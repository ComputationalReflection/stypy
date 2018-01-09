
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: ''' A sparse matrix in COOrdinate or 'triplet' format'''
2: from __future__ import division, print_function, absolute_import
3: 
4: __docformat__ = "restructuredtext en"
5: 
6: __all__ = ['coo_matrix', 'isspmatrix_coo']
7: 
8: from warnings import warn
9: 
10: import numpy as np
11: 
12: from scipy._lib.six import zip as izip
13: 
14: from ._sparsetools import coo_tocsr, coo_todense, coo_matvec
15: from .base import isspmatrix, SparseEfficiencyWarning, spmatrix
16: from .data import _data_matrix, _minmax_mixin
17: from .sputils import (upcast, upcast_char, to_native, isshape, getdtype,
18:                       get_index_dtype, downcast_intp_index)
19: 
20: 
21: class coo_matrix(_data_matrix, _minmax_mixin):
22:     '''
23:     A sparse matrix in COOrdinate format.
24: 
25:     Also known as the 'ijv' or 'triplet' format.
26: 
27:     This can be instantiated in several ways:
28:         coo_matrix(D)
29:             with a dense matrix D
30: 
31:         coo_matrix(S)
32:             with another sparse matrix S (equivalent to S.tocoo())
33: 
34:         coo_matrix((M, N), [dtype])
35:             to construct an empty matrix with shape (M, N)
36:             dtype is optional, defaulting to dtype='d'.
37: 
38:         coo_matrix((data, (i, j)), [shape=(M, N)])
39:             to construct from three arrays:
40:                 1. data[:]   the entries of the matrix, in any order
41:                 2. i[:]      the row indices of the matrix entries
42:                 3. j[:]      the column indices of the matrix entries
43: 
44:             Where ``A[i[k], j[k]] = data[k]``.  When shape is not
45:             specified, it is inferred from the index arrays
46: 
47:     Attributes
48:     ----------
49:     dtype : dtype
50:         Data type of the matrix
51:     shape : 2-tuple
52:         Shape of the matrix
53:     ndim : int
54:         Number of dimensions (this is always 2)
55:     nnz
56:         Number of nonzero elements
57:     data
58:         COO format data array of the matrix
59:     row
60:         COO format row index array of the matrix
61:     col
62:         COO format column index array of the matrix
63: 
64:     Notes
65:     -----
66: 
67:     Sparse matrices can be used in arithmetic operations: they support
68:     addition, subtraction, multiplication, division, and matrix power.
69: 
70:     Advantages of the COO format
71:         - facilitates fast conversion among sparse formats
72:         - permits duplicate entries (see example)
73:         - very fast conversion to and from CSR/CSC formats
74: 
75:     Disadvantages of the COO format
76:         - does not directly support:
77:             + arithmetic operations
78:             + slicing
79: 
80:     Intended Usage
81:         - COO is a fast format for constructing sparse matrices
82:         - Once a matrix has been constructed, convert to CSR or
83:           CSC format for fast arithmetic and matrix vector operations
84:         - By default when converting to CSR or CSC format, duplicate (i,j)
85:           entries will be summed together.  This facilitates efficient
86:           construction of finite element matrices and the like. (see example)
87: 
88:     Examples
89:     --------
90:     
91:     >>> # Constructing an empty matrix
92:     >>> from scipy.sparse import coo_matrix
93:     >>> coo_matrix((3, 4), dtype=np.int8).toarray()
94:     array([[0, 0, 0, 0],
95:            [0, 0, 0, 0],
96:            [0, 0, 0, 0]], dtype=int8)
97: 
98:     >>> # Constructing a matrix using ijv format
99:     >>> row  = np.array([0, 3, 1, 0])
100:     >>> col  = np.array([0, 3, 1, 2])
101:     >>> data = np.array([4, 5, 7, 9])
102:     >>> coo_matrix((data, (row, col)), shape=(4, 4)).toarray()
103:     array([[4, 0, 9, 0],
104:            [0, 7, 0, 0],
105:            [0, 0, 0, 0],
106:            [0, 0, 0, 5]])
107: 
108:     >>> # Constructing a matrix with duplicate indices
109:     >>> row  = np.array([0, 0, 1, 3, 1, 0, 0])
110:     >>> col  = np.array([0, 2, 1, 3, 1, 0, 0])
111:     >>> data = np.array([1, 1, 1, 1, 1, 1, 1])
112:     >>> coo = coo_matrix((data, (row, col)), shape=(4, 4))
113:     >>> # Duplicate indices are maintained until implicitly or explicitly summed
114:     >>> np.max(coo.data)
115:     1
116:     >>> coo.toarray()
117:     array([[3, 0, 1, 0],
118:            [0, 2, 0, 0],
119:            [0, 0, 0, 0],
120:            [0, 0, 0, 1]])
121: 
122:     '''
123:     format = 'coo'
124: 
125:     def __init__(self, arg1, shape=None, dtype=None, copy=False):
126:         _data_matrix.__init__(self)
127: 
128:         if isinstance(arg1, tuple):
129:             if isshape(arg1):
130:                 M, N = arg1
131:                 self.shape = (M,N)
132:                 idx_dtype = get_index_dtype(maxval=max(M, N))
133:                 self.row = np.array([], dtype=idx_dtype)
134:                 self.col = np.array([], dtype=idx_dtype)
135:                 self.data = np.array([], getdtype(dtype, default=float))
136:                 self.has_canonical_format = True
137:             else:
138:                 try:
139:                     obj, (row, col) = arg1
140:                 except (TypeError, ValueError):
141:                     raise TypeError('invalid input format')
142: 
143:                 if shape is None:
144:                     if len(row) == 0 or len(col) == 0:
145:                         raise ValueError('cannot infer dimensions from zero '
146:                                          'sized index arrays')
147:                     M = np.max(row) + 1
148:                     N = np.max(col) + 1
149:                     self.shape = (M, N)
150:                 else:
151:                     # Use 2 steps to ensure shape has length 2.
152:                     M, N = shape
153:                     self.shape = (M, N)
154: 
155:                 idx_dtype = get_index_dtype(maxval=max(self.shape))
156:                 self.row = np.array(row, copy=copy, dtype=idx_dtype)
157:                 self.col = np.array(col, copy=copy, dtype=idx_dtype)
158:                 self.data = np.array(obj, copy=copy)
159:                 self.has_canonical_format = False
160: 
161:         else:
162:             if isspmatrix(arg1):
163:                 if isspmatrix_coo(arg1) and copy:
164:                     self.row = arg1.row.copy()
165:                     self.col = arg1.col.copy()
166:                     self.data = arg1.data.copy()
167:                     self.shape = arg1.shape
168:                 else:
169:                     coo = arg1.tocoo()
170:                     self.row = coo.row
171:                     self.col = coo.col
172:                     self.data = coo.data
173:                     self.shape = coo.shape
174:                 self.has_canonical_format = False
175:             else:
176:                 #dense argument
177:                 M = np.atleast_2d(np.asarray(arg1))
178: 
179:                 if M.ndim != 2:
180:                     raise TypeError('expected dimension <= 2 array or matrix')
181:                 else:
182:                     self.shape = M.shape
183: 
184:                 self.row, self.col = M.nonzero()
185:                 self.data = M[self.row, self.col]
186:                 self.has_canonical_format = True
187: 
188:         if dtype is not None:
189:             self.data = self.data.astype(dtype, copy=False)
190: 
191:         self._check()
192: 
193:     def getnnz(self, axis=None):
194:         if axis is None:
195:             nnz = len(self.data)
196:             if nnz != len(self.row) or nnz != len(self.col):
197:                 raise ValueError('row, column, and data array must all be the '
198:                                  'same length')
199: 
200:             if self.data.ndim != 1 or self.row.ndim != 1 or \
201:                     self.col.ndim != 1:
202:                 raise ValueError('row, column, and data arrays must be 1-D')
203: 
204:             return int(nnz)
205: 
206:         if axis < 0:
207:             axis += 2
208:         if axis == 0:
209:             return np.bincount(downcast_intp_index(self.col),
210:                                minlength=self.shape[1])
211:         elif axis == 1:
212:             return np.bincount(downcast_intp_index(self.row),
213:                                minlength=self.shape[0])
214:         else:
215:             raise ValueError('axis out of bounds')
216: 
217:     getnnz.__doc__ = spmatrix.getnnz.__doc__
218: 
219:     def _check(self):
220:         ''' Checks data structure for consistency '''
221: 
222:         # index arrays should have integer data types
223:         if self.row.dtype.kind != 'i':
224:             warn("row index array has non-integer dtype (%s)  "
225:                     % self.row.dtype.name)
226:         if self.col.dtype.kind != 'i':
227:             warn("col index array has non-integer dtype (%s) "
228:                     % self.col.dtype.name)
229: 
230:         idx_dtype = get_index_dtype(maxval=max(self.shape))
231:         self.row = np.asarray(self.row, dtype=idx_dtype)
232:         self.col = np.asarray(self.col, dtype=idx_dtype)
233:         self.data = to_native(self.data)
234: 
235:         if self.nnz > 0:
236:             if self.row.max() >= self.shape[0]:
237:                 raise ValueError('row index exceeds matrix dimensions')
238:             if self.col.max() >= self.shape[1]:
239:                 raise ValueError('column index exceeds matrix dimensions')
240:             if self.row.min() < 0:
241:                 raise ValueError('negative row index found')
242:             if self.col.min() < 0:
243:                 raise ValueError('negative column index found')
244: 
245:     def transpose(self, axes=None, copy=False):
246:         if axes is not None:
247:             raise ValueError(("Sparse matrices do not support "
248:                               "an 'axes' parameter because swapping "
249:                               "dimensions is the only logical permutation."))
250: 
251:         M, N = self.shape
252:         return coo_matrix((self.data, (self.col, self.row)),
253:                           shape=(N, M), copy=copy)
254: 
255:     transpose.__doc__ = spmatrix.transpose.__doc__
256: 
257:     def toarray(self, order=None, out=None):
258:         '''See the docstring for `spmatrix.toarray`.'''
259:         B = self._process_toarray_args(order, out)
260:         fortran = int(B.flags.f_contiguous)
261:         if not fortran and not B.flags.c_contiguous:
262:             raise ValueError("Output array must be C or F contiguous")
263:         M,N = self.shape
264:         coo_todense(M, N, self.nnz, self.row, self.col, self.data,
265:                     B.ravel('A'), fortran)
266:         return B
267: 
268:     def tocsc(self, copy=False):
269:         '''Convert this matrix to Compressed Sparse Column format
270: 
271:         Duplicate entries will be summed together.
272: 
273:         Examples
274:         --------
275:         >>> from numpy import array
276:         >>> from scipy.sparse import coo_matrix
277:         >>> row  = array([0, 0, 1, 3, 1, 0, 0])
278:         >>> col  = array([0, 2, 1, 3, 1, 0, 0])
279:         >>> data = array([1, 1, 1, 1, 1, 1, 1])
280:         >>> A = coo_matrix((data, (row, col)), shape=(4, 4)).tocsc()
281:         >>> A.toarray()
282:         array([[3, 0, 1, 0],
283:                [0, 2, 0, 0],
284:                [0, 0, 0, 0],
285:                [0, 0, 0, 1]])
286: 
287:         '''
288:         from .csc import csc_matrix
289:         if self.nnz == 0:
290:             return csc_matrix(self.shape, dtype=self.dtype)
291:         else:
292:             M,N = self.shape
293:             idx_dtype = get_index_dtype((self.col, self.row),
294:                                         maxval=max(self.nnz, M))
295:             row = self.row.astype(idx_dtype, copy=False)
296:             col = self.col.astype(idx_dtype, copy=False)
297: 
298:             indptr = np.empty(N + 1, dtype=idx_dtype)
299:             indices = np.empty_like(row, dtype=idx_dtype)
300:             data = np.empty_like(self.data, dtype=upcast(self.dtype))
301: 
302:             coo_tocsr(N, M, self.nnz, col, row, self.data,
303:                       indptr, indices, data)
304: 
305:             x = csc_matrix((data, indices, indptr), shape=self.shape)
306:             if not self.has_canonical_format:
307:                 x.sum_duplicates()
308:             return x
309: 
310:     def tocsr(self, copy=False):
311:         '''Convert this matrix to Compressed Sparse Row format
312: 
313:         Duplicate entries will be summed together.
314: 
315:         Examples
316:         --------
317:         >>> from numpy import array
318:         >>> from scipy.sparse import coo_matrix
319:         >>> row  = array([0, 0, 1, 3, 1, 0, 0])
320:         >>> col  = array([0, 2, 1, 3, 1, 0, 0])
321:         >>> data = array([1, 1, 1, 1, 1, 1, 1])
322:         >>> A = coo_matrix((data, (row, col)), shape=(4, 4)).tocsr()
323:         >>> A.toarray()
324:         array([[3, 0, 1, 0],
325:                [0, 2, 0, 0],
326:                [0, 0, 0, 0],
327:                [0, 0, 0, 1]])
328: 
329:         '''
330:         from .csr import csr_matrix
331:         if self.nnz == 0:
332:             return csr_matrix(self.shape, dtype=self.dtype)
333:         else:
334:             M,N = self.shape
335:             idx_dtype = get_index_dtype((self.row, self.col),
336:                                         maxval=max(self.nnz, N))
337:             row = self.row.astype(idx_dtype, copy=False)
338:             col = self.col.astype(idx_dtype, copy=False)
339: 
340:             indptr = np.empty(M + 1, dtype=idx_dtype)
341:             indices = np.empty_like(col, dtype=idx_dtype)
342:             data = np.empty_like(self.data, dtype=upcast(self.dtype))
343: 
344:             coo_tocsr(M, N, self.nnz, row, col, self.data,
345:                       indptr, indices, data)
346: 
347:             x = csr_matrix((data, indices, indptr), shape=self.shape)
348:             if not self.has_canonical_format:
349:                 x.sum_duplicates()
350:             return x
351: 
352:     def tocoo(self, copy=False):
353:         if copy:
354:             return self.copy()
355:         else:
356:             return self
357: 
358:     tocoo.__doc__ = spmatrix.tocoo.__doc__
359: 
360:     def todia(self, copy=False):
361:         from .dia import dia_matrix
362: 
363:         self.sum_duplicates()
364:         ks = self.col - self.row  # the diagonal for each nonzero
365:         diags, diag_idx = np.unique(ks, return_inverse=True)
366: 
367:         if len(diags) > 100:
368:             # probably undesired, should todia() have a maxdiags parameter?
369:             warn("Constructing a DIA matrix with %d diagonals "
370:                  "is inefficient" % len(diags), SparseEfficiencyWarning)
371: 
372:         #initialize and fill in data array
373:         if self.data.size == 0:
374:             data = np.zeros((0, 0), dtype=self.dtype)
375:         else:
376:             data = np.zeros((len(diags), self.col.max()+1), dtype=self.dtype)
377:             data[diag_idx, self.col] = self.data
378: 
379:         return dia_matrix((data,diags), shape=self.shape)
380: 
381:     todia.__doc__ = spmatrix.todia.__doc__
382: 
383:     def todok(self, copy=False):
384:         from .dok import dok_matrix
385: 
386:         self.sum_duplicates()
387:         dok = dok_matrix((self.shape), dtype=self.dtype)
388:         dok._update(izip(izip(self.row,self.col),self.data))
389: 
390:         return dok
391: 
392:     todok.__doc__ = spmatrix.todok.__doc__
393: 
394:     def diagonal(self, k=0):
395:         rows, cols = self.shape
396:         if k <= -rows or k >= cols:
397:             raise ValueError("k exceeds matrix dimensions")
398:         diag = np.zeros(min(rows + min(k, 0), cols - max(k, 0)),
399:                         dtype=self.dtype)
400:         diag_mask = (self.row + k) == self.col
401: 
402:         if self.has_canonical_format:
403:             row = self.row[diag_mask]
404:             data = self.data[diag_mask]
405:         else:
406:             row, _, data = self._sum_duplicates(self.row[diag_mask],
407:                                                 self.col[diag_mask],
408:                                                 self.data[diag_mask])
409:         diag[row + min(k, 0)] = data
410: 
411:         return diag
412: 
413:     diagonal.__doc__ = _data_matrix.diagonal.__doc__
414: 
415:     def _setdiag(self, values, k):
416:         M, N = self.shape
417:         if values.ndim and not len(values):
418:             return
419:         idx_dtype = self.row.dtype
420: 
421:         # Determine which triples to keep and where to put the new ones.
422:         full_keep = self.col - self.row != k
423:         if k < 0:
424:             max_index = min(M+k, N)
425:             if values.ndim:
426:                 max_index = min(max_index, len(values))
427:             keep = np.logical_or(full_keep, self.col >= max_index)
428:             new_row = np.arange(-k, -k + max_index, dtype=idx_dtype)
429:             new_col = np.arange(max_index, dtype=idx_dtype)
430:         else:
431:             max_index = min(M, N-k)
432:             if values.ndim:
433:                 max_index = min(max_index, len(values))
434:             keep = np.logical_or(full_keep, self.row >= max_index)
435:             new_row = np.arange(max_index, dtype=idx_dtype)
436:             new_col = np.arange(k, k + max_index, dtype=idx_dtype)
437: 
438:         # Define the array of data consisting of the entries to be added.
439:         if values.ndim:
440:             new_data = values[:max_index]
441:         else:
442:             new_data = np.empty(max_index, dtype=self.dtype)
443:             new_data[:] = values
444: 
445:         # Update the internal structure.
446:         self.row = np.concatenate((self.row[keep], new_row))
447:         self.col = np.concatenate((self.col[keep], new_col))
448:         self.data = np.concatenate((self.data[keep], new_data))
449:         self.has_canonical_format = False
450: 
451:     # needed by _data_matrix
452:     def _with_data(self,data,copy=True):
453:         '''Returns a matrix with the same sparsity structure as self,
454:         but with different data.  By default the index arrays
455:         (i.e. .row and .col) are copied.
456:         '''
457:         if copy:
458:             return coo_matrix((data, (self.row.copy(), self.col.copy())),
459:                                    shape=self.shape, dtype=data.dtype)
460:         else:
461:             return coo_matrix((data, (self.row, self.col)),
462:                                    shape=self.shape, dtype=data.dtype)
463: 
464:     def sum_duplicates(self):
465:         '''Eliminate duplicate matrix entries by adding them together
466: 
467:         This is an *in place* operation
468:         '''
469:         if self.has_canonical_format:
470:             return
471:         summed = self._sum_duplicates(self.row, self.col, self.data)
472:         self.row, self.col, self.data = summed
473:         self.has_canonical_format = True
474: 
475:     def _sum_duplicates(self, row, col, data):
476:         # Assumes (data, row, col) not in canonical format.
477:         if len(data) == 0:
478:             return row, col, data
479:         order = np.lexsort((row, col))
480:         row = row[order]
481:         col = col[order]
482:         data = data[order]
483:         unique_mask = ((row[1:] != row[:-1]) |
484:                        (col[1:] != col[:-1]))
485:         unique_mask = np.append(True, unique_mask)
486:         row = row[unique_mask]
487:         col = col[unique_mask]
488:         unique_inds, = np.nonzero(unique_mask)
489:         data = np.add.reduceat(data, unique_inds, dtype=self.dtype)
490:         return row, col, data
491: 
492:     def eliminate_zeros(self):
493:         '''Remove zero entries from the matrix
494: 
495:         This is an *in place* operation
496:         '''
497:         mask = self.data != 0
498:         self.data = self.data[mask]
499:         self.row = self.row[mask]
500:         self.col = self.col[mask]
501: 
502:     #######################
503:     # Arithmetic handlers #
504:     #######################
505: 
506:     def _add_dense(self, other):
507:         if other.shape != self.shape:
508:             raise ValueError('Incompatible shapes.')
509:         dtype = upcast_char(self.dtype.char, other.dtype.char)
510:         result = np.array(other, dtype=dtype, copy=True)
511:         fortran = int(result.flags.f_contiguous)
512:         M, N = self.shape
513:         coo_todense(M, N, self.nnz, self.row, self.col, self.data,
514:                     result.ravel('A'), fortran)
515:         return np.matrix(result, copy=False)
516: 
517:     def _mul_vector(self, other):
518:         #output array
519:         result = np.zeros(self.shape[0], dtype=upcast_char(self.dtype.char,
520:                                                             other.dtype.char))
521:         coo_matvec(self.nnz, self.row, self.col, self.data, other, result)
522:         return result
523: 
524:     def _mul_multivector(self, other):
525:         result = np.zeros((other.shape[1], self.shape[0]),
526:                           dtype=upcast_char(self.dtype.char, other.dtype.char))
527:         for i, col in enumerate(other.T):
528:             coo_matvec(self.nnz, self.row, self.col, self.data, col, result[i])
529:         return result.T.view(type=type(other))
530: 
531: 
532: def isspmatrix_coo(x):
533:     '''Is x of coo_matrix type?
534: 
535:     Parameters
536:     ----------
537:     x
538:         object to check for being a coo matrix
539: 
540:     Returns
541:     -------
542:     bool
543:         True if x is a coo matrix, False otherwise
544: 
545:     Examples
546:     --------
547:     >>> from scipy.sparse import coo_matrix, isspmatrix_coo
548:     >>> isspmatrix_coo(coo_matrix([[5]]))
549:     True
550: 
551:     >>> from scipy.sparse import coo_matrix, csr_matrix, isspmatrix_coo
552:     >>> isspmatrix_coo(csr_matrix([[5]]))
553:     False
554:     '''
555:     return isinstance(x, coo_matrix)
556: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_368053 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1, 0), 'str', " A sparse matrix in COOrdinate or 'triplet' format")

# Assigning a Str to a Name (line 4):

# Assigning a Str to a Name (line 4):

# Assigning a Str to a Name (line 4):
str_368054 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4, 16), 'str', 'restructuredtext en')
# Assigning a type to the variable '__docformat__' (line 4)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 4, 0), '__docformat__', str_368054)

# Assigning a List to a Name (line 6):

# Assigning a List to a Name (line 6):

# Assigning a List to a Name (line 6):
__all__ = ['coo_matrix', 'isspmatrix_coo']
module_type_store.set_exportable_members(['coo_matrix', 'isspmatrix_coo'])

# Obtaining an instance of the builtin type 'list' (line 6)
list_368055 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6, 10), 'list')
# Adding type elements to the builtin type 'list' instance (line 6)
# Adding element type (line 6)
str_368056 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6, 11), 'str', 'coo_matrix')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 6, 10), list_368055, str_368056)
# Adding element type (line 6)
str_368057 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6, 25), 'str', 'isspmatrix_coo')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 6, 10), list_368055, str_368057)

# Assigning a type to the variable '__all__' (line 6)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), '__all__', list_368055)
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 8, 0))

# 'from warnings import warn' statement (line 8)
try:
    from warnings import warn

except:
    warn = UndefinedType
import_from_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'warnings', None, module_type_store, ['warn'], [warn])

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 10, 0))

# 'import numpy' statement (line 10)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/sparse/')
import_368058 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'numpy')

if (type(import_368058) is not StypyTypeError):

    if (import_368058 != 'pyd_module'):
        __import__(import_368058)
        sys_modules_368059 = sys.modules[import_368058]
        import_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'np', sys_modules_368059.module_type_store, module_type_store)
    else:
        import numpy as np

        import_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'np', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 10)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 0), 'numpy', import_368058)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/sparse/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 12, 0))

# 'from scipy._lib.six import izip' statement (line 12)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/sparse/')
import_368060 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 12, 0), 'scipy._lib.six')

if (type(import_368060) is not StypyTypeError):

    if (import_368060 != 'pyd_module'):
        __import__(import_368060)
        sys_modules_368061 = sys.modules[import_368060]
        import_from_module(stypy.reporting.localization.Localization(__file__, 12, 0), 'scipy._lib.six', sys_modules_368061.module_type_store, module_type_store, ['zip'])
        nest_module(stypy.reporting.localization.Localization(__file__, 12, 0), __file__, sys_modules_368061, sys_modules_368061.module_type_store, module_type_store)
    else:
        from scipy._lib.six import zip as izip

        import_from_module(stypy.reporting.localization.Localization(__file__, 12, 0), 'scipy._lib.six', None, module_type_store, ['zip'], [izip])

else:
    # Assigning a type to the variable 'scipy._lib.six' (line 12)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 0), 'scipy._lib.six', import_368060)

# Adding an alias
module_type_store.add_alias('izip', 'zip')
remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/sparse/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 14, 0))

# 'from scipy.sparse._sparsetools import coo_tocsr, coo_todense, coo_matvec' statement (line 14)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/sparse/')
import_368062 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 14, 0), 'scipy.sparse._sparsetools')

if (type(import_368062) is not StypyTypeError):

    if (import_368062 != 'pyd_module'):
        __import__(import_368062)
        sys_modules_368063 = sys.modules[import_368062]
        import_from_module(stypy.reporting.localization.Localization(__file__, 14, 0), 'scipy.sparse._sparsetools', sys_modules_368063.module_type_store, module_type_store, ['coo_tocsr', 'coo_todense', 'coo_matvec'])
        nest_module(stypy.reporting.localization.Localization(__file__, 14, 0), __file__, sys_modules_368063, sys_modules_368063.module_type_store, module_type_store)
    else:
        from scipy.sparse._sparsetools import coo_tocsr, coo_todense, coo_matvec

        import_from_module(stypy.reporting.localization.Localization(__file__, 14, 0), 'scipy.sparse._sparsetools', None, module_type_store, ['coo_tocsr', 'coo_todense', 'coo_matvec'], [coo_tocsr, coo_todense, coo_matvec])

else:
    # Assigning a type to the variable 'scipy.sparse._sparsetools' (line 14)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 0), 'scipy.sparse._sparsetools', import_368062)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/sparse/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 15, 0))

# 'from scipy.sparse.base import isspmatrix, SparseEfficiencyWarning, spmatrix' statement (line 15)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/sparse/')
import_368064 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 15, 0), 'scipy.sparse.base')

if (type(import_368064) is not StypyTypeError):

    if (import_368064 != 'pyd_module'):
        __import__(import_368064)
        sys_modules_368065 = sys.modules[import_368064]
        import_from_module(stypy.reporting.localization.Localization(__file__, 15, 0), 'scipy.sparse.base', sys_modules_368065.module_type_store, module_type_store, ['isspmatrix', 'SparseEfficiencyWarning', 'spmatrix'])
        nest_module(stypy.reporting.localization.Localization(__file__, 15, 0), __file__, sys_modules_368065, sys_modules_368065.module_type_store, module_type_store)
    else:
        from scipy.sparse.base import isspmatrix, SparseEfficiencyWarning, spmatrix

        import_from_module(stypy.reporting.localization.Localization(__file__, 15, 0), 'scipy.sparse.base', None, module_type_store, ['isspmatrix', 'SparseEfficiencyWarning', 'spmatrix'], [isspmatrix, SparseEfficiencyWarning, spmatrix])

else:
    # Assigning a type to the variable 'scipy.sparse.base' (line 15)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 0), 'scipy.sparse.base', import_368064)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/sparse/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 16, 0))

# 'from scipy.sparse.data import _data_matrix, _minmax_mixin' statement (line 16)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/sparse/')
import_368066 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 16, 0), 'scipy.sparse.data')

if (type(import_368066) is not StypyTypeError):

    if (import_368066 != 'pyd_module'):
        __import__(import_368066)
        sys_modules_368067 = sys.modules[import_368066]
        import_from_module(stypy.reporting.localization.Localization(__file__, 16, 0), 'scipy.sparse.data', sys_modules_368067.module_type_store, module_type_store, ['_data_matrix', '_minmax_mixin'])
        nest_module(stypy.reporting.localization.Localization(__file__, 16, 0), __file__, sys_modules_368067, sys_modules_368067.module_type_store, module_type_store)
    else:
        from scipy.sparse.data import _data_matrix, _minmax_mixin

        import_from_module(stypy.reporting.localization.Localization(__file__, 16, 0), 'scipy.sparse.data', None, module_type_store, ['_data_matrix', '_minmax_mixin'], [_data_matrix, _minmax_mixin])

else:
    # Assigning a type to the variable 'scipy.sparse.data' (line 16)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 0), 'scipy.sparse.data', import_368066)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/sparse/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 17, 0))

# 'from scipy.sparse.sputils import upcast, upcast_char, to_native, isshape, getdtype, get_index_dtype, downcast_intp_index' statement (line 17)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/sparse/')
import_368068 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 17, 0), 'scipy.sparse.sputils')

if (type(import_368068) is not StypyTypeError):

    if (import_368068 != 'pyd_module'):
        __import__(import_368068)
        sys_modules_368069 = sys.modules[import_368068]
        import_from_module(stypy.reporting.localization.Localization(__file__, 17, 0), 'scipy.sparse.sputils', sys_modules_368069.module_type_store, module_type_store, ['upcast', 'upcast_char', 'to_native', 'isshape', 'getdtype', 'get_index_dtype', 'downcast_intp_index'])
        nest_module(stypy.reporting.localization.Localization(__file__, 17, 0), __file__, sys_modules_368069, sys_modules_368069.module_type_store, module_type_store)
    else:
        from scipy.sparse.sputils import upcast, upcast_char, to_native, isshape, getdtype, get_index_dtype, downcast_intp_index

        import_from_module(stypy.reporting.localization.Localization(__file__, 17, 0), 'scipy.sparse.sputils', None, module_type_store, ['upcast', 'upcast_char', 'to_native', 'isshape', 'getdtype', 'get_index_dtype', 'downcast_intp_index'], [upcast, upcast_char, to_native, isshape, getdtype, get_index_dtype, downcast_intp_index])

else:
    # Assigning a type to the variable 'scipy.sparse.sputils' (line 17)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 17, 0), 'scipy.sparse.sputils', import_368068)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/sparse/')

# Declaration of the 'coo_matrix' class
# Getting the type of '_data_matrix' (line 21)
_data_matrix_368070 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 17), '_data_matrix')
# Getting the type of '_minmax_mixin' (line 21)
_minmax_mixin_368071 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 31), '_minmax_mixin')

class coo_matrix(_data_matrix_368070, _minmax_mixin_368071, ):
    str_368072 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 122, (-1)), 'str', "\n    A sparse matrix in COOrdinate format.\n\n    Also known as the 'ijv' or 'triplet' format.\n\n    This can be instantiated in several ways:\n        coo_matrix(D)\n            with a dense matrix D\n\n        coo_matrix(S)\n            with another sparse matrix S (equivalent to S.tocoo())\n\n        coo_matrix((M, N), [dtype])\n            to construct an empty matrix with shape (M, N)\n            dtype is optional, defaulting to dtype='d'.\n\n        coo_matrix((data, (i, j)), [shape=(M, N)])\n            to construct from three arrays:\n                1. data[:]   the entries of the matrix, in any order\n                2. i[:]      the row indices of the matrix entries\n                3. j[:]      the column indices of the matrix entries\n\n            Where ``A[i[k], j[k]] = data[k]``.  When shape is not\n            specified, it is inferred from the index arrays\n\n    Attributes\n    ----------\n    dtype : dtype\n        Data type of the matrix\n    shape : 2-tuple\n        Shape of the matrix\n    ndim : int\n        Number of dimensions (this is always 2)\n    nnz\n        Number of nonzero elements\n    data\n        COO format data array of the matrix\n    row\n        COO format row index array of the matrix\n    col\n        COO format column index array of the matrix\n\n    Notes\n    -----\n\n    Sparse matrices can be used in arithmetic operations: they support\n    addition, subtraction, multiplication, division, and matrix power.\n\n    Advantages of the COO format\n        - facilitates fast conversion among sparse formats\n        - permits duplicate entries (see example)\n        - very fast conversion to and from CSR/CSC formats\n\n    Disadvantages of the COO format\n        - does not directly support:\n            + arithmetic operations\n            + slicing\n\n    Intended Usage\n        - COO is a fast format for constructing sparse matrices\n        - Once a matrix has been constructed, convert to CSR or\n          CSC format for fast arithmetic and matrix vector operations\n        - By default when converting to CSR or CSC format, duplicate (i,j)\n          entries will be summed together.  This facilitates efficient\n          construction of finite element matrices and the like. (see example)\n\n    Examples\n    --------\n    \n    >>> # Constructing an empty matrix\n    >>> from scipy.sparse import coo_matrix\n    >>> coo_matrix((3, 4), dtype=np.int8).toarray()\n    array([[0, 0, 0, 0],\n           [0, 0, 0, 0],\n           [0, 0, 0, 0]], dtype=int8)\n\n    >>> # Constructing a matrix using ijv format\n    >>> row  = np.array([0, 3, 1, 0])\n    >>> col  = np.array([0, 3, 1, 2])\n    >>> data = np.array([4, 5, 7, 9])\n    >>> coo_matrix((data, (row, col)), shape=(4, 4)).toarray()\n    array([[4, 0, 9, 0],\n           [0, 7, 0, 0],\n           [0, 0, 0, 0],\n           [0, 0, 0, 5]])\n\n    >>> # Constructing a matrix with duplicate indices\n    >>> row  = np.array([0, 0, 1, 3, 1, 0, 0])\n    >>> col  = np.array([0, 2, 1, 3, 1, 0, 0])\n    >>> data = np.array([1, 1, 1, 1, 1, 1, 1])\n    >>> coo = coo_matrix((data, (row, col)), shape=(4, 4))\n    >>> # Duplicate indices are maintained until implicitly or explicitly summed\n    >>> np.max(coo.data)\n    1\n    >>> coo.toarray()\n    array([[3, 0, 1, 0],\n           [0, 2, 0, 0],\n           [0, 0, 0, 0],\n           [0, 0, 0, 1]])\n\n    ")
    
    # Assigning a Str to a Name (line 123):
    
    # Assigning a Str to a Name (line 123):

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 125)
        None_368073 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 35), 'None')
        # Getting the type of 'None' (line 125)
        None_368074 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 47), 'None')
        # Getting the type of 'False' (line 125)
        False_368075 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 58), 'False')
        defaults = [None_368073, None_368074, False_368075]
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 125, 4, False)
        # Assigning a type to the variable 'self' (line 126)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 126, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'coo_matrix.__init__', ['arg1', 'shape', 'dtype', 'copy'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['arg1', 'shape', 'dtype', 'copy'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        
        # Call to __init__(...): (line 126)
        # Processing the call arguments (line 126)
        # Getting the type of 'self' (line 126)
        self_368078 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 30), 'self', False)
        # Processing the call keyword arguments (line 126)
        kwargs_368079 = {}
        # Getting the type of '_data_matrix' (line 126)
        _data_matrix_368076 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 8), '_data_matrix', False)
        # Obtaining the member '__init__' of a type (line 126)
        init___368077 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 126, 8), _data_matrix_368076, '__init__')
        # Calling __init__(args, kwargs) (line 126)
        init___call_result_368080 = invoke(stypy.reporting.localization.Localization(__file__, 126, 8), init___368077, *[self_368078], **kwargs_368079)
        
        
        # Type idiom detected: calculating its left and rigth part (line 128)
        # Getting the type of 'tuple' (line 128)
        tuple_368081 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 28), 'tuple')
        # Getting the type of 'arg1' (line 128)
        arg1_368082 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 22), 'arg1')
        
        (may_be_368083, more_types_in_union_368084) = may_be_subtype(tuple_368081, arg1_368082)

        if may_be_368083:

            if more_types_in_union_368084:
                # Runtime conditional SSA (line 128)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            # Assigning a type to the variable 'arg1' (line 128)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 128, 8), 'arg1', remove_not_subtype_from_union(arg1_368082, tuple))
            
            
            # Call to isshape(...): (line 129)
            # Processing the call arguments (line 129)
            # Getting the type of 'arg1' (line 129)
            arg1_368086 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 23), 'arg1', False)
            # Processing the call keyword arguments (line 129)
            kwargs_368087 = {}
            # Getting the type of 'isshape' (line 129)
            isshape_368085 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 15), 'isshape', False)
            # Calling isshape(args, kwargs) (line 129)
            isshape_call_result_368088 = invoke(stypy.reporting.localization.Localization(__file__, 129, 15), isshape_368085, *[arg1_368086], **kwargs_368087)
            
            # Testing the type of an if condition (line 129)
            if_condition_368089 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 129, 12), isshape_call_result_368088)
            # Assigning a type to the variable 'if_condition_368089' (line 129)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 129, 12), 'if_condition_368089', if_condition_368089)
            # SSA begins for if statement (line 129)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a Name to a Tuple (line 130):
            
            # Assigning a Subscript to a Name (line 130):
            
            # Assigning a Subscript to a Name (line 130):
            
            # Obtaining the type of the subscript
            int_368090 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 130, 16), 'int')
            # Getting the type of 'arg1' (line 130)
            arg1_368091 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 23), 'arg1')
            # Obtaining the member '__getitem__' of a type (line 130)
            getitem___368092 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 130, 16), arg1_368091, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 130)
            subscript_call_result_368093 = invoke(stypy.reporting.localization.Localization(__file__, 130, 16), getitem___368092, int_368090)
            
            # Assigning a type to the variable 'tuple_var_assignment_368020' (line 130)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 130, 16), 'tuple_var_assignment_368020', subscript_call_result_368093)
            
            # Assigning a Subscript to a Name (line 130):
            
            # Assigning a Subscript to a Name (line 130):
            
            # Obtaining the type of the subscript
            int_368094 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 130, 16), 'int')
            # Getting the type of 'arg1' (line 130)
            arg1_368095 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 23), 'arg1')
            # Obtaining the member '__getitem__' of a type (line 130)
            getitem___368096 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 130, 16), arg1_368095, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 130)
            subscript_call_result_368097 = invoke(stypy.reporting.localization.Localization(__file__, 130, 16), getitem___368096, int_368094)
            
            # Assigning a type to the variable 'tuple_var_assignment_368021' (line 130)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 130, 16), 'tuple_var_assignment_368021', subscript_call_result_368097)
            
            # Assigning a Name to a Name (line 130):
            
            # Assigning a Name to a Name (line 130):
            # Getting the type of 'tuple_var_assignment_368020' (line 130)
            tuple_var_assignment_368020_368098 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 16), 'tuple_var_assignment_368020')
            # Assigning a type to the variable 'M' (line 130)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 130, 16), 'M', tuple_var_assignment_368020_368098)
            
            # Assigning a Name to a Name (line 130):
            
            # Assigning a Name to a Name (line 130):
            # Getting the type of 'tuple_var_assignment_368021' (line 130)
            tuple_var_assignment_368021_368099 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 16), 'tuple_var_assignment_368021')
            # Assigning a type to the variable 'N' (line 130)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 130, 19), 'N', tuple_var_assignment_368021_368099)
            
            # Assigning a Tuple to a Attribute (line 131):
            
            # Assigning a Tuple to a Attribute (line 131):
            
            # Assigning a Tuple to a Attribute (line 131):
            
            # Obtaining an instance of the builtin type 'tuple' (line 131)
            tuple_368100 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 131, 30), 'tuple')
            # Adding type elements to the builtin type 'tuple' instance (line 131)
            # Adding element type (line 131)
            # Getting the type of 'M' (line 131)
            M_368101 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 30), 'M')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 131, 30), tuple_368100, M_368101)
            # Adding element type (line 131)
            # Getting the type of 'N' (line 131)
            N_368102 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 32), 'N')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 131, 30), tuple_368100, N_368102)
            
            # Getting the type of 'self' (line 131)
            self_368103 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 16), 'self')
            # Setting the type of the member 'shape' of a type (line 131)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 131, 16), self_368103, 'shape', tuple_368100)
            
            # Assigning a Call to a Name (line 132):
            
            # Assigning a Call to a Name (line 132):
            
            # Assigning a Call to a Name (line 132):
            
            # Call to get_index_dtype(...): (line 132)
            # Processing the call keyword arguments (line 132)
            
            # Call to max(...): (line 132)
            # Processing the call arguments (line 132)
            # Getting the type of 'M' (line 132)
            M_368106 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 55), 'M', False)
            # Getting the type of 'N' (line 132)
            N_368107 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 58), 'N', False)
            # Processing the call keyword arguments (line 132)
            kwargs_368108 = {}
            # Getting the type of 'max' (line 132)
            max_368105 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 51), 'max', False)
            # Calling max(args, kwargs) (line 132)
            max_call_result_368109 = invoke(stypy.reporting.localization.Localization(__file__, 132, 51), max_368105, *[M_368106, N_368107], **kwargs_368108)
            
            keyword_368110 = max_call_result_368109
            kwargs_368111 = {'maxval': keyword_368110}
            # Getting the type of 'get_index_dtype' (line 132)
            get_index_dtype_368104 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 28), 'get_index_dtype', False)
            # Calling get_index_dtype(args, kwargs) (line 132)
            get_index_dtype_call_result_368112 = invoke(stypy.reporting.localization.Localization(__file__, 132, 28), get_index_dtype_368104, *[], **kwargs_368111)
            
            # Assigning a type to the variable 'idx_dtype' (line 132)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 132, 16), 'idx_dtype', get_index_dtype_call_result_368112)
            
            # Assigning a Call to a Attribute (line 133):
            
            # Assigning a Call to a Attribute (line 133):
            
            # Assigning a Call to a Attribute (line 133):
            
            # Call to array(...): (line 133)
            # Processing the call arguments (line 133)
            
            # Obtaining an instance of the builtin type 'list' (line 133)
            list_368115 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 133, 36), 'list')
            # Adding type elements to the builtin type 'list' instance (line 133)
            
            # Processing the call keyword arguments (line 133)
            # Getting the type of 'idx_dtype' (line 133)
            idx_dtype_368116 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 46), 'idx_dtype', False)
            keyword_368117 = idx_dtype_368116
            kwargs_368118 = {'dtype': keyword_368117}
            # Getting the type of 'np' (line 133)
            np_368113 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 27), 'np', False)
            # Obtaining the member 'array' of a type (line 133)
            array_368114 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 133, 27), np_368113, 'array')
            # Calling array(args, kwargs) (line 133)
            array_call_result_368119 = invoke(stypy.reporting.localization.Localization(__file__, 133, 27), array_368114, *[list_368115], **kwargs_368118)
            
            # Getting the type of 'self' (line 133)
            self_368120 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 16), 'self')
            # Setting the type of the member 'row' of a type (line 133)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 133, 16), self_368120, 'row', array_call_result_368119)
            
            # Assigning a Call to a Attribute (line 134):
            
            # Assigning a Call to a Attribute (line 134):
            
            # Assigning a Call to a Attribute (line 134):
            
            # Call to array(...): (line 134)
            # Processing the call arguments (line 134)
            
            # Obtaining an instance of the builtin type 'list' (line 134)
            list_368123 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 134, 36), 'list')
            # Adding type elements to the builtin type 'list' instance (line 134)
            
            # Processing the call keyword arguments (line 134)
            # Getting the type of 'idx_dtype' (line 134)
            idx_dtype_368124 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 46), 'idx_dtype', False)
            keyword_368125 = idx_dtype_368124
            kwargs_368126 = {'dtype': keyword_368125}
            # Getting the type of 'np' (line 134)
            np_368121 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 27), 'np', False)
            # Obtaining the member 'array' of a type (line 134)
            array_368122 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 134, 27), np_368121, 'array')
            # Calling array(args, kwargs) (line 134)
            array_call_result_368127 = invoke(stypy.reporting.localization.Localization(__file__, 134, 27), array_368122, *[list_368123], **kwargs_368126)
            
            # Getting the type of 'self' (line 134)
            self_368128 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 16), 'self')
            # Setting the type of the member 'col' of a type (line 134)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 134, 16), self_368128, 'col', array_call_result_368127)
            
            # Assigning a Call to a Attribute (line 135):
            
            # Assigning a Call to a Attribute (line 135):
            
            # Assigning a Call to a Attribute (line 135):
            
            # Call to array(...): (line 135)
            # Processing the call arguments (line 135)
            
            # Obtaining an instance of the builtin type 'list' (line 135)
            list_368131 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 135, 37), 'list')
            # Adding type elements to the builtin type 'list' instance (line 135)
            
            
            # Call to getdtype(...): (line 135)
            # Processing the call arguments (line 135)
            # Getting the type of 'dtype' (line 135)
            dtype_368133 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 50), 'dtype', False)
            # Processing the call keyword arguments (line 135)
            # Getting the type of 'float' (line 135)
            float_368134 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 65), 'float', False)
            keyword_368135 = float_368134
            kwargs_368136 = {'default': keyword_368135}
            # Getting the type of 'getdtype' (line 135)
            getdtype_368132 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 41), 'getdtype', False)
            # Calling getdtype(args, kwargs) (line 135)
            getdtype_call_result_368137 = invoke(stypy.reporting.localization.Localization(__file__, 135, 41), getdtype_368132, *[dtype_368133], **kwargs_368136)
            
            # Processing the call keyword arguments (line 135)
            kwargs_368138 = {}
            # Getting the type of 'np' (line 135)
            np_368129 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 28), 'np', False)
            # Obtaining the member 'array' of a type (line 135)
            array_368130 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 135, 28), np_368129, 'array')
            # Calling array(args, kwargs) (line 135)
            array_call_result_368139 = invoke(stypy.reporting.localization.Localization(__file__, 135, 28), array_368130, *[list_368131, getdtype_call_result_368137], **kwargs_368138)
            
            # Getting the type of 'self' (line 135)
            self_368140 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 16), 'self')
            # Setting the type of the member 'data' of a type (line 135)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 135, 16), self_368140, 'data', array_call_result_368139)
            
            # Assigning a Name to a Attribute (line 136):
            
            # Assigning a Name to a Attribute (line 136):
            
            # Assigning a Name to a Attribute (line 136):
            # Getting the type of 'True' (line 136)
            True_368141 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 44), 'True')
            # Getting the type of 'self' (line 136)
            self_368142 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 16), 'self')
            # Setting the type of the member 'has_canonical_format' of a type (line 136)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 136, 16), self_368142, 'has_canonical_format', True_368141)
            # SSA branch for the else part of an if statement (line 129)
            module_type_store.open_ssa_branch('else')
            
            
            # SSA begins for try-except statement (line 138)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
            
            # Assigning a Name to a Tuple (line 139):
            
            # Assigning a Subscript to a Name (line 139):
            
            # Assigning a Subscript to a Name (line 139):
            
            # Obtaining the type of the subscript
            int_368143 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 139, 20), 'int')
            # Getting the type of 'arg1' (line 139)
            arg1_368144 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 38), 'arg1')
            # Obtaining the member '__getitem__' of a type (line 139)
            getitem___368145 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 139, 20), arg1_368144, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 139)
            subscript_call_result_368146 = invoke(stypy.reporting.localization.Localization(__file__, 139, 20), getitem___368145, int_368143)
            
            # Assigning a type to the variable 'tuple_var_assignment_368022' (line 139)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 139, 20), 'tuple_var_assignment_368022', subscript_call_result_368146)
            
            # Assigning a Subscript to a Name (line 139):
            
            # Assigning a Subscript to a Name (line 139):
            
            # Obtaining the type of the subscript
            int_368147 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 139, 20), 'int')
            # Getting the type of 'arg1' (line 139)
            arg1_368148 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 38), 'arg1')
            # Obtaining the member '__getitem__' of a type (line 139)
            getitem___368149 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 139, 20), arg1_368148, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 139)
            subscript_call_result_368150 = invoke(stypy.reporting.localization.Localization(__file__, 139, 20), getitem___368149, int_368147)
            
            # Assigning a type to the variable 'tuple_var_assignment_368023' (line 139)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 139, 20), 'tuple_var_assignment_368023', subscript_call_result_368150)
            
            # Assigning a Name to a Name (line 139):
            
            # Assigning a Name to a Name (line 139):
            # Getting the type of 'tuple_var_assignment_368022' (line 139)
            tuple_var_assignment_368022_368151 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 20), 'tuple_var_assignment_368022')
            # Assigning a type to the variable 'obj' (line 139)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 139, 20), 'obj', tuple_var_assignment_368022_368151)
            
            # Assigning a Name to a Tuple (line 139):
            
            # Assigning a Subscript to a Name (line 139):
            
            # Obtaining the type of the subscript
            int_368152 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 139, 20), 'int')
            # Getting the type of 'tuple_var_assignment_368023' (line 139)
            tuple_var_assignment_368023_368153 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 20), 'tuple_var_assignment_368023')
            # Obtaining the member '__getitem__' of a type (line 139)
            getitem___368154 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 139, 20), tuple_var_assignment_368023_368153, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 139)
            subscript_call_result_368155 = invoke(stypy.reporting.localization.Localization(__file__, 139, 20), getitem___368154, int_368152)
            
            # Assigning a type to the variable 'tuple_var_assignment_368051' (line 139)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 139, 20), 'tuple_var_assignment_368051', subscript_call_result_368155)
            
            # Assigning a Subscript to a Name (line 139):
            
            # Obtaining the type of the subscript
            int_368156 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 139, 20), 'int')
            # Getting the type of 'tuple_var_assignment_368023' (line 139)
            tuple_var_assignment_368023_368157 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 20), 'tuple_var_assignment_368023')
            # Obtaining the member '__getitem__' of a type (line 139)
            getitem___368158 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 139, 20), tuple_var_assignment_368023_368157, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 139)
            subscript_call_result_368159 = invoke(stypy.reporting.localization.Localization(__file__, 139, 20), getitem___368158, int_368156)
            
            # Assigning a type to the variable 'tuple_var_assignment_368052' (line 139)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 139, 20), 'tuple_var_assignment_368052', subscript_call_result_368159)
            
            # Assigning a Name to a Name (line 139):
            # Getting the type of 'tuple_var_assignment_368051' (line 139)
            tuple_var_assignment_368051_368160 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 20), 'tuple_var_assignment_368051')
            # Assigning a type to the variable 'row' (line 139)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 139, 26), 'row', tuple_var_assignment_368051_368160)
            
            # Assigning a Name to a Name (line 139):
            # Getting the type of 'tuple_var_assignment_368052' (line 139)
            tuple_var_assignment_368052_368161 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 20), 'tuple_var_assignment_368052')
            # Assigning a type to the variable 'col' (line 139)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 139, 31), 'col', tuple_var_assignment_368052_368161)
            # SSA branch for the except part of a try statement (line 138)
            # SSA branch for the except 'Tuple' branch of a try statement (line 138)
            module_type_store.open_ssa_branch('except')
            
            # Call to TypeError(...): (line 141)
            # Processing the call arguments (line 141)
            str_368163 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 141, 36), 'str', 'invalid input format')
            # Processing the call keyword arguments (line 141)
            kwargs_368164 = {}
            # Getting the type of 'TypeError' (line 141)
            TypeError_368162 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 26), 'TypeError', False)
            # Calling TypeError(args, kwargs) (line 141)
            TypeError_call_result_368165 = invoke(stypy.reporting.localization.Localization(__file__, 141, 26), TypeError_368162, *[str_368163], **kwargs_368164)
            
            ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 141, 20), TypeError_call_result_368165, 'raise parameter', BaseException)
            # SSA join for try-except statement (line 138)
            module_type_store = module_type_store.join_ssa_context()
            
            
            # Type idiom detected: calculating its left and rigth part (line 143)
            # Getting the type of 'shape' (line 143)
            shape_368166 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 19), 'shape')
            # Getting the type of 'None' (line 143)
            None_368167 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 28), 'None')
            
            (may_be_368168, more_types_in_union_368169) = may_be_none(shape_368166, None_368167)

            if may_be_368168:

                if more_types_in_union_368169:
                    # Runtime conditional SSA (line 143)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
                else:
                    module_type_store = module_type_store

                
                
                # Evaluating a boolean operation
                
                
                # Call to len(...): (line 144)
                # Processing the call arguments (line 144)
                # Getting the type of 'row' (line 144)
                row_368171 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 27), 'row', False)
                # Processing the call keyword arguments (line 144)
                kwargs_368172 = {}
                # Getting the type of 'len' (line 144)
                len_368170 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 23), 'len', False)
                # Calling len(args, kwargs) (line 144)
                len_call_result_368173 = invoke(stypy.reporting.localization.Localization(__file__, 144, 23), len_368170, *[row_368171], **kwargs_368172)
                
                int_368174 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 144, 35), 'int')
                # Applying the binary operator '==' (line 144)
                result_eq_368175 = python_operator(stypy.reporting.localization.Localization(__file__, 144, 23), '==', len_call_result_368173, int_368174)
                
                
                
                # Call to len(...): (line 144)
                # Processing the call arguments (line 144)
                # Getting the type of 'col' (line 144)
                col_368177 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 44), 'col', False)
                # Processing the call keyword arguments (line 144)
                kwargs_368178 = {}
                # Getting the type of 'len' (line 144)
                len_368176 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 40), 'len', False)
                # Calling len(args, kwargs) (line 144)
                len_call_result_368179 = invoke(stypy.reporting.localization.Localization(__file__, 144, 40), len_368176, *[col_368177], **kwargs_368178)
                
                int_368180 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 144, 52), 'int')
                # Applying the binary operator '==' (line 144)
                result_eq_368181 = python_operator(stypy.reporting.localization.Localization(__file__, 144, 40), '==', len_call_result_368179, int_368180)
                
                # Applying the binary operator 'or' (line 144)
                result_or_keyword_368182 = python_operator(stypy.reporting.localization.Localization(__file__, 144, 23), 'or', result_eq_368175, result_eq_368181)
                
                # Testing the type of an if condition (line 144)
                if_condition_368183 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 144, 20), result_or_keyword_368182)
                # Assigning a type to the variable 'if_condition_368183' (line 144)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 144, 20), 'if_condition_368183', if_condition_368183)
                # SSA begins for if statement (line 144)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Call to ValueError(...): (line 145)
                # Processing the call arguments (line 145)
                str_368185 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 145, 41), 'str', 'cannot infer dimensions from zero sized index arrays')
                # Processing the call keyword arguments (line 145)
                kwargs_368186 = {}
                # Getting the type of 'ValueError' (line 145)
                ValueError_368184 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 30), 'ValueError', False)
                # Calling ValueError(args, kwargs) (line 145)
                ValueError_call_result_368187 = invoke(stypy.reporting.localization.Localization(__file__, 145, 30), ValueError_368184, *[str_368185], **kwargs_368186)
                
                ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 145, 24), ValueError_call_result_368187, 'raise parameter', BaseException)
                # SSA join for if statement (line 144)
                module_type_store = module_type_store.join_ssa_context()
                
                
                # Assigning a BinOp to a Name (line 147):
                
                # Assigning a BinOp to a Name (line 147):
                
                # Assigning a BinOp to a Name (line 147):
                
                # Call to max(...): (line 147)
                # Processing the call arguments (line 147)
                # Getting the type of 'row' (line 147)
                row_368190 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 31), 'row', False)
                # Processing the call keyword arguments (line 147)
                kwargs_368191 = {}
                # Getting the type of 'np' (line 147)
                np_368188 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 24), 'np', False)
                # Obtaining the member 'max' of a type (line 147)
                max_368189 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 147, 24), np_368188, 'max')
                # Calling max(args, kwargs) (line 147)
                max_call_result_368192 = invoke(stypy.reporting.localization.Localization(__file__, 147, 24), max_368189, *[row_368190], **kwargs_368191)
                
                int_368193 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 147, 38), 'int')
                # Applying the binary operator '+' (line 147)
                result_add_368194 = python_operator(stypy.reporting.localization.Localization(__file__, 147, 24), '+', max_call_result_368192, int_368193)
                
                # Assigning a type to the variable 'M' (line 147)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 147, 20), 'M', result_add_368194)
                
                # Assigning a BinOp to a Name (line 148):
                
                # Assigning a BinOp to a Name (line 148):
                
                # Assigning a BinOp to a Name (line 148):
                
                # Call to max(...): (line 148)
                # Processing the call arguments (line 148)
                # Getting the type of 'col' (line 148)
                col_368197 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 31), 'col', False)
                # Processing the call keyword arguments (line 148)
                kwargs_368198 = {}
                # Getting the type of 'np' (line 148)
                np_368195 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 24), 'np', False)
                # Obtaining the member 'max' of a type (line 148)
                max_368196 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 148, 24), np_368195, 'max')
                # Calling max(args, kwargs) (line 148)
                max_call_result_368199 = invoke(stypy.reporting.localization.Localization(__file__, 148, 24), max_368196, *[col_368197], **kwargs_368198)
                
                int_368200 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 148, 38), 'int')
                # Applying the binary operator '+' (line 148)
                result_add_368201 = python_operator(stypy.reporting.localization.Localization(__file__, 148, 24), '+', max_call_result_368199, int_368200)
                
                # Assigning a type to the variable 'N' (line 148)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 148, 20), 'N', result_add_368201)
                
                # Assigning a Tuple to a Attribute (line 149):
                
                # Assigning a Tuple to a Attribute (line 149):
                
                # Assigning a Tuple to a Attribute (line 149):
                
                # Obtaining an instance of the builtin type 'tuple' (line 149)
                tuple_368202 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 149, 34), 'tuple')
                # Adding type elements to the builtin type 'tuple' instance (line 149)
                # Adding element type (line 149)
                # Getting the type of 'M' (line 149)
                M_368203 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 34), 'M')
                add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 149, 34), tuple_368202, M_368203)
                # Adding element type (line 149)
                # Getting the type of 'N' (line 149)
                N_368204 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 37), 'N')
                add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 149, 34), tuple_368202, N_368204)
                
                # Getting the type of 'self' (line 149)
                self_368205 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 20), 'self')
                # Setting the type of the member 'shape' of a type (line 149)
                module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 149, 20), self_368205, 'shape', tuple_368202)

                if more_types_in_union_368169:
                    # Runtime conditional SSA for else branch (line 143)
                    module_type_store.open_ssa_branch('idiom else')



            if ((not may_be_368168) or more_types_in_union_368169):
                
                # Assigning a Name to a Tuple (line 152):
                
                # Assigning a Subscript to a Name (line 152):
                
                # Assigning a Subscript to a Name (line 152):
                
                # Obtaining the type of the subscript
                int_368206 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 152, 20), 'int')
                # Getting the type of 'shape' (line 152)
                shape_368207 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 27), 'shape')
                # Obtaining the member '__getitem__' of a type (line 152)
                getitem___368208 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 152, 20), shape_368207, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 152)
                subscript_call_result_368209 = invoke(stypy.reporting.localization.Localization(__file__, 152, 20), getitem___368208, int_368206)
                
                # Assigning a type to the variable 'tuple_var_assignment_368024' (line 152)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 152, 20), 'tuple_var_assignment_368024', subscript_call_result_368209)
                
                # Assigning a Subscript to a Name (line 152):
                
                # Assigning a Subscript to a Name (line 152):
                
                # Obtaining the type of the subscript
                int_368210 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 152, 20), 'int')
                # Getting the type of 'shape' (line 152)
                shape_368211 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 27), 'shape')
                # Obtaining the member '__getitem__' of a type (line 152)
                getitem___368212 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 152, 20), shape_368211, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 152)
                subscript_call_result_368213 = invoke(stypy.reporting.localization.Localization(__file__, 152, 20), getitem___368212, int_368210)
                
                # Assigning a type to the variable 'tuple_var_assignment_368025' (line 152)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 152, 20), 'tuple_var_assignment_368025', subscript_call_result_368213)
                
                # Assigning a Name to a Name (line 152):
                
                # Assigning a Name to a Name (line 152):
                # Getting the type of 'tuple_var_assignment_368024' (line 152)
                tuple_var_assignment_368024_368214 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 20), 'tuple_var_assignment_368024')
                # Assigning a type to the variable 'M' (line 152)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 152, 20), 'M', tuple_var_assignment_368024_368214)
                
                # Assigning a Name to a Name (line 152):
                
                # Assigning a Name to a Name (line 152):
                # Getting the type of 'tuple_var_assignment_368025' (line 152)
                tuple_var_assignment_368025_368215 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 20), 'tuple_var_assignment_368025')
                # Assigning a type to the variable 'N' (line 152)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 152, 23), 'N', tuple_var_assignment_368025_368215)
                
                # Assigning a Tuple to a Attribute (line 153):
                
                # Assigning a Tuple to a Attribute (line 153):
                
                # Assigning a Tuple to a Attribute (line 153):
                
                # Obtaining an instance of the builtin type 'tuple' (line 153)
                tuple_368216 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 153, 34), 'tuple')
                # Adding type elements to the builtin type 'tuple' instance (line 153)
                # Adding element type (line 153)
                # Getting the type of 'M' (line 153)
                M_368217 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 34), 'M')
                add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 153, 34), tuple_368216, M_368217)
                # Adding element type (line 153)
                # Getting the type of 'N' (line 153)
                N_368218 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 37), 'N')
                add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 153, 34), tuple_368216, N_368218)
                
                # Getting the type of 'self' (line 153)
                self_368219 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 20), 'self')
                # Setting the type of the member 'shape' of a type (line 153)
                module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 153, 20), self_368219, 'shape', tuple_368216)

                if (may_be_368168 and more_types_in_union_368169):
                    # SSA join for if statement (line 143)
                    module_type_store = module_type_store.join_ssa_context()


            
            
            # Assigning a Call to a Name (line 155):
            
            # Assigning a Call to a Name (line 155):
            
            # Assigning a Call to a Name (line 155):
            
            # Call to get_index_dtype(...): (line 155)
            # Processing the call keyword arguments (line 155)
            
            # Call to max(...): (line 155)
            # Processing the call arguments (line 155)
            # Getting the type of 'self' (line 155)
            self_368222 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 55), 'self', False)
            # Obtaining the member 'shape' of a type (line 155)
            shape_368223 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 155, 55), self_368222, 'shape')
            # Processing the call keyword arguments (line 155)
            kwargs_368224 = {}
            # Getting the type of 'max' (line 155)
            max_368221 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 51), 'max', False)
            # Calling max(args, kwargs) (line 155)
            max_call_result_368225 = invoke(stypy.reporting.localization.Localization(__file__, 155, 51), max_368221, *[shape_368223], **kwargs_368224)
            
            keyword_368226 = max_call_result_368225
            kwargs_368227 = {'maxval': keyword_368226}
            # Getting the type of 'get_index_dtype' (line 155)
            get_index_dtype_368220 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 28), 'get_index_dtype', False)
            # Calling get_index_dtype(args, kwargs) (line 155)
            get_index_dtype_call_result_368228 = invoke(stypy.reporting.localization.Localization(__file__, 155, 28), get_index_dtype_368220, *[], **kwargs_368227)
            
            # Assigning a type to the variable 'idx_dtype' (line 155)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 155, 16), 'idx_dtype', get_index_dtype_call_result_368228)
            
            # Assigning a Call to a Attribute (line 156):
            
            # Assigning a Call to a Attribute (line 156):
            
            # Assigning a Call to a Attribute (line 156):
            
            # Call to array(...): (line 156)
            # Processing the call arguments (line 156)
            # Getting the type of 'row' (line 156)
            row_368231 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 36), 'row', False)
            # Processing the call keyword arguments (line 156)
            # Getting the type of 'copy' (line 156)
            copy_368232 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 46), 'copy', False)
            keyword_368233 = copy_368232
            # Getting the type of 'idx_dtype' (line 156)
            idx_dtype_368234 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 58), 'idx_dtype', False)
            keyword_368235 = idx_dtype_368234
            kwargs_368236 = {'dtype': keyword_368235, 'copy': keyword_368233}
            # Getting the type of 'np' (line 156)
            np_368229 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 27), 'np', False)
            # Obtaining the member 'array' of a type (line 156)
            array_368230 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 156, 27), np_368229, 'array')
            # Calling array(args, kwargs) (line 156)
            array_call_result_368237 = invoke(stypy.reporting.localization.Localization(__file__, 156, 27), array_368230, *[row_368231], **kwargs_368236)
            
            # Getting the type of 'self' (line 156)
            self_368238 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 16), 'self')
            # Setting the type of the member 'row' of a type (line 156)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 156, 16), self_368238, 'row', array_call_result_368237)
            
            # Assigning a Call to a Attribute (line 157):
            
            # Assigning a Call to a Attribute (line 157):
            
            # Assigning a Call to a Attribute (line 157):
            
            # Call to array(...): (line 157)
            # Processing the call arguments (line 157)
            # Getting the type of 'col' (line 157)
            col_368241 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 36), 'col', False)
            # Processing the call keyword arguments (line 157)
            # Getting the type of 'copy' (line 157)
            copy_368242 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 46), 'copy', False)
            keyword_368243 = copy_368242
            # Getting the type of 'idx_dtype' (line 157)
            idx_dtype_368244 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 58), 'idx_dtype', False)
            keyword_368245 = idx_dtype_368244
            kwargs_368246 = {'dtype': keyword_368245, 'copy': keyword_368243}
            # Getting the type of 'np' (line 157)
            np_368239 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 27), 'np', False)
            # Obtaining the member 'array' of a type (line 157)
            array_368240 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 157, 27), np_368239, 'array')
            # Calling array(args, kwargs) (line 157)
            array_call_result_368247 = invoke(stypy.reporting.localization.Localization(__file__, 157, 27), array_368240, *[col_368241], **kwargs_368246)
            
            # Getting the type of 'self' (line 157)
            self_368248 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 16), 'self')
            # Setting the type of the member 'col' of a type (line 157)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 157, 16), self_368248, 'col', array_call_result_368247)
            
            # Assigning a Call to a Attribute (line 158):
            
            # Assigning a Call to a Attribute (line 158):
            
            # Assigning a Call to a Attribute (line 158):
            
            # Call to array(...): (line 158)
            # Processing the call arguments (line 158)
            # Getting the type of 'obj' (line 158)
            obj_368251 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 37), 'obj', False)
            # Processing the call keyword arguments (line 158)
            # Getting the type of 'copy' (line 158)
            copy_368252 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 47), 'copy', False)
            keyword_368253 = copy_368252
            kwargs_368254 = {'copy': keyword_368253}
            # Getting the type of 'np' (line 158)
            np_368249 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 28), 'np', False)
            # Obtaining the member 'array' of a type (line 158)
            array_368250 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 158, 28), np_368249, 'array')
            # Calling array(args, kwargs) (line 158)
            array_call_result_368255 = invoke(stypy.reporting.localization.Localization(__file__, 158, 28), array_368250, *[obj_368251], **kwargs_368254)
            
            # Getting the type of 'self' (line 158)
            self_368256 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 16), 'self')
            # Setting the type of the member 'data' of a type (line 158)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 158, 16), self_368256, 'data', array_call_result_368255)
            
            # Assigning a Name to a Attribute (line 159):
            
            # Assigning a Name to a Attribute (line 159):
            
            # Assigning a Name to a Attribute (line 159):
            # Getting the type of 'False' (line 159)
            False_368257 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 44), 'False')
            # Getting the type of 'self' (line 159)
            self_368258 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 16), 'self')
            # Setting the type of the member 'has_canonical_format' of a type (line 159)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 159, 16), self_368258, 'has_canonical_format', False_368257)
            # SSA join for if statement (line 129)
            module_type_store = module_type_store.join_ssa_context()
            

            if more_types_in_union_368084:
                # Runtime conditional SSA for else branch (line 128)
                module_type_store.open_ssa_branch('idiom else')



        if ((not may_be_368083) or more_types_in_union_368084):
            # Assigning a type to the variable 'arg1' (line 128)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 128, 8), 'arg1', remove_subtype_from_union(arg1_368082, tuple))
            
            
            # Call to isspmatrix(...): (line 162)
            # Processing the call arguments (line 162)
            # Getting the type of 'arg1' (line 162)
            arg1_368260 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 26), 'arg1', False)
            # Processing the call keyword arguments (line 162)
            kwargs_368261 = {}
            # Getting the type of 'isspmatrix' (line 162)
            isspmatrix_368259 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 15), 'isspmatrix', False)
            # Calling isspmatrix(args, kwargs) (line 162)
            isspmatrix_call_result_368262 = invoke(stypy.reporting.localization.Localization(__file__, 162, 15), isspmatrix_368259, *[arg1_368260], **kwargs_368261)
            
            # Testing the type of an if condition (line 162)
            if_condition_368263 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 162, 12), isspmatrix_call_result_368262)
            # Assigning a type to the variable 'if_condition_368263' (line 162)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 162, 12), 'if_condition_368263', if_condition_368263)
            # SSA begins for if statement (line 162)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            
            # Evaluating a boolean operation
            
            # Call to isspmatrix_coo(...): (line 163)
            # Processing the call arguments (line 163)
            # Getting the type of 'arg1' (line 163)
            arg1_368265 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 34), 'arg1', False)
            # Processing the call keyword arguments (line 163)
            kwargs_368266 = {}
            # Getting the type of 'isspmatrix_coo' (line 163)
            isspmatrix_coo_368264 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 19), 'isspmatrix_coo', False)
            # Calling isspmatrix_coo(args, kwargs) (line 163)
            isspmatrix_coo_call_result_368267 = invoke(stypy.reporting.localization.Localization(__file__, 163, 19), isspmatrix_coo_368264, *[arg1_368265], **kwargs_368266)
            
            # Getting the type of 'copy' (line 163)
            copy_368268 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 44), 'copy')
            # Applying the binary operator 'and' (line 163)
            result_and_keyword_368269 = python_operator(stypy.reporting.localization.Localization(__file__, 163, 19), 'and', isspmatrix_coo_call_result_368267, copy_368268)
            
            # Testing the type of an if condition (line 163)
            if_condition_368270 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 163, 16), result_and_keyword_368269)
            # Assigning a type to the variable 'if_condition_368270' (line 163)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 163, 16), 'if_condition_368270', if_condition_368270)
            # SSA begins for if statement (line 163)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a Call to a Attribute (line 164):
            
            # Assigning a Call to a Attribute (line 164):
            
            # Assigning a Call to a Attribute (line 164):
            
            # Call to copy(...): (line 164)
            # Processing the call keyword arguments (line 164)
            kwargs_368274 = {}
            # Getting the type of 'arg1' (line 164)
            arg1_368271 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 31), 'arg1', False)
            # Obtaining the member 'row' of a type (line 164)
            row_368272 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 164, 31), arg1_368271, 'row')
            # Obtaining the member 'copy' of a type (line 164)
            copy_368273 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 164, 31), row_368272, 'copy')
            # Calling copy(args, kwargs) (line 164)
            copy_call_result_368275 = invoke(stypy.reporting.localization.Localization(__file__, 164, 31), copy_368273, *[], **kwargs_368274)
            
            # Getting the type of 'self' (line 164)
            self_368276 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 20), 'self')
            # Setting the type of the member 'row' of a type (line 164)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 164, 20), self_368276, 'row', copy_call_result_368275)
            
            # Assigning a Call to a Attribute (line 165):
            
            # Assigning a Call to a Attribute (line 165):
            
            # Assigning a Call to a Attribute (line 165):
            
            # Call to copy(...): (line 165)
            # Processing the call keyword arguments (line 165)
            kwargs_368280 = {}
            # Getting the type of 'arg1' (line 165)
            arg1_368277 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 31), 'arg1', False)
            # Obtaining the member 'col' of a type (line 165)
            col_368278 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 165, 31), arg1_368277, 'col')
            # Obtaining the member 'copy' of a type (line 165)
            copy_368279 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 165, 31), col_368278, 'copy')
            # Calling copy(args, kwargs) (line 165)
            copy_call_result_368281 = invoke(stypy.reporting.localization.Localization(__file__, 165, 31), copy_368279, *[], **kwargs_368280)
            
            # Getting the type of 'self' (line 165)
            self_368282 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 20), 'self')
            # Setting the type of the member 'col' of a type (line 165)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 165, 20), self_368282, 'col', copy_call_result_368281)
            
            # Assigning a Call to a Attribute (line 166):
            
            # Assigning a Call to a Attribute (line 166):
            
            # Assigning a Call to a Attribute (line 166):
            
            # Call to copy(...): (line 166)
            # Processing the call keyword arguments (line 166)
            kwargs_368286 = {}
            # Getting the type of 'arg1' (line 166)
            arg1_368283 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 32), 'arg1', False)
            # Obtaining the member 'data' of a type (line 166)
            data_368284 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 166, 32), arg1_368283, 'data')
            # Obtaining the member 'copy' of a type (line 166)
            copy_368285 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 166, 32), data_368284, 'copy')
            # Calling copy(args, kwargs) (line 166)
            copy_call_result_368287 = invoke(stypy.reporting.localization.Localization(__file__, 166, 32), copy_368285, *[], **kwargs_368286)
            
            # Getting the type of 'self' (line 166)
            self_368288 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 20), 'self')
            # Setting the type of the member 'data' of a type (line 166)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 166, 20), self_368288, 'data', copy_call_result_368287)
            
            # Assigning a Attribute to a Attribute (line 167):
            
            # Assigning a Attribute to a Attribute (line 167):
            
            # Assigning a Attribute to a Attribute (line 167):
            # Getting the type of 'arg1' (line 167)
            arg1_368289 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 33), 'arg1')
            # Obtaining the member 'shape' of a type (line 167)
            shape_368290 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 167, 33), arg1_368289, 'shape')
            # Getting the type of 'self' (line 167)
            self_368291 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 20), 'self')
            # Setting the type of the member 'shape' of a type (line 167)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 167, 20), self_368291, 'shape', shape_368290)
            # SSA branch for the else part of an if statement (line 163)
            module_type_store.open_ssa_branch('else')
            
            # Assigning a Call to a Name (line 169):
            
            # Assigning a Call to a Name (line 169):
            
            # Assigning a Call to a Name (line 169):
            
            # Call to tocoo(...): (line 169)
            # Processing the call keyword arguments (line 169)
            kwargs_368294 = {}
            # Getting the type of 'arg1' (line 169)
            arg1_368292 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 26), 'arg1', False)
            # Obtaining the member 'tocoo' of a type (line 169)
            tocoo_368293 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 169, 26), arg1_368292, 'tocoo')
            # Calling tocoo(args, kwargs) (line 169)
            tocoo_call_result_368295 = invoke(stypy.reporting.localization.Localization(__file__, 169, 26), tocoo_368293, *[], **kwargs_368294)
            
            # Assigning a type to the variable 'coo' (line 169)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 169, 20), 'coo', tocoo_call_result_368295)
            
            # Assigning a Attribute to a Attribute (line 170):
            
            # Assigning a Attribute to a Attribute (line 170):
            
            # Assigning a Attribute to a Attribute (line 170):
            # Getting the type of 'coo' (line 170)
            coo_368296 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 31), 'coo')
            # Obtaining the member 'row' of a type (line 170)
            row_368297 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 170, 31), coo_368296, 'row')
            # Getting the type of 'self' (line 170)
            self_368298 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 20), 'self')
            # Setting the type of the member 'row' of a type (line 170)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 170, 20), self_368298, 'row', row_368297)
            
            # Assigning a Attribute to a Attribute (line 171):
            
            # Assigning a Attribute to a Attribute (line 171):
            
            # Assigning a Attribute to a Attribute (line 171):
            # Getting the type of 'coo' (line 171)
            coo_368299 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 31), 'coo')
            # Obtaining the member 'col' of a type (line 171)
            col_368300 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 171, 31), coo_368299, 'col')
            # Getting the type of 'self' (line 171)
            self_368301 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 20), 'self')
            # Setting the type of the member 'col' of a type (line 171)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 171, 20), self_368301, 'col', col_368300)
            
            # Assigning a Attribute to a Attribute (line 172):
            
            # Assigning a Attribute to a Attribute (line 172):
            
            # Assigning a Attribute to a Attribute (line 172):
            # Getting the type of 'coo' (line 172)
            coo_368302 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 32), 'coo')
            # Obtaining the member 'data' of a type (line 172)
            data_368303 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 172, 32), coo_368302, 'data')
            # Getting the type of 'self' (line 172)
            self_368304 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 20), 'self')
            # Setting the type of the member 'data' of a type (line 172)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 172, 20), self_368304, 'data', data_368303)
            
            # Assigning a Attribute to a Attribute (line 173):
            
            # Assigning a Attribute to a Attribute (line 173):
            
            # Assigning a Attribute to a Attribute (line 173):
            # Getting the type of 'coo' (line 173)
            coo_368305 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 33), 'coo')
            # Obtaining the member 'shape' of a type (line 173)
            shape_368306 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 173, 33), coo_368305, 'shape')
            # Getting the type of 'self' (line 173)
            self_368307 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 20), 'self')
            # Setting the type of the member 'shape' of a type (line 173)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 173, 20), self_368307, 'shape', shape_368306)
            # SSA join for if statement (line 163)
            module_type_store = module_type_store.join_ssa_context()
            
            
            # Assigning a Name to a Attribute (line 174):
            
            # Assigning a Name to a Attribute (line 174):
            
            # Assigning a Name to a Attribute (line 174):
            # Getting the type of 'False' (line 174)
            False_368308 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 44), 'False')
            # Getting the type of 'self' (line 174)
            self_368309 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 16), 'self')
            # Setting the type of the member 'has_canonical_format' of a type (line 174)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 174, 16), self_368309, 'has_canonical_format', False_368308)
            # SSA branch for the else part of an if statement (line 162)
            module_type_store.open_ssa_branch('else')
            
            # Assigning a Call to a Name (line 177):
            
            # Assigning a Call to a Name (line 177):
            
            # Assigning a Call to a Name (line 177):
            
            # Call to atleast_2d(...): (line 177)
            # Processing the call arguments (line 177)
            
            # Call to asarray(...): (line 177)
            # Processing the call arguments (line 177)
            # Getting the type of 'arg1' (line 177)
            arg1_368314 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 45), 'arg1', False)
            # Processing the call keyword arguments (line 177)
            kwargs_368315 = {}
            # Getting the type of 'np' (line 177)
            np_368312 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 34), 'np', False)
            # Obtaining the member 'asarray' of a type (line 177)
            asarray_368313 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 177, 34), np_368312, 'asarray')
            # Calling asarray(args, kwargs) (line 177)
            asarray_call_result_368316 = invoke(stypy.reporting.localization.Localization(__file__, 177, 34), asarray_368313, *[arg1_368314], **kwargs_368315)
            
            # Processing the call keyword arguments (line 177)
            kwargs_368317 = {}
            # Getting the type of 'np' (line 177)
            np_368310 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 20), 'np', False)
            # Obtaining the member 'atleast_2d' of a type (line 177)
            atleast_2d_368311 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 177, 20), np_368310, 'atleast_2d')
            # Calling atleast_2d(args, kwargs) (line 177)
            atleast_2d_call_result_368318 = invoke(stypy.reporting.localization.Localization(__file__, 177, 20), atleast_2d_368311, *[asarray_call_result_368316], **kwargs_368317)
            
            # Assigning a type to the variable 'M' (line 177)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 177, 16), 'M', atleast_2d_call_result_368318)
            
            
            # Getting the type of 'M' (line 179)
            M_368319 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 19), 'M')
            # Obtaining the member 'ndim' of a type (line 179)
            ndim_368320 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 179, 19), M_368319, 'ndim')
            int_368321 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 179, 29), 'int')
            # Applying the binary operator '!=' (line 179)
            result_ne_368322 = python_operator(stypy.reporting.localization.Localization(__file__, 179, 19), '!=', ndim_368320, int_368321)
            
            # Testing the type of an if condition (line 179)
            if_condition_368323 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 179, 16), result_ne_368322)
            # Assigning a type to the variable 'if_condition_368323' (line 179)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 179, 16), 'if_condition_368323', if_condition_368323)
            # SSA begins for if statement (line 179)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to TypeError(...): (line 180)
            # Processing the call arguments (line 180)
            str_368325 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 180, 36), 'str', 'expected dimension <= 2 array or matrix')
            # Processing the call keyword arguments (line 180)
            kwargs_368326 = {}
            # Getting the type of 'TypeError' (line 180)
            TypeError_368324 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 26), 'TypeError', False)
            # Calling TypeError(args, kwargs) (line 180)
            TypeError_call_result_368327 = invoke(stypy.reporting.localization.Localization(__file__, 180, 26), TypeError_368324, *[str_368325], **kwargs_368326)
            
            ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 180, 20), TypeError_call_result_368327, 'raise parameter', BaseException)
            # SSA branch for the else part of an if statement (line 179)
            module_type_store.open_ssa_branch('else')
            
            # Assigning a Attribute to a Attribute (line 182):
            
            # Assigning a Attribute to a Attribute (line 182):
            
            # Assigning a Attribute to a Attribute (line 182):
            # Getting the type of 'M' (line 182)
            M_368328 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 182, 33), 'M')
            # Obtaining the member 'shape' of a type (line 182)
            shape_368329 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 182, 33), M_368328, 'shape')
            # Getting the type of 'self' (line 182)
            self_368330 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 182, 20), 'self')
            # Setting the type of the member 'shape' of a type (line 182)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 182, 20), self_368330, 'shape', shape_368329)
            # SSA join for if statement (line 179)
            module_type_store = module_type_store.join_ssa_context()
            
            
            # Assigning a Call to a Tuple (line 184):
            
            # Assigning a Subscript to a Name (line 184):
            
            # Assigning a Subscript to a Name (line 184):
            
            # Obtaining the type of the subscript
            int_368331 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 184, 16), 'int')
            
            # Call to nonzero(...): (line 184)
            # Processing the call keyword arguments (line 184)
            kwargs_368334 = {}
            # Getting the type of 'M' (line 184)
            M_368332 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 37), 'M', False)
            # Obtaining the member 'nonzero' of a type (line 184)
            nonzero_368333 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 184, 37), M_368332, 'nonzero')
            # Calling nonzero(args, kwargs) (line 184)
            nonzero_call_result_368335 = invoke(stypy.reporting.localization.Localization(__file__, 184, 37), nonzero_368333, *[], **kwargs_368334)
            
            # Obtaining the member '__getitem__' of a type (line 184)
            getitem___368336 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 184, 16), nonzero_call_result_368335, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 184)
            subscript_call_result_368337 = invoke(stypy.reporting.localization.Localization(__file__, 184, 16), getitem___368336, int_368331)
            
            # Assigning a type to the variable 'tuple_var_assignment_368026' (line 184)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 184, 16), 'tuple_var_assignment_368026', subscript_call_result_368337)
            
            # Assigning a Subscript to a Name (line 184):
            
            # Assigning a Subscript to a Name (line 184):
            
            # Obtaining the type of the subscript
            int_368338 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 184, 16), 'int')
            
            # Call to nonzero(...): (line 184)
            # Processing the call keyword arguments (line 184)
            kwargs_368341 = {}
            # Getting the type of 'M' (line 184)
            M_368339 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 37), 'M', False)
            # Obtaining the member 'nonzero' of a type (line 184)
            nonzero_368340 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 184, 37), M_368339, 'nonzero')
            # Calling nonzero(args, kwargs) (line 184)
            nonzero_call_result_368342 = invoke(stypy.reporting.localization.Localization(__file__, 184, 37), nonzero_368340, *[], **kwargs_368341)
            
            # Obtaining the member '__getitem__' of a type (line 184)
            getitem___368343 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 184, 16), nonzero_call_result_368342, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 184)
            subscript_call_result_368344 = invoke(stypy.reporting.localization.Localization(__file__, 184, 16), getitem___368343, int_368338)
            
            # Assigning a type to the variable 'tuple_var_assignment_368027' (line 184)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 184, 16), 'tuple_var_assignment_368027', subscript_call_result_368344)
            
            # Assigning a Name to a Attribute (line 184):
            
            # Assigning a Name to a Attribute (line 184):
            # Getting the type of 'tuple_var_assignment_368026' (line 184)
            tuple_var_assignment_368026_368345 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 16), 'tuple_var_assignment_368026')
            # Getting the type of 'self' (line 184)
            self_368346 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 16), 'self')
            # Setting the type of the member 'row' of a type (line 184)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 184, 16), self_368346, 'row', tuple_var_assignment_368026_368345)
            
            # Assigning a Name to a Attribute (line 184):
            
            # Assigning a Name to a Attribute (line 184):
            # Getting the type of 'tuple_var_assignment_368027' (line 184)
            tuple_var_assignment_368027_368347 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 16), 'tuple_var_assignment_368027')
            # Getting the type of 'self' (line 184)
            self_368348 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 26), 'self')
            # Setting the type of the member 'col' of a type (line 184)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 184, 26), self_368348, 'col', tuple_var_assignment_368027_368347)
            
            # Assigning a Subscript to a Attribute (line 185):
            
            # Assigning a Subscript to a Attribute (line 185):
            
            # Assigning a Subscript to a Attribute (line 185):
            
            # Obtaining the type of the subscript
            
            # Obtaining an instance of the builtin type 'tuple' (line 185)
            tuple_368349 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 185, 30), 'tuple')
            # Adding type elements to the builtin type 'tuple' instance (line 185)
            # Adding element type (line 185)
            # Getting the type of 'self' (line 185)
            self_368350 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 30), 'self')
            # Obtaining the member 'row' of a type (line 185)
            row_368351 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 185, 30), self_368350, 'row')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 185, 30), tuple_368349, row_368351)
            # Adding element type (line 185)
            # Getting the type of 'self' (line 185)
            self_368352 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 40), 'self')
            # Obtaining the member 'col' of a type (line 185)
            col_368353 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 185, 40), self_368352, 'col')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 185, 30), tuple_368349, col_368353)
            
            # Getting the type of 'M' (line 185)
            M_368354 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 28), 'M')
            # Obtaining the member '__getitem__' of a type (line 185)
            getitem___368355 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 185, 28), M_368354, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 185)
            subscript_call_result_368356 = invoke(stypy.reporting.localization.Localization(__file__, 185, 28), getitem___368355, tuple_368349)
            
            # Getting the type of 'self' (line 185)
            self_368357 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 16), 'self')
            # Setting the type of the member 'data' of a type (line 185)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 185, 16), self_368357, 'data', subscript_call_result_368356)
            
            # Assigning a Name to a Attribute (line 186):
            
            # Assigning a Name to a Attribute (line 186):
            
            # Assigning a Name to a Attribute (line 186):
            # Getting the type of 'True' (line 186)
            True_368358 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 44), 'True')
            # Getting the type of 'self' (line 186)
            self_368359 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 16), 'self')
            # Setting the type of the member 'has_canonical_format' of a type (line 186)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 186, 16), self_368359, 'has_canonical_format', True_368358)
            # SSA join for if statement (line 162)
            module_type_store = module_type_store.join_ssa_context()
            

            if (may_be_368083 and more_types_in_union_368084):
                # SSA join for if statement (line 128)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Type idiom detected: calculating its left and rigth part (line 188)
        # Getting the type of 'dtype' (line 188)
        dtype_368360 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 188, 8), 'dtype')
        # Getting the type of 'None' (line 188)
        None_368361 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 188, 24), 'None')
        
        (may_be_368362, more_types_in_union_368363) = may_not_be_none(dtype_368360, None_368361)

        if may_be_368362:

            if more_types_in_union_368363:
                # Runtime conditional SSA (line 188)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a Call to a Attribute (line 189):
            
            # Assigning a Call to a Attribute (line 189):
            
            # Assigning a Call to a Attribute (line 189):
            
            # Call to astype(...): (line 189)
            # Processing the call arguments (line 189)
            # Getting the type of 'dtype' (line 189)
            dtype_368367 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 41), 'dtype', False)
            # Processing the call keyword arguments (line 189)
            # Getting the type of 'False' (line 189)
            False_368368 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 53), 'False', False)
            keyword_368369 = False_368368
            kwargs_368370 = {'copy': keyword_368369}
            # Getting the type of 'self' (line 189)
            self_368364 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 24), 'self', False)
            # Obtaining the member 'data' of a type (line 189)
            data_368365 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 189, 24), self_368364, 'data')
            # Obtaining the member 'astype' of a type (line 189)
            astype_368366 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 189, 24), data_368365, 'astype')
            # Calling astype(args, kwargs) (line 189)
            astype_call_result_368371 = invoke(stypy.reporting.localization.Localization(__file__, 189, 24), astype_368366, *[dtype_368367], **kwargs_368370)
            
            # Getting the type of 'self' (line 189)
            self_368372 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 12), 'self')
            # Setting the type of the member 'data' of a type (line 189)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 189, 12), self_368372, 'data', astype_call_result_368371)

            if more_types_in_union_368363:
                # SSA join for if statement (line 188)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Call to _check(...): (line 191)
        # Processing the call keyword arguments (line 191)
        kwargs_368375 = {}
        # Getting the type of 'self' (line 191)
        self_368373 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 8), 'self', False)
        # Obtaining the member '_check' of a type (line 191)
        _check_368374 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 191, 8), self_368373, '_check')
        # Calling _check(args, kwargs) (line 191)
        _check_call_result_368376 = invoke(stypy.reporting.localization.Localization(__file__, 191, 8), _check_368374, *[], **kwargs_368375)
        
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def getnnz(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 193)
        None_368377 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 26), 'None')
        defaults = [None_368377]
        # Create a new context for function 'getnnz'
        module_type_store = module_type_store.open_function_context('getnnz', 193, 4, False)
        # Assigning a type to the variable 'self' (line 194)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 194, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        coo_matrix.getnnz.__dict__.__setitem__('stypy_localization', localization)
        coo_matrix.getnnz.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        coo_matrix.getnnz.__dict__.__setitem__('stypy_type_store', module_type_store)
        coo_matrix.getnnz.__dict__.__setitem__('stypy_function_name', 'coo_matrix.getnnz')
        coo_matrix.getnnz.__dict__.__setitem__('stypy_param_names_list', ['axis'])
        coo_matrix.getnnz.__dict__.__setitem__('stypy_varargs_param_name', None)
        coo_matrix.getnnz.__dict__.__setitem__('stypy_kwargs_param_name', None)
        coo_matrix.getnnz.__dict__.__setitem__('stypy_call_defaults', defaults)
        coo_matrix.getnnz.__dict__.__setitem__('stypy_call_varargs', varargs)
        coo_matrix.getnnz.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        coo_matrix.getnnz.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'coo_matrix.getnnz', ['axis'], None, None, defaults, varargs, kwargs)

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

        
        # Type idiom detected: calculating its left and rigth part (line 194)
        # Getting the type of 'axis' (line 194)
        axis_368378 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 194, 11), 'axis')
        # Getting the type of 'None' (line 194)
        None_368379 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 194, 19), 'None')
        
        (may_be_368380, more_types_in_union_368381) = may_be_none(axis_368378, None_368379)

        if may_be_368380:

            if more_types_in_union_368381:
                # Runtime conditional SSA (line 194)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a Call to a Name (line 195):
            
            # Assigning a Call to a Name (line 195):
            
            # Assigning a Call to a Name (line 195):
            
            # Call to len(...): (line 195)
            # Processing the call arguments (line 195)
            # Getting the type of 'self' (line 195)
            self_368383 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 195, 22), 'self', False)
            # Obtaining the member 'data' of a type (line 195)
            data_368384 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 195, 22), self_368383, 'data')
            # Processing the call keyword arguments (line 195)
            kwargs_368385 = {}
            # Getting the type of 'len' (line 195)
            len_368382 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 195, 18), 'len', False)
            # Calling len(args, kwargs) (line 195)
            len_call_result_368386 = invoke(stypy.reporting.localization.Localization(__file__, 195, 18), len_368382, *[data_368384], **kwargs_368385)
            
            # Assigning a type to the variable 'nnz' (line 195)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 195, 12), 'nnz', len_call_result_368386)
            
            
            # Evaluating a boolean operation
            
            # Getting the type of 'nnz' (line 196)
            nnz_368387 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 196, 15), 'nnz')
            
            # Call to len(...): (line 196)
            # Processing the call arguments (line 196)
            # Getting the type of 'self' (line 196)
            self_368389 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 196, 26), 'self', False)
            # Obtaining the member 'row' of a type (line 196)
            row_368390 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 196, 26), self_368389, 'row')
            # Processing the call keyword arguments (line 196)
            kwargs_368391 = {}
            # Getting the type of 'len' (line 196)
            len_368388 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 196, 22), 'len', False)
            # Calling len(args, kwargs) (line 196)
            len_call_result_368392 = invoke(stypy.reporting.localization.Localization(__file__, 196, 22), len_368388, *[row_368390], **kwargs_368391)
            
            # Applying the binary operator '!=' (line 196)
            result_ne_368393 = python_operator(stypy.reporting.localization.Localization(__file__, 196, 15), '!=', nnz_368387, len_call_result_368392)
            
            
            # Getting the type of 'nnz' (line 196)
            nnz_368394 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 196, 39), 'nnz')
            
            # Call to len(...): (line 196)
            # Processing the call arguments (line 196)
            # Getting the type of 'self' (line 196)
            self_368396 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 196, 50), 'self', False)
            # Obtaining the member 'col' of a type (line 196)
            col_368397 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 196, 50), self_368396, 'col')
            # Processing the call keyword arguments (line 196)
            kwargs_368398 = {}
            # Getting the type of 'len' (line 196)
            len_368395 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 196, 46), 'len', False)
            # Calling len(args, kwargs) (line 196)
            len_call_result_368399 = invoke(stypy.reporting.localization.Localization(__file__, 196, 46), len_368395, *[col_368397], **kwargs_368398)
            
            # Applying the binary operator '!=' (line 196)
            result_ne_368400 = python_operator(stypy.reporting.localization.Localization(__file__, 196, 39), '!=', nnz_368394, len_call_result_368399)
            
            # Applying the binary operator 'or' (line 196)
            result_or_keyword_368401 = python_operator(stypy.reporting.localization.Localization(__file__, 196, 15), 'or', result_ne_368393, result_ne_368400)
            
            # Testing the type of an if condition (line 196)
            if_condition_368402 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 196, 12), result_or_keyword_368401)
            # Assigning a type to the variable 'if_condition_368402' (line 196)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 196, 12), 'if_condition_368402', if_condition_368402)
            # SSA begins for if statement (line 196)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to ValueError(...): (line 197)
            # Processing the call arguments (line 197)
            str_368404 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 197, 33), 'str', 'row, column, and data array must all be the same length')
            # Processing the call keyword arguments (line 197)
            kwargs_368405 = {}
            # Getting the type of 'ValueError' (line 197)
            ValueError_368403 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 197, 22), 'ValueError', False)
            # Calling ValueError(args, kwargs) (line 197)
            ValueError_call_result_368406 = invoke(stypy.reporting.localization.Localization(__file__, 197, 22), ValueError_368403, *[str_368404], **kwargs_368405)
            
            ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 197, 16), ValueError_call_result_368406, 'raise parameter', BaseException)
            # SSA join for if statement (line 196)
            module_type_store = module_type_store.join_ssa_context()
            
            
            
            # Evaluating a boolean operation
            
            # Getting the type of 'self' (line 200)
            self_368407 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 15), 'self')
            # Obtaining the member 'data' of a type (line 200)
            data_368408 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 200, 15), self_368407, 'data')
            # Obtaining the member 'ndim' of a type (line 200)
            ndim_368409 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 200, 15), data_368408, 'ndim')
            int_368410 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 200, 33), 'int')
            # Applying the binary operator '!=' (line 200)
            result_ne_368411 = python_operator(stypy.reporting.localization.Localization(__file__, 200, 15), '!=', ndim_368409, int_368410)
            
            
            # Getting the type of 'self' (line 200)
            self_368412 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 38), 'self')
            # Obtaining the member 'row' of a type (line 200)
            row_368413 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 200, 38), self_368412, 'row')
            # Obtaining the member 'ndim' of a type (line 200)
            ndim_368414 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 200, 38), row_368413, 'ndim')
            int_368415 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 200, 55), 'int')
            # Applying the binary operator '!=' (line 200)
            result_ne_368416 = python_operator(stypy.reporting.localization.Localization(__file__, 200, 38), '!=', ndim_368414, int_368415)
            
            # Applying the binary operator 'or' (line 200)
            result_or_keyword_368417 = python_operator(stypy.reporting.localization.Localization(__file__, 200, 15), 'or', result_ne_368411, result_ne_368416)
            
            # Getting the type of 'self' (line 201)
            self_368418 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 20), 'self')
            # Obtaining the member 'col' of a type (line 201)
            col_368419 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 201, 20), self_368418, 'col')
            # Obtaining the member 'ndim' of a type (line 201)
            ndim_368420 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 201, 20), col_368419, 'ndim')
            int_368421 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 201, 37), 'int')
            # Applying the binary operator '!=' (line 201)
            result_ne_368422 = python_operator(stypy.reporting.localization.Localization(__file__, 201, 20), '!=', ndim_368420, int_368421)
            
            # Applying the binary operator 'or' (line 200)
            result_or_keyword_368423 = python_operator(stypy.reporting.localization.Localization(__file__, 200, 15), 'or', result_or_keyword_368417, result_ne_368422)
            
            # Testing the type of an if condition (line 200)
            if_condition_368424 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 200, 12), result_or_keyword_368423)
            # Assigning a type to the variable 'if_condition_368424' (line 200)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 200, 12), 'if_condition_368424', if_condition_368424)
            # SSA begins for if statement (line 200)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to ValueError(...): (line 202)
            # Processing the call arguments (line 202)
            str_368426 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 202, 33), 'str', 'row, column, and data arrays must be 1-D')
            # Processing the call keyword arguments (line 202)
            kwargs_368427 = {}
            # Getting the type of 'ValueError' (line 202)
            ValueError_368425 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 22), 'ValueError', False)
            # Calling ValueError(args, kwargs) (line 202)
            ValueError_call_result_368428 = invoke(stypy.reporting.localization.Localization(__file__, 202, 22), ValueError_368425, *[str_368426], **kwargs_368427)
            
            ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 202, 16), ValueError_call_result_368428, 'raise parameter', BaseException)
            # SSA join for if statement (line 200)
            module_type_store = module_type_store.join_ssa_context()
            
            
            # Call to int(...): (line 204)
            # Processing the call arguments (line 204)
            # Getting the type of 'nnz' (line 204)
            nnz_368430 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 204, 23), 'nnz', False)
            # Processing the call keyword arguments (line 204)
            kwargs_368431 = {}
            # Getting the type of 'int' (line 204)
            int_368429 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 204, 19), 'int', False)
            # Calling int(args, kwargs) (line 204)
            int_call_result_368432 = invoke(stypy.reporting.localization.Localization(__file__, 204, 19), int_368429, *[nnz_368430], **kwargs_368431)
            
            # Assigning a type to the variable 'stypy_return_type' (line 204)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 204, 12), 'stypy_return_type', int_call_result_368432)

            if more_types_in_union_368381:
                # SSA join for if statement (line 194)
                module_type_store = module_type_store.join_ssa_context()


        
        
        
        # Getting the type of 'axis' (line 206)
        axis_368433 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 206, 11), 'axis')
        int_368434 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 206, 18), 'int')
        # Applying the binary operator '<' (line 206)
        result_lt_368435 = python_operator(stypy.reporting.localization.Localization(__file__, 206, 11), '<', axis_368433, int_368434)
        
        # Testing the type of an if condition (line 206)
        if_condition_368436 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 206, 8), result_lt_368435)
        # Assigning a type to the variable 'if_condition_368436' (line 206)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 206, 8), 'if_condition_368436', if_condition_368436)
        # SSA begins for if statement (line 206)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Getting the type of 'axis' (line 207)
        axis_368437 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 207, 12), 'axis')
        int_368438 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 207, 20), 'int')
        # Applying the binary operator '+=' (line 207)
        result_iadd_368439 = python_operator(stypy.reporting.localization.Localization(__file__, 207, 12), '+=', axis_368437, int_368438)
        # Assigning a type to the variable 'axis' (line 207)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 207, 12), 'axis', result_iadd_368439)
        
        # SSA join for if statement (line 206)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Getting the type of 'axis' (line 208)
        axis_368440 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 208, 11), 'axis')
        int_368441 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 208, 19), 'int')
        # Applying the binary operator '==' (line 208)
        result_eq_368442 = python_operator(stypy.reporting.localization.Localization(__file__, 208, 11), '==', axis_368440, int_368441)
        
        # Testing the type of an if condition (line 208)
        if_condition_368443 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 208, 8), result_eq_368442)
        # Assigning a type to the variable 'if_condition_368443' (line 208)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 208, 8), 'if_condition_368443', if_condition_368443)
        # SSA begins for if statement (line 208)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to bincount(...): (line 209)
        # Processing the call arguments (line 209)
        
        # Call to downcast_intp_index(...): (line 209)
        # Processing the call arguments (line 209)
        # Getting the type of 'self' (line 209)
        self_368447 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 51), 'self', False)
        # Obtaining the member 'col' of a type (line 209)
        col_368448 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 209, 51), self_368447, 'col')
        # Processing the call keyword arguments (line 209)
        kwargs_368449 = {}
        # Getting the type of 'downcast_intp_index' (line 209)
        downcast_intp_index_368446 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 31), 'downcast_intp_index', False)
        # Calling downcast_intp_index(args, kwargs) (line 209)
        downcast_intp_index_call_result_368450 = invoke(stypy.reporting.localization.Localization(__file__, 209, 31), downcast_intp_index_368446, *[col_368448], **kwargs_368449)
        
        # Processing the call keyword arguments (line 209)
        
        # Obtaining the type of the subscript
        int_368451 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 210, 52), 'int')
        # Getting the type of 'self' (line 210)
        self_368452 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 41), 'self', False)
        # Obtaining the member 'shape' of a type (line 210)
        shape_368453 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 210, 41), self_368452, 'shape')
        # Obtaining the member '__getitem__' of a type (line 210)
        getitem___368454 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 210, 41), shape_368453, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 210)
        subscript_call_result_368455 = invoke(stypy.reporting.localization.Localization(__file__, 210, 41), getitem___368454, int_368451)
        
        keyword_368456 = subscript_call_result_368455
        kwargs_368457 = {'minlength': keyword_368456}
        # Getting the type of 'np' (line 209)
        np_368444 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 19), 'np', False)
        # Obtaining the member 'bincount' of a type (line 209)
        bincount_368445 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 209, 19), np_368444, 'bincount')
        # Calling bincount(args, kwargs) (line 209)
        bincount_call_result_368458 = invoke(stypy.reporting.localization.Localization(__file__, 209, 19), bincount_368445, *[downcast_intp_index_call_result_368450], **kwargs_368457)
        
        # Assigning a type to the variable 'stypy_return_type' (line 209)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 209, 12), 'stypy_return_type', bincount_call_result_368458)
        # SSA branch for the else part of an if statement (line 208)
        module_type_store.open_ssa_branch('else')
        
        
        # Getting the type of 'axis' (line 211)
        axis_368459 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 211, 13), 'axis')
        int_368460 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 211, 21), 'int')
        # Applying the binary operator '==' (line 211)
        result_eq_368461 = python_operator(stypy.reporting.localization.Localization(__file__, 211, 13), '==', axis_368459, int_368460)
        
        # Testing the type of an if condition (line 211)
        if_condition_368462 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 211, 13), result_eq_368461)
        # Assigning a type to the variable 'if_condition_368462' (line 211)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 211, 13), 'if_condition_368462', if_condition_368462)
        # SSA begins for if statement (line 211)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to bincount(...): (line 212)
        # Processing the call arguments (line 212)
        
        # Call to downcast_intp_index(...): (line 212)
        # Processing the call arguments (line 212)
        # Getting the type of 'self' (line 212)
        self_368466 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 51), 'self', False)
        # Obtaining the member 'row' of a type (line 212)
        row_368467 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 212, 51), self_368466, 'row')
        # Processing the call keyword arguments (line 212)
        kwargs_368468 = {}
        # Getting the type of 'downcast_intp_index' (line 212)
        downcast_intp_index_368465 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 31), 'downcast_intp_index', False)
        # Calling downcast_intp_index(args, kwargs) (line 212)
        downcast_intp_index_call_result_368469 = invoke(stypy.reporting.localization.Localization(__file__, 212, 31), downcast_intp_index_368465, *[row_368467], **kwargs_368468)
        
        # Processing the call keyword arguments (line 212)
        
        # Obtaining the type of the subscript
        int_368470 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 213, 52), 'int')
        # Getting the type of 'self' (line 213)
        self_368471 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 41), 'self', False)
        # Obtaining the member 'shape' of a type (line 213)
        shape_368472 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 213, 41), self_368471, 'shape')
        # Obtaining the member '__getitem__' of a type (line 213)
        getitem___368473 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 213, 41), shape_368472, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 213)
        subscript_call_result_368474 = invoke(stypy.reporting.localization.Localization(__file__, 213, 41), getitem___368473, int_368470)
        
        keyword_368475 = subscript_call_result_368474
        kwargs_368476 = {'minlength': keyword_368475}
        # Getting the type of 'np' (line 212)
        np_368463 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 19), 'np', False)
        # Obtaining the member 'bincount' of a type (line 212)
        bincount_368464 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 212, 19), np_368463, 'bincount')
        # Calling bincount(args, kwargs) (line 212)
        bincount_call_result_368477 = invoke(stypy.reporting.localization.Localization(__file__, 212, 19), bincount_368464, *[downcast_intp_index_call_result_368469], **kwargs_368476)
        
        # Assigning a type to the variable 'stypy_return_type' (line 212)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 212, 12), 'stypy_return_type', bincount_call_result_368477)
        # SSA branch for the else part of an if statement (line 211)
        module_type_store.open_ssa_branch('else')
        
        # Call to ValueError(...): (line 215)
        # Processing the call arguments (line 215)
        str_368479 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 215, 29), 'str', 'axis out of bounds')
        # Processing the call keyword arguments (line 215)
        kwargs_368480 = {}
        # Getting the type of 'ValueError' (line 215)
        ValueError_368478 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 215, 18), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 215)
        ValueError_call_result_368481 = invoke(stypy.reporting.localization.Localization(__file__, 215, 18), ValueError_368478, *[str_368479], **kwargs_368480)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 215, 12), ValueError_call_result_368481, 'raise parameter', BaseException)
        # SSA join for if statement (line 211)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 208)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'getnnz(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'getnnz' in the type store
        # Getting the type of 'stypy_return_type' (line 193)
        stypy_return_type_368482 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_368482)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'getnnz'
        return stypy_return_type_368482

    
    # Assigning a Attribute to a Attribute (line 217):
    
    # Assigning a Attribute to a Attribute (line 217):

    @norecursion
    def _check(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_check'
        module_type_store = module_type_store.open_function_context('_check', 219, 4, False)
        # Assigning a type to the variable 'self' (line 220)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 220, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        coo_matrix._check.__dict__.__setitem__('stypy_localization', localization)
        coo_matrix._check.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        coo_matrix._check.__dict__.__setitem__('stypy_type_store', module_type_store)
        coo_matrix._check.__dict__.__setitem__('stypy_function_name', 'coo_matrix._check')
        coo_matrix._check.__dict__.__setitem__('stypy_param_names_list', [])
        coo_matrix._check.__dict__.__setitem__('stypy_varargs_param_name', None)
        coo_matrix._check.__dict__.__setitem__('stypy_kwargs_param_name', None)
        coo_matrix._check.__dict__.__setitem__('stypy_call_defaults', defaults)
        coo_matrix._check.__dict__.__setitem__('stypy_call_varargs', varargs)
        coo_matrix._check.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        coo_matrix._check.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'coo_matrix._check', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_check', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_check(...)' code ##################

        str_368483 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 220, 8), 'str', ' Checks data structure for consistency ')
        
        
        # Getting the type of 'self' (line 223)
        self_368484 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 11), 'self')
        # Obtaining the member 'row' of a type (line 223)
        row_368485 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 223, 11), self_368484, 'row')
        # Obtaining the member 'dtype' of a type (line 223)
        dtype_368486 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 223, 11), row_368485, 'dtype')
        # Obtaining the member 'kind' of a type (line 223)
        kind_368487 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 223, 11), dtype_368486, 'kind')
        str_368488 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 223, 34), 'str', 'i')
        # Applying the binary operator '!=' (line 223)
        result_ne_368489 = python_operator(stypy.reporting.localization.Localization(__file__, 223, 11), '!=', kind_368487, str_368488)
        
        # Testing the type of an if condition (line 223)
        if_condition_368490 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 223, 8), result_ne_368489)
        # Assigning a type to the variable 'if_condition_368490' (line 223)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 223, 8), 'if_condition_368490', if_condition_368490)
        # SSA begins for if statement (line 223)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to warn(...): (line 224)
        # Processing the call arguments (line 224)
        str_368492 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 224, 17), 'str', 'row index array has non-integer dtype (%s)  ')
        # Getting the type of 'self' (line 225)
        self_368493 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 225, 22), 'self', False)
        # Obtaining the member 'row' of a type (line 225)
        row_368494 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 225, 22), self_368493, 'row')
        # Obtaining the member 'dtype' of a type (line 225)
        dtype_368495 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 225, 22), row_368494, 'dtype')
        # Obtaining the member 'name' of a type (line 225)
        name_368496 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 225, 22), dtype_368495, 'name')
        # Applying the binary operator '%' (line 224)
        result_mod_368497 = python_operator(stypy.reporting.localization.Localization(__file__, 224, 17), '%', str_368492, name_368496)
        
        # Processing the call keyword arguments (line 224)
        kwargs_368498 = {}
        # Getting the type of 'warn' (line 224)
        warn_368491 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 224, 12), 'warn', False)
        # Calling warn(args, kwargs) (line 224)
        warn_call_result_368499 = invoke(stypy.reporting.localization.Localization(__file__, 224, 12), warn_368491, *[result_mod_368497], **kwargs_368498)
        
        # SSA join for if statement (line 223)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Getting the type of 'self' (line 226)
        self_368500 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 11), 'self')
        # Obtaining the member 'col' of a type (line 226)
        col_368501 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 226, 11), self_368500, 'col')
        # Obtaining the member 'dtype' of a type (line 226)
        dtype_368502 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 226, 11), col_368501, 'dtype')
        # Obtaining the member 'kind' of a type (line 226)
        kind_368503 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 226, 11), dtype_368502, 'kind')
        str_368504 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 226, 34), 'str', 'i')
        # Applying the binary operator '!=' (line 226)
        result_ne_368505 = python_operator(stypy.reporting.localization.Localization(__file__, 226, 11), '!=', kind_368503, str_368504)
        
        # Testing the type of an if condition (line 226)
        if_condition_368506 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 226, 8), result_ne_368505)
        # Assigning a type to the variable 'if_condition_368506' (line 226)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 226, 8), 'if_condition_368506', if_condition_368506)
        # SSA begins for if statement (line 226)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to warn(...): (line 227)
        # Processing the call arguments (line 227)
        str_368508 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 227, 17), 'str', 'col index array has non-integer dtype (%s) ')
        # Getting the type of 'self' (line 228)
        self_368509 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 22), 'self', False)
        # Obtaining the member 'col' of a type (line 228)
        col_368510 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 228, 22), self_368509, 'col')
        # Obtaining the member 'dtype' of a type (line 228)
        dtype_368511 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 228, 22), col_368510, 'dtype')
        # Obtaining the member 'name' of a type (line 228)
        name_368512 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 228, 22), dtype_368511, 'name')
        # Applying the binary operator '%' (line 227)
        result_mod_368513 = python_operator(stypy.reporting.localization.Localization(__file__, 227, 17), '%', str_368508, name_368512)
        
        # Processing the call keyword arguments (line 227)
        kwargs_368514 = {}
        # Getting the type of 'warn' (line 227)
        warn_368507 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 12), 'warn', False)
        # Calling warn(args, kwargs) (line 227)
        warn_call_result_368515 = invoke(stypy.reporting.localization.Localization(__file__, 227, 12), warn_368507, *[result_mod_368513], **kwargs_368514)
        
        # SSA join for if statement (line 226)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Name (line 230):
        
        # Assigning a Call to a Name (line 230):
        
        # Assigning a Call to a Name (line 230):
        
        # Call to get_index_dtype(...): (line 230)
        # Processing the call keyword arguments (line 230)
        
        # Call to max(...): (line 230)
        # Processing the call arguments (line 230)
        # Getting the type of 'self' (line 230)
        self_368518 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 47), 'self', False)
        # Obtaining the member 'shape' of a type (line 230)
        shape_368519 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 230, 47), self_368518, 'shape')
        # Processing the call keyword arguments (line 230)
        kwargs_368520 = {}
        # Getting the type of 'max' (line 230)
        max_368517 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 43), 'max', False)
        # Calling max(args, kwargs) (line 230)
        max_call_result_368521 = invoke(stypy.reporting.localization.Localization(__file__, 230, 43), max_368517, *[shape_368519], **kwargs_368520)
        
        keyword_368522 = max_call_result_368521
        kwargs_368523 = {'maxval': keyword_368522}
        # Getting the type of 'get_index_dtype' (line 230)
        get_index_dtype_368516 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 20), 'get_index_dtype', False)
        # Calling get_index_dtype(args, kwargs) (line 230)
        get_index_dtype_call_result_368524 = invoke(stypy.reporting.localization.Localization(__file__, 230, 20), get_index_dtype_368516, *[], **kwargs_368523)
        
        # Assigning a type to the variable 'idx_dtype' (line 230)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 230, 8), 'idx_dtype', get_index_dtype_call_result_368524)
        
        # Assigning a Call to a Attribute (line 231):
        
        # Assigning a Call to a Attribute (line 231):
        
        # Assigning a Call to a Attribute (line 231):
        
        # Call to asarray(...): (line 231)
        # Processing the call arguments (line 231)
        # Getting the type of 'self' (line 231)
        self_368527 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 231, 30), 'self', False)
        # Obtaining the member 'row' of a type (line 231)
        row_368528 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 231, 30), self_368527, 'row')
        # Processing the call keyword arguments (line 231)
        # Getting the type of 'idx_dtype' (line 231)
        idx_dtype_368529 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 231, 46), 'idx_dtype', False)
        keyword_368530 = idx_dtype_368529
        kwargs_368531 = {'dtype': keyword_368530}
        # Getting the type of 'np' (line 231)
        np_368525 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 231, 19), 'np', False)
        # Obtaining the member 'asarray' of a type (line 231)
        asarray_368526 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 231, 19), np_368525, 'asarray')
        # Calling asarray(args, kwargs) (line 231)
        asarray_call_result_368532 = invoke(stypy.reporting.localization.Localization(__file__, 231, 19), asarray_368526, *[row_368528], **kwargs_368531)
        
        # Getting the type of 'self' (line 231)
        self_368533 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 231, 8), 'self')
        # Setting the type of the member 'row' of a type (line 231)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 231, 8), self_368533, 'row', asarray_call_result_368532)
        
        # Assigning a Call to a Attribute (line 232):
        
        # Assigning a Call to a Attribute (line 232):
        
        # Assigning a Call to a Attribute (line 232):
        
        # Call to asarray(...): (line 232)
        # Processing the call arguments (line 232)
        # Getting the type of 'self' (line 232)
        self_368536 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 30), 'self', False)
        # Obtaining the member 'col' of a type (line 232)
        col_368537 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 232, 30), self_368536, 'col')
        # Processing the call keyword arguments (line 232)
        # Getting the type of 'idx_dtype' (line 232)
        idx_dtype_368538 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 46), 'idx_dtype', False)
        keyword_368539 = idx_dtype_368538
        kwargs_368540 = {'dtype': keyword_368539}
        # Getting the type of 'np' (line 232)
        np_368534 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 19), 'np', False)
        # Obtaining the member 'asarray' of a type (line 232)
        asarray_368535 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 232, 19), np_368534, 'asarray')
        # Calling asarray(args, kwargs) (line 232)
        asarray_call_result_368541 = invoke(stypy.reporting.localization.Localization(__file__, 232, 19), asarray_368535, *[col_368537], **kwargs_368540)
        
        # Getting the type of 'self' (line 232)
        self_368542 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 8), 'self')
        # Setting the type of the member 'col' of a type (line 232)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 232, 8), self_368542, 'col', asarray_call_result_368541)
        
        # Assigning a Call to a Attribute (line 233):
        
        # Assigning a Call to a Attribute (line 233):
        
        # Assigning a Call to a Attribute (line 233):
        
        # Call to to_native(...): (line 233)
        # Processing the call arguments (line 233)
        # Getting the type of 'self' (line 233)
        self_368544 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 233, 30), 'self', False)
        # Obtaining the member 'data' of a type (line 233)
        data_368545 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 233, 30), self_368544, 'data')
        # Processing the call keyword arguments (line 233)
        kwargs_368546 = {}
        # Getting the type of 'to_native' (line 233)
        to_native_368543 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 233, 20), 'to_native', False)
        # Calling to_native(args, kwargs) (line 233)
        to_native_call_result_368547 = invoke(stypy.reporting.localization.Localization(__file__, 233, 20), to_native_368543, *[data_368545], **kwargs_368546)
        
        # Getting the type of 'self' (line 233)
        self_368548 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 233, 8), 'self')
        # Setting the type of the member 'data' of a type (line 233)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 233, 8), self_368548, 'data', to_native_call_result_368547)
        
        
        # Getting the type of 'self' (line 235)
        self_368549 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 11), 'self')
        # Obtaining the member 'nnz' of a type (line 235)
        nnz_368550 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 235, 11), self_368549, 'nnz')
        int_368551 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 235, 22), 'int')
        # Applying the binary operator '>' (line 235)
        result_gt_368552 = python_operator(stypy.reporting.localization.Localization(__file__, 235, 11), '>', nnz_368550, int_368551)
        
        # Testing the type of an if condition (line 235)
        if_condition_368553 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 235, 8), result_gt_368552)
        # Assigning a type to the variable 'if_condition_368553' (line 235)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 235, 8), 'if_condition_368553', if_condition_368553)
        # SSA begins for if statement (line 235)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        
        
        # Call to max(...): (line 236)
        # Processing the call keyword arguments (line 236)
        kwargs_368557 = {}
        # Getting the type of 'self' (line 236)
        self_368554 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 236, 15), 'self', False)
        # Obtaining the member 'row' of a type (line 236)
        row_368555 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 236, 15), self_368554, 'row')
        # Obtaining the member 'max' of a type (line 236)
        max_368556 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 236, 15), row_368555, 'max')
        # Calling max(args, kwargs) (line 236)
        max_call_result_368558 = invoke(stypy.reporting.localization.Localization(__file__, 236, 15), max_368556, *[], **kwargs_368557)
        
        
        # Obtaining the type of the subscript
        int_368559 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 236, 44), 'int')
        # Getting the type of 'self' (line 236)
        self_368560 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 236, 33), 'self')
        # Obtaining the member 'shape' of a type (line 236)
        shape_368561 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 236, 33), self_368560, 'shape')
        # Obtaining the member '__getitem__' of a type (line 236)
        getitem___368562 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 236, 33), shape_368561, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 236)
        subscript_call_result_368563 = invoke(stypy.reporting.localization.Localization(__file__, 236, 33), getitem___368562, int_368559)
        
        # Applying the binary operator '>=' (line 236)
        result_ge_368564 = python_operator(stypy.reporting.localization.Localization(__file__, 236, 15), '>=', max_call_result_368558, subscript_call_result_368563)
        
        # Testing the type of an if condition (line 236)
        if_condition_368565 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 236, 12), result_ge_368564)
        # Assigning a type to the variable 'if_condition_368565' (line 236)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 236, 12), 'if_condition_368565', if_condition_368565)
        # SSA begins for if statement (line 236)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to ValueError(...): (line 237)
        # Processing the call arguments (line 237)
        str_368567 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 237, 33), 'str', 'row index exceeds matrix dimensions')
        # Processing the call keyword arguments (line 237)
        kwargs_368568 = {}
        # Getting the type of 'ValueError' (line 237)
        ValueError_368566 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 237, 22), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 237)
        ValueError_call_result_368569 = invoke(stypy.reporting.localization.Localization(__file__, 237, 22), ValueError_368566, *[str_368567], **kwargs_368568)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 237, 16), ValueError_call_result_368569, 'raise parameter', BaseException)
        # SSA join for if statement (line 236)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        
        # Call to max(...): (line 238)
        # Processing the call keyword arguments (line 238)
        kwargs_368573 = {}
        # Getting the type of 'self' (line 238)
        self_368570 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 238, 15), 'self', False)
        # Obtaining the member 'col' of a type (line 238)
        col_368571 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 238, 15), self_368570, 'col')
        # Obtaining the member 'max' of a type (line 238)
        max_368572 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 238, 15), col_368571, 'max')
        # Calling max(args, kwargs) (line 238)
        max_call_result_368574 = invoke(stypy.reporting.localization.Localization(__file__, 238, 15), max_368572, *[], **kwargs_368573)
        
        
        # Obtaining the type of the subscript
        int_368575 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 238, 44), 'int')
        # Getting the type of 'self' (line 238)
        self_368576 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 238, 33), 'self')
        # Obtaining the member 'shape' of a type (line 238)
        shape_368577 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 238, 33), self_368576, 'shape')
        # Obtaining the member '__getitem__' of a type (line 238)
        getitem___368578 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 238, 33), shape_368577, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 238)
        subscript_call_result_368579 = invoke(stypy.reporting.localization.Localization(__file__, 238, 33), getitem___368578, int_368575)
        
        # Applying the binary operator '>=' (line 238)
        result_ge_368580 = python_operator(stypy.reporting.localization.Localization(__file__, 238, 15), '>=', max_call_result_368574, subscript_call_result_368579)
        
        # Testing the type of an if condition (line 238)
        if_condition_368581 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 238, 12), result_ge_368580)
        # Assigning a type to the variable 'if_condition_368581' (line 238)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 238, 12), 'if_condition_368581', if_condition_368581)
        # SSA begins for if statement (line 238)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to ValueError(...): (line 239)
        # Processing the call arguments (line 239)
        str_368583 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 239, 33), 'str', 'column index exceeds matrix dimensions')
        # Processing the call keyword arguments (line 239)
        kwargs_368584 = {}
        # Getting the type of 'ValueError' (line 239)
        ValueError_368582 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 239, 22), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 239)
        ValueError_call_result_368585 = invoke(stypy.reporting.localization.Localization(__file__, 239, 22), ValueError_368582, *[str_368583], **kwargs_368584)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 239, 16), ValueError_call_result_368585, 'raise parameter', BaseException)
        # SSA join for if statement (line 238)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        
        # Call to min(...): (line 240)
        # Processing the call keyword arguments (line 240)
        kwargs_368589 = {}
        # Getting the type of 'self' (line 240)
        self_368586 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 15), 'self', False)
        # Obtaining the member 'row' of a type (line 240)
        row_368587 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 240, 15), self_368586, 'row')
        # Obtaining the member 'min' of a type (line 240)
        min_368588 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 240, 15), row_368587, 'min')
        # Calling min(args, kwargs) (line 240)
        min_call_result_368590 = invoke(stypy.reporting.localization.Localization(__file__, 240, 15), min_368588, *[], **kwargs_368589)
        
        int_368591 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 240, 32), 'int')
        # Applying the binary operator '<' (line 240)
        result_lt_368592 = python_operator(stypy.reporting.localization.Localization(__file__, 240, 15), '<', min_call_result_368590, int_368591)
        
        # Testing the type of an if condition (line 240)
        if_condition_368593 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 240, 12), result_lt_368592)
        # Assigning a type to the variable 'if_condition_368593' (line 240)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 240, 12), 'if_condition_368593', if_condition_368593)
        # SSA begins for if statement (line 240)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to ValueError(...): (line 241)
        # Processing the call arguments (line 241)
        str_368595 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 241, 33), 'str', 'negative row index found')
        # Processing the call keyword arguments (line 241)
        kwargs_368596 = {}
        # Getting the type of 'ValueError' (line 241)
        ValueError_368594 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 241, 22), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 241)
        ValueError_call_result_368597 = invoke(stypy.reporting.localization.Localization(__file__, 241, 22), ValueError_368594, *[str_368595], **kwargs_368596)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 241, 16), ValueError_call_result_368597, 'raise parameter', BaseException)
        # SSA join for if statement (line 240)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        
        # Call to min(...): (line 242)
        # Processing the call keyword arguments (line 242)
        kwargs_368601 = {}
        # Getting the type of 'self' (line 242)
        self_368598 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 242, 15), 'self', False)
        # Obtaining the member 'col' of a type (line 242)
        col_368599 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 242, 15), self_368598, 'col')
        # Obtaining the member 'min' of a type (line 242)
        min_368600 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 242, 15), col_368599, 'min')
        # Calling min(args, kwargs) (line 242)
        min_call_result_368602 = invoke(stypy.reporting.localization.Localization(__file__, 242, 15), min_368600, *[], **kwargs_368601)
        
        int_368603 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 242, 32), 'int')
        # Applying the binary operator '<' (line 242)
        result_lt_368604 = python_operator(stypy.reporting.localization.Localization(__file__, 242, 15), '<', min_call_result_368602, int_368603)
        
        # Testing the type of an if condition (line 242)
        if_condition_368605 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 242, 12), result_lt_368604)
        # Assigning a type to the variable 'if_condition_368605' (line 242)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 242, 12), 'if_condition_368605', if_condition_368605)
        # SSA begins for if statement (line 242)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to ValueError(...): (line 243)
        # Processing the call arguments (line 243)
        str_368607 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 243, 33), 'str', 'negative column index found')
        # Processing the call keyword arguments (line 243)
        kwargs_368608 = {}
        # Getting the type of 'ValueError' (line 243)
        ValueError_368606 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 243, 22), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 243)
        ValueError_call_result_368609 = invoke(stypy.reporting.localization.Localization(__file__, 243, 22), ValueError_368606, *[str_368607], **kwargs_368608)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 243, 16), ValueError_call_result_368609, 'raise parameter', BaseException)
        # SSA join for if statement (line 242)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 235)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of '_check(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_check' in the type store
        # Getting the type of 'stypy_return_type' (line 219)
        stypy_return_type_368610 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 219, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_368610)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_check'
        return stypy_return_type_368610


    @norecursion
    def transpose(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 245)
        None_368611 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 245, 29), 'None')
        # Getting the type of 'False' (line 245)
        False_368612 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 245, 40), 'False')
        defaults = [None_368611, False_368612]
        # Create a new context for function 'transpose'
        module_type_store = module_type_store.open_function_context('transpose', 245, 4, False)
        # Assigning a type to the variable 'self' (line 246)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 246, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        coo_matrix.transpose.__dict__.__setitem__('stypy_localization', localization)
        coo_matrix.transpose.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        coo_matrix.transpose.__dict__.__setitem__('stypy_type_store', module_type_store)
        coo_matrix.transpose.__dict__.__setitem__('stypy_function_name', 'coo_matrix.transpose')
        coo_matrix.transpose.__dict__.__setitem__('stypy_param_names_list', ['axes', 'copy'])
        coo_matrix.transpose.__dict__.__setitem__('stypy_varargs_param_name', None)
        coo_matrix.transpose.__dict__.__setitem__('stypy_kwargs_param_name', None)
        coo_matrix.transpose.__dict__.__setitem__('stypy_call_defaults', defaults)
        coo_matrix.transpose.__dict__.__setitem__('stypy_call_varargs', varargs)
        coo_matrix.transpose.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        coo_matrix.transpose.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'coo_matrix.transpose', ['axes', 'copy'], None, None, defaults, varargs, kwargs)

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

        
        # Type idiom detected: calculating its left and rigth part (line 246)
        # Getting the type of 'axes' (line 246)
        axes_368613 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 8), 'axes')
        # Getting the type of 'None' (line 246)
        None_368614 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 23), 'None')
        
        (may_be_368615, more_types_in_union_368616) = may_not_be_none(axes_368613, None_368614)

        if may_be_368615:

            if more_types_in_union_368616:
                # Runtime conditional SSA (line 246)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Call to ValueError(...): (line 247)
            # Processing the call arguments (line 247)
            str_368618 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 247, 30), 'str', "Sparse matrices do not support an 'axes' parameter because swapping dimensions is the only logical permutation.")
            # Processing the call keyword arguments (line 247)
            kwargs_368619 = {}
            # Getting the type of 'ValueError' (line 247)
            ValueError_368617 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 247, 18), 'ValueError', False)
            # Calling ValueError(args, kwargs) (line 247)
            ValueError_call_result_368620 = invoke(stypy.reporting.localization.Localization(__file__, 247, 18), ValueError_368617, *[str_368618], **kwargs_368619)
            
            ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 247, 12), ValueError_call_result_368620, 'raise parameter', BaseException)

            if more_types_in_union_368616:
                # SSA join for if statement (line 246)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Assigning a Attribute to a Tuple (line 251):
        
        # Assigning a Subscript to a Name (line 251):
        
        # Assigning a Subscript to a Name (line 251):
        
        # Obtaining the type of the subscript
        int_368621 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 251, 8), 'int')
        # Getting the type of 'self' (line 251)
        self_368622 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 251, 15), 'self')
        # Obtaining the member 'shape' of a type (line 251)
        shape_368623 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 251, 15), self_368622, 'shape')
        # Obtaining the member '__getitem__' of a type (line 251)
        getitem___368624 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 251, 8), shape_368623, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 251)
        subscript_call_result_368625 = invoke(stypy.reporting.localization.Localization(__file__, 251, 8), getitem___368624, int_368621)
        
        # Assigning a type to the variable 'tuple_var_assignment_368028' (line 251)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 251, 8), 'tuple_var_assignment_368028', subscript_call_result_368625)
        
        # Assigning a Subscript to a Name (line 251):
        
        # Assigning a Subscript to a Name (line 251):
        
        # Obtaining the type of the subscript
        int_368626 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 251, 8), 'int')
        # Getting the type of 'self' (line 251)
        self_368627 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 251, 15), 'self')
        # Obtaining the member 'shape' of a type (line 251)
        shape_368628 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 251, 15), self_368627, 'shape')
        # Obtaining the member '__getitem__' of a type (line 251)
        getitem___368629 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 251, 8), shape_368628, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 251)
        subscript_call_result_368630 = invoke(stypy.reporting.localization.Localization(__file__, 251, 8), getitem___368629, int_368626)
        
        # Assigning a type to the variable 'tuple_var_assignment_368029' (line 251)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 251, 8), 'tuple_var_assignment_368029', subscript_call_result_368630)
        
        # Assigning a Name to a Name (line 251):
        
        # Assigning a Name to a Name (line 251):
        # Getting the type of 'tuple_var_assignment_368028' (line 251)
        tuple_var_assignment_368028_368631 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 251, 8), 'tuple_var_assignment_368028')
        # Assigning a type to the variable 'M' (line 251)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 251, 8), 'M', tuple_var_assignment_368028_368631)
        
        # Assigning a Name to a Name (line 251):
        
        # Assigning a Name to a Name (line 251):
        # Getting the type of 'tuple_var_assignment_368029' (line 251)
        tuple_var_assignment_368029_368632 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 251, 8), 'tuple_var_assignment_368029')
        # Assigning a type to the variable 'N' (line 251)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 251, 11), 'N', tuple_var_assignment_368029_368632)
        
        # Call to coo_matrix(...): (line 252)
        # Processing the call arguments (line 252)
        
        # Obtaining an instance of the builtin type 'tuple' (line 252)
        tuple_368634 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 252, 27), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 252)
        # Adding element type (line 252)
        # Getting the type of 'self' (line 252)
        self_368635 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 252, 27), 'self', False)
        # Obtaining the member 'data' of a type (line 252)
        data_368636 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 252, 27), self_368635, 'data')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 252, 27), tuple_368634, data_368636)
        # Adding element type (line 252)
        
        # Obtaining an instance of the builtin type 'tuple' (line 252)
        tuple_368637 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 252, 39), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 252)
        # Adding element type (line 252)
        # Getting the type of 'self' (line 252)
        self_368638 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 252, 39), 'self', False)
        # Obtaining the member 'col' of a type (line 252)
        col_368639 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 252, 39), self_368638, 'col')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 252, 39), tuple_368637, col_368639)
        # Adding element type (line 252)
        # Getting the type of 'self' (line 252)
        self_368640 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 252, 49), 'self', False)
        # Obtaining the member 'row' of a type (line 252)
        row_368641 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 252, 49), self_368640, 'row')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 252, 39), tuple_368637, row_368641)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 252, 27), tuple_368634, tuple_368637)
        
        # Processing the call keyword arguments (line 252)
        
        # Obtaining an instance of the builtin type 'tuple' (line 253)
        tuple_368642 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 253, 33), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 253)
        # Adding element type (line 253)
        # Getting the type of 'N' (line 253)
        N_368643 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 253, 33), 'N', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 253, 33), tuple_368642, N_368643)
        # Adding element type (line 253)
        # Getting the type of 'M' (line 253)
        M_368644 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 253, 36), 'M', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 253, 33), tuple_368642, M_368644)
        
        keyword_368645 = tuple_368642
        # Getting the type of 'copy' (line 253)
        copy_368646 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 253, 45), 'copy', False)
        keyword_368647 = copy_368646
        kwargs_368648 = {'shape': keyword_368645, 'copy': keyword_368647}
        # Getting the type of 'coo_matrix' (line 252)
        coo_matrix_368633 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 252, 15), 'coo_matrix', False)
        # Calling coo_matrix(args, kwargs) (line 252)
        coo_matrix_call_result_368649 = invoke(stypy.reporting.localization.Localization(__file__, 252, 15), coo_matrix_368633, *[tuple_368634], **kwargs_368648)
        
        # Assigning a type to the variable 'stypy_return_type' (line 252)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 252, 8), 'stypy_return_type', coo_matrix_call_result_368649)
        
        # ################# End of 'transpose(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'transpose' in the type store
        # Getting the type of 'stypy_return_type' (line 245)
        stypy_return_type_368650 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 245, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_368650)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'transpose'
        return stypy_return_type_368650

    
    # Assigning a Attribute to a Attribute (line 255):
    
    # Assigning a Attribute to a Attribute (line 255):

    @norecursion
    def toarray(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 257)
        None_368651 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 257, 28), 'None')
        # Getting the type of 'None' (line 257)
        None_368652 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 257, 38), 'None')
        defaults = [None_368651, None_368652]
        # Create a new context for function 'toarray'
        module_type_store = module_type_store.open_function_context('toarray', 257, 4, False)
        # Assigning a type to the variable 'self' (line 258)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 258, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        coo_matrix.toarray.__dict__.__setitem__('stypy_localization', localization)
        coo_matrix.toarray.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        coo_matrix.toarray.__dict__.__setitem__('stypy_type_store', module_type_store)
        coo_matrix.toarray.__dict__.__setitem__('stypy_function_name', 'coo_matrix.toarray')
        coo_matrix.toarray.__dict__.__setitem__('stypy_param_names_list', ['order', 'out'])
        coo_matrix.toarray.__dict__.__setitem__('stypy_varargs_param_name', None)
        coo_matrix.toarray.__dict__.__setitem__('stypy_kwargs_param_name', None)
        coo_matrix.toarray.__dict__.__setitem__('stypy_call_defaults', defaults)
        coo_matrix.toarray.__dict__.__setitem__('stypy_call_varargs', varargs)
        coo_matrix.toarray.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        coo_matrix.toarray.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'coo_matrix.toarray', ['order', 'out'], None, None, defaults, varargs, kwargs)

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

        str_368653 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 258, 8), 'str', 'See the docstring for `spmatrix.toarray`.')
        
        # Assigning a Call to a Name (line 259):
        
        # Assigning a Call to a Name (line 259):
        
        # Assigning a Call to a Name (line 259):
        
        # Call to _process_toarray_args(...): (line 259)
        # Processing the call arguments (line 259)
        # Getting the type of 'order' (line 259)
        order_368656 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 259, 39), 'order', False)
        # Getting the type of 'out' (line 259)
        out_368657 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 259, 46), 'out', False)
        # Processing the call keyword arguments (line 259)
        kwargs_368658 = {}
        # Getting the type of 'self' (line 259)
        self_368654 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 259, 12), 'self', False)
        # Obtaining the member '_process_toarray_args' of a type (line 259)
        _process_toarray_args_368655 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 259, 12), self_368654, '_process_toarray_args')
        # Calling _process_toarray_args(args, kwargs) (line 259)
        _process_toarray_args_call_result_368659 = invoke(stypy.reporting.localization.Localization(__file__, 259, 12), _process_toarray_args_368655, *[order_368656, out_368657], **kwargs_368658)
        
        # Assigning a type to the variable 'B' (line 259)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 259, 8), 'B', _process_toarray_args_call_result_368659)
        
        # Assigning a Call to a Name (line 260):
        
        # Assigning a Call to a Name (line 260):
        
        # Assigning a Call to a Name (line 260):
        
        # Call to int(...): (line 260)
        # Processing the call arguments (line 260)
        # Getting the type of 'B' (line 260)
        B_368661 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 260, 22), 'B', False)
        # Obtaining the member 'flags' of a type (line 260)
        flags_368662 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 260, 22), B_368661, 'flags')
        # Obtaining the member 'f_contiguous' of a type (line 260)
        f_contiguous_368663 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 260, 22), flags_368662, 'f_contiguous')
        # Processing the call keyword arguments (line 260)
        kwargs_368664 = {}
        # Getting the type of 'int' (line 260)
        int_368660 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 260, 18), 'int', False)
        # Calling int(args, kwargs) (line 260)
        int_call_result_368665 = invoke(stypy.reporting.localization.Localization(__file__, 260, 18), int_368660, *[f_contiguous_368663], **kwargs_368664)
        
        # Assigning a type to the variable 'fortran' (line 260)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 260, 8), 'fortran', int_call_result_368665)
        
        
        # Evaluating a boolean operation
        
        # Getting the type of 'fortran' (line 261)
        fortran_368666 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 261, 15), 'fortran')
        # Applying the 'not' unary operator (line 261)
        result_not__368667 = python_operator(stypy.reporting.localization.Localization(__file__, 261, 11), 'not', fortran_368666)
        
        
        # Getting the type of 'B' (line 261)
        B_368668 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 261, 31), 'B')
        # Obtaining the member 'flags' of a type (line 261)
        flags_368669 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 261, 31), B_368668, 'flags')
        # Obtaining the member 'c_contiguous' of a type (line 261)
        c_contiguous_368670 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 261, 31), flags_368669, 'c_contiguous')
        # Applying the 'not' unary operator (line 261)
        result_not__368671 = python_operator(stypy.reporting.localization.Localization(__file__, 261, 27), 'not', c_contiguous_368670)
        
        # Applying the binary operator 'and' (line 261)
        result_and_keyword_368672 = python_operator(stypy.reporting.localization.Localization(__file__, 261, 11), 'and', result_not__368667, result_not__368671)
        
        # Testing the type of an if condition (line 261)
        if_condition_368673 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 261, 8), result_and_keyword_368672)
        # Assigning a type to the variable 'if_condition_368673' (line 261)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 261, 8), 'if_condition_368673', if_condition_368673)
        # SSA begins for if statement (line 261)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to ValueError(...): (line 262)
        # Processing the call arguments (line 262)
        str_368675 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 262, 29), 'str', 'Output array must be C or F contiguous')
        # Processing the call keyword arguments (line 262)
        kwargs_368676 = {}
        # Getting the type of 'ValueError' (line 262)
        ValueError_368674 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 262, 18), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 262)
        ValueError_call_result_368677 = invoke(stypy.reporting.localization.Localization(__file__, 262, 18), ValueError_368674, *[str_368675], **kwargs_368676)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 262, 12), ValueError_call_result_368677, 'raise parameter', BaseException)
        # SSA join for if statement (line 261)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Attribute to a Tuple (line 263):
        
        # Assigning a Subscript to a Name (line 263):
        
        # Assigning a Subscript to a Name (line 263):
        
        # Obtaining the type of the subscript
        int_368678 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 263, 8), 'int')
        # Getting the type of 'self' (line 263)
        self_368679 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 263, 14), 'self')
        # Obtaining the member 'shape' of a type (line 263)
        shape_368680 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 263, 14), self_368679, 'shape')
        # Obtaining the member '__getitem__' of a type (line 263)
        getitem___368681 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 263, 8), shape_368680, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 263)
        subscript_call_result_368682 = invoke(stypy.reporting.localization.Localization(__file__, 263, 8), getitem___368681, int_368678)
        
        # Assigning a type to the variable 'tuple_var_assignment_368030' (line 263)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 263, 8), 'tuple_var_assignment_368030', subscript_call_result_368682)
        
        # Assigning a Subscript to a Name (line 263):
        
        # Assigning a Subscript to a Name (line 263):
        
        # Obtaining the type of the subscript
        int_368683 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 263, 8), 'int')
        # Getting the type of 'self' (line 263)
        self_368684 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 263, 14), 'self')
        # Obtaining the member 'shape' of a type (line 263)
        shape_368685 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 263, 14), self_368684, 'shape')
        # Obtaining the member '__getitem__' of a type (line 263)
        getitem___368686 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 263, 8), shape_368685, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 263)
        subscript_call_result_368687 = invoke(stypy.reporting.localization.Localization(__file__, 263, 8), getitem___368686, int_368683)
        
        # Assigning a type to the variable 'tuple_var_assignment_368031' (line 263)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 263, 8), 'tuple_var_assignment_368031', subscript_call_result_368687)
        
        # Assigning a Name to a Name (line 263):
        
        # Assigning a Name to a Name (line 263):
        # Getting the type of 'tuple_var_assignment_368030' (line 263)
        tuple_var_assignment_368030_368688 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 263, 8), 'tuple_var_assignment_368030')
        # Assigning a type to the variable 'M' (line 263)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 263, 8), 'M', tuple_var_assignment_368030_368688)
        
        # Assigning a Name to a Name (line 263):
        
        # Assigning a Name to a Name (line 263):
        # Getting the type of 'tuple_var_assignment_368031' (line 263)
        tuple_var_assignment_368031_368689 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 263, 8), 'tuple_var_assignment_368031')
        # Assigning a type to the variable 'N' (line 263)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 263, 10), 'N', tuple_var_assignment_368031_368689)
        
        # Call to coo_todense(...): (line 264)
        # Processing the call arguments (line 264)
        # Getting the type of 'M' (line 264)
        M_368691 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 264, 20), 'M', False)
        # Getting the type of 'N' (line 264)
        N_368692 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 264, 23), 'N', False)
        # Getting the type of 'self' (line 264)
        self_368693 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 264, 26), 'self', False)
        # Obtaining the member 'nnz' of a type (line 264)
        nnz_368694 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 264, 26), self_368693, 'nnz')
        # Getting the type of 'self' (line 264)
        self_368695 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 264, 36), 'self', False)
        # Obtaining the member 'row' of a type (line 264)
        row_368696 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 264, 36), self_368695, 'row')
        # Getting the type of 'self' (line 264)
        self_368697 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 264, 46), 'self', False)
        # Obtaining the member 'col' of a type (line 264)
        col_368698 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 264, 46), self_368697, 'col')
        # Getting the type of 'self' (line 264)
        self_368699 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 264, 56), 'self', False)
        # Obtaining the member 'data' of a type (line 264)
        data_368700 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 264, 56), self_368699, 'data')
        
        # Call to ravel(...): (line 265)
        # Processing the call arguments (line 265)
        str_368703 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 265, 28), 'str', 'A')
        # Processing the call keyword arguments (line 265)
        kwargs_368704 = {}
        # Getting the type of 'B' (line 265)
        B_368701 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 265, 20), 'B', False)
        # Obtaining the member 'ravel' of a type (line 265)
        ravel_368702 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 265, 20), B_368701, 'ravel')
        # Calling ravel(args, kwargs) (line 265)
        ravel_call_result_368705 = invoke(stypy.reporting.localization.Localization(__file__, 265, 20), ravel_368702, *[str_368703], **kwargs_368704)
        
        # Getting the type of 'fortran' (line 265)
        fortran_368706 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 265, 34), 'fortran', False)
        # Processing the call keyword arguments (line 264)
        kwargs_368707 = {}
        # Getting the type of 'coo_todense' (line 264)
        coo_todense_368690 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 264, 8), 'coo_todense', False)
        # Calling coo_todense(args, kwargs) (line 264)
        coo_todense_call_result_368708 = invoke(stypy.reporting.localization.Localization(__file__, 264, 8), coo_todense_368690, *[M_368691, N_368692, nnz_368694, row_368696, col_368698, data_368700, ravel_call_result_368705, fortran_368706], **kwargs_368707)
        
        # Getting the type of 'B' (line 266)
        B_368709 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 266, 15), 'B')
        # Assigning a type to the variable 'stypy_return_type' (line 266)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 266, 8), 'stypy_return_type', B_368709)
        
        # ################# End of 'toarray(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'toarray' in the type store
        # Getting the type of 'stypy_return_type' (line 257)
        stypy_return_type_368710 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 257, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_368710)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'toarray'
        return stypy_return_type_368710


    @norecursion
    def tocsc(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'False' (line 268)
        False_368711 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 268, 25), 'False')
        defaults = [False_368711]
        # Create a new context for function 'tocsc'
        module_type_store = module_type_store.open_function_context('tocsc', 268, 4, False)
        # Assigning a type to the variable 'self' (line 269)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 269, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        coo_matrix.tocsc.__dict__.__setitem__('stypy_localization', localization)
        coo_matrix.tocsc.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        coo_matrix.tocsc.__dict__.__setitem__('stypy_type_store', module_type_store)
        coo_matrix.tocsc.__dict__.__setitem__('stypy_function_name', 'coo_matrix.tocsc')
        coo_matrix.tocsc.__dict__.__setitem__('stypy_param_names_list', ['copy'])
        coo_matrix.tocsc.__dict__.__setitem__('stypy_varargs_param_name', None)
        coo_matrix.tocsc.__dict__.__setitem__('stypy_kwargs_param_name', None)
        coo_matrix.tocsc.__dict__.__setitem__('stypy_call_defaults', defaults)
        coo_matrix.tocsc.__dict__.__setitem__('stypy_call_varargs', varargs)
        coo_matrix.tocsc.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        coo_matrix.tocsc.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'coo_matrix.tocsc', ['copy'], None, None, defaults, varargs, kwargs)

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

        str_368712 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 287, (-1)), 'str', 'Convert this matrix to Compressed Sparse Column format\n\n        Duplicate entries will be summed together.\n\n        Examples\n        --------\n        >>> from numpy import array\n        >>> from scipy.sparse import coo_matrix\n        >>> row  = array([0, 0, 1, 3, 1, 0, 0])\n        >>> col  = array([0, 2, 1, 3, 1, 0, 0])\n        >>> data = array([1, 1, 1, 1, 1, 1, 1])\n        >>> A = coo_matrix((data, (row, col)), shape=(4, 4)).tocsc()\n        >>> A.toarray()\n        array([[3, 0, 1, 0],\n               [0, 2, 0, 0],\n               [0, 0, 0, 0],\n               [0, 0, 0, 1]])\n\n        ')
        stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 288, 8))
        
        # 'from scipy.sparse.csc import csc_matrix' statement (line 288)
        update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/sparse/')
        import_368713 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 288, 8), 'scipy.sparse.csc')

        if (type(import_368713) is not StypyTypeError):

            if (import_368713 != 'pyd_module'):
                __import__(import_368713)
                sys_modules_368714 = sys.modules[import_368713]
                import_from_module(stypy.reporting.localization.Localization(__file__, 288, 8), 'scipy.sparse.csc', sys_modules_368714.module_type_store, module_type_store, ['csc_matrix'])
                nest_module(stypy.reporting.localization.Localization(__file__, 288, 8), __file__, sys_modules_368714, sys_modules_368714.module_type_store, module_type_store)
            else:
                from scipy.sparse.csc import csc_matrix

                import_from_module(stypy.reporting.localization.Localization(__file__, 288, 8), 'scipy.sparse.csc', None, module_type_store, ['csc_matrix'], [csc_matrix])

        else:
            # Assigning a type to the variable 'scipy.sparse.csc' (line 288)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 288, 8), 'scipy.sparse.csc', import_368713)

        remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/sparse/')
        
        
        
        # Getting the type of 'self' (line 289)
        self_368715 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 289, 11), 'self')
        # Obtaining the member 'nnz' of a type (line 289)
        nnz_368716 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 289, 11), self_368715, 'nnz')
        int_368717 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 289, 23), 'int')
        # Applying the binary operator '==' (line 289)
        result_eq_368718 = python_operator(stypy.reporting.localization.Localization(__file__, 289, 11), '==', nnz_368716, int_368717)
        
        # Testing the type of an if condition (line 289)
        if_condition_368719 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 289, 8), result_eq_368718)
        # Assigning a type to the variable 'if_condition_368719' (line 289)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 289, 8), 'if_condition_368719', if_condition_368719)
        # SSA begins for if statement (line 289)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to csc_matrix(...): (line 290)
        # Processing the call arguments (line 290)
        # Getting the type of 'self' (line 290)
        self_368721 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 290, 30), 'self', False)
        # Obtaining the member 'shape' of a type (line 290)
        shape_368722 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 290, 30), self_368721, 'shape')
        # Processing the call keyword arguments (line 290)
        # Getting the type of 'self' (line 290)
        self_368723 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 290, 48), 'self', False)
        # Obtaining the member 'dtype' of a type (line 290)
        dtype_368724 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 290, 48), self_368723, 'dtype')
        keyword_368725 = dtype_368724
        kwargs_368726 = {'dtype': keyword_368725}
        # Getting the type of 'csc_matrix' (line 290)
        csc_matrix_368720 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 290, 19), 'csc_matrix', False)
        # Calling csc_matrix(args, kwargs) (line 290)
        csc_matrix_call_result_368727 = invoke(stypy.reporting.localization.Localization(__file__, 290, 19), csc_matrix_368720, *[shape_368722], **kwargs_368726)
        
        # Assigning a type to the variable 'stypy_return_type' (line 290)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 290, 12), 'stypy_return_type', csc_matrix_call_result_368727)
        # SSA branch for the else part of an if statement (line 289)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Attribute to a Tuple (line 292):
        
        # Assigning a Subscript to a Name (line 292):
        
        # Assigning a Subscript to a Name (line 292):
        
        # Obtaining the type of the subscript
        int_368728 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 292, 12), 'int')
        # Getting the type of 'self' (line 292)
        self_368729 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 292, 18), 'self')
        # Obtaining the member 'shape' of a type (line 292)
        shape_368730 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 292, 18), self_368729, 'shape')
        # Obtaining the member '__getitem__' of a type (line 292)
        getitem___368731 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 292, 12), shape_368730, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 292)
        subscript_call_result_368732 = invoke(stypy.reporting.localization.Localization(__file__, 292, 12), getitem___368731, int_368728)
        
        # Assigning a type to the variable 'tuple_var_assignment_368032' (line 292)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 292, 12), 'tuple_var_assignment_368032', subscript_call_result_368732)
        
        # Assigning a Subscript to a Name (line 292):
        
        # Assigning a Subscript to a Name (line 292):
        
        # Obtaining the type of the subscript
        int_368733 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 292, 12), 'int')
        # Getting the type of 'self' (line 292)
        self_368734 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 292, 18), 'self')
        # Obtaining the member 'shape' of a type (line 292)
        shape_368735 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 292, 18), self_368734, 'shape')
        # Obtaining the member '__getitem__' of a type (line 292)
        getitem___368736 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 292, 12), shape_368735, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 292)
        subscript_call_result_368737 = invoke(stypy.reporting.localization.Localization(__file__, 292, 12), getitem___368736, int_368733)
        
        # Assigning a type to the variable 'tuple_var_assignment_368033' (line 292)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 292, 12), 'tuple_var_assignment_368033', subscript_call_result_368737)
        
        # Assigning a Name to a Name (line 292):
        
        # Assigning a Name to a Name (line 292):
        # Getting the type of 'tuple_var_assignment_368032' (line 292)
        tuple_var_assignment_368032_368738 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 292, 12), 'tuple_var_assignment_368032')
        # Assigning a type to the variable 'M' (line 292)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 292, 12), 'M', tuple_var_assignment_368032_368738)
        
        # Assigning a Name to a Name (line 292):
        
        # Assigning a Name to a Name (line 292):
        # Getting the type of 'tuple_var_assignment_368033' (line 292)
        tuple_var_assignment_368033_368739 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 292, 12), 'tuple_var_assignment_368033')
        # Assigning a type to the variable 'N' (line 292)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 292, 14), 'N', tuple_var_assignment_368033_368739)
        
        # Assigning a Call to a Name (line 293):
        
        # Assigning a Call to a Name (line 293):
        
        # Assigning a Call to a Name (line 293):
        
        # Call to get_index_dtype(...): (line 293)
        # Processing the call arguments (line 293)
        
        # Obtaining an instance of the builtin type 'tuple' (line 293)
        tuple_368741 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 293, 41), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 293)
        # Adding element type (line 293)
        # Getting the type of 'self' (line 293)
        self_368742 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 293, 41), 'self', False)
        # Obtaining the member 'col' of a type (line 293)
        col_368743 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 293, 41), self_368742, 'col')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 293, 41), tuple_368741, col_368743)
        # Adding element type (line 293)
        # Getting the type of 'self' (line 293)
        self_368744 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 293, 51), 'self', False)
        # Obtaining the member 'row' of a type (line 293)
        row_368745 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 293, 51), self_368744, 'row')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 293, 41), tuple_368741, row_368745)
        
        # Processing the call keyword arguments (line 293)
        
        # Call to max(...): (line 294)
        # Processing the call arguments (line 294)
        # Getting the type of 'self' (line 294)
        self_368747 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 294, 51), 'self', False)
        # Obtaining the member 'nnz' of a type (line 294)
        nnz_368748 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 294, 51), self_368747, 'nnz')
        # Getting the type of 'M' (line 294)
        M_368749 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 294, 61), 'M', False)
        # Processing the call keyword arguments (line 294)
        kwargs_368750 = {}
        # Getting the type of 'max' (line 294)
        max_368746 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 294, 47), 'max', False)
        # Calling max(args, kwargs) (line 294)
        max_call_result_368751 = invoke(stypy.reporting.localization.Localization(__file__, 294, 47), max_368746, *[nnz_368748, M_368749], **kwargs_368750)
        
        keyword_368752 = max_call_result_368751
        kwargs_368753 = {'maxval': keyword_368752}
        # Getting the type of 'get_index_dtype' (line 293)
        get_index_dtype_368740 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 293, 24), 'get_index_dtype', False)
        # Calling get_index_dtype(args, kwargs) (line 293)
        get_index_dtype_call_result_368754 = invoke(stypy.reporting.localization.Localization(__file__, 293, 24), get_index_dtype_368740, *[tuple_368741], **kwargs_368753)
        
        # Assigning a type to the variable 'idx_dtype' (line 293)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 293, 12), 'idx_dtype', get_index_dtype_call_result_368754)
        
        # Assigning a Call to a Name (line 295):
        
        # Assigning a Call to a Name (line 295):
        
        # Assigning a Call to a Name (line 295):
        
        # Call to astype(...): (line 295)
        # Processing the call arguments (line 295)
        # Getting the type of 'idx_dtype' (line 295)
        idx_dtype_368758 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 295, 34), 'idx_dtype', False)
        # Processing the call keyword arguments (line 295)
        # Getting the type of 'False' (line 295)
        False_368759 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 295, 50), 'False', False)
        keyword_368760 = False_368759
        kwargs_368761 = {'copy': keyword_368760}
        # Getting the type of 'self' (line 295)
        self_368755 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 295, 18), 'self', False)
        # Obtaining the member 'row' of a type (line 295)
        row_368756 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 295, 18), self_368755, 'row')
        # Obtaining the member 'astype' of a type (line 295)
        astype_368757 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 295, 18), row_368756, 'astype')
        # Calling astype(args, kwargs) (line 295)
        astype_call_result_368762 = invoke(stypy.reporting.localization.Localization(__file__, 295, 18), astype_368757, *[idx_dtype_368758], **kwargs_368761)
        
        # Assigning a type to the variable 'row' (line 295)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 295, 12), 'row', astype_call_result_368762)
        
        # Assigning a Call to a Name (line 296):
        
        # Assigning a Call to a Name (line 296):
        
        # Assigning a Call to a Name (line 296):
        
        # Call to astype(...): (line 296)
        # Processing the call arguments (line 296)
        # Getting the type of 'idx_dtype' (line 296)
        idx_dtype_368766 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 296, 34), 'idx_dtype', False)
        # Processing the call keyword arguments (line 296)
        # Getting the type of 'False' (line 296)
        False_368767 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 296, 50), 'False', False)
        keyword_368768 = False_368767
        kwargs_368769 = {'copy': keyword_368768}
        # Getting the type of 'self' (line 296)
        self_368763 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 296, 18), 'self', False)
        # Obtaining the member 'col' of a type (line 296)
        col_368764 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 296, 18), self_368763, 'col')
        # Obtaining the member 'astype' of a type (line 296)
        astype_368765 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 296, 18), col_368764, 'astype')
        # Calling astype(args, kwargs) (line 296)
        astype_call_result_368770 = invoke(stypy.reporting.localization.Localization(__file__, 296, 18), astype_368765, *[idx_dtype_368766], **kwargs_368769)
        
        # Assigning a type to the variable 'col' (line 296)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 296, 12), 'col', astype_call_result_368770)
        
        # Assigning a Call to a Name (line 298):
        
        # Assigning a Call to a Name (line 298):
        
        # Assigning a Call to a Name (line 298):
        
        # Call to empty(...): (line 298)
        # Processing the call arguments (line 298)
        # Getting the type of 'N' (line 298)
        N_368773 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 298, 30), 'N', False)
        int_368774 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 298, 34), 'int')
        # Applying the binary operator '+' (line 298)
        result_add_368775 = python_operator(stypy.reporting.localization.Localization(__file__, 298, 30), '+', N_368773, int_368774)
        
        # Processing the call keyword arguments (line 298)
        # Getting the type of 'idx_dtype' (line 298)
        idx_dtype_368776 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 298, 43), 'idx_dtype', False)
        keyword_368777 = idx_dtype_368776
        kwargs_368778 = {'dtype': keyword_368777}
        # Getting the type of 'np' (line 298)
        np_368771 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 298, 21), 'np', False)
        # Obtaining the member 'empty' of a type (line 298)
        empty_368772 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 298, 21), np_368771, 'empty')
        # Calling empty(args, kwargs) (line 298)
        empty_call_result_368779 = invoke(stypy.reporting.localization.Localization(__file__, 298, 21), empty_368772, *[result_add_368775], **kwargs_368778)
        
        # Assigning a type to the variable 'indptr' (line 298)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 298, 12), 'indptr', empty_call_result_368779)
        
        # Assigning a Call to a Name (line 299):
        
        # Assigning a Call to a Name (line 299):
        
        # Assigning a Call to a Name (line 299):
        
        # Call to empty_like(...): (line 299)
        # Processing the call arguments (line 299)
        # Getting the type of 'row' (line 299)
        row_368782 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 299, 36), 'row', False)
        # Processing the call keyword arguments (line 299)
        # Getting the type of 'idx_dtype' (line 299)
        idx_dtype_368783 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 299, 47), 'idx_dtype', False)
        keyword_368784 = idx_dtype_368783
        kwargs_368785 = {'dtype': keyword_368784}
        # Getting the type of 'np' (line 299)
        np_368780 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 299, 22), 'np', False)
        # Obtaining the member 'empty_like' of a type (line 299)
        empty_like_368781 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 299, 22), np_368780, 'empty_like')
        # Calling empty_like(args, kwargs) (line 299)
        empty_like_call_result_368786 = invoke(stypy.reporting.localization.Localization(__file__, 299, 22), empty_like_368781, *[row_368782], **kwargs_368785)
        
        # Assigning a type to the variable 'indices' (line 299)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 299, 12), 'indices', empty_like_call_result_368786)
        
        # Assigning a Call to a Name (line 300):
        
        # Assigning a Call to a Name (line 300):
        
        # Assigning a Call to a Name (line 300):
        
        # Call to empty_like(...): (line 300)
        # Processing the call arguments (line 300)
        # Getting the type of 'self' (line 300)
        self_368789 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 300, 33), 'self', False)
        # Obtaining the member 'data' of a type (line 300)
        data_368790 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 300, 33), self_368789, 'data')
        # Processing the call keyword arguments (line 300)
        
        # Call to upcast(...): (line 300)
        # Processing the call arguments (line 300)
        # Getting the type of 'self' (line 300)
        self_368792 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 300, 57), 'self', False)
        # Obtaining the member 'dtype' of a type (line 300)
        dtype_368793 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 300, 57), self_368792, 'dtype')
        # Processing the call keyword arguments (line 300)
        kwargs_368794 = {}
        # Getting the type of 'upcast' (line 300)
        upcast_368791 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 300, 50), 'upcast', False)
        # Calling upcast(args, kwargs) (line 300)
        upcast_call_result_368795 = invoke(stypy.reporting.localization.Localization(__file__, 300, 50), upcast_368791, *[dtype_368793], **kwargs_368794)
        
        keyword_368796 = upcast_call_result_368795
        kwargs_368797 = {'dtype': keyword_368796}
        # Getting the type of 'np' (line 300)
        np_368787 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 300, 19), 'np', False)
        # Obtaining the member 'empty_like' of a type (line 300)
        empty_like_368788 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 300, 19), np_368787, 'empty_like')
        # Calling empty_like(args, kwargs) (line 300)
        empty_like_call_result_368798 = invoke(stypy.reporting.localization.Localization(__file__, 300, 19), empty_like_368788, *[data_368790], **kwargs_368797)
        
        # Assigning a type to the variable 'data' (line 300)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 300, 12), 'data', empty_like_call_result_368798)
        
        # Call to coo_tocsr(...): (line 302)
        # Processing the call arguments (line 302)
        # Getting the type of 'N' (line 302)
        N_368800 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 302, 22), 'N', False)
        # Getting the type of 'M' (line 302)
        M_368801 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 302, 25), 'M', False)
        # Getting the type of 'self' (line 302)
        self_368802 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 302, 28), 'self', False)
        # Obtaining the member 'nnz' of a type (line 302)
        nnz_368803 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 302, 28), self_368802, 'nnz')
        # Getting the type of 'col' (line 302)
        col_368804 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 302, 38), 'col', False)
        # Getting the type of 'row' (line 302)
        row_368805 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 302, 43), 'row', False)
        # Getting the type of 'self' (line 302)
        self_368806 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 302, 48), 'self', False)
        # Obtaining the member 'data' of a type (line 302)
        data_368807 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 302, 48), self_368806, 'data')
        # Getting the type of 'indptr' (line 303)
        indptr_368808 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 303, 22), 'indptr', False)
        # Getting the type of 'indices' (line 303)
        indices_368809 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 303, 30), 'indices', False)
        # Getting the type of 'data' (line 303)
        data_368810 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 303, 39), 'data', False)
        # Processing the call keyword arguments (line 302)
        kwargs_368811 = {}
        # Getting the type of 'coo_tocsr' (line 302)
        coo_tocsr_368799 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 302, 12), 'coo_tocsr', False)
        # Calling coo_tocsr(args, kwargs) (line 302)
        coo_tocsr_call_result_368812 = invoke(stypy.reporting.localization.Localization(__file__, 302, 12), coo_tocsr_368799, *[N_368800, M_368801, nnz_368803, col_368804, row_368805, data_368807, indptr_368808, indices_368809, data_368810], **kwargs_368811)
        
        
        # Assigning a Call to a Name (line 305):
        
        # Assigning a Call to a Name (line 305):
        
        # Assigning a Call to a Name (line 305):
        
        # Call to csc_matrix(...): (line 305)
        # Processing the call arguments (line 305)
        
        # Obtaining an instance of the builtin type 'tuple' (line 305)
        tuple_368814 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 305, 28), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 305)
        # Adding element type (line 305)
        # Getting the type of 'data' (line 305)
        data_368815 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 305, 28), 'data', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 305, 28), tuple_368814, data_368815)
        # Adding element type (line 305)
        # Getting the type of 'indices' (line 305)
        indices_368816 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 305, 34), 'indices', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 305, 28), tuple_368814, indices_368816)
        # Adding element type (line 305)
        # Getting the type of 'indptr' (line 305)
        indptr_368817 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 305, 43), 'indptr', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 305, 28), tuple_368814, indptr_368817)
        
        # Processing the call keyword arguments (line 305)
        # Getting the type of 'self' (line 305)
        self_368818 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 305, 58), 'self', False)
        # Obtaining the member 'shape' of a type (line 305)
        shape_368819 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 305, 58), self_368818, 'shape')
        keyword_368820 = shape_368819
        kwargs_368821 = {'shape': keyword_368820}
        # Getting the type of 'csc_matrix' (line 305)
        csc_matrix_368813 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 305, 16), 'csc_matrix', False)
        # Calling csc_matrix(args, kwargs) (line 305)
        csc_matrix_call_result_368822 = invoke(stypy.reporting.localization.Localization(__file__, 305, 16), csc_matrix_368813, *[tuple_368814], **kwargs_368821)
        
        # Assigning a type to the variable 'x' (line 305)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 305, 12), 'x', csc_matrix_call_result_368822)
        
        
        # Getting the type of 'self' (line 306)
        self_368823 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 306, 19), 'self')
        # Obtaining the member 'has_canonical_format' of a type (line 306)
        has_canonical_format_368824 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 306, 19), self_368823, 'has_canonical_format')
        # Applying the 'not' unary operator (line 306)
        result_not__368825 = python_operator(stypy.reporting.localization.Localization(__file__, 306, 15), 'not', has_canonical_format_368824)
        
        # Testing the type of an if condition (line 306)
        if_condition_368826 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 306, 12), result_not__368825)
        # Assigning a type to the variable 'if_condition_368826' (line 306)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 306, 12), 'if_condition_368826', if_condition_368826)
        # SSA begins for if statement (line 306)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to sum_duplicates(...): (line 307)
        # Processing the call keyword arguments (line 307)
        kwargs_368829 = {}
        # Getting the type of 'x' (line 307)
        x_368827 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 307, 16), 'x', False)
        # Obtaining the member 'sum_duplicates' of a type (line 307)
        sum_duplicates_368828 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 307, 16), x_368827, 'sum_duplicates')
        # Calling sum_duplicates(args, kwargs) (line 307)
        sum_duplicates_call_result_368830 = invoke(stypy.reporting.localization.Localization(__file__, 307, 16), sum_duplicates_368828, *[], **kwargs_368829)
        
        # SSA join for if statement (line 306)
        module_type_store = module_type_store.join_ssa_context()
        
        # Getting the type of 'x' (line 308)
        x_368831 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 308, 19), 'x')
        # Assigning a type to the variable 'stypy_return_type' (line 308)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 308, 12), 'stypy_return_type', x_368831)
        # SSA join for if statement (line 289)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'tocsc(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'tocsc' in the type store
        # Getting the type of 'stypy_return_type' (line 268)
        stypy_return_type_368832 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 268, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_368832)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'tocsc'
        return stypy_return_type_368832


    @norecursion
    def tocsr(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'False' (line 310)
        False_368833 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 310, 25), 'False')
        defaults = [False_368833]
        # Create a new context for function 'tocsr'
        module_type_store = module_type_store.open_function_context('tocsr', 310, 4, False)
        # Assigning a type to the variable 'self' (line 311)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 311, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        coo_matrix.tocsr.__dict__.__setitem__('stypy_localization', localization)
        coo_matrix.tocsr.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        coo_matrix.tocsr.__dict__.__setitem__('stypy_type_store', module_type_store)
        coo_matrix.tocsr.__dict__.__setitem__('stypy_function_name', 'coo_matrix.tocsr')
        coo_matrix.tocsr.__dict__.__setitem__('stypy_param_names_list', ['copy'])
        coo_matrix.tocsr.__dict__.__setitem__('stypy_varargs_param_name', None)
        coo_matrix.tocsr.__dict__.__setitem__('stypy_kwargs_param_name', None)
        coo_matrix.tocsr.__dict__.__setitem__('stypy_call_defaults', defaults)
        coo_matrix.tocsr.__dict__.__setitem__('stypy_call_varargs', varargs)
        coo_matrix.tocsr.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        coo_matrix.tocsr.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'coo_matrix.tocsr', ['copy'], None, None, defaults, varargs, kwargs)

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

        str_368834 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 329, (-1)), 'str', 'Convert this matrix to Compressed Sparse Row format\n\n        Duplicate entries will be summed together.\n\n        Examples\n        --------\n        >>> from numpy import array\n        >>> from scipy.sparse import coo_matrix\n        >>> row  = array([0, 0, 1, 3, 1, 0, 0])\n        >>> col  = array([0, 2, 1, 3, 1, 0, 0])\n        >>> data = array([1, 1, 1, 1, 1, 1, 1])\n        >>> A = coo_matrix((data, (row, col)), shape=(4, 4)).tocsr()\n        >>> A.toarray()\n        array([[3, 0, 1, 0],\n               [0, 2, 0, 0],\n               [0, 0, 0, 0],\n               [0, 0, 0, 1]])\n\n        ')
        stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 330, 8))
        
        # 'from scipy.sparse.csr import csr_matrix' statement (line 330)
        update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/sparse/')
        import_368835 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 330, 8), 'scipy.sparse.csr')

        if (type(import_368835) is not StypyTypeError):

            if (import_368835 != 'pyd_module'):
                __import__(import_368835)
                sys_modules_368836 = sys.modules[import_368835]
                import_from_module(stypy.reporting.localization.Localization(__file__, 330, 8), 'scipy.sparse.csr', sys_modules_368836.module_type_store, module_type_store, ['csr_matrix'])
                nest_module(stypy.reporting.localization.Localization(__file__, 330, 8), __file__, sys_modules_368836, sys_modules_368836.module_type_store, module_type_store)
            else:
                from scipy.sparse.csr import csr_matrix

                import_from_module(stypy.reporting.localization.Localization(__file__, 330, 8), 'scipy.sparse.csr', None, module_type_store, ['csr_matrix'], [csr_matrix])

        else:
            # Assigning a type to the variable 'scipy.sparse.csr' (line 330)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 330, 8), 'scipy.sparse.csr', import_368835)

        remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/sparse/')
        
        
        
        # Getting the type of 'self' (line 331)
        self_368837 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 331, 11), 'self')
        # Obtaining the member 'nnz' of a type (line 331)
        nnz_368838 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 331, 11), self_368837, 'nnz')
        int_368839 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 331, 23), 'int')
        # Applying the binary operator '==' (line 331)
        result_eq_368840 = python_operator(stypy.reporting.localization.Localization(__file__, 331, 11), '==', nnz_368838, int_368839)
        
        # Testing the type of an if condition (line 331)
        if_condition_368841 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 331, 8), result_eq_368840)
        # Assigning a type to the variable 'if_condition_368841' (line 331)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 331, 8), 'if_condition_368841', if_condition_368841)
        # SSA begins for if statement (line 331)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to csr_matrix(...): (line 332)
        # Processing the call arguments (line 332)
        # Getting the type of 'self' (line 332)
        self_368843 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 332, 30), 'self', False)
        # Obtaining the member 'shape' of a type (line 332)
        shape_368844 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 332, 30), self_368843, 'shape')
        # Processing the call keyword arguments (line 332)
        # Getting the type of 'self' (line 332)
        self_368845 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 332, 48), 'self', False)
        # Obtaining the member 'dtype' of a type (line 332)
        dtype_368846 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 332, 48), self_368845, 'dtype')
        keyword_368847 = dtype_368846
        kwargs_368848 = {'dtype': keyword_368847}
        # Getting the type of 'csr_matrix' (line 332)
        csr_matrix_368842 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 332, 19), 'csr_matrix', False)
        # Calling csr_matrix(args, kwargs) (line 332)
        csr_matrix_call_result_368849 = invoke(stypy.reporting.localization.Localization(__file__, 332, 19), csr_matrix_368842, *[shape_368844], **kwargs_368848)
        
        # Assigning a type to the variable 'stypy_return_type' (line 332)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 332, 12), 'stypy_return_type', csr_matrix_call_result_368849)
        # SSA branch for the else part of an if statement (line 331)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Attribute to a Tuple (line 334):
        
        # Assigning a Subscript to a Name (line 334):
        
        # Assigning a Subscript to a Name (line 334):
        
        # Obtaining the type of the subscript
        int_368850 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 334, 12), 'int')
        # Getting the type of 'self' (line 334)
        self_368851 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 334, 18), 'self')
        # Obtaining the member 'shape' of a type (line 334)
        shape_368852 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 334, 18), self_368851, 'shape')
        # Obtaining the member '__getitem__' of a type (line 334)
        getitem___368853 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 334, 12), shape_368852, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 334)
        subscript_call_result_368854 = invoke(stypy.reporting.localization.Localization(__file__, 334, 12), getitem___368853, int_368850)
        
        # Assigning a type to the variable 'tuple_var_assignment_368034' (line 334)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 334, 12), 'tuple_var_assignment_368034', subscript_call_result_368854)
        
        # Assigning a Subscript to a Name (line 334):
        
        # Assigning a Subscript to a Name (line 334):
        
        # Obtaining the type of the subscript
        int_368855 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 334, 12), 'int')
        # Getting the type of 'self' (line 334)
        self_368856 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 334, 18), 'self')
        # Obtaining the member 'shape' of a type (line 334)
        shape_368857 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 334, 18), self_368856, 'shape')
        # Obtaining the member '__getitem__' of a type (line 334)
        getitem___368858 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 334, 12), shape_368857, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 334)
        subscript_call_result_368859 = invoke(stypy.reporting.localization.Localization(__file__, 334, 12), getitem___368858, int_368855)
        
        # Assigning a type to the variable 'tuple_var_assignment_368035' (line 334)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 334, 12), 'tuple_var_assignment_368035', subscript_call_result_368859)
        
        # Assigning a Name to a Name (line 334):
        
        # Assigning a Name to a Name (line 334):
        # Getting the type of 'tuple_var_assignment_368034' (line 334)
        tuple_var_assignment_368034_368860 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 334, 12), 'tuple_var_assignment_368034')
        # Assigning a type to the variable 'M' (line 334)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 334, 12), 'M', tuple_var_assignment_368034_368860)
        
        # Assigning a Name to a Name (line 334):
        
        # Assigning a Name to a Name (line 334):
        # Getting the type of 'tuple_var_assignment_368035' (line 334)
        tuple_var_assignment_368035_368861 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 334, 12), 'tuple_var_assignment_368035')
        # Assigning a type to the variable 'N' (line 334)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 334, 14), 'N', tuple_var_assignment_368035_368861)
        
        # Assigning a Call to a Name (line 335):
        
        # Assigning a Call to a Name (line 335):
        
        # Assigning a Call to a Name (line 335):
        
        # Call to get_index_dtype(...): (line 335)
        # Processing the call arguments (line 335)
        
        # Obtaining an instance of the builtin type 'tuple' (line 335)
        tuple_368863 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 335, 41), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 335)
        # Adding element type (line 335)
        # Getting the type of 'self' (line 335)
        self_368864 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 335, 41), 'self', False)
        # Obtaining the member 'row' of a type (line 335)
        row_368865 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 335, 41), self_368864, 'row')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 335, 41), tuple_368863, row_368865)
        # Adding element type (line 335)
        # Getting the type of 'self' (line 335)
        self_368866 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 335, 51), 'self', False)
        # Obtaining the member 'col' of a type (line 335)
        col_368867 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 335, 51), self_368866, 'col')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 335, 41), tuple_368863, col_368867)
        
        # Processing the call keyword arguments (line 335)
        
        # Call to max(...): (line 336)
        # Processing the call arguments (line 336)
        # Getting the type of 'self' (line 336)
        self_368869 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 336, 51), 'self', False)
        # Obtaining the member 'nnz' of a type (line 336)
        nnz_368870 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 336, 51), self_368869, 'nnz')
        # Getting the type of 'N' (line 336)
        N_368871 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 336, 61), 'N', False)
        # Processing the call keyword arguments (line 336)
        kwargs_368872 = {}
        # Getting the type of 'max' (line 336)
        max_368868 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 336, 47), 'max', False)
        # Calling max(args, kwargs) (line 336)
        max_call_result_368873 = invoke(stypy.reporting.localization.Localization(__file__, 336, 47), max_368868, *[nnz_368870, N_368871], **kwargs_368872)
        
        keyword_368874 = max_call_result_368873
        kwargs_368875 = {'maxval': keyword_368874}
        # Getting the type of 'get_index_dtype' (line 335)
        get_index_dtype_368862 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 335, 24), 'get_index_dtype', False)
        # Calling get_index_dtype(args, kwargs) (line 335)
        get_index_dtype_call_result_368876 = invoke(stypy.reporting.localization.Localization(__file__, 335, 24), get_index_dtype_368862, *[tuple_368863], **kwargs_368875)
        
        # Assigning a type to the variable 'idx_dtype' (line 335)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 335, 12), 'idx_dtype', get_index_dtype_call_result_368876)
        
        # Assigning a Call to a Name (line 337):
        
        # Assigning a Call to a Name (line 337):
        
        # Assigning a Call to a Name (line 337):
        
        # Call to astype(...): (line 337)
        # Processing the call arguments (line 337)
        # Getting the type of 'idx_dtype' (line 337)
        idx_dtype_368880 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 337, 34), 'idx_dtype', False)
        # Processing the call keyword arguments (line 337)
        # Getting the type of 'False' (line 337)
        False_368881 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 337, 50), 'False', False)
        keyword_368882 = False_368881
        kwargs_368883 = {'copy': keyword_368882}
        # Getting the type of 'self' (line 337)
        self_368877 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 337, 18), 'self', False)
        # Obtaining the member 'row' of a type (line 337)
        row_368878 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 337, 18), self_368877, 'row')
        # Obtaining the member 'astype' of a type (line 337)
        astype_368879 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 337, 18), row_368878, 'astype')
        # Calling astype(args, kwargs) (line 337)
        astype_call_result_368884 = invoke(stypy.reporting.localization.Localization(__file__, 337, 18), astype_368879, *[idx_dtype_368880], **kwargs_368883)
        
        # Assigning a type to the variable 'row' (line 337)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 337, 12), 'row', astype_call_result_368884)
        
        # Assigning a Call to a Name (line 338):
        
        # Assigning a Call to a Name (line 338):
        
        # Assigning a Call to a Name (line 338):
        
        # Call to astype(...): (line 338)
        # Processing the call arguments (line 338)
        # Getting the type of 'idx_dtype' (line 338)
        idx_dtype_368888 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 338, 34), 'idx_dtype', False)
        # Processing the call keyword arguments (line 338)
        # Getting the type of 'False' (line 338)
        False_368889 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 338, 50), 'False', False)
        keyword_368890 = False_368889
        kwargs_368891 = {'copy': keyword_368890}
        # Getting the type of 'self' (line 338)
        self_368885 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 338, 18), 'self', False)
        # Obtaining the member 'col' of a type (line 338)
        col_368886 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 338, 18), self_368885, 'col')
        # Obtaining the member 'astype' of a type (line 338)
        astype_368887 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 338, 18), col_368886, 'astype')
        # Calling astype(args, kwargs) (line 338)
        astype_call_result_368892 = invoke(stypy.reporting.localization.Localization(__file__, 338, 18), astype_368887, *[idx_dtype_368888], **kwargs_368891)
        
        # Assigning a type to the variable 'col' (line 338)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 338, 12), 'col', astype_call_result_368892)
        
        # Assigning a Call to a Name (line 340):
        
        # Assigning a Call to a Name (line 340):
        
        # Assigning a Call to a Name (line 340):
        
        # Call to empty(...): (line 340)
        # Processing the call arguments (line 340)
        # Getting the type of 'M' (line 340)
        M_368895 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 340, 30), 'M', False)
        int_368896 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 340, 34), 'int')
        # Applying the binary operator '+' (line 340)
        result_add_368897 = python_operator(stypy.reporting.localization.Localization(__file__, 340, 30), '+', M_368895, int_368896)
        
        # Processing the call keyword arguments (line 340)
        # Getting the type of 'idx_dtype' (line 340)
        idx_dtype_368898 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 340, 43), 'idx_dtype', False)
        keyword_368899 = idx_dtype_368898
        kwargs_368900 = {'dtype': keyword_368899}
        # Getting the type of 'np' (line 340)
        np_368893 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 340, 21), 'np', False)
        # Obtaining the member 'empty' of a type (line 340)
        empty_368894 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 340, 21), np_368893, 'empty')
        # Calling empty(args, kwargs) (line 340)
        empty_call_result_368901 = invoke(stypy.reporting.localization.Localization(__file__, 340, 21), empty_368894, *[result_add_368897], **kwargs_368900)
        
        # Assigning a type to the variable 'indptr' (line 340)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 340, 12), 'indptr', empty_call_result_368901)
        
        # Assigning a Call to a Name (line 341):
        
        # Assigning a Call to a Name (line 341):
        
        # Assigning a Call to a Name (line 341):
        
        # Call to empty_like(...): (line 341)
        # Processing the call arguments (line 341)
        # Getting the type of 'col' (line 341)
        col_368904 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 341, 36), 'col', False)
        # Processing the call keyword arguments (line 341)
        # Getting the type of 'idx_dtype' (line 341)
        idx_dtype_368905 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 341, 47), 'idx_dtype', False)
        keyword_368906 = idx_dtype_368905
        kwargs_368907 = {'dtype': keyword_368906}
        # Getting the type of 'np' (line 341)
        np_368902 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 341, 22), 'np', False)
        # Obtaining the member 'empty_like' of a type (line 341)
        empty_like_368903 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 341, 22), np_368902, 'empty_like')
        # Calling empty_like(args, kwargs) (line 341)
        empty_like_call_result_368908 = invoke(stypy.reporting.localization.Localization(__file__, 341, 22), empty_like_368903, *[col_368904], **kwargs_368907)
        
        # Assigning a type to the variable 'indices' (line 341)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 341, 12), 'indices', empty_like_call_result_368908)
        
        # Assigning a Call to a Name (line 342):
        
        # Assigning a Call to a Name (line 342):
        
        # Assigning a Call to a Name (line 342):
        
        # Call to empty_like(...): (line 342)
        # Processing the call arguments (line 342)
        # Getting the type of 'self' (line 342)
        self_368911 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 342, 33), 'self', False)
        # Obtaining the member 'data' of a type (line 342)
        data_368912 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 342, 33), self_368911, 'data')
        # Processing the call keyword arguments (line 342)
        
        # Call to upcast(...): (line 342)
        # Processing the call arguments (line 342)
        # Getting the type of 'self' (line 342)
        self_368914 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 342, 57), 'self', False)
        # Obtaining the member 'dtype' of a type (line 342)
        dtype_368915 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 342, 57), self_368914, 'dtype')
        # Processing the call keyword arguments (line 342)
        kwargs_368916 = {}
        # Getting the type of 'upcast' (line 342)
        upcast_368913 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 342, 50), 'upcast', False)
        # Calling upcast(args, kwargs) (line 342)
        upcast_call_result_368917 = invoke(stypy.reporting.localization.Localization(__file__, 342, 50), upcast_368913, *[dtype_368915], **kwargs_368916)
        
        keyword_368918 = upcast_call_result_368917
        kwargs_368919 = {'dtype': keyword_368918}
        # Getting the type of 'np' (line 342)
        np_368909 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 342, 19), 'np', False)
        # Obtaining the member 'empty_like' of a type (line 342)
        empty_like_368910 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 342, 19), np_368909, 'empty_like')
        # Calling empty_like(args, kwargs) (line 342)
        empty_like_call_result_368920 = invoke(stypy.reporting.localization.Localization(__file__, 342, 19), empty_like_368910, *[data_368912], **kwargs_368919)
        
        # Assigning a type to the variable 'data' (line 342)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 342, 12), 'data', empty_like_call_result_368920)
        
        # Call to coo_tocsr(...): (line 344)
        # Processing the call arguments (line 344)
        # Getting the type of 'M' (line 344)
        M_368922 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 344, 22), 'M', False)
        # Getting the type of 'N' (line 344)
        N_368923 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 344, 25), 'N', False)
        # Getting the type of 'self' (line 344)
        self_368924 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 344, 28), 'self', False)
        # Obtaining the member 'nnz' of a type (line 344)
        nnz_368925 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 344, 28), self_368924, 'nnz')
        # Getting the type of 'row' (line 344)
        row_368926 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 344, 38), 'row', False)
        # Getting the type of 'col' (line 344)
        col_368927 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 344, 43), 'col', False)
        # Getting the type of 'self' (line 344)
        self_368928 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 344, 48), 'self', False)
        # Obtaining the member 'data' of a type (line 344)
        data_368929 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 344, 48), self_368928, 'data')
        # Getting the type of 'indptr' (line 345)
        indptr_368930 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 345, 22), 'indptr', False)
        # Getting the type of 'indices' (line 345)
        indices_368931 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 345, 30), 'indices', False)
        # Getting the type of 'data' (line 345)
        data_368932 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 345, 39), 'data', False)
        # Processing the call keyword arguments (line 344)
        kwargs_368933 = {}
        # Getting the type of 'coo_tocsr' (line 344)
        coo_tocsr_368921 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 344, 12), 'coo_tocsr', False)
        # Calling coo_tocsr(args, kwargs) (line 344)
        coo_tocsr_call_result_368934 = invoke(stypy.reporting.localization.Localization(__file__, 344, 12), coo_tocsr_368921, *[M_368922, N_368923, nnz_368925, row_368926, col_368927, data_368929, indptr_368930, indices_368931, data_368932], **kwargs_368933)
        
        
        # Assigning a Call to a Name (line 347):
        
        # Assigning a Call to a Name (line 347):
        
        # Assigning a Call to a Name (line 347):
        
        # Call to csr_matrix(...): (line 347)
        # Processing the call arguments (line 347)
        
        # Obtaining an instance of the builtin type 'tuple' (line 347)
        tuple_368936 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 347, 28), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 347)
        # Adding element type (line 347)
        # Getting the type of 'data' (line 347)
        data_368937 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 347, 28), 'data', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 347, 28), tuple_368936, data_368937)
        # Adding element type (line 347)
        # Getting the type of 'indices' (line 347)
        indices_368938 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 347, 34), 'indices', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 347, 28), tuple_368936, indices_368938)
        # Adding element type (line 347)
        # Getting the type of 'indptr' (line 347)
        indptr_368939 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 347, 43), 'indptr', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 347, 28), tuple_368936, indptr_368939)
        
        # Processing the call keyword arguments (line 347)
        # Getting the type of 'self' (line 347)
        self_368940 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 347, 58), 'self', False)
        # Obtaining the member 'shape' of a type (line 347)
        shape_368941 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 347, 58), self_368940, 'shape')
        keyword_368942 = shape_368941
        kwargs_368943 = {'shape': keyword_368942}
        # Getting the type of 'csr_matrix' (line 347)
        csr_matrix_368935 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 347, 16), 'csr_matrix', False)
        # Calling csr_matrix(args, kwargs) (line 347)
        csr_matrix_call_result_368944 = invoke(stypy.reporting.localization.Localization(__file__, 347, 16), csr_matrix_368935, *[tuple_368936], **kwargs_368943)
        
        # Assigning a type to the variable 'x' (line 347)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 347, 12), 'x', csr_matrix_call_result_368944)
        
        
        # Getting the type of 'self' (line 348)
        self_368945 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 348, 19), 'self')
        # Obtaining the member 'has_canonical_format' of a type (line 348)
        has_canonical_format_368946 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 348, 19), self_368945, 'has_canonical_format')
        # Applying the 'not' unary operator (line 348)
        result_not__368947 = python_operator(stypy.reporting.localization.Localization(__file__, 348, 15), 'not', has_canonical_format_368946)
        
        # Testing the type of an if condition (line 348)
        if_condition_368948 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 348, 12), result_not__368947)
        # Assigning a type to the variable 'if_condition_368948' (line 348)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 348, 12), 'if_condition_368948', if_condition_368948)
        # SSA begins for if statement (line 348)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to sum_duplicates(...): (line 349)
        # Processing the call keyword arguments (line 349)
        kwargs_368951 = {}
        # Getting the type of 'x' (line 349)
        x_368949 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 349, 16), 'x', False)
        # Obtaining the member 'sum_duplicates' of a type (line 349)
        sum_duplicates_368950 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 349, 16), x_368949, 'sum_duplicates')
        # Calling sum_duplicates(args, kwargs) (line 349)
        sum_duplicates_call_result_368952 = invoke(stypy.reporting.localization.Localization(__file__, 349, 16), sum_duplicates_368950, *[], **kwargs_368951)
        
        # SSA join for if statement (line 348)
        module_type_store = module_type_store.join_ssa_context()
        
        # Getting the type of 'x' (line 350)
        x_368953 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 350, 19), 'x')
        # Assigning a type to the variable 'stypy_return_type' (line 350)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 350, 12), 'stypy_return_type', x_368953)
        # SSA join for if statement (line 331)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'tocsr(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'tocsr' in the type store
        # Getting the type of 'stypy_return_type' (line 310)
        stypy_return_type_368954 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 310, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_368954)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'tocsr'
        return stypy_return_type_368954


    @norecursion
    def tocoo(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'False' (line 352)
        False_368955 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 352, 25), 'False')
        defaults = [False_368955]
        # Create a new context for function 'tocoo'
        module_type_store = module_type_store.open_function_context('tocoo', 352, 4, False)
        # Assigning a type to the variable 'self' (line 353)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 353, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        coo_matrix.tocoo.__dict__.__setitem__('stypy_localization', localization)
        coo_matrix.tocoo.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        coo_matrix.tocoo.__dict__.__setitem__('stypy_type_store', module_type_store)
        coo_matrix.tocoo.__dict__.__setitem__('stypy_function_name', 'coo_matrix.tocoo')
        coo_matrix.tocoo.__dict__.__setitem__('stypy_param_names_list', ['copy'])
        coo_matrix.tocoo.__dict__.__setitem__('stypy_varargs_param_name', None)
        coo_matrix.tocoo.__dict__.__setitem__('stypy_kwargs_param_name', None)
        coo_matrix.tocoo.__dict__.__setitem__('stypy_call_defaults', defaults)
        coo_matrix.tocoo.__dict__.__setitem__('stypy_call_varargs', varargs)
        coo_matrix.tocoo.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        coo_matrix.tocoo.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'coo_matrix.tocoo', ['copy'], None, None, defaults, varargs, kwargs)

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

        
        # Getting the type of 'copy' (line 353)
        copy_368956 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 353, 11), 'copy')
        # Testing the type of an if condition (line 353)
        if_condition_368957 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 353, 8), copy_368956)
        # Assigning a type to the variable 'if_condition_368957' (line 353)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 353, 8), 'if_condition_368957', if_condition_368957)
        # SSA begins for if statement (line 353)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to copy(...): (line 354)
        # Processing the call keyword arguments (line 354)
        kwargs_368960 = {}
        # Getting the type of 'self' (line 354)
        self_368958 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 354, 19), 'self', False)
        # Obtaining the member 'copy' of a type (line 354)
        copy_368959 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 354, 19), self_368958, 'copy')
        # Calling copy(args, kwargs) (line 354)
        copy_call_result_368961 = invoke(stypy.reporting.localization.Localization(__file__, 354, 19), copy_368959, *[], **kwargs_368960)
        
        # Assigning a type to the variable 'stypy_return_type' (line 354)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 354, 12), 'stypy_return_type', copy_call_result_368961)
        # SSA branch for the else part of an if statement (line 353)
        module_type_store.open_ssa_branch('else')
        # Getting the type of 'self' (line 356)
        self_368962 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 356, 19), 'self')
        # Assigning a type to the variable 'stypy_return_type' (line 356)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 356, 12), 'stypy_return_type', self_368962)
        # SSA join for if statement (line 353)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'tocoo(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'tocoo' in the type store
        # Getting the type of 'stypy_return_type' (line 352)
        stypy_return_type_368963 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 352, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_368963)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'tocoo'
        return stypy_return_type_368963

    
    # Assigning a Attribute to a Attribute (line 358):
    
    # Assigning a Attribute to a Attribute (line 358):

    @norecursion
    def todia(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'False' (line 360)
        False_368964 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 360, 25), 'False')
        defaults = [False_368964]
        # Create a new context for function 'todia'
        module_type_store = module_type_store.open_function_context('todia', 360, 4, False)
        # Assigning a type to the variable 'self' (line 361)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 361, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        coo_matrix.todia.__dict__.__setitem__('stypy_localization', localization)
        coo_matrix.todia.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        coo_matrix.todia.__dict__.__setitem__('stypy_type_store', module_type_store)
        coo_matrix.todia.__dict__.__setitem__('stypy_function_name', 'coo_matrix.todia')
        coo_matrix.todia.__dict__.__setitem__('stypy_param_names_list', ['copy'])
        coo_matrix.todia.__dict__.__setitem__('stypy_varargs_param_name', None)
        coo_matrix.todia.__dict__.__setitem__('stypy_kwargs_param_name', None)
        coo_matrix.todia.__dict__.__setitem__('stypy_call_defaults', defaults)
        coo_matrix.todia.__dict__.__setitem__('stypy_call_varargs', varargs)
        coo_matrix.todia.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        coo_matrix.todia.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'coo_matrix.todia', ['copy'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'todia', localization, ['copy'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'todia(...)' code ##################

        stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 361, 8))
        
        # 'from scipy.sparse.dia import dia_matrix' statement (line 361)
        update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/sparse/')
        import_368965 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 361, 8), 'scipy.sparse.dia')

        if (type(import_368965) is not StypyTypeError):

            if (import_368965 != 'pyd_module'):
                __import__(import_368965)
                sys_modules_368966 = sys.modules[import_368965]
                import_from_module(stypy.reporting.localization.Localization(__file__, 361, 8), 'scipy.sparse.dia', sys_modules_368966.module_type_store, module_type_store, ['dia_matrix'])
                nest_module(stypy.reporting.localization.Localization(__file__, 361, 8), __file__, sys_modules_368966, sys_modules_368966.module_type_store, module_type_store)
            else:
                from scipy.sparse.dia import dia_matrix

                import_from_module(stypy.reporting.localization.Localization(__file__, 361, 8), 'scipy.sparse.dia', None, module_type_store, ['dia_matrix'], [dia_matrix])

        else:
            # Assigning a type to the variable 'scipy.sparse.dia' (line 361)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 361, 8), 'scipy.sparse.dia', import_368965)

        remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/sparse/')
        
        
        # Call to sum_duplicates(...): (line 363)
        # Processing the call keyword arguments (line 363)
        kwargs_368969 = {}
        # Getting the type of 'self' (line 363)
        self_368967 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 363, 8), 'self', False)
        # Obtaining the member 'sum_duplicates' of a type (line 363)
        sum_duplicates_368968 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 363, 8), self_368967, 'sum_duplicates')
        # Calling sum_duplicates(args, kwargs) (line 363)
        sum_duplicates_call_result_368970 = invoke(stypy.reporting.localization.Localization(__file__, 363, 8), sum_duplicates_368968, *[], **kwargs_368969)
        
        
        # Assigning a BinOp to a Name (line 364):
        
        # Assigning a BinOp to a Name (line 364):
        
        # Assigning a BinOp to a Name (line 364):
        # Getting the type of 'self' (line 364)
        self_368971 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 364, 13), 'self')
        # Obtaining the member 'col' of a type (line 364)
        col_368972 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 364, 13), self_368971, 'col')
        # Getting the type of 'self' (line 364)
        self_368973 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 364, 24), 'self')
        # Obtaining the member 'row' of a type (line 364)
        row_368974 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 364, 24), self_368973, 'row')
        # Applying the binary operator '-' (line 364)
        result_sub_368975 = python_operator(stypy.reporting.localization.Localization(__file__, 364, 13), '-', col_368972, row_368974)
        
        # Assigning a type to the variable 'ks' (line 364)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 364, 8), 'ks', result_sub_368975)
        
        # Assigning a Call to a Tuple (line 365):
        
        # Assigning a Subscript to a Name (line 365):
        
        # Assigning a Subscript to a Name (line 365):
        
        # Obtaining the type of the subscript
        int_368976 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 365, 8), 'int')
        
        # Call to unique(...): (line 365)
        # Processing the call arguments (line 365)
        # Getting the type of 'ks' (line 365)
        ks_368979 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 365, 36), 'ks', False)
        # Processing the call keyword arguments (line 365)
        # Getting the type of 'True' (line 365)
        True_368980 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 365, 55), 'True', False)
        keyword_368981 = True_368980
        kwargs_368982 = {'return_inverse': keyword_368981}
        # Getting the type of 'np' (line 365)
        np_368977 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 365, 26), 'np', False)
        # Obtaining the member 'unique' of a type (line 365)
        unique_368978 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 365, 26), np_368977, 'unique')
        # Calling unique(args, kwargs) (line 365)
        unique_call_result_368983 = invoke(stypy.reporting.localization.Localization(__file__, 365, 26), unique_368978, *[ks_368979], **kwargs_368982)
        
        # Obtaining the member '__getitem__' of a type (line 365)
        getitem___368984 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 365, 8), unique_call_result_368983, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 365)
        subscript_call_result_368985 = invoke(stypy.reporting.localization.Localization(__file__, 365, 8), getitem___368984, int_368976)
        
        # Assigning a type to the variable 'tuple_var_assignment_368036' (line 365)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 365, 8), 'tuple_var_assignment_368036', subscript_call_result_368985)
        
        # Assigning a Subscript to a Name (line 365):
        
        # Assigning a Subscript to a Name (line 365):
        
        # Obtaining the type of the subscript
        int_368986 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 365, 8), 'int')
        
        # Call to unique(...): (line 365)
        # Processing the call arguments (line 365)
        # Getting the type of 'ks' (line 365)
        ks_368989 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 365, 36), 'ks', False)
        # Processing the call keyword arguments (line 365)
        # Getting the type of 'True' (line 365)
        True_368990 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 365, 55), 'True', False)
        keyword_368991 = True_368990
        kwargs_368992 = {'return_inverse': keyword_368991}
        # Getting the type of 'np' (line 365)
        np_368987 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 365, 26), 'np', False)
        # Obtaining the member 'unique' of a type (line 365)
        unique_368988 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 365, 26), np_368987, 'unique')
        # Calling unique(args, kwargs) (line 365)
        unique_call_result_368993 = invoke(stypy.reporting.localization.Localization(__file__, 365, 26), unique_368988, *[ks_368989], **kwargs_368992)
        
        # Obtaining the member '__getitem__' of a type (line 365)
        getitem___368994 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 365, 8), unique_call_result_368993, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 365)
        subscript_call_result_368995 = invoke(stypy.reporting.localization.Localization(__file__, 365, 8), getitem___368994, int_368986)
        
        # Assigning a type to the variable 'tuple_var_assignment_368037' (line 365)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 365, 8), 'tuple_var_assignment_368037', subscript_call_result_368995)
        
        # Assigning a Name to a Name (line 365):
        
        # Assigning a Name to a Name (line 365):
        # Getting the type of 'tuple_var_assignment_368036' (line 365)
        tuple_var_assignment_368036_368996 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 365, 8), 'tuple_var_assignment_368036')
        # Assigning a type to the variable 'diags' (line 365)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 365, 8), 'diags', tuple_var_assignment_368036_368996)
        
        # Assigning a Name to a Name (line 365):
        
        # Assigning a Name to a Name (line 365):
        # Getting the type of 'tuple_var_assignment_368037' (line 365)
        tuple_var_assignment_368037_368997 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 365, 8), 'tuple_var_assignment_368037')
        # Assigning a type to the variable 'diag_idx' (line 365)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 365, 15), 'diag_idx', tuple_var_assignment_368037_368997)
        
        
        
        # Call to len(...): (line 367)
        # Processing the call arguments (line 367)
        # Getting the type of 'diags' (line 367)
        diags_368999 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 367, 15), 'diags', False)
        # Processing the call keyword arguments (line 367)
        kwargs_369000 = {}
        # Getting the type of 'len' (line 367)
        len_368998 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 367, 11), 'len', False)
        # Calling len(args, kwargs) (line 367)
        len_call_result_369001 = invoke(stypy.reporting.localization.Localization(__file__, 367, 11), len_368998, *[diags_368999], **kwargs_369000)
        
        int_369002 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 367, 24), 'int')
        # Applying the binary operator '>' (line 367)
        result_gt_369003 = python_operator(stypy.reporting.localization.Localization(__file__, 367, 11), '>', len_call_result_369001, int_369002)
        
        # Testing the type of an if condition (line 367)
        if_condition_369004 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 367, 8), result_gt_369003)
        # Assigning a type to the variable 'if_condition_369004' (line 367)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 367, 8), 'if_condition_369004', if_condition_369004)
        # SSA begins for if statement (line 367)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to warn(...): (line 369)
        # Processing the call arguments (line 369)
        str_369006 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 369, 17), 'str', 'Constructing a DIA matrix with %d diagonals is inefficient')
        
        # Call to len(...): (line 370)
        # Processing the call arguments (line 370)
        # Getting the type of 'diags' (line 370)
        diags_369008 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 370, 40), 'diags', False)
        # Processing the call keyword arguments (line 370)
        kwargs_369009 = {}
        # Getting the type of 'len' (line 370)
        len_369007 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 370, 36), 'len', False)
        # Calling len(args, kwargs) (line 370)
        len_call_result_369010 = invoke(stypy.reporting.localization.Localization(__file__, 370, 36), len_369007, *[diags_369008], **kwargs_369009)
        
        # Applying the binary operator '%' (line 369)
        result_mod_369011 = python_operator(stypy.reporting.localization.Localization(__file__, 369, 17), '%', str_369006, len_call_result_369010)
        
        # Getting the type of 'SparseEfficiencyWarning' (line 370)
        SparseEfficiencyWarning_369012 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 370, 48), 'SparseEfficiencyWarning', False)
        # Processing the call keyword arguments (line 369)
        kwargs_369013 = {}
        # Getting the type of 'warn' (line 369)
        warn_369005 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 369, 12), 'warn', False)
        # Calling warn(args, kwargs) (line 369)
        warn_call_result_369014 = invoke(stypy.reporting.localization.Localization(__file__, 369, 12), warn_369005, *[result_mod_369011, SparseEfficiencyWarning_369012], **kwargs_369013)
        
        # SSA join for if statement (line 367)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Getting the type of 'self' (line 373)
        self_369015 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 373, 11), 'self')
        # Obtaining the member 'data' of a type (line 373)
        data_369016 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 373, 11), self_369015, 'data')
        # Obtaining the member 'size' of a type (line 373)
        size_369017 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 373, 11), data_369016, 'size')
        int_369018 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 373, 29), 'int')
        # Applying the binary operator '==' (line 373)
        result_eq_369019 = python_operator(stypy.reporting.localization.Localization(__file__, 373, 11), '==', size_369017, int_369018)
        
        # Testing the type of an if condition (line 373)
        if_condition_369020 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 373, 8), result_eq_369019)
        # Assigning a type to the variable 'if_condition_369020' (line 373)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 373, 8), 'if_condition_369020', if_condition_369020)
        # SSA begins for if statement (line 373)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 374):
        
        # Assigning a Call to a Name (line 374):
        
        # Assigning a Call to a Name (line 374):
        
        # Call to zeros(...): (line 374)
        # Processing the call arguments (line 374)
        
        # Obtaining an instance of the builtin type 'tuple' (line 374)
        tuple_369023 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 374, 29), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 374)
        # Adding element type (line 374)
        int_369024 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 374, 29), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 374, 29), tuple_369023, int_369024)
        # Adding element type (line 374)
        int_369025 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 374, 32), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 374, 29), tuple_369023, int_369025)
        
        # Processing the call keyword arguments (line 374)
        # Getting the type of 'self' (line 374)
        self_369026 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 374, 42), 'self', False)
        # Obtaining the member 'dtype' of a type (line 374)
        dtype_369027 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 374, 42), self_369026, 'dtype')
        keyword_369028 = dtype_369027
        kwargs_369029 = {'dtype': keyword_369028}
        # Getting the type of 'np' (line 374)
        np_369021 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 374, 19), 'np', False)
        # Obtaining the member 'zeros' of a type (line 374)
        zeros_369022 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 374, 19), np_369021, 'zeros')
        # Calling zeros(args, kwargs) (line 374)
        zeros_call_result_369030 = invoke(stypy.reporting.localization.Localization(__file__, 374, 19), zeros_369022, *[tuple_369023], **kwargs_369029)
        
        # Assigning a type to the variable 'data' (line 374)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 374, 12), 'data', zeros_call_result_369030)
        # SSA branch for the else part of an if statement (line 373)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Call to a Name (line 376):
        
        # Assigning a Call to a Name (line 376):
        
        # Assigning a Call to a Name (line 376):
        
        # Call to zeros(...): (line 376)
        # Processing the call arguments (line 376)
        
        # Obtaining an instance of the builtin type 'tuple' (line 376)
        tuple_369033 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 376, 29), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 376)
        # Adding element type (line 376)
        
        # Call to len(...): (line 376)
        # Processing the call arguments (line 376)
        # Getting the type of 'diags' (line 376)
        diags_369035 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 376, 33), 'diags', False)
        # Processing the call keyword arguments (line 376)
        kwargs_369036 = {}
        # Getting the type of 'len' (line 376)
        len_369034 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 376, 29), 'len', False)
        # Calling len(args, kwargs) (line 376)
        len_call_result_369037 = invoke(stypy.reporting.localization.Localization(__file__, 376, 29), len_369034, *[diags_369035], **kwargs_369036)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 376, 29), tuple_369033, len_call_result_369037)
        # Adding element type (line 376)
        
        # Call to max(...): (line 376)
        # Processing the call keyword arguments (line 376)
        kwargs_369041 = {}
        # Getting the type of 'self' (line 376)
        self_369038 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 376, 41), 'self', False)
        # Obtaining the member 'col' of a type (line 376)
        col_369039 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 376, 41), self_369038, 'col')
        # Obtaining the member 'max' of a type (line 376)
        max_369040 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 376, 41), col_369039, 'max')
        # Calling max(args, kwargs) (line 376)
        max_call_result_369042 = invoke(stypy.reporting.localization.Localization(__file__, 376, 41), max_369040, *[], **kwargs_369041)
        
        int_369043 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 376, 56), 'int')
        # Applying the binary operator '+' (line 376)
        result_add_369044 = python_operator(stypy.reporting.localization.Localization(__file__, 376, 41), '+', max_call_result_369042, int_369043)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 376, 29), tuple_369033, result_add_369044)
        
        # Processing the call keyword arguments (line 376)
        # Getting the type of 'self' (line 376)
        self_369045 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 376, 66), 'self', False)
        # Obtaining the member 'dtype' of a type (line 376)
        dtype_369046 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 376, 66), self_369045, 'dtype')
        keyword_369047 = dtype_369046
        kwargs_369048 = {'dtype': keyword_369047}
        # Getting the type of 'np' (line 376)
        np_369031 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 376, 19), 'np', False)
        # Obtaining the member 'zeros' of a type (line 376)
        zeros_369032 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 376, 19), np_369031, 'zeros')
        # Calling zeros(args, kwargs) (line 376)
        zeros_call_result_369049 = invoke(stypy.reporting.localization.Localization(__file__, 376, 19), zeros_369032, *[tuple_369033], **kwargs_369048)
        
        # Assigning a type to the variable 'data' (line 376)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 376, 12), 'data', zeros_call_result_369049)
        
        # Assigning a Attribute to a Subscript (line 377):
        
        # Assigning a Attribute to a Subscript (line 377):
        
        # Assigning a Attribute to a Subscript (line 377):
        # Getting the type of 'self' (line 377)
        self_369050 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 377, 39), 'self')
        # Obtaining the member 'data' of a type (line 377)
        data_369051 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 377, 39), self_369050, 'data')
        # Getting the type of 'data' (line 377)
        data_369052 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 377, 12), 'data')
        
        # Obtaining an instance of the builtin type 'tuple' (line 377)
        tuple_369053 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 377, 17), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 377)
        # Adding element type (line 377)
        # Getting the type of 'diag_idx' (line 377)
        diag_idx_369054 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 377, 17), 'diag_idx')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 377, 17), tuple_369053, diag_idx_369054)
        # Adding element type (line 377)
        # Getting the type of 'self' (line 377)
        self_369055 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 377, 27), 'self')
        # Obtaining the member 'col' of a type (line 377)
        col_369056 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 377, 27), self_369055, 'col')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 377, 17), tuple_369053, col_369056)
        
        # Storing an element on a container (line 377)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 377, 12), data_369052, (tuple_369053, data_369051))
        # SSA join for if statement (line 373)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to dia_matrix(...): (line 379)
        # Processing the call arguments (line 379)
        
        # Obtaining an instance of the builtin type 'tuple' (line 379)
        tuple_369058 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 379, 27), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 379)
        # Adding element type (line 379)
        # Getting the type of 'data' (line 379)
        data_369059 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 379, 27), 'data', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 379, 27), tuple_369058, data_369059)
        # Adding element type (line 379)
        # Getting the type of 'diags' (line 379)
        diags_369060 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 379, 32), 'diags', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 379, 27), tuple_369058, diags_369060)
        
        # Processing the call keyword arguments (line 379)
        # Getting the type of 'self' (line 379)
        self_369061 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 379, 46), 'self', False)
        # Obtaining the member 'shape' of a type (line 379)
        shape_369062 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 379, 46), self_369061, 'shape')
        keyword_369063 = shape_369062
        kwargs_369064 = {'shape': keyword_369063}
        # Getting the type of 'dia_matrix' (line 379)
        dia_matrix_369057 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 379, 15), 'dia_matrix', False)
        # Calling dia_matrix(args, kwargs) (line 379)
        dia_matrix_call_result_369065 = invoke(stypy.reporting.localization.Localization(__file__, 379, 15), dia_matrix_369057, *[tuple_369058], **kwargs_369064)
        
        # Assigning a type to the variable 'stypy_return_type' (line 379)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 379, 8), 'stypy_return_type', dia_matrix_call_result_369065)
        
        # ################# End of 'todia(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'todia' in the type store
        # Getting the type of 'stypy_return_type' (line 360)
        stypy_return_type_369066 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 360, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_369066)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'todia'
        return stypy_return_type_369066

    
    # Assigning a Attribute to a Attribute (line 381):
    
    # Assigning a Attribute to a Attribute (line 381):

    @norecursion
    def todok(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'False' (line 383)
        False_369067 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 383, 25), 'False')
        defaults = [False_369067]
        # Create a new context for function 'todok'
        module_type_store = module_type_store.open_function_context('todok', 383, 4, False)
        # Assigning a type to the variable 'self' (line 384)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 384, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        coo_matrix.todok.__dict__.__setitem__('stypy_localization', localization)
        coo_matrix.todok.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        coo_matrix.todok.__dict__.__setitem__('stypy_type_store', module_type_store)
        coo_matrix.todok.__dict__.__setitem__('stypy_function_name', 'coo_matrix.todok')
        coo_matrix.todok.__dict__.__setitem__('stypy_param_names_list', ['copy'])
        coo_matrix.todok.__dict__.__setitem__('stypy_varargs_param_name', None)
        coo_matrix.todok.__dict__.__setitem__('stypy_kwargs_param_name', None)
        coo_matrix.todok.__dict__.__setitem__('stypy_call_defaults', defaults)
        coo_matrix.todok.__dict__.__setitem__('stypy_call_varargs', varargs)
        coo_matrix.todok.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        coo_matrix.todok.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'coo_matrix.todok', ['copy'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'todok', localization, ['copy'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'todok(...)' code ##################

        stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 384, 8))
        
        # 'from scipy.sparse.dok import dok_matrix' statement (line 384)
        update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/sparse/')
        import_369068 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 384, 8), 'scipy.sparse.dok')

        if (type(import_369068) is not StypyTypeError):

            if (import_369068 != 'pyd_module'):
                __import__(import_369068)
                sys_modules_369069 = sys.modules[import_369068]
                import_from_module(stypy.reporting.localization.Localization(__file__, 384, 8), 'scipy.sparse.dok', sys_modules_369069.module_type_store, module_type_store, ['dok_matrix'])
                nest_module(stypy.reporting.localization.Localization(__file__, 384, 8), __file__, sys_modules_369069, sys_modules_369069.module_type_store, module_type_store)
            else:
                from scipy.sparse.dok import dok_matrix

                import_from_module(stypy.reporting.localization.Localization(__file__, 384, 8), 'scipy.sparse.dok', None, module_type_store, ['dok_matrix'], [dok_matrix])

        else:
            # Assigning a type to the variable 'scipy.sparse.dok' (line 384)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 384, 8), 'scipy.sparse.dok', import_369068)

        remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/sparse/')
        
        
        # Call to sum_duplicates(...): (line 386)
        # Processing the call keyword arguments (line 386)
        kwargs_369072 = {}
        # Getting the type of 'self' (line 386)
        self_369070 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 386, 8), 'self', False)
        # Obtaining the member 'sum_duplicates' of a type (line 386)
        sum_duplicates_369071 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 386, 8), self_369070, 'sum_duplicates')
        # Calling sum_duplicates(args, kwargs) (line 386)
        sum_duplicates_call_result_369073 = invoke(stypy.reporting.localization.Localization(__file__, 386, 8), sum_duplicates_369071, *[], **kwargs_369072)
        
        
        # Assigning a Call to a Name (line 387):
        
        # Assigning a Call to a Name (line 387):
        
        # Assigning a Call to a Name (line 387):
        
        # Call to dok_matrix(...): (line 387)
        # Processing the call arguments (line 387)
        # Getting the type of 'self' (line 387)
        self_369075 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 387, 26), 'self', False)
        # Obtaining the member 'shape' of a type (line 387)
        shape_369076 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 387, 26), self_369075, 'shape')
        # Processing the call keyword arguments (line 387)
        # Getting the type of 'self' (line 387)
        self_369077 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 387, 45), 'self', False)
        # Obtaining the member 'dtype' of a type (line 387)
        dtype_369078 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 387, 45), self_369077, 'dtype')
        keyword_369079 = dtype_369078
        kwargs_369080 = {'dtype': keyword_369079}
        # Getting the type of 'dok_matrix' (line 387)
        dok_matrix_369074 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 387, 14), 'dok_matrix', False)
        # Calling dok_matrix(args, kwargs) (line 387)
        dok_matrix_call_result_369081 = invoke(stypy.reporting.localization.Localization(__file__, 387, 14), dok_matrix_369074, *[shape_369076], **kwargs_369080)
        
        # Assigning a type to the variable 'dok' (line 387)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 387, 8), 'dok', dok_matrix_call_result_369081)
        
        # Call to _update(...): (line 388)
        # Processing the call arguments (line 388)
        
        # Call to izip(...): (line 388)
        # Processing the call arguments (line 388)
        
        # Call to izip(...): (line 388)
        # Processing the call arguments (line 388)
        # Getting the type of 'self' (line 388)
        self_369086 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 388, 30), 'self', False)
        # Obtaining the member 'row' of a type (line 388)
        row_369087 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 388, 30), self_369086, 'row')
        # Getting the type of 'self' (line 388)
        self_369088 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 388, 39), 'self', False)
        # Obtaining the member 'col' of a type (line 388)
        col_369089 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 388, 39), self_369088, 'col')
        # Processing the call keyword arguments (line 388)
        kwargs_369090 = {}
        # Getting the type of 'izip' (line 388)
        izip_369085 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 388, 25), 'izip', False)
        # Calling izip(args, kwargs) (line 388)
        izip_call_result_369091 = invoke(stypy.reporting.localization.Localization(__file__, 388, 25), izip_369085, *[row_369087, col_369089], **kwargs_369090)
        
        # Getting the type of 'self' (line 388)
        self_369092 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 388, 49), 'self', False)
        # Obtaining the member 'data' of a type (line 388)
        data_369093 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 388, 49), self_369092, 'data')
        # Processing the call keyword arguments (line 388)
        kwargs_369094 = {}
        # Getting the type of 'izip' (line 388)
        izip_369084 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 388, 20), 'izip', False)
        # Calling izip(args, kwargs) (line 388)
        izip_call_result_369095 = invoke(stypy.reporting.localization.Localization(__file__, 388, 20), izip_369084, *[izip_call_result_369091, data_369093], **kwargs_369094)
        
        # Processing the call keyword arguments (line 388)
        kwargs_369096 = {}
        # Getting the type of 'dok' (line 388)
        dok_369082 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 388, 8), 'dok', False)
        # Obtaining the member '_update' of a type (line 388)
        _update_369083 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 388, 8), dok_369082, '_update')
        # Calling _update(args, kwargs) (line 388)
        _update_call_result_369097 = invoke(stypy.reporting.localization.Localization(__file__, 388, 8), _update_369083, *[izip_call_result_369095], **kwargs_369096)
        
        # Getting the type of 'dok' (line 390)
        dok_369098 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 390, 15), 'dok')
        # Assigning a type to the variable 'stypy_return_type' (line 390)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 390, 8), 'stypy_return_type', dok_369098)
        
        # ################# End of 'todok(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'todok' in the type store
        # Getting the type of 'stypy_return_type' (line 383)
        stypy_return_type_369099 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 383, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_369099)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'todok'
        return stypy_return_type_369099

    
    # Assigning a Attribute to a Attribute (line 392):
    
    # Assigning a Attribute to a Attribute (line 392):

    @norecursion
    def diagonal(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        int_369100 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 394, 25), 'int')
        defaults = [int_369100]
        # Create a new context for function 'diagonal'
        module_type_store = module_type_store.open_function_context('diagonal', 394, 4, False)
        # Assigning a type to the variable 'self' (line 395)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 395, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        coo_matrix.diagonal.__dict__.__setitem__('stypy_localization', localization)
        coo_matrix.diagonal.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        coo_matrix.diagonal.__dict__.__setitem__('stypy_type_store', module_type_store)
        coo_matrix.diagonal.__dict__.__setitem__('stypy_function_name', 'coo_matrix.diagonal')
        coo_matrix.diagonal.__dict__.__setitem__('stypy_param_names_list', ['k'])
        coo_matrix.diagonal.__dict__.__setitem__('stypy_varargs_param_name', None)
        coo_matrix.diagonal.__dict__.__setitem__('stypy_kwargs_param_name', None)
        coo_matrix.diagonal.__dict__.__setitem__('stypy_call_defaults', defaults)
        coo_matrix.diagonal.__dict__.__setitem__('stypy_call_varargs', varargs)
        coo_matrix.diagonal.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        coo_matrix.diagonal.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'coo_matrix.diagonal', ['k'], None, None, defaults, varargs, kwargs)

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

        
        # Assigning a Attribute to a Tuple (line 395):
        
        # Assigning a Subscript to a Name (line 395):
        
        # Assigning a Subscript to a Name (line 395):
        
        # Obtaining the type of the subscript
        int_369101 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 395, 8), 'int')
        # Getting the type of 'self' (line 395)
        self_369102 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 395, 21), 'self')
        # Obtaining the member 'shape' of a type (line 395)
        shape_369103 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 395, 21), self_369102, 'shape')
        # Obtaining the member '__getitem__' of a type (line 395)
        getitem___369104 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 395, 8), shape_369103, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 395)
        subscript_call_result_369105 = invoke(stypy.reporting.localization.Localization(__file__, 395, 8), getitem___369104, int_369101)
        
        # Assigning a type to the variable 'tuple_var_assignment_368038' (line 395)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 395, 8), 'tuple_var_assignment_368038', subscript_call_result_369105)
        
        # Assigning a Subscript to a Name (line 395):
        
        # Assigning a Subscript to a Name (line 395):
        
        # Obtaining the type of the subscript
        int_369106 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 395, 8), 'int')
        # Getting the type of 'self' (line 395)
        self_369107 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 395, 21), 'self')
        # Obtaining the member 'shape' of a type (line 395)
        shape_369108 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 395, 21), self_369107, 'shape')
        # Obtaining the member '__getitem__' of a type (line 395)
        getitem___369109 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 395, 8), shape_369108, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 395)
        subscript_call_result_369110 = invoke(stypy.reporting.localization.Localization(__file__, 395, 8), getitem___369109, int_369106)
        
        # Assigning a type to the variable 'tuple_var_assignment_368039' (line 395)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 395, 8), 'tuple_var_assignment_368039', subscript_call_result_369110)
        
        # Assigning a Name to a Name (line 395):
        
        # Assigning a Name to a Name (line 395):
        # Getting the type of 'tuple_var_assignment_368038' (line 395)
        tuple_var_assignment_368038_369111 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 395, 8), 'tuple_var_assignment_368038')
        # Assigning a type to the variable 'rows' (line 395)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 395, 8), 'rows', tuple_var_assignment_368038_369111)
        
        # Assigning a Name to a Name (line 395):
        
        # Assigning a Name to a Name (line 395):
        # Getting the type of 'tuple_var_assignment_368039' (line 395)
        tuple_var_assignment_368039_369112 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 395, 8), 'tuple_var_assignment_368039')
        # Assigning a type to the variable 'cols' (line 395)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 395, 14), 'cols', tuple_var_assignment_368039_369112)
        
        
        # Evaluating a boolean operation
        
        # Getting the type of 'k' (line 396)
        k_369113 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 396, 11), 'k')
        
        # Getting the type of 'rows' (line 396)
        rows_369114 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 396, 17), 'rows')
        # Applying the 'usub' unary operator (line 396)
        result___neg___369115 = python_operator(stypy.reporting.localization.Localization(__file__, 396, 16), 'usub', rows_369114)
        
        # Applying the binary operator '<=' (line 396)
        result_le_369116 = python_operator(stypy.reporting.localization.Localization(__file__, 396, 11), '<=', k_369113, result___neg___369115)
        
        
        # Getting the type of 'k' (line 396)
        k_369117 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 396, 25), 'k')
        # Getting the type of 'cols' (line 396)
        cols_369118 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 396, 30), 'cols')
        # Applying the binary operator '>=' (line 396)
        result_ge_369119 = python_operator(stypy.reporting.localization.Localization(__file__, 396, 25), '>=', k_369117, cols_369118)
        
        # Applying the binary operator 'or' (line 396)
        result_or_keyword_369120 = python_operator(stypy.reporting.localization.Localization(__file__, 396, 11), 'or', result_le_369116, result_ge_369119)
        
        # Testing the type of an if condition (line 396)
        if_condition_369121 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 396, 8), result_or_keyword_369120)
        # Assigning a type to the variable 'if_condition_369121' (line 396)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 396, 8), 'if_condition_369121', if_condition_369121)
        # SSA begins for if statement (line 396)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to ValueError(...): (line 397)
        # Processing the call arguments (line 397)
        str_369123 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 397, 29), 'str', 'k exceeds matrix dimensions')
        # Processing the call keyword arguments (line 397)
        kwargs_369124 = {}
        # Getting the type of 'ValueError' (line 397)
        ValueError_369122 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 397, 18), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 397)
        ValueError_call_result_369125 = invoke(stypy.reporting.localization.Localization(__file__, 397, 18), ValueError_369122, *[str_369123], **kwargs_369124)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 397, 12), ValueError_call_result_369125, 'raise parameter', BaseException)
        # SSA join for if statement (line 396)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Name (line 398):
        
        # Assigning a Call to a Name (line 398):
        
        # Assigning a Call to a Name (line 398):
        
        # Call to zeros(...): (line 398)
        # Processing the call arguments (line 398)
        
        # Call to min(...): (line 398)
        # Processing the call arguments (line 398)
        # Getting the type of 'rows' (line 398)
        rows_369129 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 398, 28), 'rows', False)
        
        # Call to min(...): (line 398)
        # Processing the call arguments (line 398)
        # Getting the type of 'k' (line 398)
        k_369131 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 398, 39), 'k', False)
        int_369132 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 398, 42), 'int')
        # Processing the call keyword arguments (line 398)
        kwargs_369133 = {}
        # Getting the type of 'min' (line 398)
        min_369130 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 398, 35), 'min', False)
        # Calling min(args, kwargs) (line 398)
        min_call_result_369134 = invoke(stypy.reporting.localization.Localization(__file__, 398, 35), min_369130, *[k_369131, int_369132], **kwargs_369133)
        
        # Applying the binary operator '+' (line 398)
        result_add_369135 = python_operator(stypy.reporting.localization.Localization(__file__, 398, 28), '+', rows_369129, min_call_result_369134)
        
        # Getting the type of 'cols' (line 398)
        cols_369136 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 398, 46), 'cols', False)
        
        # Call to max(...): (line 398)
        # Processing the call arguments (line 398)
        # Getting the type of 'k' (line 398)
        k_369138 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 398, 57), 'k', False)
        int_369139 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 398, 60), 'int')
        # Processing the call keyword arguments (line 398)
        kwargs_369140 = {}
        # Getting the type of 'max' (line 398)
        max_369137 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 398, 53), 'max', False)
        # Calling max(args, kwargs) (line 398)
        max_call_result_369141 = invoke(stypy.reporting.localization.Localization(__file__, 398, 53), max_369137, *[k_369138, int_369139], **kwargs_369140)
        
        # Applying the binary operator '-' (line 398)
        result_sub_369142 = python_operator(stypy.reporting.localization.Localization(__file__, 398, 46), '-', cols_369136, max_call_result_369141)
        
        # Processing the call keyword arguments (line 398)
        kwargs_369143 = {}
        # Getting the type of 'min' (line 398)
        min_369128 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 398, 24), 'min', False)
        # Calling min(args, kwargs) (line 398)
        min_call_result_369144 = invoke(stypy.reporting.localization.Localization(__file__, 398, 24), min_369128, *[result_add_369135, result_sub_369142], **kwargs_369143)
        
        # Processing the call keyword arguments (line 398)
        # Getting the type of 'self' (line 399)
        self_369145 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 399, 30), 'self', False)
        # Obtaining the member 'dtype' of a type (line 399)
        dtype_369146 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 399, 30), self_369145, 'dtype')
        keyword_369147 = dtype_369146
        kwargs_369148 = {'dtype': keyword_369147}
        # Getting the type of 'np' (line 398)
        np_369126 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 398, 15), 'np', False)
        # Obtaining the member 'zeros' of a type (line 398)
        zeros_369127 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 398, 15), np_369126, 'zeros')
        # Calling zeros(args, kwargs) (line 398)
        zeros_call_result_369149 = invoke(stypy.reporting.localization.Localization(__file__, 398, 15), zeros_369127, *[min_call_result_369144], **kwargs_369148)
        
        # Assigning a type to the variable 'diag' (line 398)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 398, 8), 'diag', zeros_call_result_369149)
        
        # Assigning a Compare to a Name (line 400):
        
        # Assigning a Compare to a Name (line 400):
        
        # Assigning a Compare to a Name (line 400):
        
        # Getting the type of 'self' (line 400)
        self_369150 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 400, 21), 'self')
        # Obtaining the member 'row' of a type (line 400)
        row_369151 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 400, 21), self_369150, 'row')
        # Getting the type of 'k' (line 400)
        k_369152 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 400, 32), 'k')
        # Applying the binary operator '+' (line 400)
        result_add_369153 = python_operator(stypy.reporting.localization.Localization(__file__, 400, 21), '+', row_369151, k_369152)
        
        # Getting the type of 'self' (line 400)
        self_369154 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 400, 38), 'self')
        # Obtaining the member 'col' of a type (line 400)
        col_369155 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 400, 38), self_369154, 'col')
        # Applying the binary operator '==' (line 400)
        result_eq_369156 = python_operator(stypy.reporting.localization.Localization(__file__, 400, 20), '==', result_add_369153, col_369155)
        
        # Assigning a type to the variable 'diag_mask' (line 400)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 400, 8), 'diag_mask', result_eq_369156)
        
        # Getting the type of 'self' (line 402)
        self_369157 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 402, 11), 'self')
        # Obtaining the member 'has_canonical_format' of a type (line 402)
        has_canonical_format_369158 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 402, 11), self_369157, 'has_canonical_format')
        # Testing the type of an if condition (line 402)
        if_condition_369159 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 402, 8), has_canonical_format_369158)
        # Assigning a type to the variable 'if_condition_369159' (line 402)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 402, 8), 'if_condition_369159', if_condition_369159)
        # SSA begins for if statement (line 402)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Subscript to a Name (line 403):
        
        # Assigning a Subscript to a Name (line 403):
        
        # Assigning a Subscript to a Name (line 403):
        
        # Obtaining the type of the subscript
        # Getting the type of 'diag_mask' (line 403)
        diag_mask_369160 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 403, 27), 'diag_mask')
        # Getting the type of 'self' (line 403)
        self_369161 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 403, 18), 'self')
        # Obtaining the member 'row' of a type (line 403)
        row_369162 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 403, 18), self_369161, 'row')
        # Obtaining the member '__getitem__' of a type (line 403)
        getitem___369163 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 403, 18), row_369162, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 403)
        subscript_call_result_369164 = invoke(stypy.reporting.localization.Localization(__file__, 403, 18), getitem___369163, diag_mask_369160)
        
        # Assigning a type to the variable 'row' (line 403)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 403, 12), 'row', subscript_call_result_369164)
        
        # Assigning a Subscript to a Name (line 404):
        
        # Assigning a Subscript to a Name (line 404):
        
        # Assigning a Subscript to a Name (line 404):
        
        # Obtaining the type of the subscript
        # Getting the type of 'diag_mask' (line 404)
        diag_mask_369165 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 404, 29), 'diag_mask')
        # Getting the type of 'self' (line 404)
        self_369166 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 404, 19), 'self')
        # Obtaining the member 'data' of a type (line 404)
        data_369167 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 404, 19), self_369166, 'data')
        # Obtaining the member '__getitem__' of a type (line 404)
        getitem___369168 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 404, 19), data_369167, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 404)
        subscript_call_result_369169 = invoke(stypy.reporting.localization.Localization(__file__, 404, 19), getitem___369168, diag_mask_369165)
        
        # Assigning a type to the variable 'data' (line 404)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 404, 12), 'data', subscript_call_result_369169)
        # SSA branch for the else part of an if statement (line 402)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Call to a Tuple (line 406):
        
        # Assigning a Subscript to a Name (line 406):
        
        # Assigning a Subscript to a Name (line 406):
        
        # Obtaining the type of the subscript
        int_369170 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 406, 12), 'int')
        
        # Call to _sum_duplicates(...): (line 406)
        # Processing the call arguments (line 406)
        
        # Obtaining the type of the subscript
        # Getting the type of 'diag_mask' (line 406)
        diag_mask_369173 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 406, 57), 'diag_mask', False)
        # Getting the type of 'self' (line 406)
        self_369174 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 406, 48), 'self', False)
        # Obtaining the member 'row' of a type (line 406)
        row_369175 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 406, 48), self_369174, 'row')
        # Obtaining the member '__getitem__' of a type (line 406)
        getitem___369176 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 406, 48), row_369175, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 406)
        subscript_call_result_369177 = invoke(stypy.reporting.localization.Localization(__file__, 406, 48), getitem___369176, diag_mask_369173)
        
        
        # Obtaining the type of the subscript
        # Getting the type of 'diag_mask' (line 407)
        diag_mask_369178 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 407, 57), 'diag_mask', False)
        # Getting the type of 'self' (line 407)
        self_369179 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 407, 48), 'self', False)
        # Obtaining the member 'col' of a type (line 407)
        col_369180 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 407, 48), self_369179, 'col')
        # Obtaining the member '__getitem__' of a type (line 407)
        getitem___369181 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 407, 48), col_369180, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 407)
        subscript_call_result_369182 = invoke(stypy.reporting.localization.Localization(__file__, 407, 48), getitem___369181, diag_mask_369178)
        
        
        # Obtaining the type of the subscript
        # Getting the type of 'diag_mask' (line 408)
        diag_mask_369183 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 408, 58), 'diag_mask', False)
        # Getting the type of 'self' (line 408)
        self_369184 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 408, 48), 'self', False)
        # Obtaining the member 'data' of a type (line 408)
        data_369185 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 408, 48), self_369184, 'data')
        # Obtaining the member '__getitem__' of a type (line 408)
        getitem___369186 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 408, 48), data_369185, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 408)
        subscript_call_result_369187 = invoke(stypy.reporting.localization.Localization(__file__, 408, 48), getitem___369186, diag_mask_369183)
        
        # Processing the call keyword arguments (line 406)
        kwargs_369188 = {}
        # Getting the type of 'self' (line 406)
        self_369171 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 406, 27), 'self', False)
        # Obtaining the member '_sum_duplicates' of a type (line 406)
        _sum_duplicates_369172 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 406, 27), self_369171, '_sum_duplicates')
        # Calling _sum_duplicates(args, kwargs) (line 406)
        _sum_duplicates_call_result_369189 = invoke(stypy.reporting.localization.Localization(__file__, 406, 27), _sum_duplicates_369172, *[subscript_call_result_369177, subscript_call_result_369182, subscript_call_result_369187], **kwargs_369188)
        
        # Obtaining the member '__getitem__' of a type (line 406)
        getitem___369190 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 406, 12), _sum_duplicates_call_result_369189, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 406)
        subscript_call_result_369191 = invoke(stypy.reporting.localization.Localization(__file__, 406, 12), getitem___369190, int_369170)
        
        # Assigning a type to the variable 'tuple_var_assignment_368040' (line 406)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 406, 12), 'tuple_var_assignment_368040', subscript_call_result_369191)
        
        # Assigning a Subscript to a Name (line 406):
        
        # Assigning a Subscript to a Name (line 406):
        
        # Obtaining the type of the subscript
        int_369192 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 406, 12), 'int')
        
        # Call to _sum_duplicates(...): (line 406)
        # Processing the call arguments (line 406)
        
        # Obtaining the type of the subscript
        # Getting the type of 'diag_mask' (line 406)
        diag_mask_369195 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 406, 57), 'diag_mask', False)
        # Getting the type of 'self' (line 406)
        self_369196 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 406, 48), 'self', False)
        # Obtaining the member 'row' of a type (line 406)
        row_369197 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 406, 48), self_369196, 'row')
        # Obtaining the member '__getitem__' of a type (line 406)
        getitem___369198 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 406, 48), row_369197, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 406)
        subscript_call_result_369199 = invoke(stypy.reporting.localization.Localization(__file__, 406, 48), getitem___369198, diag_mask_369195)
        
        
        # Obtaining the type of the subscript
        # Getting the type of 'diag_mask' (line 407)
        diag_mask_369200 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 407, 57), 'diag_mask', False)
        # Getting the type of 'self' (line 407)
        self_369201 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 407, 48), 'self', False)
        # Obtaining the member 'col' of a type (line 407)
        col_369202 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 407, 48), self_369201, 'col')
        # Obtaining the member '__getitem__' of a type (line 407)
        getitem___369203 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 407, 48), col_369202, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 407)
        subscript_call_result_369204 = invoke(stypy.reporting.localization.Localization(__file__, 407, 48), getitem___369203, diag_mask_369200)
        
        
        # Obtaining the type of the subscript
        # Getting the type of 'diag_mask' (line 408)
        diag_mask_369205 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 408, 58), 'diag_mask', False)
        # Getting the type of 'self' (line 408)
        self_369206 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 408, 48), 'self', False)
        # Obtaining the member 'data' of a type (line 408)
        data_369207 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 408, 48), self_369206, 'data')
        # Obtaining the member '__getitem__' of a type (line 408)
        getitem___369208 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 408, 48), data_369207, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 408)
        subscript_call_result_369209 = invoke(stypy.reporting.localization.Localization(__file__, 408, 48), getitem___369208, diag_mask_369205)
        
        # Processing the call keyword arguments (line 406)
        kwargs_369210 = {}
        # Getting the type of 'self' (line 406)
        self_369193 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 406, 27), 'self', False)
        # Obtaining the member '_sum_duplicates' of a type (line 406)
        _sum_duplicates_369194 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 406, 27), self_369193, '_sum_duplicates')
        # Calling _sum_duplicates(args, kwargs) (line 406)
        _sum_duplicates_call_result_369211 = invoke(stypy.reporting.localization.Localization(__file__, 406, 27), _sum_duplicates_369194, *[subscript_call_result_369199, subscript_call_result_369204, subscript_call_result_369209], **kwargs_369210)
        
        # Obtaining the member '__getitem__' of a type (line 406)
        getitem___369212 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 406, 12), _sum_duplicates_call_result_369211, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 406)
        subscript_call_result_369213 = invoke(stypy.reporting.localization.Localization(__file__, 406, 12), getitem___369212, int_369192)
        
        # Assigning a type to the variable 'tuple_var_assignment_368041' (line 406)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 406, 12), 'tuple_var_assignment_368041', subscript_call_result_369213)
        
        # Assigning a Subscript to a Name (line 406):
        
        # Assigning a Subscript to a Name (line 406):
        
        # Obtaining the type of the subscript
        int_369214 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 406, 12), 'int')
        
        # Call to _sum_duplicates(...): (line 406)
        # Processing the call arguments (line 406)
        
        # Obtaining the type of the subscript
        # Getting the type of 'diag_mask' (line 406)
        diag_mask_369217 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 406, 57), 'diag_mask', False)
        # Getting the type of 'self' (line 406)
        self_369218 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 406, 48), 'self', False)
        # Obtaining the member 'row' of a type (line 406)
        row_369219 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 406, 48), self_369218, 'row')
        # Obtaining the member '__getitem__' of a type (line 406)
        getitem___369220 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 406, 48), row_369219, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 406)
        subscript_call_result_369221 = invoke(stypy.reporting.localization.Localization(__file__, 406, 48), getitem___369220, diag_mask_369217)
        
        
        # Obtaining the type of the subscript
        # Getting the type of 'diag_mask' (line 407)
        diag_mask_369222 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 407, 57), 'diag_mask', False)
        # Getting the type of 'self' (line 407)
        self_369223 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 407, 48), 'self', False)
        # Obtaining the member 'col' of a type (line 407)
        col_369224 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 407, 48), self_369223, 'col')
        # Obtaining the member '__getitem__' of a type (line 407)
        getitem___369225 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 407, 48), col_369224, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 407)
        subscript_call_result_369226 = invoke(stypy.reporting.localization.Localization(__file__, 407, 48), getitem___369225, diag_mask_369222)
        
        
        # Obtaining the type of the subscript
        # Getting the type of 'diag_mask' (line 408)
        diag_mask_369227 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 408, 58), 'diag_mask', False)
        # Getting the type of 'self' (line 408)
        self_369228 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 408, 48), 'self', False)
        # Obtaining the member 'data' of a type (line 408)
        data_369229 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 408, 48), self_369228, 'data')
        # Obtaining the member '__getitem__' of a type (line 408)
        getitem___369230 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 408, 48), data_369229, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 408)
        subscript_call_result_369231 = invoke(stypy.reporting.localization.Localization(__file__, 408, 48), getitem___369230, diag_mask_369227)
        
        # Processing the call keyword arguments (line 406)
        kwargs_369232 = {}
        # Getting the type of 'self' (line 406)
        self_369215 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 406, 27), 'self', False)
        # Obtaining the member '_sum_duplicates' of a type (line 406)
        _sum_duplicates_369216 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 406, 27), self_369215, '_sum_duplicates')
        # Calling _sum_duplicates(args, kwargs) (line 406)
        _sum_duplicates_call_result_369233 = invoke(stypy.reporting.localization.Localization(__file__, 406, 27), _sum_duplicates_369216, *[subscript_call_result_369221, subscript_call_result_369226, subscript_call_result_369231], **kwargs_369232)
        
        # Obtaining the member '__getitem__' of a type (line 406)
        getitem___369234 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 406, 12), _sum_duplicates_call_result_369233, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 406)
        subscript_call_result_369235 = invoke(stypy.reporting.localization.Localization(__file__, 406, 12), getitem___369234, int_369214)
        
        # Assigning a type to the variable 'tuple_var_assignment_368042' (line 406)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 406, 12), 'tuple_var_assignment_368042', subscript_call_result_369235)
        
        # Assigning a Name to a Name (line 406):
        
        # Assigning a Name to a Name (line 406):
        # Getting the type of 'tuple_var_assignment_368040' (line 406)
        tuple_var_assignment_368040_369236 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 406, 12), 'tuple_var_assignment_368040')
        # Assigning a type to the variable 'row' (line 406)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 406, 12), 'row', tuple_var_assignment_368040_369236)
        
        # Assigning a Name to a Name (line 406):
        
        # Assigning a Name to a Name (line 406):
        # Getting the type of 'tuple_var_assignment_368041' (line 406)
        tuple_var_assignment_368041_369237 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 406, 12), 'tuple_var_assignment_368041')
        # Assigning a type to the variable '_' (line 406)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 406, 17), '_', tuple_var_assignment_368041_369237)
        
        # Assigning a Name to a Name (line 406):
        
        # Assigning a Name to a Name (line 406):
        # Getting the type of 'tuple_var_assignment_368042' (line 406)
        tuple_var_assignment_368042_369238 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 406, 12), 'tuple_var_assignment_368042')
        # Assigning a type to the variable 'data' (line 406)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 406, 20), 'data', tuple_var_assignment_368042_369238)
        # SSA join for if statement (line 402)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Name to a Subscript (line 409):
        
        # Assigning a Name to a Subscript (line 409):
        
        # Assigning a Name to a Subscript (line 409):
        # Getting the type of 'data' (line 409)
        data_369239 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 409, 32), 'data')
        # Getting the type of 'diag' (line 409)
        diag_369240 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 409, 8), 'diag')
        # Getting the type of 'row' (line 409)
        row_369241 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 409, 13), 'row')
        
        # Call to min(...): (line 409)
        # Processing the call arguments (line 409)
        # Getting the type of 'k' (line 409)
        k_369243 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 409, 23), 'k', False)
        int_369244 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 409, 26), 'int')
        # Processing the call keyword arguments (line 409)
        kwargs_369245 = {}
        # Getting the type of 'min' (line 409)
        min_369242 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 409, 19), 'min', False)
        # Calling min(args, kwargs) (line 409)
        min_call_result_369246 = invoke(stypy.reporting.localization.Localization(__file__, 409, 19), min_369242, *[k_369243, int_369244], **kwargs_369245)
        
        # Applying the binary operator '+' (line 409)
        result_add_369247 = python_operator(stypy.reporting.localization.Localization(__file__, 409, 13), '+', row_369241, min_call_result_369246)
        
        # Storing an element on a container (line 409)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 409, 8), diag_369240, (result_add_369247, data_369239))
        # Getting the type of 'diag' (line 411)
        diag_369248 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 411, 15), 'diag')
        # Assigning a type to the variable 'stypy_return_type' (line 411)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 411, 8), 'stypy_return_type', diag_369248)
        
        # ################# End of 'diagonal(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'diagonal' in the type store
        # Getting the type of 'stypy_return_type' (line 394)
        stypy_return_type_369249 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 394, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_369249)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'diagonal'
        return stypy_return_type_369249

    
    # Assigning a Attribute to a Attribute (line 413):
    
    # Assigning a Attribute to a Attribute (line 413):

    @norecursion
    def _setdiag(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_setdiag'
        module_type_store = module_type_store.open_function_context('_setdiag', 415, 4, False)
        # Assigning a type to the variable 'self' (line 416)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 416, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        coo_matrix._setdiag.__dict__.__setitem__('stypy_localization', localization)
        coo_matrix._setdiag.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        coo_matrix._setdiag.__dict__.__setitem__('stypy_type_store', module_type_store)
        coo_matrix._setdiag.__dict__.__setitem__('stypy_function_name', 'coo_matrix._setdiag')
        coo_matrix._setdiag.__dict__.__setitem__('stypy_param_names_list', ['values', 'k'])
        coo_matrix._setdiag.__dict__.__setitem__('stypy_varargs_param_name', None)
        coo_matrix._setdiag.__dict__.__setitem__('stypy_kwargs_param_name', None)
        coo_matrix._setdiag.__dict__.__setitem__('stypy_call_defaults', defaults)
        coo_matrix._setdiag.__dict__.__setitem__('stypy_call_varargs', varargs)
        coo_matrix._setdiag.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        coo_matrix._setdiag.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'coo_matrix._setdiag', ['values', 'k'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_setdiag', localization, ['values', 'k'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_setdiag(...)' code ##################

        
        # Assigning a Attribute to a Tuple (line 416):
        
        # Assigning a Subscript to a Name (line 416):
        
        # Assigning a Subscript to a Name (line 416):
        
        # Obtaining the type of the subscript
        int_369250 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 416, 8), 'int')
        # Getting the type of 'self' (line 416)
        self_369251 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 416, 15), 'self')
        # Obtaining the member 'shape' of a type (line 416)
        shape_369252 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 416, 15), self_369251, 'shape')
        # Obtaining the member '__getitem__' of a type (line 416)
        getitem___369253 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 416, 8), shape_369252, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 416)
        subscript_call_result_369254 = invoke(stypy.reporting.localization.Localization(__file__, 416, 8), getitem___369253, int_369250)
        
        # Assigning a type to the variable 'tuple_var_assignment_368043' (line 416)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 416, 8), 'tuple_var_assignment_368043', subscript_call_result_369254)
        
        # Assigning a Subscript to a Name (line 416):
        
        # Assigning a Subscript to a Name (line 416):
        
        # Obtaining the type of the subscript
        int_369255 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 416, 8), 'int')
        # Getting the type of 'self' (line 416)
        self_369256 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 416, 15), 'self')
        # Obtaining the member 'shape' of a type (line 416)
        shape_369257 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 416, 15), self_369256, 'shape')
        # Obtaining the member '__getitem__' of a type (line 416)
        getitem___369258 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 416, 8), shape_369257, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 416)
        subscript_call_result_369259 = invoke(stypy.reporting.localization.Localization(__file__, 416, 8), getitem___369258, int_369255)
        
        # Assigning a type to the variable 'tuple_var_assignment_368044' (line 416)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 416, 8), 'tuple_var_assignment_368044', subscript_call_result_369259)
        
        # Assigning a Name to a Name (line 416):
        
        # Assigning a Name to a Name (line 416):
        # Getting the type of 'tuple_var_assignment_368043' (line 416)
        tuple_var_assignment_368043_369260 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 416, 8), 'tuple_var_assignment_368043')
        # Assigning a type to the variable 'M' (line 416)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 416, 8), 'M', tuple_var_assignment_368043_369260)
        
        # Assigning a Name to a Name (line 416):
        
        # Assigning a Name to a Name (line 416):
        # Getting the type of 'tuple_var_assignment_368044' (line 416)
        tuple_var_assignment_368044_369261 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 416, 8), 'tuple_var_assignment_368044')
        # Assigning a type to the variable 'N' (line 416)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 416, 11), 'N', tuple_var_assignment_368044_369261)
        
        
        # Evaluating a boolean operation
        # Getting the type of 'values' (line 417)
        values_369262 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 417, 11), 'values')
        # Obtaining the member 'ndim' of a type (line 417)
        ndim_369263 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 417, 11), values_369262, 'ndim')
        
        
        # Call to len(...): (line 417)
        # Processing the call arguments (line 417)
        # Getting the type of 'values' (line 417)
        values_369265 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 417, 35), 'values', False)
        # Processing the call keyword arguments (line 417)
        kwargs_369266 = {}
        # Getting the type of 'len' (line 417)
        len_369264 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 417, 31), 'len', False)
        # Calling len(args, kwargs) (line 417)
        len_call_result_369267 = invoke(stypy.reporting.localization.Localization(__file__, 417, 31), len_369264, *[values_369265], **kwargs_369266)
        
        # Applying the 'not' unary operator (line 417)
        result_not__369268 = python_operator(stypy.reporting.localization.Localization(__file__, 417, 27), 'not', len_call_result_369267)
        
        # Applying the binary operator 'and' (line 417)
        result_and_keyword_369269 = python_operator(stypy.reporting.localization.Localization(__file__, 417, 11), 'and', ndim_369263, result_not__369268)
        
        # Testing the type of an if condition (line 417)
        if_condition_369270 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 417, 8), result_and_keyword_369269)
        # Assigning a type to the variable 'if_condition_369270' (line 417)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 417, 8), 'if_condition_369270', if_condition_369270)
        # SSA begins for if statement (line 417)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # Assigning a type to the variable 'stypy_return_type' (line 418)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 418, 12), 'stypy_return_type', types.NoneType)
        # SSA join for if statement (line 417)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Attribute to a Name (line 419):
        
        # Assigning a Attribute to a Name (line 419):
        
        # Assigning a Attribute to a Name (line 419):
        # Getting the type of 'self' (line 419)
        self_369271 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 419, 20), 'self')
        # Obtaining the member 'row' of a type (line 419)
        row_369272 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 419, 20), self_369271, 'row')
        # Obtaining the member 'dtype' of a type (line 419)
        dtype_369273 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 419, 20), row_369272, 'dtype')
        # Assigning a type to the variable 'idx_dtype' (line 419)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 419, 8), 'idx_dtype', dtype_369273)
        
        # Assigning a Compare to a Name (line 422):
        
        # Assigning a Compare to a Name (line 422):
        
        # Assigning a Compare to a Name (line 422):
        
        # Getting the type of 'self' (line 422)
        self_369274 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 422, 20), 'self')
        # Obtaining the member 'col' of a type (line 422)
        col_369275 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 422, 20), self_369274, 'col')
        # Getting the type of 'self' (line 422)
        self_369276 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 422, 31), 'self')
        # Obtaining the member 'row' of a type (line 422)
        row_369277 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 422, 31), self_369276, 'row')
        # Applying the binary operator '-' (line 422)
        result_sub_369278 = python_operator(stypy.reporting.localization.Localization(__file__, 422, 20), '-', col_369275, row_369277)
        
        # Getting the type of 'k' (line 422)
        k_369279 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 422, 43), 'k')
        # Applying the binary operator '!=' (line 422)
        result_ne_369280 = python_operator(stypy.reporting.localization.Localization(__file__, 422, 20), '!=', result_sub_369278, k_369279)
        
        # Assigning a type to the variable 'full_keep' (line 422)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 422, 8), 'full_keep', result_ne_369280)
        
        
        # Getting the type of 'k' (line 423)
        k_369281 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 423, 11), 'k')
        int_369282 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 423, 15), 'int')
        # Applying the binary operator '<' (line 423)
        result_lt_369283 = python_operator(stypy.reporting.localization.Localization(__file__, 423, 11), '<', k_369281, int_369282)
        
        # Testing the type of an if condition (line 423)
        if_condition_369284 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 423, 8), result_lt_369283)
        # Assigning a type to the variable 'if_condition_369284' (line 423)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 423, 8), 'if_condition_369284', if_condition_369284)
        # SSA begins for if statement (line 423)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 424):
        
        # Assigning a Call to a Name (line 424):
        
        # Assigning a Call to a Name (line 424):
        
        # Call to min(...): (line 424)
        # Processing the call arguments (line 424)
        # Getting the type of 'M' (line 424)
        M_369286 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 424, 28), 'M', False)
        # Getting the type of 'k' (line 424)
        k_369287 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 424, 30), 'k', False)
        # Applying the binary operator '+' (line 424)
        result_add_369288 = python_operator(stypy.reporting.localization.Localization(__file__, 424, 28), '+', M_369286, k_369287)
        
        # Getting the type of 'N' (line 424)
        N_369289 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 424, 33), 'N', False)
        # Processing the call keyword arguments (line 424)
        kwargs_369290 = {}
        # Getting the type of 'min' (line 424)
        min_369285 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 424, 24), 'min', False)
        # Calling min(args, kwargs) (line 424)
        min_call_result_369291 = invoke(stypy.reporting.localization.Localization(__file__, 424, 24), min_369285, *[result_add_369288, N_369289], **kwargs_369290)
        
        # Assigning a type to the variable 'max_index' (line 424)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 424, 12), 'max_index', min_call_result_369291)
        
        # Getting the type of 'values' (line 425)
        values_369292 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 425, 15), 'values')
        # Obtaining the member 'ndim' of a type (line 425)
        ndim_369293 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 425, 15), values_369292, 'ndim')
        # Testing the type of an if condition (line 425)
        if_condition_369294 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 425, 12), ndim_369293)
        # Assigning a type to the variable 'if_condition_369294' (line 425)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 425, 12), 'if_condition_369294', if_condition_369294)
        # SSA begins for if statement (line 425)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 426):
        
        # Assigning a Call to a Name (line 426):
        
        # Assigning a Call to a Name (line 426):
        
        # Call to min(...): (line 426)
        # Processing the call arguments (line 426)
        # Getting the type of 'max_index' (line 426)
        max_index_369296 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 426, 32), 'max_index', False)
        
        # Call to len(...): (line 426)
        # Processing the call arguments (line 426)
        # Getting the type of 'values' (line 426)
        values_369298 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 426, 47), 'values', False)
        # Processing the call keyword arguments (line 426)
        kwargs_369299 = {}
        # Getting the type of 'len' (line 426)
        len_369297 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 426, 43), 'len', False)
        # Calling len(args, kwargs) (line 426)
        len_call_result_369300 = invoke(stypy.reporting.localization.Localization(__file__, 426, 43), len_369297, *[values_369298], **kwargs_369299)
        
        # Processing the call keyword arguments (line 426)
        kwargs_369301 = {}
        # Getting the type of 'min' (line 426)
        min_369295 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 426, 28), 'min', False)
        # Calling min(args, kwargs) (line 426)
        min_call_result_369302 = invoke(stypy.reporting.localization.Localization(__file__, 426, 28), min_369295, *[max_index_369296, len_call_result_369300], **kwargs_369301)
        
        # Assigning a type to the variable 'max_index' (line 426)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 426, 16), 'max_index', min_call_result_369302)
        # SSA join for if statement (line 425)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Name (line 427):
        
        # Assigning a Call to a Name (line 427):
        
        # Assigning a Call to a Name (line 427):
        
        # Call to logical_or(...): (line 427)
        # Processing the call arguments (line 427)
        # Getting the type of 'full_keep' (line 427)
        full_keep_369305 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 427, 33), 'full_keep', False)
        
        # Getting the type of 'self' (line 427)
        self_369306 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 427, 44), 'self', False)
        # Obtaining the member 'col' of a type (line 427)
        col_369307 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 427, 44), self_369306, 'col')
        # Getting the type of 'max_index' (line 427)
        max_index_369308 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 427, 56), 'max_index', False)
        # Applying the binary operator '>=' (line 427)
        result_ge_369309 = python_operator(stypy.reporting.localization.Localization(__file__, 427, 44), '>=', col_369307, max_index_369308)
        
        # Processing the call keyword arguments (line 427)
        kwargs_369310 = {}
        # Getting the type of 'np' (line 427)
        np_369303 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 427, 19), 'np', False)
        # Obtaining the member 'logical_or' of a type (line 427)
        logical_or_369304 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 427, 19), np_369303, 'logical_or')
        # Calling logical_or(args, kwargs) (line 427)
        logical_or_call_result_369311 = invoke(stypy.reporting.localization.Localization(__file__, 427, 19), logical_or_369304, *[full_keep_369305, result_ge_369309], **kwargs_369310)
        
        # Assigning a type to the variable 'keep' (line 427)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 427, 12), 'keep', logical_or_call_result_369311)
        
        # Assigning a Call to a Name (line 428):
        
        # Assigning a Call to a Name (line 428):
        
        # Assigning a Call to a Name (line 428):
        
        # Call to arange(...): (line 428)
        # Processing the call arguments (line 428)
        
        # Getting the type of 'k' (line 428)
        k_369314 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 428, 33), 'k', False)
        # Applying the 'usub' unary operator (line 428)
        result___neg___369315 = python_operator(stypy.reporting.localization.Localization(__file__, 428, 32), 'usub', k_369314)
        
        
        # Getting the type of 'k' (line 428)
        k_369316 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 428, 37), 'k', False)
        # Applying the 'usub' unary operator (line 428)
        result___neg___369317 = python_operator(stypy.reporting.localization.Localization(__file__, 428, 36), 'usub', k_369316)
        
        # Getting the type of 'max_index' (line 428)
        max_index_369318 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 428, 41), 'max_index', False)
        # Applying the binary operator '+' (line 428)
        result_add_369319 = python_operator(stypy.reporting.localization.Localization(__file__, 428, 36), '+', result___neg___369317, max_index_369318)
        
        # Processing the call keyword arguments (line 428)
        # Getting the type of 'idx_dtype' (line 428)
        idx_dtype_369320 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 428, 58), 'idx_dtype', False)
        keyword_369321 = idx_dtype_369320
        kwargs_369322 = {'dtype': keyword_369321}
        # Getting the type of 'np' (line 428)
        np_369312 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 428, 22), 'np', False)
        # Obtaining the member 'arange' of a type (line 428)
        arange_369313 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 428, 22), np_369312, 'arange')
        # Calling arange(args, kwargs) (line 428)
        arange_call_result_369323 = invoke(stypy.reporting.localization.Localization(__file__, 428, 22), arange_369313, *[result___neg___369315, result_add_369319], **kwargs_369322)
        
        # Assigning a type to the variable 'new_row' (line 428)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 428, 12), 'new_row', arange_call_result_369323)
        
        # Assigning a Call to a Name (line 429):
        
        # Assigning a Call to a Name (line 429):
        
        # Assigning a Call to a Name (line 429):
        
        # Call to arange(...): (line 429)
        # Processing the call arguments (line 429)
        # Getting the type of 'max_index' (line 429)
        max_index_369326 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 429, 32), 'max_index', False)
        # Processing the call keyword arguments (line 429)
        # Getting the type of 'idx_dtype' (line 429)
        idx_dtype_369327 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 429, 49), 'idx_dtype', False)
        keyword_369328 = idx_dtype_369327
        kwargs_369329 = {'dtype': keyword_369328}
        # Getting the type of 'np' (line 429)
        np_369324 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 429, 22), 'np', False)
        # Obtaining the member 'arange' of a type (line 429)
        arange_369325 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 429, 22), np_369324, 'arange')
        # Calling arange(args, kwargs) (line 429)
        arange_call_result_369330 = invoke(stypy.reporting.localization.Localization(__file__, 429, 22), arange_369325, *[max_index_369326], **kwargs_369329)
        
        # Assigning a type to the variable 'new_col' (line 429)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 429, 12), 'new_col', arange_call_result_369330)
        # SSA branch for the else part of an if statement (line 423)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Call to a Name (line 431):
        
        # Assigning a Call to a Name (line 431):
        
        # Assigning a Call to a Name (line 431):
        
        # Call to min(...): (line 431)
        # Processing the call arguments (line 431)
        # Getting the type of 'M' (line 431)
        M_369332 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 431, 28), 'M', False)
        # Getting the type of 'N' (line 431)
        N_369333 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 431, 31), 'N', False)
        # Getting the type of 'k' (line 431)
        k_369334 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 431, 33), 'k', False)
        # Applying the binary operator '-' (line 431)
        result_sub_369335 = python_operator(stypy.reporting.localization.Localization(__file__, 431, 31), '-', N_369333, k_369334)
        
        # Processing the call keyword arguments (line 431)
        kwargs_369336 = {}
        # Getting the type of 'min' (line 431)
        min_369331 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 431, 24), 'min', False)
        # Calling min(args, kwargs) (line 431)
        min_call_result_369337 = invoke(stypy.reporting.localization.Localization(__file__, 431, 24), min_369331, *[M_369332, result_sub_369335], **kwargs_369336)
        
        # Assigning a type to the variable 'max_index' (line 431)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 431, 12), 'max_index', min_call_result_369337)
        
        # Getting the type of 'values' (line 432)
        values_369338 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 432, 15), 'values')
        # Obtaining the member 'ndim' of a type (line 432)
        ndim_369339 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 432, 15), values_369338, 'ndim')
        # Testing the type of an if condition (line 432)
        if_condition_369340 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 432, 12), ndim_369339)
        # Assigning a type to the variable 'if_condition_369340' (line 432)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 432, 12), 'if_condition_369340', if_condition_369340)
        # SSA begins for if statement (line 432)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 433):
        
        # Assigning a Call to a Name (line 433):
        
        # Assigning a Call to a Name (line 433):
        
        # Call to min(...): (line 433)
        # Processing the call arguments (line 433)
        # Getting the type of 'max_index' (line 433)
        max_index_369342 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 433, 32), 'max_index', False)
        
        # Call to len(...): (line 433)
        # Processing the call arguments (line 433)
        # Getting the type of 'values' (line 433)
        values_369344 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 433, 47), 'values', False)
        # Processing the call keyword arguments (line 433)
        kwargs_369345 = {}
        # Getting the type of 'len' (line 433)
        len_369343 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 433, 43), 'len', False)
        # Calling len(args, kwargs) (line 433)
        len_call_result_369346 = invoke(stypy.reporting.localization.Localization(__file__, 433, 43), len_369343, *[values_369344], **kwargs_369345)
        
        # Processing the call keyword arguments (line 433)
        kwargs_369347 = {}
        # Getting the type of 'min' (line 433)
        min_369341 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 433, 28), 'min', False)
        # Calling min(args, kwargs) (line 433)
        min_call_result_369348 = invoke(stypy.reporting.localization.Localization(__file__, 433, 28), min_369341, *[max_index_369342, len_call_result_369346], **kwargs_369347)
        
        # Assigning a type to the variable 'max_index' (line 433)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 433, 16), 'max_index', min_call_result_369348)
        # SSA join for if statement (line 432)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Name (line 434):
        
        # Assigning a Call to a Name (line 434):
        
        # Assigning a Call to a Name (line 434):
        
        # Call to logical_or(...): (line 434)
        # Processing the call arguments (line 434)
        # Getting the type of 'full_keep' (line 434)
        full_keep_369351 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 434, 33), 'full_keep', False)
        
        # Getting the type of 'self' (line 434)
        self_369352 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 434, 44), 'self', False)
        # Obtaining the member 'row' of a type (line 434)
        row_369353 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 434, 44), self_369352, 'row')
        # Getting the type of 'max_index' (line 434)
        max_index_369354 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 434, 56), 'max_index', False)
        # Applying the binary operator '>=' (line 434)
        result_ge_369355 = python_operator(stypy.reporting.localization.Localization(__file__, 434, 44), '>=', row_369353, max_index_369354)
        
        # Processing the call keyword arguments (line 434)
        kwargs_369356 = {}
        # Getting the type of 'np' (line 434)
        np_369349 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 434, 19), 'np', False)
        # Obtaining the member 'logical_or' of a type (line 434)
        logical_or_369350 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 434, 19), np_369349, 'logical_or')
        # Calling logical_or(args, kwargs) (line 434)
        logical_or_call_result_369357 = invoke(stypy.reporting.localization.Localization(__file__, 434, 19), logical_or_369350, *[full_keep_369351, result_ge_369355], **kwargs_369356)
        
        # Assigning a type to the variable 'keep' (line 434)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 434, 12), 'keep', logical_or_call_result_369357)
        
        # Assigning a Call to a Name (line 435):
        
        # Assigning a Call to a Name (line 435):
        
        # Assigning a Call to a Name (line 435):
        
        # Call to arange(...): (line 435)
        # Processing the call arguments (line 435)
        # Getting the type of 'max_index' (line 435)
        max_index_369360 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 435, 32), 'max_index', False)
        # Processing the call keyword arguments (line 435)
        # Getting the type of 'idx_dtype' (line 435)
        idx_dtype_369361 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 435, 49), 'idx_dtype', False)
        keyword_369362 = idx_dtype_369361
        kwargs_369363 = {'dtype': keyword_369362}
        # Getting the type of 'np' (line 435)
        np_369358 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 435, 22), 'np', False)
        # Obtaining the member 'arange' of a type (line 435)
        arange_369359 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 435, 22), np_369358, 'arange')
        # Calling arange(args, kwargs) (line 435)
        arange_call_result_369364 = invoke(stypy.reporting.localization.Localization(__file__, 435, 22), arange_369359, *[max_index_369360], **kwargs_369363)
        
        # Assigning a type to the variable 'new_row' (line 435)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 435, 12), 'new_row', arange_call_result_369364)
        
        # Assigning a Call to a Name (line 436):
        
        # Assigning a Call to a Name (line 436):
        
        # Assigning a Call to a Name (line 436):
        
        # Call to arange(...): (line 436)
        # Processing the call arguments (line 436)
        # Getting the type of 'k' (line 436)
        k_369367 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 436, 32), 'k', False)
        # Getting the type of 'k' (line 436)
        k_369368 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 436, 35), 'k', False)
        # Getting the type of 'max_index' (line 436)
        max_index_369369 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 436, 39), 'max_index', False)
        # Applying the binary operator '+' (line 436)
        result_add_369370 = python_operator(stypy.reporting.localization.Localization(__file__, 436, 35), '+', k_369368, max_index_369369)
        
        # Processing the call keyword arguments (line 436)
        # Getting the type of 'idx_dtype' (line 436)
        idx_dtype_369371 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 436, 56), 'idx_dtype', False)
        keyword_369372 = idx_dtype_369371
        kwargs_369373 = {'dtype': keyword_369372}
        # Getting the type of 'np' (line 436)
        np_369365 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 436, 22), 'np', False)
        # Obtaining the member 'arange' of a type (line 436)
        arange_369366 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 436, 22), np_369365, 'arange')
        # Calling arange(args, kwargs) (line 436)
        arange_call_result_369374 = invoke(stypy.reporting.localization.Localization(__file__, 436, 22), arange_369366, *[k_369367, result_add_369370], **kwargs_369373)
        
        # Assigning a type to the variable 'new_col' (line 436)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 436, 12), 'new_col', arange_call_result_369374)
        # SSA join for if statement (line 423)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Getting the type of 'values' (line 439)
        values_369375 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 439, 11), 'values')
        # Obtaining the member 'ndim' of a type (line 439)
        ndim_369376 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 439, 11), values_369375, 'ndim')
        # Testing the type of an if condition (line 439)
        if_condition_369377 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 439, 8), ndim_369376)
        # Assigning a type to the variable 'if_condition_369377' (line 439)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 439, 8), 'if_condition_369377', if_condition_369377)
        # SSA begins for if statement (line 439)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Subscript to a Name (line 440):
        
        # Assigning a Subscript to a Name (line 440):
        
        # Assigning a Subscript to a Name (line 440):
        
        # Obtaining the type of the subscript
        # Getting the type of 'max_index' (line 440)
        max_index_369378 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 440, 31), 'max_index')
        slice_369379 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 440, 23), None, max_index_369378, None)
        # Getting the type of 'values' (line 440)
        values_369380 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 440, 23), 'values')
        # Obtaining the member '__getitem__' of a type (line 440)
        getitem___369381 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 440, 23), values_369380, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 440)
        subscript_call_result_369382 = invoke(stypy.reporting.localization.Localization(__file__, 440, 23), getitem___369381, slice_369379)
        
        # Assigning a type to the variable 'new_data' (line 440)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 440, 12), 'new_data', subscript_call_result_369382)
        # SSA branch for the else part of an if statement (line 439)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Call to a Name (line 442):
        
        # Assigning a Call to a Name (line 442):
        
        # Assigning a Call to a Name (line 442):
        
        # Call to empty(...): (line 442)
        # Processing the call arguments (line 442)
        # Getting the type of 'max_index' (line 442)
        max_index_369385 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 442, 32), 'max_index', False)
        # Processing the call keyword arguments (line 442)
        # Getting the type of 'self' (line 442)
        self_369386 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 442, 49), 'self', False)
        # Obtaining the member 'dtype' of a type (line 442)
        dtype_369387 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 442, 49), self_369386, 'dtype')
        keyword_369388 = dtype_369387
        kwargs_369389 = {'dtype': keyword_369388}
        # Getting the type of 'np' (line 442)
        np_369383 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 442, 23), 'np', False)
        # Obtaining the member 'empty' of a type (line 442)
        empty_369384 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 442, 23), np_369383, 'empty')
        # Calling empty(args, kwargs) (line 442)
        empty_call_result_369390 = invoke(stypy.reporting.localization.Localization(__file__, 442, 23), empty_369384, *[max_index_369385], **kwargs_369389)
        
        # Assigning a type to the variable 'new_data' (line 442)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 442, 12), 'new_data', empty_call_result_369390)
        
        # Assigning a Name to a Subscript (line 443):
        
        # Assigning a Name to a Subscript (line 443):
        
        # Assigning a Name to a Subscript (line 443):
        # Getting the type of 'values' (line 443)
        values_369391 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 443, 26), 'values')
        # Getting the type of 'new_data' (line 443)
        new_data_369392 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 443, 12), 'new_data')
        slice_369393 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 443, 12), None, None, None)
        # Storing an element on a container (line 443)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 443, 12), new_data_369392, (slice_369393, values_369391))
        # SSA join for if statement (line 439)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Attribute (line 446):
        
        # Assigning a Call to a Attribute (line 446):
        
        # Assigning a Call to a Attribute (line 446):
        
        # Call to concatenate(...): (line 446)
        # Processing the call arguments (line 446)
        
        # Obtaining an instance of the builtin type 'tuple' (line 446)
        tuple_369396 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 446, 35), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 446)
        # Adding element type (line 446)
        
        # Obtaining the type of the subscript
        # Getting the type of 'keep' (line 446)
        keep_369397 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 446, 44), 'keep', False)
        # Getting the type of 'self' (line 446)
        self_369398 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 446, 35), 'self', False)
        # Obtaining the member 'row' of a type (line 446)
        row_369399 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 446, 35), self_369398, 'row')
        # Obtaining the member '__getitem__' of a type (line 446)
        getitem___369400 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 446, 35), row_369399, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 446)
        subscript_call_result_369401 = invoke(stypy.reporting.localization.Localization(__file__, 446, 35), getitem___369400, keep_369397)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 446, 35), tuple_369396, subscript_call_result_369401)
        # Adding element type (line 446)
        # Getting the type of 'new_row' (line 446)
        new_row_369402 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 446, 51), 'new_row', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 446, 35), tuple_369396, new_row_369402)
        
        # Processing the call keyword arguments (line 446)
        kwargs_369403 = {}
        # Getting the type of 'np' (line 446)
        np_369394 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 446, 19), 'np', False)
        # Obtaining the member 'concatenate' of a type (line 446)
        concatenate_369395 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 446, 19), np_369394, 'concatenate')
        # Calling concatenate(args, kwargs) (line 446)
        concatenate_call_result_369404 = invoke(stypy.reporting.localization.Localization(__file__, 446, 19), concatenate_369395, *[tuple_369396], **kwargs_369403)
        
        # Getting the type of 'self' (line 446)
        self_369405 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 446, 8), 'self')
        # Setting the type of the member 'row' of a type (line 446)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 446, 8), self_369405, 'row', concatenate_call_result_369404)
        
        # Assigning a Call to a Attribute (line 447):
        
        # Assigning a Call to a Attribute (line 447):
        
        # Assigning a Call to a Attribute (line 447):
        
        # Call to concatenate(...): (line 447)
        # Processing the call arguments (line 447)
        
        # Obtaining an instance of the builtin type 'tuple' (line 447)
        tuple_369408 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 447, 35), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 447)
        # Adding element type (line 447)
        
        # Obtaining the type of the subscript
        # Getting the type of 'keep' (line 447)
        keep_369409 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 447, 44), 'keep', False)
        # Getting the type of 'self' (line 447)
        self_369410 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 447, 35), 'self', False)
        # Obtaining the member 'col' of a type (line 447)
        col_369411 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 447, 35), self_369410, 'col')
        # Obtaining the member '__getitem__' of a type (line 447)
        getitem___369412 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 447, 35), col_369411, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 447)
        subscript_call_result_369413 = invoke(stypy.reporting.localization.Localization(__file__, 447, 35), getitem___369412, keep_369409)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 447, 35), tuple_369408, subscript_call_result_369413)
        # Adding element type (line 447)
        # Getting the type of 'new_col' (line 447)
        new_col_369414 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 447, 51), 'new_col', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 447, 35), tuple_369408, new_col_369414)
        
        # Processing the call keyword arguments (line 447)
        kwargs_369415 = {}
        # Getting the type of 'np' (line 447)
        np_369406 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 447, 19), 'np', False)
        # Obtaining the member 'concatenate' of a type (line 447)
        concatenate_369407 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 447, 19), np_369406, 'concatenate')
        # Calling concatenate(args, kwargs) (line 447)
        concatenate_call_result_369416 = invoke(stypy.reporting.localization.Localization(__file__, 447, 19), concatenate_369407, *[tuple_369408], **kwargs_369415)
        
        # Getting the type of 'self' (line 447)
        self_369417 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 447, 8), 'self')
        # Setting the type of the member 'col' of a type (line 447)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 447, 8), self_369417, 'col', concatenate_call_result_369416)
        
        # Assigning a Call to a Attribute (line 448):
        
        # Assigning a Call to a Attribute (line 448):
        
        # Assigning a Call to a Attribute (line 448):
        
        # Call to concatenate(...): (line 448)
        # Processing the call arguments (line 448)
        
        # Obtaining an instance of the builtin type 'tuple' (line 448)
        tuple_369420 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 448, 36), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 448)
        # Adding element type (line 448)
        
        # Obtaining the type of the subscript
        # Getting the type of 'keep' (line 448)
        keep_369421 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 448, 46), 'keep', False)
        # Getting the type of 'self' (line 448)
        self_369422 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 448, 36), 'self', False)
        # Obtaining the member 'data' of a type (line 448)
        data_369423 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 448, 36), self_369422, 'data')
        # Obtaining the member '__getitem__' of a type (line 448)
        getitem___369424 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 448, 36), data_369423, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 448)
        subscript_call_result_369425 = invoke(stypy.reporting.localization.Localization(__file__, 448, 36), getitem___369424, keep_369421)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 448, 36), tuple_369420, subscript_call_result_369425)
        # Adding element type (line 448)
        # Getting the type of 'new_data' (line 448)
        new_data_369426 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 448, 53), 'new_data', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 448, 36), tuple_369420, new_data_369426)
        
        # Processing the call keyword arguments (line 448)
        kwargs_369427 = {}
        # Getting the type of 'np' (line 448)
        np_369418 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 448, 20), 'np', False)
        # Obtaining the member 'concatenate' of a type (line 448)
        concatenate_369419 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 448, 20), np_369418, 'concatenate')
        # Calling concatenate(args, kwargs) (line 448)
        concatenate_call_result_369428 = invoke(stypy.reporting.localization.Localization(__file__, 448, 20), concatenate_369419, *[tuple_369420], **kwargs_369427)
        
        # Getting the type of 'self' (line 448)
        self_369429 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 448, 8), 'self')
        # Setting the type of the member 'data' of a type (line 448)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 448, 8), self_369429, 'data', concatenate_call_result_369428)
        
        # Assigning a Name to a Attribute (line 449):
        
        # Assigning a Name to a Attribute (line 449):
        
        # Assigning a Name to a Attribute (line 449):
        # Getting the type of 'False' (line 449)
        False_369430 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 449, 36), 'False')
        # Getting the type of 'self' (line 449)
        self_369431 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 449, 8), 'self')
        # Setting the type of the member 'has_canonical_format' of a type (line 449)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 449, 8), self_369431, 'has_canonical_format', False_369430)
        
        # ################# End of '_setdiag(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_setdiag' in the type store
        # Getting the type of 'stypy_return_type' (line 415)
        stypy_return_type_369432 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 415, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_369432)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_setdiag'
        return stypy_return_type_369432


    @norecursion
    def _with_data(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'True' (line 452)
        True_369433 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 452, 34), 'True')
        defaults = [True_369433]
        # Create a new context for function '_with_data'
        module_type_store = module_type_store.open_function_context('_with_data', 452, 4, False)
        # Assigning a type to the variable 'self' (line 453)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 453, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        coo_matrix._with_data.__dict__.__setitem__('stypy_localization', localization)
        coo_matrix._with_data.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        coo_matrix._with_data.__dict__.__setitem__('stypy_type_store', module_type_store)
        coo_matrix._with_data.__dict__.__setitem__('stypy_function_name', 'coo_matrix._with_data')
        coo_matrix._with_data.__dict__.__setitem__('stypy_param_names_list', ['data', 'copy'])
        coo_matrix._with_data.__dict__.__setitem__('stypy_varargs_param_name', None)
        coo_matrix._with_data.__dict__.__setitem__('stypy_kwargs_param_name', None)
        coo_matrix._with_data.__dict__.__setitem__('stypy_call_defaults', defaults)
        coo_matrix._with_data.__dict__.__setitem__('stypy_call_varargs', varargs)
        coo_matrix._with_data.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        coo_matrix._with_data.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'coo_matrix._with_data', ['data', 'copy'], None, None, defaults, varargs, kwargs)

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

        str_369434 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 456, (-1)), 'str', 'Returns a matrix with the same sparsity structure as self,\n        but with different data.  By default the index arrays\n        (i.e. .row and .col) are copied.\n        ')
        
        # Getting the type of 'copy' (line 457)
        copy_369435 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 457, 11), 'copy')
        # Testing the type of an if condition (line 457)
        if_condition_369436 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 457, 8), copy_369435)
        # Assigning a type to the variable 'if_condition_369436' (line 457)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 457, 8), 'if_condition_369436', if_condition_369436)
        # SSA begins for if statement (line 457)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to coo_matrix(...): (line 458)
        # Processing the call arguments (line 458)
        
        # Obtaining an instance of the builtin type 'tuple' (line 458)
        tuple_369438 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 458, 31), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 458)
        # Adding element type (line 458)
        # Getting the type of 'data' (line 458)
        data_369439 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 458, 31), 'data', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 458, 31), tuple_369438, data_369439)
        # Adding element type (line 458)
        
        # Obtaining an instance of the builtin type 'tuple' (line 458)
        tuple_369440 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 458, 38), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 458)
        # Adding element type (line 458)
        
        # Call to copy(...): (line 458)
        # Processing the call keyword arguments (line 458)
        kwargs_369444 = {}
        # Getting the type of 'self' (line 458)
        self_369441 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 458, 38), 'self', False)
        # Obtaining the member 'row' of a type (line 458)
        row_369442 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 458, 38), self_369441, 'row')
        # Obtaining the member 'copy' of a type (line 458)
        copy_369443 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 458, 38), row_369442, 'copy')
        # Calling copy(args, kwargs) (line 458)
        copy_call_result_369445 = invoke(stypy.reporting.localization.Localization(__file__, 458, 38), copy_369443, *[], **kwargs_369444)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 458, 38), tuple_369440, copy_call_result_369445)
        # Adding element type (line 458)
        
        # Call to copy(...): (line 458)
        # Processing the call keyword arguments (line 458)
        kwargs_369449 = {}
        # Getting the type of 'self' (line 458)
        self_369446 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 458, 55), 'self', False)
        # Obtaining the member 'col' of a type (line 458)
        col_369447 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 458, 55), self_369446, 'col')
        # Obtaining the member 'copy' of a type (line 458)
        copy_369448 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 458, 55), col_369447, 'copy')
        # Calling copy(args, kwargs) (line 458)
        copy_call_result_369450 = invoke(stypy.reporting.localization.Localization(__file__, 458, 55), copy_369448, *[], **kwargs_369449)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 458, 38), tuple_369440, copy_call_result_369450)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 458, 31), tuple_369438, tuple_369440)
        
        # Processing the call keyword arguments (line 458)
        # Getting the type of 'self' (line 459)
        self_369451 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 459, 41), 'self', False)
        # Obtaining the member 'shape' of a type (line 459)
        shape_369452 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 459, 41), self_369451, 'shape')
        keyword_369453 = shape_369452
        # Getting the type of 'data' (line 459)
        data_369454 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 459, 59), 'data', False)
        # Obtaining the member 'dtype' of a type (line 459)
        dtype_369455 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 459, 59), data_369454, 'dtype')
        keyword_369456 = dtype_369455
        kwargs_369457 = {'dtype': keyword_369456, 'shape': keyword_369453}
        # Getting the type of 'coo_matrix' (line 458)
        coo_matrix_369437 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 458, 19), 'coo_matrix', False)
        # Calling coo_matrix(args, kwargs) (line 458)
        coo_matrix_call_result_369458 = invoke(stypy.reporting.localization.Localization(__file__, 458, 19), coo_matrix_369437, *[tuple_369438], **kwargs_369457)
        
        # Assigning a type to the variable 'stypy_return_type' (line 458)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 458, 12), 'stypy_return_type', coo_matrix_call_result_369458)
        # SSA branch for the else part of an if statement (line 457)
        module_type_store.open_ssa_branch('else')
        
        # Call to coo_matrix(...): (line 461)
        # Processing the call arguments (line 461)
        
        # Obtaining an instance of the builtin type 'tuple' (line 461)
        tuple_369460 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 461, 31), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 461)
        # Adding element type (line 461)
        # Getting the type of 'data' (line 461)
        data_369461 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 461, 31), 'data', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 461, 31), tuple_369460, data_369461)
        # Adding element type (line 461)
        
        # Obtaining an instance of the builtin type 'tuple' (line 461)
        tuple_369462 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 461, 38), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 461)
        # Adding element type (line 461)
        # Getting the type of 'self' (line 461)
        self_369463 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 461, 38), 'self', False)
        # Obtaining the member 'row' of a type (line 461)
        row_369464 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 461, 38), self_369463, 'row')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 461, 38), tuple_369462, row_369464)
        # Adding element type (line 461)
        # Getting the type of 'self' (line 461)
        self_369465 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 461, 48), 'self', False)
        # Obtaining the member 'col' of a type (line 461)
        col_369466 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 461, 48), self_369465, 'col')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 461, 38), tuple_369462, col_369466)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 461, 31), tuple_369460, tuple_369462)
        
        # Processing the call keyword arguments (line 461)
        # Getting the type of 'self' (line 462)
        self_369467 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 462, 41), 'self', False)
        # Obtaining the member 'shape' of a type (line 462)
        shape_369468 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 462, 41), self_369467, 'shape')
        keyword_369469 = shape_369468
        # Getting the type of 'data' (line 462)
        data_369470 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 462, 59), 'data', False)
        # Obtaining the member 'dtype' of a type (line 462)
        dtype_369471 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 462, 59), data_369470, 'dtype')
        keyword_369472 = dtype_369471
        kwargs_369473 = {'dtype': keyword_369472, 'shape': keyword_369469}
        # Getting the type of 'coo_matrix' (line 461)
        coo_matrix_369459 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 461, 19), 'coo_matrix', False)
        # Calling coo_matrix(args, kwargs) (line 461)
        coo_matrix_call_result_369474 = invoke(stypy.reporting.localization.Localization(__file__, 461, 19), coo_matrix_369459, *[tuple_369460], **kwargs_369473)
        
        # Assigning a type to the variable 'stypy_return_type' (line 461)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 461, 12), 'stypy_return_type', coo_matrix_call_result_369474)
        # SSA join for if statement (line 457)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of '_with_data(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_with_data' in the type store
        # Getting the type of 'stypy_return_type' (line 452)
        stypy_return_type_369475 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 452, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_369475)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_with_data'
        return stypy_return_type_369475


    @norecursion
    def sum_duplicates(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'sum_duplicates'
        module_type_store = module_type_store.open_function_context('sum_duplicates', 464, 4, False)
        # Assigning a type to the variable 'self' (line 465)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 465, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        coo_matrix.sum_duplicates.__dict__.__setitem__('stypy_localization', localization)
        coo_matrix.sum_duplicates.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        coo_matrix.sum_duplicates.__dict__.__setitem__('stypy_type_store', module_type_store)
        coo_matrix.sum_duplicates.__dict__.__setitem__('stypy_function_name', 'coo_matrix.sum_duplicates')
        coo_matrix.sum_duplicates.__dict__.__setitem__('stypy_param_names_list', [])
        coo_matrix.sum_duplicates.__dict__.__setitem__('stypy_varargs_param_name', None)
        coo_matrix.sum_duplicates.__dict__.__setitem__('stypy_kwargs_param_name', None)
        coo_matrix.sum_duplicates.__dict__.__setitem__('stypy_call_defaults', defaults)
        coo_matrix.sum_duplicates.__dict__.__setitem__('stypy_call_varargs', varargs)
        coo_matrix.sum_duplicates.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        coo_matrix.sum_duplicates.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'coo_matrix.sum_duplicates', [], None, None, defaults, varargs, kwargs)

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

        str_369476 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 468, (-1)), 'str', 'Eliminate duplicate matrix entries by adding them together\n\n        This is an *in place* operation\n        ')
        
        # Getting the type of 'self' (line 469)
        self_369477 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 469, 11), 'self')
        # Obtaining the member 'has_canonical_format' of a type (line 469)
        has_canonical_format_369478 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 469, 11), self_369477, 'has_canonical_format')
        # Testing the type of an if condition (line 469)
        if_condition_369479 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 469, 8), has_canonical_format_369478)
        # Assigning a type to the variable 'if_condition_369479' (line 469)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 469, 8), 'if_condition_369479', if_condition_369479)
        # SSA begins for if statement (line 469)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # Assigning a type to the variable 'stypy_return_type' (line 470)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 470, 12), 'stypy_return_type', types.NoneType)
        # SSA join for if statement (line 469)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Name (line 471):
        
        # Assigning a Call to a Name (line 471):
        
        # Assigning a Call to a Name (line 471):
        
        # Call to _sum_duplicates(...): (line 471)
        # Processing the call arguments (line 471)
        # Getting the type of 'self' (line 471)
        self_369482 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 471, 38), 'self', False)
        # Obtaining the member 'row' of a type (line 471)
        row_369483 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 471, 38), self_369482, 'row')
        # Getting the type of 'self' (line 471)
        self_369484 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 471, 48), 'self', False)
        # Obtaining the member 'col' of a type (line 471)
        col_369485 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 471, 48), self_369484, 'col')
        # Getting the type of 'self' (line 471)
        self_369486 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 471, 58), 'self', False)
        # Obtaining the member 'data' of a type (line 471)
        data_369487 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 471, 58), self_369486, 'data')
        # Processing the call keyword arguments (line 471)
        kwargs_369488 = {}
        # Getting the type of 'self' (line 471)
        self_369480 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 471, 17), 'self', False)
        # Obtaining the member '_sum_duplicates' of a type (line 471)
        _sum_duplicates_369481 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 471, 17), self_369480, '_sum_duplicates')
        # Calling _sum_duplicates(args, kwargs) (line 471)
        _sum_duplicates_call_result_369489 = invoke(stypy.reporting.localization.Localization(__file__, 471, 17), _sum_duplicates_369481, *[row_369483, col_369485, data_369487], **kwargs_369488)
        
        # Assigning a type to the variable 'summed' (line 471)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 471, 8), 'summed', _sum_duplicates_call_result_369489)
        
        # Assigning a Name to a Tuple (line 472):
        
        # Assigning a Subscript to a Name (line 472):
        
        # Assigning a Subscript to a Name (line 472):
        
        # Obtaining the type of the subscript
        int_369490 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 472, 8), 'int')
        # Getting the type of 'summed' (line 472)
        summed_369491 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 472, 40), 'summed')
        # Obtaining the member '__getitem__' of a type (line 472)
        getitem___369492 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 472, 8), summed_369491, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 472)
        subscript_call_result_369493 = invoke(stypy.reporting.localization.Localization(__file__, 472, 8), getitem___369492, int_369490)
        
        # Assigning a type to the variable 'tuple_var_assignment_368045' (line 472)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 472, 8), 'tuple_var_assignment_368045', subscript_call_result_369493)
        
        # Assigning a Subscript to a Name (line 472):
        
        # Assigning a Subscript to a Name (line 472):
        
        # Obtaining the type of the subscript
        int_369494 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 472, 8), 'int')
        # Getting the type of 'summed' (line 472)
        summed_369495 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 472, 40), 'summed')
        # Obtaining the member '__getitem__' of a type (line 472)
        getitem___369496 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 472, 8), summed_369495, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 472)
        subscript_call_result_369497 = invoke(stypy.reporting.localization.Localization(__file__, 472, 8), getitem___369496, int_369494)
        
        # Assigning a type to the variable 'tuple_var_assignment_368046' (line 472)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 472, 8), 'tuple_var_assignment_368046', subscript_call_result_369497)
        
        # Assigning a Subscript to a Name (line 472):
        
        # Assigning a Subscript to a Name (line 472):
        
        # Obtaining the type of the subscript
        int_369498 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 472, 8), 'int')
        # Getting the type of 'summed' (line 472)
        summed_369499 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 472, 40), 'summed')
        # Obtaining the member '__getitem__' of a type (line 472)
        getitem___369500 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 472, 8), summed_369499, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 472)
        subscript_call_result_369501 = invoke(stypy.reporting.localization.Localization(__file__, 472, 8), getitem___369500, int_369498)
        
        # Assigning a type to the variable 'tuple_var_assignment_368047' (line 472)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 472, 8), 'tuple_var_assignment_368047', subscript_call_result_369501)
        
        # Assigning a Name to a Attribute (line 472):
        
        # Assigning a Name to a Attribute (line 472):
        # Getting the type of 'tuple_var_assignment_368045' (line 472)
        tuple_var_assignment_368045_369502 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 472, 8), 'tuple_var_assignment_368045')
        # Getting the type of 'self' (line 472)
        self_369503 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 472, 8), 'self')
        # Setting the type of the member 'row' of a type (line 472)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 472, 8), self_369503, 'row', tuple_var_assignment_368045_369502)
        
        # Assigning a Name to a Attribute (line 472):
        
        # Assigning a Name to a Attribute (line 472):
        # Getting the type of 'tuple_var_assignment_368046' (line 472)
        tuple_var_assignment_368046_369504 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 472, 8), 'tuple_var_assignment_368046')
        # Getting the type of 'self' (line 472)
        self_369505 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 472, 18), 'self')
        # Setting the type of the member 'col' of a type (line 472)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 472, 18), self_369505, 'col', tuple_var_assignment_368046_369504)
        
        # Assigning a Name to a Attribute (line 472):
        
        # Assigning a Name to a Attribute (line 472):
        # Getting the type of 'tuple_var_assignment_368047' (line 472)
        tuple_var_assignment_368047_369506 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 472, 8), 'tuple_var_assignment_368047')
        # Getting the type of 'self' (line 472)
        self_369507 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 472, 28), 'self')
        # Setting the type of the member 'data' of a type (line 472)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 472, 28), self_369507, 'data', tuple_var_assignment_368047_369506)
        
        # Assigning a Name to a Attribute (line 473):
        
        # Assigning a Name to a Attribute (line 473):
        
        # Assigning a Name to a Attribute (line 473):
        # Getting the type of 'True' (line 473)
        True_369508 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 473, 36), 'True')
        # Getting the type of 'self' (line 473)
        self_369509 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 473, 8), 'self')
        # Setting the type of the member 'has_canonical_format' of a type (line 473)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 473, 8), self_369509, 'has_canonical_format', True_369508)
        
        # ################# End of 'sum_duplicates(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'sum_duplicates' in the type store
        # Getting the type of 'stypy_return_type' (line 464)
        stypy_return_type_369510 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 464, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_369510)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'sum_duplicates'
        return stypy_return_type_369510


    @norecursion
    def _sum_duplicates(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_sum_duplicates'
        module_type_store = module_type_store.open_function_context('_sum_duplicates', 475, 4, False)
        # Assigning a type to the variable 'self' (line 476)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 476, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        coo_matrix._sum_duplicates.__dict__.__setitem__('stypy_localization', localization)
        coo_matrix._sum_duplicates.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        coo_matrix._sum_duplicates.__dict__.__setitem__('stypy_type_store', module_type_store)
        coo_matrix._sum_duplicates.__dict__.__setitem__('stypy_function_name', 'coo_matrix._sum_duplicates')
        coo_matrix._sum_duplicates.__dict__.__setitem__('stypy_param_names_list', ['row', 'col', 'data'])
        coo_matrix._sum_duplicates.__dict__.__setitem__('stypy_varargs_param_name', None)
        coo_matrix._sum_duplicates.__dict__.__setitem__('stypy_kwargs_param_name', None)
        coo_matrix._sum_duplicates.__dict__.__setitem__('stypy_call_defaults', defaults)
        coo_matrix._sum_duplicates.__dict__.__setitem__('stypy_call_varargs', varargs)
        coo_matrix._sum_duplicates.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        coo_matrix._sum_duplicates.__dict__.__setitem__('stypy_declared_arg_number', 4)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'coo_matrix._sum_duplicates', ['row', 'col', 'data'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_sum_duplicates', localization, ['row', 'col', 'data'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_sum_duplicates(...)' code ##################

        
        
        
        # Call to len(...): (line 477)
        # Processing the call arguments (line 477)
        # Getting the type of 'data' (line 477)
        data_369512 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 477, 15), 'data', False)
        # Processing the call keyword arguments (line 477)
        kwargs_369513 = {}
        # Getting the type of 'len' (line 477)
        len_369511 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 477, 11), 'len', False)
        # Calling len(args, kwargs) (line 477)
        len_call_result_369514 = invoke(stypy.reporting.localization.Localization(__file__, 477, 11), len_369511, *[data_369512], **kwargs_369513)
        
        int_369515 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 477, 24), 'int')
        # Applying the binary operator '==' (line 477)
        result_eq_369516 = python_operator(stypy.reporting.localization.Localization(__file__, 477, 11), '==', len_call_result_369514, int_369515)
        
        # Testing the type of an if condition (line 477)
        if_condition_369517 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 477, 8), result_eq_369516)
        # Assigning a type to the variable 'if_condition_369517' (line 477)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 477, 8), 'if_condition_369517', if_condition_369517)
        # SSA begins for if statement (line 477)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Obtaining an instance of the builtin type 'tuple' (line 478)
        tuple_369518 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 478, 19), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 478)
        # Adding element type (line 478)
        # Getting the type of 'row' (line 478)
        row_369519 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 478, 19), 'row')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 478, 19), tuple_369518, row_369519)
        # Adding element type (line 478)
        # Getting the type of 'col' (line 478)
        col_369520 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 478, 24), 'col')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 478, 19), tuple_369518, col_369520)
        # Adding element type (line 478)
        # Getting the type of 'data' (line 478)
        data_369521 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 478, 29), 'data')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 478, 19), tuple_369518, data_369521)
        
        # Assigning a type to the variable 'stypy_return_type' (line 478)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 478, 12), 'stypy_return_type', tuple_369518)
        # SSA join for if statement (line 477)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Name (line 479):
        
        # Assigning a Call to a Name (line 479):
        
        # Assigning a Call to a Name (line 479):
        
        # Call to lexsort(...): (line 479)
        # Processing the call arguments (line 479)
        
        # Obtaining an instance of the builtin type 'tuple' (line 479)
        tuple_369524 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 479, 28), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 479)
        # Adding element type (line 479)
        # Getting the type of 'row' (line 479)
        row_369525 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 479, 28), 'row', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 479, 28), tuple_369524, row_369525)
        # Adding element type (line 479)
        # Getting the type of 'col' (line 479)
        col_369526 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 479, 33), 'col', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 479, 28), tuple_369524, col_369526)
        
        # Processing the call keyword arguments (line 479)
        kwargs_369527 = {}
        # Getting the type of 'np' (line 479)
        np_369522 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 479, 16), 'np', False)
        # Obtaining the member 'lexsort' of a type (line 479)
        lexsort_369523 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 479, 16), np_369522, 'lexsort')
        # Calling lexsort(args, kwargs) (line 479)
        lexsort_call_result_369528 = invoke(stypy.reporting.localization.Localization(__file__, 479, 16), lexsort_369523, *[tuple_369524], **kwargs_369527)
        
        # Assigning a type to the variable 'order' (line 479)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 479, 8), 'order', lexsort_call_result_369528)
        
        # Assigning a Subscript to a Name (line 480):
        
        # Assigning a Subscript to a Name (line 480):
        
        # Assigning a Subscript to a Name (line 480):
        
        # Obtaining the type of the subscript
        # Getting the type of 'order' (line 480)
        order_369529 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 480, 18), 'order')
        # Getting the type of 'row' (line 480)
        row_369530 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 480, 14), 'row')
        # Obtaining the member '__getitem__' of a type (line 480)
        getitem___369531 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 480, 14), row_369530, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 480)
        subscript_call_result_369532 = invoke(stypy.reporting.localization.Localization(__file__, 480, 14), getitem___369531, order_369529)
        
        # Assigning a type to the variable 'row' (line 480)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 480, 8), 'row', subscript_call_result_369532)
        
        # Assigning a Subscript to a Name (line 481):
        
        # Assigning a Subscript to a Name (line 481):
        
        # Assigning a Subscript to a Name (line 481):
        
        # Obtaining the type of the subscript
        # Getting the type of 'order' (line 481)
        order_369533 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 481, 18), 'order')
        # Getting the type of 'col' (line 481)
        col_369534 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 481, 14), 'col')
        # Obtaining the member '__getitem__' of a type (line 481)
        getitem___369535 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 481, 14), col_369534, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 481)
        subscript_call_result_369536 = invoke(stypy.reporting.localization.Localization(__file__, 481, 14), getitem___369535, order_369533)
        
        # Assigning a type to the variable 'col' (line 481)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 481, 8), 'col', subscript_call_result_369536)
        
        # Assigning a Subscript to a Name (line 482):
        
        # Assigning a Subscript to a Name (line 482):
        
        # Assigning a Subscript to a Name (line 482):
        
        # Obtaining the type of the subscript
        # Getting the type of 'order' (line 482)
        order_369537 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 482, 20), 'order')
        # Getting the type of 'data' (line 482)
        data_369538 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 482, 15), 'data')
        # Obtaining the member '__getitem__' of a type (line 482)
        getitem___369539 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 482, 15), data_369538, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 482)
        subscript_call_result_369540 = invoke(stypy.reporting.localization.Localization(__file__, 482, 15), getitem___369539, order_369537)
        
        # Assigning a type to the variable 'data' (line 482)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 482, 8), 'data', subscript_call_result_369540)
        
        # Assigning a BinOp to a Name (line 483):
        
        # Assigning a BinOp to a Name (line 483):
        
        # Assigning a BinOp to a Name (line 483):
        
        
        # Obtaining the type of the subscript
        int_369541 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 483, 28), 'int')
        slice_369542 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 483, 24), int_369541, None, None)
        # Getting the type of 'row' (line 483)
        row_369543 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 483, 24), 'row')
        # Obtaining the member '__getitem__' of a type (line 483)
        getitem___369544 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 483, 24), row_369543, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 483)
        subscript_call_result_369545 = invoke(stypy.reporting.localization.Localization(__file__, 483, 24), getitem___369544, slice_369542)
        
        
        # Obtaining the type of the subscript
        int_369546 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 483, 40), 'int')
        slice_369547 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 483, 35), None, int_369546, None)
        # Getting the type of 'row' (line 483)
        row_369548 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 483, 35), 'row')
        # Obtaining the member '__getitem__' of a type (line 483)
        getitem___369549 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 483, 35), row_369548, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 483)
        subscript_call_result_369550 = invoke(stypy.reporting.localization.Localization(__file__, 483, 35), getitem___369549, slice_369547)
        
        # Applying the binary operator '!=' (line 483)
        result_ne_369551 = python_operator(stypy.reporting.localization.Localization(__file__, 483, 24), '!=', subscript_call_result_369545, subscript_call_result_369550)
        
        
        
        # Obtaining the type of the subscript
        int_369552 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 484, 28), 'int')
        slice_369553 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 484, 24), int_369552, None, None)
        # Getting the type of 'col' (line 484)
        col_369554 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 484, 24), 'col')
        # Obtaining the member '__getitem__' of a type (line 484)
        getitem___369555 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 484, 24), col_369554, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 484)
        subscript_call_result_369556 = invoke(stypy.reporting.localization.Localization(__file__, 484, 24), getitem___369555, slice_369553)
        
        
        # Obtaining the type of the subscript
        int_369557 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 484, 40), 'int')
        slice_369558 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 484, 35), None, int_369557, None)
        # Getting the type of 'col' (line 484)
        col_369559 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 484, 35), 'col')
        # Obtaining the member '__getitem__' of a type (line 484)
        getitem___369560 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 484, 35), col_369559, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 484)
        subscript_call_result_369561 = invoke(stypy.reporting.localization.Localization(__file__, 484, 35), getitem___369560, slice_369558)
        
        # Applying the binary operator '!=' (line 484)
        result_ne_369562 = python_operator(stypy.reporting.localization.Localization(__file__, 484, 24), '!=', subscript_call_result_369556, subscript_call_result_369561)
        
        # Applying the binary operator '|' (line 483)
        result_or__369563 = python_operator(stypy.reporting.localization.Localization(__file__, 483, 23), '|', result_ne_369551, result_ne_369562)
        
        # Assigning a type to the variable 'unique_mask' (line 483)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 483, 8), 'unique_mask', result_or__369563)
        
        # Assigning a Call to a Name (line 485):
        
        # Assigning a Call to a Name (line 485):
        
        # Assigning a Call to a Name (line 485):
        
        # Call to append(...): (line 485)
        # Processing the call arguments (line 485)
        # Getting the type of 'True' (line 485)
        True_369566 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 485, 32), 'True', False)
        # Getting the type of 'unique_mask' (line 485)
        unique_mask_369567 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 485, 38), 'unique_mask', False)
        # Processing the call keyword arguments (line 485)
        kwargs_369568 = {}
        # Getting the type of 'np' (line 485)
        np_369564 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 485, 22), 'np', False)
        # Obtaining the member 'append' of a type (line 485)
        append_369565 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 485, 22), np_369564, 'append')
        # Calling append(args, kwargs) (line 485)
        append_call_result_369569 = invoke(stypy.reporting.localization.Localization(__file__, 485, 22), append_369565, *[True_369566, unique_mask_369567], **kwargs_369568)
        
        # Assigning a type to the variable 'unique_mask' (line 485)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 485, 8), 'unique_mask', append_call_result_369569)
        
        # Assigning a Subscript to a Name (line 486):
        
        # Assigning a Subscript to a Name (line 486):
        
        # Assigning a Subscript to a Name (line 486):
        
        # Obtaining the type of the subscript
        # Getting the type of 'unique_mask' (line 486)
        unique_mask_369570 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 486, 18), 'unique_mask')
        # Getting the type of 'row' (line 486)
        row_369571 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 486, 14), 'row')
        # Obtaining the member '__getitem__' of a type (line 486)
        getitem___369572 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 486, 14), row_369571, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 486)
        subscript_call_result_369573 = invoke(stypy.reporting.localization.Localization(__file__, 486, 14), getitem___369572, unique_mask_369570)
        
        # Assigning a type to the variable 'row' (line 486)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 486, 8), 'row', subscript_call_result_369573)
        
        # Assigning a Subscript to a Name (line 487):
        
        # Assigning a Subscript to a Name (line 487):
        
        # Assigning a Subscript to a Name (line 487):
        
        # Obtaining the type of the subscript
        # Getting the type of 'unique_mask' (line 487)
        unique_mask_369574 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 487, 18), 'unique_mask')
        # Getting the type of 'col' (line 487)
        col_369575 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 487, 14), 'col')
        # Obtaining the member '__getitem__' of a type (line 487)
        getitem___369576 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 487, 14), col_369575, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 487)
        subscript_call_result_369577 = invoke(stypy.reporting.localization.Localization(__file__, 487, 14), getitem___369576, unique_mask_369574)
        
        # Assigning a type to the variable 'col' (line 487)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 487, 8), 'col', subscript_call_result_369577)
        
        # Assigning a Call to a Tuple (line 488):
        
        # Assigning a Subscript to a Name (line 488):
        
        # Assigning a Subscript to a Name (line 488):
        
        # Obtaining the type of the subscript
        int_369578 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 488, 8), 'int')
        
        # Call to nonzero(...): (line 488)
        # Processing the call arguments (line 488)
        # Getting the type of 'unique_mask' (line 488)
        unique_mask_369581 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 488, 34), 'unique_mask', False)
        # Processing the call keyword arguments (line 488)
        kwargs_369582 = {}
        # Getting the type of 'np' (line 488)
        np_369579 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 488, 23), 'np', False)
        # Obtaining the member 'nonzero' of a type (line 488)
        nonzero_369580 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 488, 23), np_369579, 'nonzero')
        # Calling nonzero(args, kwargs) (line 488)
        nonzero_call_result_369583 = invoke(stypy.reporting.localization.Localization(__file__, 488, 23), nonzero_369580, *[unique_mask_369581], **kwargs_369582)
        
        # Obtaining the member '__getitem__' of a type (line 488)
        getitem___369584 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 488, 8), nonzero_call_result_369583, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 488)
        subscript_call_result_369585 = invoke(stypy.reporting.localization.Localization(__file__, 488, 8), getitem___369584, int_369578)
        
        # Assigning a type to the variable 'tuple_var_assignment_368048' (line 488)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 488, 8), 'tuple_var_assignment_368048', subscript_call_result_369585)
        
        # Assigning a Name to a Name (line 488):
        
        # Assigning a Name to a Name (line 488):
        # Getting the type of 'tuple_var_assignment_368048' (line 488)
        tuple_var_assignment_368048_369586 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 488, 8), 'tuple_var_assignment_368048')
        # Assigning a type to the variable 'unique_inds' (line 488)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 488, 8), 'unique_inds', tuple_var_assignment_368048_369586)
        
        # Assigning a Call to a Name (line 489):
        
        # Assigning a Call to a Name (line 489):
        
        # Assigning a Call to a Name (line 489):
        
        # Call to reduceat(...): (line 489)
        # Processing the call arguments (line 489)
        # Getting the type of 'data' (line 489)
        data_369590 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 489, 31), 'data', False)
        # Getting the type of 'unique_inds' (line 489)
        unique_inds_369591 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 489, 37), 'unique_inds', False)
        # Processing the call keyword arguments (line 489)
        # Getting the type of 'self' (line 489)
        self_369592 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 489, 56), 'self', False)
        # Obtaining the member 'dtype' of a type (line 489)
        dtype_369593 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 489, 56), self_369592, 'dtype')
        keyword_369594 = dtype_369593
        kwargs_369595 = {'dtype': keyword_369594}
        # Getting the type of 'np' (line 489)
        np_369587 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 489, 15), 'np', False)
        # Obtaining the member 'add' of a type (line 489)
        add_369588 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 489, 15), np_369587, 'add')
        # Obtaining the member 'reduceat' of a type (line 489)
        reduceat_369589 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 489, 15), add_369588, 'reduceat')
        # Calling reduceat(args, kwargs) (line 489)
        reduceat_call_result_369596 = invoke(stypy.reporting.localization.Localization(__file__, 489, 15), reduceat_369589, *[data_369590, unique_inds_369591], **kwargs_369595)
        
        # Assigning a type to the variable 'data' (line 489)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 489, 8), 'data', reduceat_call_result_369596)
        
        # Obtaining an instance of the builtin type 'tuple' (line 490)
        tuple_369597 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 490, 15), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 490)
        # Adding element type (line 490)
        # Getting the type of 'row' (line 490)
        row_369598 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 490, 15), 'row')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 490, 15), tuple_369597, row_369598)
        # Adding element type (line 490)
        # Getting the type of 'col' (line 490)
        col_369599 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 490, 20), 'col')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 490, 15), tuple_369597, col_369599)
        # Adding element type (line 490)
        # Getting the type of 'data' (line 490)
        data_369600 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 490, 25), 'data')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 490, 15), tuple_369597, data_369600)
        
        # Assigning a type to the variable 'stypy_return_type' (line 490)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 490, 8), 'stypy_return_type', tuple_369597)
        
        # ################# End of '_sum_duplicates(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_sum_duplicates' in the type store
        # Getting the type of 'stypy_return_type' (line 475)
        stypy_return_type_369601 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 475, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_369601)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_sum_duplicates'
        return stypy_return_type_369601


    @norecursion
    def eliminate_zeros(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'eliminate_zeros'
        module_type_store = module_type_store.open_function_context('eliminate_zeros', 492, 4, False)
        # Assigning a type to the variable 'self' (line 493)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 493, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        coo_matrix.eliminate_zeros.__dict__.__setitem__('stypy_localization', localization)
        coo_matrix.eliminate_zeros.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        coo_matrix.eliminate_zeros.__dict__.__setitem__('stypy_type_store', module_type_store)
        coo_matrix.eliminate_zeros.__dict__.__setitem__('stypy_function_name', 'coo_matrix.eliminate_zeros')
        coo_matrix.eliminate_zeros.__dict__.__setitem__('stypy_param_names_list', [])
        coo_matrix.eliminate_zeros.__dict__.__setitem__('stypy_varargs_param_name', None)
        coo_matrix.eliminate_zeros.__dict__.__setitem__('stypy_kwargs_param_name', None)
        coo_matrix.eliminate_zeros.__dict__.__setitem__('stypy_call_defaults', defaults)
        coo_matrix.eliminate_zeros.__dict__.__setitem__('stypy_call_varargs', varargs)
        coo_matrix.eliminate_zeros.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        coo_matrix.eliminate_zeros.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'coo_matrix.eliminate_zeros', [], None, None, defaults, varargs, kwargs)

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

        str_369602 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 496, (-1)), 'str', 'Remove zero entries from the matrix\n\n        This is an *in place* operation\n        ')
        
        # Assigning a Compare to a Name (line 497):
        
        # Assigning a Compare to a Name (line 497):
        
        # Assigning a Compare to a Name (line 497):
        
        # Getting the type of 'self' (line 497)
        self_369603 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 497, 15), 'self')
        # Obtaining the member 'data' of a type (line 497)
        data_369604 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 497, 15), self_369603, 'data')
        int_369605 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 497, 28), 'int')
        # Applying the binary operator '!=' (line 497)
        result_ne_369606 = python_operator(stypy.reporting.localization.Localization(__file__, 497, 15), '!=', data_369604, int_369605)
        
        # Assigning a type to the variable 'mask' (line 497)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 497, 8), 'mask', result_ne_369606)
        
        # Assigning a Subscript to a Attribute (line 498):
        
        # Assigning a Subscript to a Attribute (line 498):
        
        # Assigning a Subscript to a Attribute (line 498):
        
        # Obtaining the type of the subscript
        # Getting the type of 'mask' (line 498)
        mask_369607 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 498, 30), 'mask')
        # Getting the type of 'self' (line 498)
        self_369608 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 498, 20), 'self')
        # Obtaining the member 'data' of a type (line 498)
        data_369609 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 498, 20), self_369608, 'data')
        # Obtaining the member '__getitem__' of a type (line 498)
        getitem___369610 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 498, 20), data_369609, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 498)
        subscript_call_result_369611 = invoke(stypy.reporting.localization.Localization(__file__, 498, 20), getitem___369610, mask_369607)
        
        # Getting the type of 'self' (line 498)
        self_369612 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 498, 8), 'self')
        # Setting the type of the member 'data' of a type (line 498)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 498, 8), self_369612, 'data', subscript_call_result_369611)
        
        # Assigning a Subscript to a Attribute (line 499):
        
        # Assigning a Subscript to a Attribute (line 499):
        
        # Assigning a Subscript to a Attribute (line 499):
        
        # Obtaining the type of the subscript
        # Getting the type of 'mask' (line 499)
        mask_369613 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 499, 28), 'mask')
        # Getting the type of 'self' (line 499)
        self_369614 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 499, 19), 'self')
        # Obtaining the member 'row' of a type (line 499)
        row_369615 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 499, 19), self_369614, 'row')
        # Obtaining the member '__getitem__' of a type (line 499)
        getitem___369616 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 499, 19), row_369615, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 499)
        subscript_call_result_369617 = invoke(stypy.reporting.localization.Localization(__file__, 499, 19), getitem___369616, mask_369613)
        
        # Getting the type of 'self' (line 499)
        self_369618 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 499, 8), 'self')
        # Setting the type of the member 'row' of a type (line 499)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 499, 8), self_369618, 'row', subscript_call_result_369617)
        
        # Assigning a Subscript to a Attribute (line 500):
        
        # Assigning a Subscript to a Attribute (line 500):
        
        # Assigning a Subscript to a Attribute (line 500):
        
        # Obtaining the type of the subscript
        # Getting the type of 'mask' (line 500)
        mask_369619 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 500, 28), 'mask')
        # Getting the type of 'self' (line 500)
        self_369620 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 500, 19), 'self')
        # Obtaining the member 'col' of a type (line 500)
        col_369621 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 500, 19), self_369620, 'col')
        # Obtaining the member '__getitem__' of a type (line 500)
        getitem___369622 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 500, 19), col_369621, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 500)
        subscript_call_result_369623 = invoke(stypy.reporting.localization.Localization(__file__, 500, 19), getitem___369622, mask_369619)
        
        # Getting the type of 'self' (line 500)
        self_369624 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 500, 8), 'self')
        # Setting the type of the member 'col' of a type (line 500)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 500, 8), self_369624, 'col', subscript_call_result_369623)
        
        # ################# End of 'eliminate_zeros(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'eliminate_zeros' in the type store
        # Getting the type of 'stypy_return_type' (line 492)
        stypy_return_type_369625 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 492, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_369625)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'eliminate_zeros'
        return stypy_return_type_369625


    @norecursion
    def _add_dense(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_add_dense'
        module_type_store = module_type_store.open_function_context('_add_dense', 506, 4, False)
        # Assigning a type to the variable 'self' (line 507)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 507, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        coo_matrix._add_dense.__dict__.__setitem__('stypy_localization', localization)
        coo_matrix._add_dense.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        coo_matrix._add_dense.__dict__.__setitem__('stypy_type_store', module_type_store)
        coo_matrix._add_dense.__dict__.__setitem__('stypy_function_name', 'coo_matrix._add_dense')
        coo_matrix._add_dense.__dict__.__setitem__('stypy_param_names_list', ['other'])
        coo_matrix._add_dense.__dict__.__setitem__('stypy_varargs_param_name', None)
        coo_matrix._add_dense.__dict__.__setitem__('stypy_kwargs_param_name', None)
        coo_matrix._add_dense.__dict__.__setitem__('stypy_call_defaults', defaults)
        coo_matrix._add_dense.__dict__.__setitem__('stypy_call_varargs', varargs)
        coo_matrix._add_dense.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        coo_matrix._add_dense.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'coo_matrix._add_dense', ['other'], None, None, defaults, varargs, kwargs)

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

        
        
        # Getting the type of 'other' (line 507)
        other_369626 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 507, 11), 'other')
        # Obtaining the member 'shape' of a type (line 507)
        shape_369627 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 507, 11), other_369626, 'shape')
        # Getting the type of 'self' (line 507)
        self_369628 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 507, 26), 'self')
        # Obtaining the member 'shape' of a type (line 507)
        shape_369629 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 507, 26), self_369628, 'shape')
        # Applying the binary operator '!=' (line 507)
        result_ne_369630 = python_operator(stypy.reporting.localization.Localization(__file__, 507, 11), '!=', shape_369627, shape_369629)
        
        # Testing the type of an if condition (line 507)
        if_condition_369631 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 507, 8), result_ne_369630)
        # Assigning a type to the variable 'if_condition_369631' (line 507)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 507, 8), 'if_condition_369631', if_condition_369631)
        # SSA begins for if statement (line 507)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to ValueError(...): (line 508)
        # Processing the call arguments (line 508)
        str_369633 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 508, 29), 'str', 'Incompatible shapes.')
        # Processing the call keyword arguments (line 508)
        kwargs_369634 = {}
        # Getting the type of 'ValueError' (line 508)
        ValueError_369632 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 508, 18), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 508)
        ValueError_call_result_369635 = invoke(stypy.reporting.localization.Localization(__file__, 508, 18), ValueError_369632, *[str_369633], **kwargs_369634)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 508, 12), ValueError_call_result_369635, 'raise parameter', BaseException)
        # SSA join for if statement (line 507)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Name (line 509):
        
        # Assigning a Call to a Name (line 509):
        
        # Assigning a Call to a Name (line 509):
        
        # Call to upcast_char(...): (line 509)
        # Processing the call arguments (line 509)
        # Getting the type of 'self' (line 509)
        self_369637 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 509, 28), 'self', False)
        # Obtaining the member 'dtype' of a type (line 509)
        dtype_369638 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 509, 28), self_369637, 'dtype')
        # Obtaining the member 'char' of a type (line 509)
        char_369639 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 509, 28), dtype_369638, 'char')
        # Getting the type of 'other' (line 509)
        other_369640 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 509, 45), 'other', False)
        # Obtaining the member 'dtype' of a type (line 509)
        dtype_369641 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 509, 45), other_369640, 'dtype')
        # Obtaining the member 'char' of a type (line 509)
        char_369642 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 509, 45), dtype_369641, 'char')
        # Processing the call keyword arguments (line 509)
        kwargs_369643 = {}
        # Getting the type of 'upcast_char' (line 509)
        upcast_char_369636 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 509, 16), 'upcast_char', False)
        # Calling upcast_char(args, kwargs) (line 509)
        upcast_char_call_result_369644 = invoke(stypy.reporting.localization.Localization(__file__, 509, 16), upcast_char_369636, *[char_369639, char_369642], **kwargs_369643)
        
        # Assigning a type to the variable 'dtype' (line 509)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 509, 8), 'dtype', upcast_char_call_result_369644)
        
        # Assigning a Call to a Name (line 510):
        
        # Assigning a Call to a Name (line 510):
        
        # Assigning a Call to a Name (line 510):
        
        # Call to array(...): (line 510)
        # Processing the call arguments (line 510)
        # Getting the type of 'other' (line 510)
        other_369647 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 510, 26), 'other', False)
        # Processing the call keyword arguments (line 510)
        # Getting the type of 'dtype' (line 510)
        dtype_369648 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 510, 39), 'dtype', False)
        keyword_369649 = dtype_369648
        # Getting the type of 'True' (line 510)
        True_369650 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 510, 51), 'True', False)
        keyword_369651 = True_369650
        kwargs_369652 = {'dtype': keyword_369649, 'copy': keyword_369651}
        # Getting the type of 'np' (line 510)
        np_369645 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 510, 17), 'np', False)
        # Obtaining the member 'array' of a type (line 510)
        array_369646 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 510, 17), np_369645, 'array')
        # Calling array(args, kwargs) (line 510)
        array_call_result_369653 = invoke(stypy.reporting.localization.Localization(__file__, 510, 17), array_369646, *[other_369647], **kwargs_369652)
        
        # Assigning a type to the variable 'result' (line 510)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 510, 8), 'result', array_call_result_369653)
        
        # Assigning a Call to a Name (line 511):
        
        # Assigning a Call to a Name (line 511):
        
        # Assigning a Call to a Name (line 511):
        
        # Call to int(...): (line 511)
        # Processing the call arguments (line 511)
        # Getting the type of 'result' (line 511)
        result_369655 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 511, 22), 'result', False)
        # Obtaining the member 'flags' of a type (line 511)
        flags_369656 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 511, 22), result_369655, 'flags')
        # Obtaining the member 'f_contiguous' of a type (line 511)
        f_contiguous_369657 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 511, 22), flags_369656, 'f_contiguous')
        # Processing the call keyword arguments (line 511)
        kwargs_369658 = {}
        # Getting the type of 'int' (line 511)
        int_369654 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 511, 18), 'int', False)
        # Calling int(args, kwargs) (line 511)
        int_call_result_369659 = invoke(stypy.reporting.localization.Localization(__file__, 511, 18), int_369654, *[f_contiguous_369657], **kwargs_369658)
        
        # Assigning a type to the variable 'fortran' (line 511)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 511, 8), 'fortran', int_call_result_369659)
        
        # Assigning a Attribute to a Tuple (line 512):
        
        # Assigning a Subscript to a Name (line 512):
        
        # Assigning a Subscript to a Name (line 512):
        
        # Obtaining the type of the subscript
        int_369660 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 512, 8), 'int')
        # Getting the type of 'self' (line 512)
        self_369661 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 512, 15), 'self')
        # Obtaining the member 'shape' of a type (line 512)
        shape_369662 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 512, 15), self_369661, 'shape')
        # Obtaining the member '__getitem__' of a type (line 512)
        getitem___369663 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 512, 8), shape_369662, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 512)
        subscript_call_result_369664 = invoke(stypy.reporting.localization.Localization(__file__, 512, 8), getitem___369663, int_369660)
        
        # Assigning a type to the variable 'tuple_var_assignment_368049' (line 512)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 512, 8), 'tuple_var_assignment_368049', subscript_call_result_369664)
        
        # Assigning a Subscript to a Name (line 512):
        
        # Assigning a Subscript to a Name (line 512):
        
        # Obtaining the type of the subscript
        int_369665 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 512, 8), 'int')
        # Getting the type of 'self' (line 512)
        self_369666 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 512, 15), 'self')
        # Obtaining the member 'shape' of a type (line 512)
        shape_369667 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 512, 15), self_369666, 'shape')
        # Obtaining the member '__getitem__' of a type (line 512)
        getitem___369668 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 512, 8), shape_369667, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 512)
        subscript_call_result_369669 = invoke(stypy.reporting.localization.Localization(__file__, 512, 8), getitem___369668, int_369665)
        
        # Assigning a type to the variable 'tuple_var_assignment_368050' (line 512)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 512, 8), 'tuple_var_assignment_368050', subscript_call_result_369669)
        
        # Assigning a Name to a Name (line 512):
        
        # Assigning a Name to a Name (line 512):
        # Getting the type of 'tuple_var_assignment_368049' (line 512)
        tuple_var_assignment_368049_369670 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 512, 8), 'tuple_var_assignment_368049')
        # Assigning a type to the variable 'M' (line 512)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 512, 8), 'M', tuple_var_assignment_368049_369670)
        
        # Assigning a Name to a Name (line 512):
        
        # Assigning a Name to a Name (line 512):
        # Getting the type of 'tuple_var_assignment_368050' (line 512)
        tuple_var_assignment_368050_369671 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 512, 8), 'tuple_var_assignment_368050')
        # Assigning a type to the variable 'N' (line 512)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 512, 11), 'N', tuple_var_assignment_368050_369671)
        
        # Call to coo_todense(...): (line 513)
        # Processing the call arguments (line 513)
        # Getting the type of 'M' (line 513)
        M_369673 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 513, 20), 'M', False)
        # Getting the type of 'N' (line 513)
        N_369674 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 513, 23), 'N', False)
        # Getting the type of 'self' (line 513)
        self_369675 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 513, 26), 'self', False)
        # Obtaining the member 'nnz' of a type (line 513)
        nnz_369676 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 513, 26), self_369675, 'nnz')
        # Getting the type of 'self' (line 513)
        self_369677 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 513, 36), 'self', False)
        # Obtaining the member 'row' of a type (line 513)
        row_369678 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 513, 36), self_369677, 'row')
        # Getting the type of 'self' (line 513)
        self_369679 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 513, 46), 'self', False)
        # Obtaining the member 'col' of a type (line 513)
        col_369680 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 513, 46), self_369679, 'col')
        # Getting the type of 'self' (line 513)
        self_369681 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 513, 56), 'self', False)
        # Obtaining the member 'data' of a type (line 513)
        data_369682 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 513, 56), self_369681, 'data')
        
        # Call to ravel(...): (line 514)
        # Processing the call arguments (line 514)
        str_369685 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 514, 33), 'str', 'A')
        # Processing the call keyword arguments (line 514)
        kwargs_369686 = {}
        # Getting the type of 'result' (line 514)
        result_369683 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 514, 20), 'result', False)
        # Obtaining the member 'ravel' of a type (line 514)
        ravel_369684 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 514, 20), result_369683, 'ravel')
        # Calling ravel(args, kwargs) (line 514)
        ravel_call_result_369687 = invoke(stypy.reporting.localization.Localization(__file__, 514, 20), ravel_369684, *[str_369685], **kwargs_369686)
        
        # Getting the type of 'fortran' (line 514)
        fortran_369688 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 514, 39), 'fortran', False)
        # Processing the call keyword arguments (line 513)
        kwargs_369689 = {}
        # Getting the type of 'coo_todense' (line 513)
        coo_todense_369672 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 513, 8), 'coo_todense', False)
        # Calling coo_todense(args, kwargs) (line 513)
        coo_todense_call_result_369690 = invoke(stypy.reporting.localization.Localization(__file__, 513, 8), coo_todense_369672, *[M_369673, N_369674, nnz_369676, row_369678, col_369680, data_369682, ravel_call_result_369687, fortran_369688], **kwargs_369689)
        
        
        # Call to matrix(...): (line 515)
        # Processing the call arguments (line 515)
        # Getting the type of 'result' (line 515)
        result_369693 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 515, 25), 'result', False)
        # Processing the call keyword arguments (line 515)
        # Getting the type of 'False' (line 515)
        False_369694 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 515, 38), 'False', False)
        keyword_369695 = False_369694
        kwargs_369696 = {'copy': keyword_369695}
        # Getting the type of 'np' (line 515)
        np_369691 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 515, 15), 'np', False)
        # Obtaining the member 'matrix' of a type (line 515)
        matrix_369692 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 515, 15), np_369691, 'matrix')
        # Calling matrix(args, kwargs) (line 515)
        matrix_call_result_369697 = invoke(stypy.reporting.localization.Localization(__file__, 515, 15), matrix_369692, *[result_369693], **kwargs_369696)
        
        # Assigning a type to the variable 'stypy_return_type' (line 515)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 515, 8), 'stypy_return_type', matrix_call_result_369697)
        
        # ################# End of '_add_dense(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_add_dense' in the type store
        # Getting the type of 'stypy_return_type' (line 506)
        stypy_return_type_369698 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 506, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_369698)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_add_dense'
        return stypy_return_type_369698


    @norecursion
    def _mul_vector(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_mul_vector'
        module_type_store = module_type_store.open_function_context('_mul_vector', 517, 4, False)
        # Assigning a type to the variable 'self' (line 518)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 518, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        coo_matrix._mul_vector.__dict__.__setitem__('stypy_localization', localization)
        coo_matrix._mul_vector.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        coo_matrix._mul_vector.__dict__.__setitem__('stypy_type_store', module_type_store)
        coo_matrix._mul_vector.__dict__.__setitem__('stypy_function_name', 'coo_matrix._mul_vector')
        coo_matrix._mul_vector.__dict__.__setitem__('stypy_param_names_list', ['other'])
        coo_matrix._mul_vector.__dict__.__setitem__('stypy_varargs_param_name', None)
        coo_matrix._mul_vector.__dict__.__setitem__('stypy_kwargs_param_name', None)
        coo_matrix._mul_vector.__dict__.__setitem__('stypy_call_defaults', defaults)
        coo_matrix._mul_vector.__dict__.__setitem__('stypy_call_varargs', varargs)
        coo_matrix._mul_vector.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        coo_matrix._mul_vector.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'coo_matrix._mul_vector', ['other'], None, None, defaults, varargs, kwargs)

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

        
        # Assigning a Call to a Name (line 519):
        
        # Assigning a Call to a Name (line 519):
        
        # Assigning a Call to a Name (line 519):
        
        # Call to zeros(...): (line 519)
        # Processing the call arguments (line 519)
        
        # Obtaining the type of the subscript
        int_369701 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 519, 37), 'int')
        # Getting the type of 'self' (line 519)
        self_369702 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 519, 26), 'self', False)
        # Obtaining the member 'shape' of a type (line 519)
        shape_369703 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 519, 26), self_369702, 'shape')
        # Obtaining the member '__getitem__' of a type (line 519)
        getitem___369704 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 519, 26), shape_369703, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 519)
        subscript_call_result_369705 = invoke(stypy.reporting.localization.Localization(__file__, 519, 26), getitem___369704, int_369701)
        
        # Processing the call keyword arguments (line 519)
        
        # Call to upcast_char(...): (line 519)
        # Processing the call arguments (line 519)
        # Getting the type of 'self' (line 519)
        self_369707 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 519, 59), 'self', False)
        # Obtaining the member 'dtype' of a type (line 519)
        dtype_369708 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 519, 59), self_369707, 'dtype')
        # Obtaining the member 'char' of a type (line 519)
        char_369709 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 519, 59), dtype_369708, 'char')
        # Getting the type of 'other' (line 520)
        other_369710 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 520, 60), 'other', False)
        # Obtaining the member 'dtype' of a type (line 520)
        dtype_369711 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 520, 60), other_369710, 'dtype')
        # Obtaining the member 'char' of a type (line 520)
        char_369712 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 520, 60), dtype_369711, 'char')
        # Processing the call keyword arguments (line 519)
        kwargs_369713 = {}
        # Getting the type of 'upcast_char' (line 519)
        upcast_char_369706 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 519, 47), 'upcast_char', False)
        # Calling upcast_char(args, kwargs) (line 519)
        upcast_char_call_result_369714 = invoke(stypy.reporting.localization.Localization(__file__, 519, 47), upcast_char_369706, *[char_369709, char_369712], **kwargs_369713)
        
        keyword_369715 = upcast_char_call_result_369714
        kwargs_369716 = {'dtype': keyword_369715}
        # Getting the type of 'np' (line 519)
        np_369699 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 519, 17), 'np', False)
        # Obtaining the member 'zeros' of a type (line 519)
        zeros_369700 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 519, 17), np_369699, 'zeros')
        # Calling zeros(args, kwargs) (line 519)
        zeros_call_result_369717 = invoke(stypy.reporting.localization.Localization(__file__, 519, 17), zeros_369700, *[subscript_call_result_369705], **kwargs_369716)
        
        # Assigning a type to the variable 'result' (line 519)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 519, 8), 'result', zeros_call_result_369717)
        
        # Call to coo_matvec(...): (line 521)
        # Processing the call arguments (line 521)
        # Getting the type of 'self' (line 521)
        self_369719 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 521, 19), 'self', False)
        # Obtaining the member 'nnz' of a type (line 521)
        nnz_369720 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 521, 19), self_369719, 'nnz')
        # Getting the type of 'self' (line 521)
        self_369721 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 521, 29), 'self', False)
        # Obtaining the member 'row' of a type (line 521)
        row_369722 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 521, 29), self_369721, 'row')
        # Getting the type of 'self' (line 521)
        self_369723 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 521, 39), 'self', False)
        # Obtaining the member 'col' of a type (line 521)
        col_369724 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 521, 39), self_369723, 'col')
        # Getting the type of 'self' (line 521)
        self_369725 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 521, 49), 'self', False)
        # Obtaining the member 'data' of a type (line 521)
        data_369726 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 521, 49), self_369725, 'data')
        # Getting the type of 'other' (line 521)
        other_369727 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 521, 60), 'other', False)
        # Getting the type of 'result' (line 521)
        result_369728 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 521, 67), 'result', False)
        # Processing the call keyword arguments (line 521)
        kwargs_369729 = {}
        # Getting the type of 'coo_matvec' (line 521)
        coo_matvec_369718 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 521, 8), 'coo_matvec', False)
        # Calling coo_matvec(args, kwargs) (line 521)
        coo_matvec_call_result_369730 = invoke(stypy.reporting.localization.Localization(__file__, 521, 8), coo_matvec_369718, *[nnz_369720, row_369722, col_369724, data_369726, other_369727, result_369728], **kwargs_369729)
        
        # Getting the type of 'result' (line 522)
        result_369731 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 522, 15), 'result')
        # Assigning a type to the variable 'stypy_return_type' (line 522)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 522, 8), 'stypy_return_type', result_369731)
        
        # ################# End of '_mul_vector(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_mul_vector' in the type store
        # Getting the type of 'stypy_return_type' (line 517)
        stypy_return_type_369732 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 517, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_369732)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_mul_vector'
        return stypy_return_type_369732


    @norecursion
    def _mul_multivector(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_mul_multivector'
        module_type_store = module_type_store.open_function_context('_mul_multivector', 524, 4, False)
        # Assigning a type to the variable 'self' (line 525)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 525, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        coo_matrix._mul_multivector.__dict__.__setitem__('stypy_localization', localization)
        coo_matrix._mul_multivector.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        coo_matrix._mul_multivector.__dict__.__setitem__('stypy_type_store', module_type_store)
        coo_matrix._mul_multivector.__dict__.__setitem__('stypy_function_name', 'coo_matrix._mul_multivector')
        coo_matrix._mul_multivector.__dict__.__setitem__('stypy_param_names_list', ['other'])
        coo_matrix._mul_multivector.__dict__.__setitem__('stypy_varargs_param_name', None)
        coo_matrix._mul_multivector.__dict__.__setitem__('stypy_kwargs_param_name', None)
        coo_matrix._mul_multivector.__dict__.__setitem__('stypy_call_defaults', defaults)
        coo_matrix._mul_multivector.__dict__.__setitem__('stypy_call_varargs', varargs)
        coo_matrix._mul_multivector.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        coo_matrix._mul_multivector.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'coo_matrix._mul_multivector', ['other'], None, None, defaults, varargs, kwargs)

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

        
        # Assigning a Call to a Name (line 525):
        
        # Assigning a Call to a Name (line 525):
        
        # Assigning a Call to a Name (line 525):
        
        # Call to zeros(...): (line 525)
        # Processing the call arguments (line 525)
        
        # Obtaining an instance of the builtin type 'tuple' (line 525)
        tuple_369735 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 525, 27), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 525)
        # Adding element type (line 525)
        
        # Obtaining the type of the subscript
        int_369736 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 525, 39), 'int')
        # Getting the type of 'other' (line 525)
        other_369737 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 525, 27), 'other', False)
        # Obtaining the member 'shape' of a type (line 525)
        shape_369738 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 525, 27), other_369737, 'shape')
        # Obtaining the member '__getitem__' of a type (line 525)
        getitem___369739 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 525, 27), shape_369738, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 525)
        subscript_call_result_369740 = invoke(stypy.reporting.localization.Localization(__file__, 525, 27), getitem___369739, int_369736)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 525, 27), tuple_369735, subscript_call_result_369740)
        # Adding element type (line 525)
        
        # Obtaining the type of the subscript
        int_369741 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 525, 54), 'int')
        # Getting the type of 'self' (line 525)
        self_369742 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 525, 43), 'self', False)
        # Obtaining the member 'shape' of a type (line 525)
        shape_369743 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 525, 43), self_369742, 'shape')
        # Obtaining the member '__getitem__' of a type (line 525)
        getitem___369744 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 525, 43), shape_369743, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 525)
        subscript_call_result_369745 = invoke(stypy.reporting.localization.Localization(__file__, 525, 43), getitem___369744, int_369741)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 525, 27), tuple_369735, subscript_call_result_369745)
        
        # Processing the call keyword arguments (line 525)
        
        # Call to upcast_char(...): (line 526)
        # Processing the call arguments (line 526)
        # Getting the type of 'self' (line 526)
        self_369747 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 526, 44), 'self', False)
        # Obtaining the member 'dtype' of a type (line 526)
        dtype_369748 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 526, 44), self_369747, 'dtype')
        # Obtaining the member 'char' of a type (line 526)
        char_369749 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 526, 44), dtype_369748, 'char')
        # Getting the type of 'other' (line 526)
        other_369750 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 526, 61), 'other', False)
        # Obtaining the member 'dtype' of a type (line 526)
        dtype_369751 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 526, 61), other_369750, 'dtype')
        # Obtaining the member 'char' of a type (line 526)
        char_369752 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 526, 61), dtype_369751, 'char')
        # Processing the call keyword arguments (line 526)
        kwargs_369753 = {}
        # Getting the type of 'upcast_char' (line 526)
        upcast_char_369746 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 526, 32), 'upcast_char', False)
        # Calling upcast_char(args, kwargs) (line 526)
        upcast_char_call_result_369754 = invoke(stypy.reporting.localization.Localization(__file__, 526, 32), upcast_char_369746, *[char_369749, char_369752], **kwargs_369753)
        
        keyword_369755 = upcast_char_call_result_369754
        kwargs_369756 = {'dtype': keyword_369755}
        # Getting the type of 'np' (line 525)
        np_369733 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 525, 17), 'np', False)
        # Obtaining the member 'zeros' of a type (line 525)
        zeros_369734 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 525, 17), np_369733, 'zeros')
        # Calling zeros(args, kwargs) (line 525)
        zeros_call_result_369757 = invoke(stypy.reporting.localization.Localization(__file__, 525, 17), zeros_369734, *[tuple_369735], **kwargs_369756)
        
        # Assigning a type to the variable 'result' (line 525)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 525, 8), 'result', zeros_call_result_369757)
        
        
        # Call to enumerate(...): (line 527)
        # Processing the call arguments (line 527)
        # Getting the type of 'other' (line 527)
        other_369759 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 527, 32), 'other', False)
        # Obtaining the member 'T' of a type (line 527)
        T_369760 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 527, 32), other_369759, 'T')
        # Processing the call keyword arguments (line 527)
        kwargs_369761 = {}
        # Getting the type of 'enumerate' (line 527)
        enumerate_369758 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 527, 22), 'enumerate', False)
        # Calling enumerate(args, kwargs) (line 527)
        enumerate_call_result_369762 = invoke(stypy.reporting.localization.Localization(__file__, 527, 22), enumerate_369758, *[T_369760], **kwargs_369761)
        
        # Testing the type of a for loop iterable (line 527)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 527, 8), enumerate_call_result_369762)
        # Getting the type of the for loop variable (line 527)
        for_loop_var_369763 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 527, 8), enumerate_call_result_369762)
        # Assigning a type to the variable 'i' (line 527)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 527, 8), 'i', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 527, 8), for_loop_var_369763))
        # Assigning a type to the variable 'col' (line 527)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 527, 8), 'col', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 527, 8), for_loop_var_369763))
        # SSA begins for a for statement (line 527)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Call to coo_matvec(...): (line 528)
        # Processing the call arguments (line 528)
        # Getting the type of 'self' (line 528)
        self_369765 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 528, 23), 'self', False)
        # Obtaining the member 'nnz' of a type (line 528)
        nnz_369766 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 528, 23), self_369765, 'nnz')
        # Getting the type of 'self' (line 528)
        self_369767 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 528, 33), 'self', False)
        # Obtaining the member 'row' of a type (line 528)
        row_369768 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 528, 33), self_369767, 'row')
        # Getting the type of 'self' (line 528)
        self_369769 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 528, 43), 'self', False)
        # Obtaining the member 'col' of a type (line 528)
        col_369770 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 528, 43), self_369769, 'col')
        # Getting the type of 'self' (line 528)
        self_369771 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 528, 53), 'self', False)
        # Obtaining the member 'data' of a type (line 528)
        data_369772 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 528, 53), self_369771, 'data')
        # Getting the type of 'col' (line 528)
        col_369773 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 528, 64), 'col', False)
        
        # Obtaining the type of the subscript
        # Getting the type of 'i' (line 528)
        i_369774 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 528, 76), 'i', False)
        # Getting the type of 'result' (line 528)
        result_369775 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 528, 69), 'result', False)
        # Obtaining the member '__getitem__' of a type (line 528)
        getitem___369776 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 528, 69), result_369775, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 528)
        subscript_call_result_369777 = invoke(stypy.reporting.localization.Localization(__file__, 528, 69), getitem___369776, i_369774)
        
        # Processing the call keyword arguments (line 528)
        kwargs_369778 = {}
        # Getting the type of 'coo_matvec' (line 528)
        coo_matvec_369764 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 528, 12), 'coo_matvec', False)
        # Calling coo_matvec(args, kwargs) (line 528)
        coo_matvec_call_result_369779 = invoke(stypy.reporting.localization.Localization(__file__, 528, 12), coo_matvec_369764, *[nnz_369766, row_369768, col_369770, data_369772, col_369773, subscript_call_result_369777], **kwargs_369778)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to view(...): (line 529)
        # Processing the call keyword arguments (line 529)
        
        # Call to type(...): (line 529)
        # Processing the call arguments (line 529)
        # Getting the type of 'other' (line 529)
        other_369784 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 529, 39), 'other', False)
        # Processing the call keyword arguments (line 529)
        kwargs_369785 = {}
        # Getting the type of 'type' (line 529)
        type_369783 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 529, 34), 'type', False)
        # Calling type(args, kwargs) (line 529)
        type_call_result_369786 = invoke(stypy.reporting.localization.Localization(__file__, 529, 34), type_369783, *[other_369784], **kwargs_369785)
        
        keyword_369787 = type_call_result_369786
        kwargs_369788 = {'type': keyword_369787}
        # Getting the type of 'result' (line 529)
        result_369780 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 529, 15), 'result', False)
        # Obtaining the member 'T' of a type (line 529)
        T_369781 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 529, 15), result_369780, 'T')
        # Obtaining the member 'view' of a type (line 529)
        view_369782 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 529, 15), T_369781, 'view')
        # Calling view(args, kwargs) (line 529)
        view_call_result_369789 = invoke(stypy.reporting.localization.Localization(__file__, 529, 15), view_369782, *[], **kwargs_369788)
        
        # Assigning a type to the variable 'stypy_return_type' (line 529)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 529, 8), 'stypy_return_type', view_call_result_369789)
        
        # ################# End of '_mul_multivector(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_mul_multivector' in the type store
        # Getting the type of 'stypy_return_type' (line 524)
        stypy_return_type_369790 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 524, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_369790)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_mul_multivector'
        return stypy_return_type_369790


# Assigning a type to the variable 'coo_matrix' (line 21)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 21, 0), 'coo_matrix', coo_matrix)

# Assigning a Str to a Name (line 123):
str_369791 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 123, 13), 'str', 'coo')
# Getting the type of 'coo_matrix'
coo_matrix_369792 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'coo_matrix')
# Setting the type of the member 'format' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), coo_matrix_369792, 'format', str_369791)

# Assigning a Attribute to a Attribute (line 217):
# Getting the type of 'spmatrix' (line 217)
spmatrix_369793 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 21), 'spmatrix')
# Obtaining the member 'getnnz' of a type (line 217)
getnnz_369794 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 217, 21), spmatrix_369793, 'getnnz')
# Obtaining the member '__doc__' of a type (line 217)
doc___369795 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 217, 21), getnnz_369794, '__doc__')
# Getting the type of 'coo_matrix'
coo_matrix_369796 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'coo_matrix')
# Obtaining the member 'getnnz' of a type
getnnz_369797 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), coo_matrix_369796, 'getnnz')
# Setting the type of the member '__doc__' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), getnnz_369797, '__doc__', doc___369795)

# Assigning a Attribute to a Attribute (line 255):
# Getting the type of 'spmatrix' (line 255)
spmatrix_369798 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 255, 24), 'spmatrix')
# Obtaining the member 'transpose' of a type (line 255)
transpose_369799 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 255, 24), spmatrix_369798, 'transpose')
# Obtaining the member '__doc__' of a type (line 255)
doc___369800 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 255, 24), transpose_369799, '__doc__')
# Getting the type of 'coo_matrix'
coo_matrix_369801 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'coo_matrix')
# Obtaining the member 'transpose' of a type
transpose_369802 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), coo_matrix_369801, 'transpose')
# Setting the type of the member '__doc__' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), transpose_369802, '__doc__', doc___369800)

# Assigning a Attribute to a Attribute (line 358):
# Getting the type of 'spmatrix' (line 358)
spmatrix_369803 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 358, 20), 'spmatrix')
# Obtaining the member 'tocoo' of a type (line 358)
tocoo_369804 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 358, 20), spmatrix_369803, 'tocoo')
# Obtaining the member '__doc__' of a type (line 358)
doc___369805 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 358, 20), tocoo_369804, '__doc__')
# Getting the type of 'coo_matrix'
coo_matrix_369806 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'coo_matrix')
# Obtaining the member 'tocoo' of a type
tocoo_369807 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), coo_matrix_369806, 'tocoo')
# Setting the type of the member '__doc__' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), tocoo_369807, '__doc__', doc___369805)

# Assigning a Attribute to a Attribute (line 381):
# Getting the type of 'spmatrix' (line 381)
spmatrix_369808 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 381, 20), 'spmatrix')
# Obtaining the member 'todia' of a type (line 381)
todia_369809 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 381, 20), spmatrix_369808, 'todia')
# Obtaining the member '__doc__' of a type (line 381)
doc___369810 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 381, 20), todia_369809, '__doc__')
# Getting the type of 'coo_matrix'
coo_matrix_369811 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'coo_matrix')
# Obtaining the member 'todia' of a type
todia_369812 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), coo_matrix_369811, 'todia')
# Setting the type of the member '__doc__' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), todia_369812, '__doc__', doc___369810)

# Assigning a Attribute to a Attribute (line 392):
# Getting the type of 'spmatrix' (line 392)
spmatrix_369813 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 392, 20), 'spmatrix')
# Obtaining the member 'todok' of a type (line 392)
todok_369814 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 392, 20), spmatrix_369813, 'todok')
# Obtaining the member '__doc__' of a type (line 392)
doc___369815 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 392, 20), todok_369814, '__doc__')
# Getting the type of 'coo_matrix'
coo_matrix_369816 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'coo_matrix')
# Obtaining the member 'todok' of a type
todok_369817 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), coo_matrix_369816, 'todok')
# Setting the type of the member '__doc__' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), todok_369817, '__doc__', doc___369815)

# Assigning a Attribute to a Attribute (line 413):
# Getting the type of '_data_matrix' (line 413)
_data_matrix_369818 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 413, 23), '_data_matrix')
# Obtaining the member 'diagonal' of a type (line 413)
diagonal_369819 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 413, 23), _data_matrix_369818, 'diagonal')
# Obtaining the member '__doc__' of a type (line 413)
doc___369820 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 413, 23), diagonal_369819, '__doc__')
# Getting the type of 'coo_matrix'
coo_matrix_369821 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'coo_matrix')
# Obtaining the member 'diagonal' of a type
diagonal_369822 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), coo_matrix_369821, 'diagonal')
# Setting the type of the member '__doc__' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), diagonal_369822, '__doc__', doc___369820)

@norecursion
def isspmatrix_coo(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'isspmatrix_coo'
    module_type_store = module_type_store.open_function_context('isspmatrix_coo', 532, 0, False)
    
    # Passed parameters checking function
    isspmatrix_coo.stypy_localization = localization
    isspmatrix_coo.stypy_type_of_self = None
    isspmatrix_coo.stypy_type_store = module_type_store
    isspmatrix_coo.stypy_function_name = 'isspmatrix_coo'
    isspmatrix_coo.stypy_param_names_list = ['x']
    isspmatrix_coo.stypy_varargs_param_name = None
    isspmatrix_coo.stypy_kwargs_param_name = None
    isspmatrix_coo.stypy_call_defaults = defaults
    isspmatrix_coo.stypy_call_varargs = varargs
    isspmatrix_coo.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'isspmatrix_coo', ['x'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'isspmatrix_coo', localization, ['x'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'isspmatrix_coo(...)' code ##################

    str_369823 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 554, (-1)), 'str', 'Is x of coo_matrix type?\n\n    Parameters\n    ----------\n    x\n        object to check for being a coo matrix\n\n    Returns\n    -------\n    bool\n        True if x is a coo matrix, False otherwise\n\n    Examples\n    --------\n    >>> from scipy.sparse import coo_matrix, isspmatrix_coo\n    >>> isspmatrix_coo(coo_matrix([[5]]))\n    True\n\n    >>> from scipy.sparse import coo_matrix, csr_matrix, isspmatrix_coo\n    >>> isspmatrix_coo(csr_matrix([[5]]))\n    False\n    ')
    
    # Call to isinstance(...): (line 555)
    # Processing the call arguments (line 555)
    # Getting the type of 'x' (line 555)
    x_369825 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 555, 22), 'x', False)
    # Getting the type of 'coo_matrix' (line 555)
    coo_matrix_369826 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 555, 25), 'coo_matrix', False)
    # Processing the call keyword arguments (line 555)
    kwargs_369827 = {}
    # Getting the type of 'isinstance' (line 555)
    isinstance_369824 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 555, 11), 'isinstance', False)
    # Calling isinstance(args, kwargs) (line 555)
    isinstance_call_result_369828 = invoke(stypy.reporting.localization.Localization(__file__, 555, 11), isinstance_369824, *[x_369825, coo_matrix_369826], **kwargs_369827)
    
    # Assigning a type to the variable 'stypy_return_type' (line 555)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 555, 4), 'stypy_return_type', isinstance_call_result_369828)
    
    # ################# End of 'isspmatrix_coo(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'isspmatrix_coo' in the type store
    # Getting the type of 'stypy_return_type' (line 532)
    stypy_return_type_369829 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 532, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_369829)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'isspmatrix_coo'
    return stypy_return_type_369829

# Assigning a type to the variable 'isspmatrix_coo' (line 532)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 532, 0), 'isspmatrix_coo', isspmatrix_coo)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
