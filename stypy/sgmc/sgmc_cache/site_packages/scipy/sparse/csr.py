
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: '''Compressed Sparse Row matrix format'''
2: 
3: from __future__ import division, print_function, absolute_import
4: 
5: __docformat__ = "restructuredtext en"
6: 
7: __all__ = ['csr_matrix', 'isspmatrix_csr']
8: 
9: 
10: import numpy as np
11: from scipy._lib.six import xrange
12: 
13: from .base import spmatrix
14: 
15: from ._sparsetools import csr_tocsc, csr_tobsr, csr_count_blocks, \
16:         get_csr_submatrix, csr_sample_values
17: from .sputils import (upcast, isintlike, IndexMixin, issequence,
18:                       get_index_dtype, ismatrix)
19: 
20: from .compressed import _cs_matrix
21: 
22: 
23: class csr_matrix(_cs_matrix, IndexMixin):
24:     '''
25:     Compressed Sparse Row matrix
26: 
27:     This can be instantiated in several ways:
28:         csr_matrix(D)
29:             with a dense matrix or rank-2 ndarray D
30: 
31:         csr_matrix(S)
32:             with another sparse matrix S (equivalent to S.tocsr())
33: 
34:         csr_matrix((M, N), [dtype])
35:             to construct an empty matrix with shape (M, N)
36:             dtype is optional, defaulting to dtype='d'.
37: 
38:         csr_matrix((data, (row_ind, col_ind)), [shape=(M, N)])
39:             where ``data``, ``row_ind`` and ``col_ind`` satisfy the
40:             relationship ``a[row_ind[k], col_ind[k]] = data[k]``.
41: 
42:         csr_matrix((data, indices, indptr), [shape=(M, N)])
43:             is the standard CSR representation where the column indices for
44:             row i are stored in ``indices[indptr[i]:indptr[i+1]]`` and their
45:             corresponding values are stored in ``data[indptr[i]:indptr[i+1]]``.
46:             If the shape parameter is not supplied, the matrix dimensions
47:             are inferred from the index arrays.
48: 
49:     Attributes
50:     ----------
51:     dtype : dtype
52:         Data type of the matrix
53:     shape : 2-tuple
54:         Shape of the matrix
55:     ndim : int
56:         Number of dimensions (this is always 2)
57:     nnz
58:         Number of nonzero elements
59:     data
60:         CSR format data array of the matrix
61:     indices
62:         CSR format index array of the matrix
63:     indptr
64:         CSR format index pointer array of the matrix
65:     has_sorted_indices
66:         Whether indices are sorted
67: 
68:     Notes
69:     -----
70: 
71:     Sparse matrices can be used in arithmetic operations: they support
72:     addition, subtraction, multiplication, division, and matrix power.
73: 
74:     Advantages of the CSR format
75:       - efficient arithmetic operations CSR + CSR, CSR * CSR, etc.
76:       - efficient row slicing
77:       - fast matrix vector products
78: 
79:     Disadvantages of the CSR format
80:       - slow column slicing operations (consider CSC)
81:       - changes to the sparsity structure are expensive (consider LIL or DOK)
82: 
83:     Examples
84:     --------
85: 
86:     >>> import numpy as np
87:     >>> from scipy.sparse import csr_matrix
88:     >>> csr_matrix((3, 4), dtype=np.int8).toarray()
89:     array([[0, 0, 0, 0],
90:            [0, 0, 0, 0],
91:            [0, 0, 0, 0]], dtype=int8)
92: 
93:     >>> row = np.array([0, 0, 1, 2, 2, 2])
94:     >>> col = np.array([0, 2, 2, 0, 1, 2])
95:     >>> data = np.array([1, 2, 3, 4, 5, 6])
96:     >>> csr_matrix((data, (row, col)), shape=(3, 3)).toarray()
97:     array([[1, 0, 2],
98:            [0, 0, 3],
99:            [4, 5, 6]])
100: 
101:     >>> indptr = np.array([0, 2, 3, 6])
102:     >>> indices = np.array([0, 2, 2, 0, 1, 2])
103:     >>> data = np.array([1, 2, 3, 4, 5, 6])
104:     >>> csr_matrix((data, indices, indptr), shape=(3, 3)).toarray()
105:     array([[1, 0, 2],
106:            [0, 0, 3],
107:            [4, 5, 6]])
108: 
109:     As an example of how to construct a CSR matrix incrementally,
110:     the following snippet builds a term-document matrix from texts:
111: 
112:     >>> docs = [["hello", "world", "hello"], ["goodbye", "cruel", "world"]]
113:     >>> indptr = [0]
114:     >>> indices = []
115:     >>> data = []
116:     >>> vocabulary = {}
117:     >>> for d in docs:
118:     ...     for term in d:
119:     ...         index = vocabulary.setdefault(term, len(vocabulary))
120:     ...         indices.append(index)
121:     ...         data.append(1)
122:     ...     indptr.append(len(indices))
123:     ...
124:     >>> csr_matrix((data, indices, indptr), dtype=int).toarray()
125:     array([[2, 1, 0, 0],
126:            [0, 1, 1, 1]])
127: 
128:     '''
129:     format = 'csr'
130: 
131:     def transpose(self, axes=None, copy=False):
132:         if axes is not None:
133:             raise ValueError(("Sparse matrices do not support "
134:                               "an 'axes' parameter because swapping "
135:                               "dimensions is the only logical permutation."))
136: 
137:         M, N = self.shape
138: 
139:         from .csc import csc_matrix
140:         return csc_matrix((self.data, self.indices,
141:                            self.indptr), shape=(N, M), copy=copy)
142: 
143:     transpose.__doc__ = spmatrix.transpose.__doc__
144: 
145:     def tolil(self, copy=False):
146:         from .lil import lil_matrix
147:         lil = lil_matrix(self.shape,dtype=self.dtype)
148: 
149:         self.sum_duplicates()
150:         ptr,ind,dat = self.indptr,self.indices,self.data
151:         rows, data = lil.rows, lil.data
152: 
153:         for n in xrange(self.shape[0]):
154:             start = ptr[n]
155:             end = ptr[n+1]
156:             rows[n] = ind[start:end].tolist()
157:             data[n] = dat[start:end].tolist()
158: 
159:         return lil
160: 
161:     tolil.__doc__ = spmatrix.tolil.__doc__
162: 
163:     def tocsr(self, copy=False):
164:         if copy:
165:             return self.copy()
166:         else:
167:             return self
168: 
169:     tocsr.__doc__ = spmatrix.tocsr.__doc__
170: 
171:     def tocsc(self, copy=False):
172:         idx_dtype = get_index_dtype((self.indptr, self.indices),
173:                                     maxval=max(self.nnz, self.shape[0]))
174:         indptr = np.empty(self.shape[1] + 1, dtype=idx_dtype)
175:         indices = np.empty(self.nnz, dtype=idx_dtype)
176:         data = np.empty(self.nnz, dtype=upcast(self.dtype))
177: 
178:         csr_tocsc(self.shape[0], self.shape[1],
179:                   self.indptr.astype(idx_dtype),
180:                   self.indices.astype(idx_dtype),
181:                   self.data,
182:                   indptr,
183:                   indices,
184:                   data)
185: 
186:         from .csc import csc_matrix
187:         A = csc_matrix((data, indices, indptr), shape=self.shape)
188:         A.has_sorted_indices = True
189:         return A
190: 
191:     tocsc.__doc__ = spmatrix.tocsc.__doc__
192: 
193:     def tobsr(self, blocksize=None, copy=True):
194:         from .bsr import bsr_matrix
195: 
196:         if blocksize is None:
197:             from .spfuncs import estimate_blocksize
198:             return self.tobsr(blocksize=estimate_blocksize(self))
199: 
200:         elif blocksize == (1,1):
201:             arg1 = (self.data.reshape(-1,1,1),self.indices,self.indptr)
202:             return bsr_matrix(arg1, shape=self.shape, copy=copy)
203: 
204:         else:
205:             R,C = blocksize
206:             M,N = self.shape
207: 
208:             if R < 1 or C < 1 or M % R != 0 or N % C != 0:
209:                 raise ValueError('invalid blocksize %s' % blocksize)
210: 
211:             blks = csr_count_blocks(M,N,R,C,self.indptr,self.indices)
212: 
213:             idx_dtype = get_index_dtype((self.indptr, self.indices),
214:                                         maxval=max(N//C, blks))
215:             indptr = np.empty(M//R+1, dtype=idx_dtype)
216:             indices = np.empty(blks, dtype=idx_dtype)
217:             data = np.zeros((blks,R,C), dtype=self.dtype)
218: 
219:             csr_tobsr(M, N, R, C,
220:                       self.indptr.astype(idx_dtype),
221:                       self.indices.astype(idx_dtype),
222:                       self.data,
223:                       indptr, indices, data.ravel())
224: 
225:             return bsr_matrix((data,indices,indptr), shape=self.shape)
226: 
227:     tobsr.__doc__ = spmatrix.tobsr.__doc__
228: 
229:     # these functions are used by the parent class (_cs_matrix)
230:     # to remove redudancy between csc_matrix and csr_matrix
231:     def _swap(self, x):
232:         '''swap the members of x if this is a column-oriented matrix
233:         '''
234:         return x
235: 
236:     def __getitem__(self, key):
237:         def asindices(x):
238:             try:
239:                 x = np.asarray(x)
240: 
241:                 # Check index contents to avoid creating 64bit arrays needlessly
242:                 idx_dtype = get_index_dtype((x,), check_contents=True)
243:                 if idx_dtype != x.dtype:
244:                     x = x.astype(idx_dtype)
245:             except:
246:                 raise IndexError('invalid index')
247:             else:
248:                 return x
249: 
250:         def check_bounds(indices, N):
251:             if indices.size == 0:
252:                 return (0, 0)
253: 
254:             max_indx = indices.max()
255:             if max_indx >= N:
256:                 raise IndexError('index (%d) out of range' % max_indx)
257: 
258:             min_indx = indices.min()
259:             if min_indx < -N:
260:                 raise IndexError('index (%d) out of range' % (N + min_indx))
261: 
262:             return min_indx, max_indx
263: 
264:         def extractor(indices,N):
265:             '''Return a sparse matrix P so that P*self implements
266:             slicing of the form self[[1,2,3],:]
267:             '''
268:             indices = asindices(indices).copy()
269: 
270:             min_indx, max_indx = check_bounds(indices, N)
271: 
272:             if min_indx < 0:
273:                 indices[indices < 0] += N
274: 
275:             indptr = np.arange(len(indices)+1, dtype=indices.dtype)
276:             data = np.ones(len(indices), dtype=self.dtype)
277:             shape = (len(indices),N)
278: 
279:             return csr_matrix((data,indices,indptr), shape=shape,
280:                               dtype=self.dtype, copy=False)
281: 
282:         row, col = self._unpack_index(key)
283: 
284:         # First attempt to use original row optimized methods
285:         # [1, ?]
286:         if isintlike(row):
287:             # [i, j]
288:             if isintlike(col):
289:                 return self._get_single_element(row, col)
290:             # [i, 1:2]
291:             elif isinstance(col, slice):
292:                 return self._get_row_slice(row, col)
293:             # [i, [1, 2]]
294:             elif issequence(col):
295:                 P = extractor(col,self.shape[1]).T
296:                 return self[row, :] * P
297:         elif isinstance(row, slice):
298:             # [1:2,??]
299:             if ((isintlike(col) and row.step in (1, None)) or
300:                     (isinstance(col, slice) and
301:                      col.step in (1, None) and
302:                      row.step in (1, None))):
303:                 # col is int or slice with step 1, row is slice with step 1.
304:                 return self._get_submatrix(row, col)
305:             elif issequence(col):
306:                 # row is slice, col is sequence.
307:                 P = extractor(col,self.shape[1]).T        # [1:2,[1,2]]
308:                 sliced = self
309:                 if row != slice(None, None, None):
310:                     sliced = sliced[row,:]
311:                 return sliced * P
312: 
313:         elif issequence(row):
314:             # [[1,2],??]
315:             if isintlike(col) or isinstance(col,slice):
316:                 P = extractor(row, self.shape[0])     # [[1,2],j] or [[1,2],1:2]
317:                 extracted = P * self
318:                 if col == slice(None, None, None):
319:                     return extracted
320:                 else:
321:                     return extracted[:,col]
322: 
323:         elif ismatrix(row) and issequence(col):
324:             if len(row[0]) == 1 and isintlike(row[0][0]):
325:                 # [[[1],[2]], [1,2]], outer indexing
326:                 row = asindices(row)
327:                 P_row = extractor(row[:,0], self.shape[0])
328:                 P_col = extractor(col, self.shape[1]).T
329:                 return P_row * self * P_col
330: 
331:         if not (issequence(col) and issequence(row)):
332:             # Sample elementwise
333:             row, col = self._index_to_arrays(row, col)
334: 
335:         row = asindices(row)
336:         col = asindices(col)
337:         if row.shape != col.shape:
338:             raise IndexError('number of row and column indices differ')
339:         assert row.ndim <= 2
340: 
341:         num_samples = np.size(row)
342:         if num_samples == 0:
343:             return csr_matrix(np.atleast_2d(row).shape, dtype=self.dtype)
344:         check_bounds(row, self.shape[0])
345:         check_bounds(col, self.shape[1])
346: 
347:         val = np.empty(num_samples, dtype=self.dtype)
348:         csr_sample_values(self.shape[0], self.shape[1],
349:                           self.indptr, self.indices, self.data,
350:                           num_samples, row.ravel(), col.ravel(), val)
351:         if row.ndim == 1:
352:             # row and col are 1d
353:             return np.asmatrix(val)
354:         return self.__class__(val.reshape(row.shape))
355: 
356:     def __iter__(self):
357:         indptr = np.zeros(2, dtype=self.indptr.dtype)
358:         shape = (1, self.shape[1])
359:         i0 = 0
360:         for i1 in self.indptr[1:]:
361:             indptr[1] = i1 - i0
362:             indices = self.indices[i0:i1]
363:             data = self.data[i0:i1]
364:             yield csr_matrix((data, indices, indptr), shape=shape, copy=True)
365:             i0 = i1
366: 
367:     def getrow(self, i):
368:         '''Returns a copy of row i of the matrix, as a (1 x n)
369:         CSR matrix (row vector).
370:         '''
371:         M, N = self.shape
372:         i = int(i)
373:         if i < 0:
374:             i += M
375:         if i < 0 or i >= M:
376:             raise IndexError('index (%d) out of range' % i)
377:         idx = slice(*self.indptr[i:i+2])
378:         data = self.data[idx].copy()
379:         indices = self.indices[idx].copy()
380:         indptr = np.array([0, len(indices)], dtype=self.indptr.dtype)
381:         return csr_matrix((data, indices, indptr), shape=(1, N),
382:                           dtype=self.dtype, copy=False)
383: 
384:     def getcol(self, i):
385:         '''Returns a copy of column i of the matrix, as a (m x 1)
386:         CSR matrix (column vector).
387:         '''
388:         return self._get_submatrix(slice(None), i)
389: 
390:     def _get_row_slice(self, i, cslice):
391:         '''Returns a copy of row self[i, cslice]
392:         '''
393:         M, N = self.shape
394: 
395:         if i < 0:
396:             i += M
397: 
398:         if i < 0 or i >= M:
399:             raise IndexError('index (%d) out of range' % i)
400: 
401:         start, stop, stride = cslice.indices(N)
402: 
403:         if stride == 1:
404:             # for stride == 1, get_csr_submatrix is faster
405:             row_indptr, row_indices, row_data = get_csr_submatrix(
406:                 M, N, self.indptr, self.indices, self.data, i, i + 1,
407:                 start, stop)
408:         else:
409:             # other strides need new code
410:             row_indices = self.indices[self.indptr[i]:self.indptr[i + 1]]
411:             row_data = self.data[self.indptr[i]:self.indptr[i + 1]]
412: 
413:             if stride > 0:
414:                 ind = (row_indices >= start) & (row_indices < stop)
415:             else:
416:                 ind = (row_indices <= start) & (row_indices > stop)
417: 
418:             if abs(stride) > 1:
419:                 ind &= (row_indices - start) % stride == 0
420: 
421:             row_indices = (row_indices[ind] - start) // stride
422:             row_data = row_data[ind]
423:             row_indptr = np.array([0, len(row_indices)])
424: 
425:             if stride < 0:
426:                 row_data = row_data[::-1]
427:                 row_indices = abs(row_indices[::-1])
428: 
429:         shape = (1, int(np.ceil(float(stop - start) / stride)))
430:         return csr_matrix((row_data, row_indices, row_indptr), shape=shape,
431:                           dtype=self.dtype, copy=False)
432: 
433:     def _get_submatrix(self, row_slice, col_slice):
434:         '''Return a submatrix of this matrix (new matrix is created).'''
435: 
436:         def process_slice(sl, num):
437:             if isinstance(sl, slice):
438:                 i0, i1, stride = sl.indices(num)
439:                 if stride != 1:
440:                     raise ValueError('slicing with step != 1 not supported')
441:             elif isintlike(sl):
442:                 if sl < 0:
443:                     sl += num
444:                 i0, i1 = sl, sl + 1
445:             else:
446:                 raise TypeError('expected slice or scalar')
447: 
448:             if not (0 <= i0 <= num) or not (0 <= i1 <= num) or not (i0 <= i1):
449:                 raise IndexError(
450:                       "index out of bounds: 0 <= %d <= %d, 0 <= %d <= %d,"
451:                       " %d <= %d" % (i0, num, i1, num, i0, i1))
452:             return i0, i1
453: 
454:         M,N = self.shape
455:         i0, i1 = process_slice(row_slice, M)
456:         j0, j1 = process_slice(col_slice, N)
457: 
458:         indptr, indices, data = get_csr_submatrix(
459:             M, N, self.indptr, self.indices, self.data, i0, i1, j0, j1)
460: 
461:         shape = (i1 - i0, j1 - j0)
462:         return self.__class__((data, indices, indptr), shape=shape,
463:                               dtype=self.dtype, copy=False)
464: 
465: 
466: def isspmatrix_csr(x):
467:     '''Is x of csr_matrix type?
468: 
469:     Parameters
470:     ----------
471:     x
472:         object to check for being a csr matrix
473: 
474:     Returns
475:     -------
476:     bool
477:         True if x is a csr matrix, False otherwise
478: 
479:     Examples
480:     --------
481:     >>> from scipy.sparse import csr_matrix, isspmatrix_csr
482:     >>> isspmatrix_csr(csr_matrix([[5]]))
483:     True
484: 
485:     >>> from scipy.sparse import csc_matrix, csr_matrix, isspmatrix_csc
486:     >>> isspmatrix_csr(csc_matrix([[5]]))
487:     False
488:     '''
489:     return isinstance(x, csr_matrix)
490: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_370359 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1, 0), 'str', 'Compressed Sparse Row matrix format')

# Assigning a Str to a Name (line 5):

# Assigning a Str to a Name (line 5):
str_370360 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 16), 'str', 'restructuredtext en')
# Assigning a type to the variable '__docformat__' (line 5)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 0), '__docformat__', str_370360)

# Assigning a List to a Name (line 7):

# Assigning a List to a Name (line 7):
__all__ = ['csr_matrix', 'isspmatrix_csr']
module_type_store.set_exportable_members(['csr_matrix', 'isspmatrix_csr'])

# Obtaining an instance of the builtin type 'list' (line 7)
list_370361 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 10), 'list')
# Adding type elements to the builtin type 'list' instance (line 7)
# Adding element type (line 7)
str_370362 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 11), 'str', 'csr_matrix')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 7, 10), list_370361, str_370362)
# Adding element type (line 7)
str_370363 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 25), 'str', 'isspmatrix_csr')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 7, 10), list_370361, str_370363)

# Assigning a type to the variable '__all__' (line 7)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), '__all__', list_370361)
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 10, 0))

# 'import numpy' statement (line 10)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/sparse/')
import_370364 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'numpy')

if (type(import_370364) is not StypyTypeError):

    if (import_370364 != 'pyd_module'):
        __import__(import_370364)
        sys_modules_370365 = sys.modules[import_370364]
        import_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'np', sys_modules_370365.module_type_store, module_type_store)
    else:
        import numpy as np

        import_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'np', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 10)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 0), 'numpy', import_370364)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/sparse/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 11, 0))

# 'from scipy._lib.six import xrange' statement (line 11)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/sparse/')
import_370366 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'scipy._lib.six')

if (type(import_370366) is not StypyTypeError):

    if (import_370366 != 'pyd_module'):
        __import__(import_370366)
        sys_modules_370367 = sys.modules[import_370366]
        import_from_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'scipy._lib.six', sys_modules_370367.module_type_store, module_type_store, ['xrange'])
        nest_module(stypy.reporting.localization.Localization(__file__, 11, 0), __file__, sys_modules_370367, sys_modules_370367.module_type_store, module_type_store)
    else:
        from scipy._lib.six import xrange

        import_from_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'scipy._lib.six', None, module_type_store, ['xrange'], [xrange])

else:
    # Assigning a type to the variable 'scipy._lib.six' (line 11)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 0), 'scipy._lib.six', import_370366)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/sparse/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 13, 0))

# 'from scipy.sparse.base import spmatrix' statement (line 13)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/sparse/')
import_370368 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 13, 0), 'scipy.sparse.base')

if (type(import_370368) is not StypyTypeError):

    if (import_370368 != 'pyd_module'):
        __import__(import_370368)
        sys_modules_370369 = sys.modules[import_370368]
        import_from_module(stypy.reporting.localization.Localization(__file__, 13, 0), 'scipy.sparse.base', sys_modules_370369.module_type_store, module_type_store, ['spmatrix'])
        nest_module(stypy.reporting.localization.Localization(__file__, 13, 0), __file__, sys_modules_370369, sys_modules_370369.module_type_store, module_type_store)
    else:
        from scipy.sparse.base import spmatrix

        import_from_module(stypy.reporting.localization.Localization(__file__, 13, 0), 'scipy.sparse.base', None, module_type_store, ['spmatrix'], [spmatrix])

else:
    # Assigning a type to the variable 'scipy.sparse.base' (line 13)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 0), 'scipy.sparse.base', import_370368)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/sparse/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 15, 0))

# 'from scipy.sparse._sparsetools import csr_tocsc, csr_tobsr, csr_count_blocks, get_csr_submatrix, csr_sample_values' statement (line 15)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/sparse/')
import_370370 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 15, 0), 'scipy.sparse._sparsetools')

if (type(import_370370) is not StypyTypeError):

    if (import_370370 != 'pyd_module'):
        __import__(import_370370)
        sys_modules_370371 = sys.modules[import_370370]
        import_from_module(stypy.reporting.localization.Localization(__file__, 15, 0), 'scipy.sparse._sparsetools', sys_modules_370371.module_type_store, module_type_store, ['csr_tocsc', 'csr_tobsr', 'csr_count_blocks', 'get_csr_submatrix', 'csr_sample_values'])
        nest_module(stypy.reporting.localization.Localization(__file__, 15, 0), __file__, sys_modules_370371, sys_modules_370371.module_type_store, module_type_store)
    else:
        from scipy.sparse._sparsetools import csr_tocsc, csr_tobsr, csr_count_blocks, get_csr_submatrix, csr_sample_values

        import_from_module(stypy.reporting.localization.Localization(__file__, 15, 0), 'scipy.sparse._sparsetools', None, module_type_store, ['csr_tocsc', 'csr_tobsr', 'csr_count_blocks', 'get_csr_submatrix', 'csr_sample_values'], [csr_tocsc, csr_tobsr, csr_count_blocks, get_csr_submatrix, csr_sample_values])

else:
    # Assigning a type to the variable 'scipy.sparse._sparsetools' (line 15)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 0), 'scipy.sparse._sparsetools', import_370370)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/sparse/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 17, 0))

# 'from scipy.sparse.sputils import upcast, isintlike, IndexMixin, issequence, get_index_dtype, ismatrix' statement (line 17)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/sparse/')
import_370372 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 17, 0), 'scipy.sparse.sputils')

if (type(import_370372) is not StypyTypeError):

    if (import_370372 != 'pyd_module'):
        __import__(import_370372)
        sys_modules_370373 = sys.modules[import_370372]
        import_from_module(stypy.reporting.localization.Localization(__file__, 17, 0), 'scipy.sparse.sputils', sys_modules_370373.module_type_store, module_type_store, ['upcast', 'isintlike', 'IndexMixin', 'issequence', 'get_index_dtype', 'ismatrix'])
        nest_module(stypy.reporting.localization.Localization(__file__, 17, 0), __file__, sys_modules_370373, sys_modules_370373.module_type_store, module_type_store)
    else:
        from scipy.sparse.sputils import upcast, isintlike, IndexMixin, issequence, get_index_dtype, ismatrix

        import_from_module(stypy.reporting.localization.Localization(__file__, 17, 0), 'scipy.sparse.sputils', None, module_type_store, ['upcast', 'isintlike', 'IndexMixin', 'issequence', 'get_index_dtype', 'ismatrix'], [upcast, isintlike, IndexMixin, issequence, get_index_dtype, ismatrix])

else:
    # Assigning a type to the variable 'scipy.sparse.sputils' (line 17)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 17, 0), 'scipy.sparse.sputils', import_370372)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/sparse/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 20, 0))

# 'from scipy.sparse.compressed import _cs_matrix' statement (line 20)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/sparse/')
import_370374 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 20, 0), 'scipy.sparse.compressed')

if (type(import_370374) is not StypyTypeError):

    if (import_370374 != 'pyd_module'):
        __import__(import_370374)
        sys_modules_370375 = sys.modules[import_370374]
        import_from_module(stypy.reporting.localization.Localization(__file__, 20, 0), 'scipy.sparse.compressed', sys_modules_370375.module_type_store, module_type_store, ['_cs_matrix'])
        nest_module(stypy.reporting.localization.Localization(__file__, 20, 0), __file__, sys_modules_370375, sys_modules_370375.module_type_store, module_type_store)
    else:
        from scipy.sparse.compressed import _cs_matrix

        import_from_module(stypy.reporting.localization.Localization(__file__, 20, 0), 'scipy.sparse.compressed', None, module_type_store, ['_cs_matrix'], [_cs_matrix])

else:
    # Assigning a type to the variable 'scipy.sparse.compressed' (line 20)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 20, 0), 'scipy.sparse.compressed', import_370374)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/sparse/')

# Declaration of the 'csr_matrix' class
# Getting the type of '_cs_matrix' (line 23)
_cs_matrix_370376 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 17), '_cs_matrix')
# Getting the type of 'IndexMixin' (line 23)
IndexMixin_370377 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 29), 'IndexMixin')

class csr_matrix(_cs_matrix_370376, IndexMixin_370377, ):
    str_370378 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 128, (-1)), 'str', '\n    Compressed Sparse Row matrix\n\n    This can be instantiated in several ways:\n        csr_matrix(D)\n            with a dense matrix or rank-2 ndarray D\n\n        csr_matrix(S)\n            with another sparse matrix S (equivalent to S.tocsr())\n\n        csr_matrix((M, N), [dtype])\n            to construct an empty matrix with shape (M, N)\n            dtype is optional, defaulting to dtype=\'d\'.\n\n        csr_matrix((data, (row_ind, col_ind)), [shape=(M, N)])\n            where ``data``, ``row_ind`` and ``col_ind`` satisfy the\n            relationship ``a[row_ind[k], col_ind[k]] = data[k]``.\n\n        csr_matrix((data, indices, indptr), [shape=(M, N)])\n            is the standard CSR representation where the column indices for\n            row i are stored in ``indices[indptr[i]:indptr[i+1]]`` and their\n            corresponding values are stored in ``data[indptr[i]:indptr[i+1]]``.\n            If the shape parameter is not supplied, the matrix dimensions\n            are inferred from the index arrays.\n\n    Attributes\n    ----------\n    dtype : dtype\n        Data type of the matrix\n    shape : 2-tuple\n        Shape of the matrix\n    ndim : int\n        Number of dimensions (this is always 2)\n    nnz\n        Number of nonzero elements\n    data\n        CSR format data array of the matrix\n    indices\n        CSR format index array of the matrix\n    indptr\n        CSR format index pointer array of the matrix\n    has_sorted_indices\n        Whether indices are sorted\n\n    Notes\n    -----\n\n    Sparse matrices can be used in arithmetic operations: they support\n    addition, subtraction, multiplication, division, and matrix power.\n\n    Advantages of the CSR format\n      - efficient arithmetic operations CSR + CSR, CSR * CSR, etc.\n      - efficient row slicing\n      - fast matrix vector products\n\n    Disadvantages of the CSR format\n      - slow column slicing operations (consider CSC)\n      - changes to the sparsity structure are expensive (consider LIL or DOK)\n\n    Examples\n    --------\n\n    >>> import numpy as np\n    >>> from scipy.sparse import csr_matrix\n    >>> csr_matrix((3, 4), dtype=np.int8).toarray()\n    array([[0, 0, 0, 0],\n           [0, 0, 0, 0],\n           [0, 0, 0, 0]], dtype=int8)\n\n    >>> row = np.array([0, 0, 1, 2, 2, 2])\n    >>> col = np.array([0, 2, 2, 0, 1, 2])\n    >>> data = np.array([1, 2, 3, 4, 5, 6])\n    >>> csr_matrix((data, (row, col)), shape=(3, 3)).toarray()\n    array([[1, 0, 2],\n           [0, 0, 3],\n           [4, 5, 6]])\n\n    >>> indptr = np.array([0, 2, 3, 6])\n    >>> indices = np.array([0, 2, 2, 0, 1, 2])\n    >>> data = np.array([1, 2, 3, 4, 5, 6])\n    >>> csr_matrix((data, indices, indptr), shape=(3, 3)).toarray()\n    array([[1, 0, 2],\n           [0, 0, 3],\n           [4, 5, 6]])\n\n    As an example of how to construct a CSR matrix incrementally,\n    the following snippet builds a term-document matrix from texts:\n\n    >>> docs = [["hello", "world", "hello"], ["goodbye", "cruel", "world"]]\n    >>> indptr = [0]\n    >>> indices = []\n    >>> data = []\n    >>> vocabulary = {}\n    >>> for d in docs:\n    ...     for term in d:\n    ...         index = vocabulary.setdefault(term, len(vocabulary))\n    ...         indices.append(index)\n    ...         data.append(1)\n    ...     indptr.append(len(indices))\n    ...\n    >>> csr_matrix((data, indices, indptr), dtype=int).toarray()\n    array([[2, 1, 0, 0],\n           [0, 1, 1, 1]])\n\n    ')
    
    # Assigning a Str to a Name (line 129):

    @norecursion
    def transpose(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 131)
        None_370379 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 29), 'None')
        # Getting the type of 'False' (line 131)
        False_370380 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 40), 'False')
        defaults = [None_370379, False_370380]
        # Create a new context for function 'transpose'
        module_type_store = module_type_store.open_function_context('transpose', 131, 4, False)
        # Assigning a type to the variable 'self' (line 132)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 132, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        csr_matrix.transpose.__dict__.__setitem__('stypy_localization', localization)
        csr_matrix.transpose.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        csr_matrix.transpose.__dict__.__setitem__('stypy_type_store', module_type_store)
        csr_matrix.transpose.__dict__.__setitem__('stypy_function_name', 'csr_matrix.transpose')
        csr_matrix.transpose.__dict__.__setitem__('stypy_param_names_list', ['axes', 'copy'])
        csr_matrix.transpose.__dict__.__setitem__('stypy_varargs_param_name', None)
        csr_matrix.transpose.__dict__.__setitem__('stypy_kwargs_param_name', None)
        csr_matrix.transpose.__dict__.__setitem__('stypy_call_defaults', defaults)
        csr_matrix.transpose.__dict__.__setitem__('stypy_call_varargs', varargs)
        csr_matrix.transpose.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        csr_matrix.transpose.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'csr_matrix.transpose', ['axes', 'copy'], None, None, defaults, varargs, kwargs)

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

        
        # Type idiom detected: calculating its left and rigth part (line 132)
        # Getting the type of 'axes' (line 132)
        axes_370381 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 8), 'axes')
        # Getting the type of 'None' (line 132)
        None_370382 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 23), 'None')
        
        (may_be_370383, more_types_in_union_370384) = may_not_be_none(axes_370381, None_370382)

        if may_be_370383:

            if more_types_in_union_370384:
                # Runtime conditional SSA (line 132)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Call to ValueError(...): (line 133)
            # Processing the call arguments (line 133)
            str_370386 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 133, 30), 'str', "Sparse matrices do not support an 'axes' parameter because swapping dimensions is the only logical permutation.")
            # Processing the call keyword arguments (line 133)
            kwargs_370387 = {}
            # Getting the type of 'ValueError' (line 133)
            ValueError_370385 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 18), 'ValueError', False)
            # Calling ValueError(args, kwargs) (line 133)
            ValueError_call_result_370388 = invoke(stypy.reporting.localization.Localization(__file__, 133, 18), ValueError_370385, *[str_370386], **kwargs_370387)
            
            ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 133, 12), ValueError_call_result_370388, 'raise parameter', BaseException)

            if more_types_in_union_370384:
                # SSA join for if statement (line 132)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Assigning a Attribute to a Tuple (line 137):
        
        # Assigning a Subscript to a Name (line 137):
        
        # Obtaining the type of the subscript
        int_370389 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 137, 8), 'int')
        # Getting the type of 'self' (line 137)
        self_370390 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 15), 'self')
        # Obtaining the member 'shape' of a type (line 137)
        shape_370391 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 137, 15), self_370390, 'shape')
        # Obtaining the member '__getitem__' of a type (line 137)
        getitem___370392 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 137, 8), shape_370391, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 137)
        subscript_call_result_370393 = invoke(stypy.reporting.localization.Localization(__file__, 137, 8), getitem___370392, int_370389)
        
        # Assigning a type to the variable 'tuple_var_assignment_370318' (line 137)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 137, 8), 'tuple_var_assignment_370318', subscript_call_result_370393)
        
        # Assigning a Subscript to a Name (line 137):
        
        # Obtaining the type of the subscript
        int_370394 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 137, 8), 'int')
        # Getting the type of 'self' (line 137)
        self_370395 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 15), 'self')
        # Obtaining the member 'shape' of a type (line 137)
        shape_370396 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 137, 15), self_370395, 'shape')
        # Obtaining the member '__getitem__' of a type (line 137)
        getitem___370397 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 137, 8), shape_370396, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 137)
        subscript_call_result_370398 = invoke(stypy.reporting.localization.Localization(__file__, 137, 8), getitem___370397, int_370394)
        
        # Assigning a type to the variable 'tuple_var_assignment_370319' (line 137)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 137, 8), 'tuple_var_assignment_370319', subscript_call_result_370398)
        
        # Assigning a Name to a Name (line 137):
        # Getting the type of 'tuple_var_assignment_370318' (line 137)
        tuple_var_assignment_370318_370399 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 8), 'tuple_var_assignment_370318')
        # Assigning a type to the variable 'M' (line 137)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 137, 8), 'M', tuple_var_assignment_370318_370399)
        
        # Assigning a Name to a Name (line 137):
        # Getting the type of 'tuple_var_assignment_370319' (line 137)
        tuple_var_assignment_370319_370400 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 8), 'tuple_var_assignment_370319')
        # Assigning a type to the variable 'N' (line 137)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 137, 11), 'N', tuple_var_assignment_370319_370400)
        stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 139, 8))
        
        # 'from scipy.sparse.csc import csc_matrix' statement (line 139)
        update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/sparse/')
        import_370401 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 139, 8), 'scipy.sparse.csc')

        if (type(import_370401) is not StypyTypeError):

            if (import_370401 != 'pyd_module'):
                __import__(import_370401)
                sys_modules_370402 = sys.modules[import_370401]
                import_from_module(stypy.reporting.localization.Localization(__file__, 139, 8), 'scipy.sparse.csc', sys_modules_370402.module_type_store, module_type_store, ['csc_matrix'])
                nest_module(stypy.reporting.localization.Localization(__file__, 139, 8), __file__, sys_modules_370402, sys_modules_370402.module_type_store, module_type_store)
            else:
                from scipy.sparse.csc import csc_matrix

                import_from_module(stypy.reporting.localization.Localization(__file__, 139, 8), 'scipy.sparse.csc', None, module_type_store, ['csc_matrix'], [csc_matrix])

        else:
            # Assigning a type to the variable 'scipy.sparse.csc' (line 139)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 139, 8), 'scipy.sparse.csc', import_370401)

        remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/sparse/')
        
        
        # Call to csc_matrix(...): (line 140)
        # Processing the call arguments (line 140)
        
        # Obtaining an instance of the builtin type 'tuple' (line 140)
        tuple_370404 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 140, 27), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 140)
        # Adding element type (line 140)
        # Getting the type of 'self' (line 140)
        self_370405 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 27), 'self', False)
        # Obtaining the member 'data' of a type (line 140)
        data_370406 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 140, 27), self_370405, 'data')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 140, 27), tuple_370404, data_370406)
        # Adding element type (line 140)
        # Getting the type of 'self' (line 140)
        self_370407 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 38), 'self', False)
        # Obtaining the member 'indices' of a type (line 140)
        indices_370408 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 140, 38), self_370407, 'indices')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 140, 27), tuple_370404, indices_370408)
        # Adding element type (line 140)
        # Getting the type of 'self' (line 141)
        self_370409 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 27), 'self', False)
        # Obtaining the member 'indptr' of a type (line 141)
        indptr_370410 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 141, 27), self_370409, 'indptr')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 140, 27), tuple_370404, indptr_370410)
        
        # Processing the call keyword arguments (line 140)
        
        # Obtaining an instance of the builtin type 'tuple' (line 141)
        tuple_370411 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 141, 48), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 141)
        # Adding element type (line 141)
        # Getting the type of 'N' (line 141)
        N_370412 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 48), 'N', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 141, 48), tuple_370411, N_370412)
        # Adding element type (line 141)
        # Getting the type of 'M' (line 141)
        M_370413 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 51), 'M', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 141, 48), tuple_370411, M_370413)
        
        keyword_370414 = tuple_370411
        # Getting the type of 'copy' (line 141)
        copy_370415 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 60), 'copy', False)
        keyword_370416 = copy_370415
        kwargs_370417 = {'shape': keyword_370414, 'copy': keyword_370416}
        # Getting the type of 'csc_matrix' (line 140)
        csc_matrix_370403 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 15), 'csc_matrix', False)
        # Calling csc_matrix(args, kwargs) (line 140)
        csc_matrix_call_result_370418 = invoke(stypy.reporting.localization.Localization(__file__, 140, 15), csc_matrix_370403, *[tuple_370404], **kwargs_370417)
        
        # Assigning a type to the variable 'stypy_return_type' (line 140)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 140, 8), 'stypy_return_type', csc_matrix_call_result_370418)
        
        # ################# End of 'transpose(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'transpose' in the type store
        # Getting the type of 'stypy_return_type' (line 131)
        stypy_return_type_370419 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_370419)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'transpose'
        return stypy_return_type_370419

    
    # Assigning a Attribute to a Attribute (line 143):

    @norecursion
    def tolil(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'False' (line 145)
        False_370420 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 25), 'False')
        defaults = [False_370420]
        # Create a new context for function 'tolil'
        module_type_store = module_type_store.open_function_context('tolil', 145, 4, False)
        # Assigning a type to the variable 'self' (line 146)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 146, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        csr_matrix.tolil.__dict__.__setitem__('stypy_localization', localization)
        csr_matrix.tolil.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        csr_matrix.tolil.__dict__.__setitem__('stypy_type_store', module_type_store)
        csr_matrix.tolil.__dict__.__setitem__('stypy_function_name', 'csr_matrix.tolil')
        csr_matrix.tolil.__dict__.__setitem__('stypy_param_names_list', ['copy'])
        csr_matrix.tolil.__dict__.__setitem__('stypy_varargs_param_name', None)
        csr_matrix.tolil.__dict__.__setitem__('stypy_kwargs_param_name', None)
        csr_matrix.tolil.__dict__.__setitem__('stypy_call_defaults', defaults)
        csr_matrix.tolil.__dict__.__setitem__('stypy_call_varargs', varargs)
        csr_matrix.tolil.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        csr_matrix.tolil.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'csr_matrix.tolil', ['copy'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'tolil', localization, ['copy'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'tolil(...)' code ##################

        stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 146, 8))
        
        # 'from scipy.sparse.lil import lil_matrix' statement (line 146)
        update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/sparse/')
        import_370421 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 146, 8), 'scipy.sparse.lil')

        if (type(import_370421) is not StypyTypeError):

            if (import_370421 != 'pyd_module'):
                __import__(import_370421)
                sys_modules_370422 = sys.modules[import_370421]
                import_from_module(stypy.reporting.localization.Localization(__file__, 146, 8), 'scipy.sparse.lil', sys_modules_370422.module_type_store, module_type_store, ['lil_matrix'])
                nest_module(stypy.reporting.localization.Localization(__file__, 146, 8), __file__, sys_modules_370422, sys_modules_370422.module_type_store, module_type_store)
            else:
                from scipy.sparse.lil import lil_matrix

                import_from_module(stypy.reporting.localization.Localization(__file__, 146, 8), 'scipy.sparse.lil', None, module_type_store, ['lil_matrix'], [lil_matrix])

        else:
            # Assigning a type to the variable 'scipy.sparse.lil' (line 146)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 146, 8), 'scipy.sparse.lil', import_370421)

        remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/sparse/')
        
        
        # Assigning a Call to a Name (line 147):
        
        # Assigning a Call to a Name (line 147):
        
        # Call to lil_matrix(...): (line 147)
        # Processing the call arguments (line 147)
        # Getting the type of 'self' (line 147)
        self_370424 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 25), 'self', False)
        # Obtaining the member 'shape' of a type (line 147)
        shape_370425 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 147, 25), self_370424, 'shape')
        # Processing the call keyword arguments (line 147)
        # Getting the type of 'self' (line 147)
        self_370426 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 42), 'self', False)
        # Obtaining the member 'dtype' of a type (line 147)
        dtype_370427 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 147, 42), self_370426, 'dtype')
        keyword_370428 = dtype_370427
        kwargs_370429 = {'dtype': keyword_370428}
        # Getting the type of 'lil_matrix' (line 147)
        lil_matrix_370423 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 14), 'lil_matrix', False)
        # Calling lil_matrix(args, kwargs) (line 147)
        lil_matrix_call_result_370430 = invoke(stypy.reporting.localization.Localization(__file__, 147, 14), lil_matrix_370423, *[shape_370425], **kwargs_370429)
        
        # Assigning a type to the variable 'lil' (line 147)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 147, 8), 'lil', lil_matrix_call_result_370430)
        
        # Call to sum_duplicates(...): (line 149)
        # Processing the call keyword arguments (line 149)
        kwargs_370433 = {}
        # Getting the type of 'self' (line 149)
        self_370431 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 8), 'self', False)
        # Obtaining the member 'sum_duplicates' of a type (line 149)
        sum_duplicates_370432 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 149, 8), self_370431, 'sum_duplicates')
        # Calling sum_duplicates(args, kwargs) (line 149)
        sum_duplicates_call_result_370434 = invoke(stypy.reporting.localization.Localization(__file__, 149, 8), sum_duplicates_370432, *[], **kwargs_370433)
        
        
        # Assigning a Tuple to a Tuple (line 150):
        
        # Assigning a Attribute to a Name (line 150):
        # Getting the type of 'self' (line 150)
        self_370435 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 22), 'self')
        # Obtaining the member 'indptr' of a type (line 150)
        indptr_370436 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 150, 22), self_370435, 'indptr')
        # Assigning a type to the variable 'tuple_assignment_370320' (line 150)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 150, 8), 'tuple_assignment_370320', indptr_370436)
        
        # Assigning a Attribute to a Name (line 150):
        # Getting the type of 'self' (line 150)
        self_370437 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 34), 'self')
        # Obtaining the member 'indices' of a type (line 150)
        indices_370438 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 150, 34), self_370437, 'indices')
        # Assigning a type to the variable 'tuple_assignment_370321' (line 150)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 150, 8), 'tuple_assignment_370321', indices_370438)
        
        # Assigning a Attribute to a Name (line 150):
        # Getting the type of 'self' (line 150)
        self_370439 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 47), 'self')
        # Obtaining the member 'data' of a type (line 150)
        data_370440 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 150, 47), self_370439, 'data')
        # Assigning a type to the variable 'tuple_assignment_370322' (line 150)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 150, 8), 'tuple_assignment_370322', data_370440)
        
        # Assigning a Name to a Name (line 150):
        # Getting the type of 'tuple_assignment_370320' (line 150)
        tuple_assignment_370320_370441 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 8), 'tuple_assignment_370320')
        # Assigning a type to the variable 'ptr' (line 150)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 150, 8), 'ptr', tuple_assignment_370320_370441)
        
        # Assigning a Name to a Name (line 150):
        # Getting the type of 'tuple_assignment_370321' (line 150)
        tuple_assignment_370321_370442 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 8), 'tuple_assignment_370321')
        # Assigning a type to the variable 'ind' (line 150)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 150, 12), 'ind', tuple_assignment_370321_370442)
        
        # Assigning a Name to a Name (line 150):
        # Getting the type of 'tuple_assignment_370322' (line 150)
        tuple_assignment_370322_370443 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 8), 'tuple_assignment_370322')
        # Assigning a type to the variable 'dat' (line 150)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 150, 16), 'dat', tuple_assignment_370322_370443)
        
        # Assigning a Tuple to a Tuple (line 151):
        
        # Assigning a Attribute to a Name (line 151):
        # Getting the type of 'lil' (line 151)
        lil_370444 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 21), 'lil')
        # Obtaining the member 'rows' of a type (line 151)
        rows_370445 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 151, 21), lil_370444, 'rows')
        # Assigning a type to the variable 'tuple_assignment_370323' (line 151)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 151, 8), 'tuple_assignment_370323', rows_370445)
        
        # Assigning a Attribute to a Name (line 151):
        # Getting the type of 'lil' (line 151)
        lil_370446 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 31), 'lil')
        # Obtaining the member 'data' of a type (line 151)
        data_370447 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 151, 31), lil_370446, 'data')
        # Assigning a type to the variable 'tuple_assignment_370324' (line 151)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 151, 8), 'tuple_assignment_370324', data_370447)
        
        # Assigning a Name to a Name (line 151):
        # Getting the type of 'tuple_assignment_370323' (line 151)
        tuple_assignment_370323_370448 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 8), 'tuple_assignment_370323')
        # Assigning a type to the variable 'rows' (line 151)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 151, 8), 'rows', tuple_assignment_370323_370448)
        
        # Assigning a Name to a Name (line 151):
        # Getting the type of 'tuple_assignment_370324' (line 151)
        tuple_assignment_370324_370449 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 8), 'tuple_assignment_370324')
        # Assigning a type to the variable 'data' (line 151)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 151, 14), 'data', tuple_assignment_370324_370449)
        
        
        # Call to xrange(...): (line 153)
        # Processing the call arguments (line 153)
        
        # Obtaining the type of the subscript
        int_370451 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 153, 35), 'int')
        # Getting the type of 'self' (line 153)
        self_370452 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 24), 'self', False)
        # Obtaining the member 'shape' of a type (line 153)
        shape_370453 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 153, 24), self_370452, 'shape')
        # Obtaining the member '__getitem__' of a type (line 153)
        getitem___370454 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 153, 24), shape_370453, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 153)
        subscript_call_result_370455 = invoke(stypy.reporting.localization.Localization(__file__, 153, 24), getitem___370454, int_370451)
        
        # Processing the call keyword arguments (line 153)
        kwargs_370456 = {}
        # Getting the type of 'xrange' (line 153)
        xrange_370450 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 17), 'xrange', False)
        # Calling xrange(args, kwargs) (line 153)
        xrange_call_result_370457 = invoke(stypy.reporting.localization.Localization(__file__, 153, 17), xrange_370450, *[subscript_call_result_370455], **kwargs_370456)
        
        # Testing the type of a for loop iterable (line 153)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 153, 8), xrange_call_result_370457)
        # Getting the type of the for loop variable (line 153)
        for_loop_var_370458 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 153, 8), xrange_call_result_370457)
        # Assigning a type to the variable 'n' (line 153)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 153, 8), 'n', for_loop_var_370458)
        # SSA begins for a for statement (line 153)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Subscript to a Name (line 154):
        
        # Assigning a Subscript to a Name (line 154):
        
        # Obtaining the type of the subscript
        # Getting the type of 'n' (line 154)
        n_370459 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 24), 'n')
        # Getting the type of 'ptr' (line 154)
        ptr_370460 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 20), 'ptr')
        # Obtaining the member '__getitem__' of a type (line 154)
        getitem___370461 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 154, 20), ptr_370460, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 154)
        subscript_call_result_370462 = invoke(stypy.reporting.localization.Localization(__file__, 154, 20), getitem___370461, n_370459)
        
        # Assigning a type to the variable 'start' (line 154)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 154, 12), 'start', subscript_call_result_370462)
        
        # Assigning a Subscript to a Name (line 155):
        
        # Assigning a Subscript to a Name (line 155):
        
        # Obtaining the type of the subscript
        # Getting the type of 'n' (line 155)
        n_370463 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 22), 'n')
        int_370464 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 155, 24), 'int')
        # Applying the binary operator '+' (line 155)
        result_add_370465 = python_operator(stypy.reporting.localization.Localization(__file__, 155, 22), '+', n_370463, int_370464)
        
        # Getting the type of 'ptr' (line 155)
        ptr_370466 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 18), 'ptr')
        # Obtaining the member '__getitem__' of a type (line 155)
        getitem___370467 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 155, 18), ptr_370466, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 155)
        subscript_call_result_370468 = invoke(stypy.reporting.localization.Localization(__file__, 155, 18), getitem___370467, result_add_370465)
        
        # Assigning a type to the variable 'end' (line 155)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 155, 12), 'end', subscript_call_result_370468)
        
        # Assigning a Call to a Subscript (line 156):
        
        # Assigning a Call to a Subscript (line 156):
        
        # Call to tolist(...): (line 156)
        # Processing the call keyword arguments (line 156)
        kwargs_370476 = {}
        
        # Obtaining the type of the subscript
        # Getting the type of 'start' (line 156)
        start_370469 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 26), 'start', False)
        # Getting the type of 'end' (line 156)
        end_370470 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 32), 'end', False)
        slice_370471 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 156, 22), start_370469, end_370470, None)
        # Getting the type of 'ind' (line 156)
        ind_370472 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 22), 'ind', False)
        # Obtaining the member '__getitem__' of a type (line 156)
        getitem___370473 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 156, 22), ind_370472, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 156)
        subscript_call_result_370474 = invoke(stypy.reporting.localization.Localization(__file__, 156, 22), getitem___370473, slice_370471)
        
        # Obtaining the member 'tolist' of a type (line 156)
        tolist_370475 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 156, 22), subscript_call_result_370474, 'tolist')
        # Calling tolist(args, kwargs) (line 156)
        tolist_call_result_370477 = invoke(stypy.reporting.localization.Localization(__file__, 156, 22), tolist_370475, *[], **kwargs_370476)
        
        # Getting the type of 'rows' (line 156)
        rows_370478 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 12), 'rows')
        # Getting the type of 'n' (line 156)
        n_370479 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 17), 'n')
        # Storing an element on a container (line 156)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 156, 12), rows_370478, (n_370479, tolist_call_result_370477))
        
        # Assigning a Call to a Subscript (line 157):
        
        # Assigning a Call to a Subscript (line 157):
        
        # Call to tolist(...): (line 157)
        # Processing the call keyword arguments (line 157)
        kwargs_370487 = {}
        
        # Obtaining the type of the subscript
        # Getting the type of 'start' (line 157)
        start_370480 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 26), 'start', False)
        # Getting the type of 'end' (line 157)
        end_370481 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 32), 'end', False)
        slice_370482 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 157, 22), start_370480, end_370481, None)
        # Getting the type of 'dat' (line 157)
        dat_370483 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 22), 'dat', False)
        # Obtaining the member '__getitem__' of a type (line 157)
        getitem___370484 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 157, 22), dat_370483, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 157)
        subscript_call_result_370485 = invoke(stypy.reporting.localization.Localization(__file__, 157, 22), getitem___370484, slice_370482)
        
        # Obtaining the member 'tolist' of a type (line 157)
        tolist_370486 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 157, 22), subscript_call_result_370485, 'tolist')
        # Calling tolist(args, kwargs) (line 157)
        tolist_call_result_370488 = invoke(stypy.reporting.localization.Localization(__file__, 157, 22), tolist_370486, *[], **kwargs_370487)
        
        # Getting the type of 'data' (line 157)
        data_370489 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 12), 'data')
        # Getting the type of 'n' (line 157)
        n_370490 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 17), 'n')
        # Storing an element on a container (line 157)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 157, 12), data_370489, (n_370490, tolist_call_result_370488))
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        # Getting the type of 'lil' (line 159)
        lil_370491 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 15), 'lil')
        # Assigning a type to the variable 'stypy_return_type' (line 159)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 159, 8), 'stypy_return_type', lil_370491)
        
        # ################# End of 'tolil(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'tolil' in the type store
        # Getting the type of 'stypy_return_type' (line 145)
        stypy_return_type_370492 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_370492)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'tolil'
        return stypy_return_type_370492

    
    # Assigning a Attribute to a Attribute (line 161):

    @norecursion
    def tocsr(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'False' (line 163)
        False_370493 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 25), 'False')
        defaults = [False_370493]
        # Create a new context for function 'tocsr'
        module_type_store = module_type_store.open_function_context('tocsr', 163, 4, False)
        # Assigning a type to the variable 'self' (line 164)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 164, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        csr_matrix.tocsr.__dict__.__setitem__('stypy_localization', localization)
        csr_matrix.tocsr.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        csr_matrix.tocsr.__dict__.__setitem__('stypy_type_store', module_type_store)
        csr_matrix.tocsr.__dict__.__setitem__('stypy_function_name', 'csr_matrix.tocsr')
        csr_matrix.tocsr.__dict__.__setitem__('stypy_param_names_list', ['copy'])
        csr_matrix.tocsr.__dict__.__setitem__('stypy_varargs_param_name', None)
        csr_matrix.tocsr.__dict__.__setitem__('stypy_kwargs_param_name', None)
        csr_matrix.tocsr.__dict__.__setitem__('stypy_call_defaults', defaults)
        csr_matrix.tocsr.__dict__.__setitem__('stypy_call_varargs', varargs)
        csr_matrix.tocsr.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        csr_matrix.tocsr.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'csr_matrix.tocsr', ['copy'], None, None, defaults, varargs, kwargs)

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

        
        # Getting the type of 'copy' (line 164)
        copy_370494 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 11), 'copy')
        # Testing the type of an if condition (line 164)
        if_condition_370495 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 164, 8), copy_370494)
        # Assigning a type to the variable 'if_condition_370495' (line 164)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 164, 8), 'if_condition_370495', if_condition_370495)
        # SSA begins for if statement (line 164)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to copy(...): (line 165)
        # Processing the call keyword arguments (line 165)
        kwargs_370498 = {}
        # Getting the type of 'self' (line 165)
        self_370496 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 19), 'self', False)
        # Obtaining the member 'copy' of a type (line 165)
        copy_370497 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 165, 19), self_370496, 'copy')
        # Calling copy(args, kwargs) (line 165)
        copy_call_result_370499 = invoke(stypy.reporting.localization.Localization(__file__, 165, 19), copy_370497, *[], **kwargs_370498)
        
        # Assigning a type to the variable 'stypy_return_type' (line 165)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 165, 12), 'stypy_return_type', copy_call_result_370499)
        # SSA branch for the else part of an if statement (line 164)
        module_type_store.open_ssa_branch('else')
        # Getting the type of 'self' (line 167)
        self_370500 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 19), 'self')
        # Assigning a type to the variable 'stypy_return_type' (line 167)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 167, 12), 'stypy_return_type', self_370500)
        # SSA join for if statement (line 164)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'tocsr(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'tocsr' in the type store
        # Getting the type of 'stypy_return_type' (line 163)
        stypy_return_type_370501 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_370501)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'tocsr'
        return stypy_return_type_370501

    
    # Assigning a Attribute to a Attribute (line 169):

    @norecursion
    def tocsc(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'False' (line 171)
        False_370502 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 25), 'False')
        defaults = [False_370502]
        # Create a new context for function 'tocsc'
        module_type_store = module_type_store.open_function_context('tocsc', 171, 4, False)
        # Assigning a type to the variable 'self' (line 172)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 172, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        csr_matrix.tocsc.__dict__.__setitem__('stypy_localization', localization)
        csr_matrix.tocsc.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        csr_matrix.tocsc.__dict__.__setitem__('stypy_type_store', module_type_store)
        csr_matrix.tocsc.__dict__.__setitem__('stypy_function_name', 'csr_matrix.tocsc')
        csr_matrix.tocsc.__dict__.__setitem__('stypy_param_names_list', ['copy'])
        csr_matrix.tocsc.__dict__.__setitem__('stypy_varargs_param_name', None)
        csr_matrix.tocsc.__dict__.__setitem__('stypy_kwargs_param_name', None)
        csr_matrix.tocsc.__dict__.__setitem__('stypy_call_defaults', defaults)
        csr_matrix.tocsc.__dict__.__setitem__('stypy_call_varargs', varargs)
        csr_matrix.tocsc.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        csr_matrix.tocsc.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'csr_matrix.tocsc', ['copy'], None, None, defaults, varargs, kwargs)

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

        
        # Assigning a Call to a Name (line 172):
        
        # Assigning a Call to a Name (line 172):
        
        # Call to get_index_dtype(...): (line 172)
        # Processing the call arguments (line 172)
        
        # Obtaining an instance of the builtin type 'tuple' (line 172)
        tuple_370504 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 172, 37), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 172)
        # Adding element type (line 172)
        # Getting the type of 'self' (line 172)
        self_370505 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 37), 'self', False)
        # Obtaining the member 'indptr' of a type (line 172)
        indptr_370506 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 172, 37), self_370505, 'indptr')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 172, 37), tuple_370504, indptr_370506)
        # Adding element type (line 172)
        # Getting the type of 'self' (line 172)
        self_370507 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 50), 'self', False)
        # Obtaining the member 'indices' of a type (line 172)
        indices_370508 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 172, 50), self_370507, 'indices')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 172, 37), tuple_370504, indices_370508)
        
        # Processing the call keyword arguments (line 172)
        
        # Call to max(...): (line 173)
        # Processing the call arguments (line 173)
        # Getting the type of 'self' (line 173)
        self_370510 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 47), 'self', False)
        # Obtaining the member 'nnz' of a type (line 173)
        nnz_370511 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 173, 47), self_370510, 'nnz')
        
        # Obtaining the type of the subscript
        int_370512 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 173, 68), 'int')
        # Getting the type of 'self' (line 173)
        self_370513 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 57), 'self', False)
        # Obtaining the member 'shape' of a type (line 173)
        shape_370514 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 173, 57), self_370513, 'shape')
        # Obtaining the member '__getitem__' of a type (line 173)
        getitem___370515 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 173, 57), shape_370514, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 173)
        subscript_call_result_370516 = invoke(stypy.reporting.localization.Localization(__file__, 173, 57), getitem___370515, int_370512)
        
        # Processing the call keyword arguments (line 173)
        kwargs_370517 = {}
        # Getting the type of 'max' (line 173)
        max_370509 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 43), 'max', False)
        # Calling max(args, kwargs) (line 173)
        max_call_result_370518 = invoke(stypy.reporting.localization.Localization(__file__, 173, 43), max_370509, *[nnz_370511, subscript_call_result_370516], **kwargs_370517)
        
        keyword_370519 = max_call_result_370518
        kwargs_370520 = {'maxval': keyword_370519}
        # Getting the type of 'get_index_dtype' (line 172)
        get_index_dtype_370503 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 20), 'get_index_dtype', False)
        # Calling get_index_dtype(args, kwargs) (line 172)
        get_index_dtype_call_result_370521 = invoke(stypy.reporting.localization.Localization(__file__, 172, 20), get_index_dtype_370503, *[tuple_370504], **kwargs_370520)
        
        # Assigning a type to the variable 'idx_dtype' (line 172)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 172, 8), 'idx_dtype', get_index_dtype_call_result_370521)
        
        # Assigning a Call to a Name (line 174):
        
        # Assigning a Call to a Name (line 174):
        
        # Call to empty(...): (line 174)
        # Processing the call arguments (line 174)
        
        # Obtaining the type of the subscript
        int_370524 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 174, 37), 'int')
        # Getting the type of 'self' (line 174)
        self_370525 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 26), 'self', False)
        # Obtaining the member 'shape' of a type (line 174)
        shape_370526 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 174, 26), self_370525, 'shape')
        # Obtaining the member '__getitem__' of a type (line 174)
        getitem___370527 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 174, 26), shape_370526, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 174)
        subscript_call_result_370528 = invoke(stypy.reporting.localization.Localization(__file__, 174, 26), getitem___370527, int_370524)
        
        int_370529 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 174, 42), 'int')
        # Applying the binary operator '+' (line 174)
        result_add_370530 = python_operator(stypy.reporting.localization.Localization(__file__, 174, 26), '+', subscript_call_result_370528, int_370529)
        
        # Processing the call keyword arguments (line 174)
        # Getting the type of 'idx_dtype' (line 174)
        idx_dtype_370531 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 51), 'idx_dtype', False)
        keyword_370532 = idx_dtype_370531
        kwargs_370533 = {'dtype': keyword_370532}
        # Getting the type of 'np' (line 174)
        np_370522 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 17), 'np', False)
        # Obtaining the member 'empty' of a type (line 174)
        empty_370523 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 174, 17), np_370522, 'empty')
        # Calling empty(args, kwargs) (line 174)
        empty_call_result_370534 = invoke(stypy.reporting.localization.Localization(__file__, 174, 17), empty_370523, *[result_add_370530], **kwargs_370533)
        
        # Assigning a type to the variable 'indptr' (line 174)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 174, 8), 'indptr', empty_call_result_370534)
        
        # Assigning a Call to a Name (line 175):
        
        # Assigning a Call to a Name (line 175):
        
        # Call to empty(...): (line 175)
        # Processing the call arguments (line 175)
        # Getting the type of 'self' (line 175)
        self_370537 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 27), 'self', False)
        # Obtaining the member 'nnz' of a type (line 175)
        nnz_370538 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 175, 27), self_370537, 'nnz')
        # Processing the call keyword arguments (line 175)
        # Getting the type of 'idx_dtype' (line 175)
        idx_dtype_370539 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 43), 'idx_dtype', False)
        keyword_370540 = idx_dtype_370539
        kwargs_370541 = {'dtype': keyword_370540}
        # Getting the type of 'np' (line 175)
        np_370535 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 18), 'np', False)
        # Obtaining the member 'empty' of a type (line 175)
        empty_370536 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 175, 18), np_370535, 'empty')
        # Calling empty(args, kwargs) (line 175)
        empty_call_result_370542 = invoke(stypy.reporting.localization.Localization(__file__, 175, 18), empty_370536, *[nnz_370538], **kwargs_370541)
        
        # Assigning a type to the variable 'indices' (line 175)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 175, 8), 'indices', empty_call_result_370542)
        
        # Assigning a Call to a Name (line 176):
        
        # Assigning a Call to a Name (line 176):
        
        # Call to empty(...): (line 176)
        # Processing the call arguments (line 176)
        # Getting the type of 'self' (line 176)
        self_370545 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 24), 'self', False)
        # Obtaining the member 'nnz' of a type (line 176)
        nnz_370546 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 176, 24), self_370545, 'nnz')
        # Processing the call keyword arguments (line 176)
        
        # Call to upcast(...): (line 176)
        # Processing the call arguments (line 176)
        # Getting the type of 'self' (line 176)
        self_370548 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 47), 'self', False)
        # Obtaining the member 'dtype' of a type (line 176)
        dtype_370549 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 176, 47), self_370548, 'dtype')
        # Processing the call keyword arguments (line 176)
        kwargs_370550 = {}
        # Getting the type of 'upcast' (line 176)
        upcast_370547 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 40), 'upcast', False)
        # Calling upcast(args, kwargs) (line 176)
        upcast_call_result_370551 = invoke(stypy.reporting.localization.Localization(__file__, 176, 40), upcast_370547, *[dtype_370549], **kwargs_370550)
        
        keyword_370552 = upcast_call_result_370551
        kwargs_370553 = {'dtype': keyword_370552}
        # Getting the type of 'np' (line 176)
        np_370543 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 15), 'np', False)
        # Obtaining the member 'empty' of a type (line 176)
        empty_370544 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 176, 15), np_370543, 'empty')
        # Calling empty(args, kwargs) (line 176)
        empty_call_result_370554 = invoke(stypy.reporting.localization.Localization(__file__, 176, 15), empty_370544, *[nnz_370546], **kwargs_370553)
        
        # Assigning a type to the variable 'data' (line 176)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 176, 8), 'data', empty_call_result_370554)
        
        # Call to csr_tocsc(...): (line 178)
        # Processing the call arguments (line 178)
        
        # Obtaining the type of the subscript
        int_370556 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 178, 29), 'int')
        # Getting the type of 'self' (line 178)
        self_370557 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 18), 'self', False)
        # Obtaining the member 'shape' of a type (line 178)
        shape_370558 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 178, 18), self_370557, 'shape')
        # Obtaining the member '__getitem__' of a type (line 178)
        getitem___370559 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 178, 18), shape_370558, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 178)
        subscript_call_result_370560 = invoke(stypy.reporting.localization.Localization(__file__, 178, 18), getitem___370559, int_370556)
        
        
        # Obtaining the type of the subscript
        int_370561 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 178, 44), 'int')
        # Getting the type of 'self' (line 178)
        self_370562 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 33), 'self', False)
        # Obtaining the member 'shape' of a type (line 178)
        shape_370563 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 178, 33), self_370562, 'shape')
        # Obtaining the member '__getitem__' of a type (line 178)
        getitem___370564 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 178, 33), shape_370563, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 178)
        subscript_call_result_370565 = invoke(stypy.reporting.localization.Localization(__file__, 178, 33), getitem___370564, int_370561)
        
        
        # Call to astype(...): (line 179)
        # Processing the call arguments (line 179)
        # Getting the type of 'idx_dtype' (line 179)
        idx_dtype_370569 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 37), 'idx_dtype', False)
        # Processing the call keyword arguments (line 179)
        kwargs_370570 = {}
        # Getting the type of 'self' (line 179)
        self_370566 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 18), 'self', False)
        # Obtaining the member 'indptr' of a type (line 179)
        indptr_370567 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 179, 18), self_370566, 'indptr')
        # Obtaining the member 'astype' of a type (line 179)
        astype_370568 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 179, 18), indptr_370567, 'astype')
        # Calling astype(args, kwargs) (line 179)
        astype_call_result_370571 = invoke(stypy.reporting.localization.Localization(__file__, 179, 18), astype_370568, *[idx_dtype_370569], **kwargs_370570)
        
        
        # Call to astype(...): (line 180)
        # Processing the call arguments (line 180)
        # Getting the type of 'idx_dtype' (line 180)
        idx_dtype_370575 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 38), 'idx_dtype', False)
        # Processing the call keyword arguments (line 180)
        kwargs_370576 = {}
        # Getting the type of 'self' (line 180)
        self_370572 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 18), 'self', False)
        # Obtaining the member 'indices' of a type (line 180)
        indices_370573 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 180, 18), self_370572, 'indices')
        # Obtaining the member 'astype' of a type (line 180)
        astype_370574 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 180, 18), indices_370573, 'astype')
        # Calling astype(args, kwargs) (line 180)
        astype_call_result_370577 = invoke(stypy.reporting.localization.Localization(__file__, 180, 18), astype_370574, *[idx_dtype_370575], **kwargs_370576)
        
        # Getting the type of 'self' (line 181)
        self_370578 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 18), 'self', False)
        # Obtaining the member 'data' of a type (line 181)
        data_370579 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 181, 18), self_370578, 'data')
        # Getting the type of 'indptr' (line 182)
        indptr_370580 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 182, 18), 'indptr', False)
        # Getting the type of 'indices' (line 183)
        indices_370581 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 18), 'indices', False)
        # Getting the type of 'data' (line 184)
        data_370582 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 18), 'data', False)
        # Processing the call keyword arguments (line 178)
        kwargs_370583 = {}
        # Getting the type of 'csr_tocsc' (line 178)
        csr_tocsc_370555 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 8), 'csr_tocsc', False)
        # Calling csr_tocsc(args, kwargs) (line 178)
        csr_tocsc_call_result_370584 = invoke(stypy.reporting.localization.Localization(__file__, 178, 8), csr_tocsc_370555, *[subscript_call_result_370560, subscript_call_result_370565, astype_call_result_370571, astype_call_result_370577, data_370579, indptr_370580, indices_370581, data_370582], **kwargs_370583)
        
        stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 186, 8))
        
        # 'from scipy.sparse.csc import csc_matrix' statement (line 186)
        update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/sparse/')
        import_370585 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 186, 8), 'scipy.sparse.csc')

        if (type(import_370585) is not StypyTypeError):

            if (import_370585 != 'pyd_module'):
                __import__(import_370585)
                sys_modules_370586 = sys.modules[import_370585]
                import_from_module(stypy.reporting.localization.Localization(__file__, 186, 8), 'scipy.sparse.csc', sys_modules_370586.module_type_store, module_type_store, ['csc_matrix'])
                nest_module(stypy.reporting.localization.Localization(__file__, 186, 8), __file__, sys_modules_370586, sys_modules_370586.module_type_store, module_type_store)
            else:
                from scipy.sparse.csc import csc_matrix

                import_from_module(stypy.reporting.localization.Localization(__file__, 186, 8), 'scipy.sparse.csc', None, module_type_store, ['csc_matrix'], [csc_matrix])

        else:
            # Assigning a type to the variable 'scipy.sparse.csc' (line 186)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 186, 8), 'scipy.sparse.csc', import_370585)

        remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/sparse/')
        
        
        # Assigning a Call to a Name (line 187):
        
        # Assigning a Call to a Name (line 187):
        
        # Call to csc_matrix(...): (line 187)
        # Processing the call arguments (line 187)
        
        # Obtaining an instance of the builtin type 'tuple' (line 187)
        tuple_370588 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 187, 24), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 187)
        # Adding element type (line 187)
        # Getting the type of 'data' (line 187)
        data_370589 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 24), 'data', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 187, 24), tuple_370588, data_370589)
        # Adding element type (line 187)
        # Getting the type of 'indices' (line 187)
        indices_370590 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 30), 'indices', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 187, 24), tuple_370588, indices_370590)
        # Adding element type (line 187)
        # Getting the type of 'indptr' (line 187)
        indptr_370591 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 39), 'indptr', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 187, 24), tuple_370588, indptr_370591)
        
        # Processing the call keyword arguments (line 187)
        # Getting the type of 'self' (line 187)
        self_370592 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 54), 'self', False)
        # Obtaining the member 'shape' of a type (line 187)
        shape_370593 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 187, 54), self_370592, 'shape')
        keyword_370594 = shape_370593
        kwargs_370595 = {'shape': keyword_370594}
        # Getting the type of 'csc_matrix' (line 187)
        csc_matrix_370587 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 12), 'csc_matrix', False)
        # Calling csc_matrix(args, kwargs) (line 187)
        csc_matrix_call_result_370596 = invoke(stypy.reporting.localization.Localization(__file__, 187, 12), csc_matrix_370587, *[tuple_370588], **kwargs_370595)
        
        # Assigning a type to the variable 'A' (line 187)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 187, 8), 'A', csc_matrix_call_result_370596)
        
        # Assigning a Name to a Attribute (line 188):
        
        # Assigning a Name to a Attribute (line 188):
        # Getting the type of 'True' (line 188)
        True_370597 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 188, 31), 'True')
        # Getting the type of 'A' (line 188)
        A_370598 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 188, 8), 'A')
        # Setting the type of the member 'has_sorted_indices' of a type (line 188)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 188, 8), A_370598, 'has_sorted_indices', True_370597)
        # Getting the type of 'A' (line 189)
        A_370599 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 15), 'A')
        # Assigning a type to the variable 'stypy_return_type' (line 189)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 189, 8), 'stypy_return_type', A_370599)
        
        # ################# End of 'tocsc(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'tocsc' in the type store
        # Getting the type of 'stypy_return_type' (line 171)
        stypy_return_type_370600 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_370600)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'tocsc'
        return stypy_return_type_370600

    
    # Assigning a Attribute to a Attribute (line 191):

    @norecursion
    def tobsr(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 193)
        None_370601 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 30), 'None')
        # Getting the type of 'True' (line 193)
        True_370602 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 41), 'True')
        defaults = [None_370601, True_370602]
        # Create a new context for function 'tobsr'
        module_type_store = module_type_store.open_function_context('tobsr', 193, 4, False)
        # Assigning a type to the variable 'self' (line 194)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 194, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        csr_matrix.tobsr.__dict__.__setitem__('stypy_localization', localization)
        csr_matrix.tobsr.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        csr_matrix.tobsr.__dict__.__setitem__('stypy_type_store', module_type_store)
        csr_matrix.tobsr.__dict__.__setitem__('stypy_function_name', 'csr_matrix.tobsr')
        csr_matrix.tobsr.__dict__.__setitem__('stypy_param_names_list', ['blocksize', 'copy'])
        csr_matrix.tobsr.__dict__.__setitem__('stypy_varargs_param_name', None)
        csr_matrix.tobsr.__dict__.__setitem__('stypy_kwargs_param_name', None)
        csr_matrix.tobsr.__dict__.__setitem__('stypy_call_defaults', defaults)
        csr_matrix.tobsr.__dict__.__setitem__('stypy_call_varargs', varargs)
        csr_matrix.tobsr.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        csr_matrix.tobsr.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'csr_matrix.tobsr', ['blocksize', 'copy'], None, None, defaults, varargs, kwargs)

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

        stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 194, 8))
        
        # 'from scipy.sparse.bsr import bsr_matrix' statement (line 194)
        update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/sparse/')
        import_370603 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 194, 8), 'scipy.sparse.bsr')

        if (type(import_370603) is not StypyTypeError):

            if (import_370603 != 'pyd_module'):
                __import__(import_370603)
                sys_modules_370604 = sys.modules[import_370603]
                import_from_module(stypy.reporting.localization.Localization(__file__, 194, 8), 'scipy.sparse.bsr', sys_modules_370604.module_type_store, module_type_store, ['bsr_matrix'])
                nest_module(stypy.reporting.localization.Localization(__file__, 194, 8), __file__, sys_modules_370604, sys_modules_370604.module_type_store, module_type_store)
            else:
                from scipy.sparse.bsr import bsr_matrix

                import_from_module(stypy.reporting.localization.Localization(__file__, 194, 8), 'scipy.sparse.bsr', None, module_type_store, ['bsr_matrix'], [bsr_matrix])

        else:
            # Assigning a type to the variable 'scipy.sparse.bsr' (line 194)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 194, 8), 'scipy.sparse.bsr', import_370603)

        remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/sparse/')
        
        
        # Type idiom detected: calculating its left and rigth part (line 196)
        # Getting the type of 'blocksize' (line 196)
        blocksize_370605 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 196, 11), 'blocksize')
        # Getting the type of 'None' (line 196)
        None_370606 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 196, 24), 'None')
        
        (may_be_370607, more_types_in_union_370608) = may_be_none(blocksize_370605, None_370606)

        if may_be_370607:

            if more_types_in_union_370608:
                # Runtime conditional SSA (line 196)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 197, 12))
            
            # 'from scipy.sparse.spfuncs import estimate_blocksize' statement (line 197)
            update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/sparse/')
            import_370609 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 197, 12), 'scipy.sparse.spfuncs')

            if (type(import_370609) is not StypyTypeError):

                if (import_370609 != 'pyd_module'):
                    __import__(import_370609)
                    sys_modules_370610 = sys.modules[import_370609]
                    import_from_module(stypy.reporting.localization.Localization(__file__, 197, 12), 'scipy.sparse.spfuncs', sys_modules_370610.module_type_store, module_type_store, ['estimate_blocksize'])
                    nest_module(stypy.reporting.localization.Localization(__file__, 197, 12), __file__, sys_modules_370610, sys_modules_370610.module_type_store, module_type_store)
                else:
                    from scipy.sparse.spfuncs import estimate_blocksize

                    import_from_module(stypy.reporting.localization.Localization(__file__, 197, 12), 'scipy.sparse.spfuncs', None, module_type_store, ['estimate_blocksize'], [estimate_blocksize])

            else:
                # Assigning a type to the variable 'scipy.sparse.spfuncs' (line 197)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 197, 12), 'scipy.sparse.spfuncs', import_370609)

            remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/sparse/')
            
            
            # Call to tobsr(...): (line 198)
            # Processing the call keyword arguments (line 198)
            
            # Call to estimate_blocksize(...): (line 198)
            # Processing the call arguments (line 198)
            # Getting the type of 'self' (line 198)
            self_370614 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 59), 'self', False)
            # Processing the call keyword arguments (line 198)
            kwargs_370615 = {}
            # Getting the type of 'estimate_blocksize' (line 198)
            estimate_blocksize_370613 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 40), 'estimate_blocksize', False)
            # Calling estimate_blocksize(args, kwargs) (line 198)
            estimate_blocksize_call_result_370616 = invoke(stypy.reporting.localization.Localization(__file__, 198, 40), estimate_blocksize_370613, *[self_370614], **kwargs_370615)
            
            keyword_370617 = estimate_blocksize_call_result_370616
            kwargs_370618 = {'blocksize': keyword_370617}
            # Getting the type of 'self' (line 198)
            self_370611 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 19), 'self', False)
            # Obtaining the member 'tobsr' of a type (line 198)
            tobsr_370612 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 198, 19), self_370611, 'tobsr')
            # Calling tobsr(args, kwargs) (line 198)
            tobsr_call_result_370619 = invoke(stypy.reporting.localization.Localization(__file__, 198, 19), tobsr_370612, *[], **kwargs_370618)
            
            # Assigning a type to the variable 'stypy_return_type' (line 198)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 198, 12), 'stypy_return_type', tobsr_call_result_370619)

            if more_types_in_union_370608:
                # Runtime conditional SSA for else branch (line 196)
                module_type_store.open_ssa_branch('idiom else')



        if ((not may_be_370607) or more_types_in_union_370608):
            
            
            # Getting the type of 'blocksize' (line 200)
            blocksize_370620 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 13), 'blocksize')
            
            # Obtaining an instance of the builtin type 'tuple' (line 200)
            tuple_370621 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 200, 27), 'tuple')
            # Adding type elements to the builtin type 'tuple' instance (line 200)
            # Adding element type (line 200)
            int_370622 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 200, 27), 'int')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 200, 27), tuple_370621, int_370622)
            # Adding element type (line 200)
            int_370623 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 200, 29), 'int')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 200, 27), tuple_370621, int_370623)
            
            # Applying the binary operator '==' (line 200)
            result_eq_370624 = python_operator(stypy.reporting.localization.Localization(__file__, 200, 13), '==', blocksize_370620, tuple_370621)
            
            # Testing the type of an if condition (line 200)
            if_condition_370625 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 200, 13), result_eq_370624)
            # Assigning a type to the variable 'if_condition_370625' (line 200)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 200, 13), 'if_condition_370625', if_condition_370625)
            # SSA begins for if statement (line 200)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a Tuple to a Name (line 201):
            
            # Assigning a Tuple to a Name (line 201):
            
            # Obtaining an instance of the builtin type 'tuple' (line 201)
            tuple_370626 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 201, 20), 'tuple')
            # Adding type elements to the builtin type 'tuple' instance (line 201)
            # Adding element type (line 201)
            
            # Call to reshape(...): (line 201)
            # Processing the call arguments (line 201)
            int_370630 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 201, 38), 'int')
            int_370631 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 201, 41), 'int')
            int_370632 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 201, 43), 'int')
            # Processing the call keyword arguments (line 201)
            kwargs_370633 = {}
            # Getting the type of 'self' (line 201)
            self_370627 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 20), 'self', False)
            # Obtaining the member 'data' of a type (line 201)
            data_370628 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 201, 20), self_370627, 'data')
            # Obtaining the member 'reshape' of a type (line 201)
            reshape_370629 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 201, 20), data_370628, 'reshape')
            # Calling reshape(args, kwargs) (line 201)
            reshape_call_result_370634 = invoke(stypy.reporting.localization.Localization(__file__, 201, 20), reshape_370629, *[int_370630, int_370631, int_370632], **kwargs_370633)
            
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 201, 20), tuple_370626, reshape_call_result_370634)
            # Adding element type (line 201)
            # Getting the type of 'self' (line 201)
            self_370635 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 46), 'self')
            # Obtaining the member 'indices' of a type (line 201)
            indices_370636 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 201, 46), self_370635, 'indices')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 201, 20), tuple_370626, indices_370636)
            # Adding element type (line 201)
            # Getting the type of 'self' (line 201)
            self_370637 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 59), 'self')
            # Obtaining the member 'indptr' of a type (line 201)
            indptr_370638 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 201, 59), self_370637, 'indptr')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 201, 20), tuple_370626, indptr_370638)
            
            # Assigning a type to the variable 'arg1' (line 201)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 201, 12), 'arg1', tuple_370626)
            
            # Call to bsr_matrix(...): (line 202)
            # Processing the call arguments (line 202)
            # Getting the type of 'arg1' (line 202)
            arg1_370640 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 30), 'arg1', False)
            # Processing the call keyword arguments (line 202)
            # Getting the type of 'self' (line 202)
            self_370641 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 42), 'self', False)
            # Obtaining the member 'shape' of a type (line 202)
            shape_370642 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 202, 42), self_370641, 'shape')
            keyword_370643 = shape_370642
            # Getting the type of 'copy' (line 202)
            copy_370644 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 59), 'copy', False)
            keyword_370645 = copy_370644
            kwargs_370646 = {'shape': keyword_370643, 'copy': keyword_370645}
            # Getting the type of 'bsr_matrix' (line 202)
            bsr_matrix_370639 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 19), 'bsr_matrix', False)
            # Calling bsr_matrix(args, kwargs) (line 202)
            bsr_matrix_call_result_370647 = invoke(stypy.reporting.localization.Localization(__file__, 202, 19), bsr_matrix_370639, *[arg1_370640], **kwargs_370646)
            
            # Assigning a type to the variable 'stypy_return_type' (line 202)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 202, 12), 'stypy_return_type', bsr_matrix_call_result_370647)
            # SSA branch for the else part of an if statement (line 200)
            module_type_store.open_ssa_branch('else')
            
            # Assigning a Name to a Tuple (line 205):
            
            # Assigning a Subscript to a Name (line 205):
            
            # Obtaining the type of the subscript
            int_370648 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 205, 12), 'int')
            # Getting the type of 'blocksize' (line 205)
            blocksize_370649 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 18), 'blocksize')
            # Obtaining the member '__getitem__' of a type (line 205)
            getitem___370650 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 205, 12), blocksize_370649, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 205)
            subscript_call_result_370651 = invoke(stypy.reporting.localization.Localization(__file__, 205, 12), getitem___370650, int_370648)
            
            # Assigning a type to the variable 'tuple_var_assignment_370325' (line 205)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 205, 12), 'tuple_var_assignment_370325', subscript_call_result_370651)
            
            # Assigning a Subscript to a Name (line 205):
            
            # Obtaining the type of the subscript
            int_370652 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 205, 12), 'int')
            # Getting the type of 'blocksize' (line 205)
            blocksize_370653 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 18), 'blocksize')
            # Obtaining the member '__getitem__' of a type (line 205)
            getitem___370654 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 205, 12), blocksize_370653, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 205)
            subscript_call_result_370655 = invoke(stypy.reporting.localization.Localization(__file__, 205, 12), getitem___370654, int_370652)
            
            # Assigning a type to the variable 'tuple_var_assignment_370326' (line 205)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 205, 12), 'tuple_var_assignment_370326', subscript_call_result_370655)
            
            # Assigning a Name to a Name (line 205):
            # Getting the type of 'tuple_var_assignment_370325' (line 205)
            tuple_var_assignment_370325_370656 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 12), 'tuple_var_assignment_370325')
            # Assigning a type to the variable 'R' (line 205)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 205, 12), 'R', tuple_var_assignment_370325_370656)
            
            # Assigning a Name to a Name (line 205):
            # Getting the type of 'tuple_var_assignment_370326' (line 205)
            tuple_var_assignment_370326_370657 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 12), 'tuple_var_assignment_370326')
            # Assigning a type to the variable 'C' (line 205)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 205, 14), 'C', tuple_var_assignment_370326_370657)
            
            # Assigning a Attribute to a Tuple (line 206):
            
            # Assigning a Subscript to a Name (line 206):
            
            # Obtaining the type of the subscript
            int_370658 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 206, 12), 'int')
            # Getting the type of 'self' (line 206)
            self_370659 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 206, 18), 'self')
            # Obtaining the member 'shape' of a type (line 206)
            shape_370660 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 206, 18), self_370659, 'shape')
            # Obtaining the member '__getitem__' of a type (line 206)
            getitem___370661 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 206, 12), shape_370660, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 206)
            subscript_call_result_370662 = invoke(stypy.reporting.localization.Localization(__file__, 206, 12), getitem___370661, int_370658)
            
            # Assigning a type to the variable 'tuple_var_assignment_370327' (line 206)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 206, 12), 'tuple_var_assignment_370327', subscript_call_result_370662)
            
            # Assigning a Subscript to a Name (line 206):
            
            # Obtaining the type of the subscript
            int_370663 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 206, 12), 'int')
            # Getting the type of 'self' (line 206)
            self_370664 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 206, 18), 'self')
            # Obtaining the member 'shape' of a type (line 206)
            shape_370665 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 206, 18), self_370664, 'shape')
            # Obtaining the member '__getitem__' of a type (line 206)
            getitem___370666 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 206, 12), shape_370665, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 206)
            subscript_call_result_370667 = invoke(stypy.reporting.localization.Localization(__file__, 206, 12), getitem___370666, int_370663)
            
            # Assigning a type to the variable 'tuple_var_assignment_370328' (line 206)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 206, 12), 'tuple_var_assignment_370328', subscript_call_result_370667)
            
            # Assigning a Name to a Name (line 206):
            # Getting the type of 'tuple_var_assignment_370327' (line 206)
            tuple_var_assignment_370327_370668 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 206, 12), 'tuple_var_assignment_370327')
            # Assigning a type to the variable 'M' (line 206)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 206, 12), 'M', tuple_var_assignment_370327_370668)
            
            # Assigning a Name to a Name (line 206):
            # Getting the type of 'tuple_var_assignment_370328' (line 206)
            tuple_var_assignment_370328_370669 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 206, 12), 'tuple_var_assignment_370328')
            # Assigning a type to the variable 'N' (line 206)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 206, 14), 'N', tuple_var_assignment_370328_370669)
            
            
            # Evaluating a boolean operation
            
            # Getting the type of 'R' (line 208)
            R_370670 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 208, 15), 'R')
            int_370671 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 208, 19), 'int')
            # Applying the binary operator '<' (line 208)
            result_lt_370672 = python_operator(stypy.reporting.localization.Localization(__file__, 208, 15), '<', R_370670, int_370671)
            
            
            # Getting the type of 'C' (line 208)
            C_370673 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 208, 24), 'C')
            int_370674 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 208, 28), 'int')
            # Applying the binary operator '<' (line 208)
            result_lt_370675 = python_operator(stypy.reporting.localization.Localization(__file__, 208, 24), '<', C_370673, int_370674)
            
            # Applying the binary operator 'or' (line 208)
            result_or_keyword_370676 = python_operator(stypy.reporting.localization.Localization(__file__, 208, 15), 'or', result_lt_370672, result_lt_370675)
            
            # Getting the type of 'M' (line 208)
            M_370677 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 208, 33), 'M')
            # Getting the type of 'R' (line 208)
            R_370678 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 208, 37), 'R')
            # Applying the binary operator '%' (line 208)
            result_mod_370679 = python_operator(stypy.reporting.localization.Localization(__file__, 208, 33), '%', M_370677, R_370678)
            
            int_370680 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 208, 42), 'int')
            # Applying the binary operator '!=' (line 208)
            result_ne_370681 = python_operator(stypy.reporting.localization.Localization(__file__, 208, 33), '!=', result_mod_370679, int_370680)
            
            # Applying the binary operator 'or' (line 208)
            result_or_keyword_370682 = python_operator(stypy.reporting.localization.Localization(__file__, 208, 15), 'or', result_or_keyword_370676, result_ne_370681)
            
            # Getting the type of 'N' (line 208)
            N_370683 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 208, 47), 'N')
            # Getting the type of 'C' (line 208)
            C_370684 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 208, 51), 'C')
            # Applying the binary operator '%' (line 208)
            result_mod_370685 = python_operator(stypy.reporting.localization.Localization(__file__, 208, 47), '%', N_370683, C_370684)
            
            int_370686 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 208, 56), 'int')
            # Applying the binary operator '!=' (line 208)
            result_ne_370687 = python_operator(stypy.reporting.localization.Localization(__file__, 208, 47), '!=', result_mod_370685, int_370686)
            
            # Applying the binary operator 'or' (line 208)
            result_or_keyword_370688 = python_operator(stypy.reporting.localization.Localization(__file__, 208, 15), 'or', result_or_keyword_370682, result_ne_370687)
            
            # Testing the type of an if condition (line 208)
            if_condition_370689 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 208, 12), result_or_keyword_370688)
            # Assigning a type to the variable 'if_condition_370689' (line 208)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 208, 12), 'if_condition_370689', if_condition_370689)
            # SSA begins for if statement (line 208)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to ValueError(...): (line 209)
            # Processing the call arguments (line 209)
            str_370691 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 209, 33), 'str', 'invalid blocksize %s')
            # Getting the type of 'blocksize' (line 209)
            blocksize_370692 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 58), 'blocksize', False)
            # Applying the binary operator '%' (line 209)
            result_mod_370693 = python_operator(stypy.reporting.localization.Localization(__file__, 209, 33), '%', str_370691, blocksize_370692)
            
            # Processing the call keyword arguments (line 209)
            kwargs_370694 = {}
            # Getting the type of 'ValueError' (line 209)
            ValueError_370690 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 22), 'ValueError', False)
            # Calling ValueError(args, kwargs) (line 209)
            ValueError_call_result_370695 = invoke(stypy.reporting.localization.Localization(__file__, 209, 22), ValueError_370690, *[result_mod_370693], **kwargs_370694)
            
            ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 209, 16), ValueError_call_result_370695, 'raise parameter', BaseException)
            # SSA join for if statement (line 208)
            module_type_store = module_type_store.join_ssa_context()
            
            
            # Assigning a Call to a Name (line 211):
            
            # Assigning a Call to a Name (line 211):
            
            # Call to csr_count_blocks(...): (line 211)
            # Processing the call arguments (line 211)
            # Getting the type of 'M' (line 211)
            M_370697 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 211, 36), 'M', False)
            # Getting the type of 'N' (line 211)
            N_370698 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 211, 38), 'N', False)
            # Getting the type of 'R' (line 211)
            R_370699 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 211, 40), 'R', False)
            # Getting the type of 'C' (line 211)
            C_370700 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 211, 42), 'C', False)
            # Getting the type of 'self' (line 211)
            self_370701 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 211, 44), 'self', False)
            # Obtaining the member 'indptr' of a type (line 211)
            indptr_370702 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 211, 44), self_370701, 'indptr')
            # Getting the type of 'self' (line 211)
            self_370703 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 211, 56), 'self', False)
            # Obtaining the member 'indices' of a type (line 211)
            indices_370704 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 211, 56), self_370703, 'indices')
            # Processing the call keyword arguments (line 211)
            kwargs_370705 = {}
            # Getting the type of 'csr_count_blocks' (line 211)
            csr_count_blocks_370696 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 211, 19), 'csr_count_blocks', False)
            # Calling csr_count_blocks(args, kwargs) (line 211)
            csr_count_blocks_call_result_370706 = invoke(stypy.reporting.localization.Localization(__file__, 211, 19), csr_count_blocks_370696, *[M_370697, N_370698, R_370699, C_370700, indptr_370702, indices_370704], **kwargs_370705)
            
            # Assigning a type to the variable 'blks' (line 211)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 211, 12), 'blks', csr_count_blocks_call_result_370706)
            
            # Assigning a Call to a Name (line 213):
            
            # Assigning a Call to a Name (line 213):
            
            # Call to get_index_dtype(...): (line 213)
            # Processing the call arguments (line 213)
            
            # Obtaining an instance of the builtin type 'tuple' (line 213)
            tuple_370708 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 213, 41), 'tuple')
            # Adding type elements to the builtin type 'tuple' instance (line 213)
            # Adding element type (line 213)
            # Getting the type of 'self' (line 213)
            self_370709 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 41), 'self', False)
            # Obtaining the member 'indptr' of a type (line 213)
            indptr_370710 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 213, 41), self_370709, 'indptr')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 213, 41), tuple_370708, indptr_370710)
            # Adding element type (line 213)
            # Getting the type of 'self' (line 213)
            self_370711 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 54), 'self', False)
            # Obtaining the member 'indices' of a type (line 213)
            indices_370712 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 213, 54), self_370711, 'indices')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 213, 41), tuple_370708, indices_370712)
            
            # Processing the call keyword arguments (line 213)
            
            # Call to max(...): (line 214)
            # Processing the call arguments (line 214)
            # Getting the type of 'N' (line 214)
            N_370714 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 51), 'N', False)
            # Getting the type of 'C' (line 214)
            C_370715 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 54), 'C', False)
            # Applying the binary operator '//' (line 214)
            result_floordiv_370716 = python_operator(stypy.reporting.localization.Localization(__file__, 214, 51), '//', N_370714, C_370715)
            
            # Getting the type of 'blks' (line 214)
            blks_370717 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 57), 'blks', False)
            # Processing the call keyword arguments (line 214)
            kwargs_370718 = {}
            # Getting the type of 'max' (line 214)
            max_370713 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 47), 'max', False)
            # Calling max(args, kwargs) (line 214)
            max_call_result_370719 = invoke(stypy.reporting.localization.Localization(__file__, 214, 47), max_370713, *[result_floordiv_370716, blks_370717], **kwargs_370718)
            
            keyword_370720 = max_call_result_370719
            kwargs_370721 = {'maxval': keyword_370720}
            # Getting the type of 'get_index_dtype' (line 213)
            get_index_dtype_370707 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 24), 'get_index_dtype', False)
            # Calling get_index_dtype(args, kwargs) (line 213)
            get_index_dtype_call_result_370722 = invoke(stypy.reporting.localization.Localization(__file__, 213, 24), get_index_dtype_370707, *[tuple_370708], **kwargs_370721)
            
            # Assigning a type to the variable 'idx_dtype' (line 213)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 213, 12), 'idx_dtype', get_index_dtype_call_result_370722)
            
            # Assigning a Call to a Name (line 215):
            
            # Assigning a Call to a Name (line 215):
            
            # Call to empty(...): (line 215)
            # Processing the call arguments (line 215)
            # Getting the type of 'M' (line 215)
            M_370725 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 215, 30), 'M', False)
            # Getting the type of 'R' (line 215)
            R_370726 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 215, 33), 'R', False)
            # Applying the binary operator '//' (line 215)
            result_floordiv_370727 = python_operator(stypy.reporting.localization.Localization(__file__, 215, 30), '//', M_370725, R_370726)
            
            int_370728 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 215, 35), 'int')
            # Applying the binary operator '+' (line 215)
            result_add_370729 = python_operator(stypy.reporting.localization.Localization(__file__, 215, 30), '+', result_floordiv_370727, int_370728)
            
            # Processing the call keyword arguments (line 215)
            # Getting the type of 'idx_dtype' (line 215)
            idx_dtype_370730 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 215, 44), 'idx_dtype', False)
            keyword_370731 = idx_dtype_370730
            kwargs_370732 = {'dtype': keyword_370731}
            # Getting the type of 'np' (line 215)
            np_370723 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 215, 21), 'np', False)
            # Obtaining the member 'empty' of a type (line 215)
            empty_370724 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 215, 21), np_370723, 'empty')
            # Calling empty(args, kwargs) (line 215)
            empty_call_result_370733 = invoke(stypy.reporting.localization.Localization(__file__, 215, 21), empty_370724, *[result_add_370729], **kwargs_370732)
            
            # Assigning a type to the variable 'indptr' (line 215)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 215, 12), 'indptr', empty_call_result_370733)
            
            # Assigning a Call to a Name (line 216):
            
            # Assigning a Call to a Name (line 216):
            
            # Call to empty(...): (line 216)
            # Processing the call arguments (line 216)
            # Getting the type of 'blks' (line 216)
            blks_370736 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 216, 31), 'blks', False)
            # Processing the call keyword arguments (line 216)
            # Getting the type of 'idx_dtype' (line 216)
            idx_dtype_370737 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 216, 43), 'idx_dtype', False)
            keyword_370738 = idx_dtype_370737
            kwargs_370739 = {'dtype': keyword_370738}
            # Getting the type of 'np' (line 216)
            np_370734 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 216, 22), 'np', False)
            # Obtaining the member 'empty' of a type (line 216)
            empty_370735 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 216, 22), np_370734, 'empty')
            # Calling empty(args, kwargs) (line 216)
            empty_call_result_370740 = invoke(stypy.reporting.localization.Localization(__file__, 216, 22), empty_370735, *[blks_370736], **kwargs_370739)
            
            # Assigning a type to the variable 'indices' (line 216)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 216, 12), 'indices', empty_call_result_370740)
            
            # Assigning a Call to a Name (line 217):
            
            # Assigning a Call to a Name (line 217):
            
            # Call to zeros(...): (line 217)
            # Processing the call arguments (line 217)
            
            # Obtaining an instance of the builtin type 'tuple' (line 217)
            tuple_370743 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 217, 29), 'tuple')
            # Adding type elements to the builtin type 'tuple' instance (line 217)
            # Adding element type (line 217)
            # Getting the type of 'blks' (line 217)
            blks_370744 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 29), 'blks', False)
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 217, 29), tuple_370743, blks_370744)
            # Adding element type (line 217)
            # Getting the type of 'R' (line 217)
            R_370745 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 34), 'R', False)
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 217, 29), tuple_370743, R_370745)
            # Adding element type (line 217)
            # Getting the type of 'C' (line 217)
            C_370746 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 36), 'C', False)
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 217, 29), tuple_370743, C_370746)
            
            # Processing the call keyword arguments (line 217)
            # Getting the type of 'self' (line 217)
            self_370747 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 46), 'self', False)
            # Obtaining the member 'dtype' of a type (line 217)
            dtype_370748 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 217, 46), self_370747, 'dtype')
            keyword_370749 = dtype_370748
            kwargs_370750 = {'dtype': keyword_370749}
            # Getting the type of 'np' (line 217)
            np_370741 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 19), 'np', False)
            # Obtaining the member 'zeros' of a type (line 217)
            zeros_370742 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 217, 19), np_370741, 'zeros')
            # Calling zeros(args, kwargs) (line 217)
            zeros_call_result_370751 = invoke(stypy.reporting.localization.Localization(__file__, 217, 19), zeros_370742, *[tuple_370743], **kwargs_370750)
            
            # Assigning a type to the variable 'data' (line 217)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 217, 12), 'data', zeros_call_result_370751)
            
            # Call to csr_tobsr(...): (line 219)
            # Processing the call arguments (line 219)
            # Getting the type of 'M' (line 219)
            M_370753 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 219, 22), 'M', False)
            # Getting the type of 'N' (line 219)
            N_370754 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 219, 25), 'N', False)
            # Getting the type of 'R' (line 219)
            R_370755 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 219, 28), 'R', False)
            # Getting the type of 'C' (line 219)
            C_370756 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 219, 31), 'C', False)
            
            # Call to astype(...): (line 220)
            # Processing the call arguments (line 220)
            # Getting the type of 'idx_dtype' (line 220)
            idx_dtype_370760 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 220, 41), 'idx_dtype', False)
            # Processing the call keyword arguments (line 220)
            kwargs_370761 = {}
            # Getting the type of 'self' (line 220)
            self_370757 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 220, 22), 'self', False)
            # Obtaining the member 'indptr' of a type (line 220)
            indptr_370758 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 220, 22), self_370757, 'indptr')
            # Obtaining the member 'astype' of a type (line 220)
            astype_370759 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 220, 22), indptr_370758, 'astype')
            # Calling astype(args, kwargs) (line 220)
            astype_call_result_370762 = invoke(stypy.reporting.localization.Localization(__file__, 220, 22), astype_370759, *[idx_dtype_370760], **kwargs_370761)
            
            
            # Call to astype(...): (line 221)
            # Processing the call arguments (line 221)
            # Getting the type of 'idx_dtype' (line 221)
            idx_dtype_370766 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 42), 'idx_dtype', False)
            # Processing the call keyword arguments (line 221)
            kwargs_370767 = {}
            # Getting the type of 'self' (line 221)
            self_370763 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 22), 'self', False)
            # Obtaining the member 'indices' of a type (line 221)
            indices_370764 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 221, 22), self_370763, 'indices')
            # Obtaining the member 'astype' of a type (line 221)
            astype_370765 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 221, 22), indices_370764, 'astype')
            # Calling astype(args, kwargs) (line 221)
            astype_call_result_370768 = invoke(stypy.reporting.localization.Localization(__file__, 221, 22), astype_370765, *[idx_dtype_370766], **kwargs_370767)
            
            # Getting the type of 'self' (line 222)
            self_370769 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 222, 22), 'self', False)
            # Obtaining the member 'data' of a type (line 222)
            data_370770 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 222, 22), self_370769, 'data')
            # Getting the type of 'indptr' (line 223)
            indptr_370771 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 22), 'indptr', False)
            # Getting the type of 'indices' (line 223)
            indices_370772 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 30), 'indices', False)
            
            # Call to ravel(...): (line 223)
            # Processing the call keyword arguments (line 223)
            kwargs_370775 = {}
            # Getting the type of 'data' (line 223)
            data_370773 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 39), 'data', False)
            # Obtaining the member 'ravel' of a type (line 223)
            ravel_370774 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 223, 39), data_370773, 'ravel')
            # Calling ravel(args, kwargs) (line 223)
            ravel_call_result_370776 = invoke(stypy.reporting.localization.Localization(__file__, 223, 39), ravel_370774, *[], **kwargs_370775)
            
            # Processing the call keyword arguments (line 219)
            kwargs_370777 = {}
            # Getting the type of 'csr_tobsr' (line 219)
            csr_tobsr_370752 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 219, 12), 'csr_tobsr', False)
            # Calling csr_tobsr(args, kwargs) (line 219)
            csr_tobsr_call_result_370778 = invoke(stypy.reporting.localization.Localization(__file__, 219, 12), csr_tobsr_370752, *[M_370753, N_370754, R_370755, C_370756, astype_call_result_370762, astype_call_result_370768, data_370770, indptr_370771, indices_370772, ravel_call_result_370776], **kwargs_370777)
            
            
            # Call to bsr_matrix(...): (line 225)
            # Processing the call arguments (line 225)
            
            # Obtaining an instance of the builtin type 'tuple' (line 225)
            tuple_370780 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 225, 31), 'tuple')
            # Adding type elements to the builtin type 'tuple' instance (line 225)
            # Adding element type (line 225)
            # Getting the type of 'data' (line 225)
            data_370781 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 225, 31), 'data', False)
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 225, 31), tuple_370780, data_370781)
            # Adding element type (line 225)
            # Getting the type of 'indices' (line 225)
            indices_370782 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 225, 36), 'indices', False)
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 225, 31), tuple_370780, indices_370782)
            # Adding element type (line 225)
            # Getting the type of 'indptr' (line 225)
            indptr_370783 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 225, 44), 'indptr', False)
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 225, 31), tuple_370780, indptr_370783)
            
            # Processing the call keyword arguments (line 225)
            # Getting the type of 'self' (line 225)
            self_370784 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 225, 59), 'self', False)
            # Obtaining the member 'shape' of a type (line 225)
            shape_370785 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 225, 59), self_370784, 'shape')
            keyword_370786 = shape_370785
            kwargs_370787 = {'shape': keyword_370786}
            # Getting the type of 'bsr_matrix' (line 225)
            bsr_matrix_370779 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 225, 19), 'bsr_matrix', False)
            # Calling bsr_matrix(args, kwargs) (line 225)
            bsr_matrix_call_result_370788 = invoke(stypy.reporting.localization.Localization(__file__, 225, 19), bsr_matrix_370779, *[tuple_370780], **kwargs_370787)
            
            # Assigning a type to the variable 'stypy_return_type' (line 225)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 225, 12), 'stypy_return_type', bsr_matrix_call_result_370788)
            # SSA join for if statement (line 200)
            module_type_store = module_type_store.join_ssa_context()
            

            if (may_be_370607 and more_types_in_union_370608):
                # SSA join for if statement (line 196)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # ################# End of 'tobsr(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'tobsr' in the type store
        # Getting the type of 'stypy_return_type' (line 193)
        stypy_return_type_370789 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_370789)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'tobsr'
        return stypy_return_type_370789

    
    # Assigning a Attribute to a Attribute (line 227):

    @norecursion
    def _swap(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_swap'
        module_type_store = module_type_store.open_function_context('_swap', 231, 4, False)
        # Assigning a type to the variable 'self' (line 232)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 232, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        csr_matrix._swap.__dict__.__setitem__('stypy_localization', localization)
        csr_matrix._swap.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        csr_matrix._swap.__dict__.__setitem__('stypy_type_store', module_type_store)
        csr_matrix._swap.__dict__.__setitem__('stypy_function_name', 'csr_matrix._swap')
        csr_matrix._swap.__dict__.__setitem__('stypy_param_names_list', ['x'])
        csr_matrix._swap.__dict__.__setitem__('stypy_varargs_param_name', None)
        csr_matrix._swap.__dict__.__setitem__('stypy_kwargs_param_name', None)
        csr_matrix._swap.__dict__.__setitem__('stypy_call_defaults', defaults)
        csr_matrix._swap.__dict__.__setitem__('stypy_call_varargs', varargs)
        csr_matrix._swap.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        csr_matrix._swap.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'csr_matrix._swap', ['x'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_swap', localization, ['x'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_swap(...)' code ##################

        str_370790 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 233, (-1)), 'str', 'swap the members of x if this is a column-oriented matrix\n        ')
        # Getting the type of 'x' (line 234)
        x_370791 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 234, 15), 'x')
        # Assigning a type to the variable 'stypy_return_type' (line 234)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 234, 8), 'stypy_return_type', x_370791)
        
        # ################# End of '_swap(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_swap' in the type store
        # Getting the type of 'stypy_return_type' (line 231)
        stypy_return_type_370792 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 231, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_370792)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_swap'
        return stypy_return_type_370792


    @norecursion
    def __getitem__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__getitem__'
        module_type_store = module_type_store.open_function_context('__getitem__', 236, 4, False)
        # Assigning a type to the variable 'self' (line 237)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 237, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        csr_matrix.__getitem__.__dict__.__setitem__('stypy_localization', localization)
        csr_matrix.__getitem__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        csr_matrix.__getitem__.__dict__.__setitem__('stypy_type_store', module_type_store)
        csr_matrix.__getitem__.__dict__.__setitem__('stypy_function_name', 'csr_matrix.__getitem__')
        csr_matrix.__getitem__.__dict__.__setitem__('stypy_param_names_list', ['key'])
        csr_matrix.__getitem__.__dict__.__setitem__('stypy_varargs_param_name', None)
        csr_matrix.__getitem__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        csr_matrix.__getitem__.__dict__.__setitem__('stypy_call_defaults', defaults)
        csr_matrix.__getitem__.__dict__.__setitem__('stypy_call_varargs', varargs)
        csr_matrix.__getitem__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        csr_matrix.__getitem__.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'csr_matrix.__getitem__', ['key'], None, None, defaults, varargs, kwargs)

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


        @norecursion
        def asindices(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function 'asindices'
            module_type_store = module_type_store.open_function_context('asindices', 237, 8, False)
            
            # Passed parameters checking function
            asindices.stypy_localization = localization
            asindices.stypy_type_of_self = None
            asindices.stypy_type_store = module_type_store
            asindices.stypy_function_name = 'asindices'
            asindices.stypy_param_names_list = ['x']
            asindices.stypy_varargs_param_name = None
            asindices.stypy_kwargs_param_name = None
            asindices.stypy_call_defaults = defaults
            asindices.stypy_call_varargs = varargs
            asindices.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, 'asindices', ['x'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, 'asindices', localization, ['x'], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of 'asindices(...)' code ##################

            
            
            # SSA begins for try-except statement (line 238)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
            
            # Assigning a Call to a Name (line 239):
            
            # Assigning a Call to a Name (line 239):
            
            # Call to asarray(...): (line 239)
            # Processing the call arguments (line 239)
            # Getting the type of 'x' (line 239)
            x_370795 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 239, 31), 'x', False)
            # Processing the call keyword arguments (line 239)
            kwargs_370796 = {}
            # Getting the type of 'np' (line 239)
            np_370793 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 239, 20), 'np', False)
            # Obtaining the member 'asarray' of a type (line 239)
            asarray_370794 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 239, 20), np_370793, 'asarray')
            # Calling asarray(args, kwargs) (line 239)
            asarray_call_result_370797 = invoke(stypy.reporting.localization.Localization(__file__, 239, 20), asarray_370794, *[x_370795], **kwargs_370796)
            
            # Assigning a type to the variable 'x' (line 239)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 239, 16), 'x', asarray_call_result_370797)
            
            # Assigning a Call to a Name (line 242):
            
            # Assigning a Call to a Name (line 242):
            
            # Call to get_index_dtype(...): (line 242)
            # Processing the call arguments (line 242)
            
            # Obtaining an instance of the builtin type 'tuple' (line 242)
            tuple_370799 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 242, 45), 'tuple')
            # Adding type elements to the builtin type 'tuple' instance (line 242)
            # Adding element type (line 242)
            # Getting the type of 'x' (line 242)
            x_370800 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 242, 45), 'x', False)
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 242, 45), tuple_370799, x_370800)
            
            # Processing the call keyword arguments (line 242)
            # Getting the type of 'True' (line 242)
            True_370801 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 242, 65), 'True', False)
            keyword_370802 = True_370801
            kwargs_370803 = {'check_contents': keyword_370802}
            # Getting the type of 'get_index_dtype' (line 242)
            get_index_dtype_370798 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 242, 28), 'get_index_dtype', False)
            # Calling get_index_dtype(args, kwargs) (line 242)
            get_index_dtype_call_result_370804 = invoke(stypy.reporting.localization.Localization(__file__, 242, 28), get_index_dtype_370798, *[tuple_370799], **kwargs_370803)
            
            # Assigning a type to the variable 'idx_dtype' (line 242)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 242, 16), 'idx_dtype', get_index_dtype_call_result_370804)
            
            
            # Getting the type of 'idx_dtype' (line 243)
            idx_dtype_370805 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 243, 19), 'idx_dtype')
            # Getting the type of 'x' (line 243)
            x_370806 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 243, 32), 'x')
            # Obtaining the member 'dtype' of a type (line 243)
            dtype_370807 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 243, 32), x_370806, 'dtype')
            # Applying the binary operator '!=' (line 243)
            result_ne_370808 = python_operator(stypy.reporting.localization.Localization(__file__, 243, 19), '!=', idx_dtype_370805, dtype_370807)
            
            # Testing the type of an if condition (line 243)
            if_condition_370809 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 243, 16), result_ne_370808)
            # Assigning a type to the variable 'if_condition_370809' (line 243)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 243, 16), 'if_condition_370809', if_condition_370809)
            # SSA begins for if statement (line 243)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a Call to a Name (line 244):
            
            # Assigning a Call to a Name (line 244):
            
            # Call to astype(...): (line 244)
            # Processing the call arguments (line 244)
            # Getting the type of 'idx_dtype' (line 244)
            idx_dtype_370812 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 244, 33), 'idx_dtype', False)
            # Processing the call keyword arguments (line 244)
            kwargs_370813 = {}
            # Getting the type of 'x' (line 244)
            x_370810 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 244, 24), 'x', False)
            # Obtaining the member 'astype' of a type (line 244)
            astype_370811 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 244, 24), x_370810, 'astype')
            # Calling astype(args, kwargs) (line 244)
            astype_call_result_370814 = invoke(stypy.reporting.localization.Localization(__file__, 244, 24), astype_370811, *[idx_dtype_370812], **kwargs_370813)
            
            # Assigning a type to the variable 'x' (line 244)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 244, 20), 'x', astype_call_result_370814)
            # SSA join for if statement (line 243)
            module_type_store = module_type_store.join_ssa_context()
            
            # SSA branch for the except part of a try statement (line 238)
            # SSA branch for the except '<any exception>' branch of a try statement (line 238)
            module_type_store.open_ssa_branch('except')
            
            # Call to IndexError(...): (line 246)
            # Processing the call arguments (line 246)
            str_370816 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 246, 33), 'str', 'invalid index')
            # Processing the call keyword arguments (line 246)
            kwargs_370817 = {}
            # Getting the type of 'IndexError' (line 246)
            IndexError_370815 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 22), 'IndexError', False)
            # Calling IndexError(args, kwargs) (line 246)
            IndexError_call_result_370818 = invoke(stypy.reporting.localization.Localization(__file__, 246, 22), IndexError_370815, *[str_370816], **kwargs_370817)
            
            ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 246, 16), IndexError_call_result_370818, 'raise parameter', BaseException)
            # SSA branch for the else branch of a try statement (line 238)
            module_type_store.open_ssa_branch('except else')
            # Getting the type of 'x' (line 248)
            x_370819 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 248, 23), 'x')
            # Assigning a type to the variable 'stypy_return_type' (line 248)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 248, 16), 'stypy_return_type', x_370819)
            # SSA join for try-except statement (line 238)
            module_type_store = module_type_store.join_ssa_context()
            
            
            # ################# End of 'asindices(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function 'asindices' in the type store
            # Getting the type of 'stypy_return_type' (line 237)
            stypy_return_type_370820 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 237, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_370820)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function 'asindices'
            return stypy_return_type_370820

        # Assigning a type to the variable 'asindices' (line 237)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 237, 8), 'asindices', asindices)

        @norecursion
        def check_bounds(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function 'check_bounds'
            module_type_store = module_type_store.open_function_context('check_bounds', 250, 8, False)
            
            # Passed parameters checking function
            check_bounds.stypy_localization = localization
            check_bounds.stypy_type_of_self = None
            check_bounds.stypy_type_store = module_type_store
            check_bounds.stypy_function_name = 'check_bounds'
            check_bounds.stypy_param_names_list = ['indices', 'N']
            check_bounds.stypy_varargs_param_name = None
            check_bounds.stypy_kwargs_param_name = None
            check_bounds.stypy_call_defaults = defaults
            check_bounds.stypy_call_varargs = varargs
            check_bounds.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, 'check_bounds', ['indices', 'N'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, 'check_bounds', localization, ['indices', 'N'], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of 'check_bounds(...)' code ##################

            
            
            # Getting the type of 'indices' (line 251)
            indices_370821 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 251, 15), 'indices')
            # Obtaining the member 'size' of a type (line 251)
            size_370822 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 251, 15), indices_370821, 'size')
            int_370823 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 251, 31), 'int')
            # Applying the binary operator '==' (line 251)
            result_eq_370824 = python_operator(stypy.reporting.localization.Localization(__file__, 251, 15), '==', size_370822, int_370823)
            
            # Testing the type of an if condition (line 251)
            if_condition_370825 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 251, 12), result_eq_370824)
            # Assigning a type to the variable 'if_condition_370825' (line 251)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 251, 12), 'if_condition_370825', if_condition_370825)
            # SSA begins for if statement (line 251)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Obtaining an instance of the builtin type 'tuple' (line 252)
            tuple_370826 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 252, 24), 'tuple')
            # Adding type elements to the builtin type 'tuple' instance (line 252)
            # Adding element type (line 252)
            int_370827 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 252, 24), 'int')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 252, 24), tuple_370826, int_370827)
            # Adding element type (line 252)
            int_370828 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 252, 27), 'int')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 252, 24), tuple_370826, int_370828)
            
            # Assigning a type to the variable 'stypy_return_type' (line 252)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 252, 16), 'stypy_return_type', tuple_370826)
            # SSA join for if statement (line 251)
            module_type_store = module_type_store.join_ssa_context()
            
            
            # Assigning a Call to a Name (line 254):
            
            # Assigning a Call to a Name (line 254):
            
            # Call to max(...): (line 254)
            # Processing the call keyword arguments (line 254)
            kwargs_370831 = {}
            # Getting the type of 'indices' (line 254)
            indices_370829 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 254, 23), 'indices', False)
            # Obtaining the member 'max' of a type (line 254)
            max_370830 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 254, 23), indices_370829, 'max')
            # Calling max(args, kwargs) (line 254)
            max_call_result_370832 = invoke(stypy.reporting.localization.Localization(__file__, 254, 23), max_370830, *[], **kwargs_370831)
            
            # Assigning a type to the variable 'max_indx' (line 254)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 254, 12), 'max_indx', max_call_result_370832)
            
            
            # Getting the type of 'max_indx' (line 255)
            max_indx_370833 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 255, 15), 'max_indx')
            # Getting the type of 'N' (line 255)
            N_370834 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 255, 27), 'N')
            # Applying the binary operator '>=' (line 255)
            result_ge_370835 = python_operator(stypy.reporting.localization.Localization(__file__, 255, 15), '>=', max_indx_370833, N_370834)
            
            # Testing the type of an if condition (line 255)
            if_condition_370836 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 255, 12), result_ge_370835)
            # Assigning a type to the variable 'if_condition_370836' (line 255)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 255, 12), 'if_condition_370836', if_condition_370836)
            # SSA begins for if statement (line 255)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to IndexError(...): (line 256)
            # Processing the call arguments (line 256)
            str_370838 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 256, 33), 'str', 'index (%d) out of range')
            # Getting the type of 'max_indx' (line 256)
            max_indx_370839 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 256, 61), 'max_indx', False)
            # Applying the binary operator '%' (line 256)
            result_mod_370840 = python_operator(stypy.reporting.localization.Localization(__file__, 256, 33), '%', str_370838, max_indx_370839)
            
            # Processing the call keyword arguments (line 256)
            kwargs_370841 = {}
            # Getting the type of 'IndexError' (line 256)
            IndexError_370837 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 256, 22), 'IndexError', False)
            # Calling IndexError(args, kwargs) (line 256)
            IndexError_call_result_370842 = invoke(stypy.reporting.localization.Localization(__file__, 256, 22), IndexError_370837, *[result_mod_370840], **kwargs_370841)
            
            ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 256, 16), IndexError_call_result_370842, 'raise parameter', BaseException)
            # SSA join for if statement (line 255)
            module_type_store = module_type_store.join_ssa_context()
            
            
            # Assigning a Call to a Name (line 258):
            
            # Assigning a Call to a Name (line 258):
            
            # Call to min(...): (line 258)
            # Processing the call keyword arguments (line 258)
            kwargs_370845 = {}
            # Getting the type of 'indices' (line 258)
            indices_370843 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 258, 23), 'indices', False)
            # Obtaining the member 'min' of a type (line 258)
            min_370844 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 258, 23), indices_370843, 'min')
            # Calling min(args, kwargs) (line 258)
            min_call_result_370846 = invoke(stypy.reporting.localization.Localization(__file__, 258, 23), min_370844, *[], **kwargs_370845)
            
            # Assigning a type to the variable 'min_indx' (line 258)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 258, 12), 'min_indx', min_call_result_370846)
            
            
            # Getting the type of 'min_indx' (line 259)
            min_indx_370847 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 259, 15), 'min_indx')
            
            # Getting the type of 'N' (line 259)
            N_370848 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 259, 27), 'N')
            # Applying the 'usub' unary operator (line 259)
            result___neg___370849 = python_operator(stypy.reporting.localization.Localization(__file__, 259, 26), 'usub', N_370848)
            
            # Applying the binary operator '<' (line 259)
            result_lt_370850 = python_operator(stypy.reporting.localization.Localization(__file__, 259, 15), '<', min_indx_370847, result___neg___370849)
            
            # Testing the type of an if condition (line 259)
            if_condition_370851 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 259, 12), result_lt_370850)
            # Assigning a type to the variable 'if_condition_370851' (line 259)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 259, 12), 'if_condition_370851', if_condition_370851)
            # SSA begins for if statement (line 259)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to IndexError(...): (line 260)
            # Processing the call arguments (line 260)
            str_370853 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 260, 33), 'str', 'index (%d) out of range')
            # Getting the type of 'N' (line 260)
            N_370854 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 260, 62), 'N', False)
            # Getting the type of 'min_indx' (line 260)
            min_indx_370855 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 260, 66), 'min_indx', False)
            # Applying the binary operator '+' (line 260)
            result_add_370856 = python_operator(stypy.reporting.localization.Localization(__file__, 260, 62), '+', N_370854, min_indx_370855)
            
            # Applying the binary operator '%' (line 260)
            result_mod_370857 = python_operator(stypy.reporting.localization.Localization(__file__, 260, 33), '%', str_370853, result_add_370856)
            
            # Processing the call keyword arguments (line 260)
            kwargs_370858 = {}
            # Getting the type of 'IndexError' (line 260)
            IndexError_370852 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 260, 22), 'IndexError', False)
            # Calling IndexError(args, kwargs) (line 260)
            IndexError_call_result_370859 = invoke(stypy.reporting.localization.Localization(__file__, 260, 22), IndexError_370852, *[result_mod_370857], **kwargs_370858)
            
            ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 260, 16), IndexError_call_result_370859, 'raise parameter', BaseException)
            # SSA join for if statement (line 259)
            module_type_store = module_type_store.join_ssa_context()
            
            
            # Obtaining an instance of the builtin type 'tuple' (line 262)
            tuple_370860 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 262, 19), 'tuple')
            # Adding type elements to the builtin type 'tuple' instance (line 262)
            # Adding element type (line 262)
            # Getting the type of 'min_indx' (line 262)
            min_indx_370861 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 262, 19), 'min_indx')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 262, 19), tuple_370860, min_indx_370861)
            # Adding element type (line 262)
            # Getting the type of 'max_indx' (line 262)
            max_indx_370862 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 262, 29), 'max_indx')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 262, 19), tuple_370860, max_indx_370862)
            
            # Assigning a type to the variable 'stypy_return_type' (line 262)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 262, 12), 'stypy_return_type', tuple_370860)
            
            # ################# End of 'check_bounds(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function 'check_bounds' in the type store
            # Getting the type of 'stypy_return_type' (line 250)
            stypy_return_type_370863 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 250, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_370863)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function 'check_bounds'
            return stypy_return_type_370863

        # Assigning a type to the variable 'check_bounds' (line 250)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 250, 8), 'check_bounds', check_bounds)

        @norecursion
        def extractor(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function 'extractor'
            module_type_store = module_type_store.open_function_context('extractor', 264, 8, False)
            
            # Passed parameters checking function
            extractor.stypy_localization = localization
            extractor.stypy_type_of_self = None
            extractor.stypy_type_store = module_type_store
            extractor.stypy_function_name = 'extractor'
            extractor.stypy_param_names_list = ['indices', 'N']
            extractor.stypy_varargs_param_name = None
            extractor.stypy_kwargs_param_name = None
            extractor.stypy_call_defaults = defaults
            extractor.stypy_call_varargs = varargs
            extractor.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, 'extractor', ['indices', 'N'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, 'extractor', localization, ['indices', 'N'], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of 'extractor(...)' code ##################

            str_370864 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 267, (-1)), 'str', 'Return a sparse matrix P so that P*self implements\n            slicing of the form self[[1,2,3],:]\n            ')
            
            # Assigning a Call to a Name (line 268):
            
            # Assigning a Call to a Name (line 268):
            
            # Call to copy(...): (line 268)
            # Processing the call keyword arguments (line 268)
            kwargs_370870 = {}
            
            # Call to asindices(...): (line 268)
            # Processing the call arguments (line 268)
            # Getting the type of 'indices' (line 268)
            indices_370866 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 268, 32), 'indices', False)
            # Processing the call keyword arguments (line 268)
            kwargs_370867 = {}
            # Getting the type of 'asindices' (line 268)
            asindices_370865 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 268, 22), 'asindices', False)
            # Calling asindices(args, kwargs) (line 268)
            asindices_call_result_370868 = invoke(stypy.reporting.localization.Localization(__file__, 268, 22), asindices_370865, *[indices_370866], **kwargs_370867)
            
            # Obtaining the member 'copy' of a type (line 268)
            copy_370869 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 268, 22), asindices_call_result_370868, 'copy')
            # Calling copy(args, kwargs) (line 268)
            copy_call_result_370871 = invoke(stypy.reporting.localization.Localization(__file__, 268, 22), copy_370869, *[], **kwargs_370870)
            
            # Assigning a type to the variable 'indices' (line 268)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 268, 12), 'indices', copy_call_result_370871)
            
            # Assigning a Call to a Tuple (line 270):
            
            # Assigning a Subscript to a Name (line 270):
            
            # Obtaining the type of the subscript
            int_370872 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 270, 12), 'int')
            
            # Call to check_bounds(...): (line 270)
            # Processing the call arguments (line 270)
            # Getting the type of 'indices' (line 270)
            indices_370874 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 270, 46), 'indices', False)
            # Getting the type of 'N' (line 270)
            N_370875 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 270, 55), 'N', False)
            # Processing the call keyword arguments (line 270)
            kwargs_370876 = {}
            # Getting the type of 'check_bounds' (line 270)
            check_bounds_370873 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 270, 33), 'check_bounds', False)
            # Calling check_bounds(args, kwargs) (line 270)
            check_bounds_call_result_370877 = invoke(stypy.reporting.localization.Localization(__file__, 270, 33), check_bounds_370873, *[indices_370874, N_370875], **kwargs_370876)
            
            # Obtaining the member '__getitem__' of a type (line 270)
            getitem___370878 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 270, 12), check_bounds_call_result_370877, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 270)
            subscript_call_result_370879 = invoke(stypy.reporting.localization.Localization(__file__, 270, 12), getitem___370878, int_370872)
            
            # Assigning a type to the variable 'tuple_var_assignment_370329' (line 270)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 270, 12), 'tuple_var_assignment_370329', subscript_call_result_370879)
            
            # Assigning a Subscript to a Name (line 270):
            
            # Obtaining the type of the subscript
            int_370880 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 270, 12), 'int')
            
            # Call to check_bounds(...): (line 270)
            # Processing the call arguments (line 270)
            # Getting the type of 'indices' (line 270)
            indices_370882 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 270, 46), 'indices', False)
            # Getting the type of 'N' (line 270)
            N_370883 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 270, 55), 'N', False)
            # Processing the call keyword arguments (line 270)
            kwargs_370884 = {}
            # Getting the type of 'check_bounds' (line 270)
            check_bounds_370881 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 270, 33), 'check_bounds', False)
            # Calling check_bounds(args, kwargs) (line 270)
            check_bounds_call_result_370885 = invoke(stypy.reporting.localization.Localization(__file__, 270, 33), check_bounds_370881, *[indices_370882, N_370883], **kwargs_370884)
            
            # Obtaining the member '__getitem__' of a type (line 270)
            getitem___370886 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 270, 12), check_bounds_call_result_370885, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 270)
            subscript_call_result_370887 = invoke(stypy.reporting.localization.Localization(__file__, 270, 12), getitem___370886, int_370880)
            
            # Assigning a type to the variable 'tuple_var_assignment_370330' (line 270)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 270, 12), 'tuple_var_assignment_370330', subscript_call_result_370887)
            
            # Assigning a Name to a Name (line 270):
            # Getting the type of 'tuple_var_assignment_370329' (line 270)
            tuple_var_assignment_370329_370888 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 270, 12), 'tuple_var_assignment_370329')
            # Assigning a type to the variable 'min_indx' (line 270)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 270, 12), 'min_indx', tuple_var_assignment_370329_370888)
            
            # Assigning a Name to a Name (line 270):
            # Getting the type of 'tuple_var_assignment_370330' (line 270)
            tuple_var_assignment_370330_370889 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 270, 12), 'tuple_var_assignment_370330')
            # Assigning a type to the variable 'max_indx' (line 270)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 270, 22), 'max_indx', tuple_var_assignment_370330_370889)
            
            
            # Getting the type of 'min_indx' (line 272)
            min_indx_370890 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 272, 15), 'min_indx')
            int_370891 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 272, 26), 'int')
            # Applying the binary operator '<' (line 272)
            result_lt_370892 = python_operator(stypy.reporting.localization.Localization(__file__, 272, 15), '<', min_indx_370890, int_370891)
            
            # Testing the type of an if condition (line 272)
            if_condition_370893 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 272, 12), result_lt_370892)
            # Assigning a type to the variable 'if_condition_370893' (line 272)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 272, 12), 'if_condition_370893', if_condition_370893)
            # SSA begins for if statement (line 272)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Getting the type of 'indices' (line 273)
            indices_370894 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 273, 16), 'indices')
            
            # Obtaining the type of the subscript
            
            # Getting the type of 'indices' (line 273)
            indices_370895 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 273, 24), 'indices')
            int_370896 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 273, 34), 'int')
            # Applying the binary operator '<' (line 273)
            result_lt_370897 = python_operator(stypy.reporting.localization.Localization(__file__, 273, 24), '<', indices_370895, int_370896)
            
            # Getting the type of 'indices' (line 273)
            indices_370898 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 273, 16), 'indices')
            # Obtaining the member '__getitem__' of a type (line 273)
            getitem___370899 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 273, 16), indices_370898, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 273)
            subscript_call_result_370900 = invoke(stypy.reporting.localization.Localization(__file__, 273, 16), getitem___370899, result_lt_370897)
            
            # Getting the type of 'N' (line 273)
            N_370901 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 273, 40), 'N')
            # Applying the binary operator '+=' (line 273)
            result_iadd_370902 = python_operator(stypy.reporting.localization.Localization(__file__, 273, 16), '+=', subscript_call_result_370900, N_370901)
            # Getting the type of 'indices' (line 273)
            indices_370903 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 273, 16), 'indices')
            
            # Getting the type of 'indices' (line 273)
            indices_370904 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 273, 24), 'indices')
            int_370905 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 273, 34), 'int')
            # Applying the binary operator '<' (line 273)
            result_lt_370906 = python_operator(stypy.reporting.localization.Localization(__file__, 273, 24), '<', indices_370904, int_370905)
            
            # Storing an element on a container (line 273)
            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 273, 16), indices_370903, (result_lt_370906, result_iadd_370902))
            
            # SSA join for if statement (line 272)
            module_type_store = module_type_store.join_ssa_context()
            
            
            # Assigning a Call to a Name (line 275):
            
            # Assigning a Call to a Name (line 275):
            
            # Call to arange(...): (line 275)
            # Processing the call arguments (line 275)
            
            # Call to len(...): (line 275)
            # Processing the call arguments (line 275)
            # Getting the type of 'indices' (line 275)
            indices_370910 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 275, 35), 'indices', False)
            # Processing the call keyword arguments (line 275)
            kwargs_370911 = {}
            # Getting the type of 'len' (line 275)
            len_370909 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 275, 31), 'len', False)
            # Calling len(args, kwargs) (line 275)
            len_call_result_370912 = invoke(stypy.reporting.localization.Localization(__file__, 275, 31), len_370909, *[indices_370910], **kwargs_370911)
            
            int_370913 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 275, 44), 'int')
            # Applying the binary operator '+' (line 275)
            result_add_370914 = python_operator(stypy.reporting.localization.Localization(__file__, 275, 31), '+', len_call_result_370912, int_370913)
            
            # Processing the call keyword arguments (line 275)
            # Getting the type of 'indices' (line 275)
            indices_370915 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 275, 53), 'indices', False)
            # Obtaining the member 'dtype' of a type (line 275)
            dtype_370916 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 275, 53), indices_370915, 'dtype')
            keyword_370917 = dtype_370916
            kwargs_370918 = {'dtype': keyword_370917}
            # Getting the type of 'np' (line 275)
            np_370907 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 275, 21), 'np', False)
            # Obtaining the member 'arange' of a type (line 275)
            arange_370908 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 275, 21), np_370907, 'arange')
            # Calling arange(args, kwargs) (line 275)
            arange_call_result_370919 = invoke(stypy.reporting.localization.Localization(__file__, 275, 21), arange_370908, *[result_add_370914], **kwargs_370918)
            
            # Assigning a type to the variable 'indptr' (line 275)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 275, 12), 'indptr', arange_call_result_370919)
            
            # Assigning a Call to a Name (line 276):
            
            # Assigning a Call to a Name (line 276):
            
            # Call to ones(...): (line 276)
            # Processing the call arguments (line 276)
            
            # Call to len(...): (line 276)
            # Processing the call arguments (line 276)
            # Getting the type of 'indices' (line 276)
            indices_370923 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 276, 31), 'indices', False)
            # Processing the call keyword arguments (line 276)
            kwargs_370924 = {}
            # Getting the type of 'len' (line 276)
            len_370922 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 276, 27), 'len', False)
            # Calling len(args, kwargs) (line 276)
            len_call_result_370925 = invoke(stypy.reporting.localization.Localization(__file__, 276, 27), len_370922, *[indices_370923], **kwargs_370924)
            
            # Processing the call keyword arguments (line 276)
            # Getting the type of 'self' (line 276)
            self_370926 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 276, 47), 'self', False)
            # Obtaining the member 'dtype' of a type (line 276)
            dtype_370927 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 276, 47), self_370926, 'dtype')
            keyword_370928 = dtype_370927
            kwargs_370929 = {'dtype': keyword_370928}
            # Getting the type of 'np' (line 276)
            np_370920 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 276, 19), 'np', False)
            # Obtaining the member 'ones' of a type (line 276)
            ones_370921 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 276, 19), np_370920, 'ones')
            # Calling ones(args, kwargs) (line 276)
            ones_call_result_370930 = invoke(stypy.reporting.localization.Localization(__file__, 276, 19), ones_370921, *[len_call_result_370925], **kwargs_370929)
            
            # Assigning a type to the variable 'data' (line 276)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 276, 12), 'data', ones_call_result_370930)
            
            # Assigning a Tuple to a Name (line 277):
            
            # Assigning a Tuple to a Name (line 277):
            
            # Obtaining an instance of the builtin type 'tuple' (line 277)
            tuple_370931 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 277, 21), 'tuple')
            # Adding type elements to the builtin type 'tuple' instance (line 277)
            # Adding element type (line 277)
            
            # Call to len(...): (line 277)
            # Processing the call arguments (line 277)
            # Getting the type of 'indices' (line 277)
            indices_370933 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 277, 25), 'indices', False)
            # Processing the call keyword arguments (line 277)
            kwargs_370934 = {}
            # Getting the type of 'len' (line 277)
            len_370932 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 277, 21), 'len', False)
            # Calling len(args, kwargs) (line 277)
            len_call_result_370935 = invoke(stypy.reporting.localization.Localization(__file__, 277, 21), len_370932, *[indices_370933], **kwargs_370934)
            
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 277, 21), tuple_370931, len_call_result_370935)
            # Adding element type (line 277)
            # Getting the type of 'N' (line 277)
            N_370936 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 277, 34), 'N')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 277, 21), tuple_370931, N_370936)
            
            # Assigning a type to the variable 'shape' (line 277)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 277, 12), 'shape', tuple_370931)
            
            # Call to csr_matrix(...): (line 279)
            # Processing the call arguments (line 279)
            
            # Obtaining an instance of the builtin type 'tuple' (line 279)
            tuple_370938 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 279, 31), 'tuple')
            # Adding type elements to the builtin type 'tuple' instance (line 279)
            # Adding element type (line 279)
            # Getting the type of 'data' (line 279)
            data_370939 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 279, 31), 'data', False)
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 279, 31), tuple_370938, data_370939)
            # Adding element type (line 279)
            # Getting the type of 'indices' (line 279)
            indices_370940 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 279, 36), 'indices', False)
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 279, 31), tuple_370938, indices_370940)
            # Adding element type (line 279)
            # Getting the type of 'indptr' (line 279)
            indptr_370941 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 279, 44), 'indptr', False)
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 279, 31), tuple_370938, indptr_370941)
            
            # Processing the call keyword arguments (line 279)
            # Getting the type of 'shape' (line 279)
            shape_370942 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 279, 59), 'shape', False)
            keyword_370943 = shape_370942
            # Getting the type of 'self' (line 280)
            self_370944 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 280, 36), 'self', False)
            # Obtaining the member 'dtype' of a type (line 280)
            dtype_370945 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 280, 36), self_370944, 'dtype')
            keyword_370946 = dtype_370945
            # Getting the type of 'False' (line 280)
            False_370947 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 280, 53), 'False', False)
            keyword_370948 = False_370947
            kwargs_370949 = {'dtype': keyword_370946, 'shape': keyword_370943, 'copy': keyword_370948}
            # Getting the type of 'csr_matrix' (line 279)
            csr_matrix_370937 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 279, 19), 'csr_matrix', False)
            # Calling csr_matrix(args, kwargs) (line 279)
            csr_matrix_call_result_370950 = invoke(stypy.reporting.localization.Localization(__file__, 279, 19), csr_matrix_370937, *[tuple_370938], **kwargs_370949)
            
            # Assigning a type to the variable 'stypy_return_type' (line 279)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 279, 12), 'stypy_return_type', csr_matrix_call_result_370950)
            
            # ################# End of 'extractor(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function 'extractor' in the type store
            # Getting the type of 'stypy_return_type' (line 264)
            stypy_return_type_370951 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 264, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_370951)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function 'extractor'
            return stypy_return_type_370951

        # Assigning a type to the variable 'extractor' (line 264)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 264, 8), 'extractor', extractor)
        
        # Assigning a Call to a Tuple (line 282):
        
        # Assigning a Subscript to a Name (line 282):
        
        # Obtaining the type of the subscript
        int_370952 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 282, 8), 'int')
        
        # Call to _unpack_index(...): (line 282)
        # Processing the call arguments (line 282)
        # Getting the type of 'key' (line 282)
        key_370955 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 282, 38), 'key', False)
        # Processing the call keyword arguments (line 282)
        kwargs_370956 = {}
        # Getting the type of 'self' (line 282)
        self_370953 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 282, 19), 'self', False)
        # Obtaining the member '_unpack_index' of a type (line 282)
        _unpack_index_370954 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 282, 19), self_370953, '_unpack_index')
        # Calling _unpack_index(args, kwargs) (line 282)
        _unpack_index_call_result_370957 = invoke(stypy.reporting.localization.Localization(__file__, 282, 19), _unpack_index_370954, *[key_370955], **kwargs_370956)
        
        # Obtaining the member '__getitem__' of a type (line 282)
        getitem___370958 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 282, 8), _unpack_index_call_result_370957, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 282)
        subscript_call_result_370959 = invoke(stypy.reporting.localization.Localization(__file__, 282, 8), getitem___370958, int_370952)
        
        # Assigning a type to the variable 'tuple_var_assignment_370331' (line 282)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 282, 8), 'tuple_var_assignment_370331', subscript_call_result_370959)
        
        # Assigning a Subscript to a Name (line 282):
        
        # Obtaining the type of the subscript
        int_370960 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 282, 8), 'int')
        
        # Call to _unpack_index(...): (line 282)
        # Processing the call arguments (line 282)
        # Getting the type of 'key' (line 282)
        key_370963 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 282, 38), 'key', False)
        # Processing the call keyword arguments (line 282)
        kwargs_370964 = {}
        # Getting the type of 'self' (line 282)
        self_370961 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 282, 19), 'self', False)
        # Obtaining the member '_unpack_index' of a type (line 282)
        _unpack_index_370962 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 282, 19), self_370961, '_unpack_index')
        # Calling _unpack_index(args, kwargs) (line 282)
        _unpack_index_call_result_370965 = invoke(stypy.reporting.localization.Localization(__file__, 282, 19), _unpack_index_370962, *[key_370963], **kwargs_370964)
        
        # Obtaining the member '__getitem__' of a type (line 282)
        getitem___370966 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 282, 8), _unpack_index_call_result_370965, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 282)
        subscript_call_result_370967 = invoke(stypy.reporting.localization.Localization(__file__, 282, 8), getitem___370966, int_370960)
        
        # Assigning a type to the variable 'tuple_var_assignment_370332' (line 282)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 282, 8), 'tuple_var_assignment_370332', subscript_call_result_370967)
        
        # Assigning a Name to a Name (line 282):
        # Getting the type of 'tuple_var_assignment_370331' (line 282)
        tuple_var_assignment_370331_370968 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 282, 8), 'tuple_var_assignment_370331')
        # Assigning a type to the variable 'row' (line 282)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 282, 8), 'row', tuple_var_assignment_370331_370968)
        
        # Assigning a Name to a Name (line 282):
        # Getting the type of 'tuple_var_assignment_370332' (line 282)
        tuple_var_assignment_370332_370969 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 282, 8), 'tuple_var_assignment_370332')
        # Assigning a type to the variable 'col' (line 282)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 282, 13), 'col', tuple_var_assignment_370332_370969)
        
        
        # Call to isintlike(...): (line 286)
        # Processing the call arguments (line 286)
        # Getting the type of 'row' (line 286)
        row_370971 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 286, 21), 'row', False)
        # Processing the call keyword arguments (line 286)
        kwargs_370972 = {}
        # Getting the type of 'isintlike' (line 286)
        isintlike_370970 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 286, 11), 'isintlike', False)
        # Calling isintlike(args, kwargs) (line 286)
        isintlike_call_result_370973 = invoke(stypy.reporting.localization.Localization(__file__, 286, 11), isintlike_370970, *[row_370971], **kwargs_370972)
        
        # Testing the type of an if condition (line 286)
        if_condition_370974 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 286, 8), isintlike_call_result_370973)
        # Assigning a type to the variable 'if_condition_370974' (line 286)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 286, 8), 'if_condition_370974', if_condition_370974)
        # SSA begins for if statement (line 286)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        
        # Call to isintlike(...): (line 288)
        # Processing the call arguments (line 288)
        # Getting the type of 'col' (line 288)
        col_370976 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 288, 25), 'col', False)
        # Processing the call keyword arguments (line 288)
        kwargs_370977 = {}
        # Getting the type of 'isintlike' (line 288)
        isintlike_370975 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 288, 15), 'isintlike', False)
        # Calling isintlike(args, kwargs) (line 288)
        isintlike_call_result_370978 = invoke(stypy.reporting.localization.Localization(__file__, 288, 15), isintlike_370975, *[col_370976], **kwargs_370977)
        
        # Testing the type of an if condition (line 288)
        if_condition_370979 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 288, 12), isintlike_call_result_370978)
        # Assigning a type to the variable 'if_condition_370979' (line 288)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 288, 12), 'if_condition_370979', if_condition_370979)
        # SSA begins for if statement (line 288)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to _get_single_element(...): (line 289)
        # Processing the call arguments (line 289)
        # Getting the type of 'row' (line 289)
        row_370982 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 289, 48), 'row', False)
        # Getting the type of 'col' (line 289)
        col_370983 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 289, 53), 'col', False)
        # Processing the call keyword arguments (line 289)
        kwargs_370984 = {}
        # Getting the type of 'self' (line 289)
        self_370980 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 289, 23), 'self', False)
        # Obtaining the member '_get_single_element' of a type (line 289)
        _get_single_element_370981 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 289, 23), self_370980, '_get_single_element')
        # Calling _get_single_element(args, kwargs) (line 289)
        _get_single_element_call_result_370985 = invoke(stypy.reporting.localization.Localization(__file__, 289, 23), _get_single_element_370981, *[row_370982, col_370983], **kwargs_370984)
        
        # Assigning a type to the variable 'stypy_return_type' (line 289)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 289, 16), 'stypy_return_type', _get_single_element_call_result_370985)
        # SSA branch for the else part of an if statement (line 288)
        module_type_store.open_ssa_branch('else')
        
        # Type idiom detected: calculating its left and rigth part (line 291)
        # Getting the type of 'slice' (line 291)
        slice_370986 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 291, 33), 'slice')
        # Getting the type of 'col' (line 291)
        col_370987 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 291, 28), 'col')
        
        (may_be_370988, more_types_in_union_370989) = may_be_subtype(slice_370986, col_370987)

        if may_be_370988:

            if more_types_in_union_370989:
                # Runtime conditional SSA (line 291)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            # Assigning a type to the variable 'col' (line 291)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 291, 17), 'col', remove_not_subtype_from_union(col_370987, slice))
            
            # Call to _get_row_slice(...): (line 292)
            # Processing the call arguments (line 292)
            # Getting the type of 'row' (line 292)
            row_370992 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 292, 43), 'row', False)
            # Getting the type of 'col' (line 292)
            col_370993 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 292, 48), 'col', False)
            # Processing the call keyword arguments (line 292)
            kwargs_370994 = {}
            # Getting the type of 'self' (line 292)
            self_370990 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 292, 23), 'self', False)
            # Obtaining the member '_get_row_slice' of a type (line 292)
            _get_row_slice_370991 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 292, 23), self_370990, '_get_row_slice')
            # Calling _get_row_slice(args, kwargs) (line 292)
            _get_row_slice_call_result_370995 = invoke(stypy.reporting.localization.Localization(__file__, 292, 23), _get_row_slice_370991, *[row_370992, col_370993], **kwargs_370994)
            
            # Assigning a type to the variable 'stypy_return_type' (line 292)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 292, 16), 'stypy_return_type', _get_row_slice_call_result_370995)

            if more_types_in_union_370989:
                # Runtime conditional SSA for else branch (line 291)
                module_type_store.open_ssa_branch('idiom else')



        if ((not may_be_370988) or more_types_in_union_370989):
            # Assigning a type to the variable 'col' (line 291)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 291, 17), 'col', remove_subtype_from_union(col_370987, slice))
            
            
            # Call to issequence(...): (line 294)
            # Processing the call arguments (line 294)
            # Getting the type of 'col' (line 294)
            col_370997 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 294, 28), 'col', False)
            # Processing the call keyword arguments (line 294)
            kwargs_370998 = {}
            # Getting the type of 'issequence' (line 294)
            issequence_370996 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 294, 17), 'issequence', False)
            # Calling issequence(args, kwargs) (line 294)
            issequence_call_result_370999 = invoke(stypy.reporting.localization.Localization(__file__, 294, 17), issequence_370996, *[col_370997], **kwargs_370998)
            
            # Testing the type of an if condition (line 294)
            if_condition_371000 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 294, 17), issequence_call_result_370999)
            # Assigning a type to the variable 'if_condition_371000' (line 294)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 294, 17), 'if_condition_371000', if_condition_371000)
            # SSA begins for if statement (line 294)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a Attribute to a Name (line 295):
            
            # Assigning a Attribute to a Name (line 295):
            
            # Call to extractor(...): (line 295)
            # Processing the call arguments (line 295)
            # Getting the type of 'col' (line 295)
            col_371002 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 295, 30), 'col', False)
            
            # Obtaining the type of the subscript
            int_371003 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 295, 45), 'int')
            # Getting the type of 'self' (line 295)
            self_371004 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 295, 34), 'self', False)
            # Obtaining the member 'shape' of a type (line 295)
            shape_371005 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 295, 34), self_371004, 'shape')
            # Obtaining the member '__getitem__' of a type (line 295)
            getitem___371006 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 295, 34), shape_371005, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 295)
            subscript_call_result_371007 = invoke(stypy.reporting.localization.Localization(__file__, 295, 34), getitem___371006, int_371003)
            
            # Processing the call keyword arguments (line 295)
            kwargs_371008 = {}
            # Getting the type of 'extractor' (line 295)
            extractor_371001 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 295, 20), 'extractor', False)
            # Calling extractor(args, kwargs) (line 295)
            extractor_call_result_371009 = invoke(stypy.reporting.localization.Localization(__file__, 295, 20), extractor_371001, *[col_371002, subscript_call_result_371007], **kwargs_371008)
            
            # Obtaining the member 'T' of a type (line 295)
            T_371010 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 295, 20), extractor_call_result_371009, 'T')
            # Assigning a type to the variable 'P' (line 295)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 295, 16), 'P', T_371010)
            
            # Obtaining the type of the subscript
            # Getting the type of 'row' (line 296)
            row_371011 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 296, 28), 'row')
            slice_371012 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 296, 23), None, None, None)
            # Getting the type of 'self' (line 296)
            self_371013 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 296, 23), 'self')
            # Obtaining the member '__getitem__' of a type (line 296)
            getitem___371014 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 296, 23), self_371013, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 296)
            subscript_call_result_371015 = invoke(stypy.reporting.localization.Localization(__file__, 296, 23), getitem___371014, (row_371011, slice_371012))
            
            # Getting the type of 'P' (line 296)
            P_371016 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 296, 38), 'P')
            # Applying the binary operator '*' (line 296)
            result_mul_371017 = python_operator(stypy.reporting.localization.Localization(__file__, 296, 23), '*', subscript_call_result_371015, P_371016)
            
            # Assigning a type to the variable 'stypy_return_type' (line 296)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 296, 16), 'stypy_return_type', result_mul_371017)
            # SSA join for if statement (line 294)
            module_type_store = module_type_store.join_ssa_context()
            

            if (may_be_370988 and more_types_in_union_370989):
                # SSA join for if statement (line 291)
                module_type_store = module_type_store.join_ssa_context()


        
        # SSA join for if statement (line 288)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA branch for the else part of an if statement (line 286)
        module_type_store.open_ssa_branch('else')
        
        # Type idiom detected: calculating its left and rigth part (line 297)
        # Getting the type of 'slice' (line 297)
        slice_371018 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 297, 29), 'slice')
        # Getting the type of 'row' (line 297)
        row_371019 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 297, 24), 'row')
        
        (may_be_371020, more_types_in_union_371021) = may_be_subtype(slice_371018, row_371019)

        if may_be_371020:

            if more_types_in_union_371021:
                # Runtime conditional SSA (line 297)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            # Assigning a type to the variable 'row' (line 297)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 297, 13), 'row', remove_not_subtype_from_union(row_371019, slice))
            
            
            # Evaluating a boolean operation
            
            # Evaluating a boolean operation
            
            # Call to isintlike(...): (line 299)
            # Processing the call arguments (line 299)
            # Getting the type of 'col' (line 299)
            col_371023 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 299, 27), 'col', False)
            # Processing the call keyword arguments (line 299)
            kwargs_371024 = {}
            # Getting the type of 'isintlike' (line 299)
            isintlike_371022 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 299, 17), 'isintlike', False)
            # Calling isintlike(args, kwargs) (line 299)
            isintlike_call_result_371025 = invoke(stypy.reporting.localization.Localization(__file__, 299, 17), isintlike_371022, *[col_371023], **kwargs_371024)
            
            
            # Getting the type of 'row' (line 299)
            row_371026 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 299, 36), 'row')
            # Obtaining the member 'step' of a type (line 299)
            step_371027 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 299, 36), row_371026, 'step')
            
            # Obtaining an instance of the builtin type 'tuple' (line 299)
            tuple_371028 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 299, 49), 'tuple')
            # Adding type elements to the builtin type 'tuple' instance (line 299)
            # Adding element type (line 299)
            int_371029 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 299, 49), 'int')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 299, 49), tuple_371028, int_371029)
            # Adding element type (line 299)
            # Getting the type of 'None' (line 299)
            None_371030 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 299, 52), 'None')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 299, 49), tuple_371028, None_371030)
            
            # Applying the binary operator 'in' (line 299)
            result_contains_371031 = python_operator(stypy.reporting.localization.Localization(__file__, 299, 36), 'in', step_371027, tuple_371028)
            
            # Applying the binary operator 'and' (line 299)
            result_and_keyword_371032 = python_operator(stypy.reporting.localization.Localization(__file__, 299, 17), 'and', isintlike_call_result_371025, result_contains_371031)
            
            
            # Evaluating a boolean operation
            
            # Call to isinstance(...): (line 300)
            # Processing the call arguments (line 300)
            # Getting the type of 'col' (line 300)
            col_371034 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 300, 32), 'col', False)
            # Getting the type of 'slice' (line 300)
            slice_371035 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 300, 37), 'slice', False)
            # Processing the call keyword arguments (line 300)
            kwargs_371036 = {}
            # Getting the type of 'isinstance' (line 300)
            isinstance_371033 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 300, 21), 'isinstance', False)
            # Calling isinstance(args, kwargs) (line 300)
            isinstance_call_result_371037 = invoke(stypy.reporting.localization.Localization(__file__, 300, 21), isinstance_371033, *[col_371034, slice_371035], **kwargs_371036)
            
            
            # Getting the type of 'col' (line 301)
            col_371038 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 301, 21), 'col')
            # Obtaining the member 'step' of a type (line 301)
            step_371039 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 301, 21), col_371038, 'step')
            
            # Obtaining an instance of the builtin type 'tuple' (line 301)
            tuple_371040 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 301, 34), 'tuple')
            # Adding type elements to the builtin type 'tuple' instance (line 301)
            # Adding element type (line 301)
            int_371041 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 301, 34), 'int')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 301, 34), tuple_371040, int_371041)
            # Adding element type (line 301)
            # Getting the type of 'None' (line 301)
            None_371042 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 301, 37), 'None')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 301, 34), tuple_371040, None_371042)
            
            # Applying the binary operator 'in' (line 301)
            result_contains_371043 = python_operator(stypy.reporting.localization.Localization(__file__, 301, 21), 'in', step_371039, tuple_371040)
            
            # Applying the binary operator 'and' (line 300)
            result_and_keyword_371044 = python_operator(stypy.reporting.localization.Localization(__file__, 300, 21), 'and', isinstance_call_result_371037, result_contains_371043)
            
            # Getting the type of 'row' (line 302)
            row_371045 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 302, 21), 'row')
            # Obtaining the member 'step' of a type (line 302)
            step_371046 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 302, 21), row_371045, 'step')
            
            # Obtaining an instance of the builtin type 'tuple' (line 302)
            tuple_371047 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 302, 34), 'tuple')
            # Adding type elements to the builtin type 'tuple' instance (line 302)
            # Adding element type (line 302)
            int_371048 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 302, 34), 'int')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 302, 34), tuple_371047, int_371048)
            # Adding element type (line 302)
            # Getting the type of 'None' (line 302)
            None_371049 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 302, 37), 'None')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 302, 34), tuple_371047, None_371049)
            
            # Applying the binary operator 'in' (line 302)
            result_contains_371050 = python_operator(stypy.reporting.localization.Localization(__file__, 302, 21), 'in', step_371046, tuple_371047)
            
            # Applying the binary operator 'and' (line 300)
            result_and_keyword_371051 = python_operator(stypy.reporting.localization.Localization(__file__, 300, 21), 'and', result_and_keyword_371044, result_contains_371050)
            
            # Applying the binary operator 'or' (line 299)
            result_or_keyword_371052 = python_operator(stypy.reporting.localization.Localization(__file__, 299, 16), 'or', result_and_keyword_371032, result_and_keyword_371051)
            
            # Testing the type of an if condition (line 299)
            if_condition_371053 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 299, 12), result_or_keyword_371052)
            # Assigning a type to the variable 'if_condition_371053' (line 299)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 299, 12), 'if_condition_371053', if_condition_371053)
            # SSA begins for if statement (line 299)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to _get_submatrix(...): (line 304)
            # Processing the call arguments (line 304)
            # Getting the type of 'row' (line 304)
            row_371056 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 304, 43), 'row', False)
            # Getting the type of 'col' (line 304)
            col_371057 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 304, 48), 'col', False)
            # Processing the call keyword arguments (line 304)
            kwargs_371058 = {}
            # Getting the type of 'self' (line 304)
            self_371054 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 304, 23), 'self', False)
            # Obtaining the member '_get_submatrix' of a type (line 304)
            _get_submatrix_371055 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 304, 23), self_371054, '_get_submatrix')
            # Calling _get_submatrix(args, kwargs) (line 304)
            _get_submatrix_call_result_371059 = invoke(stypy.reporting.localization.Localization(__file__, 304, 23), _get_submatrix_371055, *[row_371056, col_371057], **kwargs_371058)
            
            # Assigning a type to the variable 'stypy_return_type' (line 304)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 304, 16), 'stypy_return_type', _get_submatrix_call_result_371059)
            # SSA branch for the else part of an if statement (line 299)
            module_type_store.open_ssa_branch('else')
            
            
            # Call to issequence(...): (line 305)
            # Processing the call arguments (line 305)
            # Getting the type of 'col' (line 305)
            col_371061 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 305, 28), 'col', False)
            # Processing the call keyword arguments (line 305)
            kwargs_371062 = {}
            # Getting the type of 'issequence' (line 305)
            issequence_371060 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 305, 17), 'issequence', False)
            # Calling issequence(args, kwargs) (line 305)
            issequence_call_result_371063 = invoke(stypy.reporting.localization.Localization(__file__, 305, 17), issequence_371060, *[col_371061], **kwargs_371062)
            
            # Testing the type of an if condition (line 305)
            if_condition_371064 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 305, 17), issequence_call_result_371063)
            # Assigning a type to the variable 'if_condition_371064' (line 305)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 305, 17), 'if_condition_371064', if_condition_371064)
            # SSA begins for if statement (line 305)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a Attribute to a Name (line 307):
            
            # Assigning a Attribute to a Name (line 307):
            
            # Call to extractor(...): (line 307)
            # Processing the call arguments (line 307)
            # Getting the type of 'col' (line 307)
            col_371066 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 307, 30), 'col', False)
            
            # Obtaining the type of the subscript
            int_371067 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 307, 45), 'int')
            # Getting the type of 'self' (line 307)
            self_371068 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 307, 34), 'self', False)
            # Obtaining the member 'shape' of a type (line 307)
            shape_371069 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 307, 34), self_371068, 'shape')
            # Obtaining the member '__getitem__' of a type (line 307)
            getitem___371070 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 307, 34), shape_371069, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 307)
            subscript_call_result_371071 = invoke(stypy.reporting.localization.Localization(__file__, 307, 34), getitem___371070, int_371067)
            
            # Processing the call keyword arguments (line 307)
            kwargs_371072 = {}
            # Getting the type of 'extractor' (line 307)
            extractor_371065 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 307, 20), 'extractor', False)
            # Calling extractor(args, kwargs) (line 307)
            extractor_call_result_371073 = invoke(stypy.reporting.localization.Localization(__file__, 307, 20), extractor_371065, *[col_371066, subscript_call_result_371071], **kwargs_371072)
            
            # Obtaining the member 'T' of a type (line 307)
            T_371074 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 307, 20), extractor_call_result_371073, 'T')
            # Assigning a type to the variable 'P' (line 307)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 307, 16), 'P', T_371074)
            
            # Assigning a Name to a Name (line 308):
            
            # Assigning a Name to a Name (line 308):
            # Getting the type of 'self' (line 308)
            self_371075 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 308, 25), 'self')
            # Assigning a type to the variable 'sliced' (line 308)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 308, 16), 'sliced', self_371075)
            
            
            # Getting the type of 'row' (line 309)
            row_371076 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 309, 19), 'row')
            
            # Call to slice(...): (line 309)
            # Processing the call arguments (line 309)
            # Getting the type of 'None' (line 309)
            None_371078 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 309, 32), 'None', False)
            # Getting the type of 'None' (line 309)
            None_371079 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 309, 38), 'None', False)
            # Getting the type of 'None' (line 309)
            None_371080 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 309, 44), 'None', False)
            # Processing the call keyword arguments (line 309)
            kwargs_371081 = {}
            # Getting the type of 'slice' (line 309)
            slice_371077 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 309, 26), 'slice', False)
            # Calling slice(args, kwargs) (line 309)
            slice_call_result_371082 = invoke(stypy.reporting.localization.Localization(__file__, 309, 26), slice_371077, *[None_371078, None_371079, None_371080], **kwargs_371081)
            
            # Applying the binary operator '!=' (line 309)
            result_ne_371083 = python_operator(stypy.reporting.localization.Localization(__file__, 309, 19), '!=', row_371076, slice_call_result_371082)
            
            # Testing the type of an if condition (line 309)
            if_condition_371084 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 309, 16), result_ne_371083)
            # Assigning a type to the variable 'if_condition_371084' (line 309)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 309, 16), 'if_condition_371084', if_condition_371084)
            # SSA begins for if statement (line 309)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a Subscript to a Name (line 310):
            
            # Assigning a Subscript to a Name (line 310):
            
            # Obtaining the type of the subscript
            # Getting the type of 'row' (line 310)
            row_371085 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 310, 36), 'row')
            slice_371086 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 310, 29), None, None, None)
            # Getting the type of 'sliced' (line 310)
            sliced_371087 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 310, 29), 'sliced')
            # Obtaining the member '__getitem__' of a type (line 310)
            getitem___371088 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 310, 29), sliced_371087, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 310)
            subscript_call_result_371089 = invoke(stypy.reporting.localization.Localization(__file__, 310, 29), getitem___371088, (row_371085, slice_371086))
            
            # Assigning a type to the variable 'sliced' (line 310)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 310, 20), 'sliced', subscript_call_result_371089)
            # SSA join for if statement (line 309)
            module_type_store = module_type_store.join_ssa_context()
            
            # Getting the type of 'sliced' (line 311)
            sliced_371090 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 311, 23), 'sliced')
            # Getting the type of 'P' (line 311)
            P_371091 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 311, 32), 'P')
            # Applying the binary operator '*' (line 311)
            result_mul_371092 = python_operator(stypy.reporting.localization.Localization(__file__, 311, 23), '*', sliced_371090, P_371091)
            
            # Assigning a type to the variable 'stypy_return_type' (line 311)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 311, 16), 'stypy_return_type', result_mul_371092)
            # SSA join for if statement (line 305)
            module_type_store = module_type_store.join_ssa_context()
            
            # SSA join for if statement (line 299)
            module_type_store = module_type_store.join_ssa_context()
            

            if more_types_in_union_371021:
                # Runtime conditional SSA for else branch (line 297)
                module_type_store.open_ssa_branch('idiom else')



        if ((not may_be_371020) or more_types_in_union_371021):
            # Assigning a type to the variable 'row' (line 297)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 297, 13), 'row', remove_subtype_from_union(row_371019, slice))
            
            
            # Call to issequence(...): (line 313)
            # Processing the call arguments (line 313)
            # Getting the type of 'row' (line 313)
            row_371094 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 313, 24), 'row', False)
            # Processing the call keyword arguments (line 313)
            kwargs_371095 = {}
            # Getting the type of 'issequence' (line 313)
            issequence_371093 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 313, 13), 'issequence', False)
            # Calling issequence(args, kwargs) (line 313)
            issequence_call_result_371096 = invoke(stypy.reporting.localization.Localization(__file__, 313, 13), issequence_371093, *[row_371094], **kwargs_371095)
            
            # Testing the type of an if condition (line 313)
            if_condition_371097 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 313, 13), issequence_call_result_371096)
            # Assigning a type to the variable 'if_condition_371097' (line 313)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 313, 13), 'if_condition_371097', if_condition_371097)
            # SSA begins for if statement (line 313)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            
            # Evaluating a boolean operation
            
            # Call to isintlike(...): (line 315)
            # Processing the call arguments (line 315)
            # Getting the type of 'col' (line 315)
            col_371099 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 315, 25), 'col', False)
            # Processing the call keyword arguments (line 315)
            kwargs_371100 = {}
            # Getting the type of 'isintlike' (line 315)
            isintlike_371098 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 315, 15), 'isintlike', False)
            # Calling isintlike(args, kwargs) (line 315)
            isintlike_call_result_371101 = invoke(stypy.reporting.localization.Localization(__file__, 315, 15), isintlike_371098, *[col_371099], **kwargs_371100)
            
            
            # Call to isinstance(...): (line 315)
            # Processing the call arguments (line 315)
            # Getting the type of 'col' (line 315)
            col_371103 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 315, 44), 'col', False)
            # Getting the type of 'slice' (line 315)
            slice_371104 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 315, 48), 'slice', False)
            # Processing the call keyword arguments (line 315)
            kwargs_371105 = {}
            # Getting the type of 'isinstance' (line 315)
            isinstance_371102 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 315, 33), 'isinstance', False)
            # Calling isinstance(args, kwargs) (line 315)
            isinstance_call_result_371106 = invoke(stypy.reporting.localization.Localization(__file__, 315, 33), isinstance_371102, *[col_371103, slice_371104], **kwargs_371105)
            
            # Applying the binary operator 'or' (line 315)
            result_or_keyword_371107 = python_operator(stypy.reporting.localization.Localization(__file__, 315, 15), 'or', isintlike_call_result_371101, isinstance_call_result_371106)
            
            # Testing the type of an if condition (line 315)
            if_condition_371108 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 315, 12), result_or_keyword_371107)
            # Assigning a type to the variable 'if_condition_371108' (line 315)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 315, 12), 'if_condition_371108', if_condition_371108)
            # SSA begins for if statement (line 315)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a Call to a Name (line 316):
            
            # Assigning a Call to a Name (line 316):
            
            # Call to extractor(...): (line 316)
            # Processing the call arguments (line 316)
            # Getting the type of 'row' (line 316)
            row_371110 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 316, 30), 'row', False)
            
            # Obtaining the type of the subscript
            int_371111 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 316, 46), 'int')
            # Getting the type of 'self' (line 316)
            self_371112 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 316, 35), 'self', False)
            # Obtaining the member 'shape' of a type (line 316)
            shape_371113 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 316, 35), self_371112, 'shape')
            # Obtaining the member '__getitem__' of a type (line 316)
            getitem___371114 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 316, 35), shape_371113, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 316)
            subscript_call_result_371115 = invoke(stypy.reporting.localization.Localization(__file__, 316, 35), getitem___371114, int_371111)
            
            # Processing the call keyword arguments (line 316)
            kwargs_371116 = {}
            # Getting the type of 'extractor' (line 316)
            extractor_371109 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 316, 20), 'extractor', False)
            # Calling extractor(args, kwargs) (line 316)
            extractor_call_result_371117 = invoke(stypy.reporting.localization.Localization(__file__, 316, 20), extractor_371109, *[row_371110, subscript_call_result_371115], **kwargs_371116)
            
            # Assigning a type to the variable 'P' (line 316)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 316, 16), 'P', extractor_call_result_371117)
            
            # Assigning a BinOp to a Name (line 317):
            
            # Assigning a BinOp to a Name (line 317):
            # Getting the type of 'P' (line 317)
            P_371118 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 317, 28), 'P')
            # Getting the type of 'self' (line 317)
            self_371119 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 317, 32), 'self')
            # Applying the binary operator '*' (line 317)
            result_mul_371120 = python_operator(stypy.reporting.localization.Localization(__file__, 317, 28), '*', P_371118, self_371119)
            
            # Assigning a type to the variable 'extracted' (line 317)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 317, 16), 'extracted', result_mul_371120)
            
            
            # Getting the type of 'col' (line 318)
            col_371121 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 318, 19), 'col')
            
            # Call to slice(...): (line 318)
            # Processing the call arguments (line 318)
            # Getting the type of 'None' (line 318)
            None_371123 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 318, 32), 'None', False)
            # Getting the type of 'None' (line 318)
            None_371124 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 318, 38), 'None', False)
            # Getting the type of 'None' (line 318)
            None_371125 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 318, 44), 'None', False)
            # Processing the call keyword arguments (line 318)
            kwargs_371126 = {}
            # Getting the type of 'slice' (line 318)
            slice_371122 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 318, 26), 'slice', False)
            # Calling slice(args, kwargs) (line 318)
            slice_call_result_371127 = invoke(stypy.reporting.localization.Localization(__file__, 318, 26), slice_371122, *[None_371123, None_371124, None_371125], **kwargs_371126)
            
            # Applying the binary operator '==' (line 318)
            result_eq_371128 = python_operator(stypy.reporting.localization.Localization(__file__, 318, 19), '==', col_371121, slice_call_result_371127)
            
            # Testing the type of an if condition (line 318)
            if_condition_371129 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 318, 16), result_eq_371128)
            # Assigning a type to the variable 'if_condition_371129' (line 318)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 318, 16), 'if_condition_371129', if_condition_371129)
            # SSA begins for if statement (line 318)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            # Getting the type of 'extracted' (line 319)
            extracted_371130 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 319, 27), 'extracted')
            # Assigning a type to the variable 'stypy_return_type' (line 319)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 319, 20), 'stypy_return_type', extracted_371130)
            # SSA branch for the else part of an if statement (line 318)
            module_type_store.open_ssa_branch('else')
            
            # Obtaining the type of the subscript
            slice_371131 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 321, 27), None, None, None)
            # Getting the type of 'col' (line 321)
            col_371132 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 321, 39), 'col')
            # Getting the type of 'extracted' (line 321)
            extracted_371133 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 321, 27), 'extracted')
            # Obtaining the member '__getitem__' of a type (line 321)
            getitem___371134 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 321, 27), extracted_371133, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 321)
            subscript_call_result_371135 = invoke(stypy.reporting.localization.Localization(__file__, 321, 27), getitem___371134, (slice_371131, col_371132))
            
            # Assigning a type to the variable 'stypy_return_type' (line 321)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 321, 20), 'stypy_return_type', subscript_call_result_371135)
            # SSA join for if statement (line 318)
            module_type_store = module_type_store.join_ssa_context()
            
            # SSA join for if statement (line 315)
            module_type_store = module_type_store.join_ssa_context()
            
            # SSA branch for the else part of an if statement (line 313)
            module_type_store.open_ssa_branch('else')
            
            
            # Evaluating a boolean operation
            
            # Call to ismatrix(...): (line 323)
            # Processing the call arguments (line 323)
            # Getting the type of 'row' (line 323)
            row_371137 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 323, 22), 'row', False)
            # Processing the call keyword arguments (line 323)
            kwargs_371138 = {}
            # Getting the type of 'ismatrix' (line 323)
            ismatrix_371136 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 323, 13), 'ismatrix', False)
            # Calling ismatrix(args, kwargs) (line 323)
            ismatrix_call_result_371139 = invoke(stypy.reporting.localization.Localization(__file__, 323, 13), ismatrix_371136, *[row_371137], **kwargs_371138)
            
            
            # Call to issequence(...): (line 323)
            # Processing the call arguments (line 323)
            # Getting the type of 'col' (line 323)
            col_371141 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 323, 42), 'col', False)
            # Processing the call keyword arguments (line 323)
            kwargs_371142 = {}
            # Getting the type of 'issequence' (line 323)
            issequence_371140 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 323, 31), 'issequence', False)
            # Calling issequence(args, kwargs) (line 323)
            issequence_call_result_371143 = invoke(stypy.reporting.localization.Localization(__file__, 323, 31), issequence_371140, *[col_371141], **kwargs_371142)
            
            # Applying the binary operator 'and' (line 323)
            result_and_keyword_371144 = python_operator(stypy.reporting.localization.Localization(__file__, 323, 13), 'and', ismatrix_call_result_371139, issequence_call_result_371143)
            
            # Testing the type of an if condition (line 323)
            if_condition_371145 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 323, 13), result_and_keyword_371144)
            # Assigning a type to the variable 'if_condition_371145' (line 323)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 323, 13), 'if_condition_371145', if_condition_371145)
            # SSA begins for if statement (line 323)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            
            # Evaluating a boolean operation
            
            
            # Call to len(...): (line 324)
            # Processing the call arguments (line 324)
            
            # Obtaining the type of the subscript
            int_371147 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 324, 23), 'int')
            # Getting the type of 'row' (line 324)
            row_371148 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 324, 19), 'row', False)
            # Obtaining the member '__getitem__' of a type (line 324)
            getitem___371149 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 324, 19), row_371148, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 324)
            subscript_call_result_371150 = invoke(stypy.reporting.localization.Localization(__file__, 324, 19), getitem___371149, int_371147)
            
            # Processing the call keyword arguments (line 324)
            kwargs_371151 = {}
            # Getting the type of 'len' (line 324)
            len_371146 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 324, 15), 'len', False)
            # Calling len(args, kwargs) (line 324)
            len_call_result_371152 = invoke(stypy.reporting.localization.Localization(__file__, 324, 15), len_371146, *[subscript_call_result_371150], **kwargs_371151)
            
            int_371153 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 324, 30), 'int')
            # Applying the binary operator '==' (line 324)
            result_eq_371154 = python_operator(stypy.reporting.localization.Localization(__file__, 324, 15), '==', len_call_result_371152, int_371153)
            
            
            # Call to isintlike(...): (line 324)
            # Processing the call arguments (line 324)
            
            # Obtaining the type of the subscript
            int_371156 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 324, 53), 'int')
            
            # Obtaining the type of the subscript
            int_371157 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 324, 50), 'int')
            # Getting the type of 'row' (line 324)
            row_371158 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 324, 46), 'row', False)
            # Obtaining the member '__getitem__' of a type (line 324)
            getitem___371159 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 324, 46), row_371158, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 324)
            subscript_call_result_371160 = invoke(stypy.reporting.localization.Localization(__file__, 324, 46), getitem___371159, int_371157)
            
            # Obtaining the member '__getitem__' of a type (line 324)
            getitem___371161 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 324, 46), subscript_call_result_371160, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 324)
            subscript_call_result_371162 = invoke(stypy.reporting.localization.Localization(__file__, 324, 46), getitem___371161, int_371156)
            
            # Processing the call keyword arguments (line 324)
            kwargs_371163 = {}
            # Getting the type of 'isintlike' (line 324)
            isintlike_371155 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 324, 36), 'isintlike', False)
            # Calling isintlike(args, kwargs) (line 324)
            isintlike_call_result_371164 = invoke(stypy.reporting.localization.Localization(__file__, 324, 36), isintlike_371155, *[subscript_call_result_371162], **kwargs_371163)
            
            # Applying the binary operator 'and' (line 324)
            result_and_keyword_371165 = python_operator(stypy.reporting.localization.Localization(__file__, 324, 15), 'and', result_eq_371154, isintlike_call_result_371164)
            
            # Testing the type of an if condition (line 324)
            if_condition_371166 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 324, 12), result_and_keyword_371165)
            # Assigning a type to the variable 'if_condition_371166' (line 324)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 324, 12), 'if_condition_371166', if_condition_371166)
            # SSA begins for if statement (line 324)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a Call to a Name (line 326):
            
            # Assigning a Call to a Name (line 326):
            
            # Call to asindices(...): (line 326)
            # Processing the call arguments (line 326)
            # Getting the type of 'row' (line 326)
            row_371168 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 326, 32), 'row', False)
            # Processing the call keyword arguments (line 326)
            kwargs_371169 = {}
            # Getting the type of 'asindices' (line 326)
            asindices_371167 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 326, 22), 'asindices', False)
            # Calling asindices(args, kwargs) (line 326)
            asindices_call_result_371170 = invoke(stypy.reporting.localization.Localization(__file__, 326, 22), asindices_371167, *[row_371168], **kwargs_371169)
            
            # Assigning a type to the variable 'row' (line 326)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 326, 16), 'row', asindices_call_result_371170)
            
            # Assigning a Call to a Name (line 327):
            
            # Assigning a Call to a Name (line 327):
            
            # Call to extractor(...): (line 327)
            # Processing the call arguments (line 327)
            
            # Obtaining the type of the subscript
            slice_371172 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 327, 34), None, None, None)
            int_371173 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 327, 40), 'int')
            # Getting the type of 'row' (line 327)
            row_371174 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 327, 34), 'row', False)
            # Obtaining the member '__getitem__' of a type (line 327)
            getitem___371175 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 327, 34), row_371174, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 327)
            subscript_call_result_371176 = invoke(stypy.reporting.localization.Localization(__file__, 327, 34), getitem___371175, (slice_371172, int_371173))
            
            
            # Obtaining the type of the subscript
            int_371177 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 327, 55), 'int')
            # Getting the type of 'self' (line 327)
            self_371178 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 327, 44), 'self', False)
            # Obtaining the member 'shape' of a type (line 327)
            shape_371179 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 327, 44), self_371178, 'shape')
            # Obtaining the member '__getitem__' of a type (line 327)
            getitem___371180 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 327, 44), shape_371179, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 327)
            subscript_call_result_371181 = invoke(stypy.reporting.localization.Localization(__file__, 327, 44), getitem___371180, int_371177)
            
            # Processing the call keyword arguments (line 327)
            kwargs_371182 = {}
            # Getting the type of 'extractor' (line 327)
            extractor_371171 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 327, 24), 'extractor', False)
            # Calling extractor(args, kwargs) (line 327)
            extractor_call_result_371183 = invoke(stypy.reporting.localization.Localization(__file__, 327, 24), extractor_371171, *[subscript_call_result_371176, subscript_call_result_371181], **kwargs_371182)
            
            # Assigning a type to the variable 'P_row' (line 327)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 327, 16), 'P_row', extractor_call_result_371183)
            
            # Assigning a Attribute to a Name (line 328):
            
            # Assigning a Attribute to a Name (line 328):
            
            # Call to extractor(...): (line 328)
            # Processing the call arguments (line 328)
            # Getting the type of 'col' (line 328)
            col_371185 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 328, 34), 'col', False)
            
            # Obtaining the type of the subscript
            int_371186 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 328, 50), 'int')
            # Getting the type of 'self' (line 328)
            self_371187 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 328, 39), 'self', False)
            # Obtaining the member 'shape' of a type (line 328)
            shape_371188 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 328, 39), self_371187, 'shape')
            # Obtaining the member '__getitem__' of a type (line 328)
            getitem___371189 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 328, 39), shape_371188, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 328)
            subscript_call_result_371190 = invoke(stypy.reporting.localization.Localization(__file__, 328, 39), getitem___371189, int_371186)
            
            # Processing the call keyword arguments (line 328)
            kwargs_371191 = {}
            # Getting the type of 'extractor' (line 328)
            extractor_371184 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 328, 24), 'extractor', False)
            # Calling extractor(args, kwargs) (line 328)
            extractor_call_result_371192 = invoke(stypy.reporting.localization.Localization(__file__, 328, 24), extractor_371184, *[col_371185, subscript_call_result_371190], **kwargs_371191)
            
            # Obtaining the member 'T' of a type (line 328)
            T_371193 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 328, 24), extractor_call_result_371192, 'T')
            # Assigning a type to the variable 'P_col' (line 328)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 328, 16), 'P_col', T_371193)
            # Getting the type of 'P_row' (line 329)
            P_row_371194 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 329, 23), 'P_row')
            # Getting the type of 'self' (line 329)
            self_371195 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 329, 31), 'self')
            # Applying the binary operator '*' (line 329)
            result_mul_371196 = python_operator(stypy.reporting.localization.Localization(__file__, 329, 23), '*', P_row_371194, self_371195)
            
            # Getting the type of 'P_col' (line 329)
            P_col_371197 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 329, 38), 'P_col')
            # Applying the binary operator '*' (line 329)
            result_mul_371198 = python_operator(stypy.reporting.localization.Localization(__file__, 329, 36), '*', result_mul_371196, P_col_371197)
            
            # Assigning a type to the variable 'stypy_return_type' (line 329)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 329, 16), 'stypy_return_type', result_mul_371198)
            # SSA join for if statement (line 324)
            module_type_store = module_type_store.join_ssa_context()
            
            # SSA join for if statement (line 323)
            module_type_store = module_type_store.join_ssa_context()
            
            # SSA join for if statement (line 313)
            module_type_store = module_type_store.join_ssa_context()
            

            if (may_be_371020 and more_types_in_union_371021):
                # SSA join for if statement (line 297)
                module_type_store = module_type_store.join_ssa_context()


        
        # SSA join for if statement (line 286)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        
        # Evaluating a boolean operation
        
        # Call to issequence(...): (line 331)
        # Processing the call arguments (line 331)
        # Getting the type of 'col' (line 331)
        col_371200 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 331, 27), 'col', False)
        # Processing the call keyword arguments (line 331)
        kwargs_371201 = {}
        # Getting the type of 'issequence' (line 331)
        issequence_371199 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 331, 16), 'issequence', False)
        # Calling issequence(args, kwargs) (line 331)
        issequence_call_result_371202 = invoke(stypy.reporting.localization.Localization(__file__, 331, 16), issequence_371199, *[col_371200], **kwargs_371201)
        
        
        # Call to issequence(...): (line 331)
        # Processing the call arguments (line 331)
        # Getting the type of 'row' (line 331)
        row_371204 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 331, 47), 'row', False)
        # Processing the call keyword arguments (line 331)
        kwargs_371205 = {}
        # Getting the type of 'issequence' (line 331)
        issequence_371203 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 331, 36), 'issequence', False)
        # Calling issequence(args, kwargs) (line 331)
        issequence_call_result_371206 = invoke(stypy.reporting.localization.Localization(__file__, 331, 36), issequence_371203, *[row_371204], **kwargs_371205)
        
        # Applying the binary operator 'and' (line 331)
        result_and_keyword_371207 = python_operator(stypy.reporting.localization.Localization(__file__, 331, 16), 'and', issequence_call_result_371202, issequence_call_result_371206)
        
        # Applying the 'not' unary operator (line 331)
        result_not__371208 = python_operator(stypy.reporting.localization.Localization(__file__, 331, 11), 'not', result_and_keyword_371207)
        
        # Testing the type of an if condition (line 331)
        if_condition_371209 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 331, 8), result_not__371208)
        # Assigning a type to the variable 'if_condition_371209' (line 331)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 331, 8), 'if_condition_371209', if_condition_371209)
        # SSA begins for if statement (line 331)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Tuple (line 333):
        
        # Assigning a Subscript to a Name (line 333):
        
        # Obtaining the type of the subscript
        int_371210 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 333, 12), 'int')
        
        # Call to _index_to_arrays(...): (line 333)
        # Processing the call arguments (line 333)
        # Getting the type of 'row' (line 333)
        row_371213 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 333, 45), 'row', False)
        # Getting the type of 'col' (line 333)
        col_371214 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 333, 50), 'col', False)
        # Processing the call keyword arguments (line 333)
        kwargs_371215 = {}
        # Getting the type of 'self' (line 333)
        self_371211 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 333, 23), 'self', False)
        # Obtaining the member '_index_to_arrays' of a type (line 333)
        _index_to_arrays_371212 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 333, 23), self_371211, '_index_to_arrays')
        # Calling _index_to_arrays(args, kwargs) (line 333)
        _index_to_arrays_call_result_371216 = invoke(stypy.reporting.localization.Localization(__file__, 333, 23), _index_to_arrays_371212, *[row_371213, col_371214], **kwargs_371215)
        
        # Obtaining the member '__getitem__' of a type (line 333)
        getitem___371217 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 333, 12), _index_to_arrays_call_result_371216, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 333)
        subscript_call_result_371218 = invoke(stypy.reporting.localization.Localization(__file__, 333, 12), getitem___371217, int_371210)
        
        # Assigning a type to the variable 'tuple_var_assignment_370333' (line 333)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 333, 12), 'tuple_var_assignment_370333', subscript_call_result_371218)
        
        # Assigning a Subscript to a Name (line 333):
        
        # Obtaining the type of the subscript
        int_371219 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 333, 12), 'int')
        
        # Call to _index_to_arrays(...): (line 333)
        # Processing the call arguments (line 333)
        # Getting the type of 'row' (line 333)
        row_371222 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 333, 45), 'row', False)
        # Getting the type of 'col' (line 333)
        col_371223 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 333, 50), 'col', False)
        # Processing the call keyword arguments (line 333)
        kwargs_371224 = {}
        # Getting the type of 'self' (line 333)
        self_371220 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 333, 23), 'self', False)
        # Obtaining the member '_index_to_arrays' of a type (line 333)
        _index_to_arrays_371221 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 333, 23), self_371220, '_index_to_arrays')
        # Calling _index_to_arrays(args, kwargs) (line 333)
        _index_to_arrays_call_result_371225 = invoke(stypy.reporting.localization.Localization(__file__, 333, 23), _index_to_arrays_371221, *[row_371222, col_371223], **kwargs_371224)
        
        # Obtaining the member '__getitem__' of a type (line 333)
        getitem___371226 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 333, 12), _index_to_arrays_call_result_371225, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 333)
        subscript_call_result_371227 = invoke(stypy.reporting.localization.Localization(__file__, 333, 12), getitem___371226, int_371219)
        
        # Assigning a type to the variable 'tuple_var_assignment_370334' (line 333)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 333, 12), 'tuple_var_assignment_370334', subscript_call_result_371227)
        
        # Assigning a Name to a Name (line 333):
        # Getting the type of 'tuple_var_assignment_370333' (line 333)
        tuple_var_assignment_370333_371228 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 333, 12), 'tuple_var_assignment_370333')
        # Assigning a type to the variable 'row' (line 333)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 333, 12), 'row', tuple_var_assignment_370333_371228)
        
        # Assigning a Name to a Name (line 333):
        # Getting the type of 'tuple_var_assignment_370334' (line 333)
        tuple_var_assignment_370334_371229 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 333, 12), 'tuple_var_assignment_370334')
        # Assigning a type to the variable 'col' (line 333)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 333, 17), 'col', tuple_var_assignment_370334_371229)
        # SSA join for if statement (line 331)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Name (line 335):
        
        # Assigning a Call to a Name (line 335):
        
        # Call to asindices(...): (line 335)
        # Processing the call arguments (line 335)
        # Getting the type of 'row' (line 335)
        row_371231 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 335, 24), 'row', False)
        # Processing the call keyword arguments (line 335)
        kwargs_371232 = {}
        # Getting the type of 'asindices' (line 335)
        asindices_371230 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 335, 14), 'asindices', False)
        # Calling asindices(args, kwargs) (line 335)
        asindices_call_result_371233 = invoke(stypy.reporting.localization.Localization(__file__, 335, 14), asindices_371230, *[row_371231], **kwargs_371232)
        
        # Assigning a type to the variable 'row' (line 335)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 335, 8), 'row', asindices_call_result_371233)
        
        # Assigning a Call to a Name (line 336):
        
        # Assigning a Call to a Name (line 336):
        
        # Call to asindices(...): (line 336)
        # Processing the call arguments (line 336)
        # Getting the type of 'col' (line 336)
        col_371235 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 336, 24), 'col', False)
        # Processing the call keyword arguments (line 336)
        kwargs_371236 = {}
        # Getting the type of 'asindices' (line 336)
        asindices_371234 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 336, 14), 'asindices', False)
        # Calling asindices(args, kwargs) (line 336)
        asindices_call_result_371237 = invoke(stypy.reporting.localization.Localization(__file__, 336, 14), asindices_371234, *[col_371235], **kwargs_371236)
        
        # Assigning a type to the variable 'col' (line 336)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 336, 8), 'col', asindices_call_result_371237)
        
        
        # Getting the type of 'row' (line 337)
        row_371238 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 337, 11), 'row')
        # Obtaining the member 'shape' of a type (line 337)
        shape_371239 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 337, 11), row_371238, 'shape')
        # Getting the type of 'col' (line 337)
        col_371240 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 337, 24), 'col')
        # Obtaining the member 'shape' of a type (line 337)
        shape_371241 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 337, 24), col_371240, 'shape')
        # Applying the binary operator '!=' (line 337)
        result_ne_371242 = python_operator(stypy.reporting.localization.Localization(__file__, 337, 11), '!=', shape_371239, shape_371241)
        
        # Testing the type of an if condition (line 337)
        if_condition_371243 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 337, 8), result_ne_371242)
        # Assigning a type to the variable 'if_condition_371243' (line 337)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 337, 8), 'if_condition_371243', if_condition_371243)
        # SSA begins for if statement (line 337)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to IndexError(...): (line 338)
        # Processing the call arguments (line 338)
        str_371245 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 338, 29), 'str', 'number of row and column indices differ')
        # Processing the call keyword arguments (line 338)
        kwargs_371246 = {}
        # Getting the type of 'IndexError' (line 338)
        IndexError_371244 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 338, 18), 'IndexError', False)
        # Calling IndexError(args, kwargs) (line 338)
        IndexError_call_result_371247 = invoke(stypy.reporting.localization.Localization(__file__, 338, 18), IndexError_371244, *[str_371245], **kwargs_371246)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 338, 12), IndexError_call_result_371247, 'raise parameter', BaseException)
        # SSA join for if statement (line 337)
        module_type_store = module_type_store.join_ssa_context()
        
        # Evaluating assert statement condition
        
        # Getting the type of 'row' (line 339)
        row_371248 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 339, 15), 'row')
        # Obtaining the member 'ndim' of a type (line 339)
        ndim_371249 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 339, 15), row_371248, 'ndim')
        int_371250 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 339, 27), 'int')
        # Applying the binary operator '<=' (line 339)
        result_le_371251 = python_operator(stypy.reporting.localization.Localization(__file__, 339, 15), '<=', ndim_371249, int_371250)
        
        
        # Assigning a Call to a Name (line 341):
        
        # Assigning a Call to a Name (line 341):
        
        # Call to size(...): (line 341)
        # Processing the call arguments (line 341)
        # Getting the type of 'row' (line 341)
        row_371254 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 341, 30), 'row', False)
        # Processing the call keyword arguments (line 341)
        kwargs_371255 = {}
        # Getting the type of 'np' (line 341)
        np_371252 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 341, 22), 'np', False)
        # Obtaining the member 'size' of a type (line 341)
        size_371253 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 341, 22), np_371252, 'size')
        # Calling size(args, kwargs) (line 341)
        size_call_result_371256 = invoke(stypy.reporting.localization.Localization(__file__, 341, 22), size_371253, *[row_371254], **kwargs_371255)
        
        # Assigning a type to the variable 'num_samples' (line 341)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 341, 8), 'num_samples', size_call_result_371256)
        
        
        # Getting the type of 'num_samples' (line 342)
        num_samples_371257 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 342, 11), 'num_samples')
        int_371258 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 342, 26), 'int')
        # Applying the binary operator '==' (line 342)
        result_eq_371259 = python_operator(stypy.reporting.localization.Localization(__file__, 342, 11), '==', num_samples_371257, int_371258)
        
        # Testing the type of an if condition (line 342)
        if_condition_371260 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 342, 8), result_eq_371259)
        # Assigning a type to the variable 'if_condition_371260' (line 342)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 342, 8), 'if_condition_371260', if_condition_371260)
        # SSA begins for if statement (line 342)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to csr_matrix(...): (line 343)
        # Processing the call arguments (line 343)
        
        # Call to atleast_2d(...): (line 343)
        # Processing the call arguments (line 343)
        # Getting the type of 'row' (line 343)
        row_371264 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 343, 44), 'row', False)
        # Processing the call keyword arguments (line 343)
        kwargs_371265 = {}
        # Getting the type of 'np' (line 343)
        np_371262 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 343, 30), 'np', False)
        # Obtaining the member 'atleast_2d' of a type (line 343)
        atleast_2d_371263 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 343, 30), np_371262, 'atleast_2d')
        # Calling atleast_2d(args, kwargs) (line 343)
        atleast_2d_call_result_371266 = invoke(stypy.reporting.localization.Localization(__file__, 343, 30), atleast_2d_371263, *[row_371264], **kwargs_371265)
        
        # Obtaining the member 'shape' of a type (line 343)
        shape_371267 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 343, 30), atleast_2d_call_result_371266, 'shape')
        # Processing the call keyword arguments (line 343)
        # Getting the type of 'self' (line 343)
        self_371268 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 343, 62), 'self', False)
        # Obtaining the member 'dtype' of a type (line 343)
        dtype_371269 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 343, 62), self_371268, 'dtype')
        keyword_371270 = dtype_371269
        kwargs_371271 = {'dtype': keyword_371270}
        # Getting the type of 'csr_matrix' (line 343)
        csr_matrix_371261 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 343, 19), 'csr_matrix', False)
        # Calling csr_matrix(args, kwargs) (line 343)
        csr_matrix_call_result_371272 = invoke(stypy.reporting.localization.Localization(__file__, 343, 19), csr_matrix_371261, *[shape_371267], **kwargs_371271)
        
        # Assigning a type to the variable 'stypy_return_type' (line 343)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 343, 12), 'stypy_return_type', csr_matrix_call_result_371272)
        # SSA join for if statement (line 342)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to check_bounds(...): (line 344)
        # Processing the call arguments (line 344)
        # Getting the type of 'row' (line 344)
        row_371274 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 344, 21), 'row', False)
        
        # Obtaining the type of the subscript
        int_371275 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 344, 37), 'int')
        # Getting the type of 'self' (line 344)
        self_371276 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 344, 26), 'self', False)
        # Obtaining the member 'shape' of a type (line 344)
        shape_371277 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 344, 26), self_371276, 'shape')
        # Obtaining the member '__getitem__' of a type (line 344)
        getitem___371278 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 344, 26), shape_371277, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 344)
        subscript_call_result_371279 = invoke(stypy.reporting.localization.Localization(__file__, 344, 26), getitem___371278, int_371275)
        
        # Processing the call keyword arguments (line 344)
        kwargs_371280 = {}
        # Getting the type of 'check_bounds' (line 344)
        check_bounds_371273 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 344, 8), 'check_bounds', False)
        # Calling check_bounds(args, kwargs) (line 344)
        check_bounds_call_result_371281 = invoke(stypy.reporting.localization.Localization(__file__, 344, 8), check_bounds_371273, *[row_371274, subscript_call_result_371279], **kwargs_371280)
        
        
        # Call to check_bounds(...): (line 345)
        # Processing the call arguments (line 345)
        # Getting the type of 'col' (line 345)
        col_371283 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 345, 21), 'col', False)
        
        # Obtaining the type of the subscript
        int_371284 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 345, 37), 'int')
        # Getting the type of 'self' (line 345)
        self_371285 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 345, 26), 'self', False)
        # Obtaining the member 'shape' of a type (line 345)
        shape_371286 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 345, 26), self_371285, 'shape')
        # Obtaining the member '__getitem__' of a type (line 345)
        getitem___371287 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 345, 26), shape_371286, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 345)
        subscript_call_result_371288 = invoke(stypy.reporting.localization.Localization(__file__, 345, 26), getitem___371287, int_371284)
        
        # Processing the call keyword arguments (line 345)
        kwargs_371289 = {}
        # Getting the type of 'check_bounds' (line 345)
        check_bounds_371282 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 345, 8), 'check_bounds', False)
        # Calling check_bounds(args, kwargs) (line 345)
        check_bounds_call_result_371290 = invoke(stypy.reporting.localization.Localization(__file__, 345, 8), check_bounds_371282, *[col_371283, subscript_call_result_371288], **kwargs_371289)
        
        
        # Assigning a Call to a Name (line 347):
        
        # Assigning a Call to a Name (line 347):
        
        # Call to empty(...): (line 347)
        # Processing the call arguments (line 347)
        # Getting the type of 'num_samples' (line 347)
        num_samples_371293 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 347, 23), 'num_samples', False)
        # Processing the call keyword arguments (line 347)
        # Getting the type of 'self' (line 347)
        self_371294 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 347, 42), 'self', False)
        # Obtaining the member 'dtype' of a type (line 347)
        dtype_371295 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 347, 42), self_371294, 'dtype')
        keyword_371296 = dtype_371295
        kwargs_371297 = {'dtype': keyword_371296}
        # Getting the type of 'np' (line 347)
        np_371291 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 347, 14), 'np', False)
        # Obtaining the member 'empty' of a type (line 347)
        empty_371292 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 347, 14), np_371291, 'empty')
        # Calling empty(args, kwargs) (line 347)
        empty_call_result_371298 = invoke(stypy.reporting.localization.Localization(__file__, 347, 14), empty_371292, *[num_samples_371293], **kwargs_371297)
        
        # Assigning a type to the variable 'val' (line 347)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 347, 8), 'val', empty_call_result_371298)
        
        # Call to csr_sample_values(...): (line 348)
        # Processing the call arguments (line 348)
        
        # Obtaining the type of the subscript
        int_371300 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 348, 37), 'int')
        # Getting the type of 'self' (line 348)
        self_371301 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 348, 26), 'self', False)
        # Obtaining the member 'shape' of a type (line 348)
        shape_371302 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 348, 26), self_371301, 'shape')
        # Obtaining the member '__getitem__' of a type (line 348)
        getitem___371303 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 348, 26), shape_371302, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 348)
        subscript_call_result_371304 = invoke(stypy.reporting.localization.Localization(__file__, 348, 26), getitem___371303, int_371300)
        
        
        # Obtaining the type of the subscript
        int_371305 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 348, 52), 'int')
        # Getting the type of 'self' (line 348)
        self_371306 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 348, 41), 'self', False)
        # Obtaining the member 'shape' of a type (line 348)
        shape_371307 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 348, 41), self_371306, 'shape')
        # Obtaining the member '__getitem__' of a type (line 348)
        getitem___371308 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 348, 41), shape_371307, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 348)
        subscript_call_result_371309 = invoke(stypy.reporting.localization.Localization(__file__, 348, 41), getitem___371308, int_371305)
        
        # Getting the type of 'self' (line 349)
        self_371310 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 349, 26), 'self', False)
        # Obtaining the member 'indptr' of a type (line 349)
        indptr_371311 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 349, 26), self_371310, 'indptr')
        # Getting the type of 'self' (line 349)
        self_371312 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 349, 39), 'self', False)
        # Obtaining the member 'indices' of a type (line 349)
        indices_371313 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 349, 39), self_371312, 'indices')
        # Getting the type of 'self' (line 349)
        self_371314 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 349, 53), 'self', False)
        # Obtaining the member 'data' of a type (line 349)
        data_371315 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 349, 53), self_371314, 'data')
        # Getting the type of 'num_samples' (line 350)
        num_samples_371316 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 350, 26), 'num_samples', False)
        
        # Call to ravel(...): (line 350)
        # Processing the call keyword arguments (line 350)
        kwargs_371319 = {}
        # Getting the type of 'row' (line 350)
        row_371317 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 350, 39), 'row', False)
        # Obtaining the member 'ravel' of a type (line 350)
        ravel_371318 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 350, 39), row_371317, 'ravel')
        # Calling ravel(args, kwargs) (line 350)
        ravel_call_result_371320 = invoke(stypy.reporting.localization.Localization(__file__, 350, 39), ravel_371318, *[], **kwargs_371319)
        
        
        # Call to ravel(...): (line 350)
        # Processing the call keyword arguments (line 350)
        kwargs_371323 = {}
        # Getting the type of 'col' (line 350)
        col_371321 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 350, 52), 'col', False)
        # Obtaining the member 'ravel' of a type (line 350)
        ravel_371322 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 350, 52), col_371321, 'ravel')
        # Calling ravel(args, kwargs) (line 350)
        ravel_call_result_371324 = invoke(stypy.reporting.localization.Localization(__file__, 350, 52), ravel_371322, *[], **kwargs_371323)
        
        # Getting the type of 'val' (line 350)
        val_371325 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 350, 65), 'val', False)
        # Processing the call keyword arguments (line 348)
        kwargs_371326 = {}
        # Getting the type of 'csr_sample_values' (line 348)
        csr_sample_values_371299 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 348, 8), 'csr_sample_values', False)
        # Calling csr_sample_values(args, kwargs) (line 348)
        csr_sample_values_call_result_371327 = invoke(stypy.reporting.localization.Localization(__file__, 348, 8), csr_sample_values_371299, *[subscript_call_result_371304, subscript_call_result_371309, indptr_371311, indices_371313, data_371315, num_samples_371316, ravel_call_result_371320, ravel_call_result_371324, val_371325], **kwargs_371326)
        
        
        
        # Getting the type of 'row' (line 351)
        row_371328 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 351, 11), 'row')
        # Obtaining the member 'ndim' of a type (line 351)
        ndim_371329 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 351, 11), row_371328, 'ndim')
        int_371330 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 351, 23), 'int')
        # Applying the binary operator '==' (line 351)
        result_eq_371331 = python_operator(stypy.reporting.localization.Localization(__file__, 351, 11), '==', ndim_371329, int_371330)
        
        # Testing the type of an if condition (line 351)
        if_condition_371332 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 351, 8), result_eq_371331)
        # Assigning a type to the variable 'if_condition_371332' (line 351)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 351, 8), 'if_condition_371332', if_condition_371332)
        # SSA begins for if statement (line 351)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to asmatrix(...): (line 353)
        # Processing the call arguments (line 353)
        # Getting the type of 'val' (line 353)
        val_371335 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 353, 31), 'val', False)
        # Processing the call keyword arguments (line 353)
        kwargs_371336 = {}
        # Getting the type of 'np' (line 353)
        np_371333 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 353, 19), 'np', False)
        # Obtaining the member 'asmatrix' of a type (line 353)
        asmatrix_371334 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 353, 19), np_371333, 'asmatrix')
        # Calling asmatrix(args, kwargs) (line 353)
        asmatrix_call_result_371337 = invoke(stypy.reporting.localization.Localization(__file__, 353, 19), asmatrix_371334, *[val_371335], **kwargs_371336)
        
        # Assigning a type to the variable 'stypy_return_type' (line 353)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 353, 12), 'stypy_return_type', asmatrix_call_result_371337)
        # SSA join for if statement (line 351)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to __class__(...): (line 354)
        # Processing the call arguments (line 354)
        
        # Call to reshape(...): (line 354)
        # Processing the call arguments (line 354)
        # Getting the type of 'row' (line 354)
        row_371342 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 354, 42), 'row', False)
        # Obtaining the member 'shape' of a type (line 354)
        shape_371343 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 354, 42), row_371342, 'shape')
        # Processing the call keyword arguments (line 354)
        kwargs_371344 = {}
        # Getting the type of 'val' (line 354)
        val_371340 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 354, 30), 'val', False)
        # Obtaining the member 'reshape' of a type (line 354)
        reshape_371341 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 354, 30), val_371340, 'reshape')
        # Calling reshape(args, kwargs) (line 354)
        reshape_call_result_371345 = invoke(stypy.reporting.localization.Localization(__file__, 354, 30), reshape_371341, *[shape_371343], **kwargs_371344)
        
        # Processing the call keyword arguments (line 354)
        kwargs_371346 = {}
        # Getting the type of 'self' (line 354)
        self_371338 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 354, 15), 'self', False)
        # Obtaining the member '__class__' of a type (line 354)
        class___371339 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 354, 15), self_371338, '__class__')
        # Calling __class__(args, kwargs) (line 354)
        class___call_result_371347 = invoke(stypy.reporting.localization.Localization(__file__, 354, 15), class___371339, *[reshape_call_result_371345], **kwargs_371346)
        
        # Assigning a type to the variable 'stypy_return_type' (line 354)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 354, 8), 'stypy_return_type', class___call_result_371347)
        
        # ################# End of '__getitem__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__getitem__' in the type store
        # Getting the type of 'stypy_return_type' (line 236)
        stypy_return_type_371348 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 236, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_371348)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__getitem__'
        return stypy_return_type_371348


    @norecursion
    def __iter__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__iter__'
        module_type_store = module_type_store.open_function_context('__iter__', 356, 4, False)
        # Assigning a type to the variable 'self' (line 357)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 357, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        csr_matrix.__iter__.__dict__.__setitem__('stypy_localization', localization)
        csr_matrix.__iter__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        csr_matrix.__iter__.__dict__.__setitem__('stypy_type_store', module_type_store)
        csr_matrix.__iter__.__dict__.__setitem__('stypy_function_name', 'csr_matrix.__iter__')
        csr_matrix.__iter__.__dict__.__setitem__('stypy_param_names_list', [])
        csr_matrix.__iter__.__dict__.__setitem__('stypy_varargs_param_name', None)
        csr_matrix.__iter__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        csr_matrix.__iter__.__dict__.__setitem__('stypy_call_defaults', defaults)
        csr_matrix.__iter__.__dict__.__setitem__('stypy_call_varargs', varargs)
        csr_matrix.__iter__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        csr_matrix.__iter__.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'csr_matrix.__iter__', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__iter__', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__iter__(...)' code ##################

        
        # Assigning a Call to a Name (line 357):
        
        # Assigning a Call to a Name (line 357):
        
        # Call to zeros(...): (line 357)
        # Processing the call arguments (line 357)
        int_371351 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 357, 26), 'int')
        # Processing the call keyword arguments (line 357)
        # Getting the type of 'self' (line 357)
        self_371352 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 357, 35), 'self', False)
        # Obtaining the member 'indptr' of a type (line 357)
        indptr_371353 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 357, 35), self_371352, 'indptr')
        # Obtaining the member 'dtype' of a type (line 357)
        dtype_371354 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 357, 35), indptr_371353, 'dtype')
        keyword_371355 = dtype_371354
        kwargs_371356 = {'dtype': keyword_371355}
        # Getting the type of 'np' (line 357)
        np_371349 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 357, 17), 'np', False)
        # Obtaining the member 'zeros' of a type (line 357)
        zeros_371350 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 357, 17), np_371349, 'zeros')
        # Calling zeros(args, kwargs) (line 357)
        zeros_call_result_371357 = invoke(stypy.reporting.localization.Localization(__file__, 357, 17), zeros_371350, *[int_371351], **kwargs_371356)
        
        # Assigning a type to the variable 'indptr' (line 357)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 357, 8), 'indptr', zeros_call_result_371357)
        
        # Assigning a Tuple to a Name (line 358):
        
        # Assigning a Tuple to a Name (line 358):
        
        # Obtaining an instance of the builtin type 'tuple' (line 358)
        tuple_371358 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 358, 17), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 358)
        # Adding element type (line 358)
        int_371359 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 358, 17), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 358, 17), tuple_371358, int_371359)
        # Adding element type (line 358)
        
        # Obtaining the type of the subscript
        int_371360 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 358, 31), 'int')
        # Getting the type of 'self' (line 358)
        self_371361 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 358, 20), 'self')
        # Obtaining the member 'shape' of a type (line 358)
        shape_371362 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 358, 20), self_371361, 'shape')
        # Obtaining the member '__getitem__' of a type (line 358)
        getitem___371363 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 358, 20), shape_371362, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 358)
        subscript_call_result_371364 = invoke(stypy.reporting.localization.Localization(__file__, 358, 20), getitem___371363, int_371360)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 358, 17), tuple_371358, subscript_call_result_371364)
        
        # Assigning a type to the variable 'shape' (line 358)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 358, 8), 'shape', tuple_371358)
        
        # Assigning a Num to a Name (line 359):
        
        # Assigning a Num to a Name (line 359):
        int_371365 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 359, 13), 'int')
        # Assigning a type to the variable 'i0' (line 359)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 359, 8), 'i0', int_371365)
        
        
        # Obtaining the type of the subscript
        int_371366 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 360, 30), 'int')
        slice_371367 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 360, 18), int_371366, None, None)
        # Getting the type of 'self' (line 360)
        self_371368 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 360, 18), 'self')
        # Obtaining the member 'indptr' of a type (line 360)
        indptr_371369 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 360, 18), self_371368, 'indptr')
        # Obtaining the member '__getitem__' of a type (line 360)
        getitem___371370 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 360, 18), indptr_371369, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 360)
        subscript_call_result_371371 = invoke(stypy.reporting.localization.Localization(__file__, 360, 18), getitem___371370, slice_371367)
        
        # Testing the type of a for loop iterable (line 360)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 360, 8), subscript_call_result_371371)
        # Getting the type of the for loop variable (line 360)
        for_loop_var_371372 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 360, 8), subscript_call_result_371371)
        # Assigning a type to the variable 'i1' (line 360)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 360, 8), 'i1', for_loop_var_371372)
        # SSA begins for a for statement (line 360)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a BinOp to a Subscript (line 361):
        
        # Assigning a BinOp to a Subscript (line 361):
        # Getting the type of 'i1' (line 361)
        i1_371373 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 361, 24), 'i1')
        # Getting the type of 'i0' (line 361)
        i0_371374 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 361, 29), 'i0')
        # Applying the binary operator '-' (line 361)
        result_sub_371375 = python_operator(stypy.reporting.localization.Localization(__file__, 361, 24), '-', i1_371373, i0_371374)
        
        # Getting the type of 'indptr' (line 361)
        indptr_371376 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 361, 12), 'indptr')
        int_371377 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 361, 19), 'int')
        # Storing an element on a container (line 361)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 361, 12), indptr_371376, (int_371377, result_sub_371375))
        
        # Assigning a Subscript to a Name (line 362):
        
        # Assigning a Subscript to a Name (line 362):
        
        # Obtaining the type of the subscript
        # Getting the type of 'i0' (line 362)
        i0_371378 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 362, 35), 'i0')
        # Getting the type of 'i1' (line 362)
        i1_371379 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 362, 38), 'i1')
        slice_371380 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 362, 22), i0_371378, i1_371379, None)
        # Getting the type of 'self' (line 362)
        self_371381 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 362, 22), 'self')
        # Obtaining the member 'indices' of a type (line 362)
        indices_371382 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 362, 22), self_371381, 'indices')
        # Obtaining the member '__getitem__' of a type (line 362)
        getitem___371383 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 362, 22), indices_371382, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 362)
        subscript_call_result_371384 = invoke(stypy.reporting.localization.Localization(__file__, 362, 22), getitem___371383, slice_371380)
        
        # Assigning a type to the variable 'indices' (line 362)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 362, 12), 'indices', subscript_call_result_371384)
        
        # Assigning a Subscript to a Name (line 363):
        
        # Assigning a Subscript to a Name (line 363):
        
        # Obtaining the type of the subscript
        # Getting the type of 'i0' (line 363)
        i0_371385 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 363, 29), 'i0')
        # Getting the type of 'i1' (line 363)
        i1_371386 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 363, 32), 'i1')
        slice_371387 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 363, 19), i0_371385, i1_371386, None)
        # Getting the type of 'self' (line 363)
        self_371388 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 363, 19), 'self')
        # Obtaining the member 'data' of a type (line 363)
        data_371389 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 363, 19), self_371388, 'data')
        # Obtaining the member '__getitem__' of a type (line 363)
        getitem___371390 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 363, 19), data_371389, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 363)
        subscript_call_result_371391 = invoke(stypy.reporting.localization.Localization(__file__, 363, 19), getitem___371390, slice_371387)
        
        # Assigning a type to the variable 'data' (line 363)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 363, 12), 'data', subscript_call_result_371391)
        # Creating a generator
        
        # Call to csr_matrix(...): (line 364)
        # Processing the call arguments (line 364)
        
        # Obtaining an instance of the builtin type 'tuple' (line 364)
        tuple_371393 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 364, 30), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 364)
        # Adding element type (line 364)
        # Getting the type of 'data' (line 364)
        data_371394 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 364, 30), 'data', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 364, 30), tuple_371393, data_371394)
        # Adding element type (line 364)
        # Getting the type of 'indices' (line 364)
        indices_371395 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 364, 36), 'indices', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 364, 30), tuple_371393, indices_371395)
        # Adding element type (line 364)
        # Getting the type of 'indptr' (line 364)
        indptr_371396 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 364, 45), 'indptr', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 364, 30), tuple_371393, indptr_371396)
        
        # Processing the call keyword arguments (line 364)
        # Getting the type of 'shape' (line 364)
        shape_371397 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 364, 60), 'shape', False)
        keyword_371398 = shape_371397
        # Getting the type of 'True' (line 364)
        True_371399 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 364, 72), 'True', False)
        keyword_371400 = True_371399
        kwargs_371401 = {'shape': keyword_371398, 'copy': keyword_371400}
        # Getting the type of 'csr_matrix' (line 364)
        csr_matrix_371392 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 364, 18), 'csr_matrix', False)
        # Calling csr_matrix(args, kwargs) (line 364)
        csr_matrix_call_result_371402 = invoke(stypy.reporting.localization.Localization(__file__, 364, 18), csr_matrix_371392, *[tuple_371393], **kwargs_371401)
        
        GeneratorType_371403 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 364, 12), 'GeneratorType')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 364, 12), GeneratorType_371403, csr_matrix_call_result_371402)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 364, 12), 'stypy_return_type', GeneratorType_371403)
        
        # Assigning a Name to a Name (line 365):
        
        # Assigning a Name to a Name (line 365):
        # Getting the type of 'i1' (line 365)
        i1_371404 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 365, 17), 'i1')
        # Assigning a type to the variable 'i0' (line 365)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 365, 12), 'i0', i1_371404)
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of '__iter__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__iter__' in the type store
        # Getting the type of 'stypy_return_type' (line 356)
        stypy_return_type_371405 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 356, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_371405)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__iter__'
        return stypy_return_type_371405


    @norecursion
    def getrow(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'getrow'
        module_type_store = module_type_store.open_function_context('getrow', 367, 4, False)
        # Assigning a type to the variable 'self' (line 368)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 368, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        csr_matrix.getrow.__dict__.__setitem__('stypy_localization', localization)
        csr_matrix.getrow.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        csr_matrix.getrow.__dict__.__setitem__('stypy_type_store', module_type_store)
        csr_matrix.getrow.__dict__.__setitem__('stypy_function_name', 'csr_matrix.getrow')
        csr_matrix.getrow.__dict__.__setitem__('stypy_param_names_list', ['i'])
        csr_matrix.getrow.__dict__.__setitem__('stypy_varargs_param_name', None)
        csr_matrix.getrow.__dict__.__setitem__('stypy_kwargs_param_name', None)
        csr_matrix.getrow.__dict__.__setitem__('stypy_call_defaults', defaults)
        csr_matrix.getrow.__dict__.__setitem__('stypy_call_varargs', varargs)
        csr_matrix.getrow.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        csr_matrix.getrow.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'csr_matrix.getrow', ['i'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'getrow', localization, ['i'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'getrow(...)' code ##################

        str_371406 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 370, (-1)), 'str', 'Returns a copy of row i of the matrix, as a (1 x n)\n        CSR matrix (row vector).\n        ')
        
        # Assigning a Attribute to a Tuple (line 371):
        
        # Assigning a Subscript to a Name (line 371):
        
        # Obtaining the type of the subscript
        int_371407 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 371, 8), 'int')
        # Getting the type of 'self' (line 371)
        self_371408 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 371, 15), 'self')
        # Obtaining the member 'shape' of a type (line 371)
        shape_371409 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 371, 15), self_371408, 'shape')
        # Obtaining the member '__getitem__' of a type (line 371)
        getitem___371410 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 371, 8), shape_371409, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 371)
        subscript_call_result_371411 = invoke(stypy.reporting.localization.Localization(__file__, 371, 8), getitem___371410, int_371407)
        
        # Assigning a type to the variable 'tuple_var_assignment_370335' (line 371)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 371, 8), 'tuple_var_assignment_370335', subscript_call_result_371411)
        
        # Assigning a Subscript to a Name (line 371):
        
        # Obtaining the type of the subscript
        int_371412 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 371, 8), 'int')
        # Getting the type of 'self' (line 371)
        self_371413 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 371, 15), 'self')
        # Obtaining the member 'shape' of a type (line 371)
        shape_371414 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 371, 15), self_371413, 'shape')
        # Obtaining the member '__getitem__' of a type (line 371)
        getitem___371415 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 371, 8), shape_371414, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 371)
        subscript_call_result_371416 = invoke(stypy.reporting.localization.Localization(__file__, 371, 8), getitem___371415, int_371412)
        
        # Assigning a type to the variable 'tuple_var_assignment_370336' (line 371)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 371, 8), 'tuple_var_assignment_370336', subscript_call_result_371416)
        
        # Assigning a Name to a Name (line 371):
        # Getting the type of 'tuple_var_assignment_370335' (line 371)
        tuple_var_assignment_370335_371417 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 371, 8), 'tuple_var_assignment_370335')
        # Assigning a type to the variable 'M' (line 371)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 371, 8), 'M', tuple_var_assignment_370335_371417)
        
        # Assigning a Name to a Name (line 371):
        # Getting the type of 'tuple_var_assignment_370336' (line 371)
        tuple_var_assignment_370336_371418 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 371, 8), 'tuple_var_assignment_370336')
        # Assigning a type to the variable 'N' (line 371)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 371, 11), 'N', tuple_var_assignment_370336_371418)
        
        # Assigning a Call to a Name (line 372):
        
        # Assigning a Call to a Name (line 372):
        
        # Call to int(...): (line 372)
        # Processing the call arguments (line 372)
        # Getting the type of 'i' (line 372)
        i_371420 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 372, 16), 'i', False)
        # Processing the call keyword arguments (line 372)
        kwargs_371421 = {}
        # Getting the type of 'int' (line 372)
        int_371419 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 372, 12), 'int', False)
        # Calling int(args, kwargs) (line 372)
        int_call_result_371422 = invoke(stypy.reporting.localization.Localization(__file__, 372, 12), int_371419, *[i_371420], **kwargs_371421)
        
        # Assigning a type to the variable 'i' (line 372)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 372, 8), 'i', int_call_result_371422)
        
        
        # Getting the type of 'i' (line 373)
        i_371423 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 373, 11), 'i')
        int_371424 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 373, 15), 'int')
        # Applying the binary operator '<' (line 373)
        result_lt_371425 = python_operator(stypy.reporting.localization.Localization(__file__, 373, 11), '<', i_371423, int_371424)
        
        # Testing the type of an if condition (line 373)
        if_condition_371426 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 373, 8), result_lt_371425)
        # Assigning a type to the variable 'if_condition_371426' (line 373)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 373, 8), 'if_condition_371426', if_condition_371426)
        # SSA begins for if statement (line 373)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Getting the type of 'i' (line 374)
        i_371427 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 374, 12), 'i')
        # Getting the type of 'M' (line 374)
        M_371428 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 374, 17), 'M')
        # Applying the binary operator '+=' (line 374)
        result_iadd_371429 = python_operator(stypy.reporting.localization.Localization(__file__, 374, 12), '+=', i_371427, M_371428)
        # Assigning a type to the variable 'i' (line 374)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 374, 12), 'i', result_iadd_371429)
        
        # SSA join for if statement (line 373)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Evaluating a boolean operation
        
        # Getting the type of 'i' (line 375)
        i_371430 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 375, 11), 'i')
        int_371431 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 375, 15), 'int')
        # Applying the binary operator '<' (line 375)
        result_lt_371432 = python_operator(stypy.reporting.localization.Localization(__file__, 375, 11), '<', i_371430, int_371431)
        
        
        # Getting the type of 'i' (line 375)
        i_371433 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 375, 20), 'i')
        # Getting the type of 'M' (line 375)
        M_371434 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 375, 25), 'M')
        # Applying the binary operator '>=' (line 375)
        result_ge_371435 = python_operator(stypy.reporting.localization.Localization(__file__, 375, 20), '>=', i_371433, M_371434)
        
        # Applying the binary operator 'or' (line 375)
        result_or_keyword_371436 = python_operator(stypy.reporting.localization.Localization(__file__, 375, 11), 'or', result_lt_371432, result_ge_371435)
        
        # Testing the type of an if condition (line 375)
        if_condition_371437 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 375, 8), result_or_keyword_371436)
        # Assigning a type to the variable 'if_condition_371437' (line 375)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 375, 8), 'if_condition_371437', if_condition_371437)
        # SSA begins for if statement (line 375)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to IndexError(...): (line 376)
        # Processing the call arguments (line 376)
        str_371439 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 376, 29), 'str', 'index (%d) out of range')
        # Getting the type of 'i' (line 376)
        i_371440 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 376, 57), 'i', False)
        # Applying the binary operator '%' (line 376)
        result_mod_371441 = python_operator(stypy.reporting.localization.Localization(__file__, 376, 29), '%', str_371439, i_371440)
        
        # Processing the call keyword arguments (line 376)
        kwargs_371442 = {}
        # Getting the type of 'IndexError' (line 376)
        IndexError_371438 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 376, 18), 'IndexError', False)
        # Calling IndexError(args, kwargs) (line 376)
        IndexError_call_result_371443 = invoke(stypy.reporting.localization.Localization(__file__, 376, 18), IndexError_371438, *[result_mod_371441], **kwargs_371442)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 376, 12), IndexError_call_result_371443, 'raise parameter', BaseException)
        # SSA join for if statement (line 375)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Name (line 377):
        
        # Assigning a Call to a Name (line 377):
        
        # Call to slice(...): (line 377)
        
        # Obtaining the type of the subscript
        # Getting the type of 'i' (line 377)
        i_371445 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 377, 33), 'i', False)
        # Getting the type of 'i' (line 377)
        i_371446 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 377, 35), 'i', False)
        int_371447 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 377, 37), 'int')
        # Applying the binary operator '+' (line 377)
        result_add_371448 = python_operator(stypy.reporting.localization.Localization(__file__, 377, 35), '+', i_371446, int_371447)
        
        slice_371449 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 377, 21), i_371445, result_add_371448, None)
        # Getting the type of 'self' (line 377)
        self_371450 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 377, 21), 'self', False)
        # Obtaining the member 'indptr' of a type (line 377)
        indptr_371451 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 377, 21), self_371450, 'indptr')
        # Obtaining the member '__getitem__' of a type (line 377)
        getitem___371452 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 377, 21), indptr_371451, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 377)
        subscript_call_result_371453 = invoke(stypy.reporting.localization.Localization(__file__, 377, 21), getitem___371452, slice_371449)
        
        # Processing the call keyword arguments (line 377)
        kwargs_371454 = {}
        # Getting the type of 'slice' (line 377)
        slice_371444 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 377, 14), 'slice', False)
        # Calling slice(args, kwargs) (line 377)
        slice_call_result_371455 = invoke(stypy.reporting.localization.Localization(__file__, 377, 14), slice_371444, *[subscript_call_result_371453], **kwargs_371454)
        
        # Assigning a type to the variable 'idx' (line 377)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 377, 8), 'idx', slice_call_result_371455)
        
        # Assigning a Call to a Name (line 378):
        
        # Assigning a Call to a Name (line 378):
        
        # Call to copy(...): (line 378)
        # Processing the call keyword arguments (line 378)
        kwargs_371462 = {}
        
        # Obtaining the type of the subscript
        # Getting the type of 'idx' (line 378)
        idx_371456 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 378, 25), 'idx', False)
        # Getting the type of 'self' (line 378)
        self_371457 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 378, 15), 'self', False)
        # Obtaining the member 'data' of a type (line 378)
        data_371458 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 378, 15), self_371457, 'data')
        # Obtaining the member '__getitem__' of a type (line 378)
        getitem___371459 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 378, 15), data_371458, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 378)
        subscript_call_result_371460 = invoke(stypy.reporting.localization.Localization(__file__, 378, 15), getitem___371459, idx_371456)
        
        # Obtaining the member 'copy' of a type (line 378)
        copy_371461 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 378, 15), subscript_call_result_371460, 'copy')
        # Calling copy(args, kwargs) (line 378)
        copy_call_result_371463 = invoke(stypy.reporting.localization.Localization(__file__, 378, 15), copy_371461, *[], **kwargs_371462)
        
        # Assigning a type to the variable 'data' (line 378)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 378, 8), 'data', copy_call_result_371463)
        
        # Assigning a Call to a Name (line 379):
        
        # Assigning a Call to a Name (line 379):
        
        # Call to copy(...): (line 379)
        # Processing the call keyword arguments (line 379)
        kwargs_371470 = {}
        
        # Obtaining the type of the subscript
        # Getting the type of 'idx' (line 379)
        idx_371464 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 379, 31), 'idx', False)
        # Getting the type of 'self' (line 379)
        self_371465 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 379, 18), 'self', False)
        # Obtaining the member 'indices' of a type (line 379)
        indices_371466 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 379, 18), self_371465, 'indices')
        # Obtaining the member '__getitem__' of a type (line 379)
        getitem___371467 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 379, 18), indices_371466, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 379)
        subscript_call_result_371468 = invoke(stypy.reporting.localization.Localization(__file__, 379, 18), getitem___371467, idx_371464)
        
        # Obtaining the member 'copy' of a type (line 379)
        copy_371469 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 379, 18), subscript_call_result_371468, 'copy')
        # Calling copy(args, kwargs) (line 379)
        copy_call_result_371471 = invoke(stypy.reporting.localization.Localization(__file__, 379, 18), copy_371469, *[], **kwargs_371470)
        
        # Assigning a type to the variable 'indices' (line 379)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 379, 8), 'indices', copy_call_result_371471)
        
        # Assigning a Call to a Name (line 380):
        
        # Assigning a Call to a Name (line 380):
        
        # Call to array(...): (line 380)
        # Processing the call arguments (line 380)
        
        # Obtaining an instance of the builtin type 'list' (line 380)
        list_371474 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 380, 26), 'list')
        # Adding type elements to the builtin type 'list' instance (line 380)
        # Adding element type (line 380)
        int_371475 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 380, 27), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 380, 26), list_371474, int_371475)
        # Adding element type (line 380)
        
        # Call to len(...): (line 380)
        # Processing the call arguments (line 380)
        # Getting the type of 'indices' (line 380)
        indices_371477 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 380, 34), 'indices', False)
        # Processing the call keyword arguments (line 380)
        kwargs_371478 = {}
        # Getting the type of 'len' (line 380)
        len_371476 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 380, 30), 'len', False)
        # Calling len(args, kwargs) (line 380)
        len_call_result_371479 = invoke(stypy.reporting.localization.Localization(__file__, 380, 30), len_371476, *[indices_371477], **kwargs_371478)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 380, 26), list_371474, len_call_result_371479)
        
        # Processing the call keyword arguments (line 380)
        # Getting the type of 'self' (line 380)
        self_371480 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 380, 51), 'self', False)
        # Obtaining the member 'indptr' of a type (line 380)
        indptr_371481 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 380, 51), self_371480, 'indptr')
        # Obtaining the member 'dtype' of a type (line 380)
        dtype_371482 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 380, 51), indptr_371481, 'dtype')
        keyword_371483 = dtype_371482
        kwargs_371484 = {'dtype': keyword_371483}
        # Getting the type of 'np' (line 380)
        np_371472 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 380, 17), 'np', False)
        # Obtaining the member 'array' of a type (line 380)
        array_371473 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 380, 17), np_371472, 'array')
        # Calling array(args, kwargs) (line 380)
        array_call_result_371485 = invoke(stypy.reporting.localization.Localization(__file__, 380, 17), array_371473, *[list_371474], **kwargs_371484)
        
        # Assigning a type to the variable 'indptr' (line 380)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 380, 8), 'indptr', array_call_result_371485)
        
        # Call to csr_matrix(...): (line 381)
        # Processing the call arguments (line 381)
        
        # Obtaining an instance of the builtin type 'tuple' (line 381)
        tuple_371487 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 381, 27), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 381)
        # Adding element type (line 381)
        # Getting the type of 'data' (line 381)
        data_371488 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 381, 27), 'data', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 381, 27), tuple_371487, data_371488)
        # Adding element type (line 381)
        # Getting the type of 'indices' (line 381)
        indices_371489 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 381, 33), 'indices', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 381, 27), tuple_371487, indices_371489)
        # Adding element type (line 381)
        # Getting the type of 'indptr' (line 381)
        indptr_371490 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 381, 42), 'indptr', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 381, 27), tuple_371487, indptr_371490)
        
        # Processing the call keyword arguments (line 381)
        
        # Obtaining an instance of the builtin type 'tuple' (line 381)
        tuple_371491 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 381, 58), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 381)
        # Adding element type (line 381)
        int_371492 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 381, 58), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 381, 58), tuple_371491, int_371492)
        # Adding element type (line 381)
        # Getting the type of 'N' (line 381)
        N_371493 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 381, 61), 'N', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 381, 58), tuple_371491, N_371493)
        
        keyword_371494 = tuple_371491
        # Getting the type of 'self' (line 382)
        self_371495 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 382, 32), 'self', False)
        # Obtaining the member 'dtype' of a type (line 382)
        dtype_371496 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 382, 32), self_371495, 'dtype')
        keyword_371497 = dtype_371496
        # Getting the type of 'False' (line 382)
        False_371498 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 382, 49), 'False', False)
        keyword_371499 = False_371498
        kwargs_371500 = {'dtype': keyword_371497, 'shape': keyword_371494, 'copy': keyword_371499}
        # Getting the type of 'csr_matrix' (line 381)
        csr_matrix_371486 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 381, 15), 'csr_matrix', False)
        # Calling csr_matrix(args, kwargs) (line 381)
        csr_matrix_call_result_371501 = invoke(stypy.reporting.localization.Localization(__file__, 381, 15), csr_matrix_371486, *[tuple_371487], **kwargs_371500)
        
        # Assigning a type to the variable 'stypy_return_type' (line 381)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 381, 8), 'stypy_return_type', csr_matrix_call_result_371501)
        
        # ################# End of 'getrow(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'getrow' in the type store
        # Getting the type of 'stypy_return_type' (line 367)
        stypy_return_type_371502 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 367, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_371502)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'getrow'
        return stypy_return_type_371502


    @norecursion
    def getcol(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'getcol'
        module_type_store = module_type_store.open_function_context('getcol', 384, 4, False)
        # Assigning a type to the variable 'self' (line 385)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 385, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        csr_matrix.getcol.__dict__.__setitem__('stypy_localization', localization)
        csr_matrix.getcol.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        csr_matrix.getcol.__dict__.__setitem__('stypy_type_store', module_type_store)
        csr_matrix.getcol.__dict__.__setitem__('stypy_function_name', 'csr_matrix.getcol')
        csr_matrix.getcol.__dict__.__setitem__('stypy_param_names_list', ['i'])
        csr_matrix.getcol.__dict__.__setitem__('stypy_varargs_param_name', None)
        csr_matrix.getcol.__dict__.__setitem__('stypy_kwargs_param_name', None)
        csr_matrix.getcol.__dict__.__setitem__('stypy_call_defaults', defaults)
        csr_matrix.getcol.__dict__.__setitem__('stypy_call_varargs', varargs)
        csr_matrix.getcol.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        csr_matrix.getcol.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'csr_matrix.getcol', ['i'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'getcol', localization, ['i'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'getcol(...)' code ##################

        str_371503 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 387, (-1)), 'str', 'Returns a copy of column i of the matrix, as a (m x 1)\n        CSR matrix (column vector).\n        ')
        
        # Call to _get_submatrix(...): (line 388)
        # Processing the call arguments (line 388)
        
        # Call to slice(...): (line 388)
        # Processing the call arguments (line 388)
        # Getting the type of 'None' (line 388)
        None_371507 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 388, 41), 'None', False)
        # Processing the call keyword arguments (line 388)
        kwargs_371508 = {}
        # Getting the type of 'slice' (line 388)
        slice_371506 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 388, 35), 'slice', False)
        # Calling slice(args, kwargs) (line 388)
        slice_call_result_371509 = invoke(stypy.reporting.localization.Localization(__file__, 388, 35), slice_371506, *[None_371507], **kwargs_371508)
        
        # Getting the type of 'i' (line 388)
        i_371510 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 388, 48), 'i', False)
        # Processing the call keyword arguments (line 388)
        kwargs_371511 = {}
        # Getting the type of 'self' (line 388)
        self_371504 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 388, 15), 'self', False)
        # Obtaining the member '_get_submatrix' of a type (line 388)
        _get_submatrix_371505 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 388, 15), self_371504, '_get_submatrix')
        # Calling _get_submatrix(args, kwargs) (line 388)
        _get_submatrix_call_result_371512 = invoke(stypy.reporting.localization.Localization(__file__, 388, 15), _get_submatrix_371505, *[slice_call_result_371509, i_371510], **kwargs_371511)
        
        # Assigning a type to the variable 'stypy_return_type' (line 388)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 388, 8), 'stypy_return_type', _get_submatrix_call_result_371512)
        
        # ################# End of 'getcol(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'getcol' in the type store
        # Getting the type of 'stypy_return_type' (line 384)
        stypy_return_type_371513 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 384, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_371513)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'getcol'
        return stypy_return_type_371513


    @norecursion
    def _get_row_slice(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_get_row_slice'
        module_type_store = module_type_store.open_function_context('_get_row_slice', 390, 4, False)
        # Assigning a type to the variable 'self' (line 391)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 391, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        csr_matrix._get_row_slice.__dict__.__setitem__('stypy_localization', localization)
        csr_matrix._get_row_slice.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        csr_matrix._get_row_slice.__dict__.__setitem__('stypy_type_store', module_type_store)
        csr_matrix._get_row_slice.__dict__.__setitem__('stypy_function_name', 'csr_matrix._get_row_slice')
        csr_matrix._get_row_slice.__dict__.__setitem__('stypy_param_names_list', ['i', 'cslice'])
        csr_matrix._get_row_slice.__dict__.__setitem__('stypy_varargs_param_name', None)
        csr_matrix._get_row_slice.__dict__.__setitem__('stypy_kwargs_param_name', None)
        csr_matrix._get_row_slice.__dict__.__setitem__('stypy_call_defaults', defaults)
        csr_matrix._get_row_slice.__dict__.__setitem__('stypy_call_varargs', varargs)
        csr_matrix._get_row_slice.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        csr_matrix._get_row_slice.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'csr_matrix._get_row_slice', ['i', 'cslice'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_get_row_slice', localization, ['i', 'cslice'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_get_row_slice(...)' code ##################

        str_371514 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 392, (-1)), 'str', 'Returns a copy of row self[i, cslice]\n        ')
        
        # Assigning a Attribute to a Tuple (line 393):
        
        # Assigning a Subscript to a Name (line 393):
        
        # Obtaining the type of the subscript
        int_371515 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 393, 8), 'int')
        # Getting the type of 'self' (line 393)
        self_371516 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 393, 15), 'self')
        # Obtaining the member 'shape' of a type (line 393)
        shape_371517 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 393, 15), self_371516, 'shape')
        # Obtaining the member '__getitem__' of a type (line 393)
        getitem___371518 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 393, 8), shape_371517, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 393)
        subscript_call_result_371519 = invoke(stypy.reporting.localization.Localization(__file__, 393, 8), getitem___371518, int_371515)
        
        # Assigning a type to the variable 'tuple_var_assignment_370337' (line 393)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 393, 8), 'tuple_var_assignment_370337', subscript_call_result_371519)
        
        # Assigning a Subscript to a Name (line 393):
        
        # Obtaining the type of the subscript
        int_371520 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 393, 8), 'int')
        # Getting the type of 'self' (line 393)
        self_371521 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 393, 15), 'self')
        # Obtaining the member 'shape' of a type (line 393)
        shape_371522 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 393, 15), self_371521, 'shape')
        # Obtaining the member '__getitem__' of a type (line 393)
        getitem___371523 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 393, 8), shape_371522, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 393)
        subscript_call_result_371524 = invoke(stypy.reporting.localization.Localization(__file__, 393, 8), getitem___371523, int_371520)
        
        # Assigning a type to the variable 'tuple_var_assignment_370338' (line 393)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 393, 8), 'tuple_var_assignment_370338', subscript_call_result_371524)
        
        # Assigning a Name to a Name (line 393):
        # Getting the type of 'tuple_var_assignment_370337' (line 393)
        tuple_var_assignment_370337_371525 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 393, 8), 'tuple_var_assignment_370337')
        # Assigning a type to the variable 'M' (line 393)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 393, 8), 'M', tuple_var_assignment_370337_371525)
        
        # Assigning a Name to a Name (line 393):
        # Getting the type of 'tuple_var_assignment_370338' (line 393)
        tuple_var_assignment_370338_371526 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 393, 8), 'tuple_var_assignment_370338')
        # Assigning a type to the variable 'N' (line 393)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 393, 11), 'N', tuple_var_assignment_370338_371526)
        
        
        # Getting the type of 'i' (line 395)
        i_371527 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 395, 11), 'i')
        int_371528 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 395, 15), 'int')
        # Applying the binary operator '<' (line 395)
        result_lt_371529 = python_operator(stypy.reporting.localization.Localization(__file__, 395, 11), '<', i_371527, int_371528)
        
        # Testing the type of an if condition (line 395)
        if_condition_371530 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 395, 8), result_lt_371529)
        # Assigning a type to the variable 'if_condition_371530' (line 395)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 395, 8), 'if_condition_371530', if_condition_371530)
        # SSA begins for if statement (line 395)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Getting the type of 'i' (line 396)
        i_371531 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 396, 12), 'i')
        # Getting the type of 'M' (line 396)
        M_371532 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 396, 17), 'M')
        # Applying the binary operator '+=' (line 396)
        result_iadd_371533 = python_operator(stypy.reporting.localization.Localization(__file__, 396, 12), '+=', i_371531, M_371532)
        # Assigning a type to the variable 'i' (line 396)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 396, 12), 'i', result_iadd_371533)
        
        # SSA join for if statement (line 395)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Evaluating a boolean operation
        
        # Getting the type of 'i' (line 398)
        i_371534 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 398, 11), 'i')
        int_371535 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 398, 15), 'int')
        # Applying the binary operator '<' (line 398)
        result_lt_371536 = python_operator(stypy.reporting.localization.Localization(__file__, 398, 11), '<', i_371534, int_371535)
        
        
        # Getting the type of 'i' (line 398)
        i_371537 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 398, 20), 'i')
        # Getting the type of 'M' (line 398)
        M_371538 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 398, 25), 'M')
        # Applying the binary operator '>=' (line 398)
        result_ge_371539 = python_operator(stypy.reporting.localization.Localization(__file__, 398, 20), '>=', i_371537, M_371538)
        
        # Applying the binary operator 'or' (line 398)
        result_or_keyword_371540 = python_operator(stypy.reporting.localization.Localization(__file__, 398, 11), 'or', result_lt_371536, result_ge_371539)
        
        # Testing the type of an if condition (line 398)
        if_condition_371541 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 398, 8), result_or_keyword_371540)
        # Assigning a type to the variable 'if_condition_371541' (line 398)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 398, 8), 'if_condition_371541', if_condition_371541)
        # SSA begins for if statement (line 398)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to IndexError(...): (line 399)
        # Processing the call arguments (line 399)
        str_371543 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 399, 29), 'str', 'index (%d) out of range')
        # Getting the type of 'i' (line 399)
        i_371544 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 399, 57), 'i', False)
        # Applying the binary operator '%' (line 399)
        result_mod_371545 = python_operator(stypy.reporting.localization.Localization(__file__, 399, 29), '%', str_371543, i_371544)
        
        # Processing the call keyword arguments (line 399)
        kwargs_371546 = {}
        # Getting the type of 'IndexError' (line 399)
        IndexError_371542 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 399, 18), 'IndexError', False)
        # Calling IndexError(args, kwargs) (line 399)
        IndexError_call_result_371547 = invoke(stypy.reporting.localization.Localization(__file__, 399, 18), IndexError_371542, *[result_mod_371545], **kwargs_371546)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 399, 12), IndexError_call_result_371547, 'raise parameter', BaseException)
        # SSA join for if statement (line 398)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Tuple (line 401):
        
        # Assigning a Subscript to a Name (line 401):
        
        # Obtaining the type of the subscript
        int_371548 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 401, 8), 'int')
        
        # Call to indices(...): (line 401)
        # Processing the call arguments (line 401)
        # Getting the type of 'N' (line 401)
        N_371551 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 401, 45), 'N', False)
        # Processing the call keyword arguments (line 401)
        kwargs_371552 = {}
        # Getting the type of 'cslice' (line 401)
        cslice_371549 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 401, 30), 'cslice', False)
        # Obtaining the member 'indices' of a type (line 401)
        indices_371550 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 401, 30), cslice_371549, 'indices')
        # Calling indices(args, kwargs) (line 401)
        indices_call_result_371553 = invoke(stypy.reporting.localization.Localization(__file__, 401, 30), indices_371550, *[N_371551], **kwargs_371552)
        
        # Obtaining the member '__getitem__' of a type (line 401)
        getitem___371554 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 401, 8), indices_call_result_371553, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 401)
        subscript_call_result_371555 = invoke(stypy.reporting.localization.Localization(__file__, 401, 8), getitem___371554, int_371548)
        
        # Assigning a type to the variable 'tuple_var_assignment_370339' (line 401)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 401, 8), 'tuple_var_assignment_370339', subscript_call_result_371555)
        
        # Assigning a Subscript to a Name (line 401):
        
        # Obtaining the type of the subscript
        int_371556 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 401, 8), 'int')
        
        # Call to indices(...): (line 401)
        # Processing the call arguments (line 401)
        # Getting the type of 'N' (line 401)
        N_371559 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 401, 45), 'N', False)
        # Processing the call keyword arguments (line 401)
        kwargs_371560 = {}
        # Getting the type of 'cslice' (line 401)
        cslice_371557 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 401, 30), 'cslice', False)
        # Obtaining the member 'indices' of a type (line 401)
        indices_371558 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 401, 30), cslice_371557, 'indices')
        # Calling indices(args, kwargs) (line 401)
        indices_call_result_371561 = invoke(stypy.reporting.localization.Localization(__file__, 401, 30), indices_371558, *[N_371559], **kwargs_371560)
        
        # Obtaining the member '__getitem__' of a type (line 401)
        getitem___371562 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 401, 8), indices_call_result_371561, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 401)
        subscript_call_result_371563 = invoke(stypy.reporting.localization.Localization(__file__, 401, 8), getitem___371562, int_371556)
        
        # Assigning a type to the variable 'tuple_var_assignment_370340' (line 401)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 401, 8), 'tuple_var_assignment_370340', subscript_call_result_371563)
        
        # Assigning a Subscript to a Name (line 401):
        
        # Obtaining the type of the subscript
        int_371564 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 401, 8), 'int')
        
        # Call to indices(...): (line 401)
        # Processing the call arguments (line 401)
        # Getting the type of 'N' (line 401)
        N_371567 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 401, 45), 'N', False)
        # Processing the call keyword arguments (line 401)
        kwargs_371568 = {}
        # Getting the type of 'cslice' (line 401)
        cslice_371565 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 401, 30), 'cslice', False)
        # Obtaining the member 'indices' of a type (line 401)
        indices_371566 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 401, 30), cslice_371565, 'indices')
        # Calling indices(args, kwargs) (line 401)
        indices_call_result_371569 = invoke(stypy.reporting.localization.Localization(__file__, 401, 30), indices_371566, *[N_371567], **kwargs_371568)
        
        # Obtaining the member '__getitem__' of a type (line 401)
        getitem___371570 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 401, 8), indices_call_result_371569, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 401)
        subscript_call_result_371571 = invoke(stypy.reporting.localization.Localization(__file__, 401, 8), getitem___371570, int_371564)
        
        # Assigning a type to the variable 'tuple_var_assignment_370341' (line 401)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 401, 8), 'tuple_var_assignment_370341', subscript_call_result_371571)
        
        # Assigning a Name to a Name (line 401):
        # Getting the type of 'tuple_var_assignment_370339' (line 401)
        tuple_var_assignment_370339_371572 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 401, 8), 'tuple_var_assignment_370339')
        # Assigning a type to the variable 'start' (line 401)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 401, 8), 'start', tuple_var_assignment_370339_371572)
        
        # Assigning a Name to a Name (line 401):
        # Getting the type of 'tuple_var_assignment_370340' (line 401)
        tuple_var_assignment_370340_371573 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 401, 8), 'tuple_var_assignment_370340')
        # Assigning a type to the variable 'stop' (line 401)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 401, 15), 'stop', tuple_var_assignment_370340_371573)
        
        # Assigning a Name to a Name (line 401):
        # Getting the type of 'tuple_var_assignment_370341' (line 401)
        tuple_var_assignment_370341_371574 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 401, 8), 'tuple_var_assignment_370341')
        # Assigning a type to the variable 'stride' (line 401)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 401, 21), 'stride', tuple_var_assignment_370341_371574)
        
        
        # Getting the type of 'stride' (line 403)
        stride_371575 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 403, 11), 'stride')
        int_371576 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 403, 21), 'int')
        # Applying the binary operator '==' (line 403)
        result_eq_371577 = python_operator(stypy.reporting.localization.Localization(__file__, 403, 11), '==', stride_371575, int_371576)
        
        # Testing the type of an if condition (line 403)
        if_condition_371578 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 403, 8), result_eq_371577)
        # Assigning a type to the variable 'if_condition_371578' (line 403)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 403, 8), 'if_condition_371578', if_condition_371578)
        # SSA begins for if statement (line 403)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Tuple (line 405):
        
        # Assigning a Subscript to a Name (line 405):
        
        # Obtaining the type of the subscript
        int_371579 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 405, 12), 'int')
        
        # Call to get_csr_submatrix(...): (line 405)
        # Processing the call arguments (line 405)
        # Getting the type of 'M' (line 406)
        M_371581 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 406, 16), 'M', False)
        # Getting the type of 'N' (line 406)
        N_371582 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 406, 19), 'N', False)
        # Getting the type of 'self' (line 406)
        self_371583 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 406, 22), 'self', False)
        # Obtaining the member 'indptr' of a type (line 406)
        indptr_371584 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 406, 22), self_371583, 'indptr')
        # Getting the type of 'self' (line 406)
        self_371585 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 406, 35), 'self', False)
        # Obtaining the member 'indices' of a type (line 406)
        indices_371586 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 406, 35), self_371585, 'indices')
        # Getting the type of 'self' (line 406)
        self_371587 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 406, 49), 'self', False)
        # Obtaining the member 'data' of a type (line 406)
        data_371588 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 406, 49), self_371587, 'data')
        # Getting the type of 'i' (line 406)
        i_371589 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 406, 60), 'i', False)
        # Getting the type of 'i' (line 406)
        i_371590 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 406, 63), 'i', False)
        int_371591 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 406, 67), 'int')
        # Applying the binary operator '+' (line 406)
        result_add_371592 = python_operator(stypy.reporting.localization.Localization(__file__, 406, 63), '+', i_371590, int_371591)
        
        # Getting the type of 'start' (line 407)
        start_371593 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 407, 16), 'start', False)
        # Getting the type of 'stop' (line 407)
        stop_371594 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 407, 23), 'stop', False)
        # Processing the call keyword arguments (line 405)
        kwargs_371595 = {}
        # Getting the type of 'get_csr_submatrix' (line 405)
        get_csr_submatrix_371580 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 405, 48), 'get_csr_submatrix', False)
        # Calling get_csr_submatrix(args, kwargs) (line 405)
        get_csr_submatrix_call_result_371596 = invoke(stypy.reporting.localization.Localization(__file__, 405, 48), get_csr_submatrix_371580, *[M_371581, N_371582, indptr_371584, indices_371586, data_371588, i_371589, result_add_371592, start_371593, stop_371594], **kwargs_371595)
        
        # Obtaining the member '__getitem__' of a type (line 405)
        getitem___371597 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 405, 12), get_csr_submatrix_call_result_371596, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 405)
        subscript_call_result_371598 = invoke(stypy.reporting.localization.Localization(__file__, 405, 12), getitem___371597, int_371579)
        
        # Assigning a type to the variable 'tuple_var_assignment_370342' (line 405)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 405, 12), 'tuple_var_assignment_370342', subscript_call_result_371598)
        
        # Assigning a Subscript to a Name (line 405):
        
        # Obtaining the type of the subscript
        int_371599 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 405, 12), 'int')
        
        # Call to get_csr_submatrix(...): (line 405)
        # Processing the call arguments (line 405)
        # Getting the type of 'M' (line 406)
        M_371601 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 406, 16), 'M', False)
        # Getting the type of 'N' (line 406)
        N_371602 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 406, 19), 'N', False)
        # Getting the type of 'self' (line 406)
        self_371603 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 406, 22), 'self', False)
        # Obtaining the member 'indptr' of a type (line 406)
        indptr_371604 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 406, 22), self_371603, 'indptr')
        # Getting the type of 'self' (line 406)
        self_371605 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 406, 35), 'self', False)
        # Obtaining the member 'indices' of a type (line 406)
        indices_371606 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 406, 35), self_371605, 'indices')
        # Getting the type of 'self' (line 406)
        self_371607 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 406, 49), 'self', False)
        # Obtaining the member 'data' of a type (line 406)
        data_371608 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 406, 49), self_371607, 'data')
        # Getting the type of 'i' (line 406)
        i_371609 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 406, 60), 'i', False)
        # Getting the type of 'i' (line 406)
        i_371610 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 406, 63), 'i', False)
        int_371611 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 406, 67), 'int')
        # Applying the binary operator '+' (line 406)
        result_add_371612 = python_operator(stypy.reporting.localization.Localization(__file__, 406, 63), '+', i_371610, int_371611)
        
        # Getting the type of 'start' (line 407)
        start_371613 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 407, 16), 'start', False)
        # Getting the type of 'stop' (line 407)
        stop_371614 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 407, 23), 'stop', False)
        # Processing the call keyword arguments (line 405)
        kwargs_371615 = {}
        # Getting the type of 'get_csr_submatrix' (line 405)
        get_csr_submatrix_371600 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 405, 48), 'get_csr_submatrix', False)
        # Calling get_csr_submatrix(args, kwargs) (line 405)
        get_csr_submatrix_call_result_371616 = invoke(stypy.reporting.localization.Localization(__file__, 405, 48), get_csr_submatrix_371600, *[M_371601, N_371602, indptr_371604, indices_371606, data_371608, i_371609, result_add_371612, start_371613, stop_371614], **kwargs_371615)
        
        # Obtaining the member '__getitem__' of a type (line 405)
        getitem___371617 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 405, 12), get_csr_submatrix_call_result_371616, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 405)
        subscript_call_result_371618 = invoke(stypy.reporting.localization.Localization(__file__, 405, 12), getitem___371617, int_371599)
        
        # Assigning a type to the variable 'tuple_var_assignment_370343' (line 405)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 405, 12), 'tuple_var_assignment_370343', subscript_call_result_371618)
        
        # Assigning a Subscript to a Name (line 405):
        
        # Obtaining the type of the subscript
        int_371619 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 405, 12), 'int')
        
        # Call to get_csr_submatrix(...): (line 405)
        # Processing the call arguments (line 405)
        # Getting the type of 'M' (line 406)
        M_371621 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 406, 16), 'M', False)
        # Getting the type of 'N' (line 406)
        N_371622 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 406, 19), 'N', False)
        # Getting the type of 'self' (line 406)
        self_371623 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 406, 22), 'self', False)
        # Obtaining the member 'indptr' of a type (line 406)
        indptr_371624 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 406, 22), self_371623, 'indptr')
        # Getting the type of 'self' (line 406)
        self_371625 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 406, 35), 'self', False)
        # Obtaining the member 'indices' of a type (line 406)
        indices_371626 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 406, 35), self_371625, 'indices')
        # Getting the type of 'self' (line 406)
        self_371627 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 406, 49), 'self', False)
        # Obtaining the member 'data' of a type (line 406)
        data_371628 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 406, 49), self_371627, 'data')
        # Getting the type of 'i' (line 406)
        i_371629 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 406, 60), 'i', False)
        # Getting the type of 'i' (line 406)
        i_371630 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 406, 63), 'i', False)
        int_371631 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 406, 67), 'int')
        # Applying the binary operator '+' (line 406)
        result_add_371632 = python_operator(stypy.reporting.localization.Localization(__file__, 406, 63), '+', i_371630, int_371631)
        
        # Getting the type of 'start' (line 407)
        start_371633 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 407, 16), 'start', False)
        # Getting the type of 'stop' (line 407)
        stop_371634 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 407, 23), 'stop', False)
        # Processing the call keyword arguments (line 405)
        kwargs_371635 = {}
        # Getting the type of 'get_csr_submatrix' (line 405)
        get_csr_submatrix_371620 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 405, 48), 'get_csr_submatrix', False)
        # Calling get_csr_submatrix(args, kwargs) (line 405)
        get_csr_submatrix_call_result_371636 = invoke(stypy.reporting.localization.Localization(__file__, 405, 48), get_csr_submatrix_371620, *[M_371621, N_371622, indptr_371624, indices_371626, data_371628, i_371629, result_add_371632, start_371633, stop_371634], **kwargs_371635)
        
        # Obtaining the member '__getitem__' of a type (line 405)
        getitem___371637 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 405, 12), get_csr_submatrix_call_result_371636, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 405)
        subscript_call_result_371638 = invoke(stypy.reporting.localization.Localization(__file__, 405, 12), getitem___371637, int_371619)
        
        # Assigning a type to the variable 'tuple_var_assignment_370344' (line 405)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 405, 12), 'tuple_var_assignment_370344', subscript_call_result_371638)
        
        # Assigning a Name to a Name (line 405):
        # Getting the type of 'tuple_var_assignment_370342' (line 405)
        tuple_var_assignment_370342_371639 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 405, 12), 'tuple_var_assignment_370342')
        # Assigning a type to the variable 'row_indptr' (line 405)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 405, 12), 'row_indptr', tuple_var_assignment_370342_371639)
        
        # Assigning a Name to a Name (line 405):
        # Getting the type of 'tuple_var_assignment_370343' (line 405)
        tuple_var_assignment_370343_371640 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 405, 12), 'tuple_var_assignment_370343')
        # Assigning a type to the variable 'row_indices' (line 405)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 405, 24), 'row_indices', tuple_var_assignment_370343_371640)
        
        # Assigning a Name to a Name (line 405):
        # Getting the type of 'tuple_var_assignment_370344' (line 405)
        tuple_var_assignment_370344_371641 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 405, 12), 'tuple_var_assignment_370344')
        # Assigning a type to the variable 'row_data' (line 405)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 405, 37), 'row_data', tuple_var_assignment_370344_371641)
        # SSA branch for the else part of an if statement (line 403)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Subscript to a Name (line 410):
        
        # Assigning a Subscript to a Name (line 410):
        
        # Obtaining the type of the subscript
        
        # Obtaining the type of the subscript
        # Getting the type of 'i' (line 410)
        i_371642 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 410, 51), 'i')
        # Getting the type of 'self' (line 410)
        self_371643 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 410, 39), 'self')
        # Obtaining the member 'indptr' of a type (line 410)
        indptr_371644 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 410, 39), self_371643, 'indptr')
        # Obtaining the member '__getitem__' of a type (line 410)
        getitem___371645 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 410, 39), indptr_371644, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 410)
        subscript_call_result_371646 = invoke(stypy.reporting.localization.Localization(__file__, 410, 39), getitem___371645, i_371642)
        
        
        # Obtaining the type of the subscript
        # Getting the type of 'i' (line 410)
        i_371647 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 410, 66), 'i')
        int_371648 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 410, 70), 'int')
        # Applying the binary operator '+' (line 410)
        result_add_371649 = python_operator(stypy.reporting.localization.Localization(__file__, 410, 66), '+', i_371647, int_371648)
        
        # Getting the type of 'self' (line 410)
        self_371650 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 410, 54), 'self')
        # Obtaining the member 'indptr' of a type (line 410)
        indptr_371651 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 410, 54), self_371650, 'indptr')
        # Obtaining the member '__getitem__' of a type (line 410)
        getitem___371652 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 410, 54), indptr_371651, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 410)
        subscript_call_result_371653 = invoke(stypy.reporting.localization.Localization(__file__, 410, 54), getitem___371652, result_add_371649)
        
        slice_371654 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 410, 26), subscript_call_result_371646, subscript_call_result_371653, None)
        # Getting the type of 'self' (line 410)
        self_371655 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 410, 26), 'self')
        # Obtaining the member 'indices' of a type (line 410)
        indices_371656 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 410, 26), self_371655, 'indices')
        # Obtaining the member '__getitem__' of a type (line 410)
        getitem___371657 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 410, 26), indices_371656, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 410)
        subscript_call_result_371658 = invoke(stypy.reporting.localization.Localization(__file__, 410, 26), getitem___371657, slice_371654)
        
        # Assigning a type to the variable 'row_indices' (line 410)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 410, 12), 'row_indices', subscript_call_result_371658)
        
        # Assigning a Subscript to a Name (line 411):
        
        # Assigning a Subscript to a Name (line 411):
        
        # Obtaining the type of the subscript
        
        # Obtaining the type of the subscript
        # Getting the type of 'i' (line 411)
        i_371659 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 411, 45), 'i')
        # Getting the type of 'self' (line 411)
        self_371660 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 411, 33), 'self')
        # Obtaining the member 'indptr' of a type (line 411)
        indptr_371661 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 411, 33), self_371660, 'indptr')
        # Obtaining the member '__getitem__' of a type (line 411)
        getitem___371662 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 411, 33), indptr_371661, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 411)
        subscript_call_result_371663 = invoke(stypy.reporting.localization.Localization(__file__, 411, 33), getitem___371662, i_371659)
        
        
        # Obtaining the type of the subscript
        # Getting the type of 'i' (line 411)
        i_371664 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 411, 60), 'i')
        int_371665 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 411, 64), 'int')
        # Applying the binary operator '+' (line 411)
        result_add_371666 = python_operator(stypy.reporting.localization.Localization(__file__, 411, 60), '+', i_371664, int_371665)
        
        # Getting the type of 'self' (line 411)
        self_371667 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 411, 48), 'self')
        # Obtaining the member 'indptr' of a type (line 411)
        indptr_371668 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 411, 48), self_371667, 'indptr')
        # Obtaining the member '__getitem__' of a type (line 411)
        getitem___371669 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 411, 48), indptr_371668, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 411)
        subscript_call_result_371670 = invoke(stypy.reporting.localization.Localization(__file__, 411, 48), getitem___371669, result_add_371666)
        
        slice_371671 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 411, 23), subscript_call_result_371663, subscript_call_result_371670, None)
        # Getting the type of 'self' (line 411)
        self_371672 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 411, 23), 'self')
        # Obtaining the member 'data' of a type (line 411)
        data_371673 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 411, 23), self_371672, 'data')
        # Obtaining the member '__getitem__' of a type (line 411)
        getitem___371674 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 411, 23), data_371673, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 411)
        subscript_call_result_371675 = invoke(stypy.reporting.localization.Localization(__file__, 411, 23), getitem___371674, slice_371671)
        
        # Assigning a type to the variable 'row_data' (line 411)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 411, 12), 'row_data', subscript_call_result_371675)
        
        
        # Getting the type of 'stride' (line 413)
        stride_371676 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 413, 15), 'stride')
        int_371677 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 413, 24), 'int')
        # Applying the binary operator '>' (line 413)
        result_gt_371678 = python_operator(stypy.reporting.localization.Localization(__file__, 413, 15), '>', stride_371676, int_371677)
        
        # Testing the type of an if condition (line 413)
        if_condition_371679 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 413, 12), result_gt_371678)
        # Assigning a type to the variable 'if_condition_371679' (line 413)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 413, 12), 'if_condition_371679', if_condition_371679)
        # SSA begins for if statement (line 413)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a BinOp to a Name (line 414):
        
        # Assigning a BinOp to a Name (line 414):
        
        # Getting the type of 'row_indices' (line 414)
        row_indices_371680 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 414, 23), 'row_indices')
        # Getting the type of 'start' (line 414)
        start_371681 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 414, 38), 'start')
        # Applying the binary operator '>=' (line 414)
        result_ge_371682 = python_operator(stypy.reporting.localization.Localization(__file__, 414, 23), '>=', row_indices_371680, start_371681)
        
        
        # Getting the type of 'row_indices' (line 414)
        row_indices_371683 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 414, 48), 'row_indices')
        # Getting the type of 'stop' (line 414)
        stop_371684 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 414, 62), 'stop')
        # Applying the binary operator '<' (line 414)
        result_lt_371685 = python_operator(stypy.reporting.localization.Localization(__file__, 414, 48), '<', row_indices_371683, stop_371684)
        
        # Applying the binary operator '&' (line 414)
        result_and__371686 = python_operator(stypy.reporting.localization.Localization(__file__, 414, 22), '&', result_ge_371682, result_lt_371685)
        
        # Assigning a type to the variable 'ind' (line 414)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 414, 16), 'ind', result_and__371686)
        # SSA branch for the else part of an if statement (line 413)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a BinOp to a Name (line 416):
        
        # Assigning a BinOp to a Name (line 416):
        
        # Getting the type of 'row_indices' (line 416)
        row_indices_371687 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 416, 23), 'row_indices')
        # Getting the type of 'start' (line 416)
        start_371688 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 416, 38), 'start')
        # Applying the binary operator '<=' (line 416)
        result_le_371689 = python_operator(stypy.reporting.localization.Localization(__file__, 416, 23), '<=', row_indices_371687, start_371688)
        
        
        # Getting the type of 'row_indices' (line 416)
        row_indices_371690 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 416, 48), 'row_indices')
        # Getting the type of 'stop' (line 416)
        stop_371691 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 416, 62), 'stop')
        # Applying the binary operator '>' (line 416)
        result_gt_371692 = python_operator(stypy.reporting.localization.Localization(__file__, 416, 48), '>', row_indices_371690, stop_371691)
        
        # Applying the binary operator '&' (line 416)
        result_and__371693 = python_operator(stypy.reporting.localization.Localization(__file__, 416, 22), '&', result_le_371689, result_gt_371692)
        
        # Assigning a type to the variable 'ind' (line 416)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 416, 16), 'ind', result_and__371693)
        # SSA join for if statement (line 413)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        
        # Call to abs(...): (line 418)
        # Processing the call arguments (line 418)
        # Getting the type of 'stride' (line 418)
        stride_371695 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 418, 19), 'stride', False)
        # Processing the call keyword arguments (line 418)
        kwargs_371696 = {}
        # Getting the type of 'abs' (line 418)
        abs_371694 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 418, 15), 'abs', False)
        # Calling abs(args, kwargs) (line 418)
        abs_call_result_371697 = invoke(stypy.reporting.localization.Localization(__file__, 418, 15), abs_371694, *[stride_371695], **kwargs_371696)
        
        int_371698 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 418, 29), 'int')
        # Applying the binary operator '>' (line 418)
        result_gt_371699 = python_operator(stypy.reporting.localization.Localization(__file__, 418, 15), '>', abs_call_result_371697, int_371698)
        
        # Testing the type of an if condition (line 418)
        if_condition_371700 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 418, 12), result_gt_371699)
        # Assigning a type to the variable 'if_condition_371700' (line 418)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 418, 12), 'if_condition_371700', if_condition_371700)
        # SSA begins for if statement (line 418)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Getting the type of 'ind' (line 419)
        ind_371701 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 419, 16), 'ind')
        
        # Getting the type of 'row_indices' (line 419)
        row_indices_371702 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 419, 24), 'row_indices')
        # Getting the type of 'start' (line 419)
        start_371703 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 419, 38), 'start')
        # Applying the binary operator '-' (line 419)
        result_sub_371704 = python_operator(stypy.reporting.localization.Localization(__file__, 419, 24), '-', row_indices_371702, start_371703)
        
        # Getting the type of 'stride' (line 419)
        stride_371705 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 419, 47), 'stride')
        # Applying the binary operator '%' (line 419)
        result_mod_371706 = python_operator(stypy.reporting.localization.Localization(__file__, 419, 23), '%', result_sub_371704, stride_371705)
        
        int_371707 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 419, 57), 'int')
        # Applying the binary operator '==' (line 419)
        result_eq_371708 = python_operator(stypy.reporting.localization.Localization(__file__, 419, 23), '==', result_mod_371706, int_371707)
        
        # Applying the binary operator '&=' (line 419)
        result_iand_371709 = python_operator(stypy.reporting.localization.Localization(__file__, 419, 16), '&=', ind_371701, result_eq_371708)
        # Assigning a type to the variable 'ind' (line 419)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 419, 16), 'ind', result_iand_371709)
        
        # SSA join for if statement (line 418)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a BinOp to a Name (line 421):
        
        # Assigning a BinOp to a Name (line 421):
        
        # Obtaining the type of the subscript
        # Getting the type of 'ind' (line 421)
        ind_371710 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 421, 39), 'ind')
        # Getting the type of 'row_indices' (line 421)
        row_indices_371711 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 421, 27), 'row_indices')
        # Obtaining the member '__getitem__' of a type (line 421)
        getitem___371712 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 421, 27), row_indices_371711, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 421)
        subscript_call_result_371713 = invoke(stypy.reporting.localization.Localization(__file__, 421, 27), getitem___371712, ind_371710)
        
        # Getting the type of 'start' (line 421)
        start_371714 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 421, 46), 'start')
        # Applying the binary operator '-' (line 421)
        result_sub_371715 = python_operator(stypy.reporting.localization.Localization(__file__, 421, 27), '-', subscript_call_result_371713, start_371714)
        
        # Getting the type of 'stride' (line 421)
        stride_371716 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 421, 56), 'stride')
        # Applying the binary operator '//' (line 421)
        result_floordiv_371717 = python_operator(stypy.reporting.localization.Localization(__file__, 421, 26), '//', result_sub_371715, stride_371716)
        
        # Assigning a type to the variable 'row_indices' (line 421)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 421, 12), 'row_indices', result_floordiv_371717)
        
        # Assigning a Subscript to a Name (line 422):
        
        # Assigning a Subscript to a Name (line 422):
        
        # Obtaining the type of the subscript
        # Getting the type of 'ind' (line 422)
        ind_371718 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 422, 32), 'ind')
        # Getting the type of 'row_data' (line 422)
        row_data_371719 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 422, 23), 'row_data')
        # Obtaining the member '__getitem__' of a type (line 422)
        getitem___371720 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 422, 23), row_data_371719, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 422)
        subscript_call_result_371721 = invoke(stypy.reporting.localization.Localization(__file__, 422, 23), getitem___371720, ind_371718)
        
        # Assigning a type to the variable 'row_data' (line 422)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 422, 12), 'row_data', subscript_call_result_371721)
        
        # Assigning a Call to a Name (line 423):
        
        # Assigning a Call to a Name (line 423):
        
        # Call to array(...): (line 423)
        # Processing the call arguments (line 423)
        
        # Obtaining an instance of the builtin type 'list' (line 423)
        list_371724 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 423, 34), 'list')
        # Adding type elements to the builtin type 'list' instance (line 423)
        # Adding element type (line 423)
        int_371725 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 423, 35), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 423, 34), list_371724, int_371725)
        # Adding element type (line 423)
        
        # Call to len(...): (line 423)
        # Processing the call arguments (line 423)
        # Getting the type of 'row_indices' (line 423)
        row_indices_371727 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 423, 42), 'row_indices', False)
        # Processing the call keyword arguments (line 423)
        kwargs_371728 = {}
        # Getting the type of 'len' (line 423)
        len_371726 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 423, 38), 'len', False)
        # Calling len(args, kwargs) (line 423)
        len_call_result_371729 = invoke(stypy.reporting.localization.Localization(__file__, 423, 38), len_371726, *[row_indices_371727], **kwargs_371728)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 423, 34), list_371724, len_call_result_371729)
        
        # Processing the call keyword arguments (line 423)
        kwargs_371730 = {}
        # Getting the type of 'np' (line 423)
        np_371722 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 423, 25), 'np', False)
        # Obtaining the member 'array' of a type (line 423)
        array_371723 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 423, 25), np_371722, 'array')
        # Calling array(args, kwargs) (line 423)
        array_call_result_371731 = invoke(stypy.reporting.localization.Localization(__file__, 423, 25), array_371723, *[list_371724], **kwargs_371730)
        
        # Assigning a type to the variable 'row_indptr' (line 423)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 423, 12), 'row_indptr', array_call_result_371731)
        
        
        # Getting the type of 'stride' (line 425)
        stride_371732 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 425, 15), 'stride')
        int_371733 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 425, 24), 'int')
        # Applying the binary operator '<' (line 425)
        result_lt_371734 = python_operator(stypy.reporting.localization.Localization(__file__, 425, 15), '<', stride_371732, int_371733)
        
        # Testing the type of an if condition (line 425)
        if_condition_371735 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 425, 12), result_lt_371734)
        # Assigning a type to the variable 'if_condition_371735' (line 425)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 425, 12), 'if_condition_371735', if_condition_371735)
        # SSA begins for if statement (line 425)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Subscript to a Name (line 426):
        
        # Assigning a Subscript to a Name (line 426):
        
        # Obtaining the type of the subscript
        int_371736 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 426, 38), 'int')
        slice_371737 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 426, 27), None, None, int_371736)
        # Getting the type of 'row_data' (line 426)
        row_data_371738 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 426, 27), 'row_data')
        # Obtaining the member '__getitem__' of a type (line 426)
        getitem___371739 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 426, 27), row_data_371738, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 426)
        subscript_call_result_371740 = invoke(stypy.reporting.localization.Localization(__file__, 426, 27), getitem___371739, slice_371737)
        
        # Assigning a type to the variable 'row_data' (line 426)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 426, 16), 'row_data', subscript_call_result_371740)
        
        # Assigning a Call to a Name (line 427):
        
        # Assigning a Call to a Name (line 427):
        
        # Call to abs(...): (line 427)
        # Processing the call arguments (line 427)
        
        # Obtaining the type of the subscript
        int_371742 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 427, 48), 'int')
        slice_371743 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 427, 34), None, None, int_371742)
        # Getting the type of 'row_indices' (line 427)
        row_indices_371744 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 427, 34), 'row_indices', False)
        # Obtaining the member '__getitem__' of a type (line 427)
        getitem___371745 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 427, 34), row_indices_371744, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 427)
        subscript_call_result_371746 = invoke(stypy.reporting.localization.Localization(__file__, 427, 34), getitem___371745, slice_371743)
        
        # Processing the call keyword arguments (line 427)
        kwargs_371747 = {}
        # Getting the type of 'abs' (line 427)
        abs_371741 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 427, 30), 'abs', False)
        # Calling abs(args, kwargs) (line 427)
        abs_call_result_371748 = invoke(stypy.reporting.localization.Localization(__file__, 427, 30), abs_371741, *[subscript_call_result_371746], **kwargs_371747)
        
        # Assigning a type to the variable 'row_indices' (line 427)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 427, 16), 'row_indices', abs_call_result_371748)
        # SSA join for if statement (line 425)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 403)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Tuple to a Name (line 429):
        
        # Assigning a Tuple to a Name (line 429):
        
        # Obtaining an instance of the builtin type 'tuple' (line 429)
        tuple_371749 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 429, 17), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 429)
        # Adding element type (line 429)
        int_371750 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 429, 17), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 429, 17), tuple_371749, int_371750)
        # Adding element type (line 429)
        
        # Call to int(...): (line 429)
        # Processing the call arguments (line 429)
        
        # Call to ceil(...): (line 429)
        # Processing the call arguments (line 429)
        
        # Call to float(...): (line 429)
        # Processing the call arguments (line 429)
        # Getting the type of 'stop' (line 429)
        stop_371755 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 429, 38), 'stop', False)
        # Getting the type of 'start' (line 429)
        start_371756 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 429, 45), 'start', False)
        # Applying the binary operator '-' (line 429)
        result_sub_371757 = python_operator(stypy.reporting.localization.Localization(__file__, 429, 38), '-', stop_371755, start_371756)
        
        # Processing the call keyword arguments (line 429)
        kwargs_371758 = {}
        # Getting the type of 'float' (line 429)
        float_371754 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 429, 32), 'float', False)
        # Calling float(args, kwargs) (line 429)
        float_call_result_371759 = invoke(stypy.reporting.localization.Localization(__file__, 429, 32), float_371754, *[result_sub_371757], **kwargs_371758)
        
        # Getting the type of 'stride' (line 429)
        stride_371760 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 429, 54), 'stride', False)
        # Applying the binary operator 'div' (line 429)
        result_div_371761 = python_operator(stypy.reporting.localization.Localization(__file__, 429, 32), 'div', float_call_result_371759, stride_371760)
        
        # Processing the call keyword arguments (line 429)
        kwargs_371762 = {}
        # Getting the type of 'np' (line 429)
        np_371752 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 429, 24), 'np', False)
        # Obtaining the member 'ceil' of a type (line 429)
        ceil_371753 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 429, 24), np_371752, 'ceil')
        # Calling ceil(args, kwargs) (line 429)
        ceil_call_result_371763 = invoke(stypy.reporting.localization.Localization(__file__, 429, 24), ceil_371753, *[result_div_371761], **kwargs_371762)
        
        # Processing the call keyword arguments (line 429)
        kwargs_371764 = {}
        # Getting the type of 'int' (line 429)
        int_371751 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 429, 20), 'int', False)
        # Calling int(args, kwargs) (line 429)
        int_call_result_371765 = invoke(stypy.reporting.localization.Localization(__file__, 429, 20), int_371751, *[ceil_call_result_371763], **kwargs_371764)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 429, 17), tuple_371749, int_call_result_371765)
        
        # Assigning a type to the variable 'shape' (line 429)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 429, 8), 'shape', tuple_371749)
        
        # Call to csr_matrix(...): (line 430)
        # Processing the call arguments (line 430)
        
        # Obtaining an instance of the builtin type 'tuple' (line 430)
        tuple_371767 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 430, 27), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 430)
        # Adding element type (line 430)
        # Getting the type of 'row_data' (line 430)
        row_data_371768 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 430, 27), 'row_data', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 430, 27), tuple_371767, row_data_371768)
        # Adding element type (line 430)
        # Getting the type of 'row_indices' (line 430)
        row_indices_371769 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 430, 37), 'row_indices', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 430, 27), tuple_371767, row_indices_371769)
        # Adding element type (line 430)
        # Getting the type of 'row_indptr' (line 430)
        row_indptr_371770 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 430, 50), 'row_indptr', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 430, 27), tuple_371767, row_indptr_371770)
        
        # Processing the call keyword arguments (line 430)
        # Getting the type of 'shape' (line 430)
        shape_371771 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 430, 69), 'shape', False)
        keyword_371772 = shape_371771
        # Getting the type of 'self' (line 431)
        self_371773 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 431, 32), 'self', False)
        # Obtaining the member 'dtype' of a type (line 431)
        dtype_371774 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 431, 32), self_371773, 'dtype')
        keyword_371775 = dtype_371774
        # Getting the type of 'False' (line 431)
        False_371776 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 431, 49), 'False', False)
        keyword_371777 = False_371776
        kwargs_371778 = {'dtype': keyword_371775, 'shape': keyword_371772, 'copy': keyword_371777}
        # Getting the type of 'csr_matrix' (line 430)
        csr_matrix_371766 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 430, 15), 'csr_matrix', False)
        # Calling csr_matrix(args, kwargs) (line 430)
        csr_matrix_call_result_371779 = invoke(stypy.reporting.localization.Localization(__file__, 430, 15), csr_matrix_371766, *[tuple_371767], **kwargs_371778)
        
        # Assigning a type to the variable 'stypy_return_type' (line 430)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 430, 8), 'stypy_return_type', csr_matrix_call_result_371779)
        
        # ################# End of '_get_row_slice(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_get_row_slice' in the type store
        # Getting the type of 'stypy_return_type' (line 390)
        stypy_return_type_371780 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 390, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_371780)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_get_row_slice'
        return stypy_return_type_371780


    @norecursion
    def _get_submatrix(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_get_submatrix'
        module_type_store = module_type_store.open_function_context('_get_submatrix', 433, 4, False)
        # Assigning a type to the variable 'self' (line 434)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 434, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        csr_matrix._get_submatrix.__dict__.__setitem__('stypy_localization', localization)
        csr_matrix._get_submatrix.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        csr_matrix._get_submatrix.__dict__.__setitem__('stypy_type_store', module_type_store)
        csr_matrix._get_submatrix.__dict__.__setitem__('stypy_function_name', 'csr_matrix._get_submatrix')
        csr_matrix._get_submatrix.__dict__.__setitem__('stypy_param_names_list', ['row_slice', 'col_slice'])
        csr_matrix._get_submatrix.__dict__.__setitem__('stypy_varargs_param_name', None)
        csr_matrix._get_submatrix.__dict__.__setitem__('stypy_kwargs_param_name', None)
        csr_matrix._get_submatrix.__dict__.__setitem__('stypy_call_defaults', defaults)
        csr_matrix._get_submatrix.__dict__.__setitem__('stypy_call_varargs', varargs)
        csr_matrix._get_submatrix.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        csr_matrix._get_submatrix.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'csr_matrix._get_submatrix', ['row_slice', 'col_slice'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_get_submatrix', localization, ['row_slice', 'col_slice'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_get_submatrix(...)' code ##################

        str_371781 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 434, 8), 'str', 'Return a submatrix of this matrix (new matrix is created).')

        @norecursion
        def process_slice(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function 'process_slice'
            module_type_store = module_type_store.open_function_context('process_slice', 436, 8, False)
            
            # Passed parameters checking function
            process_slice.stypy_localization = localization
            process_slice.stypy_type_of_self = None
            process_slice.stypy_type_store = module_type_store
            process_slice.stypy_function_name = 'process_slice'
            process_slice.stypy_param_names_list = ['sl', 'num']
            process_slice.stypy_varargs_param_name = None
            process_slice.stypy_kwargs_param_name = None
            process_slice.stypy_call_defaults = defaults
            process_slice.stypy_call_varargs = varargs
            process_slice.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, 'process_slice', ['sl', 'num'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, 'process_slice', localization, ['sl', 'num'], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of 'process_slice(...)' code ##################

            
            # Type idiom detected: calculating its left and rigth part (line 437)
            # Getting the type of 'slice' (line 437)
            slice_371782 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 437, 30), 'slice')
            # Getting the type of 'sl' (line 437)
            sl_371783 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 437, 26), 'sl')
            
            (may_be_371784, more_types_in_union_371785) = may_be_subtype(slice_371782, sl_371783)

            if may_be_371784:

                if more_types_in_union_371785:
                    # Runtime conditional SSA (line 437)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
                else:
                    module_type_store = module_type_store

                # Assigning a type to the variable 'sl' (line 437)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 437, 12), 'sl', remove_not_subtype_from_union(sl_371783, slice))
                
                # Assigning a Call to a Tuple (line 438):
                
                # Assigning a Subscript to a Name (line 438):
                
                # Obtaining the type of the subscript
                int_371786 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 438, 16), 'int')
                
                # Call to indices(...): (line 438)
                # Processing the call arguments (line 438)
                # Getting the type of 'num' (line 438)
                num_371789 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 438, 44), 'num', False)
                # Processing the call keyword arguments (line 438)
                kwargs_371790 = {}
                # Getting the type of 'sl' (line 438)
                sl_371787 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 438, 33), 'sl', False)
                # Obtaining the member 'indices' of a type (line 438)
                indices_371788 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 438, 33), sl_371787, 'indices')
                # Calling indices(args, kwargs) (line 438)
                indices_call_result_371791 = invoke(stypy.reporting.localization.Localization(__file__, 438, 33), indices_371788, *[num_371789], **kwargs_371790)
                
                # Obtaining the member '__getitem__' of a type (line 438)
                getitem___371792 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 438, 16), indices_call_result_371791, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 438)
                subscript_call_result_371793 = invoke(stypy.reporting.localization.Localization(__file__, 438, 16), getitem___371792, int_371786)
                
                # Assigning a type to the variable 'tuple_var_assignment_370345' (line 438)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 438, 16), 'tuple_var_assignment_370345', subscript_call_result_371793)
                
                # Assigning a Subscript to a Name (line 438):
                
                # Obtaining the type of the subscript
                int_371794 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 438, 16), 'int')
                
                # Call to indices(...): (line 438)
                # Processing the call arguments (line 438)
                # Getting the type of 'num' (line 438)
                num_371797 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 438, 44), 'num', False)
                # Processing the call keyword arguments (line 438)
                kwargs_371798 = {}
                # Getting the type of 'sl' (line 438)
                sl_371795 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 438, 33), 'sl', False)
                # Obtaining the member 'indices' of a type (line 438)
                indices_371796 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 438, 33), sl_371795, 'indices')
                # Calling indices(args, kwargs) (line 438)
                indices_call_result_371799 = invoke(stypy.reporting.localization.Localization(__file__, 438, 33), indices_371796, *[num_371797], **kwargs_371798)
                
                # Obtaining the member '__getitem__' of a type (line 438)
                getitem___371800 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 438, 16), indices_call_result_371799, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 438)
                subscript_call_result_371801 = invoke(stypy.reporting.localization.Localization(__file__, 438, 16), getitem___371800, int_371794)
                
                # Assigning a type to the variable 'tuple_var_assignment_370346' (line 438)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 438, 16), 'tuple_var_assignment_370346', subscript_call_result_371801)
                
                # Assigning a Subscript to a Name (line 438):
                
                # Obtaining the type of the subscript
                int_371802 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 438, 16), 'int')
                
                # Call to indices(...): (line 438)
                # Processing the call arguments (line 438)
                # Getting the type of 'num' (line 438)
                num_371805 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 438, 44), 'num', False)
                # Processing the call keyword arguments (line 438)
                kwargs_371806 = {}
                # Getting the type of 'sl' (line 438)
                sl_371803 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 438, 33), 'sl', False)
                # Obtaining the member 'indices' of a type (line 438)
                indices_371804 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 438, 33), sl_371803, 'indices')
                # Calling indices(args, kwargs) (line 438)
                indices_call_result_371807 = invoke(stypy.reporting.localization.Localization(__file__, 438, 33), indices_371804, *[num_371805], **kwargs_371806)
                
                # Obtaining the member '__getitem__' of a type (line 438)
                getitem___371808 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 438, 16), indices_call_result_371807, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 438)
                subscript_call_result_371809 = invoke(stypy.reporting.localization.Localization(__file__, 438, 16), getitem___371808, int_371802)
                
                # Assigning a type to the variable 'tuple_var_assignment_370347' (line 438)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 438, 16), 'tuple_var_assignment_370347', subscript_call_result_371809)
                
                # Assigning a Name to a Name (line 438):
                # Getting the type of 'tuple_var_assignment_370345' (line 438)
                tuple_var_assignment_370345_371810 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 438, 16), 'tuple_var_assignment_370345')
                # Assigning a type to the variable 'i0' (line 438)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 438, 16), 'i0', tuple_var_assignment_370345_371810)
                
                # Assigning a Name to a Name (line 438):
                # Getting the type of 'tuple_var_assignment_370346' (line 438)
                tuple_var_assignment_370346_371811 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 438, 16), 'tuple_var_assignment_370346')
                # Assigning a type to the variable 'i1' (line 438)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 438, 20), 'i1', tuple_var_assignment_370346_371811)
                
                # Assigning a Name to a Name (line 438):
                # Getting the type of 'tuple_var_assignment_370347' (line 438)
                tuple_var_assignment_370347_371812 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 438, 16), 'tuple_var_assignment_370347')
                # Assigning a type to the variable 'stride' (line 438)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 438, 24), 'stride', tuple_var_assignment_370347_371812)
                
                
                # Getting the type of 'stride' (line 439)
                stride_371813 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 439, 19), 'stride')
                int_371814 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 439, 29), 'int')
                # Applying the binary operator '!=' (line 439)
                result_ne_371815 = python_operator(stypy.reporting.localization.Localization(__file__, 439, 19), '!=', stride_371813, int_371814)
                
                # Testing the type of an if condition (line 439)
                if_condition_371816 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 439, 16), result_ne_371815)
                # Assigning a type to the variable 'if_condition_371816' (line 439)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 439, 16), 'if_condition_371816', if_condition_371816)
                # SSA begins for if statement (line 439)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Call to ValueError(...): (line 440)
                # Processing the call arguments (line 440)
                str_371818 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 440, 37), 'str', 'slicing with step != 1 not supported')
                # Processing the call keyword arguments (line 440)
                kwargs_371819 = {}
                # Getting the type of 'ValueError' (line 440)
                ValueError_371817 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 440, 26), 'ValueError', False)
                # Calling ValueError(args, kwargs) (line 440)
                ValueError_call_result_371820 = invoke(stypy.reporting.localization.Localization(__file__, 440, 26), ValueError_371817, *[str_371818], **kwargs_371819)
                
                ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 440, 20), ValueError_call_result_371820, 'raise parameter', BaseException)
                # SSA join for if statement (line 439)
                module_type_store = module_type_store.join_ssa_context()
                

                if more_types_in_union_371785:
                    # Runtime conditional SSA for else branch (line 437)
                    module_type_store.open_ssa_branch('idiom else')



            if ((not may_be_371784) or more_types_in_union_371785):
                # Assigning a type to the variable 'sl' (line 437)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 437, 12), 'sl', remove_subtype_from_union(sl_371783, slice))
                
                
                # Call to isintlike(...): (line 441)
                # Processing the call arguments (line 441)
                # Getting the type of 'sl' (line 441)
                sl_371822 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 441, 27), 'sl', False)
                # Processing the call keyword arguments (line 441)
                kwargs_371823 = {}
                # Getting the type of 'isintlike' (line 441)
                isintlike_371821 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 441, 17), 'isintlike', False)
                # Calling isintlike(args, kwargs) (line 441)
                isintlike_call_result_371824 = invoke(stypy.reporting.localization.Localization(__file__, 441, 17), isintlike_371821, *[sl_371822], **kwargs_371823)
                
                # Testing the type of an if condition (line 441)
                if_condition_371825 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 441, 17), isintlike_call_result_371824)
                # Assigning a type to the variable 'if_condition_371825' (line 441)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 441, 17), 'if_condition_371825', if_condition_371825)
                # SSA begins for if statement (line 441)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                
                # Getting the type of 'sl' (line 442)
                sl_371826 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 442, 19), 'sl')
                int_371827 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 442, 24), 'int')
                # Applying the binary operator '<' (line 442)
                result_lt_371828 = python_operator(stypy.reporting.localization.Localization(__file__, 442, 19), '<', sl_371826, int_371827)
                
                # Testing the type of an if condition (line 442)
                if_condition_371829 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 442, 16), result_lt_371828)
                # Assigning a type to the variable 'if_condition_371829' (line 442)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 442, 16), 'if_condition_371829', if_condition_371829)
                # SSA begins for if statement (line 442)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Getting the type of 'sl' (line 443)
                sl_371830 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 443, 20), 'sl')
                # Getting the type of 'num' (line 443)
                num_371831 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 443, 26), 'num')
                # Applying the binary operator '+=' (line 443)
                result_iadd_371832 = python_operator(stypy.reporting.localization.Localization(__file__, 443, 20), '+=', sl_371830, num_371831)
                # Assigning a type to the variable 'sl' (line 443)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 443, 20), 'sl', result_iadd_371832)
                
                # SSA join for if statement (line 442)
                module_type_store = module_type_store.join_ssa_context()
                
                
                # Assigning a Tuple to a Tuple (line 444):
                
                # Assigning a Name to a Name (line 444):
                # Getting the type of 'sl' (line 444)
                sl_371833 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 444, 25), 'sl')
                # Assigning a type to the variable 'tuple_assignment_370348' (line 444)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 444, 16), 'tuple_assignment_370348', sl_371833)
                
                # Assigning a BinOp to a Name (line 444):
                # Getting the type of 'sl' (line 444)
                sl_371834 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 444, 29), 'sl')
                int_371835 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 444, 34), 'int')
                # Applying the binary operator '+' (line 444)
                result_add_371836 = python_operator(stypy.reporting.localization.Localization(__file__, 444, 29), '+', sl_371834, int_371835)
                
                # Assigning a type to the variable 'tuple_assignment_370349' (line 444)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 444, 16), 'tuple_assignment_370349', result_add_371836)
                
                # Assigning a Name to a Name (line 444):
                # Getting the type of 'tuple_assignment_370348' (line 444)
                tuple_assignment_370348_371837 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 444, 16), 'tuple_assignment_370348')
                # Assigning a type to the variable 'i0' (line 444)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 444, 16), 'i0', tuple_assignment_370348_371837)
                
                # Assigning a Name to a Name (line 444):
                # Getting the type of 'tuple_assignment_370349' (line 444)
                tuple_assignment_370349_371838 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 444, 16), 'tuple_assignment_370349')
                # Assigning a type to the variable 'i1' (line 444)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 444, 20), 'i1', tuple_assignment_370349_371838)
                # SSA branch for the else part of an if statement (line 441)
                module_type_store.open_ssa_branch('else')
                
                # Call to TypeError(...): (line 446)
                # Processing the call arguments (line 446)
                str_371840 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 446, 32), 'str', 'expected slice or scalar')
                # Processing the call keyword arguments (line 446)
                kwargs_371841 = {}
                # Getting the type of 'TypeError' (line 446)
                TypeError_371839 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 446, 22), 'TypeError', False)
                # Calling TypeError(args, kwargs) (line 446)
                TypeError_call_result_371842 = invoke(stypy.reporting.localization.Localization(__file__, 446, 22), TypeError_371839, *[str_371840], **kwargs_371841)
                
                ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 446, 16), TypeError_call_result_371842, 'raise parameter', BaseException)
                # SSA join for if statement (line 441)
                module_type_store = module_type_store.join_ssa_context()
                

                if (may_be_371784 and more_types_in_union_371785):
                    # SSA join for if statement (line 437)
                    module_type_store = module_type_store.join_ssa_context()


            
            
            
            # Evaluating a boolean operation
            
            
            int_371843 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 448, 20), 'int')
            # Getting the type of 'i0' (line 448)
            i0_371844 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 448, 25), 'i0')
            # Applying the binary operator '<=' (line 448)
            result_le_371845 = python_operator(stypy.reporting.localization.Localization(__file__, 448, 20), '<=', int_371843, i0_371844)
            # Getting the type of 'num' (line 448)
            num_371846 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 448, 31), 'num')
            # Applying the binary operator '<=' (line 448)
            result_le_371847 = python_operator(stypy.reporting.localization.Localization(__file__, 448, 20), '<=', i0_371844, num_371846)
            # Applying the binary operator '&' (line 448)
            result_and__371848 = python_operator(stypy.reporting.localization.Localization(__file__, 448, 20), '&', result_le_371845, result_le_371847)
            
            # Applying the 'not' unary operator (line 448)
            result_not__371849 = python_operator(stypy.reporting.localization.Localization(__file__, 448, 15), 'not', result_and__371848)
            
            
            
            int_371850 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 448, 44), 'int')
            # Getting the type of 'i1' (line 448)
            i1_371851 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 448, 49), 'i1')
            # Applying the binary operator '<=' (line 448)
            result_le_371852 = python_operator(stypy.reporting.localization.Localization(__file__, 448, 44), '<=', int_371850, i1_371851)
            # Getting the type of 'num' (line 448)
            num_371853 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 448, 55), 'num')
            # Applying the binary operator '<=' (line 448)
            result_le_371854 = python_operator(stypy.reporting.localization.Localization(__file__, 448, 44), '<=', i1_371851, num_371853)
            # Applying the binary operator '&' (line 448)
            result_and__371855 = python_operator(stypy.reporting.localization.Localization(__file__, 448, 44), '&', result_le_371852, result_le_371854)
            
            # Applying the 'not' unary operator (line 448)
            result_not__371856 = python_operator(stypy.reporting.localization.Localization(__file__, 448, 39), 'not', result_and__371855)
            
            # Applying the binary operator 'or' (line 448)
            result_or_keyword_371857 = python_operator(stypy.reporting.localization.Localization(__file__, 448, 15), 'or', result_not__371849, result_not__371856)
            
            
            # Getting the type of 'i0' (line 448)
            i0_371858 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 448, 68), 'i0')
            # Getting the type of 'i1' (line 448)
            i1_371859 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 448, 74), 'i1')
            # Applying the binary operator '<=' (line 448)
            result_le_371860 = python_operator(stypy.reporting.localization.Localization(__file__, 448, 68), '<=', i0_371858, i1_371859)
            
            # Applying the 'not' unary operator (line 448)
            result_not__371861 = python_operator(stypy.reporting.localization.Localization(__file__, 448, 63), 'not', result_le_371860)
            
            # Applying the binary operator 'or' (line 448)
            result_or_keyword_371862 = python_operator(stypy.reporting.localization.Localization(__file__, 448, 15), 'or', result_or_keyword_371857, result_not__371861)
            
            # Testing the type of an if condition (line 448)
            if_condition_371863 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 448, 12), result_or_keyword_371862)
            # Assigning a type to the variable 'if_condition_371863' (line 448)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 448, 12), 'if_condition_371863', if_condition_371863)
            # SSA begins for if statement (line 448)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to IndexError(...): (line 449)
            # Processing the call arguments (line 449)
            str_371865 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 450, 22), 'str', 'index out of bounds: 0 <= %d <= %d, 0 <= %d <= %d, %d <= %d')
            
            # Obtaining an instance of the builtin type 'tuple' (line 451)
            tuple_371866 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 451, 37), 'tuple')
            # Adding type elements to the builtin type 'tuple' instance (line 451)
            # Adding element type (line 451)
            # Getting the type of 'i0' (line 451)
            i0_371867 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 451, 37), 'i0', False)
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 451, 37), tuple_371866, i0_371867)
            # Adding element type (line 451)
            # Getting the type of 'num' (line 451)
            num_371868 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 451, 41), 'num', False)
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 451, 37), tuple_371866, num_371868)
            # Adding element type (line 451)
            # Getting the type of 'i1' (line 451)
            i1_371869 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 451, 46), 'i1', False)
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 451, 37), tuple_371866, i1_371869)
            # Adding element type (line 451)
            # Getting the type of 'num' (line 451)
            num_371870 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 451, 50), 'num', False)
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 451, 37), tuple_371866, num_371870)
            # Adding element type (line 451)
            # Getting the type of 'i0' (line 451)
            i0_371871 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 451, 55), 'i0', False)
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 451, 37), tuple_371866, i0_371871)
            # Adding element type (line 451)
            # Getting the type of 'i1' (line 451)
            i1_371872 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 451, 59), 'i1', False)
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 451, 37), tuple_371866, i1_371872)
            
            # Applying the binary operator '%' (line 450)
            result_mod_371873 = python_operator(stypy.reporting.localization.Localization(__file__, 450, 22), '%', str_371865, tuple_371866)
            
            # Processing the call keyword arguments (line 449)
            kwargs_371874 = {}
            # Getting the type of 'IndexError' (line 449)
            IndexError_371864 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 449, 22), 'IndexError', False)
            # Calling IndexError(args, kwargs) (line 449)
            IndexError_call_result_371875 = invoke(stypy.reporting.localization.Localization(__file__, 449, 22), IndexError_371864, *[result_mod_371873], **kwargs_371874)
            
            ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 449, 16), IndexError_call_result_371875, 'raise parameter', BaseException)
            # SSA join for if statement (line 448)
            module_type_store = module_type_store.join_ssa_context()
            
            
            # Obtaining an instance of the builtin type 'tuple' (line 452)
            tuple_371876 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 452, 19), 'tuple')
            # Adding type elements to the builtin type 'tuple' instance (line 452)
            # Adding element type (line 452)
            # Getting the type of 'i0' (line 452)
            i0_371877 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 452, 19), 'i0')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 452, 19), tuple_371876, i0_371877)
            # Adding element type (line 452)
            # Getting the type of 'i1' (line 452)
            i1_371878 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 452, 23), 'i1')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 452, 19), tuple_371876, i1_371878)
            
            # Assigning a type to the variable 'stypy_return_type' (line 452)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 452, 12), 'stypy_return_type', tuple_371876)
            
            # ################# End of 'process_slice(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function 'process_slice' in the type store
            # Getting the type of 'stypy_return_type' (line 436)
            stypy_return_type_371879 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 436, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_371879)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function 'process_slice'
            return stypy_return_type_371879

        # Assigning a type to the variable 'process_slice' (line 436)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 436, 8), 'process_slice', process_slice)
        
        # Assigning a Attribute to a Tuple (line 454):
        
        # Assigning a Subscript to a Name (line 454):
        
        # Obtaining the type of the subscript
        int_371880 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 454, 8), 'int')
        # Getting the type of 'self' (line 454)
        self_371881 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 454, 14), 'self')
        # Obtaining the member 'shape' of a type (line 454)
        shape_371882 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 454, 14), self_371881, 'shape')
        # Obtaining the member '__getitem__' of a type (line 454)
        getitem___371883 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 454, 8), shape_371882, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 454)
        subscript_call_result_371884 = invoke(stypy.reporting.localization.Localization(__file__, 454, 8), getitem___371883, int_371880)
        
        # Assigning a type to the variable 'tuple_var_assignment_370350' (line 454)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 454, 8), 'tuple_var_assignment_370350', subscript_call_result_371884)
        
        # Assigning a Subscript to a Name (line 454):
        
        # Obtaining the type of the subscript
        int_371885 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 454, 8), 'int')
        # Getting the type of 'self' (line 454)
        self_371886 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 454, 14), 'self')
        # Obtaining the member 'shape' of a type (line 454)
        shape_371887 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 454, 14), self_371886, 'shape')
        # Obtaining the member '__getitem__' of a type (line 454)
        getitem___371888 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 454, 8), shape_371887, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 454)
        subscript_call_result_371889 = invoke(stypy.reporting.localization.Localization(__file__, 454, 8), getitem___371888, int_371885)
        
        # Assigning a type to the variable 'tuple_var_assignment_370351' (line 454)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 454, 8), 'tuple_var_assignment_370351', subscript_call_result_371889)
        
        # Assigning a Name to a Name (line 454):
        # Getting the type of 'tuple_var_assignment_370350' (line 454)
        tuple_var_assignment_370350_371890 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 454, 8), 'tuple_var_assignment_370350')
        # Assigning a type to the variable 'M' (line 454)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 454, 8), 'M', tuple_var_assignment_370350_371890)
        
        # Assigning a Name to a Name (line 454):
        # Getting the type of 'tuple_var_assignment_370351' (line 454)
        tuple_var_assignment_370351_371891 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 454, 8), 'tuple_var_assignment_370351')
        # Assigning a type to the variable 'N' (line 454)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 454, 10), 'N', tuple_var_assignment_370351_371891)
        
        # Assigning a Call to a Tuple (line 455):
        
        # Assigning a Subscript to a Name (line 455):
        
        # Obtaining the type of the subscript
        int_371892 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 455, 8), 'int')
        
        # Call to process_slice(...): (line 455)
        # Processing the call arguments (line 455)
        # Getting the type of 'row_slice' (line 455)
        row_slice_371894 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 455, 31), 'row_slice', False)
        # Getting the type of 'M' (line 455)
        M_371895 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 455, 42), 'M', False)
        # Processing the call keyword arguments (line 455)
        kwargs_371896 = {}
        # Getting the type of 'process_slice' (line 455)
        process_slice_371893 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 455, 17), 'process_slice', False)
        # Calling process_slice(args, kwargs) (line 455)
        process_slice_call_result_371897 = invoke(stypy.reporting.localization.Localization(__file__, 455, 17), process_slice_371893, *[row_slice_371894, M_371895], **kwargs_371896)
        
        # Obtaining the member '__getitem__' of a type (line 455)
        getitem___371898 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 455, 8), process_slice_call_result_371897, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 455)
        subscript_call_result_371899 = invoke(stypy.reporting.localization.Localization(__file__, 455, 8), getitem___371898, int_371892)
        
        # Assigning a type to the variable 'tuple_var_assignment_370352' (line 455)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 455, 8), 'tuple_var_assignment_370352', subscript_call_result_371899)
        
        # Assigning a Subscript to a Name (line 455):
        
        # Obtaining the type of the subscript
        int_371900 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 455, 8), 'int')
        
        # Call to process_slice(...): (line 455)
        # Processing the call arguments (line 455)
        # Getting the type of 'row_slice' (line 455)
        row_slice_371902 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 455, 31), 'row_slice', False)
        # Getting the type of 'M' (line 455)
        M_371903 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 455, 42), 'M', False)
        # Processing the call keyword arguments (line 455)
        kwargs_371904 = {}
        # Getting the type of 'process_slice' (line 455)
        process_slice_371901 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 455, 17), 'process_slice', False)
        # Calling process_slice(args, kwargs) (line 455)
        process_slice_call_result_371905 = invoke(stypy.reporting.localization.Localization(__file__, 455, 17), process_slice_371901, *[row_slice_371902, M_371903], **kwargs_371904)
        
        # Obtaining the member '__getitem__' of a type (line 455)
        getitem___371906 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 455, 8), process_slice_call_result_371905, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 455)
        subscript_call_result_371907 = invoke(stypy.reporting.localization.Localization(__file__, 455, 8), getitem___371906, int_371900)
        
        # Assigning a type to the variable 'tuple_var_assignment_370353' (line 455)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 455, 8), 'tuple_var_assignment_370353', subscript_call_result_371907)
        
        # Assigning a Name to a Name (line 455):
        # Getting the type of 'tuple_var_assignment_370352' (line 455)
        tuple_var_assignment_370352_371908 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 455, 8), 'tuple_var_assignment_370352')
        # Assigning a type to the variable 'i0' (line 455)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 455, 8), 'i0', tuple_var_assignment_370352_371908)
        
        # Assigning a Name to a Name (line 455):
        # Getting the type of 'tuple_var_assignment_370353' (line 455)
        tuple_var_assignment_370353_371909 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 455, 8), 'tuple_var_assignment_370353')
        # Assigning a type to the variable 'i1' (line 455)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 455, 12), 'i1', tuple_var_assignment_370353_371909)
        
        # Assigning a Call to a Tuple (line 456):
        
        # Assigning a Subscript to a Name (line 456):
        
        # Obtaining the type of the subscript
        int_371910 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 456, 8), 'int')
        
        # Call to process_slice(...): (line 456)
        # Processing the call arguments (line 456)
        # Getting the type of 'col_slice' (line 456)
        col_slice_371912 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 456, 31), 'col_slice', False)
        # Getting the type of 'N' (line 456)
        N_371913 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 456, 42), 'N', False)
        # Processing the call keyword arguments (line 456)
        kwargs_371914 = {}
        # Getting the type of 'process_slice' (line 456)
        process_slice_371911 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 456, 17), 'process_slice', False)
        # Calling process_slice(args, kwargs) (line 456)
        process_slice_call_result_371915 = invoke(stypy.reporting.localization.Localization(__file__, 456, 17), process_slice_371911, *[col_slice_371912, N_371913], **kwargs_371914)
        
        # Obtaining the member '__getitem__' of a type (line 456)
        getitem___371916 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 456, 8), process_slice_call_result_371915, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 456)
        subscript_call_result_371917 = invoke(stypy.reporting.localization.Localization(__file__, 456, 8), getitem___371916, int_371910)
        
        # Assigning a type to the variable 'tuple_var_assignment_370354' (line 456)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 456, 8), 'tuple_var_assignment_370354', subscript_call_result_371917)
        
        # Assigning a Subscript to a Name (line 456):
        
        # Obtaining the type of the subscript
        int_371918 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 456, 8), 'int')
        
        # Call to process_slice(...): (line 456)
        # Processing the call arguments (line 456)
        # Getting the type of 'col_slice' (line 456)
        col_slice_371920 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 456, 31), 'col_slice', False)
        # Getting the type of 'N' (line 456)
        N_371921 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 456, 42), 'N', False)
        # Processing the call keyword arguments (line 456)
        kwargs_371922 = {}
        # Getting the type of 'process_slice' (line 456)
        process_slice_371919 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 456, 17), 'process_slice', False)
        # Calling process_slice(args, kwargs) (line 456)
        process_slice_call_result_371923 = invoke(stypy.reporting.localization.Localization(__file__, 456, 17), process_slice_371919, *[col_slice_371920, N_371921], **kwargs_371922)
        
        # Obtaining the member '__getitem__' of a type (line 456)
        getitem___371924 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 456, 8), process_slice_call_result_371923, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 456)
        subscript_call_result_371925 = invoke(stypy.reporting.localization.Localization(__file__, 456, 8), getitem___371924, int_371918)
        
        # Assigning a type to the variable 'tuple_var_assignment_370355' (line 456)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 456, 8), 'tuple_var_assignment_370355', subscript_call_result_371925)
        
        # Assigning a Name to a Name (line 456):
        # Getting the type of 'tuple_var_assignment_370354' (line 456)
        tuple_var_assignment_370354_371926 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 456, 8), 'tuple_var_assignment_370354')
        # Assigning a type to the variable 'j0' (line 456)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 456, 8), 'j0', tuple_var_assignment_370354_371926)
        
        # Assigning a Name to a Name (line 456):
        # Getting the type of 'tuple_var_assignment_370355' (line 456)
        tuple_var_assignment_370355_371927 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 456, 8), 'tuple_var_assignment_370355')
        # Assigning a type to the variable 'j1' (line 456)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 456, 12), 'j1', tuple_var_assignment_370355_371927)
        
        # Assigning a Call to a Tuple (line 458):
        
        # Assigning a Subscript to a Name (line 458):
        
        # Obtaining the type of the subscript
        int_371928 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 458, 8), 'int')
        
        # Call to get_csr_submatrix(...): (line 458)
        # Processing the call arguments (line 458)
        # Getting the type of 'M' (line 459)
        M_371930 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 459, 12), 'M', False)
        # Getting the type of 'N' (line 459)
        N_371931 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 459, 15), 'N', False)
        # Getting the type of 'self' (line 459)
        self_371932 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 459, 18), 'self', False)
        # Obtaining the member 'indptr' of a type (line 459)
        indptr_371933 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 459, 18), self_371932, 'indptr')
        # Getting the type of 'self' (line 459)
        self_371934 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 459, 31), 'self', False)
        # Obtaining the member 'indices' of a type (line 459)
        indices_371935 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 459, 31), self_371934, 'indices')
        # Getting the type of 'self' (line 459)
        self_371936 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 459, 45), 'self', False)
        # Obtaining the member 'data' of a type (line 459)
        data_371937 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 459, 45), self_371936, 'data')
        # Getting the type of 'i0' (line 459)
        i0_371938 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 459, 56), 'i0', False)
        # Getting the type of 'i1' (line 459)
        i1_371939 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 459, 60), 'i1', False)
        # Getting the type of 'j0' (line 459)
        j0_371940 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 459, 64), 'j0', False)
        # Getting the type of 'j1' (line 459)
        j1_371941 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 459, 68), 'j1', False)
        # Processing the call keyword arguments (line 458)
        kwargs_371942 = {}
        # Getting the type of 'get_csr_submatrix' (line 458)
        get_csr_submatrix_371929 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 458, 32), 'get_csr_submatrix', False)
        # Calling get_csr_submatrix(args, kwargs) (line 458)
        get_csr_submatrix_call_result_371943 = invoke(stypy.reporting.localization.Localization(__file__, 458, 32), get_csr_submatrix_371929, *[M_371930, N_371931, indptr_371933, indices_371935, data_371937, i0_371938, i1_371939, j0_371940, j1_371941], **kwargs_371942)
        
        # Obtaining the member '__getitem__' of a type (line 458)
        getitem___371944 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 458, 8), get_csr_submatrix_call_result_371943, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 458)
        subscript_call_result_371945 = invoke(stypy.reporting.localization.Localization(__file__, 458, 8), getitem___371944, int_371928)
        
        # Assigning a type to the variable 'tuple_var_assignment_370356' (line 458)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 458, 8), 'tuple_var_assignment_370356', subscript_call_result_371945)
        
        # Assigning a Subscript to a Name (line 458):
        
        # Obtaining the type of the subscript
        int_371946 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 458, 8), 'int')
        
        # Call to get_csr_submatrix(...): (line 458)
        # Processing the call arguments (line 458)
        # Getting the type of 'M' (line 459)
        M_371948 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 459, 12), 'M', False)
        # Getting the type of 'N' (line 459)
        N_371949 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 459, 15), 'N', False)
        # Getting the type of 'self' (line 459)
        self_371950 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 459, 18), 'self', False)
        # Obtaining the member 'indptr' of a type (line 459)
        indptr_371951 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 459, 18), self_371950, 'indptr')
        # Getting the type of 'self' (line 459)
        self_371952 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 459, 31), 'self', False)
        # Obtaining the member 'indices' of a type (line 459)
        indices_371953 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 459, 31), self_371952, 'indices')
        # Getting the type of 'self' (line 459)
        self_371954 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 459, 45), 'self', False)
        # Obtaining the member 'data' of a type (line 459)
        data_371955 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 459, 45), self_371954, 'data')
        # Getting the type of 'i0' (line 459)
        i0_371956 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 459, 56), 'i0', False)
        # Getting the type of 'i1' (line 459)
        i1_371957 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 459, 60), 'i1', False)
        # Getting the type of 'j0' (line 459)
        j0_371958 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 459, 64), 'j0', False)
        # Getting the type of 'j1' (line 459)
        j1_371959 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 459, 68), 'j1', False)
        # Processing the call keyword arguments (line 458)
        kwargs_371960 = {}
        # Getting the type of 'get_csr_submatrix' (line 458)
        get_csr_submatrix_371947 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 458, 32), 'get_csr_submatrix', False)
        # Calling get_csr_submatrix(args, kwargs) (line 458)
        get_csr_submatrix_call_result_371961 = invoke(stypy.reporting.localization.Localization(__file__, 458, 32), get_csr_submatrix_371947, *[M_371948, N_371949, indptr_371951, indices_371953, data_371955, i0_371956, i1_371957, j0_371958, j1_371959], **kwargs_371960)
        
        # Obtaining the member '__getitem__' of a type (line 458)
        getitem___371962 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 458, 8), get_csr_submatrix_call_result_371961, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 458)
        subscript_call_result_371963 = invoke(stypy.reporting.localization.Localization(__file__, 458, 8), getitem___371962, int_371946)
        
        # Assigning a type to the variable 'tuple_var_assignment_370357' (line 458)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 458, 8), 'tuple_var_assignment_370357', subscript_call_result_371963)
        
        # Assigning a Subscript to a Name (line 458):
        
        # Obtaining the type of the subscript
        int_371964 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 458, 8), 'int')
        
        # Call to get_csr_submatrix(...): (line 458)
        # Processing the call arguments (line 458)
        # Getting the type of 'M' (line 459)
        M_371966 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 459, 12), 'M', False)
        # Getting the type of 'N' (line 459)
        N_371967 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 459, 15), 'N', False)
        # Getting the type of 'self' (line 459)
        self_371968 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 459, 18), 'self', False)
        # Obtaining the member 'indptr' of a type (line 459)
        indptr_371969 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 459, 18), self_371968, 'indptr')
        # Getting the type of 'self' (line 459)
        self_371970 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 459, 31), 'self', False)
        # Obtaining the member 'indices' of a type (line 459)
        indices_371971 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 459, 31), self_371970, 'indices')
        # Getting the type of 'self' (line 459)
        self_371972 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 459, 45), 'self', False)
        # Obtaining the member 'data' of a type (line 459)
        data_371973 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 459, 45), self_371972, 'data')
        # Getting the type of 'i0' (line 459)
        i0_371974 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 459, 56), 'i0', False)
        # Getting the type of 'i1' (line 459)
        i1_371975 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 459, 60), 'i1', False)
        # Getting the type of 'j0' (line 459)
        j0_371976 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 459, 64), 'j0', False)
        # Getting the type of 'j1' (line 459)
        j1_371977 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 459, 68), 'j1', False)
        # Processing the call keyword arguments (line 458)
        kwargs_371978 = {}
        # Getting the type of 'get_csr_submatrix' (line 458)
        get_csr_submatrix_371965 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 458, 32), 'get_csr_submatrix', False)
        # Calling get_csr_submatrix(args, kwargs) (line 458)
        get_csr_submatrix_call_result_371979 = invoke(stypy.reporting.localization.Localization(__file__, 458, 32), get_csr_submatrix_371965, *[M_371966, N_371967, indptr_371969, indices_371971, data_371973, i0_371974, i1_371975, j0_371976, j1_371977], **kwargs_371978)
        
        # Obtaining the member '__getitem__' of a type (line 458)
        getitem___371980 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 458, 8), get_csr_submatrix_call_result_371979, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 458)
        subscript_call_result_371981 = invoke(stypy.reporting.localization.Localization(__file__, 458, 8), getitem___371980, int_371964)
        
        # Assigning a type to the variable 'tuple_var_assignment_370358' (line 458)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 458, 8), 'tuple_var_assignment_370358', subscript_call_result_371981)
        
        # Assigning a Name to a Name (line 458):
        # Getting the type of 'tuple_var_assignment_370356' (line 458)
        tuple_var_assignment_370356_371982 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 458, 8), 'tuple_var_assignment_370356')
        # Assigning a type to the variable 'indptr' (line 458)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 458, 8), 'indptr', tuple_var_assignment_370356_371982)
        
        # Assigning a Name to a Name (line 458):
        # Getting the type of 'tuple_var_assignment_370357' (line 458)
        tuple_var_assignment_370357_371983 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 458, 8), 'tuple_var_assignment_370357')
        # Assigning a type to the variable 'indices' (line 458)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 458, 16), 'indices', tuple_var_assignment_370357_371983)
        
        # Assigning a Name to a Name (line 458):
        # Getting the type of 'tuple_var_assignment_370358' (line 458)
        tuple_var_assignment_370358_371984 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 458, 8), 'tuple_var_assignment_370358')
        # Assigning a type to the variable 'data' (line 458)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 458, 25), 'data', tuple_var_assignment_370358_371984)
        
        # Assigning a Tuple to a Name (line 461):
        
        # Assigning a Tuple to a Name (line 461):
        
        # Obtaining an instance of the builtin type 'tuple' (line 461)
        tuple_371985 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 461, 17), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 461)
        # Adding element type (line 461)
        # Getting the type of 'i1' (line 461)
        i1_371986 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 461, 17), 'i1')
        # Getting the type of 'i0' (line 461)
        i0_371987 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 461, 22), 'i0')
        # Applying the binary operator '-' (line 461)
        result_sub_371988 = python_operator(stypy.reporting.localization.Localization(__file__, 461, 17), '-', i1_371986, i0_371987)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 461, 17), tuple_371985, result_sub_371988)
        # Adding element type (line 461)
        # Getting the type of 'j1' (line 461)
        j1_371989 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 461, 26), 'j1')
        # Getting the type of 'j0' (line 461)
        j0_371990 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 461, 31), 'j0')
        # Applying the binary operator '-' (line 461)
        result_sub_371991 = python_operator(stypy.reporting.localization.Localization(__file__, 461, 26), '-', j1_371989, j0_371990)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 461, 17), tuple_371985, result_sub_371991)
        
        # Assigning a type to the variable 'shape' (line 461)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 461, 8), 'shape', tuple_371985)
        
        # Call to __class__(...): (line 462)
        # Processing the call arguments (line 462)
        
        # Obtaining an instance of the builtin type 'tuple' (line 462)
        tuple_371994 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 462, 31), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 462)
        # Adding element type (line 462)
        # Getting the type of 'data' (line 462)
        data_371995 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 462, 31), 'data', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 462, 31), tuple_371994, data_371995)
        # Adding element type (line 462)
        # Getting the type of 'indices' (line 462)
        indices_371996 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 462, 37), 'indices', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 462, 31), tuple_371994, indices_371996)
        # Adding element type (line 462)
        # Getting the type of 'indptr' (line 462)
        indptr_371997 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 462, 46), 'indptr', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 462, 31), tuple_371994, indptr_371997)
        
        # Processing the call keyword arguments (line 462)
        # Getting the type of 'shape' (line 462)
        shape_371998 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 462, 61), 'shape', False)
        keyword_371999 = shape_371998
        # Getting the type of 'self' (line 463)
        self_372000 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 463, 36), 'self', False)
        # Obtaining the member 'dtype' of a type (line 463)
        dtype_372001 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 463, 36), self_372000, 'dtype')
        keyword_372002 = dtype_372001
        # Getting the type of 'False' (line 463)
        False_372003 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 463, 53), 'False', False)
        keyword_372004 = False_372003
        kwargs_372005 = {'dtype': keyword_372002, 'shape': keyword_371999, 'copy': keyword_372004}
        # Getting the type of 'self' (line 462)
        self_371992 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 462, 15), 'self', False)
        # Obtaining the member '__class__' of a type (line 462)
        class___371993 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 462, 15), self_371992, '__class__')
        # Calling __class__(args, kwargs) (line 462)
        class___call_result_372006 = invoke(stypy.reporting.localization.Localization(__file__, 462, 15), class___371993, *[tuple_371994], **kwargs_372005)
        
        # Assigning a type to the variable 'stypy_return_type' (line 462)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 462, 8), 'stypy_return_type', class___call_result_372006)
        
        # ################# End of '_get_submatrix(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_get_submatrix' in the type store
        # Getting the type of 'stypy_return_type' (line 433)
        stypy_return_type_372007 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 433, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_372007)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_get_submatrix'
        return stypy_return_type_372007


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 23, 0, False)
        # Assigning a type to the variable 'self' (line 24)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 24, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'csr_matrix.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'csr_matrix' (line 23)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 23, 0), 'csr_matrix', csr_matrix)

# Assigning a Str to a Name (line 129):
str_372008 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 129, 13), 'str', 'csr')
# Getting the type of 'csr_matrix'
csr_matrix_372009 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'csr_matrix')
# Setting the type of the member 'format' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), csr_matrix_372009, 'format', str_372008)

# Assigning a Attribute to a Attribute (line 143):
# Getting the type of 'spmatrix' (line 143)
spmatrix_372010 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 24), 'spmatrix')
# Obtaining the member 'transpose' of a type (line 143)
transpose_372011 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 143, 24), spmatrix_372010, 'transpose')
# Obtaining the member '__doc__' of a type (line 143)
doc___372012 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 143, 24), transpose_372011, '__doc__')
# Getting the type of 'csr_matrix'
csr_matrix_372013 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'csr_matrix')
# Obtaining the member 'transpose' of a type
transpose_372014 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), csr_matrix_372013, 'transpose')
# Setting the type of the member '__doc__' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), transpose_372014, '__doc__', doc___372012)

# Assigning a Attribute to a Attribute (line 161):
# Getting the type of 'spmatrix' (line 161)
spmatrix_372015 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 20), 'spmatrix')
# Obtaining the member 'tolil' of a type (line 161)
tolil_372016 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 161, 20), spmatrix_372015, 'tolil')
# Obtaining the member '__doc__' of a type (line 161)
doc___372017 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 161, 20), tolil_372016, '__doc__')
# Getting the type of 'csr_matrix'
csr_matrix_372018 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'csr_matrix')
# Obtaining the member 'tolil' of a type
tolil_372019 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), csr_matrix_372018, 'tolil')
# Setting the type of the member '__doc__' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), tolil_372019, '__doc__', doc___372017)

# Assigning a Attribute to a Attribute (line 169):
# Getting the type of 'spmatrix' (line 169)
spmatrix_372020 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 20), 'spmatrix')
# Obtaining the member 'tocsr' of a type (line 169)
tocsr_372021 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 169, 20), spmatrix_372020, 'tocsr')
# Obtaining the member '__doc__' of a type (line 169)
doc___372022 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 169, 20), tocsr_372021, '__doc__')
# Getting the type of 'csr_matrix'
csr_matrix_372023 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'csr_matrix')
# Obtaining the member 'tocsr' of a type
tocsr_372024 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), csr_matrix_372023, 'tocsr')
# Setting the type of the member '__doc__' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), tocsr_372024, '__doc__', doc___372022)

# Assigning a Attribute to a Attribute (line 191):
# Getting the type of 'spmatrix' (line 191)
spmatrix_372025 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 20), 'spmatrix')
# Obtaining the member 'tocsc' of a type (line 191)
tocsc_372026 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 191, 20), spmatrix_372025, 'tocsc')
# Obtaining the member '__doc__' of a type (line 191)
doc___372027 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 191, 20), tocsc_372026, '__doc__')
# Getting the type of 'csr_matrix'
csr_matrix_372028 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'csr_matrix')
# Obtaining the member 'tocsc' of a type
tocsc_372029 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), csr_matrix_372028, 'tocsc')
# Setting the type of the member '__doc__' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), tocsc_372029, '__doc__', doc___372027)

# Assigning a Attribute to a Attribute (line 227):
# Getting the type of 'spmatrix' (line 227)
spmatrix_372030 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 20), 'spmatrix')
# Obtaining the member 'tobsr' of a type (line 227)
tobsr_372031 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 227, 20), spmatrix_372030, 'tobsr')
# Obtaining the member '__doc__' of a type (line 227)
doc___372032 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 227, 20), tobsr_372031, '__doc__')
# Getting the type of 'csr_matrix'
csr_matrix_372033 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'csr_matrix')
# Obtaining the member 'tobsr' of a type
tobsr_372034 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), csr_matrix_372033, 'tobsr')
# Setting the type of the member '__doc__' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), tobsr_372034, '__doc__', doc___372032)

@norecursion
def isspmatrix_csr(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'isspmatrix_csr'
    module_type_store = module_type_store.open_function_context('isspmatrix_csr', 466, 0, False)
    
    # Passed parameters checking function
    isspmatrix_csr.stypy_localization = localization
    isspmatrix_csr.stypy_type_of_self = None
    isspmatrix_csr.stypy_type_store = module_type_store
    isspmatrix_csr.stypy_function_name = 'isspmatrix_csr'
    isspmatrix_csr.stypy_param_names_list = ['x']
    isspmatrix_csr.stypy_varargs_param_name = None
    isspmatrix_csr.stypy_kwargs_param_name = None
    isspmatrix_csr.stypy_call_defaults = defaults
    isspmatrix_csr.stypy_call_varargs = varargs
    isspmatrix_csr.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'isspmatrix_csr', ['x'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'isspmatrix_csr', localization, ['x'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'isspmatrix_csr(...)' code ##################

    str_372035 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 488, (-1)), 'str', 'Is x of csr_matrix type?\n\n    Parameters\n    ----------\n    x\n        object to check for being a csr matrix\n\n    Returns\n    -------\n    bool\n        True if x is a csr matrix, False otherwise\n\n    Examples\n    --------\n    >>> from scipy.sparse import csr_matrix, isspmatrix_csr\n    >>> isspmatrix_csr(csr_matrix([[5]]))\n    True\n\n    >>> from scipy.sparse import csc_matrix, csr_matrix, isspmatrix_csc\n    >>> isspmatrix_csr(csc_matrix([[5]]))\n    False\n    ')
    
    # Call to isinstance(...): (line 489)
    # Processing the call arguments (line 489)
    # Getting the type of 'x' (line 489)
    x_372037 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 489, 22), 'x', False)
    # Getting the type of 'csr_matrix' (line 489)
    csr_matrix_372038 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 489, 25), 'csr_matrix', False)
    # Processing the call keyword arguments (line 489)
    kwargs_372039 = {}
    # Getting the type of 'isinstance' (line 489)
    isinstance_372036 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 489, 11), 'isinstance', False)
    # Calling isinstance(args, kwargs) (line 489)
    isinstance_call_result_372040 = invoke(stypy.reporting.localization.Localization(__file__, 489, 11), isinstance_372036, *[x_372037, csr_matrix_372038], **kwargs_372039)
    
    # Assigning a type to the variable 'stypy_return_type' (line 489)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 489, 4), 'stypy_return_type', isinstance_call_result_372040)
    
    # ################# End of 'isspmatrix_csr(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'isspmatrix_csr' in the type store
    # Getting the type of 'stypy_return_type' (line 466)
    stypy_return_type_372041 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 466, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_372041)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'isspmatrix_csr'
    return stypy_return_type_372041

# Assigning a type to the variable 'isspmatrix_csr' (line 466)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 466, 0), 'isspmatrix_csr', isspmatrix_csr)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
