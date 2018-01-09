
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: '''LInked List sparse matrix class
2: '''
3: 
4: from __future__ import division, print_function, absolute_import
5: 
6: __docformat__ = "restructuredtext en"
7: 
8: __all__ = ['lil_matrix','isspmatrix_lil']
9: 
10: import numpy as np
11: 
12: from scipy._lib.six import xrange
13: from .base import spmatrix, isspmatrix
14: from .sputils import (getdtype, isshape, isscalarlike, IndexMixin,
15:                       upcast_scalar, get_index_dtype, isintlike)
16: from . import _csparsetools
17: 
18: 
19: class lil_matrix(spmatrix, IndexMixin):
20:     '''Row-based linked list sparse matrix
21: 
22:     This is a structure for constructing sparse matrices incrementally.
23:     Note that inserting a single item can take linear time in the worst case;
24:     to construct a matrix efficiently, make sure the items are pre-sorted by
25:     index, per row.
26: 
27:     This can be instantiated in several ways:
28:         lil_matrix(D)
29:             with a dense matrix or rank-2 ndarray D
30: 
31:         lil_matrix(S)
32:             with another sparse matrix S (equivalent to S.tolil())
33: 
34:         lil_matrix((M, N), [dtype])
35:             to construct an empty matrix with shape (M, N)
36:             dtype is optional, defaulting to dtype='d'.
37: 
38:     Attributes
39:     ----------
40:     dtype : dtype
41:         Data type of the matrix
42:     shape : 2-tuple
43:         Shape of the matrix
44:     ndim : int
45:         Number of dimensions (this is always 2)
46:     nnz
47:         Number of nonzero elements
48:     data
49:         LIL format data array of the matrix
50:     rows
51:         LIL format row index array of the matrix
52: 
53:     Notes
54:     -----
55: 
56:     Sparse matrices can be used in arithmetic operations: they support
57:     addition, subtraction, multiplication, division, and matrix power.
58: 
59:     Advantages of the LIL format
60:         - supports flexible slicing
61:         - changes to the matrix sparsity structure are efficient
62: 
63:     Disadvantages of the LIL format
64:         - arithmetic operations LIL + LIL are slow (consider CSR or CSC)
65:         - slow column slicing (consider CSC)
66:         - slow matrix vector products (consider CSR or CSC)
67: 
68:     Intended Usage
69:         - LIL is a convenient format for constructing sparse matrices
70:         - once a matrix has been constructed, convert to CSR or
71:           CSC format for fast arithmetic and matrix vector operations
72:         - consider using the COO format when constructing large matrices
73: 
74:     Data Structure
75:         - An array (``self.rows``) of rows, each of which is a sorted
76:           list of column indices of non-zero elements.
77:         - The corresponding nonzero values are stored in similar
78:           fashion in ``self.data``.
79: 
80: 
81:     '''
82:     format = 'lil'
83: 
84:     def __init__(self, arg1, shape=None, dtype=None, copy=False):
85:         spmatrix.__init__(self)
86:         self.dtype = getdtype(dtype, arg1, default=float)
87: 
88:         # First get the shape
89:         if isspmatrix(arg1):
90:             if isspmatrix_lil(arg1) and copy:
91:                 A = arg1.copy()
92:             else:
93:                 A = arg1.tolil()
94: 
95:             if dtype is not None:
96:                 A = A.astype(dtype)
97: 
98:             self.shape = A.shape
99:             self.dtype = A.dtype
100:             self.rows = A.rows
101:             self.data = A.data
102:         elif isinstance(arg1,tuple):
103:             if isshape(arg1):
104:                 if shape is not None:
105:                     raise ValueError('invalid use of shape parameter')
106:                 M, N = arg1
107:                 self.shape = (M,N)
108:                 self.rows = np.empty((M,), dtype=object)
109:                 self.data = np.empty((M,), dtype=object)
110:                 for i in range(M):
111:                     self.rows[i] = []
112:                     self.data[i] = []
113:             else:
114:                 raise TypeError('unrecognized lil_matrix constructor usage')
115:         else:
116:             # assume A is dense
117:             try:
118:                 A = np.asmatrix(arg1)
119:             except TypeError:
120:                 raise TypeError('unsupported matrix type')
121:             else:
122:                 from .csr import csr_matrix
123:                 A = csr_matrix(A, dtype=dtype).tolil()
124: 
125:                 self.shape = A.shape
126:                 self.dtype = A.dtype
127:                 self.rows = A.rows
128:                 self.data = A.data
129: 
130:     def set_shape(self,shape):
131:         shape = tuple(shape)
132: 
133:         if len(shape) != 2:
134:             raise ValueError("Only two-dimensional sparse arrays "
135:                                      "are supported.")
136:         try:
137:             shape = int(shape[0]),int(shape[1])  # floats, other weirdness
138:         except:
139:             raise TypeError('invalid shape')
140: 
141:         if not (shape[0] >= 0 and shape[1] >= 0):
142:             raise ValueError('invalid shape')
143: 
144:         if (self._shape != shape) and (self._shape is not None):
145:             try:
146:                 self = self.reshape(shape)
147:             except NotImplementedError:
148:                 raise NotImplementedError("Reshaping not implemented for %s." %
149:                                           self.__class__.__name__)
150:         self._shape = shape
151: 
152:     set_shape.__doc__ = spmatrix.set_shape.__doc__
153: 
154:     shape = property(fget=spmatrix.get_shape, fset=set_shape)
155: 
156:     def __iadd__(self,other):
157:         self[:,:] = self + other
158:         return self
159: 
160:     def __isub__(self,other):
161:         self[:,:] = self - other
162:         return self
163: 
164:     def __imul__(self,other):
165:         if isscalarlike(other):
166:             self[:,:] = self * other
167:             return self
168:         else:
169:             return NotImplemented
170: 
171:     def __itruediv__(self,other):
172:         if isscalarlike(other):
173:             self[:,:] = self / other
174:             return self
175:         else:
176:             return NotImplemented
177: 
178:     # Whenever the dimensions change, empty lists should be created for each
179:     # row
180: 
181:     def getnnz(self, axis=None):
182:         if axis is None:
183:             return sum([len(rowvals) for rowvals in self.data])
184:         if axis < 0:
185:             axis += 2
186:         if axis == 0:
187:             out = np.zeros(self.shape[1], dtype=np.intp)
188:             for row in self.rows:
189:                 out[row] += 1
190:             return out
191:         elif axis == 1:
192:             return np.array([len(rowvals) for rowvals in self.data], dtype=np.intp)
193:         else:
194:             raise ValueError('axis out of bounds')
195: 
196:     def count_nonzero(self):
197:         return sum(np.count_nonzero(rowvals) for rowvals in self.data)
198: 
199:     getnnz.__doc__ = spmatrix.getnnz.__doc__
200:     count_nonzero.__doc__ = spmatrix.count_nonzero.__doc__
201: 
202:     def __str__(self):
203:         val = ''
204:         for i, row in enumerate(self.rows):
205:             for pos, j in enumerate(row):
206:                 val += "  %s\t%s\n" % (str((i, j)), str(self.data[i][pos]))
207:         return val[:-1]
208: 
209:     def getrowview(self, i):
210:         '''Returns a view of the 'i'th row (without copying).
211:         '''
212:         new = lil_matrix((1, self.shape[1]), dtype=self.dtype)
213:         new.rows[0] = self.rows[i]
214:         new.data[0] = self.data[i]
215:         return new
216: 
217:     def getrow(self, i):
218:         '''Returns a copy of the 'i'th row.
219:         '''
220:         i = self._check_row_bounds(i)
221:         new = lil_matrix((1, self.shape[1]), dtype=self.dtype)
222:         new.rows[0] = self.rows[i][:]
223:         new.data[0] = self.data[i][:]
224:         return new
225: 
226:     def _check_row_bounds(self, i):
227:         if i < 0:
228:             i += self.shape[0]
229:         if i < 0 or i >= self.shape[0]:
230:             raise IndexError('row index out of bounds')
231:         return i
232: 
233:     def _check_col_bounds(self, j):
234:         if j < 0:
235:             j += self.shape[1]
236:         if j < 0 or j >= self.shape[1]:
237:             raise IndexError('column index out of bounds')
238:         return j
239: 
240:     def __getitem__(self, index):
241:         '''Return the element(s) index=(i, j), where j may be a slice.
242:         This always returns a copy for consistency, since slices into
243:         Python lists return copies.
244:         '''
245: 
246:         # Scalar fast path first
247:         if isinstance(index, tuple) and len(index) == 2:
248:             i, j = index
249:             # Use isinstance checks for common index types; this is
250:             # ~25-50% faster than isscalarlike. Other types are
251:             # handled below.
252:             if ((isinstance(i, int) or isinstance(i, np.integer)) and
253:                     (isinstance(j, int) or isinstance(j, np.integer))):
254:                 v = _csparsetools.lil_get1(self.shape[0], self.shape[1],
255:                                            self.rows, self.data,
256:                                            i, j)
257:                 return self.dtype.type(v)
258: 
259:         # Utilities found in IndexMixin
260:         i, j = self._unpack_index(index)
261: 
262:         # Proper check for other scalar index types
263:         i_intlike = isintlike(i)
264:         j_intlike = isintlike(j)
265: 
266:         if i_intlike and j_intlike:
267:             v = _csparsetools.lil_get1(self.shape[0], self.shape[1],
268:                                        self.rows, self.data,
269:                                        i, j)
270:             return self.dtype.type(v)
271:         elif j_intlike or isinstance(j, slice):
272:             # column slicing fast path
273:             if j_intlike:
274:                 j = self._check_col_bounds(j)
275:                 j = slice(j, j+1)
276: 
277:             if i_intlike:
278:                 i = self._check_row_bounds(i)
279:                 i = xrange(i, i+1)
280:                 i_shape = None
281:             elif isinstance(i, slice):
282:                 i = xrange(*i.indices(self.shape[0]))
283:                 i_shape = None
284:             else:
285:                 i = np.atleast_1d(i)
286:                 i_shape = i.shape
287: 
288:             if i_shape is None or len(i_shape) == 1:
289:                 return self._get_row_ranges(i, j)
290: 
291:         i, j = self._index_to_arrays(i, j)
292:         if i.size == 0:
293:             return lil_matrix(i.shape, dtype=self.dtype)
294: 
295:         new = lil_matrix(i.shape, dtype=self.dtype)
296: 
297:         i, j = _prepare_index_for_memoryview(i, j)
298:         _csparsetools.lil_fancy_get(self.shape[0], self.shape[1],
299:                                     self.rows, self.data,
300:                                     new.rows, new.data,
301:                                     i, j)
302:         return new
303: 
304:     def _get_row_ranges(self, rows, col_slice):
305:         '''
306:         Fast path for indexing in the case where column index is slice.
307: 
308:         This gains performance improvement over brute force by more
309:         efficient skipping of zeros, by accessing the elements
310:         column-wise in order.
311: 
312:         Parameters
313:         ----------
314:         rows : sequence or xrange
315:             Rows indexed. If xrange, must be within valid bounds.
316:         col_slice : slice
317:             Columns indexed
318: 
319:         '''
320:         j_start, j_stop, j_stride = col_slice.indices(self.shape[1])
321:         col_range = xrange(j_start, j_stop, j_stride)
322:         nj = len(col_range)
323:         new = lil_matrix((len(rows), nj), dtype=self.dtype)
324: 
325:         _csparsetools.lil_get_row_ranges(self.shape[0], self.shape[1],
326:                                          self.rows, self.data,
327:                                          new.rows, new.data,
328:                                          rows,
329:                                          j_start, j_stop, j_stride, nj)
330: 
331:         return new
332: 
333:     def __setitem__(self, index, x):
334:         # Scalar fast path first
335:         if isinstance(index, tuple) and len(index) == 2:
336:             i, j = index
337:             # Use isinstance checks for common index types; this is
338:             # ~25-50% faster than isscalarlike. Scalar index
339:             # assignment for other types is handled below together
340:             # with fancy indexing.
341:             if ((isinstance(i, int) or isinstance(i, np.integer)) and
342:                     (isinstance(j, int) or isinstance(j, np.integer))):
343:                 x = self.dtype.type(x)
344:                 if x.size > 1:
345:                     # Triggered if input was an ndarray
346:                     raise ValueError("Trying to assign a sequence to an item")
347:                 _csparsetools.lil_insert(self.shape[0], self.shape[1],
348:                                          self.rows, self.data, i, j, x)
349:                 return
350: 
351:         # General indexing
352:         i, j = self._unpack_index(index)
353: 
354:         # shortcut for common case of full matrix assign:
355:         if (isspmatrix(x) and isinstance(i, slice) and i == slice(None) and
356:                 isinstance(j, slice) and j == slice(None)
357:                 and x.shape == self.shape):
358:             x = lil_matrix(x, dtype=self.dtype)
359:             self.rows = x.rows
360:             self.data = x.data
361:             return
362: 
363:         i, j = self._index_to_arrays(i, j)
364: 
365:         if isspmatrix(x):
366:             x = x.toarray()
367: 
368:         # Make x and i into the same shape
369:         x = np.asarray(x, dtype=self.dtype)
370:         x, _ = np.broadcast_arrays(x, i)
371: 
372:         if x.shape != i.shape:
373:             raise ValueError("shape mismatch in assignment")
374: 
375:         # Set values
376:         i, j, x = _prepare_index_for_memoryview(i, j, x)
377:         _csparsetools.lil_fancy_set(self.shape[0], self.shape[1],
378:                                     self.rows, self.data,
379:                                     i, j, x)
380: 
381:     def _mul_scalar(self, other):
382:         if other == 0:
383:             # Multiply by zero: return the zero matrix
384:             new = lil_matrix(self.shape, dtype=self.dtype)
385:         else:
386:             res_dtype = upcast_scalar(self.dtype, other)
387: 
388:             new = self.copy()
389:             new = new.astype(res_dtype)
390:             # Multiply this scalar by every element.
391:             for j, rowvals in enumerate(new.data):
392:                 new.data[j] = [val*other for val in rowvals]
393:         return new
394: 
395:     def __truediv__(self, other):           # self / other
396:         if isscalarlike(other):
397:             new = self.copy()
398:             # Divide every element by this scalar
399:             for j, rowvals in enumerate(new.data):
400:                 new.data[j] = [val/other for val in rowvals]
401:             return new
402:         else:
403:             return self.tocsr() / other
404: 
405:     def copy(self):
406:         from copy import deepcopy
407:         new = lil_matrix(self.shape, dtype=self.dtype)
408:         new.data = deepcopy(self.data)
409:         new.rows = deepcopy(self.rows)
410:         return new
411: 
412:     copy.__doc__ = spmatrix.copy.__doc__
413: 
414:     def reshape(self, shape, order='C'):
415:         if type(order) != str or order != 'C':
416:             raise ValueError(("Sparse matrices do not support "
417:                               "an 'order' parameter."))
418: 
419:         if type(shape) != tuple:
420:             raise TypeError("a tuple must be passed in for 'shape'")
421: 
422:         if len(shape) != 2:
423:             raise ValueError("a length-2 tuple must be passed in for 'shape'")
424: 
425:         new = lil_matrix(shape, dtype=self.dtype)
426:         j_max = self.shape[1]
427: 
428:         # Size is ambiguous for sparse matrices, so in order to check 'total
429:         # dimension', we need to take the product of their dimensions instead
430:         if new.shape[0] * new.shape[1] != self.shape[0] * self.shape[1]:
431:             raise ValueError("the product of the dimensions for the new sparse "
432:                              "matrix must equal that of the original matrix")
433: 
434:         for i, row in enumerate(self.rows):
435:             for col, j in enumerate(row):
436:                 new_r, new_c = np.unravel_index(i*j_max + j, shape)
437:                 new[new_r, new_c] = self[i, j]
438:         return new
439: 
440:     reshape.__doc__ = spmatrix.reshape.__doc__
441: 
442:     def toarray(self, order=None, out=None):
443:         d = self._process_toarray_args(order, out)
444:         for i, row in enumerate(self.rows):
445:             for pos, j in enumerate(row):
446:                 d[i, j] = self.data[i][pos]
447:         return d
448: 
449:     toarray.__doc__ = spmatrix.toarray.__doc__
450: 
451:     def transpose(self, axes=None, copy=False):
452:         return self.tocsr().transpose(axes=axes, copy=copy).tolil()
453: 
454:     transpose.__doc__ = spmatrix.transpose.__doc__
455: 
456:     def tolil(self, copy=False):
457:         if copy:
458:             return self.copy()
459:         else:
460:             return self
461: 
462:     tolil.__doc__ = spmatrix.tolil.__doc__
463: 
464:     def tocsr(self, copy=False):
465:         lst = [len(x) for x in self.rows]
466:         idx_dtype = get_index_dtype(maxval=max(self.shape[1], sum(lst)))
467:         indptr = np.asarray(lst, dtype=idx_dtype)
468:         indptr = np.concatenate((np.array([0], dtype=idx_dtype),
469:                                  np.cumsum(indptr, dtype=idx_dtype)))
470: 
471:         indices = []
472:         for x in self.rows:
473:             indices.extend(x)
474:         indices = np.asarray(indices, dtype=idx_dtype)
475: 
476:         data = []
477:         for x in self.data:
478:             data.extend(x)
479:         data = np.asarray(data, dtype=self.dtype)
480: 
481:         from .csr import csr_matrix
482:         return csr_matrix((data, indices, indptr), shape=self.shape)
483: 
484:     tocsr.__doc__ = spmatrix.tocsr.__doc__
485: 
486: 
487: def _prepare_index_for_memoryview(i, j, x=None):
488:     '''
489:     Convert index and data arrays to form suitable for passing to the
490:     Cython fancy getset routines.
491: 
492:     The conversions are necessary since to (i) ensure the integer
493:     index arrays are in one of the accepted types, and (ii) to ensure
494:     the arrays are writable so that Cython memoryview support doesn't
495:     choke on them.
496: 
497:     Parameters
498:     ----------
499:     i, j
500:         Index arrays
501:     x : optional
502:         Data arrays
503: 
504:     Returns
505:     -------
506:     i, j, x
507:         Re-formatted arrays (x is omitted, if input was None)
508: 
509:     '''
510:     if i.dtype > j.dtype:
511:         j = j.astype(i.dtype)
512:     elif i.dtype < j.dtype:
513:         i = i.astype(j.dtype)
514: 
515:     if not i.flags.writeable or i.dtype not in (np.int32, np.int64):
516:         i = i.astype(np.intp)
517:     if not j.flags.writeable or j.dtype not in (np.int32, np.int64):
518:         j = j.astype(np.intp)
519: 
520:     if x is not None:
521:         if not x.flags.writeable:
522:             x = x.copy()
523:         return i, j, x
524:     else:
525:         return i, j
526: 
527: 
528: def isspmatrix_lil(x):
529:     '''Is x of lil_matrix type?
530: 
531:     Parameters
532:     ----------
533:     x
534:         object to check for being a lil matrix
535: 
536:     Returns
537:     -------
538:     bool
539:         True if x is a lil matrix, False otherwise
540: 
541:     Examples
542:     --------
543:     >>> from scipy.sparse import lil_matrix, isspmatrix_lil
544:     >>> isspmatrix_lil(lil_matrix([[5]]))
545:     True
546: 
547:     >>> from scipy.sparse import lil_matrix, csr_matrix, isspmatrix_lil
548:     >>> isspmatrix_lil(csr_matrix([[5]]))
549:     False
550:     '''
551:     return isinstance(x, lil_matrix)
552: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_377476 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2, (-1)), 'str', 'LInked List sparse matrix class\n')

# Assigning a Str to a Name (line 6):

# Assigning a Str to a Name (line 6):
str_377477 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6, 16), 'str', 'restructuredtext en')
# Assigning a type to the variable '__docformat__' (line 6)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), '__docformat__', str_377477)

# Assigning a List to a Name (line 8):

# Assigning a List to a Name (line 8):
__all__ = ['lil_matrix', 'isspmatrix_lil']
module_type_store.set_exportable_members(['lil_matrix', 'isspmatrix_lil'])

# Obtaining an instance of the builtin type 'list' (line 8)
list_377478 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 8, 10), 'list')
# Adding type elements to the builtin type 'list' instance (line 8)
# Adding element type (line 8)
str_377479 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 8, 11), 'str', 'lil_matrix')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 8, 10), list_377478, str_377479)
# Adding element type (line 8)
str_377480 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 8, 24), 'str', 'isspmatrix_lil')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 8, 10), list_377478, str_377480)

# Assigning a type to the variable '__all__' (line 8)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 0), '__all__', list_377478)
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 10, 0))

# 'import numpy' statement (line 10)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/sparse/')
import_377481 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'numpy')

if (type(import_377481) is not StypyTypeError):

    if (import_377481 != 'pyd_module'):
        __import__(import_377481)
        sys_modules_377482 = sys.modules[import_377481]
        import_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'np', sys_modules_377482.module_type_store, module_type_store)
    else:
        import numpy as np

        import_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'np', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 10)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 0), 'numpy', import_377481)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/sparse/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 12, 0))

# 'from scipy._lib.six import xrange' statement (line 12)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/sparse/')
import_377483 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 12, 0), 'scipy._lib.six')

if (type(import_377483) is not StypyTypeError):

    if (import_377483 != 'pyd_module'):
        __import__(import_377483)
        sys_modules_377484 = sys.modules[import_377483]
        import_from_module(stypy.reporting.localization.Localization(__file__, 12, 0), 'scipy._lib.six', sys_modules_377484.module_type_store, module_type_store, ['xrange'])
        nest_module(stypy.reporting.localization.Localization(__file__, 12, 0), __file__, sys_modules_377484, sys_modules_377484.module_type_store, module_type_store)
    else:
        from scipy._lib.six import xrange

        import_from_module(stypy.reporting.localization.Localization(__file__, 12, 0), 'scipy._lib.six', None, module_type_store, ['xrange'], [xrange])

else:
    # Assigning a type to the variable 'scipy._lib.six' (line 12)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 0), 'scipy._lib.six', import_377483)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/sparse/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 13, 0))

# 'from scipy.sparse.base import spmatrix, isspmatrix' statement (line 13)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/sparse/')
import_377485 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 13, 0), 'scipy.sparse.base')

if (type(import_377485) is not StypyTypeError):

    if (import_377485 != 'pyd_module'):
        __import__(import_377485)
        sys_modules_377486 = sys.modules[import_377485]
        import_from_module(stypy.reporting.localization.Localization(__file__, 13, 0), 'scipy.sparse.base', sys_modules_377486.module_type_store, module_type_store, ['spmatrix', 'isspmatrix'])
        nest_module(stypy.reporting.localization.Localization(__file__, 13, 0), __file__, sys_modules_377486, sys_modules_377486.module_type_store, module_type_store)
    else:
        from scipy.sparse.base import spmatrix, isspmatrix

        import_from_module(stypy.reporting.localization.Localization(__file__, 13, 0), 'scipy.sparse.base', None, module_type_store, ['spmatrix', 'isspmatrix'], [spmatrix, isspmatrix])

else:
    # Assigning a type to the variable 'scipy.sparse.base' (line 13)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 0), 'scipy.sparse.base', import_377485)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/sparse/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 14, 0))

# 'from scipy.sparse.sputils import getdtype, isshape, isscalarlike, IndexMixin, upcast_scalar, get_index_dtype, isintlike' statement (line 14)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/sparse/')
import_377487 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 14, 0), 'scipy.sparse.sputils')

if (type(import_377487) is not StypyTypeError):

    if (import_377487 != 'pyd_module'):
        __import__(import_377487)
        sys_modules_377488 = sys.modules[import_377487]
        import_from_module(stypy.reporting.localization.Localization(__file__, 14, 0), 'scipy.sparse.sputils', sys_modules_377488.module_type_store, module_type_store, ['getdtype', 'isshape', 'isscalarlike', 'IndexMixin', 'upcast_scalar', 'get_index_dtype', 'isintlike'])
        nest_module(stypy.reporting.localization.Localization(__file__, 14, 0), __file__, sys_modules_377488, sys_modules_377488.module_type_store, module_type_store)
    else:
        from scipy.sparse.sputils import getdtype, isshape, isscalarlike, IndexMixin, upcast_scalar, get_index_dtype, isintlike

        import_from_module(stypy.reporting.localization.Localization(__file__, 14, 0), 'scipy.sparse.sputils', None, module_type_store, ['getdtype', 'isshape', 'isscalarlike', 'IndexMixin', 'upcast_scalar', 'get_index_dtype', 'isintlike'], [getdtype, isshape, isscalarlike, IndexMixin, upcast_scalar, get_index_dtype, isintlike])

else:
    # Assigning a type to the variable 'scipy.sparse.sputils' (line 14)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 0), 'scipy.sparse.sputils', import_377487)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/sparse/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 16, 0))

# 'from scipy.sparse import _csparsetools' statement (line 16)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/sparse/')
import_377489 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 16, 0), 'scipy.sparse')

if (type(import_377489) is not StypyTypeError):

    if (import_377489 != 'pyd_module'):
        __import__(import_377489)
        sys_modules_377490 = sys.modules[import_377489]
        import_from_module(stypy.reporting.localization.Localization(__file__, 16, 0), 'scipy.sparse', sys_modules_377490.module_type_store, module_type_store, ['_csparsetools'])
        nest_module(stypy.reporting.localization.Localization(__file__, 16, 0), __file__, sys_modules_377490, sys_modules_377490.module_type_store, module_type_store)
    else:
        from scipy.sparse import _csparsetools

        import_from_module(stypy.reporting.localization.Localization(__file__, 16, 0), 'scipy.sparse', None, module_type_store, ['_csparsetools'], [_csparsetools])

else:
    # Assigning a type to the variable 'scipy.sparse' (line 16)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 0), 'scipy.sparse', import_377489)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/sparse/')

# Declaration of the 'lil_matrix' class
# Getting the type of 'spmatrix' (line 19)
spmatrix_377491 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 17), 'spmatrix')
# Getting the type of 'IndexMixin' (line 19)
IndexMixin_377492 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 27), 'IndexMixin')

class lil_matrix(spmatrix_377491, IndexMixin_377492, ):
    str_377493 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 81, (-1)), 'str', "Row-based linked list sparse matrix\n\n    This is a structure for constructing sparse matrices incrementally.\n    Note that inserting a single item can take linear time in the worst case;\n    to construct a matrix efficiently, make sure the items are pre-sorted by\n    index, per row.\n\n    This can be instantiated in several ways:\n        lil_matrix(D)\n            with a dense matrix or rank-2 ndarray D\n\n        lil_matrix(S)\n            with another sparse matrix S (equivalent to S.tolil())\n\n        lil_matrix((M, N), [dtype])\n            to construct an empty matrix with shape (M, N)\n            dtype is optional, defaulting to dtype='d'.\n\n    Attributes\n    ----------\n    dtype : dtype\n        Data type of the matrix\n    shape : 2-tuple\n        Shape of the matrix\n    ndim : int\n        Number of dimensions (this is always 2)\n    nnz\n        Number of nonzero elements\n    data\n        LIL format data array of the matrix\n    rows\n        LIL format row index array of the matrix\n\n    Notes\n    -----\n\n    Sparse matrices can be used in arithmetic operations: they support\n    addition, subtraction, multiplication, division, and matrix power.\n\n    Advantages of the LIL format\n        - supports flexible slicing\n        - changes to the matrix sparsity structure are efficient\n\n    Disadvantages of the LIL format\n        - arithmetic operations LIL + LIL are slow (consider CSR or CSC)\n        - slow column slicing (consider CSC)\n        - slow matrix vector products (consider CSR or CSC)\n\n    Intended Usage\n        - LIL is a convenient format for constructing sparse matrices\n        - once a matrix has been constructed, convert to CSR or\n          CSC format for fast arithmetic and matrix vector operations\n        - consider using the COO format when constructing large matrices\n\n    Data Structure\n        - An array (``self.rows``) of rows, each of which is a sorted\n          list of column indices of non-zero elements.\n        - The corresponding nonzero values are stored in similar\n          fashion in ``self.data``.\n\n\n    ")
    
    # Assigning a Str to a Name (line 82):

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 84)
        None_377494 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 35), 'None')
        # Getting the type of 'None' (line 84)
        None_377495 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 47), 'None')
        # Getting the type of 'False' (line 84)
        False_377496 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 58), 'False')
        defaults = [None_377494, None_377495, False_377496]
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 84, 4, False)
        # Assigning a type to the variable 'self' (line 85)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 85, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'lil_matrix.__init__', ['arg1', 'shape', 'dtype', 'copy'], None, None, defaults, varargs, kwargs)

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

        
        # Call to __init__(...): (line 85)
        # Processing the call arguments (line 85)
        # Getting the type of 'self' (line 85)
        self_377499 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 26), 'self', False)
        # Processing the call keyword arguments (line 85)
        kwargs_377500 = {}
        # Getting the type of 'spmatrix' (line 85)
        spmatrix_377497 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 8), 'spmatrix', False)
        # Obtaining the member '__init__' of a type (line 85)
        init___377498 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 85, 8), spmatrix_377497, '__init__')
        # Calling __init__(args, kwargs) (line 85)
        init___call_result_377501 = invoke(stypy.reporting.localization.Localization(__file__, 85, 8), init___377498, *[self_377499], **kwargs_377500)
        
        
        # Assigning a Call to a Attribute (line 86):
        
        # Assigning a Call to a Attribute (line 86):
        
        # Call to getdtype(...): (line 86)
        # Processing the call arguments (line 86)
        # Getting the type of 'dtype' (line 86)
        dtype_377503 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 30), 'dtype', False)
        # Getting the type of 'arg1' (line 86)
        arg1_377504 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 37), 'arg1', False)
        # Processing the call keyword arguments (line 86)
        # Getting the type of 'float' (line 86)
        float_377505 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 51), 'float', False)
        keyword_377506 = float_377505
        kwargs_377507 = {'default': keyword_377506}
        # Getting the type of 'getdtype' (line 86)
        getdtype_377502 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 21), 'getdtype', False)
        # Calling getdtype(args, kwargs) (line 86)
        getdtype_call_result_377508 = invoke(stypy.reporting.localization.Localization(__file__, 86, 21), getdtype_377502, *[dtype_377503, arg1_377504], **kwargs_377507)
        
        # Getting the type of 'self' (line 86)
        self_377509 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 8), 'self')
        # Setting the type of the member 'dtype' of a type (line 86)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 86, 8), self_377509, 'dtype', getdtype_call_result_377508)
        
        
        # Call to isspmatrix(...): (line 89)
        # Processing the call arguments (line 89)
        # Getting the type of 'arg1' (line 89)
        arg1_377511 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 22), 'arg1', False)
        # Processing the call keyword arguments (line 89)
        kwargs_377512 = {}
        # Getting the type of 'isspmatrix' (line 89)
        isspmatrix_377510 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 11), 'isspmatrix', False)
        # Calling isspmatrix(args, kwargs) (line 89)
        isspmatrix_call_result_377513 = invoke(stypy.reporting.localization.Localization(__file__, 89, 11), isspmatrix_377510, *[arg1_377511], **kwargs_377512)
        
        # Testing the type of an if condition (line 89)
        if_condition_377514 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 89, 8), isspmatrix_call_result_377513)
        # Assigning a type to the variable 'if_condition_377514' (line 89)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 89, 8), 'if_condition_377514', if_condition_377514)
        # SSA begins for if statement (line 89)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        
        # Evaluating a boolean operation
        
        # Call to isspmatrix_lil(...): (line 90)
        # Processing the call arguments (line 90)
        # Getting the type of 'arg1' (line 90)
        arg1_377516 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 30), 'arg1', False)
        # Processing the call keyword arguments (line 90)
        kwargs_377517 = {}
        # Getting the type of 'isspmatrix_lil' (line 90)
        isspmatrix_lil_377515 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 15), 'isspmatrix_lil', False)
        # Calling isspmatrix_lil(args, kwargs) (line 90)
        isspmatrix_lil_call_result_377518 = invoke(stypy.reporting.localization.Localization(__file__, 90, 15), isspmatrix_lil_377515, *[arg1_377516], **kwargs_377517)
        
        # Getting the type of 'copy' (line 90)
        copy_377519 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 40), 'copy')
        # Applying the binary operator 'and' (line 90)
        result_and_keyword_377520 = python_operator(stypy.reporting.localization.Localization(__file__, 90, 15), 'and', isspmatrix_lil_call_result_377518, copy_377519)
        
        # Testing the type of an if condition (line 90)
        if_condition_377521 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 90, 12), result_and_keyword_377520)
        # Assigning a type to the variable 'if_condition_377521' (line 90)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 90, 12), 'if_condition_377521', if_condition_377521)
        # SSA begins for if statement (line 90)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 91):
        
        # Assigning a Call to a Name (line 91):
        
        # Call to copy(...): (line 91)
        # Processing the call keyword arguments (line 91)
        kwargs_377524 = {}
        # Getting the type of 'arg1' (line 91)
        arg1_377522 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 20), 'arg1', False)
        # Obtaining the member 'copy' of a type (line 91)
        copy_377523 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 91, 20), arg1_377522, 'copy')
        # Calling copy(args, kwargs) (line 91)
        copy_call_result_377525 = invoke(stypy.reporting.localization.Localization(__file__, 91, 20), copy_377523, *[], **kwargs_377524)
        
        # Assigning a type to the variable 'A' (line 91)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 91, 16), 'A', copy_call_result_377525)
        # SSA branch for the else part of an if statement (line 90)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Call to a Name (line 93):
        
        # Assigning a Call to a Name (line 93):
        
        # Call to tolil(...): (line 93)
        # Processing the call keyword arguments (line 93)
        kwargs_377528 = {}
        # Getting the type of 'arg1' (line 93)
        arg1_377526 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 20), 'arg1', False)
        # Obtaining the member 'tolil' of a type (line 93)
        tolil_377527 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 93, 20), arg1_377526, 'tolil')
        # Calling tolil(args, kwargs) (line 93)
        tolil_call_result_377529 = invoke(stypy.reporting.localization.Localization(__file__, 93, 20), tolil_377527, *[], **kwargs_377528)
        
        # Assigning a type to the variable 'A' (line 93)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 93, 16), 'A', tolil_call_result_377529)
        # SSA join for if statement (line 90)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Type idiom detected: calculating its left and rigth part (line 95)
        # Getting the type of 'dtype' (line 95)
        dtype_377530 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 12), 'dtype')
        # Getting the type of 'None' (line 95)
        None_377531 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 28), 'None')
        
        (may_be_377532, more_types_in_union_377533) = may_not_be_none(dtype_377530, None_377531)

        if may_be_377532:

            if more_types_in_union_377533:
                # Runtime conditional SSA (line 95)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a Call to a Name (line 96):
            
            # Assigning a Call to a Name (line 96):
            
            # Call to astype(...): (line 96)
            # Processing the call arguments (line 96)
            # Getting the type of 'dtype' (line 96)
            dtype_377536 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 29), 'dtype', False)
            # Processing the call keyword arguments (line 96)
            kwargs_377537 = {}
            # Getting the type of 'A' (line 96)
            A_377534 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 20), 'A', False)
            # Obtaining the member 'astype' of a type (line 96)
            astype_377535 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 96, 20), A_377534, 'astype')
            # Calling astype(args, kwargs) (line 96)
            astype_call_result_377538 = invoke(stypy.reporting.localization.Localization(__file__, 96, 20), astype_377535, *[dtype_377536], **kwargs_377537)
            
            # Assigning a type to the variable 'A' (line 96)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 96, 16), 'A', astype_call_result_377538)

            if more_types_in_union_377533:
                # SSA join for if statement (line 95)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Assigning a Attribute to a Attribute (line 98):
        
        # Assigning a Attribute to a Attribute (line 98):
        # Getting the type of 'A' (line 98)
        A_377539 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 25), 'A')
        # Obtaining the member 'shape' of a type (line 98)
        shape_377540 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 98, 25), A_377539, 'shape')
        # Getting the type of 'self' (line 98)
        self_377541 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 12), 'self')
        # Setting the type of the member 'shape' of a type (line 98)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 98, 12), self_377541, 'shape', shape_377540)
        
        # Assigning a Attribute to a Attribute (line 99):
        
        # Assigning a Attribute to a Attribute (line 99):
        # Getting the type of 'A' (line 99)
        A_377542 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 25), 'A')
        # Obtaining the member 'dtype' of a type (line 99)
        dtype_377543 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 99, 25), A_377542, 'dtype')
        # Getting the type of 'self' (line 99)
        self_377544 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 12), 'self')
        # Setting the type of the member 'dtype' of a type (line 99)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 99, 12), self_377544, 'dtype', dtype_377543)
        
        # Assigning a Attribute to a Attribute (line 100):
        
        # Assigning a Attribute to a Attribute (line 100):
        # Getting the type of 'A' (line 100)
        A_377545 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 24), 'A')
        # Obtaining the member 'rows' of a type (line 100)
        rows_377546 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 100, 24), A_377545, 'rows')
        # Getting the type of 'self' (line 100)
        self_377547 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 12), 'self')
        # Setting the type of the member 'rows' of a type (line 100)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 100, 12), self_377547, 'rows', rows_377546)
        
        # Assigning a Attribute to a Attribute (line 101):
        
        # Assigning a Attribute to a Attribute (line 101):
        # Getting the type of 'A' (line 101)
        A_377548 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 24), 'A')
        # Obtaining the member 'data' of a type (line 101)
        data_377549 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 101, 24), A_377548, 'data')
        # Getting the type of 'self' (line 101)
        self_377550 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 12), 'self')
        # Setting the type of the member 'data' of a type (line 101)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 101, 12), self_377550, 'data', data_377549)
        # SSA branch for the else part of an if statement (line 89)
        module_type_store.open_ssa_branch('else')
        
        # Type idiom detected: calculating its left and rigth part (line 102)
        # Getting the type of 'tuple' (line 102)
        tuple_377551 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 29), 'tuple')
        # Getting the type of 'arg1' (line 102)
        arg1_377552 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 24), 'arg1')
        
        (may_be_377553, more_types_in_union_377554) = may_be_subtype(tuple_377551, arg1_377552)

        if may_be_377553:

            if more_types_in_union_377554:
                # Runtime conditional SSA (line 102)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            # Assigning a type to the variable 'arg1' (line 102)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 102, 13), 'arg1', remove_not_subtype_from_union(arg1_377552, tuple))
            
            
            # Call to isshape(...): (line 103)
            # Processing the call arguments (line 103)
            # Getting the type of 'arg1' (line 103)
            arg1_377556 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 23), 'arg1', False)
            # Processing the call keyword arguments (line 103)
            kwargs_377557 = {}
            # Getting the type of 'isshape' (line 103)
            isshape_377555 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 15), 'isshape', False)
            # Calling isshape(args, kwargs) (line 103)
            isshape_call_result_377558 = invoke(stypy.reporting.localization.Localization(__file__, 103, 15), isshape_377555, *[arg1_377556], **kwargs_377557)
            
            # Testing the type of an if condition (line 103)
            if_condition_377559 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 103, 12), isshape_call_result_377558)
            # Assigning a type to the variable 'if_condition_377559' (line 103)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 103, 12), 'if_condition_377559', if_condition_377559)
            # SSA begins for if statement (line 103)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Type idiom detected: calculating its left and rigth part (line 104)
            # Getting the type of 'shape' (line 104)
            shape_377560 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 16), 'shape')
            # Getting the type of 'None' (line 104)
            None_377561 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 32), 'None')
            
            (may_be_377562, more_types_in_union_377563) = may_not_be_none(shape_377560, None_377561)

            if may_be_377562:

                if more_types_in_union_377563:
                    # Runtime conditional SSA (line 104)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
                else:
                    module_type_store = module_type_store

                
                # Call to ValueError(...): (line 105)
                # Processing the call arguments (line 105)
                str_377565 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 105, 37), 'str', 'invalid use of shape parameter')
                # Processing the call keyword arguments (line 105)
                kwargs_377566 = {}
                # Getting the type of 'ValueError' (line 105)
                ValueError_377564 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 26), 'ValueError', False)
                # Calling ValueError(args, kwargs) (line 105)
                ValueError_call_result_377567 = invoke(stypy.reporting.localization.Localization(__file__, 105, 26), ValueError_377564, *[str_377565], **kwargs_377566)
                
                ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 105, 20), ValueError_call_result_377567, 'raise parameter', BaseException)

                if more_types_in_union_377563:
                    # SSA join for if statement (line 104)
                    module_type_store = module_type_store.join_ssa_context()


            
            
            # Assigning a Name to a Tuple (line 106):
            
            # Assigning a Subscript to a Name (line 106):
            
            # Obtaining the type of the subscript
            int_377568 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 106, 16), 'int')
            # Getting the type of 'arg1' (line 106)
            arg1_377569 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 23), 'arg1')
            # Obtaining the member '__getitem__' of a type (line 106)
            getitem___377570 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 106, 16), arg1_377569, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 106)
            subscript_call_result_377571 = invoke(stypy.reporting.localization.Localization(__file__, 106, 16), getitem___377570, int_377568)
            
            # Assigning a type to the variable 'tuple_var_assignment_377450' (line 106)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 106, 16), 'tuple_var_assignment_377450', subscript_call_result_377571)
            
            # Assigning a Subscript to a Name (line 106):
            
            # Obtaining the type of the subscript
            int_377572 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 106, 16), 'int')
            # Getting the type of 'arg1' (line 106)
            arg1_377573 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 23), 'arg1')
            # Obtaining the member '__getitem__' of a type (line 106)
            getitem___377574 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 106, 16), arg1_377573, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 106)
            subscript_call_result_377575 = invoke(stypy.reporting.localization.Localization(__file__, 106, 16), getitem___377574, int_377572)
            
            # Assigning a type to the variable 'tuple_var_assignment_377451' (line 106)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 106, 16), 'tuple_var_assignment_377451', subscript_call_result_377575)
            
            # Assigning a Name to a Name (line 106):
            # Getting the type of 'tuple_var_assignment_377450' (line 106)
            tuple_var_assignment_377450_377576 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 16), 'tuple_var_assignment_377450')
            # Assigning a type to the variable 'M' (line 106)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 106, 16), 'M', tuple_var_assignment_377450_377576)
            
            # Assigning a Name to a Name (line 106):
            # Getting the type of 'tuple_var_assignment_377451' (line 106)
            tuple_var_assignment_377451_377577 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 16), 'tuple_var_assignment_377451')
            # Assigning a type to the variable 'N' (line 106)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 106, 19), 'N', tuple_var_assignment_377451_377577)
            
            # Assigning a Tuple to a Attribute (line 107):
            
            # Assigning a Tuple to a Attribute (line 107):
            
            # Obtaining an instance of the builtin type 'tuple' (line 107)
            tuple_377578 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 107, 30), 'tuple')
            # Adding type elements to the builtin type 'tuple' instance (line 107)
            # Adding element type (line 107)
            # Getting the type of 'M' (line 107)
            M_377579 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 30), 'M')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 107, 30), tuple_377578, M_377579)
            # Adding element type (line 107)
            # Getting the type of 'N' (line 107)
            N_377580 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 32), 'N')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 107, 30), tuple_377578, N_377580)
            
            # Getting the type of 'self' (line 107)
            self_377581 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 16), 'self')
            # Setting the type of the member 'shape' of a type (line 107)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 107, 16), self_377581, 'shape', tuple_377578)
            
            # Assigning a Call to a Attribute (line 108):
            
            # Assigning a Call to a Attribute (line 108):
            
            # Call to empty(...): (line 108)
            # Processing the call arguments (line 108)
            
            # Obtaining an instance of the builtin type 'tuple' (line 108)
            tuple_377584 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 108, 38), 'tuple')
            # Adding type elements to the builtin type 'tuple' instance (line 108)
            # Adding element type (line 108)
            # Getting the type of 'M' (line 108)
            M_377585 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 38), 'M', False)
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 108, 38), tuple_377584, M_377585)
            
            # Processing the call keyword arguments (line 108)
            # Getting the type of 'object' (line 108)
            object_377586 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 49), 'object', False)
            keyword_377587 = object_377586
            kwargs_377588 = {'dtype': keyword_377587}
            # Getting the type of 'np' (line 108)
            np_377582 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 28), 'np', False)
            # Obtaining the member 'empty' of a type (line 108)
            empty_377583 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 108, 28), np_377582, 'empty')
            # Calling empty(args, kwargs) (line 108)
            empty_call_result_377589 = invoke(stypy.reporting.localization.Localization(__file__, 108, 28), empty_377583, *[tuple_377584], **kwargs_377588)
            
            # Getting the type of 'self' (line 108)
            self_377590 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 16), 'self')
            # Setting the type of the member 'rows' of a type (line 108)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 108, 16), self_377590, 'rows', empty_call_result_377589)
            
            # Assigning a Call to a Attribute (line 109):
            
            # Assigning a Call to a Attribute (line 109):
            
            # Call to empty(...): (line 109)
            # Processing the call arguments (line 109)
            
            # Obtaining an instance of the builtin type 'tuple' (line 109)
            tuple_377593 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 109, 38), 'tuple')
            # Adding type elements to the builtin type 'tuple' instance (line 109)
            # Adding element type (line 109)
            # Getting the type of 'M' (line 109)
            M_377594 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 38), 'M', False)
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 109, 38), tuple_377593, M_377594)
            
            # Processing the call keyword arguments (line 109)
            # Getting the type of 'object' (line 109)
            object_377595 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 49), 'object', False)
            keyword_377596 = object_377595
            kwargs_377597 = {'dtype': keyword_377596}
            # Getting the type of 'np' (line 109)
            np_377591 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 28), 'np', False)
            # Obtaining the member 'empty' of a type (line 109)
            empty_377592 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 109, 28), np_377591, 'empty')
            # Calling empty(args, kwargs) (line 109)
            empty_call_result_377598 = invoke(stypy.reporting.localization.Localization(__file__, 109, 28), empty_377592, *[tuple_377593], **kwargs_377597)
            
            # Getting the type of 'self' (line 109)
            self_377599 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 16), 'self')
            # Setting the type of the member 'data' of a type (line 109)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 109, 16), self_377599, 'data', empty_call_result_377598)
            
            
            # Call to range(...): (line 110)
            # Processing the call arguments (line 110)
            # Getting the type of 'M' (line 110)
            M_377601 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 31), 'M', False)
            # Processing the call keyword arguments (line 110)
            kwargs_377602 = {}
            # Getting the type of 'range' (line 110)
            range_377600 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 25), 'range', False)
            # Calling range(args, kwargs) (line 110)
            range_call_result_377603 = invoke(stypy.reporting.localization.Localization(__file__, 110, 25), range_377600, *[M_377601], **kwargs_377602)
            
            # Testing the type of a for loop iterable (line 110)
            is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 110, 16), range_call_result_377603)
            # Getting the type of the for loop variable (line 110)
            for_loop_var_377604 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 110, 16), range_call_result_377603)
            # Assigning a type to the variable 'i' (line 110)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 110, 16), 'i', for_loop_var_377604)
            # SSA begins for a for statement (line 110)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            # Assigning a List to a Subscript (line 111):
            
            # Assigning a List to a Subscript (line 111):
            
            # Obtaining an instance of the builtin type 'list' (line 111)
            list_377605 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 111, 35), 'list')
            # Adding type elements to the builtin type 'list' instance (line 111)
            
            # Getting the type of 'self' (line 111)
            self_377606 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 20), 'self')
            # Obtaining the member 'rows' of a type (line 111)
            rows_377607 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 111, 20), self_377606, 'rows')
            # Getting the type of 'i' (line 111)
            i_377608 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 30), 'i')
            # Storing an element on a container (line 111)
            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 111, 20), rows_377607, (i_377608, list_377605))
            
            # Assigning a List to a Subscript (line 112):
            
            # Assigning a List to a Subscript (line 112):
            
            # Obtaining an instance of the builtin type 'list' (line 112)
            list_377609 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 112, 35), 'list')
            # Adding type elements to the builtin type 'list' instance (line 112)
            
            # Getting the type of 'self' (line 112)
            self_377610 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 20), 'self')
            # Obtaining the member 'data' of a type (line 112)
            data_377611 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 112, 20), self_377610, 'data')
            # Getting the type of 'i' (line 112)
            i_377612 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 30), 'i')
            # Storing an element on a container (line 112)
            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 112, 20), data_377611, (i_377612, list_377609))
            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()
            
            # SSA branch for the else part of an if statement (line 103)
            module_type_store.open_ssa_branch('else')
            
            # Call to TypeError(...): (line 114)
            # Processing the call arguments (line 114)
            str_377614 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 114, 32), 'str', 'unrecognized lil_matrix constructor usage')
            # Processing the call keyword arguments (line 114)
            kwargs_377615 = {}
            # Getting the type of 'TypeError' (line 114)
            TypeError_377613 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 22), 'TypeError', False)
            # Calling TypeError(args, kwargs) (line 114)
            TypeError_call_result_377616 = invoke(stypy.reporting.localization.Localization(__file__, 114, 22), TypeError_377613, *[str_377614], **kwargs_377615)
            
            ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 114, 16), TypeError_call_result_377616, 'raise parameter', BaseException)
            # SSA join for if statement (line 103)
            module_type_store = module_type_store.join_ssa_context()
            

            if more_types_in_union_377554:
                # Runtime conditional SSA for else branch (line 102)
                module_type_store.open_ssa_branch('idiom else')



        if ((not may_be_377553) or more_types_in_union_377554):
            # Assigning a type to the variable 'arg1' (line 102)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 102, 13), 'arg1', remove_subtype_from_union(arg1_377552, tuple))
            
            
            # SSA begins for try-except statement (line 117)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
            
            # Assigning a Call to a Name (line 118):
            
            # Assigning a Call to a Name (line 118):
            
            # Call to asmatrix(...): (line 118)
            # Processing the call arguments (line 118)
            # Getting the type of 'arg1' (line 118)
            arg1_377619 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 32), 'arg1', False)
            # Processing the call keyword arguments (line 118)
            kwargs_377620 = {}
            # Getting the type of 'np' (line 118)
            np_377617 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 20), 'np', False)
            # Obtaining the member 'asmatrix' of a type (line 118)
            asmatrix_377618 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 118, 20), np_377617, 'asmatrix')
            # Calling asmatrix(args, kwargs) (line 118)
            asmatrix_call_result_377621 = invoke(stypy.reporting.localization.Localization(__file__, 118, 20), asmatrix_377618, *[arg1_377619], **kwargs_377620)
            
            # Assigning a type to the variable 'A' (line 118)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 118, 16), 'A', asmatrix_call_result_377621)
            # SSA branch for the except part of a try statement (line 117)
            # SSA branch for the except 'TypeError' branch of a try statement (line 117)
            module_type_store.open_ssa_branch('except')
            
            # Call to TypeError(...): (line 120)
            # Processing the call arguments (line 120)
            str_377623 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 120, 32), 'str', 'unsupported matrix type')
            # Processing the call keyword arguments (line 120)
            kwargs_377624 = {}
            # Getting the type of 'TypeError' (line 120)
            TypeError_377622 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 22), 'TypeError', False)
            # Calling TypeError(args, kwargs) (line 120)
            TypeError_call_result_377625 = invoke(stypy.reporting.localization.Localization(__file__, 120, 22), TypeError_377622, *[str_377623], **kwargs_377624)
            
            ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 120, 16), TypeError_call_result_377625, 'raise parameter', BaseException)
            # SSA branch for the else branch of a try statement (line 117)
            module_type_store.open_ssa_branch('except else')
            stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 122, 16))
            
            # 'from scipy.sparse.csr import csr_matrix' statement (line 122)
            update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/sparse/')
            import_377626 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 122, 16), 'scipy.sparse.csr')

            if (type(import_377626) is not StypyTypeError):

                if (import_377626 != 'pyd_module'):
                    __import__(import_377626)
                    sys_modules_377627 = sys.modules[import_377626]
                    import_from_module(stypy.reporting.localization.Localization(__file__, 122, 16), 'scipy.sparse.csr', sys_modules_377627.module_type_store, module_type_store, ['csr_matrix'])
                    nest_module(stypy.reporting.localization.Localization(__file__, 122, 16), __file__, sys_modules_377627, sys_modules_377627.module_type_store, module_type_store)
                else:
                    from scipy.sparse.csr import csr_matrix

                    import_from_module(stypy.reporting.localization.Localization(__file__, 122, 16), 'scipy.sparse.csr', None, module_type_store, ['csr_matrix'], [csr_matrix])

            else:
                # Assigning a type to the variable 'scipy.sparse.csr' (line 122)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 122, 16), 'scipy.sparse.csr', import_377626)

            remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/sparse/')
            
            
            # Assigning a Call to a Name (line 123):
            
            # Assigning a Call to a Name (line 123):
            
            # Call to tolil(...): (line 123)
            # Processing the call keyword arguments (line 123)
            kwargs_377635 = {}
            
            # Call to csr_matrix(...): (line 123)
            # Processing the call arguments (line 123)
            # Getting the type of 'A' (line 123)
            A_377629 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 31), 'A', False)
            # Processing the call keyword arguments (line 123)
            # Getting the type of 'dtype' (line 123)
            dtype_377630 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 40), 'dtype', False)
            keyword_377631 = dtype_377630
            kwargs_377632 = {'dtype': keyword_377631}
            # Getting the type of 'csr_matrix' (line 123)
            csr_matrix_377628 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 20), 'csr_matrix', False)
            # Calling csr_matrix(args, kwargs) (line 123)
            csr_matrix_call_result_377633 = invoke(stypy.reporting.localization.Localization(__file__, 123, 20), csr_matrix_377628, *[A_377629], **kwargs_377632)
            
            # Obtaining the member 'tolil' of a type (line 123)
            tolil_377634 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 123, 20), csr_matrix_call_result_377633, 'tolil')
            # Calling tolil(args, kwargs) (line 123)
            tolil_call_result_377636 = invoke(stypy.reporting.localization.Localization(__file__, 123, 20), tolil_377634, *[], **kwargs_377635)
            
            # Assigning a type to the variable 'A' (line 123)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 123, 16), 'A', tolil_call_result_377636)
            
            # Assigning a Attribute to a Attribute (line 125):
            
            # Assigning a Attribute to a Attribute (line 125):
            # Getting the type of 'A' (line 125)
            A_377637 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 29), 'A')
            # Obtaining the member 'shape' of a type (line 125)
            shape_377638 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 125, 29), A_377637, 'shape')
            # Getting the type of 'self' (line 125)
            self_377639 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 16), 'self')
            # Setting the type of the member 'shape' of a type (line 125)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 125, 16), self_377639, 'shape', shape_377638)
            
            # Assigning a Attribute to a Attribute (line 126):
            
            # Assigning a Attribute to a Attribute (line 126):
            # Getting the type of 'A' (line 126)
            A_377640 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 29), 'A')
            # Obtaining the member 'dtype' of a type (line 126)
            dtype_377641 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 126, 29), A_377640, 'dtype')
            # Getting the type of 'self' (line 126)
            self_377642 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 16), 'self')
            # Setting the type of the member 'dtype' of a type (line 126)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 126, 16), self_377642, 'dtype', dtype_377641)
            
            # Assigning a Attribute to a Attribute (line 127):
            
            # Assigning a Attribute to a Attribute (line 127):
            # Getting the type of 'A' (line 127)
            A_377643 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 28), 'A')
            # Obtaining the member 'rows' of a type (line 127)
            rows_377644 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 127, 28), A_377643, 'rows')
            # Getting the type of 'self' (line 127)
            self_377645 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 16), 'self')
            # Setting the type of the member 'rows' of a type (line 127)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 127, 16), self_377645, 'rows', rows_377644)
            
            # Assigning a Attribute to a Attribute (line 128):
            
            # Assigning a Attribute to a Attribute (line 128):
            # Getting the type of 'A' (line 128)
            A_377646 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 28), 'A')
            # Obtaining the member 'data' of a type (line 128)
            data_377647 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 128, 28), A_377646, 'data')
            # Getting the type of 'self' (line 128)
            self_377648 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 16), 'self')
            # Setting the type of the member 'data' of a type (line 128)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 128, 16), self_377648, 'data', data_377647)
            # SSA join for try-except statement (line 117)
            module_type_store = module_type_store.join_ssa_context()
            

            if (may_be_377553 and more_types_in_union_377554):
                # SSA join for if statement (line 102)
                module_type_store = module_type_store.join_ssa_context()


        
        # SSA join for if statement (line 89)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def set_shape(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'set_shape'
        module_type_store = module_type_store.open_function_context('set_shape', 130, 4, False)
        # Assigning a type to the variable 'self' (line 131)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 131, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        lil_matrix.set_shape.__dict__.__setitem__('stypy_localization', localization)
        lil_matrix.set_shape.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        lil_matrix.set_shape.__dict__.__setitem__('stypy_type_store', module_type_store)
        lil_matrix.set_shape.__dict__.__setitem__('stypy_function_name', 'lil_matrix.set_shape')
        lil_matrix.set_shape.__dict__.__setitem__('stypy_param_names_list', ['shape'])
        lil_matrix.set_shape.__dict__.__setitem__('stypy_varargs_param_name', None)
        lil_matrix.set_shape.__dict__.__setitem__('stypy_kwargs_param_name', None)
        lil_matrix.set_shape.__dict__.__setitem__('stypy_call_defaults', defaults)
        lil_matrix.set_shape.__dict__.__setitem__('stypy_call_varargs', varargs)
        lil_matrix.set_shape.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        lil_matrix.set_shape.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'lil_matrix.set_shape', ['shape'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'set_shape', localization, ['shape'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'set_shape(...)' code ##################

        
        # Assigning a Call to a Name (line 131):
        
        # Assigning a Call to a Name (line 131):
        
        # Call to tuple(...): (line 131)
        # Processing the call arguments (line 131)
        # Getting the type of 'shape' (line 131)
        shape_377650 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 22), 'shape', False)
        # Processing the call keyword arguments (line 131)
        kwargs_377651 = {}
        # Getting the type of 'tuple' (line 131)
        tuple_377649 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 16), 'tuple', False)
        # Calling tuple(args, kwargs) (line 131)
        tuple_call_result_377652 = invoke(stypy.reporting.localization.Localization(__file__, 131, 16), tuple_377649, *[shape_377650], **kwargs_377651)
        
        # Assigning a type to the variable 'shape' (line 131)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 131, 8), 'shape', tuple_call_result_377652)
        
        
        
        # Call to len(...): (line 133)
        # Processing the call arguments (line 133)
        # Getting the type of 'shape' (line 133)
        shape_377654 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 15), 'shape', False)
        # Processing the call keyword arguments (line 133)
        kwargs_377655 = {}
        # Getting the type of 'len' (line 133)
        len_377653 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 11), 'len', False)
        # Calling len(args, kwargs) (line 133)
        len_call_result_377656 = invoke(stypy.reporting.localization.Localization(__file__, 133, 11), len_377653, *[shape_377654], **kwargs_377655)
        
        int_377657 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 133, 25), 'int')
        # Applying the binary operator '!=' (line 133)
        result_ne_377658 = python_operator(stypy.reporting.localization.Localization(__file__, 133, 11), '!=', len_call_result_377656, int_377657)
        
        # Testing the type of an if condition (line 133)
        if_condition_377659 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 133, 8), result_ne_377658)
        # Assigning a type to the variable 'if_condition_377659' (line 133)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 133, 8), 'if_condition_377659', if_condition_377659)
        # SSA begins for if statement (line 133)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to ValueError(...): (line 134)
        # Processing the call arguments (line 134)
        str_377661 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 134, 29), 'str', 'Only two-dimensional sparse arrays are supported.')
        # Processing the call keyword arguments (line 134)
        kwargs_377662 = {}
        # Getting the type of 'ValueError' (line 134)
        ValueError_377660 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 18), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 134)
        ValueError_call_result_377663 = invoke(stypy.reporting.localization.Localization(__file__, 134, 18), ValueError_377660, *[str_377661], **kwargs_377662)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 134, 12), ValueError_call_result_377663, 'raise parameter', BaseException)
        # SSA join for if statement (line 133)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # SSA begins for try-except statement (line 136)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
        
        # Assigning a Tuple to a Name (line 137):
        
        # Assigning a Tuple to a Name (line 137):
        
        # Obtaining an instance of the builtin type 'tuple' (line 137)
        tuple_377664 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 137, 20), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 137)
        # Adding element type (line 137)
        
        # Call to int(...): (line 137)
        # Processing the call arguments (line 137)
        
        # Obtaining the type of the subscript
        int_377666 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 137, 30), 'int')
        # Getting the type of 'shape' (line 137)
        shape_377667 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 24), 'shape', False)
        # Obtaining the member '__getitem__' of a type (line 137)
        getitem___377668 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 137, 24), shape_377667, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 137)
        subscript_call_result_377669 = invoke(stypy.reporting.localization.Localization(__file__, 137, 24), getitem___377668, int_377666)
        
        # Processing the call keyword arguments (line 137)
        kwargs_377670 = {}
        # Getting the type of 'int' (line 137)
        int_377665 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 20), 'int', False)
        # Calling int(args, kwargs) (line 137)
        int_call_result_377671 = invoke(stypy.reporting.localization.Localization(__file__, 137, 20), int_377665, *[subscript_call_result_377669], **kwargs_377670)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 137, 20), tuple_377664, int_call_result_377671)
        # Adding element type (line 137)
        
        # Call to int(...): (line 137)
        # Processing the call arguments (line 137)
        
        # Obtaining the type of the subscript
        int_377673 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 137, 44), 'int')
        # Getting the type of 'shape' (line 137)
        shape_377674 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 38), 'shape', False)
        # Obtaining the member '__getitem__' of a type (line 137)
        getitem___377675 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 137, 38), shape_377674, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 137)
        subscript_call_result_377676 = invoke(stypy.reporting.localization.Localization(__file__, 137, 38), getitem___377675, int_377673)
        
        # Processing the call keyword arguments (line 137)
        kwargs_377677 = {}
        # Getting the type of 'int' (line 137)
        int_377672 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 34), 'int', False)
        # Calling int(args, kwargs) (line 137)
        int_call_result_377678 = invoke(stypy.reporting.localization.Localization(__file__, 137, 34), int_377672, *[subscript_call_result_377676], **kwargs_377677)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 137, 20), tuple_377664, int_call_result_377678)
        
        # Assigning a type to the variable 'shape' (line 137)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 137, 12), 'shape', tuple_377664)
        # SSA branch for the except part of a try statement (line 136)
        # SSA branch for the except '<any exception>' branch of a try statement (line 136)
        module_type_store.open_ssa_branch('except')
        
        # Call to TypeError(...): (line 139)
        # Processing the call arguments (line 139)
        str_377680 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 139, 28), 'str', 'invalid shape')
        # Processing the call keyword arguments (line 139)
        kwargs_377681 = {}
        # Getting the type of 'TypeError' (line 139)
        TypeError_377679 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 18), 'TypeError', False)
        # Calling TypeError(args, kwargs) (line 139)
        TypeError_call_result_377682 = invoke(stypy.reporting.localization.Localization(__file__, 139, 18), TypeError_377679, *[str_377680], **kwargs_377681)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 139, 12), TypeError_call_result_377682, 'raise parameter', BaseException)
        # SSA join for try-except statement (line 136)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        
        # Evaluating a boolean operation
        
        
        # Obtaining the type of the subscript
        int_377683 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 141, 22), 'int')
        # Getting the type of 'shape' (line 141)
        shape_377684 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 16), 'shape')
        # Obtaining the member '__getitem__' of a type (line 141)
        getitem___377685 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 141, 16), shape_377684, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 141)
        subscript_call_result_377686 = invoke(stypy.reporting.localization.Localization(__file__, 141, 16), getitem___377685, int_377683)
        
        int_377687 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 141, 28), 'int')
        # Applying the binary operator '>=' (line 141)
        result_ge_377688 = python_operator(stypy.reporting.localization.Localization(__file__, 141, 16), '>=', subscript_call_result_377686, int_377687)
        
        
        
        # Obtaining the type of the subscript
        int_377689 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 141, 40), 'int')
        # Getting the type of 'shape' (line 141)
        shape_377690 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 34), 'shape')
        # Obtaining the member '__getitem__' of a type (line 141)
        getitem___377691 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 141, 34), shape_377690, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 141)
        subscript_call_result_377692 = invoke(stypy.reporting.localization.Localization(__file__, 141, 34), getitem___377691, int_377689)
        
        int_377693 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 141, 46), 'int')
        # Applying the binary operator '>=' (line 141)
        result_ge_377694 = python_operator(stypy.reporting.localization.Localization(__file__, 141, 34), '>=', subscript_call_result_377692, int_377693)
        
        # Applying the binary operator 'and' (line 141)
        result_and_keyword_377695 = python_operator(stypy.reporting.localization.Localization(__file__, 141, 16), 'and', result_ge_377688, result_ge_377694)
        
        # Applying the 'not' unary operator (line 141)
        result_not__377696 = python_operator(stypy.reporting.localization.Localization(__file__, 141, 11), 'not', result_and_keyword_377695)
        
        # Testing the type of an if condition (line 141)
        if_condition_377697 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 141, 8), result_not__377696)
        # Assigning a type to the variable 'if_condition_377697' (line 141)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 141, 8), 'if_condition_377697', if_condition_377697)
        # SSA begins for if statement (line 141)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to ValueError(...): (line 142)
        # Processing the call arguments (line 142)
        str_377699 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 142, 29), 'str', 'invalid shape')
        # Processing the call keyword arguments (line 142)
        kwargs_377700 = {}
        # Getting the type of 'ValueError' (line 142)
        ValueError_377698 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 18), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 142)
        ValueError_call_result_377701 = invoke(stypy.reporting.localization.Localization(__file__, 142, 18), ValueError_377698, *[str_377699], **kwargs_377700)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 142, 12), ValueError_call_result_377701, 'raise parameter', BaseException)
        # SSA join for if statement (line 141)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Evaluating a boolean operation
        
        # Getting the type of 'self' (line 144)
        self_377702 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 12), 'self')
        # Obtaining the member '_shape' of a type (line 144)
        _shape_377703 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 144, 12), self_377702, '_shape')
        # Getting the type of 'shape' (line 144)
        shape_377704 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 27), 'shape')
        # Applying the binary operator '!=' (line 144)
        result_ne_377705 = python_operator(stypy.reporting.localization.Localization(__file__, 144, 12), '!=', _shape_377703, shape_377704)
        
        
        # Getting the type of 'self' (line 144)
        self_377706 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 39), 'self')
        # Obtaining the member '_shape' of a type (line 144)
        _shape_377707 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 144, 39), self_377706, '_shape')
        # Getting the type of 'None' (line 144)
        None_377708 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 58), 'None')
        # Applying the binary operator 'isnot' (line 144)
        result_is_not_377709 = python_operator(stypy.reporting.localization.Localization(__file__, 144, 39), 'isnot', _shape_377707, None_377708)
        
        # Applying the binary operator 'and' (line 144)
        result_and_keyword_377710 = python_operator(stypy.reporting.localization.Localization(__file__, 144, 11), 'and', result_ne_377705, result_is_not_377709)
        
        # Testing the type of an if condition (line 144)
        if_condition_377711 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 144, 8), result_and_keyword_377710)
        # Assigning a type to the variable 'if_condition_377711' (line 144)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 144, 8), 'if_condition_377711', if_condition_377711)
        # SSA begins for if statement (line 144)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        
        # SSA begins for try-except statement (line 145)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
        
        # Assigning a Call to a Name (line 146):
        
        # Assigning a Call to a Name (line 146):
        
        # Call to reshape(...): (line 146)
        # Processing the call arguments (line 146)
        # Getting the type of 'shape' (line 146)
        shape_377714 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 36), 'shape', False)
        # Processing the call keyword arguments (line 146)
        kwargs_377715 = {}
        # Getting the type of 'self' (line 146)
        self_377712 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 23), 'self', False)
        # Obtaining the member 'reshape' of a type (line 146)
        reshape_377713 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 146, 23), self_377712, 'reshape')
        # Calling reshape(args, kwargs) (line 146)
        reshape_call_result_377716 = invoke(stypy.reporting.localization.Localization(__file__, 146, 23), reshape_377713, *[shape_377714], **kwargs_377715)
        
        # Assigning a type to the variable 'self' (line 146)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 146, 16), 'self', reshape_call_result_377716)
        # SSA branch for the except part of a try statement (line 145)
        # SSA branch for the except 'NotImplementedError' branch of a try statement (line 145)
        module_type_store.open_ssa_branch('except')
        
        # Call to NotImplementedError(...): (line 148)
        # Processing the call arguments (line 148)
        str_377718 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 148, 42), 'str', 'Reshaping not implemented for %s.')
        # Getting the type of 'self' (line 149)
        self_377719 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 42), 'self', False)
        # Obtaining the member '__class__' of a type (line 149)
        class___377720 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 149, 42), self_377719, '__class__')
        # Obtaining the member '__name__' of a type (line 149)
        name___377721 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 149, 42), class___377720, '__name__')
        # Applying the binary operator '%' (line 148)
        result_mod_377722 = python_operator(stypy.reporting.localization.Localization(__file__, 148, 42), '%', str_377718, name___377721)
        
        # Processing the call keyword arguments (line 148)
        kwargs_377723 = {}
        # Getting the type of 'NotImplementedError' (line 148)
        NotImplementedError_377717 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 22), 'NotImplementedError', False)
        # Calling NotImplementedError(args, kwargs) (line 148)
        NotImplementedError_call_result_377724 = invoke(stypy.reporting.localization.Localization(__file__, 148, 22), NotImplementedError_377717, *[result_mod_377722], **kwargs_377723)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 148, 16), NotImplementedError_call_result_377724, 'raise parameter', BaseException)
        # SSA join for try-except statement (line 145)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 144)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Name to a Attribute (line 150):
        
        # Assigning a Name to a Attribute (line 150):
        # Getting the type of 'shape' (line 150)
        shape_377725 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 22), 'shape')
        # Getting the type of 'self' (line 150)
        self_377726 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 8), 'self')
        # Setting the type of the member '_shape' of a type (line 150)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 150, 8), self_377726, '_shape', shape_377725)
        
        # ################# End of 'set_shape(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'set_shape' in the type store
        # Getting the type of 'stypy_return_type' (line 130)
        stypy_return_type_377727 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_377727)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'set_shape'
        return stypy_return_type_377727

    
    # Assigning a Attribute to a Attribute (line 152):
    
    # Assigning a Call to a Name (line 154):

    @norecursion
    def __iadd__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__iadd__'
        module_type_store = module_type_store.open_function_context('__iadd__', 156, 4, False)
        # Assigning a type to the variable 'self' (line 157)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 157, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        lil_matrix.__iadd__.__dict__.__setitem__('stypy_localization', localization)
        lil_matrix.__iadd__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        lil_matrix.__iadd__.__dict__.__setitem__('stypy_type_store', module_type_store)
        lil_matrix.__iadd__.__dict__.__setitem__('stypy_function_name', 'lil_matrix.__iadd__')
        lil_matrix.__iadd__.__dict__.__setitem__('stypy_param_names_list', ['other'])
        lil_matrix.__iadd__.__dict__.__setitem__('stypy_varargs_param_name', None)
        lil_matrix.__iadd__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        lil_matrix.__iadd__.__dict__.__setitem__('stypy_call_defaults', defaults)
        lil_matrix.__iadd__.__dict__.__setitem__('stypy_call_varargs', varargs)
        lil_matrix.__iadd__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        lil_matrix.__iadd__.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'lil_matrix.__iadd__', ['other'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__iadd__', localization, ['other'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__iadd__(...)' code ##################

        
        # Assigning a BinOp to a Subscript (line 157):
        
        # Assigning a BinOp to a Subscript (line 157):
        # Getting the type of 'self' (line 157)
        self_377728 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 20), 'self')
        # Getting the type of 'other' (line 157)
        other_377729 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 27), 'other')
        # Applying the binary operator '+' (line 157)
        result_add_377730 = python_operator(stypy.reporting.localization.Localization(__file__, 157, 20), '+', self_377728, other_377729)
        
        # Getting the type of 'self' (line 157)
        self_377731 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 8), 'self')
        slice_377732 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 157, 8), None, None, None)
        slice_377733 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 157, 8), None, None, None)
        # Storing an element on a container (line 157)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 157, 8), self_377731, ((slice_377732, slice_377733), result_add_377730))
        # Getting the type of 'self' (line 158)
        self_377734 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 15), 'self')
        # Assigning a type to the variable 'stypy_return_type' (line 158)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 158, 8), 'stypy_return_type', self_377734)
        
        # ################# End of '__iadd__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__iadd__' in the type store
        # Getting the type of 'stypy_return_type' (line 156)
        stypy_return_type_377735 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_377735)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__iadd__'
        return stypy_return_type_377735


    @norecursion
    def __isub__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__isub__'
        module_type_store = module_type_store.open_function_context('__isub__', 160, 4, False)
        # Assigning a type to the variable 'self' (line 161)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 161, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        lil_matrix.__isub__.__dict__.__setitem__('stypy_localization', localization)
        lil_matrix.__isub__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        lil_matrix.__isub__.__dict__.__setitem__('stypy_type_store', module_type_store)
        lil_matrix.__isub__.__dict__.__setitem__('stypy_function_name', 'lil_matrix.__isub__')
        lil_matrix.__isub__.__dict__.__setitem__('stypy_param_names_list', ['other'])
        lil_matrix.__isub__.__dict__.__setitem__('stypy_varargs_param_name', None)
        lil_matrix.__isub__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        lil_matrix.__isub__.__dict__.__setitem__('stypy_call_defaults', defaults)
        lil_matrix.__isub__.__dict__.__setitem__('stypy_call_varargs', varargs)
        lil_matrix.__isub__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        lil_matrix.__isub__.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'lil_matrix.__isub__', ['other'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__isub__', localization, ['other'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__isub__(...)' code ##################

        
        # Assigning a BinOp to a Subscript (line 161):
        
        # Assigning a BinOp to a Subscript (line 161):
        # Getting the type of 'self' (line 161)
        self_377736 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 20), 'self')
        # Getting the type of 'other' (line 161)
        other_377737 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 27), 'other')
        # Applying the binary operator '-' (line 161)
        result_sub_377738 = python_operator(stypy.reporting.localization.Localization(__file__, 161, 20), '-', self_377736, other_377737)
        
        # Getting the type of 'self' (line 161)
        self_377739 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 8), 'self')
        slice_377740 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 161, 8), None, None, None)
        slice_377741 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 161, 8), None, None, None)
        # Storing an element on a container (line 161)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 161, 8), self_377739, ((slice_377740, slice_377741), result_sub_377738))
        # Getting the type of 'self' (line 162)
        self_377742 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 15), 'self')
        # Assigning a type to the variable 'stypy_return_type' (line 162)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 162, 8), 'stypy_return_type', self_377742)
        
        # ################# End of '__isub__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__isub__' in the type store
        # Getting the type of 'stypy_return_type' (line 160)
        stypy_return_type_377743 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_377743)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__isub__'
        return stypy_return_type_377743


    @norecursion
    def __imul__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__imul__'
        module_type_store = module_type_store.open_function_context('__imul__', 164, 4, False)
        # Assigning a type to the variable 'self' (line 165)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 165, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        lil_matrix.__imul__.__dict__.__setitem__('stypy_localization', localization)
        lil_matrix.__imul__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        lil_matrix.__imul__.__dict__.__setitem__('stypy_type_store', module_type_store)
        lil_matrix.__imul__.__dict__.__setitem__('stypy_function_name', 'lil_matrix.__imul__')
        lil_matrix.__imul__.__dict__.__setitem__('stypy_param_names_list', ['other'])
        lil_matrix.__imul__.__dict__.__setitem__('stypy_varargs_param_name', None)
        lil_matrix.__imul__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        lil_matrix.__imul__.__dict__.__setitem__('stypy_call_defaults', defaults)
        lil_matrix.__imul__.__dict__.__setitem__('stypy_call_varargs', varargs)
        lil_matrix.__imul__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        lil_matrix.__imul__.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'lil_matrix.__imul__', ['other'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__imul__', localization, ['other'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__imul__(...)' code ##################

        
        
        # Call to isscalarlike(...): (line 165)
        # Processing the call arguments (line 165)
        # Getting the type of 'other' (line 165)
        other_377745 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 24), 'other', False)
        # Processing the call keyword arguments (line 165)
        kwargs_377746 = {}
        # Getting the type of 'isscalarlike' (line 165)
        isscalarlike_377744 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 11), 'isscalarlike', False)
        # Calling isscalarlike(args, kwargs) (line 165)
        isscalarlike_call_result_377747 = invoke(stypy.reporting.localization.Localization(__file__, 165, 11), isscalarlike_377744, *[other_377745], **kwargs_377746)
        
        # Testing the type of an if condition (line 165)
        if_condition_377748 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 165, 8), isscalarlike_call_result_377747)
        # Assigning a type to the variable 'if_condition_377748' (line 165)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 165, 8), 'if_condition_377748', if_condition_377748)
        # SSA begins for if statement (line 165)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a BinOp to a Subscript (line 166):
        
        # Assigning a BinOp to a Subscript (line 166):
        # Getting the type of 'self' (line 166)
        self_377749 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 24), 'self')
        # Getting the type of 'other' (line 166)
        other_377750 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 31), 'other')
        # Applying the binary operator '*' (line 166)
        result_mul_377751 = python_operator(stypy.reporting.localization.Localization(__file__, 166, 24), '*', self_377749, other_377750)
        
        # Getting the type of 'self' (line 166)
        self_377752 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 12), 'self')
        slice_377753 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 166, 12), None, None, None)
        slice_377754 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 166, 12), None, None, None)
        # Storing an element on a container (line 166)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 166, 12), self_377752, ((slice_377753, slice_377754), result_mul_377751))
        # Getting the type of 'self' (line 167)
        self_377755 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 19), 'self')
        # Assigning a type to the variable 'stypy_return_type' (line 167)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 167, 12), 'stypy_return_type', self_377755)
        # SSA branch for the else part of an if statement (line 165)
        module_type_store.open_ssa_branch('else')
        # Getting the type of 'NotImplemented' (line 169)
        NotImplemented_377756 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 19), 'NotImplemented')
        # Assigning a type to the variable 'stypy_return_type' (line 169)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 169, 12), 'stypy_return_type', NotImplemented_377756)
        # SSA join for if statement (line 165)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of '__imul__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__imul__' in the type store
        # Getting the type of 'stypy_return_type' (line 164)
        stypy_return_type_377757 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_377757)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__imul__'
        return stypy_return_type_377757


    @norecursion
    def __itruediv__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__itruediv__'
        module_type_store = module_type_store.open_function_context('__itruediv__', 171, 4, False)
        # Assigning a type to the variable 'self' (line 172)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 172, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        lil_matrix.__itruediv__.__dict__.__setitem__('stypy_localization', localization)
        lil_matrix.__itruediv__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        lil_matrix.__itruediv__.__dict__.__setitem__('stypy_type_store', module_type_store)
        lil_matrix.__itruediv__.__dict__.__setitem__('stypy_function_name', 'lil_matrix.__itruediv__')
        lil_matrix.__itruediv__.__dict__.__setitem__('stypy_param_names_list', ['other'])
        lil_matrix.__itruediv__.__dict__.__setitem__('stypy_varargs_param_name', None)
        lil_matrix.__itruediv__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        lil_matrix.__itruediv__.__dict__.__setitem__('stypy_call_defaults', defaults)
        lil_matrix.__itruediv__.__dict__.__setitem__('stypy_call_varargs', varargs)
        lil_matrix.__itruediv__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        lil_matrix.__itruediv__.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'lil_matrix.__itruediv__', ['other'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__itruediv__', localization, ['other'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__itruediv__(...)' code ##################

        
        
        # Call to isscalarlike(...): (line 172)
        # Processing the call arguments (line 172)
        # Getting the type of 'other' (line 172)
        other_377759 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 24), 'other', False)
        # Processing the call keyword arguments (line 172)
        kwargs_377760 = {}
        # Getting the type of 'isscalarlike' (line 172)
        isscalarlike_377758 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 11), 'isscalarlike', False)
        # Calling isscalarlike(args, kwargs) (line 172)
        isscalarlike_call_result_377761 = invoke(stypy.reporting.localization.Localization(__file__, 172, 11), isscalarlike_377758, *[other_377759], **kwargs_377760)
        
        # Testing the type of an if condition (line 172)
        if_condition_377762 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 172, 8), isscalarlike_call_result_377761)
        # Assigning a type to the variable 'if_condition_377762' (line 172)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 172, 8), 'if_condition_377762', if_condition_377762)
        # SSA begins for if statement (line 172)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a BinOp to a Subscript (line 173):
        
        # Assigning a BinOp to a Subscript (line 173):
        # Getting the type of 'self' (line 173)
        self_377763 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 24), 'self')
        # Getting the type of 'other' (line 173)
        other_377764 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 31), 'other')
        # Applying the binary operator 'div' (line 173)
        result_div_377765 = python_operator(stypy.reporting.localization.Localization(__file__, 173, 24), 'div', self_377763, other_377764)
        
        # Getting the type of 'self' (line 173)
        self_377766 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 12), 'self')
        slice_377767 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 173, 12), None, None, None)
        slice_377768 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 173, 12), None, None, None)
        # Storing an element on a container (line 173)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 173, 12), self_377766, ((slice_377767, slice_377768), result_div_377765))
        # Getting the type of 'self' (line 174)
        self_377769 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 19), 'self')
        # Assigning a type to the variable 'stypy_return_type' (line 174)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 174, 12), 'stypy_return_type', self_377769)
        # SSA branch for the else part of an if statement (line 172)
        module_type_store.open_ssa_branch('else')
        # Getting the type of 'NotImplemented' (line 176)
        NotImplemented_377770 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 19), 'NotImplemented')
        # Assigning a type to the variable 'stypy_return_type' (line 176)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 176, 12), 'stypy_return_type', NotImplemented_377770)
        # SSA join for if statement (line 172)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of '__itruediv__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__itruediv__' in the type store
        # Getting the type of 'stypy_return_type' (line 171)
        stypy_return_type_377771 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_377771)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__itruediv__'
        return stypy_return_type_377771


    @norecursion
    def getnnz(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 181)
        None_377772 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 26), 'None')
        defaults = [None_377772]
        # Create a new context for function 'getnnz'
        module_type_store = module_type_store.open_function_context('getnnz', 181, 4, False)
        # Assigning a type to the variable 'self' (line 182)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 182, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        lil_matrix.getnnz.__dict__.__setitem__('stypy_localization', localization)
        lil_matrix.getnnz.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        lil_matrix.getnnz.__dict__.__setitem__('stypy_type_store', module_type_store)
        lil_matrix.getnnz.__dict__.__setitem__('stypy_function_name', 'lil_matrix.getnnz')
        lil_matrix.getnnz.__dict__.__setitem__('stypy_param_names_list', ['axis'])
        lil_matrix.getnnz.__dict__.__setitem__('stypy_varargs_param_name', None)
        lil_matrix.getnnz.__dict__.__setitem__('stypy_kwargs_param_name', None)
        lil_matrix.getnnz.__dict__.__setitem__('stypy_call_defaults', defaults)
        lil_matrix.getnnz.__dict__.__setitem__('stypy_call_varargs', varargs)
        lil_matrix.getnnz.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        lil_matrix.getnnz.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'lil_matrix.getnnz', ['axis'], None, None, defaults, varargs, kwargs)

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

        
        # Type idiom detected: calculating its left and rigth part (line 182)
        # Getting the type of 'axis' (line 182)
        axis_377773 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 182, 11), 'axis')
        # Getting the type of 'None' (line 182)
        None_377774 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 182, 19), 'None')
        
        (may_be_377775, more_types_in_union_377776) = may_be_none(axis_377773, None_377774)

        if may_be_377775:

            if more_types_in_union_377776:
                # Runtime conditional SSA (line 182)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Call to sum(...): (line 183)
            # Processing the call arguments (line 183)
            # Calculating list comprehension
            # Calculating comprehension expression
            # Getting the type of 'self' (line 183)
            self_377782 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 52), 'self', False)
            # Obtaining the member 'data' of a type (line 183)
            data_377783 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 183, 52), self_377782, 'data')
            comprehension_377784 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 183, 24), data_377783)
            # Assigning a type to the variable 'rowvals' (line 183)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 183, 24), 'rowvals', comprehension_377784)
            
            # Call to len(...): (line 183)
            # Processing the call arguments (line 183)
            # Getting the type of 'rowvals' (line 183)
            rowvals_377779 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 28), 'rowvals', False)
            # Processing the call keyword arguments (line 183)
            kwargs_377780 = {}
            # Getting the type of 'len' (line 183)
            len_377778 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 24), 'len', False)
            # Calling len(args, kwargs) (line 183)
            len_call_result_377781 = invoke(stypy.reporting.localization.Localization(__file__, 183, 24), len_377778, *[rowvals_377779], **kwargs_377780)
            
            list_377785 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 183, 24), 'list')
            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 183, 24), list_377785, len_call_result_377781)
            # Processing the call keyword arguments (line 183)
            kwargs_377786 = {}
            # Getting the type of 'sum' (line 183)
            sum_377777 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 19), 'sum', False)
            # Calling sum(args, kwargs) (line 183)
            sum_call_result_377787 = invoke(stypy.reporting.localization.Localization(__file__, 183, 19), sum_377777, *[list_377785], **kwargs_377786)
            
            # Assigning a type to the variable 'stypy_return_type' (line 183)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 183, 12), 'stypy_return_type', sum_call_result_377787)

            if more_types_in_union_377776:
                # SSA join for if statement (line 182)
                module_type_store = module_type_store.join_ssa_context()


        
        
        
        # Getting the type of 'axis' (line 184)
        axis_377788 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 11), 'axis')
        int_377789 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 184, 18), 'int')
        # Applying the binary operator '<' (line 184)
        result_lt_377790 = python_operator(stypy.reporting.localization.Localization(__file__, 184, 11), '<', axis_377788, int_377789)
        
        # Testing the type of an if condition (line 184)
        if_condition_377791 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 184, 8), result_lt_377790)
        # Assigning a type to the variable 'if_condition_377791' (line 184)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 184, 8), 'if_condition_377791', if_condition_377791)
        # SSA begins for if statement (line 184)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Getting the type of 'axis' (line 185)
        axis_377792 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 12), 'axis')
        int_377793 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 185, 20), 'int')
        # Applying the binary operator '+=' (line 185)
        result_iadd_377794 = python_operator(stypy.reporting.localization.Localization(__file__, 185, 12), '+=', axis_377792, int_377793)
        # Assigning a type to the variable 'axis' (line 185)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 185, 12), 'axis', result_iadd_377794)
        
        # SSA join for if statement (line 184)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Getting the type of 'axis' (line 186)
        axis_377795 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 11), 'axis')
        int_377796 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 186, 19), 'int')
        # Applying the binary operator '==' (line 186)
        result_eq_377797 = python_operator(stypy.reporting.localization.Localization(__file__, 186, 11), '==', axis_377795, int_377796)
        
        # Testing the type of an if condition (line 186)
        if_condition_377798 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 186, 8), result_eq_377797)
        # Assigning a type to the variable 'if_condition_377798' (line 186)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 186, 8), 'if_condition_377798', if_condition_377798)
        # SSA begins for if statement (line 186)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 187):
        
        # Assigning a Call to a Name (line 187):
        
        # Call to zeros(...): (line 187)
        # Processing the call arguments (line 187)
        
        # Obtaining the type of the subscript
        int_377801 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 187, 38), 'int')
        # Getting the type of 'self' (line 187)
        self_377802 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 27), 'self', False)
        # Obtaining the member 'shape' of a type (line 187)
        shape_377803 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 187, 27), self_377802, 'shape')
        # Obtaining the member '__getitem__' of a type (line 187)
        getitem___377804 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 187, 27), shape_377803, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 187)
        subscript_call_result_377805 = invoke(stypy.reporting.localization.Localization(__file__, 187, 27), getitem___377804, int_377801)
        
        # Processing the call keyword arguments (line 187)
        # Getting the type of 'np' (line 187)
        np_377806 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 48), 'np', False)
        # Obtaining the member 'intp' of a type (line 187)
        intp_377807 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 187, 48), np_377806, 'intp')
        keyword_377808 = intp_377807
        kwargs_377809 = {'dtype': keyword_377808}
        # Getting the type of 'np' (line 187)
        np_377799 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 18), 'np', False)
        # Obtaining the member 'zeros' of a type (line 187)
        zeros_377800 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 187, 18), np_377799, 'zeros')
        # Calling zeros(args, kwargs) (line 187)
        zeros_call_result_377810 = invoke(stypy.reporting.localization.Localization(__file__, 187, 18), zeros_377800, *[subscript_call_result_377805], **kwargs_377809)
        
        # Assigning a type to the variable 'out' (line 187)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 187, 12), 'out', zeros_call_result_377810)
        
        # Getting the type of 'self' (line 188)
        self_377811 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 188, 23), 'self')
        # Obtaining the member 'rows' of a type (line 188)
        rows_377812 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 188, 23), self_377811, 'rows')
        # Testing the type of a for loop iterable (line 188)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 188, 12), rows_377812)
        # Getting the type of the for loop variable (line 188)
        for_loop_var_377813 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 188, 12), rows_377812)
        # Assigning a type to the variable 'row' (line 188)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 188, 12), 'row', for_loop_var_377813)
        # SSA begins for a for statement (line 188)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Getting the type of 'out' (line 189)
        out_377814 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 16), 'out')
        
        # Obtaining the type of the subscript
        # Getting the type of 'row' (line 189)
        row_377815 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 20), 'row')
        # Getting the type of 'out' (line 189)
        out_377816 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 16), 'out')
        # Obtaining the member '__getitem__' of a type (line 189)
        getitem___377817 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 189, 16), out_377816, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 189)
        subscript_call_result_377818 = invoke(stypy.reporting.localization.Localization(__file__, 189, 16), getitem___377817, row_377815)
        
        int_377819 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 189, 28), 'int')
        # Applying the binary operator '+=' (line 189)
        result_iadd_377820 = python_operator(stypy.reporting.localization.Localization(__file__, 189, 16), '+=', subscript_call_result_377818, int_377819)
        # Getting the type of 'out' (line 189)
        out_377821 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 16), 'out')
        # Getting the type of 'row' (line 189)
        row_377822 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 20), 'row')
        # Storing an element on a container (line 189)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 189, 16), out_377821, (row_377822, result_iadd_377820))
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        # Getting the type of 'out' (line 190)
        out_377823 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 190, 19), 'out')
        # Assigning a type to the variable 'stypy_return_type' (line 190)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 190, 12), 'stypy_return_type', out_377823)
        # SSA branch for the else part of an if statement (line 186)
        module_type_store.open_ssa_branch('else')
        
        
        # Getting the type of 'axis' (line 191)
        axis_377824 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 13), 'axis')
        int_377825 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 191, 21), 'int')
        # Applying the binary operator '==' (line 191)
        result_eq_377826 = python_operator(stypy.reporting.localization.Localization(__file__, 191, 13), '==', axis_377824, int_377825)
        
        # Testing the type of an if condition (line 191)
        if_condition_377827 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 191, 13), result_eq_377826)
        # Assigning a type to the variable 'if_condition_377827' (line 191)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 191, 13), 'if_condition_377827', if_condition_377827)
        # SSA begins for if statement (line 191)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to array(...): (line 192)
        # Processing the call arguments (line 192)
        # Calculating list comprehension
        # Calculating comprehension expression
        # Getting the type of 'self' (line 192)
        self_377834 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 192, 57), 'self', False)
        # Obtaining the member 'data' of a type (line 192)
        data_377835 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 192, 57), self_377834, 'data')
        comprehension_377836 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 192, 29), data_377835)
        # Assigning a type to the variable 'rowvals' (line 192)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 192, 29), 'rowvals', comprehension_377836)
        
        # Call to len(...): (line 192)
        # Processing the call arguments (line 192)
        # Getting the type of 'rowvals' (line 192)
        rowvals_377831 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 192, 33), 'rowvals', False)
        # Processing the call keyword arguments (line 192)
        kwargs_377832 = {}
        # Getting the type of 'len' (line 192)
        len_377830 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 192, 29), 'len', False)
        # Calling len(args, kwargs) (line 192)
        len_call_result_377833 = invoke(stypy.reporting.localization.Localization(__file__, 192, 29), len_377830, *[rowvals_377831], **kwargs_377832)
        
        list_377837 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 192, 29), 'list')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 192, 29), list_377837, len_call_result_377833)
        # Processing the call keyword arguments (line 192)
        # Getting the type of 'np' (line 192)
        np_377838 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 192, 75), 'np', False)
        # Obtaining the member 'intp' of a type (line 192)
        intp_377839 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 192, 75), np_377838, 'intp')
        keyword_377840 = intp_377839
        kwargs_377841 = {'dtype': keyword_377840}
        # Getting the type of 'np' (line 192)
        np_377828 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 192, 19), 'np', False)
        # Obtaining the member 'array' of a type (line 192)
        array_377829 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 192, 19), np_377828, 'array')
        # Calling array(args, kwargs) (line 192)
        array_call_result_377842 = invoke(stypy.reporting.localization.Localization(__file__, 192, 19), array_377829, *[list_377837], **kwargs_377841)
        
        # Assigning a type to the variable 'stypy_return_type' (line 192)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 192, 12), 'stypy_return_type', array_call_result_377842)
        # SSA branch for the else part of an if statement (line 191)
        module_type_store.open_ssa_branch('else')
        
        # Call to ValueError(...): (line 194)
        # Processing the call arguments (line 194)
        str_377844 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 194, 29), 'str', 'axis out of bounds')
        # Processing the call keyword arguments (line 194)
        kwargs_377845 = {}
        # Getting the type of 'ValueError' (line 194)
        ValueError_377843 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 194, 18), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 194)
        ValueError_call_result_377846 = invoke(stypy.reporting.localization.Localization(__file__, 194, 18), ValueError_377843, *[str_377844], **kwargs_377845)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 194, 12), ValueError_call_result_377846, 'raise parameter', BaseException)
        # SSA join for if statement (line 191)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 186)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'getnnz(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'getnnz' in the type store
        # Getting the type of 'stypy_return_type' (line 181)
        stypy_return_type_377847 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_377847)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'getnnz'
        return stypy_return_type_377847


    @norecursion
    def count_nonzero(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'count_nonzero'
        module_type_store = module_type_store.open_function_context('count_nonzero', 196, 4, False)
        # Assigning a type to the variable 'self' (line 197)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 197, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        lil_matrix.count_nonzero.__dict__.__setitem__('stypy_localization', localization)
        lil_matrix.count_nonzero.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        lil_matrix.count_nonzero.__dict__.__setitem__('stypy_type_store', module_type_store)
        lil_matrix.count_nonzero.__dict__.__setitem__('stypy_function_name', 'lil_matrix.count_nonzero')
        lil_matrix.count_nonzero.__dict__.__setitem__('stypy_param_names_list', [])
        lil_matrix.count_nonzero.__dict__.__setitem__('stypy_varargs_param_name', None)
        lil_matrix.count_nonzero.__dict__.__setitem__('stypy_kwargs_param_name', None)
        lil_matrix.count_nonzero.__dict__.__setitem__('stypy_call_defaults', defaults)
        lil_matrix.count_nonzero.__dict__.__setitem__('stypy_call_varargs', varargs)
        lil_matrix.count_nonzero.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        lil_matrix.count_nonzero.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'lil_matrix.count_nonzero', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'count_nonzero', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'count_nonzero(...)' code ##################

        
        # Call to sum(...): (line 197)
        # Processing the call arguments (line 197)
        # Calculating generator expression
        module_type_store = module_type_store.open_function_context('list comprehension expression', 197, 19, True)
        # Calculating comprehension expression
        # Getting the type of 'self' (line 197)
        self_377854 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 197, 60), 'self', False)
        # Obtaining the member 'data' of a type (line 197)
        data_377855 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 197, 60), self_377854, 'data')
        comprehension_377856 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 197, 19), data_377855)
        # Assigning a type to the variable 'rowvals' (line 197)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 197, 19), 'rowvals', comprehension_377856)
        
        # Call to count_nonzero(...): (line 197)
        # Processing the call arguments (line 197)
        # Getting the type of 'rowvals' (line 197)
        rowvals_377851 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 197, 36), 'rowvals', False)
        # Processing the call keyword arguments (line 197)
        kwargs_377852 = {}
        # Getting the type of 'np' (line 197)
        np_377849 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 197, 19), 'np', False)
        # Obtaining the member 'count_nonzero' of a type (line 197)
        count_nonzero_377850 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 197, 19), np_377849, 'count_nonzero')
        # Calling count_nonzero(args, kwargs) (line 197)
        count_nonzero_call_result_377853 = invoke(stypy.reporting.localization.Localization(__file__, 197, 19), count_nonzero_377850, *[rowvals_377851], **kwargs_377852)
        
        list_377857 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 197, 19), 'list')
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 197, 19), list_377857, count_nonzero_call_result_377853)
        # Processing the call keyword arguments (line 197)
        kwargs_377858 = {}
        # Getting the type of 'sum' (line 197)
        sum_377848 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 197, 15), 'sum', False)
        # Calling sum(args, kwargs) (line 197)
        sum_call_result_377859 = invoke(stypy.reporting.localization.Localization(__file__, 197, 15), sum_377848, *[list_377857], **kwargs_377858)
        
        # Assigning a type to the variable 'stypy_return_type' (line 197)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 197, 8), 'stypy_return_type', sum_call_result_377859)
        
        # ################# End of 'count_nonzero(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'count_nonzero' in the type store
        # Getting the type of 'stypy_return_type' (line 196)
        stypy_return_type_377860 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 196, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_377860)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'count_nonzero'
        return stypy_return_type_377860

    
    # Assigning a Attribute to a Attribute (line 199):
    
    # Assigning a Attribute to a Attribute (line 200):

    @norecursion
    def stypy__str__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__str__'
        module_type_store = module_type_store.open_function_context('__str__', 202, 4, False)
        # Assigning a type to the variable 'self' (line 203)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 203, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        lil_matrix.stypy__str__.__dict__.__setitem__('stypy_localization', localization)
        lil_matrix.stypy__str__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        lil_matrix.stypy__str__.__dict__.__setitem__('stypy_type_store', module_type_store)
        lil_matrix.stypy__str__.__dict__.__setitem__('stypy_function_name', 'lil_matrix.stypy__str__')
        lil_matrix.stypy__str__.__dict__.__setitem__('stypy_param_names_list', [])
        lil_matrix.stypy__str__.__dict__.__setitem__('stypy_varargs_param_name', None)
        lil_matrix.stypy__str__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        lil_matrix.stypy__str__.__dict__.__setitem__('stypy_call_defaults', defaults)
        lil_matrix.stypy__str__.__dict__.__setitem__('stypy_call_varargs', varargs)
        lil_matrix.stypy__str__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        lil_matrix.stypy__str__.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'lil_matrix.stypy__str__', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__str__', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__str__(...)' code ##################

        
        # Assigning a Str to a Name (line 203):
        
        # Assigning a Str to a Name (line 203):
        str_377861 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 203, 14), 'str', '')
        # Assigning a type to the variable 'val' (line 203)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 203, 8), 'val', str_377861)
        
        
        # Call to enumerate(...): (line 204)
        # Processing the call arguments (line 204)
        # Getting the type of 'self' (line 204)
        self_377863 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 204, 32), 'self', False)
        # Obtaining the member 'rows' of a type (line 204)
        rows_377864 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 204, 32), self_377863, 'rows')
        # Processing the call keyword arguments (line 204)
        kwargs_377865 = {}
        # Getting the type of 'enumerate' (line 204)
        enumerate_377862 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 204, 22), 'enumerate', False)
        # Calling enumerate(args, kwargs) (line 204)
        enumerate_call_result_377866 = invoke(stypy.reporting.localization.Localization(__file__, 204, 22), enumerate_377862, *[rows_377864], **kwargs_377865)
        
        # Testing the type of a for loop iterable (line 204)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 204, 8), enumerate_call_result_377866)
        # Getting the type of the for loop variable (line 204)
        for_loop_var_377867 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 204, 8), enumerate_call_result_377866)
        # Assigning a type to the variable 'i' (line 204)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 204, 8), 'i', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 204, 8), for_loop_var_377867))
        # Assigning a type to the variable 'row' (line 204)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 204, 8), 'row', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 204, 8), for_loop_var_377867))
        # SSA begins for a for statement (line 204)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        
        # Call to enumerate(...): (line 205)
        # Processing the call arguments (line 205)
        # Getting the type of 'row' (line 205)
        row_377869 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 36), 'row', False)
        # Processing the call keyword arguments (line 205)
        kwargs_377870 = {}
        # Getting the type of 'enumerate' (line 205)
        enumerate_377868 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 26), 'enumerate', False)
        # Calling enumerate(args, kwargs) (line 205)
        enumerate_call_result_377871 = invoke(stypy.reporting.localization.Localization(__file__, 205, 26), enumerate_377868, *[row_377869], **kwargs_377870)
        
        # Testing the type of a for loop iterable (line 205)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 205, 12), enumerate_call_result_377871)
        # Getting the type of the for loop variable (line 205)
        for_loop_var_377872 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 205, 12), enumerate_call_result_377871)
        # Assigning a type to the variable 'pos' (line 205)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 205, 12), 'pos', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 205, 12), for_loop_var_377872))
        # Assigning a type to the variable 'j' (line 205)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 205, 12), 'j', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 205, 12), for_loop_var_377872))
        # SSA begins for a for statement (line 205)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Getting the type of 'val' (line 206)
        val_377873 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 206, 16), 'val')
        str_377874 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 206, 23), 'str', '  %s\t%s\n')
        
        # Obtaining an instance of the builtin type 'tuple' (line 206)
        tuple_377875 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 206, 39), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 206)
        # Adding element type (line 206)
        
        # Call to str(...): (line 206)
        # Processing the call arguments (line 206)
        
        # Obtaining an instance of the builtin type 'tuple' (line 206)
        tuple_377877 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 206, 44), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 206)
        # Adding element type (line 206)
        # Getting the type of 'i' (line 206)
        i_377878 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 206, 44), 'i', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 206, 44), tuple_377877, i_377878)
        # Adding element type (line 206)
        # Getting the type of 'j' (line 206)
        j_377879 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 206, 47), 'j', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 206, 44), tuple_377877, j_377879)
        
        # Processing the call keyword arguments (line 206)
        kwargs_377880 = {}
        # Getting the type of 'str' (line 206)
        str_377876 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 206, 39), 'str', False)
        # Calling str(args, kwargs) (line 206)
        str_call_result_377881 = invoke(stypy.reporting.localization.Localization(__file__, 206, 39), str_377876, *[tuple_377877], **kwargs_377880)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 206, 39), tuple_377875, str_call_result_377881)
        # Adding element type (line 206)
        
        # Call to str(...): (line 206)
        # Processing the call arguments (line 206)
        
        # Obtaining the type of the subscript
        # Getting the type of 'pos' (line 206)
        pos_377883 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 206, 69), 'pos', False)
        
        # Obtaining the type of the subscript
        # Getting the type of 'i' (line 206)
        i_377884 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 206, 66), 'i', False)
        # Getting the type of 'self' (line 206)
        self_377885 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 206, 56), 'self', False)
        # Obtaining the member 'data' of a type (line 206)
        data_377886 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 206, 56), self_377885, 'data')
        # Obtaining the member '__getitem__' of a type (line 206)
        getitem___377887 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 206, 56), data_377886, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 206)
        subscript_call_result_377888 = invoke(stypy.reporting.localization.Localization(__file__, 206, 56), getitem___377887, i_377884)
        
        # Obtaining the member '__getitem__' of a type (line 206)
        getitem___377889 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 206, 56), subscript_call_result_377888, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 206)
        subscript_call_result_377890 = invoke(stypy.reporting.localization.Localization(__file__, 206, 56), getitem___377889, pos_377883)
        
        # Processing the call keyword arguments (line 206)
        kwargs_377891 = {}
        # Getting the type of 'str' (line 206)
        str_377882 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 206, 52), 'str', False)
        # Calling str(args, kwargs) (line 206)
        str_call_result_377892 = invoke(stypy.reporting.localization.Localization(__file__, 206, 52), str_377882, *[subscript_call_result_377890], **kwargs_377891)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 206, 39), tuple_377875, str_call_result_377892)
        
        # Applying the binary operator '%' (line 206)
        result_mod_377893 = python_operator(stypy.reporting.localization.Localization(__file__, 206, 23), '%', str_377874, tuple_377875)
        
        # Applying the binary operator '+=' (line 206)
        result_iadd_377894 = python_operator(stypy.reporting.localization.Localization(__file__, 206, 16), '+=', val_377873, result_mod_377893)
        # Assigning a type to the variable 'val' (line 206)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 206, 16), 'val', result_iadd_377894)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Obtaining the type of the subscript
        int_377895 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 207, 20), 'int')
        slice_377896 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 207, 15), None, int_377895, None)
        # Getting the type of 'val' (line 207)
        val_377897 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 207, 15), 'val')
        # Obtaining the member '__getitem__' of a type (line 207)
        getitem___377898 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 207, 15), val_377897, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 207)
        subscript_call_result_377899 = invoke(stypy.reporting.localization.Localization(__file__, 207, 15), getitem___377898, slice_377896)
        
        # Assigning a type to the variable 'stypy_return_type' (line 207)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 207, 8), 'stypy_return_type', subscript_call_result_377899)
        
        # ################# End of '__str__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__str__' in the type store
        # Getting the type of 'stypy_return_type' (line 202)
        stypy_return_type_377900 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_377900)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__str__'
        return stypy_return_type_377900


    @norecursion
    def getrowview(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'getrowview'
        module_type_store = module_type_store.open_function_context('getrowview', 209, 4, False)
        # Assigning a type to the variable 'self' (line 210)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 210, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        lil_matrix.getrowview.__dict__.__setitem__('stypy_localization', localization)
        lil_matrix.getrowview.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        lil_matrix.getrowview.__dict__.__setitem__('stypy_type_store', module_type_store)
        lil_matrix.getrowview.__dict__.__setitem__('stypy_function_name', 'lil_matrix.getrowview')
        lil_matrix.getrowview.__dict__.__setitem__('stypy_param_names_list', ['i'])
        lil_matrix.getrowview.__dict__.__setitem__('stypy_varargs_param_name', None)
        lil_matrix.getrowview.__dict__.__setitem__('stypy_kwargs_param_name', None)
        lil_matrix.getrowview.__dict__.__setitem__('stypy_call_defaults', defaults)
        lil_matrix.getrowview.__dict__.__setitem__('stypy_call_varargs', varargs)
        lil_matrix.getrowview.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        lil_matrix.getrowview.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'lil_matrix.getrowview', ['i'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'getrowview', localization, ['i'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'getrowview(...)' code ##################

        str_377901 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 211, (-1)), 'str', "Returns a view of the 'i'th row (without copying).\n        ")
        
        # Assigning a Call to a Name (line 212):
        
        # Assigning a Call to a Name (line 212):
        
        # Call to lil_matrix(...): (line 212)
        # Processing the call arguments (line 212)
        
        # Obtaining an instance of the builtin type 'tuple' (line 212)
        tuple_377903 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 212, 26), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 212)
        # Adding element type (line 212)
        int_377904 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 212, 26), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 212, 26), tuple_377903, int_377904)
        # Adding element type (line 212)
        
        # Obtaining the type of the subscript
        int_377905 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 212, 40), 'int')
        # Getting the type of 'self' (line 212)
        self_377906 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 29), 'self', False)
        # Obtaining the member 'shape' of a type (line 212)
        shape_377907 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 212, 29), self_377906, 'shape')
        # Obtaining the member '__getitem__' of a type (line 212)
        getitem___377908 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 212, 29), shape_377907, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 212)
        subscript_call_result_377909 = invoke(stypy.reporting.localization.Localization(__file__, 212, 29), getitem___377908, int_377905)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 212, 26), tuple_377903, subscript_call_result_377909)
        
        # Processing the call keyword arguments (line 212)
        # Getting the type of 'self' (line 212)
        self_377910 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 51), 'self', False)
        # Obtaining the member 'dtype' of a type (line 212)
        dtype_377911 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 212, 51), self_377910, 'dtype')
        keyword_377912 = dtype_377911
        kwargs_377913 = {'dtype': keyword_377912}
        # Getting the type of 'lil_matrix' (line 212)
        lil_matrix_377902 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 14), 'lil_matrix', False)
        # Calling lil_matrix(args, kwargs) (line 212)
        lil_matrix_call_result_377914 = invoke(stypy.reporting.localization.Localization(__file__, 212, 14), lil_matrix_377902, *[tuple_377903], **kwargs_377913)
        
        # Assigning a type to the variable 'new' (line 212)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 212, 8), 'new', lil_matrix_call_result_377914)
        
        # Assigning a Subscript to a Subscript (line 213):
        
        # Assigning a Subscript to a Subscript (line 213):
        
        # Obtaining the type of the subscript
        # Getting the type of 'i' (line 213)
        i_377915 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 32), 'i')
        # Getting the type of 'self' (line 213)
        self_377916 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 22), 'self')
        # Obtaining the member 'rows' of a type (line 213)
        rows_377917 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 213, 22), self_377916, 'rows')
        # Obtaining the member '__getitem__' of a type (line 213)
        getitem___377918 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 213, 22), rows_377917, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 213)
        subscript_call_result_377919 = invoke(stypy.reporting.localization.Localization(__file__, 213, 22), getitem___377918, i_377915)
        
        # Getting the type of 'new' (line 213)
        new_377920 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 8), 'new')
        # Obtaining the member 'rows' of a type (line 213)
        rows_377921 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 213, 8), new_377920, 'rows')
        int_377922 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 213, 17), 'int')
        # Storing an element on a container (line 213)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 213, 8), rows_377921, (int_377922, subscript_call_result_377919))
        
        # Assigning a Subscript to a Subscript (line 214):
        
        # Assigning a Subscript to a Subscript (line 214):
        
        # Obtaining the type of the subscript
        # Getting the type of 'i' (line 214)
        i_377923 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 32), 'i')
        # Getting the type of 'self' (line 214)
        self_377924 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 22), 'self')
        # Obtaining the member 'data' of a type (line 214)
        data_377925 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 214, 22), self_377924, 'data')
        # Obtaining the member '__getitem__' of a type (line 214)
        getitem___377926 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 214, 22), data_377925, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 214)
        subscript_call_result_377927 = invoke(stypy.reporting.localization.Localization(__file__, 214, 22), getitem___377926, i_377923)
        
        # Getting the type of 'new' (line 214)
        new_377928 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 8), 'new')
        # Obtaining the member 'data' of a type (line 214)
        data_377929 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 214, 8), new_377928, 'data')
        int_377930 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 214, 17), 'int')
        # Storing an element on a container (line 214)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 214, 8), data_377929, (int_377930, subscript_call_result_377927))
        # Getting the type of 'new' (line 215)
        new_377931 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 215, 15), 'new')
        # Assigning a type to the variable 'stypy_return_type' (line 215)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 215, 8), 'stypy_return_type', new_377931)
        
        # ################# End of 'getrowview(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'getrowview' in the type store
        # Getting the type of 'stypy_return_type' (line 209)
        stypy_return_type_377932 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_377932)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'getrowview'
        return stypy_return_type_377932


    @norecursion
    def getrow(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'getrow'
        module_type_store = module_type_store.open_function_context('getrow', 217, 4, False)
        # Assigning a type to the variable 'self' (line 218)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 218, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        lil_matrix.getrow.__dict__.__setitem__('stypy_localization', localization)
        lil_matrix.getrow.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        lil_matrix.getrow.__dict__.__setitem__('stypy_type_store', module_type_store)
        lil_matrix.getrow.__dict__.__setitem__('stypy_function_name', 'lil_matrix.getrow')
        lil_matrix.getrow.__dict__.__setitem__('stypy_param_names_list', ['i'])
        lil_matrix.getrow.__dict__.__setitem__('stypy_varargs_param_name', None)
        lil_matrix.getrow.__dict__.__setitem__('stypy_kwargs_param_name', None)
        lil_matrix.getrow.__dict__.__setitem__('stypy_call_defaults', defaults)
        lil_matrix.getrow.__dict__.__setitem__('stypy_call_varargs', varargs)
        lil_matrix.getrow.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        lil_matrix.getrow.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'lil_matrix.getrow', ['i'], None, None, defaults, varargs, kwargs)

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

        str_377933 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 219, (-1)), 'str', "Returns a copy of the 'i'th row.\n        ")
        
        # Assigning a Call to a Name (line 220):
        
        # Assigning a Call to a Name (line 220):
        
        # Call to _check_row_bounds(...): (line 220)
        # Processing the call arguments (line 220)
        # Getting the type of 'i' (line 220)
        i_377936 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 220, 35), 'i', False)
        # Processing the call keyword arguments (line 220)
        kwargs_377937 = {}
        # Getting the type of 'self' (line 220)
        self_377934 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 220, 12), 'self', False)
        # Obtaining the member '_check_row_bounds' of a type (line 220)
        _check_row_bounds_377935 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 220, 12), self_377934, '_check_row_bounds')
        # Calling _check_row_bounds(args, kwargs) (line 220)
        _check_row_bounds_call_result_377938 = invoke(stypy.reporting.localization.Localization(__file__, 220, 12), _check_row_bounds_377935, *[i_377936], **kwargs_377937)
        
        # Assigning a type to the variable 'i' (line 220)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 220, 8), 'i', _check_row_bounds_call_result_377938)
        
        # Assigning a Call to a Name (line 221):
        
        # Assigning a Call to a Name (line 221):
        
        # Call to lil_matrix(...): (line 221)
        # Processing the call arguments (line 221)
        
        # Obtaining an instance of the builtin type 'tuple' (line 221)
        tuple_377940 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 221, 26), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 221)
        # Adding element type (line 221)
        int_377941 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 221, 26), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 221, 26), tuple_377940, int_377941)
        # Adding element type (line 221)
        
        # Obtaining the type of the subscript
        int_377942 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 221, 40), 'int')
        # Getting the type of 'self' (line 221)
        self_377943 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 29), 'self', False)
        # Obtaining the member 'shape' of a type (line 221)
        shape_377944 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 221, 29), self_377943, 'shape')
        # Obtaining the member '__getitem__' of a type (line 221)
        getitem___377945 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 221, 29), shape_377944, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 221)
        subscript_call_result_377946 = invoke(stypy.reporting.localization.Localization(__file__, 221, 29), getitem___377945, int_377942)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 221, 26), tuple_377940, subscript_call_result_377946)
        
        # Processing the call keyword arguments (line 221)
        # Getting the type of 'self' (line 221)
        self_377947 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 51), 'self', False)
        # Obtaining the member 'dtype' of a type (line 221)
        dtype_377948 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 221, 51), self_377947, 'dtype')
        keyword_377949 = dtype_377948
        kwargs_377950 = {'dtype': keyword_377949}
        # Getting the type of 'lil_matrix' (line 221)
        lil_matrix_377939 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 14), 'lil_matrix', False)
        # Calling lil_matrix(args, kwargs) (line 221)
        lil_matrix_call_result_377951 = invoke(stypy.reporting.localization.Localization(__file__, 221, 14), lil_matrix_377939, *[tuple_377940], **kwargs_377950)
        
        # Assigning a type to the variable 'new' (line 221)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 221, 8), 'new', lil_matrix_call_result_377951)
        
        # Assigning a Subscript to a Subscript (line 222):
        
        # Assigning a Subscript to a Subscript (line 222):
        
        # Obtaining the type of the subscript
        slice_377952 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 222, 22), None, None, None)
        
        # Obtaining the type of the subscript
        # Getting the type of 'i' (line 222)
        i_377953 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 222, 32), 'i')
        # Getting the type of 'self' (line 222)
        self_377954 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 222, 22), 'self')
        # Obtaining the member 'rows' of a type (line 222)
        rows_377955 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 222, 22), self_377954, 'rows')
        # Obtaining the member '__getitem__' of a type (line 222)
        getitem___377956 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 222, 22), rows_377955, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 222)
        subscript_call_result_377957 = invoke(stypy.reporting.localization.Localization(__file__, 222, 22), getitem___377956, i_377953)
        
        # Obtaining the member '__getitem__' of a type (line 222)
        getitem___377958 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 222, 22), subscript_call_result_377957, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 222)
        subscript_call_result_377959 = invoke(stypy.reporting.localization.Localization(__file__, 222, 22), getitem___377958, slice_377952)
        
        # Getting the type of 'new' (line 222)
        new_377960 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 222, 8), 'new')
        # Obtaining the member 'rows' of a type (line 222)
        rows_377961 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 222, 8), new_377960, 'rows')
        int_377962 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 222, 17), 'int')
        # Storing an element on a container (line 222)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 222, 8), rows_377961, (int_377962, subscript_call_result_377959))
        
        # Assigning a Subscript to a Subscript (line 223):
        
        # Assigning a Subscript to a Subscript (line 223):
        
        # Obtaining the type of the subscript
        slice_377963 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 223, 22), None, None, None)
        
        # Obtaining the type of the subscript
        # Getting the type of 'i' (line 223)
        i_377964 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 32), 'i')
        # Getting the type of 'self' (line 223)
        self_377965 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 22), 'self')
        # Obtaining the member 'data' of a type (line 223)
        data_377966 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 223, 22), self_377965, 'data')
        # Obtaining the member '__getitem__' of a type (line 223)
        getitem___377967 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 223, 22), data_377966, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 223)
        subscript_call_result_377968 = invoke(stypy.reporting.localization.Localization(__file__, 223, 22), getitem___377967, i_377964)
        
        # Obtaining the member '__getitem__' of a type (line 223)
        getitem___377969 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 223, 22), subscript_call_result_377968, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 223)
        subscript_call_result_377970 = invoke(stypy.reporting.localization.Localization(__file__, 223, 22), getitem___377969, slice_377963)
        
        # Getting the type of 'new' (line 223)
        new_377971 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 8), 'new')
        # Obtaining the member 'data' of a type (line 223)
        data_377972 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 223, 8), new_377971, 'data')
        int_377973 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 223, 17), 'int')
        # Storing an element on a container (line 223)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 223, 8), data_377972, (int_377973, subscript_call_result_377970))
        # Getting the type of 'new' (line 224)
        new_377974 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 224, 15), 'new')
        # Assigning a type to the variable 'stypy_return_type' (line 224)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 224, 8), 'stypy_return_type', new_377974)
        
        # ################# End of 'getrow(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'getrow' in the type store
        # Getting the type of 'stypy_return_type' (line 217)
        stypy_return_type_377975 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_377975)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'getrow'
        return stypy_return_type_377975


    @norecursion
    def _check_row_bounds(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_check_row_bounds'
        module_type_store = module_type_store.open_function_context('_check_row_bounds', 226, 4, False)
        # Assigning a type to the variable 'self' (line 227)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 227, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        lil_matrix._check_row_bounds.__dict__.__setitem__('stypy_localization', localization)
        lil_matrix._check_row_bounds.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        lil_matrix._check_row_bounds.__dict__.__setitem__('stypy_type_store', module_type_store)
        lil_matrix._check_row_bounds.__dict__.__setitem__('stypy_function_name', 'lil_matrix._check_row_bounds')
        lil_matrix._check_row_bounds.__dict__.__setitem__('stypy_param_names_list', ['i'])
        lil_matrix._check_row_bounds.__dict__.__setitem__('stypy_varargs_param_name', None)
        lil_matrix._check_row_bounds.__dict__.__setitem__('stypy_kwargs_param_name', None)
        lil_matrix._check_row_bounds.__dict__.__setitem__('stypy_call_defaults', defaults)
        lil_matrix._check_row_bounds.__dict__.__setitem__('stypy_call_varargs', varargs)
        lil_matrix._check_row_bounds.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        lil_matrix._check_row_bounds.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'lil_matrix._check_row_bounds', ['i'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_check_row_bounds', localization, ['i'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_check_row_bounds(...)' code ##################

        
        
        # Getting the type of 'i' (line 227)
        i_377976 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 11), 'i')
        int_377977 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 227, 15), 'int')
        # Applying the binary operator '<' (line 227)
        result_lt_377978 = python_operator(stypy.reporting.localization.Localization(__file__, 227, 11), '<', i_377976, int_377977)
        
        # Testing the type of an if condition (line 227)
        if_condition_377979 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 227, 8), result_lt_377978)
        # Assigning a type to the variable 'if_condition_377979' (line 227)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 227, 8), 'if_condition_377979', if_condition_377979)
        # SSA begins for if statement (line 227)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Getting the type of 'i' (line 228)
        i_377980 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 12), 'i')
        
        # Obtaining the type of the subscript
        int_377981 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 228, 28), 'int')
        # Getting the type of 'self' (line 228)
        self_377982 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 17), 'self')
        # Obtaining the member 'shape' of a type (line 228)
        shape_377983 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 228, 17), self_377982, 'shape')
        # Obtaining the member '__getitem__' of a type (line 228)
        getitem___377984 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 228, 17), shape_377983, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 228)
        subscript_call_result_377985 = invoke(stypy.reporting.localization.Localization(__file__, 228, 17), getitem___377984, int_377981)
        
        # Applying the binary operator '+=' (line 228)
        result_iadd_377986 = python_operator(stypy.reporting.localization.Localization(__file__, 228, 12), '+=', i_377980, subscript_call_result_377985)
        # Assigning a type to the variable 'i' (line 228)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 228, 12), 'i', result_iadd_377986)
        
        # SSA join for if statement (line 227)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Evaluating a boolean operation
        
        # Getting the type of 'i' (line 229)
        i_377987 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 229, 11), 'i')
        int_377988 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 229, 15), 'int')
        # Applying the binary operator '<' (line 229)
        result_lt_377989 = python_operator(stypy.reporting.localization.Localization(__file__, 229, 11), '<', i_377987, int_377988)
        
        
        # Getting the type of 'i' (line 229)
        i_377990 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 229, 20), 'i')
        
        # Obtaining the type of the subscript
        int_377991 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 229, 36), 'int')
        # Getting the type of 'self' (line 229)
        self_377992 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 229, 25), 'self')
        # Obtaining the member 'shape' of a type (line 229)
        shape_377993 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 229, 25), self_377992, 'shape')
        # Obtaining the member '__getitem__' of a type (line 229)
        getitem___377994 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 229, 25), shape_377993, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 229)
        subscript_call_result_377995 = invoke(stypy.reporting.localization.Localization(__file__, 229, 25), getitem___377994, int_377991)
        
        # Applying the binary operator '>=' (line 229)
        result_ge_377996 = python_operator(stypy.reporting.localization.Localization(__file__, 229, 20), '>=', i_377990, subscript_call_result_377995)
        
        # Applying the binary operator 'or' (line 229)
        result_or_keyword_377997 = python_operator(stypy.reporting.localization.Localization(__file__, 229, 11), 'or', result_lt_377989, result_ge_377996)
        
        # Testing the type of an if condition (line 229)
        if_condition_377998 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 229, 8), result_or_keyword_377997)
        # Assigning a type to the variable 'if_condition_377998' (line 229)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 229, 8), 'if_condition_377998', if_condition_377998)
        # SSA begins for if statement (line 229)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to IndexError(...): (line 230)
        # Processing the call arguments (line 230)
        str_378000 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 230, 29), 'str', 'row index out of bounds')
        # Processing the call keyword arguments (line 230)
        kwargs_378001 = {}
        # Getting the type of 'IndexError' (line 230)
        IndexError_377999 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 18), 'IndexError', False)
        # Calling IndexError(args, kwargs) (line 230)
        IndexError_call_result_378002 = invoke(stypy.reporting.localization.Localization(__file__, 230, 18), IndexError_377999, *[str_378000], **kwargs_378001)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 230, 12), IndexError_call_result_378002, 'raise parameter', BaseException)
        # SSA join for if statement (line 229)
        module_type_store = module_type_store.join_ssa_context()
        
        # Getting the type of 'i' (line 231)
        i_378003 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 231, 15), 'i')
        # Assigning a type to the variable 'stypy_return_type' (line 231)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 231, 8), 'stypy_return_type', i_378003)
        
        # ################# End of '_check_row_bounds(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_check_row_bounds' in the type store
        # Getting the type of 'stypy_return_type' (line 226)
        stypy_return_type_378004 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_378004)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_check_row_bounds'
        return stypy_return_type_378004


    @norecursion
    def _check_col_bounds(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_check_col_bounds'
        module_type_store = module_type_store.open_function_context('_check_col_bounds', 233, 4, False)
        # Assigning a type to the variable 'self' (line 234)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 234, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        lil_matrix._check_col_bounds.__dict__.__setitem__('stypy_localization', localization)
        lil_matrix._check_col_bounds.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        lil_matrix._check_col_bounds.__dict__.__setitem__('stypy_type_store', module_type_store)
        lil_matrix._check_col_bounds.__dict__.__setitem__('stypy_function_name', 'lil_matrix._check_col_bounds')
        lil_matrix._check_col_bounds.__dict__.__setitem__('stypy_param_names_list', ['j'])
        lil_matrix._check_col_bounds.__dict__.__setitem__('stypy_varargs_param_name', None)
        lil_matrix._check_col_bounds.__dict__.__setitem__('stypy_kwargs_param_name', None)
        lil_matrix._check_col_bounds.__dict__.__setitem__('stypy_call_defaults', defaults)
        lil_matrix._check_col_bounds.__dict__.__setitem__('stypy_call_varargs', varargs)
        lil_matrix._check_col_bounds.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        lil_matrix._check_col_bounds.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'lil_matrix._check_col_bounds', ['j'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_check_col_bounds', localization, ['j'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_check_col_bounds(...)' code ##################

        
        
        # Getting the type of 'j' (line 234)
        j_378005 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 234, 11), 'j')
        int_378006 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 234, 15), 'int')
        # Applying the binary operator '<' (line 234)
        result_lt_378007 = python_operator(stypy.reporting.localization.Localization(__file__, 234, 11), '<', j_378005, int_378006)
        
        # Testing the type of an if condition (line 234)
        if_condition_378008 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 234, 8), result_lt_378007)
        # Assigning a type to the variable 'if_condition_378008' (line 234)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 234, 8), 'if_condition_378008', if_condition_378008)
        # SSA begins for if statement (line 234)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Getting the type of 'j' (line 235)
        j_378009 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 12), 'j')
        
        # Obtaining the type of the subscript
        int_378010 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 235, 28), 'int')
        # Getting the type of 'self' (line 235)
        self_378011 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 17), 'self')
        # Obtaining the member 'shape' of a type (line 235)
        shape_378012 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 235, 17), self_378011, 'shape')
        # Obtaining the member '__getitem__' of a type (line 235)
        getitem___378013 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 235, 17), shape_378012, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 235)
        subscript_call_result_378014 = invoke(stypy.reporting.localization.Localization(__file__, 235, 17), getitem___378013, int_378010)
        
        # Applying the binary operator '+=' (line 235)
        result_iadd_378015 = python_operator(stypy.reporting.localization.Localization(__file__, 235, 12), '+=', j_378009, subscript_call_result_378014)
        # Assigning a type to the variable 'j' (line 235)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 235, 12), 'j', result_iadd_378015)
        
        # SSA join for if statement (line 234)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Evaluating a boolean operation
        
        # Getting the type of 'j' (line 236)
        j_378016 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 236, 11), 'j')
        int_378017 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 236, 15), 'int')
        # Applying the binary operator '<' (line 236)
        result_lt_378018 = python_operator(stypy.reporting.localization.Localization(__file__, 236, 11), '<', j_378016, int_378017)
        
        
        # Getting the type of 'j' (line 236)
        j_378019 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 236, 20), 'j')
        
        # Obtaining the type of the subscript
        int_378020 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 236, 36), 'int')
        # Getting the type of 'self' (line 236)
        self_378021 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 236, 25), 'self')
        # Obtaining the member 'shape' of a type (line 236)
        shape_378022 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 236, 25), self_378021, 'shape')
        # Obtaining the member '__getitem__' of a type (line 236)
        getitem___378023 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 236, 25), shape_378022, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 236)
        subscript_call_result_378024 = invoke(stypy.reporting.localization.Localization(__file__, 236, 25), getitem___378023, int_378020)
        
        # Applying the binary operator '>=' (line 236)
        result_ge_378025 = python_operator(stypy.reporting.localization.Localization(__file__, 236, 20), '>=', j_378019, subscript_call_result_378024)
        
        # Applying the binary operator 'or' (line 236)
        result_or_keyword_378026 = python_operator(stypy.reporting.localization.Localization(__file__, 236, 11), 'or', result_lt_378018, result_ge_378025)
        
        # Testing the type of an if condition (line 236)
        if_condition_378027 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 236, 8), result_or_keyword_378026)
        # Assigning a type to the variable 'if_condition_378027' (line 236)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 236, 8), 'if_condition_378027', if_condition_378027)
        # SSA begins for if statement (line 236)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to IndexError(...): (line 237)
        # Processing the call arguments (line 237)
        str_378029 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 237, 29), 'str', 'column index out of bounds')
        # Processing the call keyword arguments (line 237)
        kwargs_378030 = {}
        # Getting the type of 'IndexError' (line 237)
        IndexError_378028 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 237, 18), 'IndexError', False)
        # Calling IndexError(args, kwargs) (line 237)
        IndexError_call_result_378031 = invoke(stypy.reporting.localization.Localization(__file__, 237, 18), IndexError_378028, *[str_378029], **kwargs_378030)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 237, 12), IndexError_call_result_378031, 'raise parameter', BaseException)
        # SSA join for if statement (line 236)
        module_type_store = module_type_store.join_ssa_context()
        
        # Getting the type of 'j' (line 238)
        j_378032 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 238, 15), 'j')
        # Assigning a type to the variable 'stypy_return_type' (line 238)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 238, 8), 'stypy_return_type', j_378032)
        
        # ################# End of '_check_col_bounds(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_check_col_bounds' in the type store
        # Getting the type of 'stypy_return_type' (line 233)
        stypy_return_type_378033 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 233, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_378033)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_check_col_bounds'
        return stypy_return_type_378033


    @norecursion
    def __getitem__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__getitem__'
        module_type_store = module_type_store.open_function_context('__getitem__', 240, 4, False)
        # Assigning a type to the variable 'self' (line 241)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 241, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        lil_matrix.__getitem__.__dict__.__setitem__('stypy_localization', localization)
        lil_matrix.__getitem__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        lil_matrix.__getitem__.__dict__.__setitem__('stypy_type_store', module_type_store)
        lil_matrix.__getitem__.__dict__.__setitem__('stypy_function_name', 'lil_matrix.__getitem__')
        lil_matrix.__getitem__.__dict__.__setitem__('stypy_param_names_list', ['index'])
        lil_matrix.__getitem__.__dict__.__setitem__('stypy_varargs_param_name', None)
        lil_matrix.__getitem__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        lil_matrix.__getitem__.__dict__.__setitem__('stypy_call_defaults', defaults)
        lil_matrix.__getitem__.__dict__.__setitem__('stypy_call_varargs', varargs)
        lil_matrix.__getitem__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        lil_matrix.__getitem__.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'lil_matrix.__getitem__', ['index'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__getitem__', localization, ['index'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__getitem__(...)' code ##################

        str_378034 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 244, (-1)), 'str', 'Return the element(s) index=(i, j), where j may be a slice.\n        This always returns a copy for consistency, since slices into\n        Python lists return copies.\n        ')
        
        
        # Evaluating a boolean operation
        
        # Call to isinstance(...): (line 247)
        # Processing the call arguments (line 247)
        # Getting the type of 'index' (line 247)
        index_378036 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 247, 22), 'index', False)
        # Getting the type of 'tuple' (line 247)
        tuple_378037 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 247, 29), 'tuple', False)
        # Processing the call keyword arguments (line 247)
        kwargs_378038 = {}
        # Getting the type of 'isinstance' (line 247)
        isinstance_378035 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 247, 11), 'isinstance', False)
        # Calling isinstance(args, kwargs) (line 247)
        isinstance_call_result_378039 = invoke(stypy.reporting.localization.Localization(__file__, 247, 11), isinstance_378035, *[index_378036, tuple_378037], **kwargs_378038)
        
        
        
        # Call to len(...): (line 247)
        # Processing the call arguments (line 247)
        # Getting the type of 'index' (line 247)
        index_378041 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 247, 44), 'index', False)
        # Processing the call keyword arguments (line 247)
        kwargs_378042 = {}
        # Getting the type of 'len' (line 247)
        len_378040 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 247, 40), 'len', False)
        # Calling len(args, kwargs) (line 247)
        len_call_result_378043 = invoke(stypy.reporting.localization.Localization(__file__, 247, 40), len_378040, *[index_378041], **kwargs_378042)
        
        int_378044 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 247, 54), 'int')
        # Applying the binary operator '==' (line 247)
        result_eq_378045 = python_operator(stypy.reporting.localization.Localization(__file__, 247, 40), '==', len_call_result_378043, int_378044)
        
        # Applying the binary operator 'and' (line 247)
        result_and_keyword_378046 = python_operator(stypy.reporting.localization.Localization(__file__, 247, 11), 'and', isinstance_call_result_378039, result_eq_378045)
        
        # Testing the type of an if condition (line 247)
        if_condition_378047 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 247, 8), result_and_keyword_378046)
        # Assigning a type to the variable 'if_condition_378047' (line 247)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 247, 8), 'if_condition_378047', if_condition_378047)
        # SSA begins for if statement (line 247)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Name to a Tuple (line 248):
        
        # Assigning a Subscript to a Name (line 248):
        
        # Obtaining the type of the subscript
        int_378048 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 248, 12), 'int')
        # Getting the type of 'index' (line 248)
        index_378049 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 248, 19), 'index')
        # Obtaining the member '__getitem__' of a type (line 248)
        getitem___378050 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 248, 12), index_378049, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 248)
        subscript_call_result_378051 = invoke(stypy.reporting.localization.Localization(__file__, 248, 12), getitem___378050, int_378048)
        
        # Assigning a type to the variable 'tuple_var_assignment_377452' (line 248)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 248, 12), 'tuple_var_assignment_377452', subscript_call_result_378051)
        
        # Assigning a Subscript to a Name (line 248):
        
        # Obtaining the type of the subscript
        int_378052 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 248, 12), 'int')
        # Getting the type of 'index' (line 248)
        index_378053 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 248, 19), 'index')
        # Obtaining the member '__getitem__' of a type (line 248)
        getitem___378054 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 248, 12), index_378053, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 248)
        subscript_call_result_378055 = invoke(stypy.reporting.localization.Localization(__file__, 248, 12), getitem___378054, int_378052)
        
        # Assigning a type to the variable 'tuple_var_assignment_377453' (line 248)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 248, 12), 'tuple_var_assignment_377453', subscript_call_result_378055)
        
        # Assigning a Name to a Name (line 248):
        # Getting the type of 'tuple_var_assignment_377452' (line 248)
        tuple_var_assignment_377452_378056 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 248, 12), 'tuple_var_assignment_377452')
        # Assigning a type to the variable 'i' (line 248)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 248, 12), 'i', tuple_var_assignment_377452_378056)
        
        # Assigning a Name to a Name (line 248):
        # Getting the type of 'tuple_var_assignment_377453' (line 248)
        tuple_var_assignment_377453_378057 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 248, 12), 'tuple_var_assignment_377453')
        # Assigning a type to the variable 'j' (line 248)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 248, 15), 'j', tuple_var_assignment_377453_378057)
        
        
        # Evaluating a boolean operation
        
        # Evaluating a boolean operation
        
        # Call to isinstance(...): (line 252)
        # Processing the call arguments (line 252)
        # Getting the type of 'i' (line 252)
        i_378059 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 252, 28), 'i', False)
        # Getting the type of 'int' (line 252)
        int_378060 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 252, 31), 'int', False)
        # Processing the call keyword arguments (line 252)
        kwargs_378061 = {}
        # Getting the type of 'isinstance' (line 252)
        isinstance_378058 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 252, 17), 'isinstance', False)
        # Calling isinstance(args, kwargs) (line 252)
        isinstance_call_result_378062 = invoke(stypy.reporting.localization.Localization(__file__, 252, 17), isinstance_378058, *[i_378059, int_378060], **kwargs_378061)
        
        
        # Call to isinstance(...): (line 252)
        # Processing the call arguments (line 252)
        # Getting the type of 'i' (line 252)
        i_378064 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 252, 50), 'i', False)
        # Getting the type of 'np' (line 252)
        np_378065 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 252, 53), 'np', False)
        # Obtaining the member 'integer' of a type (line 252)
        integer_378066 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 252, 53), np_378065, 'integer')
        # Processing the call keyword arguments (line 252)
        kwargs_378067 = {}
        # Getting the type of 'isinstance' (line 252)
        isinstance_378063 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 252, 39), 'isinstance', False)
        # Calling isinstance(args, kwargs) (line 252)
        isinstance_call_result_378068 = invoke(stypy.reporting.localization.Localization(__file__, 252, 39), isinstance_378063, *[i_378064, integer_378066], **kwargs_378067)
        
        # Applying the binary operator 'or' (line 252)
        result_or_keyword_378069 = python_operator(stypy.reporting.localization.Localization(__file__, 252, 17), 'or', isinstance_call_result_378062, isinstance_call_result_378068)
        
        
        # Evaluating a boolean operation
        
        # Call to isinstance(...): (line 253)
        # Processing the call arguments (line 253)
        # Getting the type of 'j' (line 253)
        j_378071 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 253, 32), 'j', False)
        # Getting the type of 'int' (line 253)
        int_378072 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 253, 35), 'int', False)
        # Processing the call keyword arguments (line 253)
        kwargs_378073 = {}
        # Getting the type of 'isinstance' (line 253)
        isinstance_378070 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 253, 21), 'isinstance', False)
        # Calling isinstance(args, kwargs) (line 253)
        isinstance_call_result_378074 = invoke(stypy.reporting.localization.Localization(__file__, 253, 21), isinstance_378070, *[j_378071, int_378072], **kwargs_378073)
        
        
        # Call to isinstance(...): (line 253)
        # Processing the call arguments (line 253)
        # Getting the type of 'j' (line 253)
        j_378076 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 253, 54), 'j', False)
        # Getting the type of 'np' (line 253)
        np_378077 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 253, 57), 'np', False)
        # Obtaining the member 'integer' of a type (line 253)
        integer_378078 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 253, 57), np_378077, 'integer')
        # Processing the call keyword arguments (line 253)
        kwargs_378079 = {}
        # Getting the type of 'isinstance' (line 253)
        isinstance_378075 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 253, 43), 'isinstance', False)
        # Calling isinstance(args, kwargs) (line 253)
        isinstance_call_result_378080 = invoke(stypy.reporting.localization.Localization(__file__, 253, 43), isinstance_378075, *[j_378076, integer_378078], **kwargs_378079)
        
        # Applying the binary operator 'or' (line 253)
        result_or_keyword_378081 = python_operator(stypy.reporting.localization.Localization(__file__, 253, 21), 'or', isinstance_call_result_378074, isinstance_call_result_378080)
        
        # Applying the binary operator 'and' (line 252)
        result_and_keyword_378082 = python_operator(stypy.reporting.localization.Localization(__file__, 252, 16), 'and', result_or_keyword_378069, result_or_keyword_378081)
        
        # Testing the type of an if condition (line 252)
        if_condition_378083 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 252, 12), result_and_keyword_378082)
        # Assigning a type to the variable 'if_condition_378083' (line 252)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 252, 12), 'if_condition_378083', if_condition_378083)
        # SSA begins for if statement (line 252)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 254):
        
        # Assigning a Call to a Name (line 254):
        
        # Call to lil_get1(...): (line 254)
        # Processing the call arguments (line 254)
        
        # Obtaining the type of the subscript
        int_378086 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 254, 54), 'int')
        # Getting the type of 'self' (line 254)
        self_378087 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 254, 43), 'self', False)
        # Obtaining the member 'shape' of a type (line 254)
        shape_378088 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 254, 43), self_378087, 'shape')
        # Obtaining the member '__getitem__' of a type (line 254)
        getitem___378089 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 254, 43), shape_378088, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 254)
        subscript_call_result_378090 = invoke(stypy.reporting.localization.Localization(__file__, 254, 43), getitem___378089, int_378086)
        
        
        # Obtaining the type of the subscript
        int_378091 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 254, 69), 'int')
        # Getting the type of 'self' (line 254)
        self_378092 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 254, 58), 'self', False)
        # Obtaining the member 'shape' of a type (line 254)
        shape_378093 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 254, 58), self_378092, 'shape')
        # Obtaining the member '__getitem__' of a type (line 254)
        getitem___378094 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 254, 58), shape_378093, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 254)
        subscript_call_result_378095 = invoke(stypy.reporting.localization.Localization(__file__, 254, 58), getitem___378094, int_378091)
        
        # Getting the type of 'self' (line 255)
        self_378096 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 255, 43), 'self', False)
        # Obtaining the member 'rows' of a type (line 255)
        rows_378097 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 255, 43), self_378096, 'rows')
        # Getting the type of 'self' (line 255)
        self_378098 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 255, 54), 'self', False)
        # Obtaining the member 'data' of a type (line 255)
        data_378099 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 255, 54), self_378098, 'data')
        # Getting the type of 'i' (line 256)
        i_378100 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 256, 43), 'i', False)
        # Getting the type of 'j' (line 256)
        j_378101 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 256, 46), 'j', False)
        # Processing the call keyword arguments (line 254)
        kwargs_378102 = {}
        # Getting the type of '_csparsetools' (line 254)
        _csparsetools_378084 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 254, 20), '_csparsetools', False)
        # Obtaining the member 'lil_get1' of a type (line 254)
        lil_get1_378085 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 254, 20), _csparsetools_378084, 'lil_get1')
        # Calling lil_get1(args, kwargs) (line 254)
        lil_get1_call_result_378103 = invoke(stypy.reporting.localization.Localization(__file__, 254, 20), lil_get1_378085, *[subscript_call_result_378090, subscript_call_result_378095, rows_378097, data_378099, i_378100, j_378101], **kwargs_378102)
        
        # Assigning a type to the variable 'v' (line 254)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 254, 16), 'v', lil_get1_call_result_378103)
        
        # Call to type(...): (line 257)
        # Processing the call arguments (line 257)
        # Getting the type of 'v' (line 257)
        v_378107 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 257, 39), 'v', False)
        # Processing the call keyword arguments (line 257)
        kwargs_378108 = {}
        # Getting the type of 'self' (line 257)
        self_378104 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 257, 23), 'self', False)
        # Obtaining the member 'dtype' of a type (line 257)
        dtype_378105 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 257, 23), self_378104, 'dtype')
        # Obtaining the member 'type' of a type (line 257)
        type_378106 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 257, 23), dtype_378105, 'type')
        # Calling type(args, kwargs) (line 257)
        type_call_result_378109 = invoke(stypy.reporting.localization.Localization(__file__, 257, 23), type_378106, *[v_378107], **kwargs_378108)
        
        # Assigning a type to the variable 'stypy_return_type' (line 257)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 257, 16), 'stypy_return_type', type_call_result_378109)
        # SSA join for if statement (line 252)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 247)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Tuple (line 260):
        
        # Assigning a Subscript to a Name (line 260):
        
        # Obtaining the type of the subscript
        int_378110 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 260, 8), 'int')
        
        # Call to _unpack_index(...): (line 260)
        # Processing the call arguments (line 260)
        # Getting the type of 'index' (line 260)
        index_378113 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 260, 34), 'index', False)
        # Processing the call keyword arguments (line 260)
        kwargs_378114 = {}
        # Getting the type of 'self' (line 260)
        self_378111 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 260, 15), 'self', False)
        # Obtaining the member '_unpack_index' of a type (line 260)
        _unpack_index_378112 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 260, 15), self_378111, '_unpack_index')
        # Calling _unpack_index(args, kwargs) (line 260)
        _unpack_index_call_result_378115 = invoke(stypy.reporting.localization.Localization(__file__, 260, 15), _unpack_index_378112, *[index_378113], **kwargs_378114)
        
        # Obtaining the member '__getitem__' of a type (line 260)
        getitem___378116 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 260, 8), _unpack_index_call_result_378115, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 260)
        subscript_call_result_378117 = invoke(stypy.reporting.localization.Localization(__file__, 260, 8), getitem___378116, int_378110)
        
        # Assigning a type to the variable 'tuple_var_assignment_377454' (line 260)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 260, 8), 'tuple_var_assignment_377454', subscript_call_result_378117)
        
        # Assigning a Subscript to a Name (line 260):
        
        # Obtaining the type of the subscript
        int_378118 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 260, 8), 'int')
        
        # Call to _unpack_index(...): (line 260)
        # Processing the call arguments (line 260)
        # Getting the type of 'index' (line 260)
        index_378121 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 260, 34), 'index', False)
        # Processing the call keyword arguments (line 260)
        kwargs_378122 = {}
        # Getting the type of 'self' (line 260)
        self_378119 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 260, 15), 'self', False)
        # Obtaining the member '_unpack_index' of a type (line 260)
        _unpack_index_378120 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 260, 15), self_378119, '_unpack_index')
        # Calling _unpack_index(args, kwargs) (line 260)
        _unpack_index_call_result_378123 = invoke(stypy.reporting.localization.Localization(__file__, 260, 15), _unpack_index_378120, *[index_378121], **kwargs_378122)
        
        # Obtaining the member '__getitem__' of a type (line 260)
        getitem___378124 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 260, 8), _unpack_index_call_result_378123, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 260)
        subscript_call_result_378125 = invoke(stypy.reporting.localization.Localization(__file__, 260, 8), getitem___378124, int_378118)
        
        # Assigning a type to the variable 'tuple_var_assignment_377455' (line 260)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 260, 8), 'tuple_var_assignment_377455', subscript_call_result_378125)
        
        # Assigning a Name to a Name (line 260):
        # Getting the type of 'tuple_var_assignment_377454' (line 260)
        tuple_var_assignment_377454_378126 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 260, 8), 'tuple_var_assignment_377454')
        # Assigning a type to the variable 'i' (line 260)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 260, 8), 'i', tuple_var_assignment_377454_378126)
        
        # Assigning a Name to a Name (line 260):
        # Getting the type of 'tuple_var_assignment_377455' (line 260)
        tuple_var_assignment_377455_378127 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 260, 8), 'tuple_var_assignment_377455')
        # Assigning a type to the variable 'j' (line 260)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 260, 11), 'j', tuple_var_assignment_377455_378127)
        
        # Assigning a Call to a Name (line 263):
        
        # Assigning a Call to a Name (line 263):
        
        # Call to isintlike(...): (line 263)
        # Processing the call arguments (line 263)
        # Getting the type of 'i' (line 263)
        i_378129 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 263, 30), 'i', False)
        # Processing the call keyword arguments (line 263)
        kwargs_378130 = {}
        # Getting the type of 'isintlike' (line 263)
        isintlike_378128 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 263, 20), 'isintlike', False)
        # Calling isintlike(args, kwargs) (line 263)
        isintlike_call_result_378131 = invoke(stypy.reporting.localization.Localization(__file__, 263, 20), isintlike_378128, *[i_378129], **kwargs_378130)
        
        # Assigning a type to the variable 'i_intlike' (line 263)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 263, 8), 'i_intlike', isintlike_call_result_378131)
        
        # Assigning a Call to a Name (line 264):
        
        # Assigning a Call to a Name (line 264):
        
        # Call to isintlike(...): (line 264)
        # Processing the call arguments (line 264)
        # Getting the type of 'j' (line 264)
        j_378133 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 264, 30), 'j', False)
        # Processing the call keyword arguments (line 264)
        kwargs_378134 = {}
        # Getting the type of 'isintlike' (line 264)
        isintlike_378132 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 264, 20), 'isintlike', False)
        # Calling isintlike(args, kwargs) (line 264)
        isintlike_call_result_378135 = invoke(stypy.reporting.localization.Localization(__file__, 264, 20), isintlike_378132, *[j_378133], **kwargs_378134)
        
        # Assigning a type to the variable 'j_intlike' (line 264)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 264, 8), 'j_intlike', isintlike_call_result_378135)
        
        
        # Evaluating a boolean operation
        # Getting the type of 'i_intlike' (line 266)
        i_intlike_378136 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 266, 11), 'i_intlike')
        # Getting the type of 'j_intlike' (line 266)
        j_intlike_378137 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 266, 25), 'j_intlike')
        # Applying the binary operator 'and' (line 266)
        result_and_keyword_378138 = python_operator(stypy.reporting.localization.Localization(__file__, 266, 11), 'and', i_intlike_378136, j_intlike_378137)
        
        # Testing the type of an if condition (line 266)
        if_condition_378139 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 266, 8), result_and_keyword_378138)
        # Assigning a type to the variable 'if_condition_378139' (line 266)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 266, 8), 'if_condition_378139', if_condition_378139)
        # SSA begins for if statement (line 266)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 267):
        
        # Assigning a Call to a Name (line 267):
        
        # Call to lil_get1(...): (line 267)
        # Processing the call arguments (line 267)
        
        # Obtaining the type of the subscript
        int_378142 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 267, 50), 'int')
        # Getting the type of 'self' (line 267)
        self_378143 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 267, 39), 'self', False)
        # Obtaining the member 'shape' of a type (line 267)
        shape_378144 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 267, 39), self_378143, 'shape')
        # Obtaining the member '__getitem__' of a type (line 267)
        getitem___378145 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 267, 39), shape_378144, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 267)
        subscript_call_result_378146 = invoke(stypy.reporting.localization.Localization(__file__, 267, 39), getitem___378145, int_378142)
        
        
        # Obtaining the type of the subscript
        int_378147 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 267, 65), 'int')
        # Getting the type of 'self' (line 267)
        self_378148 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 267, 54), 'self', False)
        # Obtaining the member 'shape' of a type (line 267)
        shape_378149 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 267, 54), self_378148, 'shape')
        # Obtaining the member '__getitem__' of a type (line 267)
        getitem___378150 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 267, 54), shape_378149, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 267)
        subscript_call_result_378151 = invoke(stypy.reporting.localization.Localization(__file__, 267, 54), getitem___378150, int_378147)
        
        # Getting the type of 'self' (line 268)
        self_378152 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 268, 39), 'self', False)
        # Obtaining the member 'rows' of a type (line 268)
        rows_378153 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 268, 39), self_378152, 'rows')
        # Getting the type of 'self' (line 268)
        self_378154 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 268, 50), 'self', False)
        # Obtaining the member 'data' of a type (line 268)
        data_378155 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 268, 50), self_378154, 'data')
        # Getting the type of 'i' (line 269)
        i_378156 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 269, 39), 'i', False)
        # Getting the type of 'j' (line 269)
        j_378157 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 269, 42), 'j', False)
        # Processing the call keyword arguments (line 267)
        kwargs_378158 = {}
        # Getting the type of '_csparsetools' (line 267)
        _csparsetools_378140 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 267, 16), '_csparsetools', False)
        # Obtaining the member 'lil_get1' of a type (line 267)
        lil_get1_378141 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 267, 16), _csparsetools_378140, 'lil_get1')
        # Calling lil_get1(args, kwargs) (line 267)
        lil_get1_call_result_378159 = invoke(stypy.reporting.localization.Localization(__file__, 267, 16), lil_get1_378141, *[subscript_call_result_378146, subscript_call_result_378151, rows_378153, data_378155, i_378156, j_378157], **kwargs_378158)
        
        # Assigning a type to the variable 'v' (line 267)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 267, 12), 'v', lil_get1_call_result_378159)
        
        # Call to type(...): (line 270)
        # Processing the call arguments (line 270)
        # Getting the type of 'v' (line 270)
        v_378163 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 270, 35), 'v', False)
        # Processing the call keyword arguments (line 270)
        kwargs_378164 = {}
        # Getting the type of 'self' (line 270)
        self_378160 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 270, 19), 'self', False)
        # Obtaining the member 'dtype' of a type (line 270)
        dtype_378161 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 270, 19), self_378160, 'dtype')
        # Obtaining the member 'type' of a type (line 270)
        type_378162 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 270, 19), dtype_378161, 'type')
        # Calling type(args, kwargs) (line 270)
        type_call_result_378165 = invoke(stypy.reporting.localization.Localization(__file__, 270, 19), type_378162, *[v_378163], **kwargs_378164)
        
        # Assigning a type to the variable 'stypy_return_type' (line 270)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 270, 12), 'stypy_return_type', type_call_result_378165)
        # SSA branch for the else part of an if statement (line 266)
        module_type_store.open_ssa_branch('else')
        
        
        # Evaluating a boolean operation
        # Getting the type of 'j_intlike' (line 271)
        j_intlike_378166 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 271, 13), 'j_intlike')
        
        # Call to isinstance(...): (line 271)
        # Processing the call arguments (line 271)
        # Getting the type of 'j' (line 271)
        j_378168 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 271, 37), 'j', False)
        # Getting the type of 'slice' (line 271)
        slice_378169 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 271, 40), 'slice', False)
        # Processing the call keyword arguments (line 271)
        kwargs_378170 = {}
        # Getting the type of 'isinstance' (line 271)
        isinstance_378167 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 271, 26), 'isinstance', False)
        # Calling isinstance(args, kwargs) (line 271)
        isinstance_call_result_378171 = invoke(stypy.reporting.localization.Localization(__file__, 271, 26), isinstance_378167, *[j_378168, slice_378169], **kwargs_378170)
        
        # Applying the binary operator 'or' (line 271)
        result_or_keyword_378172 = python_operator(stypy.reporting.localization.Localization(__file__, 271, 13), 'or', j_intlike_378166, isinstance_call_result_378171)
        
        # Testing the type of an if condition (line 271)
        if_condition_378173 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 271, 13), result_or_keyword_378172)
        # Assigning a type to the variable 'if_condition_378173' (line 271)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 271, 13), 'if_condition_378173', if_condition_378173)
        # SSA begins for if statement (line 271)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Getting the type of 'j_intlike' (line 273)
        j_intlike_378174 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 273, 15), 'j_intlike')
        # Testing the type of an if condition (line 273)
        if_condition_378175 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 273, 12), j_intlike_378174)
        # Assigning a type to the variable 'if_condition_378175' (line 273)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 273, 12), 'if_condition_378175', if_condition_378175)
        # SSA begins for if statement (line 273)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 274):
        
        # Assigning a Call to a Name (line 274):
        
        # Call to _check_col_bounds(...): (line 274)
        # Processing the call arguments (line 274)
        # Getting the type of 'j' (line 274)
        j_378178 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 274, 43), 'j', False)
        # Processing the call keyword arguments (line 274)
        kwargs_378179 = {}
        # Getting the type of 'self' (line 274)
        self_378176 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 274, 20), 'self', False)
        # Obtaining the member '_check_col_bounds' of a type (line 274)
        _check_col_bounds_378177 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 274, 20), self_378176, '_check_col_bounds')
        # Calling _check_col_bounds(args, kwargs) (line 274)
        _check_col_bounds_call_result_378180 = invoke(stypy.reporting.localization.Localization(__file__, 274, 20), _check_col_bounds_378177, *[j_378178], **kwargs_378179)
        
        # Assigning a type to the variable 'j' (line 274)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 274, 16), 'j', _check_col_bounds_call_result_378180)
        
        # Assigning a Call to a Name (line 275):
        
        # Assigning a Call to a Name (line 275):
        
        # Call to slice(...): (line 275)
        # Processing the call arguments (line 275)
        # Getting the type of 'j' (line 275)
        j_378182 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 275, 26), 'j', False)
        # Getting the type of 'j' (line 275)
        j_378183 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 275, 29), 'j', False)
        int_378184 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 275, 31), 'int')
        # Applying the binary operator '+' (line 275)
        result_add_378185 = python_operator(stypy.reporting.localization.Localization(__file__, 275, 29), '+', j_378183, int_378184)
        
        # Processing the call keyword arguments (line 275)
        kwargs_378186 = {}
        # Getting the type of 'slice' (line 275)
        slice_378181 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 275, 20), 'slice', False)
        # Calling slice(args, kwargs) (line 275)
        slice_call_result_378187 = invoke(stypy.reporting.localization.Localization(__file__, 275, 20), slice_378181, *[j_378182, result_add_378185], **kwargs_378186)
        
        # Assigning a type to the variable 'j' (line 275)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 275, 16), 'j', slice_call_result_378187)
        # SSA join for if statement (line 273)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Getting the type of 'i_intlike' (line 277)
        i_intlike_378188 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 277, 15), 'i_intlike')
        # Testing the type of an if condition (line 277)
        if_condition_378189 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 277, 12), i_intlike_378188)
        # Assigning a type to the variable 'if_condition_378189' (line 277)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 277, 12), 'if_condition_378189', if_condition_378189)
        # SSA begins for if statement (line 277)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 278):
        
        # Assigning a Call to a Name (line 278):
        
        # Call to _check_row_bounds(...): (line 278)
        # Processing the call arguments (line 278)
        # Getting the type of 'i' (line 278)
        i_378192 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 278, 43), 'i', False)
        # Processing the call keyword arguments (line 278)
        kwargs_378193 = {}
        # Getting the type of 'self' (line 278)
        self_378190 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 278, 20), 'self', False)
        # Obtaining the member '_check_row_bounds' of a type (line 278)
        _check_row_bounds_378191 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 278, 20), self_378190, '_check_row_bounds')
        # Calling _check_row_bounds(args, kwargs) (line 278)
        _check_row_bounds_call_result_378194 = invoke(stypy.reporting.localization.Localization(__file__, 278, 20), _check_row_bounds_378191, *[i_378192], **kwargs_378193)
        
        # Assigning a type to the variable 'i' (line 278)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 278, 16), 'i', _check_row_bounds_call_result_378194)
        
        # Assigning a Call to a Name (line 279):
        
        # Assigning a Call to a Name (line 279):
        
        # Call to xrange(...): (line 279)
        # Processing the call arguments (line 279)
        # Getting the type of 'i' (line 279)
        i_378196 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 279, 27), 'i', False)
        # Getting the type of 'i' (line 279)
        i_378197 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 279, 30), 'i', False)
        int_378198 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 279, 32), 'int')
        # Applying the binary operator '+' (line 279)
        result_add_378199 = python_operator(stypy.reporting.localization.Localization(__file__, 279, 30), '+', i_378197, int_378198)
        
        # Processing the call keyword arguments (line 279)
        kwargs_378200 = {}
        # Getting the type of 'xrange' (line 279)
        xrange_378195 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 279, 20), 'xrange', False)
        # Calling xrange(args, kwargs) (line 279)
        xrange_call_result_378201 = invoke(stypy.reporting.localization.Localization(__file__, 279, 20), xrange_378195, *[i_378196, result_add_378199], **kwargs_378200)
        
        # Assigning a type to the variable 'i' (line 279)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 279, 16), 'i', xrange_call_result_378201)
        
        # Assigning a Name to a Name (line 280):
        
        # Assigning a Name to a Name (line 280):
        # Getting the type of 'None' (line 280)
        None_378202 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 280, 26), 'None')
        # Assigning a type to the variable 'i_shape' (line 280)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 280, 16), 'i_shape', None_378202)
        # SSA branch for the else part of an if statement (line 277)
        module_type_store.open_ssa_branch('else')
        
        # Type idiom detected: calculating its left and rigth part (line 281)
        # Getting the type of 'slice' (line 281)
        slice_378203 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 281, 31), 'slice')
        # Getting the type of 'i' (line 281)
        i_378204 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 281, 28), 'i')
        
        (may_be_378205, more_types_in_union_378206) = may_be_subtype(slice_378203, i_378204)

        if may_be_378205:

            if more_types_in_union_378206:
                # Runtime conditional SSA (line 281)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            # Assigning a type to the variable 'i' (line 281)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 281, 17), 'i', remove_not_subtype_from_union(i_378204, slice))
            
            # Assigning a Call to a Name (line 282):
            
            # Assigning a Call to a Name (line 282):
            
            # Call to xrange(...): (line 282)
            
            # Call to indices(...): (line 282)
            # Processing the call arguments (line 282)
            
            # Obtaining the type of the subscript
            int_378210 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 282, 49), 'int')
            # Getting the type of 'self' (line 282)
            self_378211 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 282, 38), 'self', False)
            # Obtaining the member 'shape' of a type (line 282)
            shape_378212 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 282, 38), self_378211, 'shape')
            # Obtaining the member '__getitem__' of a type (line 282)
            getitem___378213 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 282, 38), shape_378212, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 282)
            subscript_call_result_378214 = invoke(stypy.reporting.localization.Localization(__file__, 282, 38), getitem___378213, int_378210)
            
            # Processing the call keyword arguments (line 282)
            kwargs_378215 = {}
            # Getting the type of 'i' (line 282)
            i_378208 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 282, 28), 'i', False)
            # Obtaining the member 'indices' of a type (line 282)
            indices_378209 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 282, 28), i_378208, 'indices')
            # Calling indices(args, kwargs) (line 282)
            indices_call_result_378216 = invoke(stypy.reporting.localization.Localization(__file__, 282, 28), indices_378209, *[subscript_call_result_378214], **kwargs_378215)
            
            # Processing the call keyword arguments (line 282)
            kwargs_378217 = {}
            # Getting the type of 'xrange' (line 282)
            xrange_378207 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 282, 20), 'xrange', False)
            # Calling xrange(args, kwargs) (line 282)
            xrange_call_result_378218 = invoke(stypy.reporting.localization.Localization(__file__, 282, 20), xrange_378207, *[indices_call_result_378216], **kwargs_378217)
            
            # Assigning a type to the variable 'i' (line 282)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 282, 16), 'i', xrange_call_result_378218)
            
            # Assigning a Name to a Name (line 283):
            
            # Assigning a Name to a Name (line 283):
            # Getting the type of 'None' (line 283)
            None_378219 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 283, 26), 'None')
            # Assigning a type to the variable 'i_shape' (line 283)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 283, 16), 'i_shape', None_378219)

            if more_types_in_union_378206:
                # Runtime conditional SSA for else branch (line 281)
                module_type_store.open_ssa_branch('idiom else')



        if ((not may_be_378205) or more_types_in_union_378206):
            # Assigning a type to the variable 'i' (line 281)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 281, 17), 'i', remove_subtype_from_union(i_378204, slice))
            
            # Assigning a Call to a Name (line 285):
            
            # Assigning a Call to a Name (line 285):
            
            # Call to atleast_1d(...): (line 285)
            # Processing the call arguments (line 285)
            # Getting the type of 'i' (line 285)
            i_378222 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 285, 34), 'i', False)
            # Processing the call keyword arguments (line 285)
            kwargs_378223 = {}
            # Getting the type of 'np' (line 285)
            np_378220 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 285, 20), 'np', False)
            # Obtaining the member 'atleast_1d' of a type (line 285)
            atleast_1d_378221 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 285, 20), np_378220, 'atleast_1d')
            # Calling atleast_1d(args, kwargs) (line 285)
            atleast_1d_call_result_378224 = invoke(stypy.reporting.localization.Localization(__file__, 285, 20), atleast_1d_378221, *[i_378222], **kwargs_378223)
            
            # Assigning a type to the variable 'i' (line 285)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 285, 16), 'i', atleast_1d_call_result_378224)
            
            # Assigning a Attribute to a Name (line 286):
            
            # Assigning a Attribute to a Name (line 286):
            # Getting the type of 'i' (line 286)
            i_378225 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 286, 26), 'i')
            # Obtaining the member 'shape' of a type (line 286)
            shape_378226 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 286, 26), i_378225, 'shape')
            # Assigning a type to the variable 'i_shape' (line 286)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 286, 16), 'i_shape', shape_378226)

            if (may_be_378205 and more_types_in_union_378206):
                # SSA join for if statement (line 281)
                module_type_store = module_type_store.join_ssa_context()


        
        # SSA join for if statement (line 277)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Evaluating a boolean operation
        
        # Getting the type of 'i_shape' (line 288)
        i_shape_378227 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 288, 15), 'i_shape')
        # Getting the type of 'None' (line 288)
        None_378228 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 288, 26), 'None')
        # Applying the binary operator 'is' (line 288)
        result_is__378229 = python_operator(stypy.reporting.localization.Localization(__file__, 288, 15), 'is', i_shape_378227, None_378228)
        
        
        
        # Call to len(...): (line 288)
        # Processing the call arguments (line 288)
        # Getting the type of 'i_shape' (line 288)
        i_shape_378231 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 288, 38), 'i_shape', False)
        # Processing the call keyword arguments (line 288)
        kwargs_378232 = {}
        # Getting the type of 'len' (line 288)
        len_378230 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 288, 34), 'len', False)
        # Calling len(args, kwargs) (line 288)
        len_call_result_378233 = invoke(stypy.reporting.localization.Localization(__file__, 288, 34), len_378230, *[i_shape_378231], **kwargs_378232)
        
        int_378234 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 288, 50), 'int')
        # Applying the binary operator '==' (line 288)
        result_eq_378235 = python_operator(stypy.reporting.localization.Localization(__file__, 288, 34), '==', len_call_result_378233, int_378234)
        
        # Applying the binary operator 'or' (line 288)
        result_or_keyword_378236 = python_operator(stypy.reporting.localization.Localization(__file__, 288, 15), 'or', result_is__378229, result_eq_378235)
        
        # Testing the type of an if condition (line 288)
        if_condition_378237 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 288, 12), result_or_keyword_378236)
        # Assigning a type to the variable 'if_condition_378237' (line 288)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 288, 12), 'if_condition_378237', if_condition_378237)
        # SSA begins for if statement (line 288)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to _get_row_ranges(...): (line 289)
        # Processing the call arguments (line 289)
        # Getting the type of 'i' (line 289)
        i_378240 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 289, 44), 'i', False)
        # Getting the type of 'j' (line 289)
        j_378241 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 289, 47), 'j', False)
        # Processing the call keyword arguments (line 289)
        kwargs_378242 = {}
        # Getting the type of 'self' (line 289)
        self_378238 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 289, 23), 'self', False)
        # Obtaining the member '_get_row_ranges' of a type (line 289)
        _get_row_ranges_378239 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 289, 23), self_378238, '_get_row_ranges')
        # Calling _get_row_ranges(args, kwargs) (line 289)
        _get_row_ranges_call_result_378243 = invoke(stypy.reporting.localization.Localization(__file__, 289, 23), _get_row_ranges_378239, *[i_378240, j_378241], **kwargs_378242)
        
        # Assigning a type to the variable 'stypy_return_type' (line 289)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 289, 16), 'stypy_return_type', _get_row_ranges_call_result_378243)
        # SSA join for if statement (line 288)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 271)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 266)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Tuple (line 291):
        
        # Assigning a Subscript to a Name (line 291):
        
        # Obtaining the type of the subscript
        int_378244 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 291, 8), 'int')
        
        # Call to _index_to_arrays(...): (line 291)
        # Processing the call arguments (line 291)
        # Getting the type of 'i' (line 291)
        i_378247 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 291, 37), 'i', False)
        # Getting the type of 'j' (line 291)
        j_378248 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 291, 40), 'j', False)
        # Processing the call keyword arguments (line 291)
        kwargs_378249 = {}
        # Getting the type of 'self' (line 291)
        self_378245 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 291, 15), 'self', False)
        # Obtaining the member '_index_to_arrays' of a type (line 291)
        _index_to_arrays_378246 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 291, 15), self_378245, '_index_to_arrays')
        # Calling _index_to_arrays(args, kwargs) (line 291)
        _index_to_arrays_call_result_378250 = invoke(stypy.reporting.localization.Localization(__file__, 291, 15), _index_to_arrays_378246, *[i_378247, j_378248], **kwargs_378249)
        
        # Obtaining the member '__getitem__' of a type (line 291)
        getitem___378251 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 291, 8), _index_to_arrays_call_result_378250, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 291)
        subscript_call_result_378252 = invoke(stypy.reporting.localization.Localization(__file__, 291, 8), getitem___378251, int_378244)
        
        # Assigning a type to the variable 'tuple_var_assignment_377456' (line 291)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 291, 8), 'tuple_var_assignment_377456', subscript_call_result_378252)
        
        # Assigning a Subscript to a Name (line 291):
        
        # Obtaining the type of the subscript
        int_378253 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 291, 8), 'int')
        
        # Call to _index_to_arrays(...): (line 291)
        # Processing the call arguments (line 291)
        # Getting the type of 'i' (line 291)
        i_378256 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 291, 37), 'i', False)
        # Getting the type of 'j' (line 291)
        j_378257 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 291, 40), 'j', False)
        # Processing the call keyword arguments (line 291)
        kwargs_378258 = {}
        # Getting the type of 'self' (line 291)
        self_378254 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 291, 15), 'self', False)
        # Obtaining the member '_index_to_arrays' of a type (line 291)
        _index_to_arrays_378255 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 291, 15), self_378254, '_index_to_arrays')
        # Calling _index_to_arrays(args, kwargs) (line 291)
        _index_to_arrays_call_result_378259 = invoke(stypy.reporting.localization.Localization(__file__, 291, 15), _index_to_arrays_378255, *[i_378256, j_378257], **kwargs_378258)
        
        # Obtaining the member '__getitem__' of a type (line 291)
        getitem___378260 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 291, 8), _index_to_arrays_call_result_378259, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 291)
        subscript_call_result_378261 = invoke(stypy.reporting.localization.Localization(__file__, 291, 8), getitem___378260, int_378253)
        
        # Assigning a type to the variable 'tuple_var_assignment_377457' (line 291)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 291, 8), 'tuple_var_assignment_377457', subscript_call_result_378261)
        
        # Assigning a Name to a Name (line 291):
        # Getting the type of 'tuple_var_assignment_377456' (line 291)
        tuple_var_assignment_377456_378262 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 291, 8), 'tuple_var_assignment_377456')
        # Assigning a type to the variable 'i' (line 291)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 291, 8), 'i', tuple_var_assignment_377456_378262)
        
        # Assigning a Name to a Name (line 291):
        # Getting the type of 'tuple_var_assignment_377457' (line 291)
        tuple_var_assignment_377457_378263 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 291, 8), 'tuple_var_assignment_377457')
        # Assigning a type to the variable 'j' (line 291)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 291, 11), 'j', tuple_var_assignment_377457_378263)
        
        
        # Getting the type of 'i' (line 292)
        i_378264 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 292, 11), 'i')
        # Obtaining the member 'size' of a type (line 292)
        size_378265 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 292, 11), i_378264, 'size')
        int_378266 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 292, 21), 'int')
        # Applying the binary operator '==' (line 292)
        result_eq_378267 = python_operator(stypy.reporting.localization.Localization(__file__, 292, 11), '==', size_378265, int_378266)
        
        # Testing the type of an if condition (line 292)
        if_condition_378268 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 292, 8), result_eq_378267)
        # Assigning a type to the variable 'if_condition_378268' (line 292)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 292, 8), 'if_condition_378268', if_condition_378268)
        # SSA begins for if statement (line 292)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to lil_matrix(...): (line 293)
        # Processing the call arguments (line 293)
        # Getting the type of 'i' (line 293)
        i_378270 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 293, 30), 'i', False)
        # Obtaining the member 'shape' of a type (line 293)
        shape_378271 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 293, 30), i_378270, 'shape')
        # Processing the call keyword arguments (line 293)
        # Getting the type of 'self' (line 293)
        self_378272 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 293, 45), 'self', False)
        # Obtaining the member 'dtype' of a type (line 293)
        dtype_378273 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 293, 45), self_378272, 'dtype')
        keyword_378274 = dtype_378273
        kwargs_378275 = {'dtype': keyword_378274}
        # Getting the type of 'lil_matrix' (line 293)
        lil_matrix_378269 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 293, 19), 'lil_matrix', False)
        # Calling lil_matrix(args, kwargs) (line 293)
        lil_matrix_call_result_378276 = invoke(stypy.reporting.localization.Localization(__file__, 293, 19), lil_matrix_378269, *[shape_378271], **kwargs_378275)
        
        # Assigning a type to the variable 'stypy_return_type' (line 293)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 293, 12), 'stypy_return_type', lil_matrix_call_result_378276)
        # SSA join for if statement (line 292)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Name (line 295):
        
        # Assigning a Call to a Name (line 295):
        
        # Call to lil_matrix(...): (line 295)
        # Processing the call arguments (line 295)
        # Getting the type of 'i' (line 295)
        i_378278 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 295, 25), 'i', False)
        # Obtaining the member 'shape' of a type (line 295)
        shape_378279 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 295, 25), i_378278, 'shape')
        # Processing the call keyword arguments (line 295)
        # Getting the type of 'self' (line 295)
        self_378280 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 295, 40), 'self', False)
        # Obtaining the member 'dtype' of a type (line 295)
        dtype_378281 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 295, 40), self_378280, 'dtype')
        keyword_378282 = dtype_378281
        kwargs_378283 = {'dtype': keyword_378282}
        # Getting the type of 'lil_matrix' (line 295)
        lil_matrix_378277 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 295, 14), 'lil_matrix', False)
        # Calling lil_matrix(args, kwargs) (line 295)
        lil_matrix_call_result_378284 = invoke(stypy.reporting.localization.Localization(__file__, 295, 14), lil_matrix_378277, *[shape_378279], **kwargs_378283)
        
        # Assigning a type to the variable 'new' (line 295)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 295, 8), 'new', lil_matrix_call_result_378284)
        
        # Assigning a Call to a Tuple (line 297):
        
        # Assigning a Subscript to a Name (line 297):
        
        # Obtaining the type of the subscript
        int_378285 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 297, 8), 'int')
        
        # Call to _prepare_index_for_memoryview(...): (line 297)
        # Processing the call arguments (line 297)
        # Getting the type of 'i' (line 297)
        i_378287 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 297, 45), 'i', False)
        # Getting the type of 'j' (line 297)
        j_378288 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 297, 48), 'j', False)
        # Processing the call keyword arguments (line 297)
        kwargs_378289 = {}
        # Getting the type of '_prepare_index_for_memoryview' (line 297)
        _prepare_index_for_memoryview_378286 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 297, 15), '_prepare_index_for_memoryview', False)
        # Calling _prepare_index_for_memoryview(args, kwargs) (line 297)
        _prepare_index_for_memoryview_call_result_378290 = invoke(stypy.reporting.localization.Localization(__file__, 297, 15), _prepare_index_for_memoryview_378286, *[i_378287, j_378288], **kwargs_378289)
        
        # Obtaining the member '__getitem__' of a type (line 297)
        getitem___378291 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 297, 8), _prepare_index_for_memoryview_call_result_378290, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 297)
        subscript_call_result_378292 = invoke(stypy.reporting.localization.Localization(__file__, 297, 8), getitem___378291, int_378285)
        
        # Assigning a type to the variable 'tuple_var_assignment_377458' (line 297)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 297, 8), 'tuple_var_assignment_377458', subscript_call_result_378292)
        
        # Assigning a Subscript to a Name (line 297):
        
        # Obtaining the type of the subscript
        int_378293 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 297, 8), 'int')
        
        # Call to _prepare_index_for_memoryview(...): (line 297)
        # Processing the call arguments (line 297)
        # Getting the type of 'i' (line 297)
        i_378295 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 297, 45), 'i', False)
        # Getting the type of 'j' (line 297)
        j_378296 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 297, 48), 'j', False)
        # Processing the call keyword arguments (line 297)
        kwargs_378297 = {}
        # Getting the type of '_prepare_index_for_memoryview' (line 297)
        _prepare_index_for_memoryview_378294 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 297, 15), '_prepare_index_for_memoryview', False)
        # Calling _prepare_index_for_memoryview(args, kwargs) (line 297)
        _prepare_index_for_memoryview_call_result_378298 = invoke(stypy.reporting.localization.Localization(__file__, 297, 15), _prepare_index_for_memoryview_378294, *[i_378295, j_378296], **kwargs_378297)
        
        # Obtaining the member '__getitem__' of a type (line 297)
        getitem___378299 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 297, 8), _prepare_index_for_memoryview_call_result_378298, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 297)
        subscript_call_result_378300 = invoke(stypy.reporting.localization.Localization(__file__, 297, 8), getitem___378299, int_378293)
        
        # Assigning a type to the variable 'tuple_var_assignment_377459' (line 297)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 297, 8), 'tuple_var_assignment_377459', subscript_call_result_378300)
        
        # Assigning a Name to a Name (line 297):
        # Getting the type of 'tuple_var_assignment_377458' (line 297)
        tuple_var_assignment_377458_378301 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 297, 8), 'tuple_var_assignment_377458')
        # Assigning a type to the variable 'i' (line 297)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 297, 8), 'i', tuple_var_assignment_377458_378301)
        
        # Assigning a Name to a Name (line 297):
        # Getting the type of 'tuple_var_assignment_377459' (line 297)
        tuple_var_assignment_377459_378302 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 297, 8), 'tuple_var_assignment_377459')
        # Assigning a type to the variable 'j' (line 297)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 297, 11), 'j', tuple_var_assignment_377459_378302)
        
        # Call to lil_fancy_get(...): (line 298)
        # Processing the call arguments (line 298)
        
        # Obtaining the type of the subscript
        int_378305 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 298, 47), 'int')
        # Getting the type of 'self' (line 298)
        self_378306 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 298, 36), 'self', False)
        # Obtaining the member 'shape' of a type (line 298)
        shape_378307 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 298, 36), self_378306, 'shape')
        # Obtaining the member '__getitem__' of a type (line 298)
        getitem___378308 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 298, 36), shape_378307, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 298)
        subscript_call_result_378309 = invoke(stypy.reporting.localization.Localization(__file__, 298, 36), getitem___378308, int_378305)
        
        
        # Obtaining the type of the subscript
        int_378310 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 298, 62), 'int')
        # Getting the type of 'self' (line 298)
        self_378311 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 298, 51), 'self', False)
        # Obtaining the member 'shape' of a type (line 298)
        shape_378312 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 298, 51), self_378311, 'shape')
        # Obtaining the member '__getitem__' of a type (line 298)
        getitem___378313 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 298, 51), shape_378312, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 298)
        subscript_call_result_378314 = invoke(stypy.reporting.localization.Localization(__file__, 298, 51), getitem___378313, int_378310)
        
        # Getting the type of 'self' (line 299)
        self_378315 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 299, 36), 'self', False)
        # Obtaining the member 'rows' of a type (line 299)
        rows_378316 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 299, 36), self_378315, 'rows')
        # Getting the type of 'self' (line 299)
        self_378317 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 299, 47), 'self', False)
        # Obtaining the member 'data' of a type (line 299)
        data_378318 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 299, 47), self_378317, 'data')
        # Getting the type of 'new' (line 300)
        new_378319 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 300, 36), 'new', False)
        # Obtaining the member 'rows' of a type (line 300)
        rows_378320 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 300, 36), new_378319, 'rows')
        # Getting the type of 'new' (line 300)
        new_378321 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 300, 46), 'new', False)
        # Obtaining the member 'data' of a type (line 300)
        data_378322 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 300, 46), new_378321, 'data')
        # Getting the type of 'i' (line 301)
        i_378323 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 301, 36), 'i', False)
        # Getting the type of 'j' (line 301)
        j_378324 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 301, 39), 'j', False)
        # Processing the call keyword arguments (line 298)
        kwargs_378325 = {}
        # Getting the type of '_csparsetools' (line 298)
        _csparsetools_378303 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 298, 8), '_csparsetools', False)
        # Obtaining the member 'lil_fancy_get' of a type (line 298)
        lil_fancy_get_378304 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 298, 8), _csparsetools_378303, 'lil_fancy_get')
        # Calling lil_fancy_get(args, kwargs) (line 298)
        lil_fancy_get_call_result_378326 = invoke(stypy.reporting.localization.Localization(__file__, 298, 8), lil_fancy_get_378304, *[subscript_call_result_378309, subscript_call_result_378314, rows_378316, data_378318, rows_378320, data_378322, i_378323, j_378324], **kwargs_378325)
        
        # Getting the type of 'new' (line 302)
        new_378327 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 302, 15), 'new')
        # Assigning a type to the variable 'stypy_return_type' (line 302)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 302, 8), 'stypy_return_type', new_378327)
        
        # ################# End of '__getitem__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__getitem__' in the type store
        # Getting the type of 'stypy_return_type' (line 240)
        stypy_return_type_378328 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_378328)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__getitem__'
        return stypy_return_type_378328


    @norecursion
    def _get_row_ranges(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_get_row_ranges'
        module_type_store = module_type_store.open_function_context('_get_row_ranges', 304, 4, False)
        # Assigning a type to the variable 'self' (line 305)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 305, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        lil_matrix._get_row_ranges.__dict__.__setitem__('stypy_localization', localization)
        lil_matrix._get_row_ranges.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        lil_matrix._get_row_ranges.__dict__.__setitem__('stypy_type_store', module_type_store)
        lil_matrix._get_row_ranges.__dict__.__setitem__('stypy_function_name', 'lil_matrix._get_row_ranges')
        lil_matrix._get_row_ranges.__dict__.__setitem__('stypy_param_names_list', ['rows', 'col_slice'])
        lil_matrix._get_row_ranges.__dict__.__setitem__('stypy_varargs_param_name', None)
        lil_matrix._get_row_ranges.__dict__.__setitem__('stypy_kwargs_param_name', None)
        lil_matrix._get_row_ranges.__dict__.__setitem__('stypy_call_defaults', defaults)
        lil_matrix._get_row_ranges.__dict__.__setitem__('stypy_call_varargs', varargs)
        lil_matrix._get_row_ranges.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        lil_matrix._get_row_ranges.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'lil_matrix._get_row_ranges', ['rows', 'col_slice'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_get_row_ranges', localization, ['rows', 'col_slice'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_get_row_ranges(...)' code ##################

        str_378329 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 319, (-1)), 'str', '\n        Fast path for indexing in the case where column index is slice.\n\n        This gains performance improvement over brute force by more\n        efficient skipping of zeros, by accessing the elements\n        column-wise in order.\n\n        Parameters\n        ----------\n        rows : sequence or xrange\n            Rows indexed. If xrange, must be within valid bounds.\n        col_slice : slice\n            Columns indexed\n\n        ')
        
        # Assigning a Call to a Tuple (line 320):
        
        # Assigning a Subscript to a Name (line 320):
        
        # Obtaining the type of the subscript
        int_378330 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 320, 8), 'int')
        
        # Call to indices(...): (line 320)
        # Processing the call arguments (line 320)
        
        # Obtaining the type of the subscript
        int_378333 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 320, 65), 'int')
        # Getting the type of 'self' (line 320)
        self_378334 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 320, 54), 'self', False)
        # Obtaining the member 'shape' of a type (line 320)
        shape_378335 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 320, 54), self_378334, 'shape')
        # Obtaining the member '__getitem__' of a type (line 320)
        getitem___378336 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 320, 54), shape_378335, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 320)
        subscript_call_result_378337 = invoke(stypy.reporting.localization.Localization(__file__, 320, 54), getitem___378336, int_378333)
        
        # Processing the call keyword arguments (line 320)
        kwargs_378338 = {}
        # Getting the type of 'col_slice' (line 320)
        col_slice_378331 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 320, 36), 'col_slice', False)
        # Obtaining the member 'indices' of a type (line 320)
        indices_378332 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 320, 36), col_slice_378331, 'indices')
        # Calling indices(args, kwargs) (line 320)
        indices_call_result_378339 = invoke(stypy.reporting.localization.Localization(__file__, 320, 36), indices_378332, *[subscript_call_result_378337], **kwargs_378338)
        
        # Obtaining the member '__getitem__' of a type (line 320)
        getitem___378340 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 320, 8), indices_call_result_378339, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 320)
        subscript_call_result_378341 = invoke(stypy.reporting.localization.Localization(__file__, 320, 8), getitem___378340, int_378330)
        
        # Assigning a type to the variable 'tuple_var_assignment_377460' (line 320)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 320, 8), 'tuple_var_assignment_377460', subscript_call_result_378341)
        
        # Assigning a Subscript to a Name (line 320):
        
        # Obtaining the type of the subscript
        int_378342 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 320, 8), 'int')
        
        # Call to indices(...): (line 320)
        # Processing the call arguments (line 320)
        
        # Obtaining the type of the subscript
        int_378345 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 320, 65), 'int')
        # Getting the type of 'self' (line 320)
        self_378346 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 320, 54), 'self', False)
        # Obtaining the member 'shape' of a type (line 320)
        shape_378347 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 320, 54), self_378346, 'shape')
        # Obtaining the member '__getitem__' of a type (line 320)
        getitem___378348 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 320, 54), shape_378347, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 320)
        subscript_call_result_378349 = invoke(stypy.reporting.localization.Localization(__file__, 320, 54), getitem___378348, int_378345)
        
        # Processing the call keyword arguments (line 320)
        kwargs_378350 = {}
        # Getting the type of 'col_slice' (line 320)
        col_slice_378343 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 320, 36), 'col_slice', False)
        # Obtaining the member 'indices' of a type (line 320)
        indices_378344 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 320, 36), col_slice_378343, 'indices')
        # Calling indices(args, kwargs) (line 320)
        indices_call_result_378351 = invoke(stypy.reporting.localization.Localization(__file__, 320, 36), indices_378344, *[subscript_call_result_378349], **kwargs_378350)
        
        # Obtaining the member '__getitem__' of a type (line 320)
        getitem___378352 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 320, 8), indices_call_result_378351, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 320)
        subscript_call_result_378353 = invoke(stypy.reporting.localization.Localization(__file__, 320, 8), getitem___378352, int_378342)
        
        # Assigning a type to the variable 'tuple_var_assignment_377461' (line 320)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 320, 8), 'tuple_var_assignment_377461', subscript_call_result_378353)
        
        # Assigning a Subscript to a Name (line 320):
        
        # Obtaining the type of the subscript
        int_378354 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 320, 8), 'int')
        
        # Call to indices(...): (line 320)
        # Processing the call arguments (line 320)
        
        # Obtaining the type of the subscript
        int_378357 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 320, 65), 'int')
        # Getting the type of 'self' (line 320)
        self_378358 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 320, 54), 'self', False)
        # Obtaining the member 'shape' of a type (line 320)
        shape_378359 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 320, 54), self_378358, 'shape')
        # Obtaining the member '__getitem__' of a type (line 320)
        getitem___378360 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 320, 54), shape_378359, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 320)
        subscript_call_result_378361 = invoke(stypy.reporting.localization.Localization(__file__, 320, 54), getitem___378360, int_378357)
        
        # Processing the call keyword arguments (line 320)
        kwargs_378362 = {}
        # Getting the type of 'col_slice' (line 320)
        col_slice_378355 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 320, 36), 'col_slice', False)
        # Obtaining the member 'indices' of a type (line 320)
        indices_378356 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 320, 36), col_slice_378355, 'indices')
        # Calling indices(args, kwargs) (line 320)
        indices_call_result_378363 = invoke(stypy.reporting.localization.Localization(__file__, 320, 36), indices_378356, *[subscript_call_result_378361], **kwargs_378362)
        
        # Obtaining the member '__getitem__' of a type (line 320)
        getitem___378364 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 320, 8), indices_call_result_378363, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 320)
        subscript_call_result_378365 = invoke(stypy.reporting.localization.Localization(__file__, 320, 8), getitem___378364, int_378354)
        
        # Assigning a type to the variable 'tuple_var_assignment_377462' (line 320)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 320, 8), 'tuple_var_assignment_377462', subscript_call_result_378365)
        
        # Assigning a Name to a Name (line 320):
        # Getting the type of 'tuple_var_assignment_377460' (line 320)
        tuple_var_assignment_377460_378366 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 320, 8), 'tuple_var_assignment_377460')
        # Assigning a type to the variable 'j_start' (line 320)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 320, 8), 'j_start', tuple_var_assignment_377460_378366)
        
        # Assigning a Name to a Name (line 320):
        # Getting the type of 'tuple_var_assignment_377461' (line 320)
        tuple_var_assignment_377461_378367 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 320, 8), 'tuple_var_assignment_377461')
        # Assigning a type to the variable 'j_stop' (line 320)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 320, 17), 'j_stop', tuple_var_assignment_377461_378367)
        
        # Assigning a Name to a Name (line 320):
        # Getting the type of 'tuple_var_assignment_377462' (line 320)
        tuple_var_assignment_377462_378368 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 320, 8), 'tuple_var_assignment_377462')
        # Assigning a type to the variable 'j_stride' (line 320)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 320, 25), 'j_stride', tuple_var_assignment_377462_378368)
        
        # Assigning a Call to a Name (line 321):
        
        # Assigning a Call to a Name (line 321):
        
        # Call to xrange(...): (line 321)
        # Processing the call arguments (line 321)
        # Getting the type of 'j_start' (line 321)
        j_start_378370 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 321, 27), 'j_start', False)
        # Getting the type of 'j_stop' (line 321)
        j_stop_378371 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 321, 36), 'j_stop', False)
        # Getting the type of 'j_stride' (line 321)
        j_stride_378372 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 321, 44), 'j_stride', False)
        # Processing the call keyword arguments (line 321)
        kwargs_378373 = {}
        # Getting the type of 'xrange' (line 321)
        xrange_378369 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 321, 20), 'xrange', False)
        # Calling xrange(args, kwargs) (line 321)
        xrange_call_result_378374 = invoke(stypy.reporting.localization.Localization(__file__, 321, 20), xrange_378369, *[j_start_378370, j_stop_378371, j_stride_378372], **kwargs_378373)
        
        # Assigning a type to the variable 'col_range' (line 321)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 321, 8), 'col_range', xrange_call_result_378374)
        
        # Assigning a Call to a Name (line 322):
        
        # Assigning a Call to a Name (line 322):
        
        # Call to len(...): (line 322)
        # Processing the call arguments (line 322)
        # Getting the type of 'col_range' (line 322)
        col_range_378376 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 322, 17), 'col_range', False)
        # Processing the call keyword arguments (line 322)
        kwargs_378377 = {}
        # Getting the type of 'len' (line 322)
        len_378375 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 322, 13), 'len', False)
        # Calling len(args, kwargs) (line 322)
        len_call_result_378378 = invoke(stypy.reporting.localization.Localization(__file__, 322, 13), len_378375, *[col_range_378376], **kwargs_378377)
        
        # Assigning a type to the variable 'nj' (line 322)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 322, 8), 'nj', len_call_result_378378)
        
        # Assigning a Call to a Name (line 323):
        
        # Assigning a Call to a Name (line 323):
        
        # Call to lil_matrix(...): (line 323)
        # Processing the call arguments (line 323)
        
        # Obtaining an instance of the builtin type 'tuple' (line 323)
        tuple_378380 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 323, 26), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 323)
        # Adding element type (line 323)
        
        # Call to len(...): (line 323)
        # Processing the call arguments (line 323)
        # Getting the type of 'rows' (line 323)
        rows_378382 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 323, 30), 'rows', False)
        # Processing the call keyword arguments (line 323)
        kwargs_378383 = {}
        # Getting the type of 'len' (line 323)
        len_378381 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 323, 26), 'len', False)
        # Calling len(args, kwargs) (line 323)
        len_call_result_378384 = invoke(stypy.reporting.localization.Localization(__file__, 323, 26), len_378381, *[rows_378382], **kwargs_378383)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 323, 26), tuple_378380, len_call_result_378384)
        # Adding element type (line 323)
        # Getting the type of 'nj' (line 323)
        nj_378385 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 323, 37), 'nj', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 323, 26), tuple_378380, nj_378385)
        
        # Processing the call keyword arguments (line 323)
        # Getting the type of 'self' (line 323)
        self_378386 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 323, 48), 'self', False)
        # Obtaining the member 'dtype' of a type (line 323)
        dtype_378387 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 323, 48), self_378386, 'dtype')
        keyword_378388 = dtype_378387
        kwargs_378389 = {'dtype': keyword_378388}
        # Getting the type of 'lil_matrix' (line 323)
        lil_matrix_378379 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 323, 14), 'lil_matrix', False)
        # Calling lil_matrix(args, kwargs) (line 323)
        lil_matrix_call_result_378390 = invoke(stypy.reporting.localization.Localization(__file__, 323, 14), lil_matrix_378379, *[tuple_378380], **kwargs_378389)
        
        # Assigning a type to the variable 'new' (line 323)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 323, 8), 'new', lil_matrix_call_result_378390)
        
        # Call to lil_get_row_ranges(...): (line 325)
        # Processing the call arguments (line 325)
        
        # Obtaining the type of the subscript
        int_378393 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 325, 52), 'int')
        # Getting the type of 'self' (line 325)
        self_378394 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 325, 41), 'self', False)
        # Obtaining the member 'shape' of a type (line 325)
        shape_378395 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 325, 41), self_378394, 'shape')
        # Obtaining the member '__getitem__' of a type (line 325)
        getitem___378396 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 325, 41), shape_378395, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 325)
        subscript_call_result_378397 = invoke(stypy.reporting.localization.Localization(__file__, 325, 41), getitem___378396, int_378393)
        
        
        # Obtaining the type of the subscript
        int_378398 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 325, 67), 'int')
        # Getting the type of 'self' (line 325)
        self_378399 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 325, 56), 'self', False)
        # Obtaining the member 'shape' of a type (line 325)
        shape_378400 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 325, 56), self_378399, 'shape')
        # Obtaining the member '__getitem__' of a type (line 325)
        getitem___378401 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 325, 56), shape_378400, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 325)
        subscript_call_result_378402 = invoke(stypy.reporting.localization.Localization(__file__, 325, 56), getitem___378401, int_378398)
        
        # Getting the type of 'self' (line 326)
        self_378403 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 326, 41), 'self', False)
        # Obtaining the member 'rows' of a type (line 326)
        rows_378404 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 326, 41), self_378403, 'rows')
        # Getting the type of 'self' (line 326)
        self_378405 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 326, 52), 'self', False)
        # Obtaining the member 'data' of a type (line 326)
        data_378406 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 326, 52), self_378405, 'data')
        # Getting the type of 'new' (line 327)
        new_378407 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 327, 41), 'new', False)
        # Obtaining the member 'rows' of a type (line 327)
        rows_378408 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 327, 41), new_378407, 'rows')
        # Getting the type of 'new' (line 327)
        new_378409 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 327, 51), 'new', False)
        # Obtaining the member 'data' of a type (line 327)
        data_378410 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 327, 51), new_378409, 'data')
        # Getting the type of 'rows' (line 328)
        rows_378411 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 328, 41), 'rows', False)
        # Getting the type of 'j_start' (line 329)
        j_start_378412 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 329, 41), 'j_start', False)
        # Getting the type of 'j_stop' (line 329)
        j_stop_378413 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 329, 50), 'j_stop', False)
        # Getting the type of 'j_stride' (line 329)
        j_stride_378414 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 329, 58), 'j_stride', False)
        # Getting the type of 'nj' (line 329)
        nj_378415 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 329, 68), 'nj', False)
        # Processing the call keyword arguments (line 325)
        kwargs_378416 = {}
        # Getting the type of '_csparsetools' (line 325)
        _csparsetools_378391 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 325, 8), '_csparsetools', False)
        # Obtaining the member 'lil_get_row_ranges' of a type (line 325)
        lil_get_row_ranges_378392 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 325, 8), _csparsetools_378391, 'lil_get_row_ranges')
        # Calling lil_get_row_ranges(args, kwargs) (line 325)
        lil_get_row_ranges_call_result_378417 = invoke(stypy.reporting.localization.Localization(__file__, 325, 8), lil_get_row_ranges_378392, *[subscript_call_result_378397, subscript_call_result_378402, rows_378404, data_378406, rows_378408, data_378410, rows_378411, j_start_378412, j_stop_378413, j_stride_378414, nj_378415], **kwargs_378416)
        
        # Getting the type of 'new' (line 331)
        new_378418 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 331, 15), 'new')
        # Assigning a type to the variable 'stypy_return_type' (line 331)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 331, 8), 'stypy_return_type', new_378418)
        
        # ################# End of '_get_row_ranges(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_get_row_ranges' in the type store
        # Getting the type of 'stypy_return_type' (line 304)
        stypy_return_type_378419 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 304, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_378419)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_get_row_ranges'
        return stypy_return_type_378419


    @norecursion
    def __setitem__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__setitem__'
        module_type_store = module_type_store.open_function_context('__setitem__', 333, 4, False)
        # Assigning a type to the variable 'self' (line 334)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 334, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        lil_matrix.__setitem__.__dict__.__setitem__('stypy_localization', localization)
        lil_matrix.__setitem__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        lil_matrix.__setitem__.__dict__.__setitem__('stypy_type_store', module_type_store)
        lil_matrix.__setitem__.__dict__.__setitem__('stypy_function_name', 'lil_matrix.__setitem__')
        lil_matrix.__setitem__.__dict__.__setitem__('stypy_param_names_list', ['index', 'x'])
        lil_matrix.__setitem__.__dict__.__setitem__('stypy_varargs_param_name', None)
        lil_matrix.__setitem__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        lil_matrix.__setitem__.__dict__.__setitem__('stypy_call_defaults', defaults)
        lil_matrix.__setitem__.__dict__.__setitem__('stypy_call_varargs', varargs)
        lil_matrix.__setitem__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        lil_matrix.__setitem__.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'lil_matrix.__setitem__', ['index', 'x'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__setitem__', localization, ['index', 'x'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__setitem__(...)' code ##################

        
        
        # Evaluating a boolean operation
        
        # Call to isinstance(...): (line 335)
        # Processing the call arguments (line 335)
        # Getting the type of 'index' (line 335)
        index_378421 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 335, 22), 'index', False)
        # Getting the type of 'tuple' (line 335)
        tuple_378422 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 335, 29), 'tuple', False)
        # Processing the call keyword arguments (line 335)
        kwargs_378423 = {}
        # Getting the type of 'isinstance' (line 335)
        isinstance_378420 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 335, 11), 'isinstance', False)
        # Calling isinstance(args, kwargs) (line 335)
        isinstance_call_result_378424 = invoke(stypy.reporting.localization.Localization(__file__, 335, 11), isinstance_378420, *[index_378421, tuple_378422], **kwargs_378423)
        
        
        
        # Call to len(...): (line 335)
        # Processing the call arguments (line 335)
        # Getting the type of 'index' (line 335)
        index_378426 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 335, 44), 'index', False)
        # Processing the call keyword arguments (line 335)
        kwargs_378427 = {}
        # Getting the type of 'len' (line 335)
        len_378425 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 335, 40), 'len', False)
        # Calling len(args, kwargs) (line 335)
        len_call_result_378428 = invoke(stypy.reporting.localization.Localization(__file__, 335, 40), len_378425, *[index_378426], **kwargs_378427)
        
        int_378429 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 335, 54), 'int')
        # Applying the binary operator '==' (line 335)
        result_eq_378430 = python_operator(stypy.reporting.localization.Localization(__file__, 335, 40), '==', len_call_result_378428, int_378429)
        
        # Applying the binary operator 'and' (line 335)
        result_and_keyword_378431 = python_operator(stypy.reporting.localization.Localization(__file__, 335, 11), 'and', isinstance_call_result_378424, result_eq_378430)
        
        # Testing the type of an if condition (line 335)
        if_condition_378432 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 335, 8), result_and_keyword_378431)
        # Assigning a type to the variable 'if_condition_378432' (line 335)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 335, 8), 'if_condition_378432', if_condition_378432)
        # SSA begins for if statement (line 335)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Name to a Tuple (line 336):
        
        # Assigning a Subscript to a Name (line 336):
        
        # Obtaining the type of the subscript
        int_378433 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 336, 12), 'int')
        # Getting the type of 'index' (line 336)
        index_378434 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 336, 19), 'index')
        # Obtaining the member '__getitem__' of a type (line 336)
        getitem___378435 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 336, 12), index_378434, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 336)
        subscript_call_result_378436 = invoke(stypy.reporting.localization.Localization(__file__, 336, 12), getitem___378435, int_378433)
        
        # Assigning a type to the variable 'tuple_var_assignment_377463' (line 336)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 336, 12), 'tuple_var_assignment_377463', subscript_call_result_378436)
        
        # Assigning a Subscript to a Name (line 336):
        
        # Obtaining the type of the subscript
        int_378437 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 336, 12), 'int')
        # Getting the type of 'index' (line 336)
        index_378438 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 336, 19), 'index')
        # Obtaining the member '__getitem__' of a type (line 336)
        getitem___378439 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 336, 12), index_378438, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 336)
        subscript_call_result_378440 = invoke(stypy.reporting.localization.Localization(__file__, 336, 12), getitem___378439, int_378437)
        
        # Assigning a type to the variable 'tuple_var_assignment_377464' (line 336)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 336, 12), 'tuple_var_assignment_377464', subscript_call_result_378440)
        
        # Assigning a Name to a Name (line 336):
        # Getting the type of 'tuple_var_assignment_377463' (line 336)
        tuple_var_assignment_377463_378441 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 336, 12), 'tuple_var_assignment_377463')
        # Assigning a type to the variable 'i' (line 336)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 336, 12), 'i', tuple_var_assignment_377463_378441)
        
        # Assigning a Name to a Name (line 336):
        # Getting the type of 'tuple_var_assignment_377464' (line 336)
        tuple_var_assignment_377464_378442 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 336, 12), 'tuple_var_assignment_377464')
        # Assigning a type to the variable 'j' (line 336)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 336, 15), 'j', tuple_var_assignment_377464_378442)
        
        
        # Evaluating a boolean operation
        
        # Evaluating a boolean operation
        
        # Call to isinstance(...): (line 341)
        # Processing the call arguments (line 341)
        # Getting the type of 'i' (line 341)
        i_378444 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 341, 28), 'i', False)
        # Getting the type of 'int' (line 341)
        int_378445 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 341, 31), 'int', False)
        # Processing the call keyword arguments (line 341)
        kwargs_378446 = {}
        # Getting the type of 'isinstance' (line 341)
        isinstance_378443 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 341, 17), 'isinstance', False)
        # Calling isinstance(args, kwargs) (line 341)
        isinstance_call_result_378447 = invoke(stypy.reporting.localization.Localization(__file__, 341, 17), isinstance_378443, *[i_378444, int_378445], **kwargs_378446)
        
        
        # Call to isinstance(...): (line 341)
        # Processing the call arguments (line 341)
        # Getting the type of 'i' (line 341)
        i_378449 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 341, 50), 'i', False)
        # Getting the type of 'np' (line 341)
        np_378450 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 341, 53), 'np', False)
        # Obtaining the member 'integer' of a type (line 341)
        integer_378451 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 341, 53), np_378450, 'integer')
        # Processing the call keyword arguments (line 341)
        kwargs_378452 = {}
        # Getting the type of 'isinstance' (line 341)
        isinstance_378448 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 341, 39), 'isinstance', False)
        # Calling isinstance(args, kwargs) (line 341)
        isinstance_call_result_378453 = invoke(stypy.reporting.localization.Localization(__file__, 341, 39), isinstance_378448, *[i_378449, integer_378451], **kwargs_378452)
        
        # Applying the binary operator 'or' (line 341)
        result_or_keyword_378454 = python_operator(stypy.reporting.localization.Localization(__file__, 341, 17), 'or', isinstance_call_result_378447, isinstance_call_result_378453)
        
        
        # Evaluating a boolean operation
        
        # Call to isinstance(...): (line 342)
        # Processing the call arguments (line 342)
        # Getting the type of 'j' (line 342)
        j_378456 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 342, 32), 'j', False)
        # Getting the type of 'int' (line 342)
        int_378457 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 342, 35), 'int', False)
        # Processing the call keyword arguments (line 342)
        kwargs_378458 = {}
        # Getting the type of 'isinstance' (line 342)
        isinstance_378455 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 342, 21), 'isinstance', False)
        # Calling isinstance(args, kwargs) (line 342)
        isinstance_call_result_378459 = invoke(stypy.reporting.localization.Localization(__file__, 342, 21), isinstance_378455, *[j_378456, int_378457], **kwargs_378458)
        
        
        # Call to isinstance(...): (line 342)
        # Processing the call arguments (line 342)
        # Getting the type of 'j' (line 342)
        j_378461 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 342, 54), 'j', False)
        # Getting the type of 'np' (line 342)
        np_378462 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 342, 57), 'np', False)
        # Obtaining the member 'integer' of a type (line 342)
        integer_378463 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 342, 57), np_378462, 'integer')
        # Processing the call keyword arguments (line 342)
        kwargs_378464 = {}
        # Getting the type of 'isinstance' (line 342)
        isinstance_378460 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 342, 43), 'isinstance', False)
        # Calling isinstance(args, kwargs) (line 342)
        isinstance_call_result_378465 = invoke(stypy.reporting.localization.Localization(__file__, 342, 43), isinstance_378460, *[j_378461, integer_378463], **kwargs_378464)
        
        # Applying the binary operator 'or' (line 342)
        result_or_keyword_378466 = python_operator(stypy.reporting.localization.Localization(__file__, 342, 21), 'or', isinstance_call_result_378459, isinstance_call_result_378465)
        
        # Applying the binary operator 'and' (line 341)
        result_and_keyword_378467 = python_operator(stypy.reporting.localization.Localization(__file__, 341, 16), 'and', result_or_keyword_378454, result_or_keyword_378466)
        
        # Testing the type of an if condition (line 341)
        if_condition_378468 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 341, 12), result_and_keyword_378467)
        # Assigning a type to the variable 'if_condition_378468' (line 341)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 341, 12), 'if_condition_378468', if_condition_378468)
        # SSA begins for if statement (line 341)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 343):
        
        # Assigning a Call to a Name (line 343):
        
        # Call to type(...): (line 343)
        # Processing the call arguments (line 343)
        # Getting the type of 'x' (line 343)
        x_378472 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 343, 36), 'x', False)
        # Processing the call keyword arguments (line 343)
        kwargs_378473 = {}
        # Getting the type of 'self' (line 343)
        self_378469 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 343, 20), 'self', False)
        # Obtaining the member 'dtype' of a type (line 343)
        dtype_378470 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 343, 20), self_378469, 'dtype')
        # Obtaining the member 'type' of a type (line 343)
        type_378471 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 343, 20), dtype_378470, 'type')
        # Calling type(args, kwargs) (line 343)
        type_call_result_378474 = invoke(stypy.reporting.localization.Localization(__file__, 343, 20), type_378471, *[x_378472], **kwargs_378473)
        
        # Assigning a type to the variable 'x' (line 343)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 343, 16), 'x', type_call_result_378474)
        
        
        # Getting the type of 'x' (line 344)
        x_378475 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 344, 19), 'x')
        # Obtaining the member 'size' of a type (line 344)
        size_378476 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 344, 19), x_378475, 'size')
        int_378477 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 344, 28), 'int')
        # Applying the binary operator '>' (line 344)
        result_gt_378478 = python_operator(stypy.reporting.localization.Localization(__file__, 344, 19), '>', size_378476, int_378477)
        
        # Testing the type of an if condition (line 344)
        if_condition_378479 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 344, 16), result_gt_378478)
        # Assigning a type to the variable 'if_condition_378479' (line 344)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 344, 16), 'if_condition_378479', if_condition_378479)
        # SSA begins for if statement (line 344)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to ValueError(...): (line 346)
        # Processing the call arguments (line 346)
        str_378481 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 346, 37), 'str', 'Trying to assign a sequence to an item')
        # Processing the call keyword arguments (line 346)
        kwargs_378482 = {}
        # Getting the type of 'ValueError' (line 346)
        ValueError_378480 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 346, 26), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 346)
        ValueError_call_result_378483 = invoke(stypy.reporting.localization.Localization(__file__, 346, 26), ValueError_378480, *[str_378481], **kwargs_378482)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 346, 20), ValueError_call_result_378483, 'raise parameter', BaseException)
        # SSA join for if statement (line 344)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to lil_insert(...): (line 347)
        # Processing the call arguments (line 347)
        
        # Obtaining the type of the subscript
        int_378486 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 347, 52), 'int')
        # Getting the type of 'self' (line 347)
        self_378487 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 347, 41), 'self', False)
        # Obtaining the member 'shape' of a type (line 347)
        shape_378488 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 347, 41), self_378487, 'shape')
        # Obtaining the member '__getitem__' of a type (line 347)
        getitem___378489 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 347, 41), shape_378488, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 347)
        subscript_call_result_378490 = invoke(stypy.reporting.localization.Localization(__file__, 347, 41), getitem___378489, int_378486)
        
        
        # Obtaining the type of the subscript
        int_378491 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 347, 67), 'int')
        # Getting the type of 'self' (line 347)
        self_378492 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 347, 56), 'self', False)
        # Obtaining the member 'shape' of a type (line 347)
        shape_378493 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 347, 56), self_378492, 'shape')
        # Obtaining the member '__getitem__' of a type (line 347)
        getitem___378494 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 347, 56), shape_378493, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 347)
        subscript_call_result_378495 = invoke(stypy.reporting.localization.Localization(__file__, 347, 56), getitem___378494, int_378491)
        
        # Getting the type of 'self' (line 348)
        self_378496 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 348, 41), 'self', False)
        # Obtaining the member 'rows' of a type (line 348)
        rows_378497 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 348, 41), self_378496, 'rows')
        # Getting the type of 'self' (line 348)
        self_378498 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 348, 52), 'self', False)
        # Obtaining the member 'data' of a type (line 348)
        data_378499 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 348, 52), self_378498, 'data')
        # Getting the type of 'i' (line 348)
        i_378500 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 348, 63), 'i', False)
        # Getting the type of 'j' (line 348)
        j_378501 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 348, 66), 'j', False)
        # Getting the type of 'x' (line 348)
        x_378502 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 348, 69), 'x', False)
        # Processing the call keyword arguments (line 347)
        kwargs_378503 = {}
        # Getting the type of '_csparsetools' (line 347)
        _csparsetools_378484 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 347, 16), '_csparsetools', False)
        # Obtaining the member 'lil_insert' of a type (line 347)
        lil_insert_378485 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 347, 16), _csparsetools_378484, 'lil_insert')
        # Calling lil_insert(args, kwargs) (line 347)
        lil_insert_call_result_378504 = invoke(stypy.reporting.localization.Localization(__file__, 347, 16), lil_insert_378485, *[subscript_call_result_378490, subscript_call_result_378495, rows_378497, data_378499, i_378500, j_378501, x_378502], **kwargs_378503)
        
        # Assigning a type to the variable 'stypy_return_type' (line 349)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 349, 16), 'stypy_return_type', types.NoneType)
        # SSA join for if statement (line 341)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 335)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Tuple (line 352):
        
        # Assigning a Subscript to a Name (line 352):
        
        # Obtaining the type of the subscript
        int_378505 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 352, 8), 'int')
        
        # Call to _unpack_index(...): (line 352)
        # Processing the call arguments (line 352)
        # Getting the type of 'index' (line 352)
        index_378508 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 352, 34), 'index', False)
        # Processing the call keyword arguments (line 352)
        kwargs_378509 = {}
        # Getting the type of 'self' (line 352)
        self_378506 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 352, 15), 'self', False)
        # Obtaining the member '_unpack_index' of a type (line 352)
        _unpack_index_378507 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 352, 15), self_378506, '_unpack_index')
        # Calling _unpack_index(args, kwargs) (line 352)
        _unpack_index_call_result_378510 = invoke(stypy.reporting.localization.Localization(__file__, 352, 15), _unpack_index_378507, *[index_378508], **kwargs_378509)
        
        # Obtaining the member '__getitem__' of a type (line 352)
        getitem___378511 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 352, 8), _unpack_index_call_result_378510, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 352)
        subscript_call_result_378512 = invoke(stypy.reporting.localization.Localization(__file__, 352, 8), getitem___378511, int_378505)
        
        # Assigning a type to the variable 'tuple_var_assignment_377465' (line 352)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 352, 8), 'tuple_var_assignment_377465', subscript_call_result_378512)
        
        # Assigning a Subscript to a Name (line 352):
        
        # Obtaining the type of the subscript
        int_378513 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 352, 8), 'int')
        
        # Call to _unpack_index(...): (line 352)
        # Processing the call arguments (line 352)
        # Getting the type of 'index' (line 352)
        index_378516 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 352, 34), 'index', False)
        # Processing the call keyword arguments (line 352)
        kwargs_378517 = {}
        # Getting the type of 'self' (line 352)
        self_378514 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 352, 15), 'self', False)
        # Obtaining the member '_unpack_index' of a type (line 352)
        _unpack_index_378515 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 352, 15), self_378514, '_unpack_index')
        # Calling _unpack_index(args, kwargs) (line 352)
        _unpack_index_call_result_378518 = invoke(stypy.reporting.localization.Localization(__file__, 352, 15), _unpack_index_378515, *[index_378516], **kwargs_378517)
        
        # Obtaining the member '__getitem__' of a type (line 352)
        getitem___378519 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 352, 8), _unpack_index_call_result_378518, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 352)
        subscript_call_result_378520 = invoke(stypy.reporting.localization.Localization(__file__, 352, 8), getitem___378519, int_378513)
        
        # Assigning a type to the variable 'tuple_var_assignment_377466' (line 352)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 352, 8), 'tuple_var_assignment_377466', subscript_call_result_378520)
        
        # Assigning a Name to a Name (line 352):
        # Getting the type of 'tuple_var_assignment_377465' (line 352)
        tuple_var_assignment_377465_378521 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 352, 8), 'tuple_var_assignment_377465')
        # Assigning a type to the variable 'i' (line 352)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 352, 8), 'i', tuple_var_assignment_377465_378521)
        
        # Assigning a Name to a Name (line 352):
        # Getting the type of 'tuple_var_assignment_377466' (line 352)
        tuple_var_assignment_377466_378522 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 352, 8), 'tuple_var_assignment_377466')
        # Assigning a type to the variable 'j' (line 352)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 352, 11), 'j', tuple_var_assignment_377466_378522)
        
        
        # Evaluating a boolean operation
        
        # Call to isspmatrix(...): (line 355)
        # Processing the call arguments (line 355)
        # Getting the type of 'x' (line 355)
        x_378524 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 355, 23), 'x', False)
        # Processing the call keyword arguments (line 355)
        kwargs_378525 = {}
        # Getting the type of 'isspmatrix' (line 355)
        isspmatrix_378523 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 355, 12), 'isspmatrix', False)
        # Calling isspmatrix(args, kwargs) (line 355)
        isspmatrix_call_result_378526 = invoke(stypy.reporting.localization.Localization(__file__, 355, 12), isspmatrix_378523, *[x_378524], **kwargs_378525)
        
        
        # Call to isinstance(...): (line 355)
        # Processing the call arguments (line 355)
        # Getting the type of 'i' (line 355)
        i_378528 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 355, 41), 'i', False)
        # Getting the type of 'slice' (line 355)
        slice_378529 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 355, 44), 'slice', False)
        # Processing the call keyword arguments (line 355)
        kwargs_378530 = {}
        # Getting the type of 'isinstance' (line 355)
        isinstance_378527 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 355, 30), 'isinstance', False)
        # Calling isinstance(args, kwargs) (line 355)
        isinstance_call_result_378531 = invoke(stypy.reporting.localization.Localization(__file__, 355, 30), isinstance_378527, *[i_378528, slice_378529], **kwargs_378530)
        
        # Applying the binary operator 'and' (line 355)
        result_and_keyword_378532 = python_operator(stypy.reporting.localization.Localization(__file__, 355, 12), 'and', isspmatrix_call_result_378526, isinstance_call_result_378531)
        
        # Getting the type of 'i' (line 355)
        i_378533 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 355, 55), 'i')
        
        # Call to slice(...): (line 355)
        # Processing the call arguments (line 355)
        # Getting the type of 'None' (line 355)
        None_378535 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 355, 66), 'None', False)
        # Processing the call keyword arguments (line 355)
        kwargs_378536 = {}
        # Getting the type of 'slice' (line 355)
        slice_378534 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 355, 60), 'slice', False)
        # Calling slice(args, kwargs) (line 355)
        slice_call_result_378537 = invoke(stypy.reporting.localization.Localization(__file__, 355, 60), slice_378534, *[None_378535], **kwargs_378536)
        
        # Applying the binary operator '==' (line 355)
        result_eq_378538 = python_operator(stypy.reporting.localization.Localization(__file__, 355, 55), '==', i_378533, slice_call_result_378537)
        
        # Applying the binary operator 'and' (line 355)
        result_and_keyword_378539 = python_operator(stypy.reporting.localization.Localization(__file__, 355, 12), 'and', result_and_keyword_378532, result_eq_378538)
        
        # Call to isinstance(...): (line 356)
        # Processing the call arguments (line 356)
        # Getting the type of 'j' (line 356)
        j_378541 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 356, 27), 'j', False)
        # Getting the type of 'slice' (line 356)
        slice_378542 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 356, 30), 'slice', False)
        # Processing the call keyword arguments (line 356)
        kwargs_378543 = {}
        # Getting the type of 'isinstance' (line 356)
        isinstance_378540 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 356, 16), 'isinstance', False)
        # Calling isinstance(args, kwargs) (line 356)
        isinstance_call_result_378544 = invoke(stypy.reporting.localization.Localization(__file__, 356, 16), isinstance_378540, *[j_378541, slice_378542], **kwargs_378543)
        
        # Applying the binary operator 'and' (line 355)
        result_and_keyword_378545 = python_operator(stypy.reporting.localization.Localization(__file__, 355, 12), 'and', result_and_keyword_378539, isinstance_call_result_378544)
        
        # Getting the type of 'j' (line 356)
        j_378546 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 356, 41), 'j')
        
        # Call to slice(...): (line 356)
        # Processing the call arguments (line 356)
        # Getting the type of 'None' (line 356)
        None_378548 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 356, 52), 'None', False)
        # Processing the call keyword arguments (line 356)
        kwargs_378549 = {}
        # Getting the type of 'slice' (line 356)
        slice_378547 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 356, 46), 'slice', False)
        # Calling slice(args, kwargs) (line 356)
        slice_call_result_378550 = invoke(stypy.reporting.localization.Localization(__file__, 356, 46), slice_378547, *[None_378548], **kwargs_378549)
        
        # Applying the binary operator '==' (line 356)
        result_eq_378551 = python_operator(stypy.reporting.localization.Localization(__file__, 356, 41), '==', j_378546, slice_call_result_378550)
        
        # Applying the binary operator 'and' (line 355)
        result_and_keyword_378552 = python_operator(stypy.reporting.localization.Localization(__file__, 355, 12), 'and', result_and_keyword_378545, result_eq_378551)
        
        # Getting the type of 'x' (line 357)
        x_378553 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 357, 20), 'x')
        # Obtaining the member 'shape' of a type (line 357)
        shape_378554 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 357, 20), x_378553, 'shape')
        # Getting the type of 'self' (line 357)
        self_378555 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 357, 31), 'self')
        # Obtaining the member 'shape' of a type (line 357)
        shape_378556 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 357, 31), self_378555, 'shape')
        # Applying the binary operator '==' (line 357)
        result_eq_378557 = python_operator(stypy.reporting.localization.Localization(__file__, 357, 20), '==', shape_378554, shape_378556)
        
        # Applying the binary operator 'and' (line 355)
        result_and_keyword_378558 = python_operator(stypy.reporting.localization.Localization(__file__, 355, 12), 'and', result_and_keyword_378552, result_eq_378557)
        
        # Testing the type of an if condition (line 355)
        if_condition_378559 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 355, 8), result_and_keyword_378558)
        # Assigning a type to the variable 'if_condition_378559' (line 355)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 355, 8), 'if_condition_378559', if_condition_378559)
        # SSA begins for if statement (line 355)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 358):
        
        # Assigning a Call to a Name (line 358):
        
        # Call to lil_matrix(...): (line 358)
        # Processing the call arguments (line 358)
        # Getting the type of 'x' (line 358)
        x_378561 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 358, 27), 'x', False)
        # Processing the call keyword arguments (line 358)
        # Getting the type of 'self' (line 358)
        self_378562 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 358, 36), 'self', False)
        # Obtaining the member 'dtype' of a type (line 358)
        dtype_378563 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 358, 36), self_378562, 'dtype')
        keyword_378564 = dtype_378563
        kwargs_378565 = {'dtype': keyword_378564}
        # Getting the type of 'lil_matrix' (line 358)
        lil_matrix_378560 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 358, 16), 'lil_matrix', False)
        # Calling lil_matrix(args, kwargs) (line 358)
        lil_matrix_call_result_378566 = invoke(stypy.reporting.localization.Localization(__file__, 358, 16), lil_matrix_378560, *[x_378561], **kwargs_378565)
        
        # Assigning a type to the variable 'x' (line 358)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 358, 12), 'x', lil_matrix_call_result_378566)
        
        # Assigning a Attribute to a Attribute (line 359):
        
        # Assigning a Attribute to a Attribute (line 359):
        # Getting the type of 'x' (line 359)
        x_378567 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 359, 24), 'x')
        # Obtaining the member 'rows' of a type (line 359)
        rows_378568 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 359, 24), x_378567, 'rows')
        # Getting the type of 'self' (line 359)
        self_378569 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 359, 12), 'self')
        # Setting the type of the member 'rows' of a type (line 359)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 359, 12), self_378569, 'rows', rows_378568)
        
        # Assigning a Attribute to a Attribute (line 360):
        
        # Assigning a Attribute to a Attribute (line 360):
        # Getting the type of 'x' (line 360)
        x_378570 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 360, 24), 'x')
        # Obtaining the member 'data' of a type (line 360)
        data_378571 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 360, 24), x_378570, 'data')
        # Getting the type of 'self' (line 360)
        self_378572 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 360, 12), 'self')
        # Setting the type of the member 'data' of a type (line 360)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 360, 12), self_378572, 'data', data_378571)
        # Assigning a type to the variable 'stypy_return_type' (line 361)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 361, 12), 'stypy_return_type', types.NoneType)
        # SSA join for if statement (line 355)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Tuple (line 363):
        
        # Assigning a Subscript to a Name (line 363):
        
        # Obtaining the type of the subscript
        int_378573 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 363, 8), 'int')
        
        # Call to _index_to_arrays(...): (line 363)
        # Processing the call arguments (line 363)
        # Getting the type of 'i' (line 363)
        i_378576 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 363, 37), 'i', False)
        # Getting the type of 'j' (line 363)
        j_378577 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 363, 40), 'j', False)
        # Processing the call keyword arguments (line 363)
        kwargs_378578 = {}
        # Getting the type of 'self' (line 363)
        self_378574 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 363, 15), 'self', False)
        # Obtaining the member '_index_to_arrays' of a type (line 363)
        _index_to_arrays_378575 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 363, 15), self_378574, '_index_to_arrays')
        # Calling _index_to_arrays(args, kwargs) (line 363)
        _index_to_arrays_call_result_378579 = invoke(stypy.reporting.localization.Localization(__file__, 363, 15), _index_to_arrays_378575, *[i_378576, j_378577], **kwargs_378578)
        
        # Obtaining the member '__getitem__' of a type (line 363)
        getitem___378580 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 363, 8), _index_to_arrays_call_result_378579, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 363)
        subscript_call_result_378581 = invoke(stypy.reporting.localization.Localization(__file__, 363, 8), getitem___378580, int_378573)
        
        # Assigning a type to the variable 'tuple_var_assignment_377467' (line 363)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 363, 8), 'tuple_var_assignment_377467', subscript_call_result_378581)
        
        # Assigning a Subscript to a Name (line 363):
        
        # Obtaining the type of the subscript
        int_378582 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 363, 8), 'int')
        
        # Call to _index_to_arrays(...): (line 363)
        # Processing the call arguments (line 363)
        # Getting the type of 'i' (line 363)
        i_378585 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 363, 37), 'i', False)
        # Getting the type of 'j' (line 363)
        j_378586 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 363, 40), 'j', False)
        # Processing the call keyword arguments (line 363)
        kwargs_378587 = {}
        # Getting the type of 'self' (line 363)
        self_378583 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 363, 15), 'self', False)
        # Obtaining the member '_index_to_arrays' of a type (line 363)
        _index_to_arrays_378584 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 363, 15), self_378583, '_index_to_arrays')
        # Calling _index_to_arrays(args, kwargs) (line 363)
        _index_to_arrays_call_result_378588 = invoke(stypy.reporting.localization.Localization(__file__, 363, 15), _index_to_arrays_378584, *[i_378585, j_378586], **kwargs_378587)
        
        # Obtaining the member '__getitem__' of a type (line 363)
        getitem___378589 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 363, 8), _index_to_arrays_call_result_378588, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 363)
        subscript_call_result_378590 = invoke(stypy.reporting.localization.Localization(__file__, 363, 8), getitem___378589, int_378582)
        
        # Assigning a type to the variable 'tuple_var_assignment_377468' (line 363)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 363, 8), 'tuple_var_assignment_377468', subscript_call_result_378590)
        
        # Assigning a Name to a Name (line 363):
        # Getting the type of 'tuple_var_assignment_377467' (line 363)
        tuple_var_assignment_377467_378591 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 363, 8), 'tuple_var_assignment_377467')
        # Assigning a type to the variable 'i' (line 363)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 363, 8), 'i', tuple_var_assignment_377467_378591)
        
        # Assigning a Name to a Name (line 363):
        # Getting the type of 'tuple_var_assignment_377468' (line 363)
        tuple_var_assignment_377468_378592 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 363, 8), 'tuple_var_assignment_377468')
        # Assigning a type to the variable 'j' (line 363)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 363, 11), 'j', tuple_var_assignment_377468_378592)
        
        
        # Call to isspmatrix(...): (line 365)
        # Processing the call arguments (line 365)
        # Getting the type of 'x' (line 365)
        x_378594 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 365, 22), 'x', False)
        # Processing the call keyword arguments (line 365)
        kwargs_378595 = {}
        # Getting the type of 'isspmatrix' (line 365)
        isspmatrix_378593 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 365, 11), 'isspmatrix', False)
        # Calling isspmatrix(args, kwargs) (line 365)
        isspmatrix_call_result_378596 = invoke(stypy.reporting.localization.Localization(__file__, 365, 11), isspmatrix_378593, *[x_378594], **kwargs_378595)
        
        # Testing the type of an if condition (line 365)
        if_condition_378597 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 365, 8), isspmatrix_call_result_378596)
        # Assigning a type to the variable 'if_condition_378597' (line 365)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 365, 8), 'if_condition_378597', if_condition_378597)
        # SSA begins for if statement (line 365)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 366):
        
        # Assigning a Call to a Name (line 366):
        
        # Call to toarray(...): (line 366)
        # Processing the call keyword arguments (line 366)
        kwargs_378600 = {}
        # Getting the type of 'x' (line 366)
        x_378598 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 366, 16), 'x', False)
        # Obtaining the member 'toarray' of a type (line 366)
        toarray_378599 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 366, 16), x_378598, 'toarray')
        # Calling toarray(args, kwargs) (line 366)
        toarray_call_result_378601 = invoke(stypy.reporting.localization.Localization(__file__, 366, 16), toarray_378599, *[], **kwargs_378600)
        
        # Assigning a type to the variable 'x' (line 366)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 366, 12), 'x', toarray_call_result_378601)
        # SSA join for if statement (line 365)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Name (line 369):
        
        # Assigning a Call to a Name (line 369):
        
        # Call to asarray(...): (line 369)
        # Processing the call arguments (line 369)
        # Getting the type of 'x' (line 369)
        x_378604 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 369, 23), 'x', False)
        # Processing the call keyword arguments (line 369)
        # Getting the type of 'self' (line 369)
        self_378605 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 369, 32), 'self', False)
        # Obtaining the member 'dtype' of a type (line 369)
        dtype_378606 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 369, 32), self_378605, 'dtype')
        keyword_378607 = dtype_378606
        kwargs_378608 = {'dtype': keyword_378607}
        # Getting the type of 'np' (line 369)
        np_378602 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 369, 12), 'np', False)
        # Obtaining the member 'asarray' of a type (line 369)
        asarray_378603 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 369, 12), np_378602, 'asarray')
        # Calling asarray(args, kwargs) (line 369)
        asarray_call_result_378609 = invoke(stypy.reporting.localization.Localization(__file__, 369, 12), asarray_378603, *[x_378604], **kwargs_378608)
        
        # Assigning a type to the variable 'x' (line 369)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 369, 8), 'x', asarray_call_result_378609)
        
        # Assigning a Call to a Tuple (line 370):
        
        # Assigning a Subscript to a Name (line 370):
        
        # Obtaining the type of the subscript
        int_378610 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 370, 8), 'int')
        
        # Call to broadcast_arrays(...): (line 370)
        # Processing the call arguments (line 370)
        # Getting the type of 'x' (line 370)
        x_378613 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 370, 35), 'x', False)
        # Getting the type of 'i' (line 370)
        i_378614 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 370, 38), 'i', False)
        # Processing the call keyword arguments (line 370)
        kwargs_378615 = {}
        # Getting the type of 'np' (line 370)
        np_378611 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 370, 15), 'np', False)
        # Obtaining the member 'broadcast_arrays' of a type (line 370)
        broadcast_arrays_378612 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 370, 15), np_378611, 'broadcast_arrays')
        # Calling broadcast_arrays(args, kwargs) (line 370)
        broadcast_arrays_call_result_378616 = invoke(stypy.reporting.localization.Localization(__file__, 370, 15), broadcast_arrays_378612, *[x_378613, i_378614], **kwargs_378615)
        
        # Obtaining the member '__getitem__' of a type (line 370)
        getitem___378617 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 370, 8), broadcast_arrays_call_result_378616, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 370)
        subscript_call_result_378618 = invoke(stypy.reporting.localization.Localization(__file__, 370, 8), getitem___378617, int_378610)
        
        # Assigning a type to the variable 'tuple_var_assignment_377469' (line 370)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 370, 8), 'tuple_var_assignment_377469', subscript_call_result_378618)
        
        # Assigning a Subscript to a Name (line 370):
        
        # Obtaining the type of the subscript
        int_378619 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 370, 8), 'int')
        
        # Call to broadcast_arrays(...): (line 370)
        # Processing the call arguments (line 370)
        # Getting the type of 'x' (line 370)
        x_378622 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 370, 35), 'x', False)
        # Getting the type of 'i' (line 370)
        i_378623 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 370, 38), 'i', False)
        # Processing the call keyword arguments (line 370)
        kwargs_378624 = {}
        # Getting the type of 'np' (line 370)
        np_378620 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 370, 15), 'np', False)
        # Obtaining the member 'broadcast_arrays' of a type (line 370)
        broadcast_arrays_378621 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 370, 15), np_378620, 'broadcast_arrays')
        # Calling broadcast_arrays(args, kwargs) (line 370)
        broadcast_arrays_call_result_378625 = invoke(stypy.reporting.localization.Localization(__file__, 370, 15), broadcast_arrays_378621, *[x_378622, i_378623], **kwargs_378624)
        
        # Obtaining the member '__getitem__' of a type (line 370)
        getitem___378626 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 370, 8), broadcast_arrays_call_result_378625, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 370)
        subscript_call_result_378627 = invoke(stypy.reporting.localization.Localization(__file__, 370, 8), getitem___378626, int_378619)
        
        # Assigning a type to the variable 'tuple_var_assignment_377470' (line 370)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 370, 8), 'tuple_var_assignment_377470', subscript_call_result_378627)
        
        # Assigning a Name to a Name (line 370):
        # Getting the type of 'tuple_var_assignment_377469' (line 370)
        tuple_var_assignment_377469_378628 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 370, 8), 'tuple_var_assignment_377469')
        # Assigning a type to the variable 'x' (line 370)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 370, 8), 'x', tuple_var_assignment_377469_378628)
        
        # Assigning a Name to a Name (line 370):
        # Getting the type of 'tuple_var_assignment_377470' (line 370)
        tuple_var_assignment_377470_378629 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 370, 8), 'tuple_var_assignment_377470')
        # Assigning a type to the variable '_' (line 370)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 370, 11), '_', tuple_var_assignment_377470_378629)
        
        
        # Getting the type of 'x' (line 372)
        x_378630 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 372, 11), 'x')
        # Obtaining the member 'shape' of a type (line 372)
        shape_378631 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 372, 11), x_378630, 'shape')
        # Getting the type of 'i' (line 372)
        i_378632 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 372, 22), 'i')
        # Obtaining the member 'shape' of a type (line 372)
        shape_378633 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 372, 22), i_378632, 'shape')
        # Applying the binary operator '!=' (line 372)
        result_ne_378634 = python_operator(stypy.reporting.localization.Localization(__file__, 372, 11), '!=', shape_378631, shape_378633)
        
        # Testing the type of an if condition (line 372)
        if_condition_378635 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 372, 8), result_ne_378634)
        # Assigning a type to the variable 'if_condition_378635' (line 372)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 372, 8), 'if_condition_378635', if_condition_378635)
        # SSA begins for if statement (line 372)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to ValueError(...): (line 373)
        # Processing the call arguments (line 373)
        str_378637 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 373, 29), 'str', 'shape mismatch in assignment')
        # Processing the call keyword arguments (line 373)
        kwargs_378638 = {}
        # Getting the type of 'ValueError' (line 373)
        ValueError_378636 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 373, 18), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 373)
        ValueError_call_result_378639 = invoke(stypy.reporting.localization.Localization(__file__, 373, 18), ValueError_378636, *[str_378637], **kwargs_378638)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 373, 12), ValueError_call_result_378639, 'raise parameter', BaseException)
        # SSA join for if statement (line 372)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Tuple (line 376):
        
        # Assigning a Subscript to a Name (line 376):
        
        # Obtaining the type of the subscript
        int_378640 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 376, 8), 'int')
        
        # Call to _prepare_index_for_memoryview(...): (line 376)
        # Processing the call arguments (line 376)
        # Getting the type of 'i' (line 376)
        i_378642 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 376, 48), 'i', False)
        # Getting the type of 'j' (line 376)
        j_378643 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 376, 51), 'j', False)
        # Getting the type of 'x' (line 376)
        x_378644 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 376, 54), 'x', False)
        # Processing the call keyword arguments (line 376)
        kwargs_378645 = {}
        # Getting the type of '_prepare_index_for_memoryview' (line 376)
        _prepare_index_for_memoryview_378641 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 376, 18), '_prepare_index_for_memoryview', False)
        # Calling _prepare_index_for_memoryview(args, kwargs) (line 376)
        _prepare_index_for_memoryview_call_result_378646 = invoke(stypy.reporting.localization.Localization(__file__, 376, 18), _prepare_index_for_memoryview_378641, *[i_378642, j_378643, x_378644], **kwargs_378645)
        
        # Obtaining the member '__getitem__' of a type (line 376)
        getitem___378647 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 376, 8), _prepare_index_for_memoryview_call_result_378646, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 376)
        subscript_call_result_378648 = invoke(stypy.reporting.localization.Localization(__file__, 376, 8), getitem___378647, int_378640)
        
        # Assigning a type to the variable 'tuple_var_assignment_377471' (line 376)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 376, 8), 'tuple_var_assignment_377471', subscript_call_result_378648)
        
        # Assigning a Subscript to a Name (line 376):
        
        # Obtaining the type of the subscript
        int_378649 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 376, 8), 'int')
        
        # Call to _prepare_index_for_memoryview(...): (line 376)
        # Processing the call arguments (line 376)
        # Getting the type of 'i' (line 376)
        i_378651 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 376, 48), 'i', False)
        # Getting the type of 'j' (line 376)
        j_378652 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 376, 51), 'j', False)
        # Getting the type of 'x' (line 376)
        x_378653 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 376, 54), 'x', False)
        # Processing the call keyword arguments (line 376)
        kwargs_378654 = {}
        # Getting the type of '_prepare_index_for_memoryview' (line 376)
        _prepare_index_for_memoryview_378650 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 376, 18), '_prepare_index_for_memoryview', False)
        # Calling _prepare_index_for_memoryview(args, kwargs) (line 376)
        _prepare_index_for_memoryview_call_result_378655 = invoke(stypy.reporting.localization.Localization(__file__, 376, 18), _prepare_index_for_memoryview_378650, *[i_378651, j_378652, x_378653], **kwargs_378654)
        
        # Obtaining the member '__getitem__' of a type (line 376)
        getitem___378656 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 376, 8), _prepare_index_for_memoryview_call_result_378655, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 376)
        subscript_call_result_378657 = invoke(stypy.reporting.localization.Localization(__file__, 376, 8), getitem___378656, int_378649)
        
        # Assigning a type to the variable 'tuple_var_assignment_377472' (line 376)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 376, 8), 'tuple_var_assignment_377472', subscript_call_result_378657)
        
        # Assigning a Subscript to a Name (line 376):
        
        # Obtaining the type of the subscript
        int_378658 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 376, 8), 'int')
        
        # Call to _prepare_index_for_memoryview(...): (line 376)
        # Processing the call arguments (line 376)
        # Getting the type of 'i' (line 376)
        i_378660 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 376, 48), 'i', False)
        # Getting the type of 'j' (line 376)
        j_378661 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 376, 51), 'j', False)
        # Getting the type of 'x' (line 376)
        x_378662 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 376, 54), 'x', False)
        # Processing the call keyword arguments (line 376)
        kwargs_378663 = {}
        # Getting the type of '_prepare_index_for_memoryview' (line 376)
        _prepare_index_for_memoryview_378659 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 376, 18), '_prepare_index_for_memoryview', False)
        # Calling _prepare_index_for_memoryview(args, kwargs) (line 376)
        _prepare_index_for_memoryview_call_result_378664 = invoke(stypy.reporting.localization.Localization(__file__, 376, 18), _prepare_index_for_memoryview_378659, *[i_378660, j_378661, x_378662], **kwargs_378663)
        
        # Obtaining the member '__getitem__' of a type (line 376)
        getitem___378665 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 376, 8), _prepare_index_for_memoryview_call_result_378664, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 376)
        subscript_call_result_378666 = invoke(stypy.reporting.localization.Localization(__file__, 376, 8), getitem___378665, int_378658)
        
        # Assigning a type to the variable 'tuple_var_assignment_377473' (line 376)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 376, 8), 'tuple_var_assignment_377473', subscript_call_result_378666)
        
        # Assigning a Name to a Name (line 376):
        # Getting the type of 'tuple_var_assignment_377471' (line 376)
        tuple_var_assignment_377471_378667 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 376, 8), 'tuple_var_assignment_377471')
        # Assigning a type to the variable 'i' (line 376)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 376, 8), 'i', tuple_var_assignment_377471_378667)
        
        # Assigning a Name to a Name (line 376):
        # Getting the type of 'tuple_var_assignment_377472' (line 376)
        tuple_var_assignment_377472_378668 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 376, 8), 'tuple_var_assignment_377472')
        # Assigning a type to the variable 'j' (line 376)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 376, 11), 'j', tuple_var_assignment_377472_378668)
        
        # Assigning a Name to a Name (line 376):
        # Getting the type of 'tuple_var_assignment_377473' (line 376)
        tuple_var_assignment_377473_378669 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 376, 8), 'tuple_var_assignment_377473')
        # Assigning a type to the variable 'x' (line 376)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 376, 14), 'x', tuple_var_assignment_377473_378669)
        
        # Call to lil_fancy_set(...): (line 377)
        # Processing the call arguments (line 377)
        
        # Obtaining the type of the subscript
        int_378672 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 377, 47), 'int')
        # Getting the type of 'self' (line 377)
        self_378673 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 377, 36), 'self', False)
        # Obtaining the member 'shape' of a type (line 377)
        shape_378674 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 377, 36), self_378673, 'shape')
        # Obtaining the member '__getitem__' of a type (line 377)
        getitem___378675 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 377, 36), shape_378674, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 377)
        subscript_call_result_378676 = invoke(stypy.reporting.localization.Localization(__file__, 377, 36), getitem___378675, int_378672)
        
        
        # Obtaining the type of the subscript
        int_378677 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 377, 62), 'int')
        # Getting the type of 'self' (line 377)
        self_378678 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 377, 51), 'self', False)
        # Obtaining the member 'shape' of a type (line 377)
        shape_378679 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 377, 51), self_378678, 'shape')
        # Obtaining the member '__getitem__' of a type (line 377)
        getitem___378680 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 377, 51), shape_378679, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 377)
        subscript_call_result_378681 = invoke(stypy.reporting.localization.Localization(__file__, 377, 51), getitem___378680, int_378677)
        
        # Getting the type of 'self' (line 378)
        self_378682 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 378, 36), 'self', False)
        # Obtaining the member 'rows' of a type (line 378)
        rows_378683 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 378, 36), self_378682, 'rows')
        # Getting the type of 'self' (line 378)
        self_378684 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 378, 47), 'self', False)
        # Obtaining the member 'data' of a type (line 378)
        data_378685 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 378, 47), self_378684, 'data')
        # Getting the type of 'i' (line 379)
        i_378686 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 379, 36), 'i', False)
        # Getting the type of 'j' (line 379)
        j_378687 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 379, 39), 'j', False)
        # Getting the type of 'x' (line 379)
        x_378688 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 379, 42), 'x', False)
        # Processing the call keyword arguments (line 377)
        kwargs_378689 = {}
        # Getting the type of '_csparsetools' (line 377)
        _csparsetools_378670 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 377, 8), '_csparsetools', False)
        # Obtaining the member 'lil_fancy_set' of a type (line 377)
        lil_fancy_set_378671 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 377, 8), _csparsetools_378670, 'lil_fancy_set')
        # Calling lil_fancy_set(args, kwargs) (line 377)
        lil_fancy_set_call_result_378690 = invoke(stypy.reporting.localization.Localization(__file__, 377, 8), lil_fancy_set_378671, *[subscript_call_result_378676, subscript_call_result_378681, rows_378683, data_378685, i_378686, j_378687, x_378688], **kwargs_378689)
        
        
        # ################# End of '__setitem__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__setitem__' in the type store
        # Getting the type of 'stypy_return_type' (line 333)
        stypy_return_type_378691 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 333, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_378691)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__setitem__'
        return stypy_return_type_378691


    @norecursion
    def _mul_scalar(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_mul_scalar'
        module_type_store = module_type_store.open_function_context('_mul_scalar', 381, 4, False)
        # Assigning a type to the variable 'self' (line 382)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 382, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        lil_matrix._mul_scalar.__dict__.__setitem__('stypy_localization', localization)
        lil_matrix._mul_scalar.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        lil_matrix._mul_scalar.__dict__.__setitem__('stypy_type_store', module_type_store)
        lil_matrix._mul_scalar.__dict__.__setitem__('stypy_function_name', 'lil_matrix._mul_scalar')
        lil_matrix._mul_scalar.__dict__.__setitem__('stypy_param_names_list', ['other'])
        lil_matrix._mul_scalar.__dict__.__setitem__('stypy_varargs_param_name', None)
        lil_matrix._mul_scalar.__dict__.__setitem__('stypy_kwargs_param_name', None)
        lil_matrix._mul_scalar.__dict__.__setitem__('stypy_call_defaults', defaults)
        lil_matrix._mul_scalar.__dict__.__setitem__('stypy_call_varargs', varargs)
        lil_matrix._mul_scalar.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        lil_matrix._mul_scalar.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'lil_matrix._mul_scalar', ['other'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_mul_scalar', localization, ['other'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_mul_scalar(...)' code ##################

        
        
        # Getting the type of 'other' (line 382)
        other_378692 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 382, 11), 'other')
        int_378693 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 382, 20), 'int')
        # Applying the binary operator '==' (line 382)
        result_eq_378694 = python_operator(stypy.reporting.localization.Localization(__file__, 382, 11), '==', other_378692, int_378693)
        
        # Testing the type of an if condition (line 382)
        if_condition_378695 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 382, 8), result_eq_378694)
        # Assigning a type to the variable 'if_condition_378695' (line 382)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 382, 8), 'if_condition_378695', if_condition_378695)
        # SSA begins for if statement (line 382)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 384):
        
        # Assigning a Call to a Name (line 384):
        
        # Call to lil_matrix(...): (line 384)
        # Processing the call arguments (line 384)
        # Getting the type of 'self' (line 384)
        self_378697 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 384, 29), 'self', False)
        # Obtaining the member 'shape' of a type (line 384)
        shape_378698 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 384, 29), self_378697, 'shape')
        # Processing the call keyword arguments (line 384)
        # Getting the type of 'self' (line 384)
        self_378699 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 384, 47), 'self', False)
        # Obtaining the member 'dtype' of a type (line 384)
        dtype_378700 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 384, 47), self_378699, 'dtype')
        keyword_378701 = dtype_378700
        kwargs_378702 = {'dtype': keyword_378701}
        # Getting the type of 'lil_matrix' (line 384)
        lil_matrix_378696 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 384, 18), 'lil_matrix', False)
        # Calling lil_matrix(args, kwargs) (line 384)
        lil_matrix_call_result_378703 = invoke(stypy.reporting.localization.Localization(__file__, 384, 18), lil_matrix_378696, *[shape_378698], **kwargs_378702)
        
        # Assigning a type to the variable 'new' (line 384)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 384, 12), 'new', lil_matrix_call_result_378703)
        # SSA branch for the else part of an if statement (line 382)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Call to a Name (line 386):
        
        # Assigning a Call to a Name (line 386):
        
        # Call to upcast_scalar(...): (line 386)
        # Processing the call arguments (line 386)
        # Getting the type of 'self' (line 386)
        self_378705 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 386, 38), 'self', False)
        # Obtaining the member 'dtype' of a type (line 386)
        dtype_378706 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 386, 38), self_378705, 'dtype')
        # Getting the type of 'other' (line 386)
        other_378707 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 386, 50), 'other', False)
        # Processing the call keyword arguments (line 386)
        kwargs_378708 = {}
        # Getting the type of 'upcast_scalar' (line 386)
        upcast_scalar_378704 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 386, 24), 'upcast_scalar', False)
        # Calling upcast_scalar(args, kwargs) (line 386)
        upcast_scalar_call_result_378709 = invoke(stypy.reporting.localization.Localization(__file__, 386, 24), upcast_scalar_378704, *[dtype_378706, other_378707], **kwargs_378708)
        
        # Assigning a type to the variable 'res_dtype' (line 386)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 386, 12), 'res_dtype', upcast_scalar_call_result_378709)
        
        # Assigning a Call to a Name (line 388):
        
        # Assigning a Call to a Name (line 388):
        
        # Call to copy(...): (line 388)
        # Processing the call keyword arguments (line 388)
        kwargs_378712 = {}
        # Getting the type of 'self' (line 388)
        self_378710 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 388, 18), 'self', False)
        # Obtaining the member 'copy' of a type (line 388)
        copy_378711 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 388, 18), self_378710, 'copy')
        # Calling copy(args, kwargs) (line 388)
        copy_call_result_378713 = invoke(stypy.reporting.localization.Localization(__file__, 388, 18), copy_378711, *[], **kwargs_378712)
        
        # Assigning a type to the variable 'new' (line 388)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 388, 12), 'new', copy_call_result_378713)
        
        # Assigning a Call to a Name (line 389):
        
        # Assigning a Call to a Name (line 389):
        
        # Call to astype(...): (line 389)
        # Processing the call arguments (line 389)
        # Getting the type of 'res_dtype' (line 389)
        res_dtype_378716 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 389, 29), 'res_dtype', False)
        # Processing the call keyword arguments (line 389)
        kwargs_378717 = {}
        # Getting the type of 'new' (line 389)
        new_378714 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 389, 18), 'new', False)
        # Obtaining the member 'astype' of a type (line 389)
        astype_378715 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 389, 18), new_378714, 'astype')
        # Calling astype(args, kwargs) (line 389)
        astype_call_result_378718 = invoke(stypy.reporting.localization.Localization(__file__, 389, 18), astype_378715, *[res_dtype_378716], **kwargs_378717)
        
        # Assigning a type to the variable 'new' (line 389)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 389, 12), 'new', astype_call_result_378718)
        
        
        # Call to enumerate(...): (line 391)
        # Processing the call arguments (line 391)
        # Getting the type of 'new' (line 391)
        new_378720 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 391, 40), 'new', False)
        # Obtaining the member 'data' of a type (line 391)
        data_378721 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 391, 40), new_378720, 'data')
        # Processing the call keyword arguments (line 391)
        kwargs_378722 = {}
        # Getting the type of 'enumerate' (line 391)
        enumerate_378719 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 391, 30), 'enumerate', False)
        # Calling enumerate(args, kwargs) (line 391)
        enumerate_call_result_378723 = invoke(stypy.reporting.localization.Localization(__file__, 391, 30), enumerate_378719, *[data_378721], **kwargs_378722)
        
        # Testing the type of a for loop iterable (line 391)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 391, 12), enumerate_call_result_378723)
        # Getting the type of the for loop variable (line 391)
        for_loop_var_378724 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 391, 12), enumerate_call_result_378723)
        # Assigning a type to the variable 'j' (line 391)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 391, 12), 'j', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 391, 12), for_loop_var_378724))
        # Assigning a type to the variable 'rowvals' (line 391)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 391, 12), 'rowvals', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 391, 12), for_loop_var_378724))
        # SSA begins for a for statement (line 391)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a ListComp to a Subscript (line 392):
        
        # Assigning a ListComp to a Subscript (line 392):
        # Calculating list comprehension
        # Calculating comprehension expression
        # Getting the type of 'rowvals' (line 392)
        rowvals_378728 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 392, 52), 'rowvals')
        comprehension_378729 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 392, 31), rowvals_378728)
        # Assigning a type to the variable 'val' (line 392)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 392, 31), 'val', comprehension_378729)
        # Getting the type of 'val' (line 392)
        val_378725 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 392, 31), 'val')
        # Getting the type of 'other' (line 392)
        other_378726 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 392, 35), 'other')
        # Applying the binary operator '*' (line 392)
        result_mul_378727 = python_operator(stypy.reporting.localization.Localization(__file__, 392, 31), '*', val_378725, other_378726)
        
        list_378730 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 392, 31), 'list')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 392, 31), list_378730, result_mul_378727)
        # Getting the type of 'new' (line 392)
        new_378731 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 392, 16), 'new')
        # Obtaining the member 'data' of a type (line 392)
        data_378732 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 392, 16), new_378731, 'data')
        # Getting the type of 'j' (line 392)
        j_378733 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 392, 25), 'j')
        # Storing an element on a container (line 392)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 392, 16), data_378732, (j_378733, list_378730))
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 382)
        module_type_store = module_type_store.join_ssa_context()
        
        # Getting the type of 'new' (line 393)
        new_378734 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 393, 15), 'new')
        # Assigning a type to the variable 'stypy_return_type' (line 393)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 393, 8), 'stypy_return_type', new_378734)
        
        # ################# End of '_mul_scalar(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_mul_scalar' in the type store
        # Getting the type of 'stypy_return_type' (line 381)
        stypy_return_type_378735 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 381, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_378735)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_mul_scalar'
        return stypy_return_type_378735


    @norecursion
    def __truediv__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__truediv__'
        module_type_store = module_type_store.open_function_context('__truediv__', 395, 4, False)
        # Assigning a type to the variable 'self' (line 396)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 396, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        lil_matrix.__truediv__.__dict__.__setitem__('stypy_localization', localization)
        lil_matrix.__truediv__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        lil_matrix.__truediv__.__dict__.__setitem__('stypy_type_store', module_type_store)
        lil_matrix.__truediv__.__dict__.__setitem__('stypy_function_name', 'lil_matrix.__truediv__')
        lil_matrix.__truediv__.__dict__.__setitem__('stypy_param_names_list', ['other'])
        lil_matrix.__truediv__.__dict__.__setitem__('stypy_varargs_param_name', None)
        lil_matrix.__truediv__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        lil_matrix.__truediv__.__dict__.__setitem__('stypy_call_defaults', defaults)
        lil_matrix.__truediv__.__dict__.__setitem__('stypy_call_varargs', varargs)
        lil_matrix.__truediv__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        lil_matrix.__truediv__.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'lil_matrix.__truediv__', ['other'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__truediv__', localization, ['other'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__truediv__(...)' code ##################

        
        
        # Call to isscalarlike(...): (line 396)
        # Processing the call arguments (line 396)
        # Getting the type of 'other' (line 396)
        other_378737 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 396, 24), 'other', False)
        # Processing the call keyword arguments (line 396)
        kwargs_378738 = {}
        # Getting the type of 'isscalarlike' (line 396)
        isscalarlike_378736 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 396, 11), 'isscalarlike', False)
        # Calling isscalarlike(args, kwargs) (line 396)
        isscalarlike_call_result_378739 = invoke(stypy.reporting.localization.Localization(__file__, 396, 11), isscalarlike_378736, *[other_378737], **kwargs_378738)
        
        # Testing the type of an if condition (line 396)
        if_condition_378740 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 396, 8), isscalarlike_call_result_378739)
        # Assigning a type to the variable 'if_condition_378740' (line 396)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 396, 8), 'if_condition_378740', if_condition_378740)
        # SSA begins for if statement (line 396)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 397):
        
        # Assigning a Call to a Name (line 397):
        
        # Call to copy(...): (line 397)
        # Processing the call keyword arguments (line 397)
        kwargs_378743 = {}
        # Getting the type of 'self' (line 397)
        self_378741 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 397, 18), 'self', False)
        # Obtaining the member 'copy' of a type (line 397)
        copy_378742 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 397, 18), self_378741, 'copy')
        # Calling copy(args, kwargs) (line 397)
        copy_call_result_378744 = invoke(stypy.reporting.localization.Localization(__file__, 397, 18), copy_378742, *[], **kwargs_378743)
        
        # Assigning a type to the variable 'new' (line 397)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 397, 12), 'new', copy_call_result_378744)
        
        
        # Call to enumerate(...): (line 399)
        # Processing the call arguments (line 399)
        # Getting the type of 'new' (line 399)
        new_378746 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 399, 40), 'new', False)
        # Obtaining the member 'data' of a type (line 399)
        data_378747 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 399, 40), new_378746, 'data')
        # Processing the call keyword arguments (line 399)
        kwargs_378748 = {}
        # Getting the type of 'enumerate' (line 399)
        enumerate_378745 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 399, 30), 'enumerate', False)
        # Calling enumerate(args, kwargs) (line 399)
        enumerate_call_result_378749 = invoke(stypy.reporting.localization.Localization(__file__, 399, 30), enumerate_378745, *[data_378747], **kwargs_378748)
        
        # Testing the type of a for loop iterable (line 399)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 399, 12), enumerate_call_result_378749)
        # Getting the type of the for loop variable (line 399)
        for_loop_var_378750 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 399, 12), enumerate_call_result_378749)
        # Assigning a type to the variable 'j' (line 399)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 399, 12), 'j', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 399, 12), for_loop_var_378750))
        # Assigning a type to the variable 'rowvals' (line 399)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 399, 12), 'rowvals', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 399, 12), for_loop_var_378750))
        # SSA begins for a for statement (line 399)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a ListComp to a Subscript (line 400):
        
        # Assigning a ListComp to a Subscript (line 400):
        # Calculating list comprehension
        # Calculating comprehension expression
        # Getting the type of 'rowvals' (line 400)
        rowvals_378754 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 400, 52), 'rowvals')
        comprehension_378755 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 400, 31), rowvals_378754)
        # Assigning a type to the variable 'val' (line 400)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 400, 31), 'val', comprehension_378755)
        # Getting the type of 'val' (line 400)
        val_378751 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 400, 31), 'val')
        # Getting the type of 'other' (line 400)
        other_378752 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 400, 35), 'other')
        # Applying the binary operator 'div' (line 400)
        result_div_378753 = python_operator(stypy.reporting.localization.Localization(__file__, 400, 31), 'div', val_378751, other_378752)
        
        list_378756 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 400, 31), 'list')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 400, 31), list_378756, result_div_378753)
        # Getting the type of 'new' (line 400)
        new_378757 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 400, 16), 'new')
        # Obtaining the member 'data' of a type (line 400)
        data_378758 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 400, 16), new_378757, 'data')
        # Getting the type of 'j' (line 400)
        j_378759 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 400, 25), 'j')
        # Storing an element on a container (line 400)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 400, 16), data_378758, (j_378759, list_378756))
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        # Getting the type of 'new' (line 401)
        new_378760 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 401, 19), 'new')
        # Assigning a type to the variable 'stypy_return_type' (line 401)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 401, 12), 'stypy_return_type', new_378760)
        # SSA branch for the else part of an if statement (line 396)
        module_type_store.open_ssa_branch('else')
        
        # Call to tocsr(...): (line 403)
        # Processing the call keyword arguments (line 403)
        kwargs_378763 = {}
        # Getting the type of 'self' (line 403)
        self_378761 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 403, 19), 'self', False)
        # Obtaining the member 'tocsr' of a type (line 403)
        tocsr_378762 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 403, 19), self_378761, 'tocsr')
        # Calling tocsr(args, kwargs) (line 403)
        tocsr_call_result_378764 = invoke(stypy.reporting.localization.Localization(__file__, 403, 19), tocsr_378762, *[], **kwargs_378763)
        
        # Getting the type of 'other' (line 403)
        other_378765 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 403, 34), 'other')
        # Applying the binary operator 'div' (line 403)
        result_div_378766 = python_operator(stypy.reporting.localization.Localization(__file__, 403, 19), 'div', tocsr_call_result_378764, other_378765)
        
        # Assigning a type to the variable 'stypy_return_type' (line 403)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 403, 12), 'stypy_return_type', result_div_378766)
        # SSA join for if statement (line 396)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of '__truediv__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__truediv__' in the type store
        # Getting the type of 'stypy_return_type' (line 395)
        stypy_return_type_378767 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 395, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_378767)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__truediv__'
        return stypy_return_type_378767


    @norecursion
    def copy(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'copy'
        module_type_store = module_type_store.open_function_context('copy', 405, 4, False)
        # Assigning a type to the variable 'self' (line 406)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 406, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        lil_matrix.copy.__dict__.__setitem__('stypy_localization', localization)
        lil_matrix.copy.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        lil_matrix.copy.__dict__.__setitem__('stypy_type_store', module_type_store)
        lil_matrix.copy.__dict__.__setitem__('stypy_function_name', 'lil_matrix.copy')
        lil_matrix.copy.__dict__.__setitem__('stypy_param_names_list', [])
        lil_matrix.copy.__dict__.__setitem__('stypy_varargs_param_name', None)
        lil_matrix.copy.__dict__.__setitem__('stypy_kwargs_param_name', None)
        lil_matrix.copy.__dict__.__setitem__('stypy_call_defaults', defaults)
        lil_matrix.copy.__dict__.__setitem__('stypy_call_varargs', varargs)
        lil_matrix.copy.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        lil_matrix.copy.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'lil_matrix.copy', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'copy', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'copy(...)' code ##################

        stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 406, 8))
        
        # 'from copy import deepcopy' statement (line 406)
        try:
            from copy import deepcopy

        except:
            deepcopy = UndefinedType
        import_from_module(stypy.reporting.localization.Localization(__file__, 406, 8), 'copy', None, module_type_store, ['deepcopy'], [deepcopy])
        
        
        # Assigning a Call to a Name (line 407):
        
        # Assigning a Call to a Name (line 407):
        
        # Call to lil_matrix(...): (line 407)
        # Processing the call arguments (line 407)
        # Getting the type of 'self' (line 407)
        self_378769 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 407, 25), 'self', False)
        # Obtaining the member 'shape' of a type (line 407)
        shape_378770 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 407, 25), self_378769, 'shape')
        # Processing the call keyword arguments (line 407)
        # Getting the type of 'self' (line 407)
        self_378771 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 407, 43), 'self', False)
        # Obtaining the member 'dtype' of a type (line 407)
        dtype_378772 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 407, 43), self_378771, 'dtype')
        keyword_378773 = dtype_378772
        kwargs_378774 = {'dtype': keyword_378773}
        # Getting the type of 'lil_matrix' (line 407)
        lil_matrix_378768 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 407, 14), 'lil_matrix', False)
        # Calling lil_matrix(args, kwargs) (line 407)
        lil_matrix_call_result_378775 = invoke(stypy.reporting.localization.Localization(__file__, 407, 14), lil_matrix_378768, *[shape_378770], **kwargs_378774)
        
        # Assigning a type to the variable 'new' (line 407)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 407, 8), 'new', lil_matrix_call_result_378775)
        
        # Assigning a Call to a Attribute (line 408):
        
        # Assigning a Call to a Attribute (line 408):
        
        # Call to deepcopy(...): (line 408)
        # Processing the call arguments (line 408)
        # Getting the type of 'self' (line 408)
        self_378777 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 408, 28), 'self', False)
        # Obtaining the member 'data' of a type (line 408)
        data_378778 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 408, 28), self_378777, 'data')
        # Processing the call keyword arguments (line 408)
        kwargs_378779 = {}
        # Getting the type of 'deepcopy' (line 408)
        deepcopy_378776 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 408, 19), 'deepcopy', False)
        # Calling deepcopy(args, kwargs) (line 408)
        deepcopy_call_result_378780 = invoke(stypy.reporting.localization.Localization(__file__, 408, 19), deepcopy_378776, *[data_378778], **kwargs_378779)
        
        # Getting the type of 'new' (line 408)
        new_378781 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 408, 8), 'new')
        # Setting the type of the member 'data' of a type (line 408)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 408, 8), new_378781, 'data', deepcopy_call_result_378780)
        
        # Assigning a Call to a Attribute (line 409):
        
        # Assigning a Call to a Attribute (line 409):
        
        # Call to deepcopy(...): (line 409)
        # Processing the call arguments (line 409)
        # Getting the type of 'self' (line 409)
        self_378783 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 409, 28), 'self', False)
        # Obtaining the member 'rows' of a type (line 409)
        rows_378784 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 409, 28), self_378783, 'rows')
        # Processing the call keyword arguments (line 409)
        kwargs_378785 = {}
        # Getting the type of 'deepcopy' (line 409)
        deepcopy_378782 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 409, 19), 'deepcopy', False)
        # Calling deepcopy(args, kwargs) (line 409)
        deepcopy_call_result_378786 = invoke(stypy.reporting.localization.Localization(__file__, 409, 19), deepcopy_378782, *[rows_378784], **kwargs_378785)
        
        # Getting the type of 'new' (line 409)
        new_378787 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 409, 8), 'new')
        # Setting the type of the member 'rows' of a type (line 409)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 409, 8), new_378787, 'rows', deepcopy_call_result_378786)
        # Getting the type of 'new' (line 410)
        new_378788 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 410, 15), 'new')
        # Assigning a type to the variable 'stypy_return_type' (line 410)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 410, 8), 'stypy_return_type', new_378788)
        
        # ################# End of 'copy(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'copy' in the type store
        # Getting the type of 'stypy_return_type' (line 405)
        stypy_return_type_378789 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 405, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_378789)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'copy'
        return stypy_return_type_378789

    
    # Assigning a Attribute to a Attribute (line 412):

    @norecursion
    def reshape(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        str_378790 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 414, 35), 'str', 'C')
        defaults = [str_378790]
        # Create a new context for function 'reshape'
        module_type_store = module_type_store.open_function_context('reshape', 414, 4, False)
        # Assigning a type to the variable 'self' (line 415)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 415, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        lil_matrix.reshape.__dict__.__setitem__('stypy_localization', localization)
        lil_matrix.reshape.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        lil_matrix.reshape.__dict__.__setitem__('stypy_type_store', module_type_store)
        lil_matrix.reshape.__dict__.__setitem__('stypy_function_name', 'lil_matrix.reshape')
        lil_matrix.reshape.__dict__.__setitem__('stypy_param_names_list', ['shape', 'order'])
        lil_matrix.reshape.__dict__.__setitem__('stypy_varargs_param_name', None)
        lil_matrix.reshape.__dict__.__setitem__('stypy_kwargs_param_name', None)
        lil_matrix.reshape.__dict__.__setitem__('stypy_call_defaults', defaults)
        lil_matrix.reshape.__dict__.__setitem__('stypy_call_varargs', varargs)
        lil_matrix.reshape.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        lil_matrix.reshape.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'lil_matrix.reshape', ['shape', 'order'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'reshape', localization, ['shape', 'order'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'reshape(...)' code ##################

        
        
        # Evaluating a boolean operation
        
        
        # Call to type(...): (line 415)
        # Processing the call arguments (line 415)
        # Getting the type of 'order' (line 415)
        order_378792 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 415, 16), 'order', False)
        # Processing the call keyword arguments (line 415)
        kwargs_378793 = {}
        # Getting the type of 'type' (line 415)
        type_378791 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 415, 11), 'type', False)
        # Calling type(args, kwargs) (line 415)
        type_call_result_378794 = invoke(stypy.reporting.localization.Localization(__file__, 415, 11), type_378791, *[order_378792], **kwargs_378793)
        
        # Getting the type of 'str' (line 415)
        str_378795 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 415, 26), 'str')
        # Applying the binary operator '!=' (line 415)
        result_ne_378796 = python_operator(stypy.reporting.localization.Localization(__file__, 415, 11), '!=', type_call_result_378794, str_378795)
        
        
        # Getting the type of 'order' (line 415)
        order_378797 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 415, 33), 'order')
        str_378798 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 415, 42), 'str', 'C')
        # Applying the binary operator '!=' (line 415)
        result_ne_378799 = python_operator(stypy.reporting.localization.Localization(__file__, 415, 33), '!=', order_378797, str_378798)
        
        # Applying the binary operator 'or' (line 415)
        result_or_keyword_378800 = python_operator(stypy.reporting.localization.Localization(__file__, 415, 11), 'or', result_ne_378796, result_ne_378799)
        
        # Testing the type of an if condition (line 415)
        if_condition_378801 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 415, 8), result_or_keyword_378800)
        # Assigning a type to the variable 'if_condition_378801' (line 415)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 415, 8), 'if_condition_378801', if_condition_378801)
        # SSA begins for if statement (line 415)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to ValueError(...): (line 416)
        # Processing the call arguments (line 416)
        str_378803 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 416, 30), 'str', "Sparse matrices do not support an 'order' parameter.")
        # Processing the call keyword arguments (line 416)
        kwargs_378804 = {}
        # Getting the type of 'ValueError' (line 416)
        ValueError_378802 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 416, 18), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 416)
        ValueError_call_result_378805 = invoke(stypy.reporting.localization.Localization(__file__, 416, 18), ValueError_378802, *[str_378803], **kwargs_378804)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 416, 12), ValueError_call_result_378805, 'raise parameter', BaseException)
        # SSA join for if statement (line 415)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Type idiom detected: calculating its left and rigth part (line 419)
        # Getting the type of 'shape' (line 419)
        shape_378806 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 419, 16), 'shape')
        # Getting the type of 'tuple' (line 419)
        tuple_378807 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 419, 26), 'tuple')
        
        (may_be_378808, more_types_in_union_378809) = may_be_type(shape_378806, tuple_378807)

        if may_be_378808:

            if more_types_in_union_378809:
                # Runtime conditional SSA (line 419)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            # Assigning a type to the variable 'shape' (line 419)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 419, 8), 'shape', tuple_378807())
            
            # Call to TypeError(...): (line 420)
            # Processing the call arguments (line 420)
            str_378811 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 420, 28), 'str', "a tuple must be passed in for 'shape'")
            # Processing the call keyword arguments (line 420)
            kwargs_378812 = {}
            # Getting the type of 'TypeError' (line 420)
            TypeError_378810 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 420, 18), 'TypeError', False)
            # Calling TypeError(args, kwargs) (line 420)
            TypeError_call_result_378813 = invoke(stypy.reporting.localization.Localization(__file__, 420, 18), TypeError_378810, *[str_378811], **kwargs_378812)
            
            ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 420, 12), TypeError_call_result_378813, 'raise parameter', BaseException)

            if more_types_in_union_378809:
                # SSA join for if statement (line 419)
                module_type_store = module_type_store.join_ssa_context()


        
        
        
        
        # Call to len(...): (line 422)
        # Processing the call arguments (line 422)
        # Getting the type of 'shape' (line 422)
        shape_378815 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 422, 15), 'shape', False)
        # Processing the call keyword arguments (line 422)
        kwargs_378816 = {}
        # Getting the type of 'len' (line 422)
        len_378814 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 422, 11), 'len', False)
        # Calling len(args, kwargs) (line 422)
        len_call_result_378817 = invoke(stypy.reporting.localization.Localization(__file__, 422, 11), len_378814, *[shape_378815], **kwargs_378816)
        
        int_378818 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 422, 25), 'int')
        # Applying the binary operator '!=' (line 422)
        result_ne_378819 = python_operator(stypy.reporting.localization.Localization(__file__, 422, 11), '!=', len_call_result_378817, int_378818)
        
        # Testing the type of an if condition (line 422)
        if_condition_378820 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 422, 8), result_ne_378819)
        # Assigning a type to the variable 'if_condition_378820' (line 422)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 422, 8), 'if_condition_378820', if_condition_378820)
        # SSA begins for if statement (line 422)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to ValueError(...): (line 423)
        # Processing the call arguments (line 423)
        str_378822 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 423, 29), 'str', "a length-2 tuple must be passed in for 'shape'")
        # Processing the call keyword arguments (line 423)
        kwargs_378823 = {}
        # Getting the type of 'ValueError' (line 423)
        ValueError_378821 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 423, 18), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 423)
        ValueError_call_result_378824 = invoke(stypy.reporting.localization.Localization(__file__, 423, 18), ValueError_378821, *[str_378822], **kwargs_378823)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 423, 12), ValueError_call_result_378824, 'raise parameter', BaseException)
        # SSA join for if statement (line 422)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Name (line 425):
        
        # Assigning a Call to a Name (line 425):
        
        # Call to lil_matrix(...): (line 425)
        # Processing the call arguments (line 425)
        # Getting the type of 'shape' (line 425)
        shape_378826 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 425, 25), 'shape', False)
        # Processing the call keyword arguments (line 425)
        # Getting the type of 'self' (line 425)
        self_378827 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 425, 38), 'self', False)
        # Obtaining the member 'dtype' of a type (line 425)
        dtype_378828 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 425, 38), self_378827, 'dtype')
        keyword_378829 = dtype_378828
        kwargs_378830 = {'dtype': keyword_378829}
        # Getting the type of 'lil_matrix' (line 425)
        lil_matrix_378825 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 425, 14), 'lil_matrix', False)
        # Calling lil_matrix(args, kwargs) (line 425)
        lil_matrix_call_result_378831 = invoke(stypy.reporting.localization.Localization(__file__, 425, 14), lil_matrix_378825, *[shape_378826], **kwargs_378830)
        
        # Assigning a type to the variable 'new' (line 425)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 425, 8), 'new', lil_matrix_call_result_378831)
        
        # Assigning a Subscript to a Name (line 426):
        
        # Assigning a Subscript to a Name (line 426):
        
        # Obtaining the type of the subscript
        int_378832 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 426, 27), 'int')
        # Getting the type of 'self' (line 426)
        self_378833 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 426, 16), 'self')
        # Obtaining the member 'shape' of a type (line 426)
        shape_378834 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 426, 16), self_378833, 'shape')
        # Obtaining the member '__getitem__' of a type (line 426)
        getitem___378835 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 426, 16), shape_378834, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 426)
        subscript_call_result_378836 = invoke(stypy.reporting.localization.Localization(__file__, 426, 16), getitem___378835, int_378832)
        
        # Assigning a type to the variable 'j_max' (line 426)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 426, 8), 'j_max', subscript_call_result_378836)
        
        
        
        # Obtaining the type of the subscript
        int_378837 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 430, 21), 'int')
        # Getting the type of 'new' (line 430)
        new_378838 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 430, 11), 'new')
        # Obtaining the member 'shape' of a type (line 430)
        shape_378839 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 430, 11), new_378838, 'shape')
        # Obtaining the member '__getitem__' of a type (line 430)
        getitem___378840 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 430, 11), shape_378839, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 430)
        subscript_call_result_378841 = invoke(stypy.reporting.localization.Localization(__file__, 430, 11), getitem___378840, int_378837)
        
        
        # Obtaining the type of the subscript
        int_378842 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 430, 36), 'int')
        # Getting the type of 'new' (line 430)
        new_378843 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 430, 26), 'new')
        # Obtaining the member 'shape' of a type (line 430)
        shape_378844 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 430, 26), new_378843, 'shape')
        # Obtaining the member '__getitem__' of a type (line 430)
        getitem___378845 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 430, 26), shape_378844, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 430)
        subscript_call_result_378846 = invoke(stypy.reporting.localization.Localization(__file__, 430, 26), getitem___378845, int_378842)
        
        # Applying the binary operator '*' (line 430)
        result_mul_378847 = python_operator(stypy.reporting.localization.Localization(__file__, 430, 11), '*', subscript_call_result_378841, subscript_call_result_378846)
        
        
        # Obtaining the type of the subscript
        int_378848 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 430, 53), 'int')
        # Getting the type of 'self' (line 430)
        self_378849 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 430, 42), 'self')
        # Obtaining the member 'shape' of a type (line 430)
        shape_378850 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 430, 42), self_378849, 'shape')
        # Obtaining the member '__getitem__' of a type (line 430)
        getitem___378851 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 430, 42), shape_378850, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 430)
        subscript_call_result_378852 = invoke(stypy.reporting.localization.Localization(__file__, 430, 42), getitem___378851, int_378848)
        
        
        # Obtaining the type of the subscript
        int_378853 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 430, 69), 'int')
        # Getting the type of 'self' (line 430)
        self_378854 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 430, 58), 'self')
        # Obtaining the member 'shape' of a type (line 430)
        shape_378855 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 430, 58), self_378854, 'shape')
        # Obtaining the member '__getitem__' of a type (line 430)
        getitem___378856 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 430, 58), shape_378855, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 430)
        subscript_call_result_378857 = invoke(stypy.reporting.localization.Localization(__file__, 430, 58), getitem___378856, int_378853)
        
        # Applying the binary operator '*' (line 430)
        result_mul_378858 = python_operator(stypy.reporting.localization.Localization(__file__, 430, 42), '*', subscript_call_result_378852, subscript_call_result_378857)
        
        # Applying the binary operator '!=' (line 430)
        result_ne_378859 = python_operator(stypy.reporting.localization.Localization(__file__, 430, 11), '!=', result_mul_378847, result_mul_378858)
        
        # Testing the type of an if condition (line 430)
        if_condition_378860 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 430, 8), result_ne_378859)
        # Assigning a type to the variable 'if_condition_378860' (line 430)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 430, 8), 'if_condition_378860', if_condition_378860)
        # SSA begins for if statement (line 430)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to ValueError(...): (line 431)
        # Processing the call arguments (line 431)
        str_378862 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 431, 29), 'str', 'the product of the dimensions for the new sparse matrix must equal that of the original matrix')
        # Processing the call keyword arguments (line 431)
        kwargs_378863 = {}
        # Getting the type of 'ValueError' (line 431)
        ValueError_378861 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 431, 18), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 431)
        ValueError_call_result_378864 = invoke(stypy.reporting.localization.Localization(__file__, 431, 18), ValueError_378861, *[str_378862], **kwargs_378863)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 431, 12), ValueError_call_result_378864, 'raise parameter', BaseException)
        # SSA join for if statement (line 430)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Call to enumerate(...): (line 434)
        # Processing the call arguments (line 434)
        # Getting the type of 'self' (line 434)
        self_378866 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 434, 32), 'self', False)
        # Obtaining the member 'rows' of a type (line 434)
        rows_378867 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 434, 32), self_378866, 'rows')
        # Processing the call keyword arguments (line 434)
        kwargs_378868 = {}
        # Getting the type of 'enumerate' (line 434)
        enumerate_378865 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 434, 22), 'enumerate', False)
        # Calling enumerate(args, kwargs) (line 434)
        enumerate_call_result_378869 = invoke(stypy.reporting.localization.Localization(__file__, 434, 22), enumerate_378865, *[rows_378867], **kwargs_378868)
        
        # Testing the type of a for loop iterable (line 434)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 434, 8), enumerate_call_result_378869)
        # Getting the type of the for loop variable (line 434)
        for_loop_var_378870 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 434, 8), enumerate_call_result_378869)
        # Assigning a type to the variable 'i' (line 434)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 434, 8), 'i', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 434, 8), for_loop_var_378870))
        # Assigning a type to the variable 'row' (line 434)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 434, 8), 'row', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 434, 8), for_loop_var_378870))
        # SSA begins for a for statement (line 434)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        
        # Call to enumerate(...): (line 435)
        # Processing the call arguments (line 435)
        # Getting the type of 'row' (line 435)
        row_378872 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 435, 36), 'row', False)
        # Processing the call keyword arguments (line 435)
        kwargs_378873 = {}
        # Getting the type of 'enumerate' (line 435)
        enumerate_378871 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 435, 26), 'enumerate', False)
        # Calling enumerate(args, kwargs) (line 435)
        enumerate_call_result_378874 = invoke(stypy.reporting.localization.Localization(__file__, 435, 26), enumerate_378871, *[row_378872], **kwargs_378873)
        
        # Testing the type of a for loop iterable (line 435)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 435, 12), enumerate_call_result_378874)
        # Getting the type of the for loop variable (line 435)
        for_loop_var_378875 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 435, 12), enumerate_call_result_378874)
        # Assigning a type to the variable 'col' (line 435)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 435, 12), 'col', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 435, 12), for_loop_var_378875))
        # Assigning a type to the variable 'j' (line 435)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 435, 12), 'j', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 435, 12), for_loop_var_378875))
        # SSA begins for a for statement (line 435)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Call to a Tuple (line 436):
        
        # Assigning a Subscript to a Name (line 436):
        
        # Obtaining the type of the subscript
        int_378876 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 436, 16), 'int')
        
        # Call to unravel_index(...): (line 436)
        # Processing the call arguments (line 436)
        # Getting the type of 'i' (line 436)
        i_378879 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 436, 48), 'i', False)
        # Getting the type of 'j_max' (line 436)
        j_max_378880 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 436, 50), 'j_max', False)
        # Applying the binary operator '*' (line 436)
        result_mul_378881 = python_operator(stypy.reporting.localization.Localization(__file__, 436, 48), '*', i_378879, j_max_378880)
        
        # Getting the type of 'j' (line 436)
        j_378882 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 436, 58), 'j', False)
        # Applying the binary operator '+' (line 436)
        result_add_378883 = python_operator(stypy.reporting.localization.Localization(__file__, 436, 48), '+', result_mul_378881, j_378882)
        
        # Getting the type of 'shape' (line 436)
        shape_378884 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 436, 61), 'shape', False)
        # Processing the call keyword arguments (line 436)
        kwargs_378885 = {}
        # Getting the type of 'np' (line 436)
        np_378877 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 436, 31), 'np', False)
        # Obtaining the member 'unravel_index' of a type (line 436)
        unravel_index_378878 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 436, 31), np_378877, 'unravel_index')
        # Calling unravel_index(args, kwargs) (line 436)
        unravel_index_call_result_378886 = invoke(stypy.reporting.localization.Localization(__file__, 436, 31), unravel_index_378878, *[result_add_378883, shape_378884], **kwargs_378885)
        
        # Obtaining the member '__getitem__' of a type (line 436)
        getitem___378887 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 436, 16), unravel_index_call_result_378886, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 436)
        subscript_call_result_378888 = invoke(stypy.reporting.localization.Localization(__file__, 436, 16), getitem___378887, int_378876)
        
        # Assigning a type to the variable 'tuple_var_assignment_377474' (line 436)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 436, 16), 'tuple_var_assignment_377474', subscript_call_result_378888)
        
        # Assigning a Subscript to a Name (line 436):
        
        # Obtaining the type of the subscript
        int_378889 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 436, 16), 'int')
        
        # Call to unravel_index(...): (line 436)
        # Processing the call arguments (line 436)
        # Getting the type of 'i' (line 436)
        i_378892 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 436, 48), 'i', False)
        # Getting the type of 'j_max' (line 436)
        j_max_378893 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 436, 50), 'j_max', False)
        # Applying the binary operator '*' (line 436)
        result_mul_378894 = python_operator(stypy.reporting.localization.Localization(__file__, 436, 48), '*', i_378892, j_max_378893)
        
        # Getting the type of 'j' (line 436)
        j_378895 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 436, 58), 'j', False)
        # Applying the binary operator '+' (line 436)
        result_add_378896 = python_operator(stypy.reporting.localization.Localization(__file__, 436, 48), '+', result_mul_378894, j_378895)
        
        # Getting the type of 'shape' (line 436)
        shape_378897 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 436, 61), 'shape', False)
        # Processing the call keyword arguments (line 436)
        kwargs_378898 = {}
        # Getting the type of 'np' (line 436)
        np_378890 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 436, 31), 'np', False)
        # Obtaining the member 'unravel_index' of a type (line 436)
        unravel_index_378891 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 436, 31), np_378890, 'unravel_index')
        # Calling unravel_index(args, kwargs) (line 436)
        unravel_index_call_result_378899 = invoke(stypy.reporting.localization.Localization(__file__, 436, 31), unravel_index_378891, *[result_add_378896, shape_378897], **kwargs_378898)
        
        # Obtaining the member '__getitem__' of a type (line 436)
        getitem___378900 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 436, 16), unravel_index_call_result_378899, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 436)
        subscript_call_result_378901 = invoke(stypy.reporting.localization.Localization(__file__, 436, 16), getitem___378900, int_378889)
        
        # Assigning a type to the variable 'tuple_var_assignment_377475' (line 436)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 436, 16), 'tuple_var_assignment_377475', subscript_call_result_378901)
        
        # Assigning a Name to a Name (line 436):
        # Getting the type of 'tuple_var_assignment_377474' (line 436)
        tuple_var_assignment_377474_378902 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 436, 16), 'tuple_var_assignment_377474')
        # Assigning a type to the variable 'new_r' (line 436)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 436, 16), 'new_r', tuple_var_assignment_377474_378902)
        
        # Assigning a Name to a Name (line 436):
        # Getting the type of 'tuple_var_assignment_377475' (line 436)
        tuple_var_assignment_377475_378903 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 436, 16), 'tuple_var_assignment_377475')
        # Assigning a type to the variable 'new_c' (line 436)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 436, 23), 'new_c', tuple_var_assignment_377475_378903)
        
        # Assigning a Subscript to a Subscript (line 437):
        
        # Assigning a Subscript to a Subscript (line 437):
        
        # Obtaining the type of the subscript
        
        # Obtaining an instance of the builtin type 'tuple' (line 437)
        tuple_378904 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 437, 41), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 437)
        # Adding element type (line 437)
        # Getting the type of 'i' (line 437)
        i_378905 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 437, 41), 'i')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 437, 41), tuple_378904, i_378905)
        # Adding element type (line 437)
        # Getting the type of 'j' (line 437)
        j_378906 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 437, 44), 'j')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 437, 41), tuple_378904, j_378906)
        
        # Getting the type of 'self' (line 437)
        self_378907 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 437, 36), 'self')
        # Obtaining the member '__getitem__' of a type (line 437)
        getitem___378908 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 437, 36), self_378907, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 437)
        subscript_call_result_378909 = invoke(stypy.reporting.localization.Localization(__file__, 437, 36), getitem___378908, tuple_378904)
        
        # Getting the type of 'new' (line 437)
        new_378910 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 437, 16), 'new')
        
        # Obtaining an instance of the builtin type 'tuple' (line 437)
        tuple_378911 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 437, 20), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 437)
        # Adding element type (line 437)
        # Getting the type of 'new_r' (line 437)
        new_r_378912 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 437, 20), 'new_r')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 437, 20), tuple_378911, new_r_378912)
        # Adding element type (line 437)
        # Getting the type of 'new_c' (line 437)
        new_c_378913 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 437, 27), 'new_c')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 437, 20), tuple_378911, new_c_378913)
        
        # Storing an element on a container (line 437)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 437, 16), new_378910, (tuple_378911, subscript_call_result_378909))
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        # Getting the type of 'new' (line 438)
        new_378914 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 438, 15), 'new')
        # Assigning a type to the variable 'stypy_return_type' (line 438)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 438, 8), 'stypy_return_type', new_378914)
        
        # ################# End of 'reshape(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'reshape' in the type store
        # Getting the type of 'stypy_return_type' (line 414)
        stypy_return_type_378915 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 414, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_378915)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'reshape'
        return stypy_return_type_378915

    
    # Assigning a Attribute to a Attribute (line 440):

    @norecursion
    def toarray(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 442)
        None_378916 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 442, 28), 'None')
        # Getting the type of 'None' (line 442)
        None_378917 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 442, 38), 'None')
        defaults = [None_378916, None_378917]
        # Create a new context for function 'toarray'
        module_type_store = module_type_store.open_function_context('toarray', 442, 4, False)
        # Assigning a type to the variable 'self' (line 443)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 443, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        lil_matrix.toarray.__dict__.__setitem__('stypy_localization', localization)
        lil_matrix.toarray.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        lil_matrix.toarray.__dict__.__setitem__('stypy_type_store', module_type_store)
        lil_matrix.toarray.__dict__.__setitem__('stypy_function_name', 'lil_matrix.toarray')
        lil_matrix.toarray.__dict__.__setitem__('stypy_param_names_list', ['order', 'out'])
        lil_matrix.toarray.__dict__.__setitem__('stypy_varargs_param_name', None)
        lil_matrix.toarray.__dict__.__setitem__('stypy_kwargs_param_name', None)
        lil_matrix.toarray.__dict__.__setitem__('stypy_call_defaults', defaults)
        lil_matrix.toarray.__dict__.__setitem__('stypy_call_varargs', varargs)
        lil_matrix.toarray.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        lil_matrix.toarray.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'lil_matrix.toarray', ['order', 'out'], None, None, defaults, varargs, kwargs)

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

        
        # Assigning a Call to a Name (line 443):
        
        # Assigning a Call to a Name (line 443):
        
        # Call to _process_toarray_args(...): (line 443)
        # Processing the call arguments (line 443)
        # Getting the type of 'order' (line 443)
        order_378920 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 443, 39), 'order', False)
        # Getting the type of 'out' (line 443)
        out_378921 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 443, 46), 'out', False)
        # Processing the call keyword arguments (line 443)
        kwargs_378922 = {}
        # Getting the type of 'self' (line 443)
        self_378918 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 443, 12), 'self', False)
        # Obtaining the member '_process_toarray_args' of a type (line 443)
        _process_toarray_args_378919 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 443, 12), self_378918, '_process_toarray_args')
        # Calling _process_toarray_args(args, kwargs) (line 443)
        _process_toarray_args_call_result_378923 = invoke(stypy.reporting.localization.Localization(__file__, 443, 12), _process_toarray_args_378919, *[order_378920, out_378921], **kwargs_378922)
        
        # Assigning a type to the variable 'd' (line 443)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 443, 8), 'd', _process_toarray_args_call_result_378923)
        
        
        # Call to enumerate(...): (line 444)
        # Processing the call arguments (line 444)
        # Getting the type of 'self' (line 444)
        self_378925 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 444, 32), 'self', False)
        # Obtaining the member 'rows' of a type (line 444)
        rows_378926 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 444, 32), self_378925, 'rows')
        # Processing the call keyword arguments (line 444)
        kwargs_378927 = {}
        # Getting the type of 'enumerate' (line 444)
        enumerate_378924 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 444, 22), 'enumerate', False)
        # Calling enumerate(args, kwargs) (line 444)
        enumerate_call_result_378928 = invoke(stypy.reporting.localization.Localization(__file__, 444, 22), enumerate_378924, *[rows_378926], **kwargs_378927)
        
        # Testing the type of a for loop iterable (line 444)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 444, 8), enumerate_call_result_378928)
        # Getting the type of the for loop variable (line 444)
        for_loop_var_378929 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 444, 8), enumerate_call_result_378928)
        # Assigning a type to the variable 'i' (line 444)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 444, 8), 'i', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 444, 8), for_loop_var_378929))
        # Assigning a type to the variable 'row' (line 444)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 444, 8), 'row', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 444, 8), for_loop_var_378929))
        # SSA begins for a for statement (line 444)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        
        # Call to enumerate(...): (line 445)
        # Processing the call arguments (line 445)
        # Getting the type of 'row' (line 445)
        row_378931 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 445, 36), 'row', False)
        # Processing the call keyword arguments (line 445)
        kwargs_378932 = {}
        # Getting the type of 'enumerate' (line 445)
        enumerate_378930 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 445, 26), 'enumerate', False)
        # Calling enumerate(args, kwargs) (line 445)
        enumerate_call_result_378933 = invoke(stypy.reporting.localization.Localization(__file__, 445, 26), enumerate_378930, *[row_378931], **kwargs_378932)
        
        # Testing the type of a for loop iterable (line 445)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 445, 12), enumerate_call_result_378933)
        # Getting the type of the for loop variable (line 445)
        for_loop_var_378934 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 445, 12), enumerate_call_result_378933)
        # Assigning a type to the variable 'pos' (line 445)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 445, 12), 'pos', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 445, 12), for_loop_var_378934))
        # Assigning a type to the variable 'j' (line 445)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 445, 12), 'j', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 445, 12), for_loop_var_378934))
        # SSA begins for a for statement (line 445)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Subscript to a Subscript (line 446):
        
        # Assigning a Subscript to a Subscript (line 446):
        
        # Obtaining the type of the subscript
        # Getting the type of 'pos' (line 446)
        pos_378935 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 446, 39), 'pos')
        
        # Obtaining the type of the subscript
        # Getting the type of 'i' (line 446)
        i_378936 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 446, 36), 'i')
        # Getting the type of 'self' (line 446)
        self_378937 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 446, 26), 'self')
        # Obtaining the member 'data' of a type (line 446)
        data_378938 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 446, 26), self_378937, 'data')
        # Obtaining the member '__getitem__' of a type (line 446)
        getitem___378939 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 446, 26), data_378938, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 446)
        subscript_call_result_378940 = invoke(stypy.reporting.localization.Localization(__file__, 446, 26), getitem___378939, i_378936)
        
        # Obtaining the member '__getitem__' of a type (line 446)
        getitem___378941 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 446, 26), subscript_call_result_378940, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 446)
        subscript_call_result_378942 = invoke(stypy.reporting.localization.Localization(__file__, 446, 26), getitem___378941, pos_378935)
        
        # Getting the type of 'd' (line 446)
        d_378943 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 446, 16), 'd')
        
        # Obtaining an instance of the builtin type 'tuple' (line 446)
        tuple_378944 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 446, 18), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 446)
        # Adding element type (line 446)
        # Getting the type of 'i' (line 446)
        i_378945 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 446, 18), 'i')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 446, 18), tuple_378944, i_378945)
        # Adding element type (line 446)
        # Getting the type of 'j' (line 446)
        j_378946 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 446, 21), 'j')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 446, 18), tuple_378944, j_378946)
        
        # Storing an element on a container (line 446)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 446, 16), d_378943, (tuple_378944, subscript_call_result_378942))
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        # Getting the type of 'd' (line 447)
        d_378947 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 447, 15), 'd')
        # Assigning a type to the variable 'stypy_return_type' (line 447)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 447, 8), 'stypy_return_type', d_378947)
        
        # ################# End of 'toarray(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'toarray' in the type store
        # Getting the type of 'stypy_return_type' (line 442)
        stypy_return_type_378948 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 442, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_378948)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'toarray'
        return stypy_return_type_378948

    
    # Assigning a Attribute to a Attribute (line 449):

    @norecursion
    def transpose(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 451)
        None_378949 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 451, 29), 'None')
        # Getting the type of 'False' (line 451)
        False_378950 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 451, 40), 'False')
        defaults = [None_378949, False_378950]
        # Create a new context for function 'transpose'
        module_type_store = module_type_store.open_function_context('transpose', 451, 4, False)
        # Assigning a type to the variable 'self' (line 452)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 452, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        lil_matrix.transpose.__dict__.__setitem__('stypy_localization', localization)
        lil_matrix.transpose.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        lil_matrix.transpose.__dict__.__setitem__('stypy_type_store', module_type_store)
        lil_matrix.transpose.__dict__.__setitem__('stypy_function_name', 'lil_matrix.transpose')
        lil_matrix.transpose.__dict__.__setitem__('stypy_param_names_list', ['axes', 'copy'])
        lil_matrix.transpose.__dict__.__setitem__('stypy_varargs_param_name', None)
        lil_matrix.transpose.__dict__.__setitem__('stypy_kwargs_param_name', None)
        lil_matrix.transpose.__dict__.__setitem__('stypy_call_defaults', defaults)
        lil_matrix.transpose.__dict__.__setitem__('stypy_call_varargs', varargs)
        lil_matrix.transpose.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        lil_matrix.transpose.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'lil_matrix.transpose', ['axes', 'copy'], None, None, defaults, varargs, kwargs)

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

        
        # Call to tolil(...): (line 452)
        # Processing the call keyword arguments (line 452)
        kwargs_378963 = {}
        
        # Call to transpose(...): (line 452)
        # Processing the call keyword arguments (line 452)
        # Getting the type of 'axes' (line 452)
        axes_378956 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 452, 43), 'axes', False)
        keyword_378957 = axes_378956
        # Getting the type of 'copy' (line 452)
        copy_378958 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 452, 54), 'copy', False)
        keyword_378959 = copy_378958
        kwargs_378960 = {'copy': keyword_378959, 'axes': keyword_378957}
        
        # Call to tocsr(...): (line 452)
        # Processing the call keyword arguments (line 452)
        kwargs_378953 = {}
        # Getting the type of 'self' (line 452)
        self_378951 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 452, 15), 'self', False)
        # Obtaining the member 'tocsr' of a type (line 452)
        tocsr_378952 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 452, 15), self_378951, 'tocsr')
        # Calling tocsr(args, kwargs) (line 452)
        tocsr_call_result_378954 = invoke(stypy.reporting.localization.Localization(__file__, 452, 15), tocsr_378952, *[], **kwargs_378953)
        
        # Obtaining the member 'transpose' of a type (line 452)
        transpose_378955 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 452, 15), tocsr_call_result_378954, 'transpose')
        # Calling transpose(args, kwargs) (line 452)
        transpose_call_result_378961 = invoke(stypy.reporting.localization.Localization(__file__, 452, 15), transpose_378955, *[], **kwargs_378960)
        
        # Obtaining the member 'tolil' of a type (line 452)
        tolil_378962 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 452, 15), transpose_call_result_378961, 'tolil')
        # Calling tolil(args, kwargs) (line 452)
        tolil_call_result_378964 = invoke(stypy.reporting.localization.Localization(__file__, 452, 15), tolil_378962, *[], **kwargs_378963)
        
        # Assigning a type to the variable 'stypy_return_type' (line 452)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 452, 8), 'stypy_return_type', tolil_call_result_378964)
        
        # ################# End of 'transpose(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'transpose' in the type store
        # Getting the type of 'stypy_return_type' (line 451)
        stypy_return_type_378965 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 451, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_378965)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'transpose'
        return stypy_return_type_378965

    
    # Assigning a Attribute to a Attribute (line 454):

    @norecursion
    def tolil(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'False' (line 456)
        False_378966 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 456, 25), 'False')
        defaults = [False_378966]
        # Create a new context for function 'tolil'
        module_type_store = module_type_store.open_function_context('tolil', 456, 4, False)
        # Assigning a type to the variable 'self' (line 457)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 457, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        lil_matrix.tolil.__dict__.__setitem__('stypy_localization', localization)
        lil_matrix.tolil.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        lil_matrix.tolil.__dict__.__setitem__('stypy_type_store', module_type_store)
        lil_matrix.tolil.__dict__.__setitem__('stypy_function_name', 'lil_matrix.tolil')
        lil_matrix.tolil.__dict__.__setitem__('stypy_param_names_list', ['copy'])
        lil_matrix.tolil.__dict__.__setitem__('stypy_varargs_param_name', None)
        lil_matrix.tolil.__dict__.__setitem__('stypy_kwargs_param_name', None)
        lil_matrix.tolil.__dict__.__setitem__('stypy_call_defaults', defaults)
        lil_matrix.tolil.__dict__.__setitem__('stypy_call_varargs', varargs)
        lil_matrix.tolil.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        lil_matrix.tolil.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'lil_matrix.tolil', ['copy'], None, None, defaults, varargs, kwargs)

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

        
        # Getting the type of 'copy' (line 457)
        copy_378967 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 457, 11), 'copy')
        # Testing the type of an if condition (line 457)
        if_condition_378968 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 457, 8), copy_378967)
        # Assigning a type to the variable 'if_condition_378968' (line 457)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 457, 8), 'if_condition_378968', if_condition_378968)
        # SSA begins for if statement (line 457)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to copy(...): (line 458)
        # Processing the call keyword arguments (line 458)
        kwargs_378971 = {}
        # Getting the type of 'self' (line 458)
        self_378969 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 458, 19), 'self', False)
        # Obtaining the member 'copy' of a type (line 458)
        copy_378970 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 458, 19), self_378969, 'copy')
        # Calling copy(args, kwargs) (line 458)
        copy_call_result_378972 = invoke(stypy.reporting.localization.Localization(__file__, 458, 19), copy_378970, *[], **kwargs_378971)
        
        # Assigning a type to the variable 'stypy_return_type' (line 458)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 458, 12), 'stypy_return_type', copy_call_result_378972)
        # SSA branch for the else part of an if statement (line 457)
        module_type_store.open_ssa_branch('else')
        # Getting the type of 'self' (line 460)
        self_378973 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 460, 19), 'self')
        # Assigning a type to the variable 'stypy_return_type' (line 460)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 460, 12), 'stypy_return_type', self_378973)
        # SSA join for if statement (line 457)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'tolil(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'tolil' in the type store
        # Getting the type of 'stypy_return_type' (line 456)
        stypy_return_type_378974 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 456, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_378974)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'tolil'
        return stypy_return_type_378974

    
    # Assigning a Attribute to a Attribute (line 462):

    @norecursion
    def tocsr(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'False' (line 464)
        False_378975 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 464, 25), 'False')
        defaults = [False_378975]
        # Create a new context for function 'tocsr'
        module_type_store = module_type_store.open_function_context('tocsr', 464, 4, False)
        # Assigning a type to the variable 'self' (line 465)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 465, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        lil_matrix.tocsr.__dict__.__setitem__('stypy_localization', localization)
        lil_matrix.tocsr.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        lil_matrix.tocsr.__dict__.__setitem__('stypy_type_store', module_type_store)
        lil_matrix.tocsr.__dict__.__setitem__('stypy_function_name', 'lil_matrix.tocsr')
        lil_matrix.tocsr.__dict__.__setitem__('stypy_param_names_list', ['copy'])
        lil_matrix.tocsr.__dict__.__setitem__('stypy_varargs_param_name', None)
        lil_matrix.tocsr.__dict__.__setitem__('stypy_kwargs_param_name', None)
        lil_matrix.tocsr.__dict__.__setitem__('stypy_call_defaults', defaults)
        lil_matrix.tocsr.__dict__.__setitem__('stypy_call_varargs', varargs)
        lil_matrix.tocsr.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        lil_matrix.tocsr.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'lil_matrix.tocsr', ['copy'], None, None, defaults, varargs, kwargs)

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

        
        # Assigning a ListComp to a Name (line 465):
        
        # Assigning a ListComp to a Name (line 465):
        # Calculating list comprehension
        # Calculating comprehension expression
        # Getting the type of 'self' (line 465)
        self_378980 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 465, 31), 'self')
        # Obtaining the member 'rows' of a type (line 465)
        rows_378981 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 465, 31), self_378980, 'rows')
        comprehension_378982 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 465, 15), rows_378981)
        # Assigning a type to the variable 'x' (line 465)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 465, 15), 'x', comprehension_378982)
        
        # Call to len(...): (line 465)
        # Processing the call arguments (line 465)
        # Getting the type of 'x' (line 465)
        x_378977 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 465, 19), 'x', False)
        # Processing the call keyword arguments (line 465)
        kwargs_378978 = {}
        # Getting the type of 'len' (line 465)
        len_378976 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 465, 15), 'len', False)
        # Calling len(args, kwargs) (line 465)
        len_call_result_378979 = invoke(stypy.reporting.localization.Localization(__file__, 465, 15), len_378976, *[x_378977], **kwargs_378978)
        
        list_378983 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 465, 15), 'list')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 465, 15), list_378983, len_call_result_378979)
        # Assigning a type to the variable 'lst' (line 465)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 465, 8), 'lst', list_378983)
        
        # Assigning a Call to a Name (line 466):
        
        # Assigning a Call to a Name (line 466):
        
        # Call to get_index_dtype(...): (line 466)
        # Processing the call keyword arguments (line 466)
        
        # Call to max(...): (line 466)
        # Processing the call arguments (line 466)
        
        # Obtaining the type of the subscript
        int_378986 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 466, 58), 'int')
        # Getting the type of 'self' (line 466)
        self_378987 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 466, 47), 'self', False)
        # Obtaining the member 'shape' of a type (line 466)
        shape_378988 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 466, 47), self_378987, 'shape')
        # Obtaining the member '__getitem__' of a type (line 466)
        getitem___378989 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 466, 47), shape_378988, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 466)
        subscript_call_result_378990 = invoke(stypy.reporting.localization.Localization(__file__, 466, 47), getitem___378989, int_378986)
        
        
        # Call to sum(...): (line 466)
        # Processing the call arguments (line 466)
        # Getting the type of 'lst' (line 466)
        lst_378992 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 466, 66), 'lst', False)
        # Processing the call keyword arguments (line 466)
        kwargs_378993 = {}
        # Getting the type of 'sum' (line 466)
        sum_378991 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 466, 62), 'sum', False)
        # Calling sum(args, kwargs) (line 466)
        sum_call_result_378994 = invoke(stypy.reporting.localization.Localization(__file__, 466, 62), sum_378991, *[lst_378992], **kwargs_378993)
        
        # Processing the call keyword arguments (line 466)
        kwargs_378995 = {}
        # Getting the type of 'max' (line 466)
        max_378985 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 466, 43), 'max', False)
        # Calling max(args, kwargs) (line 466)
        max_call_result_378996 = invoke(stypy.reporting.localization.Localization(__file__, 466, 43), max_378985, *[subscript_call_result_378990, sum_call_result_378994], **kwargs_378995)
        
        keyword_378997 = max_call_result_378996
        kwargs_378998 = {'maxval': keyword_378997}
        # Getting the type of 'get_index_dtype' (line 466)
        get_index_dtype_378984 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 466, 20), 'get_index_dtype', False)
        # Calling get_index_dtype(args, kwargs) (line 466)
        get_index_dtype_call_result_378999 = invoke(stypy.reporting.localization.Localization(__file__, 466, 20), get_index_dtype_378984, *[], **kwargs_378998)
        
        # Assigning a type to the variable 'idx_dtype' (line 466)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 466, 8), 'idx_dtype', get_index_dtype_call_result_378999)
        
        # Assigning a Call to a Name (line 467):
        
        # Assigning a Call to a Name (line 467):
        
        # Call to asarray(...): (line 467)
        # Processing the call arguments (line 467)
        # Getting the type of 'lst' (line 467)
        lst_379002 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 467, 28), 'lst', False)
        # Processing the call keyword arguments (line 467)
        # Getting the type of 'idx_dtype' (line 467)
        idx_dtype_379003 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 467, 39), 'idx_dtype', False)
        keyword_379004 = idx_dtype_379003
        kwargs_379005 = {'dtype': keyword_379004}
        # Getting the type of 'np' (line 467)
        np_379000 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 467, 17), 'np', False)
        # Obtaining the member 'asarray' of a type (line 467)
        asarray_379001 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 467, 17), np_379000, 'asarray')
        # Calling asarray(args, kwargs) (line 467)
        asarray_call_result_379006 = invoke(stypy.reporting.localization.Localization(__file__, 467, 17), asarray_379001, *[lst_379002], **kwargs_379005)
        
        # Assigning a type to the variable 'indptr' (line 467)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 467, 8), 'indptr', asarray_call_result_379006)
        
        # Assigning a Call to a Name (line 468):
        
        # Assigning a Call to a Name (line 468):
        
        # Call to concatenate(...): (line 468)
        # Processing the call arguments (line 468)
        
        # Obtaining an instance of the builtin type 'tuple' (line 468)
        tuple_379009 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 468, 33), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 468)
        # Adding element type (line 468)
        
        # Call to array(...): (line 468)
        # Processing the call arguments (line 468)
        
        # Obtaining an instance of the builtin type 'list' (line 468)
        list_379012 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 468, 42), 'list')
        # Adding type elements to the builtin type 'list' instance (line 468)
        # Adding element type (line 468)
        int_379013 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 468, 43), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 468, 42), list_379012, int_379013)
        
        # Processing the call keyword arguments (line 468)
        # Getting the type of 'idx_dtype' (line 468)
        idx_dtype_379014 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 468, 53), 'idx_dtype', False)
        keyword_379015 = idx_dtype_379014
        kwargs_379016 = {'dtype': keyword_379015}
        # Getting the type of 'np' (line 468)
        np_379010 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 468, 33), 'np', False)
        # Obtaining the member 'array' of a type (line 468)
        array_379011 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 468, 33), np_379010, 'array')
        # Calling array(args, kwargs) (line 468)
        array_call_result_379017 = invoke(stypy.reporting.localization.Localization(__file__, 468, 33), array_379011, *[list_379012], **kwargs_379016)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 468, 33), tuple_379009, array_call_result_379017)
        # Adding element type (line 468)
        
        # Call to cumsum(...): (line 469)
        # Processing the call arguments (line 469)
        # Getting the type of 'indptr' (line 469)
        indptr_379020 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 469, 43), 'indptr', False)
        # Processing the call keyword arguments (line 469)
        # Getting the type of 'idx_dtype' (line 469)
        idx_dtype_379021 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 469, 57), 'idx_dtype', False)
        keyword_379022 = idx_dtype_379021
        kwargs_379023 = {'dtype': keyword_379022}
        # Getting the type of 'np' (line 469)
        np_379018 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 469, 33), 'np', False)
        # Obtaining the member 'cumsum' of a type (line 469)
        cumsum_379019 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 469, 33), np_379018, 'cumsum')
        # Calling cumsum(args, kwargs) (line 469)
        cumsum_call_result_379024 = invoke(stypy.reporting.localization.Localization(__file__, 469, 33), cumsum_379019, *[indptr_379020], **kwargs_379023)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 468, 33), tuple_379009, cumsum_call_result_379024)
        
        # Processing the call keyword arguments (line 468)
        kwargs_379025 = {}
        # Getting the type of 'np' (line 468)
        np_379007 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 468, 17), 'np', False)
        # Obtaining the member 'concatenate' of a type (line 468)
        concatenate_379008 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 468, 17), np_379007, 'concatenate')
        # Calling concatenate(args, kwargs) (line 468)
        concatenate_call_result_379026 = invoke(stypy.reporting.localization.Localization(__file__, 468, 17), concatenate_379008, *[tuple_379009], **kwargs_379025)
        
        # Assigning a type to the variable 'indptr' (line 468)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 468, 8), 'indptr', concatenate_call_result_379026)
        
        # Assigning a List to a Name (line 471):
        
        # Assigning a List to a Name (line 471):
        
        # Obtaining an instance of the builtin type 'list' (line 471)
        list_379027 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 471, 18), 'list')
        # Adding type elements to the builtin type 'list' instance (line 471)
        
        # Assigning a type to the variable 'indices' (line 471)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 471, 8), 'indices', list_379027)
        
        # Getting the type of 'self' (line 472)
        self_379028 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 472, 17), 'self')
        # Obtaining the member 'rows' of a type (line 472)
        rows_379029 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 472, 17), self_379028, 'rows')
        # Testing the type of a for loop iterable (line 472)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 472, 8), rows_379029)
        # Getting the type of the for loop variable (line 472)
        for_loop_var_379030 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 472, 8), rows_379029)
        # Assigning a type to the variable 'x' (line 472)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 472, 8), 'x', for_loop_var_379030)
        # SSA begins for a for statement (line 472)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Call to extend(...): (line 473)
        # Processing the call arguments (line 473)
        # Getting the type of 'x' (line 473)
        x_379033 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 473, 27), 'x', False)
        # Processing the call keyword arguments (line 473)
        kwargs_379034 = {}
        # Getting the type of 'indices' (line 473)
        indices_379031 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 473, 12), 'indices', False)
        # Obtaining the member 'extend' of a type (line 473)
        extend_379032 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 473, 12), indices_379031, 'extend')
        # Calling extend(args, kwargs) (line 473)
        extend_call_result_379035 = invoke(stypy.reporting.localization.Localization(__file__, 473, 12), extend_379032, *[x_379033], **kwargs_379034)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Name (line 474):
        
        # Assigning a Call to a Name (line 474):
        
        # Call to asarray(...): (line 474)
        # Processing the call arguments (line 474)
        # Getting the type of 'indices' (line 474)
        indices_379038 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 474, 29), 'indices', False)
        # Processing the call keyword arguments (line 474)
        # Getting the type of 'idx_dtype' (line 474)
        idx_dtype_379039 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 474, 44), 'idx_dtype', False)
        keyword_379040 = idx_dtype_379039
        kwargs_379041 = {'dtype': keyword_379040}
        # Getting the type of 'np' (line 474)
        np_379036 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 474, 18), 'np', False)
        # Obtaining the member 'asarray' of a type (line 474)
        asarray_379037 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 474, 18), np_379036, 'asarray')
        # Calling asarray(args, kwargs) (line 474)
        asarray_call_result_379042 = invoke(stypy.reporting.localization.Localization(__file__, 474, 18), asarray_379037, *[indices_379038], **kwargs_379041)
        
        # Assigning a type to the variable 'indices' (line 474)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 474, 8), 'indices', asarray_call_result_379042)
        
        # Assigning a List to a Name (line 476):
        
        # Assigning a List to a Name (line 476):
        
        # Obtaining an instance of the builtin type 'list' (line 476)
        list_379043 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 476, 15), 'list')
        # Adding type elements to the builtin type 'list' instance (line 476)
        
        # Assigning a type to the variable 'data' (line 476)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 476, 8), 'data', list_379043)
        
        # Getting the type of 'self' (line 477)
        self_379044 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 477, 17), 'self')
        # Obtaining the member 'data' of a type (line 477)
        data_379045 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 477, 17), self_379044, 'data')
        # Testing the type of a for loop iterable (line 477)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 477, 8), data_379045)
        # Getting the type of the for loop variable (line 477)
        for_loop_var_379046 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 477, 8), data_379045)
        # Assigning a type to the variable 'x' (line 477)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 477, 8), 'x', for_loop_var_379046)
        # SSA begins for a for statement (line 477)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Call to extend(...): (line 478)
        # Processing the call arguments (line 478)
        # Getting the type of 'x' (line 478)
        x_379049 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 478, 24), 'x', False)
        # Processing the call keyword arguments (line 478)
        kwargs_379050 = {}
        # Getting the type of 'data' (line 478)
        data_379047 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 478, 12), 'data', False)
        # Obtaining the member 'extend' of a type (line 478)
        extend_379048 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 478, 12), data_379047, 'extend')
        # Calling extend(args, kwargs) (line 478)
        extend_call_result_379051 = invoke(stypy.reporting.localization.Localization(__file__, 478, 12), extend_379048, *[x_379049], **kwargs_379050)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Name (line 479):
        
        # Assigning a Call to a Name (line 479):
        
        # Call to asarray(...): (line 479)
        # Processing the call arguments (line 479)
        # Getting the type of 'data' (line 479)
        data_379054 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 479, 26), 'data', False)
        # Processing the call keyword arguments (line 479)
        # Getting the type of 'self' (line 479)
        self_379055 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 479, 38), 'self', False)
        # Obtaining the member 'dtype' of a type (line 479)
        dtype_379056 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 479, 38), self_379055, 'dtype')
        keyword_379057 = dtype_379056
        kwargs_379058 = {'dtype': keyword_379057}
        # Getting the type of 'np' (line 479)
        np_379052 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 479, 15), 'np', False)
        # Obtaining the member 'asarray' of a type (line 479)
        asarray_379053 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 479, 15), np_379052, 'asarray')
        # Calling asarray(args, kwargs) (line 479)
        asarray_call_result_379059 = invoke(stypy.reporting.localization.Localization(__file__, 479, 15), asarray_379053, *[data_379054], **kwargs_379058)
        
        # Assigning a type to the variable 'data' (line 479)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 479, 8), 'data', asarray_call_result_379059)
        stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 481, 8))
        
        # 'from scipy.sparse.csr import csr_matrix' statement (line 481)
        update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/sparse/')
        import_379060 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 481, 8), 'scipy.sparse.csr')

        if (type(import_379060) is not StypyTypeError):

            if (import_379060 != 'pyd_module'):
                __import__(import_379060)
                sys_modules_379061 = sys.modules[import_379060]
                import_from_module(stypy.reporting.localization.Localization(__file__, 481, 8), 'scipy.sparse.csr', sys_modules_379061.module_type_store, module_type_store, ['csr_matrix'])
                nest_module(stypy.reporting.localization.Localization(__file__, 481, 8), __file__, sys_modules_379061, sys_modules_379061.module_type_store, module_type_store)
            else:
                from scipy.sparse.csr import csr_matrix

                import_from_module(stypy.reporting.localization.Localization(__file__, 481, 8), 'scipy.sparse.csr', None, module_type_store, ['csr_matrix'], [csr_matrix])

        else:
            # Assigning a type to the variable 'scipy.sparse.csr' (line 481)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 481, 8), 'scipy.sparse.csr', import_379060)

        remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/sparse/')
        
        
        # Call to csr_matrix(...): (line 482)
        # Processing the call arguments (line 482)
        
        # Obtaining an instance of the builtin type 'tuple' (line 482)
        tuple_379063 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 482, 27), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 482)
        # Adding element type (line 482)
        # Getting the type of 'data' (line 482)
        data_379064 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 482, 27), 'data', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 482, 27), tuple_379063, data_379064)
        # Adding element type (line 482)
        # Getting the type of 'indices' (line 482)
        indices_379065 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 482, 33), 'indices', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 482, 27), tuple_379063, indices_379065)
        # Adding element type (line 482)
        # Getting the type of 'indptr' (line 482)
        indptr_379066 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 482, 42), 'indptr', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 482, 27), tuple_379063, indptr_379066)
        
        # Processing the call keyword arguments (line 482)
        # Getting the type of 'self' (line 482)
        self_379067 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 482, 57), 'self', False)
        # Obtaining the member 'shape' of a type (line 482)
        shape_379068 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 482, 57), self_379067, 'shape')
        keyword_379069 = shape_379068
        kwargs_379070 = {'shape': keyword_379069}
        # Getting the type of 'csr_matrix' (line 482)
        csr_matrix_379062 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 482, 15), 'csr_matrix', False)
        # Calling csr_matrix(args, kwargs) (line 482)
        csr_matrix_call_result_379071 = invoke(stypy.reporting.localization.Localization(__file__, 482, 15), csr_matrix_379062, *[tuple_379063], **kwargs_379070)
        
        # Assigning a type to the variable 'stypy_return_type' (line 482)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 482, 8), 'stypy_return_type', csr_matrix_call_result_379071)
        
        # ################# End of 'tocsr(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'tocsr' in the type store
        # Getting the type of 'stypy_return_type' (line 464)
        stypy_return_type_379072 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 464, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_379072)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'tocsr'
        return stypy_return_type_379072

    
    # Assigning a Attribute to a Attribute (line 484):

# Assigning a type to the variable 'lil_matrix' (line 19)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 19, 0), 'lil_matrix', lil_matrix)

# Assigning a Str to a Name (line 82):
str_379073 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 82, 13), 'str', 'lil')
# Getting the type of 'lil_matrix'
lil_matrix_379074 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'lil_matrix')
# Setting the type of the member 'format' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), lil_matrix_379074, 'format', str_379073)

# Assigning a Attribute to a Attribute (line 152):
# Getting the type of 'spmatrix' (line 152)
spmatrix_379075 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 24), 'spmatrix')
# Obtaining the member 'set_shape' of a type (line 152)
set_shape_379076 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 152, 24), spmatrix_379075, 'set_shape')
# Obtaining the member '__doc__' of a type (line 152)
doc___379077 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 152, 24), set_shape_379076, '__doc__')
# Getting the type of 'lil_matrix'
lil_matrix_379078 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'lil_matrix')
# Obtaining the member 'set_shape' of a type
set_shape_379079 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), lil_matrix_379078, 'set_shape')
# Setting the type of the member '__doc__' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), set_shape_379079, '__doc__', doc___379077)

# Assigning a Call to a Name (line 154):

# Call to property(...): (line 154)
# Processing the call keyword arguments (line 154)
# Getting the type of 'spmatrix' (line 154)
spmatrix_379081 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 26), 'spmatrix', False)
# Obtaining the member 'get_shape' of a type (line 154)
get_shape_379082 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 154, 26), spmatrix_379081, 'get_shape')
keyword_379083 = get_shape_379082
# Getting the type of 'set_shape' (line 154)
set_shape_379084 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 51), 'set_shape', False)
keyword_379085 = set_shape_379084
kwargs_379086 = {'fset': keyword_379085, 'fget': keyword_379083}
# Getting the type of 'property' (line 154)
property_379080 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 12), 'property', False)
# Calling property(args, kwargs) (line 154)
property_call_result_379087 = invoke(stypy.reporting.localization.Localization(__file__, 154, 12), property_379080, *[], **kwargs_379086)

# Getting the type of 'lil_matrix'
lil_matrix_379088 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'lil_matrix')
# Setting the type of the member 'shape' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), lil_matrix_379088, 'shape', property_call_result_379087)

# Assigning a Attribute to a Attribute (line 199):
# Getting the type of 'spmatrix' (line 199)
spmatrix_379089 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 21), 'spmatrix')
# Obtaining the member 'getnnz' of a type (line 199)
getnnz_379090 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 199, 21), spmatrix_379089, 'getnnz')
# Obtaining the member '__doc__' of a type (line 199)
doc___379091 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 199, 21), getnnz_379090, '__doc__')
# Getting the type of 'lil_matrix'
lil_matrix_379092 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'lil_matrix')
# Obtaining the member 'getnnz' of a type
getnnz_379093 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), lil_matrix_379092, 'getnnz')
# Setting the type of the member '__doc__' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), getnnz_379093, '__doc__', doc___379091)

# Assigning a Attribute to a Attribute (line 200):
# Getting the type of 'spmatrix' (line 200)
spmatrix_379094 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 28), 'spmatrix')
# Obtaining the member 'count_nonzero' of a type (line 200)
count_nonzero_379095 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 200, 28), spmatrix_379094, 'count_nonzero')
# Obtaining the member '__doc__' of a type (line 200)
doc___379096 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 200, 28), count_nonzero_379095, '__doc__')
# Getting the type of 'lil_matrix'
lil_matrix_379097 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'lil_matrix')
# Obtaining the member 'count_nonzero' of a type
count_nonzero_379098 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), lil_matrix_379097, 'count_nonzero')
# Setting the type of the member '__doc__' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), count_nonzero_379098, '__doc__', doc___379096)

# Assigning a Attribute to a Attribute (line 412):
# Getting the type of 'spmatrix' (line 412)
spmatrix_379099 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 412, 19), 'spmatrix')
# Obtaining the member 'copy' of a type (line 412)
copy_379100 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 412, 19), spmatrix_379099, 'copy')
# Obtaining the member '__doc__' of a type (line 412)
doc___379101 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 412, 19), copy_379100, '__doc__')
# Getting the type of 'lil_matrix'
lil_matrix_379102 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'lil_matrix')
# Obtaining the member 'copy' of a type
copy_379103 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), lil_matrix_379102, 'copy')
# Setting the type of the member '__doc__' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), copy_379103, '__doc__', doc___379101)

# Assigning a Attribute to a Attribute (line 440):
# Getting the type of 'spmatrix' (line 440)
spmatrix_379104 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 440, 22), 'spmatrix')
# Obtaining the member 'reshape' of a type (line 440)
reshape_379105 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 440, 22), spmatrix_379104, 'reshape')
# Obtaining the member '__doc__' of a type (line 440)
doc___379106 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 440, 22), reshape_379105, '__doc__')
# Getting the type of 'lil_matrix'
lil_matrix_379107 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'lil_matrix')
# Obtaining the member 'reshape' of a type
reshape_379108 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), lil_matrix_379107, 'reshape')
# Setting the type of the member '__doc__' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), reshape_379108, '__doc__', doc___379106)

# Assigning a Attribute to a Attribute (line 449):
# Getting the type of 'spmatrix' (line 449)
spmatrix_379109 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 449, 22), 'spmatrix')
# Obtaining the member 'toarray' of a type (line 449)
toarray_379110 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 449, 22), spmatrix_379109, 'toarray')
# Obtaining the member '__doc__' of a type (line 449)
doc___379111 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 449, 22), toarray_379110, '__doc__')
# Getting the type of 'lil_matrix'
lil_matrix_379112 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'lil_matrix')
# Obtaining the member 'toarray' of a type
toarray_379113 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), lil_matrix_379112, 'toarray')
# Setting the type of the member '__doc__' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), toarray_379113, '__doc__', doc___379111)

# Assigning a Attribute to a Attribute (line 454):
# Getting the type of 'spmatrix' (line 454)
spmatrix_379114 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 454, 24), 'spmatrix')
# Obtaining the member 'transpose' of a type (line 454)
transpose_379115 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 454, 24), spmatrix_379114, 'transpose')
# Obtaining the member '__doc__' of a type (line 454)
doc___379116 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 454, 24), transpose_379115, '__doc__')
# Getting the type of 'lil_matrix'
lil_matrix_379117 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'lil_matrix')
# Obtaining the member 'transpose' of a type
transpose_379118 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), lil_matrix_379117, 'transpose')
# Setting the type of the member '__doc__' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), transpose_379118, '__doc__', doc___379116)

# Assigning a Attribute to a Attribute (line 462):
# Getting the type of 'spmatrix' (line 462)
spmatrix_379119 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 462, 20), 'spmatrix')
# Obtaining the member 'tolil' of a type (line 462)
tolil_379120 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 462, 20), spmatrix_379119, 'tolil')
# Obtaining the member '__doc__' of a type (line 462)
doc___379121 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 462, 20), tolil_379120, '__doc__')
# Getting the type of 'lil_matrix'
lil_matrix_379122 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'lil_matrix')
# Obtaining the member 'tolil' of a type
tolil_379123 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), lil_matrix_379122, 'tolil')
# Setting the type of the member '__doc__' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), tolil_379123, '__doc__', doc___379121)

# Assigning a Attribute to a Attribute (line 484):
# Getting the type of 'spmatrix' (line 484)
spmatrix_379124 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 484, 20), 'spmatrix')
# Obtaining the member 'tocsr' of a type (line 484)
tocsr_379125 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 484, 20), spmatrix_379124, 'tocsr')
# Obtaining the member '__doc__' of a type (line 484)
doc___379126 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 484, 20), tocsr_379125, '__doc__')
# Getting the type of 'lil_matrix'
lil_matrix_379127 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'lil_matrix')
# Obtaining the member 'tocsr' of a type
tocsr_379128 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), lil_matrix_379127, 'tocsr')
# Setting the type of the member '__doc__' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), tocsr_379128, '__doc__', doc___379126)

@norecursion
def _prepare_index_for_memoryview(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'None' (line 487)
    None_379129 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 487, 42), 'None')
    defaults = [None_379129]
    # Create a new context for function '_prepare_index_for_memoryview'
    module_type_store = module_type_store.open_function_context('_prepare_index_for_memoryview', 487, 0, False)
    
    # Passed parameters checking function
    _prepare_index_for_memoryview.stypy_localization = localization
    _prepare_index_for_memoryview.stypy_type_of_self = None
    _prepare_index_for_memoryview.stypy_type_store = module_type_store
    _prepare_index_for_memoryview.stypy_function_name = '_prepare_index_for_memoryview'
    _prepare_index_for_memoryview.stypy_param_names_list = ['i', 'j', 'x']
    _prepare_index_for_memoryview.stypy_varargs_param_name = None
    _prepare_index_for_memoryview.stypy_kwargs_param_name = None
    _prepare_index_for_memoryview.stypy_call_defaults = defaults
    _prepare_index_for_memoryview.stypy_call_varargs = varargs
    _prepare_index_for_memoryview.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_prepare_index_for_memoryview', ['i', 'j', 'x'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_prepare_index_for_memoryview', localization, ['i', 'j', 'x'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_prepare_index_for_memoryview(...)' code ##################

    str_379130 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 509, (-1)), 'str', "\n    Convert index and data arrays to form suitable for passing to the\n    Cython fancy getset routines.\n\n    The conversions are necessary since to (i) ensure the integer\n    index arrays are in one of the accepted types, and (ii) to ensure\n    the arrays are writable so that Cython memoryview support doesn't\n    choke on them.\n\n    Parameters\n    ----------\n    i, j\n        Index arrays\n    x : optional\n        Data arrays\n\n    Returns\n    -------\n    i, j, x\n        Re-formatted arrays (x is omitted, if input was None)\n\n    ")
    
    
    # Getting the type of 'i' (line 510)
    i_379131 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 510, 7), 'i')
    # Obtaining the member 'dtype' of a type (line 510)
    dtype_379132 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 510, 7), i_379131, 'dtype')
    # Getting the type of 'j' (line 510)
    j_379133 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 510, 17), 'j')
    # Obtaining the member 'dtype' of a type (line 510)
    dtype_379134 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 510, 17), j_379133, 'dtype')
    # Applying the binary operator '>' (line 510)
    result_gt_379135 = python_operator(stypy.reporting.localization.Localization(__file__, 510, 7), '>', dtype_379132, dtype_379134)
    
    # Testing the type of an if condition (line 510)
    if_condition_379136 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 510, 4), result_gt_379135)
    # Assigning a type to the variable 'if_condition_379136' (line 510)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 510, 4), 'if_condition_379136', if_condition_379136)
    # SSA begins for if statement (line 510)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 511):
    
    # Assigning a Call to a Name (line 511):
    
    # Call to astype(...): (line 511)
    # Processing the call arguments (line 511)
    # Getting the type of 'i' (line 511)
    i_379139 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 511, 21), 'i', False)
    # Obtaining the member 'dtype' of a type (line 511)
    dtype_379140 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 511, 21), i_379139, 'dtype')
    # Processing the call keyword arguments (line 511)
    kwargs_379141 = {}
    # Getting the type of 'j' (line 511)
    j_379137 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 511, 12), 'j', False)
    # Obtaining the member 'astype' of a type (line 511)
    astype_379138 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 511, 12), j_379137, 'astype')
    # Calling astype(args, kwargs) (line 511)
    astype_call_result_379142 = invoke(stypy.reporting.localization.Localization(__file__, 511, 12), astype_379138, *[dtype_379140], **kwargs_379141)
    
    # Assigning a type to the variable 'j' (line 511)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 511, 8), 'j', astype_call_result_379142)
    # SSA branch for the else part of an if statement (line 510)
    module_type_store.open_ssa_branch('else')
    
    
    # Getting the type of 'i' (line 512)
    i_379143 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 512, 9), 'i')
    # Obtaining the member 'dtype' of a type (line 512)
    dtype_379144 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 512, 9), i_379143, 'dtype')
    # Getting the type of 'j' (line 512)
    j_379145 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 512, 19), 'j')
    # Obtaining the member 'dtype' of a type (line 512)
    dtype_379146 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 512, 19), j_379145, 'dtype')
    # Applying the binary operator '<' (line 512)
    result_lt_379147 = python_operator(stypy.reporting.localization.Localization(__file__, 512, 9), '<', dtype_379144, dtype_379146)
    
    # Testing the type of an if condition (line 512)
    if_condition_379148 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 512, 9), result_lt_379147)
    # Assigning a type to the variable 'if_condition_379148' (line 512)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 512, 9), 'if_condition_379148', if_condition_379148)
    # SSA begins for if statement (line 512)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 513):
    
    # Assigning a Call to a Name (line 513):
    
    # Call to astype(...): (line 513)
    # Processing the call arguments (line 513)
    # Getting the type of 'j' (line 513)
    j_379151 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 513, 21), 'j', False)
    # Obtaining the member 'dtype' of a type (line 513)
    dtype_379152 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 513, 21), j_379151, 'dtype')
    # Processing the call keyword arguments (line 513)
    kwargs_379153 = {}
    # Getting the type of 'i' (line 513)
    i_379149 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 513, 12), 'i', False)
    # Obtaining the member 'astype' of a type (line 513)
    astype_379150 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 513, 12), i_379149, 'astype')
    # Calling astype(args, kwargs) (line 513)
    astype_call_result_379154 = invoke(stypy.reporting.localization.Localization(__file__, 513, 12), astype_379150, *[dtype_379152], **kwargs_379153)
    
    # Assigning a type to the variable 'i' (line 513)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 513, 8), 'i', astype_call_result_379154)
    # SSA join for if statement (line 512)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 510)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'i' (line 515)
    i_379155 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 515, 11), 'i')
    # Obtaining the member 'flags' of a type (line 515)
    flags_379156 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 515, 11), i_379155, 'flags')
    # Obtaining the member 'writeable' of a type (line 515)
    writeable_379157 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 515, 11), flags_379156, 'writeable')
    # Applying the 'not' unary operator (line 515)
    result_not__379158 = python_operator(stypy.reporting.localization.Localization(__file__, 515, 7), 'not', writeable_379157)
    
    
    # Getting the type of 'i' (line 515)
    i_379159 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 515, 32), 'i')
    # Obtaining the member 'dtype' of a type (line 515)
    dtype_379160 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 515, 32), i_379159, 'dtype')
    
    # Obtaining an instance of the builtin type 'tuple' (line 515)
    tuple_379161 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 515, 48), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 515)
    # Adding element type (line 515)
    # Getting the type of 'np' (line 515)
    np_379162 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 515, 48), 'np')
    # Obtaining the member 'int32' of a type (line 515)
    int32_379163 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 515, 48), np_379162, 'int32')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 515, 48), tuple_379161, int32_379163)
    # Adding element type (line 515)
    # Getting the type of 'np' (line 515)
    np_379164 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 515, 58), 'np')
    # Obtaining the member 'int64' of a type (line 515)
    int64_379165 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 515, 58), np_379164, 'int64')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 515, 48), tuple_379161, int64_379165)
    
    # Applying the binary operator 'notin' (line 515)
    result_contains_379166 = python_operator(stypy.reporting.localization.Localization(__file__, 515, 32), 'notin', dtype_379160, tuple_379161)
    
    # Applying the binary operator 'or' (line 515)
    result_or_keyword_379167 = python_operator(stypy.reporting.localization.Localization(__file__, 515, 7), 'or', result_not__379158, result_contains_379166)
    
    # Testing the type of an if condition (line 515)
    if_condition_379168 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 515, 4), result_or_keyword_379167)
    # Assigning a type to the variable 'if_condition_379168' (line 515)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 515, 4), 'if_condition_379168', if_condition_379168)
    # SSA begins for if statement (line 515)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 516):
    
    # Assigning a Call to a Name (line 516):
    
    # Call to astype(...): (line 516)
    # Processing the call arguments (line 516)
    # Getting the type of 'np' (line 516)
    np_379171 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 516, 21), 'np', False)
    # Obtaining the member 'intp' of a type (line 516)
    intp_379172 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 516, 21), np_379171, 'intp')
    # Processing the call keyword arguments (line 516)
    kwargs_379173 = {}
    # Getting the type of 'i' (line 516)
    i_379169 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 516, 12), 'i', False)
    # Obtaining the member 'astype' of a type (line 516)
    astype_379170 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 516, 12), i_379169, 'astype')
    # Calling astype(args, kwargs) (line 516)
    astype_call_result_379174 = invoke(stypy.reporting.localization.Localization(__file__, 516, 12), astype_379170, *[intp_379172], **kwargs_379173)
    
    # Assigning a type to the variable 'i' (line 516)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 516, 8), 'i', astype_call_result_379174)
    # SSA join for if statement (line 515)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'j' (line 517)
    j_379175 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 517, 11), 'j')
    # Obtaining the member 'flags' of a type (line 517)
    flags_379176 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 517, 11), j_379175, 'flags')
    # Obtaining the member 'writeable' of a type (line 517)
    writeable_379177 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 517, 11), flags_379176, 'writeable')
    # Applying the 'not' unary operator (line 517)
    result_not__379178 = python_operator(stypy.reporting.localization.Localization(__file__, 517, 7), 'not', writeable_379177)
    
    
    # Getting the type of 'j' (line 517)
    j_379179 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 517, 32), 'j')
    # Obtaining the member 'dtype' of a type (line 517)
    dtype_379180 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 517, 32), j_379179, 'dtype')
    
    # Obtaining an instance of the builtin type 'tuple' (line 517)
    tuple_379181 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 517, 48), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 517)
    # Adding element type (line 517)
    # Getting the type of 'np' (line 517)
    np_379182 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 517, 48), 'np')
    # Obtaining the member 'int32' of a type (line 517)
    int32_379183 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 517, 48), np_379182, 'int32')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 517, 48), tuple_379181, int32_379183)
    # Adding element type (line 517)
    # Getting the type of 'np' (line 517)
    np_379184 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 517, 58), 'np')
    # Obtaining the member 'int64' of a type (line 517)
    int64_379185 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 517, 58), np_379184, 'int64')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 517, 48), tuple_379181, int64_379185)
    
    # Applying the binary operator 'notin' (line 517)
    result_contains_379186 = python_operator(stypy.reporting.localization.Localization(__file__, 517, 32), 'notin', dtype_379180, tuple_379181)
    
    # Applying the binary operator 'or' (line 517)
    result_or_keyword_379187 = python_operator(stypy.reporting.localization.Localization(__file__, 517, 7), 'or', result_not__379178, result_contains_379186)
    
    # Testing the type of an if condition (line 517)
    if_condition_379188 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 517, 4), result_or_keyword_379187)
    # Assigning a type to the variable 'if_condition_379188' (line 517)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 517, 4), 'if_condition_379188', if_condition_379188)
    # SSA begins for if statement (line 517)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 518):
    
    # Assigning a Call to a Name (line 518):
    
    # Call to astype(...): (line 518)
    # Processing the call arguments (line 518)
    # Getting the type of 'np' (line 518)
    np_379191 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 518, 21), 'np', False)
    # Obtaining the member 'intp' of a type (line 518)
    intp_379192 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 518, 21), np_379191, 'intp')
    # Processing the call keyword arguments (line 518)
    kwargs_379193 = {}
    # Getting the type of 'j' (line 518)
    j_379189 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 518, 12), 'j', False)
    # Obtaining the member 'astype' of a type (line 518)
    astype_379190 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 518, 12), j_379189, 'astype')
    # Calling astype(args, kwargs) (line 518)
    astype_call_result_379194 = invoke(stypy.reporting.localization.Localization(__file__, 518, 12), astype_379190, *[intp_379192], **kwargs_379193)
    
    # Assigning a type to the variable 'j' (line 518)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 518, 8), 'j', astype_call_result_379194)
    # SSA join for if statement (line 517)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Type idiom detected: calculating its left and rigth part (line 520)
    # Getting the type of 'x' (line 520)
    x_379195 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 520, 4), 'x')
    # Getting the type of 'None' (line 520)
    None_379196 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 520, 16), 'None')
    
    (may_be_379197, more_types_in_union_379198) = may_not_be_none(x_379195, None_379196)

    if may_be_379197:

        if more_types_in_union_379198:
            # Runtime conditional SSA (line 520)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        
        # Getting the type of 'x' (line 521)
        x_379199 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 521, 15), 'x')
        # Obtaining the member 'flags' of a type (line 521)
        flags_379200 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 521, 15), x_379199, 'flags')
        # Obtaining the member 'writeable' of a type (line 521)
        writeable_379201 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 521, 15), flags_379200, 'writeable')
        # Applying the 'not' unary operator (line 521)
        result_not__379202 = python_operator(stypy.reporting.localization.Localization(__file__, 521, 11), 'not', writeable_379201)
        
        # Testing the type of an if condition (line 521)
        if_condition_379203 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 521, 8), result_not__379202)
        # Assigning a type to the variable 'if_condition_379203' (line 521)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 521, 8), 'if_condition_379203', if_condition_379203)
        # SSA begins for if statement (line 521)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 522):
        
        # Assigning a Call to a Name (line 522):
        
        # Call to copy(...): (line 522)
        # Processing the call keyword arguments (line 522)
        kwargs_379206 = {}
        # Getting the type of 'x' (line 522)
        x_379204 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 522, 16), 'x', False)
        # Obtaining the member 'copy' of a type (line 522)
        copy_379205 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 522, 16), x_379204, 'copy')
        # Calling copy(args, kwargs) (line 522)
        copy_call_result_379207 = invoke(stypy.reporting.localization.Localization(__file__, 522, 16), copy_379205, *[], **kwargs_379206)
        
        # Assigning a type to the variable 'x' (line 522)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 522, 12), 'x', copy_call_result_379207)
        # SSA join for if statement (line 521)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Obtaining an instance of the builtin type 'tuple' (line 523)
        tuple_379208 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 523, 15), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 523)
        # Adding element type (line 523)
        # Getting the type of 'i' (line 523)
        i_379209 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 523, 15), 'i')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 523, 15), tuple_379208, i_379209)
        # Adding element type (line 523)
        # Getting the type of 'j' (line 523)
        j_379210 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 523, 18), 'j')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 523, 15), tuple_379208, j_379210)
        # Adding element type (line 523)
        # Getting the type of 'x' (line 523)
        x_379211 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 523, 21), 'x')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 523, 15), tuple_379208, x_379211)
        
        # Assigning a type to the variable 'stypy_return_type' (line 523)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 523, 8), 'stypy_return_type', tuple_379208)

        if more_types_in_union_379198:
            # Runtime conditional SSA for else branch (line 520)
            module_type_store.open_ssa_branch('idiom else')



    if ((not may_be_379197) or more_types_in_union_379198):
        
        # Obtaining an instance of the builtin type 'tuple' (line 525)
        tuple_379212 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 525, 15), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 525)
        # Adding element type (line 525)
        # Getting the type of 'i' (line 525)
        i_379213 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 525, 15), 'i')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 525, 15), tuple_379212, i_379213)
        # Adding element type (line 525)
        # Getting the type of 'j' (line 525)
        j_379214 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 525, 18), 'j')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 525, 15), tuple_379212, j_379214)
        
        # Assigning a type to the variable 'stypy_return_type' (line 525)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 525, 8), 'stypy_return_type', tuple_379212)

        if (may_be_379197 and more_types_in_union_379198):
            # SSA join for if statement (line 520)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # ################# End of '_prepare_index_for_memoryview(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_prepare_index_for_memoryview' in the type store
    # Getting the type of 'stypy_return_type' (line 487)
    stypy_return_type_379215 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 487, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_379215)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_prepare_index_for_memoryview'
    return stypy_return_type_379215

# Assigning a type to the variable '_prepare_index_for_memoryview' (line 487)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 487, 0), '_prepare_index_for_memoryview', _prepare_index_for_memoryview)

@norecursion
def isspmatrix_lil(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'isspmatrix_lil'
    module_type_store = module_type_store.open_function_context('isspmatrix_lil', 528, 0, False)
    
    # Passed parameters checking function
    isspmatrix_lil.stypy_localization = localization
    isspmatrix_lil.stypy_type_of_self = None
    isspmatrix_lil.stypy_type_store = module_type_store
    isspmatrix_lil.stypy_function_name = 'isspmatrix_lil'
    isspmatrix_lil.stypy_param_names_list = ['x']
    isspmatrix_lil.stypy_varargs_param_name = None
    isspmatrix_lil.stypy_kwargs_param_name = None
    isspmatrix_lil.stypy_call_defaults = defaults
    isspmatrix_lil.stypy_call_varargs = varargs
    isspmatrix_lil.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'isspmatrix_lil', ['x'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'isspmatrix_lil', localization, ['x'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'isspmatrix_lil(...)' code ##################

    str_379216 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 550, (-1)), 'str', 'Is x of lil_matrix type?\n\n    Parameters\n    ----------\n    x\n        object to check for being a lil matrix\n\n    Returns\n    -------\n    bool\n        True if x is a lil matrix, False otherwise\n\n    Examples\n    --------\n    >>> from scipy.sparse import lil_matrix, isspmatrix_lil\n    >>> isspmatrix_lil(lil_matrix([[5]]))\n    True\n\n    >>> from scipy.sparse import lil_matrix, csr_matrix, isspmatrix_lil\n    >>> isspmatrix_lil(csr_matrix([[5]]))\n    False\n    ')
    
    # Call to isinstance(...): (line 551)
    # Processing the call arguments (line 551)
    # Getting the type of 'x' (line 551)
    x_379218 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 551, 22), 'x', False)
    # Getting the type of 'lil_matrix' (line 551)
    lil_matrix_379219 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 551, 25), 'lil_matrix', False)
    # Processing the call keyword arguments (line 551)
    kwargs_379220 = {}
    # Getting the type of 'isinstance' (line 551)
    isinstance_379217 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 551, 11), 'isinstance', False)
    # Calling isinstance(args, kwargs) (line 551)
    isinstance_call_result_379221 = invoke(stypy.reporting.localization.Localization(__file__, 551, 11), isinstance_379217, *[x_379218, lil_matrix_379219], **kwargs_379220)
    
    # Assigning a type to the variable 'stypy_return_type' (line 551)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 551, 4), 'stypy_return_type', isinstance_call_result_379221)
    
    # ################# End of 'isspmatrix_lil(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'isspmatrix_lil' in the type store
    # Getting the type of 'stypy_return_type' (line 528)
    stypy_return_type_379222 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 528, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_379222)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'isspmatrix_lil'
    return stypy_return_type_379222

# Assigning a type to the variable 'isspmatrix_lil' (line 528)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 528, 0), 'isspmatrix_lil', isspmatrix_lil)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
