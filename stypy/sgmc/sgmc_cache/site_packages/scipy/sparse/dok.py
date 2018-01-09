
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: '''Dictionary Of Keys based matrix'''
2: 
3: from __future__ import division, print_function, absolute_import
4: 
5: __docformat__ = "restructuredtext en"
6: 
7: __all__ = ['dok_matrix', 'isspmatrix_dok']
8: 
9: import functools
10: import operator
11: import itertools
12: 
13: import numpy as np
14: 
15: from scipy._lib.six import zip as izip, xrange, iteritems, iterkeys, itervalues
16: 
17: from .base import spmatrix, isspmatrix
18: from .sputils import (isdense, getdtype, isshape, isintlike, isscalarlike,
19:                       upcast, upcast_scalar, IndexMixin, get_index_dtype)
20: 
21: try:
22:     from operator import isSequenceType as _is_sequence
23: except ImportError:
24:     def _is_sequence(x):
25:         return (hasattr(x, '__len__') or hasattr(x, '__next__')
26:                 or hasattr(x, 'next'))
27: 
28: 
29: class dok_matrix(spmatrix, IndexMixin, dict):
30:     '''
31:     Dictionary Of Keys based sparse matrix.
32: 
33:     This is an efficient structure for constructing sparse
34:     matrices incrementally.
35: 
36:     This can be instantiated in several ways:
37:         dok_matrix(D)
38:             with a dense matrix, D
39: 
40:         dok_matrix(S)
41:             with a sparse matrix, S
42: 
43:         dok_matrix((M,N), [dtype])
44:             create the matrix with initial shape (M,N)
45:             dtype is optional, defaulting to dtype='d'
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
57: 
58:     Notes
59:     -----
60: 
61:     Sparse matrices can be used in arithmetic operations: they support
62:     addition, subtraction, multiplication, division, and matrix power.
63: 
64:     Allows for efficient O(1) access of individual elements.
65:     Duplicates are not allowed.
66:     Can be efficiently converted to a coo_matrix once constructed.
67: 
68:     Examples
69:     --------
70:     >>> import numpy as np
71:     >>> from scipy.sparse import dok_matrix
72:     >>> S = dok_matrix((5, 5), dtype=np.float32)
73:     >>> for i in range(5):
74:     ...     for j in range(5):
75:     ...         S[i, j] = i + j    # Update element
76: 
77:     '''
78:     format = 'dok'
79: 
80:     def __init__(self, arg1, shape=None, dtype=None, copy=False):
81:         dict.__init__(self)
82:         spmatrix.__init__(self)
83: 
84:         self.dtype = getdtype(dtype, default=float)
85:         if isinstance(arg1, tuple) and isshape(arg1):  # (M,N)
86:             M, N = arg1
87:             self.shape = (M, N)
88:         elif isspmatrix(arg1):  # Sparse ctor
89:             if isspmatrix_dok(arg1) and copy:
90:                 arg1 = arg1.copy()
91:             else:
92:                 arg1 = arg1.todok()
93: 
94:             if dtype is not None:
95:                 arg1 = arg1.astype(dtype)
96: 
97:             dict.update(self, arg1)
98:             self.shape = arg1.shape
99:             self.dtype = arg1.dtype
100:         else:  # Dense ctor
101:             try:
102:                 arg1 = np.asarray(arg1)
103:             except:
104:                 raise TypeError('Invalid input format.')
105: 
106:             if len(arg1.shape) != 2:
107:                 raise TypeError('Expected rank <=2 dense array or matrix.')
108: 
109:             from .coo import coo_matrix
110:             d = coo_matrix(arg1, dtype=dtype).todok()
111:             dict.update(self, d)
112:             self.shape = arg1.shape
113:             self.dtype = d.dtype
114: 
115:     def update(self, val):
116:         # Prevent direct usage of update
117:         raise NotImplementedError("Direct modification to dok_matrix element "
118:                                   "is not allowed.")
119: 
120:     def _update(self, data):
121:         '''An update method for dict data defined for direct access to
122:         `dok_matrix` data. Main purpose is to be used for effcient conversion
123:         from other spmatrix classes. Has no checking if `data` is valid.'''
124:         return dict.update(self, data)
125: 
126:     def getnnz(self, axis=None):
127:         if axis is not None:
128:             raise NotImplementedError("getnnz over an axis is not implemented "
129:                                       "for DOK format.")
130:         return dict.__len__(self)
131: 
132:     def count_nonzero(self):
133:         return sum(x != 0 for x in itervalues(self))
134: 
135:     getnnz.__doc__ = spmatrix.getnnz.__doc__
136:     count_nonzero.__doc__ = spmatrix.count_nonzero.__doc__
137: 
138:     def __len__(self):
139:         return dict.__len__(self)
140: 
141:     def get(self, key, default=0.):
142:         '''This overrides the dict.get method, providing type checking
143:         but otherwise equivalent functionality.
144:         '''
145:         try:
146:             i, j = key
147:             assert isintlike(i) and isintlike(j)
148:         except (AssertionError, TypeError, ValueError):
149:             raise IndexError('Index must be a pair of integers.')
150:         if (i < 0 or i >= self.shape[0] or j < 0 or j >= self.shape[1]):
151:             raise IndexError('Index out of bounds.')
152:         return dict.get(self, key, default)
153: 
154:     def __getitem__(self, index):
155:         '''If key=(i, j) is a pair of integers, return the corresponding
156:         element.  If either i or j is a slice or sequence, return a new sparse
157:         matrix with just these elements.
158:         '''
159:         zero = self.dtype.type(0)
160:         i, j = self._unpack_index(index)
161: 
162:         i_intlike = isintlike(i)
163:         j_intlike = isintlike(j)
164: 
165:         if i_intlike and j_intlike:
166:             i = int(i)
167:             j = int(j)
168:             if i < 0:
169:                 i += self.shape[0]
170:             if i < 0 or i >= self.shape[0]:
171:                 raise IndexError('Index out of bounds.')
172:             if j < 0:
173:                 j += self.shape[1]
174:             if j < 0 or j >= self.shape[1]:
175:                 raise IndexError('Index out of bounds.')
176:             return dict.get(self, (i,j), zero)
177:         elif ((i_intlike or isinstance(i, slice)) and
178:               (j_intlike or isinstance(j, slice))):
179:             # Fast path for slicing very sparse matrices
180:             i_slice = slice(i, i+1) if i_intlike else i
181:             j_slice = slice(j, j+1) if j_intlike else j
182:             i_indices = i_slice.indices(self.shape[0])
183:             j_indices = j_slice.indices(self.shape[1])
184:             i_seq = xrange(*i_indices)
185:             j_seq = xrange(*j_indices)
186:             newshape = (len(i_seq), len(j_seq))
187:             newsize = _prod(newshape)
188: 
189:             if len(self) < 2*newsize and newsize != 0:
190:                 # Switch to the fast path only when advantageous
191:                 # (count the iterations in the loops, adjust for complexity)
192:                 #
193:                 # We also don't handle newsize == 0 here (if
194:                 # i/j_intlike, it can mean index i or j was out of
195:                 # bounds)
196:                 return self._getitem_ranges(i_indices, j_indices, newshape)
197: 
198:         i, j = self._index_to_arrays(i, j)
199: 
200:         if i.size == 0:
201:             return dok_matrix(i.shape, dtype=self.dtype)
202: 
203:         min_i = i.min()
204:         if min_i < -self.shape[0] or i.max() >= self.shape[0]:
205:             raise IndexError('Index (%d) out of range -%d to %d.' %
206:                              (i.min(), self.shape[0], self.shape[0]-1))
207:         if min_i < 0:
208:             i = i.copy()
209:             i[i < 0] += self.shape[0]
210: 
211:         min_j = j.min()
212:         if min_j < -self.shape[1] or j.max() >= self.shape[1]:
213:             raise IndexError('Index (%d) out of range -%d to %d.' %
214:                              (j.min(), self.shape[1], self.shape[1]-1))
215:         if min_j < 0:
216:             j = j.copy()
217:             j[j < 0] += self.shape[1]
218: 
219:         newdok = dok_matrix(i.shape, dtype=self.dtype)
220: 
221:         for key in itertools.product(xrange(i.shape[0]), xrange(i.shape[1])):
222:             v = dict.get(self, (i[key], j[key]), zero)
223:             if v:
224:                 dict.__setitem__(newdok, key, v)
225: 
226:         return newdok
227: 
228:     def _getitem_ranges(self, i_indices, j_indices, shape):
229:         # performance golf: we don't want Numpy scalars here, they are slow
230:         i_start, i_stop, i_stride = map(int, i_indices)
231:         j_start, j_stop, j_stride = map(int, j_indices)
232: 
233:         newdok = dok_matrix(shape, dtype=self.dtype)
234: 
235:         for (ii, jj) in iterkeys(self):
236:             # ditto for numpy scalars
237:             ii = int(ii)
238:             jj = int(jj)
239:             a, ra = divmod(ii - i_start, i_stride)
240:             if a < 0 or a >= shape[0] or ra != 0:
241:                 continue
242:             b, rb = divmod(jj - j_start, j_stride)
243:             if b < 0 or b >= shape[1] or rb != 0:
244:                 continue
245:             dict.__setitem__(newdok, (a, b),
246:                              dict.__getitem__(self, (ii, jj)))
247:         return newdok
248: 
249:     def __setitem__(self, index, x):
250:         if isinstance(index, tuple) and len(index) == 2:
251:             # Integer index fast path
252:             i, j = index
253:             if (isintlike(i) and isintlike(j) and 0 <= i < self.shape[0]
254:                     and 0 <= j < self.shape[1]):
255:                 v = np.asarray(x, dtype=self.dtype)
256:                 if v.ndim == 0 and v != 0:
257:                     dict.__setitem__(self, (int(i), int(j)), v[()])
258:                     return
259: 
260:         i, j = self._unpack_index(index)
261:         i, j = self._index_to_arrays(i, j)
262: 
263:         if isspmatrix(x):
264:             x = x.toarray()
265: 
266:         # Make x and i into the same shape
267:         x = np.asarray(x, dtype=self.dtype)
268:         x, _ = np.broadcast_arrays(x, i)
269: 
270:         if x.shape != i.shape:
271:             raise ValueError("Shape mismatch in assignment.")
272: 
273:         if np.size(x) == 0:
274:             return
275: 
276:         min_i = i.min()
277:         if min_i < -self.shape[0] or i.max() >= self.shape[0]:
278:             raise IndexError('Index (%d) out of range -%d to %d.' %
279:                              (i.min(), self.shape[0], self.shape[0]-1))
280:         if min_i < 0:
281:             i = i.copy()
282:             i[i < 0] += self.shape[0]
283: 
284:         min_j = j.min()
285:         if min_j < -self.shape[1] or j.max() >= self.shape[1]:
286:             raise IndexError('Index (%d) out of range -%d to %d.' %
287:                              (j.min(), self.shape[1], self.shape[1]-1))
288:         if min_j < 0:
289:             j = j.copy()
290:             j[j < 0] += self.shape[1]
291: 
292:         dict.update(self, izip(izip(i.flat, j.flat), x.flat))
293: 
294:         if 0 in x:
295:             zeroes = x == 0
296:             for key in izip(i[zeroes].flat, j[zeroes].flat):
297:                 if dict.__getitem__(self, key) == 0:
298:                     # may have been superseded by later update
299:                     del self[key]
300: 
301:     def __add__(self, other):
302:         if isscalarlike(other):
303:             res_dtype = upcast_scalar(self.dtype, other)
304:             new = dok_matrix(self.shape, dtype=res_dtype)
305:             # Add this scalar to every element.
306:             M, N = self.shape
307:             for key in itertools.product(xrange(M), xrange(N)):
308:                 aij = dict.get(self, (key), 0) + other
309:                 if aij:
310:                     new[key] = aij
311:             # new.dtype.char = self.dtype.char
312:         elif isspmatrix_dok(other):
313:             if other.shape != self.shape:
314:                 raise ValueError("Matrix dimensions are not equal.")
315:             # We could alternatively set the dimensions to the largest of
316:             # the two matrices to be summed.  Would this be a good idea?
317:             res_dtype = upcast(self.dtype, other.dtype)
318:             new = dok_matrix(self.shape, dtype=res_dtype)
319:             dict.update(new, self)
320:             with np.errstate(over='ignore'):
321:                 dict.update(new,
322:                            ((k, new[k] + other[k]) for k in iterkeys(other)))
323:         elif isspmatrix(other):
324:             csc = self.tocsc()
325:             new = csc + other
326:         elif isdense(other):
327:             new = self.todense() + other
328:         else:
329:             return NotImplemented
330:         return new
331: 
332:     def __radd__(self, other):
333:         if isscalarlike(other):
334:             new = dok_matrix(self.shape, dtype=self.dtype)
335:             M, N = self.shape
336:             for key in itertools.product(xrange(M), xrange(N)):
337:                 aij = dict.get(self, (key), 0) + other
338:                 if aij:
339:                     new[key] = aij
340:         elif isspmatrix_dok(other):
341:             if other.shape != self.shape:
342:                 raise ValueError("Matrix dimensions are not equal.")
343:             new = dok_matrix(self.shape, dtype=self.dtype)
344:             dict.update(new, self)
345:             dict.update(new,
346:                        ((k, self[k] + other[k]) for k in iterkeys(other)))
347:         elif isspmatrix(other):
348:             csc = self.tocsc()
349:             new = csc + other
350:         elif isdense(other):
351:             new = other + self.todense()
352:         else:
353:             return NotImplemented
354:         return new
355: 
356:     def __neg__(self):
357:         if self.dtype.kind == 'b':
358:             raise NotImplementedError('Negating a sparse boolean matrix is not'
359:                                       ' supported.')
360:         new = dok_matrix(self.shape, dtype=self.dtype)
361:         dict.update(new, ((k, -self[k]) for k in iterkeys(self)))
362:         return new
363: 
364:     def _mul_scalar(self, other):
365:         res_dtype = upcast_scalar(self.dtype, other)
366:         # Multiply this scalar by every element.
367:         new = dok_matrix(self.shape, dtype=res_dtype)
368:         dict.update(new, ((k, v * other) for k, v in iteritems(self)))
369:         return new
370: 
371:     def _mul_vector(self, other):
372:         # matrix * vector
373:         result = np.zeros(self.shape[0], dtype=upcast(self.dtype, other.dtype))
374:         for (i, j), v in iteritems(self):
375:             result[i] += v * other[j]
376:         return result
377: 
378:     def _mul_multivector(self, other):
379:         # matrix * multivector
380:         result_shape = (self.shape[0], other.shape[1])
381:         result_dtype = upcast(self.dtype, other.dtype)
382:         result = np.zeros(result_shape, dtype=result_dtype)
383:         for (i, j), v in iteritems(self):
384:             result[i,:] += v * other[j,:]
385:         return result
386: 
387:     def __imul__(self, other):
388:         if isscalarlike(other):
389:             dict.update(self, ((k, v * other) for k, v in iteritems(self)))
390:             return self
391:         return NotImplemented
392: 
393:     def __truediv__(self, other):
394:         if isscalarlike(other):
395:             res_dtype = upcast_scalar(self.dtype, other)
396:             new = dok_matrix(self.shape, dtype=res_dtype)
397:             dict.update(new, ((k, v / other) for k, v in iteritems(self)))
398:             return new
399:         return self.tocsr() / other
400: 
401:     def __itruediv__(self, other):
402:         if isscalarlike(other):
403:             dict.update(self, ((k, v / other) for k, v in iteritems(self)))
404:             return self
405:         return NotImplemented
406: 
407:     def __reduce__(self):
408:         # this approach is necessary because __setstate__ is called after
409:         # __setitem__ upon unpickling and since __init__ is not called there
410:         # is no shape attribute hence it is not possible to unpickle it.
411:         return dict.__reduce__(self)
412: 
413:     # What should len(sparse) return? For consistency with dense matrices,
414:     # perhaps it should be the number of rows?  For now it returns the number
415:     # of non-zeros.
416: 
417:     def transpose(self, axes=None, copy=False):
418:         if axes is not None:
419:             raise ValueError("Sparse matrices do not support "
420:                              "an 'axes' parameter because swapping "
421:                              "dimensions is the only logical permutation.")
422: 
423:         M, N = self.shape
424:         new = dok_matrix((N, M), dtype=self.dtype, copy=copy)
425:         dict.update(new, (((right, left), val)
426:                           for (left, right), val in iteritems(self)))
427:         return new
428: 
429:     transpose.__doc__ = spmatrix.transpose.__doc__
430: 
431:     def conjtransp(self):
432:         '''Return the conjugate transpose.'''
433:         M, N = self.shape
434:         new = dok_matrix((N, M), dtype=self.dtype)
435:         dict.update(new, (((right, left), np.conj(val))
436:                           for (left, right), val in iteritems(self)))
437:         return new
438: 
439:     def copy(self):
440:         new = dok_matrix(self.shape, dtype=self.dtype)
441:         dict.update(new, self)
442:         return new
443: 
444:     copy.__doc__ = spmatrix.copy.__doc__
445: 
446:     def getrow(self, i):
447:         '''Returns the i-th row as a (1 x n) DOK matrix.'''
448:         new = dok_matrix((1, self.shape[1]), dtype=self.dtype)
449:         dict.update(new, (((0, j), self[i, j]) for j in xrange(self.shape[1])))
450:         return new
451: 
452:     def getcol(self, j):
453:         '''Returns the j-th column as a (m x 1) DOK matrix.'''
454:         new = dok_matrix((self.shape[0], 1), dtype=self.dtype)
455:         dict.update(new, (((i, 0), self[i, j]) for i in xrange(self.shape[0])))
456:         return new
457: 
458:     def tocoo(self, copy=False):
459:         from .coo import coo_matrix
460:         if self.nnz == 0:
461:             return coo_matrix(self.shape, dtype=self.dtype)
462: 
463:         idx_dtype = get_index_dtype(maxval=max(self.shape))
464:         data = np.fromiter(itervalues(self), dtype=self.dtype, count=self.nnz)
465:         I = np.fromiter((i for i, _ in iterkeys(self)), dtype=idx_dtype, count=self.nnz)
466:         J = np.fromiter((j for _, j in iterkeys(self)), dtype=idx_dtype, count=self.nnz)
467:         A = coo_matrix((data, (I, J)), shape=self.shape, dtype=self.dtype)
468:         A.has_canonical_format = True
469:         return A
470: 
471:     tocoo.__doc__ = spmatrix.tocoo.__doc__
472: 
473:     def todok(self, copy=False):
474:         if copy:
475:             return self.copy()
476:         return self
477: 
478:     todok.__doc__ = spmatrix.todok.__doc__
479: 
480:     def tocsc(self, copy=False):
481:         return self.tocoo(copy=False).tocsc(copy=copy)
482: 
483:     tocsc.__doc__ = spmatrix.tocsc.__doc__
484: 
485:     def resize(self, shape):
486:         '''Resize the matrix in-place to dimensions given by `shape`.
487: 
488:         Any non-zero elements that lie outside the new shape are removed.
489:         '''
490:         if not isshape(shape):
491:             raise TypeError("Dimensions must be a 2-tuple of positive integers")
492:         newM, newN = shape
493:         M, N = self.shape
494:         if newM < M or newN < N:
495:             # Remove all elements outside new dimensions
496:             for (i, j) in list(iterkeys(self)):
497:                 if i >= newM or j >= newN:
498:                     del self[i, j]
499:         self._shape = shape
500: 
501: 
502: def isspmatrix_dok(x):
503:     '''Is x of dok_matrix type?
504: 
505:     Parameters
506:     ----------
507:     x
508:         object to check for being a dok matrix
509: 
510:     Returns
511:     -------
512:     bool
513:         True if x is a dok matrix, False otherwise
514: 
515:     Examples
516:     --------
517:     >>> from scipy.sparse import dok_matrix, isspmatrix_dok
518:     >>> isspmatrix_dok(dok_matrix([[5]]))
519:     True
520: 
521:     >>> from scipy.sparse import dok_matrix, csr_matrix, isspmatrix_dok
522:     >>> isspmatrix_dok(csr_matrix([[5]]))
523:     False
524:     '''
525:     return isinstance(x, dok_matrix)
526: 
527: 
528: def _prod(x):
529:     '''Product of a list of numbers; ~40x faster vs np.prod for Python tuples'''
530:     if len(x) == 0:
531:         return 1
532:     return functools.reduce(operator.mul, x)
533: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_374378 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1, 0), 'str', 'Dictionary Of Keys based matrix')

# Assigning a Str to a Name (line 5):

# Assigning a Str to a Name (line 5):
str_374379 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 16), 'str', 'restructuredtext en')
# Assigning a type to the variable '__docformat__' (line 5)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 0), '__docformat__', str_374379)

# Assigning a List to a Name (line 7):

# Assigning a List to a Name (line 7):
__all__ = ['dok_matrix', 'isspmatrix_dok']
module_type_store.set_exportable_members(['dok_matrix', 'isspmatrix_dok'])

# Obtaining an instance of the builtin type 'list' (line 7)
list_374380 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 10), 'list')
# Adding type elements to the builtin type 'list' instance (line 7)
# Adding element type (line 7)
str_374381 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 11), 'str', 'dok_matrix')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 7, 10), list_374380, str_374381)
# Adding element type (line 7)
str_374382 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 25), 'str', 'isspmatrix_dok')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 7, 10), list_374380, str_374382)

# Assigning a type to the variable '__all__' (line 7)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), '__all__', list_374380)
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 9, 0))

# 'import functools' statement (line 9)
import functools

import_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'functools', functools, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 10, 0))

# 'import operator' statement (line 10)
import operator

import_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'operator', operator, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 11, 0))

# 'import itertools' statement (line 11)
import itertools

import_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'itertools', itertools, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 13, 0))

# 'import numpy' statement (line 13)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/sparse/')
import_374383 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 13, 0), 'numpy')

if (type(import_374383) is not StypyTypeError):

    if (import_374383 != 'pyd_module'):
        __import__(import_374383)
        sys_modules_374384 = sys.modules[import_374383]
        import_module(stypy.reporting.localization.Localization(__file__, 13, 0), 'np', sys_modules_374384.module_type_store, module_type_store)
    else:
        import numpy as np

        import_module(stypy.reporting.localization.Localization(__file__, 13, 0), 'np', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 13)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 0), 'numpy', import_374383)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/sparse/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 15, 0))

# 'from scipy._lib.six import izip, xrange, iteritems, iterkeys, itervalues' statement (line 15)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/sparse/')
import_374385 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 15, 0), 'scipy._lib.six')

if (type(import_374385) is not StypyTypeError):

    if (import_374385 != 'pyd_module'):
        __import__(import_374385)
        sys_modules_374386 = sys.modules[import_374385]
        import_from_module(stypy.reporting.localization.Localization(__file__, 15, 0), 'scipy._lib.six', sys_modules_374386.module_type_store, module_type_store, ['zip', 'xrange', 'iteritems', 'iterkeys', 'itervalues'])
        nest_module(stypy.reporting.localization.Localization(__file__, 15, 0), __file__, sys_modules_374386, sys_modules_374386.module_type_store, module_type_store)
    else:
        from scipy._lib.six import zip as izip, xrange, iteritems, iterkeys, itervalues

        import_from_module(stypy.reporting.localization.Localization(__file__, 15, 0), 'scipy._lib.six', None, module_type_store, ['zip', 'xrange', 'iteritems', 'iterkeys', 'itervalues'], [izip, xrange, iteritems, iterkeys, itervalues])

else:
    # Assigning a type to the variable 'scipy._lib.six' (line 15)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 0), 'scipy._lib.six', import_374385)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/sparse/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 17, 0))

# 'from scipy.sparse.base import spmatrix, isspmatrix' statement (line 17)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/sparse/')
import_374387 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 17, 0), 'scipy.sparse.base')

if (type(import_374387) is not StypyTypeError):

    if (import_374387 != 'pyd_module'):
        __import__(import_374387)
        sys_modules_374388 = sys.modules[import_374387]
        import_from_module(stypy.reporting.localization.Localization(__file__, 17, 0), 'scipy.sparse.base', sys_modules_374388.module_type_store, module_type_store, ['spmatrix', 'isspmatrix'])
        nest_module(stypy.reporting.localization.Localization(__file__, 17, 0), __file__, sys_modules_374388, sys_modules_374388.module_type_store, module_type_store)
    else:
        from scipy.sparse.base import spmatrix, isspmatrix

        import_from_module(stypy.reporting.localization.Localization(__file__, 17, 0), 'scipy.sparse.base', None, module_type_store, ['spmatrix', 'isspmatrix'], [spmatrix, isspmatrix])

else:
    # Assigning a type to the variable 'scipy.sparse.base' (line 17)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 17, 0), 'scipy.sparse.base', import_374387)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/sparse/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 18, 0))

# 'from scipy.sparse.sputils import isdense, getdtype, isshape, isintlike, isscalarlike, upcast, upcast_scalar, IndexMixin, get_index_dtype' statement (line 18)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/sparse/')
import_374389 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 18, 0), 'scipy.sparse.sputils')

if (type(import_374389) is not StypyTypeError):

    if (import_374389 != 'pyd_module'):
        __import__(import_374389)
        sys_modules_374390 = sys.modules[import_374389]
        import_from_module(stypy.reporting.localization.Localization(__file__, 18, 0), 'scipy.sparse.sputils', sys_modules_374390.module_type_store, module_type_store, ['isdense', 'getdtype', 'isshape', 'isintlike', 'isscalarlike', 'upcast', 'upcast_scalar', 'IndexMixin', 'get_index_dtype'])
        nest_module(stypy.reporting.localization.Localization(__file__, 18, 0), __file__, sys_modules_374390, sys_modules_374390.module_type_store, module_type_store)
    else:
        from scipy.sparse.sputils import isdense, getdtype, isshape, isintlike, isscalarlike, upcast, upcast_scalar, IndexMixin, get_index_dtype

        import_from_module(stypy.reporting.localization.Localization(__file__, 18, 0), 'scipy.sparse.sputils', None, module_type_store, ['isdense', 'getdtype', 'isshape', 'isintlike', 'isscalarlike', 'upcast', 'upcast_scalar', 'IndexMixin', 'get_index_dtype'], [isdense, getdtype, isshape, isintlike, isscalarlike, upcast, upcast_scalar, IndexMixin, get_index_dtype])

else:
    # Assigning a type to the variable 'scipy.sparse.sputils' (line 18)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 18, 0), 'scipy.sparse.sputils', import_374389)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/sparse/')



# SSA begins for try-except statement (line 21)
module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 22, 4))

# 'from operator import _is_sequence' statement (line 22)
try:
    from operator import isSequenceType as _is_sequence

except:
    _is_sequence = UndefinedType
import_from_module(stypy.reporting.localization.Localization(__file__, 22, 4), 'operator', None, module_type_store, ['isSequenceType'], [_is_sequence])
# Adding an alias
module_type_store.add_alias('_is_sequence', 'isSequenceType')

# SSA branch for the except part of a try statement (line 21)
# SSA branch for the except 'ImportError' branch of a try statement (line 21)
module_type_store.open_ssa_branch('except')

@norecursion
def _is_sequence(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_is_sequence'
    module_type_store = module_type_store.open_function_context('_is_sequence', 24, 4, False)
    
    # Passed parameters checking function
    _is_sequence.stypy_localization = localization
    _is_sequence.stypy_type_of_self = None
    _is_sequence.stypy_type_store = module_type_store
    _is_sequence.stypy_function_name = '_is_sequence'
    _is_sequence.stypy_param_names_list = ['x']
    _is_sequence.stypy_varargs_param_name = None
    _is_sequence.stypy_kwargs_param_name = None
    _is_sequence.stypy_call_defaults = defaults
    _is_sequence.stypy_call_varargs = varargs
    _is_sequence.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_is_sequence', ['x'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_is_sequence', localization, ['x'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_is_sequence(...)' code ##################

    
    # Evaluating a boolean operation
    
    # Call to hasattr(...): (line 25)
    # Processing the call arguments (line 25)
    # Getting the type of 'x' (line 25)
    x_374392 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 24), 'x', False)
    str_374393 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 27), 'str', '__len__')
    # Processing the call keyword arguments (line 25)
    kwargs_374394 = {}
    # Getting the type of 'hasattr' (line 25)
    hasattr_374391 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 16), 'hasattr', False)
    # Calling hasattr(args, kwargs) (line 25)
    hasattr_call_result_374395 = invoke(stypy.reporting.localization.Localization(__file__, 25, 16), hasattr_374391, *[x_374392, str_374393], **kwargs_374394)
    
    
    # Call to hasattr(...): (line 25)
    # Processing the call arguments (line 25)
    # Getting the type of 'x' (line 25)
    x_374397 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 49), 'x', False)
    str_374398 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 52), 'str', '__next__')
    # Processing the call keyword arguments (line 25)
    kwargs_374399 = {}
    # Getting the type of 'hasattr' (line 25)
    hasattr_374396 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 41), 'hasattr', False)
    # Calling hasattr(args, kwargs) (line 25)
    hasattr_call_result_374400 = invoke(stypy.reporting.localization.Localization(__file__, 25, 41), hasattr_374396, *[x_374397, str_374398], **kwargs_374399)
    
    # Applying the binary operator 'or' (line 25)
    result_or_keyword_374401 = python_operator(stypy.reporting.localization.Localization(__file__, 25, 16), 'or', hasattr_call_result_374395, hasattr_call_result_374400)
    
    # Call to hasattr(...): (line 26)
    # Processing the call arguments (line 26)
    # Getting the type of 'x' (line 26)
    x_374403 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 27), 'x', False)
    str_374404 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 30), 'str', 'next')
    # Processing the call keyword arguments (line 26)
    kwargs_374405 = {}
    # Getting the type of 'hasattr' (line 26)
    hasattr_374402 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 19), 'hasattr', False)
    # Calling hasattr(args, kwargs) (line 26)
    hasattr_call_result_374406 = invoke(stypy.reporting.localization.Localization(__file__, 26, 19), hasattr_374402, *[x_374403, str_374404], **kwargs_374405)
    
    # Applying the binary operator 'or' (line 25)
    result_or_keyword_374407 = python_operator(stypy.reporting.localization.Localization(__file__, 25, 16), 'or', result_or_keyword_374401, hasattr_call_result_374406)
    
    # Assigning a type to the variable 'stypy_return_type' (line 25)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 25, 8), 'stypy_return_type', result_or_keyword_374407)
    
    # ################# End of '_is_sequence(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_is_sequence' in the type store
    # Getting the type of 'stypy_return_type' (line 24)
    stypy_return_type_374408 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 4), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_374408)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_is_sequence'
    return stypy_return_type_374408

# Assigning a type to the variable '_is_sequence' (line 24)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 24, 4), '_is_sequence', _is_sequence)
# SSA join for try-except statement (line 21)
module_type_store = module_type_store.join_ssa_context()

# Declaration of the 'dok_matrix' class
# Getting the type of 'spmatrix' (line 29)
spmatrix_374409 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 17), 'spmatrix')
# Getting the type of 'IndexMixin' (line 29)
IndexMixin_374410 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 27), 'IndexMixin')
# Getting the type of 'dict' (line 29)
dict_374411 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 39), 'dict')

class dok_matrix(spmatrix_374409, IndexMixin_374410, dict_374411, ):
    str_374412 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 77, (-1)), 'str', "\n    Dictionary Of Keys based sparse matrix.\n\n    This is an efficient structure for constructing sparse\n    matrices incrementally.\n\n    This can be instantiated in several ways:\n        dok_matrix(D)\n            with a dense matrix, D\n\n        dok_matrix(S)\n            with a sparse matrix, S\n\n        dok_matrix((M,N), [dtype])\n            create the matrix with initial shape (M,N)\n            dtype is optional, defaulting to dtype='d'\n\n    Attributes\n    ----------\n    dtype : dtype\n        Data type of the matrix\n    shape : 2-tuple\n        Shape of the matrix\n    ndim : int\n        Number of dimensions (this is always 2)\n    nnz\n        Number of nonzero elements\n\n    Notes\n    -----\n\n    Sparse matrices can be used in arithmetic operations: they support\n    addition, subtraction, multiplication, division, and matrix power.\n\n    Allows for efficient O(1) access of individual elements.\n    Duplicates are not allowed.\n    Can be efficiently converted to a coo_matrix once constructed.\n\n    Examples\n    --------\n    >>> import numpy as np\n    >>> from scipy.sparse import dok_matrix\n    >>> S = dok_matrix((5, 5), dtype=np.float32)\n    >>> for i in range(5):\n    ...     for j in range(5):\n    ...         S[i, j] = i + j    # Update element\n\n    ")
    
    # Assigning a Str to a Name (line 78):

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 80)
        None_374413 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 35), 'None')
        # Getting the type of 'None' (line 80)
        None_374414 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 47), 'None')
        # Getting the type of 'False' (line 80)
        False_374415 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 58), 'False')
        defaults = [None_374413, None_374414, False_374415]
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 80, 4, False)
        # Assigning a type to the variable 'self' (line 81)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 81, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'dok_matrix.__init__', ['arg1', 'shape', 'dtype', 'copy'], None, None, defaults, varargs, kwargs)

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

        
        # Call to __init__(...): (line 81)
        # Processing the call arguments (line 81)
        # Getting the type of 'self' (line 81)
        self_374418 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 22), 'self', False)
        # Processing the call keyword arguments (line 81)
        kwargs_374419 = {}
        # Getting the type of 'dict' (line 81)
        dict_374416 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 8), 'dict', False)
        # Obtaining the member '__init__' of a type (line 81)
        init___374417 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 81, 8), dict_374416, '__init__')
        # Calling __init__(args, kwargs) (line 81)
        init___call_result_374420 = invoke(stypy.reporting.localization.Localization(__file__, 81, 8), init___374417, *[self_374418], **kwargs_374419)
        
        
        # Call to __init__(...): (line 82)
        # Processing the call arguments (line 82)
        # Getting the type of 'self' (line 82)
        self_374423 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 26), 'self', False)
        # Processing the call keyword arguments (line 82)
        kwargs_374424 = {}
        # Getting the type of 'spmatrix' (line 82)
        spmatrix_374421 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 8), 'spmatrix', False)
        # Obtaining the member '__init__' of a type (line 82)
        init___374422 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 82, 8), spmatrix_374421, '__init__')
        # Calling __init__(args, kwargs) (line 82)
        init___call_result_374425 = invoke(stypy.reporting.localization.Localization(__file__, 82, 8), init___374422, *[self_374423], **kwargs_374424)
        
        
        # Assigning a Call to a Attribute (line 84):
        
        # Assigning a Call to a Attribute (line 84):
        
        # Call to getdtype(...): (line 84)
        # Processing the call arguments (line 84)
        # Getting the type of 'dtype' (line 84)
        dtype_374427 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 30), 'dtype', False)
        # Processing the call keyword arguments (line 84)
        # Getting the type of 'float' (line 84)
        float_374428 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 45), 'float', False)
        keyword_374429 = float_374428
        kwargs_374430 = {'default': keyword_374429}
        # Getting the type of 'getdtype' (line 84)
        getdtype_374426 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 21), 'getdtype', False)
        # Calling getdtype(args, kwargs) (line 84)
        getdtype_call_result_374431 = invoke(stypy.reporting.localization.Localization(__file__, 84, 21), getdtype_374426, *[dtype_374427], **kwargs_374430)
        
        # Getting the type of 'self' (line 84)
        self_374432 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 8), 'self')
        # Setting the type of the member 'dtype' of a type (line 84)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 84, 8), self_374432, 'dtype', getdtype_call_result_374431)
        
        
        # Evaluating a boolean operation
        
        # Call to isinstance(...): (line 85)
        # Processing the call arguments (line 85)
        # Getting the type of 'arg1' (line 85)
        arg1_374434 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 22), 'arg1', False)
        # Getting the type of 'tuple' (line 85)
        tuple_374435 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 28), 'tuple', False)
        # Processing the call keyword arguments (line 85)
        kwargs_374436 = {}
        # Getting the type of 'isinstance' (line 85)
        isinstance_374433 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 11), 'isinstance', False)
        # Calling isinstance(args, kwargs) (line 85)
        isinstance_call_result_374437 = invoke(stypy.reporting.localization.Localization(__file__, 85, 11), isinstance_374433, *[arg1_374434, tuple_374435], **kwargs_374436)
        
        
        # Call to isshape(...): (line 85)
        # Processing the call arguments (line 85)
        # Getting the type of 'arg1' (line 85)
        arg1_374439 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 47), 'arg1', False)
        # Processing the call keyword arguments (line 85)
        kwargs_374440 = {}
        # Getting the type of 'isshape' (line 85)
        isshape_374438 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 39), 'isshape', False)
        # Calling isshape(args, kwargs) (line 85)
        isshape_call_result_374441 = invoke(stypy.reporting.localization.Localization(__file__, 85, 39), isshape_374438, *[arg1_374439], **kwargs_374440)
        
        # Applying the binary operator 'and' (line 85)
        result_and_keyword_374442 = python_operator(stypy.reporting.localization.Localization(__file__, 85, 11), 'and', isinstance_call_result_374437, isshape_call_result_374441)
        
        # Testing the type of an if condition (line 85)
        if_condition_374443 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 85, 8), result_and_keyword_374442)
        # Assigning a type to the variable 'if_condition_374443' (line 85)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 85, 8), 'if_condition_374443', if_condition_374443)
        # SSA begins for if statement (line 85)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Name to a Tuple (line 86):
        
        # Assigning a Subscript to a Name (line 86):
        
        # Obtaining the type of the subscript
        int_374444 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 86, 12), 'int')
        # Getting the type of 'arg1' (line 86)
        arg1_374445 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 19), 'arg1')
        # Obtaining the member '__getitem__' of a type (line 86)
        getitem___374446 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 86, 12), arg1_374445, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 86)
        subscript_call_result_374447 = invoke(stypy.reporting.localization.Localization(__file__, 86, 12), getitem___374446, int_374444)
        
        # Assigning a type to the variable 'tuple_var_assignment_374340' (line 86)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 86, 12), 'tuple_var_assignment_374340', subscript_call_result_374447)
        
        # Assigning a Subscript to a Name (line 86):
        
        # Obtaining the type of the subscript
        int_374448 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 86, 12), 'int')
        # Getting the type of 'arg1' (line 86)
        arg1_374449 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 19), 'arg1')
        # Obtaining the member '__getitem__' of a type (line 86)
        getitem___374450 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 86, 12), arg1_374449, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 86)
        subscript_call_result_374451 = invoke(stypy.reporting.localization.Localization(__file__, 86, 12), getitem___374450, int_374448)
        
        # Assigning a type to the variable 'tuple_var_assignment_374341' (line 86)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 86, 12), 'tuple_var_assignment_374341', subscript_call_result_374451)
        
        # Assigning a Name to a Name (line 86):
        # Getting the type of 'tuple_var_assignment_374340' (line 86)
        tuple_var_assignment_374340_374452 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 12), 'tuple_var_assignment_374340')
        # Assigning a type to the variable 'M' (line 86)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 86, 12), 'M', tuple_var_assignment_374340_374452)
        
        # Assigning a Name to a Name (line 86):
        # Getting the type of 'tuple_var_assignment_374341' (line 86)
        tuple_var_assignment_374341_374453 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 12), 'tuple_var_assignment_374341')
        # Assigning a type to the variable 'N' (line 86)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 86, 15), 'N', tuple_var_assignment_374341_374453)
        
        # Assigning a Tuple to a Attribute (line 87):
        
        # Assigning a Tuple to a Attribute (line 87):
        
        # Obtaining an instance of the builtin type 'tuple' (line 87)
        tuple_374454 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 87, 26), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 87)
        # Adding element type (line 87)
        # Getting the type of 'M' (line 87)
        M_374455 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 26), 'M')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 87, 26), tuple_374454, M_374455)
        # Adding element type (line 87)
        # Getting the type of 'N' (line 87)
        N_374456 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 29), 'N')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 87, 26), tuple_374454, N_374456)
        
        # Getting the type of 'self' (line 87)
        self_374457 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 12), 'self')
        # Setting the type of the member 'shape' of a type (line 87)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 87, 12), self_374457, 'shape', tuple_374454)
        # SSA branch for the else part of an if statement (line 85)
        module_type_store.open_ssa_branch('else')
        
        
        # Call to isspmatrix(...): (line 88)
        # Processing the call arguments (line 88)
        # Getting the type of 'arg1' (line 88)
        arg1_374459 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 24), 'arg1', False)
        # Processing the call keyword arguments (line 88)
        kwargs_374460 = {}
        # Getting the type of 'isspmatrix' (line 88)
        isspmatrix_374458 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 13), 'isspmatrix', False)
        # Calling isspmatrix(args, kwargs) (line 88)
        isspmatrix_call_result_374461 = invoke(stypy.reporting.localization.Localization(__file__, 88, 13), isspmatrix_374458, *[arg1_374459], **kwargs_374460)
        
        # Testing the type of an if condition (line 88)
        if_condition_374462 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 88, 13), isspmatrix_call_result_374461)
        # Assigning a type to the variable 'if_condition_374462' (line 88)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 88, 13), 'if_condition_374462', if_condition_374462)
        # SSA begins for if statement (line 88)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        
        # Evaluating a boolean operation
        
        # Call to isspmatrix_dok(...): (line 89)
        # Processing the call arguments (line 89)
        # Getting the type of 'arg1' (line 89)
        arg1_374464 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 30), 'arg1', False)
        # Processing the call keyword arguments (line 89)
        kwargs_374465 = {}
        # Getting the type of 'isspmatrix_dok' (line 89)
        isspmatrix_dok_374463 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 15), 'isspmatrix_dok', False)
        # Calling isspmatrix_dok(args, kwargs) (line 89)
        isspmatrix_dok_call_result_374466 = invoke(stypy.reporting.localization.Localization(__file__, 89, 15), isspmatrix_dok_374463, *[arg1_374464], **kwargs_374465)
        
        # Getting the type of 'copy' (line 89)
        copy_374467 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 40), 'copy')
        # Applying the binary operator 'and' (line 89)
        result_and_keyword_374468 = python_operator(stypy.reporting.localization.Localization(__file__, 89, 15), 'and', isspmatrix_dok_call_result_374466, copy_374467)
        
        # Testing the type of an if condition (line 89)
        if_condition_374469 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 89, 12), result_and_keyword_374468)
        # Assigning a type to the variable 'if_condition_374469' (line 89)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 89, 12), 'if_condition_374469', if_condition_374469)
        # SSA begins for if statement (line 89)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 90):
        
        # Assigning a Call to a Name (line 90):
        
        # Call to copy(...): (line 90)
        # Processing the call keyword arguments (line 90)
        kwargs_374472 = {}
        # Getting the type of 'arg1' (line 90)
        arg1_374470 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 23), 'arg1', False)
        # Obtaining the member 'copy' of a type (line 90)
        copy_374471 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 90, 23), arg1_374470, 'copy')
        # Calling copy(args, kwargs) (line 90)
        copy_call_result_374473 = invoke(stypy.reporting.localization.Localization(__file__, 90, 23), copy_374471, *[], **kwargs_374472)
        
        # Assigning a type to the variable 'arg1' (line 90)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 90, 16), 'arg1', copy_call_result_374473)
        # SSA branch for the else part of an if statement (line 89)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Call to a Name (line 92):
        
        # Assigning a Call to a Name (line 92):
        
        # Call to todok(...): (line 92)
        # Processing the call keyword arguments (line 92)
        kwargs_374476 = {}
        # Getting the type of 'arg1' (line 92)
        arg1_374474 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 23), 'arg1', False)
        # Obtaining the member 'todok' of a type (line 92)
        todok_374475 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 92, 23), arg1_374474, 'todok')
        # Calling todok(args, kwargs) (line 92)
        todok_call_result_374477 = invoke(stypy.reporting.localization.Localization(__file__, 92, 23), todok_374475, *[], **kwargs_374476)
        
        # Assigning a type to the variable 'arg1' (line 92)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 92, 16), 'arg1', todok_call_result_374477)
        # SSA join for if statement (line 89)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Type idiom detected: calculating its left and rigth part (line 94)
        # Getting the type of 'dtype' (line 94)
        dtype_374478 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 12), 'dtype')
        # Getting the type of 'None' (line 94)
        None_374479 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 28), 'None')
        
        (may_be_374480, more_types_in_union_374481) = may_not_be_none(dtype_374478, None_374479)

        if may_be_374480:

            if more_types_in_union_374481:
                # Runtime conditional SSA (line 94)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a Call to a Name (line 95):
            
            # Assigning a Call to a Name (line 95):
            
            # Call to astype(...): (line 95)
            # Processing the call arguments (line 95)
            # Getting the type of 'dtype' (line 95)
            dtype_374484 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 35), 'dtype', False)
            # Processing the call keyword arguments (line 95)
            kwargs_374485 = {}
            # Getting the type of 'arg1' (line 95)
            arg1_374482 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 23), 'arg1', False)
            # Obtaining the member 'astype' of a type (line 95)
            astype_374483 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 95, 23), arg1_374482, 'astype')
            # Calling astype(args, kwargs) (line 95)
            astype_call_result_374486 = invoke(stypy.reporting.localization.Localization(__file__, 95, 23), astype_374483, *[dtype_374484], **kwargs_374485)
            
            # Assigning a type to the variable 'arg1' (line 95)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 95, 16), 'arg1', astype_call_result_374486)

            if more_types_in_union_374481:
                # SSA join for if statement (line 94)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Call to update(...): (line 97)
        # Processing the call arguments (line 97)
        # Getting the type of 'self' (line 97)
        self_374489 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 24), 'self', False)
        # Getting the type of 'arg1' (line 97)
        arg1_374490 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 30), 'arg1', False)
        # Processing the call keyword arguments (line 97)
        kwargs_374491 = {}
        # Getting the type of 'dict' (line 97)
        dict_374487 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 12), 'dict', False)
        # Obtaining the member 'update' of a type (line 97)
        update_374488 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 97, 12), dict_374487, 'update')
        # Calling update(args, kwargs) (line 97)
        update_call_result_374492 = invoke(stypy.reporting.localization.Localization(__file__, 97, 12), update_374488, *[self_374489, arg1_374490], **kwargs_374491)
        
        
        # Assigning a Attribute to a Attribute (line 98):
        
        # Assigning a Attribute to a Attribute (line 98):
        # Getting the type of 'arg1' (line 98)
        arg1_374493 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 25), 'arg1')
        # Obtaining the member 'shape' of a type (line 98)
        shape_374494 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 98, 25), arg1_374493, 'shape')
        # Getting the type of 'self' (line 98)
        self_374495 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 12), 'self')
        # Setting the type of the member 'shape' of a type (line 98)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 98, 12), self_374495, 'shape', shape_374494)
        
        # Assigning a Attribute to a Attribute (line 99):
        
        # Assigning a Attribute to a Attribute (line 99):
        # Getting the type of 'arg1' (line 99)
        arg1_374496 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 25), 'arg1')
        # Obtaining the member 'dtype' of a type (line 99)
        dtype_374497 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 99, 25), arg1_374496, 'dtype')
        # Getting the type of 'self' (line 99)
        self_374498 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 12), 'self')
        # Setting the type of the member 'dtype' of a type (line 99)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 99, 12), self_374498, 'dtype', dtype_374497)
        # SSA branch for the else part of an if statement (line 88)
        module_type_store.open_ssa_branch('else')
        
        
        # SSA begins for try-except statement (line 101)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
        
        # Assigning a Call to a Name (line 102):
        
        # Assigning a Call to a Name (line 102):
        
        # Call to asarray(...): (line 102)
        # Processing the call arguments (line 102)
        # Getting the type of 'arg1' (line 102)
        arg1_374501 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 34), 'arg1', False)
        # Processing the call keyword arguments (line 102)
        kwargs_374502 = {}
        # Getting the type of 'np' (line 102)
        np_374499 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 23), 'np', False)
        # Obtaining the member 'asarray' of a type (line 102)
        asarray_374500 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 102, 23), np_374499, 'asarray')
        # Calling asarray(args, kwargs) (line 102)
        asarray_call_result_374503 = invoke(stypy.reporting.localization.Localization(__file__, 102, 23), asarray_374500, *[arg1_374501], **kwargs_374502)
        
        # Assigning a type to the variable 'arg1' (line 102)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 102, 16), 'arg1', asarray_call_result_374503)
        # SSA branch for the except part of a try statement (line 101)
        # SSA branch for the except '<any exception>' branch of a try statement (line 101)
        module_type_store.open_ssa_branch('except')
        
        # Call to TypeError(...): (line 104)
        # Processing the call arguments (line 104)
        str_374505 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 104, 32), 'str', 'Invalid input format.')
        # Processing the call keyword arguments (line 104)
        kwargs_374506 = {}
        # Getting the type of 'TypeError' (line 104)
        TypeError_374504 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 22), 'TypeError', False)
        # Calling TypeError(args, kwargs) (line 104)
        TypeError_call_result_374507 = invoke(stypy.reporting.localization.Localization(__file__, 104, 22), TypeError_374504, *[str_374505], **kwargs_374506)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 104, 16), TypeError_call_result_374507, 'raise parameter', BaseException)
        # SSA join for try-except statement (line 101)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        
        # Call to len(...): (line 106)
        # Processing the call arguments (line 106)
        # Getting the type of 'arg1' (line 106)
        arg1_374509 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 19), 'arg1', False)
        # Obtaining the member 'shape' of a type (line 106)
        shape_374510 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 106, 19), arg1_374509, 'shape')
        # Processing the call keyword arguments (line 106)
        kwargs_374511 = {}
        # Getting the type of 'len' (line 106)
        len_374508 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 15), 'len', False)
        # Calling len(args, kwargs) (line 106)
        len_call_result_374512 = invoke(stypy.reporting.localization.Localization(__file__, 106, 15), len_374508, *[shape_374510], **kwargs_374511)
        
        int_374513 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 106, 34), 'int')
        # Applying the binary operator '!=' (line 106)
        result_ne_374514 = python_operator(stypy.reporting.localization.Localization(__file__, 106, 15), '!=', len_call_result_374512, int_374513)
        
        # Testing the type of an if condition (line 106)
        if_condition_374515 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 106, 12), result_ne_374514)
        # Assigning a type to the variable 'if_condition_374515' (line 106)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 106, 12), 'if_condition_374515', if_condition_374515)
        # SSA begins for if statement (line 106)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to TypeError(...): (line 107)
        # Processing the call arguments (line 107)
        str_374517 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 107, 32), 'str', 'Expected rank <=2 dense array or matrix.')
        # Processing the call keyword arguments (line 107)
        kwargs_374518 = {}
        # Getting the type of 'TypeError' (line 107)
        TypeError_374516 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 22), 'TypeError', False)
        # Calling TypeError(args, kwargs) (line 107)
        TypeError_call_result_374519 = invoke(stypy.reporting.localization.Localization(__file__, 107, 22), TypeError_374516, *[str_374517], **kwargs_374518)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 107, 16), TypeError_call_result_374519, 'raise parameter', BaseException)
        # SSA join for if statement (line 106)
        module_type_store = module_type_store.join_ssa_context()
        
        stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 109, 12))
        
        # 'from scipy.sparse.coo import coo_matrix' statement (line 109)
        update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/sparse/')
        import_374520 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 109, 12), 'scipy.sparse.coo')

        if (type(import_374520) is not StypyTypeError):

            if (import_374520 != 'pyd_module'):
                __import__(import_374520)
                sys_modules_374521 = sys.modules[import_374520]
                import_from_module(stypy.reporting.localization.Localization(__file__, 109, 12), 'scipy.sparse.coo', sys_modules_374521.module_type_store, module_type_store, ['coo_matrix'])
                nest_module(stypy.reporting.localization.Localization(__file__, 109, 12), __file__, sys_modules_374521, sys_modules_374521.module_type_store, module_type_store)
            else:
                from scipy.sparse.coo import coo_matrix

                import_from_module(stypy.reporting.localization.Localization(__file__, 109, 12), 'scipy.sparse.coo', None, module_type_store, ['coo_matrix'], [coo_matrix])

        else:
            # Assigning a type to the variable 'scipy.sparse.coo' (line 109)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 109, 12), 'scipy.sparse.coo', import_374520)

        remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/sparse/')
        
        
        # Assigning a Call to a Name (line 110):
        
        # Assigning a Call to a Name (line 110):
        
        # Call to todok(...): (line 110)
        # Processing the call keyword arguments (line 110)
        kwargs_374529 = {}
        
        # Call to coo_matrix(...): (line 110)
        # Processing the call arguments (line 110)
        # Getting the type of 'arg1' (line 110)
        arg1_374523 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 27), 'arg1', False)
        # Processing the call keyword arguments (line 110)
        # Getting the type of 'dtype' (line 110)
        dtype_374524 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 39), 'dtype', False)
        keyword_374525 = dtype_374524
        kwargs_374526 = {'dtype': keyword_374525}
        # Getting the type of 'coo_matrix' (line 110)
        coo_matrix_374522 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 16), 'coo_matrix', False)
        # Calling coo_matrix(args, kwargs) (line 110)
        coo_matrix_call_result_374527 = invoke(stypy.reporting.localization.Localization(__file__, 110, 16), coo_matrix_374522, *[arg1_374523], **kwargs_374526)
        
        # Obtaining the member 'todok' of a type (line 110)
        todok_374528 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 110, 16), coo_matrix_call_result_374527, 'todok')
        # Calling todok(args, kwargs) (line 110)
        todok_call_result_374530 = invoke(stypy.reporting.localization.Localization(__file__, 110, 16), todok_374528, *[], **kwargs_374529)
        
        # Assigning a type to the variable 'd' (line 110)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 110, 12), 'd', todok_call_result_374530)
        
        # Call to update(...): (line 111)
        # Processing the call arguments (line 111)
        # Getting the type of 'self' (line 111)
        self_374533 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 24), 'self', False)
        # Getting the type of 'd' (line 111)
        d_374534 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 30), 'd', False)
        # Processing the call keyword arguments (line 111)
        kwargs_374535 = {}
        # Getting the type of 'dict' (line 111)
        dict_374531 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 12), 'dict', False)
        # Obtaining the member 'update' of a type (line 111)
        update_374532 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 111, 12), dict_374531, 'update')
        # Calling update(args, kwargs) (line 111)
        update_call_result_374536 = invoke(stypy.reporting.localization.Localization(__file__, 111, 12), update_374532, *[self_374533, d_374534], **kwargs_374535)
        
        
        # Assigning a Attribute to a Attribute (line 112):
        
        # Assigning a Attribute to a Attribute (line 112):
        # Getting the type of 'arg1' (line 112)
        arg1_374537 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 25), 'arg1')
        # Obtaining the member 'shape' of a type (line 112)
        shape_374538 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 112, 25), arg1_374537, 'shape')
        # Getting the type of 'self' (line 112)
        self_374539 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 12), 'self')
        # Setting the type of the member 'shape' of a type (line 112)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 112, 12), self_374539, 'shape', shape_374538)
        
        # Assigning a Attribute to a Attribute (line 113):
        
        # Assigning a Attribute to a Attribute (line 113):
        # Getting the type of 'd' (line 113)
        d_374540 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 25), 'd')
        # Obtaining the member 'dtype' of a type (line 113)
        dtype_374541 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 113, 25), d_374540, 'dtype')
        # Getting the type of 'self' (line 113)
        self_374542 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 12), 'self')
        # Setting the type of the member 'dtype' of a type (line 113)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 113, 12), self_374542, 'dtype', dtype_374541)
        # SSA join for if statement (line 88)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 85)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def update(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'update'
        module_type_store = module_type_store.open_function_context('update', 115, 4, False)
        # Assigning a type to the variable 'self' (line 116)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 116, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        dok_matrix.update.__dict__.__setitem__('stypy_localization', localization)
        dok_matrix.update.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        dok_matrix.update.__dict__.__setitem__('stypy_type_store', module_type_store)
        dok_matrix.update.__dict__.__setitem__('stypy_function_name', 'dok_matrix.update')
        dok_matrix.update.__dict__.__setitem__('stypy_param_names_list', ['val'])
        dok_matrix.update.__dict__.__setitem__('stypy_varargs_param_name', None)
        dok_matrix.update.__dict__.__setitem__('stypy_kwargs_param_name', None)
        dok_matrix.update.__dict__.__setitem__('stypy_call_defaults', defaults)
        dok_matrix.update.__dict__.__setitem__('stypy_call_varargs', varargs)
        dok_matrix.update.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        dok_matrix.update.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'dok_matrix.update', ['val'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'update', localization, ['val'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'update(...)' code ##################

        
        # Call to NotImplementedError(...): (line 117)
        # Processing the call arguments (line 117)
        str_374544 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 117, 34), 'str', 'Direct modification to dok_matrix element is not allowed.')
        # Processing the call keyword arguments (line 117)
        kwargs_374545 = {}
        # Getting the type of 'NotImplementedError' (line 117)
        NotImplementedError_374543 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 14), 'NotImplementedError', False)
        # Calling NotImplementedError(args, kwargs) (line 117)
        NotImplementedError_call_result_374546 = invoke(stypy.reporting.localization.Localization(__file__, 117, 14), NotImplementedError_374543, *[str_374544], **kwargs_374545)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 117, 8), NotImplementedError_call_result_374546, 'raise parameter', BaseException)
        
        # ################# End of 'update(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'update' in the type store
        # Getting the type of 'stypy_return_type' (line 115)
        stypy_return_type_374547 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_374547)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'update'
        return stypy_return_type_374547


    @norecursion
    def _update(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_update'
        module_type_store = module_type_store.open_function_context('_update', 120, 4, False)
        # Assigning a type to the variable 'self' (line 121)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 121, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        dok_matrix._update.__dict__.__setitem__('stypy_localization', localization)
        dok_matrix._update.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        dok_matrix._update.__dict__.__setitem__('stypy_type_store', module_type_store)
        dok_matrix._update.__dict__.__setitem__('stypy_function_name', 'dok_matrix._update')
        dok_matrix._update.__dict__.__setitem__('stypy_param_names_list', ['data'])
        dok_matrix._update.__dict__.__setitem__('stypy_varargs_param_name', None)
        dok_matrix._update.__dict__.__setitem__('stypy_kwargs_param_name', None)
        dok_matrix._update.__dict__.__setitem__('stypy_call_defaults', defaults)
        dok_matrix._update.__dict__.__setitem__('stypy_call_varargs', varargs)
        dok_matrix._update.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        dok_matrix._update.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'dok_matrix._update', ['data'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_update', localization, ['data'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_update(...)' code ##################

        str_374548 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 123, (-1)), 'str', 'An update method for dict data defined for direct access to\n        `dok_matrix` data. Main purpose is to be used for effcient conversion\n        from other spmatrix classes. Has no checking if `data` is valid.')
        
        # Call to update(...): (line 124)
        # Processing the call arguments (line 124)
        # Getting the type of 'self' (line 124)
        self_374551 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 27), 'self', False)
        # Getting the type of 'data' (line 124)
        data_374552 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 33), 'data', False)
        # Processing the call keyword arguments (line 124)
        kwargs_374553 = {}
        # Getting the type of 'dict' (line 124)
        dict_374549 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 15), 'dict', False)
        # Obtaining the member 'update' of a type (line 124)
        update_374550 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 124, 15), dict_374549, 'update')
        # Calling update(args, kwargs) (line 124)
        update_call_result_374554 = invoke(stypy.reporting.localization.Localization(__file__, 124, 15), update_374550, *[self_374551, data_374552], **kwargs_374553)
        
        # Assigning a type to the variable 'stypy_return_type' (line 124)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 124, 8), 'stypy_return_type', update_call_result_374554)
        
        # ################# End of '_update(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_update' in the type store
        # Getting the type of 'stypy_return_type' (line 120)
        stypy_return_type_374555 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_374555)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_update'
        return stypy_return_type_374555


    @norecursion
    def getnnz(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 126)
        None_374556 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 26), 'None')
        defaults = [None_374556]
        # Create a new context for function 'getnnz'
        module_type_store = module_type_store.open_function_context('getnnz', 126, 4, False)
        # Assigning a type to the variable 'self' (line 127)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 127, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        dok_matrix.getnnz.__dict__.__setitem__('stypy_localization', localization)
        dok_matrix.getnnz.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        dok_matrix.getnnz.__dict__.__setitem__('stypy_type_store', module_type_store)
        dok_matrix.getnnz.__dict__.__setitem__('stypy_function_name', 'dok_matrix.getnnz')
        dok_matrix.getnnz.__dict__.__setitem__('stypy_param_names_list', ['axis'])
        dok_matrix.getnnz.__dict__.__setitem__('stypy_varargs_param_name', None)
        dok_matrix.getnnz.__dict__.__setitem__('stypy_kwargs_param_name', None)
        dok_matrix.getnnz.__dict__.__setitem__('stypy_call_defaults', defaults)
        dok_matrix.getnnz.__dict__.__setitem__('stypy_call_varargs', varargs)
        dok_matrix.getnnz.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        dok_matrix.getnnz.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'dok_matrix.getnnz', ['axis'], None, None, defaults, varargs, kwargs)

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

        
        # Type idiom detected: calculating its left and rigth part (line 127)
        # Getting the type of 'axis' (line 127)
        axis_374557 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 8), 'axis')
        # Getting the type of 'None' (line 127)
        None_374558 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 23), 'None')
        
        (may_be_374559, more_types_in_union_374560) = may_not_be_none(axis_374557, None_374558)

        if may_be_374559:

            if more_types_in_union_374560:
                # Runtime conditional SSA (line 127)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Call to NotImplementedError(...): (line 128)
            # Processing the call arguments (line 128)
            str_374562 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 128, 38), 'str', 'getnnz over an axis is not implemented for DOK format.')
            # Processing the call keyword arguments (line 128)
            kwargs_374563 = {}
            # Getting the type of 'NotImplementedError' (line 128)
            NotImplementedError_374561 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 18), 'NotImplementedError', False)
            # Calling NotImplementedError(args, kwargs) (line 128)
            NotImplementedError_call_result_374564 = invoke(stypy.reporting.localization.Localization(__file__, 128, 18), NotImplementedError_374561, *[str_374562], **kwargs_374563)
            
            ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 128, 12), NotImplementedError_call_result_374564, 'raise parameter', BaseException)

            if more_types_in_union_374560:
                # SSA join for if statement (line 127)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Call to __len__(...): (line 130)
        # Processing the call arguments (line 130)
        # Getting the type of 'self' (line 130)
        self_374567 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 28), 'self', False)
        # Processing the call keyword arguments (line 130)
        kwargs_374568 = {}
        # Getting the type of 'dict' (line 130)
        dict_374565 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 15), 'dict', False)
        # Obtaining the member '__len__' of a type (line 130)
        len___374566 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 130, 15), dict_374565, '__len__')
        # Calling __len__(args, kwargs) (line 130)
        len___call_result_374569 = invoke(stypy.reporting.localization.Localization(__file__, 130, 15), len___374566, *[self_374567], **kwargs_374568)
        
        # Assigning a type to the variable 'stypy_return_type' (line 130)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 130, 8), 'stypy_return_type', len___call_result_374569)
        
        # ################# End of 'getnnz(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'getnnz' in the type store
        # Getting the type of 'stypy_return_type' (line 126)
        stypy_return_type_374570 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_374570)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'getnnz'
        return stypy_return_type_374570


    @norecursion
    def count_nonzero(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'count_nonzero'
        module_type_store = module_type_store.open_function_context('count_nonzero', 132, 4, False)
        # Assigning a type to the variable 'self' (line 133)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 133, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        dok_matrix.count_nonzero.__dict__.__setitem__('stypy_localization', localization)
        dok_matrix.count_nonzero.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        dok_matrix.count_nonzero.__dict__.__setitem__('stypy_type_store', module_type_store)
        dok_matrix.count_nonzero.__dict__.__setitem__('stypy_function_name', 'dok_matrix.count_nonzero')
        dok_matrix.count_nonzero.__dict__.__setitem__('stypy_param_names_list', [])
        dok_matrix.count_nonzero.__dict__.__setitem__('stypy_varargs_param_name', None)
        dok_matrix.count_nonzero.__dict__.__setitem__('stypy_kwargs_param_name', None)
        dok_matrix.count_nonzero.__dict__.__setitem__('stypy_call_defaults', defaults)
        dok_matrix.count_nonzero.__dict__.__setitem__('stypy_call_varargs', varargs)
        dok_matrix.count_nonzero.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        dok_matrix.count_nonzero.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'dok_matrix.count_nonzero', [], None, None, defaults, varargs, kwargs)

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

        
        # Call to sum(...): (line 133)
        # Processing the call arguments (line 133)
        # Calculating generator expression
        module_type_store = module_type_store.open_function_context('list comprehension expression', 133, 19, True)
        # Calculating comprehension expression
        
        # Call to itervalues(...): (line 133)
        # Processing the call arguments (line 133)
        # Getting the type of 'self' (line 133)
        self_374576 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 46), 'self', False)
        # Processing the call keyword arguments (line 133)
        kwargs_374577 = {}
        # Getting the type of 'itervalues' (line 133)
        itervalues_374575 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 35), 'itervalues', False)
        # Calling itervalues(args, kwargs) (line 133)
        itervalues_call_result_374578 = invoke(stypy.reporting.localization.Localization(__file__, 133, 35), itervalues_374575, *[self_374576], **kwargs_374577)
        
        comprehension_374579 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 133, 19), itervalues_call_result_374578)
        # Assigning a type to the variable 'x' (line 133)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 133, 19), 'x', comprehension_374579)
        
        # Getting the type of 'x' (line 133)
        x_374572 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 19), 'x', False)
        int_374573 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 133, 24), 'int')
        # Applying the binary operator '!=' (line 133)
        result_ne_374574 = python_operator(stypy.reporting.localization.Localization(__file__, 133, 19), '!=', x_374572, int_374573)
        
        list_374580 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 133, 19), 'list')
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 133, 19), list_374580, result_ne_374574)
        # Processing the call keyword arguments (line 133)
        kwargs_374581 = {}
        # Getting the type of 'sum' (line 133)
        sum_374571 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 15), 'sum', False)
        # Calling sum(args, kwargs) (line 133)
        sum_call_result_374582 = invoke(stypy.reporting.localization.Localization(__file__, 133, 15), sum_374571, *[list_374580], **kwargs_374581)
        
        # Assigning a type to the variable 'stypy_return_type' (line 133)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 133, 8), 'stypy_return_type', sum_call_result_374582)
        
        # ################# End of 'count_nonzero(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'count_nonzero' in the type store
        # Getting the type of 'stypy_return_type' (line 132)
        stypy_return_type_374583 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_374583)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'count_nonzero'
        return stypy_return_type_374583

    
    # Assigning a Attribute to a Attribute (line 135):
    
    # Assigning a Attribute to a Attribute (line 136):

    @norecursion
    def __len__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__len__'
        module_type_store = module_type_store.open_function_context('__len__', 138, 4, False)
        # Assigning a type to the variable 'self' (line 139)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 139, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        dok_matrix.__len__.__dict__.__setitem__('stypy_localization', localization)
        dok_matrix.__len__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        dok_matrix.__len__.__dict__.__setitem__('stypy_type_store', module_type_store)
        dok_matrix.__len__.__dict__.__setitem__('stypy_function_name', 'dok_matrix.__len__')
        dok_matrix.__len__.__dict__.__setitem__('stypy_param_names_list', [])
        dok_matrix.__len__.__dict__.__setitem__('stypy_varargs_param_name', None)
        dok_matrix.__len__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        dok_matrix.__len__.__dict__.__setitem__('stypy_call_defaults', defaults)
        dok_matrix.__len__.__dict__.__setitem__('stypy_call_varargs', varargs)
        dok_matrix.__len__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        dok_matrix.__len__.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'dok_matrix.__len__', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__len__', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__len__(...)' code ##################

        
        # Call to __len__(...): (line 139)
        # Processing the call arguments (line 139)
        # Getting the type of 'self' (line 139)
        self_374586 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 28), 'self', False)
        # Processing the call keyword arguments (line 139)
        kwargs_374587 = {}
        # Getting the type of 'dict' (line 139)
        dict_374584 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 15), 'dict', False)
        # Obtaining the member '__len__' of a type (line 139)
        len___374585 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 139, 15), dict_374584, '__len__')
        # Calling __len__(args, kwargs) (line 139)
        len___call_result_374588 = invoke(stypy.reporting.localization.Localization(__file__, 139, 15), len___374585, *[self_374586], **kwargs_374587)
        
        # Assigning a type to the variable 'stypy_return_type' (line 139)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 139, 8), 'stypy_return_type', len___call_result_374588)
        
        # ################# End of '__len__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__len__' in the type store
        # Getting the type of 'stypy_return_type' (line 138)
        stypy_return_type_374589 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_374589)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__len__'
        return stypy_return_type_374589


    @norecursion
    def get(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        float_374590 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 141, 31), 'float')
        defaults = [float_374590]
        # Create a new context for function 'get'
        module_type_store = module_type_store.open_function_context('get', 141, 4, False)
        # Assigning a type to the variable 'self' (line 142)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 142, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        dok_matrix.get.__dict__.__setitem__('stypy_localization', localization)
        dok_matrix.get.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        dok_matrix.get.__dict__.__setitem__('stypy_type_store', module_type_store)
        dok_matrix.get.__dict__.__setitem__('stypy_function_name', 'dok_matrix.get')
        dok_matrix.get.__dict__.__setitem__('stypy_param_names_list', ['key', 'default'])
        dok_matrix.get.__dict__.__setitem__('stypy_varargs_param_name', None)
        dok_matrix.get.__dict__.__setitem__('stypy_kwargs_param_name', None)
        dok_matrix.get.__dict__.__setitem__('stypy_call_defaults', defaults)
        dok_matrix.get.__dict__.__setitem__('stypy_call_varargs', varargs)
        dok_matrix.get.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        dok_matrix.get.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'dok_matrix.get', ['key', 'default'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'get', localization, ['key', 'default'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'get(...)' code ##################

        str_374591 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 144, (-1)), 'str', 'This overrides the dict.get method, providing type checking\n        but otherwise equivalent functionality.\n        ')
        
        
        # SSA begins for try-except statement (line 145)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
        
        # Assigning a Name to a Tuple (line 146):
        
        # Assigning a Subscript to a Name (line 146):
        
        # Obtaining the type of the subscript
        int_374592 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 146, 12), 'int')
        # Getting the type of 'key' (line 146)
        key_374593 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 19), 'key')
        # Obtaining the member '__getitem__' of a type (line 146)
        getitem___374594 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 146, 12), key_374593, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 146)
        subscript_call_result_374595 = invoke(stypy.reporting.localization.Localization(__file__, 146, 12), getitem___374594, int_374592)
        
        # Assigning a type to the variable 'tuple_var_assignment_374342' (line 146)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 146, 12), 'tuple_var_assignment_374342', subscript_call_result_374595)
        
        # Assigning a Subscript to a Name (line 146):
        
        # Obtaining the type of the subscript
        int_374596 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 146, 12), 'int')
        # Getting the type of 'key' (line 146)
        key_374597 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 19), 'key')
        # Obtaining the member '__getitem__' of a type (line 146)
        getitem___374598 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 146, 12), key_374597, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 146)
        subscript_call_result_374599 = invoke(stypy.reporting.localization.Localization(__file__, 146, 12), getitem___374598, int_374596)
        
        # Assigning a type to the variable 'tuple_var_assignment_374343' (line 146)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 146, 12), 'tuple_var_assignment_374343', subscript_call_result_374599)
        
        # Assigning a Name to a Name (line 146):
        # Getting the type of 'tuple_var_assignment_374342' (line 146)
        tuple_var_assignment_374342_374600 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 12), 'tuple_var_assignment_374342')
        # Assigning a type to the variable 'i' (line 146)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 146, 12), 'i', tuple_var_assignment_374342_374600)
        
        # Assigning a Name to a Name (line 146):
        # Getting the type of 'tuple_var_assignment_374343' (line 146)
        tuple_var_assignment_374343_374601 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 12), 'tuple_var_assignment_374343')
        # Assigning a type to the variable 'j' (line 146)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 146, 15), 'j', tuple_var_assignment_374343_374601)
        # Evaluating assert statement condition
        
        # Evaluating a boolean operation
        
        # Call to isintlike(...): (line 147)
        # Processing the call arguments (line 147)
        # Getting the type of 'i' (line 147)
        i_374603 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 29), 'i', False)
        # Processing the call keyword arguments (line 147)
        kwargs_374604 = {}
        # Getting the type of 'isintlike' (line 147)
        isintlike_374602 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 19), 'isintlike', False)
        # Calling isintlike(args, kwargs) (line 147)
        isintlike_call_result_374605 = invoke(stypy.reporting.localization.Localization(__file__, 147, 19), isintlike_374602, *[i_374603], **kwargs_374604)
        
        
        # Call to isintlike(...): (line 147)
        # Processing the call arguments (line 147)
        # Getting the type of 'j' (line 147)
        j_374607 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 46), 'j', False)
        # Processing the call keyword arguments (line 147)
        kwargs_374608 = {}
        # Getting the type of 'isintlike' (line 147)
        isintlike_374606 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 36), 'isintlike', False)
        # Calling isintlike(args, kwargs) (line 147)
        isintlike_call_result_374609 = invoke(stypy.reporting.localization.Localization(__file__, 147, 36), isintlike_374606, *[j_374607], **kwargs_374608)
        
        # Applying the binary operator 'and' (line 147)
        result_and_keyword_374610 = python_operator(stypy.reporting.localization.Localization(__file__, 147, 19), 'and', isintlike_call_result_374605, isintlike_call_result_374609)
        
        # SSA branch for the except part of a try statement (line 145)
        # SSA branch for the except 'Tuple' branch of a try statement (line 145)
        module_type_store.open_ssa_branch('except')
        
        # Call to IndexError(...): (line 149)
        # Processing the call arguments (line 149)
        str_374612 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 149, 29), 'str', 'Index must be a pair of integers.')
        # Processing the call keyword arguments (line 149)
        kwargs_374613 = {}
        # Getting the type of 'IndexError' (line 149)
        IndexError_374611 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 18), 'IndexError', False)
        # Calling IndexError(args, kwargs) (line 149)
        IndexError_call_result_374614 = invoke(stypy.reporting.localization.Localization(__file__, 149, 18), IndexError_374611, *[str_374612], **kwargs_374613)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 149, 12), IndexError_call_result_374614, 'raise parameter', BaseException)
        # SSA join for try-except statement (line 145)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Evaluating a boolean operation
        
        # Getting the type of 'i' (line 150)
        i_374615 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 12), 'i')
        int_374616 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 150, 16), 'int')
        # Applying the binary operator '<' (line 150)
        result_lt_374617 = python_operator(stypy.reporting.localization.Localization(__file__, 150, 12), '<', i_374615, int_374616)
        
        
        # Getting the type of 'i' (line 150)
        i_374618 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 21), 'i')
        
        # Obtaining the type of the subscript
        int_374619 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 150, 37), 'int')
        # Getting the type of 'self' (line 150)
        self_374620 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 26), 'self')
        # Obtaining the member 'shape' of a type (line 150)
        shape_374621 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 150, 26), self_374620, 'shape')
        # Obtaining the member '__getitem__' of a type (line 150)
        getitem___374622 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 150, 26), shape_374621, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 150)
        subscript_call_result_374623 = invoke(stypy.reporting.localization.Localization(__file__, 150, 26), getitem___374622, int_374619)
        
        # Applying the binary operator '>=' (line 150)
        result_ge_374624 = python_operator(stypy.reporting.localization.Localization(__file__, 150, 21), '>=', i_374618, subscript_call_result_374623)
        
        # Applying the binary operator 'or' (line 150)
        result_or_keyword_374625 = python_operator(stypy.reporting.localization.Localization(__file__, 150, 12), 'or', result_lt_374617, result_ge_374624)
        
        # Getting the type of 'j' (line 150)
        j_374626 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 43), 'j')
        int_374627 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 150, 47), 'int')
        # Applying the binary operator '<' (line 150)
        result_lt_374628 = python_operator(stypy.reporting.localization.Localization(__file__, 150, 43), '<', j_374626, int_374627)
        
        # Applying the binary operator 'or' (line 150)
        result_or_keyword_374629 = python_operator(stypy.reporting.localization.Localization(__file__, 150, 12), 'or', result_or_keyword_374625, result_lt_374628)
        
        # Getting the type of 'j' (line 150)
        j_374630 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 52), 'j')
        
        # Obtaining the type of the subscript
        int_374631 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 150, 68), 'int')
        # Getting the type of 'self' (line 150)
        self_374632 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 57), 'self')
        # Obtaining the member 'shape' of a type (line 150)
        shape_374633 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 150, 57), self_374632, 'shape')
        # Obtaining the member '__getitem__' of a type (line 150)
        getitem___374634 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 150, 57), shape_374633, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 150)
        subscript_call_result_374635 = invoke(stypy.reporting.localization.Localization(__file__, 150, 57), getitem___374634, int_374631)
        
        # Applying the binary operator '>=' (line 150)
        result_ge_374636 = python_operator(stypy.reporting.localization.Localization(__file__, 150, 52), '>=', j_374630, subscript_call_result_374635)
        
        # Applying the binary operator 'or' (line 150)
        result_or_keyword_374637 = python_operator(stypy.reporting.localization.Localization(__file__, 150, 12), 'or', result_or_keyword_374629, result_ge_374636)
        
        # Testing the type of an if condition (line 150)
        if_condition_374638 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 150, 8), result_or_keyword_374637)
        # Assigning a type to the variable 'if_condition_374638' (line 150)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 150, 8), 'if_condition_374638', if_condition_374638)
        # SSA begins for if statement (line 150)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to IndexError(...): (line 151)
        # Processing the call arguments (line 151)
        str_374640 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 151, 29), 'str', 'Index out of bounds.')
        # Processing the call keyword arguments (line 151)
        kwargs_374641 = {}
        # Getting the type of 'IndexError' (line 151)
        IndexError_374639 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 18), 'IndexError', False)
        # Calling IndexError(args, kwargs) (line 151)
        IndexError_call_result_374642 = invoke(stypy.reporting.localization.Localization(__file__, 151, 18), IndexError_374639, *[str_374640], **kwargs_374641)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 151, 12), IndexError_call_result_374642, 'raise parameter', BaseException)
        # SSA join for if statement (line 150)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to get(...): (line 152)
        # Processing the call arguments (line 152)
        # Getting the type of 'self' (line 152)
        self_374645 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 24), 'self', False)
        # Getting the type of 'key' (line 152)
        key_374646 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 30), 'key', False)
        # Getting the type of 'default' (line 152)
        default_374647 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 35), 'default', False)
        # Processing the call keyword arguments (line 152)
        kwargs_374648 = {}
        # Getting the type of 'dict' (line 152)
        dict_374643 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 15), 'dict', False)
        # Obtaining the member 'get' of a type (line 152)
        get_374644 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 152, 15), dict_374643, 'get')
        # Calling get(args, kwargs) (line 152)
        get_call_result_374649 = invoke(stypy.reporting.localization.Localization(__file__, 152, 15), get_374644, *[self_374645, key_374646, default_374647], **kwargs_374648)
        
        # Assigning a type to the variable 'stypy_return_type' (line 152)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 152, 8), 'stypy_return_type', get_call_result_374649)
        
        # ################# End of 'get(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get' in the type store
        # Getting the type of 'stypy_return_type' (line 141)
        stypy_return_type_374650 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_374650)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get'
        return stypy_return_type_374650


    @norecursion
    def __getitem__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__getitem__'
        module_type_store = module_type_store.open_function_context('__getitem__', 154, 4, False)
        # Assigning a type to the variable 'self' (line 155)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 155, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        dok_matrix.__getitem__.__dict__.__setitem__('stypy_localization', localization)
        dok_matrix.__getitem__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        dok_matrix.__getitem__.__dict__.__setitem__('stypy_type_store', module_type_store)
        dok_matrix.__getitem__.__dict__.__setitem__('stypy_function_name', 'dok_matrix.__getitem__')
        dok_matrix.__getitem__.__dict__.__setitem__('stypy_param_names_list', ['index'])
        dok_matrix.__getitem__.__dict__.__setitem__('stypy_varargs_param_name', None)
        dok_matrix.__getitem__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        dok_matrix.__getitem__.__dict__.__setitem__('stypy_call_defaults', defaults)
        dok_matrix.__getitem__.__dict__.__setitem__('stypy_call_varargs', varargs)
        dok_matrix.__getitem__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        dok_matrix.__getitem__.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'dok_matrix.__getitem__', ['index'], None, None, defaults, varargs, kwargs)

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

        str_374651 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 158, (-1)), 'str', 'If key=(i, j) is a pair of integers, return the corresponding\n        element.  If either i or j is a slice or sequence, return a new sparse\n        matrix with just these elements.\n        ')
        
        # Assigning a Call to a Name (line 159):
        
        # Assigning a Call to a Name (line 159):
        
        # Call to type(...): (line 159)
        # Processing the call arguments (line 159)
        int_374655 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 159, 31), 'int')
        # Processing the call keyword arguments (line 159)
        kwargs_374656 = {}
        # Getting the type of 'self' (line 159)
        self_374652 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 15), 'self', False)
        # Obtaining the member 'dtype' of a type (line 159)
        dtype_374653 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 159, 15), self_374652, 'dtype')
        # Obtaining the member 'type' of a type (line 159)
        type_374654 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 159, 15), dtype_374653, 'type')
        # Calling type(args, kwargs) (line 159)
        type_call_result_374657 = invoke(stypy.reporting.localization.Localization(__file__, 159, 15), type_374654, *[int_374655], **kwargs_374656)
        
        # Assigning a type to the variable 'zero' (line 159)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 159, 8), 'zero', type_call_result_374657)
        
        # Assigning a Call to a Tuple (line 160):
        
        # Assigning a Subscript to a Name (line 160):
        
        # Obtaining the type of the subscript
        int_374658 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 160, 8), 'int')
        
        # Call to _unpack_index(...): (line 160)
        # Processing the call arguments (line 160)
        # Getting the type of 'index' (line 160)
        index_374661 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 34), 'index', False)
        # Processing the call keyword arguments (line 160)
        kwargs_374662 = {}
        # Getting the type of 'self' (line 160)
        self_374659 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 15), 'self', False)
        # Obtaining the member '_unpack_index' of a type (line 160)
        _unpack_index_374660 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 160, 15), self_374659, '_unpack_index')
        # Calling _unpack_index(args, kwargs) (line 160)
        _unpack_index_call_result_374663 = invoke(stypy.reporting.localization.Localization(__file__, 160, 15), _unpack_index_374660, *[index_374661], **kwargs_374662)
        
        # Obtaining the member '__getitem__' of a type (line 160)
        getitem___374664 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 160, 8), _unpack_index_call_result_374663, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 160)
        subscript_call_result_374665 = invoke(stypy.reporting.localization.Localization(__file__, 160, 8), getitem___374664, int_374658)
        
        # Assigning a type to the variable 'tuple_var_assignment_374344' (line 160)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 160, 8), 'tuple_var_assignment_374344', subscript_call_result_374665)
        
        # Assigning a Subscript to a Name (line 160):
        
        # Obtaining the type of the subscript
        int_374666 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 160, 8), 'int')
        
        # Call to _unpack_index(...): (line 160)
        # Processing the call arguments (line 160)
        # Getting the type of 'index' (line 160)
        index_374669 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 34), 'index', False)
        # Processing the call keyword arguments (line 160)
        kwargs_374670 = {}
        # Getting the type of 'self' (line 160)
        self_374667 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 15), 'self', False)
        # Obtaining the member '_unpack_index' of a type (line 160)
        _unpack_index_374668 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 160, 15), self_374667, '_unpack_index')
        # Calling _unpack_index(args, kwargs) (line 160)
        _unpack_index_call_result_374671 = invoke(stypy.reporting.localization.Localization(__file__, 160, 15), _unpack_index_374668, *[index_374669], **kwargs_374670)
        
        # Obtaining the member '__getitem__' of a type (line 160)
        getitem___374672 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 160, 8), _unpack_index_call_result_374671, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 160)
        subscript_call_result_374673 = invoke(stypy.reporting.localization.Localization(__file__, 160, 8), getitem___374672, int_374666)
        
        # Assigning a type to the variable 'tuple_var_assignment_374345' (line 160)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 160, 8), 'tuple_var_assignment_374345', subscript_call_result_374673)
        
        # Assigning a Name to a Name (line 160):
        # Getting the type of 'tuple_var_assignment_374344' (line 160)
        tuple_var_assignment_374344_374674 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 8), 'tuple_var_assignment_374344')
        # Assigning a type to the variable 'i' (line 160)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 160, 8), 'i', tuple_var_assignment_374344_374674)
        
        # Assigning a Name to a Name (line 160):
        # Getting the type of 'tuple_var_assignment_374345' (line 160)
        tuple_var_assignment_374345_374675 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 8), 'tuple_var_assignment_374345')
        # Assigning a type to the variable 'j' (line 160)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 160, 11), 'j', tuple_var_assignment_374345_374675)
        
        # Assigning a Call to a Name (line 162):
        
        # Assigning a Call to a Name (line 162):
        
        # Call to isintlike(...): (line 162)
        # Processing the call arguments (line 162)
        # Getting the type of 'i' (line 162)
        i_374677 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 30), 'i', False)
        # Processing the call keyword arguments (line 162)
        kwargs_374678 = {}
        # Getting the type of 'isintlike' (line 162)
        isintlike_374676 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 20), 'isintlike', False)
        # Calling isintlike(args, kwargs) (line 162)
        isintlike_call_result_374679 = invoke(stypy.reporting.localization.Localization(__file__, 162, 20), isintlike_374676, *[i_374677], **kwargs_374678)
        
        # Assigning a type to the variable 'i_intlike' (line 162)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 162, 8), 'i_intlike', isintlike_call_result_374679)
        
        # Assigning a Call to a Name (line 163):
        
        # Assigning a Call to a Name (line 163):
        
        # Call to isintlike(...): (line 163)
        # Processing the call arguments (line 163)
        # Getting the type of 'j' (line 163)
        j_374681 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 30), 'j', False)
        # Processing the call keyword arguments (line 163)
        kwargs_374682 = {}
        # Getting the type of 'isintlike' (line 163)
        isintlike_374680 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 20), 'isintlike', False)
        # Calling isintlike(args, kwargs) (line 163)
        isintlike_call_result_374683 = invoke(stypy.reporting.localization.Localization(__file__, 163, 20), isintlike_374680, *[j_374681], **kwargs_374682)
        
        # Assigning a type to the variable 'j_intlike' (line 163)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 163, 8), 'j_intlike', isintlike_call_result_374683)
        
        
        # Evaluating a boolean operation
        # Getting the type of 'i_intlike' (line 165)
        i_intlike_374684 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 11), 'i_intlike')
        # Getting the type of 'j_intlike' (line 165)
        j_intlike_374685 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 25), 'j_intlike')
        # Applying the binary operator 'and' (line 165)
        result_and_keyword_374686 = python_operator(stypy.reporting.localization.Localization(__file__, 165, 11), 'and', i_intlike_374684, j_intlike_374685)
        
        # Testing the type of an if condition (line 165)
        if_condition_374687 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 165, 8), result_and_keyword_374686)
        # Assigning a type to the variable 'if_condition_374687' (line 165)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 165, 8), 'if_condition_374687', if_condition_374687)
        # SSA begins for if statement (line 165)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 166):
        
        # Assigning a Call to a Name (line 166):
        
        # Call to int(...): (line 166)
        # Processing the call arguments (line 166)
        # Getting the type of 'i' (line 166)
        i_374689 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 20), 'i', False)
        # Processing the call keyword arguments (line 166)
        kwargs_374690 = {}
        # Getting the type of 'int' (line 166)
        int_374688 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 16), 'int', False)
        # Calling int(args, kwargs) (line 166)
        int_call_result_374691 = invoke(stypy.reporting.localization.Localization(__file__, 166, 16), int_374688, *[i_374689], **kwargs_374690)
        
        # Assigning a type to the variable 'i' (line 166)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 166, 12), 'i', int_call_result_374691)
        
        # Assigning a Call to a Name (line 167):
        
        # Assigning a Call to a Name (line 167):
        
        # Call to int(...): (line 167)
        # Processing the call arguments (line 167)
        # Getting the type of 'j' (line 167)
        j_374693 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 20), 'j', False)
        # Processing the call keyword arguments (line 167)
        kwargs_374694 = {}
        # Getting the type of 'int' (line 167)
        int_374692 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 16), 'int', False)
        # Calling int(args, kwargs) (line 167)
        int_call_result_374695 = invoke(stypy.reporting.localization.Localization(__file__, 167, 16), int_374692, *[j_374693], **kwargs_374694)
        
        # Assigning a type to the variable 'j' (line 167)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 167, 12), 'j', int_call_result_374695)
        
        
        # Getting the type of 'i' (line 168)
        i_374696 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 15), 'i')
        int_374697 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 168, 19), 'int')
        # Applying the binary operator '<' (line 168)
        result_lt_374698 = python_operator(stypy.reporting.localization.Localization(__file__, 168, 15), '<', i_374696, int_374697)
        
        # Testing the type of an if condition (line 168)
        if_condition_374699 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 168, 12), result_lt_374698)
        # Assigning a type to the variable 'if_condition_374699' (line 168)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 168, 12), 'if_condition_374699', if_condition_374699)
        # SSA begins for if statement (line 168)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Getting the type of 'i' (line 169)
        i_374700 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 16), 'i')
        
        # Obtaining the type of the subscript
        int_374701 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 169, 32), 'int')
        # Getting the type of 'self' (line 169)
        self_374702 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 21), 'self')
        # Obtaining the member 'shape' of a type (line 169)
        shape_374703 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 169, 21), self_374702, 'shape')
        # Obtaining the member '__getitem__' of a type (line 169)
        getitem___374704 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 169, 21), shape_374703, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 169)
        subscript_call_result_374705 = invoke(stypy.reporting.localization.Localization(__file__, 169, 21), getitem___374704, int_374701)
        
        # Applying the binary operator '+=' (line 169)
        result_iadd_374706 = python_operator(stypy.reporting.localization.Localization(__file__, 169, 16), '+=', i_374700, subscript_call_result_374705)
        # Assigning a type to the variable 'i' (line 169)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 169, 16), 'i', result_iadd_374706)
        
        # SSA join for if statement (line 168)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Evaluating a boolean operation
        
        # Getting the type of 'i' (line 170)
        i_374707 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 15), 'i')
        int_374708 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 170, 19), 'int')
        # Applying the binary operator '<' (line 170)
        result_lt_374709 = python_operator(stypy.reporting.localization.Localization(__file__, 170, 15), '<', i_374707, int_374708)
        
        
        # Getting the type of 'i' (line 170)
        i_374710 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 24), 'i')
        
        # Obtaining the type of the subscript
        int_374711 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 170, 40), 'int')
        # Getting the type of 'self' (line 170)
        self_374712 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 29), 'self')
        # Obtaining the member 'shape' of a type (line 170)
        shape_374713 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 170, 29), self_374712, 'shape')
        # Obtaining the member '__getitem__' of a type (line 170)
        getitem___374714 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 170, 29), shape_374713, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 170)
        subscript_call_result_374715 = invoke(stypy.reporting.localization.Localization(__file__, 170, 29), getitem___374714, int_374711)
        
        # Applying the binary operator '>=' (line 170)
        result_ge_374716 = python_operator(stypy.reporting.localization.Localization(__file__, 170, 24), '>=', i_374710, subscript_call_result_374715)
        
        # Applying the binary operator 'or' (line 170)
        result_or_keyword_374717 = python_operator(stypy.reporting.localization.Localization(__file__, 170, 15), 'or', result_lt_374709, result_ge_374716)
        
        # Testing the type of an if condition (line 170)
        if_condition_374718 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 170, 12), result_or_keyword_374717)
        # Assigning a type to the variable 'if_condition_374718' (line 170)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 170, 12), 'if_condition_374718', if_condition_374718)
        # SSA begins for if statement (line 170)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to IndexError(...): (line 171)
        # Processing the call arguments (line 171)
        str_374720 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 171, 33), 'str', 'Index out of bounds.')
        # Processing the call keyword arguments (line 171)
        kwargs_374721 = {}
        # Getting the type of 'IndexError' (line 171)
        IndexError_374719 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 22), 'IndexError', False)
        # Calling IndexError(args, kwargs) (line 171)
        IndexError_call_result_374722 = invoke(stypy.reporting.localization.Localization(__file__, 171, 22), IndexError_374719, *[str_374720], **kwargs_374721)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 171, 16), IndexError_call_result_374722, 'raise parameter', BaseException)
        # SSA join for if statement (line 170)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Getting the type of 'j' (line 172)
        j_374723 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 15), 'j')
        int_374724 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 172, 19), 'int')
        # Applying the binary operator '<' (line 172)
        result_lt_374725 = python_operator(stypy.reporting.localization.Localization(__file__, 172, 15), '<', j_374723, int_374724)
        
        # Testing the type of an if condition (line 172)
        if_condition_374726 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 172, 12), result_lt_374725)
        # Assigning a type to the variable 'if_condition_374726' (line 172)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 172, 12), 'if_condition_374726', if_condition_374726)
        # SSA begins for if statement (line 172)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Getting the type of 'j' (line 173)
        j_374727 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 16), 'j')
        
        # Obtaining the type of the subscript
        int_374728 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 173, 32), 'int')
        # Getting the type of 'self' (line 173)
        self_374729 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 21), 'self')
        # Obtaining the member 'shape' of a type (line 173)
        shape_374730 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 173, 21), self_374729, 'shape')
        # Obtaining the member '__getitem__' of a type (line 173)
        getitem___374731 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 173, 21), shape_374730, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 173)
        subscript_call_result_374732 = invoke(stypy.reporting.localization.Localization(__file__, 173, 21), getitem___374731, int_374728)
        
        # Applying the binary operator '+=' (line 173)
        result_iadd_374733 = python_operator(stypy.reporting.localization.Localization(__file__, 173, 16), '+=', j_374727, subscript_call_result_374732)
        # Assigning a type to the variable 'j' (line 173)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 173, 16), 'j', result_iadd_374733)
        
        # SSA join for if statement (line 172)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Evaluating a boolean operation
        
        # Getting the type of 'j' (line 174)
        j_374734 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 15), 'j')
        int_374735 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 174, 19), 'int')
        # Applying the binary operator '<' (line 174)
        result_lt_374736 = python_operator(stypy.reporting.localization.Localization(__file__, 174, 15), '<', j_374734, int_374735)
        
        
        # Getting the type of 'j' (line 174)
        j_374737 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 24), 'j')
        
        # Obtaining the type of the subscript
        int_374738 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 174, 40), 'int')
        # Getting the type of 'self' (line 174)
        self_374739 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 29), 'self')
        # Obtaining the member 'shape' of a type (line 174)
        shape_374740 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 174, 29), self_374739, 'shape')
        # Obtaining the member '__getitem__' of a type (line 174)
        getitem___374741 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 174, 29), shape_374740, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 174)
        subscript_call_result_374742 = invoke(stypy.reporting.localization.Localization(__file__, 174, 29), getitem___374741, int_374738)
        
        # Applying the binary operator '>=' (line 174)
        result_ge_374743 = python_operator(stypy.reporting.localization.Localization(__file__, 174, 24), '>=', j_374737, subscript_call_result_374742)
        
        # Applying the binary operator 'or' (line 174)
        result_or_keyword_374744 = python_operator(stypy.reporting.localization.Localization(__file__, 174, 15), 'or', result_lt_374736, result_ge_374743)
        
        # Testing the type of an if condition (line 174)
        if_condition_374745 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 174, 12), result_or_keyword_374744)
        # Assigning a type to the variable 'if_condition_374745' (line 174)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 174, 12), 'if_condition_374745', if_condition_374745)
        # SSA begins for if statement (line 174)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to IndexError(...): (line 175)
        # Processing the call arguments (line 175)
        str_374747 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 175, 33), 'str', 'Index out of bounds.')
        # Processing the call keyword arguments (line 175)
        kwargs_374748 = {}
        # Getting the type of 'IndexError' (line 175)
        IndexError_374746 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 22), 'IndexError', False)
        # Calling IndexError(args, kwargs) (line 175)
        IndexError_call_result_374749 = invoke(stypy.reporting.localization.Localization(__file__, 175, 22), IndexError_374746, *[str_374747], **kwargs_374748)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 175, 16), IndexError_call_result_374749, 'raise parameter', BaseException)
        # SSA join for if statement (line 174)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to get(...): (line 176)
        # Processing the call arguments (line 176)
        # Getting the type of 'self' (line 176)
        self_374752 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 28), 'self', False)
        
        # Obtaining an instance of the builtin type 'tuple' (line 176)
        tuple_374753 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 176, 35), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 176)
        # Adding element type (line 176)
        # Getting the type of 'i' (line 176)
        i_374754 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 35), 'i', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 176, 35), tuple_374753, i_374754)
        # Adding element type (line 176)
        # Getting the type of 'j' (line 176)
        j_374755 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 37), 'j', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 176, 35), tuple_374753, j_374755)
        
        # Getting the type of 'zero' (line 176)
        zero_374756 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 41), 'zero', False)
        # Processing the call keyword arguments (line 176)
        kwargs_374757 = {}
        # Getting the type of 'dict' (line 176)
        dict_374750 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 19), 'dict', False)
        # Obtaining the member 'get' of a type (line 176)
        get_374751 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 176, 19), dict_374750, 'get')
        # Calling get(args, kwargs) (line 176)
        get_call_result_374758 = invoke(stypy.reporting.localization.Localization(__file__, 176, 19), get_374751, *[self_374752, tuple_374753, zero_374756], **kwargs_374757)
        
        # Assigning a type to the variable 'stypy_return_type' (line 176)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 176, 12), 'stypy_return_type', get_call_result_374758)
        # SSA branch for the else part of an if statement (line 165)
        module_type_store.open_ssa_branch('else')
        
        
        # Evaluating a boolean operation
        
        # Evaluating a boolean operation
        # Getting the type of 'i_intlike' (line 177)
        i_intlike_374759 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 15), 'i_intlike')
        
        # Call to isinstance(...): (line 177)
        # Processing the call arguments (line 177)
        # Getting the type of 'i' (line 177)
        i_374761 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 39), 'i', False)
        # Getting the type of 'slice' (line 177)
        slice_374762 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 42), 'slice', False)
        # Processing the call keyword arguments (line 177)
        kwargs_374763 = {}
        # Getting the type of 'isinstance' (line 177)
        isinstance_374760 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 28), 'isinstance', False)
        # Calling isinstance(args, kwargs) (line 177)
        isinstance_call_result_374764 = invoke(stypy.reporting.localization.Localization(__file__, 177, 28), isinstance_374760, *[i_374761, slice_374762], **kwargs_374763)
        
        # Applying the binary operator 'or' (line 177)
        result_or_keyword_374765 = python_operator(stypy.reporting.localization.Localization(__file__, 177, 15), 'or', i_intlike_374759, isinstance_call_result_374764)
        
        
        # Evaluating a boolean operation
        # Getting the type of 'j_intlike' (line 178)
        j_intlike_374766 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 15), 'j_intlike')
        
        # Call to isinstance(...): (line 178)
        # Processing the call arguments (line 178)
        # Getting the type of 'j' (line 178)
        j_374768 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 39), 'j', False)
        # Getting the type of 'slice' (line 178)
        slice_374769 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 42), 'slice', False)
        # Processing the call keyword arguments (line 178)
        kwargs_374770 = {}
        # Getting the type of 'isinstance' (line 178)
        isinstance_374767 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 28), 'isinstance', False)
        # Calling isinstance(args, kwargs) (line 178)
        isinstance_call_result_374771 = invoke(stypy.reporting.localization.Localization(__file__, 178, 28), isinstance_374767, *[j_374768, slice_374769], **kwargs_374770)
        
        # Applying the binary operator 'or' (line 178)
        result_or_keyword_374772 = python_operator(stypy.reporting.localization.Localization(__file__, 178, 15), 'or', j_intlike_374766, isinstance_call_result_374771)
        
        # Applying the binary operator 'and' (line 177)
        result_and_keyword_374773 = python_operator(stypy.reporting.localization.Localization(__file__, 177, 14), 'and', result_or_keyword_374765, result_or_keyword_374772)
        
        # Testing the type of an if condition (line 177)
        if_condition_374774 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 177, 13), result_and_keyword_374773)
        # Assigning a type to the variable 'if_condition_374774' (line 177)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 177, 13), 'if_condition_374774', if_condition_374774)
        # SSA begins for if statement (line 177)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a IfExp to a Name (line 180):
        
        # Assigning a IfExp to a Name (line 180):
        
        # Getting the type of 'i_intlike' (line 180)
        i_intlike_374775 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 39), 'i_intlike')
        # Testing the type of an if expression (line 180)
        is_suitable_condition(stypy.reporting.localization.Localization(__file__, 180, 22), i_intlike_374775)
        # SSA begins for if expression (line 180)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if expression')
        
        # Call to slice(...): (line 180)
        # Processing the call arguments (line 180)
        # Getting the type of 'i' (line 180)
        i_374777 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 28), 'i', False)
        # Getting the type of 'i' (line 180)
        i_374778 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 31), 'i', False)
        int_374779 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 180, 33), 'int')
        # Applying the binary operator '+' (line 180)
        result_add_374780 = python_operator(stypy.reporting.localization.Localization(__file__, 180, 31), '+', i_374778, int_374779)
        
        # Processing the call keyword arguments (line 180)
        kwargs_374781 = {}
        # Getting the type of 'slice' (line 180)
        slice_374776 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 22), 'slice', False)
        # Calling slice(args, kwargs) (line 180)
        slice_call_result_374782 = invoke(stypy.reporting.localization.Localization(__file__, 180, 22), slice_374776, *[i_374777, result_add_374780], **kwargs_374781)
        
        # SSA branch for the else part of an if expression (line 180)
        module_type_store.open_ssa_branch('if expression else')
        # Getting the type of 'i' (line 180)
        i_374783 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 54), 'i')
        # SSA join for if expression (line 180)
        module_type_store = module_type_store.join_ssa_context()
        if_exp_374784 = union_type.UnionType.add(slice_call_result_374782, i_374783)
        
        # Assigning a type to the variable 'i_slice' (line 180)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 180, 12), 'i_slice', if_exp_374784)
        
        # Assigning a IfExp to a Name (line 181):
        
        # Assigning a IfExp to a Name (line 181):
        
        # Getting the type of 'j_intlike' (line 181)
        j_intlike_374785 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 39), 'j_intlike')
        # Testing the type of an if expression (line 181)
        is_suitable_condition(stypy.reporting.localization.Localization(__file__, 181, 22), j_intlike_374785)
        # SSA begins for if expression (line 181)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if expression')
        
        # Call to slice(...): (line 181)
        # Processing the call arguments (line 181)
        # Getting the type of 'j' (line 181)
        j_374787 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 28), 'j', False)
        # Getting the type of 'j' (line 181)
        j_374788 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 31), 'j', False)
        int_374789 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 181, 33), 'int')
        # Applying the binary operator '+' (line 181)
        result_add_374790 = python_operator(stypy.reporting.localization.Localization(__file__, 181, 31), '+', j_374788, int_374789)
        
        # Processing the call keyword arguments (line 181)
        kwargs_374791 = {}
        # Getting the type of 'slice' (line 181)
        slice_374786 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 22), 'slice', False)
        # Calling slice(args, kwargs) (line 181)
        slice_call_result_374792 = invoke(stypy.reporting.localization.Localization(__file__, 181, 22), slice_374786, *[j_374787, result_add_374790], **kwargs_374791)
        
        # SSA branch for the else part of an if expression (line 181)
        module_type_store.open_ssa_branch('if expression else')
        # Getting the type of 'j' (line 181)
        j_374793 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 54), 'j')
        # SSA join for if expression (line 181)
        module_type_store = module_type_store.join_ssa_context()
        if_exp_374794 = union_type.UnionType.add(slice_call_result_374792, j_374793)
        
        # Assigning a type to the variable 'j_slice' (line 181)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 181, 12), 'j_slice', if_exp_374794)
        
        # Assigning a Call to a Name (line 182):
        
        # Assigning a Call to a Name (line 182):
        
        # Call to indices(...): (line 182)
        # Processing the call arguments (line 182)
        
        # Obtaining the type of the subscript
        int_374797 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 182, 51), 'int')
        # Getting the type of 'self' (line 182)
        self_374798 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 182, 40), 'self', False)
        # Obtaining the member 'shape' of a type (line 182)
        shape_374799 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 182, 40), self_374798, 'shape')
        # Obtaining the member '__getitem__' of a type (line 182)
        getitem___374800 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 182, 40), shape_374799, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 182)
        subscript_call_result_374801 = invoke(stypy.reporting.localization.Localization(__file__, 182, 40), getitem___374800, int_374797)
        
        # Processing the call keyword arguments (line 182)
        kwargs_374802 = {}
        # Getting the type of 'i_slice' (line 182)
        i_slice_374795 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 182, 24), 'i_slice', False)
        # Obtaining the member 'indices' of a type (line 182)
        indices_374796 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 182, 24), i_slice_374795, 'indices')
        # Calling indices(args, kwargs) (line 182)
        indices_call_result_374803 = invoke(stypy.reporting.localization.Localization(__file__, 182, 24), indices_374796, *[subscript_call_result_374801], **kwargs_374802)
        
        # Assigning a type to the variable 'i_indices' (line 182)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 182, 12), 'i_indices', indices_call_result_374803)
        
        # Assigning a Call to a Name (line 183):
        
        # Assigning a Call to a Name (line 183):
        
        # Call to indices(...): (line 183)
        # Processing the call arguments (line 183)
        
        # Obtaining the type of the subscript
        int_374806 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 183, 51), 'int')
        # Getting the type of 'self' (line 183)
        self_374807 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 40), 'self', False)
        # Obtaining the member 'shape' of a type (line 183)
        shape_374808 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 183, 40), self_374807, 'shape')
        # Obtaining the member '__getitem__' of a type (line 183)
        getitem___374809 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 183, 40), shape_374808, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 183)
        subscript_call_result_374810 = invoke(stypy.reporting.localization.Localization(__file__, 183, 40), getitem___374809, int_374806)
        
        # Processing the call keyword arguments (line 183)
        kwargs_374811 = {}
        # Getting the type of 'j_slice' (line 183)
        j_slice_374804 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 24), 'j_slice', False)
        # Obtaining the member 'indices' of a type (line 183)
        indices_374805 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 183, 24), j_slice_374804, 'indices')
        # Calling indices(args, kwargs) (line 183)
        indices_call_result_374812 = invoke(stypy.reporting.localization.Localization(__file__, 183, 24), indices_374805, *[subscript_call_result_374810], **kwargs_374811)
        
        # Assigning a type to the variable 'j_indices' (line 183)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 183, 12), 'j_indices', indices_call_result_374812)
        
        # Assigning a Call to a Name (line 184):
        
        # Assigning a Call to a Name (line 184):
        
        # Call to xrange(...): (line 184)
        # Getting the type of 'i_indices' (line 184)
        i_indices_374814 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 28), 'i_indices', False)
        # Processing the call keyword arguments (line 184)
        kwargs_374815 = {}
        # Getting the type of 'xrange' (line 184)
        xrange_374813 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 20), 'xrange', False)
        # Calling xrange(args, kwargs) (line 184)
        xrange_call_result_374816 = invoke(stypy.reporting.localization.Localization(__file__, 184, 20), xrange_374813, *[i_indices_374814], **kwargs_374815)
        
        # Assigning a type to the variable 'i_seq' (line 184)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 184, 12), 'i_seq', xrange_call_result_374816)
        
        # Assigning a Call to a Name (line 185):
        
        # Assigning a Call to a Name (line 185):
        
        # Call to xrange(...): (line 185)
        # Getting the type of 'j_indices' (line 185)
        j_indices_374818 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 28), 'j_indices', False)
        # Processing the call keyword arguments (line 185)
        kwargs_374819 = {}
        # Getting the type of 'xrange' (line 185)
        xrange_374817 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 20), 'xrange', False)
        # Calling xrange(args, kwargs) (line 185)
        xrange_call_result_374820 = invoke(stypy.reporting.localization.Localization(__file__, 185, 20), xrange_374817, *[j_indices_374818], **kwargs_374819)
        
        # Assigning a type to the variable 'j_seq' (line 185)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 185, 12), 'j_seq', xrange_call_result_374820)
        
        # Assigning a Tuple to a Name (line 186):
        
        # Assigning a Tuple to a Name (line 186):
        
        # Obtaining an instance of the builtin type 'tuple' (line 186)
        tuple_374821 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 186, 24), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 186)
        # Adding element type (line 186)
        
        # Call to len(...): (line 186)
        # Processing the call arguments (line 186)
        # Getting the type of 'i_seq' (line 186)
        i_seq_374823 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 28), 'i_seq', False)
        # Processing the call keyword arguments (line 186)
        kwargs_374824 = {}
        # Getting the type of 'len' (line 186)
        len_374822 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 24), 'len', False)
        # Calling len(args, kwargs) (line 186)
        len_call_result_374825 = invoke(stypy.reporting.localization.Localization(__file__, 186, 24), len_374822, *[i_seq_374823], **kwargs_374824)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 186, 24), tuple_374821, len_call_result_374825)
        # Adding element type (line 186)
        
        # Call to len(...): (line 186)
        # Processing the call arguments (line 186)
        # Getting the type of 'j_seq' (line 186)
        j_seq_374827 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 40), 'j_seq', False)
        # Processing the call keyword arguments (line 186)
        kwargs_374828 = {}
        # Getting the type of 'len' (line 186)
        len_374826 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 36), 'len', False)
        # Calling len(args, kwargs) (line 186)
        len_call_result_374829 = invoke(stypy.reporting.localization.Localization(__file__, 186, 36), len_374826, *[j_seq_374827], **kwargs_374828)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 186, 24), tuple_374821, len_call_result_374829)
        
        # Assigning a type to the variable 'newshape' (line 186)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 186, 12), 'newshape', tuple_374821)
        
        # Assigning a Call to a Name (line 187):
        
        # Assigning a Call to a Name (line 187):
        
        # Call to _prod(...): (line 187)
        # Processing the call arguments (line 187)
        # Getting the type of 'newshape' (line 187)
        newshape_374831 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 28), 'newshape', False)
        # Processing the call keyword arguments (line 187)
        kwargs_374832 = {}
        # Getting the type of '_prod' (line 187)
        _prod_374830 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 22), '_prod', False)
        # Calling _prod(args, kwargs) (line 187)
        _prod_call_result_374833 = invoke(stypy.reporting.localization.Localization(__file__, 187, 22), _prod_374830, *[newshape_374831], **kwargs_374832)
        
        # Assigning a type to the variable 'newsize' (line 187)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 187, 12), 'newsize', _prod_call_result_374833)
        
        
        # Evaluating a boolean operation
        
        
        # Call to len(...): (line 189)
        # Processing the call arguments (line 189)
        # Getting the type of 'self' (line 189)
        self_374835 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 19), 'self', False)
        # Processing the call keyword arguments (line 189)
        kwargs_374836 = {}
        # Getting the type of 'len' (line 189)
        len_374834 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 15), 'len', False)
        # Calling len(args, kwargs) (line 189)
        len_call_result_374837 = invoke(stypy.reporting.localization.Localization(__file__, 189, 15), len_374834, *[self_374835], **kwargs_374836)
        
        int_374838 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 189, 27), 'int')
        # Getting the type of 'newsize' (line 189)
        newsize_374839 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 29), 'newsize')
        # Applying the binary operator '*' (line 189)
        result_mul_374840 = python_operator(stypy.reporting.localization.Localization(__file__, 189, 27), '*', int_374838, newsize_374839)
        
        # Applying the binary operator '<' (line 189)
        result_lt_374841 = python_operator(stypy.reporting.localization.Localization(__file__, 189, 15), '<', len_call_result_374837, result_mul_374840)
        
        
        # Getting the type of 'newsize' (line 189)
        newsize_374842 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 41), 'newsize')
        int_374843 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 189, 52), 'int')
        # Applying the binary operator '!=' (line 189)
        result_ne_374844 = python_operator(stypy.reporting.localization.Localization(__file__, 189, 41), '!=', newsize_374842, int_374843)
        
        # Applying the binary operator 'and' (line 189)
        result_and_keyword_374845 = python_operator(stypy.reporting.localization.Localization(__file__, 189, 15), 'and', result_lt_374841, result_ne_374844)
        
        # Testing the type of an if condition (line 189)
        if_condition_374846 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 189, 12), result_and_keyword_374845)
        # Assigning a type to the variable 'if_condition_374846' (line 189)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 189, 12), 'if_condition_374846', if_condition_374846)
        # SSA begins for if statement (line 189)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to _getitem_ranges(...): (line 196)
        # Processing the call arguments (line 196)
        # Getting the type of 'i_indices' (line 196)
        i_indices_374849 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 196, 44), 'i_indices', False)
        # Getting the type of 'j_indices' (line 196)
        j_indices_374850 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 196, 55), 'j_indices', False)
        # Getting the type of 'newshape' (line 196)
        newshape_374851 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 196, 66), 'newshape', False)
        # Processing the call keyword arguments (line 196)
        kwargs_374852 = {}
        # Getting the type of 'self' (line 196)
        self_374847 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 196, 23), 'self', False)
        # Obtaining the member '_getitem_ranges' of a type (line 196)
        _getitem_ranges_374848 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 196, 23), self_374847, '_getitem_ranges')
        # Calling _getitem_ranges(args, kwargs) (line 196)
        _getitem_ranges_call_result_374853 = invoke(stypy.reporting.localization.Localization(__file__, 196, 23), _getitem_ranges_374848, *[i_indices_374849, j_indices_374850, newshape_374851], **kwargs_374852)
        
        # Assigning a type to the variable 'stypy_return_type' (line 196)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 196, 16), 'stypy_return_type', _getitem_ranges_call_result_374853)
        # SSA join for if statement (line 189)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 177)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 165)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Tuple (line 198):
        
        # Assigning a Subscript to a Name (line 198):
        
        # Obtaining the type of the subscript
        int_374854 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 198, 8), 'int')
        
        # Call to _index_to_arrays(...): (line 198)
        # Processing the call arguments (line 198)
        # Getting the type of 'i' (line 198)
        i_374857 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 37), 'i', False)
        # Getting the type of 'j' (line 198)
        j_374858 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 40), 'j', False)
        # Processing the call keyword arguments (line 198)
        kwargs_374859 = {}
        # Getting the type of 'self' (line 198)
        self_374855 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 15), 'self', False)
        # Obtaining the member '_index_to_arrays' of a type (line 198)
        _index_to_arrays_374856 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 198, 15), self_374855, '_index_to_arrays')
        # Calling _index_to_arrays(args, kwargs) (line 198)
        _index_to_arrays_call_result_374860 = invoke(stypy.reporting.localization.Localization(__file__, 198, 15), _index_to_arrays_374856, *[i_374857, j_374858], **kwargs_374859)
        
        # Obtaining the member '__getitem__' of a type (line 198)
        getitem___374861 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 198, 8), _index_to_arrays_call_result_374860, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 198)
        subscript_call_result_374862 = invoke(stypy.reporting.localization.Localization(__file__, 198, 8), getitem___374861, int_374854)
        
        # Assigning a type to the variable 'tuple_var_assignment_374346' (line 198)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 198, 8), 'tuple_var_assignment_374346', subscript_call_result_374862)
        
        # Assigning a Subscript to a Name (line 198):
        
        # Obtaining the type of the subscript
        int_374863 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 198, 8), 'int')
        
        # Call to _index_to_arrays(...): (line 198)
        # Processing the call arguments (line 198)
        # Getting the type of 'i' (line 198)
        i_374866 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 37), 'i', False)
        # Getting the type of 'j' (line 198)
        j_374867 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 40), 'j', False)
        # Processing the call keyword arguments (line 198)
        kwargs_374868 = {}
        # Getting the type of 'self' (line 198)
        self_374864 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 15), 'self', False)
        # Obtaining the member '_index_to_arrays' of a type (line 198)
        _index_to_arrays_374865 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 198, 15), self_374864, '_index_to_arrays')
        # Calling _index_to_arrays(args, kwargs) (line 198)
        _index_to_arrays_call_result_374869 = invoke(stypy.reporting.localization.Localization(__file__, 198, 15), _index_to_arrays_374865, *[i_374866, j_374867], **kwargs_374868)
        
        # Obtaining the member '__getitem__' of a type (line 198)
        getitem___374870 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 198, 8), _index_to_arrays_call_result_374869, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 198)
        subscript_call_result_374871 = invoke(stypy.reporting.localization.Localization(__file__, 198, 8), getitem___374870, int_374863)
        
        # Assigning a type to the variable 'tuple_var_assignment_374347' (line 198)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 198, 8), 'tuple_var_assignment_374347', subscript_call_result_374871)
        
        # Assigning a Name to a Name (line 198):
        # Getting the type of 'tuple_var_assignment_374346' (line 198)
        tuple_var_assignment_374346_374872 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 8), 'tuple_var_assignment_374346')
        # Assigning a type to the variable 'i' (line 198)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 198, 8), 'i', tuple_var_assignment_374346_374872)
        
        # Assigning a Name to a Name (line 198):
        # Getting the type of 'tuple_var_assignment_374347' (line 198)
        tuple_var_assignment_374347_374873 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 8), 'tuple_var_assignment_374347')
        # Assigning a type to the variable 'j' (line 198)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 198, 11), 'j', tuple_var_assignment_374347_374873)
        
        
        # Getting the type of 'i' (line 200)
        i_374874 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 11), 'i')
        # Obtaining the member 'size' of a type (line 200)
        size_374875 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 200, 11), i_374874, 'size')
        int_374876 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 200, 21), 'int')
        # Applying the binary operator '==' (line 200)
        result_eq_374877 = python_operator(stypy.reporting.localization.Localization(__file__, 200, 11), '==', size_374875, int_374876)
        
        # Testing the type of an if condition (line 200)
        if_condition_374878 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 200, 8), result_eq_374877)
        # Assigning a type to the variable 'if_condition_374878' (line 200)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 200, 8), 'if_condition_374878', if_condition_374878)
        # SSA begins for if statement (line 200)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to dok_matrix(...): (line 201)
        # Processing the call arguments (line 201)
        # Getting the type of 'i' (line 201)
        i_374880 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 30), 'i', False)
        # Obtaining the member 'shape' of a type (line 201)
        shape_374881 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 201, 30), i_374880, 'shape')
        # Processing the call keyword arguments (line 201)
        # Getting the type of 'self' (line 201)
        self_374882 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 45), 'self', False)
        # Obtaining the member 'dtype' of a type (line 201)
        dtype_374883 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 201, 45), self_374882, 'dtype')
        keyword_374884 = dtype_374883
        kwargs_374885 = {'dtype': keyword_374884}
        # Getting the type of 'dok_matrix' (line 201)
        dok_matrix_374879 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 19), 'dok_matrix', False)
        # Calling dok_matrix(args, kwargs) (line 201)
        dok_matrix_call_result_374886 = invoke(stypy.reporting.localization.Localization(__file__, 201, 19), dok_matrix_374879, *[shape_374881], **kwargs_374885)
        
        # Assigning a type to the variable 'stypy_return_type' (line 201)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 201, 12), 'stypy_return_type', dok_matrix_call_result_374886)
        # SSA join for if statement (line 200)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Name (line 203):
        
        # Assigning a Call to a Name (line 203):
        
        # Call to min(...): (line 203)
        # Processing the call keyword arguments (line 203)
        kwargs_374889 = {}
        # Getting the type of 'i' (line 203)
        i_374887 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 16), 'i', False)
        # Obtaining the member 'min' of a type (line 203)
        min_374888 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 203, 16), i_374887, 'min')
        # Calling min(args, kwargs) (line 203)
        min_call_result_374890 = invoke(stypy.reporting.localization.Localization(__file__, 203, 16), min_374888, *[], **kwargs_374889)
        
        # Assigning a type to the variable 'min_i' (line 203)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 203, 8), 'min_i', min_call_result_374890)
        
        
        # Evaluating a boolean operation
        
        # Getting the type of 'min_i' (line 204)
        min_i_374891 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 204, 11), 'min_i')
        
        
        # Obtaining the type of the subscript
        int_374892 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 204, 31), 'int')
        # Getting the type of 'self' (line 204)
        self_374893 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 204, 20), 'self')
        # Obtaining the member 'shape' of a type (line 204)
        shape_374894 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 204, 20), self_374893, 'shape')
        # Obtaining the member '__getitem__' of a type (line 204)
        getitem___374895 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 204, 20), shape_374894, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 204)
        subscript_call_result_374896 = invoke(stypy.reporting.localization.Localization(__file__, 204, 20), getitem___374895, int_374892)
        
        # Applying the 'usub' unary operator (line 204)
        result___neg___374897 = python_operator(stypy.reporting.localization.Localization(__file__, 204, 19), 'usub', subscript_call_result_374896)
        
        # Applying the binary operator '<' (line 204)
        result_lt_374898 = python_operator(stypy.reporting.localization.Localization(__file__, 204, 11), '<', min_i_374891, result___neg___374897)
        
        
        
        # Call to max(...): (line 204)
        # Processing the call keyword arguments (line 204)
        kwargs_374901 = {}
        # Getting the type of 'i' (line 204)
        i_374899 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 204, 37), 'i', False)
        # Obtaining the member 'max' of a type (line 204)
        max_374900 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 204, 37), i_374899, 'max')
        # Calling max(args, kwargs) (line 204)
        max_call_result_374902 = invoke(stypy.reporting.localization.Localization(__file__, 204, 37), max_374900, *[], **kwargs_374901)
        
        
        # Obtaining the type of the subscript
        int_374903 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 204, 59), 'int')
        # Getting the type of 'self' (line 204)
        self_374904 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 204, 48), 'self')
        # Obtaining the member 'shape' of a type (line 204)
        shape_374905 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 204, 48), self_374904, 'shape')
        # Obtaining the member '__getitem__' of a type (line 204)
        getitem___374906 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 204, 48), shape_374905, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 204)
        subscript_call_result_374907 = invoke(stypy.reporting.localization.Localization(__file__, 204, 48), getitem___374906, int_374903)
        
        # Applying the binary operator '>=' (line 204)
        result_ge_374908 = python_operator(stypy.reporting.localization.Localization(__file__, 204, 37), '>=', max_call_result_374902, subscript_call_result_374907)
        
        # Applying the binary operator 'or' (line 204)
        result_or_keyword_374909 = python_operator(stypy.reporting.localization.Localization(__file__, 204, 11), 'or', result_lt_374898, result_ge_374908)
        
        # Testing the type of an if condition (line 204)
        if_condition_374910 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 204, 8), result_or_keyword_374909)
        # Assigning a type to the variable 'if_condition_374910' (line 204)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 204, 8), 'if_condition_374910', if_condition_374910)
        # SSA begins for if statement (line 204)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to IndexError(...): (line 205)
        # Processing the call arguments (line 205)
        str_374912 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 205, 29), 'str', 'Index (%d) out of range -%d to %d.')
        
        # Obtaining an instance of the builtin type 'tuple' (line 206)
        tuple_374913 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 206, 30), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 206)
        # Adding element type (line 206)
        
        # Call to min(...): (line 206)
        # Processing the call keyword arguments (line 206)
        kwargs_374916 = {}
        # Getting the type of 'i' (line 206)
        i_374914 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 206, 30), 'i', False)
        # Obtaining the member 'min' of a type (line 206)
        min_374915 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 206, 30), i_374914, 'min')
        # Calling min(args, kwargs) (line 206)
        min_call_result_374917 = invoke(stypy.reporting.localization.Localization(__file__, 206, 30), min_374915, *[], **kwargs_374916)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 206, 30), tuple_374913, min_call_result_374917)
        # Adding element type (line 206)
        
        # Obtaining the type of the subscript
        int_374918 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 206, 50), 'int')
        # Getting the type of 'self' (line 206)
        self_374919 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 206, 39), 'self', False)
        # Obtaining the member 'shape' of a type (line 206)
        shape_374920 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 206, 39), self_374919, 'shape')
        # Obtaining the member '__getitem__' of a type (line 206)
        getitem___374921 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 206, 39), shape_374920, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 206)
        subscript_call_result_374922 = invoke(stypy.reporting.localization.Localization(__file__, 206, 39), getitem___374921, int_374918)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 206, 30), tuple_374913, subscript_call_result_374922)
        # Adding element type (line 206)
        
        # Obtaining the type of the subscript
        int_374923 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 206, 65), 'int')
        # Getting the type of 'self' (line 206)
        self_374924 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 206, 54), 'self', False)
        # Obtaining the member 'shape' of a type (line 206)
        shape_374925 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 206, 54), self_374924, 'shape')
        # Obtaining the member '__getitem__' of a type (line 206)
        getitem___374926 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 206, 54), shape_374925, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 206)
        subscript_call_result_374927 = invoke(stypy.reporting.localization.Localization(__file__, 206, 54), getitem___374926, int_374923)
        
        int_374928 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 206, 68), 'int')
        # Applying the binary operator '-' (line 206)
        result_sub_374929 = python_operator(stypy.reporting.localization.Localization(__file__, 206, 54), '-', subscript_call_result_374927, int_374928)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 206, 30), tuple_374913, result_sub_374929)
        
        # Applying the binary operator '%' (line 205)
        result_mod_374930 = python_operator(stypy.reporting.localization.Localization(__file__, 205, 29), '%', str_374912, tuple_374913)
        
        # Processing the call keyword arguments (line 205)
        kwargs_374931 = {}
        # Getting the type of 'IndexError' (line 205)
        IndexError_374911 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 18), 'IndexError', False)
        # Calling IndexError(args, kwargs) (line 205)
        IndexError_call_result_374932 = invoke(stypy.reporting.localization.Localization(__file__, 205, 18), IndexError_374911, *[result_mod_374930], **kwargs_374931)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 205, 12), IndexError_call_result_374932, 'raise parameter', BaseException)
        # SSA join for if statement (line 204)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Getting the type of 'min_i' (line 207)
        min_i_374933 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 207, 11), 'min_i')
        int_374934 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 207, 19), 'int')
        # Applying the binary operator '<' (line 207)
        result_lt_374935 = python_operator(stypy.reporting.localization.Localization(__file__, 207, 11), '<', min_i_374933, int_374934)
        
        # Testing the type of an if condition (line 207)
        if_condition_374936 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 207, 8), result_lt_374935)
        # Assigning a type to the variable 'if_condition_374936' (line 207)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 207, 8), 'if_condition_374936', if_condition_374936)
        # SSA begins for if statement (line 207)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 208):
        
        # Assigning a Call to a Name (line 208):
        
        # Call to copy(...): (line 208)
        # Processing the call keyword arguments (line 208)
        kwargs_374939 = {}
        # Getting the type of 'i' (line 208)
        i_374937 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 208, 16), 'i', False)
        # Obtaining the member 'copy' of a type (line 208)
        copy_374938 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 208, 16), i_374937, 'copy')
        # Calling copy(args, kwargs) (line 208)
        copy_call_result_374940 = invoke(stypy.reporting.localization.Localization(__file__, 208, 16), copy_374938, *[], **kwargs_374939)
        
        # Assigning a type to the variable 'i' (line 208)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 208, 12), 'i', copy_call_result_374940)
        
        # Getting the type of 'i' (line 209)
        i_374941 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 12), 'i')
        
        # Obtaining the type of the subscript
        
        # Getting the type of 'i' (line 209)
        i_374942 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 14), 'i')
        int_374943 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 209, 18), 'int')
        # Applying the binary operator '<' (line 209)
        result_lt_374944 = python_operator(stypy.reporting.localization.Localization(__file__, 209, 14), '<', i_374942, int_374943)
        
        # Getting the type of 'i' (line 209)
        i_374945 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 12), 'i')
        # Obtaining the member '__getitem__' of a type (line 209)
        getitem___374946 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 209, 12), i_374945, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 209)
        subscript_call_result_374947 = invoke(stypy.reporting.localization.Localization(__file__, 209, 12), getitem___374946, result_lt_374944)
        
        
        # Obtaining the type of the subscript
        int_374948 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 209, 35), 'int')
        # Getting the type of 'self' (line 209)
        self_374949 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 24), 'self')
        # Obtaining the member 'shape' of a type (line 209)
        shape_374950 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 209, 24), self_374949, 'shape')
        # Obtaining the member '__getitem__' of a type (line 209)
        getitem___374951 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 209, 24), shape_374950, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 209)
        subscript_call_result_374952 = invoke(stypy.reporting.localization.Localization(__file__, 209, 24), getitem___374951, int_374948)
        
        # Applying the binary operator '+=' (line 209)
        result_iadd_374953 = python_operator(stypy.reporting.localization.Localization(__file__, 209, 12), '+=', subscript_call_result_374947, subscript_call_result_374952)
        # Getting the type of 'i' (line 209)
        i_374954 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 12), 'i')
        
        # Getting the type of 'i' (line 209)
        i_374955 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 14), 'i')
        int_374956 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 209, 18), 'int')
        # Applying the binary operator '<' (line 209)
        result_lt_374957 = python_operator(stypy.reporting.localization.Localization(__file__, 209, 14), '<', i_374955, int_374956)
        
        # Storing an element on a container (line 209)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 209, 12), i_374954, (result_lt_374957, result_iadd_374953))
        
        # SSA join for if statement (line 207)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Name (line 211):
        
        # Assigning a Call to a Name (line 211):
        
        # Call to min(...): (line 211)
        # Processing the call keyword arguments (line 211)
        kwargs_374960 = {}
        # Getting the type of 'j' (line 211)
        j_374958 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 211, 16), 'j', False)
        # Obtaining the member 'min' of a type (line 211)
        min_374959 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 211, 16), j_374958, 'min')
        # Calling min(args, kwargs) (line 211)
        min_call_result_374961 = invoke(stypy.reporting.localization.Localization(__file__, 211, 16), min_374959, *[], **kwargs_374960)
        
        # Assigning a type to the variable 'min_j' (line 211)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 211, 8), 'min_j', min_call_result_374961)
        
        
        # Evaluating a boolean operation
        
        # Getting the type of 'min_j' (line 212)
        min_j_374962 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 11), 'min_j')
        
        
        # Obtaining the type of the subscript
        int_374963 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 212, 31), 'int')
        # Getting the type of 'self' (line 212)
        self_374964 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 20), 'self')
        # Obtaining the member 'shape' of a type (line 212)
        shape_374965 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 212, 20), self_374964, 'shape')
        # Obtaining the member '__getitem__' of a type (line 212)
        getitem___374966 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 212, 20), shape_374965, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 212)
        subscript_call_result_374967 = invoke(stypy.reporting.localization.Localization(__file__, 212, 20), getitem___374966, int_374963)
        
        # Applying the 'usub' unary operator (line 212)
        result___neg___374968 = python_operator(stypy.reporting.localization.Localization(__file__, 212, 19), 'usub', subscript_call_result_374967)
        
        # Applying the binary operator '<' (line 212)
        result_lt_374969 = python_operator(stypy.reporting.localization.Localization(__file__, 212, 11), '<', min_j_374962, result___neg___374968)
        
        
        
        # Call to max(...): (line 212)
        # Processing the call keyword arguments (line 212)
        kwargs_374972 = {}
        # Getting the type of 'j' (line 212)
        j_374970 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 37), 'j', False)
        # Obtaining the member 'max' of a type (line 212)
        max_374971 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 212, 37), j_374970, 'max')
        # Calling max(args, kwargs) (line 212)
        max_call_result_374973 = invoke(stypy.reporting.localization.Localization(__file__, 212, 37), max_374971, *[], **kwargs_374972)
        
        
        # Obtaining the type of the subscript
        int_374974 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 212, 59), 'int')
        # Getting the type of 'self' (line 212)
        self_374975 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 48), 'self')
        # Obtaining the member 'shape' of a type (line 212)
        shape_374976 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 212, 48), self_374975, 'shape')
        # Obtaining the member '__getitem__' of a type (line 212)
        getitem___374977 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 212, 48), shape_374976, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 212)
        subscript_call_result_374978 = invoke(stypy.reporting.localization.Localization(__file__, 212, 48), getitem___374977, int_374974)
        
        # Applying the binary operator '>=' (line 212)
        result_ge_374979 = python_operator(stypy.reporting.localization.Localization(__file__, 212, 37), '>=', max_call_result_374973, subscript_call_result_374978)
        
        # Applying the binary operator 'or' (line 212)
        result_or_keyword_374980 = python_operator(stypy.reporting.localization.Localization(__file__, 212, 11), 'or', result_lt_374969, result_ge_374979)
        
        # Testing the type of an if condition (line 212)
        if_condition_374981 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 212, 8), result_or_keyword_374980)
        # Assigning a type to the variable 'if_condition_374981' (line 212)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 212, 8), 'if_condition_374981', if_condition_374981)
        # SSA begins for if statement (line 212)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to IndexError(...): (line 213)
        # Processing the call arguments (line 213)
        str_374983 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 213, 29), 'str', 'Index (%d) out of range -%d to %d.')
        
        # Obtaining an instance of the builtin type 'tuple' (line 214)
        tuple_374984 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 214, 30), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 214)
        # Adding element type (line 214)
        
        # Call to min(...): (line 214)
        # Processing the call keyword arguments (line 214)
        kwargs_374987 = {}
        # Getting the type of 'j' (line 214)
        j_374985 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 30), 'j', False)
        # Obtaining the member 'min' of a type (line 214)
        min_374986 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 214, 30), j_374985, 'min')
        # Calling min(args, kwargs) (line 214)
        min_call_result_374988 = invoke(stypy.reporting.localization.Localization(__file__, 214, 30), min_374986, *[], **kwargs_374987)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 214, 30), tuple_374984, min_call_result_374988)
        # Adding element type (line 214)
        
        # Obtaining the type of the subscript
        int_374989 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 214, 50), 'int')
        # Getting the type of 'self' (line 214)
        self_374990 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 39), 'self', False)
        # Obtaining the member 'shape' of a type (line 214)
        shape_374991 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 214, 39), self_374990, 'shape')
        # Obtaining the member '__getitem__' of a type (line 214)
        getitem___374992 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 214, 39), shape_374991, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 214)
        subscript_call_result_374993 = invoke(stypy.reporting.localization.Localization(__file__, 214, 39), getitem___374992, int_374989)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 214, 30), tuple_374984, subscript_call_result_374993)
        # Adding element type (line 214)
        
        # Obtaining the type of the subscript
        int_374994 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 214, 65), 'int')
        # Getting the type of 'self' (line 214)
        self_374995 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 54), 'self', False)
        # Obtaining the member 'shape' of a type (line 214)
        shape_374996 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 214, 54), self_374995, 'shape')
        # Obtaining the member '__getitem__' of a type (line 214)
        getitem___374997 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 214, 54), shape_374996, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 214)
        subscript_call_result_374998 = invoke(stypy.reporting.localization.Localization(__file__, 214, 54), getitem___374997, int_374994)
        
        int_374999 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 214, 68), 'int')
        # Applying the binary operator '-' (line 214)
        result_sub_375000 = python_operator(stypy.reporting.localization.Localization(__file__, 214, 54), '-', subscript_call_result_374998, int_374999)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 214, 30), tuple_374984, result_sub_375000)
        
        # Applying the binary operator '%' (line 213)
        result_mod_375001 = python_operator(stypy.reporting.localization.Localization(__file__, 213, 29), '%', str_374983, tuple_374984)
        
        # Processing the call keyword arguments (line 213)
        kwargs_375002 = {}
        # Getting the type of 'IndexError' (line 213)
        IndexError_374982 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 18), 'IndexError', False)
        # Calling IndexError(args, kwargs) (line 213)
        IndexError_call_result_375003 = invoke(stypy.reporting.localization.Localization(__file__, 213, 18), IndexError_374982, *[result_mod_375001], **kwargs_375002)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 213, 12), IndexError_call_result_375003, 'raise parameter', BaseException)
        # SSA join for if statement (line 212)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Getting the type of 'min_j' (line 215)
        min_j_375004 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 215, 11), 'min_j')
        int_375005 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 215, 19), 'int')
        # Applying the binary operator '<' (line 215)
        result_lt_375006 = python_operator(stypy.reporting.localization.Localization(__file__, 215, 11), '<', min_j_375004, int_375005)
        
        # Testing the type of an if condition (line 215)
        if_condition_375007 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 215, 8), result_lt_375006)
        # Assigning a type to the variable 'if_condition_375007' (line 215)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 215, 8), 'if_condition_375007', if_condition_375007)
        # SSA begins for if statement (line 215)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 216):
        
        # Assigning a Call to a Name (line 216):
        
        # Call to copy(...): (line 216)
        # Processing the call keyword arguments (line 216)
        kwargs_375010 = {}
        # Getting the type of 'j' (line 216)
        j_375008 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 216, 16), 'j', False)
        # Obtaining the member 'copy' of a type (line 216)
        copy_375009 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 216, 16), j_375008, 'copy')
        # Calling copy(args, kwargs) (line 216)
        copy_call_result_375011 = invoke(stypy.reporting.localization.Localization(__file__, 216, 16), copy_375009, *[], **kwargs_375010)
        
        # Assigning a type to the variable 'j' (line 216)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 216, 12), 'j', copy_call_result_375011)
        
        # Getting the type of 'j' (line 217)
        j_375012 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 12), 'j')
        
        # Obtaining the type of the subscript
        
        # Getting the type of 'j' (line 217)
        j_375013 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 14), 'j')
        int_375014 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 217, 18), 'int')
        # Applying the binary operator '<' (line 217)
        result_lt_375015 = python_operator(stypy.reporting.localization.Localization(__file__, 217, 14), '<', j_375013, int_375014)
        
        # Getting the type of 'j' (line 217)
        j_375016 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 12), 'j')
        # Obtaining the member '__getitem__' of a type (line 217)
        getitem___375017 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 217, 12), j_375016, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 217)
        subscript_call_result_375018 = invoke(stypy.reporting.localization.Localization(__file__, 217, 12), getitem___375017, result_lt_375015)
        
        
        # Obtaining the type of the subscript
        int_375019 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 217, 35), 'int')
        # Getting the type of 'self' (line 217)
        self_375020 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 24), 'self')
        # Obtaining the member 'shape' of a type (line 217)
        shape_375021 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 217, 24), self_375020, 'shape')
        # Obtaining the member '__getitem__' of a type (line 217)
        getitem___375022 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 217, 24), shape_375021, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 217)
        subscript_call_result_375023 = invoke(stypy.reporting.localization.Localization(__file__, 217, 24), getitem___375022, int_375019)
        
        # Applying the binary operator '+=' (line 217)
        result_iadd_375024 = python_operator(stypy.reporting.localization.Localization(__file__, 217, 12), '+=', subscript_call_result_375018, subscript_call_result_375023)
        # Getting the type of 'j' (line 217)
        j_375025 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 12), 'j')
        
        # Getting the type of 'j' (line 217)
        j_375026 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 14), 'j')
        int_375027 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 217, 18), 'int')
        # Applying the binary operator '<' (line 217)
        result_lt_375028 = python_operator(stypy.reporting.localization.Localization(__file__, 217, 14), '<', j_375026, int_375027)
        
        # Storing an element on a container (line 217)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 217, 12), j_375025, (result_lt_375028, result_iadd_375024))
        
        # SSA join for if statement (line 215)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Name (line 219):
        
        # Assigning a Call to a Name (line 219):
        
        # Call to dok_matrix(...): (line 219)
        # Processing the call arguments (line 219)
        # Getting the type of 'i' (line 219)
        i_375030 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 219, 28), 'i', False)
        # Obtaining the member 'shape' of a type (line 219)
        shape_375031 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 219, 28), i_375030, 'shape')
        # Processing the call keyword arguments (line 219)
        # Getting the type of 'self' (line 219)
        self_375032 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 219, 43), 'self', False)
        # Obtaining the member 'dtype' of a type (line 219)
        dtype_375033 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 219, 43), self_375032, 'dtype')
        keyword_375034 = dtype_375033
        kwargs_375035 = {'dtype': keyword_375034}
        # Getting the type of 'dok_matrix' (line 219)
        dok_matrix_375029 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 219, 17), 'dok_matrix', False)
        # Calling dok_matrix(args, kwargs) (line 219)
        dok_matrix_call_result_375036 = invoke(stypy.reporting.localization.Localization(__file__, 219, 17), dok_matrix_375029, *[shape_375031], **kwargs_375035)
        
        # Assigning a type to the variable 'newdok' (line 219)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 219, 8), 'newdok', dok_matrix_call_result_375036)
        
        
        # Call to product(...): (line 221)
        # Processing the call arguments (line 221)
        
        # Call to xrange(...): (line 221)
        # Processing the call arguments (line 221)
        
        # Obtaining the type of the subscript
        int_375040 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 221, 52), 'int')
        # Getting the type of 'i' (line 221)
        i_375041 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 44), 'i', False)
        # Obtaining the member 'shape' of a type (line 221)
        shape_375042 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 221, 44), i_375041, 'shape')
        # Obtaining the member '__getitem__' of a type (line 221)
        getitem___375043 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 221, 44), shape_375042, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 221)
        subscript_call_result_375044 = invoke(stypy.reporting.localization.Localization(__file__, 221, 44), getitem___375043, int_375040)
        
        # Processing the call keyword arguments (line 221)
        kwargs_375045 = {}
        # Getting the type of 'xrange' (line 221)
        xrange_375039 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 37), 'xrange', False)
        # Calling xrange(args, kwargs) (line 221)
        xrange_call_result_375046 = invoke(stypy.reporting.localization.Localization(__file__, 221, 37), xrange_375039, *[subscript_call_result_375044], **kwargs_375045)
        
        
        # Call to xrange(...): (line 221)
        # Processing the call arguments (line 221)
        
        # Obtaining the type of the subscript
        int_375048 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 221, 72), 'int')
        # Getting the type of 'i' (line 221)
        i_375049 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 64), 'i', False)
        # Obtaining the member 'shape' of a type (line 221)
        shape_375050 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 221, 64), i_375049, 'shape')
        # Obtaining the member '__getitem__' of a type (line 221)
        getitem___375051 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 221, 64), shape_375050, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 221)
        subscript_call_result_375052 = invoke(stypy.reporting.localization.Localization(__file__, 221, 64), getitem___375051, int_375048)
        
        # Processing the call keyword arguments (line 221)
        kwargs_375053 = {}
        # Getting the type of 'xrange' (line 221)
        xrange_375047 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 57), 'xrange', False)
        # Calling xrange(args, kwargs) (line 221)
        xrange_call_result_375054 = invoke(stypy.reporting.localization.Localization(__file__, 221, 57), xrange_375047, *[subscript_call_result_375052], **kwargs_375053)
        
        # Processing the call keyword arguments (line 221)
        kwargs_375055 = {}
        # Getting the type of 'itertools' (line 221)
        itertools_375037 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 19), 'itertools', False)
        # Obtaining the member 'product' of a type (line 221)
        product_375038 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 221, 19), itertools_375037, 'product')
        # Calling product(args, kwargs) (line 221)
        product_call_result_375056 = invoke(stypy.reporting.localization.Localization(__file__, 221, 19), product_375038, *[xrange_call_result_375046, xrange_call_result_375054], **kwargs_375055)
        
        # Testing the type of a for loop iterable (line 221)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 221, 8), product_call_result_375056)
        # Getting the type of the for loop variable (line 221)
        for_loop_var_375057 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 221, 8), product_call_result_375056)
        # Assigning a type to the variable 'key' (line 221)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 221, 8), 'key', for_loop_var_375057)
        # SSA begins for a for statement (line 221)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Call to a Name (line 222):
        
        # Assigning a Call to a Name (line 222):
        
        # Call to get(...): (line 222)
        # Processing the call arguments (line 222)
        # Getting the type of 'self' (line 222)
        self_375060 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 222, 25), 'self', False)
        
        # Obtaining an instance of the builtin type 'tuple' (line 222)
        tuple_375061 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 222, 32), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 222)
        # Adding element type (line 222)
        
        # Obtaining the type of the subscript
        # Getting the type of 'key' (line 222)
        key_375062 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 222, 34), 'key', False)
        # Getting the type of 'i' (line 222)
        i_375063 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 222, 32), 'i', False)
        # Obtaining the member '__getitem__' of a type (line 222)
        getitem___375064 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 222, 32), i_375063, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 222)
        subscript_call_result_375065 = invoke(stypy.reporting.localization.Localization(__file__, 222, 32), getitem___375064, key_375062)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 222, 32), tuple_375061, subscript_call_result_375065)
        # Adding element type (line 222)
        
        # Obtaining the type of the subscript
        # Getting the type of 'key' (line 222)
        key_375066 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 222, 42), 'key', False)
        # Getting the type of 'j' (line 222)
        j_375067 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 222, 40), 'j', False)
        # Obtaining the member '__getitem__' of a type (line 222)
        getitem___375068 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 222, 40), j_375067, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 222)
        subscript_call_result_375069 = invoke(stypy.reporting.localization.Localization(__file__, 222, 40), getitem___375068, key_375066)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 222, 32), tuple_375061, subscript_call_result_375069)
        
        # Getting the type of 'zero' (line 222)
        zero_375070 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 222, 49), 'zero', False)
        # Processing the call keyword arguments (line 222)
        kwargs_375071 = {}
        # Getting the type of 'dict' (line 222)
        dict_375058 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 222, 16), 'dict', False)
        # Obtaining the member 'get' of a type (line 222)
        get_375059 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 222, 16), dict_375058, 'get')
        # Calling get(args, kwargs) (line 222)
        get_call_result_375072 = invoke(stypy.reporting.localization.Localization(__file__, 222, 16), get_375059, *[self_375060, tuple_375061, zero_375070], **kwargs_375071)
        
        # Assigning a type to the variable 'v' (line 222)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 222, 12), 'v', get_call_result_375072)
        
        # Getting the type of 'v' (line 223)
        v_375073 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 15), 'v')
        # Testing the type of an if condition (line 223)
        if_condition_375074 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 223, 12), v_375073)
        # Assigning a type to the variable 'if_condition_375074' (line 223)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 223, 12), 'if_condition_375074', if_condition_375074)
        # SSA begins for if statement (line 223)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to __setitem__(...): (line 224)
        # Processing the call arguments (line 224)
        # Getting the type of 'newdok' (line 224)
        newdok_375077 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 224, 33), 'newdok', False)
        # Getting the type of 'key' (line 224)
        key_375078 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 224, 41), 'key', False)
        # Getting the type of 'v' (line 224)
        v_375079 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 224, 46), 'v', False)
        # Processing the call keyword arguments (line 224)
        kwargs_375080 = {}
        # Getting the type of 'dict' (line 224)
        dict_375075 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 224, 16), 'dict', False)
        # Obtaining the member '__setitem__' of a type (line 224)
        setitem___375076 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 224, 16), dict_375075, '__setitem__')
        # Calling __setitem__(args, kwargs) (line 224)
        setitem___call_result_375081 = invoke(stypy.reporting.localization.Localization(__file__, 224, 16), setitem___375076, *[newdok_375077, key_375078, v_375079], **kwargs_375080)
        
        # SSA join for if statement (line 223)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        # Getting the type of 'newdok' (line 226)
        newdok_375082 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 15), 'newdok')
        # Assigning a type to the variable 'stypy_return_type' (line 226)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 226, 8), 'stypy_return_type', newdok_375082)
        
        # ################# End of '__getitem__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__getitem__' in the type store
        # Getting the type of 'stypy_return_type' (line 154)
        stypy_return_type_375083 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_375083)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__getitem__'
        return stypy_return_type_375083


    @norecursion
    def _getitem_ranges(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_getitem_ranges'
        module_type_store = module_type_store.open_function_context('_getitem_ranges', 228, 4, False)
        # Assigning a type to the variable 'self' (line 229)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 229, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        dok_matrix._getitem_ranges.__dict__.__setitem__('stypy_localization', localization)
        dok_matrix._getitem_ranges.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        dok_matrix._getitem_ranges.__dict__.__setitem__('stypy_type_store', module_type_store)
        dok_matrix._getitem_ranges.__dict__.__setitem__('stypy_function_name', 'dok_matrix._getitem_ranges')
        dok_matrix._getitem_ranges.__dict__.__setitem__('stypy_param_names_list', ['i_indices', 'j_indices', 'shape'])
        dok_matrix._getitem_ranges.__dict__.__setitem__('stypy_varargs_param_name', None)
        dok_matrix._getitem_ranges.__dict__.__setitem__('stypy_kwargs_param_name', None)
        dok_matrix._getitem_ranges.__dict__.__setitem__('stypy_call_defaults', defaults)
        dok_matrix._getitem_ranges.__dict__.__setitem__('stypy_call_varargs', varargs)
        dok_matrix._getitem_ranges.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        dok_matrix._getitem_ranges.__dict__.__setitem__('stypy_declared_arg_number', 4)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'dok_matrix._getitem_ranges', ['i_indices', 'j_indices', 'shape'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_getitem_ranges', localization, ['i_indices', 'j_indices', 'shape'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_getitem_ranges(...)' code ##################

        
        # Assigning a Call to a Tuple (line 230):
        
        # Assigning a Subscript to a Name (line 230):
        
        # Obtaining the type of the subscript
        int_375084 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 230, 8), 'int')
        
        # Call to map(...): (line 230)
        # Processing the call arguments (line 230)
        # Getting the type of 'int' (line 230)
        int_375086 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 40), 'int', False)
        # Getting the type of 'i_indices' (line 230)
        i_indices_375087 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 45), 'i_indices', False)
        # Processing the call keyword arguments (line 230)
        kwargs_375088 = {}
        # Getting the type of 'map' (line 230)
        map_375085 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 36), 'map', False)
        # Calling map(args, kwargs) (line 230)
        map_call_result_375089 = invoke(stypy.reporting.localization.Localization(__file__, 230, 36), map_375085, *[int_375086, i_indices_375087], **kwargs_375088)
        
        # Obtaining the member '__getitem__' of a type (line 230)
        getitem___375090 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 230, 8), map_call_result_375089, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 230)
        subscript_call_result_375091 = invoke(stypy.reporting.localization.Localization(__file__, 230, 8), getitem___375090, int_375084)
        
        # Assigning a type to the variable 'tuple_var_assignment_374348' (line 230)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 230, 8), 'tuple_var_assignment_374348', subscript_call_result_375091)
        
        # Assigning a Subscript to a Name (line 230):
        
        # Obtaining the type of the subscript
        int_375092 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 230, 8), 'int')
        
        # Call to map(...): (line 230)
        # Processing the call arguments (line 230)
        # Getting the type of 'int' (line 230)
        int_375094 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 40), 'int', False)
        # Getting the type of 'i_indices' (line 230)
        i_indices_375095 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 45), 'i_indices', False)
        # Processing the call keyword arguments (line 230)
        kwargs_375096 = {}
        # Getting the type of 'map' (line 230)
        map_375093 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 36), 'map', False)
        # Calling map(args, kwargs) (line 230)
        map_call_result_375097 = invoke(stypy.reporting.localization.Localization(__file__, 230, 36), map_375093, *[int_375094, i_indices_375095], **kwargs_375096)
        
        # Obtaining the member '__getitem__' of a type (line 230)
        getitem___375098 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 230, 8), map_call_result_375097, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 230)
        subscript_call_result_375099 = invoke(stypy.reporting.localization.Localization(__file__, 230, 8), getitem___375098, int_375092)
        
        # Assigning a type to the variable 'tuple_var_assignment_374349' (line 230)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 230, 8), 'tuple_var_assignment_374349', subscript_call_result_375099)
        
        # Assigning a Subscript to a Name (line 230):
        
        # Obtaining the type of the subscript
        int_375100 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 230, 8), 'int')
        
        # Call to map(...): (line 230)
        # Processing the call arguments (line 230)
        # Getting the type of 'int' (line 230)
        int_375102 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 40), 'int', False)
        # Getting the type of 'i_indices' (line 230)
        i_indices_375103 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 45), 'i_indices', False)
        # Processing the call keyword arguments (line 230)
        kwargs_375104 = {}
        # Getting the type of 'map' (line 230)
        map_375101 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 36), 'map', False)
        # Calling map(args, kwargs) (line 230)
        map_call_result_375105 = invoke(stypy.reporting.localization.Localization(__file__, 230, 36), map_375101, *[int_375102, i_indices_375103], **kwargs_375104)
        
        # Obtaining the member '__getitem__' of a type (line 230)
        getitem___375106 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 230, 8), map_call_result_375105, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 230)
        subscript_call_result_375107 = invoke(stypy.reporting.localization.Localization(__file__, 230, 8), getitem___375106, int_375100)
        
        # Assigning a type to the variable 'tuple_var_assignment_374350' (line 230)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 230, 8), 'tuple_var_assignment_374350', subscript_call_result_375107)
        
        # Assigning a Name to a Name (line 230):
        # Getting the type of 'tuple_var_assignment_374348' (line 230)
        tuple_var_assignment_374348_375108 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 8), 'tuple_var_assignment_374348')
        # Assigning a type to the variable 'i_start' (line 230)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 230, 8), 'i_start', tuple_var_assignment_374348_375108)
        
        # Assigning a Name to a Name (line 230):
        # Getting the type of 'tuple_var_assignment_374349' (line 230)
        tuple_var_assignment_374349_375109 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 8), 'tuple_var_assignment_374349')
        # Assigning a type to the variable 'i_stop' (line 230)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 230, 17), 'i_stop', tuple_var_assignment_374349_375109)
        
        # Assigning a Name to a Name (line 230):
        # Getting the type of 'tuple_var_assignment_374350' (line 230)
        tuple_var_assignment_374350_375110 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 8), 'tuple_var_assignment_374350')
        # Assigning a type to the variable 'i_stride' (line 230)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 230, 25), 'i_stride', tuple_var_assignment_374350_375110)
        
        # Assigning a Call to a Tuple (line 231):
        
        # Assigning a Subscript to a Name (line 231):
        
        # Obtaining the type of the subscript
        int_375111 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 231, 8), 'int')
        
        # Call to map(...): (line 231)
        # Processing the call arguments (line 231)
        # Getting the type of 'int' (line 231)
        int_375113 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 231, 40), 'int', False)
        # Getting the type of 'j_indices' (line 231)
        j_indices_375114 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 231, 45), 'j_indices', False)
        # Processing the call keyword arguments (line 231)
        kwargs_375115 = {}
        # Getting the type of 'map' (line 231)
        map_375112 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 231, 36), 'map', False)
        # Calling map(args, kwargs) (line 231)
        map_call_result_375116 = invoke(stypy.reporting.localization.Localization(__file__, 231, 36), map_375112, *[int_375113, j_indices_375114], **kwargs_375115)
        
        # Obtaining the member '__getitem__' of a type (line 231)
        getitem___375117 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 231, 8), map_call_result_375116, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 231)
        subscript_call_result_375118 = invoke(stypy.reporting.localization.Localization(__file__, 231, 8), getitem___375117, int_375111)
        
        # Assigning a type to the variable 'tuple_var_assignment_374351' (line 231)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 231, 8), 'tuple_var_assignment_374351', subscript_call_result_375118)
        
        # Assigning a Subscript to a Name (line 231):
        
        # Obtaining the type of the subscript
        int_375119 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 231, 8), 'int')
        
        # Call to map(...): (line 231)
        # Processing the call arguments (line 231)
        # Getting the type of 'int' (line 231)
        int_375121 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 231, 40), 'int', False)
        # Getting the type of 'j_indices' (line 231)
        j_indices_375122 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 231, 45), 'j_indices', False)
        # Processing the call keyword arguments (line 231)
        kwargs_375123 = {}
        # Getting the type of 'map' (line 231)
        map_375120 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 231, 36), 'map', False)
        # Calling map(args, kwargs) (line 231)
        map_call_result_375124 = invoke(stypy.reporting.localization.Localization(__file__, 231, 36), map_375120, *[int_375121, j_indices_375122], **kwargs_375123)
        
        # Obtaining the member '__getitem__' of a type (line 231)
        getitem___375125 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 231, 8), map_call_result_375124, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 231)
        subscript_call_result_375126 = invoke(stypy.reporting.localization.Localization(__file__, 231, 8), getitem___375125, int_375119)
        
        # Assigning a type to the variable 'tuple_var_assignment_374352' (line 231)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 231, 8), 'tuple_var_assignment_374352', subscript_call_result_375126)
        
        # Assigning a Subscript to a Name (line 231):
        
        # Obtaining the type of the subscript
        int_375127 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 231, 8), 'int')
        
        # Call to map(...): (line 231)
        # Processing the call arguments (line 231)
        # Getting the type of 'int' (line 231)
        int_375129 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 231, 40), 'int', False)
        # Getting the type of 'j_indices' (line 231)
        j_indices_375130 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 231, 45), 'j_indices', False)
        # Processing the call keyword arguments (line 231)
        kwargs_375131 = {}
        # Getting the type of 'map' (line 231)
        map_375128 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 231, 36), 'map', False)
        # Calling map(args, kwargs) (line 231)
        map_call_result_375132 = invoke(stypy.reporting.localization.Localization(__file__, 231, 36), map_375128, *[int_375129, j_indices_375130], **kwargs_375131)
        
        # Obtaining the member '__getitem__' of a type (line 231)
        getitem___375133 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 231, 8), map_call_result_375132, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 231)
        subscript_call_result_375134 = invoke(stypy.reporting.localization.Localization(__file__, 231, 8), getitem___375133, int_375127)
        
        # Assigning a type to the variable 'tuple_var_assignment_374353' (line 231)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 231, 8), 'tuple_var_assignment_374353', subscript_call_result_375134)
        
        # Assigning a Name to a Name (line 231):
        # Getting the type of 'tuple_var_assignment_374351' (line 231)
        tuple_var_assignment_374351_375135 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 231, 8), 'tuple_var_assignment_374351')
        # Assigning a type to the variable 'j_start' (line 231)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 231, 8), 'j_start', tuple_var_assignment_374351_375135)
        
        # Assigning a Name to a Name (line 231):
        # Getting the type of 'tuple_var_assignment_374352' (line 231)
        tuple_var_assignment_374352_375136 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 231, 8), 'tuple_var_assignment_374352')
        # Assigning a type to the variable 'j_stop' (line 231)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 231, 17), 'j_stop', tuple_var_assignment_374352_375136)
        
        # Assigning a Name to a Name (line 231):
        # Getting the type of 'tuple_var_assignment_374353' (line 231)
        tuple_var_assignment_374353_375137 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 231, 8), 'tuple_var_assignment_374353')
        # Assigning a type to the variable 'j_stride' (line 231)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 231, 25), 'j_stride', tuple_var_assignment_374353_375137)
        
        # Assigning a Call to a Name (line 233):
        
        # Assigning a Call to a Name (line 233):
        
        # Call to dok_matrix(...): (line 233)
        # Processing the call arguments (line 233)
        # Getting the type of 'shape' (line 233)
        shape_375139 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 233, 28), 'shape', False)
        # Processing the call keyword arguments (line 233)
        # Getting the type of 'self' (line 233)
        self_375140 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 233, 41), 'self', False)
        # Obtaining the member 'dtype' of a type (line 233)
        dtype_375141 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 233, 41), self_375140, 'dtype')
        keyword_375142 = dtype_375141
        kwargs_375143 = {'dtype': keyword_375142}
        # Getting the type of 'dok_matrix' (line 233)
        dok_matrix_375138 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 233, 17), 'dok_matrix', False)
        # Calling dok_matrix(args, kwargs) (line 233)
        dok_matrix_call_result_375144 = invoke(stypy.reporting.localization.Localization(__file__, 233, 17), dok_matrix_375138, *[shape_375139], **kwargs_375143)
        
        # Assigning a type to the variable 'newdok' (line 233)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 233, 8), 'newdok', dok_matrix_call_result_375144)
        
        
        # Call to iterkeys(...): (line 235)
        # Processing the call arguments (line 235)
        # Getting the type of 'self' (line 235)
        self_375146 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 33), 'self', False)
        # Processing the call keyword arguments (line 235)
        kwargs_375147 = {}
        # Getting the type of 'iterkeys' (line 235)
        iterkeys_375145 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 24), 'iterkeys', False)
        # Calling iterkeys(args, kwargs) (line 235)
        iterkeys_call_result_375148 = invoke(stypy.reporting.localization.Localization(__file__, 235, 24), iterkeys_375145, *[self_375146], **kwargs_375147)
        
        # Testing the type of a for loop iterable (line 235)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 235, 8), iterkeys_call_result_375148)
        # Getting the type of the for loop variable (line 235)
        for_loop_var_375149 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 235, 8), iterkeys_call_result_375148)
        # Assigning a type to the variable 'ii' (line 235)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 235, 8), 'ii', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 235, 8), for_loop_var_375149))
        # Assigning a type to the variable 'jj' (line 235)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 235, 8), 'jj', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 235, 8), for_loop_var_375149))
        # SSA begins for a for statement (line 235)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Call to a Name (line 237):
        
        # Assigning a Call to a Name (line 237):
        
        # Call to int(...): (line 237)
        # Processing the call arguments (line 237)
        # Getting the type of 'ii' (line 237)
        ii_375151 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 237, 21), 'ii', False)
        # Processing the call keyword arguments (line 237)
        kwargs_375152 = {}
        # Getting the type of 'int' (line 237)
        int_375150 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 237, 17), 'int', False)
        # Calling int(args, kwargs) (line 237)
        int_call_result_375153 = invoke(stypy.reporting.localization.Localization(__file__, 237, 17), int_375150, *[ii_375151], **kwargs_375152)
        
        # Assigning a type to the variable 'ii' (line 237)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 237, 12), 'ii', int_call_result_375153)
        
        # Assigning a Call to a Name (line 238):
        
        # Assigning a Call to a Name (line 238):
        
        # Call to int(...): (line 238)
        # Processing the call arguments (line 238)
        # Getting the type of 'jj' (line 238)
        jj_375155 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 238, 21), 'jj', False)
        # Processing the call keyword arguments (line 238)
        kwargs_375156 = {}
        # Getting the type of 'int' (line 238)
        int_375154 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 238, 17), 'int', False)
        # Calling int(args, kwargs) (line 238)
        int_call_result_375157 = invoke(stypy.reporting.localization.Localization(__file__, 238, 17), int_375154, *[jj_375155], **kwargs_375156)
        
        # Assigning a type to the variable 'jj' (line 238)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 238, 12), 'jj', int_call_result_375157)
        
        # Assigning a Call to a Tuple (line 239):
        
        # Assigning a Subscript to a Name (line 239):
        
        # Obtaining the type of the subscript
        int_375158 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 239, 12), 'int')
        
        # Call to divmod(...): (line 239)
        # Processing the call arguments (line 239)
        # Getting the type of 'ii' (line 239)
        ii_375160 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 239, 27), 'ii', False)
        # Getting the type of 'i_start' (line 239)
        i_start_375161 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 239, 32), 'i_start', False)
        # Applying the binary operator '-' (line 239)
        result_sub_375162 = python_operator(stypy.reporting.localization.Localization(__file__, 239, 27), '-', ii_375160, i_start_375161)
        
        # Getting the type of 'i_stride' (line 239)
        i_stride_375163 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 239, 41), 'i_stride', False)
        # Processing the call keyword arguments (line 239)
        kwargs_375164 = {}
        # Getting the type of 'divmod' (line 239)
        divmod_375159 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 239, 20), 'divmod', False)
        # Calling divmod(args, kwargs) (line 239)
        divmod_call_result_375165 = invoke(stypy.reporting.localization.Localization(__file__, 239, 20), divmod_375159, *[result_sub_375162, i_stride_375163], **kwargs_375164)
        
        # Obtaining the member '__getitem__' of a type (line 239)
        getitem___375166 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 239, 12), divmod_call_result_375165, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 239)
        subscript_call_result_375167 = invoke(stypy.reporting.localization.Localization(__file__, 239, 12), getitem___375166, int_375158)
        
        # Assigning a type to the variable 'tuple_var_assignment_374354' (line 239)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 239, 12), 'tuple_var_assignment_374354', subscript_call_result_375167)
        
        # Assigning a Subscript to a Name (line 239):
        
        # Obtaining the type of the subscript
        int_375168 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 239, 12), 'int')
        
        # Call to divmod(...): (line 239)
        # Processing the call arguments (line 239)
        # Getting the type of 'ii' (line 239)
        ii_375170 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 239, 27), 'ii', False)
        # Getting the type of 'i_start' (line 239)
        i_start_375171 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 239, 32), 'i_start', False)
        # Applying the binary operator '-' (line 239)
        result_sub_375172 = python_operator(stypy.reporting.localization.Localization(__file__, 239, 27), '-', ii_375170, i_start_375171)
        
        # Getting the type of 'i_stride' (line 239)
        i_stride_375173 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 239, 41), 'i_stride', False)
        # Processing the call keyword arguments (line 239)
        kwargs_375174 = {}
        # Getting the type of 'divmod' (line 239)
        divmod_375169 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 239, 20), 'divmod', False)
        # Calling divmod(args, kwargs) (line 239)
        divmod_call_result_375175 = invoke(stypy.reporting.localization.Localization(__file__, 239, 20), divmod_375169, *[result_sub_375172, i_stride_375173], **kwargs_375174)
        
        # Obtaining the member '__getitem__' of a type (line 239)
        getitem___375176 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 239, 12), divmod_call_result_375175, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 239)
        subscript_call_result_375177 = invoke(stypy.reporting.localization.Localization(__file__, 239, 12), getitem___375176, int_375168)
        
        # Assigning a type to the variable 'tuple_var_assignment_374355' (line 239)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 239, 12), 'tuple_var_assignment_374355', subscript_call_result_375177)
        
        # Assigning a Name to a Name (line 239):
        # Getting the type of 'tuple_var_assignment_374354' (line 239)
        tuple_var_assignment_374354_375178 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 239, 12), 'tuple_var_assignment_374354')
        # Assigning a type to the variable 'a' (line 239)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 239, 12), 'a', tuple_var_assignment_374354_375178)
        
        # Assigning a Name to a Name (line 239):
        # Getting the type of 'tuple_var_assignment_374355' (line 239)
        tuple_var_assignment_374355_375179 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 239, 12), 'tuple_var_assignment_374355')
        # Assigning a type to the variable 'ra' (line 239)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 239, 15), 'ra', tuple_var_assignment_374355_375179)
        
        
        # Evaluating a boolean operation
        
        # Getting the type of 'a' (line 240)
        a_375180 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 15), 'a')
        int_375181 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 240, 19), 'int')
        # Applying the binary operator '<' (line 240)
        result_lt_375182 = python_operator(stypy.reporting.localization.Localization(__file__, 240, 15), '<', a_375180, int_375181)
        
        
        # Getting the type of 'a' (line 240)
        a_375183 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 24), 'a')
        
        # Obtaining the type of the subscript
        int_375184 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 240, 35), 'int')
        # Getting the type of 'shape' (line 240)
        shape_375185 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 29), 'shape')
        # Obtaining the member '__getitem__' of a type (line 240)
        getitem___375186 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 240, 29), shape_375185, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 240)
        subscript_call_result_375187 = invoke(stypy.reporting.localization.Localization(__file__, 240, 29), getitem___375186, int_375184)
        
        # Applying the binary operator '>=' (line 240)
        result_ge_375188 = python_operator(stypy.reporting.localization.Localization(__file__, 240, 24), '>=', a_375183, subscript_call_result_375187)
        
        # Applying the binary operator 'or' (line 240)
        result_or_keyword_375189 = python_operator(stypy.reporting.localization.Localization(__file__, 240, 15), 'or', result_lt_375182, result_ge_375188)
        
        # Getting the type of 'ra' (line 240)
        ra_375190 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 41), 'ra')
        int_375191 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 240, 47), 'int')
        # Applying the binary operator '!=' (line 240)
        result_ne_375192 = python_operator(stypy.reporting.localization.Localization(__file__, 240, 41), '!=', ra_375190, int_375191)
        
        # Applying the binary operator 'or' (line 240)
        result_or_keyword_375193 = python_operator(stypy.reporting.localization.Localization(__file__, 240, 15), 'or', result_or_keyword_375189, result_ne_375192)
        
        # Testing the type of an if condition (line 240)
        if_condition_375194 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 240, 12), result_or_keyword_375193)
        # Assigning a type to the variable 'if_condition_375194' (line 240)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 240, 12), 'if_condition_375194', if_condition_375194)
        # SSA begins for if statement (line 240)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # SSA join for if statement (line 240)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Tuple (line 242):
        
        # Assigning a Subscript to a Name (line 242):
        
        # Obtaining the type of the subscript
        int_375195 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 242, 12), 'int')
        
        # Call to divmod(...): (line 242)
        # Processing the call arguments (line 242)
        # Getting the type of 'jj' (line 242)
        jj_375197 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 242, 27), 'jj', False)
        # Getting the type of 'j_start' (line 242)
        j_start_375198 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 242, 32), 'j_start', False)
        # Applying the binary operator '-' (line 242)
        result_sub_375199 = python_operator(stypy.reporting.localization.Localization(__file__, 242, 27), '-', jj_375197, j_start_375198)
        
        # Getting the type of 'j_stride' (line 242)
        j_stride_375200 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 242, 41), 'j_stride', False)
        # Processing the call keyword arguments (line 242)
        kwargs_375201 = {}
        # Getting the type of 'divmod' (line 242)
        divmod_375196 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 242, 20), 'divmod', False)
        # Calling divmod(args, kwargs) (line 242)
        divmod_call_result_375202 = invoke(stypy.reporting.localization.Localization(__file__, 242, 20), divmod_375196, *[result_sub_375199, j_stride_375200], **kwargs_375201)
        
        # Obtaining the member '__getitem__' of a type (line 242)
        getitem___375203 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 242, 12), divmod_call_result_375202, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 242)
        subscript_call_result_375204 = invoke(stypy.reporting.localization.Localization(__file__, 242, 12), getitem___375203, int_375195)
        
        # Assigning a type to the variable 'tuple_var_assignment_374356' (line 242)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 242, 12), 'tuple_var_assignment_374356', subscript_call_result_375204)
        
        # Assigning a Subscript to a Name (line 242):
        
        # Obtaining the type of the subscript
        int_375205 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 242, 12), 'int')
        
        # Call to divmod(...): (line 242)
        # Processing the call arguments (line 242)
        # Getting the type of 'jj' (line 242)
        jj_375207 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 242, 27), 'jj', False)
        # Getting the type of 'j_start' (line 242)
        j_start_375208 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 242, 32), 'j_start', False)
        # Applying the binary operator '-' (line 242)
        result_sub_375209 = python_operator(stypy.reporting.localization.Localization(__file__, 242, 27), '-', jj_375207, j_start_375208)
        
        # Getting the type of 'j_stride' (line 242)
        j_stride_375210 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 242, 41), 'j_stride', False)
        # Processing the call keyword arguments (line 242)
        kwargs_375211 = {}
        # Getting the type of 'divmod' (line 242)
        divmod_375206 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 242, 20), 'divmod', False)
        # Calling divmod(args, kwargs) (line 242)
        divmod_call_result_375212 = invoke(stypy.reporting.localization.Localization(__file__, 242, 20), divmod_375206, *[result_sub_375209, j_stride_375210], **kwargs_375211)
        
        # Obtaining the member '__getitem__' of a type (line 242)
        getitem___375213 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 242, 12), divmod_call_result_375212, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 242)
        subscript_call_result_375214 = invoke(stypy.reporting.localization.Localization(__file__, 242, 12), getitem___375213, int_375205)
        
        # Assigning a type to the variable 'tuple_var_assignment_374357' (line 242)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 242, 12), 'tuple_var_assignment_374357', subscript_call_result_375214)
        
        # Assigning a Name to a Name (line 242):
        # Getting the type of 'tuple_var_assignment_374356' (line 242)
        tuple_var_assignment_374356_375215 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 242, 12), 'tuple_var_assignment_374356')
        # Assigning a type to the variable 'b' (line 242)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 242, 12), 'b', tuple_var_assignment_374356_375215)
        
        # Assigning a Name to a Name (line 242):
        # Getting the type of 'tuple_var_assignment_374357' (line 242)
        tuple_var_assignment_374357_375216 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 242, 12), 'tuple_var_assignment_374357')
        # Assigning a type to the variable 'rb' (line 242)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 242, 15), 'rb', tuple_var_assignment_374357_375216)
        
        
        # Evaluating a boolean operation
        
        # Getting the type of 'b' (line 243)
        b_375217 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 243, 15), 'b')
        int_375218 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 243, 19), 'int')
        # Applying the binary operator '<' (line 243)
        result_lt_375219 = python_operator(stypy.reporting.localization.Localization(__file__, 243, 15), '<', b_375217, int_375218)
        
        
        # Getting the type of 'b' (line 243)
        b_375220 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 243, 24), 'b')
        
        # Obtaining the type of the subscript
        int_375221 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 243, 35), 'int')
        # Getting the type of 'shape' (line 243)
        shape_375222 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 243, 29), 'shape')
        # Obtaining the member '__getitem__' of a type (line 243)
        getitem___375223 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 243, 29), shape_375222, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 243)
        subscript_call_result_375224 = invoke(stypy.reporting.localization.Localization(__file__, 243, 29), getitem___375223, int_375221)
        
        # Applying the binary operator '>=' (line 243)
        result_ge_375225 = python_operator(stypy.reporting.localization.Localization(__file__, 243, 24), '>=', b_375220, subscript_call_result_375224)
        
        # Applying the binary operator 'or' (line 243)
        result_or_keyword_375226 = python_operator(stypy.reporting.localization.Localization(__file__, 243, 15), 'or', result_lt_375219, result_ge_375225)
        
        # Getting the type of 'rb' (line 243)
        rb_375227 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 243, 41), 'rb')
        int_375228 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 243, 47), 'int')
        # Applying the binary operator '!=' (line 243)
        result_ne_375229 = python_operator(stypy.reporting.localization.Localization(__file__, 243, 41), '!=', rb_375227, int_375228)
        
        # Applying the binary operator 'or' (line 243)
        result_or_keyword_375230 = python_operator(stypy.reporting.localization.Localization(__file__, 243, 15), 'or', result_or_keyword_375226, result_ne_375229)
        
        # Testing the type of an if condition (line 243)
        if_condition_375231 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 243, 12), result_or_keyword_375230)
        # Assigning a type to the variable 'if_condition_375231' (line 243)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 243, 12), 'if_condition_375231', if_condition_375231)
        # SSA begins for if statement (line 243)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # SSA join for if statement (line 243)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to __setitem__(...): (line 245)
        # Processing the call arguments (line 245)
        # Getting the type of 'newdok' (line 245)
        newdok_375234 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 245, 29), 'newdok', False)
        
        # Obtaining an instance of the builtin type 'tuple' (line 245)
        tuple_375235 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 245, 38), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 245)
        # Adding element type (line 245)
        # Getting the type of 'a' (line 245)
        a_375236 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 245, 38), 'a', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 245, 38), tuple_375235, a_375236)
        # Adding element type (line 245)
        # Getting the type of 'b' (line 245)
        b_375237 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 245, 41), 'b', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 245, 38), tuple_375235, b_375237)
        
        
        # Call to __getitem__(...): (line 246)
        # Processing the call arguments (line 246)
        # Getting the type of 'self' (line 246)
        self_375240 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 46), 'self', False)
        
        # Obtaining an instance of the builtin type 'tuple' (line 246)
        tuple_375241 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 246, 53), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 246)
        # Adding element type (line 246)
        # Getting the type of 'ii' (line 246)
        ii_375242 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 53), 'ii', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 246, 53), tuple_375241, ii_375242)
        # Adding element type (line 246)
        # Getting the type of 'jj' (line 246)
        jj_375243 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 57), 'jj', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 246, 53), tuple_375241, jj_375243)
        
        # Processing the call keyword arguments (line 246)
        kwargs_375244 = {}
        # Getting the type of 'dict' (line 246)
        dict_375238 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 29), 'dict', False)
        # Obtaining the member '__getitem__' of a type (line 246)
        getitem___375239 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 246, 29), dict_375238, '__getitem__')
        # Calling __getitem__(args, kwargs) (line 246)
        getitem___call_result_375245 = invoke(stypy.reporting.localization.Localization(__file__, 246, 29), getitem___375239, *[self_375240, tuple_375241], **kwargs_375244)
        
        # Processing the call keyword arguments (line 245)
        kwargs_375246 = {}
        # Getting the type of 'dict' (line 245)
        dict_375232 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 245, 12), 'dict', False)
        # Obtaining the member '__setitem__' of a type (line 245)
        setitem___375233 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 245, 12), dict_375232, '__setitem__')
        # Calling __setitem__(args, kwargs) (line 245)
        setitem___call_result_375247 = invoke(stypy.reporting.localization.Localization(__file__, 245, 12), setitem___375233, *[newdok_375234, tuple_375235, getitem___call_result_375245], **kwargs_375246)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        # Getting the type of 'newdok' (line 247)
        newdok_375248 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 247, 15), 'newdok')
        # Assigning a type to the variable 'stypy_return_type' (line 247)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 247, 8), 'stypy_return_type', newdok_375248)
        
        # ################# End of '_getitem_ranges(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_getitem_ranges' in the type store
        # Getting the type of 'stypy_return_type' (line 228)
        stypy_return_type_375249 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_375249)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_getitem_ranges'
        return stypy_return_type_375249


    @norecursion
    def __setitem__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__setitem__'
        module_type_store = module_type_store.open_function_context('__setitem__', 249, 4, False)
        # Assigning a type to the variable 'self' (line 250)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 250, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        dok_matrix.__setitem__.__dict__.__setitem__('stypy_localization', localization)
        dok_matrix.__setitem__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        dok_matrix.__setitem__.__dict__.__setitem__('stypy_type_store', module_type_store)
        dok_matrix.__setitem__.__dict__.__setitem__('stypy_function_name', 'dok_matrix.__setitem__')
        dok_matrix.__setitem__.__dict__.__setitem__('stypy_param_names_list', ['index', 'x'])
        dok_matrix.__setitem__.__dict__.__setitem__('stypy_varargs_param_name', None)
        dok_matrix.__setitem__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        dok_matrix.__setitem__.__dict__.__setitem__('stypy_call_defaults', defaults)
        dok_matrix.__setitem__.__dict__.__setitem__('stypy_call_varargs', varargs)
        dok_matrix.__setitem__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        dok_matrix.__setitem__.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'dok_matrix.__setitem__', ['index', 'x'], None, None, defaults, varargs, kwargs)

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
        
        # Call to isinstance(...): (line 250)
        # Processing the call arguments (line 250)
        # Getting the type of 'index' (line 250)
        index_375251 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 250, 22), 'index', False)
        # Getting the type of 'tuple' (line 250)
        tuple_375252 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 250, 29), 'tuple', False)
        # Processing the call keyword arguments (line 250)
        kwargs_375253 = {}
        # Getting the type of 'isinstance' (line 250)
        isinstance_375250 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 250, 11), 'isinstance', False)
        # Calling isinstance(args, kwargs) (line 250)
        isinstance_call_result_375254 = invoke(stypy.reporting.localization.Localization(__file__, 250, 11), isinstance_375250, *[index_375251, tuple_375252], **kwargs_375253)
        
        
        
        # Call to len(...): (line 250)
        # Processing the call arguments (line 250)
        # Getting the type of 'index' (line 250)
        index_375256 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 250, 44), 'index', False)
        # Processing the call keyword arguments (line 250)
        kwargs_375257 = {}
        # Getting the type of 'len' (line 250)
        len_375255 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 250, 40), 'len', False)
        # Calling len(args, kwargs) (line 250)
        len_call_result_375258 = invoke(stypy.reporting.localization.Localization(__file__, 250, 40), len_375255, *[index_375256], **kwargs_375257)
        
        int_375259 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 250, 54), 'int')
        # Applying the binary operator '==' (line 250)
        result_eq_375260 = python_operator(stypy.reporting.localization.Localization(__file__, 250, 40), '==', len_call_result_375258, int_375259)
        
        # Applying the binary operator 'and' (line 250)
        result_and_keyword_375261 = python_operator(stypy.reporting.localization.Localization(__file__, 250, 11), 'and', isinstance_call_result_375254, result_eq_375260)
        
        # Testing the type of an if condition (line 250)
        if_condition_375262 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 250, 8), result_and_keyword_375261)
        # Assigning a type to the variable 'if_condition_375262' (line 250)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 250, 8), 'if_condition_375262', if_condition_375262)
        # SSA begins for if statement (line 250)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Name to a Tuple (line 252):
        
        # Assigning a Subscript to a Name (line 252):
        
        # Obtaining the type of the subscript
        int_375263 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 252, 12), 'int')
        # Getting the type of 'index' (line 252)
        index_375264 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 252, 19), 'index')
        # Obtaining the member '__getitem__' of a type (line 252)
        getitem___375265 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 252, 12), index_375264, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 252)
        subscript_call_result_375266 = invoke(stypy.reporting.localization.Localization(__file__, 252, 12), getitem___375265, int_375263)
        
        # Assigning a type to the variable 'tuple_var_assignment_374358' (line 252)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 252, 12), 'tuple_var_assignment_374358', subscript_call_result_375266)
        
        # Assigning a Subscript to a Name (line 252):
        
        # Obtaining the type of the subscript
        int_375267 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 252, 12), 'int')
        # Getting the type of 'index' (line 252)
        index_375268 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 252, 19), 'index')
        # Obtaining the member '__getitem__' of a type (line 252)
        getitem___375269 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 252, 12), index_375268, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 252)
        subscript_call_result_375270 = invoke(stypy.reporting.localization.Localization(__file__, 252, 12), getitem___375269, int_375267)
        
        # Assigning a type to the variable 'tuple_var_assignment_374359' (line 252)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 252, 12), 'tuple_var_assignment_374359', subscript_call_result_375270)
        
        # Assigning a Name to a Name (line 252):
        # Getting the type of 'tuple_var_assignment_374358' (line 252)
        tuple_var_assignment_374358_375271 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 252, 12), 'tuple_var_assignment_374358')
        # Assigning a type to the variable 'i' (line 252)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 252, 12), 'i', tuple_var_assignment_374358_375271)
        
        # Assigning a Name to a Name (line 252):
        # Getting the type of 'tuple_var_assignment_374359' (line 252)
        tuple_var_assignment_374359_375272 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 252, 12), 'tuple_var_assignment_374359')
        # Assigning a type to the variable 'j' (line 252)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 252, 15), 'j', tuple_var_assignment_374359_375272)
        
        
        # Evaluating a boolean operation
        
        # Call to isintlike(...): (line 253)
        # Processing the call arguments (line 253)
        # Getting the type of 'i' (line 253)
        i_375274 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 253, 26), 'i', False)
        # Processing the call keyword arguments (line 253)
        kwargs_375275 = {}
        # Getting the type of 'isintlike' (line 253)
        isintlike_375273 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 253, 16), 'isintlike', False)
        # Calling isintlike(args, kwargs) (line 253)
        isintlike_call_result_375276 = invoke(stypy.reporting.localization.Localization(__file__, 253, 16), isintlike_375273, *[i_375274], **kwargs_375275)
        
        
        # Call to isintlike(...): (line 253)
        # Processing the call arguments (line 253)
        # Getting the type of 'j' (line 253)
        j_375278 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 253, 43), 'j', False)
        # Processing the call keyword arguments (line 253)
        kwargs_375279 = {}
        # Getting the type of 'isintlike' (line 253)
        isintlike_375277 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 253, 33), 'isintlike', False)
        # Calling isintlike(args, kwargs) (line 253)
        isintlike_call_result_375280 = invoke(stypy.reporting.localization.Localization(__file__, 253, 33), isintlike_375277, *[j_375278], **kwargs_375279)
        
        # Applying the binary operator 'and' (line 253)
        result_and_keyword_375281 = python_operator(stypy.reporting.localization.Localization(__file__, 253, 16), 'and', isintlike_call_result_375276, isintlike_call_result_375280)
        
        int_375282 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 253, 50), 'int')
        # Getting the type of 'i' (line 253)
        i_375283 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 253, 55), 'i')
        # Applying the binary operator '<=' (line 253)
        result_le_375284 = python_operator(stypy.reporting.localization.Localization(__file__, 253, 50), '<=', int_375282, i_375283)
        
        # Obtaining the type of the subscript
        int_375285 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 253, 70), 'int')
        # Getting the type of 'self' (line 253)
        self_375286 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 253, 59), 'self')
        # Obtaining the member 'shape' of a type (line 253)
        shape_375287 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 253, 59), self_375286, 'shape')
        # Obtaining the member '__getitem__' of a type (line 253)
        getitem___375288 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 253, 59), shape_375287, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 253)
        subscript_call_result_375289 = invoke(stypy.reporting.localization.Localization(__file__, 253, 59), getitem___375288, int_375285)
        
        # Applying the binary operator '<' (line 253)
        result_lt_375290 = python_operator(stypy.reporting.localization.Localization(__file__, 253, 50), '<', i_375283, subscript_call_result_375289)
        # Applying the binary operator '&' (line 253)
        result_and__375291 = python_operator(stypy.reporting.localization.Localization(__file__, 253, 50), '&', result_le_375284, result_lt_375290)
        
        # Applying the binary operator 'and' (line 253)
        result_and_keyword_375292 = python_operator(stypy.reporting.localization.Localization(__file__, 253, 16), 'and', result_and_keyword_375281, result_and__375291)
        
        int_375293 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 254, 24), 'int')
        # Getting the type of 'j' (line 254)
        j_375294 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 254, 29), 'j')
        # Applying the binary operator '<=' (line 254)
        result_le_375295 = python_operator(stypy.reporting.localization.Localization(__file__, 254, 24), '<=', int_375293, j_375294)
        
        # Obtaining the type of the subscript
        int_375296 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 254, 44), 'int')
        # Getting the type of 'self' (line 254)
        self_375297 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 254, 33), 'self')
        # Obtaining the member 'shape' of a type (line 254)
        shape_375298 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 254, 33), self_375297, 'shape')
        # Obtaining the member '__getitem__' of a type (line 254)
        getitem___375299 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 254, 33), shape_375298, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 254)
        subscript_call_result_375300 = invoke(stypy.reporting.localization.Localization(__file__, 254, 33), getitem___375299, int_375296)
        
        # Applying the binary operator '<' (line 254)
        result_lt_375301 = python_operator(stypy.reporting.localization.Localization(__file__, 254, 24), '<', j_375294, subscript_call_result_375300)
        # Applying the binary operator '&' (line 254)
        result_and__375302 = python_operator(stypy.reporting.localization.Localization(__file__, 254, 24), '&', result_le_375295, result_lt_375301)
        
        # Applying the binary operator 'and' (line 253)
        result_and_keyword_375303 = python_operator(stypy.reporting.localization.Localization(__file__, 253, 16), 'and', result_and_keyword_375292, result_and__375302)
        
        # Testing the type of an if condition (line 253)
        if_condition_375304 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 253, 12), result_and_keyword_375303)
        # Assigning a type to the variable 'if_condition_375304' (line 253)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 253, 12), 'if_condition_375304', if_condition_375304)
        # SSA begins for if statement (line 253)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 255):
        
        # Assigning a Call to a Name (line 255):
        
        # Call to asarray(...): (line 255)
        # Processing the call arguments (line 255)
        # Getting the type of 'x' (line 255)
        x_375307 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 255, 31), 'x', False)
        # Processing the call keyword arguments (line 255)
        # Getting the type of 'self' (line 255)
        self_375308 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 255, 40), 'self', False)
        # Obtaining the member 'dtype' of a type (line 255)
        dtype_375309 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 255, 40), self_375308, 'dtype')
        keyword_375310 = dtype_375309
        kwargs_375311 = {'dtype': keyword_375310}
        # Getting the type of 'np' (line 255)
        np_375305 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 255, 20), 'np', False)
        # Obtaining the member 'asarray' of a type (line 255)
        asarray_375306 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 255, 20), np_375305, 'asarray')
        # Calling asarray(args, kwargs) (line 255)
        asarray_call_result_375312 = invoke(stypy.reporting.localization.Localization(__file__, 255, 20), asarray_375306, *[x_375307], **kwargs_375311)
        
        # Assigning a type to the variable 'v' (line 255)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 255, 16), 'v', asarray_call_result_375312)
        
        
        # Evaluating a boolean operation
        
        # Getting the type of 'v' (line 256)
        v_375313 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 256, 19), 'v')
        # Obtaining the member 'ndim' of a type (line 256)
        ndim_375314 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 256, 19), v_375313, 'ndim')
        int_375315 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 256, 29), 'int')
        # Applying the binary operator '==' (line 256)
        result_eq_375316 = python_operator(stypy.reporting.localization.Localization(__file__, 256, 19), '==', ndim_375314, int_375315)
        
        
        # Getting the type of 'v' (line 256)
        v_375317 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 256, 35), 'v')
        int_375318 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 256, 40), 'int')
        # Applying the binary operator '!=' (line 256)
        result_ne_375319 = python_operator(stypy.reporting.localization.Localization(__file__, 256, 35), '!=', v_375317, int_375318)
        
        # Applying the binary operator 'and' (line 256)
        result_and_keyword_375320 = python_operator(stypy.reporting.localization.Localization(__file__, 256, 19), 'and', result_eq_375316, result_ne_375319)
        
        # Testing the type of an if condition (line 256)
        if_condition_375321 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 256, 16), result_and_keyword_375320)
        # Assigning a type to the variable 'if_condition_375321' (line 256)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 256, 16), 'if_condition_375321', if_condition_375321)
        # SSA begins for if statement (line 256)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to __setitem__(...): (line 257)
        # Processing the call arguments (line 257)
        # Getting the type of 'self' (line 257)
        self_375324 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 257, 37), 'self', False)
        
        # Obtaining an instance of the builtin type 'tuple' (line 257)
        tuple_375325 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 257, 44), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 257)
        # Adding element type (line 257)
        
        # Call to int(...): (line 257)
        # Processing the call arguments (line 257)
        # Getting the type of 'i' (line 257)
        i_375327 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 257, 48), 'i', False)
        # Processing the call keyword arguments (line 257)
        kwargs_375328 = {}
        # Getting the type of 'int' (line 257)
        int_375326 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 257, 44), 'int', False)
        # Calling int(args, kwargs) (line 257)
        int_call_result_375329 = invoke(stypy.reporting.localization.Localization(__file__, 257, 44), int_375326, *[i_375327], **kwargs_375328)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 257, 44), tuple_375325, int_call_result_375329)
        # Adding element type (line 257)
        
        # Call to int(...): (line 257)
        # Processing the call arguments (line 257)
        # Getting the type of 'j' (line 257)
        j_375331 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 257, 56), 'j', False)
        # Processing the call keyword arguments (line 257)
        kwargs_375332 = {}
        # Getting the type of 'int' (line 257)
        int_375330 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 257, 52), 'int', False)
        # Calling int(args, kwargs) (line 257)
        int_call_result_375333 = invoke(stypy.reporting.localization.Localization(__file__, 257, 52), int_375330, *[j_375331], **kwargs_375332)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 257, 44), tuple_375325, int_call_result_375333)
        
        
        # Obtaining the type of the subscript
        
        # Obtaining an instance of the builtin type 'tuple' (line 257)
        tuple_375334 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 257, 63), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 257)
        
        # Getting the type of 'v' (line 257)
        v_375335 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 257, 61), 'v', False)
        # Obtaining the member '__getitem__' of a type (line 257)
        getitem___375336 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 257, 61), v_375335, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 257)
        subscript_call_result_375337 = invoke(stypy.reporting.localization.Localization(__file__, 257, 61), getitem___375336, tuple_375334)
        
        # Processing the call keyword arguments (line 257)
        kwargs_375338 = {}
        # Getting the type of 'dict' (line 257)
        dict_375322 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 257, 20), 'dict', False)
        # Obtaining the member '__setitem__' of a type (line 257)
        setitem___375323 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 257, 20), dict_375322, '__setitem__')
        # Calling __setitem__(args, kwargs) (line 257)
        setitem___call_result_375339 = invoke(stypy.reporting.localization.Localization(__file__, 257, 20), setitem___375323, *[self_375324, tuple_375325, subscript_call_result_375337], **kwargs_375338)
        
        # Assigning a type to the variable 'stypy_return_type' (line 258)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 258, 20), 'stypy_return_type', types.NoneType)
        # SSA join for if statement (line 256)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 253)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 250)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Tuple (line 260):
        
        # Assigning a Subscript to a Name (line 260):
        
        # Obtaining the type of the subscript
        int_375340 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 260, 8), 'int')
        
        # Call to _unpack_index(...): (line 260)
        # Processing the call arguments (line 260)
        # Getting the type of 'index' (line 260)
        index_375343 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 260, 34), 'index', False)
        # Processing the call keyword arguments (line 260)
        kwargs_375344 = {}
        # Getting the type of 'self' (line 260)
        self_375341 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 260, 15), 'self', False)
        # Obtaining the member '_unpack_index' of a type (line 260)
        _unpack_index_375342 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 260, 15), self_375341, '_unpack_index')
        # Calling _unpack_index(args, kwargs) (line 260)
        _unpack_index_call_result_375345 = invoke(stypy.reporting.localization.Localization(__file__, 260, 15), _unpack_index_375342, *[index_375343], **kwargs_375344)
        
        # Obtaining the member '__getitem__' of a type (line 260)
        getitem___375346 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 260, 8), _unpack_index_call_result_375345, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 260)
        subscript_call_result_375347 = invoke(stypy.reporting.localization.Localization(__file__, 260, 8), getitem___375346, int_375340)
        
        # Assigning a type to the variable 'tuple_var_assignment_374360' (line 260)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 260, 8), 'tuple_var_assignment_374360', subscript_call_result_375347)
        
        # Assigning a Subscript to a Name (line 260):
        
        # Obtaining the type of the subscript
        int_375348 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 260, 8), 'int')
        
        # Call to _unpack_index(...): (line 260)
        # Processing the call arguments (line 260)
        # Getting the type of 'index' (line 260)
        index_375351 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 260, 34), 'index', False)
        # Processing the call keyword arguments (line 260)
        kwargs_375352 = {}
        # Getting the type of 'self' (line 260)
        self_375349 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 260, 15), 'self', False)
        # Obtaining the member '_unpack_index' of a type (line 260)
        _unpack_index_375350 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 260, 15), self_375349, '_unpack_index')
        # Calling _unpack_index(args, kwargs) (line 260)
        _unpack_index_call_result_375353 = invoke(stypy.reporting.localization.Localization(__file__, 260, 15), _unpack_index_375350, *[index_375351], **kwargs_375352)
        
        # Obtaining the member '__getitem__' of a type (line 260)
        getitem___375354 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 260, 8), _unpack_index_call_result_375353, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 260)
        subscript_call_result_375355 = invoke(stypy.reporting.localization.Localization(__file__, 260, 8), getitem___375354, int_375348)
        
        # Assigning a type to the variable 'tuple_var_assignment_374361' (line 260)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 260, 8), 'tuple_var_assignment_374361', subscript_call_result_375355)
        
        # Assigning a Name to a Name (line 260):
        # Getting the type of 'tuple_var_assignment_374360' (line 260)
        tuple_var_assignment_374360_375356 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 260, 8), 'tuple_var_assignment_374360')
        # Assigning a type to the variable 'i' (line 260)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 260, 8), 'i', tuple_var_assignment_374360_375356)
        
        # Assigning a Name to a Name (line 260):
        # Getting the type of 'tuple_var_assignment_374361' (line 260)
        tuple_var_assignment_374361_375357 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 260, 8), 'tuple_var_assignment_374361')
        # Assigning a type to the variable 'j' (line 260)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 260, 11), 'j', tuple_var_assignment_374361_375357)
        
        # Assigning a Call to a Tuple (line 261):
        
        # Assigning a Subscript to a Name (line 261):
        
        # Obtaining the type of the subscript
        int_375358 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 261, 8), 'int')
        
        # Call to _index_to_arrays(...): (line 261)
        # Processing the call arguments (line 261)
        # Getting the type of 'i' (line 261)
        i_375361 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 261, 37), 'i', False)
        # Getting the type of 'j' (line 261)
        j_375362 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 261, 40), 'j', False)
        # Processing the call keyword arguments (line 261)
        kwargs_375363 = {}
        # Getting the type of 'self' (line 261)
        self_375359 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 261, 15), 'self', False)
        # Obtaining the member '_index_to_arrays' of a type (line 261)
        _index_to_arrays_375360 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 261, 15), self_375359, '_index_to_arrays')
        # Calling _index_to_arrays(args, kwargs) (line 261)
        _index_to_arrays_call_result_375364 = invoke(stypy.reporting.localization.Localization(__file__, 261, 15), _index_to_arrays_375360, *[i_375361, j_375362], **kwargs_375363)
        
        # Obtaining the member '__getitem__' of a type (line 261)
        getitem___375365 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 261, 8), _index_to_arrays_call_result_375364, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 261)
        subscript_call_result_375366 = invoke(stypy.reporting.localization.Localization(__file__, 261, 8), getitem___375365, int_375358)
        
        # Assigning a type to the variable 'tuple_var_assignment_374362' (line 261)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 261, 8), 'tuple_var_assignment_374362', subscript_call_result_375366)
        
        # Assigning a Subscript to a Name (line 261):
        
        # Obtaining the type of the subscript
        int_375367 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 261, 8), 'int')
        
        # Call to _index_to_arrays(...): (line 261)
        # Processing the call arguments (line 261)
        # Getting the type of 'i' (line 261)
        i_375370 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 261, 37), 'i', False)
        # Getting the type of 'j' (line 261)
        j_375371 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 261, 40), 'j', False)
        # Processing the call keyword arguments (line 261)
        kwargs_375372 = {}
        # Getting the type of 'self' (line 261)
        self_375368 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 261, 15), 'self', False)
        # Obtaining the member '_index_to_arrays' of a type (line 261)
        _index_to_arrays_375369 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 261, 15), self_375368, '_index_to_arrays')
        # Calling _index_to_arrays(args, kwargs) (line 261)
        _index_to_arrays_call_result_375373 = invoke(stypy.reporting.localization.Localization(__file__, 261, 15), _index_to_arrays_375369, *[i_375370, j_375371], **kwargs_375372)
        
        # Obtaining the member '__getitem__' of a type (line 261)
        getitem___375374 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 261, 8), _index_to_arrays_call_result_375373, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 261)
        subscript_call_result_375375 = invoke(stypy.reporting.localization.Localization(__file__, 261, 8), getitem___375374, int_375367)
        
        # Assigning a type to the variable 'tuple_var_assignment_374363' (line 261)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 261, 8), 'tuple_var_assignment_374363', subscript_call_result_375375)
        
        # Assigning a Name to a Name (line 261):
        # Getting the type of 'tuple_var_assignment_374362' (line 261)
        tuple_var_assignment_374362_375376 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 261, 8), 'tuple_var_assignment_374362')
        # Assigning a type to the variable 'i' (line 261)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 261, 8), 'i', tuple_var_assignment_374362_375376)
        
        # Assigning a Name to a Name (line 261):
        # Getting the type of 'tuple_var_assignment_374363' (line 261)
        tuple_var_assignment_374363_375377 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 261, 8), 'tuple_var_assignment_374363')
        # Assigning a type to the variable 'j' (line 261)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 261, 11), 'j', tuple_var_assignment_374363_375377)
        
        
        # Call to isspmatrix(...): (line 263)
        # Processing the call arguments (line 263)
        # Getting the type of 'x' (line 263)
        x_375379 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 263, 22), 'x', False)
        # Processing the call keyword arguments (line 263)
        kwargs_375380 = {}
        # Getting the type of 'isspmatrix' (line 263)
        isspmatrix_375378 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 263, 11), 'isspmatrix', False)
        # Calling isspmatrix(args, kwargs) (line 263)
        isspmatrix_call_result_375381 = invoke(stypy.reporting.localization.Localization(__file__, 263, 11), isspmatrix_375378, *[x_375379], **kwargs_375380)
        
        # Testing the type of an if condition (line 263)
        if_condition_375382 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 263, 8), isspmatrix_call_result_375381)
        # Assigning a type to the variable 'if_condition_375382' (line 263)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 263, 8), 'if_condition_375382', if_condition_375382)
        # SSA begins for if statement (line 263)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 264):
        
        # Assigning a Call to a Name (line 264):
        
        # Call to toarray(...): (line 264)
        # Processing the call keyword arguments (line 264)
        kwargs_375385 = {}
        # Getting the type of 'x' (line 264)
        x_375383 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 264, 16), 'x', False)
        # Obtaining the member 'toarray' of a type (line 264)
        toarray_375384 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 264, 16), x_375383, 'toarray')
        # Calling toarray(args, kwargs) (line 264)
        toarray_call_result_375386 = invoke(stypy.reporting.localization.Localization(__file__, 264, 16), toarray_375384, *[], **kwargs_375385)
        
        # Assigning a type to the variable 'x' (line 264)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 264, 12), 'x', toarray_call_result_375386)
        # SSA join for if statement (line 263)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Name (line 267):
        
        # Assigning a Call to a Name (line 267):
        
        # Call to asarray(...): (line 267)
        # Processing the call arguments (line 267)
        # Getting the type of 'x' (line 267)
        x_375389 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 267, 23), 'x', False)
        # Processing the call keyword arguments (line 267)
        # Getting the type of 'self' (line 267)
        self_375390 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 267, 32), 'self', False)
        # Obtaining the member 'dtype' of a type (line 267)
        dtype_375391 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 267, 32), self_375390, 'dtype')
        keyword_375392 = dtype_375391
        kwargs_375393 = {'dtype': keyword_375392}
        # Getting the type of 'np' (line 267)
        np_375387 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 267, 12), 'np', False)
        # Obtaining the member 'asarray' of a type (line 267)
        asarray_375388 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 267, 12), np_375387, 'asarray')
        # Calling asarray(args, kwargs) (line 267)
        asarray_call_result_375394 = invoke(stypy.reporting.localization.Localization(__file__, 267, 12), asarray_375388, *[x_375389], **kwargs_375393)
        
        # Assigning a type to the variable 'x' (line 267)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 267, 8), 'x', asarray_call_result_375394)
        
        # Assigning a Call to a Tuple (line 268):
        
        # Assigning a Subscript to a Name (line 268):
        
        # Obtaining the type of the subscript
        int_375395 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 268, 8), 'int')
        
        # Call to broadcast_arrays(...): (line 268)
        # Processing the call arguments (line 268)
        # Getting the type of 'x' (line 268)
        x_375398 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 268, 35), 'x', False)
        # Getting the type of 'i' (line 268)
        i_375399 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 268, 38), 'i', False)
        # Processing the call keyword arguments (line 268)
        kwargs_375400 = {}
        # Getting the type of 'np' (line 268)
        np_375396 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 268, 15), 'np', False)
        # Obtaining the member 'broadcast_arrays' of a type (line 268)
        broadcast_arrays_375397 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 268, 15), np_375396, 'broadcast_arrays')
        # Calling broadcast_arrays(args, kwargs) (line 268)
        broadcast_arrays_call_result_375401 = invoke(stypy.reporting.localization.Localization(__file__, 268, 15), broadcast_arrays_375397, *[x_375398, i_375399], **kwargs_375400)
        
        # Obtaining the member '__getitem__' of a type (line 268)
        getitem___375402 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 268, 8), broadcast_arrays_call_result_375401, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 268)
        subscript_call_result_375403 = invoke(stypy.reporting.localization.Localization(__file__, 268, 8), getitem___375402, int_375395)
        
        # Assigning a type to the variable 'tuple_var_assignment_374364' (line 268)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 268, 8), 'tuple_var_assignment_374364', subscript_call_result_375403)
        
        # Assigning a Subscript to a Name (line 268):
        
        # Obtaining the type of the subscript
        int_375404 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 268, 8), 'int')
        
        # Call to broadcast_arrays(...): (line 268)
        # Processing the call arguments (line 268)
        # Getting the type of 'x' (line 268)
        x_375407 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 268, 35), 'x', False)
        # Getting the type of 'i' (line 268)
        i_375408 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 268, 38), 'i', False)
        # Processing the call keyword arguments (line 268)
        kwargs_375409 = {}
        # Getting the type of 'np' (line 268)
        np_375405 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 268, 15), 'np', False)
        # Obtaining the member 'broadcast_arrays' of a type (line 268)
        broadcast_arrays_375406 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 268, 15), np_375405, 'broadcast_arrays')
        # Calling broadcast_arrays(args, kwargs) (line 268)
        broadcast_arrays_call_result_375410 = invoke(stypy.reporting.localization.Localization(__file__, 268, 15), broadcast_arrays_375406, *[x_375407, i_375408], **kwargs_375409)
        
        # Obtaining the member '__getitem__' of a type (line 268)
        getitem___375411 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 268, 8), broadcast_arrays_call_result_375410, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 268)
        subscript_call_result_375412 = invoke(stypy.reporting.localization.Localization(__file__, 268, 8), getitem___375411, int_375404)
        
        # Assigning a type to the variable 'tuple_var_assignment_374365' (line 268)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 268, 8), 'tuple_var_assignment_374365', subscript_call_result_375412)
        
        # Assigning a Name to a Name (line 268):
        # Getting the type of 'tuple_var_assignment_374364' (line 268)
        tuple_var_assignment_374364_375413 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 268, 8), 'tuple_var_assignment_374364')
        # Assigning a type to the variable 'x' (line 268)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 268, 8), 'x', tuple_var_assignment_374364_375413)
        
        # Assigning a Name to a Name (line 268):
        # Getting the type of 'tuple_var_assignment_374365' (line 268)
        tuple_var_assignment_374365_375414 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 268, 8), 'tuple_var_assignment_374365')
        # Assigning a type to the variable '_' (line 268)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 268, 11), '_', tuple_var_assignment_374365_375414)
        
        
        # Getting the type of 'x' (line 270)
        x_375415 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 270, 11), 'x')
        # Obtaining the member 'shape' of a type (line 270)
        shape_375416 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 270, 11), x_375415, 'shape')
        # Getting the type of 'i' (line 270)
        i_375417 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 270, 22), 'i')
        # Obtaining the member 'shape' of a type (line 270)
        shape_375418 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 270, 22), i_375417, 'shape')
        # Applying the binary operator '!=' (line 270)
        result_ne_375419 = python_operator(stypy.reporting.localization.Localization(__file__, 270, 11), '!=', shape_375416, shape_375418)
        
        # Testing the type of an if condition (line 270)
        if_condition_375420 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 270, 8), result_ne_375419)
        # Assigning a type to the variable 'if_condition_375420' (line 270)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 270, 8), 'if_condition_375420', if_condition_375420)
        # SSA begins for if statement (line 270)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to ValueError(...): (line 271)
        # Processing the call arguments (line 271)
        str_375422 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 271, 29), 'str', 'Shape mismatch in assignment.')
        # Processing the call keyword arguments (line 271)
        kwargs_375423 = {}
        # Getting the type of 'ValueError' (line 271)
        ValueError_375421 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 271, 18), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 271)
        ValueError_call_result_375424 = invoke(stypy.reporting.localization.Localization(__file__, 271, 18), ValueError_375421, *[str_375422], **kwargs_375423)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 271, 12), ValueError_call_result_375424, 'raise parameter', BaseException)
        # SSA join for if statement (line 270)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        
        # Call to size(...): (line 273)
        # Processing the call arguments (line 273)
        # Getting the type of 'x' (line 273)
        x_375427 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 273, 19), 'x', False)
        # Processing the call keyword arguments (line 273)
        kwargs_375428 = {}
        # Getting the type of 'np' (line 273)
        np_375425 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 273, 11), 'np', False)
        # Obtaining the member 'size' of a type (line 273)
        size_375426 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 273, 11), np_375425, 'size')
        # Calling size(args, kwargs) (line 273)
        size_call_result_375429 = invoke(stypy.reporting.localization.Localization(__file__, 273, 11), size_375426, *[x_375427], **kwargs_375428)
        
        int_375430 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 273, 25), 'int')
        # Applying the binary operator '==' (line 273)
        result_eq_375431 = python_operator(stypy.reporting.localization.Localization(__file__, 273, 11), '==', size_call_result_375429, int_375430)
        
        # Testing the type of an if condition (line 273)
        if_condition_375432 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 273, 8), result_eq_375431)
        # Assigning a type to the variable 'if_condition_375432' (line 273)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 273, 8), 'if_condition_375432', if_condition_375432)
        # SSA begins for if statement (line 273)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # Assigning a type to the variable 'stypy_return_type' (line 274)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 274, 12), 'stypy_return_type', types.NoneType)
        # SSA join for if statement (line 273)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Name (line 276):
        
        # Assigning a Call to a Name (line 276):
        
        # Call to min(...): (line 276)
        # Processing the call keyword arguments (line 276)
        kwargs_375435 = {}
        # Getting the type of 'i' (line 276)
        i_375433 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 276, 16), 'i', False)
        # Obtaining the member 'min' of a type (line 276)
        min_375434 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 276, 16), i_375433, 'min')
        # Calling min(args, kwargs) (line 276)
        min_call_result_375436 = invoke(stypy.reporting.localization.Localization(__file__, 276, 16), min_375434, *[], **kwargs_375435)
        
        # Assigning a type to the variable 'min_i' (line 276)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 276, 8), 'min_i', min_call_result_375436)
        
        
        # Evaluating a boolean operation
        
        # Getting the type of 'min_i' (line 277)
        min_i_375437 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 277, 11), 'min_i')
        
        
        # Obtaining the type of the subscript
        int_375438 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 277, 31), 'int')
        # Getting the type of 'self' (line 277)
        self_375439 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 277, 20), 'self')
        # Obtaining the member 'shape' of a type (line 277)
        shape_375440 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 277, 20), self_375439, 'shape')
        # Obtaining the member '__getitem__' of a type (line 277)
        getitem___375441 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 277, 20), shape_375440, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 277)
        subscript_call_result_375442 = invoke(stypy.reporting.localization.Localization(__file__, 277, 20), getitem___375441, int_375438)
        
        # Applying the 'usub' unary operator (line 277)
        result___neg___375443 = python_operator(stypy.reporting.localization.Localization(__file__, 277, 19), 'usub', subscript_call_result_375442)
        
        # Applying the binary operator '<' (line 277)
        result_lt_375444 = python_operator(stypy.reporting.localization.Localization(__file__, 277, 11), '<', min_i_375437, result___neg___375443)
        
        
        
        # Call to max(...): (line 277)
        # Processing the call keyword arguments (line 277)
        kwargs_375447 = {}
        # Getting the type of 'i' (line 277)
        i_375445 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 277, 37), 'i', False)
        # Obtaining the member 'max' of a type (line 277)
        max_375446 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 277, 37), i_375445, 'max')
        # Calling max(args, kwargs) (line 277)
        max_call_result_375448 = invoke(stypy.reporting.localization.Localization(__file__, 277, 37), max_375446, *[], **kwargs_375447)
        
        
        # Obtaining the type of the subscript
        int_375449 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 277, 59), 'int')
        # Getting the type of 'self' (line 277)
        self_375450 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 277, 48), 'self')
        # Obtaining the member 'shape' of a type (line 277)
        shape_375451 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 277, 48), self_375450, 'shape')
        # Obtaining the member '__getitem__' of a type (line 277)
        getitem___375452 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 277, 48), shape_375451, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 277)
        subscript_call_result_375453 = invoke(stypy.reporting.localization.Localization(__file__, 277, 48), getitem___375452, int_375449)
        
        # Applying the binary operator '>=' (line 277)
        result_ge_375454 = python_operator(stypy.reporting.localization.Localization(__file__, 277, 37), '>=', max_call_result_375448, subscript_call_result_375453)
        
        # Applying the binary operator 'or' (line 277)
        result_or_keyword_375455 = python_operator(stypy.reporting.localization.Localization(__file__, 277, 11), 'or', result_lt_375444, result_ge_375454)
        
        # Testing the type of an if condition (line 277)
        if_condition_375456 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 277, 8), result_or_keyword_375455)
        # Assigning a type to the variable 'if_condition_375456' (line 277)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 277, 8), 'if_condition_375456', if_condition_375456)
        # SSA begins for if statement (line 277)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to IndexError(...): (line 278)
        # Processing the call arguments (line 278)
        str_375458 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 278, 29), 'str', 'Index (%d) out of range -%d to %d.')
        
        # Obtaining an instance of the builtin type 'tuple' (line 279)
        tuple_375459 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 279, 30), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 279)
        # Adding element type (line 279)
        
        # Call to min(...): (line 279)
        # Processing the call keyword arguments (line 279)
        kwargs_375462 = {}
        # Getting the type of 'i' (line 279)
        i_375460 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 279, 30), 'i', False)
        # Obtaining the member 'min' of a type (line 279)
        min_375461 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 279, 30), i_375460, 'min')
        # Calling min(args, kwargs) (line 279)
        min_call_result_375463 = invoke(stypy.reporting.localization.Localization(__file__, 279, 30), min_375461, *[], **kwargs_375462)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 279, 30), tuple_375459, min_call_result_375463)
        # Adding element type (line 279)
        
        # Obtaining the type of the subscript
        int_375464 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 279, 50), 'int')
        # Getting the type of 'self' (line 279)
        self_375465 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 279, 39), 'self', False)
        # Obtaining the member 'shape' of a type (line 279)
        shape_375466 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 279, 39), self_375465, 'shape')
        # Obtaining the member '__getitem__' of a type (line 279)
        getitem___375467 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 279, 39), shape_375466, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 279)
        subscript_call_result_375468 = invoke(stypy.reporting.localization.Localization(__file__, 279, 39), getitem___375467, int_375464)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 279, 30), tuple_375459, subscript_call_result_375468)
        # Adding element type (line 279)
        
        # Obtaining the type of the subscript
        int_375469 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 279, 65), 'int')
        # Getting the type of 'self' (line 279)
        self_375470 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 279, 54), 'self', False)
        # Obtaining the member 'shape' of a type (line 279)
        shape_375471 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 279, 54), self_375470, 'shape')
        # Obtaining the member '__getitem__' of a type (line 279)
        getitem___375472 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 279, 54), shape_375471, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 279)
        subscript_call_result_375473 = invoke(stypy.reporting.localization.Localization(__file__, 279, 54), getitem___375472, int_375469)
        
        int_375474 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 279, 68), 'int')
        # Applying the binary operator '-' (line 279)
        result_sub_375475 = python_operator(stypy.reporting.localization.Localization(__file__, 279, 54), '-', subscript_call_result_375473, int_375474)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 279, 30), tuple_375459, result_sub_375475)
        
        # Applying the binary operator '%' (line 278)
        result_mod_375476 = python_operator(stypy.reporting.localization.Localization(__file__, 278, 29), '%', str_375458, tuple_375459)
        
        # Processing the call keyword arguments (line 278)
        kwargs_375477 = {}
        # Getting the type of 'IndexError' (line 278)
        IndexError_375457 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 278, 18), 'IndexError', False)
        # Calling IndexError(args, kwargs) (line 278)
        IndexError_call_result_375478 = invoke(stypy.reporting.localization.Localization(__file__, 278, 18), IndexError_375457, *[result_mod_375476], **kwargs_375477)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 278, 12), IndexError_call_result_375478, 'raise parameter', BaseException)
        # SSA join for if statement (line 277)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Getting the type of 'min_i' (line 280)
        min_i_375479 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 280, 11), 'min_i')
        int_375480 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 280, 19), 'int')
        # Applying the binary operator '<' (line 280)
        result_lt_375481 = python_operator(stypy.reporting.localization.Localization(__file__, 280, 11), '<', min_i_375479, int_375480)
        
        # Testing the type of an if condition (line 280)
        if_condition_375482 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 280, 8), result_lt_375481)
        # Assigning a type to the variable 'if_condition_375482' (line 280)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 280, 8), 'if_condition_375482', if_condition_375482)
        # SSA begins for if statement (line 280)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 281):
        
        # Assigning a Call to a Name (line 281):
        
        # Call to copy(...): (line 281)
        # Processing the call keyword arguments (line 281)
        kwargs_375485 = {}
        # Getting the type of 'i' (line 281)
        i_375483 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 281, 16), 'i', False)
        # Obtaining the member 'copy' of a type (line 281)
        copy_375484 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 281, 16), i_375483, 'copy')
        # Calling copy(args, kwargs) (line 281)
        copy_call_result_375486 = invoke(stypy.reporting.localization.Localization(__file__, 281, 16), copy_375484, *[], **kwargs_375485)
        
        # Assigning a type to the variable 'i' (line 281)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 281, 12), 'i', copy_call_result_375486)
        
        # Getting the type of 'i' (line 282)
        i_375487 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 282, 12), 'i')
        
        # Obtaining the type of the subscript
        
        # Getting the type of 'i' (line 282)
        i_375488 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 282, 14), 'i')
        int_375489 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 282, 18), 'int')
        # Applying the binary operator '<' (line 282)
        result_lt_375490 = python_operator(stypy.reporting.localization.Localization(__file__, 282, 14), '<', i_375488, int_375489)
        
        # Getting the type of 'i' (line 282)
        i_375491 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 282, 12), 'i')
        # Obtaining the member '__getitem__' of a type (line 282)
        getitem___375492 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 282, 12), i_375491, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 282)
        subscript_call_result_375493 = invoke(stypy.reporting.localization.Localization(__file__, 282, 12), getitem___375492, result_lt_375490)
        
        
        # Obtaining the type of the subscript
        int_375494 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 282, 35), 'int')
        # Getting the type of 'self' (line 282)
        self_375495 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 282, 24), 'self')
        # Obtaining the member 'shape' of a type (line 282)
        shape_375496 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 282, 24), self_375495, 'shape')
        # Obtaining the member '__getitem__' of a type (line 282)
        getitem___375497 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 282, 24), shape_375496, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 282)
        subscript_call_result_375498 = invoke(stypy.reporting.localization.Localization(__file__, 282, 24), getitem___375497, int_375494)
        
        # Applying the binary operator '+=' (line 282)
        result_iadd_375499 = python_operator(stypy.reporting.localization.Localization(__file__, 282, 12), '+=', subscript_call_result_375493, subscript_call_result_375498)
        # Getting the type of 'i' (line 282)
        i_375500 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 282, 12), 'i')
        
        # Getting the type of 'i' (line 282)
        i_375501 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 282, 14), 'i')
        int_375502 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 282, 18), 'int')
        # Applying the binary operator '<' (line 282)
        result_lt_375503 = python_operator(stypy.reporting.localization.Localization(__file__, 282, 14), '<', i_375501, int_375502)
        
        # Storing an element on a container (line 282)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 282, 12), i_375500, (result_lt_375503, result_iadd_375499))
        
        # SSA join for if statement (line 280)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Name (line 284):
        
        # Assigning a Call to a Name (line 284):
        
        # Call to min(...): (line 284)
        # Processing the call keyword arguments (line 284)
        kwargs_375506 = {}
        # Getting the type of 'j' (line 284)
        j_375504 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 284, 16), 'j', False)
        # Obtaining the member 'min' of a type (line 284)
        min_375505 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 284, 16), j_375504, 'min')
        # Calling min(args, kwargs) (line 284)
        min_call_result_375507 = invoke(stypy.reporting.localization.Localization(__file__, 284, 16), min_375505, *[], **kwargs_375506)
        
        # Assigning a type to the variable 'min_j' (line 284)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 284, 8), 'min_j', min_call_result_375507)
        
        
        # Evaluating a boolean operation
        
        # Getting the type of 'min_j' (line 285)
        min_j_375508 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 285, 11), 'min_j')
        
        
        # Obtaining the type of the subscript
        int_375509 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 285, 31), 'int')
        # Getting the type of 'self' (line 285)
        self_375510 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 285, 20), 'self')
        # Obtaining the member 'shape' of a type (line 285)
        shape_375511 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 285, 20), self_375510, 'shape')
        # Obtaining the member '__getitem__' of a type (line 285)
        getitem___375512 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 285, 20), shape_375511, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 285)
        subscript_call_result_375513 = invoke(stypy.reporting.localization.Localization(__file__, 285, 20), getitem___375512, int_375509)
        
        # Applying the 'usub' unary operator (line 285)
        result___neg___375514 = python_operator(stypy.reporting.localization.Localization(__file__, 285, 19), 'usub', subscript_call_result_375513)
        
        # Applying the binary operator '<' (line 285)
        result_lt_375515 = python_operator(stypy.reporting.localization.Localization(__file__, 285, 11), '<', min_j_375508, result___neg___375514)
        
        
        
        # Call to max(...): (line 285)
        # Processing the call keyword arguments (line 285)
        kwargs_375518 = {}
        # Getting the type of 'j' (line 285)
        j_375516 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 285, 37), 'j', False)
        # Obtaining the member 'max' of a type (line 285)
        max_375517 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 285, 37), j_375516, 'max')
        # Calling max(args, kwargs) (line 285)
        max_call_result_375519 = invoke(stypy.reporting.localization.Localization(__file__, 285, 37), max_375517, *[], **kwargs_375518)
        
        
        # Obtaining the type of the subscript
        int_375520 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 285, 59), 'int')
        # Getting the type of 'self' (line 285)
        self_375521 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 285, 48), 'self')
        # Obtaining the member 'shape' of a type (line 285)
        shape_375522 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 285, 48), self_375521, 'shape')
        # Obtaining the member '__getitem__' of a type (line 285)
        getitem___375523 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 285, 48), shape_375522, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 285)
        subscript_call_result_375524 = invoke(stypy.reporting.localization.Localization(__file__, 285, 48), getitem___375523, int_375520)
        
        # Applying the binary operator '>=' (line 285)
        result_ge_375525 = python_operator(stypy.reporting.localization.Localization(__file__, 285, 37), '>=', max_call_result_375519, subscript_call_result_375524)
        
        # Applying the binary operator 'or' (line 285)
        result_or_keyword_375526 = python_operator(stypy.reporting.localization.Localization(__file__, 285, 11), 'or', result_lt_375515, result_ge_375525)
        
        # Testing the type of an if condition (line 285)
        if_condition_375527 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 285, 8), result_or_keyword_375526)
        # Assigning a type to the variable 'if_condition_375527' (line 285)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 285, 8), 'if_condition_375527', if_condition_375527)
        # SSA begins for if statement (line 285)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to IndexError(...): (line 286)
        # Processing the call arguments (line 286)
        str_375529 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 286, 29), 'str', 'Index (%d) out of range -%d to %d.')
        
        # Obtaining an instance of the builtin type 'tuple' (line 287)
        tuple_375530 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 287, 30), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 287)
        # Adding element type (line 287)
        
        # Call to min(...): (line 287)
        # Processing the call keyword arguments (line 287)
        kwargs_375533 = {}
        # Getting the type of 'j' (line 287)
        j_375531 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 287, 30), 'j', False)
        # Obtaining the member 'min' of a type (line 287)
        min_375532 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 287, 30), j_375531, 'min')
        # Calling min(args, kwargs) (line 287)
        min_call_result_375534 = invoke(stypy.reporting.localization.Localization(__file__, 287, 30), min_375532, *[], **kwargs_375533)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 287, 30), tuple_375530, min_call_result_375534)
        # Adding element type (line 287)
        
        # Obtaining the type of the subscript
        int_375535 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 287, 50), 'int')
        # Getting the type of 'self' (line 287)
        self_375536 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 287, 39), 'self', False)
        # Obtaining the member 'shape' of a type (line 287)
        shape_375537 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 287, 39), self_375536, 'shape')
        # Obtaining the member '__getitem__' of a type (line 287)
        getitem___375538 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 287, 39), shape_375537, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 287)
        subscript_call_result_375539 = invoke(stypy.reporting.localization.Localization(__file__, 287, 39), getitem___375538, int_375535)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 287, 30), tuple_375530, subscript_call_result_375539)
        # Adding element type (line 287)
        
        # Obtaining the type of the subscript
        int_375540 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 287, 65), 'int')
        # Getting the type of 'self' (line 287)
        self_375541 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 287, 54), 'self', False)
        # Obtaining the member 'shape' of a type (line 287)
        shape_375542 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 287, 54), self_375541, 'shape')
        # Obtaining the member '__getitem__' of a type (line 287)
        getitem___375543 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 287, 54), shape_375542, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 287)
        subscript_call_result_375544 = invoke(stypy.reporting.localization.Localization(__file__, 287, 54), getitem___375543, int_375540)
        
        int_375545 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 287, 68), 'int')
        # Applying the binary operator '-' (line 287)
        result_sub_375546 = python_operator(stypy.reporting.localization.Localization(__file__, 287, 54), '-', subscript_call_result_375544, int_375545)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 287, 30), tuple_375530, result_sub_375546)
        
        # Applying the binary operator '%' (line 286)
        result_mod_375547 = python_operator(stypy.reporting.localization.Localization(__file__, 286, 29), '%', str_375529, tuple_375530)
        
        # Processing the call keyword arguments (line 286)
        kwargs_375548 = {}
        # Getting the type of 'IndexError' (line 286)
        IndexError_375528 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 286, 18), 'IndexError', False)
        # Calling IndexError(args, kwargs) (line 286)
        IndexError_call_result_375549 = invoke(stypy.reporting.localization.Localization(__file__, 286, 18), IndexError_375528, *[result_mod_375547], **kwargs_375548)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 286, 12), IndexError_call_result_375549, 'raise parameter', BaseException)
        # SSA join for if statement (line 285)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Getting the type of 'min_j' (line 288)
        min_j_375550 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 288, 11), 'min_j')
        int_375551 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 288, 19), 'int')
        # Applying the binary operator '<' (line 288)
        result_lt_375552 = python_operator(stypy.reporting.localization.Localization(__file__, 288, 11), '<', min_j_375550, int_375551)
        
        # Testing the type of an if condition (line 288)
        if_condition_375553 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 288, 8), result_lt_375552)
        # Assigning a type to the variable 'if_condition_375553' (line 288)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 288, 8), 'if_condition_375553', if_condition_375553)
        # SSA begins for if statement (line 288)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 289):
        
        # Assigning a Call to a Name (line 289):
        
        # Call to copy(...): (line 289)
        # Processing the call keyword arguments (line 289)
        kwargs_375556 = {}
        # Getting the type of 'j' (line 289)
        j_375554 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 289, 16), 'j', False)
        # Obtaining the member 'copy' of a type (line 289)
        copy_375555 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 289, 16), j_375554, 'copy')
        # Calling copy(args, kwargs) (line 289)
        copy_call_result_375557 = invoke(stypy.reporting.localization.Localization(__file__, 289, 16), copy_375555, *[], **kwargs_375556)
        
        # Assigning a type to the variable 'j' (line 289)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 289, 12), 'j', copy_call_result_375557)
        
        # Getting the type of 'j' (line 290)
        j_375558 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 290, 12), 'j')
        
        # Obtaining the type of the subscript
        
        # Getting the type of 'j' (line 290)
        j_375559 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 290, 14), 'j')
        int_375560 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 290, 18), 'int')
        # Applying the binary operator '<' (line 290)
        result_lt_375561 = python_operator(stypy.reporting.localization.Localization(__file__, 290, 14), '<', j_375559, int_375560)
        
        # Getting the type of 'j' (line 290)
        j_375562 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 290, 12), 'j')
        # Obtaining the member '__getitem__' of a type (line 290)
        getitem___375563 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 290, 12), j_375562, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 290)
        subscript_call_result_375564 = invoke(stypy.reporting.localization.Localization(__file__, 290, 12), getitem___375563, result_lt_375561)
        
        
        # Obtaining the type of the subscript
        int_375565 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 290, 35), 'int')
        # Getting the type of 'self' (line 290)
        self_375566 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 290, 24), 'self')
        # Obtaining the member 'shape' of a type (line 290)
        shape_375567 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 290, 24), self_375566, 'shape')
        # Obtaining the member '__getitem__' of a type (line 290)
        getitem___375568 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 290, 24), shape_375567, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 290)
        subscript_call_result_375569 = invoke(stypy.reporting.localization.Localization(__file__, 290, 24), getitem___375568, int_375565)
        
        # Applying the binary operator '+=' (line 290)
        result_iadd_375570 = python_operator(stypy.reporting.localization.Localization(__file__, 290, 12), '+=', subscript_call_result_375564, subscript_call_result_375569)
        # Getting the type of 'j' (line 290)
        j_375571 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 290, 12), 'j')
        
        # Getting the type of 'j' (line 290)
        j_375572 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 290, 14), 'j')
        int_375573 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 290, 18), 'int')
        # Applying the binary operator '<' (line 290)
        result_lt_375574 = python_operator(stypy.reporting.localization.Localization(__file__, 290, 14), '<', j_375572, int_375573)
        
        # Storing an element on a container (line 290)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 290, 12), j_375571, (result_lt_375574, result_iadd_375570))
        
        # SSA join for if statement (line 288)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to update(...): (line 292)
        # Processing the call arguments (line 292)
        # Getting the type of 'self' (line 292)
        self_375577 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 292, 20), 'self', False)
        
        # Call to izip(...): (line 292)
        # Processing the call arguments (line 292)
        
        # Call to izip(...): (line 292)
        # Processing the call arguments (line 292)
        # Getting the type of 'i' (line 292)
        i_375580 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 292, 36), 'i', False)
        # Obtaining the member 'flat' of a type (line 292)
        flat_375581 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 292, 36), i_375580, 'flat')
        # Getting the type of 'j' (line 292)
        j_375582 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 292, 44), 'j', False)
        # Obtaining the member 'flat' of a type (line 292)
        flat_375583 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 292, 44), j_375582, 'flat')
        # Processing the call keyword arguments (line 292)
        kwargs_375584 = {}
        # Getting the type of 'izip' (line 292)
        izip_375579 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 292, 31), 'izip', False)
        # Calling izip(args, kwargs) (line 292)
        izip_call_result_375585 = invoke(stypy.reporting.localization.Localization(__file__, 292, 31), izip_375579, *[flat_375581, flat_375583], **kwargs_375584)
        
        # Getting the type of 'x' (line 292)
        x_375586 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 292, 53), 'x', False)
        # Obtaining the member 'flat' of a type (line 292)
        flat_375587 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 292, 53), x_375586, 'flat')
        # Processing the call keyword arguments (line 292)
        kwargs_375588 = {}
        # Getting the type of 'izip' (line 292)
        izip_375578 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 292, 26), 'izip', False)
        # Calling izip(args, kwargs) (line 292)
        izip_call_result_375589 = invoke(stypy.reporting.localization.Localization(__file__, 292, 26), izip_375578, *[izip_call_result_375585, flat_375587], **kwargs_375588)
        
        # Processing the call keyword arguments (line 292)
        kwargs_375590 = {}
        # Getting the type of 'dict' (line 292)
        dict_375575 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 292, 8), 'dict', False)
        # Obtaining the member 'update' of a type (line 292)
        update_375576 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 292, 8), dict_375575, 'update')
        # Calling update(args, kwargs) (line 292)
        update_call_result_375591 = invoke(stypy.reporting.localization.Localization(__file__, 292, 8), update_375576, *[self_375577, izip_call_result_375589], **kwargs_375590)
        
        
        
        int_375592 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 294, 11), 'int')
        # Getting the type of 'x' (line 294)
        x_375593 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 294, 16), 'x')
        # Applying the binary operator 'in' (line 294)
        result_contains_375594 = python_operator(stypy.reporting.localization.Localization(__file__, 294, 11), 'in', int_375592, x_375593)
        
        # Testing the type of an if condition (line 294)
        if_condition_375595 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 294, 8), result_contains_375594)
        # Assigning a type to the variable 'if_condition_375595' (line 294)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 294, 8), 'if_condition_375595', if_condition_375595)
        # SSA begins for if statement (line 294)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Compare to a Name (line 295):
        
        # Assigning a Compare to a Name (line 295):
        
        # Getting the type of 'x' (line 295)
        x_375596 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 295, 21), 'x')
        int_375597 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 295, 26), 'int')
        # Applying the binary operator '==' (line 295)
        result_eq_375598 = python_operator(stypy.reporting.localization.Localization(__file__, 295, 21), '==', x_375596, int_375597)
        
        # Assigning a type to the variable 'zeroes' (line 295)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 295, 12), 'zeroes', result_eq_375598)
        
        
        # Call to izip(...): (line 296)
        # Processing the call arguments (line 296)
        
        # Obtaining the type of the subscript
        # Getting the type of 'zeroes' (line 296)
        zeroes_375600 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 296, 30), 'zeroes', False)
        # Getting the type of 'i' (line 296)
        i_375601 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 296, 28), 'i', False)
        # Obtaining the member '__getitem__' of a type (line 296)
        getitem___375602 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 296, 28), i_375601, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 296)
        subscript_call_result_375603 = invoke(stypy.reporting.localization.Localization(__file__, 296, 28), getitem___375602, zeroes_375600)
        
        # Obtaining the member 'flat' of a type (line 296)
        flat_375604 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 296, 28), subscript_call_result_375603, 'flat')
        
        # Obtaining the type of the subscript
        # Getting the type of 'zeroes' (line 296)
        zeroes_375605 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 296, 46), 'zeroes', False)
        # Getting the type of 'j' (line 296)
        j_375606 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 296, 44), 'j', False)
        # Obtaining the member '__getitem__' of a type (line 296)
        getitem___375607 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 296, 44), j_375606, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 296)
        subscript_call_result_375608 = invoke(stypy.reporting.localization.Localization(__file__, 296, 44), getitem___375607, zeroes_375605)
        
        # Obtaining the member 'flat' of a type (line 296)
        flat_375609 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 296, 44), subscript_call_result_375608, 'flat')
        # Processing the call keyword arguments (line 296)
        kwargs_375610 = {}
        # Getting the type of 'izip' (line 296)
        izip_375599 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 296, 23), 'izip', False)
        # Calling izip(args, kwargs) (line 296)
        izip_call_result_375611 = invoke(stypy.reporting.localization.Localization(__file__, 296, 23), izip_375599, *[flat_375604, flat_375609], **kwargs_375610)
        
        # Testing the type of a for loop iterable (line 296)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 296, 12), izip_call_result_375611)
        # Getting the type of the for loop variable (line 296)
        for_loop_var_375612 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 296, 12), izip_call_result_375611)
        # Assigning a type to the variable 'key' (line 296)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 296, 12), 'key', for_loop_var_375612)
        # SSA begins for a for statement (line 296)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        
        
        # Call to __getitem__(...): (line 297)
        # Processing the call arguments (line 297)
        # Getting the type of 'self' (line 297)
        self_375615 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 297, 36), 'self', False)
        # Getting the type of 'key' (line 297)
        key_375616 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 297, 42), 'key', False)
        # Processing the call keyword arguments (line 297)
        kwargs_375617 = {}
        # Getting the type of 'dict' (line 297)
        dict_375613 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 297, 19), 'dict', False)
        # Obtaining the member '__getitem__' of a type (line 297)
        getitem___375614 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 297, 19), dict_375613, '__getitem__')
        # Calling __getitem__(args, kwargs) (line 297)
        getitem___call_result_375618 = invoke(stypy.reporting.localization.Localization(__file__, 297, 19), getitem___375614, *[self_375615, key_375616], **kwargs_375617)
        
        int_375619 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 297, 50), 'int')
        # Applying the binary operator '==' (line 297)
        result_eq_375620 = python_operator(stypy.reporting.localization.Localization(__file__, 297, 19), '==', getitem___call_result_375618, int_375619)
        
        # Testing the type of an if condition (line 297)
        if_condition_375621 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 297, 16), result_eq_375620)
        # Assigning a type to the variable 'if_condition_375621' (line 297)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 297, 16), 'if_condition_375621', if_condition_375621)
        # SSA begins for if statement (line 297)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # Deleting a member
        # Getting the type of 'self' (line 299)
        self_375622 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 299, 24), 'self')
        
        # Obtaining the type of the subscript
        # Getting the type of 'key' (line 299)
        key_375623 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 299, 29), 'key')
        # Getting the type of 'self' (line 299)
        self_375624 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 299, 24), 'self')
        # Obtaining the member '__getitem__' of a type (line 299)
        getitem___375625 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 299, 24), self_375624, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 299)
        subscript_call_result_375626 = invoke(stypy.reporting.localization.Localization(__file__, 299, 24), getitem___375625, key_375623)
        
        del_contained_elements_type(stypy.reporting.localization.Localization(__file__, 299, 20), self_375622, subscript_call_result_375626)
        # SSA join for if statement (line 297)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 294)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of '__setitem__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__setitem__' in the type store
        # Getting the type of 'stypy_return_type' (line 249)
        stypy_return_type_375627 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 249, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_375627)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__setitem__'
        return stypy_return_type_375627


    @norecursion
    def __add__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__add__'
        module_type_store = module_type_store.open_function_context('__add__', 301, 4, False)
        # Assigning a type to the variable 'self' (line 302)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 302, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        dok_matrix.__add__.__dict__.__setitem__('stypy_localization', localization)
        dok_matrix.__add__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        dok_matrix.__add__.__dict__.__setitem__('stypy_type_store', module_type_store)
        dok_matrix.__add__.__dict__.__setitem__('stypy_function_name', 'dok_matrix.__add__')
        dok_matrix.__add__.__dict__.__setitem__('stypy_param_names_list', ['other'])
        dok_matrix.__add__.__dict__.__setitem__('stypy_varargs_param_name', None)
        dok_matrix.__add__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        dok_matrix.__add__.__dict__.__setitem__('stypy_call_defaults', defaults)
        dok_matrix.__add__.__dict__.__setitem__('stypy_call_varargs', varargs)
        dok_matrix.__add__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        dok_matrix.__add__.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'dok_matrix.__add__', ['other'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__add__', localization, ['other'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__add__(...)' code ##################

        
        
        # Call to isscalarlike(...): (line 302)
        # Processing the call arguments (line 302)
        # Getting the type of 'other' (line 302)
        other_375629 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 302, 24), 'other', False)
        # Processing the call keyword arguments (line 302)
        kwargs_375630 = {}
        # Getting the type of 'isscalarlike' (line 302)
        isscalarlike_375628 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 302, 11), 'isscalarlike', False)
        # Calling isscalarlike(args, kwargs) (line 302)
        isscalarlike_call_result_375631 = invoke(stypy.reporting.localization.Localization(__file__, 302, 11), isscalarlike_375628, *[other_375629], **kwargs_375630)
        
        # Testing the type of an if condition (line 302)
        if_condition_375632 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 302, 8), isscalarlike_call_result_375631)
        # Assigning a type to the variable 'if_condition_375632' (line 302)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 302, 8), 'if_condition_375632', if_condition_375632)
        # SSA begins for if statement (line 302)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 303):
        
        # Assigning a Call to a Name (line 303):
        
        # Call to upcast_scalar(...): (line 303)
        # Processing the call arguments (line 303)
        # Getting the type of 'self' (line 303)
        self_375634 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 303, 38), 'self', False)
        # Obtaining the member 'dtype' of a type (line 303)
        dtype_375635 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 303, 38), self_375634, 'dtype')
        # Getting the type of 'other' (line 303)
        other_375636 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 303, 50), 'other', False)
        # Processing the call keyword arguments (line 303)
        kwargs_375637 = {}
        # Getting the type of 'upcast_scalar' (line 303)
        upcast_scalar_375633 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 303, 24), 'upcast_scalar', False)
        # Calling upcast_scalar(args, kwargs) (line 303)
        upcast_scalar_call_result_375638 = invoke(stypy.reporting.localization.Localization(__file__, 303, 24), upcast_scalar_375633, *[dtype_375635, other_375636], **kwargs_375637)
        
        # Assigning a type to the variable 'res_dtype' (line 303)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 303, 12), 'res_dtype', upcast_scalar_call_result_375638)
        
        # Assigning a Call to a Name (line 304):
        
        # Assigning a Call to a Name (line 304):
        
        # Call to dok_matrix(...): (line 304)
        # Processing the call arguments (line 304)
        # Getting the type of 'self' (line 304)
        self_375640 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 304, 29), 'self', False)
        # Obtaining the member 'shape' of a type (line 304)
        shape_375641 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 304, 29), self_375640, 'shape')
        # Processing the call keyword arguments (line 304)
        # Getting the type of 'res_dtype' (line 304)
        res_dtype_375642 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 304, 47), 'res_dtype', False)
        keyword_375643 = res_dtype_375642
        kwargs_375644 = {'dtype': keyword_375643}
        # Getting the type of 'dok_matrix' (line 304)
        dok_matrix_375639 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 304, 18), 'dok_matrix', False)
        # Calling dok_matrix(args, kwargs) (line 304)
        dok_matrix_call_result_375645 = invoke(stypy.reporting.localization.Localization(__file__, 304, 18), dok_matrix_375639, *[shape_375641], **kwargs_375644)
        
        # Assigning a type to the variable 'new' (line 304)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 304, 12), 'new', dok_matrix_call_result_375645)
        
        # Assigning a Attribute to a Tuple (line 306):
        
        # Assigning a Subscript to a Name (line 306):
        
        # Obtaining the type of the subscript
        int_375646 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 306, 12), 'int')
        # Getting the type of 'self' (line 306)
        self_375647 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 306, 19), 'self')
        # Obtaining the member 'shape' of a type (line 306)
        shape_375648 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 306, 19), self_375647, 'shape')
        # Obtaining the member '__getitem__' of a type (line 306)
        getitem___375649 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 306, 12), shape_375648, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 306)
        subscript_call_result_375650 = invoke(stypy.reporting.localization.Localization(__file__, 306, 12), getitem___375649, int_375646)
        
        # Assigning a type to the variable 'tuple_var_assignment_374366' (line 306)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 306, 12), 'tuple_var_assignment_374366', subscript_call_result_375650)
        
        # Assigning a Subscript to a Name (line 306):
        
        # Obtaining the type of the subscript
        int_375651 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 306, 12), 'int')
        # Getting the type of 'self' (line 306)
        self_375652 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 306, 19), 'self')
        # Obtaining the member 'shape' of a type (line 306)
        shape_375653 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 306, 19), self_375652, 'shape')
        # Obtaining the member '__getitem__' of a type (line 306)
        getitem___375654 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 306, 12), shape_375653, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 306)
        subscript_call_result_375655 = invoke(stypy.reporting.localization.Localization(__file__, 306, 12), getitem___375654, int_375651)
        
        # Assigning a type to the variable 'tuple_var_assignment_374367' (line 306)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 306, 12), 'tuple_var_assignment_374367', subscript_call_result_375655)
        
        # Assigning a Name to a Name (line 306):
        # Getting the type of 'tuple_var_assignment_374366' (line 306)
        tuple_var_assignment_374366_375656 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 306, 12), 'tuple_var_assignment_374366')
        # Assigning a type to the variable 'M' (line 306)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 306, 12), 'M', tuple_var_assignment_374366_375656)
        
        # Assigning a Name to a Name (line 306):
        # Getting the type of 'tuple_var_assignment_374367' (line 306)
        tuple_var_assignment_374367_375657 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 306, 12), 'tuple_var_assignment_374367')
        # Assigning a type to the variable 'N' (line 306)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 306, 15), 'N', tuple_var_assignment_374367_375657)
        
        
        # Call to product(...): (line 307)
        # Processing the call arguments (line 307)
        
        # Call to xrange(...): (line 307)
        # Processing the call arguments (line 307)
        # Getting the type of 'M' (line 307)
        M_375661 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 307, 48), 'M', False)
        # Processing the call keyword arguments (line 307)
        kwargs_375662 = {}
        # Getting the type of 'xrange' (line 307)
        xrange_375660 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 307, 41), 'xrange', False)
        # Calling xrange(args, kwargs) (line 307)
        xrange_call_result_375663 = invoke(stypy.reporting.localization.Localization(__file__, 307, 41), xrange_375660, *[M_375661], **kwargs_375662)
        
        
        # Call to xrange(...): (line 307)
        # Processing the call arguments (line 307)
        # Getting the type of 'N' (line 307)
        N_375665 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 307, 59), 'N', False)
        # Processing the call keyword arguments (line 307)
        kwargs_375666 = {}
        # Getting the type of 'xrange' (line 307)
        xrange_375664 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 307, 52), 'xrange', False)
        # Calling xrange(args, kwargs) (line 307)
        xrange_call_result_375667 = invoke(stypy.reporting.localization.Localization(__file__, 307, 52), xrange_375664, *[N_375665], **kwargs_375666)
        
        # Processing the call keyword arguments (line 307)
        kwargs_375668 = {}
        # Getting the type of 'itertools' (line 307)
        itertools_375658 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 307, 23), 'itertools', False)
        # Obtaining the member 'product' of a type (line 307)
        product_375659 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 307, 23), itertools_375658, 'product')
        # Calling product(args, kwargs) (line 307)
        product_call_result_375669 = invoke(stypy.reporting.localization.Localization(__file__, 307, 23), product_375659, *[xrange_call_result_375663, xrange_call_result_375667], **kwargs_375668)
        
        # Testing the type of a for loop iterable (line 307)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 307, 12), product_call_result_375669)
        # Getting the type of the for loop variable (line 307)
        for_loop_var_375670 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 307, 12), product_call_result_375669)
        # Assigning a type to the variable 'key' (line 307)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 307, 12), 'key', for_loop_var_375670)
        # SSA begins for a for statement (line 307)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a BinOp to a Name (line 308):
        
        # Assigning a BinOp to a Name (line 308):
        
        # Call to get(...): (line 308)
        # Processing the call arguments (line 308)
        # Getting the type of 'self' (line 308)
        self_375673 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 308, 31), 'self', False)
        # Getting the type of 'key' (line 308)
        key_375674 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 308, 38), 'key', False)
        int_375675 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 308, 44), 'int')
        # Processing the call keyword arguments (line 308)
        kwargs_375676 = {}
        # Getting the type of 'dict' (line 308)
        dict_375671 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 308, 22), 'dict', False)
        # Obtaining the member 'get' of a type (line 308)
        get_375672 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 308, 22), dict_375671, 'get')
        # Calling get(args, kwargs) (line 308)
        get_call_result_375677 = invoke(stypy.reporting.localization.Localization(__file__, 308, 22), get_375672, *[self_375673, key_375674, int_375675], **kwargs_375676)
        
        # Getting the type of 'other' (line 308)
        other_375678 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 308, 49), 'other')
        # Applying the binary operator '+' (line 308)
        result_add_375679 = python_operator(stypy.reporting.localization.Localization(__file__, 308, 22), '+', get_call_result_375677, other_375678)
        
        # Assigning a type to the variable 'aij' (line 308)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 308, 16), 'aij', result_add_375679)
        
        # Getting the type of 'aij' (line 309)
        aij_375680 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 309, 19), 'aij')
        # Testing the type of an if condition (line 309)
        if_condition_375681 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 309, 16), aij_375680)
        # Assigning a type to the variable 'if_condition_375681' (line 309)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 309, 16), 'if_condition_375681', if_condition_375681)
        # SSA begins for if statement (line 309)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Name to a Subscript (line 310):
        
        # Assigning a Name to a Subscript (line 310):
        # Getting the type of 'aij' (line 310)
        aij_375682 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 310, 31), 'aij')
        # Getting the type of 'new' (line 310)
        new_375683 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 310, 20), 'new')
        # Getting the type of 'key' (line 310)
        key_375684 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 310, 24), 'key')
        # Storing an element on a container (line 310)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 310, 20), new_375683, (key_375684, aij_375682))
        # SSA join for if statement (line 309)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA branch for the else part of an if statement (line 302)
        module_type_store.open_ssa_branch('else')
        
        
        # Call to isspmatrix_dok(...): (line 312)
        # Processing the call arguments (line 312)
        # Getting the type of 'other' (line 312)
        other_375686 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 312, 28), 'other', False)
        # Processing the call keyword arguments (line 312)
        kwargs_375687 = {}
        # Getting the type of 'isspmatrix_dok' (line 312)
        isspmatrix_dok_375685 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 312, 13), 'isspmatrix_dok', False)
        # Calling isspmatrix_dok(args, kwargs) (line 312)
        isspmatrix_dok_call_result_375688 = invoke(stypy.reporting.localization.Localization(__file__, 312, 13), isspmatrix_dok_375685, *[other_375686], **kwargs_375687)
        
        # Testing the type of an if condition (line 312)
        if_condition_375689 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 312, 13), isspmatrix_dok_call_result_375688)
        # Assigning a type to the variable 'if_condition_375689' (line 312)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 312, 13), 'if_condition_375689', if_condition_375689)
        # SSA begins for if statement (line 312)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        
        # Getting the type of 'other' (line 313)
        other_375690 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 313, 15), 'other')
        # Obtaining the member 'shape' of a type (line 313)
        shape_375691 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 313, 15), other_375690, 'shape')
        # Getting the type of 'self' (line 313)
        self_375692 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 313, 30), 'self')
        # Obtaining the member 'shape' of a type (line 313)
        shape_375693 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 313, 30), self_375692, 'shape')
        # Applying the binary operator '!=' (line 313)
        result_ne_375694 = python_operator(stypy.reporting.localization.Localization(__file__, 313, 15), '!=', shape_375691, shape_375693)
        
        # Testing the type of an if condition (line 313)
        if_condition_375695 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 313, 12), result_ne_375694)
        # Assigning a type to the variable 'if_condition_375695' (line 313)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 313, 12), 'if_condition_375695', if_condition_375695)
        # SSA begins for if statement (line 313)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to ValueError(...): (line 314)
        # Processing the call arguments (line 314)
        str_375697 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 314, 33), 'str', 'Matrix dimensions are not equal.')
        # Processing the call keyword arguments (line 314)
        kwargs_375698 = {}
        # Getting the type of 'ValueError' (line 314)
        ValueError_375696 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 314, 22), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 314)
        ValueError_call_result_375699 = invoke(stypy.reporting.localization.Localization(__file__, 314, 22), ValueError_375696, *[str_375697], **kwargs_375698)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 314, 16), ValueError_call_result_375699, 'raise parameter', BaseException)
        # SSA join for if statement (line 313)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Name (line 317):
        
        # Assigning a Call to a Name (line 317):
        
        # Call to upcast(...): (line 317)
        # Processing the call arguments (line 317)
        # Getting the type of 'self' (line 317)
        self_375701 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 317, 31), 'self', False)
        # Obtaining the member 'dtype' of a type (line 317)
        dtype_375702 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 317, 31), self_375701, 'dtype')
        # Getting the type of 'other' (line 317)
        other_375703 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 317, 43), 'other', False)
        # Obtaining the member 'dtype' of a type (line 317)
        dtype_375704 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 317, 43), other_375703, 'dtype')
        # Processing the call keyword arguments (line 317)
        kwargs_375705 = {}
        # Getting the type of 'upcast' (line 317)
        upcast_375700 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 317, 24), 'upcast', False)
        # Calling upcast(args, kwargs) (line 317)
        upcast_call_result_375706 = invoke(stypy.reporting.localization.Localization(__file__, 317, 24), upcast_375700, *[dtype_375702, dtype_375704], **kwargs_375705)
        
        # Assigning a type to the variable 'res_dtype' (line 317)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 317, 12), 'res_dtype', upcast_call_result_375706)
        
        # Assigning a Call to a Name (line 318):
        
        # Assigning a Call to a Name (line 318):
        
        # Call to dok_matrix(...): (line 318)
        # Processing the call arguments (line 318)
        # Getting the type of 'self' (line 318)
        self_375708 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 318, 29), 'self', False)
        # Obtaining the member 'shape' of a type (line 318)
        shape_375709 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 318, 29), self_375708, 'shape')
        # Processing the call keyword arguments (line 318)
        # Getting the type of 'res_dtype' (line 318)
        res_dtype_375710 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 318, 47), 'res_dtype', False)
        keyword_375711 = res_dtype_375710
        kwargs_375712 = {'dtype': keyword_375711}
        # Getting the type of 'dok_matrix' (line 318)
        dok_matrix_375707 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 318, 18), 'dok_matrix', False)
        # Calling dok_matrix(args, kwargs) (line 318)
        dok_matrix_call_result_375713 = invoke(stypy.reporting.localization.Localization(__file__, 318, 18), dok_matrix_375707, *[shape_375709], **kwargs_375712)
        
        # Assigning a type to the variable 'new' (line 318)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 318, 12), 'new', dok_matrix_call_result_375713)
        
        # Call to update(...): (line 319)
        # Processing the call arguments (line 319)
        # Getting the type of 'new' (line 319)
        new_375716 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 319, 24), 'new', False)
        # Getting the type of 'self' (line 319)
        self_375717 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 319, 29), 'self', False)
        # Processing the call keyword arguments (line 319)
        kwargs_375718 = {}
        # Getting the type of 'dict' (line 319)
        dict_375714 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 319, 12), 'dict', False)
        # Obtaining the member 'update' of a type (line 319)
        update_375715 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 319, 12), dict_375714, 'update')
        # Calling update(args, kwargs) (line 319)
        update_call_result_375719 = invoke(stypy.reporting.localization.Localization(__file__, 319, 12), update_375715, *[new_375716, self_375717], **kwargs_375718)
        
        
        # Call to errstate(...): (line 320)
        # Processing the call keyword arguments (line 320)
        str_375722 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 320, 34), 'str', 'ignore')
        keyword_375723 = str_375722
        kwargs_375724 = {'over': keyword_375723}
        # Getting the type of 'np' (line 320)
        np_375720 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 320, 17), 'np', False)
        # Obtaining the member 'errstate' of a type (line 320)
        errstate_375721 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 320, 17), np_375720, 'errstate')
        # Calling errstate(args, kwargs) (line 320)
        errstate_call_result_375725 = invoke(stypy.reporting.localization.Localization(__file__, 320, 17), errstate_375721, *[], **kwargs_375724)
        
        with_375726 = ensure_var_has_members(stypy.reporting.localization.Localization(__file__, 320, 17), errstate_call_result_375725, 'with parameter', '__enter__', '__exit__')

        if with_375726:
            # Calling the __enter__ method to initiate a with section
            # Obtaining the member '__enter__' of a type (line 320)
            enter___375727 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 320, 17), errstate_call_result_375725, '__enter__')
            with_enter_375728 = invoke(stypy.reporting.localization.Localization(__file__, 320, 17), enter___375727)
            
            # Call to update(...): (line 321)
            # Processing the call arguments (line 321)
            # Getting the type of 'new' (line 321)
            new_375731 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 321, 28), 'new', False)
            # Calculating generator expression
            module_type_store = module_type_store.open_function_context('list comprehension expression', 322, 28, True)
            # Calculating comprehension expression
            
            # Call to iterkeys(...): (line 322)
            # Processing the call arguments (line 322)
            # Getting the type of 'other' (line 322)
            other_375744 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 322, 69), 'other', False)
            # Processing the call keyword arguments (line 322)
            kwargs_375745 = {}
            # Getting the type of 'iterkeys' (line 322)
            iterkeys_375743 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 322, 60), 'iterkeys', False)
            # Calling iterkeys(args, kwargs) (line 322)
            iterkeys_call_result_375746 = invoke(stypy.reporting.localization.Localization(__file__, 322, 60), iterkeys_375743, *[other_375744], **kwargs_375745)
            
            comprehension_375747 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 322, 28), iterkeys_call_result_375746)
            # Assigning a type to the variable 'k' (line 322)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 322, 28), 'k', comprehension_375747)
            
            # Obtaining an instance of the builtin type 'tuple' (line 322)
            tuple_375732 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 322, 29), 'tuple')
            # Adding type elements to the builtin type 'tuple' instance (line 322)
            # Adding element type (line 322)
            # Getting the type of 'k' (line 322)
            k_375733 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 322, 29), 'k', False)
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 322, 29), tuple_375732, k_375733)
            # Adding element type (line 322)
            
            # Obtaining the type of the subscript
            # Getting the type of 'k' (line 322)
            k_375734 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 322, 36), 'k', False)
            # Getting the type of 'new' (line 322)
            new_375735 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 322, 32), 'new', False)
            # Obtaining the member '__getitem__' of a type (line 322)
            getitem___375736 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 322, 32), new_375735, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 322)
            subscript_call_result_375737 = invoke(stypy.reporting.localization.Localization(__file__, 322, 32), getitem___375736, k_375734)
            
            
            # Obtaining the type of the subscript
            # Getting the type of 'k' (line 322)
            k_375738 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 322, 47), 'k', False)
            # Getting the type of 'other' (line 322)
            other_375739 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 322, 41), 'other', False)
            # Obtaining the member '__getitem__' of a type (line 322)
            getitem___375740 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 322, 41), other_375739, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 322)
            subscript_call_result_375741 = invoke(stypy.reporting.localization.Localization(__file__, 322, 41), getitem___375740, k_375738)
            
            # Applying the binary operator '+' (line 322)
            result_add_375742 = python_operator(stypy.reporting.localization.Localization(__file__, 322, 32), '+', subscript_call_result_375737, subscript_call_result_375741)
            
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 322, 29), tuple_375732, result_add_375742)
            
            list_375748 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 322, 28), 'list')
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 322, 28), list_375748, tuple_375732)
            # Processing the call keyword arguments (line 321)
            kwargs_375749 = {}
            # Getting the type of 'dict' (line 321)
            dict_375729 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 321, 16), 'dict', False)
            # Obtaining the member 'update' of a type (line 321)
            update_375730 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 321, 16), dict_375729, 'update')
            # Calling update(args, kwargs) (line 321)
            update_call_result_375750 = invoke(stypy.reporting.localization.Localization(__file__, 321, 16), update_375730, *[new_375731, list_375748], **kwargs_375749)
            
            # Calling the __exit__ method to finish a with section
            # Obtaining the member '__exit__' of a type (line 320)
            exit___375751 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 320, 17), errstate_call_result_375725, '__exit__')
            with_exit_375752 = invoke(stypy.reporting.localization.Localization(__file__, 320, 17), exit___375751, None, None, None)

        # SSA branch for the else part of an if statement (line 312)
        module_type_store.open_ssa_branch('else')
        
        
        # Call to isspmatrix(...): (line 323)
        # Processing the call arguments (line 323)
        # Getting the type of 'other' (line 323)
        other_375754 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 323, 24), 'other', False)
        # Processing the call keyword arguments (line 323)
        kwargs_375755 = {}
        # Getting the type of 'isspmatrix' (line 323)
        isspmatrix_375753 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 323, 13), 'isspmatrix', False)
        # Calling isspmatrix(args, kwargs) (line 323)
        isspmatrix_call_result_375756 = invoke(stypy.reporting.localization.Localization(__file__, 323, 13), isspmatrix_375753, *[other_375754], **kwargs_375755)
        
        # Testing the type of an if condition (line 323)
        if_condition_375757 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 323, 13), isspmatrix_call_result_375756)
        # Assigning a type to the variable 'if_condition_375757' (line 323)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 323, 13), 'if_condition_375757', if_condition_375757)
        # SSA begins for if statement (line 323)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 324):
        
        # Assigning a Call to a Name (line 324):
        
        # Call to tocsc(...): (line 324)
        # Processing the call keyword arguments (line 324)
        kwargs_375760 = {}
        # Getting the type of 'self' (line 324)
        self_375758 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 324, 18), 'self', False)
        # Obtaining the member 'tocsc' of a type (line 324)
        tocsc_375759 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 324, 18), self_375758, 'tocsc')
        # Calling tocsc(args, kwargs) (line 324)
        tocsc_call_result_375761 = invoke(stypy.reporting.localization.Localization(__file__, 324, 18), tocsc_375759, *[], **kwargs_375760)
        
        # Assigning a type to the variable 'csc' (line 324)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 324, 12), 'csc', tocsc_call_result_375761)
        
        # Assigning a BinOp to a Name (line 325):
        
        # Assigning a BinOp to a Name (line 325):
        # Getting the type of 'csc' (line 325)
        csc_375762 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 325, 18), 'csc')
        # Getting the type of 'other' (line 325)
        other_375763 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 325, 24), 'other')
        # Applying the binary operator '+' (line 325)
        result_add_375764 = python_operator(stypy.reporting.localization.Localization(__file__, 325, 18), '+', csc_375762, other_375763)
        
        # Assigning a type to the variable 'new' (line 325)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 325, 12), 'new', result_add_375764)
        # SSA branch for the else part of an if statement (line 323)
        module_type_store.open_ssa_branch('else')
        
        
        # Call to isdense(...): (line 326)
        # Processing the call arguments (line 326)
        # Getting the type of 'other' (line 326)
        other_375766 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 326, 21), 'other', False)
        # Processing the call keyword arguments (line 326)
        kwargs_375767 = {}
        # Getting the type of 'isdense' (line 326)
        isdense_375765 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 326, 13), 'isdense', False)
        # Calling isdense(args, kwargs) (line 326)
        isdense_call_result_375768 = invoke(stypy.reporting.localization.Localization(__file__, 326, 13), isdense_375765, *[other_375766], **kwargs_375767)
        
        # Testing the type of an if condition (line 326)
        if_condition_375769 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 326, 13), isdense_call_result_375768)
        # Assigning a type to the variable 'if_condition_375769' (line 326)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 326, 13), 'if_condition_375769', if_condition_375769)
        # SSA begins for if statement (line 326)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a BinOp to a Name (line 327):
        
        # Assigning a BinOp to a Name (line 327):
        
        # Call to todense(...): (line 327)
        # Processing the call keyword arguments (line 327)
        kwargs_375772 = {}
        # Getting the type of 'self' (line 327)
        self_375770 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 327, 18), 'self', False)
        # Obtaining the member 'todense' of a type (line 327)
        todense_375771 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 327, 18), self_375770, 'todense')
        # Calling todense(args, kwargs) (line 327)
        todense_call_result_375773 = invoke(stypy.reporting.localization.Localization(__file__, 327, 18), todense_375771, *[], **kwargs_375772)
        
        # Getting the type of 'other' (line 327)
        other_375774 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 327, 35), 'other')
        # Applying the binary operator '+' (line 327)
        result_add_375775 = python_operator(stypy.reporting.localization.Localization(__file__, 327, 18), '+', todense_call_result_375773, other_375774)
        
        # Assigning a type to the variable 'new' (line 327)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 327, 12), 'new', result_add_375775)
        # SSA branch for the else part of an if statement (line 326)
        module_type_store.open_ssa_branch('else')
        # Getting the type of 'NotImplemented' (line 329)
        NotImplemented_375776 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 329, 19), 'NotImplemented')
        # Assigning a type to the variable 'stypy_return_type' (line 329)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 329, 12), 'stypy_return_type', NotImplemented_375776)
        # SSA join for if statement (line 326)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 323)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 312)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 302)
        module_type_store = module_type_store.join_ssa_context()
        
        # Getting the type of 'new' (line 330)
        new_375777 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 330, 15), 'new')
        # Assigning a type to the variable 'stypy_return_type' (line 330)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 330, 8), 'stypy_return_type', new_375777)
        
        # ################# End of '__add__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__add__' in the type store
        # Getting the type of 'stypy_return_type' (line 301)
        stypy_return_type_375778 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 301, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_375778)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__add__'
        return stypy_return_type_375778


    @norecursion
    def __radd__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__radd__'
        module_type_store = module_type_store.open_function_context('__radd__', 332, 4, False)
        # Assigning a type to the variable 'self' (line 333)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 333, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        dok_matrix.__radd__.__dict__.__setitem__('stypy_localization', localization)
        dok_matrix.__radd__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        dok_matrix.__radd__.__dict__.__setitem__('stypy_type_store', module_type_store)
        dok_matrix.__radd__.__dict__.__setitem__('stypy_function_name', 'dok_matrix.__radd__')
        dok_matrix.__radd__.__dict__.__setitem__('stypy_param_names_list', ['other'])
        dok_matrix.__radd__.__dict__.__setitem__('stypy_varargs_param_name', None)
        dok_matrix.__radd__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        dok_matrix.__radd__.__dict__.__setitem__('stypy_call_defaults', defaults)
        dok_matrix.__radd__.__dict__.__setitem__('stypy_call_varargs', varargs)
        dok_matrix.__radd__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        dok_matrix.__radd__.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'dok_matrix.__radd__', ['other'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__radd__', localization, ['other'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__radd__(...)' code ##################

        
        
        # Call to isscalarlike(...): (line 333)
        # Processing the call arguments (line 333)
        # Getting the type of 'other' (line 333)
        other_375780 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 333, 24), 'other', False)
        # Processing the call keyword arguments (line 333)
        kwargs_375781 = {}
        # Getting the type of 'isscalarlike' (line 333)
        isscalarlike_375779 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 333, 11), 'isscalarlike', False)
        # Calling isscalarlike(args, kwargs) (line 333)
        isscalarlike_call_result_375782 = invoke(stypy.reporting.localization.Localization(__file__, 333, 11), isscalarlike_375779, *[other_375780], **kwargs_375781)
        
        # Testing the type of an if condition (line 333)
        if_condition_375783 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 333, 8), isscalarlike_call_result_375782)
        # Assigning a type to the variable 'if_condition_375783' (line 333)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 333, 8), 'if_condition_375783', if_condition_375783)
        # SSA begins for if statement (line 333)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 334):
        
        # Assigning a Call to a Name (line 334):
        
        # Call to dok_matrix(...): (line 334)
        # Processing the call arguments (line 334)
        # Getting the type of 'self' (line 334)
        self_375785 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 334, 29), 'self', False)
        # Obtaining the member 'shape' of a type (line 334)
        shape_375786 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 334, 29), self_375785, 'shape')
        # Processing the call keyword arguments (line 334)
        # Getting the type of 'self' (line 334)
        self_375787 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 334, 47), 'self', False)
        # Obtaining the member 'dtype' of a type (line 334)
        dtype_375788 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 334, 47), self_375787, 'dtype')
        keyword_375789 = dtype_375788
        kwargs_375790 = {'dtype': keyword_375789}
        # Getting the type of 'dok_matrix' (line 334)
        dok_matrix_375784 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 334, 18), 'dok_matrix', False)
        # Calling dok_matrix(args, kwargs) (line 334)
        dok_matrix_call_result_375791 = invoke(stypy.reporting.localization.Localization(__file__, 334, 18), dok_matrix_375784, *[shape_375786], **kwargs_375790)
        
        # Assigning a type to the variable 'new' (line 334)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 334, 12), 'new', dok_matrix_call_result_375791)
        
        # Assigning a Attribute to a Tuple (line 335):
        
        # Assigning a Subscript to a Name (line 335):
        
        # Obtaining the type of the subscript
        int_375792 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 335, 12), 'int')
        # Getting the type of 'self' (line 335)
        self_375793 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 335, 19), 'self')
        # Obtaining the member 'shape' of a type (line 335)
        shape_375794 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 335, 19), self_375793, 'shape')
        # Obtaining the member '__getitem__' of a type (line 335)
        getitem___375795 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 335, 12), shape_375794, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 335)
        subscript_call_result_375796 = invoke(stypy.reporting.localization.Localization(__file__, 335, 12), getitem___375795, int_375792)
        
        # Assigning a type to the variable 'tuple_var_assignment_374368' (line 335)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 335, 12), 'tuple_var_assignment_374368', subscript_call_result_375796)
        
        # Assigning a Subscript to a Name (line 335):
        
        # Obtaining the type of the subscript
        int_375797 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 335, 12), 'int')
        # Getting the type of 'self' (line 335)
        self_375798 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 335, 19), 'self')
        # Obtaining the member 'shape' of a type (line 335)
        shape_375799 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 335, 19), self_375798, 'shape')
        # Obtaining the member '__getitem__' of a type (line 335)
        getitem___375800 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 335, 12), shape_375799, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 335)
        subscript_call_result_375801 = invoke(stypy.reporting.localization.Localization(__file__, 335, 12), getitem___375800, int_375797)
        
        # Assigning a type to the variable 'tuple_var_assignment_374369' (line 335)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 335, 12), 'tuple_var_assignment_374369', subscript_call_result_375801)
        
        # Assigning a Name to a Name (line 335):
        # Getting the type of 'tuple_var_assignment_374368' (line 335)
        tuple_var_assignment_374368_375802 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 335, 12), 'tuple_var_assignment_374368')
        # Assigning a type to the variable 'M' (line 335)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 335, 12), 'M', tuple_var_assignment_374368_375802)
        
        # Assigning a Name to a Name (line 335):
        # Getting the type of 'tuple_var_assignment_374369' (line 335)
        tuple_var_assignment_374369_375803 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 335, 12), 'tuple_var_assignment_374369')
        # Assigning a type to the variable 'N' (line 335)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 335, 15), 'N', tuple_var_assignment_374369_375803)
        
        
        # Call to product(...): (line 336)
        # Processing the call arguments (line 336)
        
        # Call to xrange(...): (line 336)
        # Processing the call arguments (line 336)
        # Getting the type of 'M' (line 336)
        M_375807 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 336, 48), 'M', False)
        # Processing the call keyword arguments (line 336)
        kwargs_375808 = {}
        # Getting the type of 'xrange' (line 336)
        xrange_375806 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 336, 41), 'xrange', False)
        # Calling xrange(args, kwargs) (line 336)
        xrange_call_result_375809 = invoke(stypy.reporting.localization.Localization(__file__, 336, 41), xrange_375806, *[M_375807], **kwargs_375808)
        
        
        # Call to xrange(...): (line 336)
        # Processing the call arguments (line 336)
        # Getting the type of 'N' (line 336)
        N_375811 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 336, 59), 'N', False)
        # Processing the call keyword arguments (line 336)
        kwargs_375812 = {}
        # Getting the type of 'xrange' (line 336)
        xrange_375810 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 336, 52), 'xrange', False)
        # Calling xrange(args, kwargs) (line 336)
        xrange_call_result_375813 = invoke(stypy.reporting.localization.Localization(__file__, 336, 52), xrange_375810, *[N_375811], **kwargs_375812)
        
        # Processing the call keyword arguments (line 336)
        kwargs_375814 = {}
        # Getting the type of 'itertools' (line 336)
        itertools_375804 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 336, 23), 'itertools', False)
        # Obtaining the member 'product' of a type (line 336)
        product_375805 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 336, 23), itertools_375804, 'product')
        # Calling product(args, kwargs) (line 336)
        product_call_result_375815 = invoke(stypy.reporting.localization.Localization(__file__, 336, 23), product_375805, *[xrange_call_result_375809, xrange_call_result_375813], **kwargs_375814)
        
        # Testing the type of a for loop iterable (line 336)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 336, 12), product_call_result_375815)
        # Getting the type of the for loop variable (line 336)
        for_loop_var_375816 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 336, 12), product_call_result_375815)
        # Assigning a type to the variable 'key' (line 336)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 336, 12), 'key', for_loop_var_375816)
        # SSA begins for a for statement (line 336)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a BinOp to a Name (line 337):
        
        # Assigning a BinOp to a Name (line 337):
        
        # Call to get(...): (line 337)
        # Processing the call arguments (line 337)
        # Getting the type of 'self' (line 337)
        self_375819 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 337, 31), 'self', False)
        # Getting the type of 'key' (line 337)
        key_375820 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 337, 38), 'key', False)
        int_375821 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 337, 44), 'int')
        # Processing the call keyword arguments (line 337)
        kwargs_375822 = {}
        # Getting the type of 'dict' (line 337)
        dict_375817 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 337, 22), 'dict', False)
        # Obtaining the member 'get' of a type (line 337)
        get_375818 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 337, 22), dict_375817, 'get')
        # Calling get(args, kwargs) (line 337)
        get_call_result_375823 = invoke(stypy.reporting.localization.Localization(__file__, 337, 22), get_375818, *[self_375819, key_375820, int_375821], **kwargs_375822)
        
        # Getting the type of 'other' (line 337)
        other_375824 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 337, 49), 'other')
        # Applying the binary operator '+' (line 337)
        result_add_375825 = python_operator(stypy.reporting.localization.Localization(__file__, 337, 22), '+', get_call_result_375823, other_375824)
        
        # Assigning a type to the variable 'aij' (line 337)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 337, 16), 'aij', result_add_375825)
        
        # Getting the type of 'aij' (line 338)
        aij_375826 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 338, 19), 'aij')
        # Testing the type of an if condition (line 338)
        if_condition_375827 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 338, 16), aij_375826)
        # Assigning a type to the variable 'if_condition_375827' (line 338)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 338, 16), 'if_condition_375827', if_condition_375827)
        # SSA begins for if statement (line 338)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Name to a Subscript (line 339):
        
        # Assigning a Name to a Subscript (line 339):
        # Getting the type of 'aij' (line 339)
        aij_375828 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 339, 31), 'aij')
        # Getting the type of 'new' (line 339)
        new_375829 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 339, 20), 'new')
        # Getting the type of 'key' (line 339)
        key_375830 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 339, 24), 'key')
        # Storing an element on a container (line 339)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 339, 20), new_375829, (key_375830, aij_375828))
        # SSA join for if statement (line 338)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA branch for the else part of an if statement (line 333)
        module_type_store.open_ssa_branch('else')
        
        
        # Call to isspmatrix_dok(...): (line 340)
        # Processing the call arguments (line 340)
        # Getting the type of 'other' (line 340)
        other_375832 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 340, 28), 'other', False)
        # Processing the call keyword arguments (line 340)
        kwargs_375833 = {}
        # Getting the type of 'isspmatrix_dok' (line 340)
        isspmatrix_dok_375831 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 340, 13), 'isspmatrix_dok', False)
        # Calling isspmatrix_dok(args, kwargs) (line 340)
        isspmatrix_dok_call_result_375834 = invoke(stypy.reporting.localization.Localization(__file__, 340, 13), isspmatrix_dok_375831, *[other_375832], **kwargs_375833)
        
        # Testing the type of an if condition (line 340)
        if_condition_375835 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 340, 13), isspmatrix_dok_call_result_375834)
        # Assigning a type to the variable 'if_condition_375835' (line 340)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 340, 13), 'if_condition_375835', if_condition_375835)
        # SSA begins for if statement (line 340)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        
        # Getting the type of 'other' (line 341)
        other_375836 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 341, 15), 'other')
        # Obtaining the member 'shape' of a type (line 341)
        shape_375837 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 341, 15), other_375836, 'shape')
        # Getting the type of 'self' (line 341)
        self_375838 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 341, 30), 'self')
        # Obtaining the member 'shape' of a type (line 341)
        shape_375839 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 341, 30), self_375838, 'shape')
        # Applying the binary operator '!=' (line 341)
        result_ne_375840 = python_operator(stypy.reporting.localization.Localization(__file__, 341, 15), '!=', shape_375837, shape_375839)
        
        # Testing the type of an if condition (line 341)
        if_condition_375841 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 341, 12), result_ne_375840)
        # Assigning a type to the variable 'if_condition_375841' (line 341)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 341, 12), 'if_condition_375841', if_condition_375841)
        # SSA begins for if statement (line 341)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to ValueError(...): (line 342)
        # Processing the call arguments (line 342)
        str_375843 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 342, 33), 'str', 'Matrix dimensions are not equal.')
        # Processing the call keyword arguments (line 342)
        kwargs_375844 = {}
        # Getting the type of 'ValueError' (line 342)
        ValueError_375842 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 342, 22), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 342)
        ValueError_call_result_375845 = invoke(stypy.reporting.localization.Localization(__file__, 342, 22), ValueError_375842, *[str_375843], **kwargs_375844)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 342, 16), ValueError_call_result_375845, 'raise parameter', BaseException)
        # SSA join for if statement (line 341)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Name (line 343):
        
        # Assigning a Call to a Name (line 343):
        
        # Call to dok_matrix(...): (line 343)
        # Processing the call arguments (line 343)
        # Getting the type of 'self' (line 343)
        self_375847 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 343, 29), 'self', False)
        # Obtaining the member 'shape' of a type (line 343)
        shape_375848 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 343, 29), self_375847, 'shape')
        # Processing the call keyword arguments (line 343)
        # Getting the type of 'self' (line 343)
        self_375849 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 343, 47), 'self', False)
        # Obtaining the member 'dtype' of a type (line 343)
        dtype_375850 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 343, 47), self_375849, 'dtype')
        keyword_375851 = dtype_375850
        kwargs_375852 = {'dtype': keyword_375851}
        # Getting the type of 'dok_matrix' (line 343)
        dok_matrix_375846 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 343, 18), 'dok_matrix', False)
        # Calling dok_matrix(args, kwargs) (line 343)
        dok_matrix_call_result_375853 = invoke(stypy.reporting.localization.Localization(__file__, 343, 18), dok_matrix_375846, *[shape_375848], **kwargs_375852)
        
        # Assigning a type to the variable 'new' (line 343)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 343, 12), 'new', dok_matrix_call_result_375853)
        
        # Call to update(...): (line 344)
        # Processing the call arguments (line 344)
        # Getting the type of 'new' (line 344)
        new_375856 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 344, 24), 'new', False)
        # Getting the type of 'self' (line 344)
        self_375857 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 344, 29), 'self', False)
        # Processing the call keyword arguments (line 344)
        kwargs_375858 = {}
        # Getting the type of 'dict' (line 344)
        dict_375854 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 344, 12), 'dict', False)
        # Obtaining the member 'update' of a type (line 344)
        update_375855 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 344, 12), dict_375854, 'update')
        # Calling update(args, kwargs) (line 344)
        update_call_result_375859 = invoke(stypy.reporting.localization.Localization(__file__, 344, 12), update_375855, *[new_375856, self_375857], **kwargs_375858)
        
        
        # Call to update(...): (line 345)
        # Processing the call arguments (line 345)
        # Getting the type of 'new' (line 345)
        new_375862 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 345, 24), 'new', False)
        # Calculating generator expression
        module_type_store = module_type_store.open_function_context('list comprehension expression', 346, 24, True)
        # Calculating comprehension expression
        
        # Call to iterkeys(...): (line 346)
        # Processing the call arguments (line 346)
        # Getting the type of 'other' (line 346)
        other_375875 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 346, 66), 'other', False)
        # Processing the call keyword arguments (line 346)
        kwargs_375876 = {}
        # Getting the type of 'iterkeys' (line 346)
        iterkeys_375874 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 346, 57), 'iterkeys', False)
        # Calling iterkeys(args, kwargs) (line 346)
        iterkeys_call_result_375877 = invoke(stypy.reporting.localization.Localization(__file__, 346, 57), iterkeys_375874, *[other_375875], **kwargs_375876)
        
        comprehension_375878 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 346, 24), iterkeys_call_result_375877)
        # Assigning a type to the variable 'k' (line 346)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 346, 24), 'k', comprehension_375878)
        
        # Obtaining an instance of the builtin type 'tuple' (line 346)
        tuple_375863 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 346, 25), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 346)
        # Adding element type (line 346)
        # Getting the type of 'k' (line 346)
        k_375864 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 346, 25), 'k', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 346, 25), tuple_375863, k_375864)
        # Adding element type (line 346)
        
        # Obtaining the type of the subscript
        # Getting the type of 'k' (line 346)
        k_375865 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 346, 33), 'k', False)
        # Getting the type of 'self' (line 346)
        self_375866 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 346, 28), 'self', False)
        # Obtaining the member '__getitem__' of a type (line 346)
        getitem___375867 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 346, 28), self_375866, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 346)
        subscript_call_result_375868 = invoke(stypy.reporting.localization.Localization(__file__, 346, 28), getitem___375867, k_375865)
        
        
        # Obtaining the type of the subscript
        # Getting the type of 'k' (line 346)
        k_375869 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 346, 44), 'k', False)
        # Getting the type of 'other' (line 346)
        other_375870 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 346, 38), 'other', False)
        # Obtaining the member '__getitem__' of a type (line 346)
        getitem___375871 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 346, 38), other_375870, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 346)
        subscript_call_result_375872 = invoke(stypy.reporting.localization.Localization(__file__, 346, 38), getitem___375871, k_375869)
        
        # Applying the binary operator '+' (line 346)
        result_add_375873 = python_operator(stypy.reporting.localization.Localization(__file__, 346, 28), '+', subscript_call_result_375868, subscript_call_result_375872)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 346, 25), tuple_375863, result_add_375873)
        
        list_375879 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 346, 24), 'list')
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 346, 24), list_375879, tuple_375863)
        # Processing the call keyword arguments (line 345)
        kwargs_375880 = {}
        # Getting the type of 'dict' (line 345)
        dict_375860 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 345, 12), 'dict', False)
        # Obtaining the member 'update' of a type (line 345)
        update_375861 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 345, 12), dict_375860, 'update')
        # Calling update(args, kwargs) (line 345)
        update_call_result_375881 = invoke(stypy.reporting.localization.Localization(__file__, 345, 12), update_375861, *[new_375862, list_375879], **kwargs_375880)
        
        # SSA branch for the else part of an if statement (line 340)
        module_type_store.open_ssa_branch('else')
        
        
        # Call to isspmatrix(...): (line 347)
        # Processing the call arguments (line 347)
        # Getting the type of 'other' (line 347)
        other_375883 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 347, 24), 'other', False)
        # Processing the call keyword arguments (line 347)
        kwargs_375884 = {}
        # Getting the type of 'isspmatrix' (line 347)
        isspmatrix_375882 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 347, 13), 'isspmatrix', False)
        # Calling isspmatrix(args, kwargs) (line 347)
        isspmatrix_call_result_375885 = invoke(stypy.reporting.localization.Localization(__file__, 347, 13), isspmatrix_375882, *[other_375883], **kwargs_375884)
        
        # Testing the type of an if condition (line 347)
        if_condition_375886 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 347, 13), isspmatrix_call_result_375885)
        # Assigning a type to the variable 'if_condition_375886' (line 347)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 347, 13), 'if_condition_375886', if_condition_375886)
        # SSA begins for if statement (line 347)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 348):
        
        # Assigning a Call to a Name (line 348):
        
        # Call to tocsc(...): (line 348)
        # Processing the call keyword arguments (line 348)
        kwargs_375889 = {}
        # Getting the type of 'self' (line 348)
        self_375887 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 348, 18), 'self', False)
        # Obtaining the member 'tocsc' of a type (line 348)
        tocsc_375888 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 348, 18), self_375887, 'tocsc')
        # Calling tocsc(args, kwargs) (line 348)
        tocsc_call_result_375890 = invoke(stypy.reporting.localization.Localization(__file__, 348, 18), tocsc_375888, *[], **kwargs_375889)
        
        # Assigning a type to the variable 'csc' (line 348)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 348, 12), 'csc', tocsc_call_result_375890)
        
        # Assigning a BinOp to a Name (line 349):
        
        # Assigning a BinOp to a Name (line 349):
        # Getting the type of 'csc' (line 349)
        csc_375891 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 349, 18), 'csc')
        # Getting the type of 'other' (line 349)
        other_375892 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 349, 24), 'other')
        # Applying the binary operator '+' (line 349)
        result_add_375893 = python_operator(stypy.reporting.localization.Localization(__file__, 349, 18), '+', csc_375891, other_375892)
        
        # Assigning a type to the variable 'new' (line 349)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 349, 12), 'new', result_add_375893)
        # SSA branch for the else part of an if statement (line 347)
        module_type_store.open_ssa_branch('else')
        
        
        # Call to isdense(...): (line 350)
        # Processing the call arguments (line 350)
        # Getting the type of 'other' (line 350)
        other_375895 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 350, 21), 'other', False)
        # Processing the call keyword arguments (line 350)
        kwargs_375896 = {}
        # Getting the type of 'isdense' (line 350)
        isdense_375894 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 350, 13), 'isdense', False)
        # Calling isdense(args, kwargs) (line 350)
        isdense_call_result_375897 = invoke(stypy.reporting.localization.Localization(__file__, 350, 13), isdense_375894, *[other_375895], **kwargs_375896)
        
        # Testing the type of an if condition (line 350)
        if_condition_375898 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 350, 13), isdense_call_result_375897)
        # Assigning a type to the variable 'if_condition_375898' (line 350)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 350, 13), 'if_condition_375898', if_condition_375898)
        # SSA begins for if statement (line 350)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a BinOp to a Name (line 351):
        
        # Assigning a BinOp to a Name (line 351):
        # Getting the type of 'other' (line 351)
        other_375899 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 351, 18), 'other')
        
        # Call to todense(...): (line 351)
        # Processing the call keyword arguments (line 351)
        kwargs_375902 = {}
        # Getting the type of 'self' (line 351)
        self_375900 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 351, 26), 'self', False)
        # Obtaining the member 'todense' of a type (line 351)
        todense_375901 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 351, 26), self_375900, 'todense')
        # Calling todense(args, kwargs) (line 351)
        todense_call_result_375903 = invoke(stypy.reporting.localization.Localization(__file__, 351, 26), todense_375901, *[], **kwargs_375902)
        
        # Applying the binary operator '+' (line 351)
        result_add_375904 = python_operator(stypy.reporting.localization.Localization(__file__, 351, 18), '+', other_375899, todense_call_result_375903)
        
        # Assigning a type to the variable 'new' (line 351)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 351, 12), 'new', result_add_375904)
        # SSA branch for the else part of an if statement (line 350)
        module_type_store.open_ssa_branch('else')
        # Getting the type of 'NotImplemented' (line 353)
        NotImplemented_375905 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 353, 19), 'NotImplemented')
        # Assigning a type to the variable 'stypy_return_type' (line 353)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 353, 12), 'stypy_return_type', NotImplemented_375905)
        # SSA join for if statement (line 350)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 347)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 340)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 333)
        module_type_store = module_type_store.join_ssa_context()
        
        # Getting the type of 'new' (line 354)
        new_375906 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 354, 15), 'new')
        # Assigning a type to the variable 'stypy_return_type' (line 354)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 354, 8), 'stypy_return_type', new_375906)
        
        # ################# End of '__radd__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__radd__' in the type store
        # Getting the type of 'stypy_return_type' (line 332)
        stypy_return_type_375907 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 332, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_375907)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__radd__'
        return stypy_return_type_375907


    @norecursion
    def __neg__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__neg__'
        module_type_store = module_type_store.open_function_context('__neg__', 356, 4, False)
        # Assigning a type to the variable 'self' (line 357)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 357, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        dok_matrix.__neg__.__dict__.__setitem__('stypy_localization', localization)
        dok_matrix.__neg__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        dok_matrix.__neg__.__dict__.__setitem__('stypy_type_store', module_type_store)
        dok_matrix.__neg__.__dict__.__setitem__('stypy_function_name', 'dok_matrix.__neg__')
        dok_matrix.__neg__.__dict__.__setitem__('stypy_param_names_list', [])
        dok_matrix.__neg__.__dict__.__setitem__('stypy_varargs_param_name', None)
        dok_matrix.__neg__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        dok_matrix.__neg__.__dict__.__setitem__('stypy_call_defaults', defaults)
        dok_matrix.__neg__.__dict__.__setitem__('stypy_call_varargs', varargs)
        dok_matrix.__neg__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        dok_matrix.__neg__.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'dok_matrix.__neg__', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__neg__', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__neg__(...)' code ##################

        
        
        # Getting the type of 'self' (line 357)
        self_375908 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 357, 11), 'self')
        # Obtaining the member 'dtype' of a type (line 357)
        dtype_375909 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 357, 11), self_375908, 'dtype')
        # Obtaining the member 'kind' of a type (line 357)
        kind_375910 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 357, 11), dtype_375909, 'kind')
        str_375911 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 357, 30), 'str', 'b')
        # Applying the binary operator '==' (line 357)
        result_eq_375912 = python_operator(stypy.reporting.localization.Localization(__file__, 357, 11), '==', kind_375910, str_375911)
        
        # Testing the type of an if condition (line 357)
        if_condition_375913 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 357, 8), result_eq_375912)
        # Assigning a type to the variable 'if_condition_375913' (line 357)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 357, 8), 'if_condition_375913', if_condition_375913)
        # SSA begins for if statement (line 357)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to NotImplementedError(...): (line 358)
        # Processing the call arguments (line 358)
        str_375915 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 358, 38), 'str', 'Negating a sparse boolean matrix is not supported.')
        # Processing the call keyword arguments (line 358)
        kwargs_375916 = {}
        # Getting the type of 'NotImplementedError' (line 358)
        NotImplementedError_375914 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 358, 18), 'NotImplementedError', False)
        # Calling NotImplementedError(args, kwargs) (line 358)
        NotImplementedError_call_result_375917 = invoke(stypy.reporting.localization.Localization(__file__, 358, 18), NotImplementedError_375914, *[str_375915], **kwargs_375916)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 358, 12), NotImplementedError_call_result_375917, 'raise parameter', BaseException)
        # SSA join for if statement (line 357)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Name (line 360):
        
        # Assigning a Call to a Name (line 360):
        
        # Call to dok_matrix(...): (line 360)
        # Processing the call arguments (line 360)
        # Getting the type of 'self' (line 360)
        self_375919 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 360, 25), 'self', False)
        # Obtaining the member 'shape' of a type (line 360)
        shape_375920 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 360, 25), self_375919, 'shape')
        # Processing the call keyword arguments (line 360)
        # Getting the type of 'self' (line 360)
        self_375921 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 360, 43), 'self', False)
        # Obtaining the member 'dtype' of a type (line 360)
        dtype_375922 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 360, 43), self_375921, 'dtype')
        keyword_375923 = dtype_375922
        kwargs_375924 = {'dtype': keyword_375923}
        # Getting the type of 'dok_matrix' (line 360)
        dok_matrix_375918 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 360, 14), 'dok_matrix', False)
        # Calling dok_matrix(args, kwargs) (line 360)
        dok_matrix_call_result_375925 = invoke(stypy.reporting.localization.Localization(__file__, 360, 14), dok_matrix_375918, *[shape_375920], **kwargs_375924)
        
        # Assigning a type to the variable 'new' (line 360)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 360, 8), 'new', dok_matrix_call_result_375925)
        
        # Call to update(...): (line 361)
        # Processing the call arguments (line 361)
        # Getting the type of 'new' (line 361)
        new_375928 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 361, 20), 'new', False)
        # Calculating generator expression
        module_type_store = module_type_store.open_function_context('list comprehension expression', 361, 26, True)
        # Calculating comprehension expression
        
        # Call to iterkeys(...): (line 361)
        # Processing the call arguments (line 361)
        # Getting the type of 'self' (line 361)
        self_375937 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 361, 58), 'self', False)
        # Processing the call keyword arguments (line 361)
        kwargs_375938 = {}
        # Getting the type of 'iterkeys' (line 361)
        iterkeys_375936 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 361, 49), 'iterkeys', False)
        # Calling iterkeys(args, kwargs) (line 361)
        iterkeys_call_result_375939 = invoke(stypy.reporting.localization.Localization(__file__, 361, 49), iterkeys_375936, *[self_375937], **kwargs_375938)
        
        comprehension_375940 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 361, 26), iterkeys_call_result_375939)
        # Assigning a type to the variable 'k' (line 361)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 361, 26), 'k', comprehension_375940)
        
        # Obtaining an instance of the builtin type 'tuple' (line 361)
        tuple_375929 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 361, 27), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 361)
        # Adding element type (line 361)
        # Getting the type of 'k' (line 361)
        k_375930 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 361, 27), 'k', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 361, 27), tuple_375929, k_375930)
        # Adding element type (line 361)
        
        
        # Obtaining the type of the subscript
        # Getting the type of 'k' (line 361)
        k_375931 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 361, 36), 'k', False)
        # Getting the type of 'self' (line 361)
        self_375932 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 361, 31), 'self', False)
        # Obtaining the member '__getitem__' of a type (line 361)
        getitem___375933 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 361, 31), self_375932, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 361)
        subscript_call_result_375934 = invoke(stypy.reporting.localization.Localization(__file__, 361, 31), getitem___375933, k_375931)
        
        # Applying the 'usub' unary operator (line 361)
        result___neg___375935 = python_operator(stypy.reporting.localization.Localization(__file__, 361, 30), 'usub', subscript_call_result_375934)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 361, 27), tuple_375929, result___neg___375935)
        
        list_375941 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 361, 26), 'list')
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 361, 26), list_375941, tuple_375929)
        # Processing the call keyword arguments (line 361)
        kwargs_375942 = {}
        # Getting the type of 'dict' (line 361)
        dict_375926 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 361, 8), 'dict', False)
        # Obtaining the member 'update' of a type (line 361)
        update_375927 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 361, 8), dict_375926, 'update')
        # Calling update(args, kwargs) (line 361)
        update_call_result_375943 = invoke(stypy.reporting.localization.Localization(__file__, 361, 8), update_375927, *[new_375928, list_375941], **kwargs_375942)
        
        # Getting the type of 'new' (line 362)
        new_375944 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 362, 15), 'new')
        # Assigning a type to the variable 'stypy_return_type' (line 362)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 362, 8), 'stypy_return_type', new_375944)
        
        # ################# End of '__neg__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__neg__' in the type store
        # Getting the type of 'stypy_return_type' (line 356)
        stypy_return_type_375945 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 356, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_375945)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__neg__'
        return stypy_return_type_375945


    @norecursion
    def _mul_scalar(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_mul_scalar'
        module_type_store = module_type_store.open_function_context('_mul_scalar', 364, 4, False)
        # Assigning a type to the variable 'self' (line 365)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 365, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        dok_matrix._mul_scalar.__dict__.__setitem__('stypy_localization', localization)
        dok_matrix._mul_scalar.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        dok_matrix._mul_scalar.__dict__.__setitem__('stypy_type_store', module_type_store)
        dok_matrix._mul_scalar.__dict__.__setitem__('stypy_function_name', 'dok_matrix._mul_scalar')
        dok_matrix._mul_scalar.__dict__.__setitem__('stypy_param_names_list', ['other'])
        dok_matrix._mul_scalar.__dict__.__setitem__('stypy_varargs_param_name', None)
        dok_matrix._mul_scalar.__dict__.__setitem__('stypy_kwargs_param_name', None)
        dok_matrix._mul_scalar.__dict__.__setitem__('stypy_call_defaults', defaults)
        dok_matrix._mul_scalar.__dict__.__setitem__('stypy_call_varargs', varargs)
        dok_matrix._mul_scalar.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        dok_matrix._mul_scalar.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'dok_matrix._mul_scalar', ['other'], None, None, defaults, varargs, kwargs)

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

        
        # Assigning a Call to a Name (line 365):
        
        # Assigning a Call to a Name (line 365):
        
        # Call to upcast_scalar(...): (line 365)
        # Processing the call arguments (line 365)
        # Getting the type of 'self' (line 365)
        self_375947 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 365, 34), 'self', False)
        # Obtaining the member 'dtype' of a type (line 365)
        dtype_375948 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 365, 34), self_375947, 'dtype')
        # Getting the type of 'other' (line 365)
        other_375949 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 365, 46), 'other', False)
        # Processing the call keyword arguments (line 365)
        kwargs_375950 = {}
        # Getting the type of 'upcast_scalar' (line 365)
        upcast_scalar_375946 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 365, 20), 'upcast_scalar', False)
        # Calling upcast_scalar(args, kwargs) (line 365)
        upcast_scalar_call_result_375951 = invoke(stypy.reporting.localization.Localization(__file__, 365, 20), upcast_scalar_375946, *[dtype_375948, other_375949], **kwargs_375950)
        
        # Assigning a type to the variable 'res_dtype' (line 365)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 365, 8), 'res_dtype', upcast_scalar_call_result_375951)
        
        # Assigning a Call to a Name (line 367):
        
        # Assigning a Call to a Name (line 367):
        
        # Call to dok_matrix(...): (line 367)
        # Processing the call arguments (line 367)
        # Getting the type of 'self' (line 367)
        self_375953 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 367, 25), 'self', False)
        # Obtaining the member 'shape' of a type (line 367)
        shape_375954 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 367, 25), self_375953, 'shape')
        # Processing the call keyword arguments (line 367)
        # Getting the type of 'res_dtype' (line 367)
        res_dtype_375955 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 367, 43), 'res_dtype', False)
        keyword_375956 = res_dtype_375955
        kwargs_375957 = {'dtype': keyword_375956}
        # Getting the type of 'dok_matrix' (line 367)
        dok_matrix_375952 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 367, 14), 'dok_matrix', False)
        # Calling dok_matrix(args, kwargs) (line 367)
        dok_matrix_call_result_375958 = invoke(stypy.reporting.localization.Localization(__file__, 367, 14), dok_matrix_375952, *[shape_375954], **kwargs_375957)
        
        # Assigning a type to the variable 'new' (line 367)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 367, 8), 'new', dok_matrix_call_result_375958)
        
        # Call to update(...): (line 368)
        # Processing the call arguments (line 368)
        # Getting the type of 'new' (line 368)
        new_375961 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 368, 20), 'new', False)
        # Calculating generator expression
        module_type_store = module_type_store.open_function_context('list comprehension expression', 368, 26, True)
        # Calculating comprehension expression
        
        # Call to iteritems(...): (line 368)
        # Processing the call arguments (line 368)
        # Getting the type of 'self' (line 368)
        self_375968 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 368, 63), 'self', False)
        # Processing the call keyword arguments (line 368)
        kwargs_375969 = {}
        # Getting the type of 'iteritems' (line 368)
        iteritems_375967 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 368, 53), 'iteritems', False)
        # Calling iteritems(args, kwargs) (line 368)
        iteritems_call_result_375970 = invoke(stypy.reporting.localization.Localization(__file__, 368, 53), iteritems_375967, *[self_375968], **kwargs_375969)
        
        comprehension_375971 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 368, 26), iteritems_call_result_375970)
        # Assigning a type to the variable 'k' (line 368)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 368, 26), 'k', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 368, 26), comprehension_375971))
        # Assigning a type to the variable 'v' (line 368)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 368, 26), 'v', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 368, 26), comprehension_375971))
        
        # Obtaining an instance of the builtin type 'tuple' (line 368)
        tuple_375962 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 368, 27), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 368)
        # Adding element type (line 368)
        # Getting the type of 'k' (line 368)
        k_375963 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 368, 27), 'k', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 368, 27), tuple_375962, k_375963)
        # Adding element type (line 368)
        # Getting the type of 'v' (line 368)
        v_375964 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 368, 30), 'v', False)
        # Getting the type of 'other' (line 368)
        other_375965 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 368, 34), 'other', False)
        # Applying the binary operator '*' (line 368)
        result_mul_375966 = python_operator(stypy.reporting.localization.Localization(__file__, 368, 30), '*', v_375964, other_375965)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 368, 27), tuple_375962, result_mul_375966)
        
        list_375972 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 368, 26), 'list')
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 368, 26), list_375972, tuple_375962)
        # Processing the call keyword arguments (line 368)
        kwargs_375973 = {}
        # Getting the type of 'dict' (line 368)
        dict_375959 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 368, 8), 'dict', False)
        # Obtaining the member 'update' of a type (line 368)
        update_375960 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 368, 8), dict_375959, 'update')
        # Calling update(args, kwargs) (line 368)
        update_call_result_375974 = invoke(stypy.reporting.localization.Localization(__file__, 368, 8), update_375960, *[new_375961, list_375972], **kwargs_375973)
        
        # Getting the type of 'new' (line 369)
        new_375975 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 369, 15), 'new')
        # Assigning a type to the variable 'stypy_return_type' (line 369)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 369, 8), 'stypy_return_type', new_375975)
        
        # ################# End of '_mul_scalar(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_mul_scalar' in the type store
        # Getting the type of 'stypy_return_type' (line 364)
        stypy_return_type_375976 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 364, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_375976)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_mul_scalar'
        return stypy_return_type_375976


    @norecursion
    def _mul_vector(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_mul_vector'
        module_type_store = module_type_store.open_function_context('_mul_vector', 371, 4, False)
        # Assigning a type to the variable 'self' (line 372)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 372, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        dok_matrix._mul_vector.__dict__.__setitem__('stypy_localization', localization)
        dok_matrix._mul_vector.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        dok_matrix._mul_vector.__dict__.__setitem__('stypy_type_store', module_type_store)
        dok_matrix._mul_vector.__dict__.__setitem__('stypy_function_name', 'dok_matrix._mul_vector')
        dok_matrix._mul_vector.__dict__.__setitem__('stypy_param_names_list', ['other'])
        dok_matrix._mul_vector.__dict__.__setitem__('stypy_varargs_param_name', None)
        dok_matrix._mul_vector.__dict__.__setitem__('stypy_kwargs_param_name', None)
        dok_matrix._mul_vector.__dict__.__setitem__('stypy_call_defaults', defaults)
        dok_matrix._mul_vector.__dict__.__setitem__('stypy_call_varargs', varargs)
        dok_matrix._mul_vector.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        dok_matrix._mul_vector.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'dok_matrix._mul_vector', ['other'], None, None, defaults, varargs, kwargs)

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

        
        # Assigning a Call to a Name (line 373):
        
        # Assigning a Call to a Name (line 373):
        
        # Call to zeros(...): (line 373)
        # Processing the call arguments (line 373)
        
        # Obtaining the type of the subscript
        int_375979 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 373, 37), 'int')
        # Getting the type of 'self' (line 373)
        self_375980 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 373, 26), 'self', False)
        # Obtaining the member 'shape' of a type (line 373)
        shape_375981 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 373, 26), self_375980, 'shape')
        # Obtaining the member '__getitem__' of a type (line 373)
        getitem___375982 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 373, 26), shape_375981, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 373)
        subscript_call_result_375983 = invoke(stypy.reporting.localization.Localization(__file__, 373, 26), getitem___375982, int_375979)
        
        # Processing the call keyword arguments (line 373)
        
        # Call to upcast(...): (line 373)
        # Processing the call arguments (line 373)
        # Getting the type of 'self' (line 373)
        self_375985 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 373, 54), 'self', False)
        # Obtaining the member 'dtype' of a type (line 373)
        dtype_375986 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 373, 54), self_375985, 'dtype')
        # Getting the type of 'other' (line 373)
        other_375987 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 373, 66), 'other', False)
        # Obtaining the member 'dtype' of a type (line 373)
        dtype_375988 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 373, 66), other_375987, 'dtype')
        # Processing the call keyword arguments (line 373)
        kwargs_375989 = {}
        # Getting the type of 'upcast' (line 373)
        upcast_375984 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 373, 47), 'upcast', False)
        # Calling upcast(args, kwargs) (line 373)
        upcast_call_result_375990 = invoke(stypy.reporting.localization.Localization(__file__, 373, 47), upcast_375984, *[dtype_375986, dtype_375988], **kwargs_375989)
        
        keyword_375991 = upcast_call_result_375990
        kwargs_375992 = {'dtype': keyword_375991}
        # Getting the type of 'np' (line 373)
        np_375977 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 373, 17), 'np', False)
        # Obtaining the member 'zeros' of a type (line 373)
        zeros_375978 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 373, 17), np_375977, 'zeros')
        # Calling zeros(args, kwargs) (line 373)
        zeros_call_result_375993 = invoke(stypy.reporting.localization.Localization(__file__, 373, 17), zeros_375978, *[subscript_call_result_375983], **kwargs_375992)
        
        # Assigning a type to the variable 'result' (line 373)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 373, 8), 'result', zeros_call_result_375993)
        
        
        # Call to iteritems(...): (line 374)
        # Processing the call arguments (line 374)
        # Getting the type of 'self' (line 374)
        self_375995 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 374, 35), 'self', False)
        # Processing the call keyword arguments (line 374)
        kwargs_375996 = {}
        # Getting the type of 'iteritems' (line 374)
        iteritems_375994 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 374, 25), 'iteritems', False)
        # Calling iteritems(args, kwargs) (line 374)
        iteritems_call_result_375997 = invoke(stypy.reporting.localization.Localization(__file__, 374, 25), iteritems_375994, *[self_375995], **kwargs_375996)
        
        # Testing the type of a for loop iterable (line 374)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 374, 8), iteritems_call_result_375997)
        # Getting the type of the for loop variable (line 374)
        for_loop_var_375998 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 374, 8), iteritems_call_result_375997)
        # Assigning a type to the variable 'i' (line 374)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 374, 8), 'i', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 374, 8), for_loop_var_375998))
        # Assigning a type to the variable 'j' (line 374)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 374, 8), 'j', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 374, 8), for_loop_var_375998))
        # Assigning a type to the variable 'v' (line 374)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 374, 8), 'v', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 374, 8), for_loop_var_375998))
        # SSA begins for a for statement (line 374)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Getting the type of 'result' (line 375)
        result_375999 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 375, 12), 'result')
        
        # Obtaining the type of the subscript
        # Getting the type of 'i' (line 375)
        i_376000 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 375, 19), 'i')
        # Getting the type of 'result' (line 375)
        result_376001 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 375, 12), 'result')
        # Obtaining the member '__getitem__' of a type (line 375)
        getitem___376002 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 375, 12), result_376001, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 375)
        subscript_call_result_376003 = invoke(stypy.reporting.localization.Localization(__file__, 375, 12), getitem___376002, i_376000)
        
        # Getting the type of 'v' (line 375)
        v_376004 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 375, 25), 'v')
        
        # Obtaining the type of the subscript
        # Getting the type of 'j' (line 375)
        j_376005 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 375, 35), 'j')
        # Getting the type of 'other' (line 375)
        other_376006 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 375, 29), 'other')
        # Obtaining the member '__getitem__' of a type (line 375)
        getitem___376007 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 375, 29), other_376006, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 375)
        subscript_call_result_376008 = invoke(stypy.reporting.localization.Localization(__file__, 375, 29), getitem___376007, j_376005)
        
        # Applying the binary operator '*' (line 375)
        result_mul_376009 = python_operator(stypy.reporting.localization.Localization(__file__, 375, 25), '*', v_376004, subscript_call_result_376008)
        
        # Applying the binary operator '+=' (line 375)
        result_iadd_376010 = python_operator(stypy.reporting.localization.Localization(__file__, 375, 12), '+=', subscript_call_result_376003, result_mul_376009)
        # Getting the type of 'result' (line 375)
        result_376011 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 375, 12), 'result')
        # Getting the type of 'i' (line 375)
        i_376012 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 375, 19), 'i')
        # Storing an element on a container (line 375)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 375, 12), result_376011, (i_376012, result_iadd_376010))
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        # Getting the type of 'result' (line 376)
        result_376013 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 376, 15), 'result')
        # Assigning a type to the variable 'stypy_return_type' (line 376)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 376, 8), 'stypy_return_type', result_376013)
        
        # ################# End of '_mul_vector(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_mul_vector' in the type store
        # Getting the type of 'stypy_return_type' (line 371)
        stypy_return_type_376014 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 371, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_376014)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_mul_vector'
        return stypy_return_type_376014


    @norecursion
    def _mul_multivector(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_mul_multivector'
        module_type_store = module_type_store.open_function_context('_mul_multivector', 378, 4, False)
        # Assigning a type to the variable 'self' (line 379)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 379, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        dok_matrix._mul_multivector.__dict__.__setitem__('stypy_localization', localization)
        dok_matrix._mul_multivector.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        dok_matrix._mul_multivector.__dict__.__setitem__('stypy_type_store', module_type_store)
        dok_matrix._mul_multivector.__dict__.__setitem__('stypy_function_name', 'dok_matrix._mul_multivector')
        dok_matrix._mul_multivector.__dict__.__setitem__('stypy_param_names_list', ['other'])
        dok_matrix._mul_multivector.__dict__.__setitem__('stypy_varargs_param_name', None)
        dok_matrix._mul_multivector.__dict__.__setitem__('stypy_kwargs_param_name', None)
        dok_matrix._mul_multivector.__dict__.__setitem__('stypy_call_defaults', defaults)
        dok_matrix._mul_multivector.__dict__.__setitem__('stypy_call_varargs', varargs)
        dok_matrix._mul_multivector.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        dok_matrix._mul_multivector.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'dok_matrix._mul_multivector', ['other'], None, None, defaults, varargs, kwargs)

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

        
        # Assigning a Tuple to a Name (line 380):
        
        # Assigning a Tuple to a Name (line 380):
        
        # Obtaining an instance of the builtin type 'tuple' (line 380)
        tuple_376015 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 380, 24), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 380)
        # Adding element type (line 380)
        
        # Obtaining the type of the subscript
        int_376016 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 380, 35), 'int')
        # Getting the type of 'self' (line 380)
        self_376017 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 380, 24), 'self')
        # Obtaining the member 'shape' of a type (line 380)
        shape_376018 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 380, 24), self_376017, 'shape')
        # Obtaining the member '__getitem__' of a type (line 380)
        getitem___376019 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 380, 24), shape_376018, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 380)
        subscript_call_result_376020 = invoke(stypy.reporting.localization.Localization(__file__, 380, 24), getitem___376019, int_376016)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 380, 24), tuple_376015, subscript_call_result_376020)
        # Adding element type (line 380)
        
        # Obtaining the type of the subscript
        int_376021 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 380, 51), 'int')
        # Getting the type of 'other' (line 380)
        other_376022 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 380, 39), 'other')
        # Obtaining the member 'shape' of a type (line 380)
        shape_376023 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 380, 39), other_376022, 'shape')
        # Obtaining the member '__getitem__' of a type (line 380)
        getitem___376024 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 380, 39), shape_376023, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 380)
        subscript_call_result_376025 = invoke(stypy.reporting.localization.Localization(__file__, 380, 39), getitem___376024, int_376021)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 380, 24), tuple_376015, subscript_call_result_376025)
        
        # Assigning a type to the variable 'result_shape' (line 380)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 380, 8), 'result_shape', tuple_376015)
        
        # Assigning a Call to a Name (line 381):
        
        # Assigning a Call to a Name (line 381):
        
        # Call to upcast(...): (line 381)
        # Processing the call arguments (line 381)
        # Getting the type of 'self' (line 381)
        self_376027 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 381, 30), 'self', False)
        # Obtaining the member 'dtype' of a type (line 381)
        dtype_376028 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 381, 30), self_376027, 'dtype')
        # Getting the type of 'other' (line 381)
        other_376029 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 381, 42), 'other', False)
        # Obtaining the member 'dtype' of a type (line 381)
        dtype_376030 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 381, 42), other_376029, 'dtype')
        # Processing the call keyword arguments (line 381)
        kwargs_376031 = {}
        # Getting the type of 'upcast' (line 381)
        upcast_376026 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 381, 23), 'upcast', False)
        # Calling upcast(args, kwargs) (line 381)
        upcast_call_result_376032 = invoke(stypy.reporting.localization.Localization(__file__, 381, 23), upcast_376026, *[dtype_376028, dtype_376030], **kwargs_376031)
        
        # Assigning a type to the variable 'result_dtype' (line 381)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 381, 8), 'result_dtype', upcast_call_result_376032)
        
        # Assigning a Call to a Name (line 382):
        
        # Assigning a Call to a Name (line 382):
        
        # Call to zeros(...): (line 382)
        # Processing the call arguments (line 382)
        # Getting the type of 'result_shape' (line 382)
        result_shape_376035 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 382, 26), 'result_shape', False)
        # Processing the call keyword arguments (line 382)
        # Getting the type of 'result_dtype' (line 382)
        result_dtype_376036 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 382, 46), 'result_dtype', False)
        keyword_376037 = result_dtype_376036
        kwargs_376038 = {'dtype': keyword_376037}
        # Getting the type of 'np' (line 382)
        np_376033 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 382, 17), 'np', False)
        # Obtaining the member 'zeros' of a type (line 382)
        zeros_376034 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 382, 17), np_376033, 'zeros')
        # Calling zeros(args, kwargs) (line 382)
        zeros_call_result_376039 = invoke(stypy.reporting.localization.Localization(__file__, 382, 17), zeros_376034, *[result_shape_376035], **kwargs_376038)
        
        # Assigning a type to the variable 'result' (line 382)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 382, 8), 'result', zeros_call_result_376039)
        
        
        # Call to iteritems(...): (line 383)
        # Processing the call arguments (line 383)
        # Getting the type of 'self' (line 383)
        self_376041 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 383, 35), 'self', False)
        # Processing the call keyword arguments (line 383)
        kwargs_376042 = {}
        # Getting the type of 'iteritems' (line 383)
        iteritems_376040 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 383, 25), 'iteritems', False)
        # Calling iteritems(args, kwargs) (line 383)
        iteritems_call_result_376043 = invoke(stypy.reporting.localization.Localization(__file__, 383, 25), iteritems_376040, *[self_376041], **kwargs_376042)
        
        # Testing the type of a for loop iterable (line 383)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 383, 8), iteritems_call_result_376043)
        # Getting the type of the for loop variable (line 383)
        for_loop_var_376044 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 383, 8), iteritems_call_result_376043)
        # Assigning a type to the variable 'i' (line 383)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 383, 8), 'i', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 383, 8), for_loop_var_376044))
        # Assigning a type to the variable 'j' (line 383)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 383, 8), 'j', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 383, 8), for_loop_var_376044))
        # Assigning a type to the variable 'v' (line 383)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 383, 8), 'v', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 383, 8), for_loop_var_376044))
        # SSA begins for a for statement (line 383)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Getting the type of 'result' (line 384)
        result_376045 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 384, 12), 'result')
        
        # Obtaining the type of the subscript
        # Getting the type of 'i' (line 384)
        i_376046 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 384, 19), 'i')
        slice_376047 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 384, 12), None, None, None)
        # Getting the type of 'result' (line 384)
        result_376048 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 384, 12), 'result')
        # Obtaining the member '__getitem__' of a type (line 384)
        getitem___376049 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 384, 12), result_376048, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 384)
        subscript_call_result_376050 = invoke(stypy.reporting.localization.Localization(__file__, 384, 12), getitem___376049, (i_376046, slice_376047))
        
        # Getting the type of 'v' (line 384)
        v_376051 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 384, 27), 'v')
        
        # Obtaining the type of the subscript
        # Getting the type of 'j' (line 384)
        j_376052 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 384, 37), 'j')
        slice_376053 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 384, 31), None, None, None)
        # Getting the type of 'other' (line 384)
        other_376054 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 384, 31), 'other')
        # Obtaining the member '__getitem__' of a type (line 384)
        getitem___376055 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 384, 31), other_376054, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 384)
        subscript_call_result_376056 = invoke(stypy.reporting.localization.Localization(__file__, 384, 31), getitem___376055, (j_376052, slice_376053))
        
        # Applying the binary operator '*' (line 384)
        result_mul_376057 = python_operator(stypy.reporting.localization.Localization(__file__, 384, 27), '*', v_376051, subscript_call_result_376056)
        
        # Applying the binary operator '+=' (line 384)
        result_iadd_376058 = python_operator(stypy.reporting.localization.Localization(__file__, 384, 12), '+=', subscript_call_result_376050, result_mul_376057)
        # Getting the type of 'result' (line 384)
        result_376059 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 384, 12), 'result')
        # Getting the type of 'i' (line 384)
        i_376060 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 384, 19), 'i')
        slice_376061 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 384, 12), None, None, None)
        # Storing an element on a container (line 384)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 384, 12), result_376059, ((i_376060, slice_376061), result_iadd_376058))
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        # Getting the type of 'result' (line 385)
        result_376062 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 385, 15), 'result')
        # Assigning a type to the variable 'stypy_return_type' (line 385)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 385, 8), 'stypy_return_type', result_376062)
        
        # ################# End of '_mul_multivector(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_mul_multivector' in the type store
        # Getting the type of 'stypy_return_type' (line 378)
        stypy_return_type_376063 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 378, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_376063)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_mul_multivector'
        return stypy_return_type_376063


    @norecursion
    def __imul__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__imul__'
        module_type_store = module_type_store.open_function_context('__imul__', 387, 4, False)
        # Assigning a type to the variable 'self' (line 388)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 388, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        dok_matrix.__imul__.__dict__.__setitem__('stypy_localization', localization)
        dok_matrix.__imul__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        dok_matrix.__imul__.__dict__.__setitem__('stypy_type_store', module_type_store)
        dok_matrix.__imul__.__dict__.__setitem__('stypy_function_name', 'dok_matrix.__imul__')
        dok_matrix.__imul__.__dict__.__setitem__('stypy_param_names_list', ['other'])
        dok_matrix.__imul__.__dict__.__setitem__('stypy_varargs_param_name', None)
        dok_matrix.__imul__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        dok_matrix.__imul__.__dict__.__setitem__('stypy_call_defaults', defaults)
        dok_matrix.__imul__.__dict__.__setitem__('stypy_call_varargs', varargs)
        dok_matrix.__imul__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        dok_matrix.__imul__.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'dok_matrix.__imul__', ['other'], None, None, defaults, varargs, kwargs)

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

        
        
        # Call to isscalarlike(...): (line 388)
        # Processing the call arguments (line 388)
        # Getting the type of 'other' (line 388)
        other_376065 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 388, 24), 'other', False)
        # Processing the call keyword arguments (line 388)
        kwargs_376066 = {}
        # Getting the type of 'isscalarlike' (line 388)
        isscalarlike_376064 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 388, 11), 'isscalarlike', False)
        # Calling isscalarlike(args, kwargs) (line 388)
        isscalarlike_call_result_376067 = invoke(stypy.reporting.localization.Localization(__file__, 388, 11), isscalarlike_376064, *[other_376065], **kwargs_376066)
        
        # Testing the type of an if condition (line 388)
        if_condition_376068 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 388, 8), isscalarlike_call_result_376067)
        # Assigning a type to the variable 'if_condition_376068' (line 388)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 388, 8), 'if_condition_376068', if_condition_376068)
        # SSA begins for if statement (line 388)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to update(...): (line 389)
        # Processing the call arguments (line 389)
        # Getting the type of 'self' (line 389)
        self_376071 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 389, 24), 'self', False)
        # Calculating generator expression
        module_type_store = module_type_store.open_function_context('list comprehension expression', 389, 31, True)
        # Calculating comprehension expression
        
        # Call to iteritems(...): (line 389)
        # Processing the call arguments (line 389)
        # Getting the type of 'self' (line 389)
        self_376078 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 389, 68), 'self', False)
        # Processing the call keyword arguments (line 389)
        kwargs_376079 = {}
        # Getting the type of 'iteritems' (line 389)
        iteritems_376077 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 389, 58), 'iteritems', False)
        # Calling iteritems(args, kwargs) (line 389)
        iteritems_call_result_376080 = invoke(stypy.reporting.localization.Localization(__file__, 389, 58), iteritems_376077, *[self_376078], **kwargs_376079)
        
        comprehension_376081 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 389, 31), iteritems_call_result_376080)
        # Assigning a type to the variable 'k' (line 389)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 389, 31), 'k', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 389, 31), comprehension_376081))
        # Assigning a type to the variable 'v' (line 389)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 389, 31), 'v', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 389, 31), comprehension_376081))
        
        # Obtaining an instance of the builtin type 'tuple' (line 389)
        tuple_376072 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 389, 32), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 389)
        # Adding element type (line 389)
        # Getting the type of 'k' (line 389)
        k_376073 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 389, 32), 'k', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 389, 32), tuple_376072, k_376073)
        # Adding element type (line 389)
        # Getting the type of 'v' (line 389)
        v_376074 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 389, 35), 'v', False)
        # Getting the type of 'other' (line 389)
        other_376075 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 389, 39), 'other', False)
        # Applying the binary operator '*' (line 389)
        result_mul_376076 = python_operator(stypy.reporting.localization.Localization(__file__, 389, 35), '*', v_376074, other_376075)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 389, 32), tuple_376072, result_mul_376076)
        
        list_376082 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 389, 31), 'list')
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 389, 31), list_376082, tuple_376072)
        # Processing the call keyword arguments (line 389)
        kwargs_376083 = {}
        # Getting the type of 'dict' (line 389)
        dict_376069 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 389, 12), 'dict', False)
        # Obtaining the member 'update' of a type (line 389)
        update_376070 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 389, 12), dict_376069, 'update')
        # Calling update(args, kwargs) (line 389)
        update_call_result_376084 = invoke(stypy.reporting.localization.Localization(__file__, 389, 12), update_376070, *[self_376071, list_376082], **kwargs_376083)
        
        # Getting the type of 'self' (line 390)
        self_376085 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 390, 19), 'self')
        # Assigning a type to the variable 'stypy_return_type' (line 390)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 390, 12), 'stypy_return_type', self_376085)
        # SSA join for if statement (line 388)
        module_type_store = module_type_store.join_ssa_context()
        
        # Getting the type of 'NotImplemented' (line 391)
        NotImplemented_376086 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 391, 15), 'NotImplemented')
        # Assigning a type to the variable 'stypy_return_type' (line 391)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 391, 8), 'stypy_return_type', NotImplemented_376086)
        
        # ################# End of '__imul__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__imul__' in the type store
        # Getting the type of 'stypy_return_type' (line 387)
        stypy_return_type_376087 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 387, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_376087)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__imul__'
        return stypy_return_type_376087


    @norecursion
    def __truediv__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__truediv__'
        module_type_store = module_type_store.open_function_context('__truediv__', 393, 4, False)
        # Assigning a type to the variable 'self' (line 394)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 394, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        dok_matrix.__truediv__.__dict__.__setitem__('stypy_localization', localization)
        dok_matrix.__truediv__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        dok_matrix.__truediv__.__dict__.__setitem__('stypy_type_store', module_type_store)
        dok_matrix.__truediv__.__dict__.__setitem__('stypy_function_name', 'dok_matrix.__truediv__')
        dok_matrix.__truediv__.__dict__.__setitem__('stypy_param_names_list', ['other'])
        dok_matrix.__truediv__.__dict__.__setitem__('stypy_varargs_param_name', None)
        dok_matrix.__truediv__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        dok_matrix.__truediv__.__dict__.__setitem__('stypy_call_defaults', defaults)
        dok_matrix.__truediv__.__dict__.__setitem__('stypy_call_varargs', varargs)
        dok_matrix.__truediv__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        dok_matrix.__truediv__.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'dok_matrix.__truediv__', ['other'], None, None, defaults, varargs, kwargs)

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

        
        
        # Call to isscalarlike(...): (line 394)
        # Processing the call arguments (line 394)
        # Getting the type of 'other' (line 394)
        other_376089 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 394, 24), 'other', False)
        # Processing the call keyword arguments (line 394)
        kwargs_376090 = {}
        # Getting the type of 'isscalarlike' (line 394)
        isscalarlike_376088 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 394, 11), 'isscalarlike', False)
        # Calling isscalarlike(args, kwargs) (line 394)
        isscalarlike_call_result_376091 = invoke(stypy.reporting.localization.Localization(__file__, 394, 11), isscalarlike_376088, *[other_376089], **kwargs_376090)
        
        # Testing the type of an if condition (line 394)
        if_condition_376092 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 394, 8), isscalarlike_call_result_376091)
        # Assigning a type to the variable 'if_condition_376092' (line 394)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 394, 8), 'if_condition_376092', if_condition_376092)
        # SSA begins for if statement (line 394)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 395):
        
        # Assigning a Call to a Name (line 395):
        
        # Call to upcast_scalar(...): (line 395)
        # Processing the call arguments (line 395)
        # Getting the type of 'self' (line 395)
        self_376094 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 395, 38), 'self', False)
        # Obtaining the member 'dtype' of a type (line 395)
        dtype_376095 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 395, 38), self_376094, 'dtype')
        # Getting the type of 'other' (line 395)
        other_376096 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 395, 50), 'other', False)
        # Processing the call keyword arguments (line 395)
        kwargs_376097 = {}
        # Getting the type of 'upcast_scalar' (line 395)
        upcast_scalar_376093 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 395, 24), 'upcast_scalar', False)
        # Calling upcast_scalar(args, kwargs) (line 395)
        upcast_scalar_call_result_376098 = invoke(stypy.reporting.localization.Localization(__file__, 395, 24), upcast_scalar_376093, *[dtype_376095, other_376096], **kwargs_376097)
        
        # Assigning a type to the variable 'res_dtype' (line 395)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 395, 12), 'res_dtype', upcast_scalar_call_result_376098)
        
        # Assigning a Call to a Name (line 396):
        
        # Assigning a Call to a Name (line 396):
        
        # Call to dok_matrix(...): (line 396)
        # Processing the call arguments (line 396)
        # Getting the type of 'self' (line 396)
        self_376100 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 396, 29), 'self', False)
        # Obtaining the member 'shape' of a type (line 396)
        shape_376101 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 396, 29), self_376100, 'shape')
        # Processing the call keyword arguments (line 396)
        # Getting the type of 'res_dtype' (line 396)
        res_dtype_376102 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 396, 47), 'res_dtype', False)
        keyword_376103 = res_dtype_376102
        kwargs_376104 = {'dtype': keyword_376103}
        # Getting the type of 'dok_matrix' (line 396)
        dok_matrix_376099 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 396, 18), 'dok_matrix', False)
        # Calling dok_matrix(args, kwargs) (line 396)
        dok_matrix_call_result_376105 = invoke(stypy.reporting.localization.Localization(__file__, 396, 18), dok_matrix_376099, *[shape_376101], **kwargs_376104)
        
        # Assigning a type to the variable 'new' (line 396)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 396, 12), 'new', dok_matrix_call_result_376105)
        
        # Call to update(...): (line 397)
        # Processing the call arguments (line 397)
        # Getting the type of 'new' (line 397)
        new_376108 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 397, 24), 'new', False)
        # Calculating generator expression
        module_type_store = module_type_store.open_function_context('list comprehension expression', 397, 30, True)
        # Calculating comprehension expression
        
        # Call to iteritems(...): (line 397)
        # Processing the call arguments (line 397)
        # Getting the type of 'self' (line 397)
        self_376115 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 397, 67), 'self', False)
        # Processing the call keyword arguments (line 397)
        kwargs_376116 = {}
        # Getting the type of 'iteritems' (line 397)
        iteritems_376114 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 397, 57), 'iteritems', False)
        # Calling iteritems(args, kwargs) (line 397)
        iteritems_call_result_376117 = invoke(stypy.reporting.localization.Localization(__file__, 397, 57), iteritems_376114, *[self_376115], **kwargs_376116)
        
        comprehension_376118 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 397, 30), iteritems_call_result_376117)
        # Assigning a type to the variable 'k' (line 397)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 397, 30), 'k', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 397, 30), comprehension_376118))
        # Assigning a type to the variable 'v' (line 397)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 397, 30), 'v', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 397, 30), comprehension_376118))
        
        # Obtaining an instance of the builtin type 'tuple' (line 397)
        tuple_376109 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 397, 31), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 397)
        # Adding element type (line 397)
        # Getting the type of 'k' (line 397)
        k_376110 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 397, 31), 'k', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 397, 31), tuple_376109, k_376110)
        # Adding element type (line 397)
        # Getting the type of 'v' (line 397)
        v_376111 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 397, 34), 'v', False)
        # Getting the type of 'other' (line 397)
        other_376112 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 397, 38), 'other', False)
        # Applying the binary operator 'div' (line 397)
        result_div_376113 = python_operator(stypy.reporting.localization.Localization(__file__, 397, 34), 'div', v_376111, other_376112)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 397, 31), tuple_376109, result_div_376113)
        
        list_376119 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 397, 30), 'list')
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 397, 30), list_376119, tuple_376109)
        # Processing the call keyword arguments (line 397)
        kwargs_376120 = {}
        # Getting the type of 'dict' (line 397)
        dict_376106 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 397, 12), 'dict', False)
        # Obtaining the member 'update' of a type (line 397)
        update_376107 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 397, 12), dict_376106, 'update')
        # Calling update(args, kwargs) (line 397)
        update_call_result_376121 = invoke(stypy.reporting.localization.Localization(__file__, 397, 12), update_376107, *[new_376108, list_376119], **kwargs_376120)
        
        # Getting the type of 'new' (line 398)
        new_376122 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 398, 19), 'new')
        # Assigning a type to the variable 'stypy_return_type' (line 398)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 398, 12), 'stypy_return_type', new_376122)
        # SSA join for if statement (line 394)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to tocsr(...): (line 399)
        # Processing the call keyword arguments (line 399)
        kwargs_376125 = {}
        # Getting the type of 'self' (line 399)
        self_376123 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 399, 15), 'self', False)
        # Obtaining the member 'tocsr' of a type (line 399)
        tocsr_376124 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 399, 15), self_376123, 'tocsr')
        # Calling tocsr(args, kwargs) (line 399)
        tocsr_call_result_376126 = invoke(stypy.reporting.localization.Localization(__file__, 399, 15), tocsr_376124, *[], **kwargs_376125)
        
        # Getting the type of 'other' (line 399)
        other_376127 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 399, 30), 'other')
        # Applying the binary operator 'div' (line 399)
        result_div_376128 = python_operator(stypy.reporting.localization.Localization(__file__, 399, 15), 'div', tocsr_call_result_376126, other_376127)
        
        # Assigning a type to the variable 'stypy_return_type' (line 399)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 399, 8), 'stypy_return_type', result_div_376128)
        
        # ################# End of '__truediv__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__truediv__' in the type store
        # Getting the type of 'stypy_return_type' (line 393)
        stypy_return_type_376129 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 393, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_376129)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__truediv__'
        return stypy_return_type_376129


    @norecursion
    def __itruediv__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__itruediv__'
        module_type_store = module_type_store.open_function_context('__itruediv__', 401, 4, False)
        # Assigning a type to the variable 'self' (line 402)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 402, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        dok_matrix.__itruediv__.__dict__.__setitem__('stypy_localization', localization)
        dok_matrix.__itruediv__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        dok_matrix.__itruediv__.__dict__.__setitem__('stypy_type_store', module_type_store)
        dok_matrix.__itruediv__.__dict__.__setitem__('stypy_function_name', 'dok_matrix.__itruediv__')
        dok_matrix.__itruediv__.__dict__.__setitem__('stypy_param_names_list', ['other'])
        dok_matrix.__itruediv__.__dict__.__setitem__('stypy_varargs_param_name', None)
        dok_matrix.__itruediv__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        dok_matrix.__itruediv__.__dict__.__setitem__('stypy_call_defaults', defaults)
        dok_matrix.__itruediv__.__dict__.__setitem__('stypy_call_varargs', varargs)
        dok_matrix.__itruediv__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        dok_matrix.__itruediv__.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'dok_matrix.__itruediv__', ['other'], None, None, defaults, varargs, kwargs)

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

        
        
        # Call to isscalarlike(...): (line 402)
        # Processing the call arguments (line 402)
        # Getting the type of 'other' (line 402)
        other_376131 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 402, 24), 'other', False)
        # Processing the call keyword arguments (line 402)
        kwargs_376132 = {}
        # Getting the type of 'isscalarlike' (line 402)
        isscalarlike_376130 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 402, 11), 'isscalarlike', False)
        # Calling isscalarlike(args, kwargs) (line 402)
        isscalarlike_call_result_376133 = invoke(stypy.reporting.localization.Localization(__file__, 402, 11), isscalarlike_376130, *[other_376131], **kwargs_376132)
        
        # Testing the type of an if condition (line 402)
        if_condition_376134 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 402, 8), isscalarlike_call_result_376133)
        # Assigning a type to the variable 'if_condition_376134' (line 402)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 402, 8), 'if_condition_376134', if_condition_376134)
        # SSA begins for if statement (line 402)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to update(...): (line 403)
        # Processing the call arguments (line 403)
        # Getting the type of 'self' (line 403)
        self_376137 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 403, 24), 'self', False)
        # Calculating generator expression
        module_type_store = module_type_store.open_function_context('list comprehension expression', 403, 31, True)
        # Calculating comprehension expression
        
        # Call to iteritems(...): (line 403)
        # Processing the call arguments (line 403)
        # Getting the type of 'self' (line 403)
        self_376144 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 403, 68), 'self', False)
        # Processing the call keyword arguments (line 403)
        kwargs_376145 = {}
        # Getting the type of 'iteritems' (line 403)
        iteritems_376143 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 403, 58), 'iteritems', False)
        # Calling iteritems(args, kwargs) (line 403)
        iteritems_call_result_376146 = invoke(stypy.reporting.localization.Localization(__file__, 403, 58), iteritems_376143, *[self_376144], **kwargs_376145)
        
        comprehension_376147 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 403, 31), iteritems_call_result_376146)
        # Assigning a type to the variable 'k' (line 403)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 403, 31), 'k', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 403, 31), comprehension_376147))
        # Assigning a type to the variable 'v' (line 403)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 403, 31), 'v', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 403, 31), comprehension_376147))
        
        # Obtaining an instance of the builtin type 'tuple' (line 403)
        tuple_376138 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 403, 32), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 403)
        # Adding element type (line 403)
        # Getting the type of 'k' (line 403)
        k_376139 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 403, 32), 'k', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 403, 32), tuple_376138, k_376139)
        # Adding element type (line 403)
        # Getting the type of 'v' (line 403)
        v_376140 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 403, 35), 'v', False)
        # Getting the type of 'other' (line 403)
        other_376141 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 403, 39), 'other', False)
        # Applying the binary operator 'div' (line 403)
        result_div_376142 = python_operator(stypy.reporting.localization.Localization(__file__, 403, 35), 'div', v_376140, other_376141)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 403, 32), tuple_376138, result_div_376142)
        
        list_376148 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 403, 31), 'list')
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 403, 31), list_376148, tuple_376138)
        # Processing the call keyword arguments (line 403)
        kwargs_376149 = {}
        # Getting the type of 'dict' (line 403)
        dict_376135 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 403, 12), 'dict', False)
        # Obtaining the member 'update' of a type (line 403)
        update_376136 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 403, 12), dict_376135, 'update')
        # Calling update(args, kwargs) (line 403)
        update_call_result_376150 = invoke(stypy.reporting.localization.Localization(__file__, 403, 12), update_376136, *[self_376137, list_376148], **kwargs_376149)
        
        # Getting the type of 'self' (line 404)
        self_376151 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 404, 19), 'self')
        # Assigning a type to the variable 'stypy_return_type' (line 404)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 404, 12), 'stypy_return_type', self_376151)
        # SSA join for if statement (line 402)
        module_type_store = module_type_store.join_ssa_context()
        
        # Getting the type of 'NotImplemented' (line 405)
        NotImplemented_376152 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 405, 15), 'NotImplemented')
        # Assigning a type to the variable 'stypy_return_type' (line 405)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 405, 8), 'stypy_return_type', NotImplemented_376152)
        
        # ################# End of '__itruediv__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__itruediv__' in the type store
        # Getting the type of 'stypy_return_type' (line 401)
        stypy_return_type_376153 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 401, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_376153)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__itruediv__'
        return stypy_return_type_376153


    @norecursion
    def __reduce__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__reduce__'
        module_type_store = module_type_store.open_function_context('__reduce__', 407, 4, False)
        # Assigning a type to the variable 'self' (line 408)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 408, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        dok_matrix.__reduce__.__dict__.__setitem__('stypy_localization', localization)
        dok_matrix.__reduce__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        dok_matrix.__reduce__.__dict__.__setitem__('stypy_type_store', module_type_store)
        dok_matrix.__reduce__.__dict__.__setitem__('stypy_function_name', 'dok_matrix.__reduce__')
        dok_matrix.__reduce__.__dict__.__setitem__('stypy_param_names_list', [])
        dok_matrix.__reduce__.__dict__.__setitem__('stypy_varargs_param_name', None)
        dok_matrix.__reduce__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        dok_matrix.__reduce__.__dict__.__setitem__('stypy_call_defaults', defaults)
        dok_matrix.__reduce__.__dict__.__setitem__('stypy_call_varargs', varargs)
        dok_matrix.__reduce__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        dok_matrix.__reduce__.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'dok_matrix.__reduce__', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__reduce__', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__reduce__(...)' code ##################

        
        # Call to __reduce__(...): (line 411)
        # Processing the call arguments (line 411)
        # Getting the type of 'self' (line 411)
        self_376156 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 411, 31), 'self', False)
        # Processing the call keyword arguments (line 411)
        kwargs_376157 = {}
        # Getting the type of 'dict' (line 411)
        dict_376154 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 411, 15), 'dict', False)
        # Obtaining the member '__reduce__' of a type (line 411)
        reduce___376155 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 411, 15), dict_376154, '__reduce__')
        # Calling __reduce__(args, kwargs) (line 411)
        reduce___call_result_376158 = invoke(stypy.reporting.localization.Localization(__file__, 411, 15), reduce___376155, *[self_376156], **kwargs_376157)
        
        # Assigning a type to the variable 'stypy_return_type' (line 411)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 411, 8), 'stypy_return_type', reduce___call_result_376158)
        
        # ################# End of '__reduce__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__reduce__' in the type store
        # Getting the type of 'stypy_return_type' (line 407)
        stypy_return_type_376159 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 407, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_376159)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__reduce__'
        return stypy_return_type_376159


    @norecursion
    def transpose(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 417)
        None_376160 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 417, 29), 'None')
        # Getting the type of 'False' (line 417)
        False_376161 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 417, 40), 'False')
        defaults = [None_376160, False_376161]
        # Create a new context for function 'transpose'
        module_type_store = module_type_store.open_function_context('transpose', 417, 4, False)
        # Assigning a type to the variable 'self' (line 418)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 418, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        dok_matrix.transpose.__dict__.__setitem__('stypy_localization', localization)
        dok_matrix.transpose.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        dok_matrix.transpose.__dict__.__setitem__('stypy_type_store', module_type_store)
        dok_matrix.transpose.__dict__.__setitem__('stypy_function_name', 'dok_matrix.transpose')
        dok_matrix.transpose.__dict__.__setitem__('stypy_param_names_list', ['axes', 'copy'])
        dok_matrix.transpose.__dict__.__setitem__('stypy_varargs_param_name', None)
        dok_matrix.transpose.__dict__.__setitem__('stypy_kwargs_param_name', None)
        dok_matrix.transpose.__dict__.__setitem__('stypy_call_defaults', defaults)
        dok_matrix.transpose.__dict__.__setitem__('stypy_call_varargs', varargs)
        dok_matrix.transpose.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        dok_matrix.transpose.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'dok_matrix.transpose', ['axes', 'copy'], None, None, defaults, varargs, kwargs)

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

        
        # Type idiom detected: calculating its left and rigth part (line 418)
        # Getting the type of 'axes' (line 418)
        axes_376162 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 418, 8), 'axes')
        # Getting the type of 'None' (line 418)
        None_376163 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 418, 23), 'None')
        
        (may_be_376164, more_types_in_union_376165) = may_not_be_none(axes_376162, None_376163)

        if may_be_376164:

            if more_types_in_union_376165:
                # Runtime conditional SSA (line 418)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Call to ValueError(...): (line 419)
            # Processing the call arguments (line 419)
            str_376167 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 419, 29), 'str', "Sparse matrices do not support an 'axes' parameter because swapping dimensions is the only logical permutation.")
            # Processing the call keyword arguments (line 419)
            kwargs_376168 = {}
            # Getting the type of 'ValueError' (line 419)
            ValueError_376166 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 419, 18), 'ValueError', False)
            # Calling ValueError(args, kwargs) (line 419)
            ValueError_call_result_376169 = invoke(stypy.reporting.localization.Localization(__file__, 419, 18), ValueError_376166, *[str_376167], **kwargs_376168)
            
            ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 419, 12), ValueError_call_result_376169, 'raise parameter', BaseException)

            if more_types_in_union_376165:
                # SSA join for if statement (line 418)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Assigning a Attribute to a Tuple (line 423):
        
        # Assigning a Subscript to a Name (line 423):
        
        # Obtaining the type of the subscript
        int_376170 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 423, 8), 'int')
        # Getting the type of 'self' (line 423)
        self_376171 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 423, 15), 'self')
        # Obtaining the member 'shape' of a type (line 423)
        shape_376172 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 423, 15), self_376171, 'shape')
        # Obtaining the member '__getitem__' of a type (line 423)
        getitem___376173 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 423, 8), shape_376172, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 423)
        subscript_call_result_376174 = invoke(stypy.reporting.localization.Localization(__file__, 423, 8), getitem___376173, int_376170)
        
        # Assigning a type to the variable 'tuple_var_assignment_374370' (line 423)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 423, 8), 'tuple_var_assignment_374370', subscript_call_result_376174)
        
        # Assigning a Subscript to a Name (line 423):
        
        # Obtaining the type of the subscript
        int_376175 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 423, 8), 'int')
        # Getting the type of 'self' (line 423)
        self_376176 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 423, 15), 'self')
        # Obtaining the member 'shape' of a type (line 423)
        shape_376177 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 423, 15), self_376176, 'shape')
        # Obtaining the member '__getitem__' of a type (line 423)
        getitem___376178 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 423, 8), shape_376177, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 423)
        subscript_call_result_376179 = invoke(stypy.reporting.localization.Localization(__file__, 423, 8), getitem___376178, int_376175)
        
        # Assigning a type to the variable 'tuple_var_assignment_374371' (line 423)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 423, 8), 'tuple_var_assignment_374371', subscript_call_result_376179)
        
        # Assigning a Name to a Name (line 423):
        # Getting the type of 'tuple_var_assignment_374370' (line 423)
        tuple_var_assignment_374370_376180 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 423, 8), 'tuple_var_assignment_374370')
        # Assigning a type to the variable 'M' (line 423)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 423, 8), 'M', tuple_var_assignment_374370_376180)
        
        # Assigning a Name to a Name (line 423):
        # Getting the type of 'tuple_var_assignment_374371' (line 423)
        tuple_var_assignment_374371_376181 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 423, 8), 'tuple_var_assignment_374371')
        # Assigning a type to the variable 'N' (line 423)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 423, 11), 'N', tuple_var_assignment_374371_376181)
        
        # Assigning a Call to a Name (line 424):
        
        # Assigning a Call to a Name (line 424):
        
        # Call to dok_matrix(...): (line 424)
        # Processing the call arguments (line 424)
        
        # Obtaining an instance of the builtin type 'tuple' (line 424)
        tuple_376183 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 424, 26), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 424)
        # Adding element type (line 424)
        # Getting the type of 'N' (line 424)
        N_376184 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 424, 26), 'N', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 424, 26), tuple_376183, N_376184)
        # Adding element type (line 424)
        # Getting the type of 'M' (line 424)
        M_376185 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 424, 29), 'M', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 424, 26), tuple_376183, M_376185)
        
        # Processing the call keyword arguments (line 424)
        # Getting the type of 'self' (line 424)
        self_376186 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 424, 39), 'self', False)
        # Obtaining the member 'dtype' of a type (line 424)
        dtype_376187 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 424, 39), self_376186, 'dtype')
        keyword_376188 = dtype_376187
        # Getting the type of 'copy' (line 424)
        copy_376189 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 424, 56), 'copy', False)
        keyword_376190 = copy_376189
        kwargs_376191 = {'dtype': keyword_376188, 'copy': keyword_376190}
        # Getting the type of 'dok_matrix' (line 424)
        dok_matrix_376182 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 424, 14), 'dok_matrix', False)
        # Calling dok_matrix(args, kwargs) (line 424)
        dok_matrix_call_result_376192 = invoke(stypy.reporting.localization.Localization(__file__, 424, 14), dok_matrix_376182, *[tuple_376183], **kwargs_376191)
        
        # Assigning a type to the variable 'new' (line 424)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 424, 8), 'new', dok_matrix_call_result_376192)
        
        # Call to update(...): (line 425)
        # Processing the call arguments (line 425)
        # Getting the type of 'new' (line 425)
        new_376195 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 425, 20), 'new', False)
        # Calculating generator expression
        module_type_store = module_type_store.open_function_context('list comprehension expression', 425, 26, True)
        # Calculating comprehension expression
        
        # Call to iteritems(...): (line 426)
        # Processing the call arguments (line 426)
        # Getting the type of 'self' (line 426)
        self_376202 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 426, 62), 'self', False)
        # Processing the call keyword arguments (line 426)
        kwargs_376203 = {}
        # Getting the type of 'iteritems' (line 426)
        iteritems_376201 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 426, 52), 'iteritems', False)
        # Calling iteritems(args, kwargs) (line 426)
        iteritems_call_result_376204 = invoke(stypy.reporting.localization.Localization(__file__, 426, 52), iteritems_376201, *[self_376202], **kwargs_376203)
        
        comprehension_376205 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 425, 26), iteritems_call_result_376204)
        # Assigning a type to the variable 'left' (line 425)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 425, 26), 'left', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 425, 26), comprehension_376205))
        # Assigning a type to the variable 'right' (line 425)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 425, 26), 'right', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 425, 26), comprehension_376205))
        # Assigning a type to the variable 'val' (line 425)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 425, 26), 'val', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 425, 26), comprehension_376205))
        
        # Obtaining an instance of the builtin type 'tuple' (line 425)
        tuple_376196 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 425, 27), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 425)
        # Adding element type (line 425)
        
        # Obtaining an instance of the builtin type 'tuple' (line 425)
        tuple_376197 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 425, 28), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 425)
        # Adding element type (line 425)
        # Getting the type of 'right' (line 425)
        right_376198 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 425, 28), 'right', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 425, 28), tuple_376197, right_376198)
        # Adding element type (line 425)
        # Getting the type of 'left' (line 425)
        left_376199 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 425, 35), 'left', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 425, 28), tuple_376197, left_376199)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 425, 27), tuple_376196, tuple_376197)
        # Adding element type (line 425)
        # Getting the type of 'val' (line 425)
        val_376200 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 425, 42), 'val', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 425, 27), tuple_376196, val_376200)
        
        list_376207 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 425, 26), 'list')
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 425, 26), list_376207, tuple_376196)
        # Processing the call keyword arguments (line 425)
        kwargs_376208 = {}
        # Getting the type of 'dict' (line 425)
        dict_376193 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 425, 8), 'dict', False)
        # Obtaining the member 'update' of a type (line 425)
        update_376194 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 425, 8), dict_376193, 'update')
        # Calling update(args, kwargs) (line 425)
        update_call_result_376209 = invoke(stypy.reporting.localization.Localization(__file__, 425, 8), update_376194, *[new_376195, list_376207], **kwargs_376208)
        
        # Getting the type of 'new' (line 427)
        new_376210 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 427, 15), 'new')
        # Assigning a type to the variable 'stypy_return_type' (line 427)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 427, 8), 'stypy_return_type', new_376210)
        
        # ################# End of 'transpose(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'transpose' in the type store
        # Getting the type of 'stypy_return_type' (line 417)
        stypy_return_type_376211 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 417, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_376211)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'transpose'
        return stypy_return_type_376211

    
    # Assigning a Attribute to a Attribute (line 429):

    @norecursion
    def conjtransp(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'conjtransp'
        module_type_store = module_type_store.open_function_context('conjtransp', 431, 4, False)
        # Assigning a type to the variable 'self' (line 432)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 432, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        dok_matrix.conjtransp.__dict__.__setitem__('stypy_localization', localization)
        dok_matrix.conjtransp.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        dok_matrix.conjtransp.__dict__.__setitem__('stypy_type_store', module_type_store)
        dok_matrix.conjtransp.__dict__.__setitem__('stypy_function_name', 'dok_matrix.conjtransp')
        dok_matrix.conjtransp.__dict__.__setitem__('stypy_param_names_list', [])
        dok_matrix.conjtransp.__dict__.__setitem__('stypy_varargs_param_name', None)
        dok_matrix.conjtransp.__dict__.__setitem__('stypy_kwargs_param_name', None)
        dok_matrix.conjtransp.__dict__.__setitem__('stypy_call_defaults', defaults)
        dok_matrix.conjtransp.__dict__.__setitem__('stypy_call_varargs', varargs)
        dok_matrix.conjtransp.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        dok_matrix.conjtransp.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'dok_matrix.conjtransp', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'conjtransp', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'conjtransp(...)' code ##################

        str_376212 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 432, 8), 'str', 'Return the conjugate transpose.')
        
        # Assigning a Attribute to a Tuple (line 433):
        
        # Assigning a Subscript to a Name (line 433):
        
        # Obtaining the type of the subscript
        int_376213 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 433, 8), 'int')
        # Getting the type of 'self' (line 433)
        self_376214 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 433, 15), 'self')
        # Obtaining the member 'shape' of a type (line 433)
        shape_376215 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 433, 15), self_376214, 'shape')
        # Obtaining the member '__getitem__' of a type (line 433)
        getitem___376216 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 433, 8), shape_376215, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 433)
        subscript_call_result_376217 = invoke(stypy.reporting.localization.Localization(__file__, 433, 8), getitem___376216, int_376213)
        
        # Assigning a type to the variable 'tuple_var_assignment_374372' (line 433)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 433, 8), 'tuple_var_assignment_374372', subscript_call_result_376217)
        
        # Assigning a Subscript to a Name (line 433):
        
        # Obtaining the type of the subscript
        int_376218 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 433, 8), 'int')
        # Getting the type of 'self' (line 433)
        self_376219 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 433, 15), 'self')
        # Obtaining the member 'shape' of a type (line 433)
        shape_376220 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 433, 15), self_376219, 'shape')
        # Obtaining the member '__getitem__' of a type (line 433)
        getitem___376221 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 433, 8), shape_376220, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 433)
        subscript_call_result_376222 = invoke(stypy.reporting.localization.Localization(__file__, 433, 8), getitem___376221, int_376218)
        
        # Assigning a type to the variable 'tuple_var_assignment_374373' (line 433)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 433, 8), 'tuple_var_assignment_374373', subscript_call_result_376222)
        
        # Assigning a Name to a Name (line 433):
        # Getting the type of 'tuple_var_assignment_374372' (line 433)
        tuple_var_assignment_374372_376223 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 433, 8), 'tuple_var_assignment_374372')
        # Assigning a type to the variable 'M' (line 433)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 433, 8), 'M', tuple_var_assignment_374372_376223)
        
        # Assigning a Name to a Name (line 433):
        # Getting the type of 'tuple_var_assignment_374373' (line 433)
        tuple_var_assignment_374373_376224 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 433, 8), 'tuple_var_assignment_374373')
        # Assigning a type to the variable 'N' (line 433)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 433, 11), 'N', tuple_var_assignment_374373_376224)
        
        # Assigning a Call to a Name (line 434):
        
        # Assigning a Call to a Name (line 434):
        
        # Call to dok_matrix(...): (line 434)
        # Processing the call arguments (line 434)
        
        # Obtaining an instance of the builtin type 'tuple' (line 434)
        tuple_376226 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 434, 26), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 434)
        # Adding element type (line 434)
        # Getting the type of 'N' (line 434)
        N_376227 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 434, 26), 'N', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 434, 26), tuple_376226, N_376227)
        # Adding element type (line 434)
        # Getting the type of 'M' (line 434)
        M_376228 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 434, 29), 'M', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 434, 26), tuple_376226, M_376228)
        
        # Processing the call keyword arguments (line 434)
        # Getting the type of 'self' (line 434)
        self_376229 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 434, 39), 'self', False)
        # Obtaining the member 'dtype' of a type (line 434)
        dtype_376230 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 434, 39), self_376229, 'dtype')
        keyword_376231 = dtype_376230
        kwargs_376232 = {'dtype': keyword_376231}
        # Getting the type of 'dok_matrix' (line 434)
        dok_matrix_376225 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 434, 14), 'dok_matrix', False)
        # Calling dok_matrix(args, kwargs) (line 434)
        dok_matrix_call_result_376233 = invoke(stypy.reporting.localization.Localization(__file__, 434, 14), dok_matrix_376225, *[tuple_376226], **kwargs_376232)
        
        # Assigning a type to the variable 'new' (line 434)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 434, 8), 'new', dok_matrix_call_result_376233)
        
        # Call to update(...): (line 435)
        # Processing the call arguments (line 435)
        # Getting the type of 'new' (line 435)
        new_376236 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 435, 20), 'new', False)
        # Calculating generator expression
        module_type_store = module_type_store.open_function_context('list comprehension expression', 435, 26, True)
        # Calculating comprehension expression
        
        # Call to iteritems(...): (line 436)
        # Processing the call arguments (line 436)
        # Getting the type of 'self' (line 436)
        self_376247 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 436, 62), 'self', False)
        # Processing the call keyword arguments (line 436)
        kwargs_376248 = {}
        # Getting the type of 'iteritems' (line 436)
        iteritems_376246 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 436, 52), 'iteritems', False)
        # Calling iteritems(args, kwargs) (line 436)
        iteritems_call_result_376249 = invoke(stypy.reporting.localization.Localization(__file__, 436, 52), iteritems_376246, *[self_376247], **kwargs_376248)
        
        comprehension_376250 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 435, 26), iteritems_call_result_376249)
        # Assigning a type to the variable 'left' (line 435)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 435, 26), 'left', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 435, 26), comprehension_376250))
        # Assigning a type to the variable 'right' (line 435)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 435, 26), 'right', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 435, 26), comprehension_376250))
        # Assigning a type to the variable 'val' (line 435)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 435, 26), 'val', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 435, 26), comprehension_376250))
        
        # Obtaining an instance of the builtin type 'tuple' (line 435)
        tuple_376237 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 435, 27), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 435)
        # Adding element type (line 435)
        
        # Obtaining an instance of the builtin type 'tuple' (line 435)
        tuple_376238 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 435, 28), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 435)
        # Adding element type (line 435)
        # Getting the type of 'right' (line 435)
        right_376239 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 435, 28), 'right', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 435, 28), tuple_376238, right_376239)
        # Adding element type (line 435)
        # Getting the type of 'left' (line 435)
        left_376240 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 435, 35), 'left', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 435, 28), tuple_376238, left_376240)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 435, 27), tuple_376237, tuple_376238)
        # Adding element type (line 435)
        
        # Call to conj(...): (line 435)
        # Processing the call arguments (line 435)
        # Getting the type of 'val' (line 435)
        val_376243 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 435, 50), 'val', False)
        # Processing the call keyword arguments (line 435)
        kwargs_376244 = {}
        # Getting the type of 'np' (line 435)
        np_376241 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 435, 42), 'np', False)
        # Obtaining the member 'conj' of a type (line 435)
        conj_376242 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 435, 42), np_376241, 'conj')
        # Calling conj(args, kwargs) (line 435)
        conj_call_result_376245 = invoke(stypy.reporting.localization.Localization(__file__, 435, 42), conj_376242, *[val_376243], **kwargs_376244)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 435, 27), tuple_376237, conj_call_result_376245)
        
        list_376252 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 435, 26), 'list')
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 435, 26), list_376252, tuple_376237)
        # Processing the call keyword arguments (line 435)
        kwargs_376253 = {}
        # Getting the type of 'dict' (line 435)
        dict_376234 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 435, 8), 'dict', False)
        # Obtaining the member 'update' of a type (line 435)
        update_376235 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 435, 8), dict_376234, 'update')
        # Calling update(args, kwargs) (line 435)
        update_call_result_376254 = invoke(stypy.reporting.localization.Localization(__file__, 435, 8), update_376235, *[new_376236, list_376252], **kwargs_376253)
        
        # Getting the type of 'new' (line 437)
        new_376255 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 437, 15), 'new')
        # Assigning a type to the variable 'stypy_return_type' (line 437)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 437, 8), 'stypy_return_type', new_376255)
        
        # ################# End of 'conjtransp(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'conjtransp' in the type store
        # Getting the type of 'stypy_return_type' (line 431)
        stypy_return_type_376256 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 431, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_376256)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'conjtransp'
        return stypy_return_type_376256


    @norecursion
    def copy(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'copy'
        module_type_store = module_type_store.open_function_context('copy', 439, 4, False)
        # Assigning a type to the variable 'self' (line 440)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 440, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        dok_matrix.copy.__dict__.__setitem__('stypy_localization', localization)
        dok_matrix.copy.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        dok_matrix.copy.__dict__.__setitem__('stypy_type_store', module_type_store)
        dok_matrix.copy.__dict__.__setitem__('stypy_function_name', 'dok_matrix.copy')
        dok_matrix.copy.__dict__.__setitem__('stypy_param_names_list', [])
        dok_matrix.copy.__dict__.__setitem__('stypy_varargs_param_name', None)
        dok_matrix.copy.__dict__.__setitem__('stypy_kwargs_param_name', None)
        dok_matrix.copy.__dict__.__setitem__('stypy_call_defaults', defaults)
        dok_matrix.copy.__dict__.__setitem__('stypy_call_varargs', varargs)
        dok_matrix.copy.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        dok_matrix.copy.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'dok_matrix.copy', [], None, None, defaults, varargs, kwargs)

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

        
        # Assigning a Call to a Name (line 440):
        
        # Assigning a Call to a Name (line 440):
        
        # Call to dok_matrix(...): (line 440)
        # Processing the call arguments (line 440)
        # Getting the type of 'self' (line 440)
        self_376258 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 440, 25), 'self', False)
        # Obtaining the member 'shape' of a type (line 440)
        shape_376259 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 440, 25), self_376258, 'shape')
        # Processing the call keyword arguments (line 440)
        # Getting the type of 'self' (line 440)
        self_376260 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 440, 43), 'self', False)
        # Obtaining the member 'dtype' of a type (line 440)
        dtype_376261 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 440, 43), self_376260, 'dtype')
        keyword_376262 = dtype_376261
        kwargs_376263 = {'dtype': keyword_376262}
        # Getting the type of 'dok_matrix' (line 440)
        dok_matrix_376257 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 440, 14), 'dok_matrix', False)
        # Calling dok_matrix(args, kwargs) (line 440)
        dok_matrix_call_result_376264 = invoke(stypy.reporting.localization.Localization(__file__, 440, 14), dok_matrix_376257, *[shape_376259], **kwargs_376263)
        
        # Assigning a type to the variable 'new' (line 440)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 440, 8), 'new', dok_matrix_call_result_376264)
        
        # Call to update(...): (line 441)
        # Processing the call arguments (line 441)
        # Getting the type of 'new' (line 441)
        new_376267 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 441, 20), 'new', False)
        # Getting the type of 'self' (line 441)
        self_376268 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 441, 25), 'self', False)
        # Processing the call keyword arguments (line 441)
        kwargs_376269 = {}
        # Getting the type of 'dict' (line 441)
        dict_376265 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 441, 8), 'dict', False)
        # Obtaining the member 'update' of a type (line 441)
        update_376266 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 441, 8), dict_376265, 'update')
        # Calling update(args, kwargs) (line 441)
        update_call_result_376270 = invoke(stypy.reporting.localization.Localization(__file__, 441, 8), update_376266, *[new_376267, self_376268], **kwargs_376269)
        
        # Getting the type of 'new' (line 442)
        new_376271 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 442, 15), 'new')
        # Assigning a type to the variable 'stypy_return_type' (line 442)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 442, 8), 'stypy_return_type', new_376271)
        
        # ################# End of 'copy(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'copy' in the type store
        # Getting the type of 'stypy_return_type' (line 439)
        stypy_return_type_376272 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 439, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_376272)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'copy'
        return stypy_return_type_376272

    
    # Assigning a Attribute to a Attribute (line 444):

    @norecursion
    def getrow(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'getrow'
        module_type_store = module_type_store.open_function_context('getrow', 446, 4, False)
        # Assigning a type to the variable 'self' (line 447)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 447, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        dok_matrix.getrow.__dict__.__setitem__('stypy_localization', localization)
        dok_matrix.getrow.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        dok_matrix.getrow.__dict__.__setitem__('stypy_type_store', module_type_store)
        dok_matrix.getrow.__dict__.__setitem__('stypy_function_name', 'dok_matrix.getrow')
        dok_matrix.getrow.__dict__.__setitem__('stypy_param_names_list', ['i'])
        dok_matrix.getrow.__dict__.__setitem__('stypy_varargs_param_name', None)
        dok_matrix.getrow.__dict__.__setitem__('stypy_kwargs_param_name', None)
        dok_matrix.getrow.__dict__.__setitem__('stypy_call_defaults', defaults)
        dok_matrix.getrow.__dict__.__setitem__('stypy_call_varargs', varargs)
        dok_matrix.getrow.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        dok_matrix.getrow.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'dok_matrix.getrow', ['i'], None, None, defaults, varargs, kwargs)

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

        str_376273 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 447, 8), 'str', 'Returns the i-th row as a (1 x n) DOK matrix.')
        
        # Assigning a Call to a Name (line 448):
        
        # Assigning a Call to a Name (line 448):
        
        # Call to dok_matrix(...): (line 448)
        # Processing the call arguments (line 448)
        
        # Obtaining an instance of the builtin type 'tuple' (line 448)
        tuple_376275 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 448, 26), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 448)
        # Adding element type (line 448)
        int_376276 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 448, 26), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 448, 26), tuple_376275, int_376276)
        # Adding element type (line 448)
        
        # Obtaining the type of the subscript
        int_376277 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 448, 40), 'int')
        # Getting the type of 'self' (line 448)
        self_376278 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 448, 29), 'self', False)
        # Obtaining the member 'shape' of a type (line 448)
        shape_376279 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 448, 29), self_376278, 'shape')
        # Obtaining the member '__getitem__' of a type (line 448)
        getitem___376280 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 448, 29), shape_376279, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 448)
        subscript_call_result_376281 = invoke(stypy.reporting.localization.Localization(__file__, 448, 29), getitem___376280, int_376277)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 448, 26), tuple_376275, subscript_call_result_376281)
        
        # Processing the call keyword arguments (line 448)
        # Getting the type of 'self' (line 448)
        self_376282 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 448, 51), 'self', False)
        # Obtaining the member 'dtype' of a type (line 448)
        dtype_376283 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 448, 51), self_376282, 'dtype')
        keyword_376284 = dtype_376283
        kwargs_376285 = {'dtype': keyword_376284}
        # Getting the type of 'dok_matrix' (line 448)
        dok_matrix_376274 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 448, 14), 'dok_matrix', False)
        # Calling dok_matrix(args, kwargs) (line 448)
        dok_matrix_call_result_376286 = invoke(stypy.reporting.localization.Localization(__file__, 448, 14), dok_matrix_376274, *[tuple_376275], **kwargs_376285)
        
        # Assigning a type to the variable 'new' (line 448)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 448, 8), 'new', dok_matrix_call_result_376286)
        
        # Call to update(...): (line 449)
        # Processing the call arguments (line 449)
        # Getting the type of 'new' (line 449)
        new_376289 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 449, 20), 'new', False)
        # Calculating generator expression
        module_type_store = module_type_store.open_function_context('list comprehension expression', 449, 26, True)
        # Calculating comprehension expression
        
        # Call to xrange(...): (line 449)
        # Processing the call arguments (line 449)
        
        # Obtaining the type of the subscript
        int_376301 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 449, 74), 'int')
        # Getting the type of 'self' (line 449)
        self_376302 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 449, 63), 'self', False)
        # Obtaining the member 'shape' of a type (line 449)
        shape_376303 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 449, 63), self_376302, 'shape')
        # Obtaining the member '__getitem__' of a type (line 449)
        getitem___376304 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 449, 63), shape_376303, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 449)
        subscript_call_result_376305 = invoke(stypy.reporting.localization.Localization(__file__, 449, 63), getitem___376304, int_376301)
        
        # Processing the call keyword arguments (line 449)
        kwargs_376306 = {}
        # Getting the type of 'xrange' (line 449)
        xrange_376300 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 449, 56), 'xrange', False)
        # Calling xrange(args, kwargs) (line 449)
        xrange_call_result_376307 = invoke(stypy.reporting.localization.Localization(__file__, 449, 56), xrange_376300, *[subscript_call_result_376305], **kwargs_376306)
        
        comprehension_376308 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 449, 26), xrange_call_result_376307)
        # Assigning a type to the variable 'j' (line 449)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 449, 26), 'j', comprehension_376308)
        
        # Obtaining an instance of the builtin type 'tuple' (line 449)
        tuple_376290 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 449, 27), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 449)
        # Adding element type (line 449)
        
        # Obtaining an instance of the builtin type 'tuple' (line 449)
        tuple_376291 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 449, 28), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 449)
        # Adding element type (line 449)
        int_376292 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 449, 28), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 449, 28), tuple_376291, int_376292)
        # Adding element type (line 449)
        # Getting the type of 'j' (line 449)
        j_376293 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 449, 31), 'j', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 449, 28), tuple_376291, j_376293)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 449, 27), tuple_376290, tuple_376291)
        # Adding element type (line 449)
        
        # Obtaining the type of the subscript
        
        # Obtaining an instance of the builtin type 'tuple' (line 449)
        tuple_376294 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 449, 40), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 449)
        # Adding element type (line 449)
        # Getting the type of 'i' (line 449)
        i_376295 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 449, 40), 'i', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 449, 40), tuple_376294, i_376295)
        # Adding element type (line 449)
        # Getting the type of 'j' (line 449)
        j_376296 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 449, 43), 'j', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 449, 40), tuple_376294, j_376296)
        
        # Getting the type of 'self' (line 449)
        self_376297 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 449, 35), 'self', False)
        # Obtaining the member '__getitem__' of a type (line 449)
        getitem___376298 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 449, 35), self_376297, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 449)
        subscript_call_result_376299 = invoke(stypy.reporting.localization.Localization(__file__, 449, 35), getitem___376298, tuple_376294)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 449, 27), tuple_376290, subscript_call_result_376299)
        
        list_376309 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 449, 26), 'list')
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 449, 26), list_376309, tuple_376290)
        # Processing the call keyword arguments (line 449)
        kwargs_376310 = {}
        # Getting the type of 'dict' (line 449)
        dict_376287 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 449, 8), 'dict', False)
        # Obtaining the member 'update' of a type (line 449)
        update_376288 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 449, 8), dict_376287, 'update')
        # Calling update(args, kwargs) (line 449)
        update_call_result_376311 = invoke(stypy.reporting.localization.Localization(__file__, 449, 8), update_376288, *[new_376289, list_376309], **kwargs_376310)
        
        # Getting the type of 'new' (line 450)
        new_376312 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 450, 15), 'new')
        # Assigning a type to the variable 'stypy_return_type' (line 450)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 450, 8), 'stypy_return_type', new_376312)
        
        # ################# End of 'getrow(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'getrow' in the type store
        # Getting the type of 'stypy_return_type' (line 446)
        stypy_return_type_376313 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 446, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_376313)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'getrow'
        return stypy_return_type_376313


    @norecursion
    def getcol(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'getcol'
        module_type_store = module_type_store.open_function_context('getcol', 452, 4, False)
        # Assigning a type to the variable 'self' (line 453)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 453, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        dok_matrix.getcol.__dict__.__setitem__('stypy_localization', localization)
        dok_matrix.getcol.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        dok_matrix.getcol.__dict__.__setitem__('stypy_type_store', module_type_store)
        dok_matrix.getcol.__dict__.__setitem__('stypy_function_name', 'dok_matrix.getcol')
        dok_matrix.getcol.__dict__.__setitem__('stypy_param_names_list', ['j'])
        dok_matrix.getcol.__dict__.__setitem__('stypy_varargs_param_name', None)
        dok_matrix.getcol.__dict__.__setitem__('stypy_kwargs_param_name', None)
        dok_matrix.getcol.__dict__.__setitem__('stypy_call_defaults', defaults)
        dok_matrix.getcol.__dict__.__setitem__('stypy_call_varargs', varargs)
        dok_matrix.getcol.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        dok_matrix.getcol.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'dok_matrix.getcol', ['j'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'getcol', localization, ['j'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'getcol(...)' code ##################

        str_376314 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 453, 8), 'str', 'Returns the j-th column as a (m x 1) DOK matrix.')
        
        # Assigning a Call to a Name (line 454):
        
        # Assigning a Call to a Name (line 454):
        
        # Call to dok_matrix(...): (line 454)
        # Processing the call arguments (line 454)
        
        # Obtaining an instance of the builtin type 'tuple' (line 454)
        tuple_376316 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 454, 26), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 454)
        # Adding element type (line 454)
        
        # Obtaining the type of the subscript
        int_376317 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 454, 37), 'int')
        # Getting the type of 'self' (line 454)
        self_376318 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 454, 26), 'self', False)
        # Obtaining the member 'shape' of a type (line 454)
        shape_376319 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 454, 26), self_376318, 'shape')
        # Obtaining the member '__getitem__' of a type (line 454)
        getitem___376320 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 454, 26), shape_376319, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 454)
        subscript_call_result_376321 = invoke(stypy.reporting.localization.Localization(__file__, 454, 26), getitem___376320, int_376317)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 454, 26), tuple_376316, subscript_call_result_376321)
        # Adding element type (line 454)
        int_376322 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 454, 41), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 454, 26), tuple_376316, int_376322)
        
        # Processing the call keyword arguments (line 454)
        # Getting the type of 'self' (line 454)
        self_376323 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 454, 51), 'self', False)
        # Obtaining the member 'dtype' of a type (line 454)
        dtype_376324 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 454, 51), self_376323, 'dtype')
        keyword_376325 = dtype_376324
        kwargs_376326 = {'dtype': keyword_376325}
        # Getting the type of 'dok_matrix' (line 454)
        dok_matrix_376315 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 454, 14), 'dok_matrix', False)
        # Calling dok_matrix(args, kwargs) (line 454)
        dok_matrix_call_result_376327 = invoke(stypy.reporting.localization.Localization(__file__, 454, 14), dok_matrix_376315, *[tuple_376316], **kwargs_376326)
        
        # Assigning a type to the variable 'new' (line 454)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 454, 8), 'new', dok_matrix_call_result_376327)
        
        # Call to update(...): (line 455)
        # Processing the call arguments (line 455)
        # Getting the type of 'new' (line 455)
        new_376330 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 455, 20), 'new', False)
        # Calculating generator expression
        module_type_store = module_type_store.open_function_context('list comprehension expression', 455, 26, True)
        # Calculating comprehension expression
        
        # Call to xrange(...): (line 455)
        # Processing the call arguments (line 455)
        
        # Obtaining the type of the subscript
        int_376342 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 455, 74), 'int')
        # Getting the type of 'self' (line 455)
        self_376343 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 455, 63), 'self', False)
        # Obtaining the member 'shape' of a type (line 455)
        shape_376344 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 455, 63), self_376343, 'shape')
        # Obtaining the member '__getitem__' of a type (line 455)
        getitem___376345 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 455, 63), shape_376344, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 455)
        subscript_call_result_376346 = invoke(stypy.reporting.localization.Localization(__file__, 455, 63), getitem___376345, int_376342)
        
        # Processing the call keyword arguments (line 455)
        kwargs_376347 = {}
        # Getting the type of 'xrange' (line 455)
        xrange_376341 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 455, 56), 'xrange', False)
        # Calling xrange(args, kwargs) (line 455)
        xrange_call_result_376348 = invoke(stypy.reporting.localization.Localization(__file__, 455, 56), xrange_376341, *[subscript_call_result_376346], **kwargs_376347)
        
        comprehension_376349 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 455, 26), xrange_call_result_376348)
        # Assigning a type to the variable 'i' (line 455)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 455, 26), 'i', comprehension_376349)
        
        # Obtaining an instance of the builtin type 'tuple' (line 455)
        tuple_376331 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 455, 27), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 455)
        # Adding element type (line 455)
        
        # Obtaining an instance of the builtin type 'tuple' (line 455)
        tuple_376332 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 455, 28), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 455)
        # Adding element type (line 455)
        # Getting the type of 'i' (line 455)
        i_376333 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 455, 28), 'i', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 455, 28), tuple_376332, i_376333)
        # Adding element type (line 455)
        int_376334 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 455, 31), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 455, 28), tuple_376332, int_376334)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 455, 27), tuple_376331, tuple_376332)
        # Adding element type (line 455)
        
        # Obtaining the type of the subscript
        
        # Obtaining an instance of the builtin type 'tuple' (line 455)
        tuple_376335 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 455, 40), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 455)
        # Adding element type (line 455)
        # Getting the type of 'i' (line 455)
        i_376336 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 455, 40), 'i', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 455, 40), tuple_376335, i_376336)
        # Adding element type (line 455)
        # Getting the type of 'j' (line 455)
        j_376337 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 455, 43), 'j', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 455, 40), tuple_376335, j_376337)
        
        # Getting the type of 'self' (line 455)
        self_376338 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 455, 35), 'self', False)
        # Obtaining the member '__getitem__' of a type (line 455)
        getitem___376339 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 455, 35), self_376338, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 455)
        subscript_call_result_376340 = invoke(stypy.reporting.localization.Localization(__file__, 455, 35), getitem___376339, tuple_376335)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 455, 27), tuple_376331, subscript_call_result_376340)
        
        list_376350 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 455, 26), 'list')
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 455, 26), list_376350, tuple_376331)
        # Processing the call keyword arguments (line 455)
        kwargs_376351 = {}
        # Getting the type of 'dict' (line 455)
        dict_376328 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 455, 8), 'dict', False)
        # Obtaining the member 'update' of a type (line 455)
        update_376329 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 455, 8), dict_376328, 'update')
        # Calling update(args, kwargs) (line 455)
        update_call_result_376352 = invoke(stypy.reporting.localization.Localization(__file__, 455, 8), update_376329, *[new_376330, list_376350], **kwargs_376351)
        
        # Getting the type of 'new' (line 456)
        new_376353 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 456, 15), 'new')
        # Assigning a type to the variable 'stypy_return_type' (line 456)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 456, 8), 'stypy_return_type', new_376353)
        
        # ################# End of 'getcol(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'getcol' in the type store
        # Getting the type of 'stypy_return_type' (line 452)
        stypy_return_type_376354 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 452, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_376354)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'getcol'
        return stypy_return_type_376354


    @norecursion
    def tocoo(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'False' (line 458)
        False_376355 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 458, 25), 'False')
        defaults = [False_376355]
        # Create a new context for function 'tocoo'
        module_type_store = module_type_store.open_function_context('tocoo', 458, 4, False)
        # Assigning a type to the variable 'self' (line 459)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 459, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        dok_matrix.tocoo.__dict__.__setitem__('stypy_localization', localization)
        dok_matrix.tocoo.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        dok_matrix.tocoo.__dict__.__setitem__('stypy_type_store', module_type_store)
        dok_matrix.tocoo.__dict__.__setitem__('stypy_function_name', 'dok_matrix.tocoo')
        dok_matrix.tocoo.__dict__.__setitem__('stypy_param_names_list', ['copy'])
        dok_matrix.tocoo.__dict__.__setitem__('stypy_varargs_param_name', None)
        dok_matrix.tocoo.__dict__.__setitem__('stypy_kwargs_param_name', None)
        dok_matrix.tocoo.__dict__.__setitem__('stypy_call_defaults', defaults)
        dok_matrix.tocoo.__dict__.__setitem__('stypy_call_varargs', varargs)
        dok_matrix.tocoo.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        dok_matrix.tocoo.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'dok_matrix.tocoo', ['copy'], None, None, defaults, varargs, kwargs)

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

        stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 459, 8))
        
        # 'from scipy.sparse.coo import coo_matrix' statement (line 459)
        update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/sparse/')
        import_376356 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 459, 8), 'scipy.sparse.coo')

        if (type(import_376356) is not StypyTypeError):

            if (import_376356 != 'pyd_module'):
                __import__(import_376356)
                sys_modules_376357 = sys.modules[import_376356]
                import_from_module(stypy.reporting.localization.Localization(__file__, 459, 8), 'scipy.sparse.coo', sys_modules_376357.module_type_store, module_type_store, ['coo_matrix'])
                nest_module(stypy.reporting.localization.Localization(__file__, 459, 8), __file__, sys_modules_376357, sys_modules_376357.module_type_store, module_type_store)
            else:
                from scipy.sparse.coo import coo_matrix

                import_from_module(stypy.reporting.localization.Localization(__file__, 459, 8), 'scipy.sparse.coo', None, module_type_store, ['coo_matrix'], [coo_matrix])

        else:
            # Assigning a type to the variable 'scipy.sparse.coo' (line 459)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 459, 8), 'scipy.sparse.coo', import_376356)

        remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/sparse/')
        
        
        
        # Getting the type of 'self' (line 460)
        self_376358 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 460, 11), 'self')
        # Obtaining the member 'nnz' of a type (line 460)
        nnz_376359 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 460, 11), self_376358, 'nnz')
        int_376360 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 460, 23), 'int')
        # Applying the binary operator '==' (line 460)
        result_eq_376361 = python_operator(stypy.reporting.localization.Localization(__file__, 460, 11), '==', nnz_376359, int_376360)
        
        # Testing the type of an if condition (line 460)
        if_condition_376362 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 460, 8), result_eq_376361)
        # Assigning a type to the variable 'if_condition_376362' (line 460)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 460, 8), 'if_condition_376362', if_condition_376362)
        # SSA begins for if statement (line 460)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to coo_matrix(...): (line 461)
        # Processing the call arguments (line 461)
        # Getting the type of 'self' (line 461)
        self_376364 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 461, 30), 'self', False)
        # Obtaining the member 'shape' of a type (line 461)
        shape_376365 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 461, 30), self_376364, 'shape')
        # Processing the call keyword arguments (line 461)
        # Getting the type of 'self' (line 461)
        self_376366 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 461, 48), 'self', False)
        # Obtaining the member 'dtype' of a type (line 461)
        dtype_376367 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 461, 48), self_376366, 'dtype')
        keyword_376368 = dtype_376367
        kwargs_376369 = {'dtype': keyword_376368}
        # Getting the type of 'coo_matrix' (line 461)
        coo_matrix_376363 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 461, 19), 'coo_matrix', False)
        # Calling coo_matrix(args, kwargs) (line 461)
        coo_matrix_call_result_376370 = invoke(stypy.reporting.localization.Localization(__file__, 461, 19), coo_matrix_376363, *[shape_376365], **kwargs_376369)
        
        # Assigning a type to the variable 'stypy_return_type' (line 461)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 461, 12), 'stypy_return_type', coo_matrix_call_result_376370)
        # SSA join for if statement (line 460)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Name (line 463):
        
        # Assigning a Call to a Name (line 463):
        
        # Call to get_index_dtype(...): (line 463)
        # Processing the call keyword arguments (line 463)
        
        # Call to max(...): (line 463)
        # Processing the call arguments (line 463)
        # Getting the type of 'self' (line 463)
        self_376373 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 463, 47), 'self', False)
        # Obtaining the member 'shape' of a type (line 463)
        shape_376374 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 463, 47), self_376373, 'shape')
        # Processing the call keyword arguments (line 463)
        kwargs_376375 = {}
        # Getting the type of 'max' (line 463)
        max_376372 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 463, 43), 'max', False)
        # Calling max(args, kwargs) (line 463)
        max_call_result_376376 = invoke(stypy.reporting.localization.Localization(__file__, 463, 43), max_376372, *[shape_376374], **kwargs_376375)
        
        keyword_376377 = max_call_result_376376
        kwargs_376378 = {'maxval': keyword_376377}
        # Getting the type of 'get_index_dtype' (line 463)
        get_index_dtype_376371 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 463, 20), 'get_index_dtype', False)
        # Calling get_index_dtype(args, kwargs) (line 463)
        get_index_dtype_call_result_376379 = invoke(stypy.reporting.localization.Localization(__file__, 463, 20), get_index_dtype_376371, *[], **kwargs_376378)
        
        # Assigning a type to the variable 'idx_dtype' (line 463)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 463, 8), 'idx_dtype', get_index_dtype_call_result_376379)
        
        # Assigning a Call to a Name (line 464):
        
        # Assigning a Call to a Name (line 464):
        
        # Call to fromiter(...): (line 464)
        # Processing the call arguments (line 464)
        
        # Call to itervalues(...): (line 464)
        # Processing the call arguments (line 464)
        # Getting the type of 'self' (line 464)
        self_376383 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 464, 38), 'self', False)
        # Processing the call keyword arguments (line 464)
        kwargs_376384 = {}
        # Getting the type of 'itervalues' (line 464)
        itervalues_376382 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 464, 27), 'itervalues', False)
        # Calling itervalues(args, kwargs) (line 464)
        itervalues_call_result_376385 = invoke(stypy.reporting.localization.Localization(__file__, 464, 27), itervalues_376382, *[self_376383], **kwargs_376384)
        
        # Processing the call keyword arguments (line 464)
        # Getting the type of 'self' (line 464)
        self_376386 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 464, 51), 'self', False)
        # Obtaining the member 'dtype' of a type (line 464)
        dtype_376387 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 464, 51), self_376386, 'dtype')
        keyword_376388 = dtype_376387
        # Getting the type of 'self' (line 464)
        self_376389 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 464, 69), 'self', False)
        # Obtaining the member 'nnz' of a type (line 464)
        nnz_376390 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 464, 69), self_376389, 'nnz')
        keyword_376391 = nnz_376390
        kwargs_376392 = {'count': keyword_376391, 'dtype': keyword_376388}
        # Getting the type of 'np' (line 464)
        np_376380 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 464, 15), 'np', False)
        # Obtaining the member 'fromiter' of a type (line 464)
        fromiter_376381 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 464, 15), np_376380, 'fromiter')
        # Calling fromiter(args, kwargs) (line 464)
        fromiter_call_result_376393 = invoke(stypy.reporting.localization.Localization(__file__, 464, 15), fromiter_376381, *[itervalues_call_result_376385], **kwargs_376392)
        
        # Assigning a type to the variable 'data' (line 464)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 464, 8), 'data', fromiter_call_result_376393)
        
        # Assigning a Call to a Name (line 465):
        
        # Assigning a Call to a Name (line 465):
        
        # Call to fromiter(...): (line 465)
        # Processing the call arguments (line 465)
        # Calculating generator expression
        module_type_store = module_type_store.open_function_context('list comprehension expression', 465, 25, True)
        # Calculating comprehension expression
        
        # Call to iterkeys(...): (line 465)
        # Processing the call arguments (line 465)
        # Getting the type of 'self' (line 465)
        self_376398 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 465, 48), 'self', False)
        # Processing the call keyword arguments (line 465)
        kwargs_376399 = {}
        # Getting the type of 'iterkeys' (line 465)
        iterkeys_376397 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 465, 39), 'iterkeys', False)
        # Calling iterkeys(args, kwargs) (line 465)
        iterkeys_call_result_376400 = invoke(stypy.reporting.localization.Localization(__file__, 465, 39), iterkeys_376397, *[self_376398], **kwargs_376399)
        
        comprehension_376401 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 465, 25), iterkeys_call_result_376400)
        # Assigning a type to the variable 'i' (line 465)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 465, 25), 'i', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 465, 25), comprehension_376401))
        # Assigning a type to the variable '_' (line 465)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 465, 25), '_', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 465, 25), comprehension_376401))
        # Getting the type of 'i' (line 465)
        i_376396 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 465, 25), 'i', False)
        list_376402 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 465, 25), 'list')
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 465, 25), list_376402, i_376396)
        # Processing the call keyword arguments (line 465)
        # Getting the type of 'idx_dtype' (line 465)
        idx_dtype_376403 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 465, 62), 'idx_dtype', False)
        keyword_376404 = idx_dtype_376403
        # Getting the type of 'self' (line 465)
        self_376405 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 465, 79), 'self', False)
        # Obtaining the member 'nnz' of a type (line 465)
        nnz_376406 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 465, 79), self_376405, 'nnz')
        keyword_376407 = nnz_376406
        kwargs_376408 = {'count': keyword_376407, 'dtype': keyword_376404}
        # Getting the type of 'np' (line 465)
        np_376394 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 465, 12), 'np', False)
        # Obtaining the member 'fromiter' of a type (line 465)
        fromiter_376395 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 465, 12), np_376394, 'fromiter')
        # Calling fromiter(args, kwargs) (line 465)
        fromiter_call_result_376409 = invoke(stypy.reporting.localization.Localization(__file__, 465, 12), fromiter_376395, *[list_376402], **kwargs_376408)
        
        # Assigning a type to the variable 'I' (line 465)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 465, 8), 'I', fromiter_call_result_376409)
        
        # Assigning a Call to a Name (line 466):
        
        # Assigning a Call to a Name (line 466):
        
        # Call to fromiter(...): (line 466)
        # Processing the call arguments (line 466)
        # Calculating generator expression
        module_type_store = module_type_store.open_function_context('list comprehension expression', 466, 25, True)
        # Calculating comprehension expression
        
        # Call to iterkeys(...): (line 466)
        # Processing the call arguments (line 466)
        # Getting the type of 'self' (line 466)
        self_376414 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 466, 48), 'self', False)
        # Processing the call keyword arguments (line 466)
        kwargs_376415 = {}
        # Getting the type of 'iterkeys' (line 466)
        iterkeys_376413 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 466, 39), 'iterkeys', False)
        # Calling iterkeys(args, kwargs) (line 466)
        iterkeys_call_result_376416 = invoke(stypy.reporting.localization.Localization(__file__, 466, 39), iterkeys_376413, *[self_376414], **kwargs_376415)
        
        comprehension_376417 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 466, 25), iterkeys_call_result_376416)
        # Assigning a type to the variable '_' (line 466)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 466, 25), '_', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 466, 25), comprehension_376417))
        # Assigning a type to the variable 'j' (line 466)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 466, 25), 'j', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 466, 25), comprehension_376417))
        # Getting the type of 'j' (line 466)
        j_376412 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 466, 25), 'j', False)
        list_376418 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 466, 25), 'list')
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 466, 25), list_376418, j_376412)
        # Processing the call keyword arguments (line 466)
        # Getting the type of 'idx_dtype' (line 466)
        idx_dtype_376419 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 466, 62), 'idx_dtype', False)
        keyword_376420 = idx_dtype_376419
        # Getting the type of 'self' (line 466)
        self_376421 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 466, 79), 'self', False)
        # Obtaining the member 'nnz' of a type (line 466)
        nnz_376422 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 466, 79), self_376421, 'nnz')
        keyword_376423 = nnz_376422
        kwargs_376424 = {'count': keyword_376423, 'dtype': keyword_376420}
        # Getting the type of 'np' (line 466)
        np_376410 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 466, 12), 'np', False)
        # Obtaining the member 'fromiter' of a type (line 466)
        fromiter_376411 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 466, 12), np_376410, 'fromiter')
        # Calling fromiter(args, kwargs) (line 466)
        fromiter_call_result_376425 = invoke(stypy.reporting.localization.Localization(__file__, 466, 12), fromiter_376411, *[list_376418], **kwargs_376424)
        
        # Assigning a type to the variable 'J' (line 466)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 466, 8), 'J', fromiter_call_result_376425)
        
        # Assigning a Call to a Name (line 467):
        
        # Assigning a Call to a Name (line 467):
        
        # Call to coo_matrix(...): (line 467)
        # Processing the call arguments (line 467)
        
        # Obtaining an instance of the builtin type 'tuple' (line 467)
        tuple_376427 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 467, 24), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 467)
        # Adding element type (line 467)
        # Getting the type of 'data' (line 467)
        data_376428 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 467, 24), 'data', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 467, 24), tuple_376427, data_376428)
        # Adding element type (line 467)
        
        # Obtaining an instance of the builtin type 'tuple' (line 467)
        tuple_376429 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 467, 31), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 467)
        # Adding element type (line 467)
        # Getting the type of 'I' (line 467)
        I_376430 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 467, 31), 'I', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 467, 31), tuple_376429, I_376430)
        # Adding element type (line 467)
        # Getting the type of 'J' (line 467)
        J_376431 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 467, 34), 'J', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 467, 31), tuple_376429, J_376431)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 467, 24), tuple_376427, tuple_376429)
        
        # Processing the call keyword arguments (line 467)
        # Getting the type of 'self' (line 467)
        self_376432 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 467, 45), 'self', False)
        # Obtaining the member 'shape' of a type (line 467)
        shape_376433 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 467, 45), self_376432, 'shape')
        keyword_376434 = shape_376433
        # Getting the type of 'self' (line 467)
        self_376435 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 467, 63), 'self', False)
        # Obtaining the member 'dtype' of a type (line 467)
        dtype_376436 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 467, 63), self_376435, 'dtype')
        keyword_376437 = dtype_376436
        kwargs_376438 = {'dtype': keyword_376437, 'shape': keyword_376434}
        # Getting the type of 'coo_matrix' (line 467)
        coo_matrix_376426 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 467, 12), 'coo_matrix', False)
        # Calling coo_matrix(args, kwargs) (line 467)
        coo_matrix_call_result_376439 = invoke(stypy.reporting.localization.Localization(__file__, 467, 12), coo_matrix_376426, *[tuple_376427], **kwargs_376438)
        
        # Assigning a type to the variable 'A' (line 467)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 467, 8), 'A', coo_matrix_call_result_376439)
        
        # Assigning a Name to a Attribute (line 468):
        
        # Assigning a Name to a Attribute (line 468):
        # Getting the type of 'True' (line 468)
        True_376440 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 468, 33), 'True')
        # Getting the type of 'A' (line 468)
        A_376441 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 468, 8), 'A')
        # Setting the type of the member 'has_canonical_format' of a type (line 468)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 468, 8), A_376441, 'has_canonical_format', True_376440)
        # Getting the type of 'A' (line 469)
        A_376442 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 469, 15), 'A')
        # Assigning a type to the variable 'stypy_return_type' (line 469)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 469, 8), 'stypy_return_type', A_376442)
        
        # ################# End of 'tocoo(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'tocoo' in the type store
        # Getting the type of 'stypy_return_type' (line 458)
        stypy_return_type_376443 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 458, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_376443)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'tocoo'
        return stypy_return_type_376443

    
    # Assigning a Attribute to a Attribute (line 471):

    @norecursion
    def todok(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'False' (line 473)
        False_376444 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 473, 25), 'False')
        defaults = [False_376444]
        # Create a new context for function 'todok'
        module_type_store = module_type_store.open_function_context('todok', 473, 4, False)
        # Assigning a type to the variable 'self' (line 474)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 474, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        dok_matrix.todok.__dict__.__setitem__('stypy_localization', localization)
        dok_matrix.todok.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        dok_matrix.todok.__dict__.__setitem__('stypy_type_store', module_type_store)
        dok_matrix.todok.__dict__.__setitem__('stypy_function_name', 'dok_matrix.todok')
        dok_matrix.todok.__dict__.__setitem__('stypy_param_names_list', ['copy'])
        dok_matrix.todok.__dict__.__setitem__('stypy_varargs_param_name', None)
        dok_matrix.todok.__dict__.__setitem__('stypy_kwargs_param_name', None)
        dok_matrix.todok.__dict__.__setitem__('stypy_call_defaults', defaults)
        dok_matrix.todok.__dict__.__setitem__('stypy_call_varargs', varargs)
        dok_matrix.todok.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        dok_matrix.todok.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'dok_matrix.todok', ['copy'], None, None, defaults, varargs, kwargs)

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

        
        # Getting the type of 'copy' (line 474)
        copy_376445 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 474, 11), 'copy')
        # Testing the type of an if condition (line 474)
        if_condition_376446 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 474, 8), copy_376445)
        # Assigning a type to the variable 'if_condition_376446' (line 474)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 474, 8), 'if_condition_376446', if_condition_376446)
        # SSA begins for if statement (line 474)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to copy(...): (line 475)
        # Processing the call keyword arguments (line 475)
        kwargs_376449 = {}
        # Getting the type of 'self' (line 475)
        self_376447 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 475, 19), 'self', False)
        # Obtaining the member 'copy' of a type (line 475)
        copy_376448 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 475, 19), self_376447, 'copy')
        # Calling copy(args, kwargs) (line 475)
        copy_call_result_376450 = invoke(stypy.reporting.localization.Localization(__file__, 475, 19), copy_376448, *[], **kwargs_376449)
        
        # Assigning a type to the variable 'stypy_return_type' (line 475)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 475, 12), 'stypy_return_type', copy_call_result_376450)
        # SSA join for if statement (line 474)
        module_type_store = module_type_store.join_ssa_context()
        
        # Getting the type of 'self' (line 476)
        self_376451 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 476, 15), 'self')
        # Assigning a type to the variable 'stypy_return_type' (line 476)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 476, 8), 'stypy_return_type', self_376451)
        
        # ################# End of 'todok(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'todok' in the type store
        # Getting the type of 'stypy_return_type' (line 473)
        stypy_return_type_376452 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 473, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_376452)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'todok'
        return stypy_return_type_376452

    
    # Assigning a Attribute to a Attribute (line 478):

    @norecursion
    def tocsc(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'False' (line 480)
        False_376453 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 480, 25), 'False')
        defaults = [False_376453]
        # Create a new context for function 'tocsc'
        module_type_store = module_type_store.open_function_context('tocsc', 480, 4, False)
        # Assigning a type to the variable 'self' (line 481)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 481, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        dok_matrix.tocsc.__dict__.__setitem__('stypy_localization', localization)
        dok_matrix.tocsc.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        dok_matrix.tocsc.__dict__.__setitem__('stypy_type_store', module_type_store)
        dok_matrix.tocsc.__dict__.__setitem__('stypy_function_name', 'dok_matrix.tocsc')
        dok_matrix.tocsc.__dict__.__setitem__('stypy_param_names_list', ['copy'])
        dok_matrix.tocsc.__dict__.__setitem__('stypy_varargs_param_name', None)
        dok_matrix.tocsc.__dict__.__setitem__('stypy_kwargs_param_name', None)
        dok_matrix.tocsc.__dict__.__setitem__('stypy_call_defaults', defaults)
        dok_matrix.tocsc.__dict__.__setitem__('stypy_call_varargs', varargs)
        dok_matrix.tocsc.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        dok_matrix.tocsc.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'dok_matrix.tocsc', ['copy'], None, None, defaults, varargs, kwargs)

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

        
        # Call to tocsc(...): (line 481)
        # Processing the call keyword arguments (line 481)
        # Getting the type of 'copy' (line 481)
        copy_376461 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 481, 49), 'copy', False)
        keyword_376462 = copy_376461
        kwargs_376463 = {'copy': keyword_376462}
        
        # Call to tocoo(...): (line 481)
        # Processing the call keyword arguments (line 481)
        # Getting the type of 'False' (line 481)
        False_376456 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 481, 31), 'False', False)
        keyword_376457 = False_376456
        kwargs_376458 = {'copy': keyword_376457}
        # Getting the type of 'self' (line 481)
        self_376454 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 481, 15), 'self', False)
        # Obtaining the member 'tocoo' of a type (line 481)
        tocoo_376455 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 481, 15), self_376454, 'tocoo')
        # Calling tocoo(args, kwargs) (line 481)
        tocoo_call_result_376459 = invoke(stypy.reporting.localization.Localization(__file__, 481, 15), tocoo_376455, *[], **kwargs_376458)
        
        # Obtaining the member 'tocsc' of a type (line 481)
        tocsc_376460 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 481, 15), tocoo_call_result_376459, 'tocsc')
        # Calling tocsc(args, kwargs) (line 481)
        tocsc_call_result_376464 = invoke(stypy.reporting.localization.Localization(__file__, 481, 15), tocsc_376460, *[], **kwargs_376463)
        
        # Assigning a type to the variable 'stypy_return_type' (line 481)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 481, 8), 'stypy_return_type', tocsc_call_result_376464)
        
        # ################# End of 'tocsc(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'tocsc' in the type store
        # Getting the type of 'stypy_return_type' (line 480)
        stypy_return_type_376465 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 480, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_376465)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'tocsc'
        return stypy_return_type_376465

    
    # Assigning a Attribute to a Attribute (line 483):

    @norecursion
    def resize(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'resize'
        module_type_store = module_type_store.open_function_context('resize', 485, 4, False)
        # Assigning a type to the variable 'self' (line 486)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 486, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        dok_matrix.resize.__dict__.__setitem__('stypy_localization', localization)
        dok_matrix.resize.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        dok_matrix.resize.__dict__.__setitem__('stypy_type_store', module_type_store)
        dok_matrix.resize.__dict__.__setitem__('stypy_function_name', 'dok_matrix.resize')
        dok_matrix.resize.__dict__.__setitem__('stypy_param_names_list', ['shape'])
        dok_matrix.resize.__dict__.__setitem__('stypy_varargs_param_name', None)
        dok_matrix.resize.__dict__.__setitem__('stypy_kwargs_param_name', None)
        dok_matrix.resize.__dict__.__setitem__('stypy_call_defaults', defaults)
        dok_matrix.resize.__dict__.__setitem__('stypy_call_varargs', varargs)
        dok_matrix.resize.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        dok_matrix.resize.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'dok_matrix.resize', ['shape'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'resize', localization, ['shape'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'resize(...)' code ##################

        str_376466 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 489, (-1)), 'str', 'Resize the matrix in-place to dimensions given by `shape`.\n\n        Any non-zero elements that lie outside the new shape are removed.\n        ')
        
        
        
        # Call to isshape(...): (line 490)
        # Processing the call arguments (line 490)
        # Getting the type of 'shape' (line 490)
        shape_376468 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 490, 23), 'shape', False)
        # Processing the call keyword arguments (line 490)
        kwargs_376469 = {}
        # Getting the type of 'isshape' (line 490)
        isshape_376467 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 490, 15), 'isshape', False)
        # Calling isshape(args, kwargs) (line 490)
        isshape_call_result_376470 = invoke(stypy.reporting.localization.Localization(__file__, 490, 15), isshape_376467, *[shape_376468], **kwargs_376469)
        
        # Applying the 'not' unary operator (line 490)
        result_not__376471 = python_operator(stypy.reporting.localization.Localization(__file__, 490, 11), 'not', isshape_call_result_376470)
        
        # Testing the type of an if condition (line 490)
        if_condition_376472 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 490, 8), result_not__376471)
        # Assigning a type to the variable 'if_condition_376472' (line 490)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 490, 8), 'if_condition_376472', if_condition_376472)
        # SSA begins for if statement (line 490)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to TypeError(...): (line 491)
        # Processing the call arguments (line 491)
        str_376474 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 491, 28), 'str', 'Dimensions must be a 2-tuple of positive integers')
        # Processing the call keyword arguments (line 491)
        kwargs_376475 = {}
        # Getting the type of 'TypeError' (line 491)
        TypeError_376473 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 491, 18), 'TypeError', False)
        # Calling TypeError(args, kwargs) (line 491)
        TypeError_call_result_376476 = invoke(stypy.reporting.localization.Localization(__file__, 491, 18), TypeError_376473, *[str_376474], **kwargs_376475)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 491, 12), TypeError_call_result_376476, 'raise parameter', BaseException)
        # SSA join for if statement (line 490)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Name to a Tuple (line 492):
        
        # Assigning a Subscript to a Name (line 492):
        
        # Obtaining the type of the subscript
        int_376477 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 492, 8), 'int')
        # Getting the type of 'shape' (line 492)
        shape_376478 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 492, 21), 'shape')
        # Obtaining the member '__getitem__' of a type (line 492)
        getitem___376479 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 492, 8), shape_376478, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 492)
        subscript_call_result_376480 = invoke(stypy.reporting.localization.Localization(__file__, 492, 8), getitem___376479, int_376477)
        
        # Assigning a type to the variable 'tuple_var_assignment_374374' (line 492)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 492, 8), 'tuple_var_assignment_374374', subscript_call_result_376480)
        
        # Assigning a Subscript to a Name (line 492):
        
        # Obtaining the type of the subscript
        int_376481 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 492, 8), 'int')
        # Getting the type of 'shape' (line 492)
        shape_376482 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 492, 21), 'shape')
        # Obtaining the member '__getitem__' of a type (line 492)
        getitem___376483 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 492, 8), shape_376482, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 492)
        subscript_call_result_376484 = invoke(stypy.reporting.localization.Localization(__file__, 492, 8), getitem___376483, int_376481)
        
        # Assigning a type to the variable 'tuple_var_assignment_374375' (line 492)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 492, 8), 'tuple_var_assignment_374375', subscript_call_result_376484)
        
        # Assigning a Name to a Name (line 492):
        # Getting the type of 'tuple_var_assignment_374374' (line 492)
        tuple_var_assignment_374374_376485 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 492, 8), 'tuple_var_assignment_374374')
        # Assigning a type to the variable 'newM' (line 492)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 492, 8), 'newM', tuple_var_assignment_374374_376485)
        
        # Assigning a Name to a Name (line 492):
        # Getting the type of 'tuple_var_assignment_374375' (line 492)
        tuple_var_assignment_374375_376486 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 492, 8), 'tuple_var_assignment_374375')
        # Assigning a type to the variable 'newN' (line 492)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 492, 14), 'newN', tuple_var_assignment_374375_376486)
        
        # Assigning a Attribute to a Tuple (line 493):
        
        # Assigning a Subscript to a Name (line 493):
        
        # Obtaining the type of the subscript
        int_376487 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 493, 8), 'int')
        # Getting the type of 'self' (line 493)
        self_376488 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 493, 15), 'self')
        # Obtaining the member 'shape' of a type (line 493)
        shape_376489 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 493, 15), self_376488, 'shape')
        # Obtaining the member '__getitem__' of a type (line 493)
        getitem___376490 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 493, 8), shape_376489, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 493)
        subscript_call_result_376491 = invoke(stypy.reporting.localization.Localization(__file__, 493, 8), getitem___376490, int_376487)
        
        # Assigning a type to the variable 'tuple_var_assignment_374376' (line 493)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 493, 8), 'tuple_var_assignment_374376', subscript_call_result_376491)
        
        # Assigning a Subscript to a Name (line 493):
        
        # Obtaining the type of the subscript
        int_376492 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 493, 8), 'int')
        # Getting the type of 'self' (line 493)
        self_376493 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 493, 15), 'self')
        # Obtaining the member 'shape' of a type (line 493)
        shape_376494 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 493, 15), self_376493, 'shape')
        # Obtaining the member '__getitem__' of a type (line 493)
        getitem___376495 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 493, 8), shape_376494, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 493)
        subscript_call_result_376496 = invoke(stypy.reporting.localization.Localization(__file__, 493, 8), getitem___376495, int_376492)
        
        # Assigning a type to the variable 'tuple_var_assignment_374377' (line 493)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 493, 8), 'tuple_var_assignment_374377', subscript_call_result_376496)
        
        # Assigning a Name to a Name (line 493):
        # Getting the type of 'tuple_var_assignment_374376' (line 493)
        tuple_var_assignment_374376_376497 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 493, 8), 'tuple_var_assignment_374376')
        # Assigning a type to the variable 'M' (line 493)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 493, 8), 'M', tuple_var_assignment_374376_376497)
        
        # Assigning a Name to a Name (line 493):
        # Getting the type of 'tuple_var_assignment_374377' (line 493)
        tuple_var_assignment_374377_376498 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 493, 8), 'tuple_var_assignment_374377')
        # Assigning a type to the variable 'N' (line 493)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 493, 11), 'N', tuple_var_assignment_374377_376498)
        
        
        # Evaluating a boolean operation
        
        # Getting the type of 'newM' (line 494)
        newM_376499 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 494, 11), 'newM')
        # Getting the type of 'M' (line 494)
        M_376500 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 494, 18), 'M')
        # Applying the binary operator '<' (line 494)
        result_lt_376501 = python_operator(stypy.reporting.localization.Localization(__file__, 494, 11), '<', newM_376499, M_376500)
        
        
        # Getting the type of 'newN' (line 494)
        newN_376502 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 494, 23), 'newN')
        # Getting the type of 'N' (line 494)
        N_376503 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 494, 30), 'N')
        # Applying the binary operator '<' (line 494)
        result_lt_376504 = python_operator(stypy.reporting.localization.Localization(__file__, 494, 23), '<', newN_376502, N_376503)
        
        # Applying the binary operator 'or' (line 494)
        result_or_keyword_376505 = python_operator(stypy.reporting.localization.Localization(__file__, 494, 11), 'or', result_lt_376501, result_lt_376504)
        
        # Testing the type of an if condition (line 494)
        if_condition_376506 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 494, 8), result_or_keyword_376505)
        # Assigning a type to the variable 'if_condition_376506' (line 494)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 494, 8), 'if_condition_376506', if_condition_376506)
        # SSA begins for if statement (line 494)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        
        # Call to list(...): (line 496)
        # Processing the call arguments (line 496)
        
        # Call to iterkeys(...): (line 496)
        # Processing the call arguments (line 496)
        # Getting the type of 'self' (line 496)
        self_376509 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 496, 40), 'self', False)
        # Processing the call keyword arguments (line 496)
        kwargs_376510 = {}
        # Getting the type of 'iterkeys' (line 496)
        iterkeys_376508 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 496, 31), 'iterkeys', False)
        # Calling iterkeys(args, kwargs) (line 496)
        iterkeys_call_result_376511 = invoke(stypy.reporting.localization.Localization(__file__, 496, 31), iterkeys_376508, *[self_376509], **kwargs_376510)
        
        # Processing the call keyword arguments (line 496)
        kwargs_376512 = {}
        # Getting the type of 'list' (line 496)
        list_376507 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 496, 26), 'list', False)
        # Calling list(args, kwargs) (line 496)
        list_call_result_376513 = invoke(stypy.reporting.localization.Localization(__file__, 496, 26), list_376507, *[iterkeys_call_result_376511], **kwargs_376512)
        
        # Testing the type of a for loop iterable (line 496)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 496, 12), list_call_result_376513)
        # Getting the type of the for loop variable (line 496)
        for_loop_var_376514 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 496, 12), list_call_result_376513)
        # Assigning a type to the variable 'i' (line 496)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 496, 12), 'i', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 496, 12), for_loop_var_376514))
        # Assigning a type to the variable 'j' (line 496)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 496, 12), 'j', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 496, 12), for_loop_var_376514))
        # SSA begins for a for statement (line 496)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        
        # Evaluating a boolean operation
        
        # Getting the type of 'i' (line 497)
        i_376515 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 497, 19), 'i')
        # Getting the type of 'newM' (line 497)
        newM_376516 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 497, 24), 'newM')
        # Applying the binary operator '>=' (line 497)
        result_ge_376517 = python_operator(stypy.reporting.localization.Localization(__file__, 497, 19), '>=', i_376515, newM_376516)
        
        
        # Getting the type of 'j' (line 497)
        j_376518 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 497, 32), 'j')
        # Getting the type of 'newN' (line 497)
        newN_376519 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 497, 37), 'newN')
        # Applying the binary operator '>=' (line 497)
        result_ge_376520 = python_operator(stypy.reporting.localization.Localization(__file__, 497, 32), '>=', j_376518, newN_376519)
        
        # Applying the binary operator 'or' (line 497)
        result_or_keyword_376521 = python_operator(stypy.reporting.localization.Localization(__file__, 497, 19), 'or', result_ge_376517, result_ge_376520)
        
        # Testing the type of an if condition (line 497)
        if_condition_376522 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 497, 16), result_or_keyword_376521)
        # Assigning a type to the variable 'if_condition_376522' (line 497)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 497, 16), 'if_condition_376522', if_condition_376522)
        # SSA begins for if statement (line 497)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # Deleting a member
        # Getting the type of 'self' (line 498)
        self_376523 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 498, 24), 'self')
        
        # Obtaining the type of the subscript
        
        # Obtaining an instance of the builtin type 'tuple' (line 498)
        tuple_376524 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 498, 29), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 498)
        # Adding element type (line 498)
        # Getting the type of 'i' (line 498)
        i_376525 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 498, 29), 'i')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 498, 29), tuple_376524, i_376525)
        # Adding element type (line 498)
        # Getting the type of 'j' (line 498)
        j_376526 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 498, 32), 'j')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 498, 29), tuple_376524, j_376526)
        
        # Getting the type of 'self' (line 498)
        self_376527 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 498, 24), 'self')
        # Obtaining the member '__getitem__' of a type (line 498)
        getitem___376528 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 498, 24), self_376527, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 498)
        subscript_call_result_376529 = invoke(stypy.reporting.localization.Localization(__file__, 498, 24), getitem___376528, tuple_376524)
        
        del_contained_elements_type(stypy.reporting.localization.Localization(__file__, 498, 20), self_376523, subscript_call_result_376529)
        # SSA join for if statement (line 497)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 494)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Name to a Attribute (line 499):
        
        # Assigning a Name to a Attribute (line 499):
        # Getting the type of 'shape' (line 499)
        shape_376530 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 499, 22), 'shape')
        # Getting the type of 'self' (line 499)
        self_376531 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 499, 8), 'self')
        # Setting the type of the member '_shape' of a type (line 499)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 499, 8), self_376531, '_shape', shape_376530)
        
        # ################# End of 'resize(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'resize' in the type store
        # Getting the type of 'stypy_return_type' (line 485)
        stypy_return_type_376532 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 485, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_376532)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'resize'
        return stypy_return_type_376532


# Assigning a type to the variable 'dok_matrix' (line 29)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 29, 0), 'dok_matrix', dok_matrix)

# Assigning a Str to a Name (line 78):
str_376533 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 78, 13), 'str', 'dok')
# Getting the type of 'dok_matrix'
dok_matrix_376534 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'dok_matrix')
# Setting the type of the member 'format' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), dok_matrix_376534, 'format', str_376533)

# Assigning a Attribute to a Attribute (line 135):
# Getting the type of 'spmatrix' (line 135)
spmatrix_376535 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 21), 'spmatrix')
# Obtaining the member 'getnnz' of a type (line 135)
getnnz_376536 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 135, 21), spmatrix_376535, 'getnnz')
# Obtaining the member '__doc__' of a type (line 135)
doc___376537 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 135, 21), getnnz_376536, '__doc__')
# Getting the type of 'dok_matrix'
dok_matrix_376538 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'dok_matrix')
# Obtaining the member 'getnnz' of a type
getnnz_376539 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), dok_matrix_376538, 'getnnz')
# Setting the type of the member '__doc__' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), getnnz_376539, '__doc__', doc___376537)

# Assigning a Attribute to a Attribute (line 136):
# Getting the type of 'spmatrix' (line 136)
spmatrix_376540 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 28), 'spmatrix')
# Obtaining the member 'count_nonzero' of a type (line 136)
count_nonzero_376541 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 136, 28), spmatrix_376540, 'count_nonzero')
# Obtaining the member '__doc__' of a type (line 136)
doc___376542 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 136, 28), count_nonzero_376541, '__doc__')
# Getting the type of 'dok_matrix'
dok_matrix_376543 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'dok_matrix')
# Obtaining the member 'count_nonzero' of a type
count_nonzero_376544 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), dok_matrix_376543, 'count_nonzero')
# Setting the type of the member '__doc__' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), count_nonzero_376544, '__doc__', doc___376542)

# Assigning a Attribute to a Attribute (line 429):
# Getting the type of 'spmatrix' (line 429)
spmatrix_376545 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 429, 24), 'spmatrix')
# Obtaining the member 'transpose' of a type (line 429)
transpose_376546 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 429, 24), spmatrix_376545, 'transpose')
# Obtaining the member '__doc__' of a type (line 429)
doc___376547 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 429, 24), transpose_376546, '__doc__')
# Getting the type of 'dok_matrix'
dok_matrix_376548 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'dok_matrix')
# Obtaining the member 'transpose' of a type
transpose_376549 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), dok_matrix_376548, 'transpose')
# Setting the type of the member '__doc__' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), transpose_376549, '__doc__', doc___376547)

# Assigning a Attribute to a Attribute (line 444):
# Getting the type of 'spmatrix' (line 444)
spmatrix_376550 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 444, 19), 'spmatrix')
# Obtaining the member 'copy' of a type (line 444)
copy_376551 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 444, 19), spmatrix_376550, 'copy')
# Obtaining the member '__doc__' of a type (line 444)
doc___376552 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 444, 19), copy_376551, '__doc__')
# Getting the type of 'dok_matrix'
dok_matrix_376553 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'dok_matrix')
# Obtaining the member 'copy' of a type
copy_376554 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), dok_matrix_376553, 'copy')
# Setting the type of the member '__doc__' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), copy_376554, '__doc__', doc___376552)

# Assigning a Attribute to a Attribute (line 471):
# Getting the type of 'spmatrix' (line 471)
spmatrix_376555 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 471, 20), 'spmatrix')
# Obtaining the member 'tocoo' of a type (line 471)
tocoo_376556 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 471, 20), spmatrix_376555, 'tocoo')
# Obtaining the member '__doc__' of a type (line 471)
doc___376557 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 471, 20), tocoo_376556, '__doc__')
# Getting the type of 'dok_matrix'
dok_matrix_376558 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'dok_matrix')
# Obtaining the member 'tocoo' of a type
tocoo_376559 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), dok_matrix_376558, 'tocoo')
# Setting the type of the member '__doc__' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), tocoo_376559, '__doc__', doc___376557)

# Assigning a Attribute to a Attribute (line 478):
# Getting the type of 'spmatrix' (line 478)
spmatrix_376560 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 478, 20), 'spmatrix')
# Obtaining the member 'todok' of a type (line 478)
todok_376561 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 478, 20), spmatrix_376560, 'todok')
# Obtaining the member '__doc__' of a type (line 478)
doc___376562 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 478, 20), todok_376561, '__doc__')
# Getting the type of 'dok_matrix'
dok_matrix_376563 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'dok_matrix')
# Obtaining the member 'todok' of a type
todok_376564 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), dok_matrix_376563, 'todok')
# Setting the type of the member '__doc__' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), todok_376564, '__doc__', doc___376562)

# Assigning a Attribute to a Attribute (line 483):
# Getting the type of 'spmatrix' (line 483)
spmatrix_376565 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 483, 20), 'spmatrix')
# Obtaining the member 'tocsc' of a type (line 483)
tocsc_376566 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 483, 20), spmatrix_376565, 'tocsc')
# Obtaining the member '__doc__' of a type (line 483)
doc___376567 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 483, 20), tocsc_376566, '__doc__')
# Getting the type of 'dok_matrix'
dok_matrix_376568 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'dok_matrix')
# Obtaining the member 'tocsc' of a type
tocsc_376569 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), dok_matrix_376568, 'tocsc')
# Setting the type of the member '__doc__' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), tocsc_376569, '__doc__', doc___376567)

@norecursion
def isspmatrix_dok(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'isspmatrix_dok'
    module_type_store = module_type_store.open_function_context('isspmatrix_dok', 502, 0, False)
    
    # Passed parameters checking function
    isspmatrix_dok.stypy_localization = localization
    isspmatrix_dok.stypy_type_of_self = None
    isspmatrix_dok.stypy_type_store = module_type_store
    isspmatrix_dok.stypy_function_name = 'isspmatrix_dok'
    isspmatrix_dok.stypy_param_names_list = ['x']
    isspmatrix_dok.stypy_varargs_param_name = None
    isspmatrix_dok.stypy_kwargs_param_name = None
    isspmatrix_dok.stypy_call_defaults = defaults
    isspmatrix_dok.stypy_call_varargs = varargs
    isspmatrix_dok.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'isspmatrix_dok', ['x'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'isspmatrix_dok', localization, ['x'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'isspmatrix_dok(...)' code ##################

    str_376570 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 524, (-1)), 'str', 'Is x of dok_matrix type?\n\n    Parameters\n    ----------\n    x\n        object to check for being a dok matrix\n\n    Returns\n    -------\n    bool\n        True if x is a dok matrix, False otherwise\n\n    Examples\n    --------\n    >>> from scipy.sparse import dok_matrix, isspmatrix_dok\n    >>> isspmatrix_dok(dok_matrix([[5]]))\n    True\n\n    >>> from scipy.sparse import dok_matrix, csr_matrix, isspmatrix_dok\n    >>> isspmatrix_dok(csr_matrix([[5]]))\n    False\n    ')
    
    # Call to isinstance(...): (line 525)
    # Processing the call arguments (line 525)
    # Getting the type of 'x' (line 525)
    x_376572 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 525, 22), 'x', False)
    # Getting the type of 'dok_matrix' (line 525)
    dok_matrix_376573 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 525, 25), 'dok_matrix', False)
    # Processing the call keyword arguments (line 525)
    kwargs_376574 = {}
    # Getting the type of 'isinstance' (line 525)
    isinstance_376571 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 525, 11), 'isinstance', False)
    # Calling isinstance(args, kwargs) (line 525)
    isinstance_call_result_376575 = invoke(stypy.reporting.localization.Localization(__file__, 525, 11), isinstance_376571, *[x_376572, dok_matrix_376573], **kwargs_376574)
    
    # Assigning a type to the variable 'stypy_return_type' (line 525)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 525, 4), 'stypy_return_type', isinstance_call_result_376575)
    
    # ################# End of 'isspmatrix_dok(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'isspmatrix_dok' in the type store
    # Getting the type of 'stypy_return_type' (line 502)
    stypy_return_type_376576 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 502, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_376576)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'isspmatrix_dok'
    return stypy_return_type_376576

# Assigning a type to the variable 'isspmatrix_dok' (line 502)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 502, 0), 'isspmatrix_dok', isspmatrix_dok)

@norecursion
def _prod(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_prod'
    module_type_store = module_type_store.open_function_context('_prod', 528, 0, False)
    
    # Passed parameters checking function
    _prod.stypy_localization = localization
    _prod.stypy_type_of_self = None
    _prod.stypy_type_store = module_type_store
    _prod.stypy_function_name = '_prod'
    _prod.stypy_param_names_list = ['x']
    _prod.stypy_varargs_param_name = None
    _prod.stypy_kwargs_param_name = None
    _prod.stypy_call_defaults = defaults
    _prod.stypy_call_varargs = varargs
    _prod.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_prod', ['x'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_prod', localization, ['x'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_prod(...)' code ##################

    str_376577 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 529, 4), 'str', 'Product of a list of numbers; ~40x faster vs np.prod for Python tuples')
    
    
    
    # Call to len(...): (line 530)
    # Processing the call arguments (line 530)
    # Getting the type of 'x' (line 530)
    x_376579 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 530, 11), 'x', False)
    # Processing the call keyword arguments (line 530)
    kwargs_376580 = {}
    # Getting the type of 'len' (line 530)
    len_376578 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 530, 7), 'len', False)
    # Calling len(args, kwargs) (line 530)
    len_call_result_376581 = invoke(stypy.reporting.localization.Localization(__file__, 530, 7), len_376578, *[x_376579], **kwargs_376580)
    
    int_376582 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 530, 17), 'int')
    # Applying the binary operator '==' (line 530)
    result_eq_376583 = python_operator(stypy.reporting.localization.Localization(__file__, 530, 7), '==', len_call_result_376581, int_376582)
    
    # Testing the type of an if condition (line 530)
    if_condition_376584 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 530, 4), result_eq_376583)
    # Assigning a type to the variable 'if_condition_376584' (line 530)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 530, 4), 'if_condition_376584', if_condition_376584)
    # SSA begins for if statement (line 530)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    int_376585 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 531, 15), 'int')
    # Assigning a type to the variable 'stypy_return_type' (line 531)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 531, 8), 'stypy_return_type', int_376585)
    # SSA join for if statement (line 530)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to reduce(...): (line 532)
    # Processing the call arguments (line 532)
    # Getting the type of 'operator' (line 532)
    operator_376588 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 532, 28), 'operator', False)
    # Obtaining the member 'mul' of a type (line 532)
    mul_376589 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 532, 28), operator_376588, 'mul')
    # Getting the type of 'x' (line 532)
    x_376590 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 532, 42), 'x', False)
    # Processing the call keyword arguments (line 532)
    kwargs_376591 = {}
    # Getting the type of 'functools' (line 532)
    functools_376586 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 532, 11), 'functools', False)
    # Obtaining the member 'reduce' of a type (line 532)
    reduce_376587 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 532, 11), functools_376586, 'reduce')
    # Calling reduce(args, kwargs) (line 532)
    reduce_call_result_376592 = invoke(stypy.reporting.localization.Localization(__file__, 532, 11), reduce_376587, *[mul_376589, x_376590], **kwargs_376591)
    
    # Assigning a type to the variable 'stypy_return_type' (line 532)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 532, 4), 'stypy_return_type', reduce_call_result_376592)
    
    # ################# End of '_prod(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_prod' in the type store
    # Getting the type of 'stypy_return_type' (line 528)
    stypy_return_type_376593 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 528, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_376593)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_prod'
    return stypy_return_type_376593

# Assigning a type to the variable '_prod' (line 528)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 528, 0), '_prod', _prod)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
