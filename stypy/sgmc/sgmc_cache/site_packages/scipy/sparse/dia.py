
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: '''Sparse DIAgonal format'''
2: 
3: from __future__ import division, print_function, absolute_import
4: 
5: __docformat__ = "restructuredtext en"
6: 
7: __all__ = ['dia_matrix', 'isspmatrix_dia']
8: 
9: import numpy as np
10: 
11: from .base import isspmatrix, _formats, spmatrix
12: from .data import _data_matrix
13: from .sputils import (isshape, upcast_char, getdtype, get_index_dtype,
14:                       get_sum_dtype, validateaxis)
15: from ._sparsetools import dia_matvec
16: 
17: 
18: class dia_matrix(_data_matrix):
19:     '''Sparse matrix with DIAgonal storage
20: 
21:     This can be instantiated in several ways:
22:         dia_matrix(D)
23:             with a dense matrix
24: 
25:         dia_matrix(S)
26:             with another sparse matrix S (equivalent to S.todia())
27: 
28:         dia_matrix((M, N), [dtype])
29:             to construct an empty matrix with shape (M, N),
30:             dtype is optional, defaulting to dtype='d'.
31: 
32:         dia_matrix((data, offsets), shape=(M, N))
33:             where the ``data[k,:]`` stores the diagonal entries for
34:             diagonal ``offsets[k]`` (See example below)
35: 
36:     Attributes
37:     ----------
38:     dtype : dtype
39:         Data type of the matrix
40:     shape : 2-tuple
41:         Shape of the matrix
42:     ndim : int
43:         Number of dimensions (this is always 2)
44:     nnz
45:         Number of nonzero elements
46:     data
47:         DIA format data array of the matrix
48:     offsets
49:         DIA format offset array of the matrix
50: 
51:     Notes
52:     -----
53: 
54:     Sparse matrices can be used in arithmetic operations: they support
55:     addition, subtraction, multiplication, division, and matrix power.
56: 
57:     Examples
58:     --------
59: 
60:     >>> import numpy as np
61:     >>> from scipy.sparse import dia_matrix
62:     >>> dia_matrix((3, 4), dtype=np.int8).toarray()
63:     array([[0, 0, 0, 0],
64:            [0, 0, 0, 0],
65:            [0, 0, 0, 0]], dtype=int8)
66: 
67:     >>> data = np.array([[1, 2, 3, 4]]).repeat(3, axis=0)
68:     >>> offsets = np.array([0, -1, 2])
69:     >>> dia_matrix((data, offsets), shape=(4, 4)).toarray()
70:     array([[1, 0, 3, 0],
71:            [1, 2, 0, 4],
72:            [0, 2, 3, 0],
73:            [0, 0, 3, 4]])
74: 
75:     '''
76:     format = 'dia'
77: 
78:     def __init__(self, arg1, shape=None, dtype=None, copy=False):
79:         _data_matrix.__init__(self)
80: 
81:         if isspmatrix_dia(arg1):
82:             if copy:
83:                 arg1 = arg1.copy()
84:             self.data = arg1.data
85:             self.offsets = arg1.offsets
86:             self.shape = arg1.shape
87:         elif isspmatrix(arg1):
88:             if isspmatrix_dia(arg1) and copy:
89:                 A = arg1.copy()
90:             else:
91:                 A = arg1.todia()
92:             self.data = A.data
93:             self.offsets = A.offsets
94:             self.shape = A.shape
95:         elif isinstance(arg1, tuple):
96:             if isshape(arg1):
97:                 # It's a tuple of matrix dimensions (M, N)
98:                 # create empty matrix
99:                 self.shape = arg1   # spmatrix checks for errors here
100:                 self.data = np.zeros((0,0), getdtype(dtype, default=float))
101:                 idx_dtype = get_index_dtype(maxval=max(self.shape))
102:                 self.offsets = np.zeros((0), dtype=idx_dtype)
103:             else:
104:                 try:
105:                     # Try interpreting it as (data, offsets)
106:                     data, offsets = arg1
107:                 except:
108:                     raise ValueError('unrecognized form for dia_matrix constructor')
109:                 else:
110:                     if shape is None:
111:                         raise ValueError('expected a shape argument')
112:                     self.data = np.atleast_2d(np.array(arg1[0], dtype=dtype, copy=copy))
113:                     self.offsets = np.atleast_1d(np.array(arg1[1],
114:                                                           dtype=get_index_dtype(maxval=max(shape)),
115:                                                           copy=copy))
116:                     self.shape = shape
117:         else:
118:             #must be dense, convert to COO first, then to DIA
119:             try:
120:                 arg1 = np.asarray(arg1)
121:             except:
122:                 raise ValueError("unrecognized form for"
123:                         " %s_matrix constructor" % self.format)
124:             from .coo import coo_matrix
125:             A = coo_matrix(arg1, dtype=dtype, shape=shape).todia()
126:             self.data = A.data
127:             self.offsets = A.offsets
128:             self.shape = A.shape
129: 
130:         if dtype is not None:
131:             self.data = self.data.astype(dtype)
132: 
133:         #check format
134:         if self.offsets.ndim != 1:
135:             raise ValueError('offsets array must have rank 1')
136: 
137:         if self.data.ndim != 2:
138:             raise ValueError('data array must have rank 2')
139: 
140:         if self.data.shape[0] != len(self.offsets):
141:             raise ValueError('number of diagonals (%d) '
142:                     'does not match the number of offsets (%d)'
143:                     % (self.data.shape[0], len(self.offsets)))
144: 
145:         if len(np.unique(self.offsets)) != len(self.offsets):
146:             raise ValueError('offset array contains duplicate values')
147: 
148:     def __repr__(self):
149:         format = _formats[self.getformat()][1]
150:         return "<%dx%d sparse matrix of type '%s'\n" \
151:                "\twith %d stored elements (%d diagonals) in %s format>" % \
152:                (self.shape + (self.dtype.type, self.nnz, self.data.shape[0],
153:                               format))
154: 
155:     def _data_mask(self):
156:         '''Returns a mask of the same shape as self.data, where
157:         mask[i,j] is True when data[i,j] corresponds to a stored element.'''
158:         num_rows, num_cols = self.shape
159:         offset_inds = np.arange(self.data.shape[1])
160:         row = offset_inds - self.offsets[:,None]
161:         mask = (row >= 0)
162:         mask &= (row < num_rows)
163:         mask &= (offset_inds < num_cols)
164:         return mask
165: 
166:     def count_nonzero(self):
167:         mask = self._data_mask()
168:         return np.count_nonzero(self.data[mask])
169: 
170:     def getnnz(self, axis=None):
171:         if axis is not None:
172:             raise NotImplementedError("getnnz over an axis is not implemented "
173:                                       "for DIA format")
174:         M,N = self.shape
175:         nnz = 0
176:         for k in self.offsets:
177:             if k > 0:
178:                 nnz += min(M,N-k)
179:             else:
180:                 nnz += min(M+k,N)
181:         return int(nnz)
182: 
183:     getnnz.__doc__ = spmatrix.getnnz.__doc__
184:     count_nonzero.__doc__ = spmatrix.count_nonzero.__doc__
185: 
186:     def sum(self, axis=None, dtype=None, out=None):
187:         validateaxis(axis)
188: 
189:         if axis is not None and axis < 0:
190:             axis += 2
191: 
192:         res_dtype = get_sum_dtype(self.dtype)
193:         num_rows, num_cols = self.shape
194:         ret = None
195: 
196:         if axis == 0:
197:             mask = self._data_mask()
198:             x = (self.data * mask).sum(axis=0)
199:             if x.shape[0] == num_cols:
200:                 res = x
201:             else:
202:                 res = np.zeros(num_cols, dtype=x.dtype)
203:                 res[:x.shape[0]] = x
204:             ret = np.matrix(res, dtype=res_dtype)
205: 
206:         else:
207:             row_sums = np.zeros(num_rows, dtype=res_dtype)
208:             one = np.ones(num_cols, dtype=res_dtype)
209:             dia_matvec(num_rows, num_cols, len(self.offsets),
210:                        self.data.shape[1], self.offsets, self.data, one, row_sums)
211: 
212:             row_sums = np.matrix(row_sums)
213: 
214:             if axis is None:
215:                 return row_sums.sum(dtype=dtype, out=out)
216: 
217:             if axis is not None:
218:                 row_sums = row_sums.T
219: 
220:             ret = np.matrix(row_sums.sum(axis=axis))
221: 
222:         if out is not None and out.shape != ret.shape:
223:             raise ValueError("dimensions do not match")
224: 
225:         return ret.sum(axis=(), dtype=dtype, out=out)
226: 
227:     sum.__doc__ = spmatrix.sum.__doc__
228: 
229:     def _mul_vector(self, other):
230:         x = other
231: 
232:         y = np.zeros(self.shape[0], dtype=upcast_char(self.dtype.char,
233:                                                        x.dtype.char))
234: 
235:         L = self.data.shape[1]
236: 
237:         M,N = self.shape
238: 
239:         dia_matvec(M,N, len(self.offsets), L, self.offsets, self.data, x.ravel(), y.ravel())
240: 
241:         return y
242: 
243:     def _mul_multimatrix(self, other):
244:         return np.hstack([self._mul_vector(col).reshape(-1,1) for col in other.T])
245: 
246:     def _setdiag(self, values, k=0):
247:         M, N = self.shape
248: 
249:         if values.ndim == 0:
250:             # broadcast
251:             values_n = np.inf
252:         else:
253:             values_n = len(values)
254: 
255:         if k < 0:
256:             n = min(M + k, N, values_n)
257:             min_index = 0
258:             max_index = n
259:         else:
260:             n = min(M, N - k, values_n)
261:             min_index = k
262:             max_index = k + n
263: 
264:         if values.ndim != 0:
265:             # allow also longer sequences
266:             values = values[:n]
267: 
268:         if k in self.offsets:
269:             self.data[self.offsets == k, min_index:max_index] = values
270:         else:
271:             self.offsets = np.append(self.offsets, self.offsets.dtype.type(k))
272:             m = max(max_index, self.data.shape[1])
273:             data = np.zeros((self.data.shape[0]+1, m), dtype=self.data.dtype)
274:             data[:-1,:self.data.shape[1]] = self.data
275:             data[-1, min_index:max_index] = values
276:             self.data = data
277: 
278:     def todia(self, copy=False):
279:         if copy:
280:             return self.copy()
281:         else:
282:             return self
283: 
284:     todia.__doc__ = spmatrix.todia.__doc__
285: 
286:     def transpose(self, axes=None, copy=False):
287:         if axes is not None:
288:             raise ValueError(("Sparse matrices do not support "
289:                               "an 'axes' parameter because swapping "
290:                               "dimensions is the only logical permutation."))
291: 
292:         num_rows, num_cols = self.shape
293:         max_dim = max(self.shape)
294: 
295:         # flip diagonal offsets
296:         offsets = -self.offsets
297: 
298:         # re-align the data matrix
299:         r = np.arange(len(offsets), dtype=np.intc)[:, None]
300:         c = np.arange(num_rows, dtype=np.intc) - (offsets % max_dim)[:, None]
301:         pad_amount = max(0, max_dim-self.data.shape[1])
302:         data = np.hstack((self.data, np.zeros((self.data.shape[0], pad_amount),
303:                                               dtype=self.data.dtype)))
304:         data = data[r, c]
305:         return dia_matrix((data, offsets), shape=(
306:             num_cols, num_rows), copy=copy)
307: 
308:     transpose.__doc__ = spmatrix.transpose.__doc__
309: 
310:     def diagonal(self, k=0):
311:         rows, cols = self.shape
312:         if k <= -rows or k >= cols:
313:             raise ValueError("k exceeds matrix dimensions")
314:         idx, = np.where(self.offsets == k)
315:         first_col, last_col = max(0, k), min(rows + k, cols)
316:         if idx.size == 0:
317:             return np.zeros(last_col - first_col, dtype=self.data.dtype)
318:         return self.data[idx[0], first_col:last_col]
319: 
320:     diagonal.__doc__ = spmatrix.diagonal.__doc__
321: 
322:     def tocsc(self, copy=False):
323:         from .csc import csc_matrix
324:         if self.nnz == 0:
325:             return csc_matrix(self.shape, dtype=self.dtype)
326: 
327:         num_rows, num_cols = self.shape
328:         num_offsets, offset_len = self.data.shape
329:         offset_inds = np.arange(offset_len)
330: 
331:         row = offset_inds - self.offsets[:,None]
332:         mask = (row >= 0)
333:         mask &= (row < num_rows)
334:         mask &= (offset_inds < num_cols)
335:         mask &= (self.data != 0)
336: 
337:         idx_dtype = get_index_dtype(maxval=max(self.shape))
338:         indptr = np.zeros(num_cols + 1, dtype=idx_dtype)
339:         indptr[1:offset_len+1] = np.cumsum(mask.sum(axis=0))
340:         indptr[offset_len+1:] = indptr[offset_len]
341:         indices = row.T[mask.T].astype(idx_dtype, copy=False)
342:         data = self.data.T[mask.T]
343:         return csc_matrix((data, indices, indptr), shape=self.shape,
344:                           dtype=self.dtype)
345: 
346:     tocsc.__doc__ = spmatrix.tocsc.__doc__
347: 
348:     def tocoo(self, copy=False):
349:         num_rows, num_cols = self.shape
350:         num_offsets, offset_len = self.data.shape
351:         offset_inds = np.arange(offset_len)
352: 
353:         row = offset_inds - self.offsets[:,None]
354:         mask = (row >= 0)
355:         mask &= (row < num_rows)
356:         mask &= (offset_inds < num_cols)
357:         mask &= (self.data != 0)
358:         row = row[mask]
359:         col = np.tile(offset_inds, num_offsets)[mask.ravel()]
360:         data = self.data[mask]
361: 
362:         from .coo import coo_matrix
363:         A = coo_matrix((data,(row,col)), shape=self.shape, dtype=self.dtype)
364:         A.has_canonical_format = True
365:         return A
366: 
367:     tocoo.__doc__ = spmatrix.tocoo.__doc__
368: 
369:     # needed by _data_matrix
370:     def _with_data(self, data, copy=True):
371:         '''Returns a matrix with the same sparsity structure as self,
372:         but with different data.  By default the structure arrays are copied.
373:         '''
374:         if copy:
375:             return dia_matrix((data, self.offsets.copy()), shape=self.shape)
376:         else:
377:             return dia_matrix((data,self.offsets), shape=self.shape)
378: 
379: 
380: def isspmatrix_dia(x):
381:     '''Is x of dia_matrix type?
382: 
383:     Parameters
384:     ----------
385:     x
386:         object to check for being a dia matrix
387: 
388:     Returns
389:     -------
390:     bool
391:         True if x is a dia matrix, False otherwise
392: 
393:     Examples
394:     --------
395:     >>> from scipy.sparse import dia_matrix, isspmatrix_dia
396:     >>> isspmatrix_dia(dia_matrix([[5]]))
397:     True
398: 
399:     >>> from scipy.sparse import dia_matrix, csr_matrix, isspmatrix_dia
400:     >>> isspmatrix_dia(csr_matrix([[5]]))
401:     False
402:     '''
403:     return isinstance(x, dia_matrix)
404: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_372973 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1, 0), 'str', 'Sparse DIAgonal format')

# Assigning a Str to a Name (line 5):

# Assigning a Str to a Name (line 5):
str_372974 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 16), 'str', 'restructuredtext en')
# Assigning a type to the variable '__docformat__' (line 5)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 0), '__docformat__', str_372974)

# Assigning a List to a Name (line 7):

# Assigning a List to a Name (line 7):
__all__ = ['dia_matrix', 'isspmatrix_dia']
module_type_store.set_exportable_members(['dia_matrix', 'isspmatrix_dia'])

# Obtaining an instance of the builtin type 'list' (line 7)
list_372975 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 10), 'list')
# Adding type elements to the builtin type 'list' instance (line 7)
# Adding element type (line 7)
str_372976 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 11), 'str', 'dia_matrix')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 7, 10), list_372975, str_372976)
# Adding element type (line 7)
str_372977 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 25), 'str', 'isspmatrix_dia')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 7, 10), list_372975, str_372977)

# Assigning a type to the variable '__all__' (line 7)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), '__all__', list_372975)
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 9, 0))

# 'import numpy' statement (line 9)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/sparse/')
import_372978 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'numpy')

if (type(import_372978) is not StypyTypeError):

    if (import_372978 != 'pyd_module'):
        __import__(import_372978)
        sys_modules_372979 = sys.modules[import_372978]
        import_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'np', sys_modules_372979.module_type_store, module_type_store)
    else:
        import numpy as np

        import_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'np', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 9)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 0), 'numpy', import_372978)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/sparse/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 11, 0))

# 'from scipy.sparse.base import isspmatrix, _formats, spmatrix' statement (line 11)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/sparse/')
import_372980 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'scipy.sparse.base')

if (type(import_372980) is not StypyTypeError):

    if (import_372980 != 'pyd_module'):
        __import__(import_372980)
        sys_modules_372981 = sys.modules[import_372980]
        import_from_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'scipy.sparse.base', sys_modules_372981.module_type_store, module_type_store, ['isspmatrix', '_formats', 'spmatrix'])
        nest_module(stypy.reporting.localization.Localization(__file__, 11, 0), __file__, sys_modules_372981, sys_modules_372981.module_type_store, module_type_store)
    else:
        from scipy.sparse.base import isspmatrix, _formats, spmatrix

        import_from_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'scipy.sparse.base', None, module_type_store, ['isspmatrix', '_formats', 'spmatrix'], [isspmatrix, _formats, spmatrix])

else:
    # Assigning a type to the variable 'scipy.sparse.base' (line 11)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 0), 'scipy.sparse.base', import_372980)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/sparse/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 12, 0))

# 'from scipy.sparse.data import _data_matrix' statement (line 12)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/sparse/')
import_372982 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 12, 0), 'scipy.sparse.data')

if (type(import_372982) is not StypyTypeError):

    if (import_372982 != 'pyd_module'):
        __import__(import_372982)
        sys_modules_372983 = sys.modules[import_372982]
        import_from_module(stypy.reporting.localization.Localization(__file__, 12, 0), 'scipy.sparse.data', sys_modules_372983.module_type_store, module_type_store, ['_data_matrix'])
        nest_module(stypy.reporting.localization.Localization(__file__, 12, 0), __file__, sys_modules_372983, sys_modules_372983.module_type_store, module_type_store)
    else:
        from scipy.sparse.data import _data_matrix

        import_from_module(stypy.reporting.localization.Localization(__file__, 12, 0), 'scipy.sparse.data', None, module_type_store, ['_data_matrix'], [_data_matrix])

else:
    # Assigning a type to the variable 'scipy.sparse.data' (line 12)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 0), 'scipy.sparse.data', import_372982)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/sparse/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 13, 0))

# 'from scipy.sparse.sputils import isshape, upcast_char, getdtype, get_index_dtype, get_sum_dtype, validateaxis' statement (line 13)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/sparse/')
import_372984 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 13, 0), 'scipy.sparse.sputils')

if (type(import_372984) is not StypyTypeError):

    if (import_372984 != 'pyd_module'):
        __import__(import_372984)
        sys_modules_372985 = sys.modules[import_372984]
        import_from_module(stypy.reporting.localization.Localization(__file__, 13, 0), 'scipy.sparse.sputils', sys_modules_372985.module_type_store, module_type_store, ['isshape', 'upcast_char', 'getdtype', 'get_index_dtype', 'get_sum_dtype', 'validateaxis'])
        nest_module(stypy.reporting.localization.Localization(__file__, 13, 0), __file__, sys_modules_372985, sys_modules_372985.module_type_store, module_type_store)
    else:
        from scipy.sparse.sputils import isshape, upcast_char, getdtype, get_index_dtype, get_sum_dtype, validateaxis

        import_from_module(stypy.reporting.localization.Localization(__file__, 13, 0), 'scipy.sparse.sputils', None, module_type_store, ['isshape', 'upcast_char', 'getdtype', 'get_index_dtype', 'get_sum_dtype', 'validateaxis'], [isshape, upcast_char, getdtype, get_index_dtype, get_sum_dtype, validateaxis])

else:
    # Assigning a type to the variable 'scipy.sparse.sputils' (line 13)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 0), 'scipy.sparse.sputils', import_372984)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/sparse/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 15, 0))

# 'from scipy.sparse._sparsetools import dia_matvec' statement (line 15)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/sparse/')
import_372986 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 15, 0), 'scipy.sparse._sparsetools')

if (type(import_372986) is not StypyTypeError):

    if (import_372986 != 'pyd_module'):
        __import__(import_372986)
        sys_modules_372987 = sys.modules[import_372986]
        import_from_module(stypy.reporting.localization.Localization(__file__, 15, 0), 'scipy.sparse._sparsetools', sys_modules_372987.module_type_store, module_type_store, ['dia_matvec'])
        nest_module(stypy.reporting.localization.Localization(__file__, 15, 0), __file__, sys_modules_372987, sys_modules_372987.module_type_store, module_type_store)
    else:
        from scipy.sparse._sparsetools import dia_matvec

        import_from_module(stypy.reporting.localization.Localization(__file__, 15, 0), 'scipy.sparse._sparsetools', None, module_type_store, ['dia_matvec'], [dia_matvec])

else:
    # Assigning a type to the variable 'scipy.sparse._sparsetools' (line 15)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 0), 'scipy.sparse._sparsetools', import_372986)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/sparse/')

# Declaration of the 'dia_matrix' class
# Getting the type of '_data_matrix' (line 18)
_data_matrix_372988 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 17), '_data_matrix')

class dia_matrix(_data_matrix_372988, ):
    str_372989 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 75, (-1)), 'str', "Sparse matrix with DIAgonal storage\n\n    This can be instantiated in several ways:\n        dia_matrix(D)\n            with a dense matrix\n\n        dia_matrix(S)\n            with another sparse matrix S (equivalent to S.todia())\n\n        dia_matrix((M, N), [dtype])\n            to construct an empty matrix with shape (M, N),\n            dtype is optional, defaulting to dtype='d'.\n\n        dia_matrix((data, offsets), shape=(M, N))\n            where the ``data[k,:]`` stores the diagonal entries for\n            diagonal ``offsets[k]`` (See example below)\n\n    Attributes\n    ----------\n    dtype : dtype\n        Data type of the matrix\n    shape : 2-tuple\n        Shape of the matrix\n    ndim : int\n        Number of dimensions (this is always 2)\n    nnz\n        Number of nonzero elements\n    data\n        DIA format data array of the matrix\n    offsets\n        DIA format offset array of the matrix\n\n    Notes\n    -----\n\n    Sparse matrices can be used in arithmetic operations: they support\n    addition, subtraction, multiplication, division, and matrix power.\n\n    Examples\n    --------\n\n    >>> import numpy as np\n    >>> from scipy.sparse import dia_matrix\n    >>> dia_matrix((3, 4), dtype=np.int8).toarray()\n    array([[0, 0, 0, 0],\n           [0, 0, 0, 0],\n           [0, 0, 0, 0]], dtype=int8)\n\n    >>> data = np.array([[1, 2, 3, 4]]).repeat(3, axis=0)\n    >>> offsets = np.array([0, -1, 2])\n    >>> dia_matrix((data, offsets), shape=(4, 4)).toarray()\n    array([[1, 0, 3, 0],\n           [1, 2, 0, 4],\n           [0, 2, 3, 0],\n           [0, 0, 3, 4]])\n\n    ")
    
    # Assigning a Str to a Name (line 76):

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 78)
        None_372990 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 35), 'None')
        # Getting the type of 'None' (line 78)
        None_372991 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 47), 'None')
        # Getting the type of 'False' (line 78)
        False_372992 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 58), 'False')
        defaults = [None_372990, None_372991, False_372992]
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 78, 4, False)
        # Assigning a type to the variable 'self' (line 79)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 79, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'dia_matrix.__init__', ['arg1', 'shape', 'dtype', 'copy'], None, None, defaults, varargs, kwargs)

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

        
        # Call to __init__(...): (line 79)
        # Processing the call arguments (line 79)
        # Getting the type of 'self' (line 79)
        self_372995 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 30), 'self', False)
        # Processing the call keyword arguments (line 79)
        kwargs_372996 = {}
        # Getting the type of '_data_matrix' (line 79)
        _data_matrix_372993 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 8), '_data_matrix', False)
        # Obtaining the member '__init__' of a type (line 79)
        init___372994 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 79, 8), _data_matrix_372993, '__init__')
        # Calling __init__(args, kwargs) (line 79)
        init___call_result_372997 = invoke(stypy.reporting.localization.Localization(__file__, 79, 8), init___372994, *[self_372995], **kwargs_372996)
        
        
        
        # Call to isspmatrix_dia(...): (line 81)
        # Processing the call arguments (line 81)
        # Getting the type of 'arg1' (line 81)
        arg1_372999 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 26), 'arg1', False)
        # Processing the call keyword arguments (line 81)
        kwargs_373000 = {}
        # Getting the type of 'isspmatrix_dia' (line 81)
        isspmatrix_dia_372998 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 11), 'isspmatrix_dia', False)
        # Calling isspmatrix_dia(args, kwargs) (line 81)
        isspmatrix_dia_call_result_373001 = invoke(stypy.reporting.localization.Localization(__file__, 81, 11), isspmatrix_dia_372998, *[arg1_372999], **kwargs_373000)
        
        # Testing the type of an if condition (line 81)
        if_condition_373002 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 81, 8), isspmatrix_dia_call_result_373001)
        # Assigning a type to the variable 'if_condition_373002' (line 81)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 81, 8), 'if_condition_373002', if_condition_373002)
        # SSA begins for if statement (line 81)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Getting the type of 'copy' (line 82)
        copy_373003 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 15), 'copy')
        # Testing the type of an if condition (line 82)
        if_condition_373004 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 82, 12), copy_373003)
        # Assigning a type to the variable 'if_condition_373004' (line 82)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 82, 12), 'if_condition_373004', if_condition_373004)
        # SSA begins for if statement (line 82)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 83):
        
        # Assigning a Call to a Name (line 83):
        
        # Call to copy(...): (line 83)
        # Processing the call keyword arguments (line 83)
        kwargs_373007 = {}
        # Getting the type of 'arg1' (line 83)
        arg1_373005 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 23), 'arg1', False)
        # Obtaining the member 'copy' of a type (line 83)
        copy_373006 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 83, 23), arg1_373005, 'copy')
        # Calling copy(args, kwargs) (line 83)
        copy_call_result_373008 = invoke(stypy.reporting.localization.Localization(__file__, 83, 23), copy_373006, *[], **kwargs_373007)
        
        # Assigning a type to the variable 'arg1' (line 83)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 83, 16), 'arg1', copy_call_result_373008)
        # SSA join for if statement (line 82)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Attribute to a Attribute (line 84):
        
        # Assigning a Attribute to a Attribute (line 84):
        # Getting the type of 'arg1' (line 84)
        arg1_373009 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 24), 'arg1')
        # Obtaining the member 'data' of a type (line 84)
        data_373010 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 84, 24), arg1_373009, 'data')
        # Getting the type of 'self' (line 84)
        self_373011 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 12), 'self')
        # Setting the type of the member 'data' of a type (line 84)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 84, 12), self_373011, 'data', data_373010)
        
        # Assigning a Attribute to a Attribute (line 85):
        
        # Assigning a Attribute to a Attribute (line 85):
        # Getting the type of 'arg1' (line 85)
        arg1_373012 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 27), 'arg1')
        # Obtaining the member 'offsets' of a type (line 85)
        offsets_373013 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 85, 27), arg1_373012, 'offsets')
        # Getting the type of 'self' (line 85)
        self_373014 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 12), 'self')
        # Setting the type of the member 'offsets' of a type (line 85)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 85, 12), self_373014, 'offsets', offsets_373013)
        
        # Assigning a Attribute to a Attribute (line 86):
        
        # Assigning a Attribute to a Attribute (line 86):
        # Getting the type of 'arg1' (line 86)
        arg1_373015 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 25), 'arg1')
        # Obtaining the member 'shape' of a type (line 86)
        shape_373016 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 86, 25), arg1_373015, 'shape')
        # Getting the type of 'self' (line 86)
        self_373017 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 12), 'self')
        # Setting the type of the member 'shape' of a type (line 86)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 86, 12), self_373017, 'shape', shape_373016)
        # SSA branch for the else part of an if statement (line 81)
        module_type_store.open_ssa_branch('else')
        
        
        # Call to isspmatrix(...): (line 87)
        # Processing the call arguments (line 87)
        # Getting the type of 'arg1' (line 87)
        arg1_373019 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 24), 'arg1', False)
        # Processing the call keyword arguments (line 87)
        kwargs_373020 = {}
        # Getting the type of 'isspmatrix' (line 87)
        isspmatrix_373018 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 13), 'isspmatrix', False)
        # Calling isspmatrix(args, kwargs) (line 87)
        isspmatrix_call_result_373021 = invoke(stypy.reporting.localization.Localization(__file__, 87, 13), isspmatrix_373018, *[arg1_373019], **kwargs_373020)
        
        # Testing the type of an if condition (line 87)
        if_condition_373022 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 87, 13), isspmatrix_call_result_373021)
        # Assigning a type to the variable 'if_condition_373022' (line 87)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 87, 13), 'if_condition_373022', if_condition_373022)
        # SSA begins for if statement (line 87)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        
        # Evaluating a boolean operation
        
        # Call to isspmatrix_dia(...): (line 88)
        # Processing the call arguments (line 88)
        # Getting the type of 'arg1' (line 88)
        arg1_373024 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 30), 'arg1', False)
        # Processing the call keyword arguments (line 88)
        kwargs_373025 = {}
        # Getting the type of 'isspmatrix_dia' (line 88)
        isspmatrix_dia_373023 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 15), 'isspmatrix_dia', False)
        # Calling isspmatrix_dia(args, kwargs) (line 88)
        isspmatrix_dia_call_result_373026 = invoke(stypy.reporting.localization.Localization(__file__, 88, 15), isspmatrix_dia_373023, *[arg1_373024], **kwargs_373025)
        
        # Getting the type of 'copy' (line 88)
        copy_373027 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 40), 'copy')
        # Applying the binary operator 'and' (line 88)
        result_and_keyword_373028 = python_operator(stypy.reporting.localization.Localization(__file__, 88, 15), 'and', isspmatrix_dia_call_result_373026, copy_373027)
        
        # Testing the type of an if condition (line 88)
        if_condition_373029 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 88, 12), result_and_keyword_373028)
        # Assigning a type to the variable 'if_condition_373029' (line 88)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 88, 12), 'if_condition_373029', if_condition_373029)
        # SSA begins for if statement (line 88)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 89):
        
        # Assigning a Call to a Name (line 89):
        
        # Call to copy(...): (line 89)
        # Processing the call keyword arguments (line 89)
        kwargs_373032 = {}
        # Getting the type of 'arg1' (line 89)
        arg1_373030 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 20), 'arg1', False)
        # Obtaining the member 'copy' of a type (line 89)
        copy_373031 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 89, 20), arg1_373030, 'copy')
        # Calling copy(args, kwargs) (line 89)
        copy_call_result_373033 = invoke(stypy.reporting.localization.Localization(__file__, 89, 20), copy_373031, *[], **kwargs_373032)
        
        # Assigning a type to the variable 'A' (line 89)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 89, 16), 'A', copy_call_result_373033)
        # SSA branch for the else part of an if statement (line 88)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Call to a Name (line 91):
        
        # Assigning a Call to a Name (line 91):
        
        # Call to todia(...): (line 91)
        # Processing the call keyword arguments (line 91)
        kwargs_373036 = {}
        # Getting the type of 'arg1' (line 91)
        arg1_373034 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 20), 'arg1', False)
        # Obtaining the member 'todia' of a type (line 91)
        todia_373035 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 91, 20), arg1_373034, 'todia')
        # Calling todia(args, kwargs) (line 91)
        todia_call_result_373037 = invoke(stypy.reporting.localization.Localization(__file__, 91, 20), todia_373035, *[], **kwargs_373036)
        
        # Assigning a type to the variable 'A' (line 91)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 91, 16), 'A', todia_call_result_373037)
        # SSA join for if statement (line 88)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Attribute to a Attribute (line 92):
        
        # Assigning a Attribute to a Attribute (line 92):
        # Getting the type of 'A' (line 92)
        A_373038 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 24), 'A')
        # Obtaining the member 'data' of a type (line 92)
        data_373039 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 92, 24), A_373038, 'data')
        # Getting the type of 'self' (line 92)
        self_373040 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 12), 'self')
        # Setting the type of the member 'data' of a type (line 92)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 92, 12), self_373040, 'data', data_373039)
        
        # Assigning a Attribute to a Attribute (line 93):
        
        # Assigning a Attribute to a Attribute (line 93):
        # Getting the type of 'A' (line 93)
        A_373041 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 27), 'A')
        # Obtaining the member 'offsets' of a type (line 93)
        offsets_373042 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 93, 27), A_373041, 'offsets')
        # Getting the type of 'self' (line 93)
        self_373043 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 12), 'self')
        # Setting the type of the member 'offsets' of a type (line 93)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 93, 12), self_373043, 'offsets', offsets_373042)
        
        # Assigning a Attribute to a Attribute (line 94):
        
        # Assigning a Attribute to a Attribute (line 94):
        # Getting the type of 'A' (line 94)
        A_373044 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 25), 'A')
        # Obtaining the member 'shape' of a type (line 94)
        shape_373045 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 94, 25), A_373044, 'shape')
        # Getting the type of 'self' (line 94)
        self_373046 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 12), 'self')
        # Setting the type of the member 'shape' of a type (line 94)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 94, 12), self_373046, 'shape', shape_373045)
        # SSA branch for the else part of an if statement (line 87)
        module_type_store.open_ssa_branch('else')
        
        # Type idiom detected: calculating its left and rigth part (line 95)
        # Getting the type of 'tuple' (line 95)
        tuple_373047 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 30), 'tuple')
        # Getting the type of 'arg1' (line 95)
        arg1_373048 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 24), 'arg1')
        
        (may_be_373049, more_types_in_union_373050) = may_be_subtype(tuple_373047, arg1_373048)

        if may_be_373049:

            if more_types_in_union_373050:
                # Runtime conditional SSA (line 95)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            # Assigning a type to the variable 'arg1' (line 95)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 95, 13), 'arg1', remove_not_subtype_from_union(arg1_373048, tuple))
            
            
            # Call to isshape(...): (line 96)
            # Processing the call arguments (line 96)
            # Getting the type of 'arg1' (line 96)
            arg1_373052 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 23), 'arg1', False)
            # Processing the call keyword arguments (line 96)
            kwargs_373053 = {}
            # Getting the type of 'isshape' (line 96)
            isshape_373051 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 15), 'isshape', False)
            # Calling isshape(args, kwargs) (line 96)
            isshape_call_result_373054 = invoke(stypy.reporting.localization.Localization(__file__, 96, 15), isshape_373051, *[arg1_373052], **kwargs_373053)
            
            # Testing the type of an if condition (line 96)
            if_condition_373055 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 96, 12), isshape_call_result_373054)
            # Assigning a type to the variable 'if_condition_373055' (line 96)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 96, 12), 'if_condition_373055', if_condition_373055)
            # SSA begins for if statement (line 96)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a Name to a Attribute (line 99):
            
            # Assigning a Name to a Attribute (line 99):
            # Getting the type of 'arg1' (line 99)
            arg1_373056 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 29), 'arg1')
            # Getting the type of 'self' (line 99)
            self_373057 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 16), 'self')
            # Setting the type of the member 'shape' of a type (line 99)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 99, 16), self_373057, 'shape', arg1_373056)
            
            # Assigning a Call to a Attribute (line 100):
            
            # Assigning a Call to a Attribute (line 100):
            
            # Call to zeros(...): (line 100)
            # Processing the call arguments (line 100)
            
            # Obtaining an instance of the builtin type 'tuple' (line 100)
            tuple_373060 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 100, 38), 'tuple')
            # Adding type elements to the builtin type 'tuple' instance (line 100)
            # Adding element type (line 100)
            int_373061 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 100, 38), 'int')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 100, 38), tuple_373060, int_373061)
            # Adding element type (line 100)
            int_373062 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 100, 40), 'int')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 100, 38), tuple_373060, int_373062)
            
            
            # Call to getdtype(...): (line 100)
            # Processing the call arguments (line 100)
            # Getting the type of 'dtype' (line 100)
            dtype_373064 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 53), 'dtype', False)
            # Processing the call keyword arguments (line 100)
            # Getting the type of 'float' (line 100)
            float_373065 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 68), 'float', False)
            keyword_373066 = float_373065
            kwargs_373067 = {'default': keyword_373066}
            # Getting the type of 'getdtype' (line 100)
            getdtype_373063 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 44), 'getdtype', False)
            # Calling getdtype(args, kwargs) (line 100)
            getdtype_call_result_373068 = invoke(stypy.reporting.localization.Localization(__file__, 100, 44), getdtype_373063, *[dtype_373064], **kwargs_373067)
            
            # Processing the call keyword arguments (line 100)
            kwargs_373069 = {}
            # Getting the type of 'np' (line 100)
            np_373058 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 28), 'np', False)
            # Obtaining the member 'zeros' of a type (line 100)
            zeros_373059 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 100, 28), np_373058, 'zeros')
            # Calling zeros(args, kwargs) (line 100)
            zeros_call_result_373070 = invoke(stypy.reporting.localization.Localization(__file__, 100, 28), zeros_373059, *[tuple_373060, getdtype_call_result_373068], **kwargs_373069)
            
            # Getting the type of 'self' (line 100)
            self_373071 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 16), 'self')
            # Setting the type of the member 'data' of a type (line 100)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 100, 16), self_373071, 'data', zeros_call_result_373070)
            
            # Assigning a Call to a Name (line 101):
            
            # Assigning a Call to a Name (line 101):
            
            # Call to get_index_dtype(...): (line 101)
            # Processing the call keyword arguments (line 101)
            
            # Call to max(...): (line 101)
            # Processing the call arguments (line 101)
            # Getting the type of 'self' (line 101)
            self_373074 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 55), 'self', False)
            # Obtaining the member 'shape' of a type (line 101)
            shape_373075 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 101, 55), self_373074, 'shape')
            # Processing the call keyword arguments (line 101)
            kwargs_373076 = {}
            # Getting the type of 'max' (line 101)
            max_373073 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 51), 'max', False)
            # Calling max(args, kwargs) (line 101)
            max_call_result_373077 = invoke(stypy.reporting.localization.Localization(__file__, 101, 51), max_373073, *[shape_373075], **kwargs_373076)
            
            keyword_373078 = max_call_result_373077
            kwargs_373079 = {'maxval': keyword_373078}
            # Getting the type of 'get_index_dtype' (line 101)
            get_index_dtype_373072 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 28), 'get_index_dtype', False)
            # Calling get_index_dtype(args, kwargs) (line 101)
            get_index_dtype_call_result_373080 = invoke(stypy.reporting.localization.Localization(__file__, 101, 28), get_index_dtype_373072, *[], **kwargs_373079)
            
            # Assigning a type to the variable 'idx_dtype' (line 101)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 101, 16), 'idx_dtype', get_index_dtype_call_result_373080)
            
            # Assigning a Call to a Attribute (line 102):
            
            # Assigning a Call to a Attribute (line 102):
            
            # Call to zeros(...): (line 102)
            # Processing the call arguments (line 102)
            int_373083 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 102, 41), 'int')
            # Processing the call keyword arguments (line 102)
            # Getting the type of 'idx_dtype' (line 102)
            idx_dtype_373084 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 51), 'idx_dtype', False)
            keyword_373085 = idx_dtype_373084
            kwargs_373086 = {'dtype': keyword_373085}
            # Getting the type of 'np' (line 102)
            np_373081 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 31), 'np', False)
            # Obtaining the member 'zeros' of a type (line 102)
            zeros_373082 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 102, 31), np_373081, 'zeros')
            # Calling zeros(args, kwargs) (line 102)
            zeros_call_result_373087 = invoke(stypy.reporting.localization.Localization(__file__, 102, 31), zeros_373082, *[int_373083], **kwargs_373086)
            
            # Getting the type of 'self' (line 102)
            self_373088 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 16), 'self')
            # Setting the type of the member 'offsets' of a type (line 102)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 102, 16), self_373088, 'offsets', zeros_call_result_373087)
            # SSA branch for the else part of an if statement (line 96)
            module_type_store.open_ssa_branch('else')
            
            
            # SSA begins for try-except statement (line 104)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
            
            # Assigning a Name to a Tuple (line 106):
            
            # Assigning a Subscript to a Name (line 106):
            
            # Obtaining the type of the subscript
            int_373089 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 106, 20), 'int')
            # Getting the type of 'arg1' (line 106)
            arg1_373090 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 36), 'arg1')
            # Obtaining the member '__getitem__' of a type (line 106)
            getitem___373091 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 106, 20), arg1_373090, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 106)
            subscript_call_result_373092 = invoke(stypy.reporting.localization.Localization(__file__, 106, 20), getitem___373091, int_373089)
            
            # Assigning a type to the variable 'tuple_var_assignment_372946' (line 106)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 106, 20), 'tuple_var_assignment_372946', subscript_call_result_373092)
            
            # Assigning a Subscript to a Name (line 106):
            
            # Obtaining the type of the subscript
            int_373093 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 106, 20), 'int')
            # Getting the type of 'arg1' (line 106)
            arg1_373094 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 36), 'arg1')
            # Obtaining the member '__getitem__' of a type (line 106)
            getitem___373095 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 106, 20), arg1_373094, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 106)
            subscript_call_result_373096 = invoke(stypy.reporting.localization.Localization(__file__, 106, 20), getitem___373095, int_373093)
            
            # Assigning a type to the variable 'tuple_var_assignment_372947' (line 106)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 106, 20), 'tuple_var_assignment_372947', subscript_call_result_373096)
            
            # Assigning a Name to a Name (line 106):
            # Getting the type of 'tuple_var_assignment_372946' (line 106)
            tuple_var_assignment_372946_373097 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 20), 'tuple_var_assignment_372946')
            # Assigning a type to the variable 'data' (line 106)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 106, 20), 'data', tuple_var_assignment_372946_373097)
            
            # Assigning a Name to a Name (line 106):
            # Getting the type of 'tuple_var_assignment_372947' (line 106)
            tuple_var_assignment_372947_373098 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 20), 'tuple_var_assignment_372947')
            # Assigning a type to the variable 'offsets' (line 106)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 106, 26), 'offsets', tuple_var_assignment_372947_373098)
            # SSA branch for the except part of a try statement (line 104)
            # SSA branch for the except '<any exception>' branch of a try statement (line 104)
            module_type_store.open_ssa_branch('except')
            
            # Call to ValueError(...): (line 108)
            # Processing the call arguments (line 108)
            str_373100 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 108, 37), 'str', 'unrecognized form for dia_matrix constructor')
            # Processing the call keyword arguments (line 108)
            kwargs_373101 = {}
            # Getting the type of 'ValueError' (line 108)
            ValueError_373099 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 26), 'ValueError', False)
            # Calling ValueError(args, kwargs) (line 108)
            ValueError_call_result_373102 = invoke(stypy.reporting.localization.Localization(__file__, 108, 26), ValueError_373099, *[str_373100], **kwargs_373101)
            
            ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 108, 20), ValueError_call_result_373102, 'raise parameter', BaseException)
            # SSA branch for the else branch of a try statement (line 104)
            module_type_store.open_ssa_branch('except else')
            
            # Type idiom detected: calculating its left and rigth part (line 110)
            # Getting the type of 'shape' (line 110)
            shape_373103 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 23), 'shape')
            # Getting the type of 'None' (line 110)
            None_373104 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 32), 'None')
            
            (may_be_373105, more_types_in_union_373106) = may_be_none(shape_373103, None_373104)

            if may_be_373105:

                if more_types_in_union_373106:
                    # Runtime conditional SSA (line 110)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
                else:
                    module_type_store = module_type_store

                
                # Call to ValueError(...): (line 111)
                # Processing the call arguments (line 111)
                str_373108 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 111, 41), 'str', 'expected a shape argument')
                # Processing the call keyword arguments (line 111)
                kwargs_373109 = {}
                # Getting the type of 'ValueError' (line 111)
                ValueError_373107 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 30), 'ValueError', False)
                # Calling ValueError(args, kwargs) (line 111)
                ValueError_call_result_373110 = invoke(stypy.reporting.localization.Localization(__file__, 111, 30), ValueError_373107, *[str_373108], **kwargs_373109)
                
                ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 111, 24), ValueError_call_result_373110, 'raise parameter', BaseException)

                if more_types_in_union_373106:
                    # SSA join for if statement (line 110)
                    module_type_store = module_type_store.join_ssa_context()


            
            
            # Assigning a Call to a Attribute (line 112):
            
            # Assigning a Call to a Attribute (line 112):
            
            # Call to atleast_2d(...): (line 112)
            # Processing the call arguments (line 112)
            
            # Call to array(...): (line 112)
            # Processing the call arguments (line 112)
            
            # Obtaining the type of the subscript
            int_373115 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 112, 60), 'int')
            # Getting the type of 'arg1' (line 112)
            arg1_373116 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 55), 'arg1', False)
            # Obtaining the member '__getitem__' of a type (line 112)
            getitem___373117 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 112, 55), arg1_373116, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 112)
            subscript_call_result_373118 = invoke(stypy.reporting.localization.Localization(__file__, 112, 55), getitem___373117, int_373115)
            
            # Processing the call keyword arguments (line 112)
            # Getting the type of 'dtype' (line 112)
            dtype_373119 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 70), 'dtype', False)
            keyword_373120 = dtype_373119
            # Getting the type of 'copy' (line 112)
            copy_373121 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 82), 'copy', False)
            keyword_373122 = copy_373121
            kwargs_373123 = {'dtype': keyword_373120, 'copy': keyword_373122}
            # Getting the type of 'np' (line 112)
            np_373113 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 46), 'np', False)
            # Obtaining the member 'array' of a type (line 112)
            array_373114 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 112, 46), np_373113, 'array')
            # Calling array(args, kwargs) (line 112)
            array_call_result_373124 = invoke(stypy.reporting.localization.Localization(__file__, 112, 46), array_373114, *[subscript_call_result_373118], **kwargs_373123)
            
            # Processing the call keyword arguments (line 112)
            kwargs_373125 = {}
            # Getting the type of 'np' (line 112)
            np_373111 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 32), 'np', False)
            # Obtaining the member 'atleast_2d' of a type (line 112)
            atleast_2d_373112 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 112, 32), np_373111, 'atleast_2d')
            # Calling atleast_2d(args, kwargs) (line 112)
            atleast_2d_call_result_373126 = invoke(stypy.reporting.localization.Localization(__file__, 112, 32), atleast_2d_373112, *[array_call_result_373124], **kwargs_373125)
            
            # Getting the type of 'self' (line 112)
            self_373127 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 20), 'self')
            # Setting the type of the member 'data' of a type (line 112)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 112, 20), self_373127, 'data', atleast_2d_call_result_373126)
            
            # Assigning a Call to a Attribute (line 113):
            
            # Assigning a Call to a Attribute (line 113):
            
            # Call to atleast_1d(...): (line 113)
            # Processing the call arguments (line 113)
            
            # Call to array(...): (line 113)
            # Processing the call arguments (line 113)
            
            # Obtaining the type of the subscript
            int_373132 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 113, 63), 'int')
            # Getting the type of 'arg1' (line 113)
            arg1_373133 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 58), 'arg1', False)
            # Obtaining the member '__getitem__' of a type (line 113)
            getitem___373134 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 113, 58), arg1_373133, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 113)
            subscript_call_result_373135 = invoke(stypy.reporting.localization.Localization(__file__, 113, 58), getitem___373134, int_373132)
            
            # Processing the call keyword arguments (line 113)
            
            # Call to get_index_dtype(...): (line 114)
            # Processing the call keyword arguments (line 114)
            
            # Call to max(...): (line 114)
            # Processing the call arguments (line 114)
            # Getting the type of 'shape' (line 114)
            shape_373138 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 91), 'shape', False)
            # Processing the call keyword arguments (line 114)
            kwargs_373139 = {}
            # Getting the type of 'max' (line 114)
            max_373137 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 87), 'max', False)
            # Calling max(args, kwargs) (line 114)
            max_call_result_373140 = invoke(stypy.reporting.localization.Localization(__file__, 114, 87), max_373137, *[shape_373138], **kwargs_373139)
            
            keyword_373141 = max_call_result_373140
            kwargs_373142 = {'maxval': keyword_373141}
            # Getting the type of 'get_index_dtype' (line 114)
            get_index_dtype_373136 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 64), 'get_index_dtype', False)
            # Calling get_index_dtype(args, kwargs) (line 114)
            get_index_dtype_call_result_373143 = invoke(stypy.reporting.localization.Localization(__file__, 114, 64), get_index_dtype_373136, *[], **kwargs_373142)
            
            keyword_373144 = get_index_dtype_call_result_373143
            # Getting the type of 'copy' (line 115)
            copy_373145 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 63), 'copy', False)
            keyword_373146 = copy_373145
            kwargs_373147 = {'dtype': keyword_373144, 'copy': keyword_373146}
            # Getting the type of 'np' (line 113)
            np_373130 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 49), 'np', False)
            # Obtaining the member 'array' of a type (line 113)
            array_373131 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 113, 49), np_373130, 'array')
            # Calling array(args, kwargs) (line 113)
            array_call_result_373148 = invoke(stypy.reporting.localization.Localization(__file__, 113, 49), array_373131, *[subscript_call_result_373135], **kwargs_373147)
            
            # Processing the call keyword arguments (line 113)
            kwargs_373149 = {}
            # Getting the type of 'np' (line 113)
            np_373128 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 35), 'np', False)
            # Obtaining the member 'atleast_1d' of a type (line 113)
            atleast_1d_373129 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 113, 35), np_373128, 'atleast_1d')
            # Calling atleast_1d(args, kwargs) (line 113)
            atleast_1d_call_result_373150 = invoke(stypy.reporting.localization.Localization(__file__, 113, 35), atleast_1d_373129, *[array_call_result_373148], **kwargs_373149)
            
            # Getting the type of 'self' (line 113)
            self_373151 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 20), 'self')
            # Setting the type of the member 'offsets' of a type (line 113)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 113, 20), self_373151, 'offsets', atleast_1d_call_result_373150)
            
            # Assigning a Name to a Attribute (line 116):
            
            # Assigning a Name to a Attribute (line 116):
            # Getting the type of 'shape' (line 116)
            shape_373152 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 33), 'shape')
            # Getting the type of 'self' (line 116)
            self_373153 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 20), 'self')
            # Setting the type of the member 'shape' of a type (line 116)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 116, 20), self_373153, 'shape', shape_373152)
            # SSA join for try-except statement (line 104)
            module_type_store = module_type_store.join_ssa_context()
            
            # SSA join for if statement (line 96)
            module_type_store = module_type_store.join_ssa_context()
            

            if more_types_in_union_373050:
                # Runtime conditional SSA for else branch (line 95)
                module_type_store.open_ssa_branch('idiom else')



        if ((not may_be_373049) or more_types_in_union_373050):
            # Assigning a type to the variable 'arg1' (line 95)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 95, 13), 'arg1', remove_subtype_from_union(arg1_373048, tuple))
            
            
            # SSA begins for try-except statement (line 119)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
            
            # Assigning a Call to a Name (line 120):
            
            # Assigning a Call to a Name (line 120):
            
            # Call to asarray(...): (line 120)
            # Processing the call arguments (line 120)
            # Getting the type of 'arg1' (line 120)
            arg1_373156 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 34), 'arg1', False)
            # Processing the call keyword arguments (line 120)
            kwargs_373157 = {}
            # Getting the type of 'np' (line 120)
            np_373154 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 23), 'np', False)
            # Obtaining the member 'asarray' of a type (line 120)
            asarray_373155 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 120, 23), np_373154, 'asarray')
            # Calling asarray(args, kwargs) (line 120)
            asarray_call_result_373158 = invoke(stypy.reporting.localization.Localization(__file__, 120, 23), asarray_373155, *[arg1_373156], **kwargs_373157)
            
            # Assigning a type to the variable 'arg1' (line 120)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 120, 16), 'arg1', asarray_call_result_373158)
            # SSA branch for the except part of a try statement (line 119)
            # SSA branch for the except '<any exception>' branch of a try statement (line 119)
            module_type_store.open_ssa_branch('except')
            
            # Call to ValueError(...): (line 122)
            # Processing the call arguments (line 122)
            str_373160 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 122, 33), 'str', 'unrecognized form for %s_matrix constructor')
            # Getting the type of 'self' (line 123)
            self_373161 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 51), 'self', False)
            # Obtaining the member 'format' of a type (line 123)
            format_373162 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 123, 51), self_373161, 'format')
            # Applying the binary operator '%' (line 122)
            result_mod_373163 = python_operator(stypy.reporting.localization.Localization(__file__, 122, 33), '%', str_373160, format_373162)
            
            # Processing the call keyword arguments (line 122)
            kwargs_373164 = {}
            # Getting the type of 'ValueError' (line 122)
            ValueError_373159 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 22), 'ValueError', False)
            # Calling ValueError(args, kwargs) (line 122)
            ValueError_call_result_373165 = invoke(stypy.reporting.localization.Localization(__file__, 122, 22), ValueError_373159, *[result_mod_373163], **kwargs_373164)
            
            ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 122, 16), ValueError_call_result_373165, 'raise parameter', BaseException)
            # SSA join for try-except statement (line 119)
            module_type_store = module_type_store.join_ssa_context()
            
            stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 124, 12))
            
            # 'from scipy.sparse.coo import coo_matrix' statement (line 124)
            update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/sparse/')
            import_373166 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 124, 12), 'scipy.sparse.coo')

            if (type(import_373166) is not StypyTypeError):

                if (import_373166 != 'pyd_module'):
                    __import__(import_373166)
                    sys_modules_373167 = sys.modules[import_373166]
                    import_from_module(stypy.reporting.localization.Localization(__file__, 124, 12), 'scipy.sparse.coo', sys_modules_373167.module_type_store, module_type_store, ['coo_matrix'])
                    nest_module(stypy.reporting.localization.Localization(__file__, 124, 12), __file__, sys_modules_373167, sys_modules_373167.module_type_store, module_type_store)
                else:
                    from scipy.sparse.coo import coo_matrix

                    import_from_module(stypy.reporting.localization.Localization(__file__, 124, 12), 'scipy.sparse.coo', None, module_type_store, ['coo_matrix'], [coo_matrix])

            else:
                # Assigning a type to the variable 'scipy.sparse.coo' (line 124)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 124, 12), 'scipy.sparse.coo', import_373166)

            remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/sparse/')
            
            
            # Assigning a Call to a Name (line 125):
            
            # Assigning a Call to a Name (line 125):
            
            # Call to todia(...): (line 125)
            # Processing the call keyword arguments (line 125)
            kwargs_373177 = {}
            
            # Call to coo_matrix(...): (line 125)
            # Processing the call arguments (line 125)
            # Getting the type of 'arg1' (line 125)
            arg1_373169 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 27), 'arg1', False)
            # Processing the call keyword arguments (line 125)
            # Getting the type of 'dtype' (line 125)
            dtype_373170 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 39), 'dtype', False)
            keyword_373171 = dtype_373170
            # Getting the type of 'shape' (line 125)
            shape_373172 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 52), 'shape', False)
            keyword_373173 = shape_373172
            kwargs_373174 = {'dtype': keyword_373171, 'shape': keyword_373173}
            # Getting the type of 'coo_matrix' (line 125)
            coo_matrix_373168 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 16), 'coo_matrix', False)
            # Calling coo_matrix(args, kwargs) (line 125)
            coo_matrix_call_result_373175 = invoke(stypy.reporting.localization.Localization(__file__, 125, 16), coo_matrix_373168, *[arg1_373169], **kwargs_373174)
            
            # Obtaining the member 'todia' of a type (line 125)
            todia_373176 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 125, 16), coo_matrix_call_result_373175, 'todia')
            # Calling todia(args, kwargs) (line 125)
            todia_call_result_373178 = invoke(stypy.reporting.localization.Localization(__file__, 125, 16), todia_373176, *[], **kwargs_373177)
            
            # Assigning a type to the variable 'A' (line 125)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 125, 12), 'A', todia_call_result_373178)
            
            # Assigning a Attribute to a Attribute (line 126):
            
            # Assigning a Attribute to a Attribute (line 126):
            # Getting the type of 'A' (line 126)
            A_373179 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 24), 'A')
            # Obtaining the member 'data' of a type (line 126)
            data_373180 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 126, 24), A_373179, 'data')
            # Getting the type of 'self' (line 126)
            self_373181 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 12), 'self')
            # Setting the type of the member 'data' of a type (line 126)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 126, 12), self_373181, 'data', data_373180)
            
            # Assigning a Attribute to a Attribute (line 127):
            
            # Assigning a Attribute to a Attribute (line 127):
            # Getting the type of 'A' (line 127)
            A_373182 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 27), 'A')
            # Obtaining the member 'offsets' of a type (line 127)
            offsets_373183 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 127, 27), A_373182, 'offsets')
            # Getting the type of 'self' (line 127)
            self_373184 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 12), 'self')
            # Setting the type of the member 'offsets' of a type (line 127)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 127, 12), self_373184, 'offsets', offsets_373183)
            
            # Assigning a Attribute to a Attribute (line 128):
            
            # Assigning a Attribute to a Attribute (line 128):
            # Getting the type of 'A' (line 128)
            A_373185 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 25), 'A')
            # Obtaining the member 'shape' of a type (line 128)
            shape_373186 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 128, 25), A_373185, 'shape')
            # Getting the type of 'self' (line 128)
            self_373187 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 12), 'self')
            # Setting the type of the member 'shape' of a type (line 128)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 128, 12), self_373187, 'shape', shape_373186)

            if (may_be_373049 and more_types_in_union_373050):
                # SSA join for if statement (line 95)
                module_type_store = module_type_store.join_ssa_context()


        
        # SSA join for if statement (line 87)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 81)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Type idiom detected: calculating its left and rigth part (line 130)
        # Getting the type of 'dtype' (line 130)
        dtype_373188 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 8), 'dtype')
        # Getting the type of 'None' (line 130)
        None_373189 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 24), 'None')
        
        (may_be_373190, more_types_in_union_373191) = may_not_be_none(dtype_373188, None_373189)

        if may_be_373190:

            if more_types_in_union_373191:
                # Runtime conditional SSA (line 130)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a Call to a Attribute (line 131):
            
            # Assigning a Call to a Attribute (line 131):
            
            # Call to astype(...): (line 131)
            # Processing the call arguments (line 131)
            # Getting the type of 'dtype' (line 131)
            dtype_373195 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 41), 'dtype', False)
            # Processing the call keyword arguments (line 131)
            kwargs_373196 = {}
            # Getting the type of 'self' (line 131)
            self_373192 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 24), 'self', False)
            # Obtaining the member 'data' of a type (line 131)
            data_373193 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 131, 24), self_373192, 'data')
            # Obtaining the member 'astype' of a type (line 131)
            astype_373194 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 131, 24), data_373193, 'astype')
            # Calling astype(args, kwargs) (line 131)
            astype_call_result_373197 = invoke(stypy.reporting.localization.Localization(__file__, 131, 24), astype_373194, *[dtype_373195], **kwargs_373196)
            
            # Getting the type of 'self' (line 131)
            self_373198 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 12), 'self')
            # Setting the type of the member 'data' of a type (line 131)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 131, 12), self_373198, 'data', astype_call_result_373197)

            if more_types_in_union_373191:
                # SSA join for if statement (line 130)
                module_type_store = module_type_store.join_ssa_context()


        
        
        
        # Getting the type of 'self' (line 134)
        self_373199 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 11), 'self')
        # Obtaining the member 'offsets' of a type (line 134)
        offsets_373200 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 134, 11), self_373199, 'offsets')
        # Obtaining the member 'ndim' of a type (line 134)
        ndim_373201 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 134, 11), offsets_373200, 'ndim')
        int_373202 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 134, 32), 'int')
        # Applying the binary operator '!=' (line 134)
        result_ne_373203 = python_operator(stypy.reporting.localization.Localization(__file__, 134, 11), '!=', ndim_373201, int_373202)
        
        # Testing the type of an if condition (line 134)
        if_condition_373204 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 134, 8), result_ne_373203)
        # Assigning a type to the variable 'if_condition_373204' (line 134)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 134, 8), 'if_condition_373204', if_condition_373204)
        # SSA begins for if statement (line 134)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to ValueError(...): (line 135)
        # Processing the call arguments (line 135)
        str_373206 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 135, 29), 'str', 'offsets array must have rank 1')
        # Processing the call keyword arguments (line 135)
        kwargs_373207 = {}
        # Getting the type of 'ValueError' (line 135)
        ValueError_373205 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 18), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 135)
        ValueError_call_result_373208 = invoke(stypy.reporting.localization.Localization(__file__, 135, 18), ValueError_373205, *[str_373206], **kwargs_373207)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 135, 12), ValueError_call_result_373208, 'raise parameter', BaseException)
        # SSA join for if statement (line 134)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Getting the type of 'self' (line 137)
        self_373209 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 11), 'self')
        # Obtaining the member 'data' of a type (line 137)
        data_373210 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 137, 11), self_373209, 'data')
        # Obtaining the member 'ndim' of a type (line 137)
        ndim_373211 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 137, 11), data_373210, 'ndim')
        int_373212 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 137, 29), 'int')
        # Applying the binary operator '!=' (line 137)
        result_ne_373213 = python_operator(stypy.reporting.localization.Localization(__file__, 137, 11), '!=', ndim_373211, int_373212)
        
        # Testing the type of an if condition (line 137)
        if_condition_373214 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 137, 8), result_ne_373213)
        # Assigning a type to the variable 'if_condition_373214' (line 137)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 137, 8), 'if_condition_373214', if_condition_373214)
        # SSA begins for if statement (line 137)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to ValueError(...): (line 138)
        # Processing the call arguments (line 138)
        str_373216 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 138, 29), 'str', 'data array must have rank 2')
        # Processing the call keyword arguments (line 138)
        kwargs_373217 = {}
        # Getting the type of 'ValueError' (line 138)
        ValueError_373215 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 18), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 138)
        ValueError_call_result_373218 = invoke(stypy.reporting.localization.Localization(__file__, 138, 18), ValueError_373215, *[str_373216], **kwargs_373217)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 138, 12), ValueError_call_result_373218, 'raise parameter', BaseException)
        # SSA join for if statement (line 137)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        
        # Obtaining the type of the subscript
        int_373219 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 140, 27), 'int')
        # Getting the type of 'self' (line 140)
        self_373220 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 11), 'self')
        # Obtaining the member 'data' of a type (line 140)
        data_373221 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 140, 11), self_373220, 'data')
        # Obtaining the member 'shape' of a type (line 140)
        shape_373222 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 140, 11), data_373221, 'shape')
        # Obtaining the member '__getitem__' of a type (line 140)
        getitem___373223 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 140, 11), shape_373222, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 140)
        subscript_call_result_373224 = invoke(stypy.reporting.localization.Localization(__file__, 140, 11), getitem___373223, int_373219)
        
        
        # Call to len(...): (line 140)
        # Processing the call arguments (line 140)
        # Getting the type of 'self' (line 140)
        self_373226 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 37), 'self', False)
        # Obtaining the member 'offsets' of a type (line 140)
        offsets_373227 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 140, 37), self_373226, 'offsets')
        # Processing the call keyword arguments (line 140)
        kwargs_373228 = {}
        # Getting the type of 'len' (line 140)
        len_373225 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 33), 'len', False)
        # Calling len(args, kwargs) (line 140)
        len_call_result_373229 = invoke(stypy.reporting.localization.Localization(__file__, 140, 33), len_373225, *[offsets_373227], **kwargs_373228)
        
        # Applying the binary operator '!=' (line 140)
        result_ne_373230 = python_operator(stypy.reporting.localization.Localization(__file__, 140, 11), '!=', subscript_call_result_373224, len_call_result_373229)
        
        # Testing the type of an if condition (line 140)
        if_condition_373231 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 140, 8), result_ne_373230)
        # Assigning a type to the variable 'if_condition_373231' (line 140)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 140, 8), 'if_condition_373231', if_condition_373231)
        # SSA begins for if statement (line 140)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to ValueError(...): (line 141)
        # Processing the call arguments (line 141)
        str_373233 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 141, 29), 'str', 'number of diagonals (%d) does not match the number of offsets (%d)')
        
        # Obtaining an instance of the builtin type 'tuple' (line 143)
        tuple_373234 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 143, 23), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 143)
        # Adding element type (line 143)
        
        # Obtaining the type of the subscript
        int_373235 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 143, 39), 'int')
        # Getting the type of 'self' (line 143)
        self_373236 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 23), 'self', False)
        # Obtaining the member 'data' of a type (line 143)
        data_373237 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 143, 23), self_373236, 'data')
        # Obtaining the member 'shape' of a type (line 143)
        shape_373238 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 143, 23), data_373237, 'shape')
        # Obtaining the member '__getitem__' of a type (line 143)
        getitem___373239 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 143, 23), shape_373238, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 143)
        subscript_call_result_373240 = invoke(stypy.reporting.localization.Localization(__file__, 143, 23), getitem___373239, int_373235)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 143, 23), tuple_373234, subscript_call_result_373240)
        # Adding element type (line 143)
        
        # Call to len(...): (line 143)
        # Processing the call arguments (line 143)
        # Getting the type of 'self' (line 143)
        self_373242 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 47), 'self', False)
        # Obtaining the member 'offsets' of a type (line 143)
        offsets_373243 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 143, 47), self_373242, 'offsets')
        # Processing the call keyword arguments (line 143)
        kwargs_373244 = {}
        # Getting the type of 'len' (line 143)
        len_373241 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 43), 'len', False)
        # Calling len(args, kwargs) (line 143)
        len_call_result_373245 = invoke(stypy.reporting.localization.Localization(__file__, 143, 43), len_373241, *[offsets_373243], **kwargs_373244)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 143, 23), tuple_373234, len_call_result_373245)
        
        # Applying the binary operator '%' (line 141)
        result_mod_373246 = python_operator(stypy.reporting.localization.Localization(__file__, 141, 29), '%', str_373233, tuple_373234)
        
        # Processing the call keyword arguments (line 141)
        kwargs_373247 = {}
        # Getting the type of 'ValueError' (line 141)
        ValueError_373232 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 18), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 141)
        ValueError_call_result_373248 = invoke(stypy.reporting.localization.Localization(__file__, 141, 18), ValueError_373232, *[result_mod_373246], **kwargs_373247)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 141, 12), ValueError_call_result_373248, 'raise parameter', BaseException)
        # SSA join for if statement (line 140)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        
        # Call to len(...): (line 145)
        # Processing the call arguments (line 145)
        
        # Call to unique(...): (line 145)
        # Processing the call arguments (line 145)
        # Getting the type of 'self' (line 145)
        self_373252 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 25), 'self', False)
        # Obtaining the member 'offsets' of a type (line 145)
        offsets_373253 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 145, 25), self_373252, 'offsets')
        # Processing the call keyword arguments (line 145)
        kwargs_373254 = {}
        # Getting the type of 'np' (line 145)
        np_373250 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 15), 'np', False)
        # Obtaining the member 'unique' of a type (line 145)
        unique_373251 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 145, 15), np_373250, 'unique')
        # Calling unique(args, kwargs) (line 145)
        unique_call_result_373255 = invoke(stypy.reporting.localization.Localization(__file__, 145, 15), unique_373251, *[offsets_373253], **kwargs_373254)
        
        # Processing the call keyword arguments (line 145)
        kwargs_373256 = {}
        # Getting the type of 'len' (line 145)
        len_373249 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 11), 'len', False)
        # Calling len(args, kwargs) (line 145)
        len_call_result_373257 = invoke(stypy.reporting.localization.Localization(__file__, 145, 11), len_373249, *[unique_call_result_373255], **kwargs_373256)
        
        
        # Call to len(...): (line 145)
        # Processing the call arguments (line 145)
        # Getting the type of 'self' (line 145)
        self_373259 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 47), 'self', False)
        # Obtaining the member 'offsets' of a type (line 145)
        offsets_373260 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 145, 47), self_373259, 'offsets')
        # Processing the call keyword arguments (line 145)
        kwargs_373261 = {}
        # Getting the type of 'len' (line 145)
        len_373258 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 43), 'len', False)
        # Calling len(args, kwargs) (line 145)
        len_call_result_373262 = invoke(stypy.reporting.localization.Localization(__file__, 145, 43), len_373258, *[offsets_373260], **kwargs_373261)
        
        # Applying the binary operator '!=' (line 145)
        result_ne_373263 = python_operator(stypy.reporting.localization.Localization(__file__, 145, 11), '!=', len_call_result_373257, len_call_result_373262)
        
        # Testing the type of an if condition (line 145)
        if_condition_373264 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 145, 8), result_ne_373263)
        # Assigning a type to the variable 'if_condition_373264' (line 145)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 145, 8), 'if_condition_373264', if_condition_373264)
        # SSA begins for if statement (line 145)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to ValueError(...): (line 146)
        # Processing the call arguments (line 146)
        str_373266 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 146, 29), 'str', 'offset array contains duplicate values')
        # Processing the call keyword arguments (line 146)
        kwargs_373267 = {}
        # Getting the type of 'ValueError' (line 146)
        ValueError_373265 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 18), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 146)
        ValueError_call_result_373268 = invoke(stypy.reporting.localization.Localization(__file__, 146, 18), ValueError_373265, *[str_373266], **kwargs_373267)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 146, 12), ValueError_call_result_373268, 'raise parameter', BaseException)
        # SSA join for if statement (line 145)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def stypy__repr__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__repr__'
        module_type_store = module_type_store.open_function_context('__repr__', 148, 4, False)
        # Assigning a type to the variable 'self' (line 149)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 149, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        dia_matrix.stypy__repr__.__dict__.__setitem__('stypy_localization', localization)
        dia_matrix.stypy__repr__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        dia_matrix.stypy__repr__.__dict__.__setitem__('stypy_type_store', module_type_store)
        dia_matrix.stypy__repr__.__dict__.__setitem__('stypy_function_name', 'dia_matrix.stypy__repr__')
        dia_matrix.stypy__repr__.__dict__.__setitem__('stypy_param_names_list', [])
        dia_matrix.stypy__repr__.__dict__.__setitem__('stypy_varargs_param_name', None)
        dia_matrix.stypy__repr__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        dia_matrix.stypy__repr__.__dict__.__setitem__('stypy_call_defaults', defaults)
        dia_matrix.stypy__repr__.__dict__.__setitem__('stypy_call_varargs', varargs)
        dia_matrix.stypy__repr__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        dia_matrix.stypy__repr__.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'dia_matrix.stypy__repr__', [], None, None, defaults, varargs, kwargs)

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

        
        # Assigning a Subscript to a Name (line 149):
        
        # Assigning a Subscript to a Name (line 149):
        
        # Obtaining the type of the subscript
        int_373269 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 149, 44), 'int')
        
        # Obtaining the type of the subscript
        
        # Call to getformat(...): (line 149)
        # Processing the call keyword arguments (line 149)
        kwargs_373272 = {}
        # Getting the type of 'self' (line 149)
        self_373270 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 26), 'self', False)
        # Obtaining the member 'getformat' of a type (line 149)
        getformat_373271 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 149, 26), self_373270, 'getformat')
        # Calling getformat(args, kwargs) (line 149)
        getformat_call_result_373273 = invoke(stypy.reporting.localization.Localization(__file__, 149, 26), getformat_373271, *[], **kwargs_373272)
        
        # Getting the type of '_formats' (line 149)
        _formats_373274 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 17), '_formats')
        # Obtaining the member '__getitem__' of a type (line 149)
        getitem___373275 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 149, 17), _formats_373274, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 149)
        subscript_call_result_373276 = invoke(stypy.reporting.localization.Localization(__file__, 149, 17), getitem___373275, getformat_call_result_373273)
        
        # Obtaining the member '__getitem__' of a type (line 149)
        getitem___373277 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 149, 17), subscript_call_result_373276, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 149)
        subscript_call_result_373278 = invoke(stypy.reporting.localization.Localization(__file__, 149, 17), getitem___373277, int_373269)
        
        # Assigning a type to the variable 'format' (line 149)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 149, 8), 'format', subscript_call_result_373278)
        str_373279 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 150, 15), 'str', "<%dx%d sparse matrix of type '%s'\n\twith %d stored elements (%d diagonals) in %s format>")
        # Getting the type of 'self' (line 152)
        self_373280 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 16), 'self')
        # Obtaining the member 'shape' of a type (line 152)
        shape_373281 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 152, 16), self_373280, 'shape')
        
        # Obtaining an instance of the builtin type 'tuple' (line 152)
        tuple_373282 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 152, 30), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 152)
        # Adding element type (line 152)
        # Getting the type of 'self' (line 152)
        self_373283 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 30), 'self')
        # Obtaining the member 'dtype' of a type (line 152)
        dtype_373284 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 152, 30), self_373283, 'dtype')
        # Obtaining the member 'type' of a type (line 152)
        type_373285 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 152, 30), dtype_373284, 'type')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 152, 30), tuple_373282, type_373285)
        # Adding element type (line 152)
        # Getting the type of 'self' (line 152)
        self_373286 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 47), 'self')
        # Obtaining the member 'nnz' of a type (line 152)
        nnz_373287 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 152, 47), self_373286, 'nnz')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 152, 30), tuple_373282, nnz_373287)
        # Adding element type (line 152)
        
        # Obtaining the type of the subscript
        int_373288 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 152, 73), 'int')
        # Getting the type of 'self' (line 152)
        self_373289 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 57), 'self')
        # Obtaining the member 'data' of a type (line 152)
        data_373290 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 152, 57), self_373289, 'data')
        # Obtaining the member 'shape' of a type (line 152)
        shape_373291 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 152, 57), data_373290, 'shape')
        # Obtaining the member '__getitem__' of a type (line 152)
        getitem___373292 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 152, 57), shape_373291, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 152)
        subscript_call_result_373293 = invoke(stypy.reporting.localization.Localization(__file__, 152, 57), getitem___373292, int_373288)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 152, 30), tuple_373282, subscript_call_result_373293)
        # Adding element type (line 152)
        # Getting the type of 'format' (line 153)
        format_373294 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 30), 'format')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 152, 30), tuple_373282, format_373294)
        
        # Applying the binary operator '+' (line 152)
        result_add_373295 = python_operator(stypy.reporting.localization.Localization(__file__, 152, 16), '+', shape_373281, tuple_373282)
        
        # Applying the binary operator '%' (line 150)
        result_mod_373296 = python_operator(stypy.reporting.localization.Localization(__file__, 150, 15), '%', str_373279, result_add_373295)
        
        # Assigning a type to the variable 'stypy_return_type' (line 150)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 150, 8), 'stypy_return_type', result_mod_373296)
        
        # ################# End of '__repr__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__repr__' in the type store
        # Getting the type of 'stypy_return_type' (line 148)
        stypy_return_type_373297 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_373297)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__repr__'
        return stypy_return_type_373297


    @norecursion
    def _data_mask(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_data_mask'
        module_type_store = module_type_store.open_function_context('_data_mask', 155, 4, False)
        # Assigning a type to the variable 'self' (line 156)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 156, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        dia_matrix._data_mask.__dict__.__setitem__('stypy_localization', localization)
        dia_matrix._data_mask.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        dia_matrix._data_mask.__dict__.__setitem__('stypy_type_store', module_type_store)
        dia_matrix._data_mask.__dict__.__setitem__('stypy_function_name', 'dia_matrix._data_mask')
        dia_matrix._data_mask.__dict__.__setitem__('stypy_param_names_list', [])
        dia_matrix._data_mask.__dict__.__setitem__('stypy_varargs_param_name', None)
        dia_matrix._data_mask.__dict__.__setitem__('stypy_kwargs_param_name', None)
        dia_matrix._data_mask.__dict__.__setitem__('stypy_call_defaults', defaults)
        dia_matrix._data_mask.__dict__.__setitem__('stypy_call_varargs', varargs)
        dia_matrix._data_mask.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        dia_matrix._data_mask.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'dia_matrix._data_mask', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_data_mask', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_data_mask(...)' code ##################

        str_373298 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 157, (-1)), 'str', 'Returns a mask of the same shape as self.data, where\n        mask[i,j] is True when data[i,j] corresponds to a stored element.')
        
        # Assigning a Attribute to a Tuple (line 158):
        
        # Assigning a Subscript to a Name (line 158):
        
        # Obtaining the type of the subscript
        int_373299 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 158, 8), 'int')
        # Getting the type of 'self' (line 158)
        self_373300 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 29), 'self')
        # Obtaining the member 'shape' of a type (line 158)
        shape_373301 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 158, 29), self_373300, 'shape')
        # Obtaining the member '__getitem__' of a type (line 158)
        getitem___373302 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 158, 8), shape_373301, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 158)
        subscript_call_result_373303 = invoke(stypy.reporting.localization.Localization(__file__, 158, 8), getitem___373302, int_373299)
        
        # Assigning a type to the variable 'tuple_var_assignment_372948' (line 158)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 158, 8), 'tuple_var_assignment_372948', subscript_call_result_373303)
        
        # Assigning a Subscript to a Name (line 158):
        
        # Obtaining the type of the subscript
        int_373304 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 158, 8), 'int')
        # Getting the type of 'self' (line 158)
        self_373305 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 29), 'self')
        # Obtaining the member 'shape' of a type (line 158)
        shape_373306 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 158, 29), self_373305, 'shape')
        # Obtaining the member '__getitem__' of a type (line 158)
        getitem___373307 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 158, 8), shape_373306, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 158)
        subscript_call_result_373308 = invoke(stypy.reporting.localization.Localization(__file__, 158, 8), getitem___373307, int_373304)
        
        # Assigning a type to the variable 'tuple_var_assignment_372949' (line 158)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 158, 8), 'tuple_var_assignment_372949', subscript_call_result_373308)
        
        # Assigning a Name to a Name (line 158):
        # Getting the type of 'tuple_var_assignment_372948' (line 158)
        tuple_var_assignment_372948_373309 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 8), 'tuple_var_assignment_372948')
        # Assigning a type to the variable 'num_rows' (line 158)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 158, 8), 'num_rows', tuple_var_assignment_372948_373309)
        
        # Assigning a Name to a Name (line 158):
        # Getting the type of 'tuple_var_assignment_372949' (line 158)
        tuple_var_assignment_372949_373310 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 8), 'tuple_var_assignment_372949')
        # Assigning a type to the variable 'num_cols' (line 158)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 158, 18), 'num_cols', tuple_var_assignment_372949_373310)
        
        # Assigning a Call to a Name (line 159):
        
        # Assigning a Call to a Name (line 159):
        
        # Call to arange(...): (line 159)
        # Processing the call arguments (line 159)
        
        # Obtaining the type of the subscript
        int_373313 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 159, 48), 'int')
        # Getting the type of 'self' (line 159)
        self_373314 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 32), 'self', False)
        # Obtaining the member 'data' of a type (line 159)
        data_373315 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 159, 32), self_373314, 'data')
        # Obtaining the member 'shape' of a type (line 159)
        shape_373316 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 159, 32), data_373315, 'shape')
        # Obtaining the member '__getitem__' of a type (line 159)
        getitem___373317 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 159, 32), shape_373316, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 159)
        subscript_call_result_373318 = invoke(stypy.reporting.localization.Localization(__file__, 159, 32), getitem___373317, int_373313)
        
        # Processing the call keyword arguments (line 159)
        kwargs_373319 = {}
        # Getting the type of 'np' (line 159)
        np_373311 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 22), 'np', False)
        # Obtaining the member 'arange' of a type (line 159)
        arange_373312 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 159, 22), np_373311, 'arange')
        # Calling arange(args, kwargs) (line 159)
        arange_call_result_373320 = invoke(stypy.reporting.localization.Localization(__file__, 159, 22), arange_373312, *[subscript_call_result_373318], **kwargs_373319)
        
        # Assigning a type to the variable 'offset_inds' (line 159)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 159, 8), 'offset_inds', arange_call_result_373320)
        
        # Assigning a BinOp to a Name (line 160):
        
        # Assigning a BinOp to a Name (line 160):
        # Getting the type of 'offset_inds' (line 160)
        offset_inds_373321 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 14), 'offset_inds')
        
        # Obtaining the type of the subscript
        slice_373322 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 160, 28), None, None, None)
        # Getting the type of 'None' (line 160)
        None_373323 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 43), 'None')
        # Getting the type of 'self' (line 160)
        self_373324 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 28), 'self')
        # Obtaining the member 'offsets' of a type (line 160)
        offsets_373325 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 160, 28), self_373324, 'offsets')
        # Obtaining the member '__getitem__' of a type (line 160)
        getitem___373326 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 160, 28), offsets_373325, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 160)
        subscript_call_result_373327 = invoke(stypy.reporting.localization.Localization(__file__, 160, 28), getitem___373326, (slice_373322, None_373323))
        
        # Applying the binary operator '-' (line 160)
        result_sub_373328 = python_operator(stypy.reporting.localization.Localization(__file__, 160, 14), '-', offset_inds_373321, subscript_call_result_373327)
        
        # Assigning a type to the variable 'row' (line 160)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 160, 8), 'row', result_sub_373328)
        
        # Assigning a Compare to a Name (line 161):
        
        # Assigning a Compare to a Name (line 161):
        
        # Getting the type of 'row' (line 161)
        row_373329 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 16), 'row')
        int_373330 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 161, 23), 'int')
        # Applying the binary operator '>=' (line 161)
        result_ge_373331 = python_operator(stypy.reporting.localization.Localization(__file__, 161, 16), '>=', row_373329, int_373330)
        
        # Assigning a type to the variable 'mask' (line 161)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 161, 8), 'mask', result_ge_373331)
        
        # Getting the type of 'mask' (line 162)
        mask_373332 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 8), 'mask')
        
        # Getting the type of 'row' (line 162)
        row_373333 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 17), 'row')
        # Getting the type of 'num_rows' (line 162)
        num_rows_373334 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 23), 'num_rows')
        # Applying the binary operator '<' (line 162)
        result_lt_373335 = python_operator(stypy.reporting.localization.Localization(__file__, 162, 17), '<', row_373333, num_rows_373334)
        
        # Applying the binary operator '&=' (line 162)
        result_iand_373336 = python_operator(stypy.reporting.localization.Localization(__file__, 162, 8), '&=', mask_373332, result_lt_373335)
        # Assigning a type to the variable 'mask' (line 162)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 162, 8), 'mask', result_iand_373336)
        
        
        # Getting the type of 'mask' (line 163)
        mask_373337 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 8), 'mask')
        
        # Getting the type of 'offset_inds' (line 163)
        offset_inds_373338 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 17), 'offset_inds')
        # Getting the type of 'num_cols' (line 163)
        num_cols_373339 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 31), 'num_cols')
        # Applying the binary operator '<' (line 163)
        result_lt_373340 = python_operator(stypy.reporting.localization.Localization(__file__, 163, 17), '<', offset_inds_373338, num_cols_373339)
        
        # Applying the binary operator '&=' (line 163)
        result_iand_373341 = python_operator(stypy.reporting.localization.Localization(__file__, 163, 8), '&=', mask_373337, result_lt_373340)
        # Assigning a type to the variable 'mask' (line 163)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 163, 8), 'mask', result_iand_373341)
        
        # Getting the type of 'mask' (line 164)
        mask_373342 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 15), 'mask')
        # Assigning a type to the variable 'stypy_return_type' (line 164)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 164, 8), 'stypy_return_type', mask_373342)
        
        # ################# End of '_data_mask(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_data_mask' in the type store
        # Getting the type of 'stypy_return_type' (line 155)
        stypy_return_type_373343 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_373343)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_data_mask'
        return stypy_return_type_373343


    @norecursion
    def count_nonzero(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'count_nonzero'
        module_type_store = module_type_store.open_function_context('count_nonzero', 166, 4, False)
        # Assigning a type to the variable 'self' (line 167)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 167, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        dia_matrix.count_nonzero.__dict__.__setitem__('stypy_localization', localization)
        dia_matrix.count_nonzero.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        dia_matrix.count_nonzero.__dict__.__setitem__('stypy_type_store', module_type_store)
        dia_matrix.count_nonzero.__dict__.__setitem__('stypy_function_name', 'dia_matrix.count_nonzero')
        dia_matrix.count_nonzero.__dict__.__setitem__('stypy_param_names_list', [])
        dia_matrix.count_nonzero.__dict__.__setitem__('stypy_varargs_param_name', None)
        dia_matrix.count_nonzero.__dict__.__setitem__('stypy_kwargs_param_name', None)
        dia_matrix.count_nonzero.__dict__.__setitem__('stypy_call_defaults', defaults)
        dia_matrix.count_nonzero.__dict__.__setitem__('stypy_call_varargs', varargs)
        dia_matrix.count_nonzero.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        dia_matrix.count_nonzero.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'dia_matrix.count_nonzero', [], None, None, defaults, varargs, kwargs)

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

        
        # Assigning a Call to a Name (line 167):
        
        # Assigning a Call to a Name (line 167):
        
        # Call to _data_mask(...): (line 167)
        # Processing the call keyword arguments (line 167)
        kwargs_373346 = {}
        # Getting the type of 'self' (line 167)
        self_373344 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 15), 'self', False)
        # Obtaining the member '_data_mask' of a type (line 167)
        _data_mask_373345 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 167, 15), self_373344, '_data_mask')
        # Calling _data_mask(args, kwargs) (line 167)
        _data_mask_call_result_373347 = invoke(stypy.reporting.localization.Localization(__file__, 167, 15), _data_mask_373345, *[], **kwargs_373346)
        
        # Assigning a type to the variable 'mask' (line 167)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 167, 8), 'mask', _data_mask_call_result_373347)
        
        # Call to count_nonzero(...): (line 168)
        # Processing the call arguments (line 168)
        
        # Obtaining the type of the subscript
        # Getting the type of 'mask' (line 168)
        mask_373350 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 42), 'mask', False)
        # Getting the type of 'self' (line 168)
        self_373351 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 32), 'self', False)
        # Obtaining the member 'data' of a type (line 168)
        data_373352 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 168, 32), self_373351, 'data')
        # Obtaining the member '__getitem__' of a type (line 168)
        getitem___373353 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 168, 32), data_373352, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 168)
        subscript_call_result_373354 = invoke(stypy.reporting.localization.Localization(__file__, 168, 32), getitem___373353, mask_373350)
        
        # Processing the call keyword arguments (line 168)
        kwargs_373355 = {}
        # Getting the type of 'np' (line 168)
        np_373348 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 15), 'np', False)
        # Obtaining the member 'count_nonzero' of a type (line 168)
        count_nonzero_373349 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 168, 15), np_373348, 'count_nonzero')
        # Calling count_nonzero(args, kwargs) (line 168)
        count_nonzero_call_result_373356 = invoke(stypy.reporting.localization.Localization(__file__, 168, 15), count_nonzero_373349, *[subscript_call_result_373354], **kwargs_373355)
        
        # Assigning a type to the variable 'stypy_return_type' (line 168)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 168, 8), 'stypy_return_type', count_nonzero_call_result_373356)
        
        # ################# End of 'count_nonzero(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'count_nonzero' in the type store
        # Getting the type of 'stypy_return_type' (line 166)
        stypy_return_type_373357 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_373357)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'count_nonzero'
        return stypy_return_type_373357


    @norecursion
    def getnnz(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 170)
        None_373358 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 26), 'None')
        defaults = [None_373358]
        # Create a new context for function 'getnnz'
        module_type_store = module_type_store.open_function_context('getnnz', 170, 4, False)
        # Assigning a type to the variable 'self' (line 171)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 171, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        dia_matrix.getnnz.__dict__.__setitem__('stypy_localization', localization)
        dia_matrix.getnnz.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        dia_matrix.getnnz.__dict__.__setitem__('stypy_type_store', module_type_store)
        dia_matrix.getnnz.__dict__.__setitem__('stypy_function_name', 'dia_matrix.getnnz')
        dia_matrix.getnnz.__dict__.__setitem__('stypy_param_names_list', ['axis'])
        dia_matrix.getnnz.__dict__.__setitem__('stypy_varargs_param_name', None)
        dia_matrix.getnnz.__dict__.__setitem__('stypy_kwargs_param_name', None)
        dia_matrix.getnnz.__dict__.__setitem__('stypy_call_defaults', defaults)
        dia_matrix.getnnz.__dict__.__setitem__('stypy_call_varargs', varargs)
        dia_matrix.getnnz.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        dia_matrix.getnnz.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'dia_matrix.getnnz', ['axis'], None, None, defaults, varargs, kwargs)

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

        
        # Type idiom detected: calculating its left and rigth part (line 171)
        # Getting the type of 'axis' (line 171)
        axis_373359 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 8), 'axis')
        # Getting the type of 'None' (line 171)
        None_373360 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 23), 'None')
        
        (may_be_373361, more_types_in_union_373362) = may_not_be_none(axis_373359, None_373360)

        if may_be_373361:

            if more_types_in_union_373362:
                # Runtime conditional SSA (line 171)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Call to NotImplementedError(...): (line 172)
            # Processing the call arguments (line 172)
            str_373364 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 172, 38), 'str', 'getnnz over an axis is not implemented for DIA format')
            # Processing the call keyword arguments (line 172)
            kwargs_373365 = {}
            # Getting the type of 'NotImplementedError' (line 172)
            NotImplementedError_373363 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 18), 'NotImplementedError', False)
            # Calling NotImplementedError(args, kwargs) (line 172)
            NotImplementedError_call_result_373366 = invoke(stypy.reporting.localization.Localization(__file__, 172, 18), NotImplementedError_373363, *[str_373364], **kwargs_373365)
            
            ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 172, 12), NotImplementedError_call_result_373366, 'raise parameter', BaseException)

            if more_types_in_union_373362:
                # SSA join for if statement (line 171)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Assigning a Attribute to a Tuple (line 174):
        
        # Assigning a Subscript to a Name (line 174):
        
        # Obtaining the type of the subscript
        int_373367 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 174, 8), 'int')
        # Getting the type of 'self' (line 174)
        self_373368 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 14), 'self')
        # Obtaining the member 'shape' of a type (line 174)
        shape_373369 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 174, 14), self_373368, 'shape')
        # Obtaining the member '__getitem__' of a type (line 174)
        getitem___373370 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 174, 8), shape_373369, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 174)
        subscript_call_result_373371 = invoke(stypy.reporting.localization.Localization(__file__, 174, 8), getitem___373370, int_373367)
        
        # Assigning a type to the variable 'tuple_var_assignment_372950' (line 174)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 174, 8), 'tuple_var_assignment_372950', subscript_call_result_373371)
        
        # Assigning a Subscript to a Name (line 174):
        
        # Obtaining the type of the subscript
        int_373372 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 174, 8), 'int')
        # Getting the type of 'self' (line 174)
        self_373373 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 14), 'self')
        # Obtaining the member 'shape' of a type (line 174)
        shape_373374 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 174, 14), self_373373, 'shape')
        # Obtaining the member '__getitem__' of a type (line 174)
        getitem___373375 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 174, 8), shape_373374, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 174)
        subscript_call_result_373376 = invoke(stypy.reporting.localization.Localization(__file__, 174, 8), getitem___373375, int_373372)
        
        # Assigning a type to the variable 'tuple_var_assignment_372951' (line 174)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 174, 8), 'tuple_var_assignment_372951', subscript_call_result_373376)
        
        # Assigning a Name to a Name (line 174):
        # Getting the type of 'tuple_var_assignment_372950' (line 174)
        tuple_var_assignment_372950_373377 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 8), 'tuple_var_assignment_372950')
        # Assigning a type to the variable 'M' (line 174)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 174, 8), 'M', tuple_var_assignment_372950_373377)
        
        # Assigning a Name to a Name (line 174):
        # Getting the type of 'tuple_var_assignment_372951' (line 174)
        tuple_var_assignment_372951_373378 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 8), 'tuple_var_assignment_372951')
        # Assigning a type to the variable 'N' (line 174)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 174, 10), 'N', tuple_var_assignment_372951_373378)
        
        # Assigning a Num to a Name (line 175):
        
        # Assigning a Num to a Name (line 175):
        int_373379 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 175, 14), 'int')
        # Assigning a type to the variable 'nnz' (line 175)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 175, 8), 'nnz', int_373379)
        
        # Getting the type of 'self' (line 176)
        self_373380 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 17), 'self')
        # Obtaining the member 'offsets' of a type (line 176)
        offsets_373381 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 176, 17), self_373380, 'offsets')
        # Testing the type of a for loop iterable (line 176)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 176, 8), offsets_373381)
        # Getting the type of the for loop variable (line 176)
        for_loop_var_373382 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 176, 8), offsets_373381)
        # Assigning a type to the variable 'k' (line 176)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 176, 8), 'k', for_loop_var_373382)
        # SSA begins for a for statement (line 176)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        
        # Getting the type of 'k' (line 177)
        k_373383 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 15), 'k')
        int_373384 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 177, 19), 'int')
        # Applying the binary operator '>' (line 177)
        result_gt_373385 = python_operator(stypy.reporting.localization.Localization(__file__, 177, 15), '>', k_373383, int_373384)
        
        # Testing the type of an if condition (line 177)
        if_condition_373386 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 177, 12), result_gt_373385)
        # Assigning a type to the variable 'if_condition_373386' (line 177)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 177, 12), 'if_condition_373386', if_condition_373386)
        # SSA begins for if statement (line 177)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Getting the type of 'nnz' (line 178)
        nnz_373387 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 16), 'nnz')
        
        # Call to min(...): (line 178)
        # Processing the call arguments (line 178)
        # Getting the type of 'M' (line 178)
        M_373389 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 27), 'M', False)
        # Getting the type of 'N' (line 178)
        N_373390 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 29), 'N', False)
        # Getting the type of 'k' (line 178)
        k_373391 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 31), 'k', False)
        # Applying the binary operator '-' (line 178)
        result_sub_373392 = python_operator(stypy.reporting.localization.Localization(__file__, 178, 29), '-', N_373390, k_373391)
        
        # Processing the call keyword arguments (line 178)
        kwargs_373393 = {}
        # Getting the type of 'min' (line 178)
        min_373388 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 23), 'min', False)
        # Calling min(args, kwargs) (line 178)
        min_call_result_373394 = invoke(stypy.reporting.localization.Localization(__file__, 178, 23), min_373388, *[M_373389, result_sub_373392], **kwargs_373393)
        
        # Applying the binary operator '+=' (line 178)
        result_iadd_373395 = python_operator(stypy.reporting.localization.Localization(__file__, 178, 16), '+=', nnz_373387, min_call_result_373394)
        # Assigning a type to the variable 'nnz' (line 178)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 178, 16), 'nnz', result_iadd_373395)
        
        # SSA branch for the else part of an if statement (line 177)
        module_type_store.open_ssa_branch('else')
        
        # Getting the type of 'nnz' (line 180)
        nnz_373396 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 16), 'nnz')
        
        # Call to min(...): (line 180)
        # Processing the call arguments (line 180)
        # Getting the type of 'M' (line 180)
        M_373398 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 27), 'M', False)
        # Getting the type of 'k' (line 180)
        k_373399 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 29), 'k', False)
        # Applying the binary operator '+' (line 180)
        result_add_373400 = python_operator(stypy.reporting.localization.Localization(__file__, 180, 27), '+', M_373398, k_373399)
        
        # Getting the type of 'N' (line 180)
        N_373401 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 31), 'N', False)
        # Processing the call keyword arguments (line 180)
        kwargs_373402 = {}
        # Getting the type of 'min' (line 180)
        min_373397 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 23), 'min', False)
        # Calling min(args, kwargs) (line 180)
        min_call_result_373403 = invoke(stypy.reporting.localization.Localization(__file__, 180, 23), min_373397, *[result_add_373400, N_373401], **kwargs_373402)
        
        # Applying the binary operator '+=' (line 180)
        result_iadd_373404 = python_operator(stypy.reporting.localization.Localization(__file__, 180, 16), '+=', nnz_373396, min_call_result_373403)
        # Assigning a type to the variable 'nnz' (line 180)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 180, 16), 'nnz', result_iadd_373404)
        
        # SSA join for if statement (line 177)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to int(...): (line 181)
        # Processing the call arguments (line 181)
        # Getting the type of 'nnz' (line 181)
        nnz_373406 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 19), 'nnz', False)
        # Processing the call keyword arguments (line 181)
        kwargs_373407 = {}
        # Getting the type of 'int' (line 181)
        int_373405 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 15), 'int', False)
        # Calling int(args, kwargs) (line 181)
        int_call_result_373408 = invoke(stypy.reporting.localization.Localization(__file__, 181, 15), int_373405, *[nnz_373406], **kwargs_373407)
        
        # Assigning a type to the variable 'stypy_return_type' (line 181)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 181, 8), 'stypy_return_type', int_call_result_373408)
        
        # ################# End of 'getnnz(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'getnnz' in the type store
        # Getting the type of 'stypy_return_type' (line 170)
        stypy_return_type_373409 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_373409)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'getnnz'
        return stypy_return_type_373409

    
    # Assigning a Attribute to a Attribute (line 183):
    
    # Assigning a Attribute to a Attribute (line 184):

    @norecursion
    def sum(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 186)
        None_373410 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 23), 'None')
        # Getting the type of 'None' (line 186)
        None_373411 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 35), 'None')
        # Getting the type of 'None' (line 186)
        None_373412 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 45), 'None')
        defaults = [None_373410, None_373411, None_373412]
        # Create a new context for function 'sum'
        module_type_store = module_type_store.open_function_context('sum', 186, 4, False)
        # Assigning a type to the variable 'self' (line 187)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 187, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        dia_matrix.sum.__dict__.__setitem__('stypy_localization', localization)
        dia_matrix.sum.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        dia_matrix.sum.__dict__.__setitem__('stypy_type_store', module_type_store)
        dia_matrix.sum.__dict__.__setitem__('stypy_function_name', 'dia_matrix.sum')
        dia_matrix.sum.__dict__.__setitem__('stypy_param_names_list', ['axis', 'dtype', 'out'])
        dia_matrix.sum.__dict__.__setitem__('stypy_varargs_param_name', None)
        dia_matrix.sum.__dict__.__setitem__('stypy_kwargs_param_name', None)
        dia_matrix.sum.__dict__.__setitem__('stypy_call_defaults', defaults)
        dia_matrix.sum.__dict__.__setitem__('stypy_call_varargs', varargs)
        dia_matrix.sum.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        dia_matrix.sum.__dict__.__setitem__('stypy_declared_arg_number', 4)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'dia_matrix.sum', ['axis', 'dtype', 'out'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'sum', localization, ['axis', 'dtype', 'out'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'sum(...)' code ##################

        
        # Call to validateaxis(...): (line 187)
        # Processing the call arguments (line 187)
        # Getting the type of 'axis' (line 187)
        axis_373414 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 21), 'axis', False)
        # Processing the call keyword arguments (line 187)
        kwargs_373415 = {}
        # Getting the type of 'validateaxis' (line 187)
        validateaxis_373413 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 8), 'validateaxis', False)
        # Calling validateaxis(args, kwargs) (line 187)
        validateaxis_call_result_373416 = invoke(stypy.reporting.localization.Localization(__file__, 187, 8), validateaxis_373413, *[axis_373414], **kwargs_373415)
        
        
        
        # Evaluating a boolean operation
        
        # Getting the type of 'axis' (line 189)
        axis_373417 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 11), 'axis')
        # Getting the type of 'None' (line 189)
        None_373418 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 23), 'None')
        # Applying the binary operator 'isnot' (line 189)
        result_is_not_373419 = python_operator(stypy.reporting.localization.Localization(__file__, 189, 11), 'isnot', axis_373417, None_373418)
        
        
        # Getting the type of 'axis' (line 189)
        axis_373420 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 32), 'axis')
        int_373421 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 189, 39), 'int')
        # Applying the binary operator '<' (line 189)
        result_lt_373422 = python_operator(stypy.reporting.localization.Localization(__file__, 189, 32), '<', axis_373420, int_373421)
        
        # Applying the binary operator 'and' (line 189)
        result_and_keyword_373423 = python_operator(stypy.reporting.localization.Localization(__file__, 189, 11), 'and', result_is_not_373419, result_lt_373422)
        
        # Testing the type of an if condition (line 189)
        if_condition_373424 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 189, 8), result_and_keyword_373423)
        # Assigning a type to the variable 'if_condition_373424' (line 189)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 189, 8), 'if_condition_373424', if_condition_373424)
        # SSA begins for if statement (line 189)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Getting the type of 'axis' (line 190)
        axis_373425 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 190, 12), 'axis')
        int_373426 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 190, 20), 'int')
        # Applying the binary operator '+=' (line 190)
        result_iadd_373427 = python_operator(stypy.reporting.localization.Localization(__file__, 190, 12), '+=', axis_373425, int_373426)
        # Assigning a type to the variable 'axis' (line 190)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 190, 12), 'axis', result_iadd_373427)
        
        # SSA join for if statement (line 189)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Name (line 192):
        
        # Assigning a Call to a Name (line 192):
        
        # Call to get_sum_dtype(...): (line 192)
        # Processing the call arguments (line 192)
        # Getting the type of 'self' (line 192)
        self_373429 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 192, 34), 'self', False)
        # Obtaining the member 'dtype' of a type (line 192)
        dtype_373430 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 192, 34), self_373429, 'dtype')
        # Processing the call keyword arguments (line 192)
        kwargs_373431 = {}
        # Getting the type of 'get_sum_dtype' (line 192)
        get_sum_dtype_373428 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 192, 20), 'get_sum_dtype', False)
        # Calling get_sum_dtype(args, kwargs) (line 192)
        get_sum_dtype_call_result_373432 = invoke(stypy.reporting.localization.Localization(__file__, 192, 20), get_sum_dtype_373428, *[dtype_373430], **kwargs_373431)
        
        # Assigning a type to the variable 'res_dtype' (line 192)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 192, 8), 'res_dtype', get_sum_dtype_call_result_373432)
        
        # Assigning a Attribute to a Tuple (line 193):
        
        # Assigning a Subscript to a Name (line 193):
        
        # Obtaining the type of the subscript
        int_373433 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 193, 8), 'int')
        # Getting the type of 'self' (line 193)
        self_373434 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 29), 'self')
        # Obtaining the member 'shape' of a type (line 193)
        shape_373435 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 193, 29), self_373434, 'shape')
        # Obtaining the member '__getitem__' of a type (line 193)
        getitem___373436 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 193, 8), shape_373435, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 193)
        subscript_call_result_373437 = invoke(stypy.reporting.localization.Localization(__file__, 193, 8), getitem___373436, int_373433)
        
        # Assigning a type to the variable 'tuple_var_assignment_372952' (line 193)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 193, 8), 'tuple_var_assignment_372952', subscript_call_result_373437)
        
        # Assigning a Subscript to a Name (line 193):
        
        # Obtaining the type of the subscript
        int_373438 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 193, 8), 'int')
        # Getting the type of 'self' (line 193)
        self_373439 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 29), 'self')
        # Obtaining the member 'shape' of a type (line 193)
        shape_373440 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 193, 29), self_373439, 'shape')
        # Obtaining the member '__getitem__' of a type (line 193)
        getitem___373441 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 193, 8), shape_373440, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 193)
        subscript_call_result_373442 = invoke(stypy.reporting.localization.Localization(__file__, 193, 8), getitem___373441, int_373438)
        
        # Assigning a type to the variable 'tuple_var_assignment_372953' (line 193)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 193, 8), 'tuple_var_assignment_372953', subscript_call_result_373442)
        
        # Assigning a Name to a Name (line 193):
        # Getting the type of 'tuple_var_assignment_372952' (line 193)
        tuple_var_assignment_372952_373443 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 8), 'tuple_var_assignment_372952')
        # Assigning a type to the variable 'num_rows' (line 193)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 193, 8), 'num_rows', tuple_var_assignment_372952_373443)
        
        # Assigning a Name to a Name (line 193):
        # Getting the type of 'tuple_var_assignment_372953' (line 193)
        tuple_var_assignment_372953_373444 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 8), 'tuple_var_assignment_372953')
        # Assigning a type to the variable 'num_cols' (line 193)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 193, 18), 'num_cols', tuple_var_assignment_372953_373444)
        
        # Assigning a Name to a Name (line 194):
        
        # Assigning a Name to a Name (line 194):
        # Getting the type of 'None' (line 194)
        None_373445 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 194, 14), 'None')
        # Assigning a type to the variable 'ret' (line 194)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 194, 8), 'ret', None_373445)
        
        
        # Getting the type of 'axis' (line 196)
        axis_373446 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 196, 11), 'axis')
        int_373447 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 196, 19), 'int')
        # Applying the binary operator '==' (line 196)
        result_eq_373448 = python_operator(stypy.reporting.localization.Localization(__file__, 196, 11), '==', axis_373446, int_373447)
        
        # Testing the type of an if condition (line 196)
        if_condition_373449 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 196, 8), result_eq_373448)
        # Assigning a type to the variable 'if_condition_373449' (line 196)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 196, 8), 'if_condition_373449', if_condition_373449)
        # SSA begins for if statement (line 196)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 197):
        
        # Assigning a Call to a Name (line 197):
        
        # Call to _data_mask(...): (line 197)
        # Processing the call keyword arguments (line 197)
        kwargs_373452 = {}
        # Getting the type of 'self' (line 197)
        self_373450 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 197, 19), 'self', False)
        # Obtaining the member '_data_mask' of a type (line 197)
        _data_mask_373451 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 197, 19), self_373450, '_data_mask')
        # Calling _data_mask(args, kwargs) (line 197)
        _data_mask_call_result_373453 = invoke(stypy.reporting.localization.Localization(__file__, 197, 19), _data_mask_373451, *[], **kwargs_373452)
        
        # Assigning a type to the variable 'mask' (line 197)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 197, 12), 'mask', _data_mask_call_result_373453)
        
        # Assigning a Call to a Name (line 198):
        
        # Assigning a Call to a Name (line 198):
        
        # Call to sum(...): (line 198)
        # Processing the call keyword arguments (line 198)
        int_373459 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 198, 44), 'int')
        keyword_373460 = int_373459
        kwargs_373461 = {'axis': keyword_373460}
        # Getting the type of 'self' (line 198)
        self_373454 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 17), 'self', False)
        # Obtaining the member 'data' of a type (line 198)
        data_373455 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 198, 17), self_373454, 'data')
        # Getting the type of 'mask' (line 198)
        mask_373456 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 29), 'mask', False)
        # Applying the binary operator '*' (line 198)
        result_mul_373457 = python_operator(stypy.reporting.localization.Localization(__file__, 198, 17), '*', data_373455, mask_373456)
        
        # Obtaining the member 'sum' of a type (line 198)
        sum_373458 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 198, 17), result_mul_373457, 'sum')
        # Calling sum(args, kwargs) (line 198)
        sum_call_result_373462 = invoke(stypy.reporting.localization.Localization(__file__, 198, 17), sum_373458, *[], **kwargs_373461)
        
        # Assigning a type to the variable 'x' (line 198)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 198, 12), 'x', sum_call_result_373462)
        
        
        
        # Obtaining the type of the subscript
        int_373463 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 199, 23), 'int')
        # Getting the type of 'x' (line 199)
        x_373464 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 15), 'x')
        # Obtaining the member 'shape' of a type (line 199)
        shape_373465 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 199, 15), x_373464, 'shape')
        # Obtaining the member '__getitem__' of a type (line 199)
        getitem___373466 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 199, 15), shape_373465, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 199)
        subscript_call_result_373467 = invoke(stypy.reporting.localization.Localization(__file__, 199, 15), getitem___373466, int_373463)
        
        # Getting the type of 'num_cols' (line 199)
        num_cols_373468 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 29), 'num_cols')
        # Applying the binary operator '==' (line 199)
        result_eq_373469 = python_operator(stypy.reporting.localization.Localization(__file__, 199, 15), '==', subscript_call_result_373467, num_cols_373468)
        
        # Testing the type of an if condition (line 199)
        if_condition_373470 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 199, 12), result_eq_373469)
        # Assigning a type to the variable 'if_condition_373470' (line 199)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 199, 12), 'if_condition_373470', if_condition_373470)
        # SSA begins for if statement (line 199)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Name to a Name (line 200):
        
        # Assigning a Name to a Name (line 200):
        # Getting the type of 'x' (line 200)
        x_373471 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 22), 'x')
        # Assigning a type to the variable 'res' (line 200)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 200, 16), 'res', x_373471)
        # SSA branch for the else part of an if statement (line 199)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Call to a Name (line 202):
        
        # Assigning a Call to a Name (line 202):
        
        # Call to zeros(...): (line 202)
        # Processing the call arguments (line 202)
        # Getting the type of 'num_cols' (line 202)
        num_cols_373474 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 31), 'num_cols', False)
        # Processing the call keyword arguments (line 202)
        # Getting the type of 'x' (line 202)
        x_373475 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 47), 'x', False)
        # Obtaining the member 'dtype' of a type (line 202)
        dtype_373476 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 202, 47), x_373475, 'dtype')
        keyword_373477 = dtype_373476
        kwargs_373478 = {'dtype': keyword_373477}
        # Getting the type of 'np' (line 202)
        np_373472 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 22), 'np', False)
        # Obtaining the member 'zeros' of a type (line 202)
        zeros_373473 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 202, 22), np_373472, 'zeros')
        # Calling zeros(args, kwargs) (line 202)
        zeros_call_result_373479 = invoke(stypy.reporting.localization.Localization(__file__, 202, 22), zeros_373473, *[num_cols_373474], **kwargs_373478)
        
        # Assigning a type to the variable 'res' (line 202)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 202, 16), 'res', zeros_call_result_373479)
        
        # Assigning a Name to a Subscript (line 203):
        
        # Assigning a Name to a Subscript (line 203):
        # Getting the type of 'x' (line 203)
        x_373480 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 35), 'x')
        # Getting the type of 'res' (line 203)
        res_373481 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 16), 'res')
        
        # Obtaining the type of the subscript
        int_373482 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 203, 29), 'int')
        # Getting the type of 'x' (line 203)
        x_373483 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 21), 'x')
        # Obtaining the member 'shape' of a type (line 203)
        shape_373484 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 203, 21), x_373483, 'shape')
        # Obtaining the member '__getitem__' of a type (line 203)
        getitem___373485 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 203, 21), shape_373484, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 203)
        subscript_call_result_373486 = invoke(stypy.reporting.localization.Localization(__file__, 203, 21), getitem___373485, int_373482)
        
        slice_373487 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 203, 16), None, subscript_call_result_373486, None)
        # Storing an element on a container (line 203)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 203, 16), res_373481, (slice_373487, x_373480))
        # SSA join for if statement (line 199)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Name (line 204):
        
        # Assigning a Call to a Name (line 204):
        
        # Call to matrix(...): (line 204)
        # Processing the call arguments (line 204)
        # Getting the type of 'res' (line 204)
        res_373490 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 204, 28), 'res', False)
        # Processing the call keyword arguments (line 204)
        # Getting the type of 'res_dtype' (line 204)
        res_dtype_373491 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 204, 39), 'res_dtype', False)
        keyword_373492 = res_dtype_373491
        kwargs_373493 = {'dtype': keyword_373492}
        # Getting the type of 'np' (line 204)
        np_373488 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 204, 18), 'np', False)
        # Obtaining the member 'matrix' of a type (line 204)
        matrix_373489 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 204, 18), np_373488, 'matrix')
        # Calling matrix(args, kwargs) (line 204)
        matrix_call_result_373494 = invoke(stypy.reporting.localization.Localization(__file__, 204, 18), matrix_373489, *[res_373490], **kwargs_373493)
        
        # Assigning a type to the variable 'ret' (line 204)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 204, 12), 'ret', matrix_call_result_373494)
        # SSA branch for the else part of an if statement (line 196)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Call to a Name (line 207):
        
        # Assigning a Call to a Name (line 207):
        
        # Call to zeros(...): (line 207)
        # Processing the call arguments (line 207)
        # Getting the type of 'num_rows' (line 207)
        num_rows_373497 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 207, 32), 'num_rows', False)
        # Processing the call keyword arguments (line 207)
        # Getting the type of 'res_dtype' (line 207)
        res_dtype_373498 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 207, 48), 'res_dtype', False)
        keyword_373499 = res_dtype_373498
        kwargs_373500 = {'dtype': keyword_373499}
        # Getting the type of 'np' (line 207)
        np_373495 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 207, 23), 'np', False)
        # Obtaining the member 'zeros' of a type (line 207)
        zeros_373496 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 207, 23), np_373495, 'zeros')
        # Calling zeros(args, kwargs) (line 207)
        zeros_call_result_373501 = invoke(stypy.reporting.localization.Localization(__file__, 207, 23), zeros_373496, *[num_rows_373497], **kwargs_373500)
        
        # Assigning a type to the variable 'row_sums' (line 207)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 207, 12), 'row_sums', zeros_call_result_373501)
        
        # Assigning a Call to a Name (line 208):
        
        # Assigning a Call to a Name (line 208):
        
        # Call to ones(...): (line 208)
        # Processing the call arguments (line 208)
        # Getting the type of 'num_cols' (line 208)
        num_cols_373504 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 208, 26), 'num_cols', False)
        # Processing the call keyword arguments (line 208)
        # Getting the type of 'res_dtype' (line 208)
        res_dtype_373505 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 208, 42), 'res_dtype', False)
        keyword_373506 = res_dtype_373505
        kwargs_373507 = {'dtype': keyword_373506}
        # Getting the type of 'np' (line 208)
        np_373502 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 208, 18), 'np', False)
        # Obtaining the member 'ones' of a type (line 208)
        ones_373503 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 208, 18), np_373502, 'ones')
        # Calling ones(args, kwargs) (line 208)
        ones_call_result_373508 = invoke(stypy.reporting.localization.Localization(__file__, 208, 18), ones_373503, *[num_cols_373504], **kwargs_373507)
        
        # Assigning a type to the variable 'one' (line 208)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 208, 12), 'one', ones_call_result_373508)
        
        # Call to dia_matvec(...): (line 209)
        # Processing the call arguments (line 209)
        # Getting the type of 'num_rows' (line 209)
        num_rows_373510 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 23), 'num_rows', False)
        # Getting the type of 'num_cols' (line 209)
        num_cols_373511 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 33), 'num_cols', False)
        
        # Call to len(...): (line 209)
        # Processing the call arguments (line 209)
        # Getting the type of 'self' (line 209)
        self_373513 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 47), 'self', False)
        # Obtaining the member 'offsets' of a type (line 209)
        offsets_373514 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 209, 47), self_373513, 'offsets')
        # Processing the call keyword arguments (line 209)
        kwargs_373515 = {}
        # Getting the type of 'len' (line 209)
        len_373512 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 43), 'len', False)
        # Calling len(args, kwargs) (line 209)
        len_call_result_373516 = invoke(stypy.reporting.localization.Localization(__file__, 209, 43), len_373512, *[offsets_373514], **kwargs_373515)
        
        
        # Obtaining the type of the subscript
        int_373517 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 210, 39), 'int')
        # Getting the type of 'self' (line 210)
        self_373518 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 23), 'self', False)
        # Obtaining the member 'data' of a type (line 210)
        data_373519 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 210, 23), self_373518, 'data')
        # Obtaining the member 'shape' of a type (line 210)
        shape_373520 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 210, 23), data_373519, 'shape')
        # Obtaining the member '__getitem__' of a type (line 210)
        getitem___373521 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 210, 23), shape_373520, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 210)
        subscript_call_result_373522 = invoke(stypy.reporting.localization.Localization(__file__, 210, 23), getitem___373521, int_373517)
        
        # Getting the type of 'self' (line 210)
        self_373523 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 43), 'self', False)
        # Obtaining the member 'offsets' of a type (line 210)
        offsets_373524 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 210, 43), self_373523, 'offsets')
        # Getting the type of 'self' (line 210)
        self_373525 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 57), 'self', False)
        # Obtaining the member 'data' of a type (line 210)
        data_373526 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 210, 57), self_373525, 'data')
        # Getting the type of 'one' (line 210)
        one_373527 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 68), 'one', False)
        # Getting the type of 'row_sums' (line 210)
        row_sums_373528 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 73), 'row_sums', False)
        # Processing the call keyword arguments (line 209)
        kwargs_373529 = {}
        # Getting the type of 'dia_matvec' (line 209)
        dia_matvec_373509 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 12), 'dia_matvec', False)
        # Calling dia_matvec(args, kwargs) (line 209)
        dia_matvec_call_result_373530 = invoke(stypy.reporting.localization.Localization(__file__, 209, 12), dia_matvec_373509, *[num_rows_373510, num_cols_373511, len_call_result_373516, subscript_call_result_373522, offsets_373524, data_373526, one_373527, row_sums_373528], **kwargs_373529)
        
        
        # Assigning a Call to a Name (line 212):
        
        # Assigning a Call to a Name (line 212):
        
        # Call to matrix(...): (line 212)
        # Processing the call arguments (line 212)
        # Getting the type of 'row_sums' (line 212)
        row_sums_373533 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 33), 'row_sums', False)
        # Processing the call keyword arguments (line 212)
        kwargs_373534 = {}
        # Getting the type of 'np' (line 212)
        np_373531 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 23), 'np', False)
        # Obtaining the member 'matrix' of a type (line 212)
        matrix_373532 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 212, 23), np_373531, 'matrix')
        # Calling matrix(args, kwargs) (line 212)
        matrix_call_result_373535 = invoke(stypy.reporting.localization.Localization(__file__, 212, 23), matrix_373532, *[row_sums_373533], **kwargs_373534)
        
        # Assigning a type to the variable 'row_sums' (line 212)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 212, 12), 'row_sums', matrix_call_result_373535)
        
        # Type idiom detected: calculating its left and rigth part (line 214)
        # Getting the type of 'axis' (line 214)
        axis_373536 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 15), 'axis')
        # Getting the type of 'None' (line 214)
        None_373537 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 23), 'None')
        
        (may_be_373538, more_types_in_union_373539) = may_be_none(axis_373536, None_373537)

        if may_be_373538:

            if more_types_in_union_373539:
                # Runtime conditional SSA (line 214)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Call to sum(...): (line 215)
            # Processing the call keyword arguments (line 215)
            # Getting the type of 'dtype' (line 215)
            dtype_373542 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 215, 42), 'dtype', False)
            keyword_373543 = dtype_373542
            # Getting the type of 'out' (line 215)
            out_373544 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 215, 53), 'out', False)
            keyword_373545 = out_373544
            kwargs_373546 = {'dtype': keyword_373543, 'out': keyword_373545}
            # Getting the type of 'row_sums' (line 215)
            row_sums_373540 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 215, 23), 'row_sums', False)
            # Obtaining the member 'sum' of a type (line 215)
            sum_373541 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 215, 23), row_sums_373540, 'sum')
            # Calling sum(args, kwargs) (line 215)
            sum_call_result_373547 = invoke(stypy.reporting.localization.Localization(__file__, 215, 23), sum_373541, *[], **kwargs_373546)
            
            # Assigning a type to the variable 'stypy_return_type' (line 215)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 215, 16), 'stypy_return_type', sum_call_result_373547)

            if more_types_in_union_373539:
                # SSA join for if statement (line 214)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Type idiom detected: calculating its left and rigth part (line 217)
        # Getting the type of 'axis' (line 217)
        axis_373548 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 12), 'axis')
        # Getting the type of 'None' (line 217)
        None_373549 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 27), 'None')
        
        (may_be_373550, more_types_in_union_373551) = may_not_be_none(axis_373548, None_373549)

        if may_be_373550:

            if more_types_in_union_373551:
                # Runtime conditional SSA (line 217)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a Attribute to a Name (line 218):
            
            # Assigning a Attribute to a Name (line 218):
            # Getting the type of 'row_sums' (line 218)
            row_sums_373552 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 218, 27), 'row_sums')
            # Obtaining the member 'T' of a type (line 218)
            T_373553 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 218, 27), row_sums_373552, 'T')
            # Assigning a type to the variable 'row_sums' (line 218)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 218, 16), 'row_sums', T_373553)

            if more_types_in_union_373551:
                # SSA join for if statement (line 217)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Assigning a Call to a Name (line 220):
        
        # Assigning a Call to a Name (line 220):
        
        # Call to matrix(...): (line 220)
        # Processing the call arguments (line 220)
        
        # Call to sum(...): (line 220)
        # Processing the call keyword arguments (line 220)
        # Getting the type of 'axis' (line 220)
        axis_373558 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 220, 46), 'axis', False)
        keyword_373559 = axis_373558
        kwargs_373560 = {'axis': keyword_373559}
        # Getting the type of 'row_sums' (line 220)
        row_sums_373556 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 220, 28), 'row_sums', False)
        # Obtaining the member 'sum' of a type (line 220)
        sum_373557 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 220, 28), row_sums_373556, 'sum')
        # Calling sum(args, kwargs) (line 220)
        sum_call_result_373561 = invoke(stypy.reporting.localization.Localization(__file__, 220, 28), sum_373557, *[], **kwargs_373560)
        
        # Processing the call keyword arguments (line 220)
        kwargs_373562 = {}
        # Getting the type of 'np' (line 220)
        np_373554 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 220, 18), 'np', False)
        # Obtaining the member 'matrix' of a type (line 220)
        matrix_373555 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 220, 18), np_373554, 'matrix')
        # Calling matrix(args, kwargs) (line 220)
        matrix_call_result_373563 = invoke(stypy.reporting.localization.Localization(__file__, 220, 18), matrix_373555, *[sum_call_result_373561], **kwargs_373562)
        
        # Assigning a type to the variable 'ret' (line 220)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 220, 12), 'ret', matrix_call_result_373563)
        # SSA join for if statement (line 196)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Evaluating a boolean operation
        
        # Getting the type of 'out' (line 222)
        out_373564 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 222, 11), 'out')
        # Getting the type of 'None' (line 222)
        None_373565 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 222, 22), 'None')
        # Applying the binary operator 'isnot' (line 222)
        result_is_not_373566 = python_operator(stypy.reporting.localization.Localization(__file__, 222, 11), 'isnot', out_373564, None_373565)
        
        
        # Getting the type of 'out' (line 222)
        out_373567 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 222, 31), 'out')
        # Obtaining the member 'shape' of a type (line 222)
        shape_373568 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 222, 31), out_373567, 'shape')
        # Getting the type of 'ret' (line 222)
        ret_373569 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 222, 44), 'ret')
        # Obtaining the member 'shape' of a type (line 222)
        shape_373570 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 222, 44), ret_373569, 'shape')
        # Applying the binary operator '!=' (line 222)
        result_ne_373571 = python_operator(stypy.reporting.localization.Localization(__file__, 222, 31), '!=', shape_373568, shape_373570)
        
        # Applying the binary operator 'and' (line 222)
        result_and_keyword_373572 = python_operator(stypy.reporting.localization.Localization(__file__, 222, 11), 'and', result_is_not_373566, result_ne_373571)
        
        # Testing the type of an if condition (line 222)
        if_condition_373573 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 222, 8), result_and_keyword_373572)
        # Assigning a type to the variable 'if_condition_373573' (line 222)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 222, 8), 'if_condition_373573', if_condition_373573)
        # SSA begins for if statement (line 222)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to ValueError(...): (line 223)
        # Processing the call arguments (line 223)
        str_373575 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 223, 29), 'str', 'dimensions do not match')
        # Processing the call keyword arguments (line 223)
        kwargs_373576 = {}
        # Getting the type of 'ValueError' (line 223)
        ValueError_373574 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 18), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 223)
        ValueError_call_result_373577 = invoke(stypy.reporting.localization.Localization(__file__, 223, 18), ValueError_373574, *[str_373575], **kwargs_373576)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 223, 12), ValueError_call_result_373577, 'raise parameter', BaseException)
        # SSA join for if statement (line 222)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to sum(...): (line 225)
        # Processing the call keyword arguments (line 225)
        
        # Obtaining an instance of the builtin type 'tuple' (line 225)
        tuple_373580 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 225, 28), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 225)
        
        keyword_373581 = tuple_373580
        # Getting the type of 'dtype' (line 225)
        dtype_373582 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 225, 38), 'dtype', False)
        keyword_373583 = dtype_373582
        # Getting the type of 'out' (line 225)
        out_373584 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 225, 49), 'out', False)
        keyword_373585 = out_373584
        kwargs_373586 = {'dtype': keyword_373583, 'out': keyword_373585, 'axis': keyword_373581}
        # Getting the type of 'ret' (line 225)
        ret_373578 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 225, 15), 'ret', False)
        # Obtaining the member 'sum' of a type (line 225)
        sum_373579 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 225, 15), ret_373578, 'sum')
        # Calling sum(args, kwargs) (line 225)
        sum_call_result_373587 = invoke(stypy.reporting.localization.Localization(__file__, 225, 15), sum_373579, *[], **kwargs_373586)
        
        # Assigning a type to the variable 'stypy_return_type' (line 225)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 225, 8), 'stypy_return_type', sum_call_result_373587)
        
        # ################# End of 'sum(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'sum' in the type store
        # Getting the type of 'stypy_return_type' (line 186)
        stypy_return_type_373588 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_373588)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'sum'
        return stypy_return_type_373588

    
    # Assigning a Attribute to a Attribute (line 227):

    @norecursion
    def _mul_vector(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_mul_vector'
        module_type_store = module_type_store.open_function_context('_mul_vector', 229, 4, False)
        # Assigning a type to the variable 'self' (line 230)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 230, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        dia_matrix._mul_vector.__dict__.__setitem__('stypy_localization', localization)
        dia_matrix._mul_vector.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        dia_matrix._mul_vector.__dict__.__setitem__('stypy_type_store', module_type_store)
        dia_matrix._mul_vector.__dict__.__setitem__('stypy_function_name', 'dia_matrix._mul_vector')
        dia_matrix._mul_vector.__dict__.__setitem__('stypy_param_names_list', ['other'])
        dia_matrix._mul_vector.__dict__.__setitem__('stypy_varargs_param_name', None)
        dia_matrix._mul_vector.__dict__.__setitem__('stypy_kwargs_param_name', None)
        dia_matrix._mul_vector.__dict__.__setitem__('stypy_call_defaults', defaults)
        dia_matrix._mul_vector.__dict__.__setitem__('stypy_call_varargs', varargs)
        dia_matrix._mul_vector.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        dia_matrix._mul_vector.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'dia_matrix._mul_vector', ['other'], None, None, defaults, varargs, kwargs)

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

        
        # Assigning a Name to a Name (line 230):
        
        # Assigning a Name to a Name (line 230):
        # Getting the type of 'other' (line 230)
        other_373589 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 12), 'other')
        # Assigning a type to the variable 'x' (line 230)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 230, 8), 'x', other_373589)
        
        # Assigning a Call to a Name (line 232):
        
        # Assigning a Call to a Name (line 232):
        
        # Call to zeros(...): (line 232)
        # Processing the call arguments (line 232)
        
        # Obtaining the type of the subscript
        int_373592 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 232, 32), 'int')
        # Getting the type of 'self' (line 232)
        self_373593 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 21), 'self', False)
        # Obtaining the member 'shape' of a type (line 232)
        shape_373594 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 232, 21), self_373593, 'shape')
        # Obtaining the member '__getitem__' of a type (line 232)
        getitem___373595 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 232, 21), shape_373594, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 232)
        subscript_call_result_373596 = invoke(stypy.reporting.localization.Localization(__file__, 232, 21), getitem___373595, int_373592)
        
        # Processing the call keyword arguments (line 232)
        
        # Call to upcast_char(...): (line 232)
        # Processing the call arguments (line 232)
        # Getting the type of 'self' (line 232)
        self_373598 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 54), 'self', False)
        # Obtaining the member 'dtype' of a type (line 232)
        dtype_373599 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 232, 54), self_373598, 'dtype')
        # Obtaining the member 'char' of a type (line 232)
        char_373600 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 232, 54), dtype_373599, 'char')
        # Getting the type of 'x' (line 233)
        x_373601 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 233, 55), 'x', False)
        # Obtaining the member 'dtype' of a type (line 233)
        dtype_373602 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 233, 55), x_373601, 'dtype')
        # Obtaining the member 'char' of a type (line 233)
        char_373603 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 233, 55), dtype_373602, 'char')
        # Processing the call keyword arguments (line 232)
        kwargs_373604 = {}
        # Getting the type of 'upcast_char' (line 232)
        upcast_char_373597 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 42), 'upcast_char', False)
        # Calling upcast_char(args, kwargs) (line 232)
        upcast_char_call_result_373605 = invoke(stypy.reporting.localization.Localization(__file__, 232, 42), upcast_char_373597, *[char_373600, char_373603], **kwargs_373604)
        
        keyword_373606 = upcast_char_call_result_373605
        kwargs_373607 = {'dtype': keyword_373606}
        # Getting the type of 'np' (line 232)
        np_373590 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 12), 'np', False)
        # Obtaining the member 'zeros' of a type (line 232)
        zeros_373591 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 232, 12), np_373590, 'zeros')
        # Calling zeros(args, kwargs) (line 232)
        zeros_call_result_373608 = invoke(stypy.reporting.localization.Localization(__file__, 232, 12), zeros_373591, *[subscript_call_result_373596], **kwargs_373607)
        
        # Assigning a type to the variable 'y' (line 232)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 232, 8), 'y', zeros_call_result_373608)
        
        # Assigning a Subscript to a Name (line 235):
        
        # Assigning a Subscript to a Name (line 235):
        
        # Obtaining the type of the subscript
        int_373609 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 235, 28), 'int')
        # Getting the type of 'self' (line 235)
        self_373610 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 12), 'self')
        # Obtaining the member 'data' of a type (line 235)
        data_373611 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 235, 12), self_373610, 'data')
        # Obtaining the member 'shape' of a type (line 235)
        shape_373612 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 235, 12), data_373611, 'shape')
        # Obtaining the member '__getitem__' of a type (line 235)
        getitem___373613 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 235, 12), shape_373612, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 235)
        subscript_call_result_373614 = invoke(stypy.reporting.localization.Localization(__file__, 235, 12), getitem___373613, int_373609)
        
        # Assigning a type to the variable 'L' (line 235)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 235, 8), 'L', subscript_call_result_373614)
        
        # Assigning a Attribute to a Tuple (line 237):
        
        # Assigning a Subscript to a Name (line 237):
        
        # Obtaining the type of the subscript
        int_373615 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 237, 8), 'int')
        # Getting the type of 'self' (line 237)
        self_373616 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 237, 14), 'self')
        # Obtaining the member 'shape' of a type (line 237)
        shape_373617 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 237, 14), self_373616, 'shape')
        # Obtaining the member '__getitem__' of a type (line 237)
        getitem___373618 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 237, 8), shape_373617, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 237)
        subscript_call_result_373619 = invoke(stypy.reporting.localization.Localization(__file__, 237, 8), getitem___373618, int_373615)
        
        # Assigning a type to the variable 'tuple_var_assignment_372954' (line 237)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 237, 8), 'tuple_var_assignment_372954', subscript_call_result_373619)
        
        # Assigning a Subscript to a Name (line 237):
        
        # Obtaining the type of the subscript
        int_373620 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 237, 8), 'int')
        # Getting the type of 'self' (line 237)
        self_373621 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 237, 14), 'self')
        # Obtaining the member 'shape' of a type (line 237)
        shape_373622 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 237, 14), self_373621, 'shape')
        # Obtaining the member '__getitem__' of a type (line 237)
        getitem___373623 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 237, 8), shape_373622, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 237)
        subscript_call_result_373624 = invoke(stypy.reporting.localization.Localization(__file__, 237, 8), getitem___373623, int_373620)
        
        # Assigning a type to the variable 'tuple_var_assignment_372955' (line 237)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 237, 8), 'tuple_var_assignment_372955', subscript_call_result_373624)
        
        # Assigning a Name to a Name (line 237):
        # Getting the type of 'tuple_var_assignment_372954' (line 237)
        tuple_var_assignment_372954_373625 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 237, 8), 'tuple_var_assignment_372954')
        # Assigning a type to the variable 'M' (line 237)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 237, 8), 'M', tuple_var_assignment_372954_373625)
        
        # Assigning a Name to a Name (line 237):
        # Getting the type of 'tuple_var_assignment_372955' (line 237)
        tuple_var_assignment_372955_373626 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 237, 8), 'tuple_var_assignment_372955')
        # Assigning a type to the variable 'N' (line 237)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 237, 10), 'N', tuple_var_assignment_372955_373626)
        
        # Call to dia_matvec(...): (line 239)
        # Processing the call arguments (line 239)
        # Getting the type of 'M' (line 239)
        M_373628 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 239, 19), 'M', False)
        # Getting the type of 'N' (line 239)
        N_373629 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 239, 21), 'N', False)
        
        # Call to len(...): (line 239)
        # Processing the call arguments (line 239)
        # Getting the type of 'self' (line 239)
        self_373631 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 239, 28), 'self', False)
        # Obtaining the member 'offsets' of a type (line 239)
        offsets_373632 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 239, 28), self_373631, 'offsets')
        # Processing the call keyword arguments (line 239)
        kwargs_373633 = {}
        # Getting the type of 'len' (line 239)
        len_373630 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 239, 24), 'len', False)
        # Calling len(args, kwargs) (line 239)
        len_call_result_373634 = invoke(stypy.reporting.localization.Localization(__file__, 239, 24), len_373630, *[offsets_373632], **kwargs_373633)
        
        # Getting the type of 'L' (line 239)
        L_373635 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 239, 43), 'L', False)
        # Getting the type of 'self' (line 239)
        self_373636 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 239, 46), 'self', False)
        # Obtaining the member 'offsets' of a type (line 239)
        offsets_373637 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 239, 46), self_373636, 'offsets')
        # Getting the type of 'self' (line 239)
        self_373638 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 239, 60), 'self', False)
        # Obtaining the member 'data' of a type (line 239)
        data_373639 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 239, 60), self_373638, 'data')
        
        # Call to ravel(...): (line 239)
        # Processing the call keyword arguments (line 239)
        kwargs_373642 = {}
        # Getting the type of 'x' (line 239)
        x_373640 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 239, 71), 'x', False)
        # Obtaining the member 'ravel' of a type (line 239)
        ravel_373641 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 239, 71), x_373640, 'ravel')
        # Calling ravel(args, kwargs) (line 239)
        ravel_call_result_373643 = invoke(stypy.reporting.localization.Localization(__file__, 239, 71), ravel_373641, *[], **kwargs_373642)
        
        
        # Call to ravel(...): (line 239)
        # Processing the call keyword arguments (line 239)
        kwargs_373646 = {}
        # Getting the type of 'y' (line 239)
        y_373644 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 239, 82), 'y', False)
        # Obtaining the member 'ravel' of a type (line 239)
        ravel_373645 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 239, 82), y_373644, 'ravel')
        # Calling ravel(args, kwargs) (line 239)
        ravel_call_result_373647 = invoke(stypy.reporting.localization.Localization(__file__, 239, 82), ravel_373645, *[], **kwargs_373646)
        
        # Processing the call keyword arguments (line 239)
        kwargs_373648 = {}
        # Getting the type of 'dia_matvec' (line 239)
        dia_matvec_373627 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 239, 8), 'dia_matvec', False)
        # Calling dia_matvec(args, kwargs) (line 239)
        dia_matvec_call_result_373649 = invoke(stypy.reporting.localization.Localization(__file__, 239, 8), dia_matvec_373627, *[M_373628, N_373629, len_call_result_373634, L_373635, offsets_373637, data_373639, ravel_call_result_373643, ravel_call_result_373647], **kwargs_373648)
        
        # Getting the type of 'y' (line 241)
        y_373650 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 241, 15), 'y')
        # Assigning a type to the variable 'stypy_return_type' (line 241)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 241, 8), 'stypy_return_type', y_373650)
        
        # ################# End of '_mul_vector(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_mul_vector' in the type store
        # Getting the type of 'stypy_return_type' (line 229)
        stypy_return_type_373651 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 229, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_373651)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_mul_vector'
        return stypy_return_type_373651


    @norecursion
    def _mul_multimatrix(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_mul_multimatrix'
        module_type_store = module_type_store.open_function_context('_mul_multimatrix', 243, 4, False)
        # Assigning a type to the variable 'self' (line 244)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 244, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        dia_matrix._mul_multimatrix.__dict__.__setitem__('stypy_localization', localization)
        dia_matrix._mul_multimatrix.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        dia_matrix._mul_multimatrix.__dict__.__setitem__('stypy_type_store', module_type_store)
        dia_matrix._mul_multimatrix.__dict__.__setitem__('stypy_function_name', 'dia_matrix._mul_multimatrix')
        dia_matrix._mul_multimatrix.__dict__.__setitem__('stypy_param_names_list', ['other'])
        dia_matrix._mul_multimatrix.__dict__.__setitem__('stypy_varargs_param_name', None)
        dia_matrix._mul_multimatrix.__dict__.__setitem__('stypy_kwargs_param_name', None)
        dia_matrix._mul_multimatrix.__dict__.__setitem__('stypy_call_defaults', defaults)
        dia_matrix._mul_multimatrix.__dict__.__setitem__('stypy_call_varargs', varargs)
        dia_matrix._mul_multimatrix.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        dia_matrix._mul_multimatrix.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'dia_matrix._mul_multimatrix', ['other'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_mul_multimatrix', localization, ['other'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_mul_multimatrix(...)' code ##################

        
        # Call to hstack(...): (line 244)
        # Processing the call arguments (line 244)
        # Calculating list comprehension
        # Calculating comprehension expression
        # Getting the type of 'other' (line 244)
        other_373664 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 244, 73), 'other', False)
        # Obtaining the member 'T' of a type (line 244)
        T_373665 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 244, 73), other_373664, 'T')
        comprehension_373666 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 244, 26), T_373665)
        # Assigning a type to the variable 'col' (line 244)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 244, 26), 'col', comprehension_373666)
        
        # Call to reshape(...): (line 244)
        # Processing the call arguments (line 244)
        int_373660 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 244, 56), 'int')
        int_373661 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 244, 59), 'int')
        # Processing the call keyword arguments (line 244)
        kwargs_373662 = {}
        
        # Call to _mul_vector(...): (line 244)
        # Processing the call arguments (line 244)
        # Getting the type of 'col' (line 244)
        col_373656 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 244, 43), 'col', False)
        # Processing the call keyword arguments (line 244)
        kwargs_373657 = {}
        # Getting the type of 'self' (line 244)
        self_373654 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 244, 26), 'self', False)
        # Obtaining the member '_mul_vector' of a type (line 244)
        _mul_vector_373655 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 244, 26), self_373654, '_mul_vector')
        # Calling _mul_vector(args, kwargs) (line 244)
        _mul_vector_call_result_373658 = invoke(stypy.reporting.localization.Localization(__file__, 244, 26), _mul_vector_373655, *[col_373656], **kwargs_373657)
        
        # Obtaining the member 'reshape' of a type (line 244)
        reshape_373659 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 244, 26), _mul_vector_call_result_373658, 'reshape')
        # Calling reshape(args, kwargs) (line 244)
        reshape_call_result_373663 = invoke(stypy.reporting.localization.Localization(__file__, 244, 26), reshape_373659, *[int_373660, int_373661], **kwargs_373662)
        
        list_373667 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 244, 26), 'list')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 244, 26), list_373667, reshape_call_result_373663)
        # Processing the call keyword arguments (line 244)
        kwargs_373668 = {}
        # Getting the type of 'np' (line 244)
        np_373652 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 244, 15), 'np', False)
        # Obtaining the member 'hstack' of a type (line 244)
        hstack_373653 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 244, 15), np_373652, 'hstack')
        # Calling hstack(args, kwargs) (line 244)
        hstack_call_result_373669 = invoke(stypy.reporting.localization.Localization(__file__, 244, 15), hstack_373653, *[list_373667], **kwargs_373668)
        
        # Assigning a type to the variable 'stypy_return_type' (line 244)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 244, 8), 'stypy_return_type', hstack_call_result_373669)
        
        # ################# End of '_mul_multimatrix(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_mul_multimatrix' in the type store
        # Getting the type of 'stypy_return_type' (line 243)
        stypy_return_type_373670 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 243, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_373670)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_mul_multimatrix'
        return stypy_return_type_373670


    @norecursion
    def _setdiag(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        int_373671 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 246, 33), 'int')
        defaults = [int_373671]
        # Create a new context for function '_setdiag'
        module_type_store = module_type_store.open_function_context('_setdiag', 246, 4, False)
        # Assigning a type to the variable 'self' (line 247)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 247, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        dia_matrix._setdiag.__dict__.__setitem__('stypy_localization', localization)
        dia_matrix._setdiag.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        dia_matrix._setdiag.__dict__.__setitem__('stypy_type_store', module_type_store)
        dia_matrix._setdiag.__dict__.__setitem__('stypy_function_name', 'dia_matrix._setdiag')
        dia_matrix._setdiag.__dict__.__setitem__('stypy_param_names_list', ['values', 'k'])
        dia_matrix._setdiag.__dict__.__setitem__('stypy_varargs_param_name', None)
        dia_matrix._setdiag.__dict__.__setitem__('stypy_kwargs_param_name', None)
        dia_matrix._setdiag.__dict__.__setitem__('stypy_call_defaults', defaults)
        dia_matrix._setdiag.__dict__.__setitem__('stypy_call_varargs', varargs)
        dia_matrix._setdiag.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        dia_matrix._setdiag.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'dia_matrix._setdiag', ['values', 'k'], None, None, defaults, varargs, kwargs)

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

        
        # Assigning a Attribute to a Tuple (line 247):
        
        # Assigning a Subscript to a Name (line 247):
        
        # Obtaining the type of the subscript
        int_373672 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 247, 8), 'int')
        # Getting the type of 'self' (line 247)
        self_373673 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 247, 15), 'self')
        # Obtaining the member 'shape' of a type (line 247)
        shape_373674 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 247, 15), self_373673, 'shape')
        # Obtaining the member '__getitem__' of a type (line 247)
        getitem___373675 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 247, 8), shape_373674, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 247)
        subscript_call_result_373676 = invoke(stypy.reporting.localization.Localization(__file__, 247, 8), getitem___373675, int_373672)
        
        # Assigning a type to the variable 'tuple_var_assignment_372956' (line 247)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 247, 8), 'tuple_var_assignment_372956', subscript_call_result_373676)
        
        # Assigning a Subscript to a Name (line 247):
        
        # Obtaining the type of the subscript
        int_373677 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 247, 8), 'int')
        # Getting the type of 'self' (line 247)
        self_373678 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 247, 15), 'self')
        # Obtaining the member 'shape' of a type (line 247)
        shape_373679 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 247, 15), self_373678, 'shape')
        # Obtaining the member '__getitem__' of a type (line 247)
        getitem___373680 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 247, 8), shape_373679, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 247)
        subscript_call_result_373681 = invoke(stypy.reporting.localization.Localization(__file__, 247, 8), getitem___373680, int_373677)
        
        # Assigning a type to the variable 'tuple_var_assignment_372957' (line 247)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 247, 8), 'tuple_var_assignment_372957', subscript_call_result_373681)
        
        # Assigning a Name to a Name (line 247):
        # Getting the type of 'tuple_var_assignment_372956' (line 247)
        tuple_var_assignment_372956_373682 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 247, 8), 'tuple_var_assignment_372956')
        # Assigning a type to the variable 'M' (line 247)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 247, 8), 'M', tuple_var_assignment_372956_373682)
        
        # Assigning a Name to a Name (line 247):
        # Getting the type of 'tuple_var_assignment_372957' (line 247)
        tuple_var_assignment_372957_373683 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 247, 8), 'tuple_var_assignment_372957')
        # Assigning a type to the variable 'N' (line 247)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 247, 11), 'N', tuple_var_assignment_372957_373683)
        
        
        # Getting the type of 'values' (line 249)
        values_373684 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 249, 11), 'values')
        # Obtaining the member 'ndim' of a type (line 249)
        ndim_373685 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 249, 11), values_373684, 'ndim')
        int_373686 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 249, 26), 'int')
        # Applying the binary operator '==' (line 249)
        result_eq_373687 = python_operator(stypy.reporting.localization.Localization(__file__, 249, 11), '==', ndim_373685, int_373686)
        
        # Testing the type of an if condition (line 249)
        if_condition_373688 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 249, 8), result_eq_373687)
        # Assigning a type to the variable 'if_condition_373688' (line 249)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 249, 8), 'if_condition_373688', if_condition_373688)
        # SSA begins for if statement (line 249)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Attribute to a Name (line 251):
        
        # Assigning a Attribute to a Name (line 251):
        # Getting the type of 'np' (line 251)
        np_373689 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 251, 23), 'np')
        # Obtaining the member 'inf' of a type (line 251)
        inf_373690 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 251, 23), np_373689, 'inf')
        # Assigning a type to the variable 'values_n' (line 251)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 251, 12), 'values_n', inf_373690)
        # SSA branch for the else part of an if statement (line 249)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Call to a Name (line 253):
        
        # Assigning a Call to a Name (line 253):
        
        # Call to len(...): (line 253)
        # Processing the call arguments (line 253)
        # Getting the type of 'values' (line 253)
        values_373692 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 253, 27), 'values', False)
        # Processing the call keyword arguments (line 253)
        kwargs_373693 = {}
        # Getting the type of 'len' (line 253)
        len_373691 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 253, 23), 'len', False)
        # Calling len(args, kwargs) (line 253)
        len_call_result_373694 = invoke(stypy.reporting.localization.Localization(__file__, 253, 23), len_373691, *[values_373692], **kwargs_373693)
        
        # Assigning a type to the variable 'values_n' (line 253)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 253, 12), 'values_n', len_call_result_373694)
        # SSA join for if statement (line 249)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Getting the type of 'k' (line 255)
        k_373695 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 255, 11), 'k')
        int_373696 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 255, 15), 'int')
        # Applying the binary operator '<' (line 255)
        result_lt_373697 = python_operator(stypy.reporting.localization.Localization(__file__, 255, 11), '<', k_373695, int_373696)
        
        # Testing the type of an if condition (line 255)
        if_condition_373698 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 255, 8), result_lt_373697)
        # Assigning a type to the variable 'if_condition_373698' (line 255)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 255, 8), 'if_condition_373698', if_condition_373698)
        # SSA begins for if statement (line 255)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 256):
        
        # Assigning a Call to a Name (line 256):
        
        # Call to min(...): (line 256)
        # Processing the call arguments (line 256)
        # Getting the type of 'M' (line 256)
        M_373700 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 256, 20), 'M', False)
        # Getting the type of 'k' (line 256)
        k_373701 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 256, 24), 'k', False)
        # Applying the binary operator '+' (line 256)
        result_add_373702 = python_operator(stypy.reporting.localization.Localization(__file__, 256, 20), '+', M_373700, k_373701)
        
        # Getting the type of 'N' (line 256)
        N_373703 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 256, 27), 'N', False)
        # Getting the type of 'values_n' (line 256)
        values_n_373704 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 256, 30), 'values_n', False)
        # Processing the call keyword arguments (line 256)
        kwargs_373705 = {}
        # Getting the type of 'min' (line 256)
        min_373699 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 256, 16), 'min', False)
        # Calling min(args, kwargs) (line 256)
        min_call_result_373706 = invoke(stypy.reporting.localization.Localization(__file__, 256, 16), min_373699, *[result_add_373702, N_373703, values_n_373704], **kwargs_373705)
        
        # Assigning a type to the variable 'n' (line 256)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 256, 12), 'n', min_call_result_373706)
        
        # Assigning a Num to a Name (line 257):
        
        # Assigning a Num to a Name (line 257):
        int_373707 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 257, 24), 'int')
        # Assigning a type to the variable 'min_index' (line 257)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 257, 12), 'min_index', int_373707)
        
        # Assigning a Name to a Name (line 258):
        
        # Assigning a Name to a Name (line 258):
        # Getting the type of 'n' (line 258)
        n_373708 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 258, 24), 'n')
        # Assigning a type to the variable 'max_index' (line 258)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 258, 12), 'max_index', n_373708)
        # SSA branch for the else part of an if statement (line 255)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Call to a Name (line 260):
        
        # Assigning a Call to a Name (line 260):
        
        # Call to min(...): (line 260)
        # Processing the call arguments (line 260)
        # Getting the type of 'M' (line 260)
        M_373710 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 260, 20), 'M', False)
        # Getting the type of 'N' (line 260)
        N_373711 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 260, 23), 'N', False)
        # Getting the type of 'k' (line 260)
        k_373712 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 260, 27), 'k', False)
        # Applying the binary operator '-' (line 260)
        result_sub_373713 = python_operator(stypy.reporting.localization.Localization(__file__, 260, 23), '-', N_373711, k_373712)
        
        # Getting the type of 'values_n' (line 260)
        values_n_373714 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 260, 30), 'values_n', False)
        # Processing the call keyword arguments (line 260)
        kwargs_373715 = {}
        # Getting the type of 'min' (line 260)
        min_373709 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 260, 16), 'min', False)
        # Calling min(args, kwargs) (line 260)
        min_call_result_373716 = invoke(stypy.reporting.localization.Localization(__file__, 260, 16), min_373709, *[M_373710, result_sub_373713, values_n_373714], **kwargs_373715)
        
        # Assigning a type to the variable 'n' (line 260)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 260, 12), 'n', min_call_result_373716)
        
        # Assigning a Name to a Name (line 261):
        
        # Assigning a Name to a Name (line 261):
        # Getting the type of 'k' (line 261)
        k_373717 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 261, 24), 'k')
        # Assigning a type to the variable 'min_index' (line 261)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 261, 12), 'min_index', k_373717)
        
        # Assigning a BinOp to a Name (line 262):
        
        # Assigning a BinOp to a Name (line 262):
        # Getting the type of 'k' (line 262)
        k_373718 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 262, 24), 'k')
        # Getting the type of 'n' (line 262)
        n_373719 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 262, 28), 'n')
        # Applying the binary operator '+' (line 262)
        result_add_373720 = python_operator(stypy.reporting.localization.Localization(__file__, 262, 24), '+', k_373718, n_373719)
        
        # Assigning a type to the variable 'max_index' (line 262)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 262, 12), 'max_index', result_add_373720)
        # SSA join for if statement (line 255)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Getting the type of 'values' (line 264)
        values_373721 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 264, 11), 'values')
        # Obtaining the member 'ndim' of a type (line 264)
        ndim_373722 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 264, 11), values_373721, 'ndim')
        int_373723 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 264, 26), 'int')
        # Applying the binary operator '!=' (line 264)
        result_ne_373724 = python_operator(stypy.reporting.localization.Localization(__file__, 264, 11), '!=', ndim_373722, int_373723)
        
        # Testing the type of an if condition (line 264)
        if_condition_373725 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 264, 8), result_ne_373724)
        # Assigning a type to the variable 'if_condition_373725' (line 264)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 264, 8), 'if_condition_373725', if_condition_373725)
        # SSA begins for if statement (line 264)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Subscript to a Name (line 266):
        
        # Assigning a Subscript to a Name (line 266):
        
        # Obtaining the type of the subscript
        # Getting the type of 'n' (line 266)
        n_373726 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 266, 29), 'n')
        slice_373727 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 266, 21), None, n_373726, None)
        # Getting the type of 'values' (line 266)
        values_373728 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 266, 21), 'values')
        # Obtaining the member '__getitem__' of a type (line 266)
        getitem___373729 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 266, 21), values_373728, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 266)
        subscript_call_result_373730 = invoke(stypy.reporting.localization.Localization(__file__, 266, 21), getitem___373729, slice_373727)
        
        # Assigning a type to the variable 'values' (line 266)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 266, 12), 'values', subscript_call_result_373730)
        # SSA join for if statement (line 264)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Getting the type of 'k' (line 268)
        k_373731 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 268, 11), 'k')
        # Getting the type of 'self' (line 268)
        self_373732 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 268, 16), 'self')
        # Obtaining the member 'offsets' of a type (line 268)
        offsets_373733 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 268, 16), self_373732, 'offsets')
        # Applying the binary operator 'in' (line 268)
        result_contains_373734 = python_operator(stypy.reporting.localization.Localization(__file__, 268, 11), 'in', k_373731, offsets_373733)
        
        # Testing the type of an if condition (line 268)
        if_condition_373735 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 268, 8), result_contains_373734)
        # Assigning a type to the variable 'if_condition_373735' (line 268)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 268, 8), 'if_condition_373735', if_condition_373735)
        # SSA begins for if statement (line 268)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Name to a Subscript (line 269):
        
        # Assigning a Name to a Subscript (line 269):
        # Getting the type of 'values' (line 269)
        values_373736 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 269, 64), 'values')
        # Getting the type of 'self' (line 269)
        self_373737 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 269, 12), 'self')
        # Obtaining the member 'data' of a type (line 269)
        data_373738 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 269, 12), self_373737, 'data')
        
        # Getting the type of 'self' (line 269)
        self_373739 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 269, 22), 'self')
        # Obtaining the member 'offsets' of a type (line 269)
        offsets_373740 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 269, 22), self_373739, 'offsets')
        # Getting the type of 'k' (line 269)
        k_373741 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 269, 38), 'k')
        # Applying the binary operator '==' (line 269)
        result_eq_373742 = python_operator(stypy.reporting.localization.Localization(__file__, 269, 22), '==', offsets_373740, k_373741)
        
        # Getting the type of 'min_index' (line 269)
        min_index_373743 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 269, 41), 'min_index')
        # Getting the type of 'max_index' (line 269)
        max_index_373744 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 269, 51), 'max_index')
        slice_373745 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 269, 12), min_index_373743, max_index_373744, None)
        # Storing an element on a container (line 269)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 269, 12), data_373738, ((result_eq_373742, slice_373745), values_373736))
        # SSA branch for the else part of an if statement (line 268)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Call to a Attribute (line 271):
        
        # Assigning a Call to a Attribute (line 271):
        
        # Call to append(...): (line 271)
        # Processing the call arguments (line 271)
        # Getting the type of 'self' (line 271)
        self_373748 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 271, 37), 'self', False)
        # Obtaining the member 'offsets' of a type (line 271)
        offsets_373749 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 271, 37), self_373748, 'offsets')
        
        # Call to type(...): (line 271)
        # Processing the call arguments (line 271)
        # Getting the type of 'k' (line 271)
        k_373754 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 271, 75), 'k', False)
        # Processing the call keyword arguments (line 271)
        kwargs_373755 = {}
        # Getting the type of 'self' (line 271)
        self_373750 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 271, 51), 'self', False)
        # Obtaining the member 'offsets' of a type (line 271)
        offsets_373751 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 271, 51), self_373750, 'offsets')
        # Obtaining the member 'dtype' of a type (line 271)
        dtype_373752 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 271, 51), offsets_373751, 'dtype')
        # Obtaining the member 'type' of a type (line 271)
        type_373753 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 271, 51), dtype_373752, 'type')
        # Calling type(args, kwargs) (line 271)
        type_call_result_373756 = invoke(stypy.reporting.localization.Localization(__file__, 271, 51), type_373753, *[k_373754], **kwargs_373755)
        
        # Processing the call keyword arguments (line 271)
        kwargs_373757 = {}
        # Getting the type of 'np' (line 271)
        np_373746 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 271, 27), 'np', False)
        # Obtaining the member 'append' of a type (line 271)
        append_373747 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 271, 27), np_373746, 'append')
        # Calling append(args, kwargs) (line 271)
        append_call_result_373758 = invoke(stypy.reporting.localization.Localization(__file__, 271, 27), append_373747, *[offsets_373749, type_call_result_373756], **kwargs_373757)
        
        # Getting the type of 'self' (line 271)
        self_373759 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 271, 12), 'self')
        # Setting the type of the member 'offsets' of a type (line 271)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 271, 12), self_373759, 'offsets', append_call_result_373758)
        
        # Assigning a Call to a Name (line 272):
        
        # Assigning a Call to a Name (line 272):
        
        # Call to max(...): (line 272)
        # Processing the call arguments (line 272)
        # Getting the type of 'max_index' (line 272)
        max_index_373761 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 272, 20), 'max_index', False)
        
        # Obtaining the type of the subscript
        int_373762 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 272, 47), 'int')
        # Getting the type of 'self' (line 272)
        self_373763 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 272, 31), 'self', False)
        # Obtaining the member 'data' of a type (line 272)
        data_373764 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 272, 31), self_373763, 'data')
        # Obtaining the member 'shape' of a type (line 272)
        shape_373765 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 272, 31), data_373764, 'shape')
        # Obtaining the member '__getitem__' of a type (line 272)
        getitem___373766 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 272, 31), shape_373765, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 272)
        subscript_call_result_373767 = invoke(stypy.reporting.localization.Localization(__file__, 272, 31), getitem___373766, int_373762)
        
        # Processing the call keyword arguments (line 272)
        kwargs_373768 = {}
        # Getting the type of 'max' (line 272)
        max_373760 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 272, 16), 'max', False)
        # Calling max(args, kwargs) (line 272)
        max_call_result_373769 = invoke(stypy.reporting.localization.Localization(__file__, 272, 16), max_373760, *[max_index_373761, subscript_call_result_373767], **kwargs_373768)
        
        # Assigning a type to the variable 'm' (line 272)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 272, 12), 'm', max_call_result_373769)
        
        # Assigning a Call to a Name (line 273):
        
        # Assigning a Call to a Name (line 273):
        
        # Call to zeros(...): (line 273)
        # Processing the call arguments (line 273)
        
        # Obtaining an instance of the builtin type 'tuple' (line 273)
        tuple_373772 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 273, 29), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 273)
        # Adding element type (line 273)
        
        # Obtaining the type of the subscript
        int_373773 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 273, 45), 'int')
        # Getting the type of 'self' (line 273)
        self_373774 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 273, 29), 'self', False)
        # Obtaining the member 'data' of a type (line 273)
        data_373775 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 273, 29), self_373774, 'data')
        # Obtaining the member 'shape' of a type (line 273)
        shape_373776 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 273, 29), data_373775, 'shape')
        # Obtaining the member '__getitem__' of a type (line 273)
        getitem___373777 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 273, 29), shape_373776, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 273)
        subscript_call_result_373778 = invoke(stypy.reporting.localization.Localization(__file__, 273, 29), getitem___373777, int_373773)
        
        int_373779 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 273, 48), 'int')
        # Applying the binary operator '+' (line 273)
        result_add_373780 = python_operator(stypy.reporting.localization.Localization(__file__, 273, 29), '+', subscript_call_result_373778, int_373779)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 273, 29), tuple_373772, result_add_373780)
        # Adding element type (line 273)
        # Getting the type of 'm' (line 273)
        m_373781 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 273, 51), 'm', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 273, 29), tuple_373772, m_373781)
        
        # Processing the call keyword arguments (line 273)
        # Getting the type of 'self' (line 273)
        self_373782 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 273, 61), 'self', False)
        # Obtaining the member 'data' of a type (line 273)
        data_373783 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 273, 61), self_373782, 'data')
        # Obtaining the member 'dtype' of a type (line 273)
        dtype_373784 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 273, 61), data_373783, 'dtype')
        keyword_373785 = dtype_373784
        kwargs_373786 = {'dtype': keyword_373785}
        # Getting the type of 'np' (line 273)
        np_373770 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 273, 19), 'np', False)
        # Obtaining the member 'zeros' of a type (line 273)
        zeros_373771 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 273, 19), np_373770, 'zeros')
        # Calling zeros(args, kwargs) (line 273)
        zeros_call_result_373787 = invoke(stypy.reporting.localization.Localization(__file__, 273, 19), zeros_373771, *[tuple_373772], **kwargs_373786)
        
        # Assigning a type to the variable 'data' (line 273)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 273, 12), 'data', zeros_call_result_373787)
        
        # Assigning a Attribute to a Subscript (line 274):
        
        # Assigning a Attribute to a Subscript (line 274):
        # Getting the type of 'self' (line 274)
        self_373788 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 274, 44), 'self')
        # Obtaining the member 'data' of a type (line 274)
        data_373789 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 274, 44), self_373788, 'data')
        # Getting the type of 'data' (line 274)
        data_373790 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 274, 12), 'data')
        int_373791 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 274, 18), 'int')
        slice_373792 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 274, 12), None, int_373791, None)
        
        # Obtaining the type of the subscript
        int_373793 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 274, 38), 'int')
        # Getting the type of 'self' (line 274)
        self_373794 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 274, 22), 'self')
        # Obtaining the member 'data' of a type (line 274)
        data_373795 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 274, 22), self_373794, 'data')
        # Obtaining the member 'shape' of a type (line 274)
        shape_373796 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 274, 22), data_373795, 'shape')
        # Obtaining the member '__getitem__' of a type (line 274)
        getitem___373797 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 274, 22), shape_373796, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 274)
        subscript_call_result_373798 = invoke(stypy.reporting.localization.Localization(__file__, 274, 22), getitem___373797, int_373793)
        
        slice_373799 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 274, 12), None, subscript_call_result_373798, None)
        # Storing an element on a container (line 274)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 274, 12), data_373790, ((slice_373792, slice_373799), data_373789))
        
        # Assigning a Name to a Subscript (line 275):
        
        # Assigning a Name to a Subscript (line 275):
        # Getting the type of 'values' (line 275)
        values_373800 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 275, 44), 'values')
        # Getting the type of 'data' (line 275)
        data_373801 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 275, 12), 'data')
        int_373802 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 275, 17), 'int')
        # Getting the type of 'min_index' (line 275)
        min_index_373803 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 275, 21), 'min_index')
        # Getting the type of 'max_index' (line 275)
        max_index_373804 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 275, 31), 'max_index')
        slice_373805 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 275, 12), min_index_373803, max_index_373804, None)
        # Storing an element on a container (line 275)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 275, 12), data_373801, ((int_373802, slice_373805), values_373800))
        
        # Assigning a Name to a Attribute (line 276):
        
        # Assigning a Name to a Attribute (line 276):
        # Getting the type of 'data' (line 276)
        data_373806 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 276, 24), 'data')
        # Getting the type of 'self' (line 276)
        self_373807 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 276, 12), 'self')
        # Setting the type of the member 'data' of a type (line 276)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 276, 12), self_373807, 'data', data_373806)
        # SSA join for if statement (line 268)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of '_setdiag(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_setdiag' in the type store
        # Getting the type of 'stypy_return_type' (line 246)
        stypy_return_type_373808 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_373808)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_setdiag'
        return stypy_return_type_373808


    @norecursion
    def todia(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'False' (line 278)
        False_373809 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 278, 25), 'False')
        defaults = [False_373809]
        # Create a new context for function 'todia'
        module_type_store = module_type_store.open_function_context('todia', 278, 4, False)
        # Assigning a type to the variable 'self' (line 279)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 279, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        dia_matrix.todia.__dict__.__setitem__('stypy_localization', localization)
        dia_matrix.todia.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        dia_matrix.todia.__dict__.__setitem__('stypy_type_store', module_type_store)
        dia_matrix.todia.__dict__.__setitem__('stypy_function_name', 'dia_matrix.todia')
        dia_matrix.todia.__dict__.__setitem__('stypy_param_names_list', ['copy'])
        dia_matrix.todia.__dict__.__setitem__('stypy_varargs_param_name', None)
        dia_matrix.todia.__dict__.__setitem__('stypy_kwargs_param_name', None)
        dia_matrix.todia.__dict__.__setitem__('stypy_call_defaults', defaults)
        dia_matrix.todia.__dict__.__setitem__('stypy_call_varargs', varargs)
        dia_matrix.todia.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        dia_matrix.todia.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'dia_matrix.todia', ['copy'], None, None, defaults, varargs, kwargs)

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

        
        # Getting the type of 'copy' (line 279)
        copy_373810 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 279, 11), 'copy')
        # Testing the type of an if condition (line 279)
        if_condition_373811 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 279, 8), copy_373810)
        # Assigning a type to the variable 'if_condition_373811' (line 279)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 279, 8), 'if_condition_373811', if_condition_373811)
        # SSA begins for if statement (line 279)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to copy(...): (line 280)
        # Processing the call keyword arguments (line 280)
        kwargs_373814 = {}
        # Getting the type of 'self' (line 280)
        self_373812 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 280, 19), 'self', False)
        # Obtaining the member 'copy' of a type (line 280)
        copy_373813 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 280, 19), self_373812, 'copy')
        # Calling copy(args, kwargs) (line 280)
        copy_call_result_373815 = invoke(stypy.reporting.localization.Localization(__file__, 280, 19), copy_373813, *[], **kwargs_373814)
        
        # Assigning a type to the variable 'stypy_return_type' (line 280)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 280, 12), 'stypy_return_type', copy_call_result_373815)
        # SSA branch for the else part of an if statement (line 279)
        module_type_store.open_ssa_branch('else')
        # Getting the type of 'self' (line 282)
        self_373816 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 282, 19), 'self')
        # Assigning a type to the variable 'stypy_return_type' (line 282)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 282, 12), 'stypy_return_type', self_373816)
        # SSA join for if statement (line 279)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'todia(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'todia' in the type store
        # Getting the type of 'stypy_return_type' (line 278)
        stypy_return_type_373817 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 278, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_373817)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'todia'
        return stypy_return_type_373817

    
    # Assigning a Attribute to a Attribute (line 284):

    @norecursion
    def transpose(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 286)
        None_373818 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 286, 29), 'None')
        # Getting the type of 'False' (line 286)
        False_373819 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 286, 40), 'False')
        defaults = [None_373818, False_373819]
        # Create a new context for function 'transpose'
        module_type_store = module_type_store.open_function_context('transpose', 286, 4, False)
        # Assigning a type to the variable 'self' (line 287)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 287, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        dia_matrix.transpose.__dict__.__setitem__('stypy_localization', localization)
        dia_matrix.transpose.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        dia_matrix.transpose.__dict__.__setitem__('stypy_type_store', module_type_store)
        dia_matrix.transpose.__dict__.__setitem__('stypy_function_name', 'dia_matrix.transpose')
        dia_matrix.transpose.__dict__.__setitem__('stypy_param_names_list', ['axes', 'copy'])
        dia_matrix.transpose.__dict__.__setitem__('stypy_varargs_param_name', None)
        dia_matrix.transpose.__dict__.__setitem__('stypy_kwargs_param_name', None)
        dia_matrix.transpose.__dict__.__setitem__('stypy_call_defaults', defaults)
        dia_matrix.transpose.__dict__.__setitem__('stypy_call_varargs', varargs)
        dia_matrix.transpose.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        dia_matrix.transpose.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'dia_matrix.transpose', ['axes', 'copy'], None, None, defaults, varargs, kwargs)

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

        
        # Type idiom detected: calculating its left and rigth part (line 287)
        # Getting the type of 'axes' (line 287)
        axes_373820 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 287, 8), 'axes')
        # Getting the type of 'None' (line 287)
        None_373821 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 287, 23), 'None')
        
        (may_be_373822, more_types_in_union_373823) = may_not_be_none(axes_373820, None_373821)

        if may_be_373822:

            if more_types_in_union_373823:
                # Runtime conditional SSA (line 287)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Call to ValueError(...): (line 288)
            # Processing the call arguments (line 288)
            str_373825 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 288, 30), 'str', "Sparse matrices do not support an 'axes' parameter because swapping dimensions is the only logical permutation.")
            # Processing the call keyword arguments (line 288)
            kwargs_373826 = {}
            # Getting the type of 'ValueError' (line 288)
            ValueError_373824 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 288, 18), 'ValueError', False)
            # Calling ValueError(args, kwargs) (line 288)
            ValueError_call_result_373827 = invoke(stypy.reporting.localization.Localization(__file__, 288, 18), ValueError_373824, *[str_373825], **kwargs_373826)
            
            ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 288, 12), ValueError_call_result_373827, 'raise parameter', BaseException)

            if more_types_in_union_373823:
                # SSA join for if statement (line 287)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Assigning a Attribute to a Tuple (line 292):
        
        # Assigning a Subscript to a Name (line 292):
        
        # Obtaining the type of the subscript
        int_373828 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 292, 8), 'int')
        # Getting the type of 'self' (line 292)
        self_373829 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 292, 29), 'self')
        # Obtaining the member 'shape' of a type (line 292)
        shape_373830 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 292, 29), self_373829, 'shape')
        # Obtaining the member '__getitem__' of a type (line 292)
        getitem___373831 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 292, 8), shape_373830, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 292)
        subscript_call_result_373832 = invoke(stypy.reporting.localization.Localization(__file__, 292, 8), getitem___373831, int_373828)
        
        # Assigning a type to the variable 'tuple_var_assignment_372958' (line 292)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 292, 8), 'tuple_var_assignment_372958', subscript_call_result_373832)
        
        # Assigning a Subscript to a Name (line 292):
        
        # Obtaining the type of the subscript
        int_373833 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 292, 8), 'int')
        # Getting the type of 'self' (line 292)
        self_373834 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 292, 29), 'self')
        # Obtaining the member 'shape' of a type (line 292)
        shape_373835 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 292, 29), self_373834, 'shape')
        # Obtaining the member '__getitem__' of a type (line 292)
        getitem___373836 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 292, 8), shape_373835, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 292)
        subscript_call_result_373837 = invoke(stypy.reporting.localization.Localization(__file__, 292, 8), getitem___373836, int_373833)
        
        # Assigning a type to the variable 'tuple_var_assignment_372959' (line 292)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 292, 8), 'tuple_var_assignment_372959', subscript_call_result_373837)
        
        # Assigning a Name to a Name (line 292):
        # Getting the type of 'tuple_var_assignment_372958' (line 292)
        tuple_var_assignment_372958_373838 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 292, 8), 'tuple_var_assignment_372958')
        # Assigning a type to the variable 'num_rows' (line 292)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 292, 8), 'num_rows', tuple_var_assignment_372958_373838)
        
        # Assigning a Name to a Name (line 292):
        # Getting the type of 'tuple_var_assignment_372959' (line 292)
        tuple_var_assignment_372959_373839 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 292, 8), 'tuple_var_assignment_372959')
        # Assigning a type to the variable 'num_cols' (line 292)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 292, 18), 'num_cols', tuple_var_assignment_372959_373839)
        
        # Assigning a Call to a Name (line 293):
        
        # Assigning a Call to a Name (line 293):
        
        # Call to max(...): (line 293)
        # Processing the call arguments (line 293)
        # Getting the type of 'self' (line 293)
        self_373841 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 293, 22), 'self', False)
        # Obtaining the member 'shape' of a type (line 293)
        shape_373842 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 293, 22), self_373841, 'shape')
        # Processing the call keyword arguments (line 293)
        kwargs_373843 = {}
        # Getting the type of 'max' (line 293)
        max_373840 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 293, 18), 'max', False)
        # Calling max(args, kwargs) (line 293)
        max_call_result_373844 = invoke(stypy.reporting.localization.Localization(__file__, 293, 18), max_373840, *[shape_373842], **kwargs_373843)
        
        # Assigning a type to the variable 'max_dim' (line 293)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 293, 8), 'max_dim', max_call_result_373844)
        
        # Assigning a UnaryOp to a Name (line 296):
        
        # Assigning a UnaryOp to a Name (line 296):
        
        # Getting the type of 'self' (line 296)
        self_373845 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 296, 19), 'self')
        # Obtaining the member 'offsets' of a type (line 296)
        offsets_373846 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 296, 19), self_373845, 'offsets')
        # Applying the 'usub' unary operator (line 296)
        result___neg___373847 = python_operator(stypy.reporting.localization.Localization(__file__, 296, 18), 'usub', offsets_373846)
        
        # Assigning a type to the variable 'offsets' (line 296)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 296, 8), 'offsets', result___neg___373847)
        
        # Assigning a Subscript to a Name (line 299):
        
        # Assigning a Subscript to a Name (line 299):
        
        # Obtaining the type of the subscript
        slice_373848 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 299, 12), None, None, None)
        # Getting the type of 'None' (line 299)
        None_373849 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 299, 54), 'None')
        
        # Call to arange(...): (line 299)
        # Processing the call arguments (line 299)
        
        # Call to len(...): (line 299)
        # Processing the call arguments (line 299)
        # Getting the type of 'offsets' (line 299)
        offsets_373853 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 299, 26), 'offsets', False)
        # Processing the call keyword arguments (line 299)
        kwargs_373854 = {}
        # Getting the type of 'len' (line 299)
        len_373852 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 299, 22), 'len', False)
        # Calling len(args, kwargs) (line 299)
        len_call_result_373855 = invoke(stypy.reporting.localization.Localization(__file__, 299, 22), len_373852, *[offsets_373853], **kwargs_373854)
        
        # Processing the call keyword arguments (line 299)
        # Getting the type of 'np' (line 299)
        np_373856 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 299, 42), 'np', False)
        # Obtaining the member 'intc' of a type (line 299)
        intc_373857 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 299, 42), np_373856, 'intc')
        keyword_373858 = intc_373857
        kwargs_373859 = {'dtype': keyword_373858}
        # Getting the type of 'np' (line 299)
        np_373850 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 299, 12), 'np', False)
        # Obtaining the member 'arange' of a type (line 299)
        arange_373851 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 299, 12), np_373850, 'arange')
        # Calling arange(args, kwargs) (line 299)
        arange_call_result_373860 = invoke(stypy.reporting.localization.Localization(__file__, 299, 12), arange_373851, *[len_call_result_373855], **kwargs_373859)
        
        # Obtaining the member '__getitem__' of a type (line 299)
        getitem___373861 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 299, 12), arange_call_result_373860, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 299)
        subscript_call_result_373862 = invoke(stypy.reporting.localization.Localization(__file__, 299, 12), getitem___373861, (slice_373848, None_373849))
        
        # Assigning a type to the variable 'r' (line 299)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 299, 8), 'r', subscript_call_result_373862)
        
        # Assigning a BinOp to a Name (line 300):
        
        # Assigning a BinOp to a Name (line 300):
        
        # Call to arange(...): (line 300)
        # Processing the call arguments (line 300)
        # Getting the type of 'num_rows' (line 300)
        num_rows_373865 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 300, 22), 'num_rows', False)
        # Processing the call keyword arguments (line 300)
        # Getting the type of 'np' (line 300)
        np_373866 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 300, 38), 'np', False)
        # Obtaining the member 'intc' of a type (line 300)
        intc_373867 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 300, 38), np_373866, 'intc')
        keyword_373868 = intc_373867
        kwargs_373869 = {'dtype': keyword_373868}
        # Getting the type of 'np' (line 300)
        np_373863 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 300, 12), 'np', False)
        # Obtaining the member 'arange' of a type (line 300)
        arange_373864 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 300, 12), np_373863, 'arange')
        # Calling arange(args, kwargs) (line 300)
        arange_call_result_373870 = invoke(stypy.reporting.localization.Localization(__file__, 300, 12), arange_373864, *[num_rows_373865], **kwargs_373869)
        
        
        # Obtaining the type of the subscript
        slice_373871 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 300, 50), None, None, None)
        # Getting the type of 'None' (line 300)
        None_373872 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 300, 72), 'None')
        # Getting the type of 'offsets' (line 300)
        offsets_373873 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 300, 50), 'offsets')
        # Getting the type of 'max_dim' (line 300)
        max_dim_373874 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 300, 60), 'max_dim')
        # Applying the binary operator '%' (line 300)
        result_mod_373875 = python_operator(stypy.reporting.localization.Localization(__file__, 300, 50), '%', offsets_373873, max_dim_373874)
        
        # Obtaining the member '__getitem__' of a type (line 300)
        getitem___373876 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 300, 50), result_mod_373875, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 300)
        subscript_call_result_373877 = invoke(stypy.reporting.localization.Localization(__file__, 300, 50), getitem___373876, (slice_373871, None_373872))
        
        # Applying the binary operator '-' (line 300)
        result_sub_373878 = python_operator(stypy.reporting.localization.Localization(__file__, 300, 12), '-', arange_call_result_373870, subscript_call_result_373877)
        
        # Assigning a type to the variable 'c' (line 300)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 300, 8), 'c', result_sub_373878)
        
        # Assigning a Call to a Name (line 301):
        
        # Assigning a Call to a Name (line 301):
        
        # Call to max(...): (line 301)
        # Processing the call arguments (line 301)
        int_373880 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 301, 25), 'int')
        # Getting the type of 'max_dim' (line 301)
        max_dim_373881 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 301, 28), 'max_dim', False)
        
        # Obtaining the type of the subscript
        int_373882 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 301, 52), 'int')
        # Getting the type of 'self' (line 301)
        self_373883 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 301, 36), 'self', False)
        # Obtaining the member 'data' of a type (line 301)
        data_373884 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 301, 36), self_373883, 'data')
        # Obtaining the member 'shape' of a type (line 301)
        shape_373885 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 301, 36), data_373884, 'shape')
        # Obtaining the member '__getitem__' of a type (line 301)
        getitem___373886 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 301, 36), shape_373885, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 301)
        subscript_call_result_373887 = invoke(stypy.reporting.localization.Localization(__file__, 301, 36), getitem___373886, int_373882)
        
        # Applying the binary operator '-' (line 301)
        result_sub_373888 = python_operator(stypy.reporting.localization.Localization(__file__, 301, 28), '-', max_dim_373881, subscript_call_result_373887)
        
        # Processing the call keyword arguments (line 301)
        kwargs_373889 = {}
        # Getting the type of 'max' (line 301)
        max_373879 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 301, 21), 'max', False)
        # Calling max(args, kwargs) (line 301)
        max_call_result_373890 = invoke(stypy.reporting.localization.Localization(__file__, 301, 21), max_373879, *[int_373880, result_sub_373888], **kwargs_373889)
        
        # Assigning a type to the variable 'pad_amount' (line 301)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 301, 8), 'pad_amount', max_call_result_373890)
        
        # Assigning a Call to a Name (line 302):
        
        # Assigning a Call to a Name (line 302):
        
        # Call to hstack(...): (line 302)
        # Processing the call arguments (line 302)
        
        # Obtaining an instance of the builtin type 'tuple' (line 302)
        tuple_373893 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 302, 26), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 302)
        # Adding element type (line 302)
        # Getting the type of 'self' (line 302)
        self_373894 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 302, 26), 'self', False)
        # Obtaining the member 'data' of a type (line 302)
        data_373895 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 302, 26), self_373894, 'data')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 302, 26), tuple_373893, data_373895)
        # Adding element type (line 302)
        
        # Call to zeros(...): (line 302)
        # Processing the call arguments (line 302)
        
        # Obtaining an instance of the builtin type 'tuple' (line 302)
        tuple_373898 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 302, 47), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 302)
        # Adding element type (line 302)
        
        # Obtaining the type of the subscript
        int_373899 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 302, 63), 'int')
        # Getting the type of 'self' (line 302)
        self_373900 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 302, 47), 'self', False)
        # Obtaining the member 'data' of a type (line 302)
        data_373901 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 302, 47), self_373900, 'data')
        # Obtaining the member 'shape' of a type (line 302)
        shape_373902 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 302, 47), data_373901, 'shape')
        # Obtaining the member '__getitem__' of a type (line 302)
        getitem___373903 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 302, 47), shape_373902, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 302)
        subscript_call_result_373904 = invoke(stypy.reporting.localization.Localization(__file__, 302, 47), getitem___373903, int_373899)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 302, 47), tuple_373898, subscript_call_result_373904)
        # Adding element type (line 302)
        # Getting the type of 'pad_amount' (line 302)
        pad_amount_373905 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 302, 67), 'pad_amount', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 302, 47), tuple_373898, pad_amount_373905)
        
        # Processing the call keyword arguments (line 302)
        # Getting the type of 'self' (line 303)
        self_373906 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 303, 52), 'self', False)
        # Obtaining the member 'data' of a type (line 303)
        data_373907 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 303, 52), self_373906, 'data')
        # Obtaining the member 'dtype' of a type (line 303)
        dtype_373908 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 303, 52), data_373907, 'dtype')
        keyword_373909 = dtype_373908
        kwargs_373910 = {'dtype': keyword_373909}
        # Getting the type of 'np' (line 302)
        np_373896 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 302, 37), 'np', False)
        # Obtaining the member 'zeros' of a type (line 302)
        zeros_373897 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 302, 37), np_373896, 'zeros')
        # Calling zeros(args, kwargs) (line 302)
        zeros_call_result_373911 = invoke(stypy.reporting.localization.Localization(__file__, 302, 37), zeros_373897, *[tuple_373898], **kwargs_373910)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 302, 26), tuple_373893, zeros_call_result_373911)
        
        # Processing the call keyword arguments (line 302)
        kwargs_373912 = {}
        # Getting the type of 'np' (line 302)
        np_373891 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 302, 15), 'np', False)
        # Obtaining the member 'hstack' of a type (line 302)
        hstack_373892 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 302, 15), np_373891, 'hstack')
        # Calling hstack(args, kwargs) (line 302)
        hstack_call_result_373913 = invoke(stypy.reporting.localization.Localization(__file__, 302, 15), hstack_373892, *[tuple_373893], **kwargs_373912)
        
        # Assigning a type to the variable 'data' (line 302)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 302, 8), 'data', hstack_call_result_373913)
        
        # Assigning a Subscript to a Name (line 304):
        
        # Assigning a Subscript to a Name (line 304):
        
        # Obtaining the type of the subscript
        
        # Obtaining an instance of the builtin type 'tuple' (line 304)
        tuple_373914 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 304, 20), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 304)
        # Adding element type (line 304)
        # Getting the type of 'r' (line 304)
        r_373915 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 304, 20), 'r')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 304, 20), tuple_373914, r_373915)
        # Adding element type (line 304)
        # Getting the type of 'c' (line 304)
        c_373916 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 304, 23), 'c')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 304, 20), tuple_373914, c_373916)
        
        # Getting the type of 'data' (line 304)
        data_373917 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 304, 15), 'data')
        # Obtaining the member '__getitem__' of a type (line 304)
        getitem___373918 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 304, 15), data_373917, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 304)
        subscript_call_result_373919 = invoke(stypy.reporting.localization.Localization(__file__, 304, 15), getitem___373918, tuple_373914)
        
        # Assigning a type to the variable 'data' (line 304)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 304, 8), 'data', subscript_call_result_373919)
        
        # Call to dia_matrix(...): (line 305)
        # Processing the call arguments (line 305)
        
        # Obtaining an instance of the builtin type 'tuple' (line 305)
        tuple_373921 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 305, 27), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 305)
        # Adding element type (line 305)
        # Getting the type of 'data' (line 305)
        data_373922 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 305, 27), 'data', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 305, 27), tuple_373921, data_373922)
        # Adding element type (line 305)
        # Getting the type of 'offsets' (line 305)
        offsets_373923 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 305, 33), 'offsets', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 305, 27), tuple_373921, offsets_373923)
        
        # Processing the call keyword arguments (line 305)
        
        # Obtaining an instance of the builtin type 'tuple' (line 306)
        tuple_373924 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 306, 12), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 306)
        # Adding element type (line 306)
        # Getting the type of 'num_cols' (line 306)
        num_cols_373925 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 306, 12), 'num_cols', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 306, 12), tuple_373924, num_cols_373925)
        # Adding element type (line 306)
        # Getting the type of 'num_rows' (line 306)
        num_rows_373926 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 306, 22), 'num_rows', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 306, 12), tuple_373924, num_rows_373926)
        
        keyword_373927 = tuple_373924
        # Getting the type of 'copy' (line 306)
        copy_373928 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 306, 38), 'copy', False)
        keyword_373929 = copy_373928
        kwargs_373930 = {'shape': keyword_373927, 'copy': keyword_373929}
        # Getting the type of 'dia_matrix' (line 305)
        dia_matrix_373920 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 305, 15), 'dia_matrix', False)
        # Calling dia_matrix(args, kwargs) (line 305)
        dia_matrix_call_result_373931 = invoke(stypy.reporting.localization.Localization(__file__, 305, 15), dia_matrix_373920, *[tuple_373921], **kwargs_373930)
        
        # Assigning a type to the variable 'stypy_return_type' (line 305)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 305, 8), 'stypy_return_type', dia_matrix_call_result_373931)
        
        # ################# End of 'transpose(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'transpose' in the type store
        # Getting the type of 'stypy_return_type' (line 286)
        stypy_return_type_373932 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 286, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_373932)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'transpose'
        return stypy_return_type_373932

    
    # Assigning a Attribute to a Attribute (line 308):

    @norecursion
    def diagonal(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        int_373933 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 310, 25), 'int')
        defaults = [int_373933]
        # Create a new context for function 'diagonal'
        module_type_store = module_type_store.open_function_context('diagonal', 310, 4, False)
        # Assigning a type to the variable 'self' (line 311)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 311, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        dia_matrix.diagonal.__dict__.__setitem__('stypy_localization', localization)
        dia_matrix.diagonal.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        dia_matrix.diagonal.__dict__.__setitem__('stypy_type_store', module_type_store)
        dia_matrix.diagonal.__dict__.__setitem__('stypy_function_name', 'dia_matrix.diagonal')
        dia_matrix.diagonal.__dict__.__setitem__('stypy_param_names_list', ['k'])
        dia_matrix.diagonal.__dict__.__setitem__('stypy_varargs_param_name', None)
        dia_matrix.diagonal.__dict__.__setitem__('stypy_kwargs_param_name', None)
        dia_matrix.diagonal.__dict__.__setitem__('stypy_call_defaults', defaults)
        dia_matrix.diagonal.__dict__.__setitem__('stypy_call_varargs', varargs)
        dia_matrix.diagonal.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        dia_matrix.diagonal.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'dia_matrix.diagonal', ['k'], None, None, defaults, varargs, kwargs)

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

        
        # Assigning a Attribute to a Tuple (line 311):
        
        # Assigning a Subscript to a Name (line 311):
        
        # Obtaining the type of the subscript
        int_373934 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 311, 8), 'int')
        # Getting the type of 'self' (line 311)
        self_373935 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 311, 21), 'self')
        # Obtaining the member 'shape' of a type (line 311)
        shape_373936 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 311, 21), self_373935, 'shape')
        # Obtaining the member '__getitem__' of a type (line 311)
        getitem___373937 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 311, 8), shape_373936, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 311)
        subscript_call_result_373938 = invoke(stypy.reporting.localization.Localization(__file__, 311, 8), getitem___373937, int_373934)
        
        # Assigning a type to the variable 'tuple_var_assignment_372960' (line 311)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 311, 8), 'tuple_var_assignment_372960', subscript_call_result_373938)
        
        # Assigning a Subscript to a Name (line 311):
        
        # Obtaining the type of the subscript
        int_373939 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 311, 8), 'int')
        # Getting the type of 'self' (line 311)
        self_373940 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 311, 21), 'self')
        # Obtaining the member 'shape' of a type (line 311)
        shape_373941 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 311, 21), self_373940, 'shape')
        # Obtaining the member '__getitem__' of a type (line 311)
        getitem___373942 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 311, 8), shape_373941, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 311)
        subscript_call_result_373943 = invoke(stypy.reporting.localization.Localization(__file__, 311, 8), getitem___373942, int_373939)
        
        # Assigning a type to the variable 'tuple_var_assignment_372961' (line 311)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 311, 8), 'tuple_var_assignment_372961', subscript_call_result_373943)
        
        # Assigning a Name to a Name (line 311):
        # Getting the type of 'tuple_var_assignment_372960' (line 311)
        tuple_var_assignment_372960_373944 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 311, 8), 'tuple_var_assignment_372960')
        # Assigning a type to the variable 'rows' (line 311)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 311, 8), 'rows', tuple_var_assignment_372960_373944)
        
        # Assigning a Name to a Name (line 311):
        # Getting the type of 'tuple_var_assignment_372961' (line 311)
        tuple_var_assignment_372961_373945 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 311, 8), 'tuple_var_assignment_372961')
        # Assigning a type to the variable 'cols' (line 311)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 311, 14), 'cols', tuple_var_assignment_372961_373945)
        
        
        # Evaluating a boolean operation
        
        # Getting the type of 'k' (line 312)
        k_373946 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 312, 11), 'k')
        
        # Getting the type of 'rows' (line 312)
        rows_373947 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 312, 17), 'rows')
        # Applying the 'usub' unary operator (line 312)
        result___neg___373948 = python_operator(stypy.reporting.localization.Localization(__file__, 312, 16), 'usub', rows_373947)
        
        # Applying the binary operator '<=' (line 312)
        result_le_373949 = python_operator(stypy.reporting.localization.Localization(__file__, 312, 11), '<=', k_373946, result___neg___373948)
        
        
        # Getting the type of 'k' (line 312)
        k_373950 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 312, 25), 'k')
        # Getting the type of 'cols' (line 312)
        cols_373951 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 312, 30), 'cols')
        # Applying the binary operator '>=' (line 312)
        result_ge_373952 = python_operator(stypy.reporting.localization.Localization(__file__, 312, 25), '>=', k_373950, cols_373951)
        
        # Applying the binary operator 'or' (line 312)
        result_or_keyword_373953 = python_operator(stypy.reporting.localization.Localization(__file__, 312, 11), 'or', result_le_373949, result_ge_373952)
        
        # Testing the type of an if condition (line 312)
        if_condition_373954 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 312, 8), result_or_keyword_373953)
        # Assigning a type to the variable 'if_condition_373954' (line 312)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 312, 8), 'if_condition_373954', if_condition_373954)
        # SSA begins for if statement (line 312)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to ValueError(...): (line 313)
        # Processing the call arguments (line 313)
        str_373956 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 313, 29), 'str', 'k exceeds matrix dimensions')
        # Processing the call keyword arguments (line 313)
        kwargs_373957 = {}
        # Getting the type of 'ValueError' (line 313)
        ValueError_373955 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 313, 18), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 313)
        ValueError_call_result_373958 = invoke(stypy.reporting.localization.Localization(__file__, 313, 18), ValueError_373955, *[str_373956], **kwargs_373957)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 313, 12), ValueError_call_result_373958, 'raise parameter', BaseException)
        # SSA join for if statement (line 312)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Tuple (line 314):
        
        # Assigning a Subscript to a Name (line 314):
        
        # Obtaining the type of the subscript
        int_373959 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 314, 8), 'int')
        
        # Call to where(...): (line 314)
        # Processing the call arguments (line 314)
        
        # Getting the type of 'self' (line 314)
        self_373962 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 314, 24), 'self', False)
        # Obtaining the member 'offsets' of a type (line 314)
        offsets_373963 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 314, 24), self_373962, 'offsets')
        # Getting the type of 'k' (line 314)
        k_373964 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 314, 40), 'k', False)
        # Applying the binary operator '==' (line 314)
        result_eq_373965 = python_operator(stypy.reporting.localization.Localization(__file__, 314, 24), '==', offsets_373963, k_373964)
        
        # Processing the call keyword arguments (line 314)
        kwargs_373966 = {}
        # Getting the type of 'np' (line 314)
        np_373960 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 314, 15), 'np', False)
        # Obtaining the member 'where' of a type (line 314)
        where_373961 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 314, 15), np_373960, 'where')
        # Calling where(args, kwargs) (line 314)
        where_call_result_373967 = invoke(stypy.reporting.localization.Localization(__file__, 314, 15), where_373961, *[result_eq_373965], **kwargs_373966)
        
        # Obtaining the member '__getitem__' of a type (line 314)
        getitem___373968 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 314, 8), where_call_result_373967, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 314)
        subscript_call_result_373969 = invoke(stypy.reporting.localization.Localization(__file__, 314, 8), getitem___373968, int_373959)
        
        # Assigning a type to the variable 'tuple_var_assignment_372962' (line 314)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 314, 8), 'tuple_var_assignment_372962', subscript_call_result_373969)
        
        # Assigning a Name to a Name (line 314):
        # Getting the type of 'tuple_var_assignment_372962' (line 314)
        tuple_var_assignment_372962_373970 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 314, 8), 'tuple_var_assignment_372962')
        # Assigning a type to the variable 'idx' (line 314)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 314, 8), 'idx', tuple_var_assignment_372962_373970)
        
        # Assigning a Tuple to a Tuple (line 315):
        
        # Assigning a Call to a Name (line 315):
        
        # Call to max(...): (line 315)
        # Processing the call arguments (line 315)
        int_373972 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 315, 34), 'int')
        # Getting the type of 'k' (line 315)
        k_373973 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 315, 37), 'k', False)
        # Processing the call keyword arguments (line 315)
        kwargs_373974 = {}
        # Getting the type of 'max' (line 315)
        max_373971 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 315, 30), 'max', False)
        # Calling max(args, kwargs) (line 315)
        max_call_result_373975 = invoke(stypy.reporting.localization.Localization(__file__, 315, 30), max_373971, *[int_373972, k_373973], **kwargs_373974)
        
        # Assigning a type to the variable 'tuple_assignment_372963' (line 315)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 315, 8), 'tuple_assignment_372963', max_call_result_373975)
        
        # Assigning a Call to a Name (line 315):
        
        # Call to min(...): (line 315)
        # Processing the call arguments (line 315)
        # Getting the type of 'rows' (line 315)
        rows_373977 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 315, 45), 'rows', False)
        # Getting the type of 'k' (line 315)
        k_373978 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 315, 52), 'k', False)
        # Applying the binary operator '+' (line 315)
        result_add_373979 = python_operator(stypy.reporting.localization.Localization(__file__, 315, 45), '+', rows_373977, k_373978)
        
        # Getting the type of 'cols' (line 315)
        cols_373980 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 315, 55), 'cols', False)
        # Processing the call keyword arguments (line 315)
        kwargs_373981 = {}
        # Getting the type of 'min' (line 315)
        min_373976 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 315, 41), 'min', False)
        # Calling min(args, kwargs) (line 315)
        min_call_result_373982 = invoke(stypy.reporting.localization.Localization(__file__, 315, 41), min_373976, *[result_add_373979, cols_373980], **kwargs_373981)
        
        # Assigning a type to the variable 'tuple_assignment_372964' (line 315)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 315, 8), 'tuple_assignment_372964', min_call_result_373982)
        
        # Assigning a Name to a Name (line 315):
        # Getting the type of 'tuple_assignment_372963' (line 315)
        tuple_assignment_372963_373983 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 315, 8), 'tuple_assignment_372963')
        # Assigning a type to the variable 'first_col' (line 315)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 315, 8), 'first_col', tuple_assignment_372963_373983)
        
        # Assigning a Name to a Name (line 315):
        # Getting the type of 'tuple_assignment_372964' (line 315)
        tuple_assignment_372964_373984 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 315, 8), 'tuple_assignment_372964')
        # Assigning a type to the variable 'last_col' (line 315)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 315, 19), 'last_col', tuple_assignment_372964_373984)
        
        
        # Getting the type of 'idx' (line 316)
        idx_373985 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 316, 11), 'idx')
        # Obtaining the member 'size' of a type (line 316)
        size_373986 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 316, 11), idx_373985, 'size')
        int_373987 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 316, 23), 'int')
        # Applying the binary operator '==' (line 316)
        result_eq_373988 = python_operator(stypy.reporting.localization.Localization(__file__, 316, 11), '==', size_373986, int_373987)
        
        # Testing the type of an if condition (line 316)
        if_condition_373989 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 316, 8), result_eq_373988)
        # Assigning a type to the variable 'if_condition_373989' (line 316)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 316, 8), 'if_condition_373989', if_condition_373989)
        # SSA begins for if statement (line 316)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to zeros(...): (line 317)
        # Processing the call arguments (line 317)
        # Getting the type of 'last_col' (line 317)
        last_col_373992 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 317, 28), 'last_col', False)
        # Getting the type of 'first_col' (line 317)
        first_col_373993 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 317, 39), 'first_col', False)
        # Applying the binary operator '-' (line 317)
        result_sub_373994 = python_operator(stypy.reporting.localization.Localization(__file__, 317, 28), '-', last_col_373992, first_col_373993)
        
        # Processing the call keyword arguments (line 317)
        # Getting the type of 'self' (line 317)
        self_373995 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 317, 56), 'self', False)
        # Obtaining the member 'data' of a type (line 317)
        data_373996 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 317, 56), self_373995, 'data')
        # Obtaining the member 'dtype' of a type (line 317)
        dtype_373997 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 317, 56), data_373996, 'dtype')
        keyword_373998 = dtype_373997
        kwargs_373999 = {'dtype': keyword_373998}
        # Getting the type of 'np' (line 317)
        np_373990 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 317, 19), 'np', False)
        # Obtaining the member 'zeros' of a type (line 317)
        zeros_373991 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 317, 19), np_373990, 'zeros')
        # Calling zeros(args, kwargs) (line 317)
        zeros_call_result_374000 = invoke(stypy.reporting.localization.Localization(__file__, 317, 19), zeros_373991, *[result_sub_373994], **kwargs_373999)
        
        # Assigning a type to the variable 'stypy_return_type' (line 317)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 317, 12), 'stypy_return_type', zeros_call_result_374000)
        # SSA join for if statement (line 316)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Obtaining the type of the subscript
        
        # Obtaining the type of the subscript
        int_374001 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 318, 29), 'int')
        # Getting the type of 'idx' (line 318)
        idx_374002 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 318, 25), 'idx')
        # Obtaining the member '__getitem__' of a type (line 318)
        getitem___374003 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 318, 25), idx_374002, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 318)
        subscript_call_result_374004 = invoke(stypy.reporting.localization.Localization(__file__, 318, 25), getitem___374003, int_374001)
        
        # Getting the type of 'first_col' (line 318)
        first_col_374005 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 318, 33), 'first_col')
        # Getting the type of 'last_col' (line 318)
        last_col_374006 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 318, 43), 'last_col')
        slice_374007 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 318, 15), first_col_374005, last_col_374006, None)
        # Getting the type of 'self' (line 318)
        self_374008 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 318, 15), 'self')
        # Obtaining the member 'data' of a type (line 318)
        data_374009 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 318, 15), self_374008, 'data')
        # Obtaining the member '__getitem__' of a type (line 318)
        getitem___374010 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 318, 15), data_374009, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 318)
        subscript_call_result_374011 = invoke(stypy.reporting.localization.Localization(__file__, 318, 15), getitem___374010, (subscript_call_result_374004, slice_374007))
        
        # Assigning a type to the variable 'stypy_return_type' (line 318)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 318, 8), 'stypy_return_type', subscript_call_result_374011)
        
        # ################# End of 'diagonal(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'diagonal' in the type store
        # Getting the type of 'stypy_return_type' (line 310)
        stypy_return_type_374012 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 310, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_374012)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'diagonal'
        return stypy_return_type_374012

    
    # Assigning a Attribute to a Attribute (line 320):

    @norecursion
    def tocsc(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'False' (line 322)
        False_374013 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 322, 25), 'False')
        defaults = [False_374013]
        # Create a new context for function 'tocsc'
        module_type_store = module_type_store.open_function_context('tocsc', 322, 4, False)
        # Assigning a type to the variable 'self' (line 323)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 323, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        dia_matrix.tocsc.__dict__.__setitem__('stypy_localization', localization)
        dia_matrix.tocsc.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        dia_matrix.tocsc.__dict__.__setitem__('stypy_type_store', module_type_store)
        dia_matrix.tocsc.__dict__.__setitem__('stypy_function_name', 'dia_matrix.tocsc')
        dia_matrix.tocsc.__dict__.__setitem__('stypy_param_names_list', ['copy'])
        dia_matrix.tocsc.__dict__.__setitem__('stypy_varargs_param_name', None)
        dia_matrix.tocsc.__dict__.__setitem__('stypy_kwargs_param_name', None)
        dia_matrix.tocsc.__dict__.__setitem__('stypy_call_defaults', defaults)
        dia_matrix.tocsc.__dict__.__setitem__('stypy_call_varargs', varargs)
        dia_matrix.tocsc.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        dia_matrix.tocsc.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'dia_matrix.tocsc', ['copy'], None, None, defaults, varargs, kwargs)

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

        stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 323, 8))
        
        # 'from scipy.sparse.csc import csc_matrix' statement (line 323)
        update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/sparse/')
        import_374014 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 323, 8), 'scipy.sparse.csc')

        if (type(import_374014) is not StypyTypeError):

            if (import_374014 != 'pyd_module'):
                __import__(import_374014)
                sys_modules_374015 = sys.modules[import_374014]
                import_from_module(stypy.reporting.localization.Localization(__file__, 323, 8), 'scipy.sparse.csc', sys_modules_374015.module_type_store, module_type_store, ['csc_matrix'])
                nest_module(stypy.reporting.localization.Localization(__file__, 323, 8), __file__, sys_modules_374015, sys_modules_374015.module_type_store, module_type_store)
            else:
                from scipy.sparse.csc import csc_matrix

                import_from_module(stypy.reporting.localization.Localization(__file__, 323, 8), 'scipy.sparse.csc', None, module_type_store, ['csc_matrix'], [csc_matrix])

        else:
            # Assigning a type to the variable 'scipy.sparse.csc' (line 323)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 323, 8), 'scipy.sparse.csc', import_374014)

        remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/sparse/')
        
        
        
        # Getting the type of 'self' (line 324)
        self_374016 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 324, 11), 'self')
        # Obtaining the member 'nnz' of a type (line 324)
        nnz_374017 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 324, 11), self_374016, 'nnz')
        int_374018 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 324, 23), 'int')
        # Applying the binary operator '==' (line 324)
        result_eq_374019 = python_operator(stypy.reporting.localization.Localization(__file__, 324, 11), '==', nnz_374017, int_374018)
        
        # Testing the type of an if condition (line 324)
        if_condition_374020 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 324, 8), result_eq_374019)
        # Assigning a type to the variable 'if_condition_374020' (line 324)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 324, 8), 'if_condition_374020', if_condition_374020)
        # SSA begins for if statement (line 324)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to csc_matrix(...): (line 325)
        # Processing the call arguments (line 325)
        # Getting the type of 'self' (line 325)
        self_374022 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 325, 30), 'self', False)
        # Obtaining the member 'shape' of a type (line 325)
        shape_374023 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 325, 30), self_374022, 'shape')
        # Processing the call keyword arguments (line 325)
        # Getting the type of 'self' (line 325)
        self_374024 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 325, 48), 'self', False)
        # Obtaining the member 'dtype' of a type (line 325)
        dtype_374025 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 325, 48), self_374024, 'dtype')
        keyword_374026 = dtype_374025
        kwargs_374027 = {'dtype': keyword_374026}
        # Getting the type of 'csc_matrix' (line 325)
        csc_matrix_374021 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 325, 19), 'csc_matrix', False)
        # Calling csc_matrix(args, kwargs) (line 325)
        csc_matrix_call_result_374028 = invoke(stypy.reporting.localization.Localization(__file__, 325, 19), csc_matrix_374021, *[shape_374023], **kwargs_374027)
        
        # Assigning a type to the variable 'stypy_return_type' (line 325)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 325, 12), 'stypy_return_type', csc_matrix_call_result_374028)
        # SSA join for if statement (line 324)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Attribute to a Tuple (line 327):
        
        # Assigning a Subscript to a Name (line 327):
        
        # Obtaining the type of the subscript
        int_374029 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 327, 8), 'int')
        # Getting the type of 'self' (line 327)
        self_374030 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 327, 29), 'self')
        # Obtaining the member 'shape' of a type (line 327)
        shape_374031 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 327, 29), self_374030, 'shape')
        # Obtaining the member '__getitem__' of a type (line 327)
        getitem___374032 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 327, 8), shape_374031, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 327)
        subscript_call_result_374033 = invoke(stypy.reporting.localization.Localization(__file__, 327, 8), getitem___374032, int_374029)
        
        # Assigning a type to the variable 'tuple_var_assignment_372965' (line 327)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 327, 8), 'tuple_var_assignment_372965', subscript_call_result_374033)
        
        # Assigning a Subscript to a Name (line 327):
        
        # Obtaining the type of the subscript
        int_374034 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 327, 8), 'int')
        # Getting the type of 'self' (line 327)
        self_374035 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 327, 29), 'self')
        # Obtaining the member 'shape' of a type (line 327)
        shape_374036 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 327, 29), self_374035, 'shape')
        # Obtaining the member '__getitem__' of a type (line 327)
        getitem___374037 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 327, 8), shape_374036, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 327)
        subscript_call_result_374038 = invoke(stypy.reporting.localization.Localization(__file__, 327, 8), getitem___374037, int_374034)
        
        # Assigning a type to the variable 'tuple_var_assignment_372966' (line 327)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 327, 8), 'tuple_var_assignment_372966', subscript_call_result_374038)
        
        # Assigning a Name to a Name (line 327):
        # Getting the type of 'tuple_var_assignment_372965' (line 327)
        tuple_var_assignment_372965_374039 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 327, 8), 'tuple_var_assignment_372965')
        # Assigning a type to the variable 'num_rows' (line 327)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 327, 8), 'num_rows', tuple_var_assignment_372965_374039)
        
        # Assigning a Name to a Name (line 327):
        # Getting the type of 'tuple_var_assignment_372966' (line 327)
        tuple_var_assignment_372966_374040 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 327, 8), 'tuple_var_assignment_372966')
        # Assigning a type to the variable 'num_cols' (line 327)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 327, 18), 'num_cols', tuple_var_assignment_372966_374040)
        
        # Assigning a Attribute to a Tuple (line 328):
        
        # Assigning a Subscript to a Name (line 328):
        
        # Obtaining the type of the subscript
        int_374041 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 328, 8), 'int')
        # Getting the type of 'self' (line 328)
        self_374042 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 328, 34), 'self')
        # Obtaining the member 'data' of a type (line 328)
        data_374043 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 328, 34), self_374042, 'data')
        # Obtaining the member 'shape' of a type (line 328)
        shape_374044 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 328, 34), data_374043, 'shape')
        # Obtaining the member '__getitem__' of a type (line 328)
        getitem___374045 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 328, 8), shape_374044, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 328)
        subscript_call_result_374046 = invoke(stypy.reporting.localization.Localization(__file__, 328, 8), getitem___374045, int_374041)
        
        # Assigning a type to the variable 'tuple_var_assignment_372967' (line 328)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 328, 8), 'tuple_var_assignment_372967', subscript_call_result_374046)
        
        # Assigning a Subscript to a Name (line 328):
        
        # Obtaining the type of the subscript
        int_374047 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 328, 8), 'int')
        # Getting the type of 'self' (line 328)
        self_374048 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 328, 34), 'self')
        # Obtaining the member 'data' of a type (line 328)
        data_374049 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 328, 34), self_374048, 'data')
        # Obtaining the member 'shape' of a type (line 328)
        shape_374050 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 328, 34), data_374049, 'shape')
        # Obtaining the member '__getitem__' of a type (line 328)
        getitem___374051 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 328, 8), shape_374050, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 328)
        subscript_call_result_374052 = invoke(stypy.reporting.localization.Localization(__file__, 328, 8), getitem___374051, int_374047)
        
        # Assigning a type to the variable 'tuple_var_assignment_372968' (line 328)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 328, 8), 'tuple_var_assignment_372968', subscript_call_result_374052)
        
        # Assigning a Name to a Name (line 328):
        # Getting the type of 'tuple_var_assignment_372967' (line 328)
        tuple_var_assignment_372967_374053 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 328, 8), 'tuple_var_assignment_372967')
        # Assigning a type to the variable 'num_offsets' (line 328)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 328, 8), 'num_offsets', tuple_var_assignment_372967_374053)
        
        # Assigning a Name to a Name (line 328):
        # Getting the type of 'tuple_var_assignment_372968' (line 328)
        tuple_var_assignment_372968_374054 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 328, 8), 'tuple_var_assignment_372968')
        # Assigning a type to the variable 'offset_len' (line 328)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 328, 21), 'offset_len', tuple_var_assignment_372968_374054)
        
        # Assigning a Call to a Name (line 329):
        
        # Assigning a Call to a Name (line 329):
        
        # Call to arange(...): (line 329)
        # Processing the call arguments (line 329)
        # Getting the type of 'offset_len' (line 329)
        offset_len_374057 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 329, 32), 'offset_len', False)
        # Processing the call keyword arguments (line 329)
        kwargs_374058 = {}
        # Getting the type of 'np' (line 329)
        np_374055 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 329, 22), 'np', False)
        # Obtaining the member 'arange' of a type (line 329)
        arange_374056 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 329, 22), np_374055, 'arange')
        # Calling arange(args, kwargs) (line 329)
        arange_call_result_374059 = invoke(stypy.reporting.localization.Localization(__file__, 329, 22), arange_374056, *[offset_len_374057], **kwargs_374058)
        
        # Assigning a type to the variable 'offset_inds' (line 329)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 329, 8), 'offset_inds', arange_call_result_374059)
        
        # Assigning a BinOp to a Name (line 331):
        
        # Assigning a BinOp to a Name (line 331):
        # Getting the type of 'offset_inds' (line 331)
        offset_inds_374060 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 331, 14), 'offset_inds')
        
        # Obtaining the type of the subscript
        slice_374061 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 331, 28), None, None, None)
        # Getting the type of 'None' (line 331)
        None_374062 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 331, 43), 'None')
        # Getting the type of 'self' (line 331)
        self_374063 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 331, 28), 'self')
        # Obtaining the member 'offsets' of a type (line 331)
        offsets_374064 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 331, 28), self_374063, 'offsets')
        # Obtaining the member '__getitem__' of a type (line 331)
        getitem___374065 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 331, 28), offsets_374064, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 331)
        subscript_call_result_374066 = invoke(stypy.reporting.localization.Localization(__file__, 331, 28), getitem___374065, (slice_374061, None_374062))
        
        # Applying the binary operator '-' (line 331)
        result_sub_374067 = python_operator(stypy.reporting.localization.Localization(__file__, 331, 14), '-', offset_inds_374060, subscript_call_result_374066)
        
        # Assigning a type to the variable 'row' (line 331)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 331, 8), 'row', result_sub_374067)
        
        # Assigning a Compare to a Name (line 332):
        
        # Assigning a Compare to a Name (line 332):
        
        # Getting the type of 'row' (line 332)
        row_374068 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 332, 16), 'row')
        int_374069 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 332, 23), 'int')
        # Applying the binary operator '>=' (line 332)
        result_ge_374070 = python_operator(stypy.reporting.localization.Localization(__file__, 332, 16), '>=', row_374068, int_374069)
        
        # Assigning a type to the variable 'mask' (line 332)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 332, 8), 'mask', result_ge_374070)
        
        # Getting the type of 'mask' (line 333)
        mask_374071 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 333, 8), 'mask')
        
        # Getting the type of 'row' (line 333)
        row_374072 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 333, 17), 'row')
        # Getting the type of 'num_rows' (line 333)
        num_rows_374073 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 333, 23), 'num_rows')
        # Applying the binary operator '<' (line 333)
        result_lt_374074 = python_operator(stypy.reporting.localization.Localization(__file__, 333, 17), '<', row_374072, num_rows_374073)
        
        # Applying the binary operator '&=' (line 333)
        result_iand_374075 = python_operator(stypy.reporting.localization.Localization(__file__, 333, 8), '&=', mask_374071, result_lt_374074)
        # Assigning a type to the variable 'mask' (line 333)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 333, 8), 'mask', result_iand_374075)
        
        
        # Getting the type of 'mask' (line 334)
        mask_374076 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 334, 8), 'mask')
        
        # Getting the type of 'offset_inds' (line 334)
        offset_inds_374077 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 334, 17), 'offset_inds')
        # Getting the type of 'num_cols' (line 334)
        num_cols_374078 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 334, 31), 'num_cols')
        # Applying the binary operator '<' (line 334)
        result_lt_374079 = python_operator(stypy.reporting.localization.Localization(__file__, 334, 17), '<', offset_inds_374077, num_cols_374078)
        
        # Applying the binary operator '&=' (line 334)
        result_iand_374080 = python_operator(stypy.reporting.localization.Localization(__file__, 334, 8), '&=', mask_374076, result_lt_374079)
        # Assigning a type to the variable 'mask' (line 334)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 334, 8), 'mask', result_iand_374080)
        
        
        # Getting the type of 'mask' (line 335)
        mask_374081 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 335, 8), 'mask')
        
        # Getting the type of 'self' (line 335)
        self_374082 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 335, 17), 'self')
        # Obtaining the member 'data' of a type (line 335)
        data_374083 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 335, 17), self_374082, 'data')
        int_374084 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 335, 30), 'int')
        # Applying the binary operator '!=' (line 335)
        result_ne_374085 = python_operator(stypy.reporting.localization.Localization(__file__, 335, 17), '!=', data_374083, int_374084)
        
        # Applying the binary operator '&=' (line 335)
        result_iand_374086 = python_operator(stypy.reporting.localization.Localization(__file__, 335, 8), '&=', mask_374081, result_ne_374085)
        # Assigning a type to the variable 'mask' (line 335)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 335, 8), 'mask', result_iand_374086)
        
        
        # Assigning a Call to a Name (line 337):
        
        # Assigning a Call to a Name (line 337):
        
        # Call to get_index_dtype(...): (line 337)
        # Processing the call keyword arguments (line 337)
        
        # Call to max(...): (line 337)
        # Processing the call arguments (line 337)
        # Getting the type of 'self' (line 337)
        self_374089 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 337, 47), 'self', False)
        # Obtaining the member 'shape' of a type (line 337)
        shape_374090 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 337, 47), self_374089, 'shape')
        # Processing the call keyword arguments (line 337)
        kwargs_374091 = {}
        # Getting the type of 'max' (line 337)
        max_374088 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 337, 43), 'max', False)
        # Calling max(args, kwargs) (line 337)
        max_call_result_374092 = invoke(stypy.reporting.localization.Localization(__file__, 337, 43), max_374088, *[shape_374090], **kwargs_374091)
        
        keyword_374093 = max_call_result_374092
        kwargs_374094 = {'maxval': keyword_374093}
        # Getting the type of 'get_index_dtype' (line 337)
        get_index_dtype_374087 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 337, 20), 'get_index_dtype', False)
        # Calling get_index_dtype(args, kwargs) (line 337)
        get_index_dtype_call_result_374095 = invoke(stypy.reporting.localization.Localization(__file__, 337, 20), get_index_dtype_374087, *[], **kwargs_374094)
        
        # Assigning a type to the variable 'idx_dtype' (line 337)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 337, 8), 'idx_dtype', get_index_dtype_call_result_374095)
        
        # Assigning a Call to a Name (line 338):
        
        # Assigning a Call to a Name (line 338):
        
        # Call to zeros(...): (line 338)
        # Processing the call arguments (line 338)
        # Getting the type of 'num_cols' (line 338)
        num_cols_374098 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 338, 26), 'num_cols', False)
        int_374099 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 338, 37), 'int')
        # Applying the binary operator '+' (line 338)
        result_add_374100 = python_operator(stypy.reporting.localization.Localization(__file__, 338, 26), '+', num_cols_374098, int_374099)
        
        # Processing the call keyword arguments (line 338)
        # Getting the type of 'idx_dtype' (line 338)
        idx_dtype_374101 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 338, 46), 'idx_dtype', False)
        keyword_374102 = idx_dtype_374101
        kwargs_374103 = {'dtype': keyword_374102}
        # Getting the type of 'np' (line 338)
        np_374096 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 338, 17), 'np', False)
        # Obtaining the member 'zeros' of a type (line 338)
        zeros_374097 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 338, 17), np_374096, 'zeros')
        # Calling zeros(args, kwargs) (line 338)
        zeros_call_result_374104 = invoke(stypy.reporting.localization.Localization(__file__, 338, 17), zeros_374097, *[result_add_374100], **kwargs_374103)
        
        # Assigning a type to the variable 'indptr' (line 338)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 338, 8), 'indptr', zeros_call_result_374104)
        
        # Assigning a Call to a Subscript (line 339):
        
        # Assigning a Call to a Subscript (line 339):
        
        # Call to cumsum(...): (line 339)
        # Processing the call arguments (line 339)
        
        # Call to sum(...): (line 339)
        # Processing the call keyword arguments (line 339)
        int_374109 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 339, 57), 'int')
        keyword_374110 = int_374109
        kwargs_374111 = {'axis': keyword_374110}
        # Getting the type of 'mask' (line 339)
        mask_374107 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 339, 43), 'mask', False)
        # Obtaining the member 'sum' of a type (line 339)
        sum_374108 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 339, 43), mask_374107, 'sum')
        # Calling sum(args, kwargs) (line 339)
        sum_call_result_374112 = invoke(stypy.reporting.localization.Localization(__file__, 339, 43), sum_374108, *[], **kwargs_374111)
        
        # Processing the call keyword arguments (line 339)
        kwargs_374113 = {}
        # Getting the type of 'np' (line 339)
        np_374105 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 339, 33), 'np', False)
        # Obtaining the member 'cumsum' of a type (line 339)
        cumsum_374106 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 339, 33), np_374105, 'cumsum')
        # Calling cumsum(args, kwargs) (line 339)
        cumsum_call_result_374114 = invoke(stypy.reporting.localization.Localization(__file__, 339, 33), cumsum_374106, *[sum_call_result_374112], **kwargs_374113)
        
        # Getting the type of 'indptr' (line 339)
        indptr_374115 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 339, 8), 'indptr')
        int_374116 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 339, 15), 'int')
        # Getting the type of 'offset_len' (line 339)
        offset_len_374117 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 339, 17), 'offset_len')
        int_374118 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 339, 28), 'int')
        # Applying the binary operator '+' (line 339)
        result_add_374119 = python_operator(stypy.reporting.localization.Localization(__file__, 339, 17), '+', offset_len_374117, int_374118)
        
        slice_374120 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 339, 8), int_374116, result_add_374119, None)
        # Storing an element on a container (line 339)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 339, 8), indptr_374115, (slice_374120, cumsum_call_result_374114))
        
        # Assigning a Subscript to a Subscript (line 340):
        
        # Assigning a Subscript to a Subscript (line 340):
        
        # Obtaining the type of the subscript
        # Getting the type of 'offset_len' (line 340)
        offset_len_374121 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 340, 39), 'offset_len')
        # Getting the type of 'indptr' (line 340)
        indptr_374122 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 340, 32), 'indptr')
        # Obtaining the member '__getitem__' of a type (line 340)
        getitem___374123 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 340, 32), indptr_374122, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 340)
        subscript_call_result_374124 = invoke(stypy.reporting.localization.Localization(__file__, 340, 32), getitem___374123, offset_len_374121)
        
        # Getting the type of 'indptr' (line 340)
        indptr_374125 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 340, 8), 'indptr')
        # Getting the type of 'offset_len' (line 340)
        offset_len_374126 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 340, 15), 'offset_len')
        int_374127 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 340, 26), 'int')
        # Applying the binary operator '+' (line 340)
        result_add_374128 = python_operator(stypy.reporting.localization.Localization(__file__, 340, 15), '+', offset_len_374126, int_374127)
        
        slice_374129 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 340, 8), result_add_374128, None, None)
        # Storing an element on a container (line 340)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 340, 8), indptr_374125, (slice_374129, subscript_call_result_374124))
        
        # Assigning a Call to a Name (line 341):
        
        # Assigning a Call to a Name (line 341):
        
        # Call to astype(...): (line 341)
        # Processing the call arguments (line 341)
        # Getting the type of 'idx_dtype' (line 341)
        idx_dtype_374137 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 341, 39), 'idx_dtype', False)
        # Processing the call keyword arguments (line 341)
        # Getting the type of 'False' (line 341)
        False_374138 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 341, 55), 'False', False)
        keyword_374139 = False_374138
        kwargs_374140 = {'copy': keyword_374139}
        
        # Obtaining the type of the subscript
        # Getting the type of 'mask' (line 341)
        mask_374130 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 341, 24), 'mask', False)
        # Obtaining the member 'T' of a type (line 341)
        T_374131 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 341, 24), mask_374130, 'T')
        # Getting the type of 'row' (line 341)
        row_374132 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 341, 18), 'row', False)
        # Obtaining the member 'T' of a type (line 341)
        T_374133 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 341, 18), row_374132, 'T')
        # Obtaining the member '__getitem__' of a type (line 341)
        getitem___374134 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 341, 18), T_374133, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 341)
        subscript_call_result_374135 = invoke(stypy.reporting.localization.Localization(__file__, 341, 18), getitem___374134, T_374131)
        
        # Obtaining the member 'astype' of a type (line 341)
        astype_374136 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 341, 18), subscript_call_result_374135, 'astype')
        # Calling astype(args, kwargs) (line 341)
        astype_call_result_374141 = invoke(stypy.reporting.localization.Localization(__file__, 341, 18), astype_374136, *[idx_dtype_374137], **kwargs_374140)
        
        # Assigning a type to the variable 'indices' (line 341)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 341, 8), 'indices', astype_call_result_374141)
        
        # Assigning a Subscript to a Name (line 342):
        
        # Assigning a Subscript to a Name (line 342):
        
        # Obtaining the type of the subscript
        # Getting the type of 'mask' (line 342)
        mask_374142 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 342, 27), 'mask')
        # Obtaining the member 'T' of a type (line 342)
        T_374143 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 342, 27), mask_374142, 'T')
        # Getting the type of 'self' (line 342)
        self_374144 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 342, 15), 'self')
        # Obtaining the member 'data' of a type (line 342)
        data_374145 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 342, 15), self_374144, 'data')
        # Obtaining the member 'T' of a type (line 342)
        T_374146 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 342, 15), data_374145, 'T')
        # Obtaining the member '__getitem__' of a type (line 342)
        getitem___374147 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 342, 15), T_374146, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 342)
        subscript_call_result_374148 = invoke(stypy.reporting.localization.Localization(__file__, 342, 15), getitem___374147, T_374143)
        
        # Assigning a type to the variable 'data' (line 342)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 342, 8), 'data', subscript_call_result_374148)
        
        # Call to csc_matrix(...): (line 343)
        # Processing the call arguments (line 343)
        
        # Obtaining an instance of the builtin type 'tuple' (line 343)
        tuple_374150 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 343, 27), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 343)
        # Adding element type (line 343)
        # Getting the type of 'data' (line 343)
        data_374151 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 343, 27), 'data', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 343, 27), tuple_374150, data_374151)
        # Adding element type (line 343)
        # Getting the type of 'indices' (line 343)
        indices_374152 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 343, 33), 'indices', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 343, 27), tuple_374150, indices_374152)
        # Adding element type (line 343)
        # Getting the type of 'indptr' (line 343)
        indptr_374153 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 343, 42), 'indptr', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 343, 27), tuple_374150, indptr_374153)
        
        # Processing the call keyword arguments (line 343)
        # Getting the type of 'self' (line 343)
        self_374154 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 343, 57), 'self', False)
        # Obtaining the member 'shape' of a type (line 343)
        shape_374155 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 343, 57), self_374154, 'shape')
        keyword_374156 = shape_374155
        # Getting the type of 'self' (line 344)
        self_374157 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 344, 32), 'self', False)
        # Obtaining the member 'dtype' of a type (line 344)
        dtype_374158 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 344, 32), self_374157, 'dtype')
        keyword_374159 = dtype_374158
        kwargs_374160 = {'dtype': keyword_374159, 'shape': keyword_374156}
        # Getting the type of 'csc_matrix' (line 343)
        csc_matrix_374149 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 343, 15), 'csc_matrix', False)
        # Calling csc_matrix(args, kwargs) (line 343)
        csc_matrix_call_result_374161 = invoke(stypy.reporting.localization.Localization(__file__, 343, 15), csc_matrix_374149, *[tuple_374150], **kwargs_374160)
        
        # Assigning a type to the variable 'stypy_return_type' (line 343)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 343, 8), 'stypy_return_type', csc_matrix_call_result_374161)
        
        # ################# End of 'tocsc(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'tocsc' in the type store
        # Getting the type of 'stypy_return_type' (line 322)
        stypy_return_type_374162 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 322, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_374162)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'tocsc'
        return stypy_return_type_374162

    
    # Assigning a Attribute to a Attribute (line 346):

    @norecursion
    def tocoo(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'False' (line 348)
        False_374163 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 348, 25), 'False')
        defaults = [False_374163]
        # Create a new context for function 'tocoo'
        module_type_store = module_type_store.open_function_context('tocoo', 348, 4, False)
        # Assigning a type to the variable 'self' (line 349)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 349, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        dia_matrix.tocoo.__dict__.__setitem__('stypy_localization', localization)
        dia_matrix.tocoo.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        dia_matrix.tocoo.__dict__.__setitem__('stypy_type_store', module_type_store)
        dia_matrix.tocoo.__dict__.__setitem__('stypy_function_name', 'dia_matrix.tocoo')
        dia_matrix.tocoo.__dict__.__setitem__('stypy_param_names_list', ['copy'])
        dia_matrix.tocoo.__dict__.__setitem__('stypy_varargs_param_name', None)
        dia_matrix.tocoo.__dict__.__setitem__('stypy_kwargs_param_name', None)
        dia_matrix.tocoo.__dict__.__setitem__('stypy_call_defaults', defaults)
        dia_matrix.tocoo.__dict__.__setitem__('stypy_call_varargs', varargs)
        dia_matrix.tocoo.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        dia_matrix.tocoo.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'dia_matrix.tocoo', ['copy'], None, None, defaults, varargs, kwargs)

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

        
        # Assigning a Attribute to a Tuple (line 349):
        
        # Assigning a Subscript to a Name (line 349):
        
        # Obtaining the type of the subscript
        int_374164 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 349, 8), 'int')
        # Getting the type of 'self' (line 349)
        self_374165 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 349, 29), 'self')
        # Obtaining the member 'shape' of a type (line 349)
        shape_374166 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 349, 29), self_374165, 'shape')
        # Obtaining the member '__getitem__' of a type (line 349)
        getitem___374167 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 349, 8), shape_374166, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 349)
        subscript_call_result_374168 = invoke(stypy.reporting.localization.Localization(__file__, 349, 8), getitem___374167, int_374164)
        
        # Assigning a type to the variable 'tuple_var_assignment_372969' (line 349)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 349, 8), 'tuple_var_assignment_372969', subscript_call_result_374168)
        
        # Assigning a Subscript to a Name (line 349):
        
        # Obtaining the type of the subscript
        int_374169 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 349, 8), 'int')
        # Getting the type of 'self' (line 349)
        self_374170 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 349, 29), 'self')
        # Obtaining the member 'shape' of a type (line 349)
        shape_374171 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 349, 29), self_374170, 'shape')
        # Obtaining the member '__getitem__' of a type (line 349)
        getitem___374172 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 349, 8), shape_374171, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 349)
        subscript_call_result_374173 = invoke(stypy.reporting.localization.Localization(__file__, 349, 8), getitem___374172, int_374169)
        
        # Assigning a type to the variable 'tuple_var_assignment_372970' (line 349)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 349, 8), 'tuple_var_assignment_372970', subscript_call_result_374173)
        
        # Assigning a Name to a Name (line 349):
        # Getting the type of 'tuple_var_assignment_372969' (line 349)
        tuple_var_assignment_372969_374174 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 349, 8), 'tuple_var_assignment_372969')
        # Assigning a type to the variable 'num_rows' (line 349)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 349, 8), 'num_rows', tuple_var_assignment_372969_374174)
        
        # Assigning a Name to a Name (line 349):
        # Getting the type of 'tuple_var_assignment_372970' (line 349)
        tuple_var_assignment_372970_374175 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 349, 8), 'tuple_var_assignment_372970')
        # Assigning a type to the variable 'num_cols' (line 349)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 349, 18), 'num_cols', tuple_var_assignment_372970_374175)
        
        # Assigning a Attribute to a Tuple (line 350):
        
        # Assigning a Subscript to a Name (line 350):
        
        # Obtaining the type of the subscript
        int_374176 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 350, 8), 'int')
        # Getting the type of 'self' (line 350)
        self_374177 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 350, 34), 'self')
        # Obtaining the member 'data' of a type (line 350)
        data_374178 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 350, 34), self_374177, 'data')
        # Obtaining the member 'shape' of a type (line 350)
        shape_374179 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 350, 34), data_374178, 'shape')
        # Obtaining the member '__getitem__' of a type (line 350)
        getitem___374180 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 350, 8), shape_374179, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 350)
        subscript_call_result_374181 = invoke(stypy.reporting.localization.Localization(__file__, 350, 8), getitem___374180, int_374176)
        
        # Assigning a type to the variable 'tuple_var_assignment_372971' (line 350)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 350, 8), 'tuple_var_assignment_372971', subscript_call_result_374181)
        
        # Assigning a Subscript to a Name (line 350):
        
        # Obtaining the type of the subscript
        int_374182 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 350, 8), 'int')
        # Getting the type of 'self' (line 350)
        self_374183 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 350, 34), 'self')
        # Obtaining the member 'data' of a type (line 350)
        data_374184 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 350, 34), self_374183, 'data')
        # Obtaining the member 'shape' of a type (line 350)
        shape_374185 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 350, 34), data_374184, 'shape')
        # Obtaining the member '__getitem__' of a type (line 350)
        getitem___374186 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 350, 8), shape_374185, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 350)
        subscript_call_result_374187 = invoke(stypy.reporting.localization.Localization(__file__, 350, 8), getitem___374186, int_374182)
        
        # Assigning a type to the variable 'tuple_var_assignment_372972' (line 350)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 350, 8), 'tuple_var_assignment_372972', subscript_call_result_374187)
        
        # Assigning a Name to a Name (line 350):
        # Getting the type of 'tuple_var_assignment_372971' (line 350)
        tuple_var_assignment_372971_374188 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 350, 8), 'tuple_var_assignment_372971')
        # Assigning a type to the variable 'num_offsets' (line 350)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 350, 8), 'num_offsets', tuple_var_assignment_372971_374188)
        
        # Assigning a Name to a Name (line 350):
        # Getting the type of 'tuple_var_assignment_372972' (line 350)
        tuple_var_assignment_372972_374189 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 350, 8), 'tuple_var_assignment_372972')
        # Assigning a type to the variable 'offset_len' (line 350)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 350, 21), 'offset_len', tuple_var_assignment_372972_374189)
        
        # Assigning a Call to a Name (line 351):
        
        # Assigning a Call to a Name (line 351):
        
        # Call to arange(...): (line 351)
        # Processing the call arguments (line 351)
        # Getting the type of 'offset_len' (line 351)
        offset_len_374192 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 351, 32), 'offset_len', False)
        # Processing the call keyword arguments (line 351)
        kwargs_374193 = {}
        # Getting the type of 'np' (line 351)
        np_374190 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 351, 22), 'np', False)
        # Obtaining the member 'arange' of a type (line 351)
        arange_374191 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 351, 22), np_374190, 'arange')
        # Calling arange(args, kwargs) (line 351)
        arange_call_result_374194 = invoke(stypy.reporting.localization.Localization(__file__, 351, 22), arange_374191, *[offset_len_374192], **kwargs_374193)
        
        # Assigning a type to the variable 'offset_inds' (line 351)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 351, 8), 'offset_inds', arange_call_result_374194)
        
        # Assigning a BinOp to a Name (line 353):
        
        # Assigning a BinOp to a Name (line 353):
        # Getting the type of 'offset_inds' (line 353)
        offset_inds_374195 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 353, 14), 'offset_inds')
        
        # Obtaining the type of the subscript
        slice_374196 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 353, 28), None, None, None)
        # Getting the type of 'None' (line 353)
        None_374197 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 353, 43), 'None')
        # Getting the type of 'self' (line 353)
        self_374198 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 353, 28), 'self')
        # Obtaining the member 'offsets' of a type (line 353)
        offsets_374199 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 353, 28), self_374198, 'offsets')
        # Obtaining the member '__getitem__' of a type (line 353)
        getitem___374200 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 353, 28), offsets_374199, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 353)
        subscript_call_result_374201 = invoke(stypy.reporting.localization.Localization(__file__, 353, 28), getitem___374200, (slice_374196, None_374197))
        
        # Applying the binary operator '-' (line 353)
        result_sub_374202 = python_operator(stypy.reporting.localization.Localization(__file__, 353, 14), '-', offset_inds_374195, subscript_call_result_374201)
        
        # Assigning a type to the variable 'row' (line 353)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 353, 8), 'row', result_sub_374202)
        
        # Assigning a Compare to a Name (line 354):
        
        # Assigning a Compare to a Name (line 354):
        
        # Getting the type of 'row' (line 354)
        row_374203 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 354, 16), 'row')
        int_374204 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 354, 23), 'int')
        # Applying the binary operator '>=' (line 354)
        result_ge_374205 = python_operator(stypy.reporting.localization.Localization(__file__, 354, 16), '>=', row_374203, int_374204)
        
        # Assigning a type to the variable 'mask' (line 354)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 354, 8), 'mask', result_ge_374205)
        
        # Getting the type of 'mask' (line 355)
        mask_374206 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 355, 8), 'mask')
        
        # Getting the type of 'row' (line 355)
        row_374207 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 355, 17), 'row')
        # Getting the type of 'num_rows' (line 355)
        num_rows_374208 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 355, 23), 'num_rows')
        # Applying the binary operator '<' (line 355)
        result_lt_374209 = python_operator(stypy.reporting.localization.Localization(__file__, 355, 17), '<', row_374207, num_rows_374208)
        
        # Applying the binary operator '&=' (line 355)
        result_iand_374210 = python_operator(stypy.reporting.localization.Localization(__file__, 355, 8), '&=', mask_374206, result_lt_374209)
        # Assigning a type to the variable 'mask' (line 355)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 355, 8), 'mask', result_iand_374210)
        
        
        # Getting the type of 'mask' (line 356)
        mask_374211 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 356, 8), 'mask')
        
        # Getting the type of 'offset_inds' (line 356)
        offset_inds_374212 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 356, 17), 'offset_inds')
        # Getting the type of 'num_cols' (line 356)
        num_cols_374213 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 356, 31), 'num_cols')
        # Applying the binary operator '<' (line 356)
        result_lt_374214 = python_operator(stypy.reporting.localization.Localization(__file__, 356, 17), '<', offset_inds_374212, num_cols_374213)
        
        # Applying the binary operator '&=' (line 356)
        result_iand_374215 = python_operator(stypy.reporting.localization.Localization(__file__, 356, 8), '&=', mask_374211, result_lt_374214)
        # Assigning a type to the variable 'mask' (line 356)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 356, 8), 'mask', result_iand_374215)
        
        
        # Getting the type of 'mask' (line 357)
        mask_374216 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 357, 8), 'mask')
        
        # Getting the type of 'self' (line 357)
        self_374217 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 357, 17), 'self')
        # Obtaining the member 'data' of a type (line 357)
        data_374218 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 357, 17), self_374217, 'data')
        int_374219 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 357, 30), 'int')
        # Applying the binary operator '!=' (line 357)
        result_ne_374220 = python_operator(stypy.reporting.localization.Localization(__file__, 357, 17), '!=', data_374218, int_374219)
        
        # Applying the binary operator '&=' (line 357)
        result_iand_374221 = python_operator(stypy.reporting.localization.Localization(__file__, 357, 8), '&=', mask_374216, result_ne_374220)
        # Assigning a type to the variable 'mask' (line 357)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 357, 8), 'mask', result_iand_374221)
        
        
        # Assigning a Subscript to a Name (line 358):
        
        # Assigning a Subscript to a Name (line 358):
        
        # Obtaining the type of the subscript
        # Getting the type of 'mask' (line 358)
        mask_374222 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 358, 18), 'mask')
        # Getting the type of 'row' (line 358)
        row_374223 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 358, 14), 'row')
        # Obtaining the member '__getitem__' of a type (line 358)
        getitem___374224 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 358, 14), row_374223, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 358)
        subscript_call_result_374225 = invoke(stypy.reporting.localization.Localization(__file__, 358, 14), getitem___374224, mask_374222)
        
        # Assigning a type to the variable 'row' (line 358)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 358, 8), 'row', subscript_call_result_374225)
        
        # Assigning a Subscript to a Name (line 359):
        
        # Assigning a Subscript to a Name (line 359):
        
        # Obtaining the type of the subscript
        
        # Call to ravel(...): (line 359)
        # Processing the call keyword arguments (line 359)
        kwargs_374228 = {}
        # Getting the type of 'mask' (line 359)
        mask_374226 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 359, 48), 'mask', False)
        # Obtaining the member 'ravel' of a type (line 359)
        ravel_374227 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 359, 48), mask_374226, 'ravel')
        # Calling ravel(args, kwargs) (line 359)
        ravel_call_result_374229 = invoke(stypy.reporting.localization.Localization(__file__, 359, 48), ravel_374227, *[], **kwargs_374228)
        
        
        # Call to tile(...): (line 359)
        # Processing the call arguments (line 359)
        # Getting the type of 'offset_inds' (line 359)
        offset_inds_374232 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 359, 22), 'offset_inds', False)
        # Getting the type of 'num_offsets' (line 359)
        num_offsets_374233 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 359, 35), 'num_offsets', False)
        # Processing the call keyword arguments (line 359)
        kwargs_374234 = {}
        # Getting the type of 'np' (line 359)
        np_374230 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 359, 14), 'np', False)
        # Obtaining the member 'tile' of a type (line 359)
        tile_374231 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 359, 14), np_374230, 'tile')
        # Calling tile(args, kwargs) (line 359)
        tile_call_result_374235 = invoke(stypy.reporting.localization.Localization(__file__, 359, 14), tile_374231, *[offset_inds_374232, num_offsets_374233], **kwargs_374234)
        
        # Obtaining the member '__getitem__' of a type (line 359)
        getitem___374236 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 359, 14), tile_call_result_374235, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 359)
        subscript_call_result_374237 = invoke(stypy.reporting.localization.Localization(__file__, 359, 14), getitem___374236, ravel_call_result_374229)
        
        # Assigning a type to the variable 'col' (line 359)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 359, 8), 'col', subscript_call_result_374237)
        
        # Assigning a Subscript to a Name (line 360):
        
        # Assigning a Subscript to a Name (line 360):
        
        # Obtaining the type of the subscript
        # Getting the type of 'mask' (line 360)
        mask_374238 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 360, 25), 'mask')
        # Getting the type of 'self' (line 360)
        self_374239 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 360, 15), 'self')
        # Obtaining the member 'data' of a type (line 360)
        data_374240 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 360, 15), self_374239, 'data')
        # Obtaining the member '__getitem__' of a type (line 360)
        getitem___374241 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 360, 15), data_374240, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 360)
        subscript_call_result_374242 = invoke(stypy.reporting.localization.Localization(__file__, 360, 15), getitem___374241, mask_374238)
        
        # Assigning a type to the variable 'data' (line 360)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 360, 8), 'data', subscript_call_result_374242)
        stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 362, 8))
        
        # 'from scipy.sparse.coo import coo_matrix' statement (line 362)
        update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/sparse/')
        import_374243 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 362, 8), 'scipy.sparse.coo')

        if (type(import_374243) is not StypyTypeError):

            if (import_374243 != 'pyd_module'):
                __import__(import_374243)
                sys_modules_374244 = sys.modules[import_374243]
                import_from_module(stypy.reporting.localization.Localization(__file__, 362, 8), 'scipy.sparse.coo', sys_modules_374244.module_type_store, module_type_store, ['coo_matrix'])
                nest_module(stypy.reporting.localization.Localization(__file__, 362, 8), __file__, sys_modules_374244, sys_modules_374244.module_type_store, module_type_store)
            else:
                from scipy.sparse.coo import coo_matrix

                import_from_module(stypy.reporting.localization.Localization(__file__, 362, 8), 'scipy.sparse.coo', None, module_type_store, ['coo_matrix'], [coo_matrix])

        else:
            # Assigning a type to the variable 'scipy.sparse.coo' (line 362)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 362, 8), 'scipy.sparse.coo', import_374243)

        remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/sparse/')
        
        
        # Assigning a Call to a Name (line 363):
        
        # Assigning a Call to a Name (line 363):
        
        # Call to coo_matrix(...): (line 363)
        # Processing the call arguments (line 363)
        
        # Obtaining an instance of the builtin type 'tuple' (line 363)
        tuple_374246 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 363, 24), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 363)
        # Adding element type (line 363)
        # Getting the type of 'data' (line 363)
        data_374247 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 363, 24), 'data', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 363, 24), tuple_374246, data_374247)
        # Adding element type (line 363)
        
        # Obtaining an instance of the builtin type 'tuple' (line 363)
        tuple_374248 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 363, 30), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 363)
        # Adding element type (line 363)
        # Getting the type of 'row' (line 363)
        row_374249 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 363, 30), 'row', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 363, 30), tuple_374248, row_374249)
        # Adding element type (line 363)
        # Getting the type of 'col' (line 363)
        col_374250 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 363, 34), 'col', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 363, 30), tuple_374248, col_374250)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 363, 24), tuple_374246, tuple_374248)
        
        # Processing the call keyword arguments (line 363)
        # Getting the type of 'self' (line 363)
        self_374251 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 363, 47), 'self', False)
        # Obtaining the member 'shape' of a type (line 363)
        shape_374252 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 363, 47), self_374251, 'shape')
        keyword_374253 = shape_374252
        # Getting the type of 'self' (line 363)
        self_374254 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 363, 65), 'self', False)
        # Obtaining the member 'dtype' of a type (line 363)
        dtype_374255 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 363, 65), self_374254, 'dtype')
        keyword_374256 = dtype_374255
        kwargs_374257 = {'dtype': keyword_374256, 'shape': keyword_374253}
        # Getting the type of 'coo_matrix' (line 363)
        coo_matrix_374245 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 363, 12), 'coo_matrix', False)
        # Calling coo_matrix(args, kwargs) (line 363)
        coo_matrix_call_result_374258 = invoke(stypy.reporting.localization.Localization(__file__, 363, 12), coo_matrix_374245, *[tuple_374246], **kwargs_374257)
        
        # Assigning a type to the variable 'A' (line 363)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 363, 8), 'A', coo_matrix_call_result_374258)
        
        # Assigning a Name to a Attribute (line 364):
        
        # Assigning a Name to a Attribute (line 364):
        # Getting the type of 'True' (line 364)
        True_374259 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 364, 33), 'True')
        # Getting the type of 'A' (line 364)
        A_374260 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 364, 8), 'A')
        # Setting the type of the member 'has_canonical_format' of a type (line 364)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 364, 8), A_374260, 'has_canonical_format', True_374259)
        # Getting the type of 'A' (line 365)
        A_374261 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 365, 15), 'A')
        # Assigning a type to the variable 'stypy_return_type' (line 365)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 365, 8), 'stypy_return_type', A_374261)
        
        # ################# End of 'tocoo(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'tocoo' in the type store
        # Getting the type of 'stypy_return_type' (line 348)
        stypy_return_type_374262 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 348, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_374262)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'tocoo'
        return stypy_return_type_374262

    
    # Assigning a Attribute to a Attribute (line 367):

    @norecursion
    def _with_data(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'True' (line 370)
        True_374263 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 370, 36), 'True')
        defaults = [True_374263]
        # Create a new context for function '_with_data'
        module_type_store = module_type_store.open_function_context('_with_data', 370, 4, False)
        # Assigning a type to the variable 'self' (line 371)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 371, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        dia_matrix._with_data.__dict__.__setitem__('stypy_localization', localization)
        dia_matrix._with_data.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        dia_matrix._with_data.__dict__.__setitem__('stypy_type_store', module_type_store)
        dia_matrix._with_data.__dict__.__setitem__('stypy_function_name', 'dia_matrix._with_data')
        dia_matrix._with_data.__dict__.__setitem__('stypy_param_names_list', ['data', 'copy'])
        dia_matrix._with_data.__dict__.__setitem__('stypy_varargs_param_name', None)
        dia_matrix._with_data.__dict__.__setitem__('stypy_kwargs_param_name', None)
        dia_matrix._with_data.__dict__.__setitem__('stypy_call_defaults', defaults)
        dia_matrix._with_data.__dict__.__setitem__('stypy_call_varargs', varargs)
        dia_matrix._with_data.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        dia_matrix._with_data.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'dia_matrix._with_data', ['data', 'copy'], None, None, defaults, varargs, kwargs)

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

        str_374264 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 373, (-1)), 'str', 'Returns a matrix with the same sparsity structure as self,\n        but with different data.  By default the structure arrays are copied.\n        ')
        
        # Getting the type of 'copy' (line 374)
        copy_374265 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 374, 11), 'copy')
        # Testing the type of an if condition (line 374)
        if_condition_374266 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 374, 8), copy_374265)
        # Assigning a type to the variable 'if_condition_374266' (line 374)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 374, 8), 'if_condition_374266', if_condition_374266)
        # SSA begins for if statement (line 374)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to dia_matrix(...): (line 375)
        # Processing the call arguments (line 375)
        
        # Obtaining an instance of the builtin type 'tuple' (line 375)
        tuple_374268 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 375, 31), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 375)
        # Adding element type (line 375)
        # Getting the type of 'data' (line 375)
        data_374269 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 375, 31), 'data', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 375, 31), tuple_374268, data_374269)
        # Adding element type (line 375)
        
        # Call to copy(...): (line 375)
        # Processing the call keyword arguments (line 375)
        kwargs_374273 = {}
        # Getting the type of 'self' (line 375)
        self_374270 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 375, 37), 'self', False)
        # Obtaining the member 'offsets' of a type (line 375)
        offsets_374271 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 375, 37), self_374270, 'offsets')
        # Obtaining the member 'copy' of a type (line 375)
        copy_374272 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 375, 37), offsets_374271, 'copy')
        # Calling copy(args, kwargs) (line 375)
        copy_call_result_374274 = invoke(stypy.reporting.localization.Localization(__file__, 375, 37), copy_374272, *[], **kwargs_374273)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 375, 31), tuple_374268, copy_call_result_374274)
        
        # Processing the call keyword arguments (line 375)
        # Getting the type of 'self' (line 375)
        self_374275 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 375, 65), 'self', False)
        # Obtaining the member 'shape' of a type (line 375)
        shape_374276 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 375, 65), self_374275, 'shape')
        keyword_374277 = shape_374276
        kwargs_374278 = {'shape': keyword_374277}
        # Getting the type of 'dia_matrix' (line 375)
        dia_matrix_374267 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 375, 19), 'dia_matrix', False)
        # Calling dia_matrix(args, kwargs) (line 375)
        dia_matrix_call_result_374279 = invoke(stypy.reporting.localization.Localization(__file__, 375, 19), dia_matrix_374267, *[tuple_374268], **kwargs_374278)
        
        # Assigning a type to the variable 'stypy_return_type' (line 375)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 375, 12), 'stypy_return_type', dia_matrix_call_result_374279)
        # SSA branch for the else part of an if statement (line 374)
        module_type_store.open_ssa_branch('else')
        
        # Call to dia_matrix(...): (line 377)
        # Processing the call arguments (line 377)
        
        # Obtaining an instance of the builtin type 'tuple' (line 377)
        tuple_374281 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 377, 31), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 377)
        # Adding element type (line 377)
        # Getting the type of 'data' (line 377)
        data_374282 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 377, 31), 'data', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 377, 31), tuple_374281, data_374282)
        # Adding element type (line 377)
        # Getting the type of 'self' (line 377)
        self_374283 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 377, 36), 'self', False)
        # Obtaining the member 'offsets' of a type (line 377)
        offsets_374284 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 377, 36), self_374283, 'offsets')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 377, 31), tuple_374281, offsets_374284)
        
        # Processing the call keyword arguments (line 377)
        # Getting the type of 'self' (line 377)
        self_374285 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 377, 57), 'self', False)
        # Obtaining the member 'shape' of a type (line 377)
        shape_374286 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 377, 57), self_374285, 'shape')
        keyword_374287 = shape_374286
        kwargs_374288 = {'shape': keyword_374287}
        # Getting the type of 'dia_matrix' (line 377)
        dia_matrix_374280 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 377, 19), 'dia_matrix', False)
        # Calling dia_matrix(args, kwargs) (line 377)
        dia_matrix_call_result_374289 = invoke(stypy.reporting.localization.Localization(__file__, 377, 19), dia_matrix_374280, *[tuple_374281], **kwargs_374288)
        
        # Assigning a type to the variable 'stypy_return_type' (line 377)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 377, 12), 'stypy_return_type', dia_matrix_call_result_374289)
        # SSA join for if statement (line 374)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of '_with_data(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_with_data' in the type store
        # Getting the type of 'stypy_return_type' (line 370)
        stypy_return_type_374290 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 370, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_374290)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_with_data'
        return stypy_return_type_374290


# Assigning a type to the variable 'dia_matrix' (line 18)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 18, 0), 'dia_matrix', dia_matrix)

# Assigning a Str to a Name (line 76):
str_374291 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 76, 13), 'str', 'dia')
# Getting the type of 'dia_matrix'
dia_matrix_374292 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'dia_matrix')
# Setting the type of the member 'format' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), dia_matrix_374292, 'format', str_374291)

# Assigning a Attribute to a Attribute (line 183):
# Getting the type of 'spmatrix' (line 183)
spmatrix_374293 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 21), 'spmatrix')
# Obtaining the member 'getnnz' of a type (line 183)
getnnz_374294 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 183, 21), spmatrix_374293, 'getnnz')
# Obtaining the member '__doc__' of a type (line 183)
doc___374295 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 183, 21), getnnz_374294, '__doc__')
# Getting the type of 'dia_matrix'
dia_matrix_374296 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'dia_matrix')
# Obtaining the member 'getnnz' of a type
getnnz_374297 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), dia_matrix_374296, 'getnnz')
# Setting the type of the member '__doc__' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), getnnz_374297, '__doc__', doc___374295)

# Assigning a Attribute to a Attribute (line 184):
# Getting the type of 'spmatrix' (line 184)
spmatrix_374298 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 28), 'spmatrix')
# Obtaining the member 'count_nonzero' of a type (line 184)
count_nonzero_374299 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 184, 28), spmatrix_374298, 'count_nonzero')
# Obtaining the member '__doc__' of a type (line 184)
doc___374300 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 184, 28), count_nonzero_374299, '__doc__')
# Getting the type of 'dia_matrix'
dia_matrix_374301 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'dia_matrix')
# Obtaining the member 'count_nonzero' of a type
count_nonzero_374302 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), dia_matrix_374301, 'count_nonzero')
# Setting the type of the member '__doc__' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), count_nonzero_374302, '__doc__', doc___374300)

# Assigning a Attribute to a Attribute (line 227):
# Getting the type of 'spmatrix' (line 227)
spmatrix_374303 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 18), 'spmatrix')
# Obtaining the member 'sum' of a type (line 227)
sum_374304 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 227, 18), spmatrix_374303, 'sum')
# Obtaining the member '__doc__' of a type (line 227)
doc___374305 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 227, 18), sum_374304, '__doc__')
# Getting the type of 'dia_matrix'
dia_matrix_374306 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'dia_matrix')
# Obtaining the member 'sum' of a type
sum_374307 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), dia_matrix_374306, 'sum')
# Setting the type of the member '__doc__' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), sum_374307, '__doc__', doc___374305)

# Assigning a Attribute to a Attribute (line 284):
# Getting the type of 'spmatrix' (line 284)
spmatrix_374308 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 284, 20), 'spmatrix')
# Obtaining the member 'todia' of a type (line 284)
todia_374309 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 284, 20), spmatrix_374308, 'todia')
# Obtaining the member '__doc__' of a type (line 284)
doc___374310 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 284, 20), todia_374309, '__doc__')
# Getting the type of 'dia_matrix'
dia_matrix_374311 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'dia_matrix')
# Obtaining the member 'todia' of a type
todia_374312 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), dia_matrix_374311, 'todia')
# Setting the type of the member '__doc__' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), todia_374312, '__doc__', doc___374310)

# Assigning a Attribute to a Attribute (line 308):
# Getting the type of 'spmatrix' (line 308)
spmatrix_374313 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 308, 24), 'spmatrix')
# Obtaining the member 'transpose' of a type (line 308)
transpose_374314 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 308, 24), spmatrix_374313, 'transpose')
# Obtaining the member '__doc__' of a type (line 308)
doc___374315 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 308, 24), transpose_374314, '__doc__')
# Getting the type of 'dia_matrix'
dia_matrix_374316 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'dia_matrix')
# Obtaining the member 'transpose' of a type
transpose_374317 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), dia_matrix_374316, 'transpose')
# Setting the type of the member '__doc__' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), transpose_374317, '__doc__', doc___374315)

# Assigning a Attribute to a Attribute (line 320):
# Getting the type of 'spmatrix' (line 320)
spmatrix_374318 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 320, 23), 'spmatrix')
# Obtaining the member 'diagonal' of a type (line 320)
diagonal_374319 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 320, 23), spmatrix_374318, 'diagonal')
# Obtaining the member '__doc__' of a type (line 320)
doc___374320 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 320, 23), diagonal_374319, '__doc__')
# Getting the type of 'dia_matrix'
dia_matrix_374321 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'dia_matrix')
# Obtaining the member 'diagonal' of a type
diagonal_374322 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), dia_matrix_374321, 'diagonal')
# Setting the type of the member '__doc__' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), diagonal_374322, '__doc__', doc___374320)

# Assigning a Attribute to a Attribute (line 346):
# Getting the type of 'spmatrix' (line 346)
spmatrix_374323 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 346, 20), 'spmatrix')
# Obtaining the member 'tocsc' of a type (line 346)
tocsc_374324 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 346, 20), spmatrix_374323, 'tocsc')
# Obtaining the member '__doc__' of a type (line 346)
doc___374325 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 346, 20), tocsc_374324, '__doc__')
# Getting the type of 'dia_matrix'
dia_matrix_374326 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'dia_matrix')
# Obtaining the member 'tocsc' of a type
tocsc_374327 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), dia_matrix_374326, 'tocsc')
# Setting the type of the member '__doc__' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), tocsc_374327, '__doc__', doc___374325)

# Assigning a Attribute to a Attribute (line 367):
# Getting the type of 'spmatrix' (line 367)
spmatrix_374328 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 367, 20), 'spmatrix')
# Obtaining the member 'tocoo' of a type (line 367)
tocoo_374329 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 367, 20), spmatrix_374328, 'tocoo')
# Obtaining the member '__doc__' of a type (line 367)
doc___374330 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 367, 20), tocoo_374329, '__doc__')
# Getting the type of 'dia_matrix'
dia_matrix_374331 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'dia_matrix')
# Obtaining the member 'tocoo' of a type
tocoo_374332 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), dia_matrix_374331, 'tocoo')
# Setting the type of the member '__doc__' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), tocoo_374332, '__doc__', doc___374330)

@norecursion
def isspmatrix_dia(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'isspmatrix_dia'
    module_type_store = module_type_store.open_function_context('isspmatrix_dia', 380, 0, False)
    
    # Passed parameters checking function
    isspmatrix_dia.stypy_localization = localization
    isspmatrix_dia.stypy_type_of_self = None
    isspmatrix_dia.stypy_type_store = module_type_store
    isspmatrix_dia.stypy_function_name = 'isspmatrix_dia'
    isspmatrix_dia.stypy_param_names_list = ['x']
    isspmatrix_dia.stypy_varargs_param_name = None
    isspmatrix_dia.stypy_kwargs_param_name = None
    isspmatrix_dia.stypy_call_defaults = defaults
    isspmatrix_dia.stypy_call_varargs = varargs
    isspmatrix_dia.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'isspmatrix_dia', ['x'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'isspmatrix_dia', localization, ['x'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'isspmatrix_dia(...)' code ##################

    str_374333 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 402, (-1)), 'str', 'Is x of dia_matrix type?\n\n    Parameters\n    ----------\n    x\n        object to check for being a dia matrix\n\n    Returns\n    -------\n    bool\n        True if x is a dia matrix, False otherwise\n\n    Examples\n    --------\n    >>> from scipy.sparse import dia_matrix, isspmatrix_dia\n    >>> isspmatrix_dia(dia_matrix([[5]]))\n    True\n\n    >>> from scipy.sparse import dia_matrix, csr_matrix, isspmatrix_dia\n    >>> isspmatrix_dia(csr_matrix([[5]]))\n    False\n    ')
    
    # Call to isinstance(...): (line 403)
    # Processing the call arguments (line 403)
    # Getting the type of 'x' (line 403)
    x_374335 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 403, 22), 'x', False)
    # Getting the type of 'dia_matrix' (line 403)
    dia_matrix_374336 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 403, 25), 'dia_matrix', False)
    # Processing the call keyword arguments (line 403)
    kwargs_374337 = {}
    # Getting the type of 'isinstance' (line 403)
    isinstance_374334 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 403, 11), 'isinstance', False)
    # Calling isinstance(args, kwargs) (line 403)
    isinstance_call_result_374338 = invoke(stypy.reporting.localization.Localization(__file__, 403, 11), isinstance_374334, *[x_374335, dia_matrix_374336], **kwargs_374337)
    
    # Assigning a type to the variable 'stypy_return_type' (line 403)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 403, 4), 'stypy_return_type', isinstance_call_result_374338)
    
    # ################# End of 'isspmatrix_dia(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'isspmatrix_dia' in the type store
    # Getting the type of 'stypy_return_type' (line 380)
    stypy_return_type_374339 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 380, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_374339)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'isspmatrix_dia'
    return stypy_return_type_374339

# Assigning a type to the variable 'isspmatrix_dia' (line 380)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 380, 0), 'isspmatrix_dia', isspmatrix_dia)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
