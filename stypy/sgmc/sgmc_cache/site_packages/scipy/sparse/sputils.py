
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: ''' Utility functions for sparse matrix module
2: '''
3: 
4: from __future__ import division, print_function, absolute_import
5: 
6: import warnings
7: import numpy as np
8: 
9: __all__ = ['upcast', 'getdtype', 'isscalarlike', 'isintlike',
10:            'isshape', 'issequence', 'isdense', 'ismatrix', 'get_sum_dtype']
11: 
12: supported_dtypes = ['bool', 'int8', 'uint8', 'short', 'ushort', 'intc',
13:                     'uintc', 'longlong', 'ulonglong', 'single', 'double',
14:                     'longdouble', 'csingle', 'cdouble', 'clongdouble']
15: supported_dtypes = [np.typeDict[x] for x in supported_dtypes]
16: 
17: _upcast_memo = {}
18: 
19: 
20: def upcast(*args):
21:     '''Returns the nearest supported sparse dtype for the
22:     combination of one or more types.
23: 
24:     upcast(t0, t1, ..., tn) -> T  where T is a supported dtype
25: 
26:     Examples
27:     --------
28: 
29:     >>> upcast('int32')
30:     <type 'numpy.int32'>
31:     >>> upcast('bool')
32:     <type 'numpy.bool_'>
33:     >>> upcast('int32','float32')
34:     <type 'numpy.float64'>
35:     >>> upcast('bool',complex,float)
36:     <type 'numpy.complex128'>
37: 
38:     '''
39: 
40:     t = _upcast_memo.get(hash(args))
41:     if t is not None:
42:         return t
43: 
44:     upcast = np.find_common_type(args, [])
45: 
46:     for t in supported_dtypes:
47:         if np.can_cast(upcast, t):
48:             _upcast_memo[hash(args)] = t
49:             return t
50: 
51:     raise TypeError('no supported conversion for types: %r' % (args,))
52: 
53: 
54: def upcast_char(*args):
55:     '''Same as `upcast` but taking dtype.char as input (faster).'''
56:     t = _upcast_memo.get(args)
57:     if t is not None:
58:         return t
59:     t = upcast(*map(np.dtype, args))
60:     _upcast_memo[args] = t
61:     return t
62: 
63: 
64: def upcast_scalar(dtype, scalar):
65:     '''Determine data type for binary operation between an array of
66:     type `dtype` and a scalar.
67:     '''
68:     return (np.array([0], dtype=dtype) * scalar).dtype
69: 
70: 
71: def downcast_intp_index(arr):
72:     '''
73:     Down-cast index array to np.intp dtype if it is of a larger dtype.
74: 
75:     Raise an error if the array contains a value that is too large for
76:     intp.
77:     '''
78:     if arr.dtype.itemsize > np.dtype(np.intp).itemsize:
79:         if arr.size == 0:
80:             return arr.astype(np.intp)
81:         maxval = arr.max()
82:         minval = arr.min()
83:         if maxval > np.iinfo(np.intp).max or minval < np.iinfo(np.intp).min:
84:             raise ValueError("Cannot deal with arrays with indices larger "
85:                              "than the machine maximum address size "
86:                              "(e.g. 64-bit indices on 32-bit machine).")
87:         return arr.astype(np.intp)
88:     return arr
89: 
90: 
91: def to_native(A):
92:     return np.asarray(A, dtype=A.dtype.newbyteorder('native'))
93: 
94: 
95: def getdtype(dtype, a=None, default=None):
96:     '''Function used to simplify argument processing.  If 'dtype' is not
97:     specified (is None), returns a.dtype; otherwise returns a np.dtype
98:     object created from the specified dtype argument.  If 'dtype' and 'a'
99:     are both None, construct a data type out of the 'default' parameter.
100:     Furthermore, 'dtype' must be in 'allowed' set.
101:     '''
102:     # TODO is this really what we want?
103:     if dtype is None:
104:         try:
105:             newdtype = a.dtype
106:         except AttributeError:
107:             if default is not None:
108:                 newdtype = np.dtype(default)
109:             else:
110:                 raise TypeError("could not interpret data type")
111:     else:
112:         newdtype = np.dtype(dtype)
113:         if newdtype == np.object_:
114:             warnings.warn("object dtype is not supported by sparse matrices")
115: 
116:     return newdtype
117: 
118: 
119: def get_index_dtype(arrays=(), maxval=None, check_contents=False):
120:     '''
121:     Based on input (integer) arrays `a`, determine a suitable index data
122:     type that can hold the data in the arrays.
123: 
124:     Parameters
125:     ----------
126:     arrays : tuple of array_like
127:         Input arrays whose types/contents to check
128:     maxval : float, optional
129:         Maximum value needed
130:     check_contents : bool, optional
131:         Whether to check the values in the arrays and not just their types.
132:         Default: False (check only the types)
133: 
134:     Returns
135:     -------
136:     dtype : dtype
137:         Suitable index data type (int32 or int64)
138: 
139:     '''
140: 
141:     int32max = np.iinfo(np.int32).max
142: 
143:     dtype = np.intc
144:     if maxval is not None:
145:         if maxval > int32max:
146:             dtype = np.int64
147: 
148:     if isinstance(arrays, np.ndarray):
149:         arrays = (arrays,)
150: 
151:     for arr in arrays:
152:         arr = np.asarray(arr)
153:         if arr.dtype > np.int32:
154:             if check_contents:
155:                 if arr.size == 0:
156:                     # a bigger type not needed
157:                     continue
158:                 elif np.issubdtype(arr.dtype, np.integer):
159:                     maxval = arr.max()
160:                     minval = arr.min()
161:                     if (minval >= np.iinfo(np.int32).min and
162:                             maxval <= np.iinfo(np.int32).max):
163:                         # a bigger type not needed
164:                         continue
165: 
166:             dtype = np.int64
167:             break
168: 
169:     return dtype
170: 
171: 
172: def get_sum_dtype(dtype):
173:     '''Mimic numpy's casting for np.sum'''
174:     if np.issubdtype(dtype, np.float_):
175:         return np.float_
176:     if dtype.kind == 'u' and np.can_cast(dtype, np.uint):
177:         return np.uint
178:     if np.can_cast(dtype, np.int_):
179:         return np.int_
180:     return dtype
181: 
182: 
183: def isscalarlike(x):
184:     '''Is x either a scalar, an array scalar, or a 0-dim array?'''
185:     return np.isscalar(x) or (isdense(x) and x.ndim == 0)
186: 
187: 
188: def isintlike(x):
189:     '''Is x appropriate as an index into a sparse matrix? Returns True
190:     if it can be cast safely to a machine int.
191:     '''
192:     if not isscalarlike(x):
193:         return False
194:     try:
195:         return bool(int(x) == x)
196:     except (TypeError, ValueError):
197:         return False
198: 
199: 
200: def isshape(x):
201:     '''Is x a valid 2-tuple of dimensions?
202:     '''
203:     try:
204:         # Assume it's a tuple of matrix dimensions (M, N)
205:         (M, N) = x
206:     except:
207:         return False
208:     else:
209:         if isintlike(M) and isintlike(N):
210:             if np.ndim(M) == 0 and np.ndim(N) == 0:
211:                 return True
212:         return False
213: 
214: 
215: def issequence(t):
216:     return ((isinstance(t, (list, tuple)) and
217:             (len(t) == 0 or np.isscalar(t[0]))) or
218:             (isinstance(t, np.ndarray) and (t.ndim == 1)))
219: 
220: 
221: def ismatrix(t):
222:     return ((isinstance(t, (list, tuple)) and
223:              len(t) > 0 and issequence(t[0])) or
224:             (isinstance(t, np.ndarray) and t.ndim == 2))
225: 
226: 
227: def isdense(x):
228:     return isinstance(x, np.ndarray)
229: 
230: 
231: def validateaxis(axis):
232:     if axis is not None:
233:         axis_type = type(axis)
234: 
235:         # In NumPy, you can pass in tuples for 'axis', but they are
236:         # not very useful for sparse matrices given their limited
237:         # dimensions, so let's make it explicit that they are not
238:         # allowed to be passed in
239:         if axis_type == tuple:
240:             raise TypeError(("Tuples are not accepted for the 'axis' "
241:                              "parameter. Please pass in one of the "
242:                              "following: {-2, -1, 0, 1, None}."))
243: 
244:         # If not a tuple, check that the provided axis is actually
245:         # an integer and raise a TypeError similar to NumPy's
246:         if not np.issubdtype(np.dtype(axis_type), np.integer):
247:             raise TypeError("axis must be an integer, not {name}"
248:                             .format(name=axis_type.__name__))
249: 
250:         if not (-2 <= axis <= 1):
251:             raise ValueError("axis out of range")
252: 
253: 
254: class IndexMixin(object):
255:     '''
256:     This class simply exists to hold the methods necessary for fancy indexing.
257:     '''
258:     def _slicetoarange(self, j, shape):
259:         ''' Given a slice object, use numpy arange to change it to a 1D
260:         array.
261:         '''
262:         start, stop, step = j.indices(shape)
263:         return np.arange(start, stop, step)
264: 
265:     def _unpack_index(self, index):
266:         ''' Parse index. Always return a tuple of the form (row, col).
267:         Where row/col is a integer, slice, or array of integers.
268:         '''
269:         # First, check if indexing with single boolean matrix.
270:         from .base import spmatrix  # This feels dirty but...
271:         if (isinstance(index, (spmatrix, np.ndarray)) and
272:            (index.ndim == 2) and index.dtype.kind == 'b'):
273:                 return index.nonzero()
274: 
275:         # Parse any ellipses.
276:         index = self._check_ellipsis(index)
277: 
278:         # Next, parse the tuple or object
279:         if isinstance(index, tuple):
280:             if len(index) == 2:
281:                 row, col = index
282:             elif len(index) == 1:
283:                 row, col = index[0], slice(None)
284:             else:
285:                 raise IndexError('invalid number of indices')
286:         else:
287:             row, col = index, slice(None)
288: 
289:         # Next, check for validity, or transform the index as needed.
290:         row, col = self._check_boolean(row, col)
291:         return row, col
292: 
293:     def _check_ellipsis(self, index):
294:         '''Process indices with Ellipsis. Returns modified index.'''
295:         if index is Ellipsis:
296:             return (slice(None), slice(None))
297:         elif isinstance(index, tuple):
298:             # Find first ellipsis
299:             for j, v in enumerate(index):
300:                 if v is Ellipsis:
301:                     first_ellipsis = j
302:                     break
303:             else:
304:                 first_ellipsis = None
305: 
306:             # Expand the first one
307:             if first_ellipsis is not None:
308:                 # Shortcuts
309:                 if len(index) == 1:
310:                     return (slice(None), slice(None))
311:                 elif len(index) == 2:
312:                     if first_ellipsis == 0:
313:                         if index[1] is Ellipsis:
314:                             return (slice(None), slice(None))
315:                         else:
316:                             return (slice(None), index[1])
317:                     else:
318:                         return (index[0], slice(None))
319: 
320:                 # General case
321:                 tail = ()
322:                 for v in index[first_ellipsis+1:]:
323:                     if v is not Ellipsis:
324:                         tail = tail + (v,)
325:                 nd = first_ellipsis + len(tail)
326:                 nslice = max(0, 2 - nd)
327:                 return index[:first_ellipsis] + (slice(None),)*nslice + tail
328: 
329:         return index
330: 
331:     def _check_boolean(self, row, col):
332:         from .base import isspmatrix  # ew...
333:         # Supporting sparse boolean indexing with both row and col does
334:         # not work because spmatrix.ndim is always 2.
335:         if isspmatrix(row) or isspmatrix(col):
336:             raise IndexError(
337:                 "Indexing with sparse matrices is not supported "
338:                 "except boolean indexing where matrix and index "
339:                 "are equal shapes.")
340:         if isinstance(row, np.ndarray) and row.dtype.kind == 'b':
341:             row = self._boolean_index_to_array(row)
342:         if isinstance(col, np.ndarray) and col.dtype.kind == 'b':
343:             col = self._boolean_index_to_array(col)
344:         return row, col
345: 
346:     def _boolean_index_to_array(self, i):
347:         if i.ndim > 1:
348:             raise IndexError('invalid index shape')
349:         return i.nonzero()[0]
350: 
351:     def _index_to_arrays(self, i, j):
352:         i, j = self._check_boolean(i, j)
353: 
354:         i_slice = isinstance(i, slice)
355:         if i_slice:
356:             i = self._slicetoarange(i, self.shape[0])[:, None]
357:         else:
358:             i = np.atleast_1d(i)
359: 
360:         if isinstance(j, slice):
361:             j = self._slicetoarange(j, self.shape[1])[None, :]
362:             if i.ndim == 1:
363:                 i = i[:, None]
364:             elif not i_slice:
365:                 raise IndexError('index returns 3-dim structure')
366:         elif isscalarlike(j):
367:             # row vector special case
368:             j = np.atleast_1d(j)
369:             if i.ndim == 1:
370:                 i, j = np.broadcast_arrays(i, j)
371:                 i = i[:, None]
372:                 j = j[:, None]
373:                 return i, j
374:         else:
375:             j = np.atleast_1d(j)
376:             if i_slice and j.ndim > 1:
377:                 raise IndexError('index returns 3-dim structure')
378: 
379:         i, j = np.broadcast_arrays(i, j)
380: 
381:         if i.ndim == 1:
382:             # return column vectors for 1-D indexing
383:             i = i[None, :]
384:             j = j[None, :]
385:         elif i.ndim > 2:
386:             raise IndexError("Index dimension must be <= 2")
387: 
388:         return i, j
389: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_379683 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2, (-1)), 'str', ' Utility functions for sparse matrix module\n')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 6, 0))

# 'import warnings' statement (line 6)
import warnings

import_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'warnings', warnings, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 7, 0))

# 'import numpy' statement (line 7)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/sparse/')
import_379684 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'numpy')

if (type(import_379684) is not StypyTypeError):

    if (import_379684 != 'pyd_module'):
        __import__(import_379684)
        sys_modules_379685 = sys.modules[import_379684]
        import_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'np', sys_modules_379685.module_type_store, module_type_store)
    else:
        import numpy as np

        import_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'np', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 7)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), 'numpy', import_379684)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/sparse/')


# Assigning a List to a Name (line 9):

# Assigning a List to a Name (line 9):
__all__ = ['upcast', 'getdtype', 'isscalarlike', 'isintlike', 'isshape', 'issequence', 'isdense', 'ismatrix', 'get_sum_dtype']
module_type_store.set_exportable_members(['upcast', 'getdtype', 'isscalarlike', 'isintlike', 'isshape', 'issequence', 'isdense', 'ismatrix', 'get_sum_dtype'])

# Obtaining an instance of the builtin type 'list' (line 9)
list_379686 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 9, 10), 'list')
# Adding type elements to the builtin type 'list' instance (line 9)
# Adding element type (line 9)
str_379687 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 9, 11), 'str', 'upcast')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 9, 10), list_379686, str_379687)
# Adding element type (line 9)
str_379688 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 9, 21), 'str', 'getdtype')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 9, 10), list_379686, str_379688)
# Adding element type (line 9)
str_379689 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 9, 33), 'str', 'isscalarlike')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 9, 10), list_379686, str_379689)
# Adding element type (line 9)
str_379690 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 9, 49), 'str', 'isintlike')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 9, 10), list_379686, str_379690)
# Adding element type (line 9)
str_379691 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, 11), 'str', 'isshape')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 9, 10), list_379686, str_379691)
# Adding element type (line 9)
str_379692 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, 22), 'str', 'issequence')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 9, 10), list_379686, str_379692)
# Adding element type (line 9)
str_379693 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, 36), 'str', 'isdense')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 9, 10), list_379686, str_379693)
# Adding element type (line 9)
str_379694 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, 47), 'str', 'ismatrix')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 9, 10), list_379686, str_379694)
# Adding element type (line 9)
str_379695 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, 59), 'str', 'get_sum_dtype')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 9, 10), list_379686, str_379695)

# Assigning a type to the variable '__all__' (line 9)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 0), '__all__', list_379686)

# Assigning a List to a Name (line 12):

# Assigning a List to a Name (line 12):

# Obtaining an instance of the builtin type 'list' (line 12)
list_379696 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 19), 'list')
# Adding type elements to the builtin type 'list' instance (line 12)
# Adding element type (line 12)
str_379697 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 20), 'str', 'bool')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 12, 19), list_379696, str_379697)
# Adding element type (line 12)
str_379698 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 28), 'str', 'int8')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 12, 19), list_379696, str_379698)
# Adding element type (line 12)
str_379699 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 36), 'str', 'uint8')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 12, 19), list_379696, str_379699)
# Adding element type (line 12)
str_379700 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 45), 'str', 'short')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 12, 19), list_379696, str_379700)
# Adding element type (line 12)
str_379701 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 54), 'str', 'ushort')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 12, 19), list_379696, str_379701)
# Adding element type (line 12)
str_379702 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 64), 'str', 'intc')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 12, 19), list_379696, str_379702)
# Adding element type (line 12)
str_379703 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 20), 'str', 'uintc')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 12, 19), list_379696, str_379703)
# Adding element type (line 12)
str_379704 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 29), 'str', 'longlong')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 12, 19), list_379696, str_379704)
# Adding element type (line 12)
str_379705 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 41), 'str', 'ulonglong')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 12, 19), list_379696, str_379705)
# Adding element type (line 12)
str_379706 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 54), 'str', 'single')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 12, 19), list_379696, str_379706)
# Adding element type (line 12)
str_379707 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 64), 'str', 'double')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 12, 19), list_379696, str_379707)
# Adding element type (line 12)
str_379708 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 20), 'str', 'longdouble')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 12, 19), list_379696, str_379708)
# Adding element type (line 12)
str_379709 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 34), 'str', 'csingle')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 12, 19), list_379696, str_379709)
# Adding element type (line 12)
str_379710 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 45), 'str', 'cdouble')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 12, 19), list_379696, str_379710)
# Adding element type (line 12)
str_379711 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 56), 'str', 'clongdouble')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 12, 19), list_379696, str_379711)

# Assigning a type to the variable 'supported_dtypes' (line 12)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 0), 'supported_dtypes', list_379696)

# Assigning a ListComp to a Name (line 15):

# Assigning a ListComp to a Name (line 15):
# Calculating list comprehension
# Calculating comprehension expression
# Getting the type of 'supported_dtypes' (line 15)
supported_dtypes_379717 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 44), 'supported_dtypes')
comprehension_379718 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 15, 20), supported_dtypes_379717)
# Assigning a type to the variable 'x' (line 15)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 20), 'x', comprehension_379718)

# Obtaining the type of the subscript
# Getting the type of 'x' (line 15)
x_379712 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 32), 'x')
# Getting the type of 'np' (line 15)
np_379713 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 20), 'np')
# Obtaining the member 'typeDict' of a type (line 15)
typeDict_379714 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 15, 20), np_379713, 'typeDict')
# Obtaining the member '__getitem__' of a type (line 15)
getitem___379715 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 15, 20), typeDict_379714, '__getitem__')
# Calling the subscript (__getitem__) to obtain the elements type (line 15)
subscript_call_result_379716 = invoke(stypy.reporting.localization.Localization(__file__, 15, 20), getitem___379715, x_379712)

list_379719 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 20), 'list')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 15, 20), list_379719, subscript_call_result_379716)
# Assigning a type to the variable 'supported_dtypes' (line 15)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 0), 'supported_dtypes', list_379719)

# Assigning a Dict to a Name (line 17):

# Assigning a Dict to a Name (line 17):

# Obtaining an instance of the builtin type 'dict' (line 17)
dict_379720 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 15), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 17)

# Assigning a type to the variable '_upcast_memo' (line 17)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 17, 0), '_upcast_memo', dict_379720)

@norecursion
def upcast(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'upcast'
    module_type_store = module_type_store.open_function_context('upcast', 20, 0, False)
    
    # Passed parameters checking function
    upcast.stypy_localization = localization
    upcast.stypy_type_of_self = None
    upcast.stypy_type_store = module_type_store
    upcast.stypy_function_name = 'upcast'
    upcast.stypy_param_names_list = []
    upcast.stypy_varargs_param_name = 'args'
    upcast.stypy_kwargs_param_name = None
    upcast.stypy_call_defaults = defaults
    upcast.stypy_call_varargs = varargs
    upcast.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'upcast', [], 'args', None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'upcast', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'upcast(...)' code ##################

    str_379721 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, (-1)), 'str', "Returns the nearest supported sparse dtype for the\n    combination of one or more types.\n\n    upcast(t0, t1, ..., tn) -> T  where T is a supported dtype\n\n    Examples\n    --------\n\n    >>> upcast('int32')\n    <type 'numpy.int32'>\n    >>> upcast('bool')\n    <type 'numpy.bool_'>\n    >>> upcast('int32','float32')\n    <type 'numpy.float64'>\n    >>> upcast('bool',complex,float)\n    <type 'numpy.complex128'>\n\n    ")
    
    # Assigning a Call to a Name (line 40):
    
    # Assigning a Call to a Name (line 40):
    
    # Call to get(...): (line 40)
    # Processing the call arguments (line 40)
    
    # Call to hash(...): (line 40)
    # Processing the call arguments (line 40)
    # Getting the type of 'args' (line 40)
    args_379725 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 30), 'args', False)
    # Processing the call keyword arguments (line 40)
    kwargs_379726 = {}
    # Getting the type of 'hash' (line 40)
    hash_379724 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 25), 'hash', False)
    # Calling hash(args, kwargs) (line 40)
    hash_call_result_379727 = invoke(stypy.reporting.localization.Localization(__file__, 40, 25), hash_379724, *[args_379725], **kwargs_379726)
    
    # Processing the call keyword arguments (line 40)
    kwargs_379728 = {}
    # Getting the type of '_upcast_memo' (line 40)
    _upcast_memo_379722 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 8), '_upcast_memo', False)
    # Obtaining the member 'get' of a type (line 40)
    get_379723 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 40, 8), _upcast_memo_379722, 'get')
    # Calling get(args, kwargs) (line 40)
    get_call_result_379729 = invoke(stypy.reporting.localization.Localization(__file__, 40, 8), get_379723, *[hash_call_result_379727], **kwargs_379728)
    
    # Assigning a type to the variable 't' (line 40)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 40, 4), 't', get_call_result_379729)
    
    # Type idiom detected: calculating its left and rigth part (line 41)
    # Getting the type of 't' (line 41)
    t_379730 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 4), 't')
    # Getting the type of 'None' (line 41)
    None_379731 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 16), 'None')
    
    (may_be_379732, more_types_in_union_379733) = may_not_be_none(t_379730, None_379731)

    if may_be_379732:

        if more_types_in_union_379733:
            # Runtime conditional SSA (line 41)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        # Getting the type of 't' (line 42)
        t_379734 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 15), 't')
        # Assigning a type to the variable 'stypy_return_type' (line 42)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 42, 8), 'stypy_return_type', t_379734)

        if more_types_in_union_379733:
            # SSA join for if statement (line 41)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Assigning a Call to a Name (line 44):
    
    # Assigning a Call to a Name (line 44):
    
    # Call to find_common_type(...): (line 44)
    # Processing the call arguments (line 44)
    # Getting the type of 'args' (line 44)
    args_379737 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 33), 'args', False)
    
    # Obtaining an instance of the builtin type 'list' (line 44)
    list_379738 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 39), 'list')
    # Adding type elements to the builtin type 'list' instance (line 44)
    
    # Processing the call keyword arguments (line 44)
    kwargs_379739 = {}
    # Getting the type of 'np' (line 44)
    np_379735 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 13), 'np', False)
    # Obtaining the member 'find_common_type' of a type (line 44)
    find_common_type_379736 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 44, 13), np_379735, 'find_common_type')
    # Calling find_common_type(args, kwargs) (line 44)
    find_common_type_call_result_379740 = invoke(stypy.reporting.localization.Localization(__file__, 44, 13), find_common_type_379736, *[args_379737, list_379738], **kwargs_379739)
    
    # Assigning a type to the variable 'upcast' (line 44)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 44, 4), 'upcast', find_common_type_call_result_379740)
    
    # Getting the type of 'supported_dtypes' (line 46)
    supported_dtypes_379741 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 13), 'supported_dtypes')
    # Testing the type of a for loop iterable (line 46)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 46, 4), supported_dtypes_379741)
    # Getting the type of the for loop variable (line 46)
    for_loop_var_379742 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 46, 4), supported_dtypes_379741)
    # Assigning a type to the variable 't' (line 46)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 46, 4), 't', for_loop_var_379742)
    # SSA begins for a for statement (line 46)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    
    # Call to can_cast(...): (line 47)
    # Processing the call arguments (line 47)
    # Getting the type of 'upcast' (line 47)
    upcast_379745 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 23), 'upcast', False)
    # Getting the type of 't' (line 47)
    t_379746 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 31), 't', False)
    # Processing the call keyword arguments (line 47)
    kwargs_379747 = {}
    # Getting the type of 'np' (line 47)
    np_379743 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 11), 'np', False)
    # Obtaining the member 'can_cast' of a type (line 47)
    can_cast_379744 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 47, 11), np_379743, 'can_cast')
    # Calling can_cast(args, kwargs) (line 47)
    can_cast_call_result_379748 = invoke(stypy.reporting.localization.Localization(__file__, 47, 11), can_cast_379744, *[upcast_379745, t_379746], **kwargs_379747)
    
    # Testing the type of an if condition (line 47)
    if_condition_379749 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 47, 8), can_cast_call_result_379748)
    # Assigning a type to the variable 'if_condition_379749' (line 47)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 47, 8), 'if_condition_379749', if_condition_379749)
    # SSA begins for if statement (line 47)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Name to a Subscript (line 48):
    
    # Assigning a Name to a Subscript (line 48):
    # Getting the type of 't' (line 48)
    t_379750 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 39), 't')
    # Getting the type of '_upcast_memo' (line 48)
    _upcast_memo_379751 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 12), '_upcast_memo')
    
    # Call to hash(...): (line 48)
    # Processing the call arguments (line 48)
    # Getting the type of 'args' (line 48)
    args_379753 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 30), 'args', False)
    # Processing the call keyword arguments (line 48)
    kwargs_379754 = {}
    # Getting the type of 'hash' (line 48)
    hash_379752 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 25), 'hash', False)
    # Calling hash(args, kwargs) (line 48)
    hash_call_result_379755 = invoke(stypy.reporting.localization.Localization(__file__, 48, 25), hash_379752, *[args_379753], **kwargs_379754)
    
    # Storing an element on a container (line 48)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 48, 12), _upcast_memo_379751, (hash_call_result_379755, t_379750))
    # Getting the type of 't' (line 49)
    t_379756 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 19), 't')
    # Assigning a type to the variable 'stypy_return_type' (line 49)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 49, 12), 'stypy_return_type', t_379756)
    # SSA join for if statement (line 47)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to TypeError(...): (line 51)
    # Processing the call arguments (line 51)
    str_379758 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 51, 20), 'str', 'no supported conversion for types: %r')
    
    # Obtaining an instance of the builtin type 'tuple' (line 51)
    tuple_379759 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 51, 63), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 51)
    # Adding element type (line 51)
    # Getting the type of 'args' (line 51)
    args_379760 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 63), 'args', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 51, 63), tuple_379759, args_379760)
    
    # Applying the binary operator '%' (line 51)
    result_mod_379761 = python_operator(stypy.reporting.localization.Localization(__file__, 51, 20), '%', str_379758, tuple_379759)
    
    # Processing the call keyword arguments (line 51)
    kwargs_379762 = {}
    # Getting the type of 'TypeError' (line 51)
    TypeError_379757 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 10), 'TypeError', False)
    # Calling TypeError(args, kwargs) (line 51)
    TypeError_call_result_379763 = invoke(stypy.reporting.localization.Localization(__file__, 51, 10), TypeError_379757, *[result_mod_379761], **kwargs_379762)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 51, 4), TypeError_call_result_379763, 'raise parameter', BaseException)
    
    # ################# End of 'upcast(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'upcast' in the type store
    # Getting the type of 'stypy_return_type' (line 20)
    stypy_return_type_379764 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_379764)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'upcast'
    return stypy_return_type_379764

# Assigning a type to the variable 'upcast' (line 20)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 20, 0), 'upcast', upcast)

@norecursion
def upcast_char(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'upcast_char'
    module_type_store = module_type_store.open_function_context('upcast_char', 54, 0, False)
    
    # Passed parameters checking function
    upcast_char.stypy_localization = localization
    upcast_char.stypy_type_of_self = None
    upcast_char.stypy_type_store = module_type_store
    upcast_char.stypy_function_name = 'upcast_char'
    upcast_char.stypy_param_names_list = []
    upcast_char.stypy_varargs_param_name = 'args'
    upcast_char.stypy_kwargs_param_name = None
    upcast_char.stypy_call_defaults = defaults
    upcast_char.stypy_call_varargs = varargs
    upcast_char.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'upcast_char', [], 'args', None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'upcast_char', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'upcast_char(...)' code ##################

    str_379765 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, 4), 'str', 'Same as `upcast` but taking dtype.char as input (faster).')
    
    # Assigning a Call to a Name (line 56):
    
    # Assigning a Call to a Name (line 56):
    
    # Call to get(...): (line 56)
    # Processing the call arguments (line 56)
    # Getting the type of 'args' (line 56)
    args_379768 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 25), 'args', False)
    # Processing the call keyword arguments (line 56)
    kwargs_379769 = {}
    # Getting the type of '_upcast_memo' (line 56)
    _upcast_memo_379766 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 8), '_upcast_memo', False)
    # Obtaining the member 'get' of a type (line 56)
    get_379767 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 56, 8), _upcast_memo_379766, 'get')
    # Calling get(args, kwargs) (line 56)
    get_call_result_379770 = invoke(stypy.reporting.localization.Localization(__file__, 56, 8), get_379767, *[args_379768], **kwargs_379769)
    
    # Assigning a type to the variable 't' (line 56)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 56, 4), 't', get_call_result_379770)
    
    # Type idiom detected: calculating its left and rigth part (line 57)
    # Getting the type of 't' (line 57)
    t_379771 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 4), 't')
    # Getting the type of 'None' (line 57)
    None_379772 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 16), 'None')
    
    (may_be_379773, more_types_in_union_379774) = may_not_be_none(t_379771, None_379772)

    if may_be_379773:

        if more_types_in_union_379774:
            # Runtime conditional SSA (line 57)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        # Getting the type of 't' (line 58)
        t_379775 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 15), 't')
        # Assigning a type to the variable 'stypy_return_type' (line 58)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 58, 8), 'stypy_return_type', t_379775)

        if more_types_in_union_379774:
            # SSA join for if statement (line 57)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Assigning a Call to a Name (line 59):
    
    # Assigning a Call to a Name (line 59):
    
    # Call to upcast(...): (line 59)
    
    # Call to map(...): (line 59)
    # Processing the call arguments (line 59)
    # Getting the type of 'np' (line 59)
    np_379778 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 20), 'np', False)
    # Obtaining the member 'dtype' of a type (line 59)
    dtype_379779 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 59, 20), np_379778, 'dtype')
    # Getting the type of 'args' (line 59)
    args_379780 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 30), 'args', False)
    # Processing the call keyword arguments (line 59)
    kwargs_379781 = {}
    # Getting the type of 'map' (line 59)
    map_379777 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 16), 'map', False)
    # Calling map(args, kwargs) (line 59)
    map_call_result_379782 = invoke(stypy.reporting.localization.Localization(__file__, 59, 16), map_379777, *[dtype_379779, args_379780], **kwargs_379781)
    
    # Processing the call keyword arguments (line 59)
    kwargs_379783 = {}
    # Getting the type of 'upcast' (line 59)
    upcast_379776 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 8), 'upcast', False)
    # Calling upcast(args, kwargs) (line 59)
    upcast_call_result_379784 = invoke(stypy.reporting.localization.Localization(__file__, 59, 8), upcast_379776, *[map_call_result_379782], **kwargs_379783)
    
    # Assigning a type to the variable 't' (line 59)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 59, 4), 't', upcast_call_result_379784)
    
    # Assigning a Name to a Subscript (line 60):
    
    # Assigning a Name to a Subscript (line 60):
    # Getting the type of 't' (line 60)
    t_379785 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 25), 't')
    # Getting the type of '_upcast_memo' (line 60)
    _upcast_memo_379786 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 4), '_upcast_memo')
    # Getting the type of 'args' (line 60)
    args_379787 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 17), 'args')
    # Storing an element on a container (line 60)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 60, 4), _upcast_memo_379786, (args_379787, t_379785))
    # Getting the type of 't' (line 61)
    t_379788 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 11), 't')
    # Assigning a type to the variable 'stypy_return_type' (line 61)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 61, 4), 'stypy_return_type', t_379788)
    
    # ################# End of 'upcast_char(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'upcast_char' in the type store
    # Getting the type of 'stypy_return_type' (line 54)
    stypy_return_type_379789 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_379789)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'upcast_char'
    return stypy_return_type_379789

# Assigning a type to the variable 'upcast_char' (line 54)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 54, 0), 'upcast_char', upcast_char)

@norecursion
def upcast_scalar(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'upcast_scalar'
    module_type_store = module_type_store.open_function_context('upcast_scalar', 64, 0, False)
    
    # Passed parameters checking function
    upcast_scalar.stypy_localization = localization
    upcast_scalar.stypy_type_of_self = None
    upcast_scalar.stypy_type_store = module_type_store
    upcast_scalar.stypy_function_name = 'upcast_scalar'
    upcast_scalar.stypy_param_names_list = ['dtype', 'scalar']
    upcast_scalar.stypy_varargs_param_name = None
    upcast_scalar.stypy_kwargs_param_name = None
    upcast_scalar.stypy_call_defaults = defaults
    upcast_scalar.stypy_call_varargs = varargs
    upcast_scalar.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'upcast_scalar', ['dtype', 'scalar'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'upcast_scalar', localization, ['dtype', 'scalar'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'upcast_scalar(...)' code ##################

    str_379790 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 67, (-1)), 'str', 'Determine data type for binary operation between an array of\n    type `dtype` and a scalar.\n    ')
    
    # Call to array(...): (line 68)
    # Processing the call arguments (line 68)
    
    # Obtaining an instance of the builtin type 'list' (line 68)
    list_379793 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 21), 'list')
    # Adding type elements to the builtin type 'list' instance (line 68)
    # Adding element type (line 68)
    int_379794 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 22), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 68, 21), list_379793, int_379794)
    
    # Processing the call keyword arguments (line 68)
    # Getting the type of 'dtype' (line 68)
    dtype_379795 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 32), 'dtype', False)
    keyword_379796 = dtype_379795
    kwargs_379797 = {'dtype': keyword_379796}
    # Getting the type of 'np' (line 68)
    np_379791 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 12), 'np', False)
    # Obtaining the member 'array' of a type (line 68)
    array_379792 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 68, 12), np_379791, 'array')
    # Calling array(args, kwargs) (line 68)
    array_call_result_379798 = invoke(stypy.reporting.localization.Localization(__file__, 68, 12), array_379792, *[list_379793], **kwargs_379797)
    
    # Getting the type of 'scalar' (line 68)
    scalar_379799 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 41), 'scalar')
    # Applying the binary operator '*' (line 68)
    result_mul_379800 = python_operator(stypy.reporting.localization.Localization(__file__, 68, 12), '*', array_call_result_379798, scalar_379799)
    
    # Obtaining the member 'dtype' of a type (line 68)
    dtype_379801 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 68, 12), result_mul_379800, 'dtype')
    # Assigning a type to the variable 'stypy_return_type' (line 68)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 68, 4), 'stypy_return_type', dtype_379801)
    
    # ################# End of 'upcast_scalar(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'upcast_scalar' in the type store
    # Getting the type of 'stypy_return_type' (line 64)
    stypy_return_type_379802 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_379802)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'upcast_scalar'
    return stypy_return_type_379802

# Assigning a type to the variable 'upcast_scalar' (line 64)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 64, 0), 'upcast_scalar', upcast_scalar)

@norecursion
def downcast_intp_index(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'downcast_intp_index'
    module_type_store = module_type_store.open_function_context('downcast_intp_index', 71, 0, False)
    
    # Passed parameters checking function
    downcast_intp_index.stypy_localization = localization
    downcast_intp_index.stypy_type_of_self = None
    downcast_intp_index.stypy_type_store = module_type_store
    downcast_intp_index.stypy_function_name = 'downcast_intp_index'
    downcast_intp_index.stypy_param_names_list = ['arr']
    downcast_intp_index.stypy_varargs_param_name = None
    downcast_intp_index.stypy_kwargs_param_name = None
    downcast_intp_index.stypy_call_defaults = defaults
    downcast_intp_index.stypy_call_varargs = varargs
    downcast_intp_index.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'downcast_intp_index', ['arr'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'downcast_intp_index', localization, ['arr'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'downcast_intp_index(...)' code ##################

    str_379803 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 77, (-1)), 'str', '\n    Down-cast index array to np.intp dtype if it is of a larger dtype.\n\n    Raise an error if the array contains a value that is too large for\n    intp.\n    ')
    
    
    # Getting the type of 'arr' (line 78)
    arr_379804 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 7), 'arr')
    # Obtaining the member 'dtype' of a type (line 78)
    dtype_379805 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 78, 7), arr_379804, 'dtype')
    # Obtaining the member 'itemsize' of a type (line 78)
    itemsize_379806 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 78, 7), dtype_379805, 'itemsize')
    
    # Call to dtype(...): (line 78)
    # Processing the call arguments (line 78)
    # Getting the type of 'np' (line 78)
    np_379809 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 37), 'np', False)
    # Obtaining the member 'intp' of a type (line 78)
    intp_379810 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 78, 37), np_379809, 'intp')
    # Processing the call keyword arguments (line 78)
    kwargs_379811 = {}
    # Getting the type of 'np' (line 78)
    np_379807 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 28), 'np', False)
    # Obtaining the member 'dtype' of a type (line 78)
    dtype_379808 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 78, 28), np_379807, 'dtype')
    # Calling dtype(args, kwargs) (line 78)
    dtype_call_result_379812 = invoke(stypy.reporting.localization.Localization(__file__, 78, 28), dtype_379808, *[intp_379810], **kwargs_379811)
    
    # Obtaining the member 'itemsize' of a type (line 78)
    itemsize_379813 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 78, 28), dtype_call_result_379812, 'itemsize')
    # Applying the binary operator '>' (line 78)
    result_gt_379814 = python_operator(stypy.reporting.localization.Localization(__file__, 78, 7), '>', itemsize_379806, itemsize_379813)
    
    # Testing the type of an if condition (line 78)
    if_condition_379815 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 78, 4), result_gt_379814)
    # Assigning a type to the variable 'if_condition_379815' (line 78)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 78, 4), 'if_condition_379815', if_condition_379815)
    # SSA begins for if statement (line 78)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    
    # Getting the type of 'arr' (line 79)
    arr_379816 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 11), 'arr')
    # Obtaining the member 'size' of a type (line 79)
    size_379817 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 79, 11), arr_379816, 'size')
    int_379818 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 79, 23), 'int')
    # Applying the binary operator '==' (line 79)
    result_eq_379819 = python_operator(stypy.reporting.localization.Localization(__file__, 79, 11), '==', size_379817, int_379818)
    
    # Testing the type of an if condition (line 79)
    if_condition_379820 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 79, 8), result_eq_379819)
    # Assigning a type to the variable 'if_condition_379820' (line 79)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 79, 8), 'if_condition_379820', if_condition_379820)
    # SSA begins for if statement (line 79)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to astype(...): (line 80)
    # Processing the call arguments (line 80)
    # Getting the type of 'np' (line 80)
    np_379823 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 30), 'np', False)
    # Obtaining the member 'intp' of a type (line 80)
    intp_379824 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 80, 30), np_379823, 'intp')
    # Processing the call keyword arguments (line 80)
    kwargs_379825 = {}
    # Getting the type of 'arr' (line 80)
    arr_379821 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 19), 'arr', False)
    # Obtaining the member 'astype' of a type (line 80)
    astype_379822 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 80, 19), arr_379821, 'astype')
    # Calling astype(args, kwargs) (line 80)
    astype_call_result_379826 = invoke(stypy.reporting.localization.Localization(__file__, 80, 19), astype_379822, *[intp_379824], **kwargs_379825)
    
    # Assigning a type to the variable 'stypy_return_type' (line 80)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 80, 12), 'stypy_return_type', astype_call_result_379826)
    # SSA join for if statement (line 79)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 81):
    
    # Assigning a Call to a Name (line 81):
    
    # Call to max(...): (line 81)
    # Processing the call keyword arguments (line 81)
    kwargs_379829 = {}
    # Getting the type of 'arr' (line 81)
    arr_379827 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 17), 'arr', False)
    # Obtaining the member 'max' of a type (line 81)
    max_379828 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 81, 17), arr_379827, 'max')
    # Calling max(args, kwargs) (line 81)
    max_call_result_379830 = invoke(stypy.reporting.localization.Localization(__file__, 81, 17), max_379828, *[], **kwargs_379829)
    
    # Assigning a type to the variable 'maxval' (line 81)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 81, 8), 'maxval', max_call_result_379830)
    
    # Assigning a Call to a Name (line 82):
    
    # Assigning a Call to a Name (line 82):
    
    # Call to min(...): (line 82)
    # Processing the call keyword arguments (line 82)
    kwargs_379833 = {}
    # Getting the type of 'arr' (line 82)
    arr_379831 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 17), 'arr', False)
    # Obtaining the member 'min' of a type (line 82)
    min_379832 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 82, 17), arr_379831, 'min')
    # Calling min(args, kwargs) (line 82)
    min_call_result_379834 = invoke(stypy.reporting.localization.Localization(__file__, 82, 17), min_379832, *[], **kwargs_379833)
    
    # Assigning a type to the variable 'minval' (line 82)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 82, 8), 'minval', min_call_result_379834)
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'maxval' (line 83)
    maxval_379835 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 11), 'maxval')
    
    # Call to iinfo(...): (line 83)
    # Processing the call arguments (line 83)
    # Getting the type of 'np' (line 83)
    np_379838 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 29), 'np', False)
    # Obtaining the member 'intp' of a type (line 83)
    intp_379839 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 83, 29), np_379838, 'intp')
    # Processing the call keyword arguments (line 83)
    kwargs_379840 = {}
    # Getting the type of 'np' (line 83)
    np_379836 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 20), 'np', False)
    # Obtaining the member 'iinfo' of a type (line 83)
    iinfo_379837 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 83, 20), np_379836, 'iinfo')
    # Calling iinfo(args, kwargs) (line 83)
    iinfo_call_result_379841 = invoke(stypy.reporting.localization.Localization(__file__, 83, 20), iinfo_379837, *[intp_379839], **kwargs_379840)
    
    # Obtaining the member 'max' of a type (line 83)
    max_379842 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 83, 20), iinfo_call_result_379841, 'max')
    # Applying the binary operator '>' (line 83)
    result_gt_379843 = python_operator(stypy.reporting.localization.Localization(__file__, 83, 11), '>', maxval_379835, max_379842)
    
    
    # Getting the type of 'minval' (line 83)
    minval_379844 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 45), 'minval')
    
    # Call to iinfo(...): (line 83)
    # Processing the call arguments (line 83)
    # Getting the type of 'np' (line 83)
    np_379847 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 63), 'np', False)
    # Obtaining the member 'intp' of a type (line 83)
    intp_379848 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 83, 63), np_379847, 'intp')
    # Processing the call keyword arguments (line 83)
    kwargs_379849 = {}
    # Getting the type of 'np' (line 83)
    np_379845 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 54), 'np', False)
    # Obtaining the member 'iinfo' of a type (line 83)
    iinfo_379846 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 83, 54), np_379845, 'iinfo')
    # Calling iinfo(args, kwargs) (line 83)
    iinfo_call_result_379850 = invoke(stypy.reporting.localization.Localization(__file__, 83, 54), iinfo_379846, *[intp_379848], **kwargs_379849)
    
    # Obtaining the member 'min' of a type (line 83)
    min_379851 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 83, 54), iinfo_call_result_379850, 'min')
    # Applying the binary operator '<' (line 83)
    result_lt_379852 = python_operator(stypy.reporting.localization.Localization(__file__, 83, 45), '<', minval_379844, min_379851)
    
    # Applying the binary operator 'or' (line 83)
    result_or_keyword_379853 = python_operator(stypy.reporting.localization.Localization(__file__, 83, 11), 'or', result_gt_379843, result_lt_379852)
    
    # Testing the type of an if condition (line 83)
    if_condition_379854 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 83, 8), result_or_keyword_379853)
    # Assigning a type to the variable 'if_condition_379854' (line 83)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 83, 8), 'if_condition_379854', if_condition_379854)
    # SSA begins for if statement (line 83)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 84)
    # Processing the call arguments (line 84)
    str_379856 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 84, 29), 'str', 'Cannot deal with arrays with indices larger than the machine maximum address size (e.g. 64-bit indices on 32-bit machine).')
    # Processing the call keyword arguments (line 84)
    kwargs_379857 = {}
    # Getting the type of 'ValueError' (line 84)
    ValueError_379855 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 18), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 84)
    ValueError_call_result_379858 = invoke(stypy.reporting.localization.Localization(__file__, 84, 18), ValueError_379855, *[str_379856], **kwargs_379857)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 84, 12), ValueError_call_result_379858, 'raise parameter', BaseException)
    # SSA join for if statement (line 83)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to astype(...): (line 87)
    # Processing the call arguments (line 87)
    # Getting the type of 'np' (line 87)
    np_379861 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 26), 'np', False)
    # Obtaining the member 'intp' of a type (line 87)
    intp_379862 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 87, 26), np_379861, 'intp')
    # Processing the call keyword arguments (line 87)
    kwargs_379863 = {}
    # Getting the type of 'arr' (line 87)
    arr_379859 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 15), 'arr', False)
    # Obtaining the member 'astype' of a type (line 87)
    astype_379860 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 87, 15), arr_379859, 'astype')
    # Calling astype(args, kwargs) (line 87)
    astype_call_result_379864 = invoke(stypy.reporting.localization.Localization(__file__, 87, 15), astype_379860, *[intp_379862], **kwargs_379863)
    
    # Assigning a type to the variable 'stypy_return_type' (line 87)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 87, 8), 'stypy_return_type', astype_call_result_379864)
    # SSA join for if statement (line 78)
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'arr' (line 88)
    arr_379865 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 11), 'arr')
    # Assigning a type to the variable 'stypy_return_type' (line 88)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 88, 4), 'stypy_return_type', arr_379865)
    
    # ################# End of 'downcast_intp_index(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'downcast_intp_index' in the type store
    # Getting the type of 'stypy_return_type' (line 71)
    stypy_return_type_379866 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_379866)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'downcast_intp_index'
    return stypy_return_type_379866

# Assigning a type to the variable 'downcast_intp_index' (line 71)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 71, 0), 'downcast_intp_index', downcast_intp_index)

@norecursion
def to_native(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'to_native'
    module_type_store = module_type_store.open_function_context('to_native', 91, 0, False)
    
    # Passed parameters checking function
    to_native.stypy_localization = localization
    to_native.stypy_type_of_self = None
    to_native.stypy_type_store = module_type_store
    to_native.stypy_function_name = 'to_native'
    to_native.stypy_param_names_list = ['A']
    to_native.stypy_varargs_param_name = None
    to_native.stypy_kwargs_param_name = None
    to_native.stypy_call_defaults = defaults
    to_native.stypy_call_varargs = varargs
    to_native.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'to_native', ['A'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'to_native', localization, ['A'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'to_native(...)' code ##################

    
    # Call to asarray(...): (line 92)
    # Processing the call arguments (line 92)
    # Getting the type of 'A' (line 92)
    A_379869 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 22), 'A', False)
    # Processing the call keyword arguments (line 92)
    
    # Call to newbyteorder(...): (line 92)
    # Processing the call arguments (line 92)
    str_379873 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 92, 52), 'str', 'native')
    # Processing the call keyword arguments (line 92)
    kwargs_379874 = {}
    # Getting the type of 'A' (line 92)
    A_379870 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 31), 'A', False)
    # Obtaining the member 'dtype' of a type (line 92)
    dtype_379871 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 92, 31), A_379870, 'dtype')
    # Obtaining the member 'newbyteorder' of a type (line 92)
    newbyteorder_379872 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 92, 31), dtype_379871, 'newbyteorder')
    # Calling newbyteorder(args, kwargs) (line 92)
    newbyteorder_call_result_379875 = invoke(stypy.reporting.localization.Localization(__file__, 92, 31), newbyteorder_379872, *[str_379873], **kwargs_379874)
    
    keyword_379876 = newbyteorder_call_result_379875
    kwargs_379877 = {'dtype': keyword_379876}
    # Getting the type of 'np' (line 92)
    np_379867 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 11), 'np', False)
    # Obtaining the member 'asarray' of a type (line 92)
    asarray_379868 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 92, 11), np_379867, 'asarray')
    # Calling asarray(args, kwargs) (line 92)
    asarray_call_result_379878 = invoke(stypy.reporting.localization.Localization(__file__, 92, 11), asarray_379868, *[A_379869], **kwargs_379877)
    
    # Assigning a type to the variable 'stypy_return_type' (line 92)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 92, 4), 'stypy_return_type', asarray_call_result_379878)
    
    # ################# End of 'to_native(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'to_native' in the type store
    # Getting the type of 'stypy_return_type' (line 91)
    stypy_return_type_379879 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_379879)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'to_native'
    return stypy_return_type_379879

# Assigning a type to the variable 'to_native' (line 91)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 91, 0), 'to_native', to_native)

@norecursion
def getdtype(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'None' (line 95)
    None_379880 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 22), 'None')
    # Getting the type of 'None' (line 95)
    None_379881 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 36), 'None')
    defaults = [None_379880, None_379881]
    # Create a new context for function 'getdtype'
    module_type_store = module_type_store.open_function_context('getdtype', 95, 0, False)
    
    # Passed parameters checking function
    getdtype.stypy_localization = localization
    getdtype.stypy_type_of_self = None
    getdtype.stypy_type_store = module_type_store
    getdtype.stypy_function_name = 'getdtype'
    getdtype.stypy_param_names_list = ['dtype', 'a', 'default']
    getdtype.stypy_varargs_param_name = None
    getdtype.stypy_kwargs_param_name = None
    getdtype.stypy_call_defaults = defaults
    getdtype.stypy_call_varargs = varargs
    getdtype.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'getdtype', ['dtype', 'a', 'default'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'getdtype', localization, ['dtype', 'a', 'default'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'getdtype(...)' code ##################

    str_379882 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 101, (-1)), 'str', "Function used to simplify argument processing.  If 'dtype' is not\n    specified (is None), returns a.dtype; otherwise returns a np.dtype\n    object created from the specified dtype argument.  If 'dtype' and 'a'\n    are both None, construct a data type out of the 'default' parameter.\n    Furthermore, 'dtype' must be in 'allowed' set.\n    ")
    
    # Type idiom detected: calculating its left and rigth part (line 103)
    # Getting the type of 'dtype' (line 103)
    dtype_379883 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 7), 'dtype')
    # Getting the type of 'None' (line 103)
    None_379884 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 16), 'None')
    
    (may_be_379885, more_types_in_union_379886) = may_be_none(dtype_379883, None_379884)

    if may_be_379885:

        if more_types_in_union_379886:
            # Runtime conditional SSA (line 103)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        
        # SSA begins for try-except statement (line 104)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
        
        # Assigning a Attribute to a Name (line 105):
        
        # Assigning a Attribute to a Name (line 105):
        # Getting the type of 'a' (line 105)
        a_379887 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 23), 'a')
        # Obtaining the member 'dtype' of a type (line 105)
        dtype_379888 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 105, 23), a_379887, 'dtype')
        # Assigning a type to the variable 'newdtype' (line 105)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 105, 12), 'newdtype', dtype_379888)
        # SSA branch for the except part of a try statement (line 104)
        # SSA branch for the except 'AttributeError' branch of a try statement (line 104)
        module_type_store.open_ssa_branch('except')
        
        # Type idiom detected: calculating its left and rigth part (line 107)
        # Getting the type of 'default' (line 107)
        default_379889 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 12), 'default')
        # Getting the type of 'None' (line 107)
        None_379890 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 30), 'None')
        
        (may_be_379891, more_types_in_union_379892) = may_not_be_none(default_379889, None_379890)

        if may_be_379891:

            if more_types_in_union_379892:
                # Runtime conditional SSA (line 107)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a Call to a Name (line 108):
            
            # Assigning a Call to a Name (line 108):
            
            # Call to dtype(...): (line 108)
            # Processing the call arguments (line 108)
            # Getting the type of 'default' (line 108)
            default_379895 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 36), 'default', False)
            # Processing the call keyword arguments (line 108)
            kwargs_379896 = {}
            # Getting the type of 'np' (line 108)
            np_379893 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 27), 'np', False)
            # Obtaining the member 'dtype' of a type (line 108)
            dtype_379894 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 108, 27), np_379893, 'dtype')
            # Calling dtype(args, kwargs) (line 108)
            dtype_call_result_379897 = invoke(stypy.reporting.localization.Localization(__file__, 108, 27), dtype_379894, *[default_379895], **kwargs_379896)
            
            # Assigning a type to the variable 'newdtype' (line 108)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 108, 16), 'newdtype', dtype_call_result_379897)

            if more_types_in_union_379892:
                # Runtime conditional SSA for else branch (line 107)
                module_type_store.open_ssa_branch('idiom else')



        if ((not may_be_379891) or more_types_in_union_379892):
            
            # Call to TypeError(...): (line 110)
            # Processing the call arguments (line 110)
            str_379899 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 110, 32), 'str', 'could not interpret data type')
            # Processing the call keyword arguments (line 110)
            kwargs_379900 = {}
            # Getting the type of 'TypeError' (line 110)
            TypeError_379898 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 22), 'TypeError', False)
            # Calling TypeError(args, kwargs) (line 110)
            TypeError_call_result_379901 = invoke(stypy.reporting.localization.Localization(__file__, 110, 22), TypeError_379898, *[str_379899], **kwargs_379900)
            
            ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 110, 16), TypeError_call_result_379901, 'raise parameter', BaseException)

            if (may_be_379891 and more_types_in_union_379892):
                # SSA join for if statement (line 107)
                module_type_store = module_type_store.join_ssa_context()


        
        # SSA join for try-except statement (line 104)
        module_type_store = module_type_store.join_ssa_context()
        

        if more_types_in_union_379886:
            # Runtime conditional SSA for else branch (line 103)
            module_type_store.open_ssa_branch('idiom else')



    if ((not may_be_379885) or more_types_in_union_379886):
        
        # Assigning a Call to a Name (line 112):
        
        # Assigning a Call to a Name (line 112):
        
        # Call to dtype(...): (line 112)
        # Processing the call arguments (line 112)
        # Getting the type of 'dtype' (line 112)
        dtype_379904 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 28), 'dtype', False)
        # Processing the call keyword arguments (line 112)
        kwargs_379905 = {}
        # Getting the type of 'np' (line 112)
        np_379902 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 19), 'np', False)
        # Obtaining the member 'dtype' of a type (line 112)
        dtype_379903 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 112, 19), np_379902, 'dtype')
        # Calling dtype(args, kwargs) (line 112)
        dtype_call_result_379906 = invoke(stypy.reporting.localization.Localization(__file__, 112, 19), dtype_379903, *[dtype_379904], **kwargs_379905)
        
        # Assigning a type to the variable 'newdtype' (line 112)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 112, 8), 'newdtype', dtype_call_result_379906)
        
        
        # Getting the type of 'newdtype' (line 113)
        newdtype_379907 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 11), 'newdtype')
        # Getting the type of 'np' (line 113)
        np_379908 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 23), 'np')
        # Obtaining the member 'object_' of a type (line 113)
        object__379909 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 113, 23), np_379908, 'object_')
        # Applying the binary operator '==' (line 113)
        result_eq_379910 = python_operator(stypy.reporting.localization.Localization(__file__, 113, 11), '==', newdtype_379907, object__379909)
        
        # Testing the type of an if condition (line 113)
        if_condition_379911 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 113, 8), result_eq_379910)
        # Assigning a type to the variable 'if_condition_379911' (line 113)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 113, 8), 'if_condition_379911', if_condition_379911)
        # SSA begins for if statement (line 113)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to warn(...): (line 114)
        # Processing the call arguments (line 114)
        str_379914 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 114, 26), 'str', 'object dtype is not supported by sparse matrices')
        # Processing the call keyword arguments (line 114)
        kwargs_379915 = {}
        # Getting the type of 'warnings' (line 114)
        warnings_379912 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 12), 'warnings', False)
        # Obtaining the member 'warn' of a type (line 114)
        warn_379913 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 114, 12), warnings_379912, 'warn')
        # Calling warn(args, kwargs) (line 114)
        warn_call_result_379916 = invoke(stypy.reporting.localization.Localization(__file__, 114, 12), warn_379913, *[str_379914], **kwargs_379915)
        
        # SSA join for if statement (line 113)
        module_type_store = module_type_store.join_ssa_context()
        

        if (may_be_379885 and more_types_in_union_379886):
            # SSA join for if statement (line 103)
            module_type_store = module_type_store.join_ssa_context()


    
    # Getting the type of 'newdtype' (line 116)
    newdtype_379917 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 11), 'newdtype')
    # Assigning a type to the variable 'stypy_return_type' (line 116)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 116, 4), 'stypy_return_type', newdtype_379917)
    
    # ################# End of 'getdtype(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'getdtype' in the type store
    # Getting the type of 'stypy_return_type' (line 95)
    stypy_return_type_379918 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_379918)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'getdtype'
    return stypy_return_type_379918

# Assigning a type to the variable 'getdtype' (line 95)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 95, 0), 'getdtype', getdtype)

@norecursion
def get_index_dtype(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    
    # Obtaining an instance of the builtin type 'tuple' (line 119)
    tuple_379919 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 119, 27), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 119)
    
    # Getting the type of 'None' (line 119)
    None_379920 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 38), 'None')
    # Getting the type of 'False' (line 119)
    False_379921 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 59), 'False')
    defaults = [tuple_379919, None_379920, False_379921]
    # Create a new context for function 'get_index_dtype'
    module_type_store = module_type_store.open_function_context('get_index_dtype', 119, 0, False)
    
    # Passed parameters checking function
    get_index_dtype.stypy_localization = localization
    get_index_dtype.stypy_type_of_self = None
    get_index_dtype.stypy_type_store = module_type_store
    get_index_dtype.stypy_function_name = 'get_index_dtype'
    get_index_dtype.stypy_param_names_list = ['arrays', 'maxval', 'check_contents']
    get_index_dtype.stypy_varargs_param_name = None
    get_index_dtype.stypy_kwargs_param_name = None
    get_index_dtype.stypy_call_defaults = defaults
    get_index_dtype.stypy_call_varargs = varargs
    get_index_dtype.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'get_index_dtype', ['arrays', 'maxval', 'check_contents'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'get_index_dtype', localization, ['arrays', 'maxval', 'check_contents'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'get_index_dtype(...)' code ##################

    str_379922 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 139, (-1)), 'str', '\n    Based on input (integer) arrays `a`, determine a suitable index data\n    type that can hold the data in the arrays.\n\n    Parameters\n    ----------\n    arrays : tuple of array_like\n        Input arrays whose types/contents to check\n    maxval : float, optional\n        Maximum value needed\n    check_contents : bool, optional\n        Whether to check the values in the arrays and not just their types.\n        Default: False (check only the types)\n\n    Returns\n    -------\n    dtype : dtype\n        Suitable index data type (int32 or int64)\n\n    ')
    
    # Assigning a Attribute to a Name (line 141):
    
    # Assigning a Attribute to a Name (line 141):
    
    # Call to iinfo(...): (line 141)
    # Processing the call arguments (line 141)
    # Getting the type of 'np' (line 141)
    np_379925 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 24), 'np', False)
    # Obtaining the member 'int32' of a type (line 141)
    int32_379926 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 141, 24), np_379925, 'int32')
    # Processing the call keyword arguments (line 141)
    kwargs_379927 = {}
    # Getting the type of 'np' (line 141)
    np_379923 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 15), 'np', False)
    # Obtaining the member 'iinfo' of a type (line 141)
    iinfo_379924 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 141, 15), np_379923, 'iinfo')
    # Calling iinfo(args, kwargs) (line 141)
    iinfo_call_result_379928 = invoke(stypy.reporting.localization.Localization(__file__, 141, 15), iinfo_379924, *[int32_379926], **kwargs_379927)
    
    # Obtaining the member 'max' of a type (line 141)
    max_379929 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 141, 15), iinfo_call_result_379928, 'max')
    # Assigning a type to the variable 'int32max' (line 141)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 141, 4), 'int32max', max_379929)
    
    # Assigning a Attribute to a Name (line 143):
    
    # Assigning a Attribute to a Name (line 143):
    # Getting the type of 'np' (line 143)
    np_379930 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 12), 'np')
    # Obtaining the member 'intc' of a type (line 143)
    intc_379931 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 143, 12), np_379930, 'intc')
    # Assigning a type to the variable 'dtype' (line 143)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 143, 4), 'dtype', intc_379931)
    
    # Type idiom detected: calculating its left and rigth part (line 144)
    # Getting the type of 'maxval' (line 144)
    maxval_379932 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 4), 'maxval')
    # Getting the type of 'None' (line 144)
    None_379933 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 21), 'None')
    
    (may_be_379934, more_types_in_union_379935) = may_not_be_none(maxval_379932, None_379933)

    if may_be_379934:

        if more_types_in_union_379935:
            # Runtime conditional SSA (line 144)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        
        # Getting the type of 'maxval' (line 145)
        maxval_379936 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 11), 'maxval')
        # Getting the type of 'int32max' (line 145)
        int32max_379937 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 20), 'int32max')
        # Applying the binary operator '>' (line 145)
        result_gt_379938 = python_operator(stypy.reporting.localization.Localization(__file__, 145, 11), '>', maxval_379936, int32max_379937)
        
        # Testing the type of an if condition (line 145)
        if_condition_379939 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 145, 8), result_gt_379938)
        # Assigning a type to the variable 'if_condition_379939' (line 145)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 145, 8), 'if_condition_379939', if_condition_379939)
        # SSA begins for if statement (line 145)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Attribute to a Name (line 146):
        
        # Assigning a Attribute to a Name (line 146):
        # Getting the type of 'np' (line 146)
        np_379940 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 20), 'np')
        # Obtaining the member 'int64' of a type (line 146)
        int64_379941 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 146, 20), np_379940, 'int64')
        # Assigning a type to the variable 'dtype' (line 146)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 146, 12), 'dtype', int64_379941)
        # SSA join for if statement (line 145)
        module_type_store = module_type_store.join_ssa_context()
        

        if more_types_in_union_379935:
            # SSA join for if statement (line 144)
            module_type_store = module_type_store.join_ssa_context()


    
    
    
    # Call to isinstance(...): (line 148)
    # Processing the call arguments (line 148)
    # Getting the type of 'arrays' (line 148)
    arrays_379943 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 18), 'arrays', False)
    # Getting the type of 'np' (line 148)
    np_379944 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 26), 'np', False)
    # Obtaining the member 'ndarray' of a type (line 148)
    ndarray_379945 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 148, 26), np_379944, 'ndarray')
    # Processing the call keyword arguments (line 148)
    kwargs_379946 = {}
    # Getting the type of 'isinstance' (line 148)
    isinstance_379942 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 7), 'isinstance', False)
    # Calling isinstance(args, kwargs) (line 148)
    isinstance_call_result_379947 = invoke(stypy.reporting.localization.Localization(__file__, 148, 7), isinstance_379942, *[arrays_379943, ndarray_379945], **kwargs_379946)
    
    # Testing the type of an if condition (line 148)
    if_condition_379948 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 148, 4), isinstance_call_result_379947)
    # Assigning a type to the variable 'if_condition_379948' (line 148)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 148, 4), 'if_condition_379948', if_condition_379948)
    # SSA begins for if statement (line 148)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Tuple to a Name (line 149):
    
    # Assigning a Tuple to a Name (line 149):
    
    # Obtaining an instance of the builtin type 'tuple' (line 149)
    tuple_379949 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 149, 18), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 149)
    # Adding element type (line 149)
    # Getting the type of 'arrays' (line 149)
    arrays_379950 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 18), 'arrays')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 149, 18), tuple_379949, arrays_379950)
    
    # Assigning a type to the variable 'arrays' (line 149)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 149, 8), 'arrays', tuple_379949)
    # SSA join for if statement (line 148)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Getting the type of 'arrays' (line 151)
    arrays_379951 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 15), 'arrays')
    # Testing the type of a for loop iterable (line 151)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 151, 4), arrays_379951)
    # Getting the type of the for loop variable (line 151)
    for_loop_var_379952 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 151, 4), arrays_379951)
    # Assigning a type to the variable 'arr' (line 151)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 151, 4), 'arr', for_loop_var_379952)
    # SSA begins for a for statement (line 151)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Call to a Name (line 152):
    
    # Assigning a Call to a Name (line 152):
    
    # Call to asarray(...): (line 152)
    # Processing the call arguments (line 152)
    # Getting the type of 'arr' (line 152)
    arr_379955 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 25), 'arr', False)
    # Processing the call keyword arguments (line 152)
    kwargs_379956 = {}
    # Getting the type of 'np' (line 152)
    np_379953 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 14), 'np', False)
    # Obtaining the member 'asarray' of a type (line 152)
    asarray_379954 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 152, 14), np_379953, 'asarray')
    # Calling asarray(args, kwargs) (line 152)
    asarray_call_result_379957 = invoke(stypy.reporting.localization.Localization(__file__, 152, 14), asarray_379954, *[arr_379955], **kwargs_379956)
    
    # Assigning a type to the variable 'arr' (line 152)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 152, 8), 'arr', asarray_call_result_379957)
    
    
    # Getting the type of 'arr' (line 153)
    arr_379958 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 11), 'arr')
    # Obtaining the member 'dtype' of a type (line 153)
    dtype_379959 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 153, 11), arr_379958, 'dtype')
    # Getting the type of 'np' (line 153)
    np_379960 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 23), 'np')
    # Obtaining the member 'int32' of a type (line 153)
    int32_379961 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 153, 23), np_379960, 'int32')
    # Applying the binary operator '>' (line 153)
    result_gt_379962 = python_operator(stypy.reporting.localization.Localization(__file__, 153, 11), '>', dtype_379959, int32_379961)
    
    # Testing the type of an if condition (line 153)
    if_condition_379963 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 153, 8), result_gt_379962)
    # Assigning a type to the variable 'if_condition_379963' (line 153)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 153, 8), 'if_condition_379963', if_condition_379963)
    # SSA begins for if statement (line 153)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Getting the type of 'check_contents' (line 154)
    check_contents_379964 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 15), 'check_contents')
    # Testing the type of an if condition (line 154)
    if_condition_379965 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 154, 12), check_contents_379964)
    # Assigning a type to the variable 'if_condition_379965' (line 154)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 154, 12), 'if_condition_379965', if_condition_379965)
    # SSA begins for if statement (line 154)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    
    # Getting the type of 'arr' (line 155)
    arr_379966 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 19), 'arr')
    # Obtaining the member 'size' of a type (line 155)
    size_379967 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 155, 19), arr_379966, 'size')
    int_379968 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 155, 31), 'int')
    # Applying the binary operator '==' (line 155)
    result_eq_379969 = python_operator(stypy.reporting.localization.Localization(__file__, 155, 19), '==', size_379967, int_379968)
    
    # Testing the type of an if condition (line 155)
    if_condition_379970 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 155, 16), result_eq_379969)
    # Assigning a type to the variable 'if_condition_379970' (line 155)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 155, 16), 'if_condition_379970', if_condition_379970)
    # SSA begins for if statement (line 155)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # SSA branch for the else part of an if statement (line 155)
    module_type_store.open_ssa_branch('else')
    
    
    # Call to issubdtype(...): (line 158)
    # Processing the call arguments (line 158)
    # Getting the type of 'arr' (line 158)
    arr_379973 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 35), 'arr', False)
    # Obtaining the member 'dtype' of a type (line 158)
    dtype_379974 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 158, 35), arr_379973, 'dtype')
    # Getting the type of 'np' (line 158)
    np_379975 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 46), 'np', False)
    # Obtaining the member 'integer' of a type (line 158)
    integer_379976 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 158, 46), np_379975, 'integer')
    # Processing the call keyword arguments (line 158)
    kwargs_379977 = {}
    # Getting the type of 'np' (line 158)
    np_379971 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 21), 'np', False)
    # Obtaining the member 'issubdtype' of a type (line 158)
    issubdtype_379972 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 158, 21), np_379971, 'issubdtype')
    # Calling issubdtype(args, kwargs) (line 158)
    issubdtype_call_result_379978 = invoke(stypy.reporting.localization.Localization(__file__, 158, 21), issubdtype_379972, *[dtype_379974, integer_379976], **kwargs_379977)
    
    # Testing the type of an if condition (line 158)
    if_condition_379979 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 158, 21), issubdtype_call_result_379978)
    # Assigning a type to the variable 'if_condition_379979' (line 158)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 158, 21), 'if_condition_379979', if_condition_379979)
    # SSA begins for if statement (line 158)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 159):
    
    # Assigning a Call to a Name (line 159):
    
    # Call to max(...): (line 159)
    # Processing the call keyword arguments (line 159)
    kwargs_379982 = {}
    # Getting the type of 'arr' (line 159)
    arr_379980 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 29), 'arr', False)
    # Obtaining the member 'max' of a type (line 159)
    max_379981 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 159, 29), arr_379980, 'max')
    # Calling max(args, kwargs) (line 159)
    max_call_result_379983 = invoke(stypy.reporting.localization.Localization(__file__, 159, 29), max_379981, *[], **kwargs_379982)
    
    # Assigning a type to the variable 'maxval' (line 159)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 159, 20), 'maxval', max_call_result_379983)
    
    # Assigning a Call to a Name (line 160):
    
    # Assigning a Call to a Name (line 160):
    
    # Call to min(...): (line 160)
    # Processing the call keyword arguments (line 160)
    kwargs_379986 = {}
    # Getting the type of 'arr' (line 160)
    arr_379984 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 29), 'arr', False)
    # Obtaining the member 'min' of a type (line 160)
    min_379985 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 160, 29), arr_379984, 'min')
    # Calling min(args, kwargs) (line 160)
    min_call_result_379987 = invoke(stypy.reporting.localization.Localization(__file__, 160, 29), min_379985, *[], **kwargs_379986)
    
    # Assigning a type to the variable 'minval' (line 160)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 160, 20), 'minval', min_call_result_379987)
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'minval' (line 161)
    minval_379988 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 24), 'minval')
    
    # Call to iinfo(...): (line 161)
    # Processing the call arguments (line 161)
    # Getting the type of 'np' (line 161)
    np_379991 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 43), 'np', False)
    # Obtaining the member 'int32' of a type (line 161)
    int32_379992 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 161, 43), np_379991, 'int32')
    # Processing the call keyword arguments (line 161)
    kwargs_379993 = {}
    # Getting the type of 'np' (line 161)
    np_379989 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 34), 'np', False)
    # Obtaining the member 'iinfo' of a type (line 161)
    iinfo_379990 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 161, 34), np_379989, 'iinfo')
    # Calling iinfo(args, kwargs) (line 161)
    iinfo_call_result_379994 = invoke(stypy.reporting.localization.Localization(__file__, 161, 34), iinfo_379990, *[int32_379992], **kwargs_379993)
    
    # Obtaining the member 'min' of a type (line 161)
    min_379995 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 161, 34), iinfo_call_result_379994, 'min')
    # Applying the binary operator '>=' (line 161)
    result_ge_379996 = python_operator(stypy.reporting.localization.Localization(__file__, 161, 24), '>=', minval_379988, min_379995)
    
    
    # Getting the type of 'maxval' (line 162)
    maxval_379997 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 28), 'maxval')
    
    # Call to iinfo(...): (line 162)
    # Processing the call arguments (line 162)
    # Getting the type of 'np' (line 162)
    np_380000 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 47), 'np', False)
    # Obtaining the member 'int32' of a type (line 162)
    int32_380001 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 162, 47), np_380000, 'int32')
    # Processing the call keyword arguments (line 162)
    kwargs_380002 = {}
    # Getting the type of 'np' (line 162)
    np_379998 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 38), 'np', False)
    # Obtaining the member 'iinfo' of a type (line 162)
    iinfo_379999 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 162, 38), np_379998, 'iinfo')
    # Calling iinfo(args, kwargs) (line 162)
    iinfo_call_result_380003 = invoke(stypy.reporting.localization.Localization(__file__, 162, 38), iinfo_379999, *[int32_380001], **kwargs_380002)
    
    # Obtaining the member 'max' of a type (line 162)
    max_380004 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 162, 38), iinfo_call_result_380003, 'max')
    # Applying the binary operator '<=' (line 162)
    result_le_380005 = python_operator(stypy.reporting.localization.Localization(__file__, 162, 28), '<=', maxval_379997, max_380004)
    
    # Applying the binary operator 'and' (line 161)
    result_and_keyword_380006 = python_operator(stypy.reporting.localization.Localization(__file__, 161, 24), 'and', result_ge_379996, result_le_380005)
    
    # Testing the type of an if condition (line 161)
    if_condition_380007 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 161, 20), result_and_keyword_380006)
    # Assigning a type to the variable 'if_condition_380007' (line 161)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 161, 20), 'if_condition_380007', if_condition_380007)
    # SSA begins for if statement (line 161)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # SSA join for if statement (line 161)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 158)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 155)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 154)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Attribute to a Name (line 166):
    
    # Assigning a Attribute to a Name (line 166):
    # Getting the type of 'np' (line 166)
    np_380008 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 20), 'np')
    # Obtaining the member 'int64' of a type (line 166)
    int64_380009 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 166, 20), np_380008, 'int64')
    # Assigning a type to the variable 'dtype' (line 166)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 166, 12), 'dtype', int64_380009)
    # SSA join for if statement (line 153)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'dtype' (line 169)
    dtype_380010 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 11), 'dtype')
    # Assigning a type to the variable 'stypy_return_type' (line 169)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 169, 4), 'stypy_return_type', dtype_380010)
    
    # ################# End of 'get_index_dtype(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'get_index_dtype' in the type store
    # Getting the type of 'stypy_return_type' (line 119)
    stypy_return_type_380011 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_380011)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'get_index_dtype'
    return stypy_return_type_380011

# Assigning a type to the variable 'get_index_dtype' (line 119)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 119, 0), 'get_index_dtype', get_index_dtype)

@norecursion
def get_sum_dtype(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'get_sum_dtype'
    module_type_store = module_type_store.open_function_context('get_sum_dtype', 172, 0, False)
    
    # Passed parameters checking function
    get_sum_dtype.stypy_localization = localization
    get_sum_dtype.stypy_type_of_self = None
    get_sum_dtype.stypy_type_store = module_type_store
    get_sum_dtype.stypy_function_name = 'get_sum_dtype'
    get_sum_dtype.stypy_param_names_list = ['dtype']
    get_sum_dtype.stypy_varargs_param_name = None
    get_sum_dtype.stypy_kwargs_param_name = None
    get_sum_dtype.stypy_call_defaults = defaults
    get_sum_dtype.stypy_call_varargs = varargs
    get_sum_dtype.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'get_sum_dtype', ['dtype'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'get_sum_dtype', localization, ['dtype'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'get_sum_dtype(...)' code ##################

    str_380012 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 173, 4), 'str', "Mimic numpy's casting for np.sum")
    
    
    # Call to issubdtype(...): (line 174)
    # Processing the call arguments (line 174)
    # Getting the type of 'dtype' (line 174)
    dtype_380015 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 21), 'dtype', False)
    # Getting the type of 'np' (line 174)
    np_380016 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 28), 'np', False)
    # Obtaining the member 'float_' of a type (line 174)
    float__380017 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 174, 28), np_380016, 'float_')
    # Processing the call keyword arguments (line 174)
    kwargs_380018 = {}
    # Getting the type of 'np' (line 174)
    np_380013 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 7), 'np', False)
    # Obtaining the member 'issubdtype' of a type (line 174)
    issubdtype_380014 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 174, 7), np_380013, 'issubdtype')
    # Calling issubdtype(args, kwargs) (line 174)
    issubdtype_call_result_380019 = invoke(stypy.reporting.localization.Localization(__file__, 174, 7), issubdtype_380014, *[dtype_380015, float__380017], **kwargs_380018)
    
    # Testing the type of an if condition (line 174)
    if_condition_380020 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 174, 4), issubdtype_call_result_380019)
    # Assigning a type to the variable 'if_condition_380020' (line 174)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 174, 4), 'if_condition_380020', if_condition_380020)
    # SSA begins for if statement (line 174)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # Getting the type of 'np' (line 175)
    np_380021 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 15), 'np')
    # Obtaining the member 'float_' of a type (line 175)
    float__380022 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 175, 15), np_380021, 'float_')
    # Assigning a type to the variable 'stypy_return_type' (line 175)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 175, 8), 'stypy_return_type', float__380022)
    # SSA join for if statement (line 174)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'dtype' (line 176)
    dtype_380023 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 7), 'dtype')
    # Obtaining the member 'kind' of a type (line 176)
    kind_380024 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 176, 7), dtype_380023, 'kind')
    str_380025 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 176, 21), 'str', 'u')
    # Applying the binary operator '==' (line 176)
    result_eq_380026 = python_operator(stypy.reporting.localization.Localization(__file__, 176, 7), '==', kind_380024, str_380025)
    
    
    # Call to can_cast(...): (line 176)
    # Processing the call arguments (line 176)
    # Getting the type of 'dtype' (line 176)
    dtype_380029 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 41), 'dtype', False)
    # Getting the type of 'np' (line 176)
    np_380030 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 48), 'np', False)
    # Obtaining the member 'uint' of a type (line 176)
    uint_380031 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 176, 48), np_380030, 'uint')
    # Processing the call keyword arguments (line 176)
    kwargs_380032 = {}
    # Getting the type of 'np' (line 176)
    np_380027 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 29), 'np', False)
    # Obtaining the member 'can_cast' of a type (line 176)
    can_cast_380028 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 176, 29), np_380027, 'can_cast')
    # Calling can_cast(args, kwargs) (line 176)
    can_cast_call_result_380033 = invoke(stypy.reporting.localization.Localization(__file__, 176, 29), can_cast_380028, *[dtype_380029, uint_380031], **kwargs_380032)
    
    # Applying the binary operator 'and' (line 176)
    result_and_keyword_380034 = python_operator(stypy.reporting.localization.Localization(__file__, 176, 7), 'and', result_eq_380026, can_cast_call_result_380033)
    
    # Testing the type of an if condition (line 176)
    if_condition_380035 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 176, 4), result_and_keyword_380034)
    # Assigning a type to the variable 'if_condition_380035' (line 176)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 176, 4), 'if_condition_380035', if_condition_380035)
    # SSA begins for if statement (line 176)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # Getting the type of 'np' (line 177)
    np_380036 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 15), 'np')
    # Obtaining the member 'uint' of a type (line 177)
    uint_380037 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 177, 15), np_380036, 'uint')
    # Assigning a type to the variable 'stypy_return_type' (line 177)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 177, 8), 'stypy_return_type', uint_380037)
    # SSA join for if statement (line 176)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Call to can_cast(...): (line 178)
    # Processing the call arguments (line 178)
    # Getting the type of 'dtype' (line 178)
    dtype_380040 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 19), 'dtype', False)
    # Getting the type of 'np' (line 178)
    np_380041 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 26), 'np', False)
    # Obtaining the member 'int_' of a type (line 178)
    int__380042 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 178, 26), np_380041, 'int_')
    # Processing the call keyword arguments (line 178)
    kwargs_380043 = {}
    # Getting the type of 'np' (line 178)
    np_380038 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 7), 'np', False)
    # Obtaining the member 'can_cast' of a type (line 178)
    can_cast_380039 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 178, 7), np_380038, 'can_cast')
    # Calling can_cast(args, kwargs) (line 178)
    can_cast_call_result_380044 = invoke(stypy.reporting.localization.Localization(__file__, 178, 7), can_cast_380039, *[dtype_380040, int__380042], **kwargs_380043)
    
    # Testing the type of an if condition (line 178)
    if_condition_380045 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 178, 4), can_cast_call_result_380044)
    # Assigning a type to the variable 'if_condition_380045' (line 178)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 178, 4), 'if_condition_380045', if_condition_380045)
    # SSA begins for if statement (line 178)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # Getting the type of 'np' (line 179)
    np_380046 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 15), 'np')
    # Obtaining the member 'int_' of a type (line 179)
    int__380047 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 179, 15), np_380046, 'int_')
    # Assigning a type to the variable 'stypy_return_type' (line 179)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 179, 8), 'stypy_return_type', int__380047)
    # SSA join for if statement (line 178)
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'dtype' (line 180)
    dtype_380048 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 11), 'dtype')
    # Assigning a type to the variable 'stypy_return_type' (line 180)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 180, 4), 'stypy_return_type', dtype_380048)
    
    # ################# End of 'get_sum_dtype(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'get_sum_dtype' in the type store
    # Getting the type of 'stypy_return_type' (line 172)
    stypy_return_type_380049 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_380049)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'get_sum_dtype'
    return stypy_return_type_380049

# Assigning a type to the variable 'get_sum_dtype' (line 172)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 172, 0), 'get_sum_dtype', get_sum_dtype)

@norecursion
def isscalarlike(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'isscalarlike'
    module_type_store = module_type_store.open_function_context('isscalarlike', 183, 0, False)
    
    # Passed parameters checking function
    isscalarlike.stypy_localization = localization
    isscalarlike.stypy_type_of_self = None
    isscalarlike.stypy_type_store = module_type_store
    isscalarlike.stypy_function_name = 'isscalarlike'
    isscalarlike.stypy_param_names_list = ['x']
    isscalarlike.stypy_varargs_param_name = None
    isscalarlike.stypy_kwargs_param_name = None
    isscalarlike.stypy_call_defaults = defaults
    isscalarlike.stypy_call_varargs = varargs
    isscalarlike.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'isscalarlike', ['x'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'isscalarlike', localization, ['x'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'isscalarlike(...)' code ##################

    str_380050 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 184, 4), 'str', 'Is x either a scalar, an array scalar, or a 0-dim array?')
    
    # Evaluating a boolean operation
    
    # Call to isscalar(...): (line 185)
    # Processing the call arguments (line 185)
    # Getting the type of 'x' (line 185)
    x_380053 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 23), 'x', False)
    # Processing the call keyword arguments (line 185)
    kwargs_380054 = {}
    # Getting the type of 'np' (line 185)
    np_380051 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 11), 'np', False)
    # Obtaining the member 'isscalar' of a type (line 185)
    isscalar_380052 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 185, 11), np_380051, 'isscalar')
    # Calling isscalar(args, kwargs) (line 185)
    isscalar_call_result_380055 = invoke(stypy.reporting.localization.Localization(__file__, 185, 11), isscalar_380052, *[x_380053], **kwargs_380054)
    
    
    # Evaluating a boolean operation
    
    # Call to isdense(...): (line 185)
    # Processing the call arguments (line 185)
    # Getting the type of 'x' (line 185)
    x_380057 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 38), 'x', False)
    # Processing the call keyword arguments (line 185)
    kwargs_380058 = {}
    # Getting the type of 'isdense' (line 185)
    isdense_380056 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 30), 'isdense', False)
    # Calling isdense(args, kwargs) (line 185)
    isdense_call_result_380059 = invoke(stypy.reporting.localization.Localization(__file__, 185, 30), isdense_380056, *[x_380057], **kwargs_380058)
    
    
    # Getting the type of 'x' (line 185)
    x_380060 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 45), 'x')
    # Obtaining the member 'ndim' of a type (line 185)
    ndim_380061 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 185, 45), x_380060, 'ndim')
    int_380062 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 185, 55), 'int')
    # Applying the binary operator '==' (line 185)
    result_eq_380063 = python_operator(stypy.reporting.localization.Localization(__file__, 185, 45), '==', ndim_380061, int_380062)
    
    # Applying the binary operator 'and' (line 185)
    result_and_keyword_380064 = python_operator(stypy.reporting.localization.Localization(__file__, 185, 30), 'and', isdense_call_result_380059, result_eq_380063)
    
    # Applying the binary operator 'or' (line 185)
    result_or_keyword_380065 = python_operator(stypy.reporting.localization.Localization(__file__, 185, 11), 'or', isscalar_call_result_380055, result_and_keyword_380064)
    
    # Assigning a type to the variable 'stypy_return_type' (line 185)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 185, 4), 'stypy_return_type', result_or_keyword_380065)
    
    # ################# End of 'isscalarlike(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'isscalarlike' in the type store
    # Getting the type of 'stypy_return_type' (line 183)
    stypy_return_type_380066 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_380066)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'isscalarlike'
    return stypy_return_type_380066

# Assigning a type to the variable 'isscalarlike' (line 183)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 183, 0), 'isscalarlike', isscalarlike)

@norecursion
def isintlike(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'isintlike'
    module_type_store = module_type_store.open_function_context('isintlike', 188, 0, False)
    
    # Passed parameters checking function
    isintlike.stypy_localization = localization
    isintlike.stypy_type_of_self = None
    isintlike.stypy_type_store = module_type_store
    isintlike.stypy_function_name = 'isintlike'
    isintlike.stypy_param_names_list = ['x']
    isintlike.stypy_varargs_param_name = None
    isintlike.stypy_kwargs_param_name = None
    isintlike.stypy_call_defaults = defaults
    isintlike.stypy_call_varargs = varargs
    isintlike.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'isintlike', ['x'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'isintlike', localization, ['x'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'isintlike(...)' code ##################

    str_380067 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 191, (-1)), 'str', 'Is x appropriate as an index into a sparse matrix? Returns True\n    if it can be cast safely to a machine int.\n    ')
    
    
    
    # Call to isscalarlike(...): (line 192)
    # Processing the call arguments (line 192)
    # Getting the type of 'x' (line 192)
    x_380069 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 192, 24), 'x', False)
    # Processing the call keyword arguments (line 192)
    kwargs_380070 = {}
    # Getting the type of 'isscalarlike' (line 192)
    isscalarlike_380068 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 192, 11), 'isscalarlike', False)
    # Calling isscalarlike(args, kwargs) (line 192)
    isscalarlike_call_result_380071 = invoke(stypy.reporting.localization.Localization(__file__, 192, 11), isscalarlike_380068, *[x_380069], **kwargs_380070)
    
    # Applying the 'not' unary operator (line 192)
    result_not__380072 = python_operator(stypy.reporting.localization.Localization(__file__, 192, 7), 'not', isscalarlike_call_result_380071)
    
    # Testing the type of an if condition (line 192)
    if_condition_380073 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 192, 4), result_not__380072)
    # Assigning a type to the variable 'if_condition_380073' (line 192)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 192, 4), 'if_condition_380073', if_condition_380073)
    # SSA begins for if statement (line 192)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # Getting the type of 'False' (line 193)
    False_380074 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 15), 'False')
    # Assigning a type to the variable 'stypy_return_type' (line 193)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 193, 8), 'stypy_return_type', False_380074)
    # SSA join for if statement (line 192)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # SSA begins for try-except statement (line 194)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
    
    # Call to bool(...): (line 195)
    # Processing the call arguments (line 195)
    
    
    # Call to int(...): (line 195)
    # Processing the call arguments (line 195)
    # Getting the type of 'x' (line 195)
    x_380077 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 195, 24), 'x', False)
    # Processing the call keyword arguments (line 195)
    kwargs_380078 = {}
    # Getting the type of 'int' (line 195)
    int_380076 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 195, 20), 'int', False)
    # Calling int(args, kwargs) (line 195)
    int_call_result_380079 = invoke(stypy.reporting.localization.Localization(__file__, 195, 20), int_380076, *[x_380077], **kwargs_380078)
    
    # Getting the type of 'x' (line 195)
    x_380080 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 195, 30), 'x', False)
    # Applying the binary operator '==' (line 195)
    result_eq_380081 = python_operator(stypy.reporting.localization.Localization(__file__, 195, 20), '==', int_call_result_380079, x_380080)
    
    # Processing the call keyword arguments (line 195)
    kwargs_380082 = {}
    # Getting the type of 'bool' (line 195)
    bool_380075 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 195, 15), 'bool', False)
    # Calling bool(args, kwargs) (line 195)
    bool_call_result_380083 = invoke(stypy.reporting.localization.Localization(__file__, 195, 15), bool_380075, *[result_eq_380081], **kwargs_380082)
    
    # Assigning a type to the variable 'stypy_return_type' (line 195)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 195, 8), 'stypy_return_type', bool_call_result_380083)
    # SSA branch for the except part of a try statement (line 194)
    # SSA branch for the except 'Tuple' branch of a try statement (line 194)
    module_type_store.open_ssa_branch('except')
    # Getting the type of 'False' (line 197)
    False_380084 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 197, 15), 'False')
    # Assigning a type to the variable 'stypy_return_type' (line 197)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 197, 8), 'stypy_return_type', False_380084)
    # SSA join for try-except statement (line 194)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'isintlike(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'isintlike' in the type store
    # Getting the type of 'stypy_return_type' (line 188)
    stypy_return_type_380085 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 188, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_380085)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'isintlike'
    return stypy_return_type_380085

# Assigning a type to the variable 'isintlike' (line 188)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 188, 0), 'isintlike', isintlike)

@norecursion
def isshape(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'isshape'
    module_type_store = module_type_store.open_function_context('isshape', 200, 0, False)
    
    # Passed parameters checking function
    isshape.stypy_localization = localization
    isshape.stypy_type_of_self = None
    isshape.stypy_type_store = module_type_store
    isshape.stypy_function_name = 'isshape'
    isshape.stypy_param_names_list = ['x']
    isshape.stypy_varargs_param_name = None
    isshape.stypy_kwargs_param_name = None
    isshape.stypy_call_defaults = defaults
    isshape.stypy_call_varargs = varargs
    isshape.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'isshape', ['x'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'isshape', localization, ['x'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'isshape(...)' code ##################

    str_380086 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 202, (-1)), 'str', 'Is x a valid 2-tuple of dimensions?\n    ')
    
    
    # SSA begins for try-except statement (line 203)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
    
    # Assigning a Name to a Tuple (line 205):
    
    # Assigning a Subscript to a Name (line 205):
    
    # Obtaining the type of the subscript
    int_380087 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 205, 8), 'int')
    # Getting the type of 'x' (line 205)
    x_380088 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 17), 'x')
    # Obtaining the member '__getitem__' of a type (line 205)
    getitem___380089 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 205, 8), x_380088, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 205)
    subscript_call_result_380090 = invoke(stypy.reporting.localization.Localization(__file__, 205, 8), getitem___380089, int_380087)
    
    # Assigning a type to the variable 'tuple_var_assignment_379664' (line 205)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 205, 8), 'tuple_var_assignment_379664', subscript_call_result_380090)
    
    # Assigning a Subscript to a Name (line 205):
    
    # Obtaining the type of the subscript
    int_380091 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 205, 8), 'int')
    # Getting the type of 'x' (line 205)
    x_380092 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 17), 'x')
    # Obtaining the member '__getitem__' of a type (line 205)
    getitem___380093 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 205, 8), x_380092, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 205)
    subscript_call_result_380094 = invoke(stypy.reporting.localization.Localization(__file__, 205, 8), getitem___380093, int_380091)
    
    # Assigning a type to the variable 'tuple_var_assignment_379665' (line 205)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 205, 8), 'tuple_var_assignment_379665', subscript_call_result_380094)
    
    # Assigning a Name to a Name (line 205):
    # Getting the type of 'tuple_var_assignment_379664' (line 205)
    tuple_var_assignment_379664_380095 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 8), 'tuple_var_assignment_379664')
    # Assigning a type to the variable 'M' (line 205)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 205, 9), 'M', tuple_var_assignment_379664_380095)
    
    # Assigning a Name to a Name (line 205):
    # Getting the type of 'tuple_var_assignment_379665' (line 205)
    tuple_var_assignment_379665_380096 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 8), 'tuple_var_assignment_379665')
    # Assigning a type to the variable 'N' (line 205)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 205, 12), 'N', tuple_var_assignment_379665_380096)
    # SSA branch for the except part of a try statement (line 203)
    # SSA branch for the except '<any exception>' branch of a try statement (line 203)
    module_type_store.open_ssa_branch('except')
    # Getting the type of 'False' (line 207)
    False_380097 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 207, 15), 'False')
    # Assigning a type to the variable 'stypy_return_type' (line 207)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 207, 8), 'stypy_return_type', False_380097)
    # SSA branch for the else branch of a try statement (line 203)
    module_type_store.open_ssa_branch('except else')
    
    
    # Evaluating a boolean operation
    
    # Call to isintlike(...): (line 209)
    # Processing the call arguments (line 209)
    # Getting the type of 'M' (line 209)
    M_380099 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 21), 'M', False)
    # Processing the call keyword arguments (line 209)
    kwargs_380100 = {}
    # Getting the type of 'isintlike' (line 209)
    isintlike_380098 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 11), 'isintlike', False)
    # Calling isintlike(args, kwargs) (line 209)
    isintlike_call_result_380101 = invoke(stypy.reporting.localization.Localization(__file__, 209, 11), isintlike_380098, *[M_380099], **kwargs_380100)
    
    
    # Call to isintlike(...): (line 209)
    # Processing the call arguments (line 209)
    # Getting the type of 'N' (line 209)
    N_380103 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 38), 'N', False)
    # Processing the call keyword arguments (line 209)
    kwargs_380104 = {}
    # Getting the type of 'isintlike' (line 209)
    isintlike_380102 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 28), 'isintlike', False)
    # Calling isintlike(args, kwargs) (line 209)
    isintlike_call_result_380105 = invoke(stypy.reporting.localization.Localization(__file__, 209, 28), isintlike_380102, *[N_380103], **kwargs_380104)
    
    # Applying the binary operator 'and' (line 209)
    result_and_keyword_380106 = python_operator(stypy.reporting.localization.Localization(__file__, 209, 11), 'and', isintlike_call_result_380101, isintlike_call_result_380105)
    
    # Testing the type of an if condition (line 209)
    if_condition_380107 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 209, 8), result_and_keyword_380106)
    # Assigning a type to the variable 'if_condition_380107' (line 209)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 209, 8), 'if_condition_380107', if_condition_380107)
    # SSA begins for if statement (line 209)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    
    # Evaluating a boolean operation
    
    
    # Call to ndim(...): (line 210)
    # Processing the call arguments (line 210)
    # Getting the type of 'M' (line 210)
    M_380110 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 23), 'M', False)
    # Processing the call keyword arguments (line 210)
    kwargs_380111 = {}
    # Getting the type of 'np' (line 210)
    np_380108 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 15), 'np', False)
    # Obtaining the member 'ndim' of a type (line 210)
    ndim_380109 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 210, 15), np_380108, 'ndim')
    # Calling ndim(args, kwargs) (line 210)
    ndim_call_result_380112 = invoke(stypy.reporting.localization.Localization(__file__, 210, 15), ndim_380109, *[M_380110], **kwargs_380111)
    
    int_380113 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 210, 29), 'int')
    # Applying the binary operator '==' (line 210)
    result_eq_380114 = python_operator(stypy.reporting.localization.Localization(__file__, 210, 15), '==', ndim_call_result_380112, int_380113)
    
    
    
    # Call to ndim(...): (line 210)
    # Processing the call arguments (line 210)
    # Getting the type of 'N' (line 210)
    N_380117 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 43), 'N', False)
    # Processing the call keyword arguments (line 210)
    kwargs_380118 = {}
    # Getting the type of 'np' (line 210)
    np_380115 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 35), 'np', False)
    # Obtaining the member 'ndim' of a type (line 210)
    ndim_380116 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 210, 35), np_380115, 'ndim')
    # Calling ndim(args, kwargs) (line 210)
    ndim_call_result_380119 = invoke(stypy.reporting.localization.Localization(__file__, 210, 35), ndim_380116, *[N_380117], **kwargs_380118)
    
    int_380120 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 210, 49), 'int')
    # Applying the binary operator '==' (line 210)
    result_eq_380121 = python_operator(stypy.reporting.localization.Localization(__file__, 210, 35), '==', ndim_call_result_380119, int_380120)
    
    # Applying the binary operator 'and' (line 210)
    result_and_keyword_380122 = python_operator(stypy.reporting.localization.Localization(__file__, 210, 15), 'and', result_eq_380114, result_eq_380121)
    
    # Testing the type of an if condition (line 210)
    if_condition_380123 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 210, 12), result_and_keyword_380122)
    # Assigning a type to the variable 'if_condition_380123' (line 210)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 210, 12), 'if_condition_380123', if_condition_380123)
    # SSA begins for if statement (line 210)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # Getting the type of 'True' (line 211)
    True_380124 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 211, 23), 'True')
    # Assigning a type to the variable 'stypy_return_type' (line 211)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 211, 16), 'stypy_return_type', True_380124)
    # SSA join for if statement (line 210)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 209)
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'False' (line 212)
    False_380125 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 15), 'False')
    # Assigning a type to the variable 'stypy_return_type' (line 212)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 212, 8), 'stypy_return_type', False_380125)
    # SSA join for try-except statement (line 203)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'isshape(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'isshape' in the type store
    # Getting the type of 'stypy_return_type' (line 200)
    stypy_return_type_380126 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_380126)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'isshape'
    return stypy_return_type_380126

# Assigning a type to the variable 'isshape' (line 200)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 200, 0), 'isshape', isshape)

@norecursion
def issequence(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'issequence'
    module_type_store = module_type_store.open_function_context('issequence', 215, 0, False)
    
    # Passed parameters checking function
    issequence.stypy_localization = localization
    issequence.stypy_type_of_self = None
    issequence.stypy_type_store = module_type_store
    issequence.stypy_function_name = 'issequence'
    issequence.stypy_param_names_list = ['t']
    issequence.stypy_varargs_param_name = None
    issequence.stypy_kwargs_param_name = None
    issequence.stypy_call_defaults = defaults
    issequence.stypy_call_varargs = varargs
    issequence.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'issequence', ['t'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'issequence', localization, ['t'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'issequence(...)' code ##################

    
    # Evaluating a boolean operation
    
    # Evaluating a boolean operation
    
    # Call to isinstance(...): (line 216)
    # Processing the call arguments (line 216)
    # Getting the type of 't' (line 216)
    t_380128 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 216, 24), 't', False)
    
    # Obtaining an instance of the builtin type 'tuple' (line 216)
    tuple_380129 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 216, 28), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 216)
    # Adding element type (line 216)
    # Getting the type of 'list' (line 216)
    list_380130 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 216, 28), 'list', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 216, 28), tuple_380129, list_380130)
    # Adding element type (line 216)
    # Getting the type of 'tuple' (line 216)
    tuple_380131 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 216, 34), 'tuple', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 216, 28), tuple_380129, tuple_380131)
    
    # Processing the call keyword arguments (line 216)
    kwargs_380132 = {}
    # Getting the type of 'isinstance' (line 216)
    isinstance_380127 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 216, 13), 'isinstance', False)
    # Calling isinstance(args, kwargs) (line 216)
    isinstance_call_result_380133 = invoke(stypy.reporting.localization.Localization(__file__, 216, 13), isinstance_380127, *[t_380128, tuple_380129], **kwargs_380132)
    
    
    # Evaluating a boolean operation
    
    
    # Call to len(...): (line 217)
    # Processing the call arguments (line 217)
    # Getting the type of 't' (line 217)
    t_380135 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 17), 't', False)
    # Processing the call keyword arguments (line 217)
    kwargs_380136 = {}
    # Getting the type of 'len' (line 217)
    len_380134 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 13), 'len', False)
    # Calling len(args, kwargs) (line 217)
    len_call_result_380137 = invoke(stypy.reporting.localization.Localization(__file__, 217, 13), len_380134, *[t_380135], **kwargs_380136)
    
    int_380138 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 217, 23), 'int')
    # Applying the binary operator '==' (line 217)
    result_eq_380139 = python_operator(stypy.reporting.localization.Localization(__file__, 217, 13), '==', len_call_result_380137, int_380138)
    
    
    # Call to isscalar(...): (line 217)
    # Processing the call arguments (line 217)
    
    # Obtaining the type of the subscript
    int_380142 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 217, 42), 'int')
    # Getting the type of 't' (line 217)
    t_380143 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 40), 't', False)
    # Obtaining the member '__getitem__' of a type (line 217)
    getitem___380144 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 217, 40), t_380143, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 217)
    subscript_call_result_380145 = invoke(stypy.reporting.localization.Localization(__file__, 217, 40), getitem___380144, int_380142)
    
    # Processing the call keyword arguments (line 217)
    kwargs_380146 = {}
    # Getting the type of 'np' (line 217)
    np_380140 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 28), 'np', False)
    # Obtaining the member 'isscalar' of a type (line 217)
    isscalar_380141 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 217, 28), np_380140, 'isscalar')
    # Calling isscalar(args, kwargs) (line 217)
    isscalar_call_result_380147 = invoke(stypy.reporting.localization.Localization(__file__, 217, 28), isscalar_380141, *[subscript_call_result_380145], **kwargs_380146)
    
    # Applying the binary operator 'or' (line 217)
    result_or_keyword_380148 = python_operator(stypy.reporting.localization.Localization(__file__, 217, 13), 'or', result_eq_380139, isscalar_call_result_380147)
    
    # Applying the binary operator 'and' (line 216)
    result_and_keyword_380149 = python_operator(stypy.reporting.localization.Localization(__file__, 216, 13), 'and', isinstance_call_result_380133, result_or_keyword_380148)
    
    
    # Evaluating a boolean operation
    
    # Call to isinstance(...): (line 218)
    # Processing the call arguments (line 218)
    # Getting the type of 't' (line 218)
    t_380151 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 218, 24), 't', False)
    # Getting the type of 'np' (line 218)
    np_380152 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 218, 27), 'np', False)
    # Obtaining the member 'ndarray' of a type (line 218)
    ndarray_380153 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 218, 27), np_380152, 'ndarray')
    # Processing the call keyword arguments (line 218)
    kwargs_380154 = {}
    # Getting the type of 'isinstance' (line 218)
    isinstance_380150 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 218, 13), 'isinstance', False)
    # Calling isinstance(args, kwargs) (line 218)
    isinstance_call_result_380155 = invoke(stypy.reporting.localization.Localization(__file__, 218, 13), isinstance_380150, *[t_380151, ndarray_380153], **kwargs_380154)
    
    
    # Getting the type of 't' (line 218)
    t_380156 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 218, 44), 't')
    # Obtaining the member 'ndim' of a type (line 218)
    ndim_380157 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 218, 44), t_380156, 'ndim')
    int_380158 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 218, 54), 'int')
    # Applying the binary operator '==' (line 218)
    result_eq_380159 = python_operator(stypy.reporting.localization.Localization(__file__, 218, 44), '==', ndim_380157, int_380158)
    
    # Applying the binary operator 'and' (line 218)
    result_and_keyword_380160 = python_operator(stypy.reporting.localization.Localization(__file__, 218, 13), 'and', isinstance_call_result_380155, result_eq_380159)
    
    # Applying the binary operator 'or' (line 216)
    result_or_keyword_380161 = python_operator(stypy.reporting.localization.Localization(__file__, 216, 12), 'or', result_and_keyword_380149, result_and_keyword_380160)
    
    # Assigning a type to the variable 'stypy_return_type' (line 216)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 216, 4), 'stypy_return_type', result_or_keyword_380161)
    
    # ################# End of 'issequence(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'issequence' in the type store
    # Getting the type of 'stypy_return_type' (line 215)
    stypy_return_type_380162 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 215, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_380162)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'issequence'
    return stypy_return_type_380162

# Assigning a type to the variable 'issequence' (line 215)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 215, 0), 'issequence', issequence)

@norecursion
def ismatrix(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'ismatrix'
    module_type_store = module_type_store.open_function_context('ismatrix', 221, 0, False)
    
    # Passed parameters checking function
    ismatrix.stypy_localization = localization
    ismatrix.stypy_type_of_self = None
    ismatrix.stypy_type_store = module_type_store
    ismatrix.stypy_function_name = 'ismatrix'
    ismatrix.stypy_param_names_list = ['t']
    ismatrix.stypy_varargs_param_name = None
    ismatrix.stypy_kwargs_param_name = None
    ismatrix.stypy_call_defaults = defaults
    ismatrix.stypy_call_varargs = varargs
    ismatrix.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'ismatrix', ['t'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'ismatrix', localization, ['t'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'ismatrix(...)' code ##################

    
    # Evaluating a boolean operation
    
    # Evaluating a boolean operation
    
    # Call to isinstance(...): (line 222)
    # Processing the call arguments (line 222)
    # Getting the type of 't' (line 222)
    t_380164 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 222, 24), 't', False)
    
    # Obtaining an instance of the builtin type 'tuple' (line 222)
    tuple_380165 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 222, 28), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 222)
    # Adding element type (line 222)
    # Getting the type of 'list' (line 222)
    list_380166 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 222, 28), 'list', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 222, 28), tuple_380165, list_380166)
    # Adding element type (line 222)
    # Getting the type of 'tuple' (line 222)
    tuple_380167 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 222, 34), 'tuple', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 222, 28), tuple_380165, tuple_380167)
    
    # Processing the call keyword arguments (line 222)
    kwargs_380168 = {}
    # Getting the type of 'isinstance' (line 222)
    isinstance_380163 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 222, 13), 'isinstance', False)
    # Calling isinstance(args, kwargs) (line 222)
    isinstance_call_result_380169 = invoke(stypy.reporting.localization.Localization(__file__, 222, 13), isinstance_380163, *[t_380164, tuple_380165], **kwargs_380168)
    
    
    
    # Call to len(...): (line 223)
    # Processing the call arguments (line 223)
    # Getting the type of 't' (line 223)
    t_380171 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 17), 't', False)
    # Processing the call keyword arguments (line 223)
    kwargs_380172 = {}
    # Getting the type of 'len' (line 223)
    len_380170 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 13), 'len', False)
    # Calling len(args, kwargs) (line 223)
    len_call_result_380173 = invoke(stypy.reporting.localization.Localization(__file__, 223, 13), len_380170, *[t_380171], **kwargs_380172)
    
    int_380174 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 223, 22), 'int')
    # Applying the binary operator '>' (line 223)
    result_gt_380175 = python_operator(stypy.reporting.localization.Localization(__file__, 223, 13), '>', len_call_result_380173, int_380174)
    
    # Applying the binary operator 'and' (line 222)
    result_and_keyword_380176 = python_operator(stypy.reporting.localization.Localization(__file__, 222, 13), 'and', isinstance_call_result_380169, result_gt_380175)
    
    # Call to issequence(...): (line 223)
    # Processing the call arguments (line 223)
    
    # Obtaining the type of the subscript
    int_380178 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 223, 41), 'int')
    # Getting the type of 't' (line 223)
    t_380179 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 39), 't', False)
    # Obtaining the member '__getitem__' of a type (line 223)
    getitem___380180 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 223, 39), t_380179, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 223)
    subscript_call_result_380181 = invoke(stypy.reporting.localization.Localization(__file__, 223, 39), getitem___380180, int_380178)
    
    # Processing the call keyword arguments (line 223)
    kwargs_380182 = {}
    # Getting the type of 'issequence' (line 223)
    issequence_380177 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 28), 'issequence', False)
    # Calling issequence(args, kwargs) (line 223)
    issequence_call_result_380183 = invoke(stypy.reporting.localization.Localization(__file__, 223, 28), issequence_380177, *[subscript_call_result_380181], **kwargs_380182)
    
    # Applying the binary operator 'and' (line 222)
    result_and_keyword_380184 = python_operator(stypy.reporting.localization.Localization(__file__, 222, 13), 'and', result_and_keyword_380176, issequence_call_result_380183)
    
    
    # Evaluating a boolean operation
    
    # Call to isinstance(...): (line 224)
    # Processing the call arguments (line 224)
    # Getting the type of 't' (line 224)
    t_380186 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 224, 24), 't', False)
    # Getting the type of 'np' (line 224)
    np_380187 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 224, 27), 'np', False)
    # Obtaining the member 'ndarray' of a type (line 224)
    ndarray_380188 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 224, 27), np_380187, 'ndarray')
    # Processing the call keyword arguments (line 224)
    kwargs_380189 = {}
    # Getting the type of 'isinstance' (line 224)
    isinstance_380185 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 224, 13), 'isinstance', False)
    # Calling isinstance(args, kwargs) (line 224)
    isinstance_call_result_380190 = invoke(stypy.reporting.localization.Localization(__file__, 224, 13), isinstance_380185, *[t_380186, ndarray_380188], **kwargs_380189)
    
    
    # Getting the type of 't' (line 224)
    t_380191 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 224, 43), 't')
    # Obtaining the member 'ndim' of a type (line 224)
    ndim_380192 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 224, 43), t_380191, 'ndim')
    int_380193 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 224, 53), 'int')
    # Applying the binary operator '==' (line 224)
    result_eq_380194 = python_operator(stypy.reporting.localization.Localization(__file__, 224, 43), '==', ndim_380192, int_380193)
    
    # Applying the binary operator 'and' (line 224)
    result_and_keyword_380195 = python_operator(stypy.reporting.localization.Localization(__file__, 224, 13), 'and', isinstance_call_result_380190, result_eq_380194)
    
    # Applying the binary operator 'or' (line 222)
    result_or_keyword_380196 = python_operator(stypy.reporting.localization.Localization(__file__, 222, 12), 'or', result_and_keyword_380184, result_and_keyword_380195)
    
    # Assigning a type to the variable 'stypy_return_type' (line 222)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 222, 4), 'stypy_return_type', result_or_keyword_380196)
    
    # ################# End of 'ismatrix(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'ismatrix' in the type store
    # Getting the type of 'stypy_return_type' (line 221)
    stypy_return_type_380197 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_380197)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'ismatrix'
    return stypy_return_type_380197

# Assigning a type to the variable 'ismatrix' (line 221)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 221, 0), 'ismatrix', ismatrix)

@norecursion
def isdense(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'isdense'
    module_type_store = module_type_store.open_function_context('isdense', 227, 0, False)
    
    # Passed parameters checking function
    isdense.stypy_localization = localization
    isdense.stypy_type_of_self = None
    isdense.stypy_type_store = module_type_store
    isdense.stypy_function_name = 'isdense'
    isdense.stypy_param_names_list = ['x']
    isdense.stypy_varargs_param_name = None
    isdense.stypy_kwargs_param_name = None
    isdense.stypy_call_defaults = defaults
    isdense.stypy_call_varargs = varargs
    isdense.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'isdense', ['x'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'isdense', localization, ['x'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'isdense(...)' code ##################

    
    # Call to isinstance(...): (line 228)
    # Processing the call arguments (line 228)
    # Getting the type of 'x' (line 228)
    x_380199 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 22), 'x', False)
    # Getting the type of 'np' (line 228)
    np_380200 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 25), 'np', False)
    # Obtaining the member 'ndarray' of a type (line 228)
    ndarray_380201 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 228, 25), np_380200, 'ndarray')
    # Processing the call keyword arguments (line 228)
    kwargs_380202 = {}
    # Getting the type of 'isinstance' (line 228)
    isinstance_380198 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 11), 'isinstance', False)
    # Calling isinstance(args, kwargs) (line 228)
    isinstance_call_result_380203 = invoke(stypy.reporting.localization.Localization(__file__, 228, 11), isinstance_380198, *[x_380199, ndarray_380201], **kwargs_380202)
    
    # Assigning a type to the variable 'stypy_return_type' (line 228)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 228, 4), 'stypy_return_type', isinstance_call_result_380203)
    
    # ################# End of 'isdense(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'isdense' in the type store
    # Getting the type of 'stypy_return_type' (line 227)
    stypy_return_type_380204 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_380204)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'isdense'
    return stypy_return_type_380204

# Assigning a type to the variable 'isdense' (line 227)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 227, 0), 'isdense', isdense)

@norecursion
def validateaxis(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'validateaxis'
    module_type_store = module_type_store.open_function_context('validateaxis', 231, 0, False)
    
    # Passed parameters checking function
    validateaxis.stypy_localization = localization
    validateaxis.stypy_type_of_self = None
    validateaxis.stypy_type_store = module_type_store
    validateaxis.stypy_function_name = 'validateaxis'
    validateaxis.stypy_param_names_list = ['axis']
    validateaxis.stypy_varargs_param_name = None
    validateaxis.stypy_kwargs_param_name = None
    validateaxis.stypy_call_defaults = defaults
    validateaxis.stypy_call_varargs = varargs
    validateaxis.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'validateaxis', ['axis'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'validateaxis', localization, ['axis'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'validateaxis(...)' code ##################

    
    # Type idiom detected: calculating its left and rigth part (line 232)
    # Getting the type of 'axis' (line 232)
    axis_380205 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 4), 'axis')
    # Getting the type of 'None' (line 232)
    None_380206 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 19), 'None')
    
    (may_be_380207, more_types_in_union_380208) = may_not_be_none(axis_380205, None_380206)

    if may_be_380207:

        if more_types_in_union_380208:
            # Runtime conditional SSA (line 232)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a Call to a Name (line 233):
        
        # Assigning a Call to a Name (line 233):
        
        # Call to type(...): (line 233)
        # Processing the call arguments (line 233)
        # Getting the type of 'axis' (line 233)
        axis_380210 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 233, 25), 'axis', False)
        # Processing the call keyword arguments (line 233)
        kwargs_380211 = {}
        # Getting the type of 'type' (line 233)
        type_380209 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 233, 20), 'type', False)
        # Calling type(args, kwargs) (line 233)
        type_call_result_380212 = invoke(stypy.reporting.localization.Localization(__file__, 233, 20), type_380209, *[axis_380210], **kwargs_380211)
        
        # Assigning a type to the variable 'axis_type' (line 233)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 233, 8), 'axis_type', type_call_result_380212)
        
        
        # Getting the type of 'axis_type' (line 239)
        axis_type_380213 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 239, 11), 'axis_type')
        # Getting the type of 'tuple' (line 239)
        tuple_380214 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 239, 24), 'tuple')
        # Applying the binary operator '==' (line 239)
        result_eq_380215 = python_operator(stypy.reporting.localization.Localization(__file__, 239, 11), '==', axis_type_380213, tuple_380214)
        
        # Testing the type of an if condition (line 239)
        if_condition_380216 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 239, 8), result_eq_380215)
        # Assigning a type to the variable 'if_condition_380216' (line 239)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 239, 8), 'if_condition_380216', if_condition_380216)
        # SSA begins for if statement (line 239)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to TypeError(...): (line 240)
        # Processing the call arguments (line 240)
        str_380218 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 240, 29), 'str', "Tuples are not accepted for the 'axis' parameter. Please pass in one of the following: {-2, -1, 0, 1, None}.")
        # Processing the call keyword arguments (line 240)
        kwargs_380219 = {}
        # Getting the type of 'TypeError' (line 240)
        TypeError_380217 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 18), 'TypeError', False)
        # Calling TypeError(args, kwargs) (line 240)
        TypeError_call_result_380220 = invoke(stypy.reporting.localization.Localization(__file__, 240, 18), TypeError_380217, *[str_380218], **kwargs_380219)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 240, 12), TypeError_call_result_380220, 'raise parameter', BaseException)
        # SSA join for if statement (line 239)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        
        # Call to issubdtype(...): (line 246)
        # Processing the call arguments (line 246)
        
        # Call to dtype(...): (line 246)
        # Processing the call arguments (line 246)
        # Getting the type of 'axis_type' (line 246)
        axis_type_380225 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 38), 'axis_type', False)
        # Processing the call keyword arguments (line 246)
        kwargs_380226 = {}
        # Getting the type of 'np' (line 246)
        np_380223 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 29), 'np', False)
        # Obtaining the member 'dtype' of a type (line 246)
        dtype_380224 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 246, 29), np_380223, 'dtype')
        # Calling dtype(args, kwargs) (line 246)
        dtype_call_result_380227 = invoke(stypy.reporting.localization.Localization(__file__, 246, 29), dtype_380224, *[axis_type_380225], **kwargs_380226)
        
        # Getting the type of 'np' (line 246)
        np_380228 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 50), 'np', False)
        # Obtaining the member 'integer' of a type (line 246)
        integer_380229 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 246, 50), np_380228, 'integer')
        # Processing the call keyword arguments (line 246)
        kwargs_380230 = {}
        # Getting the type of 'np' (line 246)
        np_380221 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 15), 'np', False)
        # Obtaining the member 'issubdtype' of a type (line 246)
        issubdtype_380222 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 246, 15), np_380221, 'issubdtype')
        # Calling issubdtype(args, kwargs) (line 246)
        issubdtype_call_result_380231 = invoke(stypy.reporting.localization.Localization(__file__, 246, 15), issubdtype_380222, *[dtype_call_result_380227, integer_380229], **kwargs_380230)
        
        # Applying the 'not' unary operator (line 246)
        result_not__380232 = python_operator(stypy.reporting.localization.Localization(__file__, 246, 11), 'not', issubdtype_call_result_380231)
        
        # Testing the type of an if condition (line 246)
        if_condition_380233 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 246, 8), result_not__380232)
        # Assigning a type to the variable 'if_condition_380233' (line 246)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 246, 8), 'if_condition_380233', if_condition_380233)
        # SSA begins for if statement (line 246)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to TypeError(...): (line 247)
        # Processing the call arguments (line 247)
        
        # Call to format(...): (line 247)
        # Processing the call keyword arguments (line 247)
        # Getting the type of 'axis_type' (line 248)
        axis_type_380237 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 248, 41), 'axis_type', False)
        # Obtaining the member '__name__' of a type (line 248)
        name___380238 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 248, 41), axis_type_380237, '__name__')
        keyword_380239 = name___380238
        kwargs_380240 = {'name': keyword_380239}
        str_380235 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 247, 28), 'str', 'axis must be an integer, not {name}')
        # Obtaining the member 'format' of a type (line 247)
        format_380236 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 247, 28), str_380235, 'format')
        # Calling format(args, kwargs) (line 247)
        format_call_result_380241 = invoke(stypy.reporting.localization.Localization(__file__, 247, 28), format_380236, *[], **kwargs_380240)
        
        # Processing the call keyword arguments (line 247)
        kwargs_380242 = {}
        # Getting the type of 'TypeError' (line 247)
        TypeError_380234 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 247, 18), 'TypeError', False)
        # Calling TypeError(args, kwargs) (line 247)
        TypeError_call_result_380243 = invoke(stypy.reporting.localization.Localization(__file__, 247, 18), TypeError_380234, *[format_call_result_380241], **kwargs_380242)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 247, 12), TypeError_call_result_380243, 'raise parameter', BaseException)
        # SSA join for if statement (line 246)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        
        int_380244 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 250, 16), 'int')
        # Getting the type of 'axis' (line 250)
        axis_380245 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 250, 22), 'axis')
        # Applying the binary operator '<=' (line 250)
        result_le_380246 = python_operator(stypy.reporting.localization.Localization(__file__, 250, 16), '<=', int_380244, axis_380245)
        int_380247 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 250, 30), 'int')
        # Applying the binary operator '<=' (line 250)
        result_le_380248 = python_operator(stypy.reporting.localization.Localization(__file__, 250, 16), '<=', axis_380245, int_380247)
        # Applying the binary operator '&' (line 250)
        result_and__380249 = python_operator(stypy.reporting.localization.Localization(__file__, 250, 16), '&', result_le_380246, result_le_380248)
        
        # Applying the 'not' unary operator (line 250)
        result_not__380250 = python_operator(stypy.reporting.localization.Localization(__file__, 250, 11), 'not', result_and__380249)
        
        # Testing the type of an if condition (line 250)
        if_condition_380251 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 250, 8), result_not__380250)
        # Assigning a type to the variable 'if_condition_380251' (line 250)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 250, 8), 'if_condition_380251', if_condition_380251)
        # SSA begins for if statement (line 250)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to ValueError(...): (line 251)
        # Processing the call arguments (line 251)
        str_380253 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 251, 29), 'str', 'axis out of range')
        # Processing the call keyword arguments (line 251)
        kwargs_380254 = {}
        # Getting the type of 'ValueError' (line 251)
        ValueError_380252 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 251, 18), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 251)
        ValueError_call_result_380255 = invoke(stypy.reporting.localization.Localization(__file__, 251, 18), ValueError_380252, *[str_380253], **kwargs_380254)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 251, 12), ValueError_call_result_380255, 'raise parameter', BaseException)
        # SSA join for if statement (line 250)
        module_type_store = module_type_store.join_ssa_context()
        

        if more_types_in_union_380208:
            # SSA join for if statement (line 232)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # ################# End of 'validateaxis(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'validateaxis' in the type store
    # Getting the type of 'stypy_return_type' (line 231)
    stypy_return_type_380256 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 231, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_380256)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'validateaxis'
    return stypy_return_type_380256

# Assigning a type to the variable 'validateaxis' (line 231)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 231, 0), 'validateaxis', validateaxis)
# Declaration of the 'IndexMixin' class

class IndexMixin(object, ):
    str_380257 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 257, (-1)), 'str', '\n    This class simply exists to hold the methods necessary for fancy indexing.\n    ')

    @norecursion
    def _slicetoarange(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_slicetoarange'
        module_type_store = module_type_store.open_function_context('_slicetoarange', 258, 4, False)
        # Assigning a type to the variable 'self' (line 259)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 259, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        IndexMixin._slicetoarange.__dict__.__setitem__('stypy_localization', localization)
        IndexMixin._slicetoarange.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        IndexMixin._slicetoarange.__dict__.__setitem__('stypy_type_store', module_type_store)
        IndexMixin._slicetoarange.__dict__.__setitem__('stypy_function_name', 'IndexMixin._slicetoarange')
        IndexMixin._slicetoarange.__dict__.__setitem__('stypy_param_names_list', ['j', 'shape'])
        IndexMixin._slicetoarange.__dict__.__setitem__('stypy_varargs_param_name', None)
        IndexMixin._slicetoarange.__dict__.__setitem__('stypy_kwargs_param_name', None)
        IndexMixin._slicetoarange.__dict__.__setitem__('stypy_call_defaults', defaults)
        IndexMixin._slicetoarange.__dict__.__setitem__('stypy_call_varargs', varargs)
        IndexMixin._slicetoarange.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        IndexMixin._slicetoarange.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'IndexMixin._slicetoarange', ['j', 'shape'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_slicetoarange', localization, ['j', 'shape'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_slicetoarange(...)' code ##################

        str_380258 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 261, (-1)), 'str', ' Given a slice object, use numpy arange to change it to a 1D\n        array.\n        ')
        
        # Assigning a Call to a Tuple (line 262):
        
        # Assigning a Subscript to a Name (line 262):
        
        # Obtaining the type of the subscript
        int_380259 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 262, 8), 'int')
        
        # Call to indices(...): (line 262)
        # Processing the call arguments (line 262)
        # Getting the type of 'shape' (line 262)
        shape_380262 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 262, 38), 'shape', False)
        # Processing the call keyword arguments (line 262)
        kwargs_380263 = {}
        # Getting the type of 'j' (line 262)
        j_380260 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 262, 28), 'j', False)
        # Obtaining the member 'indices' of a type (line 262)
        indices_380261 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 262, 28), j_380260, 'indices')
        # Calling indices(args, kwargs) (line 262)
        indices_call_result_380264 = invoke(stypy.reporting.localization.Localization(__file__, 262, 28), indices_380261, *[shape_380262], **kwargs_380263)
        
        # Obtaining the member '__getitem__' of a type (line 262)
        getitem___380265 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 262, 8), indices_call_result_380264, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 262)
        subscript_call_result_380266 = invoke(stypy.reporting.localization.Localization(__file__, 262, 8), getitem___380265, int_380259)
        
        # Assigning a type to the variable 'tuple_var_assignment_379666' (line 262)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 262, 8), 'tuple_var_assignment_379666', subscript_call_result_380266)
        
        # Assigning a Subscript to a Name (line 262):
        
        # Obtaining the type of the subscript
        int_380267 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 262, 8), 'int')
        
        # Call to indices(...): (line 262)
        # Processing the call arguments (line 262)
        # Getting the type of 'shape' (line 262)
        shape_380270 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 262, 38), 'shape', False)
        # Processing the call keyword arguments (line 262)
        kwargs_380271 = {}
        # Getting the type of 'j' (line 262)
        j_380268 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 262, 28), 'j', False)
        # Obtaining the member 'indices' of a type (line 262)
        indices_380269 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 262, 28), j_380268, 'indices')
        # Calling indices(args, kwargs) (line 262)
        indices_call_result_380272 = invoke(stypy.reporting.localization.Localization(__file__, 262, 28), indices_380269, *[shape_380270], **kwargs_380271)
        
        # Obtaining the member '__getitem__' of a type (line 262)
        getitem___380273 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 262, 8), indices_call_result_380272, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 262)
        subscript_call_result_380274 = invoke(stypy.reporting.localization.Localization(__file__, 262, 8), getitem___380273, int_380267)
        
        # Assigning a type to the variable 'tuple_var_assignment_379667' (line 262)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 262, 8), 'tuple_var_assignment_379667', subscript_call_result_380274)
        
        # Assigning a Subscript to a Name (line 262):
        
        # Obtaining the type of the subscript
        int_380275 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 262, 8), 'int')
        
        # Call to indices(...): (line 262)
        # Processing the call arguments (line 262)
        # Getting the type of 'shape' (line 262)
        shape_380278 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 262, 38), 'shape', False)
        # Processing the call keyword arguments (line 262)
        kwargs_380279 = {}
        # Getting the type of 'j' (line 262)
        j_380276 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 262, 28), 'j', False)
        # Obtaining the member 'indices' of a type (line 262)
        indices_380277 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 262, 28), j_380276, 'indices')
        # Calling indices(args, kwargs) (line 262)
        indices_call_result_380280 = invoke(stypy.reporting.localization.Localization(__file__, 262, 28), indices_380277, *[shape_380278], **kwargs_380279)
        
        # Obtaining the member '__getitem__' of a type (line 262)
        getitem___380281 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 262, 8), indices_call_result_380280, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 262)
        subscript_call_result_380282 = invoke(stypy.reporting.localization.Localization(__file__, 262, 8), getitem___380281, int_380275)
        
        # Assigning a type to the variable 'tuple_var_assignment_379668' (line 262)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 262, 8), 'tuple_var_assignment_379668', subscript_call_result_380282)
        
        # Assigning a Name to a Name (line 262):
        # Getting the type of 'tuple_var_assignment_379666' (line 262)
        tuple_var_assignment_379666_380283 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 262, 8), 'tuple_var_assignment_379666')
        # Assigning a type to the variable 'start' (line 262)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 262, 8), 'start', tuple_var_assignment_379666_380283)
        
        # Assigning a Name to a Name (line 262):
        # Getting the type of 'tuple_var_assignment_379667' (line 262)
        tuple_var_assignment_379667_380284 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 262, 8), 'tuple_var_assignment_379667')
        # Assigning a type to the variable 'stop' (line 262)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 262, 15), 'stop', tuple_var_assignment_379667_380284)
        
        # Assigning a Name to a Name (line 262):
        # Getting the type of 'tuple_var_assignment_379668' (line 262)
        tuple_var_assignment_379668_380285 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 262, 8), 'tuple_var_assignment_379668')
        # Assigning a type to the variable 'step' (line 262)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 262, 21), 'step', tuple_var_assignment_379668_380285)
        
        # Call to arange(...): (line 263)
        # Processing the call arguments (line 263)
        # Getting the type of 'start' (line 263)
        start_380288 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 263, 25), 'start', False)
        # Getting the type of 'stop' (line 263)
        stop_380289 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 263, 32), 'stop', False)
        # Getting the type of 'step' (line 263)
        step_380290 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 263, 38), 'step', False)
        # Processing the call keyword arguments (line 263)
        kwargs_380291 = {}
        # Getting the type of 'np' (line 263)
        np_380286 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 263, 15), 'np', False)
        # Obtaining the member 'arange' of a type (line 263)
        arange_380287 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 263, 15), np_380286, 'arange')
        # Calling arange(args, kwargs) (line 263)
        arange_call_result_380292 = invoke(stypy.reporting.localization.Localization(__file__, 263, 15), arange_380287, *[start_380288, stop_380289, step_380290], **kwargs_380291)
        
        # Assigning a type to the variable 'stypy_return_type' (line 263)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 263, 8), 'stypy_return_type', arange_call_result_380292)
        
        # ################# End of '_slicetoarange(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_slicetoarange' in the type store
        # Getting the type of 'stypy_return_type' (line 258)
        stypy_return_type_380293 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 258, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_380293)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_slicetoarange'
        return stypy_return_type_380293


    @norecursion
    def _unpack_index(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_unpack_index'
        module_type_store = module_type_store.open_function_context('_unpack_index', 265, 4, False)
        # Assigning a type to the variable 'self' (line 266)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 266, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        IndexMixin._unpack_index.__dict__.__setitem__('stypy_localization', localization)
        IndexMixin._unpack_index.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        IndexMixin._unpack_index.__dict__.__setitem__('stypy_type_store', module_type_store)
        IndexMixin._unpack_index.__dict__.__setitem__('stypy_function_name', 'IndexMixin._unpack_index')
        IndexMixin._unpack_index.__dict__.__setitem__('stypy_param_names_list', ['index'])
        IndexMixin._unpack_index.__dict__.__setitem__('stypy_varargs_param_name', None)
        IndexMixin._unpack_index.__dict__.__setitem__('stypy_kwargs_param_name', None)
        IndexMixin._unpack_index.__dict__.__setitem__('stypy_call_defaults', defaults)
        IndexMixin._unpack_index.__dict__.__setitem__('stypy_call_varargs', varargs)
        IndexMixin._unpack_index.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        IndexMixin._unpack_index.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'IndexMixin._unpack_index', ['index'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_unpack_index', localization, ['index'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_unpack_index(...)' code ##################

        str_380294 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 268, (-1)), 'str', ' Parse index. Always return a tuple of the form (row, col).\n        Where row/col is a integer, slice, or array of integers.\n        ')
        stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 270, 8))
        
        # 'from scipy.sparse.base import spmatrix' statement (line 270)
        update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/sparse/')
        import_380295 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 270, 8), 'scipy.sparse.base')

        if (type(import_380295) is not StypyTypeError):

            if (import_380295 != 'pyd_module'):
                __import__(import_380295)
                sys_modules_380296 = sys.modules[import_380295]
                import_from_module(stypy.reporting.localization.Localization(__file__, 270, 8), 'scipy.sparse.base', sys_modules_380296.module_type_store, module_type_store, ['spmatrix'])
                nest_module(stypy.reporting.localization.Localization(__file__, 270, 8), __file__, sys_modules_380296, sys_modules_380296.module_type_store, module_type_store)
            else:
                from scipy.sparse.base import spmatrix

                import_from_module(stypy.reporting.localization.Localization(__file__, 270, 8), 'scipy.sparse.base', None, module_type_store, ['spmatrix'], [spmatrix])

        else:
            # Assigning a type to the variable 'scipy.sparse.base' (line 270)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 270, 8), 'scipy.sparse.base', import_380295)

        remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/sparse/')
        
        
        
        # Evaluating a boolean operation
        
        # Call to isinstance(...): (line 271)
        # Processing the call arguments (line 271)
        # Getting the type of 'index' (line 271)
        index_380298 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 271, 23), 'index', False)
        
        # Obtaining an instance of the builtin type 'tuple' (line 271)
        tuple_380299 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 271, 31), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 271)
        # Adding element type (line 271)
        # Getting the type of 'spmatrix' (line 271)
        spmatrix_380300 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 271, 31), 'spmatrix', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 271, 31), tuple_380299, spmatrix_380300)
        # Adding element type (line 271)
        # Getting the type of 'np' (line 271)
        np_380301 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 271, 41), 'np', False)
        # Obtaining the member 'ndarray' of a type (line 271)
        ndarray_380302 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 271, 41), np_380301, 'ndarray')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 271, 31), tuple_380299, ndarray_380302)
        
        # Processing the call keyword arguments (line 271)
        kwargs_380303 = {}
        # Getting the type of 'isinstance' (line 271)
        isinstance_380297 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 271, 12), 'isinstance', False)
        # Calling isinstance(args, kwargs) (line 271)
        isinstance_call_result_380304 = invoke(stypy.reporting.localization.Localization(__file__, 271, 12), isinstance_380297, *[index_380298, tuple_380299], **kwargs_380303)
        
        
        # Getting the type of 'index' (line 272)
        index_380305 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 272, 12), 'index')
        # Obtaining the member 'ndim' of a type (line 272)
        ndim_380306 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 272, 12), index_380305, 'ndim')
        int_380307 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 272, 26), 'int')
        # Applying the binary operator '==' (line 272)
        result_eq_380308 = python_operator(stypy.reporting.localization.Localization(__file__, 272, 12), '==', ndim_380306, int_380307)
        
        # Applying the binary operator 'and' (line 271)
        result_and_keyword_380309 = python_operator(stypy.reporting.localization.Localization(__file__, 271, 12), 'and', isinstance_call_result_380304, result_eq_380308)
        
        # Getting the type of 'index' (line 272)
        index_380310 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 272, 33), 'index')
        # Obtaining the member 'dtype' of a type (line 272)
        dtype_380311 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 272, 33), index_380310, 'dtype')
        # Obtaining the member 'kind' of a type (line 272)
        kind_380312 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 272, 33), dtype_380311, 'kind')
        str_380313 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 272, 53), 'str', 'b')
        # Applying the binary operator '==' (line 272)
        result_eq_380314 = python_operator(stypy.reporting.localization.Localization(__file__, 272, 33), '==', kind_380312, str_380313)
        
        # Applying the binary operator 'and' (line 271)
        result_and_keyword_380315 = python_operator(stypy.reporting.localization.Localization(__file__, 271, 12), 'and', result_and_keyword_380309, result_eq_380314)
        
        # Testing the type of an if condition (line 271)
        if_condition_380316 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 271, 8), result_and_keyword_380315)
        # Assigning a type to the variable 'if_condition_380316' (line 271)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 271, 8), 'if_condition_380316', if_condition_380316)
        # SSA begins for if statement (line 271)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to nonzero(...): (line 273)
        # Processing the call keyword arguments (line 273)
        kwargs_380319 = {}
        # Getting the type of 'index' (line 273)
        index_380317 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 273, 23), 'index', False)
        # Obtaining the member 'nonzero' of a type (line 273)
        nonzero_380318 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 273, 23), index_380317, 'nonzero')
        # Calling nonzero(args, kwargs) (line 273)
        nonzero_call_result_380320 = invoke(stypy.reporting.localization.Localization(__file__, 273, 23), nonzero_380318, *[], **kwargs_380319)
        
        # Assigning a type to the variable 'stypy_return_type' (line 273)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 273, 16), 'stypy_return_type', nonzero_call_result_380320)
        # SSA join for if statement (line 271)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Name (line 276):
        
        # Assigning a Call to a Name (line 276):
        
        # Call to _check_ellipsis(...): (line 276)
        # Processing the call arguments (line 276)
        # Getting the type of 'index' (line 276)
        index_380323 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 276, 37), 'index', False)
        # Processing the call keyword arguments (line 276)
        kwargs_380324 = {}
        # Getting the type of 'self' (line 276)
        self_380321 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 276, 16), 'self', False)
        # Obtaining the member '_check_ellipsis' of a type (line 276)
        _check_ellipsis_380322 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 276, 16), self_380321, '_check_ellipsis')
        # Calling _check_ellipsis(args, kwargs) (line 276)
        _check_ellipsis_call_result_380325 = invoke(stypy.reporting.localization.Localization(__file__, 276, 16), _check_ellipsis_380322, *[index_380323], **kwargs_380324)
        
        # Assigning a type to the variable 'index' (line 276)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 276, 8), 'index', _check_ellipsis_call_result_380325)
        
        # Type idiom detected: calculating its left and rigth part (line 279)
        # Getting the type of 'tuple' (line 279)
        tuple_380326 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 279, 29), 'tuple')
        # Getting the type of 'index' (line 279)
        index_380327 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 279, 22), 'index')
        
        (may_be_380328, more_types_in_union_380329) = may_be_subtype(tuple_380326, index_380327)

        if may_be_380328:

            if more_types_in_union_380329:
                # Runtime conditional SSA (line 279)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            # Assigning a type to the variable 'index' (line 279)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 279, 8), 'index', remove_not_subtype_from_union(index_380327, tuple))
            
            
            
            # Call to len(...): (line 280)
            # Processing the call arguments (line 280)
            # Getting the type of 'index' (line 280)
            index_380331 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 280, 19), 'index', False)
            # Processing the call keyword arguments (line 280)
            kwargs_380332 = {}
            # Getting the type of 'len' (line 280)
            len_380330 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 280, 15), 'len', False)
            # Calling len(args, kwargs) (line 280)
            len_call_result_380333 = invoke(stypy.reporting.localization.Localization(__file__, 280, 15), len_380330, *[index_380331], **kwargs_380332)
            
            int_380334 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 280, 29), 'int')
            # Applying the binary operator '==' (line 280)
            result_eq_380335 = python_operator(stypy.reporting.localization.Localization(__file__, 280, 15), '==', len_call_result_380333, int_380334)
            
            # Testing the type of an if condition (line 280)
            if_condition_380336 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 280, 12), result_eq_380335)
            # Assigning a type to the variable 'if_condition_380336' (line 280)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 280, 12), 'if_condition_380336', if_condition_380336)
            # SSA begins for if statement (line 280)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a Name to a Tuple (line 281):
            
            # Assigning a Subscript to a Name (line 281):
            
            # Obtaining the type of the subscript
            int_380337 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 281, 16), 'int')
            # Getting the type of 'index' (line 281)
            index_380338 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 281, 27), 'index')
            # Obtaining the member '__getitem__' of a type (line 281)
            getitem___380339 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 281, 16), index_380338, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 281)
            subscript_call_result_380340 = invoke(stypy.reporting.localization.Localization(__file__, 281, 16), getitem___380339, int_380337)
            
            # Assigning a type to the variable 'tuple_var_assignment_379669' (line 281)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 281, 16), 'tuple_var_assignment_379669', subscript_call_result_380340)
            
            # Assigning a Subscript to a Name (line 281):
            
            # Obtaining the type of the subscript
            int_380341 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 281, 16), 'int')
            # Getting the type of 'index' (line 281)
            index_380342 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 281, 27), 'index')
            # Obtaining the member '__getitem__' of a type (line 281)
            getitem___380343 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 281, 16), index_380342, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 281)
            subscript_call_result_380344 = invoke(stypy.reporting.localization.Localization(__file__, 281, 16), getitem___380343, int_380341)
            
            # Assigning a type to the variable 'tuple_var_assignment_379670' (line 281)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 281, 16), 'tuple_var_assignment_379670', subscript_call_result_380344)
            
            # Assigning a Name to a Name (line 281):
            # Getting the type of 'tuple_var_assignment_379669' (line 281)
            tuple_var_assignment_379669_380345 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 281, 16), 'tuple_var_assignment_379669')
            # Assigning a type to the variable 'row' (line 281)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 281, 16), 'row', tuple_var_assignment_379669_380345)
            
            # Assigning a Name to a Name (line 281):
            # Getting the type of 'tuple_var_assignment_379670' (line 281)
            tuple_var_assignment_379670_380346 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 281, 16), 'tuple_var_assignment_379670')
            # Assigning a type to the variable 'col' (line 281)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 281, 21), 'col', tuple_var_assignment_379670_380346)
            # SSA branch for the else part of an if statement (line 280)
            module_type_store.open_ssa_branch('else')
            
            
            
            # Call to len(...): (line 282)
            # Processing the call arguments (line 282)
            # Getting the type of 'index' (line 282)
            index_380348 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 282, 21), 'index', False)
            # Processing the call keyword arguments (line 282)
            kwargs_380349 = {}
            # Getting the type of 'len' (line 282)
            len_380347 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 282, 17), 'len', False)
            # Calling len(args, kwargs) (line 282)
            len_call_result_380350 = invoke(stypy.reporting.localization.Localization(__file__, 282, 17), len_380347, *[index_380348], **kwargs_380349)
            
            int_380351 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 282, 31), 'int')
            # Applying the binary operator '==' (line 282)
            result_eq_380352 = python_operator(stypy.reporting.localization.Localization(__file__, 282, 17), '==', len_call_result_380350, int_380351)
            
            # Testing the type of an if condition (line 282)
            if_condition_380353 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 282, 17), result_eq_380352)
            # Assigning a type to the variable 'if_condition_380353' (line 282)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 282, 17), 'if_condition_380353', if_condition_380353)
            # SSA begins for if statement (line 282)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a Tuple to a Tuple (line 283):
            
            # Assigning a Subscript to a Name (line 283):
            
            # Obtaining the type of the subscript
            int_380354 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 283, 33), 'int')
            # Getting the type of 'index' (line 283)
            index_380355 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 283, 27), 'index')
            # Obtaining the member '__getitem__' of a type (line 283)
            getitem___380356 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 283, 27), index_380355, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 283)
            subscript_call_result_380357 = invoke(stypy.reporting.localization.Localization(__file__, 283, 27), getitem___380356, int_380354)
            
            # Assigning a type to the variable 'tuple_assignment_379671' (line 283)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 283, 16), 'tuple_assignment_379671', subscript_call_result_380357)
            
            # Assigning a Call to a Name (line 283):
            
            # Call to slice(...): (line 283)
            # Processing the call arguments (line 283)
            # Getting the type of 'None' (line 283)
            None_380359 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 283, 43), 'None', False)
            # Processing the call keyword arguments (line 283)
            kwargs_380360 = {}
            # Getting the type of 'slice' (line 283)
            slice_380358 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 283, 37), 'slice', False)
            # Calling slice(args, kwargs) (line 283)
            slice_call_result_380361 = invoke(stypy.reporting.localization.Localization(__file__, 283, 37), slice_380358, *[None_380359], **kwargs_380360)
            
            # Assigning a type to the variable 'tuple_assignment_379672' (line 283)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 283, 16), 'tuple_assignment_379672', slice_call_result_380361)
            
            # Assigning a Name to a Name (line 283):
            # Getting the type of 'tuple_assignment_379671' (line 283)
            tuple_assignment_379671_380362 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 283, 16), 'tuple_assignment_379671')
            # Assigning a type to the variable 'row' (line 283)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 283, 16), 'row', tuple_assignment_379671_380362)
            
            # Assigning a Name to a Name (line 283):
            # Getting the type of 'tuple_assignment_379672' (line 283)
            tuple_assignment_379672_380363 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 283, 16), 'tuple_assignment_379672')
            # Assigning a type to the variable 'col' (line 283)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 283, 21), 'col', tuple_assignment_379672_380363)
            # SSA branch for the else part of an if statement (line 282)
            module_type_store.open_ssa_branch('else')
            
            # Call to IndexError(...): (line 285)
            # Processing the call arguments (line 285)
            str_380365 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 285, 33), 'str', 'invalid number of indices')
            # Processing the call keyword arguments (line 285)
            kwargs_380366 = {}
            # Getting the type of 'IndexError' (line 285)
            IndexError_380364 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 285, 22), 'IndexError', False)
            # Calling IndexError(args, kwargs) (line 285)
            IndexError_call_result_380367 = invoke(stypy.reporting.localization.Localization(__file__, 285, 22), IndexError_380364, *[str_380365], **kwargs_380366)
            
            ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 285, 16), IndexError_call_result_380367, 'raise parameter', BaseException)
            # SSA join for if statement (line 282)
            module_type_store = module_type_store.join_ssa_context()
            
            # SSA join for if statement (line 280)
            module_type_store = module_type_store.join_ssa_context()
            

            if more_types_in_union_380329:
                # Runtime conditional SSA for else branch (line 279)
                module_type_store.open_ssa_branch('idiom else')



        if ((not may_be_380328) or more_types_in_union_380329):
            # Assigning a type to the variable 'index' (line 279)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 279, 8), 'index', remove_subtype_from_union(index_380327, tuple))
            
            # Assigning a Tuple to a Tuple (line 287):
            
            # Assigning a Name to a Name (line 287):
            # Getting the type of 'index' (line 287)
            index_380368 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 287, 23), 'index')
            # Assigning a type to the variable 'tuple_assignment_379673' (line 287)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 287, 12), 'tuple_assignment_379673', index_380368)
            
            # Assigning a Call to a Name (line 287):
            
            # Call to slice(...): (line 287)
            # Processing the call arguments (line 287)
            # Getting the type of 'None' (line 287)
            None_380370 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 287, 36), 'None', False)
            # Processing the call keyword arguments (line 287)
            kwargs_380371 = {}
            # Getting the type of 'slice' (line 287)
            slice_380369 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 287, 30), 'slice', False)
            # Calling slice(args, kwargs) (line 287)
            slice_call_result_380372 = invoke(stypy.reporting.localization.Localization(__file__, 287, 30), slice_380369, *[None_380370], **kwargs_380371)
            
            # Assigning a type to the variable 'tuple_assignment_379674' (line 287)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 287, 12), 'tuple_assignment_379674', slice_call_result_380372)
            
            # Assigning a Name to a Name (line 287):
            # Getting the type of 'tuple_assignment_379673' (line 287)
            tuple_assignment_379673_380373 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 287, 12), 'tuple_assignment_379673')
            # Assigning a type to the variable 'row' (line 287)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 287, 12), 'row', tuple_assignment_379673_380373)
            
            # Assigning a Name to a Name (line 287):
            # Getting the type of 'tuple_assignment_379674' (line 287)
            tuple_assignment_379674_380374 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 287, 12), 'tuple_assignment_379674')
            # Assigning a type to the variable 'col' (line 287)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 287, 17), 'col', tuple_assignment_379674_380374)

            if (may_be_380328 and more_types_in_union_380329):
                # SSA join for if statement (line 279)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Assigning a Call to a Tuple (line 290):
        
        # Assigning a Subscript to a Name (line 290):
        
        # Obtaining the type of the subscript
        int_380375 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 290, 8), 'int')
        
        # Call to _check_boolean(...): (line 290)
        # Processing the call arguments (line 290)
        # Getting the type of 'row' (line 290)
        row_380378 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 290, 39), 'row', False)
        # Getting the type of 'col' (line 290)
        col_380379 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 290, 44), 'col', False)
        # Processing the call keyword arguments (line 290)
        kwargs_380380 = {}
        # Getting the type of 'self' (line 290)
        self_380376 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 290, 19), 'self', False)
        # Obtaining the member '_check_boolean' of a type (line 290)
        _check_boolean_380377 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 290, 19), self_380376, '_check_boolean')
        # Calling _check_boolean(args, kwargs) (line 290)
        _check_boolean_call_result_380381 = invoke(stypy.reporting.localization.Localization(__file__, 290, 19), _check_boolean_380377, *[row_380378, col_380379], **kwargs_380380)
        
        # Obtaining the member '__getitem__' of a type (line 290)
        getitem___380382 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 290, 8), _check_boolean_call_result_380381, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 290)
        subscript_call_result_380383 = invoke(stypy.reporting.localization.Localization(__file__, 290, 8), getitem___380382, int_380375)
        
        # Assigning a type to the variable 'tuple_var_assignment_379675' (line 290)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 290, 8), 'tuple_var_assignment_379675', subscript_call_result_380383)
        
        # Assigning a Subscript to a Name (line 290):
        
        # Obtaining the type of the subscript
        int_380384 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 290, 8), 'int')
        
        # Call to _check_boolean(...): (line 290)
        # Processing the call arguments (line 290)
        # Getting the type of 'row' (line 290)
        row_380387 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 290, 39), 'row', False)
        # Getting the type of 'col' (line 290)
        col_380388 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 290, 44), 'col', False)
        # Processing the call keyword arguments (line 290)
        kwargs_380389 = {}
        # Getting the type of 'self' (line 290)
        self_380385 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 290, 19), 'self', False)
        # Obtaining the member '_check_boolean' of a type (line 290)
        _check_boolean_380386 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 290, 19), self_380385, '_check_boolean')
        # Calling _check_boolean(args, kwargs) (line 290)
        _check_boolean_call_result_380390 = invoke(stypy.reporting.localization.Localization(__file__, 290, 19), _check_boolean_380386, *[row_380387, col_380388], **kwargs_380389)
        
        # Obtaining the member '__getitem__' of a type (line 290)
        getitem___380391 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 290, 8), _check_boolean_call_result_380390, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 290)
        subscript_call_result_380392 = invoke(stypy.reporting.localization.Localization(__file__, 290, 8), getitem___380391, int_380384)
        
        # Assigning a type to the variable 'tuple_var_assignment_379676' (line 290)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 290, 8), 'tuple_var_assignment_379676', subscript_call_result_380392)
        
        # Assigning a Name to a Name (line 290):
        # Getting the type of 'tuple_var_assignment_379675' (line 290)
        tuple_var_assignment_379675_380393 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 290, 8), 'tuple_var_assignment_379675')
        # Assigning a type to the variable 'row' (line 290)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 290, 8), 'row', tuple_var_assignment_379675_380393)
        
        # Assigning a Name to a Name (line 290):
        # Getting the type of 'tuple_var_assignment_379676' (line 290)
        tuple_var_assignment_379676_380394 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 290, 8), 'tuple_var_assignment_379676')
        # Assigning a type to the variable 'col' (line 290)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 290, 13), 'col', tuple_var_assignment_379676_380394)
        
        # Obtaining an instance of the builtin type 'tuple' (line 291)
        tuple_380395 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 291, 15), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 291)
        # Adding element type (line 291)
        # Getting the type of 'row' (line 291)
        row_380396 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 291, 15), 'row')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 291, 15), tuple_380395, row_380396)
        # Adding element type (line 291)
        # Getting the type of 'col' (line 291)
        col_380397 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 291, 20), 'col')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 291, 15), tuple_380395, col_380397)
        
        # Assigning a type to the variable 'stypy_return_type' (line 291)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 291, 8), 'stypy_return_type', tuple_380395)
        
        # ################# End of '_unpack_index(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_unpack_index' in the type store
        # Getting the type of 'stypy_return_type' (line 265)
        stypy_return_type_380398 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 265, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_380398)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_unpack_index'
        return stypy_return_type_380398


    @norecursion
    def _check_ellipsis(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_check_ellipsis'
        module_type_store = module_type_store.open_function_context('_check_ellipsis', 293, 4, False)
        # Assigning a type to the variable 'self' (line 294)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 294, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        IndexMixin._check_ellipsis.__dict__.__setitem__('stypy_localization', localization)
        IndexMixin._check_ellipsis.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        IndexMixin._check_ellipsis.__dict__.__setitem__('stypy_type_store', module_type_store)
        IndexMixin._check_ellipsis.__dict__.__setitem__('stypy_function_name', 'IndexMixin._check_ellipsis')
        IndexMixin._check_ellipsis.__dict__.__setitem__('stypy_param_names_list', ['index'])
        IndexMixin._check_ellipsis.__dict__.__setitem__('stypy_varargs_param_name', None)
        IndexMixin._check_ellipsis.__dict__.__setitem__('stypy_kwargs_param_name', None)
        IndexMixin._check_ellipsis.__dict__.__setitem__('stypy_call_defaults', defaults)
        IndexMixin._check_ellipsis.__dict__.__setitem__('stypy_call_varargs', varargs)
        IndexMixin._check_ellipsis.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        IndexMixin._check_ellipsis.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'IndexMixin._check_ellipsis', ['index'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_check_ellipsis', localization, ['index'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_check_ellipsis(...)' code ##################

        str_380399 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 294, 8), 'str', 'Process indices with Ellipsis. Returns modified index.')
        
        
        # Getting the type of 'index' (line 295)
        index_380400 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 295, 11), 'index')
        # Getting the type of 'Ellipsis' (line 295)
        Ellipsis_380401 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 295, 20), 'Ellipsis')
        # Applying the binary operator 'is' (line 295)
        result_is__380402 = python_operator(stypy.reporting.localization.Localization(__file__, 295, 11), 'is', index_380400, Ellipsis_380401)
        
        # Testing the type of an if condition (line 295)
        if_condition_380403 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 295, 8), result_is__380402)
        # Assigning a type to the variable 'if_condition_380403' (line 295)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 295, 8), 'if_condition_380403', if_condition_380403)
        # SSA begins for if statement (line 295)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Obtaining an instance of the builtin type 'tuple' (line 296)
        tuple_380404 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 296, 20), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 296)
        # Adding element type (line 296)
        
        # Call to slice(...): (line 296)
        # Processing the call arguments (line 296)
        # Getting the type of 'None' (line 296)
        None_380406 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 296, 26), 'None', False)
        # Processing the call keyword arguments (line 296)
        kwargs_380407 = {}
        # Getting the type of 'slice' (line 296)
        slice_380405 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 296, 20), 'slice', False)
        # Calling slice(args, kwargs) (line 296)
        slice_call_result_380408 = invoke(stypy.reporting.localization.Localization(__file__, 296, 20), slice_380405, *[None_380406], **kwargs_380407)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 296, 20), tuple_380404, slice_call_result_380408)
        # Adding element type (line 296)
        
        # Call to slice(...): (line 296)
        # Processing the call arguments (line 296)
        # Getting the type of 'None' (line 296)
        None_380410 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 296, 39), 'None', False)
        # Processing the call keyword arguments (line 296)
        kwargs_380411 = {}
        # Getting the type of 'slice' (line 296)
        slice_380409 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 296, 33), 'slice', False)
        # Calling slice(args, kwargs) (line 296)
        slice_call_result_380412 = invoke(stypy.reporting.localization.Localization(__file__, 296, 33), slice_380409, *[None_380410], **kwargs_380411)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 296, 20), tuple_380404, slice_call_result_380412)
        
        # Assigning a type to the variable 'stypy_return_type' (line 296)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 296, 12), 'stypy_return_type', tuple_380404)
        # SSA branch for the else part of an if statement (line 295)
        module_type_store.open_ssa_branch('else')
        
        # Type idiom detected: calculating its left and rigth part (line 297)
        # Getting the type of 'tuple' (line 297)
        tuple_380413 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 297, 31), 'tuple')
        # Getting the type of 'index' (line 297)
        index_380414 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 297, 24), 'index')
        
        (may_be_380415, more_types_in_union_380416) = may_be_subtype(tuple_380413, index_380414)

        if may_be_380415:

            if more_types_in_union_380416:
                # Runtime conditional SSA (line 297)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            # Assigning a type to the variable 'index' (line 297)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 297, 13), 'index', remove_not_subtype_from_union(index_380414, tuple))
            
            
            # Call to enumerate(...): (line 299)
            # Processing the call arguments (line 299)
            # Getting the type of 'index' (line 299)
            index_380418 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 299, 34), 'index', False)
            # Processing the call keyword arguments (line 299)
            kwargs_380419 = {}
            # Getting the type of 'enumerate' (line 299)
            enumerate_380417 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 299, 24), 'enumerate', False)
            # Calling enumerate(args, kwargs) (line 299)
            enumerate_call_result_380420 = invoke(stypy.reporting.localization.Localization(__file__, 299, 24), enumerate_380417, *[index_380418], **kwargs_380419)
            
            # Testing the type of a for loop iterable (line 299)
            is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 299, 12), enumerate_call_result_380420)
            # Getting the type of the for loop variable (line 299)
            for_loop_var_380421 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 299, 12), enumerate_call_result_380420)
            # Assigning a type to the variable 'j' (line 299)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 299, 12), 'j', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 299, 12), for_loop_var_380421))
            # Assigning a type to the variable 'v' (line 299)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 299, 12), 'v', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 299, 12), for_loop_var_380421))
            # SSA begins for a for statement (line 299)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            
            # Getting the type of 'v' (line 300)
            v_380422 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 300, 19), 'v')
            # Getting the type of 'Ellipsis' (line 300)
            Ellipsis_380423 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 300, 24), 'Ellipsis')
            # Applying the binary operator 'is' (line 300)
            result_is__380424 = python_operator(stypy.reporting.localization.Localization(__file__, 300, 19), 'is', v_380422, Ellipsis_380423)
            
            # Testing the type of an if condition (line 300)
            if_condition_380425 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 300, 16), result_is__380424)
            # Assigning a type to the variable 'if_condition_380425' (line 300)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 300, 16), 'if_condition_380425', if_condition_380425)
            # SSA begins for if statement (line 300)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a Name to a Name (line 301):
            
            # Assigning a Name to a Name (line 301):
            # Getting the type of 'j' (line 301)
            j_380426 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 301, 37), 'j')
            # Assigning a type to the variable 'first_ellipsis' (line 301)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 301, 20), 'first_ellipsis', j_380426)
            # SSA join for if statement (line 300)
            module_type_store = module_type_store.join_ssa_context()
            
            # SSA branch for the else part of a for statement (line 299)
            module_type_store.open_ssa_branch('for loop else')
            
            # Assigning a Name to a Name (line 304):
            
            # Assigning a Name to a Name (line 304):
            # Getting the type of 'None' (line 304)
            None_380427 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 304, 33), 'None')
            # Assigning a type to the variable 'first_ellipsis' (line 304)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 304, 16), 'first_ellipsis', None_380427)
            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()
            
            
            # Type idiom detected: calculating its left and rigth part (line 307)
            # Getting the type of 'first_ellipsis' (line 307)
            first_ellipsis_380428 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 307, 12), 'first_ellipsis')
            # Getting the type of 'None' (line 307)
            None_380429 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 307, 37), 'None')
            
            (may_be_380430, more_types_in_union_380431) = may_not_be_none(first_ellipsis_380428, None_380429)

            if may_be_380430:

                if more_types_in_union_380431:
                    # Runtime conditional SSA (line 307)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
                else:
                    module_type_store = module_type_store

                
                
                
                # Call to len(...): (line 309)
                # Processing the call arguments (line 309)
                # Getting the type of 'index' (line 309)
                index_380433 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 309, 23), 'index', False)
                # Processing the call keyword arguments (line 309)
                kwargs_380434 = {}
                # Getting the type of 'len' (line 309)
                len_380432 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 309, 19), 'len', False)
                # Calling len(args, kwargs) (line 309)
                len_call_result_380435 = invoke(stypy.reporting.localization.Localization(__file__, 309, 19), len_380432, *[index_380433], **kwargs_380434)
                
                int_380436 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 309, 33), 'int')
                # Applying the binary operator '==' (line 309)
                result_eq_380437 = python_operator(stypy.reporting.localization.Localization(__file__, 309, 19), '==', len_call_result_380435, int_380436)
                
                # Testing the type of an if condition (line 309)
                if_condition_380438 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 309, 16), result_eq_380437)
                # Assigning a type to the variable 'if_condition_380438' (line 309)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 309, 16), 'if_condition_380438', if_condition_380438)
                # SSA begins for if statement (line 309)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Obtaining an instance of the builtin type 'tuple' (line 310)
                tuple_380439 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 310, 28), 'tuple')
                # Adding type elements to the builtin type 'tuple' instance (line 310)
                # Adding element type (line 310)
                
                # Call to slice(...): (line 310)
                # Processing the call arguments (line 310)
                # Getting the type of 'None' (line 310)
                None_380441 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 310, 34), 'None', False)
                # Processing the call keyword arguments (line 310)
                kwargs_380442 = {}
                # Getting the type of 'slice' (line 310)
                slice_380440 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 310, 28), 'slice', False)
                # Calling slice(args, kwargs) (line 310)
                slice_call_result_380443 = invoke(stypy.reporting.localization.Localization(__file__, 310, 28), slice_380440, *[None_380441], **kwargs_380442)
                
                add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 310, 28), tuple_380439, slice_call_result_380443)
                # Adding element type (line 310)
                
                # Call to slice(...): (line 310)
                # Processing the call arguments (line 310)
                # Getting the type of 'None' (line 310)
                None_380445 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 310, 47), 'None', False)
                # Processing the call keyword arguments (line 310)
                kwargs_380446 = {}
                # Getting the type of 'slice' (line 310)
                slice_380444 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 310, 41), 'slice', False)
                # Calling slice(args, kwargs) (line 310)
                slice_call_result_380447 = invoke(stypy.reporting.localization.Localization(__file__, 310, 41), slice_380444, *[None_380445], **kwargs_380446)
                
                add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 310, 28), tuple_380439, slice_call_result_380447)
                
                # Assigning a type to the variable 'stypy_return_type' (line 310)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 310, 20), 'stypy_return_type', tuple_380439)
                # SSA branch for the else part of an if statement (line 309)
                module_type_store.open_ssa_branch('else')
                
                
                
                # Call to len(...): (line 311)
                # Processing the call arguments (line 311)
                # Getting the type of 'index' (line 311)
                index_380449 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 311, 25), 'index', False)
                # Processing the call keyword arguments (line 311)
                kwargs_380450 = {}
                # Getting the type of 'len' (line 311)
                len_380448 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 311, 21), 'len', False)
                # Calling len(args, kwargs) (line 311)
                len_call_result_380451 = invoke(stypy.reporting.localization.Localization(__file__, 311, 21), len_380448, *[index_380449], **kwargs_380450)
                
                int_380452 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 311, 35), 'int')
                # Applying the binary operator '==' (line 311)
                result_eq_380453 = python_operator(stypy.reporting.localization.Localization(__file__, 311, 21), '==', len_call_result_380451, int_380452)
                
                # Testing the type of an if condition (line 311)
                if_condition_380454 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 311, 21), result_eq_380453)
                # Assigning a type to the variable 'if_condition_380454' (line 311)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 311, 21), 'if_condition_380454', if_condition_380454)
                # SSA begins for if statement (line 311)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                
                # Getting the type of 'first_ellipsis' (line 312)
                first_ellipsis_380455 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 312, 23), 'first_ellipsis')
                int_380456 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 312, 41), 'int')
                # Applying the binary operator '==' (line 312)
                result_eq_380457 = python_operator(stypy.reporting.localization.Localization(__file__, 312, 23), '==', first_ellipsis_380455, int_380456)
                
                # Testing the type of an if condition (line 312)
                if_condition_380458 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 312, 20), result_eq_380457)
                # Assigning a type to the variable 'if_condition_380458' (line 312)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 312, 20), 'if_condition_380458', if_condition_380458)
                # SSA begins for if statement (line 312)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                
                
                # Obtaining the type of the subscript
                int_380459 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 313, 33), 'int')
                # Getting the type of 'index' (line 313)
                index_380460 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 313, 27), 'index')
                # Obtaining the member '__getitem__' of a type (line 313)
                getitem___380461 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 313, 27), index_380460, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 313)
                subscript_call_result_380462 = invoke(stypy.reporting.localization.Localization(__file__, 313, 27), getitem___380461, int_380459)
                
                # Getting the type of 'Ellipsis' (line 313)
                Ellipsis_380463 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 313, 39), 'Ellipsis')
                # Applying the binary operator 'is' (line 313)
                result_is__380464 = python_operator(stypy.reporting.localization.Localization(__file__, 313, 27), 'is', subscript_call_result_380462, Ellipsis_380463)
                
                # Testing the type of an if condition (line 313)
                if_condition_380465 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 313, 24), result_is__380464)
                # Assigning a type to the variable 'if_condition_380465' (line 313)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 313, 24), 'if_condition_380465', if_condition_380465)
                # SSA begins for if statement (line 313)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Obtaining an instance of the builtin type 'tuple' (line 314)
                tuple_380466 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 314, 36), 'tuple')
                # Adding type elements to the builtin type 'tuple' instance (line 314)
                # Adding element type (line 314)
                
                # Call to slice(...): (line 314)
                # Processing the call arguments (line 314)
                # Getting the type of 'None' (line 314)
                None_380468 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 314, 42), 'None', False)
                # Processing the call keyword arguments (line 314)
                kwargs_380469 = {}
                # Getting the type of 'slice' (line 314)
                slice_380467 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 314, 36), 'slice', False)
                # Calling slice(args, kwargs) (line 314)
                slice_call_result_380470 = invoke(stypy.reporting.localization.Localization(__file__, 314, 36), slice_380467, *[None_380468], **kwargs_380469)
                
                add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 314, 36), tuple_380466, slice_call_result_380470)
                # Adding element type (line 314)
                
                # Call to slice(...): (line 314)
                # Processing the call arguments (line 314)
                # Getting the type of 'None' (line 314)
                None_380472 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 314, 55), 'None', False)
                # Processing the call keyword arguments (line 314)
                kwargs_380473 = {}
                # Getting the type of 'slice' (line 314)
                slice_380471 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 314, 49), 'slice', False)
                # Calling slice(args, kwargs) (line 314)
                slice_call_result_380474 = invoke(stypy.reporting.localization.Localization(__file__, 314, 49), slice_380471, *[None_380472], **kwargs_380473)
                
                add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 314, 36), tuple_380466, slice_call_result_380474)
                
                # Assigning a type to the variable 'stypy_return_type' (line 314)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 314, 28), 'stypy_return_type', tuple_380466)
                # SSA branch for the else part of an if statement (line 313)
                module_type_store.open_ssa_branch('else')
                
                # Obtaining an instance of the builtin type 'tuple' (line 316)
                tuple_380475 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 316, 36), 'tuple')
                # Adding type elements to the builtin type 'tuple' instance (line 316)
                # Adding element type (line 316)
                
                # Call to slice(...): (line 316)
                # Processing the call arguments (line 316)
                # Getting the type of 'None' (line 316)
                None_380477 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 316, 42), 'None', False)
                # Processing the call keyword arguments (line 316)
                kwargs_380478 = {}
                # Getting the type of 'slice' (line 316)
                slice_380476 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 316, 36), 'slice', False)
                # Calling slice(args, kwargs) (line 316)
                slice_call_result_380479 = invoke(stypy.reporting.localization.Localization(__file__, 316, 36), slice_380476, *[None_380477], **kwargs_380478)
                
                add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 316, 36), tuple_380475, slice_call_result_380479)
                # Adding element type (line 316)
                
                # Obtaining the type of the subscript
                int_380480 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 316, 55), 'int')
                # Getting the type of 'index' (line 316)
                index_380481 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 316, 49), 'index')
                # Obtaining the member '__getitem__' of a type (line 316)
                getitem___380482 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 316, 49), index_380481, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 316)
                subscript_call_result_380483 = invoke(stypy.reporting.localization.Localization(__file__, 316, 49), getitem___380482, int_380480)
                
                add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 316, 36), tuple_380475, subscript_call_result_380483)
                
                # Assigning a type to the variable 'stypy_return_type' (line 316)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 316, 28), 'stypy_return_type', tuple_380475)
                # SSA join for if statement (line 313)
                module_type_store = module_type_store.join_ssa_context()
                
                # SSA branch for the else part of an if statement (line 312)
                module_type_store.open_ssa_branch('else')
                
                # Obtaining an instance of the builtin type 'tuple' (line 318)
                tuple_380484 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 318, 32), 'tuple')
                # Adding type elements to the builtin type 'tuple' instance (line 318)
                # Adding element type (line 318)
                
                # Obtaining the type of the subscript
                int_380485 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 318, 38), 'int')
                # Getting the type of 'index' (line 318)
                index_380486 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 318, 32), 'index')
                # Obtaining the member '__getitem__' of a type (line 318)
                getitem___380487 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 318, 32), index_380486, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 318)
                subscript_call_result_380488 = invoke(stypy.reporting.localization.Localization(__file__, 318, 32), getitem___380487, int_380485)
                
                add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 318, 32), tuple_380484, subscript_call_result_380488)
                # Adding element type (line 318)
                
                # Call to slice(...): (line 318)
                # Processing the call arguments (line 318)
                # Getting the type of 'None' (line 318)
                None_380490 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 318, 48), 'None', False)
                # Processing the call keyword arguments (line 318)
                kwargs_380491 = {}
                # Getting the type of 'slice' (line 318)
                slice_380489 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 318, 42), 'slice', False)
                # Calling slice(args, kwargs) (line 318)
                slice_call_result_380492 = invoke(stypy.reporting.localization.Localization(__file__, 318, 42), slice_380489, *[None_380490], **kwargs_380491)
                
                add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 318, 32), tuple_380484, slice_call_result_380492)
                
                # Assigning a type to the variable 'stypy_return_type' (line 318)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 318, 24), 'stypy_return_type', tuple_380484)
                # SSA join for if statement (line 312)
                module_type_store = module_type_store.join_ssa_context()
                
                # SSA join for if statement (line 311)
                module_type_store = module_type_store.join_ssa_context()
                
                # SSA join for if statement (line 309)
                module_type_store = module_type_store.join_ssa_context()
                
                
                # Assigning a Tuple to a Name (line 321):
                
                # Assigning a Tuple to a Name (line 321):
                
                # Obtaining an instance of the builtin type 'tuple' (line 321)
                tuple_380493 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 321, 23), 'tuple')
                # Adding type elements to the builtin type 'tuple' instance (line 321)
                
                # Assigning a type to the variable 'tail' (line 321)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 321, 16), 'tail', tuple_380493)
                
                
                # Obtaining the type of the subscript
                # Getting the type of 'first_ellipsis' (line 322)
                first_ellipsis_380494 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 322, 31), 'first_ellipsis')
                int_380495 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 322, 46), 'int')
                # Applying the binary operator '+' (line 322)
                result_add_380496 = python_operator(stypy.reporting.localization.Localization(__file__, 322, 31), '+', first_ellipsis_380494, int_380495)
                
                slice_380497 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 322, 25), result_add_380496, None, None)
                # Getting the type of 'index' (line 322)
                index_380498 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 322, 25), 'index')
                # Obtaining the member '__getitem__' of a type (line 322)
                getitem___380499 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 322, 25), index_380498, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 322)
                subscript_call_result_380500 = invoke(stypy.reporting.localization.Localization(__file__, 322, 25), getitem___380499, slice_380497)
                
                # Testing the type of a for loop iterable (line 322)
                is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 322, 16), subscript_call_result_380500)
                # Getting the type of the for loop variable (line 322)
                for_loop_var_380501 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 322, 16), subscript_call_result_380500)
                # Assigning a type to the variable 'v' (line 322)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 322, 16), 'v', for_loop_var_380501)
                # SSA begins for a for statement (line 322)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
                
                
                # Getting the type of 'v' (line 323)
                v_380502 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 323, 23), 'v')
                # Getting the type of 'Ellipsis' (line 323)
                Ellipsis_380503 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 323, 32), 'Ellipsis')
                # Applying the binary operator 'isnot' (line 323)
                result_is_not_380504 = python_operator(stypy.reporting.localization.Localization(__file__, 323, 23), 'isnot', v_380502, Ellipsis_380503)
                
                # Testing the type of an if condition (line 323)
                if_condition_380505 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 323, 20), result_is_not_380504)
                # Assigning a type to the variable 'if_condition_380505' (line 323)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 323, 20), 'if_condition_380505', if_condition_380505)
                # SSA begins for if statement (line 323)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Assigning a BinOp to a Name (line 324):
                
                # Assigning a BinOp to a Name (line 324):
                # Getting the type of 'tail' (line 324)
                tail_380506 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 324, 31), 'tail')
                
                # Obtaining an instance of the builtin type 'tuple' (line 324)
                tuple_380507 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 324, 39), 'tuple')
                # Adding type elements to the builtin type 'tuple' instance (line 324)
                # Adding element type (line 324)
                # Getting the type of 'v' (line 324)
                v_380508 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 324, 39), 'v')
                add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 324, 39), tuple_380507, v_380508)
                
                # Applying the binary operator '+' (line 324)
                result_add_380509 = python_operator(stypy.reporting.localization.Localization(__file__, 324, 31), '+', tail_380506, tuple_380507)
                
                # Assigning a type to the variable 'tail' (line 324)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 324, 24), 'tail', result_add_380509)
                # SSA join for if statement (line 323)
                module_type_store = module_type_store.join_ssa_context()
                
                # SSA join for a for statement
                module_type_store = module_type_store.join_ssa_context()
                
                
                # Assigning a BinOp to a Name (line 325):
                
                # Assigning a BinOp to a Name (line 325):
                # Getting the type of 'first_ellipsis' (line 325)
                first_ellipsis_380510 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 325, 21), 'first_ellipsis')
                
                # Call to len(...): (line 325)
                # Processing the call arguments (line 325)
                # Getting the type of 'tail' (line 325)
                tail_380512 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 325, 42), 'tail', False)
                # Processing the call keyword arguments (line 325)
                kwargs_380513 = {}
                # Getting the type of 'len' (line 325)
                len_380511 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 325, 38), 'len', False)
                # Calling len(args, kwargs) (line 325)
                len_call_result_380514 = invoke(stypy.reporting.localization.Localization(__file__, 325, 38), len_380511, *[tail_380512], **kwargs_380513)
                
                # Applying the binary operator '+' (line 325)
                result_add_380515 = python_operator(stypy.reporting.localization.Localization(__file__, 325, 21), '+', first_ellipsis_380510, len_call_result_380514)
                
                # Assigning a type to the variable 'nd' (line 325)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 325, 16), 'nd', result_add_380515)
                
                # Assigning a Call to a Name (line 326):
                
                # Assigning a Call to a Name (line 326):
                
                # Call to max(...): (line 326)
                # Processing the call arguments (line 326)
                int_380517 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 326, 29), 'int')
                int_380518 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 326, 32), 'int')
                # Getting the type of 'nd' (line 326)
                nd_380519 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 326, 36), 'nd', False)
                # Applying the binary operator '-' (line 326)
                result_sub_380520 = python_operator(stypy.reporting.localization.Localization(__file__, 326, 32), '-', int_380518, nd_380519)
                
                # Processing the call keyword arguments (line 326)
                kwargs_380521 = {}
                # Getting the type of 'max' (line 326)
                max_380516 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 326, 25), 'max', False)
                # Calling max(args, kwargs) (line 326)
                max_call_result_380522 = invoke(stypy.reporting.localization.Localization(__file__, 326, 25), max_380516, *[int_380517, result_sub_380520], **kwargs_380521)
                
                # Assigning a type to the variable 'nslice' (line 326)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 326, 16), 'nslice', max_call_result_380522)
                
                # Obtaining the type of the subscript
                # Getting the type of 'first_ellipsis' (line 327)
                first_ellipsis_380523 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 327, 30), 'first_ellipsis')
                slice_380524 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 327, 23), None, first_ellipsis_380523, None)
                # Getting the type of 'index' (line 327)
                index_380525 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 327, 23), 'index')
                # Obtaining the member '__getitem__' of a type (line 327)
                getitem___380526 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 327, 23), index_380525, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 327)
                subscript_call_result_380527 = invoke(stypy.reporting.localization.Localization(__file__, 327, 23), getitem___380526, slice_380524)
                
                
                # Obtaining an instance of the builtin type 'tuple' (line 327)
                tuple_380528 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 327, 49), 'tuple')
                # Adding type elements to the builtin type 'tuple' instance (line 327)
                # Adding element type (line 327)
                
                # Call to slice(...): (line 327)
                # Processing the call arguments (line 327)
                # Getting the type of 'None' (line 327)
                None_380530 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 327, 55), 'None', False)
                # Processing the call keyword arguments (line 327)
                kwargs_380531 = {}
                # Getting the type of 'slice' (line 327)
                slice_380529 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 327, 49), 'slice', False)
                # Calling slice(args, kwargs) (line 327)
                slice_call_result_380532 = invoke(stypy.reporting.localization.Localization(__file__, 327, 49), slice_380529, *[None_380530], **kwargs_380531)
                
                add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 327, 49), tuple_380528, slice_call_result_380532)
                
                # Getting the type of 'nslice' (line 327)
                nslice_380533 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 327, 63), 'nslice')
                # Applying the binary operator '*' (line 327)
                result_mul_380534 = python_operator(stypy.reporting.localization.Localization(__file__, 327, 48), '*', tuple_380528, nslice_380533)
                
                # Applying the binary operator '+' (line 327)
                result_add_380535 = python_operator(stypy.reporting.localization.Localization(__file__, 327, 23), '+', subscript_call_result_380527, result_mul_380534)
                
                # Getting the type of 'tail' (line 327)
                tail_380536 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 327, 72), 'tail')
                # Applying the binary operator '+' (line 327)
                result_add_380537 = python_operator(stypy.reporting.localization.Localization(__file__, 327, 70), '+', result_add_380535, tail_380536)
                
                # Assigning a type to the variable 'stypy_return_type' (line 327)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 327, 16), 'stypy_return_type', result_add_380537)

                if more_types_in_union_380431:
                    # SSA join for if statement (line 307)
                    module_type_store = module_type_store.join_ssa_context()


            

            if more_types_in_union_380416:
                # SSA join for if statement (line 297)
                module_type_store = module_type_store.join_ssa_context()


        
        # SSA join for if statement (line 295)
        module_type_store = module_type_store.join_ssa_context()
        
        # Getting the type of 'index' (line 329)
        index_380538 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 329, 15), 'index')
        # Assigning a type to the variable 'stypy_return_type' (line 329)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 329, 8), 'stypy_return_type', index_380538)
        
        # ################# End of '_check_ellipsis(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_check_ellipsis' in the type store
        # Getting the type of 'stypy_return_type' (line 293)
        stypy_return_type_380539 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 293, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_380539)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_check_ellipsis'
        return stypy_return_type_380539


    @norecursion
    def _check_boolean(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_check_boolean'
        module_type_store = module_type_store.open_function_context('_check_boolean', 331, 4, False)
        # Assigning a type to the variable 'self' (line 332)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 332, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        IndexMixin._check_boolean.__dict__.__setitem__('stypy_localization', localization)
        IndexMixin._check_boolean.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        IndexMixin._check_boolean.__dict__.__setitem__('stypy_type_store', module_type_store)
        IndexMixin._check_boolean.__dict__.__setitem__('stypy_function_name', 'IndexMixin._check_boolean')
        IndexMixin._check_boolean.__dict__.__setitem__('stypy_param_names_list', ['row', 'col'])
        IndexMixin._check_boolean.__dict__.__setitem__('stypy_varargs_param_name', None)
        IndexMixin._check_boolean.__dict__.__setitem__('stypy_kwargs_param_name', None)
        IndexMixin._check_boolean.__dict__.__setitem__('stypy_call_defaults', defaults)
        IndexMixin._check_boolean.__dict__.__setitem__('stypy_call_varargs', varargs)
        IndexMixin._check_boolean.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        IndexMixin._check_boolean.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'IndexMixin._check_boolean', ['row', 'col'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_check_boolean', localization, ['row', 'col'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_check_boolean(...)' code ##################

        stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 332, 8))
        
        # 'from scipy.sparse.base import isspmatrix' statement (line 332)
        update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/sparse/')
        import_380540 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 332, 8), 'scipy.sparse.base')

        if (type(import_380540) is not StypyTypeError):

            if (import_380540 != 'pyd_module'):
                __import__(import_380540)
                sys_modules_380541 = sys.modules[import_380540]
                import_from_module(stypy.reporting.localization.Localization(__file__, 332, 8), 'scipy.sparse.base', sys_modules_380541.module_type_store, module_type_store, ['isspmatrix'])
                nest_module(stypy.reporting.localization.Localization(__file__, 332, 8), __file__, sys_modules_380541, sys_modules_380541.module_type_store, module_type_store)
            else:
                from scipy.sparse.base import isspmatrix

                import_from_module(stypy.reporting.localization.Localization(__file__, 332, 8), 'scipy.sparse.base', None, module_type_store, ['isspmatrix'], [isspmatrix])

        else:
            # Assigning a type to the variable 'scipy.sparse.base' (line 332)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 332, 8), 'scipy.sparse.base', import_380540)

        remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/sparse/')
        
        
        
        # Evaluating a boolean operation
        
        # Call to isspmatrix(...): (line 335)
        # Processing the call arguments (line 335)
        # Getting the type of 'row' (line 335)
        row_380543 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 335, 22), 'row', False)
        # Processing the call keyword arguments (line 335)
        kwargs_380544 = {}
        # Getting the type of 'isspmatrix' (line 335)
        isspmatrix_380542 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 335, 11), 'isspmatrix', False)
        # Calling isspmatrix(args, kwargs) (line 335)
        isspmatrix_call_result_380545 = invoke(stypy.reporting.localization.Localization(__file__, 335, 11), isspmatrix_380542, *[row_380543], **kwargs_380544)
        
        
        # Call to isspmatrix(...): (line 335)
        # Processing the call arguments (line 335)
        # Getting the type of 'col' (line 335)
        col_380547 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 335, 41), 'col', False)
        # Processing the call keyword arguments (line 335)
        kwargs_380548 = {}
        # Getting the type of 'isspmatrix' (line 335)
        isspmatrix_380546 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 335, 30), 'isspmatrix', False)
        # Calling isspmatrix(args, kwargs) (line 335)
        isspmatrix_call_result_380549 = invoke(stypy.reporting.localization.Localization(__file__, 335, 30), isspmatrix_380546, *[col_380547], **kwargs_380548)
        
        # Applying the binary operator 'or' (line 335)
        result_or_keyword_380550 = python_operator(stypy.reporting.localization.Localization(__file__, 335, 11), 'or', isspmatrix_call_result_380545, isspmatrix_call_result_380549)
        
        # Testing the type of an if condition (line 335)
        if_condition_380551 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 335, 8), result_or_keyword_380550)
        # Assigning a type to the variable 'if_condition_380551' (line 335)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 335, 8), 'if_condition_380551', if_condition_380551)
        # SSA begins for if statement (line 335)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to IndexError(...): (line 336)
        # Processing the call arguments (line 336)
        str_380553 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 337, 16), 'str', 'Indexing with sparse matrices is not supported except boolean indexing where matrix and index are equal shapes.')
        # Processing the call keyword arguments (line 336)
        kwargs_380554 = {}
        # Getting the type of 'IndexError' (line 336)
        IndexError_380552 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 336, 18), 'IndexError', False)
        # Calling IndexError(args, kwargs) (line 336)
        IndexError_call_result_380555 = invoke(stypy.reporting.localization.Localization(__file__, 336, 18), IndexError_380552, *[str_380553], **kwargs_380554)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 336, 12), IndexError_call_result_380555, 'raise parameter', BaseException)
        # SSA join for if statement (line 335)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Evaluating a boolean operation
        
        # Call to isinstance(...): (line 340)
        # Processing the call arguments (line 340)
        # Getting the type of 'row' (line 340)
        row_380557 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 340, 22), 'row', False)
        # Getting the type of 'np' (line 340)
        np_380558 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 340, 27), 'np', False)
        # Obtaining the member 'ndarray' of a type (line 340)
        ndarray_380559 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 340, 27), np_380558, 'ndarray')
        # Processing the call keyword arguments (line 340)
        kwargs_380560 = {}
        # Getting the type of 'isinstance' (line 340)
        isinstance_380556 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 340, 11), 'isinstance', False)
        # Calling isinstance(args, kwargs) (line 340)
        isinstance_call_result_380561 = invoke(stypy.reporting.localization.Localization(__file__, 340, 11), isinstance_380556, *[row_380557, ndarray_380559], **kwargs_380560)
        
        
        # Getting the type of 'row' (line 340)
        row_380562 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 340, 43), 'row')
        # Obtaining the member 'dtype' of a type (line 340)
        dtype_380563 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 340, 43), row_380562, 'dtype')
        # Obtaining the member 'kind' of a type (line 340)
        kind_380564 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 340, 43), dtype_380563, 'kind')
        str_380565 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 340, 61), 'str', 'b')
        # Applying the binary operator '==' (line 340)
        result_eq_380566 = python_operator(stypy.reporting.localization.Localization(__file__, 340, 43), '==', kind_380564, str_380565)
        
        # Applying the binary operator 'and' (line 340)
        result_and_keyword_380567 = python_operator(stypy.reporting.localization.Localization(__file__, 340, 11), 'and', isinstance_call_result_380561, result_eq_380566)
        
        # Testing the type of an if condition (line 340)
        if_condition_380568 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 340, 8), result_and_keyword_380567)
        # Assigning a type to the variable 'if_condition_380568' (line 340)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 340, 8), 'if_condition_380568', if_condition_380568)
        # SSA begins for if statement (line 340)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 341):
        
        # Assigning a Call to a Name (line 341):
        
        # Call to _boolean_index_to_array(...): (line 341)
        # Processing the call arguments (line 341)
        # Getting the type of 'row' (line 341)
        row_380571 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 341, 47), 'row', False)
        # Processing the call keyword arguments (line 341)
        kwargs_380572 = {}
        # Getting the type of 'self' (line 341)
        self_380569 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 341, 18), 'self', False)
        # Obtaining the member '_boolean_index_to_array' of a type (line 341)
        _boolean_index_to_array_380570 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 341, 18), self_380569, '_boolean_index_to_array')
        # Calling _boolean_index_to_array(args, kwargs) (line 341)
        _boolean_index_to_array_call_result_380573 = invoke(stypy.reporting.localization.Localization(__file__, 341, 18), _boolean_index_to_array_380570, *[row_380571], **kwargs_380572)
        
        # Assigning a type to the variable 'row' (line 341)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 341, 12), 'row', _boolean_index_to_array_call_result_380573)
        # SSA join for if statement (line 340)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Evaluating a boolean operation
        
        # Call to isinstance(...): (line 342)
        # Processing the call arguments (line 342)
        # Getting the type of 'col' (line 342)
        col_380575 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 342, 22), 'col', False)
        # Getting the type of 'np' (line 342)
        np_380576 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 342, 27), 'np', False)
        # Obtaining the member 'ndarray' of a type (line 342)
        ndarray_380577 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 342, 27), np_380576, 'ndarray')
        # Processing the call keyword arguments (line 342)
        kwargs_380578 = {}
        # Getting the type of 'isinstance' (line 342)
        isinstance_380574 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 342, 11), 'isinstance', False)
        # Calling isinstance(args, kwargs) (line 342)
        isinstance_call_result_380579 = invoke(stypy.reporting.localization.Localization(__file__, 342, 11), isinstance_380574, *[col_380575, ndarray_380577], **kwargs_380578)
        
        
        # Getting the type of 'col' (line 342)
        col_380580 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 342, 43), 'col')
        # Obtaining the member 'dtype' of a type (line 342)
        dtype_380581 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 342, 43), col_380580, 'dtype')
        # Obtaining the member 'kind' of a type (line 342)
        kind_380582 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 342, 43), dtype_380581, 'kind')
        str_380583 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 342, 61), 'str', 'b')
        # Applying the binary operator '==' (line 342)
        result_eq_380584 = python_operator(stypy.reporting.localization.Localization(__file__, 342, 43), '==', kind_380582, str_380583)
        
        # Applying the binary operator 'and' (line 342)
        result_and_keyword_380585 = python_operator(stypy.reporting.localization.Localization(__file__, 342, 11), 'and', isinstance_call_result_380579, result_eq_380584)
        
        # Testing the type of an if condition (line 342)
        if_condition_380586 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 342, 8), result_and_keyword_380585)
        # Assigning a type to the variable 'if_condition_380586' (line 342)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 342, 8), 'if_condition_380586', if_condition_380586)
        # SSA begins for if statement (line 342)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 343):
        
        # Assigning a Call to a Name (line 343):
        
        # Call to _boolean_index_to_array(...): (line 343)
        # Processing the call arguments (line 343)
        # Getting the type of 'col' (line 343)
        col_380589 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 343, 47), 'col', False)
        # Processing the call keyword arguments (line 343)
        kwargs_380590 = {}
        # Getting the type of 'self' (line 343)
        self_380587 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 343, 18), 'self', False)
        # Obtaining the member '_boolean_index_to_array' of a type (line 343)
        _boolean_index_to_array_380588 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 343, 18), self_380587, '_boolean_index_to_array')
        # Calling _boolean_index_to_array(args, kwargs) (line 343)
        _boolean_index_to_array_call_result_380591 = invoke(stypy.reporting.localization.Localization(__file__, 343, 18), _boolean_index_to_array_380588, *[col_380589], **kwargs_380590)
        
        # Assigning a type to the variable 'col' (line 343)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 343, 12), 'col', _boolean_index_to_array_call_result_380591)
        # SSA join for if statement (line 342)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Obtaining an instance of the builtin type 'tuple' (line 344)
        tuple_380592 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 344, 15), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 344)
        # Adding element type (line 344)
        # Getting the type of 'row' (line 344)
        row_380593 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 344, 15), 'row')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 344, 15), tuple_380592, row_380593)
        # Adding element type (line 344)
        # Getting the type of 'col' (line 344)
        col_380594 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 344, 20), 'col')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 344, 15), tuple_380592, col_380594)
        
        # Assigning a type to the variable 'stypy_return_type' (line 344)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 344, 8), 'stypy_return_type', tuple_380592)
        
        # ################# End of '_check_boolean(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_check_boolean' in the type store
        # Getting the type of 'stypy_return_type' (line 331)
        stypy_return_type_380595 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 331, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_380595)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_check_boolean'
        return stypy_return_type_380595


    @norecursion
    def _boolean_index_to_array(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_boolean_index_to_array'
        module_type_store = module_type_store.open_function_context('_boolean_index_to_array', 346, 4, False)
        # Assigning a type to the variable 'self' (line 347)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 347, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        IndexMixin._boolean_index_to_array.__dict__.__setitem__('stypy_localization', localization)
        IndexMixin._boolean_index_to_array.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        IndexMixin._boolean_index_to_array.__dict__.__setitem__('stypy_type_store', module_type_store)
        IndexMixin._boolean_index_to_array.__dict__.__setitem__('stypy_function_name', 'IndexMixin._boolean_index_to_array')
        IndexMixin._boolean_index_to_array.__dict__.__setitem__('stypy_param_names_list', ['i'])
        IndexMixin._boolean_index_to_array.__dict__.__setitem__('stypy_varargs_param_name', None)
        IndexMixin._boolean_index_to_array.__dict__.__setitem__('stypy_kwargs_param_name', None)
        IndexMixin._boolean_index_to_array.__dict__.__setitem__('stypy_call_defaults', defaults)
        IndexMixin._boolean_index_to_array.__dict__.__setitem__('stypy_call_varargs', varargs)
        IndexMixin._boolean_index_to_array.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        IndexMixin._boolean_index_to_array.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'IndexMixin._boolean_index_to_array', ['i'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_boolean_index_to_array', localization, ['i'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_boolean_index_to_array(...)' code ##################

        
        
        # Getting the type of 'i' (line 347)
        i_380596 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 347, 11), 'i')
        # Obtaining the member 'ndim' of a type (line 347)
        ndim_380597 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 347, 11), i_380596, 'ndim')
        int_380598 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 347, 20), 'int')
        # Applying the binary operator '>' (line 347)
        result_gt_380599 = python_operator(stypy.reporting.localization.Localization(__file__, 347, 11), '>', ndim_380597, int_380598)
        
        # Testing the type of an if condition (line 347)
        if_condition_380600 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 347, 8), result_gt_380599)
        # Assigning a type to the variable 'if_condition_380600' (line 347)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 347, 8), 'if_condition_380600', if_condition_380600)
        # SSA begins for if statement (line 347)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to IndexError(...): (line 348)
        # Processing the call arguments (line 348)
        str_380602 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 348, 29), 'str', 'invalid index shape')
        # Processing the call keyword arguments (line 348)
        kwargs_380603 = {}
        # Getting the type of 'IndexError' (line 348)
        IndexError_380601 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 348, 18), 'IndexError', False)
        # Calling IndexError(args, kwargs) (line 348)
        IndexError_call_result_380604 = invoke(stypy.reporting.localization.Localization(__file__, 348, 18), IndexError_380601, *[str_380602], **kwargs_380603)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 348, 12), IndexError_call_result_380604, 'raise parameter', BaseException)
        # SSA join for if statement (line 347)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Obtaining the type of the subscript
        int_380605 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 349, 27), 'int')
        
        # Call to nonzero(...): (line 349)
        # Processing the call keyword arguments (line 349)
        kwargs_380608 = {}
        # Getting the type of 'i' (line 349)
        i_380606 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 349, 15), 'i', False)
        # Obtaining the member 'nonzero' of a type (line 349)
        nonzero_380607 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 349, 15), i_380606, 'nonzero')
        # Calling nonzero(args, kwargs) (line 349)
        nonzero_call_result_380609 = invoke(stypy.reporting.localization.Localization(__file__, 349, 15), nonzero_380607, *[], **kwargs_380608)
        
        # Obtaining the member '__getitem__' of a type (line 349)
        getitem___380610 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 349, 15), nonzero_call_result_380609, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 349)
        subscript_call_result_380611 = invoke(stypy.reporting.localization.Localization(__file__, 349, 15), getitem___380610, int_380605)
        
        # Assigning a type to the variable 'stypy_return_type' (line 349)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 349, 8), 'stypy_return_type', subscript_call_result_380611)
        
        # ################# End of '_boolean_index_to_array(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_boolean_index_to_array' in the type store
        # Getting the type of 'stypy_return_type' (line 346)
        stypy_return_type_380612 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 346, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_380612)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_boolean_index_to_array'
        return stypy_return_type_380612


    @norecursion
    def _index_to_arrays(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_index_to_arrays'
        module_type_store = module_type_store.open_function_context('_index_to_arrays', 351, 4, False)
        # Assigning a type to the variable 'self' (line 352)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 352, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        IndexMixin._index_to_arrays.__dict__.__setitem__('stypy_localization', localization)
        IndexMixin._index_to_arrays.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        IndexMixin._index_to_arrays.__dict__.__setitem__('stypy_type_store', module_type_store)
        IndexMixin._index_to_arrays.__dict__.__setitem__('stypy_function_name', 'IndexMixin._index_to_arrays')
        IndexMixin._index_to_arrays.__dict__.__setitem__('stypy_param_names_list', ['i', 'j'])
        IndexMixin._index_to_arrays.__dict__.__setitem__('stypy_varargs_param_name', None)
        IndexMixin._index_to_arrays.__dict__.__setitem__('stypy_kwargs_param_name', None)
        IndexMixin._index_to_arrays.__dict__.__setitem__('stypy_call_defaults', defaults)
        IndexMixin._index_to_arrays.__dict__.__setitem__('stypy_call_varargs', varargs)
        IndexMixin._index_to_arrays.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        IndexMixin._index_to_arrays.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'IndexMixin._index_to_arrays', ['i', 'j'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_index_to_arrays', localization, ['i', 'j'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_index_to_arrays(...)' code ##################

        
        # Assigning a Call to a Tuple (line 352):
        
        # Assigning a Subscript to a Name (line 352):
        
        # Obtaining the type of the subscript
        int_380613 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 352, 8), 'int')
        
        # Call to _check_boolean(...): (line 352)
        # Processing the call arguments (line 352)
        # Getting the type of 'i' (line 352)
        i_380616 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 352, 35), 'i', False)
        # Getting the type of 'j' (line 352)
        j_380617 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 352, 38), 'j', False)
        # Processing the call keyword arguments (line 352)
        kwargs_380618 = {}
        # Getting the type of 'self' (line 352)
        self_380614 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 352, 15), 'self', False)
        # Obtaining the member '_check_boolean' of a type (line 352)
        _check_boolean_380615 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 352, 15), self_380614, '_check_boolean')
        # Calling _check_boolean(args, kwargs) (line 352)
        _check_boolean_call_result_380619 = invoke(stypy.reporting.localization.Localization(__file__, 352, 15), _check_boolean_380615, *[i_380616, j_380617], **kwargs_380618)
        
        # Obtaining the member '__getitem__' of a type (line 352)
        getitem___380620 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 352, 8), _check_boolean_call_result_380619, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 352)
        subscript_call_result_380621 = invoke(stypy.reporting.localization.Localization(__file__, 352, 8), getitem___380620, int_380613)
        
        # Assigning a type to the variable 'tuple_var_assignment_379677' (line 352)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 352, 8), 'tuple_var_assignment_379677', subscript_call_result_380621)
        
        # Assigning a Subscript to a Name (line 352):
        
        # Obtaining the type of the subscript
        int_380622 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 352, 8), 'int')
        
        # Call to _check_boolean(...): (line 352)
        # Processing the call arguments (line 352)
        # Getting the type of 'i' (line 352)
        i_380625 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 352, 35), 'i', False)
        # Getting the type of 'j' (line 352)
        j_380626 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 352, 38), 'j', False)
        # Processing the call keyword arguments (line 352)
        kwargs_380627 = {}
        # Getting the type of 'self' (line 352)
        self_380623 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 352, 15), 'self', False)
        # Obtaining the member '_check_boolean' of a type (line 352)
        _check_boolean_380624 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 352, 15), self_380623, '_check_boolean')
        # Calling _check_boolean(args, kwargs) (line 352)
        _check_boolean_call_result_380628 = invoke(stypy.reporting.localization.Localization(__file__, 352, 15), _check_boolean_380624, *[i_380625, j_380626], **kwargs_380627)
        
        # Obtaining the member '__getitem__' of a type (line 352)
        getitem___380629 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 352, 8), _check_boolean_call_result_380628, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 352)
        subscript_call_result_380630 = invoke(stypy.reporting.localization.Localization(__file__, 352, 8), getitem___380629, int_380622)
        
        # Assigning a type to the variable 'tuple_var_assignment_379678' (line 352)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 352, 8), 'tuple_var_assignment_379678', subscript_call_result_380630)
        
        # Assigning a Name to a Name (line 352):
        # Getting the type of 'tuple_var_assignment_379677' (line 352)
        tuple_var_assignment_379677_380631 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 352, 8), 'tuple_var_assignment_379677')
        # Assigning a type to the variable 'i' (line 352)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 352, 8), 'i', tuple_var_assignment_379677_380631)
        
        # Assigning a Name to a Name (line 352):
        # Getting the type of 'tuple_var_assignment_379678' (line 352)
        tuple_var_assignment_379678_380632 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 352, 8), 'tuple_var_assignment_379678')
        # Assigning a type to the variable 'j' (line 352)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 352, 11), 'j', tuple_var_assignment_379678_380632)
        
        # Assigning a Call to a Name (line 354):
        
        # Assigning a Call to a Name (line 354):
        
        # Call to isinstance(...): (line 354)
        # Processing the call arguments (line 354)
        # Getting the type of 'i' (line 354)
        i_380634 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 354, 29), 'i', False)
        # Getting the type of 'slice' (line 354)
        slice_380635 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 354, 32), 'slice', False)
        # Processing the call keyword arguments (line 354)
        kwargs_380636 = {}
        # Getting the type of 'isinstance' (line 354)
        isinstance_380633 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 354, 18), 'isinstance', False)
        # Calling isinstance(args, kwargs) (line 354)
        isinstance_call_result_380637 = invoke(stypy.reporting.localization.Localization(__file__, 354, 18), isinstance_380633, *[i_380634, slice_380635], **kwargs_380636)
        
        # Assigning a type to the variable 'i_slice' (line 354)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 354, 8), 'i_slice', isinstance_call_result_380637)
        
        # Getting the type of 'i_slice' (line 355)
        i_slice_380638 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 355, 11), 'i_slice')
        # Testing the type of an if condition (line 355)
        if_condition_380639 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 355, 8), i_slice_380638)
        # Assigning a type to the variable 'if_condition_380639' (line 355)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 355, 8), 'if_condition_380639', if_condition_380639)
        # SSA begins for if statement (line 355)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Subscript to a Name (line 356):
        
        # Assigning a Subscript to a Name (line 356):
        
        # Obtaining the type of the subscript
        slice_380640 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 356, 16), None, None, None)
        # Getting the type of 'None' (line 356)
        None_380641 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 356, 57), 'None')
        
        # Call to _slicetoarange(...): (line 356)
        # Processing the call arguments (line 356)
        # Getting the type of 'i' (line 356)
        i_380644 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 356, 36), 'i', False)
        
        # Obtaining the type of the subscript
        int_380645 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 356, 50), 'int')
        # Getting the type of 'self' (line 356)
        self_380646 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 356, 39), 'self', False)
        # Obtaining the member 'shape' of a type (line 356)
        shape_380647 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 356, 39), self_380646, 'shape')
        # Obtaining the member '__getitem__' of a type (line 356)
        getitem___380648 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 356, 39), shape_380647, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 356)
        subscript_call_result_380649 = invoke(stypy.reporting.localization.Localization(__file__, 356, 39), getitem___380648, int_380645)
        
        # Processing the call keyword arguments (line 356)
        kwargs_380650 = {}
        # Getting the type of 'self' (line 356)
        self_380642 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 356, 16), 'self', False)
        # Obtaining the member '_slicetoarange' of a type (line 356)
        _slicetoarange_380643 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 356, 16), self_380642, '_slicetoarange')
        # Calling _slicetoarange(args, kwargs) (line 356)
        _slicetoarange_call_result_380651 = invoke(stypy.reporting.localization.Localization(__file__, 356, 16), _slicetoarange_380643, *[i_380644, subscript_call_result_380649], **kwargs_380650)
        
        # Obtaining the member '__getitem__' of a type (line 356)
        getitem___380652 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 356, 16), _slicetoarange_call_result_380651, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 356)
        subscript_call_result_380653 = invoke(stypy.reporting.localization.Localization(__file__, 356, 16), getitem___380652, (slice_380640, None_380641))
        
        # Assigning a type to the variable 'i' (line 356)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 356, 12), 'i', subscript_call_result_380653)
        # SSA branch for the else part of an if statement (line 355)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Call to a Name (line 358):
        
        # Assigning a Call to a Name (line 358):
        
        # Call to atleast_1d(...): (line 358)
        # Processing the call arguments (line 358)
        # Getting the type of 'i' (line 358)
        i_380656 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 358, 30), 'i', False)
        # Processing the call keyword arguments (line 358)
        kwargs_380657 = {}
        # Getting the type of 'np' (line 358)
        np_380654 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 358, 16), 'np', False)
        # Obtaining the member 'atleast_1d' of a type (line 358)
        atleast_1d_380655 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 358, 16), np_380654, 'atleast_1d')
        # Calling atleast_1d(args, kwargs) (line 358)
        atleast_1d_call_result_380658 = invoke(stypy.reporting.localization.Localization(__file__, 358, 16), atleast_1d_380655, *[i_380656], **kwargs_380657)
        
        # Assigning a type to the variable 'i' (line 358)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 358, 12), 'i', atleast_1d_call_result_380658)
        # SSA join for if statement (line 355)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Type idiom detected: calculating its left and rigth part (line 360)
        # Getting the type of 'slice' (line 360)
        slice_380659 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 360, 25), 'slice')
        # Getting the type of 'j' (line 360)
        j_380660 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 360, 22), 'j')
        
        (may_be_380661, more_types_in_union_380662) = may_be_subtype(slice_380659, j_380660)

        if may_be_380661:

            if more_types_in_union_380662:
                # Runtime conditional SSA (line 360)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            # Assigning a type to the variable 'j' (line 360)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 360, 8), 'j', remove_not_subtype_from_union(j_380660, slice))
            
            # Assigning a Subscript to a Name (line 361):
            
            # Assigning a Subscript to a Name (line 361):
            
            # Obtaining the type of the subscript
            # Getting the type of 'None' (line 361)
            None_380663 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 361, 54), 'None')
            slice_380664 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 361, 16), None, None, None)
            
            # Call to _slicetoarange(...): (line 361)
            # Processing the call arguments (line 361)
            # Getting the type of 'j' (line 361)
            j_380667 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 361, 36), 'j', False)
            
            # Obtaining the type of the subscript
            int_380668 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 361, 50), 'int')
            # Getting the type of 'self' (line 361)
            self_380669 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 361, 39), 'self', False)
            # Obtaining the member 'shape' of a type (line 361)
            shape_380670 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 361, 39), self_380669, 'shape')
            # Obtaining the member '__getitem__' of a type (line 361)
            getitem___380671 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 361, 39), shape_380670, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 361)
            subscript_call_result_380672 = invoke(stypy.reporting.localization.Localization(__file__, 361, 39), getitem___380671, int_380668)
            
            # Processing the call keyword arguments (line 361)
            kwargs_380673 = {}
            # Getting the type of 'self' (line 361)
            self_380665 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 361, 16), 'self', False)
            # Obtaining the member '_slicetoarange' of a type (line 361)
            _slicetoarange_380666 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 361, 16), self_380665, '_slicetoarange')
            # Calling _slicetoarange(args, kwargs) (line 361)
            _slicetoarange_call_result_380674 = invoke(stypy.reporting.localization.Localization(__file__, 361, 16), _slicetoarange_380666, *[j_380667, subscript_call_result_380672], **kwargs_380673)
            
            # Obtaining the member '__getitem__' of a type (line 361)
            getitem___380675 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 361, 16), _slicetoarange_call_result_380674, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 361)
            subscript_call_result_380676 = invoke(stypy.reporting.localization.Localization(__file__, 361, 16), getitem___380675, (None_380663, slice_380664))
            
            # Assigning a type to the variable 'j' (line 361)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 361, 12), 'j', subscript_call_result_380676)
            
            
            # Getting the type of 'i' (line 362)
            i_380677 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 362, 15), 'i')
            # Obtaining the member 'ndim' of a type (line 362)
            ndim_380678 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 362, 15), i_380677, 'ndim')
            int_380679 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 362, 25), 'int')
            # Applying the binary operator '==' (line 362)
            result_eq_380680 = python_operator(stypy.reporting.localization.Localization(__file__, 362, 15), '==', ndim_380678, int_380679)
            
            # Testing the type of an if condition (line 362)
            if_condition_380681 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 362, 12), result_eq_380680)
            # Assigning a type to the variable 'if_condition_380681' (line 362)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 362, 12), 'if_condition_380681', if_condition_380681)
            # SSA begins for if statement (line 362)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a Subscript to a Name (line 363):
            
            # Assigning a Subscript to a Name (line 363):
            
            # Obtaining the type of the subscript
            slice_380682 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 363, 20), None, None, None)
            # Getting the type of 'None' (line 363)
            None_380683 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 363, 25), 'None')
            # Getting the type of 'i' (line 363)
            i_380684 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 363, 20), 'i')
            # Obtaining the member '__getitem__' of a type (line 363)
            getitem___380685 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 363, 20), i_380684, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 363)
            subscript_call_result_380686 = invoke(stypy.reporting.localization.Localization(__file__, 363, 20), getitem___380685, (slice_380682, None_380683))
            
            # Assigning a type to the variable 'i' (line 363)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 363, 16), 'i', subscript_call_result_380686)
            # SSA branch for the else part of an if statement (line 362)
            module_type_store.open_ssa_branch('else')
            
            
            # Getting the type of 'i_slice' (line 364)
            i_slice_380687 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 364, 21), 'i_slice')
            # Applying the 'not' unary operator (line 364)
            result_not__380688 = python_operator(stypy.reporting.localization.Localization(__file__, 364, 17), 'not', i_slice_380687)
            
            # Testing the type of an if condition (line 364)
            if_condition_380689 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 364, 17), result_not__380688)
            # Assigning a type to the variable 'if_condition_380689' (line 364)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 364, 17), 'if_condition_380689', if_condition_380689)
            # SSA begins for if statement (line 364)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to IndexError(...): (line 365)
            # Processing the call arguments (line 365)
            str_380691 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 365, 33), 'str', 'index returns 3-dim structure')
            # Processing the call keyword arguments (line 365)
            kwargs_380692 = {}
            # Getting the type of 'IndexError' (line 365)
            IndexError_380690 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 365, 22), 'IndexError', False)
            # Calling IndexError(args, kwargs) (line 365)
            IndexError_call_result_380693 = invoke(stypy.reporting.localization.Localization(__file__, 365, 22), IndexError_380690, *[str_380691], **kwargs_380692)
            
            ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 365, 16), IndexError_call_result_380693, 'raise parameter', BaseException)
            # SSA join for if statement (line 364)
            module_type_store = module_type_store.join_ssa_context()
            
            # SSA join for if statement (line 362)
            module_type_store = module_type_store.join_ssa_context()
            

            if more_types_in_union_380662:
                # Runtime conditional SSA for else branch (line 360)
                module_type_store.open_ssa_branch('idiom else')



        if ((not may_be_380661) or more_types_in_union_380662):
            # Assigning a type to the variable 'j' (line 360)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 360, 8), 'j', remove_subtype_from_union(j_380660, slice))
            
            
            # Call to isscalarlike(...): (line 366)
            # Processing the call arguments (line 366)
            # Getting the type of 'j' (line 366)
            j_380695 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 366, 26), 'j', False)
            # Processing the call keyword arguments (line 366)
            kwargs_380696 = {}
            # Getting the type of 'isscalarlike' (line 366)
            isscalarlike_380694 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 366, 13), 'isscalarlike', False)
            # Calling isscalarlike(args, kwargs) (line 366)
            isscalarlike_call_result_380697 = invoke(stypy.reporting.localization.Localization(__file__, 366, 13), isscalarlike_380694, *[j_380695], **kwargs_380696)
            
            # Testing the type of an if condition (line 366)
            if_condition_380698 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 366, 13), isscalarlike_call_result_380697)
            # Assigning a type to the variable 'if_condition_380698' (line 366)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 366, 13), 'if_condition_380698', if_condition_380698)
            # SSA begins for if statement (line 366)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a Call to a Name (line 368):
            
            # Assigning a Call to a Name (line 368):
            
            # Call to atleast_1d(...): (line 368)
            # Processing the call arguments (line 368)
            # Getting the type of 'j' (line 368)
            j_380701 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 368, 30), 'j', False)
            # Processing the call keyword arguments (line 368)
            kwargs_380702 = {}
            # Getting the type of 'np' (line 368)
            np_380699 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 368, 16), 'np', False)
            # Obtaining the member 'atleast_1d' of a type (line 368)
            atleast_1d_380700 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 368, 16), np_380699, 'atleast_1d')
            # Calling atleast_1d(args, kwargs) (line 368)
            atleast_1d_call_result_380703 = invoke(stypy.reporting.localization.Localization(__file__, 368, 16), atleast_1d_380700, *[j_380701], **kwargs_380702)
            
            # Assigning a type to the variable 'j' (line 368)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 368, 12), 'j', atleast_1d_call_result_380703)
            
            
            # Getting the type of 'i' (line 369)
            i_380704 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 369, 15), 'i')
            # Obtaining the member 'ndim' of a type (line 369)
            ndim_380705 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 369, 15), i_380704, 'ndim')
            int_380706 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 369, 25), 'int')
            # Applying the binary operator '==' (line 369)
            result_eq_380707 = python_operator(stypy.reporting.localization.Localization(__file__, 369, 15), '==', ndim_380705, int_380706)
            
            # Testing the type of an if condition (line 369)
            if_condition_380708 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 369, 12), result_eq_380707)
            # Assigning a type to the variable 'if_condition_380708' (line 369)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 369, 12), 'if_condition_380708', if_condition_380708)
            # SSA begins for if statement (line 369)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a Call to a Tuple (line 370):
            
            # Assigning a Subscript to a Name (line 370):
            
            # Obtaining the type of the subscript
            int_380709 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 370, 16), 'int')
            
            # Call to broadcast_arrays(...): (line 370)
            # Processing the call arguments (line 370)
            # Getting the type of 'i' (line 370)
            i_380712 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 370, 43), 'i', False)
            # Getting the type of 'j' (line 370)
            j_380713 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 370, 46), 'j', False)
            # Processing the call keyword arguments (line 370)
            kwargs_380714 = {}
            # Getting the type of 'np' (line 370)
            np_380710 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 370, 23), 'np', False)
            # Obtaining the member 'broadcast_arrays' of a type (line 370)
            broadcast_arrays_380711 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 370, 23), np_380710, 'broadcast_arrays')
            # Calling broadcast_arrays(args, kwargs) (line 370)
            broadcast_arrays_call_result_380715 = invoke(stypy.reporting.localization.Localization(__file__, 370, 23), broadcast_arrays_380711, *[i_380712, j_380713], **kwargs_380714)
            
            # Obtaining the member '__getitem__' of a type (line 370)
            getitem___380716 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 370, 16), broadcast_arrays_call_result_380715, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 370)
            subscript_call_result_380717 = invoke(stypy.reporting.localization.Localization(__file__, 370, 16), getitem___380716, int_380709)
            
            # Assigning a type to the variable 'tuple_var_assignment_379679' (line 370)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 370, 16), 'tuple_var_assignment_379679', subscript_call_result_380717)
            
            # Assigning a Subscript to a Name (line 370):
            
            # Obtaining the type of the subscript
            int_380718 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 370, 16), 'int')
            
            # Call to broadcast_arrays(...): (line 370)
            # Processing the call arguments (line 370)
            # Getting the type of 'i' (line 370)
            i_380721 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 370, 43), 'i', False)
            # Getting the type of 'j' (line 370)
            j_380722 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 370, 46), 'j', False)
            # Processing the call keyword arguments (line 370)
            kwargs_380723 = {}
            # Getting the type of 'np' (line 370)
            np_380719 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 370, 23), 'np', False)
            # Obtaining the member 'broadcast_arrays' of a type (line 370)
            broadcast_arrays_380720 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 370, 23), np_380719, 'broadcast_arrays')
            # Calling broadcast_arrays(args, kwargs) (line 370)
            broadcast_arrays_call_result_380724 = invoke(stypy.reporting.localization.Localization(__file__, 370, 23), broadcast_arrays_380720, *[i_380721, j_380722], **kwargs_380723)
            
            # Obtaining the member '__getitem__' of a type (line 370)
            getitem___380725 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 370, 16), broadcast_arrays_call_result_380724, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 370)
            subscript_call_result_380726 = invoke(stypy.reporting.localization.Localization(__file__, 370, 16), getitem___380725, int_380718)
            
            # Assigning a type to the variable 'tuple_var_assignment_379680' (line 370)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 370, 16), 'tuple_var_assignment_379680', subscript_call_result_380726)
            
            # Assigning a Name to a Name (line 370):
            # Getting the type of 'tuple_var_assignment_379679' (line 370)
            tuple_var_assignment_379679_380727 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 370, 16), 'tuple_var_assignment_379679')
            # Assigning a type to the variable 'i' (line 370)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 370, 16), 'i', tuple_var_assignment_379679_380727)
            
            # Assigning a Name to a Name (line 370):
            # Getting the type of 'tuple_var_assignment_379680' (line 370)
            tuple_var_assignment_379680_380728 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 370, 16), 'tuple_var_assignment_379680')
            # Assigning a type to the variable 'j' (line 370)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 370, 19), 'j', tuple_var_assignment_379680_380728)
            
            # Assigning a Subscript to a Name (line 371):
            
            # Assigning a Subscript to a Name (line 371):
            
            # Obtaining the type of the subscript
            slice_380729 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 371, 20), None, None, None)
            # Getting the type of 'None' (line 371)
            None_380730 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 371, 25), 'None')
            # Getting the type of 'i' (line 371)
            i_380731 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 371, 20), 'i')
            # Obtaining the member '__getitem__' of a type (line 371)
            getitem___380732 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 371, 20), i_380731, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 371)
            subscript_call_result_380733 = invoke(stypy.reporting.localization.Localization(__file__, 371, 20), getitem___380732, (slice_380729, None_380730))
            
            # Assigning a type to the variable 'i' (line 371)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 371, 16), 'i', subscript_call_result_380733)
            
            # Assigning a Subscript to a Name (line 372):
            
            # Assigning a Subscript to a Name (line 372):
            
            # Obtaining the type of the subscript
            slice_380734 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 372, 20), None, None, None)
            # Getting the type of 'None' (line 372)
            None_380735 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 372, 25), 'None')
            # Getting the type of 'j' (line 372)
            j_380736 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 372, 20), 'j')
            # Obtaining the member '__getitem__' of a type (line 372)
            getitem___380737 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 372, 20), j_380736, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 372)
            subscript_call_result_380738 = invoke(stypy.reporting.localization.Localization(__file__, 372, 20), getitem___380737, (slice_380734, None_380735))
            
            # Assigning a type to the variable 'j' (line 372)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 372, 16), 'j', subscript_call_result_380738)
            
            # Obtaining an instance of the builtin type 'tuple' (line 373)
            tuple_380739 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 373, 23), 'tuple')
            # Adding type elements to the builtin type 'tuple' instance (line 373)
            # Adding element type (line 373)
            # Getting the type of 'i' (line 373)
            i_380740 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 373, 23), 'i')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 373, 23), tuple_380739, i_380740)
            # Adding element type (line 373)
            # Getting the type of 'j' (line 373)
            j_380741 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 373, 26), 'j')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 373, 23), tuple_380739, j_380741)
            
            # Assigning a type to the variable 'stypy_return_type' (line 373)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 373, 16), 'stypy_return_type', tuple_380739)
            # SSA join for if statement (line 369)
            module_type_store = module_type_store.join_ssa_context()
            
            # SSA branch for the else part of an if statement (line 366)
            module_type_store.open_ssa_branch('else')
            
            # Assigning a Call to a Name (line 375):
            
            # Assigning a Call to a Name (line 375):
            
            # Call to atleast_1d(...): (line 375)
            # Processing the call arguments (line 375)
            # Getting the type of 'j' (line 375)
            j_380744 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 375, 30), 'j', False)
            # Processing the call keyword arguments (line 375)
            kwargs_380745 = {}
            # Getting the type of 'np' (line 375)
            np_380742 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 375, 16), 'np', False)
            # Obtaining the member 'atleast_1d' of a type (line 375)
            atleast_1d_380743 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 375, 16), np_380742, 'atleast_1d')
            # Calling atleast_1d(args, kwargs) (line 375)
            atleast_1d_call_result_380746 = invoke(stypy.reporting.localization.Localization(__file__, 375, 16), atleast_1d_380743, *[j_380744], **kwargs_380745)
            
            # Assigning a type to the variable 'j' (line 375)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 375, 12), 'j', atleast_1d_call_result_380746)
            
            
            # Evaluating a boolean operation
            # Getting the type of 'i_slice' (line 376)
            i_slice_380747 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 376, 15), 'i_slice')
            
            # Getting the type of 'j' (line 376)
            j_380748 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 376, 27), 'j')
            # Obtaining the member 'ndim' of a type (line 376)
            ndim_380749 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 376, 27), j_380748, 'ndim')
            int_380750 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 376, 36), 'int')
            # Applying the binary operator '>' (line 376)
            result_gt_380751 = python_operator(stypy.reporting.localization.Localization(__file__, 376, 27), '>', ndim_380749, int_380750)
            
            # Applying the binary operator 'and' (line 376)
            result_and_keyword_380752 = python_operator(stypy.reporting.localization.Localization(__file__, 376, 15), 'and', i_slice_380747, result_gt_380751)
            
            # Testing the type of an if condition (line 376)
            if_condition_380753 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 376, 12), result_and_keyword_380752)
            # Assigning a type to the variable 'if_condition_380753' (line 376)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 376, 12), 'if_condition_380753', if_condition_380753)
            # SSA begins for if statement (line 376)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to IndexError(...): (line 377)
            # Processing the call arguments (line 377)
            str_380755 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 377, 33), 'str', 'index returns 3-dim structure')
            # Processing the call keyword arguments (line 377)
            kwargs_380756 = {}
            # Getting the type of 'IndexError' (line 377)
            IndexError_380754 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 377, 22), 'IndexError', False)
            # Calling IndexError(args, kwargs) (line 377)
            IndexError_call_result_380757 = invoke(stypy.reporting.localization.Localization(__file__, 377, 22), IndexError_380754, *[str_380755], **kwargs_380756)
            
            ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 377, 16), IndexError_call_result_380757, 'raise parameter', BaseException)
            # SSA join for if statement (line 376)
            module_type_store = module_type_store.join_ssa_context()
            
            # SSA join for if statement (line 366)
            module_type_store = module_type_store.join_ssa_context()
            

            if (may_be_380661 and more_types_in_union_380662):
                # SSA join for if statement (line 360)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Assigning a Call to a Tuple (line 379):
        
        # Assigning a Subscript to a Name (line 379):
        
        # Obtaining the type of the subscript
        int_380758 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 379, 8), 'int')
        
        # Call to broadcast_arrays(...): (line 379)
        # Processing the call arguments (line 379)
        # Getting the type of 'i' (line 379)
        i_380761 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 379, 35), 'i', False)
        # Getting the type of 'j' (line 379)
        j_380762 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 379, 38), 'j', False)
        # Processing the call keyword arguments (line 379)
        kwargs_380763 = {}
        # Getting the type of 'np' (line 379)
        np_380759 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 379, 15), 'np', False)
        # Obtaining the member 'broadcast_arrays' of a type (line 379)
        broadcast_arrays_380760 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 379, 15), np_380759, 'broadcast_arrays')
        # Calling broadcast_arrays(args, kwargs) (line 379)
        broadcast_arrays_call_result_380764 = invoke(stypy.reporting.localization.Localization(__file__, 379, 15), broadcast_arrays_380760, *[i_380761, j_380762], **kwargs_380763)
        
        # Obtaining the member '__getitem__' of a type (line 379)
        getitem___380765 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 379, 8), broadcast_arrays_call_result_380764, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 379)
        subscript_call_result_380766 = invoke(stypy.reporting.localization.Localization(__file__, 379, 8), getitem___380765, int_380758)
        
        # Assigning a type to the variable 'tuple_var_assignment_379681' (line 379)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 379, 8), 'tuple_var_assignment_379681', subscript_call_result_380766)
        
        # Assigning a Subscript to a Name (line 379):
        
        # Obtaining the type of the subscript
        int_380767 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 379, 8), 'int')
        
        # Call to broadcast_arrays(...): (line 379)
        # Processing the call arguments (line 379)
        # Getting the type of 'i' (line 379)
        i_380770 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 379, 35), 'i', False)
        # Getting the type of 'j' (line 379)
        j_380771 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 379, 38), 'j', False)
        # Processing the call keyword arguments (line 379)
        kwargs_380772 = {}
        # Getting the type of 'np' (line 379)
        np_380768 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 379, 15), 'np', False)
        # Obtaining the member 'broadcast_arrays' of a type (line 379)
        broadcast_arrays_380769 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 379, 15), np_380768, 'broadcast_arrays')
        # Calling broadcast_arrays(args, kwargs) (line 379)
        broadcast_arrays_call_result_380773 = invoke(stypy.reporting.localization.Localization(__file__, 379, 15), broadcast_arrays_380769, *[i_380770, j_380771], **kwargs_380772)
        
        # Obtaining the member '__getitem__' of a type (line 379)
        getitem___380774 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 379, 8), broadcast_arrays_call_result_380773, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 379)
        subscript_call_result_380775 = invoke(stypy.reporting.localization.Localization(__file__, 379, 8), getitem___380774, int_380767)
        
        # Assigning a type to the variable 'tuple_var_assignment_379682' (line 379)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 379, 8), 'tuple_var_assignment_379682', subscript_call_result_380775)
        
        # Assigning a Name to a Name (line 379):
        # Getting the type of 'tuple_var_assignment_379681' (line 379)
        tuple_var_assignment_379681_380776 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 379, 8), 'tuple_var_assignment_379681')
        # Assigning a type to the variable 'i' (line 379)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 379, 8), 'i', tuple_var_assignment_379681_380776)
        
        # Assigning a Name to a Name (line 379):
        # Getting the type of 'tuple_var_assignment_379682' (line 379)
        tuple_var_assignment_379682_380777 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 379, 8), 'tuple_var_assignment_379682')
        # Assigning a type to the variable 'j' (line 379)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 379, 11), 'j', tuple_var_assignment_379682_380777)
        
        
        # Getting the type of 'i' (line 381)
        i_380778 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 381, 11), 'i')
        # Obtaining the member 'ndim' of a type (line 381)
        ndim_380779 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 381, 11), i_380778, 'ndim')
        int_380780 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 381, 21), 'int')
        # Applying the binary operator '==' (line 381)
        result_eq_380781 = python_operator(stypy.reporting.localization.Localization(__file__, 381, 11), '==', ndim_380779, int_380780)
        
        # Testing the type of an if condition (line 381)
        if_condition_380782 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 381, 8), result_eq_380781)
        # Assigning a type to the variable 'if_condition_380782' (line 381)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 381, 8), 'if_condition_380782', if_condition_380782)
        # SSA begins for if statement (line 381)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Subscript to a Name (line 383):
        
        # Assigning a Subscript to a Name (line 383):
        
        # Obtaining the type of the subscript
        # Getting the type of 'None' (line 383)
        None_380783 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 383, 18), 'None')
        slice_380784 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 383, 16), None, None, None)
        # Getting the type of 'i' (line 383)
        i_380785 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 383, 16), 'i')
        # Obtaining the member '__getitem__' of a type (line 383)
        getitem___380786 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 383, 16), i_380785, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 383)
        subscript_call_result_380787 = invoke(stypy.reporting.localization.Localization(__file__, 383, 16), getitem___380786, (None_380783, slice_380784))
        
        # Assigning a type to the variable 'i' (line 383)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 383, 12), 'i', subscript_call_result_380787)
        
        # Assigning a Subscript to a Name (line 384):
        
        # Assigning a Subscript to a Name (line 384):
        
        # Obtaining the type of the subscript
        # Getting the type of 'None' (line 384)
        None_380788 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 384, 18), 'None')
        slice_380789 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 384, 16), None, None, None)
        # Getting the type of 'j' (line 384)
        j_380790 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 384, 16), 'j')
        # Obtaining the member '__getitem__' of a type (line 384)
        getitem___380791 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 384, 16), j_380790, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 384)
        subscript_call_result_380792 = invoke(stypy.reporting.localization.Localization(__file__, 384, 16), getitem___380791, (None_380788, slice_380789))
        
        # Assigning a type to the variable 'j' (line 384)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 384, 12), 'j', subscript_call_result_380792)
        # SSA branch for the else part of an if statement (line 381)
        module_type_store.open_ssa_branch('else')
        
        
        # Getting the type of 'i' (line 385)
        i_380793 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 385, 13), 'i')
        # Obtaining the member 'ndim' of a type (line 385)
        ndim_380794 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 385, 13), i_380793, 'ndim')
        int_380795 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 385, 22), 'int')
        # Applying the binary operator '>' (line 385)
        result_gt_380796 = python_operator(stypy.reporting.localization.Localization(__file__, 385, 13), '>', ndim_380794, int_380795)
        
        # Testing the type of an if condition (line 385)
        if_condition_380797 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 385, 13), result_gt_380796)
        # Assigning a type to the variable 'if_condition_380797' (line 385)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 385, 13), 'if_condition_380797', if_condition_380797)
        # SSA begins for if statement (line 385)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to IndexError(...): (line 386)
        # Processing the call arguments (line 386)
        str_380799 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 386, 29), 'str', 'Index dimension must be <= 2')
        # Processing the call keyword arguments (line 386)
        kwargs_380800 = {}
        # Getting the type of 'IndexError' (line 386)
        IndexError_380798 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 386, 18), 'IndexError', False)
        # Calling IndexError(args, kwargs) (line 386)
        IndexError_call_result_380801 = invoke(stypy.reporting.localization.Localization(__file__, 386, 18), IndexError_380798, *[str_380799], **kwargs_380800)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 386, 12), IndexError_call_result_380801, 'raise parameter', BaseException)
        # SSA join for if statement (line 385)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 381)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Obtaining an instance of the builtin type 'tuple' (line 388)
        tuple_380802 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 388, 15), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 388)
        # Adding element type (line 388)
        # Getting the type of 'i' (line 388)
        i_380803 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 388, 15), 'i')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 388, 15), tuple_380802, i_380803)
        # Adding element type (line 388)
        # Getting the type of 'j' (line 388)
        j_380804 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 388, 18), 'j')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 388, 15), tuple_380802, j_380804)
        
        # Assigning a type to the variable 'stypy_return_type' (line 388)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 388, 8), 'stypy_return_type', tuple_380802)
        
        # ################# End of '_index_to_arrays(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_index_to_arrays' in the type store
        # Getting the type of 'stypy_return_type' (line 351)
        stypy_return_type_380805 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 351, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_380805)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_index_to_arrays'
        return stypy_return_type_380805


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 254, 0, False)
        # Assigning a type to the variable 'self' (line 255)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 255, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'IndexMixin.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'IndexMixin' (line 254)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 254, 0), 'IndexMixin', IndexMixin)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
