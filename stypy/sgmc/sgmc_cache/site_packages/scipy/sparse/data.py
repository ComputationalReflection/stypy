
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: '''Base class for sparse matrice with a .data attribute
2: 
3:     subclasses must provide a _with_data() method that
4:     creates a new matrix with the same sparsity pattern
5:     as self but with a different data array
6: 
7: '''
8: 
9: from __future__ import division, print_function, absolute_import
10: 
11: import numpy as np
12: 
13: from .base import spmatrix, _ufuncs_with_fixed_point_at_zero
14: from .sputils import isscalarlike, validateaxis
15: 
16: __all__ = []
17: 
18: 
19: # TODO implement all relevant operations
20: # use .data.__methods__() instead of /=, *=, etc.
21: class _data_matrix(spmatrix):
22:     def __init__(self):
23:         spmatrix.__init__(self)
24: 
25:     def _get_dtype(self):
26:         return self.data.dtype
27: 
28:     def _set_dtype(self, newtype):
29:         self.data.dtype = newtype
30:     dtype = property(fget=_get_dtype, fset=_set_dtype)
31: 
32:     def _deduped_data(self):
33:         if hasattr(self, 'sum_duplicates'):
34:             self.sum_duplicates()
35:         return self.data
36: 
37:     def __abs__(self):
38:         return self._with_data(abs(self._deduped_data()))
39: 
40:     def _real(self):
41:         return self._with_data(self.data.real)
42: 
43:     def _imag(self):
44:         return self._with_data(self.data.imag)
45: 
46:     def __neg__(self):
47:         if self.dtype.kind == 'b':
48:             raise NotImplementedError('negating a sparse boolean '
49:                                       'matrix is not supported')
50:         return self._with_data(-self.data)
51: 
52:     def __imul__(self, other):  # self *= other
53:         if isscalarlike(other):
54:             self.data *= other
55:             return self
56:         else:
57:             return NotImplemented
58: 
59:     def __itruediv__(self, other):  # self /= other
60:         if isscalarlike(other):
61:             recip = 1.0 / other
62:             self.data *= recip
63:             return self
64:         else:
65:             return NotImplemented
66: 
67:     def astype(self, dtype, casting='unsafe', copy=True):
68:         dtype = np.dtype(dtype)
69:         if self.dtype != dtype:
70:             return self._with_data(
71:                 self._deduped_data().astype(dtype, casting=casting, copy=copy),
72:                 copy=copy)
73:         elif copy:
74:             return self.copy()
75:         else:
76:             return self
77: 
78:     astype.__doc__ = spmatrix.astype.__doc__
79: 
80:     def conj(self):
81:         return self._with_data(self.data.conj())
82: 
83:     conj.__doc__ = spmatrix.conj.__doc__
84: 
85:     def copy(self):
86:         return self._with_data(self.data.copy(), copy=True)
87: 
88:     copy.__doc__ = spmatrix.copy.__doc__
89: 
90:     def count_nonzero(self):
91:         return np.count_nonzero(self._deduped_data())
92: 
93:     count_nonzero.__doc__ = spmatrix.count_nonzero.__doc__
94: 
95:     def power(self, n, dtype=None):
96:         '''
97:         This function performs element-wise power.
98: 
99:         Parameters
100:         ----------
101:         n : n is a scalar
102: 
103:         dtype : If dtype is not specified, the current dtype will be preserved.
104:         '''
105:         if not isscalarlike(n):
106:             raise NotImplementedError("input is not scalar")
107: 
108:         data = self._deduped_data()
109:         if dtype is not None:
110:             data = data.astype(dtype)
111:         return self._with_data(data ** n)
112: 
113:     ###########################
114:     # Multiplication handlers #
115:     ###########################
116: 
117:     def _mul_scalar(self, other):
118:         return self._with_data(self.data * other)
119: 
120: 
121: # Add the numpy unary ufuncs for which func(0) = 0 to _data_matrix.
122: for npfunc in _ufuncs_with_fixed_point_at_zero:
123:     name = npfunc.__name__
124: 
125:     def _create_method(op):
126:         def method(self):
127:             result = op(self.data)
128:             x = self._with_data(result, copy=True)
129:             return x
130: 
131:         method.__doc__ = ("Element-wise %s.\n\n"
132:                           "See numpy.%s for more information." % (name, name))
133:         method.__name__ = name
134: 
135:         return method
136: 
137:     setattr(_data_matrix, name, _create_method(npfunc))
138: 
139: 
140: def _find_missing_index(ind, n):
141:     for k, a in enumerate(ind):
142:         if k != a:
143:             return k
144: 
145:     k += 1
146:     if k < n:
147:         return k
148:     else:
149:         return -1
150: 
151: 
152: class _minmax_mixin(object):
153:     '''Mixin for min and max methods.
154: 
155:     These are not implemented for dia_matrix, hence the separate class.
156:     '''
157: 
158:     def _min_or_max_axis(self, axis, min_or_max):
159:         N = self.shape[axis]
160:         if N == 0:
161:             raise ValueError("zero-size array to reduction operation")
162:         M = self.shape[1 - axis]
163: 
164:         mat = self.tocsc() if axis == 0 else self.tocsr()
165:         mat.sum_duplicates()
166: 
167:         major_index, value = mat._minor_reduce(min_or_max)
168:         not_full = np.diff(mat.indptr)[major_index] < N
169:         value[not_full] = min_or_max(value[not_full], 0)
170: 
171:         mask = value != 0
172:         major_index = np.compress(mask, major_index)
173:         value = np.compress(mask, value)
174: 
175:         from . import coo_matrix
176:         if axis == 0:
177:             return coo_matrix((value, (np.zeros(len(value)), major_index)),
178:                               dtype=self.dtype, shape=(1, M))
179:         else:
180:             return coo_matrix((value, (major_index, np.zeros(len(value)))),
181:                               dtype=self.dtype, shape=(M, 1))
182: 
183:     def _min_or_max(self, axis, out, min_or_max):
184:         if out is not None:
185:             raise ValueError(("Sparse matrices do not support "
186:                               "an 'out' parameter."))
187: 
188:         validateaxis(axis)
189: 
190:         if axis is None:
191:             if 0 in self.shape:
192:                 raise ValueError("zero-size array to reduction operation")
193: 
194:             zero = self.dtype.type(0)
195:             if self.nnz == 0:
196:                 return zero
197:             m = min_or_max.reduce(self._deduped_data().ravel())
198:             if self.nnz != np.product(self.shape):
199:                 m = min_or_max(zero, m)
200:             return m
201: 
202:         if axis < 0:
203:             axis += 2
204: 
205:         if (axis == 0) or (axis == 1):
206:             return self._min_or_max_axis(axis, min_or_max)
207:         else:
208:             raise ValueError("axis out of range")
209: 
210:     def _arg_min_or_max_axis(self, axis, op, compare):
211:         if self.shape[axis] == 0:
212:             raise ValueError("Can't apply the operation along a zero-sized "
213:                              "dimension.")
214: 
215:         if axis < 0:
216:             axis += 2
217: 
218:         zero = self.dtype.type(0)
219: 
220:         mat = self.tocsc() if axis == 0 else self.tocsr()
221:         mat.sum_duplicates()
222: 
223:         ret_size, line_size = mat._swap(mat.shape)
224:         ret = np.zeros(ret_size, dtype=int)
225: 
226:         nz_lines, = np.nonzero(np.diff(mat.indptr))
227:         for i in nz_lines:
228:             p, q = mat.indptr[i:i + 2]
229:             data = mat.data[p:q]
230:             indices = mat.indices[p:q]
231:             am = op(data)
232:             m = data[am]
233:             if compare(m, zero) or q - p == line_size:
234:                 ret[i] = indices[am]
235:             else:
236:                 zero_ind = _find_missing_index(indices, line_size)
237:                 if m == zero:
238:                     ret[i] = min(am, zero_ind)
239:                 else:
240:                     ret[i] = zero_ind
241: 
242:         if axis == 1:
243:             ret = ret.reshape(-1, 1)
244: 
245:         return np.asmatrix(ret)
246: 
247:     def _arg_min_or_max(self, axis, out, op, compare):
248:         if out is not None:
249:             raise ValueError("Sparse matrices do not support "
250:                              "an 'out' parameter.")
251: 
252:         validateaxis(axis)
253: 
254:         if axis is None:
255:             if 0 in self.shape:
256:                 raise ValueError("Can't apply the operation to "
257:                                  "an empty matrix.")
258: 
259:             if self.nnz == 0:
260:                 return 0
261:             else:
262:                 zero = self.dtype.type(0)
263:                 mat = self.tocoo()
264:                 mat.sum_duplicates()
265:                 am = op(mat.data)
266:                 m = mat.data[am]
267: 
268:                 if compare(m, zero):
269:                     return mat.row[am] * mat.shape[1] + mat.col[am]
270:                 else:
271:                     size = np.product(mat.shape)
272:                     if size == mat.nnz:
273:                         return am
274:                     else:
275:                         ind = mat.row * mat.shape[1] + mat.col
276:                         zero_ind = _find_missing_index(ind, size)
277:                         if m == zero:
278:                             return min(zero_ind, am)
279:                         else:
280:                             return zero_ind
281: 
282:         return self._arg_min_or_max_axis(axis, op, compare)
283: 
284:     def max(self, axis=None, out=None):
285:         '''
286:         Return the maximum of the matrix or maximum along an axis.
287:         This takes all elements into account, not just the non-zero ones.
288: 
289:         Parameters
290:         ----------
291:         axis : {-2, -1, 0, 1, None} optional
292:             Axis along which the sum is computed. The default is to
293:             compute the maximum over all the matrix elements, returning
294:             a scalar (i.e. `axis` = `None`).
295: 
296:         out : None, optional
297:             This argument is in the signature *solely* for NumPy
298:             compatibility reasons. Do not pass in anything except
299:             for the default value, as this argument is not used.
300: 
301:         Returns
302:         -------
303:         amax : coo_matrix or scalar
304:             Maximum of `a`. If `axis` is None, the result is a scalar value.
305:             If `axis` is given, the result is a sparse.coo_matrix of dimension
306:             ``a.ndim - 1``.
307: 
308:         See Also
309:         --------
310:         min : The minimum value of a sparse matrix along a given axis.
311:         np.matrix.max : NumPy's implementation of 'max' for matrices
312: 
313:         '''
314:         return self._min_or_max(axis, out, np.maximum)
315: 
316:     def min(self, axis=None, out=None):
317:         '''
318:         Return the minimum of the matrix or maximum along an axis.
319:         This takes all elements into account, not just the non-zero ones.
320: 
321:         Parameters
322:         ----------
323:         axis : {-2, -1, 0, 1, None} optional
324:             Axis along which the sum is computed. The default is to
325:             compute the minimum over all the matrix elements, returning
326:             a scalar (i.e. `axis` = `None`).
327: 
328:         out : None, optional
329:             This argument is in the signature *solely* for NumPy
330:             compatibility reasons. Do not pass in anything except for
331:             the default value, as this argument is not used.
332: 
333:         Returns
334:         -------
335:         amin : coo_matrix or scalar
336:             Minimum of `a`. If `axis` is None, the result is a scalar value.
337:             If `axis` is given, the result is a sparse.coo_matrix of dimension
338:             ``a.ndim - 1``.
339: 
340:         See Also
341:         --------
342:         max : The maximum value of a sparse matrix along a given axis.
343:         np.matrix.min : NumPy's implementation of 'min' for matrices
344: 
345:         '''
346:         return self._min_or_max(axis, out, np.minimum)
347: 
348:     def argmax(self, axis=None, out=None):
349:         '''Return indices of maximum elements along an axis.
350: 
351:         Implicit zero elements are also taken into account. If there are
352:         several maximum values, the index of the first occurrence is returned.
353: 
354:         Parameters
355:         ----------
356:         axis : {-2, -1, 0, 1, None}, optional
357:             Axis along which the argmax is computed. If None (default), index
358:             of the maximum element in the flatten data is returned.
359:         out : None, optional
360:             This argument is in the signature *solely* for NumPy
361:             compatibility reasons. Do not pass in anything except for
362:             the default value, as this argument is not used.
363: 
364:         Returns
365:         -------
366:         ind : np.matrix or int
367:             Indices of maximum elements. If matrix, its size along `axis` is 1.
368:         '''
369:         return self._arg_min_or_max(axis, out, np.argmax, np.greater)
370: 
371:     def argmin(self, axis=None, out=None):
372:         '''Return indices of minimum elements along an axis.
373: 
374:         Implicit zero elements are also taken into account. If there are
375:         several minimum values, the index of the first occurrence is returned.
376: 
377:         Parameters
378:         ----------
379:         axis : {-2, -1, 0, 1, None}, optional
380:             Axis along which the argmin is computed. If None (default), index
381:             of the minimum element in the flatten data is returned.
382:         out : None, optional
383:             This argument is in the signature *solely* for NumPy
384:             compatibility reasons. Do not pass in anything except for
385:             the default value, as this argument is not used.
386: 
387:         Returns
388:         -------
389:          ind : np.matrix or int
390:             Indices of minimum elements. If matrix, its size along `axis` is 1.
391:         '''
392:         return self._arg_min_or_max(axis, out, np.argmin, np.less)
393: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_372049 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, (-1)), 'str', 'Base class for sparse matrice with a .data attribute\n\n    subclasses must provide a _with_data() method that\n    creates a new matrix with the same sparsity pattern\n    as self but with a different data array\n\n')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 11, 0))

# 'import numpy' statement (line 11)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/sparse/')
import_372050 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'numpy')

if (type(import_372050) is not StypyTypeError):

    if (import_372050 != 'pyd_module'):
        __import__(import_372050)
        sys_modules_372051 = sys.modules[import_372050]
        import_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'np', sys_modules_372051.module_type_store, module_type_store)
    else:
        import numpy as np

        import_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'np', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 11)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 0), 'numpy', import_372050)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/sparse/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 13, 0))

# 'from scipy.sparse.base import spmatrix, _ufuncs_with_fixed_point_at_zero' statement (line 13)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/sparse/')
import_372052 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 13, 0), 'scipy.sparse.base')

if (type(import_372052) is not StypyTypeError):

    if (import_372052 != 'pyd_module'):
        __import__(import_372052)
        sys_modules_372053 = sys.modules[import_372052]
        import_from_module(stypy.reporting.localization.Localization(__file__, 13, 0), 'scipy.sparse.base', sys_modules_372053.module_type_store, module_type_store, ['spmatrix', '_ufuncs_with_fixed_point_at_zero'])
        nest_module(stypy.reporting.localization.Localization(__file__, 13, 0), __file__, sys_modules_372053, sys_modules_372053.module_type_store, module_type_store)
    else:
        from scipy.sparse.base import spmatrix, _ufuncs_with_fixed_point_at_zero

        import_from_module(stypy.reporting.localization.Localization(__file__, 13, 0), 'scipy.sparse.base', None, module_type_store, ['spmatrix', '_ufuncs_with_fixed_point_at_zero'], [spmatrix, _ufuncs_with_fixed_point_at_zero])

else:
    # Assigning a type to the variable 'scipy.sparse.base' (line 13)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 0), 'scipy.sparse.base', import_372052)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/sparse/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 14, 0))

# 'from scipy.sparse.sputils import isscalarlike, validateaxis' statement (line 14)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/sparse/')
import_372054 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 14, 0), 'scipy.sparse.sputils')

if (type(import_372054) is not StypyTypeError):

    if (import_372054 != 'pyd_module'):
        __import__(import_372054)
        sys_modules_372055 = sys.modules[import_372054]
        import_from_module(stypy.reporting.localization.Localization(__file__, 14, 0), 'scipy.sparse.sputils', sys_modules_372055.module_type_store, module_type_store, ['isscalarlike', 'validateaxis'])
        nest_module(stypy.reporting.localization.Localization(__file__, 14, 0), __file__, sys_modules_372055, sys_modules_372055.module_type_store, module_type_store)
    else:
        from scipy.sparse.sputils import isscalarlike, validateaxis

        import_from_module(stypy.reporting.localization.Localization(__file__, 14, 0), 'scipy.sparse.sputils', None, module_type_store, ['isscalarlike', 'validateaxis'], [isscalarlike, validateaxis])

else:
    # Assigning a type to the variable 'scipy.sparse.sputils' (line 14)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 0), 'scipy.sparse.sputils', import_372054)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/sparse/')


# Assigning a List to a Name (line 16):

# Assigning a List to a Name (line 16):
__all__ = []
module_type_store.set_exportable_members([])

# Obtaining an instance of the builtin type 'list' (line 16)
list_372056 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 10), 'list')
# Adding type elements to the builtin type 'list' instance (line 16)

# Assigning a type to the variable '__all__' (line 16)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 0), '__all__', list_372056)
# Declaration of the '_data_matrix' class
# Getting the type of 'spmatrix' (line 21)
spmatrix_372057 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 19), 'spmatrix')

class _data_matrix(spmatrix_372057, ):

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 22, 4, False)
        # Assigning a type to the variable 'self' (line 23)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 23, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, '_data_matrix.__init__', [], None, None, defaults, varargs, kwargs)

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

        
        # Call to __init__(...): (line 23)
        # Processing the call arguments (line 23)
        # Getting the type of 'self' (line 23)
        self_372060 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 26), 'self', False)
        # Processing the call keyword arguments (line 23)
        kwargs_372061 = {}
        # Getting the type of 'spmatrix' (line 23)
        spmatrix_372058 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 8), 'spmatrix', False)
        # Obtaining the member '__init__' of a type (line 23)
        init___372059 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 23, 8), spmatrix_372058, '__init__')
        # Calling __init__(args, kwargs) (line 23)
        init___call_result_372062 = invoke(stypy.reporting.localization.Localization(__file__, 23, 8), init___372059, *[self_372060], **kwargs_372061)
        
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def _get_dtype(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_get_dtype'
        module_type_store = module_type_store.open_function_context('_get_dtype', 25, 4, False)
        # Assigning a type to the variable 'self' (line 26)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 26, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        _data_matrix._get_dtype.__dict__.__setitem__('stypy_localization', localization)
        _data_matrix._get_dtype.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        _data_matrix._get_dtype.__dict__.__setitem__('stypy_type_store', module_type_store)
        _data_matrix._get_dtype.__dict__.__setitem__('stypy_function_name', '_data_matrix._get_dtype')
        _data_matrix._get_dtype.__dict__.__setitem__('stypy_param_names_list', [])
        _data_matrix._get_dtype.__dict__.__setitem__('stypy_varargs_param_name', None)
        _data_matrix._get_dtype.__dict__.__setitem__('stypy_kwargs_param_name', None)
        _data_matrix._get_dtype.__dict__.__setitem__('stypy_call_defaults', defaults)
        _data_matrix._get_dtype.__dict__.__setitem__('stypy_call_varargs', varargs)
        _data_matrix._get_dtype.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        _data_matrix._get_dtype.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, '_data_matrix._get_dtype', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_get_dtype', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_get_dtype(...)' code ##################

        # Getting the type of 'self' (line 26)
        self_372063 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 15), 'self')
        # Obtaining the member 'data' of a type (line 26)
        data_372064 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 26, 15), self_372063, 'data')
        # Obtaining the member 'dtype' of a type (line 26)
        dtype_372065 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 26, 15), data_372064, 'dtype')
        # Assigning a type to the variable 'stypy_return_type' (line 26)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 26, 8), 'stypy_return_type', dtype_372065)
        
        # ################# End of '_get_dtype(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_get_dtype' in the type store
        # Getting the type of 'stypy_return_type' (line 25)
        stypy_return_type_372066 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_372066)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_get_dtype'
        return stypy_return_type_372066


    @norecursion
    def _set_dtype(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_set_dtype'
        module_type_store = module_type_store.open_function_context('_set_dtype', 28, 4, False)
        # Assigning a type to the variable 'self' (line 29)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 29, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        _data_matrix._set_dtype.__dict__.__setitem__('stypy_localization', localization)
        _data_matrix._set_dtype.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        _data_matrix._set_dtype.__dict__.__setitem__('stypy_type_store', module_type_store)
        _data_matrix._set_dtype.__dict__.__setitem__('stypy_function_name', '_data_matrix._set_dtype')
        _data_matrix._set_dtype.__dict__.__setitem__('stypy_param_names_list', ['newtype'])
        _data_matrix._set_dtype.__dict__.__setitem__('stypy_varargs_param_name', None)
        _data_matrix._set_dtype.__dict__.__setitem__('stypy_kwargs_param_name', None)
        _data_matrix._set_dtype.__dict__.__setitem__('stypy_call_defaults', defaults)
        _data_matrix._set_dtype.__dict__.__setitem__('stypy_call_varargs', varargs)
        _data_matrix._set_dtype.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        _data_matrix._set_dtype.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, '_data_matrix._set_dtype', ['newtype'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_set_dtype', localization, ['newtype'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_set_dtype(...)' code ##################

        
        # Assigning a Name to a Attribute (line 29):
        
        # Assigning a Name to a Attribute (line 29):
        # Getting the type of 'newtype' (line 29)
        newtype_372067 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 26), 'newtype')
        # Getting the type of 'self' (line 29)
        self_372068 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 8), 'self')
        # Obtaining the member 'data' of a type (line 29)
        data_372069 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 29, 8), self_372068, 'data')
        # Setting the type of the member 'dtype' of a type (line 29)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 29, 8), data_372069, 'dtype', newtype_372067)
        
        # ################# End of '_set_dtype(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_set_dtype' in the type store
        # Getting the type of 'stypy_return_type' (line 28)
        stypy_return_type_372070 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_372070)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_set_dtype'
        return stypy_return_type_372070

    
    # Assigning a Call to a Name (line 30):

    @norecursion
    def _deduped_data(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_deduped_data'
        module_type_store = module_type_store.open_function_context('_deduped_data', 32, 4, False)
        # Assigning a type to the variable 'self' (line 33)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 33, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        _data_matrix._deduped_data.__dict__.__setitem__('stypy_localization', localization)
        _data_matrix._deduped_data.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        _data_matrix._deduped_data.__dict__.__setitem__('stypy_type_store', module_type_store)
        _data_matrix._deduped_data.__dict__.__setitem__('stypy_function_name', '_data_matrix._deduped_data')
        _data_matrix._deduped_data.__dict__.__setitem__('stypy_param_names_list', [])
        _data_matrix._deduped_data.__dict__.__setitem__('stypy_varargs_param_name', None)
        _data_matrix._deduped_data.__dict__.__setitem__('stypy_kwargs_param_name', None)
        _data_matrix._deduped_data.__dict__.__setitem__('stypy_call_defaults', defaults)
        _data_matrix._deduped_data.__dict__.__setitem__('stypy_call_varargs', varargs)
        _data_matrix._deduped_data.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        _data_matrix._deduped_data.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, '_data_matrix._deduped_data', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_deduped_data', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_deduped_data(...)' code ##################

        
        # Type idiom detected: calculating its left and rigth part (line 33)
        str_372071 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 25), 'str', 'sum_duplicates')
        # Getting the type of 'self' (line 33)
        self_372072 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 19), 'self')
        
        (may_be_372073, more_types_in_union_372074) = may_provide_member(str_372071, self_372072)

        if may_be_372073:

            if more_types_in_union_372074:
                # Runtime conditional SSA (line 33)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            # Assigning a type to the variable 'self' (line 33)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 33, 8), 'self', remove_not_member_provider_from_union(self_372072, 'sum_duplicates'))
            
            # Call to sum_duplicates(...): (line 34)
            # Processing the call keyword arguments (line 34)
            kwargs_372077 = {}
            # Getting the type of 'self' (line 34)
            self_372075 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 12), 'self', False)
            # Obtaining the member 'sum_duplicates' of a type (line 34)
            sum_duplicates_372076 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 34, 12), self_372075, 'sum_duplicates')
            # Calling sum_duplicates(args, kwargs) (line 34)
            sum_duplicates_call_result_372078 = invoke(stypy.reporting.localization.Localization(__file__, 34, 12), sum_duplicates_372076, *[], **kwargs_372077)
            

            if more_types_in_union_372074:
                # SSA join for if statement (line 33)
                module_type_store = module_type_store.join_ssa_context()


        
        # Getting the type of 'self' (line 35)
        self_372079 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 15), 'self')
        # Obtaining the member 'data' of a type (line 35)
        data_372080 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 35, 15), self_372079, 'data')
        # Assigning a type to the variable 'stypy_return_type' (line 35)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 35, 8), 'stypy_return_type', data_372080)
        
        # ################# End of '_deduped_data(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_deduped_data' in the type store
        # Getting the type of 'stypy_return_type' (line 32)
        stypy_return_type_372081 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_372081)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_deduped_data'
        return stypy_return_type_372081


    @norecursion
    def __abs__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__abs__'
        module_type_store = module_type_store.open_function_context('__abs__', 37, 4, False)
        # Assigning a type to the variable 'self' (line 38)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 38, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        _data_matrix.__abs__.__dict__.__setitem__('stypy_localization', localization)
        _data_matrix.__abs__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        _data_matrix.__abs__.__dict__.__setitem__('stypy_type_store', module_type_store)
        _data_matrix.__abs__.__dict__.__setitem__('stypy_function_name', '_data_matrix.__abs__')
        _data_matrix.__abs__.__dict__.__setitem__('stypy_param_names_list', [])
        _data_matrix.__abs__.__dict__.__setitem__('stypy_varargs_param_name', None)
        _data_matrix.__abs__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        _data_matrix.__abs__.__dict__.__setitem__('stypy_call_defaults', defaults)
        _data_matrix.__abs__.__dict__.__setitem__('stypy_call_varargs', varargs)
        _data_matrix.__abs__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        _data_matrix.__abs__.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, '_data_matrix.__abs__', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__abs__', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__abs__(...)' code ##################

        
        # Call to _with_data(...): (line 38)
        # Processing the call arguments (line 38)
        
        # Call to abs(...): (line 38)
        # Processing the call arguments (line 38)
        
        # Call to _deduped_data(...): (line 38)
        # Processing the call keyword arguments (line 38)
        kwargs_372087 = {}
        # Getting the type of 'self' (line 38)
        self_372085 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 35), 'self', False)
        # Obtaining the member '_deduped_data' of a type (line 38)
        _deduped_data_372086 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 38, 35), self_372085, '_deduped_data')
        # Calling _deduped_data(args, kwargs) (line 38)
        _deduped_data_call_result_372088 = invoke(stypy.reporting.localization.Localization(__file__, 38, 35), _deduped_data_372086, *[], **kwargs_372087)
        
        # Processing the call keyword arguments (line 38)
        kwargs_372089 = {}
        # Getting the type of 'abs' (line 38)
        abs_372084 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 31), 'abs', False)
        # Calling abs(args, kwargs) (line 38)
        abs_call_result_372090 = invoke(stypy.reporting.localization.Localization(__file__, 38, 31), abs_372084, *[_deduped_data_call_result_372088], **kwargs_372089)
        
        # Processing the call keyword arguments (line 38)
        kwargs_372091 = {}
        # Getting the type of 'self' (line 38)
        self_372082 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 15), 'self', False)
        # Obtaining the member '_with_data' of a type (line 38)
        _with_data_372083 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 38, 15), self_372082, '_with_data')
        # Calling _with_data(args, kwargs) (line 38)
        _with_data_call_result_372092 = invoke(stypy.reporting.localization.Localization(__file__, 38, 15), _with_data_372083, *[abs_call_result_372090], **kwargs_372091)
        
        # Assigning a type to the variable 'stypy_return_type' (line 38)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 38, 8), 'stypy_return_type', _with_data_call_result_372092)
        
        # ################# End of '__abs__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__abs__' in the type store
        # Getting the type of 'stypy_return_type' (line 37)
        stypy_return_type_372093 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_372093)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__abs__'
        return stypy_return_type_372093


    @norecursion
    def _real(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_real'
        module_type_store = module_type_store.open_function_context('_real', 40, 4, False)
        # Assigning a type to the variable 'self' (line 41)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 41, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        _data_matrix._real.__dict__.__setitem__('stypy_localization', localization)
        _data_matrix._real.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        _data_matrix._real.__dict__.__setitem__('stypy_type_store', module_type_store)
        _data_matrix._real.__dict__.__setitem__('stypy_function_name', '_data_matrix._real')
        _data_matrix._real.__dict__.__setitem__('stypy_param_names_list', [])
        _data_matrix._real.__dict__.__setitem__('stypy_varargs_param_name', None)
        _data_matrix._real.__dict__.__setitem__('stypy_kwargs_param_name', None)
        _data_matrix._real.__dict__.__setitem__('stypy_call_defaults', defaults)
        _data_matrix._real.__dict__.__setitem__('stypy_call_varargs', varargs)
        _data_matrix._real.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        _data_matrix._real.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, '_data_matrix._real', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_real', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_real(...)' code ##################

        
        # Call to _with_data(...): (line 41)
        # Processing the call arguments (line 41)
        # Getting the type of 'self' (line 41)
        self_372096 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 31), 'self', False)
        # Obtaining the member 'data' of a type (line 41)
        data_372097 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 41, 31), self_372096, 'data')
        # Obtaining the member 'real' of a type (line 41)
        real_372098 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 41, 31), data_372097, 'real')
        # Processing the call keyword arguments (line 41)
        kwargs_372099 = {}
        # Getting the type of 'self' (line 41)
        self_372094 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 15), 'self', False)
        # Obtaining the member '_with_data' of a type (line 41)
        _with_data_372095 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 41, 15), self_372094, '_with_data')
        # Calling _with_data(args, kwargs) (line 41)
        _with_data_call_result_372100 = invoke(stypy.reporting.localization.Localization(__file__, 41, 15), _with_data_372095, *[real_372098], **kwargs_372099)
        
        # Assigning a type to the variable 'stypy_return_type' (line 41)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 41, 8), 'stypy_return_type', _with_data_call_result_372100)
        
        # ################# End of '_real(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_real' in the type store
        # Getting the type of 'stypy_return_type' (line 40)
        stypy_return_type_372101 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_372101)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_real'
        return stypy_return_type_372101


    @norecursion
    def _imag(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_imag'
        module_type_store = module_type_store.open_function_context('_imag', 43, 4, False)
        # Assigning a type to the variable 'self' (line 44)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 44, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        _data_matrix._imag.__dict__.__setitem__('stypy_localization', localization)
        _data_matrix._imag.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        _data_matrix._imag.__dict__.__setitem__('stypy_type_store', module_type_store)
        _data_matrix._imag.__dict__.__setitem__('stypy_function_name', '_data_matrix._imag')
        _data_matrix._imag.__dict__.__setitem__('stypy_param_names_list', [])
        _data_matrix._imag.__dict__.__setitem__('stypy_varargs_param_name', None)
        _data_matrix._imag.__dict__.__setitem__('stypy_kwargs_param_name', None)
        _data_matrix._imag.__dict__.__setitem__('stypy_call_defaults', defaults)
        _data_matrix._imag.__dict__.__setitem__('stypy_call_varargs', varargs)
        _data_matrix._imag.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        _data_matrix._imag.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, '_data_matrix._imag', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_imag', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_imag(...)' code ##################

        
        # Call to _with_data(...): (line 44)
        # Processing the call arguments (line 44)
        # Getting the type of 'self' (line 44)
        self_372104 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 31), 'self', False)
        # Obtaining the member 'data' of a type (line 44)
        data_372105 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 44, 31), self_372104, 'data')
        # Obtaining the member 'imag' of a type (line 44)
        imag_372106 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 44, 31), data_372105, 'imag')
        # Processing the call keyword arguments (line 44)
        kwargs_372107 = {}
        # Getting the type of 'self' (line 44)
        self_372102 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 15), 'self', False)
        # Obtaining the member '_with_data' of a type (line 44)
        _with_data_372103 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 44, 15), self_372102, '_with_data')
        # Calling _with_data(args, kwargs) (line 44)
        _with_data_call_result_372108 = invoke(stypy.reporting.localization.Localization(__file__, 44, 15), _with_data_372103, *[imag_372106], **kwargs_372107)
        
        # Assigning a type to the variable 'stypy_return_type' (line 44)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 44, 8), 'stypy_return_type', _with_data_call_result_372108)
        
        # ################# End of '_imag(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_imag' in the type store
        # Getting the type of 'stypy_return_type' (line 43)
        stypy_return_type_372109 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_372109)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_imag'
        return stypy_return_type_372109


    @norecursion
    def __neg__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__neg__'
        module_type_store = module_type_store.open_function_context('__neg__', 46, 4, False)
        # Assigning a type to the variable 'self' (line 47)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 47, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        _data_matrix.__neg__.__dict__.__setitem__('stypy_localization', localization)
        _data_matrix.__neg__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        _data_matrix.__neg__.__dict__.__setitem__('stypy_type_store', module_type_store)
        _data_matrix.__neg__.__dict__.__setitem__('stypy_function_name', '_data_matrix.__neg__')
        _data_matrix.__neg__.__dict__.__setitem__('stypy_param_names_list', [])
        _data_matrix.__neg__.__dict__.__setitem__('stypy_varargs_param_name', None)
        _data_matrix.__neg__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        _data_matrix.__neg__.__dict__.__setitem__('stypy_call_defaults', defaults)
        _data_matrix.__neg__.__dict__.__setitem__('stypy_call_varargs', varargs)
        _data_matrix.__neg__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        _data_matrix.__neg__.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, '_data_matrix.__neg__', [], None, None, defaults, varargs, kwargs)

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

        
        
        # Getting the type of 'self' (line 47)
        self_372110 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 11), 'self')
        # Obtaining the member 'dtype' of a type (line 47)
        dtype_372111 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 47, 11), self_372110, 'dtype')
        # Obtaining the member 'kind' of a type (line 47)
        kind_372112 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 47, 11), dtype_372111, 'kind')
        str_372113 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 30), 'str', 'b')
        # Applying the binary operator '==' (line 47)
        result_eq_372114 = python_operator(stypy.reporting.localization.Localization(__file__, 47, 11), '==', kind_372112, str_372113)
        
        # Testing the type of an if condition (line 47)
        if_condition_372115 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 47, 8), result_eq_372114)
        # Assigning a type to the variable 'if_condition_372115' (line 47)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 47, 8), 'if_condition_372115', if_condition_372115)
        # SSA begins for if statement (line 47)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to NotImplementedError(...): (line 48)
        # Processing the call arguments (line 48)
        str_372117 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 48, 38), 'str', 'negating a sparse boolean matrix is not supported')
        # Processing the call keyword arguments (line 48)
        kwargs_372118 = {}
        # Getting the type of 'NotImplementedError' (line 48)
        NotImplementedError_372116 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 18), 'NotImplementedError', False)
        # Calling NotImplementedError(args, kwargs) (line 48)
        NotImplementedError_call_result_372119 = invoke(stypy.reporting.localization.Localization(__file__, 48, 18), NotImplementedError_372116, *[str_372117], **kwargs_372118)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 48, 12), NotImplementedError_call_result_372119, 'raise parameter', BaseException)
        # SSA join for if statement (line 47)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to _with_data(...): (line 50)
        # Processing the call arguments (line 50)
        
        # Getting the type of 'self' (line 50)
        self_372122 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 32), 'self', False)
        # Obtaining the member 'data' of a type (line 50)
        data_372123 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 50, 32), self_372122, 'data')
        # Applying the 'usub' unary operator (line 50)
        result___neg___372124 = python_operator(stypy.reporting.localization.Localization(__file__, 50, 31), 'usub', data_372123)
        
        # Processing the call keyword arguments (line 50)
        kwargs_372125 = {}
        # Getting the type of 'self' (line 50)
        self_372120 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 15), 'self', False)
        # Obtaining the member '_with_data' of a type (line 50)
        _with_data_372121 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 50, 15), self_372120, '_with_data')
        # Calling _with_data(args, kwargs) (line 50)
        _with_data_call_result_372126 = invoke(stypy.reporting.localization.Localization(__file__, 50, 15), _with_data_372121, *[result___neg___372124], **kwargs_372125)
        
        # Assigning a type to the variable 'stypy_return_type' (line 50)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 50, 8), 'stypy_return_type', _with_data_call_result_372126)
        
        # ################# End of '__neg__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__neg__' in the type store
        # Getting the type of 'stypy_return_type' (line 46)
        stypy_return_type_372127 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_372127)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__neg__'
        return stypy_return_type_372127


    @norecursion
    def __imul__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__imul__'
        module_type_store = module_type_store.open_function_context('__imul__', 52, 4, False)
        # Assigning a type to the variable 'self' (line 53)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 53, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        _data_matrix.__imul__.__dict__.__setitem__('stypy_localization', localization)
        _data_matrix.__imul__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        _data_matrix.__imul__.__dict__.__setitem__('stypy_type_store', module_type_store)
        _data_matrix.__imul__.__dict__.__setitem__('stypy_function_name', '_data_matrix.__imul__')
        _data_matrix.__imul__.__dict__.__setitem__('stypy_param_names_list', ['other'])
        _data_matrix.__imul__.__dict__.__setitem__('stypy_varargs_param_name', None)
        _data_matrix.__imul__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        _data_matrix.__imul__.__dict__.__setitem__('stypy_call_defaults', defaults)
        _data_matrix.__imul__.__dict__.__setitem__('stypy_call_varargs', varargs)
        _data_matrix.__imul__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        _data_matrix.__imul__.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, '_data_matrix.__imul__', ['other'], None, None, defaults, varargs, kwargs)

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

        
        
        # Call to isscalarlike(...): (line 53)
        # Processing the call arguments (line 53)
        # Getting the type of 'other' (line 53)
        other_372129 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 24), 'other', False)
        # Processing the call keyword arguments (line 53)
        kwargs_372130 = {}
        # Getting the type of 'isscalarlike' (line 53)
        isscalarlike_372128 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 11), 'isscalarlike', False)
        # Calling isscalarlike(args, kwargs) (line 53)
        isscalarlike_call_result_372131 = invoke(stypy.reporting.localization.Localization(__file__, 53, 11), isscalarlike_372128, *[other_372129], **kwargs_372130)
        
        # Testing the type of an if condition (line 53)
        if_condition_372132 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 53, 8), isscalarlike_call_result_372131)
        # Assigning a type to the variable 'if_condition_372132' (line 53)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 53, 8), 'if_condition_372132', if_condition_372132)
        # SSA begins for if statement (line 53)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Getting the type of 'self' (line 54)
        self_372133 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 12), 'self')
        # Obtaining the member 'data' of a type (line 54)
        data_372134 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 54, 12), self_372133, 'data')
        # Getting the type of 'other' (line 54)
        other_372135 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 25), 'other')
        # Applying the binary operator '*=' (line 54)
        result_imul_372136 = python_operator(stypy.reporting.localization.Localization(__file__, 54, 12), '*=', data_372134, other_372135)
        # Getting the type of 'self' (line 54)
        self_372137 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 12), 'self')
        # Setting the type of the member 'data' of a type (line 54)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 54, 12), self_372137, 'data', result_imul_372136)
        
        # Getting the type of 'self' (line 55)
        self_372138 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 19), 'self')
        # Assigning a type to the variable 'stypy_return_type' (line 55)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 55, 12), 'stypy_return_type', self_372138)
        # SSA branch for the else part of an if statement (line 53)
        module_type_store.open_ssa_branch('else')
        # Getting the type of 'NotImplemented' (line 57)
        NotImplemented_372139 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 19), 'NotImplemented')
        # Assigning a type to the variable 'stypy_return_type' (line 57)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 57, 12), 'stypy_return_type', NotImplemented_372139)
        # SSA join for if statement (line 53)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of '__imul__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__imul__' in the type store
        # Getting the type of 'stypy_return_type' (line 52)
        stypy_return_type_372140 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_372140)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__imul__'
        return stypy_return_type_372140


    @norecursion
    def __itruediv__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__itruediv__'
        module_type_store = module_type_store.open_function_context('__itruediv__', 59, 4, False)
        # Assigning a type to the variable 'self' (line 60)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 60, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        _data_matrix.__itruediv__.__dict__.__setitem__('stypy_localization', localization)
        _data_matrix.__itruediv__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        _data_matrix.__itruediv__.__dict__.__setitem__('stypy_type_store', module_type_store)
        _data_matrix.__itruediv__.__dict__.__setitem__('stypy_function_name', '_data_matrix.__itruediv__')
        _data_matrix.__itruediv__.__dict__.__setitem__('stypy_param_names_list', ['other'])
        _data_matrix.__itruediv__.__dict__.__setitem__('stypy_varargs_param_name', None)
        _data_matrix.__itruediv__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        _data_matrix.__itruediv__.__dict__.__setitem__('stypy_call_defaults', defaults)
        _data_matrix.__itruediv__.__dict__.__setitem__('stypy_call_varargs', varargs)
        _data_matrix.__itruediv__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        _data_matrix.__itruediv__.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, '_data_matrix.__itruediv__', ['other'], None, None, defaults, varargs, kwargs)

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

        
        
        # Call to isscalarlike(...): (line 60)
        # Processing the call arguments (line 60)
        # Getting the type of 'other' (line 60)
        other_372142 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 24), 'other', False)
        # Processing the call keyword arguments (line 60)
        kwargs_372143 = {}
        # Getting the type of 'isscalarlike' (line 60)
        isscalarlike_372141 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 11), 'isscalarlike', False)
        # Calling isscalarlike(args, kwargs) (line 60)
        isscalarlike_call_result_372144 = invoke(stypy.reporting.localization.Localization(__file__, 60, 11), isscalarlike_372141, *[other_372142], **kwargs_372143)
        
        # Testing the type of an if condition (line 60)
        if_condition_372145 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 60, 8), isscalarlike_call_result_372144)
        # Assigning a type to the variable 'if_condition_372145' (line 60)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 60, 8), 'if_condition_372145', if_condition_372145)
        # SSA begins for if statement (line 60)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a BinOp to a Name (line 61):
        
        # Assigning a BinOp to a Name (line 61):
        float_372146 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 61, 20), 'float')
        # Getting the type of 'other' (line 61)
        other_372147 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 26), 'other')
        # Applying the binary operator 'div' (line 61)
        result_div_372148 = python_operator(stypy.reporting.localization.Localization(__file__, 61, 20), 'div', float_372146, other_372147)
        
        # Assigning a type to the variable 'recip' (line 61)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 61, 12), 'recip', result_div_372148)
        
        # Getting the type of 'self' (line 62)
        self_372149 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 12), 'self')
        # Obtaining the member 'data' of a type (line 62)
        data_372150 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 62, 12), self_372149, 'data')
        # Getting the type of 'recip' (line 62)
        recip_372151 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 25), 'recip')
        # Applying the binary operator '*=' (line 62)
        result_imul_372152 = python_operator(stypy.reporting.localization.Localization(__file__, 62, 12), '*=', data_372150, recip_372151)
        # Getting the type of 'self' (line 62)
        self_372153 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 12), 'self')
        # Setting the type of the member 'data' of a type (line 62)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 62, 12), self_372153, 'data', result_imul_372152)
        
        # Getting the type of 'self' (line 63)
        self_372154 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 19), 'self')
        # Assigning a type to the variable 'stypy_return_type' (line 63)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 63, 12), 'stypy_return_type', self_372154)
        # SSA branch for the else part of an if statement (line 60)
        module_type_store.open_ssa_branch('else')
        # Getting the type of 'NotImplemented' (line 65)
        NotImplemented_372155 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 19), 'NotImplemented')
        # Assigning a type to the variable 'stypy_return_type' (line 65)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 65, 12), 'stypy_return_type', NotImplemented_372155)
        # SSA join for if statement (line 60)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of '__itruediv__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__itruediv__' in the type store
        # Getting the type of 'stypy_return_type' (line 59)
        stypy_return_type_372156 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_372156)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__itruediv__'
        return stypy_return_type_372156


    @norecursion
    def astype(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        str_372157 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 67, 36), 'str', 'unsafe')
        # Getting the type of 'True' (line 67)
        True_372158 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 51), 'True')
        defaults = [str_372157, True_372158]
        # Create a new context for function 'astype'
        module_type_store = module_type_store.open_function_context('astype', 67, 4, False)
        # Assigning a type to the variable 'self' (line 68)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 68, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        _data_matrix.astype.__dict__.__setitem__('stypy_localization', localization)
        _data_matrix.astype.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        _data_matrix.astype.__dict__.__setitem__('stypy_type_store', module_type_store)
        _data_matrix.astype.__dict__.__setitem__('stypy_function_name', '_data_matrix.astype')
        _data_matrix.astype.__dict__.__setitem__('stypy_param_names_list', ['dtype', 'casting', 'copy'])
        _data_matrix.astype.__dict__.__setitem__('stypy_varargs_param_name', None)
        _data_matrix.astype.__dict__.__setitem__('stypy_kwargs_param_name', None)
        _data_matrix.astype.__dict__.__setitem__('stypy_call_defaults', defaults)
        _data_matrix.astype.__dict__.__setitem__('stypy_call_varargs', varargs)
        _data_matrix.astype.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        _data_matrix.astype.__dict__.__setitem__('stypy_declared_arg_number', 4)
        arguments = process_argument_values(localization, type_of_self, module_type_store, '_data_matrix.astype', ['dtype', 'casting', 'copy'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'astype', localization, ['dtype', 'casting', 'copy'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'astype(...)' code ##################

        
        # Assigning a Call to a Name (line 68):
        
        # Assigning a Call to a Name (line 68):
        
        # Call to dtype(...): (line 68)
        # Processing the call arguments (line 68)
        # Getting the type of 'dtype' (line 68)
        dtype_372161 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 25), 'dtype', False)
        # Processing the call keyword arguments (line 68)
        kwargs_372162 = {}
        # Getting the type of 'np' (line 68)
        np_372159 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 16), 'np', False)
        # Obtaining the member 'dtype' of a type (line 68)
        dtype_372160 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 68, 16), np_372159, 'dtype')
        # Calling dtype(args, kwargs) (line 68)
        dtype_call_result_372163 = invoke(stypy.reporting.localization.Localization(__file__, 68, 16), dtype_372160, *[dtype_372161], **kwargs_372162)
        
        # Assigning a type to the variable 'dtype' (line 68)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 68, 8), 'dtype', dtype_call_result_372163)
        
        
        # Getting the type of 'self' (line 69)
        self_372164 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 11), 'self')
        # Obtaining the member 'dtype' of a type (line 69)
        dtype_372165 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 69, 11), self_372164, 'dtype')
        # Getting the type of 'dtype' (line 69)
        dtype_372166 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 25), 'dtype')
        # Applying the binary operator '!=' (line 69)
        result_ne_372167 = python_operator(stypy.reporting.localization.Localization(__file__, 69, 11), '!=', dtype_372165, dtype_372166)
        
        # Testing the type of an if condition (line 69)
        if_condition_372168 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 69, 8), result_ne_372167)
        # Assigning a type to the variable 'if_condition_372168' (line 69)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 69, 8), 'if_condition_372168', if_condition_372168)
        # SSA begins for if statement (line 69)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to _with_data(...): (line 70)
        # Processing the call arguments (line 70)
        
        # Call to astype(...): (line 71)
        # Processing the call arguments (line 71)
        # Getting the type of 'dtype' (line 71)
        dtype_372176 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 44), 'dtype', False)
        # Processing the call keyword arguments (line 71)
        # Getting the type of 'casting' (line 71)
        casting_372177 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 59), 'casting', False)
        keyword_372178 = casting_372177
        # Getting the type of 'copy' (line 71)
        copy_372179 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 73), 'copy', False)
        keyword_372180 = copy_372179
        kwargs_372181 = {'copy': keyword_372180, 'casting': keyword_372178}
        
        # Call to _deduped_data(...): (line 71)
        # Processing the call keyword arguments (line 71)
        kwargs_372173 = {}
        # Getting the type of 'self' (line 71)
        self_372171 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 16), 'self', False)
        # Obtaining the member '_deduped_data' of a type (line 71)
        _deduped_data_372172 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 71, 16), self_372171, '_deduped_data')
        # Calling _deduped_data(args, kwargs) (line 71)
        _deduped_data_call_result_372174 = invoke(stypy.reporting.localization.Localization(__file__, 71, 16), _deduped_data_372172, *[], **kwargs_372173)
        
        # Obtaining the member 'astype' of a type (line 71)
        astype_372175 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 71, 16), _deduped_data_call_result_372174, 'astype')
        # Calling astype(args, kwargs) (line 71)
        astype_call_result_372182 = invoke(stypy.reporting.localization.Localization(__file__, 71, 16), astype_372175, *[dtype_372176], **kwargs_372181)
        
        # Processing the call keyword arguments (line 70)
        # Getting the type of 'copy' (line 72)
        copy_372183 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 21), 'copy', False)
        keyword_372184 = copy_372183
        kwargs_372185 = {'copy': keyword_372184}
        # Getting the type of 'self' (line 70)
        self_372169 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 19), 'self', False)
        # Obtaining the member '_with_data' of a type (line 70)
        _with_data_372170 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 70, 19), self_372169, '_with_data')
        # Calling _with_data(args, kwargs) (line 70)
        _with_data_call_result_372186 = invoke(stypy.reporting.localization.Localization(__file__, 70, 19), _with_data_372170, *[astype_call_result_372182], **kwargs_372185)
        
        # Assigning a type to the variable 'stypy_return_type' (line 70)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 70, 12), 'stypy_return_type', _with_data_call_result_372186)
        # SSA branch for the else part of an if statement (line 69)
        module_type_store.open_ssa_branch('else')
        
        # Getting the type of 'copy' (line 73)
        copy_372187 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 13), 'copy')
        # Testing the type of an if condition (line 73)
        if_condition_372188 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 73, 13), copy_372187)
        # Assigning a type to the variable 'if_condition_372188' (line 73)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 73, 13), 'if_condition_372188', if_condition_372188)
        # SSA begins for if statement (line 73)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to copy(...): (line 74)
        # Processing the call keyword arguments (line 74)
        kwargs_372191 = {}
        # Getting the type of 'self' (line 74)
        self_372189 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 19), 'self', False)
        # Obtaining the member 'copy' of a type (line 74)
        copy_372190 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 74, 19), self_372189, 'copy')
        # Calling copy(args, kwargs) (line 74)
        copy_call_result_372192 = invoke(stypy.reporting.localization.Localization(__file__, 74, 19), copy_372190, *[], **kwargs_372191)
        
        # Assigning a type to the variable 'stypy_return_type' (line 74)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 74, 12), 'stypy_return_type', copy_call_result_372192)
        # SSA branch for the else part of an if statement (line 73)
        module_type_store.open_ssa_branch('else')
        # Getting the type of 'self' (line 76)
        self_372193 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 19), 'self')
        # Assigning a type to the variable 'stypy_return_type' (line 76)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 76, 12), 'stypy_return_type', self_372193)
        # SSA join for if statement (line 73)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 69)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'astype(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'astype' in the type store
        # Getting the type of 'stypy_return_type' (line 67)
        stypy_return_type_372194 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_372194)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'astype'
        return stypy_return_type_372194

    
    # Assigning a Attribute to a Attribute (line 78):

    @norecursion
    def conj(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'conj'
        module_type_store = module_type_store.open_function_context('conj', 80, 4, False)
        # Assigning a type to the variable 'self' (line 81)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 81, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        _data_matrix.conj.__dict__.__setitem__('stypy_localization', localization)
        _data_matrix.conj.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        _data_matrix.conj.__dict__.__setitem__('stypy_type_store', module_type_store)
        _data_matrix.conj.__dict__.__setitem__('stypy_function_name', '_data_matrix.conj')
        _data_matrix.conj.__dict__.__setitem__('stypy_param_names_list', [])
        _data_matrix.conj.__dict__.__setitem__('stypy_varargs_param_name', None)
        _data_matrix.conj.__dict__.__setitem__('stypy_kwargs_param_name', None)
        _data_matrix.conj.__dict__.__setitem__('stypy_call_defaults', defaults)
        _data_matrix.conj.__dict__.__setitem__('stypy_call_varargs', varargs)
        _data_matrix.conj.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        _data_matrix.conj.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, '_data_matrix.conj', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'conj', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'conj(...)' code ##################

        
        # Call to _with_data(...): (line 81)
        # Processing the call arguments (line 81)
        
        # Call to conj(...): (line 81)
        # Processing the call keyword arguments (line 81)
        kwargs_372200 = {}
        # Getting the type of 'self' (line 81)
        self_372197 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 31), 'self', False)
        # Obtaining the member 'data' of a type (line 81)
        data_372198 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 81, 31), self_372197, 'data')
        # Obtaining the member 'conj' of a type (line 81)
        conj_372199 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 81, 31), data_372198, 'conj')
        # Calling conj(args, kwargs) (line 81)
        conj_call_result_372201 = invoke(stypy.reporting.localization.Localization(__file__, 81, 31), conj_372199, *[], **kwargs_372200)
        
        # Processing the call keyword arguments (line 81)
        kwargs_372202 = {}
        # Getting the type of 'self' (line 81)
        self_372195 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 15), 'self', False)
        # Obtaining the member '_with_data' of a type (line 81)
        _with_data_372196 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 81, 15), self_372195, '_with_data')
        # Calling _with_data(args, kwargs) (line 81)
        _with_data_call_result_372203 = invoke(stypy.reporting.localization.Localization(__file__, 81, 15), _with_data_372196, *[conj_call_result_372201], **kwargs_372202)
        
        # Assigning a type to the variable 'stypy_return_type' (line 81)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 81, 8), 'stypy_return_type', _with_data_call_result_372203)
        
        # ################# End of 'conj(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'conj' in the type store
        # Getting the type of 'stypy_return_type' (line 80)
        stypy_return_type_372204 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_372204)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'conj'
        return stypy_return_type_372204

    
    # Assigning a Attribute to a Attribute (line 83):

    @norecursion
    def copy(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'copy'
        module_type_store = module_type_store.open_function_context('copy', 85, 4, False)
        # Assigning a type to the variable 'self' (line 86)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 86, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        _data_matrix.copy.__dict__.__setitem__('stypy_localization', localization)
        _data_matrix.copy.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        _data_matrix.copy.__dict__.__setitem__('stypy_type_store', module_type_store)
        _data_matrix.copy.__dict__.__setitem__('stypy_function_name', '_data_matrix.copy')
        _data_matrix.copy.__dict__.__setitem__('stypy_param_names_list', [])
        _data_matrix.copy.__dict__.__setitem__('stypy_varargs_param_name', None)
        _data_matrix.copy.__dict__.__setitem__('stypy_kwargs_param_name', None)
        _data_matrix.copy.__dict__.__setitem__('stypy_call_defaults', defaults)
        _data_matrix.copy.__dict__.__setitem__('stypy_call_varargs', varargs)
        _data_matrix.copy.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        _data_matrix.copy.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, '_data_matrix.copy', [], None, None, defaults, varargs, kwargs)

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

        
        # Call to _with_data(...): (line 86)
        # Processing the call arguments (line 86)
        
        # Call to copy(...): (line 86)
        # Processing the call keyword arguments (line 86)
        kwargs_372210 = {}
        # Getting the type of 'self' (line 86)
        self_372207 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 31), 'self', False)
        # Obtaining the member 'data' of a type (line 86)
        data_372208 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 86, 31), self_372207, 'data')
        # Obtaining the member 'copy' of a type (line 86)
        copy_372209 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 86, 31), data_372208, 'copy')
        # Calling copy(args, kwargs) (line 86)
        copy_call_result_372211 = invoke(stypy.reporting.localization.Localization(__file__, 86, 31), copy_372209, *[], **kwargs_372210)
        
        # Processing the call keyword arguments (line 86)
        # Getting the type of 'True' (line 86)
        True_372212 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 54), 'True', False)
        keyword_372213 = True_372212
        kwargs_372214 = {'copy': keyword_372213}
        # Getting the type of 'self' (line 86)
        self_372205 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 15), 'self', False)
        # Obtaining the member '_with_data' of a type (line 86)
        _with_data_372206 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 86, 15), self_372205, '_with_data')
        # Calling _with_data(args, kwargs) (line 86)
        _with_data_call_result_372215 = invoke(stypy.reporting.localization.Localization(__file__, 86, 15), _with_data_372206, *[copy_call_result_372211], **kwargs_372214)
        
        # Assigning a type to the variable 'stypy_return_type' (line 86)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 86, 8), 'stypy_return_type', _with_data_call_result_372215)
        
        # ################# End of 'copy(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'copy' in the type store
        # Getting the type of 'stypy_return_type' (line 85)
        stypy_return_type_372216 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_372216)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'copy'
        return stypy_return_type_372216

    
    # Assigning a Attribute to a Attribute (line 88):

    @norecursion
    def count_nonzero(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'count_nonzero'
        module_type_store = module_type_store.open_function_context('count_nonzero', 90, 4, False)
        # Assigning a type to the variable 'self' (line 91)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 91, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        _data_matrix.count_nonzero.__dict__.__setitem__('stypy_localization', localization)
        _data_matrix.count_nonzero.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        _data_matrix.count_nonzero.__dict__.__setitem__('stypy_type_store', module_type_store)
        _data_matrix.count_nonzero.__dict__.__setitem__('stypy_function_name', '_data_matrix.count_nonzero')
        _data_matrix.count_nonzero.__dict__.__setitem__('stypy_param_names_list', [])
        _data_matrix.count_nonzero.__dict__.__setitem__('stypy_varargs_param_name', None)
        _data_matrix.count_nonzero.__dict__.__setitem__('stypy_kwargs_param_name', None)
        _data_matrix.count_nonzero.__dict__.__setitem__('stypy_call_defaults', defaults)
        _data_matrix.count_nonzero.__dict__.__setitem__('stypy_call_varargs', varargs)
        _data_matrix.count_nonzero.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        _data_matrix.count_nonzero.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, '_data_matrix.count_nonzero', [], None, None, defaults, varargs, kwargs)

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

        
        # Call to count_nonzero(...): (line 91)
        # Processing the call arguments (line 91)
        
        # Call to _deduped_data(...): (line 91)
        # Processing the call keyword arguments (line 91)
        kwargs_372221 = {}
        # Getting the type of 'self' (line 91)
        self_372219 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 32), 'self', False)
        # Obtaining the member '_deduped_data' of a type (line 91)
        _deduped_data_372220 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 91, 32), self_372219, '_deduped_data')
        # Calling _deduped_data(args, kwargs) (line 91)
        _deduped_data_call_result_372222 = invoke(stypy.reporting.localization.Localization(__file__, 91, 32), _deduped_data_372220, *[], **kwargs_372221)
        
        # Processing the call keyword arguments (line 91)
        kwargs_372223 = {}
        # Getting the type of 'np' (line 91)
        np_372217 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 15), 'np', False)
        # Obtaining the member 'count_nonzero' of a type (line 91)
        count_nonzero_372218 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 91, 15), np_372217, 'count_nonzero')
        # Calling count_nonzero(args, kwargs) (line 91)
        count_nonzero_call_result_372224 = invoke(stypy.reporting.localization.Localization(__file__, 91, 15), count_nonzero_372218, *[_deduped_data_call_result_372222], **kwargs_372223)
        
        # Assigning a type to the variable 'stypy_return_type' (line 91)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 91, 8), 'stypy_return_type', count_nonzero_call_result_372224)
        
        # ################# End of 'count_nonzero(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'count_nonzero' in the type store
        # Getting the type of 'stypy_return_type' (line 90)
        stypy_return_type_372225 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_372225)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'count_nonzero'
        return stypy_return_type_372225

    
    # Assigning a Attribute to a Attribute (line 93):

    @norecursion
    def power(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 95)
        None_372226 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 29), 'None')
        defaults = [None_372226]
        # Create a new context for function 'power'
        module_type_store = module_type_store.open_function_context('power', 95, 4, False)
        # Assigning a type to the variable 'self' (line 96)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 96, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        _data_matrix.power.__dict__.__setitem__('stypy_localization', localization)
        _data_matrix.power.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        _data_matrix.power.__dict__.__setitem__('stypy_type_store', module_type_store)
        _data_matrix.power.__dict__.__setitem__('stypy_function_name', '_data_matrix.power')
        _data_matrix.power.__dict__.__setitem__('stypy_param_names_list', ['n', 'dtype'])
        _data_matrix.power.__dict__.__setitem__('stypy_varargs_param_name', None)
        _data_matrix.power.__dict__.__setitem__('stypy_kwargs_param_name', None)
        _data_matrix.power.__dict__.__setitem__('stypy_call_defaults', defaults)
        _data_matrix.power.__dict__.__setitem__('stypy_call_varargs', varargs)
        _data_matrix.power.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        _data_matrix.power.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, '_data_matrix.power', ['n', 'dtype'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'power', localization, ['n', 'dtype'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'power(...)' code ##################

        str_372227 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 104, (-1)), 'str', '\n        This function performs element-wise power.\n\n        Parameters\n        ----------\n        n : n is a scalar\n\n        dtype : If dtype is not specified, the current dtype will be preserved.\n        ')
        
        
        
        # Call to isscalarlike(...): (line 105)
        # Processing the call arguments (line 105)
        # Getting the type of 'n' (line 105)
        n_372229 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 28), 'n', False)
        # Processing the call keyword arguments (line 105)
        kwargs_372230 = {}
        # Getting the type of 'isscalarlike' (line 105)
        isscalarlike_372228 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 15), 'isscalarlike', False)
        # Calling isscalarlike(args, kwargs) (line 105)
        isscalarlike_call_result_372231 = invoke(stypy.reporting.localization.Localization(__file__, 105, 15), isscalarlike_372228, *[n_372229], **kwargs_372230)
        
        # Applying the 'not' unary operator (line 105)
        result_not__372232 = python_operator(stypy.reporting.localization.Localization(__file__, 105, 11), 'not', isscalarlike_call_result_372231)
        
        # Testing the type of an if condition (line 105)
        if_condition_372233 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 105, 8), result_not__372232)
        # Assigning a type to the variable 'if_condition_372233' (line 105)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 105, 8), 'if_condition_372233', if_condition_372233)
        # SSA begins for if statement (line 105)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to NotImplementedError(...): (line 106)
        # Processing the call arguments (line 106)
        str_372235 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 106, 38), 'str', 'input is not scalar')
        # Processing the call keyword arguments (line 106)
        kwargs_372236 = {}
        # Getting the type of 'NotImplementedError' (line 106)
        NotImplementedError_372234 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 18), 'NotImplementedError', False)
        # Calling NotImplementedError(args, kwargs) (line 106)
        NotImplementedError_call_result_372237 = invoke(stypy.reporting.localization.Localization(__file__, 106, 18), NotImplementedError_372234, *[str_372235], **kwargs_372236)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 106, 12), NotImplementedError_call_result_372237, 'raise parameter', BaseException)
        # SSA join for if statement (line 105)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Name (line 108):
        
        # Assigning a Call to a Name (line 108):
        
        # Call to _deduped_data(...): (line 108)
        # Processing the call keyword arguments (line 108)
        kwargs_372240 = {}
        # Getting the type of 'self' (line 108)
        self_372238 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 15), 'self', False)
        # Obtaining the member '_deduped_data' of a type (line 108)
        _deduped_data_372239 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 108, 15), self_372238, '_deduped_data')
        # Calling _deduped_data(args, kwargs) (line 108)
        _deduped_data_call_result_372241 = invoke(stypy.reporting.localization.Localization(__file__, 108, 15), _deduped_data_372239, *[], **kwargs_372240)
        
        # Assigning a type to the variable 'data' (line 108)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 108, 8), 'data', _deduped_data_call_result_372241)
        
        # Type idiom detected: calculating its left and rigth part (line 109)
        # Getting the type of 'dtype' (line 109)
        dtype_372242 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 8), 'dtype')
        # Getting the type of 'None' (line 109)
        None_372243 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 24), 'None')
        
        (may_be_372244, more_types_in_union_372245) = may_not_be_none(dtype_372242, None_372243)

        if may_be_372244:

            if more_types_in_union_372245:
                # Runtime conditional SSA (line 109)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a Call to a Name (line 110):
            
            # Assigning a Call to a Name (line 110):
            
            # Call to astype(...): (line 110)
            # Processing the call arguments (line 110)
            # Getting the type of 'dtype' (line 110)
            dtype_372248 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 31), 'dtype', False)
            # Processing the call keyword arguments (line 110)
            kwargs_372249 = {}
            # Getting the type of 'data' (line 110)
            data_372246 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 19), 'data', False)
            # Obtaining the member 'astype' of a type (line 110)
            astype_372247 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 110, 19), data_372246, 'astype')
            # Calling astype(args, kwargs) (line 110)
            astype_call_result_372250 = invoke(stypy.reporting.localization.Localization(__file__, 110, 19), astype_372247, *[dtype_372248], **kwargs_372249)
            
            # Assigning a type to the variable 'data' (line 110)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 110, 12), 'data', astype_call_result_372250)

            if more_types_in_union_372245:
                # SSA join for if statement (line 109)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Call to _with_data(...): (line 111)
        # Processing the call arguments (line 111)
        # Getting the type of 'data' (line 111)
        data_372253 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 31), 'data', False)
        # Getting the type of 'n' (line 111)
        n_372254 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 39), 'n', False)
        # Applying the binary operator '**' (line 111)
        result_pow_372255 = python_operator(stypy.reporting.localization.Localization(__file__, 111, 31), '**', data_372253, n_372254)
        
        # Processing the call keyword arguments (line 111)
        kwargs_372256 = {}
        # Getting the type of 'self' (line 111)
        self_372251 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 15), 'self', False)
        # Obtaining the member '_with_data' of a type (line 111)
        _with_data_372252 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 111, 15), self_372251, '_with_data')
        # Calling _with_data(args, kwargs) (line 111)
        _with_data_call_result_372257 = invoke(stypy.reporting.localization.Localization(__file__, 111, 15), _with_data_372252, *[result_pow_372255], **kwargs_372256)
        
        # Assigning a type to the variable 'stypy_return_type' (line 111)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 111, 8), 'stypy_return_type', _with_data_call_result_372257)
        
        # ################# End of 'power(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'power' in the type store
        # Getting the type of 'stypy_return_type' (line 95)
        stypy_return_type_372258 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_372258)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'power'
        return stypy_return_type_372258


    @norecursion
    def _mul_scalar(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_mul_scalar'
        module_type_store = module_type_store.open_function_context('_mul_scalar', 117, 4, False)
        # Assigning a type to the variable 'self' (line 118)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 118, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        _data_matrix._mul_scalar.__dict__.__setitem__('stypy_localization', localization)
        _data_matrix._mul_scalar.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        _data_matrix._mul_scalar.__dict__.__setitem__('stypy_type_store', module_type_store)
        _data_matrix._mul_scalar.__dict__.__setitem__('stypy_function_name', '_data_matrix._mul_scalar')
        _data_matrix._mul_scalar.__dict__.__setitem__('stypy_param_names_list', ['other'])
        _data_matrix._mul_scalar.__dict__.__setitem__('stypy_varargs_param_name', None)
        _data_matrix._mul_scalar.__dict__.__setitem__('stypy_kwargs_param_name', None)
        _data_matrix._mul_scalar.__dict__.__setitem__('stypy_call_defaults', defaults)
        _data_matrix._mul_scalar.__dict__.__setitem__('stypy_call_varargs', varargs)
        _data_matrix._mul_scalar.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        _data_matrix._mul_scalar.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, '_data_matrix._mul_scalar', ['other'], None, None, defaults, varargs, kwargs)

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

        
        # Call to _with_data(...): (line 118)
        # Processing the call arguments (line 118)
        # Getting the type of 'self' (line 118)
        self_372261 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 31), 'self', False)
        # Obtaining the member 'data' of a type (line 118)
        data_372262 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 118, 31), self_372261, 'data')
        # Getting the type of 'other' (line 118)
        other_372263 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 43), 'other', False)
        # Applying the binary operator '*' (line 118)
        result_mul_372264 = python_operator(stypy.reporting.localization.Localization(__file__, 118, 31), '*', data_372262, other_372263)
        
        # Processing the call keyword arguments (line 118)
        kwargs_372265 = {}
        # Getting the type of 'self' (line 118)
        self_372259 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 15), 'self', False)
        # Obtaining the member '_with_data' of a type (line 118)
        _with_data_372260 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 118, 15), self_372259, '_with_data')
        # Calling _with_data(args, kwargs) (line 118)
        _with_data_call_result_372266 = invoke(stypy.reporting.localization.Localization(__file__, 118, 15), _with_data_372260, *[result_mul_372264], **kwargs_372265)
        
        # Assigning a type to the variable 'stypy_return_type' (line 118)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 118, 8), 'stypy_return_type', _with_data_call_result_372266)
        
        # ################# End of '_mul_scalar(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_mul_scalar' in the type store
        # Getting the type of 'stypy_return_type' (line 117)
        stypy_return_type_372267 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_372267)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_mul_scalar'
        return stypy_return_type_372267


# Assigning a type to the variable '_data_matrix' (line 21)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 21, 0), '_data_matrix', _data_matrix)

# Assigning a Call to a Name (line 30):

# Call to property(...): (line 30)
# Processing the call keyword arguments (line 30)
# Getting the type of '_get_dtype' (line 30)
_get_dtype_372269 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 26), '_get_dtype', False)
keyword_372270 = _get_dtype_372269
# Getting the type of '_set_dtype' (line 30)
_set_dtype_372271 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 43), '_set_dtype', False)
keyword_372272 = _set_dtype_372271
kwargs_372273 = {'fset': keyword_372272, 'fget': keyword_372270}
# Getting the type of 'property' (line 30)
property_372268 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 12), 'property', False)
# Calling property(args, kwargs) (line 30)
property_call_result_372274 = invoke(stypy.reporting.localization.Localization(__file__, 30, 12), property_372268, *[], **kwargs_372273)

# Getting the type of '_data_matrix'
_data_matrix_372275 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), '_data_matrix')
# Setting the type of the member 'dtype' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), _data_matrix_372275, 'dtype', property_call_result_372274)

# Assigning a Attribute to a Attribute (line 78):
# Getting the type of 'spmatrix' (line 78)
spmatrix_372276 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 21), 'spmatrix')
# Obtaining the member 'astype' of a type (line 78)
astype_372277 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 78, 21), spmatrix_372276, 'astype')
# Obtaining the member '__doc__' of a type (line 78)
doc___372278 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 78, 21), astype_372277, '__doc__')
# Getting the type of '_data_matrix'
_data_matrix_372279 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), '_data_matrix')
# Obtaining the member 'astype' of a type
astype_372280 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), _data_matrix_372279, 'astype')
# Setting the type of the member '__doc__' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), astype_372280, '__doc__', doc___372278)

# Assigning a Attribute to a Attribute (line 83):
# Getting the type of 'spmatrix' (line 83)
spmatrix_372281 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 19), 'spmatrix')
# Obtaining the member 'conj' of a type (line 83)
conj_372282 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 83, 19), spmatrix_372281, 'conj')
# Obtaining the member '__doc__' of a type (line 83)
doc___372283 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 83, 19), conj_372282, '__doc__')
# Getting the type of '_data_matrix'
_data_matrix_372284 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), '_data_matrix')
# Obtaining the member 'conj' of a type
conj_372285 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), _data_matrix_372284, 'conj')
# Setting the type of the member '__doc__' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), conj_372285, '__doc__', doc___372283)

# Assigning a Attribute to a Attribute (line 88):
# Getting the type of 'spmatrix' (line 88)
spmatrix_372286 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 19), 'spmatrix')
# Obtaining the member 'copy' of a type (line 88)
copy_372287 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 88, 19), spmatrix_372286, 'copy')
# Obtaining the member '__doc__' of a type (line 88)
doc___372288 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 88, 19), copy_372287, '__doc__')
# Getting the type of '_data_matrix'
_data_matrix_372289 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), '_data_matrix')
# Obtaining the member 'copy' of a type
copy_372290 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), _data_matrix_372289, 'copy')
# Setting the type of the member '__doc__' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), copy_372290, '__doc__', doc___372288)

# Assigning a Attribute to a Attribute (line 93):
# Getting the type of 'spmatrix' (line 93)
spmatrix_372291 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 28), 'spmatrix')
# Obtaining the member 'count_nonzero' of a type (line 93)
count_nonzero_372292 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 93, 28), spmatrix_372291, 'count_nonzero')
# Obtaining the member '__doc__' of a type (line 93)
doc___372293 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 93, 28), count_nonzero_372292, '__doc__')
# Getting the type of '_data_matrix'
_data_matrix_372294 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), '_data_matrix')
# Obtaining the member 'count_nonzero' of a type
count_nonzero_372295 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), _data_matrix_372294, 'count_nonzero')
# Setting the type of the member '__doc__' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), count_nonzero_372295, '__doc__', doc___372293)

# Getting the type of '_ufuncs_with_fixed_point_at_zero' (line 122)
_ufuncs_with_fixed_point_at_zero_372296 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 14), '_ufuncs_with_fixed_point_at_zero')
# Testing the type of a for loop iterable (line 122)
is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 122, 0), _ufuncs_with_fixed_point_at_zero_372296)
# Getting the type of the for loop variable (line 122)
for_loop_var_372297 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 122, 0), _ufuncs_with_fixed_point_at_zero_372296)
# Assigning a type to the variable 'npfunc' (line 122)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 122, 0), 'npfunc', for_loop_var_372297)
# SSA begins for a for statement (line 122)
module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')

# Assigning a Attribute to a Name (line 123):

# Assigning a Attribute to a Name (line 123):
# Getting the type of 'npfunc' (line 123)
npfunc_372298 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 11), 'npfunc')
# Obtaining the member '__name__' of a type (line 123)
name___372299 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 123, 11), npfunc_372298, '__name__')
# Assigning a type to the variable 'name' (line 123)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 123, 4), 'name', name___372299)

@norecursion
def _create_method(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_create_method'
    module_type_store = module_type_store.open_function_context('_create_method', 125, 4, False)
    
    # Passed parameters checking function
    _create_method.stypy_localization = localization
    _create_method.stypy_type_of_self = None
    _create_method.stypy_type_store = module_type_store
    _create_method.stypy_function_name = '_create_method'
    _create_method.stypy_param_names_list = ['op']
    _create_method.stypy_varargs_param_name = None
    _create_method.stypy_kwargs_param_name = None
    _create_method.stypy_call_defaults = defaults
    _create_method.stypy_call_varargs = varargs
    _create_method.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_create_method', ['op'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_create_method', localization, ['op'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_create_method(...)' code ##################


    @norecursion
    def method(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'method'
        module_type_store = module_type_store.open_function_context('method', 126, 8, False)
        
        # Passed parameters checking function
        method.stypy_localization = localization
        method.stypy_type_of_self = None
        method.stypy_type_store = module_type_store
        method.stypy_function_name = 'method'
        method.stypy_param_names_list = ['self']
        method.stypy_varargs_param_name = None
        method.stypy_kwargs_param_name = None
        method.stypy_call_defaults = defaults
        method.stypy_call_varargs = varargs
        method.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, 'method', ['self'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'method', localization, ['self'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'method(...)' code ##################

        
        # Assigning a Call to a Name (line 127):
        
        # Assigning a Call to a Name (line 127):
        
        # Call to op(...): (line 127)
        # Processing the call arguments (line 127)
        # Getting the type of 'self' (line 127)
        self_372301 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 24), 'self', False)
        # Obtaining the member 'data' of a type (line 127)
        data_372302 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 127, 24), self_372301, 'data')
        # Processing the call keyword arguments (line 127)
        kwargs_372303 = {}
        # Getting the type of 'op' (line 127)
        op_372300 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 21), 'op', False)
        # Calling op(args, kwargs) (line 127)
        op_call_result_372304 = invoke(stypy.reporting.localization.Localization(__file__, 127, 21), op_372300, *[data_372302], **kwargs_372303)
        
        # Assigning a type to the variable 'result' (line 127)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 127, 12), 'result', op_call_result_372304)
        
        # Assigning a Call to a Name (line 128):
        
        # Assigning a Call to a Name (line 128):
        
        # Call to _with_data(...): (line 128)
        # Processing the call arguments (line 128)
        # Getting the type of 'result' (line 128)
        result_372307 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 32), 'result', False)
        # Processing the call keyword arguments (line 128)
        # Getting the type of 'True' (line 128)
        True_372308 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 45), 'True', False)
        keyword_372309 = True_372308
        kwargs_372310 = {'copy': keyword_372309}
        # Getting the type of 'self' (line 128)
        self_372305 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 16), 'self', False)
        # Obtaining the member '_with_data' of a type (line 128)
        _with_data_372306 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 128, 16), self_372305, '_with_data')
        # Calling _with_data(args, kwargs) (line 128)
        _with_data_call_result_372311 = invoke(stypy.reporting.localization.Localization(__file__, 128, 16), _with_data_372306, *[result_372307], **kwargs_372310)
        
        # Assigning a type to the variable 'x' (line 128)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 128, 12), 'x', _with_data_call_result_372311)
        # Getting the type of 'x' (line 129)
        x_372312 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 19), 'x')
        # Assigning a type to the variable 'stypy_return_type' (line 129)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 129, 12), 'stypy_return_type', x_372312)
        
        # ################# End of 'method(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'method' in the type store
        # Getting the type of 'stypy_return_type' (line 126)
        stypy_return_type_372313 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 8), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_372313)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'method'
        return stypy_return_type_372313

    # Assigning a type to the variable 'method' (line 126)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 126, 8), 'method', method)
    
    # Assigning a BinOp to a Attribute (line 131):
    
    # Assigning a BinOp to a Attribute (line 131):
    str_372314 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 131, 26), 'str', 'Element-wise %s.\n\nSee numpy.%s for more information.')
    
    # Obtaining an instance of the builtin type 'tuple' (line 132)
    tuple_372315 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 132, 66), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 132)
    # Adding element type (line 132)
    # Getting the type of 'name' (line 132)
    name_372316 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 66), 'name')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 132, 66), tuple_372315, name_372316)
    # Adding element type (line 132)
    # Getting the type of 'name' (line 132)
    name_372317 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 72), 'name')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 132, 66), tuple_372315, name_372317)
    
    # Applying the binary operator '%' (line 131)
    result_mod_372318 = python_operator(stypy.reporting.localization.Localization(__file__, 131, 26), '%', str_372314, tuple_372315)
    
    # Getting the type of 'method' (line 131)
    method_372319 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 8), 'method')
    # Setting the type of the member '__doc__' of a type (line 131)
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 131, 8), method_372319, '__doc__', result_mod_372318)
    
    # Assigning a Name to a Attribute (line 133):
    
    # Assigning a Name to a Attribute (line 133):
    # Getting the type of 'name' (line 133)
    name_372320 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 26), 'name')
    # Getting the type of 'method' (line 133)
    method_372321 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 8), 'method')
    # Setting the type of the member '__name__' of a type (line 133)
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 133, 8), method_372321, '__name__', name_372320)
    # Getting the type of 'method' (line 135)
    method_372322 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 15), 'method')
    # Assigning a type to the variable 'stypy_return_type' (line 135)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 135, 8), 'stypy_return_type', method_372322)
    
    # ################# End of '_create_method(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_create_method' in the type store
    # Getting the type of 'stypy_return_type' (line 125)
    stypy_return_type_372323 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 4), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_372323)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_create_method'
    return stypy_return_type_372323

# Assigning a type to the variable '_create_method' (line 125)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 125, 4), '_create_method', _create_method)

# Call to setattr(...): (line 137)
# Processing the call arguments (line 137)
# Getting the type of '_data_matrix' (line 137)
_data_matrix_372325 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 12), '_data_matrix', False)
# Getting the type of 'name' (line 137)
name_372326 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 26), 'name', False)

# Call to _create_method(...): (line 137)
# Processing the call arguments (line 137)
# Getting the type of 'npfunc' (line 137)
npfunc_372328 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 47), 'npfunc', False)
# Processing the call keyword arguments (line 137)
kwargs_372329 = {}
# Getting the type of '_create_method' (line 137)
_create_method_372327 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 32), '_create_method', False)
# Calling _create_method(args, kwargs) (line 137)
_create_method_call_result_372330 = invoke(stypy.reporting.localization.Localization(__file__, 137, 32), _create_method_372327, *[npfunc_372328], **kwargs_372329)

# Processing the call keyword arguments (line 137)
kwargs_372331 = {}
# Getting the type of 'setattr' (line 137)
setattr_372324 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 4), 'setattr', False)
# Calling setattr(args, kwargs) (line 137)
setattr_call_result_372332 = invoke(stypy.reporting.localization.Localization(__file__, 137, 4), setattr_372324, *[_data_matrix_372325, name_372326, _create_method_call_result_372330], **kwargs_372331)

# SSA join for a for statement
module_type_store = module_type_store.join_ssa_context()


@norecursion
def _find_missing_index(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_find_missing_index'
    module_type_store = module_type_store.open_function_context('_find_missing_index', 140, 0, False)
    
    # Passed parameters checking function
    _find_missing_index.stypy_localization = localization
    _find_missing_index.stypy_type_of_self = None
    _find_missing_index.stypy_type_store = module_type_store
    _find_missing_index.stypy_function_name = '_find_missing_index'
    _find_missing_index.stypy_param_names_list = ['ind', 'n']
    _find_missing_index.stypy_varargs_param_name = None
    _find_missing_index.stypy_kwargs_param_name = None
    _find_missing_index.stypy_call_defaults = defaults
    _find_missing_index.stypy_call_varargs = varargs
    _find_missing_index.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_find_missing_index', ['ind', 'n'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_find_missing_index', localization, ['ind', 'n'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_find_missing_index(...)' code ##################

    
    
    # Call to enumerate(...): (line 141)
    # Processing the call arguments (line 141)
    # Getting the type of 'ind' (line 141)
    ind_372334 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 26), 'ind', False)
    # Processing the call keyword arguments (line 141)
    kwargs_372335 = {}
    # Getting the type of 'enumerate' (line 141)
    enumerate_372333 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 16), 'enumerate', False)
    # Calling enumerate(args, kwargs) (line 141)
    enumerate_call_result_372336 = invoke(stypy.reporting.localization.Localization(__file__, 141, 16), enumerate_372333, *[ind_372334], **kwargs_372335)
    
    # Testing the type of a for loop iterable (line 141)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 141, 4), enumerate_call_result_372336)
    # Getting the type of the for loop variable (line 141)
    for_loop_var_372337 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 141, 4), enumerate_call_result_372336)
    # Assigning a type to the variable 'k' (line 141)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 141, 4), 'k', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 141, 4), for_loop_var_372337))
    # Assigning a type to the variable 'a' (line 141)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 141, 4), 'a', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 141, 4), for_loop_var_372337))
    # SSA begins for a for statement (line 141)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    
    # Getting the type of 'k' (line 142)
    k_372338 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 11), 'k')
    # Getting the type of 'a' (line 142)
    a_372339 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 16), 'a')
    # Applying the binary operator '!=' (line 142)
    result_ne_372340 = python_operator(stypy.reporting.localization.Localization(__file__, 142, 11), '!=', k_372338, a_372339)
    
    # Testing the type of an if condition (line 142)
    if_condition_372341 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 142, 8), result_ne_372340)
    # Assigning a type to the variable 'if_condition_372341' (line 142)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 142, 8), 'if_condition_372341', if_condition_372341)
    # SSA begins for if statement (line 142)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # Getting the type of 'k' (line 143)
    k_372342 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 19), 'k')
    # Assigning a type to the variable 'stypy_return_type' (line 143)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 143, 12), 'stypy_return_type', k_372342)
    # SSA join for if statement (line 142)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Getting the type of 'k' (line 145)
    k_372343 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 4), 'k')
    int_372344 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 145, 9), 'int')
    # Applying the binary operator '+=' (line 145)
    result_iadd_372345 = python_operator(stypy.reporting.localization.Localization(__file__, 145, 4), '+=', k_372343, int_372344)
    # Assigning a type to the variable 'k' (line 145)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 145, 4), 'k', result_iadd_372345)
    
    
    
    # Getting the type of 'k' (line 146)
    k_372346 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 7), 'k')
    # Getting the type of 'n' (line 146)
    n_372347 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 11), 'n')
    # Applying the binary operator '<' (line 146)
    result_lt_372348 = python_operator(stypy.reporting.localization.Localization(__file__, 146, 7), '<', k_372346, n_372347)
    
    # Testing the type of an if condition (line 146)
    if_condition_372349 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 146, 4), result_lt_372348)
    # Assigning a type to the variable 'if_condition_372349' (line 146)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 146, 4), 'if_condition_372349', if_condition_372349)
    # SSA begins for if statement (line 146)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # Getting the type of 'k' (line 147)
    k_372350 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 15), 'k')
    # Assigning a type to the variable 'stypy_return_type' (line 147)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 147, 8), 'stypy_return_type', k_372350)
    # SSA branch for the else part of an if statement (line 146)
    module_type_store.open_ssa_branch('else')
    int_372351 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 149, 15), 'int')
    # Assigning a type to the variable 'stypy_return_type' (line 149)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 149, 8), 'stypy_return_type', int_372351)
    # SSA join for if statement (line 146)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of '_find_missing_index(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_find_missing_index' in the type store
    # Getting the type of 'stypy_return_type' (line 140)
    stypy_return_type_372352 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_372352)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_find_missing_index'
    return stypy_return_type_372352

# Assigning a type to the variable '_find_missing_index' (line 140)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 140, 0), '_find_missing_index', _find_missing_index)
# Declaration of the '_minmax_mixin' class

class _minmax_mixin(object, ):
    str_372353 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 156, (-1)), 'str', 'Mixin for min and max methods.\n\n    These are not implemented for dia_matrix, hence the separate class.\n    ')

    @norecursion
    def _min_or_max_axis(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_min_or_max_axis'
        module_type_store = module_type_store.open_function_context('_min_or_max_axis', 158, 4, False)
        # Assigning a type to the variable 'self' (line 159)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 159, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        _minmax_mixin._min_or_max_axis.__dict__.__setitem__('stypy_localization', localization)
        _minmax_mixin._min_or_max_axis.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        _minmax_mixin._min_or_max_axis.__dict__.__setitem__('stypy_type_store', module_type_store)
        _minmax_mixin._min_or_max_axis.__dict__.__setitem__('stypy_function_name', '_minmax_mixin._min_or_max_axis')
        _minmax_mixin._min_or_max_axis.__dict__.__setitem__('stypy_param_names_list', ['axis', 'min_or_max'])
        _minmax_mixin._min_or_max_axis.__dict__.__setitem__('stypy_varargs_param_name', None)
        _minmax_mixin._min_or_max_axis.__dict__.__setitem__('stypy_kwargs_param_name', None)
        _minmax_mixin._min_or_max_axis.__dict__.__setitem__('stypy_call_defaults', defaults)
        _minmax_mixin._min_or_max_axis.__dict__.__setitem__('stypy_call_varargs', varargs)
        _minmax_mixin._min_or_max_axis.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        _minmax_mixin._min_or_max_axis.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, '_minmax_mixin._min_or_max_axis', ['axis', 'min_or_max'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_min_or_max_axis', localization, ['axis', 'min_or_max'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_min_or_max_axis(...)' code ##################

        
        # Assigning a Subscript to a Name (line 159):
        
        # Assigning a Subscript to a Name (line 159):
        
        # Obtaining the type of the subscript
        # Getting the type of 'axis' (line 159)
        axis_372354 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 23), 'axis')
        # Getting the type of 'self' (line 159)
        self_372355 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 12), 'self')
        # Obtaining the member 'shape' of a type (line 159)
        shape_372356 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 159, 12), self_372355, 'shape')
        # Obtaining the member '__getitem__' of a type (line 159)
        getitem___372357 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 159, 12), shape_372356, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 159)
        subscript_call_result_372358 = invoke(stypy.reporting.localization.Localization(__file__, 159, 12), getitem___372357, axis_372354)
        
        # Assigning a type to the variable 'N' (line 159)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 159, 8), 'N', subscript_call_result_372358)
        
        
        # Getting the type of 'N' (line 160)
        N_372359 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 11), 'N')
        int_372360 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 160, 16), 'int')
        # Applying the binary operator '==' (line 160)
        result_eq_372361 = python_operator(stypy.reporting.localization.Localization(__file__, 160, 11), '==', N_372359, int_372360)
        
        # Testing the type of an if condition (line 160)
        if_condition_372362 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 160, 8), result_eq_372361)
        # Assigning a type to the variable 'if_condition_372362' (line 160)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 160, 8), 'if_condition_372362', if_condition_372362)
        # SSA begins for if statement (line 160)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to ValueError(...): (line 161)
        # Processing the call arguments (line 161)
        str_372364 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 161, 29), 'str', 'zero-size array to reduction operation')
        # Processing the call keyword arguments (line 161)
        kwargs_372365 = {}
        # Getting the type of 'ValueError' (line 161)
        ValueError_372363 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 18), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 161)
        ValueError_call_result_372366 = invoke(stypy.reporting.localization.Localization(__file__, 161, 18), ValueError_372363, *[str_372364], **kwargs_372365)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 161, 12), ValueError_call_result_372366, 'raise parameter', BaseException)
        # SSA join for if statement (line 160)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Subscript to a Name (line 162):
        
        # Assigning a Subscript to a Name (line 162):
        
        # Obtaining the type of the subscript
        int_372367 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 162, 23), 'int')
        # Getting the type of 'axis' (line 162)
        axis_372368 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 27), 'axis')
        # Applying the binary operator '-' (line 162)
        result_sub_372369 = python_operator(stypy.reporting.localization.Localization(__file__, 162, 23), '-', int_372367, axis_372368)
        
        # Getting the type of 'self' (line 162)
        self_372370 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 12), 'self')
        # Obtaining the member 'shape' of a type (line 162)
        shape_372371 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 162, 12), self_372370, 'shape')
        # Obtaining the member '__getitem__' of a type (line 162)
        getitem___372372 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 162, 12), shape_372371, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 162)
        subscript_call_result_372373 = invoke(stypy.reporting.localization.Localization(__file__, 162, 12), getitem___372372, result_sub_372369)
        
        # Assigning a type to the variable 'M' (line 162)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 162, 8), 'M', subscript_call_result_372373)
        
        # Assigning a IfExp to a Name (line 164):
        
        # Assigning a IfExp to a Name (line 164):
        
        
        # Getting the type of 'axis' (line 164)
        axis_372374 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 30), 'axis')
        int_372375 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 164, 38), 'int')
        # Applying the binary operator '==' (line 164)
        result_eq_372376 = python_operator(stypy.reporting.localization.Localization(__file__, 164, 30), '==', axis_372374, int_372375)
        
        # Testing the type of an if expression (line 164)
        is_suitable_condition(stypy.reporting.localization.Localization(__file__, 164, 14), result_eq_372376)
        # SSA begins for if expression (line 164)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if expression')
        
        # Call to tocsc(...): (line 164)
        # Processing the call keyword arguments (line 164)
        kwargs_372379 = {}
        # Getting the type of 'self' (line 164)
        self_372377 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 14), 'self', False)
        # Obtaining the member 'tocsc' of a type (line 164)
        tocsc_372378 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 164, 14), self_372377, 'tocsc')
        # Calling tocsc(args, kwargs) (line 164)
        tocsc_call_result_372380 = invoke(stypy.reporting.localization.Localization(__file__, 164, 14), tocsc_372378, *[], **kwargs_372379)
        
        # SSA branch for the else part of an if expression (line 164)
        module_type_store.open_ssa_branch('if expression else')
        
        # Call to tocsr(...): (line 164)
        # Processing the call keyword arguments (line 164)
        kwargs_372383 = {}
        # Getting the type of 'self' (line 164)
        self_372381 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 45), 'self', False)
        # Obtaining the member 'tocsr' of a type (line 164)
        tocsr_372382 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 164, 45), self_372381, 'tocsr')
        # Calling tocsr(args, kwargs) (line 164)
        tocsr_call_result_372384 = invoke(stypy.reporting.localization.Localization(__file__, 164, 45), tocsr_372382, *[], **kwargs_372383)
        
        # SSA join for if expression (line 164)
        module_type_store = module_type_store.join_ssa_context()
        if_exp_372385 = union_type.UnionType.add(tocsc_call_result_372380, tocsr_call_result_372384)
        
        # Assigning a type to the variable 'mat' (line 164)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 164, 8), 'mat', if_exp_372385)
        
        # Call to sum_duplicates(...): (line 165)
        # Processing the call keyword arguments (line 165)
        kwargs_372388 = {}
        # Getting the type of 'mat' (line 165)
        mat_372386 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 8), 'mat', False)
        # Obtaining the member 'sum_duplicates' of a type (line 165)
        sum_duplicates_372387 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 165, 8), mat_372386, 'sum_duplicates')
        # Calling sum_duplicates(args, kwargs) (line 165)
        sum_duplicates_call_result_372389 = invoke(stypy.reporting.localization.Localization(__file__, 165, 8), sum_duplicates_372387, *[], **kwargs_372388)
        
        
        # Assigning a Call to a Tuple (line 167):
        
        # Assigning a Subscript to a Name (line 167):
        
        # Obtaining the type of the subscript
        int_372390 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 167, 8), 'int')
        
        # Call to _minor_reduce(...): (line 167)
        # Processing the call arguments (line 167)
        # Getting the type of 'min_or_max' (line 167)
        min_or_max_372393 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 47), 'min_or_max', False)
        # Processing the call keyword arguments (line 167)
        kwargs_372394 = {}
        # Getting the type of 'mat' (line 167)
        mat_372391 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 29), 'mat', False)
        # Obtaining the member '_minor_reduce' of a type (line 167)
        _minor_reduce_372392 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 167, 29), mat_372391, '_minor_reduce')
        # Calling _minor_reduce(args, kwargs) (line 167)
        _minor_reduce_call_result_372395 = invoke(stypy.reporting.localization.Localization(__file__, 167, 29), _minor_reduce_372392, *[min_or_max_372393], **kwargs_372394)
        
        # Obtaining the member '__getitem__' of a type (line 167)
        getitem___372396 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 167, 8), _minor_reduce_call_result_372395, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 167)
        subscript_call_result_372397 = invoke(stypy.reporting.localization.Localization(__file__, 167, 8), getitem___372396, int_372390)
        
        # Assigning a type to the variable 'tuple_var_assignment_372042' (line 167)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 167, 8), 'tuple_var_assignment_372042', subscript_call_result_372397)
        
        # Assigning a Subscript to a Name (line 167):
        
        # Obtaining the type of the subscript
        int_372398 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 167, 8), 'int')
        
        # Call to _minor_reduce(...): (line 167)
        # Processing the call arguments (line 167)
        # Getting the type of 'min_or_max' (line 167)
        min_or_max_372401 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 47), 'min_or_max', False)
        # Processing the call keyword arguments (line 167)
        kwargs_372402 = {}
        # Getting the type of 'mat' (line 167)
        mat_372399 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 29), 'mat', False)
        # Obtaining the member '_minor_reduce' of a type (line 167)
        _minor_reduce_372400 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 167, 29), mat_372399, '_minor_reduce')
        # Calling _minor_reduce(args, kwargs) (line 167)
        _minor_reduce_call_result_372403 = invoke(stypy.reporting.localization.Localization(__file__, 167, 29), _minor_reduce_372400, *[min_or_max_372401], **kwargs_372402)
        
        # Obtaining the member '__getitem__' of a type (line 167)
        getitem___372404 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 167, 8), _minor_reduce_call_result_372403, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 167)
        subscript_call_result_372405 = invoke(stypy.reporting.localization.Localization(__file__, 167, 8), getitem___372404, int_372398)
        
        # Assigning a type to the variable 'tuple_var_assignment_372043' (line 167)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 167, 8), 'tuple_var_assignment_372043', subscript_call_result_372405)
        
        # Assigning a Name to a Name (line 167):
        # Getting the type of 'tuple_var_assignment_372042' (line 167)
        tuple_var_assignment_372042_372406 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 8), 'tuple_var_assignment_372042')
        # Assigning a type to the variable 'major_index' (line 167)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 167, 8), 'major_index', tuple_var_assignment_372042_372406)
        
        # Assigning a Name to a Name (line 167):
        # Getting the type of 'tuple_var_assignment_372043' (line 167)
        tuple_var_assignment_372043_372407 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 8), 'tuple_var_assignment_372043')
        # Assigning a type to the variable 'value' (line 167)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 167, 21), 'value', tuple_var_assignment_372043_372407)
        
        # Assigning a Compare to a Name (line 168):
        
        # Assigning a Compare to a Name (line 168):
        
        
        # Obtaining the type of the subscript
        # Getting the type of 'major_index' (line 168)
        major_index_372408 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 39), 'major_index')
        
        # Call to diff(...): (line 168)
        # Processing the call arguments (line 168)
        # Getting the type of 'mat' (line 168)
        mat_372411 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 27), 'mat', False)
        # Obtaining the member 'indptr' of a type (line 168)
        indptr_372412 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 168, 27), mat_372411, 'indptr')
        # Processing the call keyword arguments (line 168)
        kwargs_372413 = {}
        # Getting the type of 'np' (line 168)
        np_372409 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 19), 'np', False)
        # Obtaining the member 'diff' of a type (line 168)
        diff_372410 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 168, 19), np_372409, 'diff')
        # Calling diff(args, kwargs) (line 168)
        diff_call_result_372414 = invoke(stypy.reporting.localization.Localization(__file__, 168, 19), diff_372410, *[indptr_372412], **kwargs_372413)
        
        # Obtaining the member '__getitem__' of a type (line 168)
        getitem___372415 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 168, 19), diff_call_result_372414, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 168)
        subscript_call_result_372416 = invoke(stypy.reporting.localization.Localization(__file__, 168, 19), getitem___372415, major_index_372408)
        
        # Getting the type of 'N' (line 168)
        N_372417 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 54), 'N')
        # Applying the binary operator '<' (line 168)
        result_lt_372418 = python_operator(stypy.reporting.localization.Localization(__file__, 168, 19), '<', subscript_call_result_372416, N_372417)
        
        # Assigning a type to the variable 'not_full' (line 168)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 168, 8), 'not_full', result_lt_372418)
        
        # Assigning a Call to a Subscript (line 169):
        
        # Assigning a Call to a Subscript (line 169):
        
        # Call to min_or_max(...): (line 169)
        # Processing the call arguments (line 169)
        
        # Obtaining the type of the subscript
        # Getting the type of 'not_full' (line 169)
        not_full_372420 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 43), 'not_full', False)
        # Getting the type of 'value' (line 169)
        value_372421 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 37), 'value', False)
        # Obtaining the member '__getitem__' of a type (line 169)
        getitem___372422 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 169, 37), value_372421, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 169)
        subscript_call_result_372423 = invoke(stypy.reporting.localization.Localization(__file__, 169, 37), getitem___372422, not_full_372420)
        
        int_372424 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 169, 54), 'int')
        # Processing the call keyword arguments (line 169)
        kwargs_372425 = {}
        # Getting the type of 'min_or_max' (line 169)
        min_or_max_372419 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 26), 'min_or_max', False)
        # Calling min_or_max(args, kwargs) (line 169)
        min_or_max_call_result_372426 = invoke(stypy.reporting.localization.Localization(__file__, 169, 26), min_or_max_372419, *[subscript_call_result_372423, int_372424], **kwargs_372425)
        
        # Getting the type of 'value' (line 169)
        value_372427 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 8), 'value')
        # Getting the type of 'not_full' (line 169)
        not_full_372428 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 14), 'not_full')
        # Storing an element on a container (line 169)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 169, 8), value_372427, (not_full_372428, min_or_max_call_result_372426))
        
        # Assigning a Compare to a Name (line 171):
        
        # Assigning a Compare to a Name (line 171):
        
        # Getting the type of 'value' (line 171)
        value_372429 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 15), 'value')
        int_372430 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 171, 24), 'int')
        # Applying the binary operator '!=' (line 171)
        result_ne_372431 = python_operator(stypy.reporting.localization.Localization(__file__, 171, 15), '!=', value_372429, int_372430)
        
        # Assigning a type to the variable 'mask' (line 171)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 171, 8), 'mask', result_ne_372431)
        
        # Assigning a Call to a Name (line 172):
        
        # Assigning a Call to a Name (line 172):
        
        # Call to compress(...): (line 172)
        # Processing the call arguments (line 172)
        # Getting the type of 'mask' (line 172)
        mask_372434 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 34), 'mask', False)
        # Getting the type of 'major_index' (line 172)
        major_index_372435 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 40), 'major_index', False)
        # Processing the call keyword arguments (line 172)
        kwargs_372436 = {}
        # Getting the type of 'np' (line 172)
        np_372432 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 22), 'np', False)
        # Obtaining the member 'compress' of a type (line 172)
        compress_372433 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 172, 22), np_372432, 'compress')
        # Calling compress(args, kwargs) (line 172)
        compress_call_result_372437 = invoke(stypy.reporting.localization.Localization(__file__, 172, 22), compress_372433, *[mask_372434, major_index_372435], **kwargs_372436)
        
        # Assigning a type to the variable 'major_index' (line 172)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 172, 8), 'major_index', compress_call_result_372437)
        
        # Assigning a Call to a Name (line 173):
        
        # Assigning a Call to a Name (line 173):
        
        # Call to compress(...): (line 173)
        # Processing the call arguments (line 173)
        # Getting the type of 'mask' (line 173)
        mask_372440 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 28), 'mask', False)
        # Getting the type of 'value' (line 173)
        value_372441 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 34), 'value', False)
        # Processing the call keyword arguments (line 173)
        kwargs_372442 = {}
        # Getting the type of 'np' (line 173)
        np_372438 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 16), 'np', False)
        # Obtaining the member 'compress' of a type (line 173)
        compress_372439 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 173, 16), np_372438, 'compress')
        # Calling compress(args, kwargs) (line 173)
        compress_call_result_372443 = invoke(stypy.reporting.localization.Localization(__file__, 173, 16), compress_372439, *[mask_372440, value_372441], **kwargs_372442)
        
        # Assigning a type to the variable 'value' (line 173)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 173, 8), 'value', compress_call_result_372443)
        stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 175, 8))
        
        # 'from scipy.sparse import coo_matrix' statement (line 175)
        update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/sparse/')
        import_372444 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 175, 8), 'scipy.sparse')

        if (type(import_372444) is not StypyTypeError):

            if (import_372444 != 'pyd_module'):
                __import__(import_372444)
                sys_modules_372445 = sys.modules[import_372444]
                import_from_module(stypy.reporting.localization.Localization(__file__, 175, 8), 'scipy.sparse', sys_modules_372445.module_type_store, module_type_store, ['coo_matrix'])
                nest_module(stypy.reporting.localization.Localization(__file__, 175, 8), __file__, sys_modules_372445, sys_modules_372445.module_type_store, module_type_store)
            else:
                from scipy.sparse import coo_matrix

                import_from_module(stypy.reporting.localization.Localization(__file__, 175, 8), 'scipy.sparse', None, module_type_store, ['coo_matrix'], [coo_matrix])

        else:
            # Assigning a type to the variable 'scipy.sparse' (line 175)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 175, 8), 'scipy.sparse', import_372444)

        remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/sparse/')
        
        
        
        # Getting the type of 'axis' (line 176)
        axis_372446 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 11), 'axis')
        int_372447 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 176, 19), 'int')
        # Applying the binary operator '==' (line 176)
        result_eq_372448 = python_operator(stypy.reporting.localization.Localization(__file__, 176, 11), '==', axis_372446, int_372447)
        
        # Testing the type of an if condition (line 176)
        if_condition_372449 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 176, 8), result_eq_372448)
        # Assigning a type to the variable 'if_condition_372449' (line 176)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 176, 8), 'if_condition_372449', if_condition_372449)
        # SSA begins for if statement (line 176)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to coo_matrix(...): (line 177)
        # Processing the call arguments (line 177)
        
        # Obtaining an instance of the builtin type 'tuple' (line 177)
        tuple_372451 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 177, 31), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 177)
        # Adding element type (line 177)
        # Getting the type of 'value' (line 177)
        value_372452 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 31), 'value', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 177, 31), tuple_372451, value_372452)
        # Adding element type (line 177)
        
        # Obtaining an instance of the builtin type 'tuple' (line 177)
        tuple_372453 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 177, 39), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 177)
        # Adding element type (line 177)
        
        # Call to zeros(...): (line 177)
        # Processing the call arguments (line 177)
        
        # Call to len(...): (line 177)
        # Processing the call arguments (line 177)
        # Getting the type of 'value' (line 177)
        value_372457 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 52), 'value', False)
        # Processing the call keyword arguments (line 177)
        kwargs_372458 = {}
        # Getting the type of 'len' (line 177)
        len_372456 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 48), 'len', False)
        # Calling len(args, kwargs) (line 177)
        len_call_result_372459 = invoke(stypy.reporting.localization.Localization(__file__, 177, 48), len_372456, *[value_372457], **kwargs_372458)
        
        # Processing the call keyword arguments (line 177)
        kwargs_372460 = {}
        # Getting the type of 'np' (line 177)
        np_372454 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 39), 'np', False)
        # Obtaining the member 'zeros' of a type (line 177)
        zeros_372455 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 177, 39), np_372454, 'zeros')
        # Calling zeros(args, kwargs) (line 177)
        zeros_call_result_372461 = invoke(stypy.reporting.localization.Localization(__file__, 177, 39), zeros_372455, *[len_call_result_372459], **kwargs_372460)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 177, 39), tuple_372453, zeros_call_result_372461)
        # Adding element type (line 177)
        # Getting the type of 'major_index' (line 177)
        major_index_372462 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 61), 'major_index', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 177, 39), tuple_372453, major_index_372462)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 177, 31), tuple_372451, tuple_372453)
        
        # Processing the call keyword arguments (line 177)
        # Getting the type of 'self' (line 178)
        self_372463 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 36), 'self', False)
        # Obtaining the member 'dtype' of a type (line 178)
        dtype_372464 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 178, 36), self_372463, 'dtype')
        keyword_372465 = dtype_372464
        
        # Obtaining an instance of the builtin type 'tuple' (line 178)
        tuple_372466 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 178, 55), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 178)
        # Adding element type (line 178)
        int_372467 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 178, 55), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 178, 55), tuple_372466, int_372467)
        # Adding element type (line 178)
        # Getting the type of 'M' (line 178)
        M_372468 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 58), 'M', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 178, 55), tuple_372466, M_372468)
        
        keyword_372469 = tuple_372466
        kwargs_372470 = {'dtype': keyword_372465, 'shape': keyword_372469}
        # Getting the type of 'coo_matrix' (line 177)
        coo_matrix_372450 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 19), 'coo_matrix', False)
        # Calling coo_matrix(args, kwargs) (line 177)
        coo_matrix_call_result_372471 = invoke(stypy.reporting.localization.Localization(__file__, 177, 19), coo_matrix_372450, *[tuple_372451], **kwargs_372470)
        
        # Assigning a type to the variable 'stypy_return_type' (line 177)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 177, 12), 'stypy_return_type', coo_matrix_call_result_372471)
        # SSA branch for the else part of an if statement (line 176)
        module_type_store.open_ssa_branch('else')
        
        # Call to coo_matrix(...): (line 180)
        # Processing the call arguments (line 180)
        
        # Obtaining an instance of the builtin type 'tuple' (line 180)
        tuple_372473 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 180, 31), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 180)
        # Adding element type (line 180)
        # Getting the type of 'value' (line 180)
        value_372474 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 31), 'value', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 180, 31), tuple_372473, value_372474)
        # Adding element type (line 180)
        
        # Obtaining an instance of the builtin type 'tuple' (line 180)
        tuple_372475 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 180, 39), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 180)
        # Adding element type (line 180)
        # Getting the type of 'major_index' (line 180)
        major_index_372476 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 39), 'major_index', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 180, 39), tuple_372475, major_index_372476)
        # Adding element type (line 180)
        
        # Call to zeros(...): (line 180)
        # Processing the call arguments (line 180)
        
        # Call to len(...): (line 180)
        # Processing the call arguments (line 180)
        # Getting the type of 'value' (line 180)
        value_372480 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 65), 'value', False)
        # Processing the call keyword arguments (line 180)
        kwargs_372481 = {}
        # Getting the type of 'len' (line 180)
        len_372479 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 61), 'len', False)
        # Calling len(args, kwargs) (line 180)
        len_call_result_372482 = invoke(stypy.reporting.localization.Localization(__file__, 180, 61), len_372479, *[value_372480], **kwargs_372481)
        
        # Processing the call keyword arguments (line 180)
        kwargs_372483 = {}
        # Getting the type of 'np' (line 180)
        np_372477 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 52), 'np', False)
        # Obtaining the member 'zeros' of a type (line 180)
        zeros_372478 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 180, 52), np_372477, 'zeros')
        # Calling zeros(args, kwargs) (line 180)
        zeros_call_result_372484 = invoke(stypy.reporting.localization.Localization(__file__, 180, 52), zeros_372478, *[len_call_result_372482], **kwargs_372483)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 180, 39), tuple_372475, zeros_call_result_372484)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 180, 31), tuple_372473, tuple_372475)
        
        # Processing the call keyword arguments (line 180)
        # Getting the type of 'self' (line 181)
        self_372485 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 36), 'self', False)
        # Obtaining the member 'dtype' of a type (line 181)
        dtype_372486 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 181, 36), self_372485, 'dtype')
        keyword_372487 = dtype_372486
        
        # Obtaining an instance of the builtin type 'tuple' (line 181)
        tuple_372488 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 181, 55), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 181)
        # Adding element type (line 181)
        # Getting the type of 'M' (line 181)
        M_372489 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 55), 'M', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 181, 55), tuple_372488, M_372489)
        # Adding element type (line 181)
        int_372490 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 181, 58), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 181, 55), tuple_372488, int_372490)
        
        keyword_372491 = tuple_372488
        kwargs_372492 = {'dtype': keyword_372487, 'shape': keyword_372491}
        # Getting the type of 'coo_matrix' (line 180)
        coo_matrix_372472 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 19), 'coo_matrix', False)
        # Calling coo_matrix(args, kwargs) (line 180)
        coo_matrix_call_result_372493 = invoke(stypy.reporting.localization.Localization(__file__, 180, 19), coo_matrix_372472, *[tuple_372473], **kwargs_372492)
        
        # Assigning a type to the variable 'stypy_return_type' (line 180)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 180, 12), 'stypy_return_type', coo_matrix_call_result_372493)
        # SSA join for if statement (line 176)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of '_min_or_max_axis(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_min_or_max_axis' in the type store
        # Getting the type of 'stypy_return_type' (line 158)
        stypy_return_type_372494 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_372494)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_min_or_max_axis'
        return stypy_return_type_372494


    @norecursion
    def _min_or_max(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_min_or_max'
        module_type_store = module_type_store.open_function_context('_min_or_max', 183, 4, False)
        # Assigning a type to the variable 'self' (line 184)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 184, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        _minmax_mixin._min_or_max.__dict__.__setitem__('stypy_localization', localization)
        _minmax_mixin._min_or_max.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        _minmax_mixin._min_or_max.__dict__.__setitem__('stypy_type_store', module_type_store)
        _minmax_mixin._min_or_max.__dict__.__setitem__('stypy_function_name', '_minmax_mixin._min_or_max')
        _minmax_mixin._min_or_max.__dict__.__setitem__('stypy_param_names_list', ['axis', 'out', 'min_or_max'])
        _minmax_mixin._min_or_max.__dict__.__setitem__('stypy_varargs_param_name', None)
        _minmax_mixin._min_or_max.__dict__.__setitem__('stypy_kwargs_param_name', None)
        _minmax_mixin._min_or_max.__dict__.__setitem__('stypy_call_defaults', defaults)
        _minmax_mixin._min_or_max.__dict__.__setitem__('stypy_call_varargs', varargs)
        _minmax_mixin._min_or_max.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        _minmax_mixin._min_or_max.__dict__.__setitem__('stypy_declared_arg_number', 4)
        arguments = process_argument_values(localization, type_of_self, module_type_store, '_minmax_mixin._min_or_max', ['axis', 'out', 'min_or_max'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_min_or_max', localization, ['axis', 'out', 'min_or_max'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_min_or_max(...)' code ##################

        
        # Type idiom detected: calculating its left and rigth part (line 184)
        # Getting the type of 'out' (line 184)
        out_372495 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 8), 'out')
        # Getting the type of 'None' (line 184)
        None_372496 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 22), 'None')
        
        (may_be_372497, more_types_in_union_372498) = may_not_be_none(out_372495, None_372496)

        if may_be_372497:

            if more_types_in_union_372498:
                # Runtime conditional SSA (line 184)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Call to ValueError(...): (line 185)
            # Processing the call arguments (line 185)
            str_372500 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 185, 30), 'str', "Sparse matrices do not support an 'out' parameter.")
            # Processing the call keyword arguments (line 185)
            kwargs_372501 = {}
            # Getting the type of 'ValueError' (line 185)
            ValueError_372499 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 18), 'ValueError', False)
            # Calling ValueError(args, kwargs) (line 185)
            ValueError_call_result_372502 = invoke(stypy.reporting.localization.Localization(__file__, 185, 18), ValueError_372499, *[str_372500], **kwargs_372501)
            
            ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 185, 12), ValueError_call_result_372502, 'raise parameter', BaseException)

            if more_types_in_union_372498:
                # SSA join for if statement (line 184)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Call to validateaxis(...): (line 188)
        # Processing the call arguments (line 188)
        # Getting the type of 'axis' (line 188)
        axis_372504 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 188, 21), 'axis', False)
        # Processing the call keyword arguments (line 188)
        kwargs_372505 = {}
        # Getting the type of 'validateaxis' (line 188)
        validateaxis_372503 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 188, 8), 'validateaxis', False)
        # Calling validateaxis(args, kwargs) (line 188)
        validateaxis_call_result_372506 = invoke(stypy.reporting.localization.Localization(__file__, 188, 8), validateaxis_372503, *[axis_372504], **kwargs_372505)
        
        
        # Type idiom detected: calculating its left and rigth part (line 190)
        # Getting the type of 'axis' (line 190)
        axis_372507 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 190, 11), 'axis')
        # Getting the type of 'None' (line 190)
        None_372508 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 190, 19), 'None')
        
        (may_be_372509, more_types_in_union_372510) = may_be_none(axis_372507, None_372508)

        if may_be_372509:

            if more_types_in_union_372510:
                # Runtime conditional SSA (line 190)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            
            int_372511 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 191, 15), 'int')
            # Getting the type of 'self' (line 191)
            self_372512 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 20), 'self')
            # Obtaining the member 'shape' of a type (line 191)
            shape_372513 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 191, 20), self_372512, 'shape')
            # Applying the binary operator 'in' (line 191)
            result_contains_372514 = python_operator(stypy.reporting.localization.Localization(__file__, 191, 15), 'in', int_372511, shape_372513)
            
            # Testing the type of an if condition (line 191)
            if_condition_372515 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 191, 12), result_contains_372514)
            # Assigning a type to the variable 'if_condition_372515' (line 191)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 191, 12), 'if_condition_372515', if_condition_372515)
            # SSA begins for if statement (line 191)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to ValueError(...): (line 192)
            # Processing the call arguments (line 192)
            str_372517 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 192, 33), 'str', 'zero-size array to reduction operation')
            # Processing the call keyword arguments (line 192)
            kwargs_372518 = {}
            # Getting the type of 'ValueError' (line 192)
            ValueError_372516 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 192, 22), 'ValueError', False)
            # Calling ValueError(args, kwargs) (line 192)
            ValueError_call_result_372519 = invoke(stypy.reporting.localization.Localization(__file__, 192, 22), ValueError_372516, *[str_372517], **kwargs_372518)
            
            ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 192, 16), ValueError_call_result_372519, 'raise parameter', BaseException)
            # SSA join for if statement (line 191)
            module_type_store = module_type_store.join_ssa_context()
            
            
            # Assigning a Call to a Name (line 194):
            
            # Assigning a Call to a Name (line 194):
            
            # Call to type(...): (line 194)
            # Processing the call arguments (line 194)
            int_372523 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 194, 35), 'int')
            # Processing the call keyword arguments (line 194)
            kwargs_372524 = {}
            # Getting the type of 'self' (line 194)
            self_372520 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 194, 19), 'self', False)
            # Obtaining the member 'dtype' of a type (line 194)
            dtype_372521 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 194, 19), self_372520, 'dtype')
            # Obtaining the member 'type' of a type (line 194)
            type_372522 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 194, 19), dtype_372521, 'type')
            # Calling type(args, kwargs) (line 194)
            type_call_result_372525 = invoke(stypy.reporting.localization.Localization(__file__, 194, 19), type_372522, *[int_372523], **kwargs_372524)
            
            # Assigning a type to the variable 'zero' (line 194)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 194, 12), 'zero', type_call_result_372525)
            
            
            # Getting the type of 'self' (line 195)
            self_372526 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 195, 15), 'self')
            # Obtaining the member 'nnz' of a type (line 195)
            nnz_372527 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 195, 15), self_372526, 'nnz')
            int_372528 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 195, 27), 'int')
            # Applying the binary operator '==' (line 195)
            result_eq_372529 = python_operator(stypy.reporting.localization.Localization(__file__, 195, 15), '==', nnz_372527, int_372528)
            
            # Testing the type of an if condition (line 195)
            if_condition_372530 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 195, 12), result_eq_372529)
            # Assigning a type to the variable 'if_condition_372530' (line 195)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 195, 12), 'if_condition_372530', if_condition_372530)
            # SSA begins for if statement (line 195)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            # Getting the type of 'zero' (line 196)
            zero_372531 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 196, 23), 'zero')
            # Assigning a type to the variable 'stypy_return_type' (line 196)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 196, 16), 'stypy_return_type', zero_372531)
            # SSA join for if statement (line 195)
            module_type_store = module_type_store.join_ssa_context()
            
            
            # Assigning a Call to a Name (line 197):
            
            # Assigning a Call to a Name (line 197):
            
            # Call to reduce(...): (line 197)
            # Processing the call arguments (line 197)
            
            # Call to ravel(...): (line 197)
            # Processing the call keyword arguments (line 197)
            kwargs_372539 = {}
            
            # Call to _deduped_data(...): (line 197)
            # Processing the call keyword arguments (line 197)
            kwargs_372536 = {}
            # Getting the type of 'self' (line 197)
            self_372534 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 197, 34), 'self', False)
            # Obtaining the member '_deduped_data' of a type (line 197)
            _deduped_data_372535 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 197, 34), self_372534, '_deduped_data')
            # Calling _deduped_data(args, kwargs) (line 197)
            _deduped_data_call_result_372537 = invoke(stypy.reporting.localization.Localization(__file__, 197, 34), _deduped_data_372535, *[], **kwargs_372536)
            
            # Obtaining the member 'ravel' of a type (line 197)
            ravel_372538 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 197, 34), _deduped_data_call_result_372537, 'ravel')
            # Calling ravel(args, kwargs) (line 197)
            ravel_call_result_372540 = invoke(stypy.reporting.localization.Localization(__file__, 197, 34), ravel_372538, *[], **kwargs_372539)
            
            # Processing the call keyword arguments (line 197)
            kwargs_372541 = {}
            # Getting the type of 'min_or_max' (line 197)
            min_or_max_372532 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 197, 16), 'min_or_max', False)
            # Obtaining the member 'reduce' of a type (line 197)
            reduce_372533 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 197, 16), min_or_max_372532, 'reduce')
            # Calling reduce(args, kwargs) (line 197)
            reduce_call_result_372542 = invoke(stypy.reporting.localization.Localization(__file__, 197, 16), reduce_372533, *[ravel_call_result_372540], **kwargs_372541)
            
            # Assigning a type to the variable 'm' (line 197)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 197, 12), 'm', reduce_call_result_372542)
            
            
            # Getting the type of 'self' (line 198)
            self_372543 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 15), 'self')
            # Obtaining the member 'nnz' of a type (line 198)
            nnz_372544 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 198, 15), self_372543, 'nnz')
            
            # Call to product(...): (line 198)
            # Processing the call arguments (line 198)
            # Getting the type of 'self' (line 198)
            self_372547 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 38), 'self', False)
            # Obtaining the member 'shape' of a type (line 198)
            shape_372548 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 198, 38), self_372547, 'shape')
            # Processing the call keyword arguments (line 198)
            kwargs_372549 = {}
            # Getting the type of 'np' (line 198)
            np_372545 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 27), 'np', False)
            # Obtaining the member 'product' of a type (line 198)
            product_372546 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 198, 27), np_372545, 'product')
            # Calling product(args, kwargs) (line 198)
            product_call_result_372550 = invoke(stypy.reporting.localization.Localization(__file__, 198, 27), product_372546, *[shape_372548], **kwargs_372549)
            
            # Applying the binary operator '!=' (line 198)
            result_ne_372551 = python_operator(stypy.reporting.localization.Localization(__file__, 198, 15), '!=', nnz_372544, product_call_result_372550)
            
            # Testing the type of an if condition (line 198)
            if_condition_372552 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 198, 12), result_ne_372551)
            # Assigning a type to the variable 'if_condition_372552' (line 198)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 198, 12), 'if_condition_372552', if_condition_372552)
            # SSA begins for if statement (line 198)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a Call to a Name (line 199):
            
            # Assigning a Call to a Name (line 199):
            
            # Call to min_or_max(...): (line 199)
            # Processing the call arguments (line 199)
            # Getting the type of 'zero' (line 199)
            zero_372554 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 31), 'zero', False)
            # Getting the type of 'm' (line 199)
            m_372555 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 37), 'm', False)
            # Processing the call keyword arguments (line 199)
            kwargs_372556 = {}
            # Getting the type of 'min_or_max' (line 199)
            min_or_max_372553 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 20), 'min_or_max', False)
            # Calling min_or_max(args, kwargs) (line 199)
            min_or_max_call_result_372557 = invoke(stypy.reporting.localization.Localization(__file__, 199, 20), min_or_max_372553, *[zero_372554, m_372555], **kwargs_372556)
            
            # Assigning a type to the variable 'm' (line 199)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 199, 16), 'm', min_or_max_call_result_372557)
            # SSA join for if statement (line 198)
            module_type_store = module_type_store.join_ssa_context()
            
            # Getting the type of 'm' (line 200)
            m_372558 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 19), 'm')
            # Assigning a type to the variable 'stypy_return_type' (line 200)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 200, 12), 'stypy_return_type', m_372558)

            if more_types_in_union_372510:
                # SSA join for if statement (line 190)
                module_type_store = module_type_store.join_ssa_context()


        
        
        
        # Getting the type of 'axis' (line 202)
        axis_372559 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 11), 'axis')
        int_372560 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 202, 18), 'int')
        # Applying the binary operator '<' (line 202)
        result_lt_372561 = python_operator(stypy.reporting.localization.Localization(__file__, 202, 11), '<', axis_372559, int_372560)
        
        # Testing the type of an if condition (line 202)
        if_condition_372562 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 202, 8), result_lt_372561)
        # Assigning a type to the variable 'if_condition_372562' (line 202)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 202, 8), 'if_condition_372562', if_condition_372562)
        # SSA begins for if statement (line 202)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Getting the type of 'axis' (line 203)
        axis_372563 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 12), 'axis')
        int_372564 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 203, 20), 'int')
        # Applying the binary operator '+=' (line 203)
        result_iadd_372565 = python_operator(stypy.reporting.localization.Localization(__file__, 203, 12), '+=', axis_372563, int_372564)
        # Assigning a type to the variable 'axis' (line 203)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 203, 12), 'axis', result_iadd_372565)
        
        # SSA join for if statement (line 202)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Evaluating a boolean operation
        
        # Getting the type of 'axis' (line 205)
        axis_372566 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 12), 'axis')
        int_372567 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 205, 20), 'int')
        # Applying the binary operator '==' (line 205)
        result_eq_372568 = python_operator(stypy.reporting.localization.Localization(__file__, 205, 12), '==', axis_372566, int_372567)
        
        
        # Getting the type of 'axis' (line 205)
        axis_372569 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 27), 'axis')
        int_372570 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 205, 35), 'int')
        # Applying the binary operator '==' (line 205)
        result_eq_372571 = python_operator(stypy.reporting.localization.Localization(__file__, 205, 27), '==', axis_372569, int_372570)
        
        # Applying the binary operator 'or' (line 205)
        result_or_keyword_372572 = python_operator(stypy.reporting.localization.Localization(__file__, 205, 11), 'or', result_eq_372568, result_eq_372571)
        
        # Testing the type of an if condition (line 205)
        if_condition_372573 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 205, 8), result_or_keyword_372572)
        # Assigning a type to the variable 'if_condition_372573' (line 205)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 205, 8), 'if_condition_372573', if_condition_372573)
        # SSA begins for if statement (line 205)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to _min_or_max_axis(...): (line 206)
        # Processing the call arguments (line 206)
        # Getting the type of 'axis' (line 206)
        axis_372576 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 206, 41), 'axis', False)
        # Getting the type of 'min_or_max' (line 206)
        min_or_max_372577 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 206, 47), 'min_or_max', False)
        # Processing the call keyword arguments (line 206)
        kwargs_372578 = {}
        # Getting the type of 'self' (line 206)
        self_372574 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 206, 19), 'self', False)
        # Obtaining the member '_min_or_max_axis' of a type (line 206)
        _min_or_max_axis_372575 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 206, 19), self_372574, '_min_or_max_axis')
        # Calling _min_or_max_axis(args, kwargs) (line 206)
        _min_or_max_axis_call_result_372579 = invoke(stypy.reporting.localization.Localization(__file__, 206, 19), _min_or_max_axis_372575, *[axis_372576, min_or_max_372577], **kwargs_372578)
        
        # Assigning a type to the variable 'stypy_return_type' (line 206)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 206, 12), 'stypy_return_type', _min_or_max_axis_call_result_372579)
        # SSA branch for the else part of an if statement (line 205)
        module_type_store.open_ssa_branch('else')
        
        # Call to ValueError(...): (line 208)
        # Processing the call arguments (line 208)
        str_372581 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 208, 29), 'str', 'axis out of range')
        # Processing the call keyword arguments (line 208)
        kwargs_372582 = {}
        # Getting the type of 'ValueError' (line 208)
        ValueError_372580 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 208, 18), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 208)
        ValueError_call_result_372583 = invoke(stypy.reporting.localization.Localization(__file__, 208, 18), ValueError_372580, *[str_372581], **kwargs_372582)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 208, 12), ValueError_call_result_372583, 'raise parameter', BaseException)
        # SSA join for if statement (line 205)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of '_min_or_max(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_min_or_max' in the type store
        # Getting the type of 'stypy_return_type' (line 183)
        stypy_return_type_372584 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_372584)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_min_or_max'
        return stypy_return_type_372584


    @norecursion
    def _arg_min_or_max_axis(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_arg_min_or_max_axis'
        module_type_store = module_type_store.open_function_context('_arg_min_or_max_axis', 210, 4, False)
        # Assigning a type to the variable 'self' (line 211)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 211, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        _minmax_mixin._arg_min_or_max_axis.__dict__.__setitem__('stypy_localization', localization)
        _minmax_mixin._arg_min_or_max_axis.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        _minmax_mixin._arg_min_or_max_axis.__dict__.__setitem__('stypy_type_store', module_type_store)
        _minmax_mixin._arg_min_or_max_axis.__dict__.__setitem__('stypy_function_name', '_minmax_mixin._arg_min_or_max_axis')
        _minmax_mixin._arg_min_or_max_axis.__dict__.__setitem__('stypy_param_names_list', ['axis', 'op', 'compare'])
        _minmax_mixin._arg_min_or_max_axis.__dict__.__setitem__('stypy_varargs_param_name', None)
        _minmax_mixin._arg_min_or_max_axis.__dict__.__setitem__('stypy_kwargs_param_name', None)
        _minmax_mixin._arg_min_or_max_axis.__dict__.__setitem__('stypy_call_defaults', defaults)
        _minmax_mixin._arg_min_or_max_axis.__dict__.__setitem__('stypy_call_varargs', varargs)
        _minmax_mixin._arg_min_or_max_axis.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        _minmax_mixin._arg_min_or_max_axis.__dict__.__setitem__('stypy_declared_arg_number', 4)
        arguments = process_argument_values(localization, type_of_self, module_type_store, '_minmax_mixin._arg_min_or_max_axis', ['axis', 'op', 'compare'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_arg_min_or_max_axis', localization, ['axis', 'op', 'compare'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_arg_min_or_max_axis(...)' code ##################

        
        
        
        # Obtaining the type of the subscript
        # Getting the type of 'axis' (line 211)
        axis_372585 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 211, 22), 'axis')
        # Getting the type of 'self' (line 211)
        self_372586 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 211, 11), 'self')
        # Obtaining the member 'shape' of a type (line 211)
        shape_372587 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 211, 11), self_372586, 'shape')
        # Obtaining the member '__getitem__' of a type (line 211)
        getitem___372588 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 211, 11), shape_372587, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 211)
        subscript_call_result_372589 = invoke(stypy.reporting.localization.Localization(__file__, 211, 11), getitem___372588, axis_372585)
        
        int_372590 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 211, 31), 'int')
        # Applying the binary operator '==' (line 211)
        result_eq_372591 = python_operator(stypy.reporting.localization.Localization(__file__, 211, 11), '==', subscript_call_result_372589, int_372590)
        
        # Testing the type of an if condition (line 211)
        if_condition_372592 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 211, 8), result_eq_372591)
        # Assigning a type to the variable 'if_condition_372592' (line 211)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 211, 8), 'if_condition_372592', if_condition_372592)
        # SSA begins for if statement (line 211)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to ValueError(...): (line 212)
        # Processing the call arguments (line 212)
        str_372594 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 212, 29), 'str', "Can't apply the operation along a zero-sized dimension.")
        # Processing the call keyword arguments (line 212)
        kwargs_372595 = {}
        # Getting the type of 'ValueError' (line 212)
        ValueError_372593 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 18), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 212)
        ValueError_call_result_372596 = invoke(stypy.reporting.localization.Localization(__file__, 212, 18), ValueError_372593, *[str_372594], **kwargs_372595)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 212, 12), ValueError_call_result_372596, 'raise parameter', BaseException)
        # SSA join for if statement (line 211)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Getting the type of 'axis' (line 215)
        axis_372597 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 215, 11), 'axis')
        int_372598 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 215, 18), 'int')
        # Applying the binary operator '<' (line 215)
        result_lt_372599 = python_operator(stypy.reporting.localization.Localization(__file__, 215, 11), '<', axis_372597, int_372598)
        
        # Testing the type of an if condition (line 215)
        if_condition_372600 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 215, 8), result_lt_372599)
        # Assigning a type to the variable 'if_condition_372600' (line 215)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 215, 8), 'if_condition_372600', if_condition_372600)
        # SSA begins for if statement (line 215)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Getting the type of 'axis' (line 216)
        axis_372601 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 216, 12), 'axis')
        int_372602 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 216, 20), 'int')
        # Applying the binary operator '+=' (line 216)
        result_iadd_372603 = python_operator(stypy.reporting.localization.Localization(__file__, 216, 12), '+=', axis_372601, int_372602)
        # Assigning a type to the variable 'axis' (line 216)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 216, 12), 'axis', result_iadd_372603)
        
        # SSA join for if statement (line 215)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Name (line 218):
        
        # Assigning a Call to a Name (line 218):
        
        # Call to type(...): (line 218)
        # Processing the call arguments (line 218)
        int_372607 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 218, 31), 'int')
        # Processing the call keyword arguments (line 218)
        kwargs_372608 = {}
        # Getting the type of 'self' (line 218)
        self_372604 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 218, 15), 'self', False)
        # Obtaining the member 'dtype' of a type (line 218)
        dtype_372605 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 218, 15), self_372604, 'dtype')
        # Obtaining the member 'type' of a type (line 218)
        type_372606 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 218, 15), dtype_372605, 'type')
        # Calling type(args, kwargs) (line 218)
        type_call_result_372609 = invoke(stypy.reporting.localization.Localization(__file__, 218, 15), type_372606, *[int_372607], **kwargs_372608)
        
        # Assigning a type to the variable 'zero' (line 218)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 218, 8), 'zero', type_call_result_372609)
        
        # Assigning a IfExp to a Name (line 220):
        
        # Assigning a IfExp to a Name (line 220):
        
        
        # Getting the type of 'axis' (line 220)
        axis_372610 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 220, 30), 'axis')
        int_372611 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 220, 38), 'int')
        # Applying the binary operator '==' (line 220)
        result_eq_372612 = python_operator(stypy.reporting.localization.Localization(__file__, 220, 30), '==', axis_372610, int_372611)
        
        # Testing the type of an if expression (line 220)
        is_suitable_condition(stypy.reporting.localization.Localization(__file__, 220, 14), result_eq_372612)
        # SSA begins for if expression (line 220)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if expression')
        
        # Call to tocsc(...): (line 220)
        # Processing the call keyword arguments (line 220)
        kwargs_372615 = {}
        # Getting the type of 'self' (line 220)
        self_372613 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 220, 14), 'self', False)
        # Obtaining the member 'tocsc' of a type (line 220)
        tocsc_372614 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 220, 14), self_372613, 'tocsc')
        # Calling tocsc(args, kwargs) (line 220)
        tocsc_call_result_372616 = invoke(stypy.reporting.localization.Localization(__file__, 220, 14), tocsc_372614, *[], **kwargs_372615)
        
        # SSA branch for the else part of an if expression (line 220)
        module_type_store.open_ssa_branch('if expression else')
        
        # Call to tocsr(...): (line 220)
        # Processing the call keyword arguments (line 220)
        kwargs_372619 = {}
        # Getting the type of 'self' (line 220)
        self_372617 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 220, 45), 'self', False)
        # Obtaining the member 'tocsr' of a type (line 220)
        tocsr_372618 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 220, 45), self_372617, 'tocsr')
        # Calling tocsr(args, kwargs) (line 220)
        tocsr_call_result_372620 = invoke(stypy.reporting.localization.Localization(__file__, 220, 45), tocsr_372618, *[], **kwargs_372619)
        
        # SSA join for if expression (line 220)
        module_type_store = module_type_store.join_ssa_context()
        if_exp_372621 = union_type.UnionType.add(tocsc_call_result_372616, tocsr_call_result_372620)
        
        # Assigning a type to the variable 'mat' (line 220)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 220, 8), 'mat', if_exp_372621)
        
        # Call to sum_duplicates(...): (line 221)
        # Processing the call keyword arguments (line 221)
        kwargs_372624 = {}
        # Getting the type of 'mat' (line 221)
        mat_372622 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 8), 'mat', False)
        # Obtaining the member 'sum_duplicates' of a type (line 221)
        sum_duplicates_372623 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 221, 8), mat_372622, 'sum_duplicates')
        # Calling sum_duplicates(args, kwargs) (line 221)
        sum_duplicates_call_result_372625 = invoke(stypy.reporting.localization.Localization(__file__, 221, 8), sum_duplicates_372623, *[], **kwargs_372624)
        
        
        # Assigning a Call to a Tuple (line 223):
        
        # Assigning a Subscript to a Name (line 223):
        
        # Obtaining the type of the subscript
        int_372626 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 223, 8), 'int')
        
        # Call to _swap(...): (line 223)
        # Processing the call arguments (line 223)
        # Getting the type of 'mat' (line 223)
        mat_372629 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 40), 'mat', False)
        # Obtaining the member 'shape' of a type (line 223)
        shape_372630 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 223, 40), mat_372629, 'shape')
        # Processing the call keyword arguments (line 223)
        kwargs_372631 = {}
        # Getting the type of 'mat' (line 223)
        mat_372627 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 30), 'mat', False)
        # Obtaining the member '_swap' of a type (line 223)
        _swap_372628 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 223, 30), mat_372627, '_swap')
        # Calling _swap(args, kwargs) (line 223)
        _swap_call_result_372632 = invoke(stypy.reporting.localization.Localization(__file__, 223, 30), _swap_372628, *[shape_372630], **kwargs_372631)
        
        # Obtaining the member '__getitem__' of a type (line 223)
        getitem___372633 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 223, 8), _swap_call_result_372632, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 223)
        subscript_call_result_372634 = invoke(stypy.reporting.localization.Localization(__file__, 223, 8), getitem___372633, int_372626)
        
        # Assigning a type to the variable 'tuple_var_assignment_372044' (line 223)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 223, 8), 'tuple_var_assignment_372044', subscript_call_result_372634)
        
        # Assigning a Subscript to a Name (line 223):
        
        # Obtaining the type of the subscript
        int_372635 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 223, 8), 'int')
        
        # Call to _swap(...): (line 223)
        # Processing the call arguments (line 223)
        # Getting the type of 'mat' (line 223)
        mat_372638 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 40), 'mat', False)
        # Obtaining the member 'shape' of a type (line 223)
        shape_372639 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 223, 40), mat_372638, 'shape')
        # Processing the call keyword arguments (line 223)
        kwargs_372640 = {}
        # Getting the type of 'mat' (line 223)
        mat_372636 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 30), 'mat', False)
        # Obtaining the member '_swap' of a type (line 223)
        _swap_372637 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 223, 30), mat_372636, '_swap')
        # Calling _swap(args, kwargs) (line 223)
        _swap_call_result_372641 = invoke(stypy.reporting.localization.Localization(__file__, 223, 30), _swap_372637, *[shape_372639], **kwargs_372640)
        
        # Obtaining the member '__getitem__' of a type (line 223)
        getitem___372642 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 223, 8), _swap_call_result_372641, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 223)
        subscript_call_result_372643 = invoke(stypy.reporting.localization.Localization(__file__, 223, 8), getitem___372642, int_372635)
        
        # Assigning a type to the variable 'tuple_var_assignment_372045' (line 223)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 223, 8), 'tuple_var_assignment_372045', subscript_call_result_372643)
        
        # Assigning a Name to a Name (line 223):
        # Getting the type of 'tuple_var_assignment_372044' (line 223)
        tuple_var_assignment_372044_372644 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 8), 'tuple_var_assignment_372044')
        # Assigning a type to the variable 'ret_size' (line 223)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 223, 8), 'ret_size', tuple_var_assignment_372044_372644)
        
        # Assigning a Name to a Name (line 223):
        # Getting the type of 'tuple_var_assignment_372045' (line 223)
        tuple_var_assignment_372045_372645 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 8), 'tuple_var_assignment_372045')
        # Assigning a type to the variable 'line_size' (line 223)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 223, 18), 'line_size', tuple_var_assignment_372045_372645)
        
        # Assigning a Call to a Name (line 224):
        
        # Assigning a Call to a Name (line 224):
        
        # Call to zeros(...): (line 224)
        # Processing the call arguments (line 224)
        # Getting the type of 'ret_size' (line 224)
        ret_size_372648 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 224, 23), 'ret_size', False)
        # Processing the call keyword arguments (line 224)
        # Getting the type of 'int' (line 224)
        int_372649 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 224, 39), 'int', False)
        keyword_372650 = int_372649
        kwargs_372651 = {'dtype': keyword_372650}
        # Getting the type of 'np' (line 224)
        np_372646 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 224, 14), 'np', False)
        # Obtaining the member 'zeros' of a type (line 224)
        zeros_372647 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 224, 14), np_372646, 'zeros')
        # Calling zeros(args, kwargs) (line 224)
        zeros_call_result_372652 = invoke(stypy.reporting.localization.Localization(__file__, 224, 14), zeros_372647, *[ret_size_372648], **kwargs_372651)
        
        # Assigning a type to the variable 'ret' (line 224)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 224, 8), 'ret', zeros_call_result_372652)
        
        # Assigning a Call to a Tuple (line 226):
        
        # Assigning a Subscript to a Name (line 226):
        
        # Obtaining the type of the subscript
        int_372653 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 226, 8), 'int')
        
        # Call to nonzero(...): (line 226)
        # Processing the call arguments (line 226)
        
        # Call to diff(...): (line 226)
        # Processing the call arguments (line 226)
        # Getting the type of 'mat' (line 226)
        mat_372658 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 39), 'mat', False)
        # Obtaining the member 'indptr' of a type (line 226)
        indptr_372659 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 226, 39), mat_372658, 'indptr')
        # Processing the call keyword arguments (line 226)
        kwargs_372660 = {}
        # Getting the type of 'np' (line 226)
        np_372656 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 31), 'np', False)
        # Obtaining the member 'diff' of a type (line 226)
        diff_372657 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 226, 31), np_372656, 'diff')
        # Calling diff(args, kwargs) (line 226)
        diff_call_result_372661 = invoke(stypy.reporting.localization.Localization(__file__, 226, 31), diff_372657, *[indptr_372659], **kwargs_372660)
        
        # Processing the call keyword arguments (line 226)
        kwargs_372662 = {}
        # Getting the type of 'np' (line 226)
        np_372654 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 20), 'np', False)
        # Obtaining the member 'nonzero' of a type (line 226)
        nonzero_372655 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 226, 20), np_372654, 'nonzero')
        # Calling nonzero(args, kwargs) (line 226)
        nonzero_call_result_372663 = invoke(stypy.reporting.localization.Localization(__file__, 226, 20), nonzero_372655, *[diff_call_result_372661], **kwargs_372662)
        
        # Obtaining the member '__getitem__' of a type (line 226)
        getitem___372664 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 226, 8), nonzero_call_result_372663, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 226)
        subscript_call_result_372665 = invoke(stypy.reporting.localization.Localization(__file__, 226, 8), getitem___372664, int_372653)
        
        # Assigning a type to the variable 'tuple_var_assignment_372046' (line 226)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 226, 8), 'tuple_var_assignment_372046', subscript_call_result_372665)
        
        # Assigning a Name to a Name (line 226):
        # Getting the type of 'tuple_var_assignment_372046' (line 226)
        tuple_var_assignment_372046_372666 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 8), 'tuple_var_assignment_372046')
        # Assigning a type to the variable 'nz_lines' (line 226)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 226, 8), 'nz_lines', tuple_var_assignment_372046_372666)
        
        # Getting the type of 'nz_lines' (line 227)
        nz_lines_372667 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 17), 'nz_lines')
        # Testing the type of a for loop iterable (line 227)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 227, 8), nz_lines_372667)
        # Getting the type of the for loop variable (line 227)
        for_loop_var_372668 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 227, 8), nz_lines_372667)
        # Assigning a type to the variable 'i' (line 227)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 227, 8), 'i', for_loop_var_372668)
        # SSA begins for a for statement (line 227)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Subscript to a Tuple (line 228):
        
        # Assigning a Subscript to a Name (line 228):
        
        # Obtaining the type of the subscript
        int_372669 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 228, 12), 'int')
        
        # Obtaining the type of the subscript
        # Getting the type of 'i' (line 228)
        i_372670 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 30), 'i')
        # Getting the type of 'i' (line 228)
        i_372671 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 32), 'i')
        int_372672 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 228, 36), 'int')
        # Applying the binary operator '+' (line 228)
        result_add_372673 = python_operator(stypy.reporting.localization.Localization(__file__, 228, 32), '+', i_372671, int_372672)
        
        slice_372674 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 228, 19), i_372670, result_add_372673, None)
        # Getting the type of 'mat' (line 228)
        mat_372675 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 19), 'mat')
        # Obtaining the member 'indptr' of a type (line 228)
        indptr_372676 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 228, 19), mat_372675, 'indptr')
        # Obtaining the member '__getitem__' of a type (line 228)
        getitem___372677 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 228, 19), indptr_372676, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 228)
        subscript_call_result_372678 = invoke(stypy.reporting.localization.Localization(__file__, 228, 19), getitem___372677, slice_372674)
        
        # Obtaining the member '__getitem__' of a type (line 228)
        getitem___372679 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 228, 12), subscript_call_result_372678, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 228)
        subscript_call_result_372680 = invoke(stypy.reporting.localization.Localization(__file__, 228, 12), getitem___372679, int_372669)
        
        # Assigning a type to the variable 'tuple_var_assignment_372047' (line 228)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 228, 12), 'tuple_var_assignment_372047', subscript_call_result_372680)
        
        # Assigning a Subscript to a Name (line 228):
        
        # Obtaining the type of the subscript
        int_372681 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 228, 12), 'int')
        
        # Obtaining the type of the subscript
        # Getting the type of 'i' (line 228)
        i_372682 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 30), 'i')
        # Getting the type of 'i' (line 228)
        i_372683 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 32), 'i')
        int_372684 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 228, 36), 'int')
        # Applying the binary operator '+' (line 228)
        result_add_372685 = python_operator(stypy.reporting.localization.Localization(__file__, 228, 32), '+', i_372683, int_372684)
        
        slice_372686 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 228, 19), i_372682, result_add_372685, None)
        # Getting the type of 'mat' (line 228)
        mat_372687 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 19), 'mat')
        # Obtaining the member 'indptr' of a type (line 228)
        indptr_372688 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 228, 19), mat_372687, 'indptr')
        # Obtaining the member '__getitem__' of a type (line 228)
        getitem___372689 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 228, 19), indptr_372688, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 228)
        subscript_call_result_372690 = invoke(stypy.reporting.localization.Localization(__file__, 228, 19), getitem___372689, slice_372686)
        
        # Obtaining the member '__getitem__' of a type (line 228)
        getitem___372691 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 228, 12), subscript_call_result_372690, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 228)
        subscript_call_result_372692 = invoke(stypy.reporting.localization.Localization(__file__, 228, 12), getitem___372691, int_372681)
        
        # Assigning a type to the variable 'tuple_var_assignment_372048' (line 228)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 228, 12), 'tuple_var_assignment_372048', subscript_call_result_372692)
        
        # Assigning a Name to a Name (line 228):
        # Getting the type of 'tuple_var_assignment_372047' (line 228)
        tuple_var_assignment_372047_372693 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 12), 'tuple_var_assignment_372047')
        # Assigning a type to the variable 'p' (line 228)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 228, 12), 'p', tuple_var_assignment_372047_372693)
        
        # Assigning a Name to a Name (line 228):
        # Getting the type of 'tuple_var_assignment_372048' (line 228)
        tuple_var_assignment_372048_372694 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 12), 'tuple_var_assignment_372048')
        # Assigning a type to the variable 'q' (line 228)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 228, 15), 'q', tuple_var_assignment_372048_372694)
        
        # Assigning a Subscript to a Name (line 229):
        
        # Assigning a Subscript to a Name (line 229):
        
        # Obtaining the type of the subscript
        # Getting the type of 'p' (line 229)
        p_372695 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 229, 28), 'p')
        # Getting the type of 'q' (line 229)
        q_372696 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 229, 30), 'q')
        slice_372697 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 229, 19), p_372695, q_372696, None)
        # Getting the type of 'mat' (line 229)
        mat_372698 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 229, 19), 'mat')
        # Obtaining the member 'data' of a type (line 229)
        data_372699 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 229, 19), mat_372698, 'data')
        # Obtaining the member '__getitem__' of a type (line 229)
        getitem___372700 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 229, 19), data_372699, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 229)
        subscript_call_result_372701 = invoke(stypy.reporting.localization.Localization(__file__, 229, 19), getitem___372700, slice_372697)
        
        # Assigning a type to the variable 'data' (line 229)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 229, 12), 'data', subscript_call_result_372701)
        
        # Assigning a Subscript to a Name (line 230):
        
        # Assigning a Subscript to a Name (line 230):
        
        # Obtaining the type of the subscript
        # Getting the type of 'p' (line 230)
        p_372702 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 34), 'p')
        # Getting the type of 'q' (line 230)
        q_372703 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 36), 'q')
        slice_372704 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 230, 22), p_372702, q_372703, None)
        # Getting the type of 'mat' (line 230)
        mat_372705 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 22), 'mat')
        # Obtaining the member 'indices' of a type (line 230)
        indices_372706 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 230, 22), mat_372705, 'indices')
        # Obtaining the member '__getitem__' of a type (line 230)
        getitem___372707 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 230, 22), indices_372706, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 230)
        subscript_call_result_372708 = invoke(stypy.reporting.localization.Localization(__file__, 230, 22), getitem___372707, slice_372704)
        
        # Assigning a type to the variable 'indices' (line 230)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 230, 12), 'indices', subscript_call_result_372708)
        
        # Assigning a Call to a Name (line 231):
        
        # Assigning a Call to a Name (line 231):
        
        # Call to op(...): (line 231)
        # Processing the call arguments (line 231)
        # Getting the type of 'data' (line 231)
        data_372710 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 231, 20), 'data', False)
        # Processing the call keyword arguments (line 231)
        kwargs_372711 = {}
        # Getting the type of 'op' (line 231)
        op_372709 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 231, 17), 'op', False)
        # Calling op(args, kwargs) (line 231)
        op_call_result_372712 = invoke(stypy.reporting.localization.Localization(__file__, 231, 17), op_372709, *[data_372710], **kwargs_372711)
        
        # Assigning a type to the variable 'am' (line 231)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 231, 12), 'am', op_call_result_372712)
        
        # Assigning a Subscript to a Name (line 232):
        
        # Assigning a Subscript to a Name (line 232):
        
        # Obtaining the type of the subscript
        # Getting the type of 'am' (line 232)
        am_372713 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 21), 'am')
        # Getting the type of 'data' (line 232)
        data_372714 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 16), 'data')
        # Obtaining the member '__getitem__' of a type (line 232)
        getitem___372715 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 232, 16), data_372714, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 232)
        subscript_call_result_372716 = invoke(stypy.reporting.localization.Localization(__file__, 232, 16), getitem___372715, am_372713)
        
        # Assigning a type to the variable 'm' (line 232)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 232, 12), 'm', subscript_call_result_372716)
        
        
        # Evaluating a boolean operation
        
        # Call to compare(...): (line 233)
        # Processing the call arguments (line 233)
        # Getting the type of 'm' (line 233)
        m_372718 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 233, 23), 'm', False)
        # Getting the type of 'zero' (line 233)
        zero_372719 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 233, 26), 'zero', False)
        # Processing the call keyword arguments (line 233)
        kwargs_372720 = {}
        # Getting the type of 'compare' (line 233)
        compare_372717 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 233, 15), 'compare', False)
        # Calling compare(args, kwargs) (line 233)
        compare_call_result_372721 = invoke(stypy.reporting.localization.Localization(__file__, 233, 15), compare_372717, *[m_372718, zero_372719], **kwargs_372720)
        
        
        # Getting the type of 'q' (line 233)
        q_372722 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 233, 35), 'q')
        # Getting the type of 'p' (line 233)
        p_372723 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 233, 39), 'p')
        # Applying the binary operator '-' (line 233)
        result_sub_372724 = python_operator(stypy.reporting.localization.Localization(__file__, 233, 35), '-', q_372722, p_372723)
        
        # Getting the type of 'line_size' (line 233)
        line_size_372725 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 233, 44), 'line_size')
        # Applying the binary operator '==' (line 233)
        result_eq_372726 = python_operator(stypy.reporting.localization.Localization(__file__, 233, 35), '==', result_sub_372724, line_size_372725)
        
        # Applying the binary operator 'or' (line 233)
        result_or_keyword_372727 = python_operator(stypy.reporting.localization.Localization(__file__, 233, 15), 'or', compare_call_result_372721, result_eq_372726)
        
        # Testing the type of an if condition (line 233)
        if_condition_372728 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 233, 12), result_or_keyword_372727)
        # Assigning a type to the variable 'if_condition_372728' (line 233)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 233, 12), 'if_condition_372728', if_condition_372728)
        # SSA begins for if statement (line 233)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Subscript to a Subscript (line 234):
        
        # Assigning a Subscript to a Subscript (line 234):
        
        # Obtaining the type of the subscript
        # Getting the type of 'am' (line 234)
        am_372729 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 234, 33), 'am')
        # Getting the type of 'indices' (line 234)
        indices_372730 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 234, 25), 'indices')
        # Obtaining the member '__getitem__' of a type (line 234)
        getitem___372731 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 234, 25), indices_372730, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 234)
        subscript_call_result_372732 = invoke(stypy.reporting.localization.Localization(__file__, 234, 25), getitem___372731, am_372729)
        
        # Getting the type of 'ret' (line 234)
        ret_372733 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 234, 16), 'ret')
        # Getting the type of 'i' (line 234)
        i_372734 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 234, 20), 'i')
        # Storing an element on a container (line 234)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 234, 16), ret_372733, (i_372734, subscript_call_result_372732))
        # SSA branch for the else part of an if statement (line 233)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Call to a Name (line 236):
        
        # Assigning a Call to a Name (line 236):
        
        # Call to _find_missing_index(...): (line 236)
        # Processing the call arguments (line 236)
        # Getting the type of 'indices' (line 236)
        indices_372736 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 236, 47), 'indices', False)
        # Getting the type of 'line_size' (line 236)
        line_size_372737 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 236, 56), 'line_size', False)
        # Processing the call keyword arguments (line 236)
        kwargs_372738 = {}
        # Getting the type of '_find_missing_index' (line 236)
        _find_missing_index_372735 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 236, 27), '_find_missing_index', False)
        # Calling _find_missing_index(args, kwargs) (line 236)
        _find_missing_index_call_result_372739 = invoke(stypy.reporting.localization.Localization(__file__, 236, 27), _find_missing_index_372735, *[indices_372736, line_size_372737], **kwargs_372738)
        
        # Assigning a type to the variable 'zero_ind' (line 236)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 236, 16), 'zero_ind', _find_missing_index_call_result_372739)
        
        
        # Getting the type of 'm' (line 237)
        m_372740 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 237, 19), 'm')
        # Getting the type of 'zero' (line 237)
        zero_372741 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 237, 24), 'zero')
        # Applying the binary operator '==' (line 237)
        result_eq_372742 = python_operator(stypy.reporting.localization.Localization(__file__, 237, 19), '==', m_372740, zero_372741)
        
        # Testing the type of an if condition (line 237)
        if_condition_372743 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 237, 16), result_eq_372742)
        # Assigning a type to the variable 'if_condition_372743' (line 237)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 237, 16), 'if_condition_372743', if_condition_372743)
        # SSA begins for if statement (line 237)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Subscript (line 238):
        
        # Assigning a Call to a Subscript (line 238):
        
        # Call to min(...): (line 238)
        # Processing the call arguments (line 238)
        # Getting the type of 'am' (line 238)
        am_372745 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 238, 33), 'am', False)
        # Getting the type of 'zero_ind' (line 238)
        zero_ind_372746 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 238, 37), 'zero_ind', False)
        # Processing the call keyword arguments (line 238)
        kwargs_372747 = {}
        # Getting the type of 'min' (line 238)
        min_372744 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 238, 29), 'min', False)
        # Calling min(args, kwargs) (line 238)
        min_call_result_372748 = invoke(stypy.reporting.localization.Localization(__file__, 238, 29), min_372744, *[am_372745, zero_ind_372746], **kwargs_372747)
        
        # Getting the type of 'ret' (line 238)
        ret_372749 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 238, 20), 'ret')
        # Getting the type of 'i' (line 238)
        i_372750 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 238, 24), 'i')
        # Storing an element on a container (line 238)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 238, 20), ret_372749, (i_372750, min_call_result_372748))
        # SSA branch for the else part of an if statement (line 237)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Name to a Subscript (line 240):
        
        # Assigning a Name to a Subscript (line 240):
        # Getting the type of 'zero_ind' (line 240)
        zero_ind_372751 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 29), 'zero_ind')
        # Getting the type of 'ret' (line 240)
        ret_372752 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 20), 'ret')
        # Getting the type of 'i' (line 240)
        i_372753 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 24), 'i')
        # Storing an element on a container (line 240)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 240, 20), ret_372752, (i_372753, zero_ind_372751))
        # SSA join for if statement (line 237)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 233)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Getting the type of 'axis' (line 242)
        axis_372754 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 242, 11), 'axis')
        int_372755 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 242, 19), 'int')
        # Applying the binary operator '==' (line 242)
        result_eq_372756 = python_operator(stypy.reporting.localization.Localization(__file__, 242, 11), '==', axis_372754, int_372755)
        
        # Testing the type of an if condition (line 242)
        if_condition_372757 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 242, 8), result_eq_372756)
        # Assigning a type to the variable 'if_condition_372757' (line 242)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 242, 8), 'if_condition_372757', if_condition_372757)
        # SSA begins for if statement (line 242)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 243):
        
        # Assigning a Call to a Name (line 243):
        
        # Call to reshape(...): (line 243)
        # Processing the call arguments (line 243)
        int_372760 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 243, 30), 'int')
        int_372761 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 243, 34), 'int')
        # Processing the call keyword arguments (line 243)
        kwargs_372762 = {}
        # Getting the type of 'ret' (line 243)
        ret_372758 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 243, 18), 'ret', False)
        # Obtaining the member 'reshape' of a type (line 243)
        reshape_372759 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 243, 18), ret_372758, 'reshape')
        # Calling reshape(args, kwargs) (line 243)
        reshape_call_result_372763 = invoke(stypy.reporting.localization.Localization(__file__, 243, 18), reshape_372759, *[int_372760, int_372761], **kwargs_372762)
        
        # Assigning a type to the variable 'ret' (line 243)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 243, 12), 'ret', reshape_call_result_372763)
        # SSA join for if statement (line 242)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to asmatrix(...): (line 245)
        # Processing the call arguments (line 245)
        # Getting the type of 'ret' (line 245)
        ret_372766 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 245, 27), 'ret', False)
        # Processing the call keyword arguments (line 245)
        kwargs_372767 = {}
        # Getting the type of 'np' (line 245)
        np_372764 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 245, 15), 'np', False)
        # Obtaining the member 'asmatrix' of a type (line 245)
        asmatrix_372765 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 245, 15), np_372764, 'asmatrix')
        # Calling asmatrix(args, kwargs) (line 245)
        asmatrix_call_result_372768 = invoke(stypy.reporting.localization.Localization(__file__, 245, 15), asmatrix_372765, *[ret_372766], **kwargs_372767)
        
        # Assigning a type to the variable 'stypy_return_type' (line 245)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 245, 8), 'stypy_return_type', asmatrix_call_result_372768)
        
        # ################# End of '_arg_min_or_max_axis(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_arg_min_or_max_axis' in the type store
        # Getting the type of 'stypy_return_type' (line 210)
        stypy_return_type_372769 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_372769)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_arg_min_or_max_axis'
        return stypy_return_type_372769


    @norecursion
    def _arg_min_or_max(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_arg_min_or_max'
        module_type_store = module_type_store.open_function_context('_arg_min_or_max', 247, 4, False)
        # Assigning a type to the variable 'self' (line 248)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 248, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        _minmax_mixin._arg_min_or_max.__dict__.__setitem__('stypy_localization', localization)
        _minmax_mixin._arg_min_or_max.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        _minmax_mixin._arg_min_or_max.__dict__.__setitem__('stypy_type_store', module_type_store)
        _minmax_mixin._arg_min_or_max.__dict__.__setitem__('stypy_function_name', '_minmax_mixin._arg_min_or_max')
        _minmax_mixin._arg_min_or_max.__dict__.__setitem__('stypy_param_names_list', ['axis', 'out', 'op', 'compare'])
        _minmax_mixin._arg_min_or_max.__dict__.__setitem__('stypy_varargs_param_name', None)
        _minmax_mixin._arg_min_or_max.__dict__.__setitem__('stypy_kwargs_param_name', None)
        _minmax_mixin._arg_min_or_max.__dict__.__setitem__('stypy_call_defaults', defaults)
        _minmax_mixin._arg_min_or_max.__dict__.__setitem__('stypy_call_varargs', varargs)
        _minmax_mixin._arg_min_or_max.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        _minmax_mixin._arg_min_or_max.__dict__.__setitem__('stypy_declared_arg_number', 5)
        arguments = process_argument_values(localization, type_of_self, module_type_store, '_minmax_mixin._arg_min_or_max', ['axis', 'out', 'op', 'compare'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_arg_min_or_max', localization, ['axis', 'out', 'op', 'compare'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_arg_min_or_max(...)' code ##################

        
        # Type idiom detected: calculating its left and rigth part (line 248)
        # Getting the type of 'out' (line 248)
        out_372770 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 248, 8), 'out')
        # Getting the type of 'None' (line 248)
        None_372771 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 248, 22), 'None')
        
        (may_be_372772, more_types_in_union_372773) = may_not_be_none(out_372770, None_372771)

        if may_be_372772:

            if more_types_in_union_372773:
                # Runtime conditional SSA (line 248)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Call to ValueError(...): (line 249)
            # Processing the call arguments (line 249)
            str_372775 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 249, 29), 'str', "Sparse matrices do not support an 'out' parameter.")
            # Processing the call keyword arguments (line 249)
            kwargs_372776 = {}
            # Getting the type of 'ValueError' (line 249)
            ValueError_372774 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 249, 18), 'ValueError', False)
            # Calling ValueError(args, kwargs) (line 249)
            ValueError_call_result_372777 = invoke(stypy.reporting.localization.Localization(__file__, 249, 18), ValueError_372774, *[str_372775], **kwargs_372776)
            
            ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 249, 12), ValueError_call_result_372777, 'raise parameter', BaseException)

            if more_types_in_union_372773:
                # SSA join for if statement (line 248)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Call to validateaxis(...): (line 252)
        # Processing the call arguments (line 252)
        # Getting the type of 'axis' (line 252)
        axis_372779 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 252, 21), 'axis', False)
        # Processing the call keyword arguments (line 252)
        kwargs_372780 = {}
        # Getting the type of 'validateaxis' (line 252)
        validateaxis_372778 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 252, 8), 'validateaxis', False)
        # Calling validateaxis(args, kwargs) (line 252)
        validateaxis_call_result_372781 = invoke(stypy.reporting.localization.Localization(__file__, 252, 8), validateaxis_372778, *[axis_372779], **kwargs_372780)
        
        
        # Type idiom detected: calculating its left and rigth part (line 254)
        # Getting the type of 'axis' (line 254)
        axis_372782 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 254, 11), 'axis')
        # Getting the type of 'None' (line 254)
        None_372783 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 254, 19), 'None')
        
        (may_be_372784, more_types_in_union_372785) = may_be_none(axis_372782, None_372783)

        if may_be_372784:

            if more_types_in_union_372785:
                # Runtime conditional SSA (line 254)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            
            int_372786 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 255, 15), 'int')
            # Getting the type of 'self' (line 255)
            self_372787 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 255, 20), 'self')
            # Obtaining the member 'shape' of a type (line 255)
            shape_372788 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 255, 20), self_372787, 'shape')
            # Applying the binary operator 'in' (line 255)
            result_contains_372789 = python_operator(stypy.reporting.localization.Localization(__file__, 255, 15), 'in', int_372786, shape_372788)
            
            # Testing the type of an if condition (line 255)
            if_condition_372790 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 255, 12), result_contains_372789)
            # Assigning a type to the variable 'if_condition_372790' (line 255)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 255, 12), 'if_condition_372790', if_condition_372790)
            # SSA begins for if statement (line 255)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to ValueError(...): (line 256)
            # Processing the call arguments (line 256)
            str_372792 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 256, 33), 'str', "Can't apply the operation to an empty matrix.")
            # Processing the call keyword arguments (line 256)
            kwargs_372793 = {}
            # Getting the type of 'ValueError' (line 256)
            ValueError_372791 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 256, 22), 'ValueError', False)
            # Calling ValueError(args, kwargs) (line 256)
            ValueError_call_result_372794 = invoke(stypy.reporting.localization.Localization(__file__, 256, 22), ValueError_372791, *[str_372792], **kwargs_372793)
            
            ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 256, 16), ValueError_call_result_372794, 'raise parameter', BaseException)
            # SSA join for if statement (line 255)
            module_type_store = module_type_store.join_ssa_context()
            
            
            
            # Getting the type of 'self' (line 259)
            self_372795 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 259, 15), 'self')
            # Obtaining the member 'nnz' of a type (line 259)
            nnz_372796 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 259, 15), self_372795, 'nnz')
            int_372797 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 259, 27), 'int')
            # Applying the binary operator '==' (line 259)
            result_eq_372798 = python_operator(stypy.reporting.localization.Localization(__file__, 259, 15), '==', nnz_372796, int_372797)
            
            # Testing the type of an if condition (line 259)
            if_condition_372799 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 259, 12), result_eq_372798)
            # Assigning a type to the variable 'if_condition_372799' (line 259)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 259, 12), 'if_condition_372799', if_condition_372799)
            # SSA begins for if statement (line 259)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            int_372800 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 260, 23), 'int')
            # Assigning a type to the variable 'stypy_return_type' (line 260)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 260, 16), 'stypy_return_type', int_372800)
            # SSA branch for the else part of an if statement (line 259)
            module_type_store.open_ssa_branch('else')
            
            # Assigning a Call to a Name (line 262):
            
            # Assigning a Call to a Name (line 262):
            
            # Call to type(...): (line 262)
            # Processing the call arguments (line 262)
            int_372804 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 262, 39), 'int')
            # Processing the call keyword arguments (line 262)
            kwargs_372805 = {}
            # Getting the type of 'self' (line 262)
            self_372801 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 262, 23), 'self', False)
            # Obtaining the member 'dtype' of a type (line 262)
            dtype_372802 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 262, 23), self_372801, 'dtype')
            # Obtaining the member 'type' of a type (line 262)
            type_372803 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 262, 23), dtype_372802, 'type')
            # Calling type(args, kwargs) (line 262)
            type_call_result_372806 = invoke(stypy.reporting.localization.Localization(__file__, 262, 23), type_372803, *[int_372804], **kwargs_372805)
            
            # Assigning a type to the variable 'zero' (line 262)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 262, 16), 'zero', type_call_result_372806)
            
            # Assigning a Call to a Name (line 263):
            
            # Assigning a Call to a Name (line 263):
            
            # Call to tocoo(...): (line 263)
            # Processing the call keyword arguments (line 263)
            kwargs_372809 = {}
            # Getting the type of 'self' (line 263)
            self_372807 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 263, 22), 'self', False)
            # Obtaining the member 'tocoo' of a type (line 263)
            tocoo_372808 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 263, 22), self_372807, 'tocoo')
            # Calling tocoo(args, kwargs) (line 263)
            tocoo_call_result_372810 = invoke(stypy.reporting.localization.Localization(__file__, 263, 22), tocoo_372808, *[], **kwargs_372809)
            
            # Assigning a type to the variable 'mat' (line 263)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 263, 16), 'mat', tocoo_call_result_372810)
            
            # Call to sum_duplicates(...): (line 264)
            # Processing the call keyword arguments (line 264)
            kwargs_372813 = {}
            # Getting the type of 'mat' (line 264)
            mat_372811 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 264, 16), 'mat', False)
            # Obtaining the member 'sum_duplicates' of a type (line 264)
            sum_duplicates_372812 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 264, 16), mat_372811, 'sum_duplicates')
            # Calling sum_duplicates(args, kwargs) (line 264)
            sum_duplicates_call_result_372814 = invoke(stypy.reporting.localization.Localization(__file__, 264, 16), sum_duplicates_372812, *[], **kwargs_372813)
            
            
            # Assigning a Call to a Name (line 265):
            
            # Assigning a Call to a Name (line 265):
            
            # Call to op(...): (line 265)
            # Processing the call arguments (line 265)
            # Getting the type of 'mat' (line 265)
            mat_372816 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 265, 24), 'mat', False)
            # Obtaining the member 'data' of a type (line 265)
            data_372817 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 265, 24), mat_372816, 'data')
            # Processing the call keyword arguments (line 265)
            kwargs_372818 = {}
            # Getting the type of 'op' (line 265)
            op_372815 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 265, 21), 'op', False)
            # Calling op(args, kwargs) (line 265)
            op_call_result_372819 = invoke(stypy.reporting.localization.Localization(__file__, 265, 21), op_372815, *[data_372817], **kwargs_372818)
            
            # Assigning a type to the variable 'am' (line 265)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 265, 16), 'am', op_call_result_372819)
            
            # Assigning a Subscript to a Name (line 266):
            
            # Assigning a Subscript to a Name (line 266):
            
            # Obtaining the type of the subscript
            # Getting the type of 'am' (line 266)
            am_372820 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 266, 29), 'am')
            # Getting the type of 'mat' (line 266)
            mat_372821 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 266, 20), 'mat')
            # Obtaining the member 'data' of a type (line 266)
            data_372822 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 266, 20), mat_372821, 'data')
            # Obtaining the member '__getitem__' of a type (line 266)
            getitem___372823 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 266, 20), data_372822, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 266)
            subscript_call_result_372824 = invoke(stypy.reporting.localization.Localization(__file__, 266, 20), getitem___372823, am_372820)
            
            # Assigning a type to the variable 'm' (line 266)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 266, 16), 'm', subscript_call_result_372824)
            
            
            # Call to compare(...): (line 268)
            # Processing the call arguments (line 268)
            # Getting the type of 'm' (line 268)
            m_372826 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 268, 27), 'm', False)
            # Getting the type of 'zero' (line 268)
            zero_372827 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 268, 30), 'zero', False)
            # Processing the call keyword arguments (line 268)
            kwargs_372828 = {}
            # Getting the type of 'compare' (line 268)
            compare_372825 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 268, 19), 'compare', False)
            # Calling compare(args, kwargs) (line 268)
            compare_call_result_372829 = invoke(stypy.reporting.localization.Localization(__file__, 268, 19), compare_372825, *[m_372826, zero_372827], **kwargs_372828)
            
            # Testing the type of an if condition (line 268)
            if_condition_372830 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 268, 16), compare_call_result_372829)
            # Assigning a type to the variable 'if_condition_372830' (line 268)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 268, 16), 'if_condition_372830', if_condition_372830)
            # SSA begins for if statement (line 268)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Obtaining the type of the subscript
            # Getting the type of 'am' (line 269)
            am_372831 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 269, 35), 'am')
            # Getting the type of 'mat' (line 269)
            mat_372832 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 269, 27), 'mat')
            # Obtaining the member 'row' of a type (line 269)
            row_372833 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 269, 27), mat_372832, 'row')
            # Obtaining the member '__getitem__' of a type (line 269)
            getitem___372834 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 269, 27), row_372833, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 269)
            subscript_call_result_372835 = invoke(stypy.reporting.localization.Localization(__file__, 269, 27), getitem___372834, am_372831)
            
            
            # Obtaining the type of the subscript
            int_372836 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 269, 51), 'int')
            # Getting the type of 'mat' (line 269)
            mat_372837 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 269, 41), 'mat')
            # Obtaining the member 'shape' of a type (line 269)
            shape_372838 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 269, 41), mat_372837, 'shape')
            # Obtaining the member '__getitem__' of a type (line 269)
            getitem___372839 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 269, 41), shape_372838, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 269)
            subscript_call_result_372840 = invoke(stypy.reporting.localization.Localization(__file__, 269, 41), getitem___372839, int_372836)
            
            # Applying the binary operator '*' (line 269)
            result_mul_372841 = python_operator(stypy.reporting.localization.Localization(__file__, 269, 27), '*', subscript_call_result_372835, subscript_call_result_372840)
            
            
            # Obtaining the type of the subscript
            # Getting the type of 'am' (line 269)
            am_372842 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 269, 64), 'am')
            # Getting the type of 'mat' (line 269)
            mat_372843 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 269, 56), 'mat')
            # Obtaining the member 'col' of a type (line 269)
            col_372844 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 269, 56), mat_372843, 'col')
            # Obtaining the member '__getitem__' of a type (line 269)
            getitem___372845 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 269, 56), col_372844, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 269)
            subscript_call_result_372846 = invoke(stypy.reporting.localization.Localization(__file__, 269, 56), getitem___372845, am_372842)
            
            # Applying the binary operator '+' (line 269)
            result_add_372847 = python_operator(stypy.reporting.localization.Localization(__file__, 269, 27), '+', result_mul_372841, subscript_call_result_372846)
            
            # Assigning a type to the variable 'stypy_return_type' (line 269)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 269, 20), 'stypy_return_type', result_add_372847)
            # SSA branch for the else part of an if statement (line 268)
            module_type_store.open_ssa_branch('else')
            
            # Assigning a Call to a Name (line 271):
            
            # Assigning a Call to a Name (line 271):
            
            # Call to product(...): (line 271)
            # Processing the call arguments (line 271)
            # Getting the type of 'mat' (line 271)
            mat_372850 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 271, 38), 'mat', False)
            # Obtaining the member 'shape' of a type (line 271)
            shape_372851 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 271, 38), mat_372850, 'shape')
            # Processing the call keyword arguments (line 271)
            kwargs_372852 = {}
            # Getting the type of 'np' (line 271)
            np_372848 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 271, 27), 'np', False)
            # Obtaining the member 'product' of a type (line 271)
            product_372849 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 271, 27), np_372848, 'product')
            # Calling product(args, kwargs) (line 271)
            product_call_result_372853 = invoke(stypy.reporting.localization.Localization(__file__, 271, 27), product_372849, *[shape_372851], **kwargs_372852)
            
            # Assigning a type to the variable 'size' (line 271)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 271, 20), 'size', product_call_result_372853)
            
            
            # Getting the type of 'size' (line 272)
            size_372854 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 272, 23), 'size')
            # Getting the type of 'mat' (line 272)
            mat_372855 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 272, 31), 'mat')
            # Obtaining the member 'nnz' of a type (line 272)
            nnz_372856 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 272, 31), mat_372855, 'nnz')
            # Applying the binary operator '==' (line 272)
            result_eq_372857 = python_operator(stypy.reporting.localization.Localization(__file__, 272, 23), '==', size_372854, nnz_372856)
            
            # Testing the type of an if condition (line 272)
            if_condition_372858 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 272, 20), result_eq_372857)
            # Assigning a type to the variable 'if_condition_372858' (line 272)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 272, 20), 'if_condition_372858', if_condition_372858)
            # SSA begins for if statement (line 272)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            # Getting the type of 'am' (line 273)
            am_372859 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 273, 31), 'am')
            # Assigning a type to the variable 'stypy_return_type' (line 273)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 273, 24), 'stypy_return_type', am_372859)
            # SSA branch for the else part of an if statement (line 272)
            module_type_store.open_ssa_branch('else')
            
            # Assigning a BinOp to a Name (line 275):
            
            # Assigning a BinOp to a Name (line 275):
            # Getting the type of 'mat' (line 275)
            mat_372860 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 275, 30), 'mat')
            # Obtaining the member 'row' of a type (line 275)
            row_372861 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 275, 30), mat_372860, 'row')
            
            # Obtaining the type of the subscript
            int_372862 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 275, 50), 'int')
            # Getting the type of 'mat' (line 275)
            mat_372863 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 275, 40), 'mat')
            # Obtaining the member 'shape' of a type (line 275)
            shape_372864 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 275, 40), mat_372863, 'shape')
            # Obtaining the member '__getitem__' of a type (line 275)
            getitem___372865 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 275, 40), shape_372864, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 275)
            subscript_call_result_372866 = invoke(stypy.reporting.localization.Localization(__file__, 275, 40), getitem___372865, int_372862)
            
            # Applying the binary operator '*' (line 275)
            result_mul_372867 = python_operator(stypy.reporting.localization.Localization(__file__, 275, 30), '*', row_372861, subscript_call_result_372866)
            
            # Getting the type of 'mat' (line 275)
            mat_372868 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 275, 55), 'mat')
            # Obtaining the member 'col' of a type (line 275)
            col_372869 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 275, 55), mat_372868, 'col')
            # Applying the binary operator '+' (line 275)
            result_add_372870 = python_operator(stypy.reporting.localization.Localization(__file__, 275, 30), '+', result_mul_372867, col_372869)
            
            # Assigning a type to the variable 'ind' (line 275)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 275, 24), 'ind', result_add_372870)
            
            # Assigning a Call to a Name (line 276):
            
            # Assigning a Call to a Name (line 276):
            
            # Call to _find_missing_index(...): (line 276)
            # Processing the call arguments (line 276)
            # Getting the type of 'ind' (line 276)
            ind_372872 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 276, 55), 'ind', False)
            # Getting the type of 'size' (line 276)
            size_372873 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 276, 60), 'size', False)
            # Processing the call keyword arguments (line 276)
            kwargs_372874 = {}
            # Getting the type of '_find_missing_index' (line 276)
            _find_missing_index_372871 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 276, 35), '_find_missing_index', False)
            # Calling _find_missing_index(args, kwargs) (line 276)
            _find_missing_index_call_result_372875 = invoke(stypy.reporting.localization.Localization(__file__, 276, 35), _find_missing_index_372871, *[ind_372872, size_372873], **kwargs_372874)
            
            # Assigning a type to the variable 'zero_ind' (line 276)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 276, 24), 'zero_ind', _find_missing_index_call_result_372875)
            
            
            # Getting the type of 'm' (line 277)
            m_372876 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 277, 27), 'm')
            # Getting the type of 'zero' (line 277)
            zero_372877 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 277, 32), 'zero')
            # Applying the binary operator '==' (line 277)
            result_eq_372878 = python_operator(stypy.reporting.localization.Localization(__file__, 277, 27), '==', m_372876, zero_372877)
            
            # Testing the type of an if condition (line 277)
            if_condition_372879 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 277, 24), result_eq_372878)
            # Assigning a type to the variable 'if_condition_372879' (line 277)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 277, 24), 'if_condition_372879', if_condition_372879)
            # SSA begins for if statement (line 277)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to min(...): (line 278)
            # Processing the call arguments (line 278)
            # Getting the type of 'zero_ind' (line 278)
            zero_ind_372881 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 278, 39), 'zero_ind', False)
            # Getting the type of 'am' (line 278)
            am_372882 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 278, 49), 'am', False)
            # Processing the call keyword arguments (line 278)
            kwargs_372883 = {}
            # Getting the type of 'min' (line 278)
            min_372880 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 278, 35), 'min', False)
            # Calling min(args, kwargs) (line 278)
            min_call_result_372884 = invoke(stypy.reporting.localization.Localization(__file__, 278, 35), min_372880, *[zero_ind_372881, am_372882], **kwargs_372883)
            
            # Assigning a type to the variable 'stypy_return_type' (line 278)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 278, 28), 'stypy_return_type', min_call_result_372884)
            # SSA branch for the else part of an if statement (line 277)
            module_type_store.open_ssa_branch('else')
            # Getting the type of 'zero_ind' (line 280)
            zero_ind_372885 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 280, 35), 'zero_ind')
            # Assigning a type to the variable 'stypy_return_type' (line 280)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 280, 28), 'stypy_return_type', zero_ind_372885)
            # SSA join for if statement (line 277)
            module_type_store = module_type_store.join_ssa_context()
            
            # SSA join for if statement (line 272)
            module_type_store = module_type_store.join_ssa_context()
            
            # SSA join for if statement (line 268)
            module_type_store = module_type_store.join_ssa_context()
            
            # SSA join for if statement (line 259)
            module_type_store = module_type_store.join_ssa_context()
            

            if more_types_in_union_372785:
                # SSA join for if statement (line 254)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Call to _arg_min_or_max_axis(...): (line 282)
        # Processing the call arguments (line 282)
        # Getting the type of 'axis' (line 282)
        axis_372888 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 282, 41), 'axis', False)
        # Getting the type of 'op' (line 282)
        op_372889 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 282, 47), 'op', False)
        # Getting the type of 'compare' (line 282)
        compare_372890 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 282, 51), 'compare', False)
        # Processing the call keyword arguments (line 282)
        kwargs_372891 = {}
        # Getting the type of 'self' (line 282)
        self_372886 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 282, 15), 'self', False)
        # Obtaining the member '_arg_min_or_max_axis' of a type (line 282)
        _arg_min_or_max_axis_372887 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 282, 15), self_372886, '_arg_min_or_max_axis')
        # Calling _arg_min_or_max_axis(args, kwargs) (line 282)
        _arg_min_or_max_axis_call_result_372892 = invoke(stypy.reporting.localization.Localization(__file__, 282, 15), _arg_min_or_max_axis_372887, *[axis_372888, op_372889, compare_372890], **kwargs_372891)
        
        # Assigning a type to the variable 'stypy_return_type' (line 282)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 282, 8), 'stypy_return_type', _arg_min_or_max_axis_call_result_372892)
        
        # ################# End of '_arg_min_or_max(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_arg_min_or_max' in the type store
        # Getting the type of 'stypy_return_type' (line 247)
        stypy_return_type_372893 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 247, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_372893)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_arg_min_or_max'
        return stypy_return_type_372893


    @norecursion
    def max(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 284)
        None_372894 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 284, 23), 'None')
        # Getting the type of 'None' (line 284)
        None_372895 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 284, 33), 'None')
        defaults = [None_372894, None_372895]
        # Create a new context for function 'max'
        module_type_store = module_type_store.open_function_context('max', 284, 4, False)
        # Assigning a type to the variable 'self' (line 285)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 285, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        _minmax_mixin.max.__dict__.__setitem__('stypy_localization', localization)
        _minmax_mixin.max.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        _minmax_mixin.max.__dict__.__setitem__('stypy_type_store', module_type_store)
        _minmax_mixin.max.__dict__.__setitem__('stypy_function_name', '_minmax_mixin.max')
        _minmax_mixin.max.__dict__.__setitem__('stypy_param_names_list', ['axis', 'out'])
        _minmax_mixin.max.__dict__.__setitem__('stypy_varargs_param_name', None)
        _minmax_mixin.max.__dict__.__setitem__('stypy_kwargs_param_name', None)
        _minmax_mixin.max.__dict__.__setitem__('stypy_call_defaults', defaults)
        _minmax_mixin.max.__dict__.__setitem__('stypy_call_varargs', varargs)
        _minmax_mixin.max.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        _minmax_mixin.max.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, '_minmax_mixin.max', ['axis', 'out'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'max', localization, ['axis', 'out'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'max(...)' code ##################

        str_372896 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 313, (-1)), 'str', "\n        Return the maximum of the matrix or maximum along an axis.\n        This takes all elements into account, not just the non-zero ones.\n\n        Parameters\n        ----------\n        axis : {-2, -1, 0, 1, None} optional\n            Axis along which the sum is computed. The default is to\n            compute the maximum over all the matrix elements, returning\n            a scalar (i.e. `axis` = `None`).\n\n        out : None, optional\n            This argument is in the signature *solely* for NumPy\n            compatibility reasons. Do not pass in anything except\n            for the default value, as this argument is not used.\n\n        Returns\n        -------\n        amax : coo_matrix or scalar\n            Maximum of `a`. If `axis` is None, the result is a scalar value.\n            If `axis` is given, the result is a sparse.coo_matrix of dimension\n            ``a.ndim - 1``.\n\n        See Also\n        --------\n        min : The minimum value of a sparse matrix along a given axis.\n        np.matrix.max : NumPy's implementation of 'max' for matrices\n\n        ")
        
        # Call to _min_or_max(...): (line 314)
        # Processing the call arguments (line 314)
        # Getting the type of 'axis' (line 314)
        axis_372899 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 314, 32), 'axis', False)
        # Getting the type of 'out' (line 314)
        out_372900 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 314, 38), 'out', False)
        # Getting the type of 'np' (line 314)
        np_372901 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 314, 43), 'np', False)
        # Obtaining the member 'maximum' of a type (line 314)
        maximum_372902 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 314, 43), np_372901, 'maximum')
        # Processing the call keyword arguments (line 314)
        kwargs_372903 = {}
        # Getting the type of 'self' (line 314)
        self_372897 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 314, 15), 'self', False)
        # Obtaining the member '_min_or_max' of a type (line 314)
        _min_or_max_372898 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 314, 15), self_372897, '_min_or_max')
        # Calling _min_or_max(args, kwargs) (line 314)
        _min_or_max_call_result_372904 = invoke(stypy.reporting.localization.Localization(__file__, 314, 15), _min_or_max_372898, *[axis_372899, out_372900, maximum_372902], **kwargs_372903)
        
        # Assigning a type to the variable 'stypy_return_type' (line 314)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 314, 8), 'stypy_return_type', _min_or_max_call_result_372904)
        
        # ################# End of 'max(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'max' in the type store
        # Getting the type of 'stypy_return_type' (line 284)
        stypy_return_type_372905 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 284, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_372905)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'max'
        return stypy_return_type_372905


    @norecursion
    def min(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 316)
        None_372906 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 316, 23), 'None')
        # Getting the type of 'None' (line 316)
        None_372907 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 316, 33), 'None')
        defaults = [None_372906, None_372907]
        # Create a new context for function 'min'
        module_type_store = module_type_store.open_function_context('min', 316, 4, False)
        # Assigning a type to the variable 'self' (line 317)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 317, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        _minmax_mixin.min.__dict__.__setitem__('stypy_localization', localization)
        _minmax_mixin.min.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        _minmax_mixin.min.__dict__.__setitem__('stypy_type_store', module_type_store)
        _minmax_mixin.min.__dict__.__setitem__('stypy_function_name', '_minmax_mixin.min')
        _minmax_mixin.min.__dict__.__setitem__('stypy_param_names_list', ['axis', 'out'])
        _minmax_mixin.min.__dict__.__setitem__('stypy_varargs_param_name', None)
        _minmax_mixin.min.__dict__.__setitem__('stypy_kwargs_param_name', None)
        _minmax_mixin.min.__dict__.__setitem__('stypy_call_defaults', defaults)
        _minmax_mixin.min.__dict__.__setitem__('stypy_call_varargs', varargs)
        _minmax_mixin.min.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        _minmax_mixin.min.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, '_minmax_mixin.min', ['axis', 'out'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'min', localization, ['axis', 'out'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'min(...)' code ##################

        str_372908 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 345, (-1)), 'str', "\n        Return the minimum of the matrix or maximum along an axis.\n        This takes all elements into account, not just the non-zero ones.\n\n        Parameters\n        ----------\n        axis : {-2, -1, 0, 1, None} optional\n            Axis along which the sum is computed. The default is to\n            compute the minimum over all the matrix elements, returning\n            a scalar (i.e. `axis` = `None`).\n\n        out : None, optional\n            This argument is in the signature *solely* for NumPy\n            compatibility reasons. Do not pass in anything except for\n            the default value, as this argument is not used.\n\n        Returns\n        -------\n        amin : coo_matrix or scalar\n            Minimum of `a`. If `axis` is None, the result is a scalar value.\n            If `axis` is given, the result is a sparse.coo_matrix of dimension\n            ``a.ndim - 1``.\n\n        See Also\n        --------\n        max : The maximum value of a sparse matrix along a given axis.\n        np.matrix.min : NumPy's implementation of 'min' for matrices\n\n        ")
        
        # Call to _min_or_max(...): (line 346)
        # Processing the call arguments (line 346)
        # Getting the type of 'axis' (line 346)
        axis_372911 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 346, 32), 'axis', False)
        # Getting the type of 'out' (line 346)
        out_372912 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 346, 38), 'out', False)
        # Getting the type of 'np' (line 346)
        np_372913 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 346, 43), 'np', False)
        # Obtaining the member 'minimum' of a type (line 346)
        minimum_372914 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 346, 43), np_372913, 'minimum')
        # Processing the call keyword arguments (line 346)
        kwargs_372915 = {}
        # Getting the type of 'self' (line 346)
        self_372909 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 346, 15), 'self', False)
        # Obtaining the member '_min_or_max' of a type (line 346)
        _min_or_max_372910 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 346, 15), self_372909, '_min_or_max')
        # Calling _min_or_max(args, kwargs) (line 346)
        _min_or_max_call_result_372916 = invoke(stypy.reporting.localization.Localization(__file__, 346, 15), _min_or_max_372910, *[axis_372911, out_372912, minimum_372914], **kwargs_372915)
        
        # Assigning a type to the variable 'stypy_return_type' (line 346)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 346, 8), 'stypy_return_type', _min_or_max_call_result_372916)
        
        # ################# End of 'min(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'min' in the type store
        # Getting the type of 'stypy_return_type' (line 316)
        stypy_return_type_372917 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 316, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_372917)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'min'
        return stypy_return_type_372917


    @norecursion
    def argmax(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 348)
        None_372918 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 348, 26), 'None')
        # Getting the type of 'None' (line 348)
        None_372919 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 348, 36), 'None')
        defaults = [None_372918, None_372919]
        # Create a new context for function 'argmax'
        module_type_store = module_type_store.open_function_context('argmax', 348, 4, False)
        # Assigning a type to the variable 'self' (line 349)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 349, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        _minmax_mixin.argmax.__dict__.__setitem__('stypy_localization', localization)
        _minmax_mixin.argmax.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        _minmax_mixin.argmax.__dict__.__setitem__('stypy_type_store', module_type_store)
        _minmax_mixin.argmax.__dict__.__setitem__('stypy_function_name', '_minmax_mixin.argmax')
        _minmax_mixin.argmax.__dict__.__setitem__('stypy_param_names_list', ['axis', 'out'])
        _minmax_mixin.argmax.__dict__.__setitem__('stypy_varargs_param_name', None)
        _minmax_mixin.argmax.__dict__.__setitem__('stypy_kwargs_param_name', None)
        _minmax_mixin.argmax.__dict__.__setitem__('stypy_call_defaults', defaults)
        _minmax_mixin.argmax.__dict__.__setitem__('stypy_call_varargs', varargs)
        _minmax_mixin.argmax.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        _minmax_mixin.argmax.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, '_minmax_mixin.argmax', ['axis', 'out'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'argmax', localization, ['axis', 'out'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'argmax(...)' code ##################

        str_372920 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 368, (-1)), 'str', 'Return indices of maximum elements along an axis.\n\n        Implicit zero elements are also taken into account. If there are\n        several maximum values, the index of the first occurrence is returned.\n\n        Parameters\n        ----------\n        axis : {-2, -1, 0, 1, None}, optional\n            Axis along which the argmax is computed. If None (default), index\n            of the maximum element in the flatten data is returned.\n        out : None, optional\n            This argument is in the signature *solely* for NumPy\n            compatibility reasons. Do not pass in anything except for\n            the default value, as this argument is not used.\n\n        Returns\n        -------\n        ind : np.matrix or int\n            Indices of maximum elements. If matrix, its size along `axis` is 1.\n        ')
        
        # Call to _arg_min_or_max(...): (line 369)
        # Processing the call arguments (line 369)
        # Getting the type of 'axis' (line 369)
        axis_372923 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 369, 36), 'axis', False)
        # Getting the type of 'out' (line 369)
        out_372924 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 369, 42), 'out', False)
        # Getting the type of 'np' (line 369)
        np_372925 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 369, 47), 'np', False)
        # Obtaining the member 'argmax' of a type (line 369)
        argmax_372926 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 369, 47), np_372925, 'argmax')
        # Getting the type of 'np' (line 369)
        np_372927 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 369, 58), 'np', False)
        # Obtaining the member 'greater' of a type (line 369)
        greater_372928 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 369, 58), np_372927, 'greater')
        # Processing the call keyword arguments (line 369)
        kwargs_372929 = {}
        # Getting the type of 'self' (line 369)
        self_372921 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 369, 15), 'self', False)
        # Obtaining the member '_arg_min_or_max' of a type (line 369)
        _arg_min_or_max_372922 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 369, 15), self_372921, '_arg_min_or_max')
        # Calling _arg_min_or_max(args, kwargs) (line 369)
        _arg_min_or_max_call_result_372930 = invoke(stypy.reporting.localization.Localization(__file__, 369, 15), _arg_min_or_max_372922, *[axis_372923, out_372924, argmax_372926, greater_372928], **kwargs_372929)
        
        # Assigning a type to the variable 'stypy_return_type' (line 369)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 369, 8), 'stypy_return_type', _arg_min_or_max_call_result_372930)
        
        # ################# End of 'argmax(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'argmax' in the type store
        # Getting the type of 'stypy_return_type' (line 348)
        stypy_return_type_372931 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 348, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_372931)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'argmax'
        return stypy_return_type_372931


    @norecursion
    def argmin(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 371)
        None_372932 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 371, 26), 'None')
        # Getting the type of 'None' (line 371)
        None_372933 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 371, 36), 'None')
        defaults = [None_372932, None_372933]
        # Create a new context for function 'argmin'
        module_type_store = module_type_store.open_function_context('argmin', 371, 4, False)
        # Assigning a type to the variable 'self' (line 372)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 372, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        _minmax_mixin.argmin.__dict__.__setitem__('stypy_localization', localization)
        _minmax_mixin.argmin.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        _minmax_mixin.argmin.__dict__.__setitem__('stypy_type_store', module_type_store)
        _minmax_mixin.argmin.__dict__.__setitem__('stypy_function_name', '_minmax_mixin.argmin')
        _minmax_mixin.argmin.__dict__.__setitem__('stypy_param_names_list', ['axis', 'out'])
        _minmax_mixin.argmin.__dict__.__setitem__('stypy_varargs_param_name', None)
        _minmax_mixin.argmin.__dict__.__setitem__('stypy_kwargs_param_name', None)
        _minmax_mixin.argmin.__dict__.__setitem__('stypy_call_defaults', defaults)
        _minmax_mixin.argmin.__dict__.__setitem__('stypy_call_varargs', varargs)
        _minmax_mixin.argmin.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        _minmax_mixin.argmin.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, '_minmax_mixin.argmin', ['axis', 'out'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'argmin', localization, ['axis', 'out'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'argmin(...)' code ##################

        str_372934 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 391, (-1)), 'str', 'Return indices of minimum elements along an axis.\n\n        Implicit zero elements are also taken into account. If there are\n        several minimum values, the index of the first occurrence is returned.\n\n        Parameters\n        ----------\n        axis : {-2, -1, 0, 1, None}, optional\n            Axis along which the argmin is computed. If None (default), index\n            of the minimum element in the flatten data is returned.\n        out : None, optional\n            This argument is in the signature *solely* for NumPy\n            compatibility reasons. Do not pass in anything except for\n            the default value, as this argument is not used.\n\n        Returns\n        -------\n         ind : np.matrix or int\n            Indices of minimum elements. If matrix, its size along `axis` is 1.\n        ')
        
        # Call to _arg_min_or_max(...): (line 392)
        # Processing the call arguments (line 392)
        # Getting the type of 'axis' (line 392)
        axis_372937 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 392, 36), 'axis', False)
        # Getting the type of 'out' (line 392)
        out_372938 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 392, 42), 'out', False)
        # Getting the type of 'np' (line 392)
        np_372939 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 392, 47), 'np', False)
        # Obtaining the member 'argmin' of a type (line 392)
        argmin_372940 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 392, 47), np_372939, 'argmin')
        # Getting the type of 'np' (line 392)
        np_372941 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 392, 58), 'np', False)
        # Obtaining the member 'less' of a type (line 392)
        less_372942 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 392, 58), np_372941, 'less')
        # Processing the call keyword arguments (line 392)
        kwargs_372943 = {}
        # Getting the type of 'self' (line 392)
        self_372935 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 392, 15), 'self', False)
        # Obtaining the member '_arg_min_or_max' of a type (line 392)
        _arg_min_or_max_372936 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 392, 15), self_372935, '_arg_min_or_max')
        # Calling _arg_min_or_max(args, kwargs) (line 392)
        _arg_min_or_max_call_result_372944 = invoke(stypy.reporting.localization.Localization(__file__, 392, 15), _arg_min_or_max_372936, *[axis_372937, out_372938, argmin_372940, less_372942], **kwargs_372943)
        
        # Assigning a type to the variable 'stypy_return_type' (line 392)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 392, 8), 'stypy_return_type', _arg_min_or_max_call_result_372944)
        
        # ################# End of 'argmin(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'argmin' in the type store
        # Getting the type of 'stypy_return_type' (line 371)
        stypy_return_type_372945 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 371, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_372945)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'argmin'
        return stypy_return_type_372945


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 152, 0, False)
        # Assigning a type to the variable 'self' (line 153)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 153, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, '_minmax_mixin.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable '_minmax_mixin' (line 152)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 152, 0), '_minmax_mixin', _minmax_mixin)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
