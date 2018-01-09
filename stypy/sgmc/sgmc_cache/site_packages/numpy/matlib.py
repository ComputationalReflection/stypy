
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: from __future__ import division, absolute_import, print_function
2: 
3: import numpy as np
4: from numpy.matrixlib.defmatrix import matrix, asmatrix
5: # need * as we're copying the numpy namespace
6: from numpy import *
7: 
8: __version__ = np.__version__
9: 
10: __all__ = np.__all__[:] # copy numpy namespace
11: __all__ += ['rand', 'randn', 'repmat']
12: 
13: def empty(shape, dtype=None, order='C'):
14:     '''Return a new matrix of given shape and type, without initializing entries.
15: 
16:     Parameters
17:     ----------
18:     shape : int or tuple of int
19:         Shape of the empty matrix.
20:     dtype : data-type, optional
21:         Desired output data-type.
22:     order : {'C', 'F'}, optional
23:         Whether to store multi-dimensional data in row-major
24:         (C-style) or column-major (Fortran-style) order in
25:         memory.
26: 
27:     See Also
28:     --------
29:     empty_like, zeros
30: 
31:     Notes
32:     -----
33:     `empty`, unlike `zeros`, does not set the matrix values to zero,
34:     and may therefore be marginally faster.  On the other hand, it requires
35:     the user to manually set all the values in the array, and should be
36:     used with caution.
37: 
38:     Examples
39:     --------
40:     >>> import numpy.matlib
41:     >>> np.matlib.empty((2, 2))    # filled with random data
42:     matrix([[  6.76425276e-320,   9.79033856e-307],
43:             [  7.39337286e-309,   3.22135945e-309]])        #random
44:     >>> np.matlib.empty((2, 2), dtype=int)
45:     matrix([[ 6600475,        0],
46:             [ 6586976, 22740995]])                          #random
47: 
48:     '''
49:     return ndarray.__new__(matrix, shape, dtype, order=order)
50: 
51: def ones(shape, dtype=None, order='C'):
52:     '''
53:     Matrix of ones.
54: 
55:     Return a matrix of given shape and type, filled with ones.
56: 
57:     Parameters
58:     ----------
59:     shape : {sequence of ints, int}
60:         Shape of the matrix
61:     dtype : data-type, optional
62:         The desired data-type for the matrix, default is np.float64.
63:     order : {'C', 'F'}, optional
64:         Whether to store matrix in C- or Fortran-contiguous order,
65:         default is 'C'.
66: 
67:     Returns
68:     -------
69:     out : matrix
70:         Matrix of ones of given shape, dtype, and order.
71: 
72:     See Also
73:     --------
74:     ones : Array of ones.
75:     matlib.zeros : Zero matrix.
76: 
77:     Notes
78:     -----
79:     If `shape` has length one i.e. ``(N,)``, or is a scalar ``N``,
80:     `out` becomes a single row matrix of shape ``(1,N)``.
81: 
82:     Examples
83:     --------
84:     >>> np.matlib.ones((2,3))
85:     matrix([[ 1.,  1.,  1.],
86:             [ 1.,  1.,  1.]])
87: 
88:     >>> np.matlib.ones(2)
89:     matrix([[ 1.,  1.]])
90: 
91:     '''
92:     a = ndarray.__new__(matrix, shape, dtype, order=order)
93:     a.fill(1)
94:     return a
95: 
96: def zeros(shape, dtype=None, order='C'):
97:     '''
98:     Return a matrix of given shape and type, filled with zeros.
99: 
100:     Parameters
101:     ----------
102:     shape : int or sequence of ints
103:         Shape of the matrix
104:     dtype : data-type, optional
105:         The desired data-type for the matrix, default is float.
106:     order : {'C', 'F'}, optional
107:         Whether to store the result in C- or Fortran-contiguous order,
108:         default is 'C'.
109: 
110:     Returns
111:     -------
112:     out : matrix
113:         Zero matrix of given shape, dtype, and order.
114: 
115:     See Also
116:     --------
117:     numpy.zeros : Equivalent array function.
118:     matlib.ones : Return a matrix of ones.
119: 
120:     Notes
121:     -----
122:     If `shape` has length one i.e. ``(N,)``, or is a scalar ``N``,
123:     `out` becomes a single row matrix of shape ``(1,N)``.
124: 
125:     Examples
126:     --------
127:     >>> import numpy.matlib
128:     >>> np.matlib.zeros((2, 3))
129:     matrix([[ 0.,  0.,  0.],
130:             [ 0.,  0.,  0.]])
131: 
132:     >>> np.matlib.zeros(2)
133:     matrix([[ 0.,  0.]])
134: 
135:     '''
136:     a = ndarray.__new__(matrix, shape, dtype, order=order)
137:     a.fill(0)
138:     return a
139: 
140: def identity(n,dtype=None):
141:     '''
142:     Returns the square identity matrix of given size.
143: 
144:     Parameters
145:     ----------
146:     n : int
147:         Size of the returned identity matrix.
148:     dtype : data-type, optional
149:         Data-type of the output. Defaults to ``float``.
150: 
151:     Returns
152:     -------
153:     out : matrix
154:         `n` x `n` matrix with its main diagonal set to one,
155:         and all other elements zero.
156: 
157:     See Also
158:     --------
159:     numpy.identity : Equivalent array function.
160:     matlib.eye : More general matrix identity function.
161: 
162:     Examples
163:     --------
164:     >>> import numpy.matlib
165:     >>> np.matlib.identity(3, dtype=int)
166:     matrix([[1, 0, 0],
167:             [0, 1, 0],
168:             [0, 0, 1]])
169: 
170:     '''
171:     a = array([1]+n*[0], dtype=dtype)
172:     b = empty((n, n), dtype=dtype)
173:     b.flat = a
174:     return b
175: 
176: def eye(n,M=None, k=0, dtype=float):
177:     '''
178:     Return a matrix with ones on the diagonal and zeros elsewhere.
179: 
180:     Parameters
181:     ----------
182:     n : int
183:         Number of rows in the output.
184:     M : int, optional
185:         Number of columns in the output, defaults to `n`.
186:     k : int, optional
187:         Index of the diagonal: 0 refers to the main diagonal,
188:         a positive value refers to an upper diagonal,
189:         and a negative value to a lower diagonal.
190:     dtype : dtype, optional
191:         Data-type of the returned matrix.
192: 
193:     Returns
194:     -------
195:     I : matrix
196:         A `n` x `M` matrix where all elements are equal to zero,
197:         except for the `k`-th diagonal, whose values are equal to one.
198: 
199:     See Also
200:     --------
201:     numpy.eye : Equivalent array function.
202:     identity : Square identity matrix.
203: 
204:     Examples
205:     --------
206:     >>> import numpy.matlib
207:     >>> np.matlib.eye(3, k=1, dtype=float)
208:     matrix([[ 0.,  1.,  0.],
209:             [ 0.,  0.,  1.],
210:             [ 0.,  0.,  0.]])
211: 
212:     '''
213:     return asmatrix(np.eye(n, M, k, dtype))
214: 
215: def rand(*args):
216:     '''
217:     Return a matrix of random values with given shape.
218: 
219:     Create a matrix of the given shape and propagate it with
220:     random samples from a uniform distribution over ``[0, 1)``.
221: 
222:     Parameters
223:     ----------
224:     \\*args : Arguments
225:         Shape of the output.
226:         If given as N integers, each integer specifies the size of one
227:         dimension.
228:         If given as a tuple, this tuple gives the complete shape.
229: 
230:     Returns
231:     -------
232:     out : ndarray
233:         The matrix of random values with shape given by `\\*args`.
234: 
235:     See Also
236:     --------
237:     randn, numpy.random.rand
238: 
239:     Examples
240:     --------
241:     >>> import numpy.matlib
242:     >>> np.matlib.rand(2, 3)
243:     matrix([[ 0.68340382,  0.67926887,  0.83271405],
244:             [ 0.00793551,  0.20468222,  0.95253525]])       #random
245:     >>> np.matlib.rand((2, 3))
246:     matrix([[ 0.84682055,  0.73626594,  0.11308016],
247:             [ 0.85429008,  0.3294825 ,  0.89139555]])       #random
248: 
249:     If the first argument is a tuple, other arguments are ignored:
250: 
251:     >>> np.matlib.rand((2, 3), 4)
252:     matrix([[ 0.46898646,  0.15163588,  0.95188261],
253:             [ 0.59208621,  0.09561818,  0.00583606]])       #random
254: 
255:     '''
256:     if isinstance(args[0], tuple):
257:         args = args[0]
258:     return asmatrix(np.random.rand(*args))
259: 
260: def randn(*args):
261:     '''
262:     Return a random matrix with data from the "standard normal" distribution.
263: 
264:     `randn` generates a matrix filled with random floats sampled from a
265:     univariate "normal" (Gaussian) distribution of mean 0 and variance 1.
266: 
267:     Parameters
268:     ----------
269:     \\*args : Arguments
270:         Shape of the output.
271:         If given as N integers, each integer specifies the size of one
272:         dimension. If given as a tuple, this tuple gives the complete shape.
273: 
274:     Returns
275:     -------
276:     Z : matrix of floats
277:         A matrix of floating-point samples drawn from the standard normal
278:         distribution.
279: 
280:     See Also
281:     --------
282:     rand, random.randn
283: 
284:     Notes
285:     -----
286:     For random samples from :math:`N(\\mu, \\sigma^2)`, use:
287: 
288:     ``sigma * np.matlib.randn(...) + mu``
289: 
290:     Examples
291:     --------
292:     >>> import numpy.matlib
293:     >>> np.matlib.randn(1)
294:     matrix([[-0.09542833]])                                 #random
295:     >>> np.matlib.randn(1, 2, 3)
296:     matrix([[ 0.16198284,  0.0194571 ,  0.18312985],
297:             [-0.7509172 ,  1.61055   ,  0.45298599]])       #random
298: 
299:     Two-by-four matrix of samples from :math:`N(3, 6.25)`:
300: 
301:     >>> 2.5 * np.matlib.randn((2, 4)) + 3
302:     matrix([[ 4.74085004,  8.89381862,  4.09042411,  4.83721922],
303:             [ 7.52373709,  5.07933944, -2.64043543,  0.45610557]])  #random
304: 
305:     '''
306:     if isinstance(args[0], tuple):
307:         args = args[0]
308:     return asmatrix(np.random.randn(*args))
309: 
310: def repmat(a, m, n):
311:     '''
312:     Repeat a 0-D to 2-D array or matrix MxN times.
313: 
314:     Parameters
315:     ----------
316:     a : array_like
317:         The array or matrix to be repeated.
318:     m, n : int
319:         The number of times `a` is repeated along the first and second axes.
320: 
321:     Returns
322:     -------
323:     out : ndarray
324:         The result of repeating `a`.
325: 
326:     Examples
327:     --------
328:     >>> import numpy.matlib
329:     >>> a0 = np.array(1)
330:     >>> np.matlib.repmat(a0, 2, 3)
331:     array([[1, 1, 1],
332:            [1, 1, 1]])
333: 
334:     >>> a1 = np.arange(4)
335:     >>> np.matlib.repmat(a1, 2, 2)
336:     array([[0, 1, 2, 3, 0, 1, 2, 3],
337:            [0, 1, 2, 3, 0, 1, 2, 3]])
338: 
339:     >>> a2 = np.asmatrix(np.arange(6).reshape(2, 3))
340:     >>> np.matlib.repmat(a2, 2, 3)
341:     matrix([[0, 1, 2, 0, 1, 2, 0, 1, 2],
342:             [3, 4, 5, 3, 4, 5, 3, 4, 5],
343:             [0, 1, 2, 0, 1, 2, 0, 1, 2],
344:             [3, 4, 5, 3, 4, 5, 3, 4, 5]])
345: 
346:     '''
347:     a = asanyarray(a)
348:     ndim = a.ndim
349:     if ndim == 0:
350:         origrows, origcols = (1, 1)
351:     elif ndim == 1:
352:         origrows, origcols = (1, a.shape[0])
353:     else:
354:         origrows, origcols = a.shape
355:     rows = origrows * m
356:     cols = origcols * n
357:     c = a.reshape(1, a.size).repeat(m, 0).reshape(rows, origcols).repeat(n, 0)
358:     return c.reshape(rows, cols)
359: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 3, 0))

# 'import numpy' statement (line 3)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/')
import_24063 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'numpy')

if (type(import_24063) is not StypyTypeError):

    if (import_24063 != 'pyd_module'):
        __import__(import_24063)
        sys_modules_24064 = sys.modules[import_24063]
        import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'np', sys_modules_24064.module_type_store, module_type_store)
    else:
        import numpy as np

        import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'np', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 3)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 0), 'numpy', import_24063)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 4, 0))

# 'from numpy.matrixlib.defmatrix import matrix, asmatrix' statement (line 4)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/')
import_24065 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'numpy.matrixlib.defmatrix')

if (type(import_24065) is not StypyTypeError):

    if (import_24065 != 'pyd_module'):
        __import__(import_24065)
        sys_modules_24066 = sys.modules[import_24065]
        import_from_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'numpy.matrixlib.defmatrix', sys_modules_24066.module_type_store, module_type_store, ['matrix', 'asmatrix'])
        nest_module(stypy.reporting.localization.Localization(__file__, 4, 0), __file__, sys_modules_24066, sys_modules_24066.module_type_store, module_type_store)
    else:
        from numpy.matrixlib.defmatrix import matrix, asmatrix

        import_from_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'numpy.matrixlib.defmatrix', None, module_type_store, ['matrix', 'asmatrix'], [matrix, asmatrix])

else:
    # Assigning a type to the variable 'numpy.matrixlib.defmatrix' (line 4)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 4, 0), 'numpy.matrixlib.defmatrix', import_24065)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 6, 0))

# 'from numpy import ' statement (line 6)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/')
import_24067 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'numpy')

if (type(import_24067) is not StypyTypeError):

    if (import_24067 != 'pyd_module'):
        __import__(import_24067)
        sys_modules_24068 = sys.modules[import_24067]
        import_from_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'numpy', sys_modules_24068.module_type_store, module_type_store, ['*'])
        nest_module(stypy.reporting.localization.Localization(__file__, 6, 0), __file__, sys_modules_24068, sys_modules_24068.module_type_store, module_type_store)
    else:
        from numpy import *

        import_from_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'numpy', None, module_type_store, ['*'], None)

else:
    # Assigning a type to the variable 'numpy' (line 6)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), 'numpy', import_24067)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/')


# Assigning a Attribute to a Name (line 8):

# Assigning a Attribute to a Name (line 8):
# Getting the type of 'np' (line 8)
np_24069 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 8, 14), 'np')
# Obtaining the member '__version__' of a type (line 8)
version___24070 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 8, 14), np_24069, '__version__')
# Assigning a type to the variable '__version__' (line 8)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 0), '__version__', version___24070)

# Assigning a Subscript to a Name (line 10):

# Assigning a Subscript to a Name (line 10):

# Obtaining the type of the subscript
slice_24071 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 10, 10), None, None, None)
# Getting the type of 'np' (line 10)
np_24072 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 10, 10), 'np')
# Obtaining the member '__all__' of a type (line 10)
all___24073 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 10, 10), np_24072, '__all__')
# Obtaining the member '__getitem__' of a type (line 10)
getitem___24074 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 10, 10), all___24073, '__getitem__')
# Calling the subscript (__getitem__) to obtain the elements type (line 10)
subscript_call_result_24075 = invoke(stypy.reporting.localization.Localization(__file__, 10, 10), getitem___24074, slice_24071)

# Assigning a type to the variable '__all__' (line 10)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 0), '__all__', subscript_call_result_24075)

# Getting the type of '__all__' (line 11)
all___24076 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 0), '__all__')

# Obtaining an instance of the builtin type 'list' (line 11)
list_24077 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 11, 11), 'list')
# Adding type elements to the builtin type 'list' instance (line 11)
# Adding element type (line 11)
str_24078 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 11, 12), 'str', 'rand')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 11, 11), list_24077, str_24078)
# Adding element type (line 11)
str_24079 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 11, 20), 'str', 'randn')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 11, 11), list_24077, str_24079)
# Adding element type (line 11)
str_24080 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 11, 29), 'str', 'repmat')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 11, 11), list_24077, str_24080)

# Applying the binary operator '+=' (line 11)
result_iadd_24081 = python_operator(stypy.reporting.localization.Localization(__file__, 11, 0), '+=', all___24076, list_24077)
# Assigning a type to the variable '__all__' (line 11)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 0), '__all__', result_iadd_24081)


@norecursion
def empty(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'None' (line 13)
    None_24082 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 23), 'None')
    str_24083 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 35), 'str', 'C')
    defaults = [None_24082, str_24083]
    # Create a new context for function 'empty'
    module_type_store = module_type_store.open_function_context('empty', 13, 0, False)
    
    # Passed parameters checking function
    empty.stypy_localization = localization
    empty.stypy_type_of_self = None
    empty.stypy_type_store = module_type_store
    empty.stypy_function_name = 'empty'
    empty.stypy_param_names_list = ['shape', 'dtype', 'order']
    empty.stypy_varargs_param_name = None
    empty.stypy_kwargs_param_name = None
    empty.stypy_call_defaults = defaults
    empty.stypy_call_varargs = varargs
    empty.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'empty', ['shape', 'dtype', 'order'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'empty', localization, ['shape', 'dtype', 'order'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'empty(...)' code ##################

    str_24084 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 48, (-1)), 'str', "Return a new matrix of given shape and type, without initializing entries.\n\n    Parameters\n    ----------\n    shape : int or tuple of int\n        Shape of the empty matrix.\n    dtype : data-type, optional\n        Desired output data-type.\n    order : {'C', 'F'}, optional\n        Whether to store multi-dimensional data in row-major\n        (C-style) or column-major (Fortran-style) order in\n        memory.\n\n    See Also\n    --------\n    empty_like, zeros\n\n    Notes\n    -----\n    `empty`, unlike `zeros`, does not set the matrix values to zero,\n    and may therefore be marginally faster.  On the other hand, it requires\n    the user to manually set all the values in the array, and should be\n    used with caution.\n\n    Examples\n    --------\n    >>> import numpy.matlib\n    >>> np.matlib.empty((2, 2))    # filled with random data\n    matrix([[  6.76425276e-320,   9.79033856e-307],\n            [  7.39337286e-309,   3.22135945e-309]])        #random\n    >>> np.matlib.empty((2, 2), dtype=int)\n    matrix([[ 6600475,        0],\n            [ 6586976, 22740995]])                          #random\n\n    ")
    
    # Call to __new__(...): (line 49)
    # Processing the call arguments (line 49)
    # Getting the type of 'matrix' (line 49)
    matrix_24087 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 27), 'matrix', False)
    # Getting the type of 'shape' (line 49)
    shape_24088 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 35), 'shape', False)
    # Getting the type of 'dtype' (line 49)
    dtype_24089 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 42), 'dtype', False)
    # Processing the call keyword arguments (line 49)
    # Getting the type of 'order' (line 49)
    order_24090 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 55), 'order', False)
    keyword_24091 = order_24090
    kwargs_24092 = {'order': keyword_24091}
    # Getting the type of 'ndarray' (line 49)
    ndarray_24085 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 11), 'ndarray', False)
    # Obtaining the member '__new__' of a type (line 49)
    new___24086 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 49, 11), ndarray_24085, '__new__')
    # Calling __new__(args, kwargs) (line 49)
    new___call_result_24093 = invoke(stypy.reporting.localization.Localization(__file__, 49, 11), new___24086, *[matrix_24087, shape_24088, dtype_24089], **kwargs_24092)
    
    # Assigning a type to the variable 'stypy_return_type' (line 49)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 49, 4), 'stypy_return_type', new___call_result_24093)
    
    # ################# End of 'empty(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'empty' in the type store
    # Getting the type of 'stypy_return_type' (line 13)
    stypy_return_type_24094 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_24094)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'empty'
    return stypy_return_type_24094

# Assigning a type to the variable 'empty' (line 13)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 0), 'empty', empty)

@norecursion
def ones(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'None' (line 51)
    None_24095 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 22), 'None')
    str_24096 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 51, 34), 'str', 'C')
    defaults = [None_24095, str_24096]
    # Create a new context for function 'ones'
    module_type_store = module_type_store.open_function_context('ones', 51, 0, False)
    
    # Passed parameters checking function
    ones.stypy_localization = localization
    ones.stypy_type_of_self = None
    ones.stypy_type_store = module_type_store
    ones.stypy_function_name = 'ones'
    ones.stypy_param_names_list = ['shape', 'dtype', 'order']
    ones.stypy_varargs_param_name = None
    ones.stypy_kwargs_param_name = None
    ones.stypy_call_defaults = defaults
    ones.stypy_call_varargs = varargs
    ones.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'ones', ['shape', 'dtype', 'order'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'ones', localization, ['shape', 'dtype', 'order'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'ones(...)' code ##################

    str_24097 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 91, (-1)), 'str', "\n    Matrix of ones.\n\n    Return a matrix of given shape and type, filled with ones.\n\n    Parameters\n    ----------\n    shape : {sequence of ints, int}\n        Shape of the matrix\n    dtype : data-type, optional\n        The desired data-type for the matrix, default is np.float64.\n    order : {'C', 'F'}, optional\n        Whether to store matrix in C- or Fortran-contiguous order,\n        default is 'C'.\n\n    Returns\n    -------\n    out : matrix\n        Matrix of ones of given shape, dtype, and order.\n\n    See Also\n    --------\n    ones : Array of ones.\n    matlib.zeros : Zero matrix.\n\n    Notes\n    -----\n    If `shape` has length one i.e. ``(N,)``, or is a scalar ``N``,\n    `out` becomes a single row matrix of shape ``(1,N)``.\n\n    Examples\n    --------\n    >>> np.matlib.ones((2,3))\n    matrix([[ 1.,  1.,  1.],\n            [ 1.,  1.,  1.]])\n\n    >>> np.matlib.ones(2)\n    matrix([[ 1.,  1.]])\n\n    ")
    
    # Assigning a Call to a Name (line 92):
    
    # Assigning a Call to a Name (line 92):
    
    # Call to __new__(...): (line 92)
    # Processing the call arguments (line 92)
    # Getting the type of 'matrix' (line 92)
    matrix_24100 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 24), 'matrix', False)
    # Getting the type of 'shape' (line 92)
    shape_24101 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 32), 'shape', False)
    # Getting the type of 'dtype' (line 92)
    dtype_24102 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 39), 'dtype', False)
    # Processing the call keyword arguments (line 92)
    # Getting the type of 'order' (line 92)
    order_24103 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 52), 'order', False)
    keyword_24104 = order_24103
    kwargs_24105 = {'order': keyword_24104}
    # Getting the type of 'ndarray' (line 92)
    ndarray_24098 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 8), 'ndarray', False)
    # Obtaining the member '__new__' of a type (line 92)
    new___24099 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 92, 8), ndarray_24098, '__new__')
    # Calling __new__(args, kwargs) (line 92)
    new___call_result_24106 = invoke(stypy.reporting.localization.Localization(__file__, 92, 8), new___24099, *[matrix_24100, shape_24101, dtype_24102], **kwargs_24105)
    
    # Assigning a type to the variable 'a' (line 92)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 92, 4), 'a', new___call_result_24106)
    
    # Call to fill(...): (line 93)
    # Processing the call arguments (line 93)
    int_24109 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 93, 11), 'int')
    # Processing the call keyword arguments (line 93)
    kwargs_24110 = {}
    # Getting the type of 'a' (line 93)
    a_24107 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 4), 'a', False)
    # Obtaining the member 'fill' of a type (line 93)
    fill_24108 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 93, 4), a_24107, 'fill')
    # Calling fill(args, kwargs) (line 93)
    fill_call_result_24111 = invoke(stypy.reporting.localization.Localization(__file__, 93, 4), fill_24108, *[int_24109], **kwargs_24110)
    
    # Getting the type of 'a' (line 94)
    a_24112 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 11), 'a')
    # Assigning a type to the variable 'stypy_return_type' (line 94)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 94, 4), 'stypy_return_type', a_24112)
    
    # ################# End of 'ones(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'ones' in the type store
    # Getting the type of 'stypy_return_type' (line 51)
    stypy_return_type_24113 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_24113)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'ones'
    return stypy_return_type_24113

# Assigning a type to the variable 'ones' (line 51)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 51, 0), 'ones', ones)

@norecursion
def zeros(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'None' (line 96)
    None_24114 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 23), 'None')
    str_24115 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 96, 35), 'str', 'C')
    defaults = [None_24114, str_24115]
    # Create a new context for function 'zeros'
    module_type_store = module_type_store.open_function_context('zeros', 96, 0, False)
    
    # Passed parameters checking function
    zeros.stypy_localization = localization
    zeros.stypy_type_of_self = None
    zeros.stypy_type_store = module_type_store
    zeros.stypy_function_name = 'zeros'
    zeros.stypy_param_names_list = ['shape', 'dtype', 'order']
    zeros.stypy_varargs_param_name = None
    zeros.stypy_kwargs_param_name = None
    zeros.stypy_call_defaults = defaults
    zeros.stypy_call_varargs = varargs
    zeros.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'zeros', ['shape', 'dtype', 'order'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'zeros', localization, ['shape', 'dtype', 'order'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'zeros(...)' code ##################

    str_24116 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 135, (-1)), 'str', "\n    Return a matrix of given shape and type, filled with zeros.\n\n    Parameters\n    ----------\n    shape : int or sequence of ints\n        Shape of the matrix\n    dtype : data-type, optional\n        The desired data-type for the matrix, default is float.\n    order : {'C', 'F'}, optional\n        Whether to store the result in C- or Fortran-contiguous order,\n        default is 'C'.\n\n    Returns\n    -------\n    out : matrix\n        Zero matrix of given shape, dtype, and order.\n\n    See Also\n    --------\n    numpy.zeros : Equivalent array function.\n    matlib.ones : Return a matrix of ones.\n\n    Notes\n    -----\n    If `shape` has length one i.e. ``(N,)``, or is a scalar ``N``,\n    `out` becomes a single row matrix of shape ``(1,N)``.\n\n    Examples\n    --------\n    >>> import numpy.matlib\n    >>> np.matlib.zeros((2, 3))\n    matrix([[ 0.,  0.,  0.],\n            [ 0.,  0.,  0.]])\n\n    >>> np.matlib.zeros(2)\n    matrix([[ 0.,  0.]])\n\n    ")
    
    # Assigning a Call to a Name (line 136):
    
    # Assigning a Call to a Name (line 136):
    
    # Call to __new__(...): (line 136)
    # Processing the call arguments (line 136)
    # Getting the type of 'matrix' (line 136)
    matrix_24119 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 24), 'matrix', False)
    # Getting the type of 'shape' (line 136)
    shape_24120 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 32), 'shape', False)
    # Getting the type of 'dtype' (line 136)
    dtype_24121 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 39), 'dtype', False)
    # Processing the call keyword arguments (line 136)
    # Getting the type of 'order' (line 136)
    order_24122 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 52), 'order', False)
    keyword_24123 = order_24122
    kwargs_24124 = {'order': keyword_24123}
    # Getting the type of 'ndarray' (line 136)
    ndarray_24117 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 8), 'ndarray', False)
    # Obtaining the member '__new__' of a type (line 136)
    new___24118 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 136, 8), ndarray_24117, '__new__')
    # Calling __new__(args, kwargs) (line 136)
    new___call_result_24125 = invoke(stypy.reporting.localization.Localization(__file__, 136, 8), new___24118, *[matrix_24119, shape_24120, dtype_24121], **kwargs_24124)
    
    # Assigning a type to the variable 'a' (line 136)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 136, 4), 'a', new___call_result_24125)
    
    # Call to fill(...): (line 137)
    # Processing the call arguments (line 137)
    int_24128 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 137, 11), 'int')
    # Processing the call keyword arguments (line 137)
    kwargs_24129 = {}
    # Getting the type of 'a' (line 137)
    a_24126 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 4), 'a', False)
    # Obtaining the member 'fill' of a type (line 137)
    fill_24127 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 137, 4), a_24126, 'fill')
    # Calling fill(args, kwargs) (line 137)
    fill_call_result_24130 = invoke(stypy.reporting.localization.Localization(__file__, 137, 4), fill_24127, *[int_24128], **kwargs_24129)
    
    # Getting the type of 'a' (line 138)
    a_24131 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 11), 'a')
    # Assigning a type to the variable 'stypy_return_type' (line 138)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 138, 4), 'stypy_return_type', a_24131)
    
    # ################# End of 'zeros(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'zeros' in the type store
    # Getting the type of 'stypy_return_type' (line 96)
    stypy_return_type_24132 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_24132)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'zeros'
    return stypy_return_type_24132

# Assigning a type to the variable 'zeros' (line 96)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 96, 0), 'zeros', zeros)

@norecursion
def identity(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'None' (line 140)
    None_24133 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 21), 'None')
    defaults = [None_24133]
    # Create a new context for function 'identity'
    module_type_store = module_type_store.open_function_context('identity', 140, 0, False)
    
    # Passed parameters checking function
    identity.stypy_localization = localization
    identity.stypy_type_of_self = None
    identity.stypy_type_store = module_type_store
    identity.stypy_function_name = 'identity'
    identity.stypy_param_names_list = ['n', 'dtype']
    identity.stypy_varargs_param_name = None
    identity.stypy_kwargs_param_name = None
    identity.stypy_call_defaults = defaults
    identity.stypy_call_varargs = varargs
    identity.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'identity', ['n', 'dtype'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'identity', localization, ['n', 'dtype'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'identity(...)' code ##################

    str_24134 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 170, (-1)), 'str', '\n    Returns the square identity matrix of given size.\n\n    Parameters\n    ----------\n    n : int\n        Size of the returned identity matrix.\n    dtype : data-type, optional\n        Data-type of the output. Defaults to ``float``.\n\n    Returns\n    -------\n    out : matrix\n        `n` x `n` matrix with its main diagonal set to one,\n        and all other elements zero.\n\n    See Also\n    --------\n    numpy.identity : Equivalent array function.\n    matlib.eye : More general matrix identity function.\n\n    Examples\n    --------\n    >>> import numpy.matlib\n    >>> np.matlib.identity(3, dtype=int)\n    matrix([[1, 0, 0],\n            [0, 1, 0],\n            [0, 0, 1]])\n\n    ')
    
    # Assigning a Call to a Name (line 171):
    
    # Assigning a Call to a Name (line 171):
    
    # Call to array(...): (line 171)
    # Processing the call arguments (line 171)
    
    # Obtaining an instance of the builtin type 'list' (line 171)
    list_24136 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 171, 14), 'list')
    # Adding type elements to the builtin type 'list' instance (line 171)
    # Adding element type (line 171)
    int_24137 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 171, 15), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 171, 14), list_24136, int_24137)
    
    # Getting the type of 'n' (line 171)
    n_24138 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 18), 'n', False)
    
    # Obtaining an instance of the builtin type 'list' (line 171)
    list_24139 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 171, 20), 'list')
    # Adding type elements to the builtin type 'list' instance (line 171)
    # Adding element type (line 171)
    int_24140 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 171, 21), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 171, 20), list_24139, int_24140)
    
    # Applying the binary operator '*' (line 171)
    result_mul_24141 = python_operator(stypy.reporting.localization.Localization(__file__, 171, 18), '*', n_24138, list_24139)
    
    # Applying the binary operator '+' (line 171)
    result_add_24142 = python_operator(stypy.reporting.localization.Localization(__file__, 171, 14), '+', list_24136, result_mul_24141)
    
    # Processing the call keyword arguments (line 171)
    # Getting the type of 'dtype' (line 171)
    dtype_24143 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 31), 'dtype', False)
    keyword_24144 = dtype_24143
    kwargs_24145 = {'dtype': keyword_24144}
    # Getting the type of 'array' (line 171)
    array_24135 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 8), 'array', False)
    # Calling array(args, kwargs) (line 171)
    array_call_result_24146 = invoke(stypy.reporting.localization.Localization(__file__, 171, 8), array_24135, *[result_add_24142], **kwargs_24145)
    
    # Assigning a type to the variable 'a' (line 171)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 171, 4), 'a', array_call_result_24146)
    
    # Assigning a Call to a Name (line 172):
    
    # Assigning a Call to a Name (line 172):
    
    # Call to empty(...): (line 172)
    # Processing the call arguments (line 172)
    
    # Obtaining an instance of the builtin type 'tuple' (line 172)
    tuple_24148 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 172, 15), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 172)
    # Adding element type (line 172)
    # Getting the type of 'n' (line 172)
    n_24149 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 15), 'n', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 172, 15), tuple_24148, n_24149)
    # Adding element type (line 172)
    # Getting the type of 'n' (line 172)
    n_24150 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 18), 'n', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 172, 15), tuple_24148, n_24150)
    
    # Processing the call keyword arguments (line 172)
    # Getting the type of 'dtype' (line 172)
    dtype_24151 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 28), 'dtype', False)
    keyword_24152 = dtype_24151
    kwargs_24153 = {'dtype': keyword_24152}
    # Getting the type of 'empty' (line 172)
    empty_24147 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 8), 'empty', False)
    # Calling empty(args, kwargs) (line 172)
    empty_call_result_24154 = invoke(stypy.reporting.localization.Localization(__file__, 172, 8), empty_24147, *[tuple_24148], **kwargs_24153)
    
    # Assigning a type to the variable 'b' (line 172)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 172, 4), 'b', empty_call_result_24154)
    
    # Assigning a Name to a Attribute (line 173):
    
    # Assigning a Name to a Attribute (line 173):
    # Getting the type of 'a' (line 173)
    a_24155 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 13), 'a')
    # Getting the type of 'b' (line 173)
    b_24156 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 4), 'b')
    # Setting the type of the member 'flat' of a type (line 173)
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 173, 4), b_24156, 'flat', a_24155)
    # Getting the type of 'b' (line 174)
    b_24157 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 11), 'b')
    # Assigning a type to the variable 'stypy_return_type' (line 174)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 174, 4), 'stypy_return_type', b_24157)
    
    # ################# End of 'identity(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'identity' in the type store
    # Getting the type of 'stypy_return_type' (line 140)
    stypy_return_type_24158 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_24158)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'identity'
    return stypy_return_type_24158

# Assigning a type to the variable 'identity' (line 140)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 140, 0), 'identity', identity)

@norecursion
def eye(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'None' (line 176)
    None_24159 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 12), 'None')
    int_24160 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 176, 20), 'int')
    # Getting the type of 'float' (line 176)
    float_24161 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 29), 'float')
    defaults = [None_24159, int_24160, float_24161]
    # Create a new context for function 'eye'
    module_type_store = module_type_store.open_function_context('eye', 176, 0, False)
    
    # Passed parameters checking function
    eye.stypy_localization = localization
    eye.stypy_type_of_self = None
    eye.stypy_type_store = module_type_store
    eye.stypy_function_name = 'eye'
    eye.stypy_param_names_list = ['n', 'M', 'k', 'dtype']
    eye.stypy_varargs_param_name = None
    eye.stypy_kwargs_param_name = None
    eye.stypy_call_defaults = defaults
    eye.stypy_call_varargs = varargs
    eye.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'eye', ['n', 'M', 'k', 'dtype'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'eye', localization, ['n', 'M', 'k', 'dtype'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'eye(...)' code ##################

    str_24162 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 212, (-1)), 'str', '\n    Return a matrix with ones on the diagonal and zeros elsewhere.\n\n    Parameters\n    ----------\n    n : int\n        Number of rows in the output.\n    M : int, optional\n        Number of columns in the output, defaults to `n`.\n    k : int, optional\n        Index of the diagonal: 0 refers to the main diagonal,\n        a positive value refers to an upper diagonal,\n        and a negative value to a lower diagonal.\n    dtype : dtype, optional\n        Data-type of the returned matrix.\n\n    Returns\n    -------\n    I : matrix\n        A `n` x `M` matrix where all elements are equal to zero,\n        except for the `k`-th diagonal, whose values are equal to one.\n\n    See Also\n    --------\n    numpy.eye : Equivalent array function.\n    identity : Square identity matrix.\n\n    Examples\n    --------\n    >>> import numpy.matlib\n    >>> np.matlib.eye(3, k=1, dtype=float)\n    matrix([[ 0.,  1.,  0.],\n            [ 0.,  0.,  1.],\n            [ 0.,  0.,  0.]])\n\n    ')
    
    # Call to asmatrix(...): (line 213)
    # Processing the call arguments (line 213)
    
    # Call to eye(...): (line 213)
    # Processing the call arguments (line 213)
    # Getting the type of 'n' (line 213)
    n_24166 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 27), 'n', False)
    # Getting the type of 'M' (line 213)
    M_24167 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 30), 'M', False)
    # Getting the type of 'k' (line 213)
    k_24168 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 33), 'k', False)
    # Getting the type of 'dtype' (line 213)
    dtype_24169 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 36), 'dtype', False)
    # Processing the call keyword arguments (line 213)
    kwargs_24170 = {}
    # Getting the type of 'np' (line 213)
    np_24164 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 20), 'np', False)
    # Obtaining the member 'eye' of a type (line 213)
    eye_24165 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 213, 20), np_24164, 'eye')
    # Calling eye(args, kwargs) (line 213)
    eye_call_result_24171 = invoke(stypy.reporting.localization.Localization(__file__, 213, 20), eye_24165, *[n_24166, M_24167, k_24168, dtype_24169], **kwargs_24170)
    
    # Processing the call keyword arguments (line 213)
    kwargs_24172 = {}
    # Getting the type of 'asmatrix' (line 213)
    asmatrix_24163 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 11), 'asmatrix', False)
    # Calling asmatrix(args, kwargs) (line 213)
    asmatrix_call_result_24173 = invoke(stypy.reporting.localization.Localization(__file__, 213, 11), asmatrix_24163, *[eye_call_result_24171], **kwargs_24172)
    
    # Assigning a type to the variable 'stypy_return_type' (line 213)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 213, 4), 'stypy_return_type', asmatrix_call_result_24173)
    
    # ################# End of 'eye(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'eye' in the type store
    # Getting the type of 'stypy_return_type' (line 176)
    stypy_return_type_24174 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_24174)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'eye'
    return stypy_return_type_24174

# Assigning a type to the variable 'eye' (line 176)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 176, 0), 'eye', eye)

@norecursion
def rand(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'rand'
    module_type_store = module_type_store.open_function_context('rand', 215, 0, False)
    
    # Passed parameters checking function
    rand.stypy_localization = localization
    rand.stypy_type_of_self = None
    rand.stypy_type_store = module_type_store
    rand.stypy_function_name = 'rand'
    rand.stypy_param_names_list = []
    rand.stypy_varargs_param_name = 'args'
    rand.stypy_kwargs_param_name = None
    rand.stypy_call_defaults = defaults
    rand.stypy_call_varargs = varargs
    rand.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'rand', [], 'args', None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'rand', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'rand(...)' code ##################

    str_24175 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 255, (-1)), 'str', '\n    Return a matrix of random values with given shape.\n\n    Create a matrix of the given shape and propagate it with\n    random samples from a uniform distribution over ``[0, 1)``.\n\n    Parameters\n    ----------\n    \\*args : Arguments\n        Shape of the output.\n        If given as N integers, each integer specifies the size of one\n        dimension.\n        If given as a tuple, this tuple gives the complete shape.\n\n    Returns\n    -------\n    out : ndarray\n        The matrix of random values with shape given by `\\*args`.\n\n    See Also\n    --------\n    randn, numpy.random.rand\n\n    Examples\n    --------\n    >>> import numpy.matlib\n    >>> np.matlib.rand(2, 3)\n    matrix([[ 0.68340382,  0.67926887,  0.83271405],\n            [ 0.00793551,  0.20468222,  0.95253525]])       #random\n    >>> np.matlib.rand((2, 3))\n    matrix([[ 0.84682055,  0.73626594,  0.11308016],\n            [ 0.85429008,  0.3294825 ,  0.89139555]])       #random\n\n    If the first argument is a tuple, other arguments are ignored:\n\n    >>> np.matlib.rand((2, 3), 4)\n    matrix([[ 0.46898646,  0.15163588,  0.95188261],\n            [ 0.59208621,  0.09561818,  0.00583606]])       #random\n\n    ')
    
    # Type idiom detected: calculating its left and rigth part (line 256)
    # Getting the type of 'tuple' (line 256)
    tuple_24176 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 256, 27), 'tuple')
    
    # Obtaining the type of the subscript
    int_24177 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 256, 23), 'int')
    # Getting the type of 'args' (line 256)
    args_24178 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 256, 18), 'args')
    # Obtaining the member '__getitem__' of a type (line 256)
    getitem___24179 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 256, 18), args_24178, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 256)
    subscript_call_result_24180 = invoke(stypy.reporting.localization.Localization(__file__, 256, 18), getitem___24179, int_24177)
    
    
    (may_be_24181, more_types_in_union_24182) = may_be_subtype(tuple_24176, subscript_call_result_24180)

    if may_be_24181:

        if more_types_in_union_24182:
            # Runtime conditional SSA (line 256)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a Subscript to a Name (line 257):
        
        # Assigning a Subscript to a Name (line 257):
        
        # Obtaining the type of the subscript
        int_24183 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 257, 20), 'int')
        # Getting the type of 'args' (line 257)
        args_24184 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 257, 15), 'args')
        # Obtaining the member '__getitem__' of a type (line 257)
        getitem___24185 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 257, 15), args_24184, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 257)
        subscript_call_result_24186 = invoke(stypy.reporting.localization.Localization(__file__, 257, 15), getitem___24185, int_24183)
        
        # Assigning a type to the variable 'args' (line 257)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 257, 8), 'args', subscript_call_result_24186)

        if more_types_in_union_24182:
            # SSA join for if statement (line 256)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Call to asmatrix(...): (line 258)
    # Processing the call arguments (line 258)
    
    # Call to rand(...): (line 258)
    # Getting the type of 'args' (line 258)
    args_24191 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 258, 36), 'args', False)
    # Processing the call keyword arguments (line 258)
    kwargs_24192 = {}
    # Getting the type of 'np' (line 258)
    np_24188 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 258, 20), 'np', False)
    # Obtaining the member 'random' of a type (line 258)
    random_24189 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 258, 20), np_24188, 'random')
    # Obtaining the member 'rand' of a type (line 258)
    rand_24190 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 258, 20), random_24189, 'rand')
    # Calling rand(args, kwargs) (line 258)
    rand_call_result_24193 = invoke(stypy.reporting.localization.Localization(__file__, 258, 20), rand_24190, *[args_24191], **kwargs_24192)
    
    # Processing the call keyword arguments (line 258)
    kwargs_24194 = {}
    # Getting the type of 'asmatrix' (line 258)
    asmatrix_24187 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 258, 11), 'asmatrix', False)
    # Calling asmatrix(args, kwargs) (line 258)
    asmatrix_call_result_24195 = invoke(stypy.reporting.localization.Localization(__file__, 258, 11), asmatrix_24187, *[rand_call_result_24193], **kwargs_24194)
    
    # Assigning a type to the variable 'stypy_return_type' (line 258)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 258, 4), 'stypy_return_type', asmatrix_call_result_24195)
    
    # ################# End of 'rand(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'rand' in the type store
    # Getting the type of 'stypy_return_type' (line 215)
    stypy_return_type_24196 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 215, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_24196)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'rand'
    return stypy_return_type_24196

# Assigning a type to the variable 'rand' (line 215)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 215, 0), 'rand', rand)

@norecursion
def randn(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'randn'
    module_type_store = module_type_store.open_function_context('randn', 260, 0, False)
    
    # Passed parameters checking function
    randn.stypy_localization = localization
    randn.stypy_type_of_self = None
    randn.stypy_type_store = module_type_store
    randn.stypy_function_name = 'randn'
    randn.stypy_param_names_list = []
    randn.stypy_varargs_param_name = 'args'
    randn.stypy_kwargs_param_name = None
    randn.stypy_call_defaults = defaults
    randn.stypy_call_varargs = varargs
    randn.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'randn', [], 'args', None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'randn', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'randn(...)' code ##################

    str_24197 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 305, (-1)), 'str', '\n    Return a random matrix with data from the "standard normal" distribution.\n\n    `randn` generates a matrix filled with random floats sampled from a\n    univariate "normal" (Gaussian) distribution of mean 0 and variance 1.\n\n    Parameters\n    ----------\n    \\*args : Arguments\n        Shape of the output.\n        If given as N integers, each integer specifies the size of one\n        dimension. If given as a tuple, this tuple gives the complete shape.\n\n    Returns\n    -------\n    Z : matrix of floats\n        A matrix of floating-point samples drawn from the standard normal\n        distribution.\n\n    See Also\n    --------\n    rand, random.randn\n\n    Notes\n    -----\n    For random samples from :math:`N(\\mu, \\sigma^2)`, use:\n\n    ``sigma * np.matlib.randn(...) + mu``\n\n    Examples\n    --------\n    >>> import numpy.matlib\n    >>> np.matlib.randn(1)\n    matrix([[-0.09542833]])                                 #random\n    >>> np.matlib.randn(1, 2, 3)\n    matrix([[ 0.16198284,  0.0194571 ,  0.18312985],\n            [-0.7509172 ,  1.61055   ,  0.45298599]])       #random\n\n    Two-by-four matrix of samples from :math:`N(3, 6.25)`:\n\n    >>> 2.5 * np.matlib.randn((2, 4)) + 3\n    matrix([[ 4.74085004,  8.89381862,  4.09042411,  4.83721922],\n            [ 7.52373709,  5.07933944, -2.64043543,  0.45610557]])  #random\n\n    ')
    
    # Type idiom detected: calculating its left and rigth part (line 306)
    # Getting the type of 'tuple' (line 306)
    tuple_24198 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 306, 27), 'tuple')
    
    # Obtaining the type of the subscript
    int_24199 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 306, 23), 'int')
    # Getting the type of 'args' (line 306)
    args_24200 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 306, 18), 'args')
    # Obtaining the member '__getitem__' of a type (line 306)
    getitem___24201 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 306, 18), args_24200, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 306)
    subscript_call_result_24202 = invoke(stypy.reporting.localization.Localization(__file__, 306, 18), getitem___24201, int_24199)
    
    
    (may_be_24203, more_types_in_union_24204) = may_be_subtype(tuple_24198, subscript_call_result_24202)

    if may_be_24203:

        if more_types_in_union_24204:
            # Runtime conditional SSA (line 306)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a Subscript to a Name (line 307):
        
        # Assigning a Subscript to a Name (line 307):
        
        # Obtaining the type of the subscript
        int_24205 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 307, 20), 'int')
        # Getting the type of 'args' (line 307)
        args_24206 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 307, 15), 'args')
        # Obtaining the member '__getitem__' of a type (line 307)
        getitem___24207 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 307, 15), args_24206, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 307)
        subscript_call_result_24208 = invoke(stypy.reporting.localization.Localization(__file__, 307, 15), getitem___24207, int_24205)
        
        # Assigning a type to the variable 'args' (line 307)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 307, 8), 'args', subscript_call_result_24208)

        if more_types_in_union_24204:
            # SSA join for if statement (line 306)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Call to asmatrix(...): (line 308)
    # Processing the call arguments (line 308)
    
    # Call to randn(...): (line 308)
    # Getting the type of 'args' (line 308)
    args_24213 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 308, 37), 'args', False)
    # Processing the call keyword arguments (line 308)
    kwargs_24214 = {}
    # Getting the type of 'np' (line 308)
    np_24210 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 308, 20), 'np', False)
    # Obtaining the member 'random' of a type (line 308)
    random_24211 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 308, 20), np_24210, 'random')
    # Obtaining the member 'randn' of a type (line 308)
    randn_24212 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 308, 20), random_24211, 'randn')
    # Calling randn(args, kwargs) (line 308)
    randn_call_result_24215 = invoke(stypy.reporting.localization.Localization(__file__, 308, 20), randn_24212, *[args_24213], **kwargs_24214)
    
    # Processing the call keyword arguments (line 308)
    kwargs_24216 = {}
    # Getting the type of 'asmatrix' (line 308)
    asmatrix_24209 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 308, 11), 'asmatrix', False)
    # Calling asmatrix(args, kwargs) (line 308)
    asmatrix_call_result_24217 = invoke(stypy.reporting.localization.Localization(__file__, 308, 11), asmatrix_24209, *[randn_call_result_24215], **kwargs_24216)
    
    # Assigning a type to the variable 'stypy_return_type' (line 308)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 308, 4), 'stypy_return_type', asmatrix_call_result_24217)
    
    # ################# End of 'randn(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'randn' in the type store
    # Getting the type of 'stypy_return_type' (line 260)
    stypy_return_type_24218 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 260, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_24218)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'randn'
    return stypy_return_type_24218

# Assigning a type to the variable 'randn' (line 260)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 260, 0), 'randn', randn)

@norecursion
def repmat(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'repmat'
    module_type_store = module_type_store.open_function_context('repmat', 310, 0, False)
    
    # Passed parameters checking function
    repmat.stypy_localization = localization
    repmat.stypy_type_of_self = None
    repmat.stypy_type_store = module_type_store
    repmat.stypy_function_name = 'repmat'
    repmat.stypy_param_names_list = ['a', 'm', 'n']
    repmat.stypy_varargs_param_name = None
    repmat.stypy_kwargs_param_name = None
    repmat.stypy_call_defaults = defaults
    repmat.stypy_call_varargs = varargs
    repmat.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'repmat', ['a', 'm', 'n'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'repmat', localization, ['a', 'm', 'n'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'repmat(...)' code ##################

    str_24219 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 346, (-1)), 'str', '\n    Repeat a 0-D to 2-D array or matrix MxN times.\n\n    Parameters\n    ----------\n    a : array_like\n        The array or matrix to be repeated.\n    m, n : int\n        The number of times `a` is repeated along the first and second axes.\n\n    Returns\n    -------\n    out : ndarray\n        The result of repeating `a`.\n\n    Examples\n    --------\n    >>> import numpy.matlib\n    >>> a0 = np.array(1)\n    >>> np.matlib.repmat(a0, 2, 3)\n    array([[1, 1, 1],\n           [1, 1, 1]])\n\n    >>> a1 = np.arange(4)\n    >>> np.matlib.repmat(a1, 2, 2)\n    array([[0, 1, 2, 3, 0, 1, 2, 3],\n           [0, 1, 2, 3, 0, 1, 2, 3]])\n\n    >>> a2 = np.asmatrix(np.arange(6).reshape(2, 3))\n    >>> np.matlib.repmat(a2, 2, 3)\n    matrix([[0, 1, 2, 0, 1, 2, 0, 1, 2],\n            [3, 4, 5, 3, 4, 5, 3, 4, 5],\n            [0, 1, 2, 0, 1, 2, 0, 1, 2],\n            [3, 4, 5, 3, 4, 5, 3, 4, 5]])\n\n    ')
    
    # Assigning a Call to a Name (line 347):
    
    # Assigning a Call to a Name (line 347):
    
    # Call to asanyarray(...): (line 347)
    # Processing the call arguments (line 347)
    # Getting the type of 'a' (line 347)
    a_24221 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 347, 19), 'a', False)
    # Processing the call keyword arguments (line 347)
    kwargs_24222 = {}
    # Getting the type of 'asanyarray' (line 347)
    asanyarray_24220 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 347, 8), 'asanyarray', False)
    # Calling asanyarray(args, kwargs) (line 347)
    asanyarray_call_result_24223 = invoke(stypy.reporting.localization.Localization(__file__, 347, 8), asanyarray_24220, *[a_24221], **kwargs_24222)
    
    # Assigning a type to the variable 'a' (line 347)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 347, 4), 'a', asanyarray_call_result_24223)
    
    # Assigning a Attribute to a Name (line 348):
    
    # Assigning a Attribute to a Name (line 348):
    # Getting the type of 'a' (line 348)
    a_24224 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 348, 11), 'a')
    # Obtaining the member 'ndim' of a type (line 348)
    ndim_24225 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 348, 11), a_24224, 'ndim')
    # Assigning a type to the variable 'ndim' (line 348)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 348, 4), 'ndim', ndim_24225)
    
    
    # Getting the type of 'ndim' (line 349)
    ndim_24226 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 349, 7), 'ndim')
    int_24227 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 349, 15), 'int')
    # Applying the binary operator '==' (line 349)
    result_eq_24228 = python_operator(stypy.reporting.localization.Localization(__file__, 349, 7), '==', ndim_24226, int_24227)
    
    # Testing the type of an if condition (line 349)
    if_condition_24229 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 349, 4), result_eq_24228)
    # Assigning a type to the variable 'if_condition_24229' (line 349)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 349, 4), 'if_condition_24229', if_condition_24229)
    # SSA begins for if statement (line 349)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Tuple to a Tuple (line 350):
    
    # Assigning a Num to a Name (line 350):
    int_24230 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 350, 30), 'int')
    # Assigning a type to the variable 'tuple_assignment_24057' (line 350)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 350, 8), 'tuple_assignment_24057', int_24230)
    
    # Assigning a Num to a Name (line 350):
    int_24231 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 350, 33), 'int')
    # Assigning a type to the variable 'tuple_assignment_24058' (line 350)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 350, 8), 'tuple_assignment_24058', int_24231)
    
    # Assigning a Name to a Name (line 350):
    # Getting the type of 'tuple_assignment_24057' (line 350)
    tuple_assignment_24057_24232 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 350, 8), 'tuple_assignment_24057')
    # Assigning a type to the variable 'origrows' (line 350)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 350, 8), 'origrows', tuple_assignment_24057_24232)
    
    # Assigning a Name to a Name (line 350):
    # Getting the type of 'tuple_assignment_24058' (line 350)
    tuple_assignment_24058_24233 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 350, 8), 'tuple_assignment_24058')
    # Assigning a type to the variable 'origcols' (line 350)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 350, 18), 'origcols', tuple_assignment_24058_24233)
    # SSA branch for the else part of an if statement (line 349)
    module_type_store.open_ssa_branch('else')
    
    
    # Getting the type of 'ndim' (line 351)
    ndim_24234 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 351, 9), 'ndim')
    int_24235 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 351, 17), 'int')
    # Applying the binary operator '==' (line 351)
    result_eq_24236 = python_operator(stypy.reporting.localization.Localization(__file__, 351, 9), '==', ndim_24234, int_24235)
    
    # Testing the type of an if condition (line 351)
    if_condition_24237 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 351, 9), result_eq_24236)
    # Assigning a type to the variable 'if_condition_24237' (line 351)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 351, 9), 'if_condition_24237', if_condition_24237)
    # SSA begins for if statement (line 351)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Tuple to a Tuple (line 352):
    
    # Assigning a Num to a Name (line 352):
    int_24238 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 352, 30), 'int')
    # Assigning a type to the variable 'tuple_assignment_24059' (line 352)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 352, 8), 'tuple_assignment_24059', int_24238)
    
    # Assigning a Subscript to a Name (line 352):
    
    # Obtaining the type of the subscript
    int_24239 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 352, 41), 'int')
    # Getting the type of 'a' (line 352)
    a_24240 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 352, 33), 'a')
    # Obtaining the member 'shape' of a type (line 352)
    shape_24241 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 352, 33), a_24240, 'shape')
    # Obtaining the member '__getitem__' of a type (line 352)
    getitem___24242 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 352, 33), shape_24241, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 352)
    subscript_call_result_24243 = invoke(stypy.reporting.localization.Localization(__file__, 352, 33), getitem___24242, int_24239)
    
    # Assigning a type to the variable 'tuple_assignment_24060' (line 352)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 352, 8), 'tuple_assignment_24060', subscript_call_result_24243)
    
    # Assigning a Name to a Name (line 352):
    # Getting the type of 'tuple_assignment_24059' (line 352)
    tuple_assignment_24059_24244 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 352, 8), 'tuple_assignment_24059')
    # Assigning a type to the variable 'origrows' (line 352)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 352, 8), 'origrows', tuple_assignment_24059_24244)
    
    # Assigning a Name to a Name (line 352):
    # Getting the type of 'tuple_assignment_24060' (line 352)
    tuple_assignment_24060_24245 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 352, 8), 'tuple_assignment_24060')
    # Assigning a type to the variable 'origcols' (line 352)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 352, 18), 'origcols', tuple_assignment_24060_24245)
    # SSA branch for the else part of an if statement (line 351)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Attribute to a Tuple (line 354):
    
    # Assigning a Subscript to a Name (line 354):
    
    # Obtaining the type of the subscript
    int_24246 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 354, 8), 'int')
    # Getting the type of 'a' (line 354)
    a_24247 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 354, 29), 'a')
    # Obtaining the member 'shape' of a type (line 354)
    shape_24248 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 354, 29), a_24247, 'shape')
    # Obtaining the member '__getitem__' of a type (line 354)
    getitem___24249 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 354, 8), shape_24248, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 354)
    subscript_call_result_24250 = invoke(stypy.reporting.localization.Localization(__file__, 354, 8), getitem___24249, int_24246)
    
    # Assigning a type to the variable 'tuple_var_assignment_24061' (line 354)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 354, 8), 'tuple_var_assignment_24061', subscript_call_result_24250)
    
    # Assigning a Subscript to a Name (line 354):
    
    # Obtaining the type of the subscript
    int_24251 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 354, 8), 'int')
    # Getting the type of 'a' (line 354)
    a_24252 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 354, 29), 'a')
    # Obtaining the member 'shape' of a type (line 354)
    shape_24253 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 354, 29), a_24252, 'shape')
    # Obtaining the member '__getitem__' of a type (line 354)
    getitem___24254 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 354, 8), shape_24253, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 354)
    subscript_call_result_24255 = invoke(stypy.reporting.localization.Localization(__file__, 354, 8), getitem___24254, int_24251)
    
    # Assigning a type to the variable 'tuple_var_assignment_24062' (line 354)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 354, 8), 'tuple_var_assignment_24062', subscript_call_result_24255)
    
    # Assigning a Name to a Name (line 354):
    # Getting the type of 'tuple_var_assignment_24061' (line 354)
    tuple_var_assignment_24061_24256 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 354, 8), 'tuple_var_assignment_24061')
    # Assigning a type to the variable 'origrows' (line 354)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 354, 8), 'origrows', tuple_var_assignment_24061_24256)
    
    # Assigning a Name to a Name (line 354):
    # Getting the type of 'tuple_var_assignment_24062' (line 354)
    tuple_var_assignment_24062_24257 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 354, 8), 'tuple_var_assignment_24062')
    # Assigning a type to the variable 'origcols' (line 354)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 354, 18), 'origcols', tuple_var_assignment_24062_24257)
    # SSA join for if statement (line 351)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 349)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a BinOp to a Name (line 355):
    
    # Assigning a BinOp to a Name (line 355):
    # Getting the type of 'origrows' (line 355)
    origrows_24258 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 355, 11), 'origrows')
    # Getting the type of 'm' (line 355)
    m_24259 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 355, 22), 'm')
    # Applying the binary operator '*' (line 355)
    result_mul_24260 = python_operator(stypy.reporting.localization.Localization(__file__, 355, 11), '*', origrows_24258, m_24259)
    
    # Assigning a type to the variable 'rows' (line 355)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 355, 4), 'rows', result_mul_24260)
    
    # Assigning a BinOp to a Name (line 356):
    
    # Assigning a BinOp to a Name (line 356):
    # Getting the type of 'origcols' (line 356)
    origcols_24261 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 356, 11), 'origcols')
    # Getting the type of 'n' (line 356)
    n_24262 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 356, 22), 'n')
    # Applying the binary operator '*' (line 356)
    result_mul_24263 = python_operator(stypy.reporting.localization.Localization(__file__, 356, 11), '*', origcols_24261, n_24262)
    
    # Assigning a type to the variable 'cols' (line 356)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 356, 4), 'cols', result_mul_24263)
    
    # Assigning a Call to a Name (line 357):
    
    # Assigning a Call to a Name (line 357):
    
    # Call to repeat(...): (line 357)
    # Processing the call arguments (line 357)
    # Getting the type of 'n' (line 357)
    n_24282 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 357, 73), 'n', False)
    int_24283 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 357, 76), 'int')
    # Processing the call keyword arguments (line 357)
    kwargs_24284 = {}
    
    # Call to reshape(...): (line 357)
    # Processing the call arguments (line 357)
    # Getting the type of 'rows' (line 357)
    rows_24277 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 357, 50), 'rows', False)
    # Getting the type of 'origcols' (line 357)
    origcols_24278 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 357, 56), 'origcols', False)
    # Processing the call keyword arguments (line 357)
    kwargs_24279 = {}
    
    # Call to repeat(...): (line 357)
    # Processing the call arguments (line 357)
    # Getting the type of 'm' (line 357)
    m_24272 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 357, 36), 'm', False)
    int_24273 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 357, 39), 'int')
    # Processing the call keyword arguments (line 357)
    kwargs_24274 = {}
    
    # Call to reshape(...): (line 357)
    # Processing the call arguments (line 357)
    int_24266 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 357, 18), 'int')
    # Getting the type of 'a' (line 357)
    a_24267 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 357, 21), 'a', False)
    # Obtaining the member 'size' of a type (line 357)
    size_24268 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 357, 21), a_24267, 'size')
    # Processing the call keyword arguments (line 357)
    kwargs_24269 = {}
    # Getting the type of 'a' (line 357)
    a_24264 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 357, 8), 'a', False)
    # Obtaining the member 'reshape' of a type (line 357)
    reshape_24265 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 357, 8), a_24264, 'reshape')
    # Calling reshape(args, kwargs) (line 357)
    reshape_call_result_24270 = invoke(stypy.reporting.localization.Localization(__file__, 357, 8), reshape_24265, *[int_24266, size_24268], **kwargs_24269)
    
    # Obtaining the member 'repeat' of a type (line 357)
    repeat_24271 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 357, 8), reshape_call_result_24270, 'repeat')
    # Calling repeat(args, kwargs) (line 357)
    repeat_call_result_24275 = invoke(stypy.reporting.localization.Localization(__file__, 357, 8), repeat_24271, *[m_24272, int_24273], **kwargs_24274)
    
    # Obtaining the member 'reshape' of a type (line 357)
    reshape_24276 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 357, 8), repeat_call_result_24275, 'reshape')
    # Calling reshape(args, kwargs) (line 357)
    reshape_call_result_24280 = invoke(stypy.reporting.localization.Localization(__file__, 357, 8), reshape_24276, *[rows_24277, origcols_24278], **kwargs_24279)
    
    # Obtaining the member 'repeat' of a type (line 357)
    repeat_24281 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 357, 8), reshape_call_result_24280, 'repeat')
    # Calling repeat(args, kwargs) (line 357)
    repeat_call_result_24285 = invoke(stypy.reporting.localization.Localization(__file__, 357, 8), repeat_24281, *[n_24282, int_24283], **kwargs_24284)
    
    # Assigning a type to the variable 'c' (line 357)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 357, 4), 'c', repeat_call_result_24285)
    
    # Call to reshape(...): (line 358)
    # Processing the call arguments (line 358)
    # Getting the type of 'rows' (line 358)
    rows_24288 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 358, 21), 'rows', False)
    # Getting the type of 'cols' (line 358)
    cols_24289 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 358, 27), 'cols', False)
    # Processing the call keyword arguments (line 358)
    kwargs_24290 = {}
    # Getting the type of 'c' (line 358)
    c_24286 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 358, 11), 'c', False)
    # Obtaining the member 'reshape' of a type (line 358)
    reshape_24287 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 358, 11), c_24286, 'reshape')
    # Calling reshape(args, kwargs) (line 358)
    reshape_call_result_24291 = invoke(stypy.reporting.localization.Localization(__file__, 358, 11), reshape_24287, *[rows_24288, cols_24289], **kwargs_24290)
    
    # Assigning a type to the variable 'stypy_return_type' (line 358)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 358, 4), 'stypy_return_type', reshape_call_result_24291)
    
    # ################# End of 'repmat(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'repmat' in the type store
    # Getting the type of 'stypy_return_type' (line 310)
    stypy_return_type_24292 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 310, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_24292)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'repmat'
    return stypy_return_type_24292

# Assigning a type to the variable 'repmat' (line 310)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 310, 0), 'repmat', repmat)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
