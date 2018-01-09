
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: from __future__ import division, absolute_import, print_function
2: 
3: __all__ = ['atleast_1d', 'atleast_2d', 'atleast_3d', 'vstack', 'hstack',
4:            'stack']
5: 
6: from . import numeric as _nx
7: from .numeric import asanyarray, newaxis
8: 
9: def atleast_1d(*arys):
10:     '''
11:     Convert inputs to arrays with at least one dimension.
12: 
13:     Scalar inputs are converted to 1-dimensional arrays, whilst
14:     higher-dimensional inputs are preserved.
15: 
16:     Parameters
17:     ----------
18:     arys1, arys2, ... : array_like
19:         One or more input arrays.
20: 
21:     Returns
22:     -------
23:     ret : ndarray
24:         An array, or sequence of arrays, each with ``a.ndim >= 1``.
25:         Copies are made only if necessary.
26: 
27:     See Also
28:     --------
29:     atleast_2d, atleast_3d
30: 
31:     Examples
32:     --------
33:     >>> np.atleast_1d(1.0)
34:     array([ 1.])
35: 
36:     >>> x = np.arange(9.0).reshape(3,3)
37:     >>> np.atleast_1d(x)
38:     array([[ 0.,  1.,  2.],
39:            [ 3.,  4.,  5.],
40:            [ 6.,  7.,  8.]])
41:     >>> np.atleast_1d(x) is x
42:     True
43: 
44:     >>> np.atleast_1d(1, [3, 4])
45:     [array([1]), array([3, 4])]
46: 
47:     '''
48:     res = []
49:     for ary in arys:
50:         ary = asanyarray(ary)
51:         if len(ary.shape) == 0:
52:             result = ary.reshape(1)
53:         else:
54:             result = ary
55:         res.append(result)
56:     if len(res) == 1:
57:         return res[0]
58:     else:
59:         return res
60: 
61: def atleast_2d(*arys):
62:     '''
63:     View inputs as arrays with at least two dimensions.
64: 
65:     Parameters
66:     ----------
67:     arys1, arys2, ... : array_like
68:         One or more array-like sequences.  Non-array inputs are converted
69:         to arrays.  Arrays that already have two or more dimensions are
70:         preserved.
71: 
72:     Returns
73:     -------
74:     res, res2, ... : ndarray
75:         An array, or tuple of arrays, each with ``a.ndim >= 2``.
76:         Copies are avoided where possible, and views with two or more
77:         dimensions are returned.
78: 
79:     See Also
80:     --------
81:     atleast_1d, atleast_3d
82: 
83:     Examples
84:     --------
85:     >>> np.atleast_2d(3.0)
86:     array([[ 3.]])
87: 
88:     >>> x = np.arange(3.0)
89:     >>> np.atleast_2d(x)
90:     array([[ 0.,  1.,  2.]])
91:     >>> np.atleast_2d(x).base is x
92:     True
93: 
94:     >>> np.atleast_2d(1, [1, 2], [[1, 2]])
95:     [array([[1]]), array([[1, 2]]), array([[1, 2]])]
96: 
97:     '''
98:     res = []
99:     for ary in arys:
100:         ary = asanyarray(ary)
101:         if len(ary.shape) == 0:
102:             result = ary.reshape(1, 1)
103:         elif len(ary.shape) == 1:
104:             result = ary[newaxis,:]
105:         else:
106:             result = ary
107:         res.append(result)
108:     if len(res) == 1:
109:         return res[0]
110:     else:
111:         return res
112: 
113: def atleast_3d(*arys):
114:     '''
115:     View inputs as arrays with at least three dimensions.
116: 
117:     Parameters
118:     ----------
119:     arys1, arys2, ... : array_like
120:         One or more array-like sequences.  Non-array inputs are converted to
121:         arrays.  Arrays that already have three or more dimensions are
122:         preserved.
123: 
124:     Returns
125:     -------
126:     res1, res2, ... : ndarray
127:         An array, or tuple of arrays, each with ``a.ndim >= 3``.  Copies are
128:         avoided where possible, and views with three or more dimensions are
129:         returned.  For example, a 1-D array of shape ``(N,)`` becomes a view
130:         of shape ``(1, N, 1)``, and a 2-D array of shape ``(M, N)`` becomes a
131:         view of shape ``(M, N, 1)``.
132: 
133:     See Also
134:     --------
135:     atleast_1d, atleast_2d
136: 
137:     Examples
138:     --------
139:     >>> np.atleast_3d(3.0)
140:     array([[[ 3.]]])
141: 
142:     >>> x = np.arange(3.0)
143:     >>> np.atleast_3d(x).shape
144:     (1, 3, 1)
145: 
146:     >>> x = np.arange(12.0).reshape(4,3)
147:     >>> np.atleast_3d(x).shape
148:     (4, 3, 1)
149:     >>> np.atleast_3d(x).base is x
150:     True
151: 
152:     >>> for arr in np.atleast_3d([1, 2], [[1, 2]], [[[1, 2]]]):
153:     ...     print(arr, arr.shape)
154:     ...
155:     [[[1]
156:       [2]]] (1, 2, 1)
157:     [[[1]
158:       [2]]] (1, 2, 1)
159:     [[[1 2]]] (1, 1, 2)
160: 
161:     '''
162:     res = []
163:     for ary in arys:
164:         ary = asanyarray(ary)
165:         if len(ary.shape) == 0:
166:             result = ary.reshape(1, 1, 1)
167:         elif len(ary.shape) == 1:
168:             result = ary[newaxis,:, newaxis]
169:         elif len(ary.shape) == 2:
170:             result = ary[:,:, newaxis]
171:         else:
172:             result = ary
173:         res.append(result)
174:     if len(res) == 1:
175:         return res[0]
176:     else:
177:         return res
178: 
179: 
180: def vstack(tup):
181:     '''
182:     Stack arrays in sequence vertically (row wise).
183: 
184:     Take a sequence of arrays and stack them vertically to make a single
185:     array. Rebuild arrays divided by `vsplit`.
186: 
187:     Parameters
188:     ----------
189:     tup : sequence of ndarrays
190:         Tuple containing arrays to be stacked. The arrays must have the same
191:         shape along all but the first axis.
192: 
193:     Returns
194:     -------
195:     stacked : ndarray
196:         The array formed by stacking the given arrays.
197: 
198:     See Also
199:     --------
200:     stack : Join a sequence of arrays along a new axis.
201:     hstack : Stack arrays in sequence horizontally (column wise).
202:     dstack : Stack arrays in sequence depth wise (along third dimension).
203:     concatenate : Join a sequence of arrays along an existing axis.
204:     vsplit : Split array into a list of multiple sub-arrays vertically.
205: 
206:     Notes
207:     -----
208:     Equivalent to ``np.concatenate(tup, axis=0)`` if `tup` contains arrays that
209:     are at least 2-dimensional.
210: 
211:     Examples
212:     --------
213:     >>> a = np.array([1, 2, 3])
214:     >>> b = np.array([2, 3, 4])
215:     >>> np.vstack((a,b))
216:     array([[1, 2, 3],
217:            [2, 3, 4]])
218: 
219:     >>> a = np.array([[1], [2], [3]])
220:     >>> b = np.array([[2], [3], [4]])
221:     >>> np.vstack((a,b))
222:     array([[1],
223:            [2],
224:            [3],
225:            [2],
226:            [3],
227:            [4]])
228: 
229:     '''
230:     return _nx.concatenate([atleast_2d(_m) for _m in tup], 0)
231: 
232: def hstack(tup):
233:     '''
234:     Stack arrays in sequence horizontally (column wise).
235: 
236:     Take a sequence of arrays and stack them horizontally to make
237:     a single array. Rebuild arrays divided by `hsplit`.
238: 
239:     Parameters
240:     ----------
241:     tup : sequence of ndarrays
242:         All arrays must have the same shape along all but the second axis.
243: 
244:     Returns
245:     -------
246:     stacked : ndarray
247:         The array formed by stacking the given arrays.
248: 
249:     See Also
250:     --------
251:     stack : Join a sequence of arrays along a new axis.
252:     vstack : Stack arrays in sequence vertically (row wise).
253:     dstack : Stack arrays in sequence depth wise (along third axis).
254:     concatenate : Join a sequence of arrays along an existing axis.
255:     hsplit : Split array along second axis.
256: 
257:     Notes
258:     -----
259:     Equivalent to ``np.concatenate(tup, axis=1)``
260: 
261:     Examples
262:     --------
263:     >>> a = np.array((1,2,3))
264:     >>> b = np.array((2,3,4))
265:     >>> np.hstack((a,b))
266:     array([1, 2, 3, 2, 3, 4])
267:     >>> a = np.array([[1],[2],[3]])
268:     >>> b = np.array([[2],[3],[4]])
269:     >>> np.hstack((a,b))
270:     array([[1, 2],
271:            [2, 3],
272:            [3, 4]])
273: 
274:     '''
275:     arrs = [atleast_1d(_m) for _m in tup]
276:     # As a special case, dimension 0 of 1-dimensional arrays is "horizontal"
277:     if arrs[0].ndim == 1:
278:         return _nx.concatenate(arrs, 0)
279:     else:
280:         return _nx.concatenate(arrs, 1)
281: 
282: def stack(arrays, axis=0):
283:     '''
284:     Join a sequence of arrays along a new axis.
285: 
286:     The `axis` parameter specifies the index of the new axis in the dimensions
287:     of the result. For example, if ``axis=0`` it will be the first dimension
288:     and if ``axis=-1`` it will be the last dimension.
289: 
290:     .. versionadded:: 1.10.0
291: 
292:     Parameters
293:     ----------
294:     arrays : sequence of array_like
295:         Each array must have the same shape.
296:     axis : int, optional
297:         The axis in the result array along which the input arrays are stacked.
298: 
299:     Returns
300:     -------
301:     stacked : ndarray
302:         The stacked array has one more dimension than the input arrays.
303: 
304:     See Also
305:     --------
306:     concatenate : Join a sequence of arrays along an existing axis.
307:     split : Split array into a list of multiple sub-arrays of equal size.
308: 
309:     Examples
310:     --------
311:     >>> arrays = [np.random.randn(3, 4) for _ in range(10)]
312:     >>> np.stack(arrays, axis=0).shape
313:     (10, 3, 4)
314: 
315:     >>> np.stack(arrays, axis=1).shape
316:     (3, 10, 4)
317: 
318:     >>> np.stack(arrays, axis=2).shape
319:     (3, 4, 10)
320: 
321:     >>> a = np.array([1, 2, 3])
322:     >>> b = np.array([2, 3, 4])
323:     >>> np.stack((a, b))
324:     array([[1, 2, 3],
325:            [2, 3, 4]])
326: 
327:     >>> np.stack((a, b), axis=-1)
328:     array([[1, 2],
329:            [2, 3],
330:            [3, 4]])
331: 
332:     '''
333:     arrays = [asanyarray(arr) for arr in arrays]
334:     if not arrays:
335:         raise ValueError('need at least one array to stack')
336: 
337:     shapes = set(arr.shape for arr in arrays)
338:     if len(shapes) != 1:
339:         raise ValueError('all input arrays must have the same shape')
340: 
341:     result_ndim = arrays[0].ndim + 1
342:     if not -result_ndim <= axis < result_ndim:
343:         msg = 'axis {0} out of bounds [-{1}, {1})'.format(axis, result_ndim)
344:         raise IndexError(msg)
345:     if axis < 0:
346:         axis += result_ndim
347: 
348:     sl = (slice(None),) * axis + (_nx.newaxis,)
349:     expanded_arrays = [arr[sl] for arr in arrays]
350:     return _nx.concatenate(expanded_arrays, axis=axis)
351: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################


# Assigning a List to a Name (line 3):
__all__ = ['atleast_1d', 'atleast_2d', 'atleast_3d', 'vstack', 'hstack', 'stack']
module_type_store.set_exportable_members(['atleast_1d', 'atleast_2d', 'atleast_3d', 'vstack', 'hstack', 'stack'])

# Obtaining an instance of the builtin type 'list' (line 3)
list_18441 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3, 10), 'list')
# Adding type elements to the builtin type 'list' instance (line 3)
# Adding element type (line 3)
str_18442 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3, 11), 'str', 'atleast_1d')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 3, 10), list_18441, str_18442)
# Adding element type (line 3)
str_18443 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3, 25), 'str', 'atleast_2d')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 3, 10), list_18441, str_18443)
# Adding element type (line 3)
str_18444 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3, 39), 'str', 'atleast_3d')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 3, 10), list_18441, str_18444)
# Adding element type (line 3)
str_18445 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3, 53), 'str', 'vstack')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 3, 10), list_18441, str_18445)
# Adding element type (line 3)
str_18446 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3, 63), 'str', 'hstack')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 3, 10), list_18441, str_18446)
# Adding element type (line 3)
str_18447 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4, 11), 'str', 'stack')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 3, 10), list_18441, str_18447)

# Assigning a type to the variable '__all__' (line 3)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 0), '__all__', list_18441)
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 6, 0))

# 'from numpy.core import _nx' statement (line 6)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/core/')
import_18448 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'numpy.core')

if (type(import_18448) is not StypyTypeError):

    if (import_18448 != 'pyd_module'):
        __import__(import_18448)
        sys_modules_18449 = sys.modules[import_18448]
        import_from_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'numpy.core', sys_modules_18449.module_type_store, module_type_store, ['numeric'])
        nest_module(stypy.reporting.localization.Localization(__file__, 6, 0), __file__, sys_modules_18449, sys_modules_18449.module_type_store, module_type_store)
    else:
        from numpy.core import numeric as _nx

        import_from_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'numpy.core', None, module_type_store, ['numeric'], [_nx])

else:
    # Assigning a type to the variable 'numpy.core' (line 6)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), 'numpy.core', import_18448)

# Adding an alias
module_type_store.add_alias('_nx', 'numeric')
remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/core/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 7, 0))

# 'from numpy.core.numeric import asanyarray, newaxis' statement (line 7)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/core/')
import_18450 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'numpy.core.numeric')

if (type(import_18450) is not StypyTypeError):

    if (import_18450 != 'pyd_module'):
        __import__(import_18450)
        sys_modules_18451 = sys.modules[import_18450]
        import_from_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'numpy.core.numeric', sys_modules_18451.module_type_store, module_type_store, ['asanyarray', 'newaxis'])
        nest_module(stypy.reporting.localization.Localization(__file__, 7, 0), __file__, sys_modules_18451, sys_modules_18451.module_type_store, module_type_store)
    else:
        from numpy.core.numeric import asanyarray, newaxis

        import_from_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'numpy.core.numeric', None, module_type_store, ['asanyarray', 'newaxis'], [asanyarray, newaxis])

else:
    # Assigning a type to the variable 'numpy.core.numeric' (line 7)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), 'numpy.core.numeric', import_18450)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/core/')


@norecursion
def atleast_1d(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'atleast_1d'
    module_type_store = module_type_store.open_function_context('atleast_1d', 9, 0, False)
    
    # Passed parameters checking function
    atleast_1d.stypy_localization = localization
    atleast_1d.stypy_type_of_self = None
    atleast_1d.stypy_type_store = module_type_store
    atleast_1d.stypy_function_name = 'atleast_1d'
    atleast_1d.stypy_param_names_list = []
    atleast_1d.stypy_varargs_param_name = 'arys'
    atleast_1d.stypy_kwargs_param_name = None
    atleast_1d.stypy_call_defaults = defaults
    atleast_1d.stypy_call_varargs = varargs
    atleast_1d.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'atleast_1d', [], 'arys', None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'atleast_1d', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'atleast_1d(...)' code ##################

    str_18452 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, (-1)), 'str', '\n    Convert inputs to arrays with at least one dimension.\n\n    Scalar inputs are converted to 1-dimensional arrays, whilst\n    higher-dimensional inputs are preserved.\n\n    Parameters\n    ----------\n    arys1, arys2, ... : array_like\n        One or more input arrays.\n\n    Returns\n    -------\n    ret : ndarray\n        An array, or sequence of arrays, each with ``a.ndim >= 1``.\n        Copies are made only if necessary.\n\n    See Also\n    --------\n    atleast_2d, atleast_3d\n\n    Examples\n    --------\n    >>> np.atleast_1d(1.0)\n    array([ 1.])\n\n    >>> x = np.arange(9.0).reshape(3,3)\n    >>> np.atleast_1d(x)\n    array([[ 0.,  1.,  2.],\n           [ 3.,  4.,  5.],\n           [ 6.,  7.,  8.]])\n    >>> np.atleast_1d(x) is x\n    True\n\n    >>> np.atleast_1d(1, [3, 4])\n    [array([1]), array([3, 4])]\n\n    ')
    
    # Assigning a List to a Name (line 48):
    
    # Obtaining an instance of the builtin type 'list' (line 48)
    list_18453 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 48, 10), 'list')
    # Adding type elements to the builtin type 'list' instance (line 48)
    
    # Assigning a type to the variable 'res' (line 48)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 48, 4), 'res', list_18453)
    
    # Getting the type of 'arys' (line 49)
    arys_18454 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 15), 'arys')
    # Testing the type of a for loop iterable (line 49)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 49, 4), arys_18454)
    # Getting the type of the for loop variable (line 49)
    for_loop_var_18455 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 49, 4), arys_18454)
    # Assigning a type to the variable 'ary' (line 49)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 49, 4), 'ary', for_loop_var_18455)
    # SSA begins for a for statement (line 49)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Call to a Name (line 50):
    
    # Call to asanyarray(...): (line 50)
    # Processing the call arguments (line 50)
    # Getting the type of 'ary' (line 50)
    ary_18457 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 25), 'ary', False)
    # Processing the call keyword arguments (line 50)
    kwargs_18458 = {}
    # Getting the type of 'asanyarray' (line 50)
    asanyarray_18456 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 14), 'asanyarray', False)
    # Calling asanyarray(args, kwargs) (line 50)
    asanyarray_call_result_18459 = invoke(stypy.reporting.localization.Localization(__file__, 50, 14), asanyarray_18456, *[ary_18457], **kwargs_18458)
    
    # Assigning a type to the variable 'ary' (line 50)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 50, 8), 'ary', asanyarray_call_result_18459)
    
    
    
    # Call to len(...): (line 51)
    # Processing the call arguments (line 51)
    # Getting the type of 'ary' (line 51)
    ary_18461 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 15), 'ary', False)
    # Obtaining the member 'shape' of a type (line 51)
    shape_18462 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 51, 15), ary_18461, 'shape')
    # Processing the call keyword arguments (line 51)
    kwargs_18463 = {}
    # Getting the type of 'len' (line 51)
    len_18460 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 11), 'len', False)
    # Calling len(args, kwargs) (line 51)
    len_call_result_18464 = invoke(stypy.reporting.localization.Localization(__file__, 51, 11), len_18460, *[shape_18462], **kwargs_18463)
    
    int_18465 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 51, 29), 'int')
    # Applying the binary operator '==' (line 51)
    result_eq_18466 = python_operator(stypy.reporting.localization.Localization(__file__, 51, 11), '==', len_call_result_18464, int_18465)
    
    # Testing the type of an if condition (line 51)
    if_condition_18467 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 51, 8), result_eq_18466)
    # Assigning a type to the variable 'if_condition_18467' (line 51)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 51, 8), 'if_condition_18467', if_condition_18467)
    # SSA begins for if statement (line 51)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 52):
    
    # Call to reshape(...): (line 52)
    # Processing the call arguments (line 52)
    int_18470 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 52, 33), 'int')
    # Processing the call keyword arguments (line 52)
    kwargs_18471 = {}
    # Getting the type of 'ary' (line 52)
    ary_18468 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 21), 'ary', False)
    # Obtaining the member 'reshape' of a type (line 52)
    reshape_18469 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 52, 21), ary_18468, 'reshape')
    # Calling reshape(args, kwargs) (line 52)
    reshape_call_result_18472 = invoke(stypy.reporting.localization.Localization(__file__, 52, 21), reshape_18469, *[int_18470], **kwargs_18471)
    
    # Assigning a type to the variable 'result' (line 52)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 52, 12), 'result', reshape_call_result_18472)
    # SSA branch for the else part of an if statement (line 51)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Name to a Name (line 54):
    # Getting the type of 'ary' (line 54)
    ary_18473 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 21), 'ary')
    # Assigning a type to the variable 'result' (line 54)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 54, 12), 'result', ary_18473)
    # SSA join for if statement (line 51)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to append(...): (line 55)
    # Processing the call arguments (line 55)
    # Getting the type of 'result' (line 55)
    result_18476 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 19), 'result', False)
    # Processing the call keyword arguments (line 55)
    kwargs_18477 = {}
    # Getting the type of 'res' (line 55)
    res_18474 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 8), 'res', False)
    # Obtaining the member 'append' of a type (line 55)
    append_18475 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 55, 8), res_18474, 'append')
    # Calling append(args, kwargs) (line 55)
    append_call_result_18478 = invoke(stypy.reporting.localization.Localization(__file__, 55, 8), append_18475, *[result_18476], **kwargs_18477)
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    
    # Call to len(...): (line 56)
    # Processing the call arguments (line 56)
    # Getting the type of 'res' (line 56)
    res_18480 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 11), 'res', False)
    # Processing the call keyword arguments (line 56)
    kwargs_18481 = {}
    # Getting the type of 'len' (line 56)
    len_18479 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 7), 'len', False)
    # Calling len(args, kwargs) (line 56)
    len_call_result_18482 = invoke(stypy.reporting.localization.Localization(__file__, 56, 7), len_18479, *[res_18480], **kwargs_18481)
    
    int_18483 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, 19), 'int')
    # Applying the binary operator '==' (line 56)
    result_eq_18484 = python_operator(stypy.reporting.localization.Localization(__file__, 56, 7), '==', len_call_result_18482, int_18483)
    
    # Testing the type of an if condition (line 56)
    if_condition_18485 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 56, 4), result_eq_18484)
    # Assigning a type to the variable 'if_condition_18485' (line 56)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 56, 4), 'if_condition_18485', if_condition_18485)
    # SSA begins for if statement (line 56)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Obtaining the type of the subscript
    int_18486 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 57, 19), 'int')
    # Getting the type of 'res' (line 57)
    res_18487 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 15), 'res')
    # Obtaining the member '__getitem__' of a type (line 57)
    getitem___18488 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 57, 15), res_18487, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 57)
    subscript_call_result_18489 = invoke(stypy.reporting.localization.Localization(__file__, 57, 15), getitem___18488, int_18486)
    
    # Assigning a type to the variable 'stypy_return_type' (line 57)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 57, 8), 'stypy_return_type', subscript_call_result_18489)
    # SSA branch for the else part of an if statement (line 56)
    module_type_store.open_ssa_branch('else')
    # Getting the type of 'res' (line 59)
    res_18490 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 15), 'res')
    # Assigning a type to the variable 'stypy_return_type' (line 59)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 59, 8), 'stypy_return_type', res_18490)
    # SSA join for if statement (line 56)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'atleast_1d(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'atleast_1d' in the type store
    # Getting the type of 'stypy_return_type' (line 9)
    stypy_return_type_18491 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 9, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_18491)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'atleast_1d'
    return stypy_return_type_18491

# Assigning a type to the variable 'atleast_1d' (line 9)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 0), 'atleast_1d', atleast_1d)

@norecursion
def atleast_2d(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'atleast_2d'
    module_type_store = module_type_store.open_function_context('atleast_2d', 61, 0, False)
    
    # Passed parameters checking function
    atleast_2d.stypy_localization = localization
    atleast_2d.stypy_type_of_self = None
    atleast_2d.stypy_type_store = module_type_store
    atleast_2d.stypy_function_name = 'atleast_2d'
    atleast_2d.stypy_param_names_list = []
    atleast_2d.stypy_varargs_param_name = 'arys'
    atleast_2d.stypy_kwargs_param_name = None
    atleast_2d.stypy_call_defaults = defaults
    atleast_2d.stypy_call_varargs = varargs
    atleast_2d.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'atleast_2d', [], 'arys', None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'atleast_2d', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'atleast_2d(...)' code ##################

    str_18492 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 97, (-1)), 'str', '\n    View inputs as arrays with at least two dimensions.\n\n    Parameters\n    ----------\n    arys1, arys2, ... : array_like\n        One or more array-like sequences.  Non-array inputs are converted\n        to arrays.  Arrays that already have two or more dimensions are\n        preserved.\n\n    Returns\n    -------\n    res, res2, ... : ndarray\n        An array, or tuple of arrays, each with ``a.ndim >= 2``.\n        Copies are avoided where possible, and views with two or more\n        dimensions are returned.\n\n    See Also\n    --------\n    atleast_1d, atleast_3d\n\n    Examples\n    --------\n    >>> np.atleast_2d(3.0)\n    array([[ 3.]])\n\n    >>> x = np.arange(3.0)\n    >>> np.atleast_2d(x)\n    array([[ 0.,  1.,  2.]])\n    >>> np.atleast_2d(x).base is x\n    True\n\n    >>> np.atleast_2d(1, [1, 2], [[1, 2]])\n    [array([[1]]), array([[1, 2]]), array([[1, 2]])]\n\n    ')
    
    # Assigning a List to a Name (line 98):
    
    # Obtaining an instance of the builtin type 'list' (line 98)
    list_18493 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 98, 10), 'list')
    # Adding type elements to the builtin type 'list' instance (line 98)
    
    # Assigning a type to the variable 'res' (line 98)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 98, 4), 'res', list_18493)
    
    # Getting the type of 'arys' (line 99)
    arys_18494 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 15), 'arys')
    # Testing the type of a for loop iterable (line 99)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 99, 4), arys_18494)
    # Getting the type of the for loop variable (line 99)
    for_loop_var_18495 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 99, 4), arys_18494)
    # Assigning a type to the variable 'ary' (line 99)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 99, 4), 'ary', for_loop_var_18495)
    # SSA begins for a for statement (line 99)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Call to a Name (line 100):
    
    # Call to asanyarray(...): (line 100)
    # Processing the call arguments (line 100)
    # Getting the type of 'ary' (line 100)
    ary_18497 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 25), 'ary', False)
    # Processing the call keyword arguments (line 100)
    kwargs_18498 = {}
    # Getting the type of 'asanyarray' (line 100)
    asanyarray_18496 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 14), 'asanyarray', False)
    # Calling asanyarray(args, kwargs) (line 100)
    asanyarray_call_result_18499 = invoke(stypy.reporting.localization.Localization(__file__, 100, 14), asanyarray_18496, *[ary_18497], **kwargs_18498)
    
    # Assigning a type to the variable 'ary' (line 100)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 100, 8), 'ary', asanyarray_call_result_18499)
    
    
    
    # Call to len(...): (line 101)
    # Processing the call arguments (line 101)
    # Getting the type of 'ary' (line 101)
    ary_18501 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 15), 'ary', False)
    # Obtaining the member 'shape' of a type (line 101)
    shape_18502 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 101, 15), ary_18501, 'shape')
    # Processing the call keyword arguments (line 101)
    kwargs_18503 = {}
    # Getting the type of 'len' (line 101)
    len_18500 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 11), 'len', False)
    # Calling len(args, kwargs) (line 101)
    len_call_result_18504 = invoke(stypy.reporting.localization.Localization(__file__, 101, 11), len_18500, *[shape_18502], **kwargs_18503)
    
    int_18505 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 101, 29), 'int')
    # Applying the binary operator '==' (line 101)
    result_eq_18506 = python_operator(stypy.reporting.localization.Localization(__file__, 101, 11), '==', len_call_result_18504, int_18505)
    
    # Testing the type of an if condition (line 101)
    if_condition_18507 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 101, 8), result_eq_18506)
    # Assigning a type to the variable 'if_condition_18507' (line 101)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 101, 8), 'if_condition_18507', if_condition_18507)
    # SSA begins for if statement (line 101)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 102):
    
    # Call to reshape(...): (line 102)
    # Processing the call arguments (line 102)
    int_18510 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 102, 33), 'int')
    int_18511 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 102, 36), 'int')
    # Processing the call keyword arguments (line 102)
    kwargs_18512 = {}
    # Getting the type of 'ary' (line 102)
    ary_18508 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 21), 'ary', False)
    # Obtaining the member 'reshape' of a type (line 102)
    reshape_18509 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 102, 21), ary_18508, 'reshape')
    # Calling reshape(args, kwargs) (line 102)
    reshape_call_result_18513 = invoke(stypy.reporting.localization.Localization(__file__, 102, 21), reshape_18509, *[int_18510, int_18511], **kwargs_18512)
    
    # Assigning a type to the variable 'result' (line 102)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 102, 12), 'result', reshape_call_result_18513)
    # SSA branch for the else part of an if statement (line 101)
    module_type_store.open_ssa_branch('else')
    
    
    
    # Call to len(...): (line 103)
    # Processing the call arguments (line 103)
    # Getting the type of 'ary' (line 103)
    ary_18515 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 17), 'ary', False)
    # Obtaining the member 'shape' of a type (line 103)
    shape_18516 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 103, 17), ary_18515, 'shape')
    # Processing the call keyword arguments (line 103)
    kwargs_18517 = {}
    # Getting the type of 'len' (line 103)
    len_18514 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 13), 'len', False)
    # Calling len(args, kwargs) (line 103)
    len_call_result_18518 = invoke(stypy.reporting.localization.Localization(__file__, 103, 13), len_18514, *[shape_18516], **kwargs_18517)
    
    int_18519 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 103, 31), 'int')
    # Applying the binary operator '==' (line 103)
    result_eq_18520 = python_operator(stypy.reporting.localization.Localization(__file__, 103, 13), '==', len_call_result_18518, int_18519)
    
    # Testing the type of an if condition (line 103)
    if_condition_18521 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 103, 13), result_eq_18520)
    # Assigning a type to the variable 'if_condition_18521' (line 103)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 103, 13), 'if_condition_18521', if_condition_18521)
    # SSA begins for if statement (line 103)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Subscript to a Name (line 104):
    
    # Obtaining the type of the subscript
    # Getting the type of 'newaxis' (line 104)
    newaxis_18522 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 25), 'newaxis')
    slice_18523 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 104, 21), None, None, None)
    # Getting the type of 'ary' (line 104)
    ary_18524 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 21), 'ary')
    # Obtaining the member '__getitem__' of a type (line 104)
    getitem___18525 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 104, 21), ary_18524, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 104)
    subscript_call_result_18526 = invoke(stypy.reporting.localization.Localization(__file__, 104, 21), getitem___18525, (newaxis_18522, slice_18523))
    
    # Assigning a type to the variable 'result' (line 104)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 104, 12), 'result', subscript_call_result_18526)
    # SSA branch for the else part of an if statement (line 103)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Name to a Name (line 106):
    # Getting the type of 'ary' (line 106)
    ary_18527 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 21), 'ary')
    # Assigning a type to the variable 'result' (line 106)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 106, 12), 'result', ary_18527)
    # SSA join for if statement (line 103)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 101)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to append(...): (line 107)
    # Processing the call arguments (line 107)
    # Getting the type of 'result' (line 107)
    result_18530 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 19), 'result', False)
    # Processing the call keyword arguments (line 107)
    kwargs_18531 = {}
    # Getting the type of 'res' (line 107)
    res_18528 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 8), 'res', False)
    # Obtaining the member 'append' of a type (line 107)
    append_18529 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 107, 8), res_18528, 'append')
    # Calling append(args, kwargs) (line 107)
    append_call_result_18532 = invoke(stypy.reporting.localization.Localization(__file__, 107, 8), append_18529, *[result_18530], **kwargs_18531)
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    
    # Call to len(...): (line 108)
    # Processing the call arguments (line 108)
    # Getting the type of 'res' (line 108)
    res_18534 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 11), 'res', False)
    # Processing the call keyword arguments (line 108)
    kwargs_18535 = {}
    # Getting the type of 'len' (line 108)
    len_18533 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 7), 'len', False)
    # Calling len(args, kwargs) (line 108)
    len_call_result_18536 = invoke(stypy.reporting.localization.Localization(__file__, 108, 7), len_18533, *[res_18534], **kwargs_18535)
    
    int_18537 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 108, 19), 'int')
    # Applying the binary operator '==' (line 108)
    result_eq_18538 = python_operator(stypy.reporting.localization.Localization(__file__, 108, 7), '==', len_call_result_18536, int_18537)
    
    # Testing the type of an if condition (line 108)
    if_condition_18539 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 108, 4), result_eq_18538)
    # Assigning a type to the variable 'if_condition_18539' (line 108)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 108, 4), 'if_condition_18539', if_condition_18539)
    # SSA begins for if statement (line 108)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Obtaining the type of the subscript
    int_18540 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 109, 19), 'int')
    # Getting the type of 'res' (line 109)
    res_18541 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 15), 'res')
    # Obtaining the member '__getitem__' of a type (line 109)
    getitem___18542 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 109, 15), res_18541, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 109)
    subscript_call_result_18543 = invoke(stypy.reporting.localization.Localization(__file__, 109, 15), getitem___18542, int_18540)
    
    # Assigning a type to the variable 'stypy_return_type' (line 109)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 109, 8), 'stypy_return_type', subscript_call_result_18543)
    # SSA branch for the else part of an if statement (line 108)
    module_type_store.open_ssa_branch('else')
    # Getting the type of 'res' (line 111)
    res_18544 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 15), 'res')
    # Assigning a type to the variable 'stypy_return_type' (line 111)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 111, 8), 'stypy_return_type', res_18544)
    # SSA join for if statement (line 108)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'atleast_2d(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'atleast_2d' in the type store
    # Getting the type of 'stypy_return_type' (line 61)
    stypy_return_type_18545 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_18545)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'atleast_2d'
    return stypy_return_type_18545

# Assigning a type to the variable 'atleast_2d' (line 61)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 61, 0), 'atleast_2d', atleast_2d)

@norecursion
def atleast_3d(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'atleast_3d'
    module_type_store = module_type_store.open_function_context('atleast_3d', 113, 0, False)
    
    # Passed parameters checking function
    atleast_3d.stypy_localization = localization
    atleast_3d.stypy_type_of_self = None
    atleast_3d.stypy_type_store = module_type_store
    atleast_3d.stypy_function_name = 'atleast_3d'
    atleast_3d.stypy_param_names_list = []
    atleast_3d.stypy_varargs_param_name = 'arys'
    atleast_3d.stypy_kwargs_param_name = None
    atleast_3d.stypy_call_defaults = defaults
    atleast_3d.stypy_call_varargs = varargs
    atleast_3d.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'atleast_3d', [], 'arys', None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'atleast_3d', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'atleast_3d(...)' code ##################

    str_18546 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 161, (-1)), 'str', '\n    View inputs as arrays with at least three dimensions.\n\n    Parameters\n    ----------\n    arys1, arys2, ... : array_like\n        One or more array-like sequences.  Non-array inputs are converted to\n        arrays.  Arrays that already have three or more dimensions are\n        preserved.\n\n    Returns\n    -------\n    res1, res2, ... : ndarray\n        An array, or tuple of arrays, each with ``a.ndim >= 3``.  Copies are\n        avoided where possible, and views with three or more dimensions are\n        returned.  For example, a 1-D array of shape ``(N,)`` becomes a view\n        of shape ``(1, N, 1)``, and a 2-D array of shape ``(M, N)`` becomes a\n        view of shape ``(M, N, 1)``.\n\n    See Also\n    --------\n    atleast_1d, atleast_2d\n\n    Examples\n    --------\n    >>> np.atleast_3d(3.0)\n    array([[[ 3.]]])\n\n    >>> x = np.arange(3.0)\n    >>> np.atleast_3d(x).shape\n    (1, 3, 1)\n\n    >>> x = np.arange(12.0).reshape(4,3)\n    >>> np.atleast_3d(x).shape\n    (4, 3, 1)\n    >>> np.atleast_3d(x).base is x\n    True\n\n    >>> for arr in np.atleast_3d([1, 2], [[1, 2]], [[[1, 2]]]):\n    ...     print(arr, arr.shape)\n    ...\n    [[[1]\n      [2]]] (1, 2, 1)\n    [[[1]\n      [2]]] (1, 2, 1)\n    [[[1 2]]] (1, 1, 2)\n\n    ')
    
    # Assigning a List to a Name (line 162):
    
    # Obtaining an instance of the builtin type 'list' (line 162)
    list_18547 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 162, 10), 'list')
    # Adding type elements to the builtin type 'list' instance (line 162)
    
    # Assigning a type to the variable 'res' (line 162)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 162, 4), 'res', list_18547)
    
    # Getting the type of 'arys' (line 163)
    arys_18548 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 15), 'arys')
    # Testing the type of a for loop iterable (line 163)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 163, 4), arys_18548)
    # Getting the type of the for loop variable (line 163)
    for_loop_var_18549 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 163, 4), arys_18548)
    # Assigning a type to the variable 'ary' (line 163)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 163, 4), 'ary', for_loop_var_18549)
    # SSA begins for a for statement (line 163)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Call to a Name (line 164):
    
    # Call to asanyarray(...): (line 164)
    # Processing the call arguments (line 164)
    # Getting the type of 'ary' (line 164)
    ary_18551 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 25), 'ary', False)
    # Processing the call keyword arguments (line 164)
    kwargs_18552 = {}
    # Getting the type of 'asanyarray' (line 164)
    asanyarray_18550 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 14), 'asanyarray', False)
    # Calling asanyarray(args, kwargs) (line 164)
    asanyarray_call_result_18553 = invoke(stypy.reporting.localization.Localization(__file__, 164, 14), asanyarray_18550, *[ary_18551], **kwargs_18552)
    
    # Assigning a type to the variable 'ary' (line 164)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 164, 8), 'ary', asanyarray_call_result_18553)
    
    
    
    # Call to len(...): (line 165)
    # Processing the call arguments (line 165)
    # Getting the type of 'ary' (line 165)
    ary_18555 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 15), 'ary', False)
    # Obtaining the member 'shape' of a type (line 165)
    shape_18556 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 165, 15), ary_18555, 'shape')
    # Processing the call keyword arguments (line 165)
    kwargs_18557 = {}
    # Getting the type of 'len' (line 165)
    len_18554 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 11), 'len', False)
    # Calling len(args, kwargs) (line 165)
    len_call_result_18558 = invoke(stypy.reporting.localization.Localization(__file__, 165, 11), len_18554, *[shape_18556], **kwargs_18557)
    
    int_18559 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 165, 29), 'int')
    # Applying the binary operator '==' (line 165)
    result_eq_18560 = python_operator(stypy.reporting.localization.Localization(__file__, 165, 11), '==', len_call_result_18558, int_18559)
    
    # Testing the type of an if condition (line 165)
    if_condition_18561 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 165, 8), result_eq_18560)
    # Assigning a type to the variable 'if_condition_18561' (line 165)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 165, 8), 'if_condition_18561', if_condition_18561)
    # SSA begins for if statement (line 165)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 166):
    
    # Call to reshape(...): (line 166)
    # Processing the call arguments (line 166)
    int_18564 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 166, 33), 'int')
    int_18565 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 166, 36), 'int')
    int_18566 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 166, 39), 'int')
    # Processing the call keyword arguments (line 166)
    kwargs_18567 = {}
    # Getting the type of 'ary' (line 166)
    ary_18562 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 21), 'ary', False)
    # Obtaining the member 'reshape' of a type (line 166)
    reshape_18563 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 166, 21), ary_18562, 'reshape')
    # Calling reshape(args, kwargs) (line 166)
    reshape_call_result_18568 = invoke(stypy.reporting.localization.Localization(__file__, 166, 21), reshape_18563, *[int_18564, int_18565, int_18566], **kwargs_18567)
    
    # Assigning a type to the variable 'result' (line 166)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 166, 12), 'result', reshape_call_result_18568)
    # SSA branch for the else part of an if statement (line 165)
    module_type_store.open_ssa_branch('else')
    
    
    
    # Call to len(...): (line 167)
    # Processing the call arguments (line 167)
    # Getting the type of 'ary' (line 167)
    ary_18570 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 17), 'ary', False)
    # Obtaining the member 'shape' of a type (line 167)
    shape_18571 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 167, 17), ary_18570, 'shape')
    # Processing the call keyword arguments (line 167)
    kwargs_18572 = {}
    # Getting the type of 'len' (line 167)
    len_18569 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 13), 'len', False)
    # Calling len(args, kwargs) (line 167)
    len_call_result_18573 = invoke(stypy.reporting.localization.Localization(__file__, 167, 13), len_18569, *[shape_18571], **kwargs_18572)
    
    int_18574 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 167, 31), 'int')
    # Applying the binary operator '==' (line 167)
    result_eq_18575 = python_operator(stypy.reporting.localization.Localization(__file__, 167, 13), '==', len_call_result_18573, int_18574)
    
    # Testing the type of an if condition (line 167)
    if_condition_18576 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 167, 13), result_eq_18575)
    # Assigning a type to the variable 'if_condition_18576' (line 167)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 167, 13), 'if_condition_18576', if_condition_18576)
    # SSA begins for if statement (line 167)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Subscript to a Name (line 168):
    
    # Obtaining the type of the subscript
    # Getting the type of 'newaxis' (line 168)
    newaxis_18577 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 25), 'newaxis')
    slice_18578 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 168, 21), None, None, None)
    # Getting the type of 'newaxis' (line 168)
    newaxis_18579 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 36), 'newaxis')
    # Getting the type of 'ary' (line 168)
    ary_18580 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 21), 'ary')
    # Obtaining the member '__getitem__' of a type (line 168)
    getitem___18581 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 168, 21), ary_18580, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 168)
    subscript_call_result_18582 = invoke(stypy.reporting.localization.Localization(__file__, 168, 21), getitem___18581, (newaxis_18577, slice_18578, newaxis_18579))
    
    # Assigning a type to the variable 'result' (line 168)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 168, 12), 'result', subscript_call_result_18582)
    # SSA branch for the else part of an if statement (line 167)
    module_type_store.open_ssa_branch('else')
    
    
    
    # Call to len(...): (line 169)
    # Processing the call arguments (line 169)
    # Getting the type of 'ary' (line 169)
    ary_18584 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 17), 'ary', False)
    # Obtaining the member 'shape' of a type (line 169)
    shape_18585 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 169, 17), ary_18584, 'shape')
    # Processing the call keyword arguments (line 169)
    kwargs_18586 = {}
    # Getting the type of 'len' (line 169)
    len_18583 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 13), 'len', False)
    # Calling len(args, kwargs) (line 169)
    len_call_result_18587 = invoke(stypy.reporting.localization.Localization(__file__, 169, 13), len_18583, *[shape_18585], **kwargs_18586)
    
    int_18588 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 169, 31), 'int')
    # Applying the binary operator '==' (line 169)
    result_eq_18589 = python_operator(stypy.reporting.localization.Localization(__file__, 169, 13), '==', len_call_result_18587, int_18588)
    
    # Testing the type of an if condition (line 169)
    if_condition_18590 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 169, 13), result_eq_18589)
    # Assigning a type to the variable 'if_condition_18590' (line 169)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 169, 13), 'if_condition_18590', if_condition_18590)
    # SSA begins for if statement (line 169)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Subscript to a Name (line 170):
    
    # Obtaining the type of the subscript
    slice_18591 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 170, 21), None, None, None)
    slice_18592 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 170, 21), None, None, None)
    # Getting the type of 'newaxis' (line 170)
    newaxis_18593 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 30), 'newaxis')
    # Getting the type of 'ary' (line 170)
    ary_18594 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 21), 'ary')
    # Obtaining the member '__getitem__' of a type (line 170)
    getitem___18595 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 170, 21), ary_18594, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 170)
    subscript_call_result_18596 = invoke(stypy.reporting.localization.Localization(__file__, 170, 21), getitem___18595, (slice_18591, slice_18592, newaxis_18593))
    
    # Assigning a type to the variable 'result' (line 170)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 170, 12), 'result', subscript_call_result_18596)
    # SSA branch for the else part of an if statement (line 169)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Name to a Name (line 172):
    # Getting the type of 'ary' (line 172)
    ary_18597 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 21), 'ary')
    # Assigning a type to the variable 'result' (line 172)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 172, 12), 'result', ary_18597)
    # SSA join for if statement (line 169)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 167)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 165)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to append(...): (line 173)
    # Processing the call arguments (line 173)
    # Getting the type of 'result' (line 173)
    result_18600 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 19), 'result', False)
    # Processing the call keyword arguments (line 173)
    kwargs_18601 = {}
    # Getting the type of 'res' (line 173)
    res_18598 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 8), 'res', False)
    # Obtaining the member 'append' of a type (line 173)
    append_18599 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 173, 8), res_18598, 'append')
    # Calling append(args, kwargs) (line 173)
    append_call_result_18602 = invoke(stypy.reporting.localization.Localization(__file__, 173, 8), append_18599, *[result_18600], **kwargs_18601)
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    
    # Call to len(...): (line 174)
    # Processing the call arguments (line 174)
    # Getting the type of 'res' (line 174)
    res_18604 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 11), 'res', False)
    # Processing the call keyword arguments (line 174)
    kwargs_18605 = {}
    # Getting the type of 'len' (line 174)
    len_18603 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 7), 'len', False)
    # Calling len(args, kwargs) (line 174)
    len_call_result_18606 = invoke(stypy.reporting.localization.Localization(__file__, 174, 7), len_18603, *[res_18604], **kwargs_18605)
    
    int_18607 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 174, 19), 'int')
    # Applying the binary operator '==' (line 174)
    result_eq_18608 = python_operator(stypy.reporting.localization.Localization(__file__, 174, 7), '==', len_call_result_18606, int_18607)
    
    # Testing the type of an if condition (line 174)
    if_condition_18609 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 174, 4), result_eq_18608)
    # Assigning a type to the variable 'if_condition_18609' (line 174)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 174, 4), 'if_condition_18609', if_condition_18609)
    # SSA begins for if statement (line 174)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Obtaining the type of the subscript
    int_18610 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 175, 19), 'int')
    # Getting the type of 'res' (line 175)
    res_18611 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 15), 'res')
    # Obtaining the member '__getitem__' of a type (line 175)
    getitem___18612 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 175, 15), res_18611, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 175)
    subscript_call_result_18613 = invoke(stypy.reporting.localization.Localization(__file__, 175, 15), getitem___18612, int_18610)
    
    # Assigning a type to the variable 'stypy_return_type' (line 175)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 175, 8), 'stypy_return_type', subscript_call_result_18613)
    # SSA branch for the else part of an if statement (line 174)
    module_type_store.open_ssa_branch('else')
    # Getting the type of 'res' (line 177)
    res_18614 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 15), 'res')
    # Assigning a type to the variable 'stypy_return_type' (line 177)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 177, 8), 'stypy_return_type', res_18614)
    # SSA join for if statement (line 174)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'atleast_3d(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'atleast_3d' in the type store
    # Getting the type of 'stypy_return_type' (line 113)
    stypy_return_type_18615 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_18615)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'atleast_3d'
    return stypy_return_type_18615

# Assigning a type to the variable 'atleast_3d' (line 113)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 113, 0), 'atleast_3d', atleast_3d)

@norecursion
def vstack(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'vstack'
    module_type_store = module_type_store.open_function_context('vstack', 180, 0, False)
    
    # Passed parameters checking function
    vstack.stypy_localization = localization
    vstack.stypy_type_of_self = None
    vstack.stypy_type_store = module_type_store
    vstack.stypy_function_name = 'vstack'
    vstack.stypy_param_names_list = ['tup']
    vstack.stypy_varargs_param_name = None
    vstack.stypy_kwargs_param_name = None
    vstack.stypy_call_defaults = defaults
    vstack.stypy_call_varargs = varargs
    vstack.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'vstack', ['tup'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'vstack', localization, ['tup'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'vstack(...)' code ##################

    str_18616 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 229, (-1)), 'str', '\n    Stack arrays in sequence vertically (row wise).\n\n    Take a sequence of arrays and stack them vertically to make a single\n    array. Rebuild arrays divided by `vsplit`.\n\n    Parameters\n    ----------\n    tup : sequence of ndarrays\n        Tuple containing arrays to be stacked. The arrays must have the same\n        shape along all but the first axis.\n\n    Returns\n    -------\n    stacked : ndarray\n        The array formed by stacking the given arrays.\n\n    See Also\n    --------\n    stack : Join a sequence of arrays along a new axis.\n    hstack : Stack arrays in sequence horizontally (column wise).\n    dstack : Stack arrays in sequence depth wise (along third dimension).\n    concatenate : Join a sequence of arrays along an existing axis.\n    vsplit : Split array into a list of multiple sub-arrays vertically.\n\n    Notes\n    -----\n    Equivalent to ``np.concatenate(tup, axis=0)`` if `tup` contains arrays that\n    are at least 2-dimensional.\n\n    Examples\n    --------\n    >>> a = np.array([1, 2, 3])\n    >>> b = np.array([2, 3, 4])\n    >>> np.vstack((a,b))\n    array([[1, 2, 3],\n           [2, 3, 4]])\n\n    >>> a = np.array([[1], [2], [3]])\n    >>> b = np.array([[2], [3], [4]])\n    >>> np.vstack((a,b))\n    array([[1],\n           [2],\n           [3],\n           [2],\n           [3],\n           [4]])\n\n    ')
    
    # Call to concatenate(...): (line 230)
    # Processing the call arguments (line 230)
    # Calculating list comprehension
    # Calculating comprehension expression
    # Getting the type of 'tup' (line 230)
    tup_18623 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 53), 'tup', False)
    comprehension_18624 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 230, 28), tup_18623)
    # Assigning a type to the variable '_m' (line 230)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 230, 28), '_m', comprehension_18624)
    
    # Call to atleast_2d(...): (line 230)
    # Processing the call arguments (line 230)
    # Getting the type of '_m' (line 230)
    _m_18620 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 39), '_m', False)
    # Processing the call keyword arguments (line 230)
    kwargs_18621 = {}
    # Getting the type of 'atleast_2d' (line 230)
    atleast_2d_18619 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 28), 'atleast_2d', False)
    # Calling atleast_2d(args, kwargs) (line 230)
    atleast_2d_call_result_18622 = invoke(stypy.reporting.localization.Localization(__file__, 230, 28), atleast_2d_18619, *[_m_18620], **kwargs_18621)
    
    list_18625 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 230, 28), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 230, 28), list_18625, atleast_2d_call_result_18622)
    int_18626 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 230, 59), 'int')
    # Processing the call keyword arguments (line 230)
    kwargs_18627 = {}
    # Getting the type of '_nx' (line 230)
    _nx_18617 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 11), '_nx', False)
    # Obtaining the member 'concatenate' of a type (line 230)
    concatenate_18618 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 230, 11), _nx_18617, 'concatenate')
    # Calling concatenate(args, kwargs) (line 230)
    concatenate_call_result_18628 = invoke(stypy.reporting.localization.Localization(__file__, 230, 11), concatenate_18618, *[list_18625, int_18626], **kwargs_18627)
    
    # Assigning a type to the variable 'stypy_return_type' (line 230)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 230, 4), 'stypy_return_type', concatenate_call_result_18628)
    
    # ################# End of 'vstack(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'vstack' in the type store
    # Getting the type of 'stypy_return_type' (line 180)
    stypy_return_type_18629 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_18629)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'vstack'
    return stypy_return_type_18629

# Assigning a type to the variable 'vstack' (line 180)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 180, 0), 'vstack', vstack)

@norecursion
def hstack(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'hstack'
    module_type_store = module_type_store.open_function_context('hstack', 232, 0, False)
    
    # Passed parameters checking function
    hstack.stypy_localization = localization
    hstack.stypy_type_of_self = None
    hstack.stypy_type_store = module_type_store
    hstack.stypy_function_name = 'hstack'
    hstack.stypy_param_names_list = ['tup']
    hstack.stypy_varargs_param_name = None
    hstack.stypy_kwargs_param_name = None
    hstack.stypy_call_defaults = defaults
    hstack.stypy_call_varargs = varargs
    hstack.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'hstack', ['tup'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'hstack', localization, ['tup'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'hstack(...)' code ##################

    str_18630 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 274, (-1)), 'str', '\n    Stack arrays in sequence horizontally (column wise).\n\n    Take a sequence of arrays and stack them horizontally to make\n    a single array. Rebuild arrays divided by `hsplit`.\n\n    Parameters\n    ----------\n    tup : sequence of ndarrays\n        All arrays must have the same shape along all but the second axis.\n\n    Returns\n    -------\n    stacked : ndarray\n        The array formed by stacking the given arrays.\n\n    See Also\n    --------\n    stack : Join a sequence of arrays along a new axis.\n    vstack : Stack arrays in sequence vertically (row wise).\n    dstack : Stack arrays in sequence depth wise (along third axis).\n    concatenate : Join a sequence of arrays along an existing axis.\n    hsplit : Split array along second axis.\n\n    Notes\n    -----\n    Equivalent to ``np.concatenate(tup, axis=1)``\n\n    Examples\n    --------\n    >>> a = np.array((1,2,3))\n    >>> b = np.array((2,3,4))\n    >>> np.hstack((a,b))\n    array([1, 2, 3, 2, 3, 4])\n    >>> a = np.array([[1],[2],[3]])\n    >>> b = np.array([[2],[3],[4]])\n    >>> np.hstack((a,b))\n    array([[1, 2],\n           [2, 3],\n           [3, 4]])\n\n    ')
    
    # Assigning a ListComp to a Name (line 275):
    # Calculating list comprehension
    # Calculating comprehension expression
    # Getting the type of 'tup' (line 275)
    tup_18635 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 275, 37), 'tup')
    comprehension_18636 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 275, 12), tup_18635)
    # Assigning a type to the variable '_m' (line 275)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 275, 12), '_m', comprehension_18636)
    
    # Call to atleast_1d(...): (line 275)
    # Processing the call arguments (line 275)
    # Getting the type of '_m' (line 275)
    _m_18632 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 275, 23), '_m', False)
    # Processing the call keyword arguments (line 275)
    kwargs_18633 = {}
    # Getting the type of 'atleast_1d' (line 275)
    atleast_1d_18631 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 275, 12), 'atleast_1d', False)
    # Calling atleast_1d(args, kwargs) (line 275)
    atleast_1d_call_result_18634 = invoke(stypy.reporting.localization.Localization(__file__, 275, 12), atleast_1d_18631, *[_m_18632], **kwargs_18633)
    
    list_18637 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 275, 12), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 275, 12), list_18637, atleast_1d_call_result_18634)
    # Assigning a type to the variable 'arrs' (line 275)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 275, 4), 'arrs', list_18637)
    
    
    
    # Obtaining the type of the subscript
    int_18638 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 277, 12), 'int')
    # Getting the type of 'arrs' (line 277)
    arrs_18639 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 277, 7), 'arrs')
    # Obtaining the member '__getitem__' of a type (line 277)
    getitem___18640 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 277, 7), arrs_18639, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 277)
    subscript_call_result_18641 = invoke(stypy.reporting.localization.Localization(__file__, 277, 7), getitem___18640, int_18638)
    
    # Obtaining the member 'ndim' of a type (line 277)
    ndim_18642 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 277, 7), subscript_call_result_18641, 'ndim')
    int_18643 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 277, 23), 'int')
    # Applying the binary operator '==' (line 277)
    result_eq_18644 = python_operator(stypy.reporting.localization.Localization(__file__, 277, 7), '==', ndim_18642, int_18643)
    
    # Testing the type of an if condition (line 277)
    if_condition_18645 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 277, 4), result_eq_18644)
    # Assigning a type to the variable 'if_condition_18645' (line 277)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 277, 4), 'if_condition_18645', if_condition_18645)
    # SSA begins for if statement (line 277)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to concatenate(...): (line 278)
    # Processing the call arguments (line 278)
    # Getting the type of 'arrs' (line 278)
    arrs_18648 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 278, 31), 'arrs', False)
    int_18649 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 278, 37), 'int')
    # Processing the call keyword arguments (line 278)
    kwargs_18650 = {}
    # Getting the type of '_nx' (line 278)
    _nx_18646 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 278, 15), '_nx', False)
    # Obtaining the member 'concatenate' of a type (line 278)
    concatenate_18647 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 278, 15), _nx_18646, 'concatenate')
    # Calling concatenate(args, kwargs) (line 278)
    concatenate_call_result_18651 = invoke(stypy.reporting.localization.Localization(__file__, 278, 15), concatenate_18647, *[arrs_18648, int_18649], **kwargs_18650)
    
    # Assigning a type to the variable 'stypy_return_type' (line 278)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 278, 8), 'stypy_return_type', concatenate_call_result_18651)
    # SSA branch for the else part of an if statement (line 277)
    module_type_store.open_ssa_branch('else')
    
    # Call to concatenate(...): (line 280)
    # Processing the call arguments (line 280)
    # Getting the type of 'arrs' (line 280)
    arrs_18654 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 280, 31), 'arrs', False)
    int_18655 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 280, 37), 'int')
    # Processing the call keyword arguments (line 280)
    kwargs_18656 = {}
    # Getting the type of '_nx' (line 280)
    _nx_18652 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 280, 15), '_nx', False)
    # Obtaining the member 'concatenate' of a type (line 280)
    concatenate_18653 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 280, 15), _nx_18652, 'concatenate')
    # Calling concatenate(args, kwargs) (line 280)
    concatenate_call_result_18657 = invoke(stypy.reporting.localization.Localization(__file__, 280, 15), concatenate_18653, *[arrs_18654, int_18655], **kwargs_18656)
    
    # Assigning a type to the variable 'stypy_return_type' (line 280)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 280, 8), 'stypy_return_type', concatenate_call_result_18657)
    # SSA join for if statement (line 277)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'hstack(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'hstack' in the type store
    # Getting the type of 'stypy_return_type' (line 232)
    stypy_return_type_18658 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_18658)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'hstack'
    return stypy_return_type_18658

# Assigning a type to the variable 'hstack' (line 232)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 232, 0), 'hstack', hstack)

@norecursion
def stack(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    int_18659 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 282, 23), 'int')
    defaults = [int_18659]
    # Create a new context for function 'stack'
    module_type_store = module_type_store.open_function_context('stack', 282, 0, False)
    
    # Passed parameters checking function
    stack.stypy_localization = localization
    stack.stypy_type_of_self = None
    stack.stypy_type_store = module_type_store
    stack.stypy_function_name = 'stack'
    stack.stypy_param_names_list = ['arrays', 'axis']
    stack.stypy_varargs_param_name = None
    stack.stypy_kwargs_param_name = None
    stack.stypy_call_defaults = defaults
    stack.stypy_call_varargs = varargs
    stack.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'stack', ['arrays', 'axis'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'stack', localization, ['arrays', 'axis'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'stack(...)' code ##################

    str_18660 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 332, (-1)), 'str', '\n    Join a sequence of arrays along a new axis.\n\n    The `axis` parameter specifies the index of the new axis in the dimensions\n    of the result. For example, if ``axis=0`` it will be the first dimension\n    and if ``axis=-1`` it will be the last dimension.\n\n    .. versionadded:: 1.10.0\n\n    Parameters\n    ----------\n    arrays : sequence of array_like\n        Each array must have the same shape.\n    axis : int, optional\n        The axis in the result array along which the input arrays are stacked.\n\n    Returns\n    -------\n    stacked : ndarray\n        The stacked array has one more dimension than the input arrays.\n\n    See Also\n    --------\n    concatenate : Join a sequence of arrays along an existing axis.\n    split : Split array into a list of multiple sub-arrays of equal size.\n\n    Examples\n    --------\n    >>> arrays = [np.random.randn(3, 4) for _ in range(10)]\n    >>> np.stack(arrays, axis=0).shape\n    (10, 3, 4)\n\n    >>> np.stack(arrays, axis=1).shape\n    (3, 10, 4)\n\n    >>> np.stack(arrays, axis=2).shape\n    (3, 4, 10)\n\n    >>> a = np.array([1, 2, 3])\n    >>> b = np.array([2, 3, 4])\n    >>> np.stack((a, b))\n    array([[1, 2, 3],\n           [2, 3, 4]])\n\n    >>> np.stack((a, b), axis=-1)\n    array([[1, 2],\n           [2, 3],\n           [3, 4]])\n\n    ')
    
    # Assigning a ListComp to a Name (line 333):
    # Calculating list comprehension
    # Calculating comprehension expression
    # Getting the type of 'arrays' (line 333)
    arrays_18665 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 333, 41), 'arrays')
    comprehension_18666 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 333, 14), arrays_18665)
    # Assigning a type to the variable 'arr' (line 333)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 333, 14), 'arr', comprehension_18666)
    
    # Call to asanyarray(...): (line 333)
    # Processing the call arguments (line 333)
    # Getting the type of 'arr' (line 333)
    arr_18662 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 333, 25), 'arr', False)
    # Processing the call keyword arguments (line 333)
    kwargs_18663 = {}
    # Getting the type of 'asanyarray' (line 333)
    asanyarray_18661 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 333, 14), 'asanyarray', False)
    # Calling asanyarray(args, kwargs) (line 333)
    asanyarray_call_result_18664 = invoke(stypy.reporting.localization.Localization(__file__, 333, 14), asanyarray_18661, *[arr_18662], **kwargs_18663)
    
    list_18667 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 333, 14), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 333, 14), list_18667, asanyarray_call_result_18664)
    # Assigning a type to the variable 'arrays' (line 333)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 333, 4), 'arrays', list_18667)
    
    
    # Getting the type of 'arrays' (line 334)
    arrays_18668 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 334, 11), 'arrays')
    # Applying the 'not' unary operator (line 334)
    result_not__18669 = python_operator(stypy.reporting.localization.Localization(__file__, 334, 7), 'not', arrays_18668)
    
    # Testing the type of an if condition (line 334)
    if_condition_18670 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 334, 4), result_not__18669)
    # Assigning a type to the variable 'if_condition_18670' (line 334)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 334, 4), 'if_condition_18670', if_condition_18670)
    # SSA begins for if statement (line 334)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 335)
    # Processing the call arguments (line 335)
    str_18672 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 335, 25), 'str', 'need at least one array to stack')
    # Processing the call keyword arguments (line 335)
    kwargs_18673 = {}
    # Getting the type of 'ValueError' (line 335)
    ValueError_18671 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 335, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 335)
    ValueError_call_result_18674 = invoke(stypy.reporting.localization.Localization(__file__, 335, 14), ValueError_18671, *[str_18672], **kwargs_18673)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 335, 8), ValueError_call_result_18674, 'raise parameter', BaseException)
    # SSA join for if statement (line 334)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 337):
    
    # Call to set(...): (line 337)
    # Processing the call arguments (line 337)
    # Calculating generator expression
    module_type_store = module_type_store.open_function_context('list comprehension expression', 337, 17, True)
    # Calculating comprehension expression
    # Getting the type of 'arrays' (line 337)
    arrays_18678 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 337, 38), 'arrays', False)
    comprehension_18679 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 337, 17), arrays_18678)
    # Assigning a type to the variable 'arr' (line 337)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 337, 17), 'arr', comprehension_18679)
    # Getting the type of 'arr' (line 337)
    arr_18676 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 337, 17), 'arr', False)
    # Obtaining the member 'shape' of a type (line 337)
    shape_18677 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 337, 17), arr_18676, 'shape')
    list_18680 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 337, 17), 'list')
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 337, 17), list_18680, shape_18677)
    # Processing the call keyword arguments (line 337)
    kwargs_18681 = {}
    # Getting the type of 'set' (line 337)
    set_18675 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 337, 13), 'set', False)
    # Calling set(args, kwargs) (line 337)
    set_call_result_18682 = invoke(stypy.reporting.localization.Localization(__file__, 337, 13), set_18675, *[list_18680], **kwargs_18681)
    
    # Assigning a type to the variable 'shapes' (line 337)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 337, 4), 'shapes', set_call_result_18682)
    
    
    
    # Call to len(...): (line 338)
    # Processing the call arguments (line 338)
    # Getting the type of 'shapes' (line 338)
    shapes_18684 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 338, 11), 'shapes', False)
    # Processing the call keyword arguments (line 338)
    kwargs_18685 = {}
    # Getting the type of 'len' (line 338)
    len_18683 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 338, 7), 'len', False)
    # Calling len(args, kwargs) (line 338)
    len_call_result_18686 = invoke(stypy.reporting.localization.Localization(__file__, 338, 7), len_18683, *[shapes_18684], **kwargs_18685)
    
    int_18687 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 338, 22), 'int')
    # Applying the binary operator '!=' (line 338)
    result_ne_18688 = python_operator(stypy.reporting.localization.Localization(__file__, 338, 7), '!=', len_call_result_18686, int_18687)
    
    # Testing the type of an if condition (line 338)
    if_condition_18689 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 338, 4), result_ne_18688)
    # Assigning a type to the variable 'if_condition_18689' (line 338)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 338, 4), 'if_condition_18689', if_condition_18689)
    # SSA begins for if statement (line 338)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 339)
    # Processing the call arguments (line 339)
    str_18691 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 339, 25), 'str', 'all input arrays must have the same shape')
    # Processing the call keyword arguments (line 339)
    kwargs_18692 = {}
    # Getting the type of 'ValueError' (line 339)
    ValueError_18690 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 339, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 339)
    ValueError_call_result_18693 = invoke(stypy.reporting.localization.Localization(__file__, 339, 14), ValueError_18690, *[str_18691], **kwargs_18692)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 339, 8), ValueError_call_result_18693, 'raise parameter', BaseException)
    # SSA join for if statement (line 338)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a BinOp to a Name (line 341):
    
    # Obtaining the type of the subscript
    int_18694 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 341, 25), 'int')
    # Getting the type of 'arrays' (line 341)
    arrays_18695 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 341, 18), 'arrays')
    # Obtaining the member '__getitem__' of a type (line 341)
    getitem___18696 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 341, 18), arrays_18695, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 341)
    subscript_call_result_18697 = invoke(stypy.reporting.localization.Localization(__file__, 341, 18), getitem___18696, int_18694)
    
    # Obtaining the member 'ndim' of a type (line 341)
    ndim_18698 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 341, 18), subscript_call_result_18697, 'ndim')
    int_18699 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 341, 35), 'int')
    # Applying the binary operator '+' (line 341)
    result_add_18700 = python_operator(stypy.reporting.localization.Localization(__file__, 341, 18), '+', ndim_18698, int_18699)
    
    # Assigning a type to the variable 'result_ndim' (line 341)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 341, 4), 'result_ndim', result_add_18700)
    
    
    
    
    # Getting the type of 'result_ndim' (line 342)
    result_ndim_18701 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 342, 12), 'result_ndim')
    # Applying the 'usub' unary operator (line 342)
    result___neg___18702 = python_operator(stypy.reporting.localization.Localization(__file__, 342, 11), 'usub', result_ndim_18701)
    
    # Getting the type of 'axis' (line 342)
    axis_18703 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 342, 27), 'axis')
    # Applying the binary operator '<=' (line 342)
    result_le_18704 = python_operator(stypy.reporting.localization.Localization(__file__, 342, 11), '<=', result___neg___18702, axis_18703)
    # Getting the type of 'result_ndim' (line 342)
    result_ndim_18705 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 342, 34), 'result_ndim')
    # Applying the binary operator '<' (line 342)
    result_lt_18706 = python_operator(stypy.reporting.localization.Localization(__file__, 342, 11), '<', axis_18703, result_ndim_18705)
    # Applying the binary operator '&' (line 342)
    result_and__18707 = python_operator(stypy.reporting.localization.Localization(__file__, 342, 11), '&', result_le_18704, result_lt_18706)
    
    # Applying the 'not' unary operator (line 342)
    result_not__18708 = python_operator(stypy.reporting.localization.Localization(__file__, 342, 7), 'not', result_and__18707)
    
    # Testing the type of an if condition (line 342)
    if_condition_18709 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 342, 4), result_not__18708)
    # Assigning a type to the variable 'if_condition_18709' (line 342)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 342, 4), 'if_condition_18709', if_condition_18709)
    # SSA begins for if statement (line 342)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 343):
    
    # Call to format(...): (line 343)
    # Processing the call arguments (line 343)
    # Getting the type of 'axis' (line 343)
    axis_18712 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 343, 58), 'axis', False)
    # Getting the type of 'result_ndim' (line 343)
    result_ndim_18713 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 343, 64), 'result_ndim', False)
    # Processing the call keyword arguments (line 343)
    kwargs_18714 = {}
    str_18710 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 343, 14), 'str', 'axis {0} out of bounds [-{1}, {1})')
    # Obtaining the member 'format' of a type (line 343)
    format_18711 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 343, 14), str_18710, 'format')
    # Calling format(args, kwargs) (line 343)
    format_call_result_18715 = invoke(stypy.reporting.localization.Localization(__file__, 343, 14), format_18711, *[axis_18712, result_ndim_18713], **kwargs_18714)
    
    # Assigning a type to the variable 'msg' (line 343)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 343, 8), 'msg', format_call_result_18715)
    
    # Call to IndexError(...): (line 344)
    # Processing the call arguments (line 344)
    # Getting the type of 'msg' (line 344)
    msg_18717 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 344, 25), 'msg', False)
    # Processing the call keyword arguments (line 344)
    kwargs_18718 = {}
    # Getting the type of 'IndexError' (line 344)
    IndexError_18716 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 344, 14), 'IndexError', False)
    # Calling IndexError(args, kwargs) (line 344)
    IndexError_call_result_18719 = invoke(stypy.reporting.localization.Localization(__file__, 344, 14), IndexError_18716, *[msg_18717], **kwargs_18718)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 344, 8), IndexError_call_result_18719, 'raise parameter', BaseException)
    # SSA join for if statement (line 342)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'axis' (line 345)
    axis_18720 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 345, 7), 'axis')
    int_18721 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 345, 14), 'int')
    # Applying the binary operator '<' (line 345)
    result_lt_18722 = python_operator(stypy.reporting.localization.Localization(__file__, 345, 7), '<', axis_18720, int_18721)
    
    # Testing the type of an if condition (line 345)
    if_condition_18723 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 345, 4), result_lt_18722)
    # Assigning a type to the variable 'if_condition_18723' (line 345)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 345, 4), 'if_condition_18723', if_condition_18723)
    # SSA begins for if statement (line 345)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Getting the type of 'axis' (line 346)
    axis_18724 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 346, 8), 'axis')
    # Getting the type of 'result_ndim' (line 346)
    result_ndim_18725 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 346, 16), 'result_ndim')
    # Applying the binary operator '+=' (line 346)
    result_iadd_18726 = python_operator(stypy.reporting.localization.Localization(__file__, 346, 8), '+=', axis_18724, result_ndim_18725)
    # Assigning a type to the variable 'axis' (line 346)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 346, 8), 'axis', result_iadd_18726)
    
    # SSA join for if statement (line 345)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a BinOp to a Name (line 348):
    
    # Obtaining an instance of the builtin type 'tuple' (line 348)
    tuple_18727 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 348, 10), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 348)
    # Adding element type (line 348)
    
    # Call to slice(...): (line 348)
    # Processing the call arguments (line 348)
    # Getting the type of 'None' (line 348)
    None_18729 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 348, 16), 'None', False)
    # Processing the call keyword arguments (line 348)
    kwargs_18730 = {}
    # Getting the type of 'slice' (line 348)
    slice_18728 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 348, 10), 'slice', False)
    # Calling slice(args, kwargs) (line 348)
    slice_call_result_18731 = invoke(stypy.reporting.localization.Localization(__file__, 348, 10), slice_18728, *[None_18729], **kwargs_18730)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 348, 10), tuple_18727, slice_call_result_18731)
    
    # Getting the type of 'axis' (line 348)
    axis_18732 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 348, 26), 'axis')
    # Applying the binary operator '*' (line 348)
    result_mul_18733 = python_operator(stypy.reporting.localization.Localization(__file__, 348, 9), '*', tuple_18727, axis_18732)
    
    
    # Obtaining an instance of the builtin type 'tuple' (line 348)
    tuple_18734 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 348, 34), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 348)
    # Adding element type (line 348)
    # Getting the type of '_nx' (line 348)
    _nx_18735 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 348, 34), '_nx')
    # Obtaining the member 'newaxis' of a type (line 348)
    newaxis_18736 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 348, 34), _nx_18735, 'newaxis')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 348, 34), tuple_18734, newaxis_18736)
    
    # Applying the binary operator '+' (line 348)
    result_add_18737 = python_operator(stypy.reporting.localization.Localization(__file__, 348, 9), '+', result_mul_18733, tuple_18734)
    
    # Assigning a type to the variable 'sl' (line 348)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 348, 4), 'sl', result_add_18737)
    
    # Assigning a ListComp to a Name (line 349):
    # Calculating list comprehension
    # Calculating comprehension expression
    # Getting the type of 'arrays' (line 349)
    arrays_18742 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 349, 42), 'arrays')
    comprehension_18743 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 349, 23), arrays_18742)
    # Assigning a type to the variable 'arr' (line 349)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 349, 23), 'arr', comprehension_18743)
    
    # Obtaining the type of the subscript
    # Getting the type of 'sl' (line 349)
    sl_18738 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 349, 27), 'sl')
    # Getting the type of 'arr' (line 349)
    arr_18739 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 349, 23), 'arr')
    # Obtaining the member '__getitem__' of a type (line 349)
    getitem___18740 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 349, 23), arr_18739, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 349)
    subscript_call_result_18741 = invoke(stypy.reporting.localization.Localization(__file__, 349, 23), getitem___18740, sl_18738)
    
    list_18744 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 349, 23), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 349, 23), list_18744, subscript_call_result_18741)
    # Assigning a type to the variable 'expanded_arrays' (line 349)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 349, 4), 'expanded_arrays', list_18744)
    
    # Call to concatenate(...): (line 350)
    # Processing the call arguments (line 350)
    # Getting the type of 'expanded_arrays' (line 350)
    expanded_arrays_18747 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 350, 27), 'expanded_arrays', False)
    # Processing the call keyword arguments (line 350)
    # Getting the type of 'axis' (line 350)
    axis_18748 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 350, 49), 'axis', False)
    keyword_18749 = axis_18748
    kwargs_18750 = {'axis': keyword_18749}
    # Getting the type of '_nx' (line 350)
    _nx_18745 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 350, 11), '_nx', False)
    # Obtaining the member 'concatenate' of a type (line 350)
    concatenate_18746 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 350, 11), _nx_18745, 'concatenate')
    # Calling concatenate(args, kwargs) (line 350)
    concatenate_call_result_18751 = invoke(stypy.reporting.localization.Localization(__file__, 350, 11), concatenate_18746, *[expanded_arrays_18747], **kwargs_18750)
    
    # Assigning a type to the variable 'stypy_return_type' (line 350)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 350, 4), 'stypy_return_type', concatenate_call_result_18751)
    
    # ################# End of 'stack(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'stack' in the type store
    # Getting the type of 'stypy_return_type' (line 282)
    stypy_return_type_18752 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 282, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_18752)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'stack'
    return stypy_return_type_18752

# Assigning a type to the variable 'stack' (line 282)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 282, 0), 'stack', stack)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
