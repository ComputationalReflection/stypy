
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: '''
2: Set operations for 1D numeric arrays based on sorting.
3: 
4: :Contains:
5:   ediff1d,
6:   unique,
7:   intersect1d,
8:   setxor1d,
9:   in1d,
10:   union1d,
11:   setdiff1d
12: 
13: :Notes:
14: 
15: For floating point arrays, inaccurate results may appear due to usual round-off
16: and floating point comparison issues.
17: 
18: Speed could be gained in some operations by an implementation of
19: sort(), that can provide directly the permutation vectors, avoiding
20: thus calls to argsort().
21: 
22: To do: Optionally return indices analogously to unique for all functions.
23: 
24: :Author: Robert Cimrman
25: 
26: '''
27: from __future__ import division, absolute_import, print_function
28: 
29: import numpy as np
30: 
31: 
32: __all__ = [
33:     'ediff1d', 'intersect1d', 'setxor1d', 'union1d', 'setdiff1d', 'unique',
34:     'in1d'
35:     ]
36: 
37: 
38: def ediff1d(ary, to_end=None, to_begin=None):
39:     '''
40:     The differences between consecutive elements of an array.
41: 
42:     Parameters
43:     ----------
44:     ary : array_like
45:         If necessary, will be flattened before the differences are taken.
46:     to_end : array_like, optional
47:         Number(s) to append at the end of the returned differences.
48:     to_begin : array_like, optional
49:         Number(s) to prepend at the beginning of the returned differences.
50: 
51:     Returns
52:     -------
53:     ediff1d : ndarray
54:         The differences. Loosely, this is ``ary.flat[1:] - ary.flat[:-1]``.
55: 
56:     See Also
57:     --------
58:     diff, gradient
59: 
60:     Notes
61:     -----
62:     When applied to masked arrays, this function drops the mask information
63:     if the `to_begin` and/or `to_end` parameters are used.
64: 
65:     Examples
66:     --------
67:     >>> x = np.array([1, 2, 4, 7, 0])
68:     >>> np.ediff1d(x)
69:     array([ 1,  2,  3, -7])
70: 
71:     >>> np.ediff1d(x, to_begin=-99, to_end=np.array([88, 99]))
72:     array([-99,   1,   2,   3,  -7,  88,  99])
73: 
74:     The returned array is always 1D.
75: 
76:     >>> y = [[1, 2, 4], [1, 6, 24]]
77:     >>> np.ediff1d(y)
78:     array([ 1,  2, -3,  5, 18])
79: 
80:     '''
81:     ary = np.asanyarray(ary).flat
82:     ed = ary[1:] - ary[:-1]
83:     arrays = [ed]
84:     if to_begin is not None:
85:         arrays.insert(0, to_begin)
86:     if to_end is not None:
87:         arrays.append(to_end)
88: 
89:     if len(arrays) != 1:
90:         # We'll save ourselves a copy of a potentially large array in
91:         # the common case where neither to_begin or to_end was given.
92:         ed = np.hstack(arrays)
93: 
94:     return ed
95: 
96: def unique(ar, return_index=False, return_inverse=False, return_counts=False):
97:     '''
98:     Find the unique elements of an array.
99: 
100:     Returns the sorted unique elements of an array. There are three optional
101:     outputs in addition to the unique elements: the indices of the input array
102:     that give the unique values, the indices of the unique array that
103:     reconstruct the input array, and the number of times each unique value
104:     comes up in the input array.
105: 
106:     Parameters
107:     ----------
108:     ar : array_like
109:         Input array. This will be flattened if it is not already 1-D.
110:     return_index : bool, optional
111:         If True, also return the indices of `ar` that result in the unique
112:         array.
113:     return_inverse : bool, optional
114:         If True, also return the indices of the unique array that can be used
115:         to reconstruct `ar`.
116:     return_counts : bool, optional
117:         If True, also return the number of times each unique value comes up
118:         in `ar`.
119: 
120:         .. versionadded:: 1.9.0
121: 
122:     Returns
123:     -------
124:     unique : ndarray
125:         The sorted unique values.
126:     unique_indices : ndarray, optional
127:         The indices of the first occurrences of the unique values in the
128:         (flattened) original array. Only provided if `return_index` is True.
129:     unique_inverse : ndarray, optional
130:         The indices to reconstruct the (flattened) original array from the
131:         unique array. Only provided if `return_inverse` is True.
132:     unique_counts : ndarray, optional
133:         The number of times each of the unique values comes up in the
134:         original array. Only provided if `return_counts` is True.
135: 
136:         .. versionadded:: 1.9.0
137: 
138:     See Also
139:     --------
140:     numpy.lib.arraysetops : Module with a number of other functions for
141:                             performing set operations on arrays.
142: 
143:     Examples
144:     --------
145:     >>> np.unique([1, 1, 2, 2, 3, 3])
146:     array([1, 2, 3])
147:     >>> a = np.array([[1, 1], [2, 3]])
148:     >>> np.unique(a)
149:     array([1, 2, 3])
150: 
151:     Return the indices of the original array that give the unique values:
152: 
153:     >>> a = np.array(['a', 'b', 'b', 'c', 'a'])
154:     >>> u, indices = np.unique(a, return_index=True)
155:     >>> u
156:     array(['a', 'b', 'c'],
157:            dtype='|S1')
158:     >>> indices
159:     array([0, 1, 3])
160:     >>> a[indices]
161:     array(['a', 'b', 'c'],
162:            dtype='|S1')
163: 
164:     Reconstruct the input array from the unique values:
165: 
166:     >>> a = np.array([1, 2, 6, 4, 2, 3, 2])
167:     >>> u, indices = np.unique(a, return_inverse=True)
168:     >>> u
169:     array([1, 2, 3, 4, 6])
170:     >>> indices
171:     array([0, 1, 4, 3, 1, 2, 1])
172:     >>> u[indices]
173:     array([1, 2, 6, 4, 2, 3, 2])
174: 
175:     '''
176:     ar = np.asanyarray(ar).flatten()
177: 
178:     optional_indices = return_index or return_inverse
179:     optional_returns = optional_indices or return_counts
180: 
181:     if ar.size == 0:
182:         if not optional_returns:
183:             ret = ar
184:         else:
185:             ret = (ar,)
186:             if return_index:
187:                 ret += (np.empty(0, np.bool),)
188:             if return_inverse:
189:                 ret += (np.empty(0, np.bool),)
190:             if return_counts:
191:                 ret += (np.empty(0, np.intp),)
192:         return ret
193: 
194:     if optional_indices:
195:         perm = ar.argsort(kind='mergesort' if return_index else 'quicksort')
196:         aux = ar[perm]
197:     else:
198:         ar.sort()
199:         aux = ar
200:     flag = np.concatenate(([True], aux[1:] != aux[:-1]))
201: 
202:     if not optional_returns:
203:         ret = aux[flag]
204:     else:
205:         ret = (aux[flag],)
206:         if return_index:
207:             ret += (perm[flag],)
208:         if return_inverse:
209:             iflag = np.cumsum(flag) - 1
210:             inv_idx = np.empty(ar.shape, dtype=np.intp)
211:             inv_idx[perm] = iflag
212:             ret += (inv_idx,)
213:         if return_counts:
214:             idx = np.concatenate(np.nonzero(flag) + ([ar.size],))
215:             ret += (np.diff(idx),)
216:     return ret
217: 
218: def intersect1d(ar1, ar2, assume_unique=False):
219:     '''
220:     Find the intersection of two arrays.
221: 
222:     Return the sorted, unique values that are in both of the input arrays.
223: 
224:     Parameters
225:     ----------
226:     ar1, ar2 : array_like
227:         Input arrays.
228:     assume_unique : bool
229:         If True, the input arrays are both assumed to be unique, which
230:         can speed up the calculation.  Default is False.
231: 
232:     Returns
233:     -------
234:     intersect1d : ndarray
235:         Sorted 1D array of common and unique elements.
236: 
237:     See Also
238:     --------
239:     numpy.lib.arraysetops : Module with a number of other functions for
240:                             performing set operations on arrays.
241: 
242:     Examples
243:     --------
244:     >>> np.intersect1d([1, 3, 4, 3], [3, 1, 2, 1])
245:     array([1, 3])
246: 
247:     To intersect more than two arrays, use functools.reduce:
248: 
249:     >>> from functools import reduce
250:     >>> reduce(np.intersect1d, ([1, 3, 4, 3], [3, 1, 2, 1], [6, 3, 4, 2]))
251:     array([3])
252:     '''
253:     if not assume_unique:
254:         # Might be faster than unique( intersect1d( ar1, ar2 ) )?
255:         ar1 = unique(ar1)
256:         ar2 = unique(ar2)
257:     aux = np.concatenate((ar1, ar2))
258:     aux.sort()
259:     return aux[:-1][aux[1:] == aux[:-1]]
260: 
261: def setxor1d(ar1, ar2, assume_unique=False):
262:     '''
263:     Find the set exclusive-or of two arrays.
264: 
265:     Return the sorted, unique values that are in only one (not both) of the
266:     input arrays.
267: 
268:     Parameters
269:     ----------
270:     ar1, ar2 : array_like
271:         Input arrays.
272:     assume_unique : bool
273:         If True, the input arrays are both assumed to be unique, which
274:         can speed up the calculation.  Default is False.
275: 
276:     Returns
277:     -------
278:     setxor1d : ndarray
279:         Sorted 1D array of unique values that are in only one of the input
280:         arrays.
281: 
282:     Examples
283:     --------
284:     >>> a = np.array([1, 2, 3, 2, 4])
285:     >>> b = np.array([2, 3, 5, 7, 5])
286:     >>> np.setxor1d(a,b)
287:     array([1, 4, 5, 7])
288: 
289:     '''
290:     if not assume_unique:
291:         ar1 = unique(ar1)
292:         ar2 = unique(ar2)
293: 
294:     aux = np.concatenate((ar1, ar2))
295:     if aux.size == 0:
296:         return aux
297: 
298:     aux.sort()
299: #    flag = ediff1d( aux, to_end = 1, to_begin = 1 ) == 0
300:     flag = np.concatenate(([True], aux[1:] != aux[:-1], [True]))
301: #    flag2 = ediff1d( flag ) == 0
302:     flag2 = flag[1:] == flag[:-1]
303:     return aux[flag2]
304: 
305: def in1d(ar1, ar2, assume_unique=False, invert=False):
306:     '''
307:     Test whether each element of a 1-D array is also present in a second array.
308: 
309:     Returns a boolean array the same length as `ar1` that is True
310:     where an element of `ar1` is in `ar2` and False otherwise.
311: 
312:     Parameters
313:     ----------
314:     ar1 : (M,) array_like
315:         Input array.
316:     ar2 : array_like
317:         The values against which to test each value of `ar1`.
318:     assume_unique : bool, optional
319:         If True, the input arrays are both assumed to be unique, which
320:         can speed up the calculation.  Default is False.
321:     invert : bool, optional
322:         If True, the values in the returned array are inverted (that is,
323:         False where an element of `ar1` is in `ar2` and True otherwise).
324:         Default is False. ``np.in1d(a, b, invert=True)`` is equivalent
325:         to (but is faster than) ``np.invert(in1d(a, b))``.
326: 
327:         .. versionadded:: 1.8.0
328: 
329:     Returns
330:     -------
331:     in1d : (M,) ndarray, bool
332:         The values `ar1[in1d]` are in `ar2`.
333: 
334:     See Also
335:     --------
336:     numpy.lib.arraysetops : Module with a number of other functions for
337:                             performing set operations on arrays.
338: 
339:     Notes
340:     -----
341:     `in1d` can be considered as an element-wise function version of the
342:     python keyword `in`, for 1-D sequences. ``in1d(a, b)`` is roughly
343:     equivalent to ``np.array([item in b for item in a])``.
344:     However, this idea fails if `ar2` is a set, or similar (non-sequence)
345:     container:  As ``ar2`` is converted to an array, in those cases
346:     ``asarray(ar2)`` is an object array rather than the expected array of
347:     contained values.
348: 
349:     .. versionadded:: 1.4.0
350: 
351:     Examples
352:     --------
353:     >>> test = np.array([0, 1, 2, 5, 0])
354:     >>> states = [0, 2]
355:     >>> mask = np.in1d(test, states)
356:     >>> mask
357:     array([ True, False,  True, False,  True], dtype=bool)
358:     >>> test[mask]
359:     array([0, 2, 0])
360:     >>> mask = np.in1d(test, states, invert=True)
361:     >>> mask
362:     array([False,  True, False,  True, False], dtype=bool)
363:     >>> test[mask]
364:     array([1, 5])
365:     '''
366:     # Ravel both arrays, behavior for the first array could be different
367:     ar1 = np.asarray(ar1).ravel()
368:     ar2 = np.asarray(ar2).ravel()
369: 
370:     # This code is significantly faster when the condition is satisfied.
371:     if len(ar2) < 10 * len(ar1) ** 0.145:
372:         if invert:
373:             mask = np.ones(len(ar1), dtype=np.bool)
374:             for a in ar2:
375:                 mask &= (ar1 != a)
376:         else:
377:             mask = np.zeros(len(ar1), dtype=np.bool)
378:             for a in ar2:
379:                 mask |= (ar1 == a)
380:         return mask
381: 
382:     # Otherwise use sorting
383:     if not assume_unique:
384:         ar1, rev_idx = np.unique(ar1, return_inverse=True)
385:         ar2 = np.unique(ar2)
386: 
387:     ar = np.concatenate((ar1, ar2))
388:     # We need this to be a stable sort, so always use 'mergesort'
389:     # here. The values from the first array should always come before
390:     # the values from the second array.
391:     order = ar.argsort(kind='mergesort')
392:     sar = ar[order]
393:     if invert:
394:         bool_ar = (sar[1:] != sar[:-1])
395:     else:
396:         bool_ar = (sar[1:] == sar[:-1])
397:     flag = np.concatenate((bool_ar, [invert]))
398:     ret = np.empty(ar.shape, dtype=bool)
399:     ret[order] = flag
400: 
401:     if assume_unique:
402:         return ret[:len(ar1)]
403:     else:
404:         return ret[rev_idx]
405: 
406: def union1d(ar1, ar2):
407:     '''
408:     Find the union of two arrays.
409: 
410:     Return the unique, sorted array of values that are in either of the two
411:     input arrays.
412: 
413:     Parameters
414:     ----------
415:     ar1, ar2 : array_like
416:         Input arrays. They are flattened if they are not already 1D.
417: 
418:     Returns
419:     -------
420:     union1d : ndarray
421:         Unique, sorted union of the input arrays.
422: 
423:     See Also
424:     --------
425:     numpy.lib.arraysetops : Module with a number of other functions for
426:                             performing set operations on arrays.
427: 
428:     Examples
429:     --------
430:     >>> np.union1d([-1, 0, 1], [-2, 0, 2])
431:     array([-2, -1,  0,  1,  2])
432: 
433:     To find the union of more than two arrays, use functools.reduce:
434: 
435:     >>> from functools import reduce
436:     >>> reduce(np.union1d, ([1, 3, 4, 3], [3, 1, 2, 1], [6, 3, 4, 2]))
437:     array([1, 2, 3, 4, 6])
438:     '''
439:     return unique(np.concatenate((ar1, ar2)))
440: 
441: def setdiff1d(ar1, ar2, assume_unique=False):
442:     '''
443:     Find the set difference of two arrays.
444: 
445:     Return the sorted, unique values in `ar1` that are not in `ar2`.
446: 
447:     Parameters
448:     ----------
449:     ar1 : array_like
450:         Input array.
451:     ar2 : array_like
452:         Input comparison array.
453:     assume_unique : bool
454:         If True, the input arrays are both assumed to be unique, which
455:         can speed up the calculation.  Default is False.
456: 
457:     Returns
458:     -------
459:     setdiff1d : ndarray
460:         Sorted 1D array of values in `ar1` that are not in `ar2`.
461: 
462:     See Also
463:     --------
464:     numpy.lib.arraysetops : Module with a number of other functions for
465:                             performing set operations on arrays.
466: 
467:     Examples
468:     --------
469:     >>> a = np.array([1, 2, 3, 2, 4, 1])
470:     >>> b = np.array([3, 4, 5, 6])
471:     >>> np.setdiff1d(a, b)
472:     array([1, 2])
473: 
474:     '''
475:     if assume_unique:
476:         ar1 = np.asarray(ar1).ravel()
477:     else:
478:         ar1 = unique(ar1)
479:         ar2 = unique(ar2)
480:     return ar1[in1d(ar1, ar2, assume_unique=True, invert=True)]
481: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_104167 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, (-1)), 'str', '\nSet operations for 1D numeric arrays based on sorting.\n\n:Contains:\n  ediff1d,\n  unique,\n  intersect1d,\n  setxor1d,\n  in1d,\n  union1d,\n  setdiff1d\n\n:Notes:\n\nFor floating point arrays, inaccurate results may appear due to usual round-off\nand floating point comparison issues.\n\nSpeed could be gained in some operations by an implementation of\nsort(), that can provide directly the permutation vectors, avoiding\nthus calls to argsort().\n\nTo do: Optionally return indices analogously to unique for all functions.\n\n:Author: Robert Cimrman\n\n')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 29, 0))

# 'import numpy' statement (line 29)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/lib/')
import_104168 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 29, 0), 'numpy')

if (type(import_104168) is not StypyTypeError):

    if (import_104168 != 'pyd_module'):
        __import__(import_104168)
        sys_modules_104169 = sys.modules[import_104168]
        import_module(stypy.reporting.localization.Localization(__file__, 29, 0), 'np', sys_modules_104169.module_type_store, module_type_store)
    else:
        import numpy as np

        import_module(stypy.reporting.localization.Localization(__file__, 29, 0), 'np', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 29)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 29, 0), 'numpy', import_104168)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/lib/')


# Assigning a List to a Name (line 32):

# Assigning a List to a Name (line 32):
__all__ = ['ediff1d', 'intersect1d', 'setxor1d', 'union1d', 'setdiff1d', 'unique', 'in1d']
module_type_store.set_exportable_members(['ediff1d', 'intersect1d', 'setxor1d', 'union1d', 'setdiff1d', 'unique', 'in1d'])

# Obtaining an instance of the builtin type 'list' (line 32)
list_104170 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 32, 10), 'list')
# Adding type elements to the builtin type 'list' instance (line 32)
# Adding element type (line 32)
str_104171 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 4), 'str', 'ediff1d')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 32, 10), list_104170, str_104171)
# Adding element type (line 32)
str_104172 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 15), 'str', 'intersect1d')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 32, 10), list_104170, str_104172)
# Adding element type (line 32)
str_104173 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 30), 'str', 'setxor1d')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 32, 10), list_104170, str_104173)
# Adding element type (line 32)
str_104174 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 42), 'str', 'union1d')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 32, 10), list_104170, str_104174)
# Adding element type (line 32)
str_104175 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 53), 'str', 'setdiff1d')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 32, 10), list_104170, str_104175)
# Adding element type (line 32)
str_104176 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 66), 'str', 'unique')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 32, 10), list_104170, str_104176)
# Adding element type (line 32)
str_104177 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 4), 'str', 'in1d')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 32, 10), list_104170, str_104177)

# Assigning a type to the variable '__all__' (line 32)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 32, 0), '__all__', list_104170)

@norecursion
def ediff1d(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'None' (line 38)
    None_104178 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 24), 'None')
    # Getting the type of 'None' (line 38)
    None_104179 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 39), 'None')
    defaults = [None_104178, None_104179]
    # Create a new context for function 'ediff1d'
    module_type_store = module_type_store.open_function_context('ediff1d', 38, 0, False)
    
    # Passed parameters checking function
    ediff1d.stypy_localization = localization
    ediff1d.stypy_type_of_self = None
    ediff1d.stypy_type_store = module_type_store
    ediff1d.stypy_function_name = 'ediff1d'
    ediff1d.stypy_param_names_list = ['ary', 'to_end', 'to_begin']
    ediff1d.stypy_varargs_param_name = None
    ediff1d.stypy_kwargs_param_name = None
    ediff1d.stypy_call_defaults = defaults
    ediff1d.stypy_call_varargs = varargs
    ediff1d.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'ediff1d', ['ary', 'to_end', 'to_begin'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'ediff1d', localization, ['ary', 'to_end', 'to_begin'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'ediff1d(...)' code ##################

    str_104180 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 80, (-1)), 'str', '\n    The differences between consecutive elements of an array.\n\n    Parameters\n    ----------\n    ary : array_like\n        If necessary, will be flattened before the differences are taken.\n    to_end : array_like, optional\n        Number(s) to append at the end of the returned differences.\n    to_begin : array_like, optional\n        Number(s) to prepend at the beginning of the returned differences.\n\n    Returns\n    -------\n    ediff1d : ndarray\n        The differences. Loosely, this is ``ary.flat[1:] - ary.flat[:-1]``.\n\n    See Also\n    --------\n    diff, gradient\n\n    Notes\n    -----\n    When applied to masked arrays, this function drops the mask information\n    if the `to_begin` and/or `to_end` parameters are used.\n\n    Examples\n    --------\n    >>> x = np.array([1, 2, 4, 7, 0])\n    >>> np.ediff1d(x)\n    array([ 1,  2,  3, -7])\n\n    >>> np.ediff1d(x, to_begin=-99, to_end=np.array([88, 99]))\n    array([-99,   1,   2,   3,  -7,  88,  99])\n\n    The returned array is always 1D.\n\n    >>> y = [[1, 2, 4], [1, 6, 24]]\n    >>> np.ediff1d(y)\n    array([ 1,  2, -3,  5, 18])\n\n    ')
    
    # Assigning a Attribute to a Name (line 81):
    
    # Assigning a Attribute to a Name (line 81):
    
    # Call to asanyarray(...): (line 81)
    # Processing the call arguments (line 81)
    # Getting the type of 'ary' (line 81)
    ary_104183 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 24), 'ary', False)
    # Processing the call keyword arguments (line 81)
    kwargs_104184 = {}
    # Getting the type of 'np' (line 81)
    np_104181 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 10), 'np', False)
    # Obtaining the member 'asanyarray' of a type (line 81)
    asanyarray_104182 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 81, 10), np_104181, 'asanyarray')
    # Calling asanyarray(args, kwargs) (line 81)
    asanyarray_call_result_104185 = invoke(stypy.reporting.localization.Localization(__file__, 81, 10), asanyarray_104182, *[ary_104183], **kwargs_104184)
    
    # Obtaining the member 'flat' of a type (line 81)
    flat_104186 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 81, 10), asanyarray_call_result_104185, 'flat')
    # Assigning a type to the variable 'ary' (line 81)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 81, 4), 'ary', flat_104186)
    
    # Assigning a BinOp to a Name (line 82):
    
    # Assigning a BinOp to a Name (line 82):
    
    # Obtaining the type of the subscript
    int_104187 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 82, 13), 'int')
    slice_104188 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 82, 9), int_104187, None, None)
    # Getting the type of 'ary' (line 82)
    ary_104189 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 9), 'ary')
    # Obtaining the member '__getitem__' of a type (line 82)
    getitem___104190 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 82, 9), ary_104189, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 82)
    subscript_call_result_104191 = invoke(stypy.reporting.localization.Localization(__file__, 82, 9), getitem___104190, slice_104188)
    
    
    # Obtaining the type of the subscript
    int_104192 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 82, 24), 'int')
    slice_104193 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 82, 19), None, int_104192, None)
    # Getting the type of 'ary' (line 82)
    ary_104194 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 19), 'ary')
    # Obtaining the member '__getitem__' of a type (line 82)
    getitem___104195 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 82, 19), ary_104194, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 82)
    subscript_call_result_104196 = invoke(stypy.reporting.localization.Localization(__file__, 82, 19), getitem___104195, slice_104193)
    
    # Applying the binary operator '-' (line 82)
    result_sub_104197 = python_operator(stypy.reporting.localization.Localization(__file__, 82, 9), '-', subscript_call_result_104191, subscript_call_result_104196)
    
    # Assigning a type to the variable 'ed' (line 82)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 82, 4), 'ed', result_sub_104197)
    
    # Assigning a List to a Name (line 83):
    
    # Assigning a List to a Name (line 83):
    
    # Obtaining an instance of the builtin type 'list' (line 83)
    list_104198 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 83, 13), 'list')
    # Adding type elements to the builtin type 'list' instance (line 83)
    # Adding element type (line 83)
    # Getting the type of 'ed' (line 83)
    ed_104199 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 14), 'ed')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 83, 13), list_104198, ed_104199)
    
    # Assigning a type to the variable 'arrays' (line 83)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 83, 4), 'arrays', list_104198)
    
    # Type idiom detected: calculating its left and rigth part (line 84)
    # Getting the type of 'to_begin' (line 84)
    to_begin_104200 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 4), 'to_begin')
    # Getting the type of 'None' (line 84)
    None_104201 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 23), 'None')
    
    (may_be_104202, more_types_in_union_104203) = may_not_be_none(to_begin_104200, None_104201)

    if may_be_104202:

        if more_types_in_union_104203:
            # Runtime conditional SSA (line 84)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Call to insert(...): (line 85)
        # Processing the call arguments (line 85)
        int_104206 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 85, 22), 'int')
        # Getting the type of 'to_begin' (line 85)
        to_begin_104207 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 25), 'to_begin', False)
        # Processing the call keyword arguments (line 85)
        kwargs_104208 = {}
        # Getting the type of 'arrays' (line 85)
        arrays_104204 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 8), 'arrays', False)
        # Obtaining the member 'insert' of a type (line 85)
        insert_104205 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 85, 8), arrays_104204, 'insert')
        # Calling insert(args, kwargs) (line 85)
        insert_call_result_104209 = invoke(stypy.reporting.localization.Localization(__file__, 85, 8), insert_104205, *[int_104206, to_begin_104207], **kwargs_104208)
        

        if more_types_in_union_104203:
            # SSA join for if statement (line 84)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Type idiom detected: calculating its left and rigth part (line 86)
    # Getting the type of 'to_end' (line 86)
    to_end_104210 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 4), 'to_end')
    # Getting the type of 'None' (line 86)
    None_104211 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 21), 'None')
    
    (may_be_104212, more_types_in_union_104213) = may_not_be_none(to_end_104210, None_104211)

    if may_be_104212:

        if more_types_in_union_104213:
            # Runtime conditional SSA (line 86)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Call to append(...): (line 87)
        # Processing the call arguments (line 87)
        # Getting the type of 'to_end' (line 87)
        to_end_104216 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 22), 'to_end', False)
        # Processing the call keyword arguments (line 87)
        kwargs_104217 = {}
        # Getting the type of 'arrays' (line 87)
        arrays_104214 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 8), 'arrays', False)
        # Obtaining the member 'append' of a type (line 87)
        append_104215 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 87, 8), arrays_104214, 'append')
        # Calling append(args, kwargs) (line 87)
        append_call_result_104218 = invoke(stypy.reporting.localization.Localization(__file__, 87, 8), append_104215, *[to_end_104216], **kwargs_104217)
        

        if more_types_in_union_104213:
            # SSA join for if statement (line 86)
            module_type_store = module_type_store.join_ssa_context()


    
    
    
    
    # Call to len(...): (line 89)
    # Processing the call arguments (line 89)
    # Getting the type of 'arrays' (line 89)
    arrays_104220 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 11), 'arrays', False)
    # Processing the call keyword arguments (line 89)
    kwargs_104221 = {}
    # Getting the type of 'len' (line 89)
    len_104219 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 7), 'len', False)
    # Calling len(args, kwargs) (line 89)
    len_call_result_104222 = invoke(stypy.reporting.localization.Localization(__file__, 89, 7), len_104219, *[arrays_104220], **kwargs_104221)
    
    int_104223 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 89, 22), 'int')
    # Applying the binary operator '!=' (line 89)
    result_ne_104224 = python_operator(stypy.reporting.localization.Localization(__file__, 89, 7), '!=', len_call_result_104222, int_104223)
    
    # Testing the type of an if condition (line 89)
    if_condition_104225 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 89, 4), result_ne_104224)
    # Assigning a type to the variable 'if_condition_104225' (line 89)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 89, 4), 'if_condition_104225', if_condition_104225)
    # SSA begins for if statement (line 89)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 92):
    
    # Assigning a Call to a Name (line 92):
    
    # Call to hstack(...): (line 92)
    # Processing the call arguments (line 92)
    # Getting the type of 'arrays' (line 92)
    arrays_104228 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 23), 'arrays', False)
    # Processing the call keyword arguments (line 92)
    kwargs_104229 = {}
    # Getting the type of 'np' (line 92)
    np_104226 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 13), 'np', False)
    # Obtaining the member 'hstack' of a type (line 92)
    hstack_104227 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 92, 13), np_104226, 'hstack')
    # Calling hstack(args, kwargs) (line 92)
    hstack_call_result_104230 = invoke(stypy.reporting.localization.Localization(__file__, 92, 13), hstack_104227, *[arrays_104228], **kwargs_104229)
    
    # Assigning a type to the variable 'ed' (line 92)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 92, 8), 'ed', hstack_call_result_104230)
    # SSA join for if statement (line 89)
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'ed' (line 94)
    ed_104231 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 11), 'ed')
    # Assigning a type to the variable 'stypy_return_type' (line 94)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 94, 4), 'stypy_return_type', ed_104231)
    
    # ################# End of 'ediff1d(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'ediff1d' in the type store
    # Getting the type of 'stypy_return_type' (line 38)
    stypy_return_type_104232 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_104232)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'ediff1d'
    return stypy_return_type_104232

# Assigning a type to the variable 'ediff1d' (line 38)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 38, 0), 'ediff1d', ediff1d)

@norecursion
def unique(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'False' (line 96)
    False_104233 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 28), 'False')
    # Getting the type of 'False' (line 96)
    False_104234 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 50), 'False')
    # Getting the type of 'False' (line 96)
    False_104235 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 71), 'False')
    defaults = [False_104233, False_104234, False_104235]
    # Create a new context for function 'unique'
    module_type_store = module_type_store.open_function_context('unique', 96, 0, False)
    
    # Passed parameters checking function
    unique.stypy_localization = localization
    unique.stypy_type_of_self = None
    unique.stypy_type_store = module_type_store
    unique.stypy_function_name = 'unique'
    unique.stypy_param_names_list = ['ar', 'return_index', 'return_inverse', 'return_counts']
    unique.stypy_varargs_param_name = None
    unique.stypy_kwargs_param_name = None
    unique.stypy_call_defaults = defaults
    unique.stypy_call_varargs = varargs
    unique.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'unique', ['ar', 'return_index', 'return_inverse', 'return_counts'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'unique', localization, ['ar', 'return_index', 'return_inverse', 'return_counts'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'unique(...)' code ##################

    str_104236 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 175, (-1)), 'str', "\n    Find the unique elements of an array.\n\n    Returns the sorted unique elements of an array. There are three optional\n    outputs in addition to the unique elements: the indices of the input array\n    that give the unique values, the indices of the unique array that\n    reconstruct the input array, and the number of times each unique value\n    comes up in the input array.\n\n    Parameters\n    ----------\n    ar : array_like\n        Input array. This will be flattened if it is not already 1-D.\n    return_index : bool, optional\n        If True, also return the indices of `ar` that result in the unique\n        array.\n    return_inverse : bool, optional\n        If True, also return the indices of the unique array that can be used\n        to reconstruct `ar`.\n    return_counts : bool, optional\n        If True, also return the number of times each unique value comes up\n        in `ar`.\n\n        .. versionadded:: 1.9.0\n\n    Returns\n    -------\n    unique : ndarray\n        The sorted unique values.\n    unique_indices : ndarray, optional\n        The indices of the first occurrences of the unique values in the\n        (flattened) original array. Only provided if `return_index` is True.\n    unique_inverse : ndarray, optional\n        The indices to reconstruct the (flattened) original array from the\n        unique array. Only provided if `return_inverse` is True.\n    unique_counts : ndarray, optional\n        The number of times each of the unique values comes up in the\n        original array. Only provided if `return_counts` is True.\n\n        .. versionadded:: 1.9.0\n\n    See Also\n    --------\n    numpy.lib.arraysetops : Module with a number of other functions for\n                            performing set operations on arrays.\n\n    Examples\n    --------\n    >>> np.unique([1, 1, 2, 2, 3, 3])\n    array([1, 2, 3])\n    >>> a = np.array([[1, 1], [2, 3]])\n    >>> np.unique(a)\n    array([1, 2, 3])\n\n    Return the indices of the original array that give the unique values:\n\n    >>> a = np.array(['a', 'b', 'b', 'c', 'a'])\n    >>> u, indices = np.unique(a, return_index=True)\n    >>> u\n    array(['a', 'b', 'c'],\n           dtype='|S1')\n    >>> indices\n    array([0, 1, 3])\n    >>> a[indices]\n    array(['a', 'b', 'c'],\n           dtype='|S1')\n\n    Reconstruct the input array from the unique values:\n\n    >>> a = np.array([1, 2, 6, 4, 2, 3, 2])\n    >>> u, indices = np.unique(a, return_inverse=True)\n    >>> u\n    array([1, 2, 3, 4, 6])\n    >>> indices\n    array([0, 1, 4, 3, 1, 2, 1])\n    >>> u[indices]\n    array([1, 2, 6, 4, 2, 3, 2])\n\n    ")
    
    # Assigning a Call to a Name (line 176):
    
    # Assigning a Call to a Name (line 176):
    
    # Call to flatten(...): (line 176)
    # Processing the call keyword arguments (line 176)
    kwargs_104243 = {}
    
    # Call to asanyarray(...): (line 176)
    # Processing the call arguments (line 176)
    # Getting the type of 'ar' (line 176)
    ar_104239 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 23), 'ar', False)
    # Processing the call keyword arguments (line 176)
    kwargs_104240 = {}
    # Getting the type of 'np' (line 176)
    np_104237 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 9), 'np', False)
    # Obtaining the member 'asanyarray' of a type (line 176)
    asanyarray_104238 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 176, 9), np_104237, 'asanyarray')
    # Calling asanyarray(args, kwargs) (line 176)
    asanyarray_call_result_104241 = invoke(stypy.reporting.localization.Localization(__file__, 176, 9), asanyarray_104238, *[ar_104239], **kwargs_104240)
    
    # Obtaining the member 'flatten' of a type (line 176)
    flatten_104242 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 176, 9), asanyarray_call_result_104241, 'flatten')
    # Calling flatten(args, kwargs) (line 176)
    flatten_call_result_104244 = invoke(stypy.reporting.localization.Localization(__file__, 176, 9), flatten_104242, *[], **kwargs_104243)
    
    # Assigning a type to the variable 'ar' (line 176)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 176, 4), 'ar', flatten_call_result_104244)
    
    # Assigning a BoolOp to a Name (line 178):
    
    # Assigning a BoolOp to a Name (line 178):
    
    # Evaluating a boolean operation
    # Getting the type of 'return_index' (line 178)
    return_index_104245 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 23), 'return_index')
    # Getting the type of 'return_inverse' (line 178)
    return_inverse_104246 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 39), 'return_inverse')
    # Applying the binary operator 'or' (line 178)
    result_or_keyword_104247 = python_operator(stypy.reporting.localization.Localization(__file__, 178, 23), 'or', return_index_104245, return_inverse_104246)
    
    # Assigning a type to the variable 'optional_indices' (line 178)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 178, 4), 'optional_indices', result_or_keyword_104247)
    
    # Assigning a BoolOp to a Name (line 179):
    
    # Assigning a BoolOp to a Name (line 179):
    
    # Evaluating a boolean operation
    # Getting the type of 'optional_indices' (line 179)
    optional_indices_104248 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 23), 'optional_indices')
    # Getting the type of 'return_counts' (line 179)
    return_counts_104249 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 43), 'return_counts')
    # Applying the binary operator 'or' (line 179)
    result_or_keyword_104250 = python_operator(stypy.reporting.localization.Localization(__file__, 179, 23), 'or', optional_indices_104248, return_counts_104249)
    
    # Assigning a type to the variable 'optional_returns' (line 179)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 179, 4), 'optional_returns', result_or_keyword_104250)
    
    
    # Getting the type of 'ar' (line 181)
    ar_104251 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 7), 'ar')
    # Obtaining the member 'size' of a type (line 181)
    size_104252 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 181, 7), ar_104251, 'size')
    int_104253 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 181, 18), 'int')
    # Applying the binary operator '==' (line 181)
    result_eq_104254 = python_operator(stypy.reporting.localization.Localization(__file__, 181, 7), '==', size_104252, int_104253)
    
    # Testing the type of an if condition (line 181)
    if_condition_104255 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 181, 4), result_eq_104254)
    # Assigning a type to the variable 'if_condition_104255' (line 181)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 181, 4), 'if_condition_104255', if_condition_104255)
    # SSA begins for if statement (line 181)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    
    # Getting the type of 'optional_returns' (line 182)
    optional_returns_104256 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 182, 15), 'optional_returns')
    # Applying the 'not' unary operator (line 182)
    result_not__104257 = python_operator(stypy.reporting.localization.Localization(__file__, 182, 11), 'not', optional_returns_104256)
    
    # Testing the type of an if condition (line 182)
    if_condition_104258 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 182, 8), result_not__104257)
    # Assigning a type to the variable 'if_condition_104258' (line 182)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 182, 8), 'if_condition_104258', if_condition_104258)
    # SSA begins for if statement (line 182)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Name to a Name (line 183):
    
    # Assigning a Name to a Name (line 183):
    # Getting the type of 'ar' (line 183)
    ar_104259 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 18), 'ar')
    # Assigning a type to the variable 'ret' (line 183)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 183, 12), 'ret', ar_104259)
    # SSA branch for the else part of an if statement (line 182)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Tuple to a Name (line 185):
    
    # Assigning a Tuple to a Name (line 185):
    
    # Obtaining an instance of the builtin type 'tuple' (line 185)
    tuple_104260 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 185, 19), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 185)
    # Adding element type (line 185)
    # Getting the type of 'ar' (line 185)
    ar_104261 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 19), 'ar')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 185, 19), tuple_104260, ar_104261)
    
    # Assigning a type to the variable 'ret' (line 185)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 185, 12), 'ret', tuple_104260)
    
    # Getting the type of 'return_index' (line 186)
    return_index_104262 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 15), 'return_index')
    # Testing the type of an if condition (line 186)
    if_condition_104263 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 186, 12), return_index_104262)
    # Assigning a type to the variable 'if_condition_104263' (line 186)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 186, 12), 'if_condition_104263', if_condition_104263)
    # SSA begins for if statement (line 186)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Getting the type of 'ret' (line 187)
    ret_104264 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 16), 'ret')
    
    # Obtaining an instance of the builtin type 'tuple' (line 187)
    tuple_104265 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 187, 24), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 187)
    # Adding element type (line 187)
    
    # Call to empty(...): (line 187)
    # Processing the call arguments (line 187)
    int_104268 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 187, 33), 'int')
    # Getting the type of 'np' (line 187)
    np_104269 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 36), 'np', False)
    # Obtaining the member 'bool' of a type (line 187)
    bool_104270 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 187, 36), np_104269, 'bool')
    # Processing the call keyword arguments (line 187)
    kwargs_104271 = {}
    # Getting the type of 'np' (line 187)
    np_104266 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 24), 'np', False)
    # Obtaining the member 'empty' of a type (line 187)
    empty_104267 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 187, 24), np_104266, 'empty')
    # Calling empty(args, kwargs) (line 187)
    empty_call_result_104272 = invoke(stypy.reporting.localization.Localization(__file__, 187, 24), empty_104267, *[int_104268, bool_104270], **kwargs_104271)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 187, 24), tuple_104265, empty_call_result_104272)
    
    # Applying the binary operator '+=' (line 187)
    result_iadd_104273 = python_operator(stypy.reporting.localization.Localization(__file__, 187, 16), '+=', ret_104264, tuple_104265)
    # Assigning a type to the variable 'ret' (line 187)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 187, 16), 'ret', result_iadd_104273)
    
    # SSA join for if statement (line 186)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Getting the type of 'return_inverse' (line 188)
    return_inverse_104274 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 188, 15), 'return_inverse')
    # Testing the type of an if condition (line 188)
    if_condition_104275 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 188, 12), return_inverse_104274)
    # Assigning a type to the variable 'if_condition_104275' (line 188)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 188, 12), 'if_condition_104275', if_condition_104275)
    # SSA begins for if statement (line 188)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Getting the type of 'ret' (line 189)
    ret_104276 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 16), 'ret')
    
    # Obtaining an instance of the builtin type 'tuple' (line 189)
    tuple_104277 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 189, 24), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 189)
    # Adding element type (line 189)
    
    # Call to empty(...): (line 189)
    # Processing the call arguments (line 189)
    int_104280 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 189, 33), 'int')
    # Getting the type of 'np' (line 189)
    np_104281 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 36), 'np', False)
    # Obtaining the member 'bool' of a type (line 189)
    bool_104282 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 189, 36), np_104281, 'bool')
    # Processing the call keyword arguments (line 189)
    kwargs_104283 = {}
    # Getting the type of 'np' (line 189)
    np_104278 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 24), 'np', False)
    # Obtaining the member 'empty' of a type (line 189)
    empty_104279 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 189, 24), np_104278, 'empty')
    # Calling empty(args, kwargs) (line 189)
    empty_call_result_104284 = invoke(stypy.reporting.localization.Localization(__file__, 189, 24), empty_104279, *[int_104280, bool_104282], **kwargs_104283)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 189, 24), tuple_104277, empty_call_result_104284)
    
    # Applying the binary operator '+=' (line 189)
    result_iadd_104285 = python_operator(stypy.reporting.localization.Localization(__file__, 189, 16), '+=', ret_104276, tuple_104277)
    # Assigning a type to the variable 'ret' (line 189)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 189, 16), 'ret', result_iadd_104285)
    
    # SSA join for if statement (line 188)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Getting the type of 'return_counts' (line 190)
    return_counts_104286 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 190, 15), 'return_counts')
    # Testing the type of an if condition (line 190)
    if_condition_104287 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 190, 12), return_counts_104286)
    # Assigning a type to the variable 'if_condition_104287' (line 190)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 190, 12), 'if_condition_104287', if_condition_104287)
    # SSA begins for if statement (line 190)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Getting the type of 'ret' (line 191)
    ret_104288 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 16), 'ret')
    
    # Obtaining an instance of the builtin type 'tuple' (line 191)
    tuple_104289 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 191, 24), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 191)
    # Adding element type (line 191)
    
    # Call to empty(...): (line 191)
    # Processing the call arguments (line 191)
    int_104292 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 191, 33), 'int')
    # Getting the type of 'np' (line 191)
    np_104293 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 36), 'np', False)
    # Obtaining the member 'intp' of a type (line 191)
    intp_104294 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 191, 36), np_104293, 'intp')
    # Processing the call keyword arguments (line 191)
    kwargs_104295 = {}
    # Getting the type of 'np' (line 191)
    np_104290 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 24), 'np', False)
    # Obtaining the member 'empty' of a type (line 191)
    empty_104291 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 191, 24), np_104290, 'empty')
    # Calling empty(args, kwargs) (line 191)
    empty_call_result_104296 = invoke(stypy.reporting.localization.Localization(__file__, 191, 24), empty_104291, *[int_104292, intp_104294], **kwargs_104295)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 191, 24), tuple_104289, empty_call_result_104296)
    
    # Applying the binary operator '+=' (line 191)
    result_iadd_104297 = python_operator(stypy.reporting.localization.Localization(__file__, 191, 16), '+=', ret_104288, tuple_104289)
    # Assigning a type to the variable 'ret' (line 191)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 191, 16), 'ret', result_iadd_104297)
    
    # SSA join for if statement (line 190)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 182)
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'ret' (line 192)
    ret_104298 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 192, 15), 'ret')
    # Assigning a type to the variable 'stypy_return_type' (line 192)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 192, 8), 'stypy_return_type', ret_104298)
    # SSA join for if statement (line 181)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Getting the type of 'optional_indices' (line 194)
    optional_indices_104299 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 194, 7), 'optional_indices')
    # Testing the type of an if condition (line 194)
    if_condition_104300 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 194, 4), optional_indices_104299)
    # Assigning a type to the variable 'if_condition_104300' (line 194)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 194, 4), 'if_condition_104300', if_condition_104300)
    # SSA begins for if statement (line 194)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 195):
    
    # Assigning a Call to a Name (line 195):
    
    # Call to argsort(...): (line 195)
    # Processing the call keyword arguments (line 195)
    
    # Getting the type of 'return_index' (line 195)
    return_index_104303 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 195, 46), 'return_index', False)
    # Testing the type of an if expression (line 195)
    is_suitable_condition(stypy.reporting.localization.Localization(__file__, 195, 31), return_index_104303)
    # SSA begins for if expression (line 195)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if expression')
    str_104304 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 195, 31), 'str', 'mergesort')
    # SSA branch for the else part of an if expression (line 195)
    module_type_store.open_ssa_branch('if expression else')
    str_104305 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 195, 64), 'str', 'quicksort')
    # SSA join for if expression (line 195)
    module_type_store = module_type_store.join_ssa_context()
    if_exp_104306 = union_type.UnionType.add(str_104304, str_104305)
    
    keyword_104307 = if_exp_104306
    kwargs_104308 = {'kind': keyword_104307}
    # Getting the type of 'ar' (line 195)
    ar_104301 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 195, 15), 'ar', False)
    # Obtaining the member 'argsort' of a type (line 195)
    argsort_104302 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 195, 15), ar_104301, 'argsort')
    # Calling argsort(args, kwargs) (line 195)
    argsort_call_result_104309 = invoke(stypy.reporting.localization.Localization(__file__, 195, 15), argsort_104302, *[], **kwargs_104308)
    
    # Assigning a type to the variable 'perm' (line 195)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 195, 8), 'perm', argsort_call_result_104309)
    
    # Assigning a Subscript to a Name (line 196):
    
    # Assigning a Subscript to a Name (line 196):
    
    # Obtaining the type of the subscript
    # Getting the type of 'perm' (line 196)
    perm_104310 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 196, 17), 'perm')
    # Getting the type of 'ar' (line 196)
    ar_104311 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 196, 14), 'ar')
    # Obtaining the member '__getitem__' of a type (line 196)
    getitem___104312 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 196, 14), ar_104311, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 196)
    subscript_call_result_104313 = invoke(stypy.reporting.localization.Localization(__file__, 196, 14), getitem___104312, perm_104310)
    
    # Assigning a type to the variable 'aux' (line 196)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 196, 8), 'aux', subscript_call_result_104313)
    # SSA branch for the else part of an if statement (line 194)
    module_type_store.open_ssa_branch('else')
    
    # Call to sort(...): (line 198)
    # Processing the call keyword arguments (line 198)
    kwargs_104316 = {}
    # Getting the type of 'ar' (line 198)
    ar_104314 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 8), 'ar', False)
    # Obtaining the member 'sort' of a type (line 198)
    sort_104315 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 198, 8), ar_104314, 'sort')
    # Calling sort(args, kwargs) (line 198)
    sort_call_result_104317 = invoke(stypy.reporting.localization.Localization(__file__, 198, 8), sort_104315, *[], **kwargs_104316)
    
    
    # Assigning a Name to a Name (line 199):
    
    # Assigning a Name to a Name (line 199):
    # Getting the type of 'ar' (line 199)
    ar_104318 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 14), 'ar')
    # Assigning a type to the variable 'aux' (line 199)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 199, 8), 'aux', ar_104318)
    # SSA join for if statement (line 194)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 200):
    
    # Assigning a Call to a Name (line 200):
    
    # Call to concatenate(...): (line 200)
    # Processing the call arguments (line 200)
    
    # Obtaining an instance of the builtin type 'tuple' (line 200)
    tuple_104321 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 200, 27), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 200)
    # Adding element type (line 200)
    
    # Obtaining an instance of the builtin type 'list' (line 200)
    list_104322 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 200, 27), 'list')
    # Adding type elements to the builtin type 'list' instance (line 200)
    # Adding element type (line 200)
    # Getting the type of 'True' (line 200)
    True_104323 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 28), 'True', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 200, 27), list_104322, True_104323)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 200, 27), tuple_104321, list_104322)
    # Adding element type (line 200)
    
    
    # Obtaining the type of the subscript
    int_104324 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 200, 39), 'int')
    slice_104325 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 200, 35), int_104324, None, None)
    # Getting the type of 'aux' (line 200)
    aux_104326 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 35), 'aux', False)
    # Obtaining the member '__getitem__' of a type (line 200)
    getitem___104327 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 200, 35), aux_104326, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 200)
    subscript_call_result_104328 = invoke(stypy.reporting.localization.Localization(__file__, 200, 35), getitem___104327, slice_104325)
    
    
    # Obtaining the type of the subscript
    int_104329 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 200, 51), 'int')
    slice_104330 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 200, 46), None, int_104329, None)
    # Getting the type of 'aux' (line 200)
    aux_104331 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 46), 'aux', False)
    # Obtaining the member '__getitem__' of a type (line 200)
    getitem___104332 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 200, 46), aux_104331, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 200)
    subscript_call_result_104333 = invoke(stypy.reporting.localization.Localization(__file__, 200, 46), getitem___104332, slice_104330)
    
    # Applying the binary operator '!=' (line 200)
    result_ne_104334 = python_operator(stypy.reporting.localization.Localization(__file__, 200, 35), '!=', subscript_call_result_104328, subscript_call_result_104333)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 200, 27), tuple_104321, result_ne_104334)
    
    # Processing the call keyword arguments (line 200)
    kwargs_104335 = {}
    # Getting the type of 'np' (line 200)
    np_104319 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 11), 'np', False)
    # Obtaining the member 'concatenate' of a type (line 200)
    concatenate_104320 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 200, 11), np_104319, 'concatenate')
    # Calling concatenate(args, kwargs) (line 200)
    concatenate_call_result_104336 = invoke(stypy.reporting.localization.Localization(__file__, 200, 11), concatenate_104320, *[tuple_104321], **kwargs_104335)
    
    # Assigning a type to the variable 'flag' (line 200)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 200, 4), 'flag', concatenate_call_result_104336)
    
    
    # Getting the type of 'optional_returns' (line 202)
    optional_returns_104337 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 11), 'optional_returns')
    # Applying the 'not' unary operator (line 202)
    result_not__104338 = python_operator(stypy.reporting.localization.Localization(__file__, 202, 7), 'not', optional_returns_104337)
    
    # Testing the type of an if condition (line 202)
    if_condition_104339 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 202, 4), result_not__104338)
    # Assigning a type to the variable 'if_condition_104339' (line 202)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 202, 4), 'if_condition_104339', if_condition_104339)
    # SSA begins for if statement (line 202)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Subscript to a Name (line 203):
    
    # Assigning a Subscript to a Name (line 203):
    
    # Obtaining the type of the subscript
    # Getting the type of 'flag' (line 203)
    flag_104340 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 18), 'flag')
    # Getting the type of 'aux' (line 203)
    aux_104341 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 14), 'aux')
    # Obtaining the member '__getitem__' of a type (line 203)
    getitem___104342 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 203, 14), aux_104341, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 203)
    subscript_call_result_104343 = invoke(stypy.reporting.localization.Localization(__file__, 203, 14), getitem___104342, flag_104340)
    
    # Assigning a type to the variable 'ret' (line 203)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 203, 8), 'ret', subscript_call_result_104343)
    # SSA branch for the else part of an if statement (line 202)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Tuple to a Name (line 205):
    
    # Assigning a Tuple to a Name (line 205):
    
    # Obtaining an instance of the builtin type 'tuple' (line 205)
    tuple_104344 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 205, 15), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 205)
    # Adding element type (line 205)
    
    # Obtaining the type of the subscript
    # Getting the type of 'flag' (line 205)
    flag_104345 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 19), 'flag')
    # Getting the type of 'aux' (line 205)
    aux_104346 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 15), 'aux')
    # Obtaining the member '__getitem__' of a type (line 205)
    getitem___104347 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 205, 15), aux_104346, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 205)
    subscript_call_result_104348 = invoke(stypy.reporting.localization.Localization(__file__, 205, 15), getitem___104347, flag_104345)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 205, 15), tuple_104344, subscript_call_result_104348)
    
    # Assigning a type to the variable 'ret' (line 205)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 205, 8), 'ret', tuple_104344)
    
    # Getting the type of 'return_index' (line 206)
    return_index_104349 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 206, 11), 'return_index')
    # Testing the type of an if condition (line 206)
    if_condition_104350 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 206, 8), return_index_104349)
    # Assigning a type to the variable 'if_condition_104350' (line 206)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 206, 8), 'if_condition_104350', if_condition_104350)
    # SSA begins for if statement (line 206)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Getting the type of 'ret' (line 207)
    ret_104351 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 207, 12), 'ret')
    
    # Obtaining an instance of the builtin type 'tuple' (line 207)
    tuple_104352 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 207, 20), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 207)
    # Adding element type (line 207)
    
    # Obtaining the type of the subscript
    # Getting the type of 'flag' (line 207)
    flag_104353 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 207, 25), 'flag')
    # Getting the type of 'perm' (line 207)
    perm_104354 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 207, 20), 'perm')
    # Obtaining the member '__getitem__' of a type (line 207)
    getitem___104355 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 207, 20), perm_104354, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 207)
    subscript_call_result_104356 = invoke(stypy.reporting.localization.Localization(__file__, 207, 20), getitem___104355, flag_104353)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 207, 20), tuple_104352, subscript_call_result_104356)
    
    # Applying the binary operator '+=' (line 207)
    result_iadd_104357 = python_operator(stypy.reporting.localization.Localization(__file__, 207, 12), '+=', ret_104351, tuple_104352)
    # Assigning a type to the variable 'ret' (line 207)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 207, 12), 'ret', result_iadd_104357)
    
    # SSA join for if statement (line 206)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Getting the type of 'return_inverse' (line 208)
    return_inverse_104358 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 208, 11), 'return_inverse')
    # Testing the type of an if condition (line 208)
    if_condition_104359 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 208, 8), return_inverse_104358)
    # Assigning a type to the variable 'if_condition_104359' (line 208)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 208, 8), 'if_condition_104359', if_condition_104359)
    # SSA begins for if statement (line 208)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a BinOp to a Name (line 209):
    
    # Assigning a BinOp to a Name (line 209):
    
    # Call to cumsum(...): (line 209)
    # Processing the call arguments (line 209)
    # Getting the type of 'flag' (line 209)
    flag_104362 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 30), 'flag', False)
    # Processing the call keyword arguments (line 209)
    kwargs_104363 = {}
    # Getting the type of 'np' (line 209)
    np_104360 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 20), 'np', False)
    # Obtaining the member 'cumsum' of a type (line 209)
    cumsum_104361 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 209, 20), np_104360, 'cumsum')
    # Calling cumsum(args, kwargs) (line 209)
    cumsum_call_result_104364 = invoke(stypy.reporting.localization.Localization(__file__, 209, 20), cumsum_104361, *[flag_104362], **kwargs_104363)
    
    int_104365 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 209, 38), 'int')
    # Applying the binary operator '-' (line 209)
    result_sub_104366 = python_operator(stypy.reporting.localization.Localization(__file__, 209, 20), '-', cumsum_call_result_104364, int_104365)
    
    # Assigning a type to the variable 'iflag' (line 209)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 209, 12), 'iflag', result_sub_104366)
    
    # Assigning a Call to a Name (line 210):
    
    # Assigning a Call to a Name (line 210):
    
    # Call to empty(...): (line 210)
    # Processing the call arguments (line 210)
    # Getting the type of 'ar' (line 210)
    ar_104369 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 31), 'ar', False)
    # Obtaining the member 'shape' of a type (line 210)
    shape_104370 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 210, 31), ar_104369, 'shape')
    # Processing the call keyword arguments (line 210)
    # Getting the type of 'np' (line 210)
    np_104371 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 47), 'np', False)
    # Obtaining the member 'intp' of a type (line 210)
    intp_104372 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 210, 47), np_104371, 'intp')
    keyword_104373 = intp_104372
    kwargs_104374 = {'dtype': keyword_104373}
    # Getting the type of 'np' (line 210)
    np_104367 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 22), 'np', False)
    # Obtaining the member 'empty' of a type (line 210)
    empty_104368 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 210, 22), np_104367, 'empty')
    # Calling empty(args, kwargs) (line 210)
    empty_call_result_104375 = invoke(stypy.reporting.localization.Localization(__file__, 210, 22), empty_104368, *[shape_104370], **kwargs_104374)
    
    # Assigning a type to the variable 'inv_idx' (line 210)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 210, 12), 'inv_idx', empty_call_result_104375)
    
    # Assigning a Name to a Subscript (line 211):
    
    # Assigning a Name to a Subscript (line 211):
    # Getting the type of 'iflag' (line 211)
    iflag_104376 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 211, 28), 'iflag')
    # Getting the type of 'inv_idx' (line 211)
    inv_idx_104377 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 211, 12), 'inv_idx')
    # Getting the type of 'perm' (line 211)
    perm_104378 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 211, 20), 'perm')
    # Storing an element on a container (line 211)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 211, 12), inv_idx_104377, (perm_104378, iflag_104376))
    
    # Getting the type of 'ret' (line 212)
    ret_104379 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 12), 'ret')
    
    # Obtaining an instance of the builtin type 'tuple' (line 212)
    tuple_104380 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 212, 20), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 212)
    # Adding element type (line 212)
    # Getting the type of 'inv_idx' (line 212)
    inv_idx_104381 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 20), 'inv_idx')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 212, 20), tuple_104380, inv_idx_104381)
    
    # Applying the binary operator '+=' (line 212)
    result_iadd_104382 = python_operator(stypy.reporting.localization.Localization(__file__, 212, 12), '+=', ret_104379, tuple_104380)
    # Assigning a type to the variable 'ret' (line 212)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 212, 12), 'ret', result_iadd_104382)
    
    # SSA join for if statement (line 208)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Getting the type of 'return_counts' (line 213)
    return_counts_104383 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 11), 'return_counts')
    # Testing the type of an if condition (line 213)
    if_condition_104384 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 213, 8), return_counts_104383)
    # Assigning a type to the variable 'if_condition_104384' (line 213)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 213, 8), 'if_condition_104384', if_condition_104384)
    # SSA begins for if statement (line 213)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 214):
    
    # Assigning a Call to a Name (line 214):
    
    # Call to concatenate(...): (line 214)
    # Processing the call arguments (line 214)
    
    # Call to nonzero(...): (line 214)
    # Processing the call arguments (line 214)
    # Getting the type of 'flag' (line 214)
    flag_104389 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 44), 'flag', False)
    # Processing the call keyword arguments (line 214)
    kwargs_104390 = {}
    # Getting the type of 'np' (line 214)
    np_104387 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 33), 'np', False)
    # Obtaining the member 'nonzero' of a type (line 214)
    nonzero_104388 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 214, 33), np_104387, 'nonzero')
    # Calling nonzero(args, kwargs) (line 214)
    nonzero_call_result_104391 = invoke(stypy.reporting.localization.Localization(__file__, 214, 33), nonzero_104388, *[flag_104389], **kwargs_104390)
    
    
    # Obtaining an instance of the builtin type 'tuple' (line 214)
    tuple_104392 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 214, 53), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 214)
    # Adding element type (line 214)
    
    # Obtaining an instance of the builtin type 'list' (line 214)
    list_104393 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 214, 53), 'list')
    # Adding type elements to the builtin type 'list' instance (line 214)
    # Adding element type (line 214)
    # Getting the type of 'ar' (line 214)
    ar_104394 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 54), 'ar', False)
    # Obtaining the member 'size' of a type (line 214)
    size_104395 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 214, 54), ar_104394, 'size')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 214, 53), list_104393, size_104395)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 214, 53), tuple_104392, list_104393)
    
    # Applying the binary operator '+' (line 214)
    result_add_104396 = python_operator(stypy.reporting.localization.Localization(__file__, 214, 33), '+', nonzero_call_result_104391, tuple_104392)
    
    # Processing the call keyword arguments (line 214)
    kwargs_104397 = {}
    # Getting the type of 'np' (line 214)
    np_104385 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 18), 'np', False)
    # Obtaining the member 'concatenate' of a type (line 214)
    concatenate_104386 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 214, 18), np_104385, 'concatenate')
    # Calling concatenate(args, kwargs) (line 214)
    concatenate_call_result_104398 = invoke(stypy.reporting.localization.Localization(__file__, 214, 18), concatenate_104386, *[result_add_104396], **kwargs_104397)
    
    # Assigning a type to the variable 'idx' (line 214)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 214, 12), 'idx', concatenate_call_result_104398)
    
    # Getting the type of 'ret' (line 215)
    ret_104399 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 215, 12), 'ret')
    
    # Obtaining an instance of the builtin type 'tuple' (line 215)
    tuple_104400 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 215, 20), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 215)
    # Adding element type (line 215)
    
    # Call to diff(...): (line 215)
    # Processing the call arguments (line 215)
    # Getting the type of 'idx' (line 215)
    idx_104403 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 215, 28), 'idx', False)
    # Processing the call keyword arguments (line 215)
    kwargs_104404 = {}
    # Getting the type of 'np' (line 215)
    np_104401 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 215, 20), 'np', False)
    # Obtaining the member 'diff' of a type (line 215)
    diff_104402 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 215, 20), np_104401, 'diff')
    # Calling diff(args, kwargs) (line 215)
    diff_call_result_104405 = invoke(stypy.reporting.localization.Localization(__file__, 215, 20), diff_104402, *[idx_104403], **kwargs_104404)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 215, 20), tuple_104400, diff_call_result_104405)
    
    # Applying the binary operator '+=' (line 215)
    result_iadd_104406 = python_operator(stypy.reporting.localization.Localization(__file__, 215, 12), '+=', ret_104399, tuple_104400)
    # Assigning a type to the variable 'ret' (line 215)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 215, 12), 'ret', result_iadd_104406)
    
    # SSA join for if statement (line 213)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 202)
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'ret' (line 216)
    ret_104407 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 216, 11), 'ret')
    # Assigning a type to the variable 'stypy_return_type' (line 216)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 216, 4), 'stypy_return_type', ret_104407)
    
    # ################# End of 'unique(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'unique' in the type store
    # Getting the type of 'stypy_return_type' (line 96)
    stypy_return_type_104408 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_104408)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'unique'
    return stypy_return_type_104408

# Assigning a type to the variable 'unique' (line 96)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 96, 0), 'unique', unique)

@norecursion
def intersect1d(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'False' (line 218)
    False_104409 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 218, 40), 'False')
    defaults = [False_104409]
    # Create a new context for function 'intersect1d'
    module_type_store = module_type_store.open_function_context('intersect1d', 218, 0, False)
    
    # Passed parameters checking function
    intersect1d.stypy_localization = localization
    intersect1d.stypy_type_of_self = None
    intersect1d.stypy_type_store = module_type_store
    intersect1d.stypy_function_name = 'intersect1d'
    intersect1d.stypy_param_names_list = ['ar1', 'ar2', 'assume_unique']
    intersect1d.stypy_varargs_param_name = None
    intersect1d.stypy_kwargs_param_name = None
    intersect1d.stypy_call_defaults = defaults
    intersect1d.stypy_call_varargs = varargs
    intersect1d.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'intersect1d', ['ar1', 'ar2', 'assume_unique'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'intersect1d', localization, ['ar1', 'ar2', 'assume_unique'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'intersect1d(...)' code ##################

    str_104410 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 252, (-1)), 'str', '\n    Find the intersection of two arrays.\n\n    Return the sorted, unique values that are in both of the input arrays.\n\n    Parameters\n    ----------\n    ar1, ar2 : array_like\n        Input arrays.\n    assume_unique : bool\n        If True, the input arrays are both assumed to be unique, which\n        can speed up the calculation.  Default is False.\n\n    Returns\n    -------\n    intersect1d : ndarray\n        Sorted 1D array of common and unique elements.\n\n    See Also\n    --------\n    numpy.lib.arraysetops : Module with a number of other functions for\n                            performing set operations on arrays.\n\n    Examples\n    --------\n    >>> np.intersect1d([1, 3, 4, 3], [3, 1, 2, 1])\n    array([1, 3])\n\n    To intersect more than two arrays, use functools.reduce:\n\n    >>> from functools import reduce\n    >>> reduce(np.intersect1d, ([1, 3, 4, 3], [3, 1, 2, 1], [6, 3, 4, 2]))\n    array([3])\n    ')
    
    
    # Getting the type of 'assume_unique' (line 253)
    assume_unique_104411 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 253, 11), 'assume_unique')
    # Applying the 'not' unary operator (line 253)
    result_not__104412 = python_operator(stypy.reporting.localization.Localization(__file__, 253, 7), 'not', assume_unique_104411)
    
    # Testing the type of an if condition (line 253)
    if_condition_104413 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 253, 4), result_not__104412)
    # Assigning a type to the variable 'if_condition_104413' (line 253)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 253, 4), 'if_condition_104413', if_condition_104413)
    # SSA begins for if statement (line 253)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 255):
    
    # Assigning a Call to a Name (line 255):
    
    # Call to unique(...): (line 255)
    # Processing the call arguments (line 255)
    # Getting the type of 'ar1' (line 255)
    ar1_104415 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 255, 21), 'ar1', False)
    # Processing the call keyword arguments (line 255)
    kwargs_104416 = {}
    # Getting the type of 'unique' (line 255)
    unique_104414 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 255, 14), 'unique', False)
    # Calling unique(args, kwargs) (line 255)
    unique_call_result_104417 = invoke(stypy.reporting.localization.Localization(__file__, 255, 14), unique_104414, *[ar1_104415], **kwargs_104416)
    
    # Assigning a type to the variable 'ar1' (line 255)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 255, 8), 'ar1', unique_call_result_104417)
    
    # Assigning a Call to a Name (line 256):
    
    # Assigning a Call to a Name (line 256):
    
    # Call to unique(...): (line 256)
    # Processing the call arguments (line 256)
    # Getting the type of 'ar2' (line 256)
    ar2_104419 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 256, 21), 'ar2', False)
    # Processing the call keyword arguments (line 256)
    kwargs_104420 = {}
    # Getting the type of 'unique' (line 256)
    unique_104418 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 256, 14), 'unique', False)
    # Calling unique(args, kwargs) (line 256)
    unique_call_result_104421 = invoke(stypy.reporting.localization.Localization(__file__, 256, 14), unique_104418, *[ar2_104419], **kwargs_104420)
    
    # Assigning a type to the variable 'ar2' (line 256)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 256, 8), 'ar2', unique_call_result_104421)
    # SSA join for if statement (line 253)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 257):
    
    # Assigning a Call to a Name (line 257):
    
    # Call to concatenate(...): (line 257)
    # Processing the call arguments (line 257)
    
    # Obtaining an instance of the builtin type 'tuple' (line 257)
    tuple_104424 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 257, 26), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 257)
    # Adding element type (line 257)
    # Getting the type of 'ar1' (line 257)
    ar1_104425 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 257, 26), 'ar1', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 257, 26), tuple_104424, ar1_104425)
    # Adding element type (line 257)
    # Getting the type of 'ar2' (line 257)
    ar2_104426 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 257, 31), 'ar2', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 257, 26), tuple_104424, ar2_104426)
    
    # Processing the call keyword arguments (line 257)
    kwargs_104427 = {}
    # Getting the type of 'np' (line 257)
    np_104422 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 257, 10), 'np', False)
    # Obtaining the member 'concatenate' of a type (line 257)
    concatenate_104423 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 257, 10), np_104422, 'concatenate')
    # Calling concatenate(args, kwargs) (line 257)
    concatenate_call_result_104428 = invoke(stypy.reporting.localization.Localization(__file__, 257, 10), concatenate_104423, *[tuple_104424], **kwargs_104427)
    
    # Assigning a type to the variable 'aux' (line 257)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 257, 4), 'aux', concatenate_call_result_104428)
    
    # Call to sort(...): (line 258)
    # Processing the call keyword arguments (line 258)
    kwargs_104431 = {}
    # Getting the type of 'aux' (line 258)
    aux_104429 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 258, 4), 'aux', False)
    # Obtaining the member 'sort' of a type (line 258)
    sort_104430 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 258, 4), aux_104429, 'sort')
    # Calling sort(args, kwargs) (line 258)
    sort_call_result_104432 = invoke(stypy.reporting.localization.Localization(__file__, 258, 4), sort_104430, *[], **kwargs_104431)
    
    
    # Obtaining the type of the subscript
    
    
    # Obtaining the type of the subscript
    int_104433 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 259, 24), 'int')
    slice_104434 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 259, 20), int_104433, None, None)
    # Getting the type of 'aux' (line 259)
    aux_104435 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 259, 20), 'aux')
    # Obtaining the member '__getitem__' of a type (line 259)
    getitem___104436 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 259, 20), aux_104435, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 259)
    subscript_call_result_104437 = invoke(stypy.reporting.localization.Localization(__file__, 259, 20), getitem___104436, slice_104434)
    
    
    # Obtaining the type of the subscript
    int_104438 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 259, 36), 'int')
    slice_104439 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 259, 31), None, int_104438, None)
    # Getting the type of 'aux' (line 259)
    aux_104440 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 259, 31), 'aux')
    # Obtaining the member '__getitem__' of a type (line 259)
    getitem___104441 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 259, 31), aux_104440, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 259)
    subscript_call_result_104442 = invoke(stypy.reporting.localization.Localization(__file__, 259, 31), getitem___104441, slice_104439)
    
    # Applying the binary operator '==' (line 259)
    result_eq_104443 = python_operator(stypy.reporting.localization.Localization(__file__, 259, 20), '==', subscript_call_result_104437, subscript_call_result_104442)
    
    
    # Obtaining the type of the subscript
    int_104444 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 259, 16), 'int')
    slice_104445 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 259, 11), None, int_104444, None)
    # Getting the type of 'aux' (line 259)
    aux_104446 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 259, 11), 'aux')
    # Obtaining the member '__getitem__' of a type (line 259)
    getitem___104447 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 259, 11), aux_104446, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 259)
    subscript_call_result_104448 = invoke(stypy.reporting.localization.Localization(__file__, 259, 11), getitem___104447, slice_104445)
    
    # Obtaining the member '__getitem__' of a type (line 259)
    getitem___104449 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 259, 11), subscript_call_result_104448, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 259)
    subscript_call_result_104450 = invoke(stypy.reporting.localization.Localization(__file__, 259, 11), getitem___104449, result_eq_104443)
    
    # Assigning a type to the variable 'stypy_return_type' (line 259)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 259, 4), 'stypy_return_type', subscript_call_result_104450)
    
    # ################# End of 'intersect1d(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'intersect1d' in the type store
    # Getting the type of 'stypy_return_type' (line 218)
    stypy_return_type_104451 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 218, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_104451)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'intersect1d'
    return stypy_return_type_104451

# Assigning a type to the variable 'intersect1d' (line 218)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 218, 0), 'intersect1d', intersect1d)

@norecursion
def setxor1d(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'False' (line 261)
    False_104452 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 261, 37), 'False')
    defaults = [False_104452]
    # Create a new context for function 'setxor1d'
    module_type_store = module_type_store.open_function_context('setxor1d', 261, 0, False)
    
    # Passed parameters checking function
    setxor1d.stypy_localization = localization
    setxor1d.stypy_type_of_self = None
    setxor1d.stypy_type_store = module_type_store
    setxor1d.stypy_function_name = 'setxor1d'
    setxor1d.stypy_param_names_list = ['ar1', 'ar2', 'assume_unique']
    setxor1d.stypy_varargs_param_name = None
    setxor1d.stypy_kwargs_param_name = None
    setxor1d.stypy_call_defaults = defaults
    setxor1d.stypy_call_varargs = varargs
    setxor1d.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'setxor1d', ['ar1', 'ar2', 'assume_unique'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'setxor1d', localization, ['ar1', 'ar2', 'assume_unique'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'setxor1d(...)' code ##################

    str_104453 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 289, (-1)), 'str', '\n    Find the set exclusive-or of two arrays.\n\n    Return the sorted, unique values that are in only one (not both) of the\n    input arrays.\n\n    Parameters\n    ----------\n    ar1, ar2 : array_like\n        Input arrays.\n    assume_unique : bool\n        If True, the input arrays are both assumed to be unique, which\n        can speed up the calculation.  Default is False.\n\n    Returns\n    -------\n    setxor1d : ndarray\n        Sorted 1D array of unique values that are in only one of the input\n        arrays.\n\n    Examples\n    --------\n    >>> a = np.array([1, 2, 3, 2, 4])\n    >>> b = np.array([2, 3, 5, 7, 5])\n    >>> np.setxor1d(a,b)\n    array([1, 4, 5, 7])\n\n    ')
    
    
    # Getting the type of 'assume_unique' (line 290)
    assume_unique_104454 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 290, 11), 'assume_unique')
    # Applying the 'not' unary operator (line 290)
    result_not__104455 = python_operator(stypy.reporting.localization.Localization(__file__, 290, 7), 'not', assume_unique_104454)
    
    # Testing the type of an if condition (line 290)
    if_condition_104456 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 290, 4), result_not__104455)
    # Assigning a type to the variable 'if_condition_104456' (line 290)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 290, 4), 'if_condition_104456', if_condition_104456)
    # SSA begins for if statement (line 290)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 291):
    
    # Assigning a Call to a Name (line 291):
    
    # Call to unique(...): (line 291)
    # Processing the call arguments (line 291)
    # Getting the type of 'ar1' (line 291)
    ar1_104458 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 291, 21), 'ar1', False)
    # Processing the call keyword arguments (line 291)
    kwargs_104459 = {}
    # Getting the type of 'unique' (line 291)
    unique_104457 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 291, 14), 'unique', False)
    # Calling unique(args, kwargs) (line 291)
    unique_call_result_104460 = invoke(stypy.reporting.localization.Localization(__file__, 291, 14), unique_104457, *[ar1_104458], **kwargs_104459)
    
    # Assigning a type to the variable 'ar1' (line 291)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 291, 8), 'ar1', unique_call_result_104460)
    
    # Assigning a Call to a Name (line 292):
    
    # Assigning a Call to a Name (line 292):
    
    # Call to unique(...): (line 292)
    # Processing the call arguments (line 292)
    # Getting the type of 'ar2' (line 292)
    ar2_104462 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 292, 21), 'ar2', False)
    # Processing the call keyword arguments (line 292)
    kwargs_104463 = {}
    # Getting the type of 'unique' (line 292)
    unique_104461 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 292, 14), 'unique', False)
    # Calling unique(args, kwargs) (line 292)
    unique_call_result_104464 = invoke(stypy.reporting.localization.Localization(__file__, 292, 14), unique_104461, *[ar2_104462], **kwargs_104463)
    
    # Assigning a type to the variable 'ar2' (line 292)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 292, 8), 'ar2', unique_call_result_104464)
    # SSA join for if statement (line 290)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 294):
    
    # Assigning a Call to a Name (line 294):
    
    # Call to concatenate(...): (line 294)
    # Processing the call arguments (line 294)
    
    # Obtaining an instance of the builtin type 'tuple' (line 294)
    tuple_104467 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 294, 26), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 294)
    # Adding element type (line 294)
    # Getting the type of 'ar1' (line 294)
    ar1_104468 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 294, 26), 'ar1', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 294, 26), tuple_104467, ar1_104468)
    # Adding element type (line 294)
    # Getting the type of 'ar2' (line 294)
    ar2_104469 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 294, 31), 'ar2', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 294, 26), tuple_104467, ar2_104469)
    
    # Processing the call keyword arguments (line 294)
    kwargs_104470 = {}
    # Getting the type of 'np' (line 294)
    np_104465 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 294, 10), 'np', False)
    # Obtaining the member 'concatenate' of a type (line 294)
    concatenate_104466 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 294, 10), np_104465, 'concatenate')
    # Calling concatenate(args, kwargs) (line 294)
    concatenate_call_result_104471 = invoke(stypy.reporting.localization.Localization(__file__, 294, 10), concatenate_104466, *[tuple_104467], **kwargs_104470)
    
    # Assigning a type to the variable 'aux' (line 294)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 294, 4), 'aux', concatenate_call_result_104471)
    
    
    # Getting the type of 'aux' (line 295)
    aux_104472 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 295, 7), 'aux')
    # Obtaining the member 'size' of a type (line 295)
    size_104473 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 295, 7), aux_104472, 'size')
    int_104474 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 295, 19), 'int')
    # Applying the binary operator '==' (line 295)
    result_eq_104475 = python_operator(stypy.reporting.localization.Localization(__file__, 295, 7), '==', size_104473, int_104474)
    
    # Testing the type of an if condition (line 295)
    if_condition_104476 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 295, 4), result_eq_104475)
    # Assigning a type to the variable 'if_condition_104476' (line 295)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 295, 4), 'if_condition_104476', if_condition_104476)
    # SSA begins for if statement (line 295)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # Getting the type of 'aux' (line 296)
    aux_104477 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 296, 15), 'aux')
    # Assigning a type to the variable 'stypy_return_type' (line 296)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 296, 8), 'stypy_return_type', aux_104477)
    # SSA join for if statement (line 295)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to sort(...): (line 298)
    # Processing the call keyword arguments (line 298)
    kwargs_104480 = {}
    # Getting the type of 'aux' (line 298)
    aux_104478 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 298, 4), 'aux', False)
    # Obtaining the member 'sort' of a type (line 298)
    sort_104479 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 298, 4), aux_104478, 'sort')
    # Calling sort(args, kwargs) (line 298)
    sort_call_result_104481 = invoke(stypy.reporting.localization.Localization(__file__, 298, 4), sort_104479, *[], **kwargs_104480)
    
    
    # Assigning a Call to a Name (line 300):
    
    # Assigning a Call to a Name (line 300):
    
    # Call to concatenate(...): (line 300)
    # Processing the call arguments (line 300)
    
    # Obtaining an instance of the builtin type 'tuple' (line 300)
    tuple_104484 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 300, 27), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 300)
    # Adding element type (line 300)
    
    # Obtaining an instance of the builtin type 'list' (line 300)
    list_104485 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 300, 27), 'list')
    # Adding type elements to the builtin type 'list' instance (line 300)
    # Adding element type (line 300)
    # Getting the type of 'True' (line 300)
    True_104486 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 300, 28), 'True', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 300, 27), list_104485, True_104486)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 300, 27), tuple_104484, list_104485)
    # Adding element type (line 300)
    
    
    # Obtaining the type of the subscript
    int_104487 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 300, 39), 'int')
    slice_104488 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 300, 35), int_104487, None, None)
    # Getting the type of 'aux' (line 300)
    aux_104489 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 300, 35), 'aux', False)
    # Obtaining the member '__getitem__' of a type (line 300)
    getitem___104490 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 300, 35), aux_104489, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 300)
    subscript_call_result_104491 = invoke(stypy.reporting.localization.Localization(__file__, 300, 35), getitem___104490, slice_104488)
    
    
    # Obtaining the type of the subscript
    int_104492 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 300, 51), 'int')
    slice_104493 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 300, 46), None, int_104492, None)
    # Getting the type of 'aux' (line 300)
    aux_104494 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 300, 46), 'aux', False)
    # Obtaining the member '__getitem__' of a type (line 300)
    getitem___104495 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 300, 46), aux_104494, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 300)
    subscript_call_result_104496 = invoke(stypy.reporting.localization.Localization(__file__, 300, 46), getitem___104495, slice_104493)
    
    # Applying the binary operator '!=' (line 300)
    result_ne_104497 = python_operator(stypy.reporting.localization.Localization(__file__, 300, 35), '!=', subscript_call_result_104491, subscript_call_result_104496)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 300, 27), tuple_104484, result_ne_104497)
    # Adding element type (line 300)
    
    # Obtaining an instance of the builtin type 'list' (line 300)
    list_104498 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 300, 56), 'list')
    # Adding type elements to the builtin type 'list' instance (line 300)
    # Adding element type (line 300)
    # Getting the type of 'True' (line 300)
    True_104499 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 300, 57), 'True', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 300, 56), list_104498, True_104499)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 300, 27), tuple_104484, list_104498)
    
    # Processing the call keyword arguments (line 300)
    kwargs_104500 = {}
    # Getting the type of 'np' (line 300)
    np_104482 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 300, 11), 'np', False)
    # Obtaining the member 'concatenate' of a type (line 300)
    concatenate_104483 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 300, 11), np_104482, 'concatenate')
    # Calling concatenate(args, kwargs) (line 300)
    concatenate_call_result_104501 = invoke(stypy.reporting.localization.Localization(__file__, 300, 11), concatenate_104483, *[tuple_104484], **kwargs_104500)
    
    # Assigning a type to the variable 'flag' (line 300)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 300, 4), 'flag', concatenate_call_result_104501)
    
    # Assigning a Compare to a Name (line 302):
    
    # Assigning a Compare to a Name (line 302):
    
    
    # Obtaining the type of the subscript
    int_104502 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 302, 17), 'int')
    slice_104503 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 302, 12), int_104502, None, None)
    # Getting the type of 'flag' (line 302)
    flag_104504 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 302, 12), 'flag')
    # Obtaining the member '__getitem__' of a type (line 302)
    getitem___104505 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 302, 12), flag_104504, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 302)
    subscript_call_result_104506 = invoke(stypy.reporting.localization.Localization(__file__, 302, 12), getitem___104505, slice_104503)
    
    
    # Obtaining the type of the subscript
    int_104507 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 302, 30), 'int')
    slice_104508 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 302, 24), None, int_104507, None)
    # Getting the type of 'flag' (line 302)
    flag_104509 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 302, 24), 'flag')
    # Obtaining the member '__getitem__' of a type (line 302)
    getitem___104510 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 302, 24), flag_104509, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 302)
    subscript_call_result_104511 = invoke(stypy.reporting.localization.Localization(__file__, 302, 24), getitem___104510, slice_104508)
    
    # Applying the binary operator '==' (line 302)
    result_eq_104512 = python_operator(stypy.reporting.localization.Localization(__file__, 302, 12), '==', subscript_call_result_104506, subscript_call_result_104511)
    
    # Assigning a type to the variable 'flag2' (line 302)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 302, 4), 'flag2', result_eq_104512)
    
    # Obtaining the type of the subscript
    # Getting the type of 'flag2' (line 303)
    flag2_104513 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 303, 15), 'flag2')
    # Getting the type of 'aux' (line 303)
    aux_104514 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 303, 11), 'aux')
    # Obtaining the member '__getitem__' of a type (line 303)
    getitem___104515 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 303, 11), aux_104514, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 303)
    subscript_call_result_104516 = invoke(stypy.reporting.localization.Localization(__file__, 303, 11), getitem___104515, flag2_104513)
    
    # Assigning a type to the variable 'stypy_return_type' (line 303)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 303, 4), 'stypy_return_type', subscript_call_result_104516)
    
    # ################# End of 'setxor1d(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'setxor1d' in the type store
    # Getting the type of 'stypy_return_type' (line 261)
    stypy_return_type_104517 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 261, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_104517)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'setxor1d'
    return stypy_return_type_104517

# Assigning a type to the variable 'setxor1d' (line 261)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 261, 0), 'setxor1d', setxor1d)

@norecursion
def in1d(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'False' (line 305)
    False_104518 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 305, 33), 'False')
    # Getting the type of 'False' (line 305)
    False_104519 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 305, 47), 'False')
    defaults = [False_104518, False_104519]
    # Create a new context for function 'in1d'
    module_type_store = module_type_store.open_function_context('in1d', 305, 0, False)
    
    # Passed parameters checking function
    in1d.stypy_localization = localization
    in1d.stypy_type_of_self = None
    in1d.stypy_type_store = module_type_store
    in1d.stypy_function_name = 'in1d'
    in1d.stypy_param_names_list = ['ar1', 'ar2', 'assume_unique', 'invert']
    in1d.stypy_varargs_param_name = None
    in1d.stypy_kwargs_param_name = None
    in1d.stypy_call_defaults = defaults
    in1d.stypy_call_varargs = varargs
    in1d.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'in1d', ['ar1', 'ar2', 'assume_unique', 'invert'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'in1d', localization, ['ar1', 'ar2', 'assume_unique', 'invert'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'in1d(...)' code ##################

    str_104520 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 365, (-1)), 'str', '\n    Test whether each element of a 1-D array is also present in a second array.\n\n    Returns a boolean array the same length as `ar1` that is True\n    where an element of `ar1` is in `ar2` and False otherwise.\n\n    Parameters\n    ----------\n    ar1 : (M,) array_like\n        Input array.\n    ar2 : array_like\n        The values against which to test each value of `ar1`.\n    assume_unique : bool, optional\n        If True, the input arrays are both assumed to be unique, which\n        can speed up the calculation.  Default is False.\n    invert : bool, optional\n        If True, the values in the returned array are inverted (that is,\n        False where an element of `ar1` is in `ar2` and True otherwise).\n        Default is False. ``np.in1d(a, b, invert=True)`` is equivalent\n        to (but is faster than) ``np.invert(in1d(a, b))``.\n\n        .. versionadded:: 1.8.0\n\n    Returns\n    -------\n    in1d : (M,) ndarray, bool\n        The values `ar1[in1d]` are in `ar2`.\n\n    See Also\n    --------\n    numpy.lib.arraysetops : Module with a number of other functions for\n                            performing set operations on arrays.\n\n    Notes\n    -----\n    `in1d` can be considered as an element-wise function version of the\n    python keyword `in`, for 1-D sequences. ``in1d(a, b)`` is roughly\n    equivalent to ``np.array([item in b for item in a])``.\n    However, this idea fails if `ar2` is a set, or similar (non-sequence)\n    container:  As ``ar2`` is converted to an array, in those cases\n    ``asarray(ar2)`` is an object array rather than the expected array of\n    contained values.\n\n    .. versionadded:: 1.4.0\n\n    Examples\n    --------\n    >>> test = np.array([0, 1, 2, 5, 0])\n    >>> states = [0, 2]\n    >>> mask = np.in1d(test, states)\n    >>> mask\n    array([ True, False,  True, False,  True], dtype=bool)\n    >>> test[mask]\n    array([0, 2, 0])\n    >>> mask = np.in1d(test, states, invert=True)\n    >>> mask\n    array([False,  True, False,  True, False], dtype=bool)\n    >>> test[mask]\n    array([1, 5])\n    ')
    
    # Assigning a Call to a Name (line 367):
    
    # Assigning a Call to a Name (line 367):
    
    # Call to ravel(...): (line 367)
    # Processing the call keyword arguments (line 367)
    kwargs_104527 = {}
    
    # Call to asarray(...): (line 367)
    # Processing the call arguments (line 367)
    # Getting the type of 'ar1' (line 367)
    ar1_104523 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 367, 21), 'ar1', False)
    # Processing the call keyword arguments (line 367)
    kwargs_104524 = {}
    # Getting the type of 'np' (line 367)
    np_104521 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 367, 10), 'np', False)
    # Obtaining the member 'asarray' of a type (line 367)
    asarray_104522 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 367, 10), np_104521, 'asarray')
    # Calling asarray(args, kwargs) (line 367)
    asarray_call_result_104525 = invoke(stypy.reporting.localization.Localization(__file__, 367, 10), asarray_104522, *[ar1_104523], **kwargs_104524)
    
    # Obtaining the member 'ravel' of a type (line 367)
    ravel_104526 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 367, 10), asarray_call_result_104525, 'ravel')
    # Calling ravel(args, kwargs) (line 367)
    ravel_call_result_104528 = invoke(stypy.reporting.localization.Localization(__file__, 367, 10), ravel_104526, *[], **kwargs_104527)
    
    # Assigning a type to the variable 'ar1' (line 367)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 367, 4), 'ar1', ravel_call_result_104528)
    
    # Assigning a Call to a Name (line 368):
    
    # Assigning a Call to a Name (line 368):
    
    # Call to ravel(...): (line 368)
    # Processing the call keyword arguments (line 368)
    kwargs_104535 = {}
    
    # Call to asarray(...): (line 368)
    # Processing the call arguments (line 368)
    # Getting the type of 'ar2' (line 368)
    ar2_104531 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 368, 21), 'ar2', False)
    # Processing the call keyword arguments (line 368)
    kwargs_104532 = {}
    # Getting the type of 'np' (line 368)
    np_104529 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 368, 10), 'np', False)
    # Obtaining the member 'asarray' of a type (line 368)
    asarray_104530 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 368, 10), np_104529, 'asarray')
    # Calling asarray(args, kwargs) (line 368)
    asarray_call_result_104533 = invoke(stypy.reporting.localization.Localization(__file__, 368, 10), asarray_104530, *[ar2_104531], **kwargs_104532)
    
    # Obtaining the member 'ravel' of a type (line 368)
    ravel_104534 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 368, 10), asarray_call_result_104533, 'ravel')
    # Calling ravel(args, kwargs) (line 368)
    ravel_call_result_104536 = invoke(stypy.reporting.localization.Localization(__file__, 368, 10), ravel_104534, *[], **kwargs_104535)
    
    # Assigning a type to the variable 'ar2' (line 368)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 368, 4), 'ar2', ravel_call_result_104536)
    
    
    
    # Call to len(...): (line 371)
    # Processing the call arguments (line 371)
    # Getting the type of 'ar2' (line 371)
    ar2_104538 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 371, 11), 'ar2', False)
    # Processing the call keyword arguments (line 371)
    kwargs_104539 = {}
    # Getting the type of 'len' (line 371)
    len_104537 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 371, 7), 'len', False)
    # Calling len(args, kwargs) (line 371)
    len_call_result_104540 = invoke(stypy.reporting.localization.Localization(__file__, 371, 7), len_104537, *[ar2_104538], **kwargs_104539)
    
    int_104541 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 371, 18), 'int')
    
    # Call to len(...): (line 371)
    # Processing the call arguments (line 371)
    # Getting the type of 'ar1' (line 371)
    ar1_104543 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 371, 27), 'ar1', False)
    # Processing the call keyword arguments (line 371)
    kwargs_104544 = {}
    # Getting the type of 'len' (line 371)
    len_104542 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 371, 23), 'len', False)
    # Calling len(args, kwargs) (line 371)
    len_call_result_104545 = invoke(stypy.reporting.localization.Localization(__file__, 371, 23), len_104542, *[ar1_104543], **kwargs_104544)
    
    float_104546 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 371, 35), 'float')
    # Applying the binary operator '**' (line 371)
    result_pow_104547 = python_operator(stypy.reporting.localization.Localization(__file__, 371, 23), '**', len_call_result_104545, float_104546)
    
    # Applying the binary operator '*' (line 371)
    result_mul_104548 = python_operator(stypy.reporting.localization.Localization(__file__, 371, 18), '*', int_104541, result_pow_104547)
    
    # Applying the binary operator '<' (line 371)
    result_lt_104549 = python_operator(stypy.reporting.localization.Localization(__file__, 371, 7), '<', len_call_result_104540, result_mul_104548)
    
    # Testing the type of an if condition (line 371)
    if_condition_104550 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 371, 4), result_lt_104549)
    # Assigning a type to the variable 'if_condition_104550' (line 371)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 371, 4), 'if_condition_104550', if_condition_104550)
    # SSA begins for if statement (line 371)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Getting the type of 'invert' (line 372)
    invert_104551 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 372, 11), 'invert')
    # Testing the type of an if condition (line 372)
    if_condition_104552 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 372, 8), invert_104551)
    # Assigning a type to the variable 'if_condition_104552' (line 372)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 372, 8), 'if_condition_104552', if_condition_104552)
    # SSA begins for if statement (line 372)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 373):
    
    # Assigning a Call to a Name (line 373):
    
    # Call to ones(...): (line 373)
    # Processing the call arguments (line 373)
    
    # Call to len(...): (line 373)
    # Processing the call arguments (line 373)
    # Getting the type of 'ar1' (line 373)
    ar1_104556 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 373, 31), 'ar1', False)
    # Processing the call keyword arguments (line 373)
    kwargs_104557 = {}
    # Getting the type of 'len' (line 373)
    len_104555 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 373, 27), 'len', False)
    # Calling len(args, kwargs) (line 373)
    len_call_result_104558 = invoke(stypy.reporting.localization.Localization(__file__, 373, 27), len_104555, *[ar1_104556], **kwargs_104557)
    
    # Processing the call keyword arguments (line 373)
    # Getting the type of 'np' (line 373)
    np_104559 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 373, 43), 'np', False)
    # Obtaining the member 'bool' of a type (line 373)
    bool_104560 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 373, 43), np_104559, 'bool')
    keyword_104561 = bool_104560
    kwargs_104562 = {'dtype': keyword_104561}
    # Getting the type of 'np' (line 373)
    np_104553 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 373, 19), 'np', False)
    # Obtaining the member 'ones' of a type (line 373)
    ones_104554 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 373, 19), np_104553, 'ones')
    # Calling ones(args, kwargs) (line 373)
    ones_call_result_104563 = invoke(stypy.reporting.localization.Localization(__file__, 373, 19), ones_104554, *[len_call_result_104558], **kwargs_104562)
    
    # Assigning a type to the variable 'mask' (line 373)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 373, 12), 'mask', ones_call_result_104563)
    
    # Getting the type of 'ar2' (line 374)
    ar2_104564 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 374, 21), 'ar2')
    # Testing the type of a for loop iterable (line 374)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 374, 12), ar2_104564)
    # Getting the type of the for loop variable (line 374)
    for_loop_var_104565 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 374, 12), ar2_104564)
    # Assigning a type to the variable 'a' (line 374)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 374, 12), 'a', for_loop_var_104565)
    # SSA begins for a for statement (line 374)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Getting the type of 'mask' (line 375)
    mask_104566 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 375, 16), 'mask')
    
    # Getting the type of 'ar1' (line 375)
    ar1_104567 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 375, 25), 'ar1')
    # Getting the type of 'a' (line 375)
    a_104568 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 375, 32), 'a')
    # Applying the binary operator '!=' (line 375)
    result_ne_104569 = python_operator(stypy.reporting.localization.Localization(__file__, 375, 25), '!=', ar1_104567, a_104568)
    
    # Applying the binary operator '&=' (line 375)
    result_iand_104570 = python_operator(stypy.reporting.localization.Localization(__file__, 375, 16), '&=', mask_104566, result_ne_104569)
    # Assigning a type to the variable 'mask' (line 375)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 375, 16), 'mask', result_iand_104570)
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA branch for the else part of an if statement (line 372)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Call to a Name (line 377):
    
    # Assigning a Call to a Name (line 377):
    
    # Call to zeros(...): (line 377)
    # Processing the call arguments (line 377)
    
    # Call to len(...): (line 377)
    # Processing the call arguments (line 377)
    # Getting the type of 'ar1' (line 377)
    ar1_104574 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 377, 32), 'ar1', False)
    # Processing the call keyword arguments (line 377)
    kwargs_104575 = {}
    # Getting the type of 'len' (line 377)
    len_104573 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 377, 28), 'len', False)
    # Calling len(args, kwargs) (line 377)
    len_call_result_104576 = invoke(stypy.reporting.localization.Localization(__file__, 377, 28), len_104573, *[ar1_104574], **kwargs_104575)
    
    # Processing the call keyword arguments (line 377)
    # Getting the type of 'np' (line 377)
    np_104577 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 377, 44), 'np', False)
    # Obtaining the member 'bool' of a type (line 377)
    bool_104578 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 377, 44), np_104577, 'bool')
    keyword_104579 = bool_104578
    kwargs_104580 = {'dtype': keyword_104579}
    # Getting the type of 'np' (line 377)
    np_104571 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 377, 19), 'np', False)
    # Obtaining the member 'zeros' of a type (line 377)
    zeros_104572 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 377, 19), np_104571, 'zeros')
    # Calling zeros(args, kwargs) (line 377)
    zeros_call_result_104581 = invoke(stypy.reporting.localization.Localization(__file__, 377, 19), zeros_104572, *[len_call_result_104576], **kwargs_104580)
    
    # Assigning a type to the variable 'mask' (line 377)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 377, 12), 'mask', zeros_call_result_104581)
    
    # Getting the type of 'ar2' (line 378)
    ar2_104582 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 378, 21), 'ar2')
    # Testing the type of a for loop iterable (line 378)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 378, 12), ar2_104582)
    # Getting the type of the for loop variable (line 378)
    for_loop_var_104583 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 378, 12), ar2_104582)
    # Assigning a type to the variable 'a' (line 378)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 378, 12), 'a', for_loop_var_104583)
    # SSA begins for a for statement (line 378)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Getting the type of 'mask' (line 379)
    mask_104584 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 379, 16), 'mask')
    
    # Getting the type of 'ar1' (line 379)
    ar1_104585 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 379, 25), 'ar1')
    # Getting the type of 'a' (line 379)
    a_104586 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 379, 32), 'a')
    # Applying the binary operator '==' (line 379)
    result_eq_104587 = python_operator(stypy.reporting.localization.Localization(__file__, 379, 25), '==', ar1_104585, a_104586)
    
    # Applying the binary operator '|=' (line 379)
    result_ior_104588 = python_operator(stypy.reporting.localization.Localization(__file__, 379, 16), '|=', mask_104584, result_eq_104587)
    # Assigning a type to the variable 'mask' (line 379)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 379, 16), 'mask', result_ior_104588)
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 372)
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'mask' (line 380)
    mask_104589 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 380, 15), 'mask')
    # Assigning a type to the variable 'stypy_return_type' (line 380)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 380, 8), 'stypy_return_type', mask_104589)
    # SSA join for if statement (line 371)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'assume_unique' (line 383)
    assume_unique_104590 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 383, 11), 'assume_unique')
    # Applying the 'not' unary operator (line 383)
    result_not__104591 = python_operator(stypy.reporting.localization.Localization(__file__, 383, 7), 'not', assume_unique_104590)
    
    # Testing the type of an if condition (line 383)
    if_condition_104592 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 383, 4), result_not__104591)
    # Assigning a type to the variable 'if_condition_104592' (line 383)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 383, 4), 'if_condition_104592', if_condition_104592)
    # SSA begins for if statement (line 383)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Tuple (line 384):
    
    # Assigning a Call to a Name:
    
    # Call to unique(...): (line 384)
    # Processing the call arguments (line 384)
    # Getting the type of 'ar1' (line 384)
    ar1_104595 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 384, 33), 'ar1', False)
    # Processing the call keyword arguments (line 384)
    # Getting the type of 'True' (line 384)
    True_104596 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 384, 53), 'True', False)
    keyword_104597 = True_104596
    kwargs_104598 = {'return_inverse': keyword_104597}
    # Getting the type of 'np' (line 384)
    np_104593 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 384, 23), 'np', False)
    # Obtaining the member 'unique' of a type (line 384)
    unique_104594 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 384, 23), np_104593, 'unique')
    # Calling unique(args, kwargs) (line 384)
    unique_call_result_104599 = invoke(stypy.reporting.localization.Localization(__file__, 384, 23), unique_104594, *[ar1_104595], **kwargs_104598)
    
    # Assigning a type to the variable 'call_assignment_104164' (line 384)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 384, 8), 'call_assignment_104164', unique_call_result_104599)
    
    # Assigning a Call to a Name (line 384):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_104602 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 384, 8), 'int')
    # Processing the call keyword arguments
    kwargs_104603 = {}
    # Getting the type of 'call_assignment_104164' (line 384)
    call_assignment_104164_104600 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 384, 8), 'call_assignment_104164', False)
    # Obtaining the member '__getitem__' of a type (line 384)
    getitem___104601 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 384, 8), call_assignment_104164_104600, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_104604 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___104601, *[int_104602], **kwargs_104603)
    
    # Assigning a type to the variable 'call_assignment_104165' (line 384)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 384, 8), 'call_assignment_104165', getitem___call_result_104604)
    
    # Assigning a Name to a Name (line 384):
    # Getting the type of 'call_assignment_104165' (line 384)
    call_assignment_104165_104605 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 384, 8), 'call_assignment_104165')
    # Assigning a type to the variable 'ar1' (line 384)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 384, 8), 'ar1', call_assignment_104165_104605)
    
    # Assigning a Call to a Name (line 384):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_104608 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 384, 8), 'int')
    # Processing the call keyword arguments
    kwargs_104609 = {}
    # Getting the type of 'call_assignment_104164' (line 384)
    call_assignment_104164_104606 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 384, 8), 'call_assignment_104164', False)
    # Obtaining the member '__getitem__' of a type (line 384)
    getitem___104607 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 384, 8), call_assignment_104164_104606, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_104610 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___104607, *[int_104608], **kwargs_104609)
    
    # Assigning a type to the variable 'call_assignment_104166' (line 384)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 384, 8), 'call_assignment_104166', getitem___call_result_104610)
    
    # Assigning a Name to a Name (line 384):
    # Getting the type of 'call_assignment_104166' (line 384)
    call_assignment_104166_104611 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 384, 8), 'call_assignment_104166')
    # Assigning a type to the variable 'rev_idx' (line 384)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 384, 13), 'rev_idx', call_assignment_104166_104611)
    
    # Assigning a Call to a Name (line 385):
    
    # Assigning a Call to a Name (line 385):
    
    # Call to unique(...): (line 385)
    # Processing the call arguments (line 385)
    # Getting the type of 'ar2' (line 385)
    ar2_104614 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 385, 24), 'ar2', False)
    # Processing the call keyword arguments (line 385)
    kwargs_104615 = {}
    # Getting the type of 'np' (line 385)
    np_104612 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 385, 14), 'np', False)
    # Obtaining the member 'unique' of a type (line 385)
    unique_104613 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 385, 14), np_104612, 'unique')
    # Calling unique(args, kwargs) (line 385)
    unique_call_result_104616 = invoke(stypy.reporting.localization.Localization(__file__, 385, 14), unique_104613, *[ar2_104614], **kwargs_104615)
    
    # Assigning a type to the variable 'ar2' (line 385)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 385, 8), 'ar2', unique_call_result_104616)
    # SSA join for if statement (line 383)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 387):
    
    # Assigning a Call to a Name (line 387):
    
    # Call to concatenate(...): (line 387)
    # Processing the call arguments (line 387)
    
    # Obtaining an instance of the builtin type 'tuple' (line 387)
    tuple_104619 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 387, 25), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 387)
    # Adding element type (line 387)
    # Getting the type of 'ar1' (line 387)
    ar1_104620 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 387, 25), 'ar1', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 387, 25), tuple_104619, ar1_104620)
    # Adding element type (line 387)
    # Getting the type of 'ar2' (line 387)
    ar2_104621 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 387, 30), 'ar2', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 387, 25), tuple_104619, ar2_104621)
    
    # Processing the call keyword arguments (line 387)
    kwargs_104622 = {}
    # Getting the type of 'np' (line 387)
    np_104617 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 387, 9), 'np', False)
    # Obtaining the member 'concatenate' of a type (line 387)
    concatenate_104618 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 387, 9), np_104617, 'concatenate')
    # Calling concatenate(args, kwargs) (line 387)
    concatenate_call_result_104623 = invoke(stypy.reporting.localization.Localization(__file__, 387, 9), concatenate_104618, *[tuple_104619], **kwargs_104622)
    
    # Assigning a type to the variable 'ar' (line 387)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 387, 4), 'ar', concatenate_call_result_104623)
    
    # Assigning a Call to a Name (line 391):
    
    # Assigning a Call to a Name (line 391):
    
    # Call to argsort(...): (line 391)
    # Processing the call keyword arguments (line 391)
    str_104626 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 391, 28), 'str', 'mergesort')
    keyword_104627 = str_104626
    kwargs_104628 = {'kind': keyword_104627}
    # Getting the type of 'ar' (line 391)
    ar_104624 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 391, 12), 'ar', False)
    # Obtaining the member 'argsort' of a type (line 391)
    argsort_104625 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 391, 12), ar_104624, 'argsort')
    # Calling argsort(args, kwargs) (line 391)
    argsort_call_result_104629 = invoke(stypy.reporting.localization.Localization(__file__, 391, 12), argsort_104625, *[], **kwargs_104628)
    
    # Assigning a type to the variable 'order' (line 391)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 391, 4), 'order', argsort_call_result_104629)
    
    # Assigning a Subscript to a Name (line 392):
    
    # Assigning a Subscript to a Name (line 392):
    
    # Obtaining the type of the subscript
    # Getting the type of 'order' (line 392)
    order_104630 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 392, 13), 'order')
    # Getting the type of 'ar' (line 392)
    ar_104631 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 392, 10), 'ar')
    # Obtaining the member '__getitem__' of a type (line 392)
    getitem___104632 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 392, 10), ar_104631, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 392)
    subscript_call_result_104633 = invoke(stypy.reporting.localization.Localization(__file__, 392, 10), getitem___104632, order_104630)
    
    # Assigning a type to the variable 'sar' (line 392)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 392, 4), 'sar', subscript_call_result_104633)
    
    # Getting the type of 'invert' (line 393)
    invert_104634 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 393, 7), 'invert')
    # Testing the type of an if condition (line 393)
    if_condition_104635 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 393, 4), invert_104634)
    # Assigning a type to the variable 'if_condition_104635' (line 393)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 393, 4), 'if_condition_104635', if_condition_104635)
    # SSA begins for if statement (line 393)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Compare to a Name (line 394):
    
    # Assigning a Compare to a Name (line 394):
    
    
    # Obtaining the type of the subscript
    int_104636 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 394, 23), 'int')
    slice_104637 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 394, 19), int_104636, None, None)
    # Getting the type of 'sar' (line 394)
    sar_104638 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 394, 19), 'sar')
    # Obtaining the member '__getitem__' of a type (line 394)
    getitem___104639 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 394, 19), sar_104638, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 394)
    subscript_call_result_104640 = invoke(stypy.reporting.localization.Localization(__file__, 394, 19), getitem___104639, slice_104637)
    
    
    # Obtaining the type of the subscript
    int_104641 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 394, 35), 'int')
    slice_104642 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 394, 30), None, int_104641, None)
    # Getting the type of 'sar' (line 394)
    sar_104643 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 394, 30), 'sar')
    # Obtaining the member '__getitem__' of a type (line 394)
    getitem___104644 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 394, 30), sar_104643, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 394)
    subscript_call_result_104645 = invoke(stypy.reporting.localization.Localization(__file__, 394, 30), getitem___104644, slice_104642)
    
    # Applying the binary operator '!=' (line 394)
    result_ne_104646 = python_operator(stypy.reporting.localization.Localization(__file__, 394, 19), '!=', subscript_call_result_104640, subscript_call_result_104645)
    
    # Assigning a type to the variable 'bool_ar' (line 394)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 394, 8), 'bool_ar', result_ne_104646)
    # SSA branch for the else part of an if statement (line 393)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Compare to a Name (line 396):
    
    # Assigning a Compare to a Name (line 396):
    
    
    # Obtaining the type of the subscript
    int_104647 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 396, 23), 'int')
    slice_104648 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 396, 19), int_104647, None, None)
    # Getting the type of 'sar' (line 396)
    sar_104649 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 396, 19), 'sar')
    # Obtaining the member '__getitem__' of a type (line 396)
    getitem___104650 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 396, 19), sar_104649, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 396)
    subscript_call_result_104651 = invoke(stypy.reporting.localization.Localization(__file__, 396, 19), getitem___104650, slice_104648)
    
    
    # Obtaining the type of the subscript
    int_104652 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 396, 35), 'int')
    slice_104653 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 396, 30), None, int_104652, None)
    # Getting the type of 'sar' (line 396)
    sar_104654 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 396, 30), 'sar')
    # Obtaining the member '__getitem__' of a type (line 396)
    getitem___104655 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 396, 30), sar_104654, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 396)
    subscript_call_result_104656 = invoke(stypy.reporting.localization.Localization(__file__, 396, 30), getitem___104655, slice_104653)
    
    # Applying the binary operator '==' (line 396)
    result_eq_104657 = python_operator(stypy.reporting.localization.Localization(__file__, 396, 19), '==', subscript_call_result_104651, subscript_call_result_104656)
    
    # Assigning a type to the variable 'bool_ar' (line 396)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 396, 8), 'bool_ar', result_eq_104657)
    # SSA join for if statement (line 393)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 397):
    
    # Assigning a Call to a Name (line 397):
    
    # Call to concatenate(...): (line 397)
    # Processing the call arguments (line 397)
    
    # Obtaining an instance of the builtin type 'tuple' (line 397)
    tuple_104660 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 397, 27), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 397)
    # Adding element type (line 397)
    # Getting the type of 'bool_ar' (line 397)
    bool_ar_104661 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 397, 27), 'bool_ar', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 397, 27), tuple_104660, bool_ar_104661)
    # Adding element type (line 397)
    
    # Obtaining an instance of the builtin type 'list' (line 397)
    list_104662 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 397, 36), 'list')
    # Adding type elements to the builtin type 'list' instance (line 397)
    # Adding element type (line 397)
    # Getting the type of 'invert' (line 397)
    invert_104663 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 397, 37), 'invert', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 397, 36), list_104662, invert_104663)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 397, 27), tuple_104660, list_104662)
    
    # Processing the call keyword arguments (line 397)
    kwargs_104664 = {}
    # Getting the type of 'np' (line 397)
    np_104658 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 397, 11), 'np', False)
    # Obtaining the member 'concatenate' of a type (line 397)
    concatenate_104659 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 397, 11), np_104658, 'concatenate')
    # Calling concatenate(args, kwargs) (line 397)
    concatenate_call_result_104665 = invoke(stypy.reporting.localization.Localization(__file__, 397, 11), concatenate_104659, *[tuple_104660], **kwargs_104664)
    
    # Assigning a type to the variable 'flag' (line 397)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 397, 4), 'flag', concatenate_call_result_104665)
    
    # Assigning a Call to a Name (line 398):
    
    # Assigning a Call to a Name (line 398):
    
    # Call to empty(...): (line 398)
    # Processing the call arguments (line 398)
    # Getting the type of 'ar' (line 398)
    ar_104668 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 398, 19), 'ar', False)
    # Obtaining the member 'shape' of a type (line 398)
    shape_104669 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 398, 19), ar_104668, 'shape')
    # Processing the call keyword arguments (line 398)
    # Getting the type of 'bool' (line 398)
    bool_104670 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 398, 35), 'bool', False)
    keyword_104671 = bool_104670
    kwargs_104672 = {'dtype': keyword_104671}
    # Getting the type of 'np' (line 398)
    np_104666 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 398, 10), 'np', False)
    # Obtaining the member 'empty' of a type (line 398)
    empty_104667 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 398, 10), np_104666, 'empty')
    # Calling empty(args, kwargs) (line 398)
    empty_call_result_104673 = invoke(stypy.reporting.localization.Localization(__file__, 398, 10), empty_104667, *[shape_104669], **kwargs_104672)
    
    # Assigning a type to the variable 'ret' (line 398)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 398, 4), 'ret', empty_call_result_104673)
    
    # Assigning a Name to a Subscript (line 399):
    
    # Assigning a Name to a Subscript (line 399):
    # Getting the type of 'flag' (line 399)
    flag_104674 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 399, 17), 'flag')
    # Getting the type of 'ret' (line 399)
    ret_104675 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 399, 4), 'ret')
    # Getting the type of 'order' (line 399)
    order_104676 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 399, 8), 'order')
    # Storing an element on a container (line 399)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 399, 4), ret_104675, (order_104676, flag_104674))
    
    # Getting the type of 'assume_unique' (line 401)
    assume_unique_104677 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 401, 7), 'assume_unique')
    # Testing the type of an if condition (line 401)
    if_condition_104678 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 401, 4), assume_unique_104677)
    # Assigning a type to the variable 'if_condition_104678' (line 401)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 401, 4), 'if_condition_104678', if_condition_104678)
    # SSA begins for if statement (line 401)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Obtaining the type of the subscript
    
    # Call to len(...): (line 402)
    # Processing the call arguments (line 402)
    # Getting the type of 'ar1' (line 402)
    ar1_104680 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 402, 24), 'ar1', False)
    # Processing the call keyword arguments (line 402)
    kwargs_104681 = {}
    # Getting the type of 'len' (line 402)
    len_104679 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 402, 20), 'len', False)
    # Calling len(args, kwargs) (line 402)
    len_call_result_104682 = invoke(stypy.reporting.localization.Localization(__file__, 402, 20), len_104679, *[ar1_104680], **kwargs_104681)
    
    slice_104683 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 402, 15), None, len_call_result_104682, None)
    # Getting the type of 'ret' (line 402)
    ret_104684 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 402, 15), 'ret')
    # Obtaining the member '__getitem__' of a type (line 402)
    getitem___104685 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 402, 15), ret_104684, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 402)
    subscript_call_result_104686 = invoke(stypy.reporting.localization.Localization(__file__, 402, 15), getitem___104685, slice_104683)
    
    # Assigning a type to the variable 'stypy_return_type' (line 402)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 402, 8), 'stypy_return_type', subscript_call_result_104686)
    # SSA branch for the else part of an if statement (line 401)
    module_type_store.open_ssa_branch('else')
    
    # Obtaining the type of the subscript
    # Getting the type of 'rev_idx' (line 404)
    rev_idx_104687 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 404, 19), 'rev_idx')
    # Getting the type of 'ret' (line 404)
    ret_104688 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 404, 15), 'ret')
    # Obtaining the member '__getitem__' of a type (line 404)
    getitem___104689 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 404, 15), ret_104688, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 404)
    subscript_call_result_104690 = invoke(stypy.reporting.localization.Localization(__file__, 404, 15), getitem___104689, rev_idx_104687)
    
    # Assigning a type to the variable 'stypy_return_type' (line 404)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 404, 8), 'stypy_return_type', subscript_call_result_104690)
    # SSA join for if statement (line 401)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'in1d(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'in1d' in the type store
    # Getting the type of 'stypy_return_type' (line 305)
    stypy_return_type_104691 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 305, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_104691)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'in1d'
    return stypy_return_type_104691

# Assigning a type to the variable 'in1d' (line 305)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 305, 0), 'in1d', in1d)

@norecursion
def union1d(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'union1d'
    module_type_store = module_type_store.open_function_context('union1d', 406, 0, False)
    
    # Passed parameters checking function
    union1d.stypy_localization = localization
    union1d.stypy_type_of_self = None
    union1d.stypy_type_store = module_type_store
    union1d.stypy_function_name = 'union1d'
    union1d.stypy_param_names_list = ['ar1', 'ar2']
    union1d.stypy_varargs_param_name = None
    union1d.stypy_kwargs_param_name = None
    union1d.stypy_call_defaults = defaults
    union1d.stypy_call_varargs = varargs
    union1d.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'union1d', ['ar1', 'ar2'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'union1d', localization, ['ar1', 'ar2'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'union1d(...)' code ##################

    str_104692 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 438, (-1)), 'str', '\n    Find the union of two arrays.\n\n    Return the unique, sorted array of values that are in either of the two\n    input arrays.\n\n    Parameters\n    ----------\n    ar1, ar2 : array_like\n        Input arrays. They are flattened if they are not already 1D.\n\n    Returns\n    -------\n    union1d : ndarray\n        Unique, sorted union of the input arrays.\n\n    See Also\n    --------\n    numpy.lib.arraysetops : Module with a number of other functions for\n                            performing set operations on arrays.\n\n    Examples\n    --------\n    >>> np.union1d([-1, 0, 1], [-2, 0, 2])\n    array([-2, -1,  0,  1,  2])\n\n    To find the union of more than two arrays, use functools.reduce:\n\n    >>> from functools import reduce\n    >>> reduce(np.union1d, ([1, 3, 4, 3], [3, 1, 2, 1], [6, 3, 4, 2]))\n    array([1, 2, 3, 4, 6])\n    ')
    
    # Call to unique(...): (line 439)
    # Processing the call arguments (line 439)
    
    # Call to concatenate(...): (line 439)
    # Processing the call arguments (line 439)
    
    # Obtaining an instance of the builtin type 'tuple' (line 439)
    tuple_104696 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 439, 34), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 439)
    # Adding element type (line 439)
    # Getting the type of 'ar1' (line 439)
    ar1_104697 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 439, 34), 'ar1', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 439, 34), tuple_104696, ar1_104697)
    # Adding element type (line 439)
    # Getting the type of 'ar2' (line 439)
    ar2_104698 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 439, 39), 'ar2', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 439, 34), tuple_104696, ar2_104698)
    
    # Processing the call keyword arguments (line 439)
    kwargs_104699 = {}
    # Getting the type of 'np' (line 439)
    np_104694 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 439, 18), 'np', False)
    # Obtaining the member 'concatenate' of a type (line 439)
    concatenate_104695 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 439, 18), np_104694, 'concatenate')
    # Calling concatenate(args, kwargs) (line 439)
    concatenate_call_result_104700 = invoke(stypy.reporting.localization.Localization(__file__, 439, 18), concatenate_104695, *[tuple_104696], **kwargs_104699)
    
    # Processing the call keyword arguments (line 439)
    kwargs_104701 = {}
    # Getting the type of 'unique' (line 439)
    unique_104693 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 439, 11), 'unique', False)
    # Calling unique(args, kwargs) (line 439)
    unique_call_result_104702 = invoke(stypy.reporting.localization.Localization(__file__, 439, 11), unique_104693, *[concatenate_call_result_104700], **kwargs_104701)
    
    # Assigning a type to the variable 'stypy_return_type' (line 439)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 439, 4), 'stypy_return_type', unique_call_result_104702)
    
    # ################# End of 'union1d(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'union1d' in the type store
    # Getting the type of 'stypy_return_type' (line 406)
    stypy_return_type_104703 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 406, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_104703)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'union1d'
    return stypy_return_type_104703

# Assigning a type to the variable 'union1d' (line 406)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 406, 0), 'union1d', union1d)

@norecursion
def setdiff1d(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'False' (line 441)
    False_104704 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 441, 38), 'False')
    defaults = [False_104704]
    # Create a new context for function 'setdiff1d'
    module_type_store = module_type_store.open_function_context('setdiff1d', 441, 0, False)
    
    # Passed parameters checking function
    setdiff1d.stypy_localization = localization
    setdiff1d.stypy_type_of_self = None
    setdiff1d.stypy_type_store = module_type_store
    setdiff1d.stypy_function_name = 'setdiff1d'
    setdiff1d.stypy_param_names_list = ['ar1', 'ar2', 'assume_unique']
    setdiff1d.stypy_varargs_param_name = None
    setdiff1d.stypy_kwargs_param_name = None
    setdiff1d.stypy_call_defaults = defaults
    setdiff1d.stypy_call_varargs = varargs
    setdiff1d.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'setdiff1d', ['ar1', 'ar2', 'assume_unique'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'setdiff1d', localization, ['ar1', 'ar2', 'assume_unique'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'setdiff1d(...)' code ##################

    str_104705 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 474, (-1)), 'str', '\n    Find the set difference of two arrays.\n\n    Return the sorted, unique values in `ar1` that are not in `ar2`.\n\n    Parameters\n    ----------\n    ar1 : array_like\n        Input array.\n    ar2 : array_like\n        Input comparison array.\n    assume_unique : bool\n        If True, the input arrays are both assumed to be unique, which\n        can speed up the calculation.  Default is False.\n\n    Returns\n    -------\n    setdiff1d : ndarray\n        Sorted 1D array of values in `ar1` that are not in `ar2`.\n\n    See Also\n    --------\n    numpy.lib.arraysetops : Module with a number of other functions for\n                            performing set operations on arrays.\n\n    Examples\n    --------\n    >>> a = np.array([1, 2, 3, 2, 4, 1])\n    >>> b = np.array([3, 4, 5, 6])\n    >>> np.setdiff1d(a, b)\n    array([1, 2])\n\n    ')
    
    # Getting the type of 'assume_unique' (line 475)
    assume_unique_104706 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 475, 7), 'assume_unique')
    # Testing the type of an if condition (line 475)
    if_condition_104707 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 475, 4), assume_unique_104706)
    # Assigning a type to the variable 'if_condition_104707' (line 475)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 475, 4), 'if_condition_104707', if_condition_104707)
    # SSA begins for if statement (line 475)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 476):
    
    # Assigning a Call to a Name (line 476):
    
    # Call to ravel(...): (line 476)
    # Processing the call keyword arguments (line 476)
    kwargs_104714 = {}
    
    # Call to asarray(...): (line 476)
    # Processing the call arguments (line 476)
    # Getting the type of 'ar1' (line 476)
    ar1_104710 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 476, 25), 'ar1', False)
    # Processing the call keyword arguments (line 476)
    kwargs_104711 = {}
    # Getting the type of 'np' (line 476)
    np_104708 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 476, 14), 'np', False)
    # Obtaining the member 'asarray' of a type (line 476)
    asarray_104709 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 476, 14), np_104708, 'asarray')
    # Calling asarray(args, kwargs) (line 476)
    asarray_call_result_104712 = invoke(stypy.reporting.localization.Localization(__file__, 476, 14), asarray_104709, *[ar1_104710], **kwargs_104711)
    
    # Obtaining the member 'ravel' of a type (line 476)
    ravel_104713 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 476, 14), asarray_call_result_104712, 'ravel')
    # Calling ravel(args, kwargs) (line 476)
    ravel_call_result_104715 = invoke(stypy.reporting.localization.Localization(__file__, 476, 14), ravel_104713, *[], **kwargs_104714)
    
    # Assigning a type to the variable 'ar1' (line 476)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 476, 8), 'ar1', ravel_call_result_104715)
    # SSA branch for the else part of an if statement (line 475)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Call to a Name (line 478):
    
    # Assigning a Call to a Name (line 478):
    
    # Call to unique(...): (line 478)
    # Processing the call arguments (line 478)
    # Getting the type of 'ar1' (line 478)
    ar1_104717 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 478, 21), 'ar1', False)
    # Processing the call keyword arguments (line 478)
    kwargs_104718 = {}
    # Getting the type of 'unique' (line 478)
    unique_104716 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 478, 14), 'unique', False)
    # Calling unique(args, kwargs) (line 478)
    unique_call_result_104719 = invoke(stypy.reporting.localization.Localization(__file__, 478, 14), unique_104716, *[ar1_104717], **kwargs_104718)
    
    # Assigning a type to the variable 'ar1' (line 478)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 478, 8), 'ar1', unique_call_result_104719)
    
    # Assigning a Call to a Name (line 479):
    
    # Assigning a Call to a Name (line 479):
    
    # Call to unique(...): (line 479)
    # Processing the call arguments (line 479)
    # Getting the type of 'ar2' (line 479)
    ar2_104721 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 479, 21), 'ar2', False)
    # Processing the call keyword arguments (line 479)
    kwargs_104722 = {}
    # Getting the type of 'unique' (line 479)
    unique_104720 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 479, 14), 'unique', False)
    # Calling unique(args, kwargs) (line 479)
    unique_call_result_104723 = invoke(stypy.reporting.localization.Localization(__file__, 479, 14), unique_104720, *[ar2_104721], **kwargs_104722)
    
    # Assigning a type to the variable 'ar2' (line 479)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 479, 8), 'ar2', unique_call_result_104723)
    # SSA join for if statement (line 475)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Obtaining the type of the subscript
    
    # Call to in1d(...): (line 480)
    # Processing the call arguments (line 480)
    # Getting the type of 'ar1' (line 480)
    ar1_104725 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 480, 20), 'ar1', False)
    # Getting the type of 'ar2' (line 480)
    ar2_104726 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 480, 25), 'ar2', False)
    # Processing the call keyword arguments (line 480)
    # Getting the type of 'True' (line 480)
    True_104727 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 480, 44), 'True', False)
    keyword_104728 = True_104727
    # Getting the type of 'True' (line 480)
    True_104729 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 480, 57), 'True', False)
    keyword_104730 = True_104729
    kwargs_104731 = {'assume_unique': keyword_104728, 'invert': keyword_104730}
    # Getting the type of 'in1d' (line 480)
    in1d_104724 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 480, 15), 'in1d', False)
    # Calling in1d(args, kwargs) (line 480)
    in1d_call_result_104732 = invoke(stypy.reporting.localization.Localization(__file__, 480, 15), in1d_104724, *[ar1_104725, ar2_104726], **kwargs_104731)
    
    # Getting the type of 'ar1' (line 480)
    ar1_104733 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 480, 11), 'ar1')
    # Obtaining the member '__getitem__' of a type (line 480)
    getitem___104734 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 480, 11), ar1_104733, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 480)
    subscript_call_result_104735 = invoke(stypy.reporting.localization.Localization(__file__, 480, 11), getitem___104734, in1d_call_result_104732)
    
    # Assigning a type to the variable 'stypy_return_type' (line 480)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 480, 4), 'stypy_return_type', subscript_call_result_104735)
    
    # ################# End of 'setdiff1d(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'setdiff1d' in the type store
    # Getting the type of 'stypy_return_type' (line 441)
    stypy_return_type_104736 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 441, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_104736)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'setdiff1d'
    return stypy_return_type_104736

# Assigning a type to the variable 'setdiff1d' (line 441)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 441, 0), 'setdiff1d', setdiff1d)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
