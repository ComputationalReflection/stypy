
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: from __future__ import division, print_function, absolute_import
2: 
3: import functools
4: import operator
5: import sys
6: import warnings
7: import numbers
8: from collections import namedtuple
9: import inspect
10: 
11: import numpy as np
12: 
13: 
14: def _valarray(shape, value=np.nan, typecode=None):
15:     '''Return an array of all value.
16:     '''
17: 
18:     out = np.ones(shape, dtype=bool) * value
19:     if typecode is not None:
20:         out = out.astype(typecode)
21:     if not isinstance(out, np.ndarray):
22:         out = np.asarray(out)
23:     return out
24: 
25: 
26: def _lazywhere(cond, arrays, f, fillvalue=None, f2=None):
27:     '''
28:     np.where(cond, x, fillvalue) always evaluates x even where cond is False.
29:     This one only evaluates f(arr1[cond], arr2[cond], ...).
30:     For example,
31:     >>> a, b = np.array([1, 2, 3, 4]), np.array([5, 6, 7, 8])
32:     >>> def f(a, b):
33:         return a*b
34:     >>> _lazywhere(a > 2, (a, b), f, np.nan)
35:     array([ nan,  nan,  21.,  32.])
36: 
37:     Notice it assumes that all `arrays` are of the same shape, or can be
38:     broadcasted together.
39: 
40:     '''
41:     if fillvalue is None:
42:         if f2 is None:
43:             raise ValueError("One of (fillvalue, f2) must be given.")
44:         else:
45:             fillvalue = np.nan
46:     else:
47:         if f2 is not None:
48:             raise ValueError("Only one of (fillvalue, f2) can be given.")
49: 
50:     arrays = np.broadcast_arrays(*arrays)
51:     temp = tuple(np.extract(cond, arr) for arr in arrays)
52:     tcode = np.mintypecode([a.dtype.char for a in arrays])
53:     out = _valarray(np.shape(arrays[0]), value=fillvalue, typecode=tcode)
54:     np.place(out, cond, f(*temp))
55:     if f2 is not None:
56:         temp = tuple(np.extract(~cond, arr) for arr in arrays)
57:         np.place(out, ~cond, f2(*temp))
58: 
59:     return out
60: 
61: 
62: def _lazyselect(condlist, choicelist, arrays, default=0):
63:     '''
64:     Mimic `np.select(condlist, choicelist)`.
65: 
66:     Notice it assumes that all `arrays` are of the same shape, or can be
67:     broadcasted together.
68: 
69:     All functions in `choicelist` must accept array arguments in the order
70:     given in `arrays` and must return an array of the same shape as broadcasted
71:     `arrays`.
72: 
73:     Examples
74:     --------
75:     >>> x = np.arange(6)
76:     >>> np.select([x <3, x > 3], [x**2, x**3], default=0)
77:     array([  0,   1,   4,   0,  64, 125])
78: 
79:     >>> _lazyselect([x < 3, x > 3], [lambda x: x**2, lambda x: x**3], (x,))
80:     array([   0.,    1.,    4.,   0.,   64.,  125.])
81: 
82:     >>> a = -np.ones_like(x)
83:     >>> _lazyselect([x < 3, x > 3],
84:     ...             [lambda x, a: x**2, lambda x, a: a * x**3],
85:     ...             (x, a), default=np.nan)
86:     array([   0.,    1.,    4.,   nan,  -64., -125.])
87: 
88:     '''
89:     arrays = np.broadcast_arrays(*arrays)
90:     tcode = np.mintypecode([a.dtype.char for a in arrays])
91:     out = _valarray(np.shape(arrays[0]), value=default, typecode=tcode)
92:     for index in range(len(condlist)):
93:         func, cond = choicelist[index], condlist[index]
94:         if np.all(cond is False):
95:             continue
96:         cond, _ = np.broadcast_arrays(cond, arrays[0])
97:         temp = tuple(np.extract(cond, arr) for arr in arrays)
98:         np.place(out, cond, func(*temp))
99:     return out
100: 
101: 
102: def _aligned_zeros(shape, dtype=float, order="C", align=None):
103:     '''Allocate a new ndarray with aligned memory.
104: 
105:     Primary use case for this currently is working around a f2py issue
106:     in Numpy 1.9.1, where dtype.alignment is such that np.zeros() does
107:     not necessarily create arrays aligned up to it.
108: 
109:     '''
110:     dtype = np.dtype(dtype)
111:     if align is None:
112:         align = dtype.alignment
113:     if not hasattr(shape, '__len__'):
114:         shape = (shape,)
115:     size = functools.reduce(operator.mul, shape) * dtype.itemsize
116:     buf = np.empty(size + align + 1, np.uint8)
117:     offset = buf.__array_interface__['data'][0] % align
118:     if offset != 0:
119:         offset = align - offset
120:     # Note: slices producing 0-size arrays do not necessarily change
121:     # data pointer --- so we use and allocate size+1
122:     buf = buf[offset:offset+size+1][:-1]
123:     data = np.ndarray(shape, dtype, buf, order=order)
124:     data.fill(0)
125:     return data
126: 
127: 
128: def _prune_array(array):
129:     '''Return an array equivalent to the input array. If the input
130:     array is a view of a much larger array, copy its contents to a
131:     newly allocated array. Otherwise, return the input unchaged.
132:     '''
133:     if array.base is not None and array.size < array.base.size // 2:
134:         return array.copy()
135:     return array
136: 
137: 
138: class DeprecatedImport(object):
139:     '''
140:     Deprecated import, with redirection + warning.
141: 
142:     Examples
143:     --------
144:     Suppose you previously had in some module::
145: 
146:         from foo import spam
147: 
148:     If this has to be deprecated, do::
149: 
150:         spam = DeprecatedImport("foo.spam", "baz")
151: 
152:     to redirect users to use "baz" module instead.
153: 
154:     '''
155: 
156:     def __init__(self, old_module_name, new_module_name):
157:         self._old_name = old_module_name
158:         self._new_name = new_module_name
159:         __import__(self._new_name)
160:         self._mod = sys.modules[self._new_name]
161: 
162:     def __dir__(self):
163:         return dir(self._mod)
164: 
165:     def __getattr__(self, name):
166:         warnings.warn("Module %s is deprecated, use %s instead"
167:                       % (self._old_name, self._new_name),
168:                       DeprecationWarning)
169:         return getattr(self._mod, name)
170: 
171: 
172: # copy-pasted from scikit-learn utils/validation.py
173: def check_random_state(seed):
174:     '''Turn seed into a np.random.RandomState instance
175: 
176:     If seed is None (or np.random), return the RandomState singleton used
177:     by np.random.
178:     If seed is an int, return a new RandomState instance seeded with seed.
179:     If seed is already a RandomState instance, return it.
180:     Otherwise raise ValueError.
181:     '''
182:     if seed is None or seed is np.random:
183:         return np.random.mtrand._rand
184:     if isinstance(seed, (numbers.Integral, np.integer)):
185:         return np.random.RandomState(seed)
186:     if isinstance(seed, np.random.RandomState):
187:         return seed
188:     raise ValueError('%r cannot be used to seed a numpy.random.RandomState'
189:                      ' instance' % seed)
190: 
191: 
192: def _asarray_validated(a, check_finite=True,
193:                        sparse_ok=False, objects_ok=False, mask_ok=False,
194:                        as_inexact=False):
195:     '''
196:     Helper function for scipy argument validation.
197: 
198:     Many scipy linear algebra functions do support arbitrary array-like
199:     input arguments.  Examples of commonly unsupported inputs include
200:     matrices containing inf/nan, sparse matrix representations, and
201:     matrices with complicated elements.
202: 
203:     Parameters
204:     ----------
205:     a : array_like
206:         The array-like input.
207:     check_finite : bool, optional
208:         Whether to check that the input matrices contain only finite numbers.
209:         Disabling may give a performance gain, but may result in problems
210:         (crashes, non-termination) if the inputs do contain infinities or NaNs.
211:         Default: True
212:     sparse_ok : bool, optional
213:         True if scipy sparse matrices are allowed.
214:     objects_ok : bool, optional
215:         True if arrays with dype('O') are allowed.
216:     mask_ok : bool, optional
217:         True if masked arrays are allowed.
218:     as_inexact : bool, optional
219:         True to convert the input array to a np.inexact dtype.
220: 
221:     Returns
222:     -------
223:     ret : ndarray
224:         The converted validated array.
225: 
226:     '''
227:     if not sparse_ok:
228:         import scipy.sparse
229:         if scipy.sparse.issparse(a):
230:             msg = ('Sparse matrices are not supported by this function. '
231:                    'Perhaps one of the scipy.sparse.linalg functions '
232:                    'would work instead.')
233:             raise ValueError(msg)
234:     if not mask_ok:
235:         if np.ma.isMaskedArray(a):
236:             raise ValueError('masked arrays are not supported')
237:     toarray = np.asarray_chkfinite if check_finite else np.asarray
238:     a = toarray(a)
239:     if not objects_ok:
240:         if a.dtype is np.dtype('O'):
241:             raise ValueError('object arrays are not supported')
242:     if as_inexact:
243:         if not np.issubdtype(a.dtype, np.inexact):
244:             a = toarray(a, dtype=np.float_)
245:     return a
246: 
247: 
248: # Add a replacement for inspect.getargspec() which is deprecated in python 3.5
249: # The version below is borrowed from Django,
250: # https://github.com/django/django/pull/4846
251: 
252: # Note an inconsistency between inspect.getargspec(func) and
253: # inspect.signature(func). If `func` is a bound method, the latter does *not*
254: # list `self` as a first argument, while the former *does*.
255: # Hence cook up a common ground replacement: `getargspec_no_self` which
256: # mimics `inspect.getargspec` but does not list `self`.
257: #
258: # This way, the caller code does not need to know whether it uses a legacy
259: # .getargspec or bright and shiny .signature.
260: 
261: try:
262:     # is it python 3.3 or higher?
263:     inspect.signature
264: 
265:     # Apparently, yes. Wrap inspect.signature
266: 
267:     ArgSpec = namedtuple('ArgSpec', ['args', 'varargs', 'keywords', 'defaults'])
268: 
269:     def getargspec_no_self(func):
270:         '''inspect.getargspec replacement using inspect.signature.
271: 
272:         inspect.getargspec is deprecated in python 3. This is a replacement
273:         based on the (new in python 3.3) `inspect.signature`.
274: 
275:         Parameters
276:         ----------
277:         func : callable
278:             A callable to inspect
279: 
280:         Returns
281:         -------
282:         argspec : ArgSpec(args, varargs, varkw, defaults)
283:             This is similar to the result of inspect.getargspec(func) under
284:             python 2.x.
285:             NOTE: if the first argument of `func` is self, it is *not*, I repeat
286:             *not* included in argspec.args.
287:             This is done for consistency between inspect.getargspec() under
288:             python 2.x, and inspect.signature() under python 3.x.
289:         '''
290:         sig = inspect.signature(func)
291:         args = [
292:             p.name for p in sig.parameters.values()
293:             if p.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD
294:         ]
295:         varargs = [
296:             p.name for p in sig.parameters.values()
297:             if p.kind == inspect.Parameter.VAR_POSITIONAL
298:         ]
299:         varargs = varargs[0] if varargs else None
300:         varkw = [
301:             p.name for p in sig.parameters.values()
302:             if p.kind == inspect.Parameter.VAR_KEYWORD
303:         ]
304:         varkw = varkw[0] if varkw else None
305:         defaults = [
306:             p.default for p in sig.parameters.values()
307:             if (p.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD and
308:                p.default is not p.empty)
309:         ] or None
310:         return ArgSpec(args, varargs, varkw, defaults)
311: 
312: except AttributeError:
313:     # python 2.x
314:     def getargspec_no_self(func):
315:         '''inspect.getargspec replacement for compatibility with python 3.x.
316: 
317:         inspect.getargspec is deprecated in python 3. This wraps it, and
318:         *removes* `self` from the argument list of `func`, if present.
319:         This is done for forward compatibility with python 3.
320: 
321:         Parameters
322:         ----------
323:         func : callable
324:             A callable to inspect
325: 
326:         Returns
327:         -------
328:         argspec : ArgSpec(args, varargs, varkw, defaults)
329:             This is similar to the result of inspect.getargspec(func) under
330:             python 2.x.
331:             NOTE: if the first argument of `func` is self, it is *not*, I repeat
332:             *not* included in argspec.args.
333:             This is done for consistency between inspect.getargspec() under
334:             python 2.x, and inspect.signature() under python 3.x.
335:         '''
336:         argspec = inspect.getargspec(func)
337:         if argspec.args[0] == 'self':
338:             argspec.args.pop(0)
339:         return argspec
340: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 3, 0))

# 'import functools' statement (line 3)
import functools

import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'functools', functools, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 4, 0))

# 'import operator' statement (line 4)
import operator

import_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'operator', operator, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 5, 0))

# 'import sys' statement (line 5)
import sys

import_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'sys', sys, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 6, 0))

# 'import warnings' statement (line 6)
import warnings

import_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'warnings', warnings, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 7, 0))

# 'import numbers' statement (line 7)
import numbers

import_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'numbers', numbers, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 8, 0))

# 'from collections import namedtuple' statement (line 8)
try:
    from collections import namedtuple

except:
    namedtuple = UndefinedType
import_from_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'collections', None, module_type_store, ['namedtuple'], [namedtuple])

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 9, 0))

# 'import inspect' statement (line 9)
import inspect

import_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'inspect', inspect, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 11, 0))

# 'import numpy' statement (line 11)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/_lib/')
import_710079 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'numpy')

if (type(import_710079) is not StypyTypeError):

    if (import_710079 != 'pyd_module'):
        __import__(import_710079)
        sys_modules_710080 = sys.modules[import_710079]
        import_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'np', sys_modules_710080.module_type_store, module_type_store)
    else:
        import numpy as np

        import_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'np', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 11)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 0), 'numpy', import_710079)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/_lib/')


@norecursion
def _valarray(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'np' (line 14)
    np_710081 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 27), 'np')
    # Obtaining the member 'nan' of a type (line 14)
    nan_710082 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 14, 27), np_710081, 'nan')
    # Getting the type of 'None' (line 14)
    None_710083 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 44), 'None')
    defaults = [nan_710082, None_710083]
    # Create a new context for function '_valarray'
    module_type_store = module_type_store.open_function_context('_valarray', 14, 0, False)
    
    # Passed parameters checking function
    _valarray.stypy_localization = localization
    _valarray.stypy_type_of_self = None
    _valarray.stypy_type_store = module_type_store
    _valarray.stypy_function_name = '_valarray'
    _valarray.stypy_param_names_list = ['shape', 'value', 'typecode']
    _valarray.stypy_varargs_param_name = None
    _valarray.stypy_kwargs_param_name = None
    _valarray.stypy_call_defaults = defaults
    _valarray.stypy_call_varargs = varargs
    _valarray.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_valarray', ['shape', 'value', 'typecode'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_valarray', localization, ['shape', 'value', 'typecode'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_valarray(...)' code ##################

    str_710084 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, (-1)), 'str', 'Return an array of all value.\n    ')
    
    # Assigning a BinOp to a Name (line 18):
    
    # Assigning a BinOp to a Name (line 18):
    
    # Call to ones(...): (line 18)
    # Processing the call arguments (line 18)
    # Getting the type of 'shape' (line 18)
    shape_710087 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 18), 'shape', False)
    # Processing the call keyword arguments (line 18)
    # Getting the type of 'bool' (line 18)
    bool_710088 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 31), 'bool', False)
    keyword_710089 = bool_710088
    kwargs_710090 = {'dtype': keyword_710089}
    # Getting the type of 'np' (line 18)
    np_710085 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 10), 'np', False)
    # Obtaining the member 'ones' of a type (line 18)
    ones_710086 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 18, 10), np_710085, 'ones')
    # Calling ones(args, kwargs) (line 18)
    ones_call_result_710091 = invoke(stypy.reporting.localization.Localization(__file__, 18, 10), ones_710086, *[shape_710087], **kwargs_710090)
    
    # Getting the type of 'value' (line 18)
    value_710092 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 39), 'value')
    # Applying the binary operator '*' (line 18)
    result_mul_710093 = python_operator(stypy.reporting.localization.Localization(__file__, 18, 10), '*', ones_call_result_710091, value_710092)
    
    # Assigning a type to the variable 'out' (line 18)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 18, 4), 'out', result_mul_710093)
    
    # Type idiom detected: calculating its left and rigth part (line 19)
    # Getting the type of 'typecode' (line 19)
    typecode_710094 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 4), 'typecode')
    # Getting the type of 'None' (line 19)
    None_710095 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 23), 'None')
    
    (may_be_710096, more_types_in_union_710097) = may_not_be_none(typecode_710094, None_710095)

    if may_be_710096:

        if more_types_in_union_710097:
            # Runtime conditional SSA (line 19)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a Call to a Name (line 20):
        
        # Assigning a Call to a Name (line 20):
        
        # Call to astype(...): (line 20)
        # Processing the call arguments (line 20)
        # Getting the type of 'typecode' (line 20)
        typecode_710100 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 25), 'typecode', False)
        # Processing the call keyword arguments (line 20)
        kwargs_710101 = {}
        # Getting the type of 'out' (line 20)
        out_710098 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 14), 'out', False)
        # Obtaining the member 'astype' of a type (line 20)
        astype_710099 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 20, 14), out_710098, 'astype')
        # Calling astype(args, kwargs) (line 20)
        astype_call_result_710102 = invoke(stypy.reporting.localization.Localization(__file__, 20, 14), astype_710099, *[typecode_710100], **kwargs_710101)
        
        # Assigning a type to the variable 'out' (line 20)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 20, 8), 'out', astype_call_result_710102)

        if more_types_in_union_710097:
            # SSA join for if statement (line 19)
            module_type_store = module_type_store.join_ssa_context()


    
    
    
    
    # Call to isinstance(...): (line 21)
    # Processing the call arguments (line 21)
    # Getting the type of 'out' (line 21)
    out_710104 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 22), 'out', False)
    # Getting the type of 'np' (line 21)
    np_710105 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 27), 'np', False)
    # Obtaining the member 'ndarray' of a type (line 21)
    ndarray_710106 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 21, 27), np_710105, 'ndarray')
    # Processing the call keyword arguments (line 21)
    kwargs_710107 = {}
    # Getting the type of 'isinstance' (line 21)
    isinstance_710103 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 11), 'isinstance', False)
    # Calling isinstance(args, kwargs) (line 21)
    isinstance_call_result_710108 = invoke(stypy.reporting.localization.Localization(__file__, 21, 11), isinstance_710103, *[out_710104, ndarray_710106], **kwargs_710107)
    
    # Applying the 'not' unary operator (line 21)
    result_not__710109 = python_operator(stypy.reporting.localization.Localization(__file__, 21, 7), 'not', isinstance_call_result_710108)
    
    # Testing the type of an if condition (line 21)
    if_condition_710110 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 21, 4), result_not__710109)
    # Assigning a type to the variable 'if_condition_710110' (line 21)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 21, 4), 'if_condition_710110', if_condition_710110)
    # SSA begins for if statement (line 21)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 22):
    
    # Assigning a Call to a Name (line 22):
    
    # Call to asarray(...): (line 22)
    # Processing the call arguments (line 22)
    # Getting the type of 'out' (line 22)
    out_710113 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 25), 'out', False)
    # Processing the call keyword arguments (line 22)
    kwargs_710114 = {}
    # Getting the type of 'np' (line 22)
    np_710111 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 14), 'np', False)
    # Obtaining the member 'asarray' of a type (line 22)
    asarray_710112 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 22, 14), np_710111, 'asarray')
    # Calling asarray(args, kwargs) (line 22)
    asarray_call_result_710115 = invoke(stypy.reporting.localization.Localization(__file__, 22, 14), asarray_710112, *[out_710113], **kwargs_710114)
    
    # Assigning a type to the variable 'out' (line 22)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 22, 8), 'out', asarray_call_result_710115)
    # SSA join for if statement (line 21)
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'out' (line 23)
    out_710116 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 11), 'out')
    # Assigning a type to the variable 'stypy_return_type' (line 23)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 23, 4), 'stypy_return_type', out_710116)
    
    # ################# End of '_valarray(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_valarray' in the type store
    # Getting the type of 'stypy_return_type' (line 14)
    stypy_return_type_710117 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_710117)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_valarray'
    return stypy_return_type_710117

# Assigning a type to the variable '_valarray' (line 14)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 0), '_valarray', _valarray)

@norecursion
def _lazywhere(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'None' (line 26)
    None_710118 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 42), 'None')
    # Getting the type of 'None' (line 26)
    None_710119 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 51), 'None')
    defaults = [None_710118, None_710119]
    # Create a new context for function '_lazywhere'
    module_type_store = module_type_store.open_function_context('_lazywhere', 26, 0, False)
    
    # Passed parameters checking function
    _lazywhere.stypy_localization = localization
    _lazywhere.stypy_type_of_self = None
    _lazywhere.stypy_type_store = module_type_store
    _lazywhere.stypy_function_name = '_lazywhere'
    _lazywhere.stypy_param_names_list = ['cond', 'arrays', 'f', 'fillvalue', 'f2']
    _lazywhere.stypy_varargs_param_name = None
    _lazywhere.stypy_kwargs_param_name = None
    _lazywhere.stypy_call_defaults = defaults
    _lazywhere.stypy_call_varargs = varargs
    _lazywhere.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_lazywhere', ['cond', 'arrays', 'f', 'fillvalue', 'f2'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_lazywhere', localization, ['cond', 'arrays', 'f', 'fillvalue', 'f2'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_lazywhere(...)' code ##################

    str_710120 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, (-1)), 'str', '\n    np.where(cond, x, fillvalue) always evaluates x even where cond is False.\n    This one only evaluates f(arr1[cond], arr2[cond], ...).\n    For example,\n    >>> a, b = np.array([1, 2, 3, 4]), np.array([5, 6, 7, 8])\n    >>> def f(a, b):\n        return a*b\n    >>> _lazywhere(a > 2, (a, b), f, np.nan)\n    array([ nan,  nan,  21.,  32.])\n\n    Notice it assumes that all `arrays` are of the same shape, or can be\n    broadcasted together.\n\n    ')
    
    # Type idiom detected: calculating its left and rigth part (line 41)
    # Getting the type of 'fillvalue' (line 41)
    fillvalue_710121 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 7), 'fillvalue')
    # Getting the type of 'None' (line 41)
    None_710122 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 20), 'None')
    
    (may_be_710123, more_types_in_union_710124) = may_be_none(fillvalue_710121, None_710122)

    if may_be_710123:

        if more_types_in_union_710124:
            # Runtime conditional SSA (line 41)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Type idiom detected: calculating its left and rigth part (line 42)
        # Getting the type of 'f2' (line 42)
        f2_710125 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 11), 'f2')
        # Getting the type of 'None' (line 42)
        None_710126 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 17), 'None')
        
        (may_be_710127, more_types_in_union_710128) = may_be_none(f2_710125, None_710126)

        if may_be_710127:

            if more_types_in_union_710128:
                # Runtime conditional SSA (line 42)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Call to ValueError(...): (line 43)
            # Processing the call arguments (line 43)
            str_710130 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 29), 'str', 'One of (fillvalue, f2) must be given.')
            # Processing the call keyword arguments (line 43)
            kwargs_710131 = {}
            # Getting the type of 'ValueError' (line 43)
            ValueError_710129 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 18), 'ValueError', False)
            # Calling ValueError(args, kwargs) (line 43)
            ValueError_call_result_710132 = invoke(stypy.reporting.localization.Localization(__file__, 43, 18), ValueError_710129, *[str_710130], **kwargs_710131)
            
            ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 43, 12), ValueError_call_result_710132, 'raise parameter', BaseException)

            if more_types_in_union_710128:
                # Runtime conditional SSA for else branch (line 42)
                module_type_store.open_ssa_branch('idiom else')



        if ((not may_be_710127) or more_types_in_union_710128):
            
            # Assigning a Attribute to a Name (line 45):
            
            # Assigning a Attribute to a Name (line 45):
            # Getting the type of 'np' (line 45)
            np_710133 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 24), 'np')
            # Obtaining the member 'nan' of a type (line 45)
            nan_710134 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 45, 24), np_710133, 'nan')
            # Assigning a type to the variable 'fillvalue' (line 45)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 45, 12), 'fillvalue', nan_710134)

            if (may_be_710127 and more_types_in_union_710128):
                # SSA join for if statement (line 42)
                module_type_store = module_type_store.join_ssa_context()


        

        if more_types_in_union_710124:
            # Runtime conditional SSA for else branch (line 41)
            module_type_store.open_ssa_branch('idiom else')



    if ((not may_be_710123) or more_types_in_union_710124):
        
        # Type idiom detected: calculating its left and rigth part (line 47)
        # Getting the type of 'f2' (line 47)
        f2_710135 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 8), 'f2')
        # Getting the type of 'None' (line 47)
        None_710136 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 21), 'None')
        
        (may_be_710137, more_types_in_union_710138) = may_not_be_none(f2_710135, None_710136)

        if may_be_710137:

            if more_types_in_union_710138:
                # Runtime conditional SSA (line 47)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Call to ValueError(...): (line 48)
            # Processing the call arguments (line 48)
            str_710140 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 48, 29), 'str', 'Only one of (fillvalue, f2) can be given.')
            # Processing the call keyword arguments (line 48)
            kwargs_710141 = {}
            # Getting the type of 'ValueError' (line 48)
            ValueError_710139 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 18), 'ValueError', False)
            # Calling ValueError(args, kwargs) (line 48)
            ValueError_call_result_710142 = invoke(stypy.reporting.localization.Localization(__file__, 48, 18), ValueError_710139, *[str_710140], **kwargs_710141)
            
            ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 48, 12), ValueError_call_result_710142, 'raise parameter', BaseException)

            if more_types_in_union_710138:
                # SSA join for if statement (line 47)
                module_type_store = module_type_store.join_ssa_context()


        

        if (may_be_710123 and more_types_in_union_710124):
            # SSA join for if statement (line 41)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Assigning a Call to a Name (line 50):
    
    # Assigning a Call to a Name (line 50):
    
    # Call to broadcast_arrays(...): (line 50)
    # Getting the type of 'arrays' (line 50)
    arrays_710145 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 34), 'arrays', False)
    # Processing the call keyword arguments (line 50)
    kwargs_710146 = {}
    # Getting the type of 'np' (line 50)
    np_710143 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 13), 'np', False)
    # Obtaining the member 'broadcast_arrays' of a type (line 50)
    broadcast_arrays_710144 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 50, 13), np_710143, 'broadcast_arrays')
    # Calling broadcast_arrays(args, kwargs) (line 50)
    broadcast_arrays_call_result_710147 = invoke(stypy.reporting.localization.Localization(__file__, 50, 13), broadcast_arrays_710144, *[arrays_710145], **kwargs_710146)
    
    # Assigning a type to the variable 'arrays' (line 50)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 50, 4), 'arrays', broadcast_arrays_call_result_710147)
    
    # Assigning a Call to a Name (line 51):
    
    # Assigning a Call to a Name (line 51):
    
    # Call to tuple(...): (line 51)
    # Processing the call arguments (line 51)
    # Calculating generator expression
    module_type_store = module_type_store.open_function_context('list comprehension expression', 51, 17, True)
    # Calculating comprehension expression
    # Getting the type of 'arrays' (line 51)
    arrays_710155 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 50), 'arrays', False)
    comprehension_710156 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 51, 17), arrays_710155)
    # Assigning a type to the variable 'arr' (line 51)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 51, 17), 'arr', comprehension_710156)
    
    # Call to extract(...): (line 51)
    # Processing the call arguments (line 51)
    # Getting the type of 'cond' (line 51)
    cond_710151 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 28), 'cond', False)
    # Getting the type of 'arr' (line 51)
    arr_710152 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 34), 'arr', False)
    # Processing the call keyword arguments (line 51)
    kwargs_710153 = {}
    # Getting the type of 'np' (line 51)
    np_710149 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 17), 'np', False)
    # Obtaining the member 'extract' of a type (line 51)
    extract_710150 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 51, 17), np_710149, 'extract')
    # Calling extract(args, kwargs) (line 51)
    extract_call_result_710154 = invoke(stypy.reporting.localization.Localization(__file__, 51, 17), extract_710150, *[cond_710151, arr_710152], **kwargs_710153)
    
    list_710157 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 51, 17), 'list')
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 51, 17), list_710157, extract_call_result_710154)
    # Processing the call keyword arguments (line 51)
    kwargs_710158 = {}
    # Getting the type of 'tuple' (line 51)
    tuple_710148 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 11), 'tuple', False)
    # Calling tuple(args, kwargs) (line 51)
    tuple_call_result_710159 = invoke(stypy.reporting.localization.Localization(__file__, 51, 11), tuple_710148, *[list_710157], **kwargs_710158)
    
    # Assigning a type to the variable 'temp' (line 51)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 51, 4), 'temp', tuple_call_result_710159)
    
    # Assigning a Call to a Name (line 52):
    
    # Assigning a Call to a Name (line 52):
    
    # Call to mintypecode(...): (line 52)
    # Processing the call arguments (line 52)
    # Calculating list comprehension
    # Calculating comprehension expression
    # Getting the type of 'arrays' (line 52)
    arrays_710165 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 50), 'arrays', False)
    comprehension_710166 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 52, 28), arrays_710165)
    # Assigning a type to the variable 'a' (line 52)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 52, 28), 'a', comprehension_710166)
    # Getting the type of 'a' (line 52)
    a_710162 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 28), 'a', False)
    # Obtaining the member 'dtype' of a type (line 52)
    dtype_710163 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 52, 28), a_710162, 'dtype')
    # Obtaining the member 'char' of a type (line 52)
    char_710164 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 52, 28), dtype_710163, 'char')
    list_710167 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 52, 28), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 52, 28), list_710167, char_710164)
    # Processing the call keyword arguments (line 52)
    kwargs_710168 = {}
    # Getting the type of 'np' (line 52)
    np_710160 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 12), 'np', False)
    # Obtaining the member 'mintypecode' of a type (line 52)
    mintypecode_710161 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 52, 12), np_710160, 'mintypecode')
    # Calling mintypecode(args, kwargs) (line 52)
    mintypecode_call_result_710169 = invoke(stypy.reporting.localization.Localization(__file__, 52, 12), mintypecode_710161, *[list_710167], **kwargs_710168)
    
    # Assigning a type to the variable 'tcode' (line 52)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 52, 4), 'tcode', mintypecode_call_result_710169)
    
    # Assigning a Call to a Name (line 53):
    
    # Assigning a Call to a Name (line 53):
    
    # Call to _valarray(...): (line 53)
    # Processing the call arguments (line 53)
    
    # Call to shape(...): (line 53)
    # Processing the call arguments (line 53)
    
    # Obtaining the type of the subscript
    int_710173 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 53, 36), 'int')
    # Getting the type of 'arrays' (line 53)
    arrays_710174 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 29), 'arrays', False)
    # Obtaining the member '__getitem__' of a type (line 53)
    getitem___710175 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 53, 29), arrays_710174, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 53)
    subscript_call_result_710176 = invoke(stypy.reporting.localization.Localization(__file__, 53, 29), getitem___710175, int_710173)
    
    # Processing the call keyword arguments (line 53)
    kwargs_710177 = {}
    # Getting the type of 'np' (line 53)
    np_710171 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 20), 'np', False)
    # Obtaining the member 'shape' of a type (line 53)
    shape_710172 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 53, 20), np_710171, 'shape')
    # Calling shape(args, kwargs) (line 53)
    shape_call_result_710178 = invoke(stypy.reporting.localization.Localization(__file__, 53, 20), shape_710172, *[subscript_call_result_710176], **kwargs_710177)
    
    # Processing the call keyword arguments (line 53)
    # Getting the type of 'fillvalue' (line 53)
    fillvalue_710179 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 47), 'fillvalue', False)
    keyword_710180 = fillvalue_710179
    # Getting the type of 'tcode' (line 53)
    tcode_710181 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 67), 'tcode', False)
    keyword_710182 = tcode_710181
    kwargs_710183 = {'value': keyword_710180, 'typecode': keyword_710182}
    # Getting the type of '_valarray' (line 53)
    _valarray_710170 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 10), '_valarray', False)
    # Calling _valarray(args, kwargs) (line 53)
    _valarray_call_result_710184 = invoke(stypy.reporting.localization.Localization(__file__, 53, 10), _valarray_710170, *[shape_call_result_710178], **kwargs_710183)
    
    # Assigning a type to the variable 'out' (line 53)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 53, 4), 'out', _valarray_call_result_710184)
    
    # Call to place(...): (line 54)
    # Processing the call arguments (line 54)
    # Getting the type of 'out' (line 54)
    out_710187 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 13), 'out', False)
    # Getting the type of 'cond' (line 54)
    cond_710188 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 18), 'cond', False)
    
    # Call to f(...): (line 54)
    # Getting the type of 'temp' (line 54)
    temp_710190 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 27), 'temp', False)
    # Processing the call keyword arguments (line 54)
    kwargs_710191 = {}
    # Getting the type of 'f' (line 54)
    f_710189 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 24), 'f', False)
    # Calling f(args, kwargs) (line 54)
    f_call_result_710192 = invoke(stypy.reporting.localization.Localization(__file__, 54, 24), f_710189, *[temp_710190], **kwargs_710191)
    
    # Processing the call keyword arguments (line 54)
    kwargs_710193 = {}
    # Getting the type of 'np' (line 54)
    np_710185 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 4), 'np', False)
    # Obtaining the member 'place' of a type (line 54)
    place_710186 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 54, 4), np_710185, 'place')
    # Calling place(args, kwargs) (line 54)
    place_call_result_710194 = invoke(stypy.reporting.localization.Localization(__file__, 54, 4), place_710186, *[out_710187, cond_710188, f_call_result_710192], **kwargs_710193)
    
    
    # Type idiom detected: calculating its left and rigth part (line 55)
    # Getting the type of 'f2' (line 55)
    f2_710195 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 4), 'f2')
    # Getting the type of 'None' (line 55)
    None_710196 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 17), 'None')
    
    (may_be_710197, more_types_in_union_710198) = may_not_be_none(f2_710195, None_710196)

    if may_be_710197:

        if more_types_in_union_710198:
            # Runtime conditional SSA (line 55)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a Call to a Name (line 56):
        
        # Assigning a Call to a Name (line 56):
        
        # Call to tuple(...): (line 56)
        # Processing the call arguments (line 56)
        # Calculating generator expression
        module_type_store = module_type_store.open_function_context('list comprehension expression', 56, 21, True)
        # Calculating comprehension expression
        # Getting the type of 'arrays' (line 56)
        arrays_710207 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 55), 'arrays', False)
        comprehension_710208 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 56, 21), arrays_710207)
        # Assigning a type to the variable 'arr' (line 56)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 56, 21), 'arr', comprehension_710208)
        
        # Call to extract(...): (line 56)
        # Processing the call arguments (line 56)
        
        # Getting the type of 'cond' (line 56)
        cond_710202 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 33), 'cond', False)
        # Applying the '~' unary operator (line 56)
        result_inv_710203 = python_operator(stypy.reporting.localization.Localization(__file__, 56, 32), '~', cond_710202)
        
        # Getting the type of 'arr' (line 56)
        arr_710204 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 39), 'arr', False)
        # Processing the call keyword arguments (line 56)
        kwargs_710205 = {}
        # Getting the type of 'np' (line 56)
        np_710200 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 21), 'np', False)
        # Obtaining the member 'extract' of a type (line 56)
        extract_710201 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 56, 21), np_710200, 'extract')
        # Calling extract(args, kwargs) (line 56)
        extract_call_result_710206 = invoke(stypy.reporting.localization.Localization(__file__, 56, 21), extract_710201, *[result_inv_710203, arr_710204], **kwargs_710205)
        
        list_710209 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, 21), 'list')
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 56, 21), list_710209, extract_call_result_710206)
        # Processing the call keyword arguments (line 56)
        kwargs_710210 = {}
        # Getting the type of 'tuple' (line 56)
        tuple_710199 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 15), 'tuple', False)
        # Calling tuple(args, kwargs) (line 56)
        tuple_call_result_710211 = invoke(stypy.reporting.localization.Localization(__file__, 56, 15), tuple_710199, *[list_710209], **kwargs_710210)
        
        # Assigning a type to the variable 'temp' (line 56)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 56, 8), 'temp', tuple_call_result_710211)
        
        # Call to place(...): (line 57)
        # Processing the call arguments (line 57)
        # Getting the type of 'out' (line 57)
        out_710214 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 17), 'out', False)
        
        # Getting the type of 'cond' (line 57)
        cond_710215 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 23), 'cond', False)
        # Applying the '~' unary operator (line 57)
        result_inv_710216 = python_operator(stypy.reporting.localization.Localization(__file__, 57, 22), '~', cond_710215)
        
        
        # Call to f2(...): (line 57)
        # Getting the type of 'temp' (line 57)
        temp_710218 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 33), 'temp', False)
        # Processing the call keyword arguments (line 57)
        kwargs_710219 = {}
        # Getting the type of 'f2' (line 57)
        f2_710217 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 29), 'f2', False)
        # Calling f2(args, kwargs) (line 57)
        f2_call_result_710220 = invoke(stypy.reporting.localization.Localization(__file__, 57, 29), f2_710217, *[temp_710218], **kwargs_710219)
        
        # Processing the call keyword arguments (line 57)
        kwargs_710221 = {}
        # Getting the type of 'np' (line 57)
        np_710212 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 8), 'np', False)
        # Obtaining the member 'place' of a type (line 57)
        place_710213 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 57, 8), np_710212, 'place')
        # Calling place(args, kwargs) (line 57)
        place_call_result_710222 = invoke(stypy.reporting.localization.Localization(__file__, 57, 8), place_710213, *[out_710214, result_inv_710216, f2_call_result_710220], **kwargs_710221)
        

        if more_types_in_union_710198:
            # SSA join for if statement (line 55)
            module_type_store = module_type_store.join_ssa_context()


    
    # Getting the type of 'out' (line 59)
    out_710223 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 11), 'out')
    # Assigning a type to the variable 'stypy_return_type' (line 59)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 59, 4), 'stypy_return_type', out_710223)
    
    # ################# End of '_lazywhere(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_lazywhere' in the type store
    # Getting the type of 'stypy_return_type' (line 26)
    stypy_return_type_710224 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_710224)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_lazywhere'
    return stypy_return_type_710224

# Assigning a type to the variable '_lazywhere' (line 26)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 26, 0), '_lazywhere', _lazywhere)

@norecursion
def _lazyselect(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    int_710225 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 62, 54), 'int')
    defaults = [int_710225]
    # Create a new context for function '_lazyselect'
    module_type_store = module_type_store.open_function_context('_lazyselect', 62, 0, False)
    
    # Passed parameters checking function
    _lazyselect.stypy_localization = localization
    _lazyselect.stypy_type_of_self = None
    _lazyselect.stypy_type_store = module_type_store
    _lazyselect.stypy_function_name = '_lazyselect'
    _lazyselect.stypy_param_names_list = ['condlist', 'choicelist', 'arrays', 'default']
    _lazyselect.stypy_varargs_param_name = None
    _lazyselect.stypy_kwargs_param_name = None
    _lazyselect.stypy_call_defaults = defaults
    _lazyselect.stypy_call_varargs = varargs
    _lazyselect.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_lazyselect', ['condlist', 'choicelist', 'arrays', 'default'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_lazyselect', localization, ['condlist', 'choicelist', 'arrays', 'default'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_lazyselect(...)' code ##################

    str_710226 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 88, (-1)), 'str', '\n    Mimic `np.select(condlist, choicelist)`.\n\n    Notice it assumes that all `arrays` are of the same shape, or can be\n    broadcasted together.\n\n    All functions in `choicelist` must accept array arguments in the order\n    given in `arrays` and must return an array of the same shape as broadcasted\n    `arrays`.\n\n    Examples\n    --------\n    >>> x = np.arange(6)\n    >>> np.select([x <3, x > 3], [x**2, x**3], default=0)\n    array([  0,   1,   4,   0,  64, 125])\n\n    >>> _lazyselect([x < 3, x > 3], [lambda x: x**2, lambda x: x**3], (x,))\n    array([   0.,    1.,    4.,   0.,   64.,  125.])\n\n    >>> a = -np.ones_like(x)\n    >>> _lazyselect([x < 3, x > 3],\n    ...             [lambda x, a: x**2, lambda x, a: a * x**3],\n    ...             (x, a), default=np.nan)\n    array([   0.,    1.,    4.,   nan,  -64., -125.])\n\n    ')
    
    # Assigning a Call to a Name (line 89):
    
    # Assigning a Call to a Name (line 89):
    
    # Call to broadcast_arrays(...): (line 89)
    # Getting the type of 'arrays' (line 89)
    arrays_710229 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 34), 'arrays', False)
    # Processing the call keyword arguments (line 89)
    kwargs_710230 = {}
    # Getting the type of 'np' (line 89)
    np_710227 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 13), 'np', False)
    # Obtaining the member 'broadcast_arrays' of a type (line 89)
    broadcast_arrays_710228 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 89, 13), np_710227, 'broadcast_arrays')
    # Calling broadcast_arrays(args, kwargs) (line 89)
    broadcast_arrays_call_result_710231 = invoke(stypy.reporting.localization.Localization(__file__, 89, 13), broadcast_arrays_710228, *[arrays_710229], **kwargs_710230)
    
    # Assigning a type to the variable 'arrays' (line 89)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 89, 4), 'arrays', broadcast_arrays_call_result_710231)
    
    # Assigning a Call to a Name (line 90):
    
    # Assigning a Call to a Name (line 90):
    
    # Call to mintypecode(...): (line 90)
    # Processing the call arguments (line 90)
    # Calculating list comprehension
    # Calculating comprehension expression
    # Getting the type of 'arrays' (line 90)
    arrays_710237 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 50), 'arrays', False)
    comprehension_710238 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 90, 28), arrays_710237)
    # Assigning a type to the variable 'a' (line 90)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 90, 28), 'a', comprehension_710238)
    # Getting the type of 'a' (line 90)
    a_710234 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 28), 'a', False)
    # Obtaining the member 'dtype' of a type (line 90)
    dtype_710235 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 90, 28), a_710234, 'dtype')
    # Obtaining the member 'char' of a type (line 90)
    char_710236 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 90, 28), dtype_710235, 'char')
    list_710239 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 90, 28), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 90, 28), list_710239, char_710236)
    # Processing the call keyword arguments (line 90)
    kwargs_710240 = {}
    # Getting the type of 'np' (line 90)
    np_710232 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 12), 'np', False)
    # Obtaining the member 'mintypecode' of a type (line 90)
    mintypecode_710233 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 90, 12), np_710232, 'mintypecode')
    # Calling mintypecode(args, kwargs) (line 90)
    mintypecode_call_result_710241 = invoke(stypy.reporting.localization.Localization(__file__, 90, 12), mintypecode_710233, *[list_710239], **kwargs_710240)
    
    # Assigning a type to the variable 'tcode' (line 90)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 90, 4), 'tcode', mintypecode_call_result_710241)
    
    # Assigning a Call to a Name (line 91):
    
    # Assigning a Call to a Name (line 91):
    
    # Call to _valarray(...): (line 91)
    # Processing the call arguments (line 91)
    
    # Call to shape(...): (line 91)
    # Processing the call arguments (line 91)
    
    # Obtaining the type of the subscript
    int_710245 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 91, 36), 'int')
    # Getting the type of 'arrays' (line 91)
    arrays_710246 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 29), 'arrays', False)
    # Obtaining the member '__getitem__' of a type (line 91)
    getitem___710247 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 91, 29), arrays_710246, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 91)
    subscript_call_result_710248 = invoke(stypy.reporting.localization.Localization(__file__, 91, 29), getitem___710247, int_710245)
    
    # Processing the call keyword arguments (line 91)
    kwargs_710249 = {}
    # Getting the type of 'np' (line 91)
    np_710243 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 20), 'np', False)
    # Obtaining the member 'shape' of a type (line 91)
    shape_710244 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 91, 20), np_710243, 'shape')
    # Calling shape(args, kwargs) (line 91)
    shape_call_result_710250 = invoke(stypy.reporting.localization.Localization(__file__, 91, 20), shape_710244, *[subscript_call_result_710248], **kwargs_710249)
    
    # Processing the call keyword arguments (line 91)
    # Getting the type of 'default' (line 91)
    default_710251 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 47), 'default', False)
    keyword_710252 = default_710251
    # Getting the type of 'tcode' (line 91)
    tcode_710253 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 65), 'tcode', False)
    keyword_710254 = tcode_710253
    kwargs_710255 = {'value': keyword_710252, 'typecode': keyword_710254}
    # Getting the type of '_valarray' (line 91)
    _valarray_710242 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 10), '_valarray', False)
    # Calling _valarray(args, kwargs) (line 91)
    _valarray_call_result_710256 = invoke(stypy.reporting.localization.Localization(__file__, 91, 10), _valarray_710242, *[shape_call_result_710250], **kwargs_710255)
    
    # Assigning a type to the variable 'out' (line 91)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 91, 4), 'out', _valarray_call_result_710256)
    
    
    # Call to range(...): (line 92)
    # Processing the call arguments (line 92)
    
    # Call to len(...): (line 92)
    # Processing the call arguments (line 92)
    # Getting the type of 'condlist' (line 92)
    condlist_710259 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 27), 'condlist', False)
    # Processing the call keyword arguments (line 92)
    kwargs_710260 = {}
    # Getting the type of 'len' (line 92)
    len_710258 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 23), 'len', False)
    # Calling len(args, kwargs) (line 92)
    len_call_result_710261 = invoke(stypy.reporting.localization.Localization(__file__, 92, 23), len_710258, *[condlist_710259], **kwargs_710260)
    
    # Processing the call keyword arguments (line 92)
    kwargs_710262 = {}
    # Getting the type of 'range' (line 92)
    range_710257 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 17), 'range', False)
    # Calling range(args, kwargs) (line 92)
    range_call_result_710263 = invoke(stypy.reporting.localization.Localization(__file__, 92, 17), range_710257, *[len_call_result_710261], **kwargs_710262)
    
    # Testing the type of a for loop iterable (line 92)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 92, 4), range_call_result_710263)
    # Getting the type of the for loop variable (line 92)
    for_loop_var_710264 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 92, 4), range_call_result_710263)
    # Assigning a type to the variable 'index' (line 92)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 92, 4), 'index', for_loop_var_710264)
    # SSA begins for a for statement (line 92)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Tuple to a Tuple (line 93):
    
    # Assigning a Subscript to a Name (line 93):
    
    # Obtaining the type of the subscript
    # Getting the type of 'index' (line 93)
    index_710265 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 32), 'index')
    # Getting the type of 'choicelist' (line 93)
    choicelist_710266 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 21), 'choicelist')
    # Obtaining the member '__getitem__' of a type (line 93)
    getitem___710267 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 93, 21), choicelist_710266, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 93)
    subscript_call_result_710268 = invoke(stypy.reporting.localization.Localization(__file__, 93, 21), getitem___710267, index_710265)
    
    # Assigning a type to the variable 'tuple_assignment_710075' (line 93)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 93, 8), 'tuple_assignment_710075', subscript_call_result_710268)
    
    # Assigning a Subscript to a Name (line 93):
    
    # Obtaining the type of the subscript
    # Getting the type of 'index' (line 93)
    index_710269 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 49), 'index')
    # Getting the type of 'condlist' (line 93)
    condlist_710270 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 40), 'condlist')
    # Obtaining the member '__getitem__' of a type (line 93)
    getitem___710271 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 93, 40), condlist_710270, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 93)
    subscript_call_result_710272 = invoke(stypy.reporting.localization.Localization(__file__, 93, 40), getitem___710271, index_710269)
    
    # Assigning a type to the variable 'tuple_assignment_710076' (line 93)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 93, 8), 'tuple_assignment_710076', subscript_call_result_710272)
    
    # Assigning a Name to a Name (line 93):
    # Getting the type of 'tuple_assignment_710075' (line 93)
    tuple_assignment_710075_710273 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 8), 'tuple_assignment_710075')
    # Assigning a type to the variable 'func' (line 93)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 93, 8), 'func', tuple_assignment_710075_710273)
    
    # Assigning a Name to a Name (line 93):
    # Getting the type of 'tuple_assignment_710076' (line 93)
    tuple_assignment_710076_710274 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 8), 'tuple_assignment_710076')
    # Assigning a type to the variable 'cond' (line 93)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 93, 14), 'cond', tuple_assignment_710076_710274)
    
    
    # Call to all(...): (line 94)
    # Processing the call arguments (line 94)
    
    # Getting the type of 'cond' (line 94)
    cond_710277 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 18), 'cond', False)
    # Getting the type of 'False' (line 94)
    False_710278 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 26), 'False', False)
    # Applying the binary operator 'is' (line 94)
    result_is__710279 = python_operator(stypy.reporting.localization.Localization(__file__, 94, 18), 'is', cond_710277, False_710278)
    
    # Processing the call keyword arguments (line 94)
    kwargs_710280 = {}
    # Getting the type of 'np' (line 94)
    np_710275 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 11), 'np', False)
    # Obtaining the member 'all' of a type (line 94)
    all_710276 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 94, 11), np_710275, 'all')
    # Calling all(args, kwargs) (line 94)
    all_call_result_710281 = invoke(stypy.reporting.localization.Localization(__file__, 94, 11), all_710276, *[result_is__710279], **kwargs_710280)
    
    # Testing the type of an if condition (line 94)
    if_condition_710282 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 94, 8), all_call_result_710281)
    # Assigning a type to the variable 'if_condition_710282' (line 94)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 94, 8), 'if_condition_710282', if_condition_710282)
    # SSA begins for if statement (line 94)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # SSA join for if statement (line 94)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Tuple (line 96):
    
    # Assigning a Subscript to a Name (line 96):
    
    # Obtaining the type of the subscript
    int_710283 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 96, 8), 'int')
    
    # Call to broadcast_arrays(...): (line 96)
    # Processing the call arguments (line 96)
    # Getting the type of 'cond' (line 96)
    cond_710286 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 38), 'cond', False)
    
    # Obtaining the type of the subscript
    int_710287 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 96, 51), 'int')
    # Getting the type of 'arrays' (line 96)
    arrays_710288 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 44), 'arrays', False)
    # Obtaining the member '__getitem__' of a type (line 96)
    getitem___710289 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 96, 44), arrays_710288, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 96)
    subscript_call_result_710290 = invoke(stypy.reporting.localization.Localization(__file__, 96, 44), getitem___710289, int_710287)
    
    # Processing the call keyword arguments (line 96)
    kwargs_710291 = {}
    # Getting the type of 'np' (line 96)
    np_710284 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 18), 'np', False)
    # Obtaining the member 'broadcast_arrays' of a type (line 96)
    broadcast_arrays_710285 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 96, 18), np_710284, 'broadcast_arrays')
    # Calling broadcast_arrays(args, kwargs) (line 96)
    broadcast_arrays_call_result_710292 = invoke(stypy.reporting.localization.Localization(__file__, 96, 18), broadcast_arrays_710285, *[cond_710286, subscript_call_result_710290], **kwargs_710291)
    
    # Obtaining the member '__getitem__' of a type (line 96)
    getitem___710293 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 96, 8), broadcast_arrays_call_result_710292, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 96)
    subscript_call_result_710294 = invoke(stypy.reporting.localization.Localization(__file__, 96, 8), getitem___710293, int_710283)
    
    # Assigning a type to the variable 'tuple_var_assignment_710077' (line 96)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 96, 8), 'tuple_var_assignment_710077', subscript_call_result_710294)
    
    # Assigning a Subscript to a Name (line 96):
    
    # Obtaining the type of the subscript
    int_710295 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 96, 8), 'int')
    
    # Call to broadcast_arrays(...): (line 96)
    # Processing the call arguments (line 96)
    # Getting the type of 'cond' (line 96)
    cond_710298 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 38), 'cond', False)
    
    # Obtaining the type of the subscript
    int_710299 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 96, 51), 'int')
    # Getting the type of 'arrays' (line 96)
    arrays_710300 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 44), 'arrays', False)
    # Obtaining the member '__getitem__' of a type (line 96)
    getitem___710301 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 96, 44), arrays_710300, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 96)
    subscript_call_result_710302 = invoke(stypy.reporting.localization.Localization(__file__, 96, 44), getitem___710301, int_710299)
    
    # Processing the call keyword arguments (line 96)
    kwargs_710303 = {}
    # Getting the type of 'np' (line 96)
    np_710296 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 18), 'np', False)
    # Obtaining the member 'broadcast_arrays' of a type (line 96)
    broadcast_arrays_710297 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 96, 18), np_710296, 'broadcast_arrays')
    # Calling broadcast_arrays(args, kwargs) (line 96)
    broadcast_arrays_call_result_710304 = invoke(stypy.reporting.localization.Localization(__file__, 96, 18), broadcast_arrays_710297, *[cond_710298, subscript_call_result_710302], **kwargs_710303)
    
    # Obtaining the member '__getitem__' of a type (line 96)
    getitem___710305 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 96, 8), broadcast_arrays_call_result_710304, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 96)
    subscript_call_result_710306 = invoke(stypy.reporting.localization.Localization(__file__, 96, 8), getitem___710305, int_710295)
    
    # Assigning a type to the variable 'tuple_var_assignment_710078' (line 96)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 96, 8), 'tuple_var_assignment_710078', subscript_call_result_710306)
    
    # Assigning a Name to a Name (line 96):
    # Getting the type of 'tuple_var_assignment_710077' (line 96)
    tuple_var_assignment_710077_710307 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 8), 'tuple_var_assignment_710077')
    # Assigning a type to the variable 'cond' (line 96)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 96, 8), 'cond', tuple_var_assignment_710077_710307)
    
    # Assigning a Name to a Name (line 96):
    # Getting the type of 'tuple_var_assignment_710078' (line 96)
    tuple_var_assignment_710078_710308 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 8), 'tuple_var_assignment_710078')
    # Assigning a type to the variable '_' (line 96)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 96, 14), '_', tuple_var_assignment_710078_710308)
    
    # Assigning a Call to a Name (line 97):
    
    # Assigning a Call to a Name (line 97):
    
    # Call to tuple(...): (line 97)
    # Processing the call arguments (line 97)
    # Calculating generator expression
    module_type_store = module_type_store.open_function_context('list comprehension expression', 97, 21, True)
    # Calculating comprehension expression
    # Getting the type of 'arrays' (line 97)
    arrays_710316 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 54), 'arrays', False)
    comprehension_710317 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 97, 21), arrays_710316)
    # Assigning a type to the variable 'arr' (line 97)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 97, 21), 'arr', comprehension_710317)
    
    # Call to extract(...): (line 97)
    # Processing the call arguments (line 97)
    # Getting the type of 'cond' (line 97)
    cond_710312 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 32), 'cond', False)
    # Getting the type of 'arr' (line 97)
    arr_710313 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 38), 'arr', False)
    # Processing the call keyword arguments (line 97)
    kwargs_710314 = {}
    # Getting the type of 'np' (line 97)
    np_710310 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 21), 'np', False)
    # Obtaining the member 'extract' of a type (line 97)
    extract_710311 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 97, 21), np_710310, 'extract')
    # Calling extract(args, kwargs) (line 97)
    extract_call_result_710315 = invoke(stypy.reporting.localization.Localization(__file__, 97, 21), extract_710311, *[cond_710312, arr_710313], **kwargs_710314)
    
    list_710318 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 97, 21), 'list')
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 97, 21), list_710318, extract_call_result_710315)
    # Processing the call keyword arguments (line 97)
    kwargs_710319 = {}
    # Getting the type of 'tuple' (line 97)
    tuple_710309 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 15), 'tuple', False)
    # Calling tuple(args, kwargs) (line 97)
    tuple_call_result_710320 = invoke(stypy.reporting.localization.Localization(__file__, 97, 15), tuple_710309, *[list_710318], **kwargs_710319)
    
    # Assigning a type to the variable 'temp' (line 97)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 97, 8), 'temp', tuple_call_result_710320)
    
    # Call to place(...): (line 98)
    # Processing the call arguments (line 98)
    # Getting the type of 'out' (line 98)
    out_710323 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 17), 'out', False)
    # Getting the type of 'cond' (line 98)
    cond_710324 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 22), 'cond', False)
    
    # Call to func(...): (line 98)
    # Getting the type of 'temp' (line 98)
    temp_710326 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 34), 'temp', False)
    # Processing the call keyword arguments (line 98)
    kwargs_710327 = {}
    # Getting the type of 'func' (line 98)
    func_710325 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 28), 'func', False)
    # Calling func(args, kwargs) (line 98)
    func_call_result_710328 = invoke(stypy.reporting.localization.Localization(__file__, 98, 28), func_710325, *[temp_710326], **kwargs_710327)
    
    # Processing the call keyword arguments (line 98)
    kwargs_710329 = {}
    # Getting the type of 'np' (line 98)
    np_710321 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 8), 'np', False)
    # Obtaining the member 'place' of a type (line 98)
    place_710322 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 98, 8), np_710321, 'place')
    # Calling place(args, kwargs) (line 98)
    place_call_result_710330 = invoke(stypy.reporting.localization.Localization(__file__, 98, 8), place_710322, *[out_710323, cond_710324, func_call_result_710328], **kwargs_710329)
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'out' (line 99)
    out_710331 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 11), 'out')
    # Assigning a type to the variable 'stypy_return_type' (line 99)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 99, 4), 'stypy_return_type', out_710331)
    
    # ################# End of '_lazyselect(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_lazyselect' in the type store
    # Getting the type of 'stypy_return_type' (line 62)
    stypy_return_type_710332 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_710332)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_lazyselect'
    return stypy_return_type_710332

# Assigning a type to the variable '_lazyselect' (line 62)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 62, 0), '_lazyselect', _lazyselect)

@norecursion
def _aligned_zeros(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'float' (line 102)
    float_710333 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 32), 'float')
    str_710334 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 102, 45), 'str', 'C')
    # Getting the type of 'None' (line 102)
    None_710335 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 56), 'None')
    defaults = [float_710333, str_710334, None_710335]
    # Create a new context for function '_aligned_zeros'
    module_type_store = module_type_store.open_function_context('_aligned_zeros', 102, 0, False)
    
    # Passed parameters checking function
    _aligned_zeros.stypy_localization = localization
    _aligned_zeros.stypy_type_of_self = None
    _aligned_zeros.stypy_type_store = module_type_store
    _aligned_zeros.stypy_function_name = '_aligned_zeros'
    _aligned_zeros.stypy_param_names_list = ['shape', 'dtype', 'order', 'align']
    _aligned_zeros.stypy_varargs_param_name = None
    _aligned_zeros.stypy_kwargs_param_name = None
    _aligned_zeros.stypy_call_defaults = defaults
    _aligned_zeros.stypy_call_varargs = varargs
    _aligned_zeros.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_aligned_zeros', ['shape', 'dtype', 'order', 'align'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_aligned_zeros', localization, ['shape', 'dtype', 'order', 'align'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_aligned_zeros(...)' code ##################

    str_710336 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 109, (-1)), 'str', 'Allocate a new ndarray with aligned memory.\n\n    Primary use case for this currently is working around a f2py issue\n    in Numpy 1.9.1, where dtype.alignment is such that np.zeros() does\n    not necessarily create arrays aligned up to it.\n\n    ')
    
    # Assigning a Call to a Name (line 110):
    
    # Assigning a Call to a Name (line 110):
    
    # Call to dtype(...): (line 110)
    # Processing the call arguments (line 110)
    # Getting the type of 'dtype' (line 110)
    dtype_710339 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 21), 'dtype', False)
    # Processing the call keyword arguments (line 110)
    kwargs_710340 = {}
    # Getting the type of 'np' (line 110)
    np_710337 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 12), 'np', False)
    # Obtaining the member 'dtype' of a type (line 110)
    dtype_710338 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 110, 12), np_710337, 'dtype')
    # Calling dtype(args, kwargs) (line 110)
    dtype_call_result_710341 = invoke(stypy.reporting.localization.Localization(__file__, 110, 12), dtype_710338, *[dtype_710339], **kwargs_710340)
    
    # Assigning a type to the variable 'dtype' (line 110)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 110, 4), 'dtype', dtype_call_result_710341)
    
    # Type idiom detected: calculating its left and rigth part (line 111)
    # Getting the type of 'align' (line 111)
    align_710342 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 7), 'align')
    # Getting the type of 'None' (line 111)
    None_710343 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 16), 'None')
    
    (may_be_710344, more_types_in_union_710345) = may_be_none(align_710342, None_710343)

    if may_be_710344:

        if more_types_in_union_710345:
            # Runtime conditional SSA (line 111)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a Attribute to a Name (line 112):
        
        # Assigning a Attribute to a Name (line 112):
        # Getting the type of 'dtype' (line 112)
        dtype_710346 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 16), 'dtype')
        # Obtaining the member 'alignment' of a type (line 112)
        alignment_710347 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 112, 16), dtype_710346, 'alignment')
        # Assigning a type to the variable 'align' (line 112)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 112, 8), 'align', alignment_710347)

        if more_types_in_union_710345:
            # SSA join for if statement (line 111)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Type idiom detected: calculating its left and rigth part (line 113)
    str_710348 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 113, 26), 'str', '__len__')
    # Getting the type of 'shape' (line 113)
    shape_710349 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 19), 'shape')
    
    (may_be_710350, more_types_in_union_710351) = may_not_provide_member(str_710348, shape_710349)

    if may_be_710350:

        if more_types_in_union_710351:
            # Runtime conditional SSA (line 113)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        # Assigning a type to the variable 'shape' (line 113)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 113, 4), 'shape', remove_member_provider_from_union(shape_710349, '__len__'))
        
        # Assigning a Tuple to a Name (line 114):
        
        # Assigning a Tuple to a Name (line 114):
        
        # Obtaining an instance of the builtin type 'tuple' (line 114)
        tuple_710352 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 114, 17), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 114)
        # Adding element type (line 114)
        # Getting the type of 'shape' (line 114)
        shape_710353 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 17), 'shape')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 114, 17), tuple_710352, shape_710353)
        
        # Assigning a type to the variable 'shape' (line 114)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 114, 8), 'shape', tuple_710352)

        if more_types_in_union_710351:
            # SSA join for if statement (line 113)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Assigning a BinOp to a Name (line 115):
    
    # Assigning a BinOp to a Name (line 115):
    
    # Call to reduce(...): (line 115)
    # Processing the call arguments (line 115)
    # Getting the type of 'operator' (line 115)
    operator_710356 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 28), 'operator', False)
    # Obtaining the member 'mul' of a type (line 115)
    mul_710357 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 115, 28), operator_710356, 'mul')
    # Getting the type of 'shape' (line 115)
    shape_710358 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 42), 'shape', False)
    # Processing the call keyword arguments (line 115)
    kwargs_710359 = {}
    # Getting the type of 'functools' (line 115)
    functools_710354 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 11), 'functools', False)
    # Obtaining the member 'reduce' of a type (line 115)
    reduce_710355 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 115, 11), functools_710354, 'reduce')
    # Calling reduce(args, kwargs) (line 115)
    reduce_call_result_710360 = invoke(stypy.reporting.localization.Localization(__file__, 115, 11), reduce_710355, *[mul_710357, shape_710358], **kwargs_710359)
    
    # Getting the type of 'dtype' (line 115)
    dtype_710361 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 51), 'dtype')
    # Obtaining the member 'itemsize' of a type (line 115)
    itemsize_710362 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 115, 51), dtype_710361, 'itemsize')
    # Applying the binary operator '*' (line 115)
    result_mul_710363 = python_operator(stypy.reporting.localization.Localization(__file__, 115, 11), '*', reduce_call_result_710360, itemsize_710362)
    
    # Assigning a type to the variable 'size' (line 115)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 115, 4), 'size', result_mul_710363)
    
    # Assigning a Call to a Name (line 116):
    
    # Assigning a Call to a Name (line 116):
    
    # Call to empty(...): (line 116)
    # Processing the call arguments (line 116)
    # Getting the type of 'size' (line 116)
    size_710366 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 19), 'size', False)
    # Getting the type of 'align' (line 116)
    align_710367 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 26), 'align', False)
    # Applying the binary operator '+' (line 116)
    result_add_710368 = python_operator(stypy.reporting.localization.Localization(__file__, 116, 19), '+', size_710366, align_710367)
    
    int_710369 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 116, 34), 'int')
    # Applying the binary operator '+' (line 116)
    result_add_710370 = python_operator(stypy.reporting.localization.Localization(__file__, 116, 32), '+', result_add_710368, int_710369)
    
    # Getting the type of 'np' (line 116)
    np_710371 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 37), 'np', False)
    # Obtaining the member 'uint8' of a type (line 116)
    uint8_710372 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 116, 37), np_710371, 'uint8')
    # Processing the call keyword arguments (line 116)
    kwargs_710373 = {}
    # Getting the type of 'np' (line 116)
    np_710364 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 10), 'np', False)
    # Obtaining the member 'empty' of a type (line 116)
    empty_710365 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 116, 10), np_710364, 'empty')
    # Calling empty(args, kwargs) (line 116)
    empty_call_result_710374 = invoke(stypy.reporting.localization.Localization(__file__, 116, 10), empty_710365, *[result_add_710370, uint8_710372], **kwargs_710373)
    
    # Assigning a type to the variable 'buf' (line 116)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 116, 4), 'buf', empty_call_result_710374)
    
    # Assigning a BinOp to a Name (line 117):
    
    # Assigning a BinOp to a Name (line 117):
    
    # Obtaining the type of the subscript
    int_710375 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 117, 45), 'int')
    
    # Obtaining the type of the subscript
    str_710376 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 117, 37), 'str', 'data')
    # Getting the type of 'buf' (line 117)
    buf_710377 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 13), 'buf')
    # Obtaining the member '__array_interface__' of a type (line 117)
    array_interface___710378 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 117, 13), buf_710377, '__array_interface__')
    # Obtaining the member '__getitem__' of a type (line 117)
    getitem___710379 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 117, 13), array_interface___710378, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 117)
    subscript_call_result_710380 = invoke(stypy.reporting.localization.Localization(__file__, 117, 13), getitem___710379, str_710376)
    
    # Obtaining the member '__getitem__' of a type (line 117)
    getitem___710381 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 117, 13), subscript_call_result_710380, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 117)
    subscript_call_result_710382 = invoke(stypy.reporting.localization.Localization(__file__, 117, 13), getitem___710381, int_710375)
    
    # Getting the type of 'align' (line 117)
    align_710383 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 50), 'align')
    # Applying the binary operator '%' (line 117)
    result_mod_710384 = python_operator(stypy.reporting.localization.Localization(__file__, 117, 13), '%', subscript_call_result_710382, align_710383)
    
    # Assigning a type to the variable 'offset' (line 117)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 117, 4), 'offset', result_mod_710384)
    
    
    # Getting the type of 'offset' (line 118)
    offset_710385 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 7), 'offset')
    int_710386 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 118, 17), 'int')
    # Applying the binary operator '!=' (line 118)
    result_ne_710387 = python_operator(stypy.reporting.localization.Localization(__file__, 118, 7), '!=', offset_710385, int_710386)
    
    # Testing the type of an if condition (line 118)
    if_condition_710388 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 118, 4), result_ne_710387)
    # Assigning a type to the variable 'if_condition_710388' (line 118)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 118, 4), 'if_condition_710388', if_condition_710388)
    # SSA begins for if statement (line 118)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a BinOp to a Name (line 119):
    
    # Assigning a BinOp to a Name (line 119):
    # Getting the type of 'align' (line 119)
    align_710389 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 17), 'align')
    # Getting the type of 'offset' (line 119)
    offset_710390 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 25), 'offset')
    # Applying the binary operator '-' (line 119)
    result_sub_710391 = python_operator(stypy.reporting.localization.Localization(__file__, 119, 17), '-', align_710389, offset_710390)
    
    # Assigning a type to the variable 'offset' (line 119)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 119, 8), 'offset', result_sub_710391)
    # SSA join for if statement (line 118)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Subscript to a Name (line 122):
    
    # Assigning a Subscript to a Name (line 122):
    
    # Obtaining the type of the subscript
    int_710392 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 122, 37), 'int')
    slice_710393 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 122, 10), None, int_710392, None)
    
    # Obtaining the type of the subscript
    # Getting the type of 'offset' (line 122)
    offset_710394 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 14), 'offset')
    # Getting the type of 'offset' (line 122)
    offset_710395 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 21), 'offset')
    # Getting the type of 'size' (line 122)
    size_710396 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 28), 'size')
    # Applying the binary operator '+' (line 122)
    result_add_710397 = python_operator(stypy.reporting.localization.Localization(__file__, 122, 21), '+', offset_710395, size_710396)
    
    int_710398 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 122, 33), 'int')
    # Applying the binary operator '+' (line 122)
    result_add_710399 = python_operator(stypy.reporting.localization.Localization(__file__, 122, 32), '+', result_add_710397, int_710398)
    
    slice_710400 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 122, 10), offset_710394, result_add_710399, None)
    # Getting the type of 'buf' (line 122)
    buf_710401 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 10), 'buf')
    # Obtaining the member '__getitem__' of a type (line 122)
    getitem___710402 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 122, 10), buf_710401, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 122)
    subscript_call_result_710403 = invoke(stypy.reporting.localization.Localization(__file__, 122, 10), getitem___710402, slice_710400)
    
    # Obtaining the member '__getitem__' of a type (line 122)
    getitem___710404 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 122, 10), subscript_call_result_710403, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 122)
    subscript_call_result_710405 = invoke(stypy.reporting.localization.Localization(__file__, 122, 10), getitem___710404, slice_710393)
    
    # Assigning a type to the variable 'buf' (line 122)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 122, 4), 'buf', subscript_call_result_710405)
    
    # Assigning a Call to a Name (line 123):
    
    # Assigning a Call to a Name (line 123):
    
    # Call to ndarray(...): (line 123)
    # Processing the call arguments (line 123)
    # Getting the type of 'shape' (line 123)
    shape_710408 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 22), 'shape', False)
    # Getting the type of 'dtype' (line 123)
    dtype_710409 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 29), 'dtype', False)
    # Getting the type of 'buf' (line 123)
    buf_710410 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 36), 'buf', False)
    # Processing the call keyword arguments (line 123)
    # Getting the type of 'order' (line 123)
    order_710411 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 47), 'order', False)
    keyword_710412 = order_710411
    kwargs_710413 = {'order': keyword_710412}
    # Getting the type of 'np' (line 123)
    np_710406 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 11), 'np', False)
    # Obtaining the member 'ndarray' of a type (line 123)
    ndarray_710407 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 123, 11), np_710406, 'ndarray')
    # Calling ndarray(args, kwargs) (line 123)
    ndarray_call_result_710414 = invoke(stypy.reporting.localization.Localization(__file__, 123, 11), ndarray_710407, *[shape_710408, dtype_710409, buf_710410], **kwargs_710413)
    
    # Assigning a type to the variable 'data' (line 123)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 123, 4), 'data', ndarray_call_result_710414)
    
    # Call to fill(...): (line 124)
    # Processing the call arguments (line 124)
    int_710417 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 124, 14), 'int')
    # Processing the call keyword arguments (line 124)
    kwargs_710418 = {}
    # Getting the type of 'data' (line 124)
    data_710415 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 4), 'data', False)
    # Obtaining the member 'fill' of a type (line 124)
    fill_710416 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 124, 4), data_710415, 'fill')
    # Calling fill(args, kwargs) (line 124)
    fill_call_result_710419 = invoke(stypy.reporting.localization.Localization(__file__, 124, 4), fill_710416, *[int_710417], **kwargs_710418)
    
    # Getting the type of 'data' (line 125)
    data_710420 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 11), 'data')
    # Assigning a type to the variable 'stypy_return_type' (line 125)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 125, 4), 'stypy_return_type', data_710420)
    
    # ################# End of '_aligned_zeros(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_aligned_zeros' in the type store
    # Getting the type of 'stypy_return_type' (line 102)
    stypy_return_type_710421 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_710421)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_aligned_zeros'
    return stypy_return_type_710421

# Assigning a type to the variable '_aligned_zeros' (line 102)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 102, 0), '_aligned_zeros', _aligned_zeros)

@norecursion
def _prune_array(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_prune_array'
    module_type_store = module_type_store.open_function_context('_prune_array', 128, 0, False)
    
    # Passed parameters checking function
    _prune_array.stypy_localization = localization
    _prune_array.stypy_type_of_self = None
    _prune_array.stypy_type_store = module_type_store
    _prune_array.stypy_function_name = '_prune_array'
    _prune_array.stypy_param_names_list = ['array']
    _prune_array.stypy_varargs_param_name = None
    _prune_array.stypy_kwargs_param_name = None
    _prune_array.stypy_call_defaults = defaults
    _prune_array.stypy_call_varargs = varargs
    _prune_array.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_prune_array', ['array'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_prune_array', localization, ['array'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_prune_array(...)' code ##################

    str_710422 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 132, (-1)), 'str', 'Return an array equivalent to the input array. If the input\n    array is a view of a much larger array, copy its contents to a\n    newly allocated array. Otherwise, return the input unchaged.\n    ')
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'array' (line 133)
    array_710423 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 7), 'array')
    # Obtaining the member 'base' of a type (line 133)
    base_710424 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 133, 7), array_710423, 'base')
    # Getting the type of 'None' (line 133)
    None_710425 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 25), 'None')
    # Applying the binary operator 'isnot' (line 133)
    result_is_not_710426 = python_operator(stypy.reporting.localization.Localization(__file__, 133, 7), 'isnot', base_710424, None_710425)
    
    
    # Getting the type of 'array' (line 133)
    array_710427 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 34), 'array')
    # Obtaining the member 'size' of a type (line 133)
    size_710428 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 133, 34), array_710427, 'size')
    # Getting the type of 'array' (line 133)
    array_710429 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 47), 'array')
    # Obtaining the member 'base' of a type (line 133)
    base_710430 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 133, 47), array_710429, 'base')
    # Obtaining the member 'size' of a type (line 133)
    size_710431 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 133, 47), base_710430, 'size')
    int_710432 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 133, 66), 'int')
    # Applying the binary operator '//' (line 133)
    result_floordiv_710433 = python_operator(stypy.reporting.localization.Localization(__file__, 133, 47), '//', size_710431, int_710432)
    
    # Applying the binary operator '<' (line 133)
    result_lt_710434 = python_operator(stypy.reporting.localization.Localization(__file__, 133, 34), '<', size_710428, result_floordiv_710433)
    
    # Applying the binary operator 'and' (line 133)
    result_and_keyword_710435 = python_operator(stypy.reporting.localization.Localization(__file__, 133, 7), 'and', result_is_not_710426, result_lt_710434)
    
    # Testing the type of an if condition (line 133)
    if_condition_710436 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 133, 4), result_and_keyword_710435)
    # Assigning a type to the variable 'if_condition_710436' (line 133)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 133, 4), 'if_condition_710436', if_condition_710436)
    # SSA begins for if statement (line 133)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to copy(...): (line 134)
    # Processing the call keyword arguments (line 134)
    kwargs_710439 = {}
    # Getting the type of 'array' (line 134)
    array_710437 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 15), 'array', False)
    # Obtaining the member 'copy' of a type (line 134)
    copy_710438 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 134, 15), array_710437, 'copy')
    # Calling copy(args, kwargs) (line 134)
    copy_call_result_710440 = invoke(stypy.reporting.localization.Localization(__file__, 134, 15), copy_710438, *[], **kwargs_710439)
    
    # Assigning a type to the variable 'stypy_return_type' (line 134)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 134, 8), 'stypy_return_type', copy_call_result_710440)
    # SSA join for if statement (line 133)
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'array' (line 135)
    array_710441 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 11), 'array')
    # Assigning a type to the variable 'stypy_return_type' (line 135)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 135, 4), 'stypy_return_type', array_710441)
    
    # ################# End of '_prune_array(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_prune_array' in the type store
    # Getting the type of 'stypy_return_type' (line 128)
    stypy_return_type_710442 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_710442)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_prune_array'
    return stypy_return_type_710442

# Assigning a type to the variable '_prune_array' (line 128)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 128, 0), '_prune_array', _prune_array)
# Declaration of the 'DeprecatedImport' class

class DeprecatedImport(object, ):
    str_710443 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 154, (-1)), 'str', '\n    Deprecated import, with redirection + warning.\n\n    Examples\n    --------\n    Suppose you previously had in some module::\n\n        from foo import spam\n\n    If this has to be deprecated, do::\n\n        spam = DeprecatedImport("foo.spam", "baz")\n\n    to redirect users to use "baz" module instead.\n\n    ')

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 156, 4, False)
        # Assigning a type to the variable 'self' (line 157)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 157, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'DeprecatedImport.__init__', ['old_module_name', 'new_module_name'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['old_module_name', 'new_module_name'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        
        # Assigning a Name to a Attribute (line 157):
        
        # Assigning a Name to a Attribute (line 157):
        # Getting the type of 'old_module_name' (line 157)
        old_module_name_710444 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 25), 'old_module_name')
        # Getting the type of 'self' (line 157)
        self_710445 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 8), 'self')
        # Setting the type of the member '_old_name' of a type (line 157)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 157, 8), self_710445, '_old_name', old_module_name_710444)
        
        # Assigning a Name to a Attribute (line 158):
        
        # Assigning a Name to a Attribute (line 158):
        # Getting the type of 'new_module_name' (line 158)
        new_module_name_710446 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 25), 'new_module_name')
        # Getting the type of 'self' (line 158)
        self_710447 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 8), 'self')
        # Setting the type of the member '_new_name' of a type (line 158)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 158, 8), self_710447, '_new_name', new_module_name_710446)
        
        # Call to __import__(...): (line 159)
        # Processing the call arguments (line 159)
        # Getting the type of 'self' (line 159)
        self_710449 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 19), 'self', False)
        # Obtaining the member '_new_name' of a type (line 159)
        _new_name_710450 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 159, 19), self_710449, '_new_name')
        # Processing the call keyword arguments (line 159)
        kwargs_710451 = {}
        # Getting the type of '__import__' (line 159)
        import___710448 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 8), '__import__', False)
        # Calling __import__(args, kwargs) (line 159)
        import___call_result_710452 = invoke(stypy.reporting.localization.Localization(__file__, 159, 8), import___710448, *[_new_name_710450], **kwargs_710451)
        
        
        # Assigning a Subscript to a Attribute (line 160):
        
        # Assigning a Subscript to a Attribute (line 160):
        
        # Obtaining the type of the subscript
        # Getting the type of 'self' (line 160)
        self_710453 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 32), 'self')
        # Obtaining the member '_new_name' of a type (line 160)
        _new_name_710454 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 160, 32), self_710453, '_new_name')
        # Getting the type of 'sys' (line 160)
        sys_710455 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 20), 'sys')
        # Obtaining the member 'modules' of a type (line 160)
        modules_710456 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 160, 20), sys_710455, 'modules')
        # Obtaining the member '__getitem__' of a type (line 160)
        getitem___710457 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 160, 20), modules_710456, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 160)
        subscript_call_result_710458 = invoke(stypy.reporting.localization.Localization(__file__, 160, 20), getitem___710457, _new_name_710454)
        
        # Getting the type of 'self' (line 160)
        self_710459 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 8), 'self')
        # Setting the type of the member '_mod' of a type (line 160)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 160, 8), self_710459, '_mod', subscript_call_result_710458)
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def __dir__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__dir__'
        module_type_store = module_type_store.open_function_context('__dir__', 162, 4, False)
        # Assigning a type to the variable 'self' (line 163)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 163, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        DeprecatedImport.__dir__.__dict__.__setitem__('stypy_localization', localization)
        DeprecatedImport.__dir__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        DeprecatedImport.__dir__.__dict__.__setitem__('stypy_type_store', module_type_store)
        DeprecatedImport.__dir__.__dict__.__setitem__('stypy_function_name', 'DeprecatedImport.__dir__')
        DeprecatedImport.__dir__.__dict__.__setitem__('stypy_param_names_list', [])
        DeprecatedImport.__dir__.__dict__.__setitem__('stypy_varargs_param_name', None)
        DeprecatedImport.__dir__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        DeprecatedImport.__dir__.__dict__.__setitem__('stypy_call_defaults', defaults)
        DeprecatedImport.__dir__.__dict__.__setitem__('stypy_call_varargs', varargs)
        DeprecatedImport.__dir__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        DeprecatedImport.__dir__.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'DeprecatedImport.__dir__', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__dir__', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__dir__(...)' code ##################

        
        # Call to dir(...): (line 163)
        # Processing the call arguments (line 163)
        # Getting the type of 'self' (line 163)
        self_710461 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 19), 'self', False)
        # Obtaining the member '_mod' of a type (line 163)
        _mod_710462 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 163, 19), self_710461, '_mod')
        # Processing the call keyword arguments (line 163)
        kwargs_710463 = {}
        # Getting the type of 'dir' (line 163)
        dir_710460 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 15), 'dir', False)
        # Calling dir(args, kwargs) (line 163)
        dir_call_result_710464 = invoke(stypy.reporting.localization.Localization(__file__, 163, 15), dir_710460, *[_mod_710462], **kwargs_710463)
        
        # Assigning a type to the variable 'stypy_return_type' (line 163)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 163, 8), 'stypy_return_type', dir_call_result_710464)
        
        # ################# End of '__dir__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__dir__' in the type store
        # Getting the type of 'stypy_return_type' (line 162)
        stypy_return_type_710465 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_710465)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__dir__'
        return stypy_return_type_710465


    @norecursion
    def __getattr__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__getattr__'
        module_type_store = module_type_store.open_function_context('__getattr__', 165, 4, False)
        # Assigning a type to the variable 'self' (line 166)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 166, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        DeprecatedImport.__getattr__.__dict__.__setitem__('stypy_localization', localization)
        DeprecatedImport.__getattr__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        DeprecatedImport.__getattr__.__dict__.__setitem__('stypy_type_store', module_type_store)
        DeprecatedImport.__getattr__.__dict__.__setitem__('stypy_function_name', 'DeprecatedImport.__getattr__')
        DeprecatedImport.__getattr__.__dict__.__setitem__('stypy_param_names_list', ['name'])
        DeprecatedImport.__getattr__.__dict__.__setitem__('stypy_varargs_param_name', None)
        DeprecatedImport.__getattr__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        DeprecatedImport.__getattr__.__dict__.__setitem__('stypy_call_defaults', defaults)
        DeprecatedImport.__getattr__.__dict__.__setitem__('stypy_call_varargs', varargs)
        DeprecatedImport.__getattr__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        DeprecatedImport.__getattr__.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'DeprecatedImport.__getattr__', ['name'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__getattr__', localization, ['name'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__getattr__(...)' code ##################

        
        # Call to warn(...): (line 166)
        # Processing the call arguments (line 166)
        str_710468 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 166, 22), 'str', 'Module %s is deprecated, use %s instead')
        
        # Obtaining an instance of the builtin type 'tuple' (line 167)
        tuple_710469 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 167, 25), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 167)
        # Adding element type (line 167)
        # Getting the type of 'self' (line 167)
        self_710470 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 25), 'self', False)
        # Obtaining the member '_old_name' of a type (line 167)
        _old_name_710471 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 167, 25), self_710470, '_old_name')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 167, 25), tuple_710469, _old_name_710471)
        # Adding element type (line 167)
        # Getting the type of 'self' (line 167)
        self_710472 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 41), 'self', False)
        # Obtaining the member '_new_name' of a type (line 167)
        _new_name_710473 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 167, 41), self_710472, '_new_name')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 167, 25), tuple_710469, _new_name_710473)
        
        # Applying the binary operator '%' (line 166)
        result_mod_710474 = python_operator(stypy.reporting.localization.Localization(__file__, 166, 22), '%', str_710468, tuple_710469)
        
        # Getting the type of 'DeprecationWarning' (line 168)
        DeprecationWarning_710475 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 22), 'DeprecationWarning', False)
        # Processing the call keyword arguments (line 166)
        kwargs_710476 = {}
        # Getting the type of 'warnings' (line 166)
        warnings_710466 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 8), 'warnings', False)
        # Obtaining the member 'warn' of a type (line 166)
        warn_710467 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 166, 8), warnings_710466, 'warn')
        # Calling warn(args, kwargs) (line 166)
        warn_call_result_710477 = invoke(stypy.reporting.localization.Localization(__file__, 166, 8), warn_710467, *[result_mod_710474, DeprecationWarning_710475], **kwargs_710476)
        
        
        # Call to getattr(...): (line 169)
        # Processing the call arguments (line 169)
        # Getting the type of 'self' (line 169)
        self_710479 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 23), 'self', False)
        # Obtaining the member '_mod' of a type (line 169)
        _mod_710480 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 169, 23), self_710479, '_mod')
        # Getting the type of 'name' (line 169)
        name_710481 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 34), 'name', False)
        # Processing the call keyword arguments (line 169)
        kwargs_710482 = {}
        # Getting the type of 'getattr' (line 169)
        getattr_710478 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 15), 'getattr', False)
        # Calling getattr(args, kwargs) (line 169)
        getattr_call_result_710483 = invoke(stypy.reporting.localization.Localization(__file__, 169, 15), getattr_710478, *[_mod_710480, name_710481], **kwargs_710482)
        
        # Assigning a type to the variable 'stypy_return_type' (line 169)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 169, 8), 'stypy_return_type', getattr_call_result_710483)
        
        # ################# End of '__getattr__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__getattr__' in the type store
        # Getting the type of 'stypy_return_type' (line 165)
        stypy_return_type_710484 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_710484)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__getattr__'
        return stypy_return_type_710484


# Assigning a type to the variable 'DeprecatedImport' (line 138)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 138, 0), 'DeprecatedImport', DeprecatedImport)

@norecursion
def check_random_state(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'check_random_state'
    module_type_store = module_type_store.open_function_context('check_random_state', 173, 0, False)
    
    # Passed parameters checking function
    check_random_state.stypy_localization = localization
    check_random_state.stypy_type_of_self = None
    check_random_state.stypy_type_store = module_type_store
    check_random_state.stypy_function_name = 'check_random_state'
    check_random_state.stypy_param_names_list = ['seed']
    check_random_state.stypy_varargs_param_name = None
    check_random_state.stypy_kwargs_param_name = None
    check_random_state.stypy_call_defaults = defaults
    check_random_state.stypy_call_varargs = varargs
    check_random_state.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'check_random_state', ['seed'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'check_random_state', localization, ['seed'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'check_random_state(...)' code ##################

    str_710485 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 181, (-1)), 'str', 'Turn seed into a np.random.RandomState instance\n\n    If seed is None (or np.random), return the RandomState singleton used\n    by np.random.\n    If seed is an int, return a new RandomState instance seeded with seed.\n    If seed is already a RandomState instance, return it.\n    Otherwise raise ValueError.\n    ')
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'seed' (line 182)
    seed_710486 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 182, 7), 'seed')
    # Getting the type of 'None' (line 182)
    None_710487 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 182, 15), 'None')
    # Applying the binary operator 'is' (line 182)
    result_is__710488 = python_operator(stypy.reporting.localization.Localization(__file__, 182, 7), 'is', seed_710486, None_710487)
    
    
    # Getting the type of 'seed' (line 182)
    seed_710489 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 182, 23), 'seed')
    # Getting the type of 'np' (line 182)
    np_710490 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 182, 31), 'np')
    # Obtaining the member 'random' of a type (line 182)
    random_710491 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 182, 31), np_710490, 'random')
    # Applying the binary operator 'is' (line 182)
    result_is__710492 = python_operator(stypy.reporting.localization.Localization(__file__, 182, 23), 'is', seed_710489, random_710491)
    
    # Applying the binary operator 'or' (line 182)
    result_or_keyword_710493 = python_operator(stypy.reporting.localization.Localization(__file__, 182, 7), 'or', result_is__710488, result_is__710492)
    
    # Testing the type of an if condition (line 182)
    if_condition_710494 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 182, 4), result_or_keyword_710493)
    # Assigning a type to the variable 'if_condition_710494' (line 182)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 182, 4), 'if_condition_710494', if_condition_710494)
    # SSA begins for if statement (line 182)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # Getting the type of 'np' (line 183)
    np_710495 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 15), 'np')
    # Obtaining the member 'random' of a type (line 183)
    random_710496 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 183, 15), np_710495, 'random')
    # Obtaining the member 'mtrand' of a type (line 183)
    mtrand_710497 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 183, 15), random_710496, 'mtrand')
    # Obtaining the member '_rand' of a type (line 183)
    _rand_710498 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 183, 15), mtrand_710497, '_rand')
    # Assigning a type to the variable 'stypy_return_type' (line 183)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 183, 8), 'stypy_return_type', _rand_710498)
    # SSA join for if statement (line 182)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Call to isinstance(...): (line 184)
    # Processing the call arguments (line 184)
    # Getting the type of 'seed' (line 184)
    seed_710500 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 18), 'seed', False)
    
    # Obtaining an instance of the builtin type 'tuple' (line 184)
    tuple_710501 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 184, 25), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 184)
    # Adding element type (line 184)
    # Getting the type of 'numbers' (line 184)
    numbers_710502 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 25), 'numbers', False)
    # Obtaining the member 'Integral' of a type (line 184)
    Integral_710503 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 184, 25), numbers_710502, 'Integral')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 184, 25), tuple_710501, Integral_710503)
    # Adding element type (line 184)
    # Getting the type of 'np' (line 184)
    np_710504 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 43), 'np', False)
    # Obtaining the member 'integer' of a type (line 184)
    integer_710505 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 184, 43), np_710504, 'integer')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 184, 25), tuple_710501, integer_710505)
    
    # Processing the call keyword arguments (line 184)
    kwargs_710506 = {}
    # Getting the type of 'isinstance' (line 184)
    isinstance_710499 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 7), 'isinstance', False)
    # Calling isinstance(args, kwargs) (line 184)
    isinstance_call_result_710507 = invoke(stypy.reporting.localization.Localization(__file__, 184, 7), isinstance_710499, *[seed_710500, tuple_710501], **kwargs_710506)
    
    # Testing the type of an if condition (line 184)
    if_condition_710508 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 184, 4), isinstance_call_result_710507)
    # Assigning a type to the variable 'if_condition_710508' (line 184)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 184, 4), 'if_condition_710508', if_condition_710508)
    # SSA begins for if statement (line 184)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to RandomState(...): (line 185)
    # Processing the call arguments (line 185)
    # Getting the type of 'seed' (line 185)
    seed_710512 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 37), 'seed', False)
    # Processing the call keyword arguments (line 185)
    kwargs_710513 = {}
    # Getting the type of 'np' (line 185)
    np_710509 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 15), 'np', False)
    # Obtaining the member 'random' of a type (line 185)
    random_710510 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 185, 15), np_710509, 'random')
    # Obtaining the member 'RandomState' of a type (line 185)
    RandomState_710511 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 185, 15), random_710510, 'RandomState')
    # Calling RandomState(args, kwargs) (line 185)
    RandomState_call_result_710514 = invoke(stypy.reporting.localization.Localization(__file__, 185, 15), RandomState_710511, *[seed_710512], **kwargs_710513)
    
    # Assigning a type to the variable 'stypy_return_type' (line 185)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 185, 8), 'stypy_return_type', RandomState_call_result_710514)
    # SSA join for if statement (line 184)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Call to isinstance(...): (line 186)
    # Processing the call arguments (line 186)
    # Getting the type of 'seed' (line 186)
    seed_710516 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 18), 'seed', False)
    # Getting the type of 'np' (line 186)
    np_710517 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 24), 'np', False)
    # Obtaining the member 'random' of a type (line 186)
    random_710518 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 186, 24), np_710517, 'random')
    # Obtaining the member 'RandomState' of a type (line 186)
    RandomState_710519 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 186, 24), random_710518, 'RandomState')
    # Processing the call keyword arguments (line 186)
    kwargs_710520 = {}
    # Getting the type of 'isinstance' (line 186)
    isinstance_710515 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 7), 'isinstance', False)
    # Calling isinstance(args, kwargs) (line 186)
    isinstance_call_result_710521 = invoke(stypy.reporting.localization.Localization(__file__, 186, 7), isinstance_710515, *[seed_710516, RandomState_710519], **kwargs_710520)
    
    # Testing the type of an if condition (line 186)
    if_condition_710522 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 186, 4), isinstance_call_result_710521)
    # Assigning a type to the variable 'if_condition_710522' (line 186)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 186, 4), 'if_condition_710522', if_condition_710522)
    # SSA begins for if statement (line 186)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # Getting the type of 'seed' (line 187)
    seed_710523 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 15), 'seed')
    # Assigning a type to the variable 'stypy_return_type' (line 187)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 187, 8), 'stypy_return_type', seed_710523)
    # SSA join for if statement (line 186)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to ValueError(...): (line 188)
    # Processing the call arguments (line 188)
    str_710525 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 188, 21), 'str', '%r cannot be used to seed a numpy.random.RandomState instance')
    # Getting the type of 'seed' (line 189)
    seed_710526 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 35), 'seed', False)
    # Applying the binary operator '%' (line 188)
    result_mod_710527 = python_operator(stypy.reporting.localization.Localization(__file__, 188, 21), '%', str_710525, seed_710526)
    
    # Processing the call keyword arguments (line 188)
    kwargs_710528 = {}
    # Getting the type of 'ValueError' (line 188)
    ValueError_710524 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 188, 10), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 188)
    ValueError_call_result_710529 = invoke(stypy.reporting.localization.Localization(__file__, 188, 10), ValueError_710524, *[result_mod_710527], **kwargs_710528)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 188, 4), ValueError_call_result_710529, 'raise parameter', BaseException)
    
    # ################# End of 'check_random_state(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'check_random_state' in the type store
    # Getting the type of 'stypy_return_type' (line 173)
    stypy_return_type_710530 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_710530)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'check_random_state'
    return stypy_return_type_710530

# Assigning a type to the variable 'check_random_state' (line 173)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 173, 0), 'check_random_state', check_random_state)

@norecursion
def _asarray_validated(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'True' (line 192)
    True_710531 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 192, 39), 'True')
    # Getting the type of 'False' (line 193)
    False_710532 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 33), 'False')
    # Getting the type of 'False' (line 193)
    False_710533 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 51), 'False')
    # Getting the type of 'False' (line 193)
    False_710534 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 66), 'False')
    # Getting the type of 'False' (line 194)
    False_710535 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 194, 34), 'False')
    defaults = [True_710531, False_710532, False_710533, False_710534, False_710535]
    # Create a new context for function '_asarray_validated'
    module_type_store = module_type_store.open_function_context('_asarray_validated', 192, 0, False)
    
    # Passed parameters checking function
    _asarray_validated.stypy_localization = localization
    _asarray_validated.stypy_type_of_self = None
    _asarray_validated.stypy_type_store = module_type_store
    _asarray_validated.stypy_function_name = '_asarray_validated'
    _asarray_validated.stypy_param_names_list = ['a', 'check_finite', 'sparse_ok', 'objects_ok', 'mask_ok', 'as_inexact']
    _asarray_validated.stypy_varargs_param_name = None
    _asarray_validated.stypy_kwargs_param_name = None
    _asarray_validated.stypy_call_defaults = defaults
    _asarray_validated.stypy_call_varargs = varargs
    _asarray_validated.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_asarray_validated', ['a', 'check_finite', 'sparse_ok', 'objects_ok', 'mask_ok', 'as_inexact'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_asarray_validated', localization, ['a', 'check_finite', 'sparse_ok', 'objects_ok', 'mask_ok', 'as_inexact'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_asarray_validated(...)' code ##################

    str_710536 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 226, (-1)), 'str', "\n    Helper function for scipy argument validation.\n\n    Many scipy linear algebra functions do support arbitrary array-like\n    input arguments.  Examples of commonly unsupported inputs include\n    matrices containing inf/nan, sparse matrix representations, and\n    matrices with complicated elements.\n\n    Parameters\n    ----------\n    a : array_like\n        The array-like input.\n    check_finite : bool, optional\n        Whether to check that the input matrices contain only finite numbers.\n        Disabling may give a performance gain, but may result in problems\n        (crashes, non-termination) if the inputs do contain infinities or NaNs.\n        Default: True\n    sparse_ok : bool, optional\n        True if scipy sparse matrices are allowed.\n    objects_ok : bool, optional\n        True if arrays with dype('O') are allowed.\n    mask_ok : bool, optional\n        True if masked arrays are allowed.\n    as_inexact : bool, optional\n        True to convert the input array to a np.inexact dtype.\n\n    Returns\n    -------\n    ret : ndarray\n        The converted validated array.\n\n    ")
    
    
    # Getting the type of 'sparse_ok' (line 227)
    sparse_ok_710537 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 11), 'sparse_ok')
    # Applying the 'not' unary operator (line 227)
    result_not__710538 = python_operator(stypy.reporting.localization.Localization(__file__, 227, 7), 'not', sparse_ok_710537)
    
    # Testing the type of an if condition (line 227)
    if_condition_710539 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 227, 4), result_not__710538)
    # Assigning a type to the variable 'if_condition_710539' (line 227)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 227, 4), 'if_condition_710539', if_condition_710539)
    # SSA begins for if statement (line 227)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 228, 8))
    
    # 'import scipy.sparse' statement (line 228)
    update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/_lib/')
    import_710540 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 228, 8), 'scipy.sparse')

    if (type(import_710540) is not StypyTypeError):

        if (import_710540 != 'pyd_module'):
            __import__(import_710540)
            sys_modules_710541 = sys.modules[import_710540]
            import_module(stypy.reporting.localization.Localization(__file__, 228, 8), 'scipy.sparse', sys_modules_710541.module_type_store, module_type_store)
        else:
            import scipy.sparse

            import_module(stypy.reporting.localization.Localization(__file__, 228, 8), 'scipy.sparse', scipy.sparse, module_type_store)

    else:
        # Assigning a type to the variable 'scipy.sparse' (line 228)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 228, 8), 'scipy.sparse', import_710540)

    remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/_lib/')
    
    
    
    # Call to issparse(...): (line 229)
    # Processing the call arguments (line 229)
    # Getting the type of 'a' (line 229)
    a_710545 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 229, 33), 'a', False)
    # Processing the call keyword arguments (line 229)
    kwargs_710546 = {}
    # Getting the type of 'scipy' (line 229)
    scipy_710542 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 229, 11), 'scipy', False)
    # Obtaining the member 'sparse' of a type (line 229)
    sparse_710543 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 229, 11), scipy_710542, 'sparse')
    # Obtaining the member 'issparse' of a type (line 229)
    issparse_710544 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 229, 11), sparse_710543, 'issparse')
    # Calling issparse(args, kwargs) (line 229)
    issparse_call_result_710547 = invoke(stypy.reporting.localization.Localization(__file__, 229, 11), issparse_710544, *[a_710545], **kwargs_710546)
    
    # Testing the type of an if condition (line 229)
    if_condition_710548 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 229, 8), issparse_call_result_710547)
    # Assigning a type to the variable 'if_condition_710548' (line 229)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 229, 8), 'if_condition_710548', if_condition_710548)
    # SSA begins for if statement (line 229)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Str to a Name (line 230):
    
    # Assigning a Str to a Name (line 230):
    str_710549 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 230, 19), 'str', 'Sparse matrices are not supported by this function. Perhaps one of the scipy.sparse.linalg functions would work instead.')
    # Assigning a type to the variable 'msg' (line 230)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 230, 12), 'msg', str_710549)
    
    # Call to ValueError(...): (line 233)
    # Processing the call arguments (line 233)
    # Getting the type of 'msg' (line 233)
    msg_710551 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 233, 29), 'msg', False)
    # Processing the call keyword arguments (line 233)
    kwargs_710552 = {}
    # Getting the type of 'ValueError' (line 233)
    ValueError_710550 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 233, 18), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 233)
    ValueError_call_result_710553 = invoke(stypy.reporting.localization.Localization(__file__, 233, 18), ValueError_710550, *[msg_710551], **kwargs_710552)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 233, 12), ValueError_call_result_710553, 'raise parameter', BaseException)
    # SSA join for if statement (line 229)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 227)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'mask_ok' (line 234)
    mask_ok_710554 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 234, 11), 'mask_ok')
    # Applying the 'not' unary operator (line 234)
    result_not__710555 = python_operator(stypy.reporting.localization.Localization(__file__, 234, 7), 'not', mask_ok_710554)
    
    # Testing the type of an if condition (line 234)
    if_condition_710556 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 234, 4), result_not__710555)
    # Assigning a type to the variable 'if_condition_710556' (line 234)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 234, 4), 'if_condition_710556', if_condition_710556)
    # SSA begins for if statement (line 234)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    
    # Call to isMaskedArray(...): (line 235)
    # Processing the call arguments (line 235)
    # Getting the type of 'a' (line 235)
    a_710560 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 31), 'a', False)
    # Processing the call keyword arguments (line 235)
    kwargs_710561 = {}
    # Getting the type of 'np' (line 235)
    np_710557 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 11), 'np', False)
    # Obtaining the member 'ma' of a type (line 235)
    ma_710558 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 235, 11), np_710557, 'ma')
    # Obtaining the member 'isMaskedArray' of a type (line 235)
    isMaskedArray_710559 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 235, 11), ma_710558, 'isMaskedArray')
    # Calling isMaskedArray(args, kwargs) (line 235)
    isMaskedArray_call_result_710562 = invoke(stypy.reporting.localization.Localization(__file__, 235, 11), isMaskedArray_710559, *[a_710560], **kwargs_710561)
    
    # Testing the type of an if condition (line 235)
    if_condition_710563 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 235, 8), isMaskedArray_call_result_710562)
    # Assigning a type to the variable 'if_condition_710563' (line 235)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 235, 8), 'if_condition_710563', if_condition_710563)
    # SSA begins for if statement (line 235)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 236)
    # Processing the call arguments (line 236)
    str_710565 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 236, 29), 'str', 'masked arrays are not supported')
    # Processing the call keyword arguments (line 236)
    kwargs_710566 = {}
    # Getting the type of 'ValueError' (line 236)
    ValueError_710564 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 236, 18), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 236)
    ValueError_call_result_710567 = invoke(stypy.reporting.localization.Localization(__file__, 236, 18), ValueError_710564, *[str_710565], **kwargs_710566)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 236, 12), ValueError_call_result_710567, 'raise parameter', BaseException)
    # SSA join for if statement (line 235)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 234)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a IfExp to a Name (line 237):
    
    # Assigning a IfExp to a Name (line 237):
    
    # Getting the type of 'check_finite' (line 237)
    check_finite_710568 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 237, 38), 'check_finite')
    # Testing the type of an if expression (line 237)
    is_suitable_condition(stypy.reporting.localization.Localization(__file__, 237, 14), check_finite_710568)
    # SSA begins for if expression (line 237)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if expression')
    # Getting the type of 'np' (line 237)
    np_710569 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 237, 14), 'np')
    # Obtaining the member 'asarray_chkfinite' of a type (line 237)
    asarray_chkfinite_710570 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 237, 14), np_710569, 'asarray_chkfinite')
    # SSA branch for the else part of an if expression (line 237)
    module_type_store.open_ssa_branch('if expression else')
    # Getting the type of 'np' (line 237)
    np_710571 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 237, 56), 'np')
    # Obtaining the member 'asarray' of a type (line 237)
    asarray_710572 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 237, 56), np_710571, 'asarray')
    # SSA join for if expression (line 237)
    module_type_store = module_type_store.join_ssa_context()
    if_exp_710573 = union_type.UnionType.add(asarray_chkfinite_710570, asarray_710572)
    
    # Assigning a type to the variable 'toarray' (line 237)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 237, 4), 'toarray', if_exp_710573)
    
    # Assigning a Call to a Name (line 238):
    
    # Assigning a Call to a Name (line 238):
    
    # Call to toarray(...): (line 238)
    # Processing the call arguments (line 238)
    # Getting the type of 'a' (line 238)
    a_710575 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 238, 16), 'a', False)
    # Processing the call keyword arguments (line 238)
    kwargs_710576 = {}
    # Getting the type of 'toarray' (line 238)
    toarray_710574 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 238, 8), 'toarray', False)
    # Calling toarray(args, kwargs) (line 238)
    toarray_call_result_710577 = invoke(stypy.reporting.localization.Localization(__file__, 238, 8), toarray_710574, *[a_710575], **kwargs_710576)
    
    # Assigning a type to the variable 'a' (line 238)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 238, 4), 'a', toarray_call_result_710577)
    
    
    # Getting the type of 'objects_ok' (line 239)
    objects_ok_710578 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 239, 11), 'objects_ok')
    # Applying the 'not' unary operator (line 239)
    result_not__710579 = python_operator(stypy.reporting.localization.Localization(__file__, 239, 7), 'not', objects_ok_710578)
    
    # Testing the type of an if condition (line 239)
    if_condition_710580 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 239, 4), result_not__710579)
    # Assigning a type to the variable 'if_condition_710580' (line 239)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 239, 4), 'if_condition_710580', if_condition_710580)
    # SSA begins for if statement (line 239)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    
    # Getting the type of 'a' (line 240)
    a_710581 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 11), 'a')
    # Obtaining the member 'dtype' of a type (line 240)
    dtype_710582 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 240, 11), a_710581, 'dtype')
    
    # Call to dtype(...): (line 240)
    # Processing the call arguments (line 240)
    str_710585 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 240, 31), 'str', 'O')
    # Processing the call keyword arguments (line 240)
    kwargs_710586 = {}
    # Getting the type of 'np' (line 240)
    np_710583 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 22), 'np', False)
    # Obtaining the member 'dtype' of a type (line 240)
    dtype_710584 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 240, 22), np_710583, 'dtype')
    # Calling dtype(args, kwargs) (line 240)
    dtype_call_result_710587 = invoke(stypy.reporting.localization.Localization(__file__, 240, 22), dtype_710584, *[str_710585], **kwargs_710586)
    
    # Applying the binary operator 'is' (line 240)
    result_is__710588 = python_operator(stypy.reporting.localization.Localization(__file__, 240, 11), 'is', dtype_710582, dtype_call_result_710587)
    
    # Testing the type of an if condition (line 240)
    if_condition_710589 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 240, 8), result_is__710588)
    # Assigning a type to the variable 'if_condition_710589' (line 240)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 240, 8), 'if_condition_710589', if_condition_710589)
    # SSA begins for if statement (line 240)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 241)
    # Processing the call arguments (line 241)
    str_710591 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 241, 29), 'str', 'object arrays are not supported')
    # Processing the call keyword arguments (line 241)
    kwargs_710592 = {}
    # Getting the type of 'ValueError' (line 241)
    ValueError_710590 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 241, 18), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 241)
    ValueError_call_result_710593 = invoke(stypy.reporting.localization.Localization(__file__, 241, 18), ValueError_710590, *[str_710591], **kwargs_710592)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 241, 12), ValueError_call_result_710593, 'raise parameter', BaseException)
    # SSA join for if statement (line 240)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 239)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Getting the type of 'as_inexact' (line 242)
    as_inexact_710594 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 242, 7), 'as_inexact')
    # Testing the type of an if condition (line 242)
    if_condition_710595 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 242, 4), as_inexact_710594)
    # Assigning a type to the variable 'if_condition_710595' (line 242)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 242, 4), 'if_condition_710595', if_condition_710595)
    # SSA begins for if statement (line 242)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    
    
    # Call to issubdtype(...): (line 243)
    # Processing the call arguments (line 243)
    # Getting the type of 'a' (line 243)
    a_710598 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 243, 29), 'a', False)
    # Obtaining the member 'dtype' of a type (line 243)
    dtype_710599 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 243, 29), a_710598, 'dtype')
    # Getting the type of 'np' (line 243)
    np_710600 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 243, 38), 'np', False)
    # Obtaining the member 'inexact' of a type (line 243)
    inexact_710601 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 243, 38), np_710600, 'inexact')
    # Processing the call keyword arguments (line 243)
    kwargs_710602 = {}
    # Getting the type of 'np' (line 243)
    np_710596 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 243, 15), 'np', False)
    # Obtaining the member 'issubdtype' of a type (line 243)
    issubdtype_710597 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 243, 15), np_710596, 'issubdtype')
    # Calling issubdtype(args, kwargs) (line 243)
    issubdtype_call_result_710603 = invoke(stypy.reporting.localization.Localization(__file__, 243, 15), issubdtype_710597, *[dtype_710599, inexact_710601], **kwargs_710602)
    
    # Applying the 'not' unary operator (line 243)
    result_not__710604 = python_operator(stypy.reporting.localization.Localization(__file__, 243, 11), 'not', issubdtype_call_result_710603)
    
    # Testing the type of an if condition (line 243)
    if_condition_710605 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 243, 8), result_not__710604)
    # Assigning a type to the variable 'if_condition_710605' (line 243)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 243, 8), 'if_condition_710605', if_condition_710605)
    # SSA begins for if statement (line 243)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 244):
    
    # Assigning a Call to a Name (line 244):
    
    # Call to toarray(...): (line 244)
    # Processing the call arguments (line 244)
    # Getting the type of 'a' (line 244)
    a_710607 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 244, 24), 'a', False)
    # Processing the call keyword arguments (line 244)
    # Getting the type of 'np' (line 244)
    np_710608 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 244, 33), 'np', False)
    # Obtaining the member 'float_' of a type (line 244)
    float__710609 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 244, 33), np_710608, 'float_')
    keyword_710610 = float__710609
    kwargs_710611 = {'dtype': keyword_710610}
    # Getting the type of 'toarray' (line 244)
    toarray_710606 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 244, 16), 'toarray', False)
    # Calling toarray(args, kwargs) (line 244)
    toarray_call_result_710612 = invoke(stypy.reporting.localization.Localization(__file__, 244, 16), toarray_710606, *[a_710607], **kwargs_710611)
    
    # Assigning a type to the variable 'a' (line 244)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 244, 12), 'a', toarray_call_result_710612)
    # SSA join for if statement (line 243)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 242)
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'a' (line 245)
    a_710613 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 245, 11), 'a')
    # Assigning a type to the variable 'stypy_return_type' (line 245)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 245, 4), 'stypy_return_type', a_710613)
    
    # ################# End of '_asarray_validated(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_asarray_validated' in the type store
    # Getting the type of 'stypy_return_type' (line 192)
    stypy_return_type_710614 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 192, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_710614)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_asarray_validated'
    return stypy_return_type_710614

# Assigning a type to the variable '_asarray_validated' (line 192)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 192, 0), '_asarray_validated', _asarray_validated)


# SSA begins for try-except statement (line 261)
module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
# Getting the type of 'inspect' (line 263)
inspect_710615 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 263, 4), 'inspect')
# Obtaining the member 'signature' of a type (line 263)
signature_710616 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 263, 4), inspect_710615, 'signature')

# Assigning a Call to a Name (line 267):

# Assigning a Call to a Name (line 267):

# Call to namedtuple(...): (line 267)
# Processing the call arguments (line 267)
str_710618 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 267, 25), 'str', 'ArgSpec')

# Obtaining an instance of the builtin type 'list' (line 267)
list_710619 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 267, 36), 'list')
# Adding type elements to the builtin type 'list' instance (line 267)
# Adding element type (line 267)
str_710620 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 267, 37), 'str', 'args')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 267, 36), list_710619, str_710620)
# Adding element type (line 267)
str_710621 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 267, 45), 'str', 'varargs')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 267, 36), list_710619, str_710621)
# Adding element type (line 267)
str_710622 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 267, 56), 'str', 'keywords')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 267, 36), list_710619, str_710622)
# Adding element type (line 267)
str_710623 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 267, 68), 'str', 'defaults')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 267, 36), list_710619, str_710623)

# Processing the call keyword arguments (line 267)
kwargs_710624 = {}
# Getting the type of 'namedtuple' (line 267)
namedtuple_710617 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 267, 14), 'namedtuple', False)
# Calling namedtuple(args, kwargs) (line 267)
namedtuple_call_result_710625 = invoke(stypy.reporting.localization.Localization(__file__, 267, 14), namedtuple_710617, *[str_710618, list_710619], **kwargs_710624)

# Assigning a type to the variable 'ArgSpec' (line 267)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 267, 4), 'ArgSpec', namedtuple_call_result_710625)

@norecursion
def getargspec_no_self(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'getargspec_no_self'
    module_type_store = module_type_store.open_function_context('getargspec_no_self', 269, 4, False)
    
    # Passed parameters checking function
    getargspec_no_self.stypy_localization = localization
    getargspec_no_self.stypy_type_of_self = None
    getargspec_no_self.stypy_type_store = module_type_store
    getargspec_no_self.stypy_function_name = 'getargspec_no_self'
    getargspec_no_self.stypy_param_names_list = ['func']
    getargspec_no_self.stypy_varargs_param_name = None
    getargspec_no_self.stypy_kwargs_param_name = None
    getargspec_no_self.stypy_call_defaults = defaults
    getargspec_no_self.stypy_call_varargs = varargs
    getargspec_no_self.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'getargspec_no_self', ['func'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'getargspec_no_self', localization, ['func'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'getargspec_no_self(...)' code ##################

    str_710626 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 289, (-1)), 'str', 'inspect.getargspec replacement using inspect.signature.\n\n        inspect.getargspec is deprecated in python 3. This is a replacement\n        based on the (new in python 3.3) `inspect.signature`.\n\n        Parameters\n        ----------\n        func : callable\n            A callable to inspect\n\n        Returns\n        -------\n        argspec : ArgSpec(args, varargs, varkw, defaults)\n            This is similar to the result of inspect.getargspec(func) under\n            python 2.x.\n            NOTE: if the first argument of `func` is self, it is *not*, I repeat\n            *not* included in argspec.args.\n            This is done for consistency between inspect.getargspec() under\n            python 2.x, and inspect.signature() under python 3.x.\n        ')
    
    # Assigning a Call to a Name (line 290):
    
    # Assigning a Call to a Name (line 290):
    
    # Call to signature(...): (line 290)
    # Processing the call arguments (line 290)
    # Getting the type of 'func' (line 290)
    func_710629 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 290, 32), 'func', False)
    # Processing the call keyword arguments (line 290)
    kwargs_710630 = {}
    # Getting the type of 'inspect' (line 290)
    inspect_710627 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 290, 14), 'inspect', False)
    # Obtaining the member 'signature' of a type (line 290)
    signature_710628 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 290, 14), inspect_710627, 'signature')
    # Calling signature(args, kwargs) (line 290)
    signature_call_result_710631 = invoke(stypy.reporting.localization.Localization(__file__, 290, 14), signature_710628, *[func_710629], **kwargs_710630)
    
    # Assigning a type to the variable 'sig' (line 290)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 290, 8), 'sig', signature_call_result_710631)
    
    # Assigning a ListComp to a Name (line 291):
    
    # Assigning a ListComp to a Name (line 291):
    # Calculating list comprehension
    # Calculating comprehension expression
    
    # Call to values(...): (line 292)
    # Processing the call keyword arguments (line 292)
    kwargs_710643 = {}
    # Getting the type of 'sig' (line 292)
    sig_710640 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 292, 28), 'sig', False)
    # Obtaining the member 'parameters' of a type (line 292)
    parameters_710641 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 292, 28), sig_710640, 'parameters')
    # Obtaining the member 'values' of a type (line 292)
    values_710642 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 292, 28), parameters_710641, 'values')
    # Calling values(args, kwargs) (line 292)
    values_call_result_710644 = invoke(stypy.reporting.localization.Localization(__file__, 292, 28), values_710642, *[], **kwargs_710643)
    
    comprehension_710645 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 292, 12), values_call_result_710644)
    # Assigning a type to the variable 'p' (line 292)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 292, 12), 'p', comprehension_710645)
    
    # Getting the type of 'p' (line 293)
    p_710634 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 293, 15), 'p')
    # Obtaining the member 'kind' of a type (line 293)
    kind_710635 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 293, 15), p_710634, 'kind')
    # Getting the type of 'inspect' (line 293)
    inspect_710636 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 293, 25), 'inspect')
    # Obtaining the member 'Parameter' of a type (line 293)
    Parameter_710637 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 293, 25), inspect_710636, 'Parameter')
    # Obtaining the member 'POSITIONAL_OR_KEYWORD' of a type (line 293)
    POSITIONAL_OR_KEYWORD_710638 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 293, 25), Parameter_710637, 'POSITIONAL_OR_KEYWORD')
    # Applying the binary operator '==' (line 293)
    result_eq_710639 = python_operator(stypy.reporting.localization.Localization(__file__, 293, 15), '==', kind_710635, POSITIONAL_OR_KEYWORD_710638)
    
    # Getting the type of 'p' (line 292)
    p_710632 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 292, 12), 'p')
    # Obtaining the member 'name' of a type (line 292)
    name_710633 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 292, 12), p_710632, 'name')
    list_710646 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 292, 12), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 292, 12), list_710646, name_710633)
    # Assigning a type to the variable 'args' (line 291)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 291, 8), 'args', list_710646)
    
    # Assigning a ListComp to a Name (line 295):
    
    # Assigning a ListComp to a Name (line 295):
    # Calculating list comprehension
    # Calculating comprehension expression
    
    # Call to values(...): (line 296)
    # Processing the call keyword arguments (line 296)
    kwargs_710658 = {}
    # Getting the type of 'sig' (line 296)
    sig_710655 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 296, 28), 'sig', False)
    # Obtaining the member 'parameters' of a type (line 296)
    parameters_710656 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 296, 28), sig_710655, 'parameters')
    # Obtaining the member 'values' of a type (line 296)
    values_710657 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 296, 28), parameters_710656, 'values')
    # Calling values(args, kwargs) (line 296)
    values_call_result_710659 = invoke(stypy.reporting.localization.Localization(__file__, 296, 28), values_710657, *[], **kwargs_710658)
    
    comprehension_710660 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 296, 12), values_call_result_710659)
    # Assigning a type to the variable 'p' (line 296)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 296, 12), 'p', comprehension_710660)
    
    # Getting the type of 'p' (line 297)
    p_710649 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 297, 15), 'p')
    # Obtaining the member 'kind' of a type (line 297)
    kind_710650 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 297, 15), p_710649, 'kind')
    # Getting the type of 'inspect' (line 297)
    inspect_710651 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 297, 25), 'inspect')
    # Obtaining the member 'Parameter' of a type (line 297)
    Parameter_710652 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 297, 25), inspect_710651, 'Parameter')
    # Obtaining the member 'VAR_POSITIONAL' of a type (line 297)
    VAR_POSITIONAL_710653 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 297, 25), Parameter_710652, 'VAR_POSITIONAL')
    # Applying the binary operator '==' (line 297)
    result_eq_710654 = python_operator(stypy.reporting.localization.Localization(__file__, 297, 15), '==', kind_710650, VAR_POSITIONAL_710653)
    
    # Getting the type of 'p' (line 296)
    p_710647 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 296, 12), 'p')
    # Obtaining the member 'name' of a type (line 296)
    name_710648 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 296, 12), p_710647, 'name')
    list_710661 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 296, 12), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 296, 12), list_710661, name_710648)
    # Assigning a type to the variable 'varargs' (line 295)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 295, 8), 'varargs', list_710661)
    
    # Assigning a IfExp to a Name (line 299):
    
    # Assigning a IfExp to a Name (line 299):
    
    # Getting the type of 'varargs' (line 299)
    varargs_710662 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 299, 32), 'varargs')
    # Testing the type of an if expression (line 299)
    is_suitable_condition(stypy.reporting.localization.Localization(__file__, 299, 18), varargs_710662)
    # SSA begins for if expression (line 299)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if expression')
    
    # Obtaining the type of the subscript
    int_710663 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 299, 26), 'int')
    # Getting the type of 'varargs' (line 299)
    varargs_710664 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 299, 18), 'varargs')
    # Obtaining the member '__getitem__' of a type (line 299)
    getitem___710665 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 299, 18), varargs_710664, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 299)
    subscript_call_result_710666 = invoke(stypy.reporting.localization.Localization(__file__, 299, 18), getitem___710665, int_710663)
    
    # SSA branch for the else part of an if expression (line 299)
    module_type_store.open_ssa_branch('if expression else')
    # Getting the type of 'None' (line 299)
    None_710667 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 299, 45), 'None')
    # SSA join for if expression (line 299)
    module_type_store = module_type_store.join_ssa_context()
    if_exp_710668 = union_type.UnionType.add(subscript_call_result_710666, None_710667)
    
    # Assigning a type to the variable 'varargs' (line 299)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 299, 8), 'varargs', if_exp_710668)
    
    # Assigning a ListComp to a Name (line 300):
    
    # Assigning a ListComp to a Name (line 300):
    # Calculating list comprehension
    # Calculating comprehension expression
    
    # Call to values(...): (line 301)
    # Processing the call keyword arguments (line 301)
    kwargs_710680 = {}
    # Getting the type of 'sig' (line 301)
    sig_710677 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 301, 28), 'sig', False)
    # Obtaining the member 'parameters' of a type (line 301)
    parameters_710678 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 301, 28), sig_710677, 'parameters')
    # Obtaining the member 'values' of a type (line 301)
    values_710679 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 301, 28), parameters_710678, 'values')
    # Calling values(args, kwargs) (line 301)
    values_call_result_710681 = invoke(stypy.reporting.localization.Localization(__file__, 301, 28), values_710679, *[], **kwargs_710680)
    
    comprehension_710682 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 301, 12), values_call_result_710681)
    # Assigning a type to the variable 'p' (line 301)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 301, 12), 'p', comprehension_710682)
    
    # Getting the type of 'p' (line 302)
    p_710671 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 302, 15), 'p')
    # Obtaining the member 'kind' of a type (line 302)
    kind_710672 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 302, 15), p_710671, 'kind')
    # Getting the type of 'inspect' (line 302)
    inspect_710673 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 302, 25), 'inspect')
    # Obtaining the member 'Parameter' of a type (line 302)
    Parameter_710674 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 302, 25), inspect_710673, 'Parameter')
    # Obtaining the member 'VAR_KEYWORD' of a type (line 302)
    VAR_KEYWORD_710675 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 302, 25), Parameter_710674, 'VAR_KEYWORD')
    # Applying the binary operator '==' (line 302)
    result_eq_710676 = python_operator(stypy.reporting.localization.Localization(__file__, 302, 15), '==', kind_710672, VAR_KEYWORD_710675)
    
    # Getting the type of 'p' (line 301)
    p_710669 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 301, 12), 'p')
    # Obtaining the member 'name' of a type (line 301)
    name_710670 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 301, 12), p_710669, 'name')
    list_710683 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 301, 12), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 301, 12), list_710683, name_710670)
    # Assigning a type to the variable 'varkw' (line 300)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 300, 8), 'varkw', list_710683)
    
    # Assigning a IfExp to a Name (line 304):
    
    # Assigning a IfExp to a Name (line 304):
    
    # Getting the type of 'varkw' (line 304)
    varkw_710684 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 304, 28), 'varkw')
    # Testing the type of an if expression (line 304)
    is_suitable_condition(stypy.reporting.localization.Localization(__file__, 304, 16), varkw_710684)
    # SSA begins for if expression (line 304)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if expression')
    
    # Obtaining the type of the subscript
    int_710685 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 304, 22), 'int')
    # Getting the type of 'varkw' (line 304)
    varkw_710686 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 304, 16), 'varkw')
    # Obtaining the member '__getitem__' of a type (line 304)
    getitem___710687 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 304, 16), varkw_710686, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 304)
    subscript_call_result_710688 = invoke(stypy.reporting.localization.Localization(__file__, 304, 16), getitem___710687, int_710685)
    
    # SSA branch for the else part of an if expression (line 304)
    module_type_store.open_ssa_branch('if expression else')
    # Getting the type of 'None' (line 304)
    None_710689 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 304, 39), 'None')
    # SSA join for if expression (line 304)
    module_type_store = module_type_store.join_ssa_context()
    if_exp_710690 = union_type.UnionType.add(subscript_call_result_710688, None_710689)
    
    # Assigning a type to the variable 'varkw' (line 304)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 304, 8), 'varkw', if_exp_710690)
    
    # Assigning a BoolOp to a Name (line 305):
    
    # Assigning a BoolOp to a Name (line 305):
    
    # Evaluating a boolean operation
    # Calculating list comprehension
    # Calculating comprehension expression
    
    # Call to values(...): (line 306)
    # Processing the call keyword arguments (line 306)
    kwargs_710708 = {}
    # Getting the type of 'sig' (line 306)
    sig_710705 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 306, 31), 'sig', False)
    # Obtaining the member 'parameters' of a type (line 306)
    parameters_710706 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 306, 31), sig_710705, 'parameters')
    # Obtaining the member 'values' of a type (line 306)
    values_710707 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 306, 31), parameters_710706, 'values')
    # Calling values(args, kwargs) (line 306)
    values_call_result_710709 = invoke(stypy.reporting.localization.Localization(__file__, 306, 31), values_710707, *[], **kwargs_710708)
    
    comprehension_710710 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 306, 12), values_call_result_710709)
    # Assigning a type to the variable 'p' (line 306)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 306, 12), 'p', comprehension_710710)
    
    # Evaluating a boolean operation
    
    # Getting the type of 'p' (line 307)
    p_710693 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 307, 16), 'p')
    # Obtaining the member 'kind' of a type (line 307)
    kind_710694 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 307, 16), p_710693, 'kind')
    # Getting the type of 'inspect' (line 307)
    inspect_710695 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 307, 26), 'inspect')
    # Obtaining the member 'Parameter' of a type (line 307)
    Parameter_710696 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 307, 26), inspect_710695, 'Parameter')
    # Obtaining the member 'POSITIONAL_OR_KEYWORD' of a type (line 307)
    POSITIONAL_OR_KEYWORD_710697 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 307, 26), Parameter_710696, 'POSITIONAL_OR_KEYWORD')
    # Applying the binary operator '==' (line 307)
    result_eq_710698 = python_operator(stypy.reporting.localization.Localization(__file__, 307, 16), '==', kind_710694, POSITIONAL_OR_KEYWORD_710697)
    
    
    # Getting the type of 'p' (line 308)
    p_710699 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 308, 15), 'p')
    # Obtaining the member 'default' of a type (line 308)
    default_710700 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 308, 15), p_710699, 'default')
    # Getting the type of 'p' (line 308)
    p_710701 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 308, 32), 'p')
    # Obtaining the member 'empty' of a type (line 308)
    empty_710702 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 308, 32), p_710701, 'empty')
    # Applying the binary operator 'isnot' (line 308)
    result_is_not_710703 = python_operator(stypy.reporting.localization.Localization(__file__, 308, 15), 'isnot', default_710700, empty_710702)
    
    # Applying the binary operator 'and' (line 307)
    result_and_keyword_710704 = python_operator(stypy.reporting.localization.Localization(__file__, 307, 16), 'and', result_eq_710698, result_is_not_710703)
    
    # Getting the type of 'p' (line 306)
    p_710691 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 306, 12), 'p')
    # Obtaining the member 'default' of a type (line 306)
    default_710692 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 306, 12), p_710691, 'default')
    list_710711 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 306, 12), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 306, 12), list_710711, default_710692)
    # Getting the type of 'None' (line 309)
    None_710712 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 309, 13), 'None')
    # Applying the binary operator 'or' (line 305)
    result_or_keyword_710713 = python_operator(stypy.reporting.localization.Localization(__file__, 305, 19), 'or', list_710711, None_710712)
    
    # Assigning a type to the variable 'defaults' (line 305)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 305, 8), 'defaults', result_or_keyword_710713)
    
    # Call to ArgSpec(...): (line 310)
    # Processing the call arguments (line 310)
    # Getting the type of 'args' (line 310)
    args_710715 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 310, 23), 'args', False)
    # Getting the type of 'varargs' (line 310)
    varargs_710716 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 310, 29), 'varargs', False)
    # Getting the type of 'varkw' (line 310)
    varkw_710717 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 310, 38), 'varkw', False)
    # Getting the type of 'defaults' (line 310)
    defaults_710718 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 310, 45), 'defaults', False)
    # Processing the call keyword arguments (line 310)
    kwargs_710719 = {}
    # Getting the type of 'ArgSpec' (line 310)
    ArgSpec_710714 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 310, 15), 'ArgSpec', False)
    # Calling ArgSpec(args, kwargs) (line 310)
    ArgSpec_call_result_710720 = invoke(stypy.reporting.localization.Localization(__file__, 310, 15), ArgSpec_710714, *[args_710715, varargs_710716, varkw_710717, defaults_710718], **kwargs_710719)
    
    # Assigning a type to the variable 'stypy_return_type' (line 310)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 310, 8), 'stypy_return_type', ArgSpec_call_result_710720)
    
    # ################# End of 'getargspec_no_self(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'getargspec_no_self' in the type store
    # Getting the type of 'stypy_return_type' (line 269)
    stypy_return_type_710721 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 269, 4), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_710721)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'getargspec_no_self'
    return stypy_return_type_710721

# Assigning a type to the variable 'getargspec_no_self' (line 269)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 269, 4), 'getargspec_no_self', getargspec_no_self)
# SSA branch for the except part of a try statement (line 261)
# SSA branch for the except 'AttributeError' branch of a try statement (line 261)
module_type_store.open_ssa_branch('except')

@norecursion
def getargspec_no_self(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'getargspec_no_self'
    module_type_store = module_type_store.open_function_context('getargspec_no_self', 314, 4, False)
    
    # Passed parameters checking function
    getargspec_no_self.stypy_localization = localization
    getargspec_no_self.stypy_type_of_self = None
    getargspec_no_self.stypy_type_store = module_type_store
    getargspec_no_self.stypy_function_name = 'getargspec_no_self'
    getargspec_no_self.stypy_param_names_list = ['func']
    getargspec_no_self.stypy_varargs_param_name = None
    getargspec_no_self.stypy_kwargs_param_name = None
    getargspec_no_self.stypy_call_defaults = defaults
    getargspec_no_self.stypy_call_varargs = varargs
    getargspec_no_self.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'getargspec_no_self', ['func'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'getargspec_no_self', localization, ['func'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'getargspec_no_self(...)' code ##################

    str_710722 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 335, (-1)), 'str', 'inspect.getargspec replacement for compatibility with python 3.x.\n\n        inspect.getargspec is deprecated in python 3. This wraps it, and\n        *removes* `self` from the argument list of `func`, if present.\n        This is done for forward compatibility with python 3.\n\n        Parameters\n        ----------\n        func : callable\n            A callable to inspect\n\n        Returns\n        -------\n        argspec : ArgSpec(args, varargs, varkw, defaults)\n            This is similar to the result of inspect.getargspec(func) under\n            python 2.x.\n            NOTE: if the first argument of `func` is self, it is *not*, I repeat\n            *not* included in argspec.args.\n            This is done for consistency between inspect.getargspec() under\n            python 2.x, and inspect.signature() under python 3.x.\n        ')
    
    # Assigning a Call to a Name (line 336):
    
    # Assigning a Call to a Name (line 336):
    
    # Call to getargspec(...): (line 336)
    # Processing the call arguments (line 336)
    # Getting the type of 'func' (line 336)
    func_710725 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 336, 37), 'func', False)
    # Processing the call keyword arguments (line 336)
    kwargs_710726 = {}
    # Getting the type of 'inspect' (line 336)
    inspect_710723 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 336, 18), 'inspect', False)
    # Obtaining the member 'getargspec' of a type (line 336)
    getargspec_710724 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 336, 18), inspect_710723, 'getargspec')
    # Calling getargspec(args, kwargs) (line 336)
    getargspec_call_result_710727 = invoke(stypy.reporting.localization.Localization(__file__, 336, 18), getargspec_710724, *[func_710725], **kwargs_710726)
    
    # Assigning a type to the variable 'argspec' (line 336)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 336, 8), 'argspec', getargspec_call_result_710727)
    
    
    
    # Obtaining the type of the subscript
    int_710728 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 337, 24), 'int')
    # Getting the type of 'argspec' (line 337)
    argspec_710729 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 337, 11), 'argspec')
    # Obtaining the member 'args' of a type (line 337)
    args_710730 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 337, 11), argspec_710729, 'args')
    # Obtaining the member '__getitem__' of a type (line 337)
    getitem___710731 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 337, 11), args_710730, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 337)
    subscript_call_result_710732 = invoke(stypy.reporting.localization.Localization(__file__, 337, 11), getitem___710731, int_710728)
    
    str_710733 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 337, 30), 'str', 'self')
    # Applying the binary operator '==' (line 337)
    result_eq_710734 = python_operator(stypy.reporting.localization.Localization(__file__, 337, 11), '==', subscript_call_result_710732, str_710733)
    
    # Testing the type of an if condition (line 337)
    if_condition_710735 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 337, 8), result_eq_710734)
    # Assigning a type to the variable 'if_condition_710735' (line 337)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 337, 8), 'if_condition_710735', if_condition_710735)
    # SSA begins for if statement (line 337)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to pop(...): (line 338)
    # Processing the call arguments (line 338)
    int_710739 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 338, 29), 'int')
    # Processing the call keyword arguments (line 338)
    kwargs_710740 = {}
    # Getting the type of 'argspec' (line 338)
    argspec_710736 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 338, 12), 'argspec', False)
    # Obtaining the member 'args' of a type (line 338)
    args_710737 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 338, 12), argspec_710736, 'args')
    # Obtaining the member 'pop' of a type (line 338)
    pop_710738 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 338, 12), args_710737, 'pop')
    # Calling pop(args, kwargs) (line 338)
    pop_call_result_710741 = invoke(stypy.reporting.localization.Localization(__file__, 338, 12), pop_710738, *[int_710739], **kwargs_710740)
    
    # SSA join for if statement (line 337)
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'argspec' (line 339)
    argspec_710742 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 339, 15), 'argspec')
    # Assigning a type to the variable 'stypy_return_type' (line 339)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 339, 8), 'stypy_return_type', argspec_710742)
    
    # ################# End of 'getargspec_no_self(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'getargspec_no_self' in the type store
    # Getting the type of 'stypy_return_type' (line 314)
    stypy_return_type_710743 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 314, 4), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_710743)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'getargspec_no_self'
    return stypy_return_type_710743

# Assigning a type to the variable 'getargspec_no_self' (line 314)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 314, 4), 'getargspec_no_self', getargspec_no_self)
# SSA join for try-except statement (line 261)
module_type_store = module_type_store.join_ssa_context()


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
