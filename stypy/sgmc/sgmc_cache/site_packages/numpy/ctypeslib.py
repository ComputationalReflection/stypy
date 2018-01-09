
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: '''
2: ============================
3: ``ctypes`` Utility Functions
4: ============================
5: 
6: See Also
7: ---------
8: load_library : Load a C library.
9: ndpointer : Array restype/argtype with verification.
10: as_ctypes : Create a ctypes array from an ndarray.
11: as_array : Create an ndarray from a ctypes array.
12: 
13: References
14: ----------
15: .. [1] "SciPy Cookbook: ctypes", http://www.scipy.org/Cookbook/Ctypes
16: 
17: Examples
18: --------
19: Load the C library:
20: 
21: >>> _lib = np.ctypeslib.load_library('libmystuff', '.')     #doctest: +SKIP
22: 
23: Our result type, an ndarray that must be of type double, be 1-dimensional
24: and is C-contiguous in memory:
25: 
26: >>> array_1d_double = np.ctypeslib.ndpointer(
27: ...                          dtype=np.double,
28: ...                          ndim=1, flags='CONTIGUOUS')    #doctest: +SKIP
29: 
30: Our C-function typically takes an array and updates its values
31: in-place.  For example::
32: 
33:     void foo_func(double* x, int length)
34:     {
35:         int i;
36:         for (i = 0; i < length; i++) {
37:             x[i] = i*i;
38:         }
39:     }
40: 
41: We wrap it using:
42: 
43: >>> _lib.foo_func.restype = None                      #doctest: +SKIP
44: >>> _lib.foo_func.argtypes = [array_1d_double, c_int] #doctest: +SKIP
45: 
46: Then, we're ready to call ``foo_func``:
47: 
48: >>> out = np.empty(15, dtype=np.double)
49: >>> _lib.foo_func(out, len(out))                #doctest: +SKIP
50: 
51: '''
52: from __future__ import division, absolute_import, print_function
53: 
54: __all__ = ['load_library', 'ndpointer', 'test', 'ctypes_load_library',
55:            'c_intp', 'as_ctypes', 'as_array']
56: 
57: import sys, os
58: from numpy import integer, ndarray, dtype as _dtype, deprecate, array
59: from numpy.core.multiarray import _flagdict, flagsobj
60: 
61: try:
62:     import ctypes
63: except ImportError:
64:     ctypes = None
65: 
66: if ctypes is None:
67:     def _dummy(*args, **kwds):
68:         '''
69:         Dummy object that raises an ImportError if ctypes is not available.
70: 
71:         Raises
72:         ------
73:         ImportError
74:             If ctypes is not available.
75: 
76:         '''
77:         raise ImportError("ctypes is not available.")
78:     ctypes_load_library = _dummy
79:     load_library = _dummy
80:     as_ctypes = _dummy
81:     as_array = _dummy
82:     from numpy import intp as c_intp
83:     _ndptr_base = object
84: else:
85:     import numpy.core._internal as nic
86:     c_intp = nic._getintp_ctype()
87:     del nic
88:     _ndptr_base = ctypes.c_void_p
89: 
90:     # Adapted from Albert Strasheim
91:     def load_library(libname, loader_path):
92:         '''
93:         It is possible to load a library using 
94:         >>> lib = ctypes.cdll[<full_path_name>]
95: 
96:         But there are cross-platform considerations, such as library file extensions,
97:         plus the fact Windows will just load the first library it finds with that name.  
98:         Numpy supplies the load_library function as a convenience.
99: 
100:         Parameters
101:         ----------
102:         libname : str
103:             Name of the library, which can have 'lib' as a prefix,
104:             but without an extension.
105:         loader_path : str
106:             Where the library can be found.
107: 
108:         Returns
109:         -------
110:         ctypes.cdll[libpath] : library object
111:            A ctypes library object 
112: 
113:         Raises
114:         ------
115:         OSError
116:             If there is no library with the expected extension, or the 
117:             library is defective and cannot be loaded.
118:         '''
119:         if ctypes.__version__ < '1.0.1':
120:             import warnings
121:             warnings.warn("All features of ctypes interface may not work " \
122:                           "with ctypes < 1.0.1")
123: 
124:         ext = os.path.splitext(libname)[1]
125:         if not ext:
126:             # Try to load library with platform-specific name, otherwise
127:             # default to libname.[so|pyd].  Sometimes, these files are built
128:             # erroneously on non-linux platforms.
129:             from numpy.distutils.misc_util import get_shared_lib_extension
130:             so_ext = get_shared_lib_extension()
131:             libname_ext = [libname + so_ext]
132:             # mac, windows and linux >= py3.2 shared library and loadable
133:             # module have different extensions so try both
134:             so_ext2 = get_shared_lib_extension(is_python_ext=True)
135:             if not so_ext2 == so_ext:
136:                 libname_ext.insert(0, libname + so_ext2)
137:         else:
138:             libname_ext = [libname]
139: 
140:         loader_path = os.path.abspath(loader_path)
141:         if not os.path.isdir(loader_path):
142:             libdir = os.path.dirname(loader_path)
143:         else:
144:             libdir = loader_path
145: 
146:         for ln in libname_ext:
147:             libpath = os.path.join(libdir, ln)
148:             if os.path.exists(libpath):
149:                 try:
150:                     return ctypes.cdll[libpath]
151:                 except OSError:
152:                     ## defective lib file
153:                     raise
154:         ## if no successful return in the libname_ext loop:
155:         raise OSError("no file with expected extension")
156: 
157:     ctypes_load_library = deprecate(load_library, 'ctypes_load_library',
158:                                     'load_library')
159: 
160: def _num_fromflags(flaglist):
161:     num = 0
162:     for val in flaglist:
163:         num += _flagdict[val]
164:     return num
165: 
166: _flagnames = ['C_CONTIGUOUS', 'F_CONTIGUOUS', 'ALIGNED', 'WRITEABLE',
167:               'OWNDATA', 'UPDATEIFCOPY']
168: def _flags_fromnum(num):
169:     res = []
170:     for key in _flagnames:
171:         value = _flagdict[key]
172:         if (num & value):
173:             res.append(key)
174:     return res
175: 
176: 
177: class _ndptr(_ndptr_base):
178: 
179:     def _check_retval_(self):
180:         '''This method is called when this class is used as the .restype
181:         asttribute for a shared-library function.   It constructs a numpy
182:         array from a void pointer.'''
183:         return array(self)
184: 
185:     @property
186:     def __array_interface__(self):
187:         return {'descr': self._dtype_.descr,
188:                 '__ref': self,
189:                 'strides': None,
190:                 'shape': self._shape_,
191:                 'version': 3,
192:                 'typestr': self._dtype_.descr[0][1],
193:                 'data': (self.value, False),
194:                 }
195: 
196:     @classmethod
197:     def from_param(cls, obj):
198:         if not isinstance(obj, ndarray):
199:             raise TypeError("argument must be an ndarray")
200:         if cls._dtype_ is not None \
201:                and obj.dtype != cls._dtype_:
202:             raise TypeError("array must have data type %s" % cls._dtype_)
203:         if cls._ndim_ is not None \
204:                and obj.ndim != cls._ndim_:
205:             raise TypeError("array must have %d dimension(s)" % cls._ndim_)
206:         if cls._shape_ is not None \
207:                and obj.shape != cls._shape_:
208:             raise TypeError("array must have shape %s" % str(cls._shape_))
209:         if cls._flags_ is not None \
210:                and ((obj.flags.num & cls._flags_) != cls._flags_):
211:             raise TypeError("array must have flags %s" %
212:                     _flags_fromnum(cls._flags_))
213:         return obj.ctypes
214: 
215: 
216: # Factory for an array-checking class with from_param defined for
217: #  use with ctypes argtypes mechanism
218: _pointer_type_cache = {}
219: def ndpointer(dtype=None, ndim=None, shape=None, flags=None):
220:     '''
221:     Array-checking restype/argtypes.
222: 
223:     An ndpointer instance is used to describe an ndarray in restypes
224:     and argtypes specifications.  This approach is more flexible than
225:     using, for example, ``POINTER(c_double)``, since several restrictions
226:     can be specified, which are verified upon calling the ctypes function.
227:     These include data type, number of dimensions, shape and flags.  If a
228:     given array does not satisfy the specified restrictions,
229:     a ``TypeError`` is raised.
230: 
231:     Parameters
232:     ----------
233:     dtype : data-type, optional
234:         Array data-type.
235:     ndim : int, optional
236:         Number of array dimensions.
237:     shape : tuple of ints, optional
238:         Array shape.
239:     flags : str or tuple of str
240:         Array flags; may be one or more of:
241: 
242:           - C_CONTIGUOUS / C / CONTIGUOUS
243:           - F_CONTIGUOUS / F / FORTRAN
244:           - OWNDATA / O
245:           - WRITEABLE / W
246:           - ALIGNED / A
247:           - UPDATEIFCOPY / U
248: 
249:     Returns
250:     -------
251:     klass : ndpointer type object
252:         A type object, which is an ``_ndtpr`` instance containing
253:         dtype, ndim, shape and flags information.
254: 
255:     Raises
256:     ------
257:     TypeError
258:         If a given array does not satisfy the specified restrictions.
259: 
260:     Examples
261:     --------
262:     >>> clib.somefunc.argtypes = [np.ctypeslib.ndpointer(dtype=np.float64,
263:     ...                                                  ndim=1,
264:     ...                                                  flags='C_CONTIGUOUS')]
265:     ... #doctest: +SKIP
266:     >>> clib.somefunc(np.array([1, 2, 3], dtype=np.float64))
267:     ... #doctest: +SKIP
268: 
269:     '''
270: 
271:     if dtype is not None:
272:         dtype = _dtype(dtype)
273:     num = None
274:     if flags is not None:
275:         if isinstance(flags, str):
276:             flags = flags.split(',')
277:         elif isinstance(flags, (int, integer)):
278:             num = flags
279:             flags = _flags_fromnum(num)
280:         elif isinstance(flags, flagsobj):
281:             num = flags.num
282:             flags = _flags_fromnum(num)
283:         if num is None:
284:             try:
285:                 flags = [x.strip().upper() for x in flags]
286:             except:
287:                 raise TypeError("invalid flags specification")
288:             num = _num_fromflags(flags)
289:     try:
290:         return _pointer_type_cache[(dtype, ndim, shape, num)]
291:     except KeyError:
292:         pass
293:     if dtype is None:
294:         name = 'any'
295:     elif dtype.names:
296:         name = str(id(dtype))
297:     else:
298:         name = dtype.str
299:     if ndim is not None:
300:         name += "_%dd" % ndim
301:     if shape is not None:
302:         try:
303:             strshape = [str(x) for x in shape]
304:         except TypeError:
305:             strshape = [str(shape)]
306:             shape = (shape,)
307:         shape = tuple(shape)
308:         name += "_"+"x".join(strshape)
309:     if flags is not None:
310:         name += "_"+"_".join(flags)
311:     else:
312:         flags = []
313:     klass = type("ndpointer_%s"%name, (_ndptr,),
314:                  {"_dtype_": dtype,
315:                   "_shape_" : shape,
316:                   "_ndim_" : ndim,
317:                   "_flags_" : num})
318:     _pointer_type_cache[dtype] = klass
319:     return klass
320: 
321: if ctypes is not None:
322:     ct = ctypes
323:     ################################################################
324:     # simple types
325: 
326:     # maps the numpy typecodes like '<f8' to simple ctypes types like
327:     # c_double. Filled in by prep_simple.
328:     _typecodes = {}
329: 
330:     def prep_simple(simple_type, dtype):
331:         '''Given a ctypes simple type, construct and attach an
332:         __array_interface__ property to it if it does not yet have one.
333:         '''
334:         try: simple_type.__array_interface__
335:         except AttributeError: pass
336:         else: return
337: 
338:         typestr = _dtype(dtype).str
339:         _typecodes[typestr] = simple_type
340: 
341:         def __array_interface__(self):
342:             return {'descr': [('', typestr)],
343:                     '__ref': self,
344:                     'strides': None,
345:                     'shape': (),
346:                     'version': 3,
347:                     'typestr': typestr,
348:                     'data': (ct.addressof(self), False),
349:                     }
350: 
351:         simple_type.__array_interface__ = property(__array_interface__)
352: 
353:     simple_types = [
354:         ((ct.c_byte, ct.c_short, ct.c_int, ct.c_long, ct.c_longlong), "i"),
355:         ((ct.c_ubyte, ct.c_ushort, ct.c_uint, ct.c_ulong, ct.c_ulonglong), "u"),
356:         ((ct.c_float, ct.c_double), "f"),
357:     ]
358: 
359:     # Prep that numerical ctypes types:
360:     for types, code in simple_types:
361:         for tp in types:
362:             prep_simple(tp, "%c%d" % (code, ct.sizeof(tp)))
363: 
364:     ################################################################
365:     # array types
366: 
367:     _ARRAY_TYPE = type(ct.c_int * 1)
368: 
369:     def prep_array(array_type):
370:         '''Given a ctypes array type, construct and attach an
371:         __array_interface__ property to it if it does not yet have one.
372:         '''
373:         try: array_type.__array_interface__
374:         except AttributeError: pass
375:         else: return
376: 
377:         shape = []
378:         ob = array_type
379:         while type(ob) is _ARRAY_TYPE:
380:             shape.append(ob._length_)
381:             ob = ob._type_
382:         shape = tuple(shape)
383:         ai = ob().__array_interface__
384:         descr = ai['descr']
385:         typestr = ai['typestr']
386: 
387:         def __array_interface__(self):
388:             return {'descr': descr,
389:                     '__ref': self,
390:                     'strides': None,
391:                     'shape': shape,
392:                     'version': 3,
393:                     'typestr': typestr,
394:                     'data': (ct.addressof(self), False),
395:                     }
396: 
397:         array_type.__array_interface__ = property(__array_interface__)
398: 
399:     def prep_pointer(pointer_obj, shape):
400:         '''Given a ctypes pointer object, construct and
401:         attach an __array_interface__ property to it if it does not
402:         yet have one.
403:         '''
404:         try: pointer_obj.__array_interface__
405:         except AttributeError: pass
406:         else: return
407: 
408:         contents = pointer_obj.contents
409:         dtype = _dtype(type(contents))
410: 
411:         inter = {'version': 3,
412:                  'typestr': dtype.str,
413:                  'data': (ct.addressof(contents), False),
414:                  'shape': shape}
415: 
416:         pointer_obj.__array_interface__ = inter
417: 
418:     ################################################################
419:     # public functions
420: 
421:     def as_array(obj, shape=None):
422:         '''Create a numpy array from a ctypes array or a ctypes POINTER.
423:         The numpy array shares the memory with the ctypes object.
424: 
425:         The size parameter must be given if converting from a ctypes POINTER.
426:         The size parameter is ignored if converting from a ctypes array
427:         '''
428:         tp = type(obj)
429:         try: tp.__array_interface__
430:         except AttributeError:
431:             if hasattr(obj, 'contents'):
432:                 prep_pointer(obj, shape)
433:             else:
434:                 prep_array(tp)
435:         return array(obj, copy=False)
436: 
437:     def as_ctypes(obj):
438:         '''Create and return a ctypes object from a numpy array.  Actually
439:         anything that exposes the __array_interface__ is accepted.'''
440:         ai = obj.__array_interface__
441:         if ai["strides"]:
442:             raise TypeError("strided arrays not supported")
443:         if ai["version"] != 3:
444:             raise TypeError("only __array_interface__ version 3 supported")
445:         addr, readonly = ai["data"]
446:         if readonly:
447:             raise TypeError("readonly arrays unsupported")
448:         tp = _typecodes[ai["typestr"]]
449:         for dim in ai["shape"][::-1]:
450:             tp = tp * dim
451:         result = tp.from_address(addr)
452:         result.__keep = ai
453:         return result
454: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_23144 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 51, (-1)), 'str', '\n============================\n``ctypes`` Utility Functions\n============================\n\nSee Also\n---------\nload_library : Load a C library.\nndpointer : Array restype/argtype with verification.\nas_ctypes : Create a ctypes array from an ndarray.\nas_array : Create an ndarray from a ctypes array.\n\nReferences\n----------\n.. [1] "SciPy Cookbook: ctypes", http://www.scipy.org/Cookbook/Ctypes\n\nExamples\n--------\nLoad the C library:\n\n>>> _lib = np.ctypeslib.load_library(\'libmystuff\', \'.\')     #doctest: +SKIP\n\nOur result type, an ndarray that must be of type double, be 1-dimensional\nand is C-contiguous in memory:\n\n>>> array_1d_double = np.ctypeslib.ndpointer(\n...                          dtype=np.double,\n...                          ndim=1, flags=\'CONTIGUOUS\')    #doctest: +SKIP\n\nOur C-function typically takes an array and updates its values\nin-place.  For example::\n\n    void foo_func(double* x, int length)\n    {\n        int i;\n        for (i = 0; i < length; i++) {\n            x[i] = i*i;\n        }\n    }\n\nWe wrap it using:\n\n>>> _lib.foo_func.restype = None                      #doctest: +SKIP\n>>> _lib.foo_func.argtypes = [array_1d_double, c_int] #doctest: +SKIP\n\nThen, we\'re ready to call ``foo_func``:\n\n>>> out = np.empty(15, dtype=np.double)\n>>> _lib.foo_func(out, len(out))                #doctest: +SKIP\n\n')

# Assigning a List to a Name (line 54):

# Assigning a List to a Name (line 54):
__all__ = ['load_library', 'ndpointer', 'test', 'ctypes_load_library', 'c_intp', 'as_ctypes', 'as_array']
module_type_store.set_exportable_members(['load_library', 'ndpointer', 'test', 'ctypes_load_library', 'c_intp', 'as_ctypes', 'as_array'])

# Obtaining an instance of the builtin type 'list' (line 54)
list_23145 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 10), 'list')
# Adding type elements to the builtin type 'list' instance (line 54)
# Adding element type (line 54)
str_23146 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 11), 'str', 'load_library')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 54, 10), list_23145, str_23146)
# Adding element type (line 54)
str_23147 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 27), 'str', 'ndpointer')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 54, 10), list_23145, str_23147)
# Adding element type (line 54)
str_23148 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 40), 'str', 'test')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 54, 10), list_23145, str_23148)
# Adding element type (line 54)
str_23149 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 48), 'str', 'ctypes_load_library')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 54, 10), list_23145, str_23149)
# Adding element type (line 54)
str_23150 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, 11), 'str', 'c_intp')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 54, 10), list_23145, str_23150)
# Adding element type (line 54)
str_23151 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, 21), 'str', 'as_ctypes')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 54, 10), list_23145, str_23151)
# Adding element type (line 54)
str_23152 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, 34), 'str', 'as_array')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 54, 10), list_23145, str_23152)

# Assigning a type to the variable '__all__' (line 54)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 54, 0), '__all__', list_23145)
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 57, 0))

# Multiple import statement. import sys (1/2) (line 57)
import sys

import_module(stypy.reporting.localization.Localization(__file__, 57, 0), 'sys', sys, module_type_store)
# Multiple import statement. import os (2/2) (line 57)
import os

import_module(stypy.reporting.localization.Localization(__file__, 57, 0), 'os', os, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 58, 0))

# 'from numpy import integer, ndarray, _dtype, deprecate, array' statement (line 58)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/')
import_23153 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 58, 0), 'numpy')

if (type(import_23153) is not StypyTypeError):

    if (import_23153 != 'pyd_module'):
        __import__(import_23153)
        sys_modules_23154 = sys.modules[import_23153]
        import_from_module(stypy.reporting.localization.Localization(__file__, 58, 0), 'numpy', sys_modules_23154.module_type_store, module_type_store, ['integer', 'ndarray', 'dtype', 'deprecate', 'array'])
        nest_module(stypy.reporting.localization.Localization(__file__, 58, 0), __file__, sys_modules_23154, sys_modules_23154.module_type_store, module_type_store)
    else:
        from numpy import integer, ndarray, dtype as _dtype, deprecate, array

        import_from_module(stypy.reporting.localization.Localization(__file__, 58, 0), 'numpy', None, module_type_store, ['integer', 'ndarray', 'dtype', 'deprecate', 'array'], [integer, ndarray, _dtype, deprecate, array])

else:
    # Assigning a type to the variable 'numpy' (line 58)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 58, 0), 'numpy', import_23153)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 59, 0))

# 'from numpy.core.multiarray import _flagdict, flagsobj' statement (line 59)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/')
import_23155 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 59, 0), 'numpy.core.multiarray')

if (type(import_23155) is not StypyTypeError):

    if (import_23155 != 'pyd_module'):
        __import__(import_23155)
        sys_modules_23156 = sys.modules[import_23155]
        import_from_module(stypy.reporting.localization.Localization(__file__, 59, 0), 'numpy.core.multiarray', sys_modules_23156.module_type_store, module_type_store, ['_flagdict', 'flagsobj'])
        nest_module(stypy.reporting.localization.Localization(__file__, 59, 0), __file__, sys_modules_23156, sys_modules_23156.module_type_store, module_type_store)
    else:
        from numpy.core.multiarray import _flagdict, flagsobj

        import_from_module(stypy.reporting.localization.Localization(__file__, 59, 0), 'numpy.core.multiarray', None, module_type_store, ['_flagdict', 'flagsobj'], [_flagdict, flagsobj])

else:
    # Assigning a type to the variable 'numpy.core.multiarray' (line 59)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 59, 0), 'numpy.core.multiarray', import_23155)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/')



# SSA begins for try-except statement (line 61)
module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 62, 4))

# 'import ctypes' statement (line 62)
import ctypes

import_module(stypy.reporting.localization.Localization(__file__, 62, 4), 'ctypes', ctypes, module_type_store)

# SSA branch for the except part of a try statement (line 61)
# SSA branch for the except 'ImportError' branch of a try statement (line 61)
module_type_store.open_ssa_branch('except')

# Assigning a Name to a Name (line 64):

# Assigning a Name to a Name (line 64):
# Getting the type of 'None' (line 64)
None_23157 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 13), 'None')
# Assigning a type to the variable 'ctypes' (line 64)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 64, 4), 'ctypes', None_23157)
# SSA join for try-except statement (line 61)
module_type_store = module_type_store.join_ssa_context()


# Type idiom detected: calculating its left and rigth part (line 66)
# Getting the type of 'ctypes' (line 66)
ctypes_23158 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 3), 'ctypes')
# Getting the type of 'None' (line 66)
None_23159 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 13), 'None')

(may_be_23160, more_types_in_union_23161) = may_be_none(ctypes_23158, None_23159)

if may_be_23160:

    if more_types_in_union_23161:
        # Runtime conditional SSA (line 66)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
    else:
        module_type_store = module_type_store


    @norecursion
    def _dummy(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_dummy'
        module_type_store = module_type_store.open_function_context('_dummy', 67, 4, False)
        
        # Passed parameters checking function
        _dummy.stypy_localization = localization
        _dummy.stypy_type_of_self = None
        _dummy.stypy_type_store = module_type_store
        _dummy.stypy_function_name = '_dummy'
        _dummy.stypy_param_names_list = []
        _dummy.stypy_varargs_param_name = 'args'
        _dummy.stypy_kwargs_param_name = 'kwds'
        _dummy.stypy_call_defaults = defaults
        _dummy.stypy_call_varargs = varargs
        _dummy.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, '_dummy', [], 'args', 'kwds', defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_dummy', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_dummy(...)' code ##################

        str_23162 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 76, (-1)), 'str', '\n        Dummy object that raises an ImportError if ctypes is not available.\n\n        Raises\n        ------\n        ImportError\n            If ctypes is not available.\n\n        ')
        
        # Call to ImportError(...): (line 77)
        # Processing the call arguments (line 77)
        str_23164 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 77, 26), 'str', 'ctypes is not available.')
        # Processing the call keyword arguments (line 77)
        kwargs_23165 = {}
        # Getting the type of 'ImportError' (line 77)
        ImportError_23163 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 14), 'ImportError', False)
        # Calling ImportError(args, kwargs) (line 77)
        ImportError_call_result_23166 = invoke(stypy.reporting.localization.Localization(__file__, 77, 14), ImportError_23163, *[str_23164], **kwargs_23165)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 77, 8), ImportError_call_result_23166, 'raise parameter', BaseException)
        
        # ################# End of '_dummy(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_dummy' in the type store
        # Getting the type of 'stypy_return_type' (line 67)
        stypy_return_type_23167 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_23167)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_dummy'
        return stypy_return_type_23167

    # Assigning a type to the variable '_dummy' (line 67)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 67, 4), '_dummy', _dummy)
    
    # Assigning a Name to a Name (line 78):
    
    # Assigning a Name to a Name (line 78):
    # Getting the type of '_dummy' (line 78)
    _dummy_23168 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 26), '_dummy')
    # Assigning a type to the variable 'ctypes_load_library' (line 78)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 78, 4), 'ctypes_load_library', _dummy_23168)
    
    # Assigning a Name to a Name (line 79):
    
    # Assigning a Name to a Name (line 79):
    # Getting the type of '_dummy' (line 79)
    _dummy_23169 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 19), '_dummy')
    # Assigning a type to the variable 'load_library' (line 79)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 79, 4), 'load_library', _dummy_23169)
    
    # Assigning a Name to a Name (line 80):
    
    # Assigning a Name to a Name (line 80):
    # Getting the type of '_dummy' (line 80)
    _dummy_23170 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 16), '_dummy')
    # Assigning a type to the variable 'as_ctypes' (line 80)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 80, 4), 'as_ctypes', _dummy_23170)
    
    # Assigning a Name to a Name (line 81):
    
    # Assigning a Name to a Name (line 81):
    # Getting the type of '_dummy' (line 81)
    _dummy_23171 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 15), '_dummy')
    # Assigning a type to the variable 'as_array' (line 81)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 81, 4), 'as_array', _dummy_23171)
    stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 82, 4))
    
    # 'from numpy import c_intp' statement (line 82)
    update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/')
    import_23172 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 82, 4), 'numpy')

    if (type(import_23172) is not StypyTypeError):

        if (import_23172 != 'pyd_module'):
            __import__(import_23172)
            sys_modules_23173 = sys.modules[import_23172]
            import_from_module(stypy.reporting.localization.Localization(__file__, 82, 4), 'numpy', sys_modules_23173.module_type_store, module_type_store, ['intp'])
            nest_module(stypy.reporting.localization.Localization(__file__, 82, 4), __file__, sys_modules_23173, sys_modules_23173.module_type_store, module_type_store)
        else:
            from numpy import intp as c_intp

            import_from_module(stypy.reporting.localization.Localization(__file__, 82, 4), 'numpy', None, module_type_store, ['intp'], [c_intp])

    else:
        # Assigning a type to the variable 'numpy' (line 82)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 82, 4), 'numpy', import_23172)

    # Adding an alias
    module_type_store.add_alias('c_intp', 'intp')
    remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/')
    
    
    # Assigning a Name to a Name (line 83):
    
    # Assigning a Name to a Name (line 83):
    # Getting the type of 'object' (line 83)
    object_23174 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 18), 'object')
    # Assigning a type to the variable '_ndptr_base' (line 83)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 83, 4), '_ndptr_base', object_23174)

    if more_types_in_union_23161:
        # Runtime conditional SSA for else branch (line 66)
        module_type_store.open_ssa_branch('idiom else')



if ((not may_be_23160) or more_types_in_union_23161):
    stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 85, 4))
    
    # 'import numpy.core._internal' statement (line 85)
    update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/')
    import_23175 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 85, 4), 'numpy.core._internal')

    if (type(import_23175) is not StypyTypeError):

        if (import_23175 != 'pyd_module'):
            __import__(import_23175)
            sys_modules_23176 = sys.modules[import_23175]
            import_module(stypy.reporting.localization.Localization(__file__, 85, 4), 'nic', sys_modules_23176.module_type_store, module_type_store)
        else:
            import numpy.core._internal as nic

            import_module(stypy.reporting.localization.Localization(__file__, 85, 4), 'nic', numpy.core._internal, module_type_store)

    else:
        # Assigning a type to the variable 'numpy.core._internal' (line 85)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 85, 4), 'numpy.core._internal', import_23175)

    remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/')
    
    
    # Assigning a Call to a Name (line 86):
    
    # Assigning a Call to a Name (line 86):
    
    # Call to _getintp_ctype(...): (line 86)
    # Processing the call keyword arguments (line 86)
    kwargs_23179 = {}
    # Getting the type of 'nic' (line 86)
    nic_23177 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 13), 'nic', False)
    # Obtaining the member '_getintp_ctype' of a type (line 86)
    _getintp_ctype_23178 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 86, 13), nic_23177, '_getintp_ctype')
    # Calling _getintp_ctype(args, kwargs) (line 86)
    _getintp_ctype_call_result_23180 = invoke(stypy.reporting.localization.Localization(__file__, 86, 13), _getintp_ctype_23178, *[], **kwargs_23179)
    
    # Assigning a type to the variable 'c_intp' (line 86)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 86, 4), 'c_intp', _getintp_ctype_call_result_23180)
    # Deleting a member
    module_type_store.del_member(stypy.reporting.localization.Localization(__file__, 87, 4), module_type_store, 'nic')
    
    # Assigning a Attribute to a Name (line 88):
    
    # Assigning a Attribute to a Name (line 88):
    # Getting the type of 'ctypes' (line 88)
    ctypes_23181 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 18), 'ctypes')
    # Obtaining the member 'c_void_p' of a type (line 88)
    c_void_p_23182 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 88, 18), ctypes_23181, 'c_void_p')
    # Assigning a type to the variable '_ndptr_base' (line 88)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 88, 4), '_ndptr_base', c_void_p_23182)

    @norecursion
    def load_library(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'load_library'
        module_type_store = module_type_store.open_function_context('load_library', 91, 4, False)
        
        # Passed parameters checking function
        load_library.stypy_localization = localization
        load_library.stypy_type_of_self = None
        load_library.stypy_type_store = module_type_store
        load_library.stypy_function_name = 'load_library'
        load_library.stypy_param_names_list = ['libname', 'loader_path']
        load_library.stypy_varargs_param_name = None
        load_library.stypy_kwargs_param_name = None
        load_library.stypy_call_defaults = defaults
        load_library.stypy_call_varargs = varargs
        load_library.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, 'load_library', ['libname', 'loader_path'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'load_library', localization, ['libname', 'loader_path'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'load_library(...)' code ##################

        str_23183 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 118, (-1)), 'str', "\n        It is possible to load a library using \n        >>> lib = ctypes.cdll[<full_path_name>]\n\n        But there are cross-platform considerations, such as library file extensions,\n        plus the fact Windows will just load the first library it finds with that name.  \n        Numpy supplies the load_library function as a convenience.\n\n        Parameters\n        ----------\n        libname : str\n            Name of the library, which can have 'lib' as a prefix,\n            but without an extension.\n        loader_path : str\n            Where the library can be found.\n\n        Returns\n        -------\n        ctypes.cdll[libpath] : library object\n           A ctypes library object \n\n        Raises\n        ------\n        OSError\n            If there is no library with the expected extension, or the \n            library is defective and cannot be loaded.\n        ")
        
        
        # Getting the type of 'ctypes' (line 119)
        ctypes_23184 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 11), 'ctypes')
        # Obtaining the member '__version__' of a type (line 119)
        version___23185 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 119, 11), ctypes_23184, '__version__')
        str_23186 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 119, 32), 'str', '1.0.1')
        # Applying the binary operator '<' (line 119)
        result_lt_23187 = python_operator(stypy.reporting.localization.Localization(__file__, 119, 11), '<', version___23185, str_23186)
        
        # Testing the type of an if condition (line 119)
        if_condition_23188 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 119, 8), result_lt_23187)
        # Assigning a type to the variable 'if_condition_23188' (line 119)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 119, 8), 'if_condition_23188', if_condition_23188)
        # SSA begins for if statement (line 119)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 120, 12))
        
        # 'import warnings' statement (line 120)
        import warnings

        import_module(stypy.reporting.localization.Localization(__file__, 120, 12), 'warnings', warnings, module_type_store)
        
        
        # Call to warn(...): (line 121)
        # Processing the call arguments (line 121)
        str_23191 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 121, 26), 'str', 'All features of ctypes interface may not work with ctypes < 1.0.1')
        # Processing the call keyword arguments (line 121)
        kwargs_23192 = {}
        # Getting the type of 'warnings' (line 121)
        warnings_23189 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 12), 'warnings', False)
        # Obtaining the member 'warn' of a type (line 121)
        warn_23190 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 121, 12), warnings_23189, 'warn')
        # Calling warn(args, kwargs) (line 121)
        warn_call_result_23193 = invoke(stypy.reporting.localization.Localization(__file__, 121, 12), warn_23190, *[str_23191], **kwargs_23192)
        
        # SSA join for if statement (line 119)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Subscript to a Name (line 124):
        
        # Assigning a Subscript to a Name (line 124):
        
        # Obtaining the type of the subscript
        int_23194 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 124, 40), 'int')
        
        # Call to splitext(...): (line 124)
        # Processing the call arguments (line 124)
        # Getting the type of 'libname' (line 124)
        libname_23198 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 31), 'libname', False)
        # Processing the call keyword arguments (line 124)
        kwargs_23199 = {}
        # Getting the type of 'os' (line 124)
        os_23195 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 14), 'os', False)
        # Obtaining the member 'path' of a type (line 124)
        path_23196 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 124, 14), os_23195, 'path')
        # Obtaining the member 'splitext' of a type (line 124)
        splitext_23197 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 124, 14), path_23196, 'splitext')
        # Calling splitext(args, kwargs) (line 124)
        splitext_call_result_23200 = invoke(stypy.reporting.localization.Localization(__file__, 124, 14), splitext_23197, *[libname_23198], **kwargs_23199)
        
        # Obtaining the member '__getitem__' of a type (line 124)
        getitem___23201 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 124, 14), splitext_call_result_23200, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 124)
        subscript_call_result_23202 = invoke(stypy.reporting.localization.Localization(__file__, 124, 14), getitem___23201, int_23194)
        
        # Assigning a type to the variable 'ext' (line 124)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 124, 8), 'ext', subscript_call_result_23202)
        
        
        # Getting the type of 'ext' (line 125)
        ext_23203 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 15), 'ext')
        # Applying the 'not' unary operator (line 125)
        result_not__23204 = python_operator(stypy.reporting.localization.Localization(__file__, 125, 11), 'not', ext_23203)
        
        # Testing the type of an if condition (line 125)
        if_condition_23205 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 125, 8), result_not__23204)
        # Assigning a type to the variable 'if_condition_23205' (line 125)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 125, 8), 'if_condition_23205', if_condition_23205)
        # SSA begins for if statement (line 125)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 129, 12))
        
        # 'from numpy.distutils.misc_util import get_shared_lib_extension' statement (line 129)
        update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/')
        import_23206 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 129, 12), 'numpy.distutils.misc_util')

        if (type(import_23206) is not StypyTypeError):

            if (import_23206 != 'pyd_module'):
                __import__(import_23206)
                sys_modules_23207 = sys.modules[import_23206]
                import_from_module(stypy.reporting.localization.Localization(__file__, 129, 12), 'numpy.distutils.misc_util', sys_modules_23207.module_type_store, module_type_store, ['get_shared_lib_extension'])
                nest_module(stypy.reporting.localization.Localization(__file__, 129, 12), __file__, sys_modules_23207, sys_modules_23207.module_type_store, module_type_store)
            else:
                from numpy.distutils.misc_util import get_shared_lib_extension

                import_from_module(stypy.reporting.localization.Localization(__file__, 129, 12), 'numpy.distutils.misc_util', None, module_type_store, ['get_shared_lib_extension'], [get_shared_lib_extension])

        else:
            # Assigning a type to the variable 'numpy.distutils.misc_util' (line 129)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 129, 12), 'numpy.distutils.misc_util', import_23206)

        remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/')
        
        
        # Assigning a Call to a Name (line 130):
        
        # Assigning a Call to a Name (line 130):
        
        # Call to get_shared_lib_extension(...): (line 130)
        # Processing the call keyword arguments (line 130)
        kwargs_23209 = {}
        # Getting the type of 'get_shared_lib_extension' (line 130)
        get_shared_lib_extension_23208 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 21), 'get_shared_lib_extension', False)
        # Calling get_shared_lib_extension(args, kwargs) (line 130)
        get_shared_lib_extension_call_result_23210 = invoke(stypy.reporting.localization.Localization(__file__, 130, 21), get_shared_lib_extension_23208, *[], **kwargs_23209)
        
        # Assigning a type to the variable 'so_ext' (line 130)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 130, 12), 'so_ext', get_shared_lib_extension_call_result_23210)
        
        # Assigning a List to a Name (line 131):
        
        # Assigning a List to a Name (line 131):
        
        # Obtaining an instance of the builtin type 'list' (line 131)
        list_23211 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 131, 26), 'list')
        # Adding type elements to the builtin type 'list' instance (line 131)
        # Adding element type (line 131)
        # Getting the type of 'libname' (line 131)
        libname_23212 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 27), 'libname')
        # Getting the type of 'so_ext' (line 131)
        so_ext_23213 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 37), 'so_ext')
        # Applying the binary operator '+' (line 131)
        result_add_23214 = python_operator(stypy.reporting.localization.Localization(__file__, 131, 27), '+', libname_23212, so_ext_23213)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 131, 26), list_23211, result_add_23214)
        
        # Assigning a type to the variable 'libname_ext' (line 131)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 131, 12), 'libname_ext', list_23211)
        
        # Assigning a Call to a Name (line 134):
        
        # Assigning a Call to a Name (line 134):
        
        # Call to get_shared_lib_extension(...): (line 134)
        # Processing the call keyword arguments (line 134)
        # Getting the type of 'True' (line 134)
        True_23216 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 61), 'True', False)
        keyword_23217 = True_23216
        kwargs_23218 = {'is_python_ext': keyword_23217}
        # Getting the type of 'get_shared_lib_extension' (line 134)
        get_shared_lib_extension_23215 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 22), 'get_shared_lib_extension', False)
        # Calling get_shared_lib_extension(args, kwargs) (line 134)
        get_shared_lib_extension_call_result_23219 = invoke(stypy.reporting.localization.Localization(__file__, 134, 22), get_shared_lib_extension_23215, *[], **kwargs_23218)
        
        # Assigning a type to the variable 'so_ext2' (line 134)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 134, 12), 'so_ext2', get_shared_lib_extension_call_result_23219)
        
        
        
        # Getting the type of 'so_ext2' (line 135)
        so_ext2_23220 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 19), 'so_ext2')
        # Getting the type of 'so_ext' (line 135)
        so_ext_23221 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 30), 'so_ext')
        # Applying the binary operator '==' (line 135)
        result_eq_23222 = python_operator(stypy.reporting.localization.Localization(__file__, 135, 19), '==', so_ext2_23220, so_ext_23221)
        
        # Applying the 'not' unary operator (line 135)
        result_not__23223 = python_operator(stypy.reporting.localization.Localization(__file__, 135, 15), 'not', result_eq_23222)
        
        # Testing the type of an if condition (line 135)
        if_condition_23224 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 135, 12), result_not__23223)
        # Assigning a type to the variable 'if_condition_23224' (line 135)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 135, 12), 'if_condition_23224', if_condition_23224)
        # SSA begins for if statement (line 135)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to insert(...): (line 136)
        # Processing the call arguments (line 136)
        int_23227 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 136, 35), 'int')
        # Getting the type of 'libname' (line 136)
        libname_23228 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 38), 'libname', False)
        # Getting the type of 'so_ext2' (line 136)
        so_ext2_23229 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 48), 'so_ext2', False)
        # Applying the binary operator '+' (line 136)
        result_add_23230 = python_operator(stypy.reporting.localization.Localization(__file__, 136, 38), '+', libname_23228, so_ext2_23229)
        
        # Processing the call keyword arguments (line 136)
        kwargs_23231 = {}
        # Getting the type of 'libname_ext' (line 136)
        libname_ext_23225 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 16), 'libname_ext', False)
        # Obtaining the member 'insert' of a type (line 136)
        insert_23226 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 136, 16), libname_ext_23225, 'insert')
        # Calling insert(args, kwargs) (line 136)
        insert_call_result_23232 = invoke(stypy.reporting.localization.Localization(__file__, 136, 16), insert_23226, *[int_23227, result_add_23230], **kwargs_23231)
        
        # SSA join for if statement (line 135)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA branch for the else part of an if statement (line 125)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a List to a Name (line 138):
        
        # Assigning a List to a Name (line 138):
        
        # Obtaining an instance of the builtin type 'list' (line 138)
        list_23233 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 138, 26), 'list')
        # Adding type elements to the builtin type 'list' instance (line 138)
        # Adding element type (line 138)
        # Getting the type of 'libname' (line 138)
        libname_23234 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 27), 'libname')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 138, 26), list_23233, libname_23234)
        
        # Assigning a type to the variable 'libname_ext' (line 138)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 138, 12), 'libname_ext', list_23233)
        # SSA join for if statement (line 125)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Name (line 140):
        
        # Assigning a Call to a Name (line 140):
        
        # Call to abspath(...): (line 140)
        # Processing the call arguments (line 140)
        # Getting the type of 'loader_path' (line 140)
        loader_path_23238 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 38), 'loader_path', False)
        # Processing the call keyword arguments (line 140)
        kwargs_23239 = {}
        # Getting the type of 'os' (line 140)
        os_23235 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 22), 'os', False)
        # Obtaining the member 'path' of a type (line 140)
        path_23236 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 140, 22), os_23235, 'path')
        # Obtaining the member 'abspath' of a type (line 140)
        abspath_23237 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 140, 22), path_23236, 'abspath')
        # Calling abspath(args, kwargs) (line 140)
        abspath_call_result_23240 = invoke(stypy.reporting.localization.Localization(__file__, 140, 22), abspath_23237, *[loader_path_23238], **kwargs_23239)
        
        # Assigning a type to the variable 'loader_path' (line 140)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 140, 8), 'loader_path', abspath_call_result_23240)
        
        
        
        # Call to isdir(...): (line 141)
        # Processing the call arguments (line 141)
        # Getting the type of 'loader_path' (line 141)
        loader_path_23244 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 29), 'loader_path', False)
        # Processing the call keyword arguments (line 141)
        kwargs_23245 = {}
        # Getting the type of 'os' (line 141)
        os_23241 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 15), 'os', False)
        # Obtaining the member 'path' of a type (line 141)
        path_23242 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 141, 15), os_23241, 'path')
        # Obtaining the member 'isdir' of a type (line 141)
        isdir_23243 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 141, 15), path_23242, 'isdir')
        # Calling isdir(args, kwargs) (line 141)
        isdir_call_result_23246 = invoke(stypy.reporting.localization.Localization(__file__, 141, 15), isdir_23243, *[loader_path_23244], **kwargs_23245)
        
        # Applying the 'not' unary operator (line 141)
        result_not__23247 = python_operator(stypy.reporting.localization.Localization(__file__, 141, 11), 'not', isdir_call_result_23246)
        
        # Testing the type of an if condition (line 141)
        if_condition_23248 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 141, 8), result_not__23247)
        # Assigning a type to the variable 'if_condition_23248' (line 141)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 141, 8), 'if_condition_23248', if_condition_23248)
        # SSA begins for if statement (line 141)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 142):
        
        # Assigning a Call to a Name (line 142):
        
        # Call to dirname(...): (line 142)
        # Processing the call arguments (line 142)
        # Getting the type of 'loader_path' (line 142)
        loader_path_23252 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 37), 'loader_path', False)
        # Processing the call keyword arguments (line 142)
        kwargs_23253 = {}
        # Getting the type of 'os' (line 142)
        os_23249 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 21), 'os', False)
        # Obtaining the member 'path' of a type (line 142)
        path_23250 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 142, 21), os_23249, 'path')
        # Obtaining the member 'dirname' of a type (line 142)
        dirname_23251 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 142, 21), path_23250, 'dirname')
        # Calling dirname(args, kwargs) (line 142)
        dirname_call_result_23254 = invoke(stypy.reporting.localization.Localization(__file__, 142, 21), dirname_23251, *[loader_path_23252], **kwargs_23253)
        
        # Assigning a type to the variable 'libdir' (line 142)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 142, 12), 'libdir', dirname_call_result_23254)
        # SSA branch for the else part of an if statement (line 141)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Name to a Name (line 144):
        
        # Assigning a Name to a Name (line 144):
        # Getting the type of 'loader_path' (line 144)
        loader_path_23255 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 21), 'loader_path')
        # Assigning a type to the variable 'libdir' (line 144)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 144, 12), 'libdir', loader_path_23255)
        # SSA join for if statement (line 141)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Getting the type of 'libname_ext' (line 146)
        libname_ext_23256 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 18), 'libname_ext')
        # Testing the type of a for loop iterable (line 146)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 146, 8), libname_ext_23256)
        # Getting the type of the for loop variable (line 146)
        for_loop_var_23257 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 146, 8), libname_ext_23256)
        # Assigning a type to the variable 'ln' (line 146)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 146, 8), 'ln', for_loop_var_23257)
        # SSA begins for a for statement (line 146)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Call to a Name (line 147):
        
        # Assigning a Call to a Name (line 147):
        
        # Call to join(...): (line 147)
        # Processing the call arguments (line 147)
        # Getting the type of 'libdir' (line 147)
        libdir_23261 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 35), 'libdir', False)
        # Getting the type of 'ln' (line 147)
        ln_23262 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 43), 'ln', False)
        # Processing the call keyword arguments (line 147)
        kwargs_23263 = {}
        # Getting the type of 'os' (line 147)
        os_23258 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 22), 'os', False)
        # Obtaining the member 'path' of a type (line 147)
        path_23259 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 147, 22), os_23258, 'path')
        # Obtaining the member 'join' of a type (line 147)
        join_23260 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 147, 22), path_23259, 'join')
        # Calling join(args, kwargs) (line 147)
        join_call_result_23264 = invoke(stypy.reporting.localization.Localization(__file__, 147, 22), join_23260, *[libdir_23261, ln_23262], **kwargs_23263)
        
        # Assigning a type to the variable 'libpath' (line 147)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 147, 12), 'libpath', join_call_result_23264)
        
        
        # Call to exists(...): (line 148)
        # Processing the call arguments (line 148)
        # Getting the type of 'libpath' (line 148)
        libpath_23268 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 30), 'libpath', False)
        # Processing the call keyword arguments (line 148)
        kwargs_23269 = {}
        # Getting the type of 'os' (line 148)
        os_23265 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 15), 'os', False)
        # Obtaining the member 'path' of a type (line 148)
        path_23266 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 148, 15), os_23265, 'path')
        # Obtaining the member 'exists' of a type (line 148)
        exists_23267 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 148, 15), path_23266, 'exists')
        # Calling exists(args, kwargs) (line 148)
        exists_call_result_23270 = invoke(stypy.reporting.localization.Localization(__file__, 148, 15), exists_23267, *[libpath_23268], **kwargs_23269)
        
        # Testing the type of an if condition (line 148)
        if_condition_23271 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 148, 12), exists_call_result_23270)
        # Assigning a type to the variable 'if_condition_23271' (line 148)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 148, 12), 'if_condition_23271', if_condition_23271)
        # SSA begins for if statement (line 148)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        
        # SSA begins for try-except statement (line 149)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
        
        # Obtaining the type of the subscript
        # Getting the type of 'libpath' (line 150)
        libpath_23272 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 39), 'libpath')
        # Getting the type of 'ctypes' (line 150)
        ctypes_23273 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 27), 'ctypes')
        # Obtaining the member 'cdll' of a type (line 150)
        cdll_23274 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 150, 27), ctypes_23273, 'cdll')
        # Obtaining the member '__getitem__' of a type (line 150)
        getitem___23275 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 150, 27), cdll_23274, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 150)
        subscript_call_result_23276 = invoke(stypy.reporting.localization.Localization(__file__, 150, 27), getitem___23275, libpath_23272)
        
        # Assigning a type to the variable 'stypy_return_type' (line 150)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 150, 20), 'stypy_return_type', subscript_call_result_23276)
        # SSA branch for the except part of a try statement (line 149)
        # SSA branch for the except 'OSError' branch of a try statement (line 149)
        module_type_store.open_ssa_branch('except')
        # SSA join for try-except statement (line 149)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 148)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to OSError(...): (line 155)
        # Processing the call arguments (line 155)
        str_23278 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 155, 22), 'str', 'no file with expected extension')
        # Processing the call keyword arguments (line 155)
        kwargs_23279 = {}
        # Getting the type of 'OSError' (line 155)
        OSError_23277 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 14), 'OSError', False)
        # Calling OSError(args, kwargs) (line 155)
        OSError_call_result_23280 = invoke(stypy.reporting.localization.Localization(__file__, 155, 14), OSError_23277, *[str_23278], **kwargs_23279)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 155, 8), OSError_call_result_23280, 'raise parameter', BaseException)
        
        # ################# End of 'load_library(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'load_library' in the type store
        # Getting the type of 'stypy_return_type' (line 91)
        stypy_return_type_23281 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_23281)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'load_library'
        return stypy_return_type_23281

    # Assigning a type to the variable 'load_library' (line 91)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 91, 4), 'load_library', load_library)
    
    # Assigning a Call to a Name (line 157):
    
    # Assigning a Call to a Name (line 157):
    
    # Call to deprecate(...): (line 157)
    # Processing the call arguments (line 157)
    # Getting the type of 'load_library' (line 157)
    load_library_23283 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 36), 'load_library', False)
    str_23284 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 157, 50), 'str', 'ctypes_load_library')
    str_23285 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 158, 36), 'str', 'load_library')
    # Processing the call keyword arguments (line 157)
    kwargs_23286 = {}
    # Getting the type of 'deprecate' (line 157)
    deprecate_23282 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 26), 'deprecate', False)
    # Calling deprecate(args, kwargs) (line 157)
    deprecate_call_result_23287 = invoke(stypy.reporting.localization.Localization(__file__, 157, 26), deprecate_23282, *[load_library_23283, str_23284, str_23285], **kwargs_23286)
    
    # Assigning a type to the variable 'ctypes_load_library' (line 157)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 157, 4), 'ctypes_load_library', deprecate_call_result_23287)

    if (may_be_23160 and more_types_in_union_23161):
        # SSA join for if statement (line 66)
        module_type_store = module_type_store.join_ssa_context()




@norecursion
def _num_fromflags(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_num_fromflags'
    module_type_store = module_type_store.open_function_context('_num_fromflags', 160, 0, False)
    
    # Passed parameters checking function
    _num_fromflags.stypy_localization = localization
    _num_fromflags.stypy_type_of_self = None
    _num_fromflags.stypy_type_store = module_type_store
    _num_fromflags.stypy_function_name = '_num_fromflags'
    _num_fromflags.stypy_param_names_list = ['flaglist']
    _num_fromflags.stypy_varargs_param_name = None
    _num_fromflags.stypy_kwargs_param_name = None
    _num_fromflags.stypy_call_defaults = defaults
    _num_fromflags.stypy_call_varargs = varargs
    _num_fromflags.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_num_fromflags', ['flaglist'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_num_fromflags', localization, ['flaglist'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_num_fromflags(...)' code ##################

    
    # Assigning a Num to a Name (line 161):
    
    # Assigning a Num to a Name (line 161):
    int_23288 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 161, 10), 'int')
    # Assigning a type to the variable 'num' (line 161)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 161, 4), 'num', int_23288)
    
    # Getting the type of 'flaglist' (line 162)
    flaglist_23289 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 15), 'flaglist')
    # Testing the type of a for loop iterable (line 162)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 162, 4), flaglist_23289)
    # Getting the type of the for loop variable (line 162)
    for_loop_var_23290 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 162, 4), flaglist_23289)
    # Assigning a type to the variable 'val' (line 162)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 162, 4), 'val', for_loop_var_23290)
    # SSA begins for a for statement (line 162)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Getting the type of 'num' (line 163)
    num_23291 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 8), 'num')
    
    # Obtaining the type of the subscript
    # Getting the type of 'val' (line 163)
    val_23292 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 25), 'val')
    # Getting the type of '_flagdict' (line 163)
    _flagdict_23293 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 15), '_flagdict')
    # Obtaining the member '__getitem__' of a type (line 163)
    getitem___23294 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 163, 15), _flagdict_23293, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 163)
    subscript_call_result_23295 = invoke(stypy.reporting.localization.Localization(__file__, 163, 15), getitem___23294, val_23292)
    
    # Applying the binary operator '+=' (line 163)
    result_iadd_23296 = python_operator(stypy.reporting.localization.Localization(__file__, 163, 8), '+=', num_23291, subscript_call_result_23295)
    # Assigning a type to the variable 'num' (line 163)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 163, 8), 'num', result_iadd_23296)
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'num' (line 164)
    num_23297 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 11), 'num')
    # Assigning a type to the variable 'stypy_return_type' (line 164)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 164, 4), 'stypy_return_type', num_23297)
    
    # ################# End of '_num_fromflags(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_num_fromflags' in the type store
    # Getting the type of 'stypy_return_type' (line 160)
    stypy_return_type_23298 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_23298)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_num_fromflags'
    return stypy_return_type_23298

# Assigning a type to the variable '_num_fromflags' (line 160)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 160, 0), '_num_fromflags', _num_fromflags)

# Assigning a List to a Name (line 166):

# Assigning a List to a Name (line 166):

# Obtaining an instance of the builtin type 'list' (line 166)
list_23299 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 166, 13), 'list')
# Adding type elements to the builtin type 'list' instance (line 166)
# Adding element type (line 166)
str_23300 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 166, 14), 'str', 'C_CONTIGUOUS')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 166, 13), list_23299, str_23300)
# Adding element type (line 166)
str_23301 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 166, 30), 'str', 'F_CONTIGUOUS')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 166, 13), list_23299, str_23301)
# Adding element type (line 166)
str_23302 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 166, 46), 'str', 'ALIGNED')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 166, 13), list_23299, str_23302)
# Adding element type (line 166)
str_23303 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 166, 57), 'str', 'WRITEABLE')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 166, 13), list_23299, str_23303)
# Adding element type (line 166)
str_23304 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 167, 14), 'str', 'OWNDATA')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 166, 13), list_23299, str_23304)
# Adding element type (line 166)
str_23305 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 167, 25), 'str', 'UPDATEIFCOPY')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 166, 13), list_23299, str_23305)

# Assigning a type to the variable '_flagnames' (line 166)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 166, 0), '_flagnames', list_23299)

@norecursion
def _flags_fromnum(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_flags_fromnum'
    module_type_store = module_type_store.open_function_context('_flags_fromnum', 168, 0, False)
    
    # Passed parameters checking function
    _flags_fromnum.stypy_localization = localization
    _flags_fromnum.stypy_type_of_self = None
    _flags_fromnum.stypy_type_store = module_type_store
    _flags_fromnum.stypy_function_name = '_flags_fromnum'
    _flags_fromnum.stypy_param_names_list = ['num']
    _flags_fromnum.stypy_varargs_param_name = None
    _flags_fromnum.stypy_kwargs_param_name = None
    _flags_fromnum.stypy_call_defaults = defaults
    _flags_fromnum.stypy_call_varargs = varargs
    _flags_fromnum.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_flags_fromnum', ['num'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_flags_fromnum', localization, ['num'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_flags_fromnum(...)' code ##################

    
    # Assigning a List to a Name (line 169):
    
    # Assigning a List to a Name (line 169):
    
    # Obtaining an instance of the builtin type 'list' (line 169)
    list_23306 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 169, 10), 'list')
    # Adding type elements to the builtin type 'list' instance (line 169)
    
    # Assigning a type to the variable 'res' (line 169)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 169, 4), 'res', list_23306)
    
    # Getting the type of '_flagnames' (line 170)
    _flagnames_23307 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 15), '_flagnames')
    # Testing the type of a for loop iterable (line 170)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 170, 4), _flagnames_23307)
    # Getting the type of the for loop variable (line 170)
    for_loop_var_23308 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 170, 4), _flagnames_23307)
    # Assigning a type to the variable 'key' (line 170)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 170, 4), 'key', for_loop_var_23308)
    # SSA begins for a for statement (line 170)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Subscript to a Name (line 171):
    
    # Assigning a Subscript to a Name (line 171):
    
    # Obtaining the type of the subscript
    # Getting the type of 'key' (line 171)
    key_23309 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 26), 'key')
    # Getting the type of '_flagdict' (line 171)
    _flagdict_23310 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 16), '_flagdict')
    # Obtaining the member '__getitem__' of a type (line 171)
    getitem___23311 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 171, 16), _flagdict_23310, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 171)
    subscript_call_result_23312 = invoke(stypy.reporting.localization.Localization(__file__, 171, 16), getitem___23311, key_23309)
    
    # Assigning a type to the variable 'value' (line 171)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 171, 8), 'value', subscript_call_result_23312)
    
    # Getting the type of 'num' (line 172)
    num_23313 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 12), 'num')
    # Getting the type of 'value' (line 172)
    value_23314 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 18), 'value')
    # Applying the binary operator '&' (line 172)
    result_and__23315 = python_operator(stypy.reporting.localization.Localization(__file__, 172, 12), '&', num_23313, value_23314)
    
    # Testing the type of an if condition (line 172)
    if_condition_23316 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 172, 8), result_and__23315)
    # Assigning a type to the variable 'if_condition_23316' (line 172)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 172, 8), 'if_condition_23316', if_condition_23316)
    # SSA begins for if statement (line 172)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to append(...): (line 173)
    # Processing the call arguments (line 173)
    # Getting the type of 'key' (line 173)
    key_23319 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 23), 'key', False)
    # Processing the call keyword arguments (line 173)
    kwargs_23320 = {}
    # Getting the type of 'res' (line 173)
    res_23317 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 12), 'res', False)
    # Obtaining the member 'append' of a type (line 173)
    append_23318 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 173, 12), res_23317, 'append')
    # Calling append(args, kwargs) (line 173)
    append_call_result_23321 = invoke(stypy.reporting.localization.Localization(__file__, 173, 12), append_23318, *[key_23319], **kwargs_23320)
    
    # SSA join for if statement (line 172)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'res' (line 174)
    res_23322 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 11), 'res')
    # Assigning a type to the variable 'stypy_return_type' (line 174)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 174, 4), 'stypy_return_type', res_23322)
    
    # ################# End of '_flags_fromnum(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_flags_fromnum' in the type store
    # Getting the type of 'stypy_return_type' (line 168)
    stypy_return_type_23323 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_23323)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_flags_fromnum'
    return stypy_return_type_23323

# Assigning a type to the variable '_flags_fromnum' (line 168)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 168, 0), '_flags_fromnum', _flags_fromnum)
# Declaration of the '_ndptr' class
# Getting the type of '_ndptr_base' (line 177)
_ndptr_base_23324 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 13), '_ndptr_base')

class _ndptr(_ndptr_base_23324, ):

    @norecursion
    def _check_retval_(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_check_retval_'
        module_type_store = module_type_store.open_function_context('_check_retval_', 179, 4, False)
        # Assigning a type to the variable 'self' (line 180)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 180, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        _ndptr._check_retval_.__dict__.__setitem__('stypy_localization', localization)
        _ndptr._check_retval_.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        _ndptr._check_retval_.__dict__.__setitem__('stypy_type_store', module_type_store)
        _ndptr._check_retval_.__dict__.__setitem__('stypy_function_name', '_ndptr._check_retval_')
        _ndptr._check_retval_.__dict__.__setitem__('stypy_param_names_list', [])
        _ndptr._check_retval_.__dict__.__setitem__('stypy_varargs_param_name', None)
        _ndptr._check_retval_.__dict__.__setitem__('stypy_kwargs_param_name', None)
        _ndptr._check_retval_.__dict__.__setitem__('stypy_call_defaults', defaults)
        _ndptr._check_retval_.__dict__.__setitem__('stypy_call_varargs', varargs)
        _ndptr._check_retval_.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        _ndptr._check_retval_.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, '_ndptr._check_retval_', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_check_retval_', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_check_retval_(...)' code ##################

        str_23325 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 182, (-1)), 'str', 'This method is called when this class is used as the .restype\n        asttribute for a shared-library function.   It constructs a numpy\n        array from a void pointer.')
        
        # Call to array(...): (line 183)
        # Processing the call arguments (line 183)
        # Getting the type of 'self' (line 183)
        self_23327 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 21), 'self', False)
        # Processing the call keyword arguments (line 183)
        kwargs_23328 = {}
        # Getting the type of 'array' (line 183)
        array_23326 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 15), 'array', False)
        # Calling array(args, kwargs) (line 183)
        array_call_result_23329 = invoke(stypy.reporting.localization.Localization(__file__, 183, 15), array_23326, *[self_23327], **kwargs_23328)
        
        # Assigning a type to the variable 'stypy_return_type' (line 183)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 183, 8), 'stypy_return_type', array_call_result_23329)
        
        # ################# End of '_check_retval_(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_check_retval_' in the type store
        # Getting the type of 'stypy_return_type' (line 179)
        stypy_return_type_23330 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_23330)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_check_retval_'
        return stypy_return_type_23330


    @norecursion
    def __array_interface__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__array_interface__'
        module_type_store = module_type_store.open_function_context('__array_interface__', 185, 4, False)
        # Assigning a type to the variable 'self' (line 186)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 186, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        _ndptr.__array_interface__.__dict__.__setitem__('stypy_localization', localization)
        _ndptr.__array_interface__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        _ndptr.__array_interface__.__dict__.__setitem__('stypy_type_store', module_type_store)
        _ndptr.__array_interface__.__dict__.__setitem__('stypy_function_name', '_ndptr.__array_interface__')
        _ndptr.__array_interface__.__dict__.__setitem__('stypy_param_names_list', [])
        _ndptr.__array_interface__.__dict__.__setitem__('stypy_varargs_param_name', None)
        _ndptr.__array_interface__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        _ndptr.__array_interface__.__dict__.__setitem__('stypy_call_defaults', defaults)
        _ndptr.__array_interface__.__dict__.__setitem__('stypy_call_varargs', varargs)
        _ndptr.__array_interface__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        _ndptr.__array_interface__.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, '_ndptr.__array_interface__', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__array_interface__', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__array_interface__(...)' code ##################

        
        # Obtaining an instance of the builtin type 'dict' (line 187)
        dict_23331 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 187, 15), 'dict')
        # Adding type elements to the builtin type 'dict' instance (line 187)
        # Adding element type (key, value) (line 187)
        str_23332 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 187, 16), 'str', 'descr')
        # Getting the type of 'self' (line 187)
        self_23333 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 25), 'self')
        # Obtaining the member '_dtype_' of a type (line 187)
        _dtype__23334 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 187, 25), self_23333, '_dtype_')
        # Obtaining the member 'descr' of a type (line 187)
        descr_23335 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 187, 25), _dtype__23334, 'descr')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 187, 15), dict_23331, (str_23332, descr_23335))
        # Adding element type (key, value) (line 187)
        str_23336 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 188, 16), 'str', '__ref')
        # Getting the type of 'self' (line 188)
        self_23337 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 188, 25), 'self')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 187, 15), dict_23331, (str_23336, self_23337))
        # Adding element type (key, value) (line 187)
        str_23338 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 189, 16), 'str', 'strides')
        # Getting the type of 'None' (line 189)
        None_23339 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 27), 'None')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 187, 15), dict_23331, (str_23338, None_23339))
        # Adding element type (key, value) (line 187)
        str_23340 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 190, 16), 'str', 'shape')
        # Getting the type of 'self' (line 190)
        self_23341 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 190, 25), 'self')
        # Obtaining the member '_shape_' of a type (line 190)
        _shape__23342 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 190, 25), self_23341, '_shape_')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 187, 15), dict_23331, (str_23340, _shape__23342))
        # Adding element type (key, value) (line 187)
        str_23343 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 191, 16), 'str', 'version')
        int_23344 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 191, 27), 'int')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 187, 15), dict_23331, (str_23343, int_23344))
        # Adding element type (key, value) (line 187)
        str_23345 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 192, 16), 'str', 'typestr')
        
        # Obtaining the type of the subscript
        int_23346 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 192, 49), 'int')
        
        # Obtaining the type of the subscript
        int_23347 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 192, 46), 'int')
        # Getting the type of 'self' (line 192)
        self_23348 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 192, 27), 'self')
        # Obtaining the member '_dtype_' of a type (line 192)
        _dtype__23349 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 192, 27), self_23348, '_dtype_')
        # Obtaining the member 'descr' of a type (line 192)
        descr_23350 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 192, 27), _dtype__23349, 'descr')
        # Obtaining the member '__getitem__' of a type (line 192)
        getitem___23351 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 192, 27), descr_23350, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 192)
        subscript_call_result_23352 = invoke(stypy.reporting.localization.Localization(__file__, 192, 27), getitem___23351, int_23347)
        
        # Obtaining the member '__getitem__' of a type (line 192)
        getitem___23353 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 192, 27), subscript_call_result_23352, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 192)
        subscript_call_result_23354 = invoke(stypy.reporting.localization.Localization(__file__, 192, 27), getitem___23353, int_23346)
        
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 187, 15), dict_23331, (str_23345, subscript_call_result_23354))
        # Adding element type (key, value) (line 187)
        str_23355 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 193, 16), 'str', 'data')
        
        # Obtaining an instance of the builtin type 'tuple' (line 193)
        tuple_23356 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 193, 25), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 193)
        # Adding element type (line 193)
        # Getting the type of 'self' (line 193)
        self_23357 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 25), 'self')
        # Obtaining the member 'value' of a type (line 193)
        value_23358 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 193, 25), self_23357, 'value')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 193, 25), tuple_23356, value_23358)
        # Adding element type (line 193)
        # Getting the type of 'False' (line 193)
        False_23359 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 37), 'False')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 193, 25), tuple_23356, False_23359)
        
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 187, 15), dict_23331, (str_23355, tuple_23356))
        
        # Assigning a type to the variable 'stypy_return_type' (line 187)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 187, 8), 'stypy_return_type', dict_23331)
        
        # ################# End of '__array_interface__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__array_interface__' in the type store
        # Getting the type of 'stypy_return_type' (line 185)
        stypy_return_type_23360 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_23360)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__array_interface__'
        return stypy_return_type_23360


    @norecursion
    def from_param(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'from_param'
        module_type_store = module_type_store.open_function_context('from_param', 196, 4, False)
        # Assigning a type to the variable 'self' (line 197)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 197, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        _ndptr.from_param.__dict__.__setitem__('stypy_localization', localization)
        _ndptr.from_param.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        _ndptr.from_param.__dict__.__setitem__('stypy_type_store', module_type_store)
        _ndptr.from_param.__dict__.__setitem__('stypy_function_name', '_ndptr.from_param')
        _ndptr.from_param.__dict__.__setitem__('stypy_param_names_list', ['obj'])
        _ndptr.from_param.__dict__.__setitem__('stypy_varargs_param_name', None)
        _ndptr.from_param.__dict__.__setitem__('stypy_kwargs_param_name', None)
        _ndptr.from_param.__dict__.__setitem__('stypy_call_defaults', defaults)
        _ndptr.from_param.__dict__.__setitem__('stypy_call_varargs', varargs)
        _ndptr.from_param.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        _ndptr.from_param.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, '_ndptr.from_param', ['obj'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'from_param', localization, ['obj'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'from_param(...)' code ##################

        
        
        
        # Call to isinstance(...): (line 198)
        # Processing the call arguments (line 198)
        # Getting the type of 'obj' (line 198)
        obj_23362 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 26), 'obj', False)
        # Getting the type of 'ndarray' (line 198)
        ndarray_23363 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 31), 'ndarray', False)
        # Processing the call keyword arguments (line 198)
        kwargs_23364 = {}
        # Getting the type of 'isinstance' (line 198)
        isinstance_23361 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 15), 'isinstance', False)
        # Calling isinstance(args, kwargs) (line 198)
        isinstance_call_result_23365 = invoke(stypy.reporting.localization.Localization(__file__, 198, 15), isinstance_23361, *[obj_23362, ndarray_23363], **kwargs_23364)
        
        # Applying the 'not' unary operator (line 198)
        result_not__23366 = python_operator(stypy.reporting.localization.Localization(__file__, 198, 11), 'not', isinstance_call_result_23365)
        
        # Testing the type of an if condition (line 198)
        if_condition_23367 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 198, 8), result_not__23366)
        # Assigning a type to the variable 'if_condition_23367' (line 198)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 198, 8), 'if_condition_23367', if_condition_23367)
        # SSA begins for if statement (line 198)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to TypeError(...): (line 199)
        # Processing the call arguments (line 199)
        str_23369 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 199, 28), 'str', 'argument must be an ndarray')
        # Processing the call keyword arguments (line 199)
        kwargs_23370 = {}
        # Getting the type of 'TypeError' (line 199)
        TypeError_23368 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 18), 'TypeError', False)
        # Calling TypeError(args, kwargs) (line 199)
        TypeError_call_result_23371 = invoke(stypy.reporting.localization.Localization(__file__, 199, 18), TypeError_23368, *[str_23369], **kwargs_23370)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 199, 12), TypeError_call_result_23371, 'raise parameter', BaseException)
        # SSA join for if statement (line 198)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Evaluating a boolean operation
        
        # Getting the type of 'cls' (line 200)
        cls_23372 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 11), 'cls')
        # Obtaining the member '_dtype_' of a type (line 200)
        _dtype__23373 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 200, 11), cls_23372, '_dtype_')
        # Getting the type of 'None' (line 200)
        None_23374 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 30), 'None')
        # Applying the binary operator 'isnot' (line 200)
        result_is_not_23375 = python_operator(stypy.reporting.localization.Localization(__file__, 200, 11), 'isnot', _dtype__23373, None_23374)
        
        
        # Getting the type of 'obj' (line 201)
        obj_23376 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 19), 'obj')
        # Obtaining the member 'dtype' of a type (line 201)
        dtype_23377 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 201, 19), obj_23376, 'dtype')
        # Getting the type of 'cls' (line 201)
        cls_23378 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 32), 'cls')
        # Obtaining the member '_dtype_' of a type (line 201)
        _dtype__23379 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 201, 32), cls_23378, '_dtype_')
        # Applying the binary operator '!=' (line 201)
        result_ne_23380 = python_operator(stypy.reporting.localization.Localization(__file__, 201, 19), '!=', dtype_23377, _dtype__23379)
        
        # Applying the binary operator 'and' (line 200)
        result_and_keyword_23381 = python_operator(stypy.reporting.localization.Localization(__file__, 200, 11), 'and', result_is_not_23375, result_ne_23380)
        
        # Testing the type of an if condition (line 200)
        if_condition_23382 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 200, 8), result_and_keyword_23381)
        # Assigning a type to the variable 'if_condition_23382' (line 200)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 200, 8), 'if_condition_23382', if_condition_23382)
        # SSA begins for if statement (line 200)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to TypeError(...): (line 202)
        # Processing the call arguments (line 202)
        str_23384 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 202, 28), 'str', 'array must have data type %s')
        # Getting the type of 'cls' (line 202)
        cls_23385 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 61), 'cls', False)
        # Obtaining the member '_dtype_' of a type (line 202)
        _dtype__23386 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 202, 61), cls_23385, '_dtype_')
        # Applying the binary operator '%' (line 202)
        result_mod_23387 = python_operator(stypy.reporting.localization.Localization(__file__, 202, 28), '%', str_23384, _dtype__23386)
        
        # Processing the call keyword arguments (line 202)
        kwargs_23388 = {}
        # Getting the type of 'TypeError' (line 202)
        TypeError_23383 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 18), 'TypeError', False)
        # Calling TypeError(args, kwargs) (line 202)
        TypeError_call_result_23389 = invoke(stypy.reporting.localization.Localization(__file__, 202, 18), TypeError_23383, *[result_mod_23387], **kwargs_23388)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 202, 12), TypeError_call_result_23389, 'raise parameter', BaseException)
        # SSA join for if statement (line 200)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Evaluating a boolean operation
        
        # Getting the type of 'cls' (line 203)
        cls_23390 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 11), 'cls')
        # Obtaining the member '_ndim_' of a type (line 203)
        _ndim__23391 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 203, 11), cls_23390, '_ndim_')
        # Getting the type of 'None' (line 203)
        None_23392 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 29), 'None')
        # Applying the binary operator 'isnot' (line 203)
        result_is_not_23393 = python_operator(stypy.reporting.localization.Localization(__file__, 203, 11), 'isnot', _ndim__23391, None_23392)
        
        
        # Getting the type of 'obj' (line 204)
        obj_23394 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 204, 19), 'obj')
        # Obtaining the member 'ndim' of a type (line 204)
        ndim_23395 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 204, 19), obj_23394, 'ndim')
        # Getting the type of 'cls' (line 204)
        cls_23396 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 204, 31), 'cls')
        # Obtaining the member '_ndim_' of a type (line 204)
        _ndim__23397 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 204, 31), cls_23396, '_ndim_')
        # Applying the binary operator '!=' (line 204)
        result_ne_23398 = python_operator(stypy.reporting.localization.Localization(__file__, 204, 19), '!=', ndim_23395, _ndim__23397)
        
        # Applying the binary operator 'and' (line 203)
        result_and_keyword_23399 = python_operator(stypy.reporting.localization.Localization(__file__, 203, 11), 'and', result_is_not_23393, result_ne_23398)
        
        # Testing the type of an if condition (line 203)
        if_condition_23400 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 203, 8), result_and_keyword_23399)
        # Assigning a type to the variable 'if_condition_23400' (line 203)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 203, 8), 'if_condition_23400', if_condition_23400)
        # SSA begins for if statement (line 203)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to TypeError(...): (line 205)
        # Processing the call arguments (line 205)
        str_23402 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 205, 28), 'str', 'array must have %d dimension(s)')
        # Getting the type of 'cls' (line 205)
        cls_23403 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 64), 'cls', False)
        # Obtaining the member '_ndim_' of a type (line 205)
        _ndim__23404 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 205, 64), cls_23403, '_ndim_')
        # Applying the binary operator '%' (line 205)
        result_mod_23405 = python_operator(stypy.reporting.localization.Localization(__file__, 205, 28), '%', str_23402, _ndim__23404)
        
        # Processing the call keyword arguments (line 205)
        kwargs_23406 = {}
        # Getting the type of 'TypeError' (line 205)
        TypeError_23401 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 18), 'TypeError', False)
        # Calling TypeError(args, kwargs) (line 205)
        TypeError_call_result_23407 = invoke(stypy.reporting.localization.Localization(__file__, 205, 18), TypeError_23401, *[result_mod_23405], **kwargs_23406)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 205, 12), TypeError_call_result_23407, 'raise parameter', BaseException)
        # SSA join for if statement (line 203)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Evaluating a boolean operation
        
        # Getting the type of 'cls' (line 206)
        cls_23408 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 206, 11), 'cls')
        # Obtaining the member '_shape_' of a type (line 206)
        _shape__23409 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 206, 11), cls_23408, '_shape_')
        # Getting the type of 'None' (line 206)
        None_23410 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 206, 30), 'None')
        # Applying the binary operator 'isnot' (line 206)
        result_is_not_23411 = python_operator(stypy.reporting.localization.Localization(__file__, 206, 11), 'isnot', _shape__23409, None_23410)
        
        
        # Getting the type of 'obj' (line 207)
        obj_23412 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 207, 19), 'obj')
        # Obtaining the member 'shape' of a type (line 207)
        shape_23413 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 207, 19), obj_23412, 'shape')
        # Getting the type of 'cls' (line 207)
        cls_23414 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 207, 32), 'cls')
        # Obtaining the member '_shape_' of a type (line 207)
        _shape__23415 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 207, 32), cls_23414, '_shape_')
        # Applying the binary operator '!=' (line 207)
        result_ne_23416 = python_operator(stypy.reporting.localization.Localization(__file__, 207, 19), '!=', shape_23413, _shape__23415)
        
        # Applying the binary operator 'and' (line 206)
        result_and_keyword_23417 = python_operator(stypy.reporting.localization.Localization(__file__, 206, 11), 'and', result_is_not_23411, result_ne_23416)
        
        # Testing the type of an if condition (line 206)
        if_condition_23418 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 206, 8), result_and_keyword_23417)
        # Assigning a type to the variable 'if_condition_23418' (line 206)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 206, 8), 'if_condition_23418', if_condition_23418)
        # SSA begins for if statement (line 206)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to TypeError(...): (line 208)
        # Processing the call arguments (line 208)
        str_23420 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 208, 28), 'str', 'array must have shape %s')
        
        # Call to str(...): (line 208)
        # Processing the call arguments (line 208)
        # Getting the type of 'cls' (line 208)
        cls_23422 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 208, 61), 'cls', False)
        # Obtaining the member '_shape_' of a type (line 208)
        _shape__23423 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 208, 61), cls_23422, '_shape_')
        # Processing the call keyword arguments (line 208)
        kwargs_23424 = {}
        # Getting the type of 'str' (line 208)
        str_23421 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 208, 57), 'str', False)
        # Calling str(args, kwargs) (line 208)
        str_call_result_23425 = invoke(stypy.reporting.localization.Localization(__file__, 208, 57), str_23421, *[_shape__23423], **kwargs_23424)
        
        # Applying the binary operator '%' (line 208)
        result_mod_23426 = python_operator(stypy.reporting.localization.Localization(__file__, 208, 28), '%', str_23420, str_call_result_23425)
        
        # Processing the call keyword arguments (line 208)
        kwargs_23427 = {}
        # Getting the type of 'TypeError' (line 208)
        TypeError_23419 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 208, 18), 'TypeError', False)
        # Calling TypeError(args, kwargs) (line 208)
        TypeError_call_result_23428 = invoke(stypy.reporting.localization.Localization(__file__, 208, 18), TypeError_23419, *[result_mod_23426], **kwargs_23427)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 208, 12), TypeError_call_result_23428, 'raise parameter', BaseException)
        # SSA join for if statement (line 206)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Evaluating a boolean operation
        
        # Getting the type of 'cls' (line 209)
        cls_23429 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 11), 'cls')
        # Obtaining the member '_flags_' of a type (line 209)
        _flags__23430 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 209, 11), cls_23429, '_flags_')
        # Getting the type of 'None' (line 209)
        None_23431 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 30), 'None')
        # Applying the binary operator 'isnot' (line 209)
        result_is_not_23432 = python_operator(stypy.reporting.localization.Localization(__file__, 209, 11), 'isnot', _flags__23430, None_23431)
        
        
        # Getting the type of 'obj' (line 210)
        obj_23433 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 21), 'obj')
        # Obtaining the member 'flags' of a type (line 210)
        flags_23434 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 210, 21), obj_23433, 'flags')
        # Obtaining the member 'num' of a type (line 210)
        num_23435 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 210, 21), flags_23434, 'num')
        # Getting the type of 'cls' (line 210)
        cls_23436 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 37), 'cls')
        # Obtaining the member '_flags_' of a type (line 210)
        _flags__23437 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 210, 37), cls_23436, '_flags_')
        # Applying the binary operator '&' (line 210)
        result_and__23438 = python_operator(stypy.reporting.localization.Localization(__file__, 210, 21), '&', num_23435, _flags__23437)
        
        # Getting the type of 'cls' (line 210)
        cls_23439 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 53), 'cls')
        # Obtaining the member '_flags_' of a type (line 210)
        _flags__23440 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 210, 53), cls_23439, '_flags_')
        # Applying the binary operator '!=' (line 210)
        result_ne_23441 = python_operator(stypy.reporting.localization.Localization(__file__, 210, 20), '!=', result_and__23438, _flags__23440)
        
        # Applying the binary operator 'and' (line 209)
        result_and_keyword_23442 = python_operator(stypy.reporting.localization.Localization(__file__, 209, 11), 'and', result_is_not_23432, result_ne_23441)
        
        # Testing the type of an if condition (line 209)
        if_condition_23443 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 209, 8), result_and_keyword_23442)
        # Assigning a type to the variable 'if_condition_23443' (line 209)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 209, 8), 'if_condition_23443', if_condition_23443)
        # SSA begins for if statement (line 209)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to TypeError(...): (line 211)
        # Processing the call arguments (line 211)
        str_23445 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 211, 28), 'str', 'array must have flags %s')
        
        # Call to _flags_fromnum(...): (line 212)
        # Processing the call arguments (line 212)
        # Getting the type of 'cls' (line 212)
        cls_23447 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 35), 'cls', False)
        # Obtaining the member '_flags_' of a type (line 212)
        _flags__23448 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 212, 35), cls_23447, '_flags_')
        # Processing the call keyword arguments (line 212)
        kwargs_23449 = {}
        # Getting the type of '_flags_fromnum' (line 212)
        _flags_fromnum_23446 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 20), '_flags_fromnum', False)
        # Calling _flags_fromnum(args, kwargs) (line 212)
        _flags_fromnum_call_result_23450 = invoke(stypy.reporting.localization.Localization(__file__, 212, 20), _flags_fromnum_23446, *[_flags__23448], **kwargs_23449)
        
        # Applying the binary operator '%' (line 211)
        result_mod_23451 = python_operator(stypy.reporting.localization.Localization(__file__, 211, 28), '%', str_23445, _flags_fromnum_call_result_23450)
        
        # Processing the call keyword arguments (line 211)
        kwargs_23452 = {}
        # Getting the type of 'TypeError' (line 211)
        TypeError_23444 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 211, 18), 'TypeError', False)
        # Calling TypeError(args, kwargs) (line 211)
        TypeError_call_result_23453 = invoke(stypy.reporting.localization.Localization(__file__, 211, 18), TypeError_23444, *[result_mod_23451], **kwargs_23452)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 211, 12), TypeError_call_result_23453, 'raise parameter', BaseException)
        # SSA join for if statement (line 209)
        module_type_store = module_type_store.join_ssa_context()
        
        # Getting the type of 'obj' (line 213)
        obj_23454 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 15), 'obj')
        # Obtaining the member 'ctypes' of a type (line 213)
        ctypes_23455 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 213, 15), obj_23454, 'ctypes')
        # Assigning a type to the variable 'stypy_return_type' (line 213)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 213, 8), 'stypy_return_type', ctypes_23455)
        
        # ################# End of 'from_param(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'from_param' in the type store
        # Getting the type of 'stypy_return_type' (line 196)
        stypy_return_type_23456 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 196, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_23456)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'from_param'
        return stypy_return_type_23456


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 177, 0, False)
        # Assigning a type to the variable 'self' (line 178)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 178, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, '_ndptr.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable '_ndptr' (line 177)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 177, 0), '_ndptr', _ndptr)

# Assigning a Dict to a Name (line 218):

# Assigning a Dict to a Name (line 218):

# Obtaining an instance of the builtin type 'dict' (line 218)
dict_23457 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 218, 22), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 218)

# Assigning a type to the variable '_pointer_type_cache' (line 218)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 218, 0), '_pointer_type_cache', dict_23457)

@norecursion
def ndpointer(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'None' (line 219)
    None_23458 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 219, 20), 'None')
    # Getting the type of 'None' (line 219)
    None_23459 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 219, 31), 'None')
    # Getting the type of 'None' (line 219)
    None_23460 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 219, 43), 'None')
    # Getting the type of 'None' (line 219)
    None_23461 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 219, 55), 'None')
    defaults = [None_23458, None_23459, None_23460, None_23461]
    # Create a new context for function 'ndpointer'
    module_type_store = module_type_store.open_function_context('ndpointer', 219, 0, False)
    
    # Passed parameters checking function
    ndpointer.stypy_localization = localization
    ndpointer.stypy_type_of_self = None
    ndpointer.stypy_type_store = module_type_store
    ndpointer.stypy_function_name = 'ndpointer'
    ndpointer.stypy_param_names_list = ['dtype', 'ndim', 'shape', 'flags']
    ndpointer.stypy_varargs_param_name = None
    ndpointer.stypy_kwargs_param_name = None
    ndpointer.stypy_call_defaults = defaults
    ndpointer.stypy_call_varargs = varargs
    ndpointer.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'ndpointer', ['dtype', 'ndim', 'shape', 'flags'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'ndpointer', localization, ['dtype', 'ndim', 'shape', 'flags'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'ndpointer(...)' code ##################

    str_23462 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 269, (-1)), 'str', "\n    Array-checking restype/argtypes.\n\n    An ndpointer instance is used to describe an ndarray in restypes\n    and argtypes specifications.  This approach is more flexible than\n    using, for example, ``POINTER(c_double)``, since several restrictions\n    can be specified, which are verified upon calling the ctypes function.\n    These include data type, number of dimensions, shape and flags.  If a\n    given array does not satisfy the specified restrictions,\n    a ``TypeError`` is raised.\n\n    Parameters\n    ----------\n    dtype : data-type, optional\n        Array data-type.\n    ndim : int, optional\n        Number of array dimensions.\n    shape : tuple of ints, optional\n        Array shape.\n    flags : str or tuple of str\n        Array flags; may be one or more of:\n\n          - C_CONTIGUOUS / C / CONTIGUOUS\n          - F_CONTIGUOUS / F / FORTRAN\n          - OWNDATA / O\n          - WRITEABLE / W\n          - ALIGNED / A\n          - UPDATEIFCOPY / U\n\n    Returns\n    -------\n    klass : ndpointer type object\n        A type object, which is an ``_ndtpr`` instance containing\n        dtype, ndim, shape and flags information.\n\n    Raises\n    ------\n    TypeError\n        If a given array does not satisfy the specified restrictions.\n\n    Examples\n    --------\n    >>> clib.somefunc.argtypes = [np.ctypeslib.ndpointer(dtype=np.float64,\n    ...                                                  ndim=1,\n    ...                                                  flags='C_CONTIGUOUS')]\n    ... #doctest: +SKIP\n    >>> clib.somefunc(np.array([1, 2, 3], dtype=np.float64))\n    ... #doctest: +SKIP\n\n    ")
    
    # Type idiom detected: calculating its left and rigth part (line 271)
    # Getting the type of 'dtype' (line 271)
    dtype_23463 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 271, 4), 'dtype')
    # Getting the type of 'None' (line 271)
    None_23464 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 271, 20), 'None')
    
    (may_be_23465, more_types_in_union_23466) = may_not_be_none(dtype_23463, None_23464)

    if may_be_23465:

        if more_types_in_union_23466:
            # Runtime conditional SSA (line 271)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a Call to a Name (line 272):
        
        # Assigning a Call to a Name (line 272):
        
        # Call to _dtype(...): (line 272)
        # Processing the call arguments (line 272)
        # Getting the type of 'dtype' (line 272)
        dtype_23468 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 272, 23), 'dtype', False)
        # Processing the call keyword arguments (line 272)
        kwargs_23469 = {}
        # Getting the type of '_dtype' (line 272)
        _dtype_23467 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 272, 16), '_dtype', False)
        # Calling _dtype(args, kwargs) (line 272)
        _dtype_call_result_23470 = invoke(stypy.reporting.localization.Localization(__file__, 272, 16), _dtype_23467, *[dtype_23468], **kwargs_23469)
        
        # Assigning a type to the variable 'dtype' (line 272)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 272, 8), 'dtype', _dtype_call_result_23470)

        if more_types_in_union_23466:
            # SSA join for if statement (line 271)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Assigning a Name to a Name (line 273):
    
    # Assigning a Name to a Name (line 273):
    # Getting the type of 'None' (line 273)
    None_23471 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 273, 10), 'None')
    # Assigning a type to the variable 'num' (line 273)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 273, 4), 'num', None_23471)
    
    # Type idiom detected: calculating its left and rigth part (line 274)
    # Getting the type of 'flags' (line 274)
    flags_23472 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 274, 4), 'flags')
    # Getting the type of 'None' (line 274)
    None_23473 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 274, 20), 'None')
    
    (may_be_23474, more_types_in_union_23475) = may_not_be_none(flags_23472, None_23473)

    if may_be_23474:

        if more_types_in_union_23475:
            # Runtime conditional SSA (line 274)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Type idiom detected: calculating its left and rigth part (line 275)
        # Getting the type of 'str' (line 275)
        str_23476 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 275, 29), 'str')
        # Getting the type of 'flags' (line 275)
        flags_23477 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 275, 22), 'flags')
        
        (may_be_23478, more_types_in_union_23479) = may_be_subtype(str_23476, flags_23477)

        if may_be_23478:

            if more_types_in_union_23479:
                # Runtime conditional SSA (line 275)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            # Assigning a type to the variable 'flags' (line 275)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 275, 8), 'flags', remove_not_subtype_from_union(flags_23477, str))
            
            # Assigning a Call to a Name (line 276):
            
            # Assigning a Call to a Name (line 276):
            
            # Call to split(...): (line 276)
            # Processing the call arguments (line 276)
            str_23482 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 276, 32), 'str', ',')
            # Processing the call keyword arguments (line 276)
            kwargs_23483 = {}
            # Getting the type of 'flags' (line 276)
            flags_23480 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 276, 20), 'flags', False)
            # Obtaining the member 'split' of a type (line 276)
            split_23481 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 276, 20), flags_23480, 'split')
            # Calling split(args, kwargs) (line 276)
            split_call_result_23484 = invoke(stypy.reporting.localization.Localization(__file__, 276, 20), split_23481, *[str_23482], **kwargs_23483)
            
            # Assigning a type to the variable 'flags' (line 276)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 276, 12), 'flags', split_call_result_23484)

            if more_types_in_union_23479:
                # Runtime conditional SSA for else branch (line 275)
                module_type_store.open_ssa_branch('idiom else')



        if ((not may_be_23478) or more_types_in_union_23479):
            # Assigning a type to the variable 'flags' (line 275)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 275, 8), 'flags', remove_subtype_from_union(flags_23477, str))
            
            
            # Call to isinstance(...): (line 277)
            # Processing the call arguments (line 277)
            # Getting the type of 'flags' (line 277)
            flags_23486 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 277, 24), 'flags', False)
            
            # Obtaining an instance of the builtin type 'tuple' (line 277)
            tuple_23487 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 277, 32), 'tuple')
            # Adding type elements to the builtin type 'tuple' instance (line 277)
            # Adding element type (line 277)
            # Getting the type of 'int' (line 277)
            int_23488 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 277, 32), 'int', False)
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 277, 32), tuple_23487, int_23488)
            # Adding element type (line 277)
            # Getting the type of 'integer' (line 277)
            integer_23489 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 277, 37), 'integer', False)
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 277, 32), tuple_23487, integer_23489)
            
            # Processing the call keyword arguments (line 277)
            kwargs_23490 = {}
            # Getting the type of 'isinstance' (line 277)
            isinstance_23485 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 277, 13), 'isinstance', False)
            # Calling isinstance(args, kwargs) (line 277)
            isinstance_call_result_23491 = invoke(stypy.reporting.localization.Localization(__file__, 277, 13), isinstance_23485, *[flags_23486, tuple_23487], **kwargs_23490)
            
            # Testing the type of an if condition (line 277)
            if_condition_23492 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 277, 13), isinstance_call_result_23491)
            # Assigning a type to the variable 'if_condition_23492' (line 277)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 277, 13), 'if_condition_23492', if_condition_23492)
            # SSA begins for if statement (line 277)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a Name to a Name (line 278):
            
            # Assigning a Name to a Name (line 278):
            # Getting the type of 'flags' (line 278)
            flags_23493 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 278, 18), 'flags')
            # Assigning a type to the variable 'num' (line 278)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 278, 12), 'num', flags_23493)
            
            # Assigning a Call to a Name (line 279):
            
            # Assigning a Call to a Name (line 279):
            
            # Call to _flags_fromnum(...): (line 279)
            # Processing the call arguments (line 279)
            # Getting the type of 'num' (line 279)
            num_23495 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 279, 35), 'num', False)
            # Processing the call keyword arguments (line 279)
            kwargs_23496 = {}
            # Getting the type of '_flags_fromnum' (line 279)
            _flags_fromnum_23494 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 279, 20), '_flags_fromnum', False)
            # Calling _flags_fromnum(args, kwargs) (line 279)
            _flags_fromnum_call_result_23497 = invoke(stypy.reporting.localization.Localization(__file__, 279, 20), _flags_fromnum_23494, *[num_23495], **kwargs_23496)
            
            # Assigning a type to the variable 'flags' (line 279)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 279, 12), 'flags', _flags_fromnum_call_result_23497)
            # SSA branch for the else part of an if statement (line 277)
            module_type_store.open_ssa_branch('else')
            
            
            # Call to isinstance(...): (line 280)
            # Processing the call arguments (line 280)
            # Getting the type of 'flags' (line 280)
            flags_23499 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 280, 24), 'flags', False)
            # Getting the type of 'flagsobj' (line 280)
            flagsobj_23500 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 280, 31), 'flagsobj', False)
            # Processing the call keyword arguments (line 280)
            kwargs_23501 = {}
            # Getting the type of 'isinstance' (line 280)
            isinstance_23498 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 280, 13), 'isinstance', False)
            # Calling isinstance(args, kwargs) (line 280)
            isinstance_call_result_23502 = invoke(stypy.reporting.localization.Localization(__file__, 280, 13), isinstance_23498, *[flags_23499, flagsobj_23500], **kwargs_23501)
            
            # Testing the type of an if condition (line 280)
            if_condition_23503 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 280, 13), isinstance_call_result_23502)
            # Assigning a type to the variable 'if_condition_23503' (line 280)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 280, 13), 'if_condition_23503', if_condition_23503)
            # SSA begins for if statement (line 280)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a Attribute to a Name (line 281):
            
            # Assigning a Attribute to a Name (line 281):
            # Getting the type of 'flags' (line 281)
            flags_23504 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 281, 18), 'flags')
            # Obtaining the member 'num' of a type (line 281)
            num_23505 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 281, 18), flags_23504, 'num')
            # Assigning a type to the variable 'num' (line 281)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 281, 12), 'num', num_23505)
            
            # Assigning a Call to a Name (line 282):
            
            # Assigning a Call to a Name (line 282):
            
            # Call to _flags_fromnum(...): (line 282)
            # Processing the call arguments (line 282)
            # Getting the type of 'num' (line 282)
            num_23507 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 282, 35), 'num', False)
            # Processing the call keyword arguments (line 282)
            kwargs_23508 = {}
            # Getting the type of '_flags_fromnum' (line 282)
            _flags_fromnum_23506 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 282, 20), '_flags_fromnum', False)
            # Calling _flags_fromnum(args, kwargs) (line 282)
            _flags_fromnum_call_result_23509 = invoke(stypy.reporting.localization.Localization(__file__, 282, 20), _flags_fromnum_23506, *[num_23507], **kwargs_23508)
            
            # Assigning a type to the variable 'flags' (line 282)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 282, 12), 'flags', _flags_fromnum_call_result_23509)
            # SSA join for if statement (line 280)
            module_type_store = module_type_store.join_ssa_context()
            
            # SSA join for if statement (line 277)
            module_type_store = module_type_store.join_ssa_context()
            

            if (may_be_23478 and more_types_in_union_23479):
                # SSA join for if statement (line 275)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Type idiom detected: calculating its left and rigth part (line 283)
        # Getting the type of 'num' (line 283)
        num_23510 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 283, 11), 'num')
        # Getting the type of 'None' (line 283)
        None_23511 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 283, 18), 'None')
        
        (may_be_23512, more_types_in_union_23513) = may_be_none(num_23510, None_23511)

        if may_be_23512:

            if more_types_in_union_23513:
                # Runtime conditional SSA (line 283)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            
            # SSA begins for try-except statement (line 284)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
            
            # Assigning a ListComp to a Name (line 285):
            
            # Assigning a ListComp to a Name (line 285):
            # Calculating list comprehension
            # Calculating comprehension expression
            # Getting the type of 'flags' (line 285)
            flags_23521 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 285, 52), 'flags')
            comprehension_23522 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 285, 25), flags_23521)
            # Assigning a type to the variable 'x' (line 285)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 285, 25), 'x', comprehension_23522)
            
            # Call to upper(...): (line 285)
            # Processing the call keyword arguments (line 285)
            kwargs_23519 = {}
            
            # Call to strip(...): (line 285)
            # Processing the call keyword arguments (line 285)
            kwargs_23516 = {}
            # Getting the type of 'x' (line 285)
            x_23514 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 285, 25), 'x', False)
            # Obtaining the member 'strip' of a type (line 285)
            strip_23515 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 285, 25), x_23514, 'strip')
            # Calling strip(args, kwargs) (line 285)
            strip_call_result_23517 = invoke(stypy.reporting.localization.Localization(__file__, 285, 25), strip_23515, *[], **kwargs_23516)
            
            # Obtaining the member 'upper' of a type (line 285)
            upper_23518 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 285, 25), strip_call_result_23517, 'upper')
            # Calling upper(args, kwargs) (line 285)
            upper_call_result_23520 = invoke(stypy.reporting.localization.Localization(__file__, 285, 25), upper_23518, *[], **kwargs_23519)
            
            list_23523 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 285, 25), 'list')
            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 285, 25), list_23523, upper_call_result_23520)
            # Assigning a type to the variable 'flags' (line 285)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 285, 16), 'flags', list_23523)
            # SSA branch for the except part of a try statement (line 284)
            # SSA branch for the except '<any exception>' branch of a try statement (line 284)
            module_type_store.open_ssa_branch('except')
            
            # Call to TypeError(...): (line 287)
            # Processing the call arguments (line 287)
            str_23525 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 287, 32), 'str', 'invalid flags specification')
            # Processing the call keyword arguments (line 287)
            kwargs_23526 = {}
            # Getting the type of 'TypeError' (line 287)
            TypeError_23524 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 287, 22), 'TypeError', False)
            # Calling TypeError(args, kwargs) (line 287)
            TypeError_call_result_23527 = invoke(stypy.reporting.localization.Localization(__file__, 287, 22), TypeError_23524, *[str_23525], **kwargs_23526)
            
            ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 287, 16), TypeError_call_result_23527, 'raise parameter', BaseException)
            # SSA join for try-except statement (line 284)
            module_type_store = module_type_store.join_ssa_context()
            
            
            # Assigning a Call to a Name (line 288):
            
            # Assigning a Call to a Name (line 288):
            
            # Call to _num_fromflags(...): (line 288)
            # Processing the call arguments (line 288)
            # Getting the type of 'flags' (line 288)
            flags_23529 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 288, 33), 'flags', False)
            # Processing the call keyword arguments (line 288)
            kwargs_23530 = {}
            # Getting the type of '_num_fromflags' (line 288)
            _num_fromflags_23528 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 288, 18), '_num_fromflags', False)
            # Calling _num_fromflags(args, kwargs) (line 288)
            _num_fromflags_call_result_23531 = invoke(stypy.reporting.localization.Localization(__file__, 288, 18), _num_fromflags_23528, *[flags_23529], **kwargs_23530)
            
            # Assigning a type to the variable 'num' (line 288)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 288, 12), 'num', _num_fromflags_call_result_23531)

            if more_types_in_union_23513:
                # SSA join for if statement (line 283)
                module_type_store = module_type_store.join_ssa_context()


        

        if more_types_in_union_23475:
            # SSA join for if statement (line 274)
            module_type_store = module_type_store.join_ssa_context()


    
    
    
    # SSA begins for try-except statement (line 289)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
    
    # Obtaining the type of the subscript
    
    # Obtaining an instance of the builtin type 'tuple' (line 290)
    tuple_23532 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 290, 36), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 290)
    # Adding element type (line 290)
    # Getting the type of 'dtype' (line 290)
    dtype_23533 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 290, 36), 'dtype')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 290, 36), tuple_23532, dtype_23533)
    # Adding element type (line 290)
    # Getting the type of 'ndim' (line 290)
    ndim_23534 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 290, 43), 'ndim')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 290, 36), tuple_23532, ndim_23534)
    # Adding element type (line 290)
    # Getting the type of 'shape' (line 290)
    shape_23535 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 290, 49), 'shape')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 290, 36), tuple_23532, shape_23535)
    # Adding element type (line 290)
    # Getting the type of 'num' (line 290)
    num_23536 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 290, 56), 'num')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 290, 36), tuple_23532, num_23536)
    
    # Getting the type of '_pointer_type_cache' (line 290)
    _pointer_type_cache_23537 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 290, 15), '_pointer_type_cache')
    # Obtaining the member '__getitem__' of a type (line 290)
    getitem___23538 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 290, 15), _pointer_type_cache_23537, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 290)
    subscript_call_result_23539 = invoke(stypy.reporting.localization.Localization(__file__, 290, 15), getitem___23538, tuple_23532)
    
    # Assigning a type to the variable 'stypy_return_type' (line 290)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 290, 8), 'stypy_return_type', subscript_call_result_23539)
    # SSA branch for the except part of a try statement (line 289)
    # SSA branch for the except 'KeyError' branch of a try statement (line 289)
    module_type_store.open_ssa_branch('except')
    pass
    # SSA join for try-except statement (line 289)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Type idiom detected: calculating its left and rigth part (line 293)
    # Getting the type of 'dtype' (line 293)
    dtype_23540 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 293, 7), 'dtype')
    # Getting the type of 'None' (line 293)
    None_23541 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 293, 16), 'None')
    
    (may_be_23542, more_types_in_union_23543) = may_be_none(dtype_23540, None_23541)

    if may_be_23542:

        if more_types_in_union_23543:
            # Runtime conditional SSA (line 293)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a Str to a Name (line 294):
        
        # Assigning a Str to a Name (line 294):
        str_23544 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 294, 15), 'str', 'any')
        # Assigning a type to the variable 'name' (line 294)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 294, 8), 'name', str_23544)

        if more_types_in_union_23543:
            # Runtime conditional SSA for else branch (line 293)
            module_type_store.open_ssa_branch('idiom else')



    if ((not may_be_23542) or more_types_in_union_23543):
        
        # Getting the type of 'dtype' (line 295)
        dtype_23545 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 295, 9), 'dtype')
        # Obtaining the member 'names' of a type (line 295)
        names_23546 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 295, 9), dtype_23545, 'names')
        # Testing the type of an if condition (line 295)
        if_condition_23547 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 295, 9), names_23546)
        # Assigning a type to the variable 'if_condition_23547' (line 295)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 295, 9), 'if_condition_23547', if_condition_23547)
        # SSA begins for if statement (line 295)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 296):
        
        # Assigning a Call to a Name (line 296):
        
        # Call to str(...): (line 296)
        # Processing the call arguments (line 296)
        
        # Call to id(...): (line 296)
        # Processing the call arguments (line 296)
        # Getting the type of 'dtype' (line 296)
        dtype_23550 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 296, 22), 'dtype', False)
        # Processing the call keyword arguments (line 296)
        kwargs_23551 = {}
        # Getting the type of 'id' (line 296)
        id_23549 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 296, 19), 'id', False)
        # Calling id(args, kwargs) (line 296)
        id_call_result_23552 = invoke(stypy.reporting.localization.Localization(__file__, 296, 19), id_23549, *[dtype_23550], **kwargs_23551)
        
        # Processing the call keyword arguments (line 296)
        kwargs_23553 = {}
        # Getting the type of 'str' (line 296)
        str_23548 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 296, 15), 'str', False)
        # Calling str(args, kwargs) (line 296)
        str_call_result_23554 = invoke(stypy.reporting.localization.Localization(__file__, 296, 15), str_23548, *[id_call_result_23552], **kwargs_23553)
        
        # Assigning a type to the variable 'name' (line 296)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 296, 8), 'name', str_call_result_23554)
        # SSA branch for the else part of an if statement (line 295)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Attribute to a Name (line 298):
        
        # Assigning a Attribute to a Name (line 298):
        # Getting the type of 'dtype' (line 298)
        dtype_23555 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 298, 15), 'dtype')
        # Obtaining the member 'str' of a type (line 298)
        str_23556 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 298, 15), dtype_23555, 'str')
        # Assigning a type to the variable 'name' (line 298)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 298, 8), 'name', str_23556)
        # SSA join for if statement (line 295)
        module_type_store = module_type_store.join_ssa_context()
        

        if (may_be_23542 and more_types_in_union_23543):
            # SSA join for if statement (line 293)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Type idiom detected: calculating its left and rigth part (line 299)
    # Getting the type of 'ndim' (line 299)
    ndim_23557 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 299, 4), 'ndim')
    # Getting the type of 'None' (line 299)
    None_23558 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 299, 19), 'None')
    
    (may_be_23559, more_types_in_union_23560) = may_not_be_none(ndim_23557, None_23558)

    if may_be_23559:

        if more_types_in_union_23560:
            # Runtime conditional SSA (line 299)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Getting the type of 'name' (line 300)
        name_23561 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 300, 8), 'name')
        str_23562 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 300, 16), 'str', '_%dd')
        # Getting the type of 'ndim' (line 300)
        ndim_23563 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 300, 25), 'ndim')
        # Applying the binary operator '%' (line 300)
        result_mod_23564 = python_operator(stypy.reporting.localization.Localization(__file__, 300, 16), '%', str_23562, ndim_23563)
        
        # Applying the binary operator '+=' (line 300)
        result_iadd_23565 = python_operator(stypy.reporting.localization.Localization(__file__, 300, 8), '+=', name_23561, result_mod_23564)
        # Assigning a type to the variable 'name' (line 300)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 300, 8), 'name', result_iadd_23565)
        

        if more_types_in_union_23560:
            # SSA join for if statement (line 299)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Type idiom detected: calculating its left and rigth part (line 301)
    # Getting the type of 'shape' (line 301)
    shape_23566 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 301, 4), 'shape')
    # Getting the type of 'None' (line 301)
    None_23567 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 301, 20), 'None')
    
    (may_be_23568, more_types_in_union_23569) = may_not_be_none(shape_23566, None_23567)

    if may_be_23568:

        if more_types_in_union_23569:
            # Runtime conditional SSA (line 301)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        
        # SSA begins for try-except statement (line 302)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
        
        # Assigning a ListComp to a Name (line 303):
        
        # Assigning a ListComp to a Name (line 303):
        # Calculating list comprehension
        # Calculating comprehension expression
        # Getting the type of 'shape' (line 303)
        shape_23574 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 303, 40), 'shape')
        comprehension_23575 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 303, 24), shape_23574)
        # Assigning a type to the variable 'x' (line 303)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 303, 24), 'x', comprehension_23575)
        
        # Call to str(...): (line 303)
        # Processing the call arguments (line 303)
        # Getting the type of 'x' (line 303)
        x_23571 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 303, 28), 'x', False)
        # Processing the call keyword arguments (line 303)
        kwargs_23572 = {}
        # Getting the type of 'str' (line 303)
        str_23570 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 303, 24), 'str', False)
        # Calling str(args, kwargs) (line 303)
        str_call_result_23573 = invoke(stypy.reporting.localization.Localization(__file__, 303, 24), str_23570, *[x_23571], **kwargs_23572)
        
        list_23576 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 303, 24), 'list')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 303, 24), list_23576, str_call_result_23573)
        # Assigning a type to the variable 'strshape' (line 303)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 303, 12), 'strshape', list_23576)
        # SSA branch for the except part of a try statement (line 302)
        # SSA branch for the except 'TypeError' branch of a try statement (line 302)
        module_type_store.open_ssa_branch('except')
        
        # Assigning a List to a Name (line 305):
        
        # Assigning a List to a Name (line 305):
        
        # Obtaining an instance of the builtin type 'list' (line 305)
        list_23577 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 305, 23), 'list')
        # Adding type elements to the builtin type 'list' instance (line 305)
        # Adding element type (line 305)
        
        # Call to str(...): (line 305)
        # Processing the call arguments (line 305)
        # Getting the type of 'shape' (line 305)
        shape_23579 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 305, 28), 'shape', False)
        # Processing the call keyword arguments (line 305)
        kwargs_23580 = {}
        # Getting the type of 'str' (line 305)
        str_23578 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 305, 24), 'str', False)
        # Calling str(args, kwargs) (line 305)
        str_call_result_23581 = invoke(stypy.reporting.localization.Localization(__file__, 305, 24), str_23578, *[shape_23579], **kwargs_23580)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 305, 23), list_23577, str_call_result_23581)
        
        # Assigning a type to the variable 'strshape' (line 305)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 305, 12), 'strshape', list_23577)
        
        # Assigning a Tuple to a Name (line 306):
        
        # Assigning a Tuple to a Name (line 306):
        
        # Obtaining an instance of the builtin type 'tuple' (line 306)
        tuple_23582 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 306, 21), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 306)
        # Adding element type (line 306)
        # Getting the type of 'shape' (line 306)
        shape_23583 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 306, 21), 'shape')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 306, 21), tuple_23582, shape_23583)
        
        # Assigning a type to the variable 'shape' (line 306)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 306, 12), 'shape', tuple_23582)
        # SSA join for try-except statement (line 302)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Name (line 307):
        
        # Assigning a Call to a Name (line 307):
        
        # Call to tuple(...): (line 307)
        # Processing the call arguments (line 307)
        # Getting the type of 'shape' (line 307)
        shape_23585 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 307, 22), 'shape', False)
        # Processing the call keyword arguments (line 307)
        kwargs_23586 = {}
        # Getting the type of 'tuple' (line 307)
        tuple_23584 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 307, 16), 'tuple', False)
        # Calling tuple(args, kwargs) (line 307)
        tuple_call_result_23587 = invoke(stypy.reporting.localization.Localization(__file__, 307, 16), tuple_23584, *[shape_23585], **kwargs_23586)
        
        # Assigning a type to the variable 'shape' (line 307)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 307, 8), 'shape', tuple_call_result_23587)
        
        # Getting the type of 'name' (line 308)
        name_23588 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 308, 8), 'name')
        str_23589 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 308, 16), 'str', '_')
        
        # Call to join(...): (line 308)
        # Processing the call arguments (line 308)
        # Getting the type of 'strshape' (line 308)
        strshape_23592 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 308, 29), 'strshape', False)
        # Processing the call keyword arguments (line 308)
        kwargs_23593 = {}
        str_23590 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 308, 20), 'str', 'x')
        # Obtaining the member 'join' of a type (line 308)
        join_23591 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 308, 20), str_23590, 'join')
        # Calling join(args, kwargs) (line 308)
        join_call_result_23594 = invoke(stypy.reporting.localization.Localization(__file__, 308, 20), join_23591, *[strshape_23592], **kwargs_23593)
        
        # Applying the binary operator '+' (line 308)
        result_add_23595 = python_operator(stypy.reporting.localization.Localization(__file__, 308, 16), '+', str_23589, join_call_result_23594)
        
        # Applying the binary operator '+=' (line 308)
        result_iadd_23596 = python_operator(stypy.reporting.localization.Localization(__file__, 308, 8), '+=', name_23588, result_add_23595)
        # Assigning a type to the variable 'name' (line 308)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 308, 8), 'name', result_iadd_23596)
        

        if more_types_in_union_23569:
            # SSA join for if statement (line 301)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Type idiom detected: calculating its left and rigth part (line 309)
    # Getting the type of 'flags' (line 309)
    flags_23597 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 309, 4), 'flags')
    # Getting the type of 'None' (line 309)
    None_23598 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 309, 20), 'None')
    
    (may_be_23599, more_types_in_union_23600) = may_not_be_none(flags_23597, None_23598)

    if may_be_23599:

        if more_types_in_union_23600:
            # Runtime conditional SSA (line 309)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Getting the type of 'name' (line 310)
        name_23601 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 310, 8), 'name')
        str_23602 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 310, 16), 'str', '_')
        
        # Call to join(...): (line 310)
        # Processing the call arguments (line 310)
        # Getting the type of 'flags' (line 310)
        flags_23605 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 310, 29), 'flags', False)
        # Processing the call keyword arguments (line 310)
        kwargs_23606 = {}
        str_23603 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 310, 20), 'str', '_')
        # Obtaining the member 'join' of a type (line 310)
        join_23604 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 310, 20), str_23603, 'join')
        # Calling join(args, kwargs) (line 310)
        join_call_result_23607 = invoke(stypy.reporting.localization.Localization(__file__, 310, 20), join_23604, *[flags_23605], **kwargs_23606)
        
        # Applying the binary operator '+' (line 310)
        result_add_23608 = python_operator(stypy.reporting.localization.Localization(__file__, 310, 16), '+', str_23602, join_call_result_23607)
        
        # Applying the binary operator '+=' (line 310)
        result_iadd_23609 = python_operator(stypy.reporting.localization.Localization(__file__, 310, 8), '+=', name_23601, result_add_23608)
        # Assigning a type to the variable 'name' (line 310)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 310, 8), 'name', result_iadd_23609)
        

        if more_types_in_union_23600:
            # Runtime conditional SSA for else branch (line 309)
            module_type_store.open_ssa_branch('idiom else')



    if ((not may_be_23599) or more_types_in_union_23600):
        
        # Assigning a List to a Name (line 312):
        
        # Assigning a List to a Name (line 312):
        
        # Obtaining an instance of the builtin type 'list' (line 312)
        list_23610 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 312, 16), 'list')
        # Adding type elements to the builtin type 'list' instance (line 312)
        
        # Assigning a type to the variable 'flags' (line 312)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 312, 8), 'flags', list_23610)

        if (may_be_23599 and more_types_in_union_23600):
            # SSA join for if statement (line 309)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Assigning a Call to a Name (line 313):
    
    # Assigning a Call to a Name (line 313):
    
    # Call to type(...): (line 313)
    # Processing the call arguments (line 313)
    str_23612 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 313, 17), 'str', 'ndpointer_%s')
    # Getting the type of 'name' (line 313)
    name_23613 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 313, 32), 'name', False)
    # Applying the binary operator '%' (line 313)
    result_mod_23614 = python_operator(stypy.reporting.localization.Localization(__file__, 313, 17), '%', str_23612, name_23613)
    
    
    # Obtaining an instance of the builtin type 'tuple' (line 313)
    tuple_23615 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 313, 39), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 313)
    # Adding element type (line 313)
    # Getting the type of '_ndptr' (line 313)
    _ndptr_23616 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 313, 39), '_ndptr', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 313, 39), tuple_23615, _ndptr_23616)
    
    
    # Obtaining an instance of the builtin type 'dict' (line 314)
    dict_23617 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 314, 17), 'dict')
    # Adding type elements to the builtin type 'dict' instance (line 314)
    # Adding element type (key, value) (line 314)
    str_23618 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 314, 18), 'str', '_dtype_')
    # Getting the type of 'dtype' (line 314)
    dtype_23619 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 314, 29), 'dtype', False)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 314, 17), dict_23617, (str_23618, dtype_23619))
    # Adding element type (key, value) (line 314)
    str_23620 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 315, 18), 'str', '_shape_')
    # Getting the type of 'shape' (line 315)
    shape_23621 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 315, 30), 'shape', False)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 314, 17), dict_23617, (str_23620, shape_23621))
    # Adding element type (key, value) (line 314)
    str_23622 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 316, 18), 'str', '_ndim_')
    # Getting the type of 'ndim' (line 316)
    ndim_23623 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 316, 29), 'ndim', False)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 314, 17), dict_23617, (str_23622, ndim_23623))
    # Adding element type (key, value) (line 314)
    str_23624 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 317, 18), 'str', '_flags_')
    # Getting the type of 'num' (line 317)
    num_23625 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 317, 30), 'num', False)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 314, 17), dict_23617, (str_23624, num_23625))
    
    # Processing the call keyword arguments (line 313)
    kwargs_23626 = {}
    # Getting the type of 'type' (line 313)
    type_23611 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 313, 12), 'type', False)
    # Calling type(args, kwargs) (line 313)
    type_call_result_23627 = invoke(stypy.reporting.localization.Localization(__file__, 313, 12), type_23611, *[result_mod_23614, tuple_23615, dict_23617], **kwargs_23626)
    
    # Assigning a type to the variable 'klass' (line 313)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 313, 4), 'klass', type_call_result_23627)
    
    # Assigning a Name to a Subscript (line 318):
    
    # Assigning a Name to a Subscript (line 318):
    # Getting the type of 'klass' (line 318)
    klass_23628 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 318, 33), 'klass')
    # Getting the type of '_pointer_type_cache' (line 318)
    _pointer_type_cache_23629 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 318, 4), '_pointer_type_cache')
    # Getting the type of 'dtype' (line 318)
    dtype_23630 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 318, 24), 'dtype')
    # Storing an element on a container (line 318)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 318, 4), _pointer_type_cache_23629, (dtype_23630, klass_23628))
    # Getting the type of 'klass' (line 319)
    klass_23631 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 319, 11), 'klass')
    # Assigning a type to the variable 'stypy_return_type' (line 319)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 319, 4), 'stypy_return_type', klass_23631)
    
    # ################# End of 'ndpointer(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'ndpointer' in the type store
    # Getting the type of 'stypy_return_type' (line 219)
    stypy_return_type_23632 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 219, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_23632)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'ndpointer'
    return stypy_return_type_23632

# Assigning a type to the variable 'ndpointer' (line 219)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 219, 0), 'ndpointer', ndpointer)

# Type idiom detected: calculating its left and rigth part (line 321)
# Getting the type of 'ctypes' (line 321)
ctypes_23633 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 321, 0), 'ctypes')
# Getting the type of 'None' (line 321)
None_23634 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 321, 17), 'None')

(may_be_23635, more_types_in_union_23636) = may_not_be_none(ctypes_23633, None_23634)

if may_be_23635:

    if more_types_in_union_23636:
        # Runtime conditional SSA (line 321)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
    else:
        module_type_store = module_type_store

    
    # Assigning a Name to a Name (line 322):
    
    # Assigning a Name to a Name (line 322):
    # Getting the type of 'ctypes' (line 322)
    ctypes_23637 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 322, 9), 'ctypes')
    # Assigning a type to the variable 'ct' (line 322)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 322, 4), 'ct', ctypes_23637)
    
    # Assigning a Dict to a Name (line 328):
    
    # Assigning a Dict to a Name (line 328):
    
    # Obtaining an instance of the builtin type 'dict' (line 328)
    dict_23638 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 328, 17), 'dict')
    # Adding type elements to the builtin type 'dict' instance (line 328)
    
    # Assigning a type to the variable '_typecodes' (line 328)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 328, 4), '_typecodes', dict_23638)

    @norecursion
    def prep_simple(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'prep_simple'
        module_type_store = module_type_store.open_function_context('prep_simple', 330, 4, False)
        
        # Passed parameters checking function
        prep_simple.stypy_localization = localization
        prep_simple.stypy_type_of_self = None
        prep_simple.stypy_type_store = module_type_store
        prep_simple.stypy_function_name = 'prep_simple'
        prep_simple.stypy_param_names_list = ['simple_type', 'dtype']
        prep_simple.stypy_varargs_param_name = None
        prep_simple.stypy_kwargs_param_name = None
        prep_simple.stypy_call_defaults = defaults
        prep_simple.stypy_call_varargs = varargs
        prep_simple.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, 'prep_simple', ['simple_type', 'dtype'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'prep_simple', localization, ['simple_type', 'dtype'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'prep_simple(...)' code ##################

        str_23639 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 333, (-1)), 'str', 'Given a ctypes simple type, construct and attach an\n        __array_interface__ property to it if it does not yet have one.\n        ')
        
        
        # SSA begins for try-except statement (line 334)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
        # Getting the type of 'simple_type' (line 334)
        simple_type_23640 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 334, 13), 'simple_type')
        # Obtaining the member '__array_interface__' of a type (line 334)
        array_interface___23641 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 334, 13), simple_type_23640, '__array_interface__')
        # SSA branch for the except part of a try statement (line 334)
        # SSA branch for the except 'AttributeError' branch of a try statement (line 334)
        module_type_store.open_ssa_branch('except')
        pass
        # SSA branch for the else branch of a try statement (line 334)
        module_type_store.open_ssa_branch('except else')
        # Assigning a type to the variable 'stypy_return_type' (line 336)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 336, 14), 'stypy_return_type', types.NoneType)
        # SSA join for try-except statement (line 334)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Attribute to a Name (line 338):
        
        # Assigning a Attribute to a Name (line 338):
        
        # Call to _dtype(...): (line 338)
        # Processing the call arguments (line 338)
        # Getting the type of 'dtype' (line 338)
        dtype_23643 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 338, 25), 'dtype', False)
        # Processing the call keyword arguments (line 338)
        kwargs_23644 = {}
        # Getting the type of '_dtype' (line 338)
        _dtype_23642 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 338, 18), '_dtype', False)
        # Calling _dtype(args, kwargs) (line 338)
        _dtype_call_result_23645 = invoke(stypy.reporting.localization.Localization(__file__, 338, 18), _dtype_23642, *[dtype_23643], **kwargs_23644)
        
        # Obtaining the member 'str' of a type (line 338)
        str_23646 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 338, 18), _dtype_call_result_23645, 'str')
        # Assigning a type to the variable 'typestr' (line 338)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 338, 8), 'typestr', str_23646)
        
        # Assigning a Name to a Subscript (line 339):
        
        # Assigning a Name to a Subscript (line 339):
        # Getting the type of 'simple_type' (line 339)
        simple_type_23647 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 339, 30), 'simple_type')
        # Getting the type of '_typecodes' (line 339)
        _typecodes_23648 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 339, 8), '_typecodes')
        # Getting the type of 'typestr' (line 339)
        typestr_23649 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 339, 19), 'typestr')
        # Storing an element on a container (line 339)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 339, 8), _typecodes_23648, (typestr_23649, simple_type_23647))

        @norecursion
        def __array_interface__(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '__array_interface__'
            module_type_store = module_type_store.open_function_context('__array_interface__', 341, 8, False)
            
            # Passed parameters checking function
            __array_interface__.stypy_localization = localization
            __array_interface__.stypy_type_of_self = None
            __array_interface__.stypy_type_store = module_type_store
            __array_interface__.stypy_function_name = '__array_interface__'
            __array_interface__.stypy_param_names_list = ['self']
            __array_interface__.stypy_varargs_param_name = None
            __array_interface__.stypy_kwargs_param_name = None
            __array_interface__.stypy_call_defaults = defaults
            __array_interface__.stypy_call_varargs = varargs
            __array_interface__.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, '__array_interface__', ['self'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, '__array_interface__', localization, ['self'], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of '__array_interface__(...)' code ##################

            
            # Obtaining an instance of the builtin type 'dict' (line 342)
            dict_23650 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 342, 19), 'dict')
            # Adding type elements to the builtin type 'dict' instance (line 342)
            # Adding element type (key, value) (line 342)
            str_23651 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 342, 20), 'str', 'descr')
            
            # Obtaining an instance of the builtin type 'list' (line 342)
            list_23652 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 342, 29), 'list')
            # Adding type elements to the builtin type 'list' instance (line 342)
            # Adding element type (line 342)
            
            # Obtaining an instance of the builtin type 'tuple' (line 342)
            tuple_23653 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 342, 31), 'tuple')
            # Adding type elements to the builtin type 'tuple' instance (line 342)
            # Adding element type (line 342)
            str_23654 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 342, 31), 'str', '')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 342, 31), tuple_23653, str_23654)
            # Adding element type (line 342)
            # Getting the type of 'typestr' (line 342)
            typestr_23655 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 342, 35), 'typestr')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 342, 31), tuple_23653, typestr_23655)
            
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 342, 29), list_23652, tuple_23653)
            
            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 342, 19), dict_23650, (str_23651, list_23652))
            # Adding element type (key, value) (line 342)
            str_23656 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 343, 20), 'str', '__ref')
            # Getting the type of 'self' (line 343)
            self_23657 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 343, 29), 'self')
            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 342, 19), dict_23650, (str_23656, self_23657))
            # Adding element type (key, value) (line 342)
            str_23658 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 344, 20), 'str', 'strides')
            # Getting the type of 'None' (line 344)
            None_23659 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 344, 31), 'None')
            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 342, 19), dict_23650, (str_23658, None_23659))
            # Adding element type (key, value) (line 342)
            str_23660 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 345, 20), 'str', 'shape')
            
            # Obtaining an instance of the builtin type 'tuple' (line 345)
            tuple_23661 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 345, 29), 'tuple')
            # Adding type elements to the builtin type 'tuple' instance (line 345)
            
            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 342, 19), dict_23650, (str_23660, tuple_23661))
            # Adding element type (key, value) (line 342)
            str_23662 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 346, 20), 'str', 'version')
            int_23663 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 346, 31), 'int')
            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 342, 19), dict_23650, (str_23662, int_23663))
            # Adding element type (key, value) (line 342)
            str_23664 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 347, 20), 'str', 'typestr')
            # Getting the type of 'typestr' (line 347)
            typestr_23665 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 347, 31), 'typestr')
            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 342, 19), dict_23650, (str_23664, typestr_23665))
            # Adding element type (key, value) (line 342)
            str_23666 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 348, 20), 'str', 'data')
            
            # Obtaining an instance of the builtin type 'tuple' (line 348)
            tuple_23667 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 348, 29), 'tuple')
            # Adding type elements to the builtin type 'tuple' instance (line 348)
            # Adding element type (line 348)
            
            # Call to addressof(...): (line 348)
            # Processing the call arguments (line 348)
            # Getting the type of 'self' (line 348)
            self_23670 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 348, 42), 'self', False)
            # Processing the call keyword arguments (line 348)
            kwargs_23671 = {}
            # Getting the type of 'ct' (line 348)
            ct_23668 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 348, 29), 'ct', False)
            # Obtaining the member 'addressof' of a type (line 348)
            addressof_23669 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 348, 29), ct_23668, 'addressof')
            # Calling addressof(args, kwargs) (line 348)
            addressof_call_result_23672 = invoke(stypy.reporting.localization.Localization(__file__, 348, 29), addressof_23669, *[self_23670], **kwargs_23671)
            
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 348, 29), tuple_23667, addressof_call_result_23672)
            # Adding element type (line 348)
            # Getting the type of 'False' (line 348)
            False_23673 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 348, 49), 'False')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 348, 29), tuple_23667, False_23673)
            
            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 342, 19), dict_23650, (str_23666, tuple_23667))
            
            # Assigning a type to the variable 'stypy_return_type' (line 342)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 342, 12), 'stypy_return_type', dict_23650)
            
            # ################# End of '__array_interface__(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function '__array_interface__' in the type store
            # Getting the type of 'stypy_return_type' (line 341)
            stypy_return_type_23674 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 341, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_23674)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function '__array_interface__'
            return stypy_return_type_23674

        # Assigning a type to the variable '__array_interface__' (line 341)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 341, 8), '__array_interface__', __array_interface__)
        
        # Assigning a Call to a Attribute (line 351):
        
        # Assigning a Call to a Attribute (line 351):
        
        # Call to property(...): (line 351)
        # Processing the call arguments (line 351)
        # Getting the type of '__array_interface__' (line 351)
        array_interface___23676 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 351, 51), '__array_interface__', False)
        # Processing the call keyword arguments (line 351)
        kwargs_23677 = {}
        # Getting the type of 'property' (line 351)
        property_23675 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 351, 42), 'property', False)
        # Calling property(args, kwargs) (line 351)
        property_call_result_23678 = invoke(stypy.reporting.localization.Localization(__file__, 351, 42), property_23675, *[array_interface___23676], **kwargs_23677)
        
        # Getting the type of 'simple_type' (line 351)
        simple_type_23679 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 351, 8), 'simple_type')
        # Setting the type of the member '__array_interface__' of a type (line 351)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 351, 8), simple_type_23679, '__array_interface__', property_call_result_23678)
        
        # ################# End of 'prep_simple(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'prep_simple' in the type store
        # Getting the type of 'stypy_return_type' (line 330)
        stypy_return_type_23680 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 330, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_23680)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'prep_simple'
        return stypy_return_type_23680

    # Assigning a type to the variable 'prep_simple' (line 330)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 330, 4), 'prep_simple', prep_simple)
    
    # Assigning a List to a Name (line 353):
    
    # Assigning a List to a Name (line 353):
    
    # Obtaining an instance of the builtin type 'list' (line 353)
    list_23681 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 353, 19), 'list')
    # Adding type elements to the builtin type 'list' instance (line 353)
    # Adding element type (line 353)
    
    # Obtaining an instance of the builtin type 'tuple' (line 354)
    tuple_23682 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 354, 9), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 354)
    # Adding element type (line 354)
    
    # Obtaining an instance of the builtin type 'tuple' (line 354)
    tuple_23683 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 354, 10), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 354)
    # Adding element type (line 354)
    # Getting the type of 'ct' (line 354)
    ct_23684 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 354, 10), 'ct')
    # Obtaining the member 'c_byte' of a type (line 354)
    c_byte_23685 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 354, 10), ct_23684, 'c_byte')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 354, 10), tuple_23683, c_byte_23685)
    # Adding element type (line 354)
    # Getting the type of 'ct' (line 354)
    ct_23686 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 354, 21), 'ct')
    # Obtaining the member 'c_short' of a type (line 354)
    c_short_23687 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 354, 21), ct_23686, 'c_short')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 354, 10), tuple_23683, c_short_23687)
    # Adding element type (line 354)
    # Getting the type of 'ct' (line 354)
    ct_23688 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 354, 33), 'ct')
    # Obtaining the member 'c_int' of a type (line 354)
    c_int_23689 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 354, 33), ct_23688, 'c_int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 354, 10), tuple_23683, c_int_23689)
    # Adding element type (line 354)
    # Getting the type of 'ct' (line 354)
    ct_23690 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 354, 43), 'ct')
    # Obtaining the member 'c_long' of a type (line 354)
    c_long_23691 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 354, 43), ct_23690, 'c_long')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 354, 10), tuple_23683, c_long_23691)
    # Adding element type (line 354)
    # Getting the type of 'ct' (line 354)
    ct_23692 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 354, 54), 'ct')
    # Obtaining the member 'c_longlong' of a type (line 354)
    c_longlong_23693 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 354, 54), ct_23692, 'c_longlong')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 354, 10), tuple_23683, c_longlong_23693)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 354, 9), tuple_23682, tuple_23683)
    # Adding element type (line 354)
    str_23694 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 354, 70), 'str', 'i')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 354, 9), tuple_23682, str_23694)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 353, 19), list_23681, tuple_23682)
    # Adding element type (line 353)
    
    # Obtaining an instance of the builtin type 'tuple' (line 355)
    tuple_23695 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 355, 9), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 355)
    # Adding element type (line 355)
    
    # Obtaining an instance of the builtin type 'tuple' (line 355)
    tuple_23696 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 355, 10), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 355)
    # Adding element type (line 355)
    # Getting the type of 'ct' (line 355)
    ct_23697 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 355, 10), 'ct')
    # Obtaining the member 'c_ubyte' of a type (line 355)
    c_ubyte_23698 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 355, 10), ct_23697, 'c_ubyte')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 355, 10), tuple_23696, c_ubyte_23698)
    # Adding element type (line 355)
    # Getting the type of 'ct' (line 355)
    ct_23699 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 355, 22), 'ct')
    # Obtaining the member 'c_ushort' of a type (line 355)
    c_ushort_23700 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 355, 22), ct_23699, 'c_ushort')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 355, 10), tuple_23696, c_ushort_23700)
    # Adding element type (line 355)
    # Getting the type of 'ct' (line 355)
    ct_23701 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 355, 35), 'ct')
    # Obtaining the member 'c_uint' of a type (line 355)
    c_uint_23702 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 355, 35), ct_23701, 'c_uint')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 355, 10), tuple_23696, c_uint_23702)
    # Adding element type (line 355)
    # Getting the type of 'ct' (line 355)
    ct_23703 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 355, 46), 'ct')
    # Obtaining the member 'c_ulong' of a type (line 355)
    c_ulong_23704 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 355, 46), ct_23703, 'c_ulong')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 355, 10), tuple_23696, c_ulong_23704)
    # Adding element type (line 355)
    # Getting the type of 'ct' (line 355)
    ct_23705 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 355, 58), 'ct')
    # Obtaining the member 'c_ulonglong' of a type (line 355)
    c_ulonglong_23706 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 355, 58), ct_23705, 'c_ulonglong')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 355, 10), tuple_23696, c_ulonglong_23706)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 355, 9), tuple_23695, tuple_23696)
    # Adding element type (line 355)
    str_23707 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 355, 75), 'str', 'u')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 355, 9), tuple_23695, str_23707)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 353, 19), list_23681, tuple_23695)
    # Adding element type (line 353)
    
    # Obtaining an instance of the builtin type 'tuple' (line 356)
    tuple_23708 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 356, 9), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 356)
    # Adding element type (line 356)
    
    # Obtaining an instance of the builtin type 'tuple' (line 356)
    tuple_23709 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 356, 10), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 356)
    # Adding element type (line 356)
    # Getting the type of 'ct' (line 356)
    ct_23710 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 356, 10), 'ct')
    # Obtaining the member 'c_float' of a type (line 356)
    c_float_23711 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 356, 10), ct_23710, 'c_float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 356, 10), tuple_23709, c_float_23711)
    # Adding element type (line 356)
    # Getting the type of 'ct' (line 356)
    ct_23712 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 356, 22), 'ct')
    # Obtaining the member 'c_double' of a type (line 356)
    c_double_23713 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 356, 22), ct_23712, 'c_double')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 356, 10), tuple_23709, c_double_23713)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 356, 9), tuple_23708, tuple_23709)
    # Adding element type (line 356)
    str_23714 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 356, 36), 'str', 'f')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 356, 9), tuple_23708, str_23714)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 353, 19), list_23681, tuple_23708)
    
    # Assigning a type to the variable 'simple_types' (line 353)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 353, 4), 'simple_types', list_23681)
    
    # Getting the type of 'simple_types' (line 360)
    simple_types_23715 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 360, 23), 'simple_types')
    # Testing the type of a for loop iterable (line 360)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 360, 4), simple_types_23715)
    # Getting the type of the for loop variable (line 360)
    for_loop_var_23716 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 360, 4), simple_types_23715)
    # Assigning a type to the variable 'types' (line 360)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 360, 4), 'types', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 360, 4), for_loop_var_23716))
    # Assigning a type to the variable 'code' (line 360)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 360, 4), 'code', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 360, 4), for_loop_var_23716))
    # SSA begins for a for statement (line 360)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Getting the type of 'types' (line 361)
    types_23717 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 361, 18), 'types')
    # Testing the type of a for loop iterable (line 361)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 361, 8), types_23717)
    # Getting the type of the for loop variable (line 361)
    for_loop_var_23718 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 361, 8), types_23717)
    # Assigning a type to the variable 'tp' (line 361)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 361, 8), 'tp', for_loop_var_23718)
    # SSA begins for a for statement (line 361)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Call to prep_simple(...): (line 362)
    # Processing the call arguments (line 362)
    # Getting the type of 'tp' (line 362)
    tp_23720 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 362, 24), 'tp', False)
    str_23721 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 362, 28), 'str', '%c%d')
    
    # Obtaining an instance of the builtin type 'tuple' (line 362)
    tuple_23722 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 362, 38), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 362)
    # Adding element type (line 362)
    # Getting the type of 'code' (line 362)
    code_23723 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 362, 38), 'code', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 362, 38), tuple_23722, code_23723)
    # Adding element type (line 362)
    
    # Call to sizeof(...): (line 362)
    # Processing the call arguments (line 362)
    # Getting the type of 'tp' (line 362)
    tp_23726 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 362, 54), 'tp', False)
    # Processing the call keyword arguments (line 362)
    kwargs_23727 = {}
    # Getting the type of 'ct' (line 362)
    ct_23724 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 362, 44), 'ct', False)
    # Obtaining the member 'sizeof' of a type (line 362)
    sizeof_23725 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 362, 44), ct_23724, 'sizeof')
    # Calling sizeof(args, kwargs) (line 362)
    sizeof_call_result_23728 = invoke(stypy.reporting.localization.Localization(__file__, 362, 44), sizeof_23725, *[tp_23726], **kwargs_23727)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 362, 38), tuple_23722, sizeof_call_result_23728)
    
    # Applying the binary operator '%' (line 362)
    result_mod_23729 = python_operator(stypy.reporting.localization.Localization(__file__, 362, 28), '%', str_23721, tuple_23722)
    
    # Processing the call keyword arguments (line 362)
    kwargs_23730 = {}
    # Getting the type of 'prep_simple' (line 362)
    prep_simple_23719 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 362, 12), 'prep_simple', False)
    # Calling prep_simple(args, kwargs) (line 362)
    prep_simple_call_result_23731 = invoke(stypy.reporting.localization.Localization(__file__, 362, 12), prep_simple_23719, *[tp_23720, result_mod_23729], **kwargs_23730)
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 367):
    
    # Assigning a Call to a Name (line 367):
    
    # Call to type(...): (line 367)
    # Processing the call arguments (line 367)
    # Getting the type of 'ct' (line 367)
    ct_23733 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 367, 23), 'ct', False)
    # Obtaining the member 'c_int' of a type (line 367)
    c_int_23734 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 367, 23), ct_23733, 'c_int')
    int_23735 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 367, 34), 'int')
    # Applying the binary operator '*' (line 367)
    result_mul_23736 = python_operator(stypy.reporting.localization.Localization(__file__, 367, 23), '*', c_int_23734, int_23735)
    
    # Processing the call keyword arguments (line 367)
    kwargs_23737 = {}
    # Getting the type of 'type' (line 367)
    type_23732 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 367, 18), 'type', False)
    # Calling type(args, kwargs) (line 367)
    type_call_result_23738 = invoke(stypy.reporting.localization.Localization(__file__, 367, 18), type_23732, *[result_mul_23736], **kwargs_23737)
    
    # Assigning a type to the variable '_ARRAY_TYPE' (line 367)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 367, 4), '_ARRAY_TYPE', type_call_result_23738)

    @norecursion
    def prep_array(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'prep_array'
        module_type_store = module_type_store.open_function_context('prep_array', 369, 4, False)
        
        # Passed parameters checking function
        prep_array.stypy_localization = localization
        prep_array.stypy_type_of_self = None
        prep_array.stypy_type_store = module_type_store
        prep_array.stypy_function_name = 'prep_array'
        prep_array.stypy_param_names_list = ['array_type']
        prep_array.stypy_varargs_param_name = None
        prep_array.stypy_kwargs_param_name = None
        prep_array.stypy_call_defaults = defaults
        prep_array.stypy_call_varargs = varargs
        prep_array.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, 'prep_array', ['array_type'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'prep_array', localization, ['array_type'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'prep_array(...)' code ##################

        str_23739 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 372, (-1)), 'str', 'Given a ctypes array type, construct and attach an\n        __array_interface__ property to it if it does not yet have one.\n        ')
        
        
        # SSA begins for try-except statement (line 373)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
        # Getting the type of 'array_type' (line 373)
        array_type_23740 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 373, 13), 'array_type')
        # Obtaining the member '__array_interface__' of a type (line 373)
        array_interface___23741 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 373, 13), array_type_23740, '__array_interface__')
        # SSA branch for the except part of a try statement (line 373)
        # SSA branch for the except 'AttributeError' branch of a try statement (line 373)
        module_type_store.open_ssa_branch('except')
        pass
        # SSA branch for the else branch of a try statement (line 373)
        module_type_store.open_ssa_branch('except else')
        # Assigning a type to the variable 'stypy_return_type' (line 375)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 375, 14), 'stypy_return_type', types.NoneType)
        # SSA join for try-except statement (line 373)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a List to a Name (line 377):
        
        # Assigning a List to a Name (line 377):
        
        # Obtaining an instance of the builtin type 'list' (line 377)
        list_23742 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 377, 16), 'list')
        # Adding type elements to the builtin type 'list' instance (line 377)
        
        # Assigning a type to the variable 'shape' (line 377)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 377, 8), 'shape', list_23742)
        
        # Assigning a Name to a Name (line 378):
        
        # Assigning a Name to a Name (line 378):
        # Getting the type of 'array_type' (line 378)
        array_type_23743 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 378, 13), 'array_type')
        # Assigning a type to the variable 'ob' (line 378)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 378, 8), 'ob', array_type_23743)
        
        
        
        # Call to type(...): (line 379)
        # Processing the call arguments (line 379)
        # Getting the type of 'ob' (line 379)
        ob_23745 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 379, 19), 'ob', False)
        # Processing the call keyword arguments (line 379)
        kwargs_23746 = {}
        # Getting the type of 'type' (line 379)
        type_23744 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 379, 14), 'type', False)
        # Calling type(args, kwargs) (line 379)
        type_call_result_23747 = invoke(stypy.reporting.localization.Localization(__file__, 379, 14), type_23744, *[ob_23745], **kwargs_23746)
        
        # Getting the type of '_ARRAY_TYPE' (line 379)
        _ARRAY_TYPE_23748 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 379, 26), '_ARRAY_TYPE')
        # Applying the binary operator 'is' (line 379)
        result_is__23749 = python_operator(stypy.reporting.localization.Localization(__file__, 379, 14), 'is', type_call_result_23747, _ARRAY_TYPE_23748)
        
        # Testing the type of an if condition (line 379)
        is_suitable_condition(stypy.reporting.localization.Localization(__file__, 379, 8), result_is__23749)
        # SSA begins for while statement (line 379)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'while loop')
        
        # Call to append(...): (line 380)
        # Processing the call arguments (line 380)
        # Getting the type of 'ob' (line 380)
        ob_23752 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 380, 25), 'ob', False)
        # Obtaining the member '_length_' of a type (line 380)
        _length__23753 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 380, 25), ob_23752, '_length_')
        # Processing the call keyword arguments (line 380)
        kwargs_23754 = {}
        # Getting the type of 'shape' (line 380)
        shape_23750 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 380, 12), 'shape', False)
        # Obtaining the member 'append' of a type (line 380)
        append_23751 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 380, 12), shape_23750, 'append')
        # Calling append(args, kwargs) (line 380)
        append_call_result_23755 = invoke(stypy.reporting.localization.Localization(__file__, 380, 12), append_23751, *[_length__23753], **kwargs_23754)
        
        
        # Assigning a Attribute to a Name (line 381):
        
        # Assigning a Attribute to a Name (line 381):
        # Getting the type of 'ob' (line 381)
        ob_23756 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 381, 17), 'ob')
        # Obtaining the member '_type_' of a type (line 381)
        _type__23757 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 381, 17), ob_23756, '_type_')
        # Assigning a type to the variable 'ob' (line 381)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 381, 12), 'ob', _type__23757)
        # SSA join for while statement (line 379)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Name (line 382):
        
        # Assigning a Call to a Name (line 382):
        
        # Call to tuple(...): (line 382)
        # Processing the call arguments (line 382)
        # Getting the type of 'shape' (line 382)
        shape_23759 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 382, 22), 'shape', False)
        # Processing the call keyword arguments (line 382)
        kwargs_23760 = {}
        # Getting the type of 'tuple' (line 382)
        tuple_23758 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 382, 16), 'tuple', False)
        # Calling tuple(args, kwargs) (line 382)
        tuple_call_result_23761 = invoke(stypy.reporting.localization.Localization(__file__, 382, 16), tuple_23758, *[shape_23759], **kwargs_23760)
        
        # Assigning a type to the variable 'shape' (line 382)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 382, 8), 'shape', tuple_call_result_23761)
        
        # Assigning a Attribute to a Name (line 383):
        
        # Assigning a Attribute to a Name (line 383):
        
        # Call to ob(...): (line 383)
        # Processing the call keyword arguments (line 383)
        kwargs_23763 = {}
        # Getting the type of 'ob' (line 383)
        ob_23762 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 383, 13), 'ob', False)
        # Calling ob(args, kwargs) (line 383)
        ob_call_result_23764 = invoke(stypy.reporting.localization.Localization(__file__, 383, 13), ob_23762, *[], **kwargs_23763)
        
        # Obtaining the member '__array_interface__' of a type (line 383)
        array_interface___23765 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 383, 13), ob_call_result_23764, '__array_interface__')
        # Assigning a type to the variable 'ai' (line 383)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 383, 8), 'ai', array_interface___23765)
        
        # Assigning a Subscript to a Name (line 384):
        
        # Assigning a Subscript to a Name (line 384):
        
        # Obtaining the type of the subscript
        str_23766 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 384, 19), 'str', 'descr')
        # Getting the type of 'ai' (line 384)
        ai_23767 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 384, 16), 'ai')
        # Obtaining the member '__getitem__' of a type (line 384)
        getitem___23768 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 384, 16), ai_23767, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 384)
        subscript_call_result_23769 = invoke(stypy.reporting.localization.Localization(__file__, 384, 16), getitem___23768, str_23766)
        
        # Assigning a type to the variable 'descr' (line 384)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 384, 8), 'descr', subscript_call_result_23769)
        
        # Assigning a Subscript to a Name (line 385):
        
        # Assigning a Subscript to a Name (line 385):
        
        # Obtaining the type of the subscript
        str_23770 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 385, 21), 'str', 'typestr')
        # Getting the type of 'ai' (line 385)
        ai_23771 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 385, 18), 'ai')
        # Obtaining the member '__getitem__' of a type (line 385)
        getitem___23772 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 385, 18), ai_23771, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 385)
        subscript_call_result_23773 = invoke(stypy.reporting.localization.Localization(__file__, 385, 18), getitem___23772, str_23770)
        
        # Assigning a type to the variable 'typestr' (line 385)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 385, 8), 'typestr', subscript_call_result_23773)

        @norecursion
        def __array_interface__(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '__array_interface__'
            module_type_store = module_type_store.open_function_context('__array_interface__', 387, 8, False)
            
            # Passed parameters checking function
            __array_interface__.stypy_localization = localization
            __array_interface__.stypy_type_of_self = None
            __array_interface__.stypy_type_store = module_type_store
            __array_interface__.stypy_function_name = '__array_interface__'
            __array_interface__.stypy_param_names_list = ['self']
            __array_interface__.stypy_varargs_param_name = None
            __array_interface__.stypy_kwargs_param_name = None
            __array_interface__.stypy_call_defaults = defaults
            __array_interface__.stypy_call_varargs = varargs
            __array_interface__.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, '__array_interface__', ['self'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, '__array_interface__', localization, ['self'], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of '__array_interface__(...)' code ##################

            
            # Obtaining an instance of the builtin type 'dict' (line 388)
            dict_23774 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 388, 19), 'dict')
            # Adding type elements to the builtin type 'dict' instance (line 388)
            # Adding element type (key, value) (line 388)
            str_23775 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 388, 20), 'str', 'descr')
            # Getting the type of 'descr' (line 388)
            descr_23776 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 388, 29), 'descr')
            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 388, 19), dict_23774, (str_23775, descr_23776))
            # Adding element type (key, value) (line 388)
            str_23777 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 389, 20), 'str', '__ref')
            # Getting the type of 'self' (line 389)
            self_23778 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 389, 29), 'self')
            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 388, 19), dict_23774, (str_23777, self_23778))
            # Adding element type (key, value) (line 388)
            str_23779 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 390, 20), 'str', 'strides')
            # Getting the type of 'None' (line 390)
            None_23780 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 390, 31), 'None')
            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 388, 19), dict_23774, (str_23779, None_23780))
            # Adding element type (key, value) (line 388)
            str_23781 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 391, 20), 'str', 'shape')
            # Getting the type of 'shape' (line 391)
            shape_23782 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 391, 29), 'shape')
            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 388, 19), dict_23774, (str_23781, shape_23782))
            # Adding element type (key, value) (line 388)
            str_23783 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 392, 20), 'str', 'version')
            int_23784 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 392, 31), 'int')
            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 388, 19), dict_23774, (str_23783, int_23784))
            # Adding element type (key, value) (line 388)
            str_23785 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 393, 20), 'str', 'typestr')
            # Getting the type of 'typestr' (line 393)
            typestr_23786 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 393, 31), 'typestr')
            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 388, 19), dict_23774, (str_23785, typestr_23786))
            # Adding element type (key, value) (line 388)
            str_23787 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 394, 20), 'str', 'data')
            
            # Obtaining an instance of the builtin type 'tuple' (line 394)
            tuple_23788 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 394, 29), 'tuple')
            # Adding type elements to the builtin type 'tuple' instance (line 394)
            # Adding element type (line 394)
            
            # Call to addressof(...): (line 394)
            # Processing the call arguments (line 394)
            # Getting the type of 'self' (line 394)
            self_23791 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 394, 42), 'self', False)
            # Processing the call keyword arguments (line 394)
            kwargs_23792 = {}
            # Getting the type of 'ct' (line 394)
            ct_23789 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 394, 29), 'ct', False)
            # Obtaining the member 'addressof' of a type (line 394)
            addressof_23790 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 394, 29), ct_23789, 'addressof')
            # Calling addressof(args, kwargs) (line 394)
            addressof_call_result_23793 = invoke(stypy.reporting.localization.Localization(__file__, 394, 29), addressof_23790, *[self_23791], **kwargs_23792)
            
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 394, 29), tuple_23788, addressof_call_result_23793)
            # Adding element type (line 394)
            # Getting the type of 'False' (line 394)
            False_23794 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 394, 49), 'False')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 394, 29), tuple_23788, False_23794)
            
            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 388, 19), dict_23774, (str_23787, tuple_23788))
            
            # Assigning a type to the variable 'stypy_return_type' (line 388)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 388, 12), 'stypy_return_type', dict_23774)
            
            # ################# End of '__array_interface__(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function '__array_interface__' in the type store
            # Getting the type of 'stypy_return_type' (line 387)
            stypy_return_type_23795 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 387, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_23795)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function '__array_interface__'
            return stypy_return_type_23795

        # Assigning a type to the variable '__array_interface__' (line 387)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 387, 8), '__array_interface__', __array_interface__)
        
        # Assigning a Call to a Attribute (line 397):
        
        # Assigning a Call to a Attribute (line 397):
        
        # Call to property(...): (line 397)
        # Processing the call arguments (line 397)
        # Getting the type of '__array_interface__' (line 397)
        array_interface___23797 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 397, 50), '__array_interface__', False)
        # Processing the call keyword arguments (line 397)
        kwargs_23798 = {}
        # Getting the type of 'property' (line 397)
        property_23796 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 397, 41), 'property', False)
        # Calling property(args, kwargs) (line 397)
        property_call_result_23799 = invoke(stypy.reporting.localization.Localization(__file__, 397, 41), property_23796, *[array_interface___23797], **kwargs_23798)
        
        # Getting the type of 'array_type' (line 397)
        array_type_23800 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 397, 8), 'array_type')
        # Setting the type of the member '__array_interface__' of a type (line 397)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 397, 8), array_type_23800, '__array_interface__', property_call_result_23799)
        
        # ################# End of 'prep_array(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'prep_array' in the type store
        # Getting the type of 'stypy_return_type' (line 369)
        stypy_return_type_23801 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 369, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_23801)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'prep_array'
        return stypy_return_type_23801

    # Assigning a type to the variable 'prep_array' (line 369)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 369, 4), 'prep_array', prep_array)

    @norecursion
    def prep_pointer(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'prep_pointer'
        module_type_store = module_type_store.open_function_context('prep_pointer', 399, 4, False)
        
        # Passed parameters checking function
        prep_pointer.stypy_localization = localization
        prep_pointer.stypy_type_of_self = None
        prep_pointer.stypy_type_store = module_type_store
        prep_pointer.stypy_function_name = 'prep_pointer'
        prep_pointer.stypy_param_names_list = ['pointer_obj', 'shape']
        prep_pointer.stypy_varargs_param_name = None
        prep_pointer.stypy_kwargs_param_name = None
        prep_pointer.stypy_call_defaults = defaults
        prep_pointer.stypy_call_varargs = varargs
        prep_pointer.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, 'prep_pointer', ['pointer_obj', 'shape'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'prep_pointer', localization, ['pointer_obj', 'shape'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'prep_pointer(...)' code ##################

        str_23802 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 403, (-1)), 'str', 'Given a ctypes pointer object, construct and\n        attach an __array_interface__ property to it if it does not\n        yet have one.\n        ')
        
        
        # SSA begins for try-except statement (line 404)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
        # Getting the type of 'pointer_obj' (line 404)
        pointer_obj_23803 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 404, 13), 'pointer_obj')
        # Obtaining the member '__array_interface__' of a type (line 404)
        array_interface___23804 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 404, 13), pointer_obj_23803, '__array_interface__')
        # SSA branch for the except part of a try statement (line 404)
        # SSA branch for the except 'AttributeError' branch of a try statement (line 404)
        module_type_store.open_ssa_branch('except')
        pass
        # SSA branch for the else branch of a try statement (line 404)
        module_type_store.open_ssa_branch('except else')
        # Assigning a type to the variable 'stypy_return_type' (line 406)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 406, 14), 'stypy_return_type', types.NoneType)
        # SSA join for try-except statement (line 404)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Attribute to a Name (line 408):
        
        # Assigning a Attribute to a Name (line 408):
        # Getting the type of 'pointer_obj' (line 408)
        pointer_obj_23805 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 408, 19), 'pointer_obj')
        # Obtaining the member 'contents' of a type (line 408)
        contents_23806 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 408, 19), pointer_obj_23805, 'contents')
        # Assigning a type to the variable 'contents' (line 408)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 408, 8), 'contents', contents_23806)
        
        # Assigning a Call to a Name (line 409):
        
        # Assigning a Call to a Name (line 409):
        
        # Call to _dtype(...): (line 409)
        # Processing the call arguments (line 409)
        
        # Call to type(...): (line 409)
        # Processing the call arguments (line 409)
        # Getting the type of 'contents' (line 409)
        contents_23809 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 409, 28), 'contents', False)
        # Processing the call keyword arguments (line 409)
        kwargs_23810 = {}
        # Getting the type of 'type' (line 409)
        type_23808 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 409, 23), 'type', False)
        # Calling type(args, kwargs) (line 409)
        type_call_result_23811 = invoke(stypy.reporting.localization.Localization(__file__, 409, 23), type_23808, *[contents_23809], **kwargs_23810)
        
        # Processing the call keyword arguments (line 409)
        kwargs_23812 = {}
        # Getting the type of '_dtype' (line 409)
        _dtype_23807 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 409, 16), '_dtype', False)
        # Calling _dtype(args, kwargs) (line 409)
        _dtype_call_result_23813 = invoke(stypy.reporting.localization.Localization(__file__, 409, 16), _dtype_23807, *[type_call_result_23811], **kwargs_23812)
        
        # Assigning a type to the variable 'dtype' (line 409)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 409, 8), 'dtype', _dtype_call_result_23813)
        
        # Assigning a Dict to a Name (line 411):
        
        # Assigning a Dict to a Name (line 411):
        
        # Obtaining an instance of the builtin type 'dict' (line 411)
        dict_23814 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 411, 16), 'dict')
        # Adding type elements to the builtin type 'dict' instance (line 411)
        # Adding element type (key, value) (line 411)
        str_23815 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 411, 17), 'str', 'version')
        int_23816 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 411, 28), 'int')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 411, 16), dict_23814, (str_23815, int_23816))
        # Adding element type (key, value) (line 411)
        str_23817 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 412, 17), 'str', 'typestr')
        # Getting the type of 'dtype' (line 412)
        dtype_23818 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 412, 28), 'dtype')
        # Obtaining the member 'str' of a type (line 412)
        str_23819 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 412, 28), dtype_23818, 'str')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 411, 16), dict_23814, (str_23817, str_23819))
        # Adding element type (key, value) (line 411)
        str_23820 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 413, 17), 'str', 'data')
        
        # Obtaining an instance of the builtin type 'tuple' (line 413)
        tuple_23821 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 413, 26), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 413)
        # Adding element type (line 413)
        
        # Call to addressof(...): (line 413)
        # Processing the call arguments (line 413)
        # Getting the type of 'contents' (line 413)
        contents_23824 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 413, 39), 'contents', False)
        # Processing the call keyword arguments (line 413)
        kwargs_23825 = {}
        # Getting the type of 'ct' (line 413)
        ct_23822 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 413, 26), 'ct', False)
        # Obtaining the member 'addressof' of a type (line 413)
        addressof_23823 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 413, 26), ct_23822, 'addressof')
        # Calling addressof(args, kwargs) (line 413)
        addressof_call_result_23826 = invoke(stypy.reporting.localization.Localization(__file__, 413, 26), addressof_23823, *[contents_23824], **kwargs_23825)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 413, 26), tuple_23821, addressof_call_result_23826)
        # Adding element type (line 413)
        # Getting the type of 'False' (line 413)
        False_23827 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 413, 50), 'False')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 413, 26), tuple_23821, False_23827)
        
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 411, 16), dict_23814, (str_23820, tuple_23821))
        # Adding element type (key, value) (line 411)
        str_23828 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 414, 17), 'str', 'shape')
        # Getting the type of 'shape' (line 414)
        shape_23829 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 414, 26), 'shape')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 411, 16), dict_23814, (str_23828, shape_23829))
        
        # Assigning a type to the variable 'inter' (line 411)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 411, 8), 'inter', dict_23814)
        
        # Assigning a Name to a Attribute (line 416):
        
        # Assigning a Name to a Attribute (line 416):
        # Getting the type of 'inter' (line 416)
        inter_23830 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 416, 42), 'inter')
        # Getting the type of 'pointer_obj' (line 416)
        pointer_obj_23831 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 416, 8), 'pointer_obj')
        # Setting the type of the member '__array_interface__' of a type (line 416)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 416, 8), pointer_obj_23831, '__array_interface__', inter_23830)
        
        # ################# End of 'prep_pointer(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'prep_pointer' in the type store
        # Getting the type of 'stypy_return_type' (line 399)
        stypy_return_type_23832 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 399, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_23832)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'prep_pointer'
        return stypy_return_type_23832

    # Assigning a type to the variable 'prep_pointer' (line 399)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 399, 4), 'prep_pointer', prep_pointer)

    @norecursion
    def as_array(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 421)
        None_23833 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 421, 28), 'None')
        defaults = [None_23833]
        # Create a new context for function 'as_array'
        module_type_store = module_type_store.open_function_context('as_array', 421, 4, False)
        
        # Passed parameters checking function
        as_array.stypy_localization = localization
        as_array.stypy_type_of_self = None
        as_array.stypy_type_store = module_type_store
        as_array.stypy_function_name = 'as_array'
        as_array.stypy_param_names_list = ['obj', 'shape']
        as_array.stypy_varargs_param_name = None
        as_array.stypy_kwargs_param_name = None
        as_array.stypy_call_defaults = defaults
        as_array.stypy_call_varargs = varargs
        as_array.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, 'as_array', ['obj', 'shape'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'as_array', localization, ['obj', 'shape'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'as_array(...)' code ##################

        str_23834 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 427, (-1)), 'str', 'Create a numpy array from a ctypes array or a ctypes POINTER.\n        The numpy array shares the memory with the ctypes object.\n\n        The size parameter must be given if converting from a ctypes POINTER.\n        The size parameter is ignored if converting from a ctypes array\n        ')
        
        # Assigning a Call to a Name (line 428):
        
        # Assigning a Call to a Name (line 428):
        
        # Call to type(...): (line 428)
        # Processing the call arguments (line 428)
        # Getting the type of 'obj' (line 428)
        obj_23836 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 428, 18), 'obj', False)
        # Processing the call keyword arguments (line 428)
        kwargs_23837 = {}
        # Getting the type of 'type' (line 428)
        type_23835 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 428, 13), 'type', False)
        # Calling type(args, kwargs) (line 428)
        type_call_result_23838 = invoke(stypy.reporting.localization.Localization(__file__, 428, 13), type_23835, *[obj_23836], **kwargs_23837)
        
        # Assigning a type to the variable 'tp' (line 428)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 428, 8), 'tp', type_call_result_23838)
        
        
        # SSA begins for try-except statement (line 429)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
        # Getting the type of 'tp' (line 429)
        tp_23839 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 429, 13), 'tp')
        # Obtaining the member '__array_interface__' of a type (line 429)
        array_interface___23840 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 429, 13), tp_23839, '__array_interface__')
        # SSA branch for the except part of a try statement (line 429)
        # SSA branch for the except 'AttributeError' branch of a try statement (line 429)
        module_type_store.open_ssa_branch('except')
        
        # Type idiom detected: calculating its left and rigth part (line 431)
        str_23841 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 431, 28), 'str', 'contents')
        # Getting the type of 'obj' (line 431)
        obj_23842 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 431, 23), 'obj')
        
        (may_be_23843, more_types_in_union_23844) = may_provide_member(str_23841, obj_23842)

        if may_be_23843:

            if more_types_in_union_23844:
                # Runtime conditional SSA (line 431)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            # Assigning a type to the variable 'obj' (line 431)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 431, 12), 'obj', remove_not_member_provider_from_union(obj_23842, 'contents'))
            
            # Call to prep_pointer(...): (line 432)
            # Processing the call arguments (line 432)
            # Getting the type of 'obj' (line 432)
            obj_23846 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 432, 29), 'obj', False)
            # Getting the type of 'shape' (line 432)
            shape_23847 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 432, 34), 'shape', False)
            # Processing the call keyword arguments (line 432)
            kwargs_23848 = {}
            # Getting the type of 'prep_pointer' (line 432)
            prep_pointer_23845 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 432, 16), 'prep_pointer', False)
            # Calling prep_pointer(args, kwargs) (line 432)
            prep_pointer_call_result_23849 = invoke(stypy.reporting.localization.Localization(__file__, 432, 16), prep_pointer_23845, *[obj_23846, shape_23847], **kwargs_23848)
            

            if more_types_in_union_23844:
                # Runtime conditional SSA for else branch (line 431)
                module_type_store.open_ssa_branch('idiom else')



        if ((not may_be_23843) or more_types_in_union_23844):
            # Assigning a type to the variable 'obj' (line 431)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 431, 12), 'obj', remove_member_provider_from_union(obj_23842, 'contents'))
            
            # Call to prep_array(...): (line 434)
            # Processing the call arguments (line 434)
            # Getting the type of 'tp' (line 434)
            tp_23851 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 434, 27), 'tp', False)
            # Processing the call keyword arguments (line 434)
            kwargs_23852 = {}
            # Getting the type of 'prep_array' (line 434)
            prep_array_23850 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 434, 16), 'prep_array', False)
            # Calling prep_array(args, kwargs) (line 434)
            prep_array_call_result_23853 = invoke(stypy.reporting.localization.Localization(__file__, 434, 16), prep_array_23850, *[tp_23851], **kwargs_23852)
            

            if (may_be_23843 and more_types_in_union_23844):
                # SSA join for if statement (line 431)
                module_type_store = module_type_store.join_ssa_context()


        
        # SSA join for try-except statement (line 429)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to array(...): (line 435)
        # Processing the call arguments (line 435)
        # Getting the type of 'obj' (line 435)
        obj_23855 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 435, 21), 'obj', False)
        # Processing the call keyword arguments (line 435)
        # Getting the type of 'False' (line 435)
        False_23856 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 435, 31), 'False', False)
        keyword_23857 = False_23856
        kwargs_23858 = {'copy': keyword_23857}
        # Getting the type of 'array' (line 435)
        array_23854 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 435, 15), 'array', False)
        # Calling array(args, kwargs) (line 435)
        array_call_result_23859 = invoke(stypy.reporting.localization.Localization(__file__, 435, 15), array_23854, *[obj_23855], **kwargs_23858)
        
        # Assigning a type to the variable 'stypy_return_type' (line 435)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 435, 8), 'stypy_return_type', array_call_result_23859)
        
        # ################# End of 'as_array(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'as_array' in the type store
        # Getting the type of 'stypy_return_type' (line 421)
        stypy_return_type_23860 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 421, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_23860)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'as_array'
        return stypy_return_type_23860

    # Assigning a type to the variable 'as_array' (line 421)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 421, 4), 'as_array', as_array)

    @norecursion
    def as_ctypes(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'as_ctypes'
        module_type_store = module_type_store.open_function_context('as_ctypes', 437, 4, False)
        
        # Passed parameters checking function
        as_ctypes.stypy_localization = localization
        as_ctypes.stypy_type_of_self = None
        as_ctypes.stypy_type_store = module_type_store
        as_ctypes.stypy_function_name = 'as_ctypes'
        as_ctypes.stypy_param_names_list = ['obj']
        as_ctypes.stypy_varargs_param_name = None
        as_ctypes.stypy_kwargs_param_name = None
        as_ctypes.stypy_call_defaults = defaults
        as_ctypes.stypy_call_varargs = varargs
        as_ctypes.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, 'as_ctypes', ['obj'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'as_ctypes', localization, ['obj'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'as_ctypes(...)' code ##################

        str_23861 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 439, (-1)), 'str', 'Create and return a ctypes object from a numpy array.  Actually\n        anything that exposes the __array_interface__ is accepted.')
        
        # Assigning a Attribute to a Name (line 440):
        
        # Assigning a Attribute to a Name (line 440):
        # Getting the type of 'obj' (line 440)
        obj_23862 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 440, 13), 'obj')
        # Obtaining the member '__array_interface__' of a type (line 440)
        array_interface___23863 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 440, 13), obj_23862, '__array_interface__')
        # Assigning a type to the variable 'ai' (line 440)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 440, 8), 'ai', array_interface___23863)
        
        
        # Obtaining the type of the subscript
        str_23864 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 441, 14), 'str', 'strides')
        # Getting the type of 'ai' (line 441)
        ai_23865 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 441, 11), 'ai')
        # Obtaining the member '__getitem__' of a type (line 441)
        getitem___23866 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 441, 11), ai_23865, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 441)
        subscript_call_result_23867 = invoke(stypy.reporting.localization.Localization(__file__, 441, 11), getitem___23866, str_23864)
        
        # Testing the type of an if condition (line 441)
        if_condition_23868 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 441, 8), subscript_call_result_23867)
        # Assigning a type to the variable 'if_condition_23868' (line 441)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 441, 8), 'if_condition_23868', if_condition_23868)
        # SSA begins for if statement (line 441)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to TypeError(...): (line 442)
        # Processing the call arguments (line 442)
        str_23870 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 442, 28), 'str', 'strided arrays not supported')
        # Processing the call keyword arguments (line 442)
        kwargs_23871 = {}
        # Getting the type of 'TypeError' (line 442)
        TypeError_23869 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 442, 18), 'TypeError', False)
        # Calling TypeError(args, kwargs) (line 442)
        TypeError_call_result_23872 = invoke(stypy.reporting.localization.Localization(__file__, 442, 18), TypeError_23869, *[str_23870], **kwargs_23871)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 442, 12), TypeError_call_result_23872, 'raise parameter', BaseException)
        # SSA join for if statement (line 441)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        
        # Obtaining the type of the subscript
        str_23873 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 443, 14), 'str', 'version')
        # Getting the type of 'ai' (line 443)
        ai_23874 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 443, 11), 'ai')
        # Obtaining the member '__getitem__' of a type (line 443)
        getitem___23875 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 443, 11), ai_23874, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 443)
        subscript_call_result_23876 = invoke(stypy.reporting.localization.Localization(__file__, 443, 11), getitem___23875, str_23873)
        
        int_23877 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 443, 28), 'int')
        # Applying the binary operator '!=' (line 443)
        result_ne_23878 = python_operator(stypy.reporting.localization.Localization(__file__, 443, 11), '!=', subscript_call_result_23876, int_23877)
        
        # Testing the type of an if condition (line 443)
        if_condition_23879 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 443, 8), result_ne_23878)
        # Assigning a type to the variable 'if_condition_23879' (line 443)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 443, 8), 'if_condition_23879', if_condition_23879)
        # SSA begins for if statement (line 443)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to TypeError(...): (line 444)
        # Processing the call arguments (line 444)
        str_23881 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 444, 28), 'str', 'only __array_interface__ version 3 supported')
        # Processing the call keyword arguments (line 444)
        kwargs_23882 = {}
        # Getting the type of 'TypeError' (line 444)
        TypeError_23880 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 444, 18), 'TypeError', False)
        # Calling TypeError(args, kwargs) (line 444)
        TypeError_call_result_23883 = invoke(stypy.reporting.localization.Localization(__file__, 444, 18), TypeError_23880, *[str_23881], **kwargs_23882)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 444, 12), TypeError_call_result_23883, 'raise parameter', BaseException)
        # SSA join for if statement (line 443)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Subscript to a Tuple (line 445):
        
        # Assigning a Subscript to a Name (line 445):
        
        # Obtaining the type of the subscript
        int_23884 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 445, 8), 'int')
        
        # Obtaining the type of the subscript
        str_23885 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 445, 28), 'str', 'data')
        # Getting the type of 'ai' (line 445)
        ai_23886 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 445, 25), 'ai')
        # Obtaining the member '__getitem__' of a type (line 445)
        getitem___23887 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 445, 25), ai_23886, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 445)
        subscript_call_result_23888 = invoke(stypy.reporting.localization.Localization(__file__, 445, 25), getitem___23887, str_23885)
        
        # Obtaining the member '__getitem__' of a type (line 445)
        getitem___23889 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 445, 8), subscript_call_result_23888, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 445)
        subscript_call_result_23890 = invoke(stypy.reporting.localization.Localization(__file__, 445, 8), getitem___23889, int_23884)
        
        # Assigning a type to the variable 'tuple_var_assignment_23142' (line 445)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 445, 8), 'tuple_var_assignment_23142', subscript_call_result_23890)
        
        # Assigning a Subscript to a Name (line 445):
        
        # Obtaining the type of the subscript
        int_23891 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 445, 8), 'int')
        
        # Obtaining the type of the subscript
        str_23892 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 445, 28), 'str', 'data')
        # Getting the type of 'ai' (line 445)
        ai_23893 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 445, 25), 'ai')
        # Obtaining the member '__getitem__' of a type (line 445)
        getitem___23894 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 445, 25), ai_23893, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 445)
        subscript_call_result_23895 = invoke(stypy.reporting.localization.Localization(__file__, 445, 25), getitem___23894, str_23892)
        
        # Obtaining the member '__getitem__' of a type (line 445)
        getitem___23896 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 445, 8), subscript_call_result_23895, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 445)
        subscript_call_result_23897 = invoke(stypy.reporting.localization.Localization(__file__, 445, 8), getitem___23896, int_23891)
        
        # Assigning a type to the variable 'tuple_var_assignment_23143' (line 445)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 445, 8), 'tuple_var_assignment_23143', subscript_call_result_23897)
        
        # Assigning a Name to a Name (line 445):
        # Getting the type of 'tuple_var_assignment_23142' (line 445)
        tuple_var_assignment_23142_23898 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 445, 8), 'tuple_var_assignment_23142')
        # Assigning a type to the variable 'addr' (line 445)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 445, 8), 'addr', tuple_var_assignment_23142_23898)
        
        # Assigning a Name to a Name (line 445):
        # Getting the type of 'tuple_var_assignment_23143' (line 445)
        tuple_var_assignment_23143_23899 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 445, 8), 'tuple_var_assignment_23143')
        # Assigning a type to the variable 'readonly' (line 445)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 445, 14), 'readonly', tuple_var_assignment_23143_23899)
        
        # Getting the type of 'readonly' (line 446)
        readonly_23900 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 446, 11), 'readonly')
        # Testing the type of an if condition (line 446)
        if_condition_23901 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 446, 8), readonly_23900)
        # Assigning a type to the variable 'if_condition_23901' (line 446)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 446, 8), 'if_condition_23901', if_condition_23901)
        # SSA begins for if statement (line 446)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to TypeError(...): (line 447)
        # Processing the call arguments (line 447)
        str_23903 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 447, 28), 'str', 'readonly arrays unsupported')
        # Processing the call keyword arguments (line 447)
        kwargs_23904 = {}
        # Getting the type of 'TypeError' (line 447)
        TypeError_23902 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 447, 18), 'TypeError', False)
        # Calling TypeError(args, kwargs) (line 447)
        TypeError_call_result_23905 = invoke(stypy.reporting.localization.Localization(__file__, 447, 18), TypeError_23902, *[str_23903], **kwargs_23904)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 447, 12), TypeError_call_result_23905, 'raise parameter', BaseException)
        # SSA join for if statement (line 446)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Subscript to a Name (line 448):
        
        # Assigning a Subscript to a Name (line 448):
        
        # Obtaining the type of the subscript
        
        # Obtaining the type of the subscript
        str_23906 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 448, 27), 'str', 'typestr')
        # Getting the type of 'ai' (line 448)
        ai_23907 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 448, 24), 'ai')
        # Obtaining the member '__getitem__' of a type (line 448)
        getitem___23908 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 448, 24), ai_23907, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 448)
        subscript_call_result_23909 = invoke(stypy.reporting.localization.Localization(__file__, 448, 24), getitem___23908, str_23906)
        
        # Getting the type of '_typecodes' (line 448)
        _typecodes_23910 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 448, 13), '_typecodes')
        # Obtaining the member '__getitem__' of a type (line 448)
        getitem___23911 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 448, 13), _typecodes_23910, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 448)
        subscript_call_result_23912 = invoke(stypy.reporting.localization.Localization(__file__, 448, 13), getitem___23911, subscript_call_result_23909)
        
        # Assigning a type to the variable 'tp' (line 448)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 448, 8), 'tp', subscript_call_result_23912)
        
        
        # Obtaining the type of the subscript
        int_23913 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 449, 33), 'int')
        slice_23914 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 449, 19), None, None, int_23913)
        
        # Obtaining the type of the subscript
        str_23915 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 449, 22), 'str', 'shape')
        # Getting the type of 'ai' (line 449)
        ai_23916 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 449, 19), 'ai')
        # Obtaining the member '__getitem__' of a type (line 449)
        getitem___23917 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 449, 19), ai_23916, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 449)
        subscript_call_result_23918 = invoke(stypy.reporting.localization.Localization(__file__, 449, 19), getitem___23917, str_23915)
        
        # Obtaining the member '__getitem__' of a type (line 449)
        getitem___23919 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 449, 19), subscript_call_result_23918, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 449)
        subscript_call_result_23920 = invoke(stypy.reporting.localization.Localization(__file__, 449, 19), getitem___23919, slice_23914)
        
        # Testing the type of a for loop iterable (line 449)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 449, 8), subscript_call_result_23920)
        # Getting the type of the for loop variable (line 449)
        for_loop_var_23921 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 449, 8), subscript_call_result_23920)
        # Assigning a type to the variable 'dim' (line 449)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 449, 8), 'dim', for_loop_var_23921)
        # SSA begins for a for statement (line 449)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a BinOp to a Name (line 450):
        
        # Assigning a BinOp to a Name (line 450):
        # Getting the type of 'tp' (line 450)
        tp_23922 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 450, 17), 'tp')
        # Getting the type of 'dim' (line 450)
        dim_23923 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 450, 22), 'dim')
        # Applying the binary operator '*' (line 450)
        result_mul_23924 = python_operator(stypy.reporting.localization.Localization(__file__, 450, 17), '*', tp_23922, dim_23923)
        
        # Assigning a type to the variable 'tp' (line 450)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 450, 12), 'tp', result_mul_23924)
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Name (line 451):
        
        # Assigning a Call to a Name (line 451):
        
        # Call to from_address(...): (line 451)
        # Processing the call arguments (line 451)
        # Getting the type of 'addr' (line 451)
        addr_23927 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 451, 33), 'addr', False)
        # Processing the call keyword arguments (line 451)
        kwargs_23928 = {}
        # Getting the type of 'tp' (line 451)
        tp_23925 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 451, 17), 'tp', False)
        # Obtaining the member 'from_address' of a type (line 451)
        from_address_23926 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 451, 17), tp_23925, 'from_address')
        # Calling from_address(args, kwargs) (line 451)
        from_address_call_result_23929 = invoke(stypy.reporting.localization.Localization(__file__, 451, 17), from_address_23926, *[addr_23927], **kwargs_23928)
        
        # Assigning a type to the variable 'result' (line 451)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 451, 8), 'result', from_address_call_result_23929)
        
        # Assigning a Name to a Attribute (line 452):
        
        # Assigning a Name to a Attribute (line 452):
        # Getting the type of 'ai' (line 452)
        ai_23930 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 452, 24), 'ai')
        # Getting the type of 'result' (line 452)
        result_23931 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 452, 8), 'result')
        # Setting the type of the member '__keep' of a type (line 452)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 452, 8), result_23931, '__keep', ai_23930)
        # Getting the type of 'result' (line 453)
        result_23932 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 453, 15), 'result')
        # Assigning a type to the variable 'stypy_return_type' (line 453)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 453, 8), 'stypy_return_type', result_23932)
        
        # ################# End of 'as_ctypes(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'as_ctypes' in the type store
        # Getting the type of 'stypy_return_type' (line 437)
        stypy_return_type_23933 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 437, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_23933)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'as_ctypes'
        return stypy_return_type_23933

    # Assigning a type to the variable 'as_ctypes' (line 437)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 437, 4), 'as_ctypes', as_ctypes)

    if more_types_in_union_23636:
        # SSA join for if statement (line 321)
        module_type_store = module_type_store.join_ssa_context()




# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
