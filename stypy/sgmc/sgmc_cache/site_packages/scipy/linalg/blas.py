
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: '''
2: Low-level BLAS functions (:mod:`scipy.linalg.blas`)
3: ===================================================
4: 
5: This module contains low-level functions from the BLAS library.
6: 
7: .. versionadded:: 0.12.0
8: 
9: .. warning::
10: 
11:    These functions do little to no error checking.
12:    It is possible to cause crashes by mis-using them,
13:    so prefer using the higher-level routines in `scipy.linalg`.
14: 
15: Finding functions
16: -----------------
17: 
18: .. autosummary::
19:    :toctree: generated/
20: 
21:    get_blas_funcs
22:    find_best_blas_type
23: 
24: BLAS Level 1 functions
25: ----------------------
26: 
27: .. autosummary::
28:    :toctree: generated/
29: 
30:    caxpy
31:    ccopy
32:    cdotc
33:    cdotu
34:    crotg
35:    cscal
36:    csrot
37:    csscal
38:    cswap
39:    dasum
40:    daxpy
41:    dcopy
42:    ddot
43:    dnrm2
44:    drot
45:    drotg
46:    drotm
47:    drotmg
48:    dscal
49:    dswap
50:    dzasum
51:    dznrm2
52:    icamax
53:    idamax
54:    isamax
55:    izamax
56:    sasum
57:    saxpy
58:    scasum
59:    scnrm2
60:    scopy
61:    sdot
62:    snrm2
63:    srot
64:    srotg
65:    srotm
66:    srotmg
67:    sscal
68:    sswap
69:    zaxpy
70:    zcopy
71:    zdotc
72:    zdotu
73:    zdrot
74:    zdscal
75:    zrotg
76:    zscal
77:    zswap
78: 
79: BLAS Level 2 functions
80: ----------------------
81: 
82: .. autosummary::
83:    :toctree: generated/
84: 
85:    sgbmv
86:    sgemv
87:    sger
88:    ssbmv
89:    sspr
90:    sspr2
91:    ssymv
92:    ssyr
93:    ssyr2
94:    stbmv
95:    stpsv
96:    strmv
97:    strsv
98:    dgbmv
99:    dgemv
100:    dger
101:    dsbmv
102:    dspr
103:    dspr2
104:    dsymv
105:    dsyr
106:    dsyr2
107:    dtbmv
108:    dtpsv
109:    dtrmv
110:    dtrsv
111:    cgbmv
112:    cgemv
113:    cgerc
114:    cgeru
115:    chbmv
116:    chemv
117:    cher
118:    cher2
119:    chpmv
120:    chpr
121:    chpr2
122:    ctbmv
123:    ctbsv
124:    ctpmv
125:    ctpsv
126:    ctrmv
127:    ctrsv
128:    csyr
129:    zgbmv
130:    zgemv
131:    zgerc
132:    zgeru
133:    zhbmv
134:    zhemv
135:    zher
136:    zher2
137:    zhpmv
138:    zhpr
139:    zhpr2
140:    ztbmv
141:    ztbsv
142:    ztpmv
143:    ztrmv
144:    ztrsv
145:    zsyr
146: 
147: BLAS Level 3 functions
148: ----------------------
149: 
150: .. autosummary::
151:    :toctree: generated/
152: 
153:    sgemm
154:    ssymm
155:    ssyr2k
156:    ssyrk
157:    strmm
158:    strsm
159:    dgemm
160:    dsymm
161:    dsyr2k
162:    dsyrk
163:    dtrmm
164:    dtrsm
165:    cgemm
166:    chemm
167:    cher2k
168:    cherk
169:    csymm
170:    csyr2k
171:    csyrk
172:    ctrmm
173:    ctrsm
174:    zgemm
175:    zhemm
176:    zher2k
177:    zherk
178:    zsymm
179:    zsyr2k
180:    zsyrk
181:    ztrmm
182:    ztrsm
183: 
184: '''
185: #
186: # Author: Pearu Peterson, March 2002
187: #         refactoring by Fabian Pedregosa, March 2010
188: #
189: 
190: from __future__ import division, print_function, absolute_import
191: 
192: __all__ = ['get_blas_funcs', 'find_best_blas_type']
193: 
194: import numpy as _np
195: 
196: from scipy.linalg import _fblas
197: try:
198:     from scipy.linalg import _cblas
199: except ImportError:
200:     _cblas = None
201: 
202: # Expose all functions (only fblas --- cblas is an implementation detail)
203: empty_module = None
204: from scipy.linalg._fblas import *
205: del empty_module
206: 
207: # 'd' will be default for 'i',..
208: _type_conv = {'f': 's', 'd': 'd', 'F': 'c', 'D': 'z', 'G': 'z'}
209: 
210: # some convenience alias for complex functions
211: _blas_alias = {'cnrm2': 'scnrm2', 'znrm2': 'dznrm2',
212:                'cdot': 'cdotc', 'zdot': 'zdotc',
213:                'cger': 'cgerc', 'zger': 'zgerc',
214:                'sdotc': 'sdot', 'sdotu': 'sdot',
215:                'ddotc': 'ddot', 'ddotu': 'ddot'}
216: 
217: 
218: def find_best_blas_type(arrays=(), dtype=None):
219:     '''Find best-matching BLAS/LAPACK type.
220: 
221:     Arrays are used to determine the optimal prefix of BLAS routines.
222: 
223:     Parameters
224:     ----------
225:     arrays : sequence of ndarrays, optional
226:         Arrays can be given to determine optimal prefix of BLAS
227:         routines. If not given, double-precision routines will be
228:         used, otherwise the most generic type in arrays will be used.
229:     dtype : str or dtype, optional
230:         Data-type specifier. Not used if `arrays` is non-empty.
231: 
232:     Returns
233:     -------
234:     prefix : str
235:         BLAS/LAPACK prefix character.
236:     dtype : dtype
237:         Inferred Numpy data type.
238:     prefer_fortran : bool
239:         Whether to prefer Fortran order routines over C order.
240: 
241:     Examples
242:     --------
243:     >>> import scipy.linalg.blas as bla
244:     >>> a = np.random.rand(10,15)
245:     >>> b = np.asfortranarray(a)  # Change the memory layout order
246:     >>> bla.find_best_blas_type((a,))
247:     ('d', dtype('float64'), False)
248:     >>> bla.find_best_blas_type((a*1j,))
249:     ('z', dtype('complex128'), False)
250:     >>> bla.find_best_blas_type((b,))
251:     ('d', dtype('float64'), True)
252: 
253:     '''
254:     dtype = _np.dtype(dtype)
255:     prefer_fortran = False
256: 
257:     if arrays:
258:         # use the most generic type in arrays
259:         dtypes = [ar.dtype for ar in arrays]
260:         dtype = _np.find_common_type(dtypes, ())
261:         try:
262:             index = dtypes.index(dtype)
263:         except ValueError:
264:             index = 0
265:         if arrays[index].flags['FORTRAN']:
266:             # prefer Fortran for leading array with column major order
267:             prefer_fortran = True
268: 
269:     prefix = _type_conv.get(dtype.char, 'd')
270:     if dtype.char == 'G':
271:         # complex256 -> complex128 (i.e., C long double -> C double)
272:         dtype = _np.dtype('D')
273:     elif dtype.char not in 'fdFD':
274:         dtype = _np.dtype('d')
275: 
276:     return prefix, dtype, prefer_fortran
277: 
278: 
279: def _get_funcs(names, arrays, dtype,
280:                lib_name, fmodule, cmodule,
281:                fmodule_name, cmodule_name, alias):
282:     '''
283:     Return available BLAS/LAPACK functions.
284: 
285:     Used also in lapack.py. See get_blas_funcs for docstring.
286:     '''
287: 
288:     funcs = []
289:     unpack = False
290:     dtype = _np.dtype(dtype)
291:     module1 = (cmodule, cmodule_name)
292:     module2 = (fmodule, fmodule_name)
293: 
294:     if isinstance(names, str):
295:         names = (names,)
296:         unpack = True
297: 
298:     prefix, dtype, prefer_fortran = find_best_blas_type(arrays, dtype)
299: 
300:     if prefer_fortran:
301:         module1, module2 = module2, module1
302: 
303:     for i, name in enumerate(names):
304:         func_name = prefix + name
305:         func_name = alias.get(func_name, func_name)
306:         func = getattr(module1[0], func_name, None)
307:         module_name = module1[1]
308:         if func is None:
309:             func = getattr(module2[0], func_name, None)
310:             module_name = module2[1]
311:         if func is None:
312:             raise ValueError(
313:                 '%s function %s could not be found' % (lib_name, func_name))
314:         func.module_name, func.typecode = module_name, prefix
315:         func.dtype = dtype
316:         func.prefix = prefix  # Backward compatibility
317:         funcs.append(func)
318: 
319:     if unpack:
320:         return funcs[0]
321:     else:
322:         return funcs
323: 
324: 
325: def get_blas_funcs(names, arrays=(), dtype=None):
326:     '''Return available BLAS function objects from names.
327: 
328:     Arrays are used to determine the optimal prefix of BLAS routines.
329: 
330:     Parameters
331:     ----------
332:     names : str or sequence of str
333:         Name(s) of BLAS functions without type prefix.
334: 
335:     arrays : sequence of ndarrays, optional
336:         Arrays can be given to determine optimal prefix of BLAS
337:         routines. If not given, double-precision routines will be
338:         used, otherwise the most generic type in arrays will be used.
339: 
340:     dtype : str or dtype, optional
341:         Data-type specifier. Not used if `arrays` is non-empty.
342: 
343: 
344:     Returns
345:     -------
346:     funcs : list
347:         List containing the found function(s).
348: 
349: 
350:     Notes
351:     -----
352:     This routine automatically chooses between Fortran/C
353:     interfaces. Fortran code is used whenever possible for arrays with
354:     column major order. In all other cases, C code is preferred.
355: 
356:     In BLAS, the naming convention is that all functions start with a
357:     type prefix, which depends on the type of the principal
358:     matrix. These can be one of {'s', 'd', 'c', 'z'} for the numpy
359:     types {float32, float64, complex64, complex128} respectively.
360:     The code and the dtype are stored in attributes `typecode` and `dtype`
361:     of the returned functions.
362: 
363:     Examples
364:     --------
365:     >>> import scipy.linalg as LA
366:     >>> a = np.random.rand(3,2)
367:     >>> x_gemv = LA.get_blas_funcs('gemv', (a,))
368:     >>> x_gemv.typecode
369:     'd'
370:     >>> x_gemv = LA.get_blas_funcs('gemv',(a*1j,))
371:     >>> x_gemv.typecode
372:     'z'
373: 
374:     '''
375:     return _get_funcs(names, arrays, dtype,
376:                       "BLAS", _fblas, _cblas, "fblas", "cblas",
377:                       _blas_alias)
378: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_13759 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 184, (-1)), 'str', '\nLow-level BLAS functions (:mod:`scipy.linalg.blas`)\n===================================================\n\nThis module contains low-level functions from the BLAS library.\n\n.. versionadded:: 0.12.0\n\n.. warning::\n\n   These functions do little to no error checking.\n   It is possible to cause crashes by mis-using them,\n   so prefer using the higher-level routines in `scipy.linalg`.\n\nFinding functions\n-----------------\n\n.. autosummary::\n   :toctree: generated/\n\n   get_blas_funcs\n   find_best_blas_type\n\nBLAS Level 1 functions\n----------------------\n\n.. autosummary::\n   :toctree: generated/\n\n   caxpy\n   ccopy\n   cdotc\n   cdotu\n   crotg\n   cscal\n   csrot\n   csscal\n   cswap\n   dasum\n   daxpy\n   dcopy\n   ddot\n   dnrm2\n   drot\n   drotg\n   drotm\n   drotmg\n   dscal\n   dswap\n   dzasum\n   dznrm2\n   icamax\n   idamax\n   isamax\n   izamax\n   sasum\n   saxpy\n   scasum\n   scnrm2\n   scopy\n   sdot\n   snrm2\n   srot\n   srotg\n   srotm\n   srotmg\n   sscal\n   sswap\n   zaxpy\n   zcopy\n   zdotc\n   zdotu\n   zdrot\n   zdscal\n   zrotg\n   zscal\n   zswap\n\nBLAS Level 2 functions\n----------------------\n\n.. autosummary::\n   :toctree: generated/\n\n   sgbmv\n   sgemv\n   sger\n   ssbmv\n   sspr\n   sspr2\n   ssymv\n   ssyr\n   ssyr2\n   stbmv\n   stpsv\n   strmv\n   strsv\n   dgbmv\n   dgemv\n   dger\n   dsbmv\n   dspr\n   dspr2\n   dsymv\n   dsyr\n   dsyr2\n   dtbmv\n   dtpsv\n   dtrmv\n   dtrsv\n   cgbmv\n   cgemv\n   cgerc\n   cgeru\n   chbmv\n   chemv\n   cher\n   cher2\n   chpmv\n   chpr\n   chpr2\n   ctbmv\n   ctbsv\n   ctpmv\n   ctpsv\n   ctrmv\n   ctrsv\n   csyr\n   zgbmv\n   zgemv\n   zgerc\n   zgeru\n   zhbmv\n   zhemv\n   zher\n   zher2\n   zhpmv\n   zhpr\n   zhpr2\n   ztbmv\n   ztbsv\n   ztpmv\n   ztrmv\n   ztrsv\n   zsyr\n\nBLAS Level 3 functions\n----------------------\n\n.. autosummary::\n   :toctree: generated/\n\n   sgemm\n   ssymm\n   ssyr2k\n   ssyrk\n   strmm\n   strsm\n   dgemm\n   dsymm\n   dsyr2k\n   dsyrk\n   dtrmm\n   dtrsm\n   cgemm\n   chemm\n   cher2k\n   cherk\n   csymm\n   csyr2k\n   csyrk\n   ctrmm\n   ctrsm\n   zgemm\n   zhemm\n   zher2k\n   zherk\n   zsymm\n   zsyr2k\n   zsyrk\n   ztrmm\n   ztrsm\n\n')

# Assigning a List to a Name (line 192):

# Assigning a List to a Name (line 192):
__all__ = ['get_blas_funcs', 'find_best_blas_type']
module_type_store.set_exportable_members(['get_blas_funcs', 'find_best_blas_type'])

# Obtaining an instance of the builtin type 'list' (line 192)
list_13760 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 192, 10), 'list')
# Adding type elements to the builtin type 'list' instance (line 192)
# Adding element type (line 192)
str_13761 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 192, 11), 'str', 'get_blas_funcs')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 192, 10), list_13760, str_13761)
# Adding element type (line 192)
str_13762 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 192, 29), 'str', 'find_best_blas_type')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 192, 10), list_13760, str_13762)

# Assigning a type to the variable '__all__' (line 192)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 192, 0), '__all__', list_13760)
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 194, 0))

# 'import numpy' statement (line 194)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/linalg/')
import_13763 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 194, 0), 'numpy')

if (type(import_13763) is not StypyTypeError):

    if (import_13763 != 'pyd_module'):
        __import__(import_13763)
        sys_modules_13764 = sys.modules[import_13763]
        import_module(stypy.reporting.localization.Localization(__file__, 194, 0), '_np', sys_modules_13764.module_type_store, module_type_store)
    else:
        import numpy as _np

        import_module(stypy.reporting.localization.Localization(__file__, 194, 0), '_np', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 194)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 194, 0), 'numpy', import_13763)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/linalg/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 196, 0))

# 'from scipy.linalg import _fblas' statement (line 196)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/linalg/')
import_13765 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 196, 0), 'scipy.linalg')

if (type(import_13765) is not StypyTypeError):

    if (import_13765 != 'pyd_module'):
        __import__(import_13765)
        sys_modules_13766 = sys.modules[import_13765]
        import_from_module(stypy.reporting.localization.Localization(__file__, 196, 0), 'scipy.linalg', sys_modules_13766.module_type_store, module_type_store, ['_fblas'])
        nest_module(stypy.reporting.localization.Localization(__file__, 196, 0), __file__, sys_modules_13766, sys_modules_13766.module_type_store, module_type_store)
    else:
        from scipy.linalg import _fblas

        import_from_module(stypy.reporting.localization.Localization(__file__, 196, 0), 'scipy.linalg', None, module_type_store, ['_fblas'], [_fblas])

else:
    # Assigning a type to the variable 'scipy.linalg' (line 196)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 196, 0), 'scipy.linalg', import_13765)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/linalg/')



# SSA begins for try-except statement (line 197)
module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 198, 4))

# 'from scipy.linalg import _cblas' statement (line 198)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/linalg/')
import_13767 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 198, 4), 'scipy.linalg')

if (type(import_13767) is not StypyTypeError):

    if (import_13767 != 'pyd_module'):
        __import__(import_13767)
        sys_modules_13768 = sys.modules[import_13767]
        import_from_module(stypy.reporting.localization.Localization(__file__, 198, 4), 'scipy.linalg', sys_modules_13768.module_type_store, module_type_store, ['_cblas'])
        nest_module(stypy.reporting.localization.Localization(__file__, 198, 4), __file__, sys_modules_13768, sys_modules_13768.module_type_store, module_type_store)
    else:
        from scipy.linalg import _cblas

        import_from_module(stypy.reporting.localization.Localization(__file__, 198, 4), 'scipy.linalg', None, module_type_store, ['_cblas'], [_cblas])

else:
    # Assigning a type to the variable 'scipy.linalg' (line 198)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 198, 4), 'scipy.linalg', import_13767)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/linalg/')

# SSA branch for the except part of a try statement (line 197)
# SSA branch for the except 'ImportError' branch of a try statement (line 197)
module_type_store.open_ssa_branch('except')

# Assigning a Name to a Name (line 200):

# Assigning a Name to a Name (line 200):
# Getting the type of 'None' (line 200)
None_13769 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 13), 'None')
# Assigning a type to the variable '_cblas' (line 200)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 200, 4), '_cblas', None_13769)
# SSA join for try-except statement (line 197)
module_type_store = module_type_store.join_ssa_context()


# Assigning a Name to a Name (line 203):

# Assigning a Name to a Name (line 203):
# Getting the type of 'None' (line 203)
None_13770 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 15), 'None')
# Assigning a type to the variable 'empty_module' (line 203)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 203, 0), 'empty_module', None_13770)
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 204, 0))

# 'from scipy.linalg._fblas import ' statement (line 204)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/linalg/')
import_13771 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 204, 0), 'scipy.linalg._fblas')

if (type(import_13771) is not StypyTypeError):

    if (import_13771 != 'pyd_module'):
        __import__(import_13771)
        sys_modules_13772 = sys.modules[import_13771]
        import_from_module(stypy.reporting.localization.Localization(__file__, 204, 0), 'scipy.linalg._fblas', sys_modules_13772.module_type_store, module_type_store, ['*'])
        nest_module(stypy.reporting.localization.Localization(__file__, 204, 0), __file__, sys_modules_13772, sys_modules_13772.module_type_store, module_type_store)
    else:
        from scipy.linalg._fblas import *

        import_from_module(stypy.reporting.localization.Localization(__file__, 204, 0), 'scipy.linalg._fblas', None, module_type_store, ['*'], None)

else:
    # Assigning a type to the variable 'scipy.linalg._fblas' (line 204)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 204, 0), 'scipy.linalg._fblas', import_13771)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/linalg/')

# Deleting a member
module_type_store.del_member(stypy.reporting.localization.Localization(__file__, 205, 0), module_type_store, 'empty_module')

# Assigning a Dict to a Name (line 208):

# Assigning a Dict to a Name (line 208):

# Obtaining an instance of the builtin type 'dict' (line 208)
dict_13773 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 208, 13), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 208)
# Adding element type (key, value) (line 208)
str_13774 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 208, 14), 'str', 'f')
str_13775 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 208, 19), 'str', 's')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 208, 13), dict_13773, (str_13774, str_13775))
# Adding element type (key, value) (line 208)
str_13776 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 208, 24), 'str', 'd')
str_13777 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 208, 29), 'str', 'd')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 208, 13), dict_13773, (str_13776, str_13777))
# Adding element type (key, value) (line 208)
str_13778 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 208, 34), 'str', 'F')
str_13779 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 208, 39), 'str', 'c')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 208, 13), dict_13773, (str_13778, str_13779))
# Adding element type (key, value) (line 208)
str_13780 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 208, 44), 'str', 'D')
str_13781 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 208, 49), 'str', 'z')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 208, 13), dict_13773, (str_13780, str_13781))
# Adding element type (key, value) (line 208)
str_13782 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 208, 54), 'str', 'G')
str_13783 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 208, 59), 'str', 'z')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 208, 13), dict_13773, (str_13782, str_13783))

# Assigning a type to the variable '_type_conv' (line 208)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 208, 0), '_type_conv', dict_13773)

# Assigning a Dict to a Name (line 211):

# Assigning a Dict to a Name (line 211):

# Obtaining an instance of the builtin type 'dict' (line 211)
dict_13784 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 211, 14), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 211)
# Adding element type (key, value) (line 211)
str_13785 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 211, 15), 'str', 'cnrm2')
str_13786 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 211, 24), 'str', 'scnrm2')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 211, 14), dict_13784, (str_13785, str_13786))
# Adding element type (key, value) (line 211)
str_13787 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 211, 34), 'str', 'znrm2')
str_13788 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 211, 43), 'str', 'dznrm2')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 211, 14), dict_13784, (str_13787, str_13788))
# Adding element type (key, value) (line 211)
str_13789 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 212, 15), 'str', 'cdot')
str_13790 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 212, 23), 'str', 'cdotc')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 211, 14), dict_13784, (str_13789, str_13790))
# Adding element type (key, value) (line 211)
str_13791 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 212, 32), 'str', 'zdot')
str_13792 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 212, 40), 'str', 'zdotc')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 211, 14), dict_13784, (str_13791, str_13792))
# Adding element type (key, value) (line 211)
str_13793 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 213, 15), 'str', 'cger')
str_13794 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 213, 23), 'str', 'cgerc')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 211, 14), dict_13784, (str_13793, str_13794))
# Adding element type (key, value) (line 211)
str_13795 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 213, 32), 'str', 'zger')
str_13796 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 213, 40), 'str', 'zgerc')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 211, 14), dict_13784, (str_13795, str_13796))
# Adding element type (key, value) (line 211)
str_13797 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 214, 15), 'str', 'sdotc')
str_13798 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 214, 24), 'str', 'sdot')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 211, 14), dict_13784, (str_13797, str_13798))
# Adding element type (key, value) (line 211)
str_13799 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 214, 32), 'str', 'sdotu')
str_13800 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 214, 41), 'str', 'sdot')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 211, 14), dict_13784, (str_13799, str_13800))
# Adding element type (key, value) (line 211)
str_13801 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 215, 15), 'str', 'ddotc')
str_13802 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 215, 24), 'str', 'ddot')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 211, 14), dict_13784, (str_13801, str_13802))
# Adding element type (key, value) (line 211)
str_13803 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 215, 32), 'str', 'ddotu')
str_13804 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 215, 41), 'str', 'ddot')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 211, 14), dict_13784, (str_13803, str_13804))

# Assigning a type to the variable '_blas_alias' (line 211)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 211, 0), '_blas_alias', dict_13784)

@norecursion
def find_best_blas_type(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    
    # Obtaining an instance of the builtin type 'tuple' (line 218)
    tuple_13805 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 218, 31), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 218)
    
    # Getting the type of 'None' (line 218)
    None_13806 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 218, 41), 'None')
    defaults = [tuple_13805, None_13806]
    # Create a new context for function 'find_best_blas_type'
    module_type_store = module_type_store.open_function_context('find_best_blas_type', 218, 0, False)
    
    # Passed parameters checking function
    find_best_blas_type.stypy_localization = localization
    find_best_blas_type.stypy_type_of_self = None
    find_best_blas_type.stypy_type_store = module_type_store
    find_best_blas_type.stypy_function_name = 'find_best_blas_type'
    find_best_blas_type.stypy_param_names_list = ['arrays', 'dtype']
    find_best_blas_type.stypy_varargs_param_name = None
    find_best_blas_type.stypy_kwargs_param_name = None
    find_best_blas_type.stypy_call_defaults = defaults
    find_best_blas_type.stypy_call_varargs = varargs
    find_best_blas_type.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'find_best_blas_type', ['arrays', 'dtype'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'find_best_blas_type', localization, ['arrays', 'dtype'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'find_best_blas_type(...)' code ##################

    str_13807 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 253, (-1)), 'str', "Find best-matching BLAS/LAPACK type.\n\n    Arrays are used to determine the optimal prefix of BLAS routines.\n\n    Parameters\n    ----------\n    arrays : sequence of ndarrays, optional\n        Arrays can be given to determine optimal prefix of BLAS\n        routines. If not given, double-precision routines will be\n        used, otherwise the most generic type in arrays will be used.\n    dtype : str or dtype, optional\n        Data-type specifier. Not used if `arrays` is non-empty.\n\n    Returns\n    -------\n    prefix : str\n        BLAS/LAPACK prefix character.\n    dtype : dtype\n        Inferred Numpy data type.\n    prefer_fortran : bool\n        Whether to prefer Fortran order routines over C order.\n\n    Examples\n    --------\n    >>> import scipy.linalg.blas as bla\n    >>> a = np.random.rand(10,15)\n    >>> b = np.asfortranarray(a)  # Change the memory layout order\n    >>> bla.find_best_blas_type((a,))\n    ('d', dtype('float64'), False)\n    >>> bla.find_best_blas_type((a*1j,))\n    ('z', dtype('complex128'), False)\n    >>> bla.find_best_blas_type((b,))\n    ('d', dtype('float64'), True)\n\n    ")
    
    # Assigning a Call to a Name (line 254):
    
    # Assigning a Call to a Name (line 254):
    
    # Call to dtype(...): (line 254)
    # Processing the call arguments (line 254)
    # Getting the type of 'dtype' (line 254)
    dtype_13810 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 254, 22), 'dtype', False)
    # Processing the call keyword arguments (line 254)
    kwargs_13811 = {}
    # Getting the type of '_np' (line 254)
    _np_13808 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 254, 12), '_np', False)
    # Obtaining the member 'dtype' of a type (line 254)
    dtype_13809 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 254, 12), _np_13808, 'dtype')
    # Calling dtype(args, kwargs) (line 254)
    dtype_call_result_13812 = invoke(stypy.reporting.localization.Localization(__file__, 254, 12), dtype_13809, *[dtype_13810], **kwargs_13811)
    
    # Assigning a type to the variable 'dtype' (line 254)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 254, 4), 'dtype', dtype_call_result_13812)
    
    # Assigning a Name to a Name (line 255):
    
    # Assigning a Name to a Name (line 255):
    # Getting the type of 'False' (line 255)
    False_13813 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 255, 21), 'False')
    # Assigning a type to the variable 'prefer_fortran' (line 255)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 255, 4), 'prefer_fortran', False_13813)
    
    # Getting the type of 'arrays' (line 257)
    arrays_13814 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 257, 7), 'arrays')
    # Testing the type of an if condition (line 257)
    if_condition_13815 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 257, 4), arrays_13814)
    # Assigning a type to the variable 'if_condition_13815' (line 257)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 257, 4), 'if_condition_13815', if_condition_13815)
    # SSA begins for if statement (line 257)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a ListComp to a Name (line 259):
    
    # Assigning a ListComp to a Name (line 259):
    # Calculating list comprehension
    # Calculating comprehension expression
    # Getting the type of 'arrays' (line 259)
    arrays_13818 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 259, 37), 'arrays')
    comprehension_13819 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 259, 18), arrays_13818)
    # Assigning a type to the variable 'ar' (line 259)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 259, 18), 'ar', comprehension_13819)
    # Getting the type of 'ar' (line 259)
    ar_13816 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 259, 18), 'ar')
    # Obtaining the member 'dtype' of a type (line 259)
    dtype_13817 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 259, 18), ar_13816, 'dtype')
    list_13820 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 259, 18), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 259, 18), list_13820, dtype_13817)
    # Assigning a type to the variable 'dtypes' (line 259)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 259, 8), 'dtypes', list_13820)
    
    # Assigning a Call to a Name (line 260):
    
    # Assigning a Call to a Name (line 260):
    
    # Call to find_common_type(...): (line 260)
    # Processing the call arguments (line 260)
    # Getting the type of 'dtypes' (line 260)
    dtypes_13823 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 260, 37), 'dtypes', False)
    
    # Obtaining an instance of the builtin type 'tuple' (line 260)
    tuple_13824 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 260, 45), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 260)
    
    # Processing the call keyword arguments (line 260)
    kwargs_13825 = {}
    # Getting the type of '_np' (line 260)
    _np_13821 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 260, 16), '_np', False)
    # Obtaining the member 'find_common_type' of a type (line 260)
    find_common_type_13822 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 260, 16), _np_13821, 'find_common_type')
    # Calling find_common_type(args, kwargs) (line 260)
    find_common_type_call_result_13826 = invoke(stypy.reporting.localization.Localization(__file__, 260, 16), find_common_type_13822, *[dtypes_13823, tuple_13824], **kwargs_13825)
    
    # Assigning a type to the variable 'dtype' (line 260)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 260, 8), 'dtype', find_common_type_call_result_13826)
    
    
    # SSA begins for try-except statement (line 261)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
    
    # Assigning a Call to a Name (line 262):
    
    # Assigning a Call to a Name (line 262):
    
    # Call to index(...): (line 262)
    # Processing the call arguments (line 262)
    # Getting the type of 'dtype' (line 262)
    dtype_13829 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 262, 33), 'dtype', False)
    # Processing the call keyword arguments (line 262)
    kwargs_13830 = {}
    # Getting the type of 'dtypes' (line 262)
    dtypes_13827 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 262, 20), 'dtypes', False)
    # Obtaining the member 'index' of a type (line 262)
    index_13828 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 262, 20), dtypes_13827, 'index')
    # Calling index(args, kwargs) (line 262)
    index_call_result_13831 = invoke(stypy.reporting.localization.Localization(__file__, 262, 20), index_13828, *[dtype_13829], **kwargs_13830)
    
    # Assigning a type to the variable 'index' (line 262)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 262, 12), 'index', index_call_result_13831)
    # SSA branch for the except part of a try statement (line 261)
    # SSA branch for the except 'ValueError' branch of a try statement (line 261)
    module_type_store.open_ssa_branch('except')
    
    # Assigning a Num to a Name (line 264):
    
    # Assigning a Num to a Name (line 264):
    int_13832 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 264, 20), 'int')
    # Assigning a type to the variable 'index' (line 264)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 264, 12), 'index', int_13832)
    # SSA join for try-except statement (line 261)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Obtaining the type of the subscript
    str_13833 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 265, 31), 'str', 'FORTRAN')
    
    # Obtaining the type of the subscript
    # Getting the type of 'index' (line 265)
    index_13834 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 265, 18), 'index')
    # Getting the type of 'arrays' (line 265)
    arrays_13835 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 265, 11), 'arrays')
    # Obtaining the member '__getitem__' of a type (line 265)
    getitem___13836 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 265, 11), arrays_13835, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 265)
    subscript_call_result_13837 = invoke(stypy.reporting.localization.Localization(__file__, 265, 11), getitem___13836, index_13834)
    
    # Obtaining the member 'flags' of a type (line 265)
    flags_13838 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 265, 11), subscript_call_result_13837, 'flags')
    # Obtaining the member '__getitem__' of a type (line 265)
    getitem___13839 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 265, 11), flags_13838, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 265)
    subscript_call_result_13840 = invoke(stypy.reporting.localization.Localization(__file__, 265, 11), getitem___13839, str_13833)
    
    # Testing the type of an if condition (line 265)
    if_condition_13841 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 265, 8), subscript_call_result_13840)
    # Assigning a type to the variable 'if_condition_13841' (line 265)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 265, 8), 'if_condition_13841', if_condition_13841)
    # SSA begins for if statement (line 265)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Name to a Name (line 267):
    
    # Assigning a Name to a Name (line 267):
    # Getting the type of 'True' (line 267)
    True_13842 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 267, 29), 'True')
    # Assigning a type to the variable 'prefer_fortran' (line 267)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 267, 12), 'prefer_fortran', True_13842)
    # SSA join for if statement (line 265)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 257)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 269):
    
    # Assigning a Call to a Name (line 269):
    
    # Call to get(...): (line 269)
    # Processing the call arguments (line 269)
    # Getting the type of 'dtype' (line 269)
    dtype_13845 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 269, 28), 'dtype', False)
    # Obtaining the member 'char' of a type (line 269)
    char_13846 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 269, 28), dtype_13845, 'char')
    str_13847 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 269, 40), 'str', 'd')
    # Processing the call keyword arguments (line 269)
    kwargs_13848 = {}
    # Getting the type of '_type_conv' (line 269)
    _type_conv_13843 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 269, 13), '_type_conv', False)
    # Obtaining the member 'get' of a type (line 269)
    get_13844 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 269, 13), _type_conv_13843, 'get')
    # Calling get(args, kwargs) (line 269)
    get_call_result_13849 = invoke(stypy.reporting.localization.Localization(__file__, 269, 13), get_13844, *[char_13846, str_13847], **kwargs_13848)
    
    # Assigning a type to the variable 'prefix' (line 269)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 269, 4), 'prefix', get_call_result_13849)
    
    
    # Getting the type of 'dtype' (line 270)
    dtype_13850 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 270, 7), 'dtype')
    # Obtaining the member 'char' of a type (line 270)
    char_13851 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 270, 7), dtype_13850, 'char')
    str_13852 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 270, 21), 'str', 'G')
    # Applying the binary operator '==' (line 270)
    result_eq_13853 = python_operator(stypy.reporting.localization.Localization(__file__, 270, 7), '==', char_13851, str_13852)
    
    # Testing the type of an if condition (line 270)
    if_condition_13854 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 270, 4), result_eq_13853)
    # Assigning a type to the variable 'if_condition_13854' (line 270)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 270, 4), 'if_condition_13854', if_condition_13854)
    # SSA begins for if statement (line 270)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 272):
    
    # Assigning a Call to a Name (line 272):
    
    # Call to dtype(...): (line 272)
    # Processing the call arguments (line 272)
    str_13857 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 272, 26), 'str', 'D')
    # Processing the call keyword arguments (line 272)
    kwargs_13858 = {}
    # Getting the type of '_np' (line 272)
    _np_13855 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 272, 16), '_np', False)
    # Obtaining the member 'dtype' of a type (line 272)
    dtype_13856 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 272, 16), _np_13855, 'dtype')
    # Calling dtype(args, kwargs) (line 272)
    dtype_call_result_13859 = invoke(stypy.reporting.localization.Localization(__file__, 272, 16), dtype_13856, *[str_13857], **kwargs_13858)
    
    # Assigning a type to the variable 'dtype' (line 272)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 272, 8), 'dtype', dtype_call_result_13859)
    # SSA branch for the else part of an if statement (line 270)
    module_type_store.open_ssa_branch('else')
    
    
    # Getting the type of 'dtype' (line 273)
    dtype_13860 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 273, 9), 'dtype')
    # Obtaining the member 'char' of a type (line 273)
    char_13861 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 273, 9), dtype_13860, 'char')
    str_13862 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 273, 27), 'str', 'fdFD')
    # Applying the binary operator 'notin' (line 273)
    result_contains_13863 = python_operator(stypy.reporting.localization.Localization(__file__, 273, 9), 'notin', char_13861, str_13862)
    
    # Testing the type of an if condition (line 273)
    if_condition_13864 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 273, 9), result_contains_13863)
    # Assigning a type to the variable 'if_condition_13864' (line 273)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 273, 9), 'if_condition_13864', if_condition_13864)
    # SSA begins for if statement (line 273)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 274):
    
    # Assigning a Call to a Name (line 274):
    
    # Call to dtype(...): (line 274)
    # Processing the call arguments (line 274)
    str_13867 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 274, 26), 'str', 'd')
    # Processing the call keyword arguments (line 274)
    kwargs_13868 = {}
    # Getting the type of '_np' (line 274)
    _np_13865 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 274, 16), '_np', False)
    # Obtaining the member 'dtype' of a type (line 274)
    dtype_13866 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 274, 16), _np_13865, 'dtype')
    # Calling dtype(args, kwargs) (line 274)
    dtype_call_result_13869 = invoke(stypy.reporting.localization.Localization(__file__, 274, 16), dtype_13866, *[str_13867], **kwargs_13868)
    
    # Assigning a type to the variable 'dtype' (line 274)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 274, 8), 'dtype', dtype_call_result_13869)
    # SSA join for if statement (line 273)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 270)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Obtaining an instance of the builtin type 'tuple' (line 276)
    tuple_13870 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 276, 11), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 276)
    # Adding element type (line 276)
    # Getting the type of 'prefix' (line 276)
    prefix_13871 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 276, 11), 'prefix')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 276, 11), tuple_13870, prefix_13871)
    # Adding element type (line 276)
    # Getting the type of 'dtype' (line 276)
    dtype_13872 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 276, 19), 'dtype')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 276, 11), tuple_13870, dtype_13872)
    # Adding element type (line 276)
    # Getting the type of 'prefer_fortran' (line 276)
    prefer_fortran_13873 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 276, 26), 'prefer_fortran')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 276, 11), tuple_13870, prefer_fortran_13873)
    
    # Assigning a type to the variable 'stypy_return_type' (line 276)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 276, 4), 'stypy_return_type', tuple_13870)
    
    # ################# End of 'find_best_blas_type(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'find_best_blas_type' in the type store
    # Getting the type of 'stypy_return_type' (line 218)
    stypy_return_type_13874 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 218, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_13874)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'find_best_blas_type'
    return stypy_return_type_13874

# Assigning a type to the variable 'find_best_blas_type' (line 218)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 218, 0), 'find_best_blas_type', find_best_blas_type)

@norecursion
def _get_funcs(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_get_funcs'
    module_type_store = module_type_store.open_function_context('_get_funcs', 279, 0, False)
    
    # Passed parameters checking function
    _get_funcs.stypy_localization = localization
    _get_funcs.stypy_type_of_self = None
    _get_funcs.stypy_type_store = module_type_store
    _get_funcs.stypy_function_name = '_get_funcs'
    _get_funcs.stypy_param_names_list = ['names', 'arrays', 'dtype', 'lib_name', 'fmodule', 'cmodule', 'fmodule_name', 'cmodule_name', 'alias']
    _get_funcs.stypy_varargs_param_name = None
    _get_funcs.stypy_kwargs_param_name = None
    _get_funcs.stypy_call_defaults = defaults
    _get_funcs.stypy_call_varargs = varargs
    _get_funcs.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_get_funcs', ['names', 'arrays', 'dtype', 'lib_name', 'fmodule', 'cmodule', 'fmodule_name', 'cmodule_name', 'alias'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_get_funcs', localization, ['names', 'arrays', 'dtype', 'lib_name', 'fmodule', 'cmodule', 'fmodule_name', 'cmodule_name', 'alias'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_get_funcs(...)' code ##################

    str_13875 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 286, (-1)), 'str', '\n    Return available BLAS/LAPACK functions.\n\n    Used also in lapack.py. See get_blas_funcs for docstring.\n    ')
    
    # Assigning a List to a Name (line 288):
    
    # Assigning a List to a Name (line 288):
    
    # Obtaining an instance of the builtin type 'list' (line 288)
    list_13876 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 288, 12), 'list')
    # Adding type elements to the builtin type 'list' instance (line 288)
    
    # Assigning a type to the variable 'funcs' (line 288)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 288, 4), 'funcs', list_13876)
    
    # Assigning a Name to a Name (line 289):
    
    # Assigning a Name to a Name (line 289):
    # Getting the type of 'False' (line 289)
    False_13877 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 289, 13), 'False')
    # Assigning a type to the variable 'unpack' (line 289)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 289, 4), 'unpack', False_13877)
    
    # Assigning a Call to a Name (line 290):
    
    # Assigning a Call to a Name (line 290):
    
    # Call to dtype(...): (line 290)
    # Processing the call arguments (line 290)
    # Getting the type of 'dtype' (line 290)
    dtype_13880 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 290, 22), 'dtype', False)
    # Processing the call keyword arguments (line 290)
    kwargs_13881 = {}
    # Getting the type of '_np' (line 290)
    _np_13878 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 290, 12), '_np', False)
    # Obtaining the member 'dtype' of a type (line 290)
    dtype_13879 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 290, 12), _np_13878, 'dtype')
    # Calling dtype(args, kwargs) (line 290)
    dtype_call_result_13882 = invoke(stypy.reporting.localization.Localization(__file__, 290, 12), dtype_13879, *[dtype_13880], **kwargs_13881)
    
    # Assigning a type to the variable 'dtype' (line 290)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 290, 4), 'dtype', dtype_call_result_13882)
    
    # Assigning a Tuple to a Name (line 291):
    
    # Assigning a Tuple to a Name (line 291):
    
    # Obtaining an instance of the builtin type 'tuple' (line 291)
    tuple_13883 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 291, 15), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 291)
    # Adding element type (line 291)
    # Getting the type of 'cmodule' (line 291)
    cmodule_13884 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 291, 15), 'cmodule')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 291, 15), tuple_13883, cmodule_13884)
    # Adding element type (line 291)
    # Getting the type of 'cmodule_name' (line 291)
    cmodule_name_13885 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 291, 24), 'cmodule_name')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 291, 15), tuple_13883, cmodule_name_13885)
    
    # Assigning a type to the variable 'module1' (line 291)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 291, 4), 'module1', tuple_13883)
    
    # Assigning a Tuple to a Name (line 292):
    
    # Assigning a Tuple to a Name (line 292):
    
    # Obtaining an instance of the builtin type 'tuple' (line 292)
    tuple_13886 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 292, 15), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 292)
    # Adding element type (line 292)
    # Getting the type of 'fmodule' (line 292)
    fmodule_13887 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 292, 15), 'fmodule')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 292, 15), tuple_13886, fmodule_13887)
    # Adding element type (line 292)
    # Getting the type of 'fmodule_name' (line 292)
    fmodule_name_13888 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 292, 24), 'fmodule_name')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 292, 15), tuple_13886, fmodule_name_13888)
    
    # Assigning a type to the variable 'module2' (line 292)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 292, 4), 'module2', tuple_13886)
    
    # Type idiom detected: calculating its left and rigth part (line 294)
    # Getting the type of 'str' (line 294)
    str_13889 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 294, 25), 'str')
    # Getting the type of 'names' (line 294)
    names_13890 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 294, 18), 'names')
    
    (may_be_13891, more_types_in_union_13892) = may_be_subtype(str_13889, names_13890)

    if may_be_13891:

        if more_types_in_union_13892:
            # Runtime conditional SSA (line 294)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        # Assigning a type to the variable 'names' (line 294)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 294, 4), 'names', remove_not_subtype_from_union(names_13890, str))
        
        # Assigning a Tuple to a Name (line 295):
        
        # Assigning a Tuple to a Name (line 295):
        
        # Obtaining an instance of the builtin type 'tuple' (line 295)
        tuple_13893 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 295, 17), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 295)
        # Adding element type (line 295)
        # Getting the type of 'names' (line 295)
        names_13894 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 295, 17), 'names')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 295, 17), tuple_13893, names_13894)
        
        # Assigning a type to the variable 'names' (line 295)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 295, 8), 'names', tuple_13893)
        
        # Assigning a Name to a Name (line 296):
        
        # Assigning a Name to a Name (line 296):
        # Getting the type of 'True' (line 296)
        True_13895 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 296, 17), 'True')
        # Assigning a type to the variable 'unpack' (line 296)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 296, 8), 'unpack', True_13895)

        if more_types_in_union_13892:
            # SSA join for if statement (line 294)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Assigning a Call to a Tuple (line 298):
    
    # Assigning a Subscript to a Name (line 298):
    
    # Obtaining the type of the subscript
    int_13896 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 298, 4), 'int')
    
    # Call to find_best_blas_type(...): (line 298)
    # Processing the call arguments (line 298)
    # Getting the type of 'arrays' (line 298)
    arrays_13898 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 298, 56), 'arrays', False)
    # Getting the type of 'dtype' (line 298)
    dtype_13899 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 298, 64), 'dtype', False)
    # Processing the call keyword arguments (line 298)
    kwargs_13900 = {}
    # Getting the type of 'find_best_blas_type' (line 298)
    find_best_blas_type_13897 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 298, 36), 'find_best_blas_type', False)
    # Calling find_best_blas_type(args, kwargs) (line 298)
    find_best_blas_type_call_result_13901 = invoke(stypy.reporting.localization.Localization(__file__, 298, 36), find_best_blas_type_13897, *[arrays_13898, dtype_13899], **kwargs_13900)
    
    # Obtaining the member '__getitem__' of a type (line 298)
    getitem___13902 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 298, 4), find_best_blas_type_call_result_13901, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 298)
    subscript_call_result_13903 = invoke(stypy.reporting.localization.Localization(__file__, 298, 4), getitem___13902, int_13896)
    
    # Assigning a type to the variable 'tuple_var_assignment_13752' (line 298)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 298, 4), 'tuple_var_assignment_13752', subscript_call_result_13903)
    
    # Assigning a Subscript to a Name (line 298):
    
    # Obtaining the type of the subscript
    int_13904 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 298, 4), 'int')
    
    # Call to find_best_blas_type(...): (line 298)
    # Processing the call arguments (line 298)
    # Getting the type of 'arrays' (line 298)
    arrays_13906 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 298, 56), 'arrays', False)
    # Getting the type of 'dtype' (line 298)
    dtype_13907 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 298, 64), 'dtype', False)
    # Processing the call keyword arguments (line 298)
    kwargs_13908 = {}
    # Getting the type of 'find_best_blas_type' (line 298)
    find_best_blas_type_13905 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 298, 36), 'find_best_blas_type', False)
    # Calling find_best_blas_type(args, kwargs) (line 298)
    find_best_blas_type_call_result_13909 = invoke(stypy.reporting.localization.Localization(__file__, 298, 36), find_best_blas_type_13905, *[arrays_13906, dtype_13907], **kwargs_13908)
    
    # Obtaining the member '__getitem__' of a type (line 298)
    getitem___13910 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 298, 4), find_best_blas_type_call_result_13909, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 298)
    subscript_call_result_13911 = invoke(stypy.reporting.localization.Localization(__file__, 298, 4), getitem___13910, int_13904)
    
    # Assigning a type to the variable 'tuple_var_assignment_13753' (line 298)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 298, 4), 'tuple_var_assignment_13753', subscript_call_result_13911)
    
    # Assigning a Subscript to a Name (line 298):
    
    # Obtaining the type of the subscript
    int_13912 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 298, 4), 'int')
    
    # Call to find_best_blas_type(...): (line 298)
    # Processing the call arguments (line 298)
    # Getting the type of 'arrays' (line 298)
    arrays_13914 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 298, 56), 'arrays', False)
    # Getting the type of 'dtype' (line 298)
    dtype_13915 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 298, 64), 'dtype', False)
    # Processing the call keyword arguments (line 298)
    kwargs_13916 = {}
    # Getting the type of 'find_best_blas_type' (line 298)
    find_best_blas_type_13913 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 298, 36), 'find_best_blas_type', False)
    # Calling find_best_blas_type(args, kwargs) (line 298)
    find_best_blas_type_call_result_13917 = invoke(stypy.reporting.localization.Localization(__file__, 298, 36), find_best_blas_type_13913, *[arrays_13914, dtype_13915], **kwargs_13916)
    
    # Obtaining the member '__getitem__' of a type (line 298)
    getitem___13918 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 298, 4), find_best_blas_type_call_result_13917, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 298)
    subscript_call_result_13919 = invoke(stypy.reporting.localization.Localization(__file__, 298, 4), getitem___13918, int_13912)
    
    # Assigning a type to the variable 'tuple_var_assignment_13754' (line 298)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 298, 4), 'tuple_var_assignment_13754', subscript_call_result_13919)
    
    # Assigning a Name to a Name (line 298):
    # Getting the type of 'tuple_var_assignment_13752' (line 298)
    tuple_var_assignment_13752_13920 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 298, 4), 'tuple_var_assignment_13752')
    # Assigning a type to the variable 'prefix' (line 298)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 298, 4), 'prefix', tuple_var_assignment_13752_13920)
    
    # Assigning a Name to a Name (line 298):
    # Getting the type of 'tuple_var_assignment_13753' (line 298)
    tuple_var_assignment_13753_13921 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 298, 4), 'tuple_var_assignment_13753')
    # Assigning a type to the variable 'dtype' (line 298)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 298, 12), 'dtype', tuple_var_assignment_13753_13921)
    
    # Assigning a Name to a Name (line 298):
    # Getting the type of 'tuple_var_assignment_13754' (line 298)
    tuple_var_assignment_13754_13922 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 298, 4), 'tuple_var_assignment_13754')
    # Assigning a type to the variable 'prefer_fortran' (line 298)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 298, 19), 'prefer_fortran', tuple_var_assignment_13754_13922)
    
    # Getting the type of 'prefer_fortran' (line 300)
    prefer_fortran_13923 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 300, 7), 'prefer_fortran')
    # Testing the type of an if condition (line 300)
    if_condition_13924 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 300, 4), prefer_fortran_13923)
    # Assigning a type to the variable 'if_condition_13924' (line 300)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 300, 4), 'if_condition_13924', if_condition_13924)
    # SSA begins for if statement (line 300)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Tuple to a Tuple (line 301):
    
    # Assigning a Name to a Name (line 301):
    # Getting the type of 'module2' (line 301)
    module2_13925 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 301, 27), 'module2')
    # Assigning a type to the variable 'tuple_assignment_13755' (line 301)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 301, 8), 'tuple_assignment_13755', module2_13925)
    
    # Assigning a Name to a Name (line 301):
    # Getting the type of 'module1' (line 301)
    module1_13926 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 301, 36), 'module1')
    # Assigning a type to the variable 'tuple_assignment_13756' (line 301)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 301, 8), 'tuple_assignment_13756', module1_13926)
    
    # Assigning a Name to a Name (line 301):
    # Getting the type of 'tuple_assignment_13755' (line 301)
    tuple_assignment_13755_13927 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 301, 8), 'tuple_assignment_13755')
    # Assigning a type to the variable 'module1' (line 301)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 301, 8), 'module1', tuple_assignment_13755_13927)
    
    # Assigning a Name to a Name (line 301):
    # Getting the type of 'tuple_assignment_13756' (line 301)
    tuple_assignment_13756_13928 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 301, 8), 'tuple_assignment_13756')
    # Assigning a type to the variable 'module2' (line 301)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 301, 17), 'module2', tuple_assignment_13756_13928)
    # SSA join for if statement (line 300)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Call to enumerate(...): (line 303)
    # Processing the call arguments (line 303)
    # Getting the type of 'names' (line 303)
    names_13930 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 303, 29), 'names', False)
    # Processing the call keyword arguments (line 303)
    kwargs_13931 = {}
    # Getting the type of 'enumerate' (line 303)
    enumerate_13929 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 303, 19), 'enumerate', False)
    # Calling enumerate(args, kwargs) (line 303)
    enumerate_call_result_13932 = invoke(stypy.reporting.localization.Localization(__file__, 303, 19), enumerate_13929, *[names_13930], **kwargs_13931)
    
    # Testing the type of a for loop iterable (line 303)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 303, 4), enumerate_call_result_13932)
    # Getting the type of the for loop variable (line 303)
    for_loop_var_13933 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 303, 4), enumerate_call_result_13932)
    # Assigning a type to the variable 'i' (line 303)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 303, 4), 'i', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 303, 4), for_loop_var_13933))
    # Assigning a type to the variable 'name' (line 303)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 303, 4), 'name', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 303, 4), for_loop_var_13933))
    # SSA begins for a for statement (line 303)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a BinOp to a Name (line 304):
    
    # Assigning a BinOp to a Name (line 304):
    # Getting the type of 'prefix' (line 304)
    prefix_13934 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 304, 20), 'prefix')
    # Getting the type of 'name' (line 304)
    name_13935 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 304, 29), 'name')
    # Applying the binary operator '+' (line 304)
    result_add_13936 = python_operator(stypy.reporting.localization.Localization(__file__, 304, 20), '+', prefix_13934, name_13935)
    
    # Assigning a type to the variable 'func_name' (line 304)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 304, 8), 'func_name', result_add_13936)
    
    # Assigning a Call to a Name (line 305):
    
    # Assigning a Call to a Name (line 305):
    
    # Call to get(...): (line 305)
    # Processing the call arguments (line 305)
    # Getting the type of 'func_name' (line 305)
    func_name_13939 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 305, 30), 'func_name', False)
    # Getting the type of 'func_name' (line 305)
    func_name_13940 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 305, 41), 'func_name', False)
    # Processing the call keyword arguments (line 305)
    kwargs_13941 = {}
    # Getting the type of 'alias' (line 305)
    alias_13937 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 305, 20), 'alias', False)
    # Obtaining the member 'get' of a type (line 305)
    get_13938 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 305, 20), alias_13937, 'get')
    # Calling get(args, kwargs) (line 305)
    get_call_result_13942 = invoke(stypy.reporting.localization.Localization(__file__, 305, 20), get_13938, *[func_name_13939, func_name_13940], **kwargs_13941)
    
    # Assigning a type to the variable 'func_name' (line 305)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 305, 8), 'func_name', get_call_result_13942)
    
    # Assigning a Call to a Name (line 306):
    
    # Assigning a Call to a Name (line 306):
    
    # Call to getattr(...): (line 306)
    # Processing the call arguments (line 306)
    
    # Obtaining the type of the subscript
    int_13944 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 306, 31), 'int')
    # Getting the type of 'module1' (line 306)
    module1_13945 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 306, 23), 'module1', False)
    # Obtaining the member '__getitem__' of a type (line 306)
    getitem___13946 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 306, 23), module1_13945, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 306)
    subscript_call_result_13947 = invoke(stypy.reporting.localization.Localization(__file__, 306, 23), getitem___13946, int_13944)
    
    # Getting the type of 'func_name' (line 306)
    func_name_13948 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 306, 35), 'func_name', False)
    # Getting the type of 'None' (line 306)
    None_13949 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 306, 46), 'None', False)
    # Processing the call keyword arguments (line 306)
    kwargs_13950 = {}
    # Getting the type of 'getattr' (line 306)
    getattr_13943 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 306, 15), 'getattr', False)
    # Calling getattr(args, kwargs) (line 306)
    getattr_call_result_13951 = invoke(stypy.reporting.localization.Localization(__file__, 306, 15), getattr_13943, *[subscript_call_result_13947, func_name_13948, None_13949], **kwargs_13950)
    
    # Assigning a type to the variable 'func' (line 306)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 306, 8), 'func', getattr_call_result_13951)
    
    # Assigning a Subscript to a Name (line 307):
    
    # Assigning a Subscript to a Name (line 307):
    
    # Obtaining the type of the subscript
    int_13952 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 307, 30), 'int')
    # Getting the type of 'module1' (line 307)
    module1_13953 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 307, 22), 'module1')
    # Obtaining the member '__getitem__' of a type (line 307)
    getitem___13954 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 307, 22), module1_13953, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 307)
    subscript_call_result_13955 = invoke(stypy.reporting.localization.Localization(__file__, 307, 22), getitem___13954, int_13952)
    
    # Assigning a type to the variable 'module_name' (line 307)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 307, 8), 'module_name', subscript_call_result_13955)
    
    # Type idiom detected: calculating its left and rigth part (line 308)
    # Getting the type of 'func' (line 308)
    func_13956 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 308, 11), 'func')
    # Getting the type of 'None' (line 308)
    None_13957 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 308, 19), 'None')
    
    (may_be_13958, more_types_in_union_13959) = may_be_none(func_13956, None_13957)

    if may_be_13958:

        if more_types_in_union_13959:
            # Runtime conditional SSA (line 308)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a Call to a Name (line 309):
        
        # Assigning a Call to a Name (line 309):
        
        # Call to getattr(...): (line 309)
        # Processing the call arguments (line 309)
        
        # Obtaining the type of the subscript
        int_13961 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 309, 35), 'int')
        # Getting the type of 'module2' (line 309)
        module2_13962 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 309, 27), 'module2', False)
        # Obtaining the member '__getitem__' of a type (line 309)
        getitem___13963 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 309, 27), module2_13962, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 309)
        subscript_call_result_13964 = invoke(stypy.reporting.localization.Localization(__file__, 309, 27), getitem___13963, int_13961)
        
        # Getting the type of 'func_name' (line 309)
        func_name_13965 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 309, 39), 'func_name', False)
        # Getting the type of 'None' (line 309)
        None_13966 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 309, 50), 'None', False)
        # Processing the call keyword arguments (line 309)
        kwargs_13967 = {}
        # Getting the type of 'getattr' (line 309)
        getattr_13960 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 309, 19), 'getattr', False)
        # Calling getattr(args, kwargs) (line 309)
        getattr_call_result_13968 = invoke(stypy.reporting.localization.Localization(__file__, 309, 19), getattr_13960, *[subscript_call_result_13964, func_name_13965, None_13966], **kwargs_13967)
        
        # Assigning a type to the variable 'func' (line 309)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 309, 12), 'func', getattr_call_result_13968)
        
        # Assigning a Subscript to a Name (line 310):
        
        # Assigning a Subscript to a Name (line 310):
        
        # Obtaining the type of the subscript
        int_13969 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 310, 34), 'int')
        # Getting the type of 'module2' (line 310)
        module2_13970 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 310, 26), 'module2')
        # Obtaining the member '__getitem__' of a type (line 310)
        getitem___13971 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 310, 26), module2_13970, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 310)
        subscript_call_result_13972 = invoke(stypy.reporting.localization.Localization(__file__, 310, 26), getitem___13971, int_13969)
        
        # Assigning a type to the variable 'module_name' (line 310)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 310, 12), 'module_name', subscript_call_result_13972)

        if more_types_in_union_13959:
            # SSA join for if statement (line 308)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Type idiom detected: calculating its left and rigth part (line 311)
    # Getting the type of 'func' (line 311)
    func_13973 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 311, 11), 'func')
    # Getting the type of 'None' (line 311)
    None_13974 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 311, 19), 'None')
    
    (may_be_13975, more_types_in_union_13976) = may_be_none(func_13973, None_13974)

    if may_be_13975:

        if more_types_in_union_13976:
            # Runtime conditional SSA (line 311)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Call to ValueError(...): (line 312)
        # Processing the call arguments (line 312)
        str_13978 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 313, 16), 'str', '%s function %s could not be found')
        
        # Obtaining an instance of the builtin type 'tuple' (line 313)
        tuple_13979 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 313, 55), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 313)
        # Adding element type (line 313)
        # Getting the type of 'lib_name' (line 313)
        lib_name_13980 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 313, 55), 'lib_name', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 313, 55), tuple_13979, lib_name_13980)
        # Adding element type (line 313)
        # Getting the type of 'func_name' (line 313)
        func_name_13981 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 313, 65), 'func_name', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 313, 55), tuple_13979, func_name_13981)
        
        # Applying the binary operator '%' (line 313)
        result_mod_13982 = python_operator(stypy.reporting.localization.Localization(__file__, 313, 16), '%', str_13978, tuple_13979)
        
        # Processing the call keyword arguments (line 312)
        kwargs_13983 = {}
        # Getting the type of 'ValueError' (line 312)
        ValueError_13977 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 312, 18), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 312)
        ValueError_call_result_13984 = invoke(stypy.reporting.localization.Localization(__file__, 312, 18), ValueError_13977, *[result_mod_13982], **kwargs_13983)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 312, 12), ValueError_call_result_13984, 'raise parameter', BaseException)

        if more_types_in_union_13976:
            # SSA join for if statement (line 311)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Assigning a Tuple to a Tuple (line 314):
    
    # Assigning a Name to a Name (line 314):
    # Getting the type of 'module_name' (line 314)
    module_name_13985 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 314, 42), 'module_name')
    # Assigning a type to the variable 'tuple_assignment_13757' (line 314)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 314, 8), 'tuple_assignment_13757', module_name_13985)
    
    # Assigning a Name to a Name (line 314):
    # Getting the type of 'prefix' (line 314)
    prefix_13986 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 314, 55), 'prefix')
    # Assigning a type to the variable 'tuple_assignment_13758' (line 314)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 314, 8), 'tuple_assignment_13758', prefix_13986)
    
    # Assigning a Name to a Attribute (line 314):
    # Getting the type of 'tuple_assignment_13757' (line 314)
    tuple_assignment_13757_13987 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 314, 8), 'tuple_assignment_13757')
    # Getting the type of 'func' (line 314)
    func_13988 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 314, 8), 'func')
    # Setting the type of the member 'module_name' of a type (line 314)
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 314, 8), func_13988, 'module_name', tuple_assignment_13757_13987)
    
    # Assigning a Name to a Attribute (line 314):
    # Getting the type of 'tuple_assignment_13758' (line 314)
    tuple_assignment_13758_13989 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 314, 8), 'tuple_assignment_13758')
    # Getting the type of 'func' (line 314)
    func_13990 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 314, 26), 'func')
    # Setting the type of the member 'typecode' of a type (line 314)
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 314, 26), func_13990, 'typecode', tuple_assignment_13758_13989)
    
    # Assigning a Name to a Attribute (line 315):
    
    # Assigning a Name to a Attribute (line 315):
    # Getting the type of 'dtype' (line 315)
    dtype_13991 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 315, 21), 'dtype')
    # Getting the type of 'func' (line 315)
    func_13992 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 315, 8), 'func')
    # Setting the type of the member 'dtype' of a type (line 315)
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 315, 8), func_13992, 'dtype', dtype_13991)
    
    # Assigning a Name to a Attribute (line 316):
    
    # Assigning a Name to a Attribute (line 316):
    # Getting the type of 'prefix' (line 316)
    prefix_13993 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 316, 22), 'prefix')
    # Getting the type of 'func' (line 316)
    func_13994 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 316, 8), 'func')
    # Setting the type of the member 'prefix' of a type (line 316)
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 316, 8), func_13994, 'prefix', prefix_13993)
    
    # Call to append(...): (line 317)
    # Processing the call arguments (line 317)
    # Getting the type of 'func' (line 317)
    func_13997 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 317, 21), 'func', False)
    # Processing the call keyword arguments (line 317)
    kwargs_13998 = {}
    # Getting the type of 'funcs' (line 317)
    funcs_13995 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 317, 8), 'funcs', False)
    # Obtaining the member 'append' of a type (line 317)
    append_13996 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 317, 8), funcs_13995, 'append')
    # Calling append(args, kwargs) (line 317)
    append_call_result_13999 = invoke(stypy.reporting.localization.Localization(__file__, 317, 8), append_13996, *[func_13997], **kwargs_13998)
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Getting the type of 'unpack' (line 319)
    unpack_14000 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 319, 7), 'unpack')
    # Testing the type of an if condition (line 319)
    if_condition_14001 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 319, 4), unpack_14000)
    # Assigning a type to the variable 'if_condition_14001' (line 319)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 319, 4), 'if_condition_14001', if_condition_14001)
    # SSA begins for if statement (line 319)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Obtaining the type of the subscript
    int_14002 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 320, 21), 'int')
    # Getting the type of 'funcs' (line 320)
    funcs_14003 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 320, 15), 'funcs')
    # Obtaining the member '__getitem__' of a type (line 320)
    getitem___14004 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 320, 15), funcs_14003, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 320)
    subscript_call_result_14005 = invoke(stypy.reporting.localization.Localization(__file__, 320, 15), getitem___14004, int_14002)
    
    # Assigning a type to the variable 'stypy_return_type' (line 320)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 320, 8), 'stypy_return_type', subscript_call_result_14005)
    # SSA branch for the else part of an if statement (line 319)
    module_type_store.open_ssa_branch('else')
    # Getting the type of 'funcs' (line 322)
    funcs_14006 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 322, 15), 'funcs')
    # Assigning a type to the variable 'stypy_return_type' (line 322)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 322, 8), 'stypy_return_type', funcs_14006)
    # SSA join for if statement (line 319)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of '_get_funcs(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_get_funcs' in the type store
    # Getting the type of 'stypy_return_type' (line 279)
    stypy_return_type_14007 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 279, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_14007)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_get_funcs'
    return stypy_return_type_14007

# Assigning a type to the variable '_get_funcs' (line 279)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 279, 0), '_get_funcs', _get_funcs)

@norecursion
def get_blas_funcs(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    
    # Obtaining an instance of the builtin type 'tuple' (line 325)
    tuple_14008 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 325, 33), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 325)
    
    # Getting the type of 'None' (line 325)
    None_14009 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 325, 43), 'None')
    defaults = [tuple_14008, None_14009]
    # Create a new context for function 'get_blas_funcs'
    module_type_store = module_type_store.open_function_context('get_blas_funcs', 325, 0, False)
    
    # Passed parameters checking function
    get_blas_funcs.stypy_localization = localization
    get_blas_funcs.stypy_type_of_self = None
    get_blas_funcs.stypy_type_store = module_type_store
    get_blas_funcs.stypy_function_name = 'get_blas_funcs'
    get_blas_funcs.stypy_param_names_list = ['names', 'arrays', 'dtype']
    get_blas_funcs.stypy_varargs_param_name = None
    get_blas_funcs.stypy_kwargs_param_name = None
    get_blas_funcs.stypy_call_defaults = defaults
    get_blas_funcs.stypy_call_varargs = varargs
    get_blas_funcs.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'get_blas_funcs', ['names', 'arrays', 'dtype'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'get_blas_funcs', localization, ['names', 'arrays', 'dtype'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'get_blas_funcs(...)' code ##################

    str_14010 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 374, (-1)), 'str', "Return available BLAS function objects from names.\n\n    Arrays are used to determine the optimal prefix of BLAS routines.\n\n    Parameters\n    ----------\n    names : str or sequence of str\n        Name(s) of BLAS functions without type prefix.\n\n    arrays : sequence of ndarrays, optional\n        Arrays can be given to determine optimal prefix of BLAS\n        routines. If not given, double-precision routines will be\n        used, otherwise the most generic type in arrays will be used.\n\n    dtype : str or dtype, optional\n        Data-type specifier. Not used if `arrays` is non-empty.\n\n\n    Returns\n    -------\n    funcs : list\n        List containing the found function(s).\n\n\n    Notes\n    -----\n    This routine automatically chooses between Fortran/C\n    interfaces. Fortran code is used whenever possible for arrays with\n    column major order. In all other cases, C code is preferred.\n\n    In BLAS, the naming convention is that all functions start with a\n    type prefix, which depends on the type of the principal\n    matrix. These can be one of {'s', 'd', 'c', 'z'} for the numpy\n    types {float32, float64, complex64, complex128} respectively.\n    The code and the dtype are stored in attributes `typecode` and `dtype`\n    of the returned functions.\n\n    Examples\n    --------\n    >>> import scipy.linalg as LA\n    >>> a = np.random.rand(3,2)\n    >>> x_gemv = LA.get_blas_funcs('gemv', (a,))\n    >>> x_gemv.typecode\n    'd'\n    >>> x_gemv = LA.get_blas_funcs('gemv',(a*1j,))\n    >>> x_gemv.typecode\n    'z'\n\n    ")
    
    # Call to _get_funcs(...): (line 375)
    # Processing the call arguments (line 375)
    # Getting the type of 'names' (line 375)
    names_14012 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 375, 22), 'names', False)
    # Getting the type of 'arrays' (line 375)
    arrays_14013 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 375, 29), 'arrays', False)
    # Getting the type of 'dtype' (line 375)
    dtype_14014 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 375, 37), 'dtype', False)
    str_14015 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 376, 22), 'str', 'BLAS')
    # Getting the type of '_fblas' (line 376)
    _fblas_14016 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 376, 30), '_fblas', False)
    # Getting the type of '_cblas' (line 376)
    _cblas_14017 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 376, 38), '_cblas', False)
    str_14018 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 376, 46), 'str', 'fblas')
    str_14019 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 376, 55), 'str', 'cblas')
    # Getting the type of '_blas_alias' (line 377)
    _blas_alias_14020 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 377, 22), '_blas_alias', False)
    # Processing the call keyword arguments (line 375)
    kwargs_14021 = {}
    # Getting the type of '_get_funcs' (line 375)
    _get_funcs_14011 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 375, 11), '_get_funcs', False)
    # Calling _get_funcs(args, kwargs) (line 375)
    _get_funcs_call_result_14022 = invoke(stypy.reporting.localization.Localization(__file__, 375, 11), _get_funcs_14011, *[names_14012, arrays_14013, dtype_14014, str_14015, _fblas_14016, _cblas_14017, str_14018, str_14019, _blas_alias_14020], **kwargs_14021)
    
    # Assigning a type to the variable 'stypy_return_type' (line 375)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 375, 4), 'stypy_return_type', _get_funcs_call_result_14022)
    
    # ################# End of 'get_blas_funcs(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'get_blas_funcs' in the type store
    # Getting the type of 'stypy_return_type' (line 325)
    stypy_return_type_14023 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 325, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_14023)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'get_blas_funcs'
    return stypy_return_type_14023

# Assigning a type to the variable 'get_blas_funcs' (line 325)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 325, 0), 'get_blas_funcs', get_blas_funcs)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
