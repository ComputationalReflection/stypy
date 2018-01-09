
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: from __future__ import division, print_function, absolute_import
2: 
3: import warnings
4: 
5: import numpy as np
6: from numpy import asarray_chkfinite
7: 
8: from .misc import LinAlgError, _datacopied
9: from .lapack import get_lapack_funcs
10: 
11: from scipy._lib.six import callable
12: 
13: __all__ = ['qz', 'ordqz']
14: 
15: _double_precision = ['i', 'l', 'd']
16: 
17: 
18: def _select_function(sort):
19:     if callable(sort):
20:         # assume the user knows what they're doing
21:         sfunction = sort
22:     elif sort == 'lhp':
23:         sfunction = _lhp
24:     elif sort == 'rhp':
25:         sfunction = _rhp
26:     elif sort == 'iuc':
27:         sfunction = _iuc
28:     elif sort == 'ouc':
29:         sfunction = _ouc
30:     else:
31:         raise ValueError("sort parameter must be None, a callable, or "
32:                          "one of ('lhp','rhp','iuc','ouc')")
33: 
34:     return sfunction
35: 
36: 
37: def _lhp(x, y):
38:     out = np.empty_like(x, dtype=bool)
39:     nonzero = (y != 0)
40:     # handles (x, y) = (0, 0) too
41:     out[~nonzero] = False
42:     out[nonzero] = (np.real(x[nonzero]/y[nonzero]) < 0.0)
43:     return out
44: 
45: 
46: def _rhp(x, y):
47:     out = np.empty_like(x, dtype=bool)
48:     nonzero = (y != 0)
49:     # handles (x, y) = (0, 0) too
50:     out[~nonzero] = False
51:     out[nonzero] = (np.real(x[nonzero]/y[nonzero]) > 0.0)
52:     return out
53: 
54: 
55: def _iuc(x, y):
56:     out = np.empty_like(x, dtype=bool)
57:     nonzero = (y != 0)
58:     # handles (x, y) = (0, 0) too
59:     out[~nonzero] = False
60:     out[nonzero] = (abs(x[nonzero]/y[nonzero]) < 1.0)
61:     return out
62: 
63: 
64: def _ouc(x, y):
65:     out = np.empty_like(x, dtype=bool)
66:     xzero = (x == 0)
67:     yzero = (y == 0)
68:     out[xzero & yzero] = False
69:     out[~xzero & yzero] = True
70:     out[~yzero] = (abs(x[~yzero]/y[~yzero]) > 1.0)
71:     return out
72: 
73: 
74: def _qz(A, B, output='real', lwork=None, sort=None, overwrite_a=False,
75:         overwrite_b=False, check_finite=True):
76:     if sort is not None:
77:         # Disabled due to segfaults on win32, see ticket 1717.
78:         raise ValueError("The 'sort' input of qz() has to be None and will be "
79:                          "removed in a future release. Use ordqz instead.")
80: 
81:     if output not in ['real', 'complex', 'r', 'c']:
82:         raise ValueError("argument must be 'real', or 'complex'")
83: 
84:     if check_finite:
85:         a1 = asarray_chkfinite(A)
86:         b1 = asarray_chkfinite(B)
87:     else:
88:         a1 = np.asarray(A)
89:         b1 = np.asarray(B)
90: 
91:     a_m, a_n = a1.shape
92:     b_m, b_n = b1.shape
93:     if not (a_m == a_n == b_m == b_n):
94:         raise ValueError("Array dimensions must be square and agree")
95: 
96:     typa = a1.dtype.char
97:     if output in ['complex', 'c'] and typa not in ['F', 'D']:
98:         if typa in _double_precision:
99:             a1 = a1.astype('D')
100:             typa = 'D'
101:         else:
102:             a1 = a1.astype('F')
103:             typa = 'F'
104:     typb = b1.dtype.char
105:     if output in ['complex', 'c'] and typb not in ['F', 'D']:
106:         if typb in _double_precision:
107:             b1 = b1.astype('D')
108:             typb = 'D'
109:         else:
110:             b1 = b1.astype('F')
111:             typb = 'F'
112: 
113:     overwrite_a = overwrite_a or (_datacopied(a1, A))
114:     overwrite_b = overwrite_b or (_datacopied(b1, B))
115: 
116:     gges, = get_lapack_funcs(('gges',), (a1, b1))
117: 
118:     if lwork is None or lwork == -1:
119:         # get optimal work array size
120:         result = gges(lambda x: None, a1, b1, lwork=-1)
121:         lwork = result[-2][0].real.astype(np.int)
122: 
123:     sfunction = lambda x: None
124:     result = gges(sfunction, a1, b1, lwork=lwork, overwrite_a=overwrite_a,
125:                   overwrite_b=overwrite_b, sort_t=0)
126: 
127:     info = result[-1]
128:     if info < 0:
129:         raise ValueError("Illegal value in argument %d of gges" % -info)
130:     elif info > 0 and info <= a_n:
131:         warnings.warn("The QZ iteration failed. (a,b) are not in Schur "
132:                       "form, but ALPHAR(j), ALPHAI(j), and BETA(j) should be "
133:                       "correct for J=%d,...,N" % info-1, UserWarning)
134:     elif info == a_n+1:
135:         raise LinAlgError("Something other than QZ iteration failed")
136:     elif info == a_n+2:
137:         raise LinAlgError("After reordering, roundoff changed values of some "
138:                           "complex eigenvalues so that leading eigenvalues "
139:                           "in the Generalized Schur form no longer satisfy "
140:                           "sort=True. This could also be due to scaling.")
141:     elif info == a_n+3:
142:         raise LinAlgError("Reordering failed in <s,d,c,z>tgsen")
143: 
144:     return result, gges.typecode
145: 
146: 
147: def qz(A, B, output='real', lwork=None, sort=None, overwrite_a=False,
148:        overwrite_b=False, check_finite=True):
149:     '''
150:     QZ decomposition for generalized eigenvalues of a pair of matrices.
151: 
152:     The QZ, or generalized Schur, decomposition for a pair of N x N
153:     nonsymmetric matrices (A,B) is::
154: 
155:         (A,B) = (Q*AA*Z', Q*BB*Z')
156: 
157:     where AA, BB is in generalized Schur form if BB is upper-triangular
158:     with non-negative diagonal and AA is upper-triangular, or for real QZ
159:     decomposition (``output='real'``) block upper triangular with 1x1
160:     and 2x2 blocks.  In this case, the 1x1 blocks correspond to real
161:     generalized eigenvalues and 2x2 blocks are 'standardized' by making
162:     the corresponding elements of BB have the form::
163: 
164:         [ a 0 ]
165:         [ 0 b ]
166: 
167:     and the pair of corresponding 2x2 blocks in AA and BB will have a complex
168:     conjugate pair of generalized eigenvalues.  If (``output='complex'``) or
169:     A and B are complex matrices, Z' denotes the conjugate-transpose of Z.
170:     Q and Z are unitary matrices.
171: 
172:     Parameters
173:     ----------
174:     A : (N, N) array_like
175:         2d array to decompose
176:     B : (N, N) array_like
177:         2d array to decompose
178:     output : {'real', 'complex'}, optional
179:         Construct the real or complex QZ decomposition for real matrices.
180:         Default is 'real'.
181:     lwork : int, optional
182:         Work array size.  If None or -1, it is automatically computed.
183:     sort : {None, callable, 'lhp', 'rhp', 'iuc', 'ouc'}, optional
184:         NOTE: THIS INPUT IS DISABLED FOR NOW. Use ordqz instead.
185: 
186:         Specifies whether the upper eigenvalues should be sorted.  A callable
187:         may be passed that, given a eigenvalue, returns a boolean denoting
188:         whether the eigenvalue should be sorted to the top-left (True). For
189:         real matrix pairs, the sort function takes three real arguments
190:         (alphar, alphai, beta). The eigenvalue
191:         ``x = (alphar + alphai*1j)/beta``.  For complex matrix pairs or
192:         output='complex', the sort function takes two complex arguments
193:         (alpha, beta). The eigenvalue ``x = (alpha/beta)``.  Alternatively,
194:         string parameters may be used:
195: 
196:             - 'lhp'   Left-hand plane (x.real < 0.0)
197:             - 'rhp'   Right-hand plane (x.real > 0.0)
198:             - 'iuc'   Inside the unit circle (x*x.conjugate() < 1.0)
199:             - 'ouc'   Outside the unit circle (x*x.conjugate() > 1.0)
200: 
201:         Defaults to None (no sorting).
202:     overwrite_a : bool, optional
203:         Whether to overwrite data in a (may improve performance)
204:     overwrite_b : bool, optional
205:         Whether to overwrite data in b (may improve performance)
206:     check_finite : bool, optional
207:         If true checks the elements of `A` and `B` are finite numbers. If
208:         false does no checking and passes matrix through to
209:         underlying algorithm.
210: 
211:     Returns
212:     -------
213:     AA : (N, N) ndarray
214:         Generalized Schur form of A.
215:     BB : (N, N) ndarray
216:         Generalized Schur form of B.
217:     Q : (N, N) ndarray
218:         The left Schur vectors.
219:     Z : (N, N) ndarray
220:         The right Schur vectors.
221: 
222:     Notes
223:     -----
224:     Q is transposed versus the equivalent function in Matlab.
225: 
226:     .. versionadded:: 0.11.0
227: 
228:     Examples
229:     --------
230:     >>> from scipy import linalg
231:     >>> np.random.seed(1234)
232:     >>> A = np.arange(9).reshape((3, 3))
233:     >>> B = np.random.randn(3, 3)
234: 
235:     >>> AA, BB, Q, Z = linalg.qz(A, B)
236:     >>> AA
237:     array([[-13.40928183,  -4.62471562,   1.09215523],
238:            [  0.        ,   0.        ,   1.22805978],
239:            [  0.        ,   0.        ,   0.31973817]])
240:     >>> BB
241:     array([[ 0.33362547, -1.37393632,  0.02179805],
242:            [ 0.        ,  1.68144922,  0.74683866],
243:            [ 0.        ,  0.        ,  0.9258294 ]])
244:     >>> Q
245:     array([[ 0.14134727, -0.97562773,  0.16784365],
246:            [ 0.49835904, -0.07636948, -0.86360059],
247:            [ 0.85537081,  0.20571399,  0.47541828]])
248:     >>> Z
249:     array([[-0.24900855, -0.51772687,  0.81850696],
250:            [-0.79813178,  0.58842606,  0.12938478],
251:            [-0.54861681, -0.6210585 , -0.55973739]])
252: 
253:     See also
254:     --------
255:     ordqz
256:     '''
257:     # output for real
258:     # AA, BB, sdim, alphar, alphai, beta, vsl, vsr, work, info
259:     # output for complex
260:     # AA, BB, sdim, alpha, beta, vsl, vsr, work, info
261:     result, _ = _qz(A, B, output=output, lwork=lwork, sort=sort,
262:                     overwrite_a=overwrite_a, overwrite_b=overwrite_b,
263:                     check_finite=check_finite)
264:     return result[0], result[1], result[-4], result[-3]
265: 
266: 
267: def ordqz(A, B, sort='lhp', output='real', overwrite_a=False,
268:           overwrite_b=False, check_finite=True):
269:     '''QZ decomposition for a pair of matrices with reordering.
270: 
271:     .. versionadded:: 0.17.0
272: 
273:     Parameters
274:     ----------
275:     A : (N, N) array_like
276:         2d array to decompose
277:     B : (N, N) array_like
278:         2d array to decompose
279:     sort : {callable, 'lhp', 'rhp', 'iuc', 'ouc'}, optional
280:         Specifies whether the upper eigenvalues should be sorted. A
281:         callable may be passed that, given an ordered pair ``(alpha,
282:         beta)`` representing the eigenvalue ``x = (alpha/beta)``,
283:         returns a boolean denoting whether the eigenvalue should be
284:         sorted to the top-left (True). For the real matrix pairs
285:         ``beta`` is real while ``alpha`` can be complex, and for
286:         complex matrix pairs both ``alpha`` and ``beta`` can be
287:         complex. The callable must be able to accept a numpy
288:         array. Alternatively, string parameters may be used:
289: 
290:             - 'lhp'   Left-hand plane (x.real < 0.0)
291:             - 'rhp'   Right-hand plane (x.real > 0.0)
292:             - 'iuc'   Inside the unit circle (x*x.conjugate() < 1.0)
293:             - 'ouc'   Outside the unit circle (x*x.conjugate() > 1.0)
294: 
295:         With the predefined sorting functions, an infinite eigenvalue
296:         (i.e. ``alpha != 0`` and ``beta = 0``) is considered to lie in
297:         neither the left-hand nor the right-hand plane, but it is
298:         considered to lie outside the unit circle. For the eigenvalue
299:         ``(alpha, beta) = (0, 0)`` the predefined sorting functions
300:         all return `False`.
301: 
302:     output : str {'real','complex'}, optional
303:         Construct the real or complex QZ decomposition for real matrices.
304:         Default is 'real'.
305:     overwrite_a : bool, optional
306:         If True, the contents of A are overwritten.
307:     overwrite_b : bool, optional
308:         If True, the contents of B are overwritten.
309:     check_finite : bool, optional
310:         If true checks the elements of `A` and `B` are finite numbers. If
311:         false does no checking and passes matrix through to
312:         underlying algorithm.
313: 
314:     Returns
315:     -------
316:     AA : (N, N) ndarray
317:         Generalized Schur form of A.
318:     BB : (N, N) ndarray
319:         Generalized Schur form of B.
320:     alpha : (N,) ndarray
321:         alpha = alphar + alphai * 1j. See notes.
322:     beta : (N,) ndarray
323:         See notes.
324:     Q : (N, N) ndarray
325:         The left Schur vectors.
326:     Z : (N, N) ndarray
327:         The right Schur vectors.
328: 
329:     Notes
330:     -----
331:     On exit, ``(ALPHAR(j) + ALPHAI(j)*i)/BETA(j), j=1,...,N``, will be the
332:     generalized eigenvalues.  ``ALPHAR(j) + ALPHAI(j)*i`` and
333:     ``BETA(j),j=1,...,N`` are the diagonals of the complex Schur form (S,T)
334:     that would result if the 2-by-2 diagonal blocks of the real generalized
335:     Schur form of (A,B) were further reduced to triangular form using complex
336:     unitary transformations. If ALPHAI(j) is zero, then the j-th eigenvalue is
337:     real; if positive, then the ``j``-th and ``(j+1)``-st eigenvalues are a complex
338:     conjugate pair, with ``ALPHAI(j+1)`` negative.
339: 
340:     See also
341:     --------
342:     qz
343: 
344:     '''
345:     #NOTE: should users be able to set these?
346:     lwork = None
347:     result, typ = _qz(A, B, output=output, lwork=lwork, sort=None,
348:                       overwrite_a=overwrite_a, overwrite_b=overwrite_b,
349:                       check_finite=check_finite)
350:     AA, BB, Q, Z = result[0], result[1], result[-4], result[-3]
351:     if typ not in 'cz':
352:         alpha, beta = result[3] + result[4]*1.j, result[5]
353:     else:
354:         alpha, beta = result[3], result[4]
355: 
356:     sfunction = _select_function(sort)
357:     select = sfunction(alpha, beta)
358: 
359:     tgsen, = get_lapack_funcs(('tgsen',), (AA, BB))
360: 
361:     if lwork is None or lwork == -1:
362:         result = tgsen(select, AA, BB, Q, Z, lwork=-1)
363:         lwork = result[-3][0].real.astype(np.int)
364:         # looks like wrong value passed to ZTGSYL if not
365:         lwork += 1
366: 
367:     liwork = None
368:     if liwork is None or liwork == -1:
369:         result = tgsen(select, AA, BB, Q, Z, liwork=-1)
370:         liwork = result[-2][0]
371: 
372:     result = tgsen(select, AA, BB, Q, Z, lwork=lwork, liwork=liwork)
373: 
374:     info = result[-1]
375:     if info < 0:
376:         raise ValueError("Illegal value in argument %d of tgsen" % -info)
377:     elif info == 1:
378:         raise ValueError("Reordering of (A, B) failed because the transformed"
379:                          " matrix pair (A, B) would be too far from "
380:                          "generalized Schur form; the problem is very "
381:                          "ill-conditioned. (A, B) may have been partially "
382:                          "reorded. If requested, 0 is returned in DIF(*), "
383:                          "PL, and PR.")
384: 
385:     # for real results has a, b, alphar, alphai, beta, q, z, m, pl, pr, dif,
386:     # work, iwork, info
387:     if typ in ['f', 'd']:
388:         alpha = result[2] + result[3] * 1.j
389:         return (result[0], result[1], alpha, result[4], result[5], result[6])
390:     # for complex results has a, b, alpha, beta, q, z, m, pl, pr, dif, work,
391:     # iwork, info
392:     else:
393:         return result[0], result[1], result[2], result[3], result[4], result[5]
394: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 3, 0))

# 'import warnings' statement (line 3)
import warnings

import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'warnings', warnings, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 5, 0))

# 'import numpy' statement (line 5)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/linalg/')
import_25828 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'numpy')

if (type(import_25828) is not StypyTypeError):

    if (import_25828 != 'pyd_module'):
        __import__(import_25828)
        sys_modules_25829 = sys.modules[import_25828]
        import_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'np', sys_modules_25829.module_type_store, module_type_store)
    else:
        import numpy as np

        import_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'np', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 5)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 0), 'numpy', import_25828)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/linalg/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 6, 0))

# 'from numpy import asarray_chkfinite' statement (line 6)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/linalg/')
import_25830 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'numpy')

if (type(import_25830) is not StypyTypeError):

    if (import_25830 != 'pyd_module'):
        __import__(import_25830)
        sys_modules_25831 = sys.modules[import_25830]
        import_from_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'numpy', sys_modules_25831.module_type_store, module_type_store, ['asarray_chkfinite'])
        nest_module(stypy.reporting.localization.Localization(__file__, 6, 0), __file__, sys_modules_25831, sys_modules_25831.module_type_store, module_type_store)
    else:
        from numpy import asarray_chkfinite

        import_from_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'numpy', None, module_type_store, ['asarray_chkfinite'], [asarray_chkfinite])

else:
    # Assigning a type to the variable 'numpy' (line 6)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), 'numpy', import_25830)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/linalg/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 8, 0))

# 'from scipy.linalg.misc import LinAlgError, _datacopied' statement (line 8)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/linalg/')
import_25832 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'scipy.linalg.misc')

if (type(import_25832) is not StypyTypeError):

    if (import_25832 != 'pyd_module'):
        __import__(import_25832)
        sys_modules_25833 = sys.modules[import_25832]
        import_from_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'scipy.linalg.misc', sys_modules_25833.module_type_store, module_type_store, ['LinAlgError', '_datacopied'])
        nest_module(stypy.reporting.localization.Localization(__file__, 8, 0), __file__, sys_modules_25833, sys_modules_25833.module_type_store, module_type_store)
    else:
        from scipy.linalg.misc import LinAlgError, _datacopied

        import_from_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'scipy.linalg.misc', None, module_type_store, ['LinAlgError', '_datacopied'], [LinAlgError, _datacopied])

else:
    # Assigning a type to the variable 'scipy.linalg.misc' (line 8)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 0), 'scipy.linalg.misc', import_25832)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/linalg/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 9, 0))

# 'from scipy.linalg.lapack import get_lapack_funcs' statement (line 9)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/linalg/')
import_25834 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'scipy.linalg.lapack')

if (type(import_25834) is not StypyTypeError):

    if (import_25834 != 'pyd_module'):
        __import__(import_25834)
        sys_modules_25835 = sys.modules[import_25834]
        import_from_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'scipy.linalg.lapack', sys_modules_25835.module_type_store, module_type_store, ['get_lapack_funcs'])
        nest_module(stypy.reporting.localization.Localization(__file__, 9, 0), __file__, sys_modules_25835, sys_modules_25835.module_type_store, module_type_store)
    else:
        from scipy.linalg.lapack import get_lapack_funcs

        import_from_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'scipy.linalg.lapack', None, module_type_store, ['get_lapack_funcs'], [get_lapack_funcs])

else:
    # Assigning a type to the variable 'scipy.linalg.lapack' (line 9)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 0), 'scipy.linalg.lapack', import_25834)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/linalg/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 11, 0))

# 'from scipy._lib.six import callable' statement (line 11)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/linalg/')
import_25836 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'scipy._lib.six')

if (type(import_25836) is not StypyTypeError):

    if (import_25836 != 'pyd_module'):
        __import__(import_25836)
        sys_modules_25837 = sys.modules[import_25836]
        import_from_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'scipy._lib.six', sys_modules_25837.module_type_store, module_type_store, ['callable'])
        nest_module(stypy.reporting.localization.Localization(__file__, 11, 0), __file__, sys_modules_25837, sys_modules_25837.module_type_store, module_type_store)
    else:
        from scipy._lib.six import callable

        import_from_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'scipy._lib.six', None, module_type_store, ['callable'], [callable])

else:
    # Assigning a type to the variable 'scipy._lib.six' (line 11)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 0), 'scipy._lib.six', import_25836)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/linalg/')


# Assigning a List to a Name (line 13):

# Assigning a List to a Name (line 13):
__all__ = ['qz', 'ordqz']
module_type_store.set_exportable_members(['qz', 'ordqz'])

# Obtaining an instance of the builtin type 'list' (line 13)
list_25838 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 10), 'list')
# Adding type elements to the builtin type 'list' instance (line 13)
# Adding element type (line 13)
str_25839 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 11), 'str', 'qz')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 13, 10), list_25838, str_25839)
# Adding element type (line 13)
str_25840 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 17), 'str', 'ordqz')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 13, 10), list_25838, str_25840)

# Assigning a type to the variable '__all__' (line 13)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 0), '__all__', list_25838)

# Assigning a List to a Name (line 15):

# Assigning a List to a Name (line 15):

# Obtaining an instance of the builtin type 'list' (line 15)
list_25841 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 20), 'list')
# Adding type elements to the builtin type 'list' instance (line 15)
# Adding element type (line 15)
str_25842 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 21), 'str', 'i')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 15, 20), list_25841, str_25842)
# Adding element type (line 15)
str_25843 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 26), 'str', 'l')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 15, 20), list_25841, str_25843)
# Adding element type (line 15)
str_25844 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 31), 'str', 'd')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 15, 20), list_25841, str_25844)

# Assigning a type to the variable '_double_precision' (line 15)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 0), '_double_precision', list_25841)

@norecursion
def _select_function(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_select_function'
    module_type_store = module_type_store.open_function_context('_select_function', 18, 0, False)
    
    # Passed parameters checking function
    _select_function.stypy_localization = localization
    _select_function.stypy_type_of_self = None
    _select_function.stypy_type_store = module_type_store
    _select_function.stypy_function_name = '_select_function'
    _select_function.stypy_param_names_list = ['sort']
    _select_function.stypy_varargs_param_name = None
    _select_function.stypy_kwargs_param_name = None
    _select_function.stypy_call_defaults = defaults
    _select_function.stypy_call_varargs = varargs
    _select_function.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_select_function', ['sort'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_select_function', localization, ['sort'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_select_function(...)' code ##################

    
    
    # Call to callable(...): (line 19)
    # Processing the call arguments (line 19)
    # Getting the type of 'sort' (line 19)
    sort_25846 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 16), 'sort', False)
    # Processing the call keyword arguments (line 19)
    kwargs_25847 = {}
    # Getting the type of 'callable' (line 19)
    callable_25845 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 7), 'callable', False)
    # Calling callable(args, kwargs) (line 19)
    callable_call_result_25848 = invoke(stypy.reporting.localization.Localization(__file__, 19, 7), callable_25845, *[sort_25846], **kwargs_25847)
    
    # Testing the type of an if condition (line 19)
    if_condition_25849 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 19, 4), callable_call_result_25848)
    # Assigning a type to the variable 'if_condition_25849' (line 19)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 19, 4), 'if_condition_25849', if_condition_25849)
    # SSA begins for if statement (line 19)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Name to a Name (line 21):
    
    # Assigning a Name to a Name (line 21):
    # Getting the type of 'sort' (line 21)
    sort_25850 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 20), 'sort')
    # Assigning a type to the variable 'sfunction' (line 21)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 21, 8), 'sfunction', sort_25850)
    # SSA branch for the else part of an if statement (line 19)
    module_type_store.open_ssa_branch('else')
    
    
    # Getting the type of 'sort' (line 22)
    sort_25851 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 9), 'sort')
    str_25852 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 17), 'str', 'lhp')
    # Applying the binary operator '==' (line 22)
    result_eq_25853 = python_operator(stypy.reporting.localization.Localization(__file__, 22, 9), '==', sort_25851, str_25852)
    
    # Testing the type of an if condition (line 22)
    if_condition_25854 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 22, 9), result_eq_25853)
    # Assigning a type to the variable 'if_condition_25854' (line 22)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 22, 9), 'if_condition_25854', if_condition_25854)
    # SSA begins for if statement (line 22)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Name to a Name (line 23):
    
    # Assigning a Name to a Name (line 23):
    # Getting the type of '_lhp' (line 23)
    _lhp_25855 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 20), '_lhp')
    # Assigning a type to the variable 'sfunction' (line 23)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 23, 8), 'sfunction', _lhp_25855)
    # SSA branch for the else part of an if statement (line 22)
    module_type_store.open_ssa_branch('else')
    
    
    # Getting the type of 'sort' (line 24)
    sort_25856 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 9), 'sort')
    str_25857 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 17), 'str', 'rhp')
    # Applying the binary operator '==' (line 24)
    result_eq_25858 = python_operator(stypy.reporting.localization.Localization(__file__, 24, 9), '==', sort_25856, str_25857)
    
    # Testing the type of an if condition (line 24)
    if_condition_25859 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 24, 9), result_eq_25858)
    # Assigning a type to the variable 'if_condition_25859' (line 24)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 24, 9), 'if_condition_25859', if_condition_25859)
    # SSA begins for if statement (line 24)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Name to a Name (line 25):
    
    # Assigning a Name to a Name (line 25):
    # Getting the type of '_rhp' (line 25)
    _rhp_25860 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 20), '_rhp')
    # Assigning a type to the variable 'sfunction' (line 25)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 25, 8), 'sfunction', _rhp_25860)
    # SSA branch for the else part of an if statement (line 24)
    module_type_store.open_ssa_branch('else')
    
    
    # Getting the type of 'sort' (line 26)
    sort_25861 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 9), 'sort')
    str_25862 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 17), 'str', 'iuc')
    # Applying the binary operator '==' (line 26)
    result_eq_25863 = python_operator(stypy.reporting.localization.Localization(__file__, 26, 9), '==', sort_25861, str_25862)
    
    # Testing the type of an if condition (line 26)
    if_condition_25864 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 26, 9), result_eq_25863)
    # Assigning a type to the variable 'if_condition_25864' (line 26)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 26, 9), 'if_condition_25864', if_condition_25864)
    # SSA begins for if statement (line 26)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Name to a Name (line 27):
    
    # Assigning a Name to a Name (line 27):
    # Getting the type of '_iuc' (line 27)
    _iuc_25865 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 20), '_iuc')
    # Assigning a type to the variable 'sfunction' (line 27)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 27, 8), 'sfunction', _iuc_25865)
    # SSA branch for the else part of an if statement (line 26)
    module_type_store.open_ssa_branch('else')
    
    
    # Getting the type of 'sort' (line 28)
    sort_25866 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 9), 'sort')
    str_25867 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 17), 'str', 'ouc')
    # Applying the binary operator '==' (line 28)
    result_eq_25868 = python_operator(stypy.reporting.localization.Localization(__file__, 28, 9), '==', sort_25866, str_25867)
    
    # Testing the type of an if condition (line 28)
    if_condition_25869 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 28, 9), result_eq_25868)
    # Assigning a type to the variable 'if_condition_25869' (line 28)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 28, 9), 'if_condition_25869', if_condition_25869)
    # SSA begins for if statement (line 28)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Name to a Name (line 29):
    
    # Assigning a Name to a Name (line 29):
    # Getting the type of '_ouc' (line 29)
    _ouc_25870 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 20), '_ouc')
    # Assigning a type to the variable 'sfunction' (line 29)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 29, 8), 'sfunction', _ouc_25870)
    # SSA branch for the else part of an if statement (line 28)
    module_type_store.open_ssa_branch('else')
    
    # Call to ValueError(...): (line 31)
    # Processing the call arguments (line 31)
    str_25872 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 25), 'str', "sort parameter must be None, a callable, or one of ('lhp','rhp','iuc','ouc')")
    # Processing the call keyword arguments (line 31)
    kwargs_25873 = {}
    # Getting the type of 'ValueError' (line 31)
    ValueError_25871 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 31)
    ValueError_call_result_25874 = invoke(stypy.reporting.localization.Localization(__file__, 31, 14), ValueError_25871, *[str_25872], **kwargs_25873)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 31, 8), ValueError_call_result_25874, 'raise parameter', BaseException)
    # SSA join for if statement (line 28)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 26)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 24)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 22)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 19)
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'sfunction' (line 34)
    sfunction_25875 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 11), 'sfunction')
    # Assigning a type to the variable 'stypy_return_type' (line 34)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 34, 4), 'stypy_return_type', sfunction_25875)
    
    # ################# End of '_select_function(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_select_function' in the type store
    # Getting the type of 'stypy_return_type' (line 18)
    stypy_return_type_25876 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_25876)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_select_function'
    return stypy_return_type_25876

# Assigning a type to the variable '_select_function' (line 18)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 18, 0), '_select_function', _select_function)

@norecursion
def _lhp(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_lhp'
    module_type_store = module_type_store.open_function_context('_lhp', 37, 0, False)
    
    # Passed parameters checking function
    _lhp.stypy_localization = localization
    _lhp.stypy_type_of_self = None
    _lhp.stypy_type_store = module_type_store
    _lhp.stypy_function_name = '_lhp'
    _lhp.stypy_param_names_list = ['x', 'y']
    _lhp.stypy_varargs_param_name = None
    _lhp.stypy_kwargs_param_name = None
    _lhp.stypy_call_defaults = defaults
    _lhp.stypy_call_varargs = varargs
    _lhp.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_lhp', ['x', 'y'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_lhp', localization, ['x', 'y'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_lhp(...)' code ##################

    
    # Assigning a Call to a Name (line 38):
    
    # Assigning a Call to a Name (line 38):
    
    # Call to empty_like(...): (line 38)
    # Processing the call arguments (line 38)
    # Getting the type of 'x' (line 38)
    x_25879 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 24), 'x', False)
    # Processing the call keyword arguments (line 38)
    # Getting the type of 'bool' (line 38)
    bool_25880 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 33), 'bool', False)
    keyword_25881 = bool_25880
    kwargs_25882 = {'dtype': keyword_25881}
    # Getting the type of 'np' (line 38)
    np_25877 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 10), 'np', False)
    # Obtaining the member 'empty_like' of a type (line 38)
    empty_like_25878 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 38, 10), np_25877, 'empty_like')
    # Calling empty_like(args, kwargs) (line 38)
    empty_like_call_result_25883 = invoke(stypy.reporting.localization.Localization(__file__, 38, 10), empty_like_25878, *[x_25879], **kwargs_25882)
    
    # Assigning a type to the variable 'out' (line 38)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 38, 4), 'out', empty_like_call_result_25883)
    
    # Assigning a Compare to a Name (line 39):
    
    # Assigning a Compare to a Name (line 39):
    
    # Getting the type of 'y' (line 39)
    y_25884 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 15), 'y')
    int_25885 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 20), 'int')
    # Applying the binary operator '!=' (line 39)
    result_ne_25886 = python_operator(stypy.reporting.localization.Localization(__file__, 39, 15), '!=', y_25884, int_25885)
    
    # Assigning a type to the variable 'nonzero' (line 39)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 39, 4), 'nonzero', result_ne_25886)
    
    # Assigning a Name to a Subscript (line 41):
    
    # Assigning a Name to a Subscript (line 41):
    # Getting the type of 'False' (line 41)
    False_25887 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 20), 'False')
    # Getting the type of 'out' (line 41)
    out_25888 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 4), 'out')
    
    # Getting the type of 'nonzero' (line 41)
    nonzero_25889 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 9), 'nonzero')
    # Applying the '~' unary operator (line 41)
    result_inv_25890 = python_operator(stypy.reporting.localization.Localization(__file__, 41, 8), '~', nonzero_25889)
    
    # Storing an element on a container (line 41)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 41, 4), out_25888, (result_inv_25890, False_25887))
    
    # Assigning a Compare to a Subscript (line 42):
    
    # Assigning a Compare to a Subscript (line 42):
    
    
    # Call to real(...): (line 42)
    # Processing the call arguments (line 42)
    
    # Obtaining the type of the subscript
    # Getting the type of 'nonzero' (line 42)
    nonzero_25893 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 30), 'nonzero', False)
    # Getting the type of 'x' (line 42)
    x_25894 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 28), 'x', False)
    # Obtaining the member '__getitem__' of a type (line 42)
    getitem___25895 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 42, 28), x_25894, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 42)
    subscript_call_result_25896 = invoke(stypy.reporting.localization.Localization(__file__, 42, 28), getitem___25895, nonzero_25893)
    
    
    # Obtaining the type of the subscript
    # Getting the type of 'nonzero' (line 42)
    nonzero_25897 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 41), 'nonzero', False)
    # Getting the type of 'y' (line 42)
    y_25898 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 39), 'y', False)
    # Obtaining the member '__getitem__' of a type (line 42)
    getitem___25899 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 42, 39), y_25898, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 42)
    subscript_call_result_25900 = invoke(stypy.reporting.localization.Localization(__file__, 42, 39), getitem___25899, nonzero_25897)
    
    # Applying the binary operator 'div' (line 42)
    result_div_25901 = python_operator(stypy.reporting.localization.Localization(__file__, 42, 28), 'div', subscript_call_result_25896, subscript_call_result_25900)
    
    # Processing the call keyword arguments (line 42)
    kwargs_25902 = {}
    # Getting the type of 'np' (line 42)
    np_25891 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 20), 'np', False)
    # Obtaining the member 'real' of a type (line 42)
    real_25892 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 42, 20), np_25891, 'real')
    # Calling real(args, kwargs) (line 42)
    real_call_result_25903 = invoke(stypy.reporting.localization.Localization(__file__, 42, 20), real_25892, *[result_div_25901], **kwargs_25902)
    
    float_25904 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 53), 'float')
    # Applying the binary operator '<' (line 42)
    result_lt_25905 = python_operator(stypy.reporting.localization.Localization(__file__, 42, 20), '<', real_call_result_25903, float_25904)
    
    # Getting the type of 'out' (line 42)
    out_25906 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 4), 'out')
    # Getting the type of 'nonzero' (line 42)
    nonzero_25907 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 8), 'nonzero')
    # Storing an element on a container (line 42)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 42, 4), out_25906, (nonzero_25907, result_lt_25905))
    # Getting the type of 'out' (line 43)
    out_25908 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 11), 'out')
    # Assigning a type to the variable 'stypy_return_type' (line 43)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 43, 4), 'stypy_return_type', out_25908)
    
    # ################# End of '_lhp(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_lhp' in the type store
    # Getting the type of 'stypy_return_type' (line 37)
    stypy_return_type_25909 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_25909)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_lhp'
    return stypy_return_type_25909

# Assigning a type to the variable '_lhp' (line 37)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 37, 0), '_lhp', _lhp)

@norecursion
def _rhp(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_rhp'
    module_type_store = module_type_store.open_function_context('_rhp', 46, 0, False)
    
    # Passed parameters checking function
    _rhp.stypy_localization = localization
    _rhp.stypy_type_of_self = None
    _rhp.stypy_type_store = module_type_store
    _rhp.stypy_function_name = '_rhp'
    _rhp.stypy_param_names_list = ['x', 'y']
    _rhp.stypy_varargs_param_name = None
    _rhp.stypy_kwargs_param_name = None
    _rhp.stypy_call_defaults = defaults
    _rhp.stypy_call_varargs = varargs
    _rhp.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_rhp', ['x', 'y'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_rhp', localization, ['x', 'y'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_rhp(...)' code ##################

    
    # Assigning a Call to a Name (line 47):
    
    # Assigning a Call to a Name (line 47):
    
    # Call to empty_like(...): (line 47)
    # Processing the call arguments (line 47)
    # Getting the type of 'x' (line 47)
    x_25912 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 24), 'x', False)
    # Processing the call keyword arguments (line 47)
    # Getting the type of 'bool' (line 47)
    bool_25913 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 33), 'bool', False)
    keyword_25914 = bool_25913
    kwargs_25915 = {'dtype': keyword_25914}
    # Getting the type of 'np' (line 47)
    np_25910 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 10), 'np', False)
    # Obtaining the member 'empty_like' of a type (line 47)
    empty_like_25911 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 47, 10), np_25910, 'empty_like')
    # Calling empty_like(args, kwargs) (line 47)
    empty_like_call_result_25916 = invoke(stypy.reporting.localization.Localization(__file__, 47, 10), empty_like_25911, *[x_25912], **kwargs_25915)
    
    # Assigning a type to the variable 'out' (line 47)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 47, 4), 'out', empty_like_call_result_25916)
    
    # Assigning a Compare to a Name (line 48):
    
    # Assigning a Compare to a Name (line 48):
    
    # Getting the type of 'y' (line 48)
    y_25917 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 15), 'y')
    int_25918 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 48, 20), 'int')
    # Applying the binary operator '!=' (line 48)
    result_ne_25919 = python_operator(stypy.reporting.localization.Localization(__file__, 48, 15), '!=', y_25917, int_25918)
    
    # Assigning a type to the variable 'nonzero' (line 48)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 48, 4), 'nonzero', result_ne_25919)
    
    # Assigning a Name to a Subscript (line 50):
    
    # Assigning a Name to a Subscript (line 50):
    # Getting the type of 'False' (line 50)
    False_25920 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 20), 'False')
    # Getting the type of 'out' (line 50)
    out_25921 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 4), 'out')
    
    # Getting the type of 'nonzero' (line 50)
    nonzero_25922 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 9), 'nonzero')
    # Applying the '~' unary operator (line 50)
    result_inv_25923 = python_operator(stypy.reporting.localization.Localization(__file__, 50, 8), '~', nonzero_25922)
    
    # Storing an element on a container (line 50)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 50, 4), out_25921, (result_inv_25923, False_25920))
    
    # Assigning a Compare to a Subscript (line 51):
    
    # Assigning a Compare to a Subscript (line 51):
    
    
    # Call to real(...): (line 51)
    # Processing the call arguments (line 51)
    
    # Obtaining the type of the subscript
    # Getting the type of 'nonzero' (line 51)
    nonzero_25926 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 30), 'nonzero', False)
    # Getting the type of 'x' (line 51)
    x_25927 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 28), 'x', False)
    # Obtaining the member '__getitem__' of a type (line 51)
    getitem___25928 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 51, 28), x_25927, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 51)
    subscript_call_result_25929 = invoke(stypy.reporting.localization.Localization(__file__, 51, 28), getitem___25928, nonzero_25926)
    
    
    # Obtaining the type of the subscript
    # Getting the type of 'nonzero' (line 51)
    nonzero_25930 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 41), 'nonzero', False)
    # Getting the type of 'y' (line 51)
    y_25931 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 39), 'y', False)
    # Obtaining the member '__getitem__' of a type (line 51)
    getitem___25932 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 51, 39), y_25931, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 51)
    subscript_call_result_25933 = invoke(stypy.reporting.localization.Localization(__file__, 51, 39), getitem___25932, nonzero_25930)
    
    # Applying the binary operator 'div' (line 51)
    result_div_25934 = python_operator(stypy.reporting.localization.Localization(__file__, 51, 28), 'div', subscript_call_result_25929, subscript_call_result_25933)
    
    # Processing the call keyword arguments (line 51)
    kwargs_25935 = {}
    # Getting the type of 'np' (line 51)
    np_25924 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 20), 'np', False)
    # Obtaining the member 'real' of a type (line 51)
    real_25925 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 51, 20), np_25924, 'real')
    # Calling real(args, kwargs) (line 51)
    real_call_result_25936 = invoke(stypy.reporting.localization.Localization(__file__, 51, 20), real_25925, *[result_div_25934], **kwargs_25935)
    
    float_25937 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 51, 53), 'float')
    # Applying the binary operator '>' (line 51)
    result_gt_25938 = python_operator(stypy.reporting.localization.Localization(__file__, 51, 20), '>', real_call_result_25936, float_25937)
    
    # Getting the type of 'out' (line 51)
    out_25939 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 4), 'out')
    # Getting the type of 'nonzero' (line 51)
    nonzero_25940 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 8), 'nonzero')
    # Storing an element on a container (line 51)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 51, 4), out_25939, (nonzero_25940, result_gt_25938))
    # Getting the type of 'out' (line 52)
    out_25941 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 11), 'out')
    # Assigning a type to the variable 'stypy_return_type' (line 52)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 52, 4), 'stypy_return_type', out_25941)
    
    # ################# End of '_rhp(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_rhp' in the type store
    # Getting the type of 'stypy_return_type' (line 46)
    stypy_return_type_25942 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_25942)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_rhp'
    return stypy_return_type_25942

# Assigning a type to the variable '_rhp' (line 46)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 46, 0), '_rhp', _rhp)

@norecursion
def _iuc(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_iuc'
    module_type_store = module_type_store.open_function_context('_iuc', 55, 0, False)
    
    # Passed parameters checking function
    _iuc.stypy_localization = localization
    _iuc.stypy_type_of_self = None
    _iuc.stypy_type_store = module_type_store
    _iuc.stypy_function_name = '_iuc'
    _iuc.stypy_param_names_list = ['x', 'y']
    _iuc.stypy_varargs_param_name = None
    _iuc.stypy_kwargs_param_name = None
    _iuc.stypy_call_defaults = defaults
    _iuc.stypy_call_varargs = varargs
    _iuc.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_iuc', ['x', 'y'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_iuc', localization, ['x', 'y'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_iuc(...)' code ##################

    
    # Assigning a Call to a Name (line 56):
    
    # Assigning a Call to a Name (line 56):
    
    # Call to empty_like(...): (line 56)
    # Processing the call arguments (line 56)
    # Getting the type of 'x' (line 56)
    x_25945 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 24), 'x', False)
    # Processing the call keyword arguments (line 56)
    # Getting the type of 'bool' (line 56)
    bool_25946 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 33), 'bool', False)
    keyword_25947 = bool_25946
    kwargs_25948 = {'dtype': keyword_25947}
    # Getting the type of 'np' (line 56)
    np_25943 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 10), 'np', False)
    # Obtaining the member 'empty_like' of a type (line 56)
    empty_like_25944 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 56, 10), np_25943, 'empty_like')
    # Calling empty_like(args, kwargs) (line 56)
    empty_like_call_result_25949 = invoke(stypy.reporting.localization.Localization(__file__, 56, 10), empty_like_25944, *[x_25945], **kwargs_25948)
    
    # Assigning a type to the variable 'out' (line 56)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 56, 4), 'out', empty_like_call_result_25949)
    
    # Assigning a Compare to a Name (line 57):
    
    # Assigning a Compare to a Name (line 57):
    
    # Getting the type of 'y' (line 57)
    y_25950 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 15), 'y')
    int_25951 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 57, 20), 'int')
    # Applying the binary operator '!=' (line 57)
    result_ne_25952 = python_operator(stypy.reporting.localization.Localization(__file__, 57, 15), '!=', y_25950, int_25951)
    
    # Assigning a type to the variable 'nonzero' (line 57)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 57, 4), 'nonzero', result_ne_25952)
    
    # Assigning a Name to a Subscript (line 59):
    
    # Assigning a Name to a Subscript (line 59):
    # Getting the type of 'False' (line 59)
    False_25953 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 20), 'False')
    # Getting the type of 'out' (line 59)
    out_25954 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 4), 'out')
    
    # Getting the type of 'nonzero' (line 59)
    nonzero_25955 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 9), 'nonzero')
    # Applying the '~' unary operator (line 59)
    result_inv_25956 = python_operator(stypy.reporting.localization.Localization(__file__, 59, 8), '~', nonzero_25955)
    
    # Storing an element on a container (line 59)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 59, 4), out_25954, (result_inv_25956, False_25953))
    
    # Assigning a Compare to a Subscript (line 60):
    
    # Assigning a Compare to a Subscript (line 60):
    
    
    # Call to abs(...): (line 60)
    # Processing the call arguments (line 60)
    
    # Obtaining the type of the subscript
    # Getting the type of 'nonzero' (line 60)
    nonzero_25958 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 26), 'nonzero', False)
    # Getting the type of 'x' (line 60)
    x_25959 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 24), 'x', False)
    # Obtaining the member '__getitem__' of a type (line 60)
    getitem___25960 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 60, 24), x_25959, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 60)
    subscript_call_result_25961 = invoke(stypy.reporting.localization.Localization(__file__, 60, 24), getitem___25960, nonzero_25958)
    
    
    # Obtaining the type of the subscript
    # Getting the type of 'nonzero' (line 60)
    nonzero_25962 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 37), 'nonzero', False)
    # Getting the type of 'y' (line 60)
    y_25963 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 35), 'y', False)
    # Obtaining the member '__getitem__' of a type (line 60)
    getitem___25964 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 60, 35), y_25963, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 60)
    subscript_call_result_25965 = invoke(stypy.reporting.localization.Localization(__file__, 60, 35), getitem___25964, nonzero_25962)
    
    # Applying the binary operator 'div' (line 60)
    result_div_25966 = python_operator(stypy.reporting.localization.Localization(__file__, 60, 24), 'div', subscript_call_result_25961, subscript_call_result_25965)
    
    # Processing the call keyword arguments (line 60)
    kwargs_25967 = {}
    # Getting the type of 'abs' (line 60)
    abs_25957 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 20), 'abs', False)
    # Calling abs(args, kwargs) (line 60)
    abs_call_result_25968 = invoke(stypy.reporting.localization.Localization(__file__, 60, 20), abs_25957, *[result_div_25966], **kwargs_25967)
    
    float_25969 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, 49), 'float')
    # Applying the binary operator '<' (line 60)
    result_lt_25970 = python_operator(stypy.reporting.localization.Localization(__file__, 60, 20), '<', abs_call_result_25968, float_25969)
    
    # Getting the type of 'out' (line 60)
    out_25971 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 4), 'out')
    # Getting the type of 'nonzero' (line 60)
    nonzero_25972 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 8), 'nonzero')
    # Storing an element on a container (line 60)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 60, 4), out_25971, (nonzero_25972, result_lt_25970))
    # Getting the type of 'out' (line 61)
    out_25973 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 11), 'out')
    # Assigning a type to the variable 'stypy_return_type' (line 61)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 61, 4), 'stypy_return_type', out_25973)
    
    # ################# End of '_iuc(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_iuc' in the type store
    # Getting the type of 'stypy_return_type' (line 55)
    stypy_return_type_25974 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_25974)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_iuc'
    return stypy_return_type_25974

# Assigning a type to the variable '_iuc' (line 55)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 55, 0), '_iuc', _iuc)

@norecursion
def _ouc(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_ouc'
    module_type_store = module_type_store.open_function_context('_ouc', 64, 0, False)
    
    # Passed parameters checking function
    _ouc.stypy_localization = localization
    _ouc.stypy_type_of_self = None
    _ouc.stypy_type_store = module_type_store
    _ouc.stypy_function_name = '_ouc'
    _ouc.stypy_param_names_list = ['x', 'y']
    _ouc.stypy_varargs_param_name = None
    _ouc.stypy_kwargs_param_name = None
    _ouc.stypy_call_defaults = defaults
    _ouc.stypy_call_varargs = varargs
    _ouc.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_ouc', ['x', 'y'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_ouc', localization, ['x', 'y'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_ouc(...)' code ##################

    
    # Assigning a Call to a Name (line 65):
    
    # Assigning a Call to a Name (line 65):
    
    # Call to empty_like(...): (line 65)
    # Processing the call arguments (line 65)
    # Getting the type of 'x' (line 65)
    x_25977 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 24), 'x', False)
    # Processing the call keyword arguments (line 65)
    # Getting the type of 'bool' (line 65)
    bool_25978 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 33), 'bool', False)
    keyword_25979 = bool_25978
    kwargs_25980 = {'dtype': keyword_25979}
    # Getting the type of 'np' (line 65)
    np_25975 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 10), 'np', False)
    # Obtaining the member 'empty_like' of a type (line 65)
    empty_like_25976 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 65, 10), np_25975, 'empty_like')
    # Calling empty_like(args, kwargs) (line 65)
    empty_like_call_result_25981 = invoke(stypy.reporting.localization.Localization(__file__, 65, 10), empty_like_25976, *[x_25977], **kwargs_25980)
    
    # Assigning a type to the variable 'out' (line 65)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 65, 4), 'out', empty_like_call_result_25981)
    
    # Assigning a Compare to a Name (line 66):
    
    # Assigning a Compare to a Name (line 66):
    
    # Getting the type of 'x' (line 66)
    x_25982 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 13), 'x')
    int_25983 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 66, 18), 'int')
    # Applying the binary operator '==' (line 66)
    result_eq_25984 = python_operator(stypy.reporting.localization.Localization(__file__, 66, 13), '==', x_25982, int_25983)
    
    # Assigning a type to the variable 'xzero' (line 66)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 66, 4), 'xzero', result_eq_25984)
    
    # Assigning a Compare to a Name (line 67):
    
    # Assigning a Compare to a Name (line 67):
    
    # Getting the type of 'y' (line 67)
    y_25985 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 13), 'y')
    int_25986 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 67, 18), 'int')
    # Applying the binary operator '==' (line 67)
    result_eq_25987 = python_operator(stypy.reporting.localization.Localization(__file__, 67, 13), '==', y_25985, int_25986)
    
    # Assigning a type to the variable 'yzero' (line 67)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 67, 4), 'yzero', result_eq_25987)
    
    # Assigning a Name to a Subscript (line 68):
    
    # Assigning a Name to a Subscript (line 68):
    # Getting the type of 'False' (line 68)
    False_25988 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 25), 'False')
    # Getting the type of 'out' (line 68)
    out_25989 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 4), 'out')
    # Getting the type of 'xzero' (line 68)
    xzero_25990 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 8), 'xzero')
    # Getting the type of 'yzero' (line 68)
    yzero_25991 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 16), 'yzero')
    # Applying the binary operator '&' (line 68)
    result_and__25992 = python_operator(stypy.reporting.localization.Localization(__file__, 68, 8), '&', xzero_25990, yzero_25991)
    
    # Storing an element on a container (line 68)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 68, 4), out_25989, (result_and__25992, False_25988))
    
    # Assigning a Name to a Subscript (line 69):
    
    # Assigning a Name to a Subscript (line 69):
    # Getting the type of 'True' (line 69)
    True_25993 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 26), 'True')
    # Getting the type of 'out' (line 69)
    out_25994 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 4), 'out')
    
    # Getting the type of 'xzero' (line 69)
    xzero_25995 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 9), 'xzero')
    # Applying the '~' unary operator (line 69)
    result_inv_25996 = python_operator(stypy.reporting.localization.Localization(__file__, 69, 8), '~', xzero_25995)
    
    # Getting the type of 'yzero' (line 69)
    yzero_25997 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 17), 'yzero')
    # Applying the binary operator '&' (line 69)
    result_and__25998 = python_operator(stypy.reporting.localization.Localization(__file__, 69, 8), '&', result_inv_25996, yzero_25997)
    
    # Storing an element on a container (line 69)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 69, 4), out_25994, (result_and__25998, True_25993))
    
    # Assigning a Compare to a Subscript (line 70):
    
    # Assigning a Compare to a Subscript (line 70):
    
    
    # Call to abs(...): (line 70)
    # Processing the call arguments (line 70)
    
    # Obtaining the type of the subscript
    
    # Getting the type of 'yzero' (line 70)
    yzero_26000 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 26), 'yzero', False)
    # Applying the '~' unary operator (line 70)
    result_inv_26001 = python_operator(stypy.reporting.localization.Localization(__file__, 70, 25), '~', yzero_26000)
    
    # Getting the type of 'x' (line 70)
    x_26002 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 23), 'x', False)
    # Obtaining the member '__getitem__' of a type (line 70)
    getitem___26003 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 70, 23), x_26002, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 70)
    subscript_call_result_26004 = invoke(stypy.reporting.localization.Localization(__file__, 70, 23), getitem___26003, result_inv_26001)
    
    
    # Obtaining the type of the subscript
    
    # Getting the type of 'yzero' (line 70)
    yzero_26005 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 36), 'yzero', False)
    # Applying the '~' unary operator (line 70)
    result_inv_26006 = python_operator(stypy.reporting.localization.Localization(__file__, 70, 35), '~', yzero_26005)
    
    # Getting the type of 'y' (line 70)
    y_26007 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 33), 'y', False)
    # Obtaining the member '__getitem__' of a type (line 70)
    getitem___26008 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 70, 33), y_26007, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 70)
    subscript_call_result_26009 = invoke(stypy.reporting.localization.Localization(__file__, 70, 33), getitem___26008, result_inv_26006)
    
    # Applying the binary operator 'div' (line 70)
    result_div_26010 = python_operator(stypy.reporting.localization.Localization(__file__, 70, 23), 'div', subscript_call_result_26004, subscript_call_result_26009)
    
    # Processing the call keyword arguments (line 70)
    kwargs_26011 = {}
    # Getting the type of 'abs' (line 70)
    abs_25999 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 19), 'abs', False)
    # Calling abs(args, kwargs) (line 70)
    abs_call_result_26012 = invoke(stypy.reporting.localization.Localization(__file__, 70, 19), abs_25999, *[result_div_26010], **kwargs_26011)
    
    float_26013 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 70, 46), 'float')
    # Applying the binary operator '>' (line 70)
    result_gt_26014 = python_operator(stypy.reporting.localization.Localization(__file__, 70, 19), '>', abs_call_result_26012, float_26013)
    
    # Getting the type of 'out' (line 70)
    out_26015 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 4), 'out')
    
    # Getting the type of 'yzero' (line 70)
    yzero_26016 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 9), 'yzero')
    # Applying the '~' unary operator (line 70)
    result_inv_26017 = python_operator(stypy.reporting.localization.Localization(__file__, 70, 8), '~', yzero_26016)
    
    # Storing an element on a container (line 70)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 70, 4), out_26015, (result_inv_26017, result_gt_26014))
    # Getting the type of 'out' (line 71)
    out_26018 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 11), 'out')
    # Assigning a type to the variable 'stypy_return_type' (line 71)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 71, 4), 'stypy_return_type', out_26018)
    
    # ################# End of '_ouc(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_ouc' in the type store
    # Getting the type of 'stypy_return_type' (line 64)
    stypy_return_type_26019 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_26019)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_ouc'
    return stypy_return_type_26019

# Assigning a type to the variable '_ouc' (line 64)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 64, 0), '_ouc', _ouc)

@norecursion
def _qz(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    str_26020 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 74, 21), 'str', 'real')
    # Getting the type of 'None' (line 74)
    None_26021 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 35), 'None')
    # Getting the type of 'None' (line 74)
    None_26022 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 46), 'None')
    # Getting the type of 'False' (line 74)
    False_26023 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 64), 'False')
    # Getting the type of 'False' (line 75)
    False_26024 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 20), 'False')
    # Getting the type of 'True' (line 75)
    True_26025 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 40), 'True')
    defaults = [str_26020, None_26021, None_26022, False_26023, False_26024, True_26025]
    # Create a new context for function '_qz'
    module_type_store = module_type_store.open_function_context('_qz', 74, 0, False)
    
    # Passed parameters checking function
    _qz.stypy_localization = localization
    _qz.stypy_type_of_self = None
    _qz.stypy_type_store = module_type_store
    _qz.stypy_function_name = '_qz'
    _qz.stypy_param_names_list = ['A', 'B', 'output', 'lwork', 'sort', 'overwrite_a', 'overwrite_b', 'check_finite']
    _qz.stypy_varargs_param_name = None
    _qz.stypy_kwargs_param_name = None
    _qz.stypy_call_defaults = defaults
    _qz.stypy_call_varargs = varargs
    _qz.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_qz', ['A', 'B', 'output', 'lwork', 'sort', 'overwrite_a', 'overwrite_b', 'check_finite'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_qz', localization, ['A', 'B', 'output', 'lwork', 'sort', 'overwrite_a', 'overwrite_b', 'check_finite'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_qz(...)' code ##################

    
    # Type idiom detected: calculating its left and rigth part (line 76)
    # Getting the type of 'sort' (line 76)
    sort_26026 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 4), 'sort')
    # Getting the type of 'None' (line 76)
    None_26027 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 19), 'None')
    
    (may_be_26028, more_types_in_union_26029) = may_not_be_none(sort_26026, None_26027)

    if may_be_26028:

        if more_types_in_union_26029:
            # Runtime conditional SSA (line 76)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Call to ValueError(...): (line 78)
        # Processing the call arguments (line 78)
        str_26031 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 78, 25), 'str', "The 'sort' input of qz() has to be None and will be removed in a future release. Use ordqz instead.")
        # Processing the call keyword arguments (line 78)
        kwargs_26032 = {}
        # Getting the type of 'ValueError' (line 78)
        ValueError_26030 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 14), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 78)
        ValueError_call_result_26033 = invoke(stypy.reporting.localization.Localization(__file__, 78, 14), ValueError_26030, *[str_26031], **kwargs_26032)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 78, 8), ValueError_call_result_26033, 'raise parameter', BaseException)

        if more_types_in_union_26029:
            # SSA join for if statement (line 76)
            module_type_store = module_type_store.join_ssa_context()


    
    
    
    # Getting the type of 'output' (line 81)
    output_26034 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 7), 'output')
    
    # Obtaining an instance of the builtin type 'list' (line 81)
    list_26035 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 81, 21), 'list')
    # Adding type elements to the builtin type 'list' instance (line 81)
    # Adding element type (line 81)
    str_26036 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 81, 22), 'str', 'real')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 81, 21), list_26035, str_26036)
    # Adding element type (line 81)
    str_26037 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 81, 30), 'str', 'complex')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 81, 21), list_26035, str_26037)
    # Adding element type (line 81)
    str_26038 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 81, 41), 'str', 'r')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 81, 21), list_26035, str_26038)
    # Adding element type (line 81)
    str_26039 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 81, 46), 'str', 'c')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 81, 21), list_26035, str_26039)
    
    # Applying the binary operator 'notin' (line 81)
    result_contains_26040 = python_operator(stypy.reporting.localization.Localization(__file__, 81, 7), 'notin', output_26034, list_26035)
    
    # Testing the type of an if condition (line 81)
    if_condition_26041 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 81, 4), result_contains_26040)
    # Assigning a type to the variable 'if_condition_26041' (line 81)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 81, 4), 'if_condition_26041', if_condition_26041)
    # SSA begins for if statement (line 81)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 82)
    # Processing the call arguments (line 82)
    str_26043 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 82, 25), 'str', "argument must be 'real', or 'complex'")
    # Processing the call keyword arguments (line 82)
    kwargs_26044 = {}
    # Getting the type of 'ValueError' (line 82)
    ValueError_26042 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 82)
    ValueError_call_result_26045 = invoke(stypy.reporting.localization.Localization(__file__, 82, 14), ValueError_26042, *[str_26043], **kwargs_26044)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 82, 8), ValueError_call_result_26045, 'raise parameter', BaseException)
    # SSA join for if statement (line 81)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Getting the type of 'check_finite' (line 84)
    check_finite_26046 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 7), 'check_finite')
    # Testing the type of an if condition (line 84)
    if_condition_26047 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 84, 4), check_finite_26046)
    # Assigning a type to the variable 'if_condition_26047' (line 84)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 84, 4), 'if_condition_26047', if_condition_26047)
    # SSA begins for if statement (line 84)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 85):
    
    # Assigning a Call to a Name (line 85):
    
    # Call to asarray_chkfinite(...): (line 85)
    # Processing the call arguments (line 85)
    # Getting the type of 'A' (line 85)
    A_26049 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 31), 'A', False)
    # Processing the call keyword arguments (line 85)
    kwargs_26050 = {}
    # Getting the type of 'asarray_chkfinite' (line 85)
    asarray_chkfinite_26048 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 13), 'asarray_chkfinite', False)
    # Calling asarray_chkfinite(args, kwargs) (line 85)
    asarray_chkfinite_call_result_26051 = invoke(stypy.reporting.localization.Localization(__file__, 85, 13), asarray_chkfinite_26048, *[A_26049], **kwargs_26050)
    
    # Assigning a type to the variable 'a1' (line 85)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 85, 8), 'a1', asarray_chkfinite_call_result_26051)
    
    # Assigning a Call to a Name (line 86):
    
    # Assigning a Call to a Name (line 86):
    
    # Call to asarray_chkfinite(...): (line 86)
    # Processing the call arguments (line 86)
    # Getting the type of 'B' (line 86)
    B_26053 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 31), 'B', False)
    # Processing the call keyword arguments (line 86)
    kwargs_26054 = {}
    # Getting the type of 'asarray_chkfinite' (line 86)
    asarray_chkfinite_26052 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 13), 'asarray_chkfinite', False)
    # Calling asarray_chkfinite(args, kwargs) (line 86)
    asarray_chkfinite_call_result_26055 = invoke(stypy.reporting.localization.Localization(__file__, 86, 13), asarray_chkfinite_26052, *[B_26053], **kwargs_26054)
    
    # Assigning a type to the variable 'b1' (line 86)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 86, 8), 'b1', asarray_chkfinite_call_result_26055)
    # SSA branch for the else part of an if statement (line 84)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Call to a Name (line 88):
    
    # Assigning a Call to a Name (line 88):
    
    # Call to asarray(...): (line 88)
    # Processing the call arguments (line 88)
    # Getting the type of 'A' (line 88)
    A_26058 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 24), 'A', False)
    # Processing the call keyword arguments (line 88)
    kwargs_26059 = {}
    # Getting the type of 'np' (line 88)
    np_26056 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 13), 'np', False)
    # Obtaining the member 'asarray' of a type (line 88)
    asarray_26057 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 88, 13), np_26056, 'asarray')
    # Calling asarray(args, kwargs) (line 88)
    asarray_call_result_26060 = invoke(stypy.reporting.localization.Localization(__file__, 88, 13), asarray_26057, *[A_26058], **kwargs_26059)
    
    # Assigning a type to the variable 'a1' (line 88)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 88, 8), 'a1', asarray_call_result_26060)
    
    # Assigning a Call to a Name (line 89):
    
    # Assigning a Call to a Name (line 89):
    
    # Call to asarray(...): (line 89)
    # Processing the call arguments (line 89)
    # Getting the type of 'B' (line 89)
    B_26063 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 24), 'B', False)
    # Processing the call keyword arguments (line 89)
    kwargs_26064 = {}
    # Getting the type of 'np' (line 89)
    np_26061 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 13), 'np', False)
    # Obtaining the member 'asarray' of a type (line 89)
    asarray_26062 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 89, 13), np_26061, 'asarray')
    # Calling asarray(args, kwargs) (line 89)
    asarray_call_result_26065 = invoke(stypy.reporting.localization.Localization(__file__, 89, 13), asarray_26062, *[B_26063], **kwargs_26064)
    
    # Assigning a type to the variable 'b1' (line 89)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 89, 8), 'b1', asarray_call_result_26065)
    # SSA join for if statement (line 84)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Attribute to a Tuple (line 91):
    
    # Assigning a Subscript to a Name (line 91):
    
    # Obtaining the type of the subscript
    int_26066 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 91, 4), 'int')
    # Getting the type of 'a1' (line 91)
    a1_26067 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 15), 'a1')
    # Obtaining the member 'shape' of a type (line 91)
    shape_26068 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 91, 15), a1_26067, 'shape')
    # Obtaining the member '__getitem__' of a type (line 91)
    getitem___26069 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 91, 4), shape_26068, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 91)
    subscript_call_result_26070 = invoke(stypy.reporting.localization.Localization(__file__, 91, 4), getitem___26069, int_26066)
    
    # Assigning a type to the variable 'tuple_var_assignment_25810' (line 91)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 91, 4), 'tuple_var_assignment_25810', subscript_call_result_26070)
    
    # Assigning a Subscript to a Name (line 91):
    
    # Obtaining the type of the subscript
    int_26071 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 91, 4), 'int')
    # Getting the type of 'a1' (line 91)
    a1_26072 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 15), 'a1')
    # Obtaining the member 'shape' of a type (line 91)
    shape_26073 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 91, 15), a1_26072, 'shape')
    # Obtaining the member '__getitem__' of a type (line 91)
    getitem___26074 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 91, 4), shape_26073, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 91)
    subscript_call_result_26075 = invoke(stypy.reporting.localization.Localization(__file__, 91, 4), getitem___26074, int_26071)
    
    # Assigning a type to the variable 'tuple_var_assignment_25811' (line 91)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 91, 4), 'tuple_var_assignment_25811', subscript_call_result_26075)
    
    # Assigning a Name to a Name (line 91):
    # Getting the type of 'tuple_var_assignment_25810' (line 91)
    tuple_var_assignment_25810_26076 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 4), 'tuple_var_assignment_25810')
    # Assigning a type to the variable 'a_m' (line 91)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 91, 4), 'a_m', tuple_var_assignment_25810_26076)
    
    # Assigning a Name to a Name (line 91):
    # Getting the type of 'tuple_var_assignment_25811' (line 91)
    tuple_var_assignment_25811_26077 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 4), 'tuple_var_assignment_25811')
    # Assigning a type to the variable 'a_n' (line 91)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 91, 9), 'a_n', tuple_var_assignment_25811_26077)
    
    # Assigning a Attribute to a Tuple (line 92):
    
    # Assigning a Subscript to a Name (line 92):
    
    # Obtaining the type of the subscript
    int_26078 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 92, 4), 'int')
    # Getting the type of 'b1' (line 92)
    b1_26079 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 15), 'b1')
    # Obtaining the member 'shape' of a type (line 92)
    shape_26080 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 92, 15), b1_26079, 'shape')
    # Obtaining the member '__getitem__' of a type (line 92)
    getitem___26081 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 92, 4), shape_26080, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 92)
    subscript_call_result_26082 = invoke(stypy.reporting.localization.Localization(__file__, 92, 4), getitem___26081, int_26078)
    
    # Assigning a type to the variable 'tuple_var_assignment_25812' (line 92)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 92, 4), 'tuple_var_assignment_25812', subscript_call_result_26082)
    
    # Assigning a Subscript to a Name (line 92):
    
    # Obtaining the type of the subscript
    int_26083 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 92, 4), 'int')
    # Getting the type of 'b1' (line 92)
    b1_26084 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 15), 'b1')
    # Obtaining the member 'shape' of a type (line 92)
    shape_26085 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 92, 15), b1_26084, 'shape')
    # Obtaining the member '__getitem__' of a type (line 92)
    getitem___26086 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 92, 4), shape_26085, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 92)
    subscript_call_result_26087 = invoke(stypy.reporting.localization.Localization(__file__, 92, 4), getitem___26086, int_26083)
    
    # Assigning a type to the variable 'tuple_var_assignment_25813' (line 92)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 92, 4), 'tuple_var_assignment_25813', subscript_call_result_26087)
    
    # Assigning a Name to a Name (line 92):
    # Getting the type of 'tuple_var_assignment_25812' (line 92)
    tuple_var_assignment_25812_26088 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 4), 'tuple_var_assignment_25812')
    # Assigning a type to the variable 'b_m' (line 92)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 92, 4), 'b_m', tuple_var_assignment_25812_26088)
    
    # Assigning a Name to a Name (line 92):
    # Getting the type of 'tuple_var_assignment_25813' (line 92)
    tuple_var_assignment_25813_26089 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 4), 'tuple_var_assignment_25813')
    # Assigning a type to the variable 'b_n' (line 92)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 92, 9), 'b_n', tuple_var_assignment_25813_26089)
    
    
    
    # Getting the type of 'a_m' (line 93)
    a_m_26090 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 12), 'a_m')
    # Getting the type of 'a_n' (line 93)
    a_n_26091 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 19), 'a_n')
    # Applying the binary operator '==' (line 93)
    result_eq_26092 = python_operator(stypy.reporting.localization.Localization(__file__, 93, 12), '==', a_m_26090, a_n_26091)
    # Getting the type of 'b_m' (line 93)
    b_m_26093 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 26), 'b_m')
    # Applying the binary operator '==' (line 93)
    result_eq_26094 = python_operator(stypy.reporting.localization.Localization(__file__, 93, 12), '==', a_n_26091, b_m_26093)
    # Applying the binary operator '&' (line 93)
    result_and__26095 = python_operator(stypy.reporting.localization.Localization(__file__, 93, 12), '&', result_eq_26092, result_eq_26094)
    # Getting the type of 'b_n' (line 93)
    b_n_26096 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 33), 'b_n')
    # Applying the binary operator '==' (line 93)
    result_eq_26097 = python_operator(stypy.reporting.localization.Localization(__file__, 93, 12), '==', b_m_26093, b_n_26096)
    # Applying the binary operator '&' (line 93)
    result_and__26098 = python_operator(stypy.reporting.localization.Localization(__file__, 93, 12), '&', result_and__26095, result_eq_26097)
    
    # Applying the 'not' unary operator (line 93)
    result_not__26099 = python_operator(stypy.reporting.localization.Localization(__file__, 93, 7), 'not', result_and__26098)
    
    # Testing the type of an if condition (line 93)
    if_condition_26100 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 93, 4), result_not__26099)
    # Assigning a type to the variable 'if_condition_26100' (line 93)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 93, 4), 'if_condition_26100', if_condition_26100)
    # SSA begins for if statement (line 93)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 94)
    # Processing the call arguments (line 94)
    str_26102 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 94, 25), 'str', 'Array dimensions must be square and agree')
    # Processing the call keyword arguments (line 94)
    kwargs_26103 = {}
    # Getting the type of 'ValueError' (line 94)
    ValueError_26101 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 94)
    ValueError_call_result_26104 = invoke(stypy.reporting.localization.Localization(__file__, 94, 14), ValueError_26101, *[str_26102], **kwargs_26103)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 94, 8), ValueError_call_result_26104, 'raise parameter', BaseException)
    # SSA join for if statement (line 93)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Attribute to a Name (line 96):
    
    # Assigning a Attribute to a Name (line 96):
    # Getting the type of 'a1' (line 96)
    a1_26105 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 11), 'a1')
    # Obtaining the member 'dtype' of a type (line 96)
    dtype_26106 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 96, 11), a1_26105, 'dtype')
    # Obtaining the member 'char' of a type (line 96)
    char_26107 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 96, 11), dtype_26106, 'char')
    # Assigning a type to the variable 'typa' (line 96)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 96, 4), 'typa', char_26107)
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'output' (line 97)
    output_26108 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 7), 'output')
    
    # Obtaining an instance of the builtin type 'list' (line 97)
    list_26109 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 97, 17), 'list')
    # Adding type elements to the builtin type 'list' instance (line 97)
    # Adding element type (line 97)
    str_26110 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 97, 18), 'str', 'complex')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 97, 17), list_26109, str_26110)
    # Adding element type (line 97)
    str_26111 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 97, 29), 'str', 'c')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 97, 17), list_26109, str_26111)
    
    # Applying the binary operator 'in' (line 97)
    result_contains_26112 = python_operator(stypy.reporting.localization.Localization(__file__, 97, 7), 'in', output_26108, list_26109)
    
    
    # Getting the type of 'typa' (line 97)
    typa_26113 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 38), 'typa')
    
    # Obtaining an instance of the builtin type 'list' (line 97)
    list_26114 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 97, 50), 'list')
    # Adding type elements to the builtin type 'list' instance (line 97)
    # Adding element type (line 97)
    str_26115 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 97, 51), 'str', 'F')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 97, 50), list_26114, str_26115)
    # Adding element type (line 97)
    str_26116 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 97, 56), 'str', 'D')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 97, 50), list_26114, str_26116)
    
    # Applying the binary operator 'notin' (line 97)
    result_contains_26117 = python_operator(stypy.reporting.localization.Localization(__file__, 97, 38), 'notin', typa_26113, list_26114)
    
    # Applying the binary operator 'and' (line 97)
    result_and_keyword_26118 = python_operator(stypy.reporting.localization.Localization(__file__, 97, 7), 'and', result_contains_26112, result_contains_26117)
    
    # Testing the type of an if condition (line 97)
    if_condition_26119 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 97, 4), result_and_keyword_26118)
    # Assigning a type to the variable 'if_condition_26119' (line 97)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 97, 4), 'if_condition_26119', if_condition_26119)
    # SSA begins for if statement (line 97)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    
    # Getting the type of 'typa' (line 98)
    typa_26120 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 11), 'typa')
    # Getting the type of '_double_precision' (line 98)
    _double_precision_26121 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 19), '_double_precision')
    # Applying the binary operator 'in' (line 98)
    result_contains_26122 = python_operator(stypy.reporting.localization.Localization(__file__, 98, 11), 'in', typa_26120, _double_precision_26121)
    
    # Testing the type of an if condition (line 98)
    if_condition_26123 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 98, 8), result_contains_26122)
    # Assigning a type to the variable 'if_condition_26123' (line 98)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 98, 8), 'if_condition_26123', if_condition_26123)
    # SSA begins for if statement (line 98)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 99):
    
    # Assigning a Call to a Name (line 99):
    
    # Call to astype(...): (line 99)
    # Processing the call arguments (line 99)
    str_26126 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 99, 27), 'str', 'D')
    # Processing the call keyword arguments (line 99)
    kwargs_26127 = {}
    # Getting the type of 'a1' (line 99)
    a1_26124 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 17), 'a1', False)
    # Obtaining the member 'astype' of a type (line 99)
    astype_26125 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 99, 17), a1_26124, 'astype')
    # Calling astype(args, kwargs) (line 99)
    astype_call_result_26128 = invoke(stypy.reporting.localization.Localization(__file__, 99, 17), astype_26125, *[str_26126], **kwargs_26127)
    
    # Assigning a type to the variable 'a1' (line 99)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 99, 12), 'a1', astype_call_result_26128)
    
    # Assigning a Str to a Name (line 100):
    
    # Assigning a Str to a Name (line 100):
    str_26129 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 100, 19), 'str', 'D')
    # Assigning a type to the variable 'typa' (line 100)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 100, 12), 'typa', str_26129)
    # SSA branch for the else part of an if statement (line 98)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Call to a Name (line 102):
    
    # Assigning a Call to a Name (line 102):
    
    # Call to astype(...): (line 102)
    # Processing the call arguments (line 102)
    str_26132 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 102, 27), 'str', 'F')
    # Processing the call keyword arguments (line 102)
    kwargs_26133 = {}
    # Getting the type of 'a1' (line 102)
    a1_26130 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 17), 'a1', False)
    # Obtaining the member 'astype' of a type (line 102)
    astype_26131 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 102, 17), a1_26130, 'astype')
    # Calling astype(args, kwargs) (line 102)
    astype_call_result_26134 = invoke(stypy.reporting.localization.Localization(__file__, 102, 17), astype_26131, *[str_26132], **kwargs_26133)
    
    # Assigning a type to the variable 'a1' (line 102)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 102, 12), 'a1', astype_call_result_26134)
    
    # Assigning a Str to a Name (line 103):
    
    # Assigning a Str to a Name (line 103):
    str_26135 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 103, 19), 'str', 'F')
    # Assigning a type to the variable 'typa' (line 103)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 103, 12), 'typa', str_26135)
    # SSA join for if statement (line 98)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 97)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Attribute to a Name (line 104):
    
    # Assigning a Attribute to a Name (line 104):
    # Getting the type of 'b1' (line 104)
    b1_26136 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 11), 'b1')
    # Obtaining the member 'dtype' of a type (line 104)
    dtype_26137 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 104, 11), b1_26136, 'dtype')
    # Obtaining the member 'char' of a type (line 104)
    char_26138 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 104, 11), dtype_26137, 'char')
    # Assigning a type to the variable 'typb' (line 104)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 104, 4), 'typb', char_26138)
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'output' (line 105)
    output_26139 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 7), 'output')
    
    # Obtaining an instance of the builtin type 'list' (line 105)
    list_26140 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 105, 17), 'list')
    # Adding type elements to the builtin type 'list' instance (line 105)
    # Adding element type (line 105)
    str_26141 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 105, 18), 'str', 'complex')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 105, 17), list_26140, str_26141)
    # Adding element type (line 105)
    str_26142 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 105, 29), 'str', 'c')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 105, 17), list_26140, str_26142)
    
    # Applying the binary operator 'in' (line 105)
    result_contains_26143 = python_operator(stypy.reporting.localization.Localization(__file__, 105, 7), 'in', output_26139, list_26140)
    
    
    # Getting the type of 'typb' (line 105)
    typb_26144 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 38), 'typb')
    
    # Obtaining an instance of the builtin type 'list' (line 105)
    list_26145 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 105, 50), 'list')
    # Adding type elements to the builtin type 'list' instance (line 105)
    # Adding element type (line 105)
    str_26146 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 105, 51), 'str', 'F')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 105, 50), list_26145, str_26146)
    # Adding element type (line 105)
    str_26147 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 105, 56), 'str', 'D')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 105, 50), list_26145, str_26147)
    
    # Applying the binary operator 'notin' (line 105)
    result_contains_26148 = python_operator(stypy.reporting.localization.Localization(__file__, 105, 38), 'notin', typb_26144, list_26145)
    
    # Applying the binary operator 'and' (line 105)
    result_and_keyword_26149 = python_operator(stypy.reporting.localization.Localization(__file__, 105, 7), 'and', result_contains_26143, result_contains_26148)
    
    # Testing the type of an if condition (line 105)
    if_condition_26150 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 105, 4), result_and_keyword_26149)
    # Assigning a type to the variable 'if_condition_26150' (line 105)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 105, 4), 'if_condition_26150', if_condition_26150)
    # SSA begins for if statement (line 105)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    
    # Getting the type of 'typb' (line 106)
    typb_26151 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 11), 'typb')
    # Getting the type of '_double_precision' (line 106)
    _double_precision_26152 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 19), '_double_precision')
    # Applying the binary operator 'in' (line 106)
    result_contains_26153 = python_operator(stypy.reporting.localization.Localization(__file__, 106, 11), 'in', typb_26151, _double_precision_26152)
    
    # Testing the type of an if condition (line 106)
    if_condition_26154 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 106, 8), result_contains_26153)
    # Assigning a type to the variable 'if_condition_26154' (line 106)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 106, 8), 'if_condition_26154', if_condition_26154)
    # SSA begins for if statement (line 106)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 107):
    
    # Assigning a Call to a Name (line 107):
    
    # Call to astype(...): (line 107)
    # Processing the call arguments (line 107)
    str_26157 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 107, 27), 'str', 'D')
    # Processing the call keyword arguments (line 107)
    kwargs_26158 = {}
    # Getting the type of 'b1' (line 107)
    b1_26155 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 17), 'b1', False)
    # Obtaining the member 'astype' of a type (line 107)
    astype_26156 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 107, 17), b1_26155, 'astype')
    # Calling astype(args, kwargs) (line 107)
    astype_call_result_26159 = invoke(stypy.reporting.localization.Localization(__file__, 107, 17), astype_26156, *[str_26157], **kwargs_26158)
    
    # Assigning a type to the variable 'b1' (line 107)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 107, 12), 'b1', astype_call_result_26159)
    
    # Assigning a Str to a Name (line 108):
    
    # Assigning a Str to a Name (line 108):
    str_26160 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 108, 19), 'str', 'D')
    # Assigning a type to the variable 'typb' (line 108)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 108, 12), 'typb', str_26160)
    # SSA branch for the else part of an if statement (line 106)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Call to a Name (line 110):
    
    # Assigning a Call to a Name (line 110):
    
    # Call to astype(...): (line 110)
    # Processing the call arguments (line 110)
    str_26163 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 110, 27), 'str', 'F')
    # Processing the call keyword arguments (line 110)
    kwargs_26164 = {}
    # Getting the type of 'b1' (line 110)
    b1_26161 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 17), 'b1', False)
    # Obtaining the member 'astype' of a type (line 110)
    astype_26162 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 110, 17), b1_26161, 'astype')
    # Calling astype(args, kwargs) (line 110)
    astype_call_result_26165 = invoke(stypy.reporting.localization.Localization(__file__, 110, 17), astype_26162, *[str_26163], **kwargs_26164)
    
    # Assigning a type to the variable 'b1' (line 110)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 110, 12), 'b1', astype_call_result_26165)
    
    # Assigning a Str to a Name (line 111):
    
    # Assigning a Str to a Name (line 111):
    str_26166 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 111, 19), 'str', 'F')
    # Assigning a type to the variable 'typb' (line 111)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 111, 12), 'typb', str_26166)
    # SSA join for if statement (line 106)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 105)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a BoolOp to a Name (line 113):
    
    # Assigning a BoolOp to a Name (line 113):
    
    # Evaluating a boolean operation
    # Getting the type of 'overwrite_a' (line 113)
    overwrite_a_26167 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 18), 'overwrite_a')
    
    # Call to _datacopied(...): (line 113)
    # Processing the call arguments (line 113)
    # Getting the type of 'a1' (line 113)
    a1_26169 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 46), 'a1', False)
    # Getting the type of 'A' (line 113)
    A_26170 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 50), 'A', False)
    # Processing the call keyword arguments (line 113)
    kwargs_26171 = {}
    # Getting the type of '_datacopied' (line 113)
    _datacopied_26168 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 34), '_datacopied', False)
    # Calling _datacopied(args, kwargs) (line 113)
    _datacopied_call_result_26172 = invoke(stypy.reporting.localization.Localization(__file__, 113, 34), _datacopied_26168, *[a1_26169, A_26170], **kwargs_26171)
    
    # Applying the binary operator 'or' (line 113)
    result_or_keyword_26173 = python_operator(stypy.reporting.localization.Localization(__file__, 113, 18), 'or', overwrite_a_26167, _datacopied_call_result_26172)
    
    # Assigning a type to the variable 'overwrite_a' (line 113)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 113, 4), 'overwrite_a', result_or_keyword_26173)
    
    # Assigning a BoolOp to a Name (line 114):
    
    # Assigning a BoolOp to a Name (line 114):
    
    # Evaluating a boolean operation
    # Getting the type of 'overwrite_b' (line 114)
    overwrite_b_26174 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 18), 'overwrite_b')
    
    # Call to _datacopied(...): (line 114)
    # Processing the call arguments (line 114)
    # Getting the type of 'b1' (line 114)
    b1_26176 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 46), 'b1', False)
    # Getting the type of 'B' (line 114)
    B_26177 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 50), 'B', False)
    # Processing the call keyword arguments (line 114)
    kwargs_26178 = {}
    # Getting the type of '_datacopied' (line 114)
    _datacopied_26175 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 34), '_datacopied', False)
    # Calling _datacopied(args, kwargs) (line 114)
    _datacopied_call_result_26179 = invoke(stypy.reporting.localization.Localization(__file__, 114, 34), _datacopied_26175, *[b1_26176, B_26177], **kwargs_26178)
    
    # Applying the binary operator 'or' (line 114)
    result_or_keyword_26180 = python_operator(stypy.reporting.localization.Localization(__file__, 114, 18), 'or', overwrite_b_26174, _datacopied_call_result_26179)
    
    # Assigning a type to the variable 'overwrite_b' (line 114)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 114, 4), 'overwrite_b', result_or_keyword_26180)
    
    # Assigning a Call to a Tuple (line 116):
    
    # Assigning a Subscript to a Name (line 116):
    
    # Obtaining the type of the subscript
    int_26181 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 116, 4), 'int')
    
    # Call to get_lapack_funcs(...): (line 116)
    # Processing the call arguments (line 116)
    
    # Obtaining an instance of the builtin type 'tuple' (line 116)
    tuple_26183 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 116, 30), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 116)
    # Adding element type (line 116)
    str_26184 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 116, 30), 'str', 'gges')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 116, 30), tuple_26183, str_26184)
    
    
    # Obtaining an instance of the builtin type 'tuple' (line 116)
    tuple_26185 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 116, 41), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 116)
    # Adding element type (line 116)
    # Getting the type of 'a1' (line 116)
    a1_26186 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 41), 'a1', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 116, 41), tuple_26185, a1_26186)
    # Adding element type (line 116)
    # Getting the type of 'b1' (line 116)
    b1_26187 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 45), 'b1', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 116, 41), tuple_26185, b1_26187)
    
    # Processing the call keyword arguments (line 116)
    kwargs_26188 = {}
    # Getting the type of 'get_lapack_funcs' (line 116)
    get_lapack_funcs_26182 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 12), 'get_lapack_funcs', False)
    # Calling get_lapack_funcs(args, kwargs) (line 116)
    get_lapack_funcs_call_result_26189 = invoke(stypy.reporting.localization.Localization(__file__, 116, 12), get_lapack_funcs_26182, *[tuple_26183, tuple_26185], **kwargs_26188)
    
    # Obtaining the member '__getitem__' of a type (line 116)
    getitem___26190 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 116, 4), get_lapack_funcs_call_result_26189, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 116)
    subscript_call_result_26191 = invoke(stypy.reporting.localization.Localization(__file__, 116, 4), getitem___26190, int_26181)
    
    # Assigning a type to the variable 'tuple_var_assignment_25814' (line 116)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 116, 4), 'tuple_var_assignment_25814', subscript_call_result_26191)
    
    # Assigning a Name to a Name (line 116):
    # Getting the type of 'tuple_var_assignment_25814' (line 116)
    tuple_var_assignment_25814_26192 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 4), 'tuple_var_assignment_25814')
    # Assigning a type to the variable 'gges' (line 116)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 116, 4), 'gges', tuple_var_assignment_25814_26192)
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'lwork' (line 118)
    lwork_26193 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 7), 'lwork')
    # Getting the type of 'None' (line 118)
    None_26194 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 16), 'None')
    # Applying the binary operator 'is' (line 118)
    result_is__26195 = python_operator(stypy.reporting.localization.Localization(__file__, 118, 7), 'is', lwork_26193, None_26194)
    
    
    # Getting the type of 'lwork' (line 118)
    lwork_26196 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 24), 'lwork')
    int_26197 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 118, 33), 'int')
    # Applying the binary operator '==' (line 118)
    result_eq_26198 = python_operator(stypy.reporting.localization.Localization(__file__, 118, 24), '==', lwork_26196, int_26197)
    
    # Applying the binary operator 'or' (line 118)
    result_or_keyword_26199 = python_operator(stypy.reporting.localization.Localization(__file__, 118, 7), 'or', result_is__26195, result_eq_26198)
    
    # Testing the type of an if condition (line 118)
    if_condition_26200 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 118, 4), result_or_keyword_26199)
    # Assigning a type to the variable 'if_condition_26200' (line 118)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 118, 4), 'if_condition_26200', if_condition_26200)
    # SSA begins for if statement (line 118)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 120):
    
    # Assigning a Call to a Name (line 120):
    
    # Call to gges(...): (line 120)
    # Processing the call arguments (line 120)

    @norecursion
    def _stypy_temp_lambda_16(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_stypy_temp_lambda_16'
        module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_16', 120, 22, True)
        # Passed parameters checking function
        _stypy_temp_lambda_16.stypy_localization = localization
        _stypy_temp_lambda_16.stypy_type_of_self = None
        _stypy_temp_lambda_16.stypy_type_store = module_type_store
        _stypy_temp_lambda_16.stypy_function_name = '_stypy_temp_lambda_16'
        _stypy_temp_lambda_16.stypy_param_names_list = ['x']
        _stypy_temp_lambda_16.stypy_varargs_param_name = None
        _stypy_temp_lambda_16.stypy_kwargs_param_name = None
        _stypy_temp_lambda_16.stypy_call_defaults = defaults
        _stypy_temp_lambda_16.stypy_call_varargs = varargs
        _stypy_temp_lambda_16.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_16', ['x'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Stacktrace push for error reporting
        localization.set_stack_trace('_stypy_temp_lambda_16', ['x'], arguments)
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of the lambda function code ##################

        # Getting the type of 'None' (line 120)
        None_26202 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 32), 'None', False)
        # Assigning the return type of the lambda function
        # Assigning a type to the variable 'stypy_return_type' (line 120)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 120, 22), 'stypy_return_type', None_26202)
        
        # ################# End of the lambda function code ##################

        # Stacktrace pop (error reporting)
        localization.unset_stack_trace()
        
        # Storing the return type of function '_stypy_temp_lambda_16' in the type store
        # Getting the type of 'stypy_return_type' (line 120)
        stypy_return_type_26203 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 22), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_26203)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_stypy_temp_lambda_16'
        return stypy_return_type_26203

    # Assigning a type to the variable '_stypy_temp_lambda_16' (line 120)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 120, 22), '_stypy_temp_lambda_16', _stypy_temp_lambda_16)
    # Getting the type of '_stypy_temp_lambda_16' (line 120)
    _stypy_temp_lambda_16_26204 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 22), '_stypy_temp_lambda_16')
    # Getting the type of 'a1' (line 120)
    a1_26205 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 38), 'a1', False)
    # Getting the type of 'b1' (line 120)
    b1_26206 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 42), 'b1', False)
    # Processing the call keyword arguments (line 120)
    int_26207 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 120, 52), 'int')
    keyword_26208 = int_26207
    kwargs_26209 = {'lwork': keyword_26208}
    # Getting the type of 'gges' (line 120)
    gges_26201 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 17), 'gges', False)
    # Calling gges(args, kwargs) (line 120)
    gges_call_result_26210 = invoke(stypy.reporting.localization.Localization(__file__, 120, 17), gges_26201, *[_stypy_temp_lambda_16_26204, a1_26205, b1_26206], **kwargs_26209)
    
    # Assigning a type to the variable 'result' (line 120)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 120, 8), 'result', gges_call_result_26210)
    
    # Assigning a Call to a Name (line 121):
    
    # Assigning a Call to a Name (line 121):
    
    # Call to astype(...): (line 121)
    # Processing the call arguments (line 121)
    # Getting the type of 'np' (line 121)
    np_26220 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 42), 'np', False)
    # Obtaining the member 'int' of a type (line 121)
    int_26221 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 121, 42), np_26220, 'int')
    # Processing the call keyword arguments (line 121)
    kwargs_26222 = {}
    
    # Obtaining the type of the subscript
    int_26211 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 121, 27), 'int')
    
    # Obtaining the type of the subscript
    int_26212 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 121, 23), 'int')
    # Getting the type of 'result' (line 121)
    result_26213 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 16), 'result', False)
    # Obtaining the member '__getitem__' of a type (line 121)
    getitem___26214 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 121, 16), result_26213, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 121)
    subscript_call_result_26215 = invoke(stypy.reporting.localization.Localization(__file__, 121, 16), getitem___26214, int_26212)
    
    # Obtaining the member '__getitem__' of a type (line 121)
    getitem___26216 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 121, 16), subscript_call_result_26215, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 121)
    subscript_call_result_26217 = invoke(stypy.reporting.localization.Localization(__file__, 121, 16), getitem___26216, int_26211)
    
    # Obtaining the member 'real' of a type (line 121)
    real_26218 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 121, 16), subscript_call_result_26217, 'real')
    # Obtaining the member 'astype' of a type (line 121)
    astype_26219 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 121, 16), real_26218, 'astype')
    # Calling astype(args, kwargs) (line 121)
    astype_call_result_26223 = invoke(stypy.reporting.localization.Localization(__file__, 121, 16), astype_26219, *[int_26221], **kwargs_26222)
    
    # Assigning a type to the variable 'lwork' (line 121)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 121, 8), 'lwork', astype_call_result_26223)
    # SSA join for if statement (line 118)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Lambda to a Name (line 123):
    
    # Assigning a Lambda to a Name (line 123):

    @norecursion
    def _stypy_temp_lambda_17(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_stypy_temp_lambda_17'
        module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_17', 123, 16, True)
        # Passed parameters checking function
        _stypy_temp_lambda_17.stypy_localization = localization
        _stypy_temp_lambda_17.stypy_type_of_self = None
        _stypy_temp_lambda_17.stypy_type_store = module_type_store
        _stypy_temp_lambda_17.stypy_function_name = '_stypy_temp_lambda_17'
        _stypy_temp_lambda_17.stypy_param_names_list = ['x']
        _stypy_temp_lambda_17.stypy_varargs_param_name = None
        _stypy_temp_lambda_17.stypy_kwargs_param_name = None
        _stypy_temp_lambda_17.stypy_call_defaults = defaults
        _stypy_temp_lambda_17.stypy_call_varargs = varargs
        _stypy_temp_lambda_17.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_17', ['x'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Stacktrace push for error reporting
        localization.set_stack_trace('_stypy_temp_lambda_17', ['x'], arguments)
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of the lambda function code ##################

        # Getting the type of 'None' (line 123)
        None_26224 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 26), 'None')
        # Assigning the return type of the lambda function
        # Assigning a type to the variable 'stypy_return_type' (line 123)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 123, 16), 'stypy_return_type', None_26224)
        
        # ################# End of the lambda function code ##################

        # Stacktrace pop (error reporting)
        localization.unset_stack_trace()
        
        # Storing the return type of function '_stypy_temp_lambda_17' in the type store
        # Getting the type of 'stypy_return_type' (line 123)
        stypy_return_type_26225 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 16), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_26225)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_stypy_temp_lambda_17'
        return stypy_return_type_26225

    # Assigning a type to the variable '_stypy_temp_lambda_17' (line 123)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 123, 16), '_stypy_temp_lambda_17', _stypy_temp_lambda_17)
    # Getting the type of '_stypy_temp_lambda_17' (line 123)
    _stypy_temp_lambda_17_26226 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 16), '_stypy_temp_lambda_17')
    # Assigning a type to the variable 'sfunction' (line 123)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 123, 4), 'sfunction', _stypy_temp_lambda_17_26226)
    
    # Assigning a Call to a Name (line 124):
    
    # Assigning a Call to a Name (line 124):
    
    # Call to gges(...): (line 124)
    # Processing the call arguments (line 124)
    # Getting the type of 'sfunction' (line 124)
    sfunction_26228 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 18), 'sfunction', False)
    # Getting the type of 'a1' (line 124)
    a1_26229 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 29), 'a1', False)
    # Getting the type of 'b1' (line 124)
    b1_26230 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 33), 'b1', False)
    # Processing the call keyword arguments (line 124)
    # Getting the type of 'lwork' (line 124)
    lwork_26231 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 43), 'lwork', False)
    keyword_26232 = lwork_26231
    # Getting the type of 'overwrite_a' (line 124)
    overwrite_a_26233 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 62), 'overwrite_a', False)
    keyword_26234 = overwrite_a_26233
    # Getting the type of 'overwrite_b' (line 125)
    overwrite_b_26235 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 30), 'overwrite_b', False)
    keyword_26236 = overwrite_b_26235
    int_26237 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 125, 50), 'int')
    keyword_26238 = int_26237
    kwargs_26239 = {'sort_t': keyword_26238, 'overwrite_a': keyword_26234, 'lwork': keyword_26232, 'overwrite_b': keyword_26236}
    # Getting the type of 'gges' (line 124)
    gges_26227 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 13), 'gges', False)
    # Calling gges(args, kwargs) (line 124)
    gges_call_result_26240 = invoke(stypy.reporting.localization.Localization(__file__, 124, 13), gges_26227, *[sfunction_26228, a1_26229, b1_26230], **kwargs_26239)
    
    # Assigning a type to the variable 'result' (line 124)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 124, 4), 'result', gges_call_result_26240)
    
    # Assigning a Subscript to a Name (line 127):
    
    # Assigning a Subscript to a Name (line 127):
    
    # Obtaining the type of the subscript
    int_26241 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 127, 18), 'int')
    # Getting the type of 'result' (line 127)
    result_26242 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 11), 'result')
    # Obtaining the member '__getitem__' of a type (line 127)
    getitem___26243 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 127, 11), result_26242, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 127)
    subscript_call_result_26244 = invoke(stypy.reporting.localization.Localization(__file__, 127, 11), getitem___26243, int_26241)
    
    # Assigning a type to the variable 'info' (line 127)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 127, 4), 'info', subscript_call_result_26244)
    
    
    # Getting the type of 'info' (line 128)
    info_26245 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 7), 'info')
    int_26246 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 128, 14), 'int')
    # Applying the binary operator '<' (line 128)
    result_lt_26247 = python_operator(stypy.reporting.localization.Localization(__file__, 128, 7), '<', info_26245, int_26246)
    
    # Testing the type of an if condition (line 128)
    if_condition_26248 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 128, 4), result_lt_26247)
    # Assigning a type to the variable 'if_condition_26248' (line 128)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 128, 4), 'if_condition_26248', if_condition_26248)
    # SSA begins for if statement (line 128)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 129)
    # Processing the call arguments (line 129)
    str_26250 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 129, 25), 'str', 'Illegal value in argument %d of gges')
    
    # Getting the type of 'info' (line 129)
    info_26251 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 67), 'info', False)
    # Applying the 'usub' unary operator (line 129)
    result___neg___26252 = python_operator(stypy.reporting.localization.Localization(__file__, 129, 66), 'usub', info_26251)
    
    # Applying the binary operator '%' (line 129)
    result_mod_26253 = python_operator(stypy.reporting.localization.Localization(__file__, 129, 25), '%', str_26250, result___neg___26252)
    
    # Processing the call keyword arguments (line 129)
    kwargs_26254 = {}
    # Getting the type of 'ValueError' (line 129)
    ValueError_26249 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 129)
    ValueError_call_result_26255 = invoke(stypy.reporting.localization.Localization(__file__, 129, 14), ValueError_26249, *[result_mod_26253], **kwargs_26254)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 129, 8), ValueError_call_result_26255, 'raise parameter', BaseException)
    # SSA branch for the else part of an if statement (line 128)
    module_type_store.open_ssa_branch('else')
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'info' (line 130)
    info_26256 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 9), 'info')
    int_26257 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 130, 16), 'int')
    # Applying the binary operator '>' (line 130)
    result_gt_26258 = python_operator(stypy.reporting.localization.Localization(__file__, 130, 9), '>', info_26256, int_26257)
    
    
    # Getting the type of 'info' (line 130)
    info_26259 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 22), 'info')
    # Getting the type of 'a_n' (line 130)
    a_n_26260 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 30), 'a_n')
    # Applying the binary operator '<=' (line 130)
    result_le_26261 = python_operator(stypy.reporting.localization.Localization(__file__, 130, 22), '<=', info_26259, a_n_26260)
    
    # Applying the binary operator 'and' (line 130)
    result_and_keyword_26262 = python_operator(stypy.reporting.localization.Localization(__file__, 130, 9), 'and', result_gt_26258, result_le_26261)
    
    # Testing the type of an if condition (line 130)
    if_condition_26263 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 130, 9), result_and_keyword_26262)
    # Assigning a type to the variable 'if_condition_26263' (line 130)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 130, 9), 'if_condition_26263', if_condition_26263)
    # SSA begins for if statement (line 130)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to warn(...): (line 131)
    # Processing the call arguments (line 131)
    str_26266 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 131, 22), 'str', 'The QZ iteration failed. (a,b) are not in Schur form, but ALPHAR(j), ALPHAI(j), and BETA(j) should be correct for J=%d,...,N')
    # Getting the type of 'info' (line 133)
    info_26267 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 49), 'info', False)
    # Applying the binary operator '%' (line 131)
    result_mod_26268 = python_operator(stypy.reporting.localization.Localization(__file__, 131, 22), '%', str_26266, info_26267)
    
    int_26269 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 133, 54), 'int')
    # Applying the binary operator '-' (line 131)
    result_sub_26270 = python_operator(stypy.reporting.localization.Localization(__file__, 131, 22), '-', result_mod_26268, int_26269)
    
    # Getting the type of 'UserWarning' (line 133)
    UserWarning_26271 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 57), 'UserWarning', False)
    # Processing the call keyword arguments (line 131)
    kwargs_26272 = {}
    # Getting the type of 'warnings' (line 131)
    warnings_26264 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 8), 'warnings', False)
    # Obtaining the member 'warn' of a type (line 131)
    warn_26265 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 131, 8), warnings_26264, 'warn')
    # Calling warn(args, kwargs) (line 131)
    warn_call_result_26273 = invoke(stypy.reporting.localization.Localization(__file__, 131, 8), warn_26265, *[result_sub_26270, UserWarning_26271], **kwargs_26272)
    
    # SSA branch for the else part of an if statement (line 130)
    module_type_store.open_ssa_branch('else')
    
    
    # Getting the type of 'info' (line 134)
    info_26274 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 9), 'info')
    # Getting the type of 'a_n' (line 134)
    a_n_26275 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 17), 'a_n')
    int_26276 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 134, 21), 'int')
    # Applying the binary operator '+' (line 134)
    result_add_26277 = python_operator(stypy.reporting.localization.Localization(__file__, 134, 17), '+', a_n_26275, int_26276)
    
    # Applying the binary operator '==' (line 134)
    result_eq_26278 = python_operator(stypy.reporting.localization.Localization(__file__, 134, 9), '==', info_26274, result_add_26277)
    
    # Testing the type of an if condition (line 134)
    if_condition_26279 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 134, 9), result_eq_26278)
    # Assigning a type to the variable 'if_condition_26279' (line 134)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 134, 9), 'if_condition_26279', if_condition_26279)
    # SSA begins for if statement (line 134)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to LinAlgError(...): (line 135)
    # Processing the call arguments (line 135)
    str_26281 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 135, 26), 'str', 'Something other than QZ iteration failed')
    # Processing the call keyword arguments (line 135)
    kwargs_26282 = {}
    # Getting the type of 'LinAlgError' (line 135)
    LinAlgError_26280 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 14), 'LinAlgError', False)
    # Calling LinAlgError(args, kwargs) (line 135)
    LinAlgError_call_result_26283 = invoke(stypy.reporting.localization.Localization(__file__, 135, 14), LinAlgError_26280, *[str_26281], **kwargs_26282)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 135, 8), LinAlgError_call_result_26283, 'raise parameter', BaseException)
    # SSA branch for the else part of an if statement (line 134)
    module_type_store.open_ssa_branch('else')
    
    
    # Getting the type of 'info' (line 136)
    info_26284 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 9), 'info')
    # Getting the type of 'a_n' (line 136)
    a_n_26285 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 17), 'a_n')
    int_26286 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 136, 21), 'int')
    # Applying the binary operator '+' (line 136)
    result_add_26287 = python_operator(stypy.reporting.localization.Localization(__file__, 136, 17), '+', a_n_26285, int_26286)
    
    # Applying the binary operator '==' (line 136)
    result_eq_26288 = python_operator(stypy.reporting.localization.Localization(__file__, 136, 9), '==', info_26284, result_add_26287)
    
    # Testing the type of an if condition (line 136)
    if_condition_26289 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 136, 9), result_eq_26288)
    # Assigning a type to the variable 'if_condition_26289' (line 136)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 136, 9), 'if_condition_26289', if_condition_26289)
    # SSA begins for if statement (line 136)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to LinAlgError(...): (line 137)
    # Processing the call arguments (line 137)
    str_26291 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 137, 26), 'str', 'After reordering, roundoff changed values of some complex eigenvalues so that leading eigenvalues in the Generalized Schur form no longer satisfy sort=True. This could also be due to scaling.')
    # Processing the call keyword arguments (line 137)
    kwargs_26292 = {}
    # Getting the type of 'LinAlgError' (line 137)
    LinAlgError_26290 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 14), 'LinAlgError', False)
    # Calling LinAlgError(args, kwargs) (line 137)
    LinAlgError_call_result_26293 = invoke(stypy.reporting.localization.Localization(__file__, 137, 14), LinAlgError_26290, *[str_26291], **kwargs_26292)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 137, 8), LinAlgError_call_result_26293, 'raise parameter', BaseException)
    # SSA branch for the else part of an if statement (line 136)
    module_type_store.open_ssa_branch('else')
    
    
    # Getting the type of 'info' (line 141)
    info_26294 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 9), 'info')
    # Getting the type of 'a_n' (line 141)
    a_n_26295 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 17), 'a_n')
    int_26296 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 141, 21), 'int')
    # Applying the binary operator '+' (line 141)
    result_add_26297 = python_operator(stypy.reporting.localization.Localization(__file__, 141, 17), '+', a_n_26295, int_26296)
    
    # Applying the binary operator '==' (line 141)
    result_eq_26298 = python_operator(stypy.reporting.localization.Localization(__file__, 141, 9), '==', info_26294, result_add_26297)
    
    # Testing the type of an if condition (line 141)
    if_condition_26299 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 141, 9), result_eq_26298)
    # Assigning a type to the variable 'if_condition_26299' (line 141)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 141, 9), 'if_condition_26299', if_condition_26299)
    # SSA begins for if statement (line 141)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to LinAlgError(...): (line 142)
    # Processing the call arguments (line 142)
    str_26301 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 142, 26), 'str', 'Reordering failed in <s,d,c,z>tgsen')
    # Processing the call keyword arguments (line 142)
    kwargs_26302 = {}
    # Getting the type of 'LinAlgError' (line 142)
    LinAlgError_26300 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 14), 'LinAlgError', False)
    # Calling LinAlgError(args, kwargs) (line 142)
    LinAlgError_call_result_26303 = invoke(stypy.reporting.localization.Localization(__file__, 142, 14), LinAlgError_26300, *[str_26301], **kwargs_26302)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 142, 8), LinAlgError_call_result_26303, 'raise parameter', BaseException)
    # SSA join for if statement (line 141)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 136)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 134)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 130)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 128)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Obtaining an instance of the builtin type 'tuple' (line 144)
    tuple_26304 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 144, 11), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 144)
    # Adding element type (line 144)
    # Getting the type of 'result' (line 144)
    result_26305 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 11), 'result')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 144, 11), tuple_26304, result_26305)
    # Adding element type (line 144)
    # Getting the type of 'gges' (line 144)
    gges_26306 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 19), 'gges')
    # Obtaining the member 'typecode' of a type (line 144)
    typecode_26307 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 144, 19), gges_26306, 'typecode')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 144, 11), tuple_26304, typecode_26307)
    
    # Assigning a type to the variable 'stypy_return_type' (line 144)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 144, 4), 'stypy_return_type', tuple_26304)
    
    # ################# End of '_qz(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_qz' in the type store
    # Getting the type of 'stypy_return_type' (line 74)
    stypy_return_type_26308 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_26308)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_qz'
    return stypy_return_type_26308

# Assigning a type to the variable '_qz' (line 74)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 74, 0), '_qz', _qz)

@norecursion
def qz(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    str_26309 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 147, 20), 'str', 'real')
    # Getting the type of 'None' (line 147)
    None_26310 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 34), 'None')
    # Getting the type of 'None' (line 147)
    None_26311 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 45), 'None')
    # Getting the type of 'False' (line 147)
    False_26312 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 63), 'False')
    # Getting the type of 'False' (line 148)
    False_26313 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 19), 'False')
    # Getting the type of 'True' (line 148)
    True_26314 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 39), 'True')
    defaults = [str_26309, None_26310, None_26311, False_26312, False_26313, True_26314]
    # Create a new context for function 'qz'
    module_type_store = module_type_store.open_function_context('qz', 147, 0, False)
    
    # Passed parameters checking function
    qz.stypy_localization = localization
    qz.stypy_type_of_self = None
    qz.stypy_type_store = module_type_store
    qz.stypy_function_name = 'qz'
    qz.stypy_param_names_list = ['A', 'B', 'output', 'lwork', 'sort', 'overwrite_a', 'overwrite_b', 'check_finite']
    qz.stypy_varargs_param_name = None
    qz.stypy_kwargs_param_name = None
    qz.stypy_call_defaults = defaults
    qz.stypy_call_varargs = varargs
    qz.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'qz', ['A', 'B', 'output', 'lwork', 'sort', 'overwrite_a', 'overwrite_b', 'check_finite'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'qz', localization, ['A', 'B', 'output', 'lwork', 'sort', 'overwrite_a', 'overwrite_b', 'check_finite'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'qz(...)' code ##################

    str_26315 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 256, (-1)), 'str', "\n    QZ decomposition for generalized eigenvalues of a pair of matrices.\n\n    The QZ, or generalized Schur, decomposition for a pair of N x N\n    nonsymmetric matrices (A,B) is::\n\n        (A,B) = (Q*AA*Z', Q*BB*Z')\n\n    where AA, BB is in generalized Schur form if BB is upper-triangular\n    with non-negative diagonal and AA is upper-triangular, or for real QZ\n    decomposition (``output='real'``) block upper triangular with 1x1\n    and 2x2 blocks.  In this case, the 1x1 blocks correspond to real\n    generalized eigenvalues and 2x2 blocks are 'standardized' by making\n    the corresponding elements of BB have the form::\n\n        [ a 0 ]\n        [ 0 b ]\n\n    and the pair of corresponding 2x2 blocks in AA and BB will have a complex\n    conjugate pair of generalized eigenvalues.  If (``output='complex'``) or\n    A and B are complex matrices, Z' denotes the conjugate-transpose of Z.\n    Q and Z are unitary matrices.\n\n    Parameters\n    ----------\n    A : (N, N) array_like\n        2d array to decompose\n    B : (N, N) array_like\n        2d array to decompose\n    output : {'real', 'complex'}, optional\n        Construct the real or complex QZ decomposition for real matrices.\n        Default is 'real'.\n    lwork : int, optional\n        Work array size.  If None or -1, it is automatically computed.\n    sort : {None, callable, 'lhp', 'rhp', 'iuc', 'ouc'}, optional\n        NOTE: THIS INPUT IS DISABLED FOR NOW. Use ordqz instead.\n\n        Specifies whether the upper eigenvalues should be sorted.  A callable\n        may be passed that, given a eigenvalue, returns a boolean denoting\n        whether the eigenvalue should be sorted to the top-left (True). For\n        real matrix pairs, the sort function takes three real arguments\n        (alphar, alphai, beta). The eigenvalue\n        ``x = (alphar + alphai*1j)/beta``.  For complex matrix pairs or\n        output='complex', the sort function takes two complex arguments\n        (alpha, beta). The eigenvalue ``x = (alpha/beta)``.  Alternatively,\n        string parameters may be used:\n\n            - 'lhp'   Left-hand plane (x.real < 0.0)\n            - 'rhp'   Right-hand plane (x.real > 0.0)\n            - 'iuc'   Inside the unit circle (x*x.conjugate() < 1.0)\n            - 'ouc'   Outside the unit circle (x*x.conjugate() > 1.0)\n\n        Defaults to None (no sorting).\n    overwrite_a : bool, optional\n        Whether to overwrite data in a (may improve performance)\n    overwrite_b : bool, optional\n        Whether to overwrite data in b (may improve performance)\n    check_finite : bool, optional\n        If true checks the elements of `A` and `B` are finite numbers. If\n        false does no checking and passes matrix through to\n        underlying algorithm.\n\n    Returns\n    -------\n    AA : (N, N) ndarray\n        Generalized Schur form of A.\n    BB : (N, N) ndarray\n        Generalized Schur form of B.\n    Q : (N, N) ndarray\n        The left Schur vectors.\n    Z : (N, N) ndarray\n        The right Schur vectors.\n\n    Notes\n    -----\n    Q is transposed versus the equivalent function in Matlab.\n\n    .. versionadded:: 0.11.0\n\n    Examples\n    --------\n    >>> from scipy import linalg\n    >>> np.random.seed(1234)\n    >>> A = np.arange(9).reshape((3, 3))\n    >>> B = np.random.randn(3, 3)\n\n    >>> AA, BB, Q, Z = linalg.qz(A, B)\n    >>> AA\n    array([[-13.40928183,  -4.62471562,   1.09215523],\n           [  0.        ,   0.        ,   1.22805978],\n           [  0.        ,   0.        ,   0.31973817]])\n    >>> BB\n    array([[ 0.33362547, -1.37393632,  0.02179805],\n           [ 0.        ,  1.68144922,  0.74683866],\n           [ 0.        ,  0.        ,  0.9258294 ]])\n    >>> Q\n    array([[ 0.14134727, -0.97562773,  0.16784365],\n           [ 0.49835904, -0.07636948, -0.86360059],\n           [ 0.85537081,  0.20571399,  0.47541828]])\n    >>> Z\n    array([[-0.24900855, -0.51772687,  0.81850696],\n           [-0.79813178,  0.58842606,  0.12938478],\n           [-0.54861681, -0.6210585 , -0.55973739]])\n\n    See also\n    --------\n    ordqz\n    ")
    
    # Assigning a Call to a Tuple (line 261):
    
    # Assigning a Subscript to a Name (line 261):
    
    # Obtaining the type of the subscript
    int_26316 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 261, 4), 'int')
    
    # Call to _qz(...): (line 261)
    # Processing the call arguments (line 261)
    # Getting the type of 'A' (line 261)
    A_26318 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 261, 20), 'A', False)
    # Getting the type of 'B' (line 261)
    B_26319 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 261, 23), 'B', False)
    # Processing the call keyword arguments (line 261)
    # Getting the type of 'output' (line 261)
    output_26320 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 261, 33), 'output', False)
    keyword_26321 = output_26320
    # Getting the type of 'lwork' (line 261)
    lwork_26322 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 261, 47), 'lwork', False)
    keyword_26323 = lwork_26322
    # Getting the type of 'sort' (line 261)
    sort_26324 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 261, 59), 'sort', False)
    keyword_26325 = sort_26324
    # Getting the type of 'overwrite_a' (line 262)
    overwrite_a_26326 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 262, 32), 'overwrite_a', False)
    keyword_26327 = overwrite_a_26326
    # Getting the type of 'overwrite_b' (line 262)
    overwrite_b_26328 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 262, 57), 'overwrite_b', False)
    keyword_26329 = overwrite_b_26328
    # Getting the type of 'check_finite' (line 263)
    check_finite_26330 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 263, 33), 'check_finite', False)
    keyword_26331 = check_finite_26330
    kwargs_26332 = {'sort': keyword_26325, 'overwrite_a': keyword_26327, 'overwrite_b': keyword_26329, 'lwork': keyword_26323, 'output': keyword_26321, 'check_finite': keyword_26331}
    # Getting the type of '_qz' (line 261)
    _qz_26317 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 261, 16), '_qz', False)
    # Calling _qz(args, kwargs) (line 261)
    _qz_call_result_26333 = invoke(stypy.reporting.localization.Localization(__file__, 261, 16), _qz_26317, *[A_26318, B_26319], **kwargs_26332)
    
    # Obtaining the member '__getitem__' of a type (line 261)
    getitem___26334 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 261, 4), _qz_call_result_26333, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 261)
    subscript_call_result_26335 = invoke(stypy.reporting.localization.Localization(__file__, 261, 4), getitem___26334, int_26316)
    
    # Assigning a type to the variable 'tuple_var_assignment_25815' (line 261)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 261, 4), 'tuple_var_assignment_25815', subscript_call_result_26335)
    
    # Assigning a Subscript to a Name (line 261):
    
    # Obtaining the type of the subscript
    int_26336 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 261, 4), 'int')
    
    # Call to _qz(...): (line 261)
    # Processing the call arguments (line 261)
    # Getting the type of 'A' (line 261)
    A_26338 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 261, 20), 'A', False)
    # Getting the type of 'B' (line 261)
    B_26339 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 261, 23), 'B', False)
    # Processing the call keyword arguments (line 261)
    # Getting the type of 'output' (line 261)
    output_26340 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 261, 33), 'output', False)
    keyword_26341 = output_26340
    # Getting the type of 'lwork' (line 261)
    lwork_26342 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 261, 47), 'lwork', False)
    keyword_26343 = lwork_26342
    # Getting the type of 'sort' (line 261)
    sort_26344 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 261, 59), 'sort', False)
    keyword_26345 = sort_26344
    # Getting the type of 'overwrite_a' (line 262)
    overwrite_a_26346 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 262, 32), 'overwrite_a', False)
    keyword_26347 = overwrite_a_26346
    # Getting the type of 'overwrite_b' (line 262)
    overwrite_b_26348 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 262, 57), 'overwrite_b', False)
    keyword_26349 = overwrite_b_26348
    # Getting the type of 'check_finite' (line 263)
    check_finite_26350 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 263, 33), 'check_finite', False)
    keyword_26351 = check_finite_26350
    kwargs_26352 = {'sort': keyword_26345, 'overwrite_a': keyword_26347, 'overwrite_b': keyword_26349, 'lwork': keyword_26343, 'output': keyword_26341, 'check_finite': keyword_26351}
    # Getting the type of '_qz' (line 261)
    _qz_26337 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 261, 16), '_qz', False)
    # Calling _qz(args, kwargs) (line 261)
    _qz_call_result_26353 = invoke(stypy.reporting.localization.Localization(__file__, 261, 16), _qz_26337, *[A_26338, B_26339], **kwargs_26352)
    
    # Obtaining the member '__getitem__' of a type (line 261)
    getitem___26354 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 261, 4), _qz_call_result_26353, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 261)
    subscript_call_result_26355 = invoke(stypy.reporting.localization.Localization(__file__, 261, 4), getitem___26354, int_26336)
    
    # Assigning a type to the variable 'tuple_var_assignment_25816' (line 261)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 261, 4), 'tuple_var_assignment_25816', subscript_call_result_26355)
    
    # Assigning a Name to a Name (line 261):
    # Getting the type of 'tuple_var_assignment_25815' (line 261)
    tuple_var_assignment_25815_26356 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 261, 4), 'tuple_var_assignment_25815')
    # Assigning a type to the variable 'result' (line 261)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 261, 4), 'result', tuple_var_assignment_25815_26356)
    
    # Assigning a Name to a Name (line 261):
    # Getting the type of 'tuple_var_assignment_25816' (line 261)
    tuple_var_assignment_25816_26357 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 261, 4), 'tuple_var_assignment_25816')
    # Assigning a type to the variable '_' (line 261)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 261, 12), '_', tuple_var_assignment_25816_26357)
    
    # Obtaining an instance of the builtin type 'tuple' (line 264)
    tuple_26358 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 264, 11), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 264)
    # Adding element type (line 264)
    
    # Obtaining the type of the subscript
    int_26359 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 264, 18), 'int')
    # Getting the type of 'result' (line 264)
    result_26360 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 264, 11), 'result')
    # Obtaining the member '__getitem__' of a type (line 264)
    getitem___26361 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 264, 11), result_26360, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 264)
    subscript_call_result_26362 = invoke(stypy.reporting.localization.Localization(__file__, 264, 11), getitem___26361, int_26359)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 264, 11), tuple_26358, subscript_call_result_26362)
    # Adding element type (line 264)
    
    # Obtaining the type of the subscript
    int_26363 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 264, 29), 'int')
    # Getting the type of 'result' (line 264)
    result_26364 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 264, 22), 'result')
    # Obtaining the member '__getitem__' of a type (line 264)
    getitem___26365 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 264, 22), result_26364, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 264)
    subscript_call_result_26366 = invoke(stypy.reporting.localization.Localization(__file__, 264, 22), getitem___26365, int_26363)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 264, 11), tuple_26358, subscript_call_result_26366)
    # Adding element type (line 264)
    
    # Obtaining the type of the subscript
    int_26367 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 264, 40), 'int')
    # Getting the type of 'result' (line 264)
    result_26368 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 264, 33), 'result')
    # Obtaining the member '__getitem__' of a type (line 264)
    getitem___26369 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 264, 33), result_26368, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 264)
    subscript_call_result_26370 = invoke(stypy.reporting.localization.Localization(__file__, 264, 33), getitem___26369, int_26367)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 264, 11), tuple_26358, subscript_call_result_26370)
    # Adding element type (line 264)
    
    # Obtaining the type of the subscript
    int_26371 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 264, 52), 'int')
    # Getting the type of 'result' (line 264)
    result_26372 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 264, 45), 'result')
    # Obtaining the member '__getitem__' of a type (line 264)
    getitem___26373 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 264, 45), result_26372, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 264)
    subscript_call_result_26374 = invoke(stypy.reporting.localization.Localization(__file__, 264, 45), getitem___26373, int_26371)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 264, 11), tuple_26358, subscript_call_result_26374)
    
    # Assigning a type to the variable 'stypy_return_type' (line 264)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 264, 4), 'stypy_return_type', tuple_26358)
    
    # ################# End of 'qz(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'qz' in the type store
    # Getting the type of 'stypy_return_type' (line 147)
    stypy_return_type_26375 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_26375)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'qz'
    return stypy_return_type_26375

# Assigning a type to the variable 'qz' (line 147)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 147, 0), 'qz', qz)

@norecursion
def ordqz(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    str_26376 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 267, 21), 'str', 'lhp')
    str_26377 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 267, 35), 'str', 'real')
    # Getting the type of 'False' (line 267)
    False_26378 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 267, 55), 'False')
    # Getting the type of 'False' (line 268)
    False_26379 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 268, 22), 'False')
    # Getting the type of 'True' (line 268)
    True_26380 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 268, 42), 'True')
    defaults = [str_26376, str_26377, False_26378, False_26379, True_26380]
    # Create a new context for function 'ordqz'
    module_type_store = module_type_store.open_function_context('ordqz', 267, 0, False)
    
    # Passed parameters checking function
    ordqz.stypy_localization = localization
    ordqz.stypy_type_of_self = None
    ordqz.stypy_type_store = module_type_store
    ordqz.stypy_function_name = 'ordqz'
    ordqz.stypy_param_names_list = ['A', 'B', 'sort', 'output', 'overwrite_a', 'overwrite_b', 'check_finite']
    ordqz.stypy_varargs_param_name = None
    ordqz.stypy_kwargs_param_name = None
    ordqz.stypy_call_defaults = defaults
    ordqz.stypy_call_varargs = varargs
    ordqz.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'ordqz', ['A', 'B', 'sort', 'output', 'overwrite_a', 'overwrite_b', 'check_finite'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'ordqz', localization, ['A', 'B', 'sort', 'output', 'overwrite_a', 'overwrite_b', 'check_finite'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'ordqz(...)' code ##################

    str_26381 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 344, (-1)), 'str', "QZ decomposition for a pair of matrices with reordering.\n\n    .. versionadded:: 0.17.0\n\n    Parameters\n    ----------\n    A : (N, N) array_like\n        2d array to decompose\n    B : (N, N) array_like\n        2d array to decompose\n    sort : {callable, 'lhp', 'rhp', 'iuc', 'ouc'}, optional\n        Specifies whether the upper eigenvalues should be sorted. A\n        callable may be passed that, given an ordered pair ``(alpha,\n        beta)`` representing the eigenvalue ``x = (alpha/beta)``,\n        returns a boolean denoting whether the eigenvalue should be\n        sorted to the top-left (True). For the real matrix pairs\n        ``beta`` is real while ``alpha`` can be complex, and for\n        complex matrix pairs both ``alpha`` and ``beta`` can be\n        complex. The callable must be able to accept a numpy\n        array. Alternatively, string parameters may be used:\n\n            - 'lhp'   Left-hand plane (x.real < 0.0)\n            - 'rhp'   Right-hand plane (x.real > 0.0)\n            - 'iuc'   Inside the unit circle (x*x.conjugate() < 1.0)\n            - 'ouc'   Outside the unit circle (x*x.conjugate() > 1.0)\n\n        With the predefined sorting functions, an infinite eigenvalue\n        (i.e. ``alpha != 0`` and ``beta = 0``) is considered to lie in\n        neither the left-hand nor the right-hand plane, but it is\n        considered to lie outside the unit circle. For the eigenvalue\n        ``(alpha, beta) = (0, 0)`` the predefined sorting functions\n        all return `False`.\n\n    output : str {'real','complex'}, optional\n        Construct the real or complex QZ decomposition for real matrices.\n        Default is 'real'.\n    overwrite_a : bool, optional\n        If True, the contents of A are overwritten.\n    overwrite_b : bool, optional\n        If True, the contents of B are overwritten.\n    check_finite : bool, optional\n        If true checks the elements of `A` and `B` are finite numbers. If\n        false does no checking and passes matrix through to\n        underlying algorithm.\n\n    Returns\n    -------\n    AA : (N, N) ndarray\n        Generalized Schur form of A.\n    BB : (N, N) ndarray\n        Generalized Schur form of B.\n    alpha : (N,) ndarray\n        alpha = alphar + alphai * 1j. See notes.\n    beta : (N,) ndarray\n        See notes.\n    Q : (N, N) ndarray\n        The left Schur vectors.\n    Z : (N, N) ndarray\n        The right Schur vectors.\n\n    Notes\n    -----\n    On exit, ``(ALPHAR(j) + ALPHAI(j)*i)/BETA(j), j=1,...,N``, will be the\n    generalized eigenvalues.  ``ALPHAR(j) + ALPHAI(j)*i`` and\n    ``BETA(j),j=1,...,N`` are the diagonals of the complex Schur form (S,T)\n    that would result if the 2-by-2 diagonal blocks of the real generalized\n    Schur form of (A,B) were further reduced to triangular form using complex\n    unitary transformations. If ALPHAI(j) is zero, then the j-th eigenvalue is\n    real; if positive, then the ``j``-th and ``(j+1)``-st eigenvalues are a complex\n    conjugate pair, with ``ALPHAI(j+1)`` negative.\n\n    See also\n    --------\n    qz\n\n    ")
    
    # Assigning a Name to a Name (line 346):
    
    # Assigning a Name to a Name (line 346):
    # Getting the type of 'None' (line 346)
    None_26382 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 346, 12), 'None')
    # Assigning a type to the variable 'lwork' (line 346)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 346, 4), 'lwork', None_26382)
    
    # Assigning a Call to a Tuple (line 347):
    
    # Assigning a Subscript to a Name (line 347):
    
    # Obtaining the type of the subscript
    int_26383 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 347, 4), 'int')
    
    # Call to _qz(...): (line 347)
    # Processing the call arguments (line 347)
    # Getting the type of 'A' (line 347)
    A_26385 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 347, 22), 'A', False)
    # Getting the type of 'B' (line 347)
    B_26386 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 347, 25), 'B', False)
    # Processing the call keyword arguments (line 347)
    # Getting the type of 'output' (line 347)
    output_26387 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 347, 35), 'output', False)
    keyword_26388 = output_26387
    # Getting the type of 'lwork' (line 347)
    lwork_26389 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 347, 49), 'lwork', False)
    keyword_26390 = lwork_26389
    # Getting the type of 'None' (line 347)
    None_26391 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 347, 61), 'None', False)
    keyword_26392 = None_26391
    # Getting the type of 'overwrite_a' (line 348)
    overwrite_a_26393 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 348, 34), 'overwrite_a', False)
    keyword_26394 = overwrite_a_26393
    # Getting the type of 'overwrite_b' (line 348)
    overwrite_b_26395 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 348, 59), 'overwrite_b', False)
    keyword_26396 = overwrite_b_26395
    # Getting the type of 'check_finite' (line 349)
    check_finite_26397 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 349, 35), 'check_finite', False)
    keyword_26398 = check_finite_26397
    kwargs_26399 = {'sort': keyword_26392, 'overwrite_a': keyword_26394, 'overwrite_b': keyword_26396, 'lwork': keyword_26390, 'output': keyword_26388, 'check_finite': keyword_26398}
    # Getting the type of '_qz' (line 347)
    _qz_26384 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 347, 18), '_qz', False)
    # Calling _qz(args, kwargs) (line 347)
    _qz_call_result_26400 = invoke(stypy.reporting.localization.Localization(__file__, 347, 18), _qz_26384, *[A_26385, B_26386], **kwargs_26399)
    
    # Obtaining the member '__getitem__' of a type (line 347)
    getitem___26401 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 347, 4), _qz_call_result_26400, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 347)
    subscript_call_result_26402 = invoke(stypy.reporting.localization.Localization(__file__, 347, 4), getitem___26401, int_26383)
    
    # Assigning a type to the variable 'tuple_var_assignment_25817' (line 347)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 347, 4), 'tuple_var_assignment_25817', subscript_call_result_26402)
    
    # Assigning a Subscript to a Name (line 347):
    
    # Obtaining the type of the subscript
    int_26403 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 347, 4), 'int')
    
    # Call to _qz(...): (line 347)
    # Processing the call arguments (line 347)
    # Getting the type of 'A' (line 347)
    A_26405 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 347, 22), 'A', False)
    # Getting the type of 'B' (line 347)
    B_26406 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 347, 25), 'B', False)
    # Processing the call keyword arguments (line 347)
    # Getting the type of 'output' (line 347)
    output_26407 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 347, 35), 'output', False)
    keyword_26408 = output_26407
    # Getting the type of 'lwork' (line 347)
    lwork_26409 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 347, 49), 'lwork', False)
    keyword_26410 = lwork_26409
    # Getting the type of 'None' (line 347)
    None_26411 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 347, 61), 'None', False)
    keyword_26412 = None_26411
    # Getting the type of 'overwrite_a' (line 348)
    overwrite_a_26413 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 348, 34), 'overwrite_a', False)
    keyword_26414 = overwrite_a_26413
    # Getting the type of 'overwrite_b' (line 348)
    overwrite_b_26415 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 348, 59), 'overwrite_b', False)
    keyword_26416 = overwrite_b_26415
    # Getting the type of 'check_finite' (line 349)
    check_finite_26417 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 349, 35), 'check_finite', False)
    keyword_26418 = check_finite_26417
    kwargs_26419 = {'sort': keyword_26412, 'overwrite_a': keyword_26414, 'overwrite_b': keyword_26416, 'lwork': keyword_26410, 'output': keyword_26408, 'check_finite': keyword_26418}
    # Getting the type of '_qz' (line 347)
    _qz_26404 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 347, 18), '_qz', False)
    # Calling _qz(args, kwargs) (line 347)
    _qz_call_result_26420 = invoke(stypy.reporting.localization.Localization(__file__, 347, 18), _qz_26404, *[A_26405, B_26406], **kwargs_26419)
    
    # Obtaining the member '__getitem__' of a type (line 347)
    getitem___26421 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 347, 4), _qz_call_result_26420, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 347)
    subscript_call_result_26422 = invoke(stypy.reporting.localization.Localization(__file__, 347, 4), getitem___26421, int_26403)
    
    # Assigning a type to the variable 'tuple_var_assignment_25818' (line 347)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 347, 4), 'tuple_var_assignment_25818', subscript_call_result_26422)
    
    # Assigning a Name to a Name (line 347):
    # Getting the type of 'tuple_var_assignment_25817' (line 347)
    tuple_var_assignment_25817_26423 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 347, 4), 'tuple_var_assignment_25817')
    # Assigning a type to the variable 'result' (line 347)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 347, 4), 'result', tuple_var_assignment_25817_26423)
    
    # Assigning a Name to a Name (line 347):
    # Getting the type of 'tuple_var_assignment_25818' (line 347)
    tuple_var_assignment_25818_26424 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 347, 4), 'tuple_var_assignment_25818')
    # Assigning a type to the variable 'typ' (line 347)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 347, 12), 'typ', tuple_var_assignment_25818_26424)
    
    # Assigning a Tuple to a Tuple (line 350):
    
    # Assigning a Subscript to a Name (line 350):
    
    # Obtaining the type of the subscript
    int_26425 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 350, 26), 'int')
    # Getting the type of 'result' (line 350)
    result_26426 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 350, 19), 'result')
    # Obtaining the member '__getitem__' of a type (line 350)
    getitem___26427 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 350, 19), result_26426, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 350)
    subscript_call_result_26428 = invoke(stypy.reporting.localization.Localization(__file__, 350, 19), getitem___26427, int_26425)
    
    # Assigning a type to the variable 'tuple_assignment_25819' (line 350)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 350, 4), 'tuple_assignment_25819', subscript_call_result_26428)
    
    # Assigning a Subscript to a Name (line 350):
    
    # Obtaining the type of the subscript
    int_26429 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 350, 37), 'int')
    # Getting the type of 'result' (line 350)
    result_26430 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 350, 30), 'result')
    # Obtaining the member '__getitem__' of a type (line 350)
    getitem___26431 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 350, 30), result_26430, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 350)
    subscript_call_result_26432 = invoke(stypy.reporting.localization.Localization(__file__, 350, 30), getitem___26431, int_26429)
    
    # Assigning a type to the variable 'tuple_assignment_25820' (line 350)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 350, 4), 'tuple_assignment_25820', subscript_call_result_26432)
    
    # Assigning a Subscript to a Name (line 350):
    
    # Obtaining the type of the subscript
    int_26433 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 350, 48), 'int')
    # Getting the type of 'result' (line 350)
    result_26434 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 350, 41), 'result')
    # Obtaining the member '__getitem__' of a type (line 350)
    getitem___26435 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 350, 41), result_26434, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 350)
    subscript_call_result_26436 = invoke(stypy.reporting.localization.Localization(__file__, 350, 41), getitem___26435, int_26433)
    
    # Assigning a type to the variable 'tuple_assignment_25821' (line 350)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 350, 4), 'tuple_assignment_25821', subscript_call_result_26436)
    
    # Assigning a Subscript to a Name (line 350):
    
    # Obtaining the type of the subscript
    int_26437 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 350, 60), 'int')
    # Getting the type of 'result' (line 350)
    result_26438 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 350, 53), 'result')
    # Obtaining the member '__getitem__' of a type (line 350)
    getitem___26439 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 350, 53), result_26438, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 350)
    subscript_call_result_26440 = invoke(stypy.reporting.localization.Localization(__file__, 350, 53), getitem___26439, int_26437)
    
    # Assigning a type to the variable 'tuple_assignment_25822' (line 350)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 350, 4), 'tuple_assignment_25822', subscript_call_result_26440)
    
    # Assigning a Name to a Name (line 350):
    # Getting the type of 'tuple_assignment_25819' (line 350)
    tuple_assignment_25819_26441 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 350, 4), 'tuple_assignment_25819')
    # Assigning a type to the variable 'AA' (line 350)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 350, 4), 'AA', tuple_assignment_25819_26441)
    
    # Assigning a Name to a Name (line 350):
    # Getting the type of 'tuple_assignment_25820' (line 350)
    tuple_assignment_25820_26442 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 350, 4), 'tuple_assignment_25820')
    # Assigning a type to the variable 'BB' (line 350)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 350, 8), 'BB', tuple_assignment_25820_26442)
    
    # Assigning a Name to a Name (line 350):
    # Getting the type of 'tuple_assignment_25821' (line 350)
    tuple_assignment_25821_26443 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 350, 4), 'tuple_assignment_25821')
    # Assigning a type to the variable 'Q' (line 350)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 350, 12), 'Q', tuple_assignment_25821_26443)
    
    # Assigning a Name to a Name (line 350):
    # Getting the type of 'tuple_assignment_25822' (line 350)
    tuple_assignment_25822_26444 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 350, 4), 'tuple_assignment_25822')
    # Assigning a type to the variable 'Z' (line 350)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 350, 15), 'Z', tuple_assignment_25822_26444)
    
    
    # Getting the type of 'typ' (line 351)
    typ_26445 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 351, 7), 'typ')
    str_26446 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 351, 18), 'str', 'cz')
    # Applying the binary operator 'notin' (line 351)
    result_contains_26447 = python_operator(stypy.reporting.localization.Localization(__file__, 351, 7), 'notin', typ_26445, str_26446)
    
    # Testing the type of an if condition (line 351)
    if_condition_26448 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 351, 4), result_contains_26447)
    # Assigning a type to the variable 'if_condition_26448' (line 351)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 351, 4), 'if_condition_26448', if_condition_26448)
    # SSA begins for if statement (line 351)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Tuple to a Tuple (line 352):
    
    # Assigning a BinOp to a Name (line 352):
    
    # Obtaining the type of the subscript
    int_26449 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 352, 29), 'int')
    # Getting the type of 'result' (line 352)
    result_26450 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 352, 22), 'result')
    # Obtaining the member '__getitem__' of a type (line 352)
    getitem___26451 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 352, 22), result_26450, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 352)
    subscript_call_result_26452 = invoke(stypy.reporting.localization.Localization(__file__, 352, 22), getitem___26451, int_26449)
    
    
    # Obtaining the type of the subscript
    int_26453 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 352, 41), 'int')
    # Getting the type of 'result' (line 352)
    result_26454 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 352, 34), 'result')
    # Obtaining the member '__getitem__' of a type (line 352)
    getitem___26455 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 352, 34), result_26454, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 352)
    subscript_call_result_26456 = invoke(stypy.reporting.localization.Localization(__file__, 352, 34), getitem___26455, int_26453)
    
    complex_26457 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 352, 44), 'complex')
    # Applying the binary operator '*' (line 352)
    result_mul_26458 = python_operator(stypy.reporting.localization.Localization(__file__, 352, 34), '*', subscript_call_result_26456, complex_26457)
    
    # Applying the binary operator '+' (line 352)
    result_add_26459 = python_operator(stypy.reporting.localization.Localization(__file__, 352, 22), '+', subscript_call_result_26452, result_mul_26458)
    
    # Assigning a type to the variable 'tuple_assignment_25823' (line 352)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 352, 8), 'tuple_assignment_25823', result_add_26459)
    
    # Assigning a Subscript to a Name (line 352):
    
    # Obtaining the type of the subscript
    int_26460 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 352, 56), 'int')
    # Getting the type of 'result' (line 352)
    result_26461 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 352, 49), 'result')
    # Obtaining the member '__getitem__' of a type (line 352)
    getitem___26462 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 352, 49), result_26461, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 352)
    subscript_call_result_26463 = invoke(stypy.reporting.localization.Localization(__file__, 352, 49), getitem___26462, int_26460)
    
    # Assigning a type to the variable 'tuple_assignment_25824' (line 352)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 352, 8), 'tuple_assignment_25824', subscript_call_result_26463)
    
    # Assigning a Name to a Name (line 352):
    # Getting the type of 'tuple_assignment_25823' (line 352)
    tuple_assignment_25823_26464 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 352, 8), 'tuple_assignment_25823')
    # Assigning a type to the variable 'alpha' (line 352)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 352, 8), 'alpha', tuple_assignment_25823_26464)
    
    # Assigning a Name to a Name (line 352):
    # Getting the type of 'tuple_assignment_25824' (line 352)
    tuple_assignment_25824_26465 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 352, 8), 'tuple_assignment_25824')
    # Assigning a type to the variable 'beta' (line 352)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 352, 15), 'beta', tuple_assignment_25824_26465)
    # SSA branch for the else part of an if statement (line 351)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Tuple to a Tuple (line 354):
    
    # Assigning a Subscript to a Name (line 354):
    
    # Obtaining the type of the subscript
    int_26466 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 354, 29), 'int')
    # Getting the type of 'result' (line 354)
    result_26467 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 354, 22), 'result')
    # Obtaining the member '__getitem__' of a type (line 354)
    getitem___26468 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 354, 22), result_26467, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 354)
    subscript_call_result_26469 = invoke(stypy.reporting.localization.Localization(__file__, 354, 22), getitem___26468, int_26466)
    
    # Assigning a type to the variable 'tuple_assignment_25825' (line 354)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 354, 8), 'tuple_assignment_25825', subscript_call_result_26469)
    
    # Assigning a Subscript to a Name (line 354):
    
    # Obtaining the type of the subscript
    int_26470 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 354, 40), 'int')
    # Getting the type of 'result' (line 354)
    result_26471 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 354, 33), 'result')
    # Obtaining the member '__getitem__' of a type (line 354)
    getitem___26472 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 354, 33), result_26471, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 354)
    subscript_call_result_26473 = invoke(stypy.reporting.localization.Localization(__file__, 354, 33), getitem___26472, int_26470)
    
    # Assigning a type to the variable 'tuple_assignment_25826' (line 354)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 354, 8), 'tuple_assignment_25826', subscript_call_result_26473)
    
    # Assigning a Name to a Name (line 354):
    # Getting the type of 'tuple_assignment_25825' (line 354)
    tuple_assignment_25825_26474 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 354, 8), 'tuple_assignment_25825')
    # Assigning a type to the variable 'alpha' (line 354)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 354, 8), 'alpha', tuple_assignment_25825_26474)
    
    # Assigning a Name to a Name (line 354):
    # Getting the type of 'tuple_assignment_25826' (line 354)
    tuple_assignment_25826_26475 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 354, 8), 'tuple_assignment_25826')
    # Assigning a type to the variable 'beta' (line 354)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 354, 15), 'beta', tuple_assignment_25826_26475)
    # SSA join for if statement (line 351)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 356):
    
    # Assigning a Call to a Name (line 356):
    
    # Call to _select_function(...): (line 356)
    # Processing the call arguments (line 356)
    # Getting the type of 'sort' (line 356)
    sort_26477 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 356, 33), 'sort', False)
    # Processing the call keyword arguments (line 356)
    kwargs_26478 = {}
    # Getting the type of '_select_function' (line 356)
    _select_function_26476 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 356, 16), '_select_function', False)
    # Calling _select_function(args, kwargs) (line 356)
    _select_function_call_result_26479 = invoke(stypy.reporting.localization.Localization(__file__, 356, 16), _select_function_26476, *[sort_26477], **kwargs_26478)
    
    # Assigning a type to the variable 'sfunction' (line 356)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 356, 4), 'sfunction', _select_function_call_result_26479)
    
    # Assigning a Call to a Name (line 357):
    
    # Assigning a Call to a Name (line 357):
    
    # Call to sfunction(...): (line 357)
    # Processing the call arguments (line 357)
    # Getting the type of 'alpha' (line 357)
    alpha_26481 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 357, 23), 'alpha', False)
    # Getting the type of 'beta' (line 357)
    beta_26482 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 357, 30), 'beta', False)
    # Processing the call keyword arguments (line 357)
    kwargs_26483 = {}
    # Getting the type of 'sfunction' (line 357)
    sfunction_26480 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 357, 13), 'sfunction', False)
    # Calling sfunction(args, kwargs) (line 357)
    sfunction_call_result_26484 = invoke(stypy.reporting.localization.Localization(__file__, 357, 13), sfunction_26480, *[alpha_26481, beta_26482], **kwargs_26483)
    
    # Assigning a type to the variable 'select' (line 357)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 357, 4), 'select', sfunction_call_result_26484)
    
    # Assigning a Call to a Tuple (line 359):
    
    # Assigning a Subscript to a Name (line 359):
    
    # Obtaining the type of the subscript
    int_26485 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 359, 4), 'int')
    
    # Call to get_lapack_funcs(...): (line 359)
    # Processing the call arguments (line 359)
    
    # Obtaining an instance of the builtin type 'tuple' (line 359)
    tuple_26487 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 359, 31), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 359)
    # Adding element type (line 359)
    str_26488 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 359, 31), 'str', 'tgsen')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 359, 31), tuple_26487, str_26488)
    
    
    # Obtaining an instance of the builtin type 'tuple' (line 359)
    tuple_26489 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 359, 43), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 359)
    # Adding element type (line 359)
    # Getting the type of 'AA' (line 359)
    AA_26490 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 359, 43), 'AA', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 359, 43), tuple_26489, AA_26490)
    # Adding element type (line 359)
    # Getting the type of 'BB' (line 359)
    BB_26491 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 359, 47), 'BB', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 359, 43), tuple_26489, BB_26491)
    
    # Processing the call keyword arguments (line 359)
    kwargs_26492 = {}
    # Getting the type of 'get_lapack_funcs' (line 359)
    get_lapack_funcs_26486 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 359, 13), 'get_lapack_funcs', False)
    # Calling get_lapack_funcs(args, kwargs) (line 359)
    get_lapack_funcs_call_result_26493 = invoke(stypy.reporting.localization.Localization(__file__, 359, 13), get_lapack_funcs_26486, *[tuple_26487, tuple_26489], **kwargs_26492)
    
    # Obtaining the member '__getitem__' of a type (line 359)
    getitem___26494 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 359, 4), get_lapack_funcs_call_result_26493, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 359)
    subscript_call_result_26495 = invoke(stypy.reporting.localization.Localization(__file__, 359, 4), getitem___26494, int_26485)
    
    # Assigning a type to the variable 'tuple_var_assignment_25827' (line 359)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 359, 4), 'tuple_var_assignment_25827', subscript_call_result_26495)
    
    # Assigning a Name to a Name (line 359):
    # Getting the type of 'tuple_var_assignment_25827' (line 359)
    tuple_var_assignment_25827_26496 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 359, 4), 'tuple_var_assignment_25827')
    # Assigning a type to the variable 'tgsen' (line 359)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 359, 4), 'tgsen', tuple_var_assignment_25827_26496)
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'lwork' (line 361)
    lwork_26497 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 361, 7), 'lwork')
    # Getting the type of 'None' (line 361)
    None_26498 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 361, 16), 'None')
    # Applying the binary operator 'is' (line 361)
    result_is__26499 = python_operator(stypy.reporting.localization.Localization(__file__, 361, 7), 'is', lwork_26497, None_26498)
    
    
    # Getting the type of 'lwork' (line 361)
    lwork_26500 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 361, 24), 'lwork')
    int_26501 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 361, 33), 'int')
    # Applying the binary operator '==' (line 361)
    result_eq_26502 = python_operator(stypy.reporting.localization.Localization(__file__, 361, 24), '==', lwork_26500, int_26501)
    
    # Applying the binary operator 'or' (line 361)
    result_or_keyword_26503 = python_operator(stypy.reporting.localization.Localization(__file__, 361, 7), 'or', result_is__26499, result_eq_26502)
    
    # Testing the type of an if condition (line 361)
    if_condition_26504 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 361, 4), result_or_keyword_26503)
    # Assigning a type to the variable 'if_condition_26504' (line 361)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 361, 4), 'if_condition_26504', if_condition_26504)
    # SSA begins for if statement (line 361)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 362):
    
    # Assigning a Call to a Name (line 362):
    
    # Call to tgsen(...): (line 362)
    # Processing the call arguments (line 362)
    # Getting the type of 'select' (line 362)
    select_26506 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 362, 23), 'select', False)
    # Getting the type of 'AA' (line 362)
    AA_26507 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 362, 31), 'AA', False)
    # Getting the type of 'BB' (line 362)
    BB_26508 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 362, 35), 'BB', False)
    # Getting the type of 'Q' (line 362)
    Q_26509 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 362, 39), 'Q', False)
    # Getting the type of 'Z' (line 362)
    Z_26510 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 362, 42), 'Z', False)
    # Processing the call keyword arguments (line 362)
    int_26511 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 362, 51), 'int')
    keyword_26512 = int_26511
    kwargs_26513 = {'lwork': keyword_26512}
    # Getting the type of 'tgsen' (line 362)
    tgsen_26505 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 362, 17), 'tgsen', False)
    # Calling tgsen(args, kwargs) (line 362)
    tgsen_call_result_26514 = invoke(stypy.reporting.localization.Localization(__file__, 362, 17), tgsen_26505, *[select_26506, AA_26507, BB_26508, Q_26509, Z_26510], **kwargs_26513)
    
    # Assigning a type to the variable 'result' (line 362)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 362, 8), 'result', tgsen_call_result_26514)
    
    # Assigning a Call to a Name (line 363):
    
    # Assigning a Call to a Name (line 363):
    
    # Call to astype(...): (line 363)
    # Processing the call arguments (line 363)
    # Getting the type of 'np' (line 363)
    np_26524 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 363, 42), 'np', False)
    # Obtaining the member 'int' of a type (line 363)
    int_26525 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 363, 42), np_26524, 'int')
    # Processing the call keyword arguments (line 363)
    kwargs_26526 = {}
    
    # Obtaining the type of the subscript
    int_26515 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 363, 27), 'int')
    
    # Obtaining the type of the subscript
    int_26516 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 363, 23), 'int')
    # Getting the type of 'result' (line 363)
    result_26517 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 363, 16), 'result', False)
    # Obtaining the member '__getitem__' of a type (line 363)
    getitem___26518 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 363, 16), result_26517, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 363)
    subscript_call_result_26519 = invoke(stypy.reporting.localization.Localization(__file__, 363, 16), getitem___26518, int_26516)
    
    # Obtaining the member '__getitem__' of a type (line 363)
    getitem___26520 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 363, 16), subscript_call_result_26519, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 363)
    subscript_call_result_26521 = invoke(stypy.reporting.localization.Localization(__file__, 363, 16), getitem___26520, int_26515)
    
    # Obtaining the member 'real' of a type (line 363)
    real_26522 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 363, 16), subscript_call_result_26521, 'real')
    # Obtaining the member 'astype' of a type (line 363)
    astype_26523 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 363, 16), real_26522, 'astype')
    # Calling astype(args, kwargs) (line 363)
    astype_call_result_26527 = invoke(stypy.reporting.localization.Localization(__file__, 363, 16), astype_26523, *[int_26525], **kwargs_26526)
    
    # Assigning a type to the variable 'lwork' (line 363)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 363, 8), 'lwork', astype_call_result_26527)
    
    # Getting the type of 'lwork' (line 365)
    lwork_26528 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 365, 8), 'lwork')
    int_26529 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 365, 17), 'int')
    # Applying the binary operator '+=' (line 365)
    result_iadd_26530 = python_operator(stypy.reporting.localization.Localization(__file__, 365, 8), '+=', lwork_26528, int_26529)
    # Assigning a type to the variable 'lwork' (line 365)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 365, 8), 'lwork', result_iadd_26530)
    
    # SSA join for if statement (line 361)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Name to a Name (line 367):
    
    # Assigning a Name to a Name (line 367):
    # Getting the type of 'None' (line 367)
    None_26531 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 367, 13), 'None')
    # Assigning a type to the variable 'liwork' (line 367)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 367, 4), 'liwork', None_26531)
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'liwork' (line 368)
    liwork_26532 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 368, 7), 'liwork')
    # Getting the type of 'None' (line 368)
    None_26533 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 368, 17), 'None')
    # Applying the binary operator 'is' (line 368)
    result_is__26534 = python_operator(stypy.reporting.localization.Localization(__file__, 368, 7), 'is', liwork_26532, None_26533)
    
    
    # Getting the type of 'liwork' (line 368)
    liwork_26535 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 368, 25), 'liwork')
    int_26536 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 368, 35), 'int')
    # Applying the binary operator '==' (line 368)
    result_eq_26537 = python_operator(stypy.reporting.localization.Localization(__file__, 368, 25), '==', liwork_26535, int_26536)
    
    # Applying the binary operator 'or' (line 368)
    result_or_keyword_26538 = python_operator(stypy.reporting.localization.Localization(__file__, 368, 7), 'or', result_is__26534, result_eq_26537)
    
    # Testing the type of an if condition (line 368)
    if_condition_26539 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 368, 4), result_or_keyword_26538)
    # Assigning a type to the variable 'if_condition_26539' (line 368)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 368, 4), 'if_condition_26539', if_condition_26539)
    # SSA begins for if statement (line 368)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 369):
    
    # Assigning a Call to a Name (line 369):
    
    # Call to tgsen(...): (line 369)
    # Processing the call arguments (line 369)
    # Getting the type of 'select' (line 369)
    select_26541 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 369, 23), 'select', False)
    # Getting the type of 'AA' (line 369)
    AA_26542 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 369, 31), 'AA', False)
    # Getting the type of 'BB' (line 369)
    BB_26543 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 369, 35), 'BB', False)
    # Getting the type of 'Q' (line 369)
    Q_26544 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 369, 39), 'Q', False)
    # Getting the type of 'Z' (line 369)
    Z_26545 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 369, 42), 'Z', False)
    # Processing the call keyword arguments (line 369)
    int_26546 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 369, 52), 'int')
    keyword_26547 = int_26546
    kwargs_26548 = {'liwork': keyword_26547}
    # Getting the type of 'tgsen' (line 369)
    tgsen_26540 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 369, 17), 'tgsen', False)
    # Calling tgsen(args, kwargs) (line 369)
    tgsen_call_result_26549 = invoke(stypy.reporting.localization.Localization(__file__, 369, 17), tgsen_26540, *[select_26541, AA_26542, BB_26543, Q_26544, Z_26545], **kwargs_26548)
    
    # Assigning a type to the variable 'result' (line 369)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 369, 8), 'result', tgsen_call_result_26549)
    
    # Assigning a Subscript to a Name (line 370):
    
    # Assigning a Subscript to a Name (line 370):
    
    # Obtaining the type of the subscript
    int_26550 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 370, 28), 'int')
    
    # Obtaining the type of the subscript
    int_26551 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 370, 24), 'int')
    # Getting the type of 'result' (line 370)
    result_26552 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 370, 17), 'result')
    # Obtaining the member '__getitem__' of a type (line 370)
    getitem___26553 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 370, 17), result_26552, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 370)
    subscript_call_result_26554 = invoke(stypy.reporting.localization.Localization(__file__, 370, 17), getitem___26553, int_26551)
    
    # Obtaining the member '__getitem__' of a type (line 370)
    getitem___26555 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 370, 17), subscript_call_result_26554, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 370)
    subscript_call_result_26556 = invoke(stypy.reporting.localization.Localization(__file__, 370, 17), getitem___26555, int_26550)
    
    # Assigning a type to the variable 'liwork' (line 370)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 370, 8), 'liwork', subscript_call_result_26556)
    # SSA join for if statement (line 368)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 372):
    
    # Assigning a Call to a Name (line 372):
    
    # Call to tgsen(...): (line 372)
    # Processing the call arguments (line 372)
    # Getting the type of 'select' (line 372)
    select_26558 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 372, 19), 'select', False)
    # Getting the type of 'AA' (line 372)
    AA_26559 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 372, 27), 'AA', False)
    # Getting the type of 'BB' (line 372)
    BB_26560 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 372, 31), 'BB', False)
    # Getting the type of 'Q' (line 372)
    Q_26561 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 372, 35), 'Q', False)
    # Getting the type of 'Z' (line 372)
    Z_26562 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 372, 38), 'Z', False)
    # Processing the call keyword arguments (line 372)
    # Getting the type of 'lwork' (line 372)
    lwork_26563 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 372, 47), 'lwork', False)
    keyword_26564 = lwork_26563
    # Getting the type of 'liwork' (line 372)
    liwork_26565 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 372, 61), 'liwork', False)
    keyword_26566 = liwork_26565
    kwargs_26567 = {'liwork': keyword_26566, 'lwork': keyword_26564}
    # Getting the type of 'tgsen' (line 372)
    tgsen_26557 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 372, 13), 'tgsen', False)
    # Calling tgsen(args, kwargs) (line 372)
    tgsen_call_result_26568 = invoke(stypy.reporting.localization.Localization(__file__, 372, 13), tgsen_26557, *[select_26558, AA_26559, BB_26560, Q_26561, Z_26562], **kwargs_26567)
    
    # Assigning a type to the variable 'result' (line 372)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 372, 4), 'result', tgsen_call_result_26568)
    
    # Assigning a Subscript to a Name (line 374):
    
    # Assigning a Subscript to a Name (line 374):
    
    # Obtaining the type of the subscript
    int_26569 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 374, 18), 'int')
    # Getting the type of 'result' (line 374)
    result_26570 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 374, 11), 'result')
    # Obtaining the member '__getitem__' of a type (line 374)
    getitem___26571 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 374, 11), result_26570, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 374)
    subscript_call_result_26572 = invoke(stypy.reporting.localization.Localization(__file__, 374, 11), getitem___26571, int_26569)
    
    # Assigning a type to the variable 'info' (line 374)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 374, 4), 'info', subscript_call_result_26572)
    
    
    # Getting the type of 'info' (line 375)
    info_26573 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 375, 7), 'info')
    int_26574 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 375, 14), 'int')
    # Applying the binary operator '<' (line 375)
    result_lt_26575 = python_operator(stypy.reporting.localization.Localization(__file__, 375, 7), '<', info_26573, int_26574)
    
    # Testing the type of an if condition (line 375)
    if_condition_26576 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 375, 4), result_lt_26575)
    # Assigning a type to the variable 'if_condition_26576' (line 375)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 375, 4), 'if_condition_26576', if_condition_26576)
    # SSA begins for if statement (line 375)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 376)
    # Processing the call arguments (line 376)
    str_26578 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 376, 25), 'str', 'Illegal value in argument %d of tgsen')
    
    # Getting the type of 'info' (line 376)
    info_26579 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 376, 68), 'info', False)
    # Applying the 'usub' unary operator (line 376)
    result___neg___26580 = python_operator(stypy.reporting.localization.Localization(__file__, 376, 67), 'usub', info_26579)
    
    # Applying the binary operator '%' (line 376)
    result_mod_26581 = python_operator(stypy.reporting.localization.Localization(__file__, 376, 25), '%', str_26578, result___neg___26580)
    
    # Processing the call keyword arguments (line 376)
    kwargs_26582 = {}
    # Getting the type of 'ValueError' (line 376)
    ValueError_26577 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 376, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 376)
    ValueError_call_result_26583 = invoke(stypy.reporting.localization.Localization(__file__, 376, 14), ValueError_26577, *[result_mod_26581], **kwargs_26582)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 376, 8), ValueError_call_result_26583, 'raise parameter', BaseException)
    # SSA branch for the else part of an if statement (line 375)
    module_type_store.open_ssa_branch('else')
    
    
    # Getting the type of 'info' (line 377)
    info_26584 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 377, 9), 'info')
    int_26585 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 377, 17), 'int')
    # Applying the binary operator '==' (line 377)
    result_eq_26586 = python_operator(stypy.reporting.localization.Localization(__file__, 377, 9), '==', info_26584, int_26585)
    
    # Testing the type of an if condition (line 377)
    if_condition_26587 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 377, 9), result_eq_26586)
    # Assigning a type to the variable 'if_condition_26587' (line 377)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 377, 9), 'if_condition_26587', if_condition_26587)
    # SSA begins for if statement (line 377)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 378)
    # Processing the call arguments (line 378)
    str_26589 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 378, 25), 'str', 'Reordering of (A, B) failed because the transformed matrix pair (A, B) would be too far from generalized Schur form; the problem is very ill-conditioned. (A, B) may have been partially reorded. If requested, 0 is returned in DIF(*), PL, and PR.')
    # Processing the call keyword arguments (line 378)
    kwargs_26590 = {}
    # Getting the type of 'ValueError' (line 378)
    ValueError_26588 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 378, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 378)
    ValueError_call_result_26591 = invoke(stypy.reporting.localization.Localization(__file__, 378, 14), ValueError_26588, *[str_26589], **kwargs_26590)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 378, 8), ValueError_call_result_26591, 'raise parameter', BaseException)
    # SSA join for if statement (line 377)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 375)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'typ' (line 387)
    typ_26592 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 387, 7), 'typ')
    
    # Obtaining an instance of the builtin type 'list' (line 387)
    list_26593 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 387, 14), 'list')
    # Adding type elements to the builtin type 'list' instance (line 387)
    # Adding element type (line 387)
    str_26594 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 387, 15), 'str', 'f')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 387, 14), list_26593, str_26594)
    # Adding element type (line 387)
    str_26595 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 387, 20), 'str', 'd')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 387, 14), list_26593, str_26595)
    
    # Applying the binary operator 'in' (line 387)
    result_contains_26596 = python_operator(stypy.reporting.localization.Localization(__file__, 387, 7), 'in', typ_26592, list_26593)
    
    # Testing the type of an if condition (line 387)
    if_condition_26597 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 387, 4), result_contains_26596)
    # Assigning a type to the variable 'if_condition_26597' (line 387)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 387, 4), 'if_condition_26597', if_condition_26597)
    # SSA begins for if statement (line 387)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a BinOp to a Name (line 388):
    
    # Assigning a BinOp to a Name (line 388):
    
    # Obtaining the type of the subscript
    int_26598 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 388, 23), 'int')
    # Getting the type of 'result' (line 388)
    result_26599 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 388, 16), 'result')
    # Obtaining the member '__getitem__' of a type (line 388)
    getitem___26600 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 388, 16), result_26599, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 388)
    subscript_call_result_26601 = invoke(stypy.reporting.localization.Localization(__file__, 388, 16), getitem___26600, int_26598)
    
    
    # Obtaining the type of the subscript
    int_26602 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 388, 35), 'int')
    # Getting the type of 'result' (line 388)
    result_26603 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 388, 28), 'result')
    # Obtaining the member '__getitem__' of a type (line 388)
    getitem___26604 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 388, 28), result_26603, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 388)
    subscript_call_result_26605 = invoke(stypy.reporting.localization.Localization(__file__, 388, 28), getitem___26604, int_26602)
    
    complex_26606 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 388, 40), 'complex')
    # Applying the binary operator '*' (line 388)
    result_mul_26607 = python_operator(stypy.reporting.localization.Localization(__file__, 388, 28), '*', subscript_call_result_26605, complex_26606)
    
    # Applying the binary operator '+' (line 388)
    result_add_26608 = python_operator(stypy.reporting.localization.Localization(__file__, 388, 16), '+', subscript_call_result_26601, result_mul_26607)
    
    # Assigning a type to the variable 'alpha' (line 388)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 388, 8), 'alpha', result_add_26608)
    
    # Obtaining an instance of the builtin type 'tuple' (line 389)
    tuple_26609 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 389, 16), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 389)
    # Adding element type (line 389)
    
    # Obtaining the type of the subscript
    int_26610 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 389, 23), 'int')
    # Getting the type of 'result' (line 389)
    result_26611 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 389, 16), 'result')
    # Obtaining the member '__getitem__' of a type (line 389)
    getitem___26612 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 389, 16), result_26611, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 389)
    subscript_call_result_26613 = invoke(stypy.reporting.localization.Localization(__file__, 389, 16), getitem___26612, int_26610)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 389, 16), tuple_26609, subscript_call_result_26613)
    # Adding element type (line 389)
    
    # Obtaining the type of the subscript
    int_26614 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 389, 34), 'int')
    # Getting the type of 'result' (line 389)
    result_26615 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 389, 27), 'result')
    # Obtaining the member '__getitem__' of a type (line 389)
    getitem___26616 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 389, 27), result_26615, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 389)
    subscript_call_result_26617 = invoke(stypy.reporting.localization.Localization(__file__, 389, 27), getitem___26616, int_26614)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 389, 16), tuple_26609, subscript_call_result_26617)
    # Adding element type (line 389)
    # Getting the type of 'alpha' (line 389)
    alpha_26618 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 389, 38), 'alpha')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 389, 16), tuple_26609, alpha_26618)
    # Adding element type (line 389)
    
    # Obtaining the type of the subscript
    int_26619 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 389, 52), 'int')
    # Getting the type of 'result' (line 389)
    result_26620 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 389, 45), 'result')
    # Obtaining the member '__getitem__' of a type (line 389)
    getitem___26621 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 389, 45), result_26620, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 389)
    subscript_call_result_26622 = invoke(stypy.reporting.localization.Localization(__file__, 389, 45), getitem___26621, int_26619)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 389, 16), tuple_26609, subscript_call_result_26622)
    # Adding element type (line 389)
    
    # Obtaining the type of the subscript
    int_26623 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 389, 63), 'int')
    # Getting the type of 'result' (line 389)
    result_26624 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 389, 56), 'result')
    # Obtaining the member '__getitem__' of a type (line 389)
    getitem___26625 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 389, 56), result_26624, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 389)
    subscript_call_result_26626 = invoke(stypy.reporting.localization.Localization(__file__, 389, 56), getitem___26625, int_26623)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 389, 16), tuple_26609, subscript_call_result_26626)
    # Adding element type (line 389)
    
    # Obtaining the type of the subscript
    int_26627 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 389, 74), 'int')
    # Getting the type of 'result' (line 389)
    result_26628 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 389, 67), 'result')
    # Obtaining the member '__getitem__' of a type (line 389)
    getitem___26629 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 389, 67), result_26628, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 389)
    subscript_call_result_26630 = invoke(stypy.reporting.localization.Localization(__file__, 389, 67), getitem___26629, int_26627)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 389, 16), tuple_26609, subscript_call_result_26630)
    
    # Assigning a type to the variable 'stypy_return_type' (line 389)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 389, 8), 'stypy_return_type', tuple_26609)
    # SSA branch for the else part of an if statement (line 387)
    module_type_store.open_ssa_branch('else')
    
    # Obtaining an instance of the builtin type 'tuple' (line 393)
    tuple_26631 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 393, 15), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 393)
    # Adding element type (line 393)
    
    # Obtaining the type of the subscript
    int_26632 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 393, 22), 'int')
    # Getting the type of 'result' (line 393)
    result_26633 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 393, 15), 'result')
    # Obtaining the member '__getitem__' of a type (line 393)
    getitem___26634 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 393, 15), result_26633, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 393)
    subscript_call_result_26635 = invoke(stypy.reporting.localization.Localization(__file__, 393, 15), getitem___26634, int_26632)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 393, 15), tuple_26631, subscript_call_result_26635)
    # Adding element type (line 393)
    
    # Obtaining the type of the subscript
    int_26636 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 393, 33), 'int')
    # Getting the type of 'result' (line 393)
    result_26637 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 393, 26), 'result')
    # Obtaining the member '__getitem__' of a type (line 393)
    getitem___26638 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 393, 26), result_26637, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 393)
    subscript_call_result_26639 = invoke(stypy.reporting.localization.Localization(__file__, 393, 26), getitem___26638, int_26636)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 393, 15), tuple_26631, subscript_call_result_26639)
    # Adding element type (line 393)
    
    # Obtaining the type of the subscript
    int_26640 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 393, 44), 'int')
    # Getting the type of 'result' (line 393)
    result_26641 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 393, 37), 'result')
    # Obtaining the member '__getitem__' of a type (line 393)
    getitem___26642 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 393, 37), result_26641, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 393)
    subscript_call_result_26643 = invoke(stypy.reporting.localization.Localization(__file__, 393, 37), getitem___26642, int_26640)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 393, 15), tuple_26631, subscript_call_result_26643)
    # Adding element type (line 393)
    
    # Obtaining the type of the subscript
    int_26644 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 393, 55), 'int')
    # Getting the type of 'result' (line 393)
    result_26645 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 393, 48), 'result')
    # Obtaining the member '__getitem__' of a type (line 393)
    getitem___26646 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 393, 48), result_26645, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 393)
    subscript_call_result_26647 = invoke(stypy.reporting.localization.Localization(__file__, 393, 48), getitem___26646, int_26644)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 393, 15), tuple_26631, subscript_call_result_26647)
    # Adding element type (line 393)
    
    # Obtaining the type of the subscript
    int_26648 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 393, 66), 'int')
    # Getting the type of 'result' (line 393)
    result_26649 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 393, 59), 'result')
    # Obtaining the member '__getitem__' of a type (line 393)
    getitem___26650 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 393, 59), result_26649, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 393)
    subscript_call_result_26651 = invoke(stypy.reporting.localization.Localization(__file__, 393, 59), getitem___26650, int_26648)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 393, 15), tuple_26631, subscript_call_result_26651)
    # Adding element type (line 393)
    
    # Obtaining the type of the subscript
    int_26652 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 393, 77), 'int')
    # Getting the type of 'result' (line 393)
    result_26653 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 393, 70), 'result')
    # Obtaining the member '__getitem__' of a type (line 393)
    getitem___26654 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 393, 70), result_26653, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 393)
    subscript_call_result_26655 = invoke(stypy.reporting.localization.Localization(__file__, 393, 70), getitem___26654, int_26652)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 393, 15), tuple_26631, subscript_call_result_26655)
    
    # Assigning a type to the variable 'stypy_return_type' (line 393)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 393, 8), 'stypy_return_type', tuple_26631)
    # SSA join for if statement (line 387)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'ordqz(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'ordqz' in the type store
    # Getting the type of 'stypy_return_type' (line 267)
    stypy_return_type_26656 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 267, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_26656)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'ordqz'
    return stypy_return_type_26656

# Assigning a type to the variable 'ordqz' (line 267)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 267, 0), 'ordqz', ordqz)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
