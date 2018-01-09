
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: '''Frechet derivative of the matrix exponential.'''
2: from __future__ import division, print_function, absolute_import
3: 
4: import numpy as np
5: import scipy.linalg
6: 
7: __all__ = ['expm_frechet', 'expm_cond']
8: 
9: 
10: def expm_frechet(A, E, method=None, compute_expm=True, check_finite=True):
11:     '''
12:     Frechet derivative of the matrix exponential of A in the direction E.
13: 
14:     Parameters
15:     ----------
16:     A : (N, N) array_like
17:         Matrix of which to take the matrix exponential.
18:     E : (N, N) array_like
19:         Matrix direction in which to take the Frechet derivative.
20:     method : str, optional
21:         Choice of algorithm.  Should be one of
22: 
23:         - `SPS` (default)
24:         - `blockEnlarge`
25: 
26:     compute_expm : bool, optional
27:         Whether to compute also `expm_A` in addition to `expm_frechet_AE`.
28:         Default is True.
29:     check_finite : bool, optional
30:         Whether to check that the input matrix contains only finite numbers.
31:         Disabling may give a performance gain, but may result in problems
32:         (crashes, non-termination) if the inputs do contain infinities or NaNs.
33: 
34:     Returns
35:     -------
36:     expm_A : ndarray
37:         Matrix exponential of A.
38:     expm_frechet_AE : ndarray
39:         Frechet derivative of the matrix exponential of A in the direction E.
40: 
41:     For ``compute_expm = False``, only `expm_frechet_AE` is returned.
42: 
43:     See also
44:     --------
45:     expm : Compute the exponential of a matrix.
46: 
47:     Notes
48:     -----
49:     This section describes the available implementations that can be selected
50:     by the `method` parameter. The default method is *SPS*.
51: 
52:     Method *blockEnlarge* is a naive algorithm.
53: 
54:     Method *SPS* is Scaling-Pade-Squaring [1]_.
55:     It is a sophisticated implementation which should take
56:     only about 3/8 as much time as the naive implementation.
57:     The asymptotics are the same.
58: 
59:     .. versionadded:: 0.13.0
60: 
61:     References
62:     ----------
63:     .. [1] Awad H. Al-Mohy and Nicholas J. Higham (2009)
64:            Computing the Frechet Derivative of the Matrix Exponential,
65:            with an application to Condition Number Estimation.
66:            SIAM Journal On Matrix Analysis and Applications.,
67:            30 (4). pp. 1639-1657. ISSN 1095-7162
68: 
69:     Examples
70:     --------
71:     >>> import scipy.linalg
72:     >>> A = np.random.randn(3, 3)
73:     >>> E = np.random.randn(3, 3)
74:     >>> expm_A, expm_frechet_AE = scipy.linalg.expm_frechet(A, E)
75:     >>> expm_A.shape, expm_frechet_AE.shape
76:     ((3, 3), (3, 3))
77: 
78:     >>> import scipy.linalg
79:     >>> A = np.random.randn(3, 3)
80:     >>> E = np.random.randn(3, 3)
81:     >>> expm_A, expm_frechet_AE = scipy.linalg.expm_frechet(A, E)
82:     >>> M = np.zeros((6, 6))
83:     >>> M[:3, :3] = A; M[:3, 3:] = E; M[3:, 3:] = A
84:     >>> expm_M = scipy.linalg.expm(M)
85:     >>> np.allclose(expm_A, expm_M[:3, :3])
86:     True
87:     >>> np.allclose(expm_frechet_AE, expm_M[:3, 3:])
88:     True
89: 
90:     '''
91:     if check_finite:
92:         A = np.asarray_chkfinite(A)
93:         E = np.asarray_chkfinite(E)
94:     else:
95:         A = np.asarray(A)
96:         E = np.asarray(E)
97:     if A.ndim != 2 or A.shape[0] != A.shape[1]:
98:         raise ValueError('expected A to be a square matrix')
99:     if E.ndim != 2 or E.shape[0] != E.shape[1]:
100:         raise ValueError('expected E to be a square matrix')
101:     if A.shape != E.shape:
102:         raise ValueError('expected A and E to be the same shape')
103:     if method is None:
104:         method = 'SPS'
105:     if method == 'SPS':
106:         expm_A, expm_frechet_AE = expm_frechet_algo_64(A, E)
107:     elif method == 'blockEnlarge':
108:         expm_A, expm_frechet_AE = expm_frechet_block_enlarge(A, E)
109:     else:
110:         raise ValueError('Unknown implementation %s' % method)
111:     if compute_expm:
112:         return expm_A, expm_frechet_AE
113:     else:
114:         return expm_frechet_AE
115: 
116: 
117: def expm_frechet_block_enlarge(A, E):
118:     '''
119:     This is a helper function, mostly for testing and profiling.
120:     Return expm(A), frechet(A, E)
121:     '''
122:     n = A.shape[0]
123:     M = np.vstack([
124:         np.hstack([A, E]),
125:         np.hstack([np.zeros_like(A), A])])
126:     expm_M = scipy.linalg.expm(M)
127:     return expm_M[:n, :n], expm_M[:n, n:]
128: 
129: 
130: '''
131: Maximal values ell_m of ||2**-s A|| such that the backward error bound
132: does not exceed 2**-53.
133: '''
134: ell_table_61 = (
135:         None,
136:         # 1
137:         2.11e-8,
138:         3.56e-4,
139:         1.08e-2,
140:         6.49e-2,
141:         2.00e-1,
142:         4.37e-1,
143:         7.83e-1,
144:         1.23e0,
145:         1.78e0,
146:         2.42e0,
147:         # 11
148:         3.13e0,
149:         3.90e0,
150:         4.74e0,
151:         5.63e0,
152:         6.56e0,
153:         7.52e0,
154:         8.53e0,
155:         9.56e0,
156:         1.06e1,
157:         1.17e1,
158:         )
159: 
160: 
161: # The b vectors and U and V are copypasted
162: # from scipy.sparse.linalg.matfuncs.py.
163: # M, Lu, Lv follow (6.11), (6.12), (6.13), (3.3)
164: 
165: def _diff_pade3(A, E, ident):
166:     b = (120., 60., 12., 1.)
167:     A2 = A.dot(A)
168:     M2 = np.dot(A, E) + np.dot(E, A)
169:     U = A.dot(b[3]*A2 + b[1]*ident)
170:     V = b[2]*A2 + b[0]*ident
171:     Lu = A.dot(b[3]*M2) + E.dot(b[3]*A2 + b[1]*ident)
172:     Lv = b[2]*M2
173:     return U, V, Lu, Lv
174: 
175: 
176: def _diff_pade5(A, E, ident):
177:     b = (30240., 15120., 3360., 420., 30., 1.)
178:     A2 = A.dot(A)
179:     M2 = np.dot(A, E) + np.dot(E, A)
180:     A4 = np.dot(A2, A2)
181:     M4 = np.dot(A2, M2) + np.dot(M2, A2)
182:     U = A.dot(b[5]*A4 + b[3]*A2 + b[1]*ident)
183:     V = b[4]*A4 + b[2]*A2 + b[0]*ident
184:     Lu = (A.dot(b[5]*M4 + b[3]*M2) +
185:             E.dot(b[5]*A4 + b[3]*A2 + b[1]*ident))
186:     Lv = b[4]*M4 + b[2]*M2
187:     return U, V, Lu, Lv
188: 
189: 
190: def _diff_pade7(A, E, ident):
191:     b = (17297280., 8648640., 1995840., 277200., 25200., 1512., 56., 1.)
192:     A2 = A.dot(A)
193:     M2 = np.dot(A, E) + np.dot(E, A)
194:     A4 = np.dot(A2, A2)
195:     M4 = np.dot(A2, M2) + np.dot(M2, A2)
196:     A6 = np.dot(A2, A4)
197:     M6 = np.dot(A4, M2) + np.dot(M4, A2)
198:     U = A.dot(b[7]*A6 + b[5]*A4 + b[3]*A2 + b[1]*ident)
199:     V = b[6]*A6 + b[4]*A4 + b[2]*A2 + b[0]*ident
200:     Lu = (A.dot(b[7]*M6 + b[5]*M4 + b[3]*M2) +
201:             E.dot(b[7]*A6 + b[5]*A4 + b[3]*A2 + b[1]*ident))
202:     Lv = b[6]*M6 + b[4]*M4 + b[2]*M2
203:     return U, V, Lu, Lv
204: 
205: 
206: def _diff_pade9(A, E, ident):
207:     b = (17643225600., 8821612800., 2075673600., 302702400., 30270240.,
208:             2162160., 110880., 3960., 90., 1.)
209:     A2 = A.dot(A)
210:     M2 = np.dot(A, E) + np.dot(E, A)
211:     A4 = np.dot(A2, A2)
212:     M4 = np.dot(A2, M2) + np.dot(M2, A2)
213:     A6 = np.dot(A2, A4)
214:     M6 = np.dot(A4, M2) + np.dot(M4, A2)
215:     A8 = np.dot(A4, A4)
216:     M8 = np.dot(A4, M4) + np.dot(M4, A4)
217:     U = A.dot(b[9]*A8 + b[7]*A6 + b[5]*A4 + b[3]*A2 + b[1]*ident)
218:     V = b[8]*A8 + b[6]*A6 + b[4]*A4 + b[2]*A2 + b[0]*ident
219:     Lu = (A.dot(b[9]*M8 + b[7]*M6 + b[5]*M4 + b[3]*M2) +
220:             E.dot(b[9]*A8 + b[7]*A6 + b[5]*A4 + b[3]*A2 + b[1]*ident))
221:     Lv = b[8]*M8 + b[6]*M6 + b[4]*M4 + b[2]*M2
222:     return U, V, Lu, Lv
223: 
224: 
225: def expm_frechet_algo_64(A, E):
226:     n = A.shape[0]
227:     s = None
228:     ident = np.identity(n)
229:     A_norm_1 = scipy.linalg.norm(A, 1)
230:     m_pade_pairs = (
231:             (3, _diff_pade3),
232:             (5, _diff_pade5),
233:             (7, _diff_pade7),
234:             (9, _diff_pade9))
235:     for m, pade in m_pade_pairs:
236:         if A_norm_1 <= ell_table_61[m]:
237:             U, V, Lu, Lv = pade(A, E, ident)
238:             s = 0
239:             break
240:     if s is None:
241:         # scaling
242:         s = max(0, int(np.ceil(np.log2(A_norm_1 / ell_table_61[13]))))
243:         A = A * 2.0**-s
244:         E = E * 2.0**-s
245:         # pade order 13
246:         A2 = np.dot(A, A)
247:         M2 = np.dot(A, E) + np.dot(E, A)
248:         A4 = np.dot(A2, A2)
249:         M4 = np.dot(A2, M2) + np.dot(M2, A2)
250:         A6 = np.dot(A2, A4)
251:         M6 = np.dot(A4, M2) + np.dot(M4, A2)
252:         b = (64764752532480000., 32382376266240000., 7771770303897600.,
253:                 1187353796428800., 129060195264000., 10559470521600.,
254:                 670442572800., 33522128640., 1323241920., 40840800., 960960.,
255:                 16380., 182., 1.)
256:         W1 = b[13]*A6 + b[11]*A4 + b[9]*A2
257:         W2 = b[7]*A6 + b[5]*A4 + b[3]*A2 + b[1]*ident
258:         Z1 = b[12]*A6 + b[10]*A4 + b[8]*A2
259:         Z2 = b[6]*A6 + b[4]*A4 + b[2]*A2 + b[0]*ident
260:         W = np.dot(A6, W1) + W2
261:         U = np.dot(A, W)
262:         V = np.dot(A6, Z1) + Z2
263:         Lw1 = b[13]*M6 + b[11]*M4 + b[9]*M2
264:         Lw2 = b[7]*M6 + b[5]*M4 + b[3]*M2
265:         Lz1 = b[12]*M6 + b[10]*M4 + b[8]*M2
266:         Lz2 = b[6]*M6 + b[4]*M4 + b[2]*M2
267:         Lw = np.dot(A6, Lw1) + np.dot(M6, W1) + Lw2
268:         Lu = np.dot(A, Lw) + np.dot(E, W)
269:         Lv = np.dot(A6, Lz1) + np.dot(M6, Z1) + Lz2
270:     # factor once and solve twice
271:     lu_piv = scipy.linalg.lu_factor(-U + V)
272:     R = scipy.linalg.lu_solve(lu_piv, U + V)
273:     L = scipy.linalg.lu_solve(lu_piv, Lu + Lv + np.dot((Lu - Lv), R))
274:     # squaring
275:     for k in range(s):
276:         L = np.dot(R, L) + np.dot(L, R)
277:         R = np.dot(R, R)
278:     return R, L
279: 
280: 
281: def vec(M):
282:     '''
283:     Stack columns of M to construct a single vector.
284: 
285:     This is somewhat standard notation in linear algebra.
286: 
287:     Parameters
288:     ----------
289:     M : 2d array_like
290:         Input matrix
291: 
292:     Returns
293:     -------
294:     v : 1d ndarray
295:         Output vector
296: 
297:     '''
298:     return M.T.ravel()
299: 
300: 
301: def expm_frechet_kronform(A, method=None, check_finite=True):
302:     '''
303:     Construct the Kronecker form of the Frechet derivative of expm.
304: 
305:     Parameters
306:     ----------
307:     A : array_like with shape (N, N)
308:         Matrix to be expm'd.
309:     method : str, optional
310:         Extra keyword to be passed to expm_frechet.
311:     check_finite : bool, optional
312:         Whether to check that the input matrix contains only finite numbers.
313:         Disabling may give a performance gain, but may result in problems
314:         (crashes, non-termination) if the inputs do contain infinities or NaNs.
315: 
316:     Returns
317:     -------
318:     K : 2d ndarray with shape (N*N, N*N)
319:         Kronecker form of the Frechet derivative of the matrix exponential.
320: 
321:     Notes
322:     -----
323:     This function is used to help compute the condition number
324:     of the matrix exponential.
325: 
326:     See also
327:     --------
328:     expm : Compute a matrix exponential.
329:     expm_frechet : Compute the Frechet derivative of the matrix exponential.
330:     expm_cond : Compute the relative condition number of the matrix exponential
331:                 in the Frobenius norm.
332: 
333:     '''
334:     if check_finite:
335:         A = np.asarray_chkfinite(A)
336:     else:
337:         A = np.asarray(A)
338:     if len(A.shape) != 2 or A.shape[0] != A.shape[1]:
339:         raise ValueError('expected a square matrix')
340: 
341:     n = A.shape[0]
342:     ident = np.identity(n)
343:     cols = []
344:     for i in range(n):
345:         for j in range(n):
346:             E = np.outer(ident[i], ident[j])
347:             F = expm_frechet(A, E,
348:                     method=method, compute_expm=False, check_finite=False)
349:             cols.append(vec(F))
350:     return np.vstack(cols).T
351: 
352: 
353: def expm_cond(A, check_finite=True):
354:     '''
355:     Relative condition number of the matrix exponential in the Frobenius norm.
356: 
357:     Parameters
358:     ----------
359:     A : 2d array_like
360:         Square input matrix with shape (N, N).
361:     check_finite : bool, optional
362:         Whether to check that the input matrix contains only finite numbers.
363:         Disabling may give a performance gain, but may result in problems
364:         (crashes, non-termination) if the inputs do contain infinities or NaNs.
365: 
366:     Returns
367:     -------
368:     kappa : float
369:         The relative condition number of the matrix exponential
370:         in the Frobenius norm
371: 
372:     Notes
373:     -----
374:     A faster estimate for the condition number in the 1-norm
375:     has been published but is not yet implemented in scipy.
376: 
377:     .. versionadded:: 0.14.0
378: 
379:     See also
380:     --------
381:     expm : Compute the exponential of a matrix.
382:     expm_frechet : Compute the Frechet derivative of the matrix exponential.
383: 
384:     '''
385:     if check_finite:
386:         A = np.asarray_chkfinite(A)
387:     else:
388:         A = np.asarray(A)
389:     if len(A.shape) != 2 or A.shape[0] != A.shape[1]:
390:         raise ValueError('expected a square matrix')
391: 
392:     X = scipy.linalg.expm(A)
393:     K = expm_frechet_kronform(A, check_finite=False)
394: 
395:     # The following norm choices are deliberate.
396:     # The norms of A and X are Frobenius norms,
397:     # and the norm of K is the induced 2-norm.
398:     A_norm = scipy.linalg.norm(A, 'fro')
399:     X_norm = scipy.linalg.norm(X, 'fro')
400:     K_norm = scipy.linalg.norm(K, 2)
401: 
402:     kappa = (K_norm * A_norm) / X_norm
403:     return kappa
404: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_26665 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1, 0), 'str', 'Frechet derivative of the matrix exponential.')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 4, 0))

# 'import numpy' statement (line 4)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/linalg/')
import_26666 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'numpy')

if (type(import_26666) is not StypyTypeError):

    if (import_26666 != 'pyd_module'):
        __import__(import_26666)
        sys_modules_26667 = sys.modules[import_26666]
        import_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'np', sys_modules_26667.module_type_store, module_type_store)
    else:
        import numpy as np

        import_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'np', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 4)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 4, 0), 'numpy', import_26666)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/linalg/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 5, 0))

# 'import scipy.linalg' statement (line 5)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/linalg/')
import_26668 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'scipy.linalg')

if (type(import_26668) is not StypyTypeError):

    if (import_26668 != 'pyd_module'):
        __import__(import_26668)
        sys_modules_26669 = sys.modules[import_26668]
        import_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'scipy.linalg', sys_modules_26669.module_type_store, module_type_store)
    else:
        import scipy.linalg

        import_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'scipy.linalg', scipy.linalg, module_type_store)

else:
    # Assigning a type to the variable 'scipy.linalg' (line 5)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 0), 'scipy.linalg', import_26668)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/linalg/')


# Assigning a List to a Name (line 7):

# Assigning a List to a Name (line 7):
__all__ = ['expm_frechet', 'expm_cond']
module_type_store.set_exportable_members(['expm_frechet', 'expm_cond'])

# Obtaining an instance of the builtin type 'list' (line 7)
list_26670 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 10), 'list')
# Adding type elements to the builtin type 'list' instance (line 7)
# Adding element type (line 7)
str_26671 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 11), 'str', 'expm_frechet')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 7, 10), list_26670, str_26671)
# Adding element type (line 7)
str_26672 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 27), 'str', 'expm_cond')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 7, 10), list_26670, str_26672)

# Assigning a type to the variable '__all__' (line 7)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), '__all__', list_26670)

@norecursion
def expm_frechet(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'None' (line 10)
    None_26673 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 10, 30), 'None')
    # Getting the type of 'True' (line 10)
    True_26674 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 10, 49), 'True')
    # Getting the type of 'True' (line 10)
    True_26675 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 10, 68), 'True')
    defaults = [None_26673, True_26674, True_26675]
    # Create a new context for function 'expm_frechet'
    module_type_store = module_type_store.open_function_context('expm_frechet', 10, 0, False)
    
    # Passed parameters checking function
    expm_frechet.stypy_localization = localization
    expm_frechet.stypy_type_of_self = None
    expm_frechet.stypy_type_store = module_type_store
    expm_frechet.stypy_function_name = 'expm_frechet'
    expm_frechet.stypy_param_names_list = ['A', 'E', 'method', 'compute_expm', 'check_finite']
    expm_frechet.stypy_varargs_param_name = None
    expm_frechet.stypy_kwargs_param_name = None
    expm_frechet.stypy_call_defaults = defaults
    expm_frechet.stypy_call_varargs = varargs
    expm_frechet.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'expm_frechet', ['A', 'E', 'method', 'compute_expm', 'check_finite'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'expm_frechet', localization, ['A', 'E', 'method', 'compute_expm', 'check_finite'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'expm_frechet(...)' code ##################

    str_26676 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 90, (-1)), 'str', '\n    Frechet derivative of the matrix exponential of A in the direction E.\n\n    Parameters\n    ----------\n    A : (N, N) array_like\n        Matrix of which to take the matrix exponential.\n    E : (N, N) array_like\n        Matrix direction in which to take the Frechet derivative.\n    method : str, optional\n        Choice of algorithm.  Should be one of\n\n        - `SPS` (default)\n        - `blockEnlarge`\n\n    compute_expm : bool, optional\n        Whether to compute also `expm_A` in addition to `expm_frechet_AE`.\n        Default is True.\n    check_finite : bool, optional\n        Whether to check that the input matrix contains only finite numbers.\n        Disabling may give a performance gain, but may result in problems\n        (crashes, non-termination) if the inputs do contain infinities or NaNs.\n\n    Returns\n    -------\n    expm_A : ndarray\n        Matrix exponential of A.\n    expm_frechet_AE : ndarray\n        Frechet derivative of the matrix exponential of A in the direction E.\n\n    For ``compute_expm = False``, only `expm_frechet_AE` is returned.\n\n    See also\n    --------\n    expm : Compute the exponential of a matrix.\n\n    Notes\n    -----\n    This section describes the available implementations that can be selected\n    by the `method` parameter. The default method is *SPS*.\n\n    Method *blockEnlarge* is a naive algorithm.\n\n    Method *SPS* is Scaling-Pade-Squaring [1]_.\n    It is a sophisticated implementation which should take\n    only about 3/8 as much time as the naive implementation.\n    The asymptotics are the same.\n\n    .. versionadded:: 0.13.0\n\n    References\n    ----------\n    .. [1] Awad H. Al-Mohy and Nicholas J. Higham (2009)\n           Computing the Frechet Derivative of the Matrix Exponential,\n           with an application to Condition Number Estimation.\n           SIAM Journal On Matrix Analysis and Applications.,\n           30 (4). pp. 1639-1657. ISSN 1095-7162\n\n    Examples\n    --------\n    >>> import scipy.linalg\n    >>> A = np.random.randn(3, 3)\n    >>> E = np.random.randn(3, 3)\n    >>> expm_A, expm_frechet_AE = scipy.linalg.expm_frechet(A, E)\n    >>> expm_A.shape, expm_frechet_AE.shape\n    ((3, 3), (3, 3))\n\n    >>> import scipy.linalg\n    >>> A = np.random.randn(3, 3)\n    >>> E = np.random.randn(3, 3)\n    >>> expm_A, expm_frechet_AE = scipy.linalg.expm_frechet(A, E)\n    >>> M = np.zeros((6, 6))\n    >>> M[:3, :3] = A; M[:3, 3:] = E; M[3:, 3:] = A\n    >>> expm_M = scipy.linalg.expm(M)\n    >>> np.allclose(expm_A, expm_M[:3, :3])\n    True\n    >>> np.allclose(expm_frechet_AE, expm_M[:3, 3:])\n    True\n\n    ')
    
    # Getting the type of 'check_finite' (line 91)
    check_finite_26677 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 7), 'check_finite')
    # Testing the type of an if condition (line 91)
    if_condition_26678 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 91, 4), check_finite_26677)
    # Assigning a type to the variable 'if_condition_26678' (line 91)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 91, 4), 'if_condition_26678', if_condition_26678)
    # SSA begins for if statement (line 91)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 92):
    
    # Assigning a Call to a Name (line 92):
    
    # Call to asarray_chkfinite(...): (line 92)
    # Processing the call arguments (line 92)
    # Getting the type of 'A' (line 92)
    A_26681 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 33), 'A', False)
    # Processing the call keyword arguments (line 92)
    kwargs_26682 = {}
    # Getting the type of 'np' (line 92)
    np_26679 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 12), 'np', False)
    # Obtaining the member 'asarray_chkfinite' of a type (line 92)
    asarray_chkfinite_26680 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 92, 12), np_26679, 'asarray_chkfinite')
    # Calling asarray_chkfinite(args, kwargs) (line 92)
    asarray_chkfinite_call_result_26683 = invoke(stypy.reporting.localization.Localization(__file__, 92, 12), asarray_chkfinite_26680, *[A_26681], **kwargs_26682)
    
    # Assigning a type to the variable 'A' (line 92)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 92, 8), 'A', asarray_chkfinite_call_result_26683)
    
    # Assigning a Call to a Name (line 93):
    
    # Assigning a Call to a Name (line 93):
    
    # Call to asarray_chkfinite(...): (line 93)
    # Processing the call arguments (line 93)
    # Getting the type of 'E' (line 93)
    E_26686 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 33), 'E', False)
    # Processing the call keyword arguments (line 93)
    kwargs_26687 = {}
    # Getting the type of 'np' (line 93)
    np_26684 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 12), 'np', False)
    # Obtaining the member 'asarray_chkfinite' of a type (line 93)
    asarray_chkfinite_26685 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 93, 12), np_26684, 'asarray_chkfinite')
    # Calling asarray_chkfinite(args, kwargs) (line 93)
    asarray_chkfinite_call_result_26688 = invoke(stypy.reporting.localization.Localization(__file__, 93, 12), asarray_chkfinite_26685, *[E_26686], **kwargs_26687)
    
    # Assigning a type to the variable 'E' (line 93)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 93, 8), 'E', asarray_chkfinite_call_result_26688)
    # SSA branch for the else part of an if statement (line 91)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Call to a Name (line 95):
    
    # Assigning a Call to a Name (line 95):
    
    # Call to asarray(...): (line 95)
    # Processing the call arguments (line 95)
    # Getting the type of 'A' (line 95)
    A_26691 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 23), 'A', False)
    # Processing the call keyword arguments (line 95)
    kwargs_26692 = {}
    # Getting the type of 'np' (line 95)
    np_26689 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 12), 'np', False)
    # Obtaining the member 'asarray' of a type (line 95)
    asarray_26690 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 95, 12), np_26689, 'asarray')
    # Calling asarray(args, kwargs) (line 95)
    asarray_call_result_26693 = invoke(stypy.reporting.localization.Localization(__file__, 95, 12), asarray_26690, *[A_26691], **kwargs_26692)
    
    # Assigning a type to the variable 'A' (line 95)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 95, 8), 'A', asarray_call_result_26693)
    
    # Assigning a Call to a Name (line 96):
    
    # Assigning a Call to a Name (line 96):
    
    # Call to asarray(...): (line 96)
    # Processing the call arguments (line 96)
    # Getting the type of 'E' (line 96)
    E_26696 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 23), 'E', False)
    # Processing the call keyword arguments (line 96)
    kwargs_26697 = {}
    # Getting the type of 'np' (line 96)
    np_26694 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 12), 'np', False)
    # Obtaining the member 'asarray' of a type (line 96)
    asarray_26695 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 96, 12), np_26694, 'asarray')
    # Calling asarray(args, kwargs) (line 96)
    asarray_call_result_26698 = invoke(stypy.reporting.localization.Localization(__file__, 96, 12), asarray_26695, *[E_26696], **kwargs_26697)
    
    # Assigning a type to the variable 'E' (line 96)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 96, 8), 'E', asarray_call_result_26698)
    # SSA join for if statement (line 91)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'A' (line 97)
    A_26699 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 7), 'A')
    # Obtaining the member 'ndim' of a type (line 97)
    ndim_26700 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 97, 7), A_26699, 'ndim')
    int_26701 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 97, 17), 'int')
    # Applying the binary operator '!=' (line 97)
    result_ne_26702 = python_operator(stypy.reporting.localization.Localization(__file__, 97, 7), '!=', ndim_26700, int_26701)
    
    
    
    # Obtaining the type of the subscript
    int_26703 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 97, 30), 'int')
    # Getting the type of 'A' (line 97)
    A_26704 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 22), 'A')
    # Obtaining the member 'shape' of a type (line 97)
    shape_26705 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 97, 22), A_26704, 'shape')
    # Obtaining the member '__getitem__' of a type (line 97)
    getitem___26706 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 97, 22), shape_26705, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 97)
    subscript_call_result_26707 = invoke(stypy.reporting.localization.Localization(__file__, 97, 22), getitem___26706, int_26703)
    
    
    # Obtaining the type of the subscript
    int_26708 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 97, 44), 'int')
    # Getting the type of 'A' (line 97)
    A_26709 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 36), 'A')
    # Obtaining the member 'shape' of a type (line 97)
    shape_26710 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 97, 36), A_26709, 'shape')
    # Obtaining the member '__getitem__' of a type (line 97)
    getitem___26711 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 97, 36), shape_26710, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 97)
    subscript_call_result_26712 = invoke(stypy.reporting.localization.Localization(__file__, 97, 36), getitem___26711, int_26708)
    
    # Applying the binary operator '!=' (line 97)
    result_ne_26713 = python_operator(stypy.reporting.localization.Localization(__file__, 97, 22), '!=', subscript_call_result_26707, subscript_call_result_26712)
    
    # Applying the binary operator 'or' (line 97)
    result_or_keyword_26714 = python_operator(stypy.reporting.localization.Localization(__file__, 97, 7), 'or', result_ne_26702, result_ne_26713)
    
    # Testing the type of an if condition (line 97)
    if_condition_26715 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 97, 4), result_or_keyword_26714)
    # Assigning a type to the variable 'if_condition_26715' (line 97)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 97, 4), 'if_condition_26715', if_condition_26715)
    # SSA begins for if statement (line 97)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 98)
    # Processing the call arguments (line 98)
    str_26717 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 98, 25), 'str', 'expected A to be a square matrix')
    # Processing the call keyword arguments (line 98)
    kwargs_26718 = {}
    # Getting the type of 'ValueError' (line 98)
    ValueError_26716 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 98)
    ValueError_call_result_26719 = invoke(stypy.reporting.localization.Localization(__file__, 98, 14), ValueError_26716, *[str_26717], **kwargs_26718)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 98, 8), ValueError_call_result_26719, 'raise parameter', BaseException)
    # SSA join for if statement (line 97)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'E' (line 99)
    E_26720 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 7), 'E')
    # Obtaining the member 'ndim' of a type (line 99)
    ndim_26721 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 99, 7), E_26720, 'ndim')
    int_26722 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 99, 17), 'int')
    # Applying the binary operator '!=' (line 99)
    result_ne_26723 = python_operator(stypy.reporting.localization.Localization(__file__, 99, 7), '!=', ndim_26721, int_26722)
    
    
    
    # Obtaining the type of the subscript
    int_26724 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 99, 30), 'int')
    # Getting the type of 'E' (line 99)
    E_26725 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 22), 'E')
    # Obtaining the member 'shape' of a type (line 99)
    shape_26726 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 99, 22), E_26725, 'shape')
    # Obtaining the member '__getitem__' of a type (line 99)
    getitem___26727 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 99, 22), shape_26726, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 99)
    subscript_call_result_26728 = invoke(stypy.reporting.localization.Localization(__file__, 99, 22), getitem___26727, int_26724)
    
    
    # Obtaining the type of the subscript
    int_26729 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 99, 44), 'int')
    # Getting the type of 'E' (line 99)
    E_26730 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 36), 'E')
    # Obtaining the member 'shape' of a type (line 99)
    shape_26731 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 99, 36), E_26730, 'shape')
    # Obtaining the member '__getitem__' of a type (line 99)
    getitem___26732 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 99, 36), shape_26731, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 99)
    subscript_call_result_26733 = invoke(stypy.reporting.localization.Localization(__file__, 99, 36), getitem___26732, int_26729)
    
    # Applying the binary operator '!=' (line 99)
    result_ne_26734 = python_operator(stypy.reporting.localization.Localization(__file__, 99, 22), '!=', subscript_call_result_26728, subscript_call_result_26733)
    
    # Applying the binary operator 'or' (line 99)
    result_or_keyword_26735 = python_operator(stypy.reporting.localization.Localization(__file__, 99, 7), 'or', result_ne_26723, result_ne_26734)
    
    # Testing the type of an if condition (line 99)
    if_condition_26736 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 99, 4), result_or_keyword_26735)
    # Assigning a type to the variable 'if_condition_26736' (line 99)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 99, 4), 'if_condition_26736', if_condition_26736)
    # SSA begins for if statement (line 99)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 100)
    # Processing the call arguments (line 100)
    str_26738 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 100, 25), 'str', 'expected E to be a square matrix')
    # Processing the call keyword arguments (line 100)
    kwargs_26739 = {}
    # Getting the type of 'ValueError' (line 100)
    ValueError_26737 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 100)
    ValueError_call_result_26740 = invoke(stypy.reporting.localization.Localization(__file__, 100, 14), ValueError_26737, *[str_26738], **kwargs_26739)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 100, 8), ValueError_call_result_26740, 'raise parameter', BaseException)
    # SSA join for if statement (line 99)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'A' (line 101)
    A_26741 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 7), 'A')
    # Obtaining the member 'shape' of a type (line 101)
    shape_26742 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 101, 7), A_26741, 'shape')
    # Getting the type of 'E' (line 101)
    E_26743 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 18), 'E')
    # Obtaining the member 'shape' of a type (line 101)
    shape_26744 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 101, 18), E_26743, 'shape')
    # Applying the binary operator '!=' (line 101)
    result_ne_26745 = python_operator(stypy.reporting.localization.Localization(__file__, 101, 7), '!=', shape_26742, shape_26744)
    
    # Testing the type of an if condition (line 101)
    if_condition_26746 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 101, 4), result_ne_26745)
    # Assigning a type to the variable 'if_condition_26746' (line 101)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 101, 4), 'if_condition_26746', if_condition_26746)
    # SSA begins for if statement (line 101)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 102)
    # Processing the call arguments (line 102)
    str_26748 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 102, 25), 'str', 'expected A and E to be the same shape')
    # Processing the call keyword arguments (line 102)
    kwargs_26749 = {}
    # Getting the type of 'ValueError' (line 102)
    ValueError_26747 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 102)
    ValueError_call_result_26750 = invoke(stypy.reporting.localization.Localization(__file__, 102, 14), ValueError_26747, *[str_26748], **kwargs_26749)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 102, 8), ValueError_call_result_26750, 'raise parameter', BaseException)
    # SSA join for if statement (line 101)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Type idiom detected: calculating its left and rigth part (line 103)
    # Getting the type of 'method' (line 103)
    method_26751 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 7), 'method')
    # Getting the type of 'None' (line 103)
    None_26752 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 17), 'None')
    
    (may_be_26753, more_types_in_union_26754) = may_be_none(method_26751, None_26752)

    if may_be_26753:

        if more_types_in_union_26754:
            # Runtime conditional SSA (line 103)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a Str to a Name (line 104):
        
        # Assigning a Str to a Name (line 104):
        str_26755 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 104, 17), 'str', 'SPS')
        # Assigning a type to the variable 'method' (line 104)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 104, 8), 'method', str_26755)

        if more_types_in_union_26754:
            # SSA join for if statement (line 103)
            module_type_store = module_type_store.join_ssa_context()


    
    
    
    # Getting the type of 'method' (line 105)
    method_26756 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 7), 'method')
    str_26757 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 105, 17), 'str', 'SPS')
    # Applying the binary operator '==' (line 105)
    result_eq_26758 = python_operator(stypy.reporting.localization.Localization(__file__, 105, 7), '==', method_26756, str_26757)
    
    # Testing the type of an if condition (line 105)
    if_condition_26759 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 105, 4), result_eq_26758)
    # Assigning a type to the variable 'if_condition_26759' (line 105)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 105, 4), 'if_condition_26759', if_condition_26759)
    # SSA begins for if statement (line 105)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Tuple (line 106):
    
    # Assigning a Subscript to a Name (line 106):
    
    # Obtaining the type of the subscript
    int_26760 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 106, 8), 'int')
    
    # Call to expm_frechet_algo_64(...): (line 106)
    # Processing the call arguments (line 106)
    # Getting the type of 'A' (line 106)
    A_26762 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 55), 'A', False)
    # Getting the type of 'E' (line 106)
    E_26763 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 58), 'E', False)
    # Processing the call keyword arguments (line 106)
    kwargs_26764 = {}
    # Getting the type of 'expm_frechet_algo_64' (line 106)
    expm_frechet_algo_64_26761 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 34), 'expm_frechet_algo_64', False)
    # Calling expm_frechet_algo_64(args, kwargs) (line 106)
    expm_frechet_algo_64_call_result_26765 = invoke(stypy.reporting.localization.Localization(__file__, 106, 34), expm_frechet_algo_64_26761, *[A_26762, E_26763], **kwargs_26764)
    
    # Obtaining the member '__getitem__' of a type (line 106)
    getitem___26766 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 106, 8), expm_frechet_algo_64_call_result_26765, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 106)
    subscript_call_result_26767 = invoke(stypy.reporting.localization.Localization(__file__, 106, 8), getitem___26766, int_26760)
    
    # Assigning a type to the variable 'tuple_var_assignment_26657' (line 106)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 106, 8), 'tuple_var_assignment_26657', subscript_call_result_26767)
    
    # Assigning a Subscript to a Name (line 106):
    
    # Obtaining the type of the subscript
    int_26768 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 106, 8), 'int')
    
    # Call to expm_frechet_algo_64(...): (line 106)
    # Processing the call arguments (line 106)
    # Getting the type of 'A' (line 106)
    A_26770 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 55), 'A', False)
    # Getting the type of 'E' (line 106)
    E_26771 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 58), 'E', False)
    # Processing the call keyword arguments (line 106)
    kwargs_26772 = {}
    # Getting the type of 'expm_frechet_algo_64' (line 106)
    expm_frechet_algo_64_26769 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 34), 'expm_frechet_algo_64', False)
    # Calling expm_frechet_algo_64(args, kwargs) (line 106)
    expm_frechet_algo_64_call_result_26773 = invoke(stypy.reporting.localization.Localization(__file__, 106, 34), expm_frechet_algo_64_26769, *[A_26770, E_26771], **kwargs_26772)
    
    # Obtaining the member '__getitem__' of a type (line 106)
    getitem___26774 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 106, 8), expm_frechet_algo_64_call_result_26773, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 106)
    subscript_call_result_26775 = invoke(stypy.reporting.localization.Localization(__file__, 106, 8), getitem___26774, int_26768)
    
    # Assigning a type to the variable 'tuple_var_assignment_26658' (line 106)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 106, 8), 'tuple_var_assignment_26658', subscript_call_result_26775)
    
    # Assigning a Name to a Name (line 106):
    # Getting the type of 'tuple_var_assignment_26657' (line 106)
    tuple_var_assignment_26657_26776 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 8), 'tuple_var_assignment_26657')
    # Assigning a type to the variable 'expm_A' (line 106)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 106, 8), 'expm_A', tuple_var_assignment_26657_26776)
    
    # Assigning a Name to a Name (line 106):
    # Getting the type of 'tuple_var_assignment_26658' (line 106)
    tuple_var_assignment_26658_26777 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 8), 'tuple_var_assignment_26658')
    # Assigning a type to the variable 'expm_frechet_AE' (line 106)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 106, 16), 'expm_frechet_AE', tuple_var_assignment_26658_26777)
    # SSA branch for the else part of an if statement (line 105)
    module_type_store.open_ssa_branch('else')
    
    
    # Getting the type of 'method' (line 107)
    method_26778 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 9), 'method')
    str_26779 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 107, 19), 'str', 'blockEnlarge')
    # Applying the binary operator '==' (line 107)
    result_eq_26780 = python_operator(stypy.reporting.localization.Localization(__file__, 107, 9), '==', method_26778, str_26779)
    
    # Testing the type of an if condition (line 107)
    if_condition_26781 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 107, 9), result_eq_26780)
    # Assigning a type to the variable 'if_condition_26781' (line 107)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 107, 9), 'if_condition_26781', if_condition_26781)
    # SSA begins for if statement (line 107)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Tuple (line 108):
    
    # Assigning a Subscript to a Name (line 108):
    
    # Obtaining the type of the subscript
    int_26782 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 108, 8), 'int')
    
    # Call to expm_frechet_block_enlarge(...): (line 108)
    # Processing the call arguments (line 108)
    # Getting the type of 'A' (line 108)
    A_26784 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 61), 'A', False)
    # Getting the type of 'E' (line 108)
    E_26785 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 64), 'E', False)
    # Processing the call keyword arguments (line 108)
    kwargs_26786 = {}
    # Getting the type of 'expm_frechet_block_enlarge' (line 108)
    expm_frechet_block_enlarge_26783 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 34), 'expm_frechet_block_enlarge', False)
    # Calling expm_frechet_block_enlarge(args, kwargs) (line 108)
    expm_frechet_block_enlarge_call_result_26787 = invoke(stypy.reporting.localization.Localization(__file__, 108, 34), expm_frechet_block_enlarge_26783, *[A_26784, E_26785], **kwargs_26786)
    
    # Obtaining the member '__getitem__' of a type (line 108)
    getitem___26788 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 108, 8), expm_frechet_block_enlarge_call_result_26787, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 108)
    subscript_call_result_26789 = invoke(stypy.reporting.localization.Localization(__file__, 108, 8), getitem___26788, int_26782)
    
    # Assigning a type to the variable 'tuple_var_assignment_26659' (line 108)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 108, 8), 'tuple_var_assignment_26659', subscript_call_result_26789)
    
    # Assigning a Subscript to a Name (line 108):
    
    # Obtaining the type of the subscript
    int_26790 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 108, 8), 'int')
    
    # Call to expm_frechet_block_enlarge(...): (line 108)
    # Processing the call arguments (line 108)
    # Getting the type of 'A' (line 108)
    A_26792 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 61), 'A', False)
    # Getting the type of 'E' (line 108)
    E_26793 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 64), 'E', False)
    # Processing the call keyword arguments (line 108)
    kwargs_26794 = {}
    # Getting the type of 'expm_frechet_block_enlarge' (line 108)
    expm_frechet_block_enlarge_26791 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 34), 'expm_frechet_block_enlarge', False)
    # Calling expm_frechet_block_enlarge(args, kwargs) (line 108)
    expm_frechet_block_enlarge_call_result_26795 = invoke(stypy.reporting.localization.Localization(__file__, 108, 34), expm_frechet_block_enlarge_26791, *[A_26792, E_26793], **kwargs_26794)
    
    # Obtaining the member '__getitem__' of a type (line 108)
    getitem___26796 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 108, 8), expm_frechet_block_enlarge_call_result_26795, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 108)
    subscript_call_result_26797 = invoke(stypy.reporting.localization.Localization(__file__, 108, 8), getitem___26796, int_26790)
    
    # Assigning a type to the variable 'tuple_var_assignment_26660' (line 108)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 108, 8), 'tuple_var_assignment_26660', subscript_call_result_26797)
    
    # Assigning a Name to a Name (line 108):
    # Getting the type of 'tuple_var_assignment_26659' (line 108)
    tuple_var_assignment_26659_26798 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 8), 'tuple_var_assignment_26659')
    # Assigning a type to the variable 'expm_A' (line 108)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 108, 8), 'expm_A', tuple_var_assignment_26659_26798)
    
    # Assigning a Name to a Name (line 108):
    # Getting the type of 'tuple_var_assignment_26660' (line 108)
    tuple_var_assignment_26660_26799 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 8), 'tuple_var_assignment_26660')
    # Assigning a type to the variable 'expm_frechet_AE' (line 108)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 108, 16), 'expm_frechet_AE', tuple_var_assignment_26660_26799)
    # SSA branch for the else part of an if statement (line 107)
    module_type_store.open_ssa_branch('else')
    
    # Call to ValueError(...): (line 110)
    # Processing the call arguments (line 110)
    str_26801 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 110, 25), 'str', 'Unknown implementation %s')
    # Getting the type of 'method' (line 110)
    method_26802 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 55), 'method', False)
    # Applying the binary operator '%' (line 110)
    result_mod_26803 = python_operator(stypy.reporting.localization.Localization(__file__, 110, 25), '%', str_26801, method_26802)
    
    # Processing the call keyword arguments (line 110)
    kwargs_26804 = {}
    # Getting the type of 'ValueError' (line 110)
    ValueError_26800 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 110)
    ValueError_call_result_26805 = invoke(stypy.reporting.localization.Localization(__file__, 110, 14), ValueError_26800, *[result_mod_26803], **kwargs_26804)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 110, 8), ValueError_call_result_26805, 'raise parameter', BaseException)
    # SSA join for if statement (line 107)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 105)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Getting the type of 'compute_expm' (line 111)
    compute_expm_26806 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 7), 'compute_expm')
    # Testing the type of an if condition (line 111)
    if_condition_26807 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 111, 4), compute_expm_26806)
    # Assigning a type to the variable 'if_condition_26807' (line 111)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 111, 4), 'if_condition_26807', if_condition_26807)
    # SSA begins for if statement (line 111)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Obtaining an instance of the builtin type 'tuple' (line 112)
    tuple_26808 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 112, 15), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 112)
    # Adding element type (line 112)
    # Getting the type of 'expm_A' (line 112)
    expm_A_26809 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 15), 'expm_A')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 112, 15), tuple_26808, expm_A_26809)
    # Adding element type (line 112)
    # Getting the type of 'expm_frechet_AE' (line 112)
    expm_frechet_AE_26810 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 23), 'expm_frechet_AE')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 112, 15), tuple_26808, expm_frechet_AE_26810)
    
    # Assigning a type to the variable 'stypy_return_type' (line 112)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 112, 8), 'stypy_return_type', tuple_26808)
    # SSA branch for the else part of an if statement (line 111)
    module_type_store.open_ssa_branch('else')
    # Getting the type of 'expm_frechet_AE' (line 114)
    expm_frechet_AE_26811 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 15), 'expm_frechet_AE')
    # Assigning a type to the variable 'stypy_return_type' (line 114)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 114, 8), 'stypy_return_type', expm_frechet_AE_26811)
    # SSA join for if statement (line 111)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'expm_frechet(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'expm_frechet' in the type store
    # Getting the type of 'stypy_return_type' (line 10)
    stypy_return_type_26812 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 10, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_26812)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'expm_frechet'
    return stypy_return_type_26812

# Assigning a type to the variable 'expm_frechet' (line 10)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 0), 'expm_frechet', expm_frechet)

@norecursion
def expm_frechet_block_enlarge(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'expm_frechet_block_enlarge'
    module_type_store = module_type_store.open_function_context('expm_frechet_block_enlarge', 117, 0, False)
    
    # Passed parameters checking function
    expm_frechet_block_enlarge.stypy_localization = localization
    expm_frechet_block_enlarge.stypy_type_of_self = None
    expm_frechet_block_enlarge.stypy_type_store = module_type_store
    expm_frechet_block_enlarge.stypy_function_name = 'expm_frechet_block_enlarge'
    expm_frechet_block_enlarge.stypy_param_names_list = ['A', 'E']
    expm_frechet_block_enlarge.stypy_varargs_param_name = None
    expm_frechet_block_enlarge.stypy_kwargs_param_name = None
    expm_frechet_block_enlarge.stypy_call_defaults = defaults
    expm_frechet_block_enlarge.stypy_call_varargs = varargs
    expm_frechet_block_enlarge.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'expm_frechet_block_enlarge', ['A', 'E'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'expm_frechet_block_enlarge', localization, ['A', 'E'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'expm_frechet_block_enlarge(...)' code ##################

    str_26813 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 121, (-1)), 'str', '\n    This is a helper function, mostly for testing and profiling.\n    Return expm(A), frechet(A, E)\n    ')
    
    # Assigning a Subscript to a Name (line 122):
    
    # Assigning a Subscript to a Name (line 122):
    
    # Obtaining the type of the subscript
    int_26814 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 122, 16), 'int')
    # Getting the type of 'A' (line 122)
    A_26815 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 8), 'A')
    # Obtaining the member 'shape' of a type (line 122)
    shape_26816 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 122, 8), A_26815, 'shape')
    # Obtaining the member '__getitem__' of a type (line 122)
    getitem___26817 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 122, 8), shape_26816, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 122)
    subscript_call_result_26818 = invoke(stypy.reporting.localization.Localization(__file__, 122, 8), getitem___26817, int_26814)
    
    # Assigning a type to the variable 'n' (line 122)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 122, 4), 'n', subscript_call_result_26818)
    
    # Assigning a Call to a Name (line 123):
    
    # Assigning a Call to a Name (line 123):
    
    # Call to vstack(...): (line 123)
    # Processing the call arguments (line 123)
    
    # Obtaining an instance of the builtin type 'list' (line 123)
    list_26821 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 123, 18), 'list')
    # Adding type elements to the builtin type 'list' instance (line 123)
    # Adding element type (line 123)
    
    # Call to hstack(...): (line 124)
    # Processing the call arguments (line 124)
    
    # Obtaining an instance of the builtin type 'list' (line 124)
    list_26824 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 124, 18), 'list')
    # Adding type elements to the builtin type 'list' instance (line 124)
    # Adding element type (line 124)
    # Getting the type of 'A' (line 124)
    A_26825 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 19), 'A', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 124, 18), list_26824, A_26825)
    # Adding element type (line 124)
    # Getting the type of 'E' (line 124)
    E_26826 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 22), 'E', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 124, 18), list_26824, E_26826)
    
    # Processing the call keyword arguments (line 124)
    kwargs_26827 = {}
    # Getting the type of 'np' (line 124)
    np_26822 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 8), 'np', False)
    # Obtaining the member 'hstack' of a type (line 124)
    hstack_26823 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 124, 8), np_26822, 'hstack')
    # Calling hstack(args, kwargs) (line 124)
    hstack_call_result_26828 = invoke(stypy.reporting.localization.Localization(__file__, 124, 8), hstack_26823, *[list_26824], **kwargs_26827)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 123, 18), list_26821, hstack_call_result_26828)
    # Adding element type (line 123)
    
    # Call to hstack(...): (line 125)
    # Processing the call arguments (line 125)
    
    # Obtaining an instance of the builtin type 'list' (line 125)
    list_26831 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 125, 18), 'list')
    # Adding type elements to the builtin type 'list' instance (line 125)
    # Adding element type (line 125)
    
    # Call to zeros_like(...): (line 125)
    # Processing the call arguments (line 125)
    # Getting the type of 'A' (line 125)
    A_26834 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 33), 'A', False)
    # Processing the call keyword arguments (line 125)
    kwargs_26835 = {}
    # Getting the type of 'np' (line 125)
    np_26832 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 19), 'np', False)
    # Obtaining the member 'zeros_like' of a type (line 125)
    zeros_like_26833 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 125, 19), np_26832, 'zeros_like')
    # Calling zeros_like(args, kwargs) (line 125)
    zeros_like_call_result_26836 = invoke(stypy.reporting.localization.Localization(__file__, 125, 19), zeros_like_26833, *[A_26834], **kwargs_26835)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 125, 18), list_26831, zeros_like_call_result_26836)
    # Adding element type (line 125)
    # Getting the type of 'A' (line 125)
    A_26837 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 37), 'A', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 125, 18), list_26831, A_26837)
    
    # Processing the call keyword arguments (line 125)
    kwargs_26838 = {}
    # Getting the type of 'np' (line 125)
    np_26829 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 8), 'np', False)
    # Obtaining the member 'hstack' of a type (line 125)
    hstack_26830 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 125, 8), np_26829, 'hstack')
    # Calling hstack(args, kwargs) (line 125)
    hstack_call_result_26839 = invoke(stypy.reporting.localization.Localization(__file__, 125, 8), hstack_26830, *[list_26831], **kwargs_26838)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 123, 18), list_26821, hstack_call_result_26839)
    
    # Processing the call keyword arguments (line 123)
    kwargs_26840 = {}
    # Getting the type of 'np' (line 123)
    np_26819 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 8), 'np', False)
    # Obtaining the member 'vstack' of a type (line 123)
    vstack_26820 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 123, 8), np_26819, 'vstack')
    # Calling vstack(args, kwargs) (line 123)
    vstack_call_result_26841 = invoke(stypy.reporting.localization.Localization(__file__, 123, 8), vstack_26820, *[list_26821], **kwargs_26840)
    
    # Assigning a type to the variable 'M' (line 123)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 123, 4), 'M', vstack_call_result_26841)
    
    # Assigning a Call to a Name (line 126):
    
    # Assigning a Call to a Name (line 126):
    
    # Call to expm(...): (line 126)
    # Processing the call arguments (line 126)
    # Getting the type of 'M' (line 126)
    M_26845 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 31), 'M', False)
    # Processing the call keyword arguments (line 126)
    kwargs_26846 = {}
    # Getting the type of 'scipy' (line 126)
    scipy_26842 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 13), 'scipy', False)
    # Obtaining the member 'linalg' of a type (line 126)
    linalg_26843 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 126, 13), scipy_26842, 'linalg')
    # Obtaining the member 'expm' of a type (line 126)
    expm_26844 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 126, 13), linalg_26843, 'expm')
    # Calling expm(args, kwargs) (line 126)
    expm_call_result_26847 = invoke(stypy.reporting.localization.Localization(__file__, 126, 13), expm_26844, *[M_26845], **kwargs_26846)
    
    # Assigning a type to the variable 'expm_M' (line 126)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 126, 4), 'expm_M', expm_call_result_26847)
    
    # Obtaining an instance of the builtin type 'tuple' (line 127)
    tuple_26848 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 127, 11), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 127)
    # Adding element type (line 127)
    
    # Obtaining the type of the subscript
    # Getting the type of 'n' (line 127)
    n_26849 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 19), 'n')
    slice_26850 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 127, 11), None, n_26849, None)
    # Getting the type of 'n' (line 127)
    n_26851 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 23), 'n')
    slice_26852 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 127, 11), None, n_26851, None)
    # Getting the type of 'expm_M' (line 127)
    expm_M_26853 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 11), 'expm_M')
    # Obtaining the member '__getitem__' of a type (line 127)
    getitem___26854 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 127, 11), expm_M_26853, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 127)
    subscript_call_result_26855 = invoke(stypy.reporting.localization.Localization(__file__, 127, 11), getitem___26854, (slice_26850, slice_26852))
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 127, 11), tuple_26848, subscript_call_result_26855)
    # Adding element type (line 127)
    
    # Obtaining the type of the subscript
    # Getting the type of 'n' (line 127)
    n_26856 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 35), 'n')
    slice_26857 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 127, 27), None, n_26856, None)
    # Getting the type of 'n' (line 127)
    n_26858 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 38), 'n')
    slice_26859 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 127, 27), n_26858, None, None)
    # Getting the type of 'expm_M' (line 127)
    expm_M_26860 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 27), 'expm_M')
    # Obtaining the member '__getitem__' of a type (line 127)
    getitem___26861 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 127, 27), expm_M_26860, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 127)
    subscript_call_result_26862 = invoke(stypy.reporting.localization.Localization(__file__, 127, 27), getitem___26861, (slice_26857, slice_26859))
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 127, 11), tuple_26848, subscript_call_result_26862)
    
    # Assigning a type to the variable 'stypy_return_type' (line 127)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 127, 4), 'stypy_return_type', tuple_26848)
    
    # ################# End of 'expm_frechet_block_enlarge(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'expm_frechet_block_enlarge' in the type store
    # Getting the type of 'stypy_return_type' (line 117)
    stypy_return_type_26863 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_26863)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'expm_frechet_block_enlarge'
    return stypy_return_type_26863

# Assigning a type to the variable 'expm_frechet_block_enlarge' (line 117)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 117, 0), 'expm_frechet_block_enlarge', expm_frechet_block_enlarge)
str_26864 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 133, (-1)), 'str', '\nMaximal values ell_m of ||2**-s A|| such that the backward error bound\ndoes not exceed 2**-53.\n')

# Assigning a Tuple to a Name (line 134):

# Assigning a Tuple to a Name (line 134):

# Obtaining an instance of the builtin type 'tuple' (line 135)
tuple_26865 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 135, 8), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 135)
# Adding element type (line 135)
# Getting the type of 'None' (line 135)
None_26866 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 8), 'None')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 135, 8), tuple_26865, None_26866)
# Adding element type (line 135)
float_26867 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 137, 8), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 135, 8), tuple_26865, float_26867)
# Adding element type (line 135)
float_26868 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 138, 8), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 135, 8), tuple_26865, float_26868)
# Adding element type (line 135)
float_26869 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 139, 8), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 135, 8), tuple_26865, float_26869)
# Adding element type (line 135)
float_26870 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 140, 8), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 135, 8), tuple_26865, float_26870)
# Adding element type (line 135)
float_26871 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 141, 8), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 135, 8), tuple_26865, float_26871)
# Adding element type (line 135)
float_26872 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 142, 8), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 135, 8), tuple_26865, float_26872)
# Adding element type (line 135)
float_26873 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 143, 8), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 135, 8), tuple_26865, float_26873)
# Adding element type (line 135)
float_26874 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 144, 8), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 135, 8), tuple_26865, float_26874)
# Adding element type (line 135)
float_26875 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 145, 8), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 135, 8), tuple_26865, float_26875)
# Adding element type (line 135)
float_26876 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 146, 8), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 135, 8), tuple_26865, float_26876)
# Adding element type (line 135)
float_26877 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 148, 8), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 135, 8), tuple_26865, float_26877)
# Adding element type (line 135)
float_26878 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 149, 8), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 135, 8), tuple_26865, float_26878)
# Adding element type (line 135)
float_26879 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 150, 8), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 135, 8), tuple_26865, float_26879)
# Adding element type (line 135)
float_26880 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 151, 8), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 135, 8), tuple_26865, float_26880)
# Adding element type (line 135)
float_26881 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 152, 8), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 135, 8), tuple_26865, float_26881)
# Adding element type (line 135)
float_26882 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 153, 8), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 135, 8), tuple_26865, float_26882)
# Adding element type (line 135)
float_26883 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 154, 8), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 135, 8), tuple_26865, float_26883)
# Adding element type (line 135)
float_26884 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 155, 8), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 135, 8), tuple_26865, float_26884)
# Adding element type (line 135)
float_26885 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 156, 8), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 135, 8), tuple_26865, float_26885)
# Adding element type (line 135)
float_26886 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 157, 8), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 135, 8), tuple_26865, float_26886)

# Assigning a type to the variable 'ell_table_61' (line 134)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 134, 0), 'ell_table_61', tuple_26865)

@norecursion
def _diff_pade3(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_diff_pade3'
    module_type_store = module_type_store.open_function_context('_diff_pade3', 165, 0, False)
    
    # Passed parameters checking function
    _diff_pade3.stypy_localization = localization
    _diff_pade3.stypy_type_of_self = None
    _diff_pade3.stypy_type_store = module_type_store
    _diff_pade3.stypy_function_name = '_diff_pade3'
    _diff_pade3.stypy_param_names_list = ['A', 'E', 'ident']
    _diff_pade3.stypy_varargs_param_name = None
    _diff_pade3.stypy_kwargs_param_name = None
    _diff_pade3.stypy_call_defaults = defaults
    _diff_pade3.stypy_call_varargs = varargs
    _diff_pade3.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_diff_pade3', ['A', 'E', 'ident'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_diff_pade3', localization, ['A', 'E', 'ident'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_diff_pade3(...)' code ##################

    
    # Assigning a Tuple to a Name (line 166):
    
    # Assigning a Tuple to a Name (line 166):
    
    # Obtaining an instance of the builtin type 'tuple' (line 166)
    tuple_26887 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 166, 9), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 166)
    # Adding element type (line 166)
    float_26888 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 166, 9), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 166, 9), tuple_26887, float_26888)
    # Adding element type (line 166)
    float_26889 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 166, 15), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 166, 9), tuple_26887, float_26889)
    # Adding element type (line 166)
    float_26890 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 166, 20), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 166, 9), tuple_26887, float_26890)
    # Adding element type (line 166)
    float_26891 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 166, 25), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 166, 9), tuple_26887, float_26891)
    
    # Assigning a type to the variable 'b' (line 166)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 166, 4), 'b', tuple_26887)
    
    # Assigning a Call to a Name (line 167):
    
    # Assigning a Call to a Name (line 167):
    
    # Call to dot(...): (line 167)
    # Processing the call arguments (line 167)
    # Getting the type of 'A' (line 167)
    A_26894 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 15), 'A', False)
    # Processing the call keyword arguments (line 167)
    kwargs_26895 = {}
    # Getting the type of 'A' (line 167)
    A_26892 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 9), 'A', False)
    # Obtaining the member 'dot' of a type (line 167)
    dot_26893 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 167, 9), A_26892, 'dot')
    # Calling dot(args, kwargs) (line 167)
    dot_call_result_26896 = invoke(stypy.reporting.localization.Localization(__file__, 167, 9), dot_26893, *[A_26894], **kwargs_26895)
    
    # Assigning a type to the variable 'A2' (line 167)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 167, 4), 'A2', dot_call_result_26896)
    
    # Assigning a BinOp to a Name (line 168):
    
    # Assigning a BinOp to a Name (line 168):
    
    # Call to dot(...): (line 168)
    # Processing the call arguments (line 168)
    # Getting the type of 'A' (line 168)
    A_26899 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 16), 'A', False)
    # Getting the type of 'E' (line 168)
    E_26900 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 19), 'E', False)
    # Processing the call keyword arguments (line 168)
    kwargs_26901 = {}
    # Getting the type of 'np' (line 168)
    np_26897 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 9), 'np', False)
    # Obtaining the member 'dot' of a type (line 168)
    dot_26898 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 168, 9), np_26897, 'dot')
    # Calling dot(args, kwargs) (line 168)
    dot_call_result_26902 = invoke(stypy.reporting.localization.Localization(__file__, 168, 9), dot_26898, *[A_26899, E_26900], **kwargs_26901)
    
    
    # Call to dot(...): (line 168)
    # Processing the call arguments (line 168)
    # Getting the type of 'E' (line 168)
    E_26905 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 31), 'E', False)
    # Getting the type of 'A' (line 168)
    A_26906 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 34), 'A', False)
    # Processing the call keyword arguments (line 168)
    kwargs_26907 = {}
    # Getting the type of 'np' (line 168)
    np_26903 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 24), 'np', False)
    # Obtaining the member 'dot' of a type (line 168)
    dot_26904 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 168, 24), np_26903, 'dot')
    # Calling dot(args, kwargs) (line 168)
    dot_call_result_26908 = invoke(stypy.reporting.localization.Localization(__file__, 168, 24), dot_26904, *[E_26905, A_26906], **kwargs_26907)
    
    # Applying the binary operator '+' (line 168)
    result_add_26909 = python_operator(stypy.reporting.localization.Localization(__file__, 168, 9), '+', dot_call_result_26902, dot_call_result_26908)
    
    # Assigning a type to the variable 'M2' (line 168)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 168, 4), 'M2', result_add_26909)
    
    # Assigning a Call to a Name (line 169):
    
    # Assigning a Call to a Name (line 169):
    
    # Call to dot(...): (line 169)
    # Processing the call arguments (line 169)
    
    # Obtaining the type of the subscript
    int_26912 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 169, 16), 'int')
    # Getting the type of 'b' (line 169)
    b_26913 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 14), 'b', False)
    # Obtaining the member '__getitem__' of a type (line 169)
    getitem___26914 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 169, 14), b_26913, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 169)
    subscript_call_result_26915 = invoke(stypy.reporting.localization.Localization(__file__, 169, 14), getitem___26914, int_26912)
    
    # Getting the type of 'A2' (line 169)
    A2_26916 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 19), 'A2', False)
    # Applying the binary operator '*' (line 169)
    result_mul_26917 = python_operator(stypy.reporting.localization.Localization(__file__, 169, 14), '*', subscript_call_result_26915, A2_26916)
    
    
    # Obtaining the type of the subscript
    int_26918 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 169, 26), 'int')
    # Getting the type of 'b' (line 169)
    b_26919 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 24), 'b', False)
    # Obtaining the member '__getitem__' of a type (line 169)
    getitem___26920 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 169, 24), b_26919, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 169)
    subscript_call_result_26921 = invoke(stypy.reporting.localization.Localization(__file__, 169, 24), getitem___26920, int_26918)
    
    # Getting the type of 'ident' (line 169)
    ident_26922 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 29), 'ident', False)
    # Applying the binary operator '*' (line 169)
    result_mul_26923 = python_operator(stypy.reporting.localization.Localization(__file__, 169, 24), '*', subscript_call_result_26921, ident_26922)
    
    # Applying the binary operator '+' (line 169)
    result_add_26924 = python_operator(stypy.reporting.localization.Localization(__file__, 169, 14), '+', result_mul_26917, result_mul_26923)
    
    # Processing the call keyword arguments (line 169)
    kwargs_26925 = {}
    # Getting the type of 'A' (line 169)
    A_26910 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 8), 'A', False)
    # Obtaining the member 'dot' of a type (line 169)
    dot_26911 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 169, 8), A_26910, 'dot')
    # Calling dot(args, kwargs) (line 169)
    dot_call_result_26926 = invoke(stypy.reporting.localization.Localization(__file__, 169, 8), dot_26911, *[result_add_26924], **kwargs_26925)
    
    # Assigning a type to the variable 'U' (line 169)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 169, 4), 'U', dot_call_result_26926)
    
    # Assigning a BinOp to a Name (line 170):
    
    # Assigning a BinOp to a Name (line 170):
    
    # Obtaining the type of the subscript
    int_26927 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 170, 10), 'int')
    # Getting the type of 'b' (line 170)
    b_26928 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 8), 'b')
    # Obtaining the member '__getitem__' of a type (line 170)
    getitem___26929 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 170, 8), b_26928, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 170)
    subscript_call_result_26930 = invoke(stypy.reporting.localization.Localization(__file__, 170, 8), getitem___26929, int_26927)
    
    # Getting the type of 'A2' (line 170)
    A2_26931 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 13), 'A2')
    # Applying the binary operator '*' (line 170)
    result_mul_26932 = python_operator(stypy.reporting.localization.Localization(__file__, 170, 8), '*', subscript_call_result_26930, A2_26931)
    
    
    # Obtaining the type of the subscript
    int_26933 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 170, 20), 'int')
    # Getting the type of 'b' (line 170)
    b_26934 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 18), 'b')
    # Obtaining the member '__getitem__' of a type (line 170)
    getitem___26935 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 170, 18), b_26934, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 170)
    subscript_call_result_26936 = invoke(stypy.reporting.localization.Localization(__file__, 170, 18), getitem___26935, int_26933)
    
    # Getting the type of 'ident' (line 170)
    ident_26937 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 23), 'ident')
    # Applying the binary operator '*' (line 170)
    result_mul_26938 = python_operator(stypy.reporting.localization.Localization(__file__, 170, 18), '*', subscript_call_result_26936, ident_26937)
    
    # Applying the binary operator '+' (line 170)
    result_add_26939 = python_operator(stypy.reporting.localization.Localization(__file__, 170, 8), '+', result_mul_26932, result_mul_26938)
    
    # Assigning a type to the variable 'V' (line 170)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 170, 4), 'V', result_add_26939)
    
    # Assigning a BinOp to a Name (line 171):
    
    # Assigning a BinOp to a Name (line 171):
    
    # Call to dot(...): (line 171)
    # Processing the call arguments (line 171)
    
    # Obtaining the type of the subscript
    int_26942 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 171, 17), 'int')
    # Getting the type of 'b' (line 171)
    b_26943 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 15), 'b', False)
    # Obtaining the member '__getitem__' of a type (line 171)
    getitem___26944 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 171, 15), b_26943, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 171)
    subscript_call_result_26945 = invoke(stypy.reporting.localization.Localization(__file__, 171, 15), getitem___26944, int_26942)
    
    # Getting the type of 'M2' (line 171)
    M2_26946 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 20), 'M2', False)
    # Applying the binary operator '*' (line 171)
    result_mul_26947 = python_operator(stypy.reporting.localization.Localization(__file__, 171, 15), '*', subscript_call_result_26945, M2_26946)
    
    # Processing the call keyword arguments (line 171)
    kwargs_26948 = {}
    # Getting the type of 'A' (line 171)
    A_26940 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 9), 'A', False)
    # Obtaining the member 'dot' of a type (line 171)
    dot_26941 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 171, 9), A_26940, 'dot')
    # Calling dot(args, kwargs) (line 171)
    dot_call_result_26949 = invoke(stypy.reporting.localization.Localization(__file__, 171, 9), dot_26941, *[result_mul_26947], **kwargs_26948)
    
    
    # Call to dot(...): (line 171)
    # Processing the call arguments (line 171)
    
    # Obtaining the type of the subscript
    int_26952 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 171, 34), 'int')
    # Getting the type of 'b' (line 171)
    b_26953 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 32), 'b', False)
    # Obtaining the member '__getitem__' of a type (line 171)
    getitem___26954 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 171, 32), b_26953, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 171)
    subscript_call_result_26955 = invoke(stypy.reporting.localization.Localization(__file__, 171, 32), getitem___26954, int_26952)
    
    # Getting the type of 'A2' (line 171)
    A2_26956 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 37), 'A2', False)
    # Applying the binary operator '*' (line 171)
    result_mul_26957 = python_operator(stypy.reporting.localization.Localization(__file__, 171, 32), '*', subscript_call_result_26955, A2_26956)
    
    
    # Obtaining the type of the subscript
    int_26958 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 171, 44), 'int')
    # Getting the type of 'b' (line 171)
    b_26959 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 42), 'b', False)
    # Obtaining the member '__getitem__' of a type (line 171)
    getitem___26960 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 171, 42), b_26959, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 171)
    subscript_call_result_26961 = invoke(stypy.reporting.localization.Localization(__file__, 171, 42), getitem___26960, int_26958)
    
    # Getting the type of 'ident' (line 171)
    ident_26962 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 47), 'ident', False)
    # Applying the binary operator '*' (line 171)
    result_mul_26963 = python_operator(stypy.reporting.localization.Localization(__file__, 171, 42), '*', subscript_call_result_26961, ident_26962)
    
    # Applying the binary operator '+' (line 171)
    result_add_26964 = python_operator(stypy.reporting.localization.Localization(__file__, 171, 32), '+', result_mul_26957, result_mul_26963)
    
    # Processing the call keyword arguments (line 171)
    kwargs_26965 = {}
    # Getting the type of 'E' (line 171)
    E_26950 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 26), 'E', False)
    # Obtaining the member 'dot' of a type (line 171)
    dot_26951 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 171, 26), E_26950, 'dot')
    # Calling dot(args, kwargs) (line 171)
    dot_call_result_26966 = invoke(stypy.reporting.localization.Localization(__file__, 171, 26), dot_26951, *[result_add_26964], **kwargs_26965)
    
    # Applying the binary operator '+' (line 171)
    result_add_26967 = python_operator(stypy.reporting.localization.Localization(__file__, 171, 9), '+', dot_call_result_26949, dot_call_result_26966)
    
    # Assigning a type to the variable 'Lu' (line 171)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 171, 4), 'Lu', result_add_26967)
    
    # Assigning a BinOp to a Name (line 172):
    
    # Assigning a BinOp to a Name (line 172):
    
    # Obtaining the type of the subscript
    int_26968 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 172, 11), 'int')
    # Getting the type of 'b' (line 172)
    b_26969 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 9), 'b')
    # Obtaining the member '__getitem__' of a type (line 172)
    getitem___26970 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 172, 9), b_26969, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 172)
    subscript_call_result_26971 = invoke(stypy.reporting.localization.Localization(__file__, 172, 9), getitem___26970, int_26968)
    
    # Getting the type of 'M2' (line 172)
    M2_26972 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 14), 'M2')
    # Applying the binary operator '*' (line 172)
    result_mul_26973 = python_operator(stypy.reporting.localization.Localization(__file__, 172, 9), '*', subscript_call_result_26971, M2_26972)
    
    # Assigning a type to the variable 'Lv' (line 172)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 172, 4), 'Lv', result_mul_26973)
    
    # Obtaining an instance of the builtin type 'tuple' (line 173)
    tuple_26974 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 173, 11), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 173)
    # Adding element type (line 173)
    # Getting the type of 'U' (line 173)
    U_26975 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 11), 'U')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 173, 11), tuple_26974, U_26975)
    # Adding element type (line 173)
    # Getting the type of 'V' (line 173)
    V_26976 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 14), 'V')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 173, 11), tuple_26974, V_26976)
    # Adding element type (line 173)
    # Getting the type of 'Lu' (line 173)
    Lu_26977 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 17), 'Lu')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 173, 11), tuple_26974, Lu_26977)
    # Adding element type (line 173)
    # Getting the type of 'Lv' (line 173)
    Lv_26978 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 21), 'Lv')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 173, 11), tuple_26974, Lv_26978)
    
    # Assigning a type to the variable 'stypy_return_type' (line 173)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 173, 4), 'stypy_return_type', tuple_26974)
    
    # ################# End of '_diff_pade3(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_diff_pade3' in the type store
    # Getting the type of 'stypy_return_type' (line 165)
    stypy_return_type_26979 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_26979)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_diff_pade3'
    return stypy_return_type_26979

# Assigning a type to the variable '_diff_pade3' (line 165)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 165, 0), '_diff_pade3', _diff_pade3)

@norecursion
def _diff_pade5(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_diff_pade5'
    module_type_store = module_type_store.open_function_context('_diff_pade5', 176, 0, False)
    
    # Passed parameters checking function
    _diff_pade5.stypy_localization = localization
    _diff_pade5.stypy_type_of_self = None
    _diff_pade5.stypy_type_store = module_type_store
    _diff_pade5.stypy_function_name = '_diff_pade5'
    _diff_pade5.stypy_param_names_list = ['A', 'E', 'ident']
    _diff_pade5.stypy_varargs_param_name = None
    _diff_pade5.stypy_kwargs_param_name = None
    _diff_pade5.stypy_call_defaults = defaults
    _diff_pade5.stypy_call_varargs = varargs
    _diff_pade5.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_diff_pade5', ['A', 'E', 'ident'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_diff_pade5', localization, ['A', 'E', 'ident'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_diff_pade5(...)' code ##################

    
    # Assigning a Tuple to a Name (line 177):
    
    # Assigning a Tuple to a Name (line 177):
    
    # Obtaining an instance of the builtin type 'tuple' (line 177)
    tuple_26980 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 177, 9), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 177)
    # Adding element type (line 177)
    float_26981 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 177, 9), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 177, 9), tuple_26980, float_26981)
    # Adding element type (line 177)
    float_26982 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 177, 17), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 177, 9), tuple_26980, float_26982)
    # Adding element type (line 177)
    float_26983 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 177, 25), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 177, 9), tuple_26980, float_26983)
    # Adding element type (line 177)
    float_26984 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 177, 32), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 177, 9), tuple_26980, float_26984)
    # Adding element type (line 177)
    float_26985 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 177, 38), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 177, 9), tuple_26980, float_26985)
    # Adding element type (line 177)
    float_26986 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 177, 43), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 177, 9), tuple_26980, float_26986)
    
    # Assigning a type to the variable 'b' (line 177)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 177, 4), 'b', tuple_26980)
    
    # Assigning a Call to a Name (line 178):
    
    # Assigning a Call to a Name (line 178):
    
    # Call to dot(...): (line 178)
    # Processing the call arguments (line 178)
    # Getting the type of 'A' (line 178)
    A_26989 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 15), 'A', False)
    # Processing the call keyword arguments (line 178)
    kwargs_26990 = {}
    # Getting the type of 'A' (line 178)
    A_26987 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 9), 'A', False)
    # Obtaining the member 'dot' of a type (line 178)
    dot_26988 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 178, 9), A_26987, 'dot')
    # Calling dot(args, kwargs) (line 178)
    dot_call_result_26991 = invoke(stypy.reporting.localization.Localization(__file__, 178, 9), dot_26988, *[A_26989], **kwargs_26990)
    
    # Assigning a type to the variable 'A2' (line 178)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 178, 4), 'A2', dot_call_result_26991)
    
    # Assigning a BinOp to a Name (line 179):
    
    # Assigning a BinOp to a Name (line 179):
    
    # Call to dot(...): (line 179)
    # Processing the call arguments (line 179)
    # Getting the type of 'A' (line 179)
    A_26994 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 16), 'A', False)
    # Getting the type of 'E' (line 179)
    E_26995 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 19), 'E', False)
    # Processing the call keyword arguments (line 179)
    kwargs_26996 = {}
    # Getting the type of 'np' (line 179)
    np_26992 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 9), 'np', False)
    # Obtaining the member 'dot' of a type (line 179)
    dot_26993 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 179, 9), np_26992, 'dot')
    # Calling dot(args, kwargs) (line 179)
    dot_call_result_26997 = invoke(stypy.reporting.localization.Localization(__file__, 179, 9), dot_26993, *[A_26994, E_26995], **kwargs_26996)
    
    
    # Call to dot(...): (line 179)
    # Processing the call arguments (line 179)
    # Getting the type of 'E' (line 179)
    E_27000 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 31), 'E', False)
    # Getting the type of 'A' (line 179)
    A_27001 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 34), 'A', False)
    # Processing the call keyword arguments (line 179)
    kwargs_27002 = {}
    # Getting the type of 'np' (line 179)
    np_26998 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 24), 'np', False)
    # Obtaining the member 'dot' of a type (line 179)
    dot_26999 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 179, 24), np_26998, 'dot')
    # Calling dot(args, kwargs) (line 179)
    dot_call_result_27003 = invoke(stypy.reporting.localization.Localization(__file__, 179, 24), dot_26999, *[E_27000, A_27001], **kwargs_27002)
    
    # Applying the binary operator '+' (line 179)
    result_add_27004 = python_operator(stypy.reporting.localization.Localization(__file__, 179, 9), '+', dot_call_result_26997, dot_call_result_27003)
    
    # Assigning a type to the variable 'M2' (line 179)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 179, 4), 'M2', result_add_27004)
    
    # Assigning a Call to a Name (line 180):
    
    # Assigning a Call to a Name (line 180):
    
    # Call to dot(...): (line 180)
    # Processing the call arguments (line 180)
    # Getting the type of 'A2' (line 180)
    A2_27007 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 16), 'A2', False)
    # Getting the type of 'A2' (line 180)
    A2_27008 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 20), 'A2', False)
    # Processing the call keyword arguments (line 180)
    kwargs_27009 = {}
    # Getting the type of 'np' (line 180)
    np_27005 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 9), 'np', False)
    # Obtaining the member 'dot' of a type (line 180)
    dot_27006 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 180, 9), np_27005, 'dot')
    # Calling dot(args, kwargs) (line 180)
    dot_call_result_27010 = invoke(stypy.reporting.localization.Localization(__file__, 180, 9), dot_27006, *[A2_27007, A2_27008], **kwargs_27009)
    
    # Assigning a type to the variable 'A4' (line 180)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 180, 4), 'A4', dot_call_result_27010)
    
    # Assigning a BinOp to a Name (line 181):
    
    # Assigning a BinOp to a Name (line 181):
    
    # Call to dot(...): (line 181)
    # Processing the call arguments (line 181)
    # Getting the type of 'A2' (line 181)
    A2_27013 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 16), 'A2', False)
    # Getting the type of 'M2' (line 181)
    M2_27014 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 20), 'M2', False)
    # Processing the call keyword arguments (line 181)
    kwargs_27015 = {}
    # Getting the type of 'np' (line 181)
    np_27011 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 9), 'np', False)
    # Obtaining the member 'dot' of a type (line 181)
    dot_27012 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 181, 9), np_27011, 'dot')
    # Calling dot(args, kwargs) (line 181)
    dot_call_result_27016 = invoke(stypy.reporting.localization.Localization(__file__, 181, 9), dot_27012, *[A2_27013, M2_27014], **kwargs_27015)
    
    
    # Call to dot(...): (line 181)
    # Processing the call arguments (line 181)
    # Getting the type of 'M2' (line 181)
    M2_27019 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 33), 'M2', False)
    # Getting the type of 'A2' (line 181)
    A2_27020 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 37), 'A2', False)
    # Processing the call keyword arguments (line 181)
    kwargs_27021 = {}
    # Getting the type of 'np' (line 181)
    np_27017 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 26), 'np', False)
    # Obtaining the member 'dot' of a type (line 181)
    dot_27018 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 181, 26), np_27017, 'dot')
    # Calling dot(args, kwargs) (line 181)
    dot_call_result_27022 = invoke(stypy.reporting.localization.Localization(__file__, 181, 26), dot_27018, *[M2_27019, A2_27020], **kwargs_27021)
    
    # Applying the binary operator '+' (line 181)
    result_add_27023 = python_operator(stypy.reporting.localization.Localization(__file__, 181, 9), '+', dot_call_result_27016, dot_call_result_27022)
    
    # Assigning a type to the variable 'M4' (line 181)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 181, 4), 'M4', result_add_27023)
    
    # Assigning a Call to a Name (line 182):
    
    # Assigning a Call to a Name (line 182):
    
    # Call to dot(...): (line 182)
    # Processing the call arguments (line 182)
    
    # Obtaining the type of the subscript
    int_27026 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 182, 16), 'int')
    # Getting the type of 'b' (line 182)
    b_27027 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 182, 14), 'b', False)
    # Obtaining the member '__getitem__' of a type (line 182)
    getitem___27028 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 182, 14), b_27027, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 182)
    subscript_call_result_27029 = invoke(stypy.reporting.localization.Localization(__file__, 182, 14), getitem___27028, int_27026)
    
    # Getting the type of 'A4' (line 182)
    A4_27030 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 182, 19), 'A4', False)
    # Applying the binary operator '*' (line 182)
    result_mul_27031 = python_operator(stypy.reporting.localization.Localization(__file__, 182, 14), '*', subscript_call_result_27029, A4_27030)
    
    
    # Obtaining the type of the subscript
    int_27032 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 182, 26), 'int')
    # Getting the type of 'b' (line 182)
    b_27033 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 182, 24), 'b', False)
    # Obtaining the member '__getitem__' of a type (line 182)
    getitem___27034 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 182, 24), b_27033, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 182)
    subscript_call_result_27035 = invoke(stypy.reporting.localization.Localization(__file__, 182, 24), getitem___27034, int_27032)
    
    # Getting the type of 'A2' (line 182)
    A2_27036 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 182, 29), 'A2', False)
    # Applying the binary operator '*' (line 182)
    result_mul_27037 = python_operator(stypy.reporting.localization.Localization(__file__, 182, 24), '*', subscript_call_result_27035, A2_27036)
    
    # Applying the binary operator '+' (line 182)
    result_add_27038 = python_operator(stypy.reporting.localization.Localization(__file__, 182, 14), '+', result_mul_27031, result_mul_27037)
    
    
    # Obtaining the type of the subscript
    int_27039 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 182, 36), 'int')
    # Getting the type of 'b' (line 182)
    b_27040 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 182, 34), 'b', False)
    # Obtaining the member '__getitem__' of a type (line 182)
    getitem___27041 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 182, 34), b_27040, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 182)
    subscript_call_result_27042 = invoke(stypy.reporting.localization.Localization(__file__, 182, 34), getitem___27041, int_27039)
    
    # Getting the type of 'ident' (line 182)
    ident_27043 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 182, 39), 'ident', False)
    # Applying the binary operator '*' (line 182)
    result_mul_27044 = python_operator(stypy.reporting.localization.Localization(__file__, 182, 34), '*', subscript_call_result_27042, ident_27043)
    
    # Applying the binary operator '+' (line 182)
    result_add_27045 = python_operator(stypy.reporting.localization.Localization(__file__, 182, 32), '+', result_add_27038, result_mul_27044)
    
    # Processing the call keyword arguments (line 182)
    kwargs_27046 = {}
    # Getting the type of 'A' (line 182)
    A_27024 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 182, 8), 'A', False)
    # Obtaining the member 'dot' of a type (line 182)
    dot_27025 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 182, 8), A_27024, 'dot')
    # Calling dot(args, kwargs) (line 182)
    dot_call_result_27047 = invoke(stypy.reporting.localization.Localization(__file__, 182, 8), dot_27025, *[result_add_27045], **kwargs_27046)
    
    # Assigning a type to the variable 'U' (line 182)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 182, 4), 'U', dot_call_result_27047)
    
    # Assigning a BinOp to a Name (line 183):
    
    # Assigning a BinOp to a Name (line 183):
    
    # Obtaining the type of the subscript
    int_27048 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 183, 10), 'int')
    # Getting the type of 'b' (line 183)
    b_27049 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 8), 'b')
    # Obtaining the member '__getitem__' of a type (line 183)
    getitem___27050 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 183, 8), b_27049, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 183)
    subscript_call_result_27051 = invoke(stypy.reporting.localization.Localization(__file__, 183, 8), getitem___27050, int_27048)
    
    # Getting the type of 'A4' (line 183)
    A4_27052 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 13), 'A4')
    # Applying the binary operator '*' (line 183)
    result_mul_27053 = python_operator(stypy.reporting.localization.Localization(__file__, 183, 8), '*', subscript_call_result_27051, A4_27052)
    
    
    # Obtaining the type of the subscript
    int_27054 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 183, 20), 'int')
    # Getting the type of 'b' (line 183)
    b_27055 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 18), 'b')
    # Obtaining the member '__getitem__' of a type (line 183)
    getitem___27056 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 183, 18), b_27055, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 183)
    subscript_call_result_27057 = invoke(stypy.reporting.localization.Localization(__file__, 183, 18), getitem___27056, int_27054)
    
    # Getting the type of 'A2' (line 183)
    A2_27058 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 23), 'A2')
    # Applying the binary operator '*' (line 183)
    result_mul_27059 = python_operator(stypy.reporting.localization.Localization(__file__, 183, 18), '*', subscript_call_result_27057, A2_27058)
    
    # Applying the binary operator '+' (line 183)
    result_add_27060 = python_operator(stypy.reporting.localization.Localization(__file__, 183, 8), '+', result_mul_27053, result_mul_27059)
    
    
    # Obtaining the type of the subscript
    int_27061 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 183, 30), 'int')
    # Getting the type of 'b' (line 183)
    b_27062 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 28), 'b')
    # Obtaining the member '__getitem__' of a type (line 183)
    getitem___27063 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 183, 28), b_27062, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 183)
    subscript_call_result_27064 = invoke(stypy.reporting.localization.Localization(__file__, 183, 28), getitem___27063, int_27061)
    
    # Getting the type of 'ident' (line 183)
    ident_27065 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 33), 'ident')
    # Applying the binary operator '*' (line 183)
    result_mul_27066 = python_operator(stypy.reporting.localization.Localization(__file__, 183, 28), '*', subscript_call_result_27064, ident_27065)
    
    # Applying the binary operator '+' (line 183)
    result_add_27067 = python_operator(stypy.reporting.localization.Localization(__file__, 183, 26), '+', result_add_27060, result_mul_27066)
    
    # Assigning a type to the variable 'V' (line 183)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 183, 4), 'V', result_add_27067)
    
    # Assigning a BinOp to a Name (line 184):
    
    # Assigning a BinOp to a Name (line 184):
    
    # Call to dot(...): (line 184)
    # Processing the call arguments (line 184)
    
    # Obtaining the type of the subscript
    int_27070 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 184, 18), 'int')
    # Getting the type of 'b' (line 184)
    b_27071 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 16), 'b', False)
    # Obtaining the member '__getitem__' of a type (line 184)
    getitem___27072 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 184, 16), b_27071, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 184)
    subscript_call_result_27073 = invoke(stypy.reporting.localization.Localization(__file__, 184, 16), getitem___27072, int_27070)
    
    # Getting the type of 'M4' (line 184)
    M4_27074 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 21), 'M4', False)
    # Applying the binary operator '*' (line 184)
    result_mul_27075 = python_operator(stypy.reporting.localization.Localization(__file__, 184, 16), '*', subscript_call_result_27073, M4_27074)
    
    
    # Obtaining the type of the subscript
    int_27076 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 184, 28), 'int')
    # Getting the type of 'b' (line 184)
    b_27077 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 26), 'b', False)
    # Obtaining the member '__getitem__' of a type (line 184)
    getitem___27078 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 184, 26), b_27077, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 184)
    subscript_call_result_27079 = invoke(stypy.reporting.localization.Localization(__file__, 184, 26), getitem___27078, int_27076)
    
    # Getting the type of 'M2' (line 184)
    M2_27080 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 31), 'M2', False)
    # Applying the binary operator '*' (line 184)
    result_mul_27081 = python_operator(stypy.reporting.localization.Localization(__file__, 184, 26), '*', subscript_call_result_27079, M2_27080)
    
    # Applying the binary operator '+' (line 184)
    result_add_27082 = python_operator(stypy.reporting.localization.Localization(__file__, 184, 16), '+', result_mul_27075, result_mul_27081)
    
    # Processing the call keyword arguments (line 184)
    kwargs_27083 = {}
    # Getting the type of 'A' (line 184)
    A_27068 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 10), 'A', False)
    # Obtaining the member 'dot' of a type (line 184)
    dot_27069 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 184, 10), A_27068, 'dot')
    # Calling dot(args, kwargs) (line 184)
    dot_call_result_27084 = invoke(stypy.reporting.localization.Localization(__file__, 184, 10), dot_27069, *[result_add_27082], **kwargs_27083)
    
    
    # Call to dot(...): (line 185)
    # Processing the call arguments (line 185)
    
    # Obtaining the type of the subscript
    int_27087 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 185, 20), 'int')
    # Getting the type of 'b' (line 185)
    b_27088 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 18), 'b', False)
    # Obtaining the member '__getitem__' of a type (line 185)
    getitem___27089 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 185, 18), b_27088, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 185)
    subscript_call_result_27090 = invoke(stypy.reporting.localization.Localization(__file__, 185, 18), getitem___27089, int_27087)
    
    # Getting the type of 'A4' (line 185)
    A4_27091 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 23), 'A4', False)
    # Applying the binary operator '*' (line 185)
    result_mul_27092 = python_operator(stypy.reporting.localization.Localization(__file__, 185, 18), '*', subscript_call_result_27090, A4_27091)
    
    
    # Obtaining the type of the subscript
    int_27093 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 185, 30), 'int')
    # Getting the type of 'b' (line 185)
    b_27094 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 28), 'b', False)
    # Obtaining the member '__getitem__' of a type (line 185)
    getitem___27095 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 185, 28), b_27094, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 185)
    subscript_call_result_27096 = invoke(stypy.reporting.localization.Localization(__file__, 185, 28), getitem___27095, int_27093)
    
    # Getting the type of 'A2' (line 185)
    A2_27097 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 33), 'A2', False)
    # Applying the binary operator '*' (line 185)
    result_mul_27098 = python_operator(stypy.reporting.localization.Localization(__file__, 185, 28), '*', subscript_call_result_27096, A2_27097)
    
    # Applying the binary operator '+' (line 185)
    result_add_27099 = python_operator(stypy.reporting.localization.Localization(__file__, 185, 18), '+', result_mul_27092, result_mul_27098)
    
    
    # Obtaining the type of the subscript
    int_27100 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 185, 40), 'int')
    # Getting the type of 'b' (line 185)
    b_27101 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 38), 'b', False)
    # Obtaining the member '__getitem__' of a type (line 185)
    getitem___27102 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 185, 38), b_27101, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 185)
    subscript_call_result_27103 = invoke(stypy.reporting.localization.Localization(__file__, 185, 38), getitem___27102, int_27100)
    
    # Getting the type of 'ident' (line 185)
    ident_27104 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 43), 'ident', False)
    # Applying the binary operator '*' (line 185)
    result_mul_27105 = python_operator(stypy.reporting.localization.Localization(__file__, 185, 38), '*', subscript_call_result_27103, ident_27104)
    
    # Applying the binary operator '+' (line 185)
    result_add_27106 = python_operator(stypy.reporting.localization.Localization(__file__, 185, 36), '+', result_add_27099, result_mul_27105)
    
    # Processing the call keyword arguments (line 185)
    kwargs_27107 = {}
    # Getting the type of 'E' (line 185)
    E_27085 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 12), 'E', False)
    # Obtaining the member 'dot' of a type (line 185)
    dot_27086 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 185, 12), E_27085, 'dot')
    # Calling dot(args, kwargs) (line 185)
    dot_call_result_27108 = invoke(stypy.reporting.localization.Localization(__file__, 185, 12), dot_27086, *[result_add_27106], **kwargs_27107)
    
    # Applying the binary operator '+' (line 184)
    result_add_27109 = python_operator(stypy.reporting.localization.Localization(__file__, 184, 10), '+', dot_call_result_27084, dot_call_result_27108)
    
    # Assigning a type to the variable 'Lu' (line 184)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 184, 4), 'Lu', result_add_27109)
    
    # Assigning a BinOp to a Name (line 186):
    
    # Assigning a BinOp to a Name (line 186):
    
    # Obtaining the type of the subscript
    int_27110 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 186, 11), 'int')
    # Getting the type of 'b' (line 186)
    b_27111 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 9), 'b')
    # Obtaining the member '__getitem__' of a type (line 186)
    getitem___27112 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 186, 9), b_27111, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 186)
    subscript_call_result_27113 = invoke(stypy.reporting.localization.Localization(__file__, 186, 9), getitem___27112, int_27110)
    
    # Getting the type of 'M4' (line 186)
    M4_27114 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 14), 'M4')
    # Applying the binary operator '*' (line 186)
    result_mul_27115 = python_operator(stypy.reporting.localization.Localization(__file__, 186, 9), '*', subscript_call_result_27113, M4_27114)
    
    
    # Obtaining the type of the subscript
    int_27116 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 186, 21), 'int')
    # Getting the type of 'b' (line 186)
    b_27117 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 19), 'b')
    # Obtaining the member '__getitem__' of a type (line 186)
    getitem___27118 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 186, 19), b_27117, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 186)
    subscript_call_result_27119 = invoke(stypy.reporting.localization.Localization(__file__, 186, 19), getitem___27118, int_27116)
    
    # Getting the type of 'M2' (line 186)
    M2_27120 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 24), 'M2')
    # Applying the binary operator '*' (line 186)
    result_mul_27121 = python_operator(stypy.reporting.localization.Localization(__file__, 186, 19), '*', subscript_call_result_27119, M2_27120)
    
    # Applying the binary operator '+' (line 186)
    result_add_27122 = python_operator(stypy.reporting.localization.Localization(__file__, 186, 9), '+', result_mul_27115, result_mul_27121)
    
    # Assigning a type to the variable 'Lv' (line 186)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 186, 4), 'Lv', result_add_27122)
    
    # Obtaining an instance of the builtin type 'tuple' (line 187)
    tuple_27123 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 187, 11), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 187)
    # Adding element type (line 187)
    # Getting the type of 'U' (line 187)
    U_27124 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 11), 'U')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 187, 11), tuple_27123, U_27124)
    # Adding element type (line 187)
    # Getting the type of 'V' (line 187)
    V_27125 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 14), 'V')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 187, 11), tuple_27123, V_27125)
    # Adding element type (line 187)
    # Getting the type of 'Lu' (line 187)
    Lu_27126 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 17), 'Lu')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 187, 11), tuple_27123, Lu_27126)
    # Adding element type (line 187)
    # Getting the type of 'Lv' (line 187)
    Lv_27127 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 21), 'Lv')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 187, 11), tuple_27123, Lv_27127)
    
    # Assigning a type to the variable 'stypy_return_type' (line 187)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 187, 4), 'stypy_return_type', tuple_27123)
    
    # ################# End of '_diff_pade5(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_diff_pade5' in the type store
    # Getting the type of 'stypy_return_type' (line 176)
    stypy_return_type_27128 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_27128)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_diff_pade5'
    return stypy_return_type_27128

# Assigning a type to the variable '_diff_pade5' (line 176)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 176, 0), '_diff_pade5', _diff_pade5)

@norecursion
def _diff_pade7(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_diff_pade7'
    module_type_store = module_type_store.open_function_context('_diff_pade7', 190, 0, False)
    
    # Passed parameters checking function
    _diff_pade7.stypy_localization = localization
    _diff_pade7.stypy_type_of_self = None
    _diff_pade7.stypy_type_store = module_type_store
    _diff_pade7.stypy_function_name = '_diff_pade7'
    _diff_pade7.stypy_param_names_list = ['A', 'E', 'ident']
    _diff_pade7.stypy_varargs_param_name = None
    _diff_pade7.stypy_kwargs_param_name = None
    _diff_pade7.stypy_call_defaults = defaults
    _diff_pade7.stypy_call_varargs = varargs
    _diff_pade7.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_diff_pade7', ['A', 'E', 'ident'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_diff_pade7', localization, ['A', 'E', 'ident'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_diff_pade7(...)' code ##################

    
    # Assigning a Tuple to a Name (line 191):
    
    # Assigning a Tuple to a Name (line 191):
    
    # Obtaining an instance of the builtin type 'tuple' (line 191)
    tuple_27129 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 191, 9), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 191)
    # Adding element type (line 191)
    float_27130 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 191, 9), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 191, 9), tuple_27129, float_27130)
    # Adding element type (line 191)
    float_27131 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 191, 20), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 191, 9), tuple_27129, float_27131)
    # Adding element type (line 191)
    float_27132 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 191, 30), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 191, 9), tuple_27129, float_27132)
    # Adding element type (line 191)
    float_27133 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 191, 40), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 191, 9), tuple_27129, float_27133)
    # Adding element type (line 191)
    float_27134 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 191, 49), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 191, 9), tuple_27129, float_27134)
    # Adding element type (line 191)
    float_27135 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 191, 57), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 191, 9), tuple_27129, float_27135)
    # Adding element type (line 191)
    float_27136 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 191, 64), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 191, 9), tuple_27129, float_27136)
    # Adding element type (line 191)
    float_27137 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 191, 69), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 191, 9), tuple_27129, float_27137)
    
    # Assigning a type to the variable 'b' (line 191)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 191, 4), 'b', tuple_27129)
    
    # Assigning a Call to a Name (line 192):
    
    # Assigning a Call to a Name (line 192):
    
    # Call to dot(...): (line 192)
    # Processing the call arguments (line 192)
    # Getting the type of 'A' (line 192)
    A_27140 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 192, 15), 'A', False)
    # Processing the call keyword arguments (line 192)
    kwargs_27141 = {}
    # Getting the type of 'A' (line 192)
    A_27138 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 192, 9), 'A', False)
    # Obtaining the member 'dot' of a type (line 192)
    dot_27139 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 192, 9), A_27138, 'dot')
    # Calling dot(args, kwargs) (line 192)
    dot_call_result_27142 = invoke(stypy.reporting.localization.Localization(__file__, 192, 9), dot_27139, *[A_27140], **kwargs_27141)
    
    # Assigning a type to the variable 'A2' (line 192)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 192, 4), 'A2', dot_call_result_27142)
    
    # Assigning a BinOp to a Name (line 193):
    
    # Assigning a BinOp to a Name (line 193):
    
    # Call to dot(...): (line 193)
    # Processing the call arguments (line 193)
    # Getting the type of 'A' (line 193)
    A_27145 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 16), 'A', False)
    # Getting the type of 'E' (line 193)
    E_27146 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 19), 'E', False)
    # Processing the call keyword arguments (line 193)
    kwargs_27147 = {}
    # Getting the type of 'np' (line 193)
    np_27143 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 9), 'np', False)
    # Obtaining the member 'dot' of a type (line 193)
    dot_27144 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 193, 9), np_27143, 'dot')
    # Calling dot(args, kwargs) (line 193)
    dot_call_result_27148 = invoke(stypy.reporting.localization.Localization(__file__, 193, 9), dot_27144, *[A_27145, E_27146], **kwargs_27147)
    
    
    # Call to dot(...): (line 193)
    # Processing the call arguments (line 193)
    # Getting the type of 'E' (line 193)
    E_27151 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 31), 'E', False)
    # Getting the type of 'A' (line 193)
    A_27152 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 34), 'A', False)
    # Processing the call keyword arguments (line 193)
    kwargs_27153 = {}
    # Getting the type of 'np' (line 193)
    np_27149 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 24), 'np', False)
    # Obtaining the member 'dot' of a type (line 193)
    dot_27150 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 193, 24), np_27149, 'dot')
    # Calling dot(args, kwargs) (line 193)
    dot_call_result_27154 = invoke(stypy.reporting.localization.Localization(__file__, 193, 24), dot_27150, *[E_27151, A_27152], **kwargs_27153)
    
    # Applying the binary operator '+' (line 193)
    result_add_27155 = python_operator(stypy.reporting.localization.Localization(__file__, 193, 9), '+', dot_call_result_27148, dot_call_result_27154)
    
    # Assigning a type to the variable 'M2' (line 193)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 193, 4), 'M2', result_add_27155)
    
    # Assigning a Call to a Name (line 194):
    
    # Assigning a Call to a Name (line 194):
    
    # Call to dot(...): (line 194)
    # Processing the call arguments (line 194)
    # Getting the type of 'A2' (line 194)
    A2_27158 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 194, 16), 'A2', False)
    # Getting the type of 'A2' (line 194)
    A2_27159 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 194, 20), 'A2', False)
    # Processing the call keyword arguments (line 194)
    kwargs_27160 = {}
    # Getting the type of 'np' (line 194)
    np_27156 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 194, 9), 'np', False)
    # Obtaining the member 'dot' of a type (line 194)
    dot_27157 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 194, 9), np_27156, 'dot')
    # Calling dot(args, kwargs) (line 194)
    dot_call_result_27161 = invoke(stypy.reporting.localization.Localization(__file__, 194, 9), dot_27157, *[A2_27158, A2_27159], **kwargs_27160)
    
    # Assigning a type to the variable 'A4' (line 194)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 194, 4), 'A4', dot_call_result_27161)
    
    # Assigning a BinOp to a Name (line 195):
    
    # Assigning a BinOp to a Name (line 195):
    
    # Call to dot(...): (line 195)
    # Processing the call arguments (line 195)
    # Getting the type of 'A2' (line 195)
    A2_27164 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 195, 16), 'A2', False)
    # Getting the type of 'M2' (line 195)
    M2_27165 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 195, 20), 'M2', False)
    # Processing the call keyword arguments (line 195)
    kwargs_27166 = {}
    # Getting the type of 'np' (line 195)
    np_27162 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 195, 9), 'np', False)
    # Obtaining the member 'dot' of a type (line 195)
    dot_27163 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 195, 9), np_27162, 'dot')
    # Calling dot(args, kwargs) (line 195)
    dot_call_result_27167 = invoke(stypy.reporting.localization.Localization(__file__, 195, 9), dot_27163, *[A2_27164, M2_27165], **kwargs_27166)
    
    
    # Call to dot(...): (line 195)
    # Processing the call arguments (line 195)
    # Getting the type of 'M2' (line 195)
    M2_27170 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 195, 33), 'M2', False)
    # Getting the type of 'A2' (line 195)
    A2_27171 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 195, 37), 'A2', False)
    # Processing the call keyword arguments (line 195)
    kwargs_27172 = {}
    # Getting the type of 'np' (line 195)
    np_27168 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 195, 26), 'np', False)
    # Obtaining the member 'dot' of a type (line 195)
    dot_27169 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 195, 26), np_27168, 'dot')
    # Calling dot(args, kwargs) (line 195)
    dot_call_result_27173 = invoke(stypy.reporting.localization.Localization(__file__, 195, 26), dot_27169, *[M2_27170, A2_27171], **kwargs_27172)
    
    # Applying the binary operator '+' (line 195)
    result_add_27174 = python_operator(stypy.reporting.localization.Localization(__file__, 195, 9), '+', dot_call_result_27167, dot_call_result_27173)
    
    # Assigning a type to the variable 'M4' (line 195)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 195, 4), 'M4', result_add_27174)
    
    # Assigning a Call to a Name (line 196):
    
    # Assigning a Call to a Name (line 196):
    
    # Call to dot(...): (line 196)
    # Processing the call arguments (line 196)
    # Getting the type of 'A2' (line 196)
    A2_27177 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 196, 16), 'A2', False)
    # Getting the type of 'A4' (line 196)
    A4_27178 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 196, 20), 'A4', False)
    # Processing the call keyword arguments (line 196)
    kwargs_27179 = {}
    # Getting the type of 'np' (line 196)
    np_27175 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 196, 9), 'np', False)
    # Obtaining the member 'dot' of a type (line 196)
    dot_27176 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 196, 9), np_27175, 'dot')
    # Calling dot(args, kwargs) (line 196)
    dot_call_result_27180 = invoke(stypy.reporting.localization.Localization(__file__, 196, 9), dot_27176, *[A2_27177, A4_27178], **kwargs_27179)
    
    # Assigning a type to the variable 'A6' (line 196)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 196, 4), 'A6', dot_call_result_27180)
    
    # Assigning a BinOp to a Name (line 197):
    
    # Assigning a BinOp to a Name (line 197):
    
    # Call to dot(...): (line 197)
    # Processing the call arguments (line 197)
    # Getting the type of 'A4' (line 197)
    A4_27183 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 197, 16), 'A4', False)
    # Getting the type of 'M2' (line 197)
    M2_27184 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 197, 20), 'M2', False)
    # Processing the call keyword arguments (line 197)
    kwargs_27185 = {}
    # Getting the type of 'np' (line 197)
    np_27181 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 197, 9), 'np', False)
    # Obtaining the member 'dot' of a type (line 197)
    dot_27182 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 197, 9), np_27181, 'dot')
    # Calling dot(args, kwargs) (line 197)
    dot_call_result_27186 = invoke(stypy.reporting.localization.Localization(__file__, 197, 9), dot_27182, *[A4_27183, M2_27184], **kwargs_27185)
    
    
    # Call to dot(...): (line 197)
    # Processing the call arguments (line 197)
    # Getting the type of 'M4' (line 197)
    M4_27189 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 197, 33), 'M4', False)
    # Getting the type of 'A2' (line 197)
    A2_27190 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 197, 37), 'A2', False)
    # Processing the call keyword arguments (line 197)
    kwargs_27191 = {}
    # Getting the type of 'np' (line 197)
    np_27187 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 197, 26), 'np', False)
    # Obtaining the member 'dot' of a type (line 197)
    dot_27188 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 197, 26), np_27187, 'dot')
    # Calling dot(args, kwargs) (line 197)
    dot_call_result_27192 = invoke(stypy.reporting.localization.Localization(__file__, 197, 26), dot_27188, *[M4_27189, A2_27190], **kwargs_27191)
    
    # Applying the binary operator '+' (line 197)
    result_add_27193 = python_operator(stypy.reporting.localization.Localization(__file__, 197, 9), '+', dot_call_result_27186, dot_call_result_27192)
    
    # Assigning a type to the variable 'M6' (line 197)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 197, 4), 'M6', result_add_27193)
    
    # Assigning a Call to a Name (line 198):
    
    # Assigning a Call to a Name (line 198):
    
    # Call to dot(...): (line 198)
    # Processing the call arguments (line 198)
    
    # Obtaining the type of the subscript
    int_27196 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 198, 16), 'int')
    # Getting the type of 'b' (line 198)
    b_27197 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 14), 'b', False)
    # Obtaining the member '__getitem__' of a type (line 198)
    getitem___27198 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 198, 14), b_27197, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 198)
    subscript_call_result_27199 = invoke(stypy.reporting.localization.Localization(__file__, 198, 14), getitem___27198, int_27196)
    
    # Getting the type of 'A6' (line 198)
    A6_27200 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 19), 'A6', False)
    # Applying the binary operator '*' (line 198)
    result_mul_27201 = python_operator(stypy.reporting.localization.Localization(__file__, 198, 14), '*', subscript_call_result_27199, A6_27200)
    
    
    # Obtaining the type of the subscript
    int_27202 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 198, 26), 'int')
    # Getting the type of 'b' (line 198)
    b_27203 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 24), 'b', False)
    # Obtaining the member '__getitem__' of a type (line 198)
    getitem___27204 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 198, 24), b_27203, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 198)
    subscript_call_result_27205 = invoke(stypy.reporting.localization.Localization(__file__, 198, 24), getitem___27204, int_27202)
    
    # Getting the type of 'A4' (line 198)
    A4_27206 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 29), 'A4', False)
    # Applying the binary operator '*' (line 198)
    result_mul_27207 = python_operator(stypy.reporting.localization.Localization(__file__, 198, 24), '*', subscript_call_result_27205, A4_27206)
    
    # Applying the binary operator '+' (line 198)
    result_add_27208 = python_operator(stypy.reporting.localization.Localization(__file__, 198, 14), '+', result_mul_27201, result_mul_27207)
    
    
    # Obtaining the type of the subscript
    int_27209 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 198, 36), 'int')
    # Getting the type of 'b' (line 198)
    b_27210 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 34), 'b', False)
    # Obtaining the member '__getitem__' of a type (line 198)
    getitem___27211 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 198, 34), b_27210, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 198)
    subscript_call_result_27212 = invoke(stypy.reporting.localization.Localization(__file__, 198, 34), getitem___27211, int_27209)
    
    # Getting the type of 'A2' (line 198)
    A2_27213 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 39), 'A2', False)
    # Applying the binary operator '*' (line 198)
    result_mul_27214 = python_operator(stypy.reporting.localization.Localization(__file__, 198, 34), '*', subscript_call_result_27212, A2_27213)
    
    # Applying the binary operator '+' (line 198)
    result_add_27215 = python_operator(stypy.reporting.localization.Localization(__file__, 198, 32), '+', result_add_27208, result_mul_27214)
    
    
    # Obtaining the type of the subscript
    int_27216 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 198, 46), 'int')
    # Getting the type of 'b' (line 198)
    b_27217 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 44), 'b', False)
    # Obtaining the member '__getitem__' of a type (line 198)
    getitem___27218 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 198, 44), b_27217, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 198)
    subscript_call_result_27219 = invoke(stypy.reporting.localization.Localization(__file__, 198, 44), getitem___27218, int_27216)
    
    # Getting the type of 'ident' (line 198)
    ident_27220 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 49), 'ident', False)
    # Applying the binary operator '*' (line 198)
    result_mul_27221 = python_operator(stypy.reporting.localization.Localization(__file__, 198, 44), '*', subscript_call_result_27219, ident_27220)
    
    # Applying the binary operator '+' (line 198)
    result_add_27222 = python_operator(stypy.reporting.localization.Localization(__file__, 198, 42), '+', result_add_27215, result_mul_27221)
    
    # Processing the call keyword arguments (line 198)
    kwargs_27223 = {}
    # Getting the type of 'A' (line 198)
    A_27194 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 8), 'A', False)
    # Obtaining the member 'dot' of a type (line 198)
    dot_27195 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 198, 8), A_27194, 'dot')
    # Calling dot(args, kwargs) (line 198)
    dot_call_result_27224 = invoke(stypy.reporting.localization.Localization(__file__, 198, 8), dot_27195, *[result_add_27222], **kwargs_27223)
    
    # Assigning a type to the variable 'U' (line 198)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 198, 4), 'U', dot_call_result_27224)
    
    # Assigning a BinOp to a Name (line 199):
    
    # Assigning a BinOp to a Name (line 199):
    
    # Obtaining the type of the subscript
    int_27225 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 199, 10), 'int')
    # Getting the type of 'b' (line 199)
    b_27226 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 8), 'b')
    # Obtaining the member '__getitem__' of a type (line 199)
    getitem___27227 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 199, 8), b_27226, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 199)
    subscript_call_result_27228 = invoke(stypy.reporting.localization.Localization(__file__, 199, 8), getitem___27227, int_27225)
    
    # Getting the type of 'A6' (line 199)
    A6_27229 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 13), 'A6')
    # Applying the binary operator '*' (line 199)
    result_mul_27230 = python_operator(stypy.reporting.localization.Localization(__file__, 199, 8), '*', subscript_call_result_27228, A6_27229)
    
    
    # Obtaining the type of the subscript
    int_27231 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 199, 20), 'int')
    # Getting the type of 'b' (line 199)
    b_27232 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 18), 'b')
    # Obtaining the member '__getitem__' of a type (line 199)
    getitem___27233 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 199, 18), b_27232, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 199)
    subscript_call_result_27234 = invoke(stypy.reporting.localization.Localization(__file__, 199, 18), getitem___27233, int_27231)
    
    # Getting the type of 'A4' (line 199)
    A4_27235 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 23), 'A4')
    # Applying the binary operator '*' (line 199)
    result_mul_27236 = python_operator(stypy.reporting.localization.Localization(__file__, 199, 18), '*', subscript_call_result_27234, A4_27235)
    
    # Applying the binary operator '+' (line 199)
    result_add_27237 = python_operator(stypy.reporting.localization.Localization(__file__, 199, 8), '+', result_mul_27230, result_mul_27236)
    
    
    # Obtaining the type of the subscript
    int_27238 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 199, 30), 'int')
    # Getting the type of 'b' (line 199)
    b_27239 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 28), 'b')
    # Obtaining the member '__getitem__' of a type (line 199)
    getitem___27240 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 199, 28), b_27239, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 199)
    subscript_call_result_27241 = invoke(stypy.reporting.localization.Localization(__file__, 199, 28), getitem___27240, int_27238)
    
    # Getting the type of 'A2' (line 199)
    A2_27242 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 33), 'A2')
    # Applying the binary operator '*' (line 199)
    result_mul_27243 = python_operator(stypy.reporting.localization.Localization(__file__, 199, 28), '*', subscript_call_result_27241, A2_27242)
    
    # Applying the binary operator '+' (line 199)
    result_add_27244 = python_operator(stypy.reporting.localization.Localization(__file__, 199, 26), '+', result_add_27237, result_mul_27243)
    
    
    # Obtaining the type of the subscript
    int_27245 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 199, 40), 'int')
    # Getting the type of 'b' (line 199)
    b_27246 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 38), 'b')
    # Obtaining the member '__getitem__' of a type (line 199)
    getitem___27247 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 199, 38), b_27246, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 199)
    subscript_call_result_27248 = invoke(stypy.reporting.localization.Localization(__file__, 199, 38), getitem___27247, int_27245)
    
    # Getting the type of 'ident' (line 199)
    ident_27249 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 43), 'ident')
    # Applying the binary operator '*' (line 199)
    result_mul_27250 = python_operator(stypy.reporting.localization.Localization(__file__, 199, 38), '*', subscript_call_result_27248, ident_27249)
    
    # Applying the binary operator '+' (line 199)
    result_add_27251 = python_operator(stypy.reporting.localization.Localization(__file__, 199, 36), '+', result_add_27244, result_mul_27250)
    
    # Assigning a type to the variable 'V' (line 199)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 199, 4), 'V', result_add_27251)
    
    # Assigning a BinOp to a Name (line 200):
    
    # Assigning a BinOp to a Name (line 200):
    
    # Call to dot(...): (line 200)
    # Processing the call arguments (line 200)
    
    # Obtaining the type of the subscript
    int_27254 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 200, 18), 'int')
    # Getting the type of 'b' (line 200)
    b_27255 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 16), 'b', False)
    # Obtaining the member '__getitem__' of a type (line 200)
    getitem___27256 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 200, 16), b_27255, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 200)
    subscript_call_result_27257 = invoke(stypy.reporting.localization.Localization(__file__, 200, 16), getitem___27256, int_27254)
    
    # Getting the type of 'M6' (line 200)
    M6_27258 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 21), 'M6', False)
    # Applying the binary operator '*' (line 200)
    result_mul_27259 = python_operator(stypy.reporting.localization.Localization(__file__, 200, 16), '*', subscript_call_result_27257, M6_27258)
    
    
    # Obtaining the type of the subscript
    int_27260 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 200, 28), 'int')
    # Getting the type of 'b' (line 200)
    b_27261 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 26), 'b', False)
    # Obtaining the member '__getitem__' of a type (line 200)
    getitem___27262 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 200, 26), b_27261, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 200)
    subscript_call_result_27263 = invoke(stypy.reporting.localization.Localization(__file__, 200, 26), getitem___27262, int_27260)
    
    # Getting the type of 'M4' (line 200)
    M4_27264 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 31), 'M4', False)
    # Applying the binary operator '*' (line 200)
    result_mul_27265 = python_operator(stypy.reporting.localization.Localization(__file__, 200, 26), '*', subscript_call_result_27263, M4_27264)
    
    # Applying the binary operator '+' (line 200)
    result_add_27266 = python_operator(stypy.reporting.localization.Localization(__file__, 200, 16), '+', result_mul_27259, result_mul_27265)
    
    
    # Obtaining the type of the subscript
    int_27267 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 200, 38), 'int')
    # Getting the type of 'b' (line 200)
    b_27268 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 36), 'b', False)
    # Obtaining the member '__getitem__' of a type (line 200)
    getitem___27269 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 200, 36), b_27268, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 200)
    subscript_call_result_27270 = invoke(stypy.reporting.localization.Localization(__file__, 200, 36), getitem___27269, int_27267)
    
    # Getting the type of 'M2' (line 200)
    M2_27271 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 41), 'M2', False)
    # Applying the binary operator '*' (line 200)
    result_mul_27272 = python_operator(stypy.reporting.localization.Localization(__file__, 200, 36), '*', subscript_call_result_27270, M2_27271)
    
    # Applying the binary operator '+' (line 200)
    result_add_27273 = python_operator(stypy.reporting.localization.Localization(__file__, 200, 34), '+', result_add_27266, result_mul_27272)
    
    # Processing the call keyword arguments (line 200)
    kwargs_27274 = {}
    # Getting the type of 'A' (line 200)
    A_27252 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 10), 'A', False)
    # Obtaining the member 'dot' of a type (line 200)
    dot_27253 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 200, 10), A_27252, 'dot')
    # Calling dot(args, kwargs) (line 200)
    dot_call_result_27275 = invoke(stypy.reporting.localization.Localization(__file__, 200, 10), dot_27253, *[result_add_27273], **kwargs_27274)
    
    
    # Call to dot(...): (line 201)
    # Processing the call arguments (line 201)
    
    # Obtaining the type of the subscript
    int_27278 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 201, 20), 'int')
    # Getting the type of 'b' (line 201)
    b_27279 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 18), 'b', False)
    # Obtaining the member '__getitem__' of a type (line 201)
    getitem___27280 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 201, 18), b_27279, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 201)
    subscript_call_result_27281 = invoke(stypy.reporting.localization.Localization(__file__, 201, 18), getitem___27280, int_27278)
    
    # Getting the type of 'A6' (line 201)
    A6_27282 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 23), 'A6', False)
    # Applying the binary operator '*' (line 201)
    result_mul_27283 = python_operator(stypy.reporting.localization.Localization(__file__, 201, 18), '*', subscript_call_result_27281, A6_27282)
    
    
    # Obtaining the type of the subscript
    int_27284 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 201, 30), 'int')
    # Getting the type of 'b' (line 201)
    b_27285 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 28), 'b', False)
    # Obtaining the member '__getitem__' of a type (line 201)
    getitem___27286 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 201, 28), b_27285, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 201)
    subscript_call_result_27287 = invoke(stypy.reporting.localization.Localization(__file__, 201, 28), getitem___27286, int_27284)
    
    # Getting the type of 'A4' (line 201)
    A4_27288 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 33), 'A4', False)
    # Applying the binary operator '*' (line 201)
    result_mul_27289 = python_operator(stypy.reporting.localization.Localization(__file__, 201, 28), '*', subscript_call_result_27287, A4_27288)
    
    # Applying the binary operator '+' (line 201)
    result_add_27290 = python_operator(stypy.reporting.localization.Localization(__file__, 201, 18), '+', result_mul_27283, result_mul_27289)
    
    
    # Obtaining the type of the subscript
    int_27291 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 201, 40), 'int')
    # Getting the type of 'b' (line 201)
    b_27292 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 38), 'b', False)
    # Obtaining the member '__getitem__' of a type (line 201)
    getitem___27293 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 201, 38), b_27292, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 201)
    subscript_call_result_27294 = invoke(stypy.reporting.localization.Localization(__file__, 201, 38), getitem___27293, int_27291)
    
    # Getting the type of 'A2' (line 201)
    A2_27295 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 43), 'A2', False)
    # Applying the binary operator '*' (line 201)
    result_mul_27296 = python_operator(stypy.reporting.localization.Localization(__file__, 201, 38), '*', subscript_call_result_27294, A2_27295)
    
    # Applying the binary operator '+' (line 201)
    result_add_27297 = python_operator(stypy.reporting.localization.Localization(__file__, 201, 36), '+', result_add_27290, result_mul_27296)
    
    
    # Obtaining the type of the subscript
    int_27298 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 201, 50), 'int')
    # Getting the type of 'b' (line 201)
    b_27299 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 48), 'b', False)
    # Obtaining the member '__getitem__' of a type (line 201)
    getitem___27300 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 201, 48), b_27299, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 201)
    subscript_call_result_27301 = invoke(stypy.reporting.localization.Localization(__file__, 201, 48), getitem___27300, int_27298)
    
    # Getting the type of 'ident' (line 201)
    ident_27302 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 53), 'ident', False)
    # Applying the binary operator '*' (line 201)
    result_mul_27303 = python_operator(stypy.reporting.localization.Localization(__file__, 201, 48), '*', subscript_call_result_27301, ident_27302)
    
    # Applying the binary operator '+' (line 201)
    result_add_27304 = python_operator(stypy.reporting.localization.Localization(__file__, 201, 46), '+', result_add_27297, result_mul_27303)
    
    # Processing the call keyword arguments (line 201)
    kwargs_27305 = {}
    # Getting the type of 'E' (line 201)
    E_27276 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 12), 'E', False)
    # Obtaining the member 'dot' of a type (line 201)
    dot_27277 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 201, 12), E_27276, 'dot')
    # Calling dot(args, kwargs) (line 201)
    dot_call_result_27306 = invoke(stypy.reporting.localization.Localization(__file__, 201, 12), dot_27277, *[result_add_27304], **kwargs_27305)
    
    # Applying the binary operator '+' (line 200)
    result_add_27307 = python_operator(stypy.reporting.localization.Localization(__file__, 200, 10), '+', dot_call_result_27275, dot_call_result_27306)
    
    # Assigning a type to the variable 'Lu' (line 200)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 200, 4), 'Lu', result_add_27307)
    
    # Assigning a BinOp to a Name (line 202):
    
    # Assigning a BinOp to a Name (line 202):
    
    # Obtaining the type of the subscript
    int_27308 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 202, 11), 'int')
    # Getting the type of 'b' (line 202)
    b_27309 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 9), 'b')
    # Obtaining the member '__getitem__' of a type (line 202)
    getitem___27310 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 202, 9), b_27309, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 202)
    subscript_call_result_27311 = invoke(stypy.reporting.localization.Localization(__file__, 202, 9), getitem___27310, int_27308)
    
    # Getting the type of 'M6' (line 202)
    M6_27312 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 14), 'M6')
    # Applying the binary operator '*' (line 202)
    result_mul_27313 = python_operator(stypy.reporting.localization.Localization(__file__, 202, 9), '*', subscript_call_result_27311, M6_27312)
    
    
    # Obtaining the type of the subscript
    int_27314 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 202, 21), 'int')
    # Getting the type of 'b' (line 202)
    b_27315 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 19), 'b')
    # Obtaining the member '__getitem__' of a type (line 202)
    getitem___27316 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 202, 19), b_27315, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 202)
    subscript_call_result_27317 = invoke(stypy.reporting.localization.Localization(__file__, 202, 19), getitem___27316, int_27314)
    
    # Getting the type of 'M4' (line 202)
    M4_27318 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 24), 'M4')
    # Applying the binary operator '*' (line 202)
    result_mul_27319 = python_operator(stypy.reporting.localization.Localization(__file__, 202, 19), '*', subscript_call_result_27317, M4_27318)
    
    # Applying the binary operator '+' (line 202)
    result_add_27320 = python_operator(stypy.reporting.localization.Localization(__file__, 202, 9), '+', result_mul_27313, result_mul_27319)
    
    
    # Obtaining the type of the subscript
    int_27321 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 202, 31), 'int')
    # Getting the type of 'b' (line 202)
    b_27322 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 29), 'b')
    # Obtaining the member '__getitem__' of a type (line 202)
    getitem___27323 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 202, 29), b_27322, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 202)
    subscript_call_result_27324 = invoke(stypy.reporting.localization.Localization(__file__, 202, 29), getitem___27323, int_27321)
    
    # Getting the type of 'M2' (line 202)
    M2_27325 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 34), 'M2')
    # Applying the binary operator '*' (line 202)
    result_mul_27326 = python_operator(stypy.reporting.localization.Localization(__file__, 202, 29), '*', subscript_call_result_27324, M2_27325)
    
    # Applying the binary operator '+' (line 202)
    result_add_27327 = python_operator(stypy.reporting.localization.Localization(__file__, 202, 27), '+', result_add_27320, result_mul_27326)
    
    # Assigning a type to the variable 'Lv' (line 202)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 202, 4), 'Lv', result_add_27327)
    
    # Obtaining an instance of the builtin type 'tuple' (line 203)
    tuple_27328 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 203, 11), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 203)
    # Adding element type (line 203)
    # Getting the type of 'U' (line 203)
    U_27329 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 11), 'U')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 203, 11), tuple_27328, U_27329)
    # Adding element type (line 203)
    # Getting the type of 'V' (line 203)
    V_27330 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 14), 'V')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 203, 11), tuple_27328, V_27330)
    # Adding element type (line 203)
    # Getting the type of 'Lu' (line 203)
    Lu_27331 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 17), 'Lu')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 203, 11), tuple_27328, Lu_27331)
    # Adding element type (line 203)
    # Getting the type of 'Lv' (line 203)
    Lv_27332 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 21), 'Lv')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 203, 11), tuple_27328, Lv_27332)
    
    # Assigning a type to the variable 'stypy_return_type' (line 203)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 203, 4), 'stypy_return_type', tuple_27328)
    
    # ################# End of '_diff_pade7(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_diff_pade7' in the type store
    # Getting the type of 'stypy_return_type' (line 190)
    stypy_return_type_27333 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 190, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_27333)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_diff_pade7'
    return stypy_return_type_27333

# Assigning a type to the variable '_diff_pade7' (line 190)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 190, 0), '_diff_pade7', _diff_pade7)

@norecursion
def _diff_pade9(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_diff_pade9'
    module_type_store = module_type_store.open_function_context('_diff_pade9', 206, 0, False)
    
    # Passed parameters checking function
    _diff_pade9.stypy_localization = localization
    _diff_pade9.stypy_type_of_self = None
    _diff_pade9.stypy_type_store = module_type_store
    _diff_pade9.stypy_function_name = '_diff_pade9'
    _diff_pade9.stypy_param_names_list = ['A', 'E', 'ident']
    _diff_pade9.stypy_varargs_param_name = None
    _diff_pade9.stypy_kwargs_param_name = None
    _diff_pade9.stypy_call_defaults = defaults
    _diff_pade9.stypy_call_varargs = varargs
    _diff_pade9.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_diff_pade9', ['A', 'E', 'ident'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_diff_pade9', localization, ['A', 'E', 'ident'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_diff_pade9(...)' code ##################

    
    # Assigning a Tuple to a Name (line 207):
    
    # Assigning a Tuple to a Name (line 207):
    
    # Obtaining an instance of the builtin type 'tuple' (line 207)
    tuple_27334 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 207, 9), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 207)
    # Adding element type (line 207)
    float_27335 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 207, 9), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 207, 9), tuple_27334, float_27335)
    # Adding element type (line 207)
    float_27336 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 207, 23), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 207, 9), tuple_27334, float_27336)
    # Adding element type (line 207)
    float_27337 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 207, 36), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 207, 9), tuple_27334, float_27337)
    # Adding element type (line 207)
    float_27338 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 207, 49), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 207, 9), tuple_27334, float_27338)
    # Adding element type (line 207)
    float_27339 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 207, 61), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 207, 9), tuple_27334, float_27339)
    # Adding element type (line 207)
    float_27340 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 208, 12), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 207, 9), tuple_27334, float_27340)
    # Adding element type (line 207)
    float_27341 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 208, 22), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 207, 9), tuple_27334, float_27341)
    # Adding element type (line 207)
    float_27342 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 208, 31), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 207, 9), tuple_27334, float_27342)
    # Adding element type (line 207)
    float_27343 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 208, 38), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 207, 9), tuple_27334, float_27343)
    # Adding element type (line 207)
    float_27344 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 208, 43), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 207, 9), tuple_27334, float_27344)
    
    # Assigning a type to the variable 'b' (line 207)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 207, 4), 'b', tuple_27334)
    
    # Assigning a Call to a Name (line 209):
    
    # Assigning a Call to a Name (line 209):
    
    # Call to dot(...): (line 209)
    # Processing the call arguments (line 209)
    # Getting the type of 'A' (line 209)
    A_27347 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 15), 'A', False)
    # Processing the call keyword arguments (line 209)
    kwargs_27348 = {}
    # Getting the type of 'A' (line 209)
    A_27345 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 9), 'A', False)
    # Obtaining the member 'dot' of a type (line 209)
    dot_27346 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 209, 9), A_27345, 'dot')
    # Calling dot(args, kwargs) (line 209)
    dot_call_result_27349 = invoke(stypy.reporting.localization.Localization(__file__, 209, 9), dot_27346, *[A_27347], **kwargs_27348)
    
    # Assigning a type to the variable 'A2' (line 209)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 209, 4), 'A2', dot_call_result_27349)
    
    # Assigning a BinOp to a Name (line 210):
    
    # Assigning a BinOp to a Name (line 210):
    
    # Call to dot(...): (line 210)
    # Processing the call arguments (line 210)
    # Getting the type of 'A' (line 210)
    A_27352 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 16), 'A', False)
    # Getting the type of 'E' (line 210)
    E_27353 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 19), 'E', False)
    # Processing the call keyword arguments (line 210)
    kwargs_27354 = {}
    # Getting the type of 'np' (line 210)
    np_27350 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 9), 'np', False)
    # Obtaining the member 'dot' of a type (line 210)
    dot_27351 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 210, 9), np_27350, 'dot')
    # Calling dot(args, kwargs) (line 210)
    dot_call_result_27355 = invoke(stypy.reporting.localization.Localization(__file__, 210, 9), dot_27351, *[A_27352, E_27353], **kwargs_27354)
    
    
    # Call to dot(...): (line 210)
    # Processing the call arguments (line 210)
    # Getting the type of 'E' (line 210)
    E_27358 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 31), 'E', False)
    # Getting the type of 'A' (line 210)
    A_27359 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 34), 'A', False)
    # Processing the call keyword arguments (line 210)
    kwargs_27360 = {}
    # Getting the type of 'np' (line 210)
    np_27356 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 24), 'np', False)
    # Obtaining the member 'dot' of a type (line 210)
    dot_27357 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 210, 24), np_27356, 'dot')
    # Calling dot(args, kwargs) (line 210)
    dot_call_result_27361 = invoke(stypy.reporting.localization.Localization(__file__, 210, 24), dot_27357, *[E_27358, A_27359], **kwargs_27360)
    
    # Applying the binary operator '+' (line 210)
    result_add_27362 = python_operator(stypy.reporting.localization.Localization(__file__, 210, 9), '+', dot_call_result_27355, dot_call_result_27361)
    
    # Assigning a type to the variable 'M2' (line 210)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 210, 4), 'M2', result_add_27362)
    
    # Assigning a Call to a Name (line 211):
    
    # Assigning a Call to a Name (line 211):
    
    # Call to dot(...): (line 211)
    # Processing the call arguments (line 211)
    # Getting the type of 'A2' (line 211)
    A2_27365 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 211, 16), 'A2', False)
    # Getting the type of 'A2' (line 211)
    A2_27366 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 211, 20), 'A2', False)
    # Processing the call keyword arguments (line 211)
    kwargs_27367 = {}
    # Getting the type of 'np' (line 211)
    np_27363 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 211, 9), 'np', False)
    # Obtaining the member 'dot' of a type (line 211)
    dot_27364 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 211, 9), np_27363, 'dot')
    # Calling dot(args, kwargs) (line 211)
    dot_call_result_27368 = invoke(stypy.reporting.localization.Localization(__file__, 211, 9), dot_27364, *[A2_27365, A2_27366], **kwargs_27367)
    
    # Assigning a type to the variable 'A4' (line 211)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 211, 4), 'A4', dot_call_result_27368)
    
    # Assigning a BinOp to a Name (line 212):
    
    # Assigning a BinOp to a Name (line 212):
    
    # Call to dot(...): (line 212)
    # Processing the call arguments (line 212)
    # Getting the type of 'A2' (line 212)
    A2_27371 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 16), 'A2', False)
    # Getting the type of 'M2' (line 212)
    M2_27372 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 20), 'M2', False)
    # Processing the call keyword arguments (line 212)
    kwargs_27373 = {}
    # Getting the type of 'np' (line 212)
    np_27369 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 9), 'np', False)
    # Obtaining the member 'dot' of a type (line 212)
    dot_27370 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 212, 9), np_27369, 'dot')
    # Calling dot(args, kwargs) (line 212)
    dot_call_result_27374 = invoke(stypy.reporting.localization.Localization(__file__, 212, 9), dot_27370, *[A2_27371, M2_27372], **kwargs_27373)
    
    
    # Call to dot(...): (line 212)
    # Processing the call arguments (line 212)
    # Getting the type of 'M2' (line 212)
    M2_27377 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 33), 'M2', False)
    # Getting the type of 'A2' (line 212)
    A2_27378 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 37), 'A2', False)
    # Processing the call keyword arguments (line 212)
    kwargs_27379 = {}
    # Getting the type of 'np' (line 212)
    np_27375 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 26), 'np', False)
    # Obtaining the member 'dot' of a type (line 212)
    dot_27376 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 212, 26), np_27375, 'dot')
    # Calling dot(args, kwargs) (line 212)
    dot_call_result_27380 = invoke(stypy.reporting.localization.Localization(__file__, 212, 26), dot_27376, *[M2_27377, A2_27378], **kwargs_27379)
    
    # Applying the binary operator '+' (line 212)
    result_add_27381 = python_operator(stypy.reporting.localization.Localization(__file__, 212, 9), '+', dot_call_result_27374, dot_call_result_27380)
    
    # Assigning a type to the variable 'M4' (line 212)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 212, 4), 'M4', result_add_27381)
    
    # Assigning a Call to a Name (line 213):
    
    # Assigning a Call to a Name (line 213):
    
    # Call to dot(...): (line 213)
    # Processing the call arguments (line 213)
    # Getting the type of 'A2' (line 213)
    A2_27384 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 16), 'A2', False)
    # Getting the type of 'A4' (line 213)
    A4_27385 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 20), 'A4', False)
    # Processing the call keyword arguments (line 213)
    kwargs_27386 = {}
    # Getting the type of 'np' (line 213)
    np_27382 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 9), 'np', False)
    # Obtaining the member 'dot' of a type (line 213)
    dot_27383 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 213, 9), np_27382, 'dot')
    # Calling dot(args, kwargs) (line 213)
    dot_call_result_27387 = invoke(stypy.reporting.localization.Localization(__file__, 213, 9), dot_27383, *[A2_27384, A4_27385], **kwargs_27386)
    
    # Assigning a type to the variable 'A6' (line 213)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 213, 4), 'A6', dot_call_result_27387)
    
    # Assigning a BinOp to a Name (line 214):
    
    # Assigning a BinOp to a Name (line 214):
    
    # Call to dot(...): (line 214)
    # Processing the call arguments (line 214)
    # Getting the type of 'A4' (line 214)
    A4_27390 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 16), 'A4', False)
    # Getting the type of 'M2' (line 214)
    M2_27391 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 20), 'M2', False)
    # Processing the call keyword arguments (line 214)
    kwargs_27392 = {}
    # Getting the type of 'np' (line 214)
    np_27388 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 9), 'np', False)
    # Obtaining the member 'dot' of a type (line 214)
    dot_27389 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 214, 9), np_27388, 'dot')
    # Calling dot(args, kwargs) (line 214)
    dot_call_result_27393 = invoke(stypy.reporting.localization.Localization(__file__, 214, 9), dot_27389, *[A4_27390, M2_27391], **kwargs_27392)
    
    
    # Call to dot(...): (line 214)
    # Processing the call arguments (line 214)
    # Getting the type of 'M4' (line 214)
    M4_27396 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 33), 'M4', False)
    # Getting the type of 'A2' (line 214)
    A2_27397 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 37), 'A2', False)
    # Processing the call keyword arguments (line 214)
    kwargs_27398 = {}
    # Getting the type of 'np' (line 214)
    np_27394 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 26), 'np', False)
    # Obtaining the member 'dot' of a type (line 214)
    dot_27395 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 214, 26), np_27394, 'dot')
    # Calling dot(args, kwargs) (line 214)
    dot_call_result_27399 = invoke(stypy.reporting.localization.Localization(__file__, 214, 26), dot_27395, *[M4_27396, A2_27397], **kwargs_27398)
    
    # Applying the binary operator '+' (line 214)
    result_add_27400 = python_operator(stypy.reporting.localization.Localization(__file__, 214, 9), '+', dot_call_result_27393, dot_call_result_27399)
    
    # Assigning a type to the variable 'M6' (line 214)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 214, 4), 'M6', result_add_27400)
    
    # Assigning a Call to a Name (line 215):
    
    # Assigning a Call to a Name (line 215):
    
    # Call to dot(...): (line 215)
    # Processing the call arguments (line 215)
    # Getting the type of 'A4' (line 215)
    A4_27403 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 215, 16), 'A4', False)
    # Getting the type of 'A4' (line 215)
    A4_27404 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 215, 20), 'A4', False)
    # Processing the call keyword arguments (line 215)
    kwargs_27405 = {}
    # Getting the type of 'np' (line 215)
    np_27401 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 215, 9), 'np', False)
    # Obtaining the member 'dot' of a type (line 215)
    dot_27402 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 215, 9), np_27401, 'dot')
    # Calling dot(args, kwargs) (line 215)
    dot_call_result_27406 = invoke(stypy.reporting.localization.Localization(__file__, 215, 9), dot_27402, *[A4_27403, A4_27404], **kwargs_27405)
    
    # Assigning a type to the variable 'A8' (line 215)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 215, 4), 'A8', dot_call_result_27406)
    
    # Assigning a BinOp to a Name (line 216):
    
    # Assigning a BinOp to a Name (line 216):
    
    # Call to dot(...): (line 216)
    # Processing the call arguments (line 216)
    # Getting the type of 'A4' (line 216)
    A4_27409 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 216, 16), 'A4', False)
    # Getting the type of 'M4' (line 216)
    M4_27410 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 216, 20), 'M4', False)
    # Processing the call keyword arguments (line 216)
    kwargs_27411 = {}
    # Getting the type of 'np' (line 216)
    np_27407 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 216, 9), 'np', False)
    # Obtaining the member 'dot' of a type (line 216)
    dot_27408 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 216, 9), np_27407, 'dot')
    # Calling dot(args, kwargs) (line 216)
    dot_call_result_27412 = invoke(stypy.reporting.localization.Localization(__file__, 216, 9), dot_27408, *[A4_27409, M4_27410], **kwargs_27411)
    
    
    # Call to dot(...): (line 216)
    # Processing the call arguments (line 216)
    # Getting the type of 'M4' (line 216)
    M4_27415 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 216, 33), 'M4', False)
    # Getting the type of 'A4' (line 216)
    A4_27416 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 216, 37), 'A4', False)
    # Processing the call keyword arguments (line 216)
    kwargs_27417 = {}
    # Getting the type of 'np' (line 216)
    np_27413 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 216, 26), 'np', False)
    # Obtaining the member 'dot' of a type (line 216)
    dot_27414 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 216, 26), np_27413, 'dot')
    # Calling dot(args, kwargs) (line 216)
    dot_call_result_27418 = invoke(stypy.reporting.localization.Localization(__file__, 216, 26), dot_27414, *[M4_27415, A4_27416], **kwargs_27417)
    
    # Applying the binary operator '+' (line 216)
    result_add_27419 = python_operator(stypy.reporting.localization.Localization(__file__, 216, 9), '+', dot_call_result_27412, dot_call_result_27418)
    
    # Assigning a type to the variable 'M8' (line 216)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 216, 4), 'M8', result_add_27419)
    
    # Assigning a Call to a Name (line 217):
    
    # Assigning a Call to a Name (line 217):
    
    # Call to dot(...): (line 217)
    # Processing the call arguments (line 217)
    
    # Obtaining the type of the subscript
    int_27422 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 217, 16), 'int')
    # Getting the type of 'b' (line 217)
    b_27423 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 14), 'b', False)
    # Obtaining the member '__getitem__' of a type (line 217)
    getitem___27424 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 217, 14), b_27423, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 217)
    subscript_call_result_27425 = invoke(stypy.reporting.localization.Localization(__file__, 217, 14), getitem___27424, int_27422)
    
    # Getting the type of 'A8' (line 217)
    A8_27426 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 19), 'A8', False)
    # Applying the binary operator '*' (line 217)
    result_mul_27427 = python_operator(stypy.reporting.localization.Localization(__file__, 217, 14), '*', subscript_call_result_27425, A8_27426)
    
    
    # Obtaining the type of the subscript
    int_27428 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 217, 26), 'int')
    # Getting the type of 'b' (line 217)
    b_27429 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 24), 'b', False)
    # Obtaining the member '__getitem__' of a type (line 217)
    getitem___27430 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 217, 24), b_27429, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 217)
    subscript_call_result_27431 = invoke(stypy.reporting.localization.Localization(__file__, 217, 24), getitem___27430, int_27428)
    
    # Getting the type of 'A6' (line 217)
    A6_27432 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 29), 'A6', False)
    # Applying the binary operator '*' (line 217)
    result_mul_27433 = python_operator(stypy.reporting.localization.Localization(__file__, 217, 24), '*', subscript_call_result_27431, A6_27432)
    
    # Applying the binary operator '+' (line 217)
    result_add_27434 = python_operator(stypy.reporting.localization.Localization(__file__, 217, 14), '+', result_mul_27427, result_mul_27433)
    
    
    # Obtaining the type of the subscript
    int_27435 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 217, 36), 'int')
    # Getting the type of 'b' (line 217)
    b_27436 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 34), 'b', False)
    # Obtaining the member '__getitem__' of a type (line 217)
    getitem___27437 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 217, 34), b_27436, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 217)
    subscript_call_result_27438 = invoke(stypy.reporting.localization.Localization(__file__, 217, 34), getitem___27437, int_27435)
    
    # Getting the type of 'A4' (line 217)
    A4_27439 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 39), 'A4', False)
    # Applying the binary operator '*' (line 217)
    result_mul_27440 = python_operator(stypy.reporting.localization.Localization(__file__, 217, 34), '*', subscript_call_result_27438, A4_27439)
    
    # Applying the binary operator '+' (line 217)
    result_add_27441 = python_operator(stypy.reporting.localization.Localization(__file__, 217, 32), '+', result_add_27434, result_mul_27440)
    
    
    # Obtaining the type of the subscript
    int_27442 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 217, 46), 'int')
    # Getting the type of 'b' (line 217)
    b_27443 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 44), 'b', False)
    # Obtaining the member '__getitem__' of a type (line 217)
    getitem___27444 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 217, 44), b_27443, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 217)
    subscript_call_result_27445 = invoke(stypy.reporting.localization.Localization(__file__, 217, 44), getitem___27444, int_27442)
    
    # Getting the type of 'A2' (line 217)
    A2_27446 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 49), 'A2', False)
    # Applying the binary operator '*' (line 217)
    result_mul_27447 = python_operator(stypy.reporting.localization.Localization(__file__, 217, 44), '*', subscript_call_result_27445, A2_27446)
    
    # Applying the binary operator '+' (line 217)
    result_add_27448 = python_operator(stypy.reporting.localization.Localization(__file__, 217, 42), '+', result_add_27441, result_mul_27447)
    
    
    # Obtaining the type of the subscript
    int_27449 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 217, 56), 'int')
    # Getting the type of 'b' (line 217)
    b_27450 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 54), 'b', False)
    # Obtaining the member '__getitem__' of a type (line 217)
    getitem___27451 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 217, 54), b_27450, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 217)
    subscript_call_result_27452 = invoke(stypy.reporting.localization.Localization(__file__, 217, 54), getitem___27451, int_27449)
    
    # Getting the type of 'ident' (line 217)
    ident_27453 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 59), 'ident', False)
    # Applying the binary operator '*' (line 217)
    result_mul_27454 = python_operator(stypy.reporting.localization.Localization(__file__, 217, 54), '*', subscript_call_result_27452, ident_27453)
    
    # Applying the binary operator '+' (line 217)
    result_add_27455 = python_operator(stypy.reporting.localization.Localization(__file__, 217, 52), '+', result_add_27448, result_mul_27454)
    
    # Processing the call keyword arguments (line 217)
    kwargs_27456 = {}
    # Getting the type of 'A' (line 217)
    A_27420 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 8), 'A', False)
    # Obtaining the member 'dot' of a type (line 217)
    dot_27421 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 217, 8), A_27420, 'dot')
    # Calling dot(args, kwargs) (line 217)
    dot_call_result_27457 = invoke(stypy.reporting.localization.Localization(__file__, 217, 8), dot_27421, *[result_add_27455], **kwargs_27456)
    
    # Assigning a type to the variable 'U' (line 217)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 217, 4), 'U', dot_call_result_27457)
    
    # Assigning a BinOp to a Name (line 218):
    
    # Assigning a BinOp to a Name (line 218):
    
    # Obtaining the type of the subscript
    int_27458 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 218, 10), 'int')
    # Getting the type of 'b' (line 218)
    b_27459 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 218, 8), 'b')
    # Obtaining the member '__getitem__' of a type (line 218)
    getitem___27460 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 218, 8), b_27459, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 218)
    subscript_call_result_27461 = invoke(stypy.reporting.localization.Localization(__file__, 218, 8), getitem___27460, int_27458)
    
    # Getting the type of 'A8' (line 218)
    A8_27462 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 218, 13), 'A8')
    # Applying the binary operator '*' (line 218)
    result_mul_27463 = python_operator(stypy.reporting.localization.Localization(__file__, 218, 8), '*', subscript_call_result_27461, A8_27462)
    
    
    # Obtaining the type of the subscript
    int_27464 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 218, 20), 'int')
    # Getting the type of 'b' (line 218)
    b_27465 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 218, 18), 'b')
    # Obtaining the member '__getitem__' of a type (line 218)
    getitem___27466 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 218, 18), b_27465, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 218)
    subscript_call_result_27467 = invoke(stypy.reporting.localization.Localization(__file__, 218, 18), getitem___27466, int_27464)
    
    # Getting the type of 'A6' (line 218)
    A6_27468 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 218, 23), 'A6')
    # Applying the binary operator '*' (line 218)
    result_mul_27469 = python_operator(stypy.reporting.localization.Localization(__file__, 218, 18), '*', subscript_call_result_27467, A6_27468)
    
    # Applying the binary operator '+' (line 218)
    result_add_27470 = python_operator(stypy.reporting.localization.Localization(__file__, 218, 8), '+', result_mul_27463, result_mul_27469)
    
    
    # Obtaining the type of the subscript
    int_27471 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 218, 30), 'int')
    # Getting the type of 'b' (line 218)
    b_27472 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 218, 28), 'b')
    # Obtaining the member '__getitem__' of a type (line 218)
    getitem___27473 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 218, 28), b_27472, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 218)
    subscript_call_result_27474 = invoke(stypy.reporting.localization.Localization(__file__, 218, 28), getitem___27473, int_27471)
    
    # Getting the type of 'A4' (line 218)
    A4_27475 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 218, 33), 'A4')
    # Applying the binary operator '*' (line 218)
    result_mul_27476 = python_operator(stypy.reporting.localization.Localization(__file__, 218, 28), '*', subscript_call_result_27474, A4_27475)
    
    # Applying the binary operator '+' (line 218)
    result_add_27477 = python_operator(stypy.reporting.localization.Localization(__file__, 218, 26), '+', result_add_27470, result_mul_27476)
    
    
    # Obtaining the type of the subscript
    int_27478 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 218, 40), 'int')
    # Getting the type of 'b' (line 218)
    b_27479 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 218, 38), 'b')
    # Obtaining the member '__getitem__' of a type (line 218)
    getitem___27480 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 218, 38), b_27479, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 218)
    subscript_call_result_27481 = invoke(stypy.reporting.localization.Localization(__file__, 218, 38), getitem___27480, int_27478)
    
    # Getting the type of 'A2' (line 218)
    A2_27482 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 218, 43), 'A2')
    # Applying the binary operator '*' (line 218)
    result_mul_27483 = python_operator(stypy.reporting.localization.Localization(__file__, 218, 38), '*', subscript_call_result_27481, A2_27482)
    
    # Applying the binary operator '+' (line 218)
    result_add_27484 = python_operator(stypy.reporting.localization.Localization(__file__, 218, 36), '+', result_add_27477, result_mul_27483)
    
    
    # Obtaining the type of the subscript
    int_27485 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 218, 50), 'int')
    # Getting the type of 'b' (line 218)
    b_27486 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 218, 48), 'b')
    # Obtaining the member '__getitem__' of a type (line 218)
    getitem___27487 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 218, 48), b_27486, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 218)
    subscript_call_result_27488 = invoke(stypy.reporting.localization.Localization(__file__, 218, 48), getitem___27487, int_27485)
    
    # Getting the type of 'ident' (line 218)
    ident_27489 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 218, 53), 'ident')
    # Applying the binary operator '*' (line 218)
    result_mul_27490 = python_operator(stypy.reporting.localization.Localization(__file__, 218, 48), '*', subscript_call_result_27488, ident_27489)
    
    # Applying the binary operator '+' (line 218)
    result_add_27491 = python_operator(stypy.reporting.localization.Localization(__file__, 218, 46), '+', result_add_27484, result_mul_27490)
    
    # Assigning a type to the variable 'V' (line 218)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 218, 4), 'V', result_add_27491)
    
    # Assigning a BinOp to a Name (line 219):
    
    # Assigning a BinOp to a Name (line 219):
    
    # Call to dot(...): (line 219)
    # Processing the call arguments (line 219)
    
    # Obtaining the type of the subscript
    int_27494 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 219, 18), 'int')
    # Getting the type of 'b' (line 219)
    b_27495 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 219, 16), 'b', False)
    # Obtaining the member '__getitem__' of a type (line 219)
    getitem___27496 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 219, 16), b_27495, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 219)
    subscript_call_result_27497 = invoke(stypy.reporting.localization.Localization(__file__, 219, 16), getitem___27496, int_27494)
    
    # Getting the type of 'M8' (line 219)
    M8_27498 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 219, 21), 'M8', False)
    # Applying the binary operator '*' (line 219)
    result_mul_27499 = python_operator(stypy.reporting.localization.Localization(__file__, 219, 16), '*', subscript_call_result_27497, M8_27498)
    
    
    # Obtaining the type of the subscript
    int_27500 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 219, 28), 'int')
    # Getting the type of 'b' (line 219)
    b_27501 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 219, 26), 'b', False)
    # Obtaining the member '__getitem__' of a type (line 219)
    getitem___27502 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 219, 26), b_27501, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 219)
    subscript_call_result_27503 = invoke(stypy.reporting.localization.Localization(__file__, 219, 26), getitem___27502, int_27500)
    
    # Getting the type of 'M6' (line 219)
    M6_27504 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 219, 31), 'M6', False)
    # Applying the binary operator '*' (line 219)
    result_mul_27505 = python_operator(stypy.reporting.localization.Localization(__file__, 219, 26), '*', subscript_call_result_27503, M6_27504)
    
    # Applying the binary operator '+' (line 219)
    result_add_27506 = python_operator(stypy.reporting.localization.Localization(__file__, 219, 16), '+', result_mul_27499, result_mul_27505)
    
    
    # Obtaining the type of the subscript
    int_27507 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 219, 38), 'int')
    # Getting the type of 'b' (line 219)
    b_27508 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 219, 36), 'b', False)
    # Obtaining the member '__getitem__' of a type (line 219)
    getitem___27509 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 219, 36), b_27508, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 219)
    subscript_call_result_27510 = invoke(stypy.reporting.localization.Localization(__file__, 219, 36), getitem___27509, int_27507)
    
    # Getting the type of 'M4' (line 219)
    M4_27511 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 219, 41), 'M4', False)
    # Applying the binary operator '*' (line 219)
    result_mul_27512 = python_operator(stypy.reporting.localization.Localization(__file__, 219, 36), '*', subscript_call_result_27510, M4_27511)
    
    # Applying the binary operator '+' (line 219)
    result_add_27513 = python_operator(stypy.reporting.localization.Localization(__file__, 219, 34), '+', result_add_27506, result_mul_27512)
    
    
    # Obtaining the type of the subscript
    int_27514 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 219, 48), 'int')
    # Getting the type of 'b' (line 219)
    b_27515 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 219, 46), 'b', False)
    # Obtaining the member '__getitem__' of a type (line 219)
    getitem___27516 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 219, 46), b_27515, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 219)
    subscript_call_result_27517 = invoke(stypy.reporting.localization.Localization(__file__, 219, 46), getitem___27516, int_27514)
    
    # Getting the type of 'M2' (line 219)
    M2_27518 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 219, 51), 'M2', False)
    # Applying the binary operator '*' (line 219)
    result_mul_27519 = python_operator(stypy.reporting.localization.Localization(__file__, 219, 46), '*', subscript_call_result_27517, M2_27518)
    
    # Applying the binary operator '+' (line 219)
    result_add_27520 = python_operator(stypy.reporting.localization.Localization(__file__, 219, 44), '+', result_add_27513, result_mul_27519)
    
    # Processing the call keyword arguments (line 219)
    kwargs_27521 = {}
    # Getting the type of 'A' (line 219)
    A_27492 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 219, 10), 'A', False)
    # Obtaining the member 'dot' of a type (line 219)
    dot_27493 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 219, 10), A_27492, 'dot')
    # Calling dot(args, kwargs) (line 219)
    dot_call_result_27522 = invoke(stypy.reporting.localization.Localization(__file__, 219, 10), dot_27493, *[result_add_27520], **kwargs_27521)
    
    
    # Call to dot(...): (line 220)
    # Processing the call arguments (line 220)
    
    # Obtaining the type of the subscript
    int_27525 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 220, 20), 'int')
    # Getting the type of 'b' (line 220)
    b_27526 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 220, 18), 'b', False)
    # Obtaining the member '__getitem__' of a type (line 220)
    getitem___27527 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 220, 18), b_27526, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 220)
    subscript_call_result_27528 = invoke(stypy.reporting.localization.Localization(__file__, 220, 18), getitem___27527, int_27525)
    
    # Getting the type of 'A8' (line 220)
    A8_27529 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 220, 23), 'A8', False)
    # Applying the binary operator '*' (line 220)
    result_mul_27530 = python_operator(stypy.reporting.localization.Localization(__file__, 220, 18), '*', subscript_call_result_27528, A8_27529)
    
    
    # Obtaining the type of the subscript
    int_27531 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 220, 30), 'int')
    # Getting the type of 'b' (line 220)
    b_27532 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 220, 28), 'b', False)
    # Obtaining the member '__getitem__' of a type (line 220)
    getitem___27533 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 220, 28), b_27532, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 220)
    subscript_call_result_27534 = invoke(stypy.reporting.localization.Localization(__file__, 220, 28), getitem___27533, int_27531)
    
    # Getting the type of 'A6' (line 220)
    A6_27535 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 220, 33), 'A6', False)
    # Applying the binary operator '*' (line 220)
    result_mul_27536 = python_operator(stypy.reporting.localization.Localization(__file__, 220, 28), '*', subscript_call_result_27534, A6_27535)
    
    # Applying the binary operator '+' (line 220)
    result_add_27537 = python_operator(stypy.reporting.localization.Localization(__file__, 220, 18), '+', result_mul_27530, result_mul_27536)
    
    
    # Obtaining the type of the subscript
    int_27538 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 220, 40), 'int')
    # Getting the type of 'b' (line 220)
    b_27539 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 220, 38), 'b', False)
    # Obtaining the member '__getitem__' of a type (line 220)
    getitem___27540 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 220, 38), b_27539, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 220)
    subscript_call_result_27541 = invoke(stypy.reporting.localization.Localization(__file__, 220, 38), getitem___27540, int_27538)
    
    # Getting the type of 'A4' (line 220)
    A4_27542 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 220, 43), 'A4', False)
    # Applying the binary operator '*' (line 220)
    result_mul_27543 = python_operator(stypy.reporting.localization.Localization(__file__, 220, 38), '*', subscript_call_result_27541, A4_27542)
    
    # Applying the binary operator '+' (line 220)
    result_add_27544 = python_operator(stypy.reporting.localization.Localization(__file__, 220, 36), '+', result_add_27537, result_mul_27543)
    
    
    # Obtaining the type of the subscript
    int_27545 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 220, 50), 'int')
    # Getting the type of 'b' (line 220)
    b_27546 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 220, 48), 'b', False)
    # Obtaining the member '__getitem__' of a type (line 220)
    getitem___27547 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 220, 48), b_27546, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 220)
    subscript_call_result_27548 = invoke(stypy.reporting.localization.Localization(__file__, 220, 48), getitem___27547, int_27545)
    
    # Getting the type of 'A2' (line 220)
    A2_27549 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 220, 53), 'A2', False)
    # Applying the binary operator '*' (line 220)
    result_mul_27550 = python_operator(stypy.reporting.localization.Localization(__file__, 220, 48), '*', subscript_call_result_27548, A2_27549)
    
    # Applying the binary operator '+' (line 220)
    result_add_27551 = python_operator(stypy.reporting.localization.Localization(__file__, 220, 46), '+', result_add_27544, result_mul_27550)
    
    
    # Obtaining the type of the subscript
    int_27552 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 220, 60), 'int')
    # Getting the type of 'b' (line 220)
    b_27553 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 220, 58), 'b', False)
    # Obtaining the member '__getitem__' of a type (line 220)
    getitem___27554 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 220, 58), b_27553, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 220)
    subscript_call_result_27555 = invoke(stypy.reporting.localization.Localization(__file__, 220, 58), getitem___27554, int_27552)
    
    # Getting the type of 'ident' (line 220)
    ident_27556 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 220, 63), 'ident', False)
    # Applying the binary operator '*' (line 220)
    result_mul_27557 = python_operator(stypy.reporting.localization.Localization(__file__, 220, 58), '*', subscript_call_result_27555, ident_27556)
    
    # Applying the binary operator '+' (line 220)
    result_add_27558 = python_operator(stypy.reporting.localization.Localization(__file__, 220, 56), '+', result_add_27551, result_mul_27557)
    
    # Processing the call keyword arguments (line 220)
    kwargs_27559 = {}
    # Getting the type of 'E' (line 220)
    E_27523 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 220, 12), 'E', False)
    # Obtaining the member 'dot' of a type (line 220)
    dot_27524 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 220, 12), E_27523, 'dot')
    # Calling dot(args, kwargs) (line 220)
    dot_call_result_27560 = invoke(stypy.reporting.localization.Localization(__file__, 220, 12), dot_27524, *[result_add_27558], **kwargs_27559)
    
    # Applying the binary operator '+' (line 219)
    result_add_27561 = python_operator(stypy.reporting.localization.Localization(__file__, 219, 10), '+', dot_call_result_27522, dot_call_result_27560)
    
    # Assigning a type to the variable 'Lu' (line 219)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 219, 4), 'Lu', result_add_27561)
    
    # Assigning a BinOp to a Name (line 221):
    
    # Assigning a BinOp to a Name (line 221):
    
    # Obtaining the type of the subscript
    int_27562 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 221, 11), 'int')
    # Getting the type of 'b' (line 221)
    b_27563 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 9), 'b')
    # Obtaining the member '__getitem__' of a type (line 221)
    getitem___27564 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 221, 9), b_27563, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 221)
    subscript_call_result_27565 = invoke(stypy.reporting.localization.Localization(__file__, 221, 9), getitem___27564, int_27562)
    
    # Getting the type of 'M8' (line 221)
    M8_27566 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 14), 'M8')
    # Applying the binary operator '*' (line 221)
    result_mul_27567 = python_operator(stypy.reporting.localization.Localization(__file__, 221, 9), '*', subscript_call_result_27565, M8_27566)
    
    
    # Obtaining the type of the subscript
    int_27568 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 221, 21), 'int')
    # Getting the type of 'b' (line 221)
    b_27569 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 19), 'b')
    # Obtaining the member '__getitem__' of a type (line 221)
    getitem___27570 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 221, 19), b_27569, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 221)
    subscript_call_result_27571 = invoke(stypy.reporting.localization.Localization(__file__, 221, 19), getitem___27570, int_27568)
    
    # Getting the type of 'M6' (line 221)
    M6_27572 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 24), 'M6')
    # Applying the binary operator '*' (line 221)
    result_mul_27573 = python_operator(stypy.reporting.localization.Localization(__file__, 221, 19), '*', subscript_call_result_27571, M6_27572)
    
    # Applying the binary operator '+' (line 221)
    result_add_27574 = python_operator(stypy.reporting.localization.Localization(__file__, 221, 9), '+', result_mul_27567, result_mul_27573)
    
    
    # Obtaining the type of the subscript
    int_27575 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 221, 31), 'int')
    # Getting the type of 'b' (line 221)
    b_27576 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 29), 'b')
    # Obtaining the member '__getitem__' of a type (line 221)
    getitem___27577 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 221, 29), b_27576, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 221)
    subscript_call_result_27578 = invoke(stypy.reporting.localization.Localization(__file__, 221, 29), getitem___27577, int_27575)
    
    # Getting the type of 'M4' (line 221)
    M4_27579 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 34), 'M4')
    # Applying the binary operator '*' (line 221)
    result_mul_27580 = python_operator(stypy.reporting.localization.Localization(__file__, 221, 29), '*', subscript_call_result_27578, M4_27579)
    
    # Applying the binary operator '+' (line 221)
    result_add_27581 = python_operator(stypy.reporting.localization.Localization(__file__, 221, 27), '+', result_add_27574, result_mul_27580)
    
    
    # Obtaining the type of the subscript
    int_27582 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 221, 41), 'int')
    # Getting the type of 'b' (line 221)
    b_27583 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 39), 'b')
    # Obtaining the member '__getitem__' of a type (line 221)
    getitem___27584 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 221, 39), b_27583, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 221)
    subscript_call_result_27585 = invoke(stypy.reporting.localization.Localization(__file__, 221, 39), getitem___27584, int_27582)
    
    # Getting the type of 'M2' (line 221)
    M2_27586 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 44), 'M2')
    # Applying the binary operator '*' (line 221)
    result_mul_27587 = python_operator(stypy.reporting.localization.Localization(__file__, 221, 39), '*', subscript_call_result_27585, M2_27586)
    
    # Applying the binary operator '+' (line 221)
    result_add_27588 = python_operator(stypy.reporting.localization.Localization(__file__, 221, 37), '+', result_add_27581, result_mul_27587)
    
    # Assigning a type to the variable 'Lv' (line 221)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 221, 4), 'Lv', result_add_27588)
    
    # Obtaining an instance of the builtin type 'tuple' (line 222)
    tuple_27589 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 222, 11), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 222)
    # Adding element type (line 222)
    # Getting the type of 'U' (line 222)
    U_27590 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 222, 11), 'U')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 222, 11), tuple_27589, U_27590)
    # Adding element type (line 222)
    # Getting the type of 'V' (line 222)
    V_27591 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 222, 14), 'V')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 222, 11), tuple_27589, V_27591)
    # Adding element type (line 222)
    # Getting the type of 'Lu' (line 222)
    Lu_27592 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 222, 17), 'Lu')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 222, 11), tuple_27589, Lu_27592)
    # Adding element type (line 222)
    # Getting the type of 'Lv' (line 222)
    Lv_27593 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 222, 21), 'Lv')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 222, 11), tuple_27589, Lv_27593)
    
    # Assigning a type to the variable 'stypy_return_type' (line 222)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 222, 4), 'stypy_return_type', tuple_27589)
    
    # ################# End of '_diff_pade9(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_diff_pade9' in the type store
    # Getting the type of 'stypy_return_type' (line 206)
    stypy_return_type_27594 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 206, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_27594)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_diff_pade9'
    return stypy_return_type_27594

# Assigning a type to the variable '_diff_pade9' (line 206)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 206, 0), '_diff_pade9', _diff_pade9)

@norecursion
def expm_frechet_algo_64(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'expm_frechet_algo_64'
    module_type_store = module_type_store.open_function_context('expm_frechet_algo_64', 225, 0, False)
    
    # Passed parameters checking function
    expm_frechet_algo_64.stypy_localization = localization
    expm_frechet_algo_64.stypy_type_of_self = None
    expm_frechet_algo_64.stypy_type_store = module_type_store
    expm_frechet_algo_64.stypy_function_name = 'expm_frechet_algo_64'
    expm_frechet_algo_64.stypy_param_names_list = ['A', 'E']
    expm_frechet_algo_64.stypy_varargs_param_name = None
    expm_frechet_algo_64.stypy_kwargs_param_name = None
    expm_frechet_algo_64.stypy_call_defaults = defaults
    expm_frechet_algo_64.stypy_call_varargs = varargs
    expm_frechet_algo_64.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'expm_frechet_algo_64', ['A', 'E'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'expm_frechet_algo_64', localization, ['A', 'E'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'expm_frechet_algo_64(...)' code ##################

    
    # Assigning a Subscript to a Name (line 226):
    
    # Assigning a Subscript to a Name (line 226):
    
    # Obtaining the type of the subscript
    int_27595 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 226, 16), 'int')
    # Getting the type of 'A' (line 226)
    A_27596 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 8), 'A')
    # Obtaining the member 'shape' of a type (line 226)
    shape_27597 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 226, 8), A_27596, 'shape')
    # Obtaining the member '__getitem__' of a type (line 226)
    getitem___27598 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 226, 8), shape_27597, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 226)
    subscript_call_result_27599 = invoke(stypy.reporting.localization.Localization(__file__, 226, 8), getitem___27598, int_27595)
    
    # Assigning a type to the variable 'n' (line 226)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 226, 4), 'n', subscript_call_result_27599)
    
    # Assigning a Name to a Name (line 227):
    
    # Assigning a Name to a Name (line 227):
    # Getting the type of 'None' (line 227)
    None_27600 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 8), 'None')
    # Assigning a type to the variable 's' (line 227)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 227, 4), 's', None_27600)
    
    # Assigning a Call to a Name (line 228):
    
    # Assigning a Call to a Name (line 228):
    
    # Call to identity(...): (line 228)
    # Processing the call arguments (line 228)
    # Getting the type of 'n' (line 228)
    n_27603 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 24), 'n', False)
    # Processing the call keyword arguments (line 228)
    kwargs_27604 = {}
    # Getting the type of 'np' (line 228)
    np_27601 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 12), 'np', False)
    # Obtaining the member 'identity' of a type (line 228)
    identity_27602 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 228, 12), np_27601, 'identity')
    # Calling identity(args, kwargs) (line 228)
    identity_call_result_27605 = invoke(stypy.reporting.localization.Localization(__file__, 228, 12), identity_27602, *[n_27603], **kwargs_27604)
    
    # Assigning a type to the variable 'ident' (line 228)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 228, 4), 'ident', identity_call_result_27605)
    
    # Assigning a Call to a Name (line 229):
    
    # Assigning a Call to a Name (line 229):
    
    # Call to norm(...): (line 229)
    # Processing the call arguments (line 229)
    # Getting the type of 'A' (line 229)
    A_27609 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 229, 33), 'A', False)
    int_27610 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 229, 36), 'int')
    # Processing the call keyword arguments (line 229)
    kwargs_27611 = {}
    # Getting the type of 'scipy' (line 229)
    scipy_27606 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 229, 15), 'scipy', False)
    # Obtaining the member 'linalg' of a type (line 229)
    linalg_27607 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 229, 15), scipy_27606, 'linalg')
    # Obtaining the member 'norm' of a type (line 229)
    norm_27608 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 229, 15), linalg_27607, 'norm')
    # Calling norm(args, kwargs) (line 229)
    norm_call_result_27612 = invoke(stypy.reporting.localization.Localization(__file__, 229, 15), norm_27608, *[A_27609, int_27610], **kwargs_27611)
    
    # Assigning a type to the variable 'A_norm_1' (line 229)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 229, 4), 'A_norm_1', norm_call_result_27612)
    
    # Assigning a Tuple to a Name (line 230):
    
    # Assigning a Tuple to a Name (line 230):
    
    # Obtaining an instance of the builtin type 'tuple' (line 231)
    tuple_27613 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 231, 12), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 231)
    # Adding element type (line 231)
    
    # Obtaining an instance of the builtin type 'tuple' (line 231)
    tuple_27614 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 231, 13), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 231)
    # Adding element type (line 231)
    int_27615 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 231, 13), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 231, 13), tuple_27614, int_27615)
    # Adding element type (line 231)
    # Getting the type of '_diff_pade3' (line 231)
    _diff_pade3_27616 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 231, 16), '_diff_pade3')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 231, 13), tuple_27614, _diff_pade3_27616)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 231, 12), tuple_27613, tuple_27614)
    # Adding element type (line 231)
    
    # Obtaining an instance of the builtin type 'tuple' (line 232)
    tuple_27617 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 232, 13), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 232)
    # Adding element type (line 232)
    int_27618 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 232, 13), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 232, 13), tuple_27617, int_27618)
    # Adding element type (line 232)
    # Getting the type of '_diff_pade5' (line 232)
    _diff_pade5_27619 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 16), '_diff_pade5')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 232, 13), tuple_27617, _diff_pade5_27619)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 231, 12), tuple_27613, tuple_27617)
    # Adding element type (line 231)
    
    # Obtaining an instance of the builtin type 'tuple' (line 233)
    tuple_27620 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 233, 13), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 233)
    # Adding element type (line 233)
    int_27621 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 233, 13), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 233, 13), tuple_27620, int_27621)
    # Adding element type (line 233)
    # Getting the type of '_diff_pade7' (line 233)
    _diff_pade7_27622 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 233, 16), '_diff_pade7')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 233, 13), tuple_27620, _diff_pade7_27622)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 231, 12), tuple_27613, tuple_27620)
    # Adding element type (line 231)
    
    # Obtaining an instance of the builtin type 'tuple' (line 234)
    tuple_27623 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 234, 13), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 234)
    # Adding element type (line 234)
    int_27624 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 234, 13), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 234, 13), tuple_27623, int_27624)
    # Adding element type (line 234)
    # Getting the type of '_diff_pade9' (line 234)
    _diff_pade9_27625 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 234, 16), '_diff_pade9')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 234, 13), tuple_27623, _diff_pade9_27625)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 231, 12), tuple_27613, tuple_27623)
    
    # Assigning a type to the variable 'm_pade_pairs' (line 230)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 230, 4), 'm_pade_pairs', tuple_27613)
    
    # Getting the type of 'm_pade_pairs' (line 235)
    m_pade_pairs_27626 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 19), 'm_pade_pairs')
    # Testing the type of a for loop iterable (line 235)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 235, 4), m_pade_pairs_27626)
    # Getting the type of the for loop variable (line 235)
    for_loop_var_27627 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 235, 4), m_pade_pairs_27626)
    # Assigning a type to the variable 'm' (line 235)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 235, 4), 'm', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 235, 4), for_loop_var_27627))
    # Assigning a type to the variable 'pade' (line 235)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 235, 4), 'pade', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 235, 4), for_loop_var_27627))
    # SSA begins for a for statement (line 235)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    
    # Getting the type of 'A_norm_1' (line 236)
    A_norm_1_27628 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 236, 11), 'A_norm_1')
    
    # Obtaining the type of the subscript
    # Getting the type of 'm' (line 236)
    m_27629 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 236, 36), 'm')
    # Getting the type of 'ell_table_61' (line 236)
    ell_table_61_27630 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 236, 23), 'ell_table_61')
    # Obtaining the member '__getitem__' of a type (line 236)
    getitem___27631 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 236, 23), ell_table_61_27630, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 236)
    subscript_call_result_27632 = invoke(stypy.reporting.localization.Localization(__file__, 236, 23), getitem___27631, m_27629)
    
    # Applying the binary operator '<=' (line 236)
    result_le_27633 = python_operator(stypy.reporting.localization.Localization(__file__, 236, 11), '<=', A_norm_1_27628, subscript_call_result_27632)
    
    # Testing the type of an if condition (line 236)
    if_condition_27634 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 236, 8), result_le_27633)
    # Assigning a type to the variable 'if_condition_27634' (line 236)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 236, 8), 'if_condition_27634', if_condition_27634)
    # SSA begins for if statement (line 236)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Tuple (line 237):
    
    # Assigning a Subscript to a Name (line 237):
    
    # Obtaining the type of the subscript
    int_27635 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 237, 12), 'int')
    
    # Call to pade(...): (line 237)
    # Processing the call arguments (line 237)
    # Getting the type of 'A' (line 237)
    A_27637 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 237, 32), 'A', False)
    # Getting the type of 'E' (line 237)
    E_27638 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 237, 35), 'E', False)
    # Getting the type of 'ident' (line 237)
    ident_27639 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 237, 38), 'ident', False)
    # Processing the call keyword arguments (line 237)
    kwargs_27640 = {}
    # Getting the type of 'pade' (line 237)
    pade_27636 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 237, 27), 'pade', False)
    # Calling pade(args, kwargs) (line 237)
    pade_call_result_27641 = invoke(stypy.reporting.localization.Localization(__file__, 237, 27), pade_27636, *[A_27637, E_27638, ident_27639], **kwargs_27640)
    
    # Obtaining the member '__getitem__' of a type (line 237)
    getitem___27642 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 237, 12), pade_call_result_27641, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 237)
    subscript_call_result_27643 = invoke(stypy.reporting.localization.Localization(__file__, 237, 12), getitem___27642, int_27635)
    
    # Assigning a type to the variable 'tuple_var_assignment_26661' (line 237)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 237, 12), 'tuple_var_assignment_26661', subscript_call_result_27643)
    
    # Assigning a Subscript to a Name (line 237):
    
    # Obtaining the type of the subscript
    int_27644 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 237, 12), 'int')
    
    # Call to pade(...): (line 237)
    # Processing the call arguments (line 237)
    # Getting the type of 'A' (line 237)
    A_27646 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 237, 32), 'A', False)
    # Getting the type of 'E' (line 237)
    E_27647 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 237, 35), 'E', False)
    # Getting the type of 'ident' (line 237)
    ident_27648 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 237, 38), 'ident', False)
    # Processing the call keyword arguments (line 237)
    kwargs_27649 = {}
    # Getting the type of 'pade' (line 237)
    pade_27645 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 237, 27), 'pade', False)
    # Calling pade(args, kwargs) (line 237)
    pade_call_result_27650 = invoke(stypy.reporting.localization.Localization(__file__, 237, 27), pade_27645, *[A_27646, E_27647, ident_27648], **kwargs_27649)
    
    # Obtaining the member '__getitem__' of a type (line 237)
    getitem___27651 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 237, 12), pade_call_result_27650, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 237)
    subscript_call_result_27652 = invoke(stypy.reporting.localization.Localization(__file__, 237, 12), getitem___27651, int_27644)
    
    # Assigning a type to the variable 'tuple_var_assignment_26662' (line 237)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 237, 12), 'tuple_var_assignment_26662', subscript_call_result_27652)
    
    # Assigning a Subscript to a Name (line 237):
    
    # Obtaining the type of the subscript
    int_27653 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 237, 12), 'int')
    
    # Call to pade(...): (line 237)
    # Processing the call arguments (line 237)
    # Getting the type of 'A' (line 237)
    A_27655 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 237, 32), 'A', False)
    # Getting the type of 'E' (line 237)
    E_27656 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 237, 35), 'E', False)
    # Getting the type of 'ident' (line 237)
    ident_27657 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 237, 38), 'ident', False)
    # Processing the call keyword arguments (line 237)
    kwargs_27658 = {}
    # Getting the type of 'pade' (line 237)
    pade_27654 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 237, 27), 'pade', False)
    # Calling pade(args, kwargs) (line 237)
    pade_call_result_27659 = invoke(stypy.reporting.localization.Localization(__file__, 237, 27), pade_27654, *[A_27655, E_27656, ident_27657], **kwargs_27658)
    
    # Obtaining the member '__getitem__' of a type (line 237)
    getitem___27660 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 237, 12), pade_call_result_27659, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 237)
    subscript_call_result_27661 = invoke(stypy.reporting.localization.Localization(__file__, 237, 12), getitem___27660, int_27653)
    
    # Assigning a type to the variable 'tuple_var_assignment_26663' (line 237)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 237, 12), 'tuple_var_assignment_26663', subscript_call_result_27661)
    
    # Assigning a Subscript to a Name (line 237):
    
    # Obtaining the type of the subscript
    int_27662 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 237, 12), 'int')
    
    # Call to pade(...): (line 237)
    # Processing the call arguments (line 237)
    # Getting the type of 'A' (line 237)
    A_27664 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 237, 32), 'A', False)
    # Getting the type of 'E' (line 237)
    E_27665 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 237, 35), 'E', False)
    # Getting the type of 'ident' (line 237)
    ident_27666 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 237, 38), 'ident', False)
    # Processing the call keyword arguments (line 237)
    kwargs_27667 = {}
    # Getting the type of 'pade' (line 237)
    pade_27663 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 237, 27), 'pade', False)
    # Calling pade(args, kwargs) (line 237)
    pade_call_result_27668 = invoke(stypy.reporting.localization.Localization(__file__, 237, 27), pade_27663, *[A_27664, E_27665, ident_27666], **kwargs_27667)
    
    # Obtaining the member '__getitem__' of a type (line 237)
    getitem___27669 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 237, 12), pade_call_result_27668, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 237)
    subscript_call_result_27670 = invoke(stypy.reporting.localization.Localization(__file__, 237, 12), getitem___27669, int_27662)
    
    # Assigning a type to the variable 'tuple_var_assignment_26664' (line 237)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 237, 12), 'tuple_var_assignment_26664', subscript_call_result_27670)
    
    # Assigning a Name to a Name (line 237):
    # Getting the type of 'tuple_var_assignment_26661' (line 237)
    tuple_var_assignment_26661_27671 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 237, 12), 'tuple_var_assignment_26661')
    # Assigning a type to the variable 'U' (line 237)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 237, 12), 'U', tuple_var_assignment_26661_27671)
    
    # Assigning a Name to a Name (line 237):
    # Getting the type of 'tuple_var_assignment_26662' (line 237)
    tuple_var_assignment_26662_27672 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 237, 12), 'tuple_var_assignment_26662')
    # Assigning a type to the variable 'V' (line 237)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 237, 15), 'V', tuple_var_assignment_26662_27672)
    
    # Assigning a Name to a Name (line 237):
    # Getting the type of 'tuple_var_assignment_26663' (line 237)
    tuple_var_assignment_26663_27673 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 237, 12), 'tuple_var_assignment_26663')
    # Assigning a type to the variable 'Lu' (line 237)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 237, 18), 'Lu', tuple_var_assignment_26663_27673)
    
    # Assigning a Name to a Name (line 237):
    # Getting the type of 'tuple_var_assignment_26664' (line 237)
    tuple_var_assignment_26664_27674 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 237, 12), 'tuple_var_assignment_26664')
    # Assigning a type to the variable 'Lv' (line 237)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 237, 22), 'Lv', tuple_var_assignment_26664_27674)
    
    # Assigning a Num to a Name (line 238):
    
    # Assigning a Num to a Name (line 238):
    int_27675 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 238, 16), 'int')
    # Assigning a type to the variable 's' (line 238)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 238, 12), 's', int_27675)
    # SSA join for if statement (line 236)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Type idiom detected: calculating its left and rigth part (line 240)
    # Getting the type of 's' (line 240)
    s_27676 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 7), 's')
    # Getting the type of 'None' (line 240)
    None_27677 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 12), 'None')
    
    (may_be_27678, more_types_in_union_27679) = may_be_none(s_27676, None_27677)

    if may_be_27678:

        if more_types_in_union_27679:
            # Runtime conditional SSA (line 240)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a Call to a Name (line 242):
        
        # Assigning a Call to a Name (line 242):
        
        # Call to max(...): (line 242)
        # Processing the call arguments (line 242)
        int_27681 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 242, 16), 'int')
        
        # Call to int(...): (line 242)
        # Processing the call arguments (line 242)
        
        # Call to ceil(...): (line 242)
        # Processing the call arguments (line 242)
        
        # Call to log2(...): (line 242)
        # Processing the call arguments (line 242)
        # Getting the type of 'A_norm_1' (line 242)
        A_norm_1_27687 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 242, 39), 'A_norm_1', False)
        
        # Obtaining the type of the subscript
        int_27688 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 242, 63), 'int')
        # Getting the type of 'ell_table_61' (line 242)
        ell_table_61_27689 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 242, 50), 'ell_table_61', False)
        # Obtaining the member '__getitem__' of a type (line 242)
        getitem___27690 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 242, 50), ell_table_61_27689, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 242)
        subscript_call_result_27691 = invoke(stypy.reporting.localization.Localization(__file__, 242, 50), getitem___27690, int_27688)
        
        # Applying the binary operator 'div' (line 242)
        result_div_27692 = python_operator(stypy.reporting.localization.Localization(__file__, 242, 39), 'div', A_norm_1_27687, subscript_call_result_27691)
        
        # Processing the call keyword arguments (line 242)
        kwargs_27693 = {}
        # Getting the type of 'np' (line 242)
        np_27685 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 242, 31), 'np', False)
        # Obtaining the member 'log2' of a type (line 242)
        log2_27686 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 242, 31), np_27685, 'log2')
        # Calling log2(args, kwargs) (line 242)
        log2_call_result_27694 = invoke(stypy.reporting.localization.Localization(__file__, 242, 31), log2_27686, *[result_div_27692], **kwargs_27693)
        
        # Processing the call keyword arguments (line 242)
        kwargs_27695 = {}
        # Getting the type of 'np' (line 242)
        np_27683 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 242, 23), 'np', False)
        # Obtaining the member 'ceil' of a type (line 242)
        ceil_27684 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 242, 23), np_27683, 'ceil')
        # Calling ceil(args, kwargs) (line 242)
        ceil_call_result_27696 = invoke(stypy.reporting.localization.Localization(__file__, 242, 23), ceil_27684, *[log2_call_result_27694], **kwargs_27695)
        
        # Processing the call keyword arguments (line 242)
        kwargs_27697 = {}
        # Getting the type of 'int' (line 242)
        int_27682 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 242, 19), 'int', False)
        # Calling int(args, kwargs) (line 242)
        int_call_result_27698 = invoke(stypy.reporting.localization.Localization(__file__, 242, 19), int_27682, *[ceil_call_result_27696], **kwargs_27697)
        
        # Processing the call keyword arguments (line 242)
        kwargs_27699 = {}
        # Getting the type of 'max' (line 242)
        max_27680 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 242, 12), 'max', False)
        # Calling max(args, kwargs) (line 242)
        max_call_result_27700 = invoke(stypy.reporting.localization.Localization(__file__, 242, 12), max_27680, *[int_27681, int_call_result_27698], **kwargs_27699)
        
        # Assigning a type to the variable 's' (line 242)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 242, 8), 's', max_call_result_27700)
        
        # Assigning a BinOp to a Name (line 243):
        
        # Assigning a BinOp to a Name (line 243):
        # Getting the type of 'A' (line 243)
        A_27701 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 243, 12), 'A')
        float_27702 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 243, 16), 'float')
        
        # Getting the type of 's' (line 243)
        s_27703 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 243, 22), 's')
        # Applying the 'usub' unary operator (line 243)
        result___neg___27704 = python_operator(stypy.reporting.localization.Localization(__file__, 243, 21), 'usub', s_27703)
        
        # Applying the binary operator '**' (line 243)
        result_pow_27705 = python_operator(stypy.reporting.localization.Localization(__file__, 243, 16), '**', float_27702, result___neg___27704)
        
        # Applying the binary operator '*' (line 243)
        result_mul_27706 = python_operator(stypy.reporting.localization.Localization(__file__, 243, 12), '*', A_27701, result_pow_27705)
        
        # Assigning a type to the variable 'A' (line 243)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 243, 8), 'A', result_mul_27706)
        
        # Assigning a BinOp to a Name (line 244):
        
        # Assigning a BinOp to a Name (line 244):
        # Getting the type of 'E' (line 244)
        E_27707 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 244, 12), 'E')
        float_27708 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 244, 16), 'float')
        
        # Getting the type of 's' (line 244)
        s_27709 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 244, 22), 's')
        # Applying the 'usub' unary operator (line 244)
        result___neg___27710 = python_operator(stypy.reporting.localization.Localization(__file__, 244, 21), 'usub', s_27709)
        
        # Applying the binary operator '**' (line 244)
        result_pow_27711 = python_operator(stypy.reporting.localization.Localization(__file__, 244, 16), '**', float_27708, result___neg___27710)
        
        # Applying the binary operator '*' (line 244)
        result_mul_27712 = python_operator(stypy.reporting.localization.Localization(__file__, 244, 12), '*', E_27707, result_pow_27711)
        
        # Assigning a type to the variable 'E' (line 244)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 244, 8), 'E', result_mul_27712)
        
        # Assigning a Call to a Name (line 246):
        
        # Assigning a Call to a Name (line 246):
        
        # Call to dot(...): (line 246)
        # Processing the call arguments (line 246)
        # Getting the type of 'A' (line 246)
        A_27715 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 20), 'A', False)
        # Getting the type of 'A' (line 246)
        A_27716 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 23), 'A', False)
        # Processing the call keyword arguments (line 246)
        kwargs_27717 = {}
        # Getting the type of 'np' (line 246)
        np_27713 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 13), 'np', False)
        # Obtaining the member 'dot' of a type (line 246)
        dot_27714 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 246, 13), np_27713, 'dot')
        # Calling dot(args, kwargs) (line 246)
        dot_call_result_27718 = invoke(stypy.reporting.localization.Localization(__file__, 246, 13), dot_27714, *[A_27715, A_27716], **kwargs_27717)
        
        # Assigning a type to the variable 'A2' (line 246)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 246, 8), 'A2', dot_call_result_27718)
        
        # Assigning a BinOp to a Name (line 247):
        
        # Assigning a BinOp to a Name (line 247):
        
        # Call to dot(...): (line 247)
        # Processing the call arguments (line 247)
        # Getting the type of 'A' (line 247)
        A_27721 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 247, 20), 'A', False)
        # Getting the type of 'E' (line 247)
        E_27722 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 247, 23), 'E', False)
        # Processing the call keyword arguments (line 247)
        kwargs_27723 = {}
        # Getting the type of 'np' (line 247)
        np_27719 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 247, 13), 'np', False)
        # Obtaining the member 'dot' of a type (line 247)
        dot_27720 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 247, 13), np_27719, 'dot')
        # Calling dot(args, kwargs) (line 247)
        dot_call_result_27724 = invoke(stypy.reporting.localization.Localization(__file__, 247, 13), dot_27720, *[A_27721, E_27722], **kwargs_27723)
        
        
        # Call to dot(...): (line 247)
        # Processing the call arguments (line 247)
        # Getting the type of 'E' (line 247)
        E_27727 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 247, 35), 'E', False)
        # Getting the type of 'A' (line 247)
        A_27728 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 247, 38), 'A', False)
        # Processing the call keyword arguments (line 247)
        kwargs_27729 = {}
        # Getting the type of 'np' (line 247)
        np_27725 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 247, 28), 'np', False)
        # Obtaining the member 'dot' of a type (line 247)
        dot_27726 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 247, 28), np_27725, 'dot')
        # Calling dot(args, kwargs) (line 247)
        dot_call_result_27730 = invoke(stypy.reporting.localization.Localization(__file__, 247, 28), dot_27726, *[E_27727, A_27728], **kwargs_27729)
        
        # Applying the binary operator '+' (line 247)
        result_add_27731 = python_operator(stypy.reporting.localization.Localization(__file__, 247, 13), '+', dot_call_result_27724, dot_call_result_27730)
        
        # Assigning a type to the variable 'M2' (line 247)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 247, 8), 'M2', result_add_27731)
        
        # Assigning a Call to a Name (line 248):
        
        # Assigning a Call to a Name (line 248):
        
        # Call to dot(...): (line 248)
        # Processing the call arguments (line 248)
        # Getting the type of 'A2' (line 248)
        A2_27734 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 248, 20), 'A2', False)
        # Getting the type of 'A2' (line 248)
        A2_27735 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 248, 24), 'A2', False)
        # Processing the call keyword arguments (line 248)
        kwargs_27736 = {}
        # Getting the type of 'np' (line 248)
        np_27732 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 248, 13), 'np', False)
        # Obtaining the member 'dot' of a type (line 248)
        dot_27733 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 248, 13), np_27732, 'dot')
        # Calling dot(args, kwargs) (line 248)
        dot_call_result_27737 = invoke(stypy.reporting.localization.Localization(__file__, 248, 13), dot_27733, *[A2_27734, A2_27735], **kwargs_27736)
        
        # Assigning a type to the variable 'A4' (line 248)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 248, 8), 'A4', dot_call_result_27737)
        
        # Assigning a BinOp to a Name (line 249):
        
        # Assigning a BinOp to a Name (line 249):
        
        # Call to dot(...): (line 249)
        # Processing the call arguments (line 249)
        # Getting the type of 'A2' (line 249)
        A2_27740 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 249, 20), 'A2', False)
        # Getting the type of 'M2' (line 249)
        M2_27741 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 249, 24), 'M2', False)
        # Processing the call keyword arguments (line 249)
        kwargs_27742 = {}
        # Getting the type of 'np' (line 249)
        np_27738 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 249, 13), 'np', False)
        # Obtaining the member 'dot' of a type (line 249)
        dot_27739 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 249, 13), np_27738, 'dot')
        # Calling dot(args, kwargs) (line 249)
        dot_call_result_27743 = invoke(stypy.reporting.localization.Localization(__file__, 249, 13), dot_27739, *[A2_27740, M2_27741], **kwargs_27742)
        
        
        # Call to dot(...): (line 249)
        # Processing the call arguments (line 249)
        # Getting the type of 'M2' (line 249)
        M2_27746 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 249, 37), 'M2', False)
        # Getting the type of 'A2' (line 249)
        A2_27747 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 249, 41), 'A2', False)
        # Processing the call keyword arguments (line 249)
        kwargs_27748 = {}
        # Getting the type of 'np' (line 249)
        np_27744 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 249, 30), 'np', False)
        # Obtaining the member 'dot' of a type (line 249)
        dot_27745 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 249, 30), np_27744, 'dot')
        # Calling dot(args, kwargs) (line 249)
        dot_call_result_27749 = invoke(stypy.reporting.localization.Localization(__file__, 249, 30), dot_27745, *[M2_27746, A2_27747], **kwargs_27748)
        
        # Applying the binary operator '+' (line 249)
        result_add_27750 = python_operator(stypy.reporting.localization.Localization(__file__, 249, 13), '+', dot_call_result_27743, dot_call_result_27749)
        
        # Assigning a type to the variable 'M4' (line 249)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 249, 8), 'M4', result_add_27750)
        
        # Assigning a Call to a Name (line 250):
        
        # Assigning a Call to a Name (line 250):
        
        # Call to dot(...): (line 250)
        # Processing the call arguments (line 250)
        # Getting the type of 'A2' (line 250)
        A2_27753 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 250, 20), 'A2', False)
        # Getting the type of 'A4' (line 250)
        A4_27754 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 250, 24), 'A4', False)
        # Processing the call keyword arguments (line 250)
        kwargs_27755 = {}
        # Getting the type of 'np' (line 250)
        np_27751 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 250, 13), 'np', False)
        # Obtaining the member 'dot' of a type (line 250)
        dot_27752 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 250, 13), np_27751, 'dot')
        # Calling dot(args, kwargs) (line 250)
        dot_call_result_27756 = invoke(stypy.reporting.localization.Localization(__file__, 250, 13), dot_27752, *[A2_27753, A4_27754], **kwargs_27755)
        
        # Assigning a type to the variable 'A6' (line 250)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 250, 8), 'A6', dot_call_result_27756)
        
        # Assigning a BinOp to a Name (line 251):
        
        # Assigning a BinOp to a Name (line 251):
        
        # Call to dot(...): (line 251)
        # Processing the call arguments (line 251)
        # Getting the type of 'A4' (line 251)
        A4_27759 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 251, 20), 'A4', False)
        # Getting the type of 'M2' (line 251)
        M2_27760 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 251, 24), 'M2', False)
        # Processing the call keyword arguments (line 251)
        kwargs_27761 = {}
        # Getting the type of 'np' (line 251)
        np_27757 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 251, 13), 'np', False)
        # Obtaining the member 'dot' of a type (line 251)
        dot_27758 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 251, 13), np_27757, 'dot')
        # Calling dot(args, kwargs) (line 251)
        dot_call_result_27762 = invoke(stypy.reporting.localization.Localization(__file__, 251, 13), dot_27758, *[A4_27759, M2_27760], **kwargs_27761)
        
        
        # Call to dot(...): (line 251)
        # Processing the call arguments (line 251)
        # Getting the type of 'M4' (line 251)
        M4_27765 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 251, 37), 'M4', False)
        # Getting the type of 'A2' (line 251)
        A2_27766 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 251, 41), 'A2', False)
        # Processing the call keyword arguments (line 251)
        kwargs_27767 = {}
        # Getting the type of 'np' (line 251)
        np_27763 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 251, 30), 'np', False)
        # Obtaining the member 'dot' of a type (line 251)
        dot_27764 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 251, 30), np_27763, 'dot')
        # Calling dot(args, kwargs) (line 251)
        dot_call_result_27768 = invoke(stypy.reporting.localization.Localization(__file__, 251, 30), dot_27764, *[M4_27765, A2_27766], **kwargs_27767)
        
        # Applying the binary operator '+' (line 251)
        result_add_27769 = python_operator(stypy.reporting.localization.Localization(__file__, 251, 13), '+', dot_call_result_27762, dot_call_result_27768)
        
        # Assigning a type to the variable 'M6' (line 251)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 251, 8), 'M6', result_add_27769)
        
        # Assigning a Tuple to a Name (line 252):
        
        # Assigning a Tuple to a Name (line 252):
        
        # Obtaining an instance of the builtin type 'tuple' (line 252)
        tuple_27770 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 252, 13), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 252)
        # Adding element type (line 252)
        float_27771 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 252, 13), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 252, 13), tuple_27770, float_27771)
        # Adding element type (line 252)
        float_27772 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 252, 33), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 252, 13), tuple_27770, float_27772)
        # Adding element type (line 252)
        float_27773 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 252, 53), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 252, 13), tuple_27770, float_27773)
        # Adding element type (line 252)
        float_27774 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 253, 16), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 252, 13), tuple_27770, float_27774)
        # Adding element type (line 252)
        float_27775 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 253, 35), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 252, 13), tuple_27770, float_27775)
        # Adding element type (line 252)
        float_27776 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 253, 53), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 252, 13), tuple_27770, float_27776)
        # Adding element type (line 252)
        float_27777 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 254, 16), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 252, 13), tuple_27770, float_27777)
        # Adding element type (line 252)
        float_27778 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 254, 31), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 252, 13), tuple_27770, float_27778)
        # Adding element type (line 252)
        float_27779 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 254, 45), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 252, 13), tuple_27770, float_27779)
        # Adding element type (line 252)
        float_27780 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 254, 58), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 252, 13), tuple_27770, float_27780)
        # Adding element type (line 252)
        float_27781 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 254, 69), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 252, 13), tuple_27770, float_27781)
        # Adding element type (line 252)
        float_27782 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 255, 16), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 252, 13), tuple_27770, float_27782)
        # Adding element type (line 252)
        float_27783 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 255, 24), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 252, 13), tuple_27770, float_27783)
        # Adding element type (line 252)
        float_27784 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 255, 30), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 252, 13), tuple_27770, float_27784)
        
        # Assigning a type to the variable 'b' (line 252)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 252, 8), 'b', tuple_27770)
        
        # Assigning a BinOp to a Name (line 256):
        
        # Assigning a BinOp to a Name (line 256):
        
        # Obtaining the type of the subscript
        int_27785 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 256, 15), 'int')
        # Getting the type of 'b' (line 256)
        b_27786 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 256, 13), 'b')
        # Obtaining the member '__getitem__' of a type (line 256)
        getitem___27787 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 256, 13), b_27786, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 256)
        subscript_call_result_27788 = invoke(stypy.reporting.localization.Localization(__file__, 256, 13), getitem___27787, int_27785)
        
        # Getting the type of 'A6' (line 256)
        A6_27789 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 256, 19), 'A6')
        # Applying the binary operator '*' (line 256)
        result_mul_27790 = python_operator(stypy.reporting.localization.Localization(__file__, 256, 13), '*', subscript_call_result_27788, A6_27789)
        
        
        # Obtaining the type of the subscript
        int_27791 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 256, 26), 'int')
        # Getting the type of 'b' (line 256)
        b_27792 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 256, 24), 'b')
        # Obtaining the member '__getitem__' of a type (line 256)
        getitem___27793 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 256, 24), b_27792, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 256)
        subscript_call_result_27794 = invoke(stypy.reporting.localization.Localization(__file__, 256, 24), getitem___27793, int_27791)
        
        # Getting the type of 'A4' (line 256)
        A4_27795 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 256, 30), 'A4')
        # Applying the binary operator '*' (line 256)
        result_mul_27796 = python_operator(stypy.reporting.localization.Localization(__file__, 256, 24), '*', subscript_call_result_27794, A4_27795)
        
        # Applying the binary operator '+' (line 256)
        result_add_27797 = python_operator(stypy.reporting.localization.Localization(__file__, 256, 13), '+', result_mul_27790, result_mul_27796)
        
        
        # Obtaining the type of the subscript
        int_27798 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 256, 37), 'int')
        # Getting the type of 'b' (line 256)
        b_27799 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 256, 35), 'b')
        # Obtaining the member '__getitem__' of a type (line 256)
        getitem___27800 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 256, 35), b_27799, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 256)
        subscript_call_result_27801 = invoke(stypy.reporting.localization.Localization(__file__, 256, 35), getitem___27800, int_27798)
        
        # Getting the type of 'A2' (line 256)
        A2_27802 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 256, 40), 'A2')
        # Applying the binary operator '*' (line 256)
        result_mul_27803 = python_operator(stypy.reporting.localization.Localization(__file__, 256, 35), '*', subscript_call_result_27801, A2_27802)
        
        # Applying the binary operator '+' (line 256)
        result_add_27804 = python_operator(stypy.reporting.localization.Localization(__file__, 256, 33), '+', result_add_27797, result_mul_27803)
        
        # Assigning a type to the variable 'W1' (line 256)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 256, 8), 'W1', result_add_27804)
        
        # Assigning a BinOp to a Name (line 257):
        
        # Assigning a BinOp to a Name (line 257):
        
        # Obtaining the type of the subscript
        int_27805 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 257, 15), 'int')
        # Getting the type of 'b' (line 257)
        b_27806 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 257, 13), 'b')
        # Obtaining the member '__getitem__' of a type (line 257)
        getitem___27807 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 257, 13), b_27806, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 257)
        subscript_call_result_27808 = invoke(stypy.reporting.localization.Localization(__file__, 257, 13), getitem___27807, int_27805)
        
        # Getting the type of 'A6' (line 257)
        A6_27809 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 257, 18), 'A6')
        # Applying the binary operator '*' (line 257)
        result_mul_27810 = python_operator(stypy.reporting.localization.Localization(__file__, 257, 13), '*', subscript_call_result_27808, A6_27809)
        
        
        # Obtaining the type of the subscript
        int_27811 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 257, 25), 'int')
        # Getting the type of 'b' (line 257)
        b_27812 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 257, 23), 'b')
        # Obtaining the member '__getitem__' of a type (line 257)
        getitem___27813 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 257, 23), b_27812, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 257)
        subscript_call_result_27814 = invoke(stypy.reporting.localization.Localization(__file__, 257, 23), getitem___27813, int_27811)
        
        # Getting the type of 'A4' (line 257)
        A4_27815 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 257, 28), 'A4')
        # Applying the binary operator '*' (line 257)
        result_mul_27816 = python_operator(stypy.reporting.localization.Localization(__file__, 257, 23), '*', subscript_call_result_27814, A4_27815)
        
        # Applying the binary operator '+' (line 257)
        result_add_27817 = python_operator(stypy.reporting.localization.Localization(__file__, 257, 13), '+', result_mul_27810, result_mul_27816)
        
        
        # Obtaining the type of the subscript
        int_27818 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 257, 35), 'int')
        # Getting the type of 'b' (line 257)
        b_27819 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 257, 33), 'b')
        # Obtaining the member '__getitem__' of a type (line 257)
        getitem___27820 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 257, 33), b_27819, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 257)
        subscript_call_result_27821 = invoke(stypy.reporting.localization.Localization(__file__, 257, 33), getitem___27820, int_27818)
        
        # Getting the type of 'A2' (line 257)
        A2_27822 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 257, 38), 'A2')
        # Applying the binary operator '*' (line 257)
        result_mul_27823 = python_operator(stypy.reporting.localization.Localization(__file__, 257, 33), '*', subscript_call_result_27821, A2_27822)
        
        # Applying the binary operator '+' (line 257)
        result_add_27824 = python_operator(stypy.reporting.localization.Localization(__file__, 257, 31), '+', result_add_27817, result_mul_27823)
        
        
        # Obtaining the type of the subscript
        int_27825 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 257, 45), 'int')
        # Getting the type of 'b' (line 257)
        b_27826 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 257, 43), 'b')
        # Obtaining the member '__getitem__' of a type (line 257)
        getitem___27827 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 257, 43), b_27826, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 257)
        subscript_call_result_27828 = invoke(stypy.reporting.localization.Localization(__file__, 257, 43), getitem___27827, int_27825)
        
        # Getting the type of 'ident' (line 257)
        ident_27829 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 257, 48), 'ident')
        # Applying the binary operator '*' (line 257)
        result_mul_27830 = python_operator(stypy.reporting.localization.Localization(__file__, 257, 43), '*', subscript_call_result_27828, ident_27829)
        
        # Applying the binary operator '+' (line 257)
        result_add_27831 = python_operator(stypy.reporting.localization.Localization(__file__, 257, 41), '+', result_add_27824, result_mul_27830)
        
        # Assigning a type to the variable 'W2' (line 257)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 257, 8), 'W2', result_add_27831)
        
        # Assigning a BinOp to a Name (line 258):
        
        # Assigning a BinOp to a Name (line 258):
        
        # Obtaining the type of the subscript
        int_27832 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 258, 15), 'int')
        # Getting the type of 'b' (line 258)
        b_27833 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 258, 13), 'b')
        # Obtaining the member '__getitem__' of a type (line 258)
        getitem___27834 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 258, 13), b_27833, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 258)
        subscript_call_result_27835 = invoke(stypy.reporting.localization.Localization(__file__, 258, 13), getitem___27834, int_27832)
        
        # Getting the type of 'A6' (line 258)
        A6_27836 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 258, 19), 'A6')
        # Applying the binary operator '*' (line 258)
        result_mul_27837 = python_operator(stypy.reporting.localization.Localization(__file__, 258, 13), '*', subscript_call_result_27835, A6_27836)
        
        
        # Obtaining the type of the subscript
        int_27838 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 258, 26), 'int')
        # Getting the type of 'b' (line 258)
        b_27839 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 258, 24), 'b')
        # Obtaining the member '__getitem__' of a type (line 258)
        getitem___27840 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 258, 24), b_27839, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 258)
        subscript_call_result_27841 = invoke(stypy.reporting.localization.Localization(__file__, 258, 24), getitem___27840, int_27838)
        
        # Getting the type of 'A4' (line 258)
        A4_27842 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 258, 30), 'A4')
        # Applying the binary operator '*' (line 258)
        result_mul_27843 = python_operator(stypy.reporting.localization.Localization(__file__, 258, 24), '*', subscript_call_result_27841, A4_27842)
        
        # Applying the binary operator '+' (line 258)
        result_add_27844 = python_operator(stypy.reporting.localization.Localization(__file__, 258, 13), '+', result_mul_27837, result_mul_27843)
        
        
        # Obtaining the type of the subscript
        int_27845 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 258, 37), 'int')
        # Getting the type of 'b' (line 258)
        b_27846 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 258, 35), 'b')
        # Obtaining the member '__getitem__' of a type (line 258)
        getitem___27847 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 258, 35), b_27846, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 258)
        subscript_call_result_27848 = invoke(stypy.reporting.localization.Localization(__file__, 258, 35), getitem___27847, int_27845)
        
        # Getting the type of 'A2' (line 258)
        A2_27849 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 258, 40), 'A2')
        # Applying the binary operator '*' (line 258)
        result_mul_27850 = python_operator(stypy.reporting.localization.Localization(__file__, 258, 35), '*', subscript_call_result_27848, A2_27849)
        
        # Applying the binary operator '+' (line 258)
        result_add_27851 = python_operator(stypy.reporting.localization.Localization(__file__, 258, 33), '+', result_add_27844, result_mul_27850)
        
        # Assigning a type to the variable 'Z1' (line 258)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 258, 8), 'Z1', result_add_27851)
        
        # Assigning a BinOp to a Name (line 259):
        
        # Assigning a BinOp to a Name (line 259):
        
        # Obtaining the type of the subscript
        int_27852 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 259, 15), 'int')
        # Getting the type of 'b' (line 259)
        b_27853 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 259, 13), 'b')
        # Obtaining the member '__getitem__' of a type (line 259)
        getitem___27854 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 259, 13), b_27853, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 259)
        subscript_call_result_27855 = invoke(stypy.reporting.localization.Localization(__file__, 259, 13), getitem___27854, int_27852)
        
        # Getting the type of 'A6' (line 259)
        A6_27856 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 259, 18), 'A6')
        # Applying the binary operator '*' (line 259)
        result_mul_27857 = python_operator(stypy.reporting.localization.Localization(__file__, 259, 13), '*', subscript_call_result_27855, A6_27856)
        
        
        # Obtaining the type of the subscript
        int_27858 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 259, 25), 'int')
        # Getting the type of 'b' (line 259)
        b_27859 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 259, 23), 'b')
        # Obtaining the member '__getitem__' of a type (line 259)
        getitem___27860 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 259, 23), b_27859, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 259)
        subscript_call_result_27861 = invoke(stypy.reporting.localization.Localization(__file__, 259, 23), getitem___27860, int_27858)
        
        # Getting the type of 'A4' (line 259)
        A4_27862 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 259, 28), 'A4')
        # Applying the binary operator '*' (line 259)
        result_mul_27863 = python_operator(stypy.reporting.localization.Localization(__file__, 259, 23), '*', subscript_call_result_27861, A4_27862)
        
        # Applying the binary operator '+' (line 259)
        result_add_27864 = python_operator(stypy.reporting.localization.Localization(__file__, 259, 13), '+', result_mul_27857, result_mul_27863)
        
        
        # Obtaining the type of the subscript
        int_27865 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 259, 35), 'int')
        # Getting the type of 'b' (line 259)
        b_27866 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 259, 33), 'b')
        # Obtaining the member '__getitem__' of a type (line 259)
        getitem___27867 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 259, 33), b_27866, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 259)
        subscript_call_result_27868 = invoke(stypy.reporting.localization.Localization(__file__, 259, 33), getitem___27867, int_27865)
        
        # Getting the type of 'A2' (line 259)
        A2_27869 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 259, 38), 'A2')
        # Applying the binary operator '*' (line 259)
        result_mul_27870 = python_operator(stypy.reporting.localization.Localization(__file__, 259, 33), '*', subscript_call_result_27868, A2_27869)
        
        # Applying the binary operator '+' (line 259)
        result_add_27871 = python_operator(stypy.reporting.localization.Localization(__file__, 259, 31), '+', result_add_27864, result_mul_27870)
        
        
        # Obtaining the type of the subscript
        int_27872 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 259, 45), 'int')
        # Getting the type of 'b' (line 259)
        b_27873 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 259, 43), 'b')
        # Obtaining the member '__getitem__' of a type (line 259)
        getitem___27874 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 259, 43), b_27873, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 259)
        subscript_call_result_27875 = invoke(stypy.reporting.localization.Localization(__file__, 259, 43), getitem___27874, int_27872)
        
        # Getting the type of 'ident' (line 259)
        ident_27876 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 259, 48), 'ident')
        # Applying the binary operator '*' (line 259)
        result_mul_27877 = python_operator(stypy.reporting.localization.Localization(__file__, 259, 43), '*', subscript_call_result_27875, ident_27876)
        
        # Applying the binary operator '+' (line 259)
        result_add_27878 = python_operator(stypy.reporting.localization.Localization(__file__, 259, 41), '+', result_add_27871, result_mul_27877)
        
        # Assigning a type to the variable 'Z2' (line 259)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 259, 8), 'Z2', result_add_27878)
        
        # Assigning a BinOp to a Name (line 260):
        
        # Assigning a BinOp to a Name (line 260):
        
        # Call to dot(...): (line 260)
        # Processing the call arguments (line 260)
        # Getting the type of 'A6' (line 260)
        A6_27881 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 260, 19), 'A6', False)
        # Getting the type of 'W1' (line 260)
        W1_27882 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 260, 23), 'W1', False)
        # Processing the call keyword arguments (line 260)
        kwargs_27883 = {}
        # Getting the type of 'np' (line 260)
        np_27879 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 260, 12), 'np', False)
        # Obtaining the member 'dot' of a type (line 260)
        dot_27880 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 260, 12), np_27879, 'dot')
        # Calling dot(args, kwargs) (line 260)
        dot_call_result_27884 = invoke(stypy.reporting.localization.Localization(__file__, 260, 12), dot_27880, *[A6_27881, W1_27882], **kwargs_27883)
        
        # Getting the type of 'W2' (line 260)
        W2_27885 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 260, 29), 'W2')
        # Applying the binary operator '+' (line 260)
        result_add_27886 = python_operator(stypy.reporting.localization.Localization(__file__, 260, 12), '+', dot_call_result_27884, W2_27885)
        
        # Assigning a type to the variable 'W' (line 260)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 260, 8), 'W', result_add_27886)
        
        # Assigning a Call to a Name (line 261):
        
        # Assigning a Call to a Name (line 261):
        
        # Call to dot(...): (line 261)
        # Processing the call arguments (line 261)
        # Getting the type of 'A' (line 261)
        A_27889 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 261, 19), 'A', False)
        # Getting the type of 'W' (line 261)
        W_27890 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 261, 22), 'W', False)
        # Processing the call keyword arguments (line 261)
        kwargs_27891 = {}
        # Getting the type of 'np' (line 261)
        np_27887 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 261, 12), 'np', False)
        # Obtaining the member 'dot' of a type (line 261)
        dot_27888 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 261, 12), np_27887, 'dot')
        # Calling dot(args, kwargs) (line 261)
        dot_call_result_27892 = invoke(stypy.reporting.localization.Localization(__file__, 261, 12), dot_27888, *[A_27889, W_27890], **kwargs_27891)
        
        # Assigning a type to the variable 'U' (line 261)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 261, 8), 'U', dot_call_result_27892)
        
        # Assigning a BinOp to a Name (line 262):
        
        # Assigning a BinOp to a Name (line 262):
        
        # Call to dot(...): (line 262)
        # Processing the call arguments (line 262)
        # Getting the type of 'A6' (line 262)
        A6_27895 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 262, 19), 'A6', False)
        # Getting the type of 'Z1' (line 262)
        Z1_27896 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 262, 23), 'Z1', False)
        # Processing the call keyword arguments (line 262)
        kwargs_27897 = {}
        # Getting the type of 'np' (line 262)
        np_27893 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 262, 12), 'np', False)
        # Obtaining the member 'dot' of a type (line 262)
        dot_27894 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 262, 12), np_27893, 'dot')
        # Calling dot(args, kwargs) (line 262)
        dot_call_result_27898 = invoke(stypy.reporting.localization.Localization(__file__, 262, 12), dot_27894, *[A6_27895, Z1_27896], **kwargs_27897)
        
        # Getting the type of 'Z2' (line 262)
        Z2_27899 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 262, 29), 'Z2')
        # Applying the binary operator '+' (line 262)
        result_add_27900 = python_operator(stypy.reporting.localization.Localization(__file__, 262, 12), '+', dot_call_result_27898, Z2_27899)
        
        # Assigning a type to the variable 'V' (line 262)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 262, 8), 'V', result_add_27900)
        
        # Assigning a BinOp to a Name (line 263):
        
        # Assigning a BinOp to a Name (line 263):
        
        # Obtaining the type of the subscript
        int_27901 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 263, 16), 'int')
        # Getting the type of 'b' (line 263)
        b_27902 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 263, 14), 'b')
        # Obtaining the member '__getitem__' of a type (line 263)
        getitem___27903 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 263, 14), b_27902, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 263)
        subscript_call_result_27904 = invoke(stypy.reporting.localization.Localization(__file__, 263, 14), getitem___27903, int_27901)
        
        # Getting the type of 'M6' (line 263)
        M6_27905 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 263, 20), 'M6')
        # Applying the binary operator '*' (line 263)
        result_mul_27906 = python_operator(stypy.reporting.localization.Localization(__file__, 263, 14), '*', subscript_call_result_27904, M6_27905)
        
        
        # Obtaining the type of the subscript
        int_27907 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 263, 27), 'int')
        # Getting the type of 'b' (line 263)
        b_27908 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 263, 25), 'b')
        # Obtaining the member '__getitem__' of a type (line 263)
        getitem___27909 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 263, 25), b_27908, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 263)
        subscript_call_result_27910 = invoke(stypy.reporting.localization.Localization(__file__, 263, 25), getitem___27909, int_27907)
        
        # Getting the type of 'M4' (line 263)
        M4_27911 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 263, 31), 'M4')
        # Applying the binary operator '*' (line 263)
        result_mul_27912 = python_operator(stypy.reporting.localization.Localization(__file__, 263, 25), '*', subscript_call_result_27910, M4_27911)
        
        # Applying the binary operator '+' (line 263)
        result_add_27913 = python_operator(stypy.reporting.localization.Localization(__file__, 263, 14), '+', result_mul_27906, result_mul_27912)
        
        
        # Obtaining the type of the subscript
        int_27914 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 263, 38), 'int')
        # Getting the type of 'b' (line 263)
        b_27915 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 263, 36), 'b')
        # Obtaining the member '__getitem__' of a type (line 263)
        getitem___27916 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 263, 36), b_27915, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 263)
        subscript_call_result_27917 = invoke(stypy.reporting.localization.Localization(__file__, 263, 36), getitem___27916, int_27914)
        
        # Getting the type of 'M2' (line 263)
        M2_27918 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 263, 41), 'M2')
        # Applying the binary operator '*' (line 263)
        result_mul_27919 = python_operator(stypy.reporting.localization.Localization(__file__, 263, 36), '*', subscript_call_result_27917, M2_27918)
        
        # Applying the binary operator '+' (line 263)
        result_add_27920 = python_operator(stypy.reporting.localization.Localization(__file__, 263, 34), '+', result_add_27913, result_mul_27919)
        
        # Assigning a type to the variable 'Lw1' (line 263)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 263, 8), 'Lw1', result_add_27920)
        
        # Assigning a BinOp to a Name (line 264):
        
        # Assigning a BinOp to a Name (line 264):
        
        # Obtaining the type of the subscript
        int_27921 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 264, 16), 'int')
        # Getting the type of 'b' (line 264)
        b_27922 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 264, 14), 'b')
        # Obtaining the member '__getitem__' of a type (line 264)
        getitem___27923 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 264, 14), b_27922, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 264)
        subscript_call_result_27924 = invoke(stypy.reporting.localization.Localization(__file__, 264, 14), getitem___27923, int_27921)
        
        # Getting the type of 'M6' (line 264)
        M6_27925 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 264, 19), 'M6')
        # Applying the binary operator '*' (line 264)
        result_mul_27926 = python_operator(stypy.reporting.localization.Localization(__file__, 264, 14), '*', subscript_call_result_27924, M6_27925)
        
        
        # Obtaining the type of the subscript
        int_27927 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 264, 26), 'int')
        # Getting the type of 'b' (line 264)
        b_27928 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 264, 24), 'b')
        # Obtaining the member '__getitem__' of a type (line 264)
        getitem___27929 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 264, 24), b_27928, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 264)
        subscript_call_result_27930 = invoke(stypy.reporting.localization.Localization(__file__, 264, 24), getitem___27929, int_27927)
        
        # Getting the type of 'M4' (line 264)
        M4_27931 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 264, 29), 'M4')
        # Applying the binary operator '*' (line 264)
        result_mul_27932 = python_operator(stypy.reporting.localization.Localization(__file__, 264, 24), '*', subscript_call_result_27930, M4_27931)
        
        # Applying the binary operator '+' (line 264)
        result_add_27933 = python_operator(stypy.reporting.localization.Localization(__file__, 264, 14), '+', result_mul_27926, result_mul_27932)
        
        
        # Obtaining the type of the subscript
        int_27934 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 264, 36), 'int')
        # Getting the type of 'b' (line 264)
        b_27935 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 264, 34), 'b')
        # Obtaining the member '__getitem__' of a type (line 264)
        getitem___27936 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 264, 34), b_27935, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 264)
        subscript_call_result_27937 = invoke(stypy.reporting.localization.Localization(__file__, 264, 34), getitem___27936, int_27934)
        
        # Getting the type of 'M2' (line 264)
        M2_27938 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 264, 39), 'M2')
        # Applying the binary operator '*' (line 264)
        result_mul_27939 = python_operator(stypy.reporting.localization.Localization(__file__, 264, 34), '*', subscript_call_result_27937, M2_27938)
        
        # Applying the binary operator '+' (line 264)
        result_add_27940 = python_operator(stypy.reporting.localization.Localization(__file__, 264, 32), '+', result_add_27933, result_mul_27939)
        
        # Assigning a type to the variable 'Lw2' (line 264)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 264, 8), 'Lw2', result_add_27940)
        
        # Assigning a BinOp to a Name (line 265):
        
        # Assigning a BinOp to a Name (line 265):
        
        # Obtaining the type of the subscript
        int_27941 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 265, 16), 'int')
        # Getting the type of 'b' (line 265)
        b_27942 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 265, 14), 'b')
        # Obtaining the member '__getitem__' of a type (line 265)
        getitem___27943 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 265, 14), b_27942, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 265)
        subscript_call_result_27944 = invoke(stypy.reporting.localization.Localization(__file__, 265, 14), getitem___27943, int_27941)
        
        # Getting the type of 'M6' (line 265)
        M6_27945 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 265, 20), 'M6')
        # Applying the binary operator '*' (line 265)
        result_mul_27946 = python_operator(stypy.reporting.localization.Localization(__file__, 265, 14), '*', subscript_call_result_27944, M6_27945)
        
        
        # Obtaining the type of the subscript
        int_27947 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 265, 27), 'int')
        # Getting the type of 'b' (line 265)
        b_27948 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 265, 25), 'b')
        # Obtaining the member '__getitem__' of a type (line 265)
        getitem___27949 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 265, 25), b_27948, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 265)
        subscript_call_result_27950 = invoke(stypy.reporting.localization.Localization(__file__, 265, 25), getitem___27949, int_27947)
        
        # Getting the type of 'M4' (line 265)
        M4_27951 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 265, 31), 'M4')
        # Applying the binary operator '*' (line 265)
        result_mul_27952 = python_operator(stypy.reporting.localization.Localization(__file__, 265, 25), '*', subscript_call_result_27950, M4_27951)
        
        # Applying the binary operator '+' (line 265)
        result_add_27953 = python_operator(stypy.reporting.localization.Localization(__file__, 265, 14), '+', result_mul_27946, result_mul_27952)
        
        
        # Obtaining the type of the subscript
        int_27954 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 265, 38), 'int')
        # Getting the type of 'b' (line 265)
        b_27955 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 265, 36), 'b')
        # Obtaining the member '__getitem__' of a type (line 265)
        getitem___27956 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 265, 36), b_27955, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 265)
        subscript_call_result_27957 = invoke(stypy.reporting.localization.Localization(__file__, 265, 36), getitem___27956, int_27954)
        
        # Getting the type of 'M2' (line 265)
        M2_27958 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 265, 41), 'M2')
        # Applying the binary operator '*' (line 265)
        result_mul_27959 = python_operator(stypy.reporting.localization.Localization(__file__, 265, 36), '*', subscript_call_result_27957, M2_27958)
        
        # Applying the binary operator '+' (line 265)
        result_add_27960 = python_operator(stypy.reporting.localization.Localization(__file__, 265, 34), '+', result_add_27953, result_mul_27959)
        
        # Assigning a type to the variable 'Lz1' (line 265)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 265, 8), 'Lz1', result_add_27960)
        
        # Assigning a BinOp to a Name (line 266):
        
        # Assigning a BinOp to a Name (line 266):
        
        # Obtaining the type of the subscript
        int_27961 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 266, 16), 'int')
        # Getting the type of 'b' (line 266)
        b_27962 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 266, 14), 'b')
        # Obtaining the member '__getitem__' of a type (line 266)
        getitem___27963 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 266, 14), b_27962, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 266)
        subscript_call_result_27964 = invoke(stypy.reporting.localization.Localization(__file__, 266, 14), getitem___27963, int_27961)
        
        # Getting the type of 'M6' (line 266)
        M6_27965 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 266, 19), 'M6')
        # Applying the binary operator '*' (line 266)
        result_mul_27966 = python_operator(stypy.reporting.localization.Localization(__file__, 266, 14), '*', subscript_call_result_27964, M6_27965)
        
        
        # Obtaining the type of the subscript
        int_27967 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 266, 26), 'int')
        # Getting the type of 'b' (line 266)
        b_27968 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 266, 24), 'b')
        # Obtaining the member '__getitem__' of a type (line 266)
        getitem___27969 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 266, 24), b_27968, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 266)
        subscript_call_result_27970 = invoke(stypy.reporting.localization.Localization(__file__, 266, 24), getitem___27969, int_27967)
        
        # Getting the type of 'M4' (line 266)
        M4_27971 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 266, 29), 'M4')
        # Applying the binary operator '*' (line 266)
        result_mul_27972 = python_operator(stypy.reporting.localization.Localization(__file__, 266, 24), '*', subscript_call_result_27970, M4_27971)
        
        # Applying the binary operator '+' (line 266)
        result_add_27973 = python_operator(stypy.reporting.localization.Localization(__file__, 266, 14), '+', result_mul_27966, result_mul_27972)
        
        
        # Obtaining the type of the subscript
        int_27974 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 266, 36), 'int')
        # Getting the type of 'b' (line 266)
        b_27975 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 266, 34), 'b')
        # Obtaining the member '__getitem__' of a type (line 266)
        getitem___27976 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 266, 34), b_27975, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 266)
        subscript_call_result_27977 = invoke(stypy.reporting.localization.Localization(__file__, 266, 34), getitem___27976, int_27974)
        
        # Getting the type of 'M2' (line 266)
        M2_27978 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 266, 39), 'M2')
        # Applying the binary operator '*' (line 266)
        result_mul_27979 = python_operator(stypy.reporting.localization.Localization(__file__, 266, 34), '*', subscript_call_result_27977, M2_27978)
        
        # Applying the binary operator '+' (line 266)
        result_add_27980 = python_operator(stypy.reporting.localization.Localization(__file__, 266, 32), '+', result_add_27973, result_mul_27979)
        
        # Assigning a type to the variable 'Lz2' (line 266)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 266, 8), 'Lz2', result_add_27980)
        
        # Assigning a BinOp to a Name (line 267):
        
        # Assigning a BinOp to a Name (line 267):
        
        # Call to dot(...): (line 267)
        # Processing the call arguments (line 267)
        # Getting the type of 'A6' (line 267)
        A6_27983 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 267, 20), 'A6', False)
        # Getting the type of 'Lw1' (line 267)
        Lw1_27984 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 267, 24), 'Lw1', False)
        # Processing the call keyword arguments (line 267)
        kwargs_27985 = {}
        # Getting the type of 'np' (line 267)
        np_27981 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 267, 13), 'np', False)
        # Obtaining the member 'dot' of a type (line 267)
        dot_27982 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 267, 13), np_27981, 'dot')
        # Calling dot(args, kwargs) (line 267)
        dot_call_result_27986 = invoke(stypy.reporting.localization.Localization(__file__, 267, 13), dot_27982, *[A6_27983, Lw1_27984], **kwargs_27985)
        
        
        # Call to dot(...): (line 267)
        # Processing the call arguments (line 267)
        # Getting the type of 'M6' (line 267)
        M6_27989 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 267, 38), 'M6', False)
        # Getting the type of 'W1' (line 267)
        W1_27990 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 267, 42), 'W1', False)
        # Processing the call keyword arguments (line 267)
        kwargs_27991 = {}
        # Getting the type of 'np' (line 267)
        np_27987 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 267, 31), 'np', False)
        # Obtaining the member 'dot' of a type (line 267)
        dot_27988 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 267, 31), np_27987, 'dot')
        # Calling dot(args, kwargs) (line 267)
        dot_call_result_27992 = invoke(stypy.reporting.localization.Localization(__file__, 267, 31), dot_27988, *[M6_27989, W1_27990], **kwargs_27991)
        
        # Applying the binary operator '+' (line 267)
        result_add_27993 = python_operator(stypy.reporting.localization.Localization(__file__, 267, 13), '+', dot_call_result_27986, dot_call_result_27992)
        
        # Getting the type of 'Lw2' (line 267)
        Lw2_27994 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 267, 48), 'Lw2')
        # Applying the binary operator '+' (line 267)
        result_add_27995 = python_operator(stypy.reporting.localization.Localization(__file__, 267, 46), '+', result_add_27993, Lw2_27994)
        
        # Assigning a type to the variable 'Lw' (line 267)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 267, 8), 'Lw', result_add_27995)
        
        # Assigning a BinOp to a Name (line 268):
        
        # Assigning a BinOp to a Name (line 268):
        
        # Call to dot(...): (line 268)
        # Processing the call arguments (line 268)
        # Getting the type of 'A' (line 268)
        A_27998 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 268, 20), 'A', False)
        # Getting the type of 'Lw' (line 268)
        Lw_27999 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 268, 23), 'Lw', False)
        # Processing the call keyword arguments (line 268)
        kwargs_28000 = {}
        # Getting the type of 'np' (line 268)
        np_27996 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 268, 13), 'np', False)
        # Obtaining the member 'dot' of a type (line 268)
        dot_27997 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 268, 13), np_27996, 'dot')
        # Calling dot(args, kwargs) (line 268)
        dot_call_result_28001 = invoke(stypy.reporting.localization.Localization(__file__, 268, 13), dot_27997, *[A_27998, Lw_27999], **kwargs_28000)
        
        
        # Call to dot(...): (line 268)
        # Processing the call arguments (line 268)
        # Getting the type of 'E' (line 268)
        E_28004 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 268, 36), 'E', False)
        # Getting the type of 'W' (line 268)
        W_28005 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 268, 39), 'W', False)
        # Processing the call keyword arguments (line 268)
        kwargs_28006 = {}
        # Getting the type of 'np' (line 268)
        np_28002 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 268, 29), 'np', False)
        # Obtaining the member 'dot' of a type (line 268)
        dot_28003 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 268, 29), np_28002, 'dot')
        # Calling dot(args, kwargs) (line 268)
        dot_call_result_28007 = invoke(stypy.reporting.localization.Localization(__file__, 268, 29), dot_28003, *[E_28004, W_28005], **kwargs_28006)
        
        # Applying the binary operator '+' (line 268)
        result_add_28008 = python_operator(stypy.reporting.localization.Localization(__file__, 268, 13), '+', dot_call_result_28001, dot_call_result_28007)
        
        # Assigning a type to the variable 'Lu' (line 268)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 268, 8), 'Lu', result_add_28008)
        
        # Assigning a BinOp to a Name (line 269):
        
        # Assigning a BinOp to a Name (line 269):
        
        # Call to dot(...): (line 269)
        # Processing the call arguments (line 269)
        # Getting the type of 'A6' (line 269)
        A6_28011 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 269, 20), 'A6', False)
        # Getting the type of 'Lz1' (line 269)
        Lz1_28012 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 269, 24), 'Lz1', False)
        # Processing the call keyword arguments (line 269)
        kwargs_28013 = {}
        # Getting the type of 'np' (line 269)
        np_28009 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 269, 13), 'np', False)
        # Obtaining the member 'dot' of a type (line 269)
        dot_28010 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 269, 13), np_28009, 'dot')
        # Calling dot(args, kwargs) (line 269)
        dot_call_result_28014 = invoke(stypy.reporting.localization.Localization(__file__, 269, 13), dot_28010, *[A6_28011, Lz1_28012], **kwargs_28013)
        
        
        # Call to dot(...): (line 269)
        # Processing the call arguments (line 269)
        # Getting the type of 'M6' (line 269)
        M6_28017 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 269, 38), 'M6', False)
        # Getting the type of 'Z1' (line 269)
        Z1_28018 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 269, 42), 'Z1', False)
        # Processing the call keyword arguments (line 269)
        kwargs_28019 = {}
        # Getting the type of 'np' (line 269)
        np_28015 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 269, 31), 'np', False)
        # Obtaining the member 'dot' of a type (line 269)
        dot_28016 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 269, 31), np_28015, 'dot')
        # Calling dot(args, kwargs) (line 269)
        dot_call_result_28020 = invoke(stypy.reporting.localization.Localization(__file__, 269, 31), dot_28016, *[M6_28017, Z1_28018], **kwargs_28019)
        
        # Applying the binary operator '+' (line 269)
        result_add_28021 = python_operator(stypy.reporting.localization.Localization(__file__, 269, 13), '+', dot_call_result_28014, dot_call_result_28020)
        
        # Getting the type of 'Lz2' (line 269)
        Lz2_28022 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 269, 48), 'Lz2')
        # Applying the binary operator '+' (line 269)
        result_add_28023 = python_operator(stypy.reporting.localization.Localization(__file__, 269, 46), '+', result_add_28021, Lz2_28022)
        
        # Assigning a type to the variable 'Lv' (line 269)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 269, 8), 'Lv', result_add_28023)

        if more_types_in_union_27679:
            # SSA join for if statement (line 240)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Assigning a Call to a Name (line 271):
    
    # Assigning a Call to a Name (line 271):
    
    # Call to lu_factor(...): (line 271)
    # Processing the call arguments (line 271)
    
    # Getting the type of 'U' (line 271)
    U_28027 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 271, 37), 'U', False)
    # Applying the 'usub' unary operator (line 271)
    result___neg___28028 = python_operator(stypy.reporting.localization.Localization(__file__, 271, 36), 'usub', U_28027)
    
    # Getting the type of 'V' (line 271)
    V_28029 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 271, 41), 'V', False)
    # Applying the binary operator '+' (line 271)
    result_add_28030 = python_operator(stypy.reporting.localization.Localization(__file__, 271, 36), '+', result___neg___28028, V_28029)
    
    # Processing the call keyword arguments (line 271)
    kwargs_28031 = {}
    # Getting the type of 'scipy' (line 271)
    scipy_28024 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 271, 13), 'scipy', False)
    # Obtaining the member 'linalg' of a type (line 271)
    linalg_28025 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 271, 13), scipy_28024, 'linalg')
    # Obtaining the member 'lu_factor' of a type (line 271)
    lu_factor_28026 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 271, 13), linalg_28025, 'lu_factor')
    # Calling lu_factor(args, kwargs) (line 271)
    lu_factor_call_result_28032 = invoke(stypy.reporting.localization.Localization(__file__, 271, 13), lu_factor_28026, *[result_add_28030], **kwargs_28031)
    
    # Assigning a type to the variable 'lu_piv' (line 271)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 271, 4), 'lu_piv', lu_factor_call_result_28032)
    
    # Assigning a Call to a Name (line 272):
    
    # Assigning a Call to a Name (line 272):
    
    # Call to lu_solve(...): (line 272)
    # Processing the call arguments (line 272)
    # Getting the type of 'lu_piv' (line 272)
    lu_piv_28036 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 272, 30), 'lu_piv', False)
    # Getting the type of 'U' (line 272)
    U_28037 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 272, 38), 'U', False)
    # Getting the type of 'V' (line 272)
    V_28038 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 272, 42), 'V', False)
    # Applying the binary operator '+' (line 272)
    result_add_28039 = python_operator(stypy.reporting.localization.Localization(__file__, 272, 38), '+', U_28037, V_28038)
    
    # Processing the call keyword arguments (line 272)
    kwargs_28040 = {}
    # Getting the type of 'scipy' (line 272)
    scipy_28033 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 272, 8), 'scipy', False)
    # Obtaining the member 'linalg' of a type (line 272)
    linalg_28034 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 272, 8), scipy_28033, 'linalg')
    # Obtaining the member 'lu_solve' of a type (line 272)
    lu_solve_28035 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 272, 8), linalg_28034, 'lu_solve')
    # Calling lu_solve(args, kwargs) (line 272)
    lu_solve_call_result_28041 = invoke(stypy.reporting.localization.Localization(__file__, 272, 8), lu_solve_28035, *[lu_piv_28036, result_add_28039], **kwargs_28040)
    
    # Assigning a type to the variable 'R' (line 272)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 272, 4), 'R', lu_solve_call_result_28041)
    
    # Assigning a Call to a Name (line 273):
    
    # Assigning a Call to a Name (line 273):
    
    # Call to lu_solve(...): (line 273)
    # Processing the call arguments (line 273)
    # Getting the type of 'lu_piv' (line 273)
    lu_piv_28045 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 273, 30), 'lu_piv', False)
    # Getting the type of 'Lu' (line 273)
    Lu_28046 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 273, 38), 'Lu', False)
    # Getting the type of 'Lv' (line 273)
    Lv_28047 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 273, 43), 'Lv', False)
    # Applying the binary operator '+' (line 273)
    result_add_28048 = python_operator(stypy.reporting.localization.Localization(__file__, 273, 38), '+', Lu_28046, Lv_28047)
    
    
    # Call to dot(...): (line 273)
    # Processing the call arguments (line 273)
    # Getting the type of 'Lu' (line 273)
    Lu_28051 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 273, 56), 'Lu', False)
    # Getting the type of 'Lv' (line 273)
    Lv_28052 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 273, 61), 'Lv', False)
    # Applying the binary operator '-' (line 273)
    result_sub_28053 = python_operator(stypy.reporting.localization.Localization(__file__, 273, 56), '-', Lu_28051, Lv_28052)
    
    # Getting the type of 'R' (line 273)
    R_28054 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 273, 66), 'R', False)
    # Processing the call keyword arguments (line 273)
    kwargs_28055 = {}
    # Getting the type of 'np' (line 273)
    np_28049 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 273, 48), 'np', False)
    # Obtaining the member 'dot' of a type (line 273)
    dot_28050 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 273, 48), np_28049, 'dot')
    # Calling dot(args, kwargs) (line 273)
    dot_call_result_28056 = invoke(stypy.reporting.localization.Localization(__file__, 273, 48), dot_28050, *[result_sub_28053, R_28054], **kwargs_28055)
    
    # Applying the binary operator '+' (line 273)
    result_add_28057 = python_operator(stypy.reporting.localization.Localization(__file__, 273, 46), '+', result_add_28048, dot_call_result_28056)
    
    # Processing the call keyword arguments (line 273)
    kwargs_28058 = {}
    # Getting the type of 'scipy' (line 273)
    scipy_28042 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 273, 8), 'scipy', False)
    # Obtaining the member 'linalg' of a type (line 273)
    linalg_28043 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 273, 8), scipy_28042, 'linalg')
    # Obtaining the member 'lu_solve' of a type (line 273)
    lu_solve_28044 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 273, 8), linalg_28043, 'lu_solve')
    # Calling lu_solve(args, kwargs) (line 273)
    lu_solve_call_result_28059 = invoke(stypy.reporting.localization.Localization(__file__, 273, 8), lu_solve_28044, *[lu_piv_28045, result_add_28057], **kwargs_28058)
    
    # Assigning a type to the variable 'L' (line 273)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 273, 4), 'L', lu_solve_call_result_28059)
    
    
    # Call to range(...): (line 275)
    # Processing the call arguments (line 275)
    # Getting the type of 's' (line 275)
    s_28061 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 275, 19), 's', False)
    # Processing the call keyword arguments (line 275)
    kwargs_28062 = {}
    # Getting the type of 'range' (line 275)
    range_28060 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 275, 13), 'range', False)
    # Calling range(args, kwargs) (line 275)
    range_call_result_28063 = invoke(stypy.reporting.localization.Localization(__file__, 275, 13), range_28060, *[s_28061], **kwargs_28062)
    
    # Testing the type of a for loop iterable (line 275)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 275, 4), range_call_result_28063)
    # Getting the type of the for loop variable (line 275)
    for_loop_var_28064 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 275, 4), range_call_result_28063)
    # Assigning a type to the variable 'k' (line 275)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 275, 4), 'k', for_loop_var_28064)
    # SSA begins for a for statement (line 275)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a BinOp to a Name (line 276):
    
    # Assigning a BinOp to a Name (line 276):
    
    # Call to dot(...): (line 276)
    # Processing the call arguments (line 276)
    # Getting the type of 'R' (line 276)
    R_28067 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 276, 19), 'R', False)
    # Getting the type of 'L' (line 276)
    L_28068 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 276, 22), 'L', False)
    # Processing the call keyword arguments (line 276)
    kwargs_28069 = {}
    # Getting the type of 'np' (line 276)
    np_28065 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 276, 12), 'np', False)
    # Obtaining the member 'dot' of a type (line 276)
    dot_28066 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 276, 12), np_28065, 'dot')
    # Calling dot(args, kwargs) (line 276)
    dot_call_result_28070 = invoke(stypy.reporting.localization.Localization(__file__, 276, 12), dot_28066, *[R_28067, L_28068], **kwargs_28069)
    
    
    # Call to dot(...): (line 276)
    # Processing the call arguments (line 276)
    # Getting the type of 'L' (line 276)
    L_28073 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 276, 34), 'L', False)
    # Getting the type of 'R' (line 276)
    R_28074 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 276, 37), 'R', False)
    # Processing the call keyword arguments (line 276)
    kwargs_28075 = {}
    # Getting the type of 'np' (line 276)
    np_28071 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 276, 27), 'np', False)
    # Obtaining the member 'dot' of a type (line 276)
    dot_28072 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 276, 27), np_28071, 'dot')
    # Calling dot(args, kwargs) (line 276)
    dot_call_result_28076 = invoke(stypy.reporting.localization.Localization(__file__, 276, 27), dot_28072, *[L_28073, R_28074], **kwargs_28075)
    
    # Applying the binary operator '+' (line 276)
    result_add_28077 = python_operator(stypy.reporting.localization.Localization(__file__, 276, 12), '+', dot_call_result_28070, dot_call_result_28076)
    
    # Assigning a type to the variable 'L' (line 276)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 276, 8), 'L', result_add_28077)
    
    # Assigning a Call to a Name (line 277):
    
    # Assigning a Call to a Name (line 277):
    
    # Call to dot(...): (line 277)
    # Processing the call arguments (line 277)
    # Getting the type of 'R' (line 277)
    R_28080 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 277, 19), 'R', False)
    # Getting the type of 'R' (line 277)
    R_28081 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 277, 22), 'R', False)
    # Processing the call keyword arguments (line 277)
    kwargs_28082 = {}
    # Getting the type of 'np' (line 277)
    np_28078 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 277, 12), 'np', False)
    # Obtaining the member 'dot' of a type (line 277)
    dot_28079 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 277, 12), np_28078, 'dot')
    # Calling dot(args, kwargs) (line 277)
    dot_call_result_28083 = invoke(stypy.reporting.localization.Localization(__file__, 277, 12), dot_28079, *[R_28080, R_28081], **kwargs_28082)
    
    # Assigning a type to the variable 'R' (line 277)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 277, 8), 'R', dot_call_result_28083)
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Obtaining an instance of the builtin type 'tuple' (line 278)
    tuple_28084 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 278, 11), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 278)
    # Adding element type (line 278)
    # Getting the type of 'R' (line 278)
    R_28085 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 278, 11), 'R')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 278, 11), tuple_28084, R_28085)
    # Adding element type (line 278)
    # Getting the type of 'L' (line 278)
    L_28086 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 278, 14), 'L')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 278, 11), tuple_28084, L_28086)
    
    # Assigning a type to the variable 'stypy_return_type' (line 278)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 278, 4), 'stypy_return_type', tuple_28084)
    
    # ################# End of 'expm_frechet_algo_64(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'expm_frechet_algo_64' in the type store
    # Getting the type of 'stypy_return_type' (line 225)
    stypy_return_type_28087 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 225, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_28087)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'expm_frechet_algo_64'
    return stypy_return_type_28087

# Assigning a type to the variable 'expm_frechet_algo_64' (line 225)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 225, 0), 'expm_frechet_algo_64', expm_frechet_algo_64)

@norecursion
def vec(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'vec'
    module_type_store = module_type_store.open_function_context('vec', 281, 0, False)
    
    # Passed parameters checking function
    vec.stypy_localization = localization
    vec.stypy_type_of_self = None
    vec.stypy_type_store = module_type_store
    vec.stypy_function_name = 'vec'
    vec.stypy_param_names_list = ['M']
    vec.stypy_varargs_param_name = None
    vec.stypy_kwargs_param_name = None
    vec.stypy_call_defaults = defaults
    vec.stypy_call_varargs = varargs
    vec.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'vec', ['M'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'vec', localization, ['M'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'vec(...)' code ##################

    str_28088 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 297, (-1)), 'str', '\n    Stack columns of M to construct a single vector.\n\n    This is somewhat standard notation in linear algebra.\n\n    Parameters\n    ----------\n    M : 2d array_like\n        Input matrix\n\n    Returns\n    -------\n    v : 1d ndarray\n        Output vector\n\n    ')
    
    # Call to ravel(...): (line 298)
    # Processing the call keyword arguments (line 298)
    kwargs_28092 = {}
    # Getting the type of 'M' (line 298)
    M_28089 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 298, 11), 'M', False)
    # Obtaining the member 'T' of a type (line 298)
    T_28090 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 298, 11), M_28089, 'T')
    # Obtaining the member 'ravel' of a type (line 298)
    ravel_28091 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 298, 11), T_28090, 'ravel')
    # Calling ravel(args, kwargs) (line 298)
    ravel_call_result_28093 = invoke(stypy.reporting.localization.Localization(__file__, 298, 11), ravel_28091, *[], **kwargs_28092)
    
    # Assigning a type to the variable 'stypy_return_type' (line 298)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 298, 4), 'stypy_return_type', ravel_call_result_28093)
    
    # ################# End of 'vec(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'vec' in the type store
    # Getting the type of 'stypy_return_type' (line 281)
    stypy_return_type_28094 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 281, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_28094)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'vec'
    return stypy_return_type_28094

# Assigning a type to the variable 'vec' (line 281)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 281, 0), 'vec', vec)

@norecursion
def expm_frechet_kronform(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'None' (line 301)
    None_28095 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 301, 36), 'None')
    # Getting the type of 'True' (line 301)
    True_28096 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 301, 55), 'True')
    defaults = [None_28095, True_28096]
    # Create a new context for function 'expm_frechet_kronform'
    module_type_store = module_type_store.open_function_context('expm_frechet_kronform', 301, 0, False)
    
    # Passed parameters checking function
    expm_frechet_kronform.stypy_localization = localization
    expm_frechet_kronform.stypy_type_of_self = None
    expm_frechet_kronform.stypy_type_store = module_type_store
    expm_frechet_kronform.stypy_function_name = 'expm_frechet_kronform'
    expm_frechet_kronform.stypy_param_names_list = ['A', 'method', 'check_finite']
    expm_frechet_kronform.stypy_varargs_param_name = None
    expm_frechet_kronform.stypy_kwargs_param_name = None
    expm_frechet_kronform.stypy_call_defaults = defaults
    expm_frechet_kronform.stypy_call_varargs = varargs
    expm_frechet_kronform.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'expm_frechet_kronform', ['A', 'method', 'check_finite'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'expm_frechet_kronform', localization, ['A', 'method', 'check_finite'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'expm_frechet_kronform(...)' code ##################

    str_28097 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 333, (-1)), 'str', "\n    Construct the Kronecker form of the Frechet derivative of expm.\n\n    Parameters\n    ----------\n    A : array_like with shape (N, N)\n        Matrix to be expm'd.\n    method : str, optional\n        Extra keyword to be passed to expm_frechet.\n    check_finite : bool, optional\n        Whether to check that the input matrix contains only finite numbers.\n        Disabling may give a performance gain, but may result in problems\n        (crashes, non-termination) if the inputs do contain infinities or NaNs.\n\n    Returns\n    -------\n    K : 2d ndarray with shape (N*N, N*N)\n        Kronecker form of the Frechet derivative of the matrix exponential.\n\n    Notes\n    -----\n    This function is used to help compute the condition number\n    of the matrix exponential.\n\n    See also\n    --------\n    expm : Compute a matrix exponential.\n    expm_frechet : Compute the Frechet derivative of the matrix exponential.\n    expm_cond : Compute the relative condition number of the matrix exponential\n                in the Frobenius norm.\n\n    ")
    
    # Getting the type of 'check_finite' (line 334)
    check_finite_28098 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 334, 7), 'check_finite')
    # Testing the type of an if condition (line 334)
    if_condition_28099 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 334, 4), check_finite_28098)
    # Assigning a type to the variable 'if_condition_28099' (line 334)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 334, 4), 'if_condition_28099', if_condition_28099)
    # SSA begins for if statement (line 334)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 335):
    
    # Assigning a Call to a Name (line 335):
    
    # Call to asarray_chkfinite(...): (line 335)
    # Processing the call arguments (line 335)
    # Getting the type of 'A' (line 335)
    A_28102 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 335, 33), 'A', False)
    # Processing the call keyword arguments (line 335)
    kwargs_28103 = {}
    # Getting the type of 'np' (line 335)
    np_28100 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 335, 12), 'np', False)
    # Obtaining the member 'asarray_chkfinite' of a type (line 335)
    asarray_chkfinite_28101 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 335, 12), np_28100, 'asarray_chkfinite')
    # Calling asarray_chkfinite(args, kwargs) (line 335)
    asarray_chkfinite_call_result_28104 = invoke(stypy.reporting.localization.Localization(__file__, 335, 12), asarray_chkfinite_28101, *[A_28102], **kwargs_28103)
    
    # Assigning a type to the variable 'A' (line 335)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 335, 8), 'A', asarray_chkfinite_call_result_28104)
    # SSA branch for the else part of an if statement (line 334)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Call to a Name (line 337):
    
    # Assigning a Call to a Name (line 337):
    
    # Call to asarray(...): (line 337)
    # Processing the call arguments (line 337)
    # Getting the type of 'A' (line 337)
    A_28107 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 337, 23), 'A', False)
    # Processing the call keyword arguments (line 337)
    kwargs_28108 = {}
    # Getting the type of 'np' (line 337)
    np_28105 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 337, 12), 'np', False)
    # Obtaining the member 'asarray' of a type (line 337)
    asarray_28106 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 337, 12), np_28105, 'asarray')
    # Calling asarray(args, kwargs) (line 337)
    asarray_call_result_28109 = invoke(stypy.reporting.localization.Localization(__file__, 337, 12), asarray_28106, *[A_28107], **kwargs_28108)
    
    # Assigning a type to the variable 'A' (line 337)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 337, 8), 'A', asarray_call_result_28109)
    # SSA join for if statement (line 334)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Evaluating a boolean operation
    
    
    # Call to len(...): (line 338)
    # Processing the call arguments (line 338)
    # Getting the type of 'A' (line 338)
    A_28111 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 338, 11), 'A', False)
    # Obtaining the member 'shape' of a type (line 338)
    shape_28112 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 338, 11), A_28111, 'shape')
    # Processing the call keyword arguments (line 338)
    kwargs_28113 = {}
    # Getting the type of 'len' (line 338)
    len_28110 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 338, 7), 'len', False)
    # Calling len(args, kwargs) (line 338)
    len_call_result_28114 = invoke(stypy.reporting.localization.Localization(__file__, 338, 7), len_28110, *[shape_28112], **kwargs_28113)
    
    int_28115 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 338, 23), 'int')
    # Applying the binary operator '!=' (line 338)
    result_ne_28116 = python_operator(stypy.reporting.localization.Localization(__file__, 338, 7), '!=', len_call_result_28114, int_28115)
    
    
    
    # Obtaining the type of the subscript
    int_28117 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 338, 36), 'int')
    # Getting the type of 'A' (line 338)
    A_28118 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 338, 28), 'A')
    # Obtaining the member 'shape' of a type (line 338)
    shape_28119 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 338, 28), A_28118, 'shape')
    # Obtaining the member '__getitem__' of a type (line 338)
    getitem___28120 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 338, 28), shape_28119, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 338)
    subscript_call_result_28121 = invoke(stypy.reporting.localization.Localization(__file__, 338, 28), getitem___28120, int_28117)
    
    
    # Obtaining the type of the subscript
    int_28122 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 338, 50), 'int')
    # Getting the type of 'A' (line 338)
    A_28123 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 338, 42), 'A')
    # Obtaining the member 'shape' of a type (line 338)
    shape_28124 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 338, 42), A_28123, 'shape')
    # Obtaining the member '__getitem__' of a type (line 338)
    getitem___28125 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 338, 42), shape_28124, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 338)
    subscript_call_result_28126 = invoke(stypy.reporting.localization.Localization(__file__, 338, 42), getitem___28125, int_28122)
    
    # Applying the binary operator '!=' (line 338)
    result_ne_28127 = python_operator(stypy.reporting.localization.Localization(__file__, 338, 28), '!=', subscript_call_result_28121, subscript_call_result_28126)
    
    # Applying the binary operator 'or' (line 338)
    result_or_keyword_28128 = python_operator(stypy.reporting.localization.Localization(__file__, 338, 7), 'or', result_ne_28116, result_ne_28127)
    
    # Testing the type of an if condition (line 338)
    if_condition_28129 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 338, 4), result_or_keyword_28128)
    # Assigning a type to the variable 'if_condition_28129' (line 338)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 338, 4), 'if_condition_28129', if_condition_28129)
    # SSA begins for if statement (line 338)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 339)
    # Processing the call arguments (line 339)
    str_28131 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 339, 25), 'str', 'expected a square matrix')
    # Processing the call keyword arguments (line 339)
    kwargs_28132 = {}
    # Getting the type of 'ValueError' (line 339)
    ValueError_28130 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 339, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 339)
    ValueError_call_result_28133 = invoke(stypy.reporting.localization.Localization(__file__, 339, 14), ValueError_28130, *[str_28131], **kwargs_28132)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 339, 8), ValueError_call_result_28133, 'raise parameter', BaseException)
    # SSA join for if statement (line 338)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Subscript to a Name (line 341):
    
    # Assigning a Subscript to a Name (line 341):
    
    # Obtaining the type of the subscript
    int_28134 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 341, 16), 'int')
    # Getting the type of 'A' (line 341)
    A_28135 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 341, 8), 'A')
    # Obtaining the member 'shape' of a type (line 341)
    shape_28136 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 341, 8), A_28135, 'shape')
    # Obtaining the member '__getitem__' of a type (line 341)
    getitem___28137 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 341, 8), shape_28136, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 341)
    subscript_call_result_28138 = invoke(stypy.reporting.localization.Localization(__file__, 341, 8), getitem___28137, int_28134)
    
    # Assigning a type to the variable 'n' (line 341)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 341, 4), 'n', subscript_call_result_28138)
    
    # Assigning a Call to a Name (line 342):
    
    # Assigning a Call to a Name (line 342):
    
    # Call to identity(...): (line 342)
    # Processing the call arguments (line 342)
    # Getting the type of 'n' (line 342)
    n_28141 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 342, 24), 'n', False)
    # Processing the call keyword arguments (line 342)
    kwargs_28142 = {}
    # Getting the type of 'np' (line 342)
    np_28139 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 342, 12), 'np', False)
    # Obtaining the member 'identity' of a type (line 342)
    identity_28140 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 342, 12), np_28139, 'identity')
    # Calling identity(args, kwargs) (line 342)
    identity_call_result_28143 = invoke(stypy.reporting.localization.Localization(__file__, 342, 12), identity_28140, *[n_28141], **kwargs_28142)
    
    # Assigning a type to the variable 'ident' (line 342)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 342, 4), 'ident', identity_call_result_28143)
    
    # Assigning a List to a Name (line 343):
    
    # Assigning a List to a Name (line 343):
    
    # Obtaining an instance of the builtin type 'list' (line 343)
    list_28144 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 343, 11), 'list')
    # Adding type elements to the builtin type 'list' instance (line 343)
    
    # Assigning a type to the variable 'cols' (line 343)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 343, 4), 'cols', list_28144)
    
    
    # Call to range(...): (line 344)
    # Processing the call arguments (line 344)
    # Getting the type of 'n' (line 344)
    n_28146 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 344, 19), 'n', False)
    # Processing the call keyword arguments (line 344)
    kwargs_28147 = {}
    # Getting the type of 'range' (line 344)
    range_28145 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 344, 13), 'range', False)
    # Calling range(args, kwargs) (line 344)
    range_call_result_28148 = invoke(stypy.reporting.localization.Localization(__file__, 344, 13), range_28145, *[n_28146], **kwargs_28147)
    
    # Testing the type of a for loop iterable (line 344)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 344, 4), range_call_result_28148)
    # Getting the type of the for loop variable (line 344)
    for_loop_var_28149 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 344, 4), range_call_result_28148)
    # Assigning a type to the variable 'i' (line 344)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 344, 4), 'i', for_loop_var_28149)
    # SSA begins for a for statement (line 344)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    
    # Call to range(...): (line 345)
    # Processing the call arguments (line 345)
    # Getting the type of 'n' (line 345)
    n_28151 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 345, 23), 'n', False)
    # Processing the call keyword arguments (line 345)
    kwargs_28152 = {}
    # Getting the type of 'range' (line 345)
    range_28150 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 345, 17), 'range', False)
    # Calling range(args, kwargs) (line 345)
    range_call_result_28153 = invoke(stypy.reporting.localization.Localization(__file__, 345, 17), range_28150, *[n_28151], **kwargs_28152)
    
    # Testing the type of a for loop iterable (line 345)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 345, 8), range_call_result_28153)
    # Getting the type of the for loop variable (line 345)
    for_loop_var_28154 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 345, 8), range_call_result_28153)
    # Assigning a type to the variable 'j' (line 345)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 345, 8), 'j', for_loop_var_28154)
    # SSA begins for a for statement (line 345)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Call to a Name (line 346):
    
    # Assigning a Call to a Name (line 346):
    
    # Call to outer(...): (line 346)
    # Processing the call arguments (line 346)
    
    # Obtaining the type of the subscript
    # Getting the type of 'i' (line 346)
    i_28157 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 346, 31), 'i', False)
    # Getting the type of 'ident' (line 346)
    ident_28158 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 346, 25), 'ident', False)
    # Obtaining the member '__getitem__' of a type (line 346)
    getitem___28159 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 346, 25), ident_28158, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 346)
    subscript_call_result_28160 = invoke(stypy.reporting.localization.Localization(__file__, 346, 25), getitem___28159, i_28157)
    
    
    # Obtaining the type of the subscript
    # Getting the type of 'j' (line 346)
    j_28161 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 346, 41), 'j', False)
    # Getting the type of 'ident' (line 346)
    ident_28162 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 346, 35), 'ident', False)
    # Obtaining the member '__getitem__' of a type (line 346)
    getitem___28163 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 346, 35), ident_28162, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 346)
    subscript_call_result_28164 = invoke(stypy.reporting.localization.Localization(__file__, 346, 35), getitem___28163, j_28161)
    
    # Processing the call keyword arguments (line 346)
    kwargs_28165 = {}
    # Getting the type of 'np' (line 346)
    np_28155 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 346, 16), 'np', False)
    # Obtaining the member 'outer' of a type (line 346)
    outer_28156 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 346, 16), np_28155, 'outer')
    # Calling outer(args, kwargs) (line 346)
    outer_call_result_28166 = invoke(stypy.reporting.localization.Localization(__file__, 346, 16), outer_28156, *[subscript_call_result_28160, subscript_call_result_28164], **kwargs_28165)
    
    # Assigning a type to the variable 'E' (line 346)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 346, 12), 'E', outer_call_result_28166)
    
    # Assigning a Call to a Name (line 347):
    
    # Assigning a Call to a Name (line 347):
    
    # Call to expm_frechet(...): (line 347)
    # Processing the call arguments (line 347)
    # Getting the type of 'A' (line 347)
    A_28168 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 347, 29), 'A', False)
    # Getting the type of 'E' (line 347)
    E_28169 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 347, 32), 'E', False)
    # Processing the call keyword arguments (line 347)
    # Getting the type of 'method' (line 348)
    method_28170 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 348, 27), 'method', False)
    keyword_28171 = method_28170
    # Getting the type of 'False' (line 348)
    False_28172 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 348, 48), 'False', False)
    keyword_28173 = False_28172
    # Getting the type of 'False' (line 348)
    False_28174 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 348, 68), 'False', False)
    keyword_28175 = False_28174
    kwargs_28176 = {'compute_expm': keyword_28173, 'method': keyword_28171, 'check_finite': keyword_28175}
    # Getting the type of 'expm_frechet' (line 347)
    expm_frechet_28167 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 347, 16), 'expm_frechet', False)
    # Calling expm_frechet(args, kwargs) (line 347)
    expm_frechet_call_result_28177 = invoke(stypy.reporting.localization.Localization(__file__, 347, 16), expm_frechet_28167, *[A_28168, E_28169], **kwargs_28176)
    
    # Assigning a type to the variable 'F' (line 347)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 347, 12), 'F', expm_frechet_call_result_28177)
    
    # Call to append(...): (line 349)
    # Processing the call arguments (line 349)
    
    # Call to vec(...): (line 349)
    # Processing the call arguments (line 349)
    # Getting the type of 'F' (line 349)
    F_28181 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 349, 28), 'F', False)
    # Processing the call keyword arguments (line 349)
    kwargs_28182 = {}
    # Getting the type of 'vec' (line 349)
    vec_28180 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 349, 24), 'vec', False)
    # Calling vec(args, kwargs) (line 349)
    vec_call_result_28183 = invoke(stypy.reporting.localization.Localization(__file__, 349, 24), vec_28180, *[F_28181], **kwargs_28182)
    
    # Processing the call keyword arguments (line 349)
    kwargs_28184 = {}
    # Getting the type of 'cols' (line 349)
    cols_28178 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 349, 12), 'cols', False)
    # Obtaining the member 'append' of a type (line 349)
    append_28179 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 349, 12), cols_28178, 'append')
    # Calling append(args, kwargs) (line 349)
    append_call_result_28185 = invoke(stypy.reporting.localization.Localization(__file__, 349, 12), append_28179, *[vec_call_result_28183], **kwargs_28184)
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to vstack(...): (line 350)
    # Processing the call arguments (line 350)
    # Getting the type of 'cols' (line 350)
    cols_28188 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 350, 21), 'cols', False)
    # Processing the call keyword arguments (line 350)
    kwargs_28189 = {}
    # Getting the type of 'np' (line 350)
    np_28186 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 350, 11), 'np', False)
    # Obtaining the member 'vstack' of a type (line 350)
    vstack_28187 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 350, 11), np_28186, 'vstack')
    # Calling vstack(args, kwargs) (line 350)
    vstack_call_result_28190 = invoke(stypy.reporting.localization.Localization(__file__, 350, 11), vstack_28187, *[cols_28188], **kwargs_28189)
    
    # Obtaining the member 'T' of a type (line 350)
    T_28191 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 350, 11), vstack_call_result_28190, 'T')
    # Assigning a type to the variable 'stypy_return_type' (line 350)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 350, 4), 'stypy_return_type', T_28191)
    
    # ################# End of 'expm_frechet_kronform(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'expm_frechet_kronform' in the type store
    # Getting the type of 'stypy_return_type' (line 301)
    stypy_return_type_28192 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 301, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_28192)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'expm_frechet_kronform'
    return stypy_return_type_28192

# Assigning a type to the variable 'expm_frechet_kronform' (line 301)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 301, 0), 'expm_frechet_kronform', expm_frechet_kronform)

@norecursion
def expm_cond(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'True' (line 353)
    True_28193 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 353, 30), 'True')
    defaults = [True_28193]
    # Create a new context for function 'expm_cond'
    module_type_store = module_type_store.open_function_context('expm_cond', 353, 0, False)
    
    # Passed parameters checking function
    expm_cond.stypy_localization = localization
    expm_cond.stypy_type_of_self = None
    expm_cond.stypy_type_store = module_type_store
    expm_cond.stypy_function_name = 'expm_cond'
    expm_cond.stypy_param_names_list = ['A', 'check_finite']
    expm_cond.stypy_varargs_param_name = None
    expm_cond.stypy_kwargs_param_name = None
    expm_cond.stypy_call_defaults = defaults
    expm_cond.stypy_call_varargs = varargs
    expm_cond.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'expm_cond', ['A', 'check_finite'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'expm_cond', localization, ['A', 'check_finite'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'expm_cond(...)' code ##################

    str_28194 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 384, (-1)), 'str', '\n    Relative condition number of the matrix exponential in the Frobenius norm.\n\n    Parameters\n    ----------\n    A : 2d array_like\n        Square input matrix with shape (N, N).\n    check_finite : bool, optional\n        Whether to check that the input matrix contains only finite numbers.\n        Disabling may give a performance gain, but may result in problems\n        (crashes, non-termination) if the inputs do contain infinities or NaNs.\n\n    Returns\n    -------\n    kappa : float\n        The relative condition number of the matrix exponential\n        in the Frobenius norm\n\n    Notes\n    -----\n    A faster estimate for the condition number in the 1-norm\n    has been published but is not yet implemented in scipy.\n\n    .. versionadded:: 0.14.0\n\n    See also\n    --------\n    expm : Compute the exponential of a matrix.\n    expm_frechet : Compute the Frechet derivative of the matrix exponential.\n\n    ')
    
    # Getting the type of 'check_finite' (line 385)
    check_finite_28195 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 385, 7), 'check_finite')
    # Testing the type of an if condition (line 385)
    if_condition_28196 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 385, 4), check_finite_28195)
    # Assigning a type to the variable 'if_condition_28196' (line 385)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 385, 4), 'if_condition_28196', if_condition_28196)
    # SSA begins for if statement (line 385)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 386):
    
    # Assigning a Call to a Name (line 386):
    
    # Call to asarray_chkfinite(...): (line 386)
    # Processing the call arguments (line 386)
    # Getting the type of 'A' (line 386)
    A_28199 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 386, 33), 'A', False)
    # Processing the call keyword arguments (line 386)
    kwargs_28200 = {}
    # Getting the type of 'np' (line 386)
    np_28197 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 386, 12), 'np', False)
    # Obtaining the member 'asarray_chkfinite' of a type (line 386)
    asarray_chkfinite_28198 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 386, 12), np_28197, 'asarray_chkfinite')
    # Calling asarray_chkfinite(args, kwargs) (line 386)
    asarray_chkfinite_call_result_28201 = invoke(stypy.reporting.localization.Localization(__file__, 386, 12), asarray_chkfinite_28198, *[A_28199], **kwargs_28200)
    
    # Assigning a type to the variable 'A' (line 386)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 386, 8), 'A', asarray_chkfinite_call_result_28201)
    # SSA branch for the else part of an if statement (line 385)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Call to a Name (line 388):
    
    # Assigning a Call to a Name (line 388):
    
    # Call to asarray(...): (line 388)
    # Processing the call arguments (line 388)
    # Getting the type of 'A' (line 388)
    A_28204 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 388, 23), 'A', False)
    # Processing the call keyword arguments (line 388)
    kwargs_28205 = {}
    # Getting the type of 'np' (line 388)
    np_28202 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 388, 12), 'np', False)
    # Obtaining the member 'asarray' of a type (line 388)
    asarray_28203 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 388, 12), np_28202, 'asarray')
    # Calling asarray(args, kwargs) (line 388)
    asarray_call_result_28206 = invoke(stypy.reporting.localization.Localization(__file__, 388, 12), asarray_28203, *[A_28204], **kwargs_28205)
    
    # Assigning a type to the variable 'A' (line 388)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 388, 8), 'A', asarray_call_result_28206)
    # SSA join for if statement (line 385)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Evaluating a boolean operation
    
    
    # Call to len(...): (line 389)
    # Processing the call arguments (line 389)
    # Getting the type of 'A' (line 389)
    A_28208 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 389, 11), 'A', False)
    # Obtaining the member 'shape' of a type (line 389)
    shape_28209 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 389, 11), A_28208, 'shape')
    # Processing the call keyword arguments (line 389)
    kwargs_28210 = {}
    # Getting the type of 'len' (line 389)
    len_28207 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 389, 7), 'len', False)
    # Calling len(args, kwargs) (line 389)
    len_call_result_28211 = invoke(stypy.reporting.localization.Localization(__file__, 389, 7), len_28207, *[shape_28209], **kwargs_28210)
    
    int_28212 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 389, 23), 'int')
    # Applying the binary operator '!=' (line 389)
    result_ne_28213 = python_operator(stypy.reporting.localization.Localization(__file__, 389, 7), '!=', len_call_result_28211, int_28212)
    
    
    
    # Obtaining the type of the subscript
    int_28214 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 389, 36), 'int')
    # Getting the type of 'A' (line 389)
    A_28215 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 389, 28), 'A')
    # Obtaining the member 'shape' of a type (line 389)
    shape_28216 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 389, 28), A_28215, 'shape')
    # Obtaining the member '__getitem__' of a type (line 389)
    getitem___28217 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 389, 28), shape_28216, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 389)
    subscript_call_result_28218 = invoke(stypy.reporting.localization.Localization(__file__, 389, 28), getitem___28217, int_28214)
    
    
    # Obtaining the type of the subscript
    int_28219 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 389, 50), 'int')
    # Getting the type of 'A' (line 389)
    A_28220 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 389, 42), 'A')
    # Obtaining the member 'shape' of a type (line 389)
    shape_28221 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 389, 42), A_28220, 'shape')
    # Obtaining the member '__getitem__' of a type (line 389)
    getitem___28222 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 389, 42), shape_28221, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 389)
    subscript_call_result_28223 = invoke(stypy.reporting.localization.Localization(__file__, 389, 42), getitem___28222, int_28219)
    
    # Applying the binary operator '!=' (line 389)
    result_ne_28224 = python_operator(stypy.reporting.localization.Localization(__file__, 389, 28), '!=', subscript_call_result_28218, subscript_call_result_28223)
    
    # Applying the binary operator 'or' (line 389)
    result_or_keyword_28225 = python_operator(stypy.reporting.localization.Localization(__file__, 389, 7), 'or', result_ne_28213, result_ne_28224)
    
    # Testing the type of an if condition (line 389)
    if_condition_28226 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 389, 4), result_or_keyword_28225)
    # Assigning a type to the variable 'if_condition_28226' (line 389)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 389, 4), 'if_condition_28226', if_condition_28226)
    # SSA begins for if statement (line 389)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 390)
    # Processing the call arguments (line 390)
    str_28228 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 390, 25), 'str', 'expected a square matrix')
    # Processing the call keyword arguments (line 390)
    kwargs_28229 = {}
    # Getting the type of 'ValueError' (line 390)
    ValueError_28227 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 390, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 390)
    ValueError_call_result_28230 = invoke(stypy.reporting.localization.Localization(__file__, 390, 14), ValueError_28227, *[str_28228], **kwargs_28229)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 390, 8), ValueError_call_result_28230, 'raise parameter', BaseException)
    # SSA join for if statement (line 389)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 392):
    
    # Assigning a Call to a Name (line 392):
    
    # Call to expm(...): (line 392)
    # Processing the call arguments (line 392)
    # Getting the type of 'A' (line 392)
    A_28234 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 392, 26), 'A', False)
    # Processing the call keyword arguments (line 392)
    kwargs_28235 = {}
    # Getting the type of 'scipy' (line 392)
    scipy_28231 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 392, 8), 'scipy', False)
    # Obtaining the member 'linalg' of a type (line 392)
    linalg_28232 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 392, 8), scipy_28231, 'linalg')
    # Obtaining the member 'expm' of a type (line 392)
    expm_28233 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 392, 8), linalg_28232, 'expm')
    # Calling expm(args, kwargs) (line 392)
    expm_call_result_28236 = invoke(stypy.reporting.localization.Localization(__file__, 392, 8), expm_28233, *[A_28234], **kwargs_28235)
    
    # Assigning a type to the variable 'X' (line 392)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 392, 4), 'X', expm_call_result_28236)
    
    # Assigning a Call to a Name (line 393):
    
    # Assigning a Call to a Name (line 393):
    
    # Call to expm_frechet_kronform(...): (line 393)
    # Processing the call arguments (line 393)
    # Getting the type of 'A' (line 393)
    A_28238 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 393, 30), 'A', False)
    # Processing the call keyword arguments (line 393)
    # Getting the type of 'False' (line 393)
    False_28239 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 393, 46), 'False', False)
    keyword_28240 = False_28239
    kwargs_28241 = {'check_finite': keyword_28240}
    # Getting the type of 'expm_frechet_kronform' (line 393)
    expm_frechet_kronform_28237 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 393, 8), 'expm_frechet_kronform', False)
    # Calling expm_frechet_kronform(args, kwargs) (line 393)
    expm_frechet_kronform_call_result_28242 = invoke(stypy.reporting.localization.Localization(__file__, 393, 8), expm_frechet_kronform_28237, *[A_28238], **kwargs_28241)
    
    # Assigning a type to the variable 'K' (line 393)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 393, 4), 'K', expm_frechet_kronform_call_result_28242)
    
    # Assigning a Call to a Name (line 398):
    
    # Assigning a Call to a Name (line 398):
    
    # Call to norm(...): (line 398)
    # Processing the call arguments (line 398)
    # Getting the type of 'A' (line 398)
    A_28246 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 398, 31), 'A', False)
    str_28247 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 398, 34), 'str', 'fro')
    # Processing the call keyword arguments (line 398)
    kwargs_28248 = {}
    # Getting the type of 'scipy' (line 398)
    scipy_28243 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 398, 13), 'scipy', False)
    # Obtaining the member 'linalg' of a type (line 398)
    linalg_28244 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 398, 13), scipy_28243, 'linalg')
    # Obtaining the member 'norm' of a type (line 398)
    norm_28245 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 398, 13), linalg_28244, 'norm')
    # Calling norm(args, kwargs) (line 398)
    norm_call_result_28249 = invoke(stypy.reporting.localization.Localization(__file__, 398, 13), norm_28245, *[A_28246, str_28247], **kwargs_28248)
    
    # Assigning a type to the variable 'A_norm' (line 398)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 398, 4), 'A_norm', norm_call_result_28249)
    
    # Assigning a Call to a Name (line 399):
    
    # Assigning a Call to a Name (line 399):
    
    # Call to norm(...): (line 399)
    # Processing the call arguments (line 399)
    # Getting the type of 'X' (line 399)
    X_28253 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 399, 31), 'X', False)
    str_28254 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 399, 34), 'str', 'fro')
    # Processing the call keyword arguments (line 399)
    kwargs_28255 = {}
    # Getting the type of 'scipy' (line 399)
    scipy_28250 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 399, 13), 'scipy', False)
    # Obtaining the member 'linalg' of a type (line 399)
    linalg_28251 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 399, 13), scipy_28250, 'linalg')
    # Obtaining the member 'norm' of a type (line 399)
    norm_28252 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 399, 13), linalg_28251, 'norm')
    # Calling norm(args, kwargs) (line 399)
    norm_call_result_28256 = invoke(stypy.reporting.localization.Localization(__file__, 399, 13), norm_28252, *[X_28253, str_28254], **kwargs_28255)
    
    # Assigning a type to the variable 'X_norm' (line 399)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 399, 4), 'X_norm', norm_call_result_28256)
    
    # Assigning a Call to a Name (line 400):
    
    # Assigning a Call to a Name (line 400):
    
    # Call to norm(...): (line 400)
    # Processing the call arguments (line 400)
    # Getting the type of 'K' (line 400)
    K_28260 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 400, 31), 'K', False)
    int_28261 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 400, 34), 'int')
    # Processing the call keyword arguments (line 400)
    kwargs_28262 = {}
    # Getting the type of 'scipy' (line 400)
    scipy_28257 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 400, 13), 'scipy', False)
    # Obtaining the member 'linalg' of a type (line 400)
    linalg_28258 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 400, 13), scipy_28257, 'linalg')
    # Obtaining the member 'norm' of a type (line 400)
    norm_28259 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 400, 13), linalg_28258, 'norm')
    # Calling norm(args, kwargs) (line 400)
    norm_call_result_28263 = invoke(stypy.reporting.localization.Localization(__file__, 400, 13), norm_28259, *[K_28260, int_28261], **kwargs_28262)
    
    # Assigning a type to the variable 'K_norm' (line 400)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 400, 4), 'K_norm', norm_call_result_28263)
    
    # Assigning a BinOp to a Name (line 402):
    
    # Assigning a BinOp to a Name (line 402):
    # Getting the type of 'K_norm' (line 402)
    K_norm_28264 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 402, 13), 'K_norm')
    # Getting the type of 'A_norm' (line 402)
    A_norm_28265 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 402, 22), 'A_norm')
    # Applying the binary operator '*' (line 402)
    result_mul_28266 = python_operator(stypy.reporting.localization.Localization(__file__, 402, 13), '*', K_norm_28264, A_norm_28265)
    
    # Getting the type of 'X_norm' (line 402)
    X_norm_28267 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 402, 32), 'X_norm')
    # Applying the binary operator 'div' (line 402)
    result_div_28268 = python_operator(stypy.reporting.localization.Localization(__file__, 402, 12), 'div', result_mul_28266, X_norm_28267)
    
    # Assigning a type to the variable 'kappa' (line 402)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 402, 4), 'kappa', result_div_28268)
    # Getting the type of 'kappa' (line 403)
    kappa_28269 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 403, 11), 'kappa')
    # Assigning a type to the variable 'stypy_return_type' (line 403)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 403, 4), 'stypy_return_type', kappa_28269)
    
    # ################# End of 'expm_cond(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'expm_cond' in the type store
    # Getting the type of 'stypy_return_type' (line 353)
    stypy_return_type_28270 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 353, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_28270)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'expm_cond'
    return stypy_return_type_28270

# Assigning a type to the variable 'expm_cond' (line 353)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 353, 0), 'expm_cond', expm_cond)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
