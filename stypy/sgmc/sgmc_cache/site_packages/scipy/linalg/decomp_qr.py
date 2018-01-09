
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: '''QR decomposition functions.'''
2: from __future__ import division, print_function, absolute_import
3: 
4: import numpy
5: 
6: # Local imports
7: from .lapack import get_lapack_funcs
8: from .misc import _datacopied
9: 
10: __all__ = ['qr', 'qr_multiply', 'rq']
11: 
12: 
13: def safecall(f, name, *args, **kwargs):
14:     '''Call a LAPACK routine, determining lwork automatically and handling
15:     error return values'''
16:     lwork = kwargs.get("lwork", None)
17:     if lwork in (None, -1):
18:         kwargs['lwork'] = -1
19:         ret = f(*args, **kwargs)
20:         kwargs['lwork'] = ret[-2][0].real.astype(numpy.int)
21:     ret = f(*args, **kwargs)
22:     if ret[-1] < 0:
23:         raise ValueError("illegal value in %d-th argument of internal %s"
24:                          % (-ret[-1], name))
25:     return ret[:-2]
26: 
27: 
28: def qr(a, overwrite_a=False, lwork=None, mode='full', pivoting=False,
29:        check_finite=True):
30:     '''
31:     Compute QR decomposition of a matrix.
32: 
33:     Calculate the decomposition ``A = Q R`` where Q is unitary/orthogonal
34:     and R upper triangular.
35: 
36:     Parameters
37:     ----------
38:     a : (M, N) array_like
39:         Matrix to be decomposed
40:     overwrite_a : bool, optional
41:         Whether data in a is overwritten (may improve performance)
42:     lwork : int, optional
43:         Work array size, lwork >= a.shape[1]. If None or -1, an optimal size
44:         is computed.
45:     mode : {'full', 'r', 'economic', 'raw'}, optional
46:         Determines what information is to be returned: either both Q and R
47:         ('full', default), only R ('r') or both Q and R but computed in
48:         economy-size ('economic', see Notes). The final option 'raw'
49:         (added in Scipy 0.11) makes the function return two matrices
50:         (Q, TAU) in the internal format used by LAPACK.
51:     pivoting : bool, optional
52:         Whether or not factorization should include pivoting for rank-revealing
53:         qr decomposition. If pivoting, compute the decomposition
54:         ``A P = Q R`` as above, but where P is chosen such that the diagonal
55:         of R is non-increasing.
56:     check_finite : bool, optional
57:         Whether to check that the input matrix contains only finite numbers.
58:         Disabling may give a performance gain, but may result in problems
59:         (crashes, non-termination) if the inputs do contain infinities or NaNs.
60: 
61:     Returns
62:     -------
63:     Q : float or complex ndarray
64:         Of shape (M, M), or (M, K) for ``mode='economic'``.  Not returned
65:         if ``mode='r'``.
66:     R : float or complex ndarray
67:         Of shape (M, N), or (K, N) for ``mode='economic'``.  ``K = min(M, N)``.
68:     P : int ndarray
69:         Of shape (N,) for ``pivoting=True``. Not returned if
70:         ``pivoting=False``.
71: 
72:     Raises
73:     ------
74:     LinAlgError
75:         Raised if decomposition fails
76: 
77:     Notes
78:     -----
79:     This is an interface to the LAPACK routines dgeqrf, zgeqrf,
80:     dorgqr, zungqr, dgeqp3, and zgeqp3.
81: 
82:     If ``mode=economic``, the shapes of Q and R are (M, K) and (K, N) instead
83:     of (M,M) and (M,N), with ``K=min(M,N)``.
84: 
85:     Examples
86:     --------
87:     >>> from scipy import random, linalg, dot, diag, all, allclose
88:     >>> a = random.randn(9, 6)
89: 
90:     >>> q, r = linalg.qr(a)
91:     >>> allclose(a, np.dot(q, r))
92:     True
93:     >>> q.shape, r.shape
94:     ((9, 9), (9, 6))
95: 
96:     >>> r2 = linalg.qr(a, mode='r')
97:     >>> allclose(r, r2)
98:     True
99: 
100:     >>> q3, r3 = linalg.qr(a, mode='economic')
101:     >>> q3.shape, r3.shape
102:     ((9, 6), (6, 6))
103: 
104:     >>> q4, r4, p4 = linalg.qr(a, pivoting=True)
105:     >>> d = abs(diag(r4))
106:     >>> all(d[1:] <= d[:-1])
107:     True
108:     >>> allclose(a[:, p4], dot(q4, r4))
109:     True
110:     >>> q4.shape, r4.shape, p4.shape
111:     ((9, 9), (9, 6), (6,))
112: 
113:     >>> q5, r5, p5 = linalg.qr(a, mode='economic', pivoting=True)
114:     >>> q5.shape, r5.shape, p5.shape
115:     ((9, 6), (6, 6), (6,))
116: 
117:     '''
118:     # 'qr' was the old default, equivalent to 'full'. Neither 'full' nor
119:     # 'qr' are used below.
120:     # 'raw' is used internally by qr_multiply
121:     if mode not in ['full', 'qr', 'r', 'economic', 'raw']:
122:         raise ValueError(
123:                  "Mode argument should be one of ['full', 'r', 'economic', 'raw']")
124: 
125:     if check_finite:
126:         a1 = numpy.asarray_chkfinite(a)
127:     else:
128:         a1 = numpy.asarray(a)
129:     if len(a1.shape) != 2:
130:         raise ValueError("expected 2D array")
131:     M, N = a1.shape
132:     overwrite_a = overwrite_a or (_datacopied(a1, a))
133: 
134:     if pivoting:
135:         geqp3, = get_lapack_funcs(('geqp3',), (a1,))
136:         qr, jpvt, tau = safecall(geqp3, "geqp3", a1, overwrite_a=overwrite_a)
137:         jpvt -= 1  # geqp3 returns a 1-based index array, so subtract 1
138:     else:
139:         geqrf, = get_lapack_funcs(('geqrf',), (a1,))
140:         qr, tau = safecall(geqrf, "geqrf", a1, lwork=lwork,
141:             overwrite_a=overwrite_a)
142: 
143:     if mode not in ['economic', 'raw'] or M < N:
144:         R = numpy.triu(qr)
145:     else:
146:         R = numpy.triu(qr[:N, :])
147: 
148:     if pivoting:
149:         Rj = R, jpvt
150:     else:
151:         Rj = R,
152: 
153:     if mode == 'r':
154:         return Rj
155:     elif mode == 'raw':
156:         return ((qr, tau),) + Rj
157: 
158:     gor_un_gqr, = get_lapack_funcs(('orgqr',), (qr,))
159: 
160:     if M < N:
161:         Q, = safecall(gor_un_gqr, "gorgqr/gungqr", qr[:, :M], tau,
162:             lwork=lwork, overwrite_a=1)
163:     elif mode == 'economic':
164:         Q, = safecall(gor_un_gqr, "gorgqr/gungqr", qr, tau, lwork=lwork,
165:             overwrite_a=1)
166:     else:
167:         t = qr.dtype.char
168:         qqr = numpy.empty((M, M), dtype=t)
169:         qqr[:, :N] = qr
170:         Q, = safecall(gor_un_gqr, "gorgqr/gungqr", qqr, tau, lwork=lwork,
171:             overwrite_a=1)
172: 
173:     return (Q,) + Rj
174: 
175: 
176: def qr_multiply(a, c, mode='right', pivoting=False, conjugate=False,
177:     overwrite_a=False, overwrite_c=False):
178:     '''
179:     Calculate the QR decomposition and multiply Q with a matrix.
180: 
181:     Calculate the decomposition ``A = Q R`` where Q is unitary/orthogonal
182:     and R upper triangular. Multiply Q with a vector or a matrix c.
183: 
184:     Parameters
185:     ----------
186:     a : array_like, shape (M, N)
187:         Matrix to be decomposed
188:     c : array_like, one- or two-dimensional
189:         calculate the product of c and q, depending on the mode:
190:     mode : {'left', 'right'}, optional
191:         ``dot(Q, c)`` is returned if mode is 'left',
192:         ``dot(c, Q)`` is returned if mode is 'right'.
193:         The shape of c must be appropriate for the matrix multiplications,
194:         if mode is 'left', ``min(a.shape) == c.shape[0]``,
195:         if mode is 'right', ``a.shape[0] == c.shape[1]``.
196:     pivoting : bool, optional
197:         Whether or not factorization should include pivoting for rank-revealing
198:         qr decomposition, see the documentation of qr.
199:     conjugate : bool, optional
200:         Whether Q should be complex-conjugated. This might be faster
201:         than explicit conjugation.
202:     overwrite_a : bool, optional
203:         Whether data in a is overwritten (may improve performance)
204:     overwrite_c : bool, optional
205:         Whether data in c is overwritten (may improve performance).
206:         If this is used, c must be big enough to keep the result,
207:         i.e. c.shape[0] = a.shape[0] if mode is 'left'.
208: 
209: 
210:     Returns
211:     -------
212:     CQ : float or complex ndarray
213:         the product of Q and c, as defined in mode
214:     R : float or complex ndarray
215:         Of shape (K, N), ``K = min(M, N)``.
216:     P : ndarray of ints
217:         Of shape (N,) for ``pivoting=True``.
218:         Not returned if ``pivoting=False``.
219: 
220:     Raises
221:     ------
222:     LinAlgError
223:         Raised if decomposition fails
224: 
225:     Notes
226:     -----
227:     This is an interface to the LAPACK routines dgeqrf, zgeqrf,
228:     dormqr, zunmqr, dgeqp3, and zgeqp3.
229: 
230:     .. versionadded:: 0.11.0
231: 
232:     '''
233:     if mode not in ['left', 'right']:
234:         raise ValueError("Mode argument should be one of ['left', 'right']")
235:     c = numpy.asarray_chkfinite(c)
236:     onedim = c.ndim == 1
237:     if onedim:
238:         c = c.reshape(1, len(c))
239:         if mode == "left":
240:             c = c.T
241: 
242:     a = numpy.asarray(a)  # chkfinite done in qr
243:     M, N = a.shape
244:     if not (mode == "left" and
245:                 (not overwrite_c and min(M, N) == c.shape[0] or
246:                      overwrite_c and M == c.shape[0]) or
247:             mode == "right" and M == c.shape[1]):
248:         raise ValueError("objects are not aligned")
249: 
250:     raw = qr(a, overwrite_a, None, "raw", pivoting)
251:     Q, tau = raw[0]
252: 
253:     gor_un_mqr, = get_lapack_funcs(('ormqr',), (Q,))
254:     if gor_un_mqr.typecode in ('s', 'd'):
255:         trans = "T"
256:     else:
257:         trans = "C"
258: 
259:     Q = Q[:, :min(M, N)]
260:     if M > N and mode == "left" and not overwrite_c:
261:         if conjugate:
262:             cc = numpy.zeros((c.shape[1], M), dtype=c.dtype, order="F")
263:             cc[:, :N] = c.T
264:         else:
265:             cc = numpy.zeros((M, c.shape[1]), dtype=c.dtype, order="F")
266:             cc[:N, :] = c
267:             trans = "N"
268:         if conjugate:
269:             lr = "R"
270:         else:
271:             lr = "L"
272:         overwrite_c = True
273:     elif c.flags["C_CONTIGUOUS"] and trans == "T" or conjugate:
274:         cc = c.T
275:         if mode == "left":
276:             lr = "R"
277:         else:
278:             lr = "L"
279:     else:
280:         trans = "N"
281:         cc = c
282:         if mode == "left":
283:             lr = "L"
284:         else:
285:             lr = "R"
286:     cQ, = safecall(gor_un_mqr, "gormqr/gunmqr", lr, trans, Q, tau, cc,
287:             overwrite_c=overwrite_c)
288:     if trans != "N":
289:         cQ = cQ.T
290:     if mode == "right":
291:         cQ = cQ[:, :min(M, N)]
292:     if onedim:
293:         cQ = cQ.ravel()
294: 
295:     return (cQ,) + raw[1:]
296: 
297: 
298: def rq(a, overwrite_a=False, lwork=None, mode='full', check_finite=True):
299:     '''
300:     Compute RQ decomposition of a matrix.
301: 
302:     Calculate the decomposition ``A = R Q`` where Q is unitary/orthogonal
303:     and R upper triangular.
304: 
305:     Parameters
306:     ----------
307:     a : (M, N) array_like
308:         Matrix to be decomposed
309:     overwrite_a : bool, optional
310:         Whether data in a is overwritten (may improve performance)
311:     lwork : int, optional
312:         Work array size, lwork >= a.shape[1]. If None or -1, an optimal size
313:         is computed.
314:     mode : {'full', 'r', 'economic'}, optional
315:         Determines what information is to be returned: either both Q and R
316:         ('full', default), only R ('r') or both Q and R but computed in
317:         economy-size ('economic', see Notes).
318:     check_finite : bool, optional
319:         Whether to check that the input matrix contains only finite numbers.
320:         Disabling may give a performance gain, but may result in problems
321:         (crashes, non-termination) if the inputs do contain infinities or NaNs.
322: 
323:     Returns
324:     -------
325:     R : float or complex ndarray
326:         Of shape (M, N) or (M, K) for ``mode='economic'``.  ``K = min(M, N)``.
327:     Q : float or complex ndarray
328:         Of shape (N, N) or (K, N) for ``mode='economic'``.  Not returned
329:         if ``mode='r'``.
330: 
331:     Raises
332:     ------
333:     LinAlgError
334:         If decomposition fails.
335: 
336:     Notes
337:     -----
338:     This is an interface to the LAPACK routines sgerqf, dgerqf, cgerqf, zgerqf,
339:     sorgrq, dorgrq, cungrq and zungrq.
340: 
341:     If ``mode=economic``, the shapes of Q and R are (K, N) and (M, K) instead
342:     of (N,N) and (M,N), with ``K=min(M,N)``.
343: 
344:     Examples
345:     --------
346:     >>> from scipy import linalg
347:     >>> from numpy import random, dot, allclose
348:     >>> a = random.randn(6, 9)
349:     >>> r, q = linalg.rq(a)
350:     >>> allclose(a, dot(r, q))
351:     True
352:     >>> r.shape, q.shape
353:     ((6, 9), (9, 9))
354:     >>> r2 = linalg.rq(a, mode='r')
355:     >>> allclose(r, r2)
356:     True
357:     >>> r3, q3 = linalg.rq(a, mode='economic')
358:     >>> r3.shape, q3.shape
359:     ((6, 6), (6, 9))
360: 
361:     '''
362:     if mode not in ['full', 'r', 'economic']:
363:         raise ValueError(
364:                  "Mode argument should be one of ['full', 'r', 'economic']")
365: 
366:     if check_finite:
367:         a1 = numpy.asarray_chkfinite(a)
368:     else:
369:         a1 = numpy.asarray(a)
370:     if len(a1.shape) != 2:
371:         raise ValueError('expected matrix')
372:     M, N = a1.shape
373:     overwrite_a = overwrite_a or (_datacopied(a1, a))
374: 
375:     gerqf, = get_lapack_funcs(('gerqf',), (a1,))
376:     rq, tau = safecall(gerqf, 'gerqf', a1, lwork=lwork,
377:                        overwrite_a=overwrite_a)
378:     if not mode == 'economic' or N < M:
379:         R = numpy.triu(rq, N-M)
380:     else:
381:         R = numpy.triu(rq[-M:, -M:])
382: 
383:     if mode == 'r':
384:         return R
385: 
386:     gor_un_grq, = get_lapack_funcs(('orgrq',), (rq,))
387: 
388:     if N < M:
389:         Q, = safecall(gor_un_grq, "gorgrq/gungrq", rq[-N:], tau, lwork=lwork,
390:                       overwrite_a=1)
391:     elif mode == 'economic':
392:         Q, = safecall(gor_un_grq, "gorgrq/gungrq", rq, tau, lwork=lwork,
393:                       overwrite_a=1)
394:     else:
395:         rq1 = numpy.empty((N, N), dtype=rq.dtype)
396:         rq1[-M:] = rq
397:         Q, = safecall(gor_un_grq, "gorgrq/gungrq", rq1, tau, lwork=lwork,
398:                       overwrite_a=1)
399: 
400:     return R, Q
401: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_18402 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1, 0), 'str', 'QR decomposition functions.')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 4, 0))

# 'import numpy' statement (line 4)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/linalg/')
import_18403 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'numpy')

if (type(import_18403) is not StypyTypeError):

    if (import_18403 != 'pyd_module'):
        __import__(import_18403)
        sys_modules_18404 = sys.modules[import_18403]
        import_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'numpy', sys_modules_18404.module_type_store, module_type_store)
    else:
        import numpy

        import_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'numpy', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 4)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 4, 0), 'numpy', import_18403)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/linalg/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 7, 0))

# 'from scipy.linalg.lapack import get_lapack_funcs' statement (line 7)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/linalg/')
import_18405 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'scipy.linalg.lapack')

if (type(import_18405) is not StypyTypeError):

    if (import_18405 != 'pyd_module'):
        __import__(import_18405)
        sys_modules_18406 = sys.modules[import_18405]
        import_from_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'scipy.linalg.lapack', sys_modules_18406.module_type_store, module_type_store, ['get_lapack_funcs'])
        nest_module(stypy.reporting.localization.Localization(__file__, 7, 0), __file__, sys_modules_18406, sys_modules_18406.module_type_store, module_type_store)
    else:
        from scipy.linalg.lapack import get_lapack_funcs

        import_from_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'scipy.linalg.lapack', None, module_type_store, ['get_lapack_funcs'], [get_lapack_funcs])

else:
    # Assigning a type to the variable 'scipy.linalg.lapack' (line 7)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), 'scipy.linalg.lapack', import_18405)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/linalg/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 8, 0))

# 'from scipy.linalg.misc import _datacopied' statement (line 8)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/linalg/')
import_18407 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'scipy.linalg.misc')

if (type(import_18407) is not StypyTypeError):

    if (import_18407 != 'pyd_module'):
        __import__(import_18407)
        sys_modules_18408 = sys.modules[import_18407]
        import_from_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'scipy.linalg.misc', sys_modules_18408.module_type_store, module_type_store, ['_datacopied'])
        nest_module(stypy.reporting.localization.Localization(__file__, 8, 0), __file__, sys_modules_18408, sys_modules_18408.module_type_store, module_type_store)
    else:
        from scipy.linalg.misc import _datacopied

        import_from_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'scipy.linalg.misc', None, module_type_store, ['_datacopied'], [_datacopied])

else:
    # Assigning a type to the variable 'scipy.linalg.misc' (line 8)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 0), 'scipy.linalg.misc', import_18407)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/linalg/')


# Assigning a List to a Name (line 10):

# Assigning a List to a Name (line 10):
__all__ = ['qr', 'qr_multiply', 'rq']
module_type_store.set_exportable_members(['qr', 'qr_multiply', 'rq'])

# Obtaining an instance of the builtin type 'list' (line 10)
list_18409 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, 10), 'list')
# Adding type elements to the builtin type 'list' instance (line 10)
# Adding element type (line 10)
str_18410 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, 11), 'str', 'qr')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 10, 10), list_18409, str_18410)
# Adding element type (line 10)
str_18411 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, 17), 'str', 'qr_multiply')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 10, 10), list_18409, str_18411)
# Adding element type (line 10)
str_18412 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, 32), 'str', 'rq')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 10, 10), list_18409, str_18412)

# Assigning a type to the variable '__all__' (line 10)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 0), '__all__', list_18409)

@norecursion
def safecall(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'safecall'
    module_type_store = module_type_store.open_function_context('safecall', 13, 0, False)
    
    # Passed parameters checking function
    safecall.stypy_localization = localization
    safecall.stypy_type_of_self = None
    safecall.stypy_type_store = module_type_store
    safecall.stypy_function_name = 'safecall'
    safecall.stypy_param_names_list = ['f', 'name']
    safecall.stypy_varargs_param_name = 'args'
    safecall.stypy_kwargs_param_name = 'kwargs'
    safecall.stypy_call_defaults = defaults
    safecall.stypy_call_varargs = varargs
    safecall.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'safecall', ['f', 'name'], 'args', 'kwargs', defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'safecall', localization, ['f', 'name'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'safecall(...)' code ##################

    str_18413 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, (-1)), 'str', 'Call a LAPACK routine, determining lwork automatically and handling\n    error return values')
    
    # Assigning a Call to a Name (line 16):
    
    # Assigning a Call to a Name (line 16):
    
    # Call to get(...): (line 16)
    # Processing the call arguments (line 16)
    str_18416 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 23), 'str', 'lwork')
    # Getting the type of 'None' (line 16)
    None_18417 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 32), 'None', False)
    # Processing the call keyword arguments (line 16)
    kwargs_18418 = {}
    # Getting the type of 'kwargs' (line 16)
    kwargs_18414 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 12), 'kwargs', False)
    # Obtaining the member 'get' of a type (line 16)
    get_18415 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 16, 12), kwargs_18414, 'get')
    # Calling get(args, kwargs) (line 16)
    get_call_result_18419 = invoke(stypy.reporting.localization.Localization(__file__, 16, 12), get_18415, *[str_18416, None_18417], **kwargs_18418)
    
    # Assigning a type to the variable 'lwork' (line 16)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 4), 'lwork', get_call_result_18419)
    
    
    # Getting the type of 'lwork' (line 17)
    lwork_18420 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 7), 'lwork')
    
    # Obtaining an instance of the builtin type 'tuple' (line 17)
    tuple_18421 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 17), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 17)
    # Adding element type (line 17)
    # Getting the type of 'None' (line 17)
    None_18422 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 17), 'None')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 17, 17), tuple_18421, None_18422)
    # Adding element type (line 17)
    int_18423 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 23), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 17, 17), tuple_18421, int_18423)
    
    # Applying the binary operator 'in' (line 17)
    result_contains_18424 = python_operator(stypy.reporting.localization.Localization(__file__, 17, 7), 'in', lwork_18420, tuple_18421)
    
    # Testing the type of an if condition (line 17)
    if_condition_18425 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 17, 4), result_contains_18424)
    # Assigning a type to the variable 'if_condition_18425' (line 17)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 17, 4), 'if_condition_18425', if_condition_18425)
    # SSA begins for if statement (line 17)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Num to a Subscript (line 18):
    
    # Assigning a Num to a Subscript (line 18):
    int_18426 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 26), 'int')
    # Getting the type of 'kwargs' (line 18)
    kwargs_18427 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 8), 'kwargs')
    str_18428 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 15), 'str', 'lwork')
    # Storing an element on a container (line 18)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 18, 8), kwargs_18427, (str_18428, int_18426))
    
    # Assigning a Call to a Name (line 19):
    
    # Assigning a Call to a Name (line 19):
    
    # Call to f(...): (line 19)
    # Getting the type of 'args' (line 19)
    args_18430 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 17), 'args', False)
    # Processing the call keyword arguments (line 19)
    # Getting the type of 'kwargs' (line 19)
    kwargs_18431 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 25), 'kwargs', False)
    kwargs_18432 = {'kwargs_18431': kwargs_18431}
    # Getting the type of 'f' (line 19)
    f_18429 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 14), 'f', False)
    # Calling f(args, kwargs) (line 19)
    f_call_result_18433 = invoke(stypy.reporting.localization.Localization(__file__, 19, 14), f_18429, *[args_18430], **kwargs_18432)
    
    # Assigning a type to the variable 'ret' (line 19)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 19, 8), 'ret', f_call_result_18433)
    
    # Assigning a Call to a Subscript (line 20):
    
    # Assigning a Call to a Subscript (line 20):
    
    # Call to astype(...): (line 20)
    # Processing the call arguments (line 20)
    # Getting the type of 'numpy' (line 20)
    numpy_18443 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 49), 'numpy', False)
    # Obtaining the member 'int' of a type (line 20)
    int_18444 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 20, 49), numpy_18443, 'int')
    # Processing the call keyword arguments (line 20)
    kwargs_18445 = {}
    
    # Obtaining the type of the subscript
    int_18434 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 34), 'int')
    
    # Obtaining the type of the subscript
    int_18435 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 30), 'int')
    # Getting the type of 'ret' (line 20)
    ret_18436 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 26), 'ret', False)
    # Obtaining the member '__getitem__' of a type (line 20)
    getitem___18437 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 20, 26), ret_18436, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 20)
    subscript_call_result_18438 = invoke(stypy.reporting.localization.Localization(__file__, 20, 26), getitem___18437, int_18435)
    
    # Obtaining the member '__getitem__' of a type (line 20)
    getitem___18439 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 20, 26), subscript_call_result_18438, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 20)
    subscript_call_result_18440 = invoke(stypy.reporting.localization.Localization(__file__, 20, 26), getitem___18439, int_18434)
    
    # Obtaining the member 'real' of a type (line 20)
    real_18441 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 20, 26), subscript_call_result_18440, 'real')
    # Obtaining the member 'astype' of a type (line 20)
    astype_18442 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 20, 26), real_18441, 'astype')
    # Calling astype(args, kwargs) (line 20)
    astype_call_result_18446 = invoke(stypy.reporting.localization.Localization(__file__, 20, 26), astype_18442, *[int_18444], **kwargs_18445)
    
    # Getting the type of 'kwargs' (line 20)
    kwargs_18447 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 8), 'kwargs')
    str_18448 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 15), 'str', 'lwork')
    # Storing an element on a container (line 20)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 20, 8), kwargs_18447, (str_18448, astype_call_result_18446))
    # SSA join for if statement (line 17)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 21):
    
    # Assigning a Call to a Name (line 21):
    
    # Call to f(...): (line 21)
    # Getting the type of 'args' (line 21)
    args_18450 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 13), 'args', False)
    # Processing the call keyword arguments (line 21)
    # Getting the type of 'kwargs' (line 21)
    kwargs_18451 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 21), 'kwargs', False)
    kwargs_18452 = {'kwargs_18451': kwargs_18451}
    # Getting the type of 'f' (line 21)
    f_18449 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 10), 'f', False)
    # Calling f(args, kwargs) (line 21)
    f_call_result_18453 = invoke(stypy.reporting.localization.Localization(__file__, 21, 10), f_18449, *[args_18450], **kwargs_18452)
    
    # Assigning a type to the variable 'ret' (line 21)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 21, 4), 'ret', f_call_result_18453)
    
    
    
    # Obtaining the type of the subscript
    int_18454 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 11), 'int')
    # Getting the type of 'ret' (line 22)
    ret_18455 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 7), 'ret')
    # Obtaining the member '__getitem__' of a type (line 22)
    getitem___18456 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 22, 7), ret_18455, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 22)
    subscript_call_result_18457 = invoke(stypy.reporting.localization.Localization(__file__, 22, 7), getitem___18456, int_18454)
    
    int_18458 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 17), 'int')
    # Applying the binary operator '<' (line 22)
    result_lt_18459 = python_operator(stypy.reporting.localization.Localization(__file__, 22, 7), '<', subscript_call_result_18457, int_18458)
    
    # Testing the type of an if condition (line 22)
    if_condition_18460 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 22, 4), result_lt_18459)
    # Assigning a type to the variable 'if_condition_18460' (line 22)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 22, 4), 'if_condition_18460', if_condition_18460)
    # SSA begins for if statement (line 22)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 23)
    # Processing the call arguments (line 23)
    str_18462 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 25), 'str', 'illegal value in %d-th argument of internal %s')
    
    # Obtaining an instance of the builtin type 'tuple' (line 24)
    tuple_18463 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 28), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 24)
    # Adding element type (line 24)
    
    
    # Obtaining the type of the subscript
    int_18464 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 33), 'int')
    # Getting the type of 'ret' (line 24)
    ret_18465 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 29), 'ret', False)
    # Obtaining the member '__getitem__' of a type (line 24)
    getitem___18466 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 24, 29), ret_18465, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 24)
    subscript_call_result_18467 = invoke(stypy.reporting.localization.Localization(__file__, 24, 29), getitem___18466, int_18464)
    
    # Applying the 'usub' unary operator (line 24)
    result___neg___18468 = python_operator(stypy.reporting.localization.Localization(__file__, 24, 28), 'usub', subscript_call_result_18467)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 24, 28), tuple_18463, result___neg___18468)
    # Adding element type (line 24)
    # Getting the type of 'name' (line 24)
    name_18469 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 38), 'name', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 24, 28), tuple_18463, name_18469)
    
    # Applying the binary operator '%' (line 23)
    result_mod_18470 = python_operator(stypy.reporting.localization.Localization(__file__, 23, 25), '%', str_18462, tuple_18463)
    
    # Processing the call keyword arguments (line 23)
    kwargs_18471 = {}
    # Getting the type of 'ValueError' (line 23)
    ValueError_18461 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 23)
    ValueError_call_result_18472 = invoke(stypy.reporting.localization.Localization(__file__, 23, 14), ValueError_18461, *[result_mod_18470], **kwargs_18471)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 23, 8), ValueError_call_result_18472, 'raise parameter', BaseException)
    # SSA join for if statement (line 22)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Obtaining the type of the subscript
    int_18473 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 16), 'int')
    slice_18474 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 25, 11), None, int_18473, None)
    # Getting the type of 'ret' (line 25)
    ret_18475 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 11), 'ret')
    # Obtaining the member '__getitem__' of a type (line 25)
    getitem___18476 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 25, 11), ret_18475, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 25)
    subscript_call_result_18477 = invoke(stypy.reporting.localization.Localization(__file__, 25, 11), getitem___18476, slice_18474)
    
    # Assigning a type to the variable 'stypy_return_type' (line 25)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 25, 4), 'stypy_return_type', subscript_call_result_18477)
    
    # ################# End of 'safecall(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'safecall' in the type store
    # Getting the type of 'stypy_return_type' (line 13)
    stypy_return_type_18478 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_18478)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'safecall'
    return stypy_return_type_18478

# Assigning a type to the variable 'safecall' (line 13)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 0), 'safecall', safecall)

@norecursion
def qr(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'False' (line 28)
    False_18479 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 22), 'False')
    # Getting the type of 'None' (line 28)
    None_18480 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 35), 'None')
    str_18481 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 46), 'str', 'full')
    # Getting the type of 'False' (line 28)
    False_18482 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 63), 'False')
    # Getting the type of 'True' (line 29)
    True_18483 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 20), 'True')
    defaults = [False_18479, None_18480, str_18481, False_18482, True_18483]
    # Create a new context for function 'qr'
    module_type_store = module_type_store.open_function_context('qr', 28, 0, False)
    
    # Passed parameters checking function
    qr.stypy_localization = localization
    qr.stypy_type_of_self = None
    qr.stypy_type_store = module_type_store
    qr.stypy_function_name = 'qr'
    qr.stypy_param_names_list = ['a', 'overwrite_a', 'lwork', 'mode', 'pivoting', 'check_finite']
    qr.stypy_varargs_param_name = None
    qr.stypy_kwargs_param_name = None
    qr.stypy_call_defaults = defaults
    qr.stypy_call_varargs = varargs
    qr.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'qr', ['a', 'overwrite_a', 'lwork', 'mode', 'pivoting', 'check_finite'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'qr', localization, ['a', 'overwrite_a', 'lwork', 'mode', 'pivoting', 'check_finite'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'qr(...)' code ##################

    str_18484 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 117, (-1)), 'str', "\n    Compute QR decomposition of a matrix.\n\n    Calculate the decomposition ``A = Q R`` where Q is unitary/orthogonal\n    and R upper triangular.\n\n    Parameters\n    ----------\n    a : (M, N) array_like\n        Matrix to be decomposed\n    overwrite_a : bool, optional\n        Whether data in a is overwritten (may improve performance)\n    lwork : int, optional\n        Work array size, lwork >= a.shape[1]. If None or -1, an optimal size\n        is computed.\n    mode : {'full', 'r', 'economic', 'raw'}, optional\n        Determines what information is to be returned: either both Q and R\n        ('full', default), only R ('r') or both Q and R but computed in\n        economy-size ('economic', see Notes). The final option 'raw'\n        (added in Scipy 0.11) makes the function return two matrices\n        (Q, TAU) in the internal format used by LAPACK.\n    pivoting : bool, optional\n        Whether or not factorization should include pivoting for rank-revealing\n        qr decomposition. If pivoting, compute the decomposition\n        ``A P = Q R`` as above, but where P is chosen such that the diagonal\n        of R is non-increasing.\n    check_finite : bool, optional\n        Whether to check that the input matrix contains only finite numbers.\n        Disabling may give a performance gain, but may result in problems\n        (crashes, non-termination) if the inputs do contain infinities or NaNs.\n\n    Returns\n    -------\n    Q : float or complex ndarray\n        Of shape (M, M), or (M, K) for ``mode='economic'``.  Not returned\n        if ``mode='r'``.\n    R : float or complex ndarray\n        Of shape (M, N), or (K, N) for ``mode='economic'``.  ``K = min(M, N)``.\n    P : int ndarray\n        Of shape (N,) for ``pivoting=True``. Not returned if\n        ``pivoting=False``.\n\n    Raises\n    ------\n    LinAlgError\n        Raised if decomposition fails\n\n    Notes\n    -----\n    This is an interface to the LAPACK routines dgeqrf, zgeqrf,\n    dorgqr, zungqr, dgeqp3, and zgeqp3.\n\n    If ``mode=economic``, the shapes of Q and R are (M, K) and (K, N) instead\n    of (M,M) and (M,N), with ``K=min(M,N)``.\n\n    Examples\n    --------\n    >>> from scipy import random, linalg, dot, diag, all, allclose\n    >>> a = random.randn(9, 6)\n\n    >>> q, r = linalg.qr(a)\n    >>> allclose(a, np.dot(q, r))\n    True\n    >>> q.shape, r.shape\n    ((9, 9), (9, 6))\n\n    >>> r2 = linalg.qr(a, mode='r')\n    >>> allclose(r, r2)\n    True\n\n    >>> q3, r3 = linalg.qr(a, mode='economic')\n    >>> q3.shape, r3.shape\n    ((9, 6), (6, 6))\n\n    >>> q4, r4, p4 = linalg.qr(a, pivoting=True)\n    >>> d = abs(diag(r4))\n    >>> all(d[1:] <= d[:-1])\n    True\n    >>> allclose(a[:, p4], dot(q4, r4))\n    True\n    >>> q4.shape, r4.shape, p4.shape\n    ((9, 9), (9, 6), (6,))\n\n    >>> q5, r5, p5 = linalg.qr(a, mode='economic', pivoting=True)\n    >>> q5.shape, r5.shape, p5.shape\n    ((9, 6), (6, 6), (6,))\n\n    ")
    
    
    # Getting the type of 'mode' (line 121)
    mode_18485 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 7), 'mode')
    
    # Obtaining an instance of the builtin type 'list' (line 121)
    list_18486 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 121, 19), 'list')
    # Adding type elements to the builtin type 'list' instance (line 121)
    # Adding element type (line 121)
    str_18487 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 121, 20), 'str', 'full')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 121, 19), list_18486, str_18487)
    # Adding element type (line 121)
    str_18488 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 121, 28), 'str', 'qr')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 121, 19), list_18486, str_18488)
    # Adding element type (line 121)
    str_18489 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 121, 34), 'str', 'r')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 121, 19), list_18486, str_18489)
    # Adding element type (line 121)
    str_18490 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 121, 39), 'str', 'economic')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 121, 19), list_18486, str_18490)
    # Adding element type (line 121)
    str_18491 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 121, 51), 'str', 'raw')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 121, 19), list_18486, str_18491)
    
    # Applying the binary operator 'notin' (line 121)
    result_contains_18492 = python_operator(stypy.reporting.localization.Localization(__file__, 121, 7), 'notin', mode_18485, list_18486)
    
    # Testing the type of an if condition (line 121)
    if_condition_18493 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 121, 4), result_contains_18492)
    # Assigning a type to the variable 'if_condition_18493' (line 121)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 121, 4), 'if_condition_18493', if_condition_18493)
    # SSA begins for if statement (line 121)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 122)
    # Processing the call arguments (line 122)
    str_18495 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 123, 17), 'str', "Mode argument should be one of ['full', 'r', 'economic', 'raw']")
    # Processing the call keyword arguments (line 122)
    kwargs_18496 = {}
    # Getting the type of 'ValueError' (line 122)
    ValueError_18494 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 122)
    ValueError_call_result_18497 = invoke(stypy.reporting.localization.Localization(__file__, 122, 14), ValueError_18494, *[str_18495], **kwargs_18496)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 122, 8), ValueError_call_result_18497, 'raise parameter', BaseException)
    # SSA join for if statement (line 121)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Getting the type of 'check_finite' (line 125)
    check_finite_18498 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 7), 'check_finite')
    # Testing the type of an if condition (line 125)
    if_condition_18499 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 125, 4), check_finite_18498)
    # Assigning a type to the variable 'if_condition_18499' (line 125)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 125, 4), 'if_condition_18499', if_condition_18499)
    # SSA begins for if statement (line 125)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 126):
    
    # Assigning a Call to a Name (line 126):
    
    # Call to asarray_chkfinite(...): (line 126)
    # Processing the call arguments (line 126)
    # Getting the type of 'a' (line 126)
    a_18502 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 37), 'a', False)
    # Processing the call keyword arguments (line 126)
    kwargs_18503 = {}
    # Getting the type of 'numpy' (line 126)
    numpy_18500 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 13), 'numpy', False)
    # Obtaining the member 'asarray_chkfinite' of a type (line 126)
    asarray_chkfinite_18501 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 126, 13), numpy_18500, 'asarray_chkfinite')
    # Calling asarray_chkfinite(args, kwargs) (line 126)
    asarray_chkfinite_call_result_18504 = invoke(stypy.reporting.localization.Localization(__file__, 126, 13), asarray_chkfinite_18501, *[a_18502], **kwargs_18503)
    
    # Assigning a type to the variable 'a1' (line 126)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 126, 8), 'a1', asarray_chkfinite_call_result_18504)
    # SSA branch for the else part of an if statement (line 125)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Call to a Name (line 128):
    
    # Assigning a Call to a Name (line 128):
    
    # Call to asarray(...): (line 128)
    # Processing the call arguments (line 128)
    # Getting the type of 'a' (line 128)
    a_18507 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 27), 'a', False)
    # Processing the call keyword arguments (line 128)
    kwargs_18508 = {}
    # Getting the type of 'numpy' (line 128)
    numpy_18505 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 13), 'numpy', False)
    # Obtaining the member 'asarray' of a type (line 128)
    asarray_18506 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 128, 13), numpy_18505, 'asarray')
    # Calling asarray(args, kwargs) (line 128)
    asarray_call_result_18509 = invoke(stypy.reporting.localization.Localization(__file__, 128, 13), asarray_18506, *[a_18507], **kwargs_18508)
    
    # Assigning a type to the variable 'a1' (line 128)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 128, 8), 'a1', asarray_call_result_18509)
    # SSA join for if statement (line 125)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    
    # Call to len(...): (line 129)
    # Processing the call arguments (line 129)
    # Getting the type of 'a1' (line 129)
    a1_18511 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 11), 'a1', False)
    # Obtaining the member 'shape' of a type (line 129)
    shape_18512 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 129, 11), a1_18511, 'shape')
    # Processing the call keyword arguments (line 129)
    kwargs_18513 = {}
    # Getting the type of 'len' (line 129)
    len_18510 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 7), 'len', False)
    # Calling len(args, kwargs) (line 129)
    len_call_result_18514 = invoke(stypy.reporting.localization.Localization(__file__, 129, 7), len_18510, *[shape_18512], **kwargs_18513)
    
    int_18515 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 129, 24), 'int')
    # Applying the binary operator '!=' (line 129)
    result_ne_18516 = python_operator(stypy.reporting.localization.Localization(__file__, 129, 7), '!=', len_call_result_18514, int_18515)
    
    # Testing the type of an if condition (line 129)
    if_condition_18517 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 129, 4), result_ne_18516)
    # Assigning a type to the variable 'if_condition_18517' (line 129)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 129, 4), 'if_condition_18517', if_condition_18517)
    # SSA begins for if statement (line 129)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 130)
    # Processing the call arguments (line 130)
    str_18519 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 130, 25), 'str', 'expected 2D array')
    # Processing the call keyword arguments (line 130)
    kwargs_18520 = {}
    # Getting the type of 'ValueError' (line 130)
    ValueError_18518 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 130)
    ValueError_call_result_18521 = invoke(stypy.reporting.localization.Localization(__file__, 130, 14), ValueError_18518, *[str_18519], **kwargs_18520)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 130, 8), ValueError_call_result_18521, 'raise parameter', BaseException)
    # SSA join for if statement (line 129)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Attribute to a Tuple (line 131):
    
    # Assigning a Subscript to a Name (line 131):
    
    # Obtaining the type of the subscript
    int_18522 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 131, 4), 'int')
    # Getting the type of 'a1' (line 131)
    a1_18523 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 11), 'a1')
    # Obtaining the member 'shape' of a type (line 131)
    shape_18524 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 131, 11), a1_18523, 'shape')
    # Obtaining the member '__getitem__' of a type (line 131)
    getitem___18525 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 131, 4), shape_18524, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 131)
    subscript_call_result_18526 = invoke(stypy.reporting.localization.Localization(__file__, 131, 4), getitem___18525, int_18522)
    
    # Assigning a type to the variable 'tuple_var_assignment_18374' (line 131)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 131, 4), 'tuple_var_assignment_18374', subscript_call_result_18526)
    
    # Assigning a Subscript to a Name (line 131):
    
    # Obtaining the type of the subscript
    int_18527 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 131, 4), 'int')
    # Getting the type of 'a1' (line 131)
    a1_18528 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 11), 'a1')
    # Obtaining the member 'shape' of a type (line 131)
    shape_18529 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 131, 11), a1_18528, 'shape')
    # Obtaining the member '__getitem__' of a type (line 131)
    getitem___18530 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 131, 4), shape_18529, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 131)
    subscript_call_result_18531 = invoke(stypy.reporting.localization.Localization(__file__, 131, 4), getitem___18530, int_18527)
    
    # Assigning a type to the variable 'tuple_var_assignment_18375' (line 131)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 131, 4), 'tuple_var_assignment_18375', subscript_call_result_18531)
    
    # Assigning a Name to a Name (line 131):
    # Getting the type of 'tuple_var_assignment_18374' (line 131)
    tuple_var_assignment_18374_18532 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 4), 'tuple_var_assignment_18374')
    # Assigning a type to the variable 'M' (line 131)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 131, 4), 'M', tuple_var_assignment_18374_18532)
    
    # Assigning a Name to a Name (line 131):
    # Getting the type of 'tuple_var_assignment_18375' (line 131)
    tuple_var_assignment_18375_18533 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 4), 'tuple_var_assignment_18375')
    # Assigning a type to the variable 'N' (line 131)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 131, 7), 'N', tuple_var_assignment_18375_18533)
    
    # Assigning a BoolOp to a Name (line 132):
    
    # Assigning a BoolOp to a Name (line 132):
    
    # Evaluating a boolean operation
    # Getting the type of 'overwrite_a' (line 132)
    overwrite_a_18534 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 18), 'overwrite_a')
    
    # Call to _datacopied(...): (line 132)
    # Processing the call arguments (line 132)
    # Getting the type of 'a1' (line 132)
    a1_18536 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 46), 'a1', False)
    # Getting the type of 'a' (line 132)
    a_18537 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 50), 'a', False)
    # Processing the call keyword arguments (line 132)
    kwargs_18538 = {}
    # Getting the type of '_datacopied' (line 132)
    _datacopied_18535 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 34), '_datacopied', False)
    # Calling _datacopied(args, kwargs) (line 132)
    _datacopied_call_result_18539 = invoke(stypy.reporting.localization.Localization(__file__, 132, 34), _datacopied_18535, *[a1_18536, a_18537], **kwargs_18538)
    
    # Applying the binary operator 'or' (line 132)
    result_or_keyword_18540 = python_operator(stypy.reporting.localization.Localization(__file__, 132, 18), 'or', overwrite_a_18534, _datacopied_call_result_18539)
    
    # Assigning a type to the variable 'overwrite_a' (line 132)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 132, 4), 'overwrite_a', result_or_keyword_18540)
    
    # Getting the type of 'pivoting' (line 134)
    pivoting_18541 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 7), 'pivoting')
    # Testing the type of an if condition (line 134)
    if_condition_18542 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 134, 4), pivoting_18541)
    # Assigning a type to the variable 'if_condition_18542' (line 134)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 134, 4), 'if_condition_18542', if_condition_18542)
    # SSA begins for if statement (line 134)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Tuple (line 135):
    
    # Assigning a Subscript to a Name (line 135):
    
    # Obtaining the type of the subscript
    int_18543 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 135, 8), 'int')
    
    # Call to get_lapack_funcs(...): (line 135)
    # Processing the call arguments (line 135)
    
    # Obtaining an instance of the builtin type 'tuple' (line 135)
    tuple_18545 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 135, 35), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 135)
    # Adding element type (line 135)
    str_18546 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 135, 35), 'str', 'geqp3')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 135, 35), tuple_18545, str_18546)
    
    
    # Obtaining an instance of the builtin type 'tuple' (line 135)
    tuple_18547 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 135, 47), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 135)
    # Adding element type (line 135)
    # Getting the type of 'a1' (line 135)
    a1_18548 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 47), 'a1', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 135, 47), tuple_18547, a1_18548)
    
    # Processing the call keyword arguments (line 135)
    kwargs_18549 = {}
    # Getting the type of 'get_lapack_funcs' (line 135)
    get_lapack_funcs_18544 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 17), 'get_lapack_funcs', False)
    # Calling get_lapack_funcs(args, kwargs) (line 135)
    get_lapack_funcs_call_result_18550 = invoke(stypy.reporting.localization.Localization(__file__, 135, 17), get_lapack_funcs_18544, *[tuple_18545, tuple_18547], **kwargs_18549)
    
    # Obtaining the member '__getitem__' of a type (line 135)
    getitem___18551 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 135, 8), get_lapack_funcs_call_result_18550, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 135)
    subscript_call_result_18552 = invoke(stypy.reporting.localization.Localization(__file__, 135, 8), getitem___18551, int_18543)
    
    # Assigning a type to the variable 'tuple_var_assignment_18376' (line 135)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 135, 8), 'tuple_var_assignment_18376', subscript_call_result_18552)
    
    # Assigning a Name to a Name (line 135):
    # Getting the type of 'tuple_var_assignment_18376' (line 135)
    tuple_var_assignment_18376_18553 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 8), 'tuple_var_assignment_18376')
    # Assigning a type to the variable 'geqp3' (line 135)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 135, 8), 'geqp3', tuple_var_assignment_18376_18553)
    
    # Assigning a Call to a Tuple (line 136):
    
    # Assigning a Subscript to a Name (line 136):
    
    # Obtaining the type of the subscript
    int_18554 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 136, 8), 'int')
    
    # Call to safecall(...): (line 136)
    # Processing the call arguments (line 136)
    # Getting the type of 'geqp3' (line 136)
    geqp3_18556 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 33), 'geqp3', False)
    str_18557 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 136, 40), 'str', 'geqp3')
    # Getting the type of 'a1' (line 136)
    a1_18558 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 49), 'a1', False)
    # Processing the call keyword arguments (line 136)
    # Getting the type of 'overwrite_a' (line 136)
    overwrite_a_18559 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 65), 'overwrite_a', False)
    keyword_18560 = overwrite_a_18559
    kwargs_18561 = {'overwrite_a': keyword_18560}
    # Getting the type of 'safecall' (line 136)
    safecall_18555 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 24), 'safecall', False)
    # Calling safecall(args, kwargs) (line 136)
    safecall_call_result_18562 = invoke(stypy.reporting.localization.Localization(__file__, 136, 24), safecall_18555, *[geqp3_18556, str_18557, a1_18558], **kwargs_18561)
    
    # Obtaining the member '__getitem__' of a type (line 136)
    getitem___18563 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 136, 8), safecall_call_result_18562, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 136)
    subscript_call_result_18564 = invoke(stypy.reporting.localization.Localization(__file__, 136, 8), getitem___18563, int_18554)
    
    # Assigning a type to the variable 'tuple_var_assignment_18377' (line 136)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 136, 8), 'tuple_var_assignment_18377', subscript_call_result_18564)
    
    # Assigning a Subscript to a Name (line 136):
    
    # Obtaining the type of the subscript
    int_18565 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 136, 8), 'int')
    
    # Call to safecall(...): (line 136)
    # Processing the call arguments (line 136)
    # Getting the type of 'geqp3' (line 136)
    geqp3_18567 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 33), 'geqp3', False)
    str_18568 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 136, 40), 'str', 'geqp3')
    # Getting the type of 'a1' (line 136)
    a1_18569 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 49), 'a1', False)
    # Processing the call keyword arguments (line 136)
    # Getting the type of 'overwrite_a' (line 136)
    overwrite_a_18570 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 65), 'overwrite_a', False)
    keyword_18571 = overwrite_a_18570
    kwargs_18572 = {'overwrite_a': keyword_18571}
    # Getting the type of 'safecall' (line 136)
    safecall_18566 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 24), 'safecall', False)
    # Calling safecall(args, kwargs) (line 136)
    safecall_call_result_18573 = invoke(stypy.reporting.localization.Localization(__file__, 136, 24), safecall_18566, *[geqp3_18567, str_18568, a1_18569], **kwargs_18572)
    
    # Obtaining the member '__getitem__' of a type (line 136)
    getitem___18574 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 136, 8), safecall_call_result_18573, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 136)
    subscript_call_result_18575 = invoke(stypy.reporting.localization.Localization(__file__, 136, 8), getitem___18574, int_18565)
    
    # Assigning a type to the variable 'tuple_var_assignment_18378' (line 136)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 136, 8), 'tuple_var_assignment_18378', subscript_call_result_18575)
    
    # Assigning a Subscript to a Name (line 136):
    
    # Obtaining the type of the subscript
    int_18576 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 136, 8), 'int')
    
    # Call to safecall(...): (line 136)
    # Processing the call arguments (line 136)
    # Getting the type of 'geqp3' (line 136)
    geqp3_18578 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 33), 'geqp3', False)
    str_18579 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 136, 40), 'str', 'geqp3')
    # Getting the type of 'a1' (line 136)
    a1_18580 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 49), 'a1', False)
    # Processing the call keyword arguments (line 136)
    # Getting the type of 'overwrite_a' (line 136)
    overwrite_a_18581 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 65), 'overwrite_a', False)
    keyword_18582 = overwrite_a_18581
    kwargs_18583 = {'overwrite_a': keyword_18582}
    # Getting the type of 'safecall' (line 136)
    safecall_18577 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 24), 'safecall', False)
    # Calling safecall(args, kwargs) (line 136)
    safecall_call_result_18584 = invoke(stypy.reporting.localization.Localization(__file__, 136, 24), safecall_18577, *[geqp3_18578, str_18579, a1_18580], **kwargs_18583)
    
    # Obtaining the member '__getitem__' of a type (line 136)
    getitem___18585 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 136, 8), safecall_call_result_18584, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 136)
    subscript_call_result_18586 = invoke(stypy.reporting.localization.Localization(__file__, 136, 8), getitem___18585, int_18576)
    
    # Assigning a type to the variable 'tuple_var_assignment_18379' (line 136)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 136, 8), 'tuple_var_assignment_18379', subscript_call_result_18586)
    
    # Assigning a Name to a Name (line 136):
    # Getting the type of 'tuple_var_assignment_18377' (line 136)
    tuple_var_assignment_18377_18587 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 8), 'tuple_var_assignment_18377')
    # Assigning a type to the variable 'qr' (line 136)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 136, 8), 'qr', tuple_var_assignment_18377_18587)
    
    # Assigning a Name to a Name (line 136):
    # Getting the type of 'tuple_var_assignment_18378' (line 136)
    tuple_var_assignment_18378_18588 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 8), 'tuple_var_assignment_18378')
    # Assigning a type to the variable 'jpvt' (line 136)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 136, 12), 'jpvt', tuple_var_assignment_18378_18588)
    
    # Assigning a Name to a Name (line 136):
    # Getting the type of 'tuple_var_assignment_18379' (line 136)
    tuple_var_assignment_18379_18589 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 8), 'tuple_var_assignment_18379')
    # Assigning a type to the variable 'tau' (line 136)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 136, 18), 'tau', tuple_var_assignment_18379_18589)
    
    # Getting the type of 'jpvt' (line 137)
    jpvt_18590 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 8), 'jpvt')
    int_18591 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 137, 16), 'int')
    # Applying the binary operator '-=' (line 137)
    result_isub_18592 = python_operator(stypy.reporting.localization.Localization(__file__, 137, 8), '-=', jpvt_18590, int_18591)
    # Assigning a type to the variable 'jpvt' (line 137)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 137, 8), 'jpvt', result_isub_18592)
    
    # SSA branch for the else part of an if statement (line 134)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Call to a Tuple (line 139):
    
    # Assigning a Subscript to a Name (line 139):
    
    # Obtaining the type of the subscript
    int_18593 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 139, 8), 'int')
    
    # Call to get_lapack_funcs(...): (line 139)
    # Processing the call arguments (line 139)
    
    # Obtaining an instance of the builtin type 'tuple' (line 139)
    tuple_18595 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 139, 35), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 139)
    # Adding element type (line 139)
    str_18596 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 139, 35), 'str', 'geqrf')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 139, 35), tuple_18595, str_18596)
    
    
    # Obtaining an instance of the builtin type 'tuple' (line 139)
    tuple_18597 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 139, 47), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 139)
    # Adding element type (line 139)
    # Getting the type of 'a1' (line 139)
    a1_18598 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 47), 'a1', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 139, 47), tuple_18597, a1_18598)
    
    # Processing the call keyword arguments (line 139)
    kwargs_18599 = {}
    # Getting the type of 'get_lapack_funcs' (line 139)
    get_lapack_funcs_18594 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 17), 'get_lapack_funcs', False)
    # Calling get_lapack_funcs(args, kwargs) (line 139)
    get_lapack_funcs_call_result_18600 = invoke(stypy.reporting.localization.Localization(__file__, 139, 17), get_lapack_funcs_18594, *[tuple_18595, tuple_18597], **kwargs_18599)
    
    # Obtaining the member '__getitem__' of a type (line 139)
    getitem___18601 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 139, 8), get_lapack_funcs_call_result_18600, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 139)
    subscript_call_result_18602 = invoke(stypy.reporting.localization.Localization(__file__, 139, 8), getitem___18601, int_18593)
    
    # Assigning a type to the variable 'tuple_var_assignment_18380' (line 139)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 139, 8), 'tuple_var_assignment_18380', subscript_call_result_18602)
    
    # Assigning a Name to a Name (line 139):
    # Getting the type of 'tuple_var_assignment_18380' (line 139)
    tuple_var_assignment_18380_18603 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 8), 'tuple_var_assignment_18380')
    # Assigning a type to the variable 'geqrf' (line 139)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 139, 8), 'geqrf', tuple_var_assignment_18380_18603)
    
    # Assigning a Call to a Tuple (line 140):
    
    # Assigning a Subscript to a Name (line 140):
    
    # Obtaining the type of the subscript
    int_18604 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 140, 8), 'int')
    
    # Call to safecall(...): (line 140)
    # Processing the call arguments (line 140)
    # Getting the type of 'geqrf' (line 140)
    geqrf_18606 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 27), 'geqrf', False)
    str_18607 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 140, 34), 'str', 'geqrf')
    # Getting the type of 'a1' (line 140)
    a1_18608 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 43), 'a1', False)
    # Processing the call keyword arguments (line 140)
    # Getting the type of 'lwork' (line 140)
    lwork_18609 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 53), 'lwork', False)
    keyword_18610 = lwork_18609
    # Getting the type of 'overwrite_a' (line 141)
    overwrite_a_18611 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 24), 'overwrite_a', False)
    keyword_18612 = overwrite_a_18611
    kwargs_18613 = {'overwrite_a': keyword_18612, 'lwork': keyword_18610}
    # Getting the type of 'safecall' (line 140)
    safecall_18605 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 18), 'safecall', False)
    # Calling safecall(args, kwargs) (line 140)
    safecall_call_result_18614 = invoke(stypy.reporting.localization.Localization(__file__, 140, 18), safecall_18605, *[geqrf_18606, str_18607, a1_18608], **kwargs_18613)
    
    # Obtaining the member '__getitem__' of a type (line 140)
    getitem___18615 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 140, 8), safecall_call_result_18614, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 140)
    subscript_call_result_18616 = invoke(stypy.reporting.localization.Localization(__file__, 140, 8), getitem___18615, int_18604)
    
    # Assigning a type to the variable 'tuple_var_assignment_18381' (line 140)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 140, 8), 'tuple_var_assignment_18381', subscript_call_result_18616)
    
    # Assigning a Subscript to a Name (line 140):
    
    # Obtaining the type of the subscript
    int_18617 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 140, 8), 'int')
    
    # Call to safecall(...): (line 140)
    # Processing the call arguments (line 140)
    # Getting the type of 'geqrf' (line 140)
    geqrf_18619 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 27), 'geqrf', False)
    str_18620 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 140, 34), 'str', 'geqrf')
    # Getting the type of 'a1' (line 140)
    a1_18621 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 43), 'a1', False)
    # Processing the call keyword arguments (line 140)
    # Getting the type of 'lwork' (line 140)
    lwork_18622 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 53), 'lwork', False)
    keyword_18623 = lwork_18622
    # Getting the type of 'overwrite_a' (line 141)
    overwrite_a_18624 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 24), 'overwrite_a', False)
    keyword_18625 = overwrite_a_18624
    kwargs_18626 = {'overwrite_a': keyword_18625, 'lwork': keyword_18623}
    # Getting the type of 'safecall' (line 140)
    safecall_18618 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 18), 'safecall', False)
    # Calling safecall(args, kwargs) (line 140)
    safecall_call_result_18627 = invoke(stypy.reporting.localization.Localization(__file__, 140, 18), safecall_18618, *[geqrf_18619, str_18620, a1_18621], **kwargs_18626)
    
    # Obtaining the member '__getitem__' of a type (line 140)
    getitem___18628 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 140, 8), safecall_call_result_18627, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 140)
    subscript_call_result_18629 = invoke(stypy.reporting.localization.Localization(__file__, 140, 8), getitem___18628, int_18617)
    
    # Assigning a type to the variable 'tuple_var_assignment_18382' (line 140)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 140, 8), 'tuple_var_assignment_18382', subscript_call_result_18629)
    
    # Assigning a Name to a Name (line 140):
    # Getting the type of 'tuple_var_assignment_18381' (line 140)
    tuple_var_assignment_18381_18630 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 8), 'tuple_var_assignment_18381')
    # Assigning a type to the variable 'qr' (line 140)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 140, 8), 'qr', tuple_var_assignment_18381_18630)
    
    # Assigning a Name to a Name (line 140):
    # Getting the type of 'tuple_var_assignment_18382' (line 140)
    tuple_var_assignment_18382_18631 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 8), 'tuple_var_assignment_18382')
    # Assigning a type to the variable 'tau' (line 140)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 140, 12), 'tau', tuple_var_assignment_18382_18631)
    # SSA join for if statement (line 134)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'mode' (line 143)
    mode_18632 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 7), 'mode')
    
    # Obtaining an instance of the builtin type 'list' (line 143)
    list_18633 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 143, 19), 'list')
    # Adding type elements to the builtin type 'list' instance (line 143)
    # Adding element type (line 143)
    str_18634 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 143, 20), 'str', 'economic')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 143, 19), list_18633, str_18634)
    # Adding element type (line 143)
    str_18635 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 143, 32), 'str', 'raw')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 143, 19), list_18633, str_18635)
    
    # Applying the binary operator 'notin' (line 143)
    result_contains_18636 = python_operator(stypy.reporting.localization.Localization(__file__, 143, 7), 'notin', mode_18632, list_18633)
    
    
    # Getting the type of 'M' (line 143)
    M_18637 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 42), 'M')
    # Getting the type of 'N' (line 143)
    N_18638 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 46), 'N')
    # Applying the binary operator '<' (line 143)
    result_lt_18639 = python_operator(stypy.reporting.localization.Localization(__file__, 143, 42), '<', M_18637, N_18638)
    
    # Applying the binary operator 'or' (line 143)
    result_or_keyword_18640 = python_operator(stypy.reporting.localization.Localization(__file__, 143, 7), 'or', result_contains_18636, result_lt_18639)
    
    # Testing the type of an if condition (line 143)
    if_condition_18641 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 143, 4), result_or_keyword_18640)
    # Assigning a type to the variable 'if_condition_18641' (line 143)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 143, 4), 'if_condition_18641', if_condition_18641)
    # SSA begins for if statement (line 143)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 144):
    
    # Assigning a Call to a Name (line 144):
    
    # Call to triu(...): (line 144)
    # Processing the call arguments (line 144)
    # Getting the type of 'qr' (line 144)
    qr_18644 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 23), 'qr', False)
    # Processing the call keyword arguments (line 144)
    kwargs_18645 = {}
    # Getting the type of 'numpy' (line 144)
    numpy_18642 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 12), 'numpy', False)
    # Obtaining the member 'triu' of a type (line 144)
    triu_18643 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 144, 12), numpy_18642, 'triu')
    # Calling triu(args, kwargs) (line 144)
    triu_call_result_18646 = invoke(stypy.reporting.localization.Localization(__file__, 144, 12), triu_18643, *[qr_18644], **kwargs_18645)
    
    # Assigning a type to the variable 'R' (line 144)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 144, 8), 'R', triu_call_result_18646)
    # SSA branch for the else part of an if statement (line 143)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Call to a Name (line 146):
    
    # Assigning a Call to a Name (line 146):
    
    # Call to triu(...): (line 146)
    # Processing the call arguments (line 146)
    
    # Obtaining the type of the subscript
    # Getting the type of 'N' (line 146)
    N_18649 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 27), 'N', False)
    slice_18650 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 146, 23), None, N_18649, None)
    slice_18651 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 146, 23), None, None, None)
    # Getting the type of 'qr' (line 146)
    qr_18652 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 23), 'qr', False)
    # Obtaining the member '__getitem__' of a type (line 146)
    getitem___18653 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 146, 23), qr_18652, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 146)
    subscript_call_result_18654 = invoke(stypy.reporting.localization.Localization(__file__, 146, 23), getitem___18653, (slice_18650, slice_18651))
    
    # Processing the call keyword arguments (line 146)
    kwargs_18655 = {}
    # Getting the type of 'numpy' (line 146)
    numpy_18647 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 12), 'numpy', False)
    # Obtaining the member 'triu' of a type (line 146)
    triu_18648 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 146, 12), numpy_18647, 'triu')
    # Calling triu(args, kwargs) (line 146)
    triu_call_result_18656 = invoke(stypy.reporting.localization.Localization(__file__, 146, 12), triu_18648, *[subscript_call_result_18654], **kwargs_18655)
    
    # Assigning a type to the variable 'R' (line 146)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 146, 8), 'R', triu_call_result_18656)
    # SSA join for if statement (line 143)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Getting the type of 'pivoting' (line 148)
    pivoting_18657 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 7), 'pivoting')
    # Testing the type of an if condition (line 148)
    if_condition_18658 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 148, 4), pivoting_18657)
    # Assigning a type to the variable 'if_condition_18658' (line 148)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 148, 4), 'if_condition_18658', if_condition_18658)
    # SSA begins for if statement (line 148)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Tuple to a Name (line 149):
    
    # Assigning a Tuple to a Name (line 149):
    
    # Obtaining an instance of the builtin type 'tuple' (line 149)
    tuple_18659 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 149, 13), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 149)
    # Adding element type (line 149)
    # Getting the type of 'R' (line 149)
    R_18660 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 13), 'R')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 149, 13), tuple_18659, R_18660)
    # Adding element type (line 149)
    # Getting the type of 'jpvt' (line 149)
    jpvt_18661 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 16), 'jpvt')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 149, 13), tuple_18659, jpvt_18661)
    
    # Assigning a type to the variable 'Rj' (line 149)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 149, 8), 'Rj', tuple_18659)
    # SSA branch for the else part of an if statement (line 148)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Tuple to a Name (line 151):
    
    # Assigning a Tuple to a Name (line 151):
    
    # Obtaining an instance of the builtin type 'tuple' (line 151)
    tuple_18662 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 151, 13), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 151)
    # Adding element type (line 151)
    # Getting the type of 'R' (line 151)
    R_18663 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 13), 'R')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 151, 13), tuple_18662, R_18663)
    
    # Assigning a type to the variable 'Rj' (line 151)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 151, 8), 'Rj', tuple_18662)
    # SSA join for if statement (line 148)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'mode' (line 153)
    mode_18664 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 7), 'mode')
    str_18665 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 153, 15), 'str', 'r')
    # Applying the binary operator '==' (line 153)
    result_eq_18666 = python_operator(stypy.reporting.localization.Localization(__file__, 153, 7), '==', mode_18664, str_18665)
    
    # Testing the type of an if condition (line 153)
    if_condition_18667 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 153, 4), result_eq_18666)
    # Assigning a type to the variable 'if_condition_18667' (line 153)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 153, 4), 'if_condition_18667', if_condition_18667)
    # SSA begins for if statement (line 153)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # Getting the type of 'Rj' (line 154)
    Rj_18668 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 15), 'Rj')
    # Assigning a type to the variable 'stypy_return_type' (line 154)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 154, 8), 'stypy_return_type', Rj_18668)
    # SSA branch for the else part of an if statement (line 153)
    module_type_store.open_ssa_branch('else')
    
    
    # Getting the type of 'mode' (line 155)
    mode_18669 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 9), 'mode')
    str_18670 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 155, 17), 'str', 'raw')
    # Applying the binary operator '==' (line 155)
    result_eq_18671 = python_operator(stypy.reporting.localization.Localization(__file__, 155, 9), '==', mode_18669, str_18670)
    
    # Testing the type of an if condition (line 155)
    if_condition_18672 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 155, 9), result_eq_18671)
    # Assigning a type to the variable 'if_condition_18672' (line 155)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 155, 9), 'if_condition_18672', if_condition_18672)
    # SSA begins for if statement (line 155)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Obtaining an instance of the builtin type 'tuple' (line 156)
    tuple_18673 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 156, 16), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 156)
    # Adding element type (line 156)
    
    # Obtaining an instance of the builtin type 'tuple' (line 156)
    tuple_18674 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 156, 17), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 156)
    # Adding element type (line 156)
    # Getting the type of 'qr' (line 156)
    qr_18675 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 17), 'qr')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 156, 17), tuple_18674, qr_18675)
    # Adding element type (line 156)
    # Getting the type of 'tau' (line 156)
    tau_18676 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 21), 'tau')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 156, 17), tuple_18674, tau_18676)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 156, 16), tuple_18673, tuple_18674)
    
    # Getting the type of 'Rj' (line 156)
    Rj_18677 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 30), 'Rj')
    # Applying the binary operator '+' (line 156)
    result_add_18678 = python_operator(stypy.reporting.localization.Localization(__file__, 156, 15), '+', tuple_18673, Rj_18677)
    
    # Assigning a type to the variable 'stypy_return_type' (line 156)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 156, 8), 'stypy_return_type', result_add_18678)
    # SSA join for if statement (line 155)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 153)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Tuple (line 158):
    
    # Assigning a Subscript to a Name (line 158):
    
    # Obtaining the type of the subscript
    int_18679 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 158, 4), 'int')
    
    # Call to get_lapack_funcs(...): (line 158)
    # Processing the call arguments (line 158)
    
    # Obtaining an instance of the builtin type 'tuple' (line 158)
    tuple_18681 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 158, 36), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 158)
    # Adding element type (line 158)
    str_18682 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 158, 36), 'str', 'orgqr')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 158, 36), tuple_18681, str_18682)
    
    
    # Obtaining an instance of the builtin type 'tuple' (line 158)
    tuple_18683 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 158, 48), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 158)
    # Adding element type (line 158)
    # Getting the type of 'qr' (line 158)
    qr_18684 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 48), 'qr', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 158, 48), tuple_18683, qr_18684)
    
    # Processing the call keyword arguments (line 158)
    kwargs_18685 = {}
    # Getting the type of 'get_lapack_funcs' (line 158)
    get_lapack_funcs_18680 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 18), 'get_lapack_funcs', False)
    # Calling get_lapack_funcs(args, kwargs) (line 158)
    get_lapack_funcs_call_result_18686 = invoke(stypy.reporting.localization.Localization(__file__, 158, 18), get_lapack_funcs_18680, *[tuple_18681, tuple_18683], **kwargs_18685)
    
    # Obtaining the member '__getitem__' of a type (line 158)
    getitem___18687 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 158, 4), get_lapack_funcs_call_result_18686, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 158)
    subscript_call_result_18688 = invoke(stypy.reporting.localization.Localization(__file__, 158, 4), getitem___18687, int_18679)
    
    # Assigning a type to the variable 'tuple_var_assignment_18383' (line 158)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 158, 4), 'tuple_var_assignment_18383', subscript_call_result_18688)
    
    # Assigning a Name to a Name (line 158):
    # Getting the type of 'tuple_var_assignment_18383' (line 158)
    tuple_var_assignment_18383_18689 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 4), 'tuple_var_assignment_18383')
    # Assigning a type to the variable 'gor_un_gqr' (line 158)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 158, 4), 'gor_un_gqr', tuple_var_assignment_18383_18689)
    
    
    # Getting the type of 'M' (line 160)
    M_18690 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 7), 'M')
    # Getting the type of 'N' (line 160)
    N_18691 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 11), 'N')
    # Applying the binary operator '<' (line 160)
    result_lt_18692 = python_operator(stypy.reporting.localization.Localization(__file__, 160, 7), '<', M_18690, N_18691)
    
    # Testing the type of an if condition (line 160)
    if_condition_18693 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 160, 4), result_lt_18692)
    # Assigning a type to the variable 'if_condition_18693' (line 160)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 160, 4), 'if_condition_18693', if_condition_18693)
    # SSA begins for if statement (line 160)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Tuple (line 161):
    
    # Assigning a Subscript to a Name (line 161):
    
    # Obtaining the type of the subscript
    int_18694 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 161, 8), 'int')
    
    # Call to safecall(...): (line 161)
    # Processing the call arguments (line 161)
    # Getting the type of 'gor_un_gqr' (line 161)
    gor_un_gqr_18696 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 22), 'gor_un_gqr', False)
    str_18697 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 161, 34), 'str', 'gorgqr/gungqr')
    
    # Obtaining the type of the subscript
    slice_18698 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 161, 51), None, None, None)
    # Getting the type of 'M' (line 161)
    M_18699 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 58), 'M', False)
    slice_18700 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 161, 51), None, M_18699, None)
    # Getting the type of 'qr' (line 161)
    qr_18701 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 51), 'qr', False)
    # Obtaining the member '__getitem__' of a type (line 161)
    getitem___18702 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 161, 51), qr_18701, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 161)
    subscript_call_result_18703 = invoke(stypy.reporting.localization.Localization(__file__, 161, 51), getitem___18702, (slice_18698, slice_18700))
    
    # Getting the type of 'tau' (line 161)
    tau_18704 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 62), 'tau', False)
    # Processing the call keyword arguments (line 161)
    # Getting the type of 'lwork' (line 162)
    lwork_18705 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 18), 'lwork', False)
    keyword_18706 = lwork_18705
    int_18707 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 162, 37), 'int')
    keyword_18708 = int_18707
    kwargs_18709 = {'overwrite_a': keyword_18708, 'lwork': keyword_18706}
    # Getting the type of 'safecall' (line 161)
    safecall_18695 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 13), 'safecall', False)
    # Calling safecall(args, kwargs) (line 161)
    safecall_call_result_18710 = invoke(stypy.reporting.localization.Localization(__file__, 161, 13), safecall_18695, *[gor_un_gqr_18696, str_18697, subscript_call_result_18703, tau_18704], **kwargs_18709)
    
    # Obtaining the member '__getitem__' of a type (line 161)
    getitem___18711 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 161, 8), safecall_call_result_18710, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 161)
    subscript_call_result_18712 = invoke(stypy.reporting.localization.Localization(__file__, 161, 8), getitem___18711, int_18694)
    
    # Assigning a type to the variable 'tuple_var_assignment_18384' (line 161)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 161, 8), 'tuple_var_assignment_18384', subscript_call_result_18712)
    
    # Assigning a Name to a Name (line 161):
    # Getting the type of 'tuple_var_assignment_18384' (line 161)
    tuple_var_assignment_18384_18713 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 8), 'tuple_var_assignment_18384')
    # Assigning a type to the variable 'Q' (line 161)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 161, 8), 'Q', tuple_var_assignment_18384_18713)
    # SSA branch for the else part of an if statement (line 160)
    module_type_store.open_ssa_branch('else')
    
    
    # Getting the type of 'mode' (line 163)
    mode_18714 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 9), 'mode')
    str_18715 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 163, 17), 'str', 'economic')
    # Applying the binary operator '==' (line 163)
    result_eq_18716 = python_operator(stypy.reporting.localization.Localization(__file__, 163, 9), '==', mode_18714, str_18715)
    
    # Testing the type of an if condition (line 163)
    if_condition_18717 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 163, 9), result_eq_18716)
    # Assigning a type to the variable 'if_condition_18717' (line 163)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 163, 9), 'if_condition_18717', if_condition_18717)
    # SSA begins for if statement (line 163)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Tuple (line 164):
    
    # Assigning a Subscript to a Name (line 164):
    
    # Obtaining the type of the subscript
    int_18718 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 164, 8), 'int')
    
    # Call to safecall(...): (line 164)
    # Processing the call arguments (line 164)
    # Getting the type of 'gor_un_gqr' (line 164)
    gor_un_gqr_18720 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 22), 'gor_un_gqr', False)
    str_18721 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 164, 34), 'str', 'gorgqr/gungqr')
    # Getting the type of 'qr' (line 164)
    qr_18722 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 51), 'qr', False)
    # Getting the type of 'tau' (line 164)
    tau_18723 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 55), 'tau', False)
    # Processing the call keyword arguments (line 164)
    # Getting the type of 'lwork' (line 164)
    lwork_18724 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 66), 'lwork', False)
    keyword_18725 = lwork_18724
    int_18726 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 165, 24), 'int')
    keyword_18727 = int_18726
    kwargs_18728 = {'overwrite_a': keyword_18727, 'lwork': keyword_18725}
    # Getting the type of 'safecall' (line 164)
    safecall_18719 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 13), 'safecall', False)
    # Calling safecall(args, kwargs) (line 164)
    safecall_call_result_18729 = invoke(stypy.reporting.localization.Localization(__file__, 164, 13), safecall_18719, *[gor_un_gqr_18720, str_18721, qr_18722, tau_18723], **kwargs_18728)
    
    # Obtaining the member '__getitem__' of a type (line 164)
    getitem___18730 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 164, 8), safecall_call_result_18729, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 164)
    subscript_call_result_18731 = invoke(stypy.reporting.localization.Localization(__file__, 164, 8), getitem___18730, int_18718)
    
    # Assigning a type to the variable 'tuple_var_assignment_18385' (line 164)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 164, 8), 'tuple_var_assignment_18385', subscript_call_result_18731)
    
    # Assigning a Name to a Name (line 164):
    # Getting the type of 'tuple_var_assignment_18385' (line 164)
    tuple_var_assignment_18385_18732 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 8), 'tuple_var_assignment_18385')
    # Assigning a type to the variable 'Q' (line 164)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 164, 8), 'Q', tuple_var_assignment_18385_18732)
    # SSA branch for the else part of an if statement (line 163)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Attribute to a Name (line 167):
    
    # Assigning a Attribute to a Name (line 167):
    # Getting the type of 'qr' (line 167)
    qr_18733 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 12), 'qr')
    # Obtaining the member 'dtype' of a type (line 167)
    dtype_18734 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 167, 12), qr_18733, 'dtype')
    # Obtaining the member 'char' of a type (line 167)
    char_18735 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 167, 12), dtype_18734, 'char')
    # Assigning a type to the variable 't' (line 167)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 167, 8), 't', char_18735)
    
    # Assigning a Call to a Name (line 168):
    
    # Assigning a Call to a Name (line 168):
    
    # Call to empty(...): (line 168)
    # Processing the call arguments (line 168)
    
    # Obtaining an instance of the builtin type 'tuple' (line 168)
    tuple_18738 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 168, 27), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 168)
    # Adding element type (line 168)
    # Getting the type of 'M' (line 168)
    M_18739 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 27), 'M', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 168, 27), tuple_18738, M_18739)
    # Adding element type (line 168)
    # Getting the type of 'M' (line 168)
    M_18740 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 30), 'M', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 168, 27), tuple_18738, M_18740)
    
    # Processing the call keyword arguments (line 168)
    # Getting the type of 't' (line 168)
    t_18741 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 40), 't', False)
    keyword_18742 = t_18741
    kwargs_18743 = {'dtype': keyword_18742}
    # Getting the type of 'numpy' (line 168)
    numpy_18736 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 14), 'numpy', False)
    # Obtaining the member 'empty' of a type (line 168)
    empty_18737 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 168, 14), numpy_18736, 'empty')
    # Calling empty(args, kwargs) (line 168)
    empty_call_result_18744 = invoke(stypy.reporting.localization.Localization(__file__, 168, 14), empty_18737, *[tuple_18738], **kwargs_18743)
    
    # Assigning a type to the variable 'qqr' (line 168)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 168, 8), 'qqr', empty_call_result_18744)
    
    # Assigning a Name to a Subscript (line 169):
    
    # Assigning a Name to a Subscript (line 169):
    # Getting the type of 'qr' (line 169)
    qr_18745 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 21), 'qr')
    # Getting the type of 'qqr' (line 169)
    qqr_18746 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 8), 'qqr')
    slice_18747 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 169, 8), None, None, None)
    # Getting the type of 'N' (line 169)
    N_18748 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 16), 'N')
    slice_18749 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 169, 8), None, N_18748, None)
    # Storing an element on a container (line 169)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 169, 8), qqr_18746, ((slice_18747, slice_18749), qr_18745))
    
    # Assigning a Call to a Tuple (line 170):
    
    # Assigning a Subscript to a Name (line 170):
    
    # Obtaining the type of the subscript
    int_18750 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 170, 8), 'int')
    
    # Call to safecall(...): (line 170)
    # Processing the call arguments (line 170)
    # Getting the type of 'gor_un_gqr' (line 170)
    gor_un_gqr_18752 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 22), 'gor_un_gqr', False)
    str_18753 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 170, 34), 'str', 'gorgqr/gungqr')
    # Getting the type of 'qqr' (line 170)
    qqr_18754 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 51), 'qqr', False)
    # Getting the type of 'tau' (line 170)
    tau_18755 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 56), 'tau', False)
    # Processing the call keyword arguments (line 170)
    # Getting the type of 'lwork' (line 170)
    lwork_18756 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 67), 'lwork', False)
    keyword_18757 = lwork_18756
    int_18758 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 171, 24), 'int')
    keyword_18759 = int_18758
    kwargs_18760 = {'overwrite_a': keyword_18759, 'lwork': keyword_18757}
    # Getting the type of 'safecall' (line 170)
    safecall_18751 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 13), 'safecall', False)
    # Calling safecall(args, kwargs) (line 170)
    safecall_call_result_18761 = invoke(stypy.reporting.localization.Localization(__file__, 170, 13), safecall_18751, *[gor_un_gqr_18752, str_18753, qqr_18754, tau_18755], **kwargs_18760)
    
    # Obtaining the member '__getitem__' of a type (line 170)
    getitem___18762 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 170, 8), safecall_call_result_18761, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 170)
    subscript_call_result_18763 = invoke(stypy.reporting.localization.Localization(__file__, 170, 8), getitem___18762, int_18750)
    
    # Assigning a type to the variable 'tuple_var_assignment_18386' (line 170)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 170, 8), 'tuple_var_assignment_18386', subscript_call_result_18763)
    
    # Assigning a Name to a Name (line 170):
    # Getting the type of 'tuple_var_assignment_18386' (line 170)
    tuple_var_assignment_18386_18764 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 8), 'tuple_var_assignment_18386')
    # Assigning a type to the variable 'Q' (line 170)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 170, 8), 'Q', tuple_var_assignment_18386_18764)
    # SSA join for if statement (line 163)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 160)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Obtaining an instance of the builtin type 'tuple' (line 173)
    tuple_18765 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 173, 12), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 173)
    # Adding element type (line 173)
    # Getting the type of 'Q' (line 173)
    Q_18766 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 12), 'Q')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 173, 12), tuple_18765, Q_18766)
    
    # Getting the type of 'Rj' (line 173)
    Rj_18767 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 18), 'Rj')
    # Applying the binary operator '+' (line 173)
    result_add_18768 = python_operator(stypy.reporting.localization.Localization(__file__, 173, 11), '+', tuple_18765, Rj_18767)
    
    # Assigning a type to the variable 'stypy_return_type' (line 173)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 173, 4), 'stypy_return_type', result_add_18768)
    
    # ################# End of 'qr(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'qr' in the type store
    # Getting the type of 'stypy_return_type' (line 28)
    stypy_return_type_18769 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_18769)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'qr'
    return stypy_return_type_18769

# Assigning a type to the variable 'qr' (line 28)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 28, 0), 'qr', qr)

@norecursion
def qr_multiply(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    str_18770 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 176, 27), 'str', 'right')
    # Getting the type of 'False' (line 176)
    False_18771 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 45), 'False')
    # Getting the type of 'False' (line 176)
    False_18772 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 62), 'False')
    # Getting the type of 'False' (line 177)
    False_18773 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 16), 'False')
    # Getting the type of 'False' (line 177)
    False_18774 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 35), 'False')
    defaults = [str_18770, False_18771, False_18772, False_18773, False_18774]
    # Create a new context for function 'qr_multiply'
    module_type_store = module_type_store.open_function_context('qr_multiply', 176, 0, False)
    
    # Passed parameters checking function
    qr_multiply.stypy_localization = localization
    qr_multiply.stypy_type_of_self = None
    qr_multiply.stypy_type_store = module_type_store
    qr_multiply.stypy_function_name = 'qr_multiply'
    qr_multiply.stypy_param_names_list = ['a', 'c', 'mode', 'pivoting', 'conjugate', 'overwrite_a', 'overwrite_c']
    qr_multiply.stypy_varargs_param_name = None
    qr_multiply.stypy_kwargs_param_name = None
    qr_multiply.stypy_call_defaults = defaults
    qr_multiply.stypy_call_varargs = varargs
    qr_multiply.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'qr_multiply', ['a', 'c', 'mode', 'pivoting', 'conjugate', 'overwrite_a', 'overwrite_c'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'qr_multiply', localization, ['a', 'c', 'mode', 'pivoting', 'conjugate', 'overwrite_a', 'overwrite_c'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'qr_multiply(...)' code ##################

    str_18775 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 232, (-1)), 'str', "\n    Calculate the QR decomposition and multiply Q with a matrix.\n\n    Calculate the decomposition ``A = Q R`` where Q is unitary/orthogonal\n    and R upper triangular. Multiply Q with a vector or a matrix c.\n\n    Parameters\n    ----------\n    a : array_like, shape (M, N)\n        Matrix to be decomposed\n    c : array_like, one- or two-dimensional\n        calculate the product of c and q, depending on the mode:\n    mode : {'left', 'right'}, optional\n        ``dot(Q, c)`` is returned if mode is 'left',\n        ``dot(c, Q)`` is returned if mode is 'right'.\n        The shape of c must be appropriate for the matrix multiplications,\n        if mode is 'left', ``min(a.shape) == c.shape[0]``,\n        if mode is 'right', ``a.shape[0] == c.shape[1]``.\n    pivoting : bool, optional\n        Whether or not factorization should include pivoting for rank-revealing\n        qr decomposition, see the documentation of qr.\n    conjugate : bool, optional\n        Whether Q should be complex-conjugated. This might be faster\n        than explicit conjugation.\n    overwrite_a : bool, optional\n        Whether data in a is overwritten (may improve performance)\n    overwrite_c : bool, optional\n        Whether data in c is overwritten (may improve performance).\n        If this is used, c must be big enough to keep the result,\n        i.e. c.shape[0] = a.shape[0] if mode is 'left'.\n\n\n    Returns\n    -------\n    CQ : float or complex ndarray\n        the product of Q and c, as defined in mode\n    R : float or complex ndarray\n        Of shape (K, N), ``K = min(M, N)``.\n    P : ndarray of ints\n        Of shape (N,) for ``pivoting=True``.\n        Not returned if ``pivoting=False``.\n\n    Raises\n    ------\n    LinAlgError\n        Raised if decomposition fails\n\n    Notes\n    -----\n    This is an interface to the LAPACK routines dgeqrf, zgeqrf,\n    dormqr, zunmqr, dgeqp3, and zgeqp3.\n\n    .. versionadded:: 0.11.0\n\n    ")
    
    
    # Getting the type of 'mode' (line 233)
    mode_18776 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 233, 7), 'mode')
    
    # Obtaining an instance of the builtin type 'list' (line 233)
    list_18777 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 233, 19), 'list')
    # Adding type elements to the builtin type 'list' instance (line 233)
    # Adding element type (line 233)
    str_18778 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 233, 20), 'str', 'left')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 233, 19), list_18777, str_18778)
    # Adding element type (line 233)
    str_18779 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 233, 28), 'str', 'right')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 233, 19), list_18777, str_18779)
    
    # Applying the binary operator 'notin' (line 233)
    result_contains_18780 = python_operator(stypy.reporting.localization.Localization(__file__, 233, 7), 'notin', mode_18776, list_18777)
    
    # Testing the type of an if condition (line 233)
    if_condition_18781 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 233, 4), result_contains_18780)
    # Assigning a type to the variable 'if_condition_18781' (line 233)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 233, 4), 'if_condition_18781', if_condition_18781)
    # SSA begins for if statement (line 233)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 234)
    # Processing the call arguments (line 234)
    str_18783 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 234, 25), 'str', "Mode argument should be one of ['left', 'right']")
    # Processing the call keyword arguments (line 234)
    kwargs_18784 = {}
    # Getting the type of 'ValueError' (line 234)
    ValueError_18782 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 234, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 234)
    ValueError_call_result_18785 = invoke(stypy.reporting.localization.Localization(__file__, 234, 14), ValueError_18782, *[str_18783], **kwargs_18784)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 234, 8), ValueError_call_result_18785, 'raise parameter', BaseException)
    # SSA join for if statement (line 233)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 235):
    
    # Assigning a Call to a Name (line 235):
    
    # Call to asarray_chkfinite(...): (line 235)
    # Processing the call arguments (line 235)
    # Getting the type of 'c' (line 235)
    c_18788 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 32), 'c', False)
    # Processing the call keyword arguments (line 235)
    kwargs_18789 = {}
    # Getting the type of 'numpy' (line 235)
    numpy_18786 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 8), 'numpy', False)
    # Obtaining the member 'asarray_chkfinite' of a type (line 235)
    asarray_chkfinite_18787 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 235, 8), numpy_18786, 'asarray_chkfinite')
    # Calling asarray_chkfinite(args, kwargs) (line 235)
    asarray_chkfinite_call_result_18790 = invoke(stypy.reporting.localization.Localization(__file__, 235, 8), asarray_chkfinite_18787, *[c_18788], **kwargs_18789)
    
    # Assigning a type to the variable 'c' (line 235)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 235, 4), 'c', asarray_chkfinite_call_result_18790)
    
    # Assigning a Compare to a Name (line 236):
    
    # Assigning a Compare to a Name (line 236):
    
    # Getting the type of 'c' (line 236)
    c_18791 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 236, 13), 'c')
    # Obtaining the member 'ndim' of a type (line 236)
    ndim_18792 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 236, 13), c_18791, 'ndim')
    int_18793 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 236, 23), 'int')
    # Applying the binary operator '==' (line 236)
    result_eq_18794 = python_operator(stypy.reporting.localization.Localization(__file__, 236, 13), '==', ndim_18792, int_18793)
    
    # Assigning a type to the variable 'onedim' (line 236)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 236, 4), 'onedim', result_eq_18794)
    
    # Getting the type of 'onedim' (line 237)
    onedim_18795 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 237, 7), 'onedim')
    # Testing the type of an if condition (line 237)
    if_condition_18796 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 237, 4), onedim_18795)
    # Assigning a type to the variable 'if_condition_18796' (line 237)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 237, 4), 'if_condition_18796', if_condition_18796)
    # SSA begins for if statement (line 237)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 238):
    
    # Assigning a Call to a Name (line 238):
    
    # Call to reshape(...): (line 238)
    # Processing the call arguments (line 238)
    int_18799 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 238, 22), 'int')
    
    # Call to len(...): (line 238)
    # Processing the call arguments (line 238)
    # Getting the type of 'c' (line 238)
    c_18801 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 238, 29), 'c', False)
    # Processing the call keyword arguments (line 238)
    kwargs_18802 = {}
    # Getting the type of 'len' (line 238)
    len_18800 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 238, 25), 'len', False)
    # Calling len(args, kwargs) (line 238)
    len_call_result_18803 = invoke(stypy.reporting.localization.Localization(__file__, 238, 25), len_18800, *[c_18801], **kwargs_18802)
    
    # Processing the call keyword arguments (line 238)
    kwargs_18804 = {}
    # Getting the type of 'c' (line 238)
    c_18797 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 238, 12), 'c', False)
    # Obtaining the member 'reshape' of a type (line 238)
    reshape_18798 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 238, 12), c_18797, 'reshape')
    # Calling reshape(args, kwargs) (line 238)
    reshape_call_result_18805 = invoke(stypy.reporting.localization.Localization(__file__, 238, 12), reshape_18798, *[int_18799, len_call_result_18803], **kwargs_18804)
    
    # Assigning a type to the variable 'c' (line 238)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 238, 8), 'c', reshape_call_result_18805)
    
    
    # Getting the type of 'mode' (line 239)
    mode_18806 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 239, 11), 'mode')
    str_18807 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 239, 19), 'str', 'left')
    # Applying the binary operator '==' (line 239)
    result_eq_18808 = python_operator(stypy.reporting.localization.Localization(__file__, 239, 11), '==', mode_18806, str_18807)
    
    # Testing the type of an if condition (line 239)
    if_condition_18809 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 239, 8), result_eq_18808)
    # Assigning a type to the variable 'if_condition_18809' (line 239)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 239, 8), 'if_condition_18809', if_condition_18809)
    # SSA begins for if statement (line 239)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Attribute to a Name (line 240):
    
    # Assigning a Attribute to a Name (line 240):
    # Getting the type of 'c' (line 240)
    c_18810 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 16), 'c')
    # Obtaining the member 'T' of a type (line 240)
    T_18811 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 240, 16), c_18810, 'T')
    # Assigning a type to the variable 'c' (line 240)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 240, 12), 'c', T_18811)
    # SSA join for if statement (line 239)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 237)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 242):
    
    # Assigning a Call to a Name (line 242):
    
    # Call to asarray(...): (line 242)
    # Processing the call arguments (line 242)
    # Getting the type of 'a' (line 242)
    a_18814 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 242, 22), 'a', False)
    # Processing the call keyword arguments (line 242)
    kwargs_18815 = {}
    # Getting the type of 'numpy' (line 242)
    numpy_18812 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 242, 8), 'numpy', False)
    # Obtaining the member 'asarray' of a type (line 242)
    asarray_18813 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 242, 8), numpy_18812, 'asarray')
    # Calling asarray(args, kwargs) (line 242)
    asarray_call_result_18816 = invoke(stypy.reporting.localization.Localization(__file__, 242, 8), asarray_18813, *[a_18814], **kwargs_18815)
    
    # Assigning a type to the variable 'a' (line 242)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 242, 4), 'a', asarray_call_result_18816)
    
    # Assigning a Attribute to a Tuple (line 243):
    
    # Assigning a Subscript to a Name (line 243):
    
    # Obtaining the type of the subscript
    int_18817 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 243, 4), 'int')
    # Getting the type of 'a' (line 243)
    a_18818 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 243, 11), 'a')
    # Obtaining the member 'shape' of a type (line 243)
    shape_18819 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 243, 11), a_18818, 'shape')
    # Obtaining the member '__getitem__' of a type (line 243)
    getitem___18820 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 243, 4), shape_18819, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 243)
    subscript_call_result_18821 = invoke(stypy.reporting.localization.Localization(__file__, 243, 4), getitem___18820, int_18817)
    
    # Assigning a type to the variable 'tuple_var_assignment_18387' (line 243)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 243, 4), 'tuple_var_assignment_18387', subscript_call_result_18821)
    
    # Assigning a Subscript to a Name (line 243):
    
    # Obtaining the type of the subscript
    int_18822 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 243, 4), 'int')
    # Getting the type of 'a' (line 243)
    a_18823 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 243, 11), 'a')
    # Obtaining the member 'shape' of a type (line 243)
    shape_18824 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 243, 11), a_18823, 'shape')
    # Obtaining the member '__getitem__' of a type (line 243)
    getitem___18825 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 243, 4), shape_18824, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 243)
    subscript_call_result_18826 = invoke(stypy.reporting.localization.Localization(__file__, 243, 4), getitem___18825, int_18822)
    
    # Assigning a type to the variable 'tuple_var_assignment_18388' (line 243)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 243, 4), 'tuple_var_assignment_18388', subscript_call_result_18826)
    
    # Assigning a Name to a Name (line 243):
    # Getting the type of 'tuple_var_assignment_18387' (line 243)
    tuple_var_assignment_18387_18827 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 243, 4), 'tuple_var_assignment_18387')
    # Assigning a type to the variable 'M' (line 243)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 243, 4), 'M', tuple_var_assignment_18387_18827)
    
    # Assigning a Name to a Name (line 243):
    # Getting the type of 'tuple_var_assignment_18388' (line 243)
    tuple_var_assignment_18388_18828 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 243, 4), 'tuple_var_assignment_18388')
    # Assigning a type to the variable 'N' (line 243)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 243, 7), 'N', tuple_var_assignment_18388_18828)
    
    
    
    # Evaluating a boolean operation
    
    # Evaluating a boolean operation
    
    # Getting the type of 'mode' (line 244)
    mode_18829 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 244, 12), 'mode')
    str_18830 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 244, 20), 'str', 'left')
    # Applying the binary operator '==' (line 244)
    result_eq_18831 = python_operator(stypy.reporting.localization.Localization(__file__, 244, 12), '==', mode_18829, str_18830)
    
    
    # Evaluating a boolean operation
    
    # Evaluating a boolean operation
    
    # Getting the type of 'overwrite_c' (line 245)
    overwrite_c_18832 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 245, 21), 'overwrite_c')
    # Applying the 'not' unary operator (line 245)
    result_not__18833 = python_operator(stypy.reporting.localization.Localization(__file__, 245, 17), 'not', overwrite_c_18832)
    
    
    
    # Call to min(...): (line 245)
    # Processing the call arguments (line 245)
    # Getting the type of 'M' (line 245)
    M_18835 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 245, 41), 'M', False)
    # Getting the type of 'N' (line 245)
    N_18836 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 245, 44), 'N', False)
    # Processing the call keyword arguments (line 245)
    kwargs_18837 = {}
    # Getting the type of 'min' (line 245)
    min_18834 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 245, 37), 'min', False)
    # Calling min(args, kwargs) (line 245)
    min_call_result_18838 = invoke(stypy.reporting.localization.Localization(__file__, 245, 37), min_18834, *[M_18835, N_18836], **kwargs_18837)
    
    
    # Obtaining the type of the subscript
    int_18839 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 245, 58), 'int')
    # Getting the type of 'c' (line 245)
    c_18840 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 245, 50), 'c')
    # Obtaining the member 'shape' of a type (line 245)
    shape_18841 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 245, 50), c_18840, 'shape')
    # Obtaining the member '__getitem__' of a type (line 245)
    getitem___18842 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 245, 50), shape_18841, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 245)
    subscript_call_result_18843 = invoke(stypy.reporting.localization.Localization(__file__, 245, 50), getitem___18842, int_18839)
    
    # Applying the binary operator '==' (line 245)
    result_eq_18844 = python_operator(stypy.reporting.localization.Localization(__file__, 245, 37), '==', min_call_result_18838, subscript_call_result_18843)
    
    # Applying the binary operator 'and' (line 245)
    result_and_keyword_18845 = python_operator(stypy.reporting.localization.Localization(__file__, 245, 17), 'and', result_not__18833, result_eq_18844)
    
    
    # Evaluating a boolean operation
    # Getting the type of 'overwrite_c' (line 246)
    overwrite_c_18846 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 21), 'overwrite_c')
    
    # Getting the type of 'M' (line 246)
    M_18847 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 37), 'M')
    
    # Obtaining the type of the subscript
    int_18848 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 246, 50), 'int')
    # Getting the type of 'c' (line 246)
    c_18849 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 42), 'c')
    # Obtaining the member 'shape' of a type (line 246)
    shape_18850 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 246, 42), c_18849, 'shape')
    # Obtaining the member '__getitem__' of a type (line 246)
    getitem___18851 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 246, 42), shape_18850, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 246)
    subscript_call_result_18852 = invoke(stypy.reporting.localization.Localization(__file__, 246, 42), getitem___18851, int_18848)
    
    # Applying the binary operator '==' (line 246)
    result_eq_18853 = python_operator(stypy.reporting.localization.Localization(__file__, 246, 37), '==', M_18847, subscript_call_result_18852)
    
    # Applying the binary operator 'and' (line 246)
    result_and_keyword_18854 = python_operator(stypy.reporting.localization.Localization(__file__, 246, 21), 'and', overwrite_c_18846, result_eq_18853)
    
    # Applying the binary operator 'or' (line 245)
    result_or_keyword_18855 = python_operator(stypy.reporting.localization.Localization(__file__, 245, 17), 'or', result_and_keyword_18845, result_and_keyword_18854)
    
    # Applying the binary operator 'and' (line 244)
    result_and_keyword_18856 = python_operator(stypy.reporting.localization.Localization(__file__, 244, 12), 'and', result_eq_18831, result_or_keyword_18855)
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'mode' (line 247)
    mode_18857 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 247, 12), 'mode')
    str_18858 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 247, 20), 'str', 'right')
    # Applying the binary operator '==' (line 247)
    result_eq_18859 = python_operator(stypy.reporting.localization.Localization(__file__, 247, 12), '==', mode_18857, str_18858)
    
    
    # Getting the type of 'M' (line 247)
    M_18860 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 247, 32), 'M')
    
    # Obtaining the type of the subscript
    int_18861 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 247, 45), 'int')
    # Getting the type of 'c' (line 247)
    c_18862 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 247, 37), 'c')
    # Obtaining the member 'shape' of a type (line 247)
    shape_18863 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 247, 37), c_18862, 'shape')
    # Obtaining the member '__getitem__' of a type (line 247)
    getitem___18864 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 247, 37), shape_18863, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 247)
    subscript_call_result_18865 = invoke(stypy.reporting.localization.Localization(__file__, 247, 37), getitem___18864, int_18861)
    
    # Applying the binary operator '==' (line 247)
    result_eq_18866 = python_operator(stypy.reporting.localization.Localization(__file__, 247, 32), '==', M_18860, subscript_call_result_18865)
    
    # Applying the binary operator 'and' (line 247)
    result_and_keyword_18867 = python_operator(stypy.reporting.localization.Localization(__file__, 247, 12), 'and', result_eq_18859, result_eq_18866)
    
    # Applying the binary operator 'or' (line 244)
    result_or_keyword_18868 = python_operator(stypy.reporting.localization.Localization(__file__, 244, 12), 'or', result_and_keyword_18856, result_and_keyword_18867)
    
    # Applying the 'not' unary operator (line 244)
    result_not__18869 = python_operator(stypy.reporting.localization.Localization(__file__, 244, 7), 'not', result_or_keyword_18868)
    
    # Testing the type of an if condition (line 244)
    if_condition_18870 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 244, 4), result_not__18869)
    # Assigning a type to the variable 'if_condition_18870' (line 244)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 244, 4), 'if_condition_18870', if_condition_18870)
    # SSA begins for if statement (line 244)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 248)
    # Processing the call arguments (line 248)
    str_18872 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 248, 25), 'str', 'objects are not aligned')
    # Processing the call keyword arguments (line 248)
    kwargs_18873 = {}
    # Getting the type of 'ValueError' (line 248)
    ValueError_18871 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 248, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 248)
    ValueError_call_result_18874 = invoke(stypy.reporting.localization.Localization(__file__, 248, 14), ValueError_18871, *[str_18872], **kwargs_18873)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 248, 8), ValueError_call_result_18874, 'raise parameter', BaseException)
    # SSA join for if statement (line 244)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 250):
    
    # Assigning a Call to a Name (line 250):
    
    # Call to qr(...): (line 250)
    # Processing the call arguments (line 250)
    # Getting the type of 'a' (line 250)
    a_18876 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 250, 13), 'a', False)
    # Getting the type of 'overwrite_a' (line 250)
    overwrite_a_18877 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 250, 16), 'overwrite_a', False)
    # Getting the type of 'None' (line 250)
    None_18878 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 250, 29), 'None', False)
    str_18879 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 250, 35), 'str', 'raw')
    # Getting the type of 'pivoting' (line 250)
    pivoting_18880 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 250, 42), 'pivoting', False)
    # Processing the call keyword arguments (line 250)
    kwargs_18881 = {}
    # Getting the type of 'qr' (line 250)
    qr_18875 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 250, 10), 'qr', False)
    # Calling qr(args, kwargs) (line 250)
    qr_call_result_18882 = invoke(stypy.reporting.localization.Localization(__file__, 250, 10), qr_18875, *[a_18876, overwrite_a_18877, None_18878, str_18879, pivoting_18880], **kwargs_18881)
    
    # Assigning a type to the variable 'raw' (line 250)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 250, 4), 'raw', qr_call_result_18882)
    
    # Assigning a Subscript to a Tuple (line 251):
    
    # Assigning a Subscript to a Name (line 251):
    
    # Obtaining the type of the subscript
    int_18883 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 251, 4), 'int')
    
    # Obtaining the type of the subscript
    int_18884 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 251, 17), 'int')
    # Getting the type of 'raw' (line 251)
    raw_18885 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 251, 13), 'raw')
    # Obtaining the member '__getitem__' of a type (line 251)
    getitem___18886 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 251, 13), raw_18885, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 251)
    subscript_call_result_18887 = invoke(stypy.reporting.localization.Localization(__file__, 251, 13), getitem___18886, int_18884)
    
    # Obtaining the member '__getitem__' of a type (line 251)
    getitem___18888 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 251, 4), subscript_call_result_18887, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 251)
    subscript_call_result_18889 = invoke(stypy.reporting.localization.Localization(__file__, 251, 4), getitem___18888, int_18883)
    
    # Assigning a type to the variable 'tuple_var_assignment_18389' (line 251)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 251, 4), 'tuple_var_assignment_18389', subscript_call_result_18889)
    
    # Assigning a Subscript to a Name (line 251):
    
    # Obtaining the type of the subscript
    int_18890 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 251, 4), 'int')
    
    # Obtaining the type of the subscript
    int_18891 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 251, 17), 'int')
    # Getting the type of 'raw' (line 251)
    raw_18892 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 251, 13), 'raw')
    # Obtaining the member '__getitem__' of a type (line 251)
    getitem___18893 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 251, 13), raw_18892, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 251)
    subscript_call_result_18894 = invoke(stypy.reporting.localization.Localization(__file__, 251, 13), getitem___18893, int_18891)
    
    # Obtaining the member '__getitem__' of a type (line 251)
    getitem___18895 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 251, 4), subscript_call_result_18894, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 251)
    subscript_call_result_18896 = invoke(stypy.reporting.localization.Localization(__file__, 251, 4), getitem___18895, int_18890)
    
    # Assigning a type to the variable 'tuple_var_assignment_18390' (line 251)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 251, 4), 'tuple_var_assignment_18390', subscript_call_result_18896)
    
    # Assigning a Name to a Name (line 251):
    # Getting the type of 'tuple_var_assignment_18389' (line 251)
    tuple_var_assignment_18389_18897 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 251, 4), 'tuple_var_assignment_18389')
    # Assigning a type to the variable 'Q' (line 251)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 251, 4), 'Q', tuple_var_assignment_18389_18897)
    
    # Assigning a Name to a Name (line 251):
    # Getting the type of 'tuple_var_assignment_18390' (line 251)
    tuple_var_assignment_18390_18898 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 251, 4), 'tuple_var_assignment_18390')
    # Assigning a type to the variable 'tau' (line 251)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 251, 7), 'tau', tuple_var_assignment_18390_18898)
    
    # Assigning a Call to a Tuple (line 253):
    
    # Assigning a Subscript to a Name (line 253):
    
    # Obtaining the type of the subscript
    int_18899 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 253, 4), 'int')
    
    # Call to get_lapack_funcs(...): (line 253)
    # Processing the call arguments (line 253)
    
    # Obtaining an instance of the builtin type 'tuple' (line 253)
    tuple_18901 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 253, 36), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 253)
    # Adding element type (line 253)
    str_18902 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 253, 36), 'str', 'ormqr')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 253, 36), tuple_18901, str_18902)
    
    
    # Obtaining an instance of the builtin type 'tuple' (line 253)
    tuple_18903 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 253, 48), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 253)
    # Adding element type (line 253)
    # Getting the type of 'Q' (line 253)
    Q_18904 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 253, 48), 'Q', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 253, 48), tuple_18903, Q_18904)
    
    # Processing the call keyword arguments (line 253)
    kwargs_18905 = {}
    # Getting the type of 'get_lapack_funcs' (line 253)
    get_lapack_funcs_18900 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 253, 18), 'get_lapack_funcs', False)
    # Calling get_lapack_funcs(args, kwargs) (line 253)
    get_lapack_funcs_call_result_18906 = invoke(stypy.reporting.localization.Localization(__file__, 253, 18), get_lapack_funcs_18900, *[tuple_18901, tuple_18903], **kwargs_18905)
    
    # Obtaining the member '__getitem__' of a type (line 253)
    getitem___18907 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 253, 4), get_lapack_funcs_call_result_18906, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 253)
    subscript_call_result_18908 = invoke(stypy.reporting.localization.Localization(__file__, 253, 4), getitem___18907, int_18899)
    
    # Assigning a type to the variable 'tuple_var_assignment_18391' (line 253)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 253, 4), 'tuple_var_assignment_18391', subscript_call_result_18908)
    
    # Assigning a Name to a Name (line 253):
    # Getting the type of 'tuple_var_assignment_18391' (line 253)
    tuple_var_assignment_18391_18909 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 253, 4), 'tuple_var_assignment_18391')
    # Assigning a type to the variable 'gor_un_mqr' (line 253)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 253, 4), 'gor_un_mqr', tuple_var_assignment_18391_18909)
    
    
    # Getting the type of 'gor_un_mqr' (line 254)
    gor_un_mqr_18910 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 254, 7), 'gor_un_mqr')
    # Obtaining the member 'typecode' of a type (line 254)
    typecode_18911 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 254, 7), gor_un_mqr_18910, 'typecode')
    
    # Obtaining an instance of the builtin type 'tuple' (line 254)
    tuple_18912 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 254, 31), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 254)
    # Adding element type (line 254)
    str_18913 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 254, 31), 'str', 's')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 254, 31), tuple_18912, str_18913)
    # Adding element type (line 254)
    str_18914 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 254, 36), 'str', 'd')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 254, 31), tuple_18912, str_18914)
    
    # Applying the binary operator 'in' (line 254)
    result_contains_18915 = python_operator(stypy.reporting.localization.Localization(__file__, 254, 7), 'in', typecode_18911, tuple_18912)
    
    # Testing the type of an if condition (line 254)
    if_condition_18916 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 254, 4), result_contains_18915)
    # Assigning a type to the variable 'if_condition_18916' (line 254)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 254, 4), 'if_condition_18916', if_condition_18916)
    # SSA begins for if statement (line 254)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Str to a Name (line 255):
    
    # Assigning a Str to a Name (line 255):
    str_18917 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 255, 16), 'str', 'T')
    # Assigning a type to the variable 'trans' (line 255)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 255, 8), 'trans', str_18917)
    # SSA branch for the else part of an if statement (line 254)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Str to a Name (line 257):
    
    # Assigning a Str to a Name (line 257):
    str_18918 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 257, 16), 'str', 'C')
    # Assigning a type to the variable 'trans' (line 257)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 257, 8), 'trans', str_18918)
    # SSA join for if statement (line 254)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Subscript to a Name (line 259):
    
    # Assigning a Subscript to a Name (line 259):
    
    # Obtaining the type of the subscript
    slice_18919 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 259, 8), None, None, None)
    
    # Call to min(...): (line 259)
    # Processing the call arguments (line 259)
    # Getting the type of 'M' (line 259)
    M_18921 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 259, 18), 'M', False)
    # Getting the type of 'N' (line 259)
    N_18922 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 259, 21), 'N', False)
    # Processing the call keyword arguments (line 259)
    kwargs_18923 = {}
    # Getting the type of 'min' (line 259)
    min_18920 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 259, 14), 'min', False)
    # Calling min(args, kwargs) (line 259)
    min_call_result_18924 = invoke(stypy.reporting.localization.Localization(__file__, 259, 14), min_18920, *[M_18921, N_18922], **kwargs_18923)
    
    slice_18925 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 259, 8), None, min_call_result_18924, None)
    # Getting the type of 'Q' (line 259)
    Q_18926 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 259, 8), 'Q')
    # Obtaining the member '__getitem__' of a type (line 259)
    getitem___18927 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 259, 8), Q_18926, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 259)
    subscript_call_result_18928 = invoke(stypy.reporting.localization.Localization(__file__, 259, 8), getitem___18927, (slice_18919, slice_18925))
    
    # Assigning a type to the variable 'Q' (line 259)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 259, 4), 'Q', subscript_call_result_18928)
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'M' (line 260)
    M_18929 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 260, 7), 'M')
    # Getting the type of 'N' (line 260)
    N_18930 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 260, 11), 'N')
    # Applying the binary operator '>' (line 260)
    result_gt_18931 = python_operator(stypy.reporting.localization.Localization(__file__, 260, 7), '>', M_18929, N_18930)
    
    
    # Getting the type of 'mode' (line 260)
    mode_18932 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 260, 17), 'mode')
    str_18933 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 260, 25), 'str', 'left')
    # Applying the binary operator '==' (line 260)
    result_eq_18934 = python_operator(stypy.reporting.localization.Localization(__file__, 260, 17), '==', mode_18932, str_18933)
    
    # Applying the binary operator 'and' (line 260)
    result_and_keyword_18935 = python_operator(stypy.reporting.localization.Localization(__file__, 260, 7), 'and', result_gt_18931, result_eq_18934)
    
    # Getting the type of 'overwrite_c' (line 260)
    overwrite_c_18936 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 260, 40), 'overwrite_c')
    # Applying the 'not' unary operator (line 260)
    result_not__18937 = python_operator(stypy.reporting.localization.Localization(__file__, 260, 36), 'not', overwrite_c_18936)
    
    # Applying the binary operator 'and' (line 260)
    result_and_keyword_18938 = python_operator(stypy.reporting.localization.Localization(__file__, 260, 7), 'and', result_and_keyword_18935, result_not__18937)
    
    # Testing the type of an if condition (line 260)
    if_condition_18939 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 260, 4), result_and_keyword_18938)
    # Assigning a type to the variable 'if_condition_18939' (line 260)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 260, 4), 'if_condition_18939', if_condition_18939)
    # SSA begins for if statement (line 260)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Getting the type of 'conjugate' (line 261)
    conjugate_18940 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 261, 11), 'conjugate')
    # Testing the type of an if condition (line 261)
    if_condition_18941 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 261, 8), conjugate_18940)
    # Assigning a type to the variable 'if_condition_18941' (line 261)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 261, 8), 'if_condition_18941', if_condition_18941)
    # SSA begins for if statement (line 261)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 262):
    
    # Assigning a Call to a Name (line 262):
    
    # Call to zeros(...): (line 262)
    # Processing the call arguments (line 262)
    
    # Obtaining an instance of the builtin type 'tuple' (line 262)
    tuple_18944 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 262, 30), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 262)
    # Adding element type (line 262)
    
    # Obtaining the type of the subscript
    int_18945 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 262, 38), 'int')
    # Getting the type of 'c' (line 262)
    c_18946 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 262, 30), 'c', False)
    # Obtaining the member 'shape' of a type (line 262)
    shape_18947 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 262, 30), c_18946, 'shape')
    # Obtaining the member '__getitem__' of a type (line 262)
    getitem___18948 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 262, 30), shape_18947, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 262)
    subscript_call_result_18949 = invoke(stypy.reporting.localization.Localization(__file__, 262, 30), getitem___18948, int_18945)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 262, 30), tuple_18944, subscript_call_result_18949)
    # Adding element type (line 262)
    # Getting the type of 'M' (line 262)
    M_18950 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 262, 42), 'M', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 262, 30), tuple_18944, M_18950)
    
    # Processing the call keyword arguments (line 262)
    # Getting the type of 'c' (line 262)
    c_18951 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 262, 52), 'c', False)
    # Obtaining the member 'dtype' of a type (line 262)
    dtype_18952 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 262, 52), c_18951, 'dtype')
    keyword_18953 = dtype_18952
    str_18954 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 262, 67), 'str', 'F')
    keyword_18955 = str_18954
    kwargs_18956 = {'dtype': keyword_18953, 'order': keyword_18955}
    # Getting the type of 'numpy' (line 262)
    numpy_18942 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 262, 17), 'numpy', False)
    # Obtaining the member 'zeros' of a type (line 262)
    zeros_18943 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 262, 17), numpy_18942, 'zeros')
    # Calling zeros(args, kwargs) (line 262)
    zeros_call_result_18957 = invoke(stypy.reporting.localization.Localization(__file__, 262, 17), zeros_18943, *[tuple_18944], **kwargs_18956)
    
    # Assigning a type to the variable 'cc' (line 262)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 262, 12), 'cc', zeros_call_result_18957)
    
    # Assigning a Attribute to a Subscript (line 263):
    
    # Assigning a Attribute to a Subscript (line 263):
    # Getting the type of 'c' (line 263)
    c_18958 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 263, 24), 'c')
    # Obtaining the member 'T' of a type (line 263)
    T_18959 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 263, 24), c_18958, 'T')
    # Getting the type of 'cc' (line 263)
    cc_18960 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 263, 12), 'cc')
    slice_18961 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 263, 12), None, None, None)
    # Getting the type of 'N' (line 263)
    N_18962 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 263, 19), 'N')
    slice_18963 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 263, 12), None, N_18962, None)
    # Storing an element on a container (line 263)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 263, 12), cc_18960, ((slice_18961, slice_18963), T_18959))
    # SSA branch for the else part of an if statement (line 261)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Call to a Name (line 265):
    
    # Assigning a Call to a Name (line 265):
    
    # Call to zeros(...): (line 265)
    # Processing the call arguments (line 265)
    
    # Obtaining an instance of the builtin type 'tuple' (line 265)
    tuple_18966 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 265, 30), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 265)
    # Adding element type (line 265)
    # Getting the type of 'M' (line 265)
    M_18967 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 265, 30), 'M', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 265, 30), tuple_18966, M_18967)
    # Adding element type (line 265)
    
    # Obtaining the type of the subscript
    int_18968 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 265, 41), 'int')
    # Getting the type of 'c' (line 265)
    c_18969 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 265, 33), 'c', False)
    # Obtaining the member 'shape' of a type (line 265)
    shape_18970 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 265, 33), c_18969, 'shape')
    # Obtaining the member '__getitem__' of a type (line 265)
    getitem___18971 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 265, 33), shape_18970, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 265)
    subscript_call_result_18972 = invoke(stypy.reporting.localization.Localization(__file__, 265, 33), getitem___18971, int_18968)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 265, 30), tuple_18966, subscript_call_result_18972)
    
    # Processing the call keyword arguments (line 265)
    # Getting the type of 'c' (line 265)
    c_18973 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 265, 52), 'c', False)
    # Obtaining the member 'dtype' of a type (line 265)
    dtype_18974 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 265, 52), c_18973, 'dtype')
    keyword_18975 = dtype_18974
    str_18976 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 265, 67), 'str', 'F')
    keyword_18977 = str_18976
    kwargs_18978 = {'dtype': keyword_18975, 'order': keyword_18977}
    # Getting the type of 'numpy' (line 265)
    numpy_18964 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 265, 17), 'numpy', False)
    # Obtaining the member 'zeros' of a type (line 265)
    zeros_18965 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 265, 17), numpy_18964, 'zeros')
    # Calling zeros(args, kwargs) (line 265)
    zeros_call_result_18979 = invoke(stypy.reporting.localization.Localization(__file__, 265, 17), zeros_18965, *[tuple_18966], **kwargs_18978)
    
    # Assigning a type to the variable 'cc' (line 265)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 265, 12), 'cc', zeros_call_result_18979)
    
    # Assigning a Name to a Subscript (line 266):
    
    # Assigning a Name to a Subscript (line 266):
    # Getting the type of 'c' (line 266)
    c_18980 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 266, 24), 'c')
    # Getting the type of 'cc' (line 266)
    cc_18981 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 266, 12), 'cc')
    # Getting the type of 'N' (line 266)
    N_18982 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 266, 16), 'N')
    slice_18983 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 266, 12), None, N_18982, None)
    slice_18984 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 266, 12), None, None, None)
    # Storing an element on a container (line 266)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 266, 12), cc_18981, ((slice_18983, slice_18984), c_18980))
    
    # Assigning a Str to a Name (line 267):
    
    # Assigning a Str to a Name (line 267):
    str_18985 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 267, 20), 'str', 'N')
    # Assigning a type to the variable 'trans' (line 267)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 267, 12), 'trans', str_18985)
    # SSA join for if statement (line 261)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Getting the type of 'conjugate' (line 268)
    conjugate_18986 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 268, 11), 'conjugate')
    # Testing the type of an if condition (line 268)
    if_condition_18987 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 268, 8), conjugate_18986)
    # Assigning a type to the variable 'if_condition_18987' (line 268)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 268, 8), 'if_condition_18987', if_condition_18987)
    # SSA begins for if statement (line 268)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Str to a Name (line 269):
    
    # Assigning a Str to a Name (line 269):
    str_18988 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 269, 17), 'str', 'R')
    # Assigning a type to the variable 'lr' (line 269)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 269, 12), 'lr', str_18988)
    # SSA branch for the else part of an if statement (line 268)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Str to a Name (line 271):
    
    # Assigning a Str to a Name (line 271):
    str_18989 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 271, 17), 'str', 'L')
    # Assigning a type to the variable 'lr' (line 271)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 271, 12), 'lr', str_18989)
    # SSA join for if statement (line 268)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Name to a Name (line 272):
    
    # Assigning a Name to a Name (line 272):
    # Getting the type of 'True' (line 272)
    True_18990 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 272, 22), 'True')
    # Assigning a type to the variable 'overwrite_c' (line 272)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 272, 8), 'overwrite_c', True_18990)
    # SSA branch for the else part of an if statement (line 260)
    module_type_store.open_ssa_branch('else')
    
    
    # Evaluating a boolean operation
    
    # Evaluating a boolean operation
    
    # Obtaining the type of the subscript
    str_18991 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 273, 17), 'str', 'C_CONTIGUOUS')
    # Getting the type of 'c' (line 273)
    c_18992 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 273, 9), 'c')
    # Obtaining the member 'flags' of a type (line 273)
    flags_18993 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 273, 9), c_18992, 'flags')
    # Obtaining the member '__getitem__' of a type (line 273)
    getitem___18994 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 273, 9), flags_18993, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 273)
    subscript_call_result_18995 = invoke(stypy.reporting.localization.Localization(__file__, 273, 9), getitem___18994, str_18991)
    
    
    # Getting the type of 'trans' (line 273)
    trans_18996 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 273, 37), 'trans')
    str_18997 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 273, 46), 'str', 'T')
    # Applying the binary operator '==' (line 273)
    result_eq_18998 = python_operator(stypy.reporting.localization.Localization(__file__, 273, 37), '==', trans_18996, str_18997)
    
    # Applying the binary operator 'and' (line 273)
    result_and_keyword_18999 = python_operator(stypy.reporting.localization.Localization(__file__, 273, 9), 'and', subscript_call_result_18995, result_eq_18998)
    
    # Getting the type of 'conjugate' (line 273)
    conjugate_19000 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 273, 53), 'conjugate')
    # Applying the binary operator 'or' (line 273)
    result_or_keyword_19001 = python_operator(stypy.reporting.localization.Localization(__file__, 273, 9), 'or', result_and_keyword_18999, conjugate_19000)
    
    # Testing the type of an if condition (line 273)
    if_condition_19002 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 273, 9), result_or_keyword_19001)
    # Assigning a type to the variable 'if_condition_19002' (line 273)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 273, 9), 'if_condition_19002', if_condition_19002)
    # SSA begins for if statement (line 273)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Attribute to a Name (line 274):
    
    # Assigning a Attribute to a Name (line 274):
    # Getting the type of 'c' (line 274)
    c_19003 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 274, 13), 'c')
    # Obtaining the member 'T' of a type (line 274)
    T_19004 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 274, 13), c_19003, 'T')
    # Assigning a type to the variable 'cc' (line 274)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 274, 8), 'cc', T_19004)
    
    
    # Getting the type of 'mode' (line 275)
    mode_19005 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 275, 11), 'mode')
    str_19006 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 275, 19), 'str', 'left')
    # Applying the binary operator '==' (line 275)
    result_eq_19007 = python_operator(stypy.reporting.localization.Localization(__file__, 275, 11), '==', mode_19005, str_19006)
    
    # Testing the type of an if condition (line 275)
    if_condition_19008 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 275, 8), result_eq_19007)
    # Assigning a type to the variable 'if_condition_19008' (line 275)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 275, 8), 'if_condition_19008', if_condition_19008)
    # SSA begins for if statement (line 275)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Str to a Name (line 276):
    
    # Assigning a Str to a Name (line 276):
    str_19009 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 276, 17), 'str', 'R')
    # Assigning a type to the variable 'lr' (line 276)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 276, 12), 'lr', str_19009)
    # SSA branch for the else part of an if statement (line 275)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Str to a Name (line 278):
    
    # Assigning a Str to a Name (line 278):
    str_19010 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 278, 17), 'str', 'L')
    # Assigning a type to the variable 'lr' (line 278)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 278, 12), 'lr', str_19010)
    # SSA join for if statement (line 275)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA branch for the else part of an if statement (line 273)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Str to a Name (line 280):
    
    # Assigning a Str to a Name (line 280):
    str_19011 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 280, 16), 'str', 'N')
    # Assigning a type to the variable 'trans' (line 280)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 280, 8), 'trans', str_19011)
    
    # Assigning a Name to a Name (line 281):
    
    # Assigning a Name to a Name (line 281):
    # Getting the type of 'c' (line 281)
    c_19012 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 281, 13), 'c')
    # Assigning a type to the variable 'cc' (line 281)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 281, 8), 'cc', c_19012)
    
    
    # Getting the type of 'mode' (line 282)
    mode_19013 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 282, 11), 'mode')
    str_19014 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 282, 19), 'str', 'left')
    # Applying the binary operator '==' (line 282)
    result_eq_19015 = python_operator(stypy.reporting.localization.Localization(__file__, 282, 11), '==', mode_19013, str_19014)
    
    # Testing the type of an if condition (line 282)
    if_condition_19016 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 282, 8), result_eq_19015)
    # Assigning a type to the variable 'if_condition_19016' (line 282)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 282, 8), 'if_condition_19016', if_condition_19016)
    # SSA begins for if statement (line 282)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Str to a Name (line 283):
    
    # Assigning a Str to a Name (line 283):
    str_19017 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 283, 17), 'str', 'L')
    # Assigning a type to the variable 'lr' (line 283)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 283, 12), 'lr', str_19017)
    # SSA branch for the else part of an if statement (line 282)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Str to a Name (line 285):
    
    # Assigning a Str to a Name (line 285):
    str_19018 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 285, 17), 'str', 'R')
    # Assigning a type to the variable 'lr' (line 285)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 285, 12), 'lr', str_19018)
    # SSA join for if statement (line 282)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 273)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 260)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Tuple (line 286):
    
    # Assigning a Subscript to a Name (line 286):
    
    # Obtaining the type of the subscript
    int_19019 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 286, 4), 'int')
    
    # Call to safecall(...): (line 286)
    # Processing the call arguments (line 286)
    # Getting the type of 'gor_un_mqr' (line 286)
    gor_un_mqr_19021 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 286, 19), 'gor_un_mqr', False)
    str_19022 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 286, 31), 'str', 'gormqr/gunmqr')
    # Getting the type of 'lr' (line 286)
    lr_19023 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 286, 48), 'lr', False)
    # Getting the type of 'trans' (line 286)
    trans_19024 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 286, 52), 'trans', False)
    # Getting the type of 'Q' (line 286)
    Q_19025 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 286, 59), 'Q', False)
    # Getting the type of 'tau' (line 286)
    tau_19026 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 286, 62), 'tau', False)
    # Getting the type of 'cc' (line 286)
    cc_19027 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 286, 67), 'cc', False)
    # Processing the call keyword arguments (line 286)
    # Getting the type of 'overwrite_c' (line 287)
    overwrite_c_19028 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 287, 24), 'overwrite_c', False)
    keyword_19029 = overwrite_c_19028
    kwargs_19030 = {'overwrite_c': keyword_19029}
    # Getting the type of 'safecall' (line 286)
    safecall_19020 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 286, 10), 'safecall', False)
    # Calling safecall(args, kwargs) (line 286)
    safecall_call_result_19031 = invoke(stypy.reporting.localization.Localization(__file__, 286, 10), safecall_19020, *[gor_un_mqr_19021, str_19022, lr_19023, trans_19024, Q_19025, tau_19026, cc_19027], **kwargs_19030)
    
    # Obtaining the member '__getitem__' of a type (line 286)
    getitem___19032 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 286, 4), safecall_call_result_19031, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 286)
    subscript_call_result_19033 = invoke(stypy.reporting.localization.Localization(__file__, 286, 4), getitem___19032, int_19019)
    
    # Assigning a type to the variable 'tuple_var_assignment_18392' (line 286)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 286, 4), 'tuple_var_assignment_18392', subscript_call_result_19033)
    
    # Assigning a Name to a Name (line 286):
    # Getting the type of 'tuple_var_assignment_18392' (line 286)
    tuple_var_assignment_18392_19034 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 286, 4), 'tuple_var_assignment_18392')
    # Assigning a type to the variable 'cQ' (line 286)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 286, 4), 'cQ', tuple_var_assignment_18392_19034)
    
    
    # Getting the type of 'trans' (line 288)
    trans_19035 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 288, 7), 'trans')
    str_19036 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 288, 16), 'str', 'N')
    # Applying the binary operator '!=' (line 288)
    result_ne_19037 = python_operator(stypy.reporting.localization.Localization(__file__, 288, 7), '!=', trans_19035, str_19036)
    
    # Testing the type of an if condition (line 288)
    if_condition_19038 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 288, 4), result_ne_19037)
    # Assigning a type to the variable 'if_condition_19038' (line 288)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 288, 4), 'if_condition_19038', if_condition_19038)
    # SSA begins for if statement (line 288)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Attribute to a Name (line 289):
    
    # Assigning a Attribute to a Name (line 289):
    # Getting the type of 'cQ' (line 289)
    cQ_19039 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 289, 13), 'cQ')
    # Obtaining the member 'T' of a type (line 289)
    T_19040 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 289, 13), cQ_19039, 'T')
    # Assigning a type to the variable 'cQ' (line 289)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 289, 8), 'cQ', T_19040)
    # SSA join for if statement (line 288)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'mode' (line 290)
    mode_19041 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 290, 7), 'mode')
    str_19042 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 290, 15), 'str', 'right')
    # Applying the binary operator '==' (line 290)
    result_eq_19043 = python_operator(stypy.reporting.localization.Localization(__file__, 290, 7), '==', mode_19041, str_19042)
    
    # Testing the type of an if condition (line 290)
    if_condition_19044 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 290, 4), result_eq_19043)
    # Assigning a type to the variable 'if_condition_19044' (line 290)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 290, 4), 'if_condition_19044', if_condition_19044)
    # SSA begins for if statement (line 290)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Subscript to a Name (line 291):
    
    # Assigning a Subscript to a Name (line 291):
    
    # Obtaining the type of the subscript
    slice_19045 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 291, 13), None, None, None)
    
    # Call to min(...): (line 291)
    # Processing the call arguments (line 291)
    # Getting the type of 'M' (line 291)
    M_19047 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 291, 24), 'M', False)
    # Getting the type of 'N' (line 291)
    N_19048 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 291, 27), 'N', False)
    # Processing the call keyword arguments (line 291)
    kwargs_19049 = {}
    # Getting the type of 'min' (line 291)
    min_19046 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 291, 20), 'min', False)
    # Calling min(args, kwargs) (line 291)
    min_call_result_19050 = invoke(stypy.reporting.localization.Localization(__file__, 291, 20), min_19046, *[M_19047, N_19048], **kwargs_19049)
    
    slice_19051 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 291, 13), None, min_call_result_19050, None)
    # Getting the type of 'cQ' (line 291)
    cQ_19052 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 291, 13), 'cQ')
    # Obtaining the member '__getitem__' of a type (line 291)
    getitem___19053 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 291, 13), cQ_19052, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 291)
    subscript_call_result_19054 = invoke(stypy.reporting.localization.Localization(__file__, 291, 13), getitem___19053, (slice_19045, slice_19051))
    
    # Assigning a type to the variable 'cQ' (line 291)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 291, 8), 'cQ', subscript_call_result_19054)
    # SSA join for if statement (line 290)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Getting the type of 'onedim' (line 292)
    onedim_19055 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 292, 7), 'onedim')
    # Testing the type of an if condition (line 292)
    if_condition_19056 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 292, 4), onedim_19055)
    # Assigning a type to the variable 'if_condition_19056' (line 292)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 292, 4), 'if_condition_19056', if_condition_19056)
    # SSA begins for if statement (line 292)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 293):
    
    # Assigning a Call to a Name (line 293):
    
    # Call to ravel(...): (line 293)
    # Processing the call keyword arguments (line 293)
    kwargs_19059 = {}
    # Getting the type of 'cQ' (line 293)
    cQ_19057 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 293, 13), 'cQ', False)
    # Obtaining the member 'ravel' of a type (line 293)
    ravel_19058 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 293, 13), cQ_19057, 'ravel')
    # Calling ravel(args, kwargs) (line 293)
    ravel_call_result_19060 = invoke(stypy.reporting.localization.Localization(__file__, 293, 13), ravel_19058, *[], **kwargs_19059)
    
    # Assigning a type to the variable 'cQ' (line 293)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 293, 8), 'cQ', ravel_call_result_19060)
    # SSA join for if statement (line 292)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Obtaining an instance of the builtin type 'tuple' (line 295)
    tuple_19061 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 295, 12), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 295)
    # Adding element type (line 295)
    # Getting the type of 'cQ' (line 295)
    cQ_19062 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 295, 12), 'cQ')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 295, 12), tuple_19061, cQ_19062)
    
    
    # Obtaining the type of the subscript
    int_19063 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 295, 23), 'int')
    slice_19064 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 295, 19), int_19063, None, None)
    # Getting the type of 'raw' (line 295)
    raw_19065 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 295, 19), 'raw')
    # Obtaining the member '__getitem__' of a type (line 295)
    getitem___19066 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 295, 19), raw_19065, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 295)
    subscript_call_result_19067 = invoke(stypy.reporting.localization.Localization(__file__, 295, 19), getitem___19066, slice_19064)
    
    # Applying the binary operator '+' (line 295)
    result_add_19068 = python_operator(stypy.reporting.localization.Localization(__file__, 295, 11), '+', tuple_19061, subscript_call_result_19067)
    
    # Assigning a type to the variable 'stypy_return_type' (line 295)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 295, 4), 'stypy_return_type', result_add_19068)
    
    # ################# End of 'qr_multiply(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'qr_multiply' in the type store
    # Getting the type of 'stypy_return_type' (line 176)
    stypy_return_type_19069 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_19069)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'qr_multiply'
    return stypy_return_type_19069

# Assigning a type to the variable 'qr_multiply' (line 176)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 176, 0), 'qr_multiply', qr_multiply)

@norecursion
def rq(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'False' (line 298)
    False_19070 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 298, 22), 'False')
    # Getting the type of 'None' (line 298)
    None_19071 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 298, 35), 'None')
    str_19072 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 298, 46), 'str', 'full')
    # Getting the type of 'True' (line 298)
    True_19073 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 298, 67), 'True')
    defaults = [False_19070, None_19071, str_19072, True_19073]
    # Create a new context for function 'rq'
    module_type_store = module_type_store.open_function_context('rq', 298, 0, False)
    
    # Passed parameters checking function
    rq.stypy_localization = localization
    rq.stypy_type_of_self = None
    rq.stypy_type_store = module_type_store
    rq.stypy_function_name = 'rq'
    rq.stypy_param_names_list = ['a', 'overwrite_a', 'lwork', 'mode', 'check_finite']
    rq.stypy_varargs_param_name = None
    rq.stypy_kwargs_param_name = None
    rq.stypy_call_defaults = defaults
    rq.stypy_call_varargs = varargs
    rq.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'rq', ['a', 'overwrite_a', 'lwork', 'mode', 'check_finite'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'rq', localization, ['a', 'overwrite_a', 'lwork', 'mode', 'check_finite'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'rq(...)' code ##################

    str_19074 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 361, (-1)), 'str', "\n    Compute RQ decomposition of a matrix.\n\n    Calculate the decomposition ``A = R Q`` where Q is unitary/orthogonal\n    and R upper triangular.\n\n    Parameters\n    ----------\n    a : (M, N) array_like\n        Matrix to be decomposed\n    overwrite_a : bool, optional\n        Whether data in a is overwritten (may improve performance)\n    lwork : int, optional\n        Work array size, lwork >= a.shape[1]. If None or -1, an optimal size\n        is computed.\n    mode : {'full', 'r', 'economic'}, optional\n        Determines what information is to be returned: either both Q and R\n        ('full', default), only R ('r') or both Q and R but computed in\n        economy-size ('economic', see Notes).\n    check_finite : bool, optional\n        Whether to check that the input matrix contains only finite numbers.\n        Disabling may give a performance gain, but may result in problems\n        (crashes, non-termination) if the inputs do contain infinities or NaNs.\n\n    Returns\n    -------\n    R : float or complex ndarray\n        Of shape (M, N) or (M, K) for ``mode='economic'``.  ``K = min(M, N)``.\n    Q : float or complex ndarray\n        Of shape (N, N) or (K, N) for ``mode='economic'``.  Not returned\n        if ``mode='r'``.\n\n    Raises\n    ------\n    LinAlgError\n        If decomposition fails.\n\n    Notes\n    -----\n    This is an interface to the LAPACK routines sgerqf, dgerqf, cgerqf, zgerqf,\n    sorgrq, dorgrq, cungrq and zungrq.\n\n    If ``mode=economic``, the shapes of Q and R are (K, N) and (M, K) instead\n    of (N,N) and (M,N), with ``K=min(M,N)``.\n\n    Examples\n    --------\n    >>> from scipy import linalg\n    >>> from numpy import random, dot, allclose\n    >>> a = random.randn(6, 9)\n    >>> r, q = linalg.rq(a)\n    >>> allclose(a, dot(r, q))\n    True\n    >>> r.shape, q.shape\n    ((6, 9), (9, 9))\n    >>> r2 = linalg.rq(a, mode='r')\n    >>> allclose(r, r2)\n    True\n    >>> r3, q3 = linalg.rq(a, mode='economic')\n    >>> r3.shape, q3.shape\n    ((6, 6), (6, 9))\n\n    ")
    
    
    # Getting the type of 'mode' (line 362)
    mode_19075 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 362, 7), 'mode')
    
    # Obtaining an instance of the builtin type 'list' (line 362)
    list_19076 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 362, 19), 'list')
    # Adding type elements to the builtin type 'list' instance (line 362)
    # Adding element type (line 362)
    str_19077 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 362, 20), 'str', 'full')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 362, 19), list_19076, str_19077)
    # Adding element type (line 362)
    str_19078 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 362, 28), 'str', 'r')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 362, 19), list_19076, str_19078)
    # Adding element type (line 362)
    str_19079 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 362, 33), 'str', 'economic')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 362, 19), list_19076, str_19079)
    
    # Applying the binary operator 'notin' (line 362)
    result_contains_19080 = python_operator(stypy.reporting.localization.Localization(__file__, 362, 7), 'notin', mode_19075, list_19076)
    
    # Testing the type of an if condition (line 362)
    if_condition_19081 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 362, 4), result_contains_19080)
    # Assigning a type to the variable 'if_condition_19081' (line 362)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 362, 4), 'if_condition_19081', if_condition_19081)
    # SSA begins for if statement (line 362)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 363)
    # Processing the call arguments (line 363)
    str_19083 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 364, 17), 'str', "Mode argument should be one of ['full', 'r', 'economic']")
    # Processing the call keyword arguments (line 363)
    kwargs_19084 = {}
    # Getting the type of 'ValueError' (line 363)
    ValueError_19082 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 363, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 363)
    ValueError_call_result_19085 = invoke(stypy.reporting.localization.Localization(__file__, 363, 14), ValueError_19082, *[str_19083], **kwargs_19084)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 363, 8), ValueError_call_result_19085, 'raise parameter', BaseException)
    # SSA join for if statement (line 362)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Getting the type of 'check_finite' (line 366)
    check_finite_19086 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 366, 7), 'check_finite')
    # Testing the type of an if condition (line 366)
    if_condition_19087 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 366, 4), check_finite_19086)
    # Assigning a type to the variable 'if_condition_19087' (line 366)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 366, 4), 'if_condition_19087', if_condition_19087)
    # SSA begins for if statement (line 366)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 367):
    
    # Assigning a Call to a Name (line 367):
    
    # Call to asarray_chkfinite(...): (line 367)
    # Processing the call arguments (line 367)
    # Getting the type of 'a' (line 367)
    a_19090 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 367, 37), 'a', False)
    # Processing the call keyword arguments (line 367)
    kwargs_19091 = {}
    # Getting the type of 'numpy' (line 367)
    numpy_19088 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 367, 13), 'numpy', False)
    # Obtaining the member 'asarray_chkfinite' of a type (line 367)
    asarray_chkfinite_19089 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 367, 13), numpy_19088, 'asarray_chkfinite')
    # Calling asarray_chkfinite(args, kwargs) (line 367)
    asarray_chkfinite_call_result_19092 = invoke(stypy.reporting.localization.Localization(__file__, 367, 13), asarray_chkfinite_19089, *[a_19090], **kwargs_19091)
    
    # Assigning a type to the variable 'a1' (line 367)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 367, 8), 'a1', asarray_chkfinite_call_result_19092)
    # SSA branch for the else part of an if statement (line 366)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Call to a Name (line 369):
    
    # Assigning a Call to a Name (line 369):
    
    # Call to asarray(...): (line 369)
    # Processing the call arguments (line 369)
    # Getting the type of 'a' (line 369)
    a_19095 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 369, 27), 'a', False)
    # Processing the call keyword arguments (line 369)
    kwargs_19096 = {}
    # Getting the type of 'numpy' (line 369)
    numpy_19093 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 369, 13), 'numpy', False)
    # Obtaining the member 'asarray' of a type (line 369)
    asarray_19094 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 369, 13), numpy_19093, 'asarray')
    # Calling asarray(args, kwargs) (line 369)
    asarray_call_result_19097 = invoke(stypy.reporting.localization.Localization(__file__, 369, 13), asarray_19094, *[a_19095], **kwargs_19096)
    
    # Assigning a type to the variable 'a1' (line 369)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 369, 8), 'a1', asarray_call_result_19097)
    # SSA join for if statement (line 366)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    
    # Call to len(...): (line 370)
    # Processing the call arguments (line 370)
    # Getting the type of 'a1' (line 370)
    a1_19099 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 370, 11), 'a1', False)
    # Obtaining the member 'shape' of a type (line 370)
    shape_19100 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 370, 11), a1_19099, 'shape')
    # Processing the call keyword arguments (line 370)
    kwargs_19101 = {}
    # Getting the type of 'len' (line 370)
    len_19098 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 370, 7), 'len', False)
    # Calling len(args, kwargs) (line 370)
    len_call_result_19102 = invoke(stypy.reporting.localization.Localization(__file__, 370, 7), len_19098, *[shape_19100], **kwargs_19101)
    
    int_19103 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 370, 24), 'int')
    # Applying the binary operator '!=' (line 370)
    result_ne_19104 = python_operator(stypy.reporting.localization.Localization(__file__, 370, 7), '!=', len_call_result_19102, int_19103)
    
    # Testing the type of an if condition (line 370)
    if_condition_19105 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 370, 4), result_ne_19104)
    # Assigning a type to the variable 'if_condition_19105' (line 370)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 370, 4), 'if_condition_19105', if_condition_19105)
    # SSA begins for if statement (line 370)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 371)
    # Processing the call arguments (line 371)
    str_19107 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 371, 25), 'str', 'expected matrix')
    # Processing the call keyword arguments (line 371)
    kwargs_19108 = {}
    # Getting the type of 'ValueError' (line 371)
    ValueError_19106 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 371, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 371)
    ValueError_call_result_19109 = invoke(stypy.reporting.localization.Localization(__file__, 371, 14), ValueError_19106, *[str_19107], **kwargs_19108)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 371, 8), ValueError_call_result_19109, 'raise parameter', BaseException)
    # SSA join for if statement (line 370)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Attribute to a Tuple (line 372):
    
    # Assigning a Subscript to a Name (line 372):
    
    # Obtaining the type of the subscript
    int_19110 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 372, 4), 'int')
    # Getting the type of 'a1' (line 372)
    a1_19111 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 372, 11), 'a1')
    # Obtaining the member 'shape' of a type (line 372)
    shape_19112 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 372, 11), a1_19111, 'shape')
    # Obtaining the member '__getitem__' of a type (line 372)
    getitem___19113 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 372, 4), shape_19112, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 372)
    subscript_call_result_19114 = invoke(stypy.reporting.localization.Localization(__file__, 372, 4), getitem___19113, int_19110)
    
    # Assigning a type to the variable 'tuple_var_assignment_18393' (line 372)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 372, 4), 'tuple_var_assignment_18393', subscript_call_result_19114)
    
    # Assigning a Subscript to a Name (line 372):
    
    # Obtaining the type of the subscript
    int_19115 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 372, 4), 'int')
    # Getting the type of 'a1' (line 372)
    a1_19116 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 372, 11), 'a1')
    # Obtaining the member 'shape' of a type (line 372)
    shape_19117 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 372, 11), a1_19116, 'shape')
    # Obtaining the member '__getitem__' of a type (line 372)
    getitem___19118 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 372, 4), shape_19117, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 372)
    subscript_call_result_19119 = invoke(stypy.reporting.localization.Localization(__file__, 372, 4), getitem___19118, int_19115)
    
    # Assigning a type to the variable 'tuple_var_assignment_18394' (line 372)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 372, 4), 'tuple_var_assignment_18394', subscript_call_result_19119)
    
    # Assigning a Name to a Name (line 372):
    # Getting the type of 'tuple_var_assignment_18393' (line 372)
    tuple_var_assignment_18393_19120 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 372, 4), 'tuple_var_assignment_18393')
    # Assigning a type to the variable 'M' (line 372)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 372, 4), 'M', tuple_var_assignment_18393_19120)
    
    # Assigning a Name to a Name (line 372):
    # Getting the type of 'tuple_var_assignment_18394' (line 372)
    tuple_var_assignment_18394_19121 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 372, 4), 'tuple_var_assignment_18394')
    # Assigning a type to the variable 'N' (line 372)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 372, 7), 'N', tuple_var_assignment_18394_19121)
    
    # Assigning a BoolOp to a Name (line 373):
    
    # Assigning a BoolOp to a Name (line 373):
    
    # Evaluating a boolean operation
    # Getting the type of 'overwrite_a' (line 373)
    overwrite_a_19122 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 373, 18), 'overwrite_a')
    
    # Call to _datacopied(...): (line 373)
    # Processing the call arguments (line 373)
    # Getting the type of 'a1' (line 373)
    a1_19124 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 373, 46), 'a1', False)
    # Getting the type of 'a' (line 373)
    a_19125 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 373, 50), 'a', False)
    # Processing the call keyword arguments (line 373)
    kwargs_19126 = {}
    # Getting the type of '_datacopied' (line 373)
    _datacopied_19123 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 373, 34), '_datacopied', False)
    # Calling _datacopied(args, kwargs) (line 373)
    _datacopied_call_result_19127 = invoke(stypy.reporting.localization.Localization(__file__, 373, 34), _datacopied_19123, *[a1_19124, a_19125], **kwargs_19126)
    
    # Applying the binary operator 'or' (line 373)
    result_or_keyword_19128 = python_operator(stypy.reporting.localization.Localization(__file__, 373, 18), 'or', overwrite_a_19122, _datacopied_call_result_19127)
    
    # Assigning a type to the variable 'overwrite_a' (line 373)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 373, 4), 'overwrite_a', result_or_keyword_19128)
    
    # Assigning a Call to a Tuple (line 375):
    
    # Assigning a Subscript to a Name (line 375):
    
    # Obtaining the type of the subscript
    int_19129 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 375, 4), 'int')
    
    # Call to get_lapack_funcs(...): (line 375)
    # Processing the call arguments (line 375)
    
    # Obtaining an instance of the builtin type 'tuple' (line 375)
    tuple_19131 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 375, 31), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 375)
    # Adding element type (line 375)
    str_19132 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 375, 31), 'str', 'gerqf')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 375, 31), tuple_19131, str_19132)
    
    
    # Obtaining an instance of the builtin type 'tuple' (line 375)
    tuple_19133 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 375, 43), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 375)
    # Adding element type (line 375)
    # Getting the type of 'a1' (line 375)
    a1_19134 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 375, 43), 'a1', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 375, 43), tuple_19133, a1_19134)
    
    # Processing the call keyword arguments (line 375)
    kwargs_19135 = {}
    # Getting the type of 'get_lapack_funcs' (line 375)
    get_lapack_funcs_19130 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 375, 13), 'get_lapack_funcs', False)
    # Calling get_lapack_funcs(args, kwargs) (line 375)
    get_lapack_funcs_call_result_19136 = invoke(stypy.reporting.localization.Localization(__file__, 375, 13), get_lapack_funcs_19130, *[tuple_19131, tuple_19133], **kwargs_19135)
    
    # Obtaining the member '__getitem__' of a type (line 375)
    getitem___19137 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 375, 4), get_lapack_funcs_call_result_19136, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 375)
    subscript_call_result_19138 = invoke(stypy.reporting.localization.Localization(__file__, 375, 4), getitem___19137, int_19129)
    
    # Assigning a type to the variable 'tuple_var_assignment_18395' (line 375)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 375, 4), 'tuple_var_assignment_18395', subscript_call_result_19138)
    
    # Assigning a Name to a Name (line 375):
    # Getting the type of 'tuple_var_assignment_18395' (line 375)
    tuple_var_assignment_18395_19139 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 375, 4), 'tuple_var_assignment_18395')
    # Assigning a type to the variable 'gerqf' (line 375)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 375, 4), 'gerqf', tuple_var_assignment_18395_19139)
    
    # Assigning a Call to a Tuple (line 376):
    
    # Assigning a Subscript to a Name (line 376):
    
    # Obtaining the type of the subscript
    int_19140 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 376, 4), 'int')
    
    # Call to safecall(...): (line 376)
    # Processing the call arguments (line 376)
    # Getting the type of 'gerqf' (line 376)
    gerqf_19142 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 376, 23), 'gerqf', False)
    str_19143 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 376, 30), 'str', 'gerqf')
    # Getting the type of 'a1' (line 376)
    a1_19144 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 376, 39), 'a1', False)
    # Processing the call keyword arguments (line 376)
    # Getting the type of 'lwork' (line 376)
    lwork_19145 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 376, 49), 'lwork', False)
    keyword_19146 = lwork_19145
    # Getting the type of 'overwrite_a' (line 377)
    overwrite_a_19147 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 377, 35), 'overwrite_a', False)
    keyword_19148 = overwrite_a_19147
    kwargs_19149 = {'overwrite_a': keyword_19148, 'lwork': keyword_19146}
    # Getting the type of 'safecall' (line 376)
    safecall_19141 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 376, 14), 'safecall', False)
    # Calling safecall(args, kwargs) (line 376)
    safecall_call_result_19150 = invoke(stypy.reporting.localization.Localization(__file__, 376, 14), safecall_19141, *[gerqf_19142, str_19143, a1_19144], **kwargs_19149)
    
    # Obtaining the member '__getitem__' of a type (line 376)
    getitem___19151 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 376, 4), safecall_call_result_19150, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 376)
    subscript_call_result_19152 = invoke(stypy.reporting.localization.Localization(__file__, 376, 4), getitem___19151, int_19140)
    
    # Assigning a type to the variable 'tuple_var_assignment_18396' (line 376)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 376, 4), 'tuple_var_assignment_18396', subscript_call_result_19152)
    
    # Assigning a Subscript to a Name (line 376):
    
    # Obtaining the type of the subscript
    int_19153 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 376, 4), 'int')
    
    # Call to safecall(...): (line 376)
    # Processing the call arguments (line 376)
    # Getting the type of 'gerqf' (line 376)
    gerqf_19155 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 376, 23), 'gerqf', False)
    str_19156 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 376, 30), 'str', 'gerqf')
    # Getting the type of 'a1' (line 376)
    a1_19157 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 376, 39), 'a1', False)
    # Processing the call keyword arguments (line 376)
    # Getting the type of 'lwork' (line 376)
    lwork_19158 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 376, 49), 'lwork', False)
    keyword_19159 = lwork_19158
    # Getting the type of 'overwrite_a' (line 377)
    overwrite_a_19160 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 377, 35), 'overwrite_a', False)
    keyword_19161 = overwrite_a_19160
    kwargs_19162 = {'overwrite_a': keyword_19161, 'lwork': keyword_19159}
    # Getting the type of 'safecall' (line 376)
    safecall_19154 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 376, 14), 'safecall', False)
    # Calling safecall(args, kwargs) (line 376)
    safecall_call_result_19163 = invoke(stypy.reporting.localization.Localization(__file__, 376, 14), safecall_19154, *[gerqf_19155, str_19156, a1_19157], **kwargs_19162)
    
    # Obtaining the member '__getitem__' of a type (line 376)
    getitem___19164 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 376, 4), safecall_call_result_19163, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 376)
    subscript_call_result_19165 = invoke(stypy.reporting.localization.Localization(__file__, 376, 4), getitem___19164, int_19153)
    
    # Assigning a type to the variable 'tuple_var_assignment_18397' (line 376)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 376, 4), 'tuple_var_assignment_18397', subscript_call_result_19165)
    
    # Assigning a Name to a Name (line 376):
    # Getting the type of 'tuple_var_assignment_18396' (line 376)
    tuple_var_assignment_18396_19166 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 376, 4), 'tuple_var_assignment_18396')
    # Assigning a type to the variable 'rq' (line 376)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 376, 4), 'rq', tuple_var_assignment_18396_19166)
    
    # Assigning a Name to a Name (line 376):
    # Getting the type of 'tuple_var_assignment_18397' (line 376)
    tuple_var_assignment_18397_19167 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 376, 4), 'tuple_var_assignment_18397')
    # Assigning a type to the variable 'tau' (line 376)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 376, 8), 'tau', tuple_var_assignment_18397_19167)
    
    
    # Evaluating a boolean operation
    
    
    # Getting the type of 'mode' (line 378)
    mode_19168 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 378, 11), 'mode')
    str_19169 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 378, 19), 'str', 'economic')
    # Applying the binary operator '==' (line 378)
    result_eq_19170 = python_operator(stypy.reporting.localization.Localization(__file__, 378, 11), '==', mode_19168, str_19169)
    
    # Applying the 'not' unary operator (line 378)
    result_not__19171 = python_operator(stypy.reporting.localization.Localization(__file__, 378, 7), 'not', result_eq_19170)
    
    
    # Getting the type of 'N' (line 378)
    N_19172 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 378, 33), 'N')
    # Getting the type of 'M' (line 378)
    M_19173 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 378, 37), 'M')
    # Applying the binary operator '<' (line 378)
    result_lt_19174 = python_operator(stypy.reporting.localization.Localization(__file__, 378, 33), '<', N_19172, M_19173)
    
    # Applying the binary operator 'or' (line 378)
    result_or_keyword_19175 = python_operator(stypy.reporting.localization.Localization(__file__, 378, 7), 'or', result_not__19171, result_lt_19174)
    
    # Testing the type of an if condition (line 378)
    if_condition_19176 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 378, 4), result_or_keyword_19175)
    # Assigning a type to the variable 'if_condition_19176' (line 378)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 378, 4), 'if_condition_19176', if_condition_19176)
    # SSA begins for if statement (line 378)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 379):
    
    # Assigning a Call to a Name (line 379):
    
    # Call to triu(...): (line 379)
    # Processing the call arguments (line 379)
    # Getting the type of 'rq' (line 379)
    rq_19179 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 379, 23), 'rq', False)
    # Getting the type of 'N' (line 379)
    N_19180 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 379, 27), 'N', False)
    # Getting the type of 'M' (line 379)
    M_19181 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 379, 29), 'M', False)
    # Applying the binary operator '-' (line 379)
    result_sub_19182 = python_operator(stypy.reporting.localization.Localization(__file__, 379, 27), '-', N_19180, M_19181)
    
    # Processing the call keyword arguments (line 379)
    kwargs_19183 = {}
    # Getting the type of 'numpy' (line 379)
    numpy_19177 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 379, 12), 'numpy', False)
    # Obtaining the member 'triu' of a type (line 379)
    triu_19178 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 379, 12), numpy_19177, 'triu')
    # Calling triu(args, kwargs) (line 379)
    triu_call_result_19184 = invoke(stypy.reporting.localization.Localization(__file__, 379, 12), triu_19178, *[rq_19179, result_sub_19182], **kwargs_19183)
    
    # Assigning a type to the variable 'R' (line 379)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 379, 8), 'R', triu_call_result_19184)
    # SSA branch for the else part of an if statement (line 378)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Call to a Name (line 381):
    
    # Assigning a Call to a Name (line 381):
    
    # Call to triu(...): (line 381)
    # Processing the call arguments (line 381)
    
    # Obtaining the type of the subscript
    
    # Getting the type of 'M' (line 381)
    M_19187 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 381, 27), 'M', False)
    # Applying the 'usub' unary operator (line 381)
    result___neg___19188 = python_operator(stypy.reporting.localization.Localization(__file__, 381, 26), 'usub', M_19187)
    
    slice_19189 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 381, 23), result___neg___19188, None, None)
    
    # Getting the type of 'M' (line 381)
    M_19190 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 381, 32), 'M', False)
    # Applying the 'usub' unary operator (line 381)
    result___neg___19191 = python_operator(stypy.reporting.localization.Localization(__file__, 381, 31), 'usub', M_19190)
    
    slice_19192 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 381, 23), result___neg___19191, None, None)
    # Getting the type of 'rq' (line 381)
    rq_19193 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 381, 23), 'rq', False)
    # Obtaining the member '__getitem__' of a type (line 381)
    getitem___19194 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 381, 23), rq_19193, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 381)
    subscript_call_result_19195 = invoke(stypy.reporting.localization.Localization(__file__, 381, 23), getitem___19194, (slice_19189, slice_19192))
    
    # Processing the call keyword arguments (line 381)
    kwargs_19196 = {}
    # Getting the type of 'numpy' (line 381)
    numpy_19185 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 381, 12), 'numpy', False)
    # Obtaining the member 'triu' of a type (line 381)
    triu_19186 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 381, 12), numpy_19185, 'triu')
    # Calling triu(args, kwargs) (line 381)
    triu_call_result_19197 = invoke(stypy.reporting.localization.Localization(__file__, 381, 12), triu_19186, *[subscript_call_result_19195], **kwargs_19196)
    
    # Assigning a type to the variable 'R' (line 381)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 381, 8), 'R', triu_call_result_19197)
    # SSA join for if statement (line 378)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'mode' (line 383)
    mode_19198 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 383, 7), 'mode')
    str_19199 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 383, 15), 'str', 'r')
    # Applying the binary operator '==' (line 383)
    result_eq_19200 = python_operator(stypy.reporting.localization.Localization(__file__, 383, 7), '==', mode_19198, str_19199)
    
    # Testing the type of an if condition (line 383)
    if_condition_19201 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 383, 4), result_eq_19200)
    # Assigning a type to the variable 'if_condition_19201' (line 383)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 383, 4), 'if_condition_19201', if_condition_19201)
    # SSA begins for if statement (line 383)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # Getting the type of 'R' (line 384)
    R_19202 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 384, 15), 'R')
    # Assigning a type to the variable 'stypy_return_type' (line 384)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 384, 8), 'stypy_return_type', R_19202)
    # SSA join for if statement (line 383)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Tuple (line 386):
    
    # Assigning a Subscript to a Name (line 386):
    
    # Obtaining the type of the subscript
    int_19203 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 386, 4), 'int')
    
    # Call to get_lapack_funcs(...): (line 386)
    # Processing the call arguments (line 386)
    
    # Obtaining an instance of the builtin type 'tuple' (line 386)
    tuple_19205 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 386, 36), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 386)
    # Adding element type (line 386)
    str_19206 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 386, 36), 'str', 'orgrq')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 386, 36), tuple_19205, str_19206)
    
    
    # Obtaining an instance of the builtin type 'tuple' (line 386)
    tuple_19207 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 386, 48), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 386)
    # Adding element type (line 386)
    # Getting the type of 'rq' (line 386)
    rq_19208 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 386, 48), 'rq', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 386, 48), tuple_19207, rq_19208)
    
    # Processing the call keyword arguments (line 386)
    kwargs_19209 = {}
    # Getting the type of 'get_lapack_funcs' (line 386)
    get_lapack_funcs_19204 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 386, 18), 'get_lapack_funcs', False)
    # Calling get_lapack_funcs(args, kwargs) (line 386)
    get_lapack_funcs_call_result_19210 = invoke(stypy.reporting.localization.Localization(__file__, 386, 18), get_lapack_funcs_19204, *[tuple_19205, tuple_19207], **kwargs_19209)
    
    # Obtaining the member '__getitem__' of a type (line 386)
    getitem___19211 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 386, 4), get_lapack_funcs_call_result_19210, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 386)
    subscript_call_result_19212 = invoke(stypy.reporting.localization.Localization(__file__, 386, 4), getitem___19211, int_19203)
    
    # Assigning a type to the variable 'tuple_var_assignment_18398' (line 386)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 386, 4), 'tuple_var_assignment_18398', subscript_call_result_19212)
    
    # Assigning a Name to a Name (line 386):
    # Getting the type of 'tuple_var_assignment_18398' (line 386)
    tuple_var_assignment_18398_19213 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 386, 4), 'tuple_var_assignment_18398')
    # Assigning a type to the variable 'gor_un_grq' (line 386)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 386, 4), 'gor_un_grq', tuple_var_assignment_18398_19213)
    
    
    # Getting the type of 'N' (line 388)
    N_19214 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 388, 7), 'N')
    # Getting the type of 'M' (line 388)
    M_19215 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 388, 11), 'M')
    # Applying the binary operator '<' (line 388)
    result_lt_19216 = python_operator(stypy.reporting.localization.Localization(__file__, 388, 7), '<', N_19214, M_19215)
    
    # Testing the type of an if condition (line 388)
    if_condition_19217 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 388, 4), result_lt_19216)
    # Assigning a type to the variable 'if_condition_19217' (line 388)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 388, 4), 'if_condition_19217', if_condition_19217)
    # SSA begins for if statement (line 388)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Tuple (line 389):
    
    # Assigning a Subscript to a Name (line 389):
    
    # Obtaining the type of the subscript
    int_19218 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 389, 8), 'int')
    
    # Call to safecall(...): (line 389)
    # Processing the call arguments (line 389)
    # Getting the type of 'gor_un_grq' (line 389)
    gor_un_grq_19220 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 389, 22), 'gor_un_grq', False)
    str_19221 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 389, 34), 'str', 'gorgrq/gungrq')
    
    # Obtaining the type of the subscript
    
    # Getting the type of 'N' (line 389)
    N_19222 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 389, 55), 'N', False)
    # Applying the 'usub' unary operator (line 389)
    result___neg___19223 = python_operator(stypy.reporting.localization.Localization(__file__, 389, 54), 'usub', N_19222)
    
    slice_19224 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 389, 51), result___neg___19223, None, None)
    # Getting the type of 'rq' (line 389)
    rq_19225 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 389, 51), 'rq', False)
    # Obtaining the member '__getitem__' of a type (line 389)
    getitem___19226 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 389, 51), rq_19225, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 389)
    subscript_call_result_19227 = invoke(stypy.reporting.localization.Localization(__file__, 389, 51), getitem___19226, slice_19224)
    
    # Getting the type of 'tau' (line 389)
    tau_19228 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 389, 60), 'tau', False)
    # Processing the call keyword arguments (line 389)
    # Getting the type of 'lwork' (line 389)
    lwork_19229 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 389, 71), 'lwork', False)
    keyword_19230 = lwork_19229
    int_19231 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 390, 34), 'int')
    keyword_19232 = int_19231
    kwargs_19233 = {'overwrite_a': keyword_19232, 'lwork': keyword_19230}
    # Getting the type of 'safecall' (line 389)
    safecall_19219 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 389, 13), 'safecall', False)
    # Calling safecall(args, kwargs) (line 389)
    safecall_call_result_19234 = invoke(stypy.reporting.localization.Localization(__file__, 389, 13), safecall_19219, *[gor_un_grq_19220, str_19221, subscript_call_result_19227, tau_19228], **kwargs_19233)
    
    # Obtaining the member '__getitem__' of a type (line 389)
    getitem___19235 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 389, 8), safecall_call_result_19234, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 389)
    subscript_call_result_19236 = invoke(stypy.reporting.localization.Localization(__file__, 389, 8), getitem___19235, int_19218)
    
    # Assigning a type to the variable 'tuple_var_assignment_18399' (line 389)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 389, 8), 'tuple_var_assignment_18399', subscript_call_result_19236)
    
    # Assigning a Name to a Name (line 389):
    # Getting the type of 'tuple_var_assignment_18399' (line 389)
    tuple_var_assignment_18399_19237 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 389, 8), 'tuple_var_assignment_18399')
    # Assigning a type to the variable 'Q' (line 389)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 389, 8), 'Q', tuple_var_assignment_18399_19237)
    # SSA branch for the else part of an if statement (line 388)
    module_type_store.open_ssa_branch('else')
    
    
    # Getting the type of 'mode' (line 391)
    mode_19238 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 391, 9), 'mode')
    str_19239 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 391, 17), 'str', 'economic')
    # Applying the binary operator '==' (line 391)
    result_eq_19240 = python_operator(stypy.reporting.localization.Localization(__file__, 391, 9), '==', mode_19238, str_19239)
    
    # Testing the type of an if condition (line 391)
    if_condition_19241 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 391, 9), result_eq_19240)
    # Assigning a type to the variable 'if_condition_19241' (line 391)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 391, 9), 'if_condition_19241', if_condition_19241)
    # SSA begins for if statement (line 391)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Tuple (line 392):
    
    # Assigning a Subscript to a Name (line 392):
    
    # Obtaining the type of the subscript
    int_19242 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 392, 8), 'int')
    
    # Call to safecall(...): (line 392)
    # Processing the call arguments (line 392)
    # Getting the type of 'gor_un_grq' (line 392)
    gor_un_grq_19244 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 392, 22), 'gor_un_grq', False)
    str_19245 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 392, 34), 'str', 'gorgrq/gungrq')
    # Getting the type of 'rq' (line 392)
    rq_19246 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 392, 51), 'rq', False)
    # Getting the type of 'tau' (line 392)
    tau_19247 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 392, 55), 'tau', False)
    # Processing the call keyword arguments (line 392)
    # Getting the type of 'lwork' (line 392)
    lwork_19248 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 392, 66), 'lwork', False)
    keyword_19249 = lwork_19248
    int_19250 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 393, 34), 'int')
    keyword_19251 = int_19250
    kwargs_19252 = {'overwrite_a': keyword_19251, 'lwork': keyword_19249}
    # Getting the type of 'safecall' (line 392)
    safecall_19243 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 392, 13), 'safecall', False)
    # Calling safecall(args, kwargs) (line 392)
    safecall_call_result_19253 = invoke(stypy.reporting.localization.Localization(__file__, 392, 13), safecall_19243, *[gor_un_grq_19244, str_19245, rq_19246, tau_19247], **kwargs_19252)
    
    # Obtaining the member '__getitem__' of a type (line 392)
    getitem___19254 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 392, 8), safecall_call_result_19253, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 392)
    subscript_call_result_19255 = invoke(stypy.reporting.localization.Localization(__file__, 392, 8), getitem___19254, int_19242)
    
    # Assigning a type to the variable 'tuple_var_assignment_18400' (line 392)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 392, 8), 'tuple_var_assignment_18400', subscript_call_result_19255)
    
    # Assigning a Name to a Name (line 392):
    # Getting the type of 'tuple_var_assignment_18400' (line 392)
    tuple_var_assignment_18400_19256 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 392, 8), 'tuple_var_assignment_18400')
    # Assigning a type to the variable 'Q' (line 392)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 392, 8), 'Q', tuple_var_assignment_18400_19256)
    # SSA branch for the else part of an if statement (line 391)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Call to a Name (line 395):
    
    # Assigning a Call to a Name (line 395):
    
    # Call to empty(...): (line 395)
    # Processing the call arguments (line 395)
    
    # Obtaining an instance of the builtin type 'tuple' (line 395)
    tuple_19259 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 395, 27), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 395)
    # Adding element type (line 395)
    # Getting the type of 'N' (line 395)
    N_19260 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 395, 27), 'N', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 395, 27), tuple_19259, N_19260)
    # Adding element type (line 395)
    # Getting the type of 'N' (line 395)
    N_19261 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 395, 30), 'N', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 395, 27), tuple_19259, N_19261)
    
    # Processing the call keyword arguments (line 395)
    # Getting the type of 'rq' (line 395)
    rq_19262 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 395, 40), 'rq', False)
    # Obtaining the member 'dtype' of a type (line 395)
    dtype_19263 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 395, 40), rq_19262, 'dtype')
    keyword_19264 = dtype_19263
    kwargs_19265 = {'dtype': keyword_19264}
    # Getting the type of 'numpy' (line 395)
    numpy_19257 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 395, 14), 'numpy', False)
    # Obtaining the member 'empty' of a type (line 395)
    empty_19258 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 395, 14), numpy_19257, 'empty')
    # Calling empty(args, kwargs) (line 395)
    empty_call_result_19266 = invoke(stypy.reporting.localization.Localization(__file__, 395, 14), empty_19258, *[tuple_19259], **kwargs_19265)
    
    # Assigning a type to the variable 'rq1' (line 395)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 395, 8), 'rq1', empty_call_result_19266)
    
    # Assigning a Name to a Subscript (line 396):
    
    # Assigning a Name to a Subscript (line 396):
    # Getting the type of 'rq' (line 396)
    rq_19267 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 396, 19), 'rq')
    # Getting the type of 'rq1' (line 396)
    rq1_19268 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 396, 8), 'rq1')
    
    # Getting the type of 'M' (line 396)
    M_19269 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 396, 13), 'M')
    # Applying the 'usub' unary operator (line 396)
    result___neg___19270 = python_operator(stypy.reporting.localization.Localization(__file__, 396, 12), 'usub', M_19269)
    
    slice_19271 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 396, 8), result___neg___19270, None, None)
    # Storing an element on a container (line 396)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 396, 8), rq1_19268, (slice_19271, rq_19267))
    
    # Assigning a Call to a Tuple (line 397):
    
    # Assigning a Subscript to a Name (line 397):
    
    # Obtaining the type of the subscript
    int_19272 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 397, 8), 'int')
    
    # Call to safecall(...): (line 397)
    # Processing the call arguments (line 397)
    # Getting the type of 'gor_un_grq' (line 397)
    gor_un_grq_19274 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 397, 22), 'gor_un_grq', False)
    str_19275 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 397, 34), 'str', 'gorgrq/gungrq')
    # Getting the type of 'rq1' (line 397)
    rq1_19276 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 397, 51), 'rq1', False)
    # Getting the type of 'tau' (line 397)
    tau_19277 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 397, 56), 'tau', False)
    # Processing the call keyword arguments (line 397)
    # Getting the type of 'lwork' (line 397)
    lwork_19278 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 397, 67), 'lwork', False)
    keyword_19279 = lwork_19278
    int_19280 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 398, 34), 'int')
    keyword_19281 = int_19280
    kwargs_19282 = {'overwrite_a': keyword_19281, 'lwork': keyword_19279}
    # Getting the type of 'safecall' (line 397)
    safecall_19273 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 397, 13), 'safecall', False)
    # Calling safecall(args, kwargs) (line 397)
    safecall_call_result_19283 = invoke(stypy.reporting.localization.Localization(__file__, 397, 13), safecall_19273, *[gor_un_grq_19274, str_19275, rq1_19276, tau_19277], **kwargs_19282)
    
    # Obtaining the member '__getitem__' of a type (line 397)
    getitem___19284 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 397, 8), safecall_call_result_19283, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 397)
    subscript_call_result_19285 = invoke(stypy.reporting.localization.Localization(__file__, 397, 8), getitem___19284, int_19272)
    
    # Assigning a type to the variable 'tuple_var_assignment_18401' (line 397)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 397, 8), 'tuple_var_assignment_18401', subscript_call_result_19285)
    
    # Assigning a Name to a Name (line 397):
    # Getting the type of 'tuple_var_assignment_18401' (line 397)
    tuple_var_assignment_18401_19286 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 397, 8), 'tuple_var_assignment_18401')
    # Assigning a type to the variable 'Q' (line 397)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 397, 8), 'Q', tuple_var_assignment_18401_19286)
    # SSA join for if statement (line 391)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 388)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Obtaining an instance of the builtin type 'tuple' (line 400)
    tuple_19287 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 400, 11), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 400)
    # Adding element type (line 400)
    # Getting the type of 'R' (line 400)
    R_19288 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 400, 11), 'R')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 400, 11), tuple_19287, R_19288)
    # Adding element type (line 400)
    # Getting the type of 'Q' (line 400)
    Q_19289 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 400, 14), 'Q')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 400, 11), tuple_19287, Q_19289)
    
    # Assigning a type to the variable 'stypy_return_type' (line 400)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 400, 4), 'stypy_return_type', tuple_19287)
    
    # ################# End of 'rq(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'rq' in the type store
    # Getting the type of 'stypy_return_type' (line 298)
    stypy_return_type_19290 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 298, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_19290)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'rq'
    return stypy_return_type_19290

# Assigning a type to the variable 'rq' (line 298)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 298, 0), 'rq', rq)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
