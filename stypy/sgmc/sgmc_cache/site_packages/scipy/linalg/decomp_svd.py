
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: '''SVD decomposition functions.'''
2: from __future__ import division, print_function, absolute_import
3: 
4: import numpy
5: from numpy import zeros, r_, diag, dot, arccos, arcsin, where, clip
6: 
7: # Local imports.
8: from .misc import LinAlgError, _datacopied
9: from .lapack import get_lapack_funcs, _compute_lwork
10: from .decomp import _asarray_validated
11: from scipy._lib.six import string_types
12: 
13: __all__ = ['svd', 'svdvals', 'diagsvd', 'orth', 'subspace_angles']
14: 
15: 
16: def svd(a, full_matrices=True, compute_uv=True, overwrite_a=False,
17:         check_finite=True, lapack_driver='gesdd'):
18:     '''
19:     Singular Value Decomposition.
20: 
21:     Factorizes the matrix `a` into two unitary matrices ``U`` and ``Vh``, and
22:     a 1-D array ``s`` of singular values (real, non-negative) such that
23:     ``a == U @ S @ Vh``, where ``S`` is a suitably shaped matrix of zeros with
24:     main diagonal ``s``.
25: 
26:     Parameters
27:     ----------
28:     a : (M, N) array_like
29:         Matrix to decompose.
30:     full_matrices : bool, optional
31:         If True (default), `U` and `Vh` are of shape ``(M, M)``, ``(N, N)``.
32:         If False, the shapes are ``(M, K)`` and ``(K, N)``, where
33:         ``K = min(M, N)``.
34:     compute_uv : bool, optional
35:         Whether to compute also ``U`` and ``Vh`` in addition to ``s``.
36:         Default is True.
37:     overwrite_a : bool, optional
38:         Whether to overwrite `a`; may improve performance.
39:         Default is False.
40:     check_finite : bool, optional
41:         Whether to check that the input matrix contains only finite numbers.
42:         Disabling may give a performance gain, but may result in problems
43:         (crashes, non-termination) if the inputs do contain infinities or NaNs.
44:     lapack_driver : {'gesdd', 'gesvd'}, optional
45:         Whether to use the more efficient divide-and-conquer approach
46:         (``'gesdd'``) or general rectangular approach (``'gesvd'``)
47:         to compute the SVD. MATLAB and Octave use the ``'gesvd'`` approach.
48:         Default is ``'gesdd'``.
49: 
50:         .. versionadded:: 0.18
51: 
52:     Returns
53:     -------
54:     U : ndarray
55:         Unitary matrix having left singular vectors as columns.
56:         Of shape ``(M, M)`` or ``(M, K)``, depending on `full_matrices`.
57:     s : ndarray
58:         The singular values, sorted in non-increasing order.
59:         Of shape (K,), with ``K = min(M, N)``.
60:     Vh : ndarray
61:         Unitary matrix having right singular vectors as rows.
62:         Of shape ``(N, N)`` or ``(K, N)`` depending on `full_matrices`.
63: 
64:     For ``compute_uv=False``, only ``s`` is returned.
65: 
66:     Raises
67:     ------
68:     LinAlgError
69:         If SVD computation does not converge.
70: 
71:     See also
72:     --------
73:     svdvals : Compute singular values of a matrix.
74:     diagsvd : Construct the Sigma matrix, given the vector s.
75: 
76:     Examples
77:     --------
78:     >>> from scipy import linalg
79:     >>> m, n = 9, 6
80:     >>> a = np.random.randn(m, n) + 1.j*np.random.randn(m, n)
81:     >>> U, s, Vh = linalg.svd(a)
82:     >>> U.shape,  s.shape, Vh.shape
83:     ((9, 9), (6,), (6, 6))
84: 
85:     Reconstruct the original matrix from the decomposition:
86: 
87:     >>> sigma = np.zeros((m, n))
88:     >>> for i in range(min(m, n)):
89:     ...     sigma[i, i] = s[i]
90:     >>> a1 = np.dot(U, np.dot(sigma, Vh))
91:     >>> np.allclose(a, a1)
92:     True
93: 
94:     Alternatively, use ``full_matrices=False`` (notice that the shape of
95:     ``U`` is then ``(m, n)`` instead of ``(m, m)``):
96: 
97:     >>> U, s, Vh = linalg.svd(a, full_matrices=False)
98:     >>> U.shape, s.shape, Vh.shape
99:     ((9, 6), (6,), (6, 6))
100:     >>> S = np.diag(s)
101:     >>> np.allclose(a, np.dot(U, np.dot(S, Vh)))
102:     True
103: 
104:     >>> s2 = linalg.svd(a, compute_uv=False)
105:     >>> np.allclose(s, s2)
106:     True
107: 
108:     '''
109:     a1 = _asarray_validated(a, check_finite=check_finite)
110:     if len(a1.shape) != 2:
111:         raise ValueError('expected matrix')
112:     m, n = a1.shape
113:     overwrite_a = overwrite_a or (_datacopied(a1, a))
114: 
115:     if not isinstance(lapack_driver, string_types):
116:         raise TypeError('lapack_driver must be a string')
117:     if lapack_driver not in ('gesdd', 'gesvd'):
118:         raise ValueError('lapack_driver must be "gesdd" or "gesvd", not "%s"'
119:                          % (lapack_driver,))
120:     funcs = (lapack_driver, lapack_driver + '_lwork')
121:     gesXd, gesXd_lwork = get_lapack_funcs(funcs, (a1,))
122: 
123:     # compute optimal lwork
124:     lwork = _compute_lwork(gesXd_lwork, a1.shape[0], a1.shape[1],
125:                            compute_uv=compute_uv, full_matrices=full_matrices)
126: 
127:     # perform decomposition
128:     u, s, v, info = gesXd(a1, compute_uv=compute_uv, lwork=lwork,
129:                           full_matrices=full_matrices, overwrite_a=overwrite_a)
130: 
131:     if info > 0:
132:         raise LinAlgError("SVD did not converge")
133:     if info < 0:
134:         raise ValueError('illegal value in %d-th argument of internal gesdd'
135:                          % -info)
136:     if compute_uv:
137:         return u, s, v
138:     else:
139:         return s
140: 
141: 
142: def svdvals(a, overwrite_a=False, check_finite=True):
143:     '''
144:     Compute singular values of a matrix.
145: 
146:     Parameters
147:     ----------
148:     a : (M, N) array_like
149:         Matrix to decompose.
150:     overwrite_a : bool, optional
151:         Whether to overwrite `a`; may improve performance.
152:         Default is False.
153:     check_finite : bool, optional
154:         Whether to check that the input matrix contains only finite numbers.
155:         Disabling may give a performance gain, but may result in problems
156:         (crashes, non-termination) if the inputs do contain infinities or NaNs.
157: 
158:     Returns
159:     -------
160:     s : (min(M, N),) ndarray
161:         The singular values, sorted in decreasing order.
162: 
163:     Raises
164:     ------
165:     LinAlgError
166:         If SVD computation does not converge.
167: 
168:     Notes
169:     -----
170:     ``svdvals(a)`` only differs from ``svd(a, compute_uv=False)`` by its
171:     handling of the edge case of empty ``a``, where it returns an
172:     empty sequence:
173: 
174:     >>> a = np.empty((0, 2))
175:     >>> from scipy.linalg import svdvals
176:     >>> svdvals(a)
177:     array([], dtype=float64)
178: 
179:     See Also
180:     --------
181:     svd : Compute the full singular value decomposition of a matrix.
182:     diagsvd : Construct the Sigma matrix, given the vector s.
183: 
184:     Examples
185:     --------
186:     >>> from scipy.linalg import svdvals
187:     >>> m = np.array([[1.0, 0.0],
188:     ...               [2.0, 3.0],
189:     ...               [1.0, 1.0],
190:     ...               [0.0, 2.0],
191:     ...               [1.0, 0.0]])
192:     >>> svdvals(m)
193:     array([ 4.28091555,  1.63516424])
194: 
195:     We can verify the maximum singular value of `m` by computing the maximum
196:     length of `m.dot(u)` over all the unit vectors `u` in the (x,y) plane.
197:     We approximate "all" the unit vectors with a large sample.  Because
198:     of linearity, we only need the unit vectors with angles in [0, pi].
199: 
200:     >>> t = np.linspace(0, np.pi, 2000)
201:     >>> u = np.array([np.cos(t), np.sin(t)])
202:     >>> np.linalg.norm(m.dot(u), axis=0).max()
203:     4.2809152422538475
204: 
205:     `p` is a projection matrix with rank 1.  With exact arithmetic,
206:     its singular values would be [1, 0, 0, 0].
207: 
208:     >>> v = np.array([0.1, 0.3, 0.9, 0.3])
209:     >>> p = np.outer(v, v)
210:     >>> svdvals(p)
211:     array([  1.00000000e+00,   2.02021698e-17,   1.56692500e-17,
212:              8.15115104e-34])
213: 
214:     The singular values of an orthogonal matrix are all 1.  Here we
215:     create a random orthogonal matrix by using the `rvs()` method of
216:     `scipy.stats.ortho_group`.
217: 
218:     >>> from scipy.stats import ortho_group
219:     >>> np.random.seed(123)
220:     >>> orth = ortho_group.rvs(4)
221:     >>> svdvals(orth)
222:     array([ 1.,  1.,  1.,  1.])
223: 
224:     '''
225:     a = _asarray_validated(a, check_finite=check_finite)
226:     if a.size:
227:         return svd(a, compute_uv=0, overwrite_a=overwrite_a,
228:                    check_finite=False)
229:     elif len(a.shape) != 2:
230:         raise ValueError('expected matrix')
231:     else:
232:         return numpy.empty(0)
233: 
234: 
235: def diagsvd(s, M, N):
236:     '''
237:     Construct the sigma matrix in SVD from singular values and size M, N.
238: 
239:     Parameters
240:     ----------
241:     s : (M,) or (N,) array_like
242:         Singular values
243:     M : int
244:         Size of the matrix whose singular values are `s`.
245:     N : int
246:         Size of the matrix whose singular values are `s`.
247: 
248:     Returns
249:     -------
250:     S : (M, N) ndarray
251:         The S-matrix in the singular value decomposition
252: 
253:     '''
254:     part = diag(s)
255:     typ = part.dtype.char
256:     MorN = len(s)
257:     if MorN == M:
258:         return r_['-1', part, zeros((M, N-M), typ)]
259:     elif MorN == N:
260:         return r_[part, zeros((M-N, N), typ)]
261:     else:
262:         raise ValueError("Length of s must be M or N.")
263: 
264: 
265: # Orthonormal decomposition
266: 
267: def orth(A):
268:     '''
269:     Construct an orthonormal basis for the range of A using SVD
270: 
271:     Parameters
272:     ----------
273:     A : (M, N) array_like
274:         Input array
275: 
276:     Returns
277:     -------
278:     Q : (M, K) ndarray
279:         Orthonormal basis for the range of A.
280:         K = effective rank of A, as determined by automatic cutoff
281: 
282:     See also
283:     --------
284:     svd : Singular value decomposition of a matrix
285: 
286:     '''
287:     u, s, vh = svd(A, full_matrices=False)
288:     M, N = A.shape
289:     eps = numpy.finfo(float).eps
290:     tol = max(M, N) * numpy.amax(s) * eps
291:     num = numpy.sum(s > tol, dtype=int)
292:     Q = u[:, :num]
293:     return Q
294: 
295: 
296: def subspace_angles(A, B):
297:     r'''
298:     Compute the subspace angles between two matrices.
299: 
300:     Parameters
301:     ----------
302:     A : (M, N) array_like
303:         The first input array.
304:     B : (M, K) array_like
305:         The second input array.
306: 
307:     Returns
308:     -------
309:     angles : ndarray, shape (min(N, K),)
310:         The subspace angles between the column spaces of `A` and `B`.
311: 
312:     See Also
313:     --------
314:     orth
315:     svd
316: 
317:     Notes
318:     -----
319:     This computes the subspace angles according to the formula
320:     provided in [1]_. For equivalence with MATLAB and Octave behavior,
321:     use ``angles[0]``.
322: 
323:     .. versionadded:: 1.0
324: 
325:     References
326:     ----------
327:     .. [1] Knyazev A, Argentati M (2002) Principal Angles between Subspaces
328:            in an A-Based Scalar Product: Algorithms and Perturbation
329:            Estimates. SIAM J. Sci. Comput. 23:2008-2040.
330: 
331:     Examples
332:     --------
333:     A Hadamard matrix, which has orthogonal columns, so we expect that
334:     the suspace angle to be :math:`\frac{\pi}{2}`:
335: 
336:     >>> from scipy.linalg import hadamard, subspace_angles
337:     >>> H = hadamard(4)
338:     >>> print(H)
339:     [[ 1  1  1  1]
340:      [ 1 -1  1 -1]
341:      [ 1  1 -1 -1]
342:      [ 1 -1 -1  1]]
343:     >>> np.rad2deg(subspace_angles(H[:, :2], H[:, 2:]))
344:     array([ 90.,  90.])
345: 
346:     And the subspace angle of a matrix to itself should be zero:
347: 
348:     >>> subspace_angles(H[:, :2], H[:, :2]) <= 2 * np.finfo(float).eps
349:     array([ True,  True], dtype=bool)
350: 
351:     The angles between non-orthogonal subspaces are in between these extremes:
352: 
353:     >>> x = np.random.RandomState(0).randn(4, 3)
354:     >>> np.rad2deg(subspace_angles(x[:, :2], x[:, [2]]))
355:     array([ 55.832])
356:     '''
357:     # Steps here omit the U and V calculation steps from the paper
358: 
359:     # 1. Compute orthonormal bases of column-spaces
360:     A = _asarray_validated(A, check_finite=True)
361:     if len(A.shape) != 2:
362:         raise ValueError('expected 2D array, got shape %s' % (A.shape,))
363:     QA = orth(A)
364:     del A
365: 
366:     B = _asarray_validated(B, check_finite=True)
367:     if len(B.shape) != 2:
368:         raise ValueError('expected 2D array, got shape %s' % (B.shape,))
369:     if len(B) != len(QA):
370:         raise ValueError('A and B must have the same number of rows, got '
371:                          '%s and %s' % (QA.shape[0], B.shape[0]))
372:     QB = orth(B)
373:     del B
374: 
375:     # 2. Compute SVD for cosine
376:     QA_T_QB = dot(QA.T, QB)
377:     sigma = svdvals(QA_T_QB)
378: 
379:     # 3. Compute matrix B
380:     if QA.shape[1] >= QB.shape[1]:
381:         B = QB - dot(QA, QA_T_QB)
382:     else:
383:         B = QA - dot(QB, QA_T_QB.T)
384:     del QA, QB, QA_T_QB
385: 
386:     # 4. Compute SVD for sine
387:     mask = sigma ** 2 >= 0.5
388:     if mask.any():
389:         mu_arcsin = arcsin(clip(svdvals(B, overwrite_a=True), -1., 1.))
390:     else:
391:         mu_arcsin = 0.
392: 
393:     # 5. Compute the principal angles
394:     theta = where(mask, mu_arcsin, arccos(clip(sigma, -1., 1.)))
395:     return theta
396: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_20111 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1, 0), 'str', 'SVD decomposition functions.')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 4, 0))

# 'import numpy' statement (line 4)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/linalg/')
import_20112 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'numpy')

if (type(import_20112) is not StypyTypeError):

    if (import_20112 != 'pyd_module'):
        __import__(import_20112)
        sys_modules_20113 = sys.modules[import_20112]
        import_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'numpy', sys_modules_20113.module_type_store, module_type_store)
    else:
        import numpy

        import_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'numpy', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 4)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 4, 0), 'numpy', import_20112)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/linalg/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 5, 0))

# 'from numpy import zeros, r_, diag, dot, arccos, arcsin, where, clip' statement (line 5)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/linalg/')
import_20114 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'numpy')

if (type(import_20114) is not StypyTypeError):

    if (import_20114 != 'pyd_module'):
        __import__(import_20114)
        sys_modules_20115 = sys.modules[import_20114]
        import_from_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'numpy', sys_modules_20115.module_type_store, module_type_store, ['zeros', 'r_', 'diag', 'dot', 'arccos', 'arcsin', 'where', 'clip'])
        nest_module(stypy.reporting.localization.Localization(__file__, 5, 0), __file__, sys_modules_20115, sys_modules_20115.module_type_store, module_type_store)
    else:
        from numpy import zeros, r_, diag, dot, arccos, arcsin, where, clip

        import_from_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'numpy', None, module_type_store, ['zeros', 'r_', 'diag', 'dot', 'arccos', 'arcsin', 'where', 'clip'], [zeros, r_, diag, dot, arccos, arcsin, where, clip])

else:
    # Assigning a type to the variable 'numpy' (line 5)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 0), 'numpy', import_20114)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/linalg/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 8, 0))

# 'from scipy.linalg.misc import LinAlgError, _datacopied' statement (line 8)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/linalg/')
import_20116 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'scipy.linalg.misc')

if (type(import_20116) is not StypyTypeError):

    if (import_20116 != 'pyd_module'):
        __import__(import_20116)
        sys_modules_20117 = sys.modules[import_20116]
        import_from_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'scipy.linalg.misc', sys_modules_20117.module_type_store, module_type_store, ['LinAlgError', '_datacopied'])
        nest_module(stypy.reporting.localization.Localization(__file__, 8, 0), __file__, sys_modules_20117, sys_modules_20117.module_type_store, module_type_store)
    else:
        from scipy.linalg.misc import LinAlgError, _datacopied

        import_from_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'scipy.linalg.misc', None, module_type_store, ['LinAlgError', '_datacopied'], [LinAlgError, _datacopied])

else:
    # Assigning a type to the variable 'scipy.linalg.misc' (line 8)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 0), 'scipy.linalg.misc', import_20116)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/linalg/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 9, 0))

# 'from scipy.linalg.lapack import get_lapack_funcs, _compute_lwork' statement (line 9)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/linalg/')
import_20118 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'scipy.linalg.lapack')

if (type(import_20118) is not StypyTypeError):

    if (import_20118 != 'pyd_module'):
        __import__(import_20118)
        sys_modules_20119 = sys.modules[import_20118]
        import_from_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'scipy.linalg.lapack', sys_modules_20119.module_type_store, module_type_store, ['get_lapack_funcs', '_compute_lwork'])
        nest_module(stypy.reporting.localization.Localization(__file__, 9, 0), __file__, sys_modules_20119, sys_modules_20119.module_type_store, module_type_store)
    else:
        from scipy.linalg.lapack import get_lapack_funcs, _compute_lwork

        import_from_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'scipy.linalg.lapack', None, module_type_store, ['get_lapack_funcs', '_compute_lwork'], [get_lapack_funcs, _compute_lwork])

else:
    # Assigning a type to the variable 'scipy.linalg.lapack' (line 9)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 0), 'scipy.linalg.lapack', import_20118)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/linalg/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 10, 0))

# 'from scipy.linalg.decomp import _asarray_validated' statement (line 10)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/linalg/')
import_20120 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'scipy.linalg.decomp')

if (type(import_20120) is not StypyTypeError):

    if (import_20120 != 'pyd_module'):
        __import__(import_20120)
        sys_modules_20121 = sys.modules[import_20120]
        import_from_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'scipy.linalg.decomp', sys_modules_20121.module_type_store, module_type_store, ['_asarray_validated'])
        nest_module(stypy.reporting.localization.Localization(__file__, 10, 0), __file__, sys_modules_20121, sys_modules_20121.module_type_store, module_type_store)
    else:
        from scipy.linalg.decomp import _asarray_validated

        import_from_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'scipy.linalg.decomp', None, module_type_store, ['_asarray_validated'], [_asarray_validated])

else:
    # Assigning a type to the variable 'scipy.linalg.decomp' (line 10)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 0), 'scipy.linalg.decomp', import_20120)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/linalg/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 11, 0))

# 'from scipy._lib.six import string_types' statement (line 11)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/linalg/')
import_20122 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'scipy._lib.six')

if (type(import_20122) is not StypyTypeError):

    if (import_20122 != 'pyd_module'):
        __import__(import_20122)
        sys_modules_20123 = sys.modules[import_20122]
        import_from_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'scipy._lib.six', sys_modules_20123.module_type_store, module_type_store, ['string_types'])
        nest_module(stypy.reporting.localization.Localization(__file__, 11, 0), __file__, sys_modules_20123, sys_modules_20123.module_type_store, module_type_store)
    else:
        from scipy._lib.six import string_types

        import_from_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'scipy._lib.six', None, module_type_store, ['string_types'], [string_types])

else:
    # Assigning a type to the variable 'scipy._lib.six' (line 11)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 0), 'scipy._lib.six', import_20122)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/linalg/')


# Assigning a List to a Name (line 13):

# Assigning a List to a Name (line 13):
__all__ = ['svd', 'svdvals', 'diagsvd', 'orth', 'subspace_angles']
module_type_store.set_exportable_members(['svd', 'svdvals', 'diagsvd', 'orth', 'subspace_angles'])

# Obtaining an instance of the builtin type 'list' (line 13)
list_20124 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 10), 'list')
# Adding type elements to the builtin type 'list' instance (line 13)
# Adding element type (line 13)
str_20125 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 11), 'str', 'svd')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 13, 10), list_20124, str_20125)
# Adding element type (line 13)
str_20126 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 18), 'str', 'svdvals')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 13, 10), list_20124, str_20126)
# Adding element type (line 13)
str_20127 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 29), 'str', 'diagsvd')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 13, 10), list_20124, str_20127)
# Adding element type (line 13)
str_20128 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 40), 'str', 'orth')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 13, 10), list_20124, str_20128)
# Adding element type (line 13)
str_20129 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 48), 'str', 'subspace_angles')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 13, 10), list_20124, str_20129)

# Assigning a type to the variable '__all__' (line 13)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 0), '__all__', list_20124)

@norecursion
def svd(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'True' (line 16)
    True_20130 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 25), 'True')
    # Getting the type of 'True' (line 16)
    True_20131 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 42), 'True')
    # Getting the type of 'False' (line 16)
    False_20132 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 60), 'False')
    # Getting the type of 'True' (line 17)
    True_20133 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 21), 'True')
    str_20134 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 41), 'str', 'gesdd')
    defaults = [True_20130, True_20131, False_20132, True_20133, str_20134]
    # Create a new context for function 'svd'
    module_type_store = module_type_store.open_function_context('svd', 16, 0, False)
    
    # Passed parameters checking function
    svd.stypy_localization = localization
    svd.stypy_type_of_self = None
    svd.stypy_type_store = module_type_store
    svd.stypy_function_name = 'svd'
    svd.stypy_param_names_list = ['a', 'full_matrices', 'compute_uv', 'overwrite_a', 'check_finite', 'lapack_driver']
    svd.stypy_varargs_param_name = None
    svd.stypy_kwargs_param_name = None
    svd.stypy_call_defaults = defaults
    svd.stypy_call_varargs = varargs
    svd.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'svd', ['a', 'full_matrices', 'compute_uv', 'overwrite_a', 'check_finite', 'lapack_driver'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'svd', localization, ['a', 'full_matrices', 'compute_uv', 'overwrite_a', 'check_finite', 'lapack_driver'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'svd(...)' code ##################

    str_20135 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 108, (-1)), 'str', "\n    Singular Value Decomposition.\n\n    Factorizes the matrix `a` into two unitary matrices ``U`` and ``Vh``, and\n    a 1-D array ``s`` of singular values (real, non-negative) such that\n    ``a == U @ S @ Vh``, where ``S`` is a suitably shaped matrix of zeros with\n    main diagonal ``s``.\n\n    Parameters\n    ----------\n    a : (M, N) array_like\n        Matrix to decompose.\n    full_matrices : bool, optional\n        If True (default), `U` and `Vh` are of shape ``(M, M)``, ``(N, N)``.\n        If False, the shapes are ``(M, K)`` and ``(K, N)``, where\n        ``K = min(M, N)``.\n    compute_uv : bool, optional\n        Whether to compute also ``U`` and ``Vh`` in addition to ``s``.\n        Default is True.\n    overwrite_a : bool, optional\n        Whether to overwrite `a`; may improve performance.\n        Default is False.\n    check_finite : bool, optional\n        Whether to check that the input matrix contains only finite numbers.\n        Disabling may give a performance gain, but may result in problems\n        (crashes, non-termination) if the inputs do contain infinities or NaNs.\n    lapack_driver : {'gesdd', 'gesvd'}, optional\n        Whether to use the more efficient divide-and-conquer approach\n        (``'gesdd'``) or general rectangular approach (``'gesvd'``)\n        to compute the SVD. MATLAB and Octave use the ``'gesvd'`` approach.\n        Default is ``'gesdd'``.\n\n        .. versionadded:: 0.18\n\n    Returns\n    -------\n    U : ndarray\n        Unitary matrix having left singular vectors as columns.\n        Of shape ``(M, M)`` or ``(M, K)``, depending on `full_matrices`.\n    s : ndarray\n        The singular values, sorted in non-increasing order.\n        Of shape (K,), with ``K = min(M, N)``.\n    Vh : ndarray\n        Unitary matrix having right singular vectors as rows.\n        Of shape ``(N, N)`` or ``(K, N)`` depending on `full_matrices`.\n\n    For ``compute_uv=False``, only ``s`` is returned.\n\n    Raises\n    ------\n    LinAlgError\n        If SVD computation does not converge.\n\n    See also\n    --------\n    svdvals : Compute singular values of a matrix.\n    diagsvd : Construct the Sigma matrix, given the vector s.\n\n    Examples\n    --------\n    >>> from scipy import linalg\n    >>> m, n = 9, 6\n    >>> a = np.random.randn(m, n) + 1.j*np.random.randn(m, n)\n    >>> U, s, Vh = linalg.svd(a)\n    >>> U.shape,  s.shape, Vh.shape\n    ((9, 9), (6,), (6, 6))\n\n    Reconstruct the original matrix from the decomposition:\n\n    >>> sigma = np.zeros((m, n))\n    >>> for i in range(min(m, n)):\n    ...     sigma[i, i] = s[i]\n    >>> a1 = np.dot(U, np.dot(sigma, Vh))\n    >>> np.allclose(a, a1)\n    True\n\n    Alternatively, use ``full_matrices=False`` (notice that the shape of\n    ``U`` is then ``(m, n)`` instead of ``(m, m)``):\n\n    >>> U, s, Vh = linalg.svd(a, full_matrices=False)\n    >>> U.shape, s.shape, Vh.shape\n    ((9, 6), (6,), (6, 6))\n    >>> S = np.diag(s)\n    >>> np.allclose(a, np.dot(U, np.dot(S, Vh)))\n    True\n\n    >>> s2 = linalg.svd(a, compute_uv=False)\n    >>> np.allclose(s, s2)\n    True\n\n    ")
    
    # Assigning a Call to a Name (line 109):
    
    # Assigning a Call to a Name (line 109):
    
    # Call to _asarray_validated(...): (line 109)
    # Processing the call arguments (line 109)
    # Getting the type of 'a' (line 109)
    a_20137 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 28), 'a', False)
    # Processing the call keyword arguments (line 109)
    # Getting the type of 'check_finite' (line 109)
    check_finite_20138 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 44), 'check_finite', False)
    keyword_20139 = check_finite_20138
    kwargs_20140 = {'check_finite': keyword_20139}
    # Getting the type of '_asarray_validated' (line 109)
    _asarray_validated_20136 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 9), '_asarray_validated', False)
    # Calling _asarray_validated(args, kwargs) (line 109)
    _asarray_validated_call_result_20141 = invoke(stypy.reporting.localization.Localization(__file__, 109, 9), _asarray_validated_20136, *[a_20137], **kwargs_20140)
    
    # Assigning a type to the variable 'a1' (line 109)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 109, 4), 'a1', _asarray_validated_call_result_20141)
    
    
    
    # Call to len(...): (line 110)
    # Processing the call arguments (line 110)
    # Getting the type of 'a1' (line 110)
    a1_20143 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 11), 'a1', False)
    # Obtaining the member 'shape' of a type (line 110)
    shape_20144 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 110, 11), a1_20143, 'shape')
    # Processing the call keyword arguments (line 110)
    kwargs_20145 = {}
    # Getting the type of 'len' (line 110)
    len_20142 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 7), 'len', False)
    # Calling len(args, kwargs) (line 110)
    len_call_result_20146 = invoke(stypy.reporting.localization.Localization(__file__, 110, 7), len_20142, *[shape_20144], **kwargs_20145)
    
    int_20147 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 110, 24), 'int')
    # Applying the binary operator '!=' (line 110)
    result_ne_20148 = python_operator(stypy.reporting.localization.Localization(__file__, 110, 7), '!=', len_call_result_20146, int_20147)
    
    # Testing the type of an if condition (line 110)
    if_condition_20149 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 110, 4), result_ne_20148)
    # Assigning a type to the variable 'if_condition_20149' (line 110)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 110, 4), 'if_condition_20149', if_condition_20149)
    # SSA begins for if statement (line 110)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 111)
    # Processing the call arguments (line 111)
    str_20151 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 111, 25), 'str', 'expected matrix')
    # Processing the call keyword arguments (line 111)
    kwargs_20152 = {}
    # Getting the type of 'ValueError' (line 111)
    ValueError_20150 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 111)
    ValueError_call_result_20153 = invoke(stypy.reporting.localization.Localization(__file__, 111, 14), ValueError_20150, *[str_20151], **kwargs_20152)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 111, 8), ValueError_call_result_20153, 'raise parameter', BaseException)
    # SSA join for if statement (line 110)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Attribute to a Tuple (line 112):
    
    # Assigning a Subscript to a Name (line 112):
    
    # Obtaining the type of the subscript
    int_20154 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 112, 4), 'int')
    # Getting the type of 'a1' (line 112)
    a1_20155 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 11), 'a1')
    # Obtaining the member 'shape' of a type (line 112)
    shape_20156 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 112, 11), a1_20155, 'shape')
    # Obtaining the member '__getitem__' of a type (line 112)
    getitem___20157 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 112, 4), shape_20156, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 112)
    subscript_call_result_20158 = invoke(stypy.reporting.localization.Localization(__file__, 112, 4), getitem___20157, int_20154)
    
    # Assigning a type to the variable 'tuple_var_assignment_20098' (line 112)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 112, 4), 'tuple_var_assignment_20098', subscript_call_result_20158)
    
    # Assigning a Subscript to a Name (line 112):
    
    # Obtaining the type of the subscript
    int_20159 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 112, 4), 'int')
    # Getting the type of 'a1' (line 112)
    a1_20160 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 11), 'a1')
    # Obtaining the member 'shape' of a type (line 112)
    shape_20161 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 112, 11), a1_20160, 'shape')
    # Obtaining the member '__getitem__' of a type (line 112)
    getitem___20162 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 112, 4), shape_20161, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 112)
    subscript_call_result_20163 = invoke(stypy.reporting.localization.Localization(__file__, 112, 4), getitem___20162, int_20159)
    
    # Assigning a type to the variable 'tuple_var_assignment_20099' (line 112)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 112, 4), 'tuple_var_assignment_20099', subscript_call_result_20163)
    
    # Assigning a Name to a Name (line 112):
    # Getting the type of 'tuple_var_assignment_20098' (line 112)
    tuple_var_assignment_20098_20164 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 4), 'tuple_var_assignment_20098')
    # Assigning a type to the variable 'm' (line 112)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 112, 4), 'm', tuple_var_assignment_20098_20164)
    
    # Assigning a Name to a Name (line 112):
    # Getting the type of 'tuple_var_assignment_20099' (line 112)
    tuple_var_assignment_20099_20165 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 4), 'tuple_var_assignment_20099')
    # Assigning a type to the variable 'n' (line 112)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 112, 7), 'n', tuple_var_assignment_20099_20165)
    
    # Assigning a BoolOp to a Name (line 113):
    
    # Assigning a BoolOp to a Name (line 113):
    
    # Evaluating a boolean operation
    # Getting the type of 'overwrite_a' (line 113)
    overwrite_a_20166 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 18), 'overwrite_a')
    
    # Call to _datacopied(...): (line 113)
    # Processing the call arguments (line 113)
    # Getting the type of 'a1' (line 113)
    a1_20168 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 46), 'a1', False)
    # Getting the type of 'a' (line 113)
    a_20169 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 50), 'a', False)
    # Processing the call keyword arguments (line 113)
    kwargs_20170 = {}
    # Getting the type of '_datacopied' (line 113)
    _datacopied_20167 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 34), '_datacopied', False)
    # Calling _datacopied(args, kwargs) (line 113)
    _datacopied_call_result_20171 = invoke(stypy.reporting.localization.Localization(__file__, 113, 34), _datacopied_20167, *[a1_20168, a_20169], **kwargs_20170)
    
    # Applying the binary operator 'or' (line 113)
    result_or_keyword_20172 = python_operator(stypy.reporting.localization.Localization(__file__, 113, 18), 'or', overwrite_a_20166, _datacopied_call_result_20171)
    
    # Assigning a type to the variable 'overwrite_a' (line 113)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 113, 4), 'overwrite_a', result_or_keyword_20172)
    
    
    
    # Call to isinstance(...): (line 115)
    # Processing the call arguments (line 115)
    # Getting the type of 'lapack_driver' (line 115)
    lapack_driver_20174 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 22), 'lapack_driver', False)
    # Getting the type of 'string_types' (line 115)
    string_types_20175 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 37), 'string_types', False)
    # Processing the call keyword arguments (line 115)
    kwargs_20176 = {}
    # Getting the type of 'isinstance' (line 115)
    isinstance_20173 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 11), 'isinstance', False)
    # Calling isinstance(args, kwargs) (line 115)
    isinstance_call_result_20177 = invoke(stypy.reporting.localization.Localization(__file__, 115, 11), isinstance_20173, *[lapack_driver_20174, string_types_20175], **kwargs_20176)
    
    # Applying the 'not' unary operator (line 115)
    result_not__20178 = python_operator(stypy.reporting.localization.Localization(__file__, 115, 7), 'not', isinstance_call_result_20177)
    
    # Testing the type of an if condition (line 115)
    if_condition_20179 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 115, 4), result_not__20178)
    # Assigning a type to the variable 'if_condition_20179' (line 115)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 115, 4), 'if_condition_20179', if_condition_20179)
    # SSA begins for if statement (line 115)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to TypeError(...): (line 116)
    # Processing the call arguments (line 116)
    str_20181 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 116, 24), 'str', 'lapack_driver must be a string')
    # Processing the call keyword arguments (line 116)
    kwargs_20182 = {}
    # Getting the type of 'TypeError' (line 116)
    TypeError_20180 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 14), 'TypeError', False)
    # Calling TypeError(args, kwargs) (line 116)
    TypeError_call_result_20183 = invoke(stypy.reporting.localization.Localization(__file__, 116, 14), TypeError_20180, *[str_20181], **kwargs_20182)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 116, 8), TypeError_call_result_20183, 'raise parameter', BaseException)
    # SSA join for if statement (line 115)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'lapack_driver' (line 117)
    lapack_driver_20184 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 7), 'lapack_driver')
    
    # Obtaining an instance of the builtin type 'tuple' (line 117)
    tuple_20185 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 117, 29), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 117)
    # Adding element type (line 117)
    str_20186 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 117, 29), 'str', 'gesdd')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 117, 29), tuple_20185, str_20186)
    # Adding element type (line 117)
    str_20187 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 117, 38), 'str', 'gesvd')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 117, 29), tuple_20185, str_20187)
    
    # Applying the binary operator 'notin' (line 117)
    result_contains_20188 = python_operator(stypy.reporting.localization.Localization(__file__, 117, 7), 'notin', lapack_driver_20184, tuple_20185)
    
    # Testing the type of an if condition (line 117)
    if_condition_20189 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 117, 4), result_contains_20188)
    # Assigning a type to the variable 'if_condition_20189' (line 117)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 117, 4), 'if_condition_20189', if_condition_20189)
    # SSA begins for if statement (line 117)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 118)
    # Processing the call arguments (line 118)
    str_20191 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 118, 25), 'str', 'lapack_driver must be "gesdd" or "gesvd", not "%s"')
    
    # Obtaining an instance of the builtin type 'tuple' (line 119)
    tuple_20192 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 119, 28), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 119)
    # Adding element type (line 119)
    # Getting the type of 'lapack_driver' (line 119)
    lapack_driver_20193 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 28), 'lapack_driver', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 119, 28), tuple_20192, lapack_driver_20193)
    
    # Applying the binary operator '%' (line 118)
    result_mod_20194 = python_operator(stypy.reporting.localization.Localization(__file__, 118, 25), '%', str_20191, tuple_20192)
    
    # Processing the call keyword arguments (line 118)
    kwargs_20195 = {}
    # Getting the type of 'ValueError' (line 118)
    ValueError_20190 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 118)
    ValueError_call_result_20196 = invoke(stypy.reporting.localization.Localization(__file__, 118, 14), ValueError_20190, *[result_mod_20194], **kwargs_20195)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 118, 8), ValueError_call_result_20196, 'raise parameter', BaseException)
    # SSA join for if statement (line 117)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Tuple to a Name (line 120):
    
    # Assigning a Tuple to a Name (line 120):
    
    # Obtaining an instance of the builtin type 'tuple' (line 120)
    tuple_20197 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 120, 13), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 120)
    # Adding element type (line 120)
    # Getting the type of 'lapack_driver' (line 120)
    lapack_driver_20198 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 13), 'lapack_driver')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 120, 13), tuple_20197, lapack_driver_20198)
    # Adding element type (line 120)
    # Getting the type of 'lapack_driver' (line 120)
    lapack_driver_20199 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 28), 'lapack_driver')
    str_20200 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 120, 44), 'str', '_lwork')
    # Applying the binary operator '+' (line 120)
    result_add_20201 = python_operator(stypy.reporting.localization.Localization(__file__, 120, 28), '+', lapack_driver_20199, str_20200)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 120, 13), tuple_20197, result_add_20201)
    
    # Assigning a type to the variable 'funcs' (line 120)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 120, 4), 'funcs', tuple_20197)
    
    # Assigning a Call to a Tuple (line 121):
    
    # Assigning a Subscript to a Name (line 121):
    
    # Obtaining the type of the subscript
    int_20202 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 121, 4), 'int')
    
    # Call to get_lapack_funcs(...): (line 121)
    # Processing the call arguments (line 121)
    # Getting the type of 'funcs' (line 121)
    funcs_20204 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 42), 'funcs', False)
    
    # Obtaining an instance of the builtin type 'tuple' (line 121)
    tuple_20205 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 121, 50), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 121)
    # Adding element type (line 121)
    # Getting the type of 'a1' (line 121)
    a1_20206 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 50), 'a1', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 121, 50), tuple_20205, a1_20206)
    
    # Processing the call keyword arguments (line 121)
    kwargs_20207 = {}
    # Getting the type of 'get_lapack_funcs' (line 121)
    get_lapack_funcs_20203 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 25), 'get_lapack_funcs', False)
    # Calling get_lapack_funcs(args, kwargs) (line 121)
    get_lapack_funcs_call_result_20208 = invoke(stypy.reporting.localization.Localization(__file__, 121, 25), get_lapack_funcs_20203, *[funcs_20204, tuple_20205], **kwargs_20207)
    
    # Obtaining the member '__getitem__' of a type (line 121)
    getitem___20209 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 121, 4), get_lapack_funcs_call_result_20208, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 121)
    subscript_call_result_20210 = invoke(stypy.reporting.localization.Localization(__file__, 121, 4), getitem___20209, int_20202)
    
    # Assigning a type to the variable 'tuple_var_assignment_20100' (line 121)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 121, 4), 'tuple_var_assignment_20100', subscript_call_result_20210)
    
    # Assigning a Subscript to a Name (line 121):
    
    # Obtaining the type of the subscript
    int_20211 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 121, 4), 'int')
    
    # Call to get_lapack_funcs(...): (line 121)
    # Processing the call arguments (line 121)
    # Getting the type of 'funcs' (line 121)
    funcs_20213 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 42), 'funcs', False)
    
    # Obtaining an instance of the builtin type 'tuple' (line 121)
    tuple_20214 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 121, 50), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 121)
    # Adding element type (line 121)
    # Getting the type of 'a1' (line 121)
    a1_20215 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 50), 'a1', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 121, 50), tuple_20214, a1_20215)
    
    # Processing the call keyword arguments (line 121)
    kwargs_20216 = {}
    # Getting the type of 'get_lapack_funcs' (line 121)
    get_lapack_funcs_20212 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 25), 'get_lapack_funcs', False)
    # Calling get_lapack_funcs(args, kwargs) (line 121)
    get_lapack_funcs_call_result_20217 = invoke(stypy.reporting.localization.Localization(__file__, 121, 25), get_lapack_funcs_20212, *[funcs_20213, tuple_20214], **kwargs_20216)
    
    # Obtaining the member '__getitem__' of a type (line 121)
    getitem___20218 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 121, 4), get_lapack_funcs_call_result_20217, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 121)
    subscript_call_result_20219 = invoke(stypy.reporting.localization.Localization(__file__, 121, 4), getitem___20218, int_20211)
    
    # Assigning a type to the variable 'tuple_var_assignment_20101' (line 121)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 121, 4), 'tuple_var_assignment_20101', subscript_call_result_20219)
    
    # Assigning a Name to a Name (line 121):
    # Getting the type of 'tuple_var_assignment_20100' (line 121)
    tuple_var_assignment_20100_20220 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 4), 'tuple_var_assignment_20100')
    # Assigning a type to the variable 'gesXd' (line 121)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 121, 4), 'gesXd', tuple_var_assignment_20100_20220)
    
    # Assigning a Name to a Name (line 121):
    # Getting the type of 'tuple_var_assignment_20101' (line 121)
    tuple_var_assignment_20101_20221 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 4), 'tuple_var_assignment_20101')
    # Assigning a type to the variable 'gesXd_lwork' (line 121)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 121, 11), 'gesXd_lwork', tuple_var_assignment_20101_20221)
    
    # Assigning a Call to a Name (line 124):
    
    # Assigning a Call to a Name (line 124):
    
    # Call to _compute_lwork(...): (line 124)
    # Processing the call arguments (line 124)
    # Getting the type of 'gesXd_lwork' (line 124)
    gesXd_lwork_20223 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 27), 'gesXd_lwork', False)
    
    # Obtaining the type of the subscript
    int_20224 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 124, 49), 'int')
    # Getting the type of 'a1' (line 124)
    a1_20225 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 40), 'a1', False)
    # Obtaining the member 'shape' of a type (line 124)
    shape_20226 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 124, 40), a1_20225, 'shape')
    # Obtaining the member '__getitem__' of a type (line 124)
    getitem___20227 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 124, 40), shape_20226, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 124)
    subscript_call_result_20228 = invoke(stypy.reporting.localization.Localization(__file__, 124, 40), getitem___20227, int_20224)
    
    
    # Obtaining the type of the subscript
    int_20229 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 124, 62), 'int')
    # Getting the type of 'a1' (line 124)
    a1_20230 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 53), 'a1', False)
    # Obtaining the member 'shape' of a type (line 124)
    shape_20231 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 124, 53), a1_20230, 'shape')
    # Obtaining the member '__getitem__' of a type (line 124)
    getitem___20232 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 124, 53), shape_20231, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 124)
    subscript_call_result_20233 = invoke(stypy.reporting.localization.Localization(__file__, 124, 53), getitem___20232, int_20229)
    
    # Processing the call keyword arguments (line 124)
    # Getting the type of 'compute_uv' (line 125)
    compute_uv_20234 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 38), 'compute_uv', False)
    keyword_20235 = compute_uv_20234
    # Getting the type of 'full_matrices' (line 125)
    full_matrices_20236 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 64), 'full_matrices', False)
    keyword_20237 = full_matrices_20236
    kwargs_20238 = {'compute_uv': keyword_20235, 'full_matrices': keyword_20237}
    # Getting the type of '_compute_lwork' (line 124)
    _compute_lwork_20222 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 12), '_compute_lwork', False)
    # Calling _compute_lwork(args, kwargs) (line 124)
    _compute_lwork_call_result_20239 = invoke(stypy.reporting.localization.Localization(__file__, 124, 12), _compute_lwork_20222, *[gesXd_lwork_20223, subscript_call_result_20228, subscript_call_result_20233], **kwargs_20238)
    
    # Assigning a type to the variable 'lwork' (line 124)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 124, 4), 'lwork', _compute_lwork_call_result_20239)
    
    # Assigning a Call to a Tuple (line 128):
    
    # Assigning a Subscript to a Name (line 128):
    
    # Obtaining the type of the subscript
    int_20240 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 128, 4), 'int')
    
    # Call to gesXd(...): (line 128)
    # Processing the call arguments (line 128)
    # Getting the type of 'a1' (line 128)
    a1_20242 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 26), 'a1', False)
    # Processing the call keyword arguments (line 128)
    # Getting the type of 'compute_uv' (line 128)
    compute_uv_20243 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 41), 'compute_uv', False)
    keyword_20244 = compute_uv_20243
    # Getting the type of 'lwork' (line 128)
    lwork_20245 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 59), 'lwork', False)
    keyword_20246 = lwork_20245
    # Getting the type of 'full_matrices' (line 129)
    full_matrices_20247 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 40), 'full_matrices', False)
    keyword_20248 = full_matrices_20247
    # Getting the type of 'overwrite_a' (line 129)
    overwrite_a_20249 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 67), 'overwrite_a', False)
    keyword_20250 = overwrite_a_20249
    kwargs_20251 = {'compute_uv': keyword_20244, 'overwrite_a': keyword_20250, 'lwork': keyword_20246, 'full_matrices': keyword_20248}
    # Getting the type of 'gesXd' (line 128)
    gesXd_20241 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 20), 'gesXd', False)
    # Calling gesXd(args, kwargs) (line 128)
    gesXd_call_result_20252 = invoke(stypy.reporting.localization.Localization(__file__, 128, 20), gesXd_20241, *[a1_20242], **kwargs_20251)
    
    # Obtaining the member '__getitem__' of a type (line 128)
    getitem___20253 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 128, 4), gesXd_call_result_20252, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 128)
    subscript_call_result_20254 = invoke(stypy.reporting.localization.Localization(__file__, 128, 4), getitem___20253, int_20240)
    
    # Assigning a type to the variable 'tuple_var_assignment_20102' (line 128)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 128, 4), 'tuple_var_assignment_20102', subscript_call_result_20254)
    
    # Assigning a Subscript to a Name (line 128):
    
    # Obtaining the type of the subscript
    int_20255 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 128, 4), 'int')
    
    # Call to gesXd(...): (line 128)
    # Processing the call arguments (line 128)
    # Getting the type of 'a1' (line 128)
    a1_20257 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 26), 'a1', False)
    # Processing the call keyword arguments (line 128)
    # Getting the type of 'compute_uv' (line 128)
    compute_uv_20258 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 41), 'compute_uv', False)
    keyword_20259 = compute_uv_20258
    # Getting the type of 'lwork' (line 128)
    lwork_20260 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 59), 'lwork', False)
    keyword_20261 = lwork_20260
    # Getting the type of 'full_matrices' (line 129)
    full_matrices_20262 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 40), 'full_matrices', False)
    keyword_20263 = full_matrices_20262
    # Getting the type of 'overwrite_a' (line 129)
    overwrite_a_20264 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 67), 'overwrite_a', False)
    keyword_20265 = overwrite_a_20264
    kwargs_20266 = {'compute_uv': keyword_20259, 'overwrite_a': keyword_20265, 'lwork': keyword_20261, 'full_matrices': keyword_20263}
    # Getting the type of 'gesXd' (line 128)
    gesXd_20256 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 20), 'gesXd', False)
    # Calling gesXd(args, kwargs) (line 128)
    gesXd_call_result_20267 = invoke(stypy.reporting.localization.Localization(__file__, 128, 20), gesXd_20256, *[a1_20257], **kwargs_20266)
    
    # Obtaining the member '__getitem__' of a type (line 128)
    getitem___20268 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 128, 4), gesXd_call_result_20267, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 128)
    subscript_call_result_20269 = invoke(stypy.reporting.localization.Localization(__file__, 128, 4), getitem___20268, int_20255)
    
    # Assigning a type to the variable 'tuple_var_assignment_20103' (line 128)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 128, 4), 'tuple_var_assignment_20103', subscript_call_result_20269)
    
    # Assigning a Subscript to a Name (line 128):
    
    # Obtaining the type of the subscript
    int_20270 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 128, 4), 'int')
    
    # Call to gesXd(...): (line 128)
    # Processing the call arguments (line 128)
    # Getting the type of 'a1' (line 128)
    a1_20272 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 26), 'a1', False)
    # Processing the call keyword arguments (line 128)
    # Getting the type of 'compute_uv' (line 128)
    compute_uv_20273 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 41), 'compute_uv', False)
    keyword_20274 = compute_uv_20273
    # Getting the type of 'lwork' (line 128)
    lwork_20275 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 59), 'lwork', False)
    keyword_20276 = lwork_20275
    # Getting the type of 'full_matrices' (line 129)
    full_matrices_20277 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 40), 'full_matrices', False)
    keyword_20278 = full_matrices_20277
    # Getting the type of 'overwrite_a' (line 129)
    overwrite_a_20279 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 67), 'overwrite_a', False)
    keyword_20280 = overwrite_a_20279
    kwargs_20281 = {'compute_uv': keyword_20274, 'overwrite_a': keyword_20280, 'lwork': keyword_20276, 'full_matrices': keyword_20278}
    # Getting the type of 'gesXd' (line 128)
    gesXd_20271 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 20), 'gesXd', False)
    # Calling gesXd(args, kwargs) (line 128)
    gesXd_call_result_20282 = invoke(stypy.reporting.localization.Localization(__file__, 128, 20), gesXd_20271, *[a1_20272], **kwargs_20281)
    
    # Obtaining the member '__getitem__' of a type (line 128)
    getitem___20283 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 128, 4), gesXd_call_result_20282, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 128)
    subscript_call_result_20284 = invoke(stypy.reporting.localization.Localization(__file__, 128, 4), getitem___20283, int_20270)
    
    # Assigning a type to the variable 'tuple_var_assignment_20104' (line 128)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 128, 4), 'tuple_var_assignment_20104', subscript_call_result_20284)
    
    # Assigning a Subscript to a Name (line 128):
    
    # Obtaining the type of the subscript
    int_20285 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 128, 4), 'int')
    
    # Call to gesXd(...): (line 128)
    # Processing the call arguments (line 128)
    # Getting the type of 'a1' (line 128)
    a1_20287 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 26), 'a1', False)
    # Processing the call keyword arguments (line 128)
    # Getting the type of 'compute_uv' (line 128)
    compute_uv_20288 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 41), 'compute_uv', False)
    keyword_20289 = compute_uv_20288
    # Getting the type of 'lwork' (line 128)
    lwork_20290 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 59), 'lwork', False)
    keyword_20291 = lwork_20290
    # Getting the type of 'full_matrices' (line 129)
    full_matrices_20292 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 40), 'full_matrices', False)
    keyword_20293 = full_matrices_20292
    # Getting the type of 'overwrite_a' (line 129)
    overwrite_a_20294 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 67), 'overwrite_a', False)
    keyword_20295 = overwrite_a_20294
    kwargs_20296 = {'compute_uv': keyword_20289, 'overwrite_a': keyword_20295, 'lwork': keyword_20291, 'full_matrices': keyword_20293}
    # Getting the type of 'gesXd' (line 128)
    gesXd_20286 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 20), 'gesXd', False)
    # Calling gesXd(args, kwargs) (line 128)
    gesXd_call_result_20297 = invoke(stypy.reporting.localization.Localization(__file__, 128, 20), gesXd_20286, *[a1_20287], **kwargs_20296)
    
    # Obtaining the member '__getitem__' of a type (line 128)
    getitem___20298 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 128, 4), gesXd_call_result_20297, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 128)
    subscript_call_result_20299 = invoke(stypy.reporting.localization.Localization(__file__, 128, 4), getitem___20298, int_20285)
    
    # Assigning a type to the variable 'tuple_var_assignment_20105' (line 128)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 128, 4), 'tuple_var_assignment_20105', subscript_call_result_20299)
    
    # Assigning a Name to a Name (line 128):
    # Getting the type of 'tuple_var_assignment_20102' (line 128)
    tuple_var_assignment_20102_20300 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 4), 'tuple_var_assignment_20102')
    # Assigning a type to the variable 'u' (line 128)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 128, 4), 'u', tuple_var_assignment_20102_20300)
    
    # Assigning a Name to a Name (line 128):
    # Getting the type of 'tuple_var_assignment_20103' (line 128)
    tuple_var_assignment_20103_20301 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 4), 'tuple_var_assignment_20103')
    # Assigning a type to the variable 's' (line 128)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 128, 7), 's', tuple_var_assignment_20103_20301)
    
    # Assigning a Name to a Name (line 128):
    # Getting the type of 'tuple_var_assignment_20104' (line 128)
    tuple_var_assignment_20104_20302 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 4), 'tuple_var_assignment_20104')
    # Assigning a type to the variable 'v' (line 128)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 128, 10), 'v', tuple_var_assignment_20104_20302)
    
    # Assigning a Name to a Name (line 128):
    # Getting the type of 'tuple_var_assignment_20105' (line 128)
    tuple_var_assignment_20105_20303 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 4), 'tuple_var_assignment_20105')
    # Assigning a type to the variable 'info' (line 128)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 128, 13), 'info', tuple_var_assignment_20105_20303)
    
    
    # Getting the type of 'info' (line 131)
    info_20304 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 7), 'info')
    int_20305 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 131, 14), 'int')
    # Applying the binary operator '>' (line 131)
    result_gt_20306 = python_operator(stypy.reporting.localization.Localization(__file__, 131, 7), '>', info_20304, int_20305)
    
    # Testing the type of an if condition (line 131)
    if_condition_20307 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 131, 4), result_gt_20306)
    # Assigning a type to the variable 'if_condition_20307' (line 131)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 131, 4), 'if_condition_20307', if_condition_20307)
    # SSA begins for if statement (line 131)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to LinAlgError(...): (line 132)
    # Processing the call arguments (line 132)
    str_20309 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 132, 26), 'str', 'SVD did not converge')
    # Processing the call keyword arguments (line 132)
    kwargs_20310 = {}
    # Getting the type of 'LinAlgError' (line 132)
    LinAlgError_20308 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 14), 'LinAlgError', False)
    # Calling LinAlgError(args, kwargs) (line 132)
    LinAlgError_call_result_20311 = invoke(stypy.reporting.localization.Localization(__file__, 132, 14), LinAlgError_20308, *[str_20309], **kwargs_20310)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 132, 8), LinAlgError_call_result_20311, 'raise parameter', BaseException)
    # SSA join for if statement (line 131)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'info' (line 133)
    info_20312 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 7), 'info')
    int_20313 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 133, 14), 'int')
    # Applying the binary operator '<' (line 133)
    result_lt_20314 = python_operator(stypy.reporting.localization.Localization(__file__, 133, 7), '<', info_20312, int_20313)
    
    # Testing the type of an if condition (line 133)
    if_condition_20315 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 133, 4), result_lt_20314)
    # Assigning a type to the variable 'if_condition_20315' (line 133)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 133, 4), 'if_condition_20315', if_condition_20315)
    # SSA begins for if statement (line 133)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 134)
    # Processing the call arguments (line 134)
    str_20317 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 134, 25), 'str', 'illegal value in %d-th argument of internal gesdd')
    
    # Getting the type of 'info' (line 135)
    info_20318 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 28), 'info', False)
    # Applying the 'usub' unary operator (line 135)
    result___neg___20319 = python_operator(stypy.reporting.localization.Localization(__file__, 135, 27), 'usub', info_20318)
    
    # Applying the binary operator '%' (line 134)
    result_mod_20320 = python_operator(stypy.reporting.localization.Localization(__file__, 134, 25), '%', str_20317, result___neg___20319)
    
    # Processing the call keyword arguments (line 134)
    kwargs_20321 = {}
    # Getting the type of 'ValueError' (line 134)
    ValueError_20316 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 134)
    ValueError_call_result_20322 = invoke(stypy.reporting.localization.Localization(__file__, 134, 14), ValueError_20316, *[result_mod_20320], **kwargs_20321)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 134, 8), ValueError_call_result_20322, 'raise parameter', BaseException)
    # SSA join for if statement (line 133)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Getting the type of 'compute_uv' (line 136)
    compute_uv_20323 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 7), 'compute_uv')
    # Testing the type of an if condition (line 136)
    if_condition_20324 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 136, 4), compute_uv_20323)
    # Assigning a type to the variable 'if_condition_20324' (line 136)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 136, 4), 'if_condition_20324', if_condition_20324)
    # SSA begins for if statement (line 136)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Obtaining an instance of the builtin type 'tuple' (line 137)
    tuple_20325 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 137, 15), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 137)
    # Adding element type (line 137)
    # Getting the type of 'u' (line 137)
    u_20326 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 15), 'u')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 137, 15), tuple_20325, u_20326)
    # Adding element type (line 137)
    # Getting the type of 's' (line 137)
    s_20327 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 18), 's')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 137, 15), tuple_20325, s_20327)
    # Adding element type (line 137)
    # Getting the type of 'v' (line 137)
    v_20328 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 21), 'v')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 137, 15), tuple_20325, v_20328)
    
    # Assigning a type to the variable 'stypy_return_type' (line 137)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 137, 8), 'stypy_return_type', tuple_20325)
    # SSA branch for the else part of an if statement (line 136)
    module_type_store.open_ssa_branch('else')
    # Getting the type of 's' (line 139)
    s_20329 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 15), 's')
    # Assigning a type to the variable 'stypy_return_type' (line 139)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 139, 8), 'stypy_return_type', s_20329)
    # SSA join for if statement (line 136)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'svd(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'svd' in the type store
    # Getting the type of 'stypy_return_type' (line 16)
    stypy_return_type_20330 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_20330)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'svd'
    return stypy_return_type_20330

# Assigning a type to the variable 'svd' (line 16)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 0), 'svd', svd)

@norecursion
def svdvals(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'False' (line 142)
    False_20331 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 27), 'False')
    # Getting the type of 'True' (line 142)
    True_20332 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 47), 'True')
    defaults = [False_20331, True_20332]
    # Create a new context for function 'svdvals'
    module_type_store = module_type_store.open_function_context('svdvals', 142, 0, False)
    
    # Passed parameters checking function
    svdvals.stypy_localization = localization
    svdvals.stypy_type_of_self = None
    svdvals.stypy_type_store = module_type_store
    svdvals.stypy_function_name = 'svdvals'
    svdvals.stypy_param_names_list = ['a', 'overwrite_a', 'check_finite']
    svdvals.stypy_varargs_param_name = None
    svdvals.stypy_kwargs_param_name = None
    svdvals.stypy_call_defaults = defaults
    svdvals.stypy_call_varargs = varargs
    svdvals.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'svdvals', ['a', 'overwrite_a', 'check_finite'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'svdvals', localization, ['a', 'overwrite_a', 'check_finite'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'svdvals(...)' code ##################

    str_20333 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 224, (-1)), 'str', '\n    Compute singular values of a matrix.\n\n    Parameters\n    ----------\n    a : (M, N) array_like\n        Matrix to decompose.\n    overwrite_a : bool, optional\n        Whether to overwrite `a`; may improve performance.\n        Default is False.\n    check_finite : bool, optional\n        Whether to check that the input matrix contains only finite numbers.\n        Disabling may give a performance gain, but may result in problems\n        (crashes, non-termination) if the inputs do contain infinities or NaNs.\n\n    Returns\n    -------\n    s : (min(M, N),) ndarray\n        The singular values, sorted in decreasing order.\n\n    Raises\n    ------\n    LinAlgError\n        If SVD computation does not converge.\n\n    Notes\n    -----\n    ``svdvals(a)`` only differs from ``svd(a, compute_uv=False)`` by its\n    handling of the edge case of empty ``a``, where it returns an\n    empty sequence:\n\n    >>> a = np.empty((0, 2))\n    >>> from scipy.linalg import svdvals\n    >>> svdvals(a)\n    array([], dtype=float64)\n\n    See Also\n    --------\n    svd : Compute the full singular value decomposition of a matrix.\n    diagsvd : Construct the Sigma matrix, given the vector s.\n\n    Examples\n    --------\n    >>> from scipy.linalg import svdvals\n    >>> m = np.array([[1.0, 0.0],\n    ...               [2.0, 3.0],\n    ...               [1.0, 1.0],\n    ...               [0.0, 2.0],\n    ...               [1.0, 0.0]])\n    >>> svdvals(m)\n    array([ 4.28091555,  1.63516424])\n\n    We can verify the maximum singular value of `m` by computing the maximum\n    length of `m.dot(u)` over all the unit vectors `u` in the (x,y) plane.\n    We approximate "all" the unit vectors with a large sample.  Because\n    of linearity, we only need the unit vectors with angles in [0, pi].\n\n    >>> t = np.linspace(0, np.pi, 2000)\n    >>> u = np.array([np.cos(t), np.sin(t)])\n    >>> np.linalg.norm(m.dot(u), axis=0).max()\n    4.2809152422538475\n\n    `p` is a projection matrix with rank 1.  With exact arithmetic,\n    its singular values would be [1, 0, 0, 0].\n\n    >>> v = np.array([0.1, 0.3, 0.9, 0.3])\n    >>> p = np.outer(v, v)\n    >>> svdvals(p)\n    array([  1.00000000e+00,   2.02021698e-17,   1.56692500e-17,\n             8.15115104e-34])\n\n    The singular values of an orthogonal matrix are all 1.  Here we\n    create a random orthogonal matrix by using the `rvs()` method of\n    `scipy.stats.ortho_group`.\n\n    >>> from scipy.stats import ortho_group\n    >>> np.random.seed(123)\n    >>> orth = ortho_group.rvs(4)\n    >>> svdvals(orth)\n    array([ 1.,  1.,  1.,  1.])\n\n    ')
    
    # Assigning a Call to a Name (line 225):
    
    # Assigning a Call to a Name (line 225):
    
    # Call to _asarray_validated(...): (line 225)
    # Processing the call arguments (line 225)
    # Getting the type of 'a' (line 225)
    a_20335 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 225, 27), 'a', False)
    # Processing the call keyword arguments (line 225)
    # Getting the type of 'check_finite' (line 225)
    check_finite_20336 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 225, 43), 'check_finite', False)
    keyword_20337 = check_finite_20336
    kwargs_20338 = {'check_finite': keyword_20337}
    # Getting the type of '_asarray_validated' (line 225)
    _asarray_validated_20334 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 225, 8), '_asarray_validated', False)
    # Calling _asarray_validated(args, kwargs) (line 225)
    _asarray_validated_call_result_20339 = invoke(stypy.reporting.localization.Localization(__file__, 225, 8), _asarray_validated_20334, *[a_20335], **kwargs_20338)
    
    # Assigning a type to the variable 'a' (line 225)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 225, 4), 'a', _asarray_validated_call_result_20339)
    
    # Getting the type of 'a' (line 226)
    a_20340 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 7), 'a')
    # Obtaining the member 'size' of a type (line 226)
    size_20341 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 226, 7), a_20340, 'size')
    # Testing the type of an if condition (line 226)
    if_condition_20342 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 226, 4), size_20341)
    # Assigning a type to the variable 'if_condition_20342' (line 226)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 226, 4), 'if_condition_20342', if_condition_20342)
    # SSA begins for if statement (line 226)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to svd(...): (line 227)
    # Processing the call arguments (line 227)
    # Getting the type of 'a' (line 227)
    a_20344 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 19), 'a', False)
    # Processing the call keyword arguments (line 227)
    int_20345 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 227, 33), 'int')
    keyword_20346 = int_20345
    # Getting the type of 'overwrite_a' (line 227)
    overwrite_a_20347 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 48), 'overwrite_a', False)
    keyword_20348 = overwrite_a_20347
    # Getting the type of 'False' (line 228)
    False_20349 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 32), 'False', False)
    keyword_20350 = False_20349
    kwargs_20351 = {'compute_uv': keyword_20346, 'overwrite_a': keyword_20348, 'check_finite': keyword_20350}
    # Getting the type of 'svd' (line 227)
    svd_20343 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 15), 'svd', False)
    # Calling svd(args, kwargs) (line 227)
    svd_call_result_20352 = invoke(stypy.reporting.localization.Localization(__file__, 227, 15), svd_20343, *[a_20344], **kwargs_20351)
    
    # Assigning a type to the variable 'stypy_return_type' (line 227)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 227, 8), 'stypy_return_type', svd_call_result_20352)
    # SSA branch for the else part of an if statement (line 226)
    module_type_store.open_ssa_branch('else')
    
    
    
    # Call to len(...): (line 229)
    # Processing the call arguments (line 229)
    # Getting the type of 'a' (line 229)
    a_20354 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 229, 13), 'a', False)
    # Obtaining the member 'shape' of a type (line 229)
    shape_20355 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 229, 13), a_20354, 'shape')
    # Processing the call keyword arguments (line 229)
    kwargs_20356 = {}
    # Getting the type of 'len' (line 229)
    len_20353 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 229, 9), 'len', False)
    # Calling len(args, kwargs) (line 229)
    len_call_result_20357 = invoke(stypy.reporting.localization.Localization(__file__, 229, 9), len_20353, *[shape_20355], **kwargs_20356)
    
    int_20358 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 229, 25), 'int')
    # Applying the binary operator '!=' (line 229)
    result_ne_20359 = python_operator(stypy.reporting.localization.Localization(__file__, 229, 9), '!=', len_call_result_20357, int_20358)
    
    # Testing the type of an if condition (line 229)
    if_condition_20360 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 229, 9), result_ne_20359)
    # Assigning a type to the variable 'if_condition_20360' (line 229)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 229, 9), 'if_condition_20360', if_condition_20360)
    # SSA begins for if statement (line 229)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 230)
    # Processing the call arguments (line 230)
    str_20362 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 230, 25), 'str', 'expected matrix')
    # Processing the call keyword arguments (line 230)
    kwargs_20363 = {}
    # Getting the type of 'ValueError' (line 230)
    ValueError_20361 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 230)
    ValueError_call_result_20364 = invoke(stypy.reporting.localization.Localization(__file__, 230, 14), ValueError_20361, *[str_20362], **kwargs_20363)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 230, 8), ValueError_call_result_20364, 'raise parameter', BaseException)
    # SSA branch for the else part of an if statement (line 229)
    module_type_store.open_ssa_branch('else')
    
    # Call to empty(...): (line 232)
    # Processing the call arguments (line 232)
    int_20367 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 232, 27), 'int')
    # Processing the call keyword arguments (line 232)
    kwargs_20368 = {}
    # Getting the type of 'numpy' (line 232)
    numpy_20365 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 15), 'numpy', False)
    # Obtaining the member 'empty' of a type (line 232)
    empty_20366 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 232, 15), numpy_20365, 'empty')
    # Calling empty(args, kwargs) (line 232)
    empty_call_result_20369 = invoke(stypy.reporting.localization.Localization(__file__, 232, 15), empty_20366, *[int_20367], **kwargs_20368)
    
    # Assigning a type to the variable 'stypy_return_type' (line 232)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 232, 8), 'stypy_return_type', empty_call_result_20369)
    # SSA join for if statement (line 229)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 226)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'svdvals(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'svdvals' in the type store
    # Getting the type of 'stypy_return_type' (line 142)
    stypy_return_type_20370 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_20370)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'svdvals'
    return stypy_return_type_20370

# Assigning a type to the variable 'svdvals' (line 142)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 142, 0), 'svdvals', svdvals)

@norecursion
def diagsvd(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'diagsvd'
    module_type_store = module_type_store.open_function_context('diagsvd', 235, 0, False)
    
    # Passed parameters checking function
    diagsvd.stypy_localization = localization
    diagsvd.stypy_type_of_self = None
    diagsvd.stypy_type_store = module_type_store
    diagsvd.stypy_function_name = 'diagsvd'
    diagsvd.stypy_param_names_list = ['s', 'M', 'N']
    diagsvd.stypy_varargs_param_name = None
    diagsvd.stypy_kwargs_param_name = None
    diagsvd.stypy_call_defaults = defaults
    diagsvd.stypy_call_varargs = varargs
    diagsvd.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'diagsvd', ['s', 'M', 'N'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'diagsvd', localization, ['s', 'M', 'N'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'diagsvd(...)' code ##################

    str_20371 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 253, (-1)), 'str', '\n    Construct the sigma matrix in SVD from singular values and size M, N.\n\n    Parameters\n    ----------\n    s : (M,) or (N,) array_like\n        Singular values\n    M : int\n        Size of the matrix whose singular values are `s`.\n    N : int\n        Size of the matrix whose singular values are `s`.\n\n    Returns\n    -------\n    S : (M, N) ndarray\n        The S-matrix in the singular value decomposition\n\n    ')
    
    # Assigning a Call to a Name (line 254):
    
    # Assigning a Call to a Name (line 254):
    
    # Call to diag(...): (line 254)
    # Processing the call arguments (line 254)
    # Getting the type of 's' (line 254)
    s_20373 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 254, 16), 's', False)
    # Processing the call keyword arguments (line 254)
    kwargs_20374 = {}
    # Getting the type of 'diag' (line 254)
    diag_20372 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 254, 11), 'diag', False)
    # Calling diag(args, kwargs) (line 254)
    diag_call_result_20375 = invoke(stypy.reporting.localization.Localization(__file__, 254, 11), diag_20372, *[s_20373], **kwargs_20374)
    
    # Assigning a type to the variable 'part' (line 254)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 254, 4), 'part', diag_call_result_20375)
    
    # Assigning a Attribute to a Name (line 255):
    
    # Assigning a Attribute to a Name (line 255):
    # Getting the type of 'part' (line 255)
    part_20376 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 255, 10), 'part')
    # Obtaining the member 'dtype' of a type (line 255)
    dtype_20377 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 255, 10), part_20376, 'dtype')
    # Obtaining the member 'char' of a type (line 255)
    char_20378 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 255, 10), dtype_20377, 'char')
    # Assigning a type to the variable 'typ' (line 255)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 255, 4), 'typ', char_20378)
    
    # Assigning a Call to a Name (line 256):
    
    # Assigning a Call to a Name (line 256):
    
    # Call to len(...): (line 256)
    # Processing the call arguments (line 256)
    # Getting the type of 's' (line 256)
    s_20380 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 256, 15), 's', False)
    # Processing the call keyword arguments (line 256)
    kwargs_20381 = {}
    # Getting the type of 'len' (line 256)
    len_20379 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 256, 11), 'len', False)
    # Calling len(args, kwargs) (line 256)
    len_call_result_20382 = invoke(stypy.reporting.localization.Localization(__file__, 256, 11), len_20379, *[s_20380], **kwargs_20381)
    
    # Assigning a type to the variable 'MorN' (line 256)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 256, 4), 'MorN', len_call_result_20382)
    
    
    # Getting the type of 'MorN' (line 257)
    MorN_20383 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 257, 7), 'MorN')
    # Getting the type of 'M' (line 257)
    M_20384 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 257, 15), 'M')
    # Applying the binary operator '==' (line 257)
    result_eq_20385 = python_operator(stypy.reporting.localization.Localization(__file__, 257, 7), '==', MorN_20383, M_20384)
    
    # Testing the type of an if condition (line 257)
    if_condition_20386 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 257, 4), result_eq_20385)
    # Assigning a type to the variable 'if_condition_20386' (line 257)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 257, 4), 'if_condition_20386', if_condition_20386)
    # SSA begins for if statement (line 257)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Obtaining the type of the subscript
    
    # Obtaining an instance of the builtin type 'tuple' (line 258)
    tuple_20387 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 258, 18), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 258)
    # Adding element type (line 258)
    str_20388 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 258, 18), 'str', '-1')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 258, 18), tuple_20387, str_20388)
    # Adding element type (line 258)
    # Getting the type of 'part' (line 258)
    part_20389 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 258, 24), 'part')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 258, 18), tuple_20387, part_20389)
    # Adding element type (line 258)
    
    # Call to zeros(...): (line 258)
    # Processing the call arguments (line 258)
    
    # Obtaining an instance of the builtin type 'tuple' (line 258)
    tuple_20391 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 258, 37), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 258)
    # Adding element type (line 258)
    # Getting the type of 'M' (line 258)
    M_20392 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 258, 37), 'M', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 258, 37), tuple_20391, M_20392)
    # Adding element type (line 258)
    # Getting the type of 'N' (line 258)
    N_20393 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 258, 40), 'N', False)
    # Getting the type of 'M' (line 258)
    M_20394 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 258, 42), 'M', False)
    # Applying the binary operator '-' (line 258)
    result_sub_20395 = python_operator(stypy.reporting.localization.Localization(__file__, 258, 40), '-', N_20393, M_20394)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 258, 37), tuple_20391, result_sub_20395)
    
    # Getting the type of 'typ' (line 258)
    typ_20396 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 258, 46), 'typ', False)
    # Processing the call keyword arguments (line 258)
    kwargs_20397 = {}
    # Getting the type of 'zeros' (line 258)
    zeros_20390 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 258, 30), 'zeros', False)
    # Calling zeros(args, kwargs) (line 258)
    zeros_call_result_20398 = invoke(stypy.reporting.localization.Localization(__file__, 258, 30), zeros_20390, *[tuple_20391, typ_20396], **kwargs_20397)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 258, 18), tuple_20387, zeros_call_result_20398)
    
    # Getting the type of 'r_' (line 258)
    r__20399 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 258, 15), 'r_')
    # Obtaining the member '__getitem__' of a type (line 258)
    getitem___20400 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 258, 15), r__20399, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 258)
    subscript_call_result_20401 = invoke(stypy.reporting.localization.Localization(__file__, 258, 15), getitem___20400, tuple_20387)
    
    # Assigning a type to the variable 'stypy_return_type' (line 258)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 258, 8), 'stypy_return_type', subscript_call_result_20401)
    # SSA branch for the else part of an if statement (line 257)
    module_type_store.open_ssa_branch('else')
    
    
    # Getting the type of 'MorN' (line 259)
    MorN_20402 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 259, 9), 'MorN')
    # Getting the type of 'N' (line 259)
    N_20403 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 259, 17), 'N')
    # Applying the binary operator '==' (line 259)
    result_eq_20404 = python_operator(stypy.reporting.localization.Localization(__file__, 259, 9), '==', MorN_20402, N_20403)
    
    # Testing the type of an if condition (line 259)
    if_condition_20405 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 259, 9), result_eq_20404)
    # Assigning a type to the variable 'if_condition_20405' (line 259)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 259, 9), 'if_condition_20405', if_condition_20405)
    # SSA begins for if statement (line 259)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Obtaining the type of the subscript
    
    # Obtaining an instance of the builtin type 'tuple' (line 260)
    tuple_20406 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 260, 18), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 260)
    # Adding element type (line 260)
    # Getting the type of 'part' (line 260)
    part_20407 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 260, 18), 'part')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 260, 18), tuple_20406, part_20407)
    # Adding element type (line 260)
    
    # Call to zeros(...): (line 260)
    # Processing the call arguments (line 260)
    
    # Obtaining an instance of the builtin type 'tuple' (line 260)
    tuple_20409 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 260, 31), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 260)
    # Adding element type (line 260)
    # Getting the type of 'M' (line 260)
    M_20410 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 260, 31), 'M', False)
    # Getting the type of 'N' (line 260)
    N_20411 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 260, 33), 'N', False)
    # Applying the binary operator '-' (line 260)
    result_sub_20412 = python_operator(stypy.reporting.localization.Localization(__file__, 260, 31), '-', M_20410, N_20411)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 260, 31), tuple_20409, result_sub_20412)
    # Adding element type (line 260)
    # Getting the type of 'N' (line 260)
    N_20413 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 260, 36), 'N', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 260, 31), tuple_20409, N_20413)
    
    # Getting the type of 'typ' (line 260)
    typ_20414 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 260, 40), 'typ', False)
    # Processing the call keyword arguments (line 260)
    kwargs_20415 = {}
    # Getting the type of 'zeros' (line 260)
    zeros_20408 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 260, 24), 'zeros', False)
    # Calling zeros(args, kwargs) (line 260)
    zeros_call_result_20416 = invoke(stypy.reporting.localization.Localization(__file__, 260, 24), zeros_20408, *[tuple_20409, typ_20414], **kwargs_20415)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 260, 18), tuple_20406, zeros_call_result_20416)
    
    # Getting the type of 'r_' (line 260)
    r__20417 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 260, 15), 'r_')
    # Obtaining the member '__getitem__' of a type (line 260)
    getitem___20418 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 260, 15), r__20417, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 260)
    subscript_call_result_20419 = invoke(stypy.reporting.localization.Localization(__file__, 260, 15), getitem___20418, tuple_20406)
    
    # Assigning a type to the variable 'stypy_return_type' (line 260)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 260, 8), 'stypy_return_type', subscript_call_result_20419)
    # SSA branch for the else part of an if statement (line 259)
    module_type_store.open_ssa_branch('else')
    
    # Call to ValueError(...): (line 262)
    # Processing the call arguments (line 262)
    str_20421 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 262, 25), 'str', 'Length of s must be M or N.')
    # Processing the call keyword arguments (line 262)
    kwargs_20422 = {}
    # Getting the type of 'ValueError' (line 262)
    ValueError_20420 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 262, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 262)
    ValueError_call_result_20423 = invoke(stypy.reporting.localization.Localization(__file__, 262, 14), ValueError_20420, *[str_20421], **kwargs_20422)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 262, 8), ValueError_call_result_20423, 'raise parameter', BaseException)
    # SSA join for if statement (line 259)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 257)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'diagsvd(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'diagsvd' in the type store
    # Getting the type of 'stypy_return_type' (line 235)
    stypy_return_type_20424 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_20424)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'diagsvd'
    return stypy_return_type_20424

# Assigning a type to the variable 'diagsvd' (line 235)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 235, 0), 'diagsvd', diagsvd)

@norecursion
def orth(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'orth'
    module_type_store = module_type_store.open_function_context('orth', 267, 0, False)
    
    # Passed parameters checking function
    orth.stypy_localization = localization
    orth.stypy_type_of_self = None
    orth.stypy_type_store = module_type_store
    orth.stypy_function_name = 'orth'
    orth.stypy_param_names_list = ['A']
    orth.stypy_varargs_param_name = None
    orth.stypy_kwargs_param_name = None
    orth.stypy_call_defaults = defaults
    orth.stypy_call_varargs = varargs
    orth.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'orth', ['A'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'orth', localization, ['A'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'orth(...)' code ##################

    str_20425 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 286, (-1)), 'str', '\n    Construct an orthonormal basis for the range of A using SVD\n\n    Parameters\n    ----------\n    A : (M, N) array_like\n        Input array\n\n    Returns\n    -------\n    Q : (M, K) ndarray\n        Orthonormal basis for the range of A.\n        K = effective rank of A, as determined by automatic cutoff\n\n    See also\n    --------\n    svd : Singular value decomposition of a matrix\n\n    ')
    
    # Assigning a Call to a Tuple (line 287):
    
    # Assigning a Subscript to a Name (line 287):
    
    # Obtaining the type of the subscript
    int_20426 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 287, 4), 'int')
    
    # Call to svd(...): (line 287)
    # Processing the call arguments (line 287)
    # Getting the type of 'A' (line 287)
    A_20428 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 287, 19), 'A', False)
    # Processing the call keyword arguments (line 287)
    # Getting the type of 'False' (line 287)
    False_20429 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 287, 36), 'False', False)
    keyword_20430 = False_20429
    kwargs_20431 = {'full_matrices': keyword_20430}
    # Getting the type of 'svd' (line 287)
    svd_20427 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 287, 15), 'svd', False)
    # Calling svd(args, kwargs) (line 287)
    svd_call_result_20432 = invoke(stypy.reporting.localization.Localization(__file__, 287, 15), svd_20427, *[A_20428], **kwargs_20431)
    
    # Obtaining the member '__getitem__' of a type (line 287)
    getitem___20433 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 287, 4), svd_call_result_20432, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 287)
    subscript_call_result_20434 = invoke(stypy.reporting.localization.Localization(__file__, 287, 4), getitem___20433, int_20426)
    
    # Assigning a type to the variable 'tuple_var_assignment_20106' (line 287)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 287, 4), 'tuple_var_assignment_20106', subscript_call_result_20434)
    
    # Assigning a Subscript to a Name (line 287):
    
    # Obtaining the type of the subscript
    int_20435 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 287, 4), 'int')
    
    # Call to svd(...): (line 287)
    # Processing the call arguments (line 287)
    # Getting the type of 'A' (line 287)
    A_20437 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 287, 19), 'A', False)
    # Processing the call keyword arguments (line 287)
    # Getting the type of 'False' (line 287)
    False_20438 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 287, 36), 'False', False)
    keyword_20439 = False_20438
    kwargs_20440 = {'full_matrices': keyword_20439}
    # Getting the type of 'svd' (line 287)
    svd_20436 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 287, 15), 'svd', False)
    # Calling svd(args, kwargs) (line 287)
    svd_call_result_20441 = invoke(stypy.reporting.localization.Localization(__file__, 287, 15), svd_20436, *[A_20437], **kwargs_20440)
    
    # Obtaining the member '__getitem__' of a type (line 287)
    getitem___20442 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 287, 4), svd_call_result_20441, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 287)
    subscript_call_result_20443 = invoke(stypy.reporting.localization.Localization(__file__, 287, 4), getitem___20442, int_20435)
    
    # Assigning a type to the variable 'tuple_var_assignment_20107' (line 287)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 287, 4), 'tuple_var_assignment_20107', subscript_call_result_20443)
    
    # Assigning a Subscript to a Name (line 287):
    
    # Obtaining the type of the subscript
    int_20444 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 287, 4), 'int')
    
    # Call to svd(...): (line 287)
    # Processing the call arguments (line 287)
    # Getting the type of 'A' (line 287)
    A_20446 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 287, 19), 'A', False)
    # Processing the call keyword arguments (line 287)
    # Getting the type of 'False' (line 287)
    False_20447 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 287, 36), 'False', False)
    keyword_20448 = False_20447
    kwargs_20449 = {'full_matrices': keyword_20448}
    # Getting the type of 'svd' (line 287)
    svd_20445 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 287, 15), 'svd', False)
    # Calling svd(args, kwargs) (line 287)
    svd_call_result_20450 = invoke(stypy.reporting.localization.Localization(__file__, 287, 15), svd_20445, *[A_20446], **kwargs_20449)
    
    # Obtaining the member '__getitem__' of a type (line 287)
    getitem___20451 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 287, 4), svd_call_result_20450, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 287)
    subscript_call_result_20452 = invoke(stypy.reporting.localization.Localization(__file__, 287, 4), getitem___20451, int_20444)
    
    # Assigning a type to the variable 'tuple_var_assignment_20108' (line 287)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 287, 4), 'tuple_var_assignment_20108', subscript_call_result_20452)
    
    # Assigning a Name to a Name (line 287):
    # Getting the type of 'tuple_var_assignment_20106' (line 287)
    tuple_var_assignment_20106_20453 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 287, 4), 'tuple_var_assignment_20106')
    # Assigning a type to the variable 'u' (line 287)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 287, 4), 'u', tuple_var_assignment_20106_20453)
    
    # Assigning a Name to a Name (line 287):
    # Getting the type of 'tuple_var_assignment_20107' (line 287)
    tuple_var_assignment_20107_20454 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 287, 4), 'tuple_var_assignment_20107')
    # Assigning a type to the variable 's' (line 287)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 287, 7), 's', tuple_var_assignment_20107_20454)
    
    # Assigning a Name to a Name (line 287):
    # Getting the type of 'tuple_var_assignment_20108' (line 287)
    tuple_var_assignment_20108_20455 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 287, 4), 'tuple_var_assignment_20108')
    # Assigning a type to the variable 'vh' (line 287)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 287, 10), 'vh', tuple_var_assignment_20108_20455)
    
    # Assigning a Attribute to a Tuple (line 288):
    
    # Assigning a Subscript to a Name (line 288):
    
    # Obtaining the type of the subscript
    int_20456 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 288, 4), 'int')
    # Getting the type of 'A' (line 288)
    A_20457 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 288, 11), 'A')
    # Obtaining the member 'shape' of a type (line 288)
    shape_20458 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 288, 11), A_20457, 'shape')
    # Obtaining the member '__getitem__' of a type (line 288)
    getitem___20459 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 288, 4), shape_20458, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 288)
    subscript_call_result_20460 = invoke(stypy.reporting.localization.Localization(__file__, 288, 4), getitem___20459, int_20456)
    
    # Assigning a type to the variable 'tuple_var_assignment_20109' (line 288)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 288, 4), 'tuple_var_assignment_20109', subscript_call_result_20460)
    
    # Assigning a Subscript to a Name (line 288):
    
    # Obtaining the type of the subscript
    int_20461 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 288, 4), 'int')
    # Getting the type of 'A' (line 288)
    A_20462 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 288, 11), 'A')
    # Obtaining the member 'shape' of a type (line 288)
    shape_20463 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 288, 11), A_20462, 'shape')
    # Obtaining the member '__getitem__' of a type (line 288)
    getitem___20464 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 288, 4), shape_20463, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 288)
    subscript_call_result_20465 = invoke(stypy.reporting.localization.Localization(__file__, 288, 4), getitem___20464, int_20461)
    
    # Assigning a type to the variable 'tuple_var_assignment_20110' (line 288)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 288, 4), 'tuple_var_assignment_20110', subscript_call_result_20465)
    
    # Assigning a Name to a Name (line 288):
    # Getting the type of 'tuple_var_assignment_20109' (line 288)
    tuple_var_assignment_20109_20466 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 288, 4), 'tuple_var_assignment_20109')
    # Assigning a type to the variable 'M' (line 288)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 288, 4), 'M', tuple_var_assignment_20109_20466)
    
    # Assigning a Name to a Name (line 288):
    # Getting the type of 'tuple_var_assignment_20110' (line 288)
    tuple_var_assignment_20110_20467 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 288, 4), 'tuple_var_assignment_20110')
    # Assigning a type to the variable 'N' (line 288)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 288, 7), 'N', tuple_var_assignment_20110_20467)
    
    # Assigning a Attribute to a Name (line 289):
    
    # Assigning a Attribute to a Name (line 289):
    
    # Call to finfo(...): (line 289)
    # Processing the call arguments (line 289)
    # Getting the type of 'float' (line 289)
    float_20470 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 289, 22), 'float', False)
    # Processing the call keyword arguments (line 289)
    kwargs_20471 = {}
    # Getting the type of 'numpy' (line 289)
    numpy_20468 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 289, 10), 'numpy', False)
    # Obtaining the member 'finfo' of a type (line 289)
    finfo_20469 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 289, 10), numpy_20468, 'finfo')
    # Calling finfo(args, kwargs) (line 289)
    finfo_call_result_20472 = invoke(stypy.reporting.localization.Localization(__file__, 289, 10), finfo_20469, *[float_20470], **kwargs_20471)
    
    # Obtaining the member 'eps' of a type (line 289)
    eps_20473 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 289, 10), finfo_call_result_20472, 'eps')
    # Assigning a type to the variable 'eps' (line 289)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 289, 4), 'eps', eps_20473)
    
    # Assigning a BinOp to a Name (line 290):
    
    # Assigning a BinOp to a Name (line 290):
    
    # Call to max(...): (line 290)
    # Processing the call arguments (line 290)
    # Getting the type of 'M' (line 290)
    M_20475 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 290, 14), 'M', False)
    # Getting the type of 'N' (line 290)
    N_20476 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 290, 17), 'N', False)
    # Processing the call keyword arguments (line 290)
    kwargs_20477 = {}
    # Getting the type of 'max' (line 290)
    max_20474 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 290, 10), 'max', False)
    # Calling max(args, kwargs) (line 290)
    max_call_result_20478 = invoke(stypy.reporting.localization.Localization(__file__, 290, 10), max_20474, *[M_20475, N_20476], **kwargs_20477)
    
    
    # Call to amax(...): (line 290)
    # Processing the call arguments (line 290)
    # Getting the type of 's' (line 290)
    s_20481 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 290, 33), 's', False)
    # Processing the call keyword arguments (line 290)
    kwargs_20482 = {}
    # Getting the type of 'numpy' (line 290)
    numpy_20479 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 290, 22), 'numpy', False)
    # Obtaining the member 'amax' of a type (line 290)
    amax_20480 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 290, 22), numpy_20479, 'amax')
    # Calling amax(args, kwargs) (line 290)
    amax_call_result_20483 = invoke(stypy.reporting.localization.Localization(__file__, 290, 22), amax_20480, *[s_20481], **kwargs_20482)
    
    # Applying the binary operator '*' (line 290)
    result_mul_20484 = python_operator(stypy.reporting.localization.Localization(__file__, 290, 10), '*', max_call_result_20478, amax_call_result_20483)
    
    # Getting the type of 'eps' (line 290)
    eps_20485 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 290, 38), 'eps')
    # Applying the binary operator '*' (line 290)
    result_mul_20486 = python_operator(stypy.reporting.localization.Localization(__file__, 290, 36), '*', result_mul_20484, eps_20485)
    
    # Assigning a type to the variable 'tol' (line 290)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 290, 4), 'tol', result_mul_20486)
    
    # Assigning a Call to a Name (line 291):
    
    # Assigning a Call to a Name (line 291):
    
    # Call to sum(...): (line 291)
    # Processing the call arguments (line 291)
    
    # Getting the type of 's' (line 291)
    s_20489 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 291, 20), 's', False)
    # Getting the type of 'tol' (line 291)
    tol_20490 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 291, 24), 'tol', False)
    # Applying the binary operator '>' (line 291)
    result_gt_20491 = python_operator(stypy.reporting.localization.Localization(__file__, 291, 20), '>', s_20489, tol_20490)
    
    # Processing the call keyword arguments (line 291)
    # Getting the type of 'int' (line 291)
    int_20492 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 291, 35), 'int', False)
    keyword_20493 = int_20492
    kwargs_20494 = {'dtype': keyword_20493}
    # Getting the type of 'numpy' (line 291)
    numpy_20487 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 291, 10), 'numpy', False)
    # Obtaining the member 'sum' of a type (line 291)
    sum_20488 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 291, 10), numpy_20487, 'sum')
    # Calling sum(args, kwargs) (line 291)
    sum_call_result_20495 = invoke(stypy.reporting.localization.Localization(__file__, 291, 10), sum_20488, *[result_gt_20491], **kwargs_20494)
    
    # Assigning a type to the variable 'num' (line 291)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 291, 4), 'num', sum_call_result_20495)
    
    # Assigning a Subscript to a Name (line 292):
    
    # Assigning a Subscript to a Name (line 292):
    
    # Obtaining the type of the subscript
    slice_20496 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 292, 8), None, None, None)
    # Getting the type of 'num' (line 292)
    num_20497 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 292, 14), 'num')
    slice_20498 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 292, 8), None, num_20497, None)
    # Getting the type of 'u' (line 292)
    u_20499 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 292, 8), 'u')
    # Obtaining the member '__getitem__' of a type (line 292)
    getitem___20500 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 292, 8), u_20499, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 292)
    subscript_call_result_20501 = invoke(stypy.reporting.localization.Localization(__file__, 292, 8), getitem___20500, (slice_20496, slice_20498))
    
    # Assigning a type to the variable 'Q' (line 292)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 292, 4), 'Q', subscript_call_result_20501)
    # Getting the type of 'Q' (line 293)
    Q_20502 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 293, 11), 'Q')
    # Assigning a type to the variable 'stypy_return_type' (line 293)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 293, 4), 'stypy_return_type', Q_20502)
    
    # ################# End of 'orth(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'orth' in the type store
    # Getting the type of 'stypy_return_type' (line 267)
    stypy_return_type_20503 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 267, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_20503)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'orth'
    return stypy_return_type_20503

# Assigning a type to the variable 'orth' (line 267)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 267, 0), 'orth', orth)

@norecursion
def subspace_angles(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'subspace_angles'
    module_type_store = module_type_store.open_function_context('subspace_angles', 296, 0, False)
    
    # Passed parameters checking function
    subspace_angles.stypy_localization = localization
    subspace_angles.stypy_type_of_self = None
    subspace_angles.stypy_type_store = module_type_store
    subspace_angles.stypy_function_name = 'subspace_angles'
    subspace_angles.stypy_param_names_list = ['A', 'B']
    subspace_angles.stypy_varargs_param_name = None
    subspace_angles.stypy_kwargs_param_name = None
    subspace_angles.stypy_call_defaults = defaults
    subspace_angles.stypy_call_varargs = varargs
    subspace_angles.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'subspace_angles', ['A', 'B'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'subspace_angles', localization, ['A', 'B'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'subspace_angles(...)' code ##################

    str_20504 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 356, (-1)), 'str', '\n    Compute the subspace angles between two matrices.\n\n    Parameters\n    ----------\n    A : (M, N) array_like\n        The first input array.\n    B : (M, K) array_like\n        The second input array.\n\n    Returns\n    -------\n    angles : ndarray, shape (min(N, K),)\n        The subspace angles between the column spaces of `A` and `B`.\n\n    See Also\n    --------\n    orth\n    svd\n\n    Notes\n    -----\n    This computes the subspace angles according to the formula\n    provided in [1]_. For equivalence with MATLAB and Octave behavior,\n    use ``angles[0]``.\n\n    .. versionadded:: 1.0\n\n    References\n    ----------\n    .. [1] Knyazev A, Argentati M (2002) Principal Angles between Subspaces\n           in an A-Based Scalar Product: Algorithms and Perturbation\n           Estimates. SIAM J. Sci. Comput. 23:2008-2040.\n\n    Examples\n    --------\n    A Hadamard matrix, which has orthogonal columns, so we expect that\n    the suspace angle to be :math:`\\frac{\\pi}{2}`:\n\n    >>> from scipy.linalg import hadamard, subspace_angles\n    >>> H = hadamard(4)\n    >>> print(H)\n    [[ 1  1  1  1]\n     [ 1 -1  1 -1]\n     [ 1  1 -1 -1]\n     [ 1 -1 -1  1]]\n    >>> np.rad2deg(subspace_angles(H[:, :2], H[:, 2:]))\n    array([ 90.,  90.])\n\n    And the subspace angle of a matrix to itself should be zero:\n\n    >>> subspace_angles(H[:, :2], H[:, :2]) <= 2 * np.finfo(float).eps\n    array([ True,  True], dtype=bool)\n\n    The angles between non-orthogonal subspaces are in between these extremes:\n\n    >>> x = np.random.RandomState(0).randn(4, 3)\n    >>> np.rad2deg(subspace_angles(x[:, :2], x[:, [2]]))\n    array([ 55.832])\n    ')
    
    # Assigning a Call to a Name (line 360):
    
    # Assigning a Call to a Name (line 360):
    
    # Call to _asarray_validated(...): (line 360)
    # Processing the call arguments (line 360)
    # Getting the type of 'A' (line 360)
    A_20506 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 360, 27), 'A', False)
    # Processing the call keyword arguments (line 360)
    # Getting the type of 'True' (line 360)
    True_20507 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 360, 43), 'True', False)
    keyword_20508 = True_20507
    kwargs_20509 = {'check_finite': keyword_20508}
    # Getting the type of '_asarray_validated' (line 360)
    _asarray_validated_20505 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 360, 8), '_asarray_validated', False)
    # Calling _asarray_validated(args, kwargs) (line 360)
    _asarray_validated_call_result_20510 = invoke(stypy.reporting.localization.Localization(__file__, 360, 8), _asarray_validated_20505, *[A_20506], **kwargs_20509)
    
    # Assigning a type to the variable 'A' (line 360)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 360, 4), 'A', _asarray_validated_call_result_20510)
    
    
    
    # Call to len(...): (line 361)
    # Processing the call arguments (line 361)
    # Getting the type of 'A' (line 361)
    A_20512 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 361, 11), 'A', False)
    # Obtaining the member 'shape' of a type (line 361)
    shape_20513 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 361, 11), A_20512, 'shape')
    # Processing the call keyword arguments (line 361)
    kwargs_20514 = {}
    # Getting the type of 'len' (line 361)
    len_20511 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 361, 7), 'len', False)
    # Calling len(args, kwargs) (line 361)
    len_call_result_20515 = invoke(stypy.reporting.localization.Localization(__file__, 361, 7), len_20511, *[shape_20513], **kwargs_20514)
    
    int_20516 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 361, 23), 'int')
    # Applying the binary operator '!=' (line 361)
    result_ne_20517 = python_operator(stypy.reporting.localization.Localization(__file__, 361, 7), '!=', len_call_result_20515, int_20516)
    
    # Testing the type of an if condition (line 361)
    if_condition_20518 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 361, 4), result_ne_20517)
    # Assigning a type to the variable 'if_condition_20518' (line 361)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 361, 4), 'if_condition_20518', if_condition_20518)
    # SSA begins for if statement (line 361)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 362)
    # Processing the call arguments (line 362)
    str_20520 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 362, 25), 'str', 'expected 2D array, got shape %s')
    
    # Obtaining an instance of the builtin type 'tuple' (line 362)
    tuple_20521 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 362, 62), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 362)
    # Adding element type (line 362)
    # Getting the type of 'A' (line 362)
    A_20522 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 362, 62), 'A', False)
    # Obtaining the member 'shape' of a type (line 362)
    shape_20523 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 362, 62), A_20522, 'shape')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 362, 62), tuple_20521, shape_20523)
    
    # Applying the binary operator '%' (line 362)
    result_mod_20524 = python_operator(stypy.reporting.localization.Localization(__file__, 362, 25), '%', str_20520, tuple_20521)
    
    # Processing the call keyword arguments (line 362)
    kwargs_20525 = {}
    # Getting the type of 'ValueError' (line 362)
    ValueError_20519 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 362, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 362)
    ValueError_call_result_20526 = invoke(stypy.reporting.localization.Localization(__file__, 362, 14), ValueError_20519, *[result_mod_20524], **kwargs_20525)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 362, 8), ValueError_call_result_20526, 'raise parameter', BaseException)
    # SSA join for if statement (line 361)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 363):
    
    # Assigning a Call to a Name (line 363):
    
    # Call to orth(...): (line 363)
    # Processing the call arguments (line 363)
    # Getting the type of 'A' (line 363)
    A_20528 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 363, 14), 'A', False)
    # Processing the call keyword arguments (line 363)
    kwargs_20529 = {}
    # Getting the type of 'orth' (line 363)
    orth_20527 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 363, 9), 'orth', False)
    # Calling orth(args, kwargs) (line 363)
    orth_call_result_20530 = invoke(stypy.reporting.localization.Localization(__file__, 363, 9), orth_20527, *[A_20528], **kwargs_20529)
    
    # Assigning a type to the variable 'QA' (line 363)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 363, 4), 'QA', orth_call_result_20530)
    # Deleting a member
    module_type_store.del_member(stypy.reporting.localization.Localization(__file__, 364, 4), module_type_store, 'A')
    
    # Assigning a Call to a Name (line 366):
    
    # Assigning a Call to a Name (line 366):
    
    # Call to _asarray_validated(...): (line 366)
    # Processing the call arguments (line 366)
    # Getting the type of 'B' (line 366)
    B_20532 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 366, 27), 'B', False)
    # Processing the call keyword arguments (line 366)
    # Getting the type of 'True' (line 366)
    True_20533 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 366, 43), 'True', False)
    keyword_20534 = True_20533
    kwargs_20535 = {'check_finite': keyword_20534}
    # Getting the type of '_asarray_validated' (line 366)
    _asarray_validated_20531 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 366, 8), '_asarray_validated', False)
    # Calling _asarray_validated(args, kwargs) (line 366)
    _asarray_validated_call_result_20536 = invoke(stypy.reporting.localization.Localization(__file__, 366, 8), _asarray_validated_20531, *[B_20532], **kwargs_20535)
    
    # Assigning a type to the variable 'B' (line 366)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 366, 4), 'B', _asarray_validated_call_result_20536)
    
    
    
    # Call to len(...): (line 367)
    # Processing the call arguments (line 367)
    # Getting the type of 'B' (line 367)
    B_20538 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 367, 11), 'B', False)
    # Obtaining the member 'shape' of a type (line 367)
    shape_20539 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 367, 11), B_20538, 'shape')
    # Processing the call keyword arguments (line 367)
    kwargs_20540 = {}
    # Getting the type of 'len' (line 367)
    len_20537 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 367, 7), 'len', False)
    # Calling len(args, kwargs) (line 367)
    len_call_result_20541 = invoke(stypy.reporting.localization.Localization(__file__, 367, 7), len_20537, *[shape_20539], **kwargs_20540)
    
    int_20542 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 367, 23), 'int')
    # Applying the binary operator '!=' (line 367)
    result_ne_20543 = python_operator(stypy.reporting.localization.Localization(__file__, 367, 7), '!=', len_call_result_20541, int_20542)
    
    # Testing the type of an if condition (line 367)
    if_condition_20544 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 367, 4), result_ne_20543)
    # Assigning a type to the variable 'if_condition_20544' (line 367)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 367, 4), 'if_condition_20544', if_condition_20544)
    # SSA begins for if statement (line 367)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 368)
    # Processing the call arguments (line 368)
    str_20546 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 368, 25), 'str', 'expected 2D array, got shape %s')
    
    # Obtaining an instance of the builtin type 'tuple' (line 368)
    tuple_20547 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 368, 62), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 368)
    # Adding element type (line 368)
    # Getting the type of 'B' (line 368)
    B_20548 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 368, 62), 'B', False)
    # Obtaining the member 'shape' of a type (line 368)
    shape_20549 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 368, 62), B_20548, 'shape')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 368, 62), tuple_20547, shape_20549)
    
    # Applying the binary operator '%' (line 368)
    result_mod_20550 = python_operator(stypy.reporting.localization.Localization(__file__, 368, 25), '%', str_20546, tuple_20547)
    
    # Processing the call keyword arguments (line 368)
    kwargs_20551 = {}
    # Getting the type of 'ValueError' (line 368)
    ValueError_20545 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 368, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 368)
    ValueError_call_result_20552 = invoke(stypy.reporting.localization.Localization(__file__, 368, 14), ValueError_20545, *[result_mod_20550], **kwargs_20551)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 368, 8), ValueError_call_result_20552, 'raise parameter', BaseException)
    # SSA join for if statement (line 367)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    
    # Call to len(...): (line 369)
    # Processing the call arguments (line 369)
    # Getting the type of 'B' (line 369)
    B_20554 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 369, 11), 'B', False)
    # Processing the call keyword arguments (line 369)
    kwargs_20555 = {}
    # Getting the type of 'len' (line 369)
    len_20553 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 369, 7), 'len', False)
    # Calling len(args, kwargs) (line 369)
    len_call_result_20556 = invoke(stypy.reporting.localization.Localization(__file__, 369, 7), len_20553, *[B_20554], **kwargs_20555)
    
    
    # Call to len(...): (line 369)
    # Processing the call arguments (line 369)
    # Getting the type of 'QA' (line 369)
    QA_20558 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 369, 21), 'QA', False)
    # Processing the call keyword arguments (line 369)
    kwargs_20559 = {}
    # Getting the type of 'len' (line 369)
    len_20557 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 369, 17), 'len', False)
    # Calling len(args, kwargs) (line 369)
    len_call_result_20560 = invoke(stypy.reporting.localization.Localization(__file__, 369, 17), len_20557, *[QA_20558], **kwargs_20559)
    
    # Applying the binary operator '!=' (line 369)
    result_ne_20561 = python_operator(stypy.reporting.localization.Localization(__file__, 369, 7), '!=', len_call_result_20556, len_call_result_20560)
    
    # Testing the type of an if condition (line 369)
    if_condition_20562 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 369, 4), result_ne_20561)
    # Assigning a type to the variable 'if_condition_20562' (line 369)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 369, 4), 'if_condition_20562', if_condition_20562)
    # SSA begins for if statement (line 369)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 370)
    # Processing the call arguments (line 370)
    str_20564 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 370, 25), 'str', 'A and B must have the same number of rows, got %s and %s')
    
    # Obtaining an instance of the builtin type 'tuple' (line 371)
    tuple_20565 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 371, 40), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 371)
    # Adding element type (line 371)
    
    # Obtaining the type of the subscript
    int_20566 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 371, 49), 'int')
    # Getting the type of 'QA' (line 371)
    QA_20567 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 371, 40), 'QA', False)
    # Obtaining the member 'shape' of a type (line 371)
    shape_20568 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 371, 40), QA_20567, 'shape')
    # Obtaining the member '__getitem__' of a type (line 371)
    getitem___20569 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 371, 40), shape_20568, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 371)
    subscript_call_result_20570 = invoke(stypy.reporting.localization.Localization(__file__, 371, 40), getitem___20569, int_20566)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 371, 40), tuple_20565, subscript_call_result_20570)
    # Adding element type (line 371)
    
    # Obtaining the type of the subscript
    int_20571 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 371, 61), 'int')
    # Getting the type of 'B' (line 371)
    B_20572 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 371, 53), 'B', False)
    # Obtaining the member 'shape' of a type (line 371)
    shape_20573 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 371, 53), B_20572, 'shape')
    # Obtaining the member '__getitem__' of a type (line 371)
    getitem___20574 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 371, 53), shape_20573, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 371)
    subscript_call_result_20575 = invoke(stypy.reporting.localization.Localization(__file__, 371, 53), getitem___20574, int_20571)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 371, 40), tuple_20565, subscript_call_result_20575)
    
    # Applying the binary operator '%' (line 370)
    result_mod_20576 = python_operator(stypy.reporting.localization.Localization(__file__, 370, 25), '%', str_20564, tuple_20565)
    
    # Processing the call keyword arguments (line 370)
    kwargs_20577 = {}
    # Getting the type of 'ValueError' (line 370)
    ValueError_20563 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 370, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 370)
    ValueError_call_result_20578 = invoke(stypy.reporting.localization.Localization(__file__, 370, 14), ValueError_20563, *[result_mod_20576], **kwargs_20577)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 370, 8), ValueError_call_result_20578, 'raise parameter', BaseException)
    # SSA join for if statement (line 369)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 372):
    
    # Assigning a Call to a Name (line 372):
    
    # Call to orth(...): (line 372)
    # Processing the call arguments (line 372)
    # Getting the type of 'B' (line 372)
    B_20580 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 372, 14), 'B', False)
    # Processing the call keyword arguments (line 372)
    kwargs_20581 = {}
    # Getting the type of 'orth' (line 372)
    orth_20579 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 372, 9), 'orth', False)
    # Calling orth(args, kwargs) (line 372)
    orth_call_result_20582 = invoke(stypy.reporting.localization.Localization(__file__, 372, 9), orth_20579, *[B_20580], **kwargs_20581)
    
    # Assigning a type to the variable 'QB' (line 372)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 372, 4), 'QB', orth_call_result_20582)
    # Deleting a member
    module_type_store.del_member(stypy.reporting.localization.Localization(__file__, 373, 4), module_type_store, 'B')
    
    # Assigning a Call to a Name (line 376):
    
    # Assigning a Call to a Name (line 376):
    
    # Call to dot(...): (line 376)
    # Processing the call arguments (line 376)
    # Getting the type of 'QA' (line 376)
    QA_20584 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 376, 18), 'QA', False)
    # Obtaining the member 'T' of a type (line 376)
    T_20585 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 376, 18), QA_20584, 'T')
    # Getting the type of 'QB' (line 376)
    QB_20586 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 376, 24), 'QB', False)
    # Processing the call keyword arguments (line 376)
    kwargs_20587 = {}
    # Getting the type of 'dot' (line 376)
    dot_20583 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 376, 14), 'dot', False)
    # Calling dot(args, kwargs) (line 376)
    dot_call_result_20588 = invoke(stypy.reporting.localization.Localization(__file__, 376, 14), dot_20583, *[T_20585, QB_20586], **kwargs_20587)
    
    # Assigning a type to the variable 'QA_T_QB' (line 376)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 376, 4), 'QA_T_QB', dot_call_result_20588)
    
    # Assigning a Call to a Name (line 377):
    
    # Assigning a Call to a Name (line 377):
    
    # Call to svdvals(...): (line 377)
    # Processing the call arguments (line 377)
    # Getting the type of 'QA_T_QB' (line 377)
    QA_T_QB_20590 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 377, 20), 'QA_T_QB', False)
    # Processing the call keyword arguments (line 377)
    kwargs_20591 = {}
    # Getting the type of 'svdvals' (line 377)
    svdvals_20589 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 377, 12), 'svdvals', False)
    # Calling svdvals(args, kwargs) (line 377)
    svdvals_call_result_20592 = invoke(stypy.reporting.localization.Localization(__file__, 377, 12), svdvals_20589, *[QA_T_QB_20590], **kwargs_20591)
    
    # Assigning a type to the variable 'sigma' (line 377)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 377, 4), 'sigma', svdvals_call_result_20592)
    
    
    
    # Obtaining the type of the subscript
    int_20593 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 380, 16), 'int')
    # Getting the type of 'QA' (line 380)
    QA_20594 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 380, 7), 'QA')
    # Obtaining the member 'shape' of a type (line 380)
    shape_20595 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 380, 7), QA_20594, 'shape')
    # Obtaining the member '__getitem__' of a type (line 380)
    getitem___20596 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 380, 7), shape_20595, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 380)
    subscript_call_result_20597 = invoke(stypy.reporting.localization.Localization(__file__, 380, 7), getitem___20596, int_20593)
    
    
    # Obtaining the type of the subscript
    int_20598 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 380, 31), 'int')
    # Getting the type of 'QB' (line 380)
    QB_20599 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 380, 22), 'QB')
    # Obtaining the member 'shape' of a type (line 380)
    shape_20600 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 380, 22), QB_20599, 'shape')
    # Obtaining the member '__getitem__' of a type (line 380)
    getitem___20601 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 380, 22), shape_20600, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 380)
    subscript_call_result_20602 = invoke(stypy.reporting.localization.Localization(__file__, 380, 22), getitem___20601, int_20598)
    
    # Applying the binary operator '>=' (line 380)
    result_ge_20603 = python_operator(stypy.reporting.localization.Localization(__file__, 380, 7), '>=', subscript_call_result_20597, subscript_call_result_20602)
    
    # Testing the type of an if condition (line 380)
    if_condition_20604 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 380, 4), result_ge_20603)
    # Assigning a type to the variable 'if_condition_20604' (line 380)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 380, 4), 'if_condition_20604', if_condition_20604)
    # SSA begins for if statement (line 380)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a BinOp to a Name (line 381):
    
    # Assigning a BinOp to a Name (line 381):
    # Getting the type of 'QB' (line 381)
    QB_20605 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 381, 12), 'QB')
    
    # Call to dot(...): (line 381)
    # Processing the call arguments (line 381)
    # Getting the type of 'QA' (line 381)
    QA_20607 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 381, 21), 'QA', False)
    # Getting the type of 'QA_T_QB' (line 381)
    QA_T_QB_20608 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 381, 25), 'QA_T_QB', False)
    # Processing the call keyword arguments (line 381)
    kwargs_20609 = {}
    # Getting the type of 'dot' (line 381)
    dot_20606 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 381, 17), 'dot', False)
    # Calling dot(args, kwargs) (line 381)
    dot_call_result_20610 = invoke(stypy.reporting.localization.Localization(__file__, 381, 17), dot_20606, *[QA_20607, QA_T_QB_20608], **kwargs_20609)
    
    # Applying the binary operator '-' (line 381)
    result_sub_20611 = python_operator(stypy.reporting.localization.Localization(__file__, 381, 12), '-', QB_20605, dot_call_result_20610)
    
    # Assigning a type to the variable 'B' (line 381)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 381, 8), 'B', result_sub_20611)
    # SSA branch for the else part of an if statement (line 380)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a BinOp to a Name (line 383):
    
    # Assigning a BinOp to a Name (line 383):
    # Getting the type of 'QA' (line 383)
    QA_20612 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 383, 12), 'QA')
    
    # Call to dot(...): (line 383)
    # Processing the call arguments (line 383)
    # Getting the type of 'QB' (line 383)
    QB_20614 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 383, 21), 'QB', False)
    # Getting the type of 'QA_T_QB' (line 383)
    QA_T_QB_20615 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 383, 25), 'QA_T_QB', False)
    # Obtaining the member 'T' of a type (line 383)
    T_20616 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 383, 25), QA_T_QB_20615, 'T')
    # Processing the call keyword arguments (line 383)
    kwargs_20617 = {}
    # Getting the type of 'dot' (line 383)
    dot_20613 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 383, 17), 'dot', False)
    # Calling dot(args, kwargs) (line 383)
    dot_call_result_20618 = invoke(stypy.reporting.localization.Localization(__file__, 383, 17), dot_20613, *[QB_20614, T_20616], **kwargs_20617)
    
    # Applying the binary operator '-' (line 383)
    result_sub_20619 = python_operator(stypy.reporting.localization.Localization(__file__, 383, 12), '-', QA_20612, dot_call_result_20618)
    
    # Assigning a type to the variable 'B' (line 383)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 383, 8), 'B', result_sub_20619)
    # SSA join for if statement (line 380)
    module_type_store = module_type_store.join_ssa_context()
    
    # Deleting a member
    module_type_store.del_member(stypy.reporting.localization.Localization(__file__, 384, 4), module_type_store, 'QA')
    # Deleting a member
    module_type_store.del_member(stypy.reporting.localization.Localization(__file__, 384, 4), module_type_store, 'QB')
    # Deleting a member
    module_type_store.del_member(stypy.reporting.localization.Localization(__file__, 384, 4), module_type_store, 'QA_T_QB')
    
    # Assigning a Compare to a Name (line 387):
    
    # Assigning a Compare to a Name (line 387):
    
    # Getting the type of 'sigma' (line 387)
    sigma_20620 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 387, 11), 'sigma')
    int_20621 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 387, 20), 'int')
    # Applying the binary operator '**' (line 387)
    result_pow_20622 = python_operator(stypy.reporting.localization.Localization(__file__, 387, 11), '**', sigma_20620, int_20621)
    
    float_20623 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 387, 25), 'float')
    # Applying the binary operator '>=' (line 387)
    result_ge_20624 = python_operator(stypy.reporting.localization.Localization(__file__, 387, 11), '>=', result_pow_20622, float_20623)
    
    # Assigning a type to the variable 'mask' (line 387)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 387, 4), 'mask', result_ge_20624)
    
    
    # Call to any(...): (line 388)
    # Processing the call keyword arguments (line 388)
    kwargs_20627 = {}
    # Getting the type of 'mask' (line 388)
    mask_20625 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 388, 7), 'mask', False)
    # Obtaining the member 'any' of a type (line 388)
    any_20626 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 388, 7), mask_20625, 'any')
    # Calling any(args, kwargs) (line 388)
    any_call_result_20628 = invoke(stypy.reporting.localization.Localization(__file__, 388, 7), any_20626, *[], **kwargs_20627)
    
    # Testing the type of an if condition (line 388)
    if_condition_20629 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 388, 4), any_call_result_20628)
    # Assigning a type to the variable 'if_condition_20629' (line 388)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 388, 4), 'if_condition_20629', if_condition_20629)
    # SSA begins for if statement (line 388)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 389):
    
    # Assigning a Call to a Name (line 389):
    
    # Call to arcsin(...): (line 389)
    # Processing the call arguments (line 389)
    
    # Call to clip(...): (line 389)
    # Processing the call arguments (line 389)
    
    # Call to svdvals(...): (line 389)
    # Processing the call arguments (line 389)
    # Getting the type of 'B' (line 389)
    B_20633 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 389, 40), 'B', False)
    # Processing the call keyword arguments (line 389)
    # Getting the type of 'True' (line 389)
    True_20634 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 389, 55), 'True', False)
    keyword_20635 = True_20634
    kwargs_20636 = {'overwrite_a': keyword_20635}
    # Getting the type of 'svdvals' (line 389)
    svdvals_20632 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 389, 32), 'svdvals', False)
    # Calling svdvals(args, kwargs) (line 389)
    svdvals_call_result_20637 = invoke(stypy.reporting.localization.Localization(__file__, 389, 32), svdvals_20632, *[B_20633], **kwargs_20636)
    
    float_20638 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 389, 62), 'float')
    float_20639 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 389, 67), 'float')
    # Processing the call keyword arguments (line 389)
    kwargs_20640 = {}
    # Getting the type of 'clip' (line 389)
    clip_20631 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 389, 27), 'clip', False)
    # Calling clip(args, kwargs) (line 389)
    clip_call_result_20641 = invoke(stypy.reporting.localization.Localization(__file__, 389, 27), clip_20631, *[svdvals_call_result_20637, float_20638, float_20639], **kwargs_20640)
    
    # Processing the call keyword arguments (line 389)
    kwargs_20642 = {}
    # Getting the type of 'arcsin' (line 389)
    arcsin_20630 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 389, 20), 'arcsin', False)
    # Calling arcsin(args, kwargs) (line 389)
    arcsin_call_result_20643 = invoke(stypy.reporting.localization.Localization(__file__, 389, 20), arcsin_20630, *[clip_call_result_20641], **kwargs_20642)
    
    # Assigning a type to the variable 'mu_arcsin' (line 389)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 389, 8), 'mu_arcsin', arcsin_call_result_20643)
    # SSA branch for the else part of an if statement (line 388)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Num to a Name (line 391):
    
    # Assigning a Num to a Name (line 391):
    float_20644 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 391, 20), 'float')
    # Assigning a type to the variable 'mu_arcsin' (line 391)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 391, 8), 'mu_arcsin', float_20644)
    # SSA join for if statement (line 388)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 394):
    
    # Assigning a Call to a Name (line 394):
    
    # Call to where(...): (line 394)
    # Processing the call arguments (line 394)
    # Getting the type of 'mask' (line 394)
    mask_20646 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 394, 18), 'mask', False)
    # Getting the type of 'mu_arcsin' (line 394)
    mu_arcsin_20647 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 394, 24), 'mu_arcsin', False)
    
    # Call to arccos(...): (line 394)
    # Processing the call arguments (line 394)
    
    # Call to clip(...): (line 394)
    # Processing the call arguments (line 394)
    # Getting the type of 'sigma' (line 394)
    sigma_20650 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 394, 47), 'sigma', False)
    float_20651 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 394, 54), 'float')
    float_20652 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 394, 59), 'float')
    # Processing the call keyword arguments (line 394)
    kwargs_20653 = {}
    # Getting the type of 'clip' (line 394)
    clip_20649 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 394, 42), 'clip', False)
    # Calling clip(args, kwargs) (line 394)
    clip_call_result_20654 = invoke(stypy.reporting.localization.Localization(__file__, 394, 42), clip_20649, *[sigma_20650, float_20651, float_20652], **kwargs_20653)
    
    # Processing the call keyword arguments (line 394)
    kwargs_20655 = {}
    # Getting the type of 'arccos' (line 394)
    arccos_20648 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 394, 35), 'arccos', False)
    # Calling arccos(args, kwargs) (line 394)
    arccos_call_result_20656 = invoke(stypy.reporting.localization.Localization(__file__, 394, 35), arccos_20648, *[clip_call_result_20654], **kwargs_20655)
    
    # Processing the call keyword arguments (line 394)
    kwargs_20657 = {}
    # Getting the type of 'where' (line 394)
    where_20645 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 394, 12), 'where', False)
    # Calling where(args, kwargs) (line 394)
    where_call_result_20658 = invoke(stypy.reporting.localization.Localization(__file__, 394, 12), where_20645, *[mask_20646, mu_arcsin_20647, arccos_call_result_20656], **kwargs_20657)
    
    # Assigning a type to the variable 'theta' (line 394)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 394, 4), 'theta', where_call_result_20658)
    # Getting the type of 'theta' (line 395)
    theta_20659 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 395, 11), 'theta')
    # Assigning a type to the variable 'stypy_return_type' (line 395)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 395, 4), 'stypy_return_type', theta_20659)
    
    # ################# End of 'subspace_angles(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'subspace_angles' in the type store
    # Getting the type of 'stypy_return_type' (line 296)
    stypy_return_type_20660 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 296, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_20660)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'subspace_angles'
    return stypy_return_type_20660

# Assigning a type to the variable 'subspace_angles' (line 296)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 296, 0), 'subspace_angles', subspace_angles)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
