
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: #
2: # Author: Pearu Peterson, March 2002
3: #
4: # w/ additions by Travis Oliphant, March 2002
5: #              and Jake Vanderplas, August 2012
6: 
7: from __future__ import division, print_function, absolute_import
8: 
9: import warnings
10: import numpy as np
11: from numpy import atleast_1d, atleast_2d
12: from .flinalg import get_flinalg_funcs
13: from .lapack import get_lapack_funcs, _compute_lwork
14: from .misc import LinAlgError, _datacopied
15: from .decomp import _asarray_validated
16: from . import decomp, decomp_svd
17: from ._solve_toeplitz import levinson
18: 
19: __all__ = ['solve', 'solve_triangular', 'solveh_banded', 'solve_banded',
20:            'solve_toeplitz', 'solve_circulant', 'inv', 'det', 'lstsq',
21:            'pinv', 'pinv2', 'pinvh', 'matrix_balance']
22: 
23: 
24: # Linear equations
25: def _solve_check(n, info, lamch=None, rcond=None):
26:     ''' Check arguments during the different steps of the solution phase '''
27:     if info < 0:
28:         raise ValueError('LAPACK reported an illegal value in {}-th argument'
29:                          '.'.format(-info))
30:     elif 0 < info:
31:         raise LinAlgError('Matrix is singular.')
32: 
33:     if lamch is None:
34:         return
35:     E = lamch('E')
36:     if rcond < E:
37:         warnings.warn('scipy.linalg.solve\nIll-conditioned matrix detected.'
38:                       ' Result is not guaranteed to be accurate.\nReciprocal '
39:                       'condition number/precision: {} / {}'.format(rcond, E),
40:                       RuntimeWarning)
41: 
42: 
43: def solve(a, b, sym_pos=False, lower=False, overwrite_a=False,
44:           overwrite_b=False, debug=None, check_finite=True, assume_a='gen',
45:           transposed=False):
46:     '''
47:     Solves the linear equation set ``a * x = b`` for the unknown ``x``
48:     for square ``a`` matrix.
49: 
50:     If the data matrix is known to be a particular type then supplying the
51:     corresponding string to ``assume_a`` key chooses the dedicated solver.
52:     The available options are
53: 
54:     ===================  ========
55:      generic matrix       'gen'
56:      symmetric            'sym'
57:      hermitian            'her'
58:      positive definite    'pos'
59:     ===================  ========
60: 
61:     If omitted, ``'gen'`` is the default structure.
62: 
63:     The datatype of the arrays define which solver is called regardless
64:     of the values. In other words, even when the complex array entries have
65:     precisely zero imaginary parts, the complex solver will be called based
66:     on the data type of the array.
67: 
68:     Parameters
69:     ----------
70:     a : (N, N) array_like
71:         Square input data
72:     b : (N, NRHS) array_like
73:         Input data for the right hand side.
74:     sym_pos : bool, optional
75:         Assume `a` is symmetric and positive definite. This key is deprecated
76:         and assume_a = 'pos' keyword is recommended instead. The functionality
77:         is the same. It will be removed in the future.
78:     lower : bool, optional
79:         If True, only the data contained in the lower triangle of `a`. Default
80:         is to use upper triangle. (ignored for ``'gen'``)
81:     overwrite_a : bool, optional
82:         Allow overwriting data in `a` (may enhance performance).
83:         Default is False.
84:     overwrite_b : bool, optional
85:         Allow overwriting data in `b` (may enhance performance).
86:         Default is False.
87:     check_finite : bool, optional
88:         Whether to check that the input matrices contain only finite numbers.
89:         Disabling may give a performance gain, but may result in problems
90:         (crashes, non-termination) if the inputs do contain infinities or NaNs.
91:     assume_a : str, optional
92:         Valid entries are explained above.
93:     transposed: bool, optional
94:         If True, ``a^T x = b`` for real matrices, raises `NotImplementedError`
95:         for complex matrices (only for True).
96: 
97:     Returns
98:     -------
99:     x : (N, NRHS) ndarray
100:         The solution array.
101: 
102:     Raises
103:     ------
104:     ValueError
105:         If size mismatches detected or input a is not square.
106:     LinAlgError
107:         If the matrix is singular.
108:     RuntimeWarning
109:         If an ill-conditioned input a is detected.
110:     NotImplementedError
111:         If transposed is True and input a is a complex matrix.
112: 
113:     Examples
114:     --------
115:     Given `a` and `b`, solve for `x`:
116: 
117:     >>> a = np.array([[3, 2, 0], [1, -1, 0], [0, 5, 1]])
118:     >>> b = np.array([2, 4, -1])
119:     >>> from scipy import linalg
120:     >>> x = linalg.solve(a, b)
121:     >>> x
122:     array([ 2., -2.,  9.])
123:     >>> np.dot(a, x) == b
124:     array([ True,  True,  True], dtype=bool)
125: 
126:     Notes
127:     -----
128:     If the input b matrix is a 1D array with N elements, when supplied
129:     together with an NxN input a, it is assumed as a valid column vector
130:     despite the apparent size mismatch. This is compatible with the
131:     numpy.dot() behavior and the returned result is still 1D array.
132: 
133:     The generic, symmetric, hermitian and positive definite solutions are
134:     obtained via calling ?GESV, ?SYSV, ?HESV, and ?POSV routines of
135:     LAPACK respectively.
136:     '''
137:     # Flags for 1D or nD right hand side
138:     b_is_1D = False
139: 
140:     a1 = atleast_2d(_asarray_validated(a, check_finite=check_finite))
141:     b1 = atleast_1d(_asarray_validated(b, check_finite=check_finite))
142:     n = a1.shape[0]
143: 
144:     overwrite_a = overwrite_a or _datacopied(a1, a)
145:     overwrite_b = overwrite_b or _datacopied(b1, b)
146: 
147:     if a1.shape[0] != a1.shape[1]:
148:         raise ValueError('Input a needs to be a square matrix.')
149: 
150:     if n != b1.shape[0]:
151:         # Last chance to catch 1x1 scalar a and 1D b arrays
152:         if not (n == 1 and b1.size != 0):
153:             raise ValueError('Input b has to have same number of rows as '
154:                              'input a')
155: 
156:     # accomodate empty arrays
157:     if b1.size == 0:
158:         return np.asfortranarray(b1.copy())
159: 
160:     # regularize 1D b arrays to 2D
161:     if b1.ndim == 1:
162:         if n == 1:
163:             b1 = b1[None, :]
164:         else:
165:             b1 = b1[:, None]
166:         b_is_1D = True
167: 
168:     # Backwards compatibility - old keyword.
169:     if sym_pos:
170:         assume_a = 'pos'
171: 
172:     if assume_a not in ('gen', 'sym', 'her', 'pos'):
173:         raise ValueError('{} is not a recognized matrix structure'
174:                          ''.format(assume_a))
175: 
176:     # Deprecate keyword "debug"
177:     if debug is not None:
178:         warnings.warn('Use of the "debug" keyword is deprecated '
179:                       'and this keyword will be removed in future '
180:                       'versions of SciPy.', DeprecationWarning)
181: 
182:     # Get the correct lamch function.
183:     # The LAMCH functions only exists for S and D
184:     # So for complex values we have to convert to real/double.
185:     if a1.dtype.char in 'fF':  # single precision
186:         lamch = get_lapack_funcs('lamch', dtype='f')
187:     else:
188:         lamch = get_lapack_funcs('lamch', dtype='d')
189: 
190:     # Currently we do not have the other forms of the norm calculators
191:     #   lansy, lanpo, lanhe.
192:     # However, in any case they only reduce computations slightly...
193:     lange = get_lapack_funcs('lange', (a1,))
194: 
195:     # Since the I-norm and 1-norm are the same for symmetric matrices
196:     # we can collect them all in this one call
197:     # Note however, that when issuing 'gen' and form!='none', then
198:     # the I-norm should be used
199:     if transposed:
200:         trans = 1
201:         norm = 'I'
202:         if np.iscomplexobj(a1):
203:             raise NotImplementedError('scipy.linalg.solve can currently '
204:                                       'not solve a^T x = b or a^H x = b '
205:                                       'for complex matrices.')
206:     else:
207:         trans = 0
208:         norm = '1'
209: 
210:     anorm = lange(norm, a1)
211: 
212:     # Generalized case 'gesv'
213:     if assume_a == 'gen':
214:         gecon, getrf, getrs = get_lapack_funcs(('gecon', 'getrf', 'getrs'),
215:                                                (a1, b1))
216:         lu, ipvt, info = getrf(a1, overwrite_a=overwrite_a)
217:         _solve_check(n, info)
218:         x, info = getrs(lu, ipvt, b1,
219:                         trans=trans, overwrite_b=overwrite_b)
220:         _solve_check(n, info)
221:         rcond, info = gecon(lu, anorm, norm=norm)
222:     # Hermitian case 'hesv'
223:     elif assume_a == 'her':
224:         hecon, hesv, hesv_lw = get_lapack_funcs(('hecon', 'hesv', 'hesv_lwork'),
225:                                                 (a1, b1))
226:         lwork = _compute_lwork(hesv_lw, n, lower)
227:         lu, ipvt, x, info = hesv(a1, b1, lwork=lwork,
228:                                  lower=lower,
229:                                  overwrite_a=overwrite_a,
230:                                  overwrite_b=overwrite_b)
231:         _solve_check(n, info)
232:         rcond, info = hecon(lu, ipvt, anorm)
233:     # Symmetric case 'sysv'
234:     elif assume_a == 'sym':
235:         sycon, sysv, sysv_lw = get_lapack_funcs(('sycon', 'sysv', 'sysv_lwork'),
236:                                                 (a1, b1))
237:         lwork = _compute_lwork(sysv_lw, n, lower)
238:         lu, ipvt, x, info = sysv(a1, b1, lwork=lwork,
239:                                  lower=lower,
240:                                  overwrite_a=overwrite_a,
241:                                  overwrite_b=overwrite_b)
242:         _solve_check(n, info)
243:         rcond, info = sycon(lu, ipvt, anorm)
244:     # Positive definite case 'posv'
245:     else:
246:         pocon, posv = get_lapack_funcs(('pocon', 'posv'),
247:                                        (a1, b1))
248:         lu, x, info = posv(a1, b1, lower=lower,
249:                            overwrite_a=overwrite_a,
250:                            overwrite_b=overwrite_b)
251:         _solve_check(n, info)
252:         rcond, info = pocon(lu, anorm)
253: 
254:     _solve_check(n, info, lamch, rcond)
255: 
256:     if b_is_1D:
257:         x = x.ravel()
258: 
259:     return x
260: 
261: 
262: def solve_triangular(a, b, trans=0, lower=False, unit_diagonal=False,
263:                      overwrite_b=False, debug=None, check_finite=True):
264:     '''
265:     Solve the equation `a x = b` for `x`, assuming a is a triangular matrix.
266: 
267:     Parameters
268:     ----------
269:     a : (M, M) array_like
270:         A triangular matrix
271:     b : (M,) or (M, N) array_like
272:         Right-hand side matrix in `a x = b`
273:     lower : bool, optional
274:         Use only data contained in the lower triangle of `a`.
275:         Default is to use upper triangle.
276:     trans : {0, 1, 2, 'N', 'T', 'C'}, optional
277:         Type of system to solve:
278: 
279:         ========  =========
280:         trans     system
281:         ========  =========
282:         0 or 'N'  a x  = b
283:         1 or 'T'  a^T x = b
284:         2 or 'C'  a^H x = b
285:         ========  =========
286:     unit_diagonal : bool, optional
287:         If True, diagonal elements of `a` are assumed to be 1 and
288:         will not be referenced.
289:     overwrite_b : bool, optional
290:         Allow overwriting data in `b` (may enhance performance)
291:     check_finite : bool, optional
292:         Whether to check that the input matrices contain only finite numbers.
293:         Disabling may give a performance gain, but may result in problems
294:         (crashes, non-termination) if the inputs do contain infinities or NaNs.
295: 
296:     Returns
297:     -------
298:     x : (M,) or (M, N) ndarray
299:         Solution to the system `a x = b`.  Shape of return matches `b`.
300: 
301:     Raises
302:     ------
303:     LinAlgError
304:         If `a` is singular
305: 
306:     Notes
307:     -----
308:     .. versionadded:: 0.9.0
309: 
310:     Examples
311:     --------
312:     Solve the lower triangular system a x = b, where::
313: 
314:              [3  0  0  0]       [4]
315:         a =  [2  1  0  0]   b = [2]
316:              [1  0  1  0]       [4]
317:              [1  1  1  1]       [2]
318: 
319:     >>> from scipy.linalg import solve_triangular
320:     >>> a = np.array([[3, 0, 0, 0], [2, 1, 0, 0], [1, 0, 1, 0], [1, 1, 1, 1]])
321:     >>> b = np.array([4, 2, 4, 2])
322:     >>> x = solve_triangular(a, b, lower=True)
323:     >>> x
324:     array([ 1.33333333, -0.66666667,  2.66666667, -1.33333333])
325:     >>> a.dot(x)  # Check the result
326:     array([ 4.,  2.,  4.,  2.])
327: 
328:     '''
329: 
330:     # Deprecate keyword "debug"
331:     if debug is not None:
332:         warnings.warn('Use of the "debug" keyword is deprecated '
333:                       'and this keyword will be removed in the future '
334:                       'versions of SciPy.', DeprecationWarning)
335: 
336:     a1 = _asarray_validated(a, check_finite=check_finite)
337:     b1 = _asarray_validated(b, check_finite=check_finite)
338:     if len(a1.shape) != 2 or a1.shape[0] != a1.shape[1]:
339:         raise ValueError('expected square matrix')
340:     if a1.shape[0] != b1.shape[0]:
341:         raise ValueError('incompatible dimensions')
342:     overwrite_b = overwrite_b or _datacopied(b1, b)
343:     if debug:
344:         print('solve:overwrite_b=', overwrite_b)
345:     trans = {'N': 0, 'T': 1, 'C': 2}.get(trans, trans)
346:     trtrs, = get_lapack_funcs(('trtrs',), (a1, b1))
347:     x, info = trtrs(a1, b1, overwrite_b=overwrite_b, lower=lower,
348:                     trans=trans, unitdiag=unit_diagonal)
349: 
350:     if info == 0:
351:         return x
352:     if info > 0:
353:         raise LinAlgError("singular matrix: resolution failed at diagonal %d" %
354:                           (info-1))
355:     raise ValueError('illegal value in %d-th argument of internal trtrs' %
356:                      (-info))
357: 
358: 
359: def solve_banded(l_and_u, ab, b, overwrite_ab=False, overwrite_b=False,
360:                  debug=None, check_finite=True):
361:     '''
362:     Solve the equation a x = b for x, assuming a is banded matrix.
363: 
364:     The matrix a is stored in `ab` using the matrix diagonal ordered form::
365: 
366:         ab[u + i - j, j] == a[i,j]
367: 
368:     Example of `ab` (shape of a is (6,6), `u` =1, `l` =2)::
369: 
370:         *    a01  a12  a23  a34  a45
371:         a00  a11  a22  a33  a44  a55
372:         a10  a21  a32  a43  a54   *
373:         a20  a31  a42  a53   *    *
374: 
375:     Parameters
376:     ----------
377:     (l, u) : (integer, integer)
378:         Number of non-zero lower and upper diagonals
379:     ab : (`l` + `u` + 1, M) array_like
380:         Banded matrix
381:     b : (M,) or (M, K) array_like
382:         Right-hand side
383:     overwrite_ab : bool, optional
384:         Discard data in `ab` (may enhance performance)
385:     overwrite_b : bool, optional
386:         Discard data in `b` (may enhance performance)
387:     check_finite : bool, optional
388:         Whether to check that the input matrices contain only finite numbers.
389:         Disabling may give a performance gain, but may result in problems
390:         (crashes, non-termination) if the inputs do contain infinities or NaNs.
391: 
392:     Returns
393:     -------
394:     x : (M,) or (M, K) ndarray
395:         The solution to the system a x = b.  Returned shape depends on the
396:         shape of `b`.
397: 
398:     Examples
399:     --------
400:     Solve the banded system a x = b, where::
401: 
402:             [5  2 -1  0  0]       [0]
403:             [1  4  2 -1  0]       [1]
404:         a = [0  1  3  2 -1]   b = [2]
405:             [0  0  1  2  2]       [2]
406:             [0  0  0  1  1]       [3]
407: 
408:     There is one nonzero diagonal below the main diagonal (l = 1), and
409:     two above (u = 2).  The diagonal banded form of the matrix is::
410: 
411:              [*  * -1 -1 -1]
412:         ab = [*  2  2  2  2]
413:              [5  4  3  2  1]
414:              [1  1  1  1  *]
415: 
416:     >>> from scipy.linalg import solve_banded
417:     >>> ab = np.array([[0,  0, -1, -1, -1],
418:     ...                [0,  2,  2,  2,  2],
419:     ...                [5,  4,  3,  2,  1],
420:     ...                [1,  1,  1,  1,  0]])
421:     >>> b = np.array([0, 1, 2, 2, 3])
422:     >>> x = solve_banded((1, 2), ab, b)
423:     >>> x
424:     array([-2.37288136,  3.93220339, -4.        ,  4.3559322 , -1.3559322 ])
425: 
426:     '''
427: 
428:     # Deprecate keyword "debug"
429:     if debug is not None:
430:         warnings.warn('Use of the "debug" keyword is deprecated '
431:                       'and this keyword will be removed in the future '
432:                       'versions of SciPy.', DeprecationWarning)
433: 
434:     a1 = _asarray_validated(ab, check_finite=check_finite, as_inexact=True)
435:     b1 = _asarray_validated(b, check_finite=check_finite, as_inexact=True)
436:     # Validate shapes.
437:     if a1.shape[-1] != b1.shape[0]:
438:         raise ValueError("shapes of ab and b are not compatible.")
439:     (l, u) = l_and_u
440:     if l + u + 1 != a1.shape[0]:
441:         raise ValueError("invalid values for the number of lower and upper "
442:                          "diagonals: l+u+1 (%d) does not equal ab.shape[0] "
443:                          "(%d)" % (l+u+1, ab.shape[0]))
444: 
445:     overwrite_b = overwrite_b or _datacopied(b1, b)
446:     if a1.shape[-1] == 1:
447:         b2 = np.array(b1, copy=(not overwrite_b))
448:         b2 /= a1[1, 0]
449:         return b2
450:     if l == u == 1:
451:         overwrite_ab = overwrite_ab or _datacopied(a1, ab)
452:         gtsv, = get_lapack_funcs(('gtsv',), (a1, b1))
453:         du = a1[0, 1:]
454:         d = a1[1, :]
455:         dl = a1[2, :-1]
456:         du2, d, du, x, info = gtsv(dl, d, du, b1, overwrite_ab, overwrite_ab,
457:                                    overwrite_ab, overwrite_b)
458:     else:
459:         gbsv, = get_lapack_funcs(('gbsv',), (a1, b1))
460:         a2 = np.zeros((2*l+u+1, a1.shape[1]), dtype=gbsv.dtype)
461:         a2[l:, :] = a1
462:         lu, piv, x, info = gbsv(l, u, a2, b1, overwrite_ab=True,
463:                                 overwrite_b=overwrite_b)
464:     if info == 0:
465:         return x
466:     if info > 0:
467:         raise LinAlgError("singular matrix")
468:     raise ValueError('illegal value in %d-th argument of internal '
469:                      'gbsv/gtsv' % -info)
470: 
471: 
472: def solveh_banded(ab, b, overwrite_ab=False, overwrite_b=False, lower=False,
473:                   check_finite=True):
474:     '''
475:     Solve equation a x = b. a is Hermitian positive-definite banded matrix.
476: 
477:     The matrix a is stored in `ab` either in lower diagonal or upper
478:     diagonal ordered form:
479: 
480:         ab[u + i - j, j] == a[i,j]        (if upper form; i <= j)
481:         ab[    i - j, j] == a[i,j]        (if lower form; i >= j)
482: 
483:     Example of `ab` (shape of a is (6, 6), `u` =2)::
484: 
485:         upper form:
486:         *   *   a02 a13 a24 a35
487:         *   a01 a12 a23 a34 a45
488:         a00 a11 a22 a33 a44 a55
489: 
490:         lower form:
491:         a00 a11 a22 a33 a44 a55
492:         a10 a21 a32 a43 a54 *
493:         a20 a31 a42 a53 *   *
494: 
495:     Cells marked with * are not used.
496: 
497:     Parameters
498:     ----------
499:     ab : (`u` + 1, M) array_like
500:         Banded matrix
501:     b : (M,) or (M, K) array_like
502:         Right-hand side
503:     overwrite_ab : bool, optional
504:         Discard data in `ab` (may enhance performance)
505:     overwrite_b : bool, optional
506:         Discard data in `b` (may enhance performance)
507:     lower : bool, optional
508:         Is the matrix in the lower form. (Default is upper form)
509:     check_finite : bool, optional
510:         Whether to check that the input matrices contain only finite numbers.
511:         Disabling may give a performance gain, but may result in problems
512:         (crashes, non-termination) if the inputs do contain infinities or NaNs.
513: 
514:     Returns
515:     -------
516:     x : (M,) or (M, K) ndarray
517:         The solution to the system a x = b.  Shape of return matches shape
518:         of `b`.
519: 
520:     Examples
521:     --------
522:     Solve the banded system A x = b, where::
523: 
524:             [ 4  2 -1  0  0  0]       [1]
525:             [ 2  5  2 -1  0  0]       [2]
526:         A = [-1  2  6  2 -1  0]   b = [2]
527:             [ 0 -1  2  7  2 -1]       [3]
528:             [ 0  0 -1  2  8  2]       [3]
529:             [ 0  0  0 -1  2  9]       [3]
530: 
531:     >>> from scipy.linalg import solveh_banded
532: 
533:     `ab` contains the main diagonal and the nonzero diagonals below the
534:     main diagonal.  That is, we use the lower form:
535: 
536:     >>> ab = np.array([[ 4,  5,  6,  7, 8, 9],
537:     ...                [ 2,  2,  2,  2, 2, 0],
538:     ...                [-1, -1, -1, -1, 0, 0]])
539:     >>> b = np.array([1, 2, 2, 3, 3, 3])
540:     >>> x = solveh_banded(ab, b, lower=True)
541:     >>> x
542:     array([ 0.03431373,  0.45938375,  0.05602241,  0.47759104,  0.17577031,
543:             0.34733894])
544: 
545: 
546:     Solve the Hermitian banded system H x = b, where::
547: 
548:             [ 8   2-1j   0     0  ]        [ 1  ]
549:         H = [2+1j  5     1j    0  ]    b = [1+1j]
550:             [ 0   -1j    9   -2-1j]        [1-2j]
551:             [ 0    0   -2+1j   6  ]        [ 0  ]
552: 
553:     In this example, we put the upper diagonals in the array `hb`:
554: 
555:     >>> hb = np.array([[0, 2-1j, 1j, -2-1j],
556:     ...                [8,  5,    9,   6  ]])
557:     >>> b = np.array([1, 1+1j, 1-2j, 0])
558:     >>> x = solveh_banded(hb, b)
559:     >>> x
560:     array([ 0.07318536-0.02939412j,  0.11877624+0.17696461j,
561:             0.10077984-0.23035393j, -0.00479904-0.09358128j])
562: 
563:     '''
564:     a1 = _asarray_validated(ab, check_finite=check_finite)
565:     b1 = _asarray_validated(b, check_finite=check_finite)
566:     # Validate shapes.
567:     if a1.shape[-1] != b1.shape[0]:
568:         raise ValueError("shapes of ab and b are not compatible.")
569: 
570:     overwrite_b = overwrite_b or _datacopied(b1, b)
571:     overwrite_ab = overwrite_ab or _datacopied(a1, ab)
572: 
573:     if a1.shape[0] == 2:
574:         ptsv, = get_lapack_funcs(('ptsv',), (a1, b1))
575:         if lower:
576:             d = a1[0, :].real
577:             e = a1[1, :-1]
578:         else:
579:             d = a1[1, :].real
580:             e = a1[0, 1:].conj()
581:         d, du, x, info = ptsv(d, e, b1, overwrite_ab, overwrite_ab,
582:                               overwrite_b)
583:     else:
584:         pbsv, = get_lapack_funcs(('pbsv',), (a1, b1))
585:         c, x, info = pbsv(a1, b1, lower=lower, overwrite_ab=overwrite_ab,
586:                           overwrite_b=overwrite_b)
587:     if info > 0:
588:         raise LinAlgError("%d-th leading minor not positive definite" % info)
589:     if info < 0:
590:         raise ValueError('illegal value in %d-th argument of internal '
591:                          'pbsv' % -info)
592:     return x
593: 
594: 
595: def solve_toeplitz(c_or_cr, b, check_finite=True):
596:     '''Solve a Toeplitz system using Levinson Recursion
597: 
598:     The Toeplitz matrix has constant diagonals, with c as its first column
599:     and r as its first row.  If r is not given, ``r == conjugate(c)`` is
600:     assumed.
601: 
602:     Parameters
603:     ----------
604:     c_or_cr : array_like or tuple of (array_like, array_like)
605:         The vector ``c``, or a tuple of arrays (``c``, ``r``). Whatever the
606:         actual shape of ``c``, it will be converted to a 1-D array. If not
607:         supplied, ``r = conjugate(c)`` is assumed; in this case, if c[0] is
608:         real, the Toeplitz matrix is Hermitian. r[0] is ignored; the first row
609:         of the Toeplitz matrix is ``[c[0], r[1:]]``.  Whatever the actual shape
610:         of ``r``, it will be converted to a 1-D array.
611:     b : (M,) or (M, K) array_like
612:         Right-hand side in ``T x = b``.
613:     check_finite : bool, optional
614:         Whether to check that the input matrices contain only finite numbers.
615:         Disabling may give a performance gain, but may result in problems
616:         (result entirely NaNs) if the inputs do contain infinities or NaNs.
617: 
618:     Returns
619:     -------
620:     x : (M,) or (M, K) ndarray
621:         The solution to the system ``T x = b``.  Shape of return matches shape
622:         of `b`.
623: 
624:     See Also
625:     --------
626:     toeplitz : Toeplitz matrix
627: 
628:     Notes
629:     -----
630:     The solution is computed using Levinson-Durbin recursion, which is faster
631:     than generic least-squares methods, but can be less numerically stable.
632: 
633:     Examples
634:     --------
635:     Solve the Toeplitz system T x = b, where::
636: 
637:             [ 1 -1 -2 -3]       [1]
638:         T = [ 3  1 -1 -2]   b = [2]
639:             [ 6  3  1 -1]       [2]
640:             [10  6  3  1]       [5]
641: 
642:     To specify the Toeplitz matrix, only the first column and the first
643:     row are needed.
644: 
645:     >>> c = np.array([1, 3, 6, 10])    # First column of T
646:     >>> r = np.array([1, -1, -2, -3])  # First row of T
647:     >>> b = np.array([1, 2, 2, 5])
648: 
649:     >>> from scipy.linalg import solve_toeplitz, toeplitz
650:     >>> x = solve_toeplitz((c, r), b)
651:     >>> x
652:     array([ 1.66666667, -1.        , -2.66666667,  2.33333333])
653: 
654:     Check the result by creating the full Toeplitz matrix and
655:     multiplying it by `x`.  We should get `b`.
656: 
657:     >>> T = toeplitz(c, r)
658:     >>> T.dot(x)
659:     array([ 1.,  2.,  2.,  5.])
660: 
661:     '''
662:     # If numerical stability of this algorithm is a problem, a future
663:     # developer might consider implementing other O(N^2) Toeplitz solvers,
664:     # such as GKO (http://www.jstor.org/stable/2153371) or Bareiss.
665:     if isinstance(c_or_cr, tuple):
666:         c, r = c_or_cr
667:         c = _asarray_validated(c, check_finite=check_finite).ravel()
668:         r = _asarray_validated(r, check_finite=check_finite).ravel()
669:     else:
670:         c = _asarray_validated(c_or_cr, check_finite=check_finite).ravel()
671:         r = c.conjugate()
672: 
673:     # Form a 1D array of values to be used in the matrix, containing a reversed
674:     # copy of r[1:], followed by c.
675:     vals = np.concatenate((r[-1:0:-1], c))
676:     if b is None:
677:         raise ValueError('illegal value, `b` is a required argument')
678: 
679:     b = _asarray_validated(b)
680:     if vals.shape[0] != (2*b.shape[0] - 1):
681:         raise ValueError('incompatible dimensions')
682:     if np.iscomplexobj(vals) or np.iscomplexobj(b):
683:         vals = np.asarray(vals, dtype=np.complex128, order='c')
684:         b = np.asarray(b, dtype=np.complex128)
685:     else:
686:         vals = np.asarray(vals, dtype=np.double, order='c')
687:         b = np.asarray(b, dtype=np.double)
688: 
689:     if b.ndim == 1:
690:         x, _ = levinson(vals, np.ascontiguousarray(b))
691:     else:
692:         b_shape = b.shape
693:         b = b.reshape(b.shape[0], -1)
694:         x = np.column_stack(
695:             (levinson(vals, np.ascontiguousarray(b[:, i]))[0])
696:             for i in range(b.shape[1]))
697:         x = x.reshape(*b_shape)
698: 
699:     return x
700: 
701: 
702: def _get_axis_len(aname, a, axis):
703:     ax = axis
704:     if ax < 0:
705:         ax += a.ndim
706:     if 0 <= ax < a.ndim:
707:         return a.shape[ax]
708:     raise ValueError("'%saxis' entry is out of bounds" % (aname,))
709: 
710: 
711: def solve_circulant(c, b, singular='raise', tol=None,
712:                     caxis=-1, baxis=0, outaxis=0):
713:     '''Solve C x = b for x, where C is a circulant matrix.
714: 
715:     `C` is the circulant matrix associated with the vector `c`.
716: 
717:     The system is solved by doing division in Fourier space.  The
718:     calculation is::
719: 
720:         x = ifft(fft(b) / fft(c))
721: 
722:     where `fft` and `ifft` are the fast Fourier transform and its inverse,
723:     respectively.  For a large vector `c`, this is *much* faster than
724:     solving the system with the full circulant matrix.
725: 
726:     Parameters
727:     ----------
728:     c : array_like
729:         The coefficients of the circulant matrix.
730:     b : array_like
731:         Right-hand side matrix in ``a x = b``.
732:     singular : str, optional
733:         This argument controls how a near singular circulant matrix is
734:         handled.  If `singular` is "raise" and the circulant matrix is
735:         near singular, a `LinAlgError` is raised.  If `singular` is
736:         "lstsq", the least squares solution is returned.  Default is "raise".
737:     tol : float, optional
738:         If any eigenvalue of the circulant matrix has an absolute value
739:         that is less than or equal to `tol`, the matrix is considered to be
740:         near singular.  If not given, `tol` is set to::
741: 
742:             tol = abs_eigs.max() * abs_eigs.size * np.finfo(np.float64).eps
743: 
744:         where `abs_eigs` is the array of absolute values of the eigenvalues
745:         of the circulant matrix.
746:     caxis : int
747:         When `c` has dimension greater than 1, it is viewed as a collection
748:         of circulant vectors.  In this case, `caxis` is the axis of `c` that
749:         holds the vectors of circulant coefficients.
750:     baxis : int
751:         When `b` has dimension greater than 1, it is viewed as a collection
752:         of vectors.  In this case, `baxis` is the axis of `b` that holds the
753:         right-hand side vectors.
754:     outaxis : int
755:         When `c` or `b` are multidimensional, the value returned by
756:         `solve_circulant` is multidimensional.  In this case, `outaxis` is
757:         the axis of the result that holds the solution vectors.
758: 
759:     Returns
760:     -------
761:     x : ndarray
762:         Solution to the system ``C x = b``.
763: 
764:     Raises
765:     ------
766:     LinAlgError
767:         If the circulant matrix associated with `c` is near singular.
768: 
769:     See Also
770:     --------
771:     circulant : circulant matrix
772: 
773:     Notes
774:     -----
775:     For a one-dimensional vector `c` with length `m`, and an array `b`
776:     with shape ``(m, ...)``,
777: 
778:         solve_circulant(c, b)
779: 
780:     returns the same result as
781: 
782:         solve(circulant(c), b)
783: 
784:     where `solve` and `circulant` are from `scipy.linalg`.
785: 
786:     .. versionadded:: 0.16.0
787: 
788:     Examples
789:     --------
790:     >>> from scipy.linalg import solve_circulant, solve, circulant, lstsq
791: 
792:     >>> c = np.array([2, 2, 4])
793:     >>> b = np.array([1, 2, 3])
794:     >>> solve_circulant(c, b)
795:     array([ 0.75, -0.25,  0.25])
796: 
797:     Compare that result to solving the system with `scipy.linalg.solve`:
798: 
799:     >>> solve(circulant(c), b)
800:     array([ 0.75, -0.25,  0.25])
801: 
802:     A singular example:
803: 
804:     >>> c = np.array([1, 1, 0, 0])
805:     >>> b = np.array([1, 2, 3, 4])
806: 
807:     Calling ``solve_circulant(c, b)`` will raise a `LinAlgError`.  For the
808:     least square solution, use the option ``singular='lstsq'``:
809: 
810:     >>> solve_circulant(c, b, singular='lstsq')
811:     array([ 0.25,  1.25,  2.25,  1.25])
812: 
813:     Compare to `scipy.linalg.lstsq`:
814: 
815:     >>> x, resid, rnk, s = lstsq(circulant(c), b)
816:     >>> x
817:     array([ 0.25,  1.25,  2.25,  1.25])
818: 
819:     A broadcasting example:
820: 
821:     Suppose we have the vectors of two circulant matrices stored in an array
822:     with shape (2, 5), and three `b` vectors stored in an array with shape
823:     (3, 5).  For example,
824: 
825:     >>> c = np.array([[1.5, 2, 3, 0, 0], [1, 1, 4, 3, 2]])
826:     >>> b = np.arange(15).reshape(-1, 5)
827: 
828:     We want to solve all combinations of circulant matrices and `b` vectors,
829:     with the result stored in an array with shape (2, 3, 5).  When we
830:     disregard the axes of `c` and `b` that hold the vectors of coefficients,
831:     the shapes of the collections are (2,) and (3,), respectively, which are
832:     not compatible for broadcasting.  To have a broadcast result with shape
833:     (2, 3), we add a trivial dimension to `c`: ``c[:, np.newaxis, :]`` has
834:     shape (2, 1, 5).  The last dimension holds the coefficients of the
835:     circulant matrices, so when we call `solve_circulant`, we can use the
836:     default ``caxis=-1``.  The coefficients of the `b` vectors are in the last
837:     dimension of the array `b`, so we use ``baxis=-1``.  If we use the
838:     default `outaxis`, the result will have shape (5, 2, 3), so we'll use
839:     ``outaxis=-1`` to put the solution vectors in the last dimension.
840: 
841:     >>> x = solve_circulant(c[:, np.newaxis, :], b, baxis=-1, outaxis=-1)
842:     >>> x.shape
843:     (2, 3, 5)
844:     >>> np.set_printoptions(precision=3)  # For compact output of numbers.
845:     >>> x
846:     array([[[-0.118,  0.22 ,  1.277, -0.142,  0.302],
847:             [ 0.651,  0.989,  2.046,  0.627,  1.072],
848:             [ 1.42 ,  1.758,  2.816,  1.396,  1.841]],
849:            [[ 0.401,  0.304,  0.694, -0.867,  0.377],
850:             [ 0.856,  0.758,  1.149, -0.412,  0.831],
851:             [ 1.31 ,  1.213,  1.603,  0.042,  1.286]]])
852: 
853:     Check by solving one pair of `c` and `b` vectors (cf. ``x[1, 1, :]``):
854: 
855:     >>> solve_circulant(c[1], b[1, :])
856:     array([ 0.856,  0.758,  1.149, -0.412,  0.831])
857: 
858:     '''
859:     c = np.atleast_1d(c)
860:     nc = _get_axis_len("c", c, caxis)
861:     b = np.atleast_1d(b)
862:     nb = _get_axis_len("b", b, baxis)
863:     if nc != nb:
864:         raise ValueError('Incompatible c and b axis lengths')
865: 
866:     fc = np.fft.fft(np.rollaxis(c, caxis, c.ndim), axis=-1)
867:     abs_fc = np.abs(fc)
868:     if tol is None:
869:         # This is the same tolerance as used in np.linalg.matrix_rank.
870:         tol = abs_fc.max(axis=-1) * nc * np.finfo(np.float64).eps
871:         if tol.shape != ():
872:             tol.shape = tol.shape + (1,)
873:         else:
874:             tol = np.atleast_1d(tol)
875: 
876:     near_zeros = abs_fc <= tol
877:     is_near_singular = np.any(near_zeros)
878:     if is_near_singular:
879:         if singular == 'raise':
880:             raise LinAlgError("near singular circulant matrix.")
881:         else:
882:             # Replace the small values with 1 to avoid errors in the
883:             # division fb/fc below.
884:             fc[near_zeros] = 1
885: 
886:     fb = np.fft.fft(np.rollaxis(b, baxis, b.ndim), axis=-1)
887: 
888:     q = fb / fc
889: 
890:     if is_near_singular:
891:         # `near_zeros` is a boolean array, same shape as `c`, that is
892:         # True where `fc` is (near) zero.  `q` is the broadcasted result
893:         # of fb / fc, so to set the values of `q` to 0 where `fc` is near
894:         # zero, we use a mask that is the broadcast result of an array
895:         # of True values shaped like `b` with `near_zeros`.
896:         mask = np.ones_like(b, dtype=bool) & near_zeros
897:         q[mask] = 0
898: 
899:     x = np.fft.ifft(q, axis=-1)
900:     if not (np.iscomplexobj(c) or np.iscomplexobj(b)):
901:         x = x.real
902:     if outaxis != -1:
903:         x = np.rollaxis(x, -1, outaxis)
904:     return x
905: 
906: 
907: # matrix inversion
908: def inv(a, overwrite_a=False, check_finite=True):
909:     '''
910:     Compute the inverse of a matrix.
911: 
912:     Parameters
913:     ----------
914:     a : array_like
915:         Square matrix to be inverted.
916:     overwrite_a : bool, optional
917:         Discard data in `a` (may improve performance). Default is False.
918:     check_finite : bool, optional
919:         Whether to check that the input matrix contains only finite numbers.
920:         Disabling may give a performance gain, but may result in problems
921:         (crashes, non-termination) if the inputs do contain infinities or NaNs.
922: 
923:     Returns
924:     -------
925:     ainv : ndarray
926:         Inverse of the matrix `a`.
927: 
928:     Raises
929:     ------
930:     LinAlgError
931:         If `a` is singular.
932:     ValueError
933:         If `a` is not square, or not 2-dimensional.
934: 
935:     Examples
936:     --------
937:     >>> from scipy import linalg
938:     >>> a = np.array([[1., 2.], [3., 4.]])
939:     >>> linalg.inv(a)
940:     array([[-2. ,  1. ],
941:            [ 1.5, -0.5]])
942:     >>> np.dot(a, linalg.inv(a))
943:     array([[ 1.,  0.],
944:            [ 0.,  1.]])
945: 
946:     '''
947:     a1 = _asarray_validated(a, check_finite=check_finite)
948:     if len(a1.shape) != 2 or a1.shape[0] != a1.shape[1]:
949:         raise ValueError('expected square matrix')
950:     overwrite_a = overwrite_a or _datacopied(a1, a)
951:     #XXX: I found no advantage or disadvantage of using finv.
952: #     finv, = get_flinalg_funcs(('inv',),(a1,))
953: #     if finv is not None:
954: #         a_inv,info = finv(a1,overwrite_a=overwrite_a)
955: #         if info==0:
956: #             return a_inv
957: #         if info>0: raise LinAlgError, "singular matrix"
958: #         if info<0: raise ValueError('illegal value in %d-th argument of '
959: #                                     'internal inv.getrf|getri'%(-info))
960:     getrf, getri, getri_lwork = get_lapack_funcs(('getrf', 'getri',
961:                                                   'getri_lwork'),
962:                                                  (a1,))
963:     lu, piv, info = getrf(a1, overwrite_a=overwrite_a)
964:     if info == 0:
965:         lwork = _compute_lwork(getri_lwork, a1.shape[0])
966: 
967:         # XXX: the following line fixes curious SEGFAULT when
968:         # benchmarking 500x500 matrix inverse. This seems to
969:         # be a bug in LAPACK ?getri routine because if lwork is
970:         # minimal (when using lwork[0] instead of lwork[1]) then
971:         # all tests pass. Further investigation is required if
972:         # more such SEGFAULTs occur.
973:         lwork = int(1.01 * lwork)
974:         inv_a, info = getri(lu, piv, lwork=lwork, overwrite_lu=1)
975:     if info > 0:
976:         raise LinAlgError("singular matrix")
977:     if info < 0:
978:         raise ValueError('illegal value in %d-th argument of internal '
979:                          'getrf|getri' % -info)
980:     return inv_a
981: 
982: 
983: # Determinant
984: 
985: def det(a, overwrite_a=False, check_finite=True):
986:     '''
987:     Compute the determinant of a matrix
988: 
989:     The determinant of a square matrix is a value derived arithmetically
990:     from the coefficients of the matrix.
991: 
992:     The determinant for a 3x3 matrix, for example, is computed as follows::
993: 
994:         a    b    c
995:         d    e    f = A
996:         g    h    i
997: 
998:         det(A) = a*e*i + b*f*g + c*d*h - c*e*g - b*d*i - a*f*h
999: 
1000:     Parameters
1001:     ----------
1002:     a : (M, M) array_like
1003:         A square matrix.
1004:     overwrite_a : bool, optional
1005:         Allow overwriting data in a (may enhance performance).
1006:     check_finite : bool, optional
1007:         Whether to check that the input matrix contains only finite numbers.
1008:         Disabling may give a performance gain, but may result in problems
1009:         (crashes, non-termination) if the inputs do contain infinities or NaNs.
1010: 
1011:     Returns
1012:     -------
1013:     det : float or complex
1014:         Determinant of `a`.
1015: 
1016:     Notes
1017:     -----
1018:     The determinant is computed via LU factorization, LAPACK routine z/dgetrf.
1019: 
1020:     Examples
1021:     --------
1022:     >>> from scipy import linalg
1023:     >>> a = np.array([[1,2,3], [4,5,6], [7,8,9]])
1024:     >>> linalg.det(a)
1025:     0.0
1026:     >>> a = np.array([[0,2,3], [4,5,6], [7,8,9]])
1027:     >>> linalg.det(a)
1028:     3.0
1029: 
1030:     '''
1031:     a1 = _asarray_validated(a, check_finite=check_finite)
1032:     if len(a1.shape) != 2 or a1.shape[0] != a1.shape[1]:
1033:         raise ValueError('expected square matrix')
1034:     overwrite_a = overwrite_a or _datacopied(a1, a)
1035:     fdet, = get_flinalg_funcs(('det',), (a1,))
1036:     a_det, info = fdet(a1, overwrite_a=overwrite_a)
1037:     if info < 0:
1038:         raise ValueError('illegal value in %d-th argument of internal '
1039:                          'det.getrf' % -info)
1040:     return a_det
1041: 
1042: # Linear Least Squares
1043: 
1044: 
1045: class LstsqLapackError(LinAlgError):
1046:     pass
1047: 
1048: 
1049: def lstsq(a, b, cond=None, overwrite_a=False, overwrite_b=False,
1050:           check_finite=True, lapack_driver=None):
1051:     '''
1052:     Compute least-squares solution to equation Ax = b.
1053: 
1054:     Compute a vector x such that the 2-norm ``|b - A x|`` is minimized.
1055: 
1056:     Parameters
1057:     ----------
1058:     a : (M, N) array_like
1059:         Left hand side matrix (2-D array).
1060:     b : (M,) or (M, K) array_like
1061:         Right hand side matrix or vector (1-D or 2-D array).
1062:     cond : float, optional
1063:         Cutoff for 'small' singular values; used to determine effective
1064:         rank of a. Singular values smaller than
1065:         ``rcond * largest_singular_value`` are considered zero.
1066:     overwrite_a : bool, optional
1067:         Discard data in `a` (may enhance performance). Default is False.
1068:     overwrite_b : bool, optional
1069:         Discard data in `b` (may enhance performance). Default is False.
1070:     check_finite : bool, optional
1071:         Whether to check that the input matrices contain only finite numbers.
1072:         Disabling may give a performance gain, but may result in problems
1073:         (crashes, non-termination) if the inputs do contain infinities or NaNs.
1074:     lapack_driver : str, optional
1075:         Which LAPACK driver is used to solve the least-squares problem.
1076:         Options are ``'gelsd'``, ``'gelsy'``, ``'gelss'``. Default
1077:         (``'gelsd'``) is a good choice.  However, ``'gelsy'`` can be slightly
1078:         faster on many problems.  ``'gelss'`` was used historically.  It is
1079:         generally slow but uses less memory.
1080: 
1081:         .. versionadded:: 0.17.0
1082: 
1083:     Returns
1084:     -------
1085:     x : (N,) or (N, K) ndarray
1086:         Least-squares solution.  Return shape matches shape of `b`.
1087:     residues : (0,) or () or (K,) ndarray
1088:         Sums of residues, squared 2-norm for each column in ``b - a x``.
1089:         If rank of matrix a is ``< N`` or ``N > M``, or ``'gelsy'`` is used,
1090:         this is a lenght zero array. If b was 1-D, this is a () shape array
1091:         (numpy scalar), otherwise the shape is (K,).
1092:     rank : int
1093:         Effective rank of matrix `a`.
1094:     s : (min(M,N),) ndarray or None
1095:         Singular values of `a`. The condition number of a is
1096:         ``abs(s[0] / s[-1])``. None is returned when ``'gelsy'`` is used.
1097: 
1098:     Raises
1099:     ------
1100:     LinAlgError
1101:         If computation does not converge.
1102: 
1103:     ValueError
1104:         When parameters are wrong.
1105: 
1106:     See Also
1107:     --------
1108:     optimize.nnls : linear least squares with non-negativity constraint
1109: 
1110:     Examples
1111:     --------
1112:     >>> from scipy.linalg import lstsq
1113:     >>> import matplotlib.pyplot as plt
1114: 
1115:     Suppose we have the following data:
1116: 
1117:     >>> x = np.array([1, 2.5, 3.5, 4, 5, 7, 8.5])
1118:     >>> y = np.array([0.3, 1.1, 1.5, 2.0, 3.2, 6.6, 8.6])
1119: 
1120:     We want to fit a quadratic polynomial of the form ``y = a + b*x**2``
1121:     to this data.  We first form the "design matrix" M, with a constant
1122:     column of 1s and a column containing ``x**2``:
1123: 
1124:     >>> M = x[:, np.newaxis]**[0, 2]
1125:     >>> M
1126:     array([[  1.  ,   1.  ],
1127:            [  1.  ,   6.25],
1128:            [  1.  ,  12.25],
1129:            [  1.  ,  16.  ],
1130:            [  1.  ,  25.  ],
1131:            [  1.  ,  49.  ],
1132:            [  1.  ,  72.25]])
1133: 
1134:     We want to find the least-squares solution to ``M.dot(p) = y``,
1135:     where ``p`` is a vector with length 2 that holds the parameters
1136:     ``a`` and ``b``.
1137: 
1138:     >>> p, res, rnk, s = lstsq(M, y)
1139:     >>> p
1140:     array([ 0.20925829,  0.12013861])
1141: 
1142:     Plot the data and the fitted curve.
1143: 
1144:     >>> plt.plot(x, y, 'o', label='data')
1145:     >>> xx = np.linspace(0, 9, 101)
1146:     >>> yy = p[0] + p[1]*xx**2
1147:     >>> plt.plot(xx, yy, label='least squares fit, $y = a + bx^2$')
1148:     >>> plt.xlabel('x')
1149:     >>> plt.ylabel('y')
1150:     >>> plt.legend(framealpha=1, shadow=True)
1151:     >>> plt.grid(alpha=0.25)
1152:     >>> plt.show()
1153: 
1154:     '''
1155:     a1 = _asarray_validated(a, check_finite=check_finite)
1156:     b1 = _asarray_validated(b, check_finite=check_finite)
1157:     if len(a1.shape) != 2:
1158:         raise ValueError('expected matrix')
1159:     m, n = a1.shape
1160:     if len(b1.shape) == 2:
1161:         nrhs = b1.shape[1]
1162:     else:
1163:         nrhs = 1
1164:     if m != b1.shape[0]:
1165:         raise ValueError('incompatible dimensions')
1166:     if m == 0 or n == 0:  # Zero-sized problem, confuses LAPACK
1167:         x = np.zeros((n,) + b1.shape[1:], dtype=np.common_type(a1, b1))
1168:         if n == 0:
1169:             residues = np.linalg.norm(b1, axis=0)**2
1170:         else:
1171:             residues = np.empty((0,))
1172:         return x, residues, 0, np.empty((0,))
1173: 
1174:     driver = lapack_driver
1175:     if driver is None:
1176:         driver = lstsq.default_lapack_driver
1177:     if driver not in ('gelsd', 'gelsy', 'gelss'):
1178:         raise ValueError('LAPACK driver "%s" is not found' % driver)
1179: 
1180:     lapack_func, lapack_lwork = get_lapack_funcs((driver,
1181:                                                  '%s_lwork' % driver),
1182:                                                  (a1, b1))
1183:     real_data = True if (lapack_func.dtype.kind == 'f') else False
1184: 
1185:     if m < n:
1186:         # need to extend b matrix as it will be filled with
1187:         # a larger solution matrix
1188:         if len(b1.shape) == 2:
1189:             b2 = np.zeros((n, nrhs), dtype=lapack_func.dtype)
1190:             b2[:m, :] = b1
1191:         else:
1192:             b2 = np.zeros(n, dtype=lapack_func.dtype)
1193:             b2[:m] = b1
1194:         b1 = b2
1195: 
1196:     overwrite_a = overwrite_a or _datacopied(a1, a)
1197:     overwrite_b = overwrite_b or _datacopied(b1, b)
1198: 
1199:     if cond is None:
1200:         cond = np.finfo(lapack_func.dtype).eps
1201: 
1202:     if driver in ('gelss', 'gelsd'):
1203:         if driver == 'gelss':
1204:             lwork = _compute_lwork(lapack_lwork, m, n, nrhs, cond)
1205:             v, x, s, rank, work, info = lapack_func(a1, b1, cond, lwork,
1206:                                                     overwrite_a=overwrite_a,
1207:                                                     overwrite_b=overwrite_b)
1208: 
1209:         elif driver == 'gelsd':
1210:             if real_data:
1211:                 lwork, iwork = _compute_lwork(lapack_lwork, m, n, nrhs, cond)
1212:                 if iwork == 0:
1213:                     # this is LAPACK bug 0038: dgelsd does not provide the
1214:                     # size of the iwork array in query mode.  This bug was
1215:                     # fixed in LAPACK 3.2.2, released July 21, 2010.
1216:                     mesg = ("internal gelsd driver lwork query error, "
1217:                             "required iwork dimension not returned. "
1218:                             "This is likely the result of LAPACK bug "
1219:                             "0038, fixed in LAPACK 3.2.2 (released "
1220:                             "July 21, 2010). ")
1221: 
1222:                     if lapack_driver is None:
1223:                         # restart with gelss
1224:                         lstsq.default_lapack_driver = 'gelss'
1225:                         mesg += "Falling back to 'gelss' driver."
1226:                         warnings.warn(mesg, RuntimeWarning)
1227:                         return lstsq(a, b, cond, overwrite_a, overwrite_b,
1228:                                      check_finite, lapack_driver='gelss')
1229: 
1230:                     # can't proceed, bail out
1231:                     mesg += ("Use a different lapack_driver when calling lstsq"
1232:                              " or upgrade LAPACK.")
1233:                     raise LstsqLapackError(mesg)
1234: 
1235:                 x, s, rank, info = lapack_func(a1, b1, lwork,
1236:                                                iwork, cond, False, False)
1237:             else:  # complex data
1238:                 lwork, rwork, iwork = _compute_lwork(lapack_lwork, m, n,
1239:                                                      nrhs, cond)
1240:                 x, s, rank, info = lapack_func(a1, b1, lwork, rwork, iwork,
1241:                                                cond, False, False)
1242:         if info > 0:
1243:             raise LinAlgError("SVD did not converge in Linear Least Squares")
1244:         if info < 0:
1245:             raise ValueError('illegal value in %d-th argument of internal %s'
1246:                              % (-info, lapack_driver))
1247:         resids = np.asarray([], dtype=x.dtype)
1248:         if m > n:
1249:             x1 = x[:n]
1250:             if rank == n:
1251:                 resids = np.sum(np.abs(x[n:])**2, axis=0)
1252:             x = x1
1253:         return x, resids, rank, s
1254: 
1255:     elif driver == 'gelsy':
1256:         lwork = _compute_lwork(lapack_lwork, m, n, nrhs, cond)
1257:         jptv = np.zeros((a1.shape[1], 1), dtype=np.int32)
1258:         v, x, j, rank, info = lapack_func(a1, b1, jptv, cond,
1259:                                           lwork, False, False)
1260:         if info < 0:
1261:             raise ValueError("illegal value in %d-th argument of internal "
1262:                              "gelsy" % -info)
1263:         if m > n:
1264:             x1 = x[:n]
1265:             x = x1
1266:         return x, np.array([], x.dtype), rank, None
1267: lstsq.default_lapack_driver = 'gelsd'
1268: 
1269: 
1270: def pinv(a, cond=None, rcond=None, return_rank=False, check_finite=True):
1271:     '''
1272:     Compute the (Moore-Penrose) pseudo-inverse of a matrix.
1273: 
1274:     Calculate a generalized inverse of a matrix using a least-squares
1275:     solver.
1276: 
1277:     Parameters
1278:     ----------
1279:     a : (M, N) array_like
1280:         Matrix to be pseudo-inverted.
1281:     cond, rcond : float, optional
1282:         Cutoff for 'small' singular values in the least-squares solver.
1283:         Singular values smaller than ``rcond * largest_singular_value``
1284:         are considered zero.
1285:     return_rank : bool, optional
1286:         if True, return the effective rank of the matrix
1287:     check_finite : bool, optional
1288:         Whether to check that the input matrix contains only finite numbers.
1289:         Disabling may give a performance gain, but may result in problems
1290:         (crashes, non-termination) if the inputs do contain infinities or NaNs.
1291: 
1292:     Returns
1293:     -------
1294:     B : (N, M) ndarray
1295:         The pseudo-inverse of matrix `a`.
1296:     rank : int
1297:         The effective rank of the matrix.  Returned if return_rank == True
1298: 
1299:     Raises
1300:     ------
1301:     LinAlgError
1302:         If computation does not converge.
1303: 
1304:     Examples
1305:     --------
1306:     >>> from scipy import linalg
1307:     >>> a = np.random.randn(9, 6)
1308:     >>> B = linalg.pinv(a)
1309:     >>> np.allclose(a, np.dot(a, np.dot(B, a)))
1310:     True
1311:     >>> np.allclose(B, np.dot(B, np.dot(a, B)))
1312:     True
1313: 
1314:     '''
1315:     a = _asarray_validated(a, check_finite=check_finite)
1316:     b = np.identity(a.shape[0], dtype=a.dtype)
1317:     if rcond is not None:
1318:         cond = rcond
1319: 
1320:     x, resids, rank, s = lstsq(a, b, cond=cond, check_finite=False)
1321: 
1322:     if return_rank:
1323:         return x, rank
1324:     else:
1325:         return x
1326: 
1327: 
1328: def pinv2(a, cond=None, rcond=None, return_rank=False, check_finite=True):
1329:     '''
1330:     Compute the (Moore-Penrose) pseudo-inverse of a matrix.
1331: 
1332:     Calculate a generalized inverse of a matrix using its
1333:     singular-value decomposition and including all 'large' singular
1334:     values.
1335: 
1336:     Parameters
1337:     ----------
1338:     a : (M, N) array_like
1339:         Matrix to be pseudo-inverted.
1340:     cond, rcond : float or None
1341:         Cutoff for 'small' singular values.
1342:         Singular values smaller than ``rcond*largest_singular_value``
1343:         are considered zero.
1344:         If None or -1, suitable machine precision is used.
1345:     return_rank : bool, optional
1346:         if True, return the effective rank of the matrix
1347:     check_finite : bool, optional
1348:         Whether to check that the input matrix contains only finite numbers.
1349:         Disabling may give a performance gain, but may result in problems
1350:         (crashes, non-termination) if the inputs do contain infinities or NaNs.
1351: 
1352:     Returns
1353:     -------
1354:     B : (N, M) ndarray
1355:         The pseudo-inverse of matrix `a`.
1356:     rank : int
1357:         The effective rank of the matrix.  Returned if return_rank == True
1358: 
1359:     Raises
1360:     ------
1361:     LinAlgError
1362:         If SVD computation does not converge.
1363: 
1364:     Examples
1365:     --------
1366:     >>> from scipy import linalg
1367:     >>> a = np.random.randn(9, 6)
1368:     >>> B = linalg.pinv2(a)
1369:     >>> np.allclose(a, np.dot(a, np.dot(B, a)))
1370:     True
1371:     >>> np.allclose(B, np.dot(B, np.dot(a, B)))
1372:     True
1373: 
1374:     '''
1375:     a = _asarray_validated(a, check_finite=check_finite)
1376:     u, s, vh = decomp_svd.svd(a, full_matrices=False, check_finite=False)
1377: 
1378:     if rcond is not None:
1379:         cond = rcond
1380:     if cond in [None, -1]:
1381:         t = u.dtype.char.lower()
1382:         factor = {'f': 1E3, 'd': 1E6}
1383:         cond = factor[t] * np.finfo(t).eps
1384: 
1385:     rank = np.sum(s > cond * np.max(s))
1386: 
1387:     u = u[:, :rank]
1388:     u /= s[:rank]
1389:     B = np.transpose(np.conjugate(np.dot(u, vh[:rank])))
1390: 
1391:     if return_rank:
1392:         return B, rank
1393:     else:
1394:         return B
1395: 
1396: 
1397: def pinvh(a, cond=None, rcond=None, lower=True, return_rank=False,
1398:           check_finite=True):
1399:     '''
1400:     Compute the (Moore-Penrose) pseudo-inverse of a Hermitian matrix.
1401: 
1402:     Calculate a generalized inverse of a Hermitian or real symmetric matrix
1403:     using its eigenvalue decomposition and including all eigenvalues with
1404:     'large' absolute value.
1405: 
1406:     Parameters
1407:     ----------
1408:     a : (N, N) array_like
1409:         Real symmetric or complex hermetian matrix to be pseudo-inverted
1410:     cond, rcond : float or None
1411:         Cutoff for 'small' eigenvalues.
1412:         Singular values smaller than rcond * largest_eigenvalue are considered
1413:         zero.
1414: 
1415:         If None or -1, suitable machine precision is used.
1416:     lower : bool, optional
1417:         Whether the pertinent array data is taken from the lower or upper
1418:         triangle of a. (Default: lower)
1419:     return_rank : bool, optional
1420:         if True, return the effective rank of the matrix
1421:     check_finite : bool, optional
1422:         Whether to check that the input matrix contains only finite numbers.
1423:         Disabling may give a performance gain, but may result in problems
1424:         (crashes, non-termination) if the inputs do contain infinities or NaNs.
1425: 
1426:     Returns
1427:     -------
1428:     B : (N, N) ndarray
1429:         The pseudo-inverse of matrix `a`.
1430:     rank : int
1431:         The effective rank of the matrix.  Returned if return_rank == True
1432: 
1433:     Raises
1434:     ------
1435:     LinAlgError
1436:         If eigenvalue does not converge
1437: 
1438:     Examples
1439:     --------
1440:     >>> from scipy.linalg import pinvh
1441:     >>> a = np.random.randn(9, 6)
1442:     >>> a = np.dot(a, a.T)
1443:     >>> B = pinvh(a)
1444:     >>> np.allclose(a, np.dot(a, np.dot(B, a)))
1445:     True
1446:     >>> np.allclose(B, np.dot(B, np.dot(a, B)))
1447:     True
1448: 
1449:     '''
1450:     a = _asarray_validated(a, check_finite=check_finite)
1451:     s, u = decomp.eigh(a, lower=lower, check_finite=False)
1452: 
1453:     if rcond is not None:
1454:         cond = rcond
1455:     if cond in [None, -1]:
1456:         t = u.dtype.char.lower()
1457:         factor = {'f': 1E3, 'd': 1E6}
1458:         cond = factor[t] * np.finfo(t).eps
1459: 
1460:     # For Hermitian matrices, singular values equal abs(eigenvalues)
1461:     above_cutoff = (abs(s) > cond * np.max(abs(s)))
1462:     psigma_diag = 1.0 / s[above_cutoff]
1463:     u = u[:, above_cutoff]
1464: 
1465:     B = np.dot(u * psigma_diag, np.conjugate(u).T)
1466: 
1467:     if return_rank:
1468:         return B, len(psigma_diag)
1469:     else:
1470:         return B
1471: 
1472: 
1473: def matrix_balance(A, permute=True, scale=True, separate=False,
1474:                    overwrite_a=False):
1475:     '''
1476:     Compute a diagonal similarity transformation for row/column balancing.
1477: 
1478:     The balancing tries to equalize the row and column 1-norms by applying
1479:     a similarity transformation such that the magnitude variation of the
1480:     matrix entries is reflected to the scaling matrices.
1481: 
1482:     Moreover, if enabled, the matrix is first permuted to isolate the upper
1483:     triangular parts of the matrix and, again if scaling is also enabled,
1484:     only the remaining subblocks are subjected to scaling.
1485: 
1486:     The balanced matrix satisfies the following equality
1487: 
1488:     .. math::
1489: 
1490:                         B = T^{-1} A T
1491: 
1492:     The scaling coefficients are approximated to the nearest power of 2
1493:     to avoid round-off errors.
1494: 
1495:     Parameters
1496:     ----------
1497:     A : (n, n) array_like
1498:         Square data matrix for the balancing.
1499:     permute : bool, optional
1500:         The selector to define whether permutation of A is also performed
1501:         prior to scaling.
1502:     scale : bool, optional
1503:         The selector to turn on and off the scaling. If False, the matrix
1504:         will not be scaled.
1505:     separate : bool, optional
1506:         This switches from returning a full matrix of the transformation
1507:         to a tuple of two separate 1D permutation and scaling arrays.
1508:     overwrite_a : bool, optional
1509:         This is passed to xGEBAL directly. Essentially, overwrites the result
1510:         to the data. It might increase the space efficiency. See LAPACK manual
1511:         for details. This is False by default.
1512: 
1513:     Returns
1514:     -------
1515:     B : (n, n) ndarray
1516:         Balanced matrix
1517:     T : (n, n) ndarray
1518:         A possibly permuted diagonal matrix whose nonzero entries are
1519:         integer powers of 2 to avoid numerical truncation errors.
1520:     scale, perm : (n,) ndarray
1521:         If ``separate`` keyword is set to True then instead of the array
1522:         ``T`` above, the scaling and the permutation vectors are given
1523:         separately as a tuple without allocating the full array ``T``.
1524: 
1525:     .. versionadded:: 0.19.0
1526: 
1527:     Notes
1528:     -----
1529: 
1530:     This algorithm is particularly useful for eigenvalue and matrix
1531:     decompositions and in many cases it is already called by various
1532:     LAPACK routines.
1533: 
1534:     The algorithm is based on the well-known technique of [1]_ and has
1535:     been modified to account for special cases. See [2]_ for details
1536:     which have been implemented since LAPACK v3.5.0. Before this version
1537:     there are corner cases where balancing can actually worsen the
1538:     conditioning. See [3]_ for such examples.
1539: 
1540:     The code is a wrapper around LAPACK's xGEBAL routine family for matrix
1541:     balancing.
1542: 
1543:     Examples
1544:     --------
1545:     >>> from scipy import linalg
1546:     >>> x = np.array([[1,2,0], [9,1,0.01], [1,2,10*np.pi]])
1547: 
1548:     >>> y, permscale = linalg.matrix_balance(x)
1549:     >>> np.abs(x).sum(axis=0) / np.abs(x).sum(axis=1)
1550:     array([ 3.66666667,  0.4995005 ,  0.91312162])
1551: 
1552:     >>> np.abs(y).sum(axis=0) / np.abs(y).sum(axis=1)
1553:     array([ 1.2       ,  1.27041742,  0.92658316])  # may vary
1554: 
1555:     >>> permscale  # only powers of 2 (0.5 == 2^(-1))
1556:     array([[  0.5,   0. ,  0. ],  # may vary
1557:            [  0. ,   1. ,  0. ],
1558:            [  0. ,   0. ,  1. ]])
1559: 
1560:     References
1561:     ----------
1562:     .. [1] : B.N. Parlett and C. Reinsch, "Balancing a Matrix for
1563:        Calculation of Eigenvalues and Eigenvectors", Numerische Mathematik,
1564:        Vol.13(4), 1969, DOI:10.1007/BF02165404
1565: 
1566:     .. [2] : R. James, J. Langou, B.R. Lowery, "On matrix balancing and
1567:        eigenvector computation", 2014, Available online:
1568:        http://arxiv.org/abs/1401.5766
1569: 
1570:     .. [3] :  D.S. Watkins. A case where balancing is harmful.
1571:        Electron. Trans. Numer. Anal, Vol.23, 2006.
1572: 
1573:     '''
1574: 
1575:     A = np.atleast_2d(_asarray_validated(A, check_finite=True))
1576: 
1577:     if not np.equal(*A.shape):
1578:         raise ValueError('The data matrix for balancing should be square.')
1579: 
1580:     gebal = get_lapack_funcs(('gebal'), (A,))
1581:     B, lo, hi, ps, info = gebal(A, scale=scale, permute=permute,
1582:                                 overwrite_a=overwrite_a)
1583: 
1584:     if info < 0:
1585:         raise ValueError('xGEBAL exited with the internal error '
1586:                          '"illegal value in argument number {}.". See '
1587:                          'LAPACK documentation for the xGEBAL error codes.'
1588:                          ''.format(-info))
1589: 
1590:     # Separate the permutations from the scalings and then convert to int
1591:     scaling = np.ones_like(ps, dtype=float)
1592:     scaling[lo:hi+1] = ps[lo:hi+1]
1593: 
1594:     # gebal uses 1-indexing
1595:     ps = ps.astype(int, copy=False) - 1
1596:     n = A.shape[0]
1597:     perm = np.arange(n)
1598: 
1599:     # LAPACK permutes with the ordering n --> hi, then 0--> lo
1600:     if hi < n:
1601:         for ind, x in enumerate(ps[hi+1:][::-1], 1):
1602:             if n-ind == x:
1603:                 continue
1604:             perm[[x, n-ind]] = perm[[n-ind, x]]
1605: 
1606:     if lo > 0:
1607:         for ind, x in enumerate(ps[:lo]):
1608:             if ind == x:
1609:                 continue
1610:             perm[[x, ind]] = perm[[ind, x]]
1611: 
1612:     if separate:
1613:         return B, (scaling, perm)
1614: 
1615:     # get the inverse permutation
1616:     iperm = np.empty_like(perm)
1617:     iperm[perm] = np.arange(n)
1618: 
1619:     return B, np.diag(scaling)[iperm, :]
1620: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 9, 0))

# 'import warnings' statement (line 9)
import warnings

import_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'warnings', warnings, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 10, 0))

# 'import numpy' statement (line 10)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/linalg/')
import_10141 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'numpy')

if (type(import_10141) is not StypyTypeError):

    if (import_10141 != 'pyd_module'):
        __import__(import_10141)
        sys_modules_10142 = sys.modules[import_10141]
        import_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'np', sys_modules_10142.module_type_store, module_type_store)
    else:
        import numpy as np

        import_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'np', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 10)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 0), 'numpy', import_10141)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/linalg/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 11, 0))

# 'from numpy import atleast_1d, atleast_2d' statement (line 11)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/linalg/')
import_10143 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'numpy')

if (type(import_10143) is not StypyTypeError):

    if (import_10143 != 'pyd_module'):
        __import__(import_10143)
        sys_modules_10144 = sys.modules[import_10143]
        import_from_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'numpy', sys_modules_10144.module_type_store, module_type_store, ['atleast_1d', 'atleast_2d'])
        nest_module(stypy.reporting.localization.Localization(__file__, 11, 0), __file__, sys_modules_10144, sys_modules_10144.module_type_store, module_type_store)
    else:
        from numpy import atleast_1d, atleast_2d

        import_from_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'numpy', None, module_type_store, ['atleast_1d', 'atleast_2d'], [atleast_1d, atleast_2d])

else:
    # Assigning a type to the variable 'numpy' (line 11)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 0), 'numpy', import_10143)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/linalg/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 12, 0))

# 'from scipy.linalg.flinalg import get_flinalg_funcs' statement (line 12)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/linalg/')
import_10145 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 12, 0), 'scipy.linalg.flinalg')

if (type(import_10145) is not StypyTypeError):

    if (import_10145 != 'pyd_module'):
        __import__(import_10145)
        sys_modules_10146 = sys.modules[import_10145]
        import_from_module(stypy.reporting.localization.Localization(__file__, 12, 0), 'scipy.linalg.flinalg', sys_modules_10146.module_type_store, module_type_store, ['get_flinalg_funcs'])
        nest_module(stypy.reporting.localization.Localization(__file__, 12, 0), __file__, sys_modules_10146, sys_modules_10146.module_type_store, module_type_store)
    else:
        from scipy.linalg.flinalg import get_flinalg_funcs

        import_from_module(stypy.reporting.localization.Localization(__file__, 12, 0), 'scipy.linalg.flinalg', None, module_type_store, ['get_flinalg_funcs'], [get_flinalg_funcs])

else:
    # Assigning a type to the variable 'scipy.linalg.flinalg' (line 12)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 0), 'scipy.linalg.flinalg', import_10145)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/linalg/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 13, 0))

# 'from scipy.linalg.lapack import get_lapack_funcs, _compute_lwork' statement (line 13)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/linalg/')
import_10147 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 13, 0), 'scipy.linalg.lapack')

if (type(import_10147) is not StypyTypeError):

    if (import_10147 != 'pyd_module'):
        __import__(import_10147)
        sys_modules_10148 = sys.modules[import_10147]
        import_from_module(stypy.reporting.localization.Localization(__file__, 13, 0), 'scipy.linalg.lapack', sys_modules_10148.module_type_store, module_type_store, ['get_lapack_funcs', '_compute_lwork'])
        nest_module(stypy.reporting.localization.Localization(__file__, 13, 0), __file__, sys_modules_10148, sys_modules_10148.module_type_store, module_type_store)
    else:
        from scipy.linalg.lapack import get_lapack_funcs, _compute_lwork

        import_from_module(stypy.reporting.localization.Localization(__file__, 13, 0), 'scipy.linalg.lapack', None, module_type_store, ['get_lapack_funcs', '_compute_lwork'], [get_lapack_funcs, _compute_lwork])

else:
    # Assigning a type to the variable 'scipy.linalg.lapack' (line 13)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 0), 'scipy.linalg.lapack', import_10147)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/linalg/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 14, 0))

# 'from scipy.linalg.misc import LinAlgError, _datacopied' statement (line 14)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/linalg/')
import_10149 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 14, 0), 'scipy.linalg.misc')

if (type(import_10149) is not StypyTypeError):

    if (import_10149 != 'pyd_module'):
        __import__(import_10149)
        sys_modules_10150 = sys.modules[import_10149]
        import_from_module(stypy.reporting.localization.Localization(__file__, 14, 0), 'scipy.linalg.misc', sys_modules_10150.module_type_store, module_type_store, ['LinAlgError', '_datacopied'])
        nest_module(stypy.reporting.localization.Localization(__file__, 14, 0), __file__, sys_modules_10150, sys_modules_10150.module_type_store, module_type_store)
    else:
        from scipy.linalg.misc import LinAlgError, _datacopied

        import_from_module(stypy.reporting.localization.Localization(__file__, 14, 0), 'scipy.linalg.misc', None, module_type_store, ['LinAlgError', '_datacopied'], [LinAlgError, _datacopied])

else:
    # Assigning a type to the variable 'scipy.linalg.misc' (line 14)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 0), 'scipy.linalg.misc', import_10149)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/linalg/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 15, 0))

# 'from scipy.linalg.decomp import _asarray_validated' statement (line 15)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/linalg/')
import_10151 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 15, 0), 'scipy.linalg.decomp')

if (type(import_10151) is not StypyTypeError):

    if (import_10151 != 'pyd_module'):
        __import__(import_10151)
        sys_modules_10152 = sys.modules[import_10151]
        import_from_module(stypy.reporting.localization.Localization(__file__, 15, 0), 'scipy.linalg.decomp', sys_modules_10152.module_type_store, module_type_store, ['_asarray_validated'])
        nest_module(stypy.reporting.localization.Localization(__file__, 15, 0), __file__, sys_modules_10152, sys_modules_10152.module_type_store, module_type_store)
    else:
        from scipy.linalg.decomp import _asarray_validated

        import_from_module(stypy.reporting.localization.Localization(__file__, 15, 0), 'scipy.linalg.decomp', None, module_type_store, ['_asarray_validated'], [_asarray_validated])

else:
    # Assigning a type to the variable 'scipy.linalg.decomp' (line 15)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 0), 'scipy.linalg.decomp', import_10151)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/linalg/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 16, 0))

# 'from scipy.linalg import decomp, decomp_svd' statement (line 16)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/linalg/')
import_10153 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 16, 0), 'scipy.linalg')

if (type(import_10153) is not StypyTypeError):

    if (import_10153 != 'pyd_module'):
        __import__(import_10153)
        sys_modules_10154 = sys.modules[import_10153]
        import_from_module(stypy.reporting.localization.Localization(__file__, 16, 0), 'scipy.linalg', sys_modules_10154.module_type_store, module_type_store, ['decomp', 'decomp_svd'])
        nest_module(stypy.reporting.localization.Localization(__file__, 16, 0), __file__, sys_modules_10154, sys_modules_10154.module_type_store, module_type_store)
    else:
        from scipy.linalg import decomp, decomp_svd

        import_from_module(stypy.reporting.localization.Localization(__file__, 16, 0), 'scipy.linalg', None, module_type_store, ['decomp', 'decomp_svd'], [decomp, decomp_svd])

else:
    # Assigning a type to the variable 'scipy.linalg' (line 16)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 0), 'scipy.linalg', import_10153)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/linalg/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 17, 0))

# 'from scipy.linalg._solve_toeplitz import levinson' statement (line 17)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/linalg/')
import_10155 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 17, 0), 'scipy.linalg._solve_toeplitz')

if (type(import_10155) is not StypyTypeError):

    if (import_10155 != 'pyd_module'):
        __import__(import_10155)
        sys_modules_10156 = sys.modules[import_10155]
        import_from_module(stypy.reporting.localization.Localization(__file__, 17, 0), 'scipy.linalg._solve_toeplitz', sys_modules_10156.module_type_store, module_type_store, ['levinson'])
        nest_module(stypy.reporting.localization.Localization(__file__, 17, 0), __file__, sys_modules_10156, sys_modules_10156.module_type_store, module_type_store)
    else:
        from scipy.linalg._solve_toeplitz import levinson

        import_from_module(stypy.reporting.localization.Localization(__file__, 17, 0), 'scipy.linalg._solve_toeplitz', None, module_type_store, ['levinson'], [levinson])

else:
    # Assigning a type to the variable 'scipy.linalg._solve_toeplitz' (line 17)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 17, 0), 'scipy.linalg._solve_toeplitz', import_10155)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/linalg/')


# Assigning a List to a Name (line 19):

# Assigning a List to a Name (line 19):
__all__ = ['solve', 'solve_triangular', 'solveh_banded', 'solve_banded', 'solve_toeplitz', 'solve_circulant', 'inv', 'det', 'lstsq', 'pinv', 'pinv2', 'pinvh', 'matrix_balance']
module_type_store.set_exportable_members(['solve', 'solve_triangular', 'solveh_banded', 'solve_banded', 'solve_toeplitz', 'solve_circulant', 'inv', 'det', 'lstsq', 'pinv', 'pinv2', 'pinvh', 'matrix_balance'])

# Obtaining an instance of the builtin type 'list' (line 19)
list_10157 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 10), 'list')
# Adding type elements to the builtin type 'list' instance (line 19)
# Adding element type (line 19)
str_10158 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 11), 'str', 'solve')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 10), list_10157, str_10158)
# Adding element type (line 19)
str_10159 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 20), 'str', 'solve_triangular')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 10), list_10157, str_10159)
# Adding element type (line 19)
str_10160 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 40), 'str', 'solveh_banded')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 10), list_10157, str_10160)
# Adding element type (line 19)
str_10161 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 57), 'str', 'solve_banded')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 10), list_10157, str_10161)
# Adding element type (line 19)
str_10162 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 11), 'str', 'solve_toeplitz')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 10), list_10157, str_10162)
# Adding element type (line 19)
str_10163 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 29), 'str', 'solve_circulant')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 10), list_10157, str_10163)
# Adding element type (line 19)
str_10164 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 48), 'str', 'inv')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 10), list_10157, str_10164)
# Adding element type (line 19)
str_10165 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 55), 'str', 'det')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 10), list_10157, str_10165)
# Adding element type (line 19)
str_10166 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 62), 'str', 'lstsq')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 10), list_10157, str_10166)
# Adding element type (line 19)
str_10167 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 11), 'str', 'pinv')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 10), list_10157, str_10167)
# Adding element type (line 19)
str_10168 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 19), 'str', 'pinv2')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 10), list_10157, str_10168)
# Adding element type (line 19)
str_10169 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 28), 'str', 'pinvh')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 10), list_10157, str_10169)
# Adding element type (line 19)
str_10170 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 37), 'str', 'matrix_balance')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 10), list_10157, str_10170)

# Assigning a type to the variable '__all__' (line 19)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 19, 0), '__all__', list_10157)

@norecursion
def _solve_check(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'None' (line 25)
    None_10171 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 32), 'None')
    # Getting the type of 'None' (line 25)
    None_10172 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 44), 'None')
    defaults = [None_10171, None_10172]
    # Create a new context for function '_solve_check'
    module_type_store = module_type_store.open_function_context('_solve_check', 25, 0, False)
    
    # Passed parameters checking function
    _solve_check.stypy_localization = localization
    _solve_check.stypy_type_of_self = None
    _solve_check.stypy_type_store = module_type_store
    _solve_check.stypy_function_name = '_solve_check'
    _solve_check.stypy_param_names_list = ['n', 'info', 'lamch', 'rcond']
    _solve_check.stypy_varargs_param_name = None
    _solve_check.stypy_kwargs_param_name = None
    _solve_check.stypy_call_defaults = defaults
    _solve_check.stypy_call_varargs = varargs
    _solve_check.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_solve_check', ['n', 'info', 'lamch', 'rcond'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_solve_check', localization, ['n', 'info', 'lamch', 'rcond'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_solve_check(...)' code ##################

    str_10173 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 4), 'str', ' Check arguments during the different steps of the solution phase ')
    
    
    # Getting the type of 'info' (line 27)
    info_10174 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 7), 'info')
    int_10175 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 14), 'int')
    # Applying the binary operator '<' (line 27)
    result_lt_10176 = python_operator(stypy.reporting.localization.Localization(__file__, 27, 7), '<', info_10174, int_10175)
    
    # Testing the type of an if condition (line 27)
    if_condition_10177 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 27, 4), result_lt_10176)
    # Assigning a type to the variable 'if_condition_10177' (line 27)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 27, 4), 'if_condition_10177', if_condition_10177)
    # SSA begins for if statement (line 27)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 28)
    # Processing the call arguments (line 28)
    
    # Call to format(...): (line 28)
    # Processing the call arguments (line 28)
    
    # Getting the type of 'info' (line 29)
    info_10181 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 37), 'info', False)
    # Applying the 'usub' unary operator (line 29)
    result___neg___10182 = python_operator(stypy.reporting.localization.Localization(__file__, 29, 36), 'usub', info_10181)
    
    # Processing the call keyword arguments (line 28)
    kwargs_10183 = {}
    str_10179 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 25), 'str', 'LAPACK reported an illegal value in {}-th argument.')
    # Obtaining the member 'format' of a type (line 28)
    format_10180 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 28, 25), str_10179, 'format')
    # Calling format(args, kwargs) (line 28)
    format_call_result_10184 = invoke(stypy.reporting.localization.Localization(__file__, 28, 25), format_10180, *[result___neg___10182], **kwargs_10183)
    
    # Processing the call keyword arguments (line 28)
    kwargs_10185 = {}
    # Getting the type of 'ValueError' (line 28)
    ValueError_10178 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 28)
    ValueError_call_result_10186 = invoke(stypy.reporting.localization.Localization(__file__, 28, 14), ValueError_10178, *[format_call_result_10184], **kwargs_10185)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 28, 8), ValueError_call_result_10186, 'raise parameter', BaseException)
    # SSA branch for the else part of an if statement (line 27)
    module_type_store.open_ssa_branch('else')
    
    
    int_10187 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 9), 'int')
    # Getting the type of 'info' (line 30)
    info_10188 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 13), 'info')
    # Applying the binary operator '<' (line 30)
    result_lt_10189 = python_operator(stypy.reporting.localization.Localization(__file__, 30, 9), '<', int_10187, info_10188)
    
    # Testing the type of an if condition (line 30)
    if_condition_10190 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 30, 9), result_lt_10189)
    # Assigning a type to the variable 'if_condition_10190' (line 30)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 30, 9), 'if_condition_10190', if_condition_10190)
    # SSA begins for if statement (line 30)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to LinAlgError(...): (line 31)
    # Processing the call arguments (line 31)
    str_10192 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 26), 'str', 'Matrix is singular.')
    # Processing the call keyword arguments (line 31)
    kwargs_10193 = {}
    # Getting the type of 'LinAlgError' (line 31)
    LinAlgError_10191 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 14), 'LinAlgError', False)
    # Calling LinAlgError(args, kwargs) (line 31)
    LinAlgError_call_result_10194 = invoke(stypy.reporting.localization.Localization(__file__, 31, 14), LinAlgError_10191, *[str_10192], **kwargs_10193)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 31, 8), LinAlgError_call_result_10194, 'raise parameter', BaseException)
    # SSA join for if statement (line 30)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 27)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Type idiom detected: calculating its left and rigth part (line 33)
    # Getting the type of 'lamch' (line 33)
    lamch_10195 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 7), 'lamch')
    # Getting the type of 'None' (line 33)
    None_10196 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 16), 'None')
    
    (may_be_10197, more_types_in_union_10198) = may_be_none(lamch_10195, None_10196)

    if may_be_10197:

        if more_types_in_union_10198:
            # Runtime conditional SSA (line 33)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        # Assigning a type to the variable 'stypy_return_type' (line 34)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 34, 8), 'stypy_return_type', types.NoneType)

        if more_types_in_union_10198:
            # SSA join for if statement (line 33)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Assigning a Call to a Name (line 35):
    
    # Assigning a Call to a Name (line 35):
    
    # Call to lamch(...): (line 35)
    # Processing the call arguments (line 35)
    str_10200 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 14), 'str', 'E')
    # Processing the call keyword arguments (line 35)
    kwargs_10201 = {}
    # Getting the type of 'lamch' (line 35)
    lamch_10199 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 8), 'lamch', False)
    # Calling lamch(args, kwargs) (line 35)
    lamch_call_result_10202 = invoke(stypy.reporting.localization.Localization(__file__, 35, 8), lamch_10199, *[str_10200], **kwargs_10201)
    
    # Assigning a type to the variable 'E' (line 35)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 35, 4), 'E', lamch_call_result_10202)
    
    
    # Getting the type of 'rcond' (line 36)
    rcond_10203 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 7), 'rcond')
    # Getting the type of 'E' (line 36)
    E_10204 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 15), 'E')
    # Applying the binary operator '<' (line 36)
    result_lt_10205 = python_operator(stypy.reporting.localization.Localization(__file__, 36, 7), '<', rcond_10203, E_10204)
    
    # Testing the type of an if condition (line 36)
    if_condition_10206 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 36, 4), result_lt_10205)
    # Assigning a type to the variable 'if_condition_10206' (line 36)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 36, 4), 'if_condition_10206', if_condition_10206)
    # SSA begins for if statement (line 36)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to warn(...): (line 37)
    # Processing the call arguments (line 37)
    
    # Call to format(...): (line 37)
    # Processing the call arguments (line 37)
    # Getting the type of 'rcond' (line 39)
    rcond_10211 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 67), 'rcond', False)
    # Getting the type of 'E' (line 39)
    E_10212 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 74), 'E', False)
    # Processing the call keyword arguments (line 37)
    kwargs_10213 = {}
    str_10209 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 22), 'str', 'scipy.linalg.solve\nIll-conditioned matrix detected. Result is not guaranteed to be accurate.\nReciprocal condition number/precision: {} / {}')
    # Obtaining the member 'format' of a type (line 37)
    format_10210 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 37, 22), str_10209, 'format')
    # Calling format(args, kwargs) (line 37)
    format_call_result_10214 = invoke(stypy.reporting.localization.Localization(__file__, 37, 22), format_10210, *[rcond_10211, E_10212], **kwargs_10213)
    
    # Getting the type of 'RuntimeWarning' (line 40)
    RuntimeWarning_10215 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 22), 'RuntimeWarning', False)
    # Processing the call keyword arguments (line 37)
    kwargs_10216 = {}
    # Getting the type of 'warnings' (line 37)
    warnings_10207 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 8), 'warnings', False)
    # Obtaining the member 'warn' of a type (line 37)
    warn_10208 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 37, 8), warnings_10207, 'warn')
    # Calling warn(args, kwargs) (line 37)
    warn_call_result_10217 = invoke(stypy.reporting.localization.Localization(__file__, 37, 8), warn_10208, *[format_call_result_10214, RuntimeWarning_10215], **kwargs_10216)
    
    # SSA join for if statement (line 36)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of '_solve_check(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_solve_check' in the type store
    # Getting the type of 'stypy_return_type' (line 25)
    stypy_return_type_10218 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_10218)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_solve_check'
    return stypy_return_type_10218

# Assigning a type to the variable '_solve_check' (line 25)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 25, 0), '_solve_check', _solve_check)

@norecursion
def solve(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'False' (line 43)
    False_10219 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 24), 'False')
    # Getting the type of 'False' (line 43)
    False_10220 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 37), 'False')
    # Getting the type of 'False' (line 43)
    False_10221 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 56), 'False')
    # Getting the type of 'False' (line 44)
    False_10222 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 22), 'False')
    # Getting the type of 'None' (line 44)
    None_10223 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 35), 'None')
    # Getting the type of 'True' (line 44)
    True_10224 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 54), 'True')
    str_10225 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 69), 'str', 'gen')
    # Getting the type of 'False' (line 45)
    False_10226 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 21), 'False')
    defaults = [False_10219, False_10220, False_10221, False_10222, None_10223, True_10224, str_10225, False_10226]
    # Create a new context for function 'solve'
    module_type_store = module_type_store.open_function_context('solve', 43, 0, False)
    
    # Passed parameters checking function
    solve.stypy_localization = localization
    solve.stypy_type_of_self = None
    solve.stypy_type_store = module_type_store
    solve.stypy_function_name = 'solve'
    solve.stypy_param_names_list = ['a', 'b', 'sym_pos', 'lower', 'overwrite_a', 'overwrite_b', 'debug', 'check_finite', 'assume_a', 'transposed']
    solve.stypy_varargs_param_name = None
    solve.stypy_kwargs_param_name = None
    solve.stypy_call_defaults = defaults
    solve.stypy_call_varargs = varargs
    solve.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'solve', ['a', 'b', 'sym_pos', 'lower', 'overwrite_a', 'overwrite_b', 'debug', 'check_finite', 'assume_a', 'transposed'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'solve', localization, ['a', 'b', 'sym_pos', 'lower', 'overwrite_a', 'overwrite_b', 'debug', 'check_finite', 'assume_a', 'transposed'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'solve(...)' code ##################

    str_10227 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 136, (-1)), 'str', "\n    Solves the linear equation set ``a * x = b`` for the unknown ``x``\n    for square ``a`` matrix.\n\n    If the data matrix is known to be a particular type then supplying the\n    corresponding string to ``assume_a`` key chooses the dedicated solver.\n    The available options are\n\n    ===================  ========\n     generic matrix       'gen'\n     symmetric            'sym'\n     hermitian            'her'\n     positive definite    'pos'\n    ===================  ========\n\n    If omitted, ``'gen'`` is the default structure.\n\n    The datatype of the arrays define which solver is called regardless\n    of the values. In other words, even when the complex array entries have\n    precisely zero imaginary parts, the complex solver will be called based\n    on the data type of the array.\n\n    Parameters\n    ----------\n    a : (N, N) array_like\n        Square input data\n    b : (N, NRHS) array_like\n        Input data for the right hand side.\n    sym_pos : bool, optional\n        Assume `a` is symmetric and positive definite. This key is deprecated\n        and assume_a = 'pos' keyword is recommended instead. The functionality\n        is the same. It will be removed in the future.\n    lower : bool, optional\n        If True, only the data contained in the lower triangle of `a`. Default\n        is to use upper triangle. (ignored for ``'gen'``)\n    overwrite_a : bool, optional\n        Allow overwriting data in `a` (may enhance performance).\n        Default is False.\n    overwrite_b : bool, optional\n        Allow overwriting data in `b` (may enhance performance).\n        Default is False.\n    check_finite : bool, optional\n        Whether to check that the input matrices contain only finite numbers.\n        Disabling may give a performance gain, but may result in problems\n        (crashes, non-termination) if the inputs do contain infinities or NaNs.\n    assume_a : str, optional\n        Valid entries are explained above.\n    transposed: bool, optional\n        If True, ``a^T x = b`` for real matrices, raises `NotImplementedError`\n        for complex matrices (only for True).\n\n    Returns\n    -------\n    x : (N, NRHS) ndarray\n        The solution array.\n\n    Raises\n    ------\n    ValueError\n        If size mismatches detected or input a is not square.\n    LinAlgError\n        If the matrix is singular.\n    RuntimeWarning\n        If an ill-conditioned input a is detected.\n    NotImplementedError\n        If transposed is True and input a is a complex matrix.\n\n    Examples\n    --------\n    Given `a` and `b`, solve for `x`:\n\n    >>> a = np.array([[3, 2, 0], [1, -1, 0], [0, 5, 1]])\n    >>> b = np.array([2, 4, -1])\n    >>> from scipy import linalg\n    >>> x = linalg.solve(a, b)\n    >>> x\n    array([ 2., -2.,  9.])\n    >>> np.dot(a, x) == b\n    array([ True,  True,  True], dtype=bool)\n\n    Notes\n    -----\n    If the input b matrix is a 1D array with N elements, when supplied\n    together with an NxN input a, it is assumed as a valid column vector\n    despite the apparent size mismatch. This is compatible with the\n    numpy.dot() behavior and the returned result is still 1D array.\n\n    The generic, symmetric, hermitian and positive definite solutions are\n    obtained via calling ?GESV, ?SYSV, ?HESV, and ?POSV routines of\n    LAPACK respectively.\n    ")
    
    # Assigning a Name to a Name (line 138):
    
    # Assigning a Name to a Name (line 138):
    # Getting the type of 'False' (line 138)
    False_10228 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 14), 'False')
    # Assigning a type to the variable 'b_is_1D' (line 138)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 138, 4), 'b_is_1D', False_10228)
    
    # Assigning a Call to a Name (line 140):
    
    # Assigning a Call to a Name (line 140):
    
    # Call to atleast_2d(...): (line 140)
    # Processing the call arguments (line 140)
    
    # Call to _asarray_validated(...): (line 140)
    # Processing the call arguments (line 140)
    # Getting the type of 'a' (line 140)
    a_10231 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 39), 'a', False)
    # Processing the call keyword arguments (line 140)
    # Getting the type of 'check_finite' (line 140)
    check_finite_10232 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 55), 'check_finite', False)
    keyword_10233 = check_finite_10232
    kwargs_10234 = {'check_finite': keyword_10233}
    # Getting the type of '_asarray_validated' (line 140)
    _asarray_validated_10230 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 20), '_asarray_validated', False)
    # Calling _asarray_validated(args, kwargs) (line 140)
    _asarray_validated_call_result_10235 = invoke(stypy.reporting.localization.Localization(__file__, 140, 20), _asarray_validated_10230, *[a_10231], **kwargs_10234)
    
    # Processing the call keyword arguments (line 140)
    kwargs_10236 = {}
    # Getting the type of 'atleast_2d' (line 140)
    atleast_2d_10229 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 9), 'atleast_2d', False)
    # Calling atleast_2d(args, kwargs) (line 140)
    atleast_2d_call_result_10237 = invoke(stypy.reporting.localization.Localization(__file__, 140, 9), atleast_2d_10229, *[_asarray_validated_call_result_10235], **kwargs_10236)
    
    # Assigning a type to the variable 'a1' (line 140)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 140, 4), 'a1', atleast_2d_call_result_10237)
    
    # Assigning a Call to a Name (line 141):
    
    # Assigning a Call to a Name (line 141):
    
    # Call to atleast_1d(...): (line 141)
    # Processing the call arguments (line 141)
    
    # Call to _asarray_validated(...): (line 141)
    # Processing the call arguments (line 141)
    # Getting the type of 'b' (line 141)
    b_10240 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 39), 'b', False)
    # Processing the call keyword arguments (line 141)
    # Getting the type of 'check_finite' (line 141)
    check_finite_10241 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 55), 'check_finite', False)
    keyword_10242 = check_finite_10241
    kwargs_10243 = {'check_finite': keyword_10242}
    # Getting the type of '_asarray_validated' (line 141)
    _asarray_validated_10239 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 20), '_asarray_validated', False)
    # Calling _asarray_validated(args, kwargs) (line 141)
    _asarray_validated_call_result_10244 = invoke(stypy.reporting.localization.Localization(__file__, 141, 20), _asarray_validated_10239, *[b_10240], **kwargs_10243)
    
    # Processing the call keyword arguments (line 141)
    kwargs_10245 = {}
    # Getting the type of 'atleast_1d' (line 141)
    atleast_1d_10238 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 9), 'atleast_1d', False)
    # Calling atleast_1d(args, kwargs) (line 141)
    atleast_1d_call_result_10246 = invoke(stypy.reporting.localization.Localization(__file__, 141, 9), atleast_1d_10238, *[_asarray_validated_call_result_10244], **kwargs_10245)
    
    # Assigning a type to the variable 'b1' (line 141)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 141, 4), 'b1', atleast_1d_call_result_10246)
    
    # Assigning a Subscript to a Name (line 142):
    
    # Assigning a Subscript to a Name (line 142):
    
    # Obtaining the type of the subscript
    int_10247 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 142, 17), 'int')
    # Getting the type of 'a1' (line 142)
    a1_10248 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 8), 'a1')
    # Obtaining the member 'shape' of a type (line 142)
    shape_10249 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 142, 8), a1_10248, 'shape')
    # Obtaining the member '__getitem__' of a type (line 142)
    getitem___10250 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 142, 8), shape_10249, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 142)
    subscript_call_result_10251 = invoke(stypy.reporting.localization.Localization(__file__, 142, 8), getitem___10250, int_10247)
    
    # Assigning a type to the variable 'n' (line 142)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 142, 4), 'n', subscript_call_result_10251)
    
    # Assigning a BoolOp to a Name (line 144):
    
    # Assigning a BoolOp to a Name (line 144):
    
    # Evaluating a boolean operation
    # Getting the type of 'overwrite_a' (line 144)
    overwrite_a_10252 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 18), 'overwrite_a')
    
    # Call to _datacopied(...): (line 144)
    # Processing the call arguments (line 144)
    # Getting the type of 'a1' (line 144)
    a1_10254 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 45), 'a1', False)
    # Getting the type of 'a' (line 144)
    a_10255 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 49), 'a', False)
    # Processing the call keyword arguments (line 144)
    kwargs_10256 = {}
    # Getting the type of '_datacopied' (line 144)
    _datacopied_10253 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 33), '_datacopied', False)
    # Calling _datacopied(args, kwargs) (line 144)
    _datacopied_call_result_10257 = invoke(stypy.reporting.localization.Localization(__file__, 144, 33), _datacopied_10253, *[a1_10254, a_10255], **kwargs_10256)
    
    # Applying the binary operator 'or' (line 144)
    result_or_keyword_10258 = python_operator(stypy.reporting.localization.Localization(__file__, 144, 18), 'or', overwrite_a_10252, _datacopied_call_result_10257)
    
    # Assigning a type to the variable 'overwrite_a' (line 144)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 144, 4), 'overwrite_a', result_or_keyword_10258)
    
    # Assigning a BoolOp to a Name (line 145):
    
    # Assigning a BoolOp to a Name (line 145):
    
    # Evaluating a boolean operation
    # Getting the type of 'overwrite_b' (line 145)
    overwrite_b_10259 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 18), 'overwrite_b')
    
    # Call to _datacopied(...): (line 145)
    # Processing the call arguments (line 145)
    # Getting the type of 'b1' (line 145)
    b1_10261 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 45), 'b1', False)
    # Getting the type of 'b' (line 145)
    b_10262 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 49), 'b', False)
    # Processing the call keyword arguments (line 145)
    kwargs_10263 = {}
    # Getting the type of '_datacopied' (line 145)
    _datacopied_10260 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 33), '_datacopied', False)
    # Calling _datacopied(args, kwargs) (line 145)
    _datacopied_call_result_10264 = invoke(stypy.reporting.localization.Localization(__file__, 145, 33), _datacopied_10260, *[b1_10261, b_10262], **kwargs_10263)
    
    # Applying the binary operator 'or' (line 145)
    result_or_keyword_10265 = python_operator(stypy.reporting.localization.Localization(__file__, 145, 18), 'or', overwrite_b_10259, _datacopied_call_result_10264)
    
    # Assigning a type to the variable 'overwrite_b' (line 145)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 145, 4), 'overwrite_b', result_or_keyword_10265)
    
    
    
    # Obtaining the type of the subscript
    int_10266 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 147, 16), 'int')
    # Getting the type of 'a1' (line 147)
    a1_10267 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 7), 'a1')
    # Obtaining the member 'shape' of a type (line 147)
    shape_10268 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 147, 7), a1_10267, 'shape')
    # Obtaining the member '__getitem__' of a type (line 147)
    getitem___10269 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 147, 7), shape_10268, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 147)
    subscript_call_result_10270 = invoke(stypy.reporting.localization.Localization(__file__, 147, 7), getitem___10269, int_10266)
    
    
    # Obtaining the type of the subscript
    int_10271 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 147, 31), 'int')
    # Getting the type of 'a1' (line 147)
    a1_10272 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 22), 'a1')
    # Obtaining the member 'shape' of a type (line 147)
    shape_10273 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 147, 22), a1_10272, 'shape')
    # Obtaining the member '__getitem__' of a type (line 147)
    getitem___10274 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 147, 22), shape_10273, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 147)
    subscript_call_result_10275 = invoke(stypy.reporting.localization.Localization(__file__, 147, 22), getitem___10274, int_10271)
    
    # Applying the binary operator '!=' (line 147)
    result_ne_10276 = python_operator(stypy.reporting.localization.Localization(__file__, 147, 7), '!=', subscript_call_result_10270, subscript_call_result_10275)
    
    # Testing the type of an if condition (line 147)
    if_condition_10277 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 147, 4), result_ne_10276)
    # Assigning a type to the variable 'if_condition_10277' (line 147)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 147, 4), 'if_condition_10277', if_condition_10277)
    # SSA begins for if statement (line 147)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 148)
    # Processing the call arguments (line 148)
    str_10279 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 148, 25), 'str', 'Input a needs to be a square matrix.')
    # Processing the call keyword arguments (line 148)
    kwargs_10280 = {}
    # Getting the type of 'ValueError' (line 148)
    ValueError_10278 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 148)
    ValueError_call_result_10281 = invoke(stypy.reporting.localization.Localization(__file__, 148, 14), ValueError_10278, *[str_10279], **kwargs_10280)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 148, 8), ValueError_call_result_10281, 'raise parameter', BaseException)
    # SSA join for if statement (line 147)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'n' (line 150)
    n_10282 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 7), 'n')
    
    # Obtaining the type of the subscript
    int_10283 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 150, 21), 'int')
    # Getting the type of 'b1' (line 150)
    b1_10284 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 12), 'b1')
    # Obtaining the member 'shape' of a type (line 150)
    shape_10285 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 150, 12), b1_10284, 'shape')
    # Obtaining the member '__getitem__' of a type (line 150)
    getitem___10286 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 150, 12), shape_10285, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 150)
    subscript_call_result_10287 = invoke(stypy.reporting.localization.Localization(__file__, 150, 12), getitem___10286, int_10283)
    
    # Applying the binary operator '!=' (line 150)
    result_ne_10288 = python_operator(stypy.reporting.localization.Localization(__file__, 150, 7), '!=', n_10282, subscript_call_result_10287)
    
    # Testing the type of an if condition (line 150)
    if_condition_10289 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 150, 4), result_ne_10288)
    # Assigning a type to the variable 'if_condition_10289' (line 150)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 150, 4), 'if_condition_10289', if_condition_10289)
    # SSA begins for if statement (line 150)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'n' (line 152)
    n_10290 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 16), 'n')
    int_10291 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 152, 21), 'int')
    # Applying the binary operator '==' (line 152)
    result_eq_10292 = python_operator(stypy.reporting.localization.Localization(__file__, 152, 16), '==', n_10290, int_10291)
    
    
    # Getting the type of 'b1' (line 152)
    b1_10293 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 27), 'b1')
    # Obtaining the member 'size' of a type (line 152)
    size_10294 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 152, 27), b1_10293, 'size')
    int_10295 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 152, 38), 'int')
    # Applying the binary operator '!=' (line 152)
    result_ne_10296 = python_operator(stypy.reporting.localization.Localization(__file__, 152, 27), '!=', size_10294, int_10295)
    
    # Applying the binary operator 'and' (line 152)
    result_and_keyword_10297 = python_operator(stypy.reporting.localization.Localization(__file__, 152, 16), 'and', result_eq_10292, result_ne_10296)
    
    # Applying the 'not' unary operator (line 152)
    result_not__10298 = python_operator(stypy.reporting.localization.Localization(__file__, 152, 11), 'not', result_and_keyword_10297)
    
    # Testing the type of an if condition (line 152)
    if_condition_10299 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 152, 8), result_not__10298)
    # Assigning a type to the variable 'if_condition_10299' (line 152)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 152, 8), 'if_condition_10299', if_condition_10299)
    # SSA begins for if statement (line 152)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 153)
    # Processing the call arguments (line 153)
    str_10301 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 153, 29), 'str', 'Input b has to have same number of rows as input a')
    # Processing the call keyword arguments (line 153)
    kwargs_10302 = {}
    # Getting the type of 'ValueError' (line 153)
    ValueError_10300 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 18), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 153)
    ValueError_call_result_10303 = invoke(stypy.reporting.localization.Localization(__file__, 153, 18), ValueError_10300, *[str_10301], **kwargs_10302)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 153, 12), ValueError_call_result_10303, 'raise parameter', BaseException)
    # SSA join for if statement (line 152)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 150)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'b1' (line 157)
    b1_10304 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 7), 'b1')
    # Obtaining the member 'size' of a type (line 157)
    size_10305 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 157, 7), b1_10304, 'size')
    int_10306 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 157, 18), 'int')
    # Applying the binary operator '==' (line 157)
    result_eq_10307 = python_operator(stypy.reporting.localization.Localization(__file__, 157, 7), '==', size_10305, int_10306)
    
    # Testing the type of an if condition (line 157)
    if_condition_10308 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 157, 4), result_eq_10307)
    # Assigning a type to the variable 'if_condition_10308' (line 157)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 157, 4), 'if_condition_10308', if_condition_10308)
    # SSA begins for if statement (line 157)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to asfortranarray(...): (line 158)
    # Processing the call arguments (line 158)
    
    # Call to copy(...): (line 158)
    # Processing the call keyword arguments (line 158)
    kwargs_10313 = {}
    # Getting the type of 'b1' (line 158)
    b1_10311 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 33), 'b1', False)
    # Obtaining the member 'copy' of a type (line 158)
    copy_10312 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 158, 33), b1_10311, 'copy')
    # Calling copy(args, kwargs) (line 158)
    copy_call_result_10314 = invoke(stypy.reporting.localization.Localization(__file__, 158, 33), copy_10312, *[], **kwargs_10313)
    
    # Processing the call keyword arguments (line 158)
    kwargs_10315 = {}
    # Getting the type of 'np' (line 158)
    np_10309 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 15), 'np', False)
    # Obtaining the member 'asfortranarray' of a type (line 158)
    asfortranarray_10310 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 158, 15), np_10309, 'asfortranarray')
    # Calling asfortranarray(args, kwargs) (line 158)
    asfortranarray_call_result_10316 = invoke(stypy.reporting.localization.Localization(__file__, 158, 15), asfortranarray_10310, *[copy_call_result_10314], **kwargs_10315)
    
    # Assigning a type to the variable 'stypy_return_type' (line 158)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 158, 8), 'stypy_return_type', asfortranarray_call_result_10316)
    # SSA join for if statement (line 157)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'b1' (line 161)
    b1_10317 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 7), 'b1')
    # Obtaining the member 'ndim' of a type (line 161)
    ndim_10318 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 161, 7), b1_10317, 'ndim')
    int_10319 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 161, 18), 'int')
    # Applying the binary operator '==' (line 161)
    result_eq_10320 = python_operator(stypy.reporting.localization.Localization(__file__, 161, 7), '==', ndim_10318, int_10319)
    
    # Testing the type of an if condition (line 161)
    if_condition_10321 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 161, 4), result_eq_10320)
    # Assigning a type to the variable 'if_condition_10321' (line 161)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 161, 4), 'if_condition_10321', if_condition_10321)
    # SSA begins for if statement (line 161)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    
    # Getting the type of 'n' (line 162)
    n_10322 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 11), 'n')
    int_10323 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 162, 16), 'int')
    # Applying the binary operator '==' (line 162)
    result_eq_10324 = python_operator(stypy.reporting.localization.Localization(__file__, 162, 11), '==', n_10322, int_10323)
    
    # Testing the type of an if condition (line 162)
    if_condition_10325 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 162, 8), result_eq_10324)
    # Assigning a type to the variable 'if_condition_10325' (line 162)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 162, 8), 'if_condition_10325', if_condition_10325)
    # SSA begins for if statement (line 162)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Subscript to a Name (line 163):
    
    # Assigning a Subscript to a Name (line 163):
    
    # Obtaining the type of the subscript
    # Getting the type of 'None' (line 163)
    None_10326 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 20), 'None')
    slice_10327 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 163, 17), None, None, None)
    # Getting the type of 'b1' (line 163)
    b1_10328 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 17), 'b1')
    # Obtaining the member '__getitem__' of a type (line 163)
    getitem___10329 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 163, 17), b1_10328, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 163)
    subscript_call_result_10330 = invoke(stypy.reporting.localization.Localization(__file__, 163, 17), getitem___10329, (None_10326, slice_10327))
    
    # Assigning a type to the variable 'b1' (line 163)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 163, 12), 'b1', subscript_call_result_10330)
    # SSA branch for the else part of an if statement (line 162)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Subscript to a Name (line 165):
    
    # Assigning a Subscript to a Name (line 165):
    
    # Obtaining the type of the subscript
    slice_10331 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 165, 17), None, None, None)
    # Getting the type of 'None' (line 165)
    None_10332 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 23), 'None')
    # Getting the type of 'b1' (line 165)
    b1_10333 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 17), 'b1')
    # Obtaining the member '__getitem__' of a type (line 165)
    getitem___10334 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 165, 17), b1_10333, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 165)
    subscript_call_result_10335 = invoke(stypy.reporting.localization.Localization(__file__, 165, 17), getitem___10334, (slice_10331, None_10332))
    
    # Assigning a type to the variable 'b1' (line 165)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 165, 12), 'b1', subscript_call_result_10335)
    # SSA join for if statement (line 162)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Name to a Name (line 166):
    
    # Assigning a Name to a Name (line 166):
    # Getting the type of 'True' (line 166)
    True_10336 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 18), 'True')
    # Assigning a type to the variable 'b_is_1D' (line 166)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 166, 8), 'b_is_1D', True_10336)
    # SSA join for if statement (line 161)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Getting the type of 'sym_pos' (line 169)
    sym_pos_10337 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 7), 'sym_pos')
    # Testing the type of an if condition (line 169)
    if_condition_10338 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 169, 4), sym_pos_10337)
    # Assigning a type to the variable 'if_condition_10338' (line 169)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 169, 4), 'if_condition_10338', if_condition_10338)
    # SSA begins for if statement (line 169)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Str to a Name (line 170):
    
    # Assigning a Str to a Name (line 170):
    str_10339 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 170, 19), 'str', 'pos')
    # Assigning a type to the variable 'assume_a' (line 170)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 170, 8), 'assume_a', str_10339)
    # SSA join for if statement (line 169)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'assume_a' (line 172)
    assume_a_10340 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 7), 'assume_a')
    
    # Obtaining an instance of the builtin type 'tuple' (line 172)
    tuple_10341 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 172, 24), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 172)
    # Adding element type (line 172)
    str_10342 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 172, 24), 'str', 'gen')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 172, 24), tuple_10341, str_10342)
    # Adding element type (line 172)
    str_10343 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 172, 31), 'str', 'sym')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 172, 24), tuple_10341, str_10343)
    # Adding element type (line 172)
    str_10344 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 172, 38), 'str', 'her')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 172, 24), tuple_10341, str_10344)
    # Adding element type (line 172)
    str_10345 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 172, 45), 'str', 'pos')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 172, 24), tuple_10341, str_10345)
    
    # Applying the binary operator 'notin' (line 172)
    result_contains_10346 = python_operator(stypy.reporting.localization.Localization(__file__, 172, 7), 'notin', assume_a_10340, tuple_10341)
    
    # Testing the type of an if condition (line 172)
    if_condition_10347 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 172, 4), result_contains_10346)
    # Assigning a type to the variable 'if_condition_10347' (line 172)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 172, 4), 'if_condition_10347', if_condition_10347)
    # SSA begins for if statement (line 172)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 173)
    # Processing the call arguments (line 173)
    
    # Call to format(...): (line 173)
    # Processing the call arguments (line 173)
    # Getting the type of 'assume_a' (line 174)
    assume_a_10351 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 35), 'assume_a', False)
    # Processing the call keyword arguments (line 173)
    kwargs_10352 = {}
    str_10349 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 173, 25), 'str', '{} is not a recognized matrix structure')
    # Obtaining the member 'format' of a type (line 173)
    format_10350 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 173, 25), str_10349, 'format')
    # Calling format(args, kwargs) (line 173)
    format_call_result_10353 = invoke(stypy.reporting.localization.Localization(__file__, 173, 25), format_10350, *[assume_a_10351], **kwargs_10352)
    
    # Processing the call keyword arguments (line 173)
    kwargs_10354 = {}
    # Getting the type of 'ValueError' (line 173)
    ValueError_10348 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 173)
    ValueError_call_result_10355 = invoke(stypy.reporting.localization.Localization(__file__, 173, 14), ValueError_10348, *[format_call_result_10353], **kwargs_10354)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 173, 8), ValueError_call_result_10355, 'raise parameter', BaseException)
    # SSA join for if statement (line 172)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Type idiom detected: calculating its left and rigth part (line 177)
    # Getting the type of 'debug' (line 177)
    debug_10356 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 4), 'debug')
    # Getting the type of 'None' (line 177)
    None_10357 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 20), 'None')
    
    (may_be_10358, more_types_in_union_10359) = may_not_be_none(debug_10356, None_10357)

    if may_be_10358:

        if more_types_in_union_10359:
            # Runtime conditional SSA (line 177)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Call to warn(...): (line 178)
        # Processing the call arguments (line 178)
        str_10362 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 178, 22), 'str', 'Use of the "debug" keyword is deprecated and this keyword will be removed in future versions of SciPy.')
        # Getting the type of 'DeprecationWarning' (line 180)
        DeprecationWarning_10363 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 44), 'DeprecationWarning', False)
        # Processing the call keyword arguments (line 178)
        kwargs_10364 = {}
        # Getting the type of 'warnings' (line 178)
        warnings_10360 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 8), 'warnings', False)
        # Obtaining the member 'warn' of a type (line 178)
        warn_10361 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 178, 8), warnings_10360, 'warn')
        # Calling warn(args, kwargs) (line 178)
        warn_call_result_10365 = invoke(stypy.reporting.localization.Localization(__file__, 178, 8), warn_10361, *[str_10362, DeprecationWarning_10363], **kwargs_10364)
        

        if more_types_in_union_10359:
            # SSA join for if statement (line 177)
            module_type_store = module_type_store.join_ssa_context()


    
    
    
    # Getting the type of 'a1' (line 185)
    a1_10366 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 7), 'a1')
    # Obtaining the member 'dtype' of a type (line 185)
    dtype_10367 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 185, 7), a1_10366, 'dtype')
    # Obtaining the member 'char' of a type (line 185)
    char_10368 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 185, 7), dtype_10367, 'char')
    str_10369 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 185, 24), 'str', 'fF')
    # Applying the binary operator 'in' (line 185)
    result_contains_10370 = python_operator(stypy.reporting.localization.Localization(__file__, 185, 7), 'in', char_10368, str_10369)
    
    # Testing the type of an if condition (line 185)
    if_condition_10371 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 185, 4), result_contains_10370)
    # Assigning a type to the variable 'if_condition_10371' (line 185)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 185, 4), 'if_condition_10371', if_condition_10371)
    # SSA begins for if statement (line 185)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 186):
    
    # Assigning a Call to a Name (line 186):
    
    # Call to get_lapack_funcs(...): (line 186)
    # Processing the call arguments (line 186)
    str_10373 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 186, 33), 'str', 'lamch')
    # Processing the call keyword arguments (line 186)
    str_10374 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 186, 48), 'str', 'f')
    keyword_10375 = str_10374
    kwargs_10376 = {'dtype': keyword_10375}
    # Getting the type of 'get_lapack_funcs' (line 186)
    get_lapack_funcs_10372 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 16), 'get_lapack_funcs', False)
    # Calling get_lapack_funcs(args, kwargs) (line 186)
    get_lapack_funcs_call_result_10377 = invoke(stypy.reporting.localization.Localization(__file__, 186, 16), get_lapack_funcs_10372, *[str_10373], **kwargs_10376)
    
    # Assigning a type to the variable 'lamch' (line 186)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 186, 8), 'lamch', get_lapack_funcs_call_result_10377)
    # SSA branch for the else part of an if statement (line 185)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Call to a Name (line 188):
    
    # Assigning a Call to a Name (line 188):
    
    # Call to get_lapack_funcs(...): (line 188)
    # Processing the call arguments (line 188)
    str_10379 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 188, 33), 'str', 'lamch')
    # Processing the call keyword arguments (line 188)
    str_10380 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 188, 48), 'str', 'd')
    keyword_10381 = str_10380
    kwargs_10382 = {'dtype': keyword_10381}
    # Getting the type of 'get_lapack_funcs' (line 188)
    get_lapack_funcs_10378 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 188, 16), 'get_lapack_funcs', False)
    # Calling get_lapack_funcs(args, kwargs) (line 188)
    get_lapack_funcs_call_result_10383 = invoke(stypy.reporting.localization.Localization(__file__, 188, 16), get_lapack_funcs_10378, *[str_10379], **kwargs_10382)
    
    # Assigning a type to the variable 'lamch' (line 188)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 188, 8), 'lamch', get_lapack_funcs_call_result_10383)
    # SSA join for if statement (line 185)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 193):
    
    # Assigning a Call to a Name (line 193):
    
    # Call to get_lapack_funcs(...): (line 193)
    # Processing the call arguments (line 193)
    str_10385 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 193, 29), 'str', 'lange')
    
    # Obtaining an instance of the builtin type 'tuple' (line 193)
    tuple_10386 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 193, 39), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 193)
    # Adding element type (line 193)
    # Getting the type of 'a1' (line 193)
    a1_10387 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 39), 'a1', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 193, 39), tuple_10386, a1_10387)
    
    # Processing the call keyword arguments (line 193)
    kwargs_10388 = {}
    # Getting the type of 'get_lapack_funcs' (line 193)
    get_lapack_funcs_10384 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 12), 'get_lapack_funcs', False)
    # Calling get_lapack_funcs(args, kwargs) (line 193)
    get_lapack_funcs_call_result_10389 = invoke(stypy.reporting.localization.Localization(__file__, 193, 12), get_lapack_funcs_10384, *[str_10385, tuple_10386], **kwargs_10388)
    
    # Assigning a type to the variable 'lange' (line 193)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 193, 4), 'lange', get_lapack_funcs_call_result_10389)
    
    # Getting the type of 'transposed' (line 199)
    transposed_10390 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 7), 'transposed')
    # Testing the type of an if condition (line 199)
    if_condition_10391 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 199, 4), transposed_10390)
    # Assigning a type to the variable 'if_condition_10391' (line 199)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 199, 4), 'if_condition_10391', if_condition_10391)
    # SSA begins for if statement (line 199)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Num to a Name (line 200):
    
    # Assigning a Num to a Name (line 200):
    int_10392 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 200, 16), 'int')
    # Assigning a type to the variable 'trans' (line 200)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 200, 8), 'trans', int_10392)
    
    # Assigning a Str to a Name (line 201):
    
    # Assigning a Str to a Name (line 201):
    str_10393 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 201, 15), 'str', 'I')
    # Assigning a type to the variable 'norm' (line 201)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 201, 8), 'norm', str_10393)
    
    
    # Call to iscomplexobj(...): (line 202)
    # Processing the call arguments (line 202)
    # Getting the type of 'a1' (line 202)
    a1_10396 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 27), 'a1', False)
    # Processing the call keyword arguments (line 202)
    kwargs_10397 = {}
    # Getting the type of 'np' (line 202)
    np_10394 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 11), 'np', False)
    # Obtaining the member 'iscomplexobj' of a type (line 202)
    iscomplexobj_10395 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 202, 11), np_10394, 'iscomplexobj')
    # Calling iscomplexobj(args, kwargs) (line 202)
    iscomplexobj_call_result_10398 = invoke(stypy.reporting.localization.Localization(__file__, 202, 11), iscomplexobj_10395, *[a1_10396], **kwargs_10397)
    
    # Testing the type of an if condition (line 202)
    if_condition_10399 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 202, 8), iscomplexobj_call_result_10398)
    # Assigning a type to the variable 'if_condition_10399' (line 202)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 202, 8), 'if_condition_10399', if_condition_10399)
    # SSA begins for if statement (line 202)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to NotImplementedError(...): (line 203)
    # Processing the call arguments (line 203)
    str_10401 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 203, 38), 'str', 'scipy.linalg.solve can currently not solve a^T x = b or a^H x = b for complex matrices.')
    # Processing the call keyword arguments (line 203)
    kwargs_10402 = {}
    # Getting the type of 'NotImplementedError' (line 203)
    NotImplementedError_10400 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 18), 'NotImplementedError', False)
    # Calling NotImplementedError(args, kwargs) (line 203)
    NotImplementedError_call_result_10403 = invoke(stypy.reporting.localization.Localization(__file__, 203, 18), NotImplementedError_10400, *[str_10401], **kwargs_10402)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 203, 12), NotImplementedError_call_result_10403, 'raise parameter', BaseException)
    # SSA join for if statement (line 202)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA branch for the else part of an if statement (line 199)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Num to a Name (line 207):
    
    # Assigning a Num to a Name (line 207):
    int_10404 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 207, 16), 'int')
    # Assigning a type to the variable 'trans' (line 207)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 207, 8), 'trans', int_10404)
    
    # Assigning a Str to a Name (line 208):
    
    # Assigning a Str to a Name (line 208):
    str_10405 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 208, 15), 'str', '1')
    # Assigning a type to the variable 'norm' (line 208)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 208, 8), 'norm', str_10405)
    # SSA join for if statement (line 199)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 210):
    
    # Assigning a Call to a Name (line 210):
    
    # Call to lange(...): (line 210)
    # Processing the call arguments (line 210)
    # Getting the type of 'norm' (line 210)
    norm_10407 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 18), 'norm', False)
    # Getting the type of 'a1' (line 210)
    a1_10408 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 24), 'a1', False)
    # Processing the call keyword arguments (line 210)
    kwargs_10409 = {}
    # Getting the type of 'lange' (line 210)
    lange_10406 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 12), 'lange', False)
    # Calling lange(args, kwargs) (line 210)
    lange_call_result_10410 = invoke(stypy.reporting.localization.Localization(__file__, 210, 12), lange_10406, *[norm_10407, a1_10408], **kwargs_10409)
    
    # Assigning a type to the variable 'anorm' (line 210)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 210, 4), 'anorm', lange_call_result_10410)
    
    
    # Getting the type of 'assume_a' (line 213)
    assume_a_10411 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 7), 'assume_a')
    str_10412 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 213, 19), 'str', 'gen')
    # Applying the binary operator '==' (line 213)
    result_eq_10413 = python_operator(stypy.reporting.localization.Localization(__file__, 213, 7), '==', assume_a_10411, str_10412)
    
    # Testing the type of an if condition (line 213)
    if_condition_10414 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 213, 4), result_eq_10413)
    # Assigning a type to the variable 'if_condition_10414' (line 213)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 213, 4), 'if_condition_10414', if_condition_10414)
    # SSA begins for if statement (line 213)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Tuple (line 214):
    
    # Assigning a Subscript to a Name (line 214):
    
    # Obtaining the type of the subscript
    int_10415 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 214, 8), 'int')
    
    # Call to get_lapack_funcs(...): (line 214)
    # Processing the call arguments (line 214)
    
    # Obtaining an instance of the builtin type 'tuple' (line 214)
    tuple_10417 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 214, 48), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 214)
    # Adding element type (line 214)
    str_10418 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 214, 48), 'str', 'gecon')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 214, 48), tuple_10417, str_10418)
    # Adding element type (line 214)
    str_10419 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 214, 57), 'str', 'getrf')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 214, 48), tuple_10417, str_10419)
    # Adding element type (line 214)
    str_10420 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 214, 66), 'str', 'getrs')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 214, 48), tuple_10417, str_10420)
    
    
    # Obtaining an instance of the builtin type 'tuple' (line 215)
    tuple_10421 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 215, 48), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 215)
    # Adding element type (line 215)
    # Getting the type of 'a1' (line 215)
    a1_10422 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 215, 48), 'a1', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 215, 48), tuple_10421, a1_10422)
    # Adding element type (line 215)
    # Getting the type of 'b1' (line 215)
    b1_10423 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 215, 52), 'b1', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 215, 48), tuple_10421, b1_10423)
    
    # Processing the call keyword arguments (line 214)
    kwargs_10424 = {}
    # Getting the type of 'get_lapack_funcs' (line 214)
    get_lapack_funcs_10416 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 30), 'get_lapack_funcs', False)
    # Calling get_lapack_funcs(args, kwargs) (line 214)
    get_lapack_funcs_call_result_10425 = invoke(stypy.reporting.localization.Localization(__file__, 214, 30), get_lapack_funcs_10416, *[tuple_10417, tuple_10421], **kwargs_10424)
    
    # Obtaining the member '__getitem__' of a type (line 214)
    getitem___10426 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 214, 8), get_lapack_funcs_call_result_10425, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 214)
    subscript_call_result_10427 = invoke(stypy.reporting.localization.Localization(__file__, 214, 8), getitem___10426, int_10415)
    
    # Assigning a type to the variable 'tuple_var_assignment_10024' (line 214)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 214, 8), 'tuple_var_assignment_10024', subscript_call_result_10427)
    
    # Assigning a Subscript to a Name (line 214):
    
    # Obtaining the type of the subscript
    int_10428 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 214, 8), 'int')
    
    # Call to get_lapack_funcs(...): (line 214)
    # Processing the call arguments (line 214)
    
    # Obtaining an instance of the builtin type 'tuple' (line 214)
    tuple_10430 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 214, 48), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 214)
    # Adding element type (line 214)
    str_10431 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 214, 48), 'str', 'gecon')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 214, 48), tuple_10430, str_10431)
    # Adding element type (line 214)
    str_10432 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 214, 57), 'str', 'getrf')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 214, 48), tuple_10430, str_10432)
    # Adding element type (line 214)
    str_10433 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 214, 66), 'str', 'getrs')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 214, 48), tuple_10430, str_10433)
    
    
    # Obtaining an instance of the builtin type 'tuple' (line 215)
    tuple_10434 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 215, 48), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 215)
    # Adding element type (line 215)
    # Getting the type of 'a1' (line 215)
    a1_10435 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 215, 48), 'a1', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 215, 48), tuple_10434, a1_10435)
    # Adding element type (line 215)
    # Getting the type of 'b1' (line 215)
    b1_10436 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 215, 52), 'b1', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 215, 48), tuple_10434, b1_10436)
    
    # Processing the call keyword arguments (line 214)
    kwargs_10437 = {}
    # Getting the type of 'get_lapack_funcs' (line 214)
    get_lapack_funcs_10429 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 30), 'get_lapack_funcs', False)
    # Calling get_lapack_funcs(args, kwargs) (line 214)
    get_lapack_funcs_call_result_10438 = invoke(stypy.reporting.localization.Localization(__file__, 214, 30), get_lapack_funcs_10429, *[tuple_10430, tuple_10434], **kwargs_10437)
    
    # Obtaining the member '__getitem__' of a type (line 214)
    getitem___10439 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 214, 8), get_lapack_funcs_call_result_10438, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 214)
    subscript_call_result_10440 = invoke(stypy.reporting.localization.Localization(__file__, 214, 8), getitem___10439, int_10428)
    
    # Assigning a type to the variable 'tuple_var_assignment_10025' (line 214)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 214, 8), 'tuple_var_assignment_10025', subscript_call_result_10440)
    
    # Assigning a Subscript to a Name (line 214):
    
    # Obtaining the type of the subscript
    int_10441 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 214, 8), 'int')
    
    # Call to get_lapack_funcs(...): (line 214)
    # Processing the call arguments (line 214)
    
    # Obtaining an instance of the builtin type 'tuple' (line 214)
    tuple_10443 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 214, 48), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 214)
    # Adding element type (line 214)
    str_10444 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 214, 48), 'str', 'gecon')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 214, 48), tuple_10443, str_10444)
    # Adding element type (line 214)
    str_10445 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 214, 57), 'str', 'getrf')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 214, 48), tuple_10443, str_10445)
    # Adding element type (line 214)
    str_10446 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 214, 66), 'str', 'getrs')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 214, 48), tuple_10443, str_10446)
    
    
    # Obtaining an instance of the builtin type 'tuple' (line 215)
    tuple_10447 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 215, 48), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 215)
    # Adding element type (line 215)
    # Getting the type of 'a1' (line 215)
    a1_10448 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 215, 48), 'a1', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 215, 48), tuple_10447, a1_10448)
    # Adding element type (line 215)
    # Getting the type of 'b1' (line 215)
    b1_10449 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 215, 52), 'b1', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 215, 48), tuple_10447, b1_10449)
    
    # Processing the call keyword arguments (line 214)
    kwargs_10450 = {}
    # Getting the type of 'get_lapack_funcs' (line 214)
    get_lapack_funcs_10442 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 30), 'get_lapack_funcs', False)
    # Calling get_lapack_funcs(args, kwargs) (line 214)
    get_lapack_funcs_call_result_10451 = invoke(stypy.reporting.localization.Localization(__file__, 214, 30), get_lapack_funcs_10442, *[tuple_10443, tuple_10447], **kwargs_10450)
    
    # Obtaining the member '__getitem__' of a type (line 214)
    getitem___10452 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 214, 8), get_lapack_funcs_call_result_10451, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 214)
    subscript_call_result_10453 = invoke(stypy.reporting.localization.Localization(__file__, 214, 8), getitem___10452, int_10441)
    
    # Assigning a type to the variable 'tuple_var_assignment_10026' (line 214)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 214, 8), 'tuple_var_assignment_10026', subscript_call_result_10453)
    
    # Assigning a Name to a Name (line 214):
    # Getting the type of 'tuple_var_assignment_10024' (line 214)
    tuple_var_assignment_10024_10454 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 8), 'tuple_var_assignment_10024')
    # Assigning a type to the variable 'gecon' (line 214)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 214, 8), 'gecon', tuple_var_assignment_10024_10454)
    
    # Assigning a Name to a Name (line 214):
    # Getting the type of 'tuple_var_assignment_10025' (line 214)
    tuple_var_assignment_10025_10455 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 8), 'tuple_var_assignment_10025')
    # Assigning a type to the variable 'getrf' (line 214)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 214, 15), 'getrf', tuple_var_assignment_10025_10455)
    
    # Assigning a Name to a Name (line 214):
    # Getting the type of 'tuple_var_assignment_10026' (line 214)
    tuple_var_assignment_10026_10456 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 8), 'tuple_var_assignment_10026')
    # Assigning a type to the variable 'getrs' (line 214)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 214, 22), 'getrs', tuple_var_assignment_10026_10456)
    
    # Assigning a Call to a Tuple (line 216):
    
    # Assigning a Subscript to a Name (line 216):
    
    # Obtaining the type of the subscript
    int_10457 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 216, 8), 'int')
    
    # Call to getrf(...): (line 216)
    # Processing the call arguments (line 216)
    # Getting the type of 'a1' (line 216)
    a1_10459 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 216, 31), 'a1', False)
    # Processing the call keyword arguments (line 216)
    # Getting the type of 'overwrite_a' (line 216)
    overwrite_a_10460 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 216, 47), 'overwrite_a', False)
    keyword_10461 = overwrite_a_10460
    kwargs_10462 = {'overwrite_a': keyword_10461}
    # Getting the type of 'getrf' (line 216)
    getrf_10458 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 216, 25), 'getrf', False)
    # Calling getrf(args, kwargs) (line 216)
    getrf_call_result_10463 = invoke(stypy.reporting.localization.Localization(__file__, 216, 25), getrf_10458, *[a1_10459], **kwargs_10462)
    
    # Obtaining the member '__getitem__' of a type (line 216)
    getitem___10464 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 216, 8), getrf_call_result_10463, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 216)
    subscript_call_result_10465 = invoke(stypy.reporting.localization.Localization(__file__, 216, 8), getitem___10464, int_10457)
    
    # Assigning a type to the variable 'tuple_var_assignment_10027' (line 216)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 216, 8), 'tuple_var_assignment_10027', subscript_call_result_10465)
    
    # Assigning a Subscript to a Name (line 216):
    
    # Obtaining the type of the subscript
    int_10466 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 216, 8), 'int')
    
    # Call to getrf(...): (line 216)
    # Processing the call arguments (line 216)
    # Getting the type of 'a1' (line 216)
    a1_10468 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 216, 31), 'a1', False)
    # Processing the call keyword arguments (line 216)
    # Getting the type of 'overwrite_a' (line 216)
    overwrite_a_10469 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 216, 47), 'overwrite_a', False)
    keyword_10470 = overwrite_a_10469
    kwargs_10471 = {'overwrite_a': keyword_10470}
    # Getting the type of 'getrf' (line 216)
    getrf_10467 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 216, 25), 'getrf', False)
    # Calling getrf(args, kwargs) (line 216)
    getrf_call_result_10472 = invoke(stypy.reporting.localization.Localization(__file__, 216, 25), getrf_10467, *[a1_10468], **kwargs_10471)
    
    # Obtaining the member '__getitem__' of a type (line 216)
    getitem___10473 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 216, 8), getrf_call_result_10472, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 216)
    subscript_call_result_10474 = invoke(stypy.reporting.localization.Localization(__file__, 216, 8), getitem___10473, int_10466)
    
    # Assigning a type to the variable 'tuple_var_assignment_10028' (line 216)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 216, 8), 'tuple_var_assignment_10028', subscript_call_result_10474)
    
    # Assigning a Subscript to a Name (line 216):
    
    # Obtaining the type of the subscript
    int_10475 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 216, 8), 'int')
    
    # Call to getrf(...): (line 216)
    # Processing the call arguments (line 216)
    # Getting the type of 'a1' (line 216)
    a1_10477 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 216, 31), 'a1', False)
    # Processing the call keyword arguments (line 216)
    # Getting the type of 'overwrite_a' (line 216)
    overwrite_a_10478 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 216, 47), 'overwrite_a', False)
    keyword_10479 = overwrite_a_10478
    kwargs_10480 = {'overwrite_a': keyword_10479}
    # Getting the type of 'getrf' (line 216)
    getrf_10476 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 216, 25), 'getrf', False)
    # Calling getrf(args, kwargs) (line 216)
    getrf_call_result_10481 = invoke(stypy.reporting.localization.Localization(__file__, 216, 25), getrf_10476, *[a1_10477], **kwargs_10480)
    
    # Obtaining the member '__getitem__' of a type (line 216)
    getitem___10482 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 216, 8), getrf_call_result_10481, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 216)
    subscript_call_result_10483 = invoke(stypy.reporting.localization.Localization(__file__, 216, 8), getitem___10482, int_10475)
    
    # Assigning a type to the variable 'tuple_var_assignment_10029' (line 216)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 216, 8), 'tuple_var_assignment_10029', subscript_call_result_10483)
    
    # Assigning a Name to a Name (line 216):
    # Getting the type of 'tuple_var_assignment_10027' (line 216)
    tuple_var_assignment_10027_10484 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 216, 8), 'tuple_var_assignment_10027')
    # Assigning a type to the variable 'lu' (line 216)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 216, 8), 'lu', tuple_var_assignment_10027_10484)
    
    # Assigning a Name to a Name (line 216):
    # Getting the type of 'tuple_var_assignment_10028' (line 216)
    tuple_var_assignment_10028_10485 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 216, 8), 'tuple_var_assignment_10028')
    # Assigning a type to the variable 'ipvt' (line 216)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 216, 12), 'ipvt', tuple_var_assignment_10028_10485)
    
    # Assigning a Name to a Name (line 216):
    # Getting the type of 'tuple_var_assignment_10029' (line 216)
    tuple_var_assignment_10029_10486 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 216, 8), 'tuple_var_assignment_10029')
    # Assigning a type to the variable 'info' (line 216)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 216, 18), 'info', tuple_var_assignment_10029_10486)
    
    # Call to _solve_check(...): (line 217)
    # Processing the call arguments (line 217)
    # Getting the type of 'n' (line 217)
    n_10488 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 21), 'n', False)
    # Getting the type of 'info' (line 217)
    info_10489 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 24), 'info', False)
    # Processing the call keyword arguments (line 217)
    kwargs_10490 = {}
    # Getting the type of '_solve_check' (line 217)
    _solve_check_10487 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 8), '_solve_check', False)
    # Calling _solve_check(args, kwargs) (line 217)
    _solve_check_call_result_10491 = invoke(stypy.reporting.localization.Localization(__file__, 217, 8), _solve_check_10487, *[n_10488, info_10489], **kwargs_10490)
    
    
    # Assigning a Call to a Tuple (line 218):
    
    # Assigning a Subscript to a Name (line 218):
    
    # Obtaining the type of the subscript
    int_10492 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 218, 8), 'int')
    
    # Call to getrs(...): (line 218)
    # Processing the call arguments (line 218)
    # Getting the type of 'lu' (line 218)
    lu_10494 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 218, 24), 'lu', False)
    # Getting the type of 'ipvt' (line 218)
    ipvt_10495 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 218, 28), 'ipvt', False)
    # Getting the type of 'b1' (line 218)
    b1_10496 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 218, 34), 'b1', False)
    # Processing the call keyword arguments (line 218)
    # Getting the type of 'trans' (line 219)
    trans_10497 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 219, 30), 'trans', False)
    keyword_10498 = trans_10497
    # Getting the type of 'overwrite_b' (line 219)
    overwrite_b_10499 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 219, 49), 'overwrite_b', False)
    keyword_10500 = overwrite_b_10499
    kwargs_10501 = {'trans': keyword_10498, 'overwrite_b': keyword_10500}
    # Getting the type of 'getrs' (line 218)
    getrs_10493 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 218, 18), 'getrs', False)
    # Calling getrs(args, kwargs) (line 218)
    getrs_call_result_10502 = invoke(stypy.reporting.localization.Localization(__file__, 218, 18), getrs_10493, *[lu_10494, ipvt_10495, b1_10496], **kwargs_10501)
    
    # Obtaining the member '__getitem__' of a type (line 218)
    getitem___10503 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 218, 8), getrs_call_result_10502, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 218)
    subscript_call_result_10504 = invoke(stypy.reporting.localization.Localization(__file__, 218, 8), getitem___10503, int_10492)
    
    # Assigning a type to the variable 'tuple_var_assignment_10030' (line 218)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 218, 8), 'tuple_var_assignment_10030', subscript_call_result_10504)
    
    # Assigning a Subscript to a Name (line 218):
    
    # Obtaining the type of the subscript
    int_10505 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 218, 8), 'int')
    
    # Call to getrs(...): (line 218)
    # Processing the call arguments (line 218)
    # Getting the type of 'lu' (line 218)
    lu_10507 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 218, 24), 'lu', False)
    # Getting the type of 'ipvt' (line 218)
    ipvt_10508 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 218, 28), 'ipvt', False)
    # Getting the type of 'b1' (line 218)
    b1_10509 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 218, 34), 'b1', False)
    # Processing the call keyword arguments (line 218)
    # Getting the type of 'trans' (line 219)
    trans_10510 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 219, 30), 'trans', False)
    keyword_10511 = trans_10510
    # Getting the type of 'overwrite_b' (line 219)
    overwrite_b_10512 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 219, 49), 'overwrite_b', False)
    keyword_10513 = overwrite_b_10512
    kwargs_10514 = {'trans': keyword_10511, 'overwrite_b': keyword_10513}
    # Getting the type of 'getrs' (line 218)
    getrs_10506 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 218, 18), 'getrs', False)
    # Calling getrs(args, kwargs) (line 218)
    getrs_call_result_10515 = invoke(stypy.reporting.localization.Localization(__file__, 218, 18), getrs_10506, *[lu_10507, ipvt_10508, b1_10509], **kwargs_10514)
    
    # Obtaining the member '__getitem__' of a type (line 218)
    getitem___10516 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 218, 8), getrs_call_result_10515, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 218)
    subscript_call_result_10517 = invoke(stypy.reporting.localization.Localization(__file__, 218, 8), getitem___10516, int_10505)
    
    # Assigning a type to the variable 'tuple_var_assignment_10031' (line 218)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 218, 8), 'tuple_var_assignment_10031', subscript_call_result_10517)
    
    # Assigning a Name to a Name (line 218):
    # Getting the type of 'tuple_var_assignment_10030' (line 218)
    tuple_var_assignment_10030_10518 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 218, 8), 'tuple_var_assignment_10030')
    # Assigning a type to the variable 'x' (line 218)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 218, 8), 'x', tuple_var_assignment_10030_10518)
    
    # Assigning a Name to a Name (line 218):
    # Getting the type of 'tuple_var_assignment_10031' (line 218)
    tuple_var_assignment_10031_10519 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 218, 8), 'tuple_var_assignment_10031')
    # Assigning a type to the variable 'info' (line 218)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 218, 11), 'info', tuple_var_assignment_10031_10519)
    
    # Call to _solve_check(...): (line 220)
    # Processing the call arguments (line 220)
    # Getting the type of 'n' (line 220)
    n_10521 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 220, 21), 'n', False)
    # Getting the type of 'info' (line 220)
    info_10522 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 220, 24), 'info', False)
    # Processing the call keyword arguments (line 220)
    kwargs_10523 = {}
    # Getting the type of '_solve_check' (line 220)
    _solve_check_10520 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 220, 8), '_solve_check', False)
    # Calling _solve_check(args, kwargs) (line 220)
    _solve_check_call_result_10524 = invoke(stypy.reporting.localization.Localization(__file__, 220, 8), _solve_check_10520, *[n_10521, info_10522], **kwargs_10523)
    
    
    # Assigning a Call to a Tuple (line 221):
    
    # Assigning a Subscript to a Name (line 221):
    
    # Obtaining the type of the subscript
    int_10525 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 221, 8), 'int')
    
    # Call to gecon(...): (line 221)
    # Processing the call arguments (line 221)
    # Getting the type of 'lu' (line 221)
    lu_10527 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 28), 'lu', False)
    # Getting the type of 'anorm' (line 221)
    anorm_10528 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 32), 'anorm', False)
    # Processing the call keyword arguments (line 221)
    # Getting the type of 'norm' (line 221)
    norm_10529 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 44), 'norm', False)
    keyword_10530 = norm_10529
    kwargs_10531 = {'norm': keyword_10530}
    # Getting the type of 'gecon' (line 221)
    gecon_10526 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 22), 'gecon', False)
    # Calling gecon(args, kwargs) (line 221)
    gecon_call_result_10532 = invoke(stypy.reporting.localization.Localization(__file__, 221, 22), gecon_10526, *[lu_10527, anorm_10528], **kwargs_10531)
    
    # Obtaining the member '__getitem__' of a type (line 221)
    getitem___10533 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 221, 8), gecon_call_result_10532, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 221)
    subscript_call_result_10534 = invoke(stypy.reporting.localization.Localization(__file__, 221, 8), getitem___10533, int_10525)
    
    # Assigning a type to the variable 'tuple_var_assignment_10032' (line 221)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 221, 8), 'tuple_var_assignment_10032', subscript_call_result_10534)
    
    # Assigning a Subscript to a Name (line 221):
    
    # Obtaining the type of the subscript
    int_10535 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 221, 8), 'int')
    
    # Call to gecon(...): (line 221)
    # Processing the call arguments (line 221)
    # Getting the type of 'lu' (line 221)
    lu_10537 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 28), 'lu', False)
    # Getting the type of 'anorm' (line 221)
    anorm_10538 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 32), 'anorm', False)
    # Processing the call keyword arguments (line 221)
    # Getting the type of 'norm' (line 221)
    norm_10539 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 44), 'norm', False)
    keyword_10540 = norm_10539
    kwargs_10541 = {'norm': keyword_10540}
    # Getting the type of 'gecon' (line 221)
    gecon_10536 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 22), 'gecon', False)
    # Calling gecon(args, kwargs) (line 221)
    gecon_call_result_10542 = invoke(stypy.reporting.localization.Localization(__file__, 221, 22), gecon_10536, *[lu_10537, anorm_10538], **kwargs_10541)
    
    # Obtaining the member '__getitem__' of a type (line 221)
    getitem___10543 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 221, 8), gecon_call_result_10542, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 221)
    subscript_call_result_10544 = invoke(stypy.reporting.localization.Localization(__file__, 221, 8), getitem___10543, int_10535)
    
    # Assigning a type to the variable 'tuple_var_assignment_10033' (line 221)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 221, 8), 'tuple_var_assignment_10033', subscript_call_result_10544)
    
    # Assigning a Name to a Name (line 221):
    # Getting the type of 'tuple_var_assignment_10032' (line 221)
    tuple_var_assignment_10032_10545 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 8), 'tuple_var_assignment_10032')
    # Assigning a type to the variable 'rcond' (line 221)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 221, 8), 'rcond', tuple_var_assignment_10032_10545)
    
    # Assigning a Name to a Name (line 221):
    # Getting the type of 'tuple_var_assignment_10033' (line 221)
    tuple_var_assignment_10033_10546 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 8), 'tuple_var_assignment_10033')
    # Assigning a type to the variable 'info' (line 221)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 221, 15), 'info', tuple_var_assignment_10033_10546)
    # SSA branch for the else part of an if statement (line 213)
    module_type_store.open_ssa_branch('else')
    
    
    # Getting the type of 'assume_a' (line 223)
    assume_a_10547 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 9), 'assume_a')
    str_10548 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 223, 21), 'str', 'her')
    # Applying the binary operator '==' (line 223)
    result_eq_10549 = python_operator(stypy.reporting.localization.Localization(__file__, 223, 9), '==', assume_a_10547, str_10548)
    
    # Testing the type of an if condition (line 223)
    if_condition_10550 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 223, 9), result_eq_10549)
    # Assigning a type to the variable 'if_condition_10550' (line 223)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 223, 9), 'if_condition_10550', if_condition_10550)
    # SSA begins for if statement (line 223)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Tuple (line 224):
    
    # Assigning a Subscript to a Name (line 224):
    
    # Obtaining the type of the subscript
    int_10551 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 224, 8), 'int')
    
    # Call to get_lapack_funcs(...): (line 224)
    # Processing the call arguments (line 224)
    
    # Obtaining an instance of the builtin type 'tuple' (line 224)
    tuple_10553 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 224, 49), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 224)
    # Adding element type (line 224)
    str_10554 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 224, 49), 'str', 'hecon')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 224, 49), tuple_10553, str_10554)
    # Adding element type (line 224)
    str_10555 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 224, 58), 'str', 'hesv')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 224, 49), tuple_10553, str_10555)
    # Adding element type (line 224)
    str_10556 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 224, 66), 'str', 'hesv_lwork')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 224, 49), tuple_10553, str_10556)
    
    
    # Obtaining an instance of the builtin type 'tuple' (line 225)
    tuple_10557 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 225, 49), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 225)
    # Adding element type (line 225)
    # Getting the type of 'a1' (line 225)
    a1_10558 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 225, 49), 'a1', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 225, 49), tuple_10557, a1_10558)
    # Adding element type (line 225)
    # Getting the type of 'b1' (line 225)
    b1_10559 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 225, 53), 'b1', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 225, 49), tuple_10557, b1_10559)
    
    # Processing the call keyword arguments (line 224)
    kwargs_10560 = {}
    # Getting the type of 'get_lapack_funcs' (line 224)
    get_lapack_funcs_10552 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 224, 31), 'get_lapack_funcs', False)
    # Calling get_lapack_funcs(args, kwargs) (line 224)
    get_lapack_funcs_call_result_10561 = invoke(stypy.reporting.localization.Localization(__file__, 224, 31), get_lapack_funcs_10552, *[tuple_10553, tuple_10557], **kwargs_10560)
    
    # Obtaining the member '__getitem__' of a type (line 224)
    getitem___10562 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 224, 8), get_lapack_funcs_call_result_10561, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 224)
    subscript_call_result_10563 = invoke(stypy.reporting.localization.Localization(__file__, 224, 8), getitem___10562, int_10551)
    
    # Assigning a type to the variable 'tuple_var_assignment_10034' (line 224)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 224, 8), 'tuple_var_assignment_10034', subscript_call_result_10563)
    
    # Assigning a Subscript to a Name (line 224):
    
    # Obtaining the type of the subscript
    int_10564 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 224, 8), 'int')
    
    # Call to get_lapack_funcs(...): (line 224)
    # Processing the call arguments (line 224)
    
    # Obtaining an instance of the builtin type 'tuple' (line 224)
    tuple_10566 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 224, 49), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 224)
    # Adding element type (line 224)
    str_10567 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 224, 49), 'str', 'hecon')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 224, 49), tuple_10566, str_10567)
    # Adding element type (line 224)
    str_10568 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 224, 58), 'str', 'hesv')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 224, 49), tuple_10566, str_10568)
    # Adding element type (line 224)
    str_10569 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 224, 66), 'str', 'hesv_lwork')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 224, 49), tuple_10566, str_10569)
    
    
    # Obtaining an instance of the builtin type 'tuple' (line 225)
    tuple_10570 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 225, 49), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 225)
    # Adding element type (line 225)
    # Getting the type of 'a1' (line 225)
    a1_10571 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 225, 49), 'a1', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 225, 49), tuple_10570, a1_10571)
    # Adding element type (line 225)
    # Getting the type of 'b1' (line 225)
    b1_10572 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 225, 53), 'b1', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 225, 49), tuple_10570, b1_10572)
    
    # Processing the call keyword arguments (line 224)
    kwargs_10573 = {}
    # Getting the type of 'get_lapack_funcs' (line 224)
    get_lapack_funcs_10565 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 224, 31), 'get_lapack_funcs', False)
    # Calling get_lapack_funcs(args, kwargs) (line 224)
    get_lapack_funcs_call_result_10574 = invoke(stypy.reporting.localization.Localization(__file__, 224, 31), get_lapack_funcs_10565, *[tuple_10566, tuple_10570], **kwargs_10573)
    
    # Obtaining the member '__getitem__' of a type (line 224)
    getitem___10575 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 224, 8), get_lapack_funcs_call_result_10574, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 224)
    subscript_call_result_10576 = invoke(stypy.reporting.localization.Localization(__file__, 224, 8), getitem___10575, int_10564)
    
    # Assigning a type to the variable 'tuple_var_assignment_10035' (line 224)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 224, 8), 'tuple_var_assignment_10035', subscript_call_result_10576)
    
    # Assigning a Subscript to a Name (line 224):
    
    # Obtaining the type of the subscript
    int_10577 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 224, 8), 'int')
    
    # Call to get_lapack_funcs(...): (line 224)
    # Processing the call arguments (line 224)
    
    # Obtaining an instance of the builtin type 'tuple' (line 224)
    tuple_10579 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 224, 49), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 224)
    # Adding element type (line 224)
    str_10580 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 224, 49), 'str', 'hecon')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 224, 49), tuple_10579, str_10580)
    # Adding element type (line 224)
    str_10581 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 224, 58), 'str', 'hesv')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 224, 49), tuple_10579, str_10581)
    # Adding element type (line 224)
    str_10582 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 224, 66), 'str', 'hesv_lwork')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 224, 49), tuple_10579, str_10582)
    
    
    # Obtaining an instance of the builtin type 'tuple' (line 225)
    tuple_10583 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 225, 49), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 225)
    # Adding element type (line 225)
    # Getting the type of 'a1' (line 225)
    a1_10584 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 225, 49), 'a1', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 225, 49), tuple_10583, a1_10584)
    # Adding element type (line 225)
    # Getting the type of 'b1' (line 225)
    b1_10585 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 225, 53), 'b1', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 225, 49), tuple_10583, b1_10585)
    
    # Processing the call keyword arguments (line 224)
    kwargs_10586 = {}
    # Getting the type of 'get_lapack_funcs' (line 224)
    get_lapack_funcs_10578 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 224, 31), 'get_lapack_funcs', False)
    # Calling get_lapack_funcs(args, kwargs) (line 224)
    get_lapack_funcs_call_result_10587 = invoke(stypy.reporting.localization.Localization(__file__, 224, 31), get_lapack_funcs_10578, *[tuple_10579, tuple_10583], **kwargs_10586)
    
    # Obtaining the member '__getitem__' of a type (line 224)
    getitem___10588 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 224, 8), get_lapack_funcs_call_result_10587, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 224)
    subscript_call_result_10589 = invoke(stypy.reporting.localization.Localization(__file__, 224, 8), getitem___10588, int_10577)
    
    # Assigning a type to the variable 'tuple_var_assignment_10036' (line 224)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 224, 8), 'tuple_var_assignment_10036', subscript_call_result_10589)
    
    # Assigning a Name to a Name (line 224):
    # Getting the type of 'tuple_var_assignment_10034' (line 224)
    tuple_var_assignment_10034_10590 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 224, 8), 'tuple_var_assignment_10034')
    # Assigning a type to the variable 'hecon' (line 224)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 224, 8), 'hecon', tuple_var_assignment_10034_10590)
    
    # Assigning a Name to a Name (line 224):
    # Getting the type of 'tuple_var_assignment_10035' (line 224)
    tuple_var_assignment_10035_10591 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 224, 8), 'tuple_var_assignment_10035')
    # Assigning a type to the variable 'hesv' (line 224)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 224, 15), 'hesv', tuple_var_assignment_10035_10591)
    
    # Assigning a Name to a Name (line 224):
    # Getting the type of 'tuple_var_assignment_10036' (line 224)
    tuple_var_assignment_10036_10592 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 224, 8), 'tuple_var_assignment_10036')
    # Assigning a type to the variable 'hesv_lw' (line 224)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 224, 21), 'hesv_lw', tuple_var_assignment_10036_10592)
    
    # Assigning a Call to a Name (line 226):
    
    # Assigning a Call to a Name (line 226):
    
    # Call to _compute_lwork(...): (line 226)
    # Processing the call arguments (line 226)
    # Getting the type of 'hesv_lw' (line 226)
    hesv_lw_10594 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 31), 'hesv_lw', False)
    # Getting the type of 'n' (line 226)
    n_10595 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 40), 'n', False)
    # Getting the type of 'lower' (line 226)
    lower_10596 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 43), 'lower', False)
    # Processing the call keyword arguments (line 226)
    kwargs_10597 = {}
    # Getting the type of '_compute_lwork' (line 226)
    _compute_lwork_10593 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 16), '_compute_lwork', False)
    # Calling _compute_lwork(args, kwargs) (line 226)
    _compute_lwork_call_result_10598 = invoke(stypy.reporting.localization.Localization(__file__, 226, 16), _compute_lwork_10593, *[hesv_lw_10594, n_10595, lower_10596], **kwargs_10597)
    
    # Assigning a type to the variable 'lwork' (line 226)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 226, 8), 'lwork', _compute_lwork_call_result_10598)
    
    # Assigning a Call to a Tuple (line 227):
    
    # Assigning a Subscript to a Name (line 227):
    
    # Obtaining the type of the subscript
    int_10599 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 227, 8), 'int')
    
    # Call to hesv(...): (line 227)
    # Processing the call arguments (line 227)
    # Getting the type of 'a1' (line 227)
    a1_10601 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 33), 'a1', False)
    # Getting the type of 'b1' (line 227)
    b1_10602 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 37), 'b1', False)
    # Processing the call keyword arguments (line 227)
    # Getting the type of 'lwork' (line 227)
    lwork_10603 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 47), 'lwork', False)
    keyword_10604 = lwork_10603
    # Getting the type of 'lower' (line 228)
    lower_10605 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 39), 'lower', False)
    keyword_10606 = lower_10605
    # Getting the type of 'overwrite_a' (line 229)
    overwrite_a_10607 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 229, 45), 'overwrite_a', False)
    keyword_10608 = overwrite_a_10607
    # Getting the type of 'overwrite_b' (line 230)
    overwrite_b_10609 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 45), 'overwrite_b', False)
    keyword_10610 = overwrite_b_10609
    kwargs_10611 = {'lower': keyword_10606, 'overwrite_a': keyword_10608, 'lwork': keyword_10604, 'overwrite_b': keyword_10610}
    # Getting the type of 'hesv' (line 227)
    hesv_10600 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 28), 'hesv', False)
    # Calling hesv(args, kwargs) (line 227)
    hesv_call_result_10612 = invoke(stypy.reporting.localization.Localization(__file__, 227, 28), hesv_10600, *[a1_10601, b1_10602], **kwargs_10611)
    
    # Obtaining the member '__getitem__' of a type (line 227)
    getitem___10613 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 227, 8), hesv_call_result_10612, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 227)
    subscript_call_result_10614 = invoke(stypy.reporting.localization.Localization(__file__, 227, 8), getitem___10613, int_10599)
    
    # Assigning a type to the variable 'tuple_var_assignment_10037' (line 227)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 227, 8), 'tuple_var_assignment_10037', subscript_call_result_10614)
    
    # Assigning a Subscript to a Name (line 227):
    
    # Obtaining the type of the subscript
    int_10615 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 227, 8), 'int')
    
    # Call to hesv(...): (line 227)
    # Processing the call arguments (line 227)
    # Getting the type of 'a1' (line 227)
    a1_10617 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 33), 'a1', False)
    # Getting the type of 'b1' (line 227)
    b1_10618 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 37), 'b1', False)
    # Processing the call keyword arguments (line 227)
    # Getting the type of 'lwork' (line 227)
    lwork_10619 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 47), 'lwork', False)
    keyword_10620 = lwork_10619
    # Getting the type of 'lower' (line 228)
    lower_10621 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 39), 'lower', False)
    keyword_10622 = lower_10621
    # Getting the type of 'overwrite_a' (line 229)
    overwrite_a_10623 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 229, 45), 'overwrite_a', False)
    keyword_10624 = overwrite_a_10623
    # Getting the type of 'overwrite_b' (line 230)
    overwrite_b_10625 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 45), 'overwrite_b', False)
    keyword_10626 = overwrite_b_10625
    kwargs_10627 = {'lower': keyword_10622, 'overwrite_a': keyword_10624, 'lwork': keyword_10620, 'overwrite_b': keyword_10626}
    # Getting the type of 'hesv' (line 227)
    hesv_10616 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 28), 'hesv', False)
    # Calling hesv(args, kwargs) (line 227)
    hesv_call_result_10628 = invoke(stypy.reporting.localization.Localization(__file__, 227, 28), hesv_10616, *[a1_10617, b1_10618], **kwargs_10627)
    
    # Obtaining the member '__getitem__' of a type (line 227)
    getitem___10629 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 227, 8), hesv_call_result_10628, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 227)
    subscript_call_result_10630 = invoke(stypy.reporting.localization.Localization(__file__, 227, 8), getitem___10629, int_10615)
    
    # Assigning a type to the variable 'tuple_var_assignment_10038' (line 227)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 227, 8), 'tuple_var_assignment_10038', subscript_call_result_10630)
    
    # Assigning a Subscript to a Name (line 227):
    
    # Obtaining the type of the subscript
    int_10631 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 227, 8), 'int')
    
    # Call to hesv(...): (line 227)
    # Processing the call arguments (line 227)
    # Getting the type of 'a1' (line 227)
    a1_10633 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 33), 'a1', False)
    # Getting the type of 'b1' (line 227)
    b1_10634 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 37), 'b1', False)
    # Processing the call keyword arguments (line 227)
    # Getting the type of 'lwork' (line 227)
    lwork_10635 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 47), 'lwork', False)
    keyword_10636 = lwork_10635
    # Getting the type of 'lower' (line 228)
    lower_10637 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 39), 'lower', False)
    keyword_10638 = lower_10637
    # Getting the type of 'overwrite_a' (line 229)
    overwrite_a_10639 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 229, 45), 'overwrite_a', False)
    keyword_10640 = overwrite_a_10639
    # Getting the type of 'overwrite_b' (line 230)
    overwrite_b_10641 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 45), 'overwrite_b', False)
    keyword_10642 = overwrite_b_10641
    kwargs_10643 = {'lower': keyword_10638, 'overwrite_a': keyword_10640, 'lwork': keyword_10636, 'overwrite_b': keyword_10642}
    # Getting the type of 'hesv' (line 227)
    hesv_10632 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 28), 'hesv', False)
    # Calling hesv(args, kwargs) (line 227)
    hesv_call_result_10644 = invoke(stypy.reporting.localization.Localization(__file__, 227, 28), hesv_10632, *[a1_10633, b1_10634], **kwargs_10643)
    
    # Obtaining the member '__getitem__' of a type (line 227)
    getitem___10645 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 227, 8), hesv_call_result_10644, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 227)
    subscript_call_result_10646 = invoke(stypy.reporting.localization.Localization(__file__, 227, 8), getitem___10645, int_10631)
    
    # Assigning a type to the variable 'tuple_var_assignment_10039' (line 227)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 227, 8), 'tuple_var_assignment_10039', subscript_call_result_10646)
    
    # Assigning a Subscript to a Name (line 227):
    
    # Obtaining the type of the subscript
    int_10647 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 227, 8), 'int')
    
    # Call to hesv(...): (line 227)
    # Processing the call arguments (line 227)
    # Getting the type of 'a1' (line 227)
    a1_10649 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 33), 'a1', False)
    # Getting the type of 'b1' (line 227)
    b1_10650 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 37), 'b1', False)
    # Processing the call keyword arguments (line 227)
    # Getting the type of 'lwork' (line 227)
    lwork_10651 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 47), 'lwork', False)
    keyword_10652 = lwork_10651
    # Getting the type of 'lower' (line 228)
    lower_10653 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 39), 'lower', False)
    keyword_10654 = lower_10653
    # Getting the type of 'overwrite_a' (line 229)
    overwrite_a_10655 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 229, 45), 'overwrite_a', False)
    keyword_10656 = overwrite_a_10655
    # Getting the type of 'overwrite_b' (line 230)
    overwrite_b_10657 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 45), 'overwrite_b', False)
    keyword_10658 = overwrite_b_10657
    kwargs_10659 = {'lower': keyword_10654, 'overwrite_a': keyword_10656, 'lwork': keyword_10652, 'overwrite_b': keyword_10658}
    # Getting the type of 'hesv' (line 227)
    hesv_10648 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 28), 'hesv', False)
    # Calling hesv(args, kwargs) (line 227)
    hesv_call_result_10660 = invoke(stypy.reporting.localization.Localization(__file__, 227, 28), hesv_10648, *[a1_10649, b1_10650], **kwargs_10659)
    
    # Obtaining the member '__getitem__' of a type (line 227)
    getitem___10661 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 227, 8), hesv_call_result_10660, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 227)
    subscript_call_result_10662 = invoke(stypy.reporting.localization.Localization(__file__, 227, 8), getitem___10661, int_10647)
    
    # Assigning a type to the variable 'tuple_var_assignment_10040' (line 227)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 227, 8), 'tuple_var_assignment_10040', subscript_call_result_10662)
    
    # Assigning a Name to a Name (line 227):
    # Getting the type of 'tuple_var_assignment_10037' (line 227)
    tuple_var_assignment_10037_10663 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 8), 'tuple_var_assignment_10037')
    # Assigning a type to the variable 'lu' (line 227)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 227, 8), 'lu', tuple_var_assignment_10037_10663)
    
    # Assigning a Name to a Name (line 227):
    # Getting the type of 'tuple_var_assignment_10038' (line 227)
    tuple_var_assignment_10038_10664 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 8), 'tuple_var_assignment_10038')
    # Assigning a type to the variable 'ipvt' (line 227)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 227, 12), 'ipvt', tuple_var_assignment_10038_10664)
    
    # Assigning a Name to a Name (line 227):
    # Getting the type of 'tuple_var_assignment_10039' (line 227)
    tuple_var_assignment_10039_10665 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 8), 'tuple_var_assignment_10039')
    # Assigning a type to the variable 'x' (line 227)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 227, 18), 'x', tuple_var_assignment_10039_10665)
    
    # Assigning a Name to a Name (line 227):
    # Getting the type of 'tuple_var_assignment_10040' (line 227)
    tuple_var_assignment_10040_10666 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 8), 'tuple_var_assignment_10040')
    # Assigning a type to the variable 'info' (line 227)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 227, 21), 'info', tuple_var_assignment_10040_10666)
    
    # Call to _solve_check(...): (line 231)
    # Processing the call arguments (line 231)
    # Getting the type of 'n' (line 231)
    n_10668 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 231, 21), 'n', False)
    # Getting the type of 'info' (line 231)
    info_10669 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 231, 24), 'info', False)
    # Processing the call keyword arguments (line 231)
    kwargs_10670 = {}
    # Getting the type of '_solve_check' (line 231)
    _solve_check_10667 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 231, 8), '_solve_check', False)
    # Calling _solve_check(args, kwargs) (line 231)
    _solve_check_call_result_10671 = invoke(stypy.reporting.localization.Localization(__file__, 231, 8), _solve_check_10667, *[n_10668, info_10669], **kwargs_10670)
    
    
    # Assigning a Call to a Tuple (line 232):
    
    # Assigning a Subscript to a Name (line 232):
    
    # Obtaining the type of the subscript
    int_10672 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 232, 8), 'int')
    
    # Call to hecon(...): (line 232)
    # Processing the call arguments (line 232)
    # Getting the type of 'lu' (line 232)
    lu_10674 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 28), 'lu', False)
    # Getting the type of 'ipvt' (line 232)
    ipvt_10675 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 32), 'ipvt', False)
    # Getting the type of 'anorm' (line 232)
    anorm_10676 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 38), 'anorm', False)
    # Processing the call keyword arguments (line 232)
    kwargs_10677 = {}
    # Getting the type of 'hecon' (line 232)
    hecon_10673 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 22), 'hecon', False)
    # Calling hecon(args, kwargs) (line 232)
    hecon_call_result_10678 = invoke(stypy.reporting.localization.Localization(__file__, 232, 22), hecon_10673, *[lu_10674, ipvt_10675, anorm_10676], **kwargs_10677)
    
    # Obtaining the member '__getitem__' of a type (line 232)
    getitem___10679 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 232, 8), hecon_call_result_10678, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 232)
    subscript_call_result_10680 = invoke(stypy.reporting.localization.Localization(__file__, 232, 8), getitem___10679, int_10672)
    
    # Assigning a type to the variable 'tuple_var_assignment_10041' (line 232)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 232, 8), 'tuple_var_assignment_10041', subscript_call_result_10680)
    
    # Assigning a Subscript to a Name (line 232):
    
    # Obtaining the type of the subscript
    int_10681 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 232, 8), 'int')
    
    # Call to hecon(...): (line 232)
    # Processing the call arguments (line 232)
    # Getting the type of 'lu' (line 232)
    lu_10683 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 28), 'lu', False)
    # Getting the type of 'ipvt' (line 232)
    ipvt_10684 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 32), 'ipvt', False)
    # Getting the type of 'anorm' (line 232)
    anorm_10685 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 38), 'anorm', False)
    # Processing the call keyword arguments (line 232)
    kwargs_10686 = {}
    # Getting the type of 'hecon' (line 232)
    hecon_10682 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 22), 'hecon', False)
    # Calling hecon(args, kwargs) (line 232)
    hecon_call_result_10687 = invoke(stypy.reporting.localization.Localization(__file__, 232, 22), hecon_10682, *[lu_10683, ipvt_10684, anorm_10685], **kwargs_10686)
    
    # Obtaining the member '__getitem__' of a type (line 232)
    getitem___10688 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 232, 8), hecon_call_result_10687, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 232)
    subscript_call_result_10689 = invoke(stypy.reporting.localization.Localization(__file__, 232, 8), getitem___10688, int_10681)
    
    # Assigning a type to the variable 'tuple_var_assignment_10042' (line 232)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 232, 8), 'tuple_var_assignment_10042', subscript_call_result_10689)
    
    # Assigning a Name to a Name (line 232):
    # Getting the type of 'tuple_var_assignment_10041' (line 232)
    tuple_var_assignment_10041_10690 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 8), 'tuple_var_assignment_10041')
    # Assigning a type to the variable 'rcond' (line 232)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 232, 8), 'rcond', tuple_var_assignment_10041_10690)
    
    # Assigning a Name to a Name (line 232):
    # Getting the type of 'tuple_var_assignment_10042' (line 232)
    tuple_var_assignment_10042_10691 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 8), 'tuple_var_assignment_10042')
    # Assigning a type to the variable 'info' (line 232)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 232, 15), 'info', tuple_var_assignment_10042_10691)
    # SSA branch for the else part of an if statement (line 223)
    module_type_store.open_ssa_branch('else')
    
    
    # Getting the type of 'assume_a' (line 234)
    assume_a_10692 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 234, 9), 'assume_a')
    str_10693 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 234, 21), 'str', 'sym')
    # Applying the binary operator '==' (line 234)
    result_eq_10694 = python_operator(stypy.reporting.localization.Localization(__file__, 234, 9), '==', assume_a_10692, str_10693)
    
    # Testing the type of an if condition (line 234)
    if_condition_10695 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 234, 9), result_eq_10694)
    # Assigning a type to the variable 'if_condition_10695' (line 234)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 234, 9), 'if_condition_10695', if_condition_10695)
    # SSA begins for if statement (line 234)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Tuple (line 235):
    
    # Assigning a Subscript to a Name (line 235):
    
    # Obtaining the type of the subscript
    int_10696 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 235, 8), 'int')
    
    # Call to get_lapack_funcs(...): (line 235)
    # Processing the call arguments (line 235)
    
    # Obtaining an instance of the builtin type 'tuple' (line 235)
    tuple_10698 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 235, 49), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 235)
    # Adding element type (line 235)
    str_10699 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 235, 49), 'str', 'sycon')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 235, 49), tuple_10698, str_10699)
    # Adding element type (line 235)
    str_10700 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 235, 58), 'str', 'sysv')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 235, 49), tuple_10698, str_10700)
    # Adding element type (line 235)
    str_10701 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 235, 66), 'str', 'sysv_lwork')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 235, 49), tuple_10698, str_10701)
    
    
    # Obtaining an instance of the builtin type 'tuple' (line 236)
    tuple_10702 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 236, 49), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 236)
    # Adding element type (line 236)
    # Getting the type of 'a1' (line 236)
    a1_10703 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 236, 49), 'a1', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 236, 49), tuple_10702, a1_10703)
    # Adding element type (line 236)
    # Getting the type of 'b1' (line 236)
    b1_10704 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 236, 53), 'b1', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 236, 49), tuple_10702, b1_10704)
    
    # Processing the call keyword arguments (line 235)
    kwargs_10705 = {}
    # Getting the type of 'get_lapack_funcs' (line 235)
    get_lapack_funcs_10697 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 31), 'get_lapack_funcs', False)
    # Calling get_lapack_funcs(args, kwargs) (line 235)
    get_lapack_funcs_call_result_10706 = invoke(stypy.reporting.localization.Localization(__file__, 235, 31), get_lapack_funcs_10697, *[tuple_10698, tuple_10702], **kwargs_10705)
    
    # Obtaining the member '__getitem__' of a type (line 235)
    getitem___10707 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 235, 8), get_lapack_funcs_call_result_10706, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 235)
    subscript_call_result_10708 = invoke(stypy.reporting.localization.Localization(__file__, 235, 8), getitem___10707, int_10696)
    
    # Assigning a type to the variable 'tuple_var_assignment_10043' (line 235)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 235, 8), 'tuple_var_assignment_10043', subscript_call_result_10708)
    
    # Assigning a Subscript to a Name (line 235):
    
    # Obtaining the type of the subscript
    int_10709 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 235, 8), 'int')
    
    # Call to get_lapack_funcs(...): (line 235)
    # Processing the call arguments (line 235)
    
    # Obtaining an instance of the builtin type 'tuple' (line 235)
    tuple_10711 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 235, 49), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 235)
    # Adding element type (line 235)
    str_10712 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 235, 49), 'str', 'sycon')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 235, 49), tuple_10711, str_10712)
    # Adding element type (line 235)
    str_10713 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 235, 58), 'str', 'sysv')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 235, 49), tuple_10711, str_10713)
    # Adding element type (line 235)
    str_10714 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 235, 66), 'str', 'sysv_lwork')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 235, 49), tuple_10711, str_10714)
    
    
    # Obtaining an instance of the builtin type 'tuple' (line 236)
    tuple_10715 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 236, 49), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 236)
    # Adding element type (line 236)
    # Getting the type of 'a1' (line 236)
    a1_10716 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 236, 49), 'a1', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 236, 49), tuple_10715, a1_10716)
    # Adding element type (line 236)
    # Getting the type of 'b1' (line 236)
    b1_10717 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 236, 53), 'b1', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 236, 49), tuple_10715, b1_10717)
    
    # Processing the call keyword arguments (line 235)
    kwargs_10718 = {}
    # Getting the type of 'get_lapack_funcs' (line 235)
    get_lapack_funcs_10710 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 31), 'get_lapack_funcs', False)
    # Calling get_lapack_funcs(args, kwargs) (line 235)
    get_lapack_funcs_call_result_10719 = invoke(stypy.reporting.localization.Localization(__file__, 235, 31), get_lapack_funcs_10710, *[tuple_10711, tuple_10715], **kwargs_10718)
    
    # Obtaining the member '__getitem__' of a type (line 235)
    getitem___10720 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 235, 8), get_lapack_funcs_call_result_10719, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 235)
    subscript_call_result_10721 = invoke(stypy.reporting.localization.Localization(__file__, 235, 8), getitem___10720, int_10709)
    
    # Assigning a type to the variable 'tuple_var_assignment_10044' (line 235)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 235, 8), 'tuple_var_assignment_10044', subscript_call_result_10721)
    
    # Assigning a Subscript to a Name (line 235):
    
    # Obtaining the type of the subscript
    int_10722 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 235, 8), 'int')
    
    # Call to get_lapack_funcs(...): (line 235)
    # Processing the call arguments (line 235)
    
    # Obtaining an instance of the builtin type 'tuple' (line 235)
    tuple_10724 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 235, 49), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 235)
    # Adding element type (line 235)
    str_10725 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 235, 49), 'str', 'sycon')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 235, 49), tuple_10724, str_10725)
    # Adding element type (line 235)
    str_10726 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 235, 58), 'str', 'sysv')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 235, 49), tuple_10724, str_10726)
    # Adding element type (line 235)
    str_10727 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 235, 66), 'str', 'sysv_lwork')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 235, 49), tuple_10724, str_10727)
    
    
    # Obtaining an instance of the builtin type 'tuple' (line 236)
    tuple_10728 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 236, 49), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 236)
    # Adding element type (line 236)
    # Getting the type of 'a1' (line 236)
    a1_10729 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 236, 49), 'a1', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 236, 49), tuple_10728, a1_10729)
    # Adding element type (line 236)
    # Getting the type of 'b1' (line 236)
    b1_10730 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 236, 53), 'b1', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 236, 49), tuple_10728, b1_10730)
    
    # Processing the call keyword arguments (line 235)
    kwargs_10731 = {}
    # Getting the type of 'get_lapack_funcs' (line 235)
    get_lapack_funcs_10723 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 31), 'get_lapack_funcs', False)
    # Calling get_lapack_funcs(args, kwargs) (line 235)
    get_lapack_funcs_call_result_10732 = invoke(stypy.reporting.localization.Localization(__file__, 235, 31), get_lapack_funcs_10723, *[tuple_10724, tuple_10728], **kwargs_10731)
    
    # Obtaining the member '__getitem__' of a type (line 235)
    getitem___10733 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 235, 8), get_lapack_funcs_call_result_10732, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 235)
    subscript_call_result_10734 = invoke(stypy.reporting.localization.Localization(__file__, 235, 8), getitem___10733, int_10722)
    
    # Assigning a type to the variable 'tuple_var_assignment_10045' (line 235)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 235, 8), 'tuple_var_assignment_10045', subscript_call_result_10734)
    
    # Assigning a Name to a Name (line 235):
    # Getting the type of 'tuple_var_assignment_10043' (line 235)
    tuple_var_assignment_10043_10735 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 8), 'tuple_var_assignment_10043')
    # Assigning a type to the variable 'sycon' (line 235)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 235, 8), 'sycon', tuple_var_assignment_10043_10735)
    
    # Assigning a Name to a Name (line 235):
    # Getting the type of 'tuple_var_assignment_10044' (line 235)
    tuple_var_assignment_10044_10736 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 8), 'tuple_var_assignment_10044')
    # Assigning a type to the variable 'sysv' (line 235)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 235, 15), 'sysv', tuple_var_assignment_10044_10736)
    
    # Assigning a Name to a Name (line 235):
    # Getting the type of 'tuple_var_assignment_10045' (line 235)
    tuple_var_assignment_10045_10737 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 8), 'tuple_var_assignment_10045')
    # Assigning a type to the variable 'sysv_lw' (line 235)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 235, 21), 'sysv_lw', tuple_var_assignment_10045_10737)
    
    # Assigning a Call to a Name (line 237):
    
    # Assigning a Call to a Name (line 237):
    
    # Call to _compute_lwork(...): (line 237)
    # Processing the call arguments (line 237)
    # Getting the type of 'sysv_lw' (line 237)
    sysv_lw_10739 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 237, 31), 'sysv_lw', False)
    # Getting the type of 'n' (line 237)
    n_10740 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 237, 40), 'n', False)
    # Getting the type of 'lower' (line 237)
    lower_10741 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 237, 43), 'lower', False)
    # Processing the call keyword arguments (line 237)
    kwargs_10742 = {}
    # Getting the type of '_compute_lwork' (line 237)
    _compute_lwork_10738 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 237, 16), '_compute_lwork', False)
    # Calling _compute_lwork(args, kwargs) (line 237)
    _compute_lwork_call_result_10743 = invoke(stypy.reporting.localization.Localization(__file__, 237, 16), _compute_lwork_10738, *[sysv_lw_10739, n_10740, lower_10741], **kwargs_10742)
    
    # Assigning a type to the variable 'lwork' (line 237)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 237, 8), 'lwork', _compute_lwork_call_result_10743)
    
    # Assigning a Call to a Tuple (line 238):
    
    # Assigning a Subscript to a Name (line 238):
    
    # Obtaining the type of the subscript
    int_10744 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 238, 8), 'int')
    
    # Call to sysv(...): (line 238)
    # Processing the call arguments (line 238)
    # Getting the type of 'a1' (line 238)
    a1_10746 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 238, 33), 'a1', False)
    # Getting the type of 'b1' (line 238)
    b1_10747 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 238, 37), 'b1', False)
    # Processing the call keyword arguments (line 238)
    # Getting the type of 'lwork' (line 238)
    lwork_10748 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 238, 47), 'lwork', False)
    keyword_10749 = lwork_10748
    # Getting the type of 'lower' (line 239)
    lower_10750 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 239, 39), 'lower', False)
    keyword_10751 = lower_10750
    # Getting the type of 'overwrite_a' (line 240)
    overwrite_a_10752 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 45), 'overwrite_a', False)
    keyword_10753 = overwrite_a_10752
    # Getting the type of 'overwrite_b' (line 241)
    overwrite_b_10754 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 241, 45), 'overwrite_b', False)
    keyword_10755 = overwrite_b_10754
    kwargs_10756 = {'lower': keyword_10751, 'overwrite_a': keyword_10753, 'lwork': keyword_10749, 'overwrite_b': keyword_10755}
    # Getting the type of 'sysv' (line 238)
    sysv_10745 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 238, 28), 'sysv', False)
    # Calling sysv(args, kwargs) (line 238)
    sysv_call_result_10757 = invoke(stypy.reporting.localization.Localization(__file__, 238, 28), sysv_10745, *[a1_10746, b1_10747], **kwargs_10756)
    
    # Obtaining the member '__getitem__' of a type (line 238)
    getitem___10758 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 238, 8), sysv_call_result_10757, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 238)
    subscript_call_result_10759 = invoke(stypy.reporting.localization.Localization(__file__, 238, 8), getitem___10758, int_10744)
    
    # Assigning a type to the variable 'tuple_var_assignment_10046' (line 238)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 238, 8), 'tuple_var_assignment_10046', subscript_call_result_10759)
    
    # Assigning a Subscript to a Name (line 238):
    
    # Obtaining the type of the subscript
    int_10760 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 238, 8), 'int')
    
    # Call to sysv(...): (line 238)
    # Processing the call arguments (line 238)
    # Getting the type of 'a1' (line 238)
    a1_10762 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 238, 33), 'a1', False)
    # Getting the type of 'b1' (line 238)
    b1_10763 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 238, 37), 'b1', False)
    # Processing the call keyword arguments (line 238)
    # Getting the type of 'lwork' (line 238)
    lwork_10764 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 238, 47), 'lwork', False)
    keyword_10765 = lwork_10764
    # Getting the type of 'lower' (line 239)
    lower_10766 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 239, 39), 'lower', False)
    keyword_10767 = lower_10766
    # Getting the type of 'overwrite_a' (line 240)
    overwrite_a_10768 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 45), 'overwrite_a', False)
    keyword_10769 = overwrite_a_10768
    # Getting the type of 'overwrite_b' (line 241)
    overwrite_b_10770 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 241, 45), 'overwrite_b', False)
    keyword_10771 = overwrite_b_10770
    kwargs_10772 = {'lower': keyword_10767, 'overwrite_a': keyword_10769, 'lwork': keyword_10765, 'overwrite_b': keyword_10771}
    # Getting the type of 'sysv' (line 238)
    sysv_10761 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 238, 28), 'sysv', False)
    # Calling sysv(args, kwargs) (line 238)
    sysv_call_result_10773 = invoke(stypy.reporting.localization.Localization(__file__, 238, 28), sysv_10761, *[a1_10762, b1_10763], **kwargs_10772)
    
    # Obtaining the member '__getitem__' of a type (line 238)
    getitem___10774 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 238, 8), sysv_call_result_10773, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 238)
    subscript_call_result_10775 = invoke(stypy.reporting.localization.Localization(__file__, 238, 8), getitem___10774, int_10760)
    
    # Assigning a type to the variable 'tuple_var_assignment_10047' (line 238)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 238, 8), 'tuple_var_assignment_10047', subscript_call_result_10775)
    
    # Assigning a Subscript to a Name (line 238):
    
    # Obtaining the type of the subscript
    int_10776 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 238, 8), 'int')
    
    # Call to sysv(...): (line 238)
    # Processing the call arguments (line 238)
    # Getting the type of 'a1' (line 238)
    a1_10778 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 238, 33), 'a1', False)
    # Getting the type of 'b1' (line 238)
    b1_10779 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 238, 37), 'b1', False)
    # Processing the call keyword arguments (line 238)
    # Getting the type of 'lwork' (line 238)
    lwork_10780 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 238, 47), 'lwork', False)
    keyword_10781 = lwork_10780
    # Getting the type of 'lower' (line 239)
    lower_10782 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 239, 39), 'lower', False)
    keyword_10783 = lower_10782
    # Getting the type of 'overwrite_a' (line 240)
    overwrite_a_10784 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 45), 'overwrite_a', False)
    keyword_10785 = overwrite_a_10784
    # Getting the type of 'overwrite_b' (line 241)
    overwrite_b_10786 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 241, 45), 'overwrite_b', False)
    keyword_10787 = overwrite_b_10786
    kwargs_10788 = {'lower': keyword_10783, 'overwrite_a': keyword_10785, 'lwork': keyword_10781, 'overwrite_b': keyword_10787}
    # Getting the type of 'sysv' (line 238)
    sysv_10777 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 238, 28), 'sysv', False)
    # Calling sysv(args, kwargs) (line 238)
    sysv_call_result_10789 = invoke(stypy.reporting.localization.Localization(__file__, 238, 28), sysv_10777, *[a1_10778, b1_10779], **kwargs_10788)
    
    # Obtaining the member '__getitem__' of a type (line 238)
    getitem___10790 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 238, 8), sysv_call_result_10789, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 238)
    subscript_call_result_10791 = invoke(stypy.reporting.localization.Localization(__file__, 238, 8), getitem___10790, int_10776)
    
    # Assigning a type to the variable 'tuple_var_assignment_10048' (line 238)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 238, 8), 'tuple_var_assignment_10048', subscript_call_result_10791)
    
    # Assigning a Subscript to a Name (line 238):
    
    # Obtaining the type of the subscript
    int_10792 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 238, 8), 'int')
    
    # Call to sysv(...): (line 238)
    # Processing the call arguments (line 238)
    # Getting the type of 'a1' (line 238)
    a1_10794 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 238, 33), 'a1', False)
    # Getting the type of 'b1' (line 238)
    b1_10795 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 238, 37), 'b1', False)
    # Processing the call keyword arguments (line 238)
    # Getting the type of 'lwork' (line 238)
    lwork_10796 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 238, 47), 'lwork', False)
    keyword_10797 = lwork_10796
    # Getting the type of 'lower' (line 239)
    lower_10798 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 239, 39), 'lower', False)
    keyword_10799 = lower_10798
    # Getting the type of 'overwrite_a' (line 240)
    overwrite_a_10800 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 45), 'overwrite_a', False)
    keyword_10801 = overwrite_a_10800
    # Getting the type of 'overwrite_b' (line 241)
    overwrite_b_10802 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 241, 45), 'overwrite_b', False)
    keyword_10803 = overwrite_b_10802
    kwargs_10804 = {'lower': keyword_10799, 'overwrite_a': keyword_10801, 'lwork': keyword_10797, 'overwrite_b': keyword_10803}
    # Getting the type of 'sysv' (line 238)
    sysv_10793 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 238, 28), 'sysv', False)
    # Calling sysv(args, kwargs) (line 238)
    sysv_call_result_10805 = invoke(stypy.reporting.localization.Localization(__file__, 238, 28), sysv_10793, *[a1_10794, b1_10795], **kwargs_10804)
    
    # Obtaining the member '__getitem__' of a type (line 238)
    getitem___10806 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 238, 8), sysv_call_result_10805, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 238)
    subscript_call_result_10807 = invoke(stypy.reporting.localization.Localization(__file__, 238, 8), getitem___10806, int_10792)
    
    # Assigning a type to the variable 'tuple_var_assignment_10049' (line 238)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 238, 8), 'tuple_var_assignment_10049', subscript_call_result_10807)
    
    # Assigning a Name to a Name (line 238):
    # Getting the type of 'tuple_var_assignment_10046' (line 238)
    tuple_var_assignment_10046_10808 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 238, 8), 'tuple_var_assignment_10046')
    # Assigning a type to the variable 'lu' (line 238)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 238, 8), 'lu', tuple_var_assignment_10046_10808)
    
    # Assigning a Name to a Name (line 238):
    # Getting the type of 'tuple_var_assignment_10047' (line 238)
    tuple_var_assignment_10047_10809 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 238, 8), 'tuple_var_assignment_10047')
    # Assigning a type to the variable 'ipvt' (line 238)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 238, 12), 'ipvt', tuple_var_assignment_10047_10809)
    
    # Assigning a Name to a Name (line 238):
    # Getting the type of 'tuple_var_assignment_10048' (line 238)
    tuple_var_assignment_10048_10810 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 238, 8), 'tuple_var_assignment_10048')
    # Assigning a type to the variable 'x' (line 238)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 238, 18), 'x', tuple_var_assignment_10048_10810)
    
    # Assigning a Name to a Name (line 238):
    # Getting the type of 'tuple_var_assignment_10049' (line 238)
    tuple_var_assignment_10049_10811 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 238, 8), 'tuple_var_assignment_10049')
    # Assigning a type to the variable 'info' (line 238)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 238, 21), 'info', tuple_var_assignment_10049_10811)
    
    # Call to _solve_check(...): (line 242)
    # Processing the call arguments (line 242)
    # Getting the type of 'n' (line 242)
    n_10813 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 242, 21), 'n', False)
    # Getting the type of 'info' (line 242)
    info_10814 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 242, 24), 'info', False)
    # Processing the call keyword arguments (line 242)
    kwargs_10815 = {}
    # Getting the type of '_solve_check' (line 242)
    _solve_check_10812 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 242, 8), '_solve_check', False)
    # Calling _solve_check(args, kwargs) (line 242)
    _solve_check_call_result_10816 = invoke(stypy.reporting.localization.Localization(__file__, 242, 8), _solve_check_10812, *[n_10813, info_10814], **kwargs_10815)
    
    
    # Assigning a Call to a Tuple (line 243):
    
    # Assigning a Subscript to a Name (line 243):
    
    # Obtaining the type of the subscript
    int_10817 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 243, 8), 'int')
    
    # Call to sycon(...): (line 243)
    # Processing the call arguments (line 243)
    # Getting the type of 'lu' (line 243)
    lu_10819 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 243, 28), 'lu', False)
    # Getting the type of 'ipvt' (line 243)
    ipvt_10820 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 243, 32), 'ipvt', False)
    # Getting the type of 'anorm' (line 243)
    anorm_10821 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 243, 38), 'anorm', False)
    # Processing the call keyword arguments (line 243)
    kwargs_10822 = {}
    # Getting the type of 'sycon' (line 243)
    sycon_10818 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 243, 22), 'sycon', False)
    # Calling sycon(args, kwargs) (line 243)
    sycon_call_result_10823 = invoke(stypy.reporting.localization.Localization(__file__, 243, 22), sycon_10818, *[lu_10819, ipvt_10820, anorm_10821], **kwargs_10822)
    
    # Obtaining the member '__getitem__' of a type (line 243)
    getitem___10824 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 243, 8), sycon_call_result_10823, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 243)
    subscript_call_result_10825 = invoke(stypy.reporting.localization.Localization(__file__, 243, 8), getitem___10824, int_10817)
    
    # Assigning a type to the variable 'tuple_var_assignment_10050' (line 243)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 243, 8), 'tuple_var_assignment_10050', subscript_call_result_10825)
    
    # Assigning a Subscript to a Name (line 243):
    
    # Obtaining the type of the subscript
    int_10826 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 243, 8), 'int')
    
    # Call to sycon(...): (line 243)
    # Processing the call arguments (line 243)
    # Getting the type of 'lu' (line 243)
    lu_10828 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 243, 28), 'lu', False)
    # Getting the type of 'ipvt' (line 243)
    ipvt_10829 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 243, 32), 'ipvt', False)
    # Getting the type of 'anorm' (line 243)
    anorm_10830 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 243, 38), 'anorm', False)
    # Processing the call keyword arguments (line 243)
    kwargs_10831 = {}
    # Getting the type of 'sycon' (line 243)
    sycon_10827 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 243, 22), 'sycon', False)
    # Calling sycon(args, kwargs) (line 243)
    sycon_call_result_10832 = invoke(stypy.reporting.localization.Localization(__file__, 243, 22), sycon_10827, *[lu_10828, ipvt_10829, anorm_10830], **kwargs_10831)
    
    # Obtaining the member '__getitem__' of a type (line 243)
    getitem___10833 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 243, 8), sycon_call_result_10832, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 243)
    subscript_call_result_10834 = invoke(stypy.reporting.localization.Localization(__file__, 243, 8), getitem___10833, int_10826)
    
    # Assigning a type to the variable 'tuple_var_assignment_10051' (line 243)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 243, 8), 'tuple_var_assignment_10051', subscript_call_result_10834)
    
    # Assigning a Name to a Name (line 243):
    # Getting the type of 'tuple_var_assignment_10050' (line 243)
    tuple_var_assignment_10050_10835 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 243, 8), 'tuple_var_assignment_10050')
    # Assigning a type to the variable 'rcond' (line 243)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 243, 8), 'rcond', tuple_var_assignment_10050_10835)
    
    # Assigning a Name to a Name (line 243):
    # Getting the type of 'tuple_var_assignment_10051' (line 243)
    tuple_var_assignment_10051_10836 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 243, 8), 'tuple_var_assignment_10051')
    # Assigning a type to the variable 'info' (line 243)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 243, 15), 'info', tuple_var_assignment_10051_10836)
    # SSA branch for the else part of an if statement (line 234)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Call to a Tuple (line 246):
    
    # Assigning a Subscript to a Name (line 246):
    
    # Obtaining the type of the subscript
    int_10837 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 246, 8), 'int')
    
    # Call to get_lapack_funcs(...): (line 246)
    # Processing the call arguments (line 246)
    
    # Obtaining an instance of the builtin type 'tuple' (line 246)
    tuple_10839 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 246, 40), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 246)
    # Adding element type (line 246)
    str_10840 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 246, 40), 'str', 'pocon')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 246, 40), tuple_10839, str_10840)
    # Adding element type (line 246)
    str_10841 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 246, 49), 'str', 'posv')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 246, 40), tuple_10839, str_10841)
    
    
    # Obtaining an instance of the builtin type 'tuple' (line 247)
    tuple_10842 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 247, 40), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 247)
    # Adding element type (line 247)
    # Getting the type of 'a1' (line 247)
    a1_10843 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 247, 40), 'a1', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 247, 40), tuple_10842, a1_10843)
    # Adding element type (line 247)
    # Getting the type of 'b1' (line 247)
    b1_10844 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 247, 44), 'b1', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 247, 40), tuple_10842, b1_10844)
    
    # Processing the call keyword arguments (line 246)
    kwargs_10845 = {}
    # Getting the type of 'get_lapack_funcs' (line 246)
    get_lapack_funcs_10838 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 22), 'get_lapack_funcs', False)
    # Calling get_lapack_funcs(args, kwargs) (line 246)
    get_lapack_funcs_call_result_10846 = invoke(stypy.reporting.localization.Localization(__file__, 246, 22), get_lapack_funcs_10838, *[tuple_10839, tuple_10842], **kwargs_10845)
    
    # Obtaining the member '__getitem__' of a type (line 246)
    getitem___10847 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 246, 8), get_lapack_funcs_call_result_10846, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 246)
    subscript_call_result_10848 = invoke(stypy.reporting.localization.Localization(__file__, 246, 8), getitem___10847, int_10837)
    
    # Assigning a type to the variable 'tuple_var_assignment_10052' (line 246)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 246, 8), 'tuple_var_assignment_10052', subscript_call_result_10848)
    
    # Assigning a Subscript to a Name (line 246):
    
    # Obtaining the type of the subscript
    int_10849 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 246, 8), 'int')
    
    # Call to get_lapack_funcs(...): (line 246)
    # Processing the call arguments (line 246)
    
    # Obtaining an instance of the builtin type 'tuple' (line 246)
    tuple_10851 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 246, 40), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 246)
    # Adding element type (line 246)
    str_10852 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 246, 40), 'str', 'pocon')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 246, 40), tuple_10851, str_10852)
    # Adding element type (line 246)
    str_10853 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 246, 49), 'str', 'posv')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 246, 40), tuple_10851, str_10853)
    
    
    # Obtaining an instance of the builtin type 'tuple' (line 247)
    tuple_10854 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 247, 40), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 247)
    # Adding element type (line 247)
    # Getting the type of 'a1' (line 247)
    a1_10855 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 247, 40), 'a1', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 247, 40), tuple_10854, a1_10855)
    # Adding element type (line 247)
    # Getting the type of 'b1' (line 247)
    b1_10856 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 247, 44), 'b1', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 247, 40), tuple_10854, b1_10856)
    
    # Processing the call keyword arguments (line 246)
    kwargs_10857 = {}
    # Getting the type of 'get_lapack_funcs' (line 246)
    get_lapack_funcs_10850 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 22), 'get_lapack_funcs', False)
    # Calling get_lapack_funcs(args, kwargs) (line 246)
    get_lapack_funcs_call_result_10858 = invoke(stypy.reporting.localization.Localization(__file__, 246, 22), get_lapack_funcs_10850, *[tuple_10851, tuple_10854], **kwargs_10857)
    
    # Obtaining the member '__getitem__' of a type (line 246)
    getitem___10859 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 246, 8), get_lapack_funcs_call_result_10858, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 246)
    subscript_call_result_10860 = invoke(stypy.reporting.localization.Localization(__file__, 246, 8), getitem___10859, int_10849)
    
    # Assigning a type to the variable 'tuple_var_assignment_10053' (line 246)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 246, 8), 'tuple_var_assignment_10053', subscript_call_result_10860)
    
    # Assigning a Name to a Name (line 246):
    # Getting the type of 'tuple_var_assignment_10052' (line 246)
    tuple_var_assignment_10052_10861 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 8), 'tuple_var_assignment_10052')
    # Assigning a type to the variable 'pocon' (line 246)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 246, 8), 'pocon', tuple_var_assignment_10052_10861)
    
    # Assigning a Name to a Name (line 246):
    # Getting the type of 'tuple_var_assignment_10053' (line 246)
    tuple_var_assignment_10053_10862 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 8), 'tuple_var_assignment_10053')
    # Assigning a type to the variable 'posv' (line 246)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 246, 15), 'posv', tuple_var_assignment_10053_10862)
    
    # Assigning a Call to a Tuple (line 248):
    
    # Assigning a Subscript to a Name (line 248):
    
    # Obtaining the type of the subscript
    int_10863 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 248, 8), 'int')
    
    # Call to posv(...): (line 248)
    # Processing the call arguments (line 248)
    # Getting the type of 'a1' (line 248)
    a1_10865 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 248, 27), 'a1', False)
    # Getting the type of 'b1' (line 248)
    b1_10866 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 248, 31), 'b1', False)
    # Processing the call keyword arguments (line 248)
    # Getting the type of 'lower' (line 248)
    lower_10867 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 248, 41), 'lower', False)
    keyword_10868 = lower_10867
    # Getting the type of 'overwrite_a' (line 249)
    overwrite_a_10869 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 249, 39), 'overwrite_a', False)
    keyword_10870 = overwrite_a_10869
    # Getting the type of 'overwrite_b' (line 250)
    overwrite_b_10871 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 250, 39), 'overwrite_b', False)
    keyword_10872 = overwrite_b_10871
    kwargs_10873 = {'lower': keyword_10868, 'overwrite_a': keyword_10870, 'overwrite_b': keyword_10872}
    # Getting the type of 'posv' (line 248)
    posv_10864 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 248, 22), 'posv', False)
    # Calling posv(args, kwargs) (line 248)
    posv_call_result_10874 = invoke(stypy.reporting.localization.Localization(__file__, 248, 22), posv_10864, *[a1_10865, b1_10866], **kwargs_10873)
    
    # Obtaining the member '__getitem__' of a type (line 248)
    getitem___10875 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 248, 8), posv_call_result_10874, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 248)
    subscript_call_result_10876 = invoke(stypy.reporting.localization.Localization(__file__, 248, 8), getitem___10875, int_10863)
    
    # Assigning a type to the variable 'tuple_var_assignment_10054' (line 248)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 248, 8), 'tuple_var_assignment_10054', subscript_call_result_10876)
    
    # Assigning a Subscript to a Name (line 248):
    
    # Obtaining the type of the subscript
    int_10877 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 248, 8), 'int')
    
    # Call to posv(...): (line 248)
    # Processing the call arguments (line 248)
    # Getting the type of 'a1' (line 248)
    a1_10879 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 248, 27), 'a1', False)
    # Getting the type of 'b1' (line 248)
    b1_10880 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 248, 31), 'b1', False)
    # Processing the call keyword arguments (line 248)
    # Getting the type of 'lower' (line 248)
    lower_10881 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 248, 41), 'lower', False)
    keyword_10882 = lower_10881
    # Getting the type of 'overwrite_a' (line 249)
    overwrite_a_10883 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 249, 39), 'overwrite_a', False)
    keyword_10884 = overwrite_a_10883
    # Getting the type of 'overwrite_b' (line 250)
    overwrite_b_10885 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 250, 39), 'overwrite_b', False)
    keyword_10886 = overwrite_b_10885
    kwargs_10887 = {'lower': keyword_10882, 'overwrite_a': keyword_10884, 'overwrite_b': keyword_10886}
    # Getting the type of 'posv' (line 248)
    posv_10878 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 248, 22), 'posv', False)
    # Calling posv(args, kwargs) (line 248)
    posv_call_result_10888 = invoke(stypy.reporting.localization.Localization(__file__, 248, 22), posv_10878, *[a1_10879, b1_10880], **kwargs_10887)
    
    # Obtaining the member '__getitem__' of a type (line 248)
    getitem___10889 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 248, 8), posv_call_result_10888, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 248)
    subscript_call_result_10890 = invoke(stypy.reporting.localization.Localization(__file__, 248, 8), getitem___10889, int_10877)
    
    # Assigning a type to the variable 'tuple_var_assignment_10055' (line 248)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 248, 8), 'tuple_var_assignment_10055', subscript_call_result_10890)
    
    # Assigning a Subscript to a Name (line 248):
    
    # Obtaining the type of the subscript
    int_10891 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 248, 8), 'int')
    
    # Call to posv(...): (line 248)
    # Processing the call arguments (line 248)
    # Getting the type of 'a1' (line 248)
    a1_10893 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 248, 27), 'a1', False)
    # Getting the type of 'b1' (line 248)
    b1_10894 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 248, 31), 'b1', False)
    # Processing the call keyword arguments (line 248)
    # Getting the type of 'lower' (line 248)
    lower_10895 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 248, 41), 'lower', False)
    keyword_10896 = lower_10895
    # Getting the type of 'overwrite_a' (line 249)
    overwrite_a_10897 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 249, 39), 'overwrite_a', False)
    keyword_10898 = overwrite_a_10897
    # Getting the type of 'overwrite_b' (line 250)
    overwrite_b_10899 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 250, 39), 'overwrite_b', False)
    keyword_10900 = overwrite_b_10899
    kwargs_10901 = {'lower': keyword_10896, 'overwrite_a': keyword_10898, 'overwrite_b': keyword_10900}
    # Getting the type of 'posv' (line 248)
    posv_10892 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 248, 22), 'posv', False)
    # Calling posv(args, kwargs) (line 248)
    posv_call_result_10902 = invoke(stypy.reporting.localization.Localization(__file__, 248, 22), posv_10892, *[a1_10893, b1_10894], **kwargs_10901)
    
    # Obtaining the member '__getitem__' of a type (line 248)
    getitem___10903 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 248, 8), posv_call_result_10902, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 248)
    subscript_call_result_10904 = invoke(stypy.reporting.localization.Localization(__file__, 248, 8), getitem___10903, int_10891)
    
    # Assigning a type to the variable 'tuple_var_assignment_10056' (line 248)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 248, 8), 'tuple_var_assignment_10056', subscript_call_result_10904)
    
    # Assigning a Name to a Name (line 248):
    # Getting the type of 'tuple_var_assignment_10054' (line 248)
    tuple_var_assignment_10054_10905 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 248, 8), 'tuple_var_assignment_10054')
    # Assigning a type to the variable 'lu' (line 248)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 248, 8), 'lu', tuple_var_assignment_10054_10905)
    
    # Assigning a Name to a Name (line 248):
    # Getting the type of 'tuple_var_assignment_10055' (line 248)
    tuple_var_assignment_10055_10906 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 248, 8), 'tuple_var_assignment_10055')
    # Assigning a type to the variable 'x' (line 248)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 248, 12), 'x', tuple_var_assignment_10055_10906)
    
    # Assigning a Name to a Name (line 248):
    # Getting the type of 'tuple_var_assignment_10056' (line 248)
    tuple_var_assignment_10056_10907 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 248, 8), 'tuple_var_assignment_10056')
    # Assigning a type to the variable 'info' (line 248)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 248, 15), 'info', tuple_var_assignment_10056_10907)
    
    # Call to _solve_check(...): (line 251)
    # Processing the call arguments (line 251)
    # Getting the type of 'n' (line 251)
    n_10909 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 251, 21), 'n', False)
    # Getting the type of 'info' (line 251)
    info_10910 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 251, 24), 'info', False)
    # Processing the call keyword arguments (line 251)
    kwargs_10911 = {}
    # Getting the type of '_solve_check' (line 251)
    _solve_check_10908 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 251, 8), '_solve_check', False)
    # Calling _solve_check(args, kwargs) (line 251)
    _solve_check_call_result_10912 = invoke(stypy.reporting.localization.Localization(__file__, 251, 8), _solve_check_10908, *[n_10909, info_10910], **kwargs_10911)
    
    
    # Assigning a Call to a Tuple (line 252):
    
    # Assigning a Subscript to a Name (line 252):
    
    # Obtaining the type of the subscript
    int_10913 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 252, 8), 'int')
    
    # Call to pocon(...): (line 252)
    # Processing the call arguments (line 252)
    # Getting the type of 'lu' (line 252)
    lu_10915 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 252, 28), 'lu', False)
    # Getting the type of 'anorm' (line 252)
    anorm_10916 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 252, 32), 'anorm', False)
    # Processing the call keyword arguments (line 252)
    kwargs_10917 = {}
    # Getting the type of 'pocon' (line 252)
    pocon_10914 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 252, 22), 'pocon', False)
    # Calling pocon(args, kwargs) (line 252)
    pocon_call_result_10918 = invoke(stypy.reporting.localization.Localization(__file__, 252, 22), pocon_10914, *[lu_10915, anorm_10916], **kwargs_10917)
    
    # Obtaining the member '__getitem__' of a type (line 252)
    getitem___10919 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 252, 8), pocon_call_result_10918, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 252)
    subscript_call_result_10920 = invoke(stypy.reporting.localization.Localization(__file__, 252, 8), getitem___10919, int_10913)
    
    # Assigning a type to the variable 'tuple_var_assignment_10057' (line 252)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 252, 8), 'tuple_var_assignment_10057', subscript_call_result_10920)
    
    # Assigning a Subscript to a Name (line 252):
    
    # Obtaining the type of the subscript
    int_10921 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 252, 8), 'int')
    
    # Call to pocon(...): (line 252)
    # Processing the call arguments (line 252)
    # Getting the type of 'lu' (line 252)
    lu_10923 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 252, 28), 'lu', False)
    # Getting the type of 'anorm' (line 252)
    anorm_10924 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 252, 32), 'anorm', False)
    # Processing the call keyword arguments (line 252)
    kwargs_10925 = {}
    # Getting the type of 'pocon' (line 252)
    pocon_10922 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 252, 22), 'pocon', False)
    # Calling pocon(args, kwargs) (line 252)
    pocon_call_result_10926 = invoke(stypy.reporting.localization.Localization(__file__, 252, 22), pocon_10922, *[lu_10923, anorm_10924], **kwargs_10925)
    
    # Obtaining the member '__getitem__' of a type (line 252)
    getitem___10927 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 252, 8), pocon_call_result_10926, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 252)
    subscript_call_result_10928 = invoke(stypy.reporting.localization.Localization(__file__, 252, 8), getitem___10927, int_10921)
    
    # Assigning a type to the variable 'tuple_var_assignment_10058' (line 252)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 252, 8), 'tuple_var_assignment_10058', subscript_call_result_10928)
    
    # Assigning a Name to a Name (line 252):
    # Getting the type of 'tuple_var_assignment_10057' (line 252)
    tuple_var_assignment_10057_10929 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 252, 8), 'tuple_var_assignment_10057')
    # Assigning a type to the variable 'rcond' (line 252)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 252, 8), 'rcond', tuple_var_assignment_10057_10929)
    
    # Assigning a Name to a Name (line 252):
    # Getting the type of 'tuple_var_assignment_10058' (line 252)
    tuple_var_assignment_10058_10930 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 252, 8), 'tuple_var_assignment_10058')
    # Assigning a type to the variable 'info' (line 252)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 252, 15), 'info', tuple_var_assignment_10058_10930)
    # SSA join for if statement (line 234)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 223)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 213)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to _solve_check(...): (line 254)
    # Processing the call arguments (line 254)
    # Getting the type of 'n' (line 254)
    n_10932 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 254, 17), 'n', False)
    # Getting the type of 'info' (line 254)
    info_10933 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 254, 20), 'info', False)
    # Getting the type of 'lamch' (line 254)
    lamch_10934 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 254, 26), 'lamch', False)
    # Getting the type of 'rcond' (line 254)
    rcond_10935 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 254, 33), 'rcond', False)
    # Processing the call keyword arguments (line 254)
    kwargs_10936 = {}
    # Getting the type of '_solve_check' (line 254)
    _solve_check_10931 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 254, 4), '_solve_check', False)
    # Calling _solve_check(args, kwargs) (line 254)
    _solve_check_call_result_10937 = invoke(stypy.reporting.localization.Localization(__file__, 254, 4), _solve_check_10931, *[n_10932, info_10933, lamch_10934, rcond_10935], **kwargs_10936)
    
    
    # Getting the type of 'b_is_1D' (line 256)
    b_is_1D_10938 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 256, 7), 'b_is_1D')
    # Testing the type of an if condition (line 256)
    if_condition_10939 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 256, 4), b_is_1D_10938)
    # Assigning a type to the variable 'if_condition_10939' (line 256)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 256, 4), 'if_condition_10939', if_condition_10939)
    # SSA begins for if statement (line 256)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 257):
    
    # Assigning a Call to a Name (line 257):
    
    # Call to ravel(...): (line 257)
    # Processing the call keyword arguments (line 257)
    kwargs_10942 = {}
    # Getting the type of 'x' (line 257)
    x_10940 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 257, 12), 'x', False)
    # Obtaining the member 'ravel' of a type (line 257)
    ravel_10941 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 257, 12), x_10940, 'ravel')
    # Calling ravel(args, kwargs) (line 257)
    ravel_call_result_10943 = invoke(stypy.reporting.localization.Localization(__file__, 257, 12), ravel_10941, *[], **kwargs_10942)
    
    # Assigning a type to the variable 'x' (line 257)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 257, 8), 'x', ravel_call_result_10943)
    # SSA join for if statement (line 256)
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'x' (line 259)
    x_10944 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 259, 11), 'x')
    # Assigning a type to the variable 'stypy_return_type' (line 259)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 259, 4), 'stypy_return_type', x_10944)
    
    # ################# End of 'solve(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'solve' in the type store
    # Getting the type of 'stypy_return_type' (line 43)
    stypy_return_type_10945 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_10945)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'solve'
    return stypy_return_type_10945

# Assigning a type to the variable 'solve' (line 43)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 43, 0), 'solve', solve)

@norecursion
def solve_triangular(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    int_10946 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 262, 33), 'int')
    # Getting the type of 'False' (line 262)
    False_10947 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 262, 42), 'False')
    # Getting the type of 'False' (line 262)
    False_10948 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 262, 63), 'False')
    # Getting the type of 'False' (line 263)
    False_10949 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 263, 33), 'False')
    # Getting the type of 'None' (line 263)
    None_10950 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 263, 46), 'None')
    # Getting the type of 'True' (line 263)
    True_10951 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 263, 65), 'True')
    defaults = [int_10946, False_10947, False_10948, False_10949, None_10950, True_10951]
    # Create a new context for function 'solve_triangular'
    module_type_store = module_type_store.open_function_context('solve_triangular', 262, 0, False)
    
    # Passed parameters checking function
    solve_triangular.stypy_localization = localization
    solve_triangular.stypy_type_of_self = None
    solve_triangular.stypy_type_store = module_type_store
    solve_triangular.stypy_function_name = 'solve_triangular'
    solve_triangular.stypy_param_names_list = ['a', 'b', 'trans', 'lower', 'unit_diagonal', 'overwrite_b', 'debug', 'check_finite']
    solve_triangular.stypy_varargs_param_name = None
    solve_triangular.stypy_kwargs_param_name = None
    solve_triangular.stypy_call_defaults = defaults
    solve_triangular.stypy_call_varargs = varargs
    solve_triangular.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'solve_triangular', ['a', 'b', 'trans', 'lower', 'unit_diagonal', 'overwrite_b', 'debug', 'check_finite'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'solve_triangular', localization, ['a', 'b', 'trans', 'lower', 'unit_diagonal', 'overwrite_b', 'debug', 'check_finite'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'solve_triangular(...)' code ##################

    str_10952 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 328, (-1)), 'str', "\n    Solve the equation `a x = b` for `x`, assuming a is a triangular matrix.\n\n    Parameters\n    ----------\n    a : (M, M) array_like\n        A triangular matrix\n    b : (M,) or (M, N) array_like\n        Right-hand side matrix in `a x = b`\n    lower : bool, optional\n        Use only data contained in the lower triangle of `a`.\n        Default is to use upper triangle.\n    trans : {0, 1, 2, 'N', 'T', 'C'}, optional\n        Type of system to solve:\n\n        ========  =========\n        trans     system\n        ========  =========\n        0 or 'N'  a x  = b\n        1 or 'T'  a^T x = b\n        2 or 'C'  a^H x = b\n        ========  =========\n    unit_diagonal : bool, optional\n        If True, diagonal elements of `a` are assumed to be 1 and\n        will not be referenced.\n    overwrite_b : bool, optional\n        Allow overwriting data in `b` (may enhance performance)\n    check_finite : bool, optional\n        Whether to check that the input matrices contain only finite numbers.\n        Disabling may give a performance gain, but may result in problems\n        (crashes, non-termination) if the inputs do contain infinities or NaNs.\n\n    Returns\n    -------\n    x : (M,) or (M, N) ndarray\n        Solution to the system `a x = b`.  Shape of return matches `b`.\n\n    Raises\n    ------\n    LinAlgError\n        If `a` is singular\n\n    Notes\n    -----\n    .. versionadded:: 0.9.0\n\n    Examples\n    --------\n    Solve the lower triangular system a x = b, where::\n\n             [3  0  0  0]       [4]\n        a =  [2  1  0  0]   b = [2]\n             [1  0  1  0]       [4]\n             [1  1  1  1]       [2]\n\n    >>> from scipy.linalg import solve_triangular\n    >>> a = np.array([[3, 0, 0, 0], [2, 1, 0, 0], [1, 0, 1, 0], [1, 1, 1, 1]])\n    >>> b = np.array([4, 2, 4, 2])\n    >>> x = solve_triangular(a, b, lower=True)\n    >>> x\n    array([ 1.33333333, -0.66666667,  2.66666667, -1.33333333])\n    >>> a.dot(x)  # Check the result\n    array([ 4.,  2.,  4.,  2.])\n\n    ")
    
    # Type idiom detected: calculating its left and rigth part (line 331)
    # Getting the type of 'debug' (line 331)
    debug_10953 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 331, 4), 'debug')
    # Getting the type of 'None' (line 331)
    None_10954 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 331, 20), 'None')
    
    (may_be_10955, more_types_in_union_10956) = may_not_be_none(debug_10953, None_10954)

    if may_be_10955:

        if more_types_in_union_10956:
            # Runtime conditional SSA (line 331)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Call to warn(...): (line 332)
        # Processing the call arguments (line 332)
        str_10959 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 332, 22), 'str', 'Use of the "debug" keyword is deprecated and this keyword will be removed in the future versions of SciPy.')
        # Getting the type of 'DeprecationWarning' (line 334)
        DeprecationWarning_10960 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 334, 44), 'DeprecationWarning', False)
        # Processing the call keyword arguments (line 332)
        kwargs_10961 = {}
        # Getting the type of 'warnings' (line 332)
        warnings_10957 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 332, 8), 'warnings', False)
        # Obtaining the member 'warn' of a type (line 332)
        warn_10958 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 332, 8), warnings_10957, 'warn')
        # Calling warn(args, kwargs) (line 332)
        warn_call_result_10962 = invoke(stypy.reporting.localization.Localization(__file__, 332, 8), warn_10958, *[str_10959, DeprecationWarning_10960], **kwargs_10961)
        

        if more_types_in_union_10956:
            # SSA join for if statement (line 331)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Assigning a Call to a Name (line 336):
    
    # Assigning a Call to a Name (line 336):
    
    # Call to _asarray_validated(...): (line 336)
    # Processing the call arguments (line 336)
    # Getting the type of 'a' (line 336)
    a_10964 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 336, 28), 'a', False)
    # Processing the call keyword arguments (line 336)
    # Getting the type of 'check_finite' (line 336)
    check_finite_10965 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 336, 44), 'check_finite', False)
    keyword_10966 = check_finite_10965
    kwargs_10967 = {'check_finite': keyword_10966}
    # Getting the type of '_asarray_validated' (line 336)
    _asarray_validated_10963 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 336, 9), '_asarray_validated', False)
    # Calling _asarray_validated(args, kwargs) (line 336)
    _asarray_validated_call_result_10968 = invoke(stypy.reporting.localization.Localization(__file__, 336, 9), _asarray_validated_10963, *[a_10964], **kwargs_10967)
    
    # Assigning a type to the variable 'a1' (line 336)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 336, 4), 'a1', _asarray_validated_call_result_10968)
    
    # Assigning a Call to a Name (line 337):
    
    # Assigning a Call to a Name (line 337):
    
    # Call to _asarray_validated(...): (line 337)
    # Processing the call arguments (line 337)
    # Getting the type of 'b' (line 337)
    b_10970 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 337, 28), 'b', False)
    # Processing the call keyword arguments (line 337)
    # Getting the type of 'check_finite' (line 337)
    check_finite_10971 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 337, 44), 'check_finite', False)
    keyword_10972 = check_finite_10971
    kwargs_10973 = {'check_finite': keyword_10972}
    # Getting the type of '_asarray_validated' (line 337)
    _asarray_validated_10969 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 337, 9), '_asarray_validated', False)
    # Calling _asarray_validated(args, kwargs) (line 337)
    _asarray_validated_call_result_10974 = invoke(stypy.reporting.localization.Localization(__file__, 337, 9), _asarray_validated_10969, *[b_10970], **kwargs_10973)
    
    # Assigning a type to the variable 'b1' (line 337)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 337, 4), 'b1', _asarray_validated_call_result_10974)
    
    
    # Evaluating a boolean operation
    
    
    # Call to len(...): (line 338)
    # Processing the call arguments (line 338)
    # Getting the type of 'a1' (line 338)
    a1_10976 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 338, 11), 'a1', False)
    # Obtaining the member 'shape' of a type (line 338)
    shape_10977 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 338, 11), a1_10976, 'shape')
    # Processing the call keyword arguments (line 338)
    kwargs_10978 = {}
    # Getting the type of 'len' (line 338)
    len_10975 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 338, 7), 'len', False)
    # Calling len(args, kwargs) (line 338)
    len_call_result_10979 = invoke(stypy.reporting.localization.Localization(__file__, 338, 7), len_10975, *[shape_10977], **kwargs_10978)
    
    int_10980 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 338, 24), 'int')
    # Applying the binary operator '!=' (line 338)
    result_ne_10981 = python_operator(stypy.reporting.localization.Localization(__file__, 338, 7), '!=', len_call_result_10979, int_10980)
    
    
    
    # Obtaining the type of the subscript
    int_10982 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 338, 38), 'int')
    # Getting the type of 'a1' (line 338)
    a1_10983 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 338, 29), 'a1')
    # Obtaining the member 'shape' of a type (line 338)
    shape_10984 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 338, 29), a1_10983, 'shape')
    # Obtaining the member '__getitem__' of a type (line 338)
    getitem___10985 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 338, 29), shape_10984, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 338)
    subscript_call_result_10986 = invoke(stypy.reporting.localization.Localization(__file__, 338, 29), getitem___10985, int_10982)
    
    
    # Obtaining the type of the subscript
    int_10987 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 338, 53), 'int')
    # Getting the type of 'a1' (line 338)
    a1_10988 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 338, 44), 'a1')
    # Obtaining the member 'shape' of a type (line 338)
    shape_10989 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 338, 44), a1_10988, 'shape')
    # Obtaining the member '__getitem__' of a type (line 338)
    getitem___10990 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 338, 44), shape_10989, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 338)
    subscript_call_result_10991 = invoke(stypy.reporting.localization.Localization(__file__, 338, 44), getitem___10990, int_10987)
    
    # Applying the binary operator '!=' (line 338)
    result_ne_10992 = python_operator(stypy.reporting.localization.Localization(__file__, 338, 29), '!=', subscript_call_result_10986, subscript_call_result_10991)
    
    # Applying the binary operator 'or' (line 338)
    result_or_keyword_10993 = python_operator(stypy.reporting.localization.Localization(__file__, 338, 7), 'or', result_ne_10981, result_ne_10992)
    
    # Testing the type of an if condition (line 338)
    if_condition_10994 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 338, 4), result_or_keyword_10993)
    # Assigning a type to the variable 'if_condition_10994' (line 338)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 338, 4), 'if_condition_10994', if_condition_10994)
    # SSA begins for if statement (line 338)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 339)
    # Processing the call arguments (line 339)
    str_10996 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 339, 25), 'str', 'expected square matrix')
    # Processing the call keyword arguments (line 339)
    kwargs_10997 = {}
    # Getting the type of 'ValueError' (line 339)
    ValueError_10995 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 339, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 339)
    ValueError_call_result_10998 = invoke(stypy.reporting.localization.Localization(__file__, 339, 14), ValueError_10995, *[str_10996], **kwargs_10997)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 339, 8), ValueError_call_result_10998, 'raise parameter', BaseException)
    # SSA join for if statement (line 338)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    
    # Obtaining the type of the subscript
    int_10999 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 340, 16), 'int')
    # Getting the type of 'a1' (line 340)
    a1_11000 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 340, 7), 'a1')
    # Obtaining the member 'shape' of a type (line 340)
    shape_11001 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 340, 7), a1_11000, 'shape')
    # Obtaining the member '__getitem__' of a type (line 340)
    getitem___11002 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 340, 7), shape_11001, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 340)
    subscript_call_result_11003 = invoke(stypy.reporting.localization.Localization(__file__, 340, 7), getitem___11002, int_10999)
    
    
    # Obtaining the type of the subscript
    int_11004 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 340, 31), 'int')
    # Getting the type of 'b1' (line 340)
    b1_11005 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 340, 22), 'b1')
    # Obtaining the member 'shape' of a type (line 340)
    shape_11006 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 340, 22), b1_11005, 'shape')
    # Obtaining the member '__getitem__' of a type (line 340)
    getitem___11007 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 340, 22), shape_11006, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 340)
    subscript_call_result_11008 = invoke(stypy.reporting.localization.Localization(__file__, 340, 22), getitem___11007, int_11004)
    
    # Applying the binary operator '!=' (line 340)
    result_ne_11009 = python_operator(stypy.reporting.localization.Localization(__file__, 340, 7), '!=', subscript_call_result_11003, subscript_call_result_11008)
    
    # Testing the type of an if condition (line 340)
    if_condition_11010 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 340, 4), result_ne_11009)
    # Assigning a type to the variable 'if_condition_11010' (line 340)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 340, 4), 'if_condition_11010', if_condition_11010)
    # SSA begins for if statement (line 340)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 341)
    # Processing the call arguments (line 341)
    str_11012 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 341, 25), 'str', 'incompatible dimensions')
    # Processing the call keyword arguments (line 341)
    kwargs_11013 = {}
    # Getting the type of 'ValueError' (line 341)
    ValueError_11011 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 341, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 341)
    ValueError_call_result_11014 = invoke(stypy.reporting.localization.Localization(__file__, 341, 14), ValueError_11011, *[str_11012], **kwargs_11013)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 341, 8), ValueError_call_result_11014, 'raise parameter', BaseException)
    # SSA join for if statement (line 340)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a BoolOp to a Name (line 342):
    
    # Assigning a BoolOp to a Name (line 342):
    
    # Evaluating a boolean operation
    # Getting the type of 'overwrite_b' (line 342)
    overwrite_b_11015 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 342, 18), 'overwrite_b')
    
    # Call to _datacopied(...): (line 342)
    # Processing the call arguments (line 342)
    # Getting the type of 'b1' (line 342)
    b1_11017 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 342, 45), 'b1', False)
    # Getting the type of 'b' (line 342)
    b_11018 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 342, 49), 'b', False)
    # Processing the call keyword arguments (line 342)
    kwargs_11019 = {}
    # Getting the type of '_datacopied' (line 342)
    _datacopied_11016 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 342, 33), '_datacopied', False)
    # Calling _datacopied(args, kwargs) (line 342)
    _datacopied_call_result_11020 = invoke(stypy.reporting.localization.Localization(__file__, 342, 33), _datacopied_11016, *[b1_11017, b_11018], **kwargs_11019)
    
    # Applying the binary operator 'or' (line 342)
    result_or_keyword_11021 = python_operator(stypy.reporting.localization.Localization(__file__, 342, 18), 'or', overwrite_b_11015, _datacopied_call_result_11020)
    
    # Assigning a type to the variable 'overwrite_b' (line 342)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 342, 4), 'overwrite_b', result_or_keyword_11021)
    
    # Getting the type of 'debug' (line 343)
    debug_11022 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 343, 7), 'debug')
    # Testing the type of an if condition (line 343)
    if_condition_11023 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 343, 4), debug_11022)
    # Assigning a type to the variable 'if_condition_11023' (line 343)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 343, 4), 'if_condition_11023', if_condition_11023)
    # SSA begins for if statement (line 343)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to print(...): (line 344)
    # Processing the call arguments (line 344)
    str_11025 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 344, 14), 'str', 'solve:overwrite_b=')
    # Getting the type of 'overwrite_b' (line 344)
    overwrite_b_11026 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 344, 36), 'overwrite_b', False)
    # Processing the call keyword arguments (line 344)
    kwargs_11027 = {}
    # Getting the type of 'print' (line 344)
    print_11024 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 344, 8), 'print', False)
    # Calling print(args, kwargs) (line 344)
    print_call_result_11028 = invoke(stypy.reporting.localization.Localization(__file__, 344, 8), print_11024, *[str_11025, overwrite_b_11026], **kwargs_11027)
    
    # SSA join for if statement (line 343)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 345):
    
    # Assigning a Call to a Name (line 345):
    
    # Call to get(...): (line 345)
    # Processing the call arguments (line 345)
    # Getting the type of 'trans' (line 345)
    trans_11037 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 345, 41), 'trans', False)
    # Getting the type of 'trans' (line 345)
    trans_11038 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 345, 48), 'trans', False)
    # Processing the call keyword arguments (line 345)
    kwargs_11039 = {}
    
    # Obtaining an instance of the builtin type 'dict' (line 345)
    dict_11029 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 345, 12), 'dict')
    # Adding type elements to the builtin type 'dict' instance (line 345)
    # Adding element type (key, value) (line 345)
    str_11030 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 345, 13), 'str', 'N')
    int_11031 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 345, 18), 'int')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 345, 12), dict_11029, (str_11030, int_11031))
    # Adding element type (key, value) (line 345)
    str_11032 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 345, 21), 'str', 'T')
    int_11033 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 345, 26), 'int')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 345, 12), dict_11029, (str_11032, int_11033))
    # Adding element type (key, value) (line 345)
    str_11034 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 345, 29), 'str', 'C')
    int_11035 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 345, 34), 'int')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 345, 12), dict_11029, (str_11034, int_11035))
    
    # Obtaining the member 'get' of a type (line 345)
    get_11036 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 345, 12), dict_11029, 'get')
    # Calling get(args, kwargs) (line 345)
    get_call_result_11040 = invoke(stypy.reporting.localization.Localization(__file__, 345, 12), get_11036, *[trans_11037, trans_11038], **kwargs_11039)
    
    # Assigning a type to the variable 'trans' (line 345)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 345, 4), 'trans', get_call_result_11040)
    
    # Assigning a Call to a Tuple (line 346):
    
    # Assigning a Subscript to a Name (line 346):
    
    # Obtaining the type of the subscript
    int_11041 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 346, 4), 'int')
    
    # Call to get_lapack_funcs(...): (line 346)
    # Processing the call arguments (line 346)
    
    # Obtaining an instance of the builtin type 'tuple' (line 346)
    tuple_11043 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 346, 31), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 346)
    # Adding element type (line 346)
    str_11044 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 346, 31), 'str', 'trtrs')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 346, 31), tuple_11043, str_11044)
    
    
    # Obtaining an instance of the builtin type 'tuple' (line 346)
    tuple_11045 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 346, 43), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 346)
    # Adding element type (line 346)
    # Getting the type of 'a1' (line 346)
    a1_11046 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 346, 43), 'a1', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 346, 43), tuple_11045, a1_11046)
    # Adding element type (line 346)
    # Getting the type of 'b1' (line 346)
    b1_11047 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 346, 47), 'b1', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 346, 43), tuple_11045, b1_11047)
    
    # Processing the call keyword arguments (line 346)
    kwargs_11048 = {}
    # Getting the type of 'get_lapack_funcs' (line 346)
    get_lapack_funcs_11042 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 346, 13), 'get_lapack_funcs', False)
    # Calling get_lapack_funcs(args, kwargs) (line 346)
    get_lapack_funcs_call_result_11049 = invoke(stypy.reporting.localization.Localization(__file__, 346, 13), get_lapack_funcs_11042, *[tuple_11043, tuple_11045], **kwargs_11048)
    
    # Obtaining the member '__getitem__' of a type (line 346)
    getitem___11050 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 346, 4), get_lapack_funcs_call_result_11049, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 346)
    subscript_call_result_11051 = invoke(stypy.reporting.localization.Localization(__file__, 346, 4), getitem___11050, int_11041)
    
    # Assigning a type to the variable 'tuple_var_assignment_10059' (line 346)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 346, 4), 'tuple_var_assignment_10059', subscript_call_result_11051)
    
    # Assigning a Name to a Name (line 346):
    # Getting the type of 'tuple_var_assignment_10059' (line 346)
    tuple_var_assignment_10059_11052 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 346, 4), 'tuple_var_assignment_10059')
    # Assigning a type to the variable 'trtrs' (line 346)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 346, 4), 'trtrs', tuple_var_assignment_10059_11052)
    
    # Assigning a Call to a Tuple (line 347):
    
    # Assigning a Subscript to a Name (line 347):
    
    # Obtaining the type of the subscript
    int_11053 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 347, 4), 'int')
    
    # Call to trtrs(...): (line 347)
    # Processing the call arguments (line 347)
    # Getting the type of 'a1' (line 347)
    a1_11055 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 347, 20), 'a1', False)
    # Getting the type of 'b1' (line 347)
    b1_11056 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 347, 24), 'b1', False)
    # Processing the call keyword arguments (line 347)
    # Getting the type of 'overwrite_b' (line 347)
    overwrite_b_11057 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 347, 40), 'overwrite_b', False)
    keyword_11058 = overwrite_b_11057
    # Getting the type of 'lower' (line 347)
    lower_11059 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 347, 59), 'lower', False)
    keyword_11060 = lower_11059
    # Getting the type of 'trans' (line 348)
    trans_11061 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 348, 26), 'trans', False)
    keyword_11062 = trans_11061
    # Getting the type of 'unit_diagonal' (line 348)
    unit_diagonal_11063 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 348, 42), 'unit_diagonal', False)
    keyword_11064 = unit_diagonal_11063
    kwargs_11065 = {'lower': keyword_11060, 'trans': keyword_11062, 'unitdiag': keyword_11064, 'overwrite_b': keyword_11058}
    # Getting the type of 'trtrs' (line 347)
    trtrs_11054 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 347, 14), 'trtrs', False)
    # Calling trtrs(args, kwargs) (line 347)
    trtrs_call_result_11066 = invoke(stypy.reporting.localization.Localization(__file__, 347, 14), trtrs_11054, *[a1_11055, b1_11056], **kwargs_11065)
    
    # Obtaining the member '__getitem__' of a type (line 347)
    getitem___11067 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 347, 4), trtrs_call_result_11066, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 347)
    subscript_call_result_11068 = invoke(stypy.reporting.localization.Localization(__file__, 347, 4), getitem___11067, int_11053)
    
    # Assigning a type to the variable 'tuple_var_assignment_10060' (line 347)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 347, 4), 'tuple_var_assignment_10060', subscript_call_result_11068)
    
    # Assigning a Subscript to a Name (line 347):
    
    # Obtaining the type of the subscript
    int_11069 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 347, 4), 'int')
    
    # Call to trtrs(...): (line 347)
    # Processing the call arguments (line 347)
    # Getting the type of 'a1' (line 347)
    a1_11071 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 347, 20), 'a1', False)
    # Getting the type of 'b1' (line 347)
    b1_11072 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 347, 24), 'b1', False)
    # Processing the call keyword arguments (line 347)
    # Getting the type of 'overwrite_b' (line 347)
    overwrite_b_11073 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 347, 40), 'overwrite_b', False)
    keyword_11074 = overwrite_b_11073
    # Getting the type of 'lower' (line 347)
    lower_11075 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 347, 59), 'lower', False)
    keyword_11076 = lower_11075
    # Getting the type of 'trans' (line 348)
    trans_11077 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 348, 26), 'trans', False)
    keyword_11078 = trans_11077
    # Getting the type of 'unit_diagonal' (line 348)
    unit_diagonal_11079 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 348, 42), 'unit_diagonal', False)
    keyword_11080 = unit_diagonal_11079
    kwargs_11081 = {'lower': keyword_11076, 'trans': keyword_11078, 'unitdiag': keyword_11080, 'overwrite_b': keyword_11074}
    # Getting the type of 'trtrs' (line 347)
    trtrs_11070 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 347, 14), 'trtrs', False)
    # Calling trtrs(args, kwargs) (line 347)
    trtrs_call_result_11082 = invoke(stypy.reporting.localization.Localization(__file__, 347, 14), trtrs_11070, *[a1_11071, b1_11072], **kwargs_11081)
    
    # Obtaining the member '__getitem__' of a type (line 347)
    getitem___11083 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 347, 4), trtrs_call_result_11082, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 347)
    subscript_call_result_11084 = invoke(stypy.reporting.localization.Localization(__file__, 347, 4), getitem___11083, int_11069)
    
    # Assigning a type to the variable 'tuple_var_assignment_10061' (line 347)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 347, 4), 'tuple_var_assignment_10061', subscript_call_result_11084)
    
    # Assigning a Name to a Name (line 347):
    # Getting the type of 'tuple_var_assignment_10060' (line 347)
    tuple_var_assignment_10060_11085 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 347, 4), 'tuple_var_assignment_10060')
    # Assigning a type to the variable 'x' (line 347)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 347, 4), 'x', tuple_var_assignment_10060_11085)
    
    # Assigning a Name to a Name (line 347):
    # Getting the type of 'tuple_var_assignment_10061' (line 347)
    tuple_var_assignment_10061_11086 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 347, 4), 'tuple_var_assignment_10061')
    # Assigning a type to the variable 'info' (line 347)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 347, 7), 'info', tuple_var_assignment_10061_11086)
    
    
    # Getting the type of 'info' (line 350)
    info_11087 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 350, 7), 'info')
    int_11088 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 350, 15), 'int')
    # Applying the binary operator '==' (line 350)
    result_eq_11089 = python_operator(stypy.reporting.localization.Localization(__file__, 350, 7), '==', info_11087, int_11088)
    
    # Testing the type of an if condition (line 350)
    if_condition_11090 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 350, 4), result_eq_11089)
    # Assigning a type to the variable 'if_condition_11090' (line 350)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 350, 4), 'if_condition_11090', if_condition_11090)
    # SSA begins for if statement (line 350)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # Getting the type of 'x' (line 351)
    x_11091 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 351, 15), 'x')
    # Assigning a type to the variable 'stypy_return_type' (line 351)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 351, 8), 'stypy_return_type', x_11091)
    # SSA join for if statement (line 350)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'info' (line 352)
    info_11092 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 352, 7), 'info')
    int_11093 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 352, 14), 'int')
    # Applying the binary operator '>' (line 352)
    result_gt_11094 = python_operator(stypy.reporting.localization.Localization(__file__, 352, 7), '>', info_11092, int_11093)
    
    # Testing the type of an if condition (line 352)
    if_condition_11095 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 352, 4), result_gt_11094)
    # Assigning a type to the variable 'if_condition_11095' (line 352)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 352, 4), 'if_condition_11095', if_condition_11095)
    # SSA begins for if statement (line 352)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to LinAlgError(...): (line 353)
    # Processing the call arguments (line 353)
    str_11097 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 353, 26), 'str', 'singular matrix: resolution failed at diagonal %d')
    # Getting the type of 'info' (line 354)
    info_11098 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 354, 27), 'info', False)
    int_11099 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 354, 32), 'int')
    # Applying the binary operator '-' (line 354)
    result_sub_11100 = python_operator(stypy.reporting.localization.Localization(__file__, 354, 27), '-', info_11098, int_11099)
    
    # Applying the binary operator '%' (line 353)
    result_mod_11101 = python_operator(stypy.reporting.localization.Localization(__file__, 353, 26), '%', str_11097, result_sub_11100)
    
    # Processing the call keyword arguments (line 353)
    kwargs_11102 = {}
    # Getting the type of 'LinAlgError' (line 353)
    LinAlgError_11096 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 353, 14), 'LinAlgError', False)
    # Calling LinAlgError(args, kwargs) (line 353)
    LinAlgError_call_result_11103 = invoke(stypy.reporting.localization.Localization(__file__, 353, 14), LinAlgError_11096, *[result_mod_11101], **kwargs_11102)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 353, 8), LinAlgError_call_result_11103, 'raise parameter', BaseException)
    # SSA join for if statement (line 352)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to ValueError(...): (line 355)
    # Processing the call arguments (line 355)
    str_11105 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 355, 21), 'str', 'illegal value in %d-th argument of internal trtrs')
    
    # Getting the type of 'info' (line 356)
    info_11106 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 356, 23), 'info', False)
    # Applying the 'usub' unary operator (line 356)
    result___neg___11107 = python_operator(stypy.reporting.localization.Localization(__file__, 356, 22), 'usub', info_11106)
    
    # Applying the binary operator '%' (line 355)
    result_mod_11108 = python_operator(stypy.reporting.localization.Localization(__file__, 355, 21), '%', str_11105, result___neg___11107)
    
    # Processing the call keyword arguments (line 355)
    kwargs_11109 = {}
    # Getting the type of 'ValueError' (line 355)
    ValueError_11104 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 355, 10), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 355)
    ValueError_call_result_11110 = invoke(stypy.reporting.localization.Localization(__file__, 355, 10), ValueError_11104, *[result_mod_11108], **kwargs_11109)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 355, 4), ValueError_call_result_11110, 'raise parameter', BaseException)
    
    # ################# End of 'solve_triangular(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'solve_triangular' in the type store
    # Getting the type of 'stypy_return_type' (line 262)
    stypy_return_type_11111 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 262, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_11111)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'solve_triangular'
    return stypy_return_type_11111

# Assigning a type to the variable 'solve_triangular' (line 262)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 262, 0), 'solve_triangular', solve_triangular)

@norecursion
def solve_banded(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'False' (line 359)
    False_11112 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 359, 46), 'False')
    # Getting the type of 'False' (line 359)
    False_11113 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 359, 65), 'False')
    # Getting the type of 'None' (line 360)
    None_11114 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 360, 23), 'None')
    # Getting the type of 'True' (line 360)
    True_11115 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 360, 42), 'True')
    defaults = [False_11112, False_11113, None_11114, True_11115]
    # Create a new context for function 'solve_banded'
    module_type_store = module_type_store.open_function_context('solve_banded', 359, 0, False)
    
    # Passed parameters checking function
    solve_banded.stypy_localization = localization
    solve_banded.stypy_type_of_self = None
    solve_banded.stypy_type_store = module_type_store
    solve_banded.stypy_function_name = 'solve_banded'
    solve_banded.stypy_param_names_list = ['l_and_u', 'ab', 'b', 'overwrite_ab', 'overwrite_b', 'debug', 'check_finite']
    solve_banded.stypy_varargs_param_name = None
    solve_banded.stypy_kwargs_param_name = None
    solve_banded.stypy_call_defaults = defaults
    solve_banded.stypy_call_varargs = varargs
    solve_banded.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'solve_banded', ['l_and_u', 'ab', 'b', 'overwrite_ab', 'overwrite_b', 'debug', 'check_finite'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'solve_banded', localization, ['l_and_u', 'ab', 'b', 'overwrite_ab', 'overwrite_b', 'debug', 'check_finite'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'solve_banded(...)' code ##################

    str_11116 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 426, (-1)), 'str', '\n    Solve the equation a x = b for x, assuming a is banded matrix.\n\n    The matrix a is stored in `ab` using the matrix diagonal ordered form::\n\n        ab[u + i - j, j] == a[i,j]\n\n    Example of `ab` (shape of a is (6,6), `u` =1, `l` =2)::\n\n        *    a01  a12  a23  a34  a45\n        a00  a11  a22  a33  a44  a55\n        a10  a21  a32  a43  a54   *\n        a20  a31  a42  a53   *    *\n\n    Parameters\n    ----------\n    (l, u) : (integer, integer)\n        Number of non-zero lower and upper diagonals\n    ab : (`l` + `u` + 1, M) array_like\n        Banded matrix\n    b : (M,) or (M, K) array_like\n        Right-hand side\n    overwrite_ab : bool, optional\n        Discard data in `ab` (may enhance performance)\n    overwrite_b : bool, optional\n        Discard data in `b` (may enhance performance)\n    check_finite : bool, optional\n        Whether to check that the input matrices contain only finite numbers.\n        Disabling may give a performance gain, but may result in problems\n        (crashes, non-termination) if the inputs do contain infinities or NaNs.\n\n    Returns\n    -------\n    x : (M,) or (M, K) ndarray\n        The solution to the system a x = b.  Returned shape depends on the\n        shape of `b`.\n\n    Examples\n    --------\n    Solve the banded system a x = b, where::\n\n            [5  2 -1  0  0]       [0]\n            [1  4  2 -1  0]       [1]\n        a = [0  1  3  2 -1]   b = [2]\n            [0  0  1  2  2]       [2]\n            [0  0  0  1  1]       [3]\n\n    There is one nonzero diagonal below the main diagonal (l = 1), and\n    two above (u = 2).  The diagonal banded form of the matrix is::\n\n             [*  * -1 -1 -1]\n        ab = [*  2  2  2  2]\n             [5  4  3  2  1]\n             [1  1  1  1  *]\n\n    >>> from scipy.linalg import solve_banded\n    >>> ab = np.array([[0,  0, -1, -1, -1],\n    ...                [0,  2,  2,  2,  2],\n    ...                [5,  4,  3,  2,  1],\n    ...                [1,  1,  1,  1,  0]])\n    >>> b = np.array([0, 1, 2, 2, 3])\n    >>> x = solve_banded((1, 2), ab, b)\n    >>> x\n    array([-2.37288136,  3.93220339, -4.        ,  4.3559322 , -1.3559322 ])\n\n    ')
    
    # Type idiom detected: calculating its left and rigth part (line 429)
    # Getting the type of 'debug' (line 429)
    debug_11117 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 429, 4), 'debug')
    # Getting the type of 'None' (line 429)
    None_11118 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 429, 20), 'None')
    
    (may_be_11119, more_types_in_union_11120) = may_not_be_none(debug_11117, None_11118)

    if may_be_11119:

        if more_types_in_union_11120:
            # Runtime conditional SSA (line 429)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Call to warn(...): (line 430)
        # Processing the call arguments (line 430)
        str_11123 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 430, 22), 'str', 'Use of the "debug" keyword is deprecated and this keyword will be removed in the future versions of SciPy.')
        # Getting the type of 'DeprecationWarning' (line 432)
        DeprecationWarning_11124 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 432, 44), 'DeprecationWarning', False)
        # Processing the call keyword arguments (line 430)
        kwargs_11125 = {}
        # Getting the type of 'warnings' (line 430)
        warnings_11121 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 430, 8), 'warnings', False)
        # Obtaining the member 'warn' of a type (line 430)
        warn_11122 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 430, 8), warnings_11121, 'warn')
        # Calling warn(args, kwargs) (line 430)
        warn_call_result_11126 = invoke(stypy.reporting.localization.Localization(__file__, 430, 8), warn_11122, *[str_11123, DeprecationWarning_11124], **kwargs_11125)
        

        if more_types_in_union_11120:
            # SSA join for if statement (line 429)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Assigning a Call to a Name (line 434):
    
    # Assigning a Call to a Name (line 434):
    
    # Call to _asarray_validated(...): (line 434)
    # Processing the call arguments (line 434)
    # Getting the type of 'ab' (line 434)
    ab_11128 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 434, 28), 'ab', False)
    # Processing the call keyword arguments (line 434)
    # Getting the type of 'check_finite' (line 434)
    check_finite_11129 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 434, 45), 'check_finite', False)
    keyword_11130 = check_finite_11129
    # Getting the type of 'True' (line 434)
    True_11131 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 434, 70), 'True', False)
    keyword_11132 = True_11131
    kwargs_11133 = {'as_inexact': keyword_11132, 'check_finite': keyword_11130}
    # Getting the type of '_asarray_validated' (line 434)
    _asarray_validated_11127 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 434, 9), '_asarray_validated', False)
    # Calling _asarray_validated(args, kwargs) (line 434)
    _asarray_validated_call_result_11134 = invoke(stypy.reporting.localization.Localization(__file__, 434, 9), _asarray_validated_11127, *[ab_11128], **kwargs_11133)
    
    # Assigning a type to the variable 'a1' (line 434)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 434, 4), 'a1', _asarray_validated_call_result_11134)
    
    # Assigning a Call to a Name (line 435):
    
    # Assigning a Call to a Name (line 435):
    
    # Call to _asarray_validated(...): (line 435)
    # Processing the call arguments (line 435)
    # Getting the type of 'b' (line 435)
    b_11136 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 435, 28), 'b', False)
    # Processing the call keyword arguments (line 435)
    # Getting the type of 'check_finite' (line 435)
    check_finite_11137 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 435, 44), 'check_finite', False)
    keyword_11138 = check_finite_11137
    # Getting the type of 'True' (line 435)
    True_11139 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 435, 69), 'True', False)
    keyword_11140 = True_11139
    kwargs_11141 = {'as_inexact': keyword_11140, 'check_finite': keyword_11138}
    # Getting the type of '_asarray_validated' (line 435)
    _asarray_validated_11135 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 435, 9), '_asarray_validated', False)
    # Calling _asarray_validated(args, kwargs) (line 435)
    _asarray_validated_call_result_11142 = invoke(stypy.reporting.localization.Localization(__file__, 435, 9), _asarray_validated_11135, *[b_11136], **kwargs_11141)
    
    # Assigning a type to the variable 'b1' (line 435)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 435, 4), 'b1', _asarray_validated_call_result_11142)
    
    
    
    # Obtaining the type of the subscript
    int_11143 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 437, 16), 'int')
    # Getting the type of 'a1' (line 437)
    a1_11144 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 437, 7), 'a1')
    # Obtaining the member 'shape' of a type (line 437)
    shape_11145 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 437, 7), a1_11144, 'shape')
    # Obtaining the member '__getitem__' of a type (line 437)
    getitem___11146 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 437, 7), shape_11145, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 437)
    subscript_call_result_11147 = invoke(stypy.reporting.localization.Localization(__file__, 437, 7), getitem___11146, int_11143)
    
    
    # Obtaining the type of the subscript
    int_11148 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 437, 32), 'int')
    # Getting the type of 'b1' (line 437)
    b1_11149 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 437, 23), 'b1')
    # Obtaining the member 'shape' of a type (line 437)
    shape_11150 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 437, 23), b1_11149, 'shape')
    # Obtaining the member '__getitem__' of a type (line 437)
    getitem___11151 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 437, 23), shape_11150, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 437)
    subscript_call_result_11152 = invoke(stypy.reporting.localization.Localization(__file__, 437, 23), getitem___11151, int_11148)
    
    # Applying the binary operator '!=' (line 437)
    result_ne_11153 = python_operator(stypy.reporting.localization.Localization(__file__, 437, 7), '!=', subscript_call_result_11147, subscript_call_result_11152)
    
    # Testing the type of an if condition (line 437)
    if_condition_11154 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 437, 4), result_ne_11153)
    # Assigning a type to the variable 'if_condition_11154' (line 437)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 437, 4), 'if_condition_11154', if_condition_11154)
    # SSA begins for if statement (line 437)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 438)
    # Processing the call arguments (line 438)
    str_11156 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 438, 25), 'str', 'shapes of ab and b are not compatible.')
    # Processing the call keyword arguments (line 438)
    kwargs_11157 = {}
    # Getting the type of 'ValueError' (line 438)
    ValueError_11155 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 438, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 438)
    ValueError_call_result_11158 = invoke(stypy.reporting.localization.Localization(__file__, 438, 14), ValueError_11155, *[str_11156], **kwargs_11157)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 438, 8), ValueError_call_result_11158, 'raise parameter', BaseException)
    # SSA join for if statement (line 437)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Name to a Tuple (line 439):
    
    # Assigning a Subscript to a Name (line 439):
    
    # Obtaining the type of the subscript
    int_11159 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 439, 4), 'int')
    # Getting the type of 'l_and_u' (line 439)
    l_and_u_11160 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 439, 13), 'l_and_u')
    # Obtaining the member '__getitem__' of a type (line 439)
    getitem___11161 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 439, 4), l_and_u_11160, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 439)
    subscript_call_result_11162 = invoke(stypy.reporting.localization.Localization(__file__, 439, 4), getitem___11161, int_11159)
    
    # Assigning a type to the variable 'tuple_var_assignment_10062' (line 439)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 439, 4), 'tuple_var_assignment_10062', subscript_call_result_11162)
    
    # Assigning a Subscript to a Name (line 439):
    
    # Obtaining the type of the subscript
    int_11163 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 439, 4), 'int')
    # Getting the type of 'l_and_u' (line 439)
    l_and_u_11164 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 439, 13), 'l_and_u')
    # Obtaining the member '__getitem__' of a type (line 439)
    getitem___11165 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 439, 4), l_and_u_11164, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 439)
    subscript_call_result_11166 = invoke(stypy.reporting.localization.Localization(__file__, 439, 4), getitem___11165, int_11163)
    
    # Assigning a type to the variable 'tuple_var_assignment_10063' (line 439)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 439, 4), 'tuple_var_assignment_10063', subscript_call_result_11166)
    
    # Assigning a Name to a Name (line 439):
    # Getting the type of 'tuple_var_assignment_10062' (line 439)
    tuple_var_assignment_10062_11167 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 439, 4), 'tuple_var_assignment_10062')
    # Assigning a type to the variable 'l' (line 439)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 439, 5), 'l', tuple_var_assignment_10062_11167)
    
    # Assigning a Name to a Name (line 439):
    # Getting the type of 'tuple_var_assignment_10063' (line 439)
    tuple_var_assignment_10063_11168 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 439, 4), 'tuple_var_assignment_10063')
    # Assigning a type to the variable 'u' (line 439)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 439, 8), 'u', tuple_var_assignment_10063_11168)
    
    
    # Getting the type of 'l' (line 440)
    l_11169 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 440, 7), 'l')
    # Getting the type of 'u' (line 440)
    u_11170 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 440, 11), 'u')
    # Applying the binary operator '+' (line 440)
    result_add_11171 = python_operator(stypy.reporting.localization.Localization(__file__, 440, 7), '+', l_11169, u_11170)
    
    int_11172 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 440, 15), 'int')
    # Applying the binary operator '+' (line 440)
    result_add_11173 = python_operator(stypy.reporting.localization.Localization(__file__, 440, 13), '+', result_add_11171, int_11172)
    
    
    # Obtaining the type of the subscript
    int_11174 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 440, 29), 'int')
    # Getting the type of 'a1' (line 440)
    a1_11175 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 440, 20), 'a1')
    # Obtaining the member 'shape' of a type (line 440)
    shape_11176 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 440, 20), a1_11175, 'shape')
    # Obtaining the member '__getitem__' of a type (line 440)
    getitem___11177 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 440, 20), shape_11176, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 440)
    subscript_call_result_11178 = invoke(stypy.reporting.localization.Localization(__file__, 440, 20), getitem___11177, int_11174)
    
    # Applying the binary operator '!=' (line 440)
    result_ne_11179 = python_operator(stypy.reporting.localization.Localization(__file__, 440, 7), '!=', result_add_11173, subscript_call_result_11178)
    
    # Testing the type of an if condition (line 440)
    if_condition_11180 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 440, 4), result_ne_11179)
    # Assigning a type to the variable 'if_condition_11180' (line 440)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 440, 4), 'if_condition_11180', if_condition_11180)
    # SSA begins for if statement (line 440)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 441)
    # Processing the call arguments (line 441)
    str_11182 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 441, 25), 'str', 'invalid values for the number of lower and upper diagonals: l+u+1 (%d) does not equal ab.shape[0] (%d)')
    
    # Obtaining an instance of the builtin type 'tuple' (line 443)
    tuple_11183 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 443, 35), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 443)
    # Adding element type (line 443)
    # Getting the type of 'l' (line 443)
    l_11184 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 443, 35), 'l', False)
    # Getting the type of 'u' (line 443)
    u_11185 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 443, 37), 'u', False)
    # Applying the binary operator '+' (line 443)
    result_add_11186 = python_operator(stypy.reporting.localization.Localization(__file__, 443, 35), '+', l_11184, u_11185)
    
    int_11187 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 443, 39), 'int')
    # Applying the binary operator '+' (line 443)
    result_add_11188 = python_operator(stypy.reporting.localization.Localization(__file__, 443, 38), '+', result_add_11186, int_11187)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 443, 35), tuple_11183, result_add_11188)
    # Adding element type (line 443)
    
    # Obtaining the type of the subscript
    int_11189 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 443, 51), 'int')
    # Getting the type of 'ab' (line 443)
    ab_11190 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 443, 42), 'ab', False)
    # Obtaining the member 'shape' of a type (line 443)
    shape_11191 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 443, 42), ab_11190, 'shape')
    # Obtaining the member '__getitem__' of a type (line 443)
    getitem___11192 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 443, 42), shape_11191, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 443)
    subscript_call_result_11193 = invoke(stypy.reporting.localization.Localization(__file__, 443, 42), getitem___11192, int_11189)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 443, 35), tuple_11183, subscript_call_result_11193)
    
    # Applying the binary operator '%' (line 441)
    result_mod_11194 = python_operator(stypy.reporting.localization.Localization(__file__, 441, 25), '%', str_11182, tuple_11183)
    
    # Processing the call keyword arguments (line 441)
    kwargs_11195 = {}
    # Getting the type of 'ValueError' (line 441)
    ValueError_11181 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 441, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 441)
    ValueError_call_result_11196 = invoke(stypy.reporting.localization.Localization(__file__, 441, 14), ValueError_11181, *[result_mod_11194], **kwargs_11195)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 441, 8), ValueError_call_result_11196, 'raise parameter', BaseException)
    # SSA join for if statement (line 440)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a BoolOp to a Name (line 445):
    
    # Assigning a BoolOp to a Name (line 445):
    
    # Evaluating a boolean operation
    # Getting the type of 'overwrite_b' (line 445)
    overwrite_b_11197 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 445, 18), 'overwrite_b')
    
    # Call to _datacopied(...): (line 445)
    # Processing the call arguments (line 445)
    # Getting the type of 'b1' (line 445)
    b1_11199 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 445, 45), 'b1', False)
    # Getting the type of 'b' (line 445)
    b_11200 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 445, 49), 'b', False)
    # Processing the call keyword arguments (line 445)
    kwargs_11201 = {}
    # Getting the type of '_datacopied' (line 445)
    _datacopied_11198 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 445, 33), '_datacopied', False)
    # Calling _datacopied(args, kwargs) (line 445)
    _datacopied_call_result_11202 = invoke(stypy.reporting.localization.Localization(__file__, 445, 33), _datacopied_11198, *[b1_11199, b_11200], **kwargs_11201)
    
    # Applying the binary operator 'or' (line 445)
    result_or_keyword_11203 = python_operator(stypy.reporting.localization.Localization(__file__, 445, 18), 'or', overwrite_b_11197, _datacopied_call_result_11202)
    
    # Assigning a type to the variable 'overwrite_b' (line 445)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 445, 4), 'overwrite_b', result_or_keyword_11203)
    
    
    
    # Obtaining the type of the subscript
    int_11204 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 446, 16), 'int')
    # Getting the type of 'a1' (line 446)
    a1_11205 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 446, 7), 'a1')
    # Obtaining the member 'shape' of a type (line 446)
    shape_11206 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 446, 7), a1_11205, 'shape')
    # Obtaining the member '__getitem__' of a type (line 446)
    getitem___11207 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 446, 7), shape_11206, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 446)
    subscript_call_result_11208 = invoke(stypy.reporting.localization.Localization(__file__, 446, 7), getitem___11207, int_11204)
    
    int_11209 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 446, 23), 'int')
    # Applying the binary operator '==' (line 446)
    result_eq_11210 = python_operator(stypy.reporting.localization.Localization(__file__, 446, 7), '==', subscript_call_result_11208, int_11209)
    
    # Testing the type of an if condition (line 446)
    if_condition_11211 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 446, 4), result_eq_11210)
    # Assigning a type to the variable 'if_condition_11211' (line 446)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 446, 4), 'if_condition_11211', if_condition_11211)
    # SSA begins for if statement (line 446)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 447):
    
    # Assigning a Call to a Name (line 447):
    
    # Call to array(...): (line 447)
    # Processing the call arguments (line 447)
    # Getting the type of 'b1' (line 447)
    b1_11214 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 447, 22), 'b1', False)
    # Processing the call keyword arguments (line 447)
    
    # Getting the type of 'overwrite_b' (line 447)
    overwrite_b_11215 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 447, 36), 'overwrite_b', False)
    # Applying the 'not' unary operator (line 447)
    result_not__11216 = python_operator(stypy.reporting.localization.Localization(__file__, 447, 32), 'not', overwrite_b_11215)
    
    keyword_11217 = result_not__11216
    kwargs_11218 = {'copy': keyword_11217}
    # Getting the type of 'np' (line 447)
    np_11212 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 447, 13), 'np', False)
    # Obtaining the member 'array' of a type (line 447)
    array_11213 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 447, 13), np_11212, 'array')
    # Calling array(args, kwargs) (line 447)
    array_call_result_11219 = invoke(stypy.reporting.localization.Localization(__file__, 447, 13), array_11213, *[b1_11214], **kwargs_11218)
    
    # Assigning a type to the variable 'b2' (line 447)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 447, 8), 'b2', array_call_result_11219)
    
    # Getting the type of 'b2' (line 448)
    b2_11220 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 448, 8), 'b2')
    
    # Obtaining the type of the subscript
    
    # Obtaining an instance of the builtin type 'tuple' (line 448)
    tuple_11221 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 448, 17), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 448)
    # Adding element type (line 448)
    int_11222 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 448, 17), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 448, 17), tuple_11221, int_11222)
    # Adding element type (line 448)
    int_11223 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 448, 20), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 448, 17), tuple_11221, int_11223)
    
    # Getting the type of 'a1' (line 448)
    a1_11224 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 448, 14), 'a1')
    # Obtaining the member '__getitem__' of a type (line 448)
    getitem___11225 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 448, 14), a1_11224, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 448)
    subscript_call_result_11226 = invoke(stypy.reporting.localization.Localization(__file__, 448, 14), getitem___11225, tuple_11221)
    
    # Applying the binary operator 'div=' (line 448)
    result_div_11227 = python_operator(stypy.reporting.localization.Localization(__file__, 448, 8), 'div=', b2_11220, subscript_call_result_11226)
    # Assigning a type to the variable 'b2' (line 448)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 448, 8), 'b2', result_div_11227)
    
    # Getting the type of 'b2' (line 449)
    b2_11228 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 449, 15), 'b2')
    # Assigning a type to the variable 'stypy_return_type' (line 449)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 449, 8), 'stypy_return_type', b2_11228)
    # SSA join for if statement (line 446)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'l' (line 450)
    l_11229 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 450, 7), 'l')
    # Getting the type of 'u' (line 450)
    u_11230 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 450, 12), 'u')
    # Applying the binary operator '==' (line 450)
    result_eq_11231 = python_operator(stypy.reporting.localization.Localization(__file__, 450, 7), '==', l_11229, u_11230)
    int_11232 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 450, 17), 'int')
    # Applying the binary operator '==' (line 450)
    result_eq_11233 = python_operator(stypy.reporting.localization.Localization(__file__, 450, 7), '==', u_11230, int_11232)
    # Applying the binary operator '&' (line 450)
    result_and__11234 = python_operator(stypy.reporting.localization.Localization(__file__, 450, 7), '&', result_eq_11231, result_eq_11233)
    
    # Testing the type of an if condition (line 450)
    if_condition_11235 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 450, 4), result_and__11234)
    # Assigning a type to the variable 'if_condition_11235' (line 450)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 450, 4), 'if_condition_11235', if_condition_11235)
    # SSA begins for if statement (line 450)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a BoolOp to a Name (line 451):
    
    # Assigning a BoolOp to a Name (line 451):
    
    # Evaluating a boolean operation
    # Getting the type of 'overwrite_ab' (line 451)
    overwrite_ab_11236 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 451, 23), 'overwrite_ab')
    
    # Call to _datacopied(...): (line 451)
    # Processing the call arguments (line 451)
    # Getting the type of 'a1' (line 451)
    a1_11238 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 451, 51), 'a1', False)
    # Getting the type of 'ab' (line 451)
    ab_11239 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 451, 55), 'ab', False)
    # Processing the call keyword arguments (line 451)
    kwargs_11240 = {}
    # Getting the type of '_datacopied' (line 451)
    _datacopied_11237 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 451, 39), '_datacopied', False)
    # Calling _datacopied(args, kwargs) (line 451)
    _datacopied_call_result_11241 = invoke(stypy.reporting.localization.Localization(__file__, 451, 39), _datacopied_11237, *[a1_11238, ab_11239], **kwargs_11240)
    
    # Applying the binary operator 'or' (line 451)
    result_or_keyword_11242 = python_operator(stypy.reporting.localization.Localization(__file__, 451, 23), 'or', overwrite_ab_11236, _datacopied_call_result_11241)
    
    # Assigning a type to the variable 'overwrite_ab' (line 451)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 451, 8), 'overwrite_ab', result_or_keyword_11242)
    
    # Assigning a Call to a Tuple (line 452):
    
    # Assigning a Subscript to a Name (line 452):
    
    # Obtaining the type of the subscript
    int_11243 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 452, 8), 'int')
    
    # Call to get_lapack_funcs(...): (line 452)
    # Processing the call arguments (line 452)
    
    # Obtaining an instance of the builtin type 'tuple' (line 452)
    tuple_11245 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 452, 34), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 452)
    # Adding element type (line 452)
    str_11246 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 452, 34), 'str', 'gtsv')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 452, 34), tuple_11245, str_11246)
    
    
    # Obtaining an instance of the builtin type 'tuple' (line 452)
    tuple_11247 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 452, 45), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 452)
    # Adding element type (line 452)
    # Getting the type of 'a1' (line 452)
    a1_11248 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 452, 45), 'a1', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 452, 45), tuple_11247, a1_11248)
    # Adding element type (line 452)
    # Getting the type of 'b1' (line 452)
    b1_11249 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 452, 49), 'b1', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 452, 45), tuple_11247, b1_11249)
    
    # Processing the call keyword arguments (line 452)
    kwargs_11250 = {}
    # Getting the type of 'get_lapack_funcs' (line 452)
    get_lapack_funcs_11244 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 452, 16), 'get_lapack_funcs', False)
    # Calling get_lapack_funcs(args, kwargs) (line 452)
    get_lapack_funcs_call_result_11251 = invoke(stypy.reporting.localization.Localization(__file__, 452, 16), get_lapack_funcs_11244, *[tuple_11245, tuple_11247], **kwargs_11250)
    
    # Obtaining the member '__getitem__' of a type (line 452)
    getitem___11252 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 452, 8), get_lapack_funcs_call_result_11251, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 452)
    subscript_call_result_11253 = invoke(stypy.reporting.localization.Localization(__file__, 452, 8), getitem___11252, int_11243)
    
    # Assigning a type to the variable 'tuple_var_assignment_10064' (line 452)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 452, 8), 'tuple_var_assignment_10064', subscript_call_result_11253)
    
    # Assigning a Name to a Name (line 452):
    # Getting the type of 'tuple_var_assignment_10064' (line 452)
    tuple_var_assignment_10064_11254 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 452, 8), 'tuple_var_assignment_10064')
    # Assigning a type to the variable 'gtsv' (line 452)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 452, 8), 'gtsv', tuple_var_assignment_10064_11254)
    
    # Assigning a Subscript to a Name (line 453):
    
    # Assigning a Subscript to a Name (line 453):
    
    # Obtaining the type of the subscript
    int_11255 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 453, 16), 'int')
    int_11256 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 453, 19), 'int')
    slice_11257 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 453, 13), int_11256, None, None)
    # Getting the type of 'a1' (line 453)
    a1_11258 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 453, 13), 'a1')
    # Obtaining the member '__getitem__' of a type (line 453)
    getitem___11259 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 453, 13), a1_11258, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 453)
    subscript_call_result_11260 = invoke(stypy.reporting.localization.Localization(__file__, 453, 13), getitem___11259, (int_11255, slice_11257))
    
    # Assigning a type to the variable 'du' (line 453)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 453, 8), 'du', subscript_call_result_11260)
    
    # Assigning a Subscript to a Name (line 454):
    
    # Assigning a Subscript to a Name (line 454):
    
    # Obtaining the type of the subscript
    int_11261 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 454, 15), 'int')
    slice_11262 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 454, 12), None, None, None)
    # Getting the type of 'a1' (line 454)
    a1_11263 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 454, 12), 'a1')
    # Obtaining the member '__getitem__' of a type (line 454)
    getitem___11264 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 454, 12), a1_11263, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 454)
    subscript_call_result_11265 = invoke(stypy.reporting.localization.Localization(__file__, 454, 12), getitem___11264, (int_11261, slice_11262))
    
    # Assigning a type to the variable 'd' (line 454)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 454, 8), 'd', subscript_call_result_11265)
    
    # Assigning a Subscript to a Name (line 455):
    
    # Assigning a Subscript to a Name (line 455):
    
    # Obtaining the type of the subscript
    int_11266 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 455, 16), 'int')
    int_11267 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 455, 20), 'int')
    slice_11268 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 455, 13), None, int_11267, None)
    # Getting the type of 'a1' (line 455)
    a1_11269 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 455, 13), 'a1')
    # Obtaining the member '__getitem__' of a type (line 455)
    getitem___11270 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 455, 13), a1_11269, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 455)
    subscript_call_result_11271 = invoke(stypy.reporting.localization.Localization(__file__, 455, 13), getitem___11270, (int_11266, slice_11268))
    
    # Assigning a type to the variable 'dl' (line 455)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 455, 8), 'dl', subscript_call_result_11271)
    
    # Assigning a Call to a Tuple (line 456):
    
    # Assigning a Subscript to a Name (line 456):
    
    # Obtaining the type of the subscript
    int_11272 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 456, 8), 'int')
    
    # Call to gtsv(...): (line 456)
    # Processing the call arguments (line 456)
    # Getting the type of 'dl' (line 456)
    dl_11274 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 456, 35), 'dl', False)
    # Getting the type of 'd' (line 456)
    d_11275 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 456, 39), 'd', False)
    # Getting the type of 'du' (line 456)
    du_11276 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 456, 42), 'du', False)
    # Getting the type of 'b1' (line 456)
    b1_11277 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 456, 46), 'b1', False)
    # Getting the type of 'overwrite_ab' (line 456)
    overwrite_ab_11278 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 456, 50), 'overwrite_ab', False)
    # Getting the type of 'overwrite_ab' (line 456)
    overwrite_ab_11279 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 456, 64), 'overwrite_ab', False)
    # Getting the type of 'overwrite_ab' (line 457)
    overwrite_ab_11280 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 457, 35), 'overwrite_ab', False)
    # Getting the type of 'overwrite_b' (line 457)
    overwrite_b_11281 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 457, 49), 'overwrite_b', False)
    # Processing the call keyword arguments (line 456)
    kwargs_11282 = {}
    # Getting the type of 'gtsv' (line 456)
    gtsv_11273 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 456, 30), 'gtsv', False)
    # Calling gtsv(args, kwargs) (line 456)
    gtsv_call_result_11283 = invoke(stypy.reporting.localization.Localization(__file__, 456, 30), gtsv_11273, *[dl_11274, d_11275, du_11276, b1_11277, overwrite_ab_11278, overwrite_ab_11279, overwrite_ab_11280, overwrite_b_11281], **kwargs_11282)
    
    # Obtaining the member '__getitem__' of a type (line 456)
    getitem___11284 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 456, 8), gtsv_call_result_11283, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 456)
    subscript_call_result_11285 = invoke(stypy.reporting.localization.Localization(__file__, 456, 8), getitem___11284, int_11272)
    
    # Assigning a type to the variable 'tuple_var_assignment_10065' (line 456)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 456, 8), 'tuple_var_assignment_10065', subscript_call_result_11285)
    
    # Assigning a Subscript to a Name (line 456):
    
    # Obtaining the type of the subscript
    int_11286 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 456, 8), 'int')
    
    # Call to gtsv(...): (line 456)
    # Processing the call arguments (line 456)
    # Getting the type of 'dl' (line 456)
    dl_11288 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 456, 35), 'dl', False)
    # Getting the type of 'd' (line 456)
    d_11289 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 456, 39), 'd', False)
    # Getting the type of 'du' (line 456)
    du_11290 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 456, 42), 'du', False)
    # Getting the type of 'b1' (line 456)
    b1_11291 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 456, 46), 'b1', False)
    # Getting the type of 'overwrite_ab' (line 456)
    overwrite_ab_11292 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 456, 50), 'overwrite_ab', False)
    # Getting the type of 'overwrite_ab' (line 456)
    overwrite_ab_11293 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 456, 64), 'overwrite_ab', False)
    # Getting the type of 'overwrite_ab' (line 457)
    overwrite_ab_11294 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 457, 35), 'overwrite_ab', False)
    # Getting the type of 'overwrite_b' (line 457)
    overwrite_b_11295 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 457, 49), 'overwrite_b', False)
    # Processing the call keyword arguments (line 456)
    kwargs_11296 = {}
    # Getting the type of 'gtsv' (line 456)
    gtsv_11287 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 456, 30), 'gtsv', False)
    # Calling gtsv(args, kwargs) (line 456)
    gtsv_call_result_11297 = invoke(stypy.reporting.localization.Localization(__file__, 456, 30), gtsv_11287, *[dl_11288, d_11289, du_11290, b1_11291, overwrite_ab_11292, overwrite_ab_11293, overwrite_ab_11294, overwrite_b_11295], **kwargs_11296)
    
    # Obtaining the member '__getitem__' of a type (line 456)
    getitem___11298 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 456, 8), gtsv_call_result_11297, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 456)
    subscript_call_result_11299 = invoke(stypy.reporting.localization.Localization(__file__, 456, 8), getitem___11298, int_11286)
    
    # Assigning a type to the variable 'tuple_var_assignment_10066' (line 456)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 456, 8), 'tuple_var_assignment_10066', subscript_call_result_11299)
    
    # Assigning a Subscript to a Name (line 456):
    
    # Obtaining the type of the subscript
    int_11300 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 456, 8), 'int')
    
    # Call to gtsv(...): (line 456)
    # Processing the call arguments (line 456)
    # Getting the type of 'dl' (line 456)
    dl_11302 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 456, 35), 'dl', False)
    # Getting the type of 'd' (line 456)
    d_11303 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 456, 39), 'd', False)
    # Getting the type of 'du' (line 456)
    du_11304 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 456, 42), 'du', False)
    # Getting the type of 'b1' (line 456)
    b1_11305 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 456, 46), 'b1', False)
    # Getting the type of 'overwrite_ab' (line 456)
    overwrite_ab_11306 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 456, 50), 'overwrite_ab', False)
    # Getting the type of 'overwrite_ab' (line 456)
    overwrite_ab_11307 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 456, 64), 'overwrite_ab', False)
    # Getting the type of 'overwrite_ab' (line 457)
    overwrite_ab_11308 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 457, 35), 'overwrite_ab', False)
    # Getting the type of 'overwrite_b' (line 457)
    overwrite_b_11309 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 457, 49), 'overwrite_b', False)
    # Processing the call keyword arguments (line 456)
    kwargs_11310 = {}
    # Getting the type of 'gtsv' (line 456)
    gtsv_11301 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 456, 30), 'gtsv', False)
    # Calling gtsv(args, kwargs) (line 456)
    gtsv_call_result_11311 = invoke(stypy.reporting.localization.Localization(__file__, 456, 30), gtsv_11301, *[dl_11302, d_11303, du_11304, b1_11305, overwrite_ab_11306, overwrite_ab_11307, overwrite_ab_11308, overwrite_b_11309], **kwargs_11310)
    
    # Obtaining the member '__getitem__' of a type (line 456)
    getitem___11312 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 456, 8), gtsv_call_result_11311, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 456)
    subscript_call_result_11313 = invoke(stypy.reporting.localization.Localization(__file__, 456, 8), getitem___11312, int_11300)
    
    # Assigning a type to the variable 'tuple_var_assignment_10067' (line 456)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 456, 8), 'tuple_var_assignment_10067', subscript_call_result_11313)
    
    # Assigning a Subscript to a Name (line 456):
    
    # Obtaining the type of the subscript
    int_11314 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 456, 8), 'int')
    
    # Call to gtsv(...): (line 456)
    # Processing the call arguments (line 456)
    # Getting the type of 'dl' (line 456)
    dl_11316 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 456, 35), 'dl', False)
    # Getting the type of 'd' (line 456)
    d_11317 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 456, 39), 'd', False)
    # Getting the type of 'du' (line 456)
    du_11318 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 456, 42), 'du', False)
    # Getting the type of 'b1' (line 456)
    b1_11319 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 456, 46), 'b1', False)
    # Getting the type of 'overwrite_ab' (line 456)
    overwrite_ab_11320 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 456, 50), 'overwrite_ab', False)
    # Getting the type of 'overwrite_ab' (line 456)
    overwrite_ab_11321 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 456, 64), 'overwrite_ab', False)
    # Getting the type of 'overwrite_ab' (line 457)
    overwrite_ab_11322 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 457, 35), 'overwrite_ab', False)
    # Getting the type of 'overwrite_b' (line 457)
    overwrite_b_11323 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 457, 49), 'overwrite_b', False)
    # Processing the call keyword arguments (line 456)
    kwargs_11324 = {}
    # Getting the type of 'gtsv' (line 456)
    gtsv_11315 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 456, 30), 'gtsv', False)
    # Calling gtsv(args, kwargs) (line 456)
    gtsv_call_result_11325 = invoke(stypy.reporting.localization.Localization(__file__, 456, 30), gtsv_11315, *[dl_11316, d_11317, du_11318, b1_11319, overwrite_ab_11320, overwrite_ab_11321, overwrite_ab_11322, overwrite_b_11323], **kwargs_11324)
    
    # Obtaining the member '__getitem__' of a type (line 456)
    getitem___11326 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 456, 8), gtsv_call_result_11325, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 456)
    subscript_call_result_11327 = invoke(stypy.reporting.localization.Localization(__file__, 456, 8), getitem___11326, int_11314)
    
    # Assigning a type to the variable 'tuple_var_assignment_10068' (line 456)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 456, 8), 'tuple_var_assignment_10068', subscript_call_result_11327)
    
    # Assigning a Subscript to a Name (line 456):
    
    # Obtaining the type of the subscript
    int_11328 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 456, 8), 'int')
    
    # Call to gtsv(...): (line 456)
    # Processing the call arguments (line 456)
    # Getting the type of 'dl' (line 456)
    dl_11330 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 456, 35), 'dl', False)
    # Getting the type of 'd' (line 456)
    d_11331 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 456, 39), 'd', False)
    # Getting the type of 'du' (line 456)
    du_11332 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 456, 42), 'du', False)
    # Getting the type of 'b1' (line 456)
    b1_11333 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 456, 46), 'b1', False)
    # Getting the type of 'overwrite_ab' (line 456)
    overwrite_ab_11334 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 456, 50), 'overwrite_ab', False)
    # Getting the type of 'overwrite_ab' (line 456)
    overwrite_ab_11335 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 456, 64), 'overwrite_ab', False)
    # Getting the type of 'overwrite_ab' (line 457)
    overwrite_ab_11336 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 457, 35), 'overwrite_ab', False)
    # Getting the type of 'overwrite_b' (line 457)
    overwrite_b_11337 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 457, 49), 'overwrite_b', False)
    # Processing the call keyword arguments (line 456)
    kwargs_11338 = {}
    # Getting the type of 'gtsv' (line 456)
    gtsv_11329 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 456, 30), 'gtsv', False)
    # Calling gtsv(args, kwargs) (line 456)
    gtsv_call_result_11339 = invoke(stypy.reporting.localization.Localization(__file__, 456, 30), gtsv_11329, *[dl_11330, d_11331, du_11332, b1_11333, overwrite_ab_11334, overwrite_ab_11335, overwrite_ab_11336, overwrite_b_11337], **kwargs_11338)
    
    # Obtaining the member '__getitem__' of a type (line 456)
    getitem___11340 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 456, 8), gtsv_call_result_11339, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 456)
    subscript_call_result_11341 = invoke(stypy.reporting.localization.Localization(__file__, 456, 8), getitem___11340, int_11328)
    
    # Assigning a type to the variable 'tuple_var_assignment_10069' (line 456)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 456, 8), 'tuple_var_assignment_10069', subscript_call_result_11341)
    
    # Assigning a Name to a Name (line 456):
    # Getting the type of 'tuple_var_assignment_10065' (line 456)
    tuple_var_assignment_10065_11342 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 456, 8), 'tuple_var_assignment_10065')
    # Assigning a type to the variable 'du2' (line 456)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 456, 8), 'du2', tuple_var_assignment_10065_11342)
    
    # Assigning a Name to a Name (line 456):
    # Getting the type of 'tuple_var_assignment_10066' (line 456)
    tuple_var_assignment_10066_11343 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 456, 8), 'tuple_var_assignment_10066')
    # Assigning a type to the variable 'd' (line 456)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 456, 13), 'd', tuple_var_assignment_10066_11343)
    
    # Assigning a Name to a Name (line 456):
    # Getting the type of 'tuple_var_assignment_10067' (line 456)
    tuple_var_assignment_10067_11344 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 456, 8), 'tuple_var_assignment_10067')
    # Assigning a type to the variable 'du' (line 456)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 456, 16), 'du', tuple_var_assignment_10067_11344)
    
    # Assigning a Name to a Name (line 456):
    # Getting the type of 'tuple_var_assignment_10068' (line 456)
    tuple_var_assignment_10068_11345 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 456, 8), 'tuple_var_assignment_10068')
    # Assigning a type to the variable 'x' (line 456)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 456, 20), 'x', tuple_var_assignment_10068_11345)
    
    # Assigning a Name to a Name (line 456):
    # Getting the type of 'tuple_var_assignment_10069' (line 456)
    tuple_var_assignment_10069_11346 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 456, 8), 'tuple_var_assignment_10069')
    # Assigning a type to the variable 'info' (line 456)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 456, 23), 'info', tuple_var_assignment_10069_11346)
    # SSA branch for the else part of an if statement (line 450)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Call to a Tuple (line 459):
    
    # Assigning a Subscript to a Name (line 459):
    
    # Obtaining the type of the subscript
    int_11347 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 459, 8), 'int')
    
    # Call to get_lapack_funcs(...): (line 459)
    # Processing the call arguments (line 459)
    
    # Obtaining an instance of the builtin type 'tuple' (line 459)
    tuple_11349 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 459, 34), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 459)
    # Adding element type (line 459)
    str_11350 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 459, 34), 'str', 'gbsv')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 459, 34), tuple_11349, str_11350)
    
    
    # Obtaining an instance of the builtin type 'tuple' (line 459)
    tuple_11351 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 459, 45), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 459)
    # Adding element type (line 459)
    # Getting the type of 'a1' (line 459)
    a1_11352 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 459, 45), 'a1', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 459, 45), tuple_11351, a1_11352)
    # Adding element type (line 459)
    # Getting the type of 'b1' (line 459)
    b1_11353 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 459, 49), 'b1', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 459, 45), tuple_11351, b1_11353)
    
    # Processing the call keyword arguments (line 459)
    kwargs_11354 = {}
    # Getting the type of 'get_lapack_funcs' (line 459)
    get_lapack_funcs_11348 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 459, 16), 'get_lapack_funcs', False)
    # Calling get_lapack_funcs(args, kwargs) (line 459)
    get_lapack_funcs_call_result_11355 = invoke(stypy.reporting.localization.Localization(__file__, 459, 16), get_lapack_funcs_11348, *[tuple_11349, tuple_11351], **kwargs_11354)
    
    # Obtaining the member '__getitem__' of a type (line 459)
    getitem___11356 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 459, 8), get_lapack_funcs_call_result_11355, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 459)
    subscript_call_result_11357 = invoke(stypy.reporting.localization.Localization(__file__, 459, 8), getitem___11356, int_11347)
    
    # Assigning a type to the variable 'tuple_var_assignment_10070' (line 459)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 459, 8), 'tuple_var_assignment_10070', subscript_call_result_11357)
    
    # Assigning a Name to a Name (line 459):
    # Getting the type of 'tuple_var_assignment_10070' (line 459)
    tuple_var_assignment_10070_11358 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 459, 8), 'tuple_var_assignment_10070')
    # Assigning a type to the variable 'gbsv' (line 459)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 459, 8), 'gbsv', tuple_var_assignment_10070_11358)
    
    # Assigning a Call to a Name (line 460):
    
    # Assigning a Call to a Name (line 460):
    
    # Call to zeros(...): (line 460)
    # Processing the call arguments (line 460)
    
    # Obtaining an instance of the builtin type 'tuple' (line 460)
    tuple_11361 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 460, 23), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 460)
    # Adding element type (line 460)
    int_11362 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 460, 23), 'int')
    # Getting the type of 'l' (line 460)
    l_11363 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 460, 25), 'l', False)
    # Applying the binary operator '*' (line 460)
    result_mul_11364 = python_operator(stypy.reporting.localization.Localization(__file__, 460, 23), '*', int_11362, l_11363)
    
    # Getting the type of 'u' (line 460)
    u_11365 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 460, 27), 'u', False)
    # Applying the binary operator '+' (line 460)
    result_add_11366 = python_operator(stypy.reporting.localization.Localization(__file__, 460, 23), '+', result_mul_11364, u_11365)
    
    int_11367 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 460, 29), 'int')
    # Applying the binary operator '+' (line 460)
    result_add_11368 = python_operator(stypy.reporting.localization.Localization(__file__, 460, 28), '+', result_add_11366, int_11367)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 460, 23), tuple_11361, result_add_11368)
    # Adding element type (line 460)
    
    # Obtaining the type of the subscript
    int_11369 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 460, 41), 'int')
    # Getting the type of 'a1' (line 460)
    a1_11370 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 460, 32), 'a1', False)
    # Obtaining the member 'shape' of a type (line 460)
    shape_11371 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 460, 32), a1_11370, 'shape')
    # Obtaining the member '__getitem__' of a type (line 460)
    getitem___11372 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 460, 32), shape_11371, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 460)
    subscript_call_result_11373 = invoke(stypy.reporting.localization.Localization(__file__, 460, 32), getitem___11372, int_11369)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 460, 23), tuple_11361, subscript_call_result_11373)
    
    # Processing the call keyword arguments (line 460)
    # Getting the type of 'gbsv' (line 460)
    gbsv_11374 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 460, 52), 'gbsv', False)
    # Obtaining the member 'dtype' of a type (line 460)
    dtype_11375 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 460, 52), gbsv_11374, 'dtype')
    keyword_11376 = dtype_11375
    kwargs_11377 = {'dtype': keyword_11376}
    # Getting the type of 'np' (line 460)
    np_11359 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 460, 13), 'np', False)
    # Obtaining the member 'zeros' of a type (line 460)
    zeros_11360 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 460, 13), np_11359, 'zeros')
    # Calling zeros(args, kwargs) (line 460)
    zeros_call_result_11378 = invoke(stypy.reporting.localization.Localization(__file__, 460, 13), zeros_11360, *[tuple_11361], **kwargs_11377)
    
    # Assigning a type to the variable 'a2' (line 460)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 460, 8), 'a2', zeros_call_result_11378)
    
    # Assigning a Name to a Subscript (line 461):
    
    # Assigning a Name to a Subscript (line 461):
    # Getting the type of 'a1' (line 461)
    a1_11379 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 461, 20), 'a1')
    # Getting the type of 'a2' (line 461)
    a2_11380 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 461, 8), 'a2')
    # Getting the type of 'l' (line 461)
    l_11381 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 461, 11), 'l')
    slice_11382 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 461, 8), l_11381, None, None)
    slice_11383 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 461, 8), None, None, None)
    # Storing an element on a container (line 461)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 461, 8), a2_11380, ((slice_11382, slice_11383), a1_11379))
    
    # Assigning a Call to a Tuple (line 462):
    
    # Assigning a Subscript to a Name (line 462):
    
    # Obtaining the type of the subscript
    int_11384 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 462, 8), 'int')
    
    # Call to gbsv(...): (line 462)
    # Processing the call arguments (line 462)
    # Getting the type of 'l' (line 462)
    l_11386 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 462, 32), 'l', False)
    # Getting the type of 'u' (line 462)
    u_11387 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 462, 35), 'u', False)
    # Getting the type of 'a2' (line 462)
    a2_11388 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 462, 38), 'a2', False)
    # Getting the type of 'b1' (line 462)
    b1_11389 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 462, 42), 'b1', False)
    # Processing the call keyword arguments (line 462)
    # Getting the type of 'True' (line 462)
    True_11390 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 462, 59), 'True', False)
    keyword_11391 = True_11390
    # Getting the type of 'overwrite_b' (line 463)
    overwrite_b_11392 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 463, 44), 'overwrite_b', False)
    keyword_11393 = overwrite_b_11392
    kwargs_11394 = {'overwrite_ab': keyword_11391, 'overwrite_b': keyword_11393}
    # Getting the type of 'gbsv' (line 462)
    gbsv_11385 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 462, 27), 'gbsv', False)
    # Calling gbsv(args, kwargs) (line 462)
    gbsv_call_result_11395 = invoke(stypy.reporting.localization.Localization(__file__, 462, 27), gbsv_11385, *[l_11386, u_11387, a2_11388, b1_11389], **kwargs_11394)
    
    # Obtaining the member '__getitem__' of a type (line 462)
    getitem___11396 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 462, 8), gbsv_call_result_11395, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 462)
    subscript_call_result_11397 = invoke(stypy.reporting.localization.Localization(__file__, 462, 8), getitem___11396, int_11384)
    
    # Assigning a type to the variable 'tuple_var_assignment_10071' (line 462)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 462, 8), 'tuple_var_assignment_10071', subscript_call_result_11397)
    
    # Assigning a Subscript to a Name (line 462):
    
    # Obtaining the type of the subscript
    int_11398 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 462, 8), 'int')
    
    # Call to gbsv(...): (line 462)
    # Processing the call arguments (line 462)
    # Getting the type of 'l' (line 462)
    l_11400 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 462, 32), 'l', False)
    # Getting the type of 'u' (line 462)
    u_11401 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 462, 35), 'u', False)
    # Getting the type of 'a2' (line 462)
    a2_11402 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 462, 38), 'a2', False)
    # Getting the type of 'b1' (line 462)
    b1_11403 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 462, 42), 'b1', False)
    # Processing the call keyword arguments (line 462)
    # Getting the type of 'True' (line 462)
    True_11404 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 462, 59), 'True', False)
    keyword_11405 = True_11404
    # Getting the type of 'overwrite_b' (line 463)
    overwrite_b_11406 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 463, 44), 'overwrite_b', False)
    keyword_11407 = overwrite_b_11406
    kwargs_11408 = {'overwrite_ab': keyword_11405, 'overwrite_b': keyword_11407}
    # Getting the type of 'gbsv' (line 462)
    gbsv_11399 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 462, 27), 'gbsv', False)
    # Calling gbsv(args, kwargs) (line 462)
    gbsv_call_result_11409 = invoke(stypy.reporting.localization.Localization(__file__, 462, 27), gbsv_11399, *[l_11400, u_11401, a2_11402, b1_11403], **kwargs_11408)
    
    # Obtaining the member '__getitem__' of a type (line 462)
    getitem___11410 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 462, 8), gbsv_call_result_11409, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 462)
    subscript_call_result_11411 = invoke(stypy.reporting.localization.Localization(__file__, 462, 8), getitem___11410, int_11398)
    
    # Assigning a type to the variable 'tuple_var_assignment_10072' (line 462)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 462, 8), 'tuple_var_assignment_10072', subscript_call_result_11411)
    
    # Assigning a Subscript to a Name (line 462):
    
    # Obtaining the type of the subscript
    int_11412 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 462, 8), 'int')
    
    # Call to gbsv(...): (line 462)
    # Processing the call arguments (line 462)
    # Getting the type of 'l' (line 462)
    l_11414 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 462, 32), 'l', False)
    # Getting the type of 'u' (line 462)
    u_11415 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 462, 35), 'u', False)
    # Getting the type of 'a2' (line 462)
    a2_11416 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 462, 38), 'a2', False)
    # Getting the type of 'b1' (line 462)
    b1_11417 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 462, 42), 'b1', False)
    # Processing the call keyword arguments (line 462)
    # Getting the type of 'True' (line 462)
    True_11418 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 462, 59), 'True', False)
    keyword_11419 = True_11418
    # Getting the type of 'overwrite_b' (line 463)
    overwrite_b_11420 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 463, 44), 'overwrite_b', False)
    keyword_11421 = overwrite_b_11420
    kwargs_11422 = {'overwrite_ab': keyword_11419, 'overwrite_b': keyword_11421}
    # Getting the type of 'gbsv' (line 462)
    gbsv_11413 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 462, 27), 'gbsv', False)
    # Calling gbsv(args, kwargs) (line 462)
    gbsv_call_result_11423 = invoke(stypy.reporting.localization.Localization(__file__, 462, 27), gbsv_11413, *[l_11414, u_11415, a2_11416, b1_11417], **kwargs_11422)
    
    # Obtaining the member '__getitem__' of a type (line 462)
    getitem___11424 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 462, 8), gbsv_call_result_11423, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 462)
    subscript_call_result_11425 = invoke(stypy.reporting.localization.Localization(__file__, 462, 8), getitem___11424, int_11412)
    
    # Assigning a type to the variable 'tuple_var_assignment_10073' (line 462)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 462, 8), 'tuple_var_assignment_10073', subscript_call_result_11425)
    
    # Assigning a Subscript to a Name (line 462):
    
    # Obtaining the type of the subscript
    int_11426 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 462, 8), 'int')
    
    # Call to gbsv(...): (line 462)
    # Processing the call arguments (line 462)
    # Getting the type of 'l' (line 462)
    l_11428 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 462, 32), 'l', False)
    # Getting the type of 'u' (line 462)
    u_11429 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 462, 35), 'u', False)
    # Getting the type of 'a2' (line 462)
    a2_11430 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 462, 38), 'a2', False)
    # Getting the type of 'b1' (line 462)
    b1_11431 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 462, 42), 'b1', False)
    # Processing the call keyword arguments (line 462)
    # Getting the type of 'True' (line 462)
    True_11432 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 462, 59), 'True', False)
    keyword_11433 = True_11432
    # Getting the type of 'overwrite_b' (line 463)
    overwrite_b_11434 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 463, 44), 'overwrite_b', False)
    keyword_11435 = overwrite_b_11434
    kwargs_11436 = {'overwrite_ab': keyword_11433, 'overwrite_b': keyword_11435}
    # Getting the type of 'gbsv' (line 462)
    gbsv_11427 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 462, 27), 'gbsv', False)
    # Calling gbsv(args, kwargs) (line 462)
    gbsv_call_result_11437 = invoke(stypy.reporting.localization.Localization(__file__, 462, 27), gbsv_11427, *[l_11428, u_11429, a2_11430, b1_11431], **kwargs_11436)
    
    # Obtaining the member '__getitem__' of a type (line 462)
    getitem___11438 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 462, 8), gbsv_call_result_11437, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 462)
    subscript_call_result_11439 = invoke(stypy.reporting.localization.Localization(__file__, 462, 8), getitem___11438, int_11426)
    
    # Assigning a type to the variable 'tuple_var_assignment_10074' (line 462)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 462, 8), 'tuple_var_assignment_10074', subscript_call_result_11439)
    
    # Assigning a Name to a Name (line 462):
    # Getting the type of 'tuple_var_assignment_10071' (line 462)
    tuple_var_assignment_10071_11440 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 462, 8), 'tuple_var_assignment_10071')
    # Assigning a type to the variable 'lu' (line 462)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 462, 8), 'lu', tuple_var_assignment_10071_11440)
    
    # Assigning a Name to a Name (line 462):
    # Getting the type of 'tuple_var_assignment_10072' (line 462)
    tuple_var_assignment_10072_11441 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 462, 8), 'tuple_var_assignment_10072')
    # Assigning a type to the variable 'piv' (line 462)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 462, 12), 'piv', tuple_var_assignment_10072_11441)
    
    # Assigning a Name to a Name (line 462):
    # Getting the type of 'tuple_var_assignment_10073' (line 462)
    tuple_var_assignment_10073_11442 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 462, 8), 'tuple_var_assignment_10073')
    # Assigning a type to the variable 'x' (line 462)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 462, 17), 'x', tuple_var_assignment_10073_11442)
    
    # Assigning a Name to a Name (line 462):
    # Getting the type of 'tuple_var_assignment_10074' (line 462)
    tuple_var_assignment_10074_11443 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 462, 8), 'tuple_var_assignment_10074')
    # Assigning a type to the variable 'info' (line 462)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 462, 20), 'info', tuple_var_assignment_10074_11443)
    # SSA join for if statement (line 450)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'info' (line 464)
    info_11444 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 464, 7), 'info')
    int_11445 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 464, 15), 'int')
    # Applying the binary operator '==' (line 464)
    result_eq_11446 = python_operator(stypy.reporting.localization.Localization(__file__, 464, 7), '==', info_11444, int_11445)
    
    # Testing the type of an if condition (line 464)
    if_condition_11447 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 464, 4), result_eq_11446)
    # Assigning a type to the variable 'if_condition_11447' (line 464)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 464, 4), 'if_condition_11447', if_condition_11447)
    # SSA begins for if statement (line 464)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # Getting the type of 'x' (line 465)
    x_11448 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 465, 15), 'x')
    # Assigning a type to the variable 'stypy_return_type' (line 465)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 465, 8), 'stypy_return_type', x_11448)
    # SSA join for if statement (line 464)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'info' (line 466)
    info_11449 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 466, 7), 'info')
    int_11450 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 466, 14), 'int')
    # Applying the binary operator '>' (line 466)
    result_gt_11451 = python_operator(stypy.reporting.localization.Localization(__file__, 466, 7), '>', info_11449, int_11450)
    
    # Testing the type of an if condition (line 466)
    if_condition_11452 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 466, 4), result_gt_11451)
    # Assigning a type to the variable 'if_condition_11452' (line 466)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 466, 4), 'if_condition_11452', if_condition_11452)
    # SSA begins for if statement (line 466)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to LinAlgError(...): (line 467)
    # Processing the call arguments (line 467)
    str_11454 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 467, 26), 'str', 'singular matrix')
    # Processing the call keyword arguments (line 467)
    kwargs_11455 = {}
    # Getting the type of 'LinAlgError' (line 467)
    LinAlgError_11453 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 467, 14), 'LinAlgError', False)
    # Calling LinAlgError(args, kwargs) (line 467)
    LinAlgError_call_result_11456 = invoke(stypy.reporting.localization.Localization(__file__, 467, 14), LinAlgError_11453, *[str_11454], **kwargs_11455)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 467, 8), LinAlgError_call_result_11456, 'raise parameter', BaseException)
    # SSA join for if statement (line 466)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to ValueError(...): (line 468)
    # Processing the call arguments (line 468)
    str_11458 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 468, 21), 'str', 'illegal value in %d-th argument of internal gbsv/gtsv')
    
    # Getting the type of 'info' (line 469)
    info_11459 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 469, 36), 'info', False)
    # Applying the 'usub' unary operator (line 469)
    result___neg___11460 = python_operator(stypy.reporting.localization.Localization(__file__, 469, 35), 'usub', info_11459)
    
    # Applying the binary operator '%' (line 468)
    result_mod_11461 = python_operator(stypy.reporting.localization.Localization(__file__, 468, 21), '%', str_11458, result___neg___11460)
    
    # Processing the call keyword arguments (line 468)
    kwargs_11462 = {}
    # Getting the type of 'ValueError' (line 468)
    ValueError_11457 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 468, 10), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 468)
    ValueError_call_result_11463 = invoke(stypy.reporting.localization.Localization(__file__, 468, 10), ValueError_11457, *[result_mod_11461], **kwargs_11462)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 468, 4), ValueError_call_result_11463, 'raise parameter', BaseException)
    
    # ################# End of 'solve_banded(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'solve_banded' in the type store
    # Getting the type of 'stypy_return_type' (line 359)
    stypy_return_type_11464 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 359, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_11464)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'solve_banded'
    return stypy_return_type_11464

# Assigning a type to the variable 'solve_banded' (line 359)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 359, 0), 'solve_banded', solve_banded)

@norecursion
def solveh_banded(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'False' (line 472)
    False_11465 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 472, 38), 'False')
    # Getting the type of 'False' (line 472)
    False_11466 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 472, 57), 'False')
    # Getting the type of 'False' (line 472)
    False_11467 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 472, 70), 'False')
    # Getting the type of 'True' (line 473)
    True_11468 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 473, 31), 'True')
    defaults = [False_11465, False_11466, False_11467, True_11468]
    # Create a new context for function 'solveh_banded'
    module_type_store = module_type_store.open_function_context('solveh_banded', 472, 0, False)
    
    # Passed parameters checking function
    solveh_banded.stypy_localization = localization
    solveh_banded.stypy_type_of_self = None
    solveh_banded.stypy_type_store = module_type_store
    solveh_banded.stypy_function_name = 'solveh_banded'
    solveh_banded.stypy_param_names_list = ['ab', 'b', 'overwrite_ab', 'overwrite_b', 'lower', 'check_finite']
    solveh_banded.stypy_varargs_param_name = None
    solveh_banded.stypy_kwargs_param_name = None
    solveh_banded.stypy_call_defaults = defaults
    solveh_banded.stypy_call_varargs = varargs
    solveh_banded.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'solveh_banded', ['ab', 'b', 'overwrite_ab', 'overwrite_b', 'lower', 'check_finite'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'solveh_banded', localization, ['ab', 'b', 'overwrite_ab', 'overwrite_b', 'lower', 'check_finite'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'solveh_banded(...)' code ##################

    str_11469 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 563, (-1)), 'str', '\n    Solve equation a x = b. a is Hermitian positive-definite banded matrix.\n\n    The matrix a is stored in `ab` either in lower diagonal or upper\n    diagonal ordered form:\n\n        ab[u + i - j, j] == a[i,j]        (if upper form; i <= j)\n        ab[    i - j, j] == a[i,j]        (if lower form; i >= j)\n\n    Example of `ab` (shape of a is (6, 6), `u` =2)::\n\n        upper form:\n        *   *   a02 a13 a24 a35\n        *   a01 a12 a23 a34 a45\n        a00 a11 a22 a33 a44 a55\n\n        lower form:\n        a00 a11 a22 a33 a44 a55\n        a10 a21 a32 a43 a54 *\n        a20 a31 a42 a53 *   *\n\n    Cells marked with * are not used.\n\n    Parameters\n    ----------\n    ab : (`u` + 1, M) array_like\n        Banded matrix\n    b : (M,) or (M, K) array_like\n        Right-hand side\n    overwrite_ab : bool, optional\n        Discard data in `ab` (may enhance performance)\n    overwrite_b : bool, optional\n        Discard data in `b` (may enhance performance)\n    lower : bool, optional\n        Is the matrix in the lower form. (Default is upper form)\n    check_finite : bool, optional\n        Whether to check that the input matrices contain only finite numbers.\n        Disabling may give a performance gain, but may result in problems\n        (crashes, non-termination) if the inputs do contain infinities or NaNs.\n\n    Returns\n    -------\n    x : (M,) or (M, K) ndarray\n        The solution to the system a x = b.  Shape of return matches shape\n        of `b`.\n\n    Examples\n    --------\n    Solve the banded system A x = b, where::\n\n            [ 4  2 -1  0  0  0]       [1]\n            [ 2  5  2 -1  0  0]       [2]\n        A = [-1  2  6  2 -1  0]   b = [2]\n            [ 0 -1  2  7  2 -1]       [3]\n            [ 0  0 -1  2  8  2]       [3]\n            [ 0  0  0 -1  2  9]       [3]\n\n    >>> from scipy.linalg import solveh_banded\n\n    `ab` contains the main diagonal and the nonzero diagonals below the\n    main diagonal.  That is, we use the lower form:\n\n    >>> ab = np.array([[ 4,  5,  6,  7, 8, 9],\n    ...                [ 2,  2,  2,  2, 2, 0],\n    ...                [-1, -1, -1, -1, 0, 0]])\n    >>> b = np.array([1, 2, 2, 3, 3, 3])\n    >>> x = solveh_banded(ab, b, lower=True)\n    >>> x\n    array([ 0.03431373,  0.45938375,  0.05602241,  0.47759104,  0.17577031,\n            0.34733894])\n\n\n    Solve the Hermitian banded system H x = b, where::\n\n            [ 8   2-1j   0     0  ]        [ 1  ]\n        H = [2+1j  5     1j    0  ]    b = [1+1j]\n            [ 0   -1j    9   -2-1j]        [1-2j]\n            [ 0    0   -2+1j   6  ]        [ 0  ]\n\n    In this example, we put the upper diagonals in the array `hb`:\n\n    >>> hb = np.array([[0, 2-1j, 1j, -2-1j],\n    ...                [8,  5,    9,   6  ]])\n    >>> b = np.array([1, 1+1j, 1-2j, 0])\n    >>> x = solveh_banded(hb, b)\n    >>> x\n    array([ 0.07318536-0.02939412j,  0.11877624+0.17696461j,\n            0.10077984-0.23035393j, -0.00479904-0.09358128j])\n\n    ')
    
    # Assigning a Call to a Name (line 564):
    
    # Assigning a Call to a Name (line 564):
    
    # Call to _asarray_validated(...): (line 564)
    # Processing the call arguments (line 564)
    # Getting the type of 'ab' (line 564)
    ab_11471 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 564, 28), 'ab', False)
    # Processing the call keyword arguments (line 564)
    # Getting the type of 'check_finite' (line 564)
    check_finite_11472 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 564, 45), 'check_finite', False)
    keyword_11473 = check_finite_11472
    kwargs_11474 = {'check_finite': keyword_11473}
    # Getting the type of '_asarray_validated' (line 564)
    _asarray_validated_11470 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 564, 9), '_asarray_validated', False)
    # Calling _asarray_validated(args, kwargs) (line 564)
    _asarray_validated_call_result_11475 = invoke(stypy.reporting.localization.Localization(__file__, 564, 9), _asarray_validated_11470, *[ab_11471], **kwargs_11474)
    
    # Assigning a type to the variable 'a1' (line 564)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 564, 4), 'a1', _asarray_validated_call_result_11475)
    
    # Assigning a Call to a Name (line 565):
    
    # Assigning a Call to a Name (line 565):
    
    # Call to _asarray_validated(...): (line 565)
    # Processing the call arguments (line 565)
    # Getting the type of 'b' (line 565)
    b_11477 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 565, 28), 'b', False)
    # Processing the call keyword arguments (line 565)
    # Getting the type of 'check_finite' (line 565)
    check_finite_11478 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 565, 44), 'check_finite', False)
    keyword_11479 = check_finite_11478
    kwargs_11480 = {'check_finite': keyword_11479}
    # Getting the type of '_asarray_validated' (line 565)
    _asarray_validated_11476 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 565, 9), '_asarray_validated', False)
    # Calling _asarray_validated(args, kwargs) (line 565)
    _asarray_validated_call_result_11481 = invoke(stypy.reporting.localization.Localization(__file__, 565, 9), _asarray_validated_11476, *[b_11477], **kwargs_11480)
    
    # Assigning a type to the variable 'b1' (line 565)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 565, 4), 'b1', _asarray_validated_call_result_11481)
    
    
    
    # Obtaining the type of the subscript
    int_11482 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 567, 16), 'int')
    # Getting the type of 'a1' (line 567)
    a1_11483 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 567, 7), 'a1')
    # Obtaining the member 'shape' of a type (line 567)
    shape_11484 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 567, 7), a1_11483, 'shape')
    # Obtaining the member '__getitem__' of a type (line 567)
    getitem___11485 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 567, 7), shape_11484, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 567)
    subscript_call_result_11486 = invoke(stypy.reporting.localization.Localization(__file__, 567, 7), getitem___11485, int_11482)
    
    
    # Obtaining the type of the subscript
    int_11487 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 567, 32), 'int')
    # Getting the type of 'b1' (line 567)
    b1_11488 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 567, 23), 'b1')
    # Obtaining the member 'shape' of a type (line 567)
    shape_11489 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 567, 23), b1_11488, 'shape')
    # Obtaining the member '__getitem__' of a type (line 567)
    getitem___11490 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 567, 23), shape_11489, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 567)
    subscript_call_result_11491 = invoke(stypy.reporting.localization.Localization(__file__, 567, 23), getitem___11490, int_11487)
    
    # Applying the binary operator '!=' (line 567)
    result_ne_11492 = python_operator(stypy.reporting.localization.Localization(__file__, 567, 7), '!=', subscript_call_result_11486, subscript_call_result_11491)
    
    # Testing the type of an if condition (line 567)
    if_condition_11493 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 567, 4), result_ne_11492)
    # Assigning a type to the variable 'if_condition_11493' (line 567)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 567, 4), 'if_condition_11493', if_condition_11493)
    # SSA begins for if statement (line 567)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 568)
    # Processing the call arguments (line 568)
    str_11495 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 568, 25), 'str', 'shapes of ab and b are not compatible.')
    # Processing the call keyword arguments (line 568)
    kwargs_11496 = {}
    # Getting the type of 'ValueError' (line 568)
    ValueError_11494 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 568, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 568)
    ValueError_call_result_11497 = invoke(stypy.reporting.localization.Localization(__file__, 568, 14), ValueError_11494, *[str_11495], **kwargs_11496)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 568, 8), ValueError_call_result_11497, 'raise parameter', BaseException)
    # SSA join for if statement (line 567)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a BoolOp to a Name (line 570):
    
    # Assigning a BoolOp to a Name (line 570):
    
    # Evaluating a boolean operation
    # Getting the type of 'overwrite_b' (line 570)
    overwrite_b_11498 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 570, 18), 'overwrite_b')
    
    # Call to _datacopied(...): (line 570)
    # Processing the call arguments (line 570)
    # Getting the type of 'b1' (line 570)
    b1_11500 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 570, 45), 'b1', False)
    # Getting the type of 'b' (line 570)
    b_11501 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 570, 49), 'b', False)
    # Processing the call keyword arguments (line 570)
    kwargs_11502 = {}
    # Getting the type of '_datacopied' (line 570)
    _datacopied_11499 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 570, 33), '_datacopied', False)
    # Calling _datacopied(args, kwargs) (line 570)
    _datacopied_call_result_11503 = invoke(stypy.reporting.localization.Localization(__file__, 570, 33), _datacopied_11499, *[b1_11500, b_11501], **kwargs_11502)
    
    # Applying the binary operator 'or' (line 570)
    result_or_keyword_11504 = python_operator(stypy.reporting.localization.Localization(__file__, 570, 18), 'or', overwrite_b_11498, _datacopied_call_result_11503)
    
    # Assigning a type to the variable 'overwrite_b' (line 570)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 570, 4), 'overwrite_b', result_or_keyword_11504)
    
    # Assigning a BoolOp to a Name (line 571):
    
    # Assigning a BoolOp to a Name (line 571):
    
    # Evaluating a boolean operation
    # Getting the type of 'overwrite_ab' (line 571)
    overwrite_ab_11505 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 571, 19), 'overwrite_ab')
    
    # Call to _datacopied(...): (line 571)
    # Processing the call arguments (line 571)
    # Getting the type of 'a1' (line 571)
    a1_11507 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 571, 47), 'a1', False)
    # Getting the type of 'ab' (line 571)
    ab_11508 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 571, 51), 'ab', False)
    # Processing the call keyword arguments (line 571)
    kwargs_11509 = {}
    # Getting the type of '_datacopied' (line 571)
    _datacopied_11506 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 571, 35), '_datacopied', False)
    # Calling _datacopied(args, kwargs) (line 571)
    _datacopied_call_result_11510 = invoke(stypy.reporting.localization.Localization(__file__, 571, 35), _datacopied_11506, *[a1_11507, ab_11508], **kwargs_11509)
    
    # Applying the binary operator 'or' (line 571)
    result_or_keyword_11511 = python_operator(stypy.reporting.localization.Localization(__file__, 571, 19), 'or', overwrite_ab_11505, _datacopied_call_result_11510)
    
    # Assigning a type to the variable 'overwrite_ab' (line 571)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 571, 4), 'overwrite_ab', result_or_keyword_11511)
    
    
    
    # Obtaining the type of the subscript
    int_11512 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 573, 16), 'int')
    # Getting the type of 'a1' (line 573)
    a1_11513 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 573, 7), 'a1')
    # Obtaining the member 'shape' of a type (line 573)
    shape_11514 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 573, 7), a1_11513, 'shape')
    # Obtaining the member '__getitem__' of a type (line 573)
    getitem___11515 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 573, 7), shape_11514, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 573)
    subscript_call_result_11516 = invoke(stypy.reporting.localization.Localization(__file__, 573, 7), getitem___11515, int_11512)
    
    int_11517 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 573, 22), 'int')
    # Applying the binary operator '==' (line 573)
    result_eq_11518 = python_operator(stypy.reporting.localization.Localization(__file__, 573, 7), '==', subscript_call_result_11516, int_11517)
    
    # Testing the type of an if condition (line 573)
    if_condition_11519 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 573, 4), result_eq_11518)
    # Assigning a type to the variable 'if_condition_11519' (line 573)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 573, 4), 'if_condition_11519', if_condition_11519)
    # SSA begins for if statement (line 573)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Tuple (line 574):
    
    # Assigning a Subscript to a Name (line 574):
    
    # Obtaining the type of the subscript
    int_11520 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 574, 8), 'int')
    
    # Call to get_lapack_funcs(...): (line 574)
    # Processing the call arguments (line 574)
    
    # Obtaining an instance of the builtin type 'tuple' (line 574)
    tuple_11522 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 574, 34), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 574)
    # Adding element type (line 574)
    str_11523 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 574, 34), 'str', 'ptsv')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 574, 34), tuple_11522, str_11523)
    
    
    # Obtaining an instance of the builtin type 'tuple' (line 574)
    tuple_11524 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 574, 45), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 574)
    # Adding element type (line 574)
    # Getting the type of 'a1' (line 574)
    a1_11525 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 574, 45), 'a1', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 574, 45), tuple_11524, a1_11525)
    # Adding element type (line 574)
    # Getting the type of 'b1' (line 574)
    b1_11526 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 574, 49), 'b1', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 574, 45), tuple_11524, b1_11526)
    
    # Processing the call keyword arguments (line 574)
    kwargs_11527 = {}
    # Getting the type of 'get_lapack_funcs' (line 574)
    get_lapack_funcs_11521 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 574, 16), 'get_lapack_funcs', False)
    # Calling get_lapack_funcs(args, kwargs) (line 574)
    get_lapack_funcs_call_result_11528 = invoke(stypy.reporting.localization.Localization(__file__, 574, 16), get_lapack_funcs_11521, *[tuple_11522, tuple_11524], **kwargs_11527)
    
    # Obtaining the member '__getitem__' of a type (line 574)
    getitem___11529 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 574, 8), get_lapack_funcs_call_result_11528, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 574)
    subscript_call_result_11530 = invoke(stypy.reporting.localization.Localization(__file__, 574, 8), getitem___11529, int_11520)
    
    # Assigning a type to the variable 'tuple_var_assignment_10075' (line 574)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 574, 8), 'tuple_var_assignment_10075', subscript_call_result_11530)
    
    # Assigning a Name to a Name (line 574):
    # Getting the type of 'tuple_var_assignment_10075' (line 574)
    tuple_var_assignment_10075_11531 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 574, 8), 'tuple_var_assignment_10075')
    # Assigning a type to the variable 'ptsv' (line 574)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 574, 8), 'ptsv', tuple_var_assignment_10075_11531)
    
    # Getting the type of 'lower' (line 575)
    lower_11532 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 575, 11), 'lower')
    # Testing the type of an if condition (line 575)
    if_condition_11533 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 575, 8), lower_11532)
    # Assigning a type to the variable 'if_condition_11533' (line 575)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 575, 8), 'if_condition_11533', if_condition_11533)
    # SSA begins for if statement (line 575)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Attribute to a Name (line 576):
    
    # Assigning a Attribute to a Name (line 576):
    
    # Obtaining the type of the subscript
    int_11534 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 576, 19), 'int')
    slice_11535 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 576, 16), None, None, None)
    # Getting the type of 'a1' (line 576)
    a1_11536 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 576, 16), 'a1')
    # Obtaining the member '__getitem__' of a type (line 576)
    getitem___11537 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 576, 16), a1_11536, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 576)
    subscript_call_result_11538 = invoke(stypy.reporting.localization.Localization(__file__, 576, 16), getitem___11537, (int_11534, slice_11535))
    
    # Obtaining the member 'real' of a type (line 576)
    real_11539 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 576, 16), subscript_call_result_11538, 'real')
    # Assigning a type to the variable 'd' (line 576)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 576, 12), 'd', real_11539)
    
    # Assigning a Subscript to a Name (line 577):
    
    # Assigning a Subscript to a Name (line 577):
    
    # Obtaining the type of the subscript
    int_11540 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 577, 19), 'int')
    int_11541 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 577, 23), 'int')
    slice_11542 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 577, 16), None, int_11541, None)
    # Getting the type of 'a1' (line 577)
    a1_11543 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 577, 16), 'a1')
    # Obtaining the member '__getitem__' of a type (line 577)
    getitem___11544 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 577, 16), a1_11543, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 577)
    subscript_call_result_11545 = invoke(stypy.reporting.localization.Localization(__file__, 577, 16), getitem___11544, (int_11540, slice_11542))
    
    # Assigning a type to the variable 'e' (line 577)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 577, 12), 'e', subscript_call_result_11545)
    # SSA branch for the else part of an if statement (line 575)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Attribute to a Name (line 579):
    
    # Assigning a Attribute to a Name (line 579):
    
    # Obtaining the type of the subscript
    int_11546 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 579, 19), 'int')
    slice_11547 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 579, 16), None, None, None)
    # Getting the type of 'a1' (line 579)
    a1_11548 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 579, 16), 'a1')
    # Obtaining the member '__getitem__' of a type (line 579)
    getitem___11549 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 579, 16), a1_11548, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 579)
    subscript_call_result_11550 = invoke(stypy.reporting.localization.Localization(__file__, 579, 16), getitem___11549, (int_11546, slice_11547))
    
    # Obtaining the member 'real' of a type (line 579)
    real_11551 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 579, 16), subscript_call_result_11550, 'real')
    # Assigning a type to the variable 'd' (line 579)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 579, 12), 'd', real_11551)
    
    # Assigning a Call to a Name (line 580):
    
    # Assigning a Call to a Name (line 580):
    
    # Call to conj(...): (line 580)
    # Processing the call keyword arguments (line 580)
    kwargs_11559 = {}
    
    # Obtaining the type of the subscript
    int_11552 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 580, 19), 'int')
    int_11553 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 580, 22), 'int')
    slice_11554 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 580, 16), int_11553, None, None)
    # Getting the type of 'a1' (line 580)
    a1_11555 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 580, 16), 'a1', False)
    # Obtaining the member '__getitem__' of a type (line 580)
    getitem___11556 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 580, 16), a1_11555, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 580)
    subscript_call_result_11557 = invoke(stypy.reporting.localization.Localization(__file__, 580, 16), getitem___11556, (int_11552, slice_11554))
    
    # Obtaining the member 'conj' of a type (line 580)
    conj_11558 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 580, 16), subscript_call_result_11557, 'conj')
    # Calling conj(args, kwargs) (line 580)
    conj_call_result_11560 = invoke(stypy.reporting.localization.Localization(__file__, 580, 16), conj_11558, *[], **kwargs_11559)
    
    # Assigning a type to the variable 'e' (line 580)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 580, 12), 'e', conj_call_result_11560)
    # SSA join for if statement (line 575)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Tuple (line 581):
    
    # Assigning a Subscript to a Name (line 581):
    
    # Obtaining the type of the subscript
    int_11561 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 581, 8), 'int')
    
    # Call to ptsv(...): (line 581)
    # Processing the call arguments (line 581)
    # Getting the type of 'd' (line 581)
    d_11563 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 581, 30), 'd', False)
    # Getting the type of 'e' (line 581)
    e_11564 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 581, 33), 'e', False)
    # Getting the type of 'b1' (line 581)
    b1_11565 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 581, 36), 'b1', False)
    # Getting the type of 'overwrite_ab' (line 581)
    overwrite_ab_11566 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 581, 40), 'overwrite_ab', False)
    # Getting the type of 'overwrite_ab' (line 581)
    overwrite_ab_11567 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 581, 54), 'overwrite_ab', False)
    # Getting the type of 'overwrite_b' (line 582)
    overwrite_b_11568 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 582, 30), 'overwrite_b', False)
    # Processing the call keyword arguments (line 581)
    kwargs_11569 = {}
    # Getting the type of 'ptsv' (line 581)
    ptsv_11562 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 581, 25), 'ptsv', False)
    # Calling ptsv(args, kwargs) (line 581)
    ptsv_call_result_11570 = invoke(stypy.reporting.localization.Localization(__file__, 581, 25), ptsv_11562, *[d_11563, e_11564, b1_11565, overwrite_ab_11566, overwrite_ab_11567, overwrite_b_11568], **kwargs_11569)
    
    # Obtaining the member '__getitem__' of a type (line 581)
    getitem___11571 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 581, 8), ptsv_call_result_11570, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 581)
    subscript_call_result_11572 = invoke(stypy.reporting.localization.Localization(__file__, 581, 8), getitem___11571, int_11561)
    
    # Assigning a type to the variable 'tuple_var_assignment_10076' (line 581)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 581, 8), 'tuple_var_assignment_10076', subscript_call_result_11572)
    
    # Assigning a Subscript to a Name (line 581):
    
    # Obtaining the type of the subscript
    int_11573 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 581, 8), 'int')
    
    # Call to ptsv(...): (line 581)
    # Processing the call arguments (line 581)
    # Getting the type of 'd' (line 581)
    d_11575 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 581, 30), 'd', False)
    # Getting the type of 'e' (line 581)
    e_11576 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 581, 33), 'e', False)
    # Getting the type of 'b1' (line 581)
    b1_11577 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 581, 36), 'b1', False)
    # Getting the type of 'overwrite_ab' (line 581)
    overwrite_ab_11578 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 581, 40), 'overwrite_ab', False)
    # Getting the type of 'overwrite_ab' (line 581)
    overwrite_ab_11579 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 581, 54), 'overwrite_ab', False)
    # Getting the type of 'overwrite_b' (line 582)
    overwrite_b_11580 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 582, 30), 'overwrite_b', False)
    # Processing the call keyword arguments (line 581)
    kwargs_11581 = {}
    # Getting the type of 'ptsv' (line 581)
    ptsv_11574 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 581, 25), 'ptsv', False)
    # Calling ptsv(args, kwargs) (line 581)
    ptsv_call_result_11582 = invoke(stypy.reporting.localization.Localization(__file__, 581, 25), ptsv_11574, *[d_11575, e_11576, b1_11577, overwrite_ab_11578, overwrite_ab_11579, overwrite_b_11580], **kwargs_11581)
    
    # Obtaining the member '__getitem__' of a type (line 581)
    getitem___11583 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 581, 8), ptsv_call_result_11582, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 581)
    subscript_call_result_11584 = invoke(stypy.reporting.localization.Localization(__file__, 581, 8), getitem___11583, int_11573)
    
    # Assigning a type to the variable 'tuple_var_assignment_10077' (line 581)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 581, 8), 'tuple_var_assignment_10077', subscript_call_result_11584)
    
    # Assigning a Subscript to a Name (line 581):
    
    # Obtaining the type of the subscript
    int_11585 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 581, 8), 'int')
    
    # Call to ptsv(...): (line 581)
    # Processing the call arguments (line 581)
    # Getting the type of 'd' (line 581)
    d_11587 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 581, 30), 'd', False)
    # Getting the type of 'e' (line 581)
    e_11588 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 581, 33), 'e', False)
    # Getting the type of 'b1' (line 581)
    b1_11589 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 581, 36), 'b1', False)
    # Getting the type of 'overwrite_ab' (line 581)
    overwrite_ab_11590 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 581, 40), 'overwrite_ab', False)
    # Getting the type of 'overwrite_ab' (line 581)
    overwrite_ab_11591 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 581, 54), 'overwrite_ab', False)
    # Getting the type of 'overwrite_b' (line 582)
    overwrite_b_11592 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 582, 30), 'overwrite_b', False)
    # Processing the call keyword arguments (line 581)
    kwargs_11593 = {}
    # Getting the type of 'ptsv' (line 581)
    ptsv_11586 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 581, 25), 'ptsv', False)
    # Calling ptsv(args, kwargs) (line 581)
    ptsv_call_result_11594 = invoke(stypy.reporting.localization.Localization(__file__, 581, 25), ptsv_11586, *[d_11587, e_11588, b1_11589, overwrite_ab_11590, overwrite_ab_11591, overwrite_b_11592], **kwargs_11593)
    
    # Obtaining the member '__getitem__' of a type (line 581)
    getitem___11595 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 581, 8), ptsv_call_result_11594, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 581)
    subscript_call_result_11596 = invoke(stypy.reporting.localization.Localization(__file__, 581, 8), getitem___11595, int_11585)
    
    # Assigning a type to the variable 'tuple_var_assignment_10078' (line 581)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 581, 8), 'tuple_var_assignment_10078', subscript_call_result_11596)
    
    # Assigning a Subscript to a Name (line 581):
    
    # Obtaining the type of the subscript
    int_11597 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 581, 8), 'int')
    
    # Call to ptsv(...): (line 581)
    # Processing the call arguments (line 581)
    # Getting the type of 'd' (line 581)
    d_11599 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 581, 30), 'd', False)
    # Getting the type of 'e' (line 581)
    e_11600 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 581, 33), 'e', False)
    # Getting the type of 'b1' (line 581)
    b1_11601 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 581, 36), 'b1', False)
    # Getting the type of 'overwrite_ab' (line 581)
    overwrite_ab_11602 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 581, 40), 'overwrite_ab', False)
    # Getting the type of 'overwrite_ab' (line 581)
    overwrite_ab_11603 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 581, 54), 'overwrite_ab', False)
    # Getting the type of 'overwrite_b' (line 582)
    overwrite_b_11604 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 582, 30), 'overwrite_b', False)
    # Processing the call keyword arguments (line 581)
    kwargs_11605 = {}
    # Getting the type of 'ptsv' (line 581)
    ptsv_11598 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 581, 25), 'ptsv', False)
    # Calling ptsv(args, kwargs) (line 581)
    ptsv_call_result_11606 = invoke(stypy.reporting.localization.Localization(__file__, 581, 25), ptsv_11598, *[d_11599, e_11600, b1_11601, overwrite_ab_11602, overwrite_ab_11603, overwrite_b_11604], **kwargs_11605)
    
    # Obtaining the member '__getitem__' of a type (line 581)
    getitem___11607 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 581, 8), ptsv_call_result_11606, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 581)
    subscript_call_result_11608 = invoke(stypy.reporting.localization.Localization(__file__, 581, 8), getitem___11607, int_11597)
    
    # Assigning a type to the variable 'tuple_var_assignment_10079' (line 581)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 581, 8), 'tuple_var_assignment_10079', subscript_call_result_11608)
    
    # Assigning a Name to a Name (line 581):
    # Getting the type of 'tuple_var_assignment_10076' (line 581)
    tuple_var_assignment_10076_11609 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 581, 8), 'tuple_var_assignment_10076')
    # Assigning a type to the variable 'd' (line 581)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 581, 8), 'd', tuple_var_assignment_10076_11609)
    
    # Assigning a Name to a Name (line 581):
    # Getting the type of 'tuple_var_assignment_10077' (line 581)
    tuple_var_assignment_10077_11610 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 581, 8), 'tuple_var_assignment_10077')
    # Assigning a type to the variable 'du' (line 581)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 581, 11), 'du', tuple_var_assignment_10077_11610)
    
    # Assigning a Name to a Name (line 581):
    # Getting the type of 'tuple_var_assignment_10078' (line 581)
    tuple_var_assignment_10078_11611 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 581, 8), 'tuple_var_assignment_10078')
    # Assigning a type to the variable 'x' (line 581)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 581, 15), 'x', tuple_var_assignment_10078_11611)
    
    # Assigning a Name to a Name (line 581):
    # Getting the type of 'tuple_var_assignment_10079' (line 581)
    tuple_var_assignment_10079_11612 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 581, 8), 'tuple_var_assignment_10079')
    # Assigning a type to the variable 'info' (line 581)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 581, 18), 'info', tuple_var_assignment_10079_11612)
    # SSA branch for the else part of an if statement (line 573)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Call to a Tuple (line 584):
    
    # Assigning a Subscript to a Name (line 584):
    
    # Obtaining the type of the subscript
    int_11613 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 584, 8), 'int')
    
    # Call to get_lapack_funcs(...): (line 584)
    # Processing the call arguments (line 584)
    
    # Obtaining an instance of the builtin type 'tuple' (line 584)
    tuple_11615 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 584, 34), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 584)
    # Adding element type (line 584)
    str_11616 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 584, 34), 'str', 'pbsv')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 584, 34), tuple_11615, str_11616)
    
    
    # Obtaining an instance of the builtin type 'tuple' (line 584)
    tuple_11617 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 584, 45), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 584)
    # Adding element type (line 584)
    # Getting the type of 'a1' (line 584)
    a1_11618 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 584, 45), 'a1', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 584, 45), tuple_11617, a1_11618)
    # Adding element type (line 584)
    # Getting the type of 'b1' (line 584)
    b1_11619 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 584, 49), 'b1', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 584, 45), tuple_11617, b1_11619)
    
    # Processing the call keyword arguments (line 584)
    kwargs_11620 = {}
    # Getting the type of 'get_lapack_funcs' (line 584)
    get_lapack_funcs_11614 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 584, 16), 'get_lapack_funcs', False)
    # Calling get_lapack_funcs(args, kwargs) (line 584)
    get_lapack_funcs_call_result_11621 = invoke(stypy.reporting.localization.Localization(__file__, 584, 16), get_lapack_funcs_11614, *[tuple_11615, tuple_11617], **kwargs_11620)
    
    # Obtaining the member '__getitem__' of a type (line 584)
    getitem___11622 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 584, 8), get_lapack_funcs_call_result_11621, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 584)
    subscript_call_result_11623 = invoke(stypy.reporting.localization.Localization(__file__, 584, 8), getitem___11622, int_11613)
    
    # Assigning a type to the variable 'tuple_var_assignment_10080' (line 584)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 584, 8), 'tuple_var_assignment_10080', subscript_call_result_11623)
    
    # Assigning a Name to a Name (line 584):
    # Getting the type of 'tuple_var_assignment_10080' (line 584)
    tuple_var_assignment_10080_11624 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 584, 8), 'tuple_var_assignment_10080')
    # Assigning a type to the variable 'pbsv' (line 584)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 584, 8), 'pbsv', tuple_var_assignment_10080_11624)
    
    # Assigning a Call to a Tuple (line 585):
    
    # Assigning a Subscript to a Name (line 585):
    
    # Obtaining the type of the subscript
    int_11625 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 585, 8), 'int')
    
    # Call to pbsv(...): (line 585)
    # Processing the call arguments (line 585)
    # Getting the type of 'a1' (line 585)
    a1_11627 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 585, 26), 'a1', False)
    # Getting the type of 'b1' (line 585)
    b1_11628 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 585, 30), 'b1', False)
    # Processing the call keyword arguments (line 585)
    # Getting the type of 'lower' (line 585)
    lower_11629 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 585, 40), 'lower', False)
    keyword_11630 = lower_11629
    # Getting the type of 'overwrite_ab' (line 585)
    overwrite_ab_11631 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 585, 60), 'overwrite_ab', False)
    keyword_11632 = overwrite_ab_11631
    # Getting the type of 'overwrite_b' (line 586)
    overwrite_b_11633 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 586, 38), 'overwrite_b', False)
    keyword_11634 = overwrite_b_11633
    kwargs_11635 = {'lower': keyword_11630, 'overwrite_ab': keyword_11632, 'overwrite_b': keyword_11634}
    # Getting the type of 'pbsv' (line 585)
    pbsv_11626 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 585, 21), 'pbsv', False)
    # Calling pbsv(args, kwargs) (line 585)
    pbsv_call_result_11636 = invoke(stypy.reporting.localization.Localization(__file__, 585, 21), pbsv_11626, *[a1_11627, b1_11628], **kwargs_11635)
    
    # Obtaining the member '__getitem__' of a type (line 585)
    getitem___11637 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 585, 8), pbsv_call_result_11636, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 585)
    subscript_call_result_11638 = invoke(stypy.reporting.localization.Localization(__file__, 585, 8), getitem___11637, int_11625)
    
    # Assigning a type to the variable 'tuple_var_assignment_10081' (line 585)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 585, 8), 'tuple_var_assignment_10081', subscript_call_result_11638)
    
    # Assigning a Subscript to a Name (line 585):
    
    # Obtaining the type of the subscript
    int_11639 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 585, 8), 'int')
    
    # Call to pbsv(...): (line 585)
    # Processing the call arguments (line 585)
    # Getting the type of 'a1' (line 585)
    a1_11641 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 585, 26), 'a1', False)
    # Getting the type of 'b1' (line 585)
    b1_11642 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 585, 30), 'b1', False)
    # Processing the call keyword arguments (line 585)
    # Getting the type of 'lower' (line 585)
    lower_11643 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 585, 40), 'lower', False)
    keyword_11644 = lower_11643
    # Getting the type of 'overwrite_ab' (line 585)
    overwrite_ab_11645 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 585, 60), 'overwrite_ab', False)
    keyword_11646 = overwrite_ab_11645
    # Getting the type of 'overwrite_b' (line 586)
    overwrite_b_11647 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 586, 38), 'overwrite_b', False)
    keyword_11648 = overwrite_b_11647
    kwargs_11649 = {'lower': keyword_11644, 'overwrite_ab': keyword_11646, 'overwrite_b': keyword_11648}
    # Getting the type of 'pbsv' (line 585)
    pbsv_11640 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 585, 21), 'pbsv', False)
    # Calling pbsv(args, kwargs) (line 585)
    pbsv_call_result_11650 = invoke(stypy.reporting.localization.Localization(__file__, 585, 21), pbsv_11640, *[a1_11641, b1_11642], **kwargs_11649)
    
    # Obtaining the member '__getitem__' of a type (line 585)
    getitem___11651 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 585, 8), pbsv_call_result_11650, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 585)
    subscript_call_result_11652 = invoke(stypy.reporting.localization.Localization(__file__, 585, 8), getitem___11651, int_11639)
    
    # Assigning a type to the variable 'tuple_var_assignment_10082' (line 585)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 585, 8), 'tuple_var_assignment_10082', subscript_call_result_11652)
    
    # Assigning a Subscript to a Name (line 585):
    
    # Obtaining the type of the subscript
    int_11653 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 585, 8), 'int')
    
    # Call to pbsv(...): (line 585)
    # Processing the call arguments (line 585)
    # Getting the type of 'a1' (line 585)
    a1_11655 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 585, 26), 'a1', False)
    # Getting the type of 'b1' (line 585)
    b1_11656 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 585, 30), 'b1', False)
    # Processing the call keyword arguments (line 585)
    # Getting the type of 'lower' (line 585)
    lower_11657 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 585, 40), 'lower', False)
    keyword_11658 = lower_11657
    # Getting the type of 'overwrite_ab' (line 585)
    overwrite_ab_11659 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 585, 60), 'overwrite_ab', False)
    keyword_11660 = overwrite_ab_11659
    # Getting the type of 'overwrite_b' (line 586)
    overwrite_b_11661 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 586, 38), 'overwrite_b', False)
    keyword_11662 = overwrite_b_11661
    kwargs_11663 = {'lower': keyword_11658, 'overwrite_ab': keyword_11660, 'overwrite_b': keyword_11662}
    # Getting the type of 'pbsv' (line 585)
    pbsv_11654 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 585, 21), 'pbsv', False)
    # Calling pbsv(args, kwargs) (line 585)
    pbsv_call_result_11664 = invoke(stypy.reporting.localization.Localization(__file__, 585, 21), pbsv_11654, *[a1_11655, b1_11656], **kwargs_11663)
    
    # Obtaining the member '__getitem__' of a type (line 585)
    getitem___11665 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 585, 8), pbsv_call_result_11664, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 585)
    subscript_call_result_11666 = invoke(stypy.reporting.localization.Localization(__file__, 585, 8), getitem___11665, int_11653)
    
    # Assigning a type to the variable 'tuple_var_assignment_10083' (line 585)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 585, 8), 'tuple_var_assignment_10083', subscript_call_result_11666)
    
    # Assigning a Name to a Name (line 585):
    # Getting the type of 'tuple_var_assignment_10081' (line 585)
    tuple_var_assignment_10081_11667 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 585, 8), 'tuple_var_assignment_10081')
    # Assigning a type to the variable 'c' (line 585)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 585, 8), 'c', tuple_var_assignment_10081_11667)
    
    # Assigning a Name to a Name (line 585):
    # Getting the type of 'tuple_var_assignment_10082' (line 585)
    tuple_var_assignment_10082_11668 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 585, 8), 'tuple_var_assignment_10082')
    # Assigning a type to the variable 'x' (line 585)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 585, 11), 'x', tuple_var_assignment_10082_11668)
    
    # Assigning a Name to a Name (line 585):
    # Getting the type of 'tuple_var_assignment_10083' (line 585)
    tuple_var_assignment_10083_11669 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 585, 8), 'tuple_var_assignment_10083')
    # Assigning a type to the variable 'info' (line 585)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 585, 14), 'info', tuple_var_assignment_10083_11669)
    # SSA join for if statement (line 573)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'info' (line 587)
    info_11670 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 587, 7), 'info')
    int_11671 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 587, 14), 'int')
    # Applying the binary operator '>' (line 587)
    result_gt_11672 = python_operator(stypy.reporting.localization.Localization(__file__, 587, 7), '>', info_11670, int_11671)
    
    # Testing the type of an if condition (line 587)
    if_condition_11673 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 587, 4), result_gt_11672)
    # Assigning a type to the variable 'if_condition_11673' (line 587)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 587, 4), 'if_condition_11673', if_condition_11673)
    # SSA begins for if statement (line 587)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to LinAlgError(...): (line 588)
    # Processing the call arguments (line 588)
    str_11675 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 588, 26), 'str', '%d-th leading minor not positive definite')
    # Getting the type of 'info' (line 588)
    info_11676 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 588, 72), 'info', False)
    # Applying the binary operator '%' (line 588)
    result_mod_11677 = python_operator(stypy.reporting.localization.Localization(__file__, 588, 26), '%', str_11675, info_11676)
    
    # Processing the call keyword arguments (line 588)
    kwargs_11678 = {}
    # Getting the type of 'LinAlgError' (line 588)
    LinAlgError_11674 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 588, 14), 'LinAlgError', False)
    # Calling LinAlgError(args, kwargs) (line 588)
    LinAlgError_call_result_11679 = invoke(stypy.reporting.localization.Localization(__file__, 588, 14), LinAlgError_11674, *[result_mod_11677], **kwargs_11678)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 588, 8), LinAlgError_call_result_11679, 'raise parameter', BaseException)
    # SSA join for if statement (line 587)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'info' (line 589)
    info_11680 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 589, 7), 'info')
    int_11681 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 589, 14), 'int')
    # Applying the binary operator '<' (line 589)
    result_lt_11682 = python_operator(stypy.reporting.localization.Localization(__file__, 589, 7), '<', info_11680, int_11681)
    
    # Testing the type of an if condition (line 589)
    if_condition_11683 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 589, 4), result_lt_11682)
    # Assigning a type to the variable 'if_condition_11683' (line 589)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 589, 4), 'if_condition_11683', if_condition_11683)
    # SSA begins for if statement (line 589)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 590)
    # Processing the call arguments (line 590)
    str_11685 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 590, 25), 'str', 'illegal value in %d-th argument of internal pbsv')
    
    # Getting the type of 'info' (line 591)
    info_11686 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 591, 35), 'info', False)
    # Applying the 'usub' unary operator (line 591)
    result___neg___11687 = python_operator(stypy.reporting.localization.Localization(__file__, 591, 34), 'usub', info_11686)
    
    # Applying the binary operator '%' (line 590)
    result_mod_11688 = python_operator(stypy.reporting.localization.Localization(__file__, 590, 25), '%', str_11685, result___neg___11687)
    
    # Processing the call keyword arguments (line 590)
    kwargs_11689 = {}
    # Getting the type of 'ValueError' (line 590)
    ValueError_11684 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 590, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 590)
    ValueError_call_result_11690 = invoke(stypy.reporting.localization.Localization(__file__, 590, 14), ValueError_11684, *[result_mod_11688], **kwargs_11689)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 590, 8), ValueError_call_result_11690, 'raise parameter', BaseException)
    # SSA join for if statement (line 589)
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'x' (line 592)
    x_11691 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 592, 11), 'x')
    # Assigning a type to the variable 'stypy_return_type' (line 592)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 592, 4), 'stypy_return_type', x_11691)
    
    # ################# End of 'solveh_banded(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'solveh_banded' in the type store
    # Getting the type of 'stypy_return_type' (line 472)
    stypy_return_type_11692 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 472, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_11692)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'solveh_banded'
    return stypy_return_type_11692

# Assigning a type to the variable 'solveh_banded' (line 472)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 472, 0), 'solveh_banded', solveh_banded)

@norecursion
def solve_toeplitz(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'True' (line 595)
    True_11693 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 595, 44), 'True')
    defaults = [True_11693]
    # Create a new context for function 'solve_toeplitz'
    module_type_store = module_type_store.open_function_context('solve_toeplitz', 595, 0, False)
    
    # Passed parameters checking function
    solve_toeplitz.stypy_localization = localization
    solve_toeplitz.stypy_type_of_self = None
    solve_toeplitz.stypy_type_store = module_type_store
    solve_toeplitz.stypy_function_name = 'solve_toeplitz'
    solve_toeplitz.stypy_param_names_list = ['c_or_cr', 'b', 'check_finite']
    solve_toeplitz.stypy_varargs_param_name = None
    solve_toeplitz.stypy_kwargs_param_name = None
    solve_toeplitz.stypy_call_defaults = defaults
    solve_toeplitz.stypy_call_varargs = varargs
    solve_toeplitz.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'solve_toeplitz', ['c_or_cr', 'b', 'check_finite'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'solve_toeplitz', localization, ['c_or_cr', 'b', 'check_finite'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'solve_toeplitz(...)' code ##################

    str_11694 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 661, (-1)), 'str', 'Solve a Toeplitz system using Levinson Recursion\n\n    The Toeplitz matrix has constant diagonals, with c as its first column\n    and r as its first row.  If r is not given, ``r == conjugate(c)`` is\n    assumed.\n\n    Parameters\n    ----------\n    c_or_cr : array_like or tuple of (array_like, array_like)\n        The vector ``c``, or a tuple of arrays (``c``, ``r``). Whatever the\n        actual shape of ``c``, it will be converted to a 1-D array. If not\n        supplied, ``r = conjugate(c)`` is assumed; in this case, if c[0] is\n        real, the Toeplitz matrix is Hermitian. r[0] is ignored; the first row\n        of the Toeplitz matrix is ``[c[0], r[1:]]``.  Whatever the actual shape\n        of ``r``, it will be converted to a 1-D array.\n    b : (M,) or (M, K) array_like\n        Right-hand side in ``T x = b``.\n    check_finite : bool, optional\n        Whether to check that the input matrices contain only finite numbers.\n        Disabling may give a performance gain, but may result in problems\n        (result entirely NaNs) if the inputs do contain infinities or NaNs.\n\n    Returns\n    -------\n    x : (M,) or (M, K) ndarray\n        The solution to the system ``T x = b``.  Shape of return matches shape\n        of `b`.\n\n    See Also\n    --------\n    toeplitz : Toeplitz matrix\n\n    Notes\n    -----\n    The solution is computed using Levinson-Durbin recursion, which is faster\n    than generic least-squares methods, but can be less numerically stable.\n\n    Examples\n    --------\n    Solve the Toeplitz system T x = b, where::\n\n            [ 1 -1 -2 -3]       [1]\n        T = [ 3  1 -1 -2]   b = [2]\n            [ 6  3  1 -1]       [2]\n            [10  6  3  1]       [5]\n\n    To specify the Toeplitz matrix, only the first column and the first\n    row are needed.\n\n    >>> c = np.array([1, 3, 6, 10])    # First column of T\n    >>> r = np.array([1, -1, -2, -3])  # First row of T\n    >>> b = np.array([1, 2, 2, 5])\n\n    >>> from scipy.linalg import solve_toeplitz, toeplitz\n    >>> x = solve_toeplitz((c, r), b)\n    >>> x\n    array([ 1.66666667, -1.        , -2.66666667,  2.33333333])\n\n    Check the result by creating the full Toeplitz matrix and\n    multiplying it by `x`.  We should get `b`.\n\n    >>> T = toeplitz(c, r)\n    >>> T.dot(x)\n    array([ 1.,  2.,  2.,  5.])\n\n    ')
    
    # Type idiom detected: calculating its left and rigth part (line 665)
    # Getting the type of 'tuple' (line 665)
    tuple_11695 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 665, 27), 'tuple')
    # Getting the type of 'c_or_cr' (line 665)
    c_or_cr_11696 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 665, 18), 'c_or_cr')
    
    (may_be_11697, more_types_in_union_11698) = may_be_subtype(tuple_11695, c_or_cr_11696)

    if may_be_11697:

        if more_types_in_union_11698:
            # Runtime conditional SSA (line 665)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        # Assigning a type to the variable 'c_or_cr' (line 665)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 665, 4), 'c_or_cr', remove_not_subtype_from_union(c_or_cr_11696, tuple))
        
        # Assigning a Name to a Tuple (line 666):
        
        # Assigning a Subscript to a Name (line 666):
        
        # Obtaining the type of the subscript
        int_11699 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 666, 8), 'int')
        # Getting the type of 'c_or_cr' (line 666)
        c_or_cr_11700 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 666, 15), 'c_or_cr')
        # Obtaining the member '__getitem__' of a type (line 666)
        getitem___11701 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 666, 8), c_or_cr_11700, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 666)
        subscript_call_result_11702 = invoke(stypy.reporting.localization.Localization(__file__, 666, 8), getitem___11701, int_11699)
        
        # Assigning a type to the variable 'tuple_var_assignment_10084' (line 666)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 666, 8), 'tuple_var_assignment_10084', subscript_call_result_11702)
        
        # Assigning a Subscript to a Name (line 666):
        
        # Obtaining the type of the subscript
        int_11703 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 666, 8), 'int')
        # Getting the type of 'c_or_cr' (line 666)
        c_or_cr_11704 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 666, 15), 'c_or_cr')
        # Obtaining the member '__getitem__' of a type (line 666)
        getitem___11705 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 666, 8), c_or_cr_11704, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 666)
        subscript_call_result_11706 = invoke(stypy.reporting.localization.Localization(__file__, 666, 8), getitem___11705, int_11703)
        
        # Assigning a type to the variable 'tuple_var_assignment_10085' (line 666)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 666, 8), 'tuple_var_assignment_10085', subscript_call_result_11706)
        
        # Assigning a Name to a Name (line 666):
        # Getting the type of 'tuple_var_assignment_10084' (line 666)
        tuple_var_assignment_10084_11707 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 666, 8), 'tuple_var_assignment_10084')
        # Assigning a type to the variable 'c' (line 666)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 666, 8), 'c', tuple_var_assignment_10084_11707)
        
        # Assigning a Name to a Name (line 666):
        # Getting the type of 'tuple_var_assignment_10085' (line 666)
        tuple_var_assignment_10085_11708 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 666, 8), 'tuple_var_assignment_10085')
        # Assigning a type to the variable 'r' (line 666)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 666, 11), 'r', tuple_var_assignment_10085_11708)
        
        # Assigning a Call to a Name (line 667):
        
        # Assigning a Call to a Name (line 667):
        
        # Call to ravel(...): (line 667)
        # Processing the call keyword arguments (line 667)
        kwargs_11716 = {}
        
        # Call to _asarray_validated(...): (line 667)
        # Processing the call arguments (line 667)
        # Getting the type of 'c' (line 667)
        c_11710 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 667, 31), 'c', False)
        # Processing the call keyword arguments (line 667)
        # Getting the type of 'check_finite' (line 667)
        check_finite_11711 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 667, 47), 'check_finite', False)
        keyword_11712 = check_finite_11711
        kwargs_11713 = {'check_finite': keyword_11712}
        # Getting the type of '_asarray_validated' (line 667)
        _asarray_validated_11709 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 667, 12), '_asarray_validated', False)
        # Calling _asarray_validated(args, kwargs) (line 667)
        _asarray_validated_call_result_11714 = invoke(stypy.reporting.localization.Localization(__file__, 667, 12), _asarray_validated_11709, *[c_11710], **kwargs_11713)
        
        # Obtaining the member 'ravel' of a type (line 667)
        ravel_11715 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 667, 12), _asarray_validated_call_result_11714, 'ravel')
        # Calling ravel(args, kwargs) (line 667)
        ravel_call_result_11717 = invoke(stypy.reporting.localization.Localization(__file__, 667, 12), ravel_11715, *[], **kwargs_11716)
        
        # Assigning a type to the variable 'c' (line 667)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 667, 8), 'c', ravel_call_result_11717)
        
        # Assigning a Call to a Name (line 668):
        
        # Assigning a Call to a Name (line 668):
        
        # Call to ravel(...): (line 668)
        # Processing the call keyword arguments (line 668)
        kwargs_11725 = {}
        
        # Call to _asarray_validated(...): (line 668)
        # Processing the call arguments (line 668)
        # Getting the type of 'r' (line 668)
        r_11719 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 668, 31), 'r', False)
        # Processing the call keyword arguments (line 668)
        # Getting the type of 'check_finite' (line 668)
        check_finite_11720 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 668, 47), 'check_finite', False)
        keyword_11721 = check_finite_11720
        kwargs_11722 = {'check_finite': keyword_11721}
        # Getting the type of '_asarray_validated' (line 668)
        _asarray_validated_11718 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 668, 12), '_asarray_validated', False)
        # Calling _asarray_validated(args, kwargs) (line 668)
        _asarray_validated_call_result_11723 = invoke(stypy.reporting.localization.Localization(__file__, 668, 12), _asarray_validated_11718, *[r_11719], **kwargs_11722)
        
        # Obtaining the member 'ravel' of a type (line 668)
        ravel_11724 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 668, 12), _asarray_validated_call_result_11723, 'ravel')
        # Calling ravel(args, kwargs) (line 668)
        ravel_call_result_11726 = invoke(stypy.reporting.localization.Localization(__file__, 668, 12), ravel_11724, *[], **kwargs_11725)
        
        # Assigning a type to the variable 'r' (line 668)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 668, 8), 'r', ravel_call_result_11726)

        if more_types_in_union_11698:
            # Runtime conditional SSA for else branch (line 665)
            module_type_store.open_ssa_branch('idiom else')



    if ((not may_be_11697) or more_types_in_union_11698):
        # Assigning a type to the variable 'c_or_cr' (line 665)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 665, 4), 'c_or_cr', remove_subtype_from_union(c_or_cr_11696, tuple))
        
        # Assigning a Call to a Name (line 670):
        
        # Assigning a Call to a Name (line 670):
        
        # Call to ravel(...): (line 670)
        # Processing the call keyword arguments (line 670)
        kwargs_11734 = {}
        
        # Call to _asarray_validated(...): (line 670)
        # Processing the call arguments (line 670)
        # Getting the type of 'c_or_cr' (line 670)
        c_or_cr_11728 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 670, 31), 'c_or_cr', False)
        # Processing the call keyword arguments (line 670)
        # Getting the type of 'check_finite' (line 670)
        check_finite_11729 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 670, 53), 'check_finite', False)
        keyword_11730 = check_finite_11729
        kwargs_11731 = {'check_finite': keyword_11730}
        # Getting the type of '_asarray_validated' (line 670)
        _asarray_validated_11727 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 670, 12), '_asarray_validated', False)
        # Calling _asarray_validated(args, kwargs) (line 670)
        _asarray_validated_call_result_11732 = invoke(stypy.reporting.localization.Localization(__file__, 670, 12), _asarray_validated_11727, *[c_or_cr_11728], **kwargs_11731)
        
        # Obtaining the member 'ravel' of a type (line 670)
        ravel_11733 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 670, 12), _asarray_validated_call_result_11732, 'ravel')
        # Calling ravel(args, kwargs) (line 670)
        ravel_call_result_11735 = invoke(stypy.reporting.localization.Localization(__file__, 670, 12), ravel_11733, *[], **kwargs_11734)
        
        # Assigning a type to the variable 'c' (line 670)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 670, 8), 'c', ravel_call_result_11735)
        
        # Assigning a Call to a Name (line 671):
        
        # Assigning a Call to a Name (line 671):
        
        # Call to conjugate(...): (line 671)
        # Processing the call keyword arguments (line 671)
        kwargs_11738 = {}
        # Getting the type of 'c' (line 671)
        c_11736 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 671, 12), 'c', False)
        # Obtaining the member 'conjugate' of a type (line 671)
        conjugate_11737 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 671, 12), c_11736, 'conjugate')
        # Calling conjugate(args, kwargs) (line 671)
        conjugate_call_result_11739 = invoke(stypy.reporting.localization.Localization(__file__, 671, 12), conjugate_11737, *[], **kwargs_11738)
        
        # Assigning a type to the variable 'r' (line 671)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 671, 8), 'r', conjugate_call_result_11739)

        if (may_be_11697 and more_types_in_union_11698):
            # SSA join for if statement (line 665)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Assigning a Call to a Name (line 675):
    
    # Assigning a Call to a Name (line 675):
    
    # Call to concatenate(...): (line 675)
    # Processing the call arguments (line 675)
    
    # Obtaining an instance of the builtin type 'tuple' (line 675)
    tuple_11742 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 675, 27), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 675)
    # Adding element type (line 675)
    
    # Obtaining the type of the subscript
    int_11743 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 675, 29), 'int')
    int_11744 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 675, 32), 'int')
    int_11745 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 675, 34), 'int')
    slice_11746 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 675, 27), int_11743, int_11744, int_11745)
    # Getting the type of 'r' (line 675)
    r_11747 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 675, 27), 'r', False)
    # Obtaining the member '__getitem__' of a type (line 675)
    getitem___11748 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 675, 27), r_11747, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 675)
    subscript_call_result_11749 = invoke(stypy.reporting.localization.Localization(__file__, 675, 27), getitem___11748, slice_11746)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 675, 27), tuple_11742, subscript_call_result_11749)
    # Adding element type (line 675)
    # Getting the type of 'c' (line 675)
    c_11750 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 675, 39), 'c', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 675, 27), tuple_11742, c_11750)
    
    # Processing the call keyword arguments (line 675)
    kwargs_11751 = {}
    # Getting the type of 'np' (line 675)
    np_11740 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 675, 11), 'np', False)
    # Obtaining the member 'concatenate' of a type (line 675)
    concatenate_11741 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 675, 11), np_11740, 'concatenate')
    # Calling concatenate(args, kwargs) (line 675)
    concatenate_call_result_11752 = invoke(stypy.reporting.localization.Localization(__file__, 675, 11), concatenate_11741, *[tuple_11742], **kwargs_11751)
    
    # Assigning a type to the variable 'vals' (line 675)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 675, 4), 'vals', concatenate_call_result_11752)
    
    # Type idiom detected: calculating its left and rigth part (line 676)
    # Getting the type of 'b' (line 676)
    b_11753 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 676, 7), 'b')
    # Getting the type of 'None' (line 676)
    None_11754 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 676, 12), 'None')
    
    (may_be_11755, more_types_in_union_11756) = may_be_none(b_11753, None_11754)

    if may_be_11755:

        if more_types_in_union_11756:
            # Runtime conditional SSA (line 676)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Call to ValueError(...): (line 677)
        # Processing the call arguments (line 677)
        str_11758 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 677, 25), 'str', 'illegal value, `b` is a required argument')
        # Processing the call keyword arguments (line 677)
        kwargs_11759 = {}
        # Getting the type of 'ValueError' (line 677)
        ValueError_11757 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 677, 14), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 677)
        ValueError_call_result_11760 = invoke(stypy.reporting.localization.Localization(__file__, 677, 14), ValueError_11757, *[str_11758], **kwargs_11759)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 677, 8), ValueError_call_result_11760, 'raise parameter', BaseException)

        if more_types_in_union_11756:
            # SSA join for if statement (line 676)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Assigning a Call to a Name (line 679):
    
    # Assigning a Call to a Name (line 679):
    
    # Call to _asarray_validated(...): (line 679)
    # Processing the call arguments (line 679)
    # Getting the type of 'b' (line 679)
    b_11762 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 679, 27), 'b', False)
    # Processing the call keyword arguments (line 679)
    kwargs_11763 = {}
    # Getting the type of '_asarray_validated' (line 679)
    _asarray_validated_11761 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 679, 8), '_asarray_validated', False)
    # Calling _asarray_validated(args, kwargs) (line 679)
    _asarray_validated_call_result_11764 = invoke(stypy.reporting.localization.Localization(__file__, 679, 8), _asarray_validated_11761, *[b_11762], **kwargs_11763)
    
    # Assigning a type to the variable 'b' (line 679)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 679, 4), 'b', _asarray_validated_call_result_11764)
    
    
    
    # Obtaining the type of the subscript
    int_11765 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 680, 18), 'int')
    # Getting the type of 'vals' (line 680)
    vals_11766 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 680, 7), 'vals')
    # Obtaining the member 'shape' of a type (line 680)
    shape_11767 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 680, 7), vals_11766, 'shape')
    # Obtaining the member '__getitem__' of a type (line 680)
    getitem___11768 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 680, 7), shape_11767, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 680)
    subscript_call_result_11769 = invoke(stypy.reporting.localization.Localization(__file__, 680, 7), getitem___11768, int_11765)
    
    int_11770 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 680, 25), 'int')
    
    # Obtaining the type of the subscript
    int_11771 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 680, 35), 'int')
    # Getting the type of 'b' (line 680)
    b_11772 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 680, 27), 'b')
    # Obtaining the member 'shape' of a type (line 680)
    shape_11773 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 680, 27), b_11772, 'shape')
    # Obtaining the member '__getitem__' of a type (line 680)
    getitem___11774 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 680, 27), shape_11773, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 680)
    subscript_call_result_11775 = invoke(stypy.reporting.localization.Localization(__file__, 680, 27), getitem___11774, int_11771)
    
    # Applying the binary operator '*' (line 680)
    result_mul_11776 = python_operator(stypy.reporting.localization.Localization(__file__, 680, 25), '*', int_11770, subscript_call_result_11775)
    
    int_11777 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 680, 40), 'int')
    # Applying the binary operator '-' (line 680)
    result_sub_11778 = python_operator(stypy.reporting.localization.Localization(__file__, 680, 25), '-', result_mul_11776, int_11777)
    
    # Applying the binary operator '!=' (line 680)
    result_ne_11779 = python_operator(stypy.reporting.localization.Localization(__file__, 680, 7), '!=', subscript_call_result_11769, result_sub_11778)
    
    # Testing the type of an if condition (line 680)
    if_condition_11780 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 680, 4), result_ne_11779)
    # Assigning a type to the variable 'if_condition_11780' (line 680)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 680, 4), 'if_condition_11780', if_condition_11780)
    # SSA begins for if statement (line 680)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 681)
    # Processing the call arguments (line 681)
    str_11782 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 681, 25), 'str', 'incompatible dimensions')
    # Processing the call keyword arguments (line 681)
    kwargs_11783 = {}
    # Getting the type of 'ValueError' (line 681)
    ValueError_11781 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 681, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 681)
    ValueError_call_result_11784 = invoke(stypy.reporting.localization.Localization(__file__, 681, 14), ValueError_11781, *[str_11782], **kwargs_11783)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 681, 8), ValueError_call_result_11784, 'raise parameter', BaseException)
    # SSA join for if statement (line 680)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Evaluating a boolean operation
    
    # Call to iscomplexobj(...): (line 682)
    # Processing the call arguments (line 682)
    # Getting the type of 'vals' (line 682)
    vals_11787 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 682, 23), 'vals', False)
    # Processing the call keyword arguments (line 682)
    kwargs_11788 = {}
    # Getting the type of 'np' (line 682)
    np_11785 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 682, 7), 'np', False)
    # Obtaining the member 'iscomplexobj' of a type (line 682)
    iscomplexobj_11786 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 682, 7), np_11785, 'iscomplexobj')
    # Calling iscomplexobj(args, kwargs) (line 682)
    iscomplexobj_call_result_11789 = invoke(stypy.reporting.localization.Localization(__file__, 682, 7), iscomplexobj_11786, *[vals_11787], **kwargs_11788)
    
    
    # Call to iscomplexobj(...): (line 682)
    # Processing the call arguments (line 682)
    # Getting the type of 'b' (line 682)
    b_11792 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 682, 48), 'b', False)
    # Processing the call keyword arguments (line 682)
    kwargs_11793 = {}
    # Getting the type of 'np' (line 682)
    np_11790 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 682, 32), 'np', False)
    # Obtaining the member 'iscomplexobj' of a type (line 682)
    iscomplexobj_11791 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 682, 32), np_11790, 'iscomplexobj')
    # Calling iscomplexobj(args, kwargs) (line 682)
    iscomplexobj_call_result_11794 = invoke(stypy.reporting.localization.Localization(__file__, 682, 32), iscomplexobj_11791, *[b_11792], **kwargs_11793)
    
    # Applying the binary operator 'or' (line 682)
    result_or_keyword_11795 = python_operator(stypy.reporting.localization.Localization(__file__, 682, 7), 'or', iscomplexobj_call_result_11789, iscomplexobj_call_result_11794)
    
    # Testing the type of an if condition (line 682)
    if_condition_11796 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 682, 4), result_or_keyword_11795)
    # Assigning a type to the variable 'if_condition_11796' (line 682)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 682, 4), 'if_condition_11796', if_condition_11796)
    # SSA begins for if statement (line 682)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 683):
    
    # Assigning a Call to a Name (line 683):
    
    # Call to asarray(...): (line 683)
    # Processing the call arguments (line 683)
    # Getting the type of 'vals' (line 683)
    vals_11799 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 683, 26), 'vals', False)
    # Processing the call keyword arguments (line 683)
    # Getting the type of 'np' (line 683)
    np_11800 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 683, 38), 'np', False)
    # Obtaining the member 'complex128' of a type (line 683)
    complex128_11801 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 683, 38), np_11800, 'complex128')
    keyword_11802 = complex128_11801
    str_11803 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 683, 59), 'str', 'c')
    keyword_11804 = str_11803
    kwargs_11805 = {'dtype': keyword_11802, 'order': keyword_11804}
    # Getting the type of 'np' (line 683)
    np_11797 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 683, 15), 'np', False)
    # Obtaining the member 'asarray' of a type (line 683)
    asarray_11798 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 683, 15), np_11797, 'asarray')
    # Calling asarray(args, kwargs) (line 683)
    asarray_call_result_11806 = invoke(stypy.reporting.localization.Localization(__file__, 683, 15), asarray_11798, *[vals_11799], **kwargs_11805)
    
    # Assigning a type to the variable 'vals' (line 683)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 683, 8), 'vals', asarray_call_result_11806)
    
    # Assigning a Call to a Name (line 684):
    
    # Assigning a Call to a Name (line 684):
    
    # Call to asarray(...): (line 684)
    # Processing the call arguments (line 684)
    # Getting the type of 'b' (line 684)
    b_11809 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 684, 23), 'b', False)
    # Processing the call keyword arguments (line 684)
    # Getting the type of 'np' (line 684)
    np_11810 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 684, 32), 'np', False)
    # Obtaining the member 'complex128' of a type (line 684)
    complex128_11811 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 684, 32), np_11810, 'complex128')
    keyword_11812 = complex128_11811
    kwargs_11813 = {'dtype': keyword_11812}
    # Getting the type of 'np' (line 684)
    np_11807 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 684, 12), 'np', False)
    # Obtaining the member 'asarray' of a type (line 684)
    asarray_11808 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 684, 12), np_11807, 'asarray')
    # Calling asarray(args, kwargs) (line 684)
    asarray_call_result_11814 = invoke(stypy.reporting.localization.Localization(__file__, 684, 12), asarray_11808, *[b_11809], **kwargs_11813)
    
    # Assigning a type to the variable 'b' (line 684)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 684, 8), 'b', asarray_call_result_11814)
    # SSA branch for the else part of an if statement (line 682)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Call to a Name (line 686):
    
    # Assigning a Call to a Name (line 686):
    
    # Call to asarray(...): (line 686)
    # Processing the call arguments (line 686)
    # Getting the type of 'vals' (line 686)
    vals_11817 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 686, 26), 'vals', False)
    # Processing the call keyword arguments (line 686)
    # Getting the type of 'np' (line 686)
    np_11818 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 686, 38), 'np', False)
    # Obtaining the member 'double' of a type (line 686)
    double_11819 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 686, 38), np_11818, 'double')
    keyword_11820 = double_11819
    str_11821 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 686, 55), 'str', 'c')
    keyword_11822 = str_11821
    kwargs_11823 = {'dtype': keyword_11820, 'order': keyword_11822}
    # Getting the type of 'np' (line 686)
    np_11815 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 686, 15), 'np', False)
    # Obtaining the member 'asarray' of a type (line 686)
    asarray_11816 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 686, 15), np_11815, 'asarray')
    # Calling asarray(args, kwargs) (line 686)
    asarray_call_result_11824 = invoke(stypy.reporting.localization.Localization(__file__, 686, 15), asarray_11816, *[vals_11817], **kwargs_11823)
    
    # Assigning a type to the variable 'vals' (line 686)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 686, 8), 'vals', asarray_call_result_11824)
    
    # Assigning a Call to a Name (line 687):
    
    # Assigning a Call to a Name (line 687):
    
    # Call to asarray(...): (line 687)
    # Processing the call arguments (line 687)
    # Getting the type of 'b' (line 687)
    b_11827 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 687, 23), 'b', False)
    # Processing the call keyword arguments (line 687)
    # Getting the type of 'np' (line 687)
    np_11828 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 687, 32), 'np', False)
    # Obtaining the member 'double' of a type (line 687)
    double_11829 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 687, 32), np_11828, 'double')
    keyword_11830 = double_11829
    kwargs_11831 = {'dtype': keyword_11830}
    # Getting the type of 'np' (line 687)
    np_11825 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 687, 12), 'np', False)
    # Obtaining the member 'asarray' of a type (line 687)
    asarray_11826 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 687, 12), np_11825, 'asarray')
    # Calling asarray(args, kwargs) (line 687)
    asarray_call_result_11832 = invoke(stypy.reporting.localization.Localization(__file__, 687, 12), asarray_11826, *[b_11827], **kwargs_11831)
    
    # Assigning a type to the variable 'b' (line 687)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 687, 8), 'b', asarray_call_result_11832)
    # SSA join for if statement (line 682)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'b' (line 689)
    b_11833 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 689, 7), 'b')
    # Obtaining the member 'ndim' of a type (line 689)
    ndim_11834 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 689, 7), b_11833, 'ndim')
    int_11835 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 689, 17), 'int')
    # Applying the binary operator '==' (line 689)
    result_eq_11836 = python_operator(stypy.reporting.localization.Localization(__file__, 689, 7), '==', ndim_11834, int_11835)
    
    # Testing the type of an if condition (line 689)
    if_condition_11837 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 689, 4), result_eq_11836)
    # Assigning a type to the variable 'if_condition_11837' (line 689)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 689, 4), 'if_condition_11837', if_condition_11837)
    # SSA begins for if statement (line 689)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Tuple (line 690):
    
    # Assigning a Subscript to a Name (line 690):
    
    # Obtaining the type of the subscript
    int_11838 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 690, 8), 'int')
    
    # Call to levinson(...): (line 690)
    # Processing the call arguments (line 690)
    # Getting the type of 'vals' (line 690)
    vals_11840 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 690, 24), 'vals', False)
    
    # Call to ascontiguousarray(...): (line 690)
    # Processing the call arguments (line 690)
    # Getting the type of 'b' (line 690)
    b_11843 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 690, 51), 'b', False)
    # Processing the call keyword arguments (line 690)
    kwargs_11844 = {}
    # Getting the type of 'np' (line 690)
    np_11841 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 690, 30), 'np', False)
    # Obtaining the member 'ascontiguousarray' of a type (line 690)
    ascontiguousarray_11842 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 690, 30), np_11841, 'ascontiguousarray')
    # Calling ascontiguousarray(args, kwargs) (line 690)
    ascontiguousarray_call_result_11845 = invoke(stypy.reporting.localization.Localization(__file__, 690, 30), ascontiguousarray_11842, *[b_11843], **kwargs_11844)
    
    # Processing the call keyword arguments (line 690)
    kwargs_11846 = {}
    # Getting the type of 'levinson' (line 690)
    levinson_11839 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 690, 15), 'levinson', False)
    # Calling levinson(args, kwargs) (line 690)
    levinson_call_result_11847 = invoke(stypy.reporting.localization.Localization(__file__, 690, 15), levinson_11839, *[vals_11840, ascontiguousarray_call_result_11845], **kwargs_11846)
    
    # Obtaining the member '__getitem__' of a type (line 690)
    getitem___11848 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 690, 8), levinson_call_result_11847, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 690)
    subscript_call_result_11849 = invoke(stypy.reporting.localization.Localization(__file__, 690, 8), getitem___11848, int_11838)
    
    # Assigning a type to the variable 'tuple_var_assignment_10086' (line 690)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 690, 8), 'tuple_var_assignment_10086', subscript_call_result_11849)
    
    # Assigning a Subscript to a Name (line 690):
    
    # Obtaining the type of the subscript
    int_11850 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 690, 8), 'int')
    
    # Call to levinson(...): (line 690)
    # Processing the call arguments (line 690)
    # Getting the type of 'vals' (line 690)
    vals_11852 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 690, 24), 'vals', False)
    
    # Call to ascontiguousarray(...): (line 690)
    # Processing the call arguments (line 690)
    # Getting the type of 'b' (line 690)
    b_11855 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 690, 51), 'b', False)
    # Processing the call keyword arguments (line 690)
    kwargs_11856 = {}
    # Getting the type of 'np' (line 690)
    np_11853 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 690, 30), 'np', False)
    # Obtaining the member 'ascontiguousarray' of a type (line 690)
    ascontiguousarray_11854 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 690, 30), np_11853, 'ascontiguousarray')
    # Calling ascontiguousarray(args, kwargs) (line 690)
    ascontiguousarray_call_result_11857 = invoke(stypy.reporting.localization.Localization(__file__, 690, 30), ascontiguousarray_11854, *[b_11855], **kwargs_11856)
    
    # Processing the call keyword arguments (line 690)
    kwargs_11858 = {}
    # Getting the type of 'levinson' (line 690)
    levinson_11851 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 690, 15), 'levinson', False)
    # Calling levinson(args, kwargs) (line 690)
    levinson_call_result_11859 = invoke(stypy.reporting.localization.Localization(__file__, 690, 15), levinson_11851, *[vals_11852, ascontiguousarray_call_result_11857], **kwargs_11858)
    
    # Obtaining the member '__getitem__' of a type (line 690)
    getitem___11860 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 690, 8), levinson_call_result_11859, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 690)
    subscript_call_result_11861 = invoke(stypy.reporting.localization.Localization(__file__, 690, 8), getitem___11860, int_11850)
    
    # Assigning a type to the variable 'tuple_var_assignment_10087' (line 690)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 690, 8), 'tuple_var_assignment_10087', subscript_call_result_11861)
    
    # Assigning a Name to a Name (line 690):
    # Getting the type of 'tuple_var_assignment_10086' (line 690)
    tuple_var_assignment_10086_11862 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 690, 8), 'tuple_var_assignment_10086')
    # Assigning a type to the variable 'x' (line 690)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 690, 8), 'x', tuple_var_assignment_10086_11862)
    
    # Assigning a Name to a Name (line 690):
    # Getting the type of 'tuple_var_assignment_10087' (line 690)
    tuple_var_assignment_10087_11863 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 690, 8), 'tuple_var_assignment_10087')
    # Assigning a type to the variable '_' (line 690)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 690, 11), '_', tuple_var_assignment_10087_11863)
    # SSA branch for the else part of an if statement (line 689)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Attribute to a Name (line 692):
    
    # Assigning a Attribute to a Name (line 692):
    # Getting the type of 'b' (line 692)
    b_11864 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 692, 18), 'b')
    # Obtaining the member 'shape' of a type (line 692)
    shape_11865 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 692, 18), b_11864, 'shape')
    # Assigning a type to the variable 'b_shape' (line 692)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 692, 8), 'b_shape', shape_11865)
    
    # Assigning a Call to a Name (line 693):
    
    # Assigning a Call to a Name (line 693):
    
    # Call to reshape(...): (line 693)
    # Processing the call arguments (line 693)
    
    # Obtaining the type of the subscript
    int_11868 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 693, 30), 'int')
    # Getting the type of 'b' (line 693)
    b_11869 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 693, 22), 'b', False)
    # Obtaining the member 'shape' of a type (line 693)
    shape_11870 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 693, 22), b_11869, 'shape')
    # Obtaining the member '__getitem__' of a type (line 693)
    getitem___11871 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 693, 22), shape_11870, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 693)
    subscript_call_result_11872 = invoke(stypy.reporting.localization.Localization(__file__, 693, 22), getitem___11871, int_11868)
    
    int_11873 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 693, 34), 'int')
    # Processing the call keyword arguments (line 693)
    kwargs_11874 = {}
    # Getting the type of 'b' (line 693)
    b_11866 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 693, 12), 'b', False)
    # Obtaining the member 'reshape' of a type (line 693)
    reshape_11867 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 693, 12), b_11866, 'reshape')
    # Calling reshape(args, kwargs) (line 693)
    reshape_call_result_11875 = invoke(stypy.reporting.localization.Localization(__file__, 693, 12), reshape_11867, *[subscript_call_result_11872, int_11873], **kwargs_11874)
    
    # Assigning a type to the variable 'b' (line 693)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 693, 8), 'b', reshape_call_result_11875)
    
    # Assigning a Call to a Name (line 694):
    
    # Assigning a Call to a Name (line 694):
    
    # Call to column_stack(...): (line 694)
    # Processing the call arguments (line 694)
    # Calculating generator expression
    module_type_store = module_type_store.open_function_context('list comprehension expression', 695, 12, True)
    # Calculating comprehension expression
    
    # Call to range(...): (line 696)
    # Processing the call arguments (line 696)
    
    # Obtaining the type of the subscript
    int_11895 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 696, 35), 'int')
    # Getting the type of 'b' (line 696)
    b_11896 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 696, 27), 'b', False)
    # Obtaining the member 'shape' of a type (line 696)
    shape_11897 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 696, 27), b_11896, 'shape')
    # Obtaining the member '__getitem__' of a type (line 696)
    getitem___11898 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 696, 27), shape_11897, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 696)
    subscript_call_result_11899 = invoke(stypy.reporting.localization.Localization(__file__, 696, 27), getitem___11898, int_11895)
    
    # Processing the call keyword arguments (line 696)
    kwargs_11900 = {}
    # Getting the type of 'range' (line 696)
    range_11894 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 696, 21), 'range', False)
    # Calling range(args, kwargs) (line 696)
    range_call_result_11901 = invoke(stypy.reporting.localization.Localization(__file__, 696, 21), range_11894, *[subscript_call_result_11899], **kwargs_11900)
    
    comprehension_11902 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 695, 12), range_call_result_11901)
    # Assigning a type to the variable 'i' (line 695)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 695, 12), 'i', comprehension_11902)
    
    # Obtaining the type of the subscript
    int_11878 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 695, 59), 'int')
    
    # Call to levinson(...): (line 695)
    # Processing the call arguments (line 695)
    # Getting the type of 'vals' (line 695)
    vals_11880 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 695, 22), 'vals', False)
    
    # Call to ascontiguousarray(...): (line 695)
    # Processing the call arguments (line 695)
    
    # Obtaining the type of the subscript
    slice_11883 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 695, 49), None, None, None)
    # Getting the type of 'i' (line 695)
    i_11884 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 695, 54), 'i', False)
    # Getting the type of 'b' (line 695)
    b_11885 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 695, 49), 'b', False)
    # Obtaining the member '__getitem__' of a type (line 695)
    getitem___11886 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 695, 49), b_11885, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 695)
    subscript_call_result_11887 = invoke(stypy.reporting.localization.Localization(__file__, 695, 49), getitem___11886, (slice_11883, i_11884))
    
    # Processing the call keyword arguments (line 695)
    kwargs_11888 = {}
    # Getting the type of 'np' (line 695)
    np_11881 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 695, 28), 'np', False)
    # Obtaining the member 'ascontiguousarray' of a type (line 695)
    ascontiguousarray_11882 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 695, 28), np_11881, 'ascontiguousarray')
    # Calling ascontiguousarray(args, kwargs) (line 695)
    ascontiguousarray_call_result_11889 = invoke(stypy.reporting.localization.Localization(__file__, 695, 28), ascontiguousarray_11882, *[subscript_call_result_11887], **kwargs_11888)
    
    # Processing the call keyword arguments (line 695)
    kwargs_11890 = {}
    # Getting the type of 'levinson' (line 695)
    levinson_11879 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 695, 13), 'levinson', False)
    # Calling levinson(args, kwargs) (line 695)
    levinson_call_result_11891 = invoke(stypy.reporting.localization.Localization(__file__, 695, 13), levinson_11879, *[vals_11880, ascontiguousarray_call_result_11889], **kwargs_11890)
    
    # Obtaining the member '__getitem__' of a type (line 695)
    getitem___11892 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 695, 13), levinson_call_result_11891, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 695)
    subscript_call_result_11893 = invoke(stypy.reporting.localization.Localization(__file__, 695, 13), getitem___11892, int_11878)
    
    list_11903 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 695, 12), 'list')
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 695, 12), list_11903, subscript_call_result_11893)
    # Processing the call keyword arguments (line 694)
    kwargs_11904 = {}
    # Getting the type of 'np' (line 694)
    np_11876 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 694, 12), 'np', False)
    # Obtaining the member 'column_stack' of a type (line 694)
    column_stack_11877 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 694, 12), np_11876, 'column_stack')
    # Calling column_stack(args, kwargs) (line 694)
    column_stack_call_result_11905 = invoke(stypy.reporting.localization.Localization(__file__, 694, 12), column_stack_11877, *[list_11903], **kwargs_11904)
    
    # Assigning a type to the variable 'x' (line 694)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 694, 8), 'x', column_stack_call_result_11905)
    
    # Assigning a Call to a Name (line 697):
    
    # Assigning a Call to a Name (line 697):
    
    # Call to reshape(...): (line 697)
    # Getting the type of 'b_shape' (line 697)
    b_shape_11908 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 697, 23), 'b_shape', False)
    # Processing the call keyword arguments (line 697)
    kwargs_11909 = {}
    # Getting the type of 'x' (line 697)
    x_11906 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 697, 12), 'x', False)
    # Obtaining the member 'reshape' of a type (line 697)
    reshape_11907 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 697, 12), x_11906, 'reshape')
    # Calling reshape(args, kwargs) (line 697)
    reshape_call_result_11910 = invoke(stypy.reporting.localization.Localization(__file__, 697, 12), reshape_11907, *[b_shape_11908], **kwargs_11909)
    
    # Assigning a type to the variable 'x' (line 697)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 697, 8), 'x', reshape_call_result_11910)
    # SSA join for if statement (line 689)
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'x' (line 699)
    x_11911 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 699, 11), 'x')
    # Assigning a type to the variable 'stypy_return_type' (line 699)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 699, 4), 'stypy_return_type', x_11911)
    
    # ################# End of 'solve_toeplitz(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'solve_toeplitz' in the type store
    # Getting the type of 'stypy_return_type' (line 595)
    stypy_return_type_11912 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 595, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_11912)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'solve_toeplitz'
    return stypy_return_type_11912

# Assigning a type to the variable 'solve_toeplitz' (line 595)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 595, 0), 'solve_toeplitz', solve_toeplitz)

@norecursion
def _get_axis_len(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_get_axis_len'
    module_type_store = module_type_store.open_function_context('_get_axis_len', 702, 0, False)
    
    # Passed parameters checking function
    _get_axis_len.stypy_localization = localization
    _get_axis_len.stypy_type_of_self = None
    _get_axis_len.stypy_type_store = module_type_store
    _get_axis_len.stypy_function_name = '_get_axis_len'
    _get_axis_len.stypy_param_names_list = ['aname', 'a', 'axis']
    _get_axis_len.stypy_varargs_param_name = None
    _get_axis_len.stypy_kwargs_param_name = None
    _get_axis_len.stypy_call_defaults = defaults
    _get_axis_len.stypy_call_varargs = varargs
    _get_axis_len.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_get_axis_len', ['aname', 'a', 'axis'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_get_axis_len', localization, ['aname', 'a', 'axis'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_get_axis_len(...)' code ##################

    
    # Assigning a Name to a Name (line 703):
    
    # Assigning a Name to a Name (line 703):
    # Getting the type of 'axis' (line 703)
    axis_11913 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 703, 9), 'axis')
    # Assigning a type to the variable 'ax' (line 703)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 703, 4), 'ax', axis_11913)
    
    
    # Getting the type of 'ax' (line 704)
    ax_11914 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 704, 7), 'ax')
    int_11915 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 704, 12), 'int')
    # Applying the binary operator '<' (line 704)
    result_lt_11916 = python_operator(stypy.reporting.localization.Localization(__file__, 704, 7), '<', ax_11914, int_11915)
    
    # Testing the type of an if condition (line 704)
    if_condition_11917 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 704, 4), result_lt_11916)
    # Assigning a type to the variable 'if_condition_11917' (line 704)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 704, 4), 'if_condition_11917', if_condition_11917)
    # SSA begins for if statement (line 704)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Getting the type of 'ax' (line 705)
    ax_11918 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 705, 8), 'ax')
    # Getting the type of 'a' (line 705)
    a_11919 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 705, 14), 'a')
    # Obtaining the member 'ndim' of a type (line 705)
    ndim_11920 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 705, 14), a_11919, 'ndim')
    # Applying the binary operator '+=' (line 705)
    result_iadd_11921 = python_operator(stypy.reporting.localization.Localization(__file__, 705, 8), '+=', ax_11918, ndim_11920)
    # Assigning a type to the variable 'ax' (line 705)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 705, 8), 'ax', result_iadd_11921)
    
    # SSA join for if statement (line 704)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    int_11922 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 706, 7), 'int')
    # Getting the type of 'ax' (line 706)
    ax_11923 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 706, 12), 'ax')
    # Applying the binary operator '<=' (line 706)
    result_le_11924 = python_operator(stypy.reporting.localization.Localization(__file__, 706, 7), '<=', int_11922, ax_11923)
    # Getting the type of 'a' (line 706)
    a_11925 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 706, 17), 'a')
    # Obtaining the member 'ndim' of a type (line 706)
    ndim_11926 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 706, 17), a_11925, 'ndim')
    # Applying the binary operator '<' (line 706)
    result_lt_11927 = python_operator(stypy.reporting.localization.Localization(__file__, 706, 7), '<', ax_11923, ndim_11926)
    # Applying the binary operator '&' (line 706)
    result_and__11928 = python_operator(stypy.reporting.localization.Localization(__file__, 706, 7), '&', result_le_11924, result_lt_11927)
    
    # Testing the type of an if condition (line 706)
    if_condition_11929 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 706, 4), result_and__11928)
    # Assigning a type to the variable 'if_condition_11929' (line 706)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 706, 4), 'if_condition_11929', if_condition_11929)
    # SSA begins for if statement (line 706)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Obtaining the type of the subscript
    # Getting the type of 'ax' (line 707)
    ax_11930 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 707, 23), 'ax')
    # Getting the type of 'a' (line 707)
    a_11931 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 707, 15), 'a')
    # Obtaining the member 'shape' of a type (line 707)
    shape_11932 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 707, 15), a_11931, 'shape')
    # Obtaining the member '__getitem__' of a type (line 707)
    getitem___11933 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 707, 15), shape_11932, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 707)
    subscript_call_result_11934 = invoke(stypy.reporting.localization.Localization(__file__, 707, 15), getitem___11933, ax_11930)
    
    # Assigning a type to the variable 'stypy_return_type' (line 707)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 707, 8), 'stypy_return_type', subscript_call_result_11934)
    # SSA join for if statement (line 706)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to ValueError(...): (line 708)
    # Processing the call arguments (line 708)
    str_11936 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 708, 21), 'str', "'%saxis' entry is out of bounds")
    
    # Obtaining an instance of the builtin type 'tuple' (line 708)
    tuple_11937 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 708, 58), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 708)
    # Adding element type (line 708)
    # Getting the type of 'aname' (line 708)
    aname_11938 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 708, 58), 'aname', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 708, 58), tuple_11937, aname_11938)
    
    # Applying the binary operator '%' (line 708)
    result_mod_11939 = python_operator(stypy.reporting.localization.Localization(__file__, 708, 21), '%', str_11936, tuple_11937)
    
    # Processing the call keyword arguments (line 708)
    kwargs_11940 = {}
    # Getting the type of 'ValueError' (line 708)
    ValueError_11935 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 708, 10), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 708)
    ValueError_call_result_11941 = invoke(stypy.reporting.localization.Localization(__file__, 708, 10), ValueError_11935, *[result_mod_11939], **kwargs_11940)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 708, 4), ValueError_call_result_11941, 'raise parameter', BaseException)
    
    # ################# End of '_get_axis_len(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_get_axis_len' in the type store
    # Getting the type of 'stypy_return_type' (line 702)
    stypy_return_type_11942 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 702, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_11942)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_get_axis_len'
    return stypy_return_type_11942

# Assigning a type to the variable '_get_axis_len' (line 702)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 702, 0), '_get_axis_len', _get_axis_len)

@norecursion
def solve_circulant(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    str_11943 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 711, 35), 'str', 'raise')
    # Getting the type of 'None' (line 711)
    None_11944 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 711, 48), 'None')
    int_11945 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 712, 26), 'int')
    int_11946 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 712, 36), 'int')
    int_11947 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 712, 47), 'int')
    defaults = [str_11943, None_11944, int_11945, int_11946, int_11947]
    # Create a new context for function 'solve_circulant'
    module_type_store = module_type_store.open_function_context('solve_circulant', 711, 0, False)
    
    # Passed parameters checking function
    solve_circulant.stypy_localization = localization
    solve_circulant.stypy_type_of_self = None
    solve_circulant.stypy_type_store = module_type_store
    solve_circulant.stypy_function_name = 'solve_circulant'
    solve_circulant.stypy_param_names_list = ['c', 'b', 'singular', 'tol', 'caxis', 'baxis', 'outaxis']
    solve_circulant.stypy_varargs_param_name = None
    solve_circulant.stypy_kwargs_param_name = None
    solve_circulant.stypy_call_defaults = defaults
    solve_circulant.stypy_call_varargs = varargs
    solve_circulant.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'solve_circulant', ['c', 'b', 'singular', 'tol', 'caxis', 'baxis', 'outaxis'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'solve_circulant', localization, ['c', 'b', 'singular', 'tol', 'caxis', 'baxis', 'outaxis'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'solve_circulant(...)' code ##################

    str_11948 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 858, (-1)), 'str', 'Solve C x = b for x, where C is a circulant matrix.\n\n    `C` is the circulant matrix associated with the vector `c`.\n\n    The system is solved by doing division in Fourier space.  The\n    calculation is::\n\n        x = ifft(fft(b) / fft(c))\n\n    where `fft` and `ifft` are the fast Fourier transform and its inverse,\n    respectively.  For a large vector `c`, this is *much* faster than\n    solving the system with the full circulant matrix.\n\n    Parameters\n    ----------\n    c : array_like\n        The coefficients of the circulant matrix.\n    b : array_like\n        Right-hand side matrix in ``a x = b``.\n    singular : str, optional\n        This argument controls how a near singular circulant matrix is\n        handled.  If `singular` is "raise" and the circulant matrix is\n        near singular, a `LinAlgError` is raised.  If `singular` is\n        "lstsq", the least squares solution is returned.  Default is "raise".\n    tol : float, optional\n        If any eigenvalue of the circulant matrix has an absolute value\n        that is less than or equal to `tol`, the matrix is considered to be\n        near singular.  If not given, `tol` is set to::\n\n            tol = abs_eigs.max() * abs_eigs.size * np.finfo(np.float64).eps\n\n        where `abs_eigs` is the array of absolute values of the eigenvalues\n        of the circulant matrix.\n    caxis : int\n        When `c` has dimension greater than 1, it is viewed as a collection\n        of circulant vectors.  In this case, `caxis` is the axis of `c` that\n        holds the vectors of circulant coefficients.\n    baxis : int\n        When `b` has dimension greater than 1, it is viewed as a collection\n        of vectors.  In this case, `baxis` is the axis of `b` that holds the\n        right-hand side vectors.\n    outaxis : int\n        When `c` or `b` are multidimensional, the value returned by\n        `solve_circulant` is multidimensional.  In this case, `outaxis` is\n        the axis of the result that holds the solution vectors.\n\n    Returns\n    -------\n    x : ndarray\n        Solution to the system ``C x = b``.\n\n    Raises\n    ------\n    LinAlgError\n        If the circulant matrix associated with `c` is near singular.\n\n    See Also\n    --------\n    circulant : circulant matrix\n\n    Notes\n    -----\n    For a one-dimensional vector `c` with length `m`, and an array `b`\n    with shape ``(m, ...)``,\n\n        solve_circulant(c, b)\n\n    returns the same result as\n\n        solve(circulant(c), b)\n\n    where `solve` and `circulant` are from `scipy.linalg`.\n\n    .. versionadded:: 0.16.0\n\n    Examples\n    --------\n    >>> from scipy.linalg import solve_circulant, solve, circulant, lstsq\n\n    >>> c = np.array([2, 2, 4])\n    >>> b = np.array([1, 2, 3])\n    >>> solve_circulant(c, b)\n    array([ 0.75, -0.25,  0.25])\n\n    Compare that result to solving the system with `scipy.linalg.solve`:\n\n    >>> solve(circulant(c), b)\n    array([ 0.75, -0.25,  0.25])\n\n    A singular example:\n\n    >>> c = np.array([1, 1, 0, 0])\n    >>> b = np.array([1, 2, 3, 4])\n\n    Calling ``solve_circulant(c, b)`` will raise a `LinAlgError`.  For the\n    least square solution, use the option ``singular=\'lstsq\'``:\n\n    >>> solve_circulant(c, b, singular=\'lstsq\')\n    array([ 0.25,  1.25,  2.25,  1.25])\n\n    Compare to `scipy.linalg.lstsq`:\n\n    >>> x, resid, rnk, s = lstsq(circulant(c), b)\n    >>> x\n    array([ 0.25,  1.25,  2.25,  1.25])\n\n    A broadcasting example:\n\n    Suppose we have the vectors of two circulant matrices stored in an array\n    with shape (2, 5), and three `b` vectors stored in an array with shape\n    (3, 5).  For example,\n\n    >>> c = np.array([[1.5, 2, 3, 0, 0], [1, 1, 4, 3, 2]])\n    >>> b = np.arange(15).reshape(-1, 5)\n\n    We want to solve all combinations of circulant matrices and `b` vectors,\n    with the result stored in an array with shape (2, 3, 5).  When we\n    disregard the axes of `c` and `b` that hold the vectors of coefficients,\n    the shapes of the collections are (2,) and (3,), respectively, which are\n    not compatible for broadcasting.  To have a broadcast result with shape\n    (2, 3), we add a trivial dimension to `c`: ``c[:, np.newaxis, :]`` has\n    shape (2, 1, 5).  The last dimension holds the coefficients of the\n    circulant matrices, so when we call `solve_circulant`, we can use the\n    default ``caxis=-1``.  The coefficients of the `b` vectors are in the last\n    dimension of the array `b`, so we use ``baxis=-1``.  If we use the\n    default `outaxis`, the result will have shape (5, 2, 3), so we\'ll use\n    ``outaxis=-1`` to put the solution vectors in the last dimension.\n\n    >>> x = solve_circulant(c[:, np.newaxis, :], b, baxis=-1, outaxis=-1)\n    >>> x.shape\n    (2, 3, 5)\n    >>> np.set_printoptions(precision=3)  # For compact output of numbers.\n    >>> x\n    array([[[-0.118,  0.22 ,  1.277, -0.142,  0.302],\n            [ 0.651,  0.989,  2.046,  0.627,  1.072],\n            [ 1.42 ,  1.758,  2.816,  1.396,  1.841]],\n           [[ 0.401,  0.304,  0.694, -0.867,  0.377],\n            [ 0.856,  0.758,  1.149, -0.412,  0.831],\n            [ 1.31 ,  1.213,  1.603,  0.042,  1.286]]])\n\n    Check by solving one pair of `c` and `b` vectors (cf. ``x[1, 1, :]``):\n\n    >>> solve_circulant(c[1], b[1, :])\n    array([ 0.856,  0.758,  1.149, -0.412,  0.831])\n\n    ')
    
    # Assigning a Call to a Name (line 859):
    
    # Assigning a Call to a Name (line 859):
    
    # Call to atleast_1d(...): (line 859)
    # Processing the call arguments (line 859)
    # Getting the type of 'c' (line 859)
    c_11951 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 859, 22), 'c', False)
    # Processing the call keyword arguments (line 859)
    kwargs_11952 = {}
    # Getting the type of 'np' (line 859)
    np_11949 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 859, 8), 'np', False)
    # Obtaining the member 'atleast_1d' of a type (line 859)
    atleast_1d_11950 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 859, 8), np_11949, 'atleast_1d')
    # Calling atleast_1d(args, kwargs) (line 859)
    atleast_1d_call_result_11953 = invoke(stypy.reporting.localization.Localization(__file__, 859, 8), atleast_1d_11950, *[c_11951], **kwargs_11952)
    
    # Assigning a type to the variable 'c' (line 859)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 859, 4), 'c', atleast_1d_call_result_11953)
    
    # Assigning a Call to a Name (line 860):
    
    # Assigning a Call to a Name (line 860):
    
    # Call to _get_axis_len(...): (line 860)
    # Processing the call arguments (line 860)
    str_11955 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 860, 23), 'str', 'c')
    # Getting the type of 'c' (line 860)
    c_11956 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 860, 28), 'c', False)
    # Getting the type of 'caxis' (line 860)
    caxis_11957 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 860, 31), 'caxis', False)
    # Processing the call keyword arguments (line 860)
    kwargs_11958 = {}
    # Getting the type of '_get_axis_len' (line 860)
    _get_axis_len_11954 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 860, 9), '_get_axis_len', False)
    # Calling _get_axis_len(args, kwargs) (line 860)
    _get_axis_len_call_result_11959 = invoke(stypy.reporting.localization.Localization(__file__, 860, 9), _get_axis_len_11954, *[str_11955, c_11956, caxis_11957], **kwargs_11958)
    
    # Assigning a type to the variable 'nc' (line 860)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 860, 4), 'nc', _get_axis_len_call_result_11959)
    
    # Assigning a Call to a Name (line 861):
    
    # Assigning a Call to a Name (line 861):
    
    # Call to atleast_1d(...): (line 861)
    # Processing the call arguments (line 861)
    # Getting the type of 'b' (line 861)
    b_11962 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 861, 22), 'b', False)
    # Processing the call keyword arguments (line 861)
    kwargs_11963 = {}
    # Getting the type of 'np' (line 861)
    np_11960 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 861, 8), 'np', False)
    # Obtaining the member 'atleast_1d' of a type (line 861)
    atleast_1d_11961 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 861, 8), np_11960, 'atleast_1d')
    # Calling atleast_1d(args, kwargs) (line 861)
    atleast_1d_call_result_11964 = invoke(stypy.reporting.localization.Localization(__file__, 861, 8), atleast_1d_11961, *[b_11962], **kwargs_11963)
    
    # Assigning a type to the variable 'b' (line 861)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 861, 4), 'b', atleast_1d_call_result_11964)
    
    # Assigning a Call to a Name (line 862):
    
    # Assigning a Call to a Name (line 862):
    
    # Call to _get_axis_len(...): (line 862)
    # Processing the call arguments (line 862)
    str_11966 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 862, 23), 'str', 'b')
    # Getting the type of 'b' (line 862)
    b_11967 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 862, 28), 'b', False)
    # Getting the type of 'baxis' (line 862)
    baxis_11968 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 862, 31), 'baxis', False)
    # Processing the call keyword arguments (line 862)
    kwargs_11969 = {}
    # Getting the type of '_get_axis_len' (line 862)
    _get_axis_len_11965 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 862, 9), '_get_axis_len', False)
    # Calling _get_axis_len(args, kwargs) (line 862)
    _get_axis_len_call_result_11970 = invoke(stypy.reporting.localization.Localization(__file__, 862, 9), _get_axis_len_11965, *[str_11966, b_11967, baxis_11968], **kwargs_11969)
    
    # Assigning a type to the variable 'nb' (line 862)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 862, 4), 'nb', _get_axis_len_call_result_11970)
    
    
    # Getting the type of 'nc' (line 863)
    nc_11971 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 863, 7), 'nc')
    # Getting the type of 'nb' (line 863)
    nb_11972 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 863, 13), 'nb')
    # Applying the binary operator '!=' (line 863)
    result_ne_11973 = python_operator(stypy.reporting.localization.Localization(__file__, 863, 7), '!=', nc_11971, nb_11972)
    
    # Testing the type of an if condition (line 863)
    if_condition_11974 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 863, 4), result_ne_11973)
    # Assigning a type to the variable 'if_condition_11974' (line 863)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 863, 4), 'if_condition_11974', if_condition_11974)
    # SSA begins for if statement (line 863)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 864)
    # Processing the call arguments (line 864)
    str_11976 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 864, 25), 'str', 'Incompatible c and b axis lengths')
    # Processing the call keyword arguments (line 864)
    kwargs_11977 = {}
    # Getting the type of 'ValueError' (line 864)
    ValueError_11975 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 864, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 864)
    ValueError_call_result_11978 = invoke(stypy.reporting.localization.Localization(__file__, 864, 14), ValueError_11975, *[str_11976], **kwargs_11977)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 864, 8), ValueError_call_result_11978, 'raise parameter', BaseException)
    # SSA join for if statement (line 863)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 866):
    
    # Assigning a Call to a Name (line 866):
    
    # Call to fft(...): (line 866)
    # Processing the call arguments (line 866)
    
    # Call to rollaxis(...): (line 866)
    # Processing the call arguments (line 866)
    # Getting the type of 'c' (line 866)
    c_11984 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 866, 32), 'c', False)
    # Getting the type of 'caxis' (line 866)
    caxis_11985 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 866, 35), 'caxis', False)
    # Getting the type of 'c' (line 866)
    c_11986 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 866, 42), 'c', False)
    # Obtaining the member 'ndim' of a type (line 866)
    ndim_11987 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 866, 42), c_11986, 'ndim')
    # Processing the call keyword arguments (line 866)
    kwargs_11988 = {}
    # Getting the type of 'np' (line 866)
    np_11982 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 866, 20), 'np', False)
    # Obtaining the member 'rollaxis' of a type (line 866)
    rollaxis_11983 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 866, 20), np_11982, 'rollaxis')
    # Calling rollaxis(args, kwargs) (line 866)
    rollaxis_call_result_11989 = invoke(stypy.reporting.localization.Localization(__file__, 866, 20), rollaxis_11983, *[c_11984, caxis_11985, ndim_11987], **kwargs_11988)
    
    # Processing the call keyword arguments (line 866)
    int_11990 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 866, 56), 'int')
    keyword_11991 = int_11990
    kwargs_11992 = {'axis': keyword_11991}
    # Getting the type of 'np' (line 866)
    np_11979 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 866, 9), 'np', False)
    # Obtaining the member 'fft' of a type (line 866)
    fft_11980 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 866, 9), np_11979, 'fft')
    # Obtaining the member 'fft' of a type (line 866)
    fft_11981 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 866, 9), fft_11980, 'fft')
    # Calling fft(args, kwargs) (line 866)
    fft_call_result_11993 = invoke(stypy.reporting.localization.Localization(__file__, 866, 9), fft_11981, *[rollaxis_call_result_11989], **kwargs_11992)
    
    # Assigning a type to the variable 'fc' (line 866)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 866, 4), 'fc', fft_call_result_11993)
    
    # Assigning a Call to a Name (line 867):
    
    # Assigning a Call to a Name (line 867):
    
    # Call to abs(...): (line 867)
    # Processing the call arguments (line 867)
    # Getting the type of 'fc' (line 867)
    fc_11996 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 867, 20), 'fc', False)
    # Processing the call keyword arguments (line 867)
    kwargs_11997 = {}
    # Getting the type of 'np' (line 867)
    np_11994 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 867, 13), 'np', False)
    # Obtaining the member 'abs' of a type (line 867)
    abs_11995 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 867, 13), np_11994, 'abs')
    # Calling abs(args, kwargs) (line 867)
    abs_call_result_11998 = invoke(stypy.reporting.localization.Localization(__file__, 867, 13), abs_11995, *[fc_11996], **kwargs_11997)
    
    # Assigning a type to the variable 'abs_fc' (line 867)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 867, 4), 'abs_fc', abs_call_result_11998)
    
    # Type idiom detected: calculating its left and rigth part (line 868)
    # Getting the type of 'tol' (line 868)
    tol_11999 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 868, 7), 'tol')
    # Getting the type of 'None' (line 868)
    None_12000 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 868, 14), 'None')
    
    (may_be_12001, more_types_in_union_12002) = may_be_none(tol_11999, None_12000)

    if may_be_12001:

        if more_types_in_union_12002:
            # Runtime conditional SSA (line 868)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a BinOp to a Name (line 870):
        
        # Assigning a BinOp to a Name (line 870):
        
        # Call to max(...): (line 870)
        # Processing the call keyword arguments (line 870)
        int_12005 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 870, 30), 'int')
        keyword_12006 = int_12005
        kwargs_12007 = {'axis': keyword_12006}
        # Getting the type of 'abs_fc' (line 870)
        abs_fc_12003 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 870, 14), 'abs_fc', False)
        # Obtaining the member 'max' of a type (line 870)
        max_12004 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 870, 14), abs_fc_12003, 'max')
        # Calling max(args, kwargs) (line 870)
        max_call_result_12008 = invoke(stypy.reporting.localization.Localization(__file__, 870, 14), max_12004, *[], **kwargs_12007)
        
        # Getting the type of 'nc' (line 870)
        nc_12009 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 870, 36), 'nc')
        # Applying the binary operator '*' (line 870)
        result_mul_12010 = python_operator(stypy.reporting.localization.Localization(__file__, 870, 14), '*', max_call_result_12008, nc_12009)
        
        
        # Call to finfo(...): (line 870)
        # Processing the call arguments (line 870)
        # Getting the type of 'np' (line 870)
        np_12013 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 870, 50), 'np', False)
        # Obtaining the member 'float64' of a type (line 870)
        float64_12014 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 870, 50), np_12013, 'float64')
        # Processing the call keyword arguments (line 870)
        kwargs_12015 = {}
        # Getting the type of 'np' (line 870)
        np_12011 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 870, 41), 'np', False)
        # Obtaining the member 'finfo' of a type (line 870)
        finfo_12012 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 870, 41), np_12011, 'finfo')
        # Calling finfo(args, kwargs) (line 870)
        finfo_call_result_12016 = invoke(stypy.reporting.localization.Localization(__file__, 870, 41), finfo_12012, *[float64_12014], **kwargs_12015)
        
        # Obtaining the member 'eps' of a type (line 870)
        eps_12017 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 870, 41), finfo_call_result_12016, 'eps')
        # Applying the binary operator '*' (line 870)
        result_mul_12018 = python_operator(stypy.reporting.localization.Localization(__file__, 870, 39), '*', result_mul_12010, eps_12017)
        
        # Assigning a type to the variable 'tol' (line 870)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 870, 8), 'tol', result_mul_12018)
        
        
        # Getting the type of 'tol' (line 871)
        tol_12019 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 871, 11), 'tol')
        # Obtaining the member 'shape' of a type (line 871)
        shape_12020 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 871, 11), tol_12019, 'shape')
        
        # Obtaining an instance of the builtin type 'tuple' (line 871)
        tuple_12021 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 871, 24), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 871)
        
        # Applying the binary operator '!=' (line 871)
        result_ne_12022 = python_operator(stypy.reporting.localization.Localization(__file__, 871, 11), '!=', shape_12020, tuple_12021)
        
        # Testing the type of an if condition (line 871)
        if_condition_12023 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 871, 8), result_ne_12022)
        # Assigning a type to the variable 'if_condition_12023' (line 871)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 871, 8), 'if_condition_12023', if_condition_12023)
        # SSA begins for if statement (line 871)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a BinOp to a Attribute (line 872):
        
        # Assigning a BinOp to a Attribute (line 872):
        # Getting the type of 'tol' (line 872)
        tol_12024 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 872, 24), 'tol')
        # Obtaining the member 'shape' of a type (line 872)
        shape_12025 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 872, 24), tol_12024, 'shape')
        
        # Obtaining an instance of the builtin type 'tuple' (line 872)
        tuple_12026 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 872, 37), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 872)
        # Adding element type (line 872)
        int_12027 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 872, 37), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 872, 37), tuple_12026, int_12027)
        
        # Applying the binary operator '+' (line 872)
        result_add_12028 = python_operator(stypy.reporting.localization.Localization(__file__, 872, 24), '+', shape_12025, tuple_12026)
        
        # Getting the type of 'tol' (line 872)
        tol_12029 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 872, 12), 'tol')
        # Setting the type of the member 'shape' of a type (line 872)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 872, 12), tol_12029, 'shape', result_add_12028)
        # SSA branch for the else part of an if statement (line 871)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Call to a Name (line 874):
        
        # Assigning a Call to a Name (line 874):
        
        # Call to atleast_1d(...): (line 874)
        # Processing the call arguments (line 874)
        # Getting the type of 'tol' (line 874)
        tol_12032 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 874, 32), 'tol', False)
        # Processing the call keyword arguments (line 874)
        kwargs_12033 = {}
        # Getting the type of 'np' (line 874)
        np_12030 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 874, 18), 'np', False)
        # Obtaining the member 'atleast_1d' of a type (line 874)
        atleast_1d_12031 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 874, 18), np_12030, 'atleast_1d')
        # Calling atleast_1d(args, kwargs) (line 874)
        atleast_1d_call_result_12034 = invoke(stypy.reporting.localization.Localization(__file__, 874, 18), atleast_1d_12031, *[tol_12032], **kwargs_12033)
        
        # Assigning a type to the variable 'tol' (line 874)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 874, 12), 'tol', atleast_1d_call_result_12034)
        # SSA join for if statement (line 871)
        module_type_store = module_type_store.join_ssa_context()
        

        if more_types_in_union_12002:
            # SSA join for if statement (line 868)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Assigning a Compare to a Name (line 876):
    
    # Assigning a Compare to a Name (line 876):
    
    # Getting the type of 'abs_fc' (line 876)
    abs_fc_12035 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 876, 17), 'abs_fc')
    # Getting the type of 'tol' (line 876)
    tol_12036 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 876, 27), 'tol')
    # Applying the binary operator '<=' (line 876)
    result_le_12037 = python_operator(stypy.reporting.localization.Localization(__file__, 876, 17), '<=', abs_fc_12035, tol_12036)
    
    # Assigning a type to the variable 'near_zeros' (line 876)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 876, 4), 'near_zeros', result_le_12037)
    
    # Assigning a Call to a Name (line 877):
    
    # Assigning a Call to a Name (line 877):
    
    # Call to any(...): (line 877)
    # Processing the call arguments (line 877)
    # Getting the type of 'near_zeros' (line 877)
    near_zeros_12040 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 877, 30), 'near_zeros', False)
    # Processing the call keyword arguments (line 877)
    kwargs_12041 = {}
    # Getting the type of 'np' (line 877)
    np_12038 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 877, 23), 'np', False)
    # Obtaining the member 'any' of a type (line 877)
    any_12039 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 877, 23), np_12038, 'any')
    # Calling any(args, kwargs) (line 877)
    any_call_result_12042 = invoke(stypy.reporting.localization.Localization(__file__, 877, 23), any_12039, *[near_zeros_12040], **kwargs_12041)
    
    # Assigning a type to the variable 'is_near_singular' (line 877)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 877, 4), 'is_near_singular', any_call_result_12042)
    
    # Getting the type of 'is_near_singular' (line 878)
    is_near_singular_12043 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 878, 7), 'is_near_singular')
    # Testing the type of an if condition (line 878)
    if_condition_12044 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 878, 4), is_near_singular_12043)
    # Assigning a type to the variable 'if_condition_12044' (line 878)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 878, 4), 'if_condition_12044', if_condition_12044)
    # SSA begins for if statement (line 878)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    
    # Getting the type of 'singular' (line 879)
    singular_12045 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 879, 11), 'singular')
    str_12046 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 879, 23), 'str', 'raise')
    # Applying the binary operator '==' (line 879)
    result_eq_12047 = python_operator(stypy.reporting.localization.Localization(__file__, 879, 11), '==', singular_12045, str_12046)
    
    # Testing the type of an if condition (line 879)
    if_condition_12048 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 879, 8), result_eq_12047)
    # Assigning a type to the variable 'if_condition_12048' (line 879)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 879, 8), 'if_condition_12048', if_condition_12048)
    # SSA begins for if statement (line 879)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to LinAlgError(...): (line 880)
    # Processing the call arguments (line 880)
    str_12050 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 880, 30), 'str', 'near singular circulant matrix.')
    # Processing the call keyword arguments (line 880)
    kwargs_12051 = {}
    # Getting the type of 'LinAlgError' (line 880)
    LinAlgError_12049 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 880, 18), 'LinAlgError', False)
    # Calling LinAlgError(args, kwargs) (line 880)
    LinAlgError_call_result_12052 = invoke(stypy.reporting.localization.Localization(__file__, 880, 18), LinAlgError_12049, *[str_12050], **kwargs_12051)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 880, 12), LinAlgError_call_result_12052, 'raise parameter', BaseException)
    # SSA branch for the else part of an if statement (line 879)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Num to a Subscript (line 884):
    
    # Assigning a Num to a Subscript (line 884):
    int_12053 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 884, 29), 'int')
    # Getting the type of 'fc' (line 884)
    fc_12054 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 884, 12), 'fc')
    # Getting the type of 'near_zeros' (line 884)
    near_zeros_12055 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 884, 15), 'near_zeros')
    # Storing an element on a container (line 884)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 884, 12), fc_12054, (near_zeros_12055, int_12053))
    # SSA join for if statement (line 879)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 878)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 886):
    
    # Assigning a Call to a Name (line 886):
    
    # Call to fft(...): (line 886)
    # Processing the call arguments (line 886)
    
    # Call to rollaxis(...): (line 886)
    # Processing the call arguments (line 886)
    # Getting the type of 'b' (line 886)
    b_12061 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 886, 32), 'b', False)
    # Getting the type of 'baxis' (line 886)
    baxis_12062 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 886, 35), 'baxis', False)
    # Getting the type of 'b' (line 886)
    b_12063 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 886, 42), 'b', False)
    # Obtaining the member 'ndim' of a type (line 886)
    ndim_12064 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 886, 42), b_12063, 'ndim')
    # Processing the call keyword arguments (line 886)
    kwargs_12065 = {}
    # Getting the type of 'np' (line 886)
    np_12059 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 886, 20), 'np', False)
    # Obtaining the member 'rollaxis' of a type (line 886)
    rollaxis_12060 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 886, 20), np_12059, 'rollaxis')
    # Calling rollaxis(args, kwargs) (line 886)
    rollaxis_call_result_12066 = invoke(stypy.reporting.localization.Localization(__file__, 886, 20), rollaxis_12060, *[b_12061, baxis_12062, ndim_12064], **kwargs_12065)
    
    # Processing the call keyword arguments (line 886)
    int_12067 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 886, 56), 'int')
    keyword_12068 = int_12067
    kwargs_12069 = {'axis': keyword_12068}
    # Getting the type of 'np' (line 886)
    np_12056 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 886, 9), 'np', False)
    # Obtaining the member 'fft' of a type (line 886)
    fft_12057 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 886, 9), np_12056, 'fft')
    # Obtaining the member 'fft' of a type (line 886)
    fft_12058 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 886, 9), fft_12057, 'fft')
    # Calling fft(args, kwargs) (line 886)
    fft_call_result_12070 = invoke(stypy.reporting.localization.Localization(__file__, 886, 9), fft_12058, *[rollaxis_call_result_12066], **kwargs_12069)
    
    # Assigning a type to the variable 'fb' (line 886)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 886, 4), 'fb', fft_call_result_12070)
    
    # Assigning a BinOp to a Name (line 888):
    
    # Assigning a BinOp to a Name (line 888):
    # Getting the type of 'fb' (line 888)
    fb_12071 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 888, 8), 'fb')
    # Getting the type of 'fc' (line 888)
    fc_12072 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 888, 13), 'fc')
    # Applying the binary operator 'div' (line 888)
    result_div_12073 = python_operator(stypy.reporting.localization.Localization(__file__, 888, 8), 'div', fb_12071, fc_12072)
    
    # Assigning a type to the variable 'q' (line 888)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 888, 4), 'q', result_div_12073)
    
    # Getting the type of 'is_near_singular' (line 890)
    is_near_singular_12074 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 890, 7), 'is_near_singular')
    # Testing the type of an if condition (line 890)
    if_condition_12075 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 890, 4), is_near_singular_12074)
    # Assigning a type to the variable 'if_condition_12075' (line 890)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 890, 4), 'if_condition_12075', if_condition_12075)
    # SSA begins for if statement (line 890)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a BinOp to a Name (line 896):
    
    # Assigning a BinOp to a Name (line 896):
    
    # Call to ones_like(...): (line 896)
    # Processing the call arguments (line 896)
    # Getting the type of 'b' (line 896)
    b_12078 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 896, 28), 'b', False)
    # Processing the call keyword arguments (line 896)
    # Getting the type of 'bool' (line 896)
    bool_12079 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 896, 37), 'bool', False)
    keyword_12080 = bool_12079
    kwargs_12081 = {'dtype': keyword_12080}
    # Getting the type of 'np' (line 896)
    np_12076 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 896, 15), 'np', False)
    # Obtaining the member 'ones_like' of a type (line 896)
    ones_like_12077 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 896, 15), np_12076, 'ones_like')
    # Calling ones_like(args, kwargs) (line 896)
    ones_like_call_result_12082 = invoke(stypy.reporting.localization.Localization(__file__, 896, 15), ones_like_12077, *[b_12078], **kwargs_12081)
    
    # Getting the type of 'near_zeros' (line 896)
    near_zeros_12083 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 896, 45), 'near_zeros')
    # Applying the binary operator '&' (line 896)
    result_and__12084 = python_operator(stypy.reporting.localization.Localization(__file__, 896, 15), '&', ones_like_call_result_12082, near_zeros_12083)
    
    # Assigning a type to the variable 'mask' (line 896)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 896, 8), 'mask', result_and__12084)
    
    # Assigning a Num to a Subscript (line 897):
    
    # Assigning a Num to a Subscript (line 897):
    int_12085 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 897, 18), 'int')
    # Getting the type of 'q' (line 897)
    q_12086 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 897, 8), 'q')
    # Getting the type of 'mask' (line 897)
    mask_12087 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 897, 10), 'mask')
    # Storing an element on a container (line 897)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 897, 8), q_12086, (mask_12087, int_12085))
    # SSA join for if statement (line 890)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 899):
    
    # Assigning a Call to a Name (line 899):
    
    # Call to ifft(...): (line 899)
    # Processing the call arguments (line 899)
    # Getting the type of 'q' (line 899)
    q_12091 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 899, 20), 'q', False)
    # Processing the call keyword arguments (line 899)
    int_12092 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 899, 28), 'int')
    keyword_12093 = int_12092
    kwargs_12094 = {'axis': keyword_12093}
    # Getting the type of 'np' (line 899)
    np_12088 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 899, 8), 'np', False)
    # Obtaining the member 'fft' of a type (line 899)
    fft_12089 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 899, 8), np_12088, 'fft')
    # Obtaining the member 'ifft' of a type (line 899)
    ifft_12090 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 899, 8), fft_12089, 'ifft')
    # Calling ifft(args, kwargs) (line 899)
    ifft_call_result_12095 = invoke(stypy.reporting.localization.Localization(__file__, 899, 8), ifft_12090, *[q_12091], **kwargs_12094)
    
    # Assigning a type to the variable 'x' (line 899)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 899, 4), 'x', ifft_call_result_12095)
    
    
    
    # Evaluating a boolean operation
    
    # Call to iscomplexobj(...): (line 900)
    # Processing the call arguments (line 900)
    # Getting the type of 'c' (line 900)
    c_12098 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 900, 28), 'c', False)
    # Processing the call keyword arguments (line 900)
    kwargs_12099 = {}
    # Getting the type of 'np' (line 900)
    np_12096 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 900, 12), 'np', False)
    # Obtaining the member 'iscomplexobj' of a type (line 900)
    iscomplexobj_12097 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 900, 12), np_12096, 'iscomplexobj')
    # Calling iscomplexobj(args, kwargs) (line 900)
    iscomplexobj_call_result_12100 = invoke(stypy.reporting.localization.Localization(__file__, 900, 12), iscomplexobj_12097, *[c_12098], **kwargs_12099)
    
    
    # Call to iscomplexobj(...): (line 900)
    # Processing the call arguments (line 900)
    # Getting the type of 'b' (line 900)
    b_12103 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 900, 50), 'b', False)
    # Processing the call keyword arguments (line 900)
    kwargs_12104 = {}
    # Getting the type of 'np' (line 900)
    np_12101 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 900, 34), 'np', False)
    # Obtaining the member 'iscomplexobj' of a type (line 900)
    iscomplexobj_12102 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 900, 34), np_12101, 'iscomplexobj')
    # Calling iscomplexobj(args, kwargs) (line 900)
    iscomplexobj_call_result_12105 = invoke(stypy.reporting.localization.Localization(__file__, 900, 34), iscomplexobj_12102, *[b_12103], **kwargs_12104)
    
    # Applying the binary operator 'or' (line 900)
    result_or_keyword_12106 = python_operator(stypy.reporting.localization.Localization(__file__, 900, 12), 'or', iscomplexobj_call_result_12100, iscomplexobj_call_result_12105)
    
    # Applying the 'not' unary operator (line 900)
    result_not__12107 = python_operator(stypy.reporting.localization.Localization(__file__, 900, 7), 'not', result_or_keyword_12106)
    
    # Testing the type of an if condition (line 900)
    if_condition_12108 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 900, 4), result_not__12107)
    # Assigning a type to the variable 'if_condition_12108' (line 900)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 900, 4), 'if_condition_12108', if_condition_12108)
    # SSA begins for if statement (line 900)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Attribute to a Name (line 901):
    
    # Assigning a Attribute to a Name (line 901):
    # Getting the type of 'x' (line 901)
    x_12109 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 901, 12), 'x')
    # Obtaining the member 'real' of a type (line 901)
    real_12110 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 901, 12), x_12109, 'real')
    # Assigning a type to the variable 'x' (line 901)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 901, 8), 'x', real_12110)
    # SSA join for if statement (line 900)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'outaxis' (line 902)
    outaxis_12111 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 902, 7), 'outaxis')
    int_12112 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 902, 18), 'int')
    # Applying the binary operator '!=' (line 902)
    result_ne_12113 = python_operator(stypy.reporting.localization.Localization(__file__, 902, 7), '!=', outaxis_12111, int_12112)
    
    # Testing the type of an if condition (line 902)
    if_condition_12114 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 902, 4), result_ne_12113)
    # Assigning a type to the variable 'if_condition_12114' (line 902)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 902, 4), 'if_condition_12114', if_condition_12114)
    # SSA begins for if statement (line 902)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 903):
    
    # Assigning a Call to a Name (line 903):
    
    # Call to rollaxis(...): (line 903)
    # Processing the call arguments (line 903)
    # Getting the type of 'x' (line 903)
    x_12117 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 903, 24), 'x', False)
    int_12118 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 903, 27), 'int')
    # Getting the type of 'outaxis' (line 903)
    outaxis_12119 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 903, 31), 'outaxis', False)
    # Processing the call keyword arguments (line 903)
    kwargs_12120 = {}
    # Getting the type of 'np' (line 903)
    np_12115 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 903, 12), 'np', False)
    # Obtaining the member 'rollaxis' of a type (line 903)
    rollaxis_12116 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 903, 12), np_12115, 'rollaxis')
    # Calling rollaxis(args, kwargs) (line 903)
    rollaxis_call_result_12121 = invoke(stypy.reporting.localization.Localization(__file__, 903, 12), rollaxis_12116, *[x_12117, int_12118, outaxis_12119], **kwargs_12120)
    
    # Assigning a type to the variable 'x' (line 903)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 903, 8), 'x', rollaxis_call_result_12121)
    # SSA join for if statement (line 902)
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'x' (line 904)
    x_12122 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 904, 11), 'x')
    # Assigning a type to the variable 'stypy_return_type' (line 904)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 904, 4), 'stypy_return_type', x_12122)
    
    # ################# End of 'solve_circulant(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'solve_circulant' in the type store
    # Getting the type of 'stypy_return_type' (line 711)
    stypy_return_type_12123 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 711, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_12123)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'solve_circulant'
    return stypy_return_type_12123

# Assigning a type to the variable 'solve_circulant' (line 711)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 711, 0), 'solve_circulant', solve_circulant)

@norecursion
def inv(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'False' (line 908)
    False_12124 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 908, 23), 'False')
    # Getting the type of 'True' (line 908)
    True_12125 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 908, 43), 'True')
    defaults = [False_12124, True_12125]
    # Create a new context for function 'inv'
    module_type_store = module_type_store.open_function_context('inv', 908, 0, False)
    
    # Passed parameters checking function
    inv.stypy_localization = localization
    inv.stypy_type_of_self = None
    inv.stypy_type_store = module_type_store
    inv.stypy_function_name = 'inv'
    inv.stypy_param_names_list = ['a', 'overwrite_a', 'check_finite']
    inv.stypy_varargs_param_name = None
    inv.stypy_kwargs_param_name = None
    inv.stypy_call_defaults = defaults
    inv.stypy_call_varargs = varargs
    inv.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'inv', ['a', 'overwrite_a', 'check_finite'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'inv', localization, ['a', 'overwrite_a', 'check_finite'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'inv(...)' code ##################

    str_12126 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 946, (-1)), 'str', '\n    Compute the inverse of a matrix.\n\n    Parameters\n    ----------\n    a : array_like\n        Square matrix to be inverted.\n    overwrite_a : bool, optional\n        Discard data in `a` (may improve performance). Default is False.\n    check_finite : bool, optional\n        Whether to check that the input matrix contains only finite numbers.\n        Disabling may give a performance gain, but may result in problems\n        (crashes, non-termination) if the inputs do contain infinities or NaNs.\n\n    Returns\n    -------\n    ainv : ndarray\n        Inverse of the matrix `a`.\n\n    Raises\n    ------\n    LinAlgError\n        If `a` is singular.\n    ValueError\n        If `a` is not square, or not 2-dimensional.\n\n    Examples\n    --------\n    >>> from scipy import linalg\n    >>> a = np.array([[1., 2.], [3., 4.]])\n    >>> linalg.inv(a)\n    array([[-2. ,  1. ],\n           [ 1.5, -0.5]])\n    >>> np.dot(a, linalg.inv(a))\n    array([[ 1.,  0.],\n           [ 0.,  1.]])\n\n    ')
    
    # Assigning a Call to a Name (line 947):
    
    # Assigning a Call to a Name (line 947):
    
    # Call to _asarray_validated(...): (line 947)
    # Processing the call arguments (line 947)
    # Getting the type of 'a' (line 947)
    a_12128 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 947, 28), 'a', False)
    # Processing the call keyword arguments (line 947)
    # Getting the type of 'check_finite' (line 947)
    check_finite_12129 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 947, 44), 'check_finite', False)
    keyword_12130 = check_finite_12129
    kwargs_12131 = {'check_finite': keyword_12130}
    # Getting the type of '_asarray_validated' (line 947)
    _asarray_validated_12127 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 947, 9), '_asarray_validated', False)
    # Calling _asarray_validated(args, kwargs) (line 947)
    _asarray_validated_call_result_12132 = invoke(stypy.reporting.localization.Localization(__file__, 947, 9), _asarray_validated_12127, *[a_12128], **kwargs_12131)
    
    # Assigning a type to the variable 'a1' (line 947)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 947, 4), 'a1', _asarray_validated_call_result_12132)
    
    
    # Evaluating a boolean operation
    
    
    # Call to len(...): (line 948)
    # Processing the call arguments (line 948)
    # Getting the type of 'a1' (line 948)
    a1_12134 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 948, 11), 'a1', False)
    # Obtaining the member 'shape' of a type (line 948)
    shape_12135 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 948, 11), a1_12134, 'shape')
    # Processing the call keyword arguments (line 948)
    kwargs_12136 = {}
    # Getting the type of 'len' (line 948)
    len_12133 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 948, 7), 'len', False)
    # Calling len(args, kwargs) (line 948)
    len_call_result_12137 = invoke(stypy.reporting.localization.Localization(__file__, 948, 7), len_12133, *[shape_12135], **kwargs_12136)
    
    int_12138 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 948, 24), 'int')
    # Applying the binary operator '!=' (line 948)
    result_ne_12139 = python_operator(stypy.reporting.localization.Localization(__file__, 948, 7), '!=', len_call_result_12137, int_12138)
    
    
    
    # Obtaining the type of the subscript
    int_12140 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 948, 38), 'int')
    # Getting the type of 'a1' (line 948)
    a1_12141 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 948, 29), 'a1')
    # Obtaining the member 'shape' of a type (line 948)
    shape_12142 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 948, 29), a1_12141, 'shape')
    # Obtaining the member '__getitem__' of a type (line 948)
    getitem___12143 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 948, 29), shape_12142, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 948)
    subscript_call_result_12144 = invoke(stypy.reporting.localization.Localization(__file__, 948, 29), getitem___12143, int_12140)
    
    
    # Obtaining the type of the subscript
    int_12145 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 948, 53), 'int')
    # Getting the type of 'a1' (line 948)
    a1_12146 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 948, 44), 'a1')
    # Obtaining the member 'shape' of a type (line 948)
    shape_12147 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 948, 44), a1_12146, 'shape')
    # Obtaining the member '__getitem__' of a type (line 948)
    getitem___12148 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 948, 44), shape_12147, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 948)
    subscript_call_result_12149 = invoke(stypy.reporting.localization.Localization(__file__, 948, 44), getitem___12148, int_12145)
    
    # Applying the binary operator '!=' (line 948)
    result_ne_12150 = python_operator(stypy.reporting.localization.Localization(__file__, 948, 29), '!=', subscript_call_result_12144, subscript_call_result_12149)
    
    # Applying the binary operator 'or' (line 948)
    result_or_keyword_12151 = python_operator(stypy.reporting.localization.Localization(__file__, 948, 7), 'or', result_ne_12139, result_ne_12150)
    
    # Testing the type of an if condition (line 948)
    if_condition_12152 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 948, 4), result_or_keyword_12151)
    # Assigning a type to the variable 'if_condition_12152' (line 948)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 948, 4), 'if_condition_12152', if_condition_12152)
    # SSA begins for if statement (line 948)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 949)
    # Processing the call arguments (line 949)
    str_12154 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 949, 25), 'str', 'expected square matrix')
    # Processing the call keyword arguments (line 949)
    kwargs_12155 = {}
    # Getting the type of 'ValueError' (line 949)
    ValueError_12153 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 949, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 949)
    ValueError_call_result_12156 = invoke(stypy.reporting.localization.Localization(__file__, 949, 14), ValueError_12153, *[str_12154], **kwargs_12155)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 949, 8), ValueError_call_result_12156, 'raise parameter', BaseException)
    # SSA join for if statement (line 948)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a BoolOp to a Name (line 950):
    
    # Assigning a BoolOp to a Name (line 950):
    
    # Evaluating a boolean operation
    # Getting the type of 'overwrite_a' (line 950)
    overwrite_a_12157 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 950, 18), 'overwrite_a')
    
    # Call to _datacopied(...): (line 950)
    # Processing the call arguments (line 950)
    # Getting the type of 'a1' (line 950)
    a1_12159 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 950, 45), 'a1', False)
    # Getting the type of 'a' (line 950)
    a_12160 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 950, 49), 'a', False)
    # Processing the call keyword arguments (line 950)
    kwargs_12161 = {}
    # Getting the type of '_datacopied' (line 950)
    _datacopied_12158 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 950, 33), '_datacopied', False)
    # Calling _datacopied(args, kwargs) (line 950)
    _datacopied_call_result_12162 = invoke(stypy.reporting.localization.Localization(__file__, 950, 33), _datacopied_12158, *[a1_12159, a_12160], **kwargs_12161)
    
    # Applying the binary operator 'or' (line 950)
    result_or_keyword_12163 = python_operator(stypy.reporting.localization.Localization(__file__, 950, 18), 'or', overwrite_a_12157, _datacopied_call_result_12162)
    
    # Assigning a type to the variable 'overwrite_a' (line 950)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 950, 4), 'overwrite_a', result_or_keyword_12163)
    
    # Assigning a Call to a Tuple (line 960):
    
    # Assigning a Subscript to a Name (line 960):
    
    # Obtaining the type of the subscript
    int_12164 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 960, 4), 'int')
    
    # Call to get_lapack_funcs(...): (line 960)
    # Processing the call arguments (line 960)
    
    # Obtaining an instance of the builtin type 'tuple' (line 960)
    tuple_12166 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 960, 50), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 960)
    # Adding element type (line 960)
    str_12167 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 960, 50), 'str', 'getrf')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 960, 50), tuple_12166, str_12167)
    # Adding element type (line 960)
    str_12168 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 960, 59), 'str', 'getri')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 960, 50), tuple_12166, str_12168)
    # Adding element type (line 960)
    str_12169 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 961, 50), 'str', 'getri_lwork')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 960, 50), tuple_12166, str_12169)
    
    
    # Obtaining an instance of the builtin type 'tuple' (line 962)
    tuple_12170 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 962, 50), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 962)
    # Adding element type (line 962)
    # Getting the type of 'a1' (line 962)
    a1_12171 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 962, 50), 'a1', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 962, 50), tuple_12170, a1_12171)
    
    # Processing the call keyword arguments (line 960)
    kwargs_12172 = {}
    # Getting the type of 'get_lapack_funcs' (line 960)
    get_lapack_funcs_12165 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 960, 32), 'get_lapack_funcs', False)
    # Calling get_lapack_funcs(args, kwargs) (line 960)
    get_lapack_funcs_call_result_12173 = invoke(stypy.reporting.localization.Localization(__file__, 960, 32), get_lapack_funcs_12165, *[tuple_12166, tuple_12170], **kwargs_12172)
    
    # Obtaining the member '__getitem__' of a type (line 960)
    getitem___12174 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 960, 4), get_lapack_funcs_call_result_12173, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 960)
    subscript_call_result_12175 = invoke(stypy.reporting.localization.Localization(__file__, 960, 4), getitem___12174, int_12164)
    
    # Assigning a type to the variable 'tuple_var_assignment_10088' (line 960)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 960, 4), 'tuple_var_assignment_10088', subscript_call_result_12175)
    
    # Assigning a Subscript to a Name (line 960):
    
    # Obtaining the type of the subscript
    int_12176 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 960, 4), 'int')
    
    # Call to get_lapack_funcs(...): (line 960)
    # Processing the call arguments (line 960)
    
    # Obtaining an instance of the builtin type 'tuple' (line 960)
    tuple_12178 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 960, 50), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 960)
    # Adding element type (line 960)
    str_12179 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 960, 50), 'str', 'getrf')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 960, 50), tuple_12178, str_12179)
    # Adding element type (line 960)
    str_12180 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 960, 59), 'str', 'getri')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 960, 50), tuple_12178, str_12180)
    # Adding element type (line 960)
    str_12181 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 961, 50), 'str', 'getri_lwork')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 960, 50), tuple_12178, str_12181)
    
    
    # Obtaining an instance of the builtin type 'tuple' (line 962)
    tuple_12182 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 962, 50), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 962)
    # Adding element type (line 962)
    # Getting the type of 'a1' (line 962)
    a1_12183 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 962, 50), 'a1', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 962, 50), tuple_12182, a1_12183)
    
    # Processing the call keyword arguments (line 960)
    kwargs_12184 = {}
    # Getting the type of 'get_lapack_funcs' (line 960)
    get_lapack_funcs_12177 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 960, 32), 'get_lapack_funcs', False)
    # Calling get_lapack_funcs(args, kwargs) (line 960)
    get_lapack_funcs_call_result_12185 = invoke(stypy.reporting.localization.Localization(__file__, 960, 32), get_lapack_funcs_12177, *[tuple_12178, tuple_12182], **kwargs_12184)
    
    # Obtaining the member '__getitem__' of a type (line 960)
    getitem___12186 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 960, 4), get_lapack_funcs_call_result_12185, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 960)
    subscript_call_result_12187 = invoke(stypy.reporting.localization.Localization(__file__, 960, 4), getitem___12186, int_12176)
    
    # Assigning a type to the variable 'tuple_var_assignment_10089' (line 960)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 960, 4), 'tuple_var_assignment_10089', subscript_call_result_12187)
    
    # Assigning a Subscript to a Name (line 960):
    
    # Obtaining the type of the subscript
    int_12188 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 960, 4), 'int')
    
    # Call to get_lapack_funcs(...): (line 960)
    # Processing the call arguments (line 960)
    
    # Obtaining an instance of the builtin type 'tuple' (line 960)
    tuple_12190 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 960, 50), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 960)
    # Adding element type (line 960)
    str_12191 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 960, 50), 'str', 'getrf')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 960, 50), tuple_12190, str_12191)
    # Adding element type (line 960)
    str_12192 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 960, 59), 'str', 'getri')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 960, 50), tuple_12190, str_12192)
    # Adding element type (line 960)
    str_12193 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 961, 50), 'str', 'getri_lwork')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 960, 50), tuple_12190, str_12193)
    
    
    # Obtaining an instance of the builtin type 'tuple' (line 962)
    tuple_12194 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 962, 50), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 962)
    # Adding element type (line 962)
    # Getting the type of 'a1' (line 962)
    a1_12195 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 962, 50), 'a1', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 962, 50), tuple_12194, a1_12195)
    
    # Processing the call keyword arguments (line 960)
    kwargs_12196 = {}
    # Getting the type of 'get_lapack_funcs' (line 960)
    get_lapack_funcs_12189 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 960, 32), 'get_lapack_funcs', False)
    # Calling get_lapack_funcs(args, kwargs) (line 960)
    get_lapack_funcs_call_result_12197 = invoke(stypy.reporting.localization.Localization(__file__, 960, 32), get_lapack_funcs_12189, *[tuple_12190, tuple_12194], **kwargs_12196)
    
    # Obtaining the member '__getitem__' of a type (line 960)
    getitem___12198 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 960, 4), get_lapack_funcs_call_result_12197, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 960)
    subscript_call_result_12199 = invoke(stypy.reporting.localization.Localization(__file__, 960, 4), getitem___12198, int_12188)
    
    # Assigning a type to the variable 'tuple_var_assignment_10090' (line 960)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 960, 4), 'tuple_var_assignment_10090', subscript_call_result_12199)
    
    # Assigning a Name to a Name (line 960):
    # Getting the type of 'tuple_var_assignment_10088' (line 960)
    tuple_var_assignment_10088_12200 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 960, 4), 'tuple_var_assignment_10088')
    # Assigning a type to the variable 'getrf' (line 960)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 960, 4), 'getrf', tuple_var_assignment_10088_12200)
    
    # Assigning a Name to a Name (line 960):
    # Getting the type of 'tuple_var_assignment_10089' (line 960)
    tuple_var_assignment_10089_12201 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 960, 4), 'tuple_var_assignment_10089')
    # Assigning a type to the variable 'getri' (line 960)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 960, 11), 'getri', tuple_var_assignment_10089_12201)
    
    # Assigning a Name to a Name (line 960):
    # Getting the type of 'tuple_var_assignment_10090' (line 960)
    tuple_var_assignment_10090_12202 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 960, 4), 'tuple_var_assignment_10090')
    # Assigning a type to the variable 'getri_lwork' (line 960)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 960, 18), 'getri_lwork', tuple_var_assignment_10090_12202)
    
    # Assigning a Call to a Tuple (line 963):
    
    # Assigning a Subscript to a Name (line 963):
    
    # Obtaining the type of the subscript
    int_12203 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 963, 4), 'int')
    
    # Call to getrf(...): (line 963)
    # Processing the call arguments (line 963)
    # Getting the type of 'a1' (line 963)
    a1_12205 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 963, 26), 'a1', False)
    # Processing the call keyword arguments (line 963)
    # Getting the type of 'overwrite_a' (line 963)
    overwrite_a_12206 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 963, 42), 'overwrite_a', False)
    keyword_12207 = overwrite_a_12206
    kwargs_12208 = {'overwrite_a': keyword_12207}
    # Getting the type of 'getrf' (line 963)
    getrf_12204 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 963, 20), 'getrf', False)
    # Calling getrf(args, kwargs) (line 963)
    getrf_call_result_12209 = invoke(stypy.reporting.localization.Localization(__file__, 963, 20), getrf_12204, *[a1_12205], **kwargs_12208)
    
    # Obtaining the member '__getitem__' of a type (line 963)
    getitem___12210 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 963, 4), getrf_call_result_12209, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 963)
    subscript_call_result_12211 = invoke(stypy.reporting.localization.Localization(__file__, 963, 4), getitem___12210, int_12203)
    
    # Assigning a type to the variable 'tuple_var_assignment_10091' (line 963)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 963, 4), 'tuple_var_assignment_10091', subscript_call_result_12211)
    
    # Assigning a Subscript to a Name (line 963):
    
    # Obtaining the type of the subscript
    int_12212 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 963, 4), 'int')
    
    # Call to getrf(...): (line 963)
    # Processing the call arguments (line 963)
    # Getting the type of 'a1' (line 963)
    a1_12214 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 963, 26), 'a1', False)
    # Processing the call keyword arguments (line 963)
    # Getting the type of 'overwrite_a' (line 963)
    overwrite_a_12215 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 963, 42), 'overwrite_a', False)
    keyword_12216 = overwrite_a_12215
    kwargs_12217 = {'overwrite_a': keyword_12216}
    # Getting the type of 'getrf' (line 963)
    getrf_12213 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 963, 20), 'getrf', False)
    # Calling getrf(args, kwargs) (line 963)
    getrf_call_result_12218 = invoke(stypy.reporting.localization.Localization(__file__, 963, 20), getrf_12213, *[a1_12214], **kwargs_12217)
    
    # Obtaining the member '__getitem__' of a type (line 963)
    getitem___12219 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 963, 4), getrf_call_result_12218, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 963)
    subscript_call_result_12220 = invoke(stypy.reporting.localization.Localization(__file__, 963, 4), getitem___12219, int_12212)
    
    # Assigning a type to the variable 'tuple_var_assignment_10092' (line 963)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 963, 4), 'tuple_var_assignment_10092', subscript_call_result_12220)
    
    # Assigning a Subscript to a Name (line 963):
    
    # Obtaining the type of the subscript
    int_12221 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 963, 4), 'int')
    
    # Call to getrf(...): (line 963)
    # Processing the call arguments (line 963)
    # Getting the type of 'a1' (line 963)
    a1_12223 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 963, 26), 'a1', False)
    # Processing the call keyword arguments (line 963)
    # Getting the type of 'overwrite_a' (line 963)
    overwrite_a_12224 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 963, 42), 'overwrite_a', False)
    keyword_12225 = overwrite_a_12224
    kwargs_12226 = {'overwrite_a': keyword_12225}
    # Getting the type of 'getrf' (line 963)
    getrf_12222 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 963, 20), 'getrf', False)
    # Calling getrf(args, kwargs) (line 963)
    getrf_call_result_12227 = invoke(stypy.reporting.localization.Localization(__file__, 963, 20), getrf_12222, *[a1_12223], **kwargs_12226)
    
    # Obtaining the member '__getitem__' of a type (line 963)
    getitem___12228 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 963, 4), getrf_call_result_12227, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 963)
    subscript_call_result_12229 = invoke(stypy.reporting.localization.Localization(__file__, 963, 4), getitem___12228, int_12221)
    
    # Assigning a type to the variable 'tuple_var_assignment_10093' (line 963)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 963, 4), 'tuple_var_assignment_10093', subscript_call_result_12229)
    
    # Assigning a Name to a Name (line 963):
    # Getting the type of 'tuple_var_assignment_10091' (line 963)
    tuple_var_assignment_10091_12230 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 963, 4), 'tuple_var_assignment_10091')
    # Assigning a type to the variable 'lu' (line 963)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 963, 4), 'lu', tuple_var_assignment_10091_12230)
    
    # Assigning a Name to a Name (line 963):
    # Getting the type of 'tuple_var_assignment_10092' (line 963)
    tuple_var_assignment_10092_12231 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 963, 4), 'tuple_var_assignment_10092')
    # Assigning a type to the variable 'piv' (line 963)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 963, 8), 'piv', tuple_var_assignment_10092_12231)
    
    # Assigning a Name to a Name (line 963):
    # Getting the type of 'tuple_var_assignment_10093' (line 963)
    tuple_var_assignment_10093_12232 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 963, 4), 'tuple_var_assignment_10093')
    # Assigning a type to the variable 'info' (line 963)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 963, 13), 'info', tuple_var_assignment_10093_12232)
    
    
    # Getting the type of 'info' (line 964)
    info_12233 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 964, 7), 'info')
    int_12234 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 964, 15), 'int')
    # Applying the binary operator '==' (line 964)
    result_eq_12235 = python_operator(stypy.reporting.localization.Localization(__file__, 964, 7), '==', info_12233, int_12234)
    
    # Testing the type of an if condition (line 964)
    if_condition_12236 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 964, 4), result_eq_12235)
    # Assigning a type to the variable 'if_condition_12236' (line 964)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 964, 4), 'if_condition_12236', if_condition_12236)
    # SSA begins for if statement (line 964)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 965):
    
    # Assigning a Call to a Name (line 965):
    
    # Call to _compute_lwork(...): (line 965)
    # Processing the call arguments (line 965)
    # Getting the type of 'getri_lwork' (line 965)
    getri_lwork_12238 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 965, 31), 'getri_lwork', False)
    
    # Obtaining the type of the subscript
    int_12239 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 965, 53), 'int')
    # Getting the type of 'a1' (line 965)
    a1_12240 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 965, 44), 'a1', False)
    # Obtaining the member 'shape' of a type (line 965)
    shape_12241 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 965, 44), a1_12240, 'shape')
    # Obtaining the member '__getitem__' of a type (line 965)
    getitem___12242 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 965, 44), shape_12241, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 965)
    subscript_call_result_12243 = invoke(stypy.reporting.localization.Localization(__file__, 965, 44), getitem___12242, int_12239)
    
    # Processing the call keyword arguments (line 965)
    kwargs_12244 = {}
    # Getting the type of '_compute_lwork' (line 965)
    _compute_lwork_12237 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 965, 16), '_compute_lwork', False)
    # Calling _compute_lwork(args, kwargs) (line 965)
    _compute_lwork_call_result_12245 = invoke(stypy.reporting.localization.Localization(__file__, 965, 16), _compute_lwork_12237, *[getri_lwork_12238, subscript_call_result_12243], **kwargs_12244)
    
    # Assigning a type to the variable 'lwork' (line 965)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 965, 8), 'lwork', _compute_lwork_call_result_12245)
    
    # Assigning a Call to a Name (line 973):
    
    # Assigning a Call to a Name (line 973):
    
    # Call to int(...): (line 973)
    # Processing the call arguments (line 973)
    float_12247 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 973, 20), 'float')
    # Getting the type of 'lwork' (line 973)
    lwork_12248 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 973, 27), 'lwork', False)
    # Applying the binary operator '*' (line 973)
    result_mul_12249 = python_operator(stypy.reporting.localization.Localization(__file__, 973, 20), '*', float_12247, lwork_12248)
    
    # Processing the call keyword arguments (line 973)
    kwargs_12250 = {}
    # Getting the type of 'int' (line 973)
    int_12246 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 973, 16), 'int', False)
    # Calling int(args, kwargs) (line 973)
    int_call_result_12251 = invoke(stypy.reporting.localization.Localization(__file__, 973, 16), int_12246, *[result_mul_12249], **kwargs_12250)
    
    # Assigning a type to the variable 'lwork' (line 973)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 973, 8), 'lwork', int_call_result_12251)
    
    # Assigning a Call to a Tuple (line 974):
    
    # Assigning a Subscript to a Name (line 974):
    
    # Obtaining the type of the subscript
    int_12252 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 974, 8), 'int')
    
    # Call to getri(...): (line 974)
    # Processing the call arguments (line 974)
    # Getting the type of 'lu' (line 974)
    lu_12254 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 974, 28), 'lu', False)
    # Getting the type of 'piv' (line 974)
    piv_12255 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 974, 32), 'piv', False)
    # Processing the call keyword arguments (line 974)
    # Getting the type of 'lwork' (line 974)
    lwork_12256 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 974, 43), 'lwork', False)
    keyword_12257 = lwork_12256
    int_12258 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 974, 63), 'int')
    keyword_12259 = int_12258
    kwargs_12260 = {'lwork': keyword_12257, 'overwrite_lu': keyword_12259}
    # Getting the type of 'getri' (line 974)
    getri_12253 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 974, 22), 'getri', False)
    # Calling getri(args, kwargs) (line 974)
    getri_call_result_12261 = invoke(stypy.reporting.localization.Localization(__file__, 974, 22), getri_12253, *[lu_12254, piv_12255], **kwargs_12260)
    
    # Obtaining the member '__getitem__' of a type (line 974)
    getitem___12262 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 974, 8), getri_call_result_12261, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 974)
    subscript_call_result_12263 = invoke(stypy.reporting.localization.Localization(__file__, 974, 8), getitem___12262, int_12252)
    
    # Assigning a type to the variable 'tuple_var_assignment_10094' (line 974)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 974, 8), 'tuple_var_assignment_10094', subscript_call_result_12263)
    
    # Assigning a Subscript to a Name (line 974):
    
    # Obtaining the type of the subscript
    int_12264 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 974, 8), 'int')
    
    # Call to getri(...): (line 974)
    # Processing the call arguments (line 974)
    # Getting the type of 'lu' (line 974)
    lu_12266 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 974, 28), 'lu', False)
    # Getting the type of 'piv' (line 974)
    piv_12267 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 974, 32), 'piv', False)
    # Processing the call keyword arguments (line 974)
    # Getting the type of 'lwork' (line 974)
    lwork_12268 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 974, 43), 'lwork', False)
    keyword_12269 = lwork_12268
    int_12270 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 974, 63), 'int')
    keyword_12271 = int_12270
    kwargs_12272 = {'lwork': keyword_12269, 'overwrite_lu': keyword_12271}
    # Getting the type of 'getri' (line 974)
    getri_12265 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 974, 22), 'getri', False)
    # Calling getri(args, kwargs) (line 974)
    getri_call_result_12273 = invoke(stypy.reporting.localization.Localization(__file__, 974, 22), getri_12265, *[lu_12266, piv_12267], **kwargs_12272)
    
    # Obtaining the member '__getitem__' of a type (line 974)
    getitem___12274 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 974, 8), getri_call_result_12273, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 974)
    subscript_call_result_12275 = invoke(stypy.reporting.localization.Localization(__file__, 974, 8), getitem___12274, int_12264)
    
    # Assigning a type to the variable 'tuple_var_assignment_10095' (line 974)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 974, 8), 'tuple_var_assignment_10095', subscript_call_result_12275)
    
    # Assigning a Name to a Name (line 974):
    # Getting the type of 'tuple_var_assignment_10094' (line 974)
    tuple_var_assignment_10094_12276 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 974, 8), 'tuple_var_assignment_10094')
    # Assigning a type to the variable 'inv_a' (line 974)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 974, 8), 'inv_a', tuple_var_assignment_10094_12276)
    
    # Assigning a Name to a Name (line 974):
    # Getting the type of 'tuple_var_assignment_10095' (line 974)
    tuple_var_assignment_10095_12277 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 974, 8), 'tuple_var_assignment_10095')
    # Assigning a type to the variable 'info' (line 974)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 974, 15), 'info', tuple_var_assignment_10095_12277)
    # SSA join for if statement (line 964)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'info' (line 975)
    info_12278 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 975, 7), 'info')
    int_12279 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 975, 14), 'int')
    # Applying the binary operator '>' (line 975)
    result_gt_12280 = python_operator(stypy.reporting.localization.Localization(__file__, 975, 7), '>', info_12278, int_12279)
    
    # Testing the type of an if condition (line 975)
    if_condition_12281 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 975, 4), result_gt_12280)
    # Assigning a type to the variable 'if_condition_12281' (line 975)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 975, 4), 'if_condition_12281', if_condition_12281)
    # SSA begins for if statement (line 975)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to LinAlgError(...): (line 976)
    # Processing the call arguments (line 976)
    str_12283 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 976, 26), 'str', 'singular matrix')
    # Processing the call keyword arguments (line 976)
    kwargs_12284 = {}
    # Getting the type of 'LinAlgError' (line 976)
    LinAlgError_12282 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 976, 14), 'LinAlgError', False)
    # Calling LinAlgError(args, kwargs) (line 976)
    LinAlgError_call_result_12285 = invoke(stypy.reporting.localization.Localization(__file__, 976, 14), LinAlgError_12282, *[str_12283], **kwargs_12284)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 976, 8), LinAlgError_call_result_12285, 'raise parameter', BaseException)
    # SSA join for if statement (line 975)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'info' (line 977)
    info_12286 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 977, 7), 'info')
    int_12287 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 977, 14), 'int')
    # Applying the binary operator '<' (line 977)
    result_lt_12288 = python_operator(stypy.reporting.localization.Localization(__file__, 977, 7), '<', info_12286, int_12287)
    
    # Testing the type of an if condition (line 977)
    if_condition_12289 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 977, 4), result_lt_12288)
    # Assigning a type to the variable 'if_condition_12289' (line 977)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 977, 4), 'if_condition_12289', if_condition_12289)
    # SSA begins for if statement (line 977)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 978)
    # Processing the call arguments (line 978)
    str_12291 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 978, 25), 'str', 'illegal value in %d-th argument of internal getrf|getri')
    
    # Getting the type of 'info' (line 979)
    info_12292 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 979, 42), 'info', False)
    # Applying the 'usub' unary operator (line 979)
    result___neg___12293 = python_operator(stypy.reporting.localization.Localization(__file__, 979, 41), 'usub', info_12292)
    
    # Applying the binary operator '%' (line 978)
    result_mod_12294 = python_operator(stypy.reporting.localization.Localization(__file__, 978, 25), '%', str_12291, result___neg___12293)
    
    # Processing the call keyword arguments (line 978)
    kwargs_12295 = {}
    # Getting the type of 'ValueError' (line 978)
    ValueError_12290 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 978, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 978)
    ValueError_call_result_12296 = invoke(stypy.reporting.localization.Localization(__file__, 978, 14), ValueError_12290, *[result_mod_12294], **kwargs_12295)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 978, 8), ValueError_call_result_12296, 'raise parameter', BaseException)
    # SSA join for if statement (line 977)
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'inv_a' (line 980)
    inv_a_12297 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 980, 11), 'inv_a')
    # Assigning a type to the variable 'stypy_return_type' (line 980)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 980, 4), 'stypy_return_type', inv_a_12297)
    
    # ################# End of 'inv(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'inv' in the type store
    # Getting the type of 'stypy_return_type' (line 908)
    stypy_return_type_12298 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 908, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_12298)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'inv'
    return stypy_return_type_12298

# Assigning a type to the variable 'inv' (line 908)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 908, 0), 'inv', inv)

@norecursion
def det(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'False' (line 985)
    False_12299 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 985, 23), 'False')
    # Getting the type of 'True' (line 985)
    True_12300 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 985, 43), 'True')
    defaults = [False_12299, True_12300]
    # Create a new context for function 'det'
    module_type_store = module_type_store.open_function_context('det', 985, 0, False)
    
    # Passed parameters checking function
    det.stypy_localization = localization
    det.stypy_type_of_self = None
    det.stypy_type_store = module_type_store
    det.stypy_function_name = 'det'
    det.stypy_param_names_list = ['a', 'overwrite_a', 'check_finite']
    det.stypy_varargs_param_name = None
    det.stypy_kwargs_param_name = None
    det.stypy_call_defaults = defaults
    det.stypy_call_varargs = varargs
    det.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'det', ['a', 'overwrite_a', 'check_finite'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'det', localization, ['a', 'overwrite_a', 'check_finite'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'det(...)' code ##################

    str_12301 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1030, (-1)), 'str', '\n    Compute the determinant of a matrix\n\n    The determinant of a square matrix is a value derived arithmetically\n    from the coefficients of the matrix.\n\n    The determinant for a 3x3 matrix, for example, is computed as follows::\n\n        a    b    c\n        d    e    f = A\n        g    h    i\n\n        det(A) = a*e*i + b*f*g + c*d*h - c*e*g - b*d*i - a*f*h\n\n    Parameters\n    ----------\n    a : (M, M) array_like\n        A square matrix.\n    overwrite_a : bool, optional\n        Allow overwriting data in a (may enhance performance).\n    check_finite : bool, optional\n        Whether to check that the input matrix contains only finite numbers.\n        Disabling may give a performance gain, but may result in problems\n        (crashes, non-termination) if the inputs do contain infinities or NaNs.\n\n    Returns\n    -------\n    det : float or complex\n        Determinant of `a`.\n\n    Notes\n    -----\n    The determinant is computed via LU factorization, LAPACK routine z/dgetrf.\n\n    Examples\n    --------\n    >>> from scipy import linalg\n    >>> a = np.array([[1,2,3], [4,5,6], [7,8,9]])\n    >>> linalg.det(a)\n    0.0\n    >>> a = np.array([[0,2,3], [4,5,6], [7,8,9]])\n    >>> linalg.det(a)\n    3.0\n\n    ')
    
    # Assigning a Call to a Name (line 1031):
    
    # Assigning a Call to a Name (line 1031):
    
    # Call to _asarray_validated(...): (line 1031)
    # Processing the call arguments (line 1031)
    # Getting the type of 'a' (line 1031)
    a_12303 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1031, 28), 'a', False)
    # Processing the call keyword arguments (line 1031)
    # Getting the type of 'check_finite' (line 1031)
    check_finite_12304 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1031, 44), 'check_finite', False)
    keyword_12305 = check_finite_12304
    kwargs_12306 = {'check_finite': keyword_12305}
    # Getting the type of '_asarray_validated' (line 1031)
    _asarray_validated_12302 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1031, 9), '_asarray_validated', False)
    # Calling _asarray_validated(args, kwargs) (line 1031)
    _asarray_validated_call_result_12307 = invoke(stypy.reporting.localization.Localization(__file__, 1031, 9), _asarray_validated_12302, *[a_12303], **kwargs_12306)
    
    # Assigning a type to the variable 'a1' (line 1031)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1031, 4), 'a1', _asarray_validated_call_result_12307)
    
    
    # Evaluating a boolean operation
    
    
    # Call to len(...): (line 1032)
    # Processing the call arguments (line 1032)
    # Getting the type of 'a1' (line 1032)
    a1_12309 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1032, 11), 'a1', False)
    # Obtaining the member 'shape' of a type (line 1032)
    shape_12310 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1032, 11), a1_12309, 'shape')
    # Processing the call keyword arguments (line 1032)
    kwargs_12311 = {}
    # Getting the type of 'len' (line 1032)
    len_12308 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1032, 7), 'len', False)
    # Calling len(args, kwargs) (line 1032)
    len_call_result_12312 = invoke(stypy.reporting.localization.Localization(__file__, 1032, 7), len_12308, *[shape_12310], **kwargs_12311)
    
    int_12313 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1032, 24), 'int')
    # Applying the binary operator '!=' (line 1032)
    result_ne_12314 = python_operator(stypy.reporting.localization.Localization(__file__, 1032, 7), '!=', len_call_result_12312, int_12313)
    
    
    
    # Obtaining the type of the subscript
    int_12315 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1032, 38), 'int')
    # Getting the type of 'a1' (line 1032)
    a1_12316 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1032, 29), 'a1')
    # Obtaining the member 'shape' of a type (line 1032)
    shape_12317 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1032, 29), a1_12316, 'shape')
    # Obtaining the member '__getitem__' of a type (line 1032)
    getitem___12318 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1032, 29), shape_12317, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1032)
    subscript_call_result_12319 = invoke(stypy.reporting.localization.Localization(__file__, 1032, 29), getitem___12318, int_12315)
    
    
    # Obtaining the type of the subscript
    int_12320 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1032, 53), 'int')
    # Getting the type of 'a1' (line 1032)
    a1_12321 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1032, 44), 'a1')
    # Obtaining the member 'shape' of a type (line 1032)
    shape_12322 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1032, 44), a1_12321, 'shape')
    # Obtaining the member '__getitem__' of a type (line 1032)
    getitem___12323 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1032, 44), shape_12322, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1032)
    subscript_call_result_12324 = invoke(stypy.reporting.localization.Localization(__file__, 1032, 44), getitem___12323, int_12320)
    
    # Applying the binary operator '!=' (line 1032)
    result_ne_12325 = python_operator(stypy.reporting.localization.Localization(__file__, 1032, 29), '!=', subscript_call_result_12319, subscript_call_result_12324)
    
    # Applying the binary operator 'or' (line 1032)
    result_or_keyword_12326 = python_operator(stypy.reporting.localization.Localization(__file__, 1032, 7), 'or', result_ne_12314, result_ne_12325)
    
    # Testing the type of an if condition (line 1032)
    if_condition_12327 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1032, 4), result_or_keyword_12326)
    # Assigning a type to the variable 'if_condition_12327' (line 1032)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1032, 4), 'if_condition_12327', if_condition_12327)
    # SSA begins for if statement (line 1032)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 1033)
    # Processing the call arguments (line 1033)
    str_12329 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1033, 25), 'str', 'expected square matrix')
    # Processing the call keyword arguments (line 1033)
    kwargs_12330 = {}
    # Getting the type of 'ValueError' (line 1033)
    ValueError_12328 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1033, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 1033)
    ValueError_call_result_12331 = invoke(stypy.reporting.localization.Localization(__file__, 1033, 14), ValueError_12328, *[str_12329], **kwargs_12330)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 1033, 8), ValueError_call_result_12331, 'raise parameter', BaseException)
    # SSA join for if statement (line 1032)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a BoolOp to a Name (line 1034):
    
    # Assigning a BoolOp to a Name (line 1034):
    
    # Evaluating a boolean operation
    # Getting the type of 'overwrite_a' (line 1034)
    overwrite_a_12332 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1034, 18), 'overwrite_a')
    
    # Call to _datacopied(...): (line 1034)
    # Processing the call arguments (line 1034)
    # Getting the type of 'a1' (line 1034)
    a1_12334 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1034, 45), 'a1', False)
    # Getting the type of 'a' (line 1034)
    a_12335 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1034, 49), 'a', False)
    # Processing the call keyword arguments (line 1034)
    kwargs_12336 = {}
    # Getting the type of '_datacopied' (line 1034)
    _datacopied_12333 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1034, 33), '_datacopied', False)
    # Calling _datacopied(args, kwargs) (line 1034)
    _datacopied_call_result_12337 = invoke(stypy.reporting.localization.Localization(__file__, 1034, 33), _datacopied_12333, *[a1_12334, a_12335], **kwargs_12336)
    
    # Applying the binary operator 'or' (line 1034)
    result_or_keyword_12338 = python_operator(stypy.reporting.localization.Localization(__file__, 1034, 18), 'or', overwrite_a_12332, _datacopied_call_result_12337)
    
    # Assigning a type to the variable 'overwrite_a' (line 1034)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1034, 4), 'overwrite_a', result_or_keyword_12338)
    
    # Assigning a Call to a Tuple (line 1035):
    
    # Assigning a Subscript to a Name (line 1035):
    
    # Obtaining the type of the subscript
    int_12339 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1035, 4), 'int')
    
    # Call to get_flinalg_funcs(...): (line 1035)
    # Processing the call arguments (line 1035)
    
    # Obtaining an instance of the builtin type 'tuple' (line 1035)
    tuple_12341 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1035, 31), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 1035)
    # Adding element type (line 1035)
    str_12342 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1035, 31), 'str', 'det')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1035, 31), tuple_12341, str_12342)
    
    
    # Obtaining an instance of the builtin type 'tuple' (line 1035)
    tuple_12343 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1035, 41), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 1035)
    # Adding element type (line 1035)
    # Getting the type of 'a1' (line 1035)
    a1_12344 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1035, 41), 'a1', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1035, 41), tuple_12343, a1_12344)
    
    # Processing the call keyword arguments (line 1035)
    kwargs_12345 = {}
    # Getting the type of 'get_flinalg_funcs' (line 1035)
    get_flinalg_funcs_12340 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1035, 12), 'get_flinalg_funcs', False)
    # Calling get_flinalg_funcs(args, kwargs) (line 1035)
    get_flinalg_funcs_call_result_12346 = invoke(stypy.reporting.localization.Localization(__file__, 1035, 12), get_flinalg_funcs_12340, *[tuple_12341, tuple_12343], **kwargs_12345)
    
    # Obtaining the member '__getitem__' of a type (line 1035)
    getitem___12347 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1035, 4), get_flinalg_funcs_call_result_12346, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1035)
    subscript_call_result_12348 = invoke(stypy.reporting.localization.Localization(__file__, 1035, 4), getitem___12347, int_12339)
    
    # Assigning a type to the variable 'tuple_var_assignment_10096' (line 1035)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1035, 4), 'tuple_var_assignment_10096', subscript_call_result_12348)
    
    # Assigning a Name to a Name (line 1035):
    # Getting the type of 'tuple_var_assignment_10096' (line 1035)
    tuple_var_assignment_10096_12349 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1035, 4), 'tuple_var_assignment_10096')
    # Assigning a type to the variable 'fdet' (line 1035)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1035, 4), 'fdet', tuple_var_assignment_10096_12349)
    
    # Assigning a Call to a Tuple (line 1036):
    
    # Assigning a Subscript to a Name (line 1036):
    
    # Obtaining the type of the subscript
    int_12350 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1036, 4), 'int')
    
    # Call to fdet(...): (line 1036)
    # Processing the call arguments (line 1036)
    # Getting the type of 'a1' (line 1036)
    a1_12352 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1036, 23), 'a1', False)
    # Processing the call keyword arguments (line 1036)
    # Getting the type of 'overwrite_a' (line 1036)
    overwrite_a_12353 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1036, 39), 'overwrite_a', False)
    keyword_12354 = overwrite_a_12353
    kwargs_12355 = {'overwrite_a': keyword_12354}
    # Getting the type of 'fdet' (line 1036)
    fdet_12351 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1036, 18), 'fdet', False)
    # Calling fdet(args, kwargs) (line 1036)
    fdet_call_result_12356 = invoke(stypy.reporting.localization.Localization(__file__, 1036, 18), fdet_12351, *[a1_12352], **kwargs_12355)
    
    # Obtaining the member '__getitem__' of a type (line 1036)
    getitem___12357 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1036, 4), fdet_call_result_12356, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1036)
    subscript_call_result_12358 = invoke(stypy.reporting.localization.Localization(__file__, 1036, 4), getitem___12357, int_12350)
    
    # Assigning a type to the variable 'tuple_var_assignment_10097' (line 1036)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1036, 4), 'tuple_var_assignment_10097', subscript_call_result_12358)
    
    # Assigning a Subscript to a Name (line 1036):
    
    # Obtaining the type of the subscript
    int_12359 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1036, 4), 'int')
    
    # Call to fdet(...): (line 1036)
    # Processing the call arguments (line 1036)
    # Getting the type of 'a1' (line 1036)
    a1_12361 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1036, 23), 'a1', False)
    # Processing the call keyword arguments (line 1036)
    # Getting the type of 'overwrite_a' (line 1036)
    overwrite_a_12362 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1036, 39), 'overwrite_a', False)
    keyword_12363 = overwrite_a_12362
    kwargs_12364 = {'overwrite_a': keyword_12363}
    # Getting the type of 'fdet' (line 1036)
    fdet_12360 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1036, 18), 'fdet', False)
    # Calling fdet(args, kwargs) (line 1036)
    fdet_call_result_12365 = invoke(stypy.reporting.localization.Localization(__file__, 1036, 18), fdet_12360, *[a1_12361], **kwargs_12364)
    
    # Obtaining the member '__getitem__' of a type (line 1036)
    getitem___12366 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1036, 4), fdet_call_result_12365, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1036)
    subscript_call_result_12367 = invoke(stypy.reporting.localization.Localization(__file__, 1036, 4), getitem___12366, int_12359)
    
    # Assigning a type to the variable 'tuple_var_assignment_10098' (line 1036)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1036, 4), 'tuple_var_assignment_10098', subscript_call_result_12367)
    
    # Assigning a Name to a Name (line 1036):
    # Getting the type of 'tuple_var_assignment_10097' (line 1036)
    tuple_var_assignment_10097_12368 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1036, 4), 'tuple_var_assignment_10097')
    # Assigning a type to the variable 'a_det' (line 1036)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1036, 4), 'a_det', tuple_var_assignment_10097_12368)
    
    # Assigning a Name to a Name (line 1036):
    # Getting the type of 'tuple_var_assignment_10098' (line 1036)
    tuple_var_assignment_10098_12369 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1036, 4), 'tuple_var_assignment_10098')
    # Assigning a type to the variable 'info' (line 1036)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1036, 11), 'info', tuple_var_assignment_10098_12369)
    
    
    # Getting the type of 'info' (line 1037)
    info_12370 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1037, 7), 'info')
    int_12371 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1037, 14), 'int')
    # Applying the binary operator '<' (line 1037)
    result_lt_12372 = python_operator(stypy.reporting.localization.Localization(__file__, 1037, 7), '<', info_12370, int_12371)
    
    # Testing the type of an if condition (line 1037)
    if_condition_12373 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1037, 4), result_lt_12372)
    # Assigning a type to the variable 'if_condition_12373' (line 1037)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1037, 4), 'if_condition_12373', if_condition_12373)
    # SSA begins for if statement (line 1037)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 1038)
    # Processing the call arguments (line 1038)
    str_12375 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1038, 25), 'str', 'illegal value in %d-th argument of internal det.getrf')
    
    # Getting the type of 'info' (line 1039)
    info_12376 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1039, 40), 'info', False)
    # Applying the 'usub' unary operator (line 1039)
    result___neg___12377 = python_operator(stypy.reporting.localization.Localization(__file__, 1039, 39), 'usub', info_12376)
    
    # Applying the binary operator '%' (line 1038)
    result_mod_12378 = python_operator(stypy.reporting.localization.Localization(__file__, 1038, 25), '%', str_12375, result___neg___12377)
    
    # Processing the call keyword arguments (line 1038)
    kwargs_12379 = {}
    # Getting the type of 'ValueError' (line 1038)
    ValueError_12374 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1038, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 1038)
    ValueError_call_result_12380 = invoke(stypy.reporting.localization.Localization(__file__, 1038, 14), ValueError_12374, *[result_mod_12378], **kwargs_12379)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 1038, 8), ValueError_call_result_12380, 'raise parameter', BaseException)
    # SSA join for if statement (line 1037)
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'a_det' (line 1040)
    a_det_12381 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1040, 11), 'a_det')
    # Assigning a type to the variable 'stypy_return_type' (line 1040)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1040, 4), 'stypy_return_type', a_det_12381)
    
    # ################# End of 'det(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'det' in the type store
    # Getting the type of 'stypy_return_type' (line 985)
    stypy_return_type_12382 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 985, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_12382)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'det'
    return stypy_return_type_12382

# Assigning a type to the variable 'det' (line 985)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 985, 0), 'det', det)
# Declaration of the 'LstsqLapackError' class
# Getting the type of 'LinAlgError' (line 1045)
LinAlgError_12383 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1045, 23), 'LinAlgError')

class LstsqLapackError(LinAlgError_12383, ):
    pass

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 1045, 0, False)
        # Assigning a type to the variable 'self' (line 1046)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1046, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'LstsqLapackError.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'LstsqLapackError' (line 1045)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1045, 0), 'LstsqLapackError', LstsqLapackError)

@norecursion
def lstsq(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'None' (line 1049)
    None_12384 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1049, 21), 'None')
    # Getting the type of 'False' (line 1049)
    False_12385 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1049, 39), 'False')
    # Getting the type of 'False' (line 1049)
    False_12386 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1049, 58), 'False')
    # Getting the type of 'True' (line 1050)
    True_12387 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1050, 23), 'True')
    # Getting the type of 'None' (line 1050)
    None_12388 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1050, 43), 'None')
    defaults = [None_12384, False_12385, False_12386, True_12387, None_12388]
    # Create a new context for function 'lstsq'
    module_type_store = module_type_store.open_function_context('lstsq', 1049, 0, False)
    
    # Passed parameters checking function
    lstsq.stypy_localization = localization
    lstsq.stypy_type_of_self = None
    lstsq.stypy_type_store = module_type_store
    lstsq.stypy_function_name = 'lstsq'
    lstsq.stypy_param_names_list = ['a', 'b', 'cond', 'overwrite_a', 'overwrite_b', 'check_finite', 'lapack_driver']
    lstsq.stypy_varargs_param_name = None
    lstsq.stypy_kwargs_param_name = None
    lstsq.stypy_call_defaults = defaults
    lstsq.stypy_call_varargs = varargs
    lstsq.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'lstsq', ['a', 'b', 'cond', 'overwrite_a', 'overwrite_b', 'check_finite', 'lapack_driver'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'lstsq', localization, ['a', 'b', 'cond', 'overwrite_a', 'overwrite_b', 'check_finite', 'lapack_driver'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'lstsq(...)' code ##################

    str_12389 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1154, (-1)), 'str', '\n    Compute least-squares solution to equation Ax = b.\n\n    Compute a vector x such that the 2-norm ``|b - A x|`` is minimized.\n\n    Parameters\n    ----------\n    a : (M, N) array_like\n        Left hand side matrix (2-D array).\n    b : (M,) or (M, K) array_like\n        Right hand side matrix or vector (1-D or 2-D array).\n    cond : float, optional\n        Cutoff for \'small\' singular values; used to determine effective\n        rank of a. Singular values smaller than\n        ``rcond * largest_singular_value`` are considered zero.\n    overwrite_a : bool, optional\n        Discard data in `a` (may enhance performance). Default is False.\n    overwrite_b : bool, optional\n        Discard data in `b` (may enhance performance). Default is False.\n    check_finite : bool, optional\n        Whether to check that the input matrices contain only finite numbers.\n        Disabling may give a performance gain, but may result in problems\n        (crashes, non-termination) if the inputs do contain infinities or NaNs.\n    lapack_driver : str, optional\n        Which LAPACK driver is used to solve the least-squares problem.\n        Options are ``\'gelsd\'``, ``\'gelsy\'``, ``\'gelss\'``. Default\n        (``\'gelsd\'``) is a good choice.  However, ``\'gelsy\'`` can be slightly\n        faster on many problems.  ``\'gelss\'`` was used historically.  It is\n        generally slow but uses less memory.\n\n        .. versionadded:: 0.17.0\n\n    Returns\n    -------\n    x : (N,) or (N, K) ndarray\n        Least-squares solution.  Return shape matches shape of `b`.\n    residues : (0,) or () or (K,) ndarray\n        Sums of residues, squared 2-norm for each column in ``b - a x``.\n        If rank of matrix a is ``< N`` or ``N > M``, or ``\'gelsy\'`` is used,\n        this is a lenght zero array. If b was 1-D, this is a () shape array\n        (numpy scalar), otherwise the shape is (K,).\n    rank : int\n        Effective rank of matrix `a`.\n    s : (min(M,N),) ndarray or None\n        Singular values of `a`. The condition number of a is\n        ``abs(s[0] / s[-1])``. None is returned when ``\'gelsy\'`` is used.\n\n    Raises\n    ------\n    LinAlgError\n        If computation does not converge.\n\n    ValueError\n        When parameters are wrong.\n\n    See Also\n    --------\n    optimize.nnls : linear least squares with non-negativity constraint\n\n    Examples\n    --------\n    >>> from scipy.linalg import lstsq\n    >>> import matplotlib.pyplot as plt\n\n    Suppose we have the following data:\n\n    >>> x = np.array([1, 2.5, 3.5, 4, 5, 7, 8.5])\n    >>> y = np.array([0.3, 1.1, 1.5, 2.0, 3.2, 6.6, 8.6])\n\n    We want to fit a quadratic polynomial of the form ``y = a + b*x**2``\n    to this data.  We first form the "design matrix" M, with a constant\n    column of 1s and a column containing ``x**2``:\n\n    >>> M = x[:, np.newaxis]**[0, 2]\n    >>> M\n    array([[  1.  ,   1.  ],\n           [  1.  ,   6.25],\n           [  1.  ,  12.25],\n           [  1.  ,  16.  ],\n           [  1.  ,  25.  ],\n           [  1.  ,  49.  ],\n           [  1.  ,  72.25]])\n\n    We want to find the least-squares solution to ``M.dot(p) = y``,\n    where ``p`` is a vector with length 2 that holds the parameters\n    ``a`` and ``b``.\n\n    >>> p, res, rnk, s = lstsq(M, y)\n    >>> p\n    array([ 0.20925829,  0.12013861])\n\n    Plot the data and the fitted curve.\n\n    >>> plt.plot(x, y, \'o\', label=\'data\')\n    >>> xx = np.linspace(0, 9, 101)\n    >>> yy = p[0] + p[1]*xx**2\n    >>> plt.plot(xx, yy, label=\'least squares fit, $y = a + bx^2$\')\n    >>> plt.xlabel(\'x\')\n    >>> plt.ylabel(\'y\')\n    >>> plt.legend(framealpha=1, shadow=True)\n    >>> plt.grid(alpha=0.25)\n    >>> plt.show()\n\n    ')
    
    # Assigning a Call to a Name (line 1155):
    
    # Assigning a Call to a Name (line 1155):
    
    # Call to _asarray_validated(...): (line 1155)
    # Processing the call arguments (line 1155)
    # Getting the type of 'a' (line 1155)
    a_12391 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1155, 28), 'a', False)
    # Processing the call keyword arguments (line 1155)
    # Getting the type of 'check_finite' (line 1155)
    check_finite_12392 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1155, 44), 'check_finite', False)
    keyword_12393 = check_finite_12392
    kwargs_12394 = {'check_finite': keyword_12393}
    # Getting the type of '_asarray_validated' (line 1155)
    _asarray_validated_12390 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1155, 9), '_asarray_validated', False)
    # Calling _asarray_validated(args, kwargs) (line 1155)
    _asarray_validated_call_result_12395 = invoke(stypy.reporting.localization.Localization(__file__, 1155, 9), _asarray_validated_12390, *[a_12391], **kwargs_12394)
    
    # Assigning a type to the variable 'a1' (line 1155)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1155, 4), 'a1', _asarray_validated_call_result_12395)
    
    # Assigning a Call to a Name (line 1156):
    
    # Assigning a Call to a Name (line 1156):
    
    # Call to _asarray_validated(...): (line 1156)
    # Processing the call arguments (line 1156)
    # Getting the type of 'b' (line 1156)
    b_12397 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1156, 28), 'b', False)
    # Processing the call keyword arguments (line 1156)
    # Getting the type of 'check_finite' (line 1156)
    check_finite_12398 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1156, 44), 'check_finite', False)
    keyword_12399 = check_finite_12398
    kwargs_12400 = {'check_finite': keyword_12399}
    # Getting the type of '_asarray_validated' (line 1156)
    _asarray_validated_12396 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1156, 9), '_asarray_validated', False)
    # Calling _asarray_validated(args, kwargs) (line 1156)
    _asarray_validated_call_result_12401 = invoke(stypy.reporting.localization.Localization(__file__, 1156, 9), _asarray_validated_12396, *[b_12397], **kwargs_12400)
    
    # Assigning a type to the variable 'b1' (line 1156)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1156, 4), 'b1', _asarray_validated_call_result_12401)
    
    
    
    # Call to len(...): (line 1157)
    # Processing the call arguments (line 1157)
    # Getting the type of 'a1' (line 1157)
    a1_12403 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1157, 11), 'a1', False)
    # Obtaining the member 'shape' of a type (line 1157)
    shape_12404 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1157, 11), a1_12403, 'shape')
    # Processing the call keyword arguments (line 1157)
    kwargs_12405 = {}
    # Getting the type of 'len' (line 1157)
    len_12402 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1157, 7), 'len', False)
    # Calling len(args, kwargs) (line 1157)
    len_call_result_12406 = invoke(stypy.reporting.localization.Localization(__file__, 1157, 7), len_12402, *[shape_12404], **kwargs_12405)
    
    int_12407 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1157, 24), 'int')
    # Applying the binary operator '!=' (line 1157)
    result_ne_12408 = python_operator(stypy.reporting.localization.Localization(__file__, 1157, 7), '!=', len_call_result_12406, int_12407)
    
    # Testing the type of an if condition (line 1157)
    if_condition_12409 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1157, 4), result_ne_12408)
    # Assigning a type to the variable 'if_condition_12409' (line 1157)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1157, 4), 'if_condition_12409', if_condition_12409)
    # SSA begins for if statement (line 1157)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 1158)
    # Processing the call arguments (line 1158)
    str_12411 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1158, 25), 'str', 'expected matrix')
    # Processing the call keyword arguments (line 1158)
    kwargs_12412 = {}
    # Getting the type of 'ValueError' (line 1158)
    ValueError_12410 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1158, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 1158)
    ValueError_call_result_12413 = invoke(stypy.reporting.localization.Localization(__file__, 1158, 14), ValueError_12410, *[str_12411], **kwargs_12412)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 1158, 8), ValueError_call_result_12413, 'raise parameter', BaseException)
    # SSA join for if statement (line 1157)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Attribute to a Tuple (line 1159):
    
    # Assigning a Subscript to a Name (line 1159):
    
    # Obtaining the type of the subscript
    int_12414 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1159, 4), 'int')
    # Getting the type of 'a1' (line 1159)
    a1_12415 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1159, 11), 'a1')
    # Obtaining the member 'shape' of a type (line 1159)
    shape_12416 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1159, 11), a1_12415, 'shape')
    # Obtaining the member '__getitem__' of a type (line 1159)
    getitem___12417 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1159, 4), shape_12416, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1159)
    subscript_call_result_12418 = invoke(stypy.reporting.localization.Localization(__file__, 1159, 4), getitem___12417, int_12414)
    
    # Assigning a type to the variable 'tuple_var_assignment_10099' (line 1159)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1159, 4), 'tuple_var_assignment_10099', subscript_call_result_12418)
    
    # Assigning a Subscript to a Name (line 1159):
    
    # Obtaining the type of the subscript
    int_12419 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1159, 4), 'int')
    # Getting the type of 'a1' (line 1159)
    a1_12420 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1159, 11), 'a1')
    # Obtaining the member 'shape' of a type (line 1159)
    shape_12421 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1159, 11), a1_12420, 'shape')
    # Obtaining the member '__getitem__' of a type (line 1159)
    getitem___12422 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1159, 4), shape_12421, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1159)
    subscript_call_result_12423 = invoke(stypy.reporting.localization.Localization(__file__, 1159, 4), getitem___12422, int_12419)
    
    # Assigning a type to the variable 'tuple_var_assignment_10100' (line 1159)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1159, 4), 'tuple_var_assignment_10100', subscript_call_result_12423)
    
    # Assigning a Name to a Name (line 1159):
    # Getting the type of 'tuple_var_assignment_10099' (line 1159)
    tuple_var_assignment_10099_12424 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1159, 4), 'tuple_var_assignment_10099')
    # Assigning a type to the variable 'm' (line 1159)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1159, 4), 'm', tuple_var_assignment_10099_12424)
    
    # Assigning a Name to a Name (line 1159):
    # Getting the type of 'tuple_var_assignment_10100' (line 1159)
    tuple_var_assignment_10100_12425 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1159, 4), 'tuple_var_assignment_10100')
    # Assigning a type to the variable 'n' (line 1159)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1159, 7), 'n', tuple_var_assignment_10100_12425)
    
    
    
    # Call to len(...): (line 1160)
    # Processing the call arguments (line 1160)
    # Getting the type of 'b1' (line 1160)
    b1_12427 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1160, 11), 'b1', False)
    # Obtaining the member 'shape' of a type (line 1160)
    shape_12428 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1160, 11), b1_12427, 'shape')
    # Processing the call keyword arguments (line 1160)
    kwargs_12429 = {}
    # Getting the type of 'len' (line 1160)
    len_12426 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1160, 7), 'len', False)
    # Calling len(args, kwargs) (line 1160)
    len_call_result_12430 = invoke(stypy.reporting.localization.Localization(__file__, 1160, 7), len_12426, *[shape_12428], **kwargs_12429)
    
    int_12431 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1160, 24), 'int')
    # Applying the binary operator '==' (line 1160)
    result_eq_12432 = python_operator(stypy.reporting.localization.Localization(__file__, 1160, 7), '==', len_call_result_12430, int_12431)
    
    # Testing the type of an if condition (line 1160)
    if_condition_12433 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1160, 4), result_eq_12432)
    # Assigning a type to the variable 'if_condition_12433' (line 1160)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1160, 4), 'if_condition_12433', if_condition_12433)
    # SSA begins for if statement (line 1160)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Subscript to a Name (line 1161):
    
    # Assigning a Subscript to a Name (line 1161):
    
    # Obtaining the type of the subscript
    int_12434 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1161, 24), 'int')
    # Getting the type of 'b1' (line 1161)
    b1_12435 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1161, 15), 'b1')
    # Obtaining the member 'shape' of a type (line 1161)
    shape_12436 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1161, 15), b1_12435, 'shape')
    # Obtaining the member '__getitem__' of a type (line 1161)
    getitem___12437 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1161, 15), shape_12436, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1161)
    subscript_call_result_12438 = invoke(stypy.reporting.localization.Localization(__file__, 1161, 15), getitem___12437, int_12434)
    
    # Assigning a type to the variable 'nrhs' (line 1161)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1161, 8), 'nrhs', subscript_call_result_12438)
    # SSA branch for the else part of an if statement (line 1160)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Num to a Name (line 1163):
    
    # Assigning a Num to a Name (line 1163):
    int_12439 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1163, 15), 'int')
    # Assigning a type to the variable 'nrhs' (line 1163)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1163, 8), 'nrhs', int_12439)
    # SSA join for if statement (line 1160)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'm' (line 1164)
    m_12440 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1164, 7), 'm')
    
    # Obtaining the type of the subscript
    int_12441 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1164, 21), 'int')
    # Getting the type of 'b1' (line 1164)
    b1_12442 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1164, 12), 'b1')
    # Obtaining the member 'shape' of a type (line 1164)
    shape_12443 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1164, 12), b1_12442, 'shape')
    # Obtaining the member '__getitem__' of a type (line 1164)
    getitem___12444 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1164, 12), shape_12443, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1164)
    subscript_call_result_12445 = invoke(stypy.reporting.localization.Localization(__file__, 1164, 12), getitem___12444, int_12441)
    
    # Applying the binary operator '!=' (line 1164)
    result_ne_12446 = python_operator(stypy.reporting.localization.Localization(__file__, 1164, 7), '!=', m_12440, subscript_call_result_12445)
    
    # Testing the type of an if condition (line 1164)
    if_condition_12447 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1164, 4), result_ne_12446)
    # Assigning a type to the variable 'if_condition_12447' (line 1164)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1164, 4), 'if_condition_12447', if_condition_12447)
    # SSA begins for if statement (line 1164)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 1165)
    # Processing the call arguments (line 1165)
    str_12449 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1165, 25), 'str', 'incompatible dimensions')
    # Processing the call keyword arguments (line 1165)
    kwargs_12450 = {}
    # Getting the type of 'ValueError' (line 1165)
    ValueError_12448 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1165, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 1165)
    ValueError_call_result_12451 = invoke(stypy.reporting.localization.Localization(__file__, 1165, 14), ValueError_12448, *[str_12449], **kwargs_12450)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 1165, 8), ValueError_call_result_12451, 'raise parameter', BaseException)
    # SSA join for if statement (line 1164)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'm' (line 1166)
    m_12452 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1166, 7), 'm')
    int_12453 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1166, 12), 'int')
    # Applying the binary operator '==' (line 1166)
    result_eq_12454 = python_operator(stypy.reporting.localization.Localization(__file__, 1166, 7), '==', m_12452, int_12453)
    
    
    # Getting the type of 'n' (line 1166)
    n_12455 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1166, 17), 'n')
    int_12456 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1166, 22), 'int')
    # Applying the binary operator '==' (line 1166)
    result_eq_12457 = python_operator(stypy.reporting.localization.Localization(__file__, 1166, 17), '==', n_12455, int_12456)
    
    # Applying the binary operator 'or' (line 1166)
    result_or_keyword_12458 = python_operator(stypy.reporting.localization.Localization(__file__, 1166, 7), 'or', result_eq_12454, result_eq_12457)
    
    # Testing the type of an if condition (line 1166)
    if_condition_12459 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1166, 4), result_or_keyword_12458)
    # Assigning a type to the variable 'if_condition_12459' (line 1166)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1166, 4), 'if_condition_12459', if_condition_12459)
    # SSA begins for if statement (line 1166)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 1167):
    
    # Assigning a Call to a Name (line 1167):
    
    # Call to zeros(...): (line 1167)
    # Processing the call arguments (line 1167)
    
    # Obtaining an instance of the builtin type 'tuple' (line 1167)
    tuple_12462 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1167, 22), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 1167)
    # Adding element type (line 1167)
    # Getting the type of 'n' (line 1167)
    n_12463 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1167, 22), 'n', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1167, 22), tuple_12462, n_12463)
    
    
    # Obtaining the type of the subscript
    int_12464 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1167, 37), 'int')
    slice_12465 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 1167, 28), int_12464, None, None)
    # Getting the type of 'b1' (line 1167)
    b1_12466 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1167, 28), 'b1', False)
    # Obtaining the member 'shape' of a type (line 1167)
    shape_12467 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1167, 28), b1_12466, 'shape')
    # Obtaining the member '__getitem__' of a type (line 1167)
    getitem___12468 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1167, 28), shape_12467, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1167)
    subscript_call_result_12469 = invoke(stypy.reporting.localization.Localization(__file__, 1167, 28), getitem___12468, slice_12465)
    
    # Applying the binary operator '+' (line 1167)
    result_add_12470 = python_operator(stypy.reporting.localization.Localization(__file__, 1167, 21), '+', tuple_12462, subscript_call_result_12469)
    
    # Processing the call keyword arguments (line 1167)
    
    # Call to common_type(...): (line 1167)
    # Processing the call arguments (line 1167)
    # Getting the type of 'a1' (line 1167)
    a1_12473 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1167, 63), 'a1', False)
    # Getting the type of 'b1' (line 1167)
    b1_12474 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1167, 67), 'b1', False)
    # Processing the call keyword arguments (line 1167)
    kwargs_12475 = {}
    # Getting the type of 'np' (line 1167)
    np_12471 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1167, 48), 'np', False)
    # Obtaining the member 'common_type' of a type (line 1167)
    common_type_12472 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1167, 48), np_12471, 'common_type')
    # Calling common_type(args, kwargs) (line 1167)
    common_type_call_result_12476 = invoke(stypy.reporting.localization.Localization(__file__, 1167, 48), common_type_12472, *[a1_12473, b1_12474], **kwargs_12475)
    
    keyword_12477 = common_type_call_result_12476
    kwargs_12478 = {'dtype': keyword_12477}
    # Getting the type of 'np' (line 1167)
    np_12460 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1167, 12), 'np', False)
    # Obtaining the member 'zeros' of a type (line 1167)
    zeros_12461 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1167, 12), np_12460, 'zeros')
    # Calling zeros(args, kwargs) (line 1167)
    zeros_call_result_12479 = invoke(stypy.reporting.localization.Localization(__file__, 1167, 12), zeros_12461, *[result_add_12470], **kwargs_12478)
    
    # Assigning a type to the variable 'x' (line 1167)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1167, 8), 'x', zeros_call_result_12479)
    
    
    # Getting the type of 'n' (line 1168)
    n_12480 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1168, 11), 'n')
    int_12481 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1168, 16), 'int')
    # Applying the binary operator '==' (line 1168)
    result_eq_12482 = python_operator(stypy.reporting.localization.Localization(__file__, 1168, 11), '==', n_12480, int_12481)
    
    # Testing the type of an if condition (line 1168)
    if_condition_12483 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1168, 8), result_eq_12482)
    # Assigning a type to the variable 'if_condition_12483' (line 1168)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1168, 8), 'if_condition_12483', if_condition_12483)
    # SSA begins for if statement (line 1168)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a BinOp to a Name (line 1169):
    
    # Assigning a BinOp to a Name (line 1169):
    
    # Call to norm(...): (line 1169)
    # Processing the call arguments (line 1169)
    # Getting the type of 'b1' (line 1169)
    b1_12487 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1169, 38), 'b1', False)
    # Processing the call keyword arguments (line 1169)
    int_12488 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1169, 47), 'int')
    keyword_12489 = int_12488
    kwargs_12490 = {'axis': keyword_12489}
    # Getting the type of 'np' (line 1169)
    np_12484 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1169, 23), 'np', False)
    # Obtaining the member 'linalg' of a type (line 1169)
    linalg_12485 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1169, 23), np_12484, 'linalg')
    # Obtaining the member 'norm' of a type (line 1169)
    norm_12486 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1169, 23), linalg_12485, 'norm')
    # Calling norm(args, kwargs) (line 1169)
    norm_call_result_12491 = invoke(stypy.reporting.localization.Localization(__file__, 1169, 23), norm_12486, *[b1_12487], **kwargs_12490)
    
    int_12492 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1169, 51), 'int')
    # Applying the binary operator '**' (line 1169)
    result_pow_12493 = python_operator(stypy.reporting.localization.Localization(__file__, 1169, 23), '**', norm_call_result_12491, int_12492)
    
    # Assigning a type to the variable 'residues' (line 1169)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1169, 12), 'residues', result_pow_12493)
    # SSA branch for the else part of an if statement (line 1168)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Call to a Name (line 1171):
    
    # Assigning a Call to a Name (line 1171):
    
    # Call to empty(...): (line 1171)
    # Processing the call arguments (line 1171)
    
    # Obtaining an instance of the builtin type 'tuple' (line 1171)
    tuple_12496 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1171, 33), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 1171)
    # Adding element type (line 1171)
    int_12497 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1171, 33), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1171, 33), tuple_12496, int_12497)
    
    # Processing the call keyword arguments (line 1171)
    kwargs_12498 = {}
    # Getting the type of 'np' (line 1171)
    np_12494 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1171, 23), 'np', False)
    # Obtaining the member 'empty' of a type (line 1171)
    empty_12495 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1171, 23), np_12494, 'empty')
    # Calling empty(args, kwargs) (line 1171)
    empty_call_result_12499 = invoke(stypy.reporting.localization.Localization(__file__, 1171, 23), empty_12495, *[tuple_12496], **kwargs_12498)
    
    # Assigning a type to the variable 'residues' (line 1171)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1171, 12), 'residues', empty_call_result_12499)
    # SSA join for if statement (line 1168)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Obtaining an instance of the builtin type 'tuple' (line 1172)
    tuple_12500 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1172, 15), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 1172)
    # Adding element type (line 1172)
    # Getting the type of 'x' (line 1172)
    x_12501 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1172, 15), 'x')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1172, 15), tuple_12500, x_12501)
    # Adding element type (line 1172)
    # Getting the type of 'residues' (line 1172)
    residues_12502 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1172, 18), 'residues')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1172, 15), tuple_12500, residues_12502)
    # Adding element type (line 1172)
    int_12503 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1172, 28), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1172, 15), tuple_12500, int_12503)
    # Adding element type (line 1172)
    
    # Call to empty(...): (line 1172)
    # Processing the call arguments (line 1172)
    
    # Obtaining an instance of the builtin type 'tuple' (line 1172)
    tuple_12506 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1172, 41), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 1172)
    # Adding element type (line 1172)
    int_12507 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1172, 41), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1172, 41), tuple_12506, int_12507)
    
    # Processing the call keyword arguments (line 1172)
    kwargs_12508 = {}
    # Getting the type of 'np' (line 1172)
    np_12504 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1172, 31), 'np', False)
    # Obtaining the member 'empty' of a type (line 1172)
    empty_12505 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1172, 31), np_12504, 'empty')
    # Calling empty(args, kwargs) (line 1172)
    empty_call_result_12509 = invoke(stypy.reporting.localization.Localization(__file__, 1172, 31), empty_12505, *[tuple_12506], **kwargs_12508)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1172, 15), tuple_12500, empty_call_result_12509)
    
    # Assigning a type to the variable 'stypy_return_type' (line 1172)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1172, 8), 'stypy_return_type', tuple_12500)
    # SSA join for if statement (line 1166)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Name to a Name (line 1174):
    
    # Assigning a Name to a Name (line 1174):
    # Getting the type of 'lapack_driver' (line 1174)
    lapack_driver_12510 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1174, 13), 'lapack_driver')
    # Assigning a type to the variable 'driver' (line 1174)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1174, 4), 'driver', lapack_driver_12510)
    
    # Type idiom detected: calculating its left and rigth part (line 1175)
    # Getting the type of 'driver' (line 1175)
    driver_12511 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1175, 7), 'driver')
    # Getting the type of 'None' (line 1175)
    None_12512 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1175, 17), 'None')
    
    (may_be_12513, more_types_in_union_12514) = may_be_none(driver_12511, None_12512)

    if may_be_12513:

        if more_types_in_union_12514:
            # Runtime conditional SSA (line 1175)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a Attribute to a Name (line 1176):
        
        # Assigning a Attribute to a Name (line 1176):
        # Getting the type of 'lstsq' (line 1176)
        lstsq_12515 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1176, 17), 'lstsq')
        # Obtaining the member 'default_lapack_driver' of a type (line 1176)
        default_lapack_driver_12516 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1176, 17), lstsq_12515, 'default_lapack_driver')
        # Assigning a type to the variable 'driver' (line 1176)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1176, 8), 'driver', default_lapack_driver_12516)

        if more_types_in_union_12514:
            # SSA join for if statement (line 1175)
            module_type_store = module_type_store.join_ssa_context()


    
    
    
    # Getting the type of 'driver' (line 1177)
    driver_12517 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1177, 7), 'driver')
    
    # Obtaining an instance of the builtin type 'tuple' (line 1177)
    tuple_12518 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1177, 22), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 1177)
    # Adding element type (line 1177)
    str_12519 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1177, 22), 'str', 'gelsd')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1177, 22), tuple_12518, str_12519)
    # Adding element type (line 1177)
    str_12520 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1177, 31), 'str', 'gelsy')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1177, 22), tuple_12518, str_12520)
    # Adding element type (line 1177)
    str_12521 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1177, 40), 'str', 'gelss')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1177, 22), tuple_12518, str_12521)
    
    # Applying the binary operator 'notin' (line 1177)
    result_contains_12522 = python_operator(stypy.reporting.localization.Localization(__file__, 1177, 7), 'notin', driver_12517, tuple_12518)
    
    # Testing the type of an if condition (line 1177)
    if_condition_12523 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1177, 4), result_contains_12522)
    # Assigning a type to the variable 'if_condition_12523' (line 1177)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1177, 4), 'if_condition_12523', if_condition_12523)
    # SSA begins for if statement (line 1177)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 1178)
    # Processing the call arguments (line 1178)
    str_12525 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1178, 25), 'str', 'LAPACK driver "%s" is not found')
    # Getting the type of 'driver' (line 1178)
    driver_12526 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1178, 61), 'driver', False)
    # Applying the binary operator '%' (line 1178)
    result_mod_12527 = python_operator(stypy.reporting.localization.Localization(__file__, 1178, 25), '%', str_12525, driver_12526)
    
    # Processing the call keyword arguments (line 1178)
    kwargs_12528 = {}
    # Getting the type of 'ValueError' (line 1178)
    ValueError_12524 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1178, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 1178)
    ValueError_call_result_12529 = invoke(stypy.reporting.localization.Localization(__file__, 1178, 14), ValueError_12524, *[result_mod_12527], **kwargs_12528)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 1178, 8), ValueError_call_result_12529, 'raise parameter', BaseException)
    # SSA join for if statement (line 1177)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Tuple (line 1180):
    
    # Assigning a Subscript to a Name (line 1180):
    
    # Obtaining the type of the subscript
    int_12530 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1180, 4), 'int')
    
    # Call to get_lapack_funcs(...): (line 1180)
    # Processing the call arguments (line 1180)
    
    # Obtaining an instance of the builtin type 'tuple' (line 1180)
    tuple_12532 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1180, 50), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 1180)
    # Adding element type (line 1180)
    # Getting the type of 'driver' (line 1180)
    driver_12533 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1180, 50), 'driver', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1180, 50), tuple_12532, driver_12533)
    # Adding element type (line 1180)
    str_12534 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1181, 49), 'str', '%s_lwork')
    # Getting the type of 'driver' (line 1181)
    driver_12535 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1181, 62), 'driver', False)
    # Applying the binary operator '%' (line 1181)
    result_mod_12536 = python_operator(stypy.reporting.localization.Localization(__file__, 1181, 49), '%', str_12534, driver_12535)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1180, 50), tuple_12532, result_mod_12536)
    
    
    # Obtaining an instance of the builtin type 'tuple' (line 1182)
    tuple_12537 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1182, 50), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 1182)
    # Adding element type (line 1182)
    # Getting the type of 'a1' (line 1182)
    a1_12538 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1182, 50), 'a1', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1182, 50), tuple_12537, a1_12538)
    # Adding element type (line 1182)
    # Getting the type of 'b1' (line 1182)
    b1_12539 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1182, 54), 'b1', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1182, 50), tuple_12537, b1_12539)
    
    # Processing the call keyword arguments (line 1180)
    kwargs_12540 = {}
    # Getting the type of 'get_lapack_funcs' (line 1180)
    get_lapack_funcs_12531 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1180, 32), 'get_lapack_funcs', False)
    # Calling get_lapack_funcs(args, kwargs) (line 1180)
    get_lapack_funcs_call_result_12541 = invoke(stypy.reporting.localization.Localization(__file__, 1180, 32), get_lapack_funcs_12531, *[tuple_12532, tuple_12537], **kwargs_12540)
    
    # Obtaining the member '__getitem__' of a type (line 1180)
    getitem___12542 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1180, 4), get_lapack_funcs_call_result_12541, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1180)
    subscript_call_result_12543 = invoke(stypy.reporting.localization.Localization(__file__, 1180, 4), getitem___12542, int_12530)
    
    # Assigning a type to the variable 'tuple_var_assignment_10101' (line 1180)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1180, 4), 'tuple_var_assignment_10101', subscript_call_result_12543)
    
    # Assigning a Subscript to a Name (line 1180):
    
    # Obtaining the type of the subscript
    int_12544 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1180, 4), 'int')
    
    # Call to get_lapack_funcs(...): (line 1180)
    # Processing the call arguments (line 1180)
    
    # Obtaining an instance of the builtin type 'tuple' (line 1180)
    tuple_12546 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1180, 50), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 1180)
    # Adding element type (line 1180)
    # Getting the type of 'driver' (line 1180)
    driver_12547 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1180, 50), 'driver', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1180, 50), tuple_12546, driver_12547)
    # Adding element type (line 1180)
    str_12548 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1181, 49), 'str', '%s_lwork')
    # Getting the type of 'driver' (line 1181)
    driver_12549 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1181, 62), 'driver', False)
    # Applying the binary operator '%' (line 1181)
    result_mod_12550 = python_operator(stypy.reporting.localization.Localization(__file__, 1181, 49), '%', str_12548, driver_12549)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1180, 50), tuple_12546, result_mod_12550)
    
    
    # Obtaining an instance of the builtin type 'tuple' (line 1182)
    tuple_12551 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1182, 50), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 1182)
    # Adding element type (line 1182)
    # Getting the type of 'a1' (line 1182)
    a1_12552 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1182, 50), 'a1', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1182, 50), tuple_12551, a1_12552)
    # Adding element type (line 1182)
    # Getting the type of 'b1' (line 1182)
    b1_12553 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1182, 54), 'b1', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1182, 50), tuple_12551, b1_12553)
    
    # Processing the call keyword arguments (line 1180)
    kwargs_12554 = {}
    # Getting the type of 'get_lapack_funcs' (line 1180)
    get_lapack_funcs_12545 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1180, 32), 'get_lapack_funcs', False)
    # Calling get_lapack_funcs(args, kwargs) (line 1180)
    get_lapack_funcs_call_result_12555 = invoke(stypy.reporting.localization.Localization(__file__, 1180, 32), get_lapack_funcs_12545, *[tuple_12546, tuple_12551], **kwargs_12554)
    
    # Obtaining the member '__getitem__' of a type (line 1180)
    getitem___12556 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1180, 4), get_lapack_funcs_call_result_12555, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1180)
    subscript_call_result_12557 = invoke(stypy.reporting.localization.Localization(__file__, 1180, 4), getitem___12556, int_12544)
    
    # Assigning a type to the variable 'tuple_var_assignment_10102' (line 1180)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1180, 4), 'tuple_var_assignment_10102', subscript_call_result_12557)
    
    # Assigning a Name to a Name (line 1180):
    # Getting the type of 'tuple_var_assignment_10101' (line 1180)
    tuple_var_assignment_10101_12558 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1180, 4), 'tuple_var_assignment_10101')
    # Assigning a type to the variable 'lapack_func' (line 1180)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1180, 4), 'lapack_func', tuple_var_assignment_10101_12558)
    
    # Assigning a Name to a Name (line 1180):
    # Getting the type of 'tuple_var_assignment_10102' (line 1180)
    tuple_var_assignment_10102_12559 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1180, 4), 'tuple_var_assignment_10102')
    # Assigning a type to the variable 'lapack_lwork' (line 1180)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1180, 17), 'lapack_lwork', tuple_var_assignment_10102_12559)
    
    # Assigning a IfExp to a Name (line 1183):
    
    # Assigning a IfExp to a Name (line 1183):
    
    
    # Getting the type of 'lapack_func' (line 1183)
    lapack_func_12560 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1183, 25), 'lapack_func')
    # Obtaining the member 'dtype' of a type (line 1183)
    dtype_12561 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1183, 25), lapack_func_12560, 'dtype')
    # Obtaining the member 'kind' of a type (line 1183)
    kind_12562 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1183, 25), dtype_12561, 'kind')
    str_12563 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1183, 51), 'str', 'f')
    # Applying the binary operator '==' (line 1183)
    result_eq_12564 = python_operator(stypy.reporting.localization.Localization(__file__, 1183, 25), '==', kind_12562, str_12563)
    
    # Testing the type of an if expression (line 1183)
    is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1183, 16), result_eq_12564)
    # SSA begins for if expression (line 1183)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if expression')
    # Getting the type of 'True' (line 1183)
    True_12565 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1183, 16), 'True')
    # SSA branch for the else part of an if expression (line 1183)
    module_type_store.open_ssa_branch('if expression else')
    # Getting the type of 'False' (line 1183)
    False_12566 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1183, 61), 'False')
    # SSA join for if expression (line 1183)
    module_type_store = module_type_store.join_ssa_context()
    if_exp_12567 = union_type.UnionType.add(True_12565, False_12566)
    
    # Assigning a type to the variable 'real_data' (line 1183)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1183, 4), 'real_data', if_exp_12567)
    
    
    # Getting the type of 'm' (line 1185)
    m_12568 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1185, 7), 'm')
    # Getting the type of 'n' (line 1185)
    n_12569 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1185, 11), 'n')
    # Applying the binary operator '<' (line 1185)
    result_lt_12570 = python_operator(stypy.reporting.localization.Localization(__file__, 1185, 7), '<', m_12568, n_12569)
    
    # Testing the type of an if condition (line 1185)
    if_condition_12571 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1185, 4), result_lt_12570)
    # Assigning a type to the variable 'if_condition_12571' (line 1185)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1185, 4), 'if_condition_12571', if_condition_12571)
    # SSA begins for if statement (line 1185)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    
    
    # Call to len(...): (line 1188)
    # Processing the call arguments (line 1188)
    # Getting the type of 'b1' (line 1188)
    b1_12573 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1188, 15), 'b1', False)
    # Obtaining the member 'shape' of a type (line 1188)
    shape_12574 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1188, 15), b1_12573, 'shape')
    # Processing the call keyword arguments (line 1188)
    kwargs_12575 = {}
    # Getting the type of 'len' (line 1188)
    len_12572 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1188, 11), 'len', False)
    # Calling len(args, kwargs) (line 1188)
    len_call_result_12576 = invoke(stypy.reporting.localization.Localization(__file__, 1188, 11), len_12572, *[shape_12574], **kwargs_12575)
    
    int_12577 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1188, 28), 'int')
    # Applying the binary operator '==' (line 1188)
    result_eq_12578 = python_operator(stypy.reporting.localization.Localization(__file__, 1188, 11), '==', len_call_result_12576, int_12577)
    
    # Testing the type of an if condition (line 1188)
    if_condition_12579 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1188, 8), result_eq_12578)
    # Assigning a type to the variable 'if_condition_12579' (line 1188)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1188, 8), 'if_condition_12579', if_condition_12579)
    # SSA begins for if statement (line 1188)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 1189):
    
    # Assigning a Call to a Name (line 1189):
    
    # Call to zeros(...): (line 1189)
    # Processing the call arguments (line 1189)
    
    # Obtaining an instance of the builtin type 'tuple' (line 1189)
    tuple_12582 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1189, 27), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 1189)
    # Adding element type (line 1189)
    # Getting the type of 'n' (line 1189)
    n_12583 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1189, 27), 'n', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1189, 27), tuple_12582, n_12583)
    # Adding element type (line 1189)
    # Getting the type of 'nrhs' (line 1189)
    nrhs_12584 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1189, 30), 'nrhs', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1189, 27), tuple_12582, nrhs_12584)
    
    # Processing the call keyword arguments (line 1189)
    # Getting the type of 'lapack_func' (line 1189)
    lapack_func_12585 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1189, 43), 'lapack_func', False)
    # Obtaining the member 'dtype' of a type (line 1189)
    dtype_12586 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1189, 43), lapack_func_12585, 'dtype')
    keyword_12587 = dtype_12586
    kwargs_12588 = {'dtype': keyword_12587}
    # Getting the type of 'np' (line 1189)
    np_12580 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1189, 17), 'np', False)
    # Obtaining the member 'zeros' of a type (line 1189)
    zeros_12581 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1189, 17), np_12580, 'zeros')
    # Calling zeros(args, kwargs) (line 1189)
    zeros_call_result_12589 = invoke(stypy.reporting.localization.Localization(__file__, 1189, 17), zeros_12581, *[tuple_12582], **kwargs_12588)
    
    # Assigning a type to the variable 'b2' (line 1189)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1189, 12), 'b2', zeros_call_result_12589)
    
    # Assigning a Name to a Subscript (line 1190):
    
    # Assigning a Name to a Subscript (line 1190):
    # Getting the type of 'b1' (line 1190)
    b1_12590 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1190, 24), 'b1')
    # Getting the type of 'b2' (line 1190)
    b2_12591 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1190, 12), 'b2')
    # Getting the type of 'm' (line 1190)
    m_12592 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1190, 16), 'm')
    slice_12593 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 1190, 12), None, m_12592, None)
    slice_12594 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 1190, 12), None, None, None)
    # Storing an element on a container (line 1190)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1190, 12), b2_12591, ((slice_12593, slice_12594), b1_12590))
    # SSA branch for the else part of an if statement (line 1188)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Call to a Name (line 1192):
    
    # Assigning a Call to a Name (line 1192):
    
    # Call to zeros(...): (line 1192)
    # Processing the call arguments (line 1192)
    # Getting the type of 'n' (line 1192)
    n_12597 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1192, 26), 'n', False)
    # Processing the call keyword arguments (line 1192)
    # Getting the type of 'lapack_func' (line 1192)
    lapack_func_12598 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1192, 35), 'lapack_func', False)
    # Obtaining the member 'dtype' of a type (line 1192)
    dtype_12599 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1192, 35), lapack_func_12598, 'dtype')
    keyword_12600 = dtype_12599
    kwargs_12601 = {'dtype': keyword_12600}
    # Getting the type of 'np' (line 1192)
    np_12595 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1192, 17), 'np', False)
    # Obtaining the member 'zeros' of a type (line 1192)
    zeros_12596 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1192, 17), np_12595, 'zeros')
    # Calling zeros(args, kwargs) (line 1192)
    zeros_call_result_12602 = invoke(stypy.reporting.localization.Localization(__file__, 1192, 17), zeros_12596, *[n_12597], **kwargs_12601)
    
    # Assigning a type to the variable 'b2' (line 1192)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1192, 12), 'b2', zeros_call_result_12602)
    
    # Assigning a Name to a Subscript (line 1193):
    
    # Assigning a Name to a Subscript (line 1193):
    # Getting the type of 'b1' (line 1193)
    b1_12603 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1193, 21), 'b1')
    # Getting the type of 'b2' (line 1193)
    b2_12604 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1193, 12), 'b2')
    # Getting the type of 'm' (line 1193)
    m_12605 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1193, 16), 'm')
    slice_12606 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 1193, 12), None, m_12605, None)
    # Storing an element on a container (line 1193)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1193, 12), b2_12604, (slice_12606, b1_12603))
    # SSA join for if statement (line 1188)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Name to a Name (line 1194):
    
    # Assigning a Name to a Name (line 1194):
    # Getting the type of 'b2' (line 1194)
    b2_12607 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1194, 13), 'b2')
    # Assigning a type to the variable 'b1' (line 1194)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1194, 8), 'b1', b2_12607)
    # SSA join for if statement (line 1185)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a BoolOp to a Name (line 1196):
    
    # Assigning a BoolOp to a Name (line 1196):
    
    # Evaluating a boolean operation
    # Getting the type of 'overwrite_a' (line 1196)
    overwrite_a_12608 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1196, 18), 'overwrite_a')
    
    # Call to _datacopied(...): (line 1196)
    # Processing the call arguments (line 1196)
    # Getting the type of 'a1' (line 1196)
    a1_12610 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1196, 45), 'a1', False)
    # Getting the type of 'a' (line 1196)
    a_12611 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1196, 49), 'a', False)
    # Processing the call keyword arguments (line 1196)
    kwargs_12612 = {}
    # Getting the type of '_datacopied' (line 1196)
    _datacopied_12609 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1196, 33), '_datacopied', False)
    # Calling _datacopied(args, kwargs) (line 1196)
    _datacopied_call_result_12613 = invoke(stypy.reporting.localization.Localization(__file__, 1196, 33), _datacopied_12609, *[a1_12610, a_12611], **kwargs_12612)
    
    # Applying the binary operator 'or' (line 1196)
    result_or_keyword_12614 = python_operator(stypy.reporting.localization.Localization(__file__, 1196, 18), 'or', overwrite_a_12608, _datacopied_call_result_12613)
    
    # Assigning a type to the variable 'overwrite_a' (line 1196)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1196, 4), 'overwrite_a', result_or_keyword_12614)
    
    # Assigning a BoolOp to a Name (line 1197):
    
    # Assigning a BoolOp to a Name (line 1197):
    
    # Evaluating a boolean operation
    # Getting the type of 'overwrite_b' (line 1197)
    overwrite_b_12615 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1197, 18), 'overwrite_b')
    
    # Call to _datacopied(...): (line 1197)
    # Processing the call arguments (line 1197)
    # Getting the type of 'b1' (line 1197)
    b1_12617 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1197, 45), 'b1', False)
    # Getting the type of 'b' (line 1197)
    b_12618 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1197, 49), 'b', False)
    # Processing the call keyword arguments (line 1197)
    kwargs_12619 = {}
    # Getting the type of '_datacopied' (line 1197)
    _datacopied_12616 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1197, 33), '_datacopied', False)
    # Calling _datacopied(args, kwargs) (line 1197)
    _datacopied_call_result_12620 = invoke(stypy.reporting.localization.Localization(__file__, 1197, 33), _datacopied_12616, *[b1_12617, b_12618], **kwargs_12619)
    
    # Applying the binary operator 'or' (line 1197)
    result_or_keyword_12621 = python_operator(stypy.reporting.localization.Localization(__file__, 1197, 18), 'or', overwrite_b_12615, _datacopied_call_result_12620)
    
    # Assigning a type to the variable 'overwrite_b' (line 1197)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1197, 4), 'overwrite_b', result_or_keyword_12621)
    
    # Type idiom detected: calculating its left and rigth part (line 1199)
    # Getting the type of 'cond' (line 1199)
    cond_12622 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1199, 7), 'cond')
    # Getting the type of 'None' (line 1199)
    None_12623 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1199, 15), 'None')
    
    (may_be_12624, more_types_in_union_12625) = may_be_none(cond_12622, None_12623)

    if may_be_12624:

        if more_types_in_union_12625:
            # Runtime conditional SSA (line 1199)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a Attribute to a Name (line 1200):
        
        # Assigning a Attribute to a Name (line 1200):
        
        # Call to finfo(...): (line 1200)
        # Processing the call arguments (line 1200)
        # Getting the type of 'lapack_func' (line 1200)
        lapack_func_12628 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1200, 24), 'lapack_func', False)
        # Obtaining the member 'dtype' of a type (line 1200)
        dtype_12629 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1200, 24), lapack_func_12628, 'dtype')
        # Processing the call keyword arguments (line 1200)
        kwargs_12630 = {}
        # Getting the type of 'np' (line 1200)
        np_12626 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1200, 15), 'np', False)
        # Obtaining the member 'finfo' of a type (line 1200)
        finfo_12627 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1200, 15), np_12626, 'finfo')
        # Calling finfo(args, kwargs) (line 1200)
        finfo_call_result_12631 = invoke(stypy.reporting.localization.Localization(__file__, 1200, 15), finfo_12627, *[dtype_12629], **kwargs_12630)
        
        # Obtaining the member 'eps' of a type (line 1200)
        eps_12632 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1200, 15), finfo_call_result_12631, 'eps')
        # Assigning a type to the variable 'cond' (line 1200)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1200, 8), 'cond', eps_12632)

        if more_types_in_union_12625:
            # SSA join for if statement (line 1199)
            module_type_store = module_type_store.join_ssa_context()


    
    
    
    # Getting the type of 'driver' (line 1202)
    driver_12633 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1202, 7), 'driver')
    
    # Obtaining an instance of the builtin type 'tuple' (line 1202)
    tuple_12634 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1202, 18), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 1202)
    # Adding element type (line 1202)
    str_12635 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1202, 18), 'str', 'gelss')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1202, 18), tuple_12634, str_12635)
    # Adding element type (line 1202)
    str_12636 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1202, 27), 'str', 'gelsd')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1202, 18), tuple_12634, str_12636)
    
    # Applying the binary operator 'in' (line 1202)
    result_contains_12637 = python_operator(stypy.reporting.localization.Localization(__file__, 1202, 7), 'in', driver_12633, tuple_12634)
    
    # Testing the type of an if condition (line 1202)
    if_condition_12638 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1202, 4), result_contains_12637)
    # Assigning a type to the variable 'if_condition_12638' (line 1202)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1202, 4), 'if_condition_12638', if_condition_12638)
    # SSA begins for if statement (line 1202)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    
    # Getting the type of 'driver' (line 1203)
    driver_12639 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1203, 11), 'driver')
    str_12640 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1203, 21), 'str', 'gelss')
    # Applying the binary operator '==' (line 1203)
    result_eq_12641 = python_operator(stypy.reporting.localization.Localization(__file__, 1203, 11), '==', driver_12639, str_12640)
    
    # Testing the type of an if condition (line 1203)
    if_condition_12642 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1203, 8), result_eq_12641)
    # Assigning a type to the variable 'if_condition_12642' (line 1203)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1203, 8), 'if_condition_12642', if_condition_12642)
    # SSA begins for if statement (line 1203)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 1204):
    
    # Assigning a Call to a Name (line 1204):
    
    # Call to _compute_lwork(...): (line 1204)
    # Processing the call arguments (line 1204)
    # Getting the type of 'lapack_lwork' (line 1204)
    lapack_lwork_12644 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1204, 35), 'lapack_lwork', False)
    # Getting the type of 'm' (line 1204)
    m_12645 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1204, 49), 'm', False)
    # Getting the type of 'n' (line 1204)
    n_12646 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1204, 52), 'n', False)
    # Getting the type of 'nrhs' (line 1204)
    nrhs_12647 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1204, 55), 'nrhs', False)
    # Getting the type of 'cond' (line 1204)
    cond_12648 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1204, 61), 'cond', False)
    # Processing the call keyword arguments (line 1204)
    kwargs_12649 = {}
    # Getting the type of '_compute_lwork' (line 1204)
    _compute_lwork_12643 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1204, 20), '_compute_lwork', False)
    # Calling _compute_lwork(args, kwargs) (line 1204)
    _compute_lwork_call_result_12650 = invoke(stypy.reporting.localization.Localization(__file__, 1204, 20), _compute_lwork_12643, *[lapack_lwork_12644, m_12645, n_12646, nrhs_12647, cond_12648], **kwargs_12649)
    
    # Assigning a type to the variable 'lwork' (line 1204)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1204, 12), 'lwork', _compute_lwork_call_result_12650)
    
    # Assigning a Call to a Tuple (line 1205):
    
    # Assigning a Subscript to a Name (line 1205):
    
    # Obtaining the type of the subscript
    int_12651 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1205, 12), 'int')
    
    # Call to lapack_func(...): (line 1205)
    # Processing the call arguments (line 1205)
    # Getting the type of 'a1' (line 1205)
    a1_12653 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1205, 52), 'a1', False)
    # Getting the type of 'b1' (line 1205)
    b1_12654 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1205, 56), 'b1', False)
    # Getting the type of 'cond' (line 1205)
    cond_12655 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1205, 60), 'cond', False)
    # Getting the type of 'lwork' (line 1205)
    lwork_12656 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1205, 66), 'lwork', False)
    # Processing the call keyword arguments (line 1205)
    # Getting the type of 'overwrite_a' (line 1206)
    overwrite_a_12657 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1206, 64), 'overwrite_a', False)
    keyword_12658 = overwrite_a_12657
    # Getting the type of 'overwrite_b' (line 1207)
    overwrite_b_12659 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1207, 64), 'overwrite_b', False)
    keyword_12660 = overwrite_b_12659
    kwargs_12661 = {'overwrite_a': keyword_12658, 'overwrite_b': keyword_12660}
    # Getting the type of 'lapack_func' (line 1205)
    lapack_func_12652 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1205, 40), 'lapack_func', False)
    # Calling lapack_func(args, kwargs) (line 1205)
    lapack_func_call_result_12662 = invoke(stypy.reporting.localization.Localization(__file__, 1205, 40), lapack_func_12652, *[a1_12653, b1_12654, cond_12655, lwork_12656], **kwargs_12661)
    
    # Obtaining the member '__getitem__' of a type (line 1205)
    getitem___12663 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1205, 12), lapack_func_call_result_12662, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1205)
    subscript_call_result_12664 = invoke(stypy.reporting.localization.Localization(__file__, 1205, 12), getitem___12663, int_12651)
    
    # Assigning a type to the variable 'tuple_var_assignment_10103' (line 1205)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1205, 12), 'tuple_var_assignment_10103', subscript_call_result_12664)
    
    # Assigning a Subscript to a Name (line 1205):
    
    # Obtaining the type of the subscript
    int_12665 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1205, 12), 'int')
    
    # Call to lapack_func(...): (line 1205)
    # Processing the call arguments (line 1205)
    # Getting the type of 'a1' (line 1205)
    a1_12667 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1205, 52), 'a1', False)
    # Getting the type of 'b1' (line 1205)
    b1_12668 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1205, 56), 'b1', False)
    # Getting the type of 'cond' (line 1205)
    cond_12669 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1205, 60), 'cond', False)
    # Getting the type of 'lwork' (line 1205)
    lwork_12670 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1205, 66), 'lwork', False)
    # Processing the call keyword arguments (line 1205)
    # Getting the type of 'overwrite_a' (line 1206)
    overwrite_a_12671 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1206, 64), 'overwrite_a', False)
    keyword_12672 = overwrite_a_12671
    # Getting the type of 'overwrite_b' (line 1207)
    overwrite_b_12673 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1207, 64), 'overwrite_b', False)
    keyword_12674 = overwrite_b_12673
    kwargs_12675 = {'overwrite_a': keyword_12672, 'overwrite_b': keyword_12674}
    # Getting the type of 'lapack_func' (line 1205)
    lapack_func_12666 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1205, 40), 'lapack_func', False)
    # Calling lapack_func(args, kwargs) (line 1205)
    lapack_func_call_result_12676 = invoke(stypy.reporting.localization.Localization(__file__, 1205, 40), lapack_func_12666, *[a1_12667, b1_12668, cond_12669, lwork_12670], **kwargs_12675)
    
    # Obtaining the member '__getitem__' of a type (line 1205)
    getitem___12677 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1205, 12), lapack_func_call_result_12676, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1205)
    subscript_call_result_12678 = invoke(stypy.reporting.localization.Localization(__file__, 1205, 12), getitem___12677, int_12665)
    
    # Assigning a type to the variable 'tuple_var_assignment_10104' (line 1205)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1205, 12), 'tuple_var_assignment_10104', subscript_call_result_12678)
    
    # Assigning a Subscript to a Name (line 1205):
    
    # Obtaining the type of the subscript
    int_12679 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1205, 12), 'int')
    
    # Call to lapack_func(...): (line 1205)
    # Processing the call arguments (line 1205)
    # Getting the type of 'a1' (line 1205)
    a1_12681 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1205, 52), 'a1', False)
    # Getting the type of 'b1' (line 1205)
    b1_12682 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1205, 56), 'b1', False)
    # Getting the type of 'cond' (line 1205)
    cond_12683 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1205, 60), 'cond', False)
    # Getting the type of 'lwork' (line 1205)
    lwork_12684 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1205, 66), 'lwork', False)
    # Processing the call keyword arguments (line 1205)
    # Getting the type of 'overwrite_a' (line 1206)
    overwrite_a_12685 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1206, 64), 'overwrite_a', False)
    keyword_12686 = overwrite_a_12685
    # Getting the type of 'overwrite_b' (line 1207)
    overwrite_b_12687 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1207, 64), 'overwrite_b', False)
    keyword_12688 = overwrite_b_12687
    kwargs_12689 = {'overwrite_a': keyword_12686, 'overwrite_b': keyword_12688}
    # Getting the type of 'lapack_func' (line 1205)
    lapack_func_12680 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1205, 40), 'lapack_func', False)
    # Calling lapack_func(args, kwargs) (line 1205)
    lapack_func_call_result_12690 = invoke(stypy.reporting.localization.Localization(__file__, 1205, 40), lapack_func_12680, *[a1_12681, b1_12682, cond_12683, lwork_12684], **kwargs_12689)
    
    # Obtaining the member '__getitem__' of a type (line 1205)
    getitem___12691 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1205, 12), lapack_func_call_result_12690, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1205)
    subscript_call_result_12692 = invoke(stypy.reporting.localization.Localization(__file__, 1205, 12), getitem___12691, int_12679)
    
    # Assigning a type to the variable 'tuple_var_assignment_10105' (line 1205)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1205, 12), 'tuple_var_assignment_10105', subscript_call_result_12692)
    
    # Assigning a Subscript to a Name (line 1205):
    
    # Obtaining the type of the subscript
    int_12693 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1205, 12), 'int')
    
    # Call to lapack_func(...): (line 1205)
    # Processing the call arguments (line 1205)
    # Getting the type of 'a1' (line 1205)
    a1_12695 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1205, 52), 'a1', False)
    # Getting the type of 'b1' (line 1205)
    b1_12696 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1205, 56), 'b1', False)
    # Getting the type of 'cond' (line 1205)
    cond_12697 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1205, 60), 'cond', False)
    # Getting the type of 'lwork' (line 1205)
    lwork_12698 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1205, 66), 'lwork', False)
    # Processing the call keyword arguments (line 1205)
    # Getting the type of 'overwrite_a' (line 1206)
    overwrite_a_12699 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1206, 64), 'overwrite_a', False)
    keyword_12700 = overwrite_a_12699
    # Getting the type of 'overwrite_b' (line 1207)
    overwrite_b_12701 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1207, 64), 'overwrite_b', False)
    keyword_12702 = overwrite_b_12701
    kwargs_12703 = {'overwrite_a': keyword_12700, 'overwrite_b': keyword_12702}
    # Getting the type of 'lapack_func' (line 1205)
    lapack_func_12694 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1205, 40), 'lapack_func', False)
    # Calling lapack_func(args, kwargs) (line 1205)
    lapack_func_call_result_12704 = invoke(stypy.reporting.localization.Localization(__file__, 1205, 40), lapack_func_12694, *[a1_12695, b1_12696, cond_12697, lwork_12698], **kwargs_12703)
    
    # Obtaining the member '__getitem__' of a type (line 1205)
    getitem___12705 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1205, 12), lapack_func_call_result_12704, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1205)
    subscript_call_result_12706 = invoke(stypy.reporting.localization.Localization(__file__, 1205, 12), getitem___12705, int_12693)
    
    # Assigning a type to the variable 'tuple_var_assignment_10106' (line 1205)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1205, 12), 'tuple_var_assignment_10106', subscript_call_result_12706)
    
    # Assigning a Subscript to a Name (line 1205):
    
    # Obtaining the type of the subscript
    int_12707 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1205, 12), 'int')
    
    # Call to lapack_func(...): (line 1205)
    # Processing the call arguments (line 1205)
    # Getting the type of 'a1' (line 1205)
    a1_12709 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1205, 52), 'a1', False)
    # Getting the type of 'b1' (line 1205)
    b1_12710 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1205, 56), 'b1', False)
    # Getting the type of 'cond' (line 1205)
    cond_12711 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1205, 60), 'cond', False)
    # Getting the type of 'lwork' (line 1205)
    lwork_12712 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1205, 66), 'lwork', False)
    # Processing the call keyword arguments (line 1205)
    # Getting the type of 'overwrite_a' (line 1206)
    overwrite_a_12713 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1206, 64), 'overwrite_a', False)
    keyword_12714 = overwrite_a_12713
    # Getting the type of 'overwrite_b' (line 1207)
    overwrite_b_12715 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1207, 64), 'overwrite_b', False)
    keyword_12716 = overwrite_b_12715
    kwargs_12717 = {'overwrite_a': keyword_12714, 'overwrite_b': keyword_12716}
    # Getting the type of 'lapack_func' (line 1205)
    lapack_func_12708 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1205, 40), 'lapack_func', False)
    # Calling lapack_func(args, kwargs) (line 1205)
    lapack_func_call_result_12718 = invoke(stypy.reporting.localization.Localization(__file__, 1205, 40), lapack_func_12708, *[a1_12709, b1_12710, cond_12711, lwork_12712], **kwargs_12717)
    
    # Obtaining the member '__getitem__' of a type (line 1205)
    getitem___12719 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1205, 12), lapack_func_call_result_12718, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1205)
    subscript_call_result_12720 = invoke(stypy.reporting.localization.Localization(__file__, 1205, 12), getitem___12719, int_12707)
    
    # Assigning a type to the variable 'tuple_var_assignment_10107' (line 1205)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1205, 12), 'tuple_var_assignment_10107', subscript_call_result_12720)
    
    # Assigning a Subscript to a Name (line 1205):
    
    # Obtaining the type of the subscript
    int_12721 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1205, 12), 'int')
    
    # Call to lapack_func(...): (line 1205)
    # Processing the call arguments (line 1205)
    # Getting the type of 'a1' (line 1205)
    a1_12723 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1205, 52), 'a1', False)
    # Getting the type of 'b1' (line 1205)
    b1_12724 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1205, 56), 'b1', False)
    # Getting the type of 'cond' (line 1205)
    cond_12725 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1205, 60), 'cond', False)
    # Getting the type of 'lwork' (line 1205)
    lwork_12726 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1205, 66), 'lwork', False)
    # Processing the call keyword arguments (line 1205)
    # Getting the type of 'overwrite_a' (line 1206)
    overwrite_a_12727 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1206, 64), 'overwrite_a', False)
    keyword_12728 = overwrite_a_12727
    # Getting the type of 'overwrite_b' (line 1207)
    overwrite_b_12729 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1207, 64), 'overwrite_b', False)
    keyword_12730 = overwrite_b_12729
    kwargs_12731 = {'overwrite_a': keyword_12728, 'overwrite_b': keyword_12730}
    # Getting the type of 'lapack_func' (line 1205)
    lapack_func_12722 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1205, 40), 'lapack_func', False)
    # Calling lapack_func(args, kwargs) (line 1205)
    lapack_func_call_result_12732 = invoke(stypy.reporting.localization.Localization(__file__, 1205, 40), lapack_func_12722, *[a1_12723, b1_12724, cond_12725, lwork_12726], **kwargs_12731)
    
    # Obtaining the member '__getitem__' of a type (line 1205)
    getitem___12733 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1205, 12), lapack_func_call_result_12732, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1205)
    subscript_call_result_12734 = invoke(stypy.reporting.localization.Localization(__file__, 1205, 12), getitem___12733, int_12721)
    
    # Assigning a type to the variable 'tuple_var_assignment_10108' (line 1205)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1205, 12), 'tuple_var_assignment_10108', subscript_call_result_12734)
    
    # Assigning a Name to a Name (line 1205):
    # Getting the type of 'tuple_var_assignment_10103' (line 1205)
    tuple_var_assignment_10103_12735 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1205, 12), 'tuple_var_assignment_10103')
    # Assigning a type to the variable 'v' (line 1205)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1205, 12), 'v', tuple_var_assignment_10103_12735)
    
    # Assigning a Name to a Name (line 1205):
    # Getting the type of 'tuple_var_assignment_10104' (line 1205)
    tuple_var_assignment_10104_12736 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1205, 12), 'tuple_var_assignment_10104')
    # Assigning a type to the variable 'x' (line 1205)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1205, 15), 'x', tuple_var_assignment_10104_12736)
    
    # Assigning a Name to a Name (line 1205):
    # Getting the type of 'tuple_var_assignment_10105' (line 1205)
    tuple_var_assignment_10105_12737 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1205, 12), 'tuple_var_assignment_10105')
    # Assigning a type to the variable 's' (line 1205)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1205, 18), 's', tuple_var_assignment_10105_12737)
    
    # Assigning a Name to a Name (line 1205):
    # Getting the type of 'tuple_var_assignment_10106' (line 1205)
    tuple_var_assignment_10106_12738 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1205, 12), 'tuple_var_assignment_10106')
    # Assigning a type to the variable 'rank' (line 1205)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1205, 21), 'rank', tuple_var_assignment_10106_12738)
    
    # Assigning a Name to a Name (line 1205):
    # Getting the type of 'tuple_var_assignment_10107' (line 1205)
    tuple_var_assignment_10107_12739 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1205, 12), 'tuple_var_assignment_10107')
    # Assigning a type to the variable 'work' (line 1205)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1205, 27), 'work', tuple_var_assignment_10107_12739)
    
    # Assigning a Name to a Name (line 1205):
    # Getting the type of 'tuple_var_assignment_10108' (line 1205)
    tuple_var_assignment_10108_12740 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1205, 12), 'tuple_var_assignment_10108')
    # Assigning a type to the variable 'info' (line 1205)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1205, 33), 'info', tuple_var_assignment_10108_12740)
    # SSA branch for the else part of an if statement (line 1203)
    module_type_store.open_ssa_branch('else')
    
    
    # Getting the type of 'driver' (line 1209)
    driver_12741 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1209, 13), 'driver')
    str_12742 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1209, 23), 'str', 'gelsd')
    # Applying the binary operator '==' (line 1209)
    result_eq_12743 = python_operator(stypy.reporting.localization.Localization(__file__, 1209, 13), '==', driver_12741, str_12742)
    
    # Testing the type of an if condition (line 1209)
    if_condition_12744 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1209, 13), result_eq_12743)
    # Assigning a type to the variable 'if_condition_12744' (line 1209)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1209, 13), 'if_condition_12744', if_condition_12744)
    # SSA begins for if statement (line 1209)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Getting the type of 'real_data' (line 1210)
    real_data_12745 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1210, 15), 'real_data')
    # Testing the type of an if condition (line 1210)
    if_condition_12746 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1210, 12), real_data_12745)
    # Assigning a type to the variable 'if_condition_12746' (line 1210)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1210, 12), 'if_condition_12746', if_condition_12746)
    # SSA begins for if statement (line 1210)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Tuple (line 1211):
    
    # Assigning a Subscript to a Name (line 1211):
    
    # Obtaining the type of the subscript
    int_12747 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1211, 16), 'int')
    
    # Call to _compute_lwork(...): (line 1211)
    # Processing the call arguments (line 1211)
    # Getting the type of 'lapack_lwork' (line 1211)
    lapack_lwork_12749 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1211, 46), 'lapack_lwork', False)
    # Getting the type of 'm' (line 1211)
    m_12750 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1211, 60), 'm', False)
    # Getting the type of 'n' (line 1211)
    n_12751 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1211, 63), 'n', False)
    # Getting the type of 'nrhs' (line 1211)
    nrhs_12752 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1211, 66), 'nrhs', False)
    # Getting the type of 'cond' (line 1211)
    cond_12753 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1211, 72), 'cond', False)
    # Processing the call keyword arguments (line 1211)
    kwargs_12754 = {}
    # Getting the type of '_compute_lwork' (line 1211)
    _compute_lwork_12748 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1211, 31), '_compute_lwork', False)
    # Calling _compute_lwork(args, kwargs) (line 1211)
    _compute_lwork_call_result_12755 = invoke(stypy.reporting.localization.Localization(__file__, 1211, 31), _compute_lwork_12748, *[lapack_lwork_12749, m_12750, n_12751, nrhs_12752, cond_12753], **kwargs_12754)
    
    # Obtaining the member '__getitem__' of a type (line 1211)
    getitem___12756 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1211, 16), _compute_lwork_call_result_12755, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1211)
    subscript_call_result_12757 = invoke(stypy.reporting.localization.Localization(__file__, 1211, 16), getitem___12756, int_12747)
    
    # Assigning a type to the variable 'tuple_var_assignment_10109' (line 1211)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1211, 16), 'tuple_var_assignment_10109', subscript_call_result_12757)
    
    # Assigning a Subscript to a Name (line 1211):
    
    # Obtaining the type of the subscript
    int_12758 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1211, 16), 'int')
    
    # Call to _compute_lwork(...): (line 1211)
    # Processing the call arguments (line 1211)
    # Getting the type of 'lapack_lwork' (line 1211)
    lapack_lwork_12760 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1211, 46), 'lapack_lwork', False)
    # Getting the type of 'm' (line 1211)
    m_12761 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1211, 60), 'm', False)
    # Getting the type of 'n' (line 1211)
    n_12762 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1211, 63), 'n', False)
    # Getting the type of 'nrhs' (line 1211)
    nrhs_12763 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1211, 66), 'nrhs', False)
    # Getting the type of 'cond' (line 1211)
    cond_12764 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1211, 72), 'cond', False)
    # Processing the call keyword arguments (line 1211)
    kwargs_12765 = {}
    # Getting the type of '_compute_lwork' (line 1211)
    _compute_lwork_12759 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1211, 31), '_compute_lwork', False)
    # Calling _compute_lwork(args, kwargs) (line 1211)
    _compute_lwork_call_result_12766 = invoke(stypy.reporting.localization.Localization(__file__, 1211, 31), _compute_lwork_12759, *[lapack_lwork_12760, m_12761, n_12762, nrhs_12763, cond_12764], **kwargs_12765)
    
    # Obtaining the member '__getitem__' of a type (line 1211)
    getitem___12767 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1211, 16), _compute_lwork_call_result_12766, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1211)
    subscript_call_result_12768 = invoke(stypy.reporting.localization.Localization(__file__, 1211, 16), getitem___12767, int_12758)
    
    # Assigning a type to the variable 'tuple_var_assignment_10110' (line 1211)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1211, 16), 'tuple_var_assignment_10110', subscript_call_result_12768)
    
    # Assigning a Name to a Name (line 1211):
    # Getting the type of 'tuple_var_assignment_10109' (line 1211)
    tuple_var_assignment_10109_12769 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1211, 16), 'tuple_var_assignment_10109')
    # Assigning a type to the variable 'lwork' (line 1211)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1211, 16), 'lwork', tuple_var_assignment_10109_12769)
    
    # Assigning a Name to a Name (line 1211):
    # Getting the type of 'tuple_var_assignment_10110' (line 1211)
    tuple_var_assignment_10110_12770 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1211, 16), 'tuple_var_assignment_10110')
    # Assigning a type to the variable 'iwork' (line 1211)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1211, 23), 'iwork', tuple_var_assignment_10110_12770)
    
    
    # Getting the type of 'iwork' (line 1212)
    iwork_12771 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1212, 19), 'iwork')
    int_12772 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1212, 28), 'int')
    # Applying the binary operator '==' (line 1212)
    result_eq_12773 = python_operator(stypy.reporting.localization.Localization(__file__, 1212, 19), '==', iwork_12771, int_12772)
    
    # Testing the type of an if condition (line 1212)
    if_condition_12774 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1212, 16), result_eq_12773)
    # Assigning a type to the variable 'if_condition_12774' (line 1212)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1212, 16), 'if_condition_12774', if_condition_12774)
    # SSA begins for if statement (line 1212)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Str to a Name (line 1216):
    
    # Assigning a Str to a Name (line 1216):
    str_12775 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1216, 28), 'str', 'internal gelsd driver lwork query error, required iwork dimension not returned. This is likely the result of LAPACK bug 0038, fixed in LAPACK 3.2.2 (released July 21, 2010). ')
    # Assigning a type to the variable 'mesg' (line 1216)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1216, 20), 'mesg', str_12775)
    
    # Type idiom detected: calculating its left and rigth part (line 1222)
    # Getting the type of 'lapack_driver' (line 1222)
    lapack_driver_12776 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1222, 23), 'lapack_driver')
    # Getting the type of 'None' (line 1222)
    None_12777 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1222, 40), 'None')
    
    (may_be_12778, more_types_in_union_12779) = may_be_none(lapack_driver_12776, None_12777)

    if may_be_12778:

        if more_types_in_union_12779:
            # Runtime conditional SSA (line 1222)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a Str to a Attribute (line 1224):
        
        # Assigning a Str to a Attribute (line 1224):
        str_12780 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1224, 54), 'str', 'gelss')
        # Getting the type of 'lstsq' (line 1224)
        lstsq_12781 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1224, 24), 'lstsq')
        # Setting the type of the member 'default_lapack_driver' of a type (line 1224)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 1224, 24), lstsq_12781, 'default_lapack_driver', str_12780)
        
        # Getting the type of 'mesg' (line 1225)
        mesg_12782 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1225, 24), 'mesg')
        str_12783 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1225, 32), 'str', "Falling back to 'gelss' driver.")
        # Applying the binary operator '+=' (line 1225)
        result_iadd_12784 = python_operator(stypy.reporting.localization.Localization(__file__, 1225, 24), '+=', mesg_12782, str_12783)
        # Assigning a type to the variable 'mesg' (line 1225)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1225, 24), 'mesg', result_iadd_12784)
        
        
        # Call to warn(...): (line 1226)
        # Processing the call arguments (line 1226)
        # Getting the type of 'mesg' (line 1226)
        mesg_12787 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1226, 38), 'mesg', False)
        # Getting the type of 'RuntimeWarning' (line 1226)
        RuntimeWarning_12788 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1226, 44), 'RuntimeWarning', False)
        # Processing the call keyword arguments (line 1226)
        kwargs_12789 = {}
        # Getting the type of 'warnings' (line 1226)
        warnings_12785 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1226, 24), 'warnings', False)
        # Obtaining the member 'warn' of a type (line 1226)
        warn_12786 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1226, 24), warnings_12785, 'warn')
        # Calling warn(args, kwargs) (line 1226)
        warn_call_result_12790 = invoke(stypy.reporting.localization.Localization(__file__, 1226, 24), warn_12786, *[mesg_12787, RuntimeWarning_12788], **kwargs_12789)
        
        
        # Call to lstsq(...): (line 1227)
        # Processing the call arguments (line 1227)
        # Getting the type of 'a' (line 1227)
        a_12792 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1227, 37), 'a', False)
        # Getting the type of 'b' (line 1227)
        b_12793 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1227, 40), 'b', False)
        # Getting the type of 'cond' (line 1227)
        cond_12794 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1227, 43), 'cond', False)
        # Getting the type of 'overwrite_a' (line 1227)
        overwrite_a_12795 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1227, 49), 'overwrite_a', False)
        # Getting the type of 'overwrite_b' (line 1227)
        overwrite_b_12796 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1227, 62), 'overwrite_b', False)
        # Getting the type of 'check_finite' (line 1228)
        check_finite_12797 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1228, 37), 'check_finite', False)
        # Processing the call keyword arguments (line 1227)
        str_12798 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1228, 65), 'str', 'gelss')
        keyword_12799 = str_12798
        kwargs_12800 = {'lapack_driver': keyword_12799}
        # Getting the type of 'lstsq' (line 1227)
        lstsq_12791 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1227, 31), 'lstsq', False)
        # Calling lstsq(args, kwargs) (line 1227)
        lstsq_call_result_12801 = invoke(stypy.reporting.localization.Localization(__file__, 1227, 31), lstsq_12791, *[a_12792, b_12793, cond_12794, overwrite_a_12795, overwrite_b_12796, check_finite_12797], **kwargs_12800)
        
        # Assigning a type to the variable 'stypy_return_type' (line 1227)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1227, 24), 'stypy_return_type', lstsq_call_result_12801)

        if more_types_in_union_12779:
            # SSA join for if statement (line 1222)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Getting the type of 'mesg' (line 1231)
    mesg_12802 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1231, 20), 'mesg')
    str_12803 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1231, 29), 'str', 'Use a different lapack_driver when calling lstsq or upgrade LAPACK.')
    # Applying the binary operator '+=' (line 1231)
    result_iadd_12804 = python_operator(stypy.reporting.localization.Localization(__file__, 1231, 20), '+=', mesg_12802, str_12803)
    # Assigning a type to the variable 'mesg' (line 1231)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1231, 20), 'mesg', result_iadd_12804)
    
    
    # Call to LstsqLapackError(...): (line 1233)
    # Processing the call arguments (line 1233)
    # Getting the type of 'mesg' (line 1233)
    mesg_12806 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1233, 43), 'mesg', False)
    # Processing the call keyword arguments (line 1233)
    kwargs_12807 = {}
    # Getting the type of 'LstsqLapackError' (line 1233)
    LstsqLapackError_12805 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1233, 26), 'LstsqLapackError', False)
    # Calling LstsqLapackError(args, kwargs) (line 1233)
    LstsqLapackError_call_result_12808 = invoke(stypy.reporting.localization.Localization(__file__, 1233, 26), LstsqLapackError_12805, *[mesg_12806], **kwargs_12807)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 1233, 20), LstsqLapackError_call_result_12808, 'raise parameter', BaseException)
    # SSA join for if statement (line 1212)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Tuple (line 1235):
    
    # Assigning a Subscript to a Name (line 1235):
    
    # Obtaining the type of the subscript
    int_12809 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1235, 16), 'int')
    
    # Call to lapack_func(...): (line 1235)
    # Processing the call arguments (line 1235)
    # Getting the type of 'a1' (line 1235)
    a1_12811 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1235, 47), 'a1', False)
    # Getting the type of 'b1' (line 1235)
    b1_12812 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1235, 51), 'b1', False)
    # Getting the type of 'lwork' (line 1235)
    lwork_12813 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1235, 55), 'lwork', False)
    # Getting the type of 'iwork' (line 1236)
    iwork_12814 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1236, 47), 'iwork', False)
    # Getting the type of 'cond' (line 1236)
    cond_12815 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1236, 54), 'cond', False)
    # Getting the type of 'False' (line 1236)
    False_12816 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1236, 60), 'False', False)
    # Getting the type of 'False' (line 1236)
    False_12817 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1236, 67), 'False', False)
    # Processing the call keyword arguments (line 1235)
    kwargs_12818 = {}
    # Getting the type of 'lapack_func' (line 1235)
    lapack_func_12810 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1235, 35), 'lapack_func', False)
    # Calling lapack_func(args, kwargs) (line 1235)
    lapack_func_call_result_12819 = invoke(stypy.reporting.localization.Localization(__file__, 1235, 35), lapack_func_12810, *[a1_12811, b1_12812, lwork_12813, iwork_12814, cond_12815, False_12816, False_12817], **kwargs_12818)
    
    # Obtaining the member '__getitem__' of a type (line 1235)
    getitem___12820 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1235, 16), lapack_func_call_result_12819, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1235)
    subscript_call_result_12821 = invoke(stypy.reporting.localization.Localization(__file__, 1235, 16), getitem___12820, int_12809)
    
    # Assigning a type to the variable 'tuple_var_assignment_10111' (line 1235)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1235, 16), 'tuple_var_assignment_10111', subscript_call_result_12821)
    
    # Assigning a Subscript to a Name (line 1235):
    
    # Obtaining the type of the subscript
    int_12822 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1235, 16), 'int')
    
    # Call to lapack_func(...): (line 1235)
    # Processing the call arguments (line 1235)
    # Getting the type of 'a1' (line 1235)
    a1_12824 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1235, 47), 'a1', False)
    # Getting the type of 'b1' (line 1235)
    b1_12825 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1235, 51), 'b1', False)
    # Getting the type of 'lwork' (line 1235)
    lwork_12826 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1235, 55), 'lwork', False)
    # Getting the type of 'iwork' (line 1236)
    iwork_12827 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1236, 47), 'iwork', False)
    # Getting the type of 'cond' (line 1236)
    cond_12828 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1236, 54), 'cond', False)
    # Getting the type of 'False' (line 1236)
    False_12829 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1236, 60), 'False', False)
    # Getting the type of 'False' (line 1236)
    False_12830 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1236, 67), 'False', False)
    # Processing the call keyword arguments (line 1235)
    kwargs_12831 = {}
    # Getting the type of 'lapack_func' (line 1235)
    lapack_func_12823 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1235, 35), 'lapack_func', False)
    # Calling lapack_func(args, kwargs) (line 1235)
    lapack_func_call_result_12832 = invoke(stypy.reporting.localization.Localization(__file__, 1235, 35), lapack_func_12823, *[a1_12824, b1_12825, lwork_12826, iwork_12827, cond_12828, False_12829, False_12830], **kwargs_12831)
    
    # Obtaining the member '__getitem__' of a type (line 1235)
    getitem___12833 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1235, 16), lapack_func_call_result_12832, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1235)
    subscript_call_result_12834 = invoke(stypy.reporting.localization.Localization(__file__, 1235, 16), getitem___12833, int_12822)
    
    # Assigning a type to the variable 'tuple_var_assignment_10112' (line 1235)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1235, 16), 'tuple_var_assignment_10112', subscript_call_result_12834)
    
    # Assigning a Subscript to a Name (line 1235):
    
    # Obtaining the type of the subscript
    int_12835 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1235, 16), 'int')
    
    # Call to lapack_func(...): (line 1235)
    # Processing the call arguments (line 1235)
    # Getting the type of 'a1' (line 1235)
    a1_12837 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1235, 47), 'a1', False)
    # Getting the type of 'b1' (line 1235)
    b1_12838 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1235, 51), 'b1', False)
    # Getting the type of 'lwork' (line 1235)
    lwork_12839 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1235, 55), 'lwork', False)
    # Getting the type of 'iwork' (line 1236)
    iwork_12840 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1236, 47), 'iwork', False)
    # Getting the type of 'cond' (line 1236)
    cond_12841 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1236, 54), 'cond', False)
    # Getting the type of 'False' (line 1236)
    False_12842 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1236, 60), 'False', False)
    # Getting the type of 'False' (line 1236)
    False_12843 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1236, 67), 'False', False)
    # Processing the call keyword arguments (line 1235)
    kwargs_12844 = {}
    # Getting the type of 'lapack_func' (line 1235)
    lapack_func_12836 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1235, 35), 'lapack_func', False)
    # Calling lapack_func(args, kwargs) (line 1235)
    lapack_func_call_result_12845 = invoke(stypy.reporting.localization.Localization(__file__, 1235, 35), lapack_func_12836, *[a1_12837, b1_12838, lwork_12839, iwork_12840, cond_12841, False_12842, False_12843], **kwargs_12844)
    
    # Obtaining the member '__getitem__' of a type (line 1235)
    getitem___12846 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1235, 16), lapack_func_call_result_12845, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1235)
    subscript_call_result_12847 = invoke(stypy.reporting.localization.Localization(__file__, 1235, 16), getitem___12846, int_12835)
    
    # Assigning a type to the variable 'tuple_var_assignment_10113' (line 1235)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1235, 16), 'tuple_var_assignment_10113', subscript_call_result_12847)
    
    # Assigning a Subscript to a Name (line 1235):
    
    # Obtaining the type of the subscript
    int_12848 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1235, 16), 'int')
    
    # Call to lapack_func(...): (line 1235)
    # Processing the call arguments (line 1235)
    # Getting the type of 'a1' (line 1235)
    a1_12850 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1235, 47), 'a1', False)
    # Getting the type of 'b1' (line 1235)
    b1_12851 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1235, 51), 'b1', False)
    # Getting the type of 'lwork' (line 1235)
    lwork_12852 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1235, 55), 'lwork', False)
    # Getting the type of 'iwork' (line 1236)
    iwork_12853 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1236, 47), 'iwork', False)
    # Getting the type of 'cond' (line 1236)
    cond_12854 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1236, 54), 'cond', False)
    # Getting the type of 'False' (line 1236)
    False_12855 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1236, 60), 'False', False)
    # Getting the type of 'False' (line 1236)
    False_12856 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1236, 67), 'False', False)
    # Processing the call keyword arguments (line 1235)
    kwargs_12857 = {}
    # Getting the type of 'lapack_func' (line 1235)
    lapack_func_12849 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1235, 35), 'lapack_func', False)
    # Calling lapack_func(args, kwargs) (line 1235)
    lapack_func_call_result_12858 = invoke(stypy.reporting.localization.Localization(__file__, 1235, 35), lapack_func_12849, *[a1_12850, b1_12851, lwork_12852, iwork_12853, cond_12854, False_12855, False_12856], **kwargs_12857)
    
    # Obtaining the member '__getitem__' of a type (line 1235)
    getitem___12859 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1235, 16), lapack_func_call_result_12858, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1235)
    subscript_call_result_12860 = invoke(stypy.reporting.localization.Localization(__file__, 1235, 16), getitem___12859, int_12848)
    
    # Assigning a type to the variable 'tuple_var_assignment_10114' (line 1235)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1235, 16), 'tuple_var_assignment_10114', subscript_call_result_12860)
    
    # Assigning a Name to a Name (line 1235):
    # Getting the type of 'tuple_var_assignment_10111' (line 1235)
    tuple_var_assignment_10111_12861 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1235, 16), 'tuple_var_assignment_10111')
    # Assigning a type to the variable 'x' (line 1235)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1235, 16), 'x', tuple_var_assignment_10111_12861)
    
    # Assigning a Name to a Name (line 1235):
    # Getting the type of 'tuple_var_assignment_10112' (line 1235)
    tuple_var_assignment_10112_12862 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1235, 16), 'tuple_var_assignment_10112')
    # Assigning a type to the variable 's' (line 1235)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1235, 19), 's', tuple_var_assignment_10112_12862)
    
    # Assigning a Name to a Name (line 1235):
    # Getting the type of 'tuple_var_assignment_10113' (line 1235)
    tuple_var_assignment_10113_12863 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1235, 16), 'tuple_var_assignment_10113')
    # Assigning a type to the variable 'rank' (line 1235)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1235, 22), 'rank', tuple_var_assignment_10113_12863)
    
    # Assigning a Name to a Name (line 1235):
    # Getting the type of 'tuple_var_assignment_10114' (line 1235)
    tuple_var_assignment_10114_12864 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1235, 16), 'tuple_var_assignment_10114')
    # Assigning a type to the variable 'info' (line 1235)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1235, 28), 'info', tuple_var_assignment_10114_12864)
    # SSA branch for the else part of an if statement (line 1210)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Call to a Tuple (line 1238):
    
    # Assigning a Subscript to a Name (line 1238):
    
    # Obtaining the type of the subscript
    int_12865 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1238, 16), 'int')
    
    # Call to _compute_lwork(...): (line 1238)
    # Processing the call arguments (line 1238)
    # Getting the type of 'lapack_lwork' (line 1238)
    lapack_lwork_12867 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1238, 53), 'lapack_lwork', False)
    # Getting the type of 'm' (line 1238)
    m_12868 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1238, 67), 'm', False)
    # Getting the type of 'n' (line 1238)
    n_12869 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1238, 70), 'n', False)
    # Getting the type of 'nrhs' (line 1239)
    nrhs_12870 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1239, 53), 'nrhs', False)
    # Getting the type of 'cond' (line 1239)
    cond_12871 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1239, 59), 'cond', False)
    # Processing the call keyword arguments (line 1238)
    kwargs_12872 = {}
    # Getting the type of '_compute_lwork' (line 1238)
    _compute_lwork_12866 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1238, 38), '_compute_lwork', False)
    # Calling _compute_lwork(args, kwargs) (line 1238)
    _compute_lwork_call_result_12873 = invoke(stypy.reporting.localization.Localization(__file__, 1238, 38), _compute_lwork_12866, *[lapack_lwork_12867, m_12868, n_12869, nrhs_12870, cond_12871], **kwargs_12872)
    
    # Obtaining the member '__getitem__' of a type (line 1238)
    getitem___12874 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1238, 16), _compute_lwork_call_result_12873, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1238)
    subscript_call_result_12875 = invoke(stypy.reporting.localization.Localization(__file__, 1238, 16), getitem___12874, int_12865)
    
    # Assigning a type to the variable 'tuple_var_assignment_10115' (line 1238)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1238, 16), 'tuple_var_assignment_10115', subscript_call_result_12875)
    
    # Assigning a Subscript to a Name (line 1238):
    
    # Obtaining the type of the subscript
    int_12876 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1238, 16), 'int')
    
    # Call to _compute_lwork(...): (line 1238)
    # Processing the call arguments (line 1238)
    # Getting the type of 'lapack_lwork' (line 1238)
    lapack_lwork_12878 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1238, 53), 'lapack_lwork', False)
    # Getting the type of 'm' (line 1238)
    m_12879 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1238, 67), 'm', False)
    # Getting the type of 'n' (line 1238)
    n_12880 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1238, 70), 'n', False)
    # Getting the type of 'nrhs' (line 1239)
    nrhs_12881 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1239, 53), 'nrhs', False)
    # Getting the type of 'cond' (line 1239)
    cond_12882 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1239, 59), 'cond', False)
    # Processing the call keyword arguments (line 1238)
    kwargs_12883 = {}
    # Getting the type of '_compute_lwork' (line 1238)
    _compute_lwork_12877 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1238, 38), '_compute_lwork', False)
    # Calling _compute_lwork(args, kwargs) (line 1238)
    _compute_lwork_call_result_12884 = invoke(stypy.reporting.localization.Localization(__file__, 1238, 38), _compute_lwork_12877, *[lapack_lwork_12878, m_12879, n_12880, nrhs_12881, cond_12882], **kwargs_12883)
    
    # Obtaining the member '__getitem__' of a type (line 1238)
    getitem___12885 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1238, 16), _compute_lwork_call_result_12884, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1238)
    subscript_call_result_12886 = invoke(stypy.reporting.localization.Localization(__file__, 1238, 16), getitem___12885, int_12876)
    
    # Assigning a type to the variable 'tuple_var_assignment_10116' (line 1238)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1238, 16), 'tuple_var_assignment_10116', subscript_call_result_12886)
    
    # Assigning a Subscript to a Name (line 1238):
    
    # Obtaining the type of the subscript
    int_12887 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1238, 16), 'int')
    
    # Call to _compute_lwork(...): (line 1238)
    # Processing the call arguments (line 1238)
    # Getting the type of 'lapack_lwork' (line 1238)
    lapack_lwork_12889 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1238, 53), 'lapack_lwork', False)
    # Getting the type of 'm' (line 1238)
    m_12890 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1238, 67), 'm', False)
    # Getting the type of 'n' (line 1238)
    n_12891 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1238, 70), 'n', False)
    # Getting the type of 'nrhs' (line 1239)
    nrhs_12892 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1239, 53), 'nrhs', False)
    # Getting the type of 'cond' (line 1239)
    cond_12893 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1239, 59), 'cond', False)
    # Processing the call keyword arguments (line 1238)
    kwargs_12894 = {}
    # Getting the type of '_compute_lwork' (line 1238)
    _compute_lwork_12888 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1238, 38), '_compute_lwork', False)
    # Calling _compute_lwork(args, kwargs) (line 1238)
    _compute_lwork_call_result_12895 = invoke(stypy.reporting.localization.Localization(__file__, 1238, 38), _compute_lwork_12888, *[lapack_lwork_12889, m_12890, n_12891, nrhs_12892, cond_12893], **kwargs_12894)
    
    # Obtaining the member '__getitem__' of a type (line 1238)
    getitem___12896 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1238, 16), _compute_lwork_call_result_12895, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1238)
    subscript_call_result_12897 = invoke(stypy.reporting.localization.Localization(__file__, 1238, 16), getitem___12896, int_12887)
    
    # Assigning a type to the variable 'tuple_var_assignment_10117' (line 1238)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1238, 16), 'tuple_var_assignment_10117', subscript_call_result_12897)
    
    # Assigning a Name to a Name (line 1238):
    # Getting the type of 'tuple_var_assignment_10115' (line 1238)
    tuple_var_assignment_10115_12898 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1238, 16), 'tuple_var_assignment_10115')
    # Assigning a type to the variable 'lwork' (line 1238)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1238, 16), 'lwork', tuple_var_assignment_10115_12898)
    
    # Assigning a Name to a Name (line 1238):
    # Getting the type of 'tuple_var_assignment_10116' (line 1238)
    tuple_var_assignment_10116_12899 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1238, 16), 'tuple_var_assignment_10116')
    # Assigning a type to the variable 'rwork' (line 1238)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1238, 23), 'rwork', tuple_var_assignment_10116_12899)
    
    # Assigning a Name to a Name (line 1238):
    # Getting the type of 'tuple_var_assignment_10117' (line 1238)
    tuple_var_assignment_10117_12900 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1238, 16), 'tuple_var_assignment_10117')
    # Assigning a type to the variable 'iwork' (line 1238)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1238, 30), 'iwork', tuple_var_assignment_10117_12900)
    
    # Assigning a Call to a Tuple (line 1240):
    
    # Assigning a Subscript to a Name (line 1240):
    
    # Obtaining the type of the subscript
    int_12901 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1240, 16), 'int')
    
    # Call to lapack_func(...): (line 1240)
    # Processing the call arguments (line 1240)
    # Getting the type of 'a1' (line 1240)
    a1_12903 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1240, 47), 'a1', False)
    # Getting the type of 'b1' (line 1240)
    b1_12904 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1240, 51), 'b1', False)
    # Getting the type of 'lwork' (line 1240)
    lwork_12905 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1240, 55), 'lwork', False)
    # Getting the type of 'rwork' (line 1240)
    rwork_12906 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1240, 62), 'rwork', False)
    # Getting the type of 'iwork' (line 1240)
    iwork_12907 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1240, 69), 'iwork', False)
    # Getting the type of 'cond' (line 1241)
    cond_12908 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1241, 47), 'cond', False)
    # Getting the type of 'False' (line 1241)
    False_12909 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1241, 53), 'False', False)
    # Getting the type of 'False' (line 1241)
    False_12910 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1241, 60), 'False', False)
    # Processing the call keyword arguments (line 1240)
    kwargs_12911 = {}
    # Getting the type of 'lapack_func' (line 1240)
    lapack_func_12902 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1240, 35), 'lapack_func', False)
    # Calling lapack_func(args, kwargs) (line 1240)
    lapack_func_call_result_12912 = invoke(stypy.reporting.localization.Localization(__file__, 1240, 35), lapack_func_12902, *[a1_12903, b1_12904, lwork_12905, rwork_12906, iwork_12907, cond_12908, False_12909, False_12910], **kwargs_12911)
    
    # Obtaining the member '__getitem__' of a type (line 1240)
    getitem___12913 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1240, 16), lapack_func_call_result_12912, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1240)
    subscript_call_result_12914 = invoke(stypy.reporting.localization.Localization(__file__, 1240, 16), getitem___12913, int_12901)
    
    # Assigning a type to the variable 'tuple_var_assignment_10118' (line 1240)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1240, 16), 'tuple_var_assignment_10118', subscript_call_result_12914)
    
    # Assigning a Subscript to a Name (line 1240):
    
    # Obtaining the type of the subscript
    int_12915 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1240, 16), 'int')
    
    # Call to lapack_func(...): (line 1240)
    # Processing the call arguments (line 1240)
    # Getting the type of 'a1' (line 1240)
    a1_12917 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1240, 47), 'a1', False)
    # Getting the type of 'b1' (line 1240)
    b1_12918 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1240, 51), 'b1', False)
    # Getting the type of 'lwork' (line 1240)
    lwork_12919 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1240, 55), 'lwork', False)
    # Getting the type of 'rwork' (line 1240)
    rwork_12920 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1240, 62), 'rwork', False)
    # Getting the type of 'iwork' (line 1240)
    iwork_12921 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1240, 69), 'iwork', False)
    # Getting the type of 'cond' (line 1241)
    cond_12922 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1241, 47), 'cond', False)
    # Getting the type of 'False' (line 1241)
    False_12923 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1241, 53), 'False', False)
    # Getting the type of 'False' (line 1241)
    False_12924 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1241, 60), 'False', False)
    # Processing the call keyword arguments (line 1240)
    kwargs_12925 = {}
    # Getting the type of 'lapack_func' (line 1240)
    lapack_func_12916 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1240, 35), 'lapack_func', False)
    # Calling lapack_func(args, kwargs) (line 1240)
    lapack_func_call_result_12926 = invoke(stypy.reporting.localization.Localization(__file__, 1240, 35), lapack_func_12916, *[a1_12917, b1_12918, lwork_12919, rwork_12920, iwork_12921, cond_12922, False_12923, False_12924], **kwargs_12925)
    
    # Obtaining the member '__getitem__' of a type (line 1240)
    getitem___12927 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1240, 16), lapack_func_call_result_12926, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1240)
    subscript_call_result_12928 = invoke(stypy.reporting.localization.Localization(__file__, 1240, 16), getitem___12927, int_12915)
    
    # Assigning a type to the variable 'tuple_var_assignment_10119' (line 1240)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1240, 16), 'tuple_var_assignment_10119', subscript_call_result_12928)
    
    # Assigning a Subscript to a Name (line 1240):
    
    # Obtaining the type of the subscript
    int_12929 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1240, 16), 'int')
    
    # Call to lapack_func(...): (line 1240)
    # Processing the call arguments (line 1240)
    # Getting the type of 'a1' (line 1240)
    a1_12931 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1240, 47), 'a1', False)
    # Getting the type of 'b1' (line 1240)
    b1_12932 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1240, 51), 'b1', False)
    # Getting the type of 'lwork' (line 1240)
    lwork_12933 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1240, 55), 'lwork', False)
    # Getting the type of 'rwork' (line 1240)
    rwork_12934 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1240, 62), 'rwork', False)
    # Getting the type of 'iwork' (line 1240)
    iwork_12935 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1240, 69), 'iwork', False)
    # Getting the type of 'cond' (line 1241)
    cond_12936 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1241, 47), 'cond', False)
    # Getting the type of 'False' (line 1241)
    False_12937 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1241, 53), 'False', False)
    # Getting the type of 'False' (line 1241)
    False_12938 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1241, 60), 'False', False)
    # Processing the call keyword arguments (line 1240)
    kwargs_12939 = {}
    # Getting the type of 'lapack_func' (line 1240)
    lapack_func_12930 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1240, 35), 'lapack_func', False)
    # Calling lapack_func(args, kwargs) (line 1240)
    lapack_func_call_result_12940 = invoke(stypy.reporting.localization.Localization(__file__, 1240, 35), lapack_func_12930, *[a1_12931, b1_12932, lwork_12933, rwork_12934, iwork_12935, cond_12936, False_12937, False_12938], **kwargs_12939)
    
    # Obtaining the member '__getitem__' of a type (line 1240)
    getitem___12941 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1240, 16), lapack_func_call_result_12940, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1240)
    subscript_call_result_12942 = invoke(stypy.reporting.localization.Localization(__file__, 1240, 16), getitem___12941, int_12929)
    
    # Assigning a type to the variable 'tuple_var_assignment_10120' (line 1240)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1240, 16), 'tuple_var_assignment_10120', subscript_call_result_12942)
    
    # Assigning a Subscript to a Name (line 1240):
    
    # Obtaining the type of the subscript
    int_12943 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1240, 16), 'int')
    
    # Call to lapack_func(...): (line 1240)
    # Processing the call arguments (line 1240)
    # Getting the type of 'a1' (line 1240)
    a1_12945 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1240, 47), 'a1', False)
    # Getting the type of 'b1' (line 1240)
    b1_12946 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1240, 51), 'b1', False)
    # Getting the type of 'lwork' (line 1240)
    lwork_12947 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1240, 55), 'lwork', False)
    # Getting the type of 'rwork' (line 1240)
    rwork_12948 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1240, 62), 'rwork', False)
    # Getting the type of 'iwork' (line 1240)
    iwork_12949 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1240, 69), 'iwork', False)
    # Getting the type of 'cond' (line 1241)
    cond_12950 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1241, 47), 'cond', False)
    # Getting the type of 'False' (line 1241)
    False_12951 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1241, 53), 'False', False)
    # Getting the type of 'False' (line 1241)
    False_12952 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1241, 60), 'False', False)
    # Processing the call keyword arguments (line 1240)
    kwargs_12953 = {}
    # Getting the type of 'lapack_func' (line 1240)
    lapack_func_12944 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1240, 35), 'lapack_func', False)
    # Calling lapack_func(args, kwargs) (line 1240)
    lapack_func_call_result_12954 = invoke(stypy.reporting.localization.Localization(__file__, 1240, 35), lapack_func_12944, *[a1_12945, b1_12946, lwork_12947, rwork_12948, iwork_12949, cond_12950, False_12951, False_12952], **kwargs_12953)
    
    # Obtaining the member '__getitem__' of a type (line 1240)
    getitem___12955 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1240, 16), lapack_func_call_result_12954, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1240)
    subscript_call_result_12956 = invoke(stypy.reporting.localization.Localization(__file__, 1240, 16), getitem___12955, int_12943)
    
    # Assigning a type to the variable 'tuple_var_assignment_10121' (line 1240)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1240, 16), 'tuple_var_assignment_10121', subscript_call_result_12956)
    
    # Assigning a Name to a Name (line 1240):
    # Getting the type of 'tuple_var_assignment_10118' (line 1240)
    tuple_var_assignment_10118_12957 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1240, 16), 'tuple_var_assignment_10118')
    # Assigning a type to the variable 'x' (line 1240)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1240, 16), 'x', tuple_var_assignment_10118_12957)
    
    # Assigning a Name to a Name (line 1240):
    # Getting the type of 'tuple_var_assignment_10119' (line 1240)
    tuple_var_assignment_10119_12958 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1240, 16), 'tuple_var_assignment_10119')
    # Assigning a type to the variable 's' (line 1240)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1240, 19), 's', tuple_var_assignment_10119_12958)
    
    # Assigning a Name to a Name (line 1240):
    # Getting the type of 'tuple_var_assignment_10120' (line 1240)
    tuple_var_assignment_10120_12959 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1240, 16), 'tuple_var_assignment_10120')
    # Assigning a type to the variable 'rank' (line 1240)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1240, 22), 'rank', tuple_var_assignment_10120_12959)
    
    # Assigning a Name to a Name (line 1240):
    # Getting the type of 'tuple_var_assignment_10121' (line 1240)
    tuple_var_assignment_10121_12960 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1240, 16), 'tuple_var_assignment_10121')
    # Assigning a type to the variable 'info' (line 1240)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1240, 28), 'info', tuple_var_assignment_10121_12960)
    # SSA join for if statement (line 1210)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 1209)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 1203)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'info' (line 1242)
    info_12961 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1242, 11), 'info')
    int_12962 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1242, 18), 'int')
    # Applying the binary operator '>' (line 1242)
    result_gt_12963 = python_operator(stypy.reporting.localization.Localization(__file__, 1242, 11), '>', info_12961, int_12962)
    
    # Testing the type of an if condition (line 1242)
    if_condition_12964 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1242, 8), result_gt_12963)
    # Assigning a type to the variable 'if_condition_12964' (line 1242)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1242, 8), 'if_condition_12964', if_condition_12964)
    # SSA begins for if statement (line 1242)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to LinAlgError(...): (line 1243)
    # Processing the call arguments (line 1243)
    str_12966 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1243, 30), 'str', 'SVD did not converge in Linear Least Squares')
    # Processing the call keyword arguments (line 1243)
    kwargs_12967 = {}
    # Getting the type of 'LinAlgError' (line 1243)
    LinAlgError_12965 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1243, 18), 'LinAlgError', False)
    # Calling LinAlgError(args, kwargs) (line 1243)
    LinAlgError_call_result_12968 = invoke(stypy.reporting.localization.Localization(__file__, 1243, 18), LinAlgError_12965, *[str_12966], **kwargs_12967)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 1243, 12), LinAlgError_call_result_12968, 'raise parameter', BaseException)
    # SSA join for if statement (line 1242)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'info' (line 1244)
    info_12969 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1244, 11), 'info')
    int_12970 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1244, 18), 'int')
    # Applying the binary operator '<' (line 1244)
    result_lt_12971 = python_operator(stypy.reporting.localization.Localization(__file__, 1244, 11), '<', info_12969, int_12970)
    
    # Testing the type of an if condition (line 1244)
    if_condition_12972 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1244, 8), result_lt_12971)
    # Assigning a type to the variable 'if_condition_12972' (line 1244)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1244, 8), 'if_condition_12972', if_condition_12972)
    # SSA begins for if statement (line 1244)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 1245)
    # Processing the call arguments (line 1245)
    str_12974 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1245, 29), 'str', 'illegal value in %d-th argument of internal %s')
    
    # Obtaining an instance of the builtin type 'tuple' (line 1246)
    tuple_12975 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1246, 32), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 1246)
    # Adding element type (line 1246)
    
    # Getting the type of 'info' (line 1246)
    info_12976 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1246, 33), 'info', False)
    # Applying the 'usub' unary operator (line 1246)
    result___neg___12977 = python_operator(stypy.reporting.localization.Localization(__file__, 1246, 32), 'usub', info_12976)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1246, 32), tuple_12975, result___neg___12977)
    # Adding element type (line 1246)
    # Getting the type of 'lapack_driver' (line 1246)
    lapack_driver_12978 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1246, 39), 'lapack_driver', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1246, 32), tuple_12975, lapack_driver_12978)
    
    # Applying the binary operator '%' (line 1245)
    result_mod_12979 = python_operator(stypy.reporting.localization.Localization(__file__, 1245, 29), '%', str_12974, tuple_12975)
    
    # Processing the call keyword arguments (line 1245)
    kwargs_12980 = {}
    # Getting the type of 'ValueError' (line 1245)
    ValueError_12973 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1245, 18), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 1245)
    ValueError_call_result_12981 = invoke(stypy.reporting.localization.Localization(__file__, 1245, 18), ValueError_12973, *[result_mod_12979], **kwargs_12980)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 1245, 12), ValueError_call_result_12981, 'raise parameter', BaseException)
    # SSA join for if statement (line 1244)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 1247):
    
    # Assigning a Call to a Name (line 1247):
    
    # Call to asarray(...): (line 1247)
    # Processing the call arguments (line 1247)
    
    # Obtaining an instance of the builtin type 'list' (line 1247)
    list_12984 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1247, 28), 'list')
    # Adding type elements to the builtin type 'list' instance (line 1247)
    
    # Processing the call keyword arguments (line 1247)
    # Getting the type of 'x' (line 1247)
    x_12985 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1247, 38), 'x', False)
    # Obtaining the member 'dtype' of a type (line 1247)
    dtype_12986 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1247, 38), x_12985, 'dtype')
    keyword_12987 = dtype_12986
    kwargs_12988 = {'dtype': keyword_12987}
    # Getting the type of 'np' (line 1247)
    np_12982 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1247, 17), 'np', False)
    # Obtaining the member 'asarray' of a type (line 1247)
    asarray_12983 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1247, 17), np_12982, 'asarray')
    # Calling asarray(args, kwargs) (line 1247)
    asarray_call_result_12989 = invoke(stypy.reporting.localization.Localization(__file__, 1247, 17), asarray_12983, *[list_12984], **kwargs_12988)
    
    # Assigning a type to the variable 'resids' (line 1247)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1247, 8), 'resids', asarray_call_result_12989)
    
    
    # Getting the type of 'm' (line 1248)
    m_12990 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1248, 11), 'm')
    # Getting the type of 'n' (line 1248)
    n_12991 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1248, 15), 'n')
    # Applying the binary operator '>' (line 1248)
    result_gt_12992 = python_operator(stypy.reporting.localization.Localization(__file__, 1248, 11), '>', m_12990, n_12991)
    
    # Testing the type of an if condition (line 1248)
    if_condition_12993 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1248, 8), result_gt_12992)
    # Assigning a type to the variable 'if_condition_12993' (line 1248)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1248, 8), 'if_condition_12993', if_condition_12993)
    # SSA begins for if statement (line 1248)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Subscript to a Name (line 1249):
    
    # Assigning a Subscript to a Name (line 1249):
    
    # Obtaining the type of the subscript
    # Getting the type of 'n' (line 1249)
    n_12994 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1249, 20), 'n')
    slice_12995 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 1249, 17), None, n_12994, None)
    # Getting the type of 'x' (line 1249)
    x_12996 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1249, 17), 'x')
    # Obtaining the member '__getitem__' of a type (line 1249)
    getitem___12997 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1249, 17), x_12996, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1249)
    subscript_call_result_12998 = invoke(stypy.reporting.localization.Localization(__file__, 1249, 17), getitem___12997, slice_12995)
    
    # Assigning a type to the variable 'x1' (line 1249)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1249, 12), 'x1', subscript_call_result_12998)
    
    
    # Getting the type of 'rank' (line 1250)
    rank_12999 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1250, 15), 'rank')
    # Getting the type of 'n' (line 1250)
    n_13000 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1250, 23), 'n')
    # Applying the binary operator '==' (line 1250)
    result_eq_13001 = python_operator(stypy.reporting.localization.Localization(__file__, 1250, 15), '==', rank_12999, n_13000)
    
    # Testing the type of an if condition (line 1250)
    if_condition_13002 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1250, 12), result_eq_13001)
    # Assigning a type to the variable 'if_condition_13002' (line 1250)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1250, 12), 'if_condition_13002', if_condition_13002)
    # SSA begins for if statement (line 1250)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 1251):
    
    # Assigning a Call to a Name (line 1251):
    
    # Call to sum(...): (line 1251)
    # Processing the call arguments (line 1251)
    
    # Call to abs(...): (line 1251)
    # Processing the call arguments (line 1251)
    
    # Obtaining the type of the subscript
    # Getting the type of 'n' (line 1251)
    n_13007 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1251, 41), 'n', False)
    slice_13008 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 1251, 39), n_13007, None, None)
    # Getting the type of 'x' (line 1251)
    x_13009 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1251, 39), 'x', False)
    # Obtaining the member '__getitem__' of a type (line 1251)
    getitem___13010 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1251, 39), x_13009, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1251)
    subscript_call_result_13011 = invoke(stypy.reporting.localization.Localization(__file__, 1251, 39), getitem___13010, slice_13008)
    
    # Processing the call keyword arguments (line 1251)
    kwargs_13012 = {}
    # Getting the type of 'np' (line 1251)
    np_13005 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1251, 32), 'np', False)
    # Obtaining the member 'abs' of a type (line 1251)
    abs_13006 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1251, 32), np_13005, 'abs')
    # Calling abs(args, kwargs) (line 1251)
    abs_call_result_13013 = invoke(stypy.reporting.localization.Localization(__file__, 1251, 32), abs_13006, *[subscript_call_result_13011], **kwargs_13012)
    
    int_13014 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1251, 47), 'int')
    # Applying the binary operator '**' (line 1251)
    result_pow_13015 = python_operator(stypy.reporting.localization.Localization(__file__, 1251, 32), '**', abs_call_result_13013, int_13014)
    
    # Processing the call keyword arguments (line 1251)
    int_13016 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1251, 55), 'int')
    keyword_13017 = int_13016
    kwargs_13018 = {'axis': keyword_13017}
    # Getting the type of 'np' (line 1251)
    np_13003 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1251, 25), 'np', False)
    # Obtaining the member 'sum' of a type (line 1251)
    sum_13004 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1251, 25), np_13003, 'sum')
    # Calling sum(args, kwargs) (line 1251)
    sum_call_result_13019 = invoke(stypy.reporting.localization.Localization(__file__, 1251, 25), sum_13004, *[result_pow_13015], **kwargs_13018)
    
    # Assigning a type to the variable 'resids' (line 1251)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1251, 16), 'resids', sum_call_result_13019)
    # SSA join for if statement (line 1250)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Name to a Name (line 1252):
    
    # Assigning a Name to a Name (line 1252):
    # Getting the type of 'x1' (line 1252)
    x1_13020 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1252, 16), 'x1')
    # Assigning a type to the variable 'x' (line 1252)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1252, 12), 'x', x1_13020)
    # SSA join for if statement (line 1248)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Obtaining an instance of the builtin type 'tuple' (line 1253)
    tuple_13021 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1253, 15), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 1253)
    # Adding element type (line 1253)
    # Getting the type of 'x' (line 1253)
    x_13022 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1253, 15), 'x')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1253, 15), tuple_13021, x_13022)
    # Adding element type (line 1253)
    # Getting the type of 'resids' (line 1253)
    resids_13023 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1253, 18), 'resids')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1253, 15), tuple_13021, resids_13023)
    # Adding element type (line 1253)
    # Getting the type of 'rank' (line 1253)
    rank_13024 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1253, 26), 'rank')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1253, 15), tuple_13021, rank_13024)
    # Adding element type (line 1253)
    # Getting the type of 's' (line 1253)
    s_13025 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1253, 32), 's')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1253, 15), tuple_13021, s_13025)
    
    # Assigning a type to the variable 'stypy_return_type' (line 1253)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1253, 8), 'stypy_return_type', tuple_13021)
    # SSA branch for the else part of an if statement (line 1202)
    module_type_store.open_ssa_branch('else')
    
    
    # Getting the type of 'driver' (line 1255)
    driver_13026 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1255, 9), 'driver')
    str_13027 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1255, 19), 'str', 'gelsy')
    # Applying the binary operator '==' (line 1255)
    result_eq_13028 = python_operator(stypy.reporting.localization.Localization(__file__, 1255, 9), '==', driver_13026, str_13027)
    
    # Testing the type of an if condition (line 1255)
    if_condition_13029 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1255, 9), result_eq_13028)
    # Assigning a type to the variable 'if_condition_13029' (line 1255)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1255, 9), 'if_condition_13029', if_condition_13029)
    # SSA begins for if statement (line 1255)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 1256):
    
    # Assigning a Call to a Name (line 1256):
    
    # Call to _compute_lwork(...): (line 1256)
    # Processing the call arguments (line 1256)
    # Getting the type of 'lapack_lwork' (line 1256)
    lapack_lwork_13031 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1256, 31), 'lapack_lwork', False)
    # Getting the type of 'm' (line 1256)
    m_13032 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1256, 45), 'm', False)
    # Getting the type of 'n' (line 1256)
    n_13033 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1256, 48), 'n', False)
    # Getting the type of 'nrhs' (line 1256)
    nrhs_13034 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1256, 51), 'nrhs', False)
    # Getting the type of 'cond' (line 1256)
    cond_13035 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1256, 57), 'cond', False)
    # Processing the call keyword arguments (line 1256)
    kwargs_13036 = {}
    # Getting the type of '_compute_lwork' (line 1256)
    _compute_lwork_13030 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1256, 16), '_compute_lwork', False)
    # Calling _compute_lwork(args, kwargs) (line 1256)
    _compute_lwork_call_result_13037 = invoke(stypy.reporting.localization.Localization(__file__, 1256, 16), _compute_lwork_13030, *[lapack_lwork_13031, m_13032, n_13033, nrhs_13034, cond_13035], **kwargs_13036)
    
    # Assigning a type to the variable 'lwork' (line 1256)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1256, 8), 'lwork', _compute_lwork_call_result_13037)
    
    # Assigning a Call to a Name (line 1257):
    
    # Assigning a Call to a Name (line 1257):
    
    # Call to zeros(...): (line 1257)
    # Processing the call arguments (line 1257)
    
    # Obtaining an instance of the builtin type 'tuple' (line 1257)
    tuple_13040 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1257, 25), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 1257)
    # Adding element type (line 1257)
    
    # Obtaining the type of the subscript
    int_13041 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1257, 34), 'int')
    # Getting the type of 'a1' (line 1257)
    a1_13042 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1257, 25), 'a1', False)
    # Obtaining the member 'shape' of a type (line 1257)
    shape_13043 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1257, 25), a1_13042, 'shape')
    # Obtaining the member '__getitem__' of a type (line 1257)
    getitem___13044 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1257, 25), shape_13043, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1257)
    subscript_call_result_13045 = invoke(stypy.reporting.localization.Localization(__file__, 1257, 25), getitem___13044, int_13041)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1257, 25), tuple_13040, subscript_call_result_13045)
    # Adding element type (line 1257)
    int_13046 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1257, 38), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1257, 25), tuple_13040, int_13046)
    
    # Processing the call keyword arguments (line 1257)
    # Getting the type of 'np' (line 1257)
    np_13047 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1257, 48), 'np', False)
    # Obtaining the member 'int32' of a type (line 1257)
    int32_13048 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1257, 48), np_13047, 'int32')
    keyword_13049 = int32_13048
    kwargs_13050 = {'dtype': keyword_13049}
    # Getting the type of 'np' (line 1257)
    np_13038 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1257, 15), 'np', False)
    # Obtaining the member 'zeros' of a type (line 1257)
    zeros_13039 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1257, 15), np_13038, 'zeros')
    # Calling zeros(args, kwargs) (line 1257)
    zeros_call_result_13051 = invoke(stypy.reporting.localization.Localization(__file__, 1257, 15), zeros_13039, *[tuple_13040], **kwargs_13050)
    
    # Assigning a type to the variable 'jptv' (line 1257)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1257, 8), 'jptv', zeros_call_result_13051)
    
    # Assigning a Call to a Tuple (line 1258):
    
    # Assigning a Subscript to a Name (line 1258):
    
    # Obtaining the type of the subscript
    int_13052 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1258, 8), 'int')
    
    # Call to lapack_func(...): (line 1258)
    # Processing the call arguments (line 1258)
    # Getting the type of 'a1' (line 1258)
    a1_13054 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1258, 42), 'a1', False)
    # Getting the type of 'b1' (line 1258)
    b1_13055 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1258, 46), 'b1', False)
    # Getting the type of 'jptv' (line 1258)
    jptv_13056 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1258, 50), 'jptv', False)
    # Getting the type of 'cond' (line 1258)
    cond_13057 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1258, 56), 'cond', False)
    # Getting the type of 'lwork' (line 1259)
    lwork_13058 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1259, 42), 'lwork', False)
    # Getting the type of 'False' (line 1259)
    False_13059 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1259, 49), 'False', False)
    # Getting the type of 'False' (line 1259)
    False_13060 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1259, 56), 'False', False)
    # Processing the call keyword arguments (line 1258)
    kwargs_13061 = {}
    # Getting the type of 'lapack_func' (line 1258)
    lapack_func_13053 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1258, 30), 'lapack_func', False)
    # Calling lapack_func(args, kwargs) (line 1258)
    lapack_func_call_result_13062 = invoke(stypy.reporting.localization.Localization(__file__, 1258, 30), lapack_func_13053, *[a1_13054, b1_13055, jptv_13056, cond_13057, lwork_13058, False_13059, False_13060], **kwargs_13061)
    
    # Obtaining the member '__getitem__' of a type (line 1258)
    getitem___13063 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1258, 8), lapack_func_call_result_13062, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1258)
    subscript_call_result_13064 = invoke(stypy.reporting.localization.Localization(__file__, 1258, 8), getitem___13063, int_13052)
    
    # Assigning a type to the variable 'tuple_var_assignment_10122' (line 1258)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1258, 8), 'tuple_var_assignment_10122', subscript_call_result_13064)
    
    # Assigning a Subscript to a Name (line 1258):
    
    # Obtaining the type of the subscript
    int_13065 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1258, 8), 'int')
    
    # Call to lapack_func(...): (line 1258)
    # Processing the call arguments (line 1258)
    # Getting the type of 'a1' (line 1258)
    a1_13067 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1258, 42), 'a1', False)
    # Getting the type of 'b1' (line 1258)
    b1_13068 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1258, 46), 'b1', False)
    # Getting the type of 'jptv' (line 1258)
    jptv_13069 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1258, 50), 'jptv', False)
    # Getting the type of 'cond' (line 1258)
    cond_13070 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1258, 56), 'cond', False)
    # Getting the type of 'lwork' (line 1259)
    lwork_13071 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1259, 42), 'lwork', False)
    # Getting the type of 'False' (line 1259)
    False_13072 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1259, 49), 'False', False)
    # Getting the type of 'False' (line 1259)
    False_13073 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1259, 56), 'False', False)
    # Processing the call keyword arguments (line 1258)
    kwargs_13074 = {}
    # Getting the type of 'lapack_func' (line 1258)
    lapack_func_13066 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1258, 30), 'lapack_func', False)
    # Calling lapack_func(args, kwargs) (line 1258)
    lapack_func_call_result_13075 = invoke(stypy.reporting.localization.Localization(__file__, 1258, 30), lapack_func_13066, *[a1_13067, b1_13068, jptv_13069, cond_13070, lwork_13071, False_13072, False_13073], **kwargs_13074)
    
    # Obtaining the member '__getitem__' of a type (line 1258)
    getitem___13076 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1258, 8), lapack_func_call_result_13075, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1258)
    subscript_call_result_13077 = invoke(stypy.reporting.localization.Localization(__file__, 1258, 8), getitem___13076, int_13065)
    
    # Assigning a type to the variable 'tuple_var_assignment_10123' (line 1258)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1258, 8), 'tuple_var_assignment_10123', subscript_call_result_13077)
    
    # Assigning a Subscript to a Name (line 1258):
    
    # Obtaining the type of the subscript
    int_13078 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1258, 8), 'int')
    
    # Call to lapack_func(...): (line 1258)
    # Processing the call arguments (line 1258)
    # Getting the type of 'a1' (line 1258)
    a1_13080 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1258, 42), 'a1', False)
    # Getting the type of 'b1' (line 1258)
    b1_13081 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1258, 46), 'b1', False)
    # Getting the type of 'jptv' (line 1258)
    jptv_13082 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1258, 50), 'jptv', False)
    # Getting the type of 'cond' (line 1258)
    cond_13083 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1258, 56), 'cond', False)
    # Getting the type of 'lwork' (line 1259)
    lwork_13084 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1259, 42), 'lwork', False)
    # Getting the type of 'False' (line 1259)
    False_13085 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1259, 49), 'False', False)
    # Getting the type of 'False' (line 1259)
    False_13086 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1259, 56), 'False', False)
    # Processing the call keyword arguments (line 1258)
    kwargs_13087 = {}
    # Getting the type of 'lapack_func' (line 1258)
    lapack_func_13079 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1258, 30), 'lapack_func', False)
    # Calling lapack_func(args, kwargs) (line 1258)
    lapack_func_call_result_13088 = invoke(stypy.reporting.localization.Localization(__file__, 1258, 30), lapack_func_13079, *[a1_13080, b1_13081, jptv_13082, cond_13083, lwork_13084, False_13085, False_13086], **kwargs_13087)
    
    # Obtaining the member '__getitem__' of a type (line 1258)
    getitem___13089 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1258, 8), lapack_func_call_result_13088, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1258)
    subscript_call_result_13090 = invoke(stypy.reporting.localization.Localization(__file__, 1258, 8), getitem___13089, int_13078)
    
    # Assigning a type to the variable 'tuple_var_assignment_10124' (line 1258)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1258, 8), 'tuple_var_assignment_10124', subscript_call_result_13090)
    
    # Assigning a Subscript to a Name (line 1258):
    
    # Obtaining the type of the subscript
    int_13091 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1258, 8), 'int')
    
    # Call to lapack_func(...): (line 1258)
    # Processing the call arguments (line 1258)
    # Getting the type of 'a1' (line 1258)
    a1_13093 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1258, 42), 'a1', False)
    # Getting the type of 'b1' (line 1258)
    b1_13094 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1258, 46), 'b1', False)
    # Getting the type of 'jptv' (line 1258)
    jptv_13095 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1258, 50), 'jptv', False)
    # Getting the type of 'cond' (line 1258)
    cond_13096 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1258, 56), 'cond', False)
    # Getting the type of 'lwork' (line 1259)
    lwork_13097 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1259, 42), 'lwork', False)
    # Getting the type of 'False' (line 1259)
    False_13098 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1259, 49), 'False', False)
    # Getting the type of 'False' (line 1259)
    False_13099 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1259, 56), 'False', False)
    # Processing the call keyword arguments (line 1258)
    kwargs_13100 = {}
    # Getting the type of 'lapack_func' (line 1258)
    lapack_func_13092 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1258, 30), 'lapack_func', False)
    # Calling lapack_func(args, kwargs) (line 1258)
    lapack_func_call_result_13101 = invoke(stypy.reporting.localization.Localization(__file__, 1258, 30), lapack_func_13092, *[a1_13093, b1_13094, jptv_13095, cond_13096, lwork_13097, False_13098, False_13099], **kwargs_13100)
    
    # Obtaining the member '__getitem__' of a type (line 1258)
    getitem___13102 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1258, 8), lapack_func_call_result_13101, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1258)
    subscript_call_result_13103 = invoke(stypy.reporting.localization.Localization(__file__, 1258, 8), getitem___13102, int_13091)
    
    # Assigning a type to the variable 'tuple_var_assignment_10125' (line 1258)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1258, 8), 'tuple_var_assignment_10125', subscript_call_result_13103)
    
    # Assigning a Subscript to a Name (line 1258):
    
    # Obtaining the type of the subscript
    int_13104 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1258, 8), 'int')
    
    # Call to lapack_func(...): (line 1258)
    # Processing the call arguments (line 1258)
    # Getting the type of 'a1' (line 1258)
    a1_13106 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1258, 42), 'a1', False)
    # Getting the type of 'b1' (line 1258)
    b1_13107 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1258, 46), 'b1', False)
    # Getting the type of 'jptv' (line 1258)
    jptv_13108 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1258, 50), 'jptv', False)
    # Getting the type of 'cond' (line 1258)
    cond_13109 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1258, 56), 'cond', False)
    # Getting the type of 'lwork' (line 1259)
    lwork_13110 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1259, 42), 'lwork', False)
    # Getting the type of 'False' (line 1259)
    False_13111 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1259, 49), 'False', False)
    # Getting the type of 'False' (line 1259)
    False_13112 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1259, 56), 'False', False)
    # Processing the call keyword arguments (line 1258)
    kwargs_13113 = {}
    # Getting the type of 'lapack_func' (line 1258)
    lapack_func_13105 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1258, 30), 'lapack_func', False)
    # Calling lapack_func(args, kwargs) (line 1258)
    lapack_func_call_result_13114 = invoke(stypy.reporting.localization.Localization(__file__, 1258, 30), lapack_func_13105, *[a1_13106, b1_13107, jptv_13108, cond_13109, lwork_13110, False_13111, False_13112], **kwargs_13113)
    
    # Obtaining the member '__getitem__' of a type (line 1258)
    getitem___13115 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1258, 8), lapack_func_call_result_13114, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1258)
    subscript_call_result_13116 = invoke(stypy.reporting.localization.Localization(__file__, 1258, 8), getitem___13115, int_13104)
    
    # Assigning a type to the variable 'tuple_var_assignment_10126' (line 1258)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1258, 8), 'tuple_var_assignment_10126', subscript_call_result_13116)
    
    # Assigning a Name to a Name (line 1258):
    # Getting the type of 'tuple_var_assignment_10122' (line 1258)
    tuple_var_assignment_10122_13117 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1258, 8), 'tuple_var_assignment_10122')
    # Assigning a type to the variable 'v' (line 1258)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1258, 8), 'v', tuple_var_assignment_10122_13117)
    
    # Assigning a Name to a Name (line 1258):
    # Getting the type of 'tuple_var_assignment_10123' (line 1258)
    tuple_var_assignment_10123_13118 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1258, 8), 'tuple_var_assignment_10123')
    # Assigning a type to the variable 'x' (line 1258)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1258, 11), 'x', tuple_var_assignment_10123_13118)
    
    # Assigning a Name to a Name (line 1258):
    # Getting the type of 'tuple_var_assignment_10124' (line 1258)
    tuple_var_assignment_10124_13119 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1258, 8), 'tuple_var_assignment_10124')
    # Assigning a type to the variable 'j' (line 1258)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1258, 14), 'j', tuple_var_assignment_10124_13119)
    
    # Assigning a Name to a Name (line 1258):
    # Getting the type of 'tuple_var_assignment_10125' (line 1258)
    tuple_var_assignment_10125_13120 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1258, 8), 'tuple_var_assignment_10125')
    # Assigning a type to the variable 'rank' (line 1258)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1258, 17), 'rank', tuple_var_assignment_10125_13120)
    
    # Assigning a Name to a Name (line 1258):
    # Getting the type of 'tuple_var_assignment_10126' (line 1258)
    tuple_var_assignment_10126_13121 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1258, 8), 'tuple_var_assignment_10126')
    # Assigning a type to the variable 'info' (line 1258)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1258, 23), 'info', tuple_var_assignment_10126_13121)
    
    
    # Getting the type of 'info' (line 1260)
    info_13122 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1260, 11), 'info')
    int_13123 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1260, 18), 'int')
    # Applying the binary operator '<' (line 1260)
    result_lt_13124 = python_operator(stypy.reporting.localization.Localization(__file__, 1260, 11), '<', info_13122, int_13123)
    
    # Testing the type of an if condition (line 1260)
    if_condition_13125 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1260, 8), result_lt_13124)
    # Assigning a type to the variable 'if_condition_13125' (line 1260)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1260, 8), 'if_condition_13125', if_condition_13125)
    # SSA begins for if statement (line 1260)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 1261)
    # Processing the call arguments (line 1261)
    str_13127 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1261, 29), 'str', 'illegal value in %d-th argument of internal gelsy')
    
    # Getting the type of 'info' (line 1262)
    info_13128 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1262, 40), 'info', False)
    # Applying the 'usub' unary operator (line 1262)
    result___neg___13129 = python_operator(stypy.reporting.localization.Localization(__file__, 1262, 39), 'usub', info_13128)
    
    # Applying the binary operator '%' (line 1261)
    result_mod_13130 = python_operator(stypy.reporting.localization.Localization(__file__, 1261, 29), '%', str_13127, result___neg___13129)
    
    # Processing the call keyword arguments (line 1261)
    kwargs_13131 = {}
    # Getting the type of 'ValueError' (line 1261)
    ValueError_13126 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1261, 18), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 1261)
    ValueError_call_result_13132 = invoke(stypy.reporting.localization.Localization(__file__, 1261, 18), ValueError_13126, *[result_mod_13130], **kwargs_13131)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 1261, 12), ValueError_call_result_13132, 'raise parameter', BaseException)
    # SSA join for if statement (line 1260)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'm' (line 1263)
    m_13133 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1263, 11), 'm')
    # Getting the type of 'n' (line 1263)
    n_13134 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1263, 15), 'n')
    # Applying the binary operator '>' (line 1263)
    result_gt_13135 = python_operator(stypy.reporting.localization.Localization(__file__, 1263, 11), '>', m_13133, n_13134)
    
    # Testing the type of an if condition (line 1263)
    if_condition_13136 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1263, 8), result_gt_13135)
    # Assigning a type to the variable 'if_condition_13136' (line 1263)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1263, 8), 'if_condition_13136', if_condition_13136)
    # SSA begins for if statement (line 1263)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Subscript to a Name (line 1264):
    
    # Assigning a Subscript to a Name (line 1264):
    
    # Obtaining the type of the subscript
    # Getting the type of 'n' (line 1264)
    n_13137 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1264, 20), 'n')
    slice_13138 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 1264, 17), None, n_13137, None)
    # Getting the type of 'x' (line 1264)
    x_13139 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1264, 17), 'x')
    # Obtaining the member '__getitem__' of a type (line 1264)
    getitem___13140 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1264, 17), x_13139, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1264)
    subscript_call_result_13141 = invoke(stypy.reporting.localization.Localization(__file__, 1264, 17), getitem___13140, slice_13138)
    
    # Assigning a type to the variable 'x1' (line 1264)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1264, 12), 'x1', subscript_call_result_13141)
    
    # Assigning a Name to a Name (line 1265):
    
    # Assigning a Name to a Name (line 1265):
    # Getting the type of 'x1' (line 1265)
    x1_13142 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1265, 16), 'x1')
    # Assigning a type to the variable 'x' (line 1265)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1265, 12), 'x', x1_13142)
    # SSA join for if statement (line 1263)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Obtaining an instance of the builtin type 'tuple' (line 1266)
    tuple_13143 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1266, 15), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 1266)
    # Adding element type (line 1266)
    # Getting the type of 'x' (line 1266)
    x_13144 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1266, 15), 'x')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1266, 15), tuple_13143, x_13144)
    # Adding element type (line 1266)
    
    # Call to array(...): (line 1266)
    # Processing the call arguments (line 1266)
    
    # Obtaining an instance of the builtin type 'list' (line 1266)
    list_13147 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1266, 27), 'list')
    # Adding type elements to the builtin type 'list' instance (line 1266)
    
    # Getting the type of 'x' (line 1266)
    x_13148 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1266, 31), 'x', False)
    # Obtaining the member 'dtype' of a type (line 1266)
    dtype_13149 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1266, 31), x_13148, 'dtype')
    # Processing the call keyword arguments (line 1266)
    kwargs_13150 = {}
    # Getting the type of 'np' (line 1266)
    np_13145 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1266, 18), 'np', False)
    # Obtaining the member 'array' of a type (line 1266)
    array_13146 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1266, 18), np_13145, 'array')
    # Calling array(args, kwargs) (line 1266)
    array_call_result_13151 = invoke(stypy.reporting.localization.Localization(__file__, 1266, 18), array_13146, *[list_13147, dtype_13149], **kwargs_13150)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1266, 15), tuple_13143, array_call_result_13151)
    # Adding element type (line 1266)
    # Getting the type of 'rank' (line 1266)
    rank_13152 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1266, 41), 'rank')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1266, 15), tuple_13143, rank_13152)
    # Adding element type (line 1266)
    # Getting the type of 'None' (line 1266)
    None_13153 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1266, 47), 'None')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1266, 15), tuple_13143, None_13153)
    
    # Assigning a type to the variable 'stypy_return_type' (line 1266)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1266, 8), 'stypy_return_type', tuple_13143)
    # SSA join for if statement (line 1255)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 1202)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'lstsq(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'lstsq' in the type store
    # Getting the type of 'stypy_return_type' (line 1049)
    stypy_return_type_13154 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1049, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_13154)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'lstsq'
    return stypy_return_type_13154

# Assigning a type to the variable 'lstsq' (line 1049)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1049, 0), 'lstsq', lstsq)

# Assigning a Str to a Attribute (line 1267):

# Assigning a Str to a Attribute (line 1267):
str_13155 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1267, 30), 'str', 'gelsd')
# Getting the type of 'lstsq' (line 1267)
lstsq_13156 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1267, 0), 'lstsq')
# Setting the type of the member 'default_lapack_driver' of a type (line 1267)
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 1267, 0), lstsq_13156, 'default_lapack_driver', str_13155)

@norecursion
def pinv(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'None' (line 1270)
    None_13157 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1270, 17), 'None')
    # Getting the type of 'None' (line 1270)
    None_13158 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1270, 29), 'None')
    # Getting the type of 'False' (line 1270)
    False_13159 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1270, 47), 'False')
    # Getting the type of 'True' (line 1270)
    True_13160 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1270, 67), 'True')
    defaults = [None_13157, None_13158, False_13159, True_13160]
    # Create a new context for function 'pinv'
    module_type_store = module_type_store.open_function_context('pinv', 1270, 0, False)
    
    # Passed parameters checking function
    pinv.stypy_localization = localization
    pinv.stypy_type_of_self = None
    pinv.stypy_type_store = module_type_store
    pinv.stypy_function_name = 'pinv'
    pinv.stypy_param_names_list = ['a', 'cond', 'rcond', 'return_rank', 'check_finite']
    pinv.stypy_varargs_param_name = None
    pinv.stypy_kwargs_param_name = None
    pinv.stypy_call_defaults = defaults
    pinv.stypy_call_varargs = varargs
    pinv.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'pinv', ['a', 'cond', 'rcond', 'return_rank', 'check_finite'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'pinv', localization, ['a', 'cond', 'rcond', 'return_rank', 'check_finite'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'pinv(...)' code ##################

    str_13161 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1314, (-1)), 'str', "\n    Compute the (Moore-Penrose) pseudo-inverse of a matrix.\n\n    Calculate a generalized inverse of a matrix using a least-squares\n    solver.\n\n    Parameters\n    ----------\n    a : (M, N) array_like\n        Matrix to be pseudo-inverted.\n    cond, rcond : float, optional\n        Cutoff for 'small' singular values in the least-squares solver.\n        Singular values smaller than ``rcond * largest_singular_value``\n        are considered zero.\n    return_rank : bool, optional\n        if True, return the effective rank of the matrix\n    check_finite : bool, optional\n        Whether to check that the input matrix contains only finite numbers.\n        Disabling may give a performance gain, but may result in problems\n        (crashes, non-termination) if the inputs do contain infinities or NaNs.\n\n    Returns\n    -------\n    B : (N, M) ndarray\n        The pseudo-inverse of matrix `a`.\n    rank : int\n        The effective rank of the matrix.  Returned if return_rank == True\n\n    Raises\n    ------\n    LinAlgError\n        If computation does not converge.\n\n    Examples\n    --------\n    >>> from scipy import linalg\n    >>> a = np.random.randn(9, 6)\n    >>> B = linalg.pinv(a)\n    >>> np.allclose(a, np.dot(a, np.dot(B, a)))\n    True\n    >>> np.allclose(B, np.dot(B, np.dot(a, B)))\n    True\n\n    ")
    
    # Assigning a Call to a Name (line 1315):
    
    # Assigning a Call to a Name (line 1315):
    
    # Call to _asarray_validated(...): (line 1315)
    # Processing the call arguments (line 1315)
    # Getting the type of 'a' (line 1315)
    a_13163 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1315, 27), 'a', False)
    # Processing the call keyword arguments (line 1315)
    # Getting the type of 'check_finite' (line 1315)
    check_finite_13164 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1315, 43), 'check_finite', False)
    keyword_13165 = check_finite_13164
    kwargs_13166 = {'check_finite': keyword_13165}
    # Getting the type of '_asarray_validated' (line 1315)
    _asarray_validated_13162 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1315, 8), '_asarray_validated', False)
    # Calling _asarray_validated(args, kwargs) (line 1315)
    _asarray_validated_call_result_13167 = invoke(stypy.reporting.localization.Localization(__file__, 1315, 8), _asarray_validated_13162, *[a_13163], **kwargs_13166)
    
    # Assigning a type to the variable 'a' (line 1315)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1315, 4), 'a', _asarray_validated_call_result_13167)
    
    # Assigning a Call to a Name (line 1316):
    
    # Assigning a Call to a Name (line 1316):
    
    # Call to identity(...): (line 1316)
    # Processing the call arguments (line 1316)
    
    # Obtaining the type of the subscript
    int_13170 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1316, 28), 'int')
    # Getting the type of 'a' (line 1316)
    a_13171 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1316, 20), 'a', False)
    # Obtaining the member 'shape' of a type (line 1316)
    shape_13172 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1316, 20), a_13171, 'shape')
    # Obtaining the member '__getitem__' of a type (line 1316)
    getitem___13173 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1316, 20), shape_13172, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1316)
    subscript_call_result_13174 = invoke(stypy.reporting.localization.Localization(__file__, 1316, 20), getitem___13173, int_13170)
    
    # Processing the call keyword arguments (line 1316)
    # Getting the type of 'a' (line 1316)
    a_13175 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1316, 38), 'a', False)
    # Obtaining the member 'dtype' of a type (line 1316)
    dtype_13176 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1316, 38), a_13175, 'dtype')
    keyword_13177 = dtype_13176
    kwargs_13178 = {'dtype': keyword_13177}
    # Getting the type of 'np' (line 1316)
    np_13168 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1316, 8), 'np', False)
    # Obtaining the member 'identity' of a type (line 1316)
    identity_13169 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1316, 8), np_13168, 'identity')
    # Calling identity(args, kwargs) (line 1316)
    identity_call_result_13179 = invoke(stypy.reporting.localization.Localization(__file__, 1316, 8), identity_13169, *[subscript_call_result_13174], **kwargs_13178)
    
    # Assigning a type to the variable 'b' (line 1316)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1316, 4), 'b', identity_call_result_13179)
    
    # Type idiom detected: calculating its left and rigth part (line 1317)
    # Getting the type of 'rcond' (line 1317)
    rcond_13180 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1317, 4), 'rcond')
    # Getting the type of 'None' (line 1317)
    None_13181 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1317, 20), 'None')
    
    (may_be_13182, more_types_in_union_13183) = may_not_be_none(rcond_13180, None_13181)

    if may_be_13182:

        if more_types_in_union_13183:
            # Runtime conditional SSA (line 1317)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a Name to a Name (line 1318):
        
        # Assigning a Name to a Name (line 1318):
        # Getting the type of 'rcond' (line 1318)
        rcond_13184 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1318, 15), 'rcond')
        # Assigning a type to the variable 'cond' (line 1318)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1318, 8), 'cond', rcond_13184)

        if more_types_in_union_13183:
            # SSA join for if statement (line 1317)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Assigning a Call to a Tuple (line 1320):
    
    # Assigning a Subscript to a Name (line 1320):
    
    # Obtaining the type of the subscript
    int_13185 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1320, 4), 'int')
    
    # Call to lstsq(...): (line 1320)
    # Processing the call arguments (line 1320)
    # Getting the type of 'a' (line 1320)
    a_13187 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1320, 31), 'a', False)
    # Getting the type of 'b' (line 1320)
    b_13188 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1320, 34), 'b', False)
    # Processing the call keyword arguments (line 1320)
    # Getting the type of 'cond' (line 1320)
    cond_13189 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1320, 42), 'cond', False)
    keyword_13190 = cond_13189
    # Getting the type of 'False' (line 1320)
    False_13191 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1320, 61), 'False', False)
    keyword_13192 = False_13191
    kwargs_13193 = {'cond': keyword_13190, 'check_finite': keyword_13192}
    # Getting the type of 'lstsq' (line 1320)
    lstsq_13186 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1320, 25), 'lstsq', False)
    # Calling lstsq(args, kwargs) (line 1320)
    lstsq_call_result_13194 = invoke(stypy.reporting.localization.Localization(__file__, 1320, 25), lstsq_13186, *[a_13187, b_13188], **kwargs_13193)
    
    # Obtaining the member '__getitem__' of a type (line 1320)
    getitem___13195 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1320, 4), lstsq_call_result_13194, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1320)
    subscript_call_result_13196 = invoke(stypy.reporting.localization.Localization(__file__, 1320, 4), getitem___13195, int_13185)
    
    # Assigning a type to the variable 'tuple_var_assignment_10127' (line 1320)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1320, 4), 'tuple_var_assignment_10127', subscript_call_result_13196)
    
    # Assigning a Subscript to a Name (line 1320):
    
    # Obtaining the type of the subscript
    int_13197 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1320, 4), 'int')
    
    # Call to lstsq(...): (line 1320)
    # Processing the call arguments (line 1320)
    # Getting the type of 'a' (line 1320)
    a_13199 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1320, 31), 'a', False)
    # Getting the type of 'b' (line 1320)
    b_13200 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1320, 34), 'b', False)
    # Processing the call keyword arguments (line 1320)
    # Getting the type of 'cond' (line 1320)
    cond_13201 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1320, 42), 'cond', False)
    keyword_13202 = cond_13201
    # Getting the type of 'False' (line 1320)
    False_13203 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1320, 61), 'False', False)
    keyword_13204 = False_13203
    kwargs_13205 = {'cond': keyword_13202, 'check_finite': keyword_13204}
    # Getting the type of 'lstsq' (line 1320)
    lstsq_13198 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1320, 25), 'lstsq', False)
    # Calling lstsq(args, kwargs) (line 1320)
    lstsq_call_result_13206 = invoke(stypy.reporting.localization.Localization(__file__, 1320, 25), lstsq_13198, *[a_13199, b_13200], **kwargs_13205)
    
    # Obtaining the member '__getitem__' of a type (line 1320)
    getitem___13207 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1320, 4), lstsq_call_result_13206, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1320)
    subscript_call_result_13208 = invoke(stypy.reporting.localization.Localization(__file__, 1320, 4), getitem___13207, int_13197)
    
    # Assigning a type to the variable 'tuple_var_assignment_10128' (line 1320)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1320, 4), 'tuple_var_assignment_10128', subscript_call_result_13208)
    
    # Assigning a Subscript to a Name (line 1320):
    
    # Obtaining the type of the subscript
    int_13209 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1320, 4), 'int')
    
    # Call to lstsq(...): (line 1320)
    # Processing the call arguments (line 1320)
    # Getting the type of 'a' (line 1320)
    a_13211 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1320, 31), 'a', False)
    # Getting the type of 'b' (line 1320)
    b_13212 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1320, 34), 'b', False)
    # Processing the call keyword arguments (line 1320)
    # Getting the type of 'cond' (line 1320)
    cond_13213 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1320, 42), 'cond', False)
    keyword_13214 = cond_13213
    # Getting the type of 'False' (line 1320)
    False_13215 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1320, 61), 'False', False)
    keyword_13216 = False_13215
    kwargs_13217 = {'cond': keyword_13214, 'check_finite': keyword_13216}
    # Getting the type of 'lstsq' (line 1320)
    lstsq_13210 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1320, 25), 'lstsq', False)
    # Calling lstsq(args, kwargs) (line 1320)
    lstsq_call_result_13218 = invoke(stypy.reporting.localization.Localization(__file__, 1320, 25), lstsq_13210, *[a_13211, b_13212], **kwargs_13217)
    
    # Obtaining the member '__getitem__' of a type (line 1320)
    getitem___13219 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1320, 4), lstsq_call_result_13218, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1320)
    subscript_call_result_13220 = invoke(stypy.reporting.localization.Localization(__file__, 1320, 4), getitem___13219, int_13209)
    
    # Assigning a type to the variable 'tuple_var_assignment_10129' (line 1320)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1320, 4), 'tuple_var_assignment_10129', subscript_call_result_13220)
    
    # Assigning a Subscript to a Name (line 1320):
    
    # Obtaining the type of the subscript
    int_13221 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1320, 4), 'int')
    
    # Call to lstsq(...): (line 1320)
    # Processing the call arguments (line 1320)
    # Getting the type of 'a' (line 1320)
    a_13223 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1320, 31), 'a', False)
    # Getting the type of 'b' (line 1320)
    b_13224 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1320, 34), 'b', False)
    # Processing the call keyword arguments (line 1320)
    # Getting the type of 'cond' (line 1320)
    cond_13225 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1320, 42), 'cond', False)
    keyword_13226 = cond_13225
    # Getting the type of 'False' (line 1320)
    False_13227 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1320, 61), 'False', False)
    keyword_13228 = False_13227
    kwargs_13229 = {'cond': keyword_13226, 'check_finite': keyword_13228}
    # Getting the type of 'lstsq' (line 1320)
    lstsq_13222 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1320, 25), 'lstsq', False)
    # Calling lstsq(args, kwargs) (line 1320)
    lstsq_call_result_13230 = invoke(stypy.reporting.localization.Localization(__file__, 1320, 25), lstsq_13222, *[a_13223, b_13224], **kwargs_13229)
    
    # Obtaining the member '__getitem__' of a type (line 1320)
    getitem___13231 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1320, 4), lstsq_call_result_13230, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1320)
    subscript_call_result_13232 = invoke(stypy.reporting.localization.Localization(__file__, 1320, 4), getitem___13231, int_13221)
    
    # Assigning a type to the variable 'tuple_var_assignment_10130' (line 1320)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1320, 4), 'tuple_var_assignment_10130', subscript_call_result_13232)
    
    # Assigning a Name to a Name (line 1320):
    # Getting the type of 'tuple_var_assignment_10127' (line 1320)
    tuple_var_assignment_10127_13233 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1320, 4), 'tuple_var_assignment_10127')
    # Assigning a type to the variable 'x' (line 1320)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1320, 4), 'x', tuple_var_assignment_10127_13233)
    
    # Assigning a Name to a Name (line 1320):
    # Getting the type of 'tuple_var_assignment_10128' (line 1320)
    tuple_var_assignment_10128_13234 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1320, 4), 'tuple_var_assignment_10128')
    # Assigning a type to the variable 'resids' (line 1320)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1320, 7), 'resids', tuple_var_assignment_10128_13234)
    
    # Assigning a Name to a Name (line 1320):
    # Getting the type of 'tuple_var_assignment_10129' (line 1320)
    tuple_var_assignment_10129_13235 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1320, 4), 'tuple_var_assignment_10129')
    # Assigning a type to the variable 'rank' (line 1320)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1320, 15), 'rank', tuple_var_assignment_10129_13235)
    
    # Assigning a Name to a Name (line 1320):
    # Getting the type of 'tuple_var_assignment_10130' (line 1320)
    tuple_var_assignment_10130_13236 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1320, 4), 'tuple_var_assignment_10130')
    # Assigning a type to the variable 's' (line 1320)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1320, 21), 's', tuple_var_assignment_10130_13236)
    
    # Getting the type of 'return_rank' (line 1322)
    return_rank_13237 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1322, 7), 'return_rank')
    # Testing the type of an if condition (line 1322)
    if_condition_13238 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1322, 4), return_rank_13237)
    # Assigning a type to the variable 'if_condition_13238' (line 1322)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1322, 4), 'if_condition_13238', if_condition_13238)
    # SSA begins for if statement (line 1322)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Obtaining an instance of the builtin type 'tuple' (line 1323)
    tuple_13239 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1323, 15), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 1323)
    # Adding element type (line 1323)
    # Getting the type of 'x' (line 1323)
    x_13240 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1323, 15), 'x')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1323, 15), tuple_13239, x_13240)
    # Adding element type (line 1323)
    # Getting the type of 'rank' (line 1323)
    rank_13241 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1323, 18), 'rank')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1323, 15), tuple_13239, rank_13241)
    
    # Assigning a type to the variable 'stypy_return_type' (line 1323)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1323, 8), 'stypy_return_type', tuple_13239)
    # SSA branch for the else part of an if statement (line 1322)
    module_type_store.open_ssa_branch('else')
    # Getting the type of 'x' (line 1325)
    x_13242 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1325, 15), 'x')
    # Assigning a type to the variable 'stypy_return_type' (line 1325)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1325, 8), 'stypy_return_type', x_13242)
    # SSA join for if statement (line 1322)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'pinv(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'pinv' in the type store
    # Getting the type of 'stypy_return_type' (line 1270)
    stypy_return_type_13243 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1270, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_13243)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'pinv'
    return stypy_return_type_13243

# Assigning a type to the variable 'pinv' (line 1270)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1270, 0), 'pinv', pinv)

@norecursion
def pinv2(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'None' (line 1328)
    None_13244 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1328, 18), 'None')
    # Getting the type of 'None' (line 1328)
    None_13245 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1328, 30), 'None')
    # Getting the type of 'False' (line 1328)
    False_13246 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1328, 48), 'False')
    # Getting the type of 'True' (line 1328)
    True_13247 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1328, 68), 'True')
    defaults = [None_13244, None_13245, False_13246, True_13247]
    # Create a new context for function 'pinv2'
    module_type_store = module_type_store.open_function_context('pinv2', 1328, 0, False)
    
    # Passed parameters checking function
    pinv2.stypy_localization = localization
    pinv2.stypy_type_of_self = None
    pinv2.stypy_type_store = module_type_store
    pinv2.stypy_function_name = 'pinv2'
    pinv2.stypy_param_names_list = ['a', 'cond', 'rcond', 'return_rank', 'check_finite']
    pinv2.stypy_varargs_param_name = None
    pinv2.stypy_kwargs_param_name = None
    pinv2.stypy_call_defaults = defaults
    pinv2.stypy_call_varargs = varargs
    pinv2.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'pinv2', ['a', 'cond', 'rcond', 'return_rank', 'check_finite'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'pinv2', localization, ['a', 'cond', 'rcond', 'return_rank', 'check_finite'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'pinv2(...)' code ##################

    str_13248 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1374, (-1)), 'str', "\n    Compute the (Moore-Penrose) pseudo-inverse of a matrix.\n\n    Calculate a generalized inverse of a matrix using its\n    singular-value decomposition and including all 'large' singular\n    values.\n\n    Parameters\n    ----------\n    a : (M, N) array_like\n        Matrix to be pseudo-inverted.\n    cond, rcond : float or None\n        Cutoff for 'small' singular values.\n        Singular values smaller than ``rcond*largest_singular_value``\n        are considered zero.\n        If None or -1, suitable machine precision is used.\n    return_rank : bool, optional\n        if True, return the effective rank of the matrix\n    check_finite : bool, optional\n        Whether to check that the input matrix contains only finite numbers.\n        Disabling may give a performance gain, but may result in problems\n        (crashes, non-termination) if the inputs do contain infinities or NaNs.\n\n    Returns\n    -------\n    B : (N, M) ndarray\n        The pseudo-inverse of matrix `a`.\n    rank : int\n        The effective rank of the matrix.  Returned if return_rank == True\n\n    Raises\n    ------\n    LinAlgError\n        If SVD computation does not converge.\n\n    Examples\n    --------\n    >>> from scipy import linalg\n    >>> a = np.random.randn(9, 6)\n    >>> B = linalg.pinv2(a)\n    >>> np.allclose(a, np.dot(a, np.dot(B, a)))\n    True\n    >>> np.allclose(B, np.dot(B, np.dot(a, B)))\n    True\n\n    ")
    
    # Assigning a Call to a Name (line 1375):
    
    # Assigning a Call to a Name (line 1375):
    
    # Call to _asarray_validated(...): (line 1375)
    # Processing the call arguments (line 1375)
    # Getting the type of 'a' (line 1375)
    a_13250 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1375, 27), 'a', False)
    # Processing the call keyword arguments (line 1375)
    # Getting the type of 'check_finite' (line 1375)
    check_finite_13251 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1375, 43), 'check_finite', False)
    keyword_13252 = check_finite_13251
    kwargs_13253 = {'check_finite': keyword_13252}
    # Getting the type of '_asarray_validated' (line 1375)
    _asarray_validated_13249 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1375, 8), '_asarray_validated', False)
    # Calling _asarray_validated(args, kwargs) (line 1375)
    _asarray_validated_call_result_13254 = invoke(stypy.reporting.localization.Localization(__file__, 1375, 8), _asarray_validated_13249, *[a_13250], **kwargs_13253)
    
    # Assigning a type to the variable 'a' (line 1375)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1375, 4), 'a', _asarray_validated_call_result_13254)
    
    # Assigning a Call to a Tuple (line 1376):
    
    # Assigning a Subscript to a Name (line 1376):
    
    # Obtaining the type of the subscript
    int_13255 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1376, 4), 'int')
    
    # Call to svd(...): (line 1376)
    # Processing the call arguments (line 1376)
    # Getting the type of 'a' (line 1376)
    a_13258 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1376, 30), 'a', False)
    # Processing the call keyword arguments (line 1376)
    # Getting the type of 'False' (line 1376)
    False_13259 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1376, 47), 'False', False)
    keyword_13260 = False_13259
    # Getting the type of 'False' (line 1376)
    False_13261 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1376, 67), 'False', False)
    keyword_13262 = False_13261
    kwargs_13263 = {'check_finite': keyword_13262, 'full_matrices': keyword_13260}
    # Getting the type of 'decomp_svd' (line 1376)
    decomp_svd_13256 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1376, 15), 'decomp_svd', False)
    # Obtaining the member 'svd' of a type (line 1376)
    svd_13257 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1376, 15), decomp_svd_13256, 'svd')
    # Calling svd(args, kwargs) (line 1376)
    svd_call_result_13264 = invoke(stypy.reporting.localization.Localization(__file__, 1376, 15), svd_13257, *[a_13258], **kwargs_13263)
    
    # Obtaining the member '__getitem__' of a type (line 1376)
    getitem___13265 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1376, 4), svd_call_result_13264, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1376)
    subscript_call_result_13266 = invoke(stypy.reporting.localization.Localization(__file__, 1376, 4), getitem___13265, int_13255)
    
    # Assigning a type to the variable 'tuple_var_assignment_10131' (line 1376)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1376, 4), 'tuple_var_assignment_10131', subscript_call_result_13266)
    
    # Assigning a Subscript to a Name (line 1376):
    
    # Obtaining the type of the subscript
    int_13267 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1376, 4), 'int')
    
    # Call to svd(...): (line 1376)
    # Processing the call arguments (line 1376)
    # Getting the type of 'a' (line 1376)
    a_13270 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1376, 30), 'a', False)
    # Processing the call keyword arguments (line 1376)
    # Getting the type of 'False' (line 1376)
    False_13271 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1376, 47), 'False', False)
    keyword_13272 = False_13271
    # Getting the type of 'False' (line 1376)
    False_13273 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1376, 67), 'False', False)
    keyword_13274 = False_13273
    kwargs_13275 = {'check_finite': keyword_13274, 'full_matrices': keyword_13272}
    # Getting the type of 'decomp_svd' (line 1376)
    decomp_svd_13268 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1376, 15), 'decomp_svd', False)
    # Obtaining the member 'svd' of a type (line 1376)
    svd_13269 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1376, 15), decomp_svd_13268, 'svd')
    # Calling svd(args, kwargs) (line 1376)
    svd_call_result_13276 = invoke(stypy.reporting.localization.Localization(__file__, 1376, 15), svd_13269, *[a_13270], **kwargs_13275)
    
    # Obtaining the member '__getitem__' of a type (line 1376)
    getitem___13277 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1376, 4), svd_call_result_13276, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1376)
    subscript_call_result_13278 = invoke(stypy.reporting.localization.Localization(__file__, 1376, 4), getitem___13277, int_13267)
    
    # Assigning a type to the variable 'tuple_var_assignment_10132' (line 1376)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1376, 4), 'tuple_var_assignment_10132', subscript_call_result_13278)
    
    # Assigning a Subscript to a Name (line 1376):
    
    # Obtaining the type of the subscript
    int_13279 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1376, 4), 'int')
    
    # Call to svd(...): (line 1376)
    # Processing the call arguments (line 1376)
    # Getting the type of 'a' (line 1376)
    a_13282 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1376, 30), 'a', False)
    # Processing the call keyword arguments (line 1376)
    # Getting the type of 'False' (line 1376)
    False_13283 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1376, 47), 'False', False)
    keyword_13284 = False_13283
    # Getting the type of 'False' (line 1376)
    False_13285 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1376, 67), 'False', False)
    keyword_13286 = False_13285
    kwargs_13287 = {'check_finite': keyword_13286, 'full_matrices': keyword_13284}
    # Getting the type of 'decomp_svd' (line 1376)
    decomp_svd_13280 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1376, 15), 'decomp_svd', False)
    # Obtaining the member 'svd' of a type (line 1376)
    svd_13281 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1376, 15), decomp_svd_13280, 'svd')
    # Calling svd(args, kwargs) (line 1376)
    svd_call_result_13288 = invoke(stypy.reporting.localization.Localization(__file__, 1376, 15), svd_13281, *[a_13282], **kwargs_13287)
    
    # Obtaining the member '__getitem__' of a type (line 1376)
    getitem___13289 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1376, 4), svd_call_result_13288, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1376)
    subscript_call_result_13290 = invoke(stypy.reporting.localization.Localization(__file__, 1376, 4), getitem___13289, int_13279)
    
    # Assigning a type to the variable 'tuple_var_assignment_10133' (line 1376)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1376, 4), 'tuple_var_assignment_10133', subscript_call_result_13290)
    
    # Assigning a Name to a Name (line 1376):
    # Getting the type of 'tuple_var_assignment_10131' (line 1376)
    tuple_var_assignment_10131_13291 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1376, 4), 'tuple_var_assignment_10131')
    # Assigning a type to the variable 'u' (line 1376)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1376, 4), 'u', tuple_var_assignment_10131_13291)
    
    # Assigning a Name to a Name (line 1376):
    # Getting the type of 'tuple_var_assignment_10132' (line 1376)
    tuple_var_assignment_10132_13292 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1376, 4), 'tuple_var_assignment_10132')
    # Assigning a type to the variable 's' (line 1376)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1376, 7), 's', tuple_var_assignment_10132_13292)
    
    # Assigning a Name to a Name (line 1376):
    # Getting the type of 'tuple_var_assignment_10133' (line 1376)
    tuple_var_assignment_10133_13293 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1376, 4), 'tuple_var_assignment_10133')
    # Assigning a type to the variable 'vh' (line 1376)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1376, 10), 'vh', tuple_var_assignment_10133_13293)
    
    # Type idiom detected: calculating its left and rigth part (line 1378)
    # Getting the type of 'rcond' (line 1378)
    rcond_13294 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1378, 4), 'rcond')
    # Getting the type of 'None' (line 1378)
    None_13295 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1378, 20), 'None')
    
    (may_be_13296, more_types_in_union_13297) = may_not_be_none(rcond_13294, None_13295)

    if may_be_13296:

        if more_types_in_union_13297:
            # Runtime conditional SSA (line 1378)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a Name to a Name (line 1379):
        
        # Assigning a Name to a Name (line 1379):
        # Getting the type of 'rcond' (line 1379)
        rcond_13298 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1379, 15), 'rcond')
        # Assigning a type to the variable 'cond' (line 1379)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1379, 8), 'cond', rcond_13298)

        if more_types_in_union_13297:
            # SSA join for if statement (line 1378)
            module_type_store = module_type_store.join_ssa_context()


    
    
    
    # Getting the type of 'cond' (line 1380)
    cond_13299 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1380, 7), 'cond')
    
    # Obtaining an instance of the builtin type 'list' (line 1380)
    list_13300 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1380, 15), 'list')
    # Adding type elements to the builtin type 'list' instance (line 1380)
    # Adding element type (line 1380)
    # Getting the type of 'None' (line 1380)
    None_13301 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1380, 16), 'None')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1380, 15), list_13300, None_13301)
    # Adding element type (line 1380)
    int_13302 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1380, 22), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1380, 15), list_13300, int_13302)
    
    # Applying the binary operator 'in' (line 1380)
    result_contains_13303 = python_operator(stypy.reporting.localization.Localization(__file__, 1380, 7), 'in', cond_13299, list_13300)
    
    # Testing the type of an if condition (line 1380)
    if_condition_13304 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1380, 4), result_contains_13303)
    # Assigning a type to the variable 'if_condition_13304' (line 1380)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1380, 4), 'if_condition_13304', if_condition_13304)
    # SSA begins for if statement (line 1380)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 1381):
    
    # Assigning a Call to a Name (line 1381):
    
    # Call to lower(...): (line 1381)
    # Processing the call keyword arguments (line 1381)
    kwargs_13309 = {}
    # Getting the type of 'u' (line 1381)
    u_13305 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1381, 12), 'u', False)
    # Obtaining the member 'dtype' of a type (line 1381)
    dtype_13306 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1381, 12), u_13305, 'dtype')
    # Obtaining the member 'char' of a type (line 1381)
    char_13307 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1381, 12), dtype_13306, 'char')
    # Obtaining the member 'lower' of a type (line 1381)
    lower_13308 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1381, 12), char_13307, 'lower')
    # Calling lower(args, kwargs) (line 1381)
    lower_call_result_13310 = invoke(stypy.reporting.localization.Localization(__file__, 1381, 12), lower_13308, *[], **kwargs_13309)
    
    # Assigning a type to the variable 't' (line 1381)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1381, 8), 't', lower_call_result_13310)
    
    # Assigning a Dict to a Name (line 1382):
    
    # Assigning a Dict to a Name (line 1382):
    
    # Obtaining an instance of the builtin type 'dict' (line 1382)
    dict_13311 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1382, 17), 'dict')
    # Adding type elements to the builtin type 'dict' instance (line 1382)
    # Adding element type (key, value) (line 1382)
    str_13312 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1382, 18), 'str', 'f')
    float_13313 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1382, 23), 'float')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1382, 17), dict_13311, (str_13312, float_13313))
    # Adding element type (key, value) (line 1382)
    str_13314 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1382, 28), 'str', 'd')
    float_13315 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1382, 33), 'float')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1382, 17), dict_13311, (str_13314, float_13315))
    
    # Assigning a type to the variable 'factor' (line 1382)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1382, 8), 'factor', dict_13311)
    
    # Assigning a BinOp to a Name (line 1383):
    
    # Assigning a BinOp to a Name (line 1383):
    
    # Obtaining the type of the subscript
    # Getting the type of 't' (line 1383)
    t_13316 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1383, 22), 't')
    # Getting the type of 'factor' (line 1383)
    factor_13317 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1383, 15), 'factor')
    # Obtaining the member '__getitem__' of a type (line 1383)
    getitem___13318 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1383, 15), factor_13317, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1383)
    subscript_call_result_13319 = invoke(stypy.reporting.localization.Localization(__file__, 1383, 15), getitem___13318, t_13316)
    
    
    # Call to finfo(...): (line 1383)
    # Processing the call arguments (line 1383)
    # Getting the type of 't' (line 1383)
    t_13322 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1383, 36), 't', False)
    # Processing the call keyword arguments (line 1383)
    kwargs_13323 = {}
    # Getting the type of 'np' (line 1383)
    np_13320 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1383, 27), 'np', False)
    # Obtaining the member 'finfo' of a type (line 1383)
    finfo_13321 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1383, 27), np_13320, 'finfo')
    # Calling finfo(args, kwargs) (line 1383)
    finfo_call_result_13324 = invoke(stypy.reporting.localization.Localization(__file__, 1383, 27), finfo_13321, *[t_13322], **kwargs_13323)
    
    # Obtaining the member 'eps' of a type (line 1383)
    eps_13325 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1383, 27), finfo_call_result_13324, 'eps')
    # Applying the binary operator '*' (line 1383)
    result_mul_13326 = python_operator(stypy.reporting.localization.Localization(__file__, 1383, 15), '*', subscript_call_result_13319, eps_13325)
    
    # Assigning a type to the variable 'cond' (line 1383)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1383, 8), 'cond', result_mul_13326)
    # SSA join for if statement (line 1380)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 1385):
    
    # Assigning a Call to a Name (line 1385):
    
    # Call to sum(...): (line 1385)
    # Processing the call arguments (line 1385)
    
    # Getting the type of 's' (line 1385)
    s_13329 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1385, 18), 's', False)
    # Getting the type of 'cond' (line 1385)
    cond_13330 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1385, 22), 'cond', False)
    
    # Call to max(...): (line 1385)
    # Processing the call arguments (line 1385)
    # Getting the type of 's' (line 1385)
    s_13333 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1385, 36), 's', False)
    # Processing the call keyword arguments (line 1385)
    kwargs_13334 = {}
    # Getting the type of 'np' (line 1385)
    np_13331 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1385, 29), 'np', False)
    # Obtaining the member 'max' of a type (line 1385)
    max_13332 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1385, 29), np_13331, 'max')
    # Calling max(args, kwargs) (line 1385)
    max_call_result_13335 = invoke(stypy.reporting.localization.Localization(__file__, 1385, 29), max_13332, *[s_13333], **kwargs_13334)
    
    # Applying the binary operator '*' (line 1385)
    result_mul_13336 = python_operator(stypy.reporting.localization.Localization(__file__, 1385, 22), '*', cond_13330, max_call_result_13335)
    
    # Applying the binary operator '>' (line 1385)
    result_gt_13337 = python_operator(stypy.reporting.localization.Localization(__file__, 1385, 18), '>', s_13329, result_mul_13336)
    
    # Processing the call keyword arguments (line 1385)
    kwargs_13338 = {}
    # Getting the type of 'np' (line 1385)
    np_13327 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1385, 11), 'np', False)
    # Obtaining the member 'sum' of a type (line 1385)
    sum_13328 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1385, 11), np_13327, 'sum')
    # Calling sum(args, kwargs) (line 1385)
    sum_call_result_13339 = invoke(stypy.reporting.localization.Localization(__file__, 1385, 11), sum_13328, *[result_gt_13337], **kwargs_13338)
    
    # Assigning a type to the variable 'rank' (line 1385)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1385, 4), 'rank', sum_call_result_13339)
    
    # Assigning a Subscript to a Name (line 1387):
    
    # Assigning a Subscript to a Name (line 1387):
    
    # Obtaining the type of the subscript
    slice_13340 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 1387, 8), None, None, None)
    # Getting the type of 'rank' (line 1387)
    rank_13341 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1387, 14), 'rank')
    slice_13342 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 1387, 8), None, rank_13341, None)
    # Getting the type of 'u' (line 1387)
    u_13343 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1387, 8), 'u')
    # Obtaining the member '__getitem__' of a type (line 1387)
    getitem___13344 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1387, 8), u_13343, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1387)
    subscript_call_result_13345 = invoke(stypy.reporting.localization.Localization(__file__, 1387, 8), getitem___13344, (slice_13340, slice_13342))
    
    # Assigning a type to the variable 'u' (line 1387)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1387, 4), 'u', subscript_call_result_13345)
    
    # Getting the type of 'u' (line 1388)
    u_13346 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1388, 4), 'u')
    
    # Obtaining the type of the subscript
    # Getting the type of 'rank' (line 1388)
    rank_13347 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1388, 12), 'rank')
    slice_13348 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 1388, 9), None, rank_13347, None)
    # Getting the type of 's' (line 1388)
    s_13349 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1388, 9), 's')
    # Obtaining the member '__getitem__' of a type (line 1388)
    getitem___13350 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1388, 9), s_13349, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1388)
    subscript_call_result_13351 = invoke(stypy.reporting.localization.Localization(__file__, 1388, 9), getitem___13350, slice_13348)
    
    # Applying the binary operator 'div=' (line 1388)
    result_div_13352 = python_operator(stypy.reporting.localization.Localization(__file__, 1388, 4), 'div=', u_13346, subscript_call_result_13351)
    # Assigning a type to the variable 'u' (line 1388)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1388, 4), 'u', result_div_13352)
    
    
    # Assigning a Call to a Name (line 1389):
    
    # Assigning a Call to a Name (line 1389):
    
    # Call to transpose(...): (line 1389)
    # Processing the call arguments (line 1389)
    
    # Call to conjugate(...): (line 1389)
    # Processing the call arguments (line 1389)
    
    # Call to dot(...): (line 1389)
    # Processing the call arguments (line 1389)
    # Getting the type of 'u' (line 1389)
    u_13359 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1389, 41), 'u', False)
    
    # Obtaining the type of the subscript
    # Getting the type of 'rank' (line 1389)
    rank_13360 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1389, 48), 'rank', False)
    slice_13361 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 1389, 44), None, rank_13360, None)
    # Getting the type of 'vh' (line 1389)
    vh_13362 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1389, 44), 'vh', False)
    # Obtaining the member '__getitem__' of a type (line 1389)
    getitem___13363 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1389, 44), vh_13362, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1389)
    subscript_call_result_13364 = invoke(stypy.reporting.localization.Localization(__file__, 1389, 44), getitem___13363, slice_13361)
    
    # Processing the call keyword arguments (line 1389)
    kwargs_13365 = {}
    # Getting the type of 'np' (line 1389)
    np_13357 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1389, 34), 'np', False)
    # Obtaining the member 'dot' of a type (line 1389)
    dot_13358 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1389, 34), np_13357, 'dot')
    # Calling dot(args, kwargs) (line 1389)
    dot_call_result_13366 = invoke(stypy.reporting.localization.Localization(__file__, 1389, 34), dot_13358, *[u_13359, subscript_call_result_13364], **kwargs_13365)
    
    # Processing the call keyword arguments (line 1389)
    kwargs_13367 = {}
    # Getting the type of 'np' (line 1389)
    np_13355 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1389, 21), 'np', False)
    # Obtaining the member 'conjugate' of a type (line 1389)
    conjugate_13356 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1389, 21), np_13355, 'conjugate')
    # Calling conjugate(args, kwargs) (line 1389)
    conjugate_call_result_13368 = invoke(stypy.reporting.localization.Localization(__file__, 1389, 21), conjugate_13356, *[dot_call_result_13366], **kwargs_13367)
    
    # Processing the call keyword arguments (line 1389)
    kwargs_13369 = {}
    # Getting the type of 'np' (line 1389)
    np_13353 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1389, 8), 'np', False)
    # Obtaining the member 'transpose' of a type (line 1389)
    transpose_13354 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1389, 8), np_13353, 'transpose')
    # Calling transpose(args, kwargs) (line 1389)
    transpose_call_result_13370 = invoke(stypy.reporting.localization.Localization(__file__, 1389, 8), transpose_13354, *[conjugate_call_result_13368], **kwargs_13369)
    
    # Assigning a type to the variable 'B' (line 1389)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1389, 4), 'B', transpose_call_result_13370)
    
    # Getting the type of 'return_rank' (line 1391)
    return_rank_13371 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1391, 7), 'return_rank')
    # Testing the type of an if condition (line 1391)
    if_condition_13372 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1391, 4), return_rank_13371)
    # Assigning a type to the variable 'if_condition_13372' (line 1391)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1391, 4), 'if_condition_13372', if_condition_13372)
    # SSA begins for if statement (line 1391)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Obtaining an instance of the builtin type 'tuple' (line 1392)
    tuple_13373 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1392, 15), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 1392)
    # Adding element type (line 1392)
    # Getting the type of 'B' (line 1392)
    B_13374 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1392, 15), 'B')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1392, 15), tuple_13373, B_13374)
    # Adding element type (line 1392)
    # Getting the type of 'rank' (line 1392)
    rank_13375 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1392, 18), 'rank')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1392, 15), tuple_13373, rank_13375)
    
    # Assigning a type to the variable 'stypy_return_type' (line 1392)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1392, 8), 'stypy_return_type', tuple_13373)
    # SSA branch for the else part of an if statement (line 1391)
    module_type_store.open_ssa_branch('else')
    # Getting the type of 'B' (line 1394)
    B_13376 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1394, 15), 'B')
    # Assigning a type to the variable 'stypy_return_type' (line 1394)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1394, 8), 'stypy_return_type', B_13376)
    # SSA join for if statement (line 1391)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'pinv2(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'pinv2' in the type store
    # Getting the type of 'stypy_return_type' (line 1328)
    stypy_return_type_13377 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1328, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_13377)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'pinv2'
    return stypy_return_type_13377

# Assigning a type to the variable 'pinv2' (line 1328)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1328, 0), 'pinv2', pinv2)

@norecursion
def pinvh(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'None' (line 1397)
    None_13378 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1397, 18), 'None')
    # Getting the type of 'None' (line 1397)
    None_13379 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1397, 30), 'None')
    # Getting the type of 'True' (line 1397)
    True_13380 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1397, 42), 'True')
    # Getting the type of 'False' (line 1397)
    False_13381 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1397, 60), 'False')
    # Getting the type of 'True' (line 1398)
    True_13382 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1398, 23), 'True')
    defaults = [None_13378, None_13379, True_13380, False_13381, True_13382]
    # Create a new context for function 'pinvh'
    module_type_store = module_type_store.open_function_context('pinvh', 1397, 0, False)
    
    # Passed parameters checking function
    pinvh.stypy_localization = localization
    pinvh.stypy_type_of_self = None
    pinvh.stypy_type_store = module_type_store
    pinvh.stypy_function_name = 'pinvh'
    pinvh.stypy_param_names_list = ['a', 'cond', 'rcond', 'lower', 'return_rank', 'check_finite']
    pinvh.stypy_varargs_param_name = None
    pinvh.stypy_kwargs_param_name = None
    pinvh.stypy_call_defaults = defaults
    pinvh.stypy_call_varargs = varargs
    pinvh.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'pinvh', ['a', 'cond', 'rcond', 'lower', 'return_rank', 'check_finite'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'pinvh', localization, ['a', 'cond', 'rcond', 'lower', 'return_rank', 'check_finite'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'pinvh(...)' code ##################

    str_13383 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1449, (-1)), 'str', "\n    Compute the (Moore-Penrose) pseudo-inverse of a Hermitian matrix.\n\n    Calculate a generalized inverse of a Hermitian or real symmetric matrix\n    using its eigenvalue decomposition and including all eigenvalues with\n    'large' absolute value.\n\n    Parameters\n    ----------\n    a : (N, N) array_like\n        Real symmetric or complex hermetian matrix to be pseudo-inverted\n    cond, rcond : float or None\n        Cutoff for 'small' eigenvalues.\n        Singular values smaller than rcond * largest_eigenvalue are considered\n        zero.\n\n        If None or -1, suitable machine precision is used.\n    lower : bool, optional\n        Whether the pertinent array data is taken from the lower or upper\n        triangle of a. (Default: lower)\n    return_rank : bool, optional\n        if True, return the effective rank of the matrix\n    check_finite : bool, optional\n        Whether to check that the input matrix contains only finite numbers.\n        Disabling may give a performance gain, but may result in problems\n        (crashes, non-termination) if the inputs do contain infinities or NaNs.\n\n    Returns\n    -------\n    B : (N, N) ndarray\n        The pseudo-inverse of matrix `a`.\n    rank : int\n        The effective rank of the matrix.  Returned if return_rank == True\n\n    Raises\n    ------\n    LinAlgError\n        If eigenvalue does not converge\n\n    Examples\n    --------\n    >>> from scipy.linalg import pinvh\n    >>> a = np.random.randn(9, 6)\n    >>> a = np.dot(a, a.T)\n    >>> B = pinvh(a)\n    >>> np.allclose(a, np.dot(a, np.dot(B, a)))\n    True\n    >>> np.allclose(B, np.dot(B, np.dot(a, B)))\n    True\n\n    ")
    
    # Assigning a Call to a Name (line 1450):
    
    # Assigning a Call to a Name (line 1450):
    
    # Call to _asarray_validated(...): (line 1450)
    # Processing the call arguments (line 1450)
    # Getting the type of 'a' (line 1450)
    a_13385 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1450, 27), 'a', False)
    # Processing the call keyword arguments (line 1450)
    # Getting the type of 'check_finite' (line 1450)
    check_finite_13386 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1450, 43), 'check_finite', False)
    keyword_13387 = check_finite_13386
    kwargs_13388 = {'check_finite': keyword_13387}
    # Getting the type of '_asarray_validated' (line 1450)
    _asarray_validated_13384 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1450, 8), '_asarray_validated', False)
    # Calling _asarray_validated(args, kwargs) (line 1450)
    _asarray_validated_call_result_13389 = invoke(stypy.reporting.localization.Localization(__file__, 1450, 8), _asarray_validated_13384, *[a_13385], **kwargs_13388)
    
    # Assigning a type to the variable 'a' (line 1450)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1450, 4), 'a', _asarray_validated_call_result_13389)
    
    # Assigning a Call to a Tuple (line 1451):
    
    # Assigning a Subscript to a Name (line 1451):
    
    # Obtaining the type of the subscript
    int_13390 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1451, 4), 'int')
    
    # Call to eigh(...): (line 1451)
    # Processing the call arguments (line 1451)
    # Getting the type of 'a' (line 1451)
    a_13393 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1451, 23), 'a', False)
    # Processing the call keyword arguments (line 1451)
    # Getting the type of 'lower' (line 1451)
    lower_13394 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1451, 32), 'lower', False)
    keyword_13395 = lower_13394
    # Getting the type of 'False' (line 1451)
    False_13396 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1451, 52), 'False', False)
    keyword_13397 = False_13396
    kwargs_13398 = {'lower': keyword_13395, 'check_finite': keyword_13397}
    # Getting the type of 'decomp' (line 1451)
    decomp_13391 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1451, 11), 'decomp', False)
    # Obtaining the member 'eigh' of a type (line 1451)
    eigh_13392 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1451, 11), decomp_13391, 'eigh')
    # Calling eigh(args, kwargs) (line 1451)
    eigh_call_result_13399 = invoke(stypy.reporting.localization.Localization(__file__, 1451, 11), eigh_13392, *[a_13393], **kwargs_13398)
    
    # Obtaining the member '__getitem__' of a type (line 1451)
    getitem___13400 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1451, 4), eigh_call_result_13399, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1451)
    subscript_call_result_13401 = invoke(stypy.reporting.localization.Localization(__file__, 1451, 4), getitem___13400, int_13390)
    
    # Assigning a type to the variable 'tuple_var_assignment_10134' (line 1451)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1451, 4), 'tuple_var_assignment_10134', subscript_call_result_13401)
    
    # Assigning a Subscript to a Name (line 1451):
    
    # Obtaining the type of the subscript
    int_13402 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1451, 4), 'int')
    
    # Call to eigh(...): (line 1451)
    # Processing the call arguments (line 1451)
    # Getting the type of 'a' (line 1451)
    a_13405 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1451, 23), 'a', False)
    # Processing the call keyword arguments (line 1451)
    # Getting the type of 'lower' (line 1451)
    lower_13406 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1451, 32), 'lower', False)
    keyword_13407 = lower_13406
    # Getting the type of 'False' (line 1451)
    False_13408 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1451, 52), 'False', False)
    keyword_13409 = False_13408
    kwargs_13410 = {'lower': keyword_13407, 'check_finite': keyword_13409}
    # Getting the type of 'decomp' (line 1451)
    decomp_13403 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1451, 11), 'decomp', False)
    # Obtaining the member 'eigh' of a type (line 1451)
    eigh_13404 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1451, 11), decomp_13403, 'eigh')
    # Calling eigh(args, kwargs) (line 1451)
    eigh_call_result_13411 = invoke(stypy.reporting.localization.Localization(__file__, 1451, 11), eigh_13404, *[a_13405], **kwargs_13410)
    
    # Obtaining the member '__getitem__' of a type (line 1451)
    getitem___13412 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1451, 4), eigh_call_result_13411, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1451)
    subscript_call_result_13413 = invoke(stypy.reporting.localization.Localization(__file__, 1451, 4), getitem___13412, int_13402)
    
    # Assigning a type to the variable 'tuple_var_assignment_10135' (line 1451)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1451, 4), 'tuple_var_assignment_10135', subscript_call_result_13413)
    
    # Assigning a Name to a Name (line 1451):
    # Getting the type of 'tuple_var_assignment_10134' (line 1451)
    tuple_var_assignment_10134_13414 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1451, 4), 'tuple_var_assignment_10134')
    # Assigning a type to the variable 's' (line 1451)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1451, 4), 's', tuple_var_assignment_10134_13414)
    
    # Assigning a Name to a Name (line 1451):
    # Getting the type of 'tuple_var_assignment_10135' (line 1451)
    tuple_var_assignment_10135_13415 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1451, 4), 'tuple_var_assignment_10135')
    # Assigning a type to the variable 'u' (line 1451)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1451, 7), 'u', tuple_var_assignment_10135_13415)
    
    # Type idiom detected: calculating its left and rigth part (line 1453)
    # Getting the type of 'rcond' (line 1453)
    rcond_13416 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1453, 4), 'rcond')
    # Getting the type of 'None' (line 1453)
    None_13417 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1453, 20), 'None')
    
    (may_be_13418, more_types_in_union_13419) = may_not_be_none(rcond_13416, None_13417)

    if may_be_13418:

        if more_types_in_union_13419:
            # Runtime conditional SSA (line 1453)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a Name to a Name (line 1454):
        
        # Assigning a Name to a Name (line 1454):
        # Getting the type of 'rcond' (line 1454)
        rcond_13420 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1454, 15), 'rcond')
        # Assigning a type to the variable 'cond' (line 1454)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1454, 8), 'cond', rcond_13420)

        if more_types_in_union_13419:
            # SSA join for if statement (line 1453)
            module_type_store = module_type_store.join_ssa_context()


    
    
    
    # Getting the type of 'cond' (line 1455)
    cond_13421 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1455, 7), 'cond')
    
    # Obtaining an instance of the builtin type 'list' (line 1455)
    list_13422 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1455, 15), 'list')
    # Adding type elements to the builtin type 'list' instance (line 1455)
    # Adding element type (line 1455)
    # Getting the type of 'None' (line 1455)
    None_13423 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1455, 16), 'None')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1455, 15), list_13422, None_13423)
    # Adding element type (line 1455)
    int_13424 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1455, 22), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1455, 15), list_13422, int_13424)
    
    # Applying the binary operator 'in' (line 1455)
    result_contains_13425 = python_operator(stypy.reporting.localization.Localization(__file__, 1455, 7), 'in', cond_13421, list_13422)
    
    # Testing the type of an if condition (line 1455)
    if_condition_13426 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1455, 4), result_contains_13425)
    # Assigning a type to the variable 'if_condition_13426' (line 1455)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1455, 4), 'if_condition_13426', if_condition_13426)
    # SSA begins for if statement (line 1455)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 1456):
    
    # Assigning a Call to a Name (line 1456):
    
    # Call to lower(...): (line 1456)
    # Processing the call keyword arguments (line 1456)
    kwargs_13431 = {}
    # Getting the type of 'u' (line 1456)
    u_13427 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1456, 12), 'u', False)
    # Obtaining the member 'dtype' of a type (line 1456)
    dtype_13428 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1456, 12), u_13427, 'dtype')
    # Obtaining the member 'char' of a type (line 1456)
    char_13429 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1456, 12), dtype_13428, 'char')
    # Obtaining the member 'lower' of a type (line 1456)
    lower_13430 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1456, 12), char_13429, 'lower')
    # Calling lower(args, kwargs) (line 1456)
    lower_call_result_13432 = invoke(stypy.reporting.localization.Localization(__file__, 1456, 12), lower_13430, *[], **kwargs_13431)
    
    # Assigning a type to the variable 't' (line 1456)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1456, 8), 't', lower_call_result_13432)
    
    # Assigning a Dict to a Name (line 1457):
    
    # Assigning a Dict to a Name (line 1457):
    
    # Obtaining an instance of the builtin type 'dict' (line 1457)
    dict_13433 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1457, 17), 'dict')
    # Adding type elements to the builtin type 'dict' instance (line 1457)
    # Adding element type (key, value) (line 1457)
    str_13434 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1457, 18), 'str', 'f')
    float_13435 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1457, 23), 'float')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1457, 17), dict_13433, (str_13434, float_13435))
    # Adding element type (key, value) (line 1457)
    str_13436 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1457, 28), 'str', 'd')
    float_13437 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1457, 33), 'float')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1457, 17), dict_13433, (str_13436, float_13437))
    
    # Assigning a type to the variable 'factor' (line 1457)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1457, 8), 'factor', dict_13433)
    
    # Assigning a BinOp to a Name (line 1458):
    
    # Assigning a BinOp to a Name (line 1458):
    
    # Obtaining the type of the subscript
    # Getting the type of 't' (line 1458)
    t_13438 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1458, 22), 't')
    # Getting the type of 'factor' (line 1458)
    factor_13439 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1458, 15), 'factor')
    # Obtaining the member '__getitem__' of a type (line 1458)
    getitem___13440 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1458, 15), factor_13439, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1458)
    subscript_call_result_13441 = invoke(stypy.reporting.localization.Localization(__file__, 1458, 15), getitem___13440, t_13438)
    
    
    # Call to finfo(...): (line 1458)
    # Processing the call arguments (line 1458)
    # Getting the type of 't' (line 1458)
    t_13444 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1458, 36), 't', False)
    # Processing the call keyword arguments (line 1458)
    kwargs_13445 = {}
    # Getting the type of 'np' (line 1458)
    np_13442 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1458, 27), 'np', False)
    # Obtaining the member 'finfo' of a type (line 1458)
    finfo_13443 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1458, 27), np_13442, 'finfo')
    # Calling finfo(args, kwargs) (line 1458)
    finfo_call_result_13446 = invoke(stypy.reporting.localization.Localization(__file__, 1458, 27), finfo_13443, *[t_13444], **kwargs_13445)
    
    # Obtaining the member 'eps' of a type (line 1458)
    eps_13447 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1458, 27), finfo_call_result_13446, 'eps')
    # Applying the binary operator '*' (line 1458)
    result_mul_13448 = python_operator(stypy.reporting.localization.Localization(__file__, 1458, 15), '*', subscript_call_result_13441, eps_13447)
    
    # Assigning a type to the variable 'cond' (line 1458)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1458, 8), 'cond', result_mul_13448)
    # SSA join for if statement (line 1455)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Compare to a Name (line 1461):
    
    # Assigning a Compare to a Name (line 1461):
    
    
    # Call to abs(...): (line 1461)
    # Processing the call arguments (line 1461)
    # Getting the type of 's' (line 1461)
    s_13450 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1461, 24), 's', False)
    # Processing the call keyword arguments (line 1461)
    kwargs_13451 = {}
    # Getting the type of 'abs' (line 1461)
    abs_13449 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1461, 20), 'abs', False)
    # Calling abs(args, kwargs) (line 1461)
    abs_call_result_13452 = invoke(stypy.reporting.localization.Localization(__file__, 1461, 20), abs_13449, *[s_13450], **kwargs_13451)
    
    # Getting the type of 'cond' (line 1461)
    cond_13453 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1461, 29), 'cond')
    
    # Call to max(...): (line 1461)
    # Processing the call arguments (line 1461)
    
    # Call to abs(...): (line 1461)
    # Processing the call arguments (line 1461)
    # Getting the type of 's' (line 1461)
    s_13457 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1461, 47), 's', False)
    # Processing the call keyword arguments (line 1461)
    kwargs_13458 = {}
    # Getting the type of 'abs' (line 1461)
    abs_13456 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1461, 43), 'abs', False)
    # Calling abs(args, kwargs) (line 1461)
    abs_call_result_13459 = invoke(stypy.reporting.localization.Localization(__file__, 1461, 43), abs_13456, *[s_13457], **kwargs_13458)
    
    # Processing the call keyword arguments (line 1461)
    kwargs_13460 = {}
    # Getting the type of 'np' (line 1461)
    np_13454 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1461, 36), 'np', False)
    # Obtaining the member 'max' of a type (line 1461)
    max_13455 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1461, 36), np_13454, 'max')
    # Calling max(args, kwargs) (line 1461)
    max_call_result_13461 = invoke(stypy.reporting.localization.Localization(__file__, 1461, 36), max_13455, *[abs_call_result_13459], **kwargs_13460)
    
    # Applying the binary operator '*' (line 1461)
    result_mul_13462 = python_operator(stypy.reporting.localization.Localization(__file__, 1461, 29), '*', cond_13453, max_call_result_13461)
    
    # Applying the binary operator '>' (line 1461)
    result_gt_13463 = python_operator(stypy.reporting.localization.Localization(__file__, 1461, 20), '>', abs_call_result_13452, result_mul_13462)
    
    # Assigning a type to the variable 'above_cutoff' (line 1461)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1461, 4), 'above_cutoff', result_gt_13463)
    
    # Assigning a BinOp to a Name (line 1462):
    
    # Assigning a BinOp to a Name (line 1462):
    float_13464 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1462, 18), 'float')
    
    # Obtaining the type of the subscript
    # Getting the type of 'above_cutoff' (line 1462)
    above_cutoff_13465 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1462, 26), 'above_cutoff')
    # Getting the type of 's' (line 1462)
    s_13466 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1462, 24), 's')
    # Obtaining the member '__getitem__' of a type (line 1462)
    getitem___13467 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1462, 24), s_13466, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1462)
    subscript_call_result_13468 = invoke(stypy.reporting.localization.Localization(__file__, 1462, 24), getitem___13467, above_cutoff_13465)
    
    # Applying the binary operator 'div' (line 1462)
    result_div_13469 = python_operator(stypy.reporting.localization.Localization(__file__, 1462, 18), 'div', float_13464, subscript_call_result_13468)
    
    # Assigning a type to the variable 'psigma_diag' (line 1462)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1462, 4), 'psigma_diag', result_div_13469)
    
    # Assigning a Subscript to a Name (line 1463):
    
    # Assigning a Subscript to a Name (line 1463):
    
    # Obtaining the type of the subscript
    slice_13470 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 1463, 8), None, None, None)
    # Getting the type of 'above_cutoff' (line 1463)
    above_cutoff_13471 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1463, 13), 'above_cutoff')
    # Getting the type of 'u' (line 1463)
    u_13472 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1463, 8), 'u')
    # Obtaining the member '__getitem__' of a type (line 1463)
    getitem___13473 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1463, 8), u_13472, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1463)
    subscript_call_result_13474 = invoke(stypy.reporting.localization.Localization(__file__, 1463, 8), getitem___13473, (slice_13470, above_cutoff_13471))
    
    # Assigning a type to the variable 'u' (line 1463)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1463, 4), 'u', subscript_call_result_13474)
    
    # Assigning a Call to a Name (line 1465):
    
    # Assigning a Call to a Name (line 1465):
    
    # Call to dot(...): (line 1465)
    # Processing the call arguments (line 1465)
    # Getting the type of 'u' (line 1465)
    u_13477 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1465, 15), 'u', False)
    # Getting the type of 'psigma_diag' (line 1465)
    psigma_diag_13478 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1465, 19), 'psigma_diag', False)
    # Applying the binary operator '*' (line 1465)
    result_mul_13479 = python_operator(stypy.reporting.localization.Localization(__file__, 1465, 15), '*', u_13477, psigma_diag_13478)
    
    
    # Call to conjugate(...): (line 1465)
    # Processing the call arguments (line 1465)
    # Getting the type of 'u' (line 1465)
    u_13482 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1465, 45), 'u', False)
    # Processing the call keyword arguments (line 1465)
    kwargs_13483 = {}
    # Getting the type of 'np' (line 1465)
    np_13480 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1465, 32), 'np', False)
    # Obtaining the member 'conjugate' of a type (line 1465)
    conjugate_13481 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1465, 32), np_13480, 'conjugate')
    # Calling conjugate(args, kwargs) (line 1465)
    conjugate_call_result_13484 = invoke(stypy.reporting.localization.Localization(__file__, 1465, 32), conjugate_13481, *[u_13482], **kwargs_13483)
    
    # Obtaining the member 'T' of a type (line 1465)
    T_13485 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1465, 32), conjugate_call_result_13484, 'T')
    # Processing the call keyword arguments (line 1465)
    kwargs_13486 = {}
    # Getting the type of 'np' (line 1465)
    np_13475 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1465, 8), 'np', False)
    # Obtaining the member 'dot' of a type (line 1465)
    dot_13476 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1465, 8), np_13475, 'dot')
    # Calling dot(args, kwargs) (line 1465)
    dot_call_result_13487 = invoke(stypy.reporting.localization.Localization(__file__, 1465, 8), dot_13476, *[result_mul_13479, T_13485], **kwargs_13486)
    
    # Assigning a type to the variable 'B' (line 1465)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1465, 4), 'B', dot_call_result_13487)
    
    # Getting the type of 'return_rank' (line 1467)
    return_rank_13488 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1467, 7), 'return_rank')
    # Testing the type of an if condition (line 1467)
    if_condition_13489 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1467, 4), return_rank_13488)
    # Assigning a type to the variable 'if_condition_13489' (line 1467)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1467, 4), 'if_condition_13489', if_condition_13489)
    # SSA begins for if statement (line 1467)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Obtaining an instance of the builtin type 'tuple' (line 1468)
    tuple_13490 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1468, 15), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 1468)
    # Adding element type (line 1468)
    # Getting the type of 'B' (line 1468)
    B_13491 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1468, 15), 'B')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1468, 15), tuple_13490, B_13491)
    # Adding element type (line 1468)
    
    # Call to len(...): (line 1468)
    # Processing the call arguments (line 1468)
    # Getting the type of 'psigma_diag' (line 1468)
    psigma_diag_13493 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1468, 22), 'psigma_diag', False)
    # Processing the call keyword arguments (line 1468)
    kwargs_13494 = {}
    # Getting the type of 'len' (line 1468)
    len_13492 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1468, 18), 'len', False)
    # Calling len(args, kwargs) (line 1468)
    len_call_result_13495 = invoke(stypy.reporting.localization.Localization(__file__, 1468, 18), len_13492, *[psigma_diag_13493], **kwargs_13494)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1468, 15), tuple_13490, len_call_result_13495)
    
    # Assigning a type to the variable 'stypy_return_type' (line 1468)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1468, 8), 'stypy_return_type', tuple_13490)
    # SSA branch for the else part of an if statement (line 1467)
    module_type_store.open_ssa_branch('else')
    # Getting the type of 'B' (line 1470)
    B_13496 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1470, 15), 'B')
    # Assigning a type to the variable 'stypy_return_type' (line 1470)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1470, 8), 'stypy_return_type', B_13496)
    # SSA join for if statement (line 1467)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'pinvh(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'pinvh' in the type store
    # Getting the type of 'stypy_return_type' (line 1397)
    stypy_return_type_13497 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1397, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_13497)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'pinvh'
    return stypy_return_type_13497

# Assigning a type to the variable 'pinvh' (line 1397)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1397, 0), 'pinvh', pinvh)

@norecursion
def matrix_balance(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'True' (line 1473)
    True_13498 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1473, 30), 'True')
    # Getting the type of 'True' (line 1473)
    True_13499 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1473, 42), 'True')
    # Getting the type of 'False' (line 1473)
    False_13500 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1473, 57), 'False')
    # Getting the type of 'False' (line 1474)
    False_13501 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1474, 31), 'False')
    defaults = [True_13498, True_13499, False_13500, False_13501]
    # Create a new context for function 'matrix_balance'
    module_type_store = module_type_store.open_function_context('matrix_balance', 1473, 0, False)
    
    # Passed parameters checking function
    matrix_balance.stypy_localization = localization
    matrix_balance.stypy_type_of_self = None
    matrix_balance.stypy_type_store = module_type_store
    matrix_balance.stypy_function_name = 'matrix_balance'
    matrix_balance.stypy_param_names_list = ['A', 'permute', 'scale', 'separate', 'overwrite_a']
    matrix_balance.stypy_varargs_param_name = None
    matrix_balance.stypy_kwargs_param_name = None
    matrix_balance.stypy_call_defaults = defaults
    matrix_balance.stypy_call_varargs = varargs
    matrix_balance.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'matrix_balance', ['A', 'permute', 'scale', 'separate', 'overwrite_a'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'matrix_balance', localization, ['A', 'permute', 'scale', 'separate', 'overwrite_a'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'matrix_balance(...)' code ##################

    str_13502 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1573, (-1)), 'str', '\n    Compute a diagonal similarity transformation for row/column balancing.\n\n    The balancing tries to equalize the row and column 1-norms by applying\n    a similarity transformation such that the magnitude variation of the\n    matrix entries is reflected to the scaling matrices.\n\n    Moreover, if enabled, the matrix is first permuted to isolate the upper\n    triangular parts of the matrix and, again if scaling is also enabled,\n    only the remaining subblocks are subjected to scaling.\n\n    The balanced matrix satisfies the following equality\n\n    .. math::\n\n                        B = T^{-1} A T\n\n    The scaling coefficients are approximated to the nearest power of 2\n    to avoid round-off errors.\n\n    Parameters\n    ----------\n    A : (n, n) array_like\n        Square data matrix for the balancing.\n    permute : bool, optional\n        The selector to define whether permutation of A is also performed\n        prior to scaling.\n    scale : bool, optional\n        The selector to turn on and off the scaling. If False, the matrix\n        will not be scaled.\n    separate : bool, optional\n        This switches from returning a full matrix of the transformation\n        to a tuple of two separate 1D permutation and scaling arrays.\n    overwrite_a : bool, optional\n        This is passed to xGEBAL directly. Essentially, overwrites the result\n        to the data. It might increase the space efficiency. See LAPACK manual\n        for details. This is False by default.\n\n    Returns\n    -------\n    B : (n, n) ndarray\n        Balanced matrix\n    T : (n, n) ndarray\n        A possibly permuted diagonal matrix whose nonzero entries are\n        integer powers of 2 to avoid numerical truncation errors.\n    scale, perm : (n,) ndarray\n        If ``separate`` keyword is set to True then instead of the array\n        ``T`` above, the scaling and the permutation vectors are given\n        separately as a tuple without allocating the full array ``T``.\n\n    .. versionadded:: 0.19.0\n\n    Notes\n    -----\n\n    This algorithm is particularly useful for eigenvalue and matrix\n    decompositions and in many cases it is already called by various\n    LAPACK routines.\n\n    The algorithm is based on the well-known technique of [1]_ and has\n    been modified to account for special cases. See [2]_ for details\n    which have been implemented since LAPACK v3.5.0. Before this version\n    there are corner cases where balancing can actually worsen the\n    conditioning. See [3]_ for such examples.\n\n    The code is a wrapper around LAPACK\'s xGEBAL routine family for matrix\n    balancing.\n\n    Examples\n    --------\n    >>> from scipy import linalg\n    >>> x = np.array([[1,2,0], [9,1,0.01], [1,2,10*np.pi]])\n\n    >>> y, permscale = linalg.matrix_balance(x)\n    >>> np.abs(x).sum(axis=0) / np.abs(x).sum(axis=1)\n    array([ 3.66666667,  0.4995005 ,  0.91312162])\n\n    >>> np.abs(y).sum(axis=0) / np.abs(y).sum(axis=1)\n    array([ 1.2       ,  1.27041742,  0.92658316])  # may vary\n\n    >>> permscale  # only powers of 2 (0.5 == 2^(-1))\n    array([[  0.5,   0. ,  0. ],  # may vary\n           [  0. ,   1. ,  0. ],\n           [  0. ,   0. ,  1. ]])\n\n    References\n    ----------\n    .. [1] : B.N. Parlett and C. Reinsch, "Balancing a Matrix for\n       Calculation of Eigenvalues and Eigenvectors", Numerische Mathematik,\n       Vol.13(4), 1969, DOI:10.1007/BF02165404\n\n    .. [2] : R. James, J. Langou, B.R. Lowery, "On matrix balancing and\n       eigenvector computation", 2014, Available online:\n       http://arxiv.org/abs/1401.5766\n\n    .. [3] :  D.S. Watkins. A case where balancing is harmful.\n       Electron. Trans. Numer. Anal, Vol.23, 2006.\n\n    ')
    
    # Assigning a Call to a Name (line 1575):
    
    # Assigning a Call to a Name (line 1575):
    
    # Call to atleast_2d(...): (line 1575)
    # Processing the call arguments (line 1575)
    
    # Call to _asarray_validated(...): (line 1575)
    # Processing the call arguments (line 1575)
    # Getting the type of 'A' (line 1575)
    A_13506 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1575, 41), 'A', False)
    # Processing the call keyword arguments (line 1575)
    # Getting the type of 'True' (line 1575)
    True_13507 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1575, 57), 'True', False)
    keyword_13508 = True_13507
    kwargs_13509 = {'check_finite': keyword_13508}
    # Getting the type of '_asarray_validated' (line 1575)
    _asarray_validated_13505 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1575, 22), '_asarray_validated', False)
    # Calling _asarray_validated(args, kwargs) (line 1575)
    _asarray_validated_call_result_13510 = invoke(stypy.reporting.localization.Localization(__file__, 1575, 22), _asarray_validated_13505, *[A_13506], **kwargs_13509)
    
    # Processing the call keyword arguments (line 1575)
    kwargs_13511 = {}
    # Getting the type of 'np' (line 1575)
    np_13503 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1575, 8), 'np', False)
    # Obtaining the member 'atleast_2d' of a type (line 1575)
    atleast_2d_13504 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1575, 8), np_13503, 'atleast_2d')
    # Calling atleast_2d(args, kwargs) (line 1575)
    atleast_2d_call_result_13512 = invoke(stypy.reporting.localization.Localization(__file__, 1575, 8), atleast_2d_13504, *[_asarray_validated_call_result_13510], **kwargs_13511)
    
    # Assigning a type to the variable 'A' (line 1575)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1575, 4), 'A', atleast_2d_call_result_13512)
    
    
    
    # Call to equal(...): (line 1577)
    # Getting the type of 'A' (line 1577)
    A_13515 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1577, 21), 'A', False)
    # Obtaining the member 'shape' of a type (line 1577)
    shape_13516 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1577, 21), A_13515, 'shape')
    # Processing the call keyword arguments (line 1577)
    kwargs_13517 = {}
    # Getting the type of 'np' (line 1577)
    np_13513 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1577, 11), 'np', False)
    # Obtaining the member 'equal' of a type (line 1577)
    equal_13514 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1577, 11), np_13513, 'equal')
    # Calling equal(args, kwargs) (line 1577)
    equal_call_result_13518 = invoke(stypy.reporting.localization.Localization(__file__, 1577, 11), equal_13514, *[shape_13516], **kwargs_13517)
    
    # Applying the 'not' unary operator (line 1577)
    result_not__13519 = python_operator(stypy.reporting.localization.Localization(__file__, 1577, 7), 'not', equal_call_result_13518)
    
    # Testing the type of an if condition (line 1577)
    if_condition_13520 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1577, 4), result_not__13519)
    # Assigning a type to the variable 'if_condition_13520' (line 1577)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1577, 4), 'if_condition_13520', if_condition_13520)
    # SSA begins for if statement (line 1577)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 1578)
    # Processing the call arguments (line 1578)
    str_13522 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1578, 25), 'str', 'The data matrix for balancing should be square.')
    # Processing the call keyword arguments (line 1578)
    kwargs_13523 = {}
    # Getting the type of 'ValueError' (line 1578)
    ValueError_13521 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1578, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 1578)
    ValueError_call_result_13524 = invoke(stypy.reporting.localization.Localization(__file__, 1578, 14), ValueError_13521, *[str_13522], **kwargs_13523)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 1578, 8), ValueError_call_result_13524, 'raise parameter', BaseException)
    # SSA join for if statement (line 1577)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 1580):
    
    # Assigning a Call to a Name (line 1580):
    
    # Call to get_lapack_funcs(...): (line 1580)
    # Processing the call arguments (line 1580)
    str_13526 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1580, 30), 'str', 'gebal')
    
    # Obtaining an instance of the builtin type 'tuple' (line 1580)
    tuple_13527 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1580, 41), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 1580)
    # Adding element type (line 1580)
    # Getting the type of 'A' (line 1580)
    A_13528 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1580, 41), 'A', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1580, 41), tuple_13527, A_13528)
    
    # Processing the call keyword arguments (line 1580)
    kwargs_13529 = {}
    # Getting the type of 'get_lapack_funcs' (line 1580)
    get_lapack_funcs_13525 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1580, 12), 'get_lapack_funcs', False)
    # Calling get_lapack_funcs(args, kwargs) (line 1580)
    get_lapack_funcs_call_result_13530 = invoke(stypy.reporting.localization.Localization(__file__, 1580, 12), get_lapack_funcs_13525, *[str_13526, tuple_13527], **kwargs_13529)
    
    # Assigning a type to the variable 'gebal' (line 1580)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1580, 4), 'gebal', get_lapack_funcs_call_result_13530)
    
    # Assigning a Call to a Tuple (line 1581):
    
    # Assigning a Subscript to a Name (line 1581):
    
    # Obtaining the type of the subscript
    int_13531 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1581, 4), 'int')
    
    # Call to gebal(...): (line 1581)
    # Processing the call arguments (line 1581)
    # Getting the type of 'A' (line 1581)
    A_13533 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1581, 32), 'A', False)
    # Processing the call keyword arguments (line 1581)
    # Getting the type of 'scale' (line 1581)
    scale_13534 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1581, 41), 'scale', False)
    keyword_13535 = scale_13534
    # Getting the type of 'permute' (line 1581)
    permute_13536 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1581, 56), 'permute', False)
    keyword_13537 = permute_13536
    # Getting the type of 'overwrite_a' (line 1582)
    overwrite_a_13538 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1582, 44), 'overwrite_a', False)
    keyword_13539 = overwrite_a_13538
    kwargs_13540 = {'scale': keyword_13535, 'overwrite_a': keyword_13539, 'permute': keyword_13537}
    # Getting the type of 'gebal' (line 1581)
    gebal_13532 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1581, 26), 'gebal', False)
    # Calling gebal(args, kwargs) (line 1581)
    gebal_call_result_13541 = invoke(stypy.reporting.localization.Localization(__file__, 1581, 26), gebal_13532, *[A_13533], **kwargs_13540)
    
    # Obtaining the member '__getitem__' of a type (line 1581)
    getitem___13542 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1581, 4), gebal_call_result_13541, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1581)
    subscript_call_result_13543 = invoke(stypy.reporting.localization.Localization(__file__, 1581, 4), getitem___13542, int_13531)
    
    # Assigning a type to the variable 'tuple_var_assignment_10136' (line 1581)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1581, 4), 'tuple_var_assignment_10136', subscript_call_result_13543)
    
    # Assigning a Subscript to a Name (line 1581):
    
    # Obtaining the type of the subscript
    int_13544 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1581, 4), 'int')
    
    # Call to gebal(...): (line 1581)
    # Processing the call arguments (line 1581)
    # Getting the type of 'A' (line 1581)
    A_13546 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1581, 32), 'A', False)
    # Processing the call keyword arguments (line 1581)
    # Getting the type of 'scale' (line 1581)
    scale_13547 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1581, 41), 'scale', False)
    keyword_13548 = scale_13547
    # Getting the type of 'permute' (line 1581)
    permute_13549 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1581, 56), 'permute', False)
    keyword_13550 = permute_13549
    # Getting the type of 'overwrite_a' (line 1582)
    overwrite_a_13551 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1582, 44), 'overwrite_a', False)
    keyword_13552 = overwrite_a_13551
    kwargs_13553 = {'scale': keyword_13548, 'overwrite_a': keyword_13552, 'permute': keyword_13550}
    # Getting the type of 'gebal' (line 1581)
    gebal_13545 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1581, 26), 'gebal', False)
    # Calling gebal(args, kwargs) (line 1581)
    gebal_call_result_13554 = invoke(stypy.reporting.localization.Localization(__file__, 1581, 26), gebal_13545, *[A_13546], **kwargs_13553)
    
    # Obtaining the member '__getitem__' of a type (line 1581)
    getitem___13555 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1581, 4), gebal_call_result_13554, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1581)
    subscript_call_result_13556 = invoke(stypy.reporting.localization.Localization(__file__, 1581, 4), getitem___13555, int_13544)
    
    # Assigning a type to the variable 'tuple_var_assignment_10137' (line 1581)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1581, 4), 'tuple_var_assignment_10137', subscript_call_result_13556)
    
    # Assigning a Subscript to a Name (line 1581):
    
    # Obtaining the type of the subscript
    int_13557 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1581, 4), 'int')
    
    # Call to gebal(...): (line 1581)
    # Processing the call arguments (line 1581)
    # Getting the type of 'A' (line 1581)
    A_13559 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1581, 32), 'A', False)
    # Processing the call keyword arguments (line 1581)
    # Getting the type of 'scale' (line 1581)
    scale_13560 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1581, 41), 'scale', False)
    keyword_13561 = scale_13560
    # Getting the type of 'permute' (line 1581)
    permute_13562 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1581, 56), 'permute', False)
    keyword_13563 = permute_13562
    # Getting the type of 'overwrite_a' (line 1582)
    overwrite_a_13564 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1582, 44), 'overwrite_a', False)
    keyword_13565 = overwrite_a_13564
    kwargs_13566 = {'scale': keyword_13561, 'overwrite_a': keyword_13565, 'permute': keyword_13563}
    # Getting the type of 'gebal' (line 1581)
    gebal_13558 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1581, 26), 'gebal', False)
    # Calling gebal(args, kwargs) (line 1581)
    gebal_call_result_13567 = invoke(stypy.reporting.localization.Localization(__file__, 1581, 26), gebal_13558, *[A_13559], **kwargs_13566)
    
    # Obtaining the member '__getitem__' of a type (line 1581)
    getitem___13568 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1581, 4), gebal_call_result_13567, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1581)
    subscript_call_result_13569 = invoke(stypy.reporting.localization.Localization(__file__, 1581, 4), getitem___13568, int_13557)
    
    # Assigning a type to the variable 'tuple_var_assignment_10138' (line 1581)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1581, 4), 'tuple_var_assignment_10138', subscript_call_result_13569)
    
    # Assigning a Subscript to a Name (line 1581):
    
    # Obtaining the type of the subscript
    int_13570 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1581, 4), 'int')
    
    # Call to gebal(...): (line 1581)
    # Processing the call arguments (line 1581)
    # Getting the type of 'A' (line 1581)
    A_13572 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1581, 32), 'A', False)
    # Processing the call keyword arguments (line 1581)
    # Getting the type of 'scale' (line 1581)
    scale_13573 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1581, 41), 'scale', False)
    keyword_13574 = scale_13573
    # Getting the type of 'permute' (line 1581)
    permute_13575 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1581, 56), 'permute', False)
    keyword_13576 = permute_13575
    # Getting the type of 'overwrite_a' (line 1582)
    overwrite_a_13577 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1582, 44), 'overwrite_a', False)
    keyword_13578 = overwrite_a_13577
    kwargs_13579 = {'scale': keyword_13574, 'overwrite_a': keyword_13578, 'permute': keyword_13576}
    # Getting the type of 'gebal' (line 1581)
    gebal_13571 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1581, 26), 'gebal', False)
    # Calling gebal(args, kwargs) (line 1581)
    gebal_call_result_13580 = invoke(stypy.reporting.localization.Localization(__file__, 1581, 26), gebal_13571, *[A_13572], **kwargs_13579)
    
    # Obtaining the member '__getitem__' of a type (line 1581)
    getitem___13581 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1581, 4), gebal_call_result_13580, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1581)
    subscript_call_result_13582 = invoke(stypy.reporting.localization.Localization(__file__, 1581, 4), getitem___13581, int_13570)
    
    # Assigning a type to the variable 'tuple_var_assignment_10139' (line 1581)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1581, 4), 'tuple_var_assignment_10139', subscript_call_result_13582)
    
    # Assigning a Subscript to a Name (line 1581):
    
    # Obtaining the type of the subscript
    int_13583 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1581, 4), 'int')
    
    # Call to gebal(...): (line 1581)
    # Processing the call arguments (line 1581)
    # Getting the type of 'A' (line 1581)
    A_13585 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1581, 32), 'A', False)
    # Processing the call keyword arguments (line 1581)
    # Getting the type of 'scale' (line 1581)
    scale_13586 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1581, 41), 'scale', False)
    keyword_13587 = scale_13586
    # Getting the type of 'permute' (line 1581)
    permute_13588 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1581, 56), 'permute', False)
    keyword_13589 = permute_13588
    # Getting the type of 'overwrite_a' (line 1582)
    overwrite_a_13590 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1582, 44), 'overwrite_a', False)
    keyword_13591 = overwrite_a_13590
    kwargs_13592 = {'scale': keyword_13587, 'overwrite_a': keyword_13591, 'permute': keyword_13589}
    # Getting the type of 'gebal' (line 1581)
    gebal_13584 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1581, 26), 'gebal', False)
    # Calling gebal(args, kwargs) (line 1581)
    gebal_call_result_13593 = invoke(stypy.reporting.localization.Localization(__file__, 1581, 26), gebal_13584, *[A_13585], **kwargs_13592)
    
    # Obtaining the member '__getitem__' of a type (line 1581)
    getitem___13594 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1581, 4), gebal_call_result_13593, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1581)
    subscript_call_result_13595 = invoke(stypy.reporting.localization.Localization(__file__, 1581, 4), getitem___13594, int_13583)
    
    # Assigning a type to the variable 'tuple_var_assignment_10140' (line 1581)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1581, 4), 'tuple_var_assignment_10140', subscript_call_result_13595)
    
    # Assigning a Name to a Name (line 1581):
    # Getting the type of 'tuple_var_assignment_10136' (line 1581)
    tuple_var_assignment_10136_13596 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1581, 4), 'tuple_var_assignment_10136')
    # Assigning a type to the variable 'B' (line 1581)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1581, 4), 'B', tuple_var_assignment_10136_13596)
    
    # Assigning a Name to a Name (line 1581):
    # Getting the type of 'tuple_var_assignment_10137' (line 1581)
    tuple_var_assignment_10137_13597 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1581, 4), 'tuple_var_assignment_10137')
    # Assigning a type to the variable 'lo' (line 1581)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1581, 7), 'lo', tuple_var_assignment_10137_13597)
    
    # Assigning a Name to a Name (line 1581):
    # Getting the type of 'tuple_var_assignment_10138' (line 1581)
    tuple_var_assignment_10138_13598 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1581, 4), 'tuple_var_assignment_10138')
    # Assigning a type to the variable 'hi' (line 1581)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1581, 11), 'hi', tuple_var_assignment_10138_13598)
    
    # Assigning a Name to a Name (line 1581):
    # Getting the type of 'tuple_var_assignment_10139' (line 1581)
    tuple_var_assignment_10139_13599 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1581, 4), 'tuple_var_assignment_10139')
    # Assigning a type to the variable 'ps' (line 1581)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1581, 15), 'ps', tuple_var_assignment_10139_13599)
    
    # Assigning a Name to a Name (line 1581):
    # Getting the type of 'tuple_var_assignment_10140' (line 1581)
    tuple_var_assignment_10140_13600 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1581, 4), 'tuple_var_assignment_10140')
    # Assigning a type to the variable 'info' (line 1581)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1581, 19), 'info', tuple_var_assignment_10140_13600)
    
    
    # Getting the type of 'info' (line 1584)
    info_13601 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1584, 7), 'info')
    int_13602 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1584, 14), 'int')
    # Applying the binary operator '<' (line 1584)
    result_lt_13603 = python_operator(stypy.reporting.localization.Localization(__file__, 1584, 7), '<', info_13601, int_13602)
    
    # Testing the type of an if condition (line 1584)
    if_condition_13604 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1584, 4), result_lt_13603)
    # Assigning a type to the variable 'if_condition_13604' (line 1584)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1584, 4), 'if_condition_13604', if_condition_13604)
    # SSA begins for if statement (line 1584)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 1585)
    # Processing the call arguments (line 1585)
    
    # Call to format(...): (line 1585)
    # Processing the call arguments (line 1585)
    
    # Getting the type of 'info' (line 1588)
    info_13608 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1588, 36), 'info', False)
    # Applying the 'usub' unary operator (line 1588)
    result___neg___13609 = python_operator(stypy.reporting.localization.Localization(__file__, 1588, 35), 'usub', info_13608)
    
    # Processing the call keyword arguments (line 1585)
    kwargs_13610 = {}
    str_13606 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1585, 25), 'str', 'xGEBAL exited with the internal error "illegal value in argument number {}.". See LAPACK documentation for the xGEBAL error codes.')
    # Obtaining the member 'format' of a type (line 1585)
    format_13607 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1585, 25), str_13606, 'format')
    # Calling format(args, kwargs) (line 1585)
    format_call_result_13611 = invoke(stypy.reporting.localization.Localization(__file__, 1585, 25), format_13607, *[result___neg___13609], **kwargs_13610)
    
    # Processing the call keyword arguments (line 1585)
    kwargs_13612 = {}
    # Getting the type of 'ValueError' (line 1585)
    ValueError_13605 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1585, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 1585)
    ValueError_call_result_13613 = invoke(stypy.reporting.localization.Localization(__file__, 1585, 14), ValueError_13605, *[format_call_result_13611], **kwargs_13612)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 1585, 8), ValueError_call_result_13613, 'raise parameter', BaseException)
    # SSA join for if statement (line 1584)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 1591):
    
    # Assigning a Call to a Name (line 1591):
    
    # Call to ones_like(...): (line 1591)
    # Processing the call arguments (line 1591)
    # Getting the type of 'ps' (line 1591)
    ps_13616 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1591, 27), 'ps', False)
    # Processing the call keyword arguments (line 1591)
    # Getting the type of 'float' (line 1591)
    float_13617 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1591, 37), 'float', False)
    keyword_13618 = float_13617
    kwargs_13619 = {'dtype': keyword_13618}
    # Getting the type of 'np' (line 1591)
    np_13614 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1591, 14), 'np', False)
    # Obtaining the member 'ones_like' of a type (line 1591)
    ones_like_13615 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1591, 14), np_13614, 'ones_like')
    # Calling ones_like(args, kwargs) (line 1591)
    ones_like_call_result_13620 = invoke(stypy.reporting.localization.Localization(__file__, 1591, 14), ones_like_13615, *[ps_13616], **kwargs_13619)
    
    # Assigning a type to the variable 'scaling' (line 1591)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1591, 4), 'scaling', ones_like_call_result_13620)
    
    # Assigning a Subscript to a Subscript (line 1592):
    
    # Assigning a Subscript to a Subscript (line 1592):
    
    # Obtaining the type of the subscript
    # Getting the type of 'lo' (line 1592)
    lo_13621 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1592, 26), 'lo')
    # Getting the type of 'hi' (line 1592)
    hi_13622 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1592, 29), 'hi')
    int_13623 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1592, 32), 'int')
    # Applying the binary operator '+' (line 1592)
    result_add_13624 = python_operator(stypy.reporting.localization.Localization(__file__, 1592, 29), '+', hi_13622, int_13623)
    
    slice_13625 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 1592, 23), lo_13621, result_add_13624, None)
    # Getting the type of 'ps' (line 1592)
    ps_13626 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1592, 23), 'ps')
    # Obtaining the member '__getitem__' of a type (line 1592)
    getitem___13627 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1592, 23), ps_13626, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1592)
    subscript_call_result_13628 = invoke(stypy.reporting.localization.Localization(__file__, 1592, 23), getitem___13627, slice_13625)
    
    # Getting the type of 'scaling' (line 1592)
    scaling_13629 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1592, 4), 'scaling')
    # Getting the type of 'lo' (line 1592)
    lo_13630 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1592, 12), 'lo')
    # Getting the type of 'hi' (line 1592)
    hi_13631 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1592, 15), 'hi')
    int_13632 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1592, 18), 'int')
    # Applying the binary operator '+' (line 1592)
    result_add_13633 = python_operator(stypy.reporting.localization.Localization(__file__, 1592, 15), '+', hi_13631, int_13632)
    
    slice_13634 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 1592, 4), lo_13630, result_add_13633, None)
    # Storing an element on a container (line 1592)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1592, 4), scaling_13629, (slice_13634, subscript_call_result_13628))
    
    # Assigning a BinOp to a Name (line 1595):
    
    # Assigning a BinOp to a Name (line 1595):
    
    # Call to astype(...): (line 1595)
    # Processing the call arguments (line 1595)
    # Getting the type of 'int' (line 1595)
    int_13637 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1595, 19), 'int', False)
    # Processing the call keyword arguments (line 1595)
    # Getting the type of 'False' (line 1595)
    False_13638 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1595, 29), 'False', False)
    keyword_13639 = False_13638
    kwargs_13640 = {'copy': keyword_13639}
    # Getting the type of 'ps' (line 1595)
    ps_13635 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1595, 9), 'ps', False)
    # Obtaining the member 'astype' of a type (line 1595)
    astype_13636 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1595, 9), ps_13635, 'astype')
    # Calling astype(args, kwargs) (line 1595)
    astype_call_result_13641 = invoke(stypy.reporting.localization.Localization(__file__, 1595, 9), astype_13636, *[int_13637], **kwargs_13640)
    
    int_13642 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1595, 38), 'int')
    # Applying the binary operator '-' (line 1595)
    result_sub_13643 = python_operator(stypy.reporting.localization.Localization(__file__, 1595, 9), '-', astype_call_result_13641, int_13642)
    
    # Assigning a type to the variable 'ps' (line 1595)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1595, 4), 'ps', result_sub_13643)
    
    # Assigning a Subscript to a Name (line 1596):
    
    # Assigning a Subscript to a Name (line 1596):
    
    # Obtaining the type of the subscript
    int_13644 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1596, 16), 'int')
    # Getting the type of 'A' (line 1596)
    A_13645 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1596, 8), 'A')
    # Obtaining the member 'shape' of a type (line 1596)
    shape_13646 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1596, 8), A_13645, 'shape')
    # Obtaining the member '__getitem__' of a type (line 1596)
    getitem___13647 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1596, 8), shape_13646, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1596)
    subscript_call_result_13648 = invoke(stypy.reporting.localization.Localization(__file__, 1596, 8), getitem___13647, int_13644)
    
    # Assigning a type to the variable 'n' (line 1596)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1596, 4), 'n', subscript_call_result_13648)
    
    # Assigning a Call to a Name (line 1597):
    
    # Assigning a Call to a Name (line 1597):
    
    # Call to arange(...): (line 1597)
    # Processing the call arguments (line 1597)
    # Getting the type of 'n' (line 1597)
    n_13651 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1597, 21), 'n', False)
    # Processing the call keyword arguments (line 1597)
    kwargs_13652 = {}
    # Getting the type of 'np' (line 1597)
    np_13649 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1597, 11), 'np', False)
    # Obtaining the member 'arange' of a type (line 1597)
    arange_13650 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1597, 11), np_13649, 'arange')
    # Calling arange(args, kwargs) (line 1597)
    arange_call_result_13653 = invoke(stypy.reporting.localization.Localization(__file__, 1597, 11), arange_13650, *[n_13651], **kwargs_13652)
    
    # Assigning a type to the variable 'perm' (line 1597)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1597, 4), 'perm', arange_call_result_13653)
    
    
    # Getting the type of 'hi' (line 1600)
    hi_13654 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1600, 7), 'hi')
    # Getting the type of 'n' (line 1600)
    n_13655 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1600, 12), 'n')
    # Applying the binary operator '<' (line 1600)
    result_lt_13656 = python_operator(stypy.reporting.localization.Localization(__file__, 1600, 7), '<', hi_13654, n_13655)
    
    # Testing the type of an if condition (line 1600)
    if_condition_13657 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1600, 4), result_lt_13656)
    # Assigning a type to the variable 'if_condition_13657' (line 1600)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1600, 4), 'if_condition_13657', if_condition_13657)
    # SSA begins for if statement (line 1600)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    
    # Call to enumerate(...): (line 1601)
    # Processing the call arguments (line 1601)
    
    # Obtaining the type of the subscript
    int_13659 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1601, 44), 'int')
    slice_13660 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 1601, 32), None, None, int_13659)
    
    # Obtaining the type of the subscript
    # Getting the type of 'hi' (line 1601)
    hi_13661 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1601, 35), 'hi', False)
    int_13662 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1601, 38), 'int')
    # Applying the binary operator '+' (line 1601)
    result_add_13663 = python_operator(stypy.reporting.localization.Localization(__file__, 1601, 35), '+', hi_13661, int_13662)
    
    slice_13664 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 1601, 32), result_add_13663, None, None)
    # Getting the type of 'ps' (line 1601)
    ps_13665 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1601, 32), 'ps', False)
    # Obtaining the member '__getitem__' of a type (line 1601)
    getitem___13666 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1601, 32), ps_13665, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1601)
    subscript_call_result_13667 = invoke(stypy.reporting.localization.Localization(__file__, 1601, 32), getitem___13666, slice_13664)
    
    # Obtaining the member '__getitem__' of a type (line 1601)
    getitem___13668 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1601, 32), subscript_call_result_13667, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1601)
    subscript_call_result_13669 = invoke(stypy.reporting.localization.Localization(__file__, 1601, 32), getitem___13668, slice_13660)
    
    int_13670 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1601, 49), 'int')
    # Processing the call keyword arguments (line 1601)
    kwargs_13671 = {}
    # Getting the type of 'enumerate' (line 1601)
    enumerate_13658 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1601, 22), 'enumerate', False)
    # Calling enumerate(args, kwargs) (line 1601)
    enumerate_call_result_13672 = invoke(stypy.reporting.localization.Localization(__file__, 1601, 22), enumerate_13658, *[subscript_call_result_13669, int_13670], **kwargs_13671)
    
    # Testing the type of a for loop iterable (line 1601)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 1601, 8), enumerate_call_result_13672)
    # Getting the type of the for loop variable (line 1601)
    for_loop_var_13673 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 1601, 8), enumerate_call_result_13672)
    # Assigning a type to the variable 'ind' (line 1601)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1601, 8), 'ind', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1601, 8), for_loop_var_13673))
    # Assigning a type to the variable 'x' (line 1601)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1601, 8), 'x', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1601, 8), for_loop_var_13673))
    # SSA begins for a for statement (line 1601)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    
    # Getting the type of 'n' (line 1602)
    n_13674 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1602, 15), 'n')
    # Getting the type of 'ind' (line 1602)
    ind_13675 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1602, 17), 'ind')
    # Applying the binary operator '-' (line 1602)
    result_sub_13676 = python_operator(stypy.reporting.localization.Localization(__file__, 1602, 15), '-', n_13674, ind_13675)
    
    # Getting the type of 'x' (line 1602)
    x_13677 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1602, 24), 'x')
    # Applying the binary operator '==' (line 1602)
    result_eq_13678 = python_operator(stypy.reporting.localization.Localization(__file__, 1602, 15), '==', result_sub_13676, x_13677)
    
    # Testing the type of an if condition (line 1602)
    if_condition_13679 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1602, 12), result_eq_13678)
    # Assigning a type to the variable 'if_condition_13679' (line 1602)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1602, 12), 'if_condition_13679', if_condition_13679)
    # SSA begins for if statement (line 1602)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # SSA join for if statement (line 1602)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Subscript to a Subscript (line 1604):
    
    # Assigning a Subscript to a Subscript (line 1604):
    
    # Obtaining the type of the subscript
    
    # Obtaining an instance of the builtin type 'list' (line 1604)
    list_13680 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1604, 36), 'list')
    # Adding type elements to the builtin type 'list' instance (line 1604)
    # Adding element type (line 1604)
    # Getting the type of 'n' (line 1604)
    n_13681 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1604, 37), 'n')
    # Getting the type of 'ind' (line 1604)
    ind_13682 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1604, 39), 'ind')
    # Applying the binary operator '-' (line 1604)
    result_sub_13683 = python_operator(stypy.reporting.localization.Localization(__file__, 1604, 37), '-', n_13681, ind_13682)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1604, 36), list_13680, result_sub_13683)
    # Adding element type (line 1604)
    # Getting the type of 'x' (line 1604)
    x_13684 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1604, 44), 'x')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1604, 36), list_13680, x_13684)
    
    # Getting the type of 'perm' (line 1604)
    perm_13685 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1604, 31), 'perm')
    # Obtaining the member '__getitem__' of a type (line 1604)
    getitem___13686 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1604, 31), perm_13685, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1604)
    subscript_call_result_13687 = invoke(stypy.reporting.localization.Localization(__file__, 1604, 31), getitem___13686, list_13680)
    
    # Getting the type of 'perm' (line 1604)
    perm_13688 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1604, 12), 'perm')
    
    # Obtaining an instance of the builtin type 'list' (line 1604)
    list_13689 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1604, 17), 'list')
    # Adding type elements to the builtin type 'list' instance (line 1604)
    # Adding element type (line 1604)
    # Getting the type of 'x' (line 1604)
    x_13690 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1604, 18), 'x')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1604, 17), list_13689, x_13690)
    # Adding element type (line 1604)
    # Getting the type of 'n' (line 1604)
    n_13691 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1604, 21), 'n')
    # Getting the type of 'ind' (line 1604)
    ind_13692 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1604, 23), 'ind')
    # Applying the binary operator '-' (line 1604)
    result_sub_13693 = python_operator(stypy.reporting.localization.Localization(__file__, 1604, 21), '-', n_13691, ind_13692)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1604, 17), list_13689, result_sub_13693)
    
    # Storing an element on a container (line 1604)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1604, 12), perm_13688, (list_13689, subscript_call_result_13687))
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 1600)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'lo' (line 1606)
    lo_13694 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1606, 7), 'lo')
    int_13695 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1606, 12), 'int')
    # Applying the binary operator '>' (line 1606)
    result_gt_13696 = python_operator(stypy.reporting.localization.Localization(__file__, 1606, 7), '>', lo_13694, int_13695)
    
    # Testing the type of an if condition (line 1606)
    if_condition_13697 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1606, 4), result_gt_13696)
    # Assigning a type to the variable 'if_condition_13697' (line 1606)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1606, 4), 'if_condition_13697', if_condition_13697)
    # SSA begins for if statement (line 1606)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    
    # Call to enumerate(...): (line 1607)
    # Processing the call arguments (line 1607)
    
    # Obtaining the type of the subscript
    # Getting the type of 'lo' (line 1607)
    lo_13699 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1607, 36), 'lo', False)
    slice_13700 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 1607, 32), None, lo_13699, None)
    # Getting the type of 'ps' (line 1607)
    ps_13701 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1607, 32), 'ps', False)
    # Obtaining the member '__getitem__' of a type (line 1607)
    getitem___13702 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1607, 32), ps_13701, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1607)
    subscript_call_result_13703 = invoke(stypy.reporting.localization.Localization(__file__, 1607, 32), getitem___13702, slice_13700)
    
    # Processing the call keyword arguments (line 1607)
    kwargs_13704 = {}
    # Getting the type of 'enumerate' (line 1607)
    enumerate_13698 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1607, 22), 'enumerate', False)
    # Calling enumerate(args, kwargs) (line 1607)
    enumerate_call_result_13705 = invoke(stypy.reporting.localization.Localization(__file__, 1607, 22), enumerate_13698, *[subscript_call_result_13703], **kwargs_13704)
    
    # Testing the type of a for loop iterable (line 1607)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 1607, 8), enumerate_call_result_13705)
    # Getting the type of the for loop variable (line 1607)
    for_loop_var_13706 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 1607, 8), enumerate_call_result_13705)
    # Assigning a type to the variable 'ind' (line 1607)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1607, 8), 'ind', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1607, 8), for_loop_var_13706))
    # Assigning a type to the variable 'x' (line 1607)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1607, 8), 'x', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1607, 8), for_loop_var_13706))
    # SSA begins for a for statement (line 1607)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    
    # Getting the type of 'ind' (line 1608)
    ind_13707 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1608, 15), 'ind')
    # Getting the type of 'x' (line 1608)
    x_13708 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1608, 22), 'x')
    # Applying the binary operator '==' (line 1608)
    result_eq_13709 = python_operator(stypy.reporting.localization.Localization(__file__, 1608, 15), '==', ind_13707, x_13708)
    
    # Testing the type of an if condition (line 1608)
    if_condition_13710 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1608, 12), result_eq_13709)
    # Assigning a type to the variable 'if_condition_13710' (line 1608)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1608, 12), 'if_condition_13710', if_condition_13710)
    # SSA begins for if statement (line 1608)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # SSA join for if statement (line 1608)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Subscript to a Subscript (line 1610):
    
    # Assigning a Subscript to a Subscript (line 1610):
    
    # Obtaining the type of the subscript
    
    # Obtaining an instance of the builtin type 'list' (line 1610)
    list_13711 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1610, 34), 'list')
    # Adding type elements to the builtin type 'list' instance (line 1610)
    # Adding element type (line 1610)
    # Getting the type of 'ind' (line 1610)
    ind_13712 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1610, 35), 'ind')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1610, 34), list_13711, ind_13712)
    # Adding element type (line 1610)
    # Getting the type of 'x' (line 1610)
    x_13713 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1610, 40), 'x')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1610, 34), list_13711, x_13713)
    
    # Getting the type of 'perm' (line 1610)
    perm_13714 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1610, 29), 'perm')
    # Obtaining the member '__getitem__' of a type (line 1610)
    getitem___13715 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1610, 29), perm_13714, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1610)
    subscript_call_result_13716 = invoke(stypy.reporting.localization.Localization(__file__, 1610, 29), getitem___13715, list_13711)
    
    # Getting the type of 'perm' (line 1610)
    perm_13717 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1610, 12), 'perm')
    
    # Obtaining an instance of the builtin type 'list' (line 1610)
    list_13718 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1610, 17), 'list')
    # Adding type elements to the builtin type 'list' instance (line 1610)
    # Adding element type (line 1610)
    # Getting the type of 'x' (line 1610)
    x_13719 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1610, 18), 'x')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1610, 17), list_13718, x_13719)
    # Adding element type (line 1610)
    # Getting the type of 'ind' (line 1610)
    ind_13720 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1610, 21), 'ind')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1610, 17), list_13718, ind_13720)
    
    # Storing an element on a container (line 1610)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1610, 12), perm_13717, (list_13718, subscript_call_result_13716))
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 1606)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Getting the type of 'separate' (line 1612)
    separate_13721 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1612, 7), 'separate')
    # Testing the type of an if condition (line 1612)
    if_condition_13722 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1612, 4), separate_13721)
    # Assigning a type to the variable 'if_condition_13722' (line 1612)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1612, 4), 'if_condition_13722', if_condition_13722)
    # SSA begins for if statement (line 1612)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Obtaining an instance of the builtin type 'tuple' (line 1613)
    tuple_13723 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1613, 15), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 1613)
    # Adding element type (line 1613)
    # Getting the type of 'B' (line 1613)
    B_13724 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1613, 15), 'B')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1613, 15), tuple_13723, B_13724)
    # Adding element type (line 1613)
    
    # Obtaining an instance of the builtin type 'tuple' (line 1613)
    tuple_13725 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1613, 19), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 1613)
    # Adding element type (line 1613)
    # Getting the type of 'scaling' (line 1613)
    scaling_13726 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1613, 19), 'scaling')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1613, 19), tuple_13725, scaling_13726)
    # Adding element type (line 1613)
    # Getting the type of 'perm' (line 1613)
    perm_13727 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1613, 28), 'perm')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1613, 19), tuple_13725, perm_13727)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1613, 15), tuple_13723, tuple_13725)
    
    # Assigning a type to the variable 'stypy_return_type' (line 1613)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1613, 8), 'stypy_return_type', tuple_13723)
    # SSA join for if statement (line 1612)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 1616):
    
    # Assigning a Call to a Name (line 1616):
    
    # Call to empty_like(...): (line 1616)
    # Processing the call arguments (line 1616)
    # Getting the type of 'perm' (line 1616)
    perm_13730 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1616, 26), 'perm', False)
    # Processing the call keyword arguments (line 1616)
    kwargs_13731 = {}
    # Getting the type of 'np' (line 1616)
    np_13728 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1616, 12), 'np', False)
    # Obtaining the member 'empty_like' of a type (line 1616)
    empty_like_13729 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1616, 12), np_13728, 'empty_like')
    # Calling empty_like(args, kwargs) (line 1616)
    empty_like_call_result_13732 = invoke(stypy.reporting.localization.Localization(__file__, 1616, 12), empty_like_13729, *[perm_13730], **kwargs_13731)
    
    # Assigning a type to the variable 'iperm' (line 1616)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1616, 4), 'iperm', empty_like_call_result_13732)
    
    # Assigning a Call to a Subscript (line 1617):
    
    # Assigning a Call to a Subscript (line 1617):
    
    # Call to arange(...): (line 1617)
    # Processing the call arguments (line 1617)
    # Getting the type of 'n' (line 1617)
    n_13735 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1617, 28), 'n', False)
    # Processing the call keyword arguments (line 1617)
    kwargs_13736 = {}
    # Getting the type of 'np' (line 1617)
    np_13733 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1617, 18), 'np', False)
    # Obtaining the member 'arange' of a type (line 1617)
    arange_13734 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1617, 18), np_13733, 'arange')
    # Calling arange(args, kwargs) (line 1617)
    arange_call_result_13737 = invoke(stypy.reporting.localization.Localization(__file__, 1617, 18), arange_13734, *[n_13735], **kwargs_13736)
    
    # Getting the type of 'iperm' (line 1617)
    iperm_13738 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1617, 4), 'iperm')
    # Getting the type of 'perm' (line 1617)
    perm_13739 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1617, 10), 'perm')
    # Storing an element on a container (line 1617)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1617, 4), iperm_13738, (perm_13739, arange_call_result_13737))
    
    # Obtaining an instance of the builtin type 'tuple' (line 1619)
    tuple_13740 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1619, 11), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 1619)
    # Adding element type (line 1619)
    # Getting the type of 'B' (line 1619)
    B_13741 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1619, 11), 'B')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1619, 11), tuple_13740, B_13741)
    # Adding element type (line 1619)
    
    # Obtaining the type of the subscript
    # Getting the type of 'iperm' (line 1619)
    iperm_13742 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1619, 31), 'iperm')
    slice_13743 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 1619, 14), None, None, None)
    
    # Call to diag(...): (line 1619)
    # Processing the call arguments (line 1619)
    # Getting the type of 'scaling' (line 1619)
    scaling_13746 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1619, 22), 'scaling', False)
    # Processing the call keyword arguments (line 1619)
    kwargs_13747 = {}
    # Getting the type of 'np' (line 1619)
    np_13744 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1619, 14), 'np', False)
    # Obtaining the member 'diag' of a type (line 1619)
    diag_13745 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1619, 14), np_13744, 'diag')
    # Calling diag(args, kwargs) (line 1619)
    diag_call_result_13748 = invoke(stypy.reporting.localization.Localization(__file__, 1619, 14), diag_13745, *[scaling_13746], **kwargs_13747)
    
    # Obtaining the member '__getitem__' of a type (line 1619)
    getitem___13749 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1619, 14), diag_call_result_13748, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1619)
    subscript_call_result_13750 = invoke(stypy.reporting.localization.Localization(__file__, 1619, 14), getitem___13749, (iperm_13742, slice_13743))
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1619, 11), tuple_13740, subscript_call_result_13750)
    
    # Assigning a type to the variable 'stypy_return_type' (line 1619)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1619, 4), 'stypy_return_type', tuple_13740)
    
    # ################# End of 'matrix_balance(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'matrix_balance' in the type store
    # Getting the type of 'stypy_return_type' (line 1473)
    stypy_return_type_13751 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1473, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_13751)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'matrix_balance'
    return stypy_return_type_13751

# Assigning a type to the variable 'matrix_balance' (line 1473)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1473, 0), 'matrix_balance', matrix_balance)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
