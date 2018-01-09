
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: #
2: # Author: Pearu Peterson, March 2002
3: #
4: # additions by Travis Oliphant, March 2002
5: # additions by Eric Jones,      June 2002
6: # additions by Johannes Loehnert, June 2006
7: # additions by Bart Vandereycken, June 2006
8: # additions by Andrew D Straw, May 2007
9: # additions by Tiziano Zito, November 2008
10: #
11: # April 2010: Functions for LU, QR, SVD, Schur and Cholesky decompositions were
12: # moved to their own files.  Still in this file are functions for eigenstuff
13: # and for the Hessenberg form.
14: 
15: from __future__ import division, print_function, absolute_import
16: 
17: __all__ = ['eig', 'eigvals', 'eigh', 'eigvalsh',
18:            'eig_banded', 'eigvals_banded',
19:            'eigh_tridiagonal', 'eigvalsh_tridiagonal', 'hessenberg']
20: 
21: import numpy
22: from numpy import (array, isfinite, inexact, nonzero, iscomplexobj, cast,
23:                    flatnonzero, conj, asarray, argsort, empty)
24: # Local imports
25: from scipy._lib.six import xrange
26: from scipy._lib._util import _asarray_validated
27: from scipy._lib.six import string_types
28: from .misc import LinAlgError, _datacopied, norm
29: from .lapack import get_lapack_funcs, _compute_lwork
30: 
31: 
32: _I = cast['F'](1j)
33: 
34: 
35: def _make_complex_eigvecs(w, vin, dtype):
36:     '''
37:     Produce complex-valued eigenvectors from LAPACK DGGEV real-valued output
38:     '''
39:     # - see LAPACK man page DGGEV at ALPHAI
40:     v = numpy.array(vin, dtype=dtype)
41:     m = (w.imag > 0)
42:     m[:-1] |= (w.imag[1:] < 0)  # workaround for LAPACK bug, cf. ticket #709
43:     for i in flatnonzero(m):
44:         v.imag[:, i] = vin[:, i+1]
45:         conj(v[:, i], v[:, i+1])
46:     return v
47: 
48: 
49: def _make_eigvals(alpha, beta, homogeneous_eigvals):
50:     if homogeneous_eigvals:
51:         if beta is None:
52:             return numpy.vstack((alpha, numpy.ones_like(alpha)))
53:         else:
54:             return numpy.vstack((alpha, beta))
55:     else:
56:         if beta is None:
57:             return alpha
58:         else:
59:             w = numpy.empty_like(alpha)
60:             alpha_zero = (alpha == 0)
61:             beta_zero = (beta == 0)
62:             beta_nonzero = ~beta_zero
63:             w[beta_nonzero] = alpha[beta_nonzero]/beta[beta_nonzero]
64:             # Use numpy.inf for complex values too since
65:             # 1/numpy.inf = 0, i.e. it correctly behaves as projective
66:             # infinity.
67:             w[~alpha_zero & beta_zero] = numpy.inf
68:             if numpy.all(alpha.imag == 0):
69:                 w[alpha_zero & beta_zero] = numpy.nan
70:             else:
71:                 w[alpha_zero & beta_zero] = complex(numpy.nan, numpy.nan)
72:             return w
73: 
74: 
75: def _geneig(a1, b1, left, right, overwrite_a, overwrite_b,
76:             homogeneous_eigvals):
77:     ggev, = get_lapack_funcs(('ggev',), (a1, b1))
78:     cvl, cvr = left, right
79:     res = ggev(a1, b1, lwork=-1)
80:     lwork = res[-2][0].real.astype(numpy.int)
81:     if ggev.typecode in 'cz':
82:         alpha, beta, vl, vr, work, info = ggev(a1, b1, cvl, cvr, lwork,
83:                                                overwrite_a, overwrite_b)
84:         w = _make_eigvals(alpha, beta, homogeneous_eigvals)
85:     else:
86:         alphar, alphai, beta, vl, vr, work, info = ggev(a1, b1, cvl, cvr,
87:                                                         lwork, overwrite_a,
88:                                                         overwrite_b)
89:         alpha = alphar + _I * alphai
90:         w = _make_eigvals(alpha, beta, homogeneous_eigvals)
91:     _check_info(info, 'generalized eig algorithm (ggev)')
92: 
93:     only_real = numpy.all(w.imag == 0.0)
94:     if not (ggev.typecode in 'cz' or only_real):
95:         t = w.dtype.char
96:         if left:
97:             vl = _make_complex_eigvecs(w, vl, t)
98:         if right:
99:             vr = _make_complex_eigvecs(w, vr, t)
100: 
101:     # the eigenvectors returned by the lapack function are NOT normalized
102:     for i in xrange(vr.shape[0]):
103:         if right:
104:             vr[:, i] /= norm(vr[:, i])
105:         if left:
106:             vl[:, i] /= norm(vl[:, i])
107: 
108:     if not (left or right):
109:         return w
110:     if left:
111:         if right:
112:             return w, vl, vr
113:         return w, vl
114:     return w, vr
115: 
116: 
117: def eig(a, b=None, left=False, right=True, overwrite_a=False,
118:         overwrite_b=False, check_finite=True, homogeneous_eigvals=False):
119:     '''
120:     Solve an ordinary or generalized eigenvalue problem of a square matrix.
121: 
122:     Find eigenvalues w and right or left eigenvectors of a general matrix::
123: 
124:         a   vr[:,i] = w[i]        b   vr[:,i]
125:         a.H vl[:,i] = w[i].conj() b.H vl[:,i]
126: 
127:     where ``.H`` is the Hermitian conjugation.
128: 
129:     Parameters
130:     ----------
131:     a : (M, M) array_like
132:         A complex or real matrix whose eigenvalues and eigenvectors
133:         will be computed.
134:     b : (M, M) array_like, optional
135:         Right-hand side matrix in a generalized eigenvalue problem.
136:         Default is None, identity matrix is assumed.
137:     left : bool, optional
138:         Whether to calculate and return left eigenvectors.  Default is False.
139:     right : bool, optional
140:         Whether to calculate and return right eigenvectors.  Default is True.
141:     overwrite_a : bool, optional
142:         Whether to overwrite `a`; may improve performance.  Default is False.
143:     overwrite_b : bool, optional
144:         Whether to overwrite `b`; may improve performance.  Default is False.
145:     check_finite : bool, optional
146:         Whether to check that the input matrices contain only finite numbers.
147:         Disabling may give a performance gain, but may result in problems
148:         (crashes, non-termination) if the inputs do contain infinities or NaNs.
149:     homogeneous_eigvals : bool, optional
150:         If True, return the eigenvalues in homogeneous coordinates.
151:         In this case ``w`` is a (2, M) array so that::
152: 
153:             w[1,i] a vr[:,i] = w[0,i] b vr[:,i]
154: 
155:         Default is False.
156: 
157:     Returns
158:     -------
159:     w : (M,) or (2, M) double or complex ndarray
160:         The eigenvalues, each repeated according to its
161:         multiplicity. The shape is (M,) unless
162:         ``homogeneous_eigvals=True``.
163:     vl : (M, M) double or complex ndarray
164:         The normalized left eigenvector corresponding to the eigenvalue
165:         ``w[i]`` is the column vl[:,i]. Only returned if ``left=True``.
166:     vr : (M, M) double or complex ndarray
167:         The normalized right eigenvector corresponding to the eigenvalue
168:         ``w[i]`` is the column ``vr[:,i]``.  Only returned if ``right=True``.
169: 
170:     Raises
171:     ------
172:     LinAlgError
173:         If eigenvalue computation does not converge.
174: 
175:     See Also
176:     --------
177:     eigvals : eigenvalues of general arrays
178:     eigh : Eigenvalues and right eigenvectors for symmetric/Hermitian arrays.
179:     eig_banded : eigenvalues and right eigenvectors for symmetric/Hermitian
180:         band matrices
181:     eigh_tridiagonal : eigenvalues and right eiegenvectors for
182:         symmetric/Hermitian tridiagonal matrices
183:     '''
184:     a1 = _asarray_validated(a, check_finite=check_finite)
185:     if len(a1.shape) != 2 or a1.shape[0] != a1.shape[1]:
186:         raise ValueError('expected square matrix')
187:     overwrite_a = overwrite_a or (_datacopied(a1, a))
188:     if b is not None:
189:         b1 = _asarray_validated(b, check_finite=check_finite)
190:         overwrite_b = overwrite_b or _datacopied(b1, b)
191:         if len(b1.shape) != 2 or b1.shape[0] != b1.shape[1]:
192:             raise ValueError('expected square matrix')
193:         if b1.shape != a1.shape:
194:             raise ValueError('a and b must have the same shape')
195:         return _geneig(a1, b1, left, right, overwrite_a, overwrite_b,
196:                        homogeneous_eigvals)
197: 
198:     geev, geev_lwork = get_lapack_funcs(('geev', 'geev_lwork'), (a1,))
199:     compute_vl, compute_vr = left, right
200: 
201:     lwork = _compute_lwork(geev_lwork, a1.shape[0],
202:                            compute_vl=compute_vl,
203:                            compute_vr=compute_vr)
204: 
205:     if geev.typecode in 'cz':
206:         w, vl, vr, info = geev(a1, lwork=lwork,
207:                                compute_vl=compute_vl,
208:                                compute_vr=compute_vr,
209:                                overwrite_a=overwrite_a)
210:         w = _make_eigvals(w, None, homogeneous_eigvals)
211:     else:
212:         wr, wi, vl, vr, info = geev(a1, lwork=lwork,
213:                                     compute_vl=compute_vl,
214:                                     compute_vr=compute_vr,
215:                                     overwrite_a=overwrite_a)
216:         t = {'f': 'F', 'd': 'D'}[wr.dtype.char]
217:         w = wr + _I * wi
218:         w = _make_eigvals(w, None, homogeneous_eigvals)
219: 
220:     _check_info(info, 'eig algorithm (geev)',
221:                 positive='did not converge (only eigenvalues '
222:                          'with order >= %d have converged)')
223: 
224:     only_real = numpy.all(w.imag == 0.0)
225:     if not (geev.typecode in 'cz' or only_real):
226:         t = w.dtype.char
227:         if left:
228:             vl = _make_complex_eigvecs(w, vl, t)
229:         if right:
230:             vr = _make_complex_eigvecs(w, vr, t)
231:     if not (left or right):
232:         return w
233:     if left:
234:         if right:
235:             return w, vl, vr
236:         return w, vl
237:     return w, vr
238: 
239: 
240: def eigh(a, b=None, lower=True, eigvals_only=False, overwrite_a=False,
241:          overwrite_b=False, turbo=True, eigvals=None, type=1,
242:          check_finite=True):
243:     '''
244:     Solve an ordinary or generalized eigenvalue problem for a complex
245:     Hermitian or real symmetric matrix.
246: 
247:     Find eigenvalues w and optionally eigenvectors v of matrix `a`, where
248:     `b` is positive definite::
249: 
250:                       a v[:,i] = w[i] b v[:,i]
251:         v[i,:].conj() a v[:,i] = w[i]
252:         v[i,:].conj() b v[:,i] = 1
253: 
254:     Parameters
255:     ----------
256:     a : (M, M) array_like
257:         A complex Hermitian or real symmetric matrix whose eigenvalues and
258:         eigenvectors will be computed.
259:     b : (M, M) array_like, optional
260:         A complex Hermitian or real symmetric definite positive matrix in.
261:         If omitted, identity matrix is assumed.
262:     lower : bool, optional
263:         Whether the pertinent array data is taken from the lower or upper
264:         triangle of `a`. (Default: lower)
265:     eigvals_only : bool, optional
266:         Whether to calculate only eigenvalues and no eigenvectors.
267:         (Default: both are calculated)
268:     turbo : bool, optional
269:         Use divide and conquer algorithm (faster but expensive in memory,
270:         only for generalized eigenvalue problem and if eigvals=None)
271:     eigvals : tuple (lo, hi), optional
272:         Indexes of the smallest and largest (in ascending order) eigenvalues
273:         and corresponding eigenvectors to be returned: 0 <= lo <= hi <= M-1.
274:         If omitted, all eigenvalues and eigenvectors are returned.
275:     type : int, optional
276:         Specifies the problem type to be solved:
277: 
278:            type = 1: a   v[:,i] = w[i] b v[:,i]
279: 
280:            type = 2: a b v[:,i] = w[i]   v[:,i]
281: 
282:            type = 3: b a v[:,i] = w[i]   v[:,i]
283:     overwrite_a : bool, optional
284:         Whether to overwrite data in `a` (may improve performance)
285:     overwrite_b : bool, optional
286:         Whether to overwrite data in `b` (may improve performance)
287:     check_finite : bool, optional
288:         Whether to check that the input matrices contain only finite numbers.
289:         Disabling may give a performance gain, but may result in problems
290:         (crashes, non-termination) if the inputs do contain infinities or NaNs.
291: 
292:     Returns
293:     -------
294:     w : (N,) float ndarray
295:         The N (1<=N<=M) selected eigenvalues, in ascending order, each
296:         repeated according to its multiplicity.
297:     v : (M, N) complex ndarray
298:         (if eigvals_only == False)
299: 
300:         The normalized selected eigenvector corresponding to the
301:         eigenvalue w[i] is the column v[:,i].
302: 
303:         Normalization:
304: 
305:             type 1 and 3: v.conj() a      v  = w
306: 
307:             type 2: inv(v).conj() a  inv(v) = w
308: 
309:             type = 1 or 2: v.conj() b      v  = I
310: 
311:             type = 3: v.conj() inv(b) v  = I
312: 
313:     Raises
314:     ------
315:     LinAlgError
316:         If eigenvalue computation does not converge,
317:         an error occurred, or b matrix is not definite positive. Note that
318:         if input matrices are not symmetric or hermitian, no error is reported
319:         but results will be wrong.
320: 
321:     See Also
322:     --------
323:     eigvalsh : eigenvalues of symmetric or Hermitian arrays
324:     eig : eigenvalues and right eigenvectors for non-symmetric arrays
325:     eigh : eigenvalues and right eigenvectors for symmetric/Hermitian arrays
326:     eigh_tridiagonal : eigenvalues and right eiegenvectors for
327:         symmetric/Hermitian tridiagonal matrices
328:     '''
329:     a1 = _asarray_validated(a, check_finite=check_finite)
330:     if len(a1.shape) != 2 or a1.shape[0] != a1.shape[1]:
331:         raise ValueError('expected square matrix')
332:     overwrite_a = overwrite_a or (_datacopied(a1, a))
333:     if iscomplexobj(a1):
334:         cplx = True
335:     else:
336:         cplx = False
337:     if b is not None:
338:         b1 = _asarray_validated(b, check_finite=check_finite)
339:         overwrite_b = overwrite_b or _datacopied(b1, b)
340:         if len(b1.shape) != 2 or b1.shape[0] != b1.shape[1]:
341:             raise ValueError('expected square matrix')
342: 
343:         if b1.shape != a1.shape:
344:             raise ValueError("wrong b dimensions %s, should "
345:                              "be %s" % (str(b1.shape), str(a1.shape)))
346:         if iscomplexobj(b1):
347:             cplx = True
348:         else:
349:             cplx = cplx or False
350:     else:
351:         b1 = None
352: 
353:     # Set job for fortran routines
354:     _job = (eigvals_only and 'N') or 'V'
355: 
356:     # port eigenvalue range from python to fortran convention
357:     if eigvals is not None:
358:         lo, hi = eigvals
359:         if lo < 0 or hi >= a1.shape[0]:
360:             raise ValueError('The eigenvalue range specified is not valid.\n'
361:                              'Valid range is [%s,%s]' % (0, a1.shape[0]-1))
362:         lo += 1
363:         hi += 1
364:         eigvals = (lo, hi)
365: 
366:     # set lower
367:     if lower:
368:         uplo = 'L'
369:     else:
370:         uplo = 'U'
371: 
372:     # fix prefix for lapack routines
373:     if cplx:
374:         pfx = 'he'
375:     else:
376:         pfx = 'sy'
377: 
378:     #  Standard Eigenvalue Problem
379:     #  Use '*evr' routines
380:     # FIXME: implement calculation of optimal lwork
381:     #        for all lapack routines
382:     if b1 is None:
383:         driver = pfx+'evr'
384:         (evr,) = get_lapack_funcs((driver,), (a1,))
385:         if eigvals is None:
386:             w, v, info = evr(a1, uplo=uplo, jobz=_job, range="A", il=1,
387:                              iu=a1.shape[0], overwrite_a=overwrite_a)
388:         else:
389:             (lo, hi) = eigvals
390:             w_tot, v, info = evr(a1, uplo=uplo, jobz=_job, range="I",
391:                                  il=lo, iu=hi, overwrite_a=overwrite_a)
392:             w = w_tot[0:hi-lo+1]
393: 
394:     # Generalized Eigenvalue Problem
395:     else:
396:         # Use '*gvx' routines if range is specified
397:         if eigvals is not None:
398:             driver = pfx+'gvx'
399:             (gvx,) = get_lapack_funcs((driver,), (a1, b1))
400:             (lo, hi) = eigvals
401:             w_tot, v, ifail, info = gvx(a1, b1, uplo=uplo, iu=hi,
402:                                         itype=type, jobz=_job, il=lo,
403:                                         overwrite_a=overwrite_a,
404:                                         overwrite_b=overwrite_b)
405:             w = w_tot[0:hi-lo+1]
406:         # Use '*gvd' routine if turbo is on and no eigvals are specified
407:         elif turbo:
408:             driver = pfx+'gvd'
409:             (gvd,) = get_lapack_funcs((driver,), (a1, b1))
410:             v, w, info = gvd(a1, b1, uplo=uplo, itype=type, jobz=_job,
411:                              overwrite_a=overwrite_a,
412:                              overwrite_b=overwrite_b)
413:         # Use '*gv' routine if turbo is off and no eigvals are specified
414:         else:
415:             driver = pfx+'gv'
416:             (gv,) = get_lapack_funcs((driver,), (a1, b1))
417:             v, w, info = gv(a1, b1, uplo=uplo, itype=type, jobz=_job,
418:                             overwrite_a=overwrite_a,
419:                             overwrite_b=overwrite_b)
420: 
421:     # Check if we had a  successful exit
422:     if info == 0:
423:         if eigvals_only:
424:             return w
425:         else:
426:             return w, v
427:     _check_info(info, driver, positive=False)  # triage more specifically
428:     if info > 0 and b1 is None:
429:         raise LinAlgError("unrecoverable internal error.")
430: 
431:     # The algorithm failed to converge.
432:     elif 0 < info <= b1.shape[0]:
433:         if eigvals is not None:
434:             raise LinAlgError("the eigenvectors %s failed to"
435:                               " converge." % nonzero(ifail)-1)
436:         else:
437:             raise LinAlgError("internal fortran routine failed to converge: "
438:                               "%i off-diagonal elements of an "
439:                               "intermediate tridiagonal form did not converge"
440:                               " to zero." % info)
441: 
442:     # This occurs when b is not positive definite
443:     else:
444:         raise LinAlgError("the leading minor of order %i"
445:                           " of 'b' is not positive definite. The"
446:                           " factorization of 'b' could not be completed"
447:                           " and no eigenvalues or eigenvectors were"
448:                           " computed." % (info-b1.shape[0]))
449: 
450: 
451: _conv_dict = {0: 0, 1: 1, 2: 2,
452:               'all': 0, 'value': 1, 'index': 2,
453:               'a': 0, 'v': 1, 'i': 2}
454: 
455: 
456: def _check_select(select, select_range, max_ev, max_len):
457:     '''Check that select is valid, convert to Fortran style.'''
458:     if isinstance(select, string_types):
459:         select = select.lower()
460:     try:
461:         select = _conv_dict[select]
462:     except KeyError:
463:         raise ValueError('invalid argument for select')
464:     vl, vu = 0., 1.
465:     il = iu = 1
466:     if select != 0:  # (non-all)
467:         sr = asarray(select_range)
468:         if sr.ndim != 1 or sr.size != 2 or sr[1] < sr[0]:
469:             raise ValueError('select_range must be a 2-element array-like '
470:                              'in nondecreasing order')
471:         if select == 1:  # (value)
472:             vl, vu = sr
473:             if max_ev == 0:
474:                 max_ev = max_len
475:         else:  # 2 (index)
476:             if sr.dtype.char.lower() not in 'lih':
477:                 raise ValueError('when using select="i", select_range must '
478:                                  'contain integers, got dtype %s' % sr.dtype)
479:             # translate Python (0 ... N-1) into Fortran (1 ... N) with + 1
480:             il, iu = sr + 1
481:             if min(il, iu) < 1 or max(il, iu) > max_len:
482:                 raise ValueError('select_range out of bounds')
483:             max_ev = iu - il + 1
484:     return select, vl, vu, il, iu, max_ev
485: 
486: 
487: def eig_banded(a_band, lower=False, eigvals_only=False, overwrite_a_band=False,
488:                select='a', select_range=None, max_ev=0, check_finite=True):
489:     '''
490:     Solve real symmetric or complex hermitian band matrix eigenvalue problem.
491: 
492:     Find eigenvalues w and optionally right eigenvectors v of a::
493: 
494:         a v[:,i] = w[i] v[:,i]
495:         v.H v    = identity
496: 
497:     The matrix a is stored in a_band either in lower diagonal or upper
498:     diagonal ordered form:
499: 
500:         a_band[u + i - j, j] == a[i,j]        (if upper form; i <= j)
501:         a_band[    i - j, j] == a[i,j]        (if lower form; i >= j)
502: 
503:     where u is the number of bands above the diagonal.
504: 
505:     Example of a_band (shape of a is (6,6), u=2)::
506: 
507:         upper form:
508:         *   *   a02 a13 a24 a35
509:         *   a01 a12 a23 a34 a45
510:         a00 a11 a22 a33 a44 a55
511: 
512:         lower form:
513:         a00 a11 a22 a33 a44 a55
514:         a10 a21 a32 a43 a54 *
515:         a20 a31 a42 a53 *   *
516: 
517:     Cells marked with * are not used.
518: 
519:     Parameters
520:     ----------
521:     a_band : (u+1, M) array_like
522:         The bands of the M by M matrix a.
523:     lower : bool, optional
524:         Is the matrix in the lower form. (Default is upper form)
525:     eigvals_only : bool, optional
526:         Compute only the eigenvalues and no eigenvectors.
527:         (Default: calculate also eigenvectors)
528:     overwrite_a_band : bool, optional
529:         Discard data in a_band (may enhance performance)
530:     select : {'a', 'v', 'i'}, optional
531:         Which eigenvalues to calculate
532: 
533:         ======  ========================================
534:         select  calculated
535:         ======  ========================================
536:         'a'     All eigenvalues
537:         'v'     Eigenvalues in the interval (min, max]
538:         'i'     Eigenvalues with indices min <= i <= max
539:         ======  ========================================
540:     select_range : (min, max), optional
541:         Range of selected eigenvalues
542:     max_ev : int, optional
543:         For select=='v', maximum number of eigenvalues expected.
544:         For other values of select, has no meaning.
545: 
546:         In doubt, leave this parameter untouched.
547: 
548:     check_finite : bool, optional
549:         Whether to check that the input matrix contains only finite numbers.
550:         Disabling may give a performance gain, but may result in problems
551:         (crashes, non-termination) if the inputs do contain infinities or NaNs.
552: 
553:     Returns
554:     -------
555:     w : (M,) ndarray
556:         The eigenvalues, in ascending order, each repeated according to its
557:         multiplicity.
558:     v : (M, M) float or complex ndarray
559:         The normalized eigenvector corresponding to the eigenvalue w[i] is
560:         the column v[:,i].
561: 
562:     Raises
563:     ------
564:     LinAlgError
565:         If eigenvalue computation does not converge.
566: 
567:     See Also
568:     --------
569:     eigvals_banded : eigenvalues for symmetric/Hermitian band matrices
570:     eig : eigenvalues and right eigenvectors of general arrays.
571:     eigh : eigenvalues and right eigenvectors for symmetric/Hermitian arrays
572:     eigh_tridiagonal : eigenvalues and right eiegenvectors for
573:         symmetric/Hermitian tridiagonal matrices
574:     '''
575:     if eigvals_only or overwrite_a_band:
576:         a1 = _asarray_validated(a_band, check_finite=check_finite)
577:         overwrite_a_band = overwrite_a_band or (_datacopied(a1, a_band))
578:     else:
579:         a1 = array(a_band)
580:         if issubclass(a1.dtype.type, inexact) and not isfinite(a1).all():
581:             raise ValueError("array must not contain infs or NaNs")
582:         overwrite_a_band = 1
583: 
584:     if len(a1.shape) != 2:
585:         raise ValueError('expected two-dimensional array')
586:     select, vl, vu, il, iu, max_ev = _check_select(
587:         select, select_range, max_ev, a1.shape[1])
588:     del select_range
589:     if select == 0:
590:         if a1.dtype.char in 'GFD':
591:             # FIXME: implement this somewhen, for now go with builtin values
592:             # FIXME: calc optimal lwork by calling ?hbevd(lwork=-1)
593:             #        or by using calc_lwork.f ???
594:             # lwork = calc_lwork.hbevd(bevd.typecode, a1.shape[0], lower)
595:             internal_name = 'hbevd'
596:         else:  # a1.dtype.char in 'fd':
597:             # FIXME: implement this somewhen, for now go with builtin values
598:             #         see above
599:             # lwork = calc_lwork.sbevd(bevd.typecode, a1.shape[0], lower)
600:             internal_name = 'sbevd'
601:         bevd, = get_lapack_funcs((internal_name,), (a1,))
602:         w, v, info = bevd(a1, compute_v=not eigvals_only,
603:                           lower=lower, overwrite_ab=overwrite_a_band)
604:     else:  # select in [1, 2]
605:         if eigvals_only:
606:             max_ev = 1
607:         # calculate optimal abstol for dsbevx (see manpage)
608:         if a1.dtype.char in 'fF':  # single precision
609:             lamch, = get_lapack_funcs(('lamch',), (array(0, dtype='f'),))
610:         else:
611:             lamch, = get_lapack_funcs(('lamch',), (array(0, dtype='d'),))
612:         abstol = 2 * lamch('s')
613:         if a1.dtype.char in 'GFD':
614:             internal_name = 'hbevx'
615:         else:  # a1.dtype.char in 'gfd'
616:             internal_name = 'sbevx'
617:         bevx, = get_lapack_funcs((internal_name,), (a1,))
618:         w, v, m, ifail, info = bevx(
619:             a1, vl, vu, il, iu, compute_v=not eigvals_only, mmax=max_ev,
620:             range=select, lower=lower, overwrite_ab=overwrite_a_band,
621:             abstol=abstol)
622:         # crop off w and v
623:         w = w[:m]
624:         if not eigvals_only:
625:             v = v[:, :m]
626:     _check_info(info, internal_name)
627: 
628:     if eigvals_only:
629:         return w
630:     return w, v
631: 
632: 
633: def eigvals(a, b=None, overwrite_a=False, check_finite=True,
634:             homogeneous_eigvals=False):
635:     '''
636:     Compute eigenvalues from an ordinary or generalized eigenvalue problem.
637: 
638:     Find eigenvalues of a general matrix::
639: 
640:         a   vr[:,i] = w[i]        b   vr[:,i]
641: 
642:     Parameters
643:     ----------
644:     a : (M, M) array_like
645:         A complex or real matrix whose eigenvalues and eigenvectors
646:         will be computed.
647:     b : (M, M) array_like, optional
648:         Right-hand side matrix in a generalized eigenvalue problem.
649:         If omitted, identity matrix is assumed.
650:     overwrite_a : bool, optional
651:         Whether to overwrite data in a (may improve performance)
652:     check_finite : bool, optional
653:         Whether to check that the input matrices contain only finite numbers.
654:         Disabling may give a performance gain, but may result in problems
655:         (crashes, non-termination) if the inputs do contain infinities
656:         or NaNs.
657:     homogeneous_eigvals : bool, optional
658:         If True, return the eigenvalues in homogeneous coordinates.
659:         In this case ``w`` is a (2, M) array so that::
660: 
661:             w[1,i] a vr[:,i] = w[0,i] b vr[:,i]
662: 
663:         Default is False.
664: 
665:     Returns
666:     -------
667:     w : (M,) or (2, M) double or complex ndarray
668:         The eigenvalues, each repeated according to its multiplicity
669:         but not in any specific order. The shape is (M,) unless
670:         ``homogeneous_eigvals=True``.
671: 
672:     Raises
673:     ------
674:     LinAlgError
675:         If eigenvalue computation does not converge
676: 
677:     See Also
678:     --------
679:     eig : eigenvalues and right eigenvectors of general arrays.
680:     eigvalsh : eigenvalues of symmetric or Hermitian arrays
681:     eigvals_banded : eigenvalues for symmetric/Hermitian band matrices
682:     eigvalsh_tridiagonal : eigenvalues of symmetric/Hermitian tridiagonal
683:         matrices
684:     '''
685:     return eig(a, b=b, left=0, right=0, overwrite_a=overwrite_a,
686:                check_finite=check_finite,
687:                homogeneous_eigvals=homogeneous_eigvals)
688: 
689: 
690: def eigvalsh(a, b=None, lower=True, overwrite_a=False,
691:              overwrite_b=False, turbo=True, eigvals=None, type=1,
692:              check_finite=True):
693:     '''
694:     Solve an ordinary or generalized eigenvalue problem for a complex
695:     Hermitian or real symmetric matrix.
696: 
697:     Find eigenvalues w of matrix a, where b is positive definite::
698: 
699:                       a v[:,i] = w[i] b v[:,i]
700:         v[i,:].conj() a v[:,i] = w[i]
701:         v[i,:].conj() b v[:,i] = 1
702: 
703: 
704:     Parameters
705:     ----------
706:     a : (M, M) array_like
707:         A complex Hermitian or real symmetric matrix whose eigenvalues and
708:         eigenvectors will be computed.
709:     b : (M, M) array_like, optional
710:         A complex Hermitian or real symmetric definite positive matrix in.
711:         If omitted, identity matrix is assumed.
712:     lower : bool, optional
713:         Whether the pertinent array data is taken from the lower or upper
714:         triangle of `a`. (Default: lower)
715:     turbo : bool, optional
716:         Use divide and conquer algorithm (faster but expensive in memory,
717:         only for generalized eigenvalue problem and if eigvals=None)
718:     eigvals : tuple (lo, hi), optional
719:         Indexes of the smallest and largest (in ascending order) eigenvalues
720:         and corresponding eigenvectors to be returned: 0 <= lo < hi <= M-1.
721:         If omitted, all eigenvalues and eigenvectors are returned.
722:     type : int, optional
723:         Specifies the problem type to be solved:
724: 
725:            type = 1: a   v[:,i] = w[i] b v[:,i]
726: 
727:            type = 2: a b v[:,i] = w[i]   v[:,i]
728: 
729:            type = 3: b a v[:,i] = w[i]   v[:,i]
730:     overwrite_a : bool, optional
731:         Whether to overwrite data in `a` (may improve performance)
732:     overwrite_b : bool, optional
733:         Whether to overwrite data in `b` (may improve performance)
734:     check_finite : bool, optional
735:         Whether to check that the input matrices contain only finite numbers.
736:         Disabling may give a performance gain, but may result in problems
737:         (crashes, non-termination) if the inputs do contain infinities or NaNs.
738: 
739:     Returns
740:     -------
741:     w : (N,) float ndarray
742:         The N (1<=N<=M) selected eigenvalues, in ascending order, each
743:         repeated according to its multiplicity.
744: 
745:     Raises
746:     ------
747:     LinAlgError
748:         If eigenvalue computation does not converge,
749:         an error occurred, or b matrix is not definite positive. Note that
750:         if input matrices are not symmetric or hermitian, no error is reported
751:         but results will be wrong.
752: 
753:     See Also
754:     --------
755:     eigh : eigenvalues and right eigenvectors for symmetric/Hermitian arrays
756:     eigvals : eigenvalues of general arrays
757:     eigvals_banded : eigenvalues for symmetric/Hermitian band matrices
758:     eigvalsh_tridiagonal : eigenvalues of symmetric/Hermitian tridiagonal
759:         matrices
760:     '''
761:     return eigh(a, b=b, lower=lower, eigvals_only=True,
762:                 overwrite_a=overwrite_a, overwrite_b=overwrite_b,
763:                 turbo=turbo, eigvals=eigvals, type=type,
764:                 check_finite=check_finite)
765: 
766: 
767: def eigvals_banded(a_band, lower=False, overwrite_a_band=False,
768:                    select='a', select_range=None, check_finite=True):
769:     '''
770:     Solve real symmetric or complex hermitian band matrix eigenvalue problem.
771: 
772:     Find eigenvalues w of a::
773: 
774:         a v[:,i] = w[i] v[:,i]
775:         v.H v    = identity
776: 
777:     The matrix a is stored in a_band either in lower diagonal or upper
778:     diagonal ordered form:
779: 
780:         a_band[u + i - j, j] == a[i,j]        (if upper form; i <= j)
781:         a_band[    i - j, j] == a[i,j]        (if lower form; i >= j)
782: 
783:     where u is the number of bands above the diagonal.
784: 
785:     Example of a_band (shape of a is (6,6), u=2)::
786: 
787:         upper form:
788:         *   *   a02 a13 a24 a35
789:         *   a01 a12 a23 a34 a45
790:         a00 a11 a22 a33 a44 a55
791: 
792:         lower form:
793:         a00 a11 a22 a33 a44 a55
794:         a10 a21 a32 a43 a54 *
795:         a20 a31 a42 a53 *   *
796: 
797:     Cells marked with * are not used.
798: 
799:     Parameters
800:     ----------
801:     a_band : (u+1, M) array_like
802:         The bands of the M by M matrix a.
803:     lower : bool, optional
804:         Is the matrix in the lower form. (Default is upper form)
805:     overwrite_a_band : bool, optional
806:         Discard data in a_band (may enhance performance)
807:     select : {'a', 'v', 'i'}, optional
808:         Which eigenvalues to calculate
809: 
810:         ======  ========================================
811:         select  calculated
812:         ======  ========================================
813:         'a'     All eigenvalues
814:         'v'     Eigenvalues in the interval (min, max]
815:         'i'     Eigenvalues with indices min <= i <= max
816:         ======  ========================================
817:     select_range : (min, max), optional
818:         Range of selected eigenvalues
819:     check_finite : bool, optional
820:         Whether to check that the input matrix contains only finite numbers.
821:         Disabling may give a performance gain, but may result in problems
822:         (crashes, non-termination) if the inputs do contain infinities or NaNs.
823: 
824:     Returns
825:     -------
826:     w : (M,) ndarray
827:         The eigenvalues, in ascending order, each repeated according to its
828:         multiplicity.
829: 
830:     Raises
831:     ------
832:     LinAlgError
833:         If eigenvalue computation does not converge.
834: 
835:     See Also
836:     --------
837:     eig_banded : eigenvalues and right eigenvectors for symmetric/Hermitian
838:         band matrices
839:     eigvalsh_tridiagonal : eigenvalues of symmetric/Hermitian tridiagonal
840:         matrices
841:     eigvals : eigenvalues of general arrays
842:     eigh : eigenvalues and right eigenvectors for symmetric/Hermitian arrays
843:     eig : eigenvalues and right eigenvectors for non-symmetric arrays
844:     '''
845:     return eig_banded(a_band, lower=lower, eigvals_only=1,
846:                       overwrite_a_band=overwrite_a_band, select=select,
847:                       select_range=select_range, check_finite=check_finite)
848: 
849: 
850: def eigvalsh_tridiagonal(d, e, select='a', select_range=None,
851:                          check_finite=True, tol=0., lapack_driver='auto'):
852:     '''
853:     Solve eigenvalue problem for a real symmetric tridiagonal matrix.
854: 
855:     Find eigenvalues `w` of ``a``::
856: 
857:         a v[:,i] = w[i] v[:,i]
858:         v.H v    = identity
859: 
860:     For a real symmetric matrix ``a`` with diagonal elements `d` and
861:     off-diagonal elements `e`.
862: 
863:     Parameters
864:     ----------
865:     d : ndarray, shape (ndim,)
866:         The diagonal elements of the array.
867:     e : ndarray, shape (ndim-1,)
868:         The off-diagonal elements of the array.
869:     select : {'a', 'v', 'i'}, optional
870:         Which eigenvalues to calculate
871: 
872:         ======  ========================================
873:         select  calculated
874:         ======  ========================================
875:         'a'     All eigenvalues
876:         'v'     Eigenvalues in the interval (min, max]
877:         'i'     Eigenvalues with indices min <= i <= max
878:         ======  ========================================
879:     select_range : (min, max), optional
880:         Range of selected eigenvalues
881:     check_finite : bool, optional
882:         Whether to check that the input matrix contains only finite numbers.
883:         Disabling may give a performance gain, but may result in problems
884:         (crashes, non-termination) if the inputs do contain infinities or NaNs.
885:     tol : float
886:         The absolute tolerance to which each eigenvalue is required
887:         (only used when ``lapack_driver='stebz'``).
888:         An eigenvalue (or cluster) is considered to have converged if it
889:         lies in an interval of this width. If <= 0. (default),
890:         the value ``eps*|a|`` is used where eps is the machine precision,
891:         and ``|a|`` is the 1-norm of the matrix ``a``.
892:     lapack_driver : str
893:         LAPACK function to use, can be 'auto', 'stemr', 'stebz',  'sterf',
894:         or 'stev'. When 'auto' (default), it will use 'stemr' if ``select='a'``
895:         and 'stebz' otherwise. 'sterf' and 'stev' can only be used when
896:         ``select='a'``.
897: 
898:     Returns
899:     -------
900:     w : (M,) ndarray
901:         The eigenvalues, in ascending order, each repeated according to its
902:         multiplicity.
903: 
904:     Raises
905:     ------
906:     LinAlgError
907:         If eigenvalue computation does not converge.
908: 
909:     See Also
910:     --------
911:     eigh_tridiagonal : eigenvalues and right eiegenvectors for
912:         symmetric/Hermitian tridiagonal matrices
913:     '''
914:     return eigh_tridiagonal(
915:         d, e, eigvals_only=True, select=select, select_range=select_range,
916:         check_finite=check_finite, tol=tol, lapack_driver=lapack_driver)
917: 
918: 
919: def eigh_tridiagonal(d, e, eigvals_only=False, select='a', select_range=None,
920:                      check_finite=True, tol=0., lapack_driver='auto'):
921:     '''
922:     Solve eigenvalue problem for a real symmetric tridiagonal matrix.
923: 
924:     Find eigenvalues `w` and optionally right eigenvectors `v` of ``a``::
925: 
926:         a v[:,i] = w[i] v[:,i]
927:         v.H v    = identity
928: 
929:     For a real symmetric matrix ``a`` with diagonal elements `d` and
930:     off-diagonal elements `e`.
931: 
932:     Parameters
933:     ----------
934:     d : ndarray, shape (ndim,)
935:         The diagonal elements of the array.
936:     e : ndarray, shape (ndim-1,)
937:         The off-diagonal elements of the array.
938:     select : {'a', 'v', 'i'}, optional
939:         Which eigenvalues to calculate
940: 
941:         ======  ========================================
942:         select  calculated
943:         ======  ========================================
944:         'a'     All eigenvalues
945:         'v'     Eigenvalues in the interval (min, max]
946:         'i'     Eigenvalues with indices min <= i <= max
947:         ======  ========================================
948:     select_range : (min, max), optional
949:         Range of selected eigenvalues
950:     check_finite : bool, optional
951:         Whether to check that the input matrix contains only finite numbers.
952:         Disabling may give a performance gain, but may result in problems
953:         (crashes, non-termination) if the inputs do contain infinities or NaNs.
954:     tol : float
955:         The absolute tolerance to which each eigenvalue is required
956:         (only used when 'stebz' is the `lapack_driver`).
957:         An eigenvalue (or cluster) is considered to have converged if it
958:         lies in an interval of this width. If <= 0. (default),
959:         the value ``eps*|a|`` is used where eps is the machine precision,
960:         and ``|a|`` is the 1-norm of the matrix ``a``.
961:     lapack_driver : str
962:         LAPACK function to use, can be 'auto', 'stemr', 'stebz', 'sterf',
963:         or 'stev'. When 'auto' (default), it will use 'stemr' if ``select='a'``
964:         and 'stebz' otherwise. When 'stebz' is used to find the eigenvalues and
965:         ``eigvals_only=False``, then a second LAPACK call (to ``?STEIN``) is
966:         used to find the corresponding eigenvectors. 'sterf' can only be
967:         used when ``eigvals_only=True`` and ``select='a'``. 'stev' can only
968:         be used when ``select='a'``.
969: 
970:     Returns
971:     -------
972:     w : (M,) ndarray
973:         The eigenvalues, in ascending order, each repeated according to its
974:         multiplicity.
975:     v : (M, M) ndarray
976:         The normalized eigenvector corresponding to the eigenvalue ``w[i]`` is
977:         the column ``v[:,i]``.
978: 
979:     Raises
980:     ------
981:     LinAlgError
982:         If eigenvalue computation does not converge.
983: 
984:     See Also
985:     --------
986:     eigvalsh_tridiagonal : eigenvalues of symmetric/Hermitian tridiagonal
987:         matrices
988:     eig : eigenvalues and right eigenvectors for non-symmetric arrays
989:     eigh : eigenvalues and right eigenvectors for symmetric/Hermitian arrays
990:     eig_banded : eigenvalues and right eigenvectors for symmetric/Hermitian
991:         band matrices
992: 
993:     Notes
994:     -----
995:     This function makes use of LAPACK ``S/DSTEMR`` routines.
996:     '''
997:     d = _asarray_validated(d, check_finite=check_finite)
998:     e = _asarray_validated(e, check_finite=check_finite)
999:     for check in (d, e):
1000:         if check.ndim != 1:
1001:             raise ValueError('expected one-dimensional array')
1002:         if check.dtype.char in 'GFD':  # complex
1003:             raise TypeError('Only real arrays currently supported')
1004:     if d.size != e.size + 1:
1005:         raise ValueError('d (%s) must have one more element than e (%s)'
1006:                          % (d.size, e.size))
1007:     select, vl, vu, il, iu, _ = _check_select(
1008:         select, select_range, 0, d.size)
1009:     if not isinstance(lapack_driver, string_types):
1010:         raise TypeError('lapack_driver must be str')
1011:     drivers = ('auto', 'stemr', 'sterf', 'stebz', 'stev')
1012:     if lapack_driver not in drivers:
1013:         raise ValueError('lapack_driver must be one of %s, got %s'
1014:                          % (drivers, lapack_driver))
1015:     if lapack_driver == 'auto':
1016:         lapack_driver = 'stemr' if select == 0 else 'stebz'
1017:     func, = get_lapack_funcs((lapack_driver,), (d, e))
1018:     compute_v = not eigvals_only
1019:     if lapack_driver == 'sterf':
1020:         if select != 0:
1021:             raise ValueError('sterf can only be used when select == "a"')
1022:         if not eigvals_only:
1023:             raise ValueError('sterf can only be used when eigvals_only is '
1024:                              'True')
1025:         w, info = func(d, e)
1026:         m = len(w)
1027:     elif lapack_driver == 'stev':
1028:         if select != 0:
1029:             raise ValueError('stev can only be used when select == "a"')
1030:         w, v, info = func(d, e, compute_v=compute_v)
1031:         m = len(w)
1032:     elif lapack_driver == 'stebz':
1033:         tol = float(tol)
1034:         internal_name = 'stebz'
1035:         stebz, = get_lapack_funcs((internal_name,), (d, e))
1036:         # If getting eigenvectors, needs to be block-ordered (B) instead of
1037:         # matirx-ordered (E), and we will reorder later
1038:         order = 'E' if eigvals_only else 'B'
1039:         m, w, iblock, isplit, info = stebz(d, e, select, vl, vu, il, iu, tol,
1040:                                            order)
1041:     else:   # 'stemr'
1042:         # ?STEMR annoyingly requires size N instead of N-1
1043:         e_ = empty(e.size+1, e.dtype)
1044:         e_[:-1] = e
1045:         stemr_lwork, = get_lapack_funcs(('stemr_lwork',), (d, e))
1046:         lwork, liwork, info = stemr_lwork(d, e_, select, vl, vu, il, iu,
1047:                                           compute_v=compute_v)
1048:         _check_info(info, 'stemr_lwork')
1049:         m, w, v, info = func(d, e_, select, vl, vu, il, iu,
1050:                              compute_v=compute_v, lwork=lwork, liwork=liwork)
1051:     _check_info(info, lapack_driver + ' (eigh_tridiagonal)')
1052:     w = w[:m]
1053:     if eigvals_only:
1054:         return w
1055:     else:
1056:         # Do we still need to compute the eigenvalues?
1057:         if lapack_driver == 'stebz':
1058:             func, = get_lapack_funcs(('stein',), (d, e))
1059:             v, info = func(d, e, w, iblock, isplit)
1060:             _check_info(info, 'stein (eigh_tridiagonal)',
1061:                         positive='%d eigenvectors failed to converge')
1062:             # Convert block-order to matrix-order
1063:             order = argsort(w)
1064:             w, v = w[order], v[:, order]
1065:         else:
1066:             v = v[:, :m]
1067:         return w, v
1068: 
1069: 
1070: def _check_info(info, driver, positive='did not converge (LAPACK info=%d)'):
1071:     '''Check info return value.'''
1072:     if info < 0:
1073:         raise ValueError('illegal value in argument %d of internal %s'
1074:                          % (-info, driver))
1075:     if info > 0 and positive:
1076:         raise LinAlgError(("%s " + positive) % (driver, info,))
1077: 
1078: 
1079: def hessenberg(a, calc_q=False, overwrite_a=False, check_finite=True):
1080:     '''
1081:     Compute Hessenberg form of a matrix.
1082: 
1083:     The Hessenberg decomposition is::
1084: 
1085:         A = Q H Q^H
1086: 
1087:     where `Q` is unitary/orthogonal and `H` has only zero elements below
1088:     the first sub-diagonal.
1089: 
1090:     Parameters
1091:     ----------
1092:     a : (M, M) array_like
1093:         Matrix to bring into Hessenberg form.
1094:     calc_q : bool, optional
1095:         Whether to compute the transformation matrix.  Default is False.
1096:     overwrite_a : bool, optional
1097:         Whether to overwrite `a`; may improve performance.
1098:         Default is False.
1099:     check_finite : bool, optional
1100:         Whether to check that the input matrix contains only finite numbers.
1101:         Disabling may give a performance gain, but may result in problems
1102:         (crashes, non-termination) if the inputs do contain infinities or NaNs.
1103: 
1104:     Returns
1105:     -------
1106:     H : (M, M) ndarray
1107:         Hessenberg form of `a`.
1108:     Q : (M, M) ndarray
1109:         Unitary/orthogonal similarity transformation matrix ``A = Q H Q^H``.
1110:         Only returned if ``calc_q=True``.
1111: 
1112:     '''
1113:     a1 = _asarray_validated(a, check_finite=check_finite)
1114:     if len(a1.shape) != 2 or (a1.shape[0] != a1.shape[1]):
1115:         raise ValueError('expected square matrix')
1116:     overwrite_a = overwrite_a or (_datacopied(a1, a))
1117: 
1118:     # if 2x2 or smaller: already in Hessenberg
1119:     if a1.shape[0] <= 2:
1120:         if calc_q:
1121:             return a1, numpy.eye(a1.shape[0])
1122:         return a1
1123: 
1124:     gehrd, gebal, gehrd_lwork = get_lapack_funcs(('gehrd', 'gebal',
1125:                                                   'gehrd_lwork'), (a1,))
1126:     ba, lo, hi, pivscale, info = gebal(a1, permute=0, overwrite_a=overwrite_a)
1127:     _check_info(info, 'gebal (hessenberg)', positive=False)
1128:     n = len(a1)
1129: 
1130:     lwork = _compute_lwork(gehrd_lwork, ba.shape[0], lo=lo, hi=hi)
1131: 
1132:     hq, tau, info = gehrd(ba, lo=lo, hi=hi, lwork=lwork, overwrite_a=1)
1133:     _check_info(info, 'gehrd (hessenberg)', positive=False)
1134:     h = numpy.triu(hq, -1)
1135:     if not calc_q:
1136:         return h
1137: 
1138:     # use orghr/unghr to compute q
1139:     orghr, orghr_lwork = get_lapack_funcs(('orghr', 'orghr_lwork'), (a1,))
1140:     lwork = _compute_lwork(orghr_lwork, n, lo=lo, hi=hi)
1141: 
1142:     q, info = orghr(a=hq, tau=tau, lo=lo, hi=hi, lwork=lwork, overwrite_a=1)
1143:     _check_info(info, 'orghr (hessenberg)', positive=False)
1144:     return h, q
1145: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################


# Assigning a List to a Name (line 17):

# Assigning a List to a Name (line 17):
__all__ = ['eig', 'eigvals', 'eigh', 'eigvalsh', 'eig_banded', 'eigvals_banded', 'eigh_tridiagonal', 'eigvalsh_tridiagonal', 'hessenberg']
module_type_store.set_exportable_members(['eig', 'eigvals', 'eigh', 'eigvalsh', 'eig_banded', 'eigvals_banded', 'eigh_tridiagonal', 'eigvalsh_tridiagonal', 'hessenberg'])

# Obtaining an instance of the builtin type 'list' (line 17)
list_14149 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 10), 'list')
# Adding type elements to the builtin type 'list' instance (line 17)
# Adding element type (line 17)
str_14150 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 11), 'str', 'eig')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 17, 10), list_14149, str_14150)
# Adding element type (line 17)
str_14151 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 18), 'str', 'eigvals')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 17, 10), list_14149, str_14151)
# Adding element type (line 17)
str_14152 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 29), 'str', 'eigh')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 17, 10), list_14149, str_14152)
# Adding element type (line 17)
str_14153 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 37), 'str', 'eigvalsh')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 17, 10), list_14149, str_14153)
# Adding element type (line 17)
str_14154 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 11), 'str', 'eig_banded')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 17, 10), list_14149, str_14154)
# Adding element type (line 17)
str_14155 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 25), 'str', 'eigvals_banded')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 17, 10), list_14149, str_14155)
# Adding element type (line 17)
str_14156 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 11), 'str', 'eigh_tridiagonal')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 17, 10), list_14149, str_14156)
# Adding element type (line 17)
str_14157 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 31), 'str', 'eigvalsh_tridiagonal')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 17, 10), list_14149, str_14157)
# Adding element type (line 17)
str_14158 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 55), 'str', 'hessenberg')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 17, 10), list_14149, str_14158)

# Assigning a type to the variable '__all__' (line 17)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 17, 0), '__all__', list_14149)
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 21, 0))

# 'import numpy' statement (line 21)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/linalg/')
import_14159 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 21, 0), 'numpy')

if (type(import_14159) is not StypyTypeError):

    if (import_14159 != 'pyd_module'):
        __import__(import_14159)
        sys_modules_14160 = sys.modules[import_14159]
        import_module(stypy.reporting.localization.Localization(__file__, 21, 0), 'numpy', sys_modules_14160.module_type_store, module_type_store)
    else:
        import numpy

        import_module(stypy.reporting.localization.Localization(__file__, 21, 0), 'numpy', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 21)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 21, 0), 'numpy', import_14159)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/linalg/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 22, 0))

# 'from numpy import array, isfinite, inexact, nonzero, iscomplexobj, cast, flatnonzero, conj, asarray, argsort, empty' statement (line 22)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/linalg/')
import_14161 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 22, 0), 'numpy')

if (type(import_14161) is not StypyTypeError):

    if (import_14161 != 'pyd_module'):
        __import__(import_14161)
        sys_modules_14162 = sys.modules[import_14161]
        import_from_module(stypy.reporting.localization.Localization(__file__, 22, 0), 'numpy', sys_modules_14162.module_type_store, module_type_store, ['array', 'isfinite', 'inexact', 'nonzero', 'iscomplexobj', 'cast', 'flatnonzero', 'conj', 'asarray', 'argsort', 'empty'])
        nest_module(stypy.reporting.localization.Localization(__file__, 22, 0), __file__, sys_modules_14162, sys_modules_14162.module_type_store, module_type_store)
    else:
        from numpy import array, isfinite, inexact, nonzero, iscomplexobj, cast, flatnonzero, conj, asarray, argsort, empty

        import_from_module(stypy.reporting.localization.Localization(__file__, 22, 0), 'numpy', None, module_type_store, ['array', 'isfinite', 'inexact', 'nonzero', 'iscomplexobj', 'cast', 'flatnonzero', 'conj', 'asarray', 'argsort', 'empty'], [array, isfinite, inexact, nonzero, iscomplexobj, cast, flatnonzero, conj, asarray, argsort, empty])

else:
    # Assigning a type to the variable 'numpy' (line 22)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 22, 0), 'numpy', import_14161)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/linalg/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 25, 0))

# 'from scipy._lib.six import xrange' statement (line 25)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/linalg/')
import_14163 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 25, 0), 'scipy._lib.six')

if (type(import_14163) is not StypyTypeError):

    if (import_14163 != 'pyd_module'):
        __import__(import_14163)
        sys_modules_14164 = sys.modules[import_14163]
        import_from_module(stypy.reporting.localization.Localization(__file__, 25, 0), 'scipy._lib.six', sys_modules_14164.module_type_store, module_type_store, ['xrange'])
        nest_module(stypy.reporting.localization.Localization(__file__, 25, 0), __file__, sys_modules_14164, sys_modules_14164.module_type_store, module_type_store)
    else:
        from scipy._lib.six import xrange

        import_from_module(stypy.reporting.localization.Localization(__file__, 25, 0), 'scipy._lib.six', None, module_type_store, ['xrange'], [xrange])

else:
    # Assigning a type to the variable 'scipy._lib.six' (line 25)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 25, 0), 'scipy._lib.six', import_14163)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/linalg/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 26, 0))

# 'from scipy._lib._util import _asarray_validated' statement (line 26)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/linalg/')
import_14165 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 26, 0), 'scipy._lib._util')

if (type(import_14165) is not StypyTypeError):

    if (import_14165 != 'pyd_module'):
        __import__(import_14165)
        sys_modules_14166 = sys.modules[import_14165]
        import_from_module(stypy.reporting.localization.Localization(__file__, 26, 0), 'scipy._lib._util', sys_modules_14166.module_type_store, module_type_store, ['_asarray_validated'])
        nest_module(stypy.reporting.localization.Localization(__file__, 26, 0), __file__, sys_modules_14166, sys_modules_14166.module_type_store, module_type_store)
    else:
        from scipy._lib._util import _asarray_validated

        import_from_module(stypy.reporting.localization.Localization(__file__, 26, 0), 'scipy._lib._util', None, module_type_store, ['_asarray_validated'], [_asarray_validated])

else:
    # Assigning a type to the variable 'scipy._lib._util' (line 26)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 26, 0), 'scipy._lib._util', import_14165)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/linalg/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 27, 0))

# 'from scipy._lib.six import string_types' statement (line 27)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/linalg/')
import_14167 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 27, 0), 'scipy._lib.six')

if (type(import_14167) is not StypyTypeError):

    if (import_14167 != 'pyd_module'):
        __import__(import_14167)
        sys_modules_14168 = sys.modules[import_14167]
        import_from_module(stypy.reporting.localization.Localization(__file__, 27, 0), 'scipy._lib.six', sys_modules_14168.module_type_store, module_type_store, ['string_types'])
        nest_module(stypy.reporting.localization.Localization(__file__, 27, 0), __file__, sys_modules_14168, sys_modules_14168.module_type_store, module_type_store)
    else:
        from scipy._lib.six import string_types

        import_from_module(stypy.reporting.localization.Localization(__file__, 27, 0), 'scipy._lib.six', None, module_type_store, ['string_types'], [string_types])

else:
    # Assigning a type to the variable 'scipy._lib.six' (line 27)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 27, 0), 'scipy._lib.six', import_14167)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/linalg/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 28, 0))

# 'from scipy.linalg.misc import LinAlgError, _datacopied, norm' statement (line 28)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/linalg/')
import_14169 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 28, 0), 'scipy.linalg.misc')

if (type(import_14169) is not StypyTypeError):

    if (import_14169 != 'pyd_module'):
        __import__(import_14169)
        sys_modules_14170 = sys.modules[import_14169]
        import_from_module(stypy.reporting.localization.Localization(__file__, 28, 0), 'scipy.linalg.misc', sys_modules_14170.module_type_store, module_type_store, ['LinAlgError', '_datacopied', 'norm'])
        nest_module(stypy.reporting.localization.Localization(__file__, 28, 0), __file__, sys_modules_14170, sys_modules_14170.module_type_store, module_type_store)
    else:
        from scipy.linalg.misc import LinAlgError, _datacopied, norm

        import_from_module(stypy.reporting.localization.Localization(__file__, 28, 0), 'scipy.linalg.misc', None, module_type_store, ['LinAlgError', '_datacopied', 'norm'], [LinAlgError, _datacopied, norm])

else:
    # Assigning a type to the variable 'scipy.linalg.misc' (line 28)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 28, 0), 'scipy.linalg.misc', import_14169)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/linalg/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 29, 0))

# 'from scipy.linalg.lapack import get_lapack_funcs, _compute_lwork' statement (line 29)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/linalg/')
import_14171 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 29, 0), 'scipy.linalg.lapack')

if (type(import_14171) is not StypyTypeError):

    if (import_14171 != 'pyd_module'):
        __import__(import_14171)
        sys_modules_14172 = sys.modules[import_14171]
        import_from_module(stypy.reporting.localization.Localization(__file__, 29, 0), 'scipy.linalg.lapack', sys_modules_14172.module_type_store, module_type_store, ['get_lapack_funcs', '_compute_lwork'])
        nest_module(stypy.reporting.localization.Localization(__file__, 29, 0), __file__, sys_modules_14172, sys_modules_14172.module_type_store, module_type_store)
    else:
        from scipy.linalg.lapack import get_lapack_funcs, _compute_lwork

        import_from_module(stypy.reporting.localization.Localization(__file__, 29, 0), 'scipy.linalg.lapack', None, module_type_store, ['get_lapack_funcs', '_compute_lwork'], [get_lapack_funcs, _compute_lwork])

else:
    # Assigning a type to the variable 'scipy.linalg.lapack' (line 29)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 29, 0), 'scipy.linalg.lapack', import_14171)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/linalg/')


# Assigning a Call to a Name (line 32):

# Assigning a Call to a Name (line 32):

# Call to (...): (line 32)
# Processing the call arguments (line 32)
complex_14177 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 32, 15), 'complex')
# Processing the call keyword arguments (line 32)
kwargs_14178 = {}

# Obtaining the type of the subscript
str_14173 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 32, 10), 'str', 'F')
# Getting the type of 'cast' (line 32)
cast_14174 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 5), 'cast', False)
# Obtaining the member '__getitem__' of a type (line 32)
getitem___14175 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 32, 5), cast_14174, '__getitem__')
# Calling the subscript (__getitem__) to obtain the elements type (line 32)
subscript_call_result_14176 = invoke(stypy.reporting.localization.Localization(__file__, 32, 5), getitem___14175, str_14173)

# Calling (args, kwargs) (line 32)
_call_result_14179 = invoke(stypy.reporting.localization.Localization(__file__, 32, 5), subscript_call_result_14176, *[complex_14177], **kwargs_14178)

# Assigning a type to the variable '_I' (line 32)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 32, 0), '_I', _call_result_14179)

@norecursion
def _make_complex_eigvecs(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_make_complex_eigvecs'
    module_type_store = module_type_store.open_function_context('_make_complex_eigvecs', 35, 0, False)
    
    # Passed parameters checking function
    _make_complex_eigvecs.stypy_localization = localization
    _make_complex_eigvecs.stypy_type_of_self = None
    _make_complex_eigvecs.stypy_type_store = module_type_store
    _make_complex_eigvecs.stypy_function_name = '_make_complex_eigvecs'
    _make_complex_eigvecs.stypy_param_names_list = ['w', 'vin', 'dtype']
    _make_complex_eigvecs.stypy_varargs_param_name = None
    _make_complex_eigvecs.stypy_kwargs_param_name = None
    _make_complex_eigvecs.stypy_call_defaults = defaults
    _make_complex_eigvecs.stypy_call_varargs = varargs
    _make_complex_eigvecs.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_make_complex_eigvecs', ['w', 'vin', 'dtype'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_make_complex_eigvecs', localization, ['w', 'vin', 'dtype'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_make_complex_eigvecs(...)' code ##################

    str_14180 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, (-1)), 'str', '\n    Produce complex-valued eigenvectors from LAPACK DGGEV real-valued output\n    ')
    
    # Assigning a Call to a Name (line 40):
    
    # Assigning a Call to a Name (line 40):
    
    # Call to array(...): (line 40)
    # Processing the call arguments (line 40)
    # Getting the type of 'vin' (line 40)
    vin_14183 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 20), 'vin', False)
    # Processing the call keyword arguments (line 40)
    # Getting the type of 'dtype' (line 40)
    dtype_14184 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 31), 'dtype', False)
    keyword_14185 = dtype_14184
    kwargs_14186 = {'dtype': keyword_14185}
    # Getting the type of 'numpy' (line 40)
    numpy_14181 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 8), 'numpy', False)
    # Obtaining the member 'array' of a type (line 40)
    array_14182 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 40, 8), numpy_14181, 'array')
    # Calling array(args, kwargs) (line 40)
    array_call_result_14187 = invoke(stypy.reporting.localization.Localization(__file__, 40, 8), array_14182, *[vin_14183], **kwargs_14186)
    
    # Assigning a type to the variable 'v' (line 40)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 40, 4), 'v', array_call_result_14187)
    
    # Assigning a Compare to a Name (line 41):
    
    # Assigning a Compare to a Name (line 41):
    
    # Getting the type of 'w' (line 41)
    w_14188 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 9), 'w')
    # Obtaining the member 'imag' of a type (line 41)
    imag_14189 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 41, 9), w_14188, 'imag')
    int_14190 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 18), 'int')
    # Applying the binary operator '>' (line 41)
    result_gt_14191 = python_operator(stypy.reporting.localization.Localization(__file__, 41, 9), '>', imag_14189, int_14190)
    
    # Assigning a type to the variable 'm' (line 41)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 41, 4), 'm', result_gt_14191)
    
    # Getting the type of 'm' (line 42)
    m_14192 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 4), 'm')
    
    # Obtaining the type of the subscript
    int_14193 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 7), 'int')
    slice_14194 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 42, 4), None, int_14193, None)
    # Getting the type of 'm' (line 42)
    m_14195 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 4), 'm')
    # Obtaining the member '__getitem__' of a type (line 42)
    getitem___14196 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 42, 4), m_14195, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 42)
    subscript_call_result_14197 = invoke(stypy.reporting.localization.Localization(__file__, 42, 4), getitem___14196, slice_14194)
    
    
    
    # Obtaining the type of the subscript
    int_14198 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 22), 'int')
    slice_14199 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 42, 15), int_14198, None, None)
    # Getting the type of 'w' (line 42)
    w_14200 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 15), 'w')
    # Obtaining the member 'imag' of a type (line 42)
    imag_14201 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 42, 15), w_14200, 'imag')
    # Obtaining the member '__getitem__' of a type (line 42)
    getitem___14202 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 42, 15), imag_14201, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 42)
    subscript_call_result_14203 = invoke(stypy.reporting.localization.Localization(__file__, 42, 15), getitem___14202, slice_14199)
    
    int_14204 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 28), 'int')
    # Applying the binary operator '<' (line 42)
    result_lt_14205 = python_operator(stypy.reporting.localization.Localization(__file__, 42, 15), '<', subscript_call_result_14203, int_14204)
    
    # Applying the binary operator '|=' (line 42)
    result_ior_14206 = python_operator(stypy.reporting.localization.Localization(__file__, 42, 4), '|=', subscript_call_result_14197, result_lt_14205)
    # Getting the type of 'm' (line 42)
    m_14207 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 4), 'm')
    int_14208 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 7), 'int')
    slice_14209 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 42, 4), None, int_14208, None)
    # Storing an element on a container (line 42)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 42, 4), m_14207, (slice_14209, result_ior_14206))
    
    
    
    # Call to flatnonzero(...): (line 43)
    # Processing the call arguments (line 43)
    # Getting the type of 'm' (line 43)
    m_14211 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 25), 'm', False)
    # Processing the call keyword arguments (line 43)
    kwargs_14212 = {}
    # Getting the type of 'flatnonzero' (line 43)
    flatnonzero_14210 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 13), 'flatnonzero', False)
    # Calling flatnonzero(args, kwargs) (line 43)
    flatnonzero_call_result_14213 = invoke(stypy.reporting.localization.Localization(__file__, 43, 13), flatnonzero_14210, *[m_14211], **kwargs_14212)
    
    # Testing the type of a for loop iterable (line 43)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 43, 4), flatnonzero_call_result_14213)
    # Getting the type of the for loop variable (line 43)
    for_loop_var_14214 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 43, 4), flatnonzero_call_result_14213)
    # Assigning a type to the variable 'i' (line 43)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 43, 4), 'i', for_loop_var_14214)
    # SSA begins for a for statement (line 43)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Subscript to a Subscript (line 44):
    
    # Assigning a Subscript to a Subscript (line 44):
    
    # Obtaining the type of the subscript
    slice_14215 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 44, 23), None, None, None)
    # Getting the type of 'i' (line 44)
    i_14216 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 30), 'i')
    int_14217 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 32), 'int')
    # Applying the binary operator '+' (line 44)
    result_add_14218 = python_operator(stypy.reporting.localization.Localization(__file__, 44, 30), '+', i_14216, int_14217)
    
    # Getting the type of 'vin' (line 44)
    vin_14219 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 23), 'vin')
    # Obtaining the member '__getitem__' of a type (line 44)
    getitem___14220 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 44, 23), vin_14219, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 44)
    subscript_call_result_14221 = invoke(stypy.reporting.localization.Localization(__file__, 44, 23), getitem___14220, (slice_14215, result_add_14218))
    
    # Getting the type of 'v' (line 44)
    v_14222 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 8), 'v')
    # Obtaining the member 'imag' of a type (line 44)
    imag_14223 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 44, 8), v_14222, 'imag')
    slice_14224 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 44, 8), None, None, None)
    # Getting the type of 'i' (line 44)
    i_14225 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 18), 'i')
    # Storing an element on a container (line 44)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 44, 8), imag_14223, ((slice_14224, i_14225), subscript_call_result_14221))
    
    # Call to conj(...): (line 45)
    # Processing the call arguments (line 45)
    
    # Obtaining the type of the subscript
    slice_14227 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 45, 13), None, None, None)
    # Getting the type of 'i' (line 45)
    i_14228 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 18), 'i', False)
    # Getting the type of 'v' (line 45)
    v_14229 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 13), 'v', False)
    # Obtaining the member '__getitem__' of a type (line 45)
    getitem___14230 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 45, 13), v_14229, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 45)
    subscript_call_result_14231 = invoke(stypy.reporting.localization.Localization(__file__, 45, 13), getitem___14230, (slice_14227, i_14228))
    
    
    # Obtaining the type of the subscript
    slice_14232 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 45, 22), None, None, None)
    # Getting the type of 'i' (line 45)
    i_14233 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 27), 'i', False)
    int_14234 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 45, 29), 'int')
    # Applying the binary operator '+' (line 45)
    result_add_14235 = python_operator(stypy.reporting.localization.Localization(__file__, 45, 27), '+', i_14233, int_14234)
    
    # Getting the type of 'v' (line 45)
    v_14236 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 22), 'v', False)
    # Obtaining the member '__getitem__' of a type (line 45)
    getitem___14237 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 45, 22), v_14236, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 45)
    subscript_call_result_14238 = invoke(stypy.reporting.localization.Localization(__file__, 45, 22), getitem___14237, (slice_14232, result_add_14235))
    
    # Processing the call keyword arguments (line 45)
    kwargs_14239 = {}
    # Getting the type of 'conj' (line 45)
    conj_14226 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 8), 'conj', False)
    # Calling conj(args, kwargs) (line 45)
    conj_call_result_14240 = invoke(stypy.reporting.localization.Localization(__file__, 45, 8), conj_14226, *[subscript_call_result_14231, subscript_call_result_14238], **kwargs_14239)
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'v' (line 46)
    v_14241 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 11), 'v')
    # Assigning a type to the variable 'stypy_return_type' (line 46)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 46, 4), 'stypy_return_type', v_14241)
    
    # ################# End of '_make_complex_eigvecs(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_make_complex_eigvecs' in the type store
    # Getting the type of 'stypy_return_type' (line 35)
    stypy_return_type_14242 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_14242)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_make_complex_eigvecs'
    return stypy_return_type_14242

# Assigning a type to the variable '_make_complex_eigvecs' (line 35)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 35, 0), '_make_complex_eigvecs', _make_complex_eigvecs)

@norecursion
def _make_eigvals(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_make_eigvals'
    module_type_store = module_type_store.open_function_context('_make_eigvals', 49, 0, False)
    
    # Passed parameters checking function
    _make_eigvals.stypy_localization = localization
    _make_eigvals.stypy_type_of_self = None
    _make_eigvals.stypy_type_store = module_type_store
    _make_eigvals.stypy_function_name = '_make_eigvals'
    _make_eigvals.stypy_param_names_list = ['alpha', 'beta', 'homogeneous_eigvals']
    _make_eigvals.stypy_varargs_param_name = None
    _make_eigvals.stypy_kwargs_param_name = None
    _make_eigvals.stypy_call_defaults = defaults
    _make_eigvals.stypy_call_varargs = varargs
    _make_eigvals.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_make_eigvals', ['alpha', 'beta', 'homogeneous_eigvals'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_make_eigvals', localization, ['alpha', 'beta', 'homogeneous_eigvals'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_make_eigvals(...)' code ##################

    
    # Getting the type of 'homogeneous_eigvals' (line 50)
    homogeneous_eigvals_14243 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 7), 'homogeneous_eigvals')
    # Testing the type of an if condition (line 50)
    if_condition_14244 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 50, 4), homogeneous_eigvals_14243)
    # Assigning a type to the variable 'if_condition_14244' (line 50)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 50, 4), 'if_condition_14244', if_condition_14244)
    # SSA begins for if statement (line 50)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Type idiom detected: calculating its left and rigth part (line 51)
    # Getting the type of 'beta' (line 51)
    beta_14245 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 11), 'beta')
    # Getting the type of 'None' (line 51)
    None_14246 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 19), 'None')
    
    (may_be_14247, more_types_in_union_14248) = may_be_none(beta_14245, None_14246)

    if may_be_14247:

        if more_types_in_union_14248:
            # Runtime conditional SSA (line 51)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Call to vstack(...): (line 52)
        # Processing the call arguments (line 52)
        
        # Obtaining an instance of the builtin type 'tuple' (line 52)
        tuple_14251 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 52, 33), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 52)
        # Adding element type (line 52)
        # Getting the type of 'alpha' (line 52)
        alpha_14252 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 33), 'alpha', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 52, 33), tuple_14251, alpha_14252)
        # Adding element type (line 52)
        
        # Call to ones_like(...): (line 52)
        # Processing the call arguments (line 52)
        # Getting the type of 'alpha' (line 52)
        alpha_14255 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 56), 'alpha', False)
        # Processing the call keyword arguments (line 52)
        kwargs_14256 = {}
        # Getting the type of 'numpy' (line 52)
        numpy_14253 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 40), 'numpy', False)
        # Obtaining the member 'ones_like' of a type (line 52)
        ones_like_14254 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 52, 40), numpy_14253, 'ones_like')
        # Calling ones_like(args, kwargs) (line 52)
        ones_like_call_result_14257 = invoke(stypy.reporting.localization.Localization(__file__, 52, 40), ones_like_14254, *[alpha_14255], **kwargs_14256)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 52, 33), tuple_14251, ones_like_call_result_14257)
        
        # Processing the call keyword arguments (line 52)
        kwargs_14258 = {}
        # Getting the type of 'numpy' (line 52)
        numpy_14249 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 19), 'numpy', False)
        # Obtaining the member 'vstack' of a type (line 52)
        vstack_14250 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 52, 19), numpy_14249, 'vstack')
        # Calling vstack(args, kwargs) (line 52)
        vstack_call_result_14259 = invoke(stypy.reporting.localization.Localization(__file__, 52, 19), vstack_14250, *[tuple_14251], **kwargs_14258)
        
        # Assigning a type to the variable 'stypy_return_type' (line 52)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 52, 12), 'stypy_return_type', vstack_call_result_14259)

        if more_types_in_union_14248:
            # Runtime conditional SSA for else branch (line 51)
            module_type_store.open_ssa_branch('idiom else')



    if ((not may_be_14247) or more_types_in_union_14248):
        
        # Call to vstack(...): (line 54)
        # Processing the call arguments (line 54)
        
        # Obtaining an instance of the builtin type 'tuple' (line 54)
        tuple_14262 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 33), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 54)
        # Adding element type (line 54)
        # Getting the type of 'alpha' (line 54)
        alpha_14263 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 33), 'alpha', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 54, 33), tuple_14262, alpha_14263)
        # Adding element type (line 54)
        # Getting the type of 'beta' (line 54)
        beta_14264 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 40), 'beta', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 54, 33), tuple_14262, beta_14264)
        
        # Processing the call keyword arguments (line 54)
        kwargs_14265 = {}
        # Getting the type of 'numpy' (line 54)
        numpy_14260 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 19), 'numpy', False)
        # Obtaining the member 'vstack' of a type (line 54)
        vstack_14261 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 54, 19), numpy_14260, 'vstack')
        # Calling vstack(args, kwargs) (line 54)
        vstack_call_result_14266 = invoke(stypy.reporting.localization.Localization(__file__, 54, 19), vstack_14261, *[tuple_14262], **kwargs_14265)
        
        # Assigning a type to the variable 'stypy_return_type' (line 54)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 54, 12), 'stypy_return_type', vstack_call_result_14266)

        if (may_be_14247 and more_types_in_union_14248):
            # SSA join for if statement (line 51)
            module_type_store = module_type_store.join_ssa_context()


    
    # SSA branch for the else part of an if statement (line 50)
    module_type_store.open_ssa_branch('else')
    
    # Type idiom detected: calculating its left and rigth part (line 56)
    # Getting the type of 'beta' (line 56)
    beta_14267 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 11), 'beta')
    # Getting the type of 'None' (line 56)
    None_14268 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 19), 'None')
    
    (may_be_14269, more_types_in_union_14270) = may_be_none(beta_14267, None_14268)

    if may_be_14269:

        if more_types_in_union_14270:
            # Runtime conditional SSA (line 56)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        # Getting the type of 'alpha' (line 57)
        alpha_14271 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 19), 'alpha')
        # Assigning a type to the variable 'stypy_return_type' (line 57)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 57, 12), 'stypy_return_type', alpha_14271)

        if more_types_in_union_14270:
            # Runtime conditional SSA for else branch (line 56)
            module_type_store.open_ssa_branch('idiom else')



    if ((not may_be_14269) or more_types_in_union_14270):
        
        # Assigning a Call to a Name (line 59):
        
        # Assigning a Call to a Name (line 59):
        
        # Call to empty_like(...): (line 59)
        # Processing the call arguments (line 59)
        # Getting the type of 'alpha' (line 59)
        alpha_14274 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 33), 'alpha', False)
        # Processing the call keyword arguments (line 59)
        kwargs_14275 = {}
        # Getting the type of 'numpy' (line 59)
        numpy_14272 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 16), 'numpy', False)
        # Obtaining the member 'empty_like' of a type (line 59)
        empty_like_14273 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 59, 16), numpy_14272, 'empty_like')
        # Calling empty_like(args, kwargs) (line 59)
        empty_like_call_result_14276 = invoke(stypy.reporting.localization.Localization(__file__, 59, 16), empty_like_14273, *[alpha_14274], **kwargs_14275)
        
        # Assigning a type to the variable 'w' (line 59)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 59, 12), 'w', empty_like_call_result_14276)
        
        # Assigning a Compare to a Name (line 60):
        
        # Assigning a Compare to a Name (line 60):
        
        # Getting the type of 'alpha' (line 60)
        alpha_14277 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 26), 'alpha')
        int_14278 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, 35), 'int')
        # Applying the binary operator '==' (line 60)
        result_eq_14279 = python_operator(stypy.reporting.localization.Localization(__file__, 60, 26), '==', alpha_14277, int_14278)
        
        # Assigning a type to the variable 'alpha_zero' (line 60)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 60, 12), 'alpha_zero', result_eq_14279)
        
        # Assigning a Compare to a Name (line 61):
        
        # Assigning a Compare to a Name (line 61):
        
        # Getting the type of 'beta' (line 61)
        beta_14280 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 25), 'beta')
        int_14281 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 61, 33), 'int')
        # Applying the binary operator '==' (line 61)
        result_eq_14282 = python_operator(stypy.reporting.localization.Localization(__file__, 61, 25), '==', beta_14280, int_14281)
        
        # Assigning a type to the variable 'beta_zero' (line 61)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 61, 12), 'beta_zero', result_eq_14282)
        
        # Assigning a UnaryOp to a Name (line 62):
        
        # Assigning a UnaryOp to a Name (line 62):
        
        # Getting the type of 'beta_zero' (line 62)
        beta_zero_14283 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 28), 'beta_zero')
        # Applying the '~' unary operator (line 62)
        result_inv_14284 = python_operator(stypy.reporting.localization.Localization(__file__, 62, 27), '~', beta_zero_14283)
        
        # Assigning a type to the variable 'beta_nonzero' (line 62)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 62, 12), 'beta_nonzero', result_inv_14284)
        
        # Assigning a BinOp to a Subscript (line 63):
        
        # Assigning a BinOp to a Subscript (line 63):
        
        # Obtaining the type of the subscript
        # Getting the type of 'beta_nonzero' (line 63)
        beta_nonzero_14285 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 36), 'beta_nonzero')
        # Getting the type of 'alpha' (line 63)
        alpha_14286 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 30), 'alpha')
        # Obtaining the member '__getitem__' of a type (line 63)
        getitem___14287 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 63, 30), alpha_14286, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 63)
        subscript_call_result_14288 = invoke(stypy.reporting.localization.Localization(__file__, 63, 30), getitem___14287, beta_nonzero_14285)
        
        
        # Obtaining the type of the subscript
        # Getting the type of 'beta_nonzero' (line 63)
        beta_nonzero_14289 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 55), 'beta_nonzero')
        # Getting the type of 'beta' (line 63)
        beta_14290 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 50), 'beta')
        # Obtaining the member '__getitem__' of a type (line 63)
        getitem___14291 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 63, 50), beta_14290, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 63)
        subscript_call_result_14292 = invoke(stypy.reporting.localization.Localization(__file__, 63, 50), getitem___14291, beta_nonzero_14289)
        
        # Applying the binary operator 'div' (line 63)
        result_div_14293 = python_operator(stypy.reporting.localization.Localization(__file__, 63, 30), 'div', subscript_call_result_14288, subscript_call_result_14292)
        
        # Getting the type of 'w' (line 63)
        w_14294 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 12), 'w')
        # Getting the type of 'beta_nonzero' (line 63)
        beta_nonzero_14295 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 14), 'beta_nonzero')
        # Storing an element on a container (line 63)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 63, 12), w_14294, (beta_nonzero_14295, result_div_14293))
        
        # Assigning a Attribute to a Subscript (line 67):
        
        # Assigning a Attribute to a Subscript (line 67):
        # Getting the type of 'numpy' (line 67)
        numpy_14296 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 41), 'numpy')
        # Obtaining the member 'inf' of a type (line 67)
        inf_14297 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 67, 41), numpy_14296, 'inf')
        # Getting the type of 'w' (line 67)
        w_14298 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 12), 'w')
        
        # Getting the type of 'alpha_zero' (line 67)
        alpha_zero_14299 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 15), 'alpha_zero')
        # Applying the '~' unary operator (line 67)
        result_inv_14300 = python_operator(stypy.reporting.localization.Localization(__file__, 67, 14), '~', alpha_zero_14299)
        
        # Getting the type of 'beta_zero' (line 67)
        beta_zero_14301 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 28), 'beta_zero')
        # Applying the binary operator '&' (line 67)
        result_and__14302 = python_operator(stypy.reporting.localization.Localization(__file__, 67, 14), '&', result_inv_14300, beta_zero_14301)
        
        # Storing an element on a container (line 67)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 67, 12), w_14298, (result_and__14302, inf_14297))
        
        
        # Call to all(...): (line 68)
        # Processing the call arguments (line 68)
        
        # Getting the type of 'alpha' (line 68)
        alpha_14305 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 25), 'alpha', False)
        # Obtaining the member 'imag' of a type (line 68)
        imag_14306 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 68, 25), alpha_14305, 'imag')
        int_14307 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 39), 'int')
        # Applying the binary operator '==' (line 68)
        result_eq_14308 = python_operator(stypy.reporting.localization.Localization(__file__, 68, 25), '==', imag_14306, int_14307)
        
        # Processing the call keyword arguments (line 68)
        kwargs_14309 = {}
        # Getting the type of 'numpy' (line 68)
        numpy_14303 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 15), 'numpy', False)
        # Obtaining the member 'all' of a type (line 68)
        all_14304 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 68, 15), numpy_14303, 'all')
        # Calling all(args, kwargs) (line 68)
        all_call_result_14310 = invoke(stypy.reporting.localization.Localization(__file__, 68, 15), all_14304, *[result_eq_14308], **kwargs_14309)
        
        # Testing the type of an if condition (line 68)
        if_condition_14311 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 68, 12), all_call_result_14310)
        # Assigning a type to the variable 'if_condition_14311' (line 68)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 68, 12), 'if_condition_14311', if_condition_14311)
        # SSA begins for if statement (line 68)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Attribute to a Subscript (line 69):
        
        # Assigning a Attribute to a Subscript (line 69):
        # Getting the type of 'numpy' (line 69)
        numpy_14312 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 44), 'numpy')
        # Obtaining the member 'nan' of a type (line 69)
        nan_14313 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 69, 44), numpy_14312, 'nan')
        # Getting the type of 'w' (line 69)
        w_14314 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 16), 'w')
        # Getting the type of 'alpha_zero' (line 69)
        alpha_zero_14315 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 18), 'alpha_zero')
        # Getting the type of 'beta_zero' (line 69)
        beta_zero_14316 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 31), 'beta_zero')
        # Applying the binary operator '&' (line 69)
        result_and__14317 = python_operator(stypy.reporting.localization.Localization(__file__, 69, 18), '&', alpha_zero_14315, beta_zero_14316)
        
        # Storing an element on a container (line 69)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 69, 16), w_14314, (result_and__14317, nan_14313))
        # SSA branch for the else part of an if statement (line 68)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Call to a Subscript (line 71):
        
        # Assigning a Call to a Subscript (line 71):
        
        # Call to complex(...): (line 71)
        # Processing the call arguments (line 71)
        # Getting the type of 'numpy' (line 71)
        numpy_14319 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 52), 'numpy', False)
        # Obtaining the member 'nan' of a type (line 71)
        nan_14320 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 71, 52), numpy_14319, 'nan')
        # Getting the type of 'numpy' (line 71)
        numpy_14321 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 63), 'numpy', False)
        # Obtaining the member 'nan' of a type (line 71)
        nan_14322 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 71, 63), numpy_14321, 'nan')
        # Processing the call keyword arguments (line 71)
        kwargs_14323 = {}
        # Getting the type of 'complex' (line 71)
        complex_14318 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 44), 'complex', False)
        # Calling complex(args, kwargs) (line 71)
        complex_call_result_14324 = invoke(stypy.reporting.localization.Localization(__file__, 71, 44), complex_14318, *[nan_14320, nan_14322], **kwargs_14323)
        
        # Getting the type of 'w' (line 71)
        w_14325 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 16), 'w')
        # Getting the type of 'alpha_zero' (line 71)
        alpha_zero_14326 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 18), 'alpha_zero')
        # Getting the type of 'beta_zero' (line 71)
        beta_zero_14327 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 31), 'beta_zero')
        # Applying the binary operator '&' (line 71)
        result_and__14328 = python_operator(stypy.reporting.localization.Localization(__file__, 71, 18), '&', alpha_zero_14326, beta_zero_14327)
        
        # Storing an element on a container (line 71)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 71, 16), w_14325, (result_and__14328, complex_call_result_14324))
        # SSA join for if statement (line 68)
        module_type_store = module_type_store.join_ssa_context()
        
        # Getting the type of 'w' (line 72)
        w_14329 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 19), 'w')
        # Assigning a type to the variable 'stypy_return_type' (line 72)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 72, 12), 'stypy_return_type', w_14329)

        if (may_be_14269 and more_types_in_union_14270):
            # SSA join for if statement (line 56)
            module_type_store = module_type_store.join_ssa_context()


    
    # SSA join for if statement (line 50)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of '_make_eigvals(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_make_eigvals' in the type store
    # Getting the type of 'stypy_return_type' (line 49)
    stypy_return_type_14330 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_14330)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_make_eigvals'
    return stypy_return_type_14330

# Assigning a type to the variable '_make_eigvals' (line 49)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 49, 0), '_make_eigvals', _make_eigvals)

@norecursion
def _geneig(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_geneig'
    module_type_store = module_type_store.open_function_context('_geneig', 75, 0, False)
    
    # Passed parameters checking function
    _geneig.stypy_localization = localization
    _geneig.stypy_type_of_self = None
    _geneig.stypy_type_store = module_type_store
    _geneig.stypy_function_name = '_geneig'
    _geneig.stypy_param_names_list = ['a1', 'b1', 'left', 'right', 'overwrite_a', 'overwrite_b', 'homogeneous_eigvals']
    _geneig.stypy_varargs_param_name = None
    _geneig.stypy_kwargs_param_name = None
    _geneig.stypy_call_defaults = defaults
    _geneig.stypy_call_varargs = varargs
    _geneig.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_geneig', ['a1', 'b1', 'left', 'right', 'overwrite_a', 'overwrite_b', 'homogeneous_eigvals'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_geneig', localization, ['a1', 'b1', 'left', 'right', 'overwrite_a', 'overwrite_b', 'homogeneous_eigvals'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_geneig(...)' code ##################

    
    # Assigning a Call to a Tuple (line 77):
    
    # Assigning a Subscript to a Name (line 77):
    
    # Obtaining the type of the subscript
    int_14331 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 77, 4), 'int')
    
    # Call to get_lapack_funcs(...): (line 77)
    # Processing the call arguments (line 77)
    
    # Obtaining an instance of the builtin type 'tuple' (line 77)
    tuple_14333 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 77, 30), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 77)
    # Adding element type (line 77)
    str_14334 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 77, 30), 'str', 'ggev')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 77, 30), tuple_14333, str_14334)
    
    
    # Obtaining an instance of the builtin type 'tuple' (line 77)
    tuple_14335 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 77, 41), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 77)
    # Adding element type (line 77)
    # Getting the type of 'a1' (line 77)
    a1_14336 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 41), 'a1', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 77, 41), tuple_14335, a1_14336)
    # Adding element type (line 77)
    # Getting the type of 'b1' (line 77)
    b1_14337 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 45), 'b1', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 77, 41), tuple_14335, b1_14337)
    
    # Processing the call keyword arguments (line 77)
    kwargs_14338 = {}
    # Getting the type of 'get_lapack_funcs' (line 77)
    get_lapack_funcs_14332 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 12), 'get_lapack_funcs', False)
    # Calling get_lapack_funcs(args, kwargs) (line 77)
    get_lapack_funcs_call_result_14339 = invoke(stypy.reporting.localization.Localization(__file__, 77, 12), get_lapack_funcs_14332, *[tuple_14333, tuple_14335], **kwargs_14338)
    
    # Obtaining the member '__getitem__' of a type (line 77)
    getitem___14340 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 77, 4), get_lapack_funcs_call_result_14339, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 77)
    subscript_call_result_14341 = invoke(stypy.reporting.localization.Localization(__file__, 77, 4), getitem___14340, int_14331)
    
    # Assigning a type to the variable 'tuple_var_assignment_14024' (line 77)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 77, 4), 'tuple_var_assignment_14024', subscript_call_result_14341)
    
    # Assigning a Name to a Name (line 77):
    # Getting the type of 'tuple_var_assignment_14024' (line 77)
    tuple_var_assignment_14024_14342 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 4), 'tuple_var_assignment_14024')
    # Assigning a type to the variable 'ggev' (line 77)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 77, 4), 'ggev', tuple_var_assignment_14024_14342)
    
    # Assigning a Tuple to a Tuple (line 78):
    
    # Assigning a Name to a Name (line 78):
    # Getting the type of 'left' (line 78)
    left_14343 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 15), 'left')
    # Assigning a type to the variable 'tuple_assignment_14025' (line 78)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 78, 4), 'tuple_assignment_14025', left_14343)
    
    # Assigning a Name to a Name (line 78):
    # Getting the type of 'right' (line 78)
    right_14344 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 21), 'right')
    # Assigning a type to the variable 'tuple_assignment_14026' (line 78)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 78, 4), 'tuple_assignment_14026', right_14344)
    
    # Assigning a Name to a Name (line 78):
    # Getting the type of 'tuple_assignment_14025' (line 78)
    tuple_assignment_14025_14345 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 4), 'tuple_assignment_14025')
    # Assigning a type to the variable 'cvl' (line 78)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 78, 4), 'cvl', tuple_assignment_14025_14345)
    
    # Assigning a Name to a Name (line 78):
    # Getting the type of 'tuple_assignment_14026' (line 78)
    tuple_assignment_14026_14346 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 4), 'tuple_assignment_14026')
    # Assigning a type to the variable 'cvr' (line 78)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 78, 9), 'cvr', tuple_assignment_14026_14346)
    
    # Assigning a Call to a Name (line 79):
    
    # Assigning a Call to a Name (line 79):
    
    # Call to ggev(...): (line 79)
    # Processing the call arguments (line 79)
    # Getting the type of 'a1' (line 79)
    a1_14348 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 15), 'a1', False)
    # Getting the type of 'b1' (line 79)
    b1_14349 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 19), 'b1', False)
    # Processing the call keyword arguments (line 79)
    int_14350 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 79, 29), 'int')
    keyword_14351 = int_14350
    kwargs_14352 = {'lwork': keyword_14351}
    # Getting the type of 'ggev' (line 79)
    ggev_14347 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 10), 'ggev', False)
    # Calling ggev(args, kwargs) (line 79)
    ggev_call_result_14353 = invoke(stypy.reporting.localization.Localization(__file__, 79, 10), ggev_14347, *[a1_14348, b1_14349], **kwargs_14352)
    
    # Assigning a type to the variable 'res' (line 79)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 79, 4), 'res', ggev_call_result_14353)
    
    # Assigning a Call to a Name (line 80):
    
    # Assigning a Call to a Name (line 80):
    
    # Call to astype(...): (line 80)
    # Processing the call arguments (line 80)
    # Getting the type of 'numpy' (line 80)
    numpy_14363 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 35), 'numpy', False)
    # Obtaining the member 'int' of a type (line 80)
    int_14364 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 80, 35), numpy_14363, 'int')
    # Processing the call keyword arguments (line 80)
    kwargs_14365 = {}
    
    # Obtaining the type of the subscript
    int_14354 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 80, 20), 'int')
    
    # Obtaining the type of the subscript
    int_14355 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 80, 16), 'int')
    # Getting the type of 'res' (line 80)
    res_14356 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 12), 'res', False)
    # Obtaining the member '__getitem__' of a type (line 80)
    getitem___14357 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 80, 12), res_14356, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 80)
    subscript_call_result_14358 = invoke(stypy.reporting.localization.Localization(__file__, 80, 12), getitem___14357, int_14355)
    
    # Obtaining the member '__getitem__' of a type (line 80)
    getitem___14359 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 80, 12), subscript_call_result_14358, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 80)
    subscript_call_result_14360 = invoke(stypy.reporting.localization.Localization(__file__, 80, 12), getitem___14359, int_14354)
    
    # Obtaining the member 'real' of a type (line 80)
    real_14361 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 80, 12), subscript_call_result_14360, 'real')
    # Obtaining the member 'astype' of a type (line 80)
    astype_14362 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 80, 12), real_14361, 'astype')
    # Calling astype(args, kwargs) (line 80)
    astype_call_result_14366 = invoke(stypy.reporting.localization.Localization(__file__, 80, 12), astype_14362, *[int_14364], **kwargs_14365)
    
    # Assigning a type to the variable 'lwork' (line 80)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 80, 4), 'lwork', astype_call_result_14366)
    
    
    # Getting the type of 'ggev' (line 81)
    ggev_14367 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 7), 'ggev')
    # Obtaining the member 'typecode' of a type (line 81)
    typecode_14368 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 81, 7), ggev_14367, 'typecode')
    str_14369 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 81, 24), 'str', 'cz')
    # Applying the binary operator 'in' (line 81)
    result_contains_14370 = python_operator(stypy.reporting.localization.Localization(__file__, 81, 7), 'in', typecode_14368, str_14369)
    
    # Testing the type of an if condition (line 81)
    if_condition_14371 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 81, 4), result_contains_14370)
    # Assigning a type to the variable 'if_condition_14371' (line 81)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 81, 4), 'if_condition_14371', if_condition_14371)
    # SSA begins for if statement (line 81)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Tuple (line 82):
    
    # Assigning a Subscript to a Name (line 82):
    
    # Obtaining the type of the subscript
    int_14372 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 82, 8), 'int')
    
    # Call to ggev(...): (line 82)
    # Processing the call arguments (line 82)
    # Getting the type of 'a1' (line 82)
    a1_14374 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 47), 'a1', False)
    # Getting the type of 'b1' (line 82)
    b1_14375 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 51), 'b1', False)
    # Getting the type of 'cvl' (line 82)
    cvl_14376 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 55), 'cvl', False)
    # Getting the type of 'cvr' (line 82)
    cvr_14377 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 60), 'cvr', False)
    # Getting the type of 'lwork' (line 82)
    lwork_14378 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 65), 'lwork', False)
    # Getting the type of 'overwrite_a' (line 83)
    overwrite_a_14379 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 47), 'overwrite_a', False)
    # Getting the type of 'overwrite_b' (line 83)
    overwrite_b_14380 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 60), 'overwrite_b', False)
    # Processing the call keyword arguments (line 82)
    kwargs_14381 = {}
    # Getting the type of 'ggev' (line 82)
    ggev_14373 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 42), 'ggev', False)
    # Calling ggev(args, kwargs) (line 82)
    ggev_call_result_14382 = invoke(stypy.reporting.localization.Localization(__file__, 82, 42), ggev_14373, *[a1_14374, b1_14375, cvl_14376, cvr_14377, lwork_14378, overwrite_a_14379, overwrite_b_14380], **kwargs_14381)
    
    # Obtaining the member '__getitem__' of a type (line 82)
    getitem___14383 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 82, 8), ggev_call_result_14382, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 82)
    subscript_call_result_14384 = invoke(stypy.reporting.localization.Localization(__file__, 82, 8), getitem___14383, int_14372)
    
    # Assigning a type to the variable 'tuple_var_assignment_14027' (line 82)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 82, 8), 'tuple_var_assignment_14027', subscript_call_result_14384)
    
    # Assigning a Subscript to a Name (line 82):
    
    # Obtaining the type of the subscript
    int_14385 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 82, 8), 'int')
    
    # Call to ggev(...): (line 82)
    # Processing the call arguments (line 82)
    # Getting the type of 'a1' (line 82)
    a1_14387 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 47), 'a1', False)
    # Getting the type of 'b1' (line 82)
    b1_14388 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 51), 'b1', False)
    # Getting the type of 'cvl' (line 82)
    cvl_14389 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 55), 'cvl', False)
    # Getting the type of 'cvr' (line 82)
    cvr_14390 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 60), 'cvr', False)
    # Getting the type of 'lwork' (line 82)
    lwork_14391 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 65), 'lwork', False)
    # Getting the type of 'overwrite_a' (line 83)
    overwrite_a_14392 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 47), 'overwrite_a', False)
    # Getting the type of 'overwrite_b' (line 83)
    overwrite_b_14393 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 60), 'overwrite_b', False)
    # Processing the call keyword arguments (line 82)
    kwargs_14394 = {}
    # Getting the type of 'ggev' (line 82)
    ggev_14386 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 42), 'ggev', False)
    # Calling ggev(args, kwargs) (line 82)
    ggev_call_result_14395 = invoke(stypy.reporting.localization.Localization(__file__, 82, 42), ggev_14386, *[a1_14387, b1_14388, cvl_14389, cvr_14390, lwork_14391, overwrite_a_14392, overwrite_b_14393], **kwargs_14394)
    
    # Obtaining the member '__getitem__' of a type (line 82)
    getitem___14396 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 82, 8), ggev_call_result_14395, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 82)
    subscript_call_result_14397 = invoke(stypy.reporting.localization.Localization(__file__, 82, 8), getitem___14396, int_14385)
    
    # Assigning a type to the variable 'tuple_var_assignment_14028' (line 82)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 82, 8), 'tuple_var_assignment_14028', subscript_call_result_14397)
    
    # Assigning a Subscript to a Name (line 82):
    
    # Obtaining the type of the subscript
    int_14398 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 82, 8), 'int')
    
    # Call to ggev(...): (line 82)
    # Processing the call arguments (line 82)
    # Getting the type of 'a1' (line 82)
    a1_14400 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 47), 'a1', False)
    # Getting the type of 'b1' (line 82)
    b1_14401 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 51), 'b1', False)
    # Getting the type of 'cvl' (line 82)
    cvl_14402 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 55), 'cvl', False)
    # Getting the type of 'cvr' (line 82)
    cvr_14403 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 60), 'cvr', False)
    # Getting the type of 'lwork' (line 82)
    lwork_14404 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 65), 'lwork', False)
    # Getting the type of 'overwrite_a' (line 83)
    overwrite_a_14405 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 47), 'overwrite_a', False)
    # Getting the type of 'overwrite_b' (line 83)
    overwrite_b_14406 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 60), 'overwrite_b', False)
    # Processing the call keyword arguments (line 82)
    kwargs_14407 = {}
    # Getting the type of 'ggev' (line 82)
    ggev_14399 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 42), 'ggev', False)
    # Calling ggev(args, kwargs) (line 82)
    ggev_call_result_14408 = invoke(stypy.reporting.localization.Localization(__file__, 82, 42), ggev_14399, *[a1_14400, b1_14401, cvl_14402, cvr_14403, lwork_14404, overwrite_a_14405, overwrite_b_14406], **kwargs_14407)
    
    # Obtaining the member '__getitem__' of a type (line 82)
    getitem___14409 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 82, 8), ggev_call_result_14408, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 82)
    subscript_call_result_14410 = invoke(stypy.reporting.localization.Localization(__file__, 82, 8), getitem___14409, int_14398)
    
    # Assigning a type to the variable 'tuple_var_assignment_14029' (line 82)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 82, 8), 'tuple_var_assignment_14029', subscript_call_result_14410)
    
    # Assigning a Subscript to a Name (line 82):
    
    # Obtaining the type of the subscript
    int_14411 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 82, 8), 'int')
    
    # Call to ggev(...): (line 82)
    # Processing the call arguments (line 82)
    # Getting the type of 'a1' (line 82)
    a1_14413 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 47), 'a1', False)
    # Getting the type of 'b1' (line 82)
    b1_14414 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 51), 'b1', False)
    # Getting the type of 'cvl' (line 82)
    cvl_14415 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 55), 'cvl', False)
    # Getting the type of 'cvr' (line 82)
    cvr_14416 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 60), 'cvr', False)
    # Getting the type of 'lwork' (line 82)
    lwork_14417 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 65), 'lwork', False)
    # Getting the type of 'overwrite_a' (line 83)
    overwrite_a_14418 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 47), 'overwrite_a', False)
    # Getting the type of 'overwrite_b' (line 83)
    overwrite_b_14419 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 60), 'overwrite_b', False)
    # Processing the call keyword arguments (line 82)
    kwargs_14420 = {}
    # Getting the type of 'ggev' (line 82)
    ggev_14412 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 42), 'ggev', False)
    # Calling ggev(args, kwargs) (line 82)
    ggev_call_result_14421 = invoke(stypy.reporting.localization.Localization(__file__, 82, 42), ggev_14412, *[a1_14413, b1_14414, cvl_14415, cvr_14416, lwork_14417, overwrite_a_14418, overwrite_b_14419], **kwargs_14420)
    
    # Obtaining the member '__getitem__' of a type (line 82)
    getitem___14422 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 82, 8), ggev_call_result_14421, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 82)
    subscript_call_result_14423 = invoke(stypy.reporting.localization.Localization(__file__, 82, 8), getitem___14422, int_14411)
    
    # Assigning a type to the variable 'tuple_var_assignment_14030' (line 82)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 82, 8), 'tuple_var_assignment_14030', subscript_call_result_14423)
    
    # Assigning a Subscript to a Name (line 82):
    
    # Obtaining the type of the subscript
    int_14424 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 82, 8), 'int')
    
    # Call to ggev(...): (line 82)
    # Processing the call arguments (line 82)
    # Getting the type of 'a1' (line 82)
    a1_14426 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 47), 'a1', False)
    # Getting the type of 'b1' (line 82)
    b1_14427 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 51), 'b1', False)
    # Getting the type of 'cvl' (line 82)
    cvl_14428 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 55), 'cvl', False)
    # Getting the type of 'cvr' (line 82)
    cvr_14429 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 60), 'cvr', False)
    # Getting the type of 'lwork' (line 82)
    lwork_14430 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 65), 'lwork', False)
    # Getting the type of 'overwrite_a' (line 83)
    overwrite_a_14431 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 47), 'overwrite_a', False)
    # Getting the type of 'overwrite_b' (line 83)
    overwrite_b_14432 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 60), 'overwrite_b', False)
    # Processing the call keyword arguments (line 82)
    kwargs_14433 = {}
    # Getting the type of 'ggev' (line 82)
    ggev_14425 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 42), 'ggev', False)
    # Calling ggev(args, kwargs) (line 82)
    ggev_call_result_14434 = invoke(stypy.reporting.localization.Localization(__file__, 82, 42), ggev_14425, *[a1_14426, b1_14427, cvl_14428, cvr_14429, lwork_14430, overwrite_a_14431, overwrite_b_14432], **kwargs_14433)
    
    # Obtaining the member '__getitem__' of a type (line 82)
    getitem___14435 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 82, 8), ggev_call_result_14434, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 82)
    subscript_call_result_14436 = invoke(stypy.reporting.localization.Localization(__file__, 82, 8), getitem___14435, int_14424)
    
    # Assigning a type to the variable 'tuple_var_assignment_14031' (line 82)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 82, 8), 'tuple_var_assignment_14031', subscript_call_result_14436)
    
    # Assigning a Subscript to a Name (line 82):
    
    # Obtaining the type of the subscript
    int_14437 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 82, 8), 'int')
    
    # Call to ggev(...): (line 82)
    # Processing the call arguments (line 82)
    # Getting the type of 'a1' (line 82)
    a1_14439 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 47), 'a1', False)
    # Getting the type of 'b1' (line 82)
    b1_14440 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 51), 'b1', False)
    # Getting the type of 'cvl' (line 82)
    cvl_14441 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 55), 'cvl', False)
    # Getting the type of 'cvr' (line 82)
    cvr_14442 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 60), 'cvr', False)
    # Getting the type of 'lwork' (line 82)
    lwork_14443 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 65), 'lwork', False)
    # Getting the type of 'overwrite_a' (line 83)
    overwrite_a_14444 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 47), 'overwrite_a', False)
    # Getting the type of 'overwrite_b' (line 83)
    overwrite_b_14445 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 60), 'overwrite_b', False)
    # Processing the call keyword arguments (line 82)
    kwargs_14446 = {}
    # Getting the type of 'ggev' (line 82)
    ggev_14438 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 42), 'ggev', False)
    # Calling ggev(args, kwargs) (line 82)
    ggev_call_result_14447 = invoke(stypy.reporting.localization.Localization(__file__, 82, 42), ggev_14438, *[a1_14439, b1_14440, cvl_14441, cvr_14442, lwork_14443, overwrite_a_14444, overwrite_b_14445], **kwargs_14446)
    
    # Obtaining the member '__getitem__' of a type (line 82)
    getitem___14448 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 82, 8), ggev_call_result_14447, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 82)
    subscript_call_result_14449 = invoke(stypy.reporting.localization.Localization(__file__, 82, 8), getitem___14448, int_14437)
    
    # Assigning a type to the variable 'tuple_var_assignment_14032' (line 82)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 82, 8), 'tuple_var_assignment_14032', subscript_call_result_14449)
    
    # Assigning a Name to a Name (line 82):
    # Getting the type of 'tuple_var_assignment_14027' (line 82)
    tuple_var_assignment_14027_14450 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 8), 'tuple_var_assignment_14027')
    # Assigning a type to the variable 'alpha' (line 82)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 82, 8), 'alpha', tuple_var_assignment_14027_14450)
    
    # Assigning a Name to a Name (line 82):
    # Getting the type of 'tuple_var_assignment_14028' (line 82)
    tuple_var_assignment_14028_14451 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 8), 'tuple_var_assignment_14028')
    # Assigning a type to the variable 'beta' (line 82)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 82, 15), 'beta', tuple_var_assignment_14028_14451)
    
    # Assigning a Name to a Name (line 82):
    # Getting the type of 'tuple_var_assignment_14029' (line 82)
    tuple_var_assignment_14029_14452 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 8), 'tuple_var_assignment_14029')
    # Assigning a type to the variable 'vl' (line 82)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 82, 21), 'vl', tuple_var_assignment_14029_14452)
    
    # Assigning a Name to a Name (line 82):
    # Getting the type of 'tuple_var_assignment_14030' (line 82)
    tuple_var_assignment_14030_14453 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 8), 'tuple_var_assignment_14030')
    # Assigning a type to the variable 'vr' (line 82)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 82, 25), 'vr', tuple_var_assignment_14030_14453)
    
    # Assigning a Name to a Name (line 82):
    # Getting the type of 'tuple_var_assignment_14031' (line 82)
    tuple_var_assignment_14031_14454 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 8), 'tuple_var_assignment_14031')
    # Assigning a type to the variable 'work' (line 82)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 82, 29), 'work', tuple_var_assignment_14031_14454)
    
    # Assigning a Name to a Name (line 82):
    # Getting the type of 'tuple_var_assignment_14032' (line 82)
    tuple_var_assignment_14032_14455 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 8), 'tuple_var_assignment_14032')
    # Assigning a type to the variable 'info' (line 82)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 82, 35), 'info', tuple_var_assignment_14032_14455)
    
    # Assigning a Call to a Name (line 84):
    
    # Assigning a Call to a Name (line 84):
    
    # Call to _make_eigvals(...): (line 84)
    # Processing the call arguments (line 84)
    # Getting the type of 'alpha' (line 84)
    alpha_14457 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 26), 'alpha', False)
    # Getting the type of 'beta' (line 84)
    beta_14458 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 33), 'beta', False)
    # Getting the type of 'homogeneous_eigvals' (line 84)
    homogeneous_eigvals_14459 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 39), 'homogeneous_eigvals', False)
    # Processing the call keyword arguments (line 84)
    kwargs_14460 = {}
    # Getting the type of '_make_eigvals' (line 84)
    _make_eigvals_14456 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 12), '_make_eigvals', False)
    # Calling _make_eigvals(args, kwargs) (line 84)
    _make_eigvals_call_result_14461 = invoke(stypy.reporting.localization.Localization(__file__, 84, 12), _make_eigvals_14456, *[alpha_14457, beta_14458, homogeneous_eigvals_14459], **kwargs_14460)
    
    # Assigning a type to the variable 'w' (line 84)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 84, 8), 'w', _make_eigvals_call_result_14461)
    # SSA branch for the else part of an if statement (line 81)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Call to a Tuple (line 86):
    
    # Assigning a Subscript to a Name (line 86):
    
    # Obtaining the type of the subscript
    int_14462 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 86, 8), 'int')
    
    # Call to ggev(...): (line 86)
    # Processing the call arguments (line 86)
    # Getting the type of 'a1' (line 86)
    a1_14464 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 56), 'a1', False)
    # Getting the type of 'b1' (line 86)
    b1_14465 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 60), 'b1', False)
    # Getting the type of 'cvl' (line 86)
    cvl_14466 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 64), 'cvl', False)
    # Getting the type of 'cvr' (line 86)
    cvr_14467 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 69), 'cvr', False)
    # Getting the type of 'lwork' (line 87)
    lwork_14468 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 56), 'lwork', False)
    # Getting the type of 'overwrite_a' (line 87)
    overwrite_a_14469 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 63), 'overwrite_a', False)
    # Getting the type of 'overwrite_b' (line 88)
    overwrite_b_14470 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 56), 'overwrite_b', False)
    # Processing the call keyword arguments (line 86)
    kwargs_14471 = {}
    # Getting the type of 'ggev' (line 86)
    ggev_14463 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 51), 'ggev', False)
    # Calling ggev(args, kwargs) (line 86)
    ggev_call_result_14472 = invoke(stypy.reporting.localization.Localization(__file__, 86, 51), ggev_14463, *[a1_14464, b1_14465, cvl_14466, cvr_14467, lwork_14468, overwrite_a_14469, overwrite_b_14470], **kwargs_14471)
    
    # Obtaining the member '__getitem__' of a type (line 86)
    getitem___14473 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 86, 8), ggev_call_result_14472, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 86)
    subscript_call_result_14474 = invoke(stypy.reporting.localization.Localization(__file__, 86, 8), getitem___14473, int_14462)
    
    # Assigning a type to the variable 'tuple_var_assignment_14033' (line 86)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 86, 8), 'tuple_var_assignment_14033', subscript_call_result_14474)
    
    # Assigning a Subscript to a Name (line 86):
    
    # Obtaining the type of the subscript
    int_14475 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 86, 8), 'int')
    
    # Call to ggev(...): (line 86)
    # Processing the call arguments (line 86)
    # Getting the type of 'a1' (line 86)
    a1_14477 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 56), 'a1', False)
    # Getting the type of 'b1' (line 86)
    b1_14478 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 60), 'b1', False)
    # Getting the type of 'cvl' (line 86)
    cvl_14479 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 64), 'cvl', False)
    # Getting the type of 'cvr' (line 86)
    cvr_14480 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 69), 'cvr', False)
    # Getting the type of 'lwork' (line 87)
    lwork_14481 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 56), 'lwork', False)
    # Getting the type of 'overwrite_a' (line 87)
    overwrite_a_14482 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 63), 'overwrite_a', False)
    # Getting the type of 'overwrite_b' (line 88)
    overwrite_b_14483 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 56), 'overwrite_b', False)
    # Processing the call keyword arguments (line 86)
    kwargs_14484 = {}
    # Getting the type of 'ggev' (line 86)
    ggev_14476 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 51), 'ggev', False)
    # Calling ggev(args, kwargs) (line 86)
    ggev_call_result_14485 = invoke(stypy.reporting.localization.Localization(__file__, 86, 51), ggev_14476, *[a1_14477, b1_14478, cvl_14479, cvr_14480, lwork_14481, overwrite_a_14482, overwrite_b_14483], **kwargs_14484)
    
    # Obtaining the member '__getitem__' of a type (line 86)
    getitem___14486 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 86, 8), ggev_call_result_14485, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 86)
    subscript_call_result_14487 = invoke(stypy.reporting.localization.Localization(__file__, 86, 8), getitem___14486, int_14475)
    
    # Assigning a type to the variable 'tuple_var_assignment_14034' (line 86)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 86, 8), 'tuple_var_assignment_14034', subscript_call_result_14487)
    
    # Assigning a Subscript to a Name (line 86):
    
    # Obtaining the type of the subscript
    int_14488 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 86, 8), 'int')
    
    # Call to ggev(...): (line 86)
    # Processing the call arguments (line 86)
    # Getting the type of 'a1' (line 86)
    a1_14490 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 56), 'a1', False)
    # Getting the type of 'b1' (line 86)
    b1_14491 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 60), 'b1', False)
    # Getting the type of 'cvl' (line 86)
    cvl_14492 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 64), 'cvl', False)
    # Getting the type of 'cvr' (line 86)
    cvr_14493 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 69), 'cvr', False)
    # Getting the type of 'lwork' (line 87)
    lwork_14494 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 56), 'lwork', False)
    # Getting the type of 'overwrite_a' (line 87)
    overwrite_a_14495 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 63), 'overwrite_a', False)
    # Getting the type of 'overwrite_b' (line 88)
    overwrite_b_14496 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 56), 'overwrite_b', False)
    # Processing the call keyword arguments (line 86)
    kwargs_14497 = {}
    # Getting the type of 'ggev' (line 86)
    ggev_14489 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 51), 'ggev', False)
    # Calling ggev(args, kwargs) (line 86)
    ggev_call_result_14498 = invoke(stypy.reporting.localization.Localization(__file__, 86, 51), ggev_14489, *[a1_14490, b1_14491, cvl_14492, cvr_14493, lwork_14494, overwrite_a_14495, overwrite_b_14496], **kwargs_14497)
    
    # Obtaining the member '__getitem__' of a type (line 86)
    getitem___14499 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 86, 8), ggev_call_result_14498, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 86)
    subscript_call_result_14500 = invoke(stypy.reporting.localization.Localization(__file__, 86, 8), getitem___14499, int_14488)
    
    # Assigning a type to the variable 'tuple_var_assignment_14035' (line 86)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 86, 8), 'tuple_var_assignment_14035', subscript_call_result_14500)
    
    # Assigning a Subscript to a Name (line 86):
    
    # Obtaining the type of the subscript
    int_14501 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 86, 8), 'int')
    
    # Call to ggev(...): (line 86)
    # Processing the call arguments (line 86)
    # Getting the type of 'a1' (line 86)
    a1_14503 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 56), 'a1', False)
    # Getting the type of 'b1' (line 86)
    b1_14504 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 60), 'b1', False)
    # Getting the type of 'cvl' (line 86)
    cvl_14505 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 64), 'cvl', False)
    # Getting the type of 'cvr' (line 86)
    cvr_14506 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 69), 'cvr', False)
    # Getting the type of 'lwork' (line 87)
    lwork_14507 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 56), 'lwork', False)
    # Getting the type of 'overwrite_a' (line 87)
    overwrite_a_14508 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 63), 'overwrite_a', False)
    # Getting the type of 'overwrite_b' (line 88)
    overwrite_b_14509 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 56), 'overwrite_b', False)
    # Processing the call keyword arguments (line 86)
    kwargs_14510 = {}
    # Getting the type of 'ggev' (line 86)
    ggev_14502 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 51), 'ggev', False)
    # Calling ggev(args, kwargs) (line 86)
    ggev_call_result_14511 = invoke(stypy.reporting.localization.Localization(__file__, 86, 51), ggev_14502, *[a1_14503, b1_14504, cvl_14505, cvr_14506, lwork_14507, overwrite_a_14508, overwrite_b_14509], **kwargs_14510)
    
    # Obtaining the member '__getitem__' of a type (line 86)
    getitem___14512 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 86, 8), ggev_call_result_14511, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 86)
    subscript_call_result_14513 = invoke(stypy.reporting.localization.Localization(__file__, 86, 8), getitem___14512, int_14501)
    
    # Assigning a type to the variable 'tuple_var_assignment_14036' (line 86)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 86, 8), 'tuple_var_assignment_14036', subscript_call_result_14513)
    
    # Assigning a Subscript to a Name (line 86):
    
    # Obtaining the type of the subscript
    int_14514 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 86, 8), 'int')
    
    # Call to ggev(...): (line 86)
    # Processing the call arguments (line 86)
    # Getting the type of 'a1' (line 86)
    a1_14516 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 56), 'a1', False)
    # Getting the type of 'b1' (line 86)
    b1_14517 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 60), 'b1', False)
    # Getting the type of 'cvl' (line 86)
    cvl_14518 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 64), 'cvl', False)
    # Getting the type of 'cvr' (line 86)
    cvr_14519 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 69), 'cvr', False)
    # Getting the type of 'lwork' (line 87)
    lwork_14520 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 56), 'lwork', False)
    # Getting the type of 'overwrite_a' (line 87)
    overwrite_a_14521 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 63), 'overwrite_a', False)
    # Getting the type of 'overwrite_b' (line 88)
    overwrite_b_14522 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 56), 'overwrite_b', False)
    # Processing the call keyword arguments (line 86)
    kwargs_14523 = {}
    # Getting the type of 'ggev' (line 86)
    ggev_14515 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 51), 'ggev', False)
    # Calling ggev(args, kwargs) (line 86)
    ggev_call_result_14524 = invoke(stypy.reporting.localization.Localization(__file__, 86, 51), ggev_14515, *[a1_14516, b1_14517, cvl_14518, cvr_14519, lwork_14520, overwrite_a_14521, overwrite_b_14522], **kwargs_14523)
    
    # Obtaining the member '__getitem__' of a type (line 86)
    getitem___14525 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 86, 8), ggev_call_result_14524, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 86)
    subscript_call_result_14526 = invoke(stypy.reporting.localization.Localization(__file__, 86, 8), getitem___14525, int_14514)
    
    # Assigning a type to the variable 'tuple_var_assignment_14037' (line 86)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 86, 8), 'tuple_var_assignment_14037', subscript_call_result_14526)
    
    # Assigning a Subscript to a Name (line 86):
    
    # Obtaining the type of the subscript
    int_14527 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 86, 8), 'int')
    
    # Call to ggev(...): (line 86)
    # Processing the call arguments (line 86)
    # Getting the type of 'a1' (line 86)
    a1_14529 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 56), 'a1', False)
    # Getting the type of 'b1' (line 86)
    b1_14530 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 60), 'b1', False)
    # Getting the type of 'cvl' (line 86)
    cvl_14531 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 64), 'cvl', False)
    # Getting the type of 'cvr' (line 86)
    cvr_14532 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 69), 'cvr', False)
    # Getting the type of 'lwork' (line 87)
    lwork_14533 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 56), 'lwork', False)
    # Getting the type of 'overwrite_a' (line 87)
    overwrite_a_14534 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 63), 'overwrite_a', False)
    # Getting the type of 'overwrite_b' (line 88)
    overwrite_b_14535 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 56), 'overwrite_b', False)
    # Processing the call keyword arguments (line 86)
    kwargs_14536 = {}
    # Getting the type of 'ggev' (line 86)
    ggev_14528 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 51), 'ggev', False)
    # Calling ggev(args, kwargs) (line 86)
    ggev_call_result_14537 = invoke(stypy.reporting.localization.Localization(__file__, 86, 51), ggev_14528, *[a1_14529, b1_14530, cvl_14531, cvr_14532, lwork_14533, overwrite_a_14534, overwrite_b_14535], **kwargs_14536)
    
    # Obtaining the member '__getitem__' of a type (line 86)
    getitem___14538 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 86, 8), ggev_call_result_14537, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 86)
    subscript_call_result_14539 = invoke(stypy.reporting.localization.Localization(__file__, 86, 8), getitem___14538, int_14527)
    
    # Assigning a type to the variable 'tuple_var_assignment_14038' (line 86)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 86, 8), 'tuple_var_assignment_14038', subscript_call_result_14539)
    
    # Assigning a Subscript to a Name (line 86):
    
    # Obtaining the type of the subscript
    int_14540 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 86, 8), 'int')
    
    # Call to ggev(...): (line 86)
    # Processing the call arguments (line 86)
    # Getting the type of 'a1' (line 86)
    a1_14542 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 56), 'a1', False)
    # Getting the type of 'b1' (line 86)
    b1_14543 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 60), 'b1', False)
    # Getting the type of 'cvl' (line 86)
    cvl_14544 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 64), 'cvl', False)
    # Getting the type of 'cvr' (line 86)
    cvr_14545 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 69), 'cvr', False)
    # Getting the type of 'lwork' (line 87)
    lwork_14546 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 56), 'lwork', False)
    # Getting the type of 'overwrite_a' (line 87)
    overwrite_a_14547 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 63), 'overwrite_a', False)
    # Getting the type of 'overwrite_b' (line 88)
    overwrite_b_14548 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 56), 'overwrite_b', False)
    # Processing the call keyword arguments (line 86)
    kwargs_14549 = {}
    # Getting the type of 'ggev' (line 86)
    ggev_14541 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 51), 'ggev', False)
    # Calling ggev(args, kwargs) (line 86)
    ggev_call_result_14550 = invoke(stypy.reporting.localization.Localization(__file__, 86, 51), ggev_14541, *[a1_14542, b1_14543, cvl_14544, cvr_14545, lwork_14546, overwrite_a_14547, overwrite_b_14548], **kwargs_14549)
    
    # Obtaining the member '__getitem__' of a type (line 86)
    getitem___14551 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 86, 8), ggev_call_result_14550, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 86)
    subscript_call_result_14552 = invoke(stypy.reporting.localization.Localization(__file__, 86, 8), getitem___14551, int_14540)
    
    # Assigning a type to the variable 'tuple_var_assignment_14039' (line 86)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 86, 8), 'tuple_var_assignment_14039', subscript_call_result_14552)
    
    # Assigning a Name to a Name (line 86):
    # Getting the type of 'tuple_var_assignment_14033' (line 86)
    tuple_var_assignment_14033_14553 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 8), 'tuple_var_assignment_14033')
    # Assigning a type to the variable 'alphar' (line 86)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 86, 8), 'alphar', tuple_var_assignment_14033_14553)
    
    # Assigning a Name to a Name (line 86):
    # Getting the type of 'tuple_var_assignment_14034' (line 86)
    tuple_var_assignment_14034_14554 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 8), 'tuple_var_assignment_14034')
    # Assigning a type to the variable 'alphai' (line 86)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 86, 16), 'alphai', tuple_var_assignment_14034_14554)
    
    # Assigning a Name to a Name (line 86):
    # Getting the type of 'tuple_var_assignment_14035' (line 86)
    tuple_var_assignment_14035_14555 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 8), 'tuple_var_assignment_14035')
    # Assigning a type to the variable 'beta' (line 86)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 86, 24), 'beta', tuple_var_assignment_14035_14555)
    
    # Assigning a Name to a Name (line 86):
    # Getting the type of 'tuple_var_assignment_14036' (line 86)
    tuple_var_assignment_14036_14556 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 8), 'tuple_var_assignment_14036')
    # Assigning a type to the variable 'vl' (line 86)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 86, 30), 'vl', tuple_var_assignment_14036_14556)
    
    # Assigning a Name to a Name (line 86):
    # Getting the type of 'tuple_var_assignment_14037' (line 86)
    tuple_var_assignment_14037_14557 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 8), 'tuple_var_assignment_14037')
    # Assigning a type to the variable 'vr' (line 86)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 86, 34), 'vr', tuple_var_assignment_14037_14557)
    
    # Assigning a Name to a Name (line 86):
    # Getting the type of 'tuple_var_assignment_14038' (line 86)
    tuple_var_assignment_14038_14558 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 8), 'tuple_var_assignment_14038')
    # Assigning a type to the variable 'work' (line 86)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 86, 38), 'work', tuple_var_assignment_14038_14558)
    
    # Assigning a Name to a Name (line 86):
    # Getting the type of 'tuple_var_assignment_14039' (line 86)
    tuple_var_assignment_14039_14559 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 8), 'tuple_var_assignment_14039')
    # Assigning a type to the variable 'info' (line 86)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 86, 44), 'info', tuple_var_assignment_14039_14559)
    
    # Assigning a BinOp to a Name (line 89):
    
    # Assigning a BinOp to a Name (line 89):
    # Getting the type of 'alphar' (line 89)
    alphar_14560 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 16), 'alphar')
    # Getting the type of '_I' (line 89)
    _I_14561 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 25), '_I')
    # Getting the type of 'alphai' (line 89)
    alphai_14562 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 30), 'alphai')
    # Applying the binary operator '*' (line 89)
    result_mul_14563 = python_operator(stypy.reporting.localization.Localization(__file__, 89, 25), '*', _I_14561, alphai_14562)
    
    # Applying the binary operator '+' (line 89)
    result_add_14564 = python_operator(stypy.reporting.localization.Localization(__file__, 89, 16), '+', alphar_14560, result_mul_14563)
    
    # Assigning a type to the variable 'alpha' (line 89)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 89, 8), 'alpha', result_add_14564)
    
    # Assigning a Call to a Name (line 90):
    
    # Assigning a Call to a Name (line 90):
    
    # Call to _make_eigvals(...): (line 90)
    # Processing the call arguments (line 90)
    # Getting the type of 'alpha' (line 90)
    alpha_14566 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 26), 'alpha', False)
    # Getting the type of 'beta' (line 90)
    beta_14567 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 33), 'beta', False)
    # Getting the type of 'homogeneous_eigvals' (line 90)
    homogeneous_eigvals_14568 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 39), 'homogeneous_eigvals', False)
    # Processing the call keyword arguments (line 90)
    kwargs_14569 = {}
    # Getting the type of '_make_eigvals' (line 90)
    _make_eigvals_14565 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 12), '_make_eigvals', False)
    # Calling _make_eigvals(args, kwargs) (line 90)
    _make_eigvals_call_result_14570 = invoke(stypy.reporting.localization.Localization(__file__, 90, 12), _make_eigvals_14565, *[alpha_14566, beta_14567, homogeneous_eigvals_14568], **kwargs_14569)
    
    # Assigning a type to the variable 'w' (line 90)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 90, 8), 'w', _make_eigvals_call_result_14570)
    # SSA join for if statement (line 81)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to _check_info(...): (line 91)
    # Processing the call arguments (line 91)
    # Getting the type of 'info' (line 91)
    info_14572 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 16), 'info', False)
    str_14573 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 91, 22), 'str', 'generalized eig algorithm (ggev)')
    # Processing the call keyword arguments (line 91)
    kwargs_14574 = {}
    # Getting the type of '_check_info' (line 91)
    _check_info_14571 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 4), '_check_info', False)
    # Calling _check_info(args, kwargs) (line 91)
    _check_info_call_result_14575 = invoke(stypy.reporting.localization.Localization(__file__, 91, 4), _check_info_14571, *[info_14572, str_14573], **kwargs_14574)
    
    
    # Assigning a Call to a Name (line 93):
    
    # Assigning a Call to a Name (line 93):
    
    # Call to all(...): (line 93)
    # Processing the call arguments (line 93)
    
    # Getting the type of 'w' (line 93)
    w_14578 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 26), 'w', False)
    # Obtaining the member 'imag' of a type (line 93)
    imag_14579 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 93, 26), w_14578, 'imag')
    float_14580 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 93, 36), 'float')
    # Applying the binary operator '==' (line 93)
    result_eq_14581 = python_operator(stypy.reporting.localization.Localization(__file__, 93, 26), '==', imag_14579, float_14580)
    
    # Processing the call keyword arguments (line 93)
    kwargs_14582 = {}
    # Getting the type of 'numpy' (line 93)
    numpy_14576 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 16), 'numpy', False)
    # Obtaining the member 'all' of a type (line 93)
    all_14577 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 93, 16), numpy_14576, 'all')
    # Calling all(args, kwargs) (line 93)
    all_call_result_14583 = invoke(stypy.reporting.localization.Localization(__file__, 93, 16), all_14577, *[result_eq_14581], **kwargs_14582)
    
    # Assigning a type to the variable 'only_real' (line 93)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 93, 4), 'only_real', all_call_result_14583)
    
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'ggev' (line 94)
    ggev_14584 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 12), 'ggev')
    # Obtaining the member 'typecode' of a type (line 94)
    typecode_14585 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 94, 12), ggev_14584, 'typecode')
    str_14586 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 94, 29), 'str', 'cz')
    # Applying the binary operator 'in' (line 94)
    result_contains_14587 = python_operator(stypy.reporting.localization.Localization(__file__, 94, 12), 'in', typecode_14585, str_14586)
    
    # Getting the type of 'only_real' (line 94)
    only_real_14588 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 37), 'only_real')
    # Applying the binary operator 'or' (line 94)
    result_or_keyword_14589 = python_operator(stypy.reporting.localization.Localization(__file__, 94, 12), 'or', result_contains_14587, only_real_14588)
    
    # Applying the 'not' unary operator (line 94)
    result_not__14590 = python_operator(stypy.reporting.localization.Localization(__file__, 94, 7), 'not', result_or_keyword_14589)
    
    # Testing the type of an if condition (line 94)
    if_condition_14591 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 94, 4), result_not__14590)
    # Assigning a type to the variable 'if_condition_14591' (line 94)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 94, 4), 'if_condition_14591', if_condition_14591)
    # SSA begins for if statement (line 94)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Attribute to a Name (line 95):
    
    # Assigning a Attribute to a Name (line 95):
    # Getting the type of 'w' (line 95)
    w_14592 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 12), 'w')
    # Obtaining the member 'dtype' of a type (line 95)
    dtype_14593 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 95, 12), w_14592, 'dtype')
    # Obtaining the member 'char' of a type (line 95)
    char_14594 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 95, 12), dtype_14593, 'char')
    # Assigning a type to the variable 't' (line 95)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 95, 8), 't', char_14594)
    
    # Getting the type of 'left' (line 96)
    left_14595 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 11), 'left')
    # Testing the type of an if condition (line 96)
    if_condition_14596 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 96, 8), left_14595)
    # Assigning a type to the variable 'if_condition_14596' (line 96)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 96, 8), 'if_condition_14596', if_condition_14596)
    # SSA begins for if statement (line 96)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 97):
    
    # Assigning a Call to a Name (line 97):
    
    # Call to _make_complex_eigvecs(...): (line 97)
    # Processing the call arguments (line 97)
    # Getting the type of 'w' (line 97)
    w_14598 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 39), 'w', False)
    # Getting the type of 'vl' (line 97)
    vl_14599 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 42), 'vl', False)
    # Getting the type of 't' (line 97)
    t_14600 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 46), 't', False)
    # Processing the call keyword arguments (line 97)
    kwargs_14601 = {}
    # Getting the type of '_make_complex_eigvecs' (line 97)
    _make_complex_eigvecs_14597 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 17), '_make_complex_eigvecs', False)
    # Calling _make_complex_eigvecs(args, kwargs) (line 97)
    _make_complex_eigvecs_call_result_14602 = invoke(stypy.reporting.localization.Localization(__file__, 97, 17), _make_complex_eigvecs_14597, *[w_14598, vl_14599, t_14600], **kwargs_14601)
    
    # Assigning a type to the variable 'vl' (line 97)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 97, 12), 'vl', _make_complex_eigvecs_call_result_14602)
    # SSA join for if statement (line 96)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Getting the type of 'right' (line 98)
    right_14603 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 11), 'right')
    # Testing the type of an if condition (line 98)
    if_condition_14604 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 98, 8), right_14603)
    # Assigning a type to the variable 'if_condition_14604' (line 98)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 98, 8), 'if_condition_14604', if_condition_14604)
    # SSA begins for if statement (line 98)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 99):
    
    # Assigning a Call to a Name (line 99):
    
    # Call to _make_complex_eigvecs(...): (line 99)
    # Processing the call arguments (line 99)
    # Getting the type of 'w' (line 99)
    w_14606 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 39), 'w', False)
    # Getting the type of 'vr' (line 99)
    vr_14607 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 42), 'vr', False)
    # Getting the type of 't' (line 99)
    t_14608 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 46), 't', False)
    # Processing the call keyword arguments (line 99)
    kwargs_14609 = {}
    # Getting the type of '_make_complex_eigvecs' (line 99)
    _make_complex_eigvecs_14605 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 17), '_make_complex_eigvecs', False)
    # Calling _make_complex_eigvecs(args, kwargs) (line 99)
    _make_complex_eigvecs_call_result_14610 = invoke(stypy.reporting.localization.Localization(__file__, 99, 17), _make_complex_eigvecs_14605, *[w_14606, vr_14607, t_14608], **kwargs_14609)
    
    # Assigning a type to the variable 'vr' (line 99)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 99, 12), 'vr', _make_complex_eigvecs_call_result_14610)
    # SSA join for if statement (line 98)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 94)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Call to xrange(...): (line 102)
    # Processing the call arguments (line 102)
    
    # Obtaining the type of the subscript
    int_14612 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 102, 29), 'int')
    # Getting the type of 'vr' (line 102)
    vr_14613 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 20), 'vr', False)
    # Obtaining the member 'shape' of a type (line 102)
    shape_14614 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 102, 20), vr_14613, 'shape')
    # Obtaining the member '__getitem__' of a type (line 102)
    getitem___14615 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 102, 20), shape_14614, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 102)
    subscript_call_result_14616 = invoke(stypy.reporting.localization.Localization(__file__, 102, 20), getitem___14615, int_14612)
    
    # Processing the call keyword arguments (line 102)
    kwargs_14617 = {}
    # Getting the type of 'xrange' (line 102)
    xrange_14611 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 13), 'xrange', False)
    # Calling xrange(args, kwargs) (line 102)
    xrange_call_result_14618 = invoke(stypy.reporting.localization.Localization(__file__, 102, 13), xrange_14611, *[subscript_call_result_14616], **kwargs_14617)
    
    # Testing the type of a for loop iterable (line 102)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 102, 4), xrange_call_result_14618)
    # Getting the type of the for loop variable (line 102)
    for_loop_var_14619 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 102, 4), xrange_call_result_14618)
    # Assigning a type to the variable 'i' (line 102)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 102, 4), 'i', for_loop_var_14619)
    # SSA begins for a for statement (line 102)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Getting the type of 'right' (line 103)
    right_14620 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 11), 'right')
    # Testing the type of an if condition (line 103)
    if_condition_14621 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 103, 8), right_14620)
    # Assigning a type to the variable 'if_condition_14621' (line 103)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 103, 8), 'if_condition_14621', if_condition_14621)
    # SSA begins for if statement (line 103)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Getting the type of 'vr' (line 104)
    vr_14622 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 12), 'vr')
    
    # Obtaining the type of the subscript
    slice_14623 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 104, 12), None, None, None)
    # Getting the type of 'i' (line 104)
    i_14624 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 18), 'i')
    # Getting the type of 'vr' (line 104)
    vr_14625 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 12), 'vr')
    # Obtaining the member '__getitem__' of a type (line 104)
    getitem___14626 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 104, 12), vr_14625, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 104)
    subscript_call_result_14627 = invoke(stypy.reporting.localization.Localization(__file__, 104, 12), getitem___14626, (slice_14623, i_14624))
    
    
    # Call to norm(...): (line 104)
    # Processing the call arguments (line 104)
    
    # Obtaining the type of the subscript
    slice_14629 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 104, 29), None, None, None)
    # Getting the type of 'i' (line 104)
    i_14630 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 35), 'i', False)
    # Getting the type of 'vr' (line 104)
    vr_14631 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 29), 'vr', False)
    # Obtaining the member '__getitem__' of a type (line 104)
    getitem___14632 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 104, 29), vr_14631, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 104)
    subscript_call_result_14633 = invoke(stypy.reporting.localization.Localization(__file__, 104, 29), getitem___14632, (slice_14629, i_14630))
    
    # Processing the call keyword arguments (line 104)
    kwargs_14634 = {}
    # Getting the type of 'norm' (line 104)
    norm_14628 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 24), 'norm', False)
    # Calling norm(args, kwargs) (line 104)
    norm_call_result_14635 = invoke(stypy.reporting.localization.Localization(__file__, 104, 24), norm_14628, *[subscript_call_result_14633], **kwargs_14634)
    
    # Applying the binary operator 'div=' (line 104)
    result_div_14636 = python_operator(stypy.reporting.localization.Localization(__file__, 104, 12), 'div=', subscript_call_result_14627, norm_call_result_14635)
    # Getting the type of 'vr' (line 104)
    vr_14637 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 12), 'vr')
    slice_14638 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 104, 12), None, None, None)
    # Getting the type of 'i' (line 104)
    i_14639 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 18), 'i')
    # Storing an element on a container (line 104)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 104, 12), vr_14637, ((slice_14638, i_14639), result_div_14636))
    
    # SSA join for if statement (line 103)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Getting the type of 'left' (line 105)
    left_14640 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 11), 'left')
    # Testing the type of an if condition (line 105)
    if_condition_14641 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 105, 8), left_14640)
    # Assigning a type to the variable 'if_condition_14641' (line 105)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 105, 8), 'if_condition_14641', if_condition_14641)
    # SSA begins for if statement (line 105)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Getting the type of 'vl' (line 106)
    vl_14642 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 12), 'vl')
    
    # Obtaining the type of the subscript
    slice_14643 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 106, 12), None, None, None)
    # Getting the type of 'i' (line 106)
    i_14644 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 18), 'i')
    # Getting the type of 'vl' (line 106)
    vl_14645 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 12), 'vl')
    # Obtaining the member '__getitem__' of a type (line 106)
    getitem___14646 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 106, 12), vl_14645, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 106)
    subscript_call_result_14647 = invoke(stypy.reporting.localization.Localization(__file__, 106, 12), getitem___14646, (slice_14643, i_14644))
    
    
    # Call to norm(...): (line 106)
    # Processing the call arguments (line 106)
    
    # Obtaining the type of the subscript
    slice_14649 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 106, 29), None, None, None)
    # Getting the type of 'i' (line 106)
    i_14650 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 35), 'i', False)
    # Getting the type of 'vl' (line 106)
    vl_14651 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 29), 'vl', False)
    # Obtaining the member '__getitem__' of a type (line 106)
    getitem___14652 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 106, 29), vl_14651, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 106)
    subscript_call_result_14653 = invoke(stypy.reporting.localization.Localization(__file__, 106, 29), getitem___14652, (slice_14649, i_14650))
    
    # Processing the call keyword arguments (line 106)
    kwargs_14654 = {}
    # Getting the type of 'norm' (line 106)
    norm_14648 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 24), 'norm', False)
    # Calling norm(args, kwargs) (line 106)
    norm_call_result_14655 = invoke(stypy.reporting.localization.Localization(__file__, 106, 24), norm_14648, *[subscript_call_result_14653], **kwargs_14654)
    
    # Applying the binary operator 'div=' (line 106)
    result_div_14656 = python_operator(stypy.reporting.localization.Localization(__file__, 106, 12), 'div=', subscript_call_result_14647, norm_call_result_14655)
    # Getting the type of 'vl' (line 106)
    vl_14657 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 12), 'vl')
    slice_14658 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 106, 12), None, None, None)
    # Getting the type of 'i' (line 106)
    i_14659 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 18), 'i')
    # Storing an element on a container (line 106)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 106, 12), vl_14657, ((slice_14658, i_14659), result_div_14656))
    
    # SSA join for if statement (line 105)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    
    # Evaluating a boolean operation
    # Getting the type of 'left' (line 108)
    left_14660 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 12), 'left')
    # Getting the type of 'right' (line 108)
    right_14661 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 20), 'right')
    # Applying the binary operator 'or' (line 108)
    result_or_keyword_14662 = python_operator(stypy.reporting.localization.Localization(__file__, 108, 12), 'or', left_14660, right_14661)
    
    # Applying the 'not' unary operator (line 108)
    result_not__14663 = python_operator(stypy.reporting.localization.Localization(__file__, 108, 7), 'not', result_or_keyword_14662)
    
    # Testing the type of an if condition (line 108)
    if_condition_14664 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 108, 4), result_not__14663)
    # Assigning a type to the variable 'if_condition_14664' (line 108)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 108, 4), 'if_condition_14664', if_condition_14664)
    # SSA begins for if statement (line 108)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # Getting the type of 'w' (line 109)
    w_14665 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 15), 'w')
    # Assigning a type to the variable 'stypy_return_type' (line 109)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 109, 8), 'stypy_return_type', w_14665)
    # SSA join for if statement (line 108)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Getting the type of 'left' (line 110)
    left_14666 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 7), 'left')
    # Testing the type of an if condition (line 110)
    if_condition_14667 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 110, 4), left_14666)
    # Assigning a type to the variable 'if_condition_14667' (line 110)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 110, 4), 'if_condition_14667', if_condition_14667)
    # SSA begins for if statement (line 110)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Getting the type of 'right' (line 111)
    right_14668 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 11), 'right')
    # Testing the type of an if condition (line 111)
    if_condition_14669 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 111, 8), right_14668)
    # Assigning a type to the variable 'if_condition_14669' (line 111)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 111, 8), 'if_condition_14669', if_condition_14669)
    # SSA begins for if statement (line 111)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Obtaining an instance of the builtin type 'tuple' (line 112)
    tuple_14670 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 112, 19), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 112)
    # Adding element type (line 112)
    # Getting the type of 'w' (line 112)
    w_14671 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 19), 'w')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 112, 19), tuple_14670, w_14671)
    # Adding element type (line 112)
    # Getting the type of 'vl' (line 112)
    vl_14672 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 22), 'vl')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 112, 19), tuple_14670, vl_14672)
    # Adding element type (line 112)
    # Getting the type of 'vr' (line 112)
    vr_14673 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 26), 'vr')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 112, 19), tuple_14670, vr_14673)
    
    # Assigning a type to the variable 'stypy_return_type' (line 112)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 112, 12), 'stypy_return_type', tuple_14670)
    # SSA join for if statement (line 111)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Obtaining an instance of the builtin type 'tuple' (line 113)
    tuple_14674 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 113, 15), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 113)
    # Adding element type (line 113)
    # Getting the type of 'w' (line 113)
    w_14675 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 15), 'w')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 113, 15), tuple_14674, w_14675)
    # Adding element type (line 113)
    # Getting the type of 'vl' (line 113)
    vl_14676 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 18), 'vl')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 113, 15), tuple_14674, vl_14676)
    
    # Assigning a type to the variable 'stypy_return_type' (line 113)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 113, 8), 'stypy_return_type', tuple_14674)
    # SSA join for if statement (line 110)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Obtaining an instance of the builtin type 'tuple' (line 114)
    tuple_14677 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 114, 11), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 114)
    # Adding element type (line 114)
    # Getting the type of 'w' (line 114)
    w_14678 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 11), 'w')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 114, 11), tuple_14677, w_14678)
    # Adding element type (line 114)
    # Getting the type of 'vr' (line 114)
    vr_14679 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 14), 'vr')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 114, 11), tuple_14677, vr_14679)
    
    # Assigning a type to the variable 'stypy_return_type' (line 114)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 114, 4), 'stypy_return_type', tuple_14677)
    
    # ################# End of '_geneig(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_geneig' in the type store
    # Getting the type of 'stypy_return_type' (line 75)
    stypy_return_type_14680 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_14680)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_geneig'
    return stypy_return_type_14680

# Assigning a type to the variable '_geneig' (line 75)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 75, 0), '_geneig', _geneig)

@norecursion
def eig(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'None' (line 117)
    None_14681 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 13), 'None')
    # Getting the type of 'False' (line 117)
    False_14682 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 24), 'False')
    # Getting the type of 'True' (line 117)
    True_14683 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 37), 'True')
    # Getting the type of 'False' (line 117)
    False_14684 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 55), 'False')
    # Getting the type of 'False' (line 118)
    False_14685 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 20), 'False')
    # Getting the type of 'True' (line 118)
    True_14686 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 40), 'True')
    # Getting the type of 'False' (line 118)
    False_14687 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 66), 'False')
    defaults = [None_14681, False_14682, True_14683, False_14684, False_14685, True_14686, False_14687]
    # Create a new context for function 'eig'
    module_type_store = module_type_store.open_function_context('eig', 117, 0, False)
    
    # Passed parameters checking function
    eig.stypy_localization = localization
    eig.stypy_type_of_self = None
    eig.stypy_type_store = module_type_store
    eig.stypy_function_name = 'eig'
    eig.stypy_param_names_list = ['a', 'b', 'left', 'right', 'overwrite_a', 'overwrite_b', 'check_finite', 'homogeneous_eigvals']
    eig.stypy_varargs_param_name = None
    eig.stypy_kwargs_param_name = None
    eig.stypy_call_defaults = defaults
    eig.stypy_call_varargs = varargs
    eig.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'eig', ['a', 'b', 'left', 'right', 'overwrite_a', 'overwrite_b', 'check_finite', 'homogeneous_eigvals'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'eig', localization, ['a', 'b', 'left', 'right', 'overwrite_a', 'overwrite_b', 'check_finite', 'homogeneous_eigvals'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'eig(...)' code ##################

    str_14688 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 183, (-1)), 'str', '\n    Solve an ordinary or generalized eigenvalue problem of a square matrix.\n\n    Find eigenvalues w and right or left eigenvectors of a general matrix::\n\n        a   vr[:,i] = w[i]        b   vr[:,i]\n        a.H vl[:,i] = w[i].conj() b.H vl[:,i]\n\n    where ``.H`` is the Hermitian conjugation.\n\n    Parameters\n    ----------\n    a : (M, M) array_like\n        A complex or real matrix whose eigenvalues and eigenvectors\n        will be computed.\n    b : (M, M) array_like, optional\n        Right-hand side matrix in a generalized eigenvalue problem.\n        Default is None, identity matrix is assumed.\n    left : bool, optional\n        Whether to calculate and return left eigenvectors.  Default is False.\n    right : bool, optional\n        Whether to calculate and return right eigenvectors.  Default is True.\n    overwrite_a : bool, optional\n        Whether to overwrite `a`; may improve performance.  Default is False.\n    overwrite_b : bool, optional\n        Whether to overwrite `b`; may improve performance.  Default is False.\n    check_finite : bool, optional\n        Whether to check that the input matrices contain only finite numbers.\n        Disabling may give a performance gain, but may result in problems\n        (crashes, non-termination) if the inputs do contain infinities or NaNs.\n    homogeneous_eigvals : bool, optional\n        If True, return the eigenvalues in homogeneous coordinates.\n        In this case ``w`` is a (2, M) array so that::\n\n            w[1,i] a vr[:,i] = w[0,i] b vr[:,i]\n\n        Default is False.\n\n    Returns\n    -------\n    w : (M,) or (2, M) double or complex ndarray\n        The eigenvalues, each repeated according to its\n        multiplicity. The shape is (M,) unless\n        ``homogeneous_eigvals=True``.\n    vl : (M, M) double or complex ndarray\n        The normalized left eigenvector corresponding to the eigenvalue\n        ``w[i]`` is the column vl[:,i]. Only returned if ``left=True``.\n    vr : (M, M) double or complex ndarray\n        The normalized right eigenvector corresponding to the eigenvalue\n        ``w[i]`` is the column ``vr[:,i]``.  Only returned if ``right=True``.\n\n    Raises\n    ------\n    LinAlgError\n        If eigenvalue computation does not converge.\n\n    See Also\n    --------\n    eigvals : eigenvalues of general arrays\n    eigh : Eigenvalues and right eigenvectors for symmetric/Hermitian arrays.\n    eig_banded : eigenvalues and right eigenvectors for symmetric/Hermitian\n        band matrices\n    eigh_tridiagonal : eigenvalues and right eiegenvectors for\n        symmetric/Hermitian tridiagonal matrices\n    ')
    
    # Assigning a Call to a Name (line 184):
    
    # Assigning a Call to a Name (line 184):
    
    # Call to _asarray_validated(...): (line 184)
    # Processing the call arguments (line 184)
    # Getting the type of 'a' (line 184)
    a_14690 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 28), 'a', False)
    # Processing the call keyword arguments (line 184)
    # Getting the type of 'check_finite' (line 184)
    check_finite_14691 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 44), 'check_finite', False)
    keyword_14692 = check_finite_14691
    kwargs_14693 = {'check_finite': keyword_14692}
    # Getting the type of '_asarray_validated' (line 184)
    _asarray_validated_14689 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 9), '_asarray_validated', False)
    # Calling _asarray_validated(args, kwargs) (line 184)
    _asarray_validated_call_result_14694 = invoke(stypy.reporting.localization.Localization(__file__, 184, 9), _asarray_validated_14689, *[a_14690], **kwargs_14693)
    
    # Assigning a type to the variable 'a1' (line 184)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 184, 4), 'a1', _asarray_validated_call_result_14694)
    
    
    # Evaluating a boolean operation
    
    
    # Call to len(...): (line 185)
    # Processing the call arguments (line 185)
    # Getting the type of 'a1' (line 185)
    a1_14696 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 11), 'a1', False)
    # Obtaining the member 'shape' of a type (line 185)
    shape_14697 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 185, 11), a1_14696, 'shape')
    # Processing the call keyword arguments (line 185)
    kwargs_14698 = {}
    # Getting the type of 'len' (line 185)
    len_14695 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 7), 'len', False)
    # Calling len(args, kwargs) (line 185)
    len_call_result_14699 = invoke(stypy.reporting.localization.Localization(__file__, 185, 7), len_14695, *[shape_14697], **kwargs_14698)
    
    int_14700 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 185, 24), 'int')
    # Applying the binary operator '!=' (line 185)
    result_ne_14701 = python_operator(stypy.reporting.localization.Localization(__file__, 185, 7), '!=', len_call_result_14699, int_14700)
    
    
    
    # Obtaining the type of the subscript
    int_14702 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 185, 38), 'int')
    # Getting the type of 'a1' (line 185)
    a1_14703 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 29), 'a1')
    # Obtaining the member 'shape' of a type (line 185)
    shape_14704 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 185, 29), a1_14703, 'shape')
    # Obtaining the member '__getitem__' of a type (line 185)
    getitem___14705 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 185, 29), shape_14704, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 185)
    subscript_call_result_14706 = invoke(stypy.reporting.localization.Localization(__file__, 185, 29), getitem___14705, int_14702)
    
    
    # Obtaining the type of the subscript
    int_14707 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 185, 53), 'int')
    # Getting the type of 'a1' (line 185)
    a1_14708 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 44), 'a1')
    # Obtaining the member 'shape' of a type (line 185)
    shape_14709 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 185, 44), a1_14708, 'shape')
    # Obtaining the member '__getitem__' of a type (line 185)
    getitem___14710 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 185, 44), shape_14709, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 185)
    subscript_call_result_14711 = invoke(stypy.reporting.localization.Localization(__file__, 185, 44), getitem___14710, int_14707)
    
    # Applying the binary operator '!=' (line 185)
    result_ne_14712 = python_operator(stypy.reporting.localization.Localization(__file__, 185, 29), '!=', subscript_call_result_14706, subscript_call_result_14711)
    
    # Applying the binary operator 'or' (line 185)
    result_or_keyword_14713 = python_operator(stypy.reporting.localization.Localization(__file__, 185, 7), 'or', result_ne_14701, result_ne_14712)
    
    # Testing the type of an if condition (line 185)
    if_condition_14714 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 185, 4), result_or_keyword_14713)
    # Assigning a type to the variable 'if_condition_14714' (line 185)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 185, 4), 'if_condition_14714', if_condition_14714)
    # SSA begins for if statement (line 185)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 186)
    # Processing the call arguments (line 186)
    str_14716 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 186, 25), 'str', 'expected square matrix')
    # Processing the call keyword arguments (line 186)
    kwargs_14717 = {}
    # Getting the type of 'ValueError' (line 186)
    ValueError_14715 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 186)
    ValueError_call_result_14718 = invoke(stypy.reporting.localization.Localization(__file__, 186, 14), ValueError_14715, *[str_14716], **kwargs_14717)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 186, 8), ValueError_call_result_14718, 'raise parameter', BaseException)
    # SSA join for if statement (line 185)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a BoolOp to a Name (line 187):
    
    # Assigning a BoolOp to a Name (line 187):
    
    # Evaluating a boolean operation
    # Getting the type of 'overwrite_a' (line 187)
    overwrite_a_14719 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 18), 'overwrite_a')
    
    # Call to _datacopied(...): (line 187)
    # Processing the call arguments (line 187)
    # Getting the type of 'a1' (line 187)
    a1_14721 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 46), 'a1', False)
    # Getting the type of 'a' (line 187)
    a_14722 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 50), 'a', False)
    # Processing the call keyword arguments (line 187)
    kwargs_14723 = {}
    # Getting the type of '_datacopied' (line 187)
    _datacopied_14720 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 34), '_datacopied', False)
    # Calling _datacopied(args, kwargs) (line 187)
    _datacopied_call_result_14724 = invoke(stypy.reporting.localization.Localization(__file__, 187, 34), _datacopied_14720, *[a1_14721, a_14722], **kwargs_14723)
    
    # Applying the binary operator 'or' (line 187)
    result_or_keyword_14725 = python_operator(stypy.reporting.localization.Localization(__file__, 187, 18), 'or', overwrite_a_14719, _datacopied_call_result_14724)
    
    # Assigning a type to the variable 'overwrite_a' (line 187)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 187, 4), 'overwrite_a', result_or_keyword_14725)
    
    # Type idiom detected: calculating its left and rigth part (line 188)
    # Getting the type of 'b' (line 188)
    b_14726 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 188, 4), 'b')
    # Getting the type of 'None' (line 188)
    None_14727 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 188, 16), 'None')
    
    (may_be_14728, more_types_in_union_14729) = may_not_be_none(b_14726, None_14727)

    if may_be_14728:

        if more_types_in_union_14729:
            # Runtime conditional SSA (line 188)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a Call to a Name (line 189):
        
        # Assigning a Call to a Name (line 189):
        
        # Call to _asarray_validated(...): (line 189)
        # Processing the call arguments (line 189)
        # Getting the type of 'b' (line 189)
        b_14731 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 32), 'b', False)
        # Processing the call keyword arguments (line 189)
        # Getting the type of 'check_finite' (line 189)
        check_finite_14732 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 48), 'check_finite', False)
        keyword_14733 = check_finite_14732
        kwargs_14734 = {'check_finite': keyword_14733}
        # Getting the type of '_asarray_validated' (line 189)
        _asarray_validated_14730 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 13), '_asarray_validated', False)
        # Calling _asarray_validated(args, kwargs) (line 189)
        _asarray_validated_call_result_14735 = invoke(stypy.reporting.localization.Localization(__file__, 189, 13), _asarray_validated_14730, *[b_14731], **kwargs_14734)
        
        # Assigning a type to the variable 'b1' (line 189)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 189, 8), 'b1', _asarray_validated_call_result_14735)
        
        # Assigning a BoolOp to a Name (line 190):
        
        # Assigning a BoolOp to a Name (line 190):
        
        # Evaluating a boolean operation
        # Getting the type of 'overwrite_b' (line 190)
        overwrite_b_14736 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 190, 22), 'overwrite_b')
        
        # Call to _datacopied(...): (line 190)
        # Processing the call arguments (line 190)
        # Getting the type of 'b1' (line 190)
        b1_14738 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 190, 49), 'b1', False)
        # Getting the type of 'b' (line 190)
        b_14739 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 190, 53), 'b', False)
        # Processing the call keyword arguments (line 190)
        kwargs_14740 = {}
        # Getting the type of '_datacopied' (line 190)
        _datacopied_14737 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 190, 37), '_datacopied', False)
        # Calling _datacopied(args, kwargs) (line 190)
        _datacopied_call_result_14741 = invoke(stypy.reporting.localization.Localization(__file__, 190, 37), _datacopied_14737, *[b1_14738, b_14739], **kwargs_14740)
        
        # Applying the binary operator 'or' (line 190)
        result_or_keyword_14742 = python_operator(stypy.reporting.localization.Localization(__file__, 190, 22), 'or', overwrite_b_14736, _datacopied_call_result_14741)
        
        # Assigning a type to the variable 'overwrite_b' (line 190)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 190, 8), 'overwrite_b', result_or_keyword_14742)
        
        
        # Evaluating a boolean operation
        
        
        # Call to len(...): (line 191)
        # Processing the call arguments (line 191)
        # Getting the type of 'b1' (line 191)
        b1_14744 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 15), 'b1', False)
        # Obtaining the member 'shape' of a type (line 191)
        shape_14745 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 191, 15), b1_14744, 'shape')
        # Processing the call keyword arguments (line 191)
        kwargs_14746 = {}
        # Getting the type of 'len' (line 191)
        len_14743 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 11), 'len', False)
        # Calling len(args, kwargs) (line 191)
        len_call_result_14747 = invoke(stypy.reporting.localization.Localization(__file__, 191, 11), len_14743, *[shape_14745], **kwargs_14746)
        
        int_14748 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 191, 28), 'int')
        # Applying the binary operator '!=' (line 191)
        result_ne_14749 = python_operator(stypy.reporting.localization.Localization(__file__, 191, 11), '!=', len_call_result_14747, int_14748)
        
        
        
        # Obtaining the type of the subscript
        int_14750 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 191, 42), 'int')
        # Getting the type of 'b1' (line 191)
        b1_14751 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 33), 'b1')
        # Obtaining the member 'shape' of a type (line 191)
        shape_14752 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 191, 33), b1_14751, 'shape')
        # Obtaining the member '__getitem__' of a type (line 191)
        getitem___14753 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 191, 33), shape_14752, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 191)
        subscript_call_result_14754 = invoke(stypy.reporting.localization.Localization(__file__, 191, 33), getitem___14753, int_14750)
        
        
        # Obtaining the type of the subscript
        int_14755 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 191, 57), 'int')
        # Getting the type of 'b1' (line 191)
        b1_14756 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 48), 'b1')
        # Obtaining the member 'shape' of a type (line 191)
        shape_14757 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 191, 48), b1_14756, 'shape')
        # Obtaining the member '__getitem__' of a type (line 191)
        getitem___14758 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 191, 48), shape_14757, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 191)
        subscript_call_result_14759 = invoke(stypy.reporting.localization.Localization(__file__, 191, 48), getitem___14758, int_14755)
        
        # Applying the binary operator '!=' (line 191)
        result_ne_14760 = python_operator(stypy.reporting.localization.Localization(__file__, 191, 33), '!=', subscript_call_result_14754, subscript_call_result_14759)
        
        # Applying the binary operator 'or' (line 191)
        result_or_keyword_14761 = python_operator(stypy.reporting.localization.Localization(__file__, 191, 11), 'or', result_ne_14749, result_ne_14760)
        
        # Testing the type of an if condition (line 191)
        if_condition_14762 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 191, 8), result_or_keyword_14761)
        # Assigning a type to the variable 'if_condition_14762' (line 191)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 191, 8), 'if_condition_14762', if_condition_14762)
        # SSA begins for if statement (line 191)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to ValueError(...): (line 192)
        # Processing the call arguments (line 192)
        str_14764 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 192, 29), 'str', 'expected square matrix')
        # Processing the call keyword arguments (line 192)
        kwargs_14765 = {}
        # Getting the type of 'ValueError' (line 192)
        ValueError_14763 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 192, 18), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 192)
        ValueError_call_result_14766 = invoke(stypy.reporting.localization.Localization(__file__, 192, 18), ValueError_14763, *[str_14764], **kwargs_14765)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 192, 12), ValueError_call_result_14766, 'raise parameter', BaseException)
        # SSA join for if statement (line 191)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Getting the type of 'b1' (line 193)
        b1_14767 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 11), 'b1')
        # Obtaining the member 'shape' of a type (line 193)
        shape_14768 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 193, 11), b1_14767, 'shape')
        # Getting the type of 'a1' (line 193)
        a1_14769 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 23), 'a1')
        # Obtaining the member 'shape' of a type (line 193)
        shape_14770 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 193, 23), a1_14769, 'shape')
        # Applying the binary operator '!=' (line 193)
        result_ne_14771 = python_operator(stypy.reporting.localization.Localization(__file__, 193, 11), '!=', shape_14768, shape_14770)
        
        # Testing the type of an if condition (line 193)
        if_condition_14772 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 193, 8), result_ne_14771)
        # Assigning a type to the variable 'if_condition_14772' (line 193)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 193, 8), 'if_condition_14772', if_condition_14772)
        # SSA begins for if statement (line 193)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to ValueError(...): (line 194)
        # Processing the call arguments (line 194)
        str_14774 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 194, 29), 'str', 'a and b must have the same shape')
        # Processing the call keyword arguments (line 194)
        kwargs_14775 = {}
        # Getting the type of 'ValueError' (line 194)
        ValueError_14773 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 194, 18), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 194)
        ValueError_call_result_14776 = invoke(stypy.reporting.localization.Localization(__file__, 194, 18), ValueError_14773, *[str_14774], **kwargs_14775)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 194, 12), ValueError_call_result_14776, 'raise parameter', BaseException)
        # SSA join for if statement (line 193)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to _geneig(...): (line 195)
        # Processing the call arguments (line 195)
        # Getting the type of 'a1' (line 195)
        a1_14778 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 195, 23), 'a1', False)
        # Getting the type of 'b1' (line 195)
        b1_14779 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 195, 27), 'b1', False)
        # Getting the type of 'left' (line 195)
        left_14780 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 195, 31), 'left', False)
        # Getting the type of 'right' (line 195)
        right_14781 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 195, 37), 'right', False)
        # Getting the type of 'overwrite_a' (line 195)
        overwrite_a_14782 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 195, 44), 'overwrite_a', False)
        # Getting the type of 'overwrite_b' (line 195)
        overwrite_b_14783 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 195, 57), 'overwrite_b', False)
        # Getting the type of 'homogeneous_eigvals' (line 196)
        homogeneous_eigvals_14784 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 196, 23), 'homogeneous_eigvals', False)
        # Processing the call keyword arguments (line 195)
        kwargs_14785 = {}
        # Getting the type of '_geneig' (line 195)
        _geneig_14777 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 195, 15), '_geneig', False)
        # Calling _geneig(args, kwargs) (line 195)
        _geneig_call_result_14786 = invoke(stypy.reporting.localization.Localization(__file__, 195, 15), _geneig_14777, *[a1_14778, b1_14779, left_14780, right_14781, overwrite_a_14782, overwrite_b_14783, homogeneous_eigvals_14784], **kwargs_14785)
        
        # Assigning a type to the variable 'stypy_return_type' (line 195)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 195, 8), 'stypy_return_type', _geneig_call_result_14786)

        if more_types_in_union_14729:
            # SSA join for if statement (line 188)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Assigning a Call to a Tuple (line 198):
    
    # Assigning a Subscript to a Name (line 198):
    
    # Obtaining the type of the subscript
    int_14787 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 198, 4), 'int')
    
    # Call to get_lapack_funcs(...): (line 198)
    # Processing the call arguments (line 198)
    
    # Obtaining an instance of the builtin type 'tuple' (line 198)
    tuple_14789 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 198, 41), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 198)
    # Adding element type (line 198)
    str_14790 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 198, 41), 'str', 'geev')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 198, 41), tuple_14789, str_14790)
    # Adding element type (line 198)
    str_14791 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 198, 49), 'str', 'geev_lwork')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 198, 41), tuple_14789, str_14791)
    
    
    # Obtaining an instance of the builtin type 'tuple' (line 198)
    tuple_14792 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 198, 65), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 198)
    # Adding element type (line 198)
    # Getting the type of 'a1' (line 198)
    a1_14793 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 65), 'a1', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 198, 65), tuple_14792, a1_14793)
    
    # Processing the call keyword arguments (line 198)
    kwargs_14794 = {}
    # Getting the type of 'get_lapack_funcs' (line 198)
    get_lapack_funcs_14788 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 23), 'get_lapack_funcs', False)
    # Calling get_lapack_funcs(args, kwargs) (line 198)
    get_lapack_funcs_call_result_14795 = invoke(stypy.reporting.localization.Localization(__file__, 198, 23), get_lapack_funcs_14788, *[tuple_14789, tuple_14792], **kwargs_14794)
    
    # Obtaining the member '__getitem__' of a type (line 198)
    getitem___14796 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 198, 4), get_lapack_funcs_call_result_14795, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 198)
    subscript_call_result_14797 = invoke(stypy.reporting.localization.Localization(__file__, 198, 4), getitem___14796, int_14787)
    
    # Assigning a type to the variable 'tuple_var_assignment_14040' (line 198)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 198, 4), 'tuple_var_assignment_14040', subscript_call_result_14797)
    
    # Assigning a Subscript to a Name (line 198):
    
    # Obtaining the type of the subscript
    int_14798 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 198, 4), 'int')
    
    # Call to get_lapack_funcs(...): (line 198)
    # Processing the call arguments (line 198)
    
    # Obtaining an instance of the builtin type 'tuple' (line 198)
    tuple_14800 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 198, 41), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 198)
    # Adding element type (line 198)
    str_14801 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 198, 41), 'str', 'geev')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 198, 41), tuple_14800, str_14801)
    # Adding element type (line 198)
    str_14802 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 198, 49), 'str', 'geev_lwork')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 198, 41), tuple_14800, str_14802)
    
    
    # Obtaining an instance of the builtin type 'tuple' (line 198)
    tuple_14803 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 198, 65), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 198)
    # Adding element type (line 198)
    # Getting the type of 'a1' (line 198)
    a1_14804 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 65), 'a1', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 198, 65), tuple_14803, a1_14804)
    
    # Processing the call keyword arguments (line 198)
    kwargs_14805 = {}
    # Getting the type of 'get_lapack_funcs' (line 198)
    get_lapack_funcs_14799 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 23), 'get_lapack_funcs', False)
    # Calling get_lapack_funcs(args, kwargs) (line 198)
    get_lapack_funcs_call_result_14806 = invoke(stypy.reporting.localization.Localization(__file__, 198, 23), get_lapack_funcs_14799, *[tuple_14800, tuple_14803], **kwargs_14805)
    
    # Obtaining the member '__getitem__' of a type (line 198)
    getitem___14807 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 198, 4), get_lapack_funcs_call_result_14806, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 198)
    subscript_call_result_14808 = invoke(stypy.reporting.localization.Localization(__file__, 198, 4), getitem___14807, int_14798)
    
    # Assigning a type to the variable 'tuple_var_assignment_14041' (line 198)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 198, 4), 'tuple_var_assignment_14041', subscript_call_result_14808)
    
    # Assigning a Name to a Name (line 198):
    # Getting the type of 'tuple_var_assignment_14040' (line 198)
    tuple_var_assignment_14040_14809 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 4), 'tuple_var_assignment_14040')
    # Assigning a type to the variable 'geev' (line 198)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 198, 4), 'geev', tuple_var_assignment_14040_14809)
    
    # Assigning a Name to a Name (line 198):
    # Getting the type of 'tuple_var_assignment_14041' (line 198)
    tuple_var_assignment_14041_14810 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 4), 'tuple_var_assignment_14041')
    # Assigning a type to the variable 'geev_lwork' (line 198)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 198, 10), 'geev_lwork', tuple_var_assignment_14041_14810)
    
    # Assigning a Tuple to a Tuple (line 199):
    
    # Assigning a Name to a Name (line 199):
    # Getting the type of 'left' (line 199)
    left_14811 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 29), 'left')
    # Assigning a type to the variable 'tuple_assignment_14042' (line 199)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 199, 4), 'tuple_assignment_14042', left_14811)
    
    # Assigning a Name to a Name (line 199):
    # Getting the type of 'right' (line 199)
    right_14812 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 35), 'right')
    # Assigning a type to the variable 'tuple_assignment_14043' (line 199)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 199, 4), 'tuple_assignment_14043', right_14812)
    
    # Assigning a Name to a Name (line 199):
    # Getting the type of 'tuple_assignment_14042' (line 199)
    tuple_assignment_14042_14813 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 4), 'tuple_assignment_14042')
    # Assigning a type to the variable 'compute_vl' (line 199)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 199, 4), 'compute_vl', tuple_assignment_14042_14813)
    
    # Assigning a Name to a Name (line 199):
    # Getting the type of 'tuple_assignment_14043' (line 199)
    tuple_assignment_14043_14814 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 4), 'tuple_assignment_14043')
    # Assigning a type to the variable 'compute_vr' (line 199)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 199, 16), 'compute_vr', tuple_assignment_14043_14814)
    
    # Assigning a Call to a Name (line 201):
    
    # Assigning a Call to a Name (line 201):
    
    # Call to _compute_lwork(...): (line 201)
    # Processing the call arguments (line 201)
    # Getting the type of 'geev_lwork' (line 201)
    geev_lwork_14816 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 27), 'geev_lwork', False)
    
    # Obtaining the type of the subscript
    int_14817 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 201, 48), 'int')
    # Getting the type of 'a1' (line 201)
    a1_14818 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 39), 'a1', False)
    # Obtaining the member 'shape' of a type (line 201)
    shape_14819 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 201, 39), a1_14818, 'shape')
    # Obtaining the member '__getitem__' of a type (line 201)
    getitem___14820 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 201, 39), shape_14819, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 201)
    subscript_call_result_14821 = invoke(stypy.reporting.localization.Localization(__file__, 201, 39), getitem___14820, int_14817)
    
    # Processing the call keyword arguments (line 201)
    # Getting the type of 'compute_vl' (line 202)
    compute_vl_14822 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 38), 'compute_vl', False)
    keyword_14823 = compute_vl_14822
    # Getting the type of 'compute_vr' (line 203)
    compute_vr_14824 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 38), 'compute_vr', False)
    keyword_14825 = compute_vr_14824
    kwargs_14826 = {'compute_vl': keyword_14823, 'compute_vr': keyword_14825}
    # Getting the type of '_compute_lwork' (line 201)
    _compute_lwork_14815 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 12), '_compute_lwork', False)
    # Calling _compute_lwork(args, kwargs) (line 201)
    _compute_lwork_call_result_14827 = invoke(stypy.reporting.localization.Localization(__file__, 201, 12), _compute_lwork_14815, *[geev_lwork_14816, subscript_call_result_14821], **kwargs_14826)
    
    # Assigning a type to the variable 'lwork' (line 201)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 201, 4), 'lwork', _compute_lwork_call_result_14827)
    
    
    # Getting the type of 'geev' (line 205)
    geev_14828 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 7), 'geev')
    # Obtaining the member 'typecode' of a type (line 205)
    typecode_14829 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 205, 7), geev_14828, 'typecode')
    str_14830 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 205, 24), 'str', 'cz')
    # Applying the binary operator 'in' (line 205)
    result_contains_14831 = python_operator(stypy.reporting.localization.Localization(__file__, 205, 7), 'in', typecode_14829, str_14830)
    
    # Testing the type of an if condition (line 205)
    if_condition_14832 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 205, 4), result_contains_14831)
    # Assigning a type to the variable 'if_condition_14832' (line 205)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 205, 4), 'if_condition_14832', if_condition_14832)
    # SSA begins for if statement (line 205)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Tuple (line 206):
    
    # Assigning a Subscript to a Name (line 206):
    
    # Obtaining the type of the subscript
    int_14833 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 206, 8), 'int')
    
    # Call to geev(...): (line 206)
    # Processing the call arguments (line 206)
    # Getting the type of 'a1' (line 206)
    a1_14835 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 206, 31), 'a1', False)
    # Processing the call keyword arguments (line 206)
    # Getting the type of 'lwork' (line 206)
    lwork_14836 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 206, 41), 'lwork', False)
    keyword_14837 = lwork_14836
    # Getting the type of 'compute_vl' (line 207)
    compute_vl_14838 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 207, 42), 'compute_vl', False)
    keyword_14839 = compute_vl_14838
    # Getting the type of 'compute_vr' (line 208)
    compute_vr_14840 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 208, 42), 'compute_vr', False)
    keyword_14841 = compute_vr_14840
    # Getting the type of 'overwrite_a' (line 209)
    overwrite_a_14842 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 43), 'overwrite_a', False)
    keyword_14843 = overwrite_a_14842
    kwargs_14844 = {'overwrite_a': keyword_14843, 'compute_vl': keyword_14839, 'compute_vr': keyword_14841, 'lwork': keyword_14837}
    # Getting the type of 'geev' (line 206)
    geev_14834 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 206, 26), 'geev', False)
    # Calling geev(args, kwargs) (line 206)
    geev_call_result_14845 = invoke(stypy.reporting.localization.Localization(__file__, 206, 26), geev_14834, *[a1_14835], **kwargs_14844)
    
    # Obtaining the member '__getitem__' of a type (line 206)
    getitem___14846 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 206, 8), geev_call_result_14845, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 206)
    subscript_call_result_14847 = invoke(stypy.reporting.localization.Localization(__file__, 206, 8), getitem___14846, int_14833)
    
    # Assigning a type to the variable 'tuple_var_assignment_14044' (line 206)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 206, 8), 'tuple_var_assignment_14044', subscript_call_result_14847)
    
    # Assigning a Subscript to a Name (line 206):
    
    # Obtaining the type of the subscript
    int_14848 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 206, 8), 'int')
    
    # Call to geev(...): (line 206)
    # Processing the call arguments (line 206)
    # Getting the type of 'a1' (line 206)
    a1_14850 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 206, 31), 'a1', False)
    # Processing the call keyword arguments (line 206)
    # Getting the type of 'lwork' (line 206)
    lwork_14851 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 206, 41), 'lwork', False)
    keyword_14852 = lwork_14851
    # Getting the type of 'compute_vl' (line 207)
    compute_vl_14853 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 207, 42), 'compute_vl', False)
    keyword_14854 = compute_vl_14853
    # Getting the type of 'compute_vr' (line 208)
    compute_vr_14855 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 208, 42), 'compute_vr', False)
    keyword_14856 = compute_vr_14855
    # Getting the type of 'overwrite_a' (line 209)
    overwrite_a_14857 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 43), 'overwrite_a', False)
    keyword_14858 = overwrite_a_14857
    kwargs_14859 = {'overwrite_a': keyword_14858, 'compute_vl': keyword_14854, 'compute_vr': keyword_14856, 'lwork': keyword_14852}
    # Getting the type of 'geev' (line 206)
    geev_14849 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 206, 26), 'geev', False)
    # Calling geev(args, kwargs) (line 206)
    geev_call_result_14860 = invoke(stypy.reporting.localization.Localization(__file__, 206, 26), geev_14849, *[a1_14850], **kwargs_14859)
    
    # Obtaining the member '__getitem__' of a type (line 206)
    getitem___14861 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 206, 8), geev_call_result_14860, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 206)
    subscript_call_result_14862 = invoke(stypy.reporting.localization.Localization(__file__, 206, 8), getitem___14861, int_14848)
    
    # Assigning a type to the variable 'tuple_var_assignment_14045' (line 206)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 206, 8), 'tuple_var_assignment_14045', subscript_call_result_14862)
    
    # Assigning a Subscript to a Name (line 206):
    
    # Obtaining the type of the subscript
    int_14863 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 206, 8), 'int')
    
    # Call to geev(...): (line 206)
    # Processing the call arguments (line 206)
    # Getting the type of 'a1' (line 206)
    a1_14865 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 206, 31), 'a1', False)
    # Processing the call keyword arguments (line 206)
    # Getting the type of 'lwork' (line 206)
    lwork_14866 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 206, 41), 'lwork', False)
    keyword_14867 = lwork_14866
    # Getting the type of 'compute_vl' (line 207)
    compute_vl_14868 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 207, 42), 'compute_vl', False)
    keyword_14869 = compute_vl_14868
    # Getting the type of 'compute_vr' (line 208)
    compute_vr_14870 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 208, 42), 'compute_vr', False)
    keyword_14871 = compute_vr_14870
    # Getting the type of 'overwrite_a' (line 209)
    overwrite_a_14872 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 43), 'overwrite_a', False)
    keyword_14873 = overwrite_a_14872
    kwargs_14874 = {'overwrite_a': keyword_14873, 'compute_vl': keyword_14869, 'compute_vr': keyword_14871, 'lwork': keyword_14867}
    # Getting the type of 'geev' (line 206)
    geev_14864 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 206, 26), 'geev', False)
    # Calling geev(args, kwargs) (line 206)
    geev_call_result_14875 = invoke(stypy.reporting.localization.Localization(__file__, 206, 26), geev_14864, *[a1_14865], **kwargs_14874)
    
    # Obtaining the member '__getitem__' of a type (line 206)
    getitem___14876 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 206, 8), geev_call_result_14875, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 206)
    subscript_call_result_14877 = invoke(stypy.reporting.localization.Localization(__file__, 206, 8), getitem___14876, int_14863)
    
    # Assigning a type to the variable 'tuple_var_assignment_14046' (line 206)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 206, 8), 'tuple_var_assignment_14046', subscript_call_result_14877)
    
    # Assigning a Subscript to a Name (line 206):
    
    # Obtaining the type of the subscript
    int_14878 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 206, 8), 'int')
    
    # Call to geev(...): (line 206)
    # Processing the call arguments (line 206)
    # Getting the type of 'a1' (line 206)
    a1_14880 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 206, 31), 'a1', False)
    # Processing the call keyword arguments (line 206)
    # Getting the type of 'lwork' (line 206)
    lwork_14881 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 206, 41), 'lwork', False)
    keyword_14882 = lwork_14881
    # Getting the type of 'compute_vl' (line 207)
    compute_vl_14883 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 207, 42), 'compute_vl', False)
    keyword_14884 = compute_vl_14883
    # Getting the type of 'compute_vr' (line 208)
    compute_vr_14885 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 208, 42), 'compute_vr', False)
    keyword_14886 = compute_vr_14885
    # Getting the type of 'overwrite_a' (line 209)
    overwrite_a_14887 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 43), 'overwrite_a', False)
    keyword_14888 = overwrite_a_14887
    kwargs_14889 = {'overwrite_a': keyword_14888, 'compute_vl': keyword_14884, 'compute_vr': keyword_14886, 'lwork': keyword_14882}
    # Getting the type of 'geev' (line 206)
    geev_14879 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 206, 26), 'geev', False)
    # Calling geev(args, kwargs) (line 206)
    geev_call_result_14890 = invoke(stypy.reporting.localization.Localization(__file__, 206, 26), geev_14879, *[a1_14880], **kwargs_14889)
    
    # Obtaining the member '__getitem__' of a type (line 206)
    getitem___14891 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 206, 8), geev_call_result_14890, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 206)
    subscript_call_result_14892 = invoke(stypy.reporting.localization.Localization(__file__, 206, 8), getitem___14891, int_14878)
    
    # Assigning a type to the variable 'tuple_var_assignment_14047' (line 206)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 206, 8), 'tuple_var_assignment_14047', subscript_call_result_14892)
    
    # Assigning a Name to a Name (line 206):
    # Getting the type of 'tuple_var_assignment_14044' (line 206)
    tuple_var_assignment_14044_14893 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 206, 8), 'tuple_var_assignment_14044')
    # Assigning a type to the variable 'w' (line 206)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 206, 8), 'w', tuple_var_assignment_14044_14893)
    
    # Assigning a Name to a Name (line 206):
    # Getting the type of 'tuple_var_assignment_14045' (line 206)
    tuple_var_assignment_14045_14894 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 206, 8), 'tuple_var_assignment_14045')
    # Assigning a type to the variable 'vl' (line 206)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 206, 11), 'vl', tuple_var_assignment_14045_14894)
    
    # Assigning a Name to a Name (line 206):
    # Getting the type of 'tuple_var_assignment_14046' (line 206)
    tuple_var_assignment_14046_14895 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 206, 8), 'tuple_var_assignment_14046')
    # Assigning a type to the variable 'vr' (line 206)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 206, 15), 'vr', tuple_var_assignment_14046_14895)
    
    # Assigning a Name to a Name (line 206):
    # Getting the type of 'tuple_var_assignment_14047' (line 206)
    tuple_var_assignment_14047_14896 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 206, 8), 'tuple_var_assignment_14047')
    # Assigning a type to the variable 'info' (line 206)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 206, 19), 'info', tuple_var_assignment_14047_14896)
    
    # Assigning a Call to a Name (line 210):
    
    # Assigning a Call to a Name (line 210):
    
    # Call to _make_eigvals(...): (line 210)
    # Processing the call arguments (line 210)
    # Getting the type of 'w' (line 210)
    w_14898 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 26), 'w', False)
    # Getting the type of 'None' (line 210)
    None_14899 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 29), 'None', False)
    # Getting the type of 'homogeneous_eigvals' (line 210)
    homogeneous_eigvals_14900 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 35), 'homogeneous_eigvals', False)
    # Processing the call keyword arguments (line 210)
    kwargs_14901 = {}
    # Getting the type of '_make_eigvals' (line 210)
    _make_eigvals_14897 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 12), '_make_eigvals', False)
    # Calling _make_eigvals(args, kwargs) (line 210)
    _make_eigvals_call_result_14902 = invoke(stypy.reporting.localization.Localization(__file__, 210, 12), _make_eigvals_14897, *[w_14898, None_14899, homogeneous_eigvals_14900], **kwargs_14901)
    
    # Assigning a type to the variable 'w' (line 210)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 210, 8), 'w', _make_eigvals_call_result_14902)
    # SSA branch for the else part of an if statement (line 205)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Call to a Tuple (line 212):
    
    # Assigning a Subscript to a Name (line 212):
    
    # Obtaining the type of the subscript
    int_14903 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 212, 8), 'int')
    
    # Call to geev(...): (line 212)
    # Processing the call arguments (line 212)
    # Getting the type of 'a1' (line 212)
    a1_14905 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 36), 'a1', False)
    # Processing the call keyword arguments (line 212)
    # Getting the type of 'lwork' (line 212)
    lwork_14906 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 46), 'lwork', False)
    keyword_14907 = lwork_14906
    # Getting the type of 'compute_vl' (line 213)
    compute_vl_14908 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 47), 'compute_vl', False)
    keyword_14909 = compute_vl_14908
    # Getting the type of 'compute_vr' (line 214)
    compute_vr_14910 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 47), 'compute_vr', False)
    keyword_14911 = compute_vr_14910
    # Getting the type of 'overwrite_a' (line 215)
    overwrite_a_14912 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 215, 48), 'overwrite_a', False)
    keyword_14913 = overwrite_a_14912
    kwargs_14914 = {'overwrite_a': keyword_14913, 'compute_vl': keyword_14909, 'compute_vr': keyword_14911, 'lwork': keyword_14907}
    # Getting the type of 'geev' (line 212)
    geev_14904 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 31), 'geev', False)
    # Calling geev(args, kwargs) (line 212)
    geev_call_result_14915 = invoke(stypy.reporting.localization.Localization(__file__, 212, 31), geev_14904, *[a1_14905], **kwargs_14914)
    
    # Obtaining the member '__getitem__' of a type (line 212)
    getitem___14916 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 212, 8), geev_call_result_14915, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 212)
    subscript_call_result_14917 = invoke(stypy.reporting.localization.Localization(__file__, 212, 8), getitem___14916, int_14903)
    
    # Assigning a type to the variable 'tuple_var_assignment_14048' (line 212)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 212, 8), 'tuple_var_assignment_14048', subscript_call_result_14917)
    
    # Assigning a Subscript to a Name (line 212):
    
    # Obtaining the type of the subscript
    int_14918 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 212, 8), 'int')
    
    # Call to geev(...): (line 212)
    # Processing the call arguments (line 212)
    # Getting the type of 'a1' (line 212)
    a1_14920 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 36), 'a1', False)
    # Processing the call keyword arguments (line 212)
    # Getting the type of 'lwork' (line 212)
    lwork_14921 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 46), 'lwork', False)
    keyword_14922 = lwork_14921
    # Getting the type of 'compute_vl' (line 213)
    compute_vl_14923 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 47), 'compute_vl', False)
    keyword_14924 = compute_vl_14923
    # Getting the type of 'compute_vr' (line 214)
    compute_vr_14925 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 47), 'compute_vr', False)
    keyword_14926 = compute_vr_14925
    # Getting the type of 'overwrite_a' (line 215)
    overwrite_a_14927 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 215, 48), 'overwrite_a', False)
    keyword_14928 = overwrite_a_14927
    kwargs_14929 = {'overwrite_a': keyword_14928, 'compute_vl': keyword_14924, 'compute_vr': keyword_14926, 'lwork': keyword_14922}
    # Getting the type of 'geev' (line 212)
    geev_14919 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 31), 'geev', False)
    # Calling geev(args, kwargs) (line 212)
    geev_call_result_14930 = invoke(stypy.reporting.localization.Localization(__file__, 212, 31), geev_14919, *[a1_14920], **kwargs_14929)
    
    # Obtaining the member '__getitem__' of a type (line 212)
    getitem___14931 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 212, 8), geev_call_result_14930, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 212)
    subscript_call_result_14932 = invoke(stypy.reporting.localization.Localization(__file__, 212, 8), getitem___14931, int_14918)
    
    # Assigning a type to the variable 'tuple_var_assignment_14049' (line 212)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 212, 8), 'tuple_var_assignment_14049', subscript_call_result_14932)
    
    # Assigning a Subscript to a Name (line 212):
    
    # Obtaining the type of the subscript
    int_14933 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 212, 8), 'int')
    
    # Call to geev(...): (line 212)
    # Processing the call arguments (line 212)
    # Getting the type of 'a1' (line 212)
    a1_14935 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 36), 'a1', False)
    # Processing the call keyword arguments (line 212)
    # Getting the type of 'lwork' (line 212)
    lwork_14936 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 46), 'lwork', False)
    keyword_14937 = lwork_14936
    # Getting the type of 'compute_vl' (line 213)
    compute_vl_14938 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 47), 'compute_vl', False)
    keyword_14939 = compute_vl_14938
    # Getting the type of 'compute_vr' (line 214)
    compute_vr_14940 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 47), 'compute_vr', False)
    keyword_14941 = compute_vr_14940
    # Getting the type of 'overwrite_a' (line 215)
    overwrite_a_14942 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 215, 48), 'overwrite_a', False)
    keyword_14943 = overwrite_a_14942
    kwargs_14944 = {'overwrite_a': keyword_14943, 'compute_vl': keyword_14939, 'compute_vr': keyword_14941, 'lwork': keyword_14937}
    # Getting the type of 'geev' (line 212)
    geev_14934 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 31), 'geev', False)
    # Calling geev(args, kwargs) (line 212)
    geev_call_result_14945 = invoke(stypy.reporting.localization.Localization(__file__, 212, 31), geev_14934, *[a1_14935], **kwargs_14944)
    
    # Obtaining the member '__getitem__' of a type (line 212)
    getitem___14946 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 212, 8), geev_call_result_14945, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 212)
    subscript_call_result_14947 = invoke(stypy.reporting.localization.Localization(__file__, 212, 8), getitem___14946, int_14933)
    
    # Assigning a type to the variable 'tuple_var_assignment_14050' (line 212)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 212, 8), 'tuple_var_assignment_14050', subscript_call_result_14947)
    
    # Assigning a Subscript to a Name (line 212):
    
    # Obtaining the type of the subscript
    int_14948 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 212, 8), 'int')
    
    # Call to geev(...): (line 212)
    # Processing the call arguments (line 212)
    # Getting the type of 'a1' (line 212)
    a1_14950 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 36), 'a1', False)
    # Processing the call keyword arguments (line 212)
    # Getting the type of 'lwork' (line 212)
    lwork_14951 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 46), 'lwork', False)
    keyword_14952 = lwork_14951
    # Getting the type of 'compute_vl' (line 213)
    compute_vl_14953 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 47), 'compute_vl', False)
    keyword_14954 = compute_vl_14953
    # Getting the type of 'compute_vr' (line 214)
    compute_vr_14955 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 47), 'compute_vr', False)
    keyword_14956 = compute_vr_14955
    # Getting the type of 'overwrite_a' (line 215)
    overwrite_a_14957 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 215, 48), 'overwrite_a', False)
    keyword_14958 = overwrite_a_14957
    kwargs_14959 = {'overwrite_a': keyword_14958, 'compute_vl': keyword_14954, 'compute_vr': keyword_14956, 'lwork': keyword_14952}
    # Getting the type of 'geev' (line 212)
    geev_14949 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 31), 'geev', False)
    # Calling geev(args, kwargs) (line 212)
    geev_call_result_14960 = invoke(stypy.reporting.localization.Localization(__file__, 212, 31), geev_14949, *[a1_14950], **kwargs_14959)
    
    # Obtaining the member '__getitem__' of a type (line 212)
    getitem___14961 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 212, 8), geev_call_result_14960, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 212)
    subscript_call_result_14962 = invoke(stypy.reporting.localization.Localization(__file__, 212, 8), getitem___14961, int_14948)
    
    # Assigning a type to the variable 'tuple_var_assignment_14051' (line 212)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 212, 8), 'tuple_var_assignment_14051', subscript_call_result_14962)
    
    # Assigning a Subscript to a Name (line 212):
    
    # Obtaining the type of the subscript
    int_14963 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 212, 8), 'int')
    
    # Call to geev(...): (line 212)
    # Processing the call arguments (line 212)
    # Getting the type of 'a1' (line 212)
    a1_14965 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 36), 'a1', False)
    # Processing the call keyword arguments (line 212)
    # Getting the type of 'lwork' (line 212)
    lwork_14966 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 46), 'lwork', False)
    keyword_14967 = lwork_14966
    # Getting the type of 'compute_vl' (line 213)
    compute_vl_14968 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 47), 'compute_vl', False)
    keyword_14969 = compute_vl_14968
    # Getting the type of 'compute_vr' (line 214)
    compute_vr_14970 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 47), 'compute_vr', False)
    keyword_14971 = compute_vr_14970
    # Getting the type of 'overwrite_a' (line 215)
    overwrite_a_14972 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 215, 48), 'overwrite_a', False)
    keyword_14973 = overwrite_a_14972
    kwargs_14974 = {'overwrite_a': keyword_14973, 'compute_vl': keyword_14969, 'compute_vr': keyword_14971, 'lwork': keyword_14967}
    # Getting the type of 'geev' (line 212)
    geev_14964 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 31), 'geev', False)
    # Calling geev(args, kwargs) (line 212)
    geev_call_result_14975 = invoke(stypy.reporting.localization.Localization(__file__, 212, 31), geev_14964, *[a1_14965], **kwargs_14974)
    
    # Obtaining the member '__getitem__' of a type (line 212)
    getitem___14976 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 212, 8), geev_call_result_14975, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 212)
    subscript_call_result_14977 = invoke(stypy.reporting.localization.Localization(__file__, 212, 8), getitem___14976, int_14963)
    
    # Assigning a type to the variable 'tuple_var_assignment_14052' (line 212)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 212, 8), 'tuple_var_assignment_14052', subscript_call_result_14977)
    
    # Assigning a Name to a Name (line 212):
    # Getting the type of 'tuple_var_assignment_14048' (line 212)
    tuple_var_assignment_14048_14978 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 8), 'tuple_var_assignment_14048')
    # Assigning a type to the variable 'wr' (line 212)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 212, 8), 'wr', tuple_var_assignment_14048_14978)
    
    # Assigning a Name to a Name (line 212):
    # Getting the type of 'tuple_var_assignment_14049' (line 212)
    tuple_var_assignment_14049_14979 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 8), 'tuple_var_assignment_14049')
    # Assigning a type to the variable 'wi' (line 212)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 212, 12), 'wi', tuple_var_assignment_14049_14979)
    
    # Assigning a Name to a Name (line 212):
    # Getting the type of 'tuple_var_assignment_14050' (line 212)
    tuple_var_assignment_14050_14980 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 8), 'tuple_var_assignment_14050')
    # Assigning a type to the variable 'vl' (line 212)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 212, 16), 'vl', tuple_var_assignment_14050_14980)
    
    # Assigning a Name to a Name (line 212):
    # Getting the type of 'tuple_var_assignment_14051' (line 212)
    tuple_var_assignment_14051_14981 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 8), 'tuple_var_assignment_14051')
    # Assigning a type to the variable 'vr' (line 212)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 212, 20), 'vr', tuple_var_assignment_14051_14981)
    
    # Assigning a Name to a Name (line 212):
    # Getting the type of 'tuple_var_assignment_14052' (line 212)
    tuple_var_assignment_14052_14982 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 8), 'tuple_var_assignment_14052')
    # Assigning a type to the variable 'info' (line 212)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 212, 24), 'info', tuple_var_assignment_14052_14982)
    
    # Assigning a Subscript to a Name (line 216):
    
    # Assigning a Subscript to a Name (line 216):
    
    # Obtaining the type of the subscript
    # Getting the type of 'wr' (line 216)
    wr_14983 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 216, 33), 'wr')
    # Obtaining the member 'dtype' of a type (line 216)
    dtype_14984 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 216, 33), wr_14983, 'dtype')
    # Obtaining the member 'char' of a type (line 216)
    char_14985 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 216, 33), dtype_14984, 'char')
    
    # Obtaining an instance of the builtin type 'dict' (line 216)
    dict_14986 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 216, 12), 'dict')
    # Adding type elements to the builtin type 'dict' instance (line 216)
    # Adding element type (key, value) (line 216)
    str_14987 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 216, 13), 'str', 'f')
    str_14988 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 216, 18), 'str', 'F')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 216, 12), dict_14986, (str_14987, str_14988))
    # Adding element type (key, value) (line 216)
    str_14989 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 216, 23), 'str', 'd')
    str_14990 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 216, 28), 'str', 'D')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 216, 12), dict_14986, (str_14989, str_14990))
    
    # Obtaining the member '__getitem__' of a type (line 216)
    getitem___14991 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 216, 12), dict_14986, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 216)
    subscript_call_result_14992 = invoke(stypy.reporting.localization.Localization(__file__, 216, 12), getitem___14991, char_14985)
    
    # Assigning a type to the variable 't' (line 216)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 216, 8), 't', subscript_call_result_14992)
    
    # Assigning a BinOp to a Name (line 217):
    
    # Assigning a BinOp to a Name (line 217):
    # Getting the type of 'wr' (line 217)
    wr_14993 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 12), 'wr')
    # Getting the type of '_I' (line 217)
    _I_14994 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 17), '_I')
    # Getting the type of 'wi' (line 217)
    wi_14995 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 22), 'wi')
    # Applying the binary operator '*' (line 217)
    result_mul_14996 = python_operator(stypy.reporting.localization.Localization(__file__, 217, 17), '*', _I_14994, wi_14995)
    
    # Applying the binary operator '+' (line 217)
    result_add_14997 = python_operator(stypy.reporting.localization.Localization(__file__, 217, 12), '+', wr_14993, result_mul_14996)
    
    # Assigning a type to the variable 'w' (line 217)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 217, 8), 'w', result_add_14997)
    
    # Assigning a Call to a Name (line 218):
    
    # Assigning a Call to a Name (line 218):
    
    # Call to _make_eigvals(...): (line 218)
    # Processing the call arguments (line 218)
    # Getting the type of 'w' (line 218)
    w_14999 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 218, 26), 'w', False)
    # Getting the type of 'None' (line 218)
    None_15000 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 218, 29), 'None', False)
    # Getting the type of 'homogeneous_eigvals' (line 218)
    homogeneous_eigvals_15001 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 218, 35), 'homogeneous_eigvals', False)
    # Processing the call keyword arguments (line 218)
    kwargs_15002 = {}
    # Getting the type of '_make_eigvals' (line 218)
    _make_eigvals_14998 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 218, 12), '_make_eigvals', False)
    # Calling _make_eigvals(args, kwargs) (line 218)
    _make_eigvals_call_result_15003 = invoke(stypy.reporting.localization.Localization(__file__, 218, 12), _make_eigvals_14998, *[w_14999, None_15000, homogeneous_eigvals_15001], **kwargs_15002)
    
    # Assigning a type to the variable 'w' (line 218)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 218, 8), 'w', _make_eigvals_call_result_15003)
    # SSA join for if statement (line 205)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to _check_info(...): (line 220)
    # Processing the call arguments (line 220)
    # Getting the type of 'info' (line 220)
    info_15005 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 220, 16), 'info', False)
    str_15006 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 220, 22), 'str', 'eig algorithm (geev)')
    # Processing the call keyword arguments (line 220)
    str_15007 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 221, 25), 'str', 'did not converge (only eigenvalues with order >= %d have converged)')
    keyword_15008 = str_15007
    kwargs_15009 = {'positive': keyword_15008}
    # Getting the type of '_check_info' (line 220)
    _check_info_15004 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 220, 4), '_check_info', False)
    # Calling _check_info(args, kwargs) (line 220)
    _check_info_call_result_15010 = invoke(stypy.reporting.localization.Localization(__file__, 220, 4), _check_info_15004, *[info_15005, str_15006], **kwargs_15009)
    
    
    # Assigning a Call to a Name (line 224):
    
    # Assigning a Call to a Name (line 224):
    
    # Call to all(...): (line 224)
    # Processing the call arguments (line 224)
    
    # Getting the type of 'w' (line 224)
    w_15013 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 224, 26), 'w', False)
    # Obtaining the member 'imag' of a type (line 224)
    imag_15014 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 224, 26), w_15013, 'imag')
    float_15015 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 224, 36), 'float')
    # Applying the binary operator '==' (line 224)
    result_eq_15016 = python_operator(stypy.reporting.localization.Localization(__file__, 224, 26), '==', imag_15014, float_15015)
    
    # Processing the call keyword arguments (line 224)
    kwargs_15017 = {}
    # Getting the type of 'numpy' (line 224)
    numpy_15011 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 224, 16), 'numpy', False)
    # Obtaining the member 'all' of a type (line 224)
    all_15012 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 224, 16), numpy_15011, 'all')
    # Calling all(args, kwargs) (line 224)
    all_call_result_15018 = invoke(stypy.reporting.localization.Localization(__file__, 224, 16), all_15012, *[result_eq_15016], **kwargs_15017)
    
    # Assigning a type to the variable 'only_real' (line 224)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 224, 4), 'only_real', all_call_result_15018)
    
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'geev' (line 225)
    geev_15019 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 225, 12), 'geev')
    # Obtaining the member 'typecode' of a type (line 225)
    typecode_15020 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 225, 12), geev_15019, 'typecode')
    str_15021 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 225, 29), 'str', 'cz')
    # Applying the binary operator 'in' (line 225)
    result_contains_15022 = python_operator(stypy.reporting.localization.Localization(__file__, 225, 12), 'in', typecode_15020, str_15021)
    
    # Getting the type of 'only_real' (line 225)
    only_real_15023 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 225, 37), 'only_real')
    # Applying the binary operator 'or' (line 225)
    result_or_keyword_15024 = python_operator(stypy.reporting.localization.Localization(__file__, 225, 12), 'or', result_contains_15022, only_real_15023)
    
    # Applying the 'not' unary operator (line 225)
    result_not__15025 = python_operator(stypy.reporting.localization.Localization(__file__, 225, 7), 'not', result_or_keyword_15024)
    
    # Testing the type of an if condition (line 225)
    if_condition_15026 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 225, 4), result_not__15025)
    # Assigning a type to the variable 'if_condition_15026' (line 225)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 225, 4), 'if_condition_15026', if_condition_15026)
    # SSA begins for if statement (line 225)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Attribute to a Name (line 226):
    
    # Assigning a Attribute to a Name (line 226):
    # Getting the type of 'w' (line 226)
    w_15027 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 12), 'w')
    # Obtaining the member 'dtype' of a type (line 226)
    dtype_15028 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 226, 12), w_15027, 'dtype')
    # Obtaining the member 'char' of a type (line 226)
    char_15029 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 226, 12), dtype_15028, 'char')
    # Assigning a type to the variable 't' (line 226)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 226, 8), 't', char_15029)
    
    # Getting the type of 'left' (line 227)
    left_15030 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 11), 'left')
    # Testing the type of an if condition (line 227)
    if_condition_15031 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 227, 8), left_15030)
    # Assigning a type to the variable 'if_condition_15031' (line 227)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 227, 8), 'if_condition_15031', if_condition_15031)
    # SSA begins for if statement (line 227)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 228):
    
    # Assigning a Call to a Name (line 228):
    
    # Call to _make_complex_eigvecs(...): (line 228)
    # Processing the call arguments (line 228)
    # Getting the type of 'w' (line 228)
    w_15033 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 39), 'w', False)
    # Getting the type of 'vl' (line 228)
    vl_15034 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 42), 'vl', False)
    # Getting the type of 't' (line 228)
    t_15035 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 46), 't', False)
    # Processing the call keyword arguments (line 228)
    kwargs_15036 = {}
    # Getting the type of '_make_complex_eigvecs' (line 228)
    _make_complex_eigvecs_15032 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 17), '_make_complex_eigvecs', False)
    # Calling _make_complex_eigvecs(args, kwargs) (line 228)
    _make_complex_eigvecs_call_result_15037 = invoke(stypy.reporting.localization.Localization(__file__, 228, 17), _make_complex_eigvecs_15032, *[w_15033, vl_15034, t_15035], **kwargs_15036)
    
    # Assigning a type to the variable 'vl' (line 228)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 228, 12), 'vl', _make_complex_eigvecs_call_result_15037)
    # SSA join for if statement (line 227)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Getting the type of 'right' (line 229)
    right_15038 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 229, 11), 'right')
    # Testing the type of an if condition (line 229)
    if_condition_15039 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 229, 8), right_15038)
    # Assigning a type to the variable 'if_condition_15039' (line 229)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 229, 8), 'if_condition_15039', if_condition_15039)
    # SSA begins for if statement (line 229)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 230):
    
    # Assigning a Call to a Name (line 230):
    
    # Call to _make_complex_eigvecs(...): (line 230)
    # Processing the call arguments (line 230)
    # Getting the type of 'w' (line 230)
    w_15041 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 39), 'w', False)
    # Getting the type of 'vr' (line 230)
    vr_15042 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 42), 'vr', False)
    # Getting the type of 't' (line 230)
    t_15043 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 46), 't', False)
    # Processing the call keyword arguments (line 230)
    kwargs_15044 = {}
    # Getting the type of '_make_complex_eigvecs' (line 230)
    _make_complex_eigvecs_15040 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 17), '_make_complex_eigvecs', False)
    # Calling _make_complex_eigvecs(args, kwargs) (line 230)
    _make_complex_eigvecs_call_result_15045 = invoke(stypy.reporting.localization.Localization(__file__, 230, 17), _make_complex_eigvecs_15040, *[w_15041, vr_15042, t_15043], **kwargs_15044)
    
    # Assigning a type to the variable 'vr' (line 230)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 230, 12), 'vr', _make_complex_eigvecs_call_result_15045)
    # SSA join for if statement (line 229)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 225)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    
    # Evaluating a boolean operation
    # Getting the type of 'left' (line 231)
    left_15046 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 231, 12), 'left')
    # Getting the type of 'right' (line 231)
    right_15047 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 231, 20), 'right')
    # Applying the binary operator 'or' (line 231)
    result_or_keyword_15048 = python_operator(stypy.reporting.localization.Localization(__file__, 231, 12), 'or', left_15046, right_15047)
    
    # Applying the 'not' unary operator (line 231)
    result_not__15049 = python_operator(stypy.reporting.localization.Localization(__file__, 231, 7), 'not', result_or_keyword_15048)
    
    # Testing the type of an if condition (line 231)
    if_condition_15050 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 231, 4), result_not__15049)
    # Assigning a type to the variable 'if_condition_15050' (line 231)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 231, 4), 'if_condition_15050', if_condition_15050)
    # SSA begins for if statement (line 231)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # Getting the type of 'w' (line 232)
    w_15051 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 15), 'w')
    # Assigning a type to the variable 'stypy_return_type' (line 232)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 232, 8), 'stypy_return_type', w_15051)
    # SSA join for if statement (line 231)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Getting the type of 'left' (line 233)
    left_15052 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 233, 7), 'left')
    # Testing the type of an if condition (line 233)
    if_condition_15053 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 233, 4), left_15052)
    # Assigning a type to the variable 'if_condition_15053' (line 233)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 233, 4), 'if_condition_15053', if_condition_15053)
    # SSA begins for if statement (line 233)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Getting the type of 'right' (line 234)
    right_15054 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 234, 11), 'right')
    # Testing the type of an if condition (line 234)
    if_condition_15055 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 234, 8), right_15054)
    # Assigning a type to the variable 'if_condition_15055' (line 234)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 234, 8), 'if_condition_15055', if_condition_15055)
    # SSA begins for if statement (line 234)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Obtaining an instance of the builtin type 'tuple' (line 235)
    tuple_15056 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 235, 19), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 235)
    # Adding element type (line 235)
    # Getting the type of 'w' (line 235)
    w_15057 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 19), 'w')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 235, 19), tuple_15056, w_15057)
    # Adding element type (line 235)
    # Getting the type of 'vl' (line 235)
    vl_15058 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 22), 'vl')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 235, 19), tuple_15056, vl_15058)
    # Adding element type (line 235)
    # Getting the type of 'vr' (line 235)
    vr_15059 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 26), 'vr')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 235, 19), tuple_15056, vr_15059)
    
    # Assigning a type to the variable 'stypy_return_type' (line 235)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 235, 12), 'stypy_return_type', tuple_15056)
    # SSA join for if statement (line 234)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Obtaining an instance of the builtin type 'tuple' (line 236)
    tuple_15060 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 236, 15), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 236)
    # Adding element type (line 236)
    # Getting the type of 'w' (line 236)
    w_15061 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 236, 15), 'w')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 236, 15), tuple_15060, w_15061)
    # Adding element type (line 236)
    # Getting the type of 'vl' (line 236)
    vl_15062 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 236, 18), 'vl')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 236, 15), tuple_15060, vl_15062)
    
    # Assigning a type to the variable 'stypy_return_type' (line 236)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 236, 8), 'stypy_return_type', tuple_15060)
    # SSA join for if statement (line 233)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Obtaining an instance of the builtin type 'tuple' (line 237)
    tuple_15063 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 237, 11), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 237)
    # Adding element type (line 237)
    # Getting the type of 'w' (line 237)
    w_15064 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 237, 11), 'w')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 237, 11), tuple_15063, w_15064)
    # Adding element type (line 237)
    # Getting the type of 'vr' (line 237)
    vr_15065 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 237, 14), 'vr')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 237, 11), tuple_15063, vr_15065)
    
    # Assigning a type to the variable 'stypy_return_type' (line 237)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 237, 4), 'stypy_return_type', tuple_15063)
    
    # ################# End of 'eig(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'eig' in the type store
    # Getting the type of 'stypy_return_type' (line 117)
    stypy_return_type_15066 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_15066)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'eig'
    return stypy_return_type_15066

# Assigning a type to the variable 'eig' (line 117)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 117, 0), 'eig', eig)

@norecursion
def eigh(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'None' (line 240)
    None_15067 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 14), 'None')
    # Getting the type of 'True' (line 240)
    True_15068 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 26), 'True')
    # Getting the type of 'False' (line 240)
    False_15069 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 45), 'False')
    # Getting the type of 'False' (line 240)
    False_15070 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 64), 'False')
    # Getting the type of 'False' (line 241)
    False_15071 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 241, 21), 'False')
    # Getting the type of 'True' (line 241)
    True_15072 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 241, 34), 'True')
    # Getting the type of 'None' (line 241)
    None_15073 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 241, 48), 'None')
    int_15074 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 241, 59), 'int')
    # Getting the type of 'True' (line 242)
    True_15075 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 242, 22), 'True')
    defaults = [None_15067, True_15068, False_15069, False_15070, False_15071, True_15072, None_15073, int_15074, True_15075]
    # Create a new context for function 'eigh'
    module_type_store = module_type_store.open_function_context('eigh', 240, 0, False)
    
    # Passed parameters checking function
    eigh.stypy_localization = localization
    eigh.stypy_type_of_self = None
    eigh.stypy_type_store = module_type_store
    eigh.stypy_function_name = 'eigh'
    eigh.stypy_param_names_list = ['a', 'b', 'lower', 'eigvals_only', 'overwrite_a', 'overwrite_b', 'turbo', 'eigvals', 'type', 'check_finite']
    eigh.stypy_varargs_param_name = None
    eigh.stypy_kwargs_param_name = None
    eigh.stypy_call_defaults = defaults
    eigh.stypy_call_varargs = varargs
    eigh.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'eigh', ['a', 'b', 'lower', 'eigvals_only', 'overwrite_a', 'overwrite_b', 'turbo', 'eigvals', 'type', 'check_finite'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'eigh', localization, ['a', 'b', 'lower', 'eigvals_only', 'overwrite_a', 'overwrite_b', 'turbo', 'eigvals', 'type', 'check_finite'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'eigh(...)' code ##################

    str_15076 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 328, (-1)), 'str', '\n    Solve an ordinary or generalized eigenvalue problem for a complex\n    Hermitian or real symmetric matrix.\n\n    Find eigenvalues w and optionally eigenvectors v of matrix `a`, where\n    `b` is positive definite::\n\n                      a v[:,i] = w[i] b v[:,i]\n        v[i,:].conj() a v[:,i] = w[i]\n        v[i,:].conj() b v[:,i] = 1\n\n    Parameters\n    ----------\n    a : (M, M) array_like\n        A complex Hermitian or real symmetric matrix whose eigenvalues and\n        eigenvectors will be computed.\n    b : (M, M) array_like, optional\n        A complex Hermitian or real symmetric definite positive matrix in.\n        If omitted, identity matrix is assumed.\n    lower : bool, optional\n        Whether the pertinent array data is taken from the lower or upper\n        triangle of `a`. (Default: lower)\n    eigvals_only : bool, optional\n        Whether to calculate only eigenvalues and no eigenvectors.\n        (Default: both are calculated)\n    turbo : bool, optional\n        Use divide and conquer algorithm (faster but expensive in memory,\n        only for generalized eigenvalue problem and if eigvals=None)\n    eigvals : tuple (lo, hi), optional\n        Indexes of the smallest and largest (in ascending order) eigenvalues\n        and corresponding eigenvectors to be returned: 0 <= lo <= hi <= M-1.\n        If omitted, all eigenvalues and eigenvectors are returned.\n    type : int, optional\n        Specifies the problem type to be solved:\n\n           type = 1: a   v[:,i] = w[i] b v[:,i]\n\n           type = 2: a b v[:,i] = w[i]   v[:,i]\n\n           type = 3: b a v[:,i] = w[i]   v[:,i]\n    overwrite_a : bool, optional\n        Whether to overwrite data in `a` (may improve performance)\n    overwrite_b : bool, optional\n        Whether to overwrite data in `b` (may improve performance)\n    check_finite : bool, optional\n        Whether to check that the input matrices contain only finite numbers.\n        Disabling may give a performance gain, but may result in problems\n        (crashes, non-termination) if the inputs do contain infinities or NaNs.\n\n    Returns\n    -------\n    w : (N,) float ndarray\n        The N (1<=N<=M) selected eigenvalues, in ascending order, each\n        repeated according to its multiplicity.\n    v : (M, N) complex ndarray\n        (if eigvals_only == False)\n\n        The normalized selected eigenvector corresponding to the\n        eigenvalue w[i] is the column v[:,i].\n\n        Normalization:\n\n            type 1 and 3: v.conj() a      v  = w\n\n            type 2: inv(v).conj() a  inv(v) = w\n\n            type = 1 or 2: v.conj() b      v  = I\n\n            type = 3: v.conj() inv(b) v  = I\n\n    Raises\n    ------\n    LinAlgError\n        If eigenvalue computation does not converge,\n        an error occurred, or b matrix is not definite positive. Note that\n        if input matrices are not symmetric or hermitian, no error is reported\n        but results will be wrong.\n\n    See Also\n    --------\n    eigvalsh : eigenvalues of symmetric or Hermitian arrays\n    eig : eigenvalues and right eigenvectors for non-symmetric arrays\n    eigh : eigenvalues and right eigenvectors for symmetric/Hermitian arrays\n    eigh_tridiagonal : eigenvalues and right eiegenvectors for\n        symmetric/Hermitian tridiagonal matrices\n    ')
    
    # Assigning a Call to a Name (line 329):
    
    # Assigning a Call to a Name (line 329):
    
    # Call to _asarray_validated(...): (line 329)
    # Processing the call arguments (line 329)
    # Getting the type of 'a' (line 329)
    a_15078 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 329, 28), 'a', False)
    # Processing the call keyword arguments (line 329)
    # Getting the type of 'check_finite' (line 329)
    check_finite_15079 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 329, 44), 'check_finite', False)
    keyword_15080 = check_finite_15079
    kwargs_15081 = {'check_finite': keyword_15080}
    # Getting the type of '_asarray_validated' (line 329)
    _asarray_validated_15077 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 329, 9), '_asarray_validated', False)
    # Calling _asarray_validated(args, kwargs) (line 329)
    _asarray_validated_call_result_15082 = invoke(stypy.reporting.localization.Localization(__file__, 329, 9), _asarray_validated_15077, *[a_15078], **kwargs_15081)
    
    # Assigning a type to the variable 'a1' (line 329)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 329, 4), 'a1', _asarray_validated_call_result_15082)
    
    
    # Evaluating a boolean operation
    
    
    # Call to len(...): (line 330)
    # Processing the call arguments (line 330)
    # Getting the type of 'a1' (line 330)
    a1_15084 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 330, 11), 'a1', False)
    # Obtaining the member 'shape' of a type (line 330)
    shape_15085 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 330, 11), a1_15084, 'shape')
    # Processing the call keyword arguments (line 330)
    kwargs_15086 = {}
    # Getting the type of 'len' (line 330)
    len_15083 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 330, 7), 'len', False)
    # Calling len(args, kwargs) (line 330)
    len_call_result_15087 = invoke(stypy.reporting.localization.Localization(__file__, 330, 7), len_15083, *[shape_15085], **kwargs_15086)
    
    int_15088 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 330, 24), 'int')
    # Applying the binary operator '!=' (line 330)
    result_ne_15089 = python_operator(stypy.reporting.localization.Localization(__file__, 330, 7), '!=', len_call_result_15087, int_15088)
    
    
    
    # Obtaining the type of the subscript
    int_15090 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 330, 38), 'int')
    # Getting the type of 'a1' (line 330)
    a1_15091 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 330, 29), 'a1')
    # Obtaining the member 'shape' of a type (line 330)
    shape_15092 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 330, 29), a1_15091, 'shape')
    # Obtaining the member '__getitem__' of a type (line 330)
    getitem___15093 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 330, 29), shape_15092, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 330)
    subscript_call_result_15094 = invoke(stypy.reporting.localization.Localization(__file__, 330, 29), getitem___15093, int_15090)
    
    
    # Obtaining the type of the subscript
    int_15095 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 330, 53), 'int')
    # Getting the type of 'a1' (line 330)
    a1_15096 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 330, 44), 'a1')
    # Obtaining the member 'shape' of a type (line 330)
    shape_15097 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 330, 44), a1_15096, 'shape')
    # Obtaining the member '__getitem__' of a type (line 330)
    getitem___15098 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 330, 44), shape_15097, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 330)
    subscript_call_result_15099 = invoke(stypy.reporting.localization.Localization(__file__, 330, 44), getitem___15098, int_15095)
    
    # Applying the binary operator '!=' (line 330)
    result_ne_15100 = python_operator(stypy.reporting.localization.Localization(__file__, 330, 29), '!=', subscript_call_result_15094, subscript_call_result_15099)
    
    # Applying the binary operator 'or' (line 330)
    result_or_keyword_15101 = python_operator(stypy.reporting.localization.Localization(__file__, 330, 7), 'or', result_ne_15089, result_ne_15100)
    
    # Testing the type of an if condition (line 330)
    if_condition_15102 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 330, 4), result_or_keyword_15101)
    # Assigning a type to the variable 'if_condition_15102' (line 330)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 330, 4), 'if_condition_15102', if_condition_15102)
    # SSA begins for if statement (line 330)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 331)
    # Processing the call arguments (line 331)
    str_15104 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 331, 25), 'str', 'expected square matrix')
    # Processing the call keyword arguments (line 331)
    kwargs_15105 = {}
    # Getting the type of 'ValueError' (line 331)
    ValueError_15103 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 331, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 331)
    ValueError_call_result_15106 = invoke(stypy.reporting.localization.Localization(__file__, 331, 14), ValueError_15103, *[str_15104], **kwargs_15105)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 331, 8), ValueError_call_result_15106, 'raise parameter', BaseException)
    # SSA join for if statement (line 330)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a BoolOp to a Name (line 332):
    
    # Assigning a BoolOp to a Name (line 332):
    
    # Evaluating a boolean operation
    # Getting the type of 'overwrite_a' (line 332)
    overwrite_a_15107 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 332, 18), 'overwrite_a')
    
    # Call to _datacopied(...): (line 332)
    # Processing the call arguments (line 332)
    # Getting the type of 'a1' (line 332)
    a1_15109 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 332, 46), 'a1', False)
    # Getting the type of 'a' (line 332)
    a_15110 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 332, 50), 'a', False)
    # Processing the call keyword arguments (line 332)
    kwargs_15111 = {}
    # Getting the type of '_datacopied' (line 332)
    _datacopied_15108 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 332, 34), '_datacopied', False)
    # Calling _datacopied(args, kwargs) (line 332)
    _datacopied_call_result_15112 = invoke(stypy.reporting.localization.Localization(__file__, 332, 34), _datacopied_15108, *[a1_15109, a_15110], **kwargs_15111)
    
    # Applying the binary operator 'or' (line 332)
    result_or_keyword_15113 = python_operator(stypy.reporting.localization.Localization(__file__, 332, 18), 'or', overwrite_a_15107, _datacopied_call_result_15112)
    
    # Assigning a type to the variable 'overwrite_a' (line 332)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 332, 4), 'overwrite_a', result_or_keyword_15113)
    
    
    # Call to iscomplexobj(...): (line 333)
    # Processing the call arguments (line 333)
    # Getting the type of 'a1' (line 333)
    a1_15115 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 333, 20), 'a1', False)
    # Processing the call keyword arguments (line 333)
    kwargs_15116 = {}
    # Getting the type of 'iscomplexobj' (line 333)
    iscomplexobj_15114 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 333, 7), 'iscomplexobj', False)
    # Calling iscomplexobj(args, kwargs) (line 333)
    iscomplexobj_call_result_15117 = invoke(stypy.reporting.localization.Localization(__file__, 333, 7), iscomplexobj_15114, *[a1_15115], **kwargs_15116)
    
    # Testing the type of an if condition (line 333)
    if_condition_15118 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 333, 4), iscomplexobj_call_result_15117)
    # Assigning a type to the variable 'if_condition_15118' (line 333)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 333, 4), 'if_condition_15118', if_condition_15118)
    # SSA begins for if statement (line 333)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Name to a Name (line 334):
    
    # Assigning a Name to a Name (line 334):
    # Getting the type of 'True' (line 334)
    True_15119 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 334, 15), 'True')
    # Assigning a type to the variable 'cplx' (line 334)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 334, 8), 'cplx', True_15119)
    # SSA branch for the else part of an if statement (line 333)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Name to a Name (line 336):
    
    # Assigning a Name to a Name (line 336):
    # Getting the type of 'False' (line 336)
    False_15120 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 336, 15), 'False')
    # Assigning a type to the variable 'cplx' (line 336)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 336, 8), 'cplx', False_15120)
    # SSA join for if statement (line 333)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Type idiom detected: calculating its left and rigth part (line 337)
    # Getting the type of 'b' (line 337)
    b_15121 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 337, 4), 'b')
    # Getting the type of 'None' (line 337)
    None_15122 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 337, 16), 'None')
    
    (may_be_15123, more_types_in_union_15124) = may_not_be_none(b_15121, None_15122)

    if may_be_15123:

        if more_types_in_union_15124:
            # Runtime conditional SSA (line 337)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a Call to a Name (line 338):
        
        # Assigning a Call to a Name (line 338):
        
        # Call to _asarray_validated(...): (line 338)
        # Processing the call arguments (line 338)
        # Getting the type of 'b' (line 338)
        b_15126 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 338, 32), 'b', False)
        # Processing the call keyword arguments (line 338)
        # Getting the type of 'check_finite' (line 338)
        check_finite_15127 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 338, 48), 'check_finite', False)
        keyword_15128 = check_finite_15127
        kwargs_15129 = {'check_finite': keyword_15128}
        # Getting the type of '_asarray_validated' (line 338)
        _asarray_validated_15125 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 338, 13), '_asarray_validated', False)
        # Calling _asarray_validated(args, kwargs) (line 338)
        _asarray_validated_call_result_15130 = invoke(stypy.reporting.localization.Localization(__file__, 338, 13), _asarray_validated_15125, *[b_15126], **kwargs_15129)
        
        # Assigning a type to the variable 'b1' (line 338)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 338, 8), 'b1', _asarray_validated_call_result_15130)
        
        # Assigning a BoolOp to a Name (line 339):
        
        # Assigning a BoolOp to a Name (line 339):
        
        # Evaluating a boolean operation
        # Getting the type of 'overwrite_b' (line 339)
        overwrite_b_15131 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 339, 22), 'overwrite_b')
        
        # Call to _datacopied(...): (line 339)
        # Processing the call arguments (line 339)
        # Getting the type of 'b1' (line 339)
        b1_15133 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 339, 49), 'b1', False)
        # Getting the type of 'b' (line 339)
        b_15134 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 339, 53), 'b', False)
        # Processing the call keyword arguments (line 339)
        kwargs_15135 = {}
        # Getting the type of '_datacopied' (line 339)
        _datacopied_15132 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 339, 37), '_datacopied', False)
        # Calling _datacopied(args, kwargs) (line 339)
        _datacopied_call_result_15136 = invoke(stypy.reporting.localization.Localization(__file__, 339, 37), _datacopied_15132, *[b1_15133, b_15134], **kwargs_15135)
        
        # Applying the binary operator 'or' (line 339)
        result_or_keyword_15137 = python_operator(stypy.reporting.localization.Localization(__file__, 339, 22), 'or', overwrite_b_15131, _datacopied_call_result_15136)
        
        # Assigning a type to the variable 'overwrite_b' (line 339)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 339, 8), 'overwrite_b', result_or_keyword_15137)
        
        
        # Evaluating a boolean operation
        
        
        # Call to len(...): (line 340)
        # Processing the call arguments (line 340)
        # Getting the type of 'b1' (line 340)
        b1_15139 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 340, 15), 'b1', False)
        # Obtaining the member 'shape' of a type (line 340)
        shape_15140 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 340, 15), b1_15139, 'shape')
        # Processing the call keyword arguments (line 340)
        kwargs_15141 = {}
        # Getting the type of 'len' (line 340)
        len_15138 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 340, 11), 'len', False)
        # Calling len(args, kwargs) (line 340)
        len_call_result_15142 = invoke(stypy.reporting.localization.Localization(__file__, 340, 11), len_15138, *[shape_15140], **kwargs_15141)
        
        int_15143 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 340, 28), 'int')
        # Applying the binary operator '!=' (line 340)
        result_ne_15144 = python_operator(stypy.reporting.localization.Localization(__file__, 340, 11), '!=', len_call_result_15142, int_15143)
        
        
        
        # Obtaining the type of the subscript
        int_15145 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 340, 42), 'int')
        # Getting the type of 'b1' (line 340)
        b1_15146 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 340, 33), 'b1')
        # Obtaining the member 'shape' of a type (line 340)
        shape_15147 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 340, 33), b1_15146, 'shape')
        # Obtaining the member '__getitem__' of a type (line 340)
        getitem___15148 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 340, 33), shape_15147, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 340)
        subscript_call_result_15149 = invoke(stypy.reporting.localization.Localization(__file__, 340, 33), getitem___15148, int_15145)
        
        
        # Obtaining the type of the subscript
        int_15150 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 340, 57), 'int')
        # Getting the type of 'b1' (line 340)
        b1_15151 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 340, 48), 'b1')
        # Obtaining the member 'shape' of a type (line 340)
        shape_15152 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 340, 48), b1_15151, 'shape')
        # Obtaining the member '__getitem__' of a type (line 340)
        getitem___15153 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 340, 48), shape_15152, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 340)
        subscript_call_result_15154 = invoke(stypy.reporting.localization.Localization(__file__, 340, 48), getitem___15153, int_15150)
        
        # Applying the binary operator '!=' (line 340)
        result_ne_15155 = python_operator(stypy.reporting.localization.Localization(__file__, 340, 33), '!=', subscript_call_result_15149, subscript_call_result_15154)
        
        # Applying the binary operator 'or' (line 340)
        result_or_keyword_15156 = python_operator(stypy.reporting.localization.Localization(__file__, 340, 11), 'or', result_ne_15144, result_ne_15155)
        
        # Testing the type of an if condition (line 340)
        if_condition_15157 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 340, 8), result_or_keyword_15156)
        # Assigning a type to the variable 'if_condition_15157' (line 340)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 340, 8), 'if_condition_15157', if_condition_15157)
        # SSA begins for if statement (line 340)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to ValueError(...): (line 341)
        # Processing the call arguments (line 341)
        str_15159 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 341, 29), 'str', 'expected square matrix')
        # Processing the call keyword arguments (line 341)
        kwargs_15160 = {}
        # Getting the type of 'ValueError' (line 341)
        ValueError_15158 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 341, 18), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 341)
        ValueError_call_result_15161 = invoke(stypy.reporting.localization.Localization(__file__, 341, 18), ValueError_15158, *[str_15159], **kwargs_15160)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 341, 12), ValueError_call_result_15161, 'raise parameter', BaseException)
        # SSA join for if statement (line 340)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Getting the type of 'b1' (line 343)
        b1_15162 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 343, 11), 'b1')
        # Obtaining the member 'shape' of a type (line 343)
        shape_15163 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 343, 11), b1_15162, 'shape')
        # Getting the type of 'a1' (line 343)
        a1_15164 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 343, 23), 'a1')
        # Obtaining the member 'shape' of a type (line 343)
        shape_15165 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 343, 23), a1_15164, 'shape')
        # Applying the binary operator '!=' (line 343)
        result_ne_15166 = python_operator(stypy.reporting.localization.Localization(__file__, 343, 11), '!=', shape_15163, shape_15165)
        
        # Testing the type of an if condition (line 343)
        if_condition_15167 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 343, 8), result_ne_15166)
        # Assigning a type to the variable 'if_condition_15167' (line 343)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 343, 8), 'if_condition_15167', if_condition_15167)
        # SSA begins for if statement (line 343)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to ValueError(...): (line 344)
        # Processing the call arguments (line 344)
        str_15169 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 344, 29), 'str', 'wrong b dimensions %s, should be %s')
        
        # Obtaining an instance of the builtin type 'tuple' (line 345)
        tuple_15170 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 345, 40), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 345)
        # Adding element type (line 345)
        
        # Call to str(...): (line 345)
        # Processing the call arguments (line 345)
        # Getting the type of 'b1' (line 345)
        b1_15172 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 345, 44), 'b1', False)
        # Obtaining the member 'shape' of a type (line 345)
        shape_15173 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 345, 44), b1_15172, 'shape')
        # Processing the call keyword arguments (line 345)
        kwargs_15174 = {}
        # Getting the type of 'str' (line 345)
        str_15171 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 345, 40), 'str', False)
        # Calling str(args, kwargs) (line 345)
        str_call_result_15175 = invoke(stypy.reporting.localization.Localization(__file__, 345, 40), str_15171, *[shape_15173], **kwargs_15174)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 345, 40), tuple_15170, str_call_result_15175)
        # Adding element type (line 345)
        
        # Call to str(...): (line 345)
        # Processing the call arguments (line 345)
        # Getting the type of 'a1' (line 345)
        a1_15177 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 345, 59), 'a1', False)
        # Obtaining the member 'shape' of a type (line 345)
        shape_15178 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 345, 59), a1_15177, 'shape')
        # Processing the call keyword arguments (line 345)
        kwargs_15179 = {}
        # Getting the type of 'str' (line 345)
        str_15176 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 345, 55), 'str', False)
        # Calling str(args, kwargs) (line 345)
        str_call_result_15180 = invoke(stypy.reporting.localization.Localization(__file__, 345, 55), str_15176, *[shape_15178], **kwargs_15179)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 345, 40), tuple_15170, str_call_result_15180)
        
        # Applying the binary operator '%' (line 344)
        result_mod_15181 = python_operator(stypy.reporting.localization.Localization(__file__, 344, 29), '%', str_15169, tuple_15170)
        
        # Processing the call keyword arguments (line 344)
        kwargs_15182 = {}
        # Getting the type of 'ValueError' (line 344)
        ValueError_15168 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 344, 18), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 344)
        ValueError_call_result_15183 = invoke(stypy.reporting.localization.Localization(__file__, 344, 18), ValueError_15168, *[result_mod_15181], **kwargs_15182)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 344, 12), ValueError_call_result_15183, 'raise parameter', BaseException)
        # SSA join for if statement (line 343)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Call to iscomplexobj(...): (line 346)
        # Processing the call arguments (line 346)
        # Getting the type of 'b1' (line 346)
        b1_15185 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 346, 24), 'b1', False)
        # Processing the call keyword arguments (line 346)
        kwargs_15186 = {}
        # Getting the type of 'iscomplexobj' (line 346)
        iscomplexobj_15184 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 346, 11), 'iscomplexobj', False)
        # Calling iscomplexobj(args, kwargs) (line 346)
        iscomplexobj_call_result_15187 = invoke(stypy.reporting.localization.Localization(__file__, 346, 11), iscomplexobj_15184, *[b1_15185], **kwargs_15186)
        
        # Testing the type of an if condition (line 346)
        if_condition_15188 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 346, 8), iscomplexobj_call_result_15187)
        # Assigning a type to the variable 'if_condition_15188' (line 346)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 346, 8), 'if_condition_15188', if_condition_15188)
        # SSA begins for if statement (line 346)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Name to a Name (line 347):
        
        # Assigning a Name to a Name (line 347):
        # Getting the type of 'True' (line 347)
        True_15189 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 347, 19), 'True')
        # Assigning a type to the variable 'cplx' (line 347)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 347, 12), 'cplx', True_15189)
        # SSA branch for the else part of an if statement (line 346)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a BoolOp to a Name (line 349):
        
        # Assigning a BoolOp to a Name (line 349):
        
        # Evaluating a boolean operation
        # Getting the type of 'cplx' (line 349)
        cplx_15190 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 349, 19), 'cplx')
        # Getting the type of 'False' (line 349)
        False_15191 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 349, 27), 'False')
        # Applying the binary operator 'or' (line 349)
        result_or_keyword_15192 = python_operator(stypy.reporting.localization.Localization(__file__, 349, 19), 'or', cplx_15190, False_15191)
        
        # Assigning a type to the variable 'cplx' (line 349)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 349, 12), 'cplx', result_or_keyword_15192)
        # SSA join for if statement (line 346)
        module_type_store = module_type_store.join_ssa_context()
        

        if more_types_in_union_15124:
            # Runtime conditional SSA for else branch (line 337)
            module_type_store.open_ssa_branch('idiom else')



    if ((not may_be_15123) or more_types_in_union_15124):
        
        # Assigning a Name to a Name (line 351):
        
        # Assigning a Name to a Name (line 351):
        # Getting the type of 'None' (line 351)
        None_15193 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 351, 13), 'None')
        # Assigning a type to the variable 'b1' (line 351)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 351, 8), 'b1', None_15193)

        if (may_be_15123 and more_types_in_union_15124):
            # SSA join for if statement (line 337)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Assigning a BoolOp to a Name (line 354):
    
    # Assigning a BoolOp to a Name (line 354):
    
    # Evaluating a boolean operation
    
    # Evaluating a boolean operation
    # Getting the type of 'eigvals_only' (line 354)
    eigvals_only_15194 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 354, 12), 'eigvals_only')
    str_15195 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 354, 29), 'str', 'N')
    # Applying the binary operator 'and' (line 354)
    result_and_keyword_15196 = python_operator(stypy.reporting.localization.Localization(__file__, 354, 12), 'and', eigvals_only_15194, str_15195)
    
    str_15197 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 354, 37), 'str', 'V')
    # Applying the binary operator 'or' (line 354)
    result_or_keyword_15198 = python_operator(stypy.reporting.localization.Localization(__file__, 354, 11), 'or', result_and_keyword_15196, str_15197)
    
    # Assigning a type to the variable '_job' (line 354)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 354, 4), '_job', result_or_keyword_15198)
    
    # Type idiom detected: calculating its left and rigth part (line 357)
    # Getting the type of 'eigvals' (line 357)
    eigvals_15199 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 357, 4), 'eigvals')
    # Getting the type of 'None' (line 357)
    None_15200 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 357, 22), 'None')
    
    (may_be_15201, more_types_in_union_15202) = may_not_be_none(eigvals_15199, None_15200)

    if may_be_15201:

        if more_types_in_union_15202:
            # Runtime conditional SSA (line 357)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a Name to a Tuple (line 358):
        
        # Assigning a Subscript to a Name (line 358):
        
        # Obtaining the type of the subscript
        int_15203 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 358, 8), 'int')
        # Getting the type of 'eigvals' (line 358)
        eigvals_15204 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 358, 17), 'eigvals')
        # Obtaining the member '__getitem__' of a type (line 358)
        getitem___15205 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 358, 8), eigvals_15204, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 358)
        subscript_call_result_15206 = invoke(stypy.reporting.localization.Localization(__file__, 358, 8), getitem___15205, int_15203)
        
        # Assigning a type to the variable 'tuple_var_assignment_14053' (line 358)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 358, 8), 'tuple_var_assignment_14053', subscript_call_result_15206)
        
        # Assigning a Subscript to a Name (line 358):
        
        # Obtaining the type of the subscript
        int_15207 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 358, 8), 'int')
        # Getting the type of 'eigvals' (line 358)
        eigvals_15208 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 358, 17), 'eigvals')
        # Obtaining the member '__getitem__' of a type (line 358)
        getitem___15209 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 358, 8), eigvals_15208, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 358)
        subscript_call_result_15210 = invoke(stypy.reporting.localization.Localization(__file__, 358, 8), getitem___15209, int_15207)
        
        # Assigning a type to the variable 'tuple_var_assignment_14054' (line 358)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 358, 8), 'tuple_var_assignment_14054', subscript_call_result_15210)
        
        # Assigning a Name to a Name (line 358):
        # Getting the type of 'tuple_var_assignment_14053' (line 358)
        tuple_var_assignment_14053_15211 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 358, 8), 'tuple_var_assignment_14053')
        # Assigning a type to the variable 'lo' (line 358)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 358, 8), 'lo', tuple_var_assignment_14053_15211)
        
        # Assigning a Name to a Name (line 358):
        # Getting the type of 'tuple_var_assignment_14054' (line 358)
        tuple_var_assignment_14054_15212 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 358, 8), 'tuple_var_assignment_14054')
        # Assigning a type to the variable 'hi' (line 358)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 358, 12), 'hi', tuple_var_assignment_14054_15212)
        
        
        # Evaluating a boolean operation
        
        # Getting the type of 'lo' (line 359)
        lo_15213 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 359, 11), 'lo')
        int_15214 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 359, 16), 'int')
        # Applying the binary operator '<' (line 359)
        result_lt_15215 = python_operator(stypy.reporting.localization.Localization(__file__, 359, 11), '<', lo_15213, int_15214)
        
        
        # Getting the type of 'hi' (line 359)
        hi_15216 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 359, 21), 'hi')
        
        # Obtaining the type of the subscript
        int_15217 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 359, 36), 'int')
        # Getting the type of 'a1' (line 359)
        a1_15218 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 359, 27), 'a1')
        # Obtaining the member 'shape' of a type (line 359)
        shape_15219 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 359, 27), a1_15218, 'shape')
        # Obtaining the member '__getitem__' of a type (line 359)
        getitem___15220 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 359, 27), shape_15219, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 359)
        subscript_call_result_15221 = invoke(stypy.reporting.localization.Localization(__file__, 359, 27), getitem___15220, int_15217)
        
        # Applying the binary operator '>=' (line 359)
        result_ge_15222 = python_operator(stypy.reporting.localization.Localization(__file__, 359, 21), '>=', hi_15216, subscript_call_result_15221)
        
        # Applying the binary operator 'or' (line 359)
        result_or_keyword_15223 = python_operator(stypy.reporting.localization.Localization(__file__, 359, 11), 'or', result_lt_15215, result_ge_15222)
        
        # Testing the type of an if condition (line 359)
        if_condition_15224 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 359, 8), result_or_keyword_15223)
        # Assigning a type to the variable 'if_condition_15224' (line 359)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 359, 8), 'if_condition_15224', if_condition_15224)
        # SSA begins for if statement (line 359)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to ValueError(...): (line 360)
        # Processing the call arguments (line 360)
        str_15226 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 360, 29), 'str', 'The eigenvalue range specified is not valid.\nValid range is [%s,%s]')
        
        # Obtaining an instance of the builtin type 'tuple' (line 361)
        tuple_15227 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 361, 57), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 361)
        # Adding element type (line 361)
        int_15228 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 361, 57), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 361, 57), tuple_15227, int_15228)
        # Adding element type (line 361)
        
        # Obtaining the type of the subscript
        int_15229 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 361, 69), 'int')
        # Getting the type of 'a1' (line 361)
        a1_15230 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 361, 60), 'a1', False)
        # Obtaining the member 'shape' of a type (line 361)
        shape_15231 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 361, 60), a1_15230, 'shape')
        # Obtaining the member '__getitem__' of a type (line 361)
        getitem___15232 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 361, 60), shape_15231, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 361)
        subscript_call_result_15233 = invoke(stypy.reporting.localization.Localization(__file__, 361, 60), getitem___15232, int_15229)
        
        int_15234 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 361, 72), 'int')
        # Applying the binary operator '-' (line 361)
        result_sub_15235 = python_operator(stypy.reporting.localization.Localization(__file__, 361, 60), '-', subscript_call_result_15233, int_15234)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 361, 57), tuple_15227, result_sub_15235)
        
        # Applying the binary operator '%' (line 360)
        result_mod_15236 = python_operator(stypy.reporting.localization.Localization(__file__, 360, 29), '%', str_15226, tuple_15227)
        
        # Processing the call keyword arguments (line 360)
        kwargs_15237 = {}
        # Getting the type of 'ValueError' (line 360)
        ValueError_15225 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 360, 18), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 360)
        ValueError_call_result_15238 = invoke(stypy.reporting.localization.Localization(__file__, 360, 18), ValueError_15225, *[result_mod_15236], **kwargs_15237)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 360, 12), ValueError_call_result_15238, 'raise parameter', BaseException)
        # SSA join for if statement (line 359)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Getting the type of 'lo' (line 362)
        lo_15239 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 362, 8), 'lo')
        int_15240 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 362, 14), 'int')
        # Applying the binary operator '+=' (line 362)
        result_iadd_15241 = python_operator(stypy.reporting.localization.Localization(__file__, 362, 8), '+=', lo_15239, int_15240)
        # Assigning a type to the variable 'lo' (line 362)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 362, 8), 'lo', result_iadd_15241)
        
        
        # Getting the type of 'hi' (line 363)
        hi_15242 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 363, 8), 'hi')
        int_15243 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 363, 14), 'int')
        # Applying the binary operator '+=' (line 363)
        result_iadd_15244 = python_operator(stypy.reporting.localization.Localization(__file__, 363, 8), '+=', hi_15242, int_15243)
        # Assigning a type to the variable 'hi' (line 363)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 363, 8), 'hi', result_iadd_15244)
        
        
        # Assigning a Tuple to a Name (line 364):
        
        # Assigning a Tuple to a Name (line 364):
        
        # Obtaining an instance of the builtin type 'tuple' (line 364)
        tuple_15245 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 364, 19), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 364)
        # Adding element type (line 364)
        # Getting the type of 'lo' (line 364)
        lo_15246 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 364, 19), 'lo')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 364, 19), tuple_15245, lo_15246)
        # Adding element type (line 364)
        # Getting the type of 'hi' (line 364)
        hi_15247 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 364, 23), 'hi')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 364, 19), tuple_15245, hi_15247)
        
        # Assigning a type to the variable 'eigvals' (line 364)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 364, 8), 'eigvals', tuple_15245)

        if more_types_in_union_15202:
            # SSA join for if statement (line 357)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Getting the type of 'lower' (line 367)
    lower_15248 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 367, 7), 'lower')
    # Testing the type of an if condition (line 367)
    if_condition_15249 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 367, 4), lower_15248)
    # Assigning a type to the variable 'if_condition_15249' (line 367)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 367, 4), 'if_condition_15249', if_condition_15249)
    # SSA begins for if statement (line 367)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Str to a Name (line 368):
    
    # Assigning a Str to a Name (line 368):
    str_15250 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 368, 15), 'str', 'L')
    # Assigning a type to the variable 'uplo' (line 368)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 368, 8), 'uplo', str_15250)
    # SSA branch for the else part of an if statement (line 367)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Str to a Name (line 370):
    
    # Assigning a Str to a Name (line 370):
    str_15251 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 370, 15), 'str', 'U')
    # Assigning a type to the variable 'uplo' (line 370)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 370, 8), 'uplo', str_15251)
    # SSA join for if statement (line 367)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Getting the type of 'cplx' (line 373)
    cplx_15252 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 373, 7), 'cplx')
    # Testing the type of an if condition (line 373)
    if_condition_15253 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 373, 4), cplx_15252)
    # Assigning a type to the variable 'if_condition_15253' (line 373)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 373, 4), 'if_condition_15253', if_condition_15253)
    # SSA begins for if statement (line 373)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Str to a Name (line 374):
    
    # Assigning a Str to a Name (line 374):
    str_15254 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 374, 14), 'str', 'he')
    # Assigning a type to the variable 'pfx' (line 374)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 374, 8), 'pfx', str_15254)
    # SSA branch for the else part of an if statement (line 373)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Str to a Name (line 376):
    
    # Assigning a Str to a Name (line 376):
    str_15255 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 376, 14), 'str', 'sy')
    # Assigning a type to the variable 'pfx' (line 376)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 376, 8), 'pfx', str_15255)
    # SSA join for if statement (line 373)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Type idiom detected: calculating its left and rigth part (line 382)
    # Getting the type of 'b1' (line 382)
    b1_15256 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 382, 7), 'b1')
    # Getting the type of 'None' (line 382)
    None_15257 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 382, 13), 'None')
    
    (may_be_15258, more_types_in_union_15259) = may_be_none(b1_15256, None_15257)

    if may_be_15258:

        if more_types_in_union_15259:
            # Runtime conditional SSA (line 382)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a BinOp to a Name (line 383):
        
        # Assigning a BinOp to a Name (line 383):
        # Getting the type of 'pfx' (line 383)
        pfx_15260 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 383, 17), 'pfx')
        str_15261 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 383, 21), 'str', 'evr')
        # Applying the binary operator '+' (line 383)
        result_add_15262 = python_operator(stypy.reporting.localization.Localization(__file__, 383, 17), '+', pfx_15260, str_15261)
        
        # Assigning a type to the variable 'driver' (line 383)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 383, 8), 'driver', result_add_15262)
        
        # Assigning a Call to a Tuple (line 384):
        
        # Assigning a Subscript to a Name (line 384):
        
        # Obtaining the type of the subscript
        int_15263 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 384, 8), 'int')
        
        # Call to get_lapack_funcs(...): (line 384)
        # Processing the call arguments (line 384)
        
        # Obtaining an instance of the builtin type 'tuple' (line 384)
        tuple_15265 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 384, 35), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 384)
        # Adding element type (line 384)
        # Getting the type of 'driver' (line 384)
        driver_15266 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 384, 35), 'driver', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 384, 35), tuple_15265, driver_15266)
        
        
        # Obtaining an instance of the builtin type 'tuple' (line 384)
        tuple_15267 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 384, 46), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 384)
        # Adding element type (line 384)
        # Getting the type of 'a1' (line 384)
        a1_15268 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 384, 46), 'a1', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 384, 46), tuple_15267, a1_15268)
        
        # Processing the call keyword arguments (line 384)
        kwargs_15269 = {}
        # Getting the type of 'get_lapack_funcs' (line 384)
        get_lapack_funcs_15264 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 384, 17), 'get_lapack_funcs', False)
        # Calling get_lapack_funcs(args, kwargs) (line 384)
        get_lapack_funcs_call_result_15270 = invoke(stypy.reporting.localization.Localization(__file__, 384, 17), get_lapack_funcs_15264, *[tuple_15265, tuple_15267], **kwargs_15269)
        
        # Obtaining the member '__getitem__' of a type (line 384)
        getitem___15271 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 384, 8), get_lapack_funcs_call_result_15270, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 384)
        subscript_call_result_15272 = invoke(stypy.reporting.localization.Localization(__file__, 384, 8), getitem___15271, int_15263)
        
        # Assigning a type to the variable 'tuple_var_assignment_14055' (line 384)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 384, 8), 'tuple_var_assignment_14055', subscript_call_result_15272)
        
        # Assigning a Name to a Name (line 384):
        # Getting the type of 'tuple_var_assignment_14055' (line 384)
        tuple_var_assignment_14055_15273 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 384, 8), 'tuple_var_assignment_14055')
        # Assigning a type to the variable 'evr' (line 384)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 384, 9), 'evr', tuple_var_assignment_14055_15273)
        
        # Type idiom detected: calculating its left and rigth part (line 385)
        # Getting the type of 'eigvals' (line 385)
        eigvals_15274 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 385, 11), 'eigvals')
        # Getting the type of 'None' (line 385)
        None_15275 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 385, 22), 'None')
        
        (may_be_15276, more_types_in_union_15277) = may_be_none(eigvals_15274, None_15275)

        if may_be_15276:

            if more_types_in_union_15277:
                # Runtime conditional SSA (line 385)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a Call to a Tuple (line 386):
            
            # Assigning a Subscript to a Name (line 386):
            
            # Obtaining the type of the subscript
            int_15278 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 386, 12), 'int')
            
            # Call to evr(...): (line 386)
            # Processing the call arguments (line 386)
            # Getting the type of 'a1' (line 386)
            a1_15280 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 386, 29), 'a1', False)
            # Processing the call keyword arguments (line 386)
            # Getting the type of 'uplo' (line 386)
            uplo_15281 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 386, 38), 'uplo', False)
            keyword_15282 = uplo_15281
            # Getting the type of '_job' (line 386)
            _job_15283 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 386, 49), '_job', False)
            keyword_15284 = _job_15283
            str_15285 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 386, 61), 'str', 'A')
            keyword_15286 = str_15285
            int_15287 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 386, 69), 'int')
            keyword_15288 = int_15287
            
            # Obtaining the type of the subscript
            int_15289 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 387, 41), 'int')
            # Getting the type of 'a1' (line 387)
            a1_15290 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 387, 32), 'a1', False)
            # Obtaining the member 'shape' of a type (line 387)
            shape_15291 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 387, 32), a1_15290, 'shape')
            # Obtaining the member '__getitem__' of a type (line 387)
            getitem___15292 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 387, 32), shape_15291, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 387)
            subscript_call_result_15293 = invoke(stypy.reporting.localization.Localization(__file__, 387, 32), getitem___15292, int_15289)
            
            keyword_15294 = subscript_call_result_15293
            # Getting the type of 'overwrite_a' (line 387)
            overwrite_a_15295 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 387, 57), 'overwrite_a', False)
            keyword_15296 = overwrite_a_15295
            kwargs_15297 = {'overwrite_a': keyword_15296, 'uplo': keyword_15282, 'iu': keyword_15294, 'jobz': keyword_15284, 'range': keyword_15286, 'il': keyword_15288}
            # Getting the type of 'evr' (line 386)
            evr_15279 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 386, 25), 'evr', False)
            # Calling evr(args, kwargs) (line 386)
            evr_call_result_15298 = invoke(stypy.reporting.localization.Localization(__file__, 386, 25), evr_15279, *[a1_15280], **kwargs_15297)
            
            # Obtaining the member '__getitem__' of a type (line 386)
            getitem___15299 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 386, 12), evr_call_result_15298, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 386)
            subscript_call_result_15300 = invoke(stypy.reporting.localization.Localization(__file__, 386, 12), getitem___15299, int_15278)
            
            # Assigning a type to the variable 'tuple_var_assignment_14056' (line 386)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 386, 12), 'tuple_var_assignment_14056', subscript_call_result_15300)
            
            # Assigning a Subscript to a Name (line 386):
            
            # Obtaining the type of the subscript
            int_15301 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 386, 12), 'int')
            
            # Call to evr(...): (line 386)
            # Processing the call arguments (line 386)
            # Getting the type of 'a1' (line 386)
            a1_15303 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 386, 29), 'a1', False)
            # Processing the call keyword arguments (line 386)
            # Getting the type of 'uplo' (line 386)
            uplo_15304 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 386, 38), 'uplo', False)
            keyword_15305 = uplo_15304
            # Getting the type of '_job' (line 386)
            _job_15306 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 386, 49), '_job', False)
            keyword_15307 = _job_15306
            str_15308 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 386, 61), 'str', 'A')
            keyword_15309 = str_15308
            int_15310 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 386, 69), 'int')
            keyword_15311 = int_15310
            
            # Obtaining the type of the subscript
            int_15312 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 387, 41), 'int')
            # Getting the type of 'a1' (line 387)
            a1_15313 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 387, 32), 'a1', False)
            # Obtaining the member 'shape' of a type (line 387)
            shape_15314 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 387, 32), a1_15313, 'shape')
            # Obtaining the member '__getitem__' of a type (line 387)
            getitem___15315 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 387, 32), shape_15314, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 387)
            subscript_call_result_15316 = invoke(stypy.reporting.localization.Localization(__file__, 387, 32), getitem___15315, int_15312)
            
            keyword_15317 = subscript_call_result_15316
            # Getting the type of 'overwrite_a' (line 387)
            overwrite_a_15318 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 387, 57), 'overwrite_a', False)
            keyword_15319 = overwrite_a_15318
            kwargs_15320 = {'overwrite_a': keyword_15319, 'uplo': keyword_15305, 'iu': keyword_15317, 'jobz': keyword_15307, 'range': keyword_15309, 'il': keyword_15311}
            # Getting the type of 'evr' (line 386)
            evr_15302 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 386, 25), 'evr', False)
            # Calling evr(args, kwargs) (line 386)
            evr_call_result_15321 = invoke(stypy.reporting.localization.Localization(__file__, 386, 25), evr_15302, *[a1_15303], **kwargs_15320)
            
            # Obtaining the member '__getitem__' of a type (line 386)
            getitem___15322 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 386, 12), evr_call_result_15321, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 386)
            subscript_call_result_15323 = invoke(stypy.reporting.localization.Localization(__file__, 386, 12), getitem___15322, int_15301)
            
            # Assigning a type to the variable 'tuple_var_assignment_14057' (line 386)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 386, 12), 'tuple_var_assignment_14057', subscript_call_result_15323)
            
            # Assigning a Subscript to a Name (line 386):
            
            # Obtaining the type of the subscript
            int_15324 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 386, 12), 'int')
            
            # Call to evr(...): (line 386)
            # Processing the call arguments (line 386)
            # Getting the type of 'a1' (line 386)
            a1_15326 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 386, 29), 'a1', False)
            # Processing the call keyword arguments (line 386)
            # Getting the type of 'uplo' (line 386)
            uplo_15327 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 386, 38), 'uplo', False)
            keyword_15328 = uplo_15327
            # Getting the type of '_job' (line 386)
            _job_15329 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 386, 49), '_job', False)
            keyword_15330 = _job_15329
            str_15331 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 386, 61), 'str', 'A')
            keyword_15332 = str_15331
            int_15333 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 386, 69), 'int')
            keyword_15334 = int_15333
            
            # Obtaining the type of the subscript
            int_15335 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 387, 41), 'int')
            # Getting the type of 'a1' (line 387)
            a1_15336 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 387, 32), 'a1', False)
            # Obtaining the member 'shape' of a type (line 387)
            shape_15337 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 387, 32), a1_15336, 'shape')
            # Obtaining the member '__getitem__' of a type (line 387)
            getitem___15338 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 387, 32), shape_15337, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 387)
            subscript_call_result_15339 = invoke(stypy.reporting.localization.Localization(__file__, 387, 32), getitem___15338, int_15335)
            
            keyword_15340 = subscript_call_result_15339
            # Getting the type of 'overwrite_a' (line 387)
            overwrite_a_15341 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 387, 57), 'overwrite_a', False)
            keyword_15342 = overwrite_a_15341
            kwargs_15343 = {'overwrite_a': keyword_15342, 'uplo': keyword_15328, 'iu': keyword_15340, 'jobz': keyword_15330, 'range': keyword_15332, 'il': keyword_15334}
            # Getting the type of 'evr' (line 386)
            evr_15325 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 386, 25), 'evr', False)
            # Calling evr(args, kwargs) (line 386)
            evr_call_result_15344 = invoke(stypy.reporting.localization.Localization(__file__, 386, 25), evr_15325, *[a1_15326], **kwargs_15343)
            
            # Obtaining the member '__getitem__' of a type (line 386)
            getitem___15345 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 386, 12), evr_call_result_15344, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 386)
            subscript_call_result_15346 = invoke(stypy.reporting.localization.Localization(__file__, 386, 12), getitem___15345, int_15324)
            
            # Assigning a type to the variable 'tuple_var_assignment_14058' (line 386)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 386, 12), 'tuple_var_assignment_14058', subscript_call_result_15346)
            
            # Assigning a Name to a Name (line 386):
            # Getting the type of 'tuple_var_assignment_14056' (line 386)
            tuple_var_assignment_14056_15347 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 386, 12), 'tuple_var_assignment_14056')
            # Assigning a type to the variable 'w' (line 386)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 386, 12), 'w', tuple_var_assignment_14056_15347)
            
            # Assigning a Name to a Name (line 386):
            # Getting the type of 'tuple_var_assignment_14057' (line 386)
            tuple_var_assignment_14057_15348 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 386, 12), 'tuple_var_assignment_14057')
            # Assigning a type to the variable 'v' (line 386)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 386, 15), 'v', tuple_var_assignment_14057_15348)
            
            # Assigning a Name to a Name (line 386):
            # Getting the type of 'tuple_var_assignment_14058' (line 386)
            tuple_var_assignment_14058_15349 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 386, 12), 'tuple_var_assignment_14058')
            # Assigning a type to the variable 'info' (line 386)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 386, 18), 'info', tuple_var_assignment_14058_15349)

            if more_types_in_union_15277:
                # Runtime conditional SSA for else branch (line 385)
                module_type_store.open_ssa_branch('idiom else')



        if ((not may_be_15276) or more_types_in_union_15277):
            
            # Assigning a Name to a Tuple (line 389):
            
            # Assigning a Subscript to a Name (line 389):
            
            # Obtaining the type of the subscript
            int_15350 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 389, 12), 'int')
            # Getting the type of 'eigvals' (line 389)
            eigvals_15351 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 389, 23), 'eigvals')
            # Obtaining the member '__getitem__' of a type (line 389)
            getitem___15352 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 389, 12), eigvals_15351, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 389)
            subscript_call_result_15353 = invoke(stypy.reporting.localization.Localization(__file__, 389, 12), getitem___15352, int_15350)
            
            # Assigning a type to the variable 'tuple_var_assignment_14059' (line 389)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 389, 12), 'tuple_var_assignment_14059', subscript_call_result_15353)
            
            # Assigning a Subscript to a Name (line 389):
            
            # Obtaining the type of the subscript
            int_15354 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 389, 12), 'int')
            # Getting the type of 'eigvals' (line 389)
            eigvals_15355 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 389, 23), 'eigvals')
            # Obtaining the member '__getitem__' of a type (line 389)
            getitem___15356 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 389, 12), eigvals_15355, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 389)
            subscript_call_result_15357 = invoke(stypy.reporting.localization.Localization(__file__, 389, 12), getitem___15356, int_15354)
            
            # Assigning a type to the variable 'tuple_var_assignment_14060' (line 389)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 389, 12), 'tuple_var_assignment_14060', subscript_call_result_15357)
            
            # Assigning a Name to a Name (line 389):
            # Getting the type of 'tuple_var_assignment_14059' (line 389)
            tuple_var_assignment_14059_15358 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 389, 12), 'tuple_var_assignment_14059')
            # Assigning a type to the variable 'lo' (line 389)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 389, 13), 'lo', tuple_var_assignment_14059_15358)
            
            # Assigning a Name to a Name (line 389):
            # Getting the type of 'tuple_var_assignment_14060' (line 389)
            tuple_var_assignment_14060_15359 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 389, 12), 'tuple_var_assignment_14060')
            # Assigning a type to the variable 'hi' (line 389)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 389, 17), 'hi', tuple_var_assignment_14060_15359)
            
            # Assigning a Call to a Tuple (line 390):
            
            # Assigning a Subscript to a Name (line 390):
            
            # Obtaining the type of the subscript
            int_15360 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 390, 12), 'int')
            
            # Call to evr(...): (line 390)
            # Processing the call arguments (line 390)
            # Getting the type of 'a1' (line 390)
            a1_15362 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 390, 33), 'a1', False)
            # Processing the call keyword arguments (line 390)
            # Getting the type of 'uplo' (line 390)
            uplo_15363 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 390, 42), 'uplo', False)
            keyword_15364 = uplo_15363
            # Getting the type of '_job' (line 390)
            _job_15365 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 390, 53), '_job', False)
            keyword_15366 = _job_15365
            str_15367 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 390, 65), 'str', 'I')
            keyword_15368 = str_15367
            # Getting the type of 'lo' (line 391)
            lo_15369 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 391, 36), 'lo', False)
            keyword_15370 = lo_15369
            # Getting the type of 'hi' (line 391)
            hi_15371 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 391, 43), 'hi', False)
            keyword_15372 = hi_15371
            # Getting the type of 'overwrite_a' (line 391)
            overwrite_a_15373 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 391, 59), 'overwrite_a', False)
            keyword_15374 = overwrite_a_15373
            kwargs_15375 = {'overwrite_a': keyword_15374, 'uplo': keyword_15364, 'iu': keyword_15372, 'jobz': keyword_15366, 'range': keyword_15368, 'il': keyword_15370}
            # Getting the type of 'evr' (line 390)
            evr_15361 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 390, 29), 'evr', False)
            # Calling evr(args, kwargs) (line 390)
            evr_call_result_15376 = invoke(stypy.reporting.localization.Localization(__file__, 390, 29), evr_15361, *[a1_15362], **kwargs_15375)
            
            # Obtaining the member '__getitem__' of a type (line 390)
            getitem___15377 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 390, 12), evr_call_result_15376, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 390)
            subscript_call_result_15378 = invoke(stypy.reporting.localization.Localization(__file__, 390, 12), getitem___15377, int_15360)
            
            # Assigning a type to the variable 'tuple_var_assignment_14061' (line 390)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 390, 12), 'tuple_var_assignment_14061', subscript_call_result_15378)
            
            # Assigning a Subscript to a Name (line 390):
            
            # Obtaining the type of the subscript
            int_15379 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 390, 12), 'int')
            
            # Call to evr(...): (line 390)
            # Processing the call arguments (line 390)
            # Getting the type of 'a1' (line 390)
            a1_15381 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 390, 33), 'a1', False)
            # Processing the call keyword arguments (line 390)
            # Getting the type of 'uplo' (line 390)
            uplo_15382 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 390, 42), 'uplo', False)
            keyword_15383 = uplo_15382
            # Getting the type of '_job' (line 390)
            _job_15384 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 390, 53), '_job', False)
            keyword_15385 = _job_15384
            str_15386 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 390, 65), 'str', 'I')
            keyword_15387 = str_15386
            # Getting the type of 'lo' (line 391)
            lo_15388 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 391, 36), 'lo', False)
            keyword_15389 = lo_15388
            # Getting the type of 'hi' (line 391)
            hi_15390 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 391, 43), 'hi', False)
            keyword_15391 = hi_15390
            # Getting the type of 'overwrite_a' (line 391)
            overwrite_a_15392 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 391, 59), 'overwrite_a', False)
            keyword_15393 = overwrite_a_15392
            kwargs_15394 = {'overwrite_a': keyword_15393, 'uplo': keyword_15383, 'iu': keyword_15391, 'jobz': keyword_15385, 'range': keyword_15387, 'il': keyword_15389}
            # Getting the type of 'evr' (line 390)
            evr_15380 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 390, 29), 'evr', False)
            # Calling evr(args, kwargs) (line 390)
            evr_call_result_15395 = invoke(stypy.reporting.localization.Localization(__file__, 390, 29), evr_15380, *[a1_15381], **kwargs_15394)
            
            # Obtaining the member '__getitem__' of a type (line 390)
            getitem___15396 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 390, 12), evr_call_result_15395, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 390)
            subscript_call_result_15397 = invoke(stypy.reporting.localization.Localization(__file__, 390, 12), getitem___15396, int_15379)
            
            # Assigning a type to the variable 'tuple_var_assignment_14062' (line 390)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 390, 12), 'tuple_var_assignment_14062', subscript_call_result_15397)
            
            # Assigning a Subscript to a Name (line 390):
            
            # Obtaining the type of the subscript
            int_15398 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 390, 12), 'int')
            
            # Call to evr(...): (line 390)
            # Processing the call arguments (line 390)
            # Getting the type of 'a1' (line 390)
            a1_15400 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 390, 33), 'a1', False)
            # Processing the call keyword arguments (line 390)
            # Getting the type of 'uplo' (line 390)
            uplo_15401 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 390, 42), 'uplo', False)
            keyword_15402 = uplo_15401
            # Getting the type of '_job' (line 390)
            _job_15403 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 390, 53), '_job', False)
            keyword_15404 = _job_15403
            str_15405 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 390, 65), 'str', 'I')
            keyword_15406 = str_15405
            # Getting the type of 'lo' (line 391)
            lo_15407 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 391, 36), 'lo', False)
            keyword_15408 = lo_15407
            # Getting the type of 'hi' (line 391)
            hi_15409 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 391, 43), 'hi', False)
            keyword_15410 = hi_15409
            # Getting the type of 'overwrite_a' (line 391)
            overwrite_a_15411 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 391, 59), 'overwrite_a', False)
            keyword_15412 = overwrite_a_15411
            kwargs_15413 = {'overwrite_a': keyword_15412, 'uplo': keyword_15402, 'iu': keyword_15410, 'jobz': keyword_15404, 'range': keyword_15406, 'il': keyword_15408}
            # Getting the type of 'evr' (line 390)
            evr_15399 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 390, 29), 'evr', False)
            # Calling evr(args, kwargs) (line 390)
            evr_call_result_15414 = invoke(stypy.reporting.localization.Localization(__file__, 390, 29), evr_15399, *[a1_15400], **kwargs_15413)
            
            # Obtaining the member '__getitem__' of a type (line 390)
            getitem___15415 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 390, 12), evr_call_result_15414, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 390)
            subscript_call_result_15416 = invoke(stypy.reporting.localization.Localization(__file__, 390, 12), getitem___15415, int_15398)
            
            # Assigning a type to the variable 'tuple_var_assignment_14063' (line 390)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 390, 12), 'tuple_var_assignment_14063', subscript_call_result_15416)
            
            # Assigning a Name to a Name (line 390):
            # Getting the type of 'tuple_var_assignment_14061' (line 390)
            tuple_var_assignment_14061_15417 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 390, 12), 'tuple_var_assignment_14061')
            # Assigning a type to the variable 'w_tot' (line 390)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 390, 12), 'w_tot', tuple_var_assignment_14061_15417)
            
            # Assigning a Name to a Name (line 390):
            # Getting the type of 'tuple_var_assignment_14062' (line 390)
            tuple_var_assignment_14062_15418 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 390, 12), 'tuple_var_assignment_14062')
            # Assigning a type to the variable 'v' (line 390)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 390, 19), 'v', tuple_var_assignment_14062_15418)
            
            # Assigning a Name to a Name (line 390):
            # Getting the type of 'tuple_var_assignment_14063' (line 390)
            tuple_var_assignment_14063_15419 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 390, 12), 'tuple_var_assignment_14063')
            # Assigning a type to the variable 'info' (line 390)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 390, 22), 'info', tuple_var_assignment_14063_15419)
            
            # Assigning a Subscript to a Name (line 392):
            
            # Assigning a Subscript to a Name (line 392):
            
            # Obtaining the type of the subscript
            int_15420 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 392, 22), 'int')
            # Getting the type of 'hi' (line 392)
            hi_15421 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 392, 24), 'hi')
            # Getting the type of 'lo' (line 392)
            lo_15422 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 392, 27), 'lo')
            # Applying the binary operator '-' (line 392)
            result_sub_15423 = python_operator(stypy.reporting.localization.Localization(__file__, 392, 24), '-', hi_15421, lo_15422)
            
            int_15424 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 392, 30), 'int')
            # Applying the binary operator '+' (line 392)
            result_add_15425 = python_operator(stypy.reporting.localization.Localization(__file__, 392, 29), '+', result_sub_15423, int_15424)
            
            slice_15426 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 392, 16), int_15420, result_add_15425, None)
            # Getting the type of 'w_tot' (line 392)
            w_tot_15427 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 392, 16), 'w_tot')
            # Obtaining the member '__getitem__' of a type (line 392)
            getitem___15428 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 392, 16), w_tot_15427, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 392)
            subscript_call_result_15429 = invoke(stypy.reporting.localization.Localization(__file__, 392, 16), getitem___15428, slice_15426)
            
            # Assigning a type to the variable 'w' (line 392)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 392, 12), 'w', subscript_call_result_15429)

            if (may_be_15276 and more_types_in_union_15277):
                # SSA join for if statement (line 385)
                module_type_store = module_type_store.join_ssa_context()


        

        if more_types_in_union_15259:
            # Runtime conditional SSA for else branch (line 382)
            module_type_store.open_ssa_branch('idiom else')



    if ((not may_be_15258) or more_types_in_union_15259):
        
        # Type idiom detected: calculating its left and rigth part (line 397)
        # Getting the type of 'eigvals' (line 397)
        eigvals_15430 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 397, 8), 'eigvals')
        # Getting the type of 'None' (line 397)
        None_15431 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 397, 26), 'None')
        
        (may_be_15432, more_types_in_union_15433) = may_not_be_none(eigvals_15430, None_15431)

        if may_be_15432:

            if more_types_in_union_15433:
                # Runtime conditional SSA (line 397)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a BinOp to a Name (line 398):
            
            # Assigning a BinOp to a Name (line 398):
            # Getting the type of 'pfx' (line 398)
            pfx_15434 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 398, 21), 'pfx')
            str_15435 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 398, 25), 'str', 'gvx')
            # Applying the binary operator '+' (line 398)
            result_add_15436 = python_operator(stypy.reporting.localization.Localization(__file__, 398, 21), '+', pfx_15434, str_15435)
            
            # Assigning a type to the variable 'driver' (line 398)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 398, 12), 'driver', result_add_15436)
            
            # Assigning a Call to a Tuple (line 399):
            
            # Assigning a Subscript to a Name (line 399):
            
            # Obtaining the type of the subscript
            int_15437 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 399, 12), 'int')
            
            # Call to get_lapack_funcs(...): (line 399)
            # Processing the call arguments (line 399)
            
            # Obtaining an instance of the builtin type 'tuple' (line 399)
            tuple_15439 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 399, 39), 'tuple')
            # Adding type elements to the builtin type 'tuple' instance (line 399)
            # Adding element type (line 399)
            # Getting the type of 'driver' (line 399)
            driver_15440 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 399, 39), 'driver', False)
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 399, 39), tuple_15439, driver_15440)
            
            
            # Obtaining an instance of the builtin type 'tuple' (line 399)
            tuple_15441 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 399, 50), 'tuple')
            # Adding type elements to the builtin type 'tuple' instance (line 399)
            # Adding element type (line 399)
            # Getting the type of 'a1' (line 399)
            a1_15442 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 399, 50), 'a1', False)
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 399, 50), tuple_15441, a1_15442)
            # Adding element type (line 399)
            # Getting the type of 'b1' (line 399)
            b1_15443 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 399, 54), 'b1', False)
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 399, 50), tuple_15441, b1_15443)
            
            # Processing the call keyword arguments (line 399)
            kwargs_15444 = {}
            # Getting the type of 'get_lapack_funcs' (line 399)
            get_lapack_funcs_15438 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 399, 21), 'get_lapack_funcs', False)
            # Calling get_lapack_funcs(args, kwargs) (line 399)
            get_lapack_funcs_call_result_15445 = invoke(stypy.reporting.localization.Localization(__file__, 399, 21), get_lapack_funcs_15438, *[tuple_15439, tuple_15441], **kwargs_15444)
            
            # Obtaining the member '__getitem__' of a type (line 399)
            getitem___15446 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 399, 12), get_lapack_funcs_call_result_15445, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 399)
            subscript_call_result_15447 = invoke(stypy.reporting.localization.Localization(__file__, 399, 12), getitem___15446, int_15437)
            
            # Assigning a type to the variable 'tuple_var_assignment_14064' (line 399)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 399, 12), 'tuple_var_assignment_14064', subscript_call_result_15447)
            
            # Assigning a Name to a Name (line 399):
            # Getting the type of 'tuple_var_assignment_14064' (line 399)
            tuple_var_assignment_14064_15448 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 399, 12), 'tuple_var_assignment_14064')
            # Assigning a type to the variable 'gvx' (line 399)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 399, 13), 'gvx', tuple_var_assignment_14064_15448)
            
            # Assigning a Name to a Tuple (line 400):
            
            # Assigning a Subscript to a Name (line 400):
            
            # Obtaining the type of the subscript
            int_15449 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 400, 12), 'int')
            # Getting the type of 'eigvals' (line 400)
            eigvals_15450 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 400, 23), 'eigvals')
            # Obtaining the member '__getitem__' of a type (line 400)
            getitem___15451 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 400, 12), eigvals_15450, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 400)
            subscript_call_result_15452 = invoke(stypy.reporting.localization.Localization(__file__, 400, 12), getitem___15451, int_15449)
            
            # Assigning a type to the variable 'tuple_var_assignment_14065' (line 400)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 400, 12), 'tuple_var_assignment_14065', subscript_call_result_15452)
            
            # Assigning a Subscript to a Name (line 400):
            
            # Obtaining the type of the subscript
            int_15453 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 400, 12), 'int')
            # Getting the type of 'eigvals' (line 400)
            eigvals_15454 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 400, 23), 'eigvals')
            # Obtaining the member '__getitem__' of a type (line 400)
            getitem___15455 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 400, 12), eigvals_15454, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 400)
            subscript_call_result_15456 = invoke(stypy.reporting.localization.Localization(__file__, 400, 12), getitem___15455, int_15453)
            
            # Assigning a type to the variable 'tuple_var_assignment_14066' (line 400)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 400, 12), 'tuple_var_assignment_14066', subscript_call_result_15456)
            
            # Assigning a Name to a Name (line 400):
            # Getting the type of 'tuple_var_assignment_14065' (line 400)
            tuple_var_assignment_14065_15457 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 400, 12), 'tuple_var_assignment_14065')
            # Assigning a type to the variable 'lo' (line 400)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 400, 13), 'lo', tuple_var_assignment_14065_15457)
            
            # Assigning a Name to a Name (line 400):
            # Getting the type of 'tuple_var_assignment_14066' (line 400)
            tuple_var_assignment_14066_15458 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 400, 12), 'tuple_var_assignment_14066')
            # Assigning a type to the variable 'hi' (line 400)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 400, 17), 'hi', tuple_var_assignment_14066_15458)
            
            # Assigning a Call to a Tuple (line 401):
            
            # Assigning a Subscript to a Name (line 401):
            
            # Obtaining the type of the subscript
            int_15459 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 401, 12), 'int')
            
            # Call to gvx(...): (line 401)
            # Processing the call arguments (line 401)
            # Getting the type of 'a1' (line 401)
            a1_15461 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 401, 40), 'a1', False)
            # Getting the type of 'b1' (line 401)
            b1_15462 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 401, 44), 'b1', False)
            # Processing the call keyword arguments (line 401)
            # Getting the type of 'uplo' (line 401)
            uplo_15463 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 401, 53), 'uplo', False)
            keyword_15464 = uplo_15463
            # Getting the type of 'hi' (line 401)
            hi_15465 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 401, 62), 'hi', False)
            keyword_15466 = hi_15465
            # Getting the type of 'type' (line 402)
            type_15467 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 402, 46), 'type', False)
            keyword_15468 = type_15467
            # Getting the type of '_job' (line 402)
            _job_15469 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 402, 57), '_job', False)
            keyword_15470 = _job_15469
            # Getting the type of 'lo' (line 402)
            lo_15471 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 402, 66), 'lo', False)
            keyword_15472 = lo_15471
            # Getting the type of 'overwrite_a' (line 403)
            overwrite_a_15473 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 403, 52), 'overwrite_a', False)
            keyword_15474 = overwrite_a_15473
            # Getting the type of 'overwrite_b' (line 404)
            overwrite_b_15475 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 404, 52), 'overwrite_b', False)
            keyword_15476 = overwrite_b_15475
            kwargs_15477 = {'itype': keyword_15468, 'overwrite_b': keyword_15476, 'uplo': keyword_15464, 'iu': keyword_15466, 'jobz': keyword_15470, 'il': keyword_15472, 'overwrite_a': keyword_15474}
            # Getting the type of 'gvx' (line 401)
            gvx_15460 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 401, 36), 'gvx', False)
            # Calling gvx(args, kwargs) (line 401)
            gvx_call_result_15478 = invoke(stypy.reporting.localization.Localization(__file__, 401, 36), gvx_15460, *[a1_15461, b1_15462], **kwargs_15477)
            
            # Obtaining the member '__getitem__' of a type (line 401)
            getitem___15479 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 401, 12), gvx_call_result_15478, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 401)
            subscript_call_result_15480 = invoke(stypy.reporting.localization.Localization(__file__, 401, 12), getitem___15479, int_15459)
            
            # Assigning a type to the variable 'tuple_var_assignment_14067' (line 401)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 401, 12), 'tuple_var_assignment_14067', subscript_call_result_15480)
            
            # Assigning a Subscript to a Name (line 401):
            
            # Obtaining the type of the subscript
            int_15481 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 401, 12), 'int')
            
            # Call to gvx(...): (line 401)
            # Processing the call arguments (line 401)
            # Getting the type of 'a1' (line 401)
            a1_15483 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 401, 40), 'a1', False)
            # Getting the type of 'b1' (line 401)
            b1_15484 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 401, 44), 'b1', False)
            # Processing the call keyword arguments (line 401)
            # Getting the type of 'uplo' (line 401)
            uplo_15485 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 401, 53), 'uplo', False)
            keyword_15486 = uplo_15485
            # Getting the type of 'hi' (line 401)
            hi_15487 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 401, 62), 'hi', False)
            keyword_15488 = hi_15487
            # Getting the type of 'type' (line 402)
            type_15489 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 402, 46), 'type', False)
            keyword_15490 = type_15489
            # Getting the type of '_job' (line 402)
            _job_15491 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 402, 57), '_job', False)
            keyword_15492 = _job_15491
            # Getting the type of 'lo' (line 402)
            lo_15493 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 402, 66), 'lo', False)
            keyword_15494 = lo_15493
            # Getting the type of 'overwrite_a' (line 403)
            overwrite_a_15495 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 403, 52), 'overwrite_a', False)
            keyword_15496 = overwrite_a_15495
            # Getting the type of 'overwrite_b' (line 404)
            overwrite_b_15497 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 404, 52), 'overwrite_b', False)
            keyword_15498 = overwrite_b_15497
            kwargs_15499 = {'itype': keyword_15490, 'overwrite_b': keyword_15498, 'uplo': keyword_15486, 'iu': keyword_15488, 'jobz': keyword_15492, 'il': keyword_15494, 'overwrite_a': keyword_15496}
            # Getting the type of 'gvx' (line 401)
            gvx_15482 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 401, 36), 'gvx', False)
            # Calling gvx(args, kwargs) (line 401)
            gvx_call_result_15500 = invoke(stypy.reporting.localization.Localization(__file__, 401, 36), gvx_15482, *[a1_15483, b1_15484], **kwargs_15499)
            
            # Obtaining the member '__getitem__' of a type (line 401)
            getitem___15501 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 401, 12), gvx_call_result_15500, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 401)
            subscript_call_result_15502 = invoke(stypy.reporting.localization.Localization(__file__, 401, 12), getitem___15501, int_15481)
            
            # Assigning a type to the variable 'tuple_var_assignment_14068' (line 401)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 401, 12), 'tuple_var_assignment_14068', subscript_call_result_15502)
            
            # Assigning a Subscript to a Name (line 401):
            
            # Obtaining the type of the subscript
            int_15503 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 401, 12), 'int')
            
            # Call to gvx(...): (line 401)
            # Processing the call arguments (line 401)
            # Getting the type of 'a1' (line 401)
            a1_15505 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 401, 40), 'a1', False)
            # Getting the type of 'b1' (line 401)
            b1_15506 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 401, 44), 'b1', False)
            # Processing the call keyword arguments (line 401)
            # Getting the type of 'uplo' (line 401)
            uplo_15507 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 401, 53), 'uplo', False)
            keyword_15508 = uplo_15507
            # Getting the type of 'hi' (line 401)
            hi_15509 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 401, 62), 'hi', False)
            keyword_15510 = hi_15509
            # Getting the type of 'type' (line 402)
            type_15511 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 402, 46), 'type', False)
            keyword_15512 = type_15511
            # Getting the type of '_job' (line 402)
            _job_15513 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 402, 57), '_job', False)
            keyword_15514 = _job_15513
            # Getting the type of 'lo' (line 402)
            lo_15515 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 402, 66), 'lo', False)
            keyword_15516 = lo_15515
            # Getting the type of 'overwrite_a' (line 403)
            overwrite_a_15517 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 403, 52), 'overwrite_a', False)
            keyword_15518 = overwrite_a_15517
            # Getting the type of 'overwrite_b' (line 404)
            overwrite_b_15519 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 404, 52), 'overwrite_b', False)
            keyword_15520 = overwrite_b_15519
            kwargs_15521 = {'itype': keyword_15512, 'overwrite_b': keyword_15520, 'uplo': keyword_15508, 'iu': keyword_15510, 'jobz': keyword_15514, 'il': keyword_15516, 'overwrite_a': keyword_15518}
            # Getting the type of 'gvx' (line 401)
            gvx_15504 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 401, 36), 'gvx', False)
            # Calling gvx(args, kwargs) (line 401)
            gvx_call_result_15522 = invoke(stypy.reporting.localization.Localization(__file__, 401, 36), gvx_15504, *[a1_15505, b1_15506], **kwargs_15521)
            
            # Obtaining the member '__getitem__' of a type (line 401)
            getitem___15523 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 401, 12), gvx_call_result_15522, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 401)
            subscript_call_result_15524 = invoke(stypy.reporting.localization.Localization(__file__, 401, 12), getitem___15523, int_15503)
            
            # Assigning a type to the variable 'tuple_var_assignment_14069' (line 401)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 401, 12), 'tuple_var_assignment_14069', subscript_call_result_15524)
            
            # Assigning a Subscript to a Name (line 401):
            
            # Obtaining the type of the subscript
            int_15525 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 401, 12), 'int')
            
            # Call to gvx(...): (line 401)
            # Processing the call arguments (line 401)
            # Getting the type of 'a1' (line 401)
            a1_15527 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 401, 40), 'a1', False)
            # Getting the type of 'b1' (line 401)
            b1_15528 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 401, 44), 'b1', False)
            # Processing the call keyword arguments (line 401)
            # Getting the type of 'uplo' (line 401)
            uplo_15529 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 401, 53), 'uplo', False)
            keyword_15530 = uplo_15529
            # Getting the type of 'hi' (line 401)
            hi_15531 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 401, 62), 'hi', False)
            keyword_15532 = hi_15531
            # Getting the type of 'type' (line 402)
            type_15533 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 402, 46), 'type', False)
            keyword_15534 = type_15533
            # Getting the type of '_job' (line 402)
            _job_15535 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 402, 57), '_job', False)
            keyword_15536 = _job_15535
            # Getting the type of 'lo' (line 402)
            lo_15537 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 402, 66), 'lo', False)
            keyword_15538 = lo_15537
            # Getting the type of 'overwrite_a' (line 403)
            overwrite_a_15539 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 403, 52), 'overwrite_a', False)
            keyword_15540 = overwrite_a_15539
            # Getting the type of 'overwrite_b' (line 404)
            overwrite_b_15541 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 404, 52), 'overwrite_b', False)
            keyword_15542 = overwrite_b_15541
            kwargs_15543 = {'itype': keyword_15534, 'overwrite_b': keyword_15542, 'uplo': keyword_15530, 'iu': keyword_15532, 'jobz': keyword_15536, 'il': keyword_15538, 'overwrite_a': keyword_15540}
            # Getting the type of 'gvx' (line 401)
            gvx_15526 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 401, 36), 'gvx', False)
            # Calling gvx(args, kwargs) (line 401)
            gvx_call_result_15544 = invoke(stypy.reporting.localization.Localization(__file__, 401, 36), gvx_15526, *[a1_15527, b1_15528], **kwargs_15543)
            
            # Obtaining the member '__getitem__' of a type (line 401)
            getitem___15545 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 401, 12), gvx_call_result_15544, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 401)
            subscript_call_result_15546 = invoke(stypy.reporting.localization.Localization(__file__, 401, 12), getitem___15545, int_15525)
            
            # Assigning a type to the variable 'tuple_var_assignment_14070' (line 401)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 401, 12), 'tuple_var_assignment_14070', subscript_call_result_15546)
            
            # Assigning a Name to a Name (line 401):
            # Getting the type of 'tuple_var_assignment_14067' (line 401)
            tuple_var_assignment_14067_15547 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 401, 12), 'tuple_var_assignment_14067')
            # Assigning a type to the variable 'w_tot' (line 401)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 401, 12), 'w_tot', tuple_var_assignment_14067_15547)
            
            # Assigning a Name to a Name (line 401):
            # Getting the type of 'tuple_var_assignment_14068' (line 401)
            tuple_var_assignment_14068_15548 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 401, 12), 'tuple_var_assignment_14068')
            # Assigning a type to the variable 'v' (line 401)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 401, 19), 'v', tuple_var_assignment_14068_15548)
            
            # Assigning a Name to a Name (line 401):
            # Getting the type of 'tuple_var_assignment_14069' (line 401)
            tuple_var_assignment_14069_15549 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 401, 12), 'tuple_var_assignment_14069')
            # Assigning a type to the variable 'ifail' (line 401)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 401, 22), 'ifail', tuple_var_assignment_14069_15549)
            
            # Assigning a Name to a Name (line 401):
            # Getting the type of 'tuple_var_assignment_14070' (line 401)
            tuple_var_assignment_14070_15550 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 401, 12), 'tuple_var_assignment_14070')
            # Assigning a type to the variable 'info' (line 401)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 401, 29), 'info', tuple_var_assignment_14070_15550)
            
            # Assigning a Subscript to a Name (line 405):
            
            # Assigning a Subscript to a Name (line 405):
            
            # Obtaining the type of the subscript
            int_15551 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 405, 22), 'int')
            # Getting the type of 'hi' (line 405)
            hi_15552 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 405, 24), 'hi')
            # Getting the type of 'lo' (line 405)
            lo_15553 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 405, 27), 'lo')
            # Applying the binary operator '-' (line 405)
            result_sub_15554 = python_operator(stypy.reporting.localization.Localization(__file__, 405, 24), '-', hi_15552, lo_15553)
            
            int_15555 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 405, 30), 'int')
            # Applying the binary operator '+' (line 405)
            result_add_15556 = python_operator(stypy.reporting.localization.Localization(__file__, 405, 29), '+', result_sub_15554, int_15555)
            
            slice_15557 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 405, 16), int_15551, result_add_15556, None)
            # Getting the type of 'w_tot' (line 405)
            w_tot_15558 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 405, 16), 'w_tot')
            # Obtaining the member '__getitem__' of a type (line 405)
            getitem___15559 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 405, 16), w_tot_15558, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 405)
            subscript_call_result_15560 = invoke(stypy.reporting.localization.Localization(__file__, 405, 16), getitem___15559, slice_15557)
            
            # Assigning a type to the variable 'w' (line 405)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 405, 12), 'w', subscript_call_result_15560)

            if more_types_in_union_15433:
                # Runtime conditional SSA for else branch (line 397)
                module_type_store.open_ssa_branch('idiom else')



        if ((not may_be_15432) or more_types_in_union_15433):
            
            # Getting the type of 'turbo' (line 407)
            turbo_15561 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 407, 13), 'turbo')
            # Testing the type of an if condition (line 407)
            if_condition_15562 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 407, 13), turbo_15561)
            # Assigning a type to the variable 'if_condition_15562' (line 407)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 407, 13), 'if_condition_15562', if_condition_15562)
            # SSA begins for if statement (line 407)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a BinOp to a Name (line 408):
            
            # Assigning a BinOp to a Name (line 408):
            # Getting the type of 'pfx' (line 408)
            pfx_15563 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 408, 21), 'pfx')
            str_15564 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 408, 25), 'str', 'gvd')
            # Applying the binary operator '+' (line 408)
            result_add_15565 = python_operator(stypy.reporting.localization.Localization(__file__, 408, 21), '+', pfx_15563, str_15564)
            
            # Assigning a type to the variable 'driver' (line 408)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 408, 12), 'driver', result_add_15565)
            
            # Assigning a Call to a Tuple (line 409):
            
            # Assigning a Subscript to a Name (line 409):
            
            # Obtaining the type of the subscript
            int_15566 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 409, 12), 'int')
            
            # Call to get_lapack_funcs(...): (line 409)
            # Processing the call arguments (line 409)
            
            # Obtaining an instance of the builtin type 'tuple' (line 409)
            tuple_15568 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 409, 39), 'tuple')
            # Adding type elements to the builtin type 'tuple' instance (line 409)
            # Adding element type (line 409)
            # Getting the type of 'driver' (line 409)
            driver_15569 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 409, 39), 'driver', False)
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 409, 39), tuple_15568, driver_15569)
            
            
            # Obtaining an instance of the builtin type 'tuple' (line 409)
            tuple_15570 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 409, 50), 'tuple')
            # Adding type elements to the builtin type 'tuple' instance (line 409)
            # Adding element type (line 409)
            # Getting the type of 'a1' (line 409)
            a1_15571 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 409, 50), 'a1', False)
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 409, 50), tuple_15570, a1_15571)
            # Adding element type (line 409)
            # Getting the type of 'b1' (line 409)
            b1_15572 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 409, 54), 'b1', False)
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 409, 50), tuple_15570, b1_15572)
            
            # Processing the call keyword arguments (line 409)
            kwargs_15573 = {}
            # Getting the type of 'get_lapack_funcs' (line 409)
            get_lapack_funcs_15567 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 409, 21), 'get_lapack_funcs', False)
            # Calling get_lapack_funcs(args, kwargs) (line 409)
            get_lapack_funcs_call_result_15574 = invoke(stypy.reporting.localization.Localization(__file__, 409, 21), get_lapack_funcs_15567, *[tuple_15568, tuple_15570], **kwargs_15573)
            
            # Obtaining the member '__getitem__' of a type (line 409)
            getitem___15575 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 409, 12), get_lapack_funcs_call_result_15574, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 409)
            subscript_call_result_15576 = invoke(stypy.reporting.localization.Localization(__file__, 409, 12), getitem___15575, int_15566)
            
            # Assigning a type to the variable 'tuple_var_assignment_14071' (line 409)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 409, 12), 'tuple_var_assignment_14071', subscript_call_result_15576)
            
            # Assigning a Name to a Name (line 409):
            # Getting the type of 'tuple_var_assignment_14071' (line 409)
            tuple_var_assignment_14071_15577 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 409, 12), 'tuple_var_assignment_14071')
            # Assigning a type to the variable 'gvd' (line 409)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 409, 13), 'gvd', tuple_var_assignment_14071_15577)
            
            # Assigning a Call to a Tuple (line 410):
            
            # Assigning a Subscript to a Name (line 410):
            
            # Obtaining the type of the subscript
            int_15578 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 410, 12), 'int')
            
            # Call to gvd(...): (line 410)
            # Processing the call arguments (line 410)
            # Getting the type of 'a1' (line 410)
            a1_15580 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 410, 29), 'a1', False)
            # Getting the type of 'b1' (line 410)
            b1_15581 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 410, 33), 'b1', False)
            # Processing the call keyword arguments (line 410)
            # Getting the type of 'uplo' (line 410)
            uplo_15582 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 410, 42), 'uplo', False)
            keyword_15583 = uplo_15582
            # Getting the type of 'type' (line 410)
            type_15584 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 410, 54), 'type', False)
            keyword_15585 = type_15584
            # Getting the type of '_job' (line 410)
            _job_15586 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 410, 65), '_job', False)
            keyword_15587 = _job_15586
            # Getting the type of 'overwrite_a' (line 411)
            overwrite_a_15588 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 411, 41), 'overwrite_a', False)
            keyword_15589 = overwrite_a_15588
            # Getting the type of 'overwrite_b' (line 412)
            overwrite_b_15590 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 412, 41), 'overwrite_b', False)
            keyword_15591 = overwrite_b_15590
            kwargs_15592 = {'uplo': keyword_15583, 'overwrite_a': keyword_15589, 'itype': keyword_15585, 'jobz': keyword_15587, 'overwrite_b': keyword_15591}
            # Getting the type of 'gvd' (line 410)
            gvd_15579 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 410, 25), 'gvd', False)
            # Calling gvd(args, kwargs) (line 410)
            gvd_call_result_15593 = invoke(stypy.reporting.localization.Localization(__file__, 410, 25), gvd_15579, *[a1_15580, b1_15581], **kwargs_15592)
            
            # Obtaining the member '__getitem__' of a type (line 410)
            getitem___15594 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 410, 12), gvd_call_result_15593, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 410)
            subscript_call_result_15595 = invoke(stypy.reporting.localization.Localization(__file__, 410, 12), getitem___15594, int_15578)
            
            # Assigning a type to the variable 'tuple_var_assignment_14072' (line 410)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 410, 12), 'tuple_var_assignment_14072', subscript_call_result_15595)
            
            # Assigning a Subscript to a Name (line 410):
            
            # Obtaining the type of the subscript
            int_15596 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 410, 12), 'int')
            
            # Call to gvd(...): (line 410)
            # Processing the call arguments (line 410)
            # Getting the type of 'a1' (line 410)
            a1_15598 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 410, 29), 'a1', False)
            # Getting the type of 'b1' (line 410)
            b1_15599 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 410, 33), 'b1', False)
            # Processing the call keyword arguments (line 410)
            # Getting the type of 'uplo' (line 410)
            uplo_15600 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 410, 42), 'uplo', False)
            keyword_15601 = uplo_15600
            # Getting the type of 'type' (line 410)
            type_15602 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 410, 54), 'type', False)
            keyword_15603 = type_15602
            # Getting the type of '_job' (line 410)
            _job_15604 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 410, 65), '_job', False)
            keyword_15605 = _job_15604
            # Getting the type of 'overwrite_a' (line 411)
            overwrite_a_15606 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 411, 41), 'overwrite_a', False)
            keyword_15607 = overwrite_a_15606
            # Getting the type of 'overwrite_b' (line 412)
            overwrite_b_15608 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 412, 41), 'overwrite_b', False)
            keyword_15609 = overwrite_b_15608
            kwargs_15610 = {'uplo': keyword_15601, 'overwrite_a': keyword_15607, 'itype': keyword_15603, 'jobz': keyword_15605, 'overwrite_b': keyword_15609}
            # Getting the type of 'gvd' (line 410)
            gvd_15597 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 410, 25), 'gvd', False)
            # Calling gvd(args, kwargs) (line 410)
            gvd_call_result_15611 = invoke(stypy.reporting.localization.Localization(__file__, 410, 25), gvd_15597, *[a1_15598, b1_15599], **kwargs_15610)
            
            # Obtaining the member '__getitem__' of a type (line 410)
            getitem___15612 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 410, 12), gvd_call_result_15611, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 410)
            subscript_call_result_15613 = invoke(stypy.reporting.localization.Localization(__file__, 410, 12), getitem___15612, int_15596)
            
            # Assigning a type to the variable 'tuple_var_assignment_14073' (line 410)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 410, 12), 'tuple_var_assignment_14073', subscript_call_result_15613)
            
            # Assigning a Subscript to a Name (line 410):
            
            # Obtaining the type of the subscript
            int_15614 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 410, 12), 'int')
            
            # Call to gvd(...): (line 410)
            # Processing the call arguments (line 410)
            # Getting the type of 'a1' (line 410)
            a1_15616 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 410, 29), 'a1', False)
            # Getting the type of 'b1' (line 410)
            b1_15617 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 410, 33), 'b1', False)
            # Processing the call keyword arguments (line 410)
            # Getting the type of 'uplo' (line 410)
            uplo_15618 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 410, 42), 'uplo', False)
            keyword_15619 = uplo_15618
            # Getting the type of 'type' (line 410)
            type_15620 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 410, 54), 'type', False)
            keyword_15621 = type_15620
            # Getting the type of '_job' (line 410)
            _job_15622 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 410, 65), '_job', False)
            keyword_15623 = _job_15622
            # Getting the type of 'overwrite_a' (line 411)
            overwrite_a_15624 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 411, 41), 'overwrite_a', False)
            keyword_15625 = overwrite_a_15624
            # Getting the type of 'overwrite_b' (line 412)
            overwrite_b_15626 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 412, 41), 'overwrite_b', False)
            keyword_15627 = overwrite_b_15626
            kwargs_15628 = {'uplo': keyword_15619, 'overwrite_a': keyword_15625, 'itype': keyword_15621, 'jobz': keyword_15623, 'overwrite_b': keyword_15627}
            # Getting the type of 'gvd' (line 410)
            gvd_15615 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 410, 25), 'gvd', False)
            # Calling gvd(args, kwargs) (line 410)
            gvd_call_result_15629 = invoke(stypy.reporting.localization.Localization(__file__, 410, 25), gvd_15615, *[a1_15616, b1_15617], **kwargs_15628)
            
            # Obtaining the member '__getitem__' of a type (line 410)
            getitem___15630 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 410, 12), gvd_call_result_15629, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 410)
            subscript_call_result_15631 = invoke(stypy.reporting.localization.Localization(__file__, 410, 12), getitem___15630, int_15614)
            
            # Assigning a type to the variable 'tuple_var_assignment_14074' (line 410)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 410, 12), 'tuple_var_assignment_14074', subscript_call_result_15631)
            
            # Assigning a Name to a Name (line 410):
            # Getting the type of 'tuple_var_assignment_14072' (line 410)
            tuple_var_assignment_14072_15632 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 410, 12), 'tuple_var_assignment_14072')
            # Assigning a type to the variable 'v' (line 410)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 410, 12), 'v', tuple_var_assignment_14072_15632)
            
            # Assigning a Name to a Name (line 410):
            # Getting the type of 'tuple_var_assignment_14073' (line 410)
            tuple_var_assignment_14073_15633 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 410, 12), 'tuple_var_assignment_14073')
            # Assigning a type to the variable 'w' (line 410)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 410, 15), 'w', tuple_var_assignment_14073_15633)
            
            # Assigning a Name to a Name (line 410):
            # Getting the type of 'tuple_var_assignment_14074' (line 410)
            tuple_var_assignment_14074_15634 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 410, 12), 'tuple_var_assignment_14074')
            # Assigning a type to the variable 'info' (line 410)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 410, 18), 'info', tuple_var_assignment_14074_15634)
            # SSA branch for the else part of an if statement (line 407)
            module_type_store.open_ssa_branch('else')
            
            # Assigning a BinOp to a Name (line 415):
            
            # Assigning a BinOp to a Name (line 415):
            # Getting the type of 'pfx' (line 415)
            pfx_15635 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 415, 21), 'pfx')
            str_15636 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 415, 25), 'str', 'gv')
            # Applying the binary operator '+' (line 415)
            result_add_15637 = python_operator(stypy.reporting.localization.Localization(__file__, 415, 21), '+', pfx_15635, str_15636)
            
            # Assigning a type to the variable 'driver' (line 415)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 415, 12), 'driver', result_add_15637)
            
            # Assigning a Call to a Tuple (line 416):
            
            # Assigning a Subscript to a Name (line 416):
            
            # Obtaining the type of the subscript
            int_15638 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 416, 12), 'int')
            
            # Call to get_lapack_funcs(...): (line 416)
            # Processing the call arguments (line 416)
            
            # Obtaining an instance of the builtin type 'tuple' (line 416)
            tuple_15640 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 416, 38), 'tuple')
            # Adding type elements to the builtin type 'tuple' instance (line 416)
            # Adding element type (line 416)
            # Getting the type of 'driver' (line 416)
            driver_15641 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 416, 38), 'driver', False)
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 416, 38), tuple_15640, driver_15641)
            
            
            # Obtaining an instance of the builtin type 'tuple' (line 416)
            tuple_15642 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 416, 49), 'tuple')
            # Adding type elements to the builtin type 'tuple' instance (line 416)
            # Adding element type (line 416)
            # Getting the type of 'a1' (line 416)
            a1_15643 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 416, 49), 'a1', False)
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 416, 49), tuple_15642, a1_15643)
            # Adding element type (line 416)
            # Getting the type of 'b1' (line 416)
            b1_15644 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 416, 53), 'b1', False)
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 416, 49), tuple_15642, b1_15644)
            
            # Processing the call keyword arguments (line 416)
            kwargs_15645 = {}
            # Getting the type of 'get_lapack_funcs' (line 416)
            get_lapack_funcs_15639 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 416, 20), 'get_lapack_funcs', False)
            # Calling get_lapack_funcs(args, kwargs) (line 416)
            get_lapack_funcs_call_result_15646 = invoke(stypy.reporting.localization.Localization(__file__, 416, 20), get_lapack_funcs_15639, *[tuple_15640, tuple_15642], **kwargs_15645)
            
            # Obtaining the member '__getitem__' of a type (line 416)
            getitem___15647 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 416, 12), get_lapack_funcs_call_result_15646, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 416)
            subscript_call_result_15648 = invoke(stypy.reporting.localization.Localization(__file__, 416, 12), getitem___15647, int_15638)
            
            # Assigning a type to the variable 'tuple_var_assignment_14075' (line 416)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 416, 12), 'tuple_var_assignment_14075', subscript_call_result_15648)
            
            # Assigning a Name to a Name (line 416):
            # Getting the type of 'tuple_var_assignment_14075' (line 416)
            tuple_var_assignment_14075_15649 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 416, 12), 'tuple_var_assignment_14075')
            # Assigning a type to the variable 'gv' (line 416)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 416, 13), 'gv', tuple_var_assignment_14075_15649)
            
            # Assigning a Call to a Tuple (line 417):
            
            # Assigning a Subscript to a Name (line 417):
            
            # Obtaining the type of the subscript
            int_15650 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 417, 12), 'int')
            
            # Call to gv(...): (line 417)
            # Processing the call arguments (line 417)
            # Getting the type of 'a1' (line 417)
            a1_15652 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 417, 28), 'a1', False)
            # Getting the type of 'b1' (line 417)
            b1_15653 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 417, 32), 'b1', False)
            # Processing the call keyword arguments (line 417)
            # Getting the type of 'uplo' (line 417)
            uplo_15654 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 417, 41), 'uplo', False)
            keyword_15655 = uplo_15654
            # Getting the type of 'type' (line 417)
            type_15656 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 417, 53), 'type', False)
            keyword_15657 = type_15656
            # Getting the type of '_job' (line 417)
            _job_15658 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 417, 64), '_job', False)
            keyword_15659 = _job_15658
            # Getting the type of 'overwrite_a' (line 418)
            overwrite_a_15660 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 418, 40), 'overwrite_a', False)
            keyword_15661 = overwrite_a_15660
            # Getting the type of 'overwrite_b' (line 419)
            overwrite_b_15662 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 419, 40), 'overwrite_b', False)
            keyword_15663 = overwrite_b_15662
            kwargs_15664 = {'uplo': keyword_15655, 'overwrite_a': keyword_15661, 'itype': keyword_15657, 'jobz': keyword_15659, 'overwrite_b': keyword_15663}
            # Getting the type of 'gv' (line 417)
            gv_15651 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 417, 25), 'gv', False)
            # Calling gv(args, kwargs) (line 417)
            gv_call_result_15665 = invoke(stypy.reporting.localization.Localization(__file__, 417, 25), gv_15651, *[a1_15652, b1_15653], **kwargs_15664)
            
            # Obtaining the member '__getitem__' of a type (line 417)
            getitem___15666 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 417, 12), gv_call_result_15665, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 417)
            subscript_call_result_15667 = invoke(stypy.reporting.localization.Localization(__file__, 417, 12), getitem___15666, int_15650)
            
            # Assigning a type to the variable 'tuple_var_assignment_14076' (line 417)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 417, 12), 'tuple_var_assignment_14076', subscript_call_result_15667)
            
            # Assigning a Subscript to a Name (line 417):
            
            # Obtaining the type of the subscript
            int_15668 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 417, 12), 'int')
            
            # Call to gv(...): (line 417)
            # Processing the call arguments (line 417)
            # Getting the type of 'a1' (line 417)
            a1_15670 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 417, 28), 'a1', False)
            # Getting the type of 'b1' (line 417)
            b1_15671 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 417, 32), 'b1', False)
            # Processing the call keyword arguments (line 417)
            # Getting the type of 'uplo' (line 417)
            uplo_15672 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 417, 41), 'uplo', False)
            keyword_15673 = uplo_15672
            # Getting the type of 'type' (line 417)
            type_15674 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 417, 53), 'type', False)
            keyword_15675 = type_15674
            # Getting the type of '_job' (line 417)
            _job_15676 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 417, 64), '_job', False)
            keyword_15677 = _job_15676
            # Getting the type of 'overwrite_a' (line 418)
            overwrite_a_15678 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 418, 40), 'overwrite_a', False)
            keyword_15679 = overwrite_a_15678
            # Getting the type of 'overwrite_b' (line 419)
            overwrite_b_15680 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 419, 40), 'overwrite_b', False)
            keyword_15681 = overwrite_b_15680
            kwargs_15682 = {'uplo': keyword_15673, 'overwrite_a': keyword_15679, 'itype': keyword_15675, 'jobz': keyword_15677, 'overwrite_b': keyword_15681}
            # Getting the type of 'gv' (line 417)
            gv_15669 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 417, 25), 'gv', False)
            # Calling gv(args, kwargs) (line 417)
            gv_call_result_15683 = invoke(stypy.reporting.localization.Localization(__file__, 417, 25), gv_15669, *[a1_15670, b1_15671], **kwargs_15682)
            
            # Obtaining the member '__getitem__' of a type (line 417)
            getitem___15684 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 417, 12), gv_call_result_15683, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 417)
            subscript_call_result_15685 = invoke(stypy.reporting.localization.Localization(__file__, 417, 12), getitem___15684, int_15668)
            
            # Assigning a type to the variable 'tuple_var_assignment_14077' (line 417)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 417, 12), 'tuple_var_assignment_14077', subscript_call_result_15685)
            
            # Assigning a Subscript to a Name (line 417):
            
            # Obtaining the type of the subscript
            int_15686 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 417, 12), 'int')
            
            # Call to gv(...): (line 417)
            # Processing the call arguments (line 417)
            # Getting the type of 'a1' (line 417)
            a1_15688 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 417, 28), 'a1', False)
            # Getting the type of 'b1' (line 417)
            b1_15689 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 417, 32), 'b1', False)
            # Processing the call keyword arguments (line 417)
            # Getting the type of 'uplo' (line 417)
            uplo_15690 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 417, 41), 'uplo', False)
            keyword_15691 = uplo_15690
            # Getting the type of 'type' (line 417)
            type_15692 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 417, 53), 'type', False)
            keyword_15693 = type_15692
            # Getting the type of '_job' (line 417)
            _job_15694 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 417, 64), '_job', False)
            keyword_15695 = _job_15694
            # Getting the type of 'overwrite_a' (line 418)
            overwrite_a_15696 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 418, 40), 'overwrite_a', False)
            keyword_15697 = overwrite_a_15696
            # Getting the type of 'overwrite_b' (line 419)
            overwrite_b_15698 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 419, 40), 'overwrite_b', False)
            keyword_15699 = overwrite_b_15698
            kwargs_15700 = {'uplo': keyword_15691, 'overwrite_a': keyword_15697, 'itype': keyword_15693, 'jobz': keyword_15695, 'overwrite_b': keyword_15699}
            # Getting the type of 'gv' (line 417)
            gv_15687 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 417, 25), 'gv', False)
            # Calling gv(args, kwargs) (line 417)
            gv_call_result_15701 = invoke(stypy.reporting.localization.Localization(__file__, 417, 25), gv_15687, *[a1_15688, b1_15689], **kwargs_15700)
            
            # Obtaining the member '__getitem__' of a type (line 417)
            getitem___15702 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 417, 12), gv_call_result_15701, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 417)
            subscript_call_result_15703 = invoke(stypy.reporting.localization.Localization(__file__, 417, 12), getitem___15702, int_15686)
            
            # Assigning a type to the variable 'tuple_var_assignment_14078' (line 417)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 417, 12), 'tuple_var_assignment_14078', subscript_call_result_15703)
            
            # Assigning a Name to a Name (line 417):
            # Getting the type of 'tuple_var_assignment_14076' (line 417)
            tuple_var_assignment_14076_15704 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 417, 12), 'tuple_var_assignment_14076')
            # Assigning a type to the variable 'v' (line 417)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 417, 12), 'v', tuple_var_assignment_14076_15704)
            
            # Assigning a Name to a Name (line 417):
            # Getting the type of 'tuple_var_assignment_14077' (line 417)
            tuple_var_assignment_14077_15705 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 417, 12), 'tuple_var_assignment_14077')
            # Assigning a type to the variable 'w' (line 417)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 417, 15), 'w', tuple_var_assignment_14077_15705)
            
            # Assigning a Name to a Name (line 417):
            # Getting the type of 'tuple_var_assignment_14078' (line 417)
            tuple_var_assignment_14078_15706 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 417, 12), 'tuple_var_assignment_14078')
            # Assigning a type to the variable 'info' (line 417)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 417, 18), 'info', tuple_var_assignment_14078_15706)
            # SSA join for if statement (line 407)
            module_type_store = module_type_store.join_ssa_context()
            

            if (may_be_15432 and more_types_in_union_15433):
                # SSA join for if statement (line 397)
                module_type_store = module_type_store.join_ssa_context()


        

        if (may_be_15258 and more_types_in_union_15259):
            # SSA join for if statement (line 382)
            module_type_store = module_type_store.join_ssa_context()


    
    
    
    # Getting the type of 'info' (line 422)
    info_15707 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 422, 7), 'info')
    int_15708 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 422, 15), 'int')
    # Applying the binary operator '==' (line 422)
    result_eq_15709 = python_operator(stypy.reporting.localization.Localization(__file__, 422, 7), '==', info_15707, int_15708)
    
    # Testing the type of an if condition (line 422)
    if_condition_15710 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 422, 4), result_eq_15709)
    # Assigning a type to the variable 'if_condition_15710' (line 422)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 422, 4), 'if_condition_15710', if_condition_15710)
    # SSA begins for if statement (line 422)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Getting the type of 'eigvals_only' (line 423)
    eigvals_only_15711 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 423, 11), 'eigvals_only')
    # Testing the type of an if condition (line 423)
    if_condition_15712 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 423, 8), eigvals_only_15711)
    # Assigning a type to the variable 'if_condition_15712' (line 423)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 423, 8), 'if_condition_15712', if_condition_15712)
    # SSA begins for if statement (line 423)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # Getting the type of 'w' (line 424)
    w_15713 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 424, 19), 'w')
    # Assigning a type to the variable 'stypy_return_type' (line 424)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 424, 12), 'stypy_return_type', w_15713)
    # SSA branch for the else part of an if statement (line 423)
    module_type_store.open_ssa_branch('else')
    
    # Obtaining an instance of the builtin type 'tuple' (line 426)
    tuple_15714 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 426, 19), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 426)
    # Adding element type (line 426)
    # Getting the type of 'w' (line 426)
    w_15715 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 426, 19), 'w')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 426, 19), tuple_15714, w_15715)
    # Adding element type (line 426)
    # Getting the type of 'v' (line 426)
    v_15716 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 426, 22), 'v')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 426, 19), tuple_15714, v_15716)
    
    # Assigning a type to the variable 'stypy_return_type' (line 426)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 426, 12), 'stypy_return_type', tuple_15714)
    # SSA join for if statement (line 423)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 422)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to _check_info(...): (line 427)
    # Processing the call arguments (line 427)
    # Getting the type of 'info' (line 427)
    info_15718 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 427, 16), 'info', False)
    # Getting the type of 'driver' (line 427)
    driver_15719 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 427, 22), 'driver', False)
    # Processing the call keyword arguments (line 427)
    # Getting the type of 'False' (line 427)
    False_15720 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 427, 39), 'False', False)
    keyword_15721 = False_15720
    kwargs_15722 = {'positive': keyword_15721}
    # Getting the type of '_check_info' (line 427)
    _check_info_15717 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 427, 4), '_check_info', False)
    # Calling _check_info(args, kwargs) (line 427)
    _check_info_call_result_15723 = invoke(stypy.reporting.localization.Localization(__file__, 427, 4), _check_info_15717, *[info_15718, driver_15719], **kwargs_15722)
    
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'info' (line 428)
    info_15724 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 428, 7), 'info')
    int_15725 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 428, 14), 'int')
    # Applying the binary operator '>' (line 428)
    result_gt_15726 = python_operator(stypy.reporting.localization.Localization(__file__, 428, 7), '>', info_15724, int_15725)
    
    
    # Getting the type of 'b1' (line 428)
    b1_15727 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 428, 20), 'b1')
    # Getting the type of 'None' (line 428)
    None_15728 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 428, 26), 'None')
    # Applying the binary operator 'is' (line 428)
    result_is__15729 = python_operator(stypy.reporting.localization.Localization(__file__, 428, 20), 'is', b1_15727, None_15728)
    
    # Applying the binary operator 'and' (line 428)
    result_and_keyword_15730 = python_operator(stypy.reporting.localization.Localization(__file__, 428, 7), 'and', result_gt_15726, result_is__15729)
    
    # Testing the type of an if condition (line 428)
    if_condition_15731 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 428, 4), result_and_keyword_15730)
    # Assigning a type to the variable 'if_condition_15731' (line 428)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 428, 4), 'if_condition_15731', if_condition_15731)
    # SSA begins for if statement (line 428)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to LinAlgError(...): (line 429)
    # Processing the call arguments (line 429)
    str_15733 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 429, 26), 'str', 'unrecoverable internal error.')
    # Processing the call keyword arguments (line 429)
    kwargs_15734 = {}
    # Getting the type of 'LinAlgError' (line 429)
    LinAlgError_15732 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 429, 14), 'LinAlgError', False)
    # Calling LinAlgError(args, kwargs) (line 429)
    LinAlgError_call_result_15735 = invoke(stypy.reporting.localization.Localization(__file__, 429, 14), LinAlgError_15732, *[str_15733], **kwargs_15734)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 429, 8), LinAlgError_call_result_15735, 'raise parameter', BaseException)
    # SSA branch for the else part of an if statement (line 428)
    module_type_store.open_ssa_branch('else')
    
    
    int_15736 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 432, 9), 'int')
    # Getting the type of 'info' (line 432)
    info_15737 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 432, 13), 'info')
    # Applying the binary operator '<' (line 432)
    result_lt_15738 = python_operator(stypy.reporting.localization.Localization(__file__, 432, 9), '<', int_15736, info_15737)
    
    # Obtaining the type of the subscript
    int_15739 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 432, 30), 'int')
    # Getting the type of 'b1' (line 432)
    b1_15740 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 432, 21), 'b1')
    # Obtaining the member 'shape' of a type (line 432)
    shape_15741 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 432, 21), b1_15740, 'shape')
    # Obtaining the member '__getitem__' of a type (line 432)
    getitem___15742 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 432, 21), shape_15741, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 432)
    subscript_call_result_15743 = invoke(stypy.reporting.localization.Localization(__file__, 432, 21), getitem___15742, int_15739)
    
    # Applying the binary operator '<=' (line 432)
    result_le_15744 = python_operator(stypy.reporting.localization.Localization(__file__, 432, 9), '<=', info_15737, subscript_call_result_15743)
    # Applying the binary operator '&' (line 432)
    result_and__15745 = python_operator(stypy.reporting.localization.Localization(__file__, 432, 9), '&', result_lt_15738, result_le_15744)
    
    # Testing the type of an if condition (line 432)
    if_condition_15746 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 432, 9), result_and__15745)
    # Assigning a type to the variable 'if_condition_15746' (line 432)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 432, 9), 'if_condition_15746', if_condition_15746)
    # SSA begins for if statement (line 432)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Type idiom detected: calculating its left and rigth part (line 433)
    # Getting the type of 'eigvals' (line 433)
    eigvals_15747 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 433, 8), 'eigvals')
    # Getting the type of 'None' (line 433)
    None_15748 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 433, 26), 'None')
    
    (may_be_15749, more_types_in_union_15750) = may_not_be_none(eigvals_15747, None_15748)

    if may_be_15749:

        if more_types_in_union_15750:
            # Runtime conditional SSA (line 433)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Call to LinAlgError(...): (line 434)
        # Processing the call arguments (line 434)
        str_15752 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 434, 30), 'str', 'the eigenvectors %s failed to converge.')
        
        # Call to nonzero(...): (line 435)
        # Processing the call arguments (line 435)
        # Getting the type of 'ifail' (line 435)
        ifail_15754 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 435, 53), 'ifail', False)
        # Processing the call keyword arguments (line 435)
        kwargs_15755 = {}
        # Getting the type of 'nonzero' (line 435)
        nonzero_15753 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 435, 45), 'nonzero', False)
        # Calling nonzero(args, kwargs) (line 435)
        nonzero_call_result_15756 = invoke(stypy.reporting.localization.Localization(__file__, 435, 45), nonzero_15753, *[ifail_15754], **kwargs_15755)
        
        # Applying the binary operator '%' (line 434)
        result_mod_15757 = python_operator(stypy.reporting.localization.Localization(__file__, 434, 30), '%', str_15752, nonzero_call_result_15756)
        
        int_15758 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 435, 60), 'int')
        # Applying the binary operator '-' (line 434)
        result_sub_15759 = python_operator(stypy.reporting.localization.Localization(__file__, 434, 30), '-', result_mod_15757, int_15758)
        
        # Processing the call keyword arguments (line 434)
        kwargs_15760 = {}
        # Getting the type of 'LinAlgError' (line 434)
        LinAlgError_15751 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 434, 18), 'LinAlgError', False)
        # Calling LinAlgError(args, kwargs) (line 434)
        LinAlgError_call_result_15761 = invoke(stypy.reporting.localization.Localization(__file__, 434, 18), LinAlgError_15751, *[result_sub_15759], **kwargs_15760)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 434, 12), LinAlgError_call_result_15761, 'raise parameter', BaseException)

        if more_types_in_union_15750:
            # Runtime conditional SSA for else branch (line 433)
            module_type_store.open_ssa_branch('idiom else')



    if ((not may_be_15749) or more_types_in_union_15750):
        
        # Call to LinAlgError(...): (line 437)
        # Processing the call arguments (line 437)
        str_15763 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 437, 30), 'str', 'internal fortran routine failed to converge: %i off-diagonal elements of an intermediate tridiagonal form did not converge to zero.')
        # Getting the type of 'info' (line 440)
        info_15764 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 440, 44), 'info', False)
        # Applying the binary operator '%' (line 437)
        result_mod_15765 = python_operator(stypy.reporting.localization.Localization(__file__, 437, 30), '%', str_15763, info_15764)
        
        # Processing the call keyword arguments (line 437)
        kwargs_15766 = {}
        # Getting the type of 'LinAlgError' (line 437)
        LinAlgError_15762 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 437, 18), 'LinAlgError', False)
        # Calling LinAlgError(args, kwargs) (line 437)
        LinAlgError_call_result_15767 = invoke(stypy.reporting.localization.Localization(__file__, 437, 18), LinAlgError_15762, *[result_mod_15765], **kwargs_15766)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 437, 12), LinAlgError_call_result_15767, 'raise parameter', BaseException)

        if (may_be_15749 and more_types_in_union_15750):
            # SSA join for if statement (line 433)
            module_type_store = module_type_store.join_ssa_context()


    
    # SSA branch for the else part of an if statement (line 432)
    module_type_store.open_ssa_branch('else')
    
    # Call to LinAlgError(...): (line 444)
    # Processing the call arguments (line 444)
    str_15769 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 444, 26), 'str', "the leading minor of order %i of 'b' is not positive definite. The factorization of 'b' could not be completed and no eigenvalues or eigenvectors were computed.")
    # Getting the type of 'info' (line 448)
    info_15770 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 448, 42), 'info', False)
    
    # Obtaining the type of the subscript
    int_15771 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 448, 56), 'int')
    # Getting the type of 'b1' (line 448)
    b1_15772 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 448, 47), 'b1', False)
    # Obtaining the member 'shape' of a type (line 448)
    shape_15773 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 448, 47), b1_15772, 'shape')
    # Obtaining the member '__getitem__' of a type (line 448)
    getitem___15774 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 448, 47), shape_15773, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 448)
    subscript_call_result_15775 = invoke(stypy.reporting.localization.Localization(__file__, 448, 47), getitem___15774, int_15771)
    
    # Applying the binary operator '-' (line 448)
    result_sub_15776 = python_operator(stypy.reporting.localization.Localization(__file__, 448, 42), '-', info_15770, subscript_call_result_15775)
    
    # Applying the binary operator '%' (line 444)
    result_mod_15777 = python_operator(stypy.reporting.localization.Localization(__file__, 444, 26), '%', str_15769, result_sub_15776)
    
    # Processing the call keyword arguments (line 444)
    kwargs_15778 = {}
    # Getting the type of 'LinAlgError' (line 444)
    LinAlgError_15768 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 444, 14), 'LinAlgError', False)
    # Calling LinAlgError(args, kwargs) (line 444)
    LinAlgError_call_result_15779 = invoke(stypy.reporting.localization.Localization(__file__, 444, 14), LinAlgError_15768, *[result_mod_15777], **kwargs_15778)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 444, 8), LinAlgError_call_result_15779, 'raise parameter', BaseException)
    # SSA join for if statement (line 432)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 428)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'eigh(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'eigh' in the type store
    # Getting the type of 'stypy_return_type' (line 240)
    stypy_return_type_15780 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_15780)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'eigh'
    return stypy_return_type_15780

# Assigning a type to the variable 'eigh' (line 240)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 240, 0), 'eigh', eigh)

# Assigning a Dict to a Name (line 451):

# Assigning a Dict to a Name (line 451):

# Obtaining an instance of the builtin type 'dict' (line 451)
dict_15781 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 451, 13), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 451)
# Adding element type (key, value) (line 451)
int_15782 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 451, 14), 'int')
int_15783 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 451, 17), 'int')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 451, 13), dict_15781, (int_15782, int_15783))
# Adding element type (key, value) (line 451)
int_15784 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 451, 20), 'int')
int_15785 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 451, 23), 'int')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 451, 13), dict_15781, (int_15784, int_15785))
# Adding element type (key, value) (line 451)
int_15786 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 451, 26), 'int')
int_15787 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 451, 29), 'int')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 451, 13), dict_15781, (int_15786, int_15787))
# Adding element type (key, value) (line 451)
str_15788 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 452, 14), 'str', 'all')
int_15789 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 452, 21), 'int')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 451, 13), dict_15781, (str_15788, int_15789))
# Adding element type (key, value) (line 451)
str_15790 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 452, 24), 'str', 'value')
int_15791 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 452, 33), 'int')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 451, 13), dict_15781, (str_15790, int_15791))
# Adding element type (key, value) (line 451)
str_15792 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 452, 36), 'str', 'index')
int_15793 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 452, 45), 'int')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 451, 13), dict_15781, (str_15792, int_15793))
# Adding element type (key, value) (line 451)
str_15794 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 453, 14), 'str', 'a')
int_15795 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 453, 19), 'int')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 451, 13), dict_15781, (str_15794, int_15795))
# Adding element type (key, value) (line 451)
str_15796 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 453, 22), 'str', 'v')
int_15797 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 453, 27), 'int')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 451, 13), dict_15781, (str_15796, int_15797))
# Adding element type (key, value) (line 451)
str_15798 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 453, 30), 'str', 'i')
int_15799 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 453, 35), 'int')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 451, 13), dict_15781, (str_15798, int_15799))

# Assigning a type to the variable '_conv_dict' (line 451)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 451, 0), '_conv_dict', dict_15781)

@norecursion
def _check_select(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_check_select'
    module_type_store = module_type_store.open_function_context('_check_select', 456, 0, False)
    
    # Passed parameters checking function
    _check_select.stypy_localization = localization
    _check_select.stypy_type_of_self = None
    _check_select.stypy_type_store = module_type_store
    _check_select.stypy_function_name = '_check_select'
    _check_select.stypy_param_names_list = ['select', 'select_range', 'max_ev', 'max_len']
    _check_select.stypy_varargs_param_name = None
    _check_select.stypy_kwargs_param_name = None
    _check_select.stypy_call_defaults = defaults
    _check_select.stypy_call_varargs = varargs
    _check_select.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_check_select', ['select', 'select_range', 'max_ev', 'max_len'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_check_select', localization, ['select', 'select_range', 'max_ev', 'max_len'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_check_select(...)' code ##################

    str_15800 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 457, 4), 'str', 'Check that select is valid, convert to Fortran style.')
    
    
    # Call to isinstance(...): (line 458)
    # Processing the call arguments (line 458)
    # Getting the type of 'select' (line 458)
    select_15802 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 458, 18), 'select', False)
    # Getting the type of 'string_types' (line 458)
    string_types_15803 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 458, 26), 'string_types', False)
    # Processing the call keyword arguments (line 458)
    kwargs_15804 = {}
    # Getting the type of 'isinstance' (line 458)
    isinstance_15801 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 458, 7), 'isinstance', False)
    # Calling isinstance(args, kwargs) (line 458)
    isinstance_call_result_15805 = invoke(stypy.reporting.localization.Localization(__file__, 458, 7), isinstance_15801, *[select_15802, string_types_15803], **kwargs_15804)
    
    # Testing the type of an if condition (line 458)
    if_condition_15806 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 458, 4), isinstance_call_result_15805)
    # Assigning a type to the variable 'if_condition_15806' (line 458)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 458, 4), 'if_condition_15806', if_condition_15806)
    # SSA begins for if statement (line 458)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 459):
    
    # Assigning a Call to a Name (line 459):
    
    # Call to lower(...): (line 459)
    # Processing the call keyword arguments (line 459)
    kwargs_15809 = {}
    # Getting the type of 'select' (line 459)
    select_15807 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 459, 17), 'select', False)
    # Obtaining the member 'lower' of a type (line 459)
    lower_15808 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 459, 17), select_15807, 'lower')
    # Calling lower(args, kwargs) (line 459)
    lower_call_result_15810 = invoke(stypy.reporting.localization.Localization(__file__, 459, 17), lower_15808, *[], **kwargs_15809)
    
    # Assigning a type to the variable 'select' (line 459)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 459, 8), 'select', lower_call_result_15810)
    # SSA join for if statement (line 458)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # SSA begins for try-except statement (line 460)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
    
    # Assigning a Subscript to a Name (line 461):
    
    # Assigning a Subscript to a Name (line 461):
    
    # Obtaining the type of the subscript
    # Getting the type of 'select' (line 461)
    select_15811 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 461, 28), 'select')
    # Getting the type of '_conv_dict' (line 461)
    _conv_dict_15812 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 461, 17), '_conv_dict')
    # Obtaining the member '__getitem__' of a type (line 461)
    getitem___15813 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 461, 17), _conv_dict_15812, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 461)
    subscript_call_result_15814 = invoke(stypy.reporting.localization.Localization(__file__, 461, 17), getitem___15813, select_15811)
    
    # Assigning a type to the variable 'select' (line 461)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 461, 8), 'select', subscript_call_result_15814)
    # SSA branch for the except part of a try statement (line 460)
    # SSA branch for the except 'KeyError' branch of a try statement (line 460)
    module_type_store.open_ssa_branch('except')
    
    # Call to ValueError(...): (line 463)
    # Processing the call arguments (line 463)
    str_15816 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 463, 25), 'str', 'invalid argument for select')
    # Processing the call keyword arguments (line 463)
    kwargs_15817 = {}
    # Getting the type of 'ValueError' (line 463)
    ValueError_15815 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 463, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 463)
    ValueError_call_result_15818 = invoke(stypy.reporting.localization.Localization(__file__, 463, 14), ValueError_15815, *[str_15816], **kwargs_15817)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 463, 8), ValueError_call_result_15818, 'raise parameter', BaseException)
    # SSA join for try-except statement (line 460)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Tuple to a Tuple (line 464):
    
    # Assigning a Num to a Name (line 464):
    float_15819 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 464, 13), 'float')
    # Assigning a type to the variable 'tuple_assignment_14079' (line 464)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 464, 4), 'tuple_assignment_14079', float_15819)
    
    # Assigning a Num to a Name (line 464):
    float_15820 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 464, 17), 'float')
    # Assigning a type to the variable 'tuple_assignment_14080' (line 464)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 464, 4), 'tuple_assignment_14080', float_15820)
    
    # Assigning a Name to a Name (line 464):
    # Getting the type of 'tuple_assignment_14079' (line 464)
    tuple_assignment_14079_15821 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 464, 4), 'tuple_assignment_14079')
    # Assigning a type to the variable 'vl' (line 464)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 464, 4), 'vl', tuple_assignment_14079_15821)
    
    # Assigning a Name to a Name (line 464):
    # Getting the type of 'tuple_assignment_14080' (line 464)
    tuple_assignment_14080_15822 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 464, 4), 'tuple_assignment_14080')
    # Assigning a type to the variable 'vu' (line 464)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 464, 8), 'vu', tuple_assignment_14080_15822)
    
    # Multiple assignment of 2 elements.
    
    # Assigning a Num to a Name (line 465):
    int_15823 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 465, 14), 'int')
    # Assigning a type to the variable 'iu' (line 465)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 465, 9), 'iu', int_15823)
    
    # Assigning a Name to a Name (line 465):
    # Getting the type of 'iu' (line 465)
    iu_15824 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 465, 9), 'iu')
    # Assigning a type to the variable 'il' (line 465)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 465, 4), 'il', iu_15824)
    
    
    # Getting the type of 'select' (line 466)
    select_15825 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 466, 7), 'select')
    int_15826 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 466, 17), 'int')
    # Applying the binary operator '!=' (line 466)
    result_ne_15827 = python_operator(stypy.reporting.localization.Localization(__file__, 466, 7), '!=', select_15825, int_15826)
    
    # Testing the type of an if condition (line 466)
    if_condition_15828 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 466, 4), result_ne_15827)
    # Assigning a type to the variable 'if_condition_15828' (line 466)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 466, 4), 'if_condition_15828', if_condition_15828)
    # SSA begins for if statement (line 466)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 467):
    
    # Assigning a Call to a Name (line 467):
    
    # Call to asarray(...): (line 467)
    # Processing the call arguments (line 467)
    # Getting the type of 'select_range' (line 467)
    select_range_15830 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 467, 21), 'select_range', False)
    # Processing the call keyword arguments (line 467)
    kwargs_15831 = {}
    # Getting the type of 'asarray' (line 467)
    asarray_15829 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 467, 13), 'asarray', False)
    # Calling asarray(args, kwargs) (line 467)
    asarray_call_result_15832 = invoke(stypy.reporting.localization.Localization(__file__, 467, 13), asarray_15829, *[select_range_15830], **kwargs_15831)
    
    # Assigning a type to the variable 'sr' (line 467)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 467, 8), 'sr', asarray_call_result_15832)
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'sr' (line 468)
    sr_15833 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 468, 11), 'sr')
    # Obtaining the member 'ndim' of a type (line 468)
    ndim_15834 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 468, 11), sr_15833, 'ndim')
    int_15835 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 468, 22), 'int')
    # Applying the binary operator '!=' (line 468)
    result_ne_15836 = python_operator(stypy.reporting.localization.Localization(__file__, 468, 11), '!=', ndim_15834, int_15835)
    
    
    # Getting the type of 'sr' (line 468)
    sr_15837 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 468, 27), 'sr')
    # Obtaining the member 'size' of a type (line 468)
    size_15838 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 468, 27), sr_15837, 'size')
    int_15839 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 468, 38), 'int')
    # Applying the binary operator '!=' (line 468)
    result_ne_15840 = python_operator(stypy.reporting.localization.Localization(__file__, 468, 27), '!=', size_15838, int_15839)
    
    # Applying the binary operator 'or' (line 468)
    result_or_keyword_15841 = python_operator(stypy.reporting.localization.Localization(__file__, 468, 11), 'or', result_ne_15836, result_ne_15840)
    
    
    # Obtaining the type of the subscript
    int_15842 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 468, 46), 'int')
    # Getting the type of 'sr' (line 468)
    sr_15843 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 468, 43), 'sr')
    # Obtaining the member '__getitem__' of a type (line 468)
    getitem___15844 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 468, 43), sr_15843, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 468)
    subscript_call_result_15845 = invoke(stypy.reporting.localization.Localization(__file__, 468, 43), getitem___15844, int_15842)
    
    
    # Obtaining the type of the subscript
    int_15846 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 468, 54), 'int')
    # Getting the type of 'sr' (line 468)
    sr_15847 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 468, 51), 'sr')
    # Obtaining the member '__getitem__' of a type (line 468)
    getitem___15848 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 468, 51), sr_15847, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 468)
    subscript_call_result_15849 = invoke(stypy.reporting.localization.Localization(__file__, 468, 51), getitem___15848, int_15846)
    
    # Applying the binary operator '<' (line 468)
    result_lt_15850 = python_operator(stypy.reporting.localization.Localization(__file__, 468, 43), '<', subscript_call_result_15845, subscript_call_result_15849)
    
    # Applying the binary operator 'or' (line 468)
    result_or_keyword_15851 = python_operator(stypy.reporting.localization.Localization(__file__, 468, 11), 'or', result_or_keyword_15841, result_lt_15850)
    
    # Testing the type of an if condition (line 468)
    if_condition_15852 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 468, 8), result_or_keyword_15851)
    # Assigning a type to the variable 'if_condition_15852' (line 468)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 468, 8), 'if_condition_15852', if_condition_15852)
    # SSA begins for if statement (line 468)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 469)
    # Processing the call arguments (line 469)
    str_15854 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 469, 29), 'str', 'select_range must be a 2-element array-like in nondecreasing order')
    # Processing the call keyword arguments (line 469)
    kwargs_15855 = {}
    # Getting the type of 'ValueError' (line 469)
    ValueError_15853 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 469, 18), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 469)
    ValueError_call_result_15856 = invoke(stypy.reporting.localization.Localization(__file__, 469, 18), ValueError_15853, *[str_15854], **kwargs_15855)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 469, 12), ValueError_call_result_15856, 'raise parameter', BaseException)
    # SSA join for if statement (line 468)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'select' (line 471)
    select_15857 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 471, 11), 'select')
    int_15858 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 471, 21), 'int')
    # Applying the binary operator '==' (line 471)
    result_eq_15859 = python_operator(stypy.reporting.localization.Localization(__file__, 471, 11), '==', select_15857, int_15858)
    
    # Testing the type of an if condition (line 471)
    if_condition_15860 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 471, 8), result_eq_15859)
    # Assigning a type to the variable 'if_condition_15860' (line 471)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 471, 8), 'if_condition_15860', if_condition_15860)
    # SSA begins for if statement (line 471)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Name to a Tuple (line 472):
    
    # Assigning a Subscript to a Name (line 472):
    
    # Obtaining the type of the subscript
    int_15861 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 472, 12), 'int')
    # Getting the type of 'sr' (line 472)
    sr_15862 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 472, 21), 'sr')
    # Obtaining the member '__getitem__' of a type (line 472)
    getitem___15863 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 472, 12), sr_15862, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 472)
    subscript_call_result_15864 = invoke(stypy.reporting.localization.Localization(__file__, 472, 12), getitem___15863, int_15861)
    
    # Assigning a type to the variable 'tuple_var_assignment_14081' (line 472)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 472, 12), 'tuple_var_assignment_14081', subscript_call_result_15864)
    
    # Assigning a Subscript to a Name (line 472):
    
    # Obtaining the type of the subscript
    int_15865 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 472, 12), 'int')
    # Getting the type of 'sr' (line 472)
    sr_15866 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 472, 21), 'sr')
    # Obtaining the member '__getitem__' of a type (line 472)
    getitem___15867 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 472, 12), sr_15866, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 472)
    subscript_call_result_15868 = invoke(stypy.reporting.localization.Localization(__file__, 472, 12), getitem___15867, int_15865)
    
    # Assigning a type to the variable 'tuple_var_assignment_14082' (line 472)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 472, 12), 'tuple_var_assignment_14082', subscript_call_result_15868)
    
    # Assigning a Name to a Name (line 472):
    # Getting the type of 'tuple_var_assignment_14081' (line 472)
    tuple_var_assignment_14081_15869 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 472, 12), 'tuple_var_assignment_14081')
    # Assigning a type to the variable 'vl' (line 472)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 472, 12), 'vl', tuple_var_assignment_14081_15869)
    
    # Assigning a Name to a Name (line 472):
    # Getting the type of 'tuple_var_assignment_14082' (line 472)
    tuple_var_assignment_14082_15870 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 472, 12), 'tuple_var_assignment_14082')
    # Assigning a type to the variable 'vu' (line 472)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 472, 16), 'vu', tuple_var_assignment_14082_15870)
    
    
    # Getting the type of 'max_ev' (line 473)
    max_ev_15871 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 473, 15), 'max_ev')
    int_15872 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 473, 25), 'int')
    # Applying the binary operator '==' (line 473)
    result_eq_15873 = python_operator(stypy.reporting.localization.Localization(__file__, 473, 15), '==', max_ev_15871, int_15872)
    
    # Testing the type of an if condition (line 473)
    if_condition_15874 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 473, 12), result_eq_15873)
    # Assigning a type to the variable 'if_condition_15874' (line 473)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 473, 12), 'if_condition_15874', if_condition_15874)
    # SSA begins for if statement (line 473)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Name to a Name (line 474):
    
    # Assigning a Name to a Name (line 474):
    # Getting the type of 'max_len' (line 474)
    max_len_15875 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 474, 25), 'max_len')
    # Assigning a type to the variable 'max_ev' (line 474)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 474, 16), 'max_ev', max_len_15875)
    # SSA join for if statement (line 473)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA branch for the else part of an if statement (line 471)
    module_type_store.open_ssa_branch('else')
    
    
    
    # Call to lower(...): (line 476)
    # Processing the call keyword arguments (line 476)
    kwargs_15880 = {}
    # Getting the type of 'sr' (line 476)
    sr_15876 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 476, 15), 'sr', False)
    # Obtaining the member 'dtype' of a type (line 476)
    dtype_15877 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 476, 15), sr_15876, 'dtype')
    # Obtaining the member 'char' of a type (line 476)
    char_15878 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 476, 15), dtype_15877, 'char')
    # Obtaining the member 'lower' of a type (line 476)
    lower_15879 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 476, 15), char_15878, 'lower')
    # Calling lower(args, kwargs) (line 476)
    lower_call_result_15881 = invoke(stypy.reporting.localization.Localization(__file__, 476, 15), lower_15879, *[], **kwargs_15880)
    
    str_15882 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 476, 44), 'str', 'lih')
    # Applying the binary operator 'notin' (line 476)
    result_contains_15883 = python_operator(stypy.reporting.localization.Localization(__file__, 476, 15), 'notin', lower_call_result_15881, str_15882)
    
    # Testing the type of an if condition (line 476)
    if_condition_15884 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 476, 12), result_contains_15883)
    # Assigning a type to the variable 'if_condition_15884' (line 476)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 476, 12), 'if_condition_15884', if_condition_15884)
    # SSA begins for if statement (line 476)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 477)
    # Processing the call arguments (line 477)
    str_15886 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 477, 33), 'str', 'when using select="i", select_range must contain integers, got dtype %s')
    # Getting the type of 'sr' (line 478)
    sr_15887 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 478, 68), 'sr', False)
    # Obtaining the member 'dtype' of a type (line 478)
    dtype_15888 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 478, 68), sr_15887, 'dtype')
    # Applying the binary operator '%' (line 477)
    result_mod_15889 = python_operator(stypy.reporting.localization.Localization(__file__, 477, 33), '%', str_15886, dtype_15888)
    
    # Processing the call keyword arguments (line 477)
    kwargs_15890 = {}
    # Getting the type of 'ValueError' (line 477)
    ValueError_15885 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 477, 22), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 477)
    ValueError_call_result_15891 = invoke(stypy.reporting.localization.Localization(__file__, 477, 22), ValueError_15885, *[result_mod_15889], **kwargs_15890)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 477, 16), ValueError_call_result_15891, 'raise parameter', BaseException)
    # SSA join for if statement (line 476)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a BinOp to a Tuple (line 480):
    
    # Assigning a Subscript to a Name (line 480):
    
    # Obtaining the type of the subscript
    int_15892 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 480, 12), 'int')
    # Getting the type of 'sr' (line 480)
    sr_15893 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 480, 21), 'sr')
    int_15894 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 480, 26), 'int')
    # Applying the binary operator '+' (line 480)
    result_add_15895 = python_operator(stypy.reporting.localization.Localization(__file__, 480, 21), '+', sr_15893, int_15894)
    
    # Obtaining the member '__getitem__' of a type (line 480)
    getitem___15896 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 480, 12), result_add_15895, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 480)
    subscript_call_result_15897 = invoke(stypy.reporting.localization.Localization(__file__, 480, 12), getitem___15896, int_15892)
    
    # Assigning a type to the variable 'tuple_var_assignment_14083' (line 480)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 480, 12), 'tuple_var_assignment_14083', subscript_call_result_15897)
    
    # Assigning a Subscript to a Name (line 480):
    
    # Obtaining the type of the subscript
    int_15898 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 480, 12), 'int')
    # Getting the type of 'sr' (line 480)
    sr_15899 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 480, 21), 'sr')
    int_15900 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 480, 26), 'int')
    # Applying the binary operator '+' (line 480)
    result_add_15901 = python_operator(stypy.reporting.localization.Localization(__file__, 480, 21), '+', sr_15899, int_15900)
    
    # Obtaining the member '__getitem__' of a type (line 480)
    getitem___15902 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 480, 12), result_add_15901, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 480)
    subscript_call_result_15903 = invoke(stypy.reporting.localization.Localization(__file__, 480, 12), getitem___15902, int_15898)
    
    # Assigning a type to the variable 'tuple_var_assignment_14084' (line 480)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 480, 12), 'tuple_var_assignment_14084', subscript_call_result_15903)
    
    # Assigning a Name to a Name (line 480):
    # Getting the type of 'tuple_var_assignment_14083' (line 480)
    tuple_var_assignment_14083_15904 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 480, 12), 'tuple_var_assignment_14083')
    # Assigning a type to the variable 'il' (line 480)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 480, 12), 'il', tuple_var_assignment_14083_15904)
    
    # Assigning a Name to a Name (line 480):
    # Getting the type of 'tuple_var_assignment_14084' (line 480)
    tuple_var_assignment_14084_15905 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 480, 12), 'tuple_var_assignment_14084')
    # Assigning a type to the variable 'iu' (line 480)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 480, 16), 'iu', tuple_var_assignment_14084_15905)
    
    
    # Evaluating a boolean operation
    
    
    # Call to min(...): (line 481)
    # Processing the call arguments (line 481)
    # Getting the type of 'il' (line 481)
    il_15907 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 481, 19), 'il', False)
    # Getting the type of 'iu' (line 481)
    iu_15908 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 481, 23), 'iu', False)
    # Processing the call keyword arguments (line 481)
    kwargs_15909 = {}
    # Getting the type of 'min' (line 481)
    min_15906 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 481, 15), 'min', False)
    # Calling min(args, kwargs) (line 481)
    min_call_result_15910 = invoke(stypy.reporting.localization.Localization(__file__, 481, 15), min_15906, *[il_15907, iu_15908], **kwargs_15909)
    
    int_15911 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 481, 29), 'int')
    # Applying the binary operator '<' (line 481)
    result_lt_15912 = python_operator(stypy.reporting.localization.Localization(__file__, 481, 15), '<', min_call_result_15910, int_15911)
    
    
    
    # Call to max(...): (line 481)
    # Processing the call arguments (line 481)
    # Getting the type of 'il' (line 481)
    il_15914 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 481, 38), 'il', False)
    # Getting the type of 'iu' (line 481)
    iu_15915 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 481, 42), 'iu', False)
    # Processing the call keyword arguments (line 481)
    kwargs_15916 = {}
    # Getting the type of 'max' (line 481)
    max_15913 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 481, 34), 'max', False)
    # Calling max(args, kwargs) (line 481)
    max_call_result_15917 = invoke(stypy.reporting.localization.Localization(__file__, 481, 34), max_15913, *[il_15914, iu_15915], **kwargs_15916)
    
    # Getting the type of 'max_len' (line 481)
    max_len_15918 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 481, 48), 'max_len')
    # Applying the binary operator '>' (line 481)
    result_gt_15919 = python_operator(stypy.reporting.localization.Localization(__file__, 481, 34), '>', max_call_result_15917, max_len_15918)
    
    # Applying the binary operator 'or' (line 481)
    result_or_keyword_15920 = python_operator(stypy.reporting.localization.Localization(__file__, 481, 15), 'or', result_lt_15912, result_gt_15919)
    
    # Testing the type of an if condition (line 481)
    if_condition_15921 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 481, 12), result_or_keyword_15920)
    # Assigning a type to the variable 'if_condition_15921' (line 481)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 481, 12), 'if_condition_15921', if_condition_15921)
    # SSA begins for if statement (line 481)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 482)
    # Processing the call arguments (line 482)
    str_15923 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 482, 33), 'str', 'select_range out of bounds')
    # Processing the call keyword arguments (line 482)
    kwargs_15924 = {}
    # Getting the type of 'ValueError' (line 482)
    ValueError_15922 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 482, 22), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 482)
    ValueError_call_result_15925 = invoke(stypy.reporting.localization.Localization(__file__, 482, 22), ValueError_15922, *[str_15923], **kwargs_15924)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 482, 16), ValueError_call_result_15925, 'raise parameter', BaseException)
    # SSA join for if statement (line 481)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a BinOp to a Name (line 483):
    
    # Assigning a BinOp to a Name (line 483):
    # Getting the type of 'iu' (line 483)
    iu_15926 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 483, 21), 'iu')
    # Getting the type of 'il' (line 483)
    il_15927 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 483, 26), 'il')
    # Applying the binary operator '-' (line 483)
    result_sub_15928 = python_operator(stypy.reporting.localization.Localization(__file__, 483, 21), '-', iu_15926, il_15927)
    
    int_15929 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 483, 31), 'int')
    # Applying the binary operator '+' (line 483)
    result_add_15930 = python_operator(stypy.reporting.localization.Localization(__file__, 483, 29), '+', result_sub_15928, int_15929)
    
    # Assigning a type to the variable 'max_ev' (line 483)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 483, 12), 'max_ev', result_add_15930)
    # SSA join for if statement (line 471)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 466)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Obtaining an instance of the builtin type 'tuple' (line 484)
    tuple_15931 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 484, 11), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 484)
    # Adding element type (line 484)
    # Getting the type of 'select' (line 484)
    select_15932 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 484, 11), 'select')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 484, 11), tuple_15931, select_15932)
    # Adding element type (line 484)
    # Getting the type of 'vl' (line 484)
    vl_15933 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 484, 19), 'vl')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 484, 11), tuple_15931, vl_15933)
    # Adding element type (line 484)
    # Getting the type of 'vu' (line 484)
    vu_15934 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 484, 23), 'vu')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 484, 11), tuple_15931, vu_15934)
    # Adding element type (line 484)
    # Getting the type of 'il' (line 484)
    il_15935 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 484, 27), 'il')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 484, 11), tuple_15931, il_15935)
    # Adding element type (line 484)
    # Getting the type of 'iu' (line 484)
    iu_15936 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 484, 31), 'iu')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 484, 11), tuple_15931, iu_15936)
    # Adding element type (line 484)
    # Getting the type of 'max_ev' (line 484)
    max_ev_15937 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 484, 35), 'max_ev')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 484, 11), tuple_15931, max_ev_15937)
    
    # Assigning a type to the variable 'stypy_return_type' (line 484)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 484, 4), 'stypy_return_type', tuple_15931)
    
    # ################# End of '_check_select(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_check_select' in the type store
    # Getting the type of 'stypy_return_type' (line 456)
    stypy_return_type_15938 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 456, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_15938)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_check_select'
    return stypy_return_type_15938

# Assigning a type to the variable '_check_select' (line 456)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 456, 0), '_check_select', _check_select)

@norecursion
def eig_banded(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'False' (line 487)
    False_15939 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 487, 29), 'False')
    # Getting the type of 'False' (line 487)
    False_15940 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 487, 49), 'False')
    # Getting the type of 'False' (line 487)
    False_15941 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 487, 73), 'False')
    str_15942 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 488, 22), 'str', 'a')
    # Getting the type of 'None' (line 488)
    None_15943 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 488, 40), 'None')
    int_15944 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 488, 53), 'int')
    # Getting the type of 'True' (line 488)
    True_15945 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 488, 69), 'True')
    defaults = [False_15939, False_15940, False_15941, str_15942, None_15943, int_15944, True_15945]
    # Create a new context for function 'eig_banded'
    module_type_store = module_type_store.open_function_context('eig_banded', 487, 0, False)
    
    # Passed parameters checking function
    eig_banded.stypy_localization = localization
    eig_banded.stypy_type_of_self = None
    eig_banded.stypy_type_store = module_type_store
    eig_banded.stypy_function_name = 'eig_banded'
    eig_banded.stypy_param_names_list = ['a_band', 'lower', 'eigvals_only', 'overwrite_a_band', 'select', 'select_range', 'max_ev', 'check_finite']
    eig_banded.stypy_varargs_param_name = None
    eig_banded.stypy_kwargs_param_name = None
    eig_banded.stypy_call_defaults = defaults
    eig_banded.stypy_call_varargs = varargs
    eig_banded.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'eig_banded', ['a_band', 'lower', 'eigvals_only', 'overwrite_a_band', 'select', 'select_range', 'max_ev', 'check_finite'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'eig_banded', localization, ['a_band', 'lower', 'eigvals_only', 'overwrite_a_band', 'select', 'select_range', 'max_ev', 'check_finite'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'eig_banded(...)' code ##################

    str_15946 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 574, (-1)), 'str', "\n    Solve real symmetric or complex hermitian band matrix eigenvalue problem.\n\n    Find eigenvalues w and optionally right eigenvectors v of a::\n\n        a v[:,i] = w[i] v[:,i]\n        v.H v    = identity\n\n    The matrix a is stored in a_band either in lower diagonal or upper\n    diagonal ordered form:\n\n        a_band[u + i - j, j] == a[i,j]        (if upper form; i <= j)\n        a_band[    i - j, j] == a[i,j]        (if lower form; i >= j)\n\n    where u is the number of bands above the diagonal.\n\n    Example of a_band (shape of a is (6,6), u=2)::\n\n        upper form:\n        *   *   a02 a13 a24 a35\n        *   a01 a12 a23 a34 a45\n        a00 a11 a22 a33 a44 a55\n\n        lower form:\n        a00 a11 a22 a33 a44 a55\n        a10 a21 a32 a43 a54 *\n        a20 a31 a42 a53 *   *\n\n    Cells marked with * are not used.\n\n    Parameters\n    ----------\n    a_band : (u+1, M) array_like\n        The bands of the M by M matrix a.\n    lower : bool, optional\n        Is the matrix in the lower form. (Default is upper form)\n    eigvals_only : bool, optional\n        Compute only the eigenvalues and no eigenvectors.\n        (Default: calculate also eigenvectors)\n    overwrite_a_band : bool, optional\n        Discard data in a_band (may enhance performance)\n    select : {'a', 'v', 'i'}, optional\n        Which eigenvalues to calculate\n\n        ======  ========================================\n        select  calculated\n        ======  ========================================\n        'a'     All eigenvalues\n        'v'     Eigenvalues in the interval (min, max]\n        'i'     Eigenvalues with indices min <= i <= max\n        ======  ========================================\n    select_range : (min, max), optional\n        Range of selected eigenvalues\n    max_ev : int, optional\n        For select=='v', maximum number of eigenvalues expected.\n        For other values of select, has no meaning.\n\n        In doubt, leave this parameter untouched.\n\n    check_finite : bool, optional\n        Whether to check that the input matrix contains only finite numbers.\n        Disabling may give a performance gain, but may result in problems\n        (crashes, non-termination) if the inputs do contain infinities or NaNs.\n\n    Returns\n    -------\n    w : (M,) ndarray\n        The eigenvalues, in ascending order, each repeated according to its\n        multiplicity.\n    v : (M, M) float or complex ndarray\n        The normalized eigenvector corresponding to the eigenvalue w[i] is\n        the column v[:,i].\n\n    Raises\n    ------\n    LinAlgError\n        If eigenvalue computation does not converge.\n\n    See Also\n    --------\n    eigvals_banded : eigenvalues for symmetric/Hermitian band matrices\n    eig : eigenvalues and right eigenvectors of general arrays.\n    eigh : eigenvalues and right eigenvectors for symmetric/Hermitian arrays\n    eigh_tridiagonal : eigenvalues and right eiegenvectors for\n        symmetric/Hermitian tridiagonal matrices\n    ")
    
    
    # Evaluating a boolean operation
    # Getting the type of 'eigvals_only' (line 575)
    eigvals_only_15947 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 575, 7), 'eigvals_only')
    # Getting the type of 'overwrite_a_band' (line 575)
    overwrite_a_band_15948 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 575, 23), 'overwrite_a_band')
    # Applying the binary operator 'or' (line 575)
    result_or_keyword_15949 = python_operator(stypy.reporting.localization.Localization(__file__, 575, 7), 'or', eigvals_only_15947, overwrite_a_band_15948)
    
    # Testing the type of an if condition (line 575)
    if_condition_15950 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 575, 4), result_or_keyword_15949)
    # Assigning a type to the variable 'if_condition_15950' (line 575)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 575, 4), 'if_condition_15950', if_condition_15950)
    # SSA begins for if statement (line 575)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 576):
    
    # Assigning a Call to a Name (line 576):
    
    # Call to _asarray_validated(...): (line 576)
    # Processing the call arguments (line 576)
    # Getting the type of 'a_band' (line 576)
    a_band_15952 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 576, 32), 'a_band', False)
    # Processing the call keyword arguments (line 576)
    # Getting the type of 'check_finite' (line 576)
    check_finite_15953 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 576, 53), 'check_finite', False)
    keyword_15954 = check_finite_15953
    kwargs_15955 = {'check_finite': keyword_15954}
    # Getting the type of '_asarray_validated' (line 576)
    _asarray_validated_15951 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 576, 13), '_asarray_validated', False)
    # Calling _asarray_validated(args, kwargs) (line 576)
    _asarray_validated_call_result_15956 = invoke(stypy.reporting.localization.Localization(__file__, 576, 13), _asarray_validated_15951, *[a_band_15952], **kwargs_15955)
    
    # Assigning a type to the variable 'a1' (line 576)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 576, 8), 'a1', _asarray_validated_call_result_15956)
    
    # Assigning a BoolOp to a Name (line 577):
    
    # Assigning a BoolOp to a Name (line 577):
    
    # Evaluating a boolean operation
    # Getting the type of 'overwrite_a_band' (line 577)
    overwrite_a_band_15957 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 577, 27), 'overwrite_a_band')
    
    # Call to _datacopied(...): (line 577)
    # Processing the call arguments (line 577)
    # Getting the type of 'a1' (line 577)
    a1_15959 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 577, 60), 'a1', False)
    # Getting the type of 'a_band' (line 577)
    a_band_15960 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 577, 64), 'a_band', False)
    # Processing the call keyword arguments (line 577)
    kwargs_15961 = {}
    # Getting the type of '_datacopied' (line 577)
    _datacopied_15958 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 577, 48), '_datacopied', False)
    # Calling _datacopied(args, kwargs) (line 577)
    _datacopied_call_result_15962 = invoke(stypy.reporting.localization.Localization(__file__, 577, 48), _datacopied_15958, *[a1_15959, a_band_15960], **kwargs_15961)
    
    # Applying the binary operator 'or' (line 577)
    result_or_keyword_15963 = python_operator(stypy.reporting.localization.Localization(__file__, 577, 27), 'or', overwrite_a_band_15957, _datacopied_call_result_15962)
    
    # Assigning a type to the variable 'overwrite_a_band' (line 577)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 577, 8), 'overwrite_a_band', result_or_keyword_15963)
    # SSA branch for the else part of an if statement (line 575)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Call to a Name (line 579):
    
    # Assigning a Call to a Name (line 579):
    
    # Call to array(...): (line 579)
    # Processing the call arguments (line 579)
    # Getting the type of 'a_band' (line 579)
    a_band_15965 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 579, 19), 'a_band', False)
    # Processing the call keyword arguments (line 579)
    kwargs_15966 = {}
    # Getting the type of 'array' (line 579)
    array_15964 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 579, 13), 'array', False)
    # Calling array(args, kwargs) (line 579)
    array_call_result_15967 = invoke(stypy.reporting.localization.Localization(__file__, 579, 13), array_15964, *[a_band_15965], **kwargs_15966)
    
    # Assigning a type to the variable 'a1' (line 579)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 579, 8), 'a1', array_call_result_15967)
    
    
    # Evaluating a boolean operation
    
    # Call to issubclass(...): (line 580)
    # Processing the call arguments (line 580)
    # Getting the type of 'a1' (line 580)
    a1_15969 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 580, 22), 'a1', False)
    # Obtaining the member 'dtype' of a type (line 580)
    dtype_15970 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 580, 22), a1_15969, 'dtype')
    # Obtaining the member 'type' of a type (line 580)
    type_15971 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 580, 22), dtype_15970, 'type')
    # Getting the type of 'inexact' (line 580)
    inexact_15972 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 580, 37), 'inexact', False)
    # Processing the call keyword arguments (line 580)
    kwargs_15973 = {}
    # Getting the type of 'issubclass' (line 580)
    issubclass_15968 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 580, 11), 'issubclass', False)
    # Calling issubclass(args, kwargs) (line 580)
    issubclass_call_result_15974 = invoke(stypy.reporting.localization.Localization(__file__, 580, 11), issubclass_15968, *[type_15971, inexact_15972], **kwargs_15973)
    
    
    
    # Call to all(...): (line 580)
    # Processing the call keyword arguments (line 580)
    kwargs_15980 = {}
    
    # Call to isfinite(...): (line 580)
    # Processing the call arguments (line 580)
    # Getting the type of 'a1' (line 580)
    a1_15976 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 580, 63), 'a1', False)
    # Processing the call keyword arguments (line 580)
    kwargs_15977 = {}
    # Getting the type of 'isfinite' (line 580)
    isfinite_15975 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 580, 54), 'isfinite', False)
    # Calling isfinite(args, kwargs) (line 580)
    isfinite_call_result_15978 = invoke(stypy.reporting.localization.Localization(__file__, 580, 54), isfinite_15975, *[a1_15976], **kwargs_15977)
    
    # Obtaining the member 'all' of a type (line 580)
    all_15979 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 580, 54), isfinite_call_result_15978, 'all')
    # Calling all(args, kwargs) (line 580)
    all_call_result_15981 = invoke(stypy.reporting.localization.Localization(__file__, 580, 54), all_15979, *[], **kwargs_15980)
    
    # Applying the 'not' unary operator (line 580)
    result_not__15982 = python_operator(stypy.reporting.localization.Localization(__file__, 580, 50), 'not', all_call_result_15981)
    
    # Applying the binary operator 'and' (line 580)
    result_and_keyword_15983 = python_operator(stypy.reporting.localization.Localization(__file__, 580, 11), 'and', issubclass_call_result_15974, result_not__15982)
    
    # Testing the type of an if condition (line 580)
    if_condition_15984 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 580, 8), result_and_keyword_15983)
    # Assigning a type to the variable 'if_condition_15984' (line 580)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 580, 8), 'if_condition_15984', if_condition_15984)
    # SSA begins for if statement (line 580)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 581)
    # Processing the call arguments (line 581)
    str_15986 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 581, 29), 'str', 'array must not contain infs or NaNs')
    # Processing the call keyword arguments (line 581)
    kwargs_15987 = {}
    # Getting the type of 'ValueError' (line 581)
    ValueError_15985 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 581, 18), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 581)
    ValueError_call_result_15988 = invoke(stypy.reporting.localization.Localization(__file__, 581, 18), ValueError_15985, *[str_15986], **kwargs_15987)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 581, 12), ValueError_call_result_15988, 'raise parameter', BaseException)
    # SSA join for if statement (line 580)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Num to a Name (line 582):
    
    # Assigning a Num to a Name (line 582):
    int_15989 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 582, 27), 'int')
    # Assigning a type to the variable 'overwrite_a_band' (line 582)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 582, 8), 'overwrite_a_band', int_15989)
    # SSA join for if statement (line 575)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    
    # Call to len(...): (line 584)
    # Processing the call arguments (line 584)
    # Getting the type of 'a1' (line 584)
    a1_15991 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 584, 11), 'a1', False)
    # Obtaining the member 'shape' of a type (line 584)
    shape_15992 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 584, 11), a1_15991, 'shape')
    # Processing the call keyword arguments (line 584)
    kwargs_15993 = {}
    # Getting the type of 'len' (line 584)
    len_15990 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 584, 7), 'len', False)
    # Calling len(args, kwargs) (line 584)
    len_call_result_15994 = invoke(stypy.reporting.localization.Localization(__file__, 584, 7), len_15990, *[shape_15992], **kwargs_15993)
    
    int_15995 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 584, 24), 'int')
    # Applying the binary operator '!=' (line 584)
    result_ne_15996 = python_operator(stypy.reporting.localization.Localization(__file__, 584, 7), '!=', len_call_result_15994, int_15995)
    
    # Testing the type of an if condition (line 584)
    if_condition_15997 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 584, 4), result_ne_15996)
    # Assigning a type to the variable 'if_condition_15997' (line 584)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 584, 4), 'if_condition_15997', if_condition_15997)
    # SSA begins for if statement (line 584)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 585)
    # Processing the call arguments (line 585)
    str_15999 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 585, 25), 'str', 'expected two-dimensional array')
    # Processing the call keyword arguments (line 585)
    kwargs_16000 = {}
    # Getting the type of 'ValueError' (line 585)
    ValueError_15998 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 585, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 585)
    ValueError_call_result_16001 = invoke(stypy.reporting.localization.Localization(__file__, 585, 14), ValueError_15998, *[str_15999], **kwargs_16000)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 585, 8), ValueError_call_result_16001, 'raise parameter', BaseException)
    # SSA join for if statement (line 584)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Tuple (line 586):
    
    # Assigning a Subscript to a Name (line 586):
    
    # Obtaining the type of the subscript
    int_16002 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 586, 4), 'int')
    
    # Call to _check_select(...): (line 586)
    # Processing the call arguments (line 586)
    # Getting the type of 'select' (line 587)
    select_16004 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 587, 8), 'select', False)
    # Getting the type of 'select_range' (line 587)
    select_range_16005 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 587, 16), 'select_range', False)
    # Getting the type of 'max_ev' (line 587)
    max_ev_16006 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 587, 30), 'max_ev', False)
    
    # Obtaining the type of the subscript
    int_16007 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 587, 47), 'int')
    # Getting the type of 'a1' (line 587)
    a1_16008 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 587, 38), 'a1', False)
    # Obtaining the member 'shape' of a type (line 587)
    shape_16009 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 587, 38), a1_16008, 'shape')
    # Obtaining the member '__getitem__' of a type (line 587)
    getitem___16010 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 587, 38), shape_16009, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 587)
    subscript_call_result_16011 = invoke(stypy.reporting.localization.Localization(__file__, 587, 38), getitem___16010, int_16007)
    
    # Processing the call keyword arguments (line 586)
    kwargs_16012 = {}
    # Getting the type of '_check_select' (line 586)
    _check_select_16003 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 586, 37), '_check_select', False)
    # Calling _check_select(args, kwargs) (line 586)
    _check_select_call_result_16013 = invoke(stypy.reporting.localization.Localization(__file__, 586, 37), _check_select_16003, *[select_16004, select_range_16005, max_ev_16006, subscript_call_result_16011], **kwargs_16012)
    
    # Obtaining the member '__getitem__' of a type (line 586)
    getitem___16014 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 586, 4), _check_select_call_result_16013, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 586)
    subscript_call_result_16015 = invoke(stypy.reporting.localization.Localization(__file__, 586, 4), getitem___16014, int_16002)
    
    # Assigning a type to the variable 'tuple_var_assignment_14085' (line 586)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 586, 4), 'tuple_var_assignment_14085', subscript_call_result_16015)
    
    # Assigning a Subscript to a Name (line 586):
    
    # Obtaining the type of the subscript
    int_16016 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 586, 4), 'int')
    
    # Call to _check_select(...): (line 586)
    # Processing the call arguments (line 586)
    # Getting the type of 'select' (line 587)
    select_16018 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 587, 8), 'select', False)
    # Getting the type of 'select_range' (line 587)
    select_range_16019 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 587, 16), 'select_range', False)
    # Getting the type of 'max_ev' (line 587)
    max_ev_16020 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 587, 30), 'max_ev', False)
    
    # Obtaining the type of the subscript
    int_16021 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 587, 47), 'int')
    # Getting the type of 'a1' (line 587)
    a1_16022 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 587, 38), 'a1', False)
    # Obtaining the member 'shape' of a type (line 587)
    shape_16023 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 587, 38), a1_16022, 'shape')
    # Obtaining the member '__getitem__' of a type (line 587)
    getitem___16024 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 587, 38), shape_16023, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 587)
    subscript_call_result_16025 = invoke(stypy.reporting.localization.Localization(__file__, 587, 38), getitem___16024, int_16021)
    
    # Processing the call keyword arguments (line 586)
    kwargs_16026 = {}
    # Getting the type of '_check_select' (line 586)
    _check_select_16017 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 586, 37), '_check_select', False)
    # Calling _check_select(args, kwargs) (line 586)
    _check_select_call_result_16027 = invoke(stypy.reporting.localization.Localization(__file__, 586, 37), _check_select_16017, *[select_16018, select_range_16019, max_ev_16020, subscript_call_result_16025], **kwargs_16026)
    
    # Obtaining the member '__getitem__' of a type (line 586)
    getitem___16028 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 586, 4), _check_select_call_result_16027, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 586)
    subscript_call_result_16029 = invoke(stypy.reporting.localization.Localization(__file__, 586, 4), getitem___16028, int_16016)
    
    # Assigning a type to the variable 'tuple_var_assignment_14086' (line 586)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 586, 4), 'tuple_var_assignment_14086', subscript_call_result_16029)
    
    # Assigning a Subscript to a Name (line 586):
    
    # Obtaining the type of the subscript
    int_16030 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 586, 4), 'int')
    
    # Call to _check_select(...): (line 586)
    # Processing the call arguments (line 586)
    # Getting the type of 'select' (line 587)
    select_16032 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 587, 8), 'select', False)
    # Getting the type of 'select_range' (line 587)
    select_range_16033 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 587, 16), 'select_range', False)
    # Getting the type of 'max_ev' (line 587)
    max_ev_16034 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 587, 30), 'max_ev', False)
    
    # Obtaining the type of the subscript
    int_16035 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 587, 47), 'int')
    # Getting the type of 'a1' (line 587)
    a1_16036 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 587, 38), 'a1', False)
    # Obtaining the member 'shape' of a type (line 587)
    shape_16037 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 587, 38), a1_16036, 'shape')
    # Obtaining the member '__getitem__' of a type (line 587)
    getitem___16038 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 587, 38), shape_16037, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 587)
    subscript_call_result_16039 = invoke(stypy.reporting.localization.Localization(__file__, 587, 38), getitem___16038, int_16035)
    
    # Processing the call keyword arguments (line 586)
    kwargs_16040 = {}
    # Getting the type of '_check_select' (line 586)
    _check_select_16031 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 586, 37), '_check_select', False)
    # Calling _check_select(args, kwargs) (line 586)
    _check_select_call_result_16041 = invoke(stypy.reporting.localization.Localization(__file__, 586, 37), _check_select_16031, *[select_16032, select_range_16033, max_ev_16034, subscript_call_result_16039], **kwargs_16040)
    
    # Obtaining the member '__getitem__' of a type (line 586)
    getitem___16042 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 586, 4), _check_select_call_result_16041, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 586)
    subscript_call_result_16043 = invoke(stypy.reporting.localization.Localization(__file__, 586, 4), getitem___16042, int_16030)
    
    # Assigning a type to the variable 'tuple_var_assignment_14087' (line 586)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 586, 4), 'tuple_var_assignment_14087', subscript_call_result_16043)
    
    # Assigning a Subscript to a Name (line 586):
    
    # Obtaining the type of the subscript
    int_16044 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 586, 4), 'int')
    
    # Call to _check_select(...): (line 586)
    # Processing the call arguments (line 586)
    # Getting the type of 'select' (line 587)
    select_16046 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 587, 8), 'select', False)
    # Getting the type of 'select_range' (line 587)
    select_range_16047 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 587, 16), 'select_range', False)
    # Getting the type of 'max_ev' (line 587)
    max_ev_16048 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 587, 30), 'max_ev', False)
    
    # Obtaining the type of the subscript
    int_16049 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 587, 47), 'int')
    # Getting the type of 'a1' (line 587)
    a1_16050 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 587, 38), 'a1', False)
    # Obtaining the member 'shape' of a type (line 587)
    shape_16051 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 587, 38), a1_16050, 'shape')
    # Obtaining the member '__getitem__' of a type (line 587)
    getitem___16052 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 587, 38), shape_16051, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 587)
    subscript_call_result_16053 = invoke(stypy.reporting.localization.Localization(__file__, 587, 38), getitem___16052, int_16049)
    
    # Processing the call keyword arguments (line 586)
    kwargs_16054 = {}
    # Getting the type of '_check_select' (line 586)
    _check_select_16045 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 586, 37), '_check_select', False)
    # Calling _check_select(args, kwargs) (line 586)
    _check_select_call_result_16055 = invoke(stypy.reporting.localization.Localization(__file__, 586, 37), _check_select_16045, *[select_16046, select_range_16047, max_ev_16048, subscript_call_result_16053], **kwargs_16054)
    
    # Obtaining the member '__getitem__' of a type (line 586)
    getitem___16056 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 586, 4), _check_select_call_result_16055, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 586)
    subscript_call_result_16057 = invoke(stypy.reporting.localization.Localization(__file__, 586, 4), getitem___16056, int_16044)
    
    # Assigning a type to the variable 'tuple_var_assignment_14088' (line 586)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 586, 4), 'tuple_var_assignment_14088', subscript_call_result_16057)
    
    # Assigning a Subscript to a Name (line 586):
    
    # Obtaining the type of the subscript
    int_16058 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 586, 4), 'int')
    
    # Call to _check_select(...): (line 586)
    # Processing the call arguments (line 586)
    # Getting the type of 'select' (line 587)
    select_16060 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 587, 8), 'select', False)
    # Getting the type of 'select_range' (line 587)
    select_range_16061 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 587, 16), 'select_range', False)
    # Getting the type of 'max_ev' (line 587)
    max_ev_16062 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 587, 30), 'max_ev', False)
    
    # Obtaining the type of the subscript
    int_16063 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 587, 47), 'int')
    # Getting the type of 'a1' (line 587)
    a1_16064 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 587, 38), 'a1', False)
    # Obtaining the member 'shape' of a type (line 587)
    shape_16065 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 587, 38), a1_16064, 'shape')
    # Obtaining the member '__getitem__' of a type (line 587)
    getitem___16066 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 587, 38), shape_16065, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 587)
    subscript_call_result_16067 = invoke(stypy.reporting.localization.Localization(__file__, 587, 38), getitem___16066, int_16063)
    
    # Processing the call keyword arguments (line 586)
    kwargs_16068 = {}
    # Getting the type of '_check_select' (line 586)
    _check_select_16059 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 586, 37), '_check_select', False)
    # Calling _check_select(args, kwargs) (line 586)
    _check_select_call_result_16069 = invoke(stypy.reporting.localization.Localization(__file__, 586, 37), _check_select_16059, *[select_16060, select_range_16061, max_ev_16062, subscript_call_result_16067], **kwargs_16068)
    
    # Obtaining the member '__getitem__' of a type (line 586)
    getitem___16070 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 586, 4), _check_select_call_result_16069, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 586)
    subscript_call_result_16071 = invoke(stypy.reporting.localization.Localization(__file__, 586, 4), getitem___16070, int_16058)
    
    # Assigning a type to the variable 'tuple_var_assignment_14089' (line 586)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 586, 4), 'tuple_var_assignment_14089', subscript_call_result_16071)
    
    # Assigning a Subscript to a Name (line 586):
    
    # Obtaining the type of the subscript
    int_16072 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 586, 4), 'int')
    
    # Call to _check_select(...): (line 586)
    # Processing the call arguments (line 586)
    # Getting the type of 'select' (line 587)
    select_16074 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 587, 8), 'select', False)
    # Getting the type of 'select_range' (line 587)
    select_range_16075 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 587, 16), 'select_range', False)
    # Getting the type of 'max_ev' (line 587)
    max_ev_16076 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 587, 30), 'max_ev', False)
    
    # Obtaining the type of the subscript
    int_16077 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 587, 47), 'int')
    # Getting the type of 'a1' (line 587)
    a1_16078 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 587, 38), 'a1', False)
    # Obtaining the member 'shape' of a type (line 587)
    shape_16079 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 587, 38), a1_16078, 'shape')
    # Obtaining the member '__getitem__' of a type (line 587)
    getitem___16080 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 587, 38), shape_16079, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 587)
    subscript_call_result_16081 = invoke(stypy.reporting.localization.Localization(__file__, 587, 38), getitem___16080, int_16077)
    
    # Processing the call keyword arguments (line 586)
    kwargs_16082 = {}
    # Getting the type of '_check_select' (line 586)
    _check_select_16073 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 586, 37), '_check_select', False)
    # Calling _check_select(args, kwargs) (line 586)
    _check_select_call_result_16083 = invoke(stypy.reporting.localization.Localization(__file__, 586, 37), _check_select_16073, *[select_16074, select_range_16075, max_ev_16076, subscript_call_result_16081], **kwargs_16082)
    
    # Obtaining the member '__getitem__' of a type (line 586)
    getitem___16084 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 586, 4), _check_select_call_result_16083, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 586)
    subscript_call_result_16085 = invoke(stypy.reporting.localization.Localization(__file__, 586, 4), getitem___16084, int_16072)
    
    # Assigning a type to the variable 'tuple_var_assignment_14090' (line 586)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 586, 4), 'tuple_var_assignment_14090', subscript_call_result_16085)
    
    # Assigning a Name to a Name (line 586):
    # Getting the type of 'tuple_var_assignment_14085' (line 586)
    tuple_var_assignment_14085_16086 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 586, 4), 'tuple_var_assignment_14085')
    # Assigning a type to the variable 'select' (line 586)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 586, 4), 'select', tuple_var_assignment_14085_16086)
    
    # Assigning a Name to a Name (line 586):
    # Getting the type of 'tuple_var_assignment_14086' (line 586)
    tuple_var_assignment_14086_16087 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 586, 4), 'tuple_var_assignment_14086')
    # Assigning a type to the variable 'vl' (line 586)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 586, 12), 'vl', tuple_var_assignment_14086_16087)
    
    # Assigning a Name to a Name (line 586):
    # Getting the type of 'tuple_var_assignment_14087' (line 586)
    tuple_var_assignment_14087_16088 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 586, 4), 'tuple_var_assignment_14087')
    # Assigning a type to the variable 'vu' (line 586)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 586, 16), 'vu', tuple_var_assignment_14087_16088)
    
    # Assigning a Name to a Name (line 586):
    # Getting the type of 'tuple_var_assignment_14088' (line 586)
    tuple_var_assignment_14088_16089 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 586, 4), 'tuple_var_assignment_14088')
    # Assigning a type to the variable 'il' (line 586)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 586, 20), 'il', tuple_var_assignment_14088_16089)
    
    # Assigning a Name to a Name (line 586):
    # Getting the type of 'tuple_var_assignment_14089' (line 586)
    tuple_var_assignment_14089_16090 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 586, 4), 'tuple_var_assignment_14089')
    # Assigning a type to the variable 'iu' (line 586)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 586, 24), 'iu', tuple_var_assignment_14089_16090)
    
    # Assigning a Name to a Name (line 586):
    # Getting the type of 'tuple_var_assignment_14090' (line 586)
    tuple_var_assignment_14090_16091 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 586, 4), 'tuple_var_assignment_14090')
    # Assigning a type to the variable 'max_ev' (line 586)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 586, 28), 'max_ev', tuple_var_assignment_14090_16091)
    # Deleting a member
    module_type_store.del_member(stypy.reporting.localization.Localization(__file__, 588, 4), module_type_store, 'select_range')
    
    
    # Getting the type of 'select' (line 589)
    select_16092 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 589, 7), 'select')
    int_16093 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 589, 17), 'int')
    # Applying the binary operator '==' (line 589)
    result_eq_16094 = python_operator(stypy.reporting.localization.Localization(__file__, 589, 7), '==', select_16092, int_16093)
    
    # Testing the type of an if condition (line 589)
    if_condition_16095 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 589, 4), result_eq_16094)
    # Assigning a type to the variable 'if_condition_16095' (line 589)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 589, 4), 'if_condition_16095', if_condition_16095)
    # SSA begins for if statement (line 589)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    
    # Getting the type of 'a1' (line 590)
    a1_16096 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 590, 11), 'a1')
    # Obtaining the member 'dtype' of a type (line 590)
    dtype_16097 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 590, 11), a1_16096, 'dtype')
    # Obtaining the member 'char' of a type (line 590)
    char_16098 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 590, 11), dtype_16097, 'char')
    str_16099 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 590, 28), 'str', 'GFD')
    # Applying the binary operator 'in' (line 590)
    result_contains_16100 = python_operator(stypy.reporting.localization.Localization(__file__, 590, 11), 'in', char_16098, str_16099)
    
    # Testing the type of an if condition (line 590)
    if_condition_16101 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 590, 8), result_contains_16100)
    # Assigning a type to the variable 'if_condition_16101' (line 590)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 590, 8), 'if_condition_16101', if_condition_16101)
    # SSA begins for if statement (line 590)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Str to a Name (line 595):
    
    # Assigning a Str to a Name (line 595):
    str_16102 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 595, 28), 'str', 'hbevd')
    # Assigning a type to the variable 'internal_name' (line 595)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 595, 12), 'internal_name', str_16102)
    # SSA branch for the else part of an if statement (line 590)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Str to a Name (line 600):
    
    # Assigning a Str to a Name (line 600):
    str_16103 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 600, 28), 'str', 'sbevd')
    # Assigning a type to the variable 'internal_name' (line 600)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 600, 12), 'internal_name', str_16103)
    # SSA join for if statement (line 590)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Tuple (line 601):
    
    # Assigning a Subscript to a Name (line 601):
    
    # Obtaining the type of the subscript
    int_16104 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 601, 8), 'int')
    
    # Call to get_lapack_funcs(...): (line 601)
    # Processing the call arguments (line 601)
    
    # Obtaining an instance of the builtin type 'tuple' (line 601)
    tuple_16106 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 601, 34), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 601)
    # Adding element type (line 601)
    # Getting the type of 'internal_name' (line 601)
    internal_name_16107 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 601, 34), 'internal_name', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 601, 34), tuple_16106, internal_name_16107)
    
    
    # Obtaining an instance of the builtin type 'tuple' (line 601)
    tuple_16108 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 601, 52), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 601)
    # Adding element type (line 601)
    # Getting the type of 'a1' (line 601)
    a1_16109 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 601, 52), 'a1', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 601, 52), tuple_16108, a1_16109)
    
    # Processing the call keyword arguments (line 601)
    kwargs_16110 = {}
    # Getting the type of 'get_lapack_funcs' (line 601)
    get_lapack_funcs_16105 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 601, 16), 'get_lapack_funcs', False)
    # Calling get_lapack_funcs(args, kwargs) (line 601)
    get_lapack_funcs_call_result_16111 = invoke(stypy.reporting.localization.Localization(__file__, 601, 16), get_lapack_funcs_16105, *[tuple_16106, tuple_16108], **kwargs_16110)
    
    # Obtaining the member '__getitem__' of a type (line 601)
    getitem___16112 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 601, 8), get_lapack_funcs_call_result_16111, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 601)
    subscript_call_result_16113 = invoke(stypy.reporting.localization.Localization(__file__, 601, 8), getitem___16112, int_16104)
    
    # Assigning a type to the variable 'tuple_var_assignment_14091' (line 601)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 601, 8), 'tuple_var_assignment_14091', subscript_call_result_16113)
    
    # Assigning a Name to a Name (line 601):
    # Getting the type of 'tuple_var_assignment_14091' (line 601)
    tuple_var_assignment_14091_16114 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 601, 8), 'tuple_var_assignment_14091')
    # Assigning a type to the variable 'bevd' (line 601)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 601, 8), 'bevd', tuple_var_assignment_14091_16114)
    
    # Assigning a Call to a Tuple (line 602):
    
    # Assigning a Subscript to a Name (line 602):
    
    # Obtaining the type of the subscript
    int_16115 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 602, 8), 'int')
    
    # Call to bevd(...): (line 602)
    # Processing the call arguments (line 602)
    # Getting the type of 'a1' (line 602)
    a1_16117 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 602, 26), 'a1', False)
    # Processing the call keyword arguments (line 602)
    
    # Getting the type of 'eigvals_only' (line 602)
    eigvals_only_16118 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 602, 44), 'eigvals_only', False)
    # Applying the 'not' unary operator (line 602)
    result_not__16119 = python_operator(stypy.reporting.localization.Localization(__file__, 602, 40), 'not', eigvals_only_16118)
    
    keyword_16120 = result_not__16119
    # Getting the type of 'lower' (line 603)
    lower_16121 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 603, 32), 'lower', False)
    keyword_16122 = lower_16121
    # Getting the type of 'overwrite_a_band' (line 603)
    overwrite_a_band_16123 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 603, 52), 'overwrite_a_band', False)
    keyword_16124 = overwrite_a_band_16123
    kwargs_16125 = {'lower': keyword_16122, 'overwrite_ab': keyword_16124, 'compute_v': keyword_16120}
    # Getting the type of 'bevd' (line 602)
    bevd_16116 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 602, 21), 'bevd', False)
    # Calling bevd(args, kwargs) (line 602)
    bevd_call_result_16126 = invoke(stypy.reporting.localization.Localization(__file__, 602, 21), bevd_16116, *[a1_16117], **kwargs_16125)
    
    # Obtaining the member '__getitem__' of a type (line 602)
    getitem___16127 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 602, 8), bevd_call_result_16126, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 602)
    subscript_call_result_16128 = invoke(stypy.reporting.localization.Localization(__file__, 602, 8), getitem___16127, int_16115)
    
    # Assigning a type to the variable 'tuple_var_assignment_14092' (line 602)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 602, 8), 'tuple_var_assignment_14092', subscript_call_result_16128)
    
    # Assigning a Subscript to a Name (line 602):
    
    # Obtaining the type of the subscript
    int_16129 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 602, 8), 'int')
    
    # Call to bevd(...): (line 602)
    # Processing the call arguments (line 602)
    # Getting the type of 'a1' (line 602)
    a1_16131 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 602, 26), 'a1', False)
    # Processing the call keyword arguments (line 602)
    
    # Getting the type of 'eigvals_only' (line 602)
    eigvals_only_16132 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 602, 44), 'eigvals_only', False)
    # Applying the 'not' unary operator (line 602)
    result_not__16133 = python_operator(stypy.reporting.localization.Localization(__file__, 602, 40), 'not', eigvals_only_16132)
    
    keyword_16134 = result_not__16133
    # Getting the type of 'lower' (line 603)
    lower_16135 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 603, 32), 'lower', False)
    keyword_16136 = lower_16135
    # Getting the type of 'overwrite_a_band' (line 603)
    overwrite_a_band_16137 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 603, 52), 'overwrite_a_band', False)
    keyword_16138 = overwrite_a_band_16137
    kwargs_16139 = {'lower': keyword_16136, 'overwrite_ab': keyword_16138, 'compute_v': keyword_16134}
    # Getting the type of 'bevd' (line 602)
    bevd_16130 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 602, 21), 'bevd', False)
    # Calling bevd(args, kwargs) (line 602)
    bevd_call_result_16140 = invoke(stypy.reporting.localization.Localization(__file__, 602, 21), bevd_16130, *[a1_16131], **kwargs_16139)
    
    # Obtaining the member '__getitem__' of a type (line 602)
    getitem___16141 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 602, 8), bevd_call_result_16140, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 602)
    subscript_call_result_16142 = invoke(stypy.reporting.localization.Localization(__file__, 602, 8), getitem___16141, int_16129)
    
    # Assigning a type to the variable 'tuple_var_assignment_14093' (line 602)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 602, 8), 'tuple_var_assignment_14093', subscript_call_result_16142)
    
    # Assigning a Subscript to a Name (line 602):
    
    # Obtaining the type of the subscript
    int_16143 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 602, 8), 'int')
    
    # Call to bevd(...): (line 602)
    # Processing the call arguments (line 602)
    # Getting the type of 'a1' (line 602)
    a1_16145 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 602, 26), 'a1', False)
    # Processing the call keyword arguments (line 602)
    
    # Getting the type of 'eigvals_only' (line 602)
    eigvals_only_16146 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 602, 44), 'eigvals_only', False)
    # Applying the 'not' unary operator (line 602)
    result_not__16147 = python_operator(stypy.reporting.localization.Localization(__file__, 602, 40), 'not', eigvals_only_16146)
    
    keyword_16148 = result_not__16147
    # Getting the type of 'lower' (line 603)
    lower_16149 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 603, 32), 'lower', False)
    keyword_16150 = lower_16149
    # Getting the type of 'overwrite_a_band' (line 603)
    overwrite_a_band_16151 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 603, 52), 'overwrite_a_band', False)
    keyword_16152 = overwrite_a_band_16151
    kwargs_16153 = {'lower': keyword_16150, 'overwrite_ab': keyword_16152, 'compute_v': keyword_16148}
    # Getting the type of 'bevd' (line 602)
    bevd_16144 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 602, 21), 'bevd', False)
    # Calling bevd(args, kwargs) (line 602)
    bevd_call_result_16154 = invoke(stypy.reporting.localization.Localization(__file__, 602, 21), bevd_16144, *[a1_16145], **kwargs_16153)
    
    # Obtaining the member '__getitem__' of a type (line 602)
    getitem___16155 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 602, 8), bevd_call_result_16154, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 602)
    subscript_call_result_16156 = invoke(stypy.reporting.localization.Localization(__file__, 602, 8), getitem___16155, int_16143)
    
    # Assigning a type to the variable 'tuple_var_assignment_14094' (line 602)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 602, 8), 'tuple_var_assignment_14094', subscript_call_result_16156)
    
    # Assigning a Name to a Name (line 602):
    # Getting the type of 'tuple_var_assignment_14092' (line 602)
    tuple_var_assignment_14092_16157 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 602, 8), 'tuple_var_assignment_14092')
    # Assigning a type to the variable 'w' (line 602)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 602, 8), 'w', tuple_var_assignment_14092_16157)
    
    # Assigning a Name to a Name (line 602):
    # Getting the type of 'tuple_var_assignment_14093' (line 602)
    tuple_var_assignment_14093_16158 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 602, 8), 'tuple_var_assignment_14093')
    # Assigning a type to the variable 'v' (line 602)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 602, 11), 'v', tuple_var_assignment_14093_16158)
    
    # Assigning a Name to a Name (line 602):
    # Getting the type of 'tuple_var_assignment_14094' (line 602)
    tuple_var_assignment_14094_16159 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 602, 8), 'tuple_var_assignment_14094')
    # Assigning a type to the variable 'info' (line 602)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 602, 14), 'info', tuple_var_assignment_14094_16159)
    # SSA branch for the else part of an if statement (line 589)
    module_type_store.open_ssa_branch('else')
    
    # Getting the type of 'eigvals_only' (line 605)
    eigvals_only_16160 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 605, 11), 'eigvals_only')
    # Testing the type of an if condition (line 605)
    if_condition_16161 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 605, 8), eigvals_only_16160)
    # Assigning a type to the variable 'if_condition_16161' (line 605)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 605, 8), 'if_condition_16161', if_condition_16161)
    # SSA begins for if statement (line 605)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Num to a Name (line 606):
    
    # Assigning a Num to a Name (line 606):
    int_16162 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 606, 21), 'int')
    # Assigning a type to the variable 'max_ev' (line 606)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 606, 12), 'max_ev', int_16162)
    # SSA join for if statement (line 605)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'a1' (line 608)
    a1_16163 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 608, 11), 'a1')
    # Obtaining the member 'dtype' of a type (line 608)
    dtype_16164 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 608, 11), a1_16163, 'dtype')
    # Obtaining the member 'char' of a type (line 608)
    char_16165 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 608, 11), dtype_16164, 'char')
    str_16166 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 608, 28), 'str', 'fF')
    # Applying the binary operator 'in' (line 608)
    result_contains_16167 = python_operator(stypy.reporting.localization.Localization(__file__, 608, 11), 'in', char_16165, str_16166)
    
    # Testing the type of an if condition (line 608)
    if_condition_16168 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 608, 8), result_contains_16167)
    # Assigning a type to the variable 'if_condition_16168' (line 608)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 608, 8), 'if_condition_16168', if_condition_16168)
    # SSA begins for if statement (line 608)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Tuple (line 609):
    
    # Assigning a Subscript to a Name (line 609):
    
    # Obtaining the type of the subscript
    int_16169 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 609, 12), 'int')
    
    # Call to get_lapack_funcs(...): (line 609)
    # Processing the call arguments (line 609)
    
    # Obtaining an instance of the builtin type 'tuple' (line 609)
    tuple_16171 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 609, 39), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 609)
    # Adding element type (line 609)
    str_16172 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 609, 39), 'str', 'lamch')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 609, 39), tuple_16171, str_16172)
    
    
    # Obtaining an instance of the builtin type 'tuple' (line 609)
    tuple_16173 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 609, 51), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 609)
    # Adding element type (line 609)
    
    # Call to array(...): (line 609)
    # Processing the call arguments (line 609)
    int_16175 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 609, 57), 'int')
    # Processing the call keyword arguments (line 609)
    str_16176 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 609, 66), 'str', 'f')
    keyword_16177 = str_16176
    kwargs_16178 = {'dtype': keyword_16177}
    # Getting the type of 'array' (line 609)
    array_16174 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 609, 51), 'array', False)
    # Calling array(args, kwargs) (line 609)
    array_call_result_16179 = invoke(stypy.reporting.localization.Localization(__file__, 609, 51), array_16174, *[int_16175], **kwargs_16178)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 609, 51), tuple_16173, array_call_result_16179)
    
    # Processing the call keyword arguments (line 609)
    kwargs_16180 = {}
    # Getting the type of 'get_lapack_funcs' (line 609)
    get_lapack_funcs_16170 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 609, 21), 'get_lapack_funcs', False)
    # Calling get_lapack_funcs(args, kwargs) (line 609)
    get_lapack_funcs_call_result_16181 = invoke(stypy.reporting.localization.Localization(__file__, 609, 21), get_lapack_funcs_16170, *[tuple_16171, tuple_16173], **kwargs_16180)
    
    # Obtaining the member '__getitem__' of a type (line 609)
    getitem___16182 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 609, 12), get_lapack_funcs_call_result_16181, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 609)
    subscript_call_result_16183 = invoke(stypy.reporting.localization.Localization(__file__, 609, 12), getitem___16182, int_16169)
    
    # Assigning a type to the variable 'tuple_var_assignment_14095' (line 609)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 609, 12), 'tuple_var_assignment_14095', subscript_call_result_16183)
    
    # Assigning a Name to a Name (line 609):
    # Getting the type of 'tuple_var_assignment_14095' (line 609)
    tuple_var_assignment_14095_16184 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 609, 12), 'tuple_var_assignment_14095')
    # Assigning a type to the variable 'lamch' (line 609)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 609, 12), 'lamch', tuple_var_assignment_14095_16184)
    # SSA branch for the else part of an if statement (line 608)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Call to a Tuple (line 611):
    
    # Assigning a Subscript to a Name (line 611):
    
    # Obtaining the type of the subscript
    int_16185 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 611, 12), 'int')
    
    # Call to get_lapack_funcs(...): (line 611)
    # Processing the call arguments (line 611)
    
    # Obtaining an instance of the builtin type 'tuple' (line 611)
    tuple_16187 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 611, 39), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 611)
    # Adding element type (line 611)
    str_16188 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 611, 39), 'str', 'lamch')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 611, 39), tuple_16187, str_16188)
    
    
    # Obtaining an instance of the builtin type 'tuple' (line 611)
    tuple_16189 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 611, 51), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 611)
    # Adding element type (line 611)
    
    # Call to array(...): (line 611)
    # Processing the call arguments (line 611)
    int_16191 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 611, 57), 'int')
    # Processing the call keyword arguments (line 611)
    str_16192 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 611, 66), 'str', 'd')
    keyword_16193 = str_16192
    kwargs_16194 = {'dtype': keyword_16193}
    # Getting the type of 'array' (line 611)
    array_16190 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 611, 51), 'array', False)
    # Calling array(args, kwargs) (line 611)
    array_call_result_16195 = invoke(stypy.reporting.localization.Localization(__file__, 611, 51), array_16190, *[int_16191], **kwargs_16194)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 611, 51), tuple_16189, array_call_result_16195)
    
    # Processing the call keyword arguments (line 611)
    kwargs_16196 = {}
    # Getting the type of 'get_lapack_funcs' (line 611)
    get_lapack_funcs_16186 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 611, 21), 'get_lapack_funcs', False)
    # Calling get_lapack_funcs(args, kwargs) (line 611)
    get_lapack_funcs_call_result_16197 = invoke(stypy.reporting.localization.Localization(__file__, 611, 21), get_lapack_funcs_16186, *[tuple_16187, tuple_16189], **kwargs_16196)
    
    # Obtaining the member '__getitem__' of a type (line 611)
    getitem___16198 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 611, 12), get_lapack_funcs_call_result_16197, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 611)
    subscript_call_result_16199 = invoke(stypy.reporting.localization.Localization(__file__, 611, 12), getitem___16198, int_16185)
    
    # Assigning a type to the variable 'tuple_var_assignment_14096' (line 611)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 611, 12), 'tuple_var_assignment_14096', subscript_call_result_16199)
    
    # Assigning a Name to a Name (line 611):
    # Getting the type of 'tuple_var_assignment_14096' (line 611)
    tuple_var_assignment_14096_16200 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 611, 12), 'tuple_var_assignment_14096')
    # Assigning a type to the variable 'lamch' (line 611)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 611, 12), 'lamch', tuple_var_assignment_14096_16200)
    # SSA join for if statement (line 608)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a BinOp to a Name (line 612):
    
    # Assigning a BinOp to a Name (line 612):
    int_16201 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 612, 17), 'int')
    
    # Call to lamch(...): (line 612)
    # Processing the call arguments (line 612)
    str_16203 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 612, 27), 'str', 's')
    # Processing the call keyword arguments (line 612)
    kwargs_16204 = {}
    # Getting the type of 'lamch' (line 612)
    lamch_16202 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 612, 21), 'lamch', False)
    # Calling lamch(args, kwargs) (line 612)
    lamch_call_result_16205 = invoke(stypy.reporting.localization.Localization(__file__, 612, 21), lamch_16202, *[str_16203], **kwargs_16204)
    
    # Applying the binary operator '*' (line 612)
    result_mul_16206 = python_operator(stypy.reporting.localization.Localization(__file__, 612, 17), '*', int_16201, lamch_call_result_16205)
    
    # Assigning a type to the variable 'abstol' (line 612)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 612, 8), 'abstol', result_mul_16206)
    
    
    # Getting the type of 'a1' (line 613)
    a1_16207 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 613, 11), 'a1')
    # Obtaining the member 'dtype' of a type (line 613)
    dtype_16208 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 613, 11), a1_16207, 'dtype')
    # Obtaining the member 'char' of a type (line 613)
    char_16209 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 613, 11), dtype_16208, 'char')
    str_16210 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 613, 28), 'str', 'GFD')
    # Applying the binary operator 'in' (line 613)
    result_contains_16211 = python_operator(stypy.reporting.localization.Localization(__file__, 613, 11), 'in', char_16209, str_16210)
    
    # Testing the type of an if condition (line 613)
    if_condition_16212 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 613, 8), result_contains_16211)
    # Assigning a type to the variable 'if_condition_16212' (line 613)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 613, 8), 'if_condition_16212', if_condition_16212)
    # SSA begins for if statement (line 613)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Str to a Name (line 614):
    
    # Assigning a Str to a Name (line 614):
    str_16213 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 614, 28), 'str', 'hbevx')
    # Assigning a type to the variable 'internal_name' (line 614)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 614, 12), 'internal_name', str_16213)
    # SSA branch for the else part of an if statement (line 613)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Str to a Name (line 616):
    
    # Assigning a Str to a Name (line 616):
    str_16214 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 616, 28), 'str', 'sbevx')
    # Assigning a type to the variable 'internal_name' (line 616)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 616, 12), 'internal_name', str_16214)
    # SSA join for if statement (line 613)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Tuple (line 617):
    
    # Assigning a Subscript to a Name (line 617):
    
    # Obtaining the type of the subscript
    int_16215 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 617, 8), 'int')
    
    # Call to get_lapack_funcs(...): (line 617)
    # Processing the call arguments (line 617)
    
    # Obtaining an instance of the builtin type 'tuple' (line 617)
    tuple_16217 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 617, 34), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 617)
    # Adding element type (line 617)
    # Getting the type of 'internal_name' (line 617)
    internal_name_16218 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 617, 34), 'internal_name', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 617, 34), tuple_16217, internal_name_16218)
    
    
    # Obtaining an instance of the builtin type 'tuple' (line 617)
    tuple_16219 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 617, 52), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 617)
    # Adding element type (line 617)
    # Getting the type of 'a1' (line 617)
    a1_16220 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 617, 52), 'a1', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 617, 52), tuple_16219, a1_16220)
    
    # Processing the call keyword arguments (line 617)
    kwargs_16221 = {}
    # Getting the type of 'get_lapack_funcs' (line 617)
    get_lapack_funcs_16216 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 617, 16), 'get_lapack_funcs', False)
    # Calling get_lapack_funcs(args, kwargs) (line 617)
    get_lapack_funcs_call_result_16222 = invoke(stypy.reporting.localization.Localization(__file__, 617, 16), get_lapack_funcs_16216, *[tuple_16217, tuple_16219], **kwargs_16221)
    
    # Obtaining the member '__getitem__' of a type (line 617)
    getitem___16223 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 617, 8), get_lapack_funcs_call_result_16222, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 617)
    subscript_call_result_16224 = invoke(stypy.reporting.localization.Localization(__file__, 617, 8), getitem___16223, int_16215)
    
    # Assigning a type to the variable 'tuple_var_assignment_14097' (line 617)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 617, 8), 'tuple_var_assignment_14097', subscript_call_result_16224)
    
    # Assigning a Name to a Name (line 617):
    # Getting the type of 'tuple_var_assignment_14097' (line 617)
    tuple_var_assignment_14097_16225 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 617, 8), 'tuple_var_assignment_14097')
    # Assigning a type to the variable 'bevx' (line 617)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 617, 8), 'bevx', tuple_var_assignment_14097_16225)
    
    # Assigning a Call to a Tuple (line 618):
    
    # Assigning a Subscript to a Name (line 618):
    
    # Obtaining the type of the subscript
    int_16226 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 618, 8), 'int')
    
    # Call to bevx(...): (line 618)
    # Processing the call arguments (line 618)
    # Getting the type of 'a1' (line 619)
    a1_16228 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 619, 12), 'a1', False)
    # Getting the type of 'vl' (line 619)
    vl_16229 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 619, 16), 'vl', False)
    # Getting the type of 'vu' (line 619)
    vu_16230 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 619, 20), 'vu', False)
    # Getting the type of 'il' (line 619)
    il_16231 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 619, 24), 'il', False)
    # Getting the type of 'iu' (line 619)
    iu_16232 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 619, 28), 'iu', False)
    # Processing the call keyword arguments (line 618)
    
    # Getting the type of 'eigvals_only' (line 619)
    eigvals_only_16233 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 619, 46), 'eigvals_only', False)
    # Applying the 'not' unary operator (line 619)
    result_not__16234 = python_operator(stypy.reporting.localization.Localization(__file__, 619, 42), 'not', eigvals_only_16233)
    
    keyword_16235 = result_not__16234
    # Getting the type of 'max_ev' (line 619)
    max_ev_16236 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 619, 65), 'max_ev', False)
    keyword_16237 = max_ev_16236
    # Getting the type of 'select' (line 620)
    select_16238 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 620, 18), 'select', False)
    keyword_16239 = select_16238
    # Getting the type of 'lower' (line 620)
    lower_16240 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 620, 32), 'lower', False)
    keyword_16241 = lower_16240
    # Getting the type of 'overwrite_a_band' (line 620)
    overwrite_a_band_16242 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 620, 52), 'overwrite_a_band', False)
    keyword_16243 = overwrite_a_band_16242
    # Getting the type of 'abstol' (line 621)
    abstol_16244 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 621, 19), 'abstol', False)
    keyword_16245 = abstol_16244
    kwargs_16246 = {'mmax': keyword_16237, 'lower': keyword_16241, 'overwrite_ab': keyword_16243, 'range': keyword_16239, 'abstol': keyword_16245, 'compute_v': keyword_16235}
    # Getting the type of 'bevx' (line 618)
    bevx_16227 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 618, 31), 'bevx', False)
    # Calling bevx(args, kwargs) (line 618)
    bevx_call_result_16247 = invoke(stypy.reporting.localization.Localization(__file__, 618, 31), bevx_16227, *[a1_16228, vl_16229, vu_16230, il_16231, iu_16232], **kwargs_16246)
    
    # Obtaining the member '__getitem__' of a type (line 618)
    getitem___16248 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 618, 8), bevx_call_result_16247, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 618)
    subscript_call_result_16249 = invoke(stypy.reporting.localization.Localization(__file__, 618, 8), getitem___16248, int_16226)
    
    # Assigning a type to the variable 'tuple_var_assignment_14098' (line 618)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 618, 8), 'tuple_var_assignment_14098', subscript_call_result_16249)
    
    # Assigning a Subscript to a Name (line 618):
    
    # Obtaining the type of the subscript
    int_16250 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 618, 8), 'int')
    
    # Call to bevx(...): (line 618)
    # Processing the call arguments (line 618)
    # Getting the type of 'a1' (line 619)
    a1_16252 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 619, 12), 'a1', False)
    # Getting the type of 'vl' (line 619)
    vl_16253 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 619, 16), 'vl', False)
    # Getting the type of 'vu' (line 619)
    vu_16254 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 619, 20), 'vu', False)
    # Getting the type of 'il' (line 619)
    il_16255 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 619, 24), 'il', False)
    # Getting the type of 'iu' (line 619)
    iu_16256 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 619, 28), 'iu', False)
    # Processing the call keyword arguments (line 618)
    
    # Getting the type of 'eigvals_only' (line 619)
    eigvals_only_16257 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 619, 46), 'eigvals_only', False)
    # Applying the 'not' unary operator (line 619)
    result_not__16258 = python_operator(stypy.reporting.localization.Localization(__file__, 619, 42), 'not', eigvals_only_16257)
    
    keyword_16259 = result_not__16258
    # Getting the type of 'max_ev' (line 619)
    max_ev_16260 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 619, 65), 'max_ev', False)
    keyword_16261 = max_ev_16260
    # Getting the type of 'select' (line 620)
    select_16262 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 620, 18), 'select', False)
    keyword_16263 = select_16262
    # Getting the type of 'lower' (line 620)
    lower_16264 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 620, 32), 'lower', False)
    keyword_16265 = lower_16264
    # Getting the type of 'overwrite_a_band' (line 620)
    overwrite_a_band_16266 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 620, 52), 'overwrite_a_band', False)
    keyword_16267 = overwrite_a_band_16266
    # Getting the type of 'abstol' (line 621)
    abstol_16268 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 621, 19), 'abstol', False)
    keyword_16269 = abstol_16268
    kwargs_16270 = {'mmax': keyword_16261, 'lower': keyword_16265, 'overwrite_ab': keyword_16267, 'range': keyword_16263, 'abstol': keyword_16269, 'compute_v': keyword_16259}
    # Getting the type of 'bevx' (line 618)
    bevx_16251 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 618, 31), 'bevx', False)
    # Calling bevx(args, kwargs) (line 618)
    bevx_call_result_16271 = invoke(stypy.reporting.localization.Localization(__file__, 618, 31), bevx_16251, *[a1_16252, vl_16253, vu_16254, il_16255, iu_16256], **kwargs_16270)
    
    # Obtaining the member '__getitem__' of a type (line 618)
    getitem___16272 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 618, 8), bevx_call_result_16271, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 618)
    subscript_call_result_16273 = invoke(stypy.reporting.localization.Localization(__file__, 618, 8), getitem___16272, int_16250)
    
    # Assigning a type to the variable 'tuple_var_assignment_14099' (line 618)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 618, 8), 'tuple_var_assignment_14099', subscript_call_result_16273)
    
    # Assigning a Subscript to a Name (line 618):
    
    # Obtaining the type of the subscript
    int_16274 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 618, 8), 'int')
    
    # Call to bevx(...): (line 618)
    # Processing the call arguments (line 618)
    # Getting the type of 'a1' (line 619)
    a1_16276 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 619, 12), 'a1', False)
    # Getting the type of 'vl' (line 619)
    vl_16277 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 619, 16), 'vl', False)
    # Getting the type of 'vu' (line 619)
    vu_16278 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 619, 20), 'vu', False)
    # Getting the type of 'il' (line 619)
    il_16279 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 619, 24), 'il', False)
    # Getting the type of 'iu' (line 619)
    iu_16280 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 619, 28), 'iu', False)
    # Processing the call keyword arguments (line 618)
    
    # Getting the type of 'eigvals_only' (line 619)
    eigvals_only_16281 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 619, 46), 'eigvals_only', False)
    # Applying the 'not' unary operator (line 619)
    result_not__16282 = python_operator(stypy.reporting.localization.Localization(__file__, 619, 42), 'not', eigvals_only_16281)
    
    keyword_16283 = result_not__16282
    # Getting the type of 'max_ev' (line 619)
    max_ev_16284 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 619, 65), 'max_ev', False)
    keyword_16285 = max_ev_16284
    # Getting the type of 'select' (line 620)
    select_16286 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 620, 18), 'select', False)
    keyword_16287 = select_16286
    # Getting the type of 'lower' (line 620)
    lower_16288 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 620, 32), 'lower', False)
    keyword_16289 = lower_16288
    # Getting the type of 'overwrite_a_band' (line 620)
    overwrite_a_band_16290 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 620, 52), 'overwrite_a_band', False)
    keyword_16291 = overwrite_a_band_16290
    # Getting the type of 'abstol' (line 621)
    abstol_16292 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 621, 19), 'abstol', False)
    keyword_16293 = abstol_16292
    kwargs_16294 = {'mmax': keyword_16285, 'lower': keyword_16289, 'overwrite_ab': keyword_16291, 'range': keyword_16287, 'abstol': keyword_16293, 'compute_v': keyword_16283}
    # Getting the type of 'bevx' (line 618)
    bevx_16275 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 618, 31), 'bevx', False)
    # Calling bevx(args, kwargs) (line 618)
    bevx_call_result_16295 = invoke(stypy.reporting.localization.Localization(__file__, 618, 31), bevx_16275, *[a1_16276, vl_16277, vu_16278, il_16279, iu_16280], **kwargs_16294)
    
    # Obtaining the member '__getitem__' of a type (line 618)
    getitem___16296 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 618, 8), bevx_call_result_16295, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 618)
    subscript_call_result_16297 = invoke(stypy.reporting.localization.Localization(__file__, 618, 8), getitem___16296, int_16274)
    
    # Assigning a type to the variable 'tuple_var_assignment_14100' (line 618)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 618, 8), 'tuple_var_assignment_14100', subscript_call_result_16297)
    
    # Assigning a Subscript to a Name (line 618):
    
    # Obtaining the type of the subscript
    int_16298 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 618, 8), 'int')
    
    # Call to bevx(...): (line 618)
    # Processing the call arguments (line 618)
    # Getting the type of 'a1' (line 619)
    a1_16300 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 619, 12), 'a1', False)
    # Getting the type of 'vl' (line 619)
    vl_16301 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 619, 16), 'vl', False)
    # Getting the type of 'vu' (line 619)
    vu_16302 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 619, 20), 'vu', False)
    # Getting the type of 'il' (line 619)
    il_16303 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 619, 24), 'il', False)
    # Getting the type of 'iu' (line 619)
    iu_16304 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 619, 28), 'iu', False)
    # Processing the call keyword arguments (line 618)
    
    # Getting the type of 'eigvals_only' (line 619)
    eigvals_only_16305 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 619, 46), 'eigvals_only', False)
    # Applying the 'not' unary operator (line 619)
    result_not__16306 = python_operator(stypy.reporting.localization.Localization(__file__, 619, 42), 'not', eigvals_only_16305)
    
    keyword_16307 = result_not__16306
    # Getting the type of 'max_ev' (line 619)
    max_ev_16308 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 619, 65), 'max_ev', False)
    keyword_16309 = max_ev_16308
    # Getting the type of 'select' (line 620)
    select_16310 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 620, 18), 'select', False)
    keyword_16311 = select_16310
    # Getting the type of 'lower' (line 620)
    lower_16312 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 620, 32), 'lower', False)
    keyword_16313 = lower_16312
    # Getting the type of 'overwrite_a_band' (line 620)
    overwrite_a_band_16314 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 620, 52), 'overwrite_a_band', False)
    keyword_16315 = overwrite_a_band_16314
    # Getting the type of 'abstol' (line 621)
    abstol_16316 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 621, 19), 'abstol', False)
    keyword_16317 = abstol_16316
    kwargs_16318 = {'mmax': keyword_16309, 'lower': keyword_16313, 'overwrite_ab': keyword_16315, 'range': keyword_16311, 'abstol': keyword_16317, 'compute_v': keyword_16307}
    # Getting the type of 'bevx' (line 618)
    bevx_16299 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 618, 31), 'bevx', False)
    # Calling bevx(args, kwargs) (line 618)
    bevx_call_result_16319 = invoke(stypy.reporting.localization.Localization(__file__, 618, 31), bevx_16299, *[a1_16300, vl_16301, vu_16302, il_16303, iu_16304], **kwargs_16318)
    
    # Obtaining the member '__getitem__' of a type (line 618)
    getitem___16320 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 618, 8), bevx_call_result_16319, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 618)
    subscript_call_result_16321 = invoke(stypy.reporting.localization.Localization(__file__, 618, 8), getitem___16320, int_16298)
    
    # Assigning a type to the variable 'tuple_var_assignment_14101' (line 618)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 618, 8), 'tuple_var_assignment_14101', subscript_call_result_16321)
    
    # Assigning a Subscript to a Name (line 618):
    
    # Obtaining the type of the subscript
    int_16322 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 618, 8), 'int')
    
    # Call to bevx(...): (line 618)
    # Processing the call arguments (line 618)
    # Getting the type of 'a1' (line 619)
    a1_16324 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 619, 12), 'a1', False)
    # Getting the type of 'vl' (line 619)
    vl_16325 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 619, 16), 'vl', False)
    # Getting the type of 'vu' (line 619)
    vu_16326 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 619, 20), 'vu', False)
    # Getting the type of 'il' (line 619)
    il_16327 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 619, 24), 'il', False)
    # Getting the type of 'iu' (line 619)
    iu_16328 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 619, 28), 'iu', False)
    # Processing the call keyword arguments (line 618)
    
    # Getting the type of 'eigvals_only' (line 619)
    eigvals_only_16329 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 619, 46), 'eigvals_only', False)
    # Applying the 'not' unary operator (line 619)
    result_not__16330 = python_operator(stypy.reporting.localization.Localization(__file__, 619, 42), 'not', eigvals_only_16329)
    
    keyword_16331 = result_not__16330
    # Getting the type of 'max_ev' (line 619)
    max_ev_16332 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 619, 65), 'max_ev', False)
    keyword_16333 = max_ev_16332
    # Getting the type of 'select' (line 620)
    select_16334 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 620, 18), 'select', False)
    keyword_16335 = select_16334
    # Getting the type of 'lower' (line 620)
    lower_16336 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 620, 32), 'lower', False)
    keyword_16337 = lower_16336
    # Getting the type of 'overwrite_a_band' (line 620)
    overwrite_a_band_16338 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 620, 52), 'overwrite_a_band', False)
    keyword_16339 = overwrite_a_band_16338
    # Getting the type of 'abstol' (line 621)
    abstol_16340 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 621, 19), 'abstol', False)
    keyword_16341 = abstol_16340
    kwargs_16342 = {'mmax': keyword_16333, 'lower': keyword_16337, 'overwrite_ab': keyword_16339, 'range': keyword_16335, 'abstol': keyword_16341, 'compute_v': keyword_16331}
    # Getting the type of 'bevx' (line 618)
    bevx_16323 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 618, 31), 'bevx', False)
    # Calling bevx(args, kwargs) (line 618)
    bevx_call_result_16343 = invoke(stypy.reporting.localization.Localization(__file__, 618, 31), bevx_16323, *[a1_16324, vl_16325, vu_16326, il_16327, iu_16328], **kwargs_16342)
    
    # Obtaining the member '__getitem__' of a type (line 618)
    getitem___16344 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 618, 8), bevx_call_result_16343, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 618)
    subscript_call_result_16345 = invoke(stypy.reporting.localization.Localization(__file__, 618, 8), getitem___16344, int_16322)
    
    # Assigning a type to the variable 'tuple_var_assignment_14102' (line 618)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 618, 8), 'tuple_var_assignment_14102', subscript_call_result_16345)
    
    # Assigning a Name to a Name (line 618):
    # Getting the type of 'tuple_var_assignment_14098' (line 618)
    tuple_var_assignment_14098_16346 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 618, 8), 'tuple_var_assignment_14098')
    # Assigning a type to the variable 'w' (line 618)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 618, 8), 'w', tuple_var_assignment_14098_16346)
    
    # Assigning a Name to a Name (line 618):
    # Getting the type of 'tuple_var_assignment_14099' (line 618)
    tuple_var_assignment_14099_16347 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 618, 8), 'tuple_var_assignment_14099')
    # Assigning a type to the variable 'v' (line 618)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 618, 11), 'v', tuple_var_assignment_14099_16347)
    
    # Assigning a Name to a Name (line 618):
    # Getting the type of 'tuple_var_assignment_14100' (line 618)
    tuple_var_assignment_14100_16348 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 618, 8), 'tuple_var_assignment_14100')
    # Assigning a type to the variable 'm' (line 618)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 618, 14), 'm', tuple_var_assignment_14100_16348)
    
    # Assigning a Name to a Name (line 618):
    # Getting the type of 'tuple_var_assignment_14101' (line 618)
    tuple_var_assignment_14101_16349 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 618, 8), 'tuple_var_assignment_14101')
    # Assigning a type to the variable 'ifail' (line 618)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 618, 17), 'ifail', tuple_var_assignment_14101_16349)
    
    # Assigning a Name to a Name (line 618):
    # Getting the type of 'tuple_var_assignment_14102' (line 618)
    tuple_var_assignment_14102_16350 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 618, 8), 'tuple_var_assignment_14102')
    # Assigning a type to the variable 'info' (line 618)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 618, 24), 'info', tuple_var_assignment_14102_16350)
    
    # Assigning a Subscript to a Name (line 623):
    
    # Assigning a Subscript to a Name (line 623):
    
    # Obtaining the type of the subscript
    # Getting the type of 'm' (line 623)
    m_16351 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 623, 15), 'm')
    slice_16352 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 623, 12), None, m_16351, None)
    # Getting the type of 'w' (line 623)
    w_16353 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 623, 12), 'w')
    # Obtaining the member '__getitem__' of a type (line 623)
    getitem___16354 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 623, 12), w_16353, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 623)
    subscript_call_result_16355 = invoke(stypy.reporting.localization.Localization(__file__, 623, 12), getitem___16354, slice_16352)
    
    # Assigning a type to the variable 'w' (line 623)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 623, 8), 'w', subscript_call_result_16355)
    
    
    # Getting the type of 'eigvals_only' (line 624)
    eigvals_only_16356 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 624, 15), 'eigvals_only')
    # Applying the 'not' unary operator (line 624)
    result_not__16357 = python_operator(stypy.reporting.localization.Localization(__file__, 624, 11), 'not', eigvals_only_16356)
    
    # Testing the type of an if condition (line 624)
    if_condition_16358 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 624, 8), result_not__16357)
    # Assigning a type to the variable 'if_condition_16358' (line 624)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 624, 8), 'if_condition_16358', if_condition_16358)
    # SSA begins for if statement (line 624)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Subscript to a Name (line 625):
    
    # Assigning a Subscript to a Name (line 625):
    
    # Obtaining the type of the subscript
    slice_16359 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 625, 16), None, None, None)
    # Getting the type of 'm' (line 625)
    m_16360 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 625, 22), 'm')
    slice_16361 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 625, 16), None, m_16360, None)
    # Getting the type of 'v' (line 625)
    v_16362 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 625, 16), 'v')
    # Obtaining the member '__getitem__' of a type (line 625)
    getitem___16363 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 625, 16), v_16362, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 625)
    subscript_call_result_16364 = invoke(stypy.reporting.localization.Localization(__file__, 625, 16), getitem___16363, (slice_16359, slice_16361))
    
    # Assigning a type to the variable 'v' (line 625)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 625, 12), 'v', subscript_call_result_16364)
    # SSA join for if statement (line 624)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 589)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to _check_info(...): (line 626)
    # Processing the call arguments (line 626)
    # Getting the type of 'info' (line 626)
    info_16366 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 626, 16), 'info', False)
    # Getting the type of 'internal_name' (line 626)
    internal_name_16367 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 626, 22), 'internal_name', False)
    # Processing the call keyword arguments (line 626)
    kwargs_16368 = {}
    # Getting the type of '_check_info' (line 626)
    _check_info_16365 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 626, 4), '_check_info', False)
    # Calling _check_info(args, kwargs) (line 626)
    _check_info_call_result_16369 = invoke(stypy.reporting.localization.Localization(__file__, 626, 4), _check_info_16365, *[info_16366, internal_name_16367], **kwargs_16368)
    
    
    # Getting the type of 'eigvals_only' (line 628)
    eigvals_only_16370 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 628, 7), 'eigvals_only')
    # Testing the type of an if condition (line 628)
    if_condition_16371 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 628, 4), eigvals_only_16370)
    # Assigning a type to the variable 'if_condition_16371' (line 628)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 628, 4), 'if_condition_16371', if_condition_16371)
    # SSA begins for if statement (line 628)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # Getting the type of 'w' (line 629)
    w_16372 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 629, 15), 'w')
    # Assigning a type to the variable 'stypy_return_type' (line 629)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 629, 8), 'stypy_return_type', w_16372)
    # SSA join for if statement (line 628)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Obtaining an instance of the builtin type 'tuple' (line 630)
    tuple_16373 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 630, 11), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 630)
    # Adding element type (line 630)
    # Getting the type of 'w' (line 630)
    w_16374 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 630, 11), 'w')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 630, 11), tuple_16373, w_16374)
    # Adding element type (line 630)
    # Getting the type of 'v' (line 630)
    v_16375 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 630, 14), 'v')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 630, 11), tuple_16373, v_16375)
    
    # Assigning a type to the variable 'stypy_return_type' (line 630)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 630, 4), 'stypy_return_type', tuple_16373)
    
    # ################# End of 'eig_banded(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'eig_banded' in the type store
    # Getting the type of 'stypy_return_type' (line 487)
    stypy_return_type_16376 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 487, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_16376)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'eig_banded'
    return stypy_return_type_16376

# Assigning a type to the variable 'eig_banded' (line 487)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 487, 0), 'eig_banded', eig_banded)

@norecursion
def eigvals(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'None' (line 633)
    None_16377 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 633, 17), 'None')
    # Getting the type of 'False' (line 633)
    False_16378 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 633, 35), 'False')
    # Getting the type of 'True' (line 633)
    True_16379 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 633, 55), 'True')
    # Getting the type of 'False' (line 634)
    False_16380 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 634, 32), 'False')
    defaults = [None_16377, False_16378, True_16379, False_16380]
    # Create a new context for function 'eigvals'
    module_type_store = module_type_store.open_function_context('eigvals', 633, 0, False)
    
    # Passed parameters checking function
    eigvals.stypy_localization = localization
    eigvals.stypy_type_of_self = None
    eigvals.stypy_type_store = module_type_store
    eigvals.stypy_function_name = 'eigvals'
    eigvals.stypy_param_names_list = ['a', 'b', 'overwrite_a', 'check_finite', 'homogeneous_eigvals']
    eigvals.stypy_varargs_param_name = None
    eigvals.stypy_kwargs_param_name = None
    eigvals.stypy_call_defaults = defaults
    eigvals.stypy_call_varargs = varargs
    eigvals.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'eigvals', ['a', 'b', 'overwrite_a', 'check_finite', 'homogeneous_eigvals'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'eigvals', localization, ['a', 'b', 'overwrite_a', 'check_finite', 'homogeneous_eigvals'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'eigvals(...)' code ##################

    str_16381 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 684, (-1)), 'str', '\n    Compute eigenvalues from an ordinary or generalized eigenvalue problem.\n\n    Find eigenvalues of a general matrix::\n\n        a   vr[:,i] = w[i]        b   vr[:,i]\n\n    Parameters\n    ----------\n    a : (M, M) array_like\n        A complex or real matrix whose eigenvalues and eigenvectors\n        will be computed.\n    b : (M, M) array_like, optional\n        Right-hand side matrix in a generalized eigenvalue problem.\n        If omitted, identity matrix is assumed.\n    overwrite_a : bool, optional\n        Whether to overwrite data in a (may improve performance)\n    check_finite : bool, optional\n        Whether to check that the input matrices contain only finite numbers.\n        Disabling may give a performance gain, but may result in problems\n        (crashes, non-termination) if the inputs do contain infinities\n        or NaNs.\n    homogeneous_eigvals : bool, optional\n        If True, return the eigenvalues in homogeneous coordinates.\n        In this case ``w`` is a (2, M) array so that::\n\n            w[1,i] a vr[:,i] = w[0,i] b vr[:,i]\n\n        Default is False.\n\n    Returns\n    -------\n    w : (M,) or (2, M) double or complex ndarray\n        The eigenvalues, each repeated according to its multiplicity\n        but not in any specific order. The shape is (M,) unless\n        ``homogeneous_eigvals=True``.\n\n    Raises\n    ------\n    LinAlgError\n        If eigenvalue computation does not converge\n\n    See Also\n    --------\n    eig : eigenvalues and right eigenvectors of general arrays.\n    eigvalsh : eigenvalues of symmetric or Hermitian arrays\n    eigvals_banded : eigenvalues for symmetric/Hermitian band matrices\n    eigvalsh_tridiagonal : eigenvalues of symmetric/Hermitian tridiagonal\n        matrices\n    ')
    
    # Call to eig(...): (line 685)
    # Processing the call arguments (line 685)
    # Getting the type of 'a' (line 685)
    a_16383 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 685, 15), 'a', False)
    # Processing the call keyword arguments (line 685)
    # Getting the type of 'b' (line 685)
    b_16384 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 685, 20), 'b', False)
    keyword_16385 = b_16384
    int_16386 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 685, 28), 'int')
    keyword_16387 = int_16386
    int_16388 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 685, 37), 'int')
    keyword_16389 = int_16388
    # Getting the type of 'overwrite_a' (line 685)
    overwrite_a_16390 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 685, 52), 'overwrite_a', False)
    keyword_16391 = overwrite_a_16390
    # Getting the type of 'check_finite' (line 686)
    check_finite_16392 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 686, 28), 'check_finite', False)
    keyword_16393 = check_finite_16392
    # Getting the type of 'homogeneous_eigvals' (line 687)
    homogeneous_eigvals_16394 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 687, 35), 'homogeneous_eigvals', False)
    keyword_16395 = homogeneous_eigvals_16394
    kwargs_16396 = {'b': keyword_16385, 'overwrite_a': keyword_16391, 'homogeneous_eigvals': keyword_16395, 'right': keyword_16389, 'check_finite': keyword_16393, 'left': keyword_16387}
    # Getting the type of 'eig' (line 685)
    eig_16382 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 685, 11), 'eig', False)
    # Calling eig(args, kwargs) (line 685)
    eig_call_result_16397 = invoke(stypy.reporting.localization.Localization(__file__, 685, 11), eig_16382, *[a_16383], **kwargs_16396)
    
    # Assigning a type to the variable 'stypy_return_type' (line 685)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 685, 4), 'stypy_return_type', eig_call_result_16397)
    
    # ################# End of 'eigvals(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'eigvals' in the type store
    # Getting the type of 'stypy_return_type' (line 633)
    stypy_return_type_16398 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 633, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_16398)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'eigvals'
    return stypy_return_type_16398

# Assigning a type to the variable 'eigvals' (line 633)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 633, 0), 'eigvals', eigvals)

@norecursion
def eigvalsh(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'None' (line 690)
    None_16399 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 690, 18), 'None')
    # Getting the type of 'True' (line 690)
    True_16400 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 690, 30), 'True')
    # Getting the type of 'False' (line 690)
    False_16401 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 690, 48), 'False')
    # Getting the type of 'False' (line 691)
    False_16402 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 691, 25), 'False')
    # Getting the type of 'True' (line 691)
    True_16403 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 691, 38), 'True')
    # Getting the type of 'None' (line 691)
    None_16404 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 691, 52), 'None')
    int_16405 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 691, 63), 'int')
    # Getting the type of 'True' (line 692)
    True_16406 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 692, 26), 'True')
    defaults = [None_16399, True_16400, False_16401, False_16402, True_16403, None_16404, int_16405, True_16406]
    # Create a new context for function 'eigvalsh'
    module_type_store = module_type_store.open_function_context('eigvalsh', 690, 0, False)
    
    # Passed parameters checking function
    eigvalsh.stypy_localization = localization
    eigvalsh.stypy_type_of_self = None
    eigvalsh.stypy_type_store = module_type_store
    eigvalsh.stypy_function_name = 'eigvalsh'
    eigvalsh.stypy_param_names_list = ['a', 'b', 'lower', 'overwrite_a', 'overwrite_b', 'turbo', 'eigvals', 'type', 'check_finite']
    eigvalsh.stypy_varargs_param_name = None
    eigvalsh.stypy_kwargs_param_name = None
    eigvalsh.stypy_call_defaults = defaults
    eigvalsh.stypy_call_varargs = varargs
    eigvalsh.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'eigvalsh', ['a', 'b', 'lower', 'overwrite_a', 'overwrite_b', 'turbo', 'eigvals', 'type', 'check_finite'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'eigvalsh', localization, ['a', 'b', 'lower', 'overwrite_a', 'overwrite_b', 'turbo', 'eigvals', 'type', 'check_finite'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'eigvalsh(...)' code ##################

    str_16407 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 760, (-1)), 'str', '\n    Solve an ordinary or generalized eigenvalue problem for a complex\n    Hermitian or real symmetric matrix.\n\n    Find eigenvalues w of matrix a, where b is positive definite::\n\n                      a v[:,i] = w[i] b v[:,i]\n        v[i,:].conj() a v[:,i] = w[i]\n        v[i,:].conj() b v[:,i] = 1\n\n\n    Parameters\n    ----------\n    a : (M, M) array_like\n        A complex Hermitian or real symmetric matrix whose eigenvalues and\n        eigenvectors will be computed.\n    b : (M, M) array_like, optional\n        A complex Hermitian or real symmetric definite positive matrix in.\n        If omitted, identity matrix is assumed.\n    lower : bool, optional\n        Whether the pertinent array data is taken from the lower or upper\n        triangle of `a`. (Default: lower)\n    turbo : bool, optional\n        Use divide and conquer algorithm (faster but expensive in memory,\n        only for generalized eigenvalue problem and if eigvals=None)\n    eigvals : tuple (lo, hi), optional\n        Indexes of the smallest and largest (in ascending order) eigenvalues\n        and corresponding eigenvectors to be returned: 0 <= lo < hi <= M-1.\n        If omitted, all eigenvalues and eigenvectors are returned.\n    type : int, optional\n        Specifies the problem type to be solved:\n\n           type = 1: a   v[:,i] = w[i] b v[:,i]\n\n           type = 2: a b v[:,i] = w[i]   v[:,i]\n\n           type = 3: b a v[:,i] = w[i]   v[:,i]\n    overwrite_a : bool, optional\n        Whether to overwrite data in `a` (may improve performance)\n    overwrite_b : bool, optional\n        Whether to overwrite data in `b` (may improve performance)\n    check_finite : bool, optional\n        Whether to check that the input matrices contain only finite numbers.\n        Disabling may give a performance gain, but may result in problems\n        (crashes, non-termination) if the inputs do contain infinities or NaNs.\n\n    Returns\n    -------\n    w : (N,) float ndarray\n        The N (1<=N<=M) selected eigenvalues, in ascending order, each\n        repeated according to its multiplicity.\n\n    Raises\n    ------\n    LinAlgError\n        If eigenvalue computation does not converge,\n        an error occurred, or b matrix is not definite positive. Note that\n        if input matrices are not symmetric or hermitian, no error is reported\n        but results will be wrong.\n\n    See Also\n    --------\n    eigh : eigenvalues and right eigenvectors for symmetric/Hermitian arrays\n    eigvals : eigenvalues of general arrays\n    eigvals_banded : eigenvalues for symmetric/Hermitian band matrices\n    eigvalsh_tridiagonal : eigenvalues of symmetric/Hermitian tridiagonal\n        matrices\n    ')
    
    # Call to eigh(...): (line 761)
    # Processing the call arguments (line 761)
    # Getting the type of 'a' (line 761)
    a_16409 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 761, 16), 'a', False)
    # Processing the call keyword arguments (line 761)
    # Getting the type of 'b' (line 761)
    b_16410 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 761, 21), 'b', False)
    keyword_16411 = b_16410
    # Getting the type of 'lower' (line 761)
    lower_16412 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 761, 30), 'lower', False)
    keyword_16413 = lower_16412
    # Getting the type of 'True' (line 761)
    True_16414 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 761, 50), 'True', False)
    keyword_16415 = True_16414
    # Getting the type of 'overwrite_a' (line 762)
    overwrite_a_16416 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 762, 28), 'overwrite_a', False)
    keyword_16417 = overwrite_a_16416
    # Getting the type of 'overwrite_b' (line 762)
    overwrite_b_16418 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 762, 53), 'overwrite_b', False)
    keyword_16419 = overwrite_b_16418
    # Getting the type of 'turbo' (line 763)
    turbo_16420 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 763, 22), 'turbo', False)
    keyword_16421 = turbo_16420
    # Getting the type of 'eigvals' (line 763)
    eigvals_16422 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 763, 37), 'eigvals', False)
    keyword_16423 = eigvals_16422
    # Getting the type of 'type' (line 763)
    type_16424 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 763, 51), 'type', False)
    keyword_16425 = type_16424
    # Getting the type of 'check_finite' (line 764)
    check_finite_16426 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 764, 29), 'check_finite', False)
    keyword_16427 = check_finite_16426
    kwargs_16428 = {'turbo': keyword_16421, 'lower': keyword_16413, 'b': keyword_16411, 'overwrite_a': keyword_16417, 'overwrite_b': keyword_16419, 'eigvals': keyword_16423, 'eigvals_only': keyword_16415, 'type': keyword_16425, 'check_finite': keyword_16427}
    # Getting the type of 'eigh' (line 761)
    eigh_16408 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 761, 11), 'eigh', False)
    # Calling eigh(args, kwargs) (line 761)
    eigh_call_result_16429 = invoke(stypy.reporting.localization.Localization(__file__, 761, 11), eigh_16408, *[a_16409], **kwargs_16428)
    
    # Assigning a type to the variable 'stypy_return_type' (line 761)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 761, 4), 'stypy_return_type', eigh_call_result_16429)
    
    # ################# End of 'eigvalsh(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'eigvalsh' in the type store
    # Getting the type of 'stypy_return_type' (line 690)
    stypy_return_type_16430 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 690, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_16430)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'eigvalsh'
    return stypy_return_type_16430

# Assigning a type to the variable 'eigvalsh' (line 690)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 690, 0), 'eigvalsh', eigvalsh)

@norecursion
def eigvals_banded(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'False' (line 767)
    False_16431 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 767, 33), 'False')
    # Getting the type of 'False' (line 767)
    False_16432 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 767, 57), 'False')
    str_16433 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 768, 26), 'str', 'a')
    # Getting the type of 'None' (line 768)
    None_16434 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 768, 44), 'None')
    # Getting the type of 'True' (line 768)
    True_16435 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 768, 63), 'True')
    defaults = [False_16431, False_16432, str_16433, None_16434, True_16435]
    # Create a new context for function 'eigvals_banded'
    module_type_store = module_type_store.open_function_context('eigvals_banded', 767, 0, False)
    
    # Passed parameters checking function
    eigvals_banded.stypy_localization = localization
    eigvals_banded.stypy_type_of_self = None
    eigvals_banded.stypy_type_store = module_type_store
    eigvals_banded.stypy_function_name = 'eigvals_banded'
    eigvals_banded.stypy_param_names_list = ['a_band', 'lower', 'overwrite_a_band', 'select', 'select_range', 'check_finite']
    eigvals_banded.stypy_varargs_param_name = None
    eigvals_banded.stypy_kwargs_param_name = None
    eigvals_banded.stypy_call_defaults = defaults
    eigvals_banded.stypy_call_varargs = varargs
    eigvals_banded.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'eigvals_banded', ['a_band', 'lower', 'overwrite_a_band', 'select', 'select_range', 'check_finite'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'eigvals_banded', localization, ['a_band', 'lower', 'overwrite_a_band', 'select', 'select_range', 'check_finite'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'eigvals_banded(...)' code ##################

    str_16436 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 844, (-1)), 'str', "\n    Solve real symmetric or complex hermitian band matrix eigenvalue problem.\n\n    Find eigenvalues w of a::\n\n        a v[:,i] = w[i] v[:,i]\n        v.H v    = identity\n\n    The matrix a is stored in a_band either in lower diagonal or upper\n    diagonal ordered form:\n\n        a_band[u + i - j, j] == a[i,j]        (if upper form; i <= j)\n        a_band[    i - j, j] == a[i,j]        (if lower form; i >= j)\n\n    where u is the number of bands above the diagonal.\n\n    Example of a_band (shape of a is (6,6), u=2)::\n\n        upper form:\n        *   *   a02 a13 a24 a35\n        *   a01 a12 a23 a34 a45\n        a00 a11 a22 a33 a44 a55\n\n        lower form:\n        a00 a11 a22 a33 a44 a55\n        a10 a21 a32 a43 a54 *\n        a20 a31 a42 a53 *   *\n\n    Cells marked with * are not used.\n\n    Parameters\n    ----------\n    a_band : (u+1, M) array_like\n        The bands of the M by M matrix a.\n    lower : bool, optional\n        Is the matrix in the lower form. (Default is upper form)\n    overwrite_a_band : bool, optional\n        Discard data in a_band (may enhance performance)\n    select : {'a', 'v', 'i'}, optional\n        Which eigenvalues to calculate\n\n        ======  ========================================\n        select  calculated\n        ======  ========================================\n        'a'     All eigenvalues\n        'v'     Eigenvalues in the interval (min, max]\n        'i'     Eigenvalues with indices min <= i <= max\n        ======  ========================================\n    select_range : (min, max), optional\n        Range of selected eigenvalues\n    check_finite : bool, optional\n        Whether to check that the input matrix contains only finite numbers.\n        Disabling may give a performance gain, but may result in problems\n        (crashes, non-termination) if the inputs do contain infinities or NaNs.\n\n    Returns\n    -------\n    w : (M,) ndarray\n        The eigenvalues, in ascending order, each repeated according to its\n        multiplicity.\n\n    Raises\n    ------\n    LinAlgError\n        If eigenvalue computation does not converge.\n\n    See Also\n    --------\n    eig_banded : eigenvalues and right eigenvectors for symmetric/Hermitian\n        band matrices\n    eigvalsh_tridiagonal : eigenvalues of symmetric/Hermitian tridiagonal\n        matrices\n    eigvals : eigenvalues of general arrays\n    eigh : eigenvalues and right eigenvectors for symmetric/Hermitian arrays\n    eig : eigenvalues and right eigenvectors for non-symmetric arrays\n    ")
    
    # Call to eig_banded(...): (line 845)
    # Processing the call arguments (line 845)
    # Getting the type of 'a_band' (line 845)
    a_band_16438 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 845, 22), 'a_band', False)
    # Processing the call keyword arguments (line 845)
    # Getting the type of 'lower' (line 845)
    lower_16439 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 845, 36), 'lower', False)
    keyword_16440 = lower_16439
    int_16441 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 845, 56), 'int')
    keyword_16442 = int_16441
    # Getting the type of 'overwrite_a_band' (line 846)
    overwrite_a_band_16443 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 846, 39), 'overwrite_a_band', False)
    keyword_16444 = overwrite_a_band_16443
    # Getting the type of 'select' (line 846)
    select_16445 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 846, 64), 'select', False)
    keyword_16446 = select_16445
    # Getting the type of 'select_range' (line 847)
    select_range_16447 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 847, 35), 'select_range', False)
    keyword_16448 = select_range_16447
    # Getting the type of 'check_finite' (line 847)
    check_finite_16449 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 847, 62), 'check_finite', False)
    keyword_16450 = check_finite_16449
    kwargs_16451 = {'lower': keyword_16440, 'select_range': keyword_16448, 'select': keyword_16446, 'eigvals_only': keyword_16442, 'check_finite': keyword_16450, 'overwrite_a_band': keyword_16444}
    # Getting the type of 'eig_banded' (line 845)
    eig_banded_16437 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 845, 11), 'eig_banded', False)
    # Calling eig_banded(args, kwargs) (line 845)
    eig_banded_call_result_16452 = invoke(stypy.reporting.localization.Localization(__file__, 845, 11), eig_banded_16437, *[a_band_16438], **kwargs_16451)
    
    # Assigning a type to the variable 'stypy_return_type' (line 845)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 845, 4), 'stypy_return_type', eig_banded_call_result_16452)
    
    # ################# End of 'eigvals_banded(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'eigvals_banded' in the type store
    # Getting the type of 'stypy_return_type' (line 767)
    stypy_return_type_16453 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 767, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_16453)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'eigvals_banded'
    return stypy_return_type_16453

# Assigning a type to the variable 'eigvals_banded' (line 767)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 767, 0), 'eigvals_banded', eigvals_banded)

@norecursion
def eigvalsh_tridiagonal(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    str_16454 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 850, 38), 'str', 'a')
    # Getting the type of 'None' (line 850)
    None_16455 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 850, 56), 'None')
    # Getting the type of 'True' (line 851)
    True_16456 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 851, 38), 'True')
    float_16457 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 851, 48), 'float')
    str_16458 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 851, 66), 'str', 'auto')
    defaults = [str_16454, None_16455, True_16456, float_16457, str_16458]
    # Create a new context for function 'eigvalsh_tridiagonal'
    module_type_store = module_type_store.open_function_context('eigvalsh_tridiagonal', 850, 0, False)
    
    # Passed parameters checking function
    eigvalsh_tridiagonal.stypy_localization = localization
    eigvalsh_tridiagonal.stypy_type_of_self = None
    eigvalsh_tridiagonal.stypy_type_store = module_type_store
    eigvalsh_tridiagonal.stypy_function_name = 'eigvalsh_tridiagonal'
    eigvalsh_tridiagonal.stypy_param_names_list = ['d', 'e', 'select', 'select_range', 'check_finite', 'tol', 'lapack_driver']
    eigvalsh_tridiagonal.stypy_varargs_param_name = None
    eigvalsh_tridiagonal.stypy_kwargs_param_name = None
    eigvalsh_tridiagonal.stypy_call_defaults = defaults
    eigvalsh_tridiagonal.stypy_call_varargs = varargs
    eigvalsh_tridiagonal.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'eigvalsh_tridiagonal', ['d', 'e', 'select', 'select_range', 'check_finite', 'tol', 'lapack_driver'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'eigvalsh_tridiagonal', localization, ['d', 'e', 'select', 'select_range', 'check_finite', 'tol', 'lapack_driver'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'eigvalsh_tridiagonal(...)' code ##################

    str_16459 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 913, (-1)), 'str', "\n    Solve eigenvalue problem for a real symmetric tridiagonal matrix.\n\n    Find eigenvalues `w` of ``a``::\n\n        a v[:,i] = w[i] v[:,i]\n        v.H v    = identity\n\n    For a real symmetric matrix ``a`` with diagonal elements `d` and\n    off-diagonal elements `e`.\n\n    Parameters\n    ----------\n    d : ndarray, shape (ndim,)\n        The diagonal elements of the array.\n    e : ndarray, shape (ndim-1,)\n        The off-diagonal elements of the array.\n    select : {'a', 'v', 'i'}, optional\n        Which eigenvalues to calculate\n\n        ======  ========================================\n        select  calculated\n        ======  ========================================\n        'a'     All eigenvalues\n        'v'     Eigenvalues in the interval (min, max]\n        'i'     Eigenvalues with indices min <= i <= max\n        ======  ========================================\n    select_range : (min, max), optional\n        Range of selected eigenvalues\n    check_finite : bool, optional\n        Whether to check that the input matrix contains only finite numbers.\n        Disabling may give a performance gain, but may result in problems\n        (crashes, non-termination) if the inputs do contain infinities or NaNs.\n    tol : float\n        The absolute tolerance to which each eigenvalue is required\n        (only used when ``lapack_driver='stebz'``).\n        An eigenvalue (or cluster) is considered to have converged if it\n        lies in an interval of this width. If <= 0. (default),\n        the value ``eps*|a|`` is used where eps is the machine precision,\n        and ``|a|`` is the 1-norm of the matrix ``a``.\n    lapack_driver : str\n        LAPACK function to use, can be 'auto', 'stemr', 'stebz',  'sterf',\n        or 'stev'. When 'auto' (default), it will use 'stemr' if ``select='a'``\n        and 'stebz' otherwise. 'sterf' and 'stev' can only be used when\n        ``select='a'``.\n\n    Returns\n    -------\n    w : (M,) ndarray\n        The eigenvalues, in ascending order, each repeated according to its\n        multiplicity.\n\n    Raises\n    ------\n    LinAlgError\n        If eigenvalue computation does not converge.\n\n    See Also\n    --------\n    eigh_tridiagonal : eigenvalues and right eiegenvectors for\n        symmetric/Hermitian tridiagonal matrices\n    ")
    
    # Call to eigh_tridiagonal(...): (line 914)
    # Processing the call arguments (line 914)
    # Getting the type of 'd' (line 915)
    d_16461 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 915, 8), 'd', False)
    # Getting the type of 'e' (line 915)
    e_16462 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 915, 11), 'e', False)
    # Processing the call keyword arguments (line 914)
    # Getting the type of 'True' (line 915)
    True_16463 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 915, 27), 'True', False)
    keyword_16464 = True_16463
    # Getting the type of 'select' (line 915)
    select_16465 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 915, 40), 'select', False)
    keyword_16466 = select_16465
    # Getting the type of 'select_range' (line 915)
    select_range_16467 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 915, 61), 'select_range', False)
    keyword_16468 = select_range_16467
    # Getting the type of 'check_finite' (line 916)
    check_finite_16469 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 916, 21), 'check_finite', False)
    keyword_16470 = check_finite_16469
    # Getting the type of 'tol' (line 916)
    tol_16471 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 916, 39), 'tol', False)
    keyword_16472 = tol_16471
    # Getting the type of 'lapack_driver' (line 916)
    lapack_driver_16473 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 916, 58), 'lapack_driver', False)
    keyword_16474 = lapack_driver_16473
    kwargs_16475 = {'lapack_driver': keyword_16474, 'check_finite': keyword_16470, 'tol': keyword_16472, 'eigvals_only': keyword_16464, 'select_range': keyword_16468, 'select': keyword_16466}
    # Getting the type of 'eigh_tridiagonal' (line 914)
    eigh_tridiagonal_16460 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 914, 11), 'eigh_tridiagonal', False)
    # Calling eigh_tridiagonal(args, kwargs) (line 914)
    eigh_tridiagonal_call_result_16476 = invoke(stypy.reporting.localization.Localization(__file__, 914, 11), eigh_tridiagonal_16460, *[d_16461, e_16462], **kwargs_16475)
    
    # Assigning a type to the variable 'stypy_return_type' (line 914)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 914, 4), 'stypy_return_type', eigh_tridiagonal_call_result_16476)
    
    # ################# End of 'eigvalsh_tridiagonal(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'eigvalsh_tridiagonal' in the type store
    # Getting the type of 'stypy_return_type' (line 850)
    stypy_return_type_16477 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 850, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_16477)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'eigvalsh_tridiagonal'
    return stypy_return_type_16477

# Assigning a type to the variable 'eigvalsh_tridiagonal' (line 850)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 850, 0), 'eigvalsh_tridiagonal', eigvalsh_tridiagonal)

@norecursion
def eigh_tridiagonal(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'False' (line 919)
    False_16478 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 919, 40), 'False')
    str_16479 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 919, 54), 'str', 'a')
    # Getting the type of 'None' (line 919)
    None_16480 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 919, 72), 'None')
    # Getting the type of 'True' (line 920)
    True_16481 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 920, 34), 'True')
    float_16482 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 920, 44), 'float')
    str_16483 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 920, 62), 'str', 'auto')
    defaults = [False_16478, str_16479, None_16480, True_16481, float_16482, str_16483]
    # Create a new context for function 'eigh_tridiagonal'
    module_type_store = module_type_store.open_function_context('eigh_tridiagonal', 919, 0, False)
    
    # Passed parameters checking function
    eigh_tridiagonal.stypy_localization = localization
    eigh_tridiagonal.stypy_type_of_self = None
    eigh_tridiagonal.stypy_type_store = module_type_store
    eigh_tridiagonal.stypy_function_name = 'eigh_tridiagonal'
    eigh_tridiagonal.stypy_param_names_list = ['d', 'e', 'eigvals_only', 'select', 'select_range', 'check_finite', 'tol', 'lapack_driver']
    eigh_tridiagonal.stypy_varargs_param_name = None
    eigh_tridiagonal.stypy_kwargs_param_name = None
    eigh_tridiagonal.stypy_call_defaults = defaults
    eigh_tridiagonal.stypy_call_varargs = varargs
    eigh_tridiagonal.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'eigh_tridiagonal', ['d', 'e', 'eigvals_only', 'select', 'select_range', 'check_finite', 'tol', 'lapack_driver'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'eigh_tridiagonal', localization, ['d', 'e', 'eigvals_only', 'select', 'select_range', 'check_finite', 'tol', 'lapack_driver'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'eigh_tridiagonal(...)' code ##################

    str_16484 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 996, (-1)), 'str', "\n    Solve eigenvalue problem for a real symmetric tridiagonal matrix.\n\n    Find eigenvalues `w` and optionally right eigenvectors `v` of ``a``::\n\n        a v[:,i] = w[i] v[:,i]\n        v.H v    = identity\n\n    For a real symmetric matrix ``a`` with diagonal elements `d` and\n    off-diagonal elements `e`.\n\n    Parameters\n    ----------\n    d : ndarray, shape (ndim,)\n        The diagonal elements of the array.\n    e : ndarray, shape (ndim-1,)\n        The off-diagonal elements of the array.\n    select : {'a', 'v', 'i'}, optional\n        Which eigenvalues to calculate\n\n        ======  ========================================\n        select  calculated\n        ======  ========================================\n        'a'     All eigenvalues\n        'v'     Eigenvalues in the interval (min, max]\n        'i'     Eigenvalues with indices min <= i <= max\n        ======  ========================================\n    select_range : (min, max), optional\n        Range of selected eigenvalues\n    check_finite : bool, optional\n        Whether to check that the input matrix contains only finite numbers.\n        Disabling may give a performance gain, but may result in problems\n        (crashes, non-termination) if the inputs do contain infinities or NaNs.\n    tol : float\n        The absolute tolerance to which each eigenvalue is required\n        (only used when 'stebz' is the `lapack_driver`).\n        An eigenvalue (or cluster) is considered to have converged if it\n        lies in an interval of this width. If <= 0. (default),\n        the value ``eps*|a|`` is used where eps is the machine precision,\n        and ``|a|`` is the 1-norm of the matrix ``a``.\n    lapack_driver : str\n        LAPACK function to use, can be 'auto', 'stemr', 'stebz', 'sterf',\n        or 'stev'. When 'auto' (default), it will use 'stemr' if ``select='a'``\n        and 'stebz' otherwise. When 'stebz' is used to find the eigenvalues and\n        ``eigvals_only=False``, then a second LAPACK call (to ``?STEIN``) is\n        used to find the corresponding eigenvectors. 'sterf' can only be\n        used when ``eigvals_only=True`` and ``select='a'``. 'stev' can only\n        be used when ``select='a'``.\n\n    Returns\n    -------\n    w : (M,) ndarray\n        The eigenvalues, in ascending order, each repeated according to its\n        multiplicity.\n    v : (M, M) ndarray\n        The normalized eigenvector corresponding to the eigenvalue ``w[i]`` is\n        the column ``v[:,i]``.\n\n    Raises\n    ------\n    LinAlgError\n        If eigenvalue computation does not converge.\n\n    See Also\n    --------\n    eigvalsh_tridiagonal : eigenvalues of symmetric/Hermitian tridiagonal\n        matrices\n    eig : eigenvalues and right eigenvectors for non-symmetric arrays\n    eigh : eigenvalues and right eigenvectors for symmetric/Hermitian arrays\n    eig_banded : eigenvalues and right eigenvectors for symmetric/Hermitian\n        band matrices\n\n    Notes\n    -----\n    This function makes use of LAPACK ``S/DSTEMR`` routines.\n    ")
    
    # Assigning a Call to a Name (line 997):
    
    # Assigning a Call to a Name (line 997):
    
    # Call to _asarray_validated(...): (line 997)
    # Processing the call arguments (line 997)
    # Getting the type of 'd' (line 997)
    d_16486 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 997, 27), 'd', False)
    # Processing the call keyword arguments (line 997)
    # Getting the type of 'check_finite' (line 997)
    check_finite_16487 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 997, 43), 'check_finite', False)
    keyword_16488 = check_finite_16487
    kwargs_16489 = {'check_finite': keyword_16488}
    # Getting the type of '_asarray_validated' (line 997)
    _asarray_validated_16485 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 997, 8), '_asarray_validated', False)
    # Calling _asarray_validated(args, kwargs) (line 997)
    _asarray_validated_call_result_16490 = invoke(stypy.reporting.localization.Localization(__file__, 997, 8), _asarray_validated_16485, *[d_16486], **kwargs_16489)
    
    # Assigning a type to the variable 'd' (line 997)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 997, 4), 'd', _asarray_validated_call_result_16490)
    
    # Assigning a Call to a Name (line 998):
    
    # Assigning a Call to a Name (line 998):
    
    # Call to _asarray_validated(...): (line 998)
    # Processing the call arguments (line 998)
    # Getting the type of 'e' (line 998)
    e_16492 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 998, 27), 'e', False)
    # Processing the call keyword arguments (line 998)
    # Getting the type of 'check_finite' (line 998)
    check_finite_16493 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 998, 43), 'check_finite', False)
    keyword_16494 = check_finite_16493
    kwargs_16495 = {'check_finite': keyword_16494}
    # Getting the type of '_asarray_validated' (line 998)
    _asarray_validated_16491 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 998, 8), '_asarray_validated', False)
    # Calling _asarray_validated(args, kwargs) (line 998)
    _asarray_validated_call_result_16496 = invoke(stypy.reporting.localization.Localization(__file__, 998, 8), _asarray_validated_16491, *[e_16492], **kwargs_16495)
    
    # Assigning a type to the variable 'e' (line 998)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 998, 4), 'e', _asarray_validated_call_result_16496)
    
    
    # Obtaining an instance of the builtin type 'tuple' (line 999)
    tuple_16497 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 999, 18), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 999)
    # Adding element type (line 999)
    # Getting the type of 'd' (line 999)
    d_16498 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 999, 18), 'd')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 999, 18), tuple_16497, d_16498)
    # Adding element type (line 999)
    # Getting the type of 'e' (line 999)
    e_16499 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 999, 21), 'e')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 999, 18), tuple_16497, e_16499)
    
    # Testing the type of a for loop iterable (line 999)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 999, 4), tuple_16497)
    # Getting the type of the for loop variable (line 999)
    for_loop_var_16500 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 999, 4), tuple_16497)
    # Assigning a type to the variable 'check' (line 999)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 999, 4), 'check', for_loop_var_16500)
    # SSA begins for a for statement (line 999)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    
    # Getting the type of 'check' (line 1000)
    check_16501 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1000, 11), 'check')
    # Obtaining the member 'ndim' of a type (line 1000)
    ndim_16502 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1000, 11), check_16501, 'ndim')
    int_16503 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1000, 25), 'int')
    # Applying the binary operator '!=' (line 1000)
    result_ne_16504 = python_operator(stypy.reporting.localization.Localization(__file__, 1000, 11), '!=', ndim_16502, int_16503)
    
    # Testing the type of an if condition (line 1000)
    if_condition_16505 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1000, 8), result_ne_16504)
    # Assigning a type to the variable 'if_condition_16505' (line 1000)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1000, 8), 'if_condition_16505', if_condition_16505)
    # SSA begins for if statement (line 1000)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 1001)
    # Processing the call arguments (line 1001)
    str_16507 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1001, 29), 'str', 'expected one-dimensional array')
    # Processing the call keyword arguments (line 1001)
    kwargs_16508 = {}
    # Getting the type of 'ValueError' (line 1001)
    ValueError_16506 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1001, 18), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 1001)
    ValueError_call_result_16509 = invoke(stypy.reporting.localization.Localization(__file__, 1001, 18), ValueError_16506, *[str_16507], **kwargs_16508)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 1001, 12), ValueError_call_result_16509, 'raise parameter', BaseException)
    # SSA join for if statement (line 1000)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'check' (line 1002)
    check_16510 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1002, 11), 'check')
    # Obtaining the member 'dtype' of a type (line 1002)
    dtype_16511 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1002, 11), check_16510, 'dtype')
    # Obtaining the member 'char' of a type (line 1002)
    char_16512 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1002, 11), dtype_16511, 'char')
    str_16513 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1002, 31), 'str', 'GFD')
    # Applying the binary operator 'in' (line 1002)
    result_contains_16514 = python_operator(stypy.reporting.localization.Localization(__file__, 1002, 11), 'in', char_16512, str_16513)
    
    # Testing the type of an if condition (line 1002)
    if_condition_16515 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1002, 8), result_contains_16514)
    # Assigning a type to the variable 'if_condition_16515' (line 1002)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1002, 8), 'if_condition_16515', if_condition_16515)
    # SSA begins for if statement (line 1002)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to TypeError(...): (line 1003)
    # Processing the call arguments (line 1003)
    str_16517 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1003, 28), 'str', 'Only real arrays currently supported')
    # Processing the call keyword arguments (line 1003)
    kwargs_16518 = {}
    # Getting the type of 'TypeError' (line 1003)
    TypeError_16516 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1003, 18), 'TypeError', False)
    # Calling TypeError(args, kwargs) (line 1003)
    TypeError_call_result_16519 = invoke(stypy.reporting.localization.Localization(__file__, 1003, 18), TypeError_16516, *[str_16517], **kwargs_16518)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 1003, 12), TypeError_call_result_16519, 'raise parameter', BaseException)
    # SSA join for if statement (line 1002)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'd' (line 1004)
    d_16520 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1004, 7), 'd')
    # Obtaining the member 'size' of a type (line 1004)
    size_16521 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1004, 7), d_16520, 'size')
    # Getting the type of 'e' (line 1004)
    e_16522 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1004, 17), 'e')
    # Obtaining the member 'size' of a type (line 1004)
    size_16523 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1004, 17), e_16522, 'size')
    int_16524 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1004, 26), 'int')
    # Applying the binary operator '+' (line 1004)
    result_add_16525 = python_operator(stypy.reporting.localization.Localization(__file__, 1004, 17), '+', size_16523, int_16524)
    
    # Applying the binary operator '!=' (line 1004)
    result_ne_16526 = python_operator(stypy.reporting.localization.Localization(__file__, 1004, 7), '!=', size_16521, result_add_16525)
    
    # Testing the type of an if condition (line 1004)
    if_condition_16527 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1004, 4), result_ne_16526)
    # Assigning a type to the variable 'if_condition_16527' (line 1004)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1004, 4), 'if_condition_16527', if_condition_16527)
    # SSA begins for if statement (line 1004)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 1005)
    # Processing the call arguments (line 1005)
    str_16529 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1005, 25), 'str', 'd (%s) must have one more element than e (%s)')
    
    # Obtaining an instance of the builtin type 'tuple' (line 1006)
    tuple_16530 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1006, 28), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 1006)
    # Adding element type (line 1006)
    # Getting the type of 'd' (line 1006)
    d_16531 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1006, 28), 'd', False)
    # Obtaining the member 'size' of a type (line 1006)
    size_16532 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1006, 28), d_16531, 'size')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1006, 28), tuple_16530, size_16532)
    # Adding element type (line 1006)
    # Getting the type of 'e' (line 1006)
    e_16533 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1006, 36), 'e', False)
    # Obtaining the member 'size' of a type (line 1006)
    size_16534 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1006, 36), e_16533, 'size')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1006, 28), tuple_16530, size_16534)
    
    # Applying the binary operator '%' (line 1005)
    result_mod_16535 = python_operator(stypy.reporting.localization.Localization(__file__, 1005, 25), '%', str_16529, tuple_16530)
    
    # Processing the call keyword arguments (line 1005)
    kwargs_16536 = {}
    # Getting the type of 'ValueError' (line 1005)
    ValueError_16528 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1005, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 1005)
    ValueError_call_result_16537 = invoke(stypy.reporting.localization.Localization(__file__, 1005, 14), ValueError_16528, *[result_mod_16535], **kwargs_16536)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 1005, 8), ValueError_call_result_16537, 'raise parameter', BaseException)
    # SSA join for if statement (line 1004)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Tuple (line 1007):
    
    # Assigning a Subscript to a Name (line 1007):
    
    # Obtaining the type of the subscript
    int_16538 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1007, 4), 'int')
    
    # Call to _check_select(...): (line 1007)
    # Processing the call arguments (line 1007)
    # Getting the type of 'select' (line 1008)
    select_16540 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1008, 8), 'select', False)
    # Getting the type of 'select_range' (line 1008)
    select_range_16541 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1008, 16), 'select_range', False)
    int_16542 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1008, 30), 'int')
    # Getting the type of 'd' (line 1008)
    d_16543 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1008, 33), 'd', False)
    # Obtaining the member 'size' of a type (line 1008)
    size_16544 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1008, 33), d_16543, 'size')
    # Processing the call keyword arguments (line 1007)
    kwargs_16545 = {}
    # Getting the type of '_check_select' (line 1007)
    _check_select_16539 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1007, 32), '_check_select', False)
    # Calling _check_select(args, kwargs) (line 1007)
    _check_select_call_result_16546 = invoke(stypy.reporting.localization.Localization(__file__, 1007, 32), _check_select_16539, *[select_16540, select_range_16541, int_16542, size_16544], **kwargs_16545)
    
    # Obtaining the member '__getitem__' of a type (line 1007)
    getitem___16547 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1007, 4), _check_select_call_result_16546, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1007)
    subscript_call_result_16548 = invoke(stypy.reporting.localization.Localization(__file__, 1007, 4), getitem___16547, int_16538)
    
    # Assigning a type to the variable 'tuple_var_assignment_14103' (line 1007)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1007, 4), 'tuple_var_assignment_14103', subscript_call_result_16548)
    
    # Assigning a Subscript to a Name (line 1007):
    
    # Obtaining the type of the subscript
    int_16549 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1007, 4), 'int')
    
    # Call to _check_select(...): (line 1007)
    # Processing the call arguments (line 1007)
    # Getting the type of 'select' (line 1008)
    select_16551 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1008, 8), 'select', False)
    # Getting the type of 'select_range' (line 1008)
    select_range_16552 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1008, 16), 'select_range', False)
    int_16553 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1008, 30), 'int')
    # Getting the type of 'd' (line 1008)
    d_16554 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1008, 33), 'd', False)
    # Obtaining the member 'size' of a type (line 1008)
    size_16555 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1008, 33), d_16554, 'size')
    # Processing the call keyword arguments (line 1007)
    kwargs_16556 = {}
    # Getting the type of '_check_select' (line 1007)
    _check_select_16550 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1007, 32), '_check_select', False)
    # Calling _check_select(args, kwargs) (line 1007)
    _check_select_call_result_16557 = invoke(stypy.reporting.localization.Localization(__file__, 1007, 32), _check_select_16550, *[select_16551, select_range_16552, int_16553, size_16555], **kwargs_16556)
    
    # Obtaining the member '__getitem__' of a type (line 1007)
    getitem___16558 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1007, 4), _check_select_call_result_16557, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1007)
    subscript_call_result_16559 = invoke(stypy.reporting.localization.Localization(__file__, 1007, 4), getitem___16558, int_16549)
    
    # Assigning a type to the variable 'tuple_var_assignment_14104' (line 1007)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1007, 4), 'tuple_var_assignment_14104', subscript_call_result_16559)
    
    # Assigning a Subscript to a Name (line 1007):
    
    # Obtaining the type of the subscript
    int_16560 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1007, 4), 'int')
    
    # Call to _check_select(...): (line 1007)
    # Processing the call arguments (line 1007)
    # Getting the type of 'select' (line 1008)
    select_16562 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1008, 8), 'select', False)
    # Getting the type of 'select_range' (line 1008)
    select_range_16563 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1008, 16), 'select_range', False)
    int_16564 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1008, 30), 'int')
    # Getting the type of 'd' (line 1008)
    d_16565 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1008, 33), 'd', False)
    # Obtaining the member 'size' of a type (line 1008)
    size_16566 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1008, 33), d_16565, 'size')
    # Processing the call keyword arguments (line 1007)
    kwargs_16567 = {}
    # Getting the type of '_check_select' (line 1007)
    _check_select_16561 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1007, 32), '_check_select', False)
    # Calling _check_select(args, kwargs) (line 1007)
    _check_select_call_result_16568 = invoke(stypy.reporting.localization.Localization(__file__, 1007, 32), _check_select_16561, *[select_16562, select_range_16563, int_16564, size_16566], **kwargs_16567)
    
    # Obtaining the member '__getitem__' of a type (line 1007)
    getitem___16569 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1007, 4), _check_select_call_result_16568, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1007)
    subscript_call_result_16570 = invoke(stypy.reporting.localization.Localization(__file__, 1007, 4), getitem___16569, int_16560)
    
    # Assigning a type to the variable 'tuple_var_assignment_14105' (line 1007)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1007, 4), 'tuple_var_assignment_14105', subscript_call_result_16570)
    
    # Assigning a Subscript to a Name (line 1007):
    
    # Obtaining the type of the subscript
    int_16571 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1007, 4), 'int')
    
    # Call to _check_select(...): (line 1007)
    # Processing the call arguments (line 1007)
    # Getting the type of 'select' (line 1008)
    select_16573 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1008, 8), 'select', False)
    # Getting the type of 'select_range' (line 1008)
    select_range_16574 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1008, 16), 'select_range', False)
    int_16575 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1008, 30), 'int')
    # Getting the type of 'd' (line 1008)
    d_16576 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1008, 33), 'd', False)
    # Obtaining the member 'size' of a type (line 1008)
    size_16577 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1008, 33), d_16576, 'size')
    # Processing the call keyword arguments (line 1007)
    kwargs_16578 = {}
    # Getting the type of '_check_select' (line 1007)
    _check_select_16572 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1007, 32), '_check_select', False)
    # Calling _check_select(args, kwargs) (line 1007)
    _check_select_call_result_16579 = invoke(stypy.reporting.localization.Localization(__file__, 1007, 32), _check_select_16572, *[select_16573, select_range_16574, int_16575, size_16577], **kwargs_16578)
    
    # Obtaining the member '__getitem__' of a type (line 1007)
    getitem___16580 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1007, 4), _check_select_call_result_16579, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1007)
    subscript_call_result_16581 = invoke(stypy.reporting.localization.Localization(__file__, 1007, 4), getitem___16580, int_16571)
    
    # Assigning a type to the variable 'tuple_var_assignment_14106' (line 1007)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1007, 4), 'tuple_var_assignment_14106', subscript_call_result_16581)
    
    # Assigning a Subscript to a Name (line 1007):
    
    # Obtaining the type of the subscript
    int_16582 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1007, 4), 'int')
    
    # Call to _check_select(...): (line 1007)
    # Processing the call arguments (line 1007)
    # Getting the type of 'select' (line 1008)
    select_16584 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1008, 8), 'select', False)
    # Getting the type of 'select_range' (line 1008)
    select_range_16585 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1008, 16), 'select_range', False)
    int_16586 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1008, 30), 'int')
    # Getting the type of 'd' (line 1008)
    d_16587 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1008, 33), 'd', False)
    # Obtaining the member 'size' of a type (line 1008)
    size_16588 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1008, 33), d_16587, 'size')
    # Processing the call keyword arguments (line 1007)
    kwargs_16589 = {}
    # Getting the type of '_check_select' (line 1007)
    _check_select_16583 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1007, 32), '_check_select', False)
    # Calling _check_select(args, kwargs) (line 1007)
    _check_select_call_result_16590 = invoke(stypy.reporting.localization.Localization(__file__, 1007, 32), _check_select_16583, *[select_16584, select_range_16585, int_16586, size_16588], **kwargs_16589)
    
    # Obtaining the member '__getitem__' of a type (line 1007)
    getitem___16591 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1007, 4), _check_select_call_result_16590, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1007)
    subscript_call_result_16592 = invoke(stypy.reporting.localization.Localization(__file__, 1007, 4), getitem___16591, int_16582)
    
    # Assigning a type to the variable 'tuple_var_assignment_14107' (line 1007)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1007, 4), 'tuple_var_assignment_14107', subscript_call_result_16592)
    
    # Assigning a Subscript to a Name (line 1007):
    
    # Obtaining the type of the subscript
    int_16593 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1007, 4), 'int')
    
    # Call to _check_select(...): (line 1007)
    # Processing the call arguments (line 1007)
    # Getting the type of 'select' (line 1008)
    select_16595 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1008, 8), 'select', False)
    # Getting the type of 'select_range' (line 1008)
    select_range_16596 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1008, 16), 'select_range', False)
    int_16597 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1008, 30), 'int')
    # Getting the type of 'd' (line 1008)
    d_16598 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1008, 33), 'd', False)
    # Obtaining the member 'size' of a type (line 1008)
    size_16599 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1008, 33), d_16598, 'size')
    # Processing the call keyword arguments (line 1007)
    kwargs_16600 = {}
    # Getting the type of '_check_select' (line 1007)
    _check_select_16594 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1007, 32), '_check_select', False)
    # Calling _check_select(args, kwargs) (line 1007)
    _check_select_call_result_16601 = invoke(stypy.reporting.localization.Localization(__file__, 1007, 32), _check_select_16594, *[select_16595, select_range_16596, int_16597, size_16599], **kwargs_16600)
    
    # Obtaining the member '__getitem__' of a type (line 1007)
    getitem___16602 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1007, 4), _check_select_call_result_16601, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1007)
    subscript_call_result_16603 = invoke(stypy.reporting.localization.Localization(__file__, 1007, 4), getitem___16602, int_16593)
    
    # Assigning a type to the variable 'tuple_var_assignment_14108' (line 1007)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1007, 4), 'tuple_var_assignment_14108', subscript_call_result_16603)
    
    # Assigning a Name to a Name (line 1007):
    # Getting the type of 'tuple_var_assignment_14103' (line 1007)
    tuple_var_assignment_14103_16604 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1007, 4), 'tuple_var_assignment_14103')
    # Assigning a type to the variable 'select' (line 1007)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1007, 4), 'select', tuple_var_assignment_14103_16604)
    
    # Assigning a Name to a Name (line 1007):
    # Getting the type of 'tuple_var_assignment_14104' (line 1007)
    tuple_var_assignment_14104_16605 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1007, 4), 'tuple_var_assignment_14104')
    # Assigning a type to the variable 'vl' (line 1007)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1007, 12), 'vl', tuple_var_assignment_14104_16605)
    
    # Assigning a Name to a Name (line 1007):
    # Getting the type of 'tuple_var_assignment_14105' (line 1007)
    tuple_var_assignment_14105_16606 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1007, 4), 'tuple_var_assignment_14105')
    # Assigning a type to the variable 'vu' (line 1007)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1007, 16), 'vu', tuple_var_assignment_14105_16606)
    
    # Assigning a Name to a Name (line 1007):
    # Getting the type of 'tuple_var_assignment_14106' (line 1007)
    tuple_var_assignment_14106_16607 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1007, 4), 'tuple_var_assignment_14106')
    # Assigning a type to the variable 'il' (line 1007)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1007, 20), 'il', tuple_var_assignment_14106_16607)
    
    # Assigning a Name to a Name (line 1007):
    # Getting the type of 'tuple_var_assignment_14107' (line 1007)
    tuple_var_assignment_14107_16608 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1007, 4), 'tuple_var_assignment_14107')
    # Assigning a type to the variable 'iu' (line 1007)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1007, 24), 'iu', tuple_var_assignment_14107_16608)
    
    # Assigning a Name to a Name (line 1007):
    # Getting the type of 'tuple_var_assignment_14108' (line 1007)
    tuple_var_assignment_14108_16609 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1007, 4), 'tuple_var_assignment_14108')
    # Assigning a type to the variable '_' (line 1007)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1007, 28), '_', tuple_var_assignment_14108_16609)
    
    
    
    # Call to isinstance(...): (line 1009)
    # Processing the call arguments (line 1009)
    # Getting the type of 'lapack_driver' (line 1009)
    lapack_driver_16611 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1009, 22), 'lapack_driver', False)
    # Getting the type of 'string_types' (line 1009)
    string_types_16612 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1009, 37), 'string_types', False)
    # Processing the call keyword arguments (line 1009)
    kwargs_16613 = {}
    # Getting the type of 'isinstance' (line 1009)
    isinstance_16610 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1009, 11), 'isinstance', False)
    # Calling isinstance(args, kwargs) (line 1009)
    isinstance_call_result_16614 = invoke(stypy.reporting.localization.Localization(__file__, 1009, 11), isinstance_16610, *[lapack_driver_16611, string_types_16612], **kwargs_16613)
    
    # Applying the 'not' unary operator (line 1009)
    result_not__16615 = python_operator(stypy.reporting.localization.Localization(__file__, 1009, 7), 'not', isinstance_call_result_16614)
    
    # Testing the type of an if condition (line 1009)
    if_condition_16616 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1009, 4), result_not__16615)
    # Assigning a type to the variable 'if_condition_16616' (line 1009)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1009, 4), 'if_condition_16616', if_condition_16616)
    # SSA begins for if statement (line 1009)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to TypeError(...): (line 1010)
    # Processing the call arguments (line 1010)
    str_16618 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1010, 24), 'str', 'lapack_driver must be str')
    # Processing the call keyword arguments (line 1010)
    kwargs_16619 = {}
    # Getting the type of 'TypeError' (line 1010)
    TypeError_16617 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1010, 14), 'TypeError', False)
    # Calling TypeError(args, kwargs) (line 1010)
    TypeError_call_result_16620 = invoke(stypy.reporting.localization.Localization(__file__, 1010, 14), TypeError_16617, *[str_16618], **kwargs_16619)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 1010, 8), TypeError_call_result_16620, 'raise parameter', BaseException)
    # SSA join for if statement (line 1009)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Tuple to a Name (line 1011):
    
    # Assigning a Tuple to a Name (line 1011):
    
    # Obtaining an instance of the builtin type 'tuple' (line 1011)
    tuple_16621 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1011, 15), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 1011)
    # Adding element type (line 1011)
    str_16622 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1011, 15), 'str', 'auto')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1011, 15), tuple_16621, str_16622)
    # Adding element type (line 1011)
    str_16623 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1011, 23), 'str', 'stemr')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1011, 15), tuple_16621, str_16623)
    # Adding element type (line 1011)
    str_16624 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1011, 32), 'str', 'sterf')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1011, 15), tuple_16621, str_16624)
    # Adding element type (line 1011)
    str_16625 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1011, 41), 'str', 'stebz')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1011, 15), tuple_16621, str_16625)
    # Adding element type (line 1011)
    str_16626 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1011, 50), 'str', 'stev')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1011, 15), tuple_16621, str_16626)
    
    # Assigning a type to the variable 'drivers' (line 1011)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1011, 4), 'drivers', tuple_16621)
    
    
    # Getting the type of 'lapack_driver' (line 1012)
    lapack_driver_16627 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1012, 7), 'lapack_driver')
    # Getting the type of 'drivers' (line 1012)
    drivers_16628 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1012, 28), 'drivers')
    # Applying the binary operator 'notin' (line 1012)
    result_contains_16629 = python_operator(stypy.reporting.localization.Localization(__file__, 1012, 7), 'notin', lapack_driver_16627, drivers_16628)
    
    # Testing the type of an if condition (line 1012)
    if_condition_16630 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1012, 4), result_contains_16629)
    # Assigning a type to the variable 'if_condition_16630' (line 1012)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1012, 4), 'if_condition_16630', if_condition_16630)
    # SSA begins for if statement (line 1012)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 1013)
    # Processing the call arguments (line 1013)
    str_16632 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1013, 25), 'str', 'lapack_driver must be one of %s, got %s')
    
    # Obtaining an instance of the builtin type 'tuple' (line 1014)
    tuple_16633 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1014, 28), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 1014)
    # Adding element type (line 1014)
    # Getting the type of 'drivers' (line 1014)
    drivers_16634 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1014, 28), 'drivers', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1014, 28), tuple_16633, drivers_16634)
    # Adding element type (line 1014)
    # Getting the type of 'lapack_driver' (line 1014)
    lapack_driver_16635 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1014, 37), 'lapack_driver', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1014, 28), tuple_16633, lapack_driver_16635)
    
    # Applying the binary operator '%' (line 1013)
    result_mod_16636 = python_operator(stypy.reporting.localization.Localization(__file__, 1013, 25), '%', str_16632, tuple_16633)
    
    # Processing the call keyword arguments (line 1013)
    kwargs_16637 = {}
    # Getting the type of 'ValueError' (line 1013)
    ValueError_16631 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1013, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 1013)
    ValueError_call_result_16638 = invoke(stypy.reporting.localization.Localization(__file__, 1013, 14), ValueError_16631, *[result_mod_16636], **kwargs_16637)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 1013, 8), ValueError_call_result_16638, 'raise parameter', BaseException)
    # SSA join for if statement (line 1012)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'lapack_driver' (line 1015)
    lapack_driver_16639 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1015, 7), 'lapack_driver')
    str_16640 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1015, 24), 'str', 'auto')
    # Applying the binary operator '==' (line 1015)
    result_eq_16641 = python_operator(stypy.reporting.localization.Localization(__file__, 1015, 7), '==', lapack_driver_16639, str_16640)
    
    # Testing the type of an if condition (line 1015)
    if_condition_16642 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1015, 4), result_eq_16641)
    # Assigning a type to the variable 'if_condition_16642' (line 1015)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1015, 4), 'if_condition_16642', if_condition_16642)
    # SSA begins for if statement (line 1015)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a IfExp to a Name (line 1016):
    
    # Assigning a IfExp to a Name (line 1016):
    
    
    # Getting the type of 'select' (line 1016)
    select_16643 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1016, 35), 'select')
    int_16644 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1016, 45), 'int')
    # Applying the binary operator '==' (line 1016)
    result_eq_16645 = python_operator(stypy.reporting.localization.Localization(__file__, 1016, 35), '==', select_16643, int_16644)
    
    # Testing the type of an if expression (line 1016)
    is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1016, 24), result_eq_16645)
    # SSA begins for if expression (line 1016)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if expression')
    str_16646 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1016, 24), 'str', 'stemr')
    # SSA branch for the else part of an if expression (line 1016)
    module_type_store.open_ssa_branch('if expression else')
    str_16647 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1016, 52), 'str', 'stebz')
    # SSA join for if expression (line 1016)
    module_type_store = module_type_store.join_ssa_context()
    if_exp_16648 = union_type.UnionType.add(str_16646, str_16647)
    
    # Assigning a type to the variable 'lapack_driver' (line 1016)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1016, 8), 'lapack_driver', if_exp_16648)
    # SSA join for if statement (line 1015)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Tuple (line 1017):
    
    # Assigning a Subscript to a Name (line 1017):
    
    # Obtaining the type of the subscript
    int_16649 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1017, 4), 'int')
    
    # Call to get_lapack_funcs(...): (line 1017)
    # Processing the call arguments (line 1017)
    
    # Obtaining an instance of the builtin type 'tuple' (line 1017)
    tuple_16651 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1017, 30), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 1017)
    # Adding element type (line 1017)
    # Getting the type of 'lapack_driver' (line 1017)
    lapack_driver_16652 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1017, 30), 'lapack_driver', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1017, 30), tuple_16651, lapack_driver_16652)
    
    
    # Obtaining an instance of the builtin type 'tuple' (line 1017)
    tuple_16653 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1017, 48), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 1017)
    # Adding element type (line 1017)
    # Getting the type of 'd' (line 1017)
    d_16654 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1017, 48), 'd', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1017, 48), tuple_16653, d_16654)
    # Adding element type (line 1017)
    # Getting the type of 'e' (line 1017)
    e_16655 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1017, 51), 'e', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1017, 48), tuple_16653, e_16655)
    
    # Processing the call keyword arguments (line 1017)
    kwargs_16656 = {}
    # Getting the type of 'get_lapack_funcs' (line 1017)
    get_lapack_funcs_16650 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1017, 12), 'get_lapack_funcs', False)
    # Calling get_lapack_funcs(args, kwargs) (line 1017)
    get_lapack_funcs_call_result_16657 = invoke(stypy.reporting.localization.Localization(__file__, 1017, 12), get_lapack_funcs_16650, *[tuple_16651, tuple_16653], **kwargs_16656)
    
    # Obtaining the member '__getitem__' of a type (line 1017)
    getitem___16658 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1017, 4), get_lapack_funcs_call_result_16657, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1017)
    subscript_call_result_16659 = invoke(stypy.reporting.localization.Localization(__file__, 1017, 4), getitem___16658, int_16649)
    
    # Assigning a type to the variable 'tuple_var_assignment_14109' (line 1017)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1017, 4), 'tuple_var_assignment_14109', subscript_call_result_16659)
    
    # Assigning a Name to a Name (line 1017):
    # Getting the type of 'tuple_var_assignment_14109' (line 1017)
    tuple_var_assignment_14109_16660 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1017, 4), 'tuple_var_assignment_14109')
    # Assigning a type to the variable 'func' (line 1017)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1017, 4), 'func', tuple_var_assignment_14109_16660)
    
    # Assigning a UnaryOp to a Name (line 1018):
    
    # Assigning a UnaryOp to a Name (line 1018):
    
    # Getting the type of 'eigvals_only' (line 1018)
    eigvals_only_16661 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1018, 20), 'eigvals_only')
    # Applying the 'not' unary operator (line 1018)
    result_not__16662 = python_operator(stypy.reporting.localization.Localization(__file__, 1018, 16), 'not', eigvals_only_16661)
    
    # Assigning a type to the variable 'compute_v' (line 1018)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1018, 4), 'compute_v', result_not__16662)
    
    
    # Getting the type of 'lapack_driver' (line 1019)
    lapack_driver_16663 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1019, 7), 'lapack_driver')
    str_16664 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1019, 24), 'str', 'sterf')
    # Applying the binary operator '==' (line 1019)
    result_eq_16665 = python_operator(stypy.reporting.localization.Localization(__file__, 1019, 7), '==', lapack_driver_16663, str_16664)
    
    # Testing the type of an if condition (line 1019)
    if_condition_16666 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1019, 4), result_eq_16665)
    # Assigning a type to the variable 'if_condition_16666' (line 1019)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1019, 4), 'if_condition_16666', if_condition_16666)
    # SSA begins for if statement (line 1019)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    
    # Getting the type of 'select' (line 1020)
    select_16667 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1020, 11), 'select')
    int_16668 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1020, 21), 'int')
    # Applying the binary operator '!=' (line 1020)
    result_ne_16669 = python_operator(stypy.reporting.localization.Localization(__file__, 1020, 11), '!=', select_16667, int_16668)
    
    # Testing the type of an if condition (line 1020)
    if_condition_16670 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1020, 8), result_ne_16669)
    # Assigning a type to the variable 'if_condition_16670' (line 1020)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1020, 8), 'if_condition_16670', if_condition_16670)
    # SSA begins for if statement (line 1020)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 1021)
    # Processing the call arguments (line 1021)
    str_16672 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1021, 29), 'str', 'sterf can only be used when select == "a"')
    # Processing the call keyword arguments (line 1021)
    kwargs_16673 = {}
    # Getting the type of 'ValueError' (line 1021)
    ValueError_16671 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1021, 18), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 1021)
    ValueError_call_result_16674 = invoke(stypy.reporting.localization.Localization(__file__, 1021, 18), ValueError_16671, *[str_16672], **kwargs_16673)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 1021, 12), ValueError_call_result_16674, 'raise parameter', BaseException)
    # SSA join for if statement (line 1020)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'eigvals_only' (line 1022)
    eigvals_only_16675 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1022, 15), 'eigvals_only')
    # Applying the 'not' unary operator (line 1022)
    result_not__16676 = python_operator(stypy.reporting.localization.Localization(__file__, 1022, 11), 'not', eigvals_only_16675)
    
    # Testing the type of an if condition (line 1022)
    if_condition_16677 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1022, 8), result_not__16676)
    # Assigning a type to the variable 'if_condition_16677' (line 1022)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1022, 8), 'if_condition_16677', if_condition_16677)
    # SSA begins for if statement (line 1022)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 1023)
    # Processing the call arguments (line 1023)
    str_16679 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1023, 29), 'str', 'sterf can only be used when eigvals_only is True')
    # Processing the call keyword arguments (line 1023)
    kwargs_16680 = {}
    # Getting the type of 'ValueError' (line 1023)
    ValueError_16678 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1023, 18), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 1023)
    ValueError_call_result_16681 = invoke(stypy.reporting.localization.Localization(__file__, 1023, 18), ValueError_16678, *[str_16679], **kwargs_16680)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 1023, 12), ValueError_call_result_16681, 'raise parameter', BaseException)
    # SSA join for if statement (line 1022)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Tuple (line 1025):
    
    # Assigning a Subscript to a Name (line 1025):
    
    # Obtaining the type of the subscript
    int_16682 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1025, 8), 'int')
    
    # Call to func(...): (line 1025)
    # Processing the call arguments (line 1025)
    # Getting the type of 'd' (line 1025)
    d_16684 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1025, 23), 'd', False)
    # Getting the type of 'e' (line 1025)
    e_16685 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1025, 26), 'e', False)
    # Processing the call keyword arguments (line 1025)
    kwargs_16686 = {}
    # Getting the type of 'func' (line 1025)
    func_16683 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1025, 18), 'func', False)
    # Calling func(args, kwargs) (line 1025)
    func_call_result_16687 = invoke(stypy.reporting.localization.Localization(__file__, 1025, 18), func_16683, *[d_16684, e_16685], **kwargs_16686)
    
    # Obtaining the member '__getitem__' of a type (line 1025)
    getitem___16688 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1025, 8), func_call_result_16687, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1025)
    subscript_call_result_16689 = invoke(stypy.reporting.localization.Localization(__file__, 1025, 8), getitem___16688, int_16682)
    
    # Assigning a type to the variable 'tuple_var_assignment_14110' (line 1025)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1025, 8), 'tuple_var_assignment_14110', subscript_call_result_16689)
    
    # Assigning a Subscript to a Name (line 1025):
    
    # Obtaining the type of the subscript
    int_16690 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1025, 8), 'int')
    
    # Call to func(...): (line 1025)
    # Processing the call arguments (line 1025)
    # Getting the type of 'd' (line 1025)
    d_16692 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1025, 23), 'd', False)
    # Getting the type of 'e' (line 1025)
    e_16693 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1025, 26), 'e', False)
    # Processing the call keyword arguments (line 1025)
    kwargs_16694 = {}
    # Getting the type of 'func' (line 1025)
    func_16691 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1025, 18), 'func', False)
    # Calling func(args, kwargs) (line 1025)
    func_call_result_16695 = invoke(stypy.reporting.localization.Localization(__file__, 1025, 18), func_16691, *[d_16692, e_16693], **kwargs_16694)
    
    # Obtaining the member '__getitem__' of a type (line 1025)
    getitem___16696 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1025, 8), func_call_result_16695, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1025)
    subscript_call_result_16697 = invoke(stypy.reporting.localization.Localization(__file__, 1025, 8), getitem___16696, int_16690)
    
    # Assigning a type to the variable 'tuple_var_assignment_14111' (line 1025)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1025, 8), 'tuple_var_assignment_14111', subscript_call_result_16697)
    
    # Assigning a Name to a Name (line 1025):
    # Getting the type of 'tuple_var_assignment_14110' (line 1025)
    tuple_var_assignment_14110_16698 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1025, 8), 'tuple_var_assignment_14110')
    # Assigning a type to the variable 'w' (line 1025)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1025, 8), 'w', tuple_var_assignment_14110_16698)
    
    # Assigning a Name to a Name (line 1025):
    # Getting the type of 'tuple_var_assignment_14111' (line 1025)
    tuple_var_assignment_14111_16699 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1025, 8), 'tuple_var_assignment_14111')
    # Assigning a type to the variable 'info' (line 1025)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1025, 11), 'info', tuple_var_assignment_14111_16699)
    
    # Assigning a Call to a Name (line 1026):
    
    # Assigning a Call to a Name (line 1026):
    
    # Call to len(...): (line 1026)
    # Processing the call arguments (line 1026)
    # Getting the type of 'w' (line 1026)
    w_16701 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1026, 16), 'w', False)
    # Processing the call keyword arguments (line 1026)
    kwargs_16702 = {}
    # Getting the type of 'len' (line 1026)
    len_16700 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1026, 12), 'len', False)
    # Calling len(args, kwargs) (line 1026)
    len_call_result_16703 = invoke(stypy.reporting.localization.Localization(__file__, 1026, 12), len_16700, *[w_16701], **kwargs_16702)
    
    # Assigning a type to the variable 'm' (line 1026)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1026, 8), 'm', len_call_result_16703)
    # SSA branch for the else part of an if statement (line 1019)
    module_type_store.open_ssa_branch('else')
    
    
    # Getting the type of 'lapack_driver' (line 1027)
    lapack_driver_16704 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1027, 9), 'lapack_driver')
    str_16705 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1027, 26), 'str', 'stev')
    # Applying the binary operator '==' (line 1027)
    result_eq_16706 = python_operator(stypy.reporting.localization.Localization(__file__, 1027, 9), '==', lapack_driver_16704, str_16705)
    
    # Testing the type of an if condition (line 1027)
    if_condition_16707 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1027, 9), result_eq_16706)
    # Assigning a type to the variable 'if_condition_16707' (line 1027)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1027, 9), 'if_condition_16707', if_condition_16707)
    # SSA begins for if statement (line 1027)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    
    # Getting the type of 'select' (line 1028)
    select_16708 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1028, 11), 'select')
    int_16709 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1028, 21), 'int')
    # Applying the binary operator '!=' (line 1028)
    result_ne_16710 = python_operator(stypy.reporting.localization.Localization(__file__, 1028, 11), '!=', select_16708, int_16709)
    
    # Testing the type of an if condition (line 1028)
    if_condition_16711 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1028, 8), result_ne_16710)
    # Assigning a type to the variable 'if_condition_16711' (line 1028)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1028, 8), 'if_condition_16711', if_condition_16711)
    # SSA begins for if statement (line 1028)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 1029)
    # Processing the call arguments (line 1029)
    str_16713 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1029, 29), 'str', 'stev can only be used when select == "a"')
    # Processing the call keyword arguments (line 1029)
    kwargs_16714 = {}
    # Getting the type of 'ValueError' (line 1029)
    ValueError_16712 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1029, 18), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 1029)
    ValueError_call_result_16715 = invoke(stypy.reporting.localization.Localization(__file__, 1029, 18), ValueError_16712, *[str_16713], **kwargs_16714)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 1029, 12), ValueError_call_result_16715, 'raise parameter', BaseException)
    # SSA join for if statement (line 1028)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Tuple (line 1030):
    
    # Assigning a Subscript to a Name (line 1030):
    
    # Obtaining the type of the subscript
    int_16716 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1030, 8), 'int')
    
    # Call to func(...): (line 1030)
    # Processing the call arguments (line 1030)
    # Getting the type of 'd' (line 1030)
    d_16718 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1030, 26), 'd', False)
    # Getting the type of 'e' (line 1030)
    e_16719 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1030, 29), 'e', False)
    # Processing the call keyword arguments (line 1030)
    # Getting the type of 'compute_v' (line 1030)
    compute_v_16720 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1030, 42), 'compute_v', False)
    keyword_16721 = compute_v_16720
    kwargs_16722 = {'compute_v': keyword_16721}
    # Getting the type of 'func' (line 1030)
    func_16717 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1030, 21), 'func', False)
    # Calling func(args, kwargs) (line 1030)
    func_call_result_16723 = invoke(stypy.reporting.localization.Localization(__file__, 1030, 21), func_16717, *[d_16718, e_16719], **kwargs_16722)
    
    # Obtaining the member '__getitem__' of a type (line 1030)
    getitem___16724 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1030, 8), func_call_result_16723, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1030)
    subscript_call_result_16725 = invoke(stypy.reporting.localization.Localization(__file__, 1030, 8), getitem___16724, int_16716)
    
    # Assigning a type to the variable 'tuple_var_assignment_14112' (line 1030)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1030, 8), 'tuple_var_assignment_14112', subscript_call_result_16725)
    
    # Assigning a Subscript to a Name (line 1030):
    
    # Obtaining the type of the subscript
    int_16726 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1030, 8), 'int')
    
    # Call to func(...): (line 1030)
    # Processing the call arguments (line 1030)
    # Getting the type of 'd' (line 1030)
    d_16728 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1030, 26), 'd', False)
    # Getting the type of 'e' (line 1030)
    e_16729 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1030, 29), 'e', False)
    # Processing the call keyword arguments (line 1030)
    # Getting the type of 'compute_v' (line 1030)
    compute_v_16730 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1030, 42), 'compute_v', False)
    keyword_16731 = compute_v_16730
    kwargs_16732 = {'compute_v': keyword_16731}
    # Getting the type of 'func' (line 1030)
    func_16727 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1030, 21), 'func', False)
    # Calling func(args, kwargs) (line 1030)
    func_call_result_16733 = invoke(stypy.reporting.localization.Localization(__file__, 1030, 21), func_16727, *[d_16728, e_16729], **kwargs_16732)
    
    # Obtaining the member '__getitem__' of a type (line 1030)
    getitem___16734 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1030, 8), func_call_result_16733, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1030)
    subscript_call_result_16735 = invoke(stypy.reporting.localization.Localization(__file__, 1030, 8), getitem___16734, int_16726)
    
    # Assigning a type to the variable 'tuple_var_assignment_14113' (line 1030)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1030, 8), 'tuple_var_assignment_14113', subscript_call_result_16735)
    
    # Assigning a Subscript to a Name (line 1030):
    
    # Obtaining the type of the subscript
    int_16736 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1030, 8), 'int')
    
    # Call to func(...): (line 1030)
    # Processing the call arguments (line 1030)
    # Getting the type of 'd' (line 1030)
    d_16738 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1030, 26), 'd', False)
    # Getting the type of 'e' (line 1030)
    e_16739 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1030, 29), 'e', False)
    # Processing the call keyword arguments (line 1030)
    # Getting the type of 'compute_v' (line 1030)
    compute_v_16740 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1030, 42), 'compute_v', False)
    keyword_16741 = compute_v_16740
    kwargs_16742 = {'compute_v': keyword_16741}
    # Getting the type of 'func' (line 1030)
    func_16737 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1030, 21), 'func', False)
    # Calling func(args, kwargs) (line 1030)
    func_call_result_16743 = invoke(stypy.reporting.localization.Localization(__file__, 1030, 21), func_16737, *[d_16738, e_16739], **kwargs_16742)
    
    # Obtaining the member '__getitem__' of a type (line 1030)
    getitem___16744 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1030, 8), func_call_result_16743, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1030)
    subscript_call_result_16745 = invoke(stypy.reporting.localization.Localization(__file__, 1030, 8), getitem___16744, int_16736)
    
    # Assigning a type to the variable 'tuple_var_assignment_14114' (line 1030)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1030, 8), 'tuple_var_assignment_14114', subscript_call_result_16745)
    
    # Assigning a Name to a Name (line 1030):
    # Getting the type of 'tuple_var_assignment_14112' (line 1030)
    tuple_var_assignment_14112_16746 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1030, 8), 'tuple_var_assignment_14112')
    # Assigning a type to the variable 'w' (line 1030)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1030, 8), 'w', tuple_var_assignment_14112_16746)
    
    # Assigning a Name to a Name (line 1030):
    # Getting the type of 'tuple_var_assignment_14113' (line 1030)
    tuple_var_assignment_14113_16747 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1030, 8), 'tuple_var_assignment_14113')
    # Assigning a type to the variable 'v' (line 1030)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1030, 11), 'v', tuple_var_assignment_14113_16747)
    
    # Assigning a Name to a Name (line 1030):
    # Getting the type of 'tuple_var_assignment_14114' (line 1030)
    tuple_var_assignment_14114_16748 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1030, 8), 'tuple_var_assignment_14114')
    # Assigning a type to the variable 'info' (line 1030)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1030, 14), 'info', tuple_var_assignment_14114_16748)
    
    # Assigning a Call to a Name (line 1031):
    
    # Assigning a Call to a Name (line 1031):
    
    # Call to len(...): (line 1031)
    # Processing the call arguments (line 1031)
    # Getting the type of 'w' (line 1031)
    w_16750 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1031, 16), 'w', False)
    # Processing the call keyword arguments (line 1031)
    kwargs_16751 = {}
    # Getting the type of 'len' (line 1031)
    len_16749 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1031, 12), 'len', False)
    # Calling len(args, kwargs) (line 1031)
    len_call_result_16752 = invoke(stypy.reporting.localization.Localization(__file__, 1031, 12), len_16749, *[w_16750], **kwargs_16751)
    
    # Assigning a type to the variable 'm' (line 1031)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1031, 8), 'm', len_call_result_16752)
    # SSA branch for the else part of an if statement (line 1027)
    module_type_store.open_ssa_branch('else')
    
    
    # Getting the type of 'lapack_driver' (line 1032)
    lapack_driver_16753 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1032, 9), 'lapack_driver')
    str_16754 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1032, 26), 'str', 'stebz')
    # Applying the binary operator '==' (line 1032)
    result_eq_16755 = python_operator(stypy.reporting.localization.Localization(__file__, 1032, 9), '==', lapack_driver_16753, str_16754)
    
    # Testing the type of an if condition (line 1032)
    if_condition_16756 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1032, 9), result_eq_16755)
    # Assigning a type to the variable 'if_condition_16756' (line 1032)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1032, 9), 'if_condition_16756', if_condition_16756)
    # SSA begins for if statement (line 1032)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 1033):
    
    # Assigning a Call to a Name (line 1033):
    
    # Call to float(...): (line 1033)
    # Processing the call arguments (line 1033)
    # Getting the type of 'tol' (line 1033)
    tol_16758 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1033, 20), 'tol', False)
    # Processing the call keyword arguments (line 1033)
    kwargs_16759 = {}
    # Getting the type of 'float' (line 1033)
    float_16757 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1033, 14), 'float', False)
    # Calling float(args, kwargs) (line 1033)
    float_call_result_16760 = invoke(stypy.reporting.localization.Localization(__file__, 1033, 14), float_16757, *[tol_16758], **kwargs_16759)
    
    # Assigning a type to the variable 'tol' (line 1033)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1033, 8), 'tol', float_call_result_16760)
    
    # Assigning a Str to a Name (line 1034):
    
    # Assigning a Str to a Name (line 1034):
    str_16761 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1034, 24), 'str', 'stebz')
    # Assigning a type to the variable 'internal_name' (line 1034)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1034, 8), 'internal_name', str_16761)
    
    # Assigning a Call to a Tuple (line 1035):
    
    # Assigning a Subscript to a Name (line 1035):
    
    # Obtaining the type of the subscript
    int_16762 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1035, 8), 'int')
    
    # Call to get_lapack_funcs(...): (line 1035)
    # Processing the call arguments (line 1035)
    
    # Obtaining an instance of the builtin type 'tuple' (line 1035)
    tuple_16764 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1035, 35), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 1035)
    # Adding element type (line 1035)
    # Getting the type of 'internal_name' (line 1035)
    internal_name_16765 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1035, 35), 'internal_name', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1035, 35), tuple_16764, internal_name_16765)
    
    
    # Obtaining an instance of the builtin type 'tuple' (line 1035)
    tuple_16766 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1035, 53), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 1035)
    # Adding element type (line 1035)
    # Getting the type of 'd' (line 1035)
    d_16767 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1035, 53), 'd', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1035, 53), tuple_16766, d_16767)
    # Adding element type (line 1035)
    # Getting the type of 'e' (line 1035)
    e_16768 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1035, 56), 'e', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1035, 53), tuple_16766, e_16768)
    
    # Processing the call keyword arguments (line 1035)
    kwargs_16769 = {}
    # Getting the type of 'get_lapack_funcs' (line 1035)
    get_lapack_funcs_16763 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1035, 17), 'get_lapack_funcs', False)
    # Calling get_lapack_funcs(args, kwargs) (line 1035)
    get_lapack_funcs_call_result_16770 = invoke(stypy.reporting.localization.Localization(__file__, 1035, 17), get_lapack_funcs_16763, *[tuple_16764, tuple_16766], **kwargs_16769)
    
    # Obtaining the member '__getitem__' of a type (line 1035)
    getitem___16771 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1035, 8), get_lapack_funcs_call_result_16770, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1035)
    subscript_call_result_16772 = invoke(stypy.reporting.localization.Localization(__file__, 1035, 8), getitem___16771, int_16762)
    
    # Assigning a type to the variable 'tuple_var_assignment_14115' (line 1035)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1035, 8), 'tuple_var_assignment_14115', subscript_call_result_16772)
    
    # Assigning a Name to a Name (line 1035):
    # Getting the type of 'tuple_var_assignment_14115' (line 1035)
    tuple_var_assignment_14115_16773 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1035, 8), 'tuple_var_assignment_14115')
    # Assigning a type to the variable 'stebz' (line 1035)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1035, 8), 'stebz', tuple_var_assignment_14115_16773)
    
    # Assigning a IfExp to a Name (line 1038):
    
    # Assigning a IfExp to a Name (line 1038):
    
    # Getting the type of 'eigvals_only' (line 1038)
    eigvals_only_16774 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1038, 23), 'eigvals_only')
    # Testing the type of an if expression (line 1038)
    is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1038, 16), eigvals_only_16774)
    # SSA begins for if expression (line 1038)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if expression')
    str_16775 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1038, 16), 'str', 'E')
    # SSA branch for the else part of an if expression (line 1038)
    module_type_store.open_ssa_branch('if expression else')
    str_16776 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1038, 41), 'str', 'B')
    # SSA join for if expression (line 1038)
    module_type_store = module_type_store.join_ssa_context()
    if_exp_16777 = union_type.UnionType.add(str_16775, str_16776)
    
    # Assigning a type to the variable 'order' (line 1038)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1038, 8), 'order', if_exp_16777)
    
    # Assigning a Call to a Tuple (line 1039):
    
    # Assigning a Subscript to a Name (line 1039):
    
    # Obtaining the type of the subscript
    int_16778 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1039, 8), 'int')
    
    # Call to stebz(...): (line 1039)
    # Processing the call arguments (line 1039)
    # Getting the type of 'd' (line 1039)
    d_16780 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1039, 43), 'd', False)
    # Getting the type of 'e' (line 1039)
    e_16781 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1039, 46), 'e', False)
    # Getting the type of 'select' (line 1039)
    select_16782 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1039, 49), 'select', False)
    # Getting the type of 'vl' (line 1039)
    vl_16783 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1039, 57), 'vl', False)
    # Getting the type of 'vu' (line 1039)
    vu_16784 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1039, 61), 'vu', False)
    # Getting the type of 'il' (line 1039)
    il_16785 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1039, 65), 'il', False)
    # Getting the type of 'iu' (line 1039)
    iu_16786 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1039, 69), 'iu', False)
    # Getting the type of 'tol' (line 1039)
    tol_16787 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1039, 73), 'tol', False)
    # Getting the type of 'order' (line 1040)
    order_16788 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1040, 43), 'order', False)
    # Processing the call keyword arguments (line 1039)
    kwargs_16789 = {}
    # Getting the type of 'stebz' (line 1039)
    stebz_16779 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1039, 37), 'stebz', False)
    # Calling stebz(args, kwargs) (line 1039)
    stebz_call_result_16790 = invoke(stypy.reporting.localization.Localization(__file__, 1039, 37), stebz_16779, *[d_16780, e_16781, select_16782, vl_16783, vu_16784, il_16785, iu_16786, tol_16787, order_16788], **kwargs_16789)
    
    # Obtaining the member '__getitem__' of a type (line 1039)
    getitem___16791 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1039, 8), stebz_call_result_16790, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1039)
    subscript_call_result_16792 = invoke(stypy.reporting.localization.Localization(__file__, 1039, 8), getitem___16791, int_16778)
    
    # Assigning a type to the variable 'tuple_var_assignment_14116' (line 1039)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1039, 8), 'tuple_var_assignment_14116', subscript_call_result_16792)
    
    # Assigning a Subscript to a Name (line 1039):
    
    # Obtaining the type of the subscript
    int_16793 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1039, 8), 'int')
    
    # Call to stebz(...): (line 1039)
    # Processing the call arguments (line 1039)
    # Getting the type of 'd' (line 1039)
    d_16795 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1039, 43), 'd', False)
    # Getting the type of 'e' (line 1039)
    e_16796 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1039, 46), 'e', False)
    # Getting the type of 'select' (line 1039)
    select_16797 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1039, 49), 'select', False)
    # Getting the type of 'vl' (line 1039)
    vl_16798 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1039, 57), 'vl', False)
    # Getting the type of 'vu' (line 1039)
    vu_16799 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1039, 61), 'vu', False)
    # Getting the type of 'il' (line 1039)
    il_16800 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1039, 65), 'il', False)
    # Getting the type of 'iu' (line 1039)
    iu_16801 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1039, 69), 'iu', False)
    # Getting the type of 'tol' (line 1039)
    tol_16802 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1039, 73), 'tol', False)
    # Getting the type of 'order' (line 1040)
    order_16803 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1040, 43), 'order', False)
    # Processing the call keyword arguments (line 1039)
    kwargs_16804 = {}
    # Getting the type of 'stebz' (line 1039)
    stebz_16794 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1039, 37), 'stebz', False)
    # Calling stebz(args, kwargs) (line 1039)
    stebz_call_result_16805 = invoke(stypy.reporting.localization.Localization(__file__, 1039, 37), stebz_16794, *[d_16795, e_16796, select_16797, vl_16798, vu_16799, il_16800, iu_16801, tol_16802, order_16803], **kwargs_16804)
    
    # Obtaining the member '__getitem__' of a type (line 1039)
    getitem___16806 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1039, 8), stebz_call_result_16805, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1039)
    subscript_call_result_16807 = invoke(stypy.reporting.localization.Localization(__file__, 1039, 8), getitem___16806, int_16793)
    
    # Assigning a type to the variable 'tuple_var_assignment_14117' (line 1039)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1039, 8), 'tuple_var_assignment_14117', subscript_call_result_16807)
    
    # Assigning a Subscript to a Name (line 1039):
    
    # Obtaining the type of the subscript
    int_16808 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1039, 8), 'int')
    
    # Call to stebz(...): (line 1039)
    # Processing the call arguments (line 1039)
    # Getting the type of 'd' (line 1039)
    d_16810 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1039, 43), 'd', False)
    # Getting the type of 'e' (line 1039)
    e_16811 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1039, 46), 'e', False)
    # Getting the type of 'select' (line 1039)
    select_16812 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1039, 49), 'select', False)
    # Getting the type of 'vl' (line 1039)
    vl_16813 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1039, 57), 'vl', False)
    # Getting the type of 'vu' (line 1039)
    vu_16814 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1039, 61), 'vu', False)
    # Getting the type of 'il' (line 1039)
    il_16815 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1039, 65), 'il', False)
    # Getting the type of 'iu' (line 1039)
    iu_16816 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1039, 69), 'iu', False)
    # Getting the type of 'tol' (line 1039)
    tol_16817 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1039, 73), 'tol', False)
    # Getting the type of 'order' (line 1040)
    order_16818 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1040, 43), 'order', False)
    # Processing the call keyword arguments (line 1039)
    kwargs_16819 = {}
    # Getting the type of 'stebz' (line 1039)
    stebz_16809 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1039, 37), 'stebz', False)
    # Calling stebz(args, kwargs) (line 1039)
    stebz_call_result_16820 = invoke(stypy.reporting.localization.Localization(__file__, 1039, 37), stebz_16809, *[d_16810, e_16811, select_16812, vl_16813, vu_16814, il_16815, iu_16816, tol_16817, order_16818], **kwargs_16819)
    
    # Obtaining the member '__getitem__' of a type (line 1039)
    getitem___16821 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1039, 8), stebz_call_result_16820, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1039)
    subscript_call_result_16822 = invoke(stypy.reporting.localization.Localization(__file__, 1039, 8), getitem___16821, int_16808)
    
    # Assigning a type to the variable 'tuple_var_assignment_14118' (line 1039)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1039, 8), 'tuple_var_assignment_14118', subscript_call_result_16822)
    
    # Assigning a Subscript to a Name (line 1039):
    
    # Obtaining the type of the subscript
    int_16823 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1039, 8), 'int')
    
    # Call to stebz(...): (line 1039)
    # Processing the call arguments (line 1039)
    # Getting the type of 'd' (line 1039)
    d_16825 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1039, 43), 'd', False)
    # Getting the type of 'e' (line 1039)
    e_16826 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1039, 46), 'e', False)
    # Getting the type of 'select' (line 1039)
    select_16827 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1039, 49), 'select', False)
    # Getting the type of 'vl' (line 1039)
    vl_16828 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1039, 57), 'vl', False)
    # Getting the type of 'vu' (line 1039)
    vu_16829 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1039, 61), 'vu', False)
    # Getting the type of 'il' (line 1039)
    il_16830 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1039, 65), 'il', False)
    # Getting the type of 'iu' (line 1039)
    iu_16831 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1039, 69), 'iu', False)
    # Getting the type of 'tol' (line 1039)
    tol_16832 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1039, 73), 'tol', False)
    # Getting the type of 'order' (line 1040)
    order_16833 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1040, 43), 'order', False)
    # Processing the call keyword arguments (line 1039)
    kwargs_16834 = {}
    # Getting the type of 'stebz' (line 1039)
    stebz_16824 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1039, 37), 'stebz', False)
    # Calling stebz(args, kwargs) (line 1039)
    stebz_call_result_16835 = invoke(stypy.reporting.localization.Localization(__file__, 1039, 37), stebz_16824, *[d_16825, e_16826, select_16827, vl_16828, vu_16829, il_16830, iu_16831, tol_16832, order_16833], **kwargs_16834)
    
    # Obtaining the member '__getitem__' of a type (line 1039)
    getitem___16836 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1039, 8), stebz_call_result_16835, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1039)
    subscript_call_result_16837 = invoke(stypy.reporting.localization.Localization(__file__, 1039, 8), getitem___16836, int_16823)
    
    # Assigning a type to the variable 'tuple_var_assignment_14119' (line 1039)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1039, 8), 'tuple_var_assignment_14119', subscript_call_result_16837)
    
    # Assigning a Subscript to a Name (line 1039):
    
    # Obtaining the type of the subscript
    int_16838 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1039, 8), 'int')
    
    # Call to stebz(...): (line 1039)
    # Processing the call arguments (line 1039)
    # Getting the type of 'd' (line 1039)
    d_16840 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1039, 43), 'd', False)
    # Getting the type of 'e' (line 1039)
    e_16841 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1039, 46), 'e', False)
    # Getting the type of 'select' (line 1039)
    select_16842 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1039, 49), 'select', False)
    # Getting the type of 'vl' (line 1039)
    vl_16843 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1039, 57), 'vl', False)
    # Getting the type of 'vu' (line 1039)
    vu_16844 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1039, 61), 'vu', False)
    # Getting the type of 'il' (line 1039)
    il_16845 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1039, 65), 'il', False)
    # Getting the type of 'iu' (line 1039)
    iu_16846 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1039, 69), 'iu', False)
    # Getting the type of 'tol' (line 1039)
    tol_16847 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1039, 73), 'tol', False)
    # Getting the type of 'order' (line 1040)
    order_16848 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1040, 43), 'order', False)
    # Processing the call keyword arguments (line 1039)
    kwargs_16849 = {}
    # Getting the type of 'stebz' (line 1039)
    stebz_16839 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1039, 37), 'stebz', False)
    # Calling stebz(args, kwargs) (line 1039)
    stebz_call_result_16850 = invoke(stypy.reporting.localization.Localization(__file__, 1039, 37), stebz_16839, *[d_16840, e_16841, select_16842, vl_16843, vu_16844, il_16845, iu_16846, tol_16847, order_16848], **kwargs_16849)
    
    # Obtaining the member '__getitem__' of a type (line 1039)
    getitem___16851 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1039, 8), stebz_call_result_16850, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1039)
    subscript_call_result_16852 = invoke(stypy.reporting.localization.Localization(__file__, 1039, 8), getitem___16851, int_16838)
    
    # Assigning a type to the variable 'tuple_var_assignment_14120' (line 1039)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1039, 8), 'tuple_var_assignment_14120', subscript_call_result_16852)
    
    # Assigning a Name to a Name (line 1039):
    # Getting the type of 'tuple_var_assignment_14116' (line 1039)
    tuple_var_assignment_14116_16853 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1039, 8), 'tuple_var_assignment_14116')
    # Assigning a type to the variable 'm' (line 1039)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1039, 8), 'm', tuple_var_assignment_14116_16853)
    
    # Assigning a Name to a Name (line 1039):
    # Getting the type of 'tuple_var_assignment_14117' (line 1039)
    tuple_var_assignment_14117_16854 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1039, 8), 'tuple_var_assignment_14117')
    # Assigning a type to the variable 'w' (line 1039)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1039, 11), 'w', tuple_var_assignment_14117_16854)
    
    # Assigning a Name to a Name (line 1039):
    # Getting the type of 'tuple_var_assignment_14118' (line 1039)
    tuple_var_assignment_14118_16855 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1039, 8), 'tuple_var_assignment_14118')
    # Assigning a type to the variable 'iblock' (line 1039)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1039, 14), 'iblock', tuple_var_assignment_14118_16855)
    
    # Assigning a Name to a Name (line 1039):
    # Getting the type of 'tuple_var_assignment_14119' (line 1039)
    tuple_var_assignment_14119_16856 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1039, 8), 'tuple_var_assignment_14119')
    # Assigning a type to the variable 'isplit' (line 1039)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1039, 22), 'isplit', tuple_var_assignment_14119_16856)
    
    # Assigning a Name to a Name (line 1039):
    # Getting the type of 'tuple_var_assignment_14120' (line 1039)
    tuple_var_assignment_14120_16857 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1039, 8), 'tuple_var_assignment_14120')
    # Assigning a type to the variable 'info' (line 1039)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1039, 30), 'info', tuple_var_assignment_14120_16857)
    # SSA branch for the else part of an if statement (line 1032)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Call to a Name (line 1043):
    
    # Assigning a Call to a Name (line 1043):
    
    # Call to empty(...): (line 1043)
    # Processing the call arguments (line 1043)
    # Getting the type of 'e' (line 1043)
    e_16859 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1043, 19), 'e', False)
    # Obtaining the member 'size' of a type (line 1043)
    size_16860 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1043, 19), e_16859, 'size')
    int_16861 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1043, 26), 'int')
    # Applying the binary operator '+' (line 1043)
    result_add_16862 = python_operator(stypy.reporting.localization.Localization(__file__, 1043, 19), '+', size_16860, int_16861)
    
    # Getting the type of 'e' (line 1043)
    e_16863 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1043, 29), 'e', False)
    # Obtaining the member 'dtype' of a type (line 1043)
    dtype_16864 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1043, 29), e_16863, 'dtype')
    # Processing the call keyword arguments (line 1043)
    kwargs_16865 = {}
    # Getting the type of 'empty' (line 1043)
    empty_16858 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1043, 13), 'empty', False)
    # Calling empty(args, kwargs) (line 1043)
    empty_call_result_16866 = invoke(stypy.reporting.localization.Localization(__file__, 1043, 13), empty_16858, *[result_add_16862, dtype_16864], **kwargs_16865)
    
    # Assigning a type to the variable 'e_' (line 1043)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1043, 8), 'e_', empty_call_result_16866)
    
    # Assigning a Name to a Subscript (line 1044):
    
    # Assigning a Name to a Subscript (line 1044):
    # Getting the type of 'e' (line 1044)
    e_16867 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1044, 18), 'e')
    # Getting the type of 'e_' (line 1044)
    e__16868 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1044, 8), 'e_')
    int_16869 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1044, 12), 'int')
    slice_16870 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 1044, 8), None, int_16869, None)
    # Storing an element on a container (line 1044)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1044, 8), e__16868, (slice_16870, e_16867))
    
    # Assigning a Call to a Tuple (line 1045):
    
    # Assigning a Subscript to a Name (line 1045):
    
    # Obtaining the type of the subscript
    int_16871 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1045, 8), 'int')
    
    # Call to get_lapack_funcs(...): (line 1045)
    # Processing the call arguments (line 1045)
    
    # Obtaining an instance of the builtin type 'tuple' (line 1045)
    tuple_16873 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1045, 41), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 1045)
    # Adding element type (line 1045)
    str_16874 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1045, 41), 'str', 'stemr_lwork')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1045, 41), tuple_16873, str_16874)
    
    
    # Obtaining an instance of the builtin type 'tuple' (line 1045)
    tuple_16875 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1045, 59), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 1045)
    # Adding element type (line 1045)
    # Getting the type of 'd' (line 1045)
    d_16876 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1045, 59), 'd', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1045, 59), tuple_16875, d_16876)
    # Adding element type (line 1045)
    # Getting the type of 'e' (line 1045)
    e_16877 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1045, 62), 'e', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1045, 59), tuple_16875, e_16877)
    
    # Processing the call keyword arguments (line 1045)
    kwargs_16878 = {}
    # Getting the type of 'get_lapack_funcs' (line 1045)
    get_lapack_funcs_16872 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1045, 23), 'get_lapack_funcs', False)
    # Calling get_lapack_funcs(args, kwargs) (line 1045)
    get_lapack_funcs_call_result_16879 = invoke(stypy.reporting.localization.Localization(__file__, 1045, 23), get_lapack_funcs_16872, *[tuple_16873, tuple_16875], **kwargs_16878)
    
    # Obtaining the member '__getitem__' of a type (line 1045)
    getitem___16880 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1045, 8), get_lapack_funcs_call_result_16879, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1045)
    subscript_call_result_16881 = invoke(stypy.reporting.localization.Localization(__file__, 1045, 8), getitem___16880, int_16871)
    
    # Assigning a type to the variable 'tuple_var_assignment_14121' (line 1045)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1045, 8), 'tuple_var_assignment_14121', subscript_call_result_16881)
    
    # Assigning a Name to a Name (line 1045):
    # Getting the type of 'tuple_var_assignment_14121' (line 1045)
    tuple_var_assignment_14121_16882 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1045, 8), 'tuple_var_assignment_14121')
    # Assigning a type to the variable 'stemr_lwork' (line 1045)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1045, 8), 'stemr_lwork', tuple_var_assignment_14121_16882)
    
    # Assigning a Call to a Tuple (line 1046):
    
    # Assigning a Subscript to a Name (line 1046):
    
    # Obtaining the type of the subscript
    int_16883 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1046, 8), 'int')
    
    # Call to stemr_lwork(...): (line 1046)
    # Processing the call arguments (line 1046)
    # Getting the type of 'd' (line 1046)
    d_16885 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1046, 42), 'd', False)
    # Getting the type of 'e_' (line 1046)
    e__16886 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1046, 45), 'e_', False)
    # Getting the type of 'select' (line 1046)
    select_16887 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1046, 49), 'select', False)
    # Getting the type of 'vl' (line 1046)
    vl_16888 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1046, 57), 'vl', False)
    # Getting the type of 'vu' (line 1046)
    vu_16889 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1046, 61), 'vu', False)
    # Getting the type of 'il' (line 1046)
    il_16890 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1046, 65), 'il', False)
    # Getting the type of 'iu' (line 1046)
    iu_16891 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1046, 69), 'iu', False)
    # Processing the call keyword arguments (line 1046)
    # Getting the type of 'compute_v' (line 1047)
    compute_v_16892 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1047, 52), 'compute_v', False)
    keyword_16893 = compute_v_16892
    kwargs_16894 = {'compute_v': keyword_16893}
    # Getting the type of 'stemr_lwork' (line 1046)
    stemr_lwork_16884 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1046, 30), 'stemr_lwork', False)
    # Calling stemr_lwork(args, kwargs) (line 1046)
    stemr_lwork_call_result_16895 = invoke(stypy.reporting.localization.Localization(__file__, 1046, 30), stemr_lwork_16884, *[d_16885, e__16886, select_16887, vl_16888, vu_16889, il_16890, iu_16891], **kwargs_16894)
    
    # Obtaining the member '__getitem__' of a type (line 1046)
    getitem___16896 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1046, 8), stemr_lwork_call_result_16895, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1046)
    subscript_call_result_16897 = invoke(stypy.reporting.localization.Localization(__file__, 1046, 8), getitem___16896, int_16883)
    
    # Assigning a type to the variable 'tuple_var_assignment_14122' (line 1046)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1046, 8), 'tuple_var_assignment_14122', subscript_call_result_16897)
    
    # Assigning a Subscript to a Name (line 1046):
    
    # Obtaining the type of the subscript
    int_16898 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1046, 8), 'int')
    
    # Call to stemr_lwork(...): (line 1046)
    # Processing the call arguments (line 1046)
    # Getting the type of 'd' (line 1046)
    d_16900 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1046, 42), 'd', False)
    # Getting the type of 'e_' (line 1046)
    e__16901 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1046, 45), 'e_', False)
    # Getting the type of 'select' (line 1046)
    select_16902 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1046, 49), 'select', False)
    # Getting the type of 'vl' (line 1046)
    vl_16903 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1046, 57), 'vl', False)
    # Getting the type of 'vu' (line 1046)
    vu_16904 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1046, 61), 'vu', False)
    # Getting the type of 'il' (line 1046)
    il_16905 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1046, 65), 'il', False)
    # Getting the type of 'iu' (line 1046)
    iu_16906 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1046, 69), 'iu', False)
    # Processing the call keyword arguments (line 1046)
    # Getting the type of 'compute_v' (line 1047)
    compute_v_16907 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1047, 52), 'compute_v', False)
    keyword_16908 = compute_v_16907
    kwargs_16909 = {'compute_v': keyword_16908}
    # Getting the type of 'stemr_lwork' (line 1046)
    stemr_lwork_16899 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1046, 30), 'stemr_lwork', False)
    # Calling stemr_lwork(args, kwargs) (line 1046)
    stemr_lwork_call_result_16910 = invoke(stypy.reporting.localization.Localization(__file__, 1046, 30), stemr_lwork_16899, *[d_16900, e__16901, select_16902, vl_16903, vu_16904, il_16905, iu_16906], **kwargs_16909)
    
    # Obtaining the member '__getitem__' of a type (line 1046)
    getitem___16911 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1046, 8), stemr_lwork_call_result_16910, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1046)
    subscript_call_result_16912 = invoke(stypy.reporting.localization.Localization(__file__, 1046, 8), getitem___16911, int_16898)
    
    # Assigning a type to the variable 'tuple_var_assignment_14123' (line 1046)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1046, 8), 'tuple_var_assignment_14123', subscript_call_result_16912)
    
    # Assigning a Subscript to a Name (line 1046):
    
    # Obtaining the type of the subscript
    int_16913 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1046, 8), 'int')
    
    # Call to stemr_lwork(...): (line 1046)
    # Processing the call arguments (line 1046)
    # Getting the type of 'd' (line 1046)
    d_16915 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1046, 42), 'd', False)
    # Getting the type of 'e_' (line 1046)
    e__16916 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1046, 45), 'e_', False)
    # Getting the type of 'select' (line 1046)
    select_16917 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1046, 49), 'select', False)
    # Getting the type of 'vl' (line 1046)
    vl_16918 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1046, 57), 'vl', False)
    # Getting the type of 'vu' (line 1046)
    vu_16919 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1046, 61), 'vu', False)
    # Getting the type of 'il' (line 1046)
    il_16920 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1046, 65), 'il', False)
    # Getting the type of 'iu' (line 1046)
    iu_16921 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1046, 69), 'iu', False)
    # Processing the call keyword arguments (line 1046)
    # Getting the type of 'compute_v' (line 1047)
    compute_v_16922 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1047, 52), 'compute_v', False)
    keyword_16923 = compute_v_16922
    kwargs_16924 = {'compute_v': keyword_16923}
    # Getting the type of 'stemr_lwork' (line 1046)
    stemr_lwork_16914 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1046, 30), 'stemr_lwork', False)
    # Calling stemr_lwork(args, kwargs) (line 1046)
    stemr_lwork_call_result_16925 = invoke(stypy.reporting.localization.Localization(__file__, 1046, 30), stemr_lwork_16914, *[d_16915, e__16916, select_16917, vl_16918, vu_16919, il_16920, iu_16921], **kwargs_16924)
    
    # Obtaining the member '__getitem__' of a type (line 1046)
    getitem___16926 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1046, 8), stemr_lwork_call_result_16925, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1046)
    subscript_call_result_16927 = invoke(stypy.reporting.localization.Localization(__file__, 1046, 8), getitem___16926, int_16913)
    
    # Assigning a type to the variable 'tuple_var_assignment_14124' (line 1046)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1046, 8), 'tuple_var_assignment_14124', subscript_call_result_16927)
    
    # Assigning a Name to a Name (line 1046):
    # Getting the type of 'tuple_var_assignment_14122' (line 1046)
    tuple_var_assignment_14122_16928 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1046, 8), 'tuple_var_assignment_14122')
    # Assigning a type to the variable 'lwork' (line 1046)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1046, 8), 'lwork', tuple_var_assignment_14122_16928)
    
    # Assigning a Name to a Name (line 1046):
    # Getting the type of 'tuple_var_assignment_14123' (line 1046)
    tuple_var_assignment_14123_16929 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1046, 8), 'tuple_var_assignment_14123')
    # Assigning a type to the variable 'liwork' (line 1046)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1046, 15), 'liwork', tuple_var_assignment_14123_16929)
    
    # Assigning a Name to a Name (line 1046):
    # Getting the type of 'tuple_var_assignment_14124' (line 1046)
    tuple_var_assignment_14124_16930 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1046, 8), 'tuple_var_assignment_14124')
    # Assigning a type to the variable 'info' (line 1046)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1046, 23), 'info', tuple_var_assignment_14124_16930)
    
    # Call to _check_info(...): (line 1048)
    # Processing the call arguments (line 1048)
    # Getting the type of 'info' (line 1048)
    info_16932 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1048, 20), 'info', False)
    str_16933 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1048, 26), 'str', 'stemr_lwork')
    # Processing the call keyword arguments (line 1048)
    kwargs_16934 = {}
    # Getting the type of '_check_info' (line 1048)
    _check_info_16931 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1048, 8), '_check_info', False)
    # Calling _check_info(args, kwargs) (line 1048)
    _check_info_call_result_16935 = invoke(stypy.reporting.localization.Localization(__file__, 1048, 8), _check_info_16931, *[info_16932, str_16933], **kwargs_16934)
    
    
    # Assigning a Call to a Tuple (line 1049):
    
    # Assigning a Subscript to a Name (line 1049):
    
    # Obtaining the type of the subscript
    int_16936 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1049, 8), 'int')
    
    # Call to func(...): (line 1049)
    # Processing the call arguments (line 1049)
    # Getting the type of 'd' (line 1049)
    d_16938 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1049, 29), 'd', False)
    # Getting the type of 'e_' (line 1049)
    e__16939 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1049, 32), 'e_', False)
    # Getting the type of 'select' (line 1049)
    select_16940 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1049, 36), 'select', False)
    # Getting the type of 'vl' (line 1049)
    vl_16941 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1049, 44), 'vl', False)
    # Getting the type of 'vu' (line 1049)
    vu_16942 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1049, 48), 'vu', False)
    # Getting the type of 'il' (line 1049)
    il_16943 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1049, 52), 'il', False)
    # Getting the type of 'iu' (line 1049)
    iu_16944 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1049, 56), 'iu', False)
    # Processing the call keyword arguments (line 1049)
    # Getting the type of 'compute_v' (line 1050)
    compute_v_16945 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1050, 39), 'compute_v', False)
    keyword_16946 = compute_v_16945
    # Getting the type of 'lwork' (line 1050)
    lwork_16947 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1050, 56), 'lwork', False)
    keyword_16948 = lwork_16947
    # Getting the type of 'liwork' (line 1050)
    liwork_16949 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1050, 70), 'liwork', False)
    keyword_16950 = liwork_16949
    kwargs_16951 = {'liwork': keyword_16950, 'compute_v': keyword_16946, 'lwork': keyword_16948}
    # Getting the type of 'func' (line 1049)
    func_16937 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1049, 24), 'func', False)
    # Calling func(args, kwargs) (line 1049)
    func_call_result_16952 = invoke(stypy.reporting.localization.Localization(__file__, 1049, 24), func_16937, *[d_16938, e__16939, select_16940, vl_16941, vu_16942, il_16943, iu_16944], **kwargs_16951)
    
    # Obtaining the member '__getitem__' of a type (line 1049)
    getitem___16953 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1049, 8), func_call_result_16952, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1049)
    subscript_call_result_16954 = invoke(stypy.reporting.localization.Localization(__file__, 1049, 8), getitem___16953, int_16936)
    
    # Assigning a type to the variable 'tuple_var_assignment_14125' (line 1049)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1049, 8), 'tuple_var_assignment_14125', subscript_call_result_16954)
    
    # Assigning a Subscript to a Name (line 1049):
    
    # Obtaining the type of the subscript
    int_16955 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1049, 8), 'int')
    
    # Call to func(...): (line 1049)
    # Processing the call arguments (line 1049)
    # Getting the type of 'd' (line 1049)
    d_16957 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1049, 29), 'd', False)
    # Getting the type of 'e_' (line 1049)
    e__16958 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1049, 32), 'e_', False)
    # Getting the type of 'select' (line 1049)
    select_16959 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1049, 36), 'select', False)
    # Getting the type of 'vl' (line 1049)
    vl_16960 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1049, 44), 'vl', False)
    # Getting the type of 'vu' (line 1049)
    vu_16961 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1049, 48), 'vu', False)
    # Getting the type of 'il' (line 1049)
    il_16962 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1049, 52), 'il', False)
    # Getting the type of 'iu' (line 1049)
    iu_16963 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1049, 56), 'iu', False)
    # Processing the call keyword arguments (line 1049)
    # Getting the type of 'compute_v' (line 1050)
    compute_v_16964 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1050, 39), 'compute_v', False)
    keyword_16965 = compute_v_16964
    # Getting the type of 'lwork' (line 1050)
    lwork_16966 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1050, 56), 'lwork', False)
    keyword_16967 = lwork_16966
    # Getting the type of 'liwork' (line 1050)
    liwork_16968 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1050, 70), 'liwork', False)
    keyword_16969 = liwork_16968
    kwargs_16970 = {'liwork': keyword_16969, 'compute_v': keyword_16965, 'lwork': keyword_16967}
    # Getting the type of 'func' (line 1049)
    func_16956 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1049, 24), 'func', False)
    # Calling func(args, kwargs) (line 1049)
    func_call_result_16971 = invoke(stypy.reporting.localization.Localization(__file__, 1049, 24), func_16956, *[d_16957, e__16958, select_16959, vl_16960, vu_16961, il_16962, iu_16963], **kwargs_16970)
    
    # Obtaining the member '__getitem__' of a type (line 1049)
    getitem___16972 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1049, 8), func_call_result_16971, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1049)
    subscript_call_result_16973 = invoke(stypy.reporting.localization.Localization(__file__, 1049, 8), getitem___16972, int_16955)
    
    # Assigning a type to the variable 'tuple_var_assignment_14126' (line 1049)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1049, 8), 'tuple_var_assignment_14126', subscript_call_result_16973)
    
    # Assigning a Subscript to a Name (line 1049):
    
    # Obtaining the type of the subscript
    int_16974 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1049, 8), 'int')
    
    # Call to func(...): (line 1049)
    # Processing the call arguments (line 1049)
    # Getting the type of 'd' (line 1049)
    d_16976 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1049, 29), 'd', False)
    # Getting the type of 'e_' (line 1049)
    e__16977 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1049, 32), 'e_', False)
    # Getting the type of 'select' (line 1049)
    select_16978 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1049, 36), 'select', False)
    # Getting the type of 'vl' (line 1049)
    vl_16979 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1049, 44), 'vl', False)
    # Getting the type of 'vu' (line 1049)
    vu_16980 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1049, 48), 'vu', False)
    # Getting the type of 'il' (line 1049)
    il_16981 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1049, 52), 'il', False)
    # Getting the type of 'iu' (line 1049)
    iu_16982 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1049, 56), 'iu', False)
    # Processing the call keyword arguments (line 1049)
    # Getting the type of 'compute_v' (line 1050)
    compute_v_16983 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1050, 39), 'compute_v', False)
    keyword_16984 = compute_v_16983
    # Getting the type of 'lwork' (line 1050)
    lwork_16985 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1050, 56), 'lwork', False)
    keyword_16986 = lwork_16985
    # Getting the type of 'liwork' (line 1050)
    liwork_16987 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1050, 70), 'liwork', False)
    keyword_16988 = liwork_16987
    kwargs_16989 = {'liwork': keyword_16988, 'compute_v': keyword_16984, 'lwork': keyword_16986}
    # Getting the type of 'func' (line 1049)
    func_16975 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1049, 24), 'func', False)
    # Calling func(args, kwargs) (line 1049)
    func_call_result_16990 = invoke(stypy.reporting.localization.Localization(__file__, 1049, 24), func_16975, *[d_16976, e__16977, select_16978, vl_16979, vu_16980, il_16981, iu_16982], **kwargs_16989)
    
    # Obtaining the member '__getitem__' of a type (line 1049)
    getitem___16991 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1049, 8), func_call_result_16990, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1049)
    subscript_call_result_16992 = invoke(stypy.reporting.localization.Localization(__file__, 1049, 8), getitem___16991, int_16974)
    
    # Assigning a type to the variable 'tuple_var_assignment_14127' (line 1049)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1049, 8), 'tuple_var_assignment_14127', subscript_call_result_16992)
    
    # Assigning a Subscript to a Name (line 1049):
    
    # Obtaining the type of the subscript
    int_16993 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1049, 8), 'int')
    
    # Call to func(...): (line 1049)
    # Processing the call arguments (line 1049)
    # Getting the type of 'd' (line 1049)
    d_16995 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1049, 29), 'd', False)
    # Getting the type of 'e_' (line 1049)
    e__16996 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1049, 32), 'e_', False)
    # Getting the type of 'select' (line 1049)
    select_16997 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1049, 36), 'select', False)
    # Getting the type of 'vl' (line 1049)
    vl_16998 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1049, 44), 'vl', False)
    # Getting the type of 'vu' (line 1049)
    vu_16999 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1049, 48), 'vu', False)
    # Getting the type of 'il' (line 1049)
    il_17000 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1049, 52), 'il', False)
    # Getting the type of 'iu' (line 1049)
    iu_17001 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1049, 56), 'iu', False)
    # Processing the call keyword arguments (line 1049)
    # Getting the type of 'compute_v' (line 1050)
    compute_v_17002 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1050, 39), 'compute_v', False)
    keyword_17003 = compute_v_17002
    # Getting the type of 'lwork' (line 1050)
    lwork_17004 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1050, 56), 'lwork', False)
    keyword_17005 = lwork_17004
    # Getting the type of 'liwork' (line 1050)
    liwork_17006 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1050, 70), 'liwork', False)
    keyword_17007 = liwork_17006
    kwargs_17008 = {'liwork': keyword_17007, 'compute_v': keyword_17003, 'lwork': keyword_17005}
    # Getting the type of 'func' (line 1049)
    func_16994 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1049, 24), 'func', False)
    # Calling func(args, kwargs) (line 1049)
    func_call_result_17009 = invoke(stypy.reporting.localization.Localization(__file__, 1049, 24), func_16994, *[d_16995, e__16996, select_16997, vl_16998, vu_16999, il_17000, iu_17001], **kwargs_17008)
    
    # Obtaining the member '__getitem__' of a type (line 1049)
    getitem___17010 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1049, 8), func_call_result_17009, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1049)
    subscript_call_result_17011 = invoke(stypy.reporting.localization.Localization(__file__, 1049, 8), getitem___17010, int_16993)
    
    # Assigning a type to the variable 'tuple_var_assignment_14128' (line 1049)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1049, 8), 'tuple_var_assignment_14128', subscript_call_result_17011)
    
    # Assigning a Name to a Name (line 1049):
    # Getting the type of 'tuple_var_assignment_14125' (line 1049)
    tuple_var_assignment_14125_17012 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1049, 8), 'tuple_var_assignment_14125')
    # Assigning a type to the variable 'm' (line 1049)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1049, 8), 'm', tuple_var_assignment_14125_17012)
    
    # Assigning a Name to a Name (line 1049):
    # Getting the type of 'tuple_var_assignment_14126' (line 1049)
    tuple_var_assignment_14126_17013 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1049, 8), 'tuple_var_assignment_14126')
    # Assigning a type to the variable 'w' (line 1049)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1049, 11), 'w', tuple_var_assignment_14126_17013)
    
    # Assigning a Name to a Name (line 1049):
    # Getting the type of 'tuple_var_assignment_14127' (line 1049)
    tuple_var_assignment_14127_17014 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1049, 8), 'tuple_var_assignment_14127')
    # Assigning a type to the variable 'v' (line 1049)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1049, 14), 'v', tuple_var_assignment_14127_17014)
    
    # Assigning a Name to a Name (line 1049):
    # Getting the type of 'tuple_var_assignment_14128' (line 1049)
    tuple_var_assignment_14128_17015 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1049, 8), 'tuple_var_assignment_14128')
    # Assigning a type to the variable 'info' (line 1049)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1049, 17), 'info', tuple_var_assignment_14128_17015)
    # SSA join for if statement (line 1032)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 1027)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 1019)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to _check_info(...): (line 1051)
    # Processing the call arguments (line 1051)
    # Getting the type of 'info' (line 1051)
    info_17017 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1051, 16), 'info', False)
    # Getting the type of 'lapack_driver' (line 1051)
    lapack_driver_17018 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1051, 22), 'lapack_driver', False)
    str_17019 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1051, 38), 'str', ' (eigh_tridiagonal)')
    # Applying the binary operator '+' (line 1051)
    result_add_17020 = python_operator(stypy.reporting.localization.Localization(__file__, 1051, 22), '+', lapack_driver_17018, str_17019)
    
    # Processing the call keyword arguments (line 1051)
    kwargs_17021 = {}
    # Getting the type of '_check_info' (line 1051)
    _check_info_17016 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1051, 4), '_check_info', False)
    # Calling _check_info(args, kwargs) (line 1051)
    _check_info_call_result_17022 = invoke(stypy.reporting.localization.Localization(__file__, 1051, 4), _check_info_17016, *[info_17017, result_add_17020], **kwargs_17021)
    
    
    # Assigning a Subscript to a Name (line 1052):
    
    # Assigning a Subscript to a Name (line 1052):
    
    # Obtaining the type of the subscript
    # Getting the type of 'm' (line 1052)
    m_17023 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1052, 11), 'm')
    slice_17024 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 1052, 8), None, m_17023, None)
    # Getting the type of 'w' (line 1052)
    w_17025 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1052, 8), 'w')
    # Obtaining the member '__getitem__' of a type (line 1052)
    getitem___17026 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1052, 8), w_17025, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1052)
    subscript_call_result_17027 = invoke(stypy.reporting.localization.Localization(__file__, 1052, 8), getitem___17026, slice_17024)
    
    # Assigning a type to the variable 'w' (line 1052)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1052, 4), 'w', subscript_call_result_17027)
    
    # Getting the type of 'eigvals_only' (line 1053)
    eigvals_only_17028 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1053, 7), 'eigvals_only')
    # Testing the type of an if condition (line 1053)
    if_condition_17029 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1053, 4), eigvals_only_17028)
    # Assigning a type to the variable 'if_condition_17029' (line 1053)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1053, 4), 'if_condition_17029', if_condition_17029)
    # SSA begins for if statement (line 1053)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # Getting the type of 'w' (line 1054)
    w_17030 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1054, 15), 'w')
    # Assigning a type to the variable 'stypy_return_type' (line 1054)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1054, 8), 'stypy_return_type', w_17030)
    # SSA branch for the else part of an if statement (line 1053)
    module_type_store.open_ssa_branch('else')
    
    
    # Getting the type of 'lapack_driver' (line 1057)
    lapack_driver_17031 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1057, 11), 'lapack_driver')
    str_17032 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1057, 28), 'str', 'stebz')
    # Applying the binary operator '==' (line 1057)
    result_eq_17033 = python_operator(stypy.reporting.localization.Localization(__file__, 1057, 11), '==', lapack_driver_17031, str_17032)
    
    # Testing the type of an if condition (line 1057)
    if_condition_17034 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1057, 8), result_eq_17033)
    # Assigning a type to the variable 'if_condition_17034' (line 1057)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1057, 8), 'if_condition_17034', if_condition_17034)
    # SSA begins for if statement (line 1057)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Tuple (line 1058):
    
    # Assigning a Subscript to a Name (line 1058):
    
    # Obtaining the type of the subscript
    int_17035 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1058, 12), 'int')
    
    # Call to get_lapack_funcs(...): (line 1058)
    # Processing the call arguments (line 1058)
    
    # Obtaining an instance of the builtin type 'tuple' (line 1058)
    tuple_17037 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1058, 38), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 1058)
    # Adding element type (line 1058)
    str_17038 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1058, 38), 'str', 'stein')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1058, 38), tuple_17037, str_17038)
    
    
    # Obtaining an instance of the builtin type 'tuple' (line 1058)
    tuple_17039 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1058, 50), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 1058)
    # Adding element type (line 1058)
    # Getting the type of 'd' (line 1058)
    d_17040 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1058, 50), 'd', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1058, 50), tuple_17039, d_17040)
    # Adding element type (line 1058)
    # Getting the type of 'e' (line 1058)
    e_17041 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1058, 53), 'e', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1058, 50), tuple_17039, e_17041)
    
    # Processing the call keyword arguments (line 1058)
    kwargs_17042 = {}
    # Getting the type of 'get_lapack_funcs' (line 1058)
    get_lapack_funcs_17036 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1058, 20), 'get_lapack_funcs', False)
    # Calling get_lapack_funcs(args, kwargs) (line 1058)
    get_lapack_funcs_call_result_17043 = invoke(stypy.reporting.localization.Localization(__file__, 1058, 20), get_lapack_funcs_17036, *[tuple_17037, tuple_17039], **kwargs_17042)
    
    # Obtaining the member '__getitem__' of a type (line 1058)
    getitem___17044 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1058, 12), get_lapack_funcs_call_result_17043, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1058)
    subscript_call_result_17045 = invoke(stypy.reporting.localization.Localization(__file__, 1058, 12), getitem___17044, int_17035)
    
    # Assigning a type to the variable 'tuple_var_assignment_14129' (line 1058)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1058, 12), 'tuple_var_assignment_14129', subscript_call_result_17045)
    
    # Assigning a Name to a Name (line 1058):
    # Getting the type of 'tuple_var_assignment_14129' (line 1058)
    tuple_var_assignment_14129_17046 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1058, 12), 'tuple_var_assignment_14129')
    # Assigning a type to the variable 'func' (line 1058)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1058, 12), 'func', tuple_var_assignment_14129_17046)
    
    # Assigning a Call to a Tuple (line 1059):
    
    # Assigning a Subscript to a Name (line 1059):
    
    # Obtaining the type of the subscript
    int_17047 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1059, 12), 'int')
    
    # Call to func(...): (line 1059)
    # Processing the call arguments (line 1059)
    # Getting the type of 'd' (line 1059)
    d_17049 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1059, 27), 'd', False)
    # Getting the type of 'e' (line 1059)
    e_17050 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1059, 30), 'e', False)
    # Getting the type of 'w' (line 1059)
    w_17051 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1059, 33), 'w', False)
    # Getting the type of 'iblock' (line 1059)
    iblock_17052 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1059, 36), 'iblock', False)
    # Getting the type of 'isplit' (line 1059)
    isplit_17053 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1059, 44), 'isplit', False)
    # Processing the call keyword arguments (line 1059)
    kwargs_17054 = {}
    # Getting the type of 'func' (line 1059)
    func_17048 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1059, 22), 'func', False)
    # Calling func(args, kwargs) (line 1059)
    func_call_result_17055 = invoke(stypy.reporting.localization.Localization(__file__, 1059, 22), func_17048, *[d_17049, e_17050, w_17051, iblock_17052, isplit_17053], **kwargs_17054)
    
    # Obtaining the member '__getitem__' of a type (line 1059)
    getitem___17056 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1059, 12), func_call_result_17055, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1059)
    subscript_call_result_17057 = invoke(stypy.reporting.localization.Localization(__file__, 1059, 12), getitem___17056, int_17047)
    
    # Assigning a type to the variable 'tuple_var_assignment_14130' (line 1059)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1059, 12), 'tuple_var_assignment_14130', subscript_call_result_17057)
    
    # Assigning a Subscript to a Name (line 1059):
    
    # Obtaining the type of the subscript
    int_17058 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1059, 12), 'int')
    
    # Call to func(...): (line 1059)
    # Processing the call arguments (line 1059)
    # Getting the type of 'd' (line 1059)
    d_17060 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1059, 27), 'd', False)
    # Getting the type of 'e' (line 1059)
    e_17061 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1059, 30), 'e', False)
    # Getting the type of 'w' (line 1059)
    w_17062 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1059, 33), 'w', False)
    # Getting the type of 'iblock' (line 1059)
    iblock_17063 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1059, 36), 'iblock', False)
    # Getting the type of 'isplit' (line 1059)
    isplit_17064 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1059, 44), 'isplit', False)
    # Processing the call keyword arguments (line 1059)
    kwargs_17065 = {}
    # Getting the type of 'func' (line 1059)
    func_17059 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1059, 22), 'func', False)
    # Calling func(args, kwargs) (line 1059)
    func_call_result_17066 = invoke(stypy.reporting.localization.Localization(__file__, 1059, 22), func_17059, *[d_17060, e_17061, w_17062, iblock_17063, isplit_17064], **kwargs_17065)
    
    # Obtaining the member '__getitem__' of a type (line 1059)
    getitem___17067 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1059, 12), func_call_result_17066, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1059)
    subscript_call_result_17068 = invoke(stypy.reporting.localization.Localization(__file__, 1059, 12), getitem___17067, int_17058)
    
    # Assigning a type to the variable 'tuple_var_assignment_14131' (line 1059)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1059, 12), 'tuple_var_assignment_14131', subscript_call_result_17068)
    
    # Assigning a Name to a Name (line 1059):
    # Getting the type of 'tuple_var_assignment_14130' (line 1059)
    tuple_var_assignment_14130_17069 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1059, 12), 'tuple_var_assignment_14130')
    # Assigning a type to the variable 'v' (line 1059)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1059, 12), 'v', tuple_var_assignment_14130_17069)
    
    # Assigning a Name to a Name (line 1059):
    # Getting the type of 'tuple_var_assignment_14131' (line 1059)
    tuple_var_assignment_14131_17070 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1059, 12), 'tuple_var_assignment_14131')
    # Assigning a type to the variable 'info' (line 1059)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1059, 15), 'info', tuple_var_assignment_14131_17070)
    
    # Call to _check_info(...): (line 1060)
    # Processing the call arguments (line 1060)
    # Getting the type of 'info' (line 1060)
    info_17072 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1060, 24), 'info', False)
    str_17073 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1060, 30), 'str', 'stein (eigh_tridiagonal)')
    # Processing the call keyword arguments (line 1060)
    str_17074 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1061, 33), 'str', '%d eigenvectors failed to converge')
    keyword_17075 = str_17074
    kwargs_17076 = {'positive': keyword_17075}
    # Getting the type of '_check_info' (line 1060)
    _check_info_17071 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1060, 12), '_check_info', False)
    # Calling _check_info(args, kwargs) (line 1060)
    _check_info_call_result_17077 = invoke(stypy.reporting.localization.Localization(__file__, 1060, 12), _check_info_17071, *[info_17072, str_17073], **kwargs_17076)
    
    
    # Assigning a Call to a Name (line 1063):
    
    # Assigning a Call to a Name (line 1063):
    
    # Call to argsort(...): (line 1063)
    # Processing the call arguments (line 1063)
    # Getting the type of 'w' (line 1063)
    w_17079 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1063, 28), 'w', False)
    # Processing the call keyword arguments (line 1063)
    kwargs_17080 = {}
    # Getting the type of 'argsort' (line 1063)
    argsort_17078 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1063, 20), 'argsort', False)
    # Calling argsort(args, kwargs) (line 1063)
    argsort_call_result_17081 = invoke(stypy.reporting.localization.Localization(__file__, 1063, 20), argsort_17078, *[w_17079], **kwargs_17080)
    
    # Assigning a type to the variable 'order' (line 1063)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1063, 12), 'order', argsort_call_result_17081)
    
    # Assigning a Tuple to a Tuple (line 1064):
    
    # Assigning a Subscript to a Name (line 1064):
    
    # Obtaining the type of the subscript
    # Getting the type of 'order' (line 1064)
    order_17082 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1064, 21), 'order')
    # Getting the type of 'w' (line 1064)
    w_17083 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1064, 19), 'w')
    # Obtaining the member '__getitem__' of a type (line 1064)
    getitem___17084 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1064, 19), w_17083, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1064)
    subscript_call_result_17085 = invoke(stypy.reporting.localization.Localization(__file__, 1064, 19), getitem___17084, order_17082)
    
    # Assigning a type to the variable 'tuple_assignment_14132' (line 1064)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1064, 12), 'tuple_assignment_14132', subscript_call_result_17085)
    
    # Assigning a Subscript to a Name (line 1064):
    
    # Obtaining the type of the subscript
    slice_17086 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 1064, 29), None, None, None)
    # Getting the type of 'order' (line 1064)
    order_17087 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1064, 34), 'order')
    # Getting the type of 'v' (line 1064)
    v_17088 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1064, 29), 'v')
    # Obtaining the member '__getitem__' of a type (line 1064)
    getitem___17089 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1064, 29), v_17088, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1064)
    subscript_call_result_17090 = invoke(stypy.reporting.localization.Localization(__file__, 1064, 29), getitem___17089, (slice_17086, order_17087))
    
    # Assigning a type to the variable 'tuple_assignment_14133' (line 1064)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1064, 12), 'tuple_assignment_14133', subscript_call_result_17090)
    
    # Assigning a Name to a Name (line 1064):
    # Getting the type of 'tuple_assignment_14132' (line 1064)
    tuple_assignment_14132_17091 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1064, 12), 'tuple_assignment_14132')
    # Assigning a type to the variable 'w' (line 1064)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1064, 12), 'w', tuple_assignment_14132_17091)
    
    # Assigning a Name to a Name (line 1064):
    # Getting the type of 'tuple_assignment_14133' (line 1064)
    tuple_assignment_14133_17092 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1064, 12), 'tuple_assignment_14133')
    # Assigning a type to the variable 'v' (line 1064)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1064, 15), 'v', tuple_assignment_14133_17092)
    # SSA branch for the else part of an if statement (line 1057)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Subscript to a Name (line 1066):
    
    # Assigning a Subscript to a Name (line 1066):
    
    # Obtaining the type of the subscript
    slice_17093 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 1066, 16), None, None, None)
    # Getting the type of 'm' (line 1066)
    m_17094 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1066, 22), 'm')
    slice_17095 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 1066, 16), None, m_17094, None)
    # Getting the type of 'v' (line 1066)
    v_17096 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1066, 16), 'v')
    # Obtaining the member '__getitem__' of a type (line 1066)
    getitem___17097 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1066, 16), v_17096, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1066)
    subscript_call_result_17098 = invoke(stypy.reporting.localization.Localization(__file__, 1066, 16), getitem___17097, (slice_17093, slice_17095))
    
    # Assigning a type to the variable 'v' (line 1066)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1066, 12), 'v', subscript_call_result_17098)
    # SSA join for if statement (line 1057)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Obtaining an instance of the builtin type 'tuple' (line 1067)
    tuple_17099 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1067, 15), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 1067)
    # Adding element type (line 1067)
    # Getting the type of 'w' (line 1067)
    w_17100 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1067, 15), 'w')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1067, 15), tuple_17099, w_17100)
    # Adding element type (line 1067)
    # Getting the type of 'v' (line 1067)
    v_17101 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1067, 18), 'v')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1067, 15), tuple_17099, v_17101)
    
    # Assigning a type to the variable 'stypy_return_type' (line 1067)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1067, 8), 'stypy_return_type', tuple_17099)
    # SSA join for if statement (line 1053)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'eigh_tridiagonal(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'eigh_tridiagonal' in the type store
    # Getting the type of 'stypy_return_type' (line 919)
    stypy_return_type_17102 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 919, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_17102)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'eigh_tridiagonal'
    return stypy_return_type_17102

# Assigning a type to the variable 'eigh_tridiagonal' (line 919)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 919, 0), 'eigh_tridiagonal', eigh_tridiagonal)

@norecursion
def _check_info(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    str_17103 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1070, 39), 'str', 'did not converge (LAPACK info=%d)')
    defaults = [str_17103]
    # Create a new context for function '_check_info'
    module_type_store = module_type_store.open_function_context('_check_info', 1070, 0, False)
    
    # Passed parameters checking function
    _check_info.stypy_localization = localization
    _check_info.stypy_type_of_self = None
    _check_info.stypy_type_store = module_type_store
    _check_info.stypy_function_name = '_check_info'
    _check_info.stypy_param_names_list = ['info', 'driver', 'positive']
    _check_info.stypy_varargs_param_name = None
    _check_info.stypy_kwargs_param_name = None
    _check_info.stypy_call_defaults = defaults
    _check_info.stypy_call_varargs = varargs
    _check_info.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_check_info', ['info', 'driver', 'positive'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_check_info', localization, ['info', 'driver', 'positive'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_check_info(...)' code ##################

    str_17104 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1071, 4), 'str', 'Check info return value.')
    
    
    # Getting the type of 'info' (line 1072)
    info_17105 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1072, 7), 'info')
    int_17106 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1072, 14), 'int')
    # Applying the binary operator '<' (line 1072)
    result_lt_17107 = python_operator(stypy.reporting.localization.Localization(__file__, 1072, 7), '<', info_17105, int_17106)
    
    # Testing the type of an if condition (line 1072)
    if_condition_17108 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1072, 4), result_lt_17107)
    # Assigning a type to the variable 'if_condition_17108' (line 1072)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1072, 4), 'if_condition_17108', if_condition_17108)
    # SSA begins for if statement (line 1072)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 1073)
    # Processing the call arguments (line 1073)
    str_17110 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1073, 25), 'str', 'illegal value in argument %d of internal %s')
    
    # Obtaining an instance of the builtin type 'tuple' (line 1074)
    tuple_17111 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1074, 28), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 1074)
    # Adding element type (line 1074)
    
    # Getting the type of 'info' (line 1074)
    info_17112 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1074, 29), 'info', False)
    # Applying the 'usub' unary operator (line 1074)
    result___neg___17113 = python_operator(stypy.reporting.localization.Localization(__file__, 1074, 28), 'usub', info_17112)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1074, 28), tuple_17111, result___neg___17113)
    # Adding element type (line 1074)
    # Getting the type of 'driver' (line 1074)
    driver_17114 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1074, 35), 'driver', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1074, 28), tuple_17111, driver_17114)
    
    # Applying the binary operator '%' (line 1073)
    result_mod_17115 = python_operator(stypy.reporting.localization.Localization(__file__, 1073, 25), '%', str_17110, tuple_17111)
    
    # Processing the call keyword arguments (line 1073)
    kwargs_17116 = {}
    # Getting the type of 'ValueError' (line 1073)
    ValueError_17109 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1073, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 1073)
    ValueError_call_result_17117 = invoke(stypy.reporting.localization.Localization(__file__, 1073, 14), ValueError_17109, *[result_mod_17115], **kwargs_17116)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 1073, 8), ValueError_call_result_17117, 'raise parameter', BaseException)
    # SSA join for if statement (line 1072)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'info' (line 1075)
    info_17118 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1075, 7), 'info')
    int_17119 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1075, 14), 'int')
    # Applying the binary operator '>' (line 1075)
    result_gt_17120 = python_operator(stypy.reporting.localization.Localization(__file__, 1075, 7), '>', info_17118, int_17119)
    
    # Getting the type of 'positive' (line 1075)
    positive_17121 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1075, 20), 'positive')
    # Applying the binary operator 'and' (line 1075)
    result_and_keyword_17122 = python_operator(stypy.reporting.localization.Localization(__file__, 1075, 7), 'and', result_gt_17120, positive_17121)
    
    # Testing the type of an if condition (line 1075)
    if_condition_17123 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1075, 4), result_and_keyword_17122)
    # Assigning a type to the variable 'if_condition_17123' (line 1075)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1075, 4), 'if_condition_17123', if_condition_17123)
    # SSA begins for if statement (line 1075)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to LinAlgError(...): (line 1076)
    # Processing the call arguments (line 1076)
    str_17125 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1076, 27), 'str', '%s ')
    # Getting the type of 'positive' (line 1076)
    positive_17126 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1076, 35), 'positive', False)
    # Applying the binary operator '+' (line 1076)
    result_add_17127 = python_operator(stypy.reporting.localization.Localization(__file__, 1076, 27), '+', str_17125, positive_17126)
    
    
    # Obtaining an instance of the builtin type 'tuple' (line 1076)
    tuple_17128 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1076, 48), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 1076)
    # Adding element type (line 1076)
    # Getting the type of 'driver' (line 1076)
    driver_17129 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1076, 48), 'driver', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1076, 48), tuple_17128, driver_17129)
    # Adding element type (line 1076)
    # Getting the type of 'info' (line 1076)
    info_17130 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1076, 56), 'info', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1076, 48), tuple_17128, info_17130)
    
    # Applying the binary operator '%' (line 1076)
    result_mod_17131 = python_operator(stypy.reporting.localization.Localization(__file__, 1076, 26), '%', result_add_17127, tuple_17128)
    
    # Processing the call keyword arguments (line 1076)
    kwargs_17132 = {}
    # Getting the type of 'LinAlgError' (line 1076)
    LinAlgError_17124 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1076, 14), 'LinAlgError', False)
    # Calling LinAlgError(args, kwargs) (line 1076)
    LinAlgError_call_result_17133 = invoke(stypy.reporting.localization.Localization(__file__, 1076, 14), LinAlgError_17124, *[result_mod_17131], **kwargs_17132)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 1076, 8), LinAlgError_call_result_17133, 'raise parameter', BaseException)
    # SSA join for if statement (line 1075)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of '_check_info(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_check_info' in the type store
    # Getting the type of 'stypy_return_type' (line 1070)
    stypy_return_type_17134 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1070, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_17134)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_check_info'
    return stypy_return_type_17134

# Assigning a type to the variable '_check_info' (line 1070)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1070, 0), '_check_info', _check_info)

@norecursion
def hessenberg(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'False' (line 1079)
    False_17135 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1079, 25), 'False')
    # Getting the type of 'False' (line 1079)
    False_17136 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1079, 44), 'False')
    # Getting the type of 'True' (line 1079)
    True_17137 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1079, 64), 'True')
    defaults = [False_17135, False_17136, True_17137]
    # Create a new context for function 'hessenberg'
    module_type_store = module_type_store.open_function_context('hessenberg', 1079, 0, False)
    
    # Passed parameters checking function
    hessenberg.stypy_localization = localization
    hessenberg.stypy_type_of_self = None
    hessenberg.stypy_type_store = module_type_store
    hessenberg.stypy_function_name = 'hessenberg'
    hessenberg.stypy_param_names_list = ['a', 'calc_q', 'overwrite_a', 'check_finite']
    hessenberg.stypy_varargs_param_name = None
    hessenberg.stypy_kwargs_param_name = None
    hessenberg.stypy_call_defaults = defaults
    hessenberg.stypy_call_varargs = varargs
    hessenberg.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'hessenberg', ['a', 'calc_q', 'overwrite_a', 'check_finite'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'hessenberg', localization, ['a', 'calc_q', 'overwrite_a', 'check_finite'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'hessenberg(...)' code ##################

    str_17138 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1112, (-1)), 'str', '\n    Compute Hessenberg form of a matrix.\n\n    The Hessenberg decomposition is::\n\n        A = Q H Q^H\n\n    where `Q` is unitary/orthogonal and `H` has only zero elements below\n    the first sub-diagonal.\n\n    Parameters\n    ----------\n    a : (M, M) array_like\n        Matrix to bring into Hessenberg form.\n    calc_q : bool, optional\n        Whether to compute the transformation matrix.  Default is False.\n    overwrite_a : bool, optional\n        Whether to overwrite `a`; may improve performance.\n        Default is False.\n    check_finite : bool, optional\n        Whether to check that the input matrix contains only finite numbers.\n        Disabling may give a performance gain, but may result in problems\n        (crashes, non-termination) if the inputs do contain infinities or NaNs.\n\n    Returns\n    -------\n    H : (M, M) ndarray\n        Hessenberg form of `a`.\n    Q : (M, M) ndarray\n        Unitary/orthogonal similarity transformation matrix ``A = Q H Q^H``.\n        Only returned if ``calc_q=True``.\n\n    ')
    
    # Assigning a Call to a Name (line 1113):
    
    # Assigning a Call to a Name (line 1113):
    
    # Call to _asarray_validated(...): (line 1113)
    # Processing the call arguments (line 1113)
    # Getting the type of 'a' (line 1113)
    a_17140 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1113, 28), 'a', False)
    # Processing the call keyword arguments (line 1113)
    # Getting the type of 'check_finite' (line 1113)
    check_finite_17141 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1113, 44), 'check_finite', False)
    keyword_17142 = check_finite_17141
    kwargs_17143 = {'check_finite': keyword_17142}
    # Getting the type of '_asarray_validated' (line 1113)
    _asarray_validated_17139 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1113, 9), '_asarray_validated', False)
    # Calling _asarray_validated(args, kwargs) (line 1113)
    _asarray_validated_call_result_17144 = invoke(stypy.reporting.localization.Localization(__file__, 1113, 9), _asarray_validated_17139, *[a_17140], **kwargs_17143)
    
    # Assigning a type to the variable 'a1' (line 1113)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1113, 4), 'a1', _asarray_validated_call_result_17144)
    
    
    # Evaluating a boolean operation
    
    
    # Call to len(...): (line 1114)
    # Processing the call arguments (line 1114)
    # Getting the type of 'a1' (line 1114)
    a1_17146 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1114, 11), 'a1', False)
    # Obtaining the member 'shape' of a type (line 1114)
    shape_17147 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1114, 11), a1_17146, 'shape')
    # Processing the call keyword arguments (line 1114)
    kwargs_17148 = {}
    # Getting the type of 'len' (line 1114)
    len_17145 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1114, 7), 'len', False)
    # Calling len(args, kwargs) (line 1114)
    len_call_result_17149 = invoke(stypy.reporting.localization.Localization(__file__, 1114, 7), len_17145, *[shape_17147], **kwargs_17148)
    
    int_17150 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1114, 24), 'int')
    # Applying the binary operator '!=' (line 1114)
    result_ne_17151 = python_operator(stypy.reporting.localization.Localization(__file__, 1114, 7), '!=', len_call_result_17149, int_17150)
    
    
    
    # Obtaining the type of the subscript
    int_17152 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1114, 39), 'int')
    # Getting the type of 'a1' (line 1114)
    a1_17153 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1114, 30), 'a1')
    # Obtaining the member 'shape' of a type (line 1114)
    shape_17154 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1114, 30), a1_17153, 'shape')
    # Obtaining the member '__getitem__' of a type (line 1114)
    getitem___17155 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1114, 30), shape_17154, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1114)
    subscript_call_result_17156 = invoke(stypy.reporting.localization.Localization(__file__, 1114, 30), getitem___17155, int_17152)
    
    
    # Obtaining the type of the subscript
    int_17157 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1114, 54), 'int')
    # Getting the type of 'a1' (line 1114)
    a1_17158 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1114, 45), 'a1')
    # Obtaining the member 'shape' of a type (line 1114)
    shape_17159 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1114, 45), a1_17158, 'shape')
    # Obtaining the member '__getitem__' of a type (line 1114)
    getitem___17160 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1114, 45), shape_17159, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1114)
    subscript_call_result_17161 = invoke(stypy.reporting.localization.Localization(__file__, 1114, 45), getitem___17160, int_17157)
    
    # Applying the binary operator '!=' (line 1114)
    result_ne_17162 = python_operator(stypy.reporting.localization.Localization(__file__, 1114, 30), '!=', subscript_call_result_17156, subscript_call_result_17161)
    
    # Applying the binary operator 'or' (line 1114)
    result_or_keyword_17163 = python_operator(stypy.reporting.localization.Localization(__file__, 1114, 7), 'or', result_ne_17151, result_ne_17162)
    
    # Testing the type of an if condition (line 1114)
    if_condition_17164 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1114, 4), result_or_keyword_17163)
    # Assigning a type to the variable 'if_condition_17164' (line 1114)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1114, 4), 'if_condition_17164', if_condition_17164)
    # SSA begins for if statement (line 1114)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 1115)
    # Processing the call arguments (line 1115)
    str_17166 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1115, 25), 'str', 'expected square matrix')
    # Processing the call keyword arguments (line 1115)
    kwargs_17167 = {}
    # Getting the type of 'ValueError' (line 1115)
    ValueError_17165 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1115, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 1115)
    ValueError_call_result_17168 = invoke(stypy.reporting.localization.Localization(__file__, 1115, 14), ValueError_17165, *[str_17166], **kwargs_17167)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 1115, 8), ValueError_call_result_17168, 'raise parameter', BaseException)
    # SSA join for if statement (line 1114)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a BoolOp to a Name (line 1116):
    
    # Assigning a BoolOp to a Name (line 1116):
    
    # Evaluating a boolean operation
    # Getting the type of 'overwrite_a' (line 1116)
    overwrite_a_17169 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1116, 18), 'overwrite_a')
    
    # Call to _datacopied(...): (line 1116)
    # Processing the call arguments (line 1116)
    # Getting the type of 'a1' (line 1116)
    a1_17171 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1116, 46), 'a1', False)
    # Getting the type of 'a' (line 1116)
    a_17172 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1116, 50), 'a', False)
    # Processing the call keyword arguments (line 1116)
    kwargs_17173 = {}
    # Getting the type of '_datacopied' (line 1116)
    _datacopied_17170 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1116, 34), '_datacopied', False)
    # Calling _datacopied(args, kwargs) (line 1116)
    _datacopied_call_result_17174 = invoke(stypy.reporting.localization.Localization(__file__, 1116, 34), _datacopied_17170, *[a1_17171, a_17172], **kwargs_17173)
    
    # Applying the binary operator 'or' (line 1116)
    result_or_keyword_17175 = python_operator(stypy.reporting.localization.Localization(__file__, 1116, 18), 'or', overwrite_a_17169, _datacopied_call_result_17174)
    
    # Assigning a type to the variable 'overwrite_a' (line 1116)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1116, 4), 'overwrite_a', result_or_keyword_17175)
    
    
    
    # Obtaining the type of the subscript
    int_17176 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1119, 16), 'int')
    # Getting the type of 'a1' (line 1119)
    a1_17177 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1119, 7), 'a1')
    # Obtaining the member 'shape' of a type (line 1119)
    shape_17178 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1119, 7), a1_17177, 'shape')
    # Obtaining the member '__getitem__' of a type (line 1119)
    getitem___17179 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1119, 7), shape_17178, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1119)
    subscript_call_result_17180 = invoke(stypy.reporting.localization.Localization(__file__, 1119, 7), getitem___17179, int_17176)
    
    int_17181 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1119, 22), 'int')
    # Applying the binary operator '<=' (line 1119)
    result_le_17182 = python_operator(stypy.reporting.localization.Localization(__file__, 1119, 7), '<=', subscript_call_result_17180, int_17181)
    
    # Testing the type of an if condition (line 1119)
    if_condition_17183 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1119, 4), result_le_17182)
    # Assigning a type to the variable 'if_condition_17183' (line 1119)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1119, 4), 'if_condition_17183', if_condition_17183)
    # SSA begins for if statement (line 1119)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Getting the type of 'calc_q' (line 1120)
    calc_q_17184 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1120, 11), 'calc_q')
    # Testing the type of an if condition (line 1120)
    if_condition_17185 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1120, 8), calc_q_17184)
    # Assigning a type to the variable 'if_condition_17185' (line 1120)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1120, 8), 'if_condition_17185', if_condition_17185)
    # SSA begins for if statement (line 1120)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Obtaining an instance of the builtin type 'tuple' (line 1121)
    tuple_17186 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1121, 19), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 1121)
    # Adding element type (line 1121)
    # Getting the type of 'a1' (line 1121)
    a1_17187 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1121, 19), 'a1')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1121, 19), tuple_17186, a1_17187)
    # Adding element type (line 1121)
    
    # Call to eye(...): (line 1121)
    # Processing the call arguments (line 1121)
    
    # Obtaining the type of the subscript
    int_17190 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1121, 42), 'int')
    # Getting the type of 'a1' (line 1121)
    a1_17191 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1121, 33), 'a1', False)
    # Obtaining the member 'shape' of a type (line 1121)
    shape_17192 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1121, 33), a1_17191, 'shape')
    # Obtaining the member '__getitem__' of a type (line 1121)
    getitem___17193 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1121, 33), shape_17192, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1121)
    subscript_call_result_17194 = invoke(stypy.reporting.localization.Localization(__file__, 1121, 33), getitem___17193, int_17190)
    
    # Processing the call keyword arguments (line 1121)
    kwargs_17195 = {}
    # Getting the type of 'numpy' (line 1121)
    numpy_17188 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1121, 23), 'numpy', False)
    # Obtaining the member 'eye' of a type (line 1121)
    eye_17189 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1121, 23), numpy_17188, 'eye')
    # Calling eye(args, kwargs) (line 1121)
    eye_call_result_17196 = invoke(stypy.reporting.localization.Localization(__file__, 1121, 23), eye_17189, *[subscript_call_result_17194], **kwargs_17195)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1121, 19), tuple_17186, eye_call_result_17196)
    
    # Assigning a type to the variable 'stypy_return_type' (line 1121)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1121, 12), 'stypy_return_type', tuple_17186)
    # SSA join for if statement (line 1120)
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'a1' (line 1122)
    a1_17197 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1122, 15), 'a1')
    # Assigning a type to the variable 'stypy_return_type' (line 1122)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1122, 8), 'stypy_return_type', a1_17197)
    # SSA join for if statement (line 1119)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Tuple (line 1124):
    
    # Assigning a Subscript to a Name (line 1124):
    
    # Obtaining the type of the subscript
    int_17198 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1124, 4), 'int')
    
    # Call to get_lapack_funcs(...): (line 1124)
    # Processing the call arguments (line 1124)
    
    # Obtaining an instance of the builtin type 'tuple' (line 1124)
    tuple_17200 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1124, 50), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 1124)
    # Adding element type (line 1124)
    str_17201 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1124, 50), 'str', 'gehrd')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1124, 50), tuple_17200, str_17201)
    # Adding element type (line 1124)
    str_17202 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1124, 59), 'str', 'gebal')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1124, 50), tuple_17200, str_17202)
    # Adding element type (line 1124)
    str_17203 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1125, 50), 'str', 'gehrd_lwork')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1124, 50), tuple_17200, str_17203)
    
    
    # Obtaining an instance of the builtin type 'tuple' (line 1125)
    tuple_17204 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1125, 67), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 1125)
    # Adding element type (line 1125)
    # Getting the type of 'a1' (line 1125)
    a1_17205 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1125, 67), 'a1', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1125, 67), tuple_17204, a1_17205)
    
    # Processing the call keyword arguments (line 1124)
    kwargs_17206 = {}
    # Getting the type of 'get_lapack_funcs' (line 1124)
    get_lapack_funcs_17199 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1124, 32), 'get_lapack_funcs', False)
    # Calling get_lapack_funcs(args, kwargs) (line 1124)
    get_lapack_funcs_call_result_17207 = invoke(stypy.reporting.localization.Localization(__file__, 1124, 32), get_lapack_funcs_17199, *[tuple_17200, tuple_17204], **kwargs_17206)
    
    # Obtaining the member '__getitem__' of a type (line 1124)
    getitem___17208 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1124, 4), get_lapack_funcs_call_result_17207, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1124)
    subscript_call_result_17209 = invoke(stypy.reporting.localization.Localization(__file__, 1124, 4), getitem___17208, int_17198)
    
    # Assigning a type to the variable 'tuple_var_assignment_14134' (line 1124)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1124, 4), 'tuple_var_assignment_14134', subscript_call_result_17209)
    
    # Assigning a Subscript to a Name (line 1124):
    
    # Obtaining the type of the subscript
    int_17210 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1124, 4), 'int')
    
    # Call to get_lapack_funcs(...): (line 1124)
    # Processing the call arguments (line 1124)
    
    # Obtaining an instance of the builtin type 'tuple' (line 1124)
    tuple_17212 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1124, 50), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 1124)
    # Adding element type (line 1124)
    str_17213 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1124, 50), 'str', 'gehrd')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1124, 50), tuple_17212, str_17213)
    # Adding element type (line 1124)
    str_17214 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1124, 59), 'str', 'gebal')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1124, 50), tuple_17212, str_17214)
    # Adding element type (line 1124)
    str_17215 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1125, 50), 'str', 'gehrd_lwork')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1124, 50), tuple_17212, str_17215)
    
    
    # Obtaining an instance of the builtin type 'tuple' (line 1125)
    tuple_17216 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1125, 67), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 1125)
    # Adding element type (line 1125)
    # Getting the type of 'a1' (line 1125)
    a1_17217 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1125, 67), 'a1', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1125, 67), tuple_17216, a1_17217)
    
    # Processing the call keyword arguments (line 1124)
    kwargs_17218 = {}
    # Getting the type of 'get_lapack_funcs' (line 1124)
    get_lapack_funcs_17211 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1124, 32), 'get_lapack_funcs', False)
    # Calling get_lapack_funcs(args, kwargs) (line 1124)
    get_lapack_funcs_call_result_17219 = invoke(stypy.reporting.localization.Localization(__file__, 1124, 32), get_lapack_funcs_17211, *[tuple_17212, tuple_17216], **kwargs_17218)
    
    # Obtaining the member '__getitem__' of a type (line 1124)
    getitem___17220 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1124, 4), get_lapack_funcs_call_result_17219, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1124)
    subscript_call_result_17221 = invoke(stypy.reporting.localization.Localization(__file__, 1124, 4), getitem___17220, int_17210)
    
    # Assigning a type to the variable 'tuple_var_assignment_14135' (line 1124)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1124, 4), 'tuple_var_assignment_14135', subscript_call_result_17221)
    
    # Assigning a Subscript to a Name (line 1124):
    
    # Obtaining the type of the subscript
    int_17222 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1124, 4), 'int')
    
    # Call to get_lapack_funcs(...): (line 1124)
    # Processing the call arguments (line 1124)
    
    # Obtaining an instance of the builtin type 'tuple' (line 1124)
    tuple_17224 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1124, 50), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 1124)
    # Adding element type (line 1124)
    str_17225 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1124, 50), 'str', 'gehrd')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1124, 50), tuple_17224, str_17225)
    # Adding element type (line 1124)
    str_17226 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1124, 59), 'str', 'gebal')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1124, 50), tuple_17224, str_17226)
    # Adding element type (line 1124)
    str_17227 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1125, 50), 'str', 'gehrd_lwork')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1124, 50), tuple_17224, str_17227)
    
    
    # Obtaining an instance of the builtin type 'tuple' (line 1125)
    tuple_17228 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1125, 67), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 1125)
    # Adding element type (line 1125)
    # Getting the type of 'a1' (line 1125)
    a1_17229 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1125, 67), 'a1', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1125, 67), tuple_17228, a1_17229)
    
    # Processing the call keyword arguments (line 1124)
    kwargs_17230 = {}
    # Getting the type of 'get_lapack_funcs' (line 1124)
    get_lapack_funcs_17223 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1124, 32), 'get_lapack_funcs', False)
    # Calling get_lapack_funcs(args, kwargs) (line 1124)
    get_lapack_funcs_call_result_17231 = invoke(stypy.reporting.localization.Localization(__file__, 1124, 32), get_lapack_funcs_17223, *[tuple_17224, tuple_17228], **kwargs_17230)
    
    # Obtaining the member '__getitem__' of a type (line 1124)
    getitem___17232 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1124, 4), get_lapack_funcs_call_result_17231, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1124)
    subscript_call_result_17233 = invoke(stypy.reporting.localization.Localization(__file__, 1124, 4), getitem___17232, int_17222)
    
    # Assigning a type to the variable 'tuple_var_assignment_14136' (line 1124)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1124, 4), 'tuple_var_assignment_14136', subscript_call_result_17233)
    
    # Assigning a Name to a Name (line 1124):
    # Getting the type of 'tuple_var_assignment_14134' (line 1124)
    tuple_var_assignment_14134_17234 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1124, 4), 'tuple_var_assignment_14134')
    # Assigning a type to the variable 'gehrd' (line 1124)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1124, 4), 'gehrd', tuple_var_assignment_14134_17234)
    
    # Assigning a Name to a Name (line 1124):
    # Getting the type of 'tuple_var_assignment_14135' (line 1124)
    tuple_var_assignment_14135_17235 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1124, 4), 'tuple_var_assignment_14135')
    # Assigning a type to the variable 'gebal' (line 1124)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1124, 11), 'gebal', tuple_var_assignment_14135_17235)
    
    # Assigning a Name to a Name (line 1124):
    # Getting the type of 'tuple_var_assignment_14136' (line 1124)
    tuple_var_assignment_14136_17236 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1124, 4), 'tuple_var_assignment_14136')
    # Assigning a type to the variable 'gehrd_lwork' (line 1124)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1124, 18), 'gehrd_lwork', tuple_var_assignment_14136_17236)
    
    # Assigning a Call to a Tuple (line 1126):
    
    # Assigning a Subscript to a Name (line 1126):
    
    # Obtaining the type of the subscript
    int_17237 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1126, 4), 'int')
    
    # Call to gebal(...): (line 1126)
    # Processing the call arguments (line 1126)
    # Getting the type of 'a1' (line 1126)
    a1_17239 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1126, 39), 'a1', False)
    # Processing the call keyword arguments (line 1126)
    int_17240 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1126, 51), 'int')
    keyword_17241 = int_17240
    # Getting the type of 'overwrite_a' (line 1126)
    overwrite_a_17242 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1126, 66), 'overwrite_a', False)
    keyword_17243 = overwrite_a_17242
    kwargs_17244 = {'overwrite_a': keyword_17243, 'permute': keyword_17241}
    # Getting the type of 'gebal' (line 1126)
    gebal_17238 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1126, 33), 'gebal', False)
    # Calling gebal(args, kwargs) (line 1126)
    gebal_call_result_17245 = invoke(stypy.reporting.localization.Localization(__file__, 1126, 33), gebal_17238, *[a1_17239], **kwargs_17244)
    
    # Obtaining the member '__getitem__' of a type (line 1126)
    getitem___17246 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1126, 4), gebal_call_result_17245, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1126)
    subscript_call_result_17247 = invoke(stypy.reporting.localization.Localization(__file__, 1126, 4), getitem___17246, int_17237)
    
    # Assigning a type to the variable 'tuple_var_assignment_14137' (line 1126)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1126, 4), 'tuple_var_assignment_14137', subscript_call_result_17247)
    
    # Assigning a Subscript to a Name (line 1126):
    
    # Obtaining the type of the subscript
    int_17248 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1126, 4), 'int')
    
    # Call to gebal(...): (line 1126)
    # Processing the call arguments (line 1126)
    # Getting the type of 'a1' (line 1126)
    a1_17250 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1126, 39), 'a1', False)
    # Processing the call keyword arguments (line 1126)
    int_17251 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1126, 51), 'int')
    keyword_17252 = int_17251
    # Getting the type of 'overwrite_a' (line 1126)
    overwrite_a_17253 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1126, 66), 'overwrite_a', False)
    keyword_17254 = overwrite_a_17253
    kwargs_17255 = {'overwrite_a': keyword_17254, 'permute': keyword_17252}
    # Getting the type of 'gebal' (line 1126)
    gebal_17249 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1126, 33), 'gebal', False)
    # Calling gebal(args, kwargs) (line 1126)
    gebal_call_result_17256 = invoke(stypy.reporting.localization.Localization(__file__, 1126, 33), gebal_17249, *[a1_17250], **kwargs_17255)
    
    # Obtaining the member '__getitem__' of a type (line 1126)
    getitem___17257 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1126, 4), gebal_call_result_17256, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1126)
    subscript_call_result_17258 = invoke(stypy.reporting.localization.Localization(__file__, 1126, 4), getitem___17257, int_17248)
    
    # Assigning a type to the variable 'tuple_var_assignment_14138' (line 1126)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1126, 4), 'tuple_var_assignment_14138', subscript_call_result_17258)
    
    # Assigning a Subscript to a Name (line 1126):
    
    # Obtaining the type of the subscript
    int_17259 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1126, 4), 'int')
    
    # Call to gebal(...): (line 1126)
    # Processing the call arguments (line 1126)
    # Getting the type of 'a1' (line 1126)
    a1_17261 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1126, 39), 'a1', False)
    # Processing the call keyword arguments (line 1126)
    int_17262 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1126, 51), 'int')
    keyword_17263 = int_17262
    # Getting the type of 'overwrite_a' (line 1126)
    overwrite_a_17264 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1126, 66), 'overwrite_a', False)
    keyword_17265 = overwrite_a_17264
    kwargs_17266 = {'overwrite_a': keyword_17265, 'permute': keyword_17263}
    # Getting the type of 'gebal' (line 1126)
    gebal_17260 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1126, 33), 'gebal', False)
    # Calling gebal(args, kwargs) (line 1126)
    gebal_call_result_17267 = invoke(stypy.reporting.localization.Localization(__file__, 1126, 33), gebal_17260, *[a1_17261], **kwargs_17266)
    
    # Obtaining the member '__getitem__' of a type (line 1126)
    getitem___17268 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1126, 4), gebal_call_result_17267, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1126)
    subscript_call_result_17269 = invoke(stypy.reporting.localization.Localization(__file__, 1126, 4), getitem___17268, int_17259)
    
    # Assigning a type to the variable 'tuple_var_assignment_14139' (line 1126)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1126, 4), 'tuple_var_assignment_14139', subscript_call_result_17269)
    
    # Assigning a Subscript to a Name (line 1126):
    
    # Obtaining the type of the subscript
    int_17270 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1126, 4), 'int')
    
    # Call to gebal(...): (line 1126)
    # Processing the call arguments (line 1126)
    # Getting the type of 'a1' (line 1126)
    a1_17272 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1126, 39), 'a1', False)
    # Processing the call keyword arguments (line 1126)
    int_17273 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1126, 51), 'int')
    keyword_17274 = int_17273
    # Getting the type of 'overwrite_a' (line 1126)
    overwrite_a_17275 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1126, 66), 'overwrite_a', False)
    keyword_17276 = overwrite_a_17275
    kwargs_17277 = {'overwrite_a': keyword_17276, 'permute': keyword_17274}
    # Getting the type of 'gebal' (line 1126)
    gebal_17271 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1126, 33), 'gebal', False)
    # Calling gebal(args, kwargs) (line 1126)
    gebal_call_result_17278 = invoke(stypy.reporting.localization.Localization(__file__, 1126, 33), gebal_17271, *[a1_17272], **kwargs_17277)
    
    # Obtaining the member '__getitem__' of a type (line 1126)
    getitem___17279 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1126, 4), gebal_call_result_17278, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1126)
    subscript_call_result_17280 = invoke(stypy.reporting.localization.Localization(__file__, 1126, 4), getitem___17279, int_17270)
    
    # Assigning a type to the variable 'tuple_var_assignment_14140' (line 1126)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1126, 4), 'tuple_var_assignment_14140', subscript_call_result_17280)
    
    # Assigning a Subscript to a Name (line 1126):
    
    # Obtaining the type of the subscript
    int_17281 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1126, 4), 'int')
    
    # Call to gebal(...): (line 1126)
    # Processing the call arguments (line 1126)
    # Getting the type of 'a1' (line 1126)
    a1_17283 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1126, 39), 'a1', False)
    # Processing the call keyword arguments (line 1126)
    int_17284 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1126, 51), 'int')
    keyword_17285 = int_17284
    # Getting the type of 'overwrite_a' (line 1126)
    overwrite_a_17286 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1126, 66), 'overwrite_a', False)
    keyword_17287 = overwrite_a_17286
    kwargs_17288 = {'overwrite_a': keyword_17287, 'permute': keyword_17285}
    # Getting the type of 'gebal' (line 1126)
    gebal_17282 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1126, 33), 'gebal', False)
    # Calling gebal(args, kwargs) (line 1126)
    gebal_call_result_17289 = invoke(stypy.reporting.localization.Localization(__file__, 1126, 33), gebal_17282, *[a1_17283], **kwargs_17288)
    
    # Obtaining the member '__getitem__' of a type (line 1126)
    getitem___17290 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1126, 4), gebal_call_result_17289, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1126)
    subscript_call_result_17291 = invoke(stypy.reporting.localization.Localization(__file__, 1126, 4), getitem___17290, int_17281)
    
    # Assigning a type to the variable 'tuple_var_assignment_14141' (line 1126)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1126, 4), 'tuple_var_assignment_14141', subscript_call_result_17291)
    
    # Assigning a Name to a Name (line 1126):
    # Getting the type of 'tuple_var_assignment_14137' (line 1126)
    tuple_var_assignment_14137_17292 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1126, 4), 'tuple_var_assignment_14137')
    # Assigning a type to the variable 'ba' (line 1126)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1126, 4), 'ba', tuple_var_assignment_14137_17292)
    
    # Assigning a Name to a Name (line 1126):
    # Getting the type of 'tuple_var_assignment_14138' (line 1126)
    tuple_var_assignment_14138_17293 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1126, 4), 'tuple_var_assignment_14138')
    # Assigning a type to the variable 'lo' (line 1126)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1126, 8), 'lo', tuple_var_assignment_14138_17293)
    
    # Assigning a Name to a Name (line 1126):
    # Getting the type of 'tuple_var_assignment_14139' (line 1126)
    tuple_var_assignment_14139_17294 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1126, 4), 'tuple_var_assignment_14139')
    # Assigning a type to the variable 'hi' (line 1126)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1126, 12), 'hi', tuple_var_assignment_14139_17294)
    
    # Assigning a Name to a Name (line 1126):
    # Getting the type of 'tuple_var_assignment_14140' (line 1126)
    tuple_var_assignment_14140_17295 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1126, 4), 'tuple_var_assignment_14140')
    # Assigning a type to the variable 'pivscale' (line 1126)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1126, 16), 'pivscale', tuple_var_assignment_14140_17295)
    
    # Assigning a Name to a Name (line 1126):
    # Getting the type of 'tuple_var_assignment_14141' (line 1126)
    tuple_var_assignment_14141_17296 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1126, 4), 'tuple_var_assignment_14141')
    # Assigning a type to the variable 'info' (line 1126)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1126, 26), 'info', tuple_var_assignment_14141_17296)
    
    # Call to _check_info(...): (line 1127)
    # Processing the call arguments (line 1127)
    # Getting the type of 'info' (line 1127)
    info_17298 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1127, 16), 'info', False)
    str_17299 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1127, 22), 'str', 'gebal (hessenberg)')
    # Processing the call keyword arguments (line 1127)
    # Getting the type of 'False' (line 1127)
    False_17300 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1127, 53), 'False', False)
    keyword_17301 = False_17300
    kwargs_17302 = {'positive': keyword_17301}
    # Getting the type of '_check_info' (line 1127)
    _check_info_17297 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1127, 4), '_check_info', False)
    # Calling _check_info(args, kwargs) (line 1127)
    _check_info_call_result_17303 = invoke(stypy.reporting.localization.Localization(__file__, 1127, 4), _check_info_17297, *[info_17298, str_17299], **kwargs_17302)
    
    
    # Assigning a Call to a Name (line 1128):
    
    # Assigning a Call to a Name (line 1128):
    
    # Call to len(...): (line 1128)
    # Processing the call arguments (line 1128)
    # Getting the type of 'a1' (line 1128)
    a1_17305 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1128, 12), 'a1', False)
    # Processing the call keyword arguments (line 1128)
    kwargs_17306 = {}
    # Getting the type of 'len' (line 1128)
    len_17304 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1128, 8), 'len', False)
    # Calling len(args, kwargs) (line 1128)
    len_call_result_17307 = invoke(stypy.reporting.localization.Localization(__file__, 1128, 8), len_17304, *[a1_17305], **kwargs_17306)
    
    # Assigning a type to the variable 'n' (line 1128)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1128, 4), 'n', len_call_result_17307)
    
    # Assigning a Call to a Name (line 1130):
    
    # Assigning a Call to a Name (line 1130):
    
    # Call to _compute_lwork(...): (line 1130)
    # Processing the call arguments (line 1130)
    # Getting the type of 'gehrd_lwork' (line 1130)
    gehrd_lwork_17309 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1130, 27), 'gehrd_lwork', False)
    
    # Obtaining the type of the subscript
    int_17310 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1130, 49), 'int')
    # Getting the type of 'ba' (line 1130)
    ba_17311 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1130, 40), 'ba', False)
    # Obtaining the member 'shape' of a type (line 1130)
    shape_17312 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1130, 40), ba_17311, 'shape')
    # Obtaining the member '__getitem__' of a type (line 1130)
    getitem___17313 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1130, 40), shape_17312, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1130)
    subscript_call_result_17314 = invoke(stypy.reporting.localization.Localization(__file__, 1130, 40), getitem___17313, int_17310)
    
    # Processing the call keyword arguments (line 1130)
    # Getting the type of 'lo' (line 1130)
    lo_17315 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1130, 56), 'lo', False)
    keyword_17316 = lo_17315
    # Getting the type of 'hi' (line 1130)
    hi_17317 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1130, 63), 'hi', False)
    keyword_17318 = hi_17317
    kwargs_17319 = {'lo': keyword_17316, 'hi': keyword_17318}
    # Getting the type of '_compute_lwork' (line 1130)
    _compute_lwork_17308 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1130, 12), '_compute_lwork', False)
    # Calling _compute_lwork(args, kwargs) (line 1130)
    _compute_lwork_call_result_17320 = invoke(stypy.reporting.localization.Localization(__file__, 1130, 12), _compute_lwork_17308, *[gehrd_lwork_17309, subscript_call_result_17314], **kwargs_17319)
    
    # Assigning a type to the variable 'lwork' (line 1130)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1130, 4), 'lwork', _compute_lwork_call_result_17320)
    
    # Assigning a Call to a Tuple (line 1132):
    
    # Assigning a Subscript to a Name (line 1132):
    
    # Obtaining the type of the subscript
    int_17321 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1132, 4), 'int')
    
    # Call to gehrd(...): (line 1132)
    # Processing the call arguments (line 1132)
    # Getting the type of 'ba' (line 1132)
    ba_17323 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1132, 26), 'ba', False)
    # Processing the call keyword arguments (line 1132)
    # Getting the type of 'lo' (line 1132)
    lo_17324 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1132, 33), 'lo', False)
    keyword_17325 = lo_17324
    # Getting the type of 'hi' (line 1132)
    hi_17326 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1132, 40), 'hi', False)
    keyword_17327 = hi_17326
    # Getting the type of 'lwork' (line 1132)
    lwork_17328 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1132, 50), 'lwork', False)
    keyword_17329 = lwork_17328
    int_17330 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1132, 69), 'int')
    keyword_17331 = int_17330
    kwargs_17332 = {'lo': keyword_17325, 'hi': keyword_17327, 'overwrite_a': keyword_17331, 'lwork': keyword_17329}
    # Getting the type of 'gehrd' (line 1132)
    gehrd_17322 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1132, 20), 'gehrd', False)
    # Calling gehrd(args, kwargs) (line 1132)
    gehrd_call_result_17333 = invoke(stypy.reporting.localization.Localization(__file__, 1132, 20), gehrd_17322, *[ba_17323], **kwargs_17332)
    
    # Obtaining the member '__getitem__' of a type (line 1132)
    getitem___17334 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1132, 4), gehrd_call_result_17333, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1132)
    subscript_call_result_17335 = invoke(stypy.reporting.localization.Localization(__file__, 1132, 4), getitem___17334, int_17321)
    
    # Assigning a type to the variable 'tuple_var_assignment_14142' (line 1132)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1132, 4), 'tuple_var_assignment_14142', subscript_call_result_17335)
    
    # Assigning a Subscript to a Name (line 1132):
    
    # Obtaining the type of the subscript
    int_17336 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1132, 4), 'int')
    
    # Call to gehrd(...): (line 1132)
    # Processing the call arguments (line 1132)
    # Getting the type of 'ba' (line 1132)
    ba_17338 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1132, 26), 'ba', False)
    # Processing the call keyword arguments (line 1132)
    # Getting the type of 'lo' (line 1132)
    lo_17339 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1132, 33), 'lo', False)
    keyword_17340 = lo_17339
    # Getting the type of 'hi' (line 1132)
    hi_17341 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1132, 40), 'hi', False)
    keyword_17342 = hi_17341
    # Getting the type of 'lwork' (line 1132)
    lwork_17343 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1132, 50), 'lwork', False)
    keyword_17344 = lwork_17343
    int_17345 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1132, 69), 'int')
    keyword_17346 = int_17345
    kwargs_17347 = {'lo': keyword_17340, 'hi': keyword_17342, 'overwrite_a': keyword_17346, 'lwork': keyword_17344}
    # Getting the type of 'gehrd' (line 1132)
    gehrd_17337 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1132, 20), 'gehrd', False)
    # Calling gehrd(args, kwargs) (line 1132)
    gehrd_call_result_17348 = invoke(stypy.reporting.localization.Localization(__file__, 1132, 20), gehrd_17337, *[ba_17338], **kwargs_17347)
    
    # Obtaining the member '__getitem__' of a type (line 1132)
    getitem___17349 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1132, 4), gehrd_call_result_17348, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1132)
    subscript_call_result_17350 = invoke(stypy.reporting.localization.Localization(__file__, 1132, 4), getitem___17349, int_17336)
    
    # Assigning a type to the variable 'tuple_var_assignment_14143' (line 1132)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1132, 4), 'tuple_var_assignment_14143', subscript_call_result_17350)
    
    # Assigning a Subscript to a Name (line 1132):
    
    # Obtaining the type of the subscript
    int_17351 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1132, 4), 'int')
    
    # Call to gehrd(...): (line 1132)
    # Processing the call arguments (line 1132)
    # Getting the type of 'ba' (line 1132)
    ba_17353 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1132, 26), 'ba', False)
    # Processing the call keyword arguments (line 1132)
    # Getting the type of 'lo' (line 1132)
    lo_17354 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1132, 33), 'lo', False)
    keyword_17355 = lo_17354
    # Getting the type of 'hi' (line 1132)
    hi_17356 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1132, 40), 'hi', False)
    keyword_17357 = hi_17356
    # Getting the type of 'lwork' (line 1132)
    lwork_17358 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1132, 50), 'lwork', False)
    keyword_17359 = lwork_17358
    int_17360 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1132, 69), 'int')
    keyword_17361 = int_17360
    kwargs_17362 = {'lo': keyword_17355, 'hi': keyword_17357, 'overwrite_a': keyword_17361, 'lwork': keyword_17359}
    # Getting the type of 'gehrd' (line 1132)
    gehrd_17352 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1132, 20), 'gehrd', False)
    # Calling gehrd(args, kwargs) (line 1132)
    gehrd_call_result_17363 = invoke(stypy.reporting.localization.Localization(__file__, 1132, 20), gehrd_17352, *[ba_17353], **kwargs_17362)
    
    # Obtaining the member '__getitem__' of a type (line 1132)
    getitem___17364 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1132, 4), gehrd_call_result_17363, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1132)
    subscript_call_result_17365 = invoke(stypy.reporting.localization.Localization(__file__, 1132, 4), getitem___17364, int_17351)
    
    # Assigning a type to the variable 'tuple_var_assignment_14144' (line 1132)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1132, 4), 'tuple_var_assignment_14144', subscript_call_result_17365)
    
    # Assigning a Name to a Name (line 1132):
    # Getting the type of 'tuple_var_assignment_14142' (line 1132)
    tuple_var_assignment_14142_17366 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1132, 4), 'tuple_var_assignment_14142')
    # Assigning a type to the variable 'hq' (line 1132)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1132, 4), 'hq', tuple_var_assignment_14142_17366)
    
    # Assigning a Name to a Name (line 1132):
    # Getting the type of 'tuple_var_assignment_14143' (line 1132)
    tuple_var_assignment_14143_17367 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1132, 4), 'tuple_var_assignment_14143')
    # Assigning a type to the variable 'tau' (line 1132)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1132, 8), 'tau', tuple_var_assignment_14143_17367)
    
    # Assigning a Name to a Name (line 1132):
    # Getting the type of 'tuple_var_assignment_14144' (line 1132)
    tuple_var_assignment_14144_17368 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1132, 4), 'tuple_var_assignment_14144')
    # Assigning a type to the variable 'info' (line 1132)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1132, 13), 'info', tuple_var_assignment_14144_17368)
    
    # Call to _check_info(...): (line 1133)
    # Processing the call arguments (line 1133)
    # Getting the type of 'info' (line 1133)
    info_17370 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1133, 16), 'info', False)
    str_17371 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1133, 22), 'str', 'gehrd (hessenberg)')
    # Processing the call keyword arguments (line 1133)
    # Getting the type of 'False' (line 1133)
    False_17372 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1133, 53), 'False', False)
    keyword_17373 = False_17372
    kwargs_17374 = {'positive': keyword_17373}
    # Getting the type of '_check_info' (line 1133)
    _check_info_17369 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1133, 4), '_check_info', False)
    # Calling _check_info(args, kwargs) (line 1133)
    _check_info_call_result_17375 = invoke(stypy.reporting.localization.Localization(__file__, 1133, 4), _check_info_17369, *[info_17370, str_17371], **kwargs_17374)
    
    
    # Assigning a Call to a Name (line 1134):
    
    # Assigning a Call to a Name (line 1134):
    
    # Call to triu(...): (line 1134)
    # Processing the call arguments (line 1134)
    # Getting the type of 'hq' (line 1134)
    hq_17378 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1134, 19), 'hq', False)
    int_17379 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1134, 23), 'int')
    # Processing the call keyword arguments (line 1134)
    kwargs_17380 = {}
    # Getting the type of 'numpy' (line 1134)
    numpy_17376 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1134, 8), 'numpy', False)
    # Obtaining the member 'triu' of a type (line 1134)
    triu_17377 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1134, 8), numpy_17376, 'triu')
    # Calling triu(args, kwargs) (line 1134)
    triu_call_result_17381 = invoke(stypy.reporting.localization.Localization(__file__, 1134, 8), triu_17377, *[hq_17378, int_17379], **kwargs_17380)
    
    # Assigning a type to the variable 'h' (line 1134)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1134, 4), 'h', triu_call_result_17381)
    
    
    # Getting the type of 'calc_q' (line 1135)
    calc_q_17382 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1135, 11), 'calc_q')
    # Applying the 'not' unary operator (line 1135)
    result_not__17383 = python_operator(stypy.reporting.localization.Localization(__file__, 1135, 7), 'not', calc_q_17382)
    
    # Testing the type of an if condition (line 1135)
    if_condition_17384 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1135, 4), result_not__17383)
    # Assigning a type to the variable 'if_condition_17384' (line 1135)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1135, 4), 'if_condition_17384', if_condition_17384)
    # SSA begins for if statement (line 1135)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # Getting the type of 'h' (line 1136)
    h_17385 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1136, 15), 'h')
    # Assigning a type to the variable 'stypy_return_type' (line 1136)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1136, 8), 'stypy_return_type', h_17385)
    # SSA join for if statement (line 1135)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Tuple (line 1139):
    
    # Assigning a Subscript to a Name (line 1139):
    
    # Obtaining the type of the subscript
    int_17386 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1139, 4), 'int')
    
    # Call to get_lapack_funcs(...): (line 1139)
    # Processing the call arguments (line 1139)
    
    # Obtaining an instance of the builtin type 'tuple' (line 1139)
    tuple_17388 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1139, 43), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 1139)
    # Adding element type (line 1139)
    str_17389 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1139, 43), 'str', 'orghr')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1139, 43), tuple_17388, str_17389)
    # Adding element type (line 1139)
    str_17390 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1139, 52), 'str', 'orghr_lwork')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1139, 43), tuple_17388, str_17390)
    
    
    # Obtaining an instance of the builtin type 'tuple' (line 1139)
    tuple_17391 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1139, 69), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 1139)
    # Adding element type (line 1139)
    # Getting the type of 'a1' (line 1139)
    a1_17392 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1139, 69), 'a1', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1139, 69), tuple_17391, a1_17392)
    
    # Processing the call keyword arguments (line 1139)
    kwargs_17393 = {}
    # Getting the type of 'get_lapack_funcs' (line 1139)
    get_lapack_funcs_17387 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1139, 25), 'get_lapack_funcs', False)
    # Calling get_lapack_funcs(args, kwargs) (line 1139)
    get_lapack_funcs_call_result_17394 = invoke(stypy.reporting.localization.Localization(__file__, 1139, 25), get_lapack_funcs_17387, *[tuple_17388, tuple_17391], **kwargs_17393)
    
    # Obtaining the member '__getitem__' of a type (line 1139)
    getitem___17395 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1139, 4), get_lapack_funcs_call_result_17394, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1139)
    subscript_call_result_17396 = invoke(stypy.reporting.localization.Localization(__file__, 1139, 4), getitem___17395, int_17386)
    
    # Assigning a type to the variable 'tuple_var_assignment_14145' (line 1139)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1139, 4), 'tuple_var_assignment_14145', subscript_call_result_17396)
    
    # Assigning a Subscript to a Name (line 1139):
    
    # Obtaining the type of the subscript
    int_17397 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1139, 4), 'int')
    
    # Call to get_lapack_funcs(...): (line 1139)
    # Processing the call arguments (line 1139)
    
    # Obtaining an instance of the builtin type 'tuple' (line 1139)
    tuple_17399 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1139, 43), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 1139)
    # Adding element type (line 1139)
    str_17400 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1139, 43), 'str', 'orghr')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1139, 43), tuple_17399, str_17400)
    # Adding element type (line 1139)
    str_17401 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1139, 52), 'str', 'orghr_lwork')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1139, 43), tuple_17399, str_17401)
    
    
    # Obtaining an instance of the builtin type 'tuple' (line 1139)
    tuple_17402 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1139, 69), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 1139)
    # Adding element type (line 1139)
    # Getting the type of 'a1' (line 1139)
    a1_17403 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1139, 69), 'a1', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1139, 69), tuple_17402, a1_17403)
    
    # Processing the call keyword arguments (line 1139)
    kwargs_17404 = {}
    # Getting the type of 'get_lapack_funcs' (line 1139)
    get_lapack_funcs_17398 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1139, 25), 'get_lapack_funcs', False)
    # Calling get_lapack_funcs(args, kwargs) (line 1139)
    get_lapack_funcs_call_result_17405 = invoke(stypy.reporting.localization.Localization(__file__, 1139, 25), get_lapack_funcs_17398, *[tuple_17399, tuple_17402], **kwargs_17404)
    
    # Obtaining the member '__getitem__' of a type (line 1139)
    getitem___17406 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1139, 4), get_lapack_funcs_call_result_17405, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1139)
    subscript_call_result_17407 = invoke(stypy.reporting.localization.Localization(__file__, 1139, 4), getitem___17406, int_17397)
    
    # Assigning a type to the variable 'tuple_var_assignment_14146' (line 1139)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1139, 4), 'tuple_var_assignment_14146', subscript_call_result_17407)
    
    # Assigning a Name to a Name (line 1139):
    # Getting the type of 'tuple_var_assignment_14145' (line 1139)
    tuple_var_assignment_14145_17408 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1139, 4), 'tuple_var_assignment_14145')
    # Assigning a type to the variable 'orghr' (line 1139)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1139, 4), 'orghr', tuple_var_assignment_14145_17408)
    
    # Assigning a Name to a Name (line 1139):
    # Getting the type of 'tuple_var_assignment_14146' (line 1139)
    tuple_var_assignment_14146_17409 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1139, 4), 'tuple_var_assignment_14146')
    # Assigning a type to the variable 'orghr_lwork' (line 1139)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1139, 11), 'orghr_lwork', tuple_var_assignment_14146_17409)
    
    # Assigning a Call to a Name (line 1140):
    
    # Assigning a Call to a Name (line 1140):
    
    # Call to _compute_lwork(...): (line 1140)
    # Processing the call arguments (line 1140)
    # Getting the type of 'orghr_lwork' (line 1140)
    orghr_lwork_17411 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1140, 27), 'orghr_lwork', False)
    # Getting the type of 'n' (line 1140)
    n_17412 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1140, 40), 'n', False)
    # Processing the call keyword arguments (line 1140)
    # Getting the type of 'lo' (line 1140)
    lo_17413 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1140, 46), 'lo', False)
    keyword_17414 = lo_17413
    # Getting the type of 'hi' (line 1140)
    hi_17415 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1140, 53), 'hi', False)
    keyword_17416 = hi_17415
    kwargs_17417 = {'lo': keyword_17414, 'hi': keyword_17416}
    # Getting the type of '_compute_lwork' (line 1140)
    _compute_lwork_17410 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1140, 12), '_compute_lwork', False)
    # Calling _compute_lwork(args, kwargs) (line 1140)
    _compute_lwork_call_result_17418 = invoke(stypy.reporting.localization.Localization(__file__, 1140, 12), _compute_lwork_17410, *[orghr_lwork_17411, n_17412], **kwargs_17417)
    
    # Assigning a type to the variable 'lwork' (line 1140)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1140, 4), 'lwork', _compute_lwork_call_result_17418)
    
    # Assigning a Call to a Tuple (line 1142):
    
    # Assigning a Subscript to a Name (line 1142):
    
    # Obtaining the type of the subscript
    int_17419 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1142, 4), 'int')
    
    # Call to orghr(...): (line 1142)
    # Processing the call keyword arguments (line 1142)
    # Getting the type of 'hq' (line 1142)
    hq_17421 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1142, 22), 'hq', False)
    keyword_17422 = hq_17421
    # Getting the type of 'tau' (line 1142)
    tau_17423 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1142, 30), 'tau', False)
    keyword_17424 = tau_17423
    # Getting the type of 'lo' (line 1142)
    lo_17425 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1142, 38), 'lo', False)
    keyword_17426 = lo_17425
    # Getting the type of 'hi' (line 1142)
    hi_17427 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1142, 45), 'hi', False)
    keyword_17428 = hi_17427
    # Getting the type of 'lwork' (line 1142)
    lwork_17429 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1142, 55), 'lwork', False)
    keyword_17430 = lwork_17429
    int_17431 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1142, 74), 'int')
    keyword_17432 = int_17431
    kwargs_17433 = {'a': keyword_17422, 'tau': keyword_17424, 'overwrite_a': keyword_17432, 'lo': keyword_17426, 'hi': keyword_17428, 'lwork': keyword_17430}
    # Getting the type of 'orghr' (line 1142)
    orghr_17420 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1142, 14), 'orghr', False)
    # Calling orghr(args, kwargs) (line 1142)
    orghr_call_result_17434 = invoke(stypy.reporting.localization.Localization(__file__, 1142, 14), orghr_17420, *[], **kwargs_17433)
    
    # Obtaining the member '__getitem__' of a type (line 1142)
    getitem___17435 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1142, 4), orghr_call_result_17434, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1142)
    subscript_call_result_17436 = invoke(stypy.reporting.localization.Localization(__file__, 1142, 4), getitem___17435, int_17419)
    
    # Assigning a type to the variable 'tuple_var_assignment_14147' (line 1142)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1142, 4), 'tuple_var_assignment_14147', subscript_call_result_17436)
    
    # Assigning a Subscript to a Name (line 1142):
    
    # Obtaining the type of the subscript
    int_17437 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1142, 4), 'int')
    
    # Call to orghr(...): (line 1142)
    # Processing the call keyword arguments (line 1142)
    # Getting the type of 'hq' (line 1142)
    hq_17439 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1142, 22), 'hq', False)
    keyword_17440 = hq_17439
    # Getting the type of 'tau' (line 1142)
    tau_17441 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1142, 30), 'tau', False)
    keyword_17442 = tau_17441
    # Getting the type of 'lo' (line 1142)
    lo_17443 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1142, 38), 'lo', False)
    keyword_17444 = lo_17443
    # Getting the type of 'hi' (line 1142)
    hi_17445 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1142, 45), 'hi', False)
    keyword_17446 = hi_17445
    # Getting the type of 'lwork' (line 1142)
    lwork_17447 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1142, 55), 'lwork', False)
    keyword_17448 = lwork_17447
    int_17449 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1142, 74), 'int')
    keyword_17450 = int_17449
    kwargs_17451 = {'a': keyword_17440, 'tau': keyword_17442, 'overwrite_a': keyword_17450, 'lo': keyword_17444, 'hi': keyword_17446, 'lwork': keyword_17448}
    # Getting the type of 'orghr' (line 1142)
    orghr_17438 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1142, 14), 'orghr', False)
    # Calling orghr(args, kwargs) (line 1142)
    orghr_call_result_17452 = invoke(stypy.reporting.localization.Localization(__file__, 1142, 14), orghr_17438, *[], **kwargs_17451)
    
    # Obtaining the member '__getitem__' of a type (line 1142)
    getitem___17453 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1142, 4), orghr_call_result_17452, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1142)
    subscript_call_result_17454 = invoke(stypy.reporting.localization.Localization(__file__, 1142, 4), getitem___17453, int_17437)
    
    # Assigning a type to the variable 'tuple_var_assignment_14148' (line 1142)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1142, 4), 'tuple_var_assignment_14148', subscript_call_result_17454)
    
    # Assigning a Name to a Name (line 1142):
    # Getting the type of 'tuple_var_assignment_14147' (line 1142)
    tuple_var_assignment_14147_17455 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1142, 4), 'tuple_var_assignment_14147')
    # Assigning a type to the variable 'q' (line 1142)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1142, 4), 'q', tuple_var_assignment_14147_17455)
    
    # Assigning a Name to a Name (line 1142):
    # Getting the type of 'tuple_var_assignment_14148' (line 1142)
    tuple_var_assignment_14148_17456 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1142, 4), 'tuple_var_assignment_14148')
    # Assigning a type to the variable 'info' (line 1142)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1142, 7), 'info', tuple_var_assignment_14148_17456)
    
    # Call to _check_info(...): (line 1143)
    # Processing the call arguments (line 1143)
    # Getting the type of 'info' (line 1143)
    info_17458 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1143, 16), 'info', False)
    str_17459 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1143, 22), 'str', 'orghr (hessenberg)')
    # Processing the call keyword arguments (line 1143)
    # Getting the type of 'False' (line 1143)
    False_17460 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1143, 53), 'False', False)
    keyword_17461 = False_17460
    kwargs_17462 = {'positive': keyword_17461}
    # Getting the type of '_check_info' (line 1143)
    _check_info_17457 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1143, 4), '_check_info', False)
    # Calling _check_info(args, kwargs) (line 1143)
    _check_info_call_result_17463 = invoke(stypy.reporting.localization.Localization(__file__, 1143, 4), _check_info_17457, *[info_17458, str_17459], **kwargs_17462)
    
    
    # Obtaining an instance of the builtin type 'tuple' (line 1144)
    tuple_17464 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1144, 11), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 1144)
    # Adding element type (line 1144)
    # Getting the type of 'h' (line 1144)
    h_17465 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1144, 11), 'h')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1144, 11), tuple_17464, h_17465)
    # Adding element type (line 1144)
    # Getting the type of 'q' (line 1144)
    q_17466 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1144, 14), 'q')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1144, 11), tuple_17464, q_17466)
    
    # Assigning a type to the variable 'stypy_return_type' (line 1144)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1144, 4), 'stypy_return_type', tuple_17464)
    
    # ################# End of 'hessenberg(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'hessenberg' in the type store
    # Getting the type of 'stypy_return_type' (line 1079)
    stypy_return_type_17467 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1079, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_17467)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'hessenberg'
    return stypy_return_type_17467

# Assigning a type to the variable 'hessenberg' (line 1079)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1079, 0), 'hessenberg', hessenberg)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
