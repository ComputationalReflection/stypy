
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: '''
2: Low-level LAPACK functions (:mod:`scipy.linalg.lapack`)
3: =======================================================
4: 
5: This module contains low-level functions from the LAPACK library.
6: 
7: The `*gegv` family of routines have been removed from LAPACK 3.6.0
8: and have been deprecated in SciPy 0.17.0. They will be removed in
9: a future release.
10: 
11: .. versionadded:: 0.12.0
12: 
13: .. warning::
14: 
15:    These functions do little to no error checking.
16:    It is possible to cause crashes by mis-using them,
17:    so prefer using the higher-level routines in `scipy.linalg`.
18: 
19: Finding functions
20: -----------------
21: 
22: .. autosummary::
23: 
24:    get_lapack_funcs
25: 
26: All functions
27: -------------
28: 
29: .. autosummary::
30:    :toctree: generated/
31: 
32: 
33:    sgbsv
34:    dgbsv
35:    cgbsv
36:    zgbsv
37: 
38:    sgbtrf
39:    dgbtrf
40:    cgbtrf
41:    zgbtrf
42: 
43:    sgbtrs
44:    dgbtrs
45:    cgbtrs
46:    zgbtrs
47: 
48:    sgebal
49:    dgebal
50:    cgebal
51:    zgebal
52: 
53:    sgees
54:    dgees
55:    cgees
56:    zgees
57: 
58:    sgeev
59:    dgeev
60:    cgeev
61:    zgeev
62: 
63:    sgeev_lwork
64:    dgeev_lwork
65:    cgeev_lwork
66:    zgeev_lwork
67: 
68:    sgegv
69:    dgegv
70:    cgegv
71:    zgegv
72: 
73:    sgehrd
74:    dgehrd
75:    cgehrd
76:    zgehrd
77: 
78:    sgehrd_lwork
79:    dgehrd_lwork
80:    cgehrd_lwork
81:    zgehrd_lwork
82: 
83:    sgelss
84:    dgelss
85:    cgelss
86:    zgelss
87: 
88:    sgelss_lwork
89:    dgelss_lwork
90:    cgelss_lwork
91:    zgelss_lwork
92: 
93:    sgelsd
94:    dgelsd
95:    cgelsd
96:    zgelsd
97: 
98:    sgelsd_lwork
99:    dgelsd_lwork
100:    cgelsd_lwork
101:    zgelsd_lwork
102: 
103:    sgelsy
104:    dgelsy
105:    cgelsy
106:    zgelsy
107: 
108:    sgelsy_lwork
109:    dgelsy_lwork
110:    cgelsy_lwork
111:    zgelsy_lwork
112: 
113:    sgeqp3
114:    dgeqp3
115:    cgeqp3
116:    zgeqp3
117: 
118:    sgeqrf
119:    dgeqrf
120:    cgeqrf
121:    zgeqrf
122: 
123:    sgerqf
124:    dgerqf
125:    cgerqf
126:    zgerqf
127: 
128:    sgesdd
129:    dgesdd
130:    cgesdd
131:    zgesdd
132: 
133:    sgesdd_lwork
134:    dgesdd_lwork
135:    cgesdd_lwork
136:    zgesdd_lwork
137: 
138:    sgesvd
139:    dgesvd
140:    cgesvd
141:    zgesvd
142: 
143:    sgesvd_lwork
144:    dgesvd_lwork
145:    cgesvd_lwork
146:    zgesvd_lwork
147: 
148:    sgesv
149:    dgesv
150:    cgesv
151:    zgesv
152: 
153:    sgesvx
154:    dgesvx
155:    cgesvx
156:    zgesvx
157: 
158:    sgecon
159:    dgecon
160:    cgecon
161:    zgecon
162: 
163:    ssysv
164:    dsysv
165:    csysv
166:    zsysv
167: 
168:    ssysv_lwork
169:    dsysv_lwork
170:    csysv_lwork
171:    zsysv_lwork
172: 
173:    ssysvx
174:    dsysvx
175:    csysvx
176:    zsysvx
177: 
178:    ssysvx_lwork
179:    dsysvx_lwork
180:    csysvx_lwork
181:    zsysvx_lwork
182: 
183:    ssytrd
184:    dsytrd
185: 
186:    ssytrd_lwork
187:    dsytrd_lwork
188: 
189:    chetrd
190:    zhetrd
191: 
192:    chetrd_lwork
193:    zhetrd_lwork
194: 
195:    chesv
196:    zhesv
197: 
198:    chesv_lwork
199:    zhesv_lwork
200: 
201:    chesvx
202:    zhesvx
203: 
204:    chesvx_lwork
205:    zhesvx_lwork
206: 
207:    sgetrf
208:    dgetrf
209:    cgetrf
210:    zgetrf
211: 
212:    sgetri
213:    dgetri
214:    cgetri
215:    zgetri
216: 
217:    sgetri_lwork
218:    dgetri_lwork
219:    cgetri_lwork
220:    zgetri_lwork
221: 
222:    sgetrs
223:    dgetrs
224:    cgetrs
225:    zgetrs
226: 
227:    sgges
228:    dgges
229:    cgges
230:    zgges
231: 
232:    sggev
233:    dggev
234:    cggev
235:    zggev
236: 
237:    chbevd
238:    zhbevd
239: 
240:    chbevx
241:    zhbevx
242: 
243:    cheev
244:    zheev
245: 
246:    cheevd
247:    zheevd
248: 
249:    cheevr
250:    zheevr
251: 
252:    chegv
253:    zhegv
254: 
255:    chegvd
256:    zhegvd
257: 
258:    chegvx
259:    zhegvx
260: 
261:    slarf
262:    dlarf
263:    clarf
264:    zlarf
265: 
266:    slarfg
267:    dlarfg
268:    clarfg
269:    zlarfg
270: 
271:    slartg
272:    dlartg
273:    clartg
274:    zlartg
275: 
276:    slasd4
277:    dlasd4
278: 
279:    slaswp
280:    dlaswp
281:    claswp
282:    zlaswp
283: 
284:    slauum
285:    dlauum
286:    clauum
287:    zlauum
288: 
289:    spbsv
290:    dpbsv
291:    cpbsv
292:    zpbsv
293: 
294:    spbtrf
295:    dpbtrf
296:    cpbtrf
297:    zpbtrf
298: 
299:    spbtrs
300:    dpbtrs
301:    cpbtrs
302:    zpbtrs
303: 
304:    sposv
305:    dposv
306:    cposv
307:    zposv
308: 
309:    sposvx
310:    dposvx
311:    cposvx
312:    zposvx
313: 
314:    spocon
315:    dpocon
316:    cpocon
317:    zpocon
318: 
319:    spotrf
320:    dpotrf
321:    cpotrf
322:    zpotrf
323: 
324:    spotri
325:    dpotri
326:    cpotri
327:    zpotri
328: 
329:    spotrs
330:    dpotrs
331:    cpotrs
332:    zpotrs
333: 
334:    crot
335:    zrot
336: 
337:    strsyl
338:    dtrsyl
339:    ctrsyl
340:    ztrsyl
341: 
342:    strtri
343:    dtrtri
344:    ctrtri
345:    ztrtri
346: 
347:    strtrs
348:    dtrtrs
349:    ctrtrs
350:    ztrtrs
351: 
352:    cunghr
353:    zunghr
354: 
355:    cungqr
356:    zungqr
357: 
358:    cungrq
359:    zungrq
360: 
361:    cunmqr
362:    zunmqr
363: 
364:    sgtsv
365:    dgtsv
366:    cgtsv
367:    zgtsv
368: 
369:    sptsv
370:    dptsv
371:    cptsv
372:    zptsv
373: 
374:    slamch
375:    dlamch
376: 
377:    sorghr
378:    dorghr
379:    sorgqr
380:    dorgqr
381: 
382:    sorgrq
383:    dorgrq
384: 
385:    sormqr
386:    dormqr
387: 
388:    ssbev
389:    dsbev
390: 
391:    ssbevd
392:    dsbevd
393: 
394:    ssbevx
395:    dsbevx
396: 
397:    sstebz
398:    dstebz
399: 
400:    sstemr
401:    dstemr
402: 
403:    ssterf
404:    dsterf
405: 
406:    sstein
407:    dstein
408: 
409:    sstev
410:    dstev
411: 
412:    ssyev
413:    dsyev
414: 
415:    ssyevd
416:    dsyevd
417: 
418:    ssyevr
419:    dsyevr
420: 
421:    ssygv
422:    dsygv
423: 
424:    ssygvd
425:    dsygvd
426: 
427:    ssygvx
428:    dsygvx
429: 
430:    slange
431:    dlange
432:    clange
433:    zlange
434: 
435:    ilaver
436: 
437: '''
438: #
439: # Author: Pearu Peterson, March 2002
440: #
441: 
442: from __future__ import division, print_function, absolute_import
443: 
444: __all__ = ['get_lapack_funcs']
445: 
446: import numpy as _np
447: 
448: from .blas import _get_funcs
449: 
450: # Backward compatibility:
451: from .blas import find_best_blas_type as find_best_lapack_type
452: 
453: from scipy.linalg import _flapack
454: try:
455:     from scipy.linalg import _clapack
456: except ImportError:
457:     _clapack = None
458: 
459: # Backward compatibility
460: from scipy._lib._util import DeprecatedImport as _DeprecatedImport
461: clapack = _DeprecatedImport("scipy.linalg.blas.clapack", "scipy.linalg.lapack")
462: flapack = _DeprecatedImport("scipy.linalg.blas.flapack", "scipy.linalg.lapack")
463: 
464: # Expose all functions (only flapack --- clapack is an implementation detail)
465: empty_module = None
466: from scipy.linalg._flapack import *
467: del empty_module
468: 
469: _dep_message = '''The `*gegv` family of routines has been deprecated in
470: LAPACK 3.6.0 in favor of the `*ggev` family of routines.
471: The corresponding wrappers will be removed from SciPy in
472: a future release.'''
473: 
474: cgegv = _np.deprecate(cgegv, old_name='cgegv', message=_dep_message)
475: dgegv = _np.deprecate(dgegv, old_name='dgegv', message=_dep_message)
476: sgegv = _np.deprecate(sgegv, old_name='sgegv', message=_dep_message)
477: zgegv = _np.deprecate(zgegv, old_name='zgegv', message=_dep_message)
478: 
479: # Modyfy _flapack in this scope so the deprecation warnings apply to
480: # functions returned by get_lapack_funcs.
481: _flapack.cgegv = cgegv
482: _flapack.dgegv = dgegv
483: _flapack.sgegv = sgegv
484: _flapack.zgegv = zgegv
485: 
486: # some convenience alias for complex functions
487: _lapack_alias = {
488:     'corghr': 'cunghr', 'zorghr': 'zunghr',
489:     'corghr_lwork': 'cunghr_lwork', 'zorghr_lwork': 'zunghr_lwork',
490:     'corgqr': 'cungqr', 'zorgqr': 'zungqr',
491:     'cormqr': 'cunmqr', 'zormqr': 'zunmqr',
492:     'corgrq': 'cungrq', 'zorgrq': 'zungrq',
493: }
494: 
495: 
496: def get_lapack_funcs(names, arrays=(), dtype=None):
497:     '''Return available LAPACK function objects from names.
498: 
499:     Arrays are used to determine the optimal prefix of LAPACK routines.
500: 
501:     Parameters
502:     ----------
503:     names : str or sequence of str
504:         Name(s) of LAPACK functions without type prefix.
505: 
506:     arrays : sequence of ndarrays, optional
507:         Arrays can be given to determine optimal prefix of LAPACK
508:         routines. If not given, double-precision routines will be
509:         used, otherwise the most generic type in arrays will be used.
510: 
511:     dtype : str or dtype, optional
512:         Data-type specifier. Not used if `arrays` is non-empty.
513: 
514:     Returns
515:     -------
516:     funcs : list
517:         List containing the found function(s).
518: 
519:     Notes
520:     -----
521:     This routine automatically chooses between Fortran/C
522:     interfaces. Fortran code is used whenever possible for arrays with
523:     column major order. In all other cases, C code is preferred.
524: 
525:     In LAPACK, the naming convention is that all functions start with a
526:     type prefix, which depends on the type of the principal
527:     matrix. These can be one of {'s', 'd', 'c', 'z'} for the numpy
528:     types {float32, float64, complex64, complex128} respectively, and
529:     are stored in attribute ``typecode`` of the returned functions.
530: 
531:     Examples
532:     --------
533:     Suppose we would like to use '?lange' routine which computes the selected
534:     norm of an array. We pass our array in order to get the correct 'lange'
535:     flavor.
536: 
537:     >>> import scipy.linalg as LA
538:     >>> a = np.random.rand(3,2)
539:     >>> x_lange = LA.get_lapack_funcs('lange', (a,))
540:     >>> x_lange.typecode
541:     'd'
542:     >>> x_lange = LA.get_lapack_funcs('lange',(a*1j,))
543:     >>> x_lange.typecode
544:     'z'
545: 
546:     Several LAPACK routines work best when its internal WORK array has
547:     the optimal size (big enough for fast computation and small enough to
548:     avoid waste of memory). This size is determined also by a dedicated query
549:     to the function which is often wrapped as a standalone function and
550:     commonly denoted as ``###_lwork``. Below is an example for ``?sysv``
551: 
552:     >>> import scipy.linalg as LA
553:     >>> a = np.random.rand(1000,1000)
554:     >>> b = np.random.rand(1000,1)*1j
555:     >>> # We pick up zsysv and zsysv_lwork due to b array
556:     ... xsysv, xlwork = LA.get_lapack_funcs(('sysv', 'sysv_lwork'), (a, b))
557:     >>> opt_lwork, _ = xlwork(a.shape[0])  # returns a complex for 'z' prefix
558:     >>> udut, ipiv, x, info = xsysv(a, b, lwork=int(opt_lwork.real))
559: 
560:     '''
561:     return _get_funcs(names, arrays, dtype,
562:                       "LAPACK", _flapack, _clapack,
563:                       "flapack", "clapack", _lapack_alias)
564: 
565: 
566: def _compute_lwork(routine, *args, **kwargs):
567:     '''
568:     Round floating-point lwork returned by lapack to integer.
569: 
570:     Several LAPACK routines compute optimal values for LWORK, which
571:     they return in a floating-point variable. However, for large
572:     values of LWORK, single-precision floating point is not sufficient
573:     to hold the exact value --- some LAPACK versions (<= 3.5.0 at
574:     least) truncate the returned integer to single precision and in
575:     some cases this can be smaller than the required value.
576: 
577:     Examples
578:     --------
579:     >>> from scipy.linalg import lapack
580:     >>> n = 5000
581:     >>> s_r, s_lw = lapack.get_lapack_funcs(('sysvx', 'sysvx_lwork'))
582:     >>> lwork = lapack._compute_lwork(s_lw, n)
583:     >>> lwork
584:     32000
585: 
586:     '''
587:     wi = routine(*args, **kwargs)
588:     if len(wi) < 2:
589:         raise ValueError('')
590:     info = wi[-1]
591:     if info != 0:
592:         raise ValueError("Internal work array size computation failed: "
593:                          "%d" % (info,))
594: 
595:     lwork = [w.real for w in wi[:-1]]
596: 
597:     dtype = getattr(routine, 'dtype', None)
598:     if dtype == _np.float32 or dtype == _np.complex64:
599:         # Single-precision routine -- take next fp value to work
600:         # around possible truncation in LAPACK code
601:         lwork = _np.nextafter(lwork, _np.inf, dtype=_np.float32)
602: 
603:     lwork = _np.array(lwork, _np.int64)
604:     if _np.any(_np.logical_or(lwork < 0, lwork > _np.iinfo(_np.int32).max)):
605:         raise ValueError("Too large work array required -- computation cannot "
606:                          "be performed with standard 32-bit LAPACK.")
607:     lwork = lwork.astype(_np.int32)
608:     if lwork.size == 1:
609:         return lwork[0]
610:     return lwork
611: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_22305 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 437, (-1)), 'str', '\nLow-level LAPACK functions (:mod:`scipy.linalg.lapack`)\n=======================================================\n\nThis module contains low-level functions from the LAPACK library.\n\nThe `*gegv` family of routines have been removed from LAPACK 3.6.0\nand have been deprecated in SciPy 0.17.0. They will be removed in\na future release.\n\n.. versionadded:: 0.12.0\n\n.. warning::\n\n   These functions do little to no error checking.\n   It is possible to cause crashes by mis-using them,\n   so prefer using the higher-level routines in `scipy.linalg`.\n\nFinding functions\n-----------------\n\n.. autosummary::\n\n   get_lapack_funcs\n\nAll functions\n-------------\n\n.. autosummary::\n   :toctree: generated/\n\n\n   sgbsv\n   dgbsv\n   cgbsv\n   zgbsv\n\n   sgbtrf\n   dgbtrf\n   cgbtrf\n   zgbtrf\n\n   sgbtrs\n   dgbtrs\n   cgbtrs\n   zgbtrs\n\n   sgebal\n   dgebal\n   cgebal\n   zgebal\n\n   sgees\n   dgees\n   cgees\n   zgees\n\n   sgeev\n   dgeev\n   cgeev\n   zgeev\n\n   sgeev_lwork\n   dgeev_lwork\n   cgeev_lwork\n   zgeev_lwork\n\n   sgegv\n   dgegv\n   cgegv\n   zgegv\n\n   sgehrd\n   dgehrd\n   cgehrd\n   zgehrd\n\n   sgehrd_lwork\n   dgehrd_lwork\n   cgehrd_lwork\n   zgehrd_lwork\n\n   sgelss\n   dgelss\n   cgelss\n   zgelss\n\n   sgelss_lwork\n   dgelss_lwork\n   cgelss_lwork\n   zgelss_lwork\n\n   sgelsd\n   dgelsd\n   cgelsd\n   zgelsd\n\n   sgelsd_lwork\n   dgelsd_lwork\n   cgelsd_lwork\n   zgelsd_lwork\n\n   sgelsy\n   dgelsy\n   cgelsy\n   zgelsy\n\n   sgelsy_lwork\n   dgelsy_lwork\n   cgelsy_lwork\n   zgelsy_lwork\n\n   sgeqp3\n   dgeqp3\n   cgeqp3\n   zgeqp3\n\n   sgeqrf\n   dgeqrf\n   cgeqrf\n   zgeqrf\n\n   sgerqf\n   dgerqf\n   cgerqf\n   zgerqf\n\n   sgesdd\n   dgesdd\n   cgesdd\n   zgesdd\n\n   sgesdd_lwork\n   dgesdd_lwork\n   cgesdd_lwork\n   zgesdd_lwork\n\n   sgesvd\n   dgesvd\n   cgesvd\n   zgesvd\n\n   sgesvd_lwork\n   dgesvd_lwork\n   cgesvd_lwork\n   zgesvd_lwork\n\n   sgesv\n   dgesv\n   cgesv\n   zgesv\n\n   sgesvx\n   dgesvx\n   cgesvx\n   zgesvx\n\n   sgecon\n   dgecon\n   cgecon\n   zgecon\n\n   ssysv\n   dsysv\n   csysv\n   zsysv\n\n   ssysv_lwork\n   dsysv_lwork\n   csysv_lwork\n   zsysv_lwork\n\n   ssysvx\n   dsysvx\n   csysvx\n   zsysvx\n\n   ssysvx_lwork\n   dsysvx_lwork\n   csysvx_lwork\n   zsysvx_lwork\n\n   ssytrd\n   dsytrd\n\n   ssytrd_lwork\n   dsytrd_lwork\n\n   chetrd\n   zhetrd\n\n   chetrd_lwork\n   zhetrd_lwork\n\n   chesv\n   zhesv\n\n   chesv_lwork\n   zhesv_lwork\n\n   chesvx\n   zhesvx\n\n   chesvx_lwork\n   zhesvx_lwork\n\n   sgetrf\n   dgetrf\n   cgetrf\n   zgetrf\n\n   sgetri\n   dgetri\n   cgetri\n   zgetri\n\n   sgetri_lwork\n   dgetri_lwork\n   cgetri_lwork\n   zgetri_lwork\n\n   sgetrs\n   dgetrs\n   cgetrs\n   zgetrs\n\n   sgges\n   dgges\n   cgges\n   zgges\n\n   sggev\n   dggev\n   cggev\n   zggev\n\n   chbevd\n   zhbevd\n\n   chbevx\n   zhbevx\n\n   cheev\n   zheev\n\n   cheevd\n   zheevd\n\n   cheevr\n   zheevr\n\n   chegv\n   zhegv\n\n   chegvd\n   zhegvd\n\n   chegvx\n   zhegvx\n\n   slarf\n   dlarf\n   clarf\n   zlarf\n\n   slarfg\n   dlarfg\n   clarfg\n   zlarfg\n\n   slartg\n   dlartg\n   clartg\n   zlartg\n\n   slasd4\n   dlasd4\n\n   slaswp\n   dlaswp\n   claswp\n   zlaswp\n\n   slauum\n   dlauum\n   clauum\n   zlauum\n\n   spbsv\n   dpbsv\n   cpbsv\n   zpbsv\n\n   spbtrf\n   dpbtrf\n   cpbtrf\n   zpbtrf\n\n   spbtrs\n   dpbtrs\n   cpbtrs\n   zpbtrs\n\n   sposv\n   dposv\n   cposv\n   zposv\n\n   sposvx\n   dposvx\n   cposvx\n   zposvx\n\n   spocon\n   dpocon\n   cpocon\n   zpocon\n\n   spotrf\n   dpotrf\n   cpotrf\n   zpotrf\n\n   spotri\n   dpotri\n   cpotri\n   zpotri\n\n   spotrs\n   dpotrs\n   cpotrs\n   zpotrs\n\n   crot\n   zrot\n\n   strsyl\n   dtrsyl\n   ctrsyl\n   ztrsyl\n\n   strtri\n   dtrtri\n   ctrtri\n   ztrtri\n\n   strtrs\n   dtrtrs\n   ctrtrs\n   ztrtrs\n\n   cunghr\n   zunghr\n\n   cungqr\n   zungqr\n\n   cungrq\n   zungrq\n\n   cunmqr\n   zunmqr\n\n   sgtsv\n   dgtsv\n   cgtsv\n   zgtsv\n\n   sptsv\n   dptsv\n   cptsv\n   zptsv\n\n   slamch\n   dlamch\n\n   sorghr\n   dorghr\n   sorgqr\n   dorgqr\n\n   sorgrq\n   dorgrq\n\n   sormqr\n   dormqr\n\n   ssbev\n   dsbev\n\n   ssbevd\n   dsbevd\n\n   ssbevx\n   dsbevx\n\n   sstebz\n   dstebz\n\n   sstemr\n   dstemr\n\n   ssterf\n   dsterf\n\n   sstein\n   dstein\n\n   sstev\n   dstev\n\n   ssyev\n   dsyev\n\n   ssyevd\n   dsyevd\n\n   ssyevr\n   dsyevr\n\n   ssygv\n   dsygv\n\n   ssygvd\n   dsygvd\n\n   ssygvx\n   dsygvx\n\n   slange\n   dlange\n   clange\n   zlange\n\n   ilaver\n\n')

# Assigning a List to a Name (line 444):
__all__ = ['get_lapack_funcs']
module_type_store.set_exportable_members(['get_lapack_funcs'])

# Obtaining an instance of the builtin type 'list' (line 444)
list_22306 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 444, 10), 'list')
# Adding type elements to the builtin type 'list' instance (line 444)
# Adding element type (line 444)
str_22307 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 444, 11), 'str', 'get_lapack_funcs')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 444, 10), list_22306, str_22307)

# Assigning a type to the variable '__all__' (line 444)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 444, 0), '__all__', list_22306)
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 446, 0))

# 'import numpy' statement (line 446)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/linalg/')
import_22308 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 446, 0), 'numpy')

if (type(import_22308) is not StypyTypeError):

    if (import_22308 != 'pyd_module'):
        __import__(import_22308)
        sys_modules_22309 = sys.modules[import_22308]
        import_module(stypy.reporting.localization.Localization(__file__, 446, 0), '_np', sys_modules_22309.module_type_store, module_type_store)
    else:
        import numpy as _np

        import_module(stypy.reporting.localization.Localization(__file__, 446, 0), '_np', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 446)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 446, 0), 'numpy', import_22308)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/linalg/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 448, 0))

# 'from scipy.linalg.blas import _get_funcs' statement (line 448)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/linalg/')
import_22310 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 448, 0), 'scipy.linalg.blas')

if (type(import_22310) is not StypyTypeError):

    if (import_22310 != 'pyd_module'):
        __import__(import_22310)
        sys_modules_22311 = sys.modules[import_22310]
        import_from_module(stypy.reporting.localization.Localization(__file__, 448, 0), 'scipy.linalg.blas', sys_modules_22311.module_type_store, module_type_store, ['_get_funcs'])
        nest_module(stypy.reporting.localization.Localization(__file__, 448, 0), __file__, sys_modules_22311, sys_modules_22311.module_type_store, module_type_store)
    else:
        from scipy.linalg.blas import _get_funcs

        import_from_module(stypy.reporting.localization.Localization(__file__, 448, 0), 'scipy.linalg.blas', None, module_type_store, ['_get_funcs'], [_get_funcs])

else:
    # Assigning a type to the variable 'scipy.linalg.blas' (line 448)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 448, 0), 'scipy.linalg.blas', import_22310)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/linalg/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 451, 0))

# 'from scipy.linalg.blas import find_best_lapack_type' statement (line 451)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/linalg/')
import_22312 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 451, 0), 'scipy.linalg.blas')

if (type(import_22312) is not StypyTypeError):

    if (import_22312 != 'pyd_module'):
        __import__(import_22312)
        sys_modules_22313 = sys.modules[import_22312]
        import_from_module(stypy.reporting.localization.Localization(__file__, 451, 0), 'scipy.linalg.blas', sys_modules_22313.module_type_store, module_type_store, ['find_best_blas_type'])
        nest_module(stypy.reporting.localization.Localization(__file__, 451, 0), __file__, sys_modules_22313, sys_modules_22313.module_type_store, module_type_store)
    else:
        from scipy.linalg.blas import find_best_blas_type as find_best_lapack_type

        import_from_module(stypy.reporting.localization.Localization(__file__, 451, 0), 'scipy.linalg.blas', None, module_type_store, ['find_best_blas_type'], [find_best_lapack_type])

else:
    # Assigning a type to the variable 'scipy.linalg.blas' (line 451)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 451, 0), 'scipy.linalg.blas', import_22312)

# Adding an alias
module_type_store.add_alias('find_best_lapack_type', 'find_best_blas_type')
remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/linalg/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 453, 0))

# 'from scipy.linalg import _flapack' statement (line 453)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/linalg/')
import_22314 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 453, 0), 'scipy.linalg')

if (type(import_22314) is not StypyTypeError):

    if (import_22314 != 'pyd_module'):
        __import__(import_22314)
        sys_modules_22315 = sys.modules[import_22314]
        import_from_module(stypy.reporting.localization.Localization(__file__, 453, 0), 'scipy.linalg', sys_modules_22315.module_type_store, module_type_store, ['_flapack'])
        nest_module(stypy.reporting.localization.Localization(__file__, 453, 0), __file__, sys_modules_22315, sys_modules_22315.module_type_store, module_type_store)
    else:
        from scipy.linalg import _flapack

        import_from_module(stypy.reporting.localization.Localization(__file__, 453, 0), 'scipy.linalg', None, module_type_store, ['_flapack'], [_flapack])

else:
    # Assigning a type to the variable 'scipy.linalg' (line 453)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 453, 0), 'scipy.linalg', import_22314)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/linalg/')



# SSA begins for try-except statement (line 454)
module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 455, 4))

# 'from scipy.linalg import _clapack' statement (line 455)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/linalg/')
import_22316 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 455, 4), 'scipy.linalg')

if (type(import_22316) is not StypyTypeError):

    if (import_22316 != 'pyd_module'):
        __import__(import_22316)
        sys_modules_22317 = sys.modules[import_22316]
        import_from_module(stypy.reporting.localization.Localization(__file__, 455, 4), 'scipy.linalg', sys_modules_22317.module_type_store, module_type_store, ['_clapack'])
        nest_module(stypy.reporting.localization.Localization(__file__, 455, 4), __file__, sys_modules_22317, sys_modules_22317.module_type_store, module_type_store)
    else:
        from scipy.linalg import _clapack

        import_from_module(stypy.reporting.localization.Localization(__file__, 455, 4), 'scipy.linalg', None, module_type_store, ['_clapack'], [_clapack])

else:
    # Assigning a type to the variable 'scipy.linalg' (line 455)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 455, 4), 'scipy.linalg', import_22316)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/linalg/')

# SSA branch for the except part of a try statement (line 454)
# SSA branch for the except 'ImportError' branch of a try statement (line 454)
module_type_store.open_ssa_branch('except')

# Assigning a Name to a Name (line 457):
# Getting the type of 'None' (line 457)
None_22318 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 457, 15), 'None')
# Assigning a type to the variable '_clapack' (line 457)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 457, 4), '_clapack', None_22318)
# SSA join for try-except statement (line 454)
module_type_store = module_type_store.join_ssa_context()

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 460, 0))

# 'from scipy._lib._util import _DeprecatedImport' statement (line 460)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/linalg/')
import_22319 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 460, 0), 'scipy._lib._util')

if (type(import_22319) is not StypyTypeError):

    if (import_22319 != 'pyd_module'):
        __import__(import_22319)
        sys_modules_22320 = sys.modules[import_22319]
        import_from_module(stypy.reporting.localization.Localization(__file__, 460, 0), 'scipy._lib._util', sys_modules_22320.module_type_store, module_type_store, ['DeprecatedImport'])
        nest_module(stypy.reporting.localization.Localization(__file__, 460, 0), __file__, sys_modules_22320, sys_modules_22320.module_type_store, module_type_store)
    else:
        from scipy._lib._util import DeprecatedImport as _DeprecatedImport

        import_from_module(stypy.reporting.localization.Localization(__file__, 460, 0), 'scipy._lib._util', None, module_type_store, ['DeprecatedImport'], [_DeprecatedImport])

else:
    # Assigning a type to the variable 'scipy._lib._util' (line 460)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 460, 0), 'scipy._lib._util', import_22319)

# Adding an alias
module_type_store.add_alias('_DeprecatedImport', 'DeprecatedImport')
remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/linalg/')


# Assigning a Call to a Name (line 461):

# Call to _DeprecatedImport(...): (line 461)
# Processing the call arguments (line 461)
str_22322 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 461, 28), 'str', 'scipy.linalg.blas.clapack')
str_22323 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 461, 57), 'str', 'scipy.linalg.lapack')
# Processing the call keyword arguments (line 461)
kwargs_22324 = {}
# Getting the type of '_DeprecatedImport' (line 461)
_DeprecatedImport_22321 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 461, 10), '_DeprecatedImport', False)
# Calling _DeprecatedImport(args, kwargs) (line 461)
_DeprecatedImport_call_result_22325 = invoke(stypy.reporting.localization.Localization(__file__, 461, 10), _DeprecatedImport_22321, *[str_22322, str_22323], **kwargs_22324)

# Assigning a type to the variable 'clapack' (line 461)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 461, 0), 'clapack', _DeprecatedImport_call_result_22325)

# Assigning a Call to a Name (line 462):

# Call to _DeprecatedImport(...): (line 462)
# Processing the call arguments (line 462)
str_22327 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 462, 28), 'str', 'scipy.linalg.blas.flapack')
str_22328 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 462, 57), 'str', 'scipy.linalg.lapack')
# Processing the call keyword arguments (line 462)
kwargs_22329 = {}
# Getting the type of '_DeprecatedImport' (line 462)
_DeprecatedImport_22326 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 462, 10), '_DeprecatedImport', False)
# Calling _DeprecatedImport(args, kwargs) (line 462)
_DeprecatedImport_call_result_22330 = invoke(stypy.reporting.localization.Localization(__file__, 462, 10), _DeprecatedImport_22326, *[str_22327, str_22328], **kwargs_22329)

# Assigning a type to the variable 'flapack' (line 462)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 462, 0), 'flapack', _DeprecatedImport_call_result_22330)

# Assigning a Name to a Name (line 465):
# Getting the type of 'None' (line 465)
None_22331 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 465, 15), 'None')
# Assigning a type to the variable 'empty_module' (line 465)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 465, 0), 'empty_module', None_22331)
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 466, 0))

# 'from scipy.linalg._flapack import ' statement (line 466)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/linalg/')
import_22332 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 466, 0), 'scipy.linalg._flapack')

if (type(import_22332) is not StypyTypeError):

    if (import_22332 != 'pyd_module'):
        __import__(import_22332)
        sys_modules_22333 = sys.modules[import_22332]
        import_from_module(stypy.reporting.localization.Localization(__file__, 466, 0), 'scipy.linalg._flapack', sys_modules_22333.module_type_store, module_type_store, ['*'])
        nest_module(stypy.reporting.localization.Localization(__file__, 466, 0), __file__, sys_modules_22333, sys_modules_22333.module_type_store, module_type_store)
    else:
        from scipy.linalg._flapack import *

        import_from_module(stypy.reporting.localization.Localization(__file__, 466, 0), 'scipy.linalg._flapack', None, module_type_store, ['*'], None)

else:
    # Assigning a type to the variable 'scipy.linalg._flapack' (line 466)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 466, 0), 'scipy.linalg._flapack', import_22332)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/linalg/')

# Deleting a member
module_type_store.del_member(stypy.reporting.localization.Localization(__file__, 467, 0), module_type_store, 'empty_module')

# Assigning a Str to a Name (line 469):
str_22334 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 472, (-1)), 'str', 'The `*gegv` family of routines has been deprecated in\nLAPACK 3.6.0 in favor of the `*ggev` family of routines.\nThe corresponding wrappers will be removed from SciPy in\na future release.')
# Assigning a type to the variable '_dep_message' (line 469)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 469, 0), '_dep_message', str_22334)

# Assigning a Call to a Name (line 474):

# Call to deprecate(...): (line 474)
# Processing the call arguments (line 474)
# Getting the type of 'cgegv' (line 474)
cgegv_22337 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 474, 22), 'cgegv', False)
# Processing the call keyword arguments (line 474)
str_22338 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 474, 38), 'str', 'cgegv')
keyword_22339 = str_22338
# Getting the type of '_dep_message' (line 474)
_dep_message_22340 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 474, 55), '_dep_message', False)
keyword_22341 = _dep_message_22340
kwargs_22342 = {'message': keyword_22341, 'old_name': keyword_22339}
# Getting the type of '_np' (line 474)
_np_22335 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 474, 8), '_np', False)
# Obtaining the member 'deprecate' of a type (line 474)
deprecate_22336 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 474, 8), _np_22335, 'deprecate')
# Calling deprecate(args, kwargs) (line 474)
deprecate_call_result_22343 = invoke(stypy.reporting.localization.Localization(__file__, 474, 8), deprecate_22336, *[cgegv_22337], **kwargs_22342)

# Assigning a type to the variable 'cgegv' (line 474)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 474, 0), 'cgegv', deprecate_call_result_22343)

# Assigning a Call to a Name (line 475):

# Call to deprecate(...): (line 475)
# Processing the call arguments (line 475)
# Getting the type of 'dgegv' (line 475)
dgegv_22346 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 475, 22), 'dgegv', False)
# Processing the call keyword arguments (line 475)
str_22347 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 475, 38), 'str', 'dgegv')
keyword_22348 = str_22347
# Getting the type of '_dep_message' (line 475)
_dep_message_22349 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 475, 55), '_dep_message', False)
keyword_22350 = _dep_message_22349
kwargs_22351 = {'message': keyword_22350, 'old_name': keyword_22348}
# Getting the type of '_np' (line 475)
_np_22344 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 475, 8), '_np', False)
# Obtaining the member 'deprecate' of a type (line 475)
deprecate_22345 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 475, 8), _np_22344, 'deprecate')
# Calling deprecate(args, kwargs) (line 475)
deprecate_call_result_22352 = invoke(stypy.reporting.localization.Localization(__file__, 475, 8), deprecate_22345, *[dgegv_22346], **kwargs_22351)

# Assigning a type to the variable 'dgegv' (line 475)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 475, 0), 'dgegv', deprecate_call_result_22352)

# Assigning a Call to a Name (line 476):

# Call to deprecate(...): (line 476)
# Processing the call arguments (line 476)
# Getting the type of 'sgegv' (line 476)
sgegv_22355 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 476, 22), 'sgegv', False)
# Processing the call keyword arguments (line 476)
str_22356 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 476, 38), 'str', 'sgegv')
keyword_22357 = str_22356
# Getting the type of '_dep_message' (line 476)
_dep_message_22358 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 476, 55), '_dep_message', False)
keyword_22359 = _dep_message_22358
kwargs_22360 = {'message': keyword_22359, 'old_name': keyword_22357}
# Getting the type of '_np' (line 476)
_np_22353 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 476, 8), '_np', False)
# Obtaining the member 'deprecate' of a type (line 476)
deprecate_22354 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 476, 8), _np_22353, 'deprecate')
# Calling deprecate(args, kwargs) (line 476)
deprecate_call_result_22361 = invoke(stypy.reporting.localization.Localization(__file__, 476, 8), deprecate_22354, *[sgegv_22355], **kwargs_22360)

# Assigning a type to the variable 'sgegv' (line 476)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 476, 0), 'sgegv', deprecate_call_result_22361)

# Assigning a Call to a Name (line 477):

# Call to deprecate(...): (line 477)
# Processing the call arguments (line 477)
# Getting the type of 'zgegv' (line 477)
zgegv_22364 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 477, 22), 'zgegv', False)
# Processing the call keyword arguments (line 477)
str_22365 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 477, 38), 'str', 'zgegv')
keyword_22366 = str_22365
# Getting the type of '_dep_message' (line 477)
_dep_message_22367 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 477, 55), '_dep_message', False)
keyword_22368 = _dep_message_22367
kwargs_22369 = {'message': keyword_22368, 'old_name': keyword_22366}
# Getting the type of '_np' (line 477)
_np_22362 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 477, 8), '_np', False)
# Obtaining the member 'deprecate' of a type (line 477)
deprecate_22363 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 477, 8), _np_22362, 'deprecate')
# Calling deprecate(args, kwargs) (line 477)
deprecate_call_result_22370 = invoke(stypy.reporting.localization.Localization(__file__, 477, 8), deprecate_22363, *[zgegv_22364], **kwargs_22369)

# Assigning a type to the variable 'zgegv' (line 477)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 477, 0), 'zgegv', deprecate_call_result_22370)

# Assigning a Name to a Attribute (line 481):
# Getting the type of 'cgegv' (line 481)
cgegv_22371 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 481, 17), 'cgegv')
# Getting the type of '_flapack' (line 481)
_flapack_22372 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 481, 0), '_flapack')
# Setting the type of the member 'cgegv' of a type (line 481)
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 481, 0), _flapack_22372, 'cgegv', cgegv_22371)

# Assigning a Name to a Attribute (line 482):
# Getting the type of 'dgegv' (line 482)
dgegv_22373 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 482, 17), 'dgegv')
# Getting the type of '_flapack' (line 482)
_flapack_22374 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 482, 0), '_flapack')
# Setting the type of the member 'dgegv' of a type (line 482)
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 482, 0), _flapack_22374, 'dgegv', dgegv_22373)

# Assigning a Name to a Attribute (line 483):
# Getting the type of 'sgegv' (line 483)
sgegv_22375 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 483, 17), 'sgegv')
# Getting the type of '_flapack' (line 483)
_flapack_22376 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 483, 0), '_flapack')
# Setting the type of the member 'sgegv' of a type (line 483)
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 483, 0), _flapack_22376, 'sgegv', sgegv_22375)

# Assigning a Name to a Attribute (line 484):
# Getting the type of 'zgegv' (line 484)
zgegv_22377 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 484, 17), 'zgegv')
# Getting the type of '_flapack' (line 484)
_flapack_22378 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 484, 0), '_flapack')
# Setting the type of the member 'zgegv' of a type (line 484)
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 484, 0), _flapack_22378, 'zgegv', zgegv_22377)

# Assigning a Dict to a Name (line 487):

# Obtaining an instance of the builtin type 'dict' (line 487)
dict_22379 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 487, 16), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 487)
# Adding element type (key, value) (line 487)
str_22380 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 488, 4), 'str', 'corghr')
str_22381 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 488, 14), 'str', 'cunghr')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 487, 16), dict_22379, (str_22380, str_22381))
# Adding element type (key, value) (line 487)
str_22382 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 488, 24), 'str', 'zorghr')
str_22383 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 488, 34), 'str', 'zunghr')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 487, 16), dict_22379, (str_22382, str_22383))
# Adding element type (key, value) (line 487)
str_22384 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 489, 4), 'str', 'corghr_lwork')
str_22385 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 489, 20), 'str', 'cunghr_lwork')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 487, 16), dict_22379, (str_22384, str_22385))
# Adding element type (key, value) (line 487)
str_22386 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 489, 36), 'str', 'zorghr_lwork')
str_22387 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 489, 52), 'str', 'zunghr_lwork')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 487, 16), dict_22379, (str_22386, str_22387))
# Adding element type (key, value) (line 487)
str_22388 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 490, 4), 'str', 'corgqr')
str_22389 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 490, 14), 'str', 'cungqr')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 487, 16), dict_22379, (str_22388, str_22389))
# Adding element type (key, value) (line 487)
str_22390 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 490, 24), 'str', 'zorgqr')
str_22391 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 490, 34), 'str', 'zungqr')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 487, 16), dict_22379, (str_22390, str_22391))
# Adding element type (key, value) (line 487)
str_22392 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 491, 4), 'str', 'cormqr')
str_22393 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 491, 14), 'str', 'cunmqr')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 487, 16), dict_22379, (str_22392, str_22393))
# Adding element type (key, value) (line 487)
str_22394 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 491, 24), 'str', 'zormqr')
str_22395 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 491, 34), 'str', 'zunmqr')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 487, 16), dict_22379, (str_22394, str_22395))
# Adding element type (key, value) (line 487)
str_22396 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 492, 4), 'str', 'corgrq')
str_22397 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 492, 14), 'str', 'cungrq')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 487, 16), dict_22379, (str_22396, str_22397))
# Adding element type (key, value) (line 487)
str_22398 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 492, 24), 'str', 'zorgrq')
str_22399 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 492, 34), 'str', 'zungrq')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 487, 16), dict_22379, (str_22398, str_22399))

# Assigning a type to the variable '_lapack_alias' (line 487)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 487, 0), '_lapack_alias', dict_22379)

@norecursion
def get_lapack_funcs(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    
    # Obtaining an instance of the builtin type 'tuple' (line 496)
    tuple_22400 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 496, 35), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 496)
    
    # Getting the type of 'None' (line 496)
    None_22401 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 496, 45), 'None')
    defaults = [tuple_22400, None_22401]
    # Create a new context for function 'get_lapack_funcs'
    module_type_store = module_type_store.open_function_context('get_lapack_funcs', 496, 0, False)
    
    # Passed parameters checking function
    get_lapack_funcs.stypy_localization = localization
    get_lapack_funcs.stypy_type_of_self = None
    get_lapack_funcs.stypy_type_store = module_type_store
    get_lapack_funcs.stypy_function_name = 'get_lapack_funcs'
    get_lapack_funcs.stypy_param_names_list = ['names', 'arrays', 'dtype']
    get_lapack_funcs.stypy_varargs_param_name = None
    get_lapack_funcs.stypy_kwargs_param_name = None
    get_lapack_funcs.stypy_call_defaults = defaults
    get_lapack_funcs.stypy_call_varargs = varargs
    get_lapack_funcs.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'get_lapack_funcs', ['names', 'arrays', 'dtype'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'get_lapack_funcs', localization, ['names', 'arrays', 'dtype'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'get_lapack_funcs(...)' code ##################

    str_22402 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 560, (-1)), 'str', "Return available LAPACK function objects from names.\n\n    Arrays are used to determine the optimal prefix of LAPACK routines.\n\n    Parameters\n    ----------\n    names : str or sequence of str\n        Name(s) of LAPACK functions without type prefix.\n\n    arrays : sequence of ndarrays, optional\n        Arrays can be given to determine optimal prefix of LAPACK\n        routines. If not given, double-precision routines will be\n        used, otherwise the most generic type in arrays will be used.\n\n    dtype : str or dtype, optional\n        Data-type specifier. Not used if `arrays` is non-empty.\n\n    Returns\n    -------\n    funcs : list\n        List containing the found function(s).\n\n    Notes\n    -----\n    This routine automatically chooses between Fortran/C\n    interfaces. Fortran code is used whenever possible for arrays with\n    column major order. In all other cases, C code is preferred.\n\n    In LAPACK, the naming convention is that all functions start with a\n    type prefix, which depends on the type of the principal\n    matrix. These can be one of {'s', 'd', 'c', 'z'} for the numpy\n    types {float32, float64, complex64, complex128} respectively, and\n    are stored in attribute ``typecode`` of the returned functions.\n\n    Examples\n    --------\n    Suppose we would like to use '?lange' routine which computes the selected\n    norm of an array. We pass our array in order to get the correct 'lange'\n    flavor.\n\n    >>> import scipy.linalg as LA\n    >>> a = np.random.rand(3,2)\n    >>> x_lange = LA.get_lapack_funcs('lange', (a,))\n    >>> x_lange.typecode\n    'd'\n    >>> x_lange = LA.get_lapack_funcs('lange',(a*1j,))\n    >>> x_lange.typecode\n    'z'\n\n    Several LAPACK routines work best when its internal WORK array has\n    the optimal size (big enough for fast computation and small enough to\n    avoid waste of memory). This size is determined also by a dedicated query\n    to the function which is often wrapped as a standalone function and\n    commonly denoted as ``###_lwork``. Below is an example for ``?sysv``\n\n    >>> import scipy.linalg as LA\n    >>> a = np.random.rand(1000,1000)\n    >>> b = np.random.rand(1000,1)*1j\n    >>> # We pick up zsysv and zsysv_lwork due to b array\n    ... xsysv, xlwork = LA.get_lapack_funcs(('sysv', 'sysv_lwork'), (a, b))\n    >>> opt_lwork, _ = xlwork(a.shape[0])  # returns a complex for 'z' prefix\n    >>> udut, ipiv, x, info = xsysv(a, b, lwork=int(opt_lwork.real))\n\n    ")
    
    # Call to _get_funcs(...): (line 561)
    # Processing the call arguments (line 561)
    # Getting the type of 'names' (line 561)
    names_22404 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 561, 22), 'names', False)
    # Getting the type of 'arrays' (line 561)
    arrays_22405 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 561, 29), 'arrays', False)
    # Getting the type of 'dtype' (line 561)
    dtype_22406 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 561, 37), 'dtype', False)
    str_22407 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 562, 22), 'str', 'LAPACK')
    # Getting the type of '_flapack' (line 562)
    _flapack_22408 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 562, 32), '_flapack', False)
    # Getting the type of '_clapack' (line 562)
    _clapack_22409 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 562, 42), '_clapack', False)
    str_22410 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 563, 22), 'str', 'flapack')
    str_22411 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 563, 33), 'str', 'clapack')
    # Getting the type of '_lapack_alias' (line 563)
    _lapack_alias_22412 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 563, 44), '_lapack_alias', False)
    # Processing the call keyword arguments (line 561)
    kwargs_22413 = {}
    # Getting the type of '_get_funcs' (line 561)
    _get_funcs_22403 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 561, 11), '_get_funcs', False)
    # Calling _get_funcs(args, kwargs) (line 561)
    _get_funcs_call_result_22414 = invoke(stypy.reporting.localization.Localization(__file__, 561, 11), _get_funcs_22403, *[names_22404, arrays_22405, dtype_22406, str_22407, _flapack_22408, _clapack_22409, str_22410, str_22411, _lapack_alias_22412], **kwargs_22413)
    
    # Assigning a type to the variable 'stypy_return_type' (line 561)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 561, 4), 'stypy_return_type', _get_funcs_call_result_22414)
    
    # ################# End of 'get_lapack_funcs(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'get_lapack_funcs' in the type store
    # Getting the type of 'stypy_return_type' (line 496)
    stypy_return_type_22415 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 496, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_22415)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'get_lapack_funcs'
    return stypy_return_type_22415

# Assigning a type to the variable 'get_lapack_funcs' (line 496)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 496, 0), 'get_lapack_funcs', get_lapack_funcs)

@norecursion
def _compute_lwork(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_compute_lwork'
    module_type_store = module_type_store.open_function_context('_compute_lwork', 566, 0, False)
    
    # Passed parameters checking function
    _compute_lwork.stypy_localization = localization
    _compute_lwork.stypy_type_of_self = None
    _compute_lwork.stypy_type_store = module_type_store
    _compute_lwork.stypy_function_name = '_compute_lwork'
    _compute_lwork.stypy_param_names_list = ['routine']
    _compute_lwork.stypy_varargs_param_name = 'args'
    _compute_lwork.stypy_kwargs_param_name = 'kwargs'
    _compute_lwork.stypy_call_defaults = defaults
    _compute_lwork.stypy_call_varargs = varargs
    _compute_lwork.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_compute_lwork', ['routine'], 'args', 'kwargs', defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_compute_lwork', localization, ['routine'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_compute_lwork(...)' code ##################

    str_22416 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 586, (-1)), 'str', "\n    Round floating-point lwork returned by lapack to integer.\n\n    Several LAPACK routines compute optimal values for LWORK, which\n    they return in a floating-point variable. However, for large\n    values of LWORK, single-precision floating point is not sufficient\n    to hold the exact value --- some LAPACK versions (<= 3.5.0 at\n    least) truncate the returned integer to single precision and in\n    some cases this can be smaller than the required value.\n\n    Examples\n    --------\n    >>> from scipy.linalg import lapack\n    >>> n = 5000\n    >>> s_r, s_lw = lapack.get_lapack_funcs(('sysvx', 'sysvx_lwork'))\n    >>> lwork = lapack._compute_lwork(s_lw, n)\n    >>> lwork\n    32000\n\n    ")
    
    # Assigning a Call to a Name (line 587):
    
    # Call to routine(...): (line 587)
    # Getting the type of 'args' (line 587)
    args_22418 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 587, 18), 'args', False)
    # Processing the call keyword arguments (line 587)
    # Getting the type of 'kwargs' (line 587)
    kwargs_22419 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 587, 26), 'kwargs', False)
    kwargs_22420 = {'kwargs_22419': kwargs_22419}
    # Getting the type of 'routine' (line 587)
    routine_22417 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 587, 9), 'routine', False)
    # Calling routine(args, kwargs) (line 587)
    routine_call_result_22421 = invoke(stypy.reporting.localization.Localization(__file__, 587, 9), routine_22417, *[args_22418], **kwargs_22420)
    
    # Assigning a type to the variable 'wi' (line 587)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 587, 4), 'wi', routine_call_result_22421)
    
    
    
    # Call to len(...): (line 588)
    # Processing the call arguments (line 588)
    # Getting the type of 'wi' (line 588)
    wi_22423 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 588, 11), 'wi', False)
    # Processing the call keyword arguments (line 588)
    kwargs_22424 = {}
    # Getting the type of 'len' (line 588)
    len_22422 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 588, 7), 'len', False)
    # Calling len(args, kwargs) (line 588)
    len_call_result_22425 = invoke(stypy.reporting.localization.Localization(__file__, 588, 7), len_22422, *[wi_22423], **kwargs_22424)
    
    int_22426 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 588, 17), 'int')
    # Applying the binary operator '<' (line 588)
    result_lt_22427 = python_operator(stypy.reporting.localization.Localization(__file__, 588, 7), '<', len_call_result_22425, int_22426)
    
    # Testing the type of an if condition (line 588)
    if_condition_22428 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 588, 4), result_lt_22427)
    # Assigning a type to the variable 'if_condition_22428' (line 588)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 588, 4), 'if_condition_22428', if_condition_22428)
    # SSA begins for if statement (line 588)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 589)
    # Processing the call arguments (line 589)
    str_22430 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 589, 25), 'str', '')
    # Processing the call keyword arguments (line 589)
    kwargs_22431 = {}
    # Getting the type of 'ValueError' (line 589)
    ValueError_22429 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 589, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 589)
    ValueError_call_result_22432 = invoke(stypy.reporting.localization.Localization(__file__, 589, 14), ValueError_22429, *[str_22430], **kwargs_22431)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 589, 8), ValueError_call_result_22432, 'raise parameter', BaseException)
    # SSA join for if statement (line 588)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Subscript to a Name (line 590):
    
    # Obtaining the type of the subscript
    int_22433 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 590, 14), 'int')
    # Getting the type of 'wi' (line 590)
    wi_22434 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 590, 11), 'wi')
    # Obtaining the member '__getitem__' of a type (line 590)
    getitem___22435 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 590, 11), wi_22434, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 590)
    subscript_call_result_22436 = invoke(stypy.reporting.localization.Localization(__file__, 590, 11), getitem___22435, int_22433)
    
    # Assigning a type to the variable 'info' (line 590)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 590, 4), 'info', subscript_call_result_22436)
    
    
    # Getting the type of 'info' (line 591)
    info_22437 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 591, 7), 'info')
    int_22438 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 591, 15), 'int')
    # Applying the binary operator '!=' (line 591)
    result_ne_22439 = python_operator(stypy.reporting.localization.Localization(__file__, 591, 7), '!=', info_22437, int_22438)
    
    # Testing the type of an if condition (line 591)
    if_condition_22440 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 591, 4), result_ne_22439)
    # Assigning a type to the variable 'if_condition_22440' (line 591)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 591, 4), 'if_condition_22440', if_condition_22440)
    # SSA begins for if statement (line 591)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 592)
    # Processing the call arguments (line 592)
    str_22442 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 592, 25), 'str', 'Internal work array size computation failed: %d')
    
    # Obtaining an instance of the builtin type 'tuple' (line 593)
    tuple_22443 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 593, 33), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 593)
    # Adding element type (line 593)
    # Getting the type of 'info' (line 593)
    info_22444 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 593, 33), 'info', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 593, 33), tuple_22443, info_22444)
    
    # Applying the binary operator '%' (line 592)
    result_mod_22445 = python_operator(stypy.reporting.localization.Localization(__file__, 592, 25), '%', str_22442, tuple_22443)
    
    # Processing the call keyword arguments (line 592)
    kwargs_22446 = {}
    # Getting the type of 'ValueError' (line 592)
    ValueError_22441 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 592, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 592)
    ValueError_call_result_22447 = invoke(stypy.reporting.localization.Localization(__file__, 592, 14), ValueError_22441, *[result_mod_22445], **kwargs_22446)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 592, 8), ValueError_call_result_22447, 'raise parameter', BaseException)
    # SSA join for if statement (line 591)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a ListComp to a Name (line 595):
    # Calculating list comprehension
    # Calculating comprehension expression
    
    # Obtaining the type of the subscript
    int_22450 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 595, 33), 'int')
    slice_22451 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 595, 29), None, int_22450, None)
    # Getting the type of 'wi' (line 595)
    wi_22452 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 595, 29), 'wi')
    # Obtaining the member '__getitem__' of a type (line 595)
    getitem___22453 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 595, 29), wi_22452, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 595)
    subscript_call_result_22454 = invoke(stypy.reporting.localization.Localization(__file__, 595, 29), getitem___22453, slice_22451)
    
    comprehension_22455 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 595, 13), subscript_call_result_22454)
    # Assigning a type to the variable 'w' (line 595)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 595, 13), 'w', comprehension_22455)
    # Getting the type of 'w' (line 595)
    w_22448 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 595, 13), 'w')
    # Obtaining the member 'real' of a type (line 595)
    real_22449 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 595, 13), w_22448, 'real')
    list_22456 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 595, 13), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 595, 13), list_22456, real_22449)
    # Assigning a type to the variable 'lwork' (line 595)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 595, 4), 'lwork', list_22456)
    
    # Assigning a Call to a Name (line 597):
    
    # Call to getattr(...): (line 597)
    # Processing the call arguments (line 597)
    # Getting the type of 'routine' (line 597)
    routine_22458 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 597, 20), 'routine', False)
    str_22459 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 597, 29), 'str', 'dtype')
    # Getting the type of 'None' (line 597)
    None_22460 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 597, 38), 'None', False)
    # Processing the call keyword arguments (line 597)
    kwargs_22461 = {}
    # Getting the type of 'getattr' (line 597)
    getattr_22457 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 597, 12), 'getattr', False)
    # Calling getattr(args, kwargs) (line 597)
    getattr_call_result_22462 = invoke(stypy.reporting.localization.Localization(__file__, 597, 12), getattr_22457, *[routine_22458, str_22459, None_22460], **kwargs_22461)
    
    # Assigning a type to the variable 'dtype' (line 597)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 597, 4), 'dtype', getattr_call_result_22462)
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'dtype' (line 598)
    dtype_22463 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 598, 7), 'dtype')
    # Getting the type of '_np' (line 598)
    _np_22464 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 598, 16), '_np')
    # Obtaining the member 'float32' of a type (line 598)
    float32_22465 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 598, 16), _np_22464, 'float32')
    # Applying the binary operator '==' (line 598)
    result_eq_22466 = python_operator(stypy.reporting.localization.Localization(__file__, 598, 7), '==', dtype_22463, float32_22465)
    
    
    # Getting the type of 'dtype' (line 598)
    dtype_22467 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 598, 31), 'dtype')
    # Getting the type of '_np' (line 598)
    _np_22468 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 598, 40), '_np')
    # Obtaining the member 'complex64' of a type (line 598)
    complex64_22469 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 598, 40), _np_22468, 'complex64')
    # Applying the binary operator '==' (line 598)
    result_eq_22470 = python_operator(stypy.reporting.localization.Localization(__file__, 598, 31), '==', dtype_22467, complex64_22469)
    
    # Applying the binary operator 'or' (line 598)
    result_or_keyword_22471 = python_operator(stypy.reporting.localization.Localization(__file__, 598, 7), 'or', result_eq_22466, result_eq_22470)
    
    # Testing the type of an if condition (line 598)
    if_condition_22472 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 598, 4), result_or_keyword_22471)
    # Assigning a type to the variable 'if_condition_22472' (line 598)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 598, 4), 'if_condition_22472', if_condition_22472)
    # SSA begins for if statement (line 598)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 601):
    
    # Call to nextafter(...): (line 601)
    # Processing the call arguments (line 601)
    # Getting the type of 'lwork' (line 601)
    lwork_22475 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 601, 30), 'lwork', False)
    # Getting the type of '_np' (line 601)
    _np_22476 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 601, 37), '_np', False)
    # Obtaining the member 'inf' of a type (line 601)
    inf_22477 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 601, 37), _np_22476, 'inf')
    # Processing the call keyword arguments (line 601)
    # Getting the type of '_np' (line 601)
    _np_22478 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 601, 52), '_np', False)
    # Obtaining the member 'float32' of a type (line 601)
    float32_22479 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 601, 52), _np_22478, 'float32')
    keyword_22480 = float32_22479
    kwargs_22481 = {'dtype': keyword_22480}
    # Getting the type of '_np' (line 601)
    _np_22473 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 601, 16), '_np', False)
    # Obtaining the member 'nextafter' of a type (line 601)
    nextafter_22474 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 601, 16), _np_22473, 'nextafter')
    # Calling nextafter(args, kwargs) (line 601)
    nextafter_call_result_22482 = invoke(stypy.reporting.localization.Localization(__file__, 601, 16), nextafter_22474, *[lwork_22475, inf_22477], **kwargs_22481)
    
    # Assigning a type to the variable 'lwork' (line 601)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 601, 8), 'lwork', nextafter_call_result_22482)
    # SSA join for if statement (line 598)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 603):
    
    # Call to array(...): (line 603)
    # Processing the call arguments (line 603)
    # Getting the type of 'lwork' (line 603)
    lwork_22485 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 603, 22), 'lwork', False)
    # Getting the type of '_np' (line 603)
    _np_22486 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 603, 29), '_np', False)
    # Obtaining the member 'int64' of a type (line 603)
    int64_22487 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 603, 29), _np_22486, 'int64')
    # Processing the call keyword arguments (line 603)
    kwargs_22488 = {}
    # Getting the type of '_np' (line 603)
    _np_22483 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 603, 12), '_np', False)
    # Obtaining the member 'array' of a type (line 603)
    array_22484 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 603, 12), _np_22483, 'array')
    # Calling array(args, kwargs) (line 603)
    array_call_result_22489 = invoke(stypy.reporting.localization.Localization(__file__, 603, 12), array_22484, *[lwork_22485, int64_22487], **kwargs_22488)
    
    # Assigning a type to the variable 'lwork' (line 603)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 603, 4), 'lwork', array_call_result_22489)
    
    
    # Call to any(...): (line 604)
    # Processing the call arguments (line 604)
    
    # Call to logical_or(...): (line 604)
    # Processing the call arguments (line 604)
    
    # Getting the type of 'lwork' (line 604)
    lwork_22494 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 604, 30), 'lwork', False)
    int_22495 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 604, 38), 'int')
    # Applying the binary operator '<' (line 604)
    result_lt_22496 = python_operator(stypy.reporting.localization.Localization(__file__, 604, 30), '<', lwork_22494, int_22495)
    
    
    # Getting the type of 'lwork' (line 604)
    lwork_22497 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 604, 41), 'lwork', False)
    
    # Call to iinfo(...): (line 604)
    # Processing the call arguments (line 604)
    # Getting the type of '_np' (line 604)
    _np_22500 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 604, 59), '_np', False)
    # Obtaining the member 'int32' of a type (line 604)
    int32_22501 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 604, 59), _np_22500, 'int32')
    # Processing the call keyword arguments (line 604)
    kwargs_22502 = {}
    # Getting the type of '_np' (line 604)
    _np_22498 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 604, 49), '_np', False)
    # Obtaining the member 'iinfo' of a type (line 604)
    iinfo_22499 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 604, 49), _np_22498, 'iinfo')
    # Calling iinfo(args, kwargs) (line 604)
    iinfo_call_result_22503 = invoke(stypy.reporting.localization.Localization(__file__, 604, 49), iinfo_22499, *[int32_22501], **kwargs_22502)
    
    # Obtaining the member 'max' of a type (line 604)
    max_22504 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 604, 49), iinfo_call_result_22503, 'max')
    # Applying the binary operator '>' (line 604)
    result_gt_22505 = python_operator(stypy.reporting.localization.Localization(__file__, 604, 41), '>', lwork_22497, max_22504)
    
    # Processing the call keyword arguments (line 604)
    kwargs_22506 = {}
    # Getting the type of '_np' (line 604)
    _np_22492 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 604, 15), '_np', False)
    # Obtaining the member 'logical_or' of a type (line 604)
    logical_or_22493 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 604, 15), _np_22492, 'logical_or')
    # Calling logical_or(args, kwargs) (line 604)
    logical_or_call_result_22507 = invoke(stypy.reporting.localization.Localization(__file__, 604, 15), logical_or_22493, *[result_lt_22496, result_gt_22505], **kwargs_22506)
    
    # Processing the call keyword arguments (line 604)
    kwargs_22508 = {}
    # Getting the type of '_np' (line 604)
    _np_22490 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 604, 7), '_np', False)
    # Obtaining the member 'any' of a type (line 604)
    any_22491 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 604, 7), _np_22490, 'any')
    # Calling any(args, kwargs) (line 604)
    any_call_result_22509 = invoke(stypy.reporting.localization.Localization(__file__, 604, 7), any_22491, *[logical_or_call_result_22507], **kwargs_22508)
    
    # Testing the type of an if condition (line 604)
    if_condition_22510 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 604, 4), any_call_result_22509)
    # Assigning a type to the variable 'if_condition_22510' (line 604)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 604, 4), 'if_condition_22510', if_condition_22510)
    # SSA begins for if statement (line 604)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 605)
    # Processing the call arguments (line 605)
    str_22512 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 605, 25), 'str', 'Too large work array required -- computation cannot be performed with standard 32-bit LAPACK.')
    # Processing the call keyword arguments (line 605)
    kwargs_22513 = {}
    # Getting the type of 'ValueError' (line 605)
    ValueError_22511 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 605, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 605)
    ValueError_call_result_22514 = invoke(stypy.reporting.localization.Localization(__file__, 605, 14), ValueError_22511, *[str_22512], **kwargs_22513)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 605, 8), ValueError_call_result_22514, 'raise parameter', BaseException)
    # SSA join for if statement (line 604)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 607):
    
    # Call to astype(...): (line 607)
    # Processing the call arguments (line 607)
    # Getting the type of '_np' (line 607)
    _np_22517 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 607, 25), '_np', False)
    # Obtaining the member 'int32' of a type (line 607)
    int32_22518 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 607, 25), _np_22517, 'int32')
    # Processing the call keyword arguments (line 607)
    kwargs_22519 = {}
    # Getting the type of 'lwork' (line 607)
    lwork_22515 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 607, 12), 'lwork', False)
    # Obtaining the member 'astype' of a type (line 607)
    astype_22516 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 607, 12), lwork_22515, 'astype')
    # Calling astype(args, kwargs) (line 607)
    astype_call_result_22520 = invoke(stypy.reporting.localization.Localization(__file__, 607, 12), astype_22516, *[int32_22518], **kwargs_22519)
    
    # Assigning a type to the variable 'lwork' (line 607)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 607, 4), 'lwork', astype_call_result_22520)
    
    
    # Getting the type of 'lwork' (line 608)
    lwork_22521 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 608, 7), 'lwork')
    # Obtaining the member 'size' of a type (line 608)
    size_22522 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 608, 7), lwork_22521, 'size')
    int_22523 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 608, 21), 'int')
    # Applying the binary operator '==' (line 608)
    result_eq_22524 = python_operator(stypy.reporting.localization.Localization(__file__, 608, 7), '==', size_22522, int_22523)
    
    # Testing the type of an if condition (line 608)
    if_condition_22525 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 608, 4), result_eq_22524)
    # Assigning a type to the variable 'if_condition_22525' (line 608)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 608, 4), 'if_condition_22525', if_condition_22525)
    # SSA begins for if statement (line 608)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Obtaining the type of the subscript
    int_22526 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 609, 21), 'int')
    # Getting the type of 'lwork' (line 609)
    lwork_22527 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 609, 15), 'lwork')
    # Obtaining the member '__getitem__' of a type (line 609)
    getitem___22528 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 609, 15), lwork_22527, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 609)
    subscript_call_result_22529 = invoke(stypy.reporting.localization.Localization(__file__, 609, 15), getitem___22528, int_22526)
    
    # Assigning a type to the variable 'stypy_return_type' (line 609)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 609, 8), 'stypy_return_type', subscript_call_result_22529)
    # SSA join for if statement (line 608)
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'lwork' (line 610)
    lwork_22530 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 610, 11), 'lwork')
    # Assigning a type to the variable 'stypy_return_type' (line 610)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 610, 4), 'stypy_return_type', lwork_22530)
    
    # ################# End of '_compute_lwork(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_compute_lwork' in the type store
    # Getting the type of 'stypy_return_type' (line 566)
    stypy_return_type_22531 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 566, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_22531)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_compute_lwork'
    return stypy_return_type_22531

# Assigning a type to the variable '_compute_lwork' (line 566)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 566, 0), '_compute_lwork', _compute_lwork)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
