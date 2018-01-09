
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: #******************************************************************************
2: #   Copyright (C) 2013 Kenneth L. Ho
3: #
4: #   Redistribution and use in source and binary forms, with or without
5: #   modification, are permitted provided that the following conditions are met:
6: #
7: #   Redistributions of source code must retain the above copyright notice, this
8: #   list of conditions and the following disclaimer. Redistributions in binary
9: #   form must reproduce the above copyright notice, this list of conditions and
10: #   the following disclaimer in the documentation and/or other materials
11: #   provided with the distribution.
12: #
13: #   None of the names of the copyright holders may be used to endorse or
14: #   promote products derived from this software without specific prior written
15: #   permission.
16: #
17: #   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
18: #   AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
19: #   IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
20: #   ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
21: #   LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
22: #   CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
23: #   SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
24: #   INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
25: #   CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
26: #   ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
27: #   POSSIBILITY OF SUCH DAMAGE.
28: #******************************************************************************
29: 
30: # Python module for interfacing with `id_dist`.
31: 
32: r'''
33: ======================================================================
34: Interpolative matrix decomposition (:mod:`scipy.linalg.interpolative`)
35: ======================================================================
36: 
37: .. moduleauthor:: Kenneth L. Ho <klho@stanford.edu>
38: 
39: .. versionadded:: 0.13
40: 
41: .. currentmodule:: scipy.linalg.interpolative
42: 
43: An interpolative decomposition (ID) of a matrix :math:`A \in
44: \mathbb{C}^{m \times n}` of rank :math:`k \leq \min \{ m, n \}` is a
45: factorization
46: 
47: .. math::
48:   A \Pi =
49:   \begin{bmatrix}
50:    A \Pi_{1} & A \Pi_{2}
51:   \end{bmatrix} =
52:   A \Pi_{1}
53:   \begin{bmatrix}
54:    I & T
55:   \end{bmatrix},
56: 
57: where :math:`\Pi = [\Pi_{1}, \Pi_{2}]` is a permutation matrix with
58: :math:`\Pi_{1} \in \{ 0, 1 \}^{n \times k}`, i.e., :math:`A \Pi_{2} =
59: A \Pi_{1} T`. This can equivalently be written as :math:`A = BP`,
60: where :math:`B = A \Pi_{1}` and :math:`P = [I, T] \Pi^{\mathsf{T}}`
61: are the *skeleton* and *interpolation matrices*, respectively.
62: 
63: If :math:`A` does not have exact rank :math:`k`, then there exists an
64: approximation in the form of an ID such that :math:`A = BP + E`, where
65: :math:`\| E \| \sim \sigma_{k + 1}` is on the order of the :math:`(k +
66: 1)`-th largest singular value of :math:`A`. Note that :math:`\sigma_{k
67: + 1}` is the best possible error for a rank-:math:`k` approximation
68: and, in fact, is achieved by the singular value decomposition (SVD)
69: :math:`A \approx U S V^{*}`, where :math:`U \in \mathbb{C}^{m \times
70: k}` and :math:`V \in \mathbb{C}^{n \times k}` have orthonormal columns
71: and :math:`S = \mathop{\mathrm{diag}} (\sigma_{i}) \in \mathbb{C}^{k
72: \times k}` is diagonal with nonnegative entries. The principal
73: advantages of using an ID over an SVD are that:
74: 
75: - it is cheaper to construct;
76: - it preserves the structure of :math:`A`; and
77: - it is more efficient to compute with in light of the identity submatrix of :math:`P`.
78: 
79: Routines
80: ========
81: 
82: Main functionality:
83: 
84: .. autosummary::
85:    :toctree: generated/
86: 
87:    interp_decomp
88:    reconstruct_matrix_from_id
89:    reconstruct_interp_matrix
90:    reconstruct_skel_matrix
91:    id_to_svd
92:    svd
93:    estimate_spectral_norm
94:    estimate_spectral_norm_diff
95:    estimate_rank
96: 
97: Support functions:
98: 
99: .. autosummary::
100:    :toctree: generated/
101: 
102:    seed
103:    rand
104: 
105: 
106: References
107: ==========
108: 
109: This module uses the ID software package [1]_ by Martinsson, Rokhlin,
110: Shkolnisky, and Tygert, which is a Fortran library for computing IDs
111: using various algorithms, including the rank-revealing QR approach of
112: [2]_ and the more recent randomized methods described in [3]_, [4]_,
113: and [5]_. This module exposes its functionality in a way convenient
114: for Python users. Note that this module does not add any functionality
115: beyond that of organizing a simpler and more consistent interface.
116: 
117: We advise the user to consult also the `documentation for the ID package
118: <http://tygert.com/id_doc.4.pdf>`_.
119: 
120: .. [1] P.G. Martinsson, V. Rokhlin, Y. Shkolnisky, M. Tygert. "ID: a
121:     software package for low-rank approximation of matrices via interpolative
122:     decompositions, version 0.2." http://tygert.com/id_doc.4.pdf.
123: 
124: .. [2] H. Cheng, Z. Gimbutas, P.G. Martinsson, V. Rokhlin. "On the
125:     compression of low rank matrices." *SIAM J. Sci. Comput.* 26 (4): 1389--1404,
126:     2005. `doi:10.1137/030602678 <http://dx.doi.org/10.1137/030602678>`_.
127: 
128: .. [3] E. Liberty, F. Woolfe, P.G. Martinsson, V. Rokhlin, M.
129:     Tygert. "Randomized algorithms for the low-rank approximation of matrices."
130:     *Proc. Natl. Acad. Sci. U.S.A.* 104 (51): 20167--20172, 2007.
131:     `doi:10.1073/pnas.0709640104 <http://dx.doi.org/10.1073/pnas.0709640104>`_.
132: 
133: .. [4] P.G. Martinsson, V. Rokhlin, M. Tygert. "A randomized
134:     algorithm for the decomposition of matrices." *Appl. Comput. Harmon. Anal.* 30
135:     (1): 47--68,  2011. `doi:10.1016/j.acha.2010.02.003
136:     <http://dx.doi.org/10.1016/j.acha.2010.02.003>`_.
137: 
138: .. [5] F. Woolfe, E. Liberty, V. Rokhlin, M. Tygert. "A fast
139:     randomized algorithm for the approximation of matrices." *Appl. Comput.
140:     Harmon. Anal.* 25 (3): 335--366, 2008. `doi:10.1016/j.acha.2007.12.002
141:     <http://dx.doi.org/10.1016/j.acha.2007.12.002>`_.
142: 
143: 
144: Tutorial
145: ========
146: 
147: Initializing
148: ------------
149: 
150: The first step is to import :mod:`scipy.linalg.interpolative` by issuing the
151: command:
152: 
153: >>> import scipy.linalg.interpolative as sli
154: 
155: Now let's build a matrix. For this, we consider a Hilbert matrix, which is well
156: know to have low rank:
157: 
158: >>> from scipy.linalg import hilbert
159: >>> n = 1000
160: >>> A = hilbert(n)
161: 
162: We can also do this explicitly via:
163: 
164: >>> import numpy as np
165: >>> n = 1000
166: >>> A = np.empty((n, n), order='F')
167: >>> for j in range(n):
168: >>>     for i in range(m):
169: >>>         A[i,j] = 1. / (i + j + 1)
170: 
171: Note the use of the flag ``order='F'`` in :func:`numpy.empty`. This
172: instantiates the matrix in Fortran-contiguous order and is important for
173: avoiding data copying when passing to the backend.
174: 
175: We then define multiplication routines for the matrix by regarding it as a
176: :class:`scipy.sparse.linalg.LinearOperator`:
177: 
178: >>> from scipy.sparse.linalg import aslinearoperator
179: >>> L = aslinearoperator(A)
180: 
181: This automatically sets up methods describing the action of the matrix and its
182: adjoint on a vector.
183: 
184: Computing an ID
185: ---------------
186: 
187: We have several choices of algorithm to compute an ID. These fall largely
188: according to two dichotomies:
189: 
190: 1. how the matrix is represented, i.e., via its entries or via its action on a
191:    vector; and
192: 2. whether to approximate it to a fixed relative precision or to a fixed rank.
193: 
194: We step through each choice in turn below.
195: 
196: In all cases, the ID is represented by three parameters:
197: 
198: 1. a rank ``k``;
199: 2. an index array ``idx``; and
200: 3. interpolation coefficients ``proj``.
201: 
202: The ID is specified by the relation
203: ``np.dot(A[:,idx[:k]], proj) == A[:,idx[k:]]``.
204: 
205: From matrix entries
206: ...................
207: 
208: We first consider a matrix given in terms of its entries.
209: 
210: To compute an ID to a fixed precision, type:
211: 
212: >>> k, idx, proj = sli.interp_decomp(A, eps)
213: 
214: where ``eps < 1`` is the desired precision.
215: 
216: To compute an ID to a fixed rank, use:
217: 
218: >>> idx, proj = sli.interp_decomp(A, k)
219: 
220: where ``k >= 1`` is the desired rank.
221: 
222: Both algorithms use random sampling and are usually faster than the
223: corresponding older, deterministic algorithms, which can be accessed via the
224: commands:
225: 
226: >>> k, idx, proj = sli.interp_decomp(A, eps, rand=False)
227: 
228: and:
229: 
230: >>> idx, proj = sli.interp_decomp(A, k, rand=False)
231: 
232: respectively.
233: 
234: From matrix action
235: ..................
236: 
237: Now consider a matrix given in terms of its action on a vector as a
238: :class:`scipy.sparse.linalg.LinearOperator`.
239: 
240: To compute an ID to a fixed precision, type:
241: 
242: >>> k, idx, proj = sli.interp_decomp(L, eps)
243: 
244: To compute an ID to a fixed rank, use:
245: 
246: >>> idx, proj = sli.interp_decomp(L, k)
247: 
248: These algorithms are randomized.
249: 
250: Reconstructing an ID
251: --------------------
252: 
253: The ID routines above do not output the skeleton and interpolation matrices
254: explicitly but instead return the relevant information in a more compact (and
255: sometimes more useful) form. To build these matrices, write:
256: 
257: >>> B = sli.reconstruct_skel_matrix(A, k, idx)
258: 
259: for the skeleton matrix and:
260: 
261: >>> P = sli.reconstruct_interp_matrix(idx, proj)
262: 
263: for the interpolation matrix. The ID approximation can then be computed as:
264: 
265: >>> C = np.dot(B, P)
266: 
267: This can also be constructed directly using:
268: 
269: >>> C = sli.reconstruct_matrix_from_id(B, idx, proj)
270: 
271: without having to first compute ``P``.
272: 
273: Alternatively, this can be done explicitly as well using:
274: 
275: >>> B = A[:,idx[:k]]
276: >>> P = np.hstack([np.eye(k), proj])[:,np.argsort(idx)]
277: >>> C = np.dot(B, P)
278: 
279: Computing an SVD
280: ----------------
281: 
282: An ID can be converted to an SVD via the command:
283: 
284: >>> U, S, V = sli.id_to_svd(B, idx, proj)
285: 
286: The SVD approximation is then:
287: 
288: >>> C = np.dot(U, np.dot(np.diag(S), np.dot(V.conj().T)))
289: 
290: The SVD can also be computed "fresh" by combining both the ID and conversion
291: steps into one command. Following the various ID algorithms above, there are
292: correspondingly various SVD algorithms that one can employ.
293: 
294: From matrix entries
295: ...................
296: 
297: We consider first SVD algorithms for a matrix given in terms of its entries.
298: 
299: To compute an SVD to a fixed precision, type:
300: 
301: >>> U, S, V = sli.svd(A, eps)
302: 
303: To compute an SVD to a fixed rank, use:
304: 
305: >>> U, S, V = sli.svd(A, k)
306: 
307: Both algorithms use random sampling; for the determinstic versions, issue the
308: keyword ``rand=False`` as above.
309: 
310: From matrix action
311: ..................
312: 
313: Now consider a matrix given in terms of its action on a vector.
314: 
315: To compute an SVD to a fixed precision, type:
316: 
317: >>> U, S, V = sli.svd(L, eps)
318: 
319: To compute an SVD to a fixed rank, use:
320: 
321: >>> U, S, V = sli.svd(L, k)
322: 
323: Utility routines
324: ----------------
325: 
326: Several utility routines are also available.
327: 
328: To estimate the spectral norm of a matrix, use:
329: 
330: >>> snorm = sli.estimate_spectral_norm(A)
331: 
332: This algorithm is based on the randomized power method and thus requires only
333: matrix-vector products. The number of iterations to take can be set using the
334: keyword ``its`` (default: ``its=20``). The matrix is interpreted as a
335: :class:`scipy.sparse.linalg.LinearOperator`, but it is also valid to supply it
336: as a :class:`numpy.ndarray`, in which case it is trivially converted using
337: :func:`scipy.sparse.linalg.aslinearoperator`.
338: 
339: The same algorithm can also estimate the spectral norm of the difference of two
340: matrices ``A1`` and ``A2`` as follows:
341: 
342: >>> diff = sli.estimate_spectral_norm_diff(A1, A2)
343: 
344: This is often useful for checking the accuracy of a matrix approximation.
345: 
346: Some routines in :mod:`scipy.linalg.interpolative` require estimating the rank
347: of a matrix as well. This can be done with either:
348: 
349: >>> k = sli.estimate_rank(A, eps)
350: 
351: or:
352: 
353: >>> k = sli.estimate_rank(L, eps)
354: 
355: depending on the representation. The parameter ``eps`` controls the definition
356: of the numerical rank.
357: 
358: Finally, the random number generation required for all randomized routines can
359: be controlled via :func:`scipy.linalg.interpolative.seed`. To reset the seed
360: values to their original values, use:
361: 
362: >>> sli.seed('default')
363: 
364: To specify the seed values, use:
365: 
366: >>> sli.seed(s)
367: 
368: where ``s`` must be an integer or array of 55 floats. If an integer, the array
369: of floats is obtained by using `np.random.rand` with the given integer seed.
370: 
371: To simply generate some random numbers, type:
372: 
373: >>> sli.rand(n)
374: 
375: where ``n`` is the number of random numbers to generate.
376: 
377: Remarks
378: -------
379: 
380: The above functions all automatically detect the appropriate interface and work
381: with both real and complex data types, passing input arguments to the proper
382: backend routine.
383: 
384: '''
385: 
386: import scipy.linalg._interpolative_backend as backend
387: import numpy as np
388: 
389: _DTYPE_ERROR = ValueError("invalid input dtype (input must be float64 or complex128)")
390: _TYPE_ERROR = TypeError("invalid input type (must be array or LinearOperator)")
391: 
392: 
393: def _is_real(A):
394:     try:
395:         if A.dtype == np.complex128:
396:             return False
397:         elif A.dtype == np.float64:
398:             return True
399:         else:
400:             raise _DTYPE_ERROR
401:     except AttributeError:
402:         raise _TYPE_ERROR
403: 
404: 
405: def seed(seed=None):
406:     '''
407:     Seed the internal random number generator used in this ID package.
408: 
409:     The generator is a lagged Fibonacci method with 55-element internal state.
410: 
411:     Parameters
412:     ----------
413:     seed : int, sequence, 'default', optional
414:         If 'default', the random seed is reset to a default value.
415: 
416:         If `seed` is a sequence containing 55 floating-point numbers
417:         in range [0,1], these are used to set the internal state of
418:         the generator.
419: 
420:         If the value is an integer, the internal state is obtained
421:         from `numpy.random.RandomState` (MT19937) with the integer
422:         used as the initial seed.
423: 
424:         If `seed` is omitted (None), `numpy.random` is used to
425:         initialize the generator.
426: 
427:     '''
428:     # For details, see :func:`backend.id_srand`, :func:`backend.id_srandi`,
429:     # and :func:`backend.id_srando`.
430: 
431:     if isinstance(seed, str) and seed == 'default':
432:         backend.id_srando()
433:     elif hasattr(seed, '__len__'):
434:         state = np.asfortranarray(seed, dtype=float)
435:         if state.shape != (55,):
436:             raise ValueError("invalid input size")
437:         elif state.min() < 0 or state.max() > 1:
438:             raise ValueError("values not in range [0,1]")
439:         backend.id_srandi(state)
440:     elif seed is None:
441:         backend.id_srandi(np.random.rand(55))
442:     else:
443:         rnd = np.random.RandomState(seed)
444:         backend.id_srandi(rnd.rand(55))
445: 
446: 
447: def rand(*shape):
448:     '''
449:     Generate standard uniform pseudorandom numbers via a very efficient lagged
450:     Fibonacci method.
451: 
452:     This routine is used for all random number generation in this package and
453:     can affect ID and SVD results.
454: 
455:     Parameters
456:     ----------
457:     shape
458:         Shape of output array
459: 
460:     '''
461:     # For details, see :func:`backend.id_srand`, and :func:`backend.id_srando`.
462:     return backend.id_srand(np.prod(shape)).reshape(shape)
463: 
464: 
465: def interp_decomp(A, eps_or_k, rand=True):
466:     '''
467:     Compute ID of a matrix.
468: 
469:     An ID of a matrix `A` is a factorization defined by a rank `k`, a column
470:     index array `idx`, and interpolation coefficients `proj` such that::
471: 
472:         numpy.dot(A[:,idx[:k]], proj) = A[:,idx[k:]]
473: 
474:     The original matrix can then be reconstructed as::
475: 
476:         numpy.hstack([A[:,idx[:k]],
477:                                     numpy.dot(A[:,idx[:k]], proj)]
478:                                 )[:,numpy.argsort(idx)]
479: 
480:     or via the routine :func:`reconstruct_matrix_from_id`. This can
481:     equivalently be written as::
482: 
483:         numpy.dot(A[:,idx[:k]],
484:                             numpy.hstack([numpy.eye(k), proj])
485:                           )[:,np.argsort(idx)]
486: 
487:     in terms of the skeleton and interpolation matrices::
488: 
489:         B = A[:,idx[:k]]
490: 
491:     and::
492: 
493:         P = numpy.hstack([numpy.eye(k), proj])[:,np.argsort(idx)]
494: 
495:     respectively. See also :func:`reconstruct_interp_matrix` and
496:     :func:`reconstruct_skel_matrix`.
497: 
498:     The ID can be computed to any relative precision or rank (depending on the
499:     value of `eps_or_k`). If a precision is specified (`eps_or_k < 1`), then
500:     this function has the output signature::
501: 
502:         k, idx, proj = interp_decomp(A, eps_or_k)
503: 
504:     Otherwise, if a rank is specified (`eps_or_k >= 1`), then the output
505:     signature is::
506: 
507:         idx, proj = interp_decomp(A, eps_or_k)
508: 
509:     ..  This function automatically detects the form of the input parameters
510:         and passes them to the appropriate backend. For details, see
511:         :func:`backend.iddp_id`, :func:`backend.iddp_aid`,
512:         :func:`backend.iddp_rid`, :func:`backend.iddr_id`,
513:         :func:`backend.iddr_aid`, :func:`backend.iddr_rid`,
514:         :func:`backend.idzp_id`, :func:`backend.idzp_aid`,
515:         :func:`backend.idzp_rid`, :func:`backend.idzr_id`,
516:         :func:`backend.idzr_aid`, and :func:`backend.idzr_rid`.
517: 
518:     Parameters
519:     ----------
520:     A : :class:`numpy.ndarray` or :class:`scipy.sparse.linalg.LinearOperator` with `rmatvec`
521:         Matrix to be factored
522:     eps_or_k : float or int
523:         Relative error (if `eps_or_k < 1`) or rank (if `eps_or_k >= 1`) of
524:         approximation.
525:     rand : bool, optional
526:         Whether to use random sampling if `A` is of type :class:`numpy.ndarray`
527:         (randomized algorithms are always used if `A` is of type
528:         :class:`scipy.sparse.linalg.LinearOperator`).
529: 
530:     Returns
531:     -------
532:     k : int
533:         Rank required to achieve specified relative precision if
534:         `eps_or_k < 1`.
535:     idx : :class:`numpy.ndarray`
536:         Column index array.
537:     proj : :class:`numpy.ndarray`
538:         Interpolation coefficients.
539:     '''
540:     from scipy.sparse.linalg import LinearOperator
541: 
542:     real = _is_real(A)
543: 
544:     if isinstance(A, np.ndarray):
545:         if eps_or_k < 1:
546:             eps = eps_or_k
547:             if rand:
548:                 if real:
549:                     k, idx, proj = backend.iddp_aid(eps, A)
550:                 else:
551:                     k, idx, proj = backend.idzp_aid(eps, A)
552:             else:
553:                 if real:
554:                     k, idx, proj = backend.iddp_id(eps, A)
555:                 else:
556:                     k, idx, proj = backend.idzp_id(eps, A)
557:             return k, idx - 1, proj
558:         else:
559:             k = int(eps_or_k)
560:             if rand:
561:                 if real:
562:                     idx, proj = backend.iddr_aid(A, k)
563:                 else:
564:                     idx, proj = backend.idzr_aid(A, k)
565:             else:
566:                 if real:
567:                     idx, proj = backend.iddr_id(A, k)
568:                 else:
569:                     idx, proj = backend.idzr_id(A, k)
570:             return idx - 1, proj
571:     elif isinstance(A, LinearOperator):
572:         m, n = A.shape
573:         matveca = A.rmatvec
574:         if eps_or_k < 1:
575:             eps = eps_or_k
576:             if real:
577:                 k, idx, proj = backend.iddp_rid(eps, m, n, matveca)
578:             else:
579:                 k, idx, proj = backend.idzp_rid(eps, m, n, matveca)
580:             return k, idx - 1, proj
581:         else:
582:             k = int(eps_or_k)
583:             if real:
584:                 idx, proj = backend.iddr_rid(m, n, matveca, k)
585:             else:
586:                 idx, proj = backend.idzr_rid(m, n, matveca, k)
587:             return idx - 1, proj
588:     else:
589:         raise _TYPE_ERROR
590: 
591: 
592: def reconstruct_matrix_from_id(B, idx, proj):
593:     '''
594:     Reconstruct matrix from its ID.
595: 
596:     A matrix `A` with skeleton matrix `B` and ID indices and coefficients `idx`
597:     and `proj`, respectively, can be reconstructed as::
598: 
599:         numpy.hstack([B, numpy.dot(B, proj)])[:,numpy.argsort(idx)]
600: 
601:     See also :func:`reconstruct_interp_matrix` and
602:     :func:`reconstruct_skel_matrix`.
603: 
604:     ..  This function automatically detects the matrix data type and calls the
605:         appropriate backend. For details, see :func:`backend.idd_reconid` and
606:         :func:`backend.idz_reconid`.
607: 
608:     Parameters
609:     ----------
610:     B : :class:`numpy.ndarray`
611:         Skeleton matrix.
612:     idx : :class:`numpy.ndarray`
613:         Column index array.
614:     proj : :class:`numpy.ndarray`
615:         Interpolation coefficients.
616: 
617:     Returns
618:     -------
619:     :class:`numpy.ndarray`
620:         Reconstructed matrix.
621:     '''
622:     if _is_real(B):
623:         return backend.idd_reconid(B, idx + 1, proj)
624:     else:
625:         return backend.idz_reconid(B, idx + 1, proj)
626: 
627: 
628: def reconstruct_interp_matrix(idx, proj):
629:     '''
630:     Reconstruct interpolation matrix from ID.
631: 
632:     The interpolation matrix can be reconstructed from the ID indices and
633:     coefficients `idx` and `proj`, respectively, as::
634: 
635:         P = numpy.hstack([numpy.eye(proj.shape[0]), proj])[:,numpy.argsort(idx)]
636: 
637:     The original matrix can then be reconstructed from its skeleton matrix `B`
638:     via::
639: 
640:         numpy.dot(B, P)
641: 
642:     See also :func:`reconstruct_matrix_from_id` and
643:     :func:`reconstruct_skel_matrix`.
644: 
645:     ..  This function automatically detects the matrix data type and calls the
646:         appropriate backend. For details, see :func:`backend.idd_reconint` and
647:         :func:`backend.idz_reconint`.
648: 
649:     Parameters
650:     ----------
651:     idx : :class:`numpy.ndarray`
652:         Column index array.
653:     proj : :class:`numpy.ndarray`
654:         Interpolation coefficients.
655: 
656:     Returns
657:     -------
658:     :class:`numpy.ndarray`
659:         Interpolation matrix.
660:     '''
661:     if _is_real(proj):
662:         return backend.idd_reconint(idx + 1, proj)
663:     else:
664:         return backend.idz_reconint(idx + 1, proj)
665: 
666: 
667: def reconstruct_skel_matrix(A, k, idx):
668:     '''
669:     Reconstruct skeleton matrix from ID.
670: 
671:     The skeleton matrix can be reconstructed from the original matrix `A` and its
672:     ID rank and indices `k` and `idx`, respectively, as::
673: 
674:         B = A[:,idx[:k]]
675: 
676:     The original matrix can then be reconstructed via::
677: 
678:         numpy.hstack([B, numpy.dot(B, proj)])[:,numpy.argsort(idx)]
679: 
680:     See also :func:`reconstruct_matrix_from_id` and
681:     :func:`reconstruct_interp_matrix`.
682: 
683:     ..  This function automatically detects the matrix data type and calls the
684:         appropriate backend. For details, see :func:`backend.idd_copycols` and
685:         :func:`backend.idz_copycols`.
686: 
687:     Parameters
688:     ----------
689:     A : :class:`numpy.ndarray`
690:         Original matrix.
691:     k : int
692:         Rank of ID.
693:     idx : :class:`numpy.ndarray`
694:         Column index array.
695: 
696:     Returns
697:     -------
698:     :class:`numpy.ndarray`
699:         Skeleton matrix.
700:     '''
701:     if _is_real(A):
702:         return backend.idd_copycols(A, k, idx + 1)
703:     else:
704:         return backend.idz_copycols(A, k, idx + 1)
705: 
706: 
707: def id_to_svd(B, idx, proj):
708:     '''
709:     Convert ID to SVD.
710: 
711:     The SVD reconstruction of a matrix with skeleton matrix `B` and ID indices and
712:     coefficients `idx` and `proj`, respectively, is::
713: 
714:         U, S, V = id_to_svd(B, idx, proj)
715:         A = numpy.dot(U, numpy.dot(numpy.diag(S), V.conj().T))
716: 
717:     See also :func:`svd`.
718: 
719:     ..  This function automatically detects the matrix data type and calls the
720:         appropriate backend. For details, see :func:`backend.idd_id2svd` and
721:         :func:`backend.idz_id2svd`.
722: 
723:     Parameters
724:     ----------
725:     B : :class:`numpy.ndarray`
726:         Skeleton matrix.
727:     idx : :class:`numpy.ndarray`
728:         Column index array.
729:     proj : :class:`numpy.ndarray`
730:         Interpolation coefficients.
731: 
732:     Returns
733:     -------
734:     U : :class:`numpy.ndarray`
735:         Left singular vectors.
736:     S : :class:`numpy.ndarray`
737:         Singular values.
738:     V : :class:`numpy.ndarray`
739:         Right singular vectors.
740:     '''
741:     if _is_real(B):
742:         U, V, S = backend.idd_id2svd(B, idx + 1, proj)
743:     else:
744:         U, V, S = backend.idz_id2svd(B, idx + 1, proj)
745:     return U, S, V
746: 
747: 
748: def estimate_spectral_norm(A, its=20):
749:     '''
750:     Estimate spectral norm of a matrix by the randomized power method.
751: 
752:     ..  This function automatically detects the matrix data type and calls the
753:         appropriate backend. For details, see :func:`backend.idd_snorm` and
754:         :func:`backend.idz_snorm`.
755: 
756:     Parameters
757:     ----------
758:     A : :class:`scipy.sparse.linalg.LinearOperator`
759:         Matrix given as a :class:`scipy.sparse.linalg.LinearOperator` with the
760:         `matvec` and `rmatvec` methods (to apply the matrix and its adjoint).
761:     its : int, optional
762:         Number of power method iterations.
763: 
764:     Returns
765:     -------
766:     float
767:         Spectral norm estimate.
768:     '''
769:     from scipy.sparse.linalg import aslinearoperator
770:     A = aslinearoperator(A)
771:     m, n = A.shape
772:     matvec = lambda x: A. matvec(x)
773:     matveca = lambda x: A.rmatvec(x)
774:     if _is_real(A):
775:         return backend.idd_snorm(m, n, matveca, matvec, its=its)
776:     else:
777:         return backend.idz_snorm(m, n, matveca, matvec, its=its)
778: 
779: 
780: def estimate_spectral_norm_diff(A, B, its=20):
781:     '''
782:     Estimate spectral norm of the difference of two matrices by the randomized
783:     power method.
784: 
785:     ..  This function automatically detects the matrix data type and calls the
786:         appropriate backend. For details, see :func:`backend.idd_diffsnorm` and
787:         :func:`backend.idz_diffsnorm`.
788: 
789:     Parameters
790:     ----------
791:     A : :class:`scipy.sparse.linalg.LinearOperator`
792:         First matrix given as a :class:`scipy.sparse.linalg.LinearOperator` with the
793:         `matvec` and `rmatvec` methods (to apply the matrix and its adjoint).
794:     B : :class:`scipy.sparse.linalg.LinearOperator`
795:         Second matrix given as a :class:`scipy.sparse.linalg.LinearOperator` with
796:         the `matvec` and `rmatvec` methods (to apply the matrix and its adjoint).
797:     its : int, optional
798:         Number of power method iterations.
799: 
800:     Returns
801:     -------
802:     float
803:         Spectral norm estimate of matrix difference.
804:     '''
805:     from scipy.sparse.linalg import aslinearoperator
806:     A = aslinearoperator(A)
807:     B = aslinearoperator(B)
808:     m, n = A.shape
809:     matvec1 = lambda x: A. matvec(x)
810:     matveca1 = lambda x: A.rmatvec(x)
811:     matvec2 = lambda x: B. matvec(x)
812:     matveca2 = lambda x: B.rmatvec(x)
813:     if _is_real(A):
814:         return backend.idd_diffsnorm(
815:             m, n, matveca1, matveca2, matvec1, matvec2, its=its)
816:     else:
817:         return backend.idz_diffsnorm(
818:             m, n, matveca1, matveca2, matvec1, matvec2, its=its)
819: 
820: 
821: def svd(A, eps_or_k, rand=True):
822:     '''
823:     Compute SVD of a matrix via an ID.
824: 
825:     An SVD of a matrix `A` is a factorization::
826: 
827:         A = numpy.dot(U, numpy.dot(numpy.diag(S), V.conj().T))
828: 
829:     where `U` and `V` have orthonormal columns and `S` is nonnegative.
830: 
831:     The SVD can be computed to any relative precision or rank (depending on the
832:     value of `eps_or_k`).
833: 
834:     See also :func:`interp_decomp` and :func:`id_to_svd`.
835: 
836:     ..  This function automatically detects the form of the input parameters and
837:         passes them to the appropriate backend. For details, see
838:         :func:`backend.iddp_svd`, :func:`backend.iddp_asvd`,
839:         :func:`backend.iddp_rsvd`, :func:`backend.iddr_svd`,
840:         :func:`backend.iddr_asvd`, :func:`backend.iddr_rsvd`,
841:         :func:`backend.idzp_svd`, :func:`backend.idzp_asvd`,
842:         :func:`backend.idzp_rsvd`, :func:`backend.idzr_svd`,
843:         :func:`backend.idzr_asvd`, and :func:`backend.idzr_rsvd`.
844: 
845:     Parameters
846:     ----------
847:     A : :class:`numpy.ndarray` or :class:`scipy.sparse.linalg.LinearOperator`
848:         Matrix to be factored, given as either a :class:`numpy.ndarray` or a
849:         :class:`scipy.sparse.linalg.LinearOperator` with the `matvec` and
850:         `rmatvec` methods (to apply the matrix and its adjoint).
851:     eps_or_k : float or int
852:         Relative error (if `eps_or_k < 1`) or rank (if `eps_or_k >= 1`) of
853:         approximation.
854:     rand : bool, optional
855:         Whether to use random sampling if `A` is of type :class:`numpy.ndarray`
856:         (randomized algorithms are always used if `A` is of type
857:         :class:`scipy.sparse.linalg.LinearOperator`).
858: 
859:     Returns
860:     -------
861:     U : :class:`numpy.ndarray`
862:         Left singular vectors.
863:     S : :class:`numpy.ndarray`
864:         Singular values.
865:     V : :class:`numpy.ndarray`
866:         Right singular vectors.
867:     '''
868:     from scipy.sparse.linalg import LinearOperator
869: 
870:     real = _is_real(A)
871: 
872:     if isinstance(A, np.ndarray):
873:         if eps_or_k < 1:
874:             eps = eps_or_k
875:             if rand:
876:                 if real:
877:                     U, V, S = backend.iddp_asvd(eps, A)
878:                 else:
879:                     U, V, S = backend.idzp_asvd(eps, A)
880:             else:
881:                 if real:
882:                     U, V, S = backend.iddp_svd(eps, A)
883:                 else:
884:                     U, V, S = backend.idzp_svd(eps, A)
885:         else:
886:             k = int(eps_or_k)
887:             if k > min(A.shape):
888:                 raise ValueError("Approximation rank %s exceeds min(A.shape) = "
889:                                  " %s " % (k, min(A.shape)))
890:             if rand:
891:                 if real:
892:                     U, V, S = backend.iddr_asvd(A, k)
893:                 else:
894:                     U, V, S = backend.idzr_asvd(A, k)
895:             else:
896:                 if real:
897:                     U, V, S = backend.iddr_svd(A, k)
898:                 else:
899:                     U, V, S = backend.idzr_svd(A, k)
900:     elif isinstance(A, LinearOperator):
901:         m, n = A.shape
902:         matvec = lambda x: A.matvec(x)
903:         matveca = lambda x: A.rmatvec(x)
904:         if eps_or_k < 1:
905:             eps = eps_or_k
906:             if real:
907:                 U, V, S = backend.iddp_rsvd(eps, m, n, matveca, matvec)
908:             else:
909:                 U, V, S = backend.idzp_rsvd(eps, m, n, matveca, matvec)
910:         else:
911:             k = int(eps_or_k)
912:             if real:
913:                 U, V, S = backend.iddr_rsvd(m, n, matveca, matvec, k)
914:             else:
915:                 U, V, S = backend.idzr_rsvd(m, n, matveca, matvec, k)
916:     else:
917:         raise _TYPE_ERROR
918:     return U, S, V
919: 
920: 
921: def estimate_rank(A, eps):
922:     '''
923:     Estimate matrix rank to a specified relative precision using randomized
924:     methods.
925: 
926:     The matrix `A` can be given as either a :class:`numpy.ndarray` or a
927:     :class:`scipy.sparse.linalg.LinearOperator`, with different algorithms used
928:     for each case. If `A` is of type :class:`numpy.ndarray`, then the output
929:     rank is typically about 8 higher than the actual numerical rank.
930: 
931:     ..  This function automatically detects the form of the input parameters and
932:         passes them to the appropriate backend. For details,
933:         see :func:`backend.idd_estrank`, :func:`backend.idd_findrank`,
934:         :func:`backend.idz_estrank`, and :func:`backend.idz_findrank`.
935: 
936:     Parameters
937:     ----------
938:     A : :class:`numpy.ndarray` or :class:`scipy.sparse.linalg.LinearOperator`
939:         Matrix whose rank is to be estimated, given as either a
940:         :class:`numpy.ndarray` or a :class:`scipy.sparse.linalg.LinearOperator`
941:         with the `rmatvec` method (to apply the matrix adjoint).
942:     eps : float
943:         Relative error for numerical rank definition.
944: 
945:     Returns
946:     -------
947:     int
948:         Estimated matrix rank.
949:     '''
950:     from scipy.sparse.linalg import LinearOperator
951: 
952:     real = _is_real(A)
953: 
954:     if isinstance(A, np.ndarray):
955:         if real:
956:             rank = backend.idd_estrank(eps, A)
957:         else:
958:             rank = backend.idz_estrank(eps, A)
959:         if rank == 0:
960:             # special return value for nearly full rank
961:             rank = min(A.shape)
962:         return rank
963:     elif isinstance(A, LinearOperator):
964:         m, n = A.shape
965:         matveca = A.rmatvec
966:         if real:
967:             return backend.idd_findrank(eps, m, n, matveca)
968:         else:
969:             return backend.idz_findrank(eps, m, n, matveca)
970:     else:
971:         raise _TYPE_ERROR
972: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_20872 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 384, (-1)), 'str', '\n======================================================================\nInterpolative matrix decomposition (:mod:`scipy.linalg.interpolative`)\n======================================================================\n\n.. moduleauthor:: Kenneth L. Ho <klho@stanford.edu>\n\n.. versionadded:: 0.13\n\n.. currentmodule:: scipy.linalg.interpolative\n\nAn interpolative decomposition (ID) of a matrix :math:`A \\in\n\\mathbb{C}^{m \\times n}` of rank :math:`k \\leq \\min \\{ m, n \\}` is a\nfactorization\n\n.. math::\n  A \\Pi =\n  \\begin{bmatrix}\n   A \\Pi_{1} & A \\Pi_{2}\n  \\end{bmatrix} =\n  A \\Pi_{1}\n  \\begin{bmatrix}\n   I & T\n  \\end{bmatrix},\n\nwhere :math:`\\Pi = [\\Pi_{1}, \\Pi_{2}]` is a permutation matrix with\n:math:`\\Pi_{1} \\in \\{ 0, 1 \\}^{n \\times k}`, i.e., :math:`A \\Pi_{2} =\nA \\Pi_{1} T`. This can equivalently be written as :math:`A = BP`,\nwhere :math:`B = A \\Pi_{1}` and :math:`P = [I, T] \\Pi^{\\mathsf{T}}`\nare the *skeleton* and *interpolation matrices*, respectively.\n\nIf :math:`A` does not have exact rank :math:`k`, then there exists an\napproximation in the form of an ID such that :math:`A = BP + E`, where\n:math:`\\| E \\| \\sim \\sigma_{k + 1}` is on the order of the :math:`(k +\n1)`-th largest singular value of :math:`A`. Note that :math:`\\sigma_{k\n+ 1}` is the best possible error for a rank-:math:`k` approximation\nand, in fact, is achieved by the singular value decomposition (SVD)\n:math:`A \\approx U S V^{*}`, where :math:`U \\in \\mathbb{C}^{m \\times\nk}` and :math:`V \\in \\mathbb{C}^{n \\times k}` have orthonormal columns\nand :math:`S = \\mathop{\\mathrm{diag}} (\\sigma_{i}) \\in \\mathbb{C}^{k\n\\times k}` is diagonal with nonnegative entries. The principal\nadvantages of using an ID over an SVD are that:\n\n- it is cheaper to construct;\n- it preserves the structure of :math:`A`; and\n- it is more efficient to compute with in light of the identity submatrix of :math:`P`.\n\nRoutines\n========\n\nMain functionality:\n\n.. autosummary::\n   :toctree: generated/\n\n   interp_decomp\n   reconstruct_matrix_from_id\n   reconstruct_interp_matrix\n   reconstruct_skel_matrix\n   id_to_svd\n   svd\n   estimate_spectral_norm\n   estimate_spectral_norm_diff\n   estimate_rank\n\nSupport functions:\n\n.. autosummary::\n   :toctree: generated/\n\n   seed\n   rand\n\n\nReferences\n==========\n\nThis module uses the ID software package [1]_ by Martinsson, Rokhlin,\nShkolnisky, and Tygert, which is a Fortran library for computing IDs\nusing various algorithms, including the rank-revealing QR approach of\n[2]_ and the more recent randomized methods described in [3]_, [4]_,\nand [5]_. This module exposes its functionality in a way convenient\nfor Python users. Note that this module does not add any functionality\nbeyond that of organizing a simpler and more consistent interface.\n\nWe advise the user to consult also the `documentation for the ID package\n<http://tygert.com/id_doc.4.pdf>`_.\n\n.. [1] P.G. Martinsson, V. Rokhlin, Y. Shkolnisky, M. Tygert. "ID: a\n    software package for low-rank approximation of matrices via interpolative\n    decompositions, version 0.2." http://tygert.com/id_doc.4.pdf.\n\n.. [2] H. Cheng, Z. Gimbutas, P.G. Martinsson, V. Rokhlin. "On the\n    compression of low rank matrices." *SIAM J. Sci. Comput.* 26 (4): 1389--1404,\n    2005. `doi:10.1137/030602678 <http://dx.doi.org/10.1137/030602678>`_.\n\n.. [3] E. Liberty, F. Woolfe, P.G. Martinsson, V. Rokhlin, M.\n    Tygert. "Randomized algorithms for the low-rank approximation of matrices."\n    *Proc. Natl. Acad. Sci. U.S.A.* 104 (51): 20167--20172, 2007.\n    `doi:10.1073/pnas.0709640104 <http://dx.doi.org/10.1073/pnas.0709640104>`_.\n\n.. [4] P.G. Martinsson, V. Rokhlin, M. Tygert. "A randomized\n    algorithm for the decomposition of matrices." *Appl. Comput. Harmon. Anal.* 30\n    (1): 47--68,  2011. `doi:10.1016/j.acha.2010.02.003\n    <http://dx.doi.org/10.1016/j.acha.2010.02.003>`_.\n\n.. [5] F. Woolfe, E. Liberty, V. Rokhlin, M. Tygert. "A fast\n    randomized algorithm for the approximation of matrices." *Appl. Comput.\n    Harmon. Anal.* 25 (3): 335--366, 2008. `doi:10.1016/j.acha.2007.12.002\n    <http://dx.doi.org/10.1016/j.acha.2007.12.002>`_.\n\n\nTutorial\n========\n\nInitializing\n------------\n\nThe first step is to import :mod:`scipy.linalg.interpolative` by issuing the\ncommand:\n\n>>> import scipy.linalg.interpolative as sli\n\nNow let\'s build a matrix. For this, we consider a Hilbert matrix, which is well\nknow to have low rank:\n\n>>> from scipy.linalg import hilbert\n>>> n = 1000\n>>> A = hilbert(n)\n\nWe can also do this explicitly via:\n\n>>> import numpy as np\n>>> n = 1000\n>>> A = np.empty((n, n), order=\'F\')\n>>> for j in range(n):\n>>>     for i in range(m):\n>>>         A[i,j] = 1. / (i + j + 1)\n\nNote the use of the flag ``order=\'F\'`` in :func:`numpy.empty`. This\ninstantiates the matrix in Fortran-contiguous order and is important for\navoiding data copying when passing to the backend.\n\nWe then define multiplication routines for the matrix by regarding it as a\n:class:`scipy.sparse.linalg.LinearOperator`:\n\n>>> from scipy.sparse.linalg import aslinearoperator\n>>> L = aslinearoperator(A)\n\nThis automatically sets up methods describing the action of the matrix and its\nadjoint on a vector.\n\nComputing an ID\n---------------\n\nWe have several choices of algorithm to compute an ID. These fall largely\naccording to two dichotomies:\n\n1. how the matrix is represented, i.e., via its entries or via its action on a\n   vector; and\n2. whether to approximate it to a fixed relative precision or to a fixed rank.\n\nWe step through each choice in turn below.\n\nIn all cases, the ID is represented by three parameters:\n\n1. a rank ``k``;\n2. an index array ``idx``; and\n3. interpolation coefficients ``proj``.\n\nThe ID is specified by the relation\n``np.dot(A[:,idx[:k]], proj) == A[:,idx[k:]]``.\n\nFrom matrix entries\n...................\n\nWe first consider a matrix given in terms of its entries.\n\nTo compute an ID to a fixed precision, type:\n\n>>> k, idx, proj = sli.interp_decomp(A, eps)\n\nwhere ``eps < 1`` is the desired precision.\n\nTo compute an ID to a fixed rank, use:\n\n>>> idx, proj = sli.interp_decomp(A, k)\n\nwhere ``k >= 1`` is the desired rank.\n\nBoth algorithms use random sampling and are usually faster than the\ncorresponding older, deterministic algorithms, which can be accessed via the\ncommands:\n\n>>> k, idx, proj = sli.interp_decomp(A, eps, rand=False)\n\nand:\n\n>>> idx, proj = sli.interp_decomp(A, k, rand=False)\n\nrespectively.\n\nFrom matrix action\n..................\n\nNow consider a matrix given in terms of its action on a vector as a\n:class:`scipy.sparse.linalg.LinearOperator`.\n\nTo compute an ID to a fixed precision, type:\n\n>>> k, idx, proj = sli.interp_decomp(L, eps)\n\nTo compute an ID to a fixed rank, use:\n\n>>> idx, proj = sli.interp_decomp(L, k)\n\nThese algorithms are randomized.\n\nReconstructing an ID\n--------------------\n\nThe ID routines above do not output the skeleton and interpolation matrices\nexplicitly but instead return the relevant information in a more compact (and\nsometimes more useful) form. To build these matrices, write:\n\n>>> B = sli.reconstruct_skel_matrix(A, k, idx)\n\nfor the skeleton matrix and:\n\n>>> P = sli.reconstruct_interp_matrix(idx, proj)\n\nfor the interpolation matrix. The ID approximation can then be computed as:\n\n>>> C = np.dot(B, P)\n\nThis can also be constructed directly using:\n\n>>> C = sli.reconstruct_matrix_from_id(B, idx, proj)\n\nwithout having to first compute ``P``.\n\nAlternatively, this can be done explicitly as well using:\n\n>>> B = A[:,idx[:k]]\n>>> P = np.hstack([np.eye(k), proj])[:,np.argsort(idx)]\n>>> C = np.dot(B, P)\n\nComputing an SVD\n----------------\n\nAn ID can be converted to an SVD via the command:\n\n>>> U, S, V = sli.id_to_svd(B, idx, proj)\n\nThe SVD approximation is then:\n\n>>> C = np.dot(U, np.dot(np.diag(S), np.dot(V.conj().T)))\n\nThe SVD can also be computed "fresh" by combining both the ID and conversion\nsteps into one command. Following the various ID algorithms above, there are\ncorrespondingly various SVD algorithms that one can employ.\n\nFrom matrix entries\n...................\n\nWe consider first SVD algorithms for a matrix given in terms of its entries.\n\nTo compute an SVD to a fixed precision, type:\n\n>>> U, S, V = sli.svd(A, eps)\n\nTo compute an SVD to a fixed rank, use:\n\n>>> U, S, V = sli.svd(A, k)\n\nBoth algorithms use random sampling; for the determinstic versions, issue the\nkeyword ``rand=False`` as above.\n\nFrom matrix action\n..................\n\nNow consider a matrix given in terms of its action on a vector.\n\nTo compute an SVD to a fixed precision, type:\n\n>>> U, S, V = sli.svd(L, eps)\n\nTo compute an SVD to a fixed rank, use:\n\n>>> U, S, V = sli.svd(L, k)\n\nUtility routines\n----------------\n\nSeveral utility routines are also available.\n\nTo estimate the spectral norm of a matrix, use:\n\n>>> snorm = sli.estimate_spectral_norm(A)\n\nThis algorithm is based on the randomized power method and thus requires only\nmatrix-vector products. The number of iterations to take can be set using the\nkeyword ``its`` (default: ``its=20``). The matrix is interpreted as a\n:class:`scipy.sparse.linalg.LinearOperator`, but it is also valid to supply it\nas a :class:`numpy.ndarray`, in which case it is trivially converted using\n:func:`scipy.sparse.linalg.aslinearoperator`.\n\nThe same algorithm can also estimate the spectral norm of the difference of two\nmatrices ``A1`` and ``A2`` as follows:\n\n>>> diff = sli.estimate_spectral_norm_diff(A1, A2)\n\nThis is often useful for checking the accuracy of a matrix approximation.\n\nSome routines in :mod:`scipy.linalg.interpolative` require estimating the rank\nof a matrix as well. This can be done with either:\n\n>>> k = sli.estimate_rank(A, eps)\n\nor:\n\n>>> k = sli.estimate_rank(L, eps)\n\ndepending on the representation. The parameter ``eps`` controls the definition\nof the numerical rank.\n\nFinally, the random number generation required for all randomized routines can\nbe controlled via :func:`scipy.linalg.interpolative.seed`. To reset the seed\nvalues to their original values, use:\n\n>>> sli.seed(\'default\')\n\nTo specify the seed values, use:\n\n>>> sli.seed(s)\n\nwhere ``s`` must be an integer or array of 55 floats. If an integer, the array\nof floats is obtained by using `np.random.rand` with the given integer seed.\n\nTo simply generate some random numbers, type:\n\n>>> sli.rand(n)\n\nwhere ``n`` is the number of random numbers to generate.\n\nRemarks\n-------\n\nThe above functions all automatically detect the appropriate interface and work\nwith both real and complex data types, passing input arguments to the proper\nbackend routine.\n\n')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 386, 0))

# 'import scipy.linalg._interpolative_backend' statement (line 386)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/linalg/')
import_20873 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 386, 0), 'scipy.linalg._interpolative_backend')

if (type(import_20873) is not StypyTypeError):

    if (import_20873 != 'pyd_module'):
        __import__(import_20873)
        sys_modules_20874 = sys.modules[import_20873]
        import_module(stypy.reporting.localization.Localization(__file__, 386, 0), 'backend', sys_modules_20874.module_type_store, module_type_store)
    else:
        import scipy.linalg._interpolative_backend as backend

        import_module(stypy.reporting.localization.Localization(__file__, 386, 0), 'backend', scipy.linalg._interpolative_backend, module_type_store)

else:
    # Assigning a type to the variable 'scipy.linalg._interpolative_backend' (line 386)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 386, 0), 'scipy.linalg._interpolative_backend', import_20873)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/linalg/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 387, 0))

# 'import numpy' statement (line 387)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/linalg/')
import_20875 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 387, 0), 'numpy')

if (type(import_20875) is not StypyTypeError):

    if (import_20875 != 'pyd_module'):
        __import__(import_20875)
        sys_modules_20876 = sys.modules[import_20875]
        import_module(stypy.reporting.localization.Localization(__file__, 387, 0), 'np', sys_modules_20876.module_type_store, module_type_store)
    else:
        import numpy as np

        import_module(stypy.reporting.localization.Localization(__file__, 387, 0), 'np', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 387)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 387, 0), 'numpy', import_20875)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/linalg/')


# Assigning a Call to a Name (line 389):

# Assigning a Call to a Name (line 389):

# Call to ValueError(...): (line 389)
# Processing the call arguments (line 389)
str_20878 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 389, 26), 'str', 'invalid input dtype (input must be float64 or complex128)')
# Processing the call keyword arguments (line 389)
kwargs_20879 = {}
# Getting the type of 'ValueError' (line 389)
ValueError_20877 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 389, 15), 'ValueError', False)
# Calling ValueError(args, kwargs) (line 389)
ValueError_call_result_20880 = invoke(stypy.reporting.localization.Localization(__file__, 389, 15), ValueError_20877, *[str_20878], **kwargs_20879)

# Assigning a type to the variable '_DTYPE_ERROR' (line 389)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 389, 0), '_DTYPE_ERROR', ValueError_call_result_20880)

# Assigning a Call to a Name (line 390):

# Assigning a Call to a Name (line 390):

# Call to TypeError(...): (line 390)
# Processing the call arguments (line 390)
str_20882 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 390, 24), 'str', 'invalid input type (must be array or LinearOperator)')
# Processing the call keyword arguments (line 390)
kwargs_20883 = {}
# Getting the type of 'TypeError' (line 390)
TypeError_20881 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 390, 14), 'TypeError', False)
# Calling TypeError(args, kwargs) (line 390)
TypeError_call_result_20884 = invoke(stypy.reporting.localization.Localization(__file__, 390, 14), TypeError_20881, *[str_20882], **kwargs_20883)

# Assigning a type to the variable '_TYPE_ERROR' (line 390)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 390, 0), '_TYPE_ERROR', TypeError_call_result_20884)

@norecursion
def _is_real(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_is_real'
    module_type_store = module_type_store.open_function_context('_is_real', 393, 0, False)
    
    # Passed parameters checking function
    _is_real.stypy_localization = localization
    _is_real.stypy_type_of_self = None
    _is_real.stypy_type_store = module_type_store
    _is_real.stypy_function_name = '_is_real'
    _is_real.stypy_param_names_list = ['A']
    _is_real.stypy_varargs_param_name = None
    _is_real.stypy_kwargs_param_name = None
    _is_real.stypy_call_defaults = defaults
    _is_real.stypy_call_varargs = varargs
    _is_real.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_is_real', ['A'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_is_real', localization, ['A'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_is_real(...)' code ##################

    
    
    # SSA begins for try-except statement (line 394)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
    
    
    # Getting the type of 'A' (line 395)
    A_20885 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 395, 11), 'A')
    # Obtaining the member 'dtype' of a type (line 395)
    dtype_20886 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 395, 11), A_20885, 'dtype')
    # Getting the type of 'np' (line 395)
    np_20887 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 395, 22), 'np')
    # Obtaining the member 'complex128' of a type (line 395)
    complex128_20888 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 395, 22), np_20887, 'complex128')
    # Applying the binary operator '==' (line 395)
    result_eq_20889 = python_operator(stypy.reporting.localization.Localization(__file__, 395, 11), '==', dtype_20886, complex128_20888)
    
    # Testing the type of an if condition (line 395)
    if_condition_20890 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 395, 8), result_eq_20889)
    # Assigning a type to the variable 'if_condition_20890' (line 395)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 395, 8), 'if_condition_20890', if_condition_20890)
    # SSA begins for if statement (line 395)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # Getting the type of 'False' (line 396)
    False_20891 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 396, 19), 'False')
    # Assigning a type to the variable 'stypy_return_type' (line 396)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 396, 12), 'stypy_return_type', False_20891)
    # SSA branch for the else part of an if statement (line 395)
    module_type_store.open_ssa_branch('else')
    
    
    # Getting the type of 'A' (line 397)
    A_20892 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 397, 13), 'A')
    # Obtaining the member 'dtype' of a type (line 397)
    dtype_20893 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 397, 13), A_20892, 'dtype')
    # Getting the type of 'np' (line 397)
    np_20894 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 397, 24), 'np')
    # Obtaining the member 'float64' of a type (line 397)
    float64_20895 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 397, 24), np_20894, 'float64')
    # Applying the binary operator '==' (line 397)
    result_eq_20896 = python_operator(stypy.reporting.localization.Localization(__file__, 397, 13), '==', dtype_20893, float64_20895)
    
    # Testing the type of an if condition (line 397)
    if_condition_20897 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 397, 13), result_eq_20896)
    # Assigning a type to the variable 'if_condition_20897' (line 397)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 397, 13), 'if_condition_20897', if_condition_20897)
    # SSA begins for if statement (line 397)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # Getting the type of 'True' (line 398)
    True_20898 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 398, 19), 'True')
    # Assigning a type to the variable 'stypy_return_type' (line 398)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 398, 12), 'stypy_return_type', True_20898)
    # SSA branch for the else part of an if statement (line 397)
    module_type_store.open_ssa_branch('else')
    # Getting the type of '_DTYPE_ERROR' (line 400)
    _DTYPE_ERROR_20899 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 400, 18), '_DTYPE_ERROR')
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 400, 12), _DTYPE_ERROR_20899, 'raise parameter', BaseException)
    # SSA join for if statement (line 397)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 395)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA branch for the except part of a try statement (line 394)
    # SSA branch for the except 'AttributeError' branch of a try statement (line 394)
    module_type_store.open_ssa_branch('except')
    # Getting the type of '_TYPE_ERROR' (line 402)
    _TYPE_ERROR_20900 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 402, 14), '_TYPE_ERROR')
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 402, 8), _TYPE_ERROR_20900, 'raise parameter', BaseException)
    # SSA join for try-except statement (line 394)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of '_is_real(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_is_real' in the type store
    # Getting the type of 'stypy_return_type' (line 393)
    stypy_return_type_20901 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 393, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_20901)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_is_real'
    return stypy_return_type_20901

# Assigning a type to the variable '_is_real' (line 393)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 393, 0), '_is_real', _is_real)

@norecursion
def seed(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'None' (line 405)
    None_20902 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 405, 14), 'None')
    defaults = [None_20902]
    # Create a new context for function 'seed'
    module_type_store = module_type_store.open_function_context('seed', 405, 0, False)
    
    # Passed parameters checking function
    seed.stypy_localization = localization
    seed.stypy_type_of_self = None
    seed.stypy_type_store = module_type_store
    seed.stypy_function_name = 'seed'
    seed.stypy_param_names_list = ['seed']
    seed.stypy_varargs_param_name = None
    seed.stypy_kwargs_param_name = None
    seed.stypy_call_defaults = defaults
    seed.stypy_call_varargs = varargs
    seed.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'seed', ['seed'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'seed', localization, ['seed'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'seed(...)' code ##################

    str_20903 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 427, (-1)), 'str', "\n    Seed the internal random number generator used in this ID package.\n\n    The generator is a lagged Fibonacci method with 55-element internal state.\n\n    Parameters\n    ----------\n    seed : int, sequence, 'default', optional\n        If 'default', the random seed is reset to a default value.\n\n        If `seed` is a sequence containing 55 floating-point numbers\n        in range [0,1], these are used to set the internal state of\n        the generator.\n\n        If the value is an integer, the internal state is obtained\n        from `numpy.random.RandomState` (MT19937) with the integer\n        used as the initial seed.\n\n        If `seed` is omitted (None), `numpy.random` is used to\n        initialize the generator.\n\n    ")
    
    
    # Evaluating a boolean operation
    
    # Call to isinstance(...): (line 431)
    # Processing the call arguments (line 431)
    # Getting the type of 'seed' (line 431)
    seed_20905 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 431, 18), 'seed', False)
    # Getting the type of 'str' (line 431)
    str_20906 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 431, 24), 'str', False)
    # Processing the call keyword arguments (line 431)
    kwargs_20907 = {}
    # Getting the type of 'isinstance' (line 431)
    isinstance_20904 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 431, 7), 'isinstance', False)
    # Calling isinstance(args, kwargs) (line 431)
    isinstance_call_result_20908 = invoke(stypy.reporting.localization.Localization(__file__, 431, 7), isinstance_20904, *[seed_20905, str_20906], **kwargs_20907)
    
    
    # Getting the type of 'seed' (line 431)
    seed_20909 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 431, 33), 'seed')
    str_20910 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 431, 41), 'str', 'default')
    # Applying the binary operator '==' (line 431)
    result_eq_20911 = python_operator(stypy.reporting.localization.Localization(__file__, 431, 33), '==', seed_20909, str_20910)
    
    # Applying the binary operator 'and' (line 431)
    result_and_keyword_20912 = python_operator(stypy.reporting.localization.Localization(__file__, 431, 7), 'and', isinstance_call_result_20908, result_eq_20911)
    
    # Testing the type of an if condition (line 431)
    if_condition_20913 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 431, 4), result_and_keyword_20912)
    # Assigning a type to the variable 'if_condition_20913' (line 431)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 431, 4), 'if_condition_20913', if_condition_20913)
    # SSA begins for if statement (line 431)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to id_srando(...): (line 432)
    # Processing the call keyword arguments (line 432)
    kwargs_20916 = {}
    # Getting the type of 'backend' (line 432)
    backend_20914 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 432, 8), 'backend', False)
    # Obtaining the member 'id_srando' of a type (line 432)
    id_srando_20915 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 432, 8), backend_20914, 'id_srando')
    # Calling id_srando(args, kwargs) (line 432)
    id_srando_call_result_20917 = invoke(stypy.reporting.localization.Localization(__file__, 432, 8), id_srando_20915, *[], **kwargs_20916)
    
    # SSA branch for the else part of an if statement (line 431)
    module_type_store.open_ssa_branch('else')
    
    # Type idiom detected: calculating its left and rigth part (line 433)
    str_20918 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 433, 23), 'str', '__len__')
    # Getting the type of 'seed' (line 433)
    seed_20919 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 433, 17), 'seed')
    
    (may_be_20920, more_types_in_union_20921) = may_provide_member(str_20918, seed_20919)

    if may_be_20920:

        if more_types_in_union_20921:
            # Runtime conditional SSA (line 433)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        # Assigning a type to the variable 'seed' (line 433)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 433, 9), 'seed', remove_not_member_provider_from_union(seed_20919, '__len__'))
        
        # Assigning a Call to a Name (line 434):
        
        # Assigning a Call to a Name (line 434):
        
        # Call to asfortranarray(...): (line 434)
        # Processing the call arguments (line 434)
        # Getting the type of 'seed' (line 434)
        seed_20924 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 434, 34), 'seed', False)
        # Processing the call keyword arguments (line 434)
        # Getting the type of 'float' (line 434)
        float_20925 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 434, 46), 'float', False)
        keyword_20926 = float_20925
        kwargs_20927 = {'dtype': keyword_20926}
        # Getting the type of 'np' (line 434)
        np_20922 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 434, 16), 'np', False)
        # Obtaining the member 'asfortranarray' of a type (line 434)
        asfortranarray_20923 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 434, 16), np_20922, 'asfortranarray')
        # Calling asfortranarray(args, kwargs) (line 434)
        asfortranarray_call_result_20928 = invoke(stypy.reporting.localization.Localization(__file__, 434, 16), asfortranarray_20923, *[seed_20924], **kwargs_20927)
        
        # Assigning a type to the variable 'state' (line 434)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 434, 8), 'state', asfortranarray_call_result_20928)
        
        
        # Getting the type of 'state' (line 435)
        state_20929 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 435, 11), 'state')
        # Obtaining the member 'shape' of a type (line 435)
        shape_20930 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 435, 11), state_20929, 'shape')
        
        # Obtaining an instance of the builtin type 'tuple' (line 435)
        tuple_20931 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 435, 27), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 435)
        # Adding element type (line 435)
        int_20932 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 435, 27), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 435, 27), tuple_20931, int_20932)
        
        # Applying the binary operator '!=' (line 435)
        result_ne_20933 = python_operator(stypy.reporting.localization.Localization(__file__, 435, 11), '!=', shape_20930, tuple_20931)
        
        # Testing the type of an if condition (line 435)
        if_condition_20934 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 435, 8), result_ne_20933)
        # Assigning a type to the variable 'if_condition_20934' (line 435)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 435, 8), 'if_condition_20934', if_condition_20934)
        # SSA begins for if statement (line 435)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to ValueError(...): (line 436)
        # Processing the call arguments (line 436)
        str_20936 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 436, 29), 'str', 'invalid input size')
        # Processing the call keyword arguments (line 436)
        kwargs_20937 = {}
        # Getting the type of 'ValueError' (line 436)
        ValueError_20935 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 436, 18), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 436)
        ValueError_call_result_20938 = invoke(stypy.reporting.localization.Localization(__file__, 436, 18), ValueError_20935, *[str_20936], **kwargs_20937)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 436, 12), ValueError_call_result_20938, 'raise parameter', BaseException)
        # SSA branch for the else part of an if statement (line 435)
        module_type_store.open_ssa_branch('else')
        
        
        # Evaluating a boolean operation
        
        
        # Call to min(...): (line 437)
        # Processing the call keyword arguments (line 437)
        kwargs_20941 = {}
        # Getting the type of 'state' (line 437)
        state_20939 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 437, 13), 'state', False)
        # Obtaining the member 'min' of a type (line 437)
        min_20940 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 437, 13), state_20939, 'min')
        # Calling min(args, kwargs) (line 437)
        min_call_result_20942 = invoke(stypy.reporting.localization.Localization(__file__, 437, 13), min_20940, *[], **kwargs_20941)
        
        int_20943 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 437, 27), 'int')
        # Applying the binary operator '<' (line 437)
        result_lt_20944 = python_operator(stypy.reporting.localization.Localization(__file__, 437, 13), '<', min_call_result_20942, int_20943)
        
        
        
        # Call to max(...): (line 437)
        # Processing the call keyword arguments (line 437)
        kwargs_20947 = {}
        # Getting the type of 'state' (line 437)
        state_20945 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 437, 32), 'state', False)
        # Obtaining the member 'max' of a type (line 437)
        max_20946 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 437, 32), state_20945, 'max')
        # Calling max(args, kwargs) (line 437)
        max_call_result_20948 = invoke(stypy.reporting.localization.Localization(__file__, 437, 32), max_20946, *[], **kwargs_20947)
        
        int_20949 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 437, 46), 'int')
        # Applying the binary operator '>' (line 437)
        result_gt_20950 = python_operator(stypy.reporting.localization.Localization(__file__, 437, 32), '>', max_call_result_20948, int_20949)
        
        # Applying the binary operator 'or' (line 437)
        result_or_keyword_20951 = python_operator(stypy.reporting.localization.Localization(__file__, 437, 13), 'or', result_lt_20944, result_gt_20950)
        
        # Testing the type of an if condition (line 437)
        if_condition_20952 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 437, 13), result_or_keyword_20951)
        # Assigning a type to the variable 'if_condition_20952' (line 437)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 437, 13), 'if_condition_20952', if_condition_20952)
        # SSA begins for if statement (line 437)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to ValueError(...): (line 438)
        # Processing the call arguments (line 438)
        str_20954 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 438, 29), 'str', 'values not in range [0,1]')
        # Processing the call keyword arguments (line 438)
        kwargs_20955 = {}
        # Getting the type of 'ValueError' (line 438)
        ValueError_20953 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 438, 18), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 438)
        ValueError_call_result_20956 = invoke(stypy.reporting.localization.Localization(__file__, 438, 18), ValueError_20953, *[str_20954], **kwargs_20955)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 438, 12), ValueError_call_result_20956, 'raise parameter', BaseException)
        # SSA join for if statement (line 437)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 435)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to id_srandi(...): (line 439)
        # Processing the call arguments (line 439)
        # Getting the type of 'state' (line 439)
        state_20959 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 439, 26), 'state', False)
        # Processing the call keyword arguments (line 439)
        kwargs_20960 = {}
        # Getting the type of 'backend' (line 439)
        backend_20957 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 439, 8), 'backend', False)
        # Obtaining the member 'id_srandi' of a type (line 439)
        id_srandi_20958 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 439, 8), backend_20957, 'id_srandi')
        # Calling id_srandi(args, kwargs) (line 439)
        id_srandi_call_result_20961 = invoke(stypy.reporting.localization.Localization(__file__, 439, 8), id_srandi_20958, *[state_20959], **kwargs_20960)
        

        if more_types_in_union_20921:
            # Runtime conditional SSA for else branch (line 433)
            module_type_store.open_ssa_branch('idiom else')



    if ((not may_be_20920) or more_types_in_union_20921):
        # Assigning a type to the variable 'seed' (line 433)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 433, 9), 'seed', remove_member_provider_from_union(seed_20919, '__len__'))
        
        # Type idiom detected: calculating its left and rigth part (line 440)
        # Getting the type of 'seed' (line 440)
        seed_20962 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 440, 9), 'seed')
        # Getting the type of 'None' (line 440)
        None_20963 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 440, 17), 'None')
        
        (may_be_20964, more_types_in_union_20965) = may_be_none(seed_20962, None_20963)

        if may_be_20964:

            if more_types_in_union_20965:
                # Runtime conditional SSA (line 440)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Call to id_srandi(...): (line 441)
            # Processing the call arguments (line 441)
            
            # Call to rand(...): (line 441)
            # Processing the call arguments (line 441)
            int_20971 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 441, 41), 'int')
            # Processing the call keyword arguments (line 441)
            kwargs_20972 = {}
            # Getting the type of 'np' (line 441)
            np_20968 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 441, 26), 'np', False)
            # Obtaining the member 'random' of a type (line 441)
            random_20969 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 441, 26), np_20968, 'random')
            # Obtaining the member 'rand' of a type (line 441)
            rand_20970 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 441, 26), random_20969, 'rand')
            # Calling rand(args, kwargs) (line 441)
            rand_call_result_20973 = invoke(stypy.reporting.localization.Localization(__file__, 441, 26), rand_20970, *[int_20971], **kwargs_20972)
            
            # Processing the call keyword arguments (line 441)
            kwargs_20974 = {}
            # Getting the type of 'backend' (line 441)
            backend_20966 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 441, 8), 'backend', False)
            # Obtaining the member 'id_srandi' of a type (line 441)
            id_srandi_20967 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 441, 8), backend_20966, 'id_srandi')
            # Calling id_srandi(args, kwargs) (line 441)
            id_srandi_call_result_20975 = invoke(stypy.reporting.localization.Localization(__file__, 441, 8), id_srandi_20967, *[rand_call_result_20973], **kwargs_20974)
            

            if more_types_in_union_20965:
                # Runtime conditional SSA for else branch (line 440)
                module_type_store.open_ssa_branch('idiom else')



        if ((not may_be_20964) or more_types_in_union_20965):
            
            # Assigning a Call to a Name (line 443):
            
            # Assigning a Call to a Name (line 443):
            
            # Call to RandomState(...): (line 443)
            # Processing the call arguments (line 443)
            # Getting the type of 'seed' (line 443)
            seed_20979 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 443, 36), 'seed', False)
            # Processing the call keyword arguments (line 443)
            kwargs_20980 = {}
            # Getting the type of 'np' (line 443)
            np_20976 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 443, 14), 'np', False)
            # Obtaining the member 'random' of a type (line 443)
            random_20977 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 443, 14), np_20976, 'random')
            # Obtaining the member 'RandomState' of a type (line 443)
            RandomState_20978 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 443, 14), random_20977, 'RandomState')
            # Calling RandomState(args, kwargs) (line 443)
            RandomState_call_result_20981 = invoke(stypy.reporting.localization.Localization(__file__, 443, 14), RandomState_20978, *[seed_20979], **kwargs_20980)
            
            # Assigning a type to the variable 'rnd' (line 443)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 443, 8), 'rnd', RandomState_call_result_20981)
            
            # Call to id_srandi(...): (line 444)
            # Processing the call arguments (line 444)
            
            # Call to rand(...): (line 444)
            # Processing the call arguments (line 444)
            int_20986 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 444, 35), 'int')
            # Processing the call keyword arguments (line 444)
            kwargs_20987 = {}
            # Getting the type of 'rnd' (line 444)
            rnd_20984 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 444, 26), 'rnd', False)
            # Obtaining the member 'rand' of a type (line 444)
            rand_20985 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 444, 26), rnd_20984, 'rand')
            # Calling rand(args, kwargs) (line 444)
            rand_call_result_20988 = invoke(stypy.reporting.localization.Localization(__file__, 444, 26), rand_20985, *[int_20986], **kwargs_20987)
            
            # Processing the call keyword arguments (line 444)
            kwargs_20989 = {}
            # Getting the type of 'backend' (line 444)
            backend_20982 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 444, 8), 'backend', False)
            # Obtaining the member 'id_srandi' of a type (line 444)
            id_srandi_20983 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 444, 8), backend_20982, 'id_srandi')
            # Calling id_srandi(args, kwargs) (line 444)
            id_srandi_call_result_20990 = invoke(stypy.reporting.localization.Localization(__file__, 444, 8), id_srandi_20983, *[rand_call_result_20988], **kwargs_20989)
            

            if (may_be_20964 and more_types_in_union_20965):
                # SSA join for if statement (line 440)
                module_type_store = module_type_store.join_ssa_context()


        

        if (may_be_20920 and more_types_in_union_20921):
            # SSA join for if statement (line 433)
            module_type_store = module_type_store.join_ssa_context()


    
    # SSA join for if statement (line 431)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'seed(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'seed' in the type store
    # Getting the type of 'stypy_return_type' (line 405)
    stypy_return_type_20991 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 405, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_20991)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'seed'
    return stypy_return_type_20991

# Assigning a type to the variable 'seed' (line 405)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 405, 0), 'seed', seed)

@norecursion
def rand(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'rand'
    module_type_store = module_type_store.open_function_context('rand', 447, 0, False)
    
    # Passed parameters checking function
    rand.stypy_localization = localization
    rand.stypy_type_of_self = None
    rand.stypy_type_store = module_type_store
    rand.stypy_function_name = 'rand'
    rand.stypy_param_names_list = []
    rand.stypy_varargs_param_name = 'shape'
    rand.stypy_kwargs_param_name = None
    rand.stypy_call_defaults = defaults
    rand.stypy_call_varargs = varargs
    rand.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'rand', [], 'shape', None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'rand', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'rand(...)' code ##################

    str_20992 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 460, (-1)), 'str', '\n    Generate standard uniform pseudorandom numbers via a very efficient lagged\n    Fibonacci method.\n\n    This routine is used for all random number generation in this package and\n    can affect ID and SVD results.\n\n    Parameters\n    ----------\n    shape\n        Shape of output array\n\n    ')
    
    # Call to reshape(...): (line 462)
    # Processing the call arguments (line 462)
    # Getting the type of 'shape' (line 462)
    shape_21003 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 462, 52), 'shape', False)
    # Processing the call keyword arguments (line 462)
    kwargs_21004 = {}
    
    # Call to id_srand(...): (line 462)
    # Processing the call arguments (line 462)
    
    # Call to prod(...): (line 462)
    # Processing the call arguments (line 462)
    # Getting the type of 'shape' (line 462)
    shape_20997 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 462, 36), 'shape', False)
    # Processing the call keyword arguments (line 462)
    kwargs_20998 = {}
    # Getting the type of 'np' (line 462)
    np_20995 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 462, 28), 'np', False)
    # Obtaining the member 'prod' of a type (line 462)
    prod_20996 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 462, 28), np_20995, 'prod')
    # Calling prod(args, kwargs) (line 462)
    prod_call_result_20999 = invoke(stypy.reporting.localization.Localization(__file__, 462, 28), prod_20996, *[shape_20997], **kwargs_20998)
    
    # Processing the call keyword arguments (line 462)
    kwargs_21000 = {}
    # Getting the type of 'backend' (line 462)
    backend_20993 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 462, 11), 'backend', False)
    # Obtaining the member 'id_srand' of a type (line 462)
    id_srand_20994 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 462, 11), backend_20993, 'id_srand')
    # Calling id_srand(args, kwargs) (line 462)
    id_srand_call_result_21001 = invoke(stypy.reporting.localization.Localization(__file__, 462, 11), id_srand_20994, *[prod_call_result_20999], **kwargs_21000)
    
    # Obtaining the member 'reshape' of a type (line 462)
    reshape_21002 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 462, 11), id_srand_call_result_21001, 'reshape')
    # Calling reshape(args, kwargs) (line 462)
    reshape_call_result_21005 = invoke(stypy.reporting.localization.Localization(__file__, 462, 11), reshape_21002, *[shape_21003], **kwargs_21004)
    
    # Assigning a type to the variable 'stypy_return_type' (line 462)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 462, 4), 'stypy_return_type', reshape_call_result_21005)
    
    # ################# End of 'rand(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'rand' in the type store
    # Getting the type of 'stypy_return_type' (line 447)
    stypy_return_type_21006 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 447, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_21006)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'rand'
    return stypy_return_type_21006

# Assigning a type to the variable 'rand' (line 447)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 447, 0), 'rand', rand)

@norecursion
def interp_decomp(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'True' (line 465)
    True_21007 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 465, 36), 'True')
    defaults = [True_21007]
    # Create a new context for function 'interp_decomp'
    module_type_store = module_type_store.open_function_context('interp_decomp', 465, 0, False)
    
    # Passed parameters checking function
    interp_decomp.stypy_localization = localization
    interp_decomp.stypy_type_of_self = None
    interp_decomp.stypy_type_store = module_type_store
    interp_decomp.stypy_function_name = 'interp_decomp'
    interp_decomp.stypy_param_names_list = ['A', 'eps_or_k', 'rand']
    interp_decomp.stypy_varargs_param_name = None
    interp_decomp.stypy_kwargs_param_name = None
    interp_decomp.stypy_call_defaults = defaults
    interp_decomp.stypy_call_varargs = varargs
    interp_decomp.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'interp_decomp', ['A', 'eps_or_k', 'rand'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'interp_decomp', localization, ['A', 'eps_or_k', 'rand'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'interp_decomp(...)' code ##################

    str_21008 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 539, (-1)), 'str', '\n    Compute ID of a matrix.\n\n    An ID of a matrix `A` is a factorization defined by a rank `k`, a column\n    index array `idx`, and interpolation coefficients `proj` such that::\n\n        numpy.dot(A[:,idx[:k]], proj) = A[:,idx[k:]]\n\n    The original matrix can then be reconstructed as::\n\n        numpy.hstack([A[:,idx[:k]],\n                                    numpy.dot(A[:,idx[:k]], proj)]\n                                )[:,numpy.argsort(idx)]\n\n    or via the routine :func:`reconstruct_matrix_from_id`. This can\n    equivalently be written as::\n\n        numpy.dot(A[:,idx[:k]],\n                            numpy.hstack([numpy.eye(k), proj])\n                          )[:,np.argsort(idx)]\n\n    in terms of the skeleton and interpolation matrices::\n\n        B = A[:,idx[:k]]\n\n    and::\n\n        P = numpy.hstack([numpy.eye(k), proj])[:,np.argsort(idx)]\n\n    respectively. See also :func:`reconstruct_interp_matrix` and\n    :func:`reconstruct_skel_matrix`.\n\n    The ID can be computed to any relative precision or rank (depending on the\n    value of `eps_or_k`). If a precision is specified (`eps_or_k < 1`), then\n    this function has the output signature::\n\n        k, idx, proj = interp_decomp(A, eps_or_k)\n\n    Otherwise, if a rank is specified (`eps_or_k >= 1`), then the output\n    signature is::\n\n        idx, proj = interp_decomp(A, eps_or_k)\n\n    ..  This function automatically detects the form of the input parameters\n        and passes them to the appropriate backend. For details, see\n        :func:`backend.iddp_id`, :func:`backend.iddp_aid`,\n        :func:`backend.iddp_rid`, :func:`backend.iddr_id`,\n        :func:`backend.iddr_aid`, :func:`backend.iddr_rid`,\n        :func:`backend.idzp_id`, :func:`backend.idzp_aid`,\n        :func:`backend.idzp_rid`, :func:`backend.idzr_id`,\n        :func:`backend.idzr_aid`, and :func:`backend.idzr_rid`.\n\n    Parameters\n    ----------\n    A : :class:`numpy.ndarray` or :class:`scipy.sparse.linalg.LinearOperator` with `rmatvec`\n        Matrix to be factored\n    eps_or_k : float or int\n        Relative error (if `eps_or_k < 1`) or rank (if `eps_or_k >= 1`) of\n        approximation.\n    rand : bool, optional\n        Whether to use random sampling if `A` is of type :class:`numpy.ndarray`\n        (randomized algorithms are always used if `A` is of type\n        :class:`scipy.sparse.linalg.LinearOperator`).\n\n    Returns\n    -------\n    k : int\n        Rank required to achieve specified relative precision if\n        `eps_or_k < 1`.\n    idx : :class:`numpy.ndarray`\n        Column index array.\n    proj : :class:`numpy.ndarray`\n        Interpolation coefficients.\n    ')
    stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 540, 4))
    
    # 'from scipy.sparse.linalg import LinearOperator' statement (line 540)
    update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/linalg/')
    import_21009 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 540, 4), 'scipy.sparse.linalg')

    if (type(import_21009) is not StypyTypeError):

        if (import_21009 != 'pyd_module'):
            __import__(import_21009)
            sys_modules_21010 = sys.modules[import_21009]
            import_from_module(stypy.reporting.localization.Localization(__file__, 540, 4), 'scipy.sparse.linalg', sys_modules_21010.module_type_store, module_type_store, ['LinearOperator'])
            nest_module(stypy.reporting.localization.Localization(__file__, 540, 4), __file__, sys_modules_21010, sys_modules_21010.module_type_store, module_type_store)
        else:
            from scipy.sparse.linalg import LinearOperator

            import_from_module(stypy.reporting.localization.Localization(__file__, 540, 4), 'scipy.sparse.linalg', None, module_type_store, ['LinearOperator'], [LinearOperator])

    else:
        # Assigning a type to the variable 'scipy.sparse.linalg' (line 540)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 540, 4), 'scipy.sparse.linalg', import_21009)

    remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/linalg/')
    
    
    # Assigning a Call to a Name (line 542):
    
    # Assigning a Call to a Name (line 542):
    
    # Call to _is_real(...): (line 542)
    # Processing the call arguments (line 542)
    # Getting the type of 'A' (line 542)
    A_21012 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 542, 20), 'A', False)
    # Processing the call keyword arguments (line 542)
    kwargs_21013 = {}
    # Getting the type of '_is_real' (line 542)
    _is_real_21011 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 542, 11), '_is_real', False)
    # Calling _is_real(args, kwargs) (line 542)
    _is_real_call_result_21014 = invoke(stypy.reporting.localization.Localization(__file__, 542, 11), _is_real_21011, *[A_21012], **kwargs_21013)
    
    # Assigning a type to the variable 'real' (line 542)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 542, 4), 'real', _is_real_call_result_21014)
    
    
    # Call to isinstance(...): (line 544)
    # Processing the call arguments (line 544)
    # Getting the type of 'A' (line 544)
    A_21016 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 544, 18), 'A', False)
    # Getting the type of 'np' (line 544)
    np_21017 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 544, 21), 'np', False)
    # Obtaining the member 'ndarray' of a type (line 544)
    ndarray_21018 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 544, 21), np_21017, 'ndarray')
    # Processing the call keyword arguments (line 544)
    kwargs_21019 = {}
    # Getting the type of 'isinstance' (line 544)
    isinstance_21015 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 544, 7), 'isinstance', False)
    # Calling isinstance(args, kwargs) (line 544)
    isinstance_call_result_21020 = invoke(stypy.reporting.localization.Localization(__file__, 544, 7), isinstance_21015, *[A_21016, ndarray_21018], **kwargs_21019)
    
    # Testing the type of an if condition (line 544)
    if_condition_21021 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 544, 4), isinstance_call_result_21020)
    # Assigning a type to the variable 'if_condition_21021' (line 544)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 544, 4), 'if_condition_21021', if_condition_21021)
    # SSA begins for if statement (line 544)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    
    # Getting the type of 'eps_or_k' (line 545)
    eps_or_k_21022 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 545, 11), 'eps_or_k')
    int_21023 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 545, 22), 'int')
    # Applying the binary operator '<' (line 545)
    result_lt_21024 = python_operator(stypy.reporting.localization.Localization(__file__, 545, 11), '<', eps_or_k_21022, int_21023)
    
    # Testing the type of an if condition (line 545)
    if_condition_21025 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 545, 8), result_lt_21024)
    # Assigning a type to the variable 'if_condition_21025' (line 545)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 545, 8), 'if_condition_21025', if_condition_21025)
    # SSA begins for if statement (line 545)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Name to a Name (line 546):
    
    # Assigning a Name to a Name (line 546):
    # Getting the type of 'eps_or_k' (line 546)
    eps_or_k_21026 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 546, 18), 'eps_or_k')
    # Assigning a type to the variable 'eps' (line 546)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 546, 12), 'eps', eps_or_k_21026)
    
    # Getting the type of 'rand' (line 547)
    rand_21027 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 547, 15), 'rand')
    # Testing the type of an if condition (line 547)
    if_condition_21028 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 547, 12), rand_21027)
    # Assigning a type to the variable 'if_condition_21028' (line 547)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 547, 12), 'if_condition_21028', if_condition_21028)
    # SSA begins for if statement (line 547)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Getting the type of 'real' (line 548)
    real_21029 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 548, 19), 'real')
    # Testing the type of an if condition (line 548)
    if_condition_21030 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 548, 16), real_21029)
    # Assigning a type to the variable 'if_condition_21030' (line 548)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 548, 16), 'if_condition_21030', if_condition_21030)
    # SSA begins for if statement (line 548)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Tuple (line 549):
    
    # Assigning a Subscript to a Name (line 549):
    
    # Obtaining the type of the subscript
    int_21031 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 549, 20), 'int')
    
    # Call to iddp_aid(...): (line 549)
    # Processing the call arguments (line 549)
    # Getting the type of 'eps' (line 549)
    eps_21034 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 549, 52), 'eps', False)
    # Getting the type of 'A' (line 549)
    A_21035 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 549, 57), 'A', False)
    # Processing the call keyword arguments (line 549)
    kwargs_21036 = {}
    # Getting the type of 'backend' (line 549)
    backend_21032 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 549, 35), 'backend', False)
    # Obtaining the member 'iddp_aid' of a type (line 549)
    iddp_aid_21033 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 549, 35), backend_21032, 'iddp_aid')
    # Calling iddp_aid(args, kwargs) (line 549)
    iddp_aid_call_result_21037 = invoke(stypy.reporting.localization.Localization(__file__, 549, 35), iddp_aid_21033, *[eps_21034, A_21035], **kwargs_21036)
    
    # Obtaining the member '__getitem__' of a type (line 549)
    getitem___21038 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 549, 20), iddp_aid_call_result_21037, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 549)
    subscript_call_result_21039 = invoke(stypy.reporting.localization.Localization(__file__, 549, 20), getitem___21038, int_21031)
    
    # Assigning a type to the variable 'tuple_var_assignment_20790' (line 549)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 549, 20), 'tuple_var_assignment_20790', subscript_call_result_21039)
    
    # Assigning a Subscript to a Name (line 549):
    
    # Obtaining the type of the subscript
    int_21040 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 549, 20), 'int')
    
    # Call to iddp_aid(...): (line 549)
    # Processing the call arguments (line 549)
    # Getting the type of 'eps' (line 549)
    eps_21043 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 549, 52), 'eps', False)
    # Getting the type of 'A' (line 549)
    A_21044 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 549, 57), 'A', False)
    # Processing the call keyword arguments (line 549)
    kwargs_21045 = {}
    # Getting the type of 'backend' (line 549)
    backend_21041 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 549, 35), 'backend', False)
    # Obtaining the member 'iddp_aid' of a type (line 549)
    iddp_aid_21042 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 549, 35), backend_21041, 'iddp_aid')
    # Calling iddp_aid(args, kwargs) (line 549)
    iddp_aid_call_result_21046 = invoke(stypy.reporting.localization.Localization(__file__, 549, 35), iddp_aid_21042, *[eps_21043, A_21044], **kwargs_21045)
    
    # Obtaining the member '__getitem__' of a type (line 549)
    getitem___21047 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 549, 20), iddp_aid_call_result_21046, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 549)
    subscript_call_result_21048 = invoke(stypy.reporting.localization.Localization(__file__, 549, 20), getitem___21047, int_21040)
    
    # Assigning a type to the variable 'tuple_var_assignment_20791' (line 549)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 549, 20), 'tuple_var_assignment_20791', subscript_call_result_21048)
    
    # Assigning a Subscript to a Name (line 549):
    
    # Obtaining the type of the subscript
    int_21049 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 549, 20), 'int')
    
    # Call to iddp_aid(...): (line 549)
    # Processing the call arguments (line 549)
    # Getting the type of 'eps' (line 549)
    eps_21052 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 549, 52), 'eps', False)
    # Getting the type of 'A' (line 549)
    A_21053 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 549, 57), 'A', False)
    # Processing the call keyword arguments (line 549)
    kwargs_21054 = {}
    # Getting the type of 'backend' (line 549)
    backend_21050 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 549, 35), 'backend', False)
    # Obtaining the member 'iddp_aid' of a type (line 549)
    iddp_aid_21051 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 549, 35), backend_21050, 'iddp_aid')
    # Calling iddp_aid(args, kwargs) (line 549)
    iddp_aid_call_result_21055 = invoke(stypy.reporting.localization.Localization(__file__, 549, 35), iddp_aid_21051, *[eps_21052, A_21053], **kwargs_21054)
    
    # Obtaining the member '__getitem__' of a type (line 549)
    getitem___21056 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 549, 20), iddp_aid_call_result_21055, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 549)
    subscript_call_result_21057 = invoke(stypy.reporting.localization.Localization(__file__, 549, 20), getitem___21056, int_21049)
    
    # Assigning a type to the variable 'tuple_var_assignment_20792' (line 549)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 549, 20), 'tuple_var_assignment_20792', subscript_call_result_21057)
    
    # Assigning a Name to a Name (line 549):
    # Getting the type of 'tuple_var_assignment_20790' (line 549)
    tuple_var_assignment_20790_21058 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 549, 20), 'tuple_var_assignment_20790')
    # Assigning a type to the variable 'k' (line 549)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 549, 20), 'k', tuple_var_assignment_20790_21058)
    
    # Assigning a Name to a Name (line 549):
    # Getting the type of 'tuple_var_assignment_20791' (line 549)
    tuple_var_assignment_20791_21059 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 549, 20), 'tuple_var_assignment_20791')
    # Assigning a type to the variable 'idx' (line 549)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 549, 23), 'idx', tuple_var_assignment_20791_21059)
    
    # Assigning a Name to a Name (line 549):
    # Getting the type of 'tuple_var_assignment_20792' (line 549)
    tuple_var_assignment_20792_21060 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 549, 20), 'tuple_var_assignment_20792')
    # Assigning a type to the variable 'proj' (line 549)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 549, 28), 'proj', tuple_var_assignment_20792_21060)
    # SSA branch for the else part of an if statement (line 548)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Call to a Tuple (line 551):
    
    # Assigning a Subscript to a Name (line 551):
    
    # Obtaining the type of the subscript
    int_21061 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 551, 20), 'int')
    
    # Call to idzp_aid(...): (line 551)
    # Processing the call arguments (line 551)
    # Getting the type of 'eps' (line 551)
    eps_21064 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 551, 52), 'eps', False)
    # Getting the type of 'A' (line 551)
    A_21065 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 551, 57), 'A', False)
    # Processing the call keyword arguments (line 551)
    kwargs_21066 = {}
    # Getting the type of 'backend' (line 551)
    backend_21062 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 551, 35), 'backend', False)
    # Obtaining the member 'idzp_aid' of a type (line 551)
    idzp_aid_21063 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 551, 35), backend_21062, 'idzp_aid')
    # Calling idzp_aid(args, kwargs) (line 551)
    idzp_aid_call_result_21067 = invoke(stypy.reporting.localization.Localization(__file__, 551, 35), idzp_aid_21063, *[eps_21064, A_21065], **kwargs_21066)
    
    # Obtaining the member '__getitem__' of a type (line 551)
    getitem___21068 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 551, 20), idzp_aid_call_result_21067, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 551)
    subscript_call_result_21069 = invoke(stypy.reporting.localization.Localization(__file__, 551, 20), getitem___21068, int_21061)
    
    # Assigning a type to the variable 'tuple_var_assignment_20793' (line 551)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 551, 20), 'tuple_var_assignment_20793', subscript_call_result_21069)
    
    # Assigning a Subscript to a Name (line 551):
    
    # Obtaining the type of the subscript
    int_21070 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 551, 20), 'int')
    
    # Call to idzp_aid(...): (line 551)
    # Processing the call arguments (line 551)
    # Getting the type of 'eps' (line 551)
    eps_21073 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 551, 52), 'eps', False)
    # Getting the type of 'A' (line 551)
    A_21074 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 551, 57), 'A', False)
    # Processing the call keyword arguments (line 551)
    kwargs_21075 = {}
    # Getting the type of 'backend' (line 551)
    backend_21071 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 551, 35), 'backend', False)
    # Obtaining the member 'idzp_aid' of a type (line 551)
    idzp_aid_21072 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 551, 35), backend_21071, 'idzp_aid')
    # Calling idzp_aid(args, kwargs) (line 551)
    idzp_aid_call_result_21076 = invoke(stypy.reporting.localization.Localization(__file__, 551, 35), idzp_aid_21072, *[eps_21073, A_21074], **kwargs_21075)
    
    # Obtaining the member '__getitem__' of a type (line 551)
    getitem___21077 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 551, 20), idzp_aid_call_result_21076, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 551)
    subscript_call_result_21078 = invoke(stypy.reporting.localization.Localization(__file__, 551, 20), getitem___21077, int_21070)
    
    # Assigning a type to the variable 'tuple_var_assignment_20794' (line 551)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 551, 20), 'tuple_var_assignment_20794', subscript_call_result_21078)
    
    # Assigning a Subscript to a Name (line 551):
    
    # Obtaining the type of the subscript
    int_21079 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 551, 20), 'int')
    
    # Call to idzp_aid(...): (line 551)
    # Processing the call arguments (line 551)
    # Getting the type of 'eps' (line 551)
    eps_21082 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 551, 52), 'eps', False)
    # Getting the type of 'A' (line 551)
    A_21083 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 551, 57), 'A', False)
    # Processing the call keyword arguments (line 551)
    kwargs_21084 = {}
    # Getting the type of 'backend' (line 551)
    backend_21080 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 551, 35), 'backend', False)
    # Obtaining the member 'idzp_aid' of a type (line 551)
    idzp_aid_21081 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 551, 35), backend_21080, 'idzp_aid')
    # Calling idzp_aid(args, kwargs) (line 551)
    idzp_aid_call_result_21085 = invoke(stypy.reporting.localization.Localization(__file__, 551, 35), idzp_aid_21081, *[eps_21082, A_21083], **kwargs_21084)
    
    # Obtaining the member '__getitem__' of a type (line 551)
    getitem___21086 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 551, 20), idzp_aid_call_result_21085, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 551)
    subscript_call_result_21087 = invoke(stypy.reporting.localization.Localization(__file__, 551, 20), getitem___21086, int_21079)
    
    # Assigning a type to the variable 'tuple_var_assignment_20795' (line 551)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 551, 20), 'tuple_var_assignment_20795', subscript_call_result_21087)
    
    # Assigning a Name to a Name (line 551):
    # Getting the type of 'tuple_var_assignment_20793' (line 551)
    tuple_var_assignment_20793_21088 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 551, 20), 'tuple_var_assignment_20793')
    # Assigning a type to the variable 'k' (line 551)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 551, 20), 'k', tuple_var_assignment_20793_21088)
    
    # Assigning a Name to a Name (line 551):
    # Getting the type of 'tuple_var_assignment_20794' (line 551)
    tuple_var_assignment_20794_21089 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 551, 20), 'tuple_var_assignment_20794')
    # Assigning a type to the variable 'idx' (line 551)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 551, 23), 'idx', tuple_var_assignment_20794_21089)
    
    # Assigning a Name to a Name (line 551):
    # Getting the type of 'tuple_var_assignment_20795' (line 551)
    tuple_var_assignment_20795_21090 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 551, 20), 'tuple_var_assignment_20795')
    # Assigning a type to the variable 'proj' (line 551)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 551, 28), 'proj', tuple_var_assignment_20795_21090)
    # SSA join for if statement (line 548)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA branch for the else part of an if statement (line 547)
    module_type_store.open_ssa_branch('else')
    
    # Getting the type of 'real' (line 553)
    real_21091 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 553, 19), 'real')
    # Testing the type of an if condition (line 553)
    if_condition_21092 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 553, 16), real_21091)
    # Assigning a type to the variable 'if_condition_21092' (line 553)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 553, 16), 'if_condition_21092', if_condition_21092)
    # SSA begins for if statement (line 553)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Tuple (line 554):
    
    # Assigning a Subscript to a Name (line 554):
    
    # Obtaining the type of the subscript
    int_21093 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 554, 20), 'int')
    
    # Call to iddp_id(...): (line 554)
    # Processing the call arguments (line 554)
    # Getting the type of 'eps' (line 554)
    eps_21096 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 554, 51), 'eps', False)
    # Getting the type of 'A' (line 554)
    A_21097 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 554, 56), 'A', False)
    # Processing the call keyword arguments (line 554)
    kwargs_21098 = {}
    # Getting the type of 'backend' (line 554)
    backend_21094 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 554, 35), 'backend', False)
    # Obtaining the member 'iddp_id' of a type (line 554)
    iddp_id_21095 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 554, 35), backend_21094, 'iddp_id')
    # Calling iddp_id(args, kwargs) (line 554)
    iddp_id_call_result_21099 = invoke(stypy.reporting.localization.Localization(__file__, 554, 35), iddp_id_21095, *[eps_21096, A_21097], **kwargs_21098)
    
    # Obtaining the member '__getitem__' of a type (line 554)
    getitem___21100 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 554, 20), iddp_id_call_result_21099, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 554)
    subscript_call_result_21101 = invoke(stypy.reporting.localization.Localization(__file__, 554, 20), getitem___21100, int_21093)
    
    # Assigning a type to the variable 'tuple_var_assignment_20796' (line 554)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 554, 20), 'tuple_var_assignment_20796', subscript_call_result_21101)
    
    # Assigning a Subscript to a Name (line 554):
    
    # Obtaining the type of the subscript
    int_21102 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 554, 20), 'int')
    
    # Call to iddp_id(...): (line 554)
    # Processing the call arguments (line 554)
    # Getting the type of 'eps' (line 554)
    eps_21105 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 554, 51), 'eps', False)
    # Getting the type of 'A' (line 554)
    A_21106 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 554, 56), 'A', False)
    # Processing the call keyword arguments (line 554)
    kwargs_21107 = {}
    # Getting the type of 'backend' (line 554)
    backend_21103 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 554, 35), 'backend', False)
    # Obtaining the member 'iddp_id' of a type (line 554)
    iddp_id_21104 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 554, 35), backend_21103, 'iddp_id')
    # Calling iddp_id(args, kwargs) (line 554)
    iddp_id_call_result_21108 = invoke(stypy.reporting.localization.Localization(__file__, 554, 35), iddp_id_21104, *[eps_21105, A_21106], **kwargs_21107)
    
    # Obtaining the member '__getitem__' of a type (line 554)
    getitem___21109 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 554, 20), iddp_id_call_result_21108, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 554)
    subscript_call_result_21110 = invoke(stypy.reporting.localization.Localization(__file__, 554, 20), getitem___21109, int_21102)
    
    # Assigning a type to the variable 'tuple_var_assignment_20797' (line 554)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 554, 20), 'tuple_var_assignment_20797', subscript_call_result_21110)
    
    # Assigning a Subscript to a Name (line 554):
    
    # Obtaining the type of the subscript
    int_21111 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 554, 20), 'int')
    
    # Call to iddp_id(...): (line 554)
    # Processing the call arguments (line 554)
    # Getting the type of 'eps' (line 554)
    eps_21114 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 554, 51), 'eps', False)
    # Getting the type of 'A' (line 554)
    A_21115 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 554, 56), 'A', False)
    # Processing the call keyword arguments (line 554)
    kwargs_21116 = {}
    # Getting the type of 'backend' (line 554)
    backend_21112 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 554, 35), 'backend', False)
    # Obtaining the member 'iddp_id' of a type (line 554)
    iddp_id_21113 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 554, 35), backend_21112, 'iddp_id')
    # Calling iddp_id(args, kwargs) (line 554)
    iddp_id_call_result_21117 = invoke(stypy.reporting.localization.Localization(__file__, 554, 35), iddp_id_21113, *[eps_21114, A_21115], **kwargs_21116)
    
    # Obtaining the member '__getitem__' of a type (line 554)
    getitem___21118 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 554, 20), iddp_id_call_result_21117, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 554)
    subscript_call_result_21119 = invoke(stypy.reporting.localization.Localization(__file__, 554, 20), getitem___21118, int_21111)
    
    # Assigning a type to the variable 'tuple_var_assignment_20798' (line 554)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 554, 20), 'tuple_var_assignment_20798', subscript_call_result_21119)
    
    # Assigning a Name to a Name (line 554):
    # Getting the type of 'tuple_var_assignment_20796' (line 554)
    tuple_var_assignment_20796_21120 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 554, 20), 'tuple_var_assignment_20796')
    # Assigning a type to the variable 'k' (line 554)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 554, 20), 'k', tuple_var_assignment_20796_21120)
    
    # Assigning a Name to a Name (line 554):
    # Getting the type of 'tuple_var_assignment_20797' (line 554)
    tuple_var_assignment_20797_21121 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 554, 20), 'tuple_var_assignment_20797')
    # Assigning a type to the variable 'idx' (line 554)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 554, 23), 'idx', tuple_var_assignment_20797_21121)
    
    # Assigning a Name to a Name (line 554):
    # Getting the type of 'tuple_var_assignment_20798' (line 554)
    tuple_var_assignment_20798_21122 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 554, 20), 'tuple_var_assignment_20798')
    # Assigning a type to the variable 'proj' (line 554)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 554, 28), 'proj', tuple_var_assignment_20798_21122)
    # SSA branch for the else part of an if statement (line 553)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Call to a Tuple (line 556):
    
    # Assigning a Subscript to a Name (line 556):
    
    # Obtaining the type of the subscript
    int_21123 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 556, 20), 'int')
    
    # Call to idzp_id(...): (line 556)
    # Processing the call arguments (line 556)
    # Getting the type of 'eps' (line 556)
    eps_21126 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 556, 51), 'eps', False)
    # Getting the type of 'A' (line 556)
    A_21127 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 556, 56), 'A', False)
    # Processing the call keyword arguments (line 556)
    kwargs_21128 = {}
    # Getting the type of 'backend' (line 556)
    backend_21124 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 556, 35), 'backend', False)
    # Obtaining the member 'idzp_id' of a type (line 556)
    idzp_id_21125 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 556, 35), backend_21124, 'idzp_id')
    # Calling idzp_id(args, kwargs) (line 556)
    idzp_id_call_result_21129 = invoke(stypy.reporting.localization.Localization(__file__, 556, 35), idzp_id_21125, *[eps_21126, A_21127], **kwargs_21128)
    
    # Obtaining the member '__getitem__' of a type (line 556)
    getitem___21130 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 556, 20), idzp_id_call_result_21129, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 556)
    subscript_call_result_21131 = invoke(stypy.reporting.localization.Localization(__file__, 556, 20), getitem___21130, int_21123)
    
    # Assigning a type to the variable 'tuple_var_assignment_20799' (line 556)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 556, 20), 'tuple_var_assignment_20799', subscript_call_result_21131)
    
    # Assigning a Subscript to a Name (line 556):
    
    # Obtaining the type of the subscript
    int_21132 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 556, 20), 'int')
    
    # Call to idzp_id(...): (line 556)
    # Processing the call arguments (line 556)
    # Getting the type of 'eps' (line 556)
    eps_21135 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 556, 51), 'eps', False)
    # Getting the type of 'A' (line 556)
    A_21136 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 556, 56), 'A', False)
    # Processing the call keyword arguments (line 556)
    kwargs_21137 = {}
    # Getting the type of 'backend' (line 556)
    backend_21133 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 556, 35), 'backend', False)
    # Obtaining the member 'idzp_id' of a type (line 556)
    idzp_id_21134 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 556, 35), backend_21133, 'idzp_id')
    # Calling idzp_id(args, kwargs) (line 556)
    idzp_id_call_result_21138 = invoke(stypy.reporting.localization.Localization(__file__, 556, 35), idzp_id_21134, *[eps_21135, A_21136], **kwargs_21137)
    
    # Obtaining the member '__getitem__' of a type (line 556)
    getitem___21139 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 556, 20), idzp_id_call_result_21138, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 556)
    subscript_call_result_21140 = invoke(stypy.reporting.localization.Localization(__file__, 556, 20), getitem___21139, int_21132)
    
    # Assigning a type to the variable 'tuple_var_assignment_20800' (line 556)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 556, 20), 'tuple_var_assignment_20800', subscript_call_result_21140)
    
    # Assigning a Subscript to a Name (line 556):
    
    # Obtaining the type of the subscript
    int_21141 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 556, 20), 'int')
    
    # Call to idzp_id(...): (line 556)
    # Processing the call arguments (line 556)
    # Getting the type of 'eps' (line 556)
    eps_21144 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 556, 51), 'eps', False)
    # Getting the type of 'A' (line 556)
    A_21145 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 556, 56), 'A', False)
    # Processing the call keyword arguments (line 556)
    kwargs_21146 = {}
    # Getting the type of 'backend' (line 556)
    backend_21142 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 556, 35), 'backend', False)
    # Obtaining the member 'idzp_id' of a type (line 556)
    idzp_id_21143 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 556, 35), backend_21142, 'idzp_id')
    # Calling idzp_id(args, kwargs) (line 556)
    idzp_id_call_result_21147 = invoke(stypy.reporting.localization.Localization(__file__, 556, 35), idzp_id_21143, *[eps_21144, A_21145], **kwargs_21146)
    
    # Obtaining the member '__getitem__' of a type (line 556)
    getitem___21148 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 556, 20), idzp_id_call_result_21147, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 556)
    subscript_call_result_21149 = invoke(stypy.reporting.localization.Localization(__file__, 556, 20), getitem___21148, int_21141)
    
    # Assigning a type to the variable 'tuple_var_assignment_20801' (line 556)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 556, 20), 'tuple_var_assignment_20801', subscript_call_result_21149)
    
    # Assigning a Name to a Name (line 556):
    # Getting the type of 'tuple_var_assignment_20799' (line 556)
    tuple_var_assignment_20799_21150 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 556, 20), 'tuple_var_assignment_20799')
    # Assigning a type to the variable 'k' (line 556)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 556, 20), 'k', tuple_var_assignment_20799_21150)
    
    # Assigning a Name to a Name (line 556):
    # Getting the type of 'tuple_var_assignment_20800' (line 556)
    tuple_var_assignment_20800_21151 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 556, 20), 'tuple_var_assignment_20800')
    # Assigning a type to the variable 'idx' (line 556)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 556, 23), 'idx', tuple_var_assignment_20800_21151)
    
    # Assigning a Name to a Name (line 556):
    # Getting the type of 'tuple_var_assignment_20801' (line 556)
    tuple_var_assignment_20801_21152 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 556, 20), 'tuple_var_assignment_20801')
    # Assigning a type to the variable 'proj' (line 556)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 556, 28), 'proj', tuple_var_assignment_20801_21152)
    # SSA join for if statement (line 553)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 547)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Obtaining an instance of the builtin type 'tuple' (line 557)
    tuple_21153 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 557, 19), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 557)
    # Adding element type (line 557)
    # Getting the type of 'k' (line 557)
    k_21154 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 557, 19), 'k')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 557, 19), tuple_21153, k_21154)
    # Adding element type (line 557)
    # Getting the type of 'idx' (line 557)
    idx_21155 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 557, 22), 'idx')
    int_21156 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 557, 28), 'int')
    # Applying the binary operator '-' (line 557)
    result_sub_21157 = python_operator(stypy.reporting.localization.Localization(__file__, 557, 22), '-', idx_21155, int_21156)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 557, 19), tuple_21153, result_sub_21157)
    # Adding element type (line 557)
    # Getting the type of 'proj' (line 557)
    proj_21158 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 557, 31), 'proj')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 557, 19), tuple_21153, proj_21158)
    
    # Assigning a type to the variable 'stypy_return_type' (line 557)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 557, 12), 'stypy_return_type', tuple_21153)
    # SSA branch for the else part of an if statement (line 545)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Call to a Name (line 559):
    
    # Assigning a Call to a Name (line 559):
    
    # Call to int(...): (line 559)
    # Processing the call arguments (line 559)
    # Getting the type of 'eps_or_k' (line 559)
    eps_or_k_21160 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 559, 20), 'eps_or_k', False)
    # Processing the call keyword arguments (line 559)
    kwargs_21161 = {}
    # Getting the type of 'int' (line 559)
    int_21159 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 559, 16), 'int', False)
    # Calling int(args, kwargs) (line 559)
    int_call_result_21162 = invoke(stypy.reporting.localization.Localization(__file__, 559, 16), int_21159, *[eps_or_k_21160], **kwargs_21161)
    
    # Assigning a type to the variable 'k' (line 559)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 559, 12), 'k', int_call_result_21162)
    
    # Getting the type of 'rand' (line 560)
    rand_21163 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 560, 15), 'rand')
    # Testing the type of an if condition (line 560)
    if_condition_21164 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 560, 12), rand_21163)
    # Assigning a type to the variable 'if_condition_21164' (line 560)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 560, 12), 'if_condition_21164', if_condition_21164)
    # SSA begins for if statement (line 560)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Getting the type of 'real' (line 561)
    real_21165 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 561, 19), 'real')
    # Testing the type of an if condition (line 561)
    if_condition_21166 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 561, 16), real_21165)
    # Assigning a type to the variable 'if_condition_21166' (line 561)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 561, 16), 'if_condition_21166', if_condition_21166)
    # SSA begins for if statement (line 561)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Tuple (line 562):
    
    # Assigning a Subscript to a Name (line 562):
    
    # Obtaining the type of the subscript
    int_21167 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 562, 20), 'int')
    
    # Call to iddr_aid(...): (line 562)
    # Processing the call arguments (line 562)
    # Getting the type of 'A' (line 562)
    A_21170 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 562, 49), 'A', False)
    # Getting the type of 'k' (line 562)
    k_21171 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 562, 52), 'k', False)
    # Processing the call keyword arguments (line 562)
    kwargs_21172 = {}
    # Getting the type of 'backend' (line 562)
    backend_21168 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 562, 32), 'backend', False)
    # Obtaining the member 'iddr_aid' of a type (line 562)
    iddr_aid_21169 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 562, 32), backend_21168, 'iddr_aid')
    # Calling iddr_aid(args, kwargs) (line 562)
    iddr_aid_call_result_21173 = invoke(stypy.reporting.localization.Localization(__file__, 562, 32), iddr_aid_21169, *[A_21170, k_21171], **kwargs_21172)
    
    # Obtaining the member '__getitem__' of a type (line 562)
    getitem___21174 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 562, 20), iddr_aid_call_result_21173, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 562)
    subscript_call_result_21175 = invoke(stypy.reporting.localization.Localization(__file__, 562, 20), getitem___21174, int_21167)
    
    # Assigning a type to the variable 'tuple_var_assignment_20802' (line 562)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 562, 20), 'tuple_var_assignment_20802', subscript_call_result_21175)
    
    # Assigning a Subscript to a Name (line 562):
    
    # Obtaining the type of the subscript
    int_21176 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 562, 20), 'int')
    
    # Call to iddr_aid(...): (line 562)
    # Processing the call arguments (line 562)
    # Getting the type of 'A' (line 562)
    A_21179 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 562, 49), 'A', False)
    # Getting the type of 'k' (line 562)
    k_21180 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 562, 52), 'k', False)
    # Processing the call keyword arguments (line 562)
    kwargs_21181 = {}
    # Getting the type of 'backend' (line 562)
    backend_21177 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 562, 32), 'backend', False)
    # Obtaining the member 'iddr_aid' of a type (line 562)
    iddr_aid_21178 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 562, 32), backend_21177, 'iddr_aid')
    # Calling iddr_aid(args, kwargs) (line 562)
    iddr_aid_call_result_21182 = invoke(stypy.reporting.localization.Localization(__file__, 562, 32), iddr_aid_21178, *[A_21179, k_21180], **kwargs_21181)
    
    # Obtaining the member '__getitem__' of a type (line 562)
    getitem___21183 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 562, 20), iddr_aid_call_result_21182, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 562)
    subscript_call_result_21184 = invoke(stypy.reporting.localization.Localization(__file__, 562, 20), getitem___21183, int_21176)
    
    # Assigning a type to the variable 'tuple_var_assignment_20803' (line 562)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 562, 20), 'tuple_var_assignment_20803', subscript_call_result_21184)
    
    # Assigning a Name to a Name (line 562):
    # Getting the type of 'tuple_var_assignment_20802' (line 562)
    tuple_var_assignment_20802_21185 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 562, 20), 'tuple_var_assignment_20802')
    # Assigning a type to the variable 'idx' (line 562)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 562, 20), 'idx', tuple_var_assignment_20802_21185)
    
    # Assigning a Name to a Name (line 562):
    # Getting the type of 'tuple_var_assignment_20803' (line 562)
    tuple_var_assignment_20803_21186 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 562, 20), 'tuple_var_assignment_20803')
    # Assigning a type to the variable 'proj' (line 562)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 562, 25), 'proj', tuple_var_assignment_20803_21186)
    # SSA branch for the else part of an if statement (line 561)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Call to a Tuple (line 564):
    
    # Assigning a Subscript to a Name (line 564):
    
    # Obtaining the type of the subscript
    int_21187 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 564, 20), 'int')
    
    # Call to idzr_aid(...): (line 564)
    # Processing the call arguments (line 564)
    # Getting the type of 'A' (line 564)
    A_21190 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 564, 49), 'A', False)
    # Getting the type of 'k' (line 564)
    k_21191 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 564, 52), 'k', False)
    # Processing the call keyword arguments (line 564)
    kwargs_21192 = {}
    # Getting the type of 'backend' (line 564)
    backend_21188 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 564, 32), 'backend', False)
    # Obtaining the member 'idzr_aid' of a type (line 564)
    idzr_aid_21189 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 564, 32), backend_21188, 'idzr_aid')
    # Calling idzr_aid(args, kwargs) (line 564)
    idzr_aid_call_result_21193 = invoke(stypy.reporting.localization.Localization(__file__, 564, 32), idzr_aid_21189, *[A_21190, k_21191], **kwargs_21192)
    
    # Obtaining the member '__getitem__' of a type (line 564)
    getitem___21194 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 564, 20), idzr_aid_call_result_21193, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 564)
    subscript_call_result_21195 = invoke(stypy.reporting.localization.Localization(__file__, 564, 20), getitem___21194, int_21187)
    
    # Assigning a type to the variable 'tuple_var_assignment_20804' (line 564)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 564, 20), 'tuple_var_assignment_20804', subscript_call_result_21195)
    
    # Assigning a Subscript to a Name (line 564):
    
    # Obtaining the type of the subscript
    int_21196 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 564, 20), 'int')
    
    # Call to idzr_aid(...): (line 564)
    # Processing the call arguments (line 564)
    # Getting the type of 'A' (line 564)
    A_21199 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 564, 49), 'A', False)
    # Getting the type of 'k' (line 564)
    k_21200 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 564, 52), 'k', False)
    # Processing the call keyword arguments (line 564)
    kwargs_21201 = {}
    # Getting the type of 'backend' (line 564)
    backend_21197 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 564, 32), 'backend', False)
    # Obtaining the member 'idzr_aid' of a type (line 564)
    idzr_aid_21198 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 564, 32), backend_21197, 'idzr_aid')
    # Calling idzr_aid(args, kwargs) (line 564)
    idzr_aid_call_result_21202 = invoke(stypy.reporting.localization.Localization(__file__, 564, 32), idzr_aid_21198, *[A_21199, k_21200], **kwargs_21201)
    
    # Obtaining the member '__getitem__' of a type (line 564)
    getitem___21203 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 564, 20), idzr_aid_call_result_21202, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 564)
    subscript_call_result_21204 = invoke(stypy.reporting.localization.Localization(__file__, 564, 20), getitem___21203, int_21196)
    
    # Assigning a type to the variable 'tuple_var_assignment_20805' (line 564)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 564, 20), 'tuple_var_assignment_20805', subscript_call_result_21204)
    
    # Assigning a Name to a Name (line 564):
    # Getting the type of 'tuple_var_assignment_20804' (line 564)
    tuple_var_assignment_20804_21205 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 564, 20), 'tuple_var_assignment_20804')
    # Assigning a type to the variable 'idx' (line 564)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 564, 20), 'idx', tuple_var_assignment_20804_21205)
    
    # Assigning a Name to a Name (line 564):
    # Getting the type of 'tuple_var_assignment_20805' (line 564)
    tuple_var_assignment_20805_21206 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 564, 20), 'tuple_var_assignment_20805')
    # Assigning a type to the variable 'proj' (line 564)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 564, 25), 'proj', tuple_var_assignment_20805_21206)
    # SSA join for if statement (line 561)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA branch for the else part of an if statement (line 560)
    module_type_store.open_ssa_branch('else')
    
    # Getting the type of 'real' (line 566)
    real_21207 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 566, 19), 'real')
    # Testing the type of an if condition (line 566)
    if_condition_21208 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 566, 16), real_21207)
    # Assigning a type to the variable 'if_condition_21208' (line 566)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 566, 16), 'if_condition_21208', if_condition_21208)
    # SSA begins for if statement (line 566)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Tuple (line 567):
    
    # Assigning a Subscript to a Name (line 567):
    
    # Obtaining the type of the subscript
    int_21209 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 567, 20), 'int')
    
    # Call to iddr_id(...): (line 567)
    # Processing the call arguments (line 567)
    # Getting the type of 'A' (line 567)
    A_21212 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 567, 48), 'A', False)
    # Getting the type of 'k' (line 567)
    k_21213 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 567, 51), 'k', False)
    # Processing the call keyword arguments (line 567)
    kwargs_21214 = {}
    # Getting the type of 'backend' (line 567)
    backend_21210 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 567, 32), 'backend', False)
    # Obtaining the member 'iddr_id' of a type (line 567)
    iddr_id_21211 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 567, 32), backend_21210, 'iddr_id')
    # Calling iddr_id(args, kwargs) (line 567)
    iddr_id_call_result_21215 = invoke(stypy.reporting.localization.Localization(__file__, 567, 32), iddr_id_21211, *[A_21212, k_21213], **kwargs_21214)
    
    # Obtaining the member '__getitem__' of a type (line 567)
    getitem___21216 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 567, 20), iddr_id_call_result_21215, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 567)
    subscript_call_result_21217 = invoke(stypy.reporting.localization.Localization(__file__, 567, 20), getitem___21216, int_21209)
    
    # Assigning a type to the variable 'tuple_var_assignment_20806' (line 567)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 567, 20), 'tuple_var_assignment_20806', subscript_call_result_21217)
    
    # Assigning a Subscript to a Name (line 567):
    
    # Obtaining the type of the subscript
    int_21218 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 567, 20), 'int')
    
    # Call to iddr_id(...): (line 567)
    # Processing the call arguments (line 567)
    # Getting the type of 'A' (line 567)
    A_21221 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 567, 48), 'A', False)
    # Getting the type of 'k' (line 567)
    k_21222 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 567, 51), 'k', False)
    # Processing the call keyword arguments (line 567)
    kwargs_21223 = {}
    # Getting the type of 'backend' (line 567)
    backend_21219 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 567, 32), 'backend', False)
    # Obtaining the member 'iddr_id' of a type (line 567)
    iddr_id_21220 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 567, 32), backend_21219, 'iddr_id')
    # Calling iddr_id(args, kwargs) (line 567)
    iddr_id_call_result_21224 = invoke(stypy.reporting.localization.Localization(__file__, 567, 32), iddr_id_21220, *[A_21221, k_21222], **kwargs_21223)
    
    # Obtaining the member '__getitem__' of a type (line 567)
    getitem___21225 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 567, 20), iddr_id_call_result_21224, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 567)
    subscript_call_result_21226 = invoke(stypy.reporting.localization.Localization(__file__, 567, 20), getitem___21225, int_21218)
    
    # Assigning a type to the variable 'tuple_var_assignment_20807' (line 567)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 567, 20), 'tuple_var_assignment_20807', subscript_call_result_21226)
    
    # Assigning a Name to a Name (line 567):
    # Getting the type of 'tuple_var_assignment_20806' (line 567)
    tuple_var_assignment_20806_21227 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 567, 20), 'tuple_var_assignment_20806')
    # Assigning a type to the variable 'idx' (line 567)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 567, 20), 'idx', tuple_var_assignment_20806_21227)
    
    # Assigning a Name to a Name (line 567):
    # Getting the type of 'tuple_var_assignment_20807' (line 567)
    tuple_var_assignment_20807_21228 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 567, 20), 'tuple_var_assignment_20807')
    # Assigning a type to the variable 'proj' (line 567)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 567, 25), 'proj', tuple_var_assignment_20807_21228)
    # SSA branch for the else part of an if statement (line 566)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Call to a Tuple (line 569):
    
    # Assigning a Subscript to a Name (line 569):
    
    # Obtaining the type of the subscript
    int_21229 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 569, 20), 'int')
    
    # Call to idzr_id(...): (line 569)
    # Processing the call arguments (line 569)
    # Getting the type of 'A' (line 569)
    A_21232 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 569, 48), 'A', False)
    # Getting the type of 'k' (line 569)
    k_21233 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 569, 51), 'k', False)
    # Processing the call keyword arguments (line 569)
    kwargs_21234 = {}
    # Getting the type of 'backend' (line 569)
    backend_21230 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 569, 32), 'backend', False)
    # Obtaining the member 'idzr_id' of a type (line 569)
    idzr_id_21231 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 569, 32), backend_21230, 'idzr_id')
    # Calling idzr_id(args, kwargs) (line 569)
    idzr_id_call_result_21235 = invoke(stypy.reporting.localization.Localization(__file__, 569, 32), idzr_id_21231, *[A_21232, k_21233], **kwargs_21234)
    
    # Obtaining the member '__getitem__' of a type (line 569)
    getitem___21236 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 569, 20), idzr_id_call_result_21235, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 569)
    subscript_call_result_21237 = invoke(stypy.reporting.localization.Localization(__file__, 569, 20), getitem___21236, int_21229)
    
    # Assigning a type to the variable 'tuple_var_assignment_20808' (line 569)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 569, 20), 'tuple_var_assignment_20808', subscript_call_result_21237)
    
    # Assigning a Subscript to a Name (line 569):
    
    # Obtaining the type of the subscript
    int_21238 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 569, 20), 'int')
    
    # Call to idzr_id(...): (line 569)
    # Processing the call arguments (line 569)
    # Getting the type of 'A' (line 569)
    A_21241 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 569, 48), 'A', False)
    # Getting the type of 'k' (line 569)
    k_21242 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 569, 51), 'k', False)
    # Processing the call keyword arguments (line 569)
    kwargs_21243 = {}
    # Getting the type of 'backend' (line 569)
    backend_21239 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 569, 32), 'backend', False)
    # Obtaining the member 'idzr_id' of a type (line 569)
    idzr_id_21240 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 569, 32), backend_21239, 'idzr_id')
    # Calling idzr_id(args, kwargs) (line 569)
    idzr_id_call_result_21244 = invoke(stypy.reporting.localization.Localization(__file__, 569, 32), idzr_id_21240, *[A_21241, k_21242], **kwargs_21243)
    
    # Obtaining the member '__getitem__' of a type (line 569)
    getitem___21245 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 569, 20), idzr_id_call_result_21244, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 569)
    subscript_call_result_21246 = invoke(stypy.reporting.localization.Localization(__file__, 569, 20), getitem___21245, int_21238)
    
    # Assigning a type to the variable 'tuple_var_assignment_20809' (line 569)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 569, 20), 'tuple_var_assignment_20809', subscript_call_result_21246)
    
    # Assigning a Name to a Name (line 569):
    # Getting the type of 'tuple_var_assignment_20808' (line 569)
    tuple_var_assignment_20808_21247 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 569, 20), 'tuple_var_assignment_20808')
    # Assigning a type to the variable 'idx' (line 569)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 569, 20), 'idx', tuple_var_assignment_20808_21247)
    
    # Assigning a Name to a Name (line 569):
    # Getting the type of 'tuple_var_assignment_20809' (line 569)
    tuple_var_assignment_20809_21248 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 569, 20), 'tuple_var_assignment_20809')
    # Assigning a type to the variable 'proj' (line 569)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 569, 25), 'proj', tuple_var_assignment_20809_21248)
    # SSA join for if statement (line 566)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 560)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Obtaining an instance of the builtin type 'tuple' (line 570)
    tuple_21249 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 570, 19), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 570)
    # Adding element type (line 570)
    # Getting the type of 'idx' (line 570)
    idx_21250 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 570, 19), 'idx')
    int_21251 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 570, 25), 'int')
    # Applying the binary operator '-' (line 570)
    result_sub_21252 = python_operator(stypy.reporting.localization.Localization(__file__, 570, 19), '-', idx_21250, int_21251)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 570, 19), tuple_21249, result_sub_21252)
    # Adding element type (line 570)
    # Getting the type of 'proj' (line 570)
    proj_21253 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 570, 28), 'proj')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 570, 19), tuple_21249, proj_21253)
    
    # Assigning a type to the variable 'stypy_return_type' (line 570)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 570, 12), 'stypy_return_type', tuple_21249)
    # SSA join for if statement (line 545)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA branch for the else part of an if statement (line 544)
    module_type_store.open_ssa_branch('else')
    
    
    # Call to isinstance(...): (line 571)
    # Processing the call arguments (line 571)
    # Getting the type of 'A' (line 571)
    A_21255 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 571, 20), 'A', False)
    # Getting the type of 'LinearOperator' (line 571)
    LinearOperator_21256 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 571, 23), 'LinearOperator', False)
    # Processing the call keyword arguments (line 571)
    kwargs_21257 = {}
    # Getting the type of 'isinstance' (line 571)
    isinstance_21254 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 571, 9), 'isinstance', False)
    # Calling isinstance(args, kwargs) (line 571)
    isinstance_call_result_21258 = invoke(stypy.reporting.localization.Localization(__file__, 571, 9), isinstance_21254, *[A_21255, LinearOperator_21256], **kwargs_21257)
    
    # Testing the type of an if condition (line 571)
    if_condition_21259 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 571, 9), isinstance_call_result_21258)
    # Assigning a type to the variable 'if_condition_21259' (line 571)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 571, 9), 'if_condition_21259', if_condition_21259)
    # SSA begins for if statement (line 571)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Attribute to a Tuple (line 572):
    
    # Assigning a Subscript to a Name (line 572):
    
    # Obtaining the type of the subscript
    int_21260 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 572, 8), 'int')
    # Getting the type of 'A' (line 572)
    A_21261 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 572, 15), 'A')
    # Obtaining the member 'shape' of a type (line 572)
    shape_21262 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 572, 15), A_21261, 'shape')
    # Obtaining the member '__getitem__' of a type (line 572)
    getitem___21263 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 572, 8), shape_21262, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 572)
    subscript_call_result_21264 = invoke(stypy.reporting.localization.Localization(__file__, 572, 8), getitem___21263, int_21260)
    
    # Assigning a type to the variable 'tuple_var_assignment_20810' (line 572)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 572, 8), 'tuple_var_assignment_20810', subscript_call_result_21264)
    
    # Assigning a Subscript to a Name (line 572):
    
    # Obtaining the type of the subscript
    int_21265 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 572, 8), 'int')
    # Getting the type of 'A' (line 572)
    A_21266 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 572, 15), 'A')
    # Obtaining the member 'shape' of a type (line 572)
    shape_21267 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 572, 15), A_21266, 'shape')
    # Obtaining the member '__getitem__' of a type (line 572)
    getitem___21268 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 572, 8), shape_21267, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 572)
    subscript_call_result_21269 = invoke(stypy.reporting.localization.Localization(__file__, 572, 8), getitem___21268, int_21265)
    
    # Assigning a type to the variable 'tuple_var_assignment_20811' (line 572)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 572, 8), 'tuple_var_assignment_20811', subscript_call_result_21269)
    
    # Assigning a Name to a Name (line 572):
    # Getting the type of 'tuple_var_assignment_20810' (line 572)
    tuple_var_assignment_20810_21270 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 572, 8), 'tuple_var_assignment_20810')
    # Assigning a type to the variable 'm' (line 572)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 572, 8), 'm', tuple_var_assignment_20810_21270)
    
    # Assigning a Name to a Name (line 572):
    # Getting the type of 'tuple_var_assignment_20811' (line 572)
    tuple_var_assignment_20811_21271 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 572, 8), 'tuple_var_assignment_20811')
    # Assigning a type to the variable 'n' (line 572)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 572, 11), 'n', tuple_var_assignment_20811_21271)
    
    # Assigning a Attribute to a Name (line 573):
    
    # Assigning a Attribute to a Name (line 573):
    # Getting the type of 'A' (line 573)
    A_21272 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 573, 18), 'A')
    # Obtaining the member 'rmatvec' of a type (line 573)
    rmatvec_21273 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 573, 18), A_21272, 'rmatvec')
    # Assigning a type to the variable 'matveca' (line 573)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 573, 8), 'matveca', rmatvec_21273)
    
    
    # Getting the type of 'eps_or_k' (line 574)
    eps_or_k_21274 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 574, 11), 'eps_or_k')
    int_21275 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 574, 22), 'int')
    # Applying the binary operator '<' (line 574)
    result_lt_21276 = python_operator(stypy.reporting.localization.Localization(__file__, 574, 11), '<', eps_or_k_21274, int_21275)
    
    # Testing the type of an if condition (line 574)
    if_condition_21277 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 574, 8), result_lt_21276)
    # Assigning a type to the variable 'if_condition_21277' (line 574)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 574, 8), 'if_condition_21277', if_condition_21277)
    # SSA begins for if statement (line 574)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Name to a Name (line 575):
    
    # Assigning a Name to a Name (line 575):
    # Getting the type of 'eps_or_k' (line 575)
    eps_or_k_21278 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 575, 18), 'eps_or_k')
    # Assigning a type to the variable 'eps' (line 575)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 575, 12), 'eps', eps_or_k_21278)
    
    # Getting the type of 'real' (line 576)
    real_21279 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 576, 15), 'real')
    # Testing the type of an if condition (line 576)
    if_condition_21280 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 576, 12), real_21279)
    # Assigning a type to the variable 'if_condition_21280' (line 576)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 576, 12), 'if_condition_21280', if_condition_21280)
    # SSA begins for if statement (line 576)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Tuple (line 577):
    
    # Assigning a Subscript to a Name (line 577):
    
    # Obtaining the type of the subscript
    int_21281 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 577, 16), 'int')
    
    # Call to iddp_rid(...): (line 577)
    # Processing the call arguments (line 577)
    # Getting the type of 'eps' (line 577)
    eps_21284 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 577, 48), 'eps', False)
    # Getting the type of 'm' (line 577)
    m_21285 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 577, 53), 'm', False)
    # Getting the type of 'n' (line 577)
    n_21286 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 577, 56), 'n', False)
    # Getting the type of 'matveca' (line 577)
    matveca_21287 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 577, 59), 'matveca', False)
    # Processing the call keyword arguments (line 577)
    kwargs_21288 = {}
    # Getting the type of 'backend' (line 577)
    backend_21282 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 577, 31), 'backend', False)
    # Obtaining the member 'iddp_rid' of a type (line 577)
    iddp_rid_21283 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 577, 31), backend_21282, 'iddp_rid')
    # Calling iddp_rid(args, kwargs) (line 577)
    iddp_rid_call_result_21289 = invoke(stypy.reporting.localization.Localization(__file__, 577, 31), iddp_rid_21283, *[eps_21284, m_21285, n_21286, matveca_21287], **kwargs_21288)
    
    # Obtaining the member '__getitem__' of a type (line 577)
    getitem___21290 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 577, 16), iddp_rid_call_result_21289, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 577)
    subscript_call_result_21291 = invoke(stypy.reporting.localization.Localization(__file__, 577, 16), getitem___21290, int_21281)
    
    # Assigning a type to the variable 'tuple_var_assignment_20812' (line 577)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 577, 16), 'tuple_var_assignment_20812', subscript_call_result_21291)
    
    # Assigning a Subscript to a Name (line 577):
    
    # Obtaining the type of the subscript
    int_21292 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 577, 16), 'int')
    
    # Call to iddp_rid(...): (line 577)
    # Processing the call arguments (line 577)
    # Getting the type of 'eps' (line 577)
    eps_21295 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 577, 48), 'eps', False)
    # Getting the type of 'm' (line 577)
    m_21296 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 577, 53), 'm', False)
    # Getting the type of 'n' (line 577)
    n_21297 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 577, 56), 'n', False)
    # Getting the type of 'matveca' (line 577)
    matveca_21298 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 577, 59), 'matveca', False)
    # Processing the call keyword arguments (line 577)
    kwargs_21299 = {}
    # Getting the type of 'backend' (line 577)
    backend_21293 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 577, 31), 'backend', False)
    # Obtaining the member 'iddp_rid' of a type (line 577)
    iddp_rid_21294 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 577, 31), backend_21293, 'iddp_rid')
    # Calling iddp_rid(args, kwargs) (line 577)
    iddp_rid_call_result_21300 = invoke(stypy.reporting.localization.Localization(__file__, 577, 31), iddp_rid_21294, *[eps_21295, m_21296, n_21297, matveca_21298], **kwargs_21299)
    
    # Obtaining the member '__getitem__' of a type (line 577)
    getitem___21301 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 577, 16), iddp_rid_call_result_21300, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 577)
    subscript_call_result_21302 = invoke(stypy.reporting.localization.Localization(__file__, 577, 16), getitem___21301, int_21292)
    
    # Assigning a type to the variable 'tuple_var_assignment_20813' (line 577)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 577, 16), 'tuple_var_assignment_20813', subscript_call_result_21302)
    
    # Assigning a Subscript to a Name (line 577):
    
    # Obtaining the type of the subscript
    int_21303 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 577, 16), 'int')
    
    # Call to iddp_rid(...): (line 577)
    # Processing the call arguments (line 577)
    # Getting the type of 'eps' (line 577)
    eps_21306 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 577, 48), 'eps', False)
    # Getting the type of 'm' (line 577)
    m_21307 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 577, 53), 'm', False)
    # Getting the type of 'n' (line 577)
    n_21308 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 577, 56), 'n', False)
    # Getting the type of 'matveca' (line 577)
    matveca_21309 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 577, 59), 'matveca', False)
    # Processing the call keyword arguments (line 577)
    kwargs_21310 = {}
    # Getting the type of 'backend' (line 577)
    backend_21304 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 577, 31), 'backend', False)
    # Obtaining the member 'iddp_rid' of a type (line 577)
    iddp_rid_21305 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 577, 31), backend_21304, 'iddp_rid')
    # Calling iddp_rid(args, kwargs) (line 577)
    iddp_rid_call_result_21311 = invoke(stypy.reporting.localization.Localization(__file__, 577, 31), iddp_rid_21305, *[eps_21306, m_21307, n_21308, matveca_21309], **kwargs_21310)
    
    # Obtaining the member '__getitem__' of a type (line 577)
    getitem___21312 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 577, 16), iddp_rid_call_result_21311, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 577)
    subscript_call_result_21313 = invoke(stypy.reporting.localization.Localization(__file__, 577, 16), getitem___21312, int_21303)
    
    # Assigning a type to the variable 'tuple_var_assignment_20814' (line 577)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 577, 16), 'tuple_var_assignment_20814', subscript_call_result_21313)
    
    # Assigning a Name to a Name (line 577):
    # Getting the type of 'tuple_var_assignment_20812' (line 577)
    tuple_var_assignment_20812_21314 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 577, 16), 'tuple_var_assignment_20812')
    # Assigning a type to the variable 'k' (line 577)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 577, 16), 'k', tuple_var_assignment_20812_21314)
    
    # Assigning a Name to a Name (line 577):
    # Getting the type of 'tuple_var_assignment_20813' (line 577)
    tuple_var_assignment_20813_21315 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 577, 16), 'tuple_var_assignment_20813')
    # Assigning a type to the variable 'idx' (line 577)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 577, 19), 'idx', tuple_var_assignment_20813_21315)
    
    # Assigning a Name to a Name (line 577):
    # Getting the type of 'tuple_var_assignment_20814' (line 577)
    tuple_var_assignment_20814_21316 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 577, 16), 'tuple_var_assignment_20814')
    # Assigning a type to the variable 'proj' (line 577)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 577, 24), 'proj', tuple_var_assignment_20814_21316)
    # SSA branch for the else part of an if statement (line 576)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Call to a Tuple (line 579):
    
    # Assigning a Subscript to a Name (line 579):
    
    # Obtaining the type of the subscript
    int_21317 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 579, 16), 'int')
    
    # Call to idzp_rid(...): (line 579)
    # Processing the call arguments (line 579)
    # Getting the type of 'eps' (line 579)
    eps_21320 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 579, 48), 'eps', False)
    # Getting the type of 'm' (line 579)
    m_21321 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 579, 53), 'm', False)
    # Getting the type of 'n' (line 579)
    n_21322 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 579, 56), 'n', False)
    # Getting the type of 'matveca' (line 579)
    matveca_21323 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 579, 59), 'matveca', False)
    # Processing the call keyword arguments (line 579)
    kwargs_21324 = {}
    # Getting the type of 'backend' (line 579)
    backend_21318 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 579, 31), 'backend', False)
    # Obtaining the member 'idzp_rid' of a type (line 579)
    idzp_rid_21319 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 579, 31), backend_21318, 'idzp_rid')
    # Calling idzp_rid(args, kwargs) (line 579)
    idzp_rid_call_result_21325 = invoke(stypy.reporting.localization.Localization(__file__, 579, 31), idzp_rid_21319, *[eps_21320, m_21321, n_21322, matveca_21323], **kwargs_21324)
    
    # Obtaining the member '__getitem__' of a type (line 579)
    getitem___21326 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 579, 16), idzp_rid_call_result_21325, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 579)
    subscript_call_result_21327 = invoke(stypy.reporting.localization.Localization(__file__, 579, 16), getitem___21326, int_21317)
    
    # Assigning a type to the variable 'tuple_var_assignment_20815' (line 579)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 579, 16), 'tuple_var_assignment_20815', subscript_call_result_21327)
    
    # Assigning a Subscript to a Name (line 579):
    
    # Obtaining the type of the subscript
    int_21328 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 579, 16), 'int')
    
    # Call to idzp_rid(...): (line 579)
    # Processing the call arguments (line 579)
    # Getting the type of 'eps' (line 579)
    eps_21331 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 579, 48), 'eps', False)
    # Getting the type of 'm' (line 579)
    m_21332 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 579, 53), 'm', False)
    # Getting the type of 'n' (line 579)
    n_21333 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 579, 56), 'n', False)
    # Getting the type of 'matveca' (line 579)
    matveca_21334 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 579, 59), 'matveca', False)
    # Processing the call keyword arguments (line 579)
    kwargs_21335 = {}
    # Getting the type of 'backend' (line 579)
    backend_21329 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 579, 31), 'backend', False)
    # Obtaining the member 'idzp_rid' of a type (line 579)
    idzp_rid_21330 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 579, 31), backend_21329, 'idzp_rid')
    # Calling idzp_rid(args, kwargs) (line 579)
    idzp_rid_call_result_21336 = invoke(stypy.reporting.localization.Localization(__file__, 579, 31), idzp_rid_21330, *[eps_21331, m_21332, n_21333, matveca_21334], **kwargs_21335)
    
    # Obtaining the member '__getitem__' of a type (line 579)
    getitem___21337 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 579, 16), idzp_rid_call_result_21336, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 579)
    subscript_call_result_21338 = invoke(stypy.reporting.localization.Localization(__file__, 579, 16), getitem___21337, int_21328)
    
    # Assigning a type to the variable 'tuple_var_assignment_20816' (line 579)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 579, 16), 'tuple_var_assignment_20816', subscript_call_result_21338)
    
    # Assigning a Subscript to a Name (line 579):
    
    # Obtaining the type of the subscript
    int_21339 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 579, 16), 'int')
    
    # Call to idzp_rid(...): (line 579)
    # Processing the call arguments (line 579)
    # Getting the type of 'eps' (line 579)
    eps_21342 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 579, 48), 'eps', False)
    # Getting the type of 'm' (line 579)
    m_21343 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 579, 53), 'm', False)
    # Getting the type of 'n' (line 579)
    n_21344 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 579, 56), 'n', False)
    # Getting the type of 'matveca' (line 579)
    matveca_21345 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 579, 59), 'matveca', False)
    # Processing the call keyword arguments (line 579)
    kwargs_21346 = {}
    # Getting the type of 'backend' (line 579)
    backend_21340 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 579, 31), 'backend', False)
    # Obtaining the member 'idzp_rid' of a type (line 579)
    idzp_rid_21341 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 579, 31), backend_21340, 'idzp_rid')
    # Calling idzp_rid(args, kwargs) (line 579)
    idzp_rid_call_result_21347 = invoke(stypy.reporting.localization.Localization(__file__, 579, 31), idzp_rid_21341, *[eps_21342, m_21343, n_21344, matveca_21345], **kwargs_21346)
    
    # Obtaining the member '__getitem__' of a type (line 579)
    getitem___21348 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 579, 16), idzp_rid_call_result_21347, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 579)
    subscript_call_result_21349 = invoke(stypy.reporting.localization.Localization(__file__, 579, 16), getitem___21348, int_21339)
    
    # Assigning a type to the variable 'tuple_var_assignment_20817' (line 579)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 579, 16), 'tuple_var_assignment_20817', subscript_call_result_21349)
    
    # Assigning a Name to a Name (line 579):
    # Getting the type of 'tuple_var_assignment_20815' (line 579)
    tuple_var_assignment_20815_21350 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 579, 16), 'tuple_var_assignment_20815')
    # Assigning a type to the variable 'k' (line 579)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 579, 16), 'k', tuple_var_assignment_20815_21350)
    
    # Assigning a Name to a Name (line 579):
    # Getting the type of 'tuple_var_assignment_20816' (line 579)
    tuple_var_assignment_20816_21351 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 579, 16), 'tuple_var_assignment_20816')
    # Assigning a type to the variable 'idx' (line 579)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 579, 19), 'idx', tuple_var_assignment_20816_21351)
    
    # Assigning a Name to a Name (line 579):
    # Getting the type of 'tuple_var_assignment_20817' (line 579)
    tuple_var_assignment_20817_21352 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 579, 16), 'tuple_var_assignment_20817')
    # Assigning a type to the variable 'proj' (line 579)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 579, 24), 'proj', tuple_var_assignment_20817_21352)
    # SSA join for if statement (line 576)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Obtaining an instance of the builtin type 'tuple' (line 580)
    tuple_21353 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 580, 19), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 580)
    # Adding element type (line 580)
    # Getting the type of 'k' (line 580)
    k_21354 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 580, 19), 'k')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 580, 19), tuple_21353, k_21354)
    # Adding element type (line 580)
    # Getting the type of 'idx' (line 580)
    idx_21355 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 580, 22), 'idx')
    int_21356 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 580, 28), 'int')
    # Applying the binary operator '-' (line 580)
    result_sub_21357 = python_operator(stypy.reporting.localization.Localization(__file__, 580, 22), '-', idx_21355, int_21356)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 580, 19), tuple_21353, result_sub_21357)
    # Adding element type (line 580)
    # Getting the type of 'proj' (line 580)
    proj_21358 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 580, 31), 'proj')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 580, 19), tuple_21353, proj_21358)
    
    # Assigning a type to the variable 'stypy_return_type' (line 580)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 580, 12), 'stypy_return_type', tuple_21353)
    # SSA branch for the else part of an if statement (line 574)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Call to a Name (line 582):
    
    # Assigning a Call to a Name (line 582):
    
    # Call to int(...): (line 582)
    # Processing the call arguments (line 582)
    # Getting the type of 'eps_or_k' (line 582)
    eps_or_k_21360 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 582, 20), 'eps_or_k', False)
    # Processing the call keyword arguments (line 582)
    kwargs_21361 = {}
    # Getting the type of 'int' (line 582)
    int_21359 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 582, 16), 'int', False)
    # Calling int(args, kwargs) (line 582)
    int_call_result_21362 = invoke(stypy.reporting.localization.Localization(__file__, 582, 16), int_21359, *[eps_or_k_21360], **kwargs_21361)
    
    # Assigning a type to the variable 'k' (line 582)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 582, 12), 'k', int_call_result_21362)
    
    # Getting the type of 'real' (line 583)
    real_21363 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 583, 15), 'real')
    # Testing the type of an if condition (line 583)
    if_condition_21364 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 583, 12), real_21363)
    # Assigning a type to the variable 'if_condition_21364' (line 583)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 583, 12), 'if_condition_21364', if_condition_21364)
    # SSA begins for if statement (line 583)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Tuple (line 584):
    
    # Assigning a Subscript to a Name (line 584):
    
    # Obtaining the type of the subscript
    int_21365 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 584, 16), 'int')
    
    # Call to iddr_rid(...): (line 584)
    # Processing the call arguments (line 584)
    # Getting the type of 'm' (line 584)
    m_21368 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 584, 45), 'm', False)
    # Getting the type of 'n' (line 584)
    n_21369 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 584, 48), 'n', False)
    # Getting the type of 'matveca' (line 584)
    matveca_21370 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 584, 51), 'matveca', False)
    # Getting the type of 'k' (line 584)
    k_21371 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 584, 60), 'k', False)
    # Processing the call keyword arguments (line 584)
    kwargs_21372 = {}
    # Getting the type of 'backend' (line 584)
    backend_21366 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 584, 28), 'backend', False)
    # Obtaining the member 'iddr_rid' of a type (line 584)
    iddr_rid_21367 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 584, 28), backend_21366, 'iddr_rid')
    # Calling iddr_rid(args, kwargs) (line 584)
    iddr_rid_call_result_21373 = invoke(stypy.reporting.localization.Localization(__file__, 584, 28), iddr_rid_21367, *[m_21368, n_21369, matveca_21370, k_21371], **kwargs_21372)
    
    # Obtaining the member '__getitem__' of a type (line 584)
    getitem___21374 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 584, 16), iddr_rid_call_result_21373, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 584)
    subscript_call_result_21375 = invoke(stypy.reporting.localization.Localization(__file__, 584, 16), getitem___21374, int_21365)
    
    # Assigning a type to the variable 'tuple_var_assignment_20818' (line 584)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 584, 16), 'tuple_var_assignment_20818', subscript_call_result_21375)
    
    # Assigning a Subscript to a Name (line 584):
    
    # Obtaining the type of the subscript
    int_21376 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 584, 16), 'int')
    
    # Call to iddr_rid(...): (line 584)
    # Processing the call arguments (line 584)
    # Getting the type of 'm' (line 584)
    m_21379 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 584, 45), 'm', False)
    # Getting the type of 'n' (line 584)
    n_21380 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 584, 48), 'n', False)
    # Getting the type of 'matveca' (line 584)
    matveca_21381 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 584, 51), 'matveca', False)
    # Getting the type of 'k' (line 584)
    k_21382 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 584, 60), 'k', False)
    # Processing the call keyword arguments (line 584)
    kwargs_21383 = {}
    # Getting the type of 'backend' (line 584)
    backend_21377 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 584, 28), 'backend', False)
    # Obtaining the member 'iddr_rid' of a type (line 584)
    iddr_rid_21378 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 584, 28), backend_21377, 'iddr_rid')
    # Calling iddr_rid(args, kwargs) (line 584)
    iddr_rid_call_result_21384 = invoke(stypy.reporting.localization.Localization(__file__, 584, 28), iddr_rid_21378, *[m_21379, n_21380, matveca_21381, k_21382], **kwargs_21383)
    
    # Obtaining the member '__getitem__' of a type (line 584)
    getitem___21385 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 584, 16), iddr_rid_call_result_21384, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 584)
    subscript_call_result_21386 = invoke(stypy.reporting.localization.Localization(__file__, 584, 16), getitem___21385, int_21376)
    
    # Assigning a type to the variable 'tuple_var_assignment_20819' (line 584)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 584, 16), 'tuple_var_assignment_20819', subscript_call_result_21386)
    
    # Assigning a Name to a Name (line 584):
    # Getting the type of 'tuple_var_assignment_20818' (line 584)
    tuple_var_assignment_20818_21387 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 584, 16), 'tuple_var_assignment_20818')
    # Assigning a type to the variable 'idx' (line 584)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 584, 16), 'idx', tuple_var_assignment_20818_21387)
    
    # Assigning a Name to a Name (line 584):
    # Getting the type of 'tuple_var_assignment_20819' (line 584)
    tuple_var_assignment_20819_21388 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 584, 16), 'tuple_var_assignment_20819')
    # Assigning a type to the variable 'proj' (line 584)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 584, 21), 'proj', tuple_var_assignment_20819_21388)
    # SSA branch for the else part of an if statement (line 583)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Call to a Tuple (line 586):
    
    # Assigning a Subscript to a Name (line 586):
    
    # Obtaining the type of the subscript
    int_21389 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 586, 16), 'int')
    
    # Call to idzr_rid(...): (line 586)
    # Processing the call arguments (line 586)
    # Getting the type of 'm' (line 586)
    m_21392 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 586, 45), 'm', False)
    # Getting the type of 'n' (line 586)
    n_21393 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 586, 48), 'n', False)
    # Getting the type of 'matveca' (line 586)
    matveca_21394 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 586, 51), 'matveca', False)
    # Getting the type of 'k' (line 586)
    k_21395 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 586, 60), 'k', False)
    # Processing the call keyword arguments (line 586)
    kwargs_21396 = {}
    # Getting the type of 'backend' (line 586)
    backend_21390 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 586, 28), 'backend', False)
    # Obtaining the member 'idzr_rid' of a type (line 586)
    idzr_rid_21391 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 586, 28), backend_21390, 'idzr_rid')
    # Calling idzr_rid(args, kwargs) (line 586)
    idzr_rid_call_result_21397 = invoke(stypy.reporting.localization.Localization(__file__, 586, 28), idzr_rid_21391, *[m_21392, n_21393, matveca_21394, k_21395], **kwargs_21396)
    
    # Obtaining the member '__getitem__' of a type (line 586)
    getitem___21398 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 586, 16), idzr_rid_call_result_21397, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 586)
    subscript_call_result_21399 = invoke(stypy.reporting.localization.Localization(__file__, 586, 16), getitem___21398, int_21389)
    
    # Assigning a type to the variable 'tuple_var_assignment_20820' (line 586)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 586, 16), 'tuple_var_assignment_20820', subscript_call_result_21399)
    
    # Assigning a Subscript to a Name (line 586):
    
    # Obtaining the type of the subscript
    int_21400 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 586, 16), 'int')
    
    # Call to idzr_rid(...): (line 586)
    # Processing the call arguments (line 586)
    # Getting the type of 'm' (line 586)
    m_21403 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 586, 45), 'm', False)
    # Getting the type of 'n' (line 586)
    n_21404 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 586, 48), 'n', False)
    # Getting the type of 'matveca' (line 586)
    matveca_21405 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 586, 51), 'matveca', False)
    # Getting the type of 'k' (line 586)
    k_21406 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 586, 60), 'k', False)
    # Processing the call keyword arguments (line 586)
    kwargs_21407 = {}
    # Getting the type of 'backend' (line 586)
    backend_21401 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 586, 28), 'backend', False)
    # Obtaining the member 'idzr_rid' of a type (line 586)
    idzr_rid_21402 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 586, 28), backend_21401, 'idzr_rid')
    # Calling idzr_rid(args, kwargs) (line 586)
    idzr_rid_call_result_21408 = invoke(stypy.reporting.localization.Localization(__file__, 586, 28), idzr_rid_21402, *[m_21403, n_21404, matveca_21405, k_21406], **kwargs_21407)
    
    # Obtaining the member '__getitem__' of a type (line 586)
    getitem___21409 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 586, 16), idzr_rid_call_result_21408, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 586)
    subscript_call_result_21410 = invoke(stypy.reporting.localization.Localization(__file__, 586, 16), getitem___21409, int_21400)
    
    # Assigning a type to the variable 'tuple_var_assignment_20821' (line 586)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 586, 16), 'tuple_var_assignment_20821', subscript_call_result_21410)
    
    # Assigning a Name to a Name (line 586):
    # Getting the type of 'tuple_var_assignment_20820' (line 586)
    tuple_var_assignment_20820_21411 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 586, 16), 'tuple_var_assignment_20820')
    # Assigning a type to the variable 'idx' (line 586)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 586, 16), 'idx', tuple_var_assignment_20820_21411)
    
    # Assigning a Name to a Name (line 586):
    # Getting the type of 'tuple_var_assignment_20821' (line 586)
    tuple_var_assignment_20821_21412 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 586, 16), 'tuple_var_assignment_20821')
    # Assigning a type to the variable 'proj' (line 586)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 586, 21), 'proj', tuple_var_assignment_20821_21412)
    # SSA join for if statement (line 583)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Obtaining an instance of the builtin type 'tuple' (line 587)
    tuple_21413 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 587, 19), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 587)
    # Adding element type (line 587)
    # Getting the type of 'idx' (line 587)
    idx_21414 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 587, 19), 'idx')
    int_21415 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 587, 25), 'int')
    # Applying the binary operator '-' (line 587)
    result_sub_21416 = python_operator(stypy.reporting.localization.Localization(__file__, 587, 19), '-', idx_21414, int_21415)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 587, 19), tuple_21413, result_sub_21416)
    # Adding element type (line 587)
    # Getting the type of 'proj' (line 587)
    proj_21417 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 587, 28), 'proj')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 587, 19), tuple_21413, proj_21417)
    
    # Assigning a type to the variable 'stypy_return_type' (line 587)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 587, 12), 'stypy_return_type', tuple_21413)
    # SSA join for if statement (line 574)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA branch for the else part of an if statement (line 571)
    module_type_store.open_ssa_branch('else')
    # Getting the type of '_TYPE_ERROR' (line 589)
    _TYPE_ERROR_21418 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 589, 14), '_TYPE_ERROR')
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 589, 8), _TYPE_ERROR_21418, 'raise parameter', BaseException)
    # SSA join for if statement (line 571)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 544)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'interp_decomp(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'interp_decomp' in the type store
    # Getting the type of 'stypy_return_type' (line 465)
    stypy_return_type_21419 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 465, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_21419)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'interp_decomp'
    return stypy_return_type_21419

# Assigning a type to the variable 'interp_decomp' (line 465)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 465, 0), 'interp_decomp', interp_decomp)

@norecursion
def reconstruct_matrix_from_id(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'reconstruct_matrix_from_id'
    module_type_store = module_type_store.open_function_context('reconstruct_matrix_from_id', 592, 0, False)
    
    # Passed parameters checking function
    reconstruct_matrix_from_id.stypy_localization = localization
    reconstruct_matrix_from_id.stypy_type_of_self = None
    reconstruct_matrix_from_id.stypy_type_store = module_type_store
    reconstruct_matrix_from_id.stypy_function_name = 'reconstruct_matrix_from_id'
    reconstruct_matrix_from_id.stypy_param_names_list = ['B', 'idx', 'proj']
    reconstruct_matrix_from_id.stypy_varargs_param_name = None
    reconstruct_matrix_from_id.stypy_kwargs_param_name = None
    reconstruct_matrix_from_id.stypy_call_defaults = defaults
    reconstruct_matrix_from_id.stypy_call_varargs = varargs
    reconstruct_matrix_from_id.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'reconstruct_matrix_from_id', ['B', 'idx', 'proj'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'reconstruct_matrix_from_id', localization, ['B', 'idx', 'proj'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'reconstruct_matrix_from_id(...)' code ##################

    str_21420 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 621, (-1)), 'str', '\n    Reconstruct matrix from its ID.\n\n    A matrix `A` with skeleton matrix `B` and ID indices and coefficients `idx`\n    and `proj`, respectively, can be reconstructed as::\n\n        numpy.hstack([B, numpy.dot(B, proj)])[:,numpy.argsort(idx)]\n\n    See also :func:`reconstruct_interp_matrix` and\n    :func:`reconstruct_skel_matrix`.\n\n    ..  This function automatically detects the matrix data type and calls the\n        appropriate backend. For details, see :func:`backend.idd_reconid` and\n        :func:`backend.idz_reconid`.\n\n    Parameters\n    ----------\n    B : :class:`numpy.ndarray`\n        Skeleton matrix.\n    idx : :class:`numpy.ndarray`\n        Column index array.\n    proj : :class:`numpy.ndarray`\n        Interpolation coefficients.\n\n    Returns\n    -------\n    :class:`numpy.ndarray`\n        Reconstructed matrix.\n    ')
    
    
    # Call to _is_real(...): (line 622)
    # Processing the call arguments (line 622)
    # Getting the type of 'B' (line 622)
    B_21422 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 622, 16), 'B', False)
    # Processing the call keyword arguments (line 622)
    kwargs_21423 = {}
    # Getting the type of '_is_real' (line 622)
    _is_real_21421 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 622, 7), '_is_real', False)
    # Calling _is_real(args, kwargs) (line 622)
    _is_real_call_result_21424 = invoke(stypy.reporting.localization.Localization(__file__, 622, 7), _is_real_21421, *[B_21422], **kwargs_21423)
    
    # Testing the type of an if condition (line 622)
    if_condition_21425 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 622, 4), _is_real_call_result_21424)
    # Assigning a type to the variable 'if_condition_21425' (line 622)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 622, 4), 'if_condition_21425', if_condition_21425)
    # SSA begins for if statement (line 622)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to idd_reconid(...): (line 623)
    # Processing the call arguments (line 623)
    # Getting the type of 'B' (line 623)
    B_21428 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 623, 35), 'B', False)
    # Getting the type of 'idx' (line 623)
    idx_21429 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 623, 38), 'idx', False)
    int_21430 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 623, 44), 'int')
    # Applying the binary operator '+' (line 623)
    result_add_21431 = python_operator(stypy.reporting.localization.Localization(__file__, 623, 38), '+', idx_21429, int_21430)
    
    # Getting the type of 'proj' (line 623)
    proj_21432 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 623, 47), 'proj', False)
    # Processing the call keyword arguments (line 623)
    kwargs_21433 = {}
    # Getting the type of 'backend' (line 623)
    backend_21426 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 623, 15), 'backend', False)
    # Obtaining the member 'idd_reconid' of a type (line 623)
    idd_reconid_21427 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 623, 15), backend_21426, 'idd_reconid')
    # Calling idd_reconid(args, kwargs) (line 623)
    idd_reconid_call_result_21434 = invoke(stypy.reporting.localization.Localization(__file__, 623, 15), idd_reconid_21427, *[B_21428, result_add_21431, proj_21432], **kwargs_21433)
    
    # Assigning a type to the variable 'stypy_return_type' (line 623)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 623, 8), 'stypy_return_type', idd_reconid_call_result_21434)
    # SSA branch for the else part of an if statement (line 622)
    module_type_store.open_ssa_branch('else')
    
    # Call to idz_reconid(...): (line 625)
    # Processing the call arguments (line 625)
    # Getting the type of 'B' (line 625)
    B_21437 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 625, 35), 'B', False)
    # Getting the type of 'idx' (line 625)
    idx_21438 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 625, 38), 'idx', False)
    int_21439 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 625, 44), 'int')
    # Applying the binary operator '+' (line 625)
    result_add_21440 = python_operator(stypy.reporting.localization.Localization(__file__, 625, 38), '+', idx_21438, int_21439)
    
    # Getting the type of 'proj' (line 625)
    proj_21441 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 625, 47), 'proj', False)
    # Processing the call keyword arguments (line 625)
    kwargs_21442 = {}
    # Getting the type of 'backend' (line 625)
    backend_21435 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 625, 15), 'backend', False)
    # Obtaining the member 'idz_reconid' of a type (line 625)
    idz_reconid_21436 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 625, 15), backend_21435, 'idz_reconid')
    # Calling idz_reconid(args, kwargs) (line 625)
    idz_reconid_call_result_21443 = invoke(stypy.reporting.localization.Localization(__file__, 625, 15), idz_reconid_21436, *[B_21437, result_add_21440, proj_21441], **kwargs_21442)
    
    # Assigning a type to the variable 'stypy_return_type' (line 625)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 625, 8), 'stypy_return_type', idz_reconid_call_result_21443)
    # SSA join for if statement (line 622)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'reconstruct_matrix_from_id(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'reconstruct_matrix_from_id' in the type store
    # Getting the type of 'stypy_return_type' (line 592)
    stypy_return_type_21444 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 592, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_21444)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'reconstruct_matrix_from_id'
    return stypy_return_type_21444

# Assigning a type to the variable 'reconstruct_matrix_from_id' (line 592)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 592, 0), 'reconstruct_matrix_from_id', reconstruct_matrix_from_id)

@norecursion
def reconstruct_interp_matrix(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'reconstruct_interp_matrix'
    module_type_store = module_type_store.open_function_context('reconstruct_interp_matrix', 628, 0, False)
    
    # Passed parameters checking function
    reconstruct_interp_matrix.stypy_localization = localization
    reconstruct_interp_matrix.stypy_type_of_self = None
    reconstruct_interp_matrix.stypy_type_store = module_type_store
    reconstruct_interp_matrix.stypy_function_name = 'reconstruct_interp_matrix'
    reconstruct_interp_matrix.stypy_param_names_list = ['idx', 'proj']
    reconstruct_interp_matrix.stypy_varargs_param_name = None
    reconstruct_interp_matrix.stypy_kwargs_param_name = None
    reconstruct_interp_matrix.stypy_call_defaults = defaults
    reconstruct_interp_matrix.stypy_call_varargs = varargs
    reconstruct_interp_matrix.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'reconstruct_interp_matrix', ['idx', 'proj'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'reconstruct_interp_matrix', localization, ['idx', 'proj'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'reconstruct_interp_matrix(...)' code ##################

    str_21445 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 660, (-1)), 'str', '\n    Reconstruct interpolation matrix from ID.\n\n    The interpolation matrix can be reconstructed from the ID indices and\n    coefficients `idx` and `proj`, respectively, as::\n\n        P = numpy.hstack([numpy.eye(proj.shape[0]), proj])[:,numpy.argsort(idx)]\n\n    The original matrix can then be reconstructed from its skeleton matrix `B`\n    via::\n\n        numpy.dot(B, P)\n\n    See also :func:`reconstruct_matrix_from_id` and\n    :func:`reconstruct_skel_matrix`.\n\n    ..  This function automatically detects the matrix data type and calls the\n        appropriate backend. For details, see :func:`backend.idd_reconint` and\n        :func:`backend.idz_reconint`.\n\n    Parameters\n    ----------\n    idx : :class:`numpy.ndarray`\n        Column index array.\n    proj : :class:`numpy.ndarray`\n        Interpolation coefficients.\n\n    Returns\n    -------\n    :class:`numpy.ndarray`\n        Interpolation matrix.\n    ')
    
    
    # Call to _is_real(...): (line 661)
    # Processing the call arguments (line 661)
    # Getting the type of 'proj' (line 661)
    proj_21447 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 661, 16), 'proj', False)
    # Processing the call keyword arguments (line 661)
    kwargs_21448 = {}
    # Getting the type of '_is_real' (line 661)
    _is_real_21446 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 661, 7), '_is_real', False)
    # Calling _is_real(args, kwargs) (line 661)
    _is_real_call_result_21449 = invoke(stypy.reporting.localization.Localization(__file__, 661, 7), _is_real_21446, *[proj_21447], **kwargs_21448)
    
    # Testing the type of an if condition (line 661)
    if_condition_21450 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 661, 4), _is_real_call_result_21449)
    # Assigning a type to the variable 'if_condition_21450' (line 661)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 661, 4), 'if_condition_21450', if_condition_21450)
    # SSA begins for if statement (line 661)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to idd_reconint(...): (line 662)
    # Processing the call arguments (line 662)
    # Getting the type of 'idx' (line 662)
    idx_21453 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 662, 36), 'idx', False)
    int_21454 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 662, 42), 'int')
    # Applying the binary operator '+' (line 662)
    result_add_21455 = python_operator(stypy.reporting.localization.Localization(__file__, 662, 36), '+', idx_21453, int_21454)
    
    # Getting the type of 'proj' (line 662)
    proj_21456 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 662, 45), 'proj', False)
    # Processing the call keyword arguments (line 662)
    kwargs_21457 = {}
    # Getting the type of 'backend' (line 662)
    backend_21451 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 662, 15), 'backend', False)
    # Obtaining the member 'idd_reconint' of a type (line 662)
    idd_reconint_21452 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 662, 15), backend_21451, 'idd_reconint')
    # Calling idd_reconint(args, kwargs) (line 662)
    idd_reconint_call_result_21458 = invoke(stypy.reporting.localization.Localization(__file__, 662, 15), idd_reconint_21452, *[result_add_21455, proj_21456], **kwargs_21457)
    
    # Assigning a type to the variable 'stypy_return_type' (line 662)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 662, 8), 'stypy_return_type', idd_reconint_call_result_21458)
    # SSA branch for the else part of an if statement (line 661)
    module_type_store.open_ssa_branch('else')
    
    # Call to idz_reconint(...): (line 664)
    # Processing the call arguments (line 664)
    # Getting the type of 'idx' (line 664)
    idx_21461 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 664, 36), 'idx', False)
    int_21462 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 664, 42), 'int')
    # Applying the binary operator '+' (line 664)
    result_add_21463 = python_operator(stypy.reporting.localization.Localization(__file__, 664, 36), '+', idx_21461, int_21462)
    
    # Getting the type of 'proj' (line 664)
    proj_21464 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 664, 45), 'proj', False)
    # Processing the call keyword arguments (line 664)
    kwargs_21465 = {}
    # Getting the type of 'backend' (line 664)
    backend_21459 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 664, 15), 'backend', False)
    # Obtaining the member 'idz_reconint' of a type (line 664)
    idz_reconint_21460 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 664, 15), backend_21459, 'idz_reconint')
    # Calling idz_reconint(args, kwargs) (line 664)
    idz_reconint_call_result_21466 = invoke(stypy.reporting.localization.Localization(__file__, 664, 15), idz_reconint_21460, *[result_add_21463, proj_21464], **kwargs_21465)
    
    # Assigning a type to the variable 'stypy_return_type' (line 664)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 664, 8), 'stypy_return_type', idz_reconint_call_result_21466)
    # SSA join for if statement (line 661)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'reconstruct_interp_matrix(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'reconstruct_interp_matrix' in the type store
    # Getting the type of 'stypy_return_type' (line 628)
    stypy_return_type_21467 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 628, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_21467)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'reconstruct_interp_matrix'
    return stypy_return_type_21467

# Assigning a type to the variable 'reconstruct_interp_matrix' (line 628)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 628, 0), 'reconstruct_interp_matrix', reconstruct_interp_matrix)

@norecursion
def reconstruct_skel_matrix(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'reconstruct_skel_matrix'
    module_type_store = module_type_store.open_function_context('reconstruct_skel_matrix', 667, 0, False)
    
    # Passed parameters checking function
    reconstruct_skel_matrix.stypy_localization = localization
    reconstruct_skel_matrix.stypy_type_of_self = None
    reconstruct_skel_matrix.stypy_type_store = module_type_store
    reconstruct_skel_matrix.stypy_function_name = 'reconstruct_skel_matrix'
    reconstruct_skel_matrix.stypy_param_names_list = ['A', 'k', 'idx']
    reconstruct_skel_matrix.stypy_varargs_param_name = None
    reconstruct_skel_matrix.stypy_kwargs_param_name = None
    reconstruct_skel_matrix.stypy_call_defaults = defaults
    reconstruct_skel_matrix.stypy_call_varargs = varargs
    reconstruct_skel_matrix.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'reconstruct_skel_matrix', ['A', 'k', 'idx'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'reconstruct_skel_matrix', localization, ['A', 'k', 'idx'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'reconstruct_skel_matrix(...)' code ##################

    str_21468 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 700, (-1)), 'str', '\n    Reconstruct skeleton matrix from ID.\n\n    The skeleton matrix can be reconstructed from the original matrix `A` and its\n    ID rank and indices `k` and `idx`, respectively, as::\n\n        B = A[:,idx[:k]]\n\n    The original matrix can then be reconstructed via::\n\n        numpy.hstack([B, numpy.dot(B, proj)])[:,numpy.argsort(idx)]\n\n    See also :func:`reconstruct_matrix_from_id` and\n    :func:`reconstruct_interp_matrix`.\n\n    ..  This function automatically detects the matrix data type and calls the\n        appropriate backend. For details, see :func:`backend.idd_copycols` and\n        :func:`backend.idz_copycols`.\n\n    Parameters\n    ----------\n    A : :class:`numpy.ndarray`\n        Original matrix.\n    k : int\n        Rank of ID.\n    idx : :class:`numpy.ndarray`\n        Column index array.\n\n    Returns\n    -------\n    :class:`numpy.ndarray`\n        Skeleton matrix.\n    ')
    
    
    # Call to _is_real(...): (line 701)
    # Processing the call arguments (line 701)
    # Getting the type of 'A' (line 701)
    A_21470 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 701, 16), 'A', False)
    # Processing the call keyword arguments (line 701)
    kwargs_21471 = {}
    # Getting the type of '_is_real' (line 701)
    _is_real_21469 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 701, 7), '_is_real', False)
    # Calling _is_real(args, kwargs) (line 701)
    _is_real_call_result_21472 = invoke(stypy.reporting.localization.Localization(__file__, 701, 7), _is_real_21469, *[A_21470], **kwargs_21471)
    
    # Testing the type of an if condition (line 701)
    if_condition_21473 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 701, 4), _is_real_call_result_21472)
    # Assigning a type to the variable 'if_condition_21473' (line 701)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 701, 4), 'if_condition_21473', if_condition_21473)
    # SSA begins for if statement (line 701)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to idd_copycols(...): (line 702)
    # Processing the call arguments (line 702)
    # Getting the type of 'A' (line 702)
    A_21476 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 702, 36), 'A', False)
    # Getting the type of 'k' (line 702)
    k_21477 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 702, 39), 'k', False)
    # Getting the type of 'idx' (line 702)
    idx_21478 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 702, 42), 'idx', False)
    int_21479 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 702, 48), 'int')
    # Applying the binary operator '+' (line 702)
    result_add_21480 = python_operator(stypy.reporting.localization.Localization(__file__, 702, 42), '+', idx_21478, int_21479)
    
    # Processing the call keyword arguments (line 702)
    kwargs_21481 = {}
    # Getting the type of 'backend' (line 702)
    backend_21474 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 702, 15), 'backend', False)
    # Obtaining the member 'idd_copycols' of a type (line 702)
    idd_copycols_21475 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 702, 15), backend_21474, 'idd_copycols')
    # Calling idd_copycols(args, kwargs) (line 702)
    idd_copycols_call_result_21482 = invoke(stypy.reporting.localization.Localization(__file__, 702, 15), idd_copycols_21475, *[A_21476, k_21477, result_add_21480], **kwargs_21481)
    
    # Assigning a type to the variable 'stypy_return_type' (line 702)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 702, 8), 'stypy_return_type', idd_copycols_call_result_21482)
    # SSA branch for the else part of an if statement (line 701)
    module_type_store.open_ssa_branch('else')
    
    # Call to idz_copycols(...): (line 704)
    # Processing the call arguments (line 704)
    # Getting the type of 'A' (line 704)
    A_21485 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 704, 36), 'A', False)
    # Getting the type of 'k' (line 704)
    k_21486 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 704, 39), 'k', False)
    # Getting the type of 'idx' (line 704)
    idx_21487 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 704, 42), 'idx', False)
    int_21488 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 704, 48), 'int')
    # Applying the binary operator '+' (line 704)
    result_add_21489 = python_operator(stypy.reporting.localization.Localization(__file__, 704, 42), '+', idx_21487, int_21488)
    
    # Processing the call keyword arguments (line 704)
    kwargs_21490 = {}
    # Getting the type of 'backend' (line 704)
    backend_21483 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 704, 15), 'backend', False)
    # Obtaining the member 'idz_copycols' of a type (line 704)
    idz_copycols_21484 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 704, 15), backend_21483, 'idz_copycols')
    # Calling idz_copycols(args, kwargs) (line 704)
    idz_copycols_call_result_21491 = invoke(stypy.reporting.localization.Localization(__file__, 704, 15), idz_copycols_21484, *[A_21485, k_21486, result_add_21489], **kwargs_21490)
    
    # Assigning a type to the variable 'stypy_return_type' (line 704)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 704, 8), 'stypy_return_type', idz_copycols_call_result_21491)
    # SSA join for if statement (line 701)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'reconstruct_skel_matrix(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'reconstruct_skel_matrix' in the type store
    # Getting the type of 'stypy_return_type' (line 667)
    stypy_return_type_21492 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 667, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_21492)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'reconstruct_skel_matrix'
    return stypy_return_type_21492

# Assigning a type to the variable 'reconstruct_skel_matrix' (line 667)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 667, 0), 'reconstruct_skel_matrix', reconstruct_skel_matrix)

@norecursion
def id_to_svd(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'id_to_svd'
    module_type_store = module_type_store.open_function_context('id_to_svd', 707, 0, False)
    
    # Passed parameters checking function
    id_to_svd.stypy_localization = localization
    id_to_svd.stypy_type_of_self = None
    id_to_svd.stypy_type_store = module_type_store
    id_to_svd.stypy_function_name = 'id_to_svd'
    id_to_svd.stypy_param_names_list = ['B', 'idx', 'proj']
    id_to_svd.stypy_varargs_param_name = None
    id_to_svd.stypy_kwargs_param_name = None
    id_to_svd.stypy_call_defaults = defaults
    id_to_svd.stypy_call_varargs = varargs
    id_to_svd.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'id_to_svd', ['B', 'idx', 'proj'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'id_to_svd', localization, ['B', 'idx', 'proj'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'id_to_svd(...)' code ##################

    str_21493 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 740, (-1)), 'str', '\n    Convert ID to SVD.\n\n    The SVD reconstruction of a matrix with skeleton matrix `B` and ID indices and\n    coefficients `idx` and `proj`, respectively, is::\n\n        U, S, V = id_to_svd(B, idx, proj)\n        A = numpy.dot(U, numpy.dot(numpy.diag(S), V.conj().T))\n\n    See also :func:`svd`.\n\n    ..  This function automatically detects the matrix data type and calls the\n        appropriate backend. For details, see :func:`backend.idd_id2svd` and\n        :func:`backend.idz_id2svd`.\n\n    Parameters\n    ----------\n    B : :class:`numpy.ndarray`\n        Skeleton matrix.\n    idx : :class:`numpy.ndarray`\n        Column index array.\n    proj : :class:`numpy.ndarray`\n        Interpolation coefficients.\n\n    Returns\n    -------\n    U : :class:`numpy.ndarray`\n        Left singular vectors.\n    S : :class:`numpy.ndarray`\n        Singular values.\n    V : :class:`numpy.ndarray`\n        Right singular vectors.\n    ')
    
    
    # Call to _is_real(...): (line 741)
    # Processing the call arguments (line 741)
    # Getting the type of 'B' (line 741)
    B_21495 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 741, 16), 'B', False)
    # Processing the call keyword arguments (line 741)
    kwargs_21496 = {}
    # Getting the type of '_is_real' (line 741)
    _is_real_21494 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 741, 7), '_is_real', False)
    # Calling _is_real(args, kwargs) (line 741)
    _is_real_call_result_21497 = invoke(stypy.reporting.localization.Localization(__file__, 741, 7), _is_real_21494, *[B_21495], **kwargs_21496)
    
    # Testing the type of an if condition (line 741)
    if_condition_21498 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 741, 4), _is_real_call_result_21497)
    # Assigning a type to the variable 'if_condition_21498' (line 741)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 741, 4), 'if_condition_21498', if_condition_21498)
    # SSA begins for if statement (line 741)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Tuple (line 742):
    
    # Assigning a Subscript to a Name (line 742):
    
    # Obtaining the type of the subscript
    int_21499 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 742, 8), 'int')
    
    # Call to idd_id2svd(...): (line 742)
    # Processing the call arguments (line 742)
    # Getting the type of 'B' (line 742)
    B_21502 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 742, 37), 'B', False)
    # Getting the type of 'idx' (line 742)
    idx_21503 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 742, 40), 'idx', False)
    int_21504 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 742, 46), 'int')
    # Applying the binary operator '+' (line 742)
    result_add_21505 = python_operator(stypy.reporting.localization.Localization(__file__, 742, 40), '+', idx_21503, int_21504)
    
    # Getting the type of 'proj' (line 742)
    proj_21506 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 742, 49), 'proj', False)
    # Processing the call keyword arguments (line 742)
    kwargs_21507 = {}
    # Getting the type of 'backend' (line 742)
    backend_21500 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 742, 18), 'backend', False)
    # Obtaining the member 'idd_id2svd' of a type (line 742)
    idd_id2svd_21501 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 742, 18), backend_21500, 'idd_id2svd')
    # Calling idd_id2svd(args, kwargs) (line 742)
    idd_id2svd_call_result_21508 = invoke(stypy.reporting.localization.Localization(__file__, 742, 18), idd_id2svd_21501, *[B_21502, result_add_21505, proj_21506], **kwargs_21507)
    
    # Obtaining the member '__getitem__' of a type (line 742)
    getitem___21509 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 742, 8), idd_id2svd_call_result_21508, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 742)
    subscript_call_result_21510 = invoke(stypy.reporting.localization.Localization(__file__, 742, 8), getitem___21509, int_21499)
    
    # Assigning a type to the variable 'tuple_var_assignment_20822' (line 742)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 742, 8), 'tuple_var_assignment_20822', subscript_call_result_21510)
    
    # Assigning a Subscript to a Name (line 742):
    
    # Obtaining the type of the subscript
    int_21511 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 742, 8), 'int')
    
    # Call to idd_id2svd(...): (line 742)
    # Processing the call arguments (line 742)
    # Getting the type of 'B' (line 742)
    B_21514 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 742, 37), 'B', False)
    # Getting the type of 'idx' (line 742)
    idx_21515 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 742, 40), 'idx', False)
    int_21516 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 742, 46), 'int')
    # Applying the binary operator '+' (line 742)
    result_add_21517 = python_operator(stypy.reporting.localization.Localization(__file__, 742, 40), '+', idx_21515, int_21516)
    
    # Getting the type of 'proj' (line 742)
    proj_21518 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 742, 49), 'proj', False)
    # Processing the call keyword arguments (line 742)
    kwargs_21519 = {}
    # Getting the type of 'backend' (line 742)
    backend_21512 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 742, 18), 'backend', False)
    # Obtaining the member 'idd_id2svd' of a type (line 742)
    idd_id2svd_21513 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 742, 18), backend_21512, 'idd_id2svd')
    # Calling idd_id2svd(args, kwargs) (line 742)
    idd_id2svd_call_result_21520 = invoke(stypy.reporting.localization.Localization(__file__, 742, 18), idd_id2svd_21513, *[B_21514, result_add_21517, proj_21518], **kwargs_21519)
    
    # Obtaining the member '__getitem__' of a type (line 742)
    getitem___21521 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 742, 8), idd_id2svd_call_result_21520, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 742)
    subscript_call_result_21522 = invoke(stypy.reporting.localization.Localization(__file__, 742, 8), getitem___21521, int_21511)
    
    # Assigning a type to the variable 'tuple_var_assignment_20823' (line 742)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 742, 8), 'tuple_var_assignment_20823', subscript_call_result_21522)
    
    # Assigning a Subscript to a Name (line 742):
    
    # Obtaining the type of the subscript
    int_21523 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 742, 8), 'int')
    
    # Call to idd_id2svd(...): (line 742)
    # Processing the call arguments (line 742)
    # Getting the type of 'B' (line 742)
    B_21526 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 742, 37), 'B', False)
    # Getting the type of 'idx' (line 742)
    idx_21527 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 742, 40), 'idx', False)
    int_21528 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 742, 46), 'int')
    # Applying the binary operator '+' (line 742)
    result_add_21529 = python_operator(stypy.reporting.localization.Localization(__file__, 742, 40), '+', idx_21527, int_21528)
    
    # Getting the type of 'proj' (line 742)
    proj_21530 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 742, 49), 'proj', False)
    # Processing the call keyword arguments (line 742)
    kwargs_21531 = {}
    # Getting the type of 'backend' (line 742)
    backend_21524 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 742, 18), 'backend', False)
    # Obtaining the member 'idd_id2svd' of a type (line 742)
    idd_id2svd_21525 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 742, 18), backend_21524, 'idd_id2svd')
    # Calling idd_id2svd(args, kwargs) (line 742)
    idd_id2svd_call_result_21532 = invoke(stypy.reporting.localization.Localization(__file__, 742, 18), idd_id2svd_21525, *[B_21526, result_add_21529, proj_21530], **kwargs_21531)
    
    # Obtaining the member '__getitem__' of a type (line 742)
    getitem___21533 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 742, 8), idd_id2svd_call_result_21532, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 742)
    subscript_call_result_21534 = invoke(stypy.reporting.localization.Localization(__file__, 742, 8), getitem___21533, int_21523)
    
    # Assigning a type to the variable 'tuple_var_assignment_20824' (line 742)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 742, 8), 'tuple_var_assignment_20824', subscript_call_result_21534)
    
    # Assigning a Name to a Name (line 742):
    # Getting the type of 'tuple_var_assignment_20822' (line 742)
    tuple_var_assignment_20822_21535 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 742, 8), 'tuple_var_assignment_20822')
    # Assigning a type to the variable 'U' (line 742)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 742, 8), 'U', tuple_var_assignment_20822_21535)
    
    # Assigning a Name to a Name (line 742):
    # Getting the type of 'tuple_var_assignment_20823' (line 742)
    tuple_var_assignment_20823_21536 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 742, 8), 'tuple_var_assignment_20823')
    # Assigning a type to the variable 'V' (line 742)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 742, 11), 'V', tuple_var_assignment_20823_21536)
    
    # Assigning a Name to a Name (line 742):
    # Getting the type of 'tuple_var_assignment_20824' (line 742)
    tuple_var_assignment_20824_21537 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 742, 8), 'tuple_var_assignment_20824')
    # Assigning a type to the variable 'S' (line 742)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 742, 14), 'S', tuple_var_assignment_20824_21537)
    # SSA branch for the else part of an if statement (line 741)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Call to a Tuple (line 744):
    
    # Assigning a Subscript to a Name (line 744):
    
    # Obtaining the type of the subscript
    int_21538 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 744, 8), 'int')
    
    # Call to idz_id2svd(...): (line 744)
    # Processing the call arguments (line 744)
    # Getting the type of 'B' (line 744)
    B_21541 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 744, 37), 'B', False)
    # Getting the type of 'idx' (line 744)
    idx_21542 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 744, 40), 'idx', False)
    int_21543 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 744, 46), 'int')
    # Applying the binary operator '+' (line 744)
    result_add_21544 = python_operator(stypy.reporting.localization.Localization(__file__, 744, 40), '+', idx_21542, int_21543)
    
    # Getting the type of 'proj' (line 744)
    proj_21545 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 744, 49), 'proj', False)
    # Processing the call keyword arguments (line 744)
    kwargs_21546 = {}
    # Getting the type of 'backend' (line 744)
    backend_21539 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 744, 18), 'backend', False)
    # Obtaining the member 'idz_id2svd' of a type (line 744)
    idz_id2svd_21540 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 744, 18), backend_21539, 'idz_id2svd')
    # Calling idz_id2svd(args, kwargs) (line 744)
    idz_id2svd_call_result_21547 = invoke(stypy.reporting.localization.Localization(__file__, 744, 18), idz_id2svd_21540, *[B_21541, result_add_21544, proj_21545], **kwargs_21546)
    
    # Obtaining the member '__getitem__' of a type (line 744)
    getitem___21548 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 744, 8), idz_id2svd_call_result_21547, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 744)
    subscript_call_result_21549 = invoke(stypy.reporting.localization.Localization(__file__, 744, 8), getitem___21548, int_21538)
    
    # Assigning a type to the variable 'tuple_var_assignment_20825' (line 744)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 744, 8), 'tuple_var_assignment_20825', subscript_call_result_21549)
    
    # Assigning a Subscript to a Name (line 744):
    
    # Obtaining the type of the subscript
    int_21550 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 744, 8), 'int')
    
    # Call to idz_id2svd(...): (line 744)
    # Processing the call arguments (line 744)
    # Getting the type of 'B' (line 744)
    B_21553 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 744, 37), 'B', False)
    # Getting the type of 'idx' (line 744)
    idx_21554 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 744, 40), 'idx', False)
    int_21555 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 744, 46), 'int')
    # Applying the binary operator '+' (line 744)
    result_add_21556 = python_operator(stypy.reporting.localization.Localization(__file__, 744, 40), '+', idx_21554, int_21555)
    
    # Getting the type of 'proj' (line 744)
    proj_21557 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 744, 49), 'proj', False)
    # Processing the call keyword arguments (line 744)
    kwargs_21558 = {}
    # Getting the type of 'backend' (line 744)
    backend_21551 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 744, 18), 'backend', False)
    # Obtaining the member 'idz_id2svd' of a type (line 744)
    idz_id2svd_21552 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 744, 18), backend_21551, 'idz_id2svd')
    # Calling idz_id2svd(args, kwargs) (line 744)
    idz_id2svd_call_result_21559 = invoke(stypy.reporting.localization.Localization(__file__, 744, 18), idz_id2svd_21552, *[B_21553, result_add_21556, proj_21557], **kwargs_21558)
    
    # Obtaining the member '__getitem__' of a type (line 744)
    getitem___21560 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 744, 8), idz_id2svd_call_result_21559, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 744)
    subscript_call_result_21561 = invoke(stypy.reporting.localization.Localization(__file__, 744, 8), getitem___21560, int_21550)
    
    # Assigning a type to the variable 'tuple_var_assignment_20826' (line 744)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 744, 8), 'tuple_var_assignment_20826', subscript_call_result_21561)
    
    # Assigning a Subscript to a Name (line 744):
    
    # Obtaining the type of the subscript
    int_21562 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 744, 8), 'int')
    
    # Call to idz_id2svd(...): (line 744)
    # Processing the call arguments (line 744)
    # Getting the type of 'B' (line 744)
    B_21565 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 744, 37), 'B', False)
    # Getting the type of 'idx' (line 744)
    idx_21566 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 744, 40), 'idx', False)
    int_21567 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 744, 46), 'int')
    # Applying the binary operator '+' (line 744)
    result_add_21568 = python_operator(stypy.reporting.localization.Localization(__file__, 744, 40), '+', idx_21566, int_21567)
    
    # Getting the type of 'proj' (line 744)
    proj_21569 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 744, 49), 'proj', False)
    # Processing the call keyword arguments (line 744)
    kwargs_21570 = {}
    # Getting the type of 'backend' (line 744)
    backend_21563 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 744, 18), 'backend', False)
    # Obtaining the member 'idz_id2svd' of a type (line 744)
    idz_id2svd_21564 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 744, 18), backend_21563, 'idz_id2svd')
    # Calling idz_id2svd(args, kwargs) (line 744)
    idz_id2svd_call_result_21571 = invoke(stypy.reporting.localization.Localization(__file__, 744, 18), idz_id2svd_21564, *[B_21565, result_add_21568, proj_21569], **kwargs_21570)
    
    # Obtaining the member '__getitem__' of a type (line 744)
    getitem___21572 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 744, 8), idz_id2svd_call_result_21571, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 744)
    subscript_call_result_21573 = invoke(stypy.reporting.localization.Localization(__file__, 744, 8), getitem___21572, int_21562)
    
    # Assigning a type to the variable 'tuple_var_assignment_20827' (line 744)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 744, 8), 'tuple_var_assignment_20827', subscript_call_result_21573)
    
    # Assigning a Name to a Name (line 744):
    # Getting the type of 'tuple_var_assignment_20825' (line 744)
    tuple_var_assignment_20825_21574 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 744, 8), 'tuple_var_assignment_20825')
    # Assigning a type to the variable 'U' (line 744)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 744, 8), 'U', tuple_var_assignment_20825_21574)
    
    # Assigning a Name to a Name (line 744):
    # Getting the type of 'tuple_var_assignment_20826' (line 744)
    tuple_var_assignment_20826_21575 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 744, 8), 'tuple_var_assignment_20826')
    # Assigning a type to the variable 'V' (line 744)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 744, 11), 'V', tuple_var_assignment_20826_21575)
    
    # Assigning a Name to a Name (line 744):
    # Getting the type of 'tuple_var_assignment_20827' (line 744)
    tuple_var_assignment_20827_21576 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 744, 8), 'tuple_var_assignment_20827')
    # Assigning a type to the variable 'S' (line 744)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 744, 14), 'S', tuple_var_assignment_20827_21576)
    # SSA join for if statement (line 741)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Obtaining an instance of the builtin type 'tuple' (line 745)
    tuple_21577 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 745, 11), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 745)
    # Adding element type (line 745)
    # Getting the type of 'U' (line 745)
    U_21578 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 745, 11), 'U')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 745, 11), tuple_21577, U_21578)
    # Adding element type (line 745)
    # Getting the type of 'S' (line 745)
    S_21579 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 745, 14), 'S')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 745, 11), tuple_21577, S_21579)
    # Adding element type (line 745)
    # Getting the type of 'V' (line 745)
    V_21580 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 745, 17), 'V')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 745, 11), tuple_21577, V_21580)
    
    # Assigning a type to the variable 'stypy_return_type' (line 745)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 745, 4), 'stypy_return_type', tuple_21577)
    
    # ################# End of 'id_to_svd(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'id_to_svd' in the type store
    # Getting the type of 'stypy_return_type' (line 707)
    stypy_return_type_21581 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 707, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_21581)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'id_to_svd'
    return stypy_return_type_21581

# Assigning a type to the variable 'id_to_svd' (line 707)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 707, 0), 'id_to_svd', id_to_svd)

@norecursion
def estimate_spectral_norm(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    int_21582 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 748, 34), 'int')
    defaults = [int_21582]
    # Create a new context for function 'estimate_spectral_norm'
    module_type_store = module_type_store.open_function_context('estimate_spectral_norm', 748, 0, False)
    
    # Passed parameters checking function
    estimate_spectral_norm.stypy_localization = localization
    estimate_spectral_norm.stypy_type_of_self = None
    estimate_spectral_norm.stypy_type_store = module_type_store
    estimate_spectral_norm.stypy_function_name = 'estimate_spectral_norm'
    estimate_spectral_norm.stypy_param_names_list = ['A', 'its']
    estimate_spectral_norm.stypy_varargs_param_name = None
    estimate_spectral_norm.stypy_kwargs_param_name = None
    estimate_spectral_norm.stypy_call_defaults = defaults
    estimate_spectral_norm.stypy_call_varargs = varargs
    estimate_spectral_norm.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'estimate_spectral_norm', ['A', 'its'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'estimate_spectral_norm', localization, ['A', 'its'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'estimate_spectral_norm(...)' code ##################

    str_21583 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 768, (-1)), 'str', '\n    Estimate spectral norm of a matrix by the randomized power method.\n\n    ..  This function automatically detects the matrix data type and calls the\n        appropriate backend. For details, see :func:`backend.idd_snorm` and\n        :func:`backend.idz_snorm`.\n\n    Parameters\n    ----------\n    A : :class:`scipy.sparse.linalg.LinearOperator`\n        Matrix given as a :class:`scipy.sparse.linalg.LinearOperator` with the\n        `matvec` and `rmatvec` methods (to apply the matrix and its adjoint).\n    its : int, optional\n        Number of power method iterations.\n\n    Returns\n    -------\n    float\n        Spectral norm estimate.\n    ')
    stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 769, 4))
    
    # 'from scipy.sparse.linalg import aslinearoperator' statement (line 769)
    update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/linalg/')
    import_21584 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 769, 4), 'scipy.sparse.linalg')

    if (type(import_21584) is not StypyTypeError):

        if (import_21584 != 'pyd_module'):
            __import__(import_21584)
            sys_modules_21585 = sys.modules[import_21584]
            import_from_module(stypy.reporting.localization.Localization(__file__, 769, 4), 'scipy.sparse.linalg', sys_modules_21585.module_type_store, module_type_store, ['aslinearoperator'])
            nest_module(stypy.reporting.localization.Localization(__file__, 769, 4), __file__, sys_modules_21585, sys_modules_21585.module_type_store, module_type_store)
        else:
            from scipy.sparse.linalg import aslinearoperator

            import_from_module(stypy.reporting.localization.Localization(__file__, 769, 4), 'scipy.sparse.linalg', None, module_type_store, ['aslinearoperator'], [aslinearoperator])

    else:
        # Assigning a type to the variable 'scipy.sparse.linalg' (line 769)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 769, 4), 'scipy.sparse.linalg', import_21584)

    remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/linalg/')
    
    
    # Assigning a Call to a Name (line 770):
    
    # Assigning a Call to a Name (line 770):
    
    # Call to aslinearoperator(...): (line 770)
    # Processing the call arguments (line 770)
    # Getting the type of 'A' (line 770)
    A_21587 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 770, 25), 'A', False)
    # Processing the call keyword arguments (line 770)
    kwargs_21588 = {}
    # Getting the type of 'aslinearoperator' (line 770)
    aslinearoperator_21586 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 770, 8), 'aslinearoperator', False)
    # Calling aslinearoperator(args, kwargs) (line 770)
    aslinearoperator_call_result_21589 = invoke(stypy.reporting.localization.Localization(__file__, 770, 8), aslinearoperator_21586, *[A_21587], **kwargs_21588)
    
    # Assigning a type to the variable 'A' (line 770)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 770, 4), 'A', aslinearoperator_call_result_21589)
    
    # Assigning a Attribute to a Tuple (line 771):
    
    # Assigning a Subscript to a Name (line 771):
    
    # Obtaining the type of the subscript
    int_21590 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 771, 4), 'int')
    # Getting the type of 'A' (line 771)
    A_21591 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 771, 11), 'A')
    # Obtaining the member 'shape' of a type (line 771)
    shape_21592 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 771, 11), A_21591, 'shape')
    # Obtaining the member '__getitem__' of a type (line 771)
    getitem___21593 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 771, 4), shape_21592, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 771)
    subscript_call_result_21594 = invoke(stypy.reporting.localization.Localization(__file__, 771, 4), getitem___21593, int_21590)
    
    # Assigning a type to the variable 'tuple_var_assignment_20828' (line 771)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 771, 4), 'tuple_var_assignment_20828', subscript_call_result_21594)
    
    # Assigning a Subscript to a Name (line 771):
    
    # Obtaining the type of the subscript
    int_21595 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 771, 4), 'int')
    # Getting the type of 'A' (line 771)
    A_21596 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 771, 11), 'A')
    # Obtaining the member 'shape' of a type (line 771)
    shape_21597 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 771, 11), A_21596, 'shape')
    # Obtaining the member '__getitem__' of a type (line 771)
    getitem___21598 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 771, 4), shape_21597, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 771)
    subscript_call_result_21599 = invoke(stypy.reporting.localization.Localization(__file__, 771, 4), getitem___21598, int_21595)
    
    # Assigning a type to the variable 'tuple_var_assignment_20829' (line 771)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 771, 4), 'tuple_var_assignment_20829', subscript_call_result_21599)
    
    # Assigning a Name to a Name (line 771):
    # Getting the type of 'tuple_var_assignment_20828' (line 771)
    tuple_var_assignment_20828_21600 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 771, 4), 'tuple_var_assignment_20828')
    # Assigning a type to the variable 'm' (line 771)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 771, 4), 'm', tuple_var_assignment_20828_21600)
    
    # Assigning a Name to a Name (line 771):
    # Getting the type of 'tuple_var_assignment_20829' (line 771)
    tuple_var_assignment_20829_21601 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 771, 4), 'tuple_var_assignment_20829')
    # Assigning a type to the variable 'n' (line 771)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 771, 7), 'n', tuple_var_assignment_20829_21601)
    
    # Assigning a Lambda to a Name (line 772):
    
    # Assigning a Lambda to a Name (line 772):

    @norecursion
    def _stypy_temp_lambda_8(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_stypy_temp_lambda_8'
        module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_8', 772, 13, True)
        # Passed parameters checking function
        _stypy_temp_lambda_8.stypy_localization = localization
        _stypy_temp_lambda_8.stypy_type_of_self = None
        _stypy_temp_lambda_8.stypy_type_store = module_type_store
        _stypy_temp_lambda_8.stypy_function_name = '_stypy_temp_lambda_8'
        _stypy_temp_lambda_8.stypy_param_names_list = ['x']
        _stypy_temp_lambda_8.stypy_varargs_param_name = None
        _stypy_temp_lambda_8.stypy_kwargs_param_name = None
        _stypy_temp_lambda_8.stypy_call_defaults = defaults
        _stypy_temp_lambda_8.stypy_call_varargs = varargs
        _stypy_temp_lambda_8.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_8', ['x'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Stacktrace push for error reporting
        localization.set_stack_trace('_stypy_temp_lambda_8', ['x'], arguments)
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of the lambda function code ##################

        
        # Call to matvec(...): (line 772)
        # Processing the call arguments (line 772)
        # Getting the type of 'x' (line 772)
        x_21604 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 772, 33), 'x', False)
        # Processing the call keyword arguments (line 772)
        kwargs_21605 = {}
        # Getting the type of 'A' (line 772)
        A_21602 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 772, 23), 'A', False)
        # Obtaining the member 'matvec' of a type (line 772)
        matvec_21603 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 772, 23), A_21602, 'matvec')
        # Calling matvec(args, kwargs) (line 772)
        matvec_call_result_21606 = invoke(stypy.reporting.localization.Localization(__file__, 772, 23), matvec_21603, *[x_21604], **kwargs_21605)
        
        # Assigning the return type of the lambda function
        # Assigning a type to the variable 'stypy_return_type' (line 772)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 772, 13), 'stypy_return_type', matvec_call_result_21606)
        
        # ################# End of the lambda function code ##################

        # Stacktrace pop (error reporting)
        localization.unset_stack_trace()
        
        # Storing the return type of function '_stypy_temp_lambda_8' in the type store
        # Getting the type of 'stypy_return_type' (line 772)
        stypy_return_type_21607 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 772, 13), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_21607)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_stypy_temp_lambda_8'
        return stypy_return_type_21607

    # Assigning a type to the variable '_stypy_temp_lambda_8' (line 772)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 772, 13), '_stypy_temp_lambda_8', _stypy_temp_lambda_8)
    # Getting the type of '_stypy_temp_lambda_8' (line 772)
    _stypy_temp_lambda_8_21608 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 772, 13), '_stypy_temp_lambda_8')
    # Assigning a type to the variable 'matvec' (line 772)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 772, 4), 'matvec', _stypy_temp_lambda_8_21608)
    
    # Assigning a Lambda to a Name (line 773):
    
    # Assigning a Lambda to a Name (line 773):

    @norecursion
    def _stypy_temp_lambda_9(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_stypy_temp_lambda_9'
        module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_9', 773, 14, True)
        # Passed parameters checking function
        _stypy_temp_lambda_9.stypy_localization = localization
        _stypy_temp_lambda_9.stypy_type_of_self = None
        _stypy_temp_lambda_9.stypy_type_store = module_type_store
        _stypy_temp_lambda_9.stypy_function_name = '_stypy_temp_lambda_9'
        _stypy_temp_lambda_9.stypy_param_names_list = ['x']
        _stypy_temp_lambda_9.stypy_varargs_param_name = None
        _stypy_temp_lambda_9.stypy_kwargs_param_name = None
        _stypy_temp_lambda_9.stypy_call_defaults = defaults
        _stypy_temp_lambda_9.stypy_call_varargs = varargs
        _stypy_temp_lambda_9.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_9', ['x'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Stacktrace push for error reporting
        localization.set_stack_trace('_stypy_temp_lambda_9', ['x'], arguments)
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of the lambda function code ##################

        
        # Call to rmatvec(...): (line 773)
        # Processing the call arguments (line 773)
        # Getting the type of 'x' (line 773)
        x_21611 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 773, 34), 'x', False)
        # Processing the call keyword arguments (line 773)
        kwargs_21612 = {}
        # Getting the type of 'A' (line 773)
        A_21609 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 773, 24), 'A', False)
        # Obtaining the member 'rmatvec' of a type (line 773)
        rmatvec_21610 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 773, 24), A_21609, 'rmatvec')
        # Calling rmatvec(args, kwargs) (line 773)
        rmatvec_call_result_21613 = invoke(stypy.reporting.localization.Localization(__file__, 773, 24), rmatvec_21610, *[x_21611], **kwargs_21612)
        
        # Assigning the return type of the lambda function
        # Assigning a type to the variable 'stypy_return_type' (line 773)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 773, 14), 'stypy_return_type', rmatvec_call_result_21613)
        
        # ################# End of the lambda function code ##################

        # Stacktrace pop (error reporting)
        localization.unset_stack_trace()
        
        # Storing the return type of function '_stypy_temp_lambda_9' in the type store
        # Getting the type of 'stypy_return_type' (line 773)
        stypy_return_type_21614 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 773, 14), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_21614)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_stypy_temp_lambda_9'
        return stypy_return_type_21614

    # Assigning a type to the variable '_stypy_temp_lambda_9' (line 773)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 773, 14), '_stypy_temp_lambda_9', _stypy_temp_lambda_9)
    # Getting the type of '_stypy_temp_lambda_9' (line 773)
    _stypy_temp_lambda_9_21615 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 773, 14), '_stypy_temp_lambda_9')
    # Assigning a type to the variable 'matveca' (line 773)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 773, 4), 'matveca', _stypy_temp_lambda_9_21615)
    
    
    # Call to _is_real(...): (line 774)
    # Processing the call arguments (line 774)
    # Getting the type of 'A' (line 774)
    A_21617 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 774, 16), 'A', False)
    # Processing the call keyword arguments (line 774)
    kwargs_21618 = {}
    # Getting the type of '_is_real' (line 774)
    _is_real_21616 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 774, 7), '_is_real', False)
    # Calling _is_real(args, kwargs) (line 774)
    _is_real_call_result_21619 = invoke(stypy.reporting.localization.Localization(__file__, 774, 7), _is_real_21616, *[A_21617], **kwargs_21618)
    
    # Testing the type of an if condition (line 774)
    if_condition_21620 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 774, 4), _is_real_call_result_21619)
    # Assigning a type to the variable 'if_condition_21620' (line 774)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 774, 4), 'if_condition_21620', if_condition_21620)
    # SSA begins for if statement (line 774)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to idd_snorm(...): (line 775)
    # Processing the call arguments (line 775)
    # Getting the type of 'm' (line 775)
    m_21623 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 775, 33), 'm', False)
    # Getting the type of 'n' (line 775)
    n_21624 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 775, 36), 'n', False)
    # Getting the type of 'matveca' (line 775)
    matveca_21625 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 775, 39), 'matveca', False)
    # Getting the type of 'matvec' (line 775)
    matvec_21626 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 775, 48), 'matvec', False)
    # Processing the call keyword arguments (line 775)
    # Getting the type of 'its' (line 775)
    its_21627 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 775, 60), 'its', False)
    keyword_21628 = its_21627
    kwargs_21629 = {'its': keyword_21628}
    # Getting the type of 'backend' (line 775)
    backend_21621 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 775, 15), 'backend', False)
    # Obtaining the member 'idd_snorm' of a type (line 775)
    idd_snorm_21622 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 775, 15), backend_21621, 'idd_snorm')
    # Calling idd_snorm(args, kwargs) (line 775)
    idd_snorm_call_result_21630 = invoke(stypy.reporting.localization.Localization(__file__, 775, 15), idd_snorm_21622, *[m_21623, n_21624, matveca_21625, matvec_21626], **kwargs_21629)
    
    # Assigning a type to the variable 'stypy_return_type' (line 775)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 775, 8), 'stypy_return_type', idd_snorm_call_result_21630)
    # SSA branch for the else part of an if statement (line 774)
    module_type_store.open_ssa_branch('else')
    
    # Call to idz_snorm(...): (line 777)
    # Processing the call arguments (line 777)
    # Getting the type of 'm' (line 777)
    m_21633 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 777, 33), 'm', False)
    # Getting the type of 'n' (line 777)
    n_21634 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 777, 36), 'n', False)
    # Getting the type of 'matveca' (line 777)
    matveca_21635 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 777, 39), 'matveca', False)
    # Getting the type of 'matvec' (line 777)
    matvec_21636 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 777, 48), 'matvec', False)
    # Processing the call keyword arguments (line 777)
    # Getting the type of 'its' (line 777)
    its_21637 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 777, 60), 'its', False)
    keyword_21638 = its_21637
    kwargs_21639 = {'its': keyword_21638}
    # Getting the type of 'backend' (line 777)
    backend_21631 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 777, 15), 'backend', False)
    # Obtaining the member 'idz_snorm' of a type (line 777)
    idz_snorm_21632 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 777, 15), backend_21631, 'idz_snorm')
    # Calling idz_snorm(args, kwargs) (line 777)
    idz_snorm_call_result_21640 = invoke(stypy.reporting.localization.Localization(__file__, 777, 15), idz_snorm_21632, *[m_21633, n_21634, matveca_21635, matvec_21636], **kwargs_21639)
    
    # Assigning a type to the variable 'stypy_return_type' (line 777)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 777, 8), 'stypy_return_type', idz_snorm_call_result_21640)
    # SSA join for if statement (line 774)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'estimate_spectral_norm(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'estimate_spectral_norm' in the type store
    # Getting the type of 'stypy_return_type' (line 748)
    stypy_return_type_21641 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 748, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_21641)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'estimate_spectral_norm'
    return stypy_return_type_21641

# Assigning a type to the variable 'estimate_spectral_norm' (line 748)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 748, 0), 'estimate_spectral_norm', estimate_spectral_norm)

@norecursion
def estimate_spectral_norm_diff(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    int_21642 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 780, 42), 'int')
    defaults = [int_21642]
    # Create a new context for function 'estimate_spectral_norm_diff'
    module_type_store = module_type_store.open_function_context('estimate_spectral_norm_diff', 780, 0, False)
    
    # Passed parameters checking function
    estimate_spectral_norm_diff.stypy_localization = localization
    estimate_spectral_norm_diff.stypy_type_of_self = None
    estimate_spectral_norm_diff.stypy_type_store = module_type_store
    estimate_spectral_norm_diff.stypy_function_name = 'estimate_spectral_norm_diff'
    estimate_spectral_norm_diff.stypy_param_names_list = ['A', 'B', 'its']
    estimate_spectral_norm_diff.stypy_varargs_param_name = None
    estimate_spectral_norm_diff.stypy_kwargs_param_name = None
    estimate_spectral_norm_diff.stypy_call_defaults = defaults
    estimate_spectral_norm_diff.stypy_call_varargs = varargs
    estimate_spectral_norm_diff.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'estimate_spectral_norm_diff', ['A', 'B', 'its'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'estimate_spectral_norm_diff', localization, ['A', 'B', 'its'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'estimate_spectral_norm_diff(...)' code ##################

    str_21643 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 804, (-1)), 'str', '\n    Estimate spectral norm of the difference of two matrices by the randomized\n    power method.\n\n    ..  This function automatically detects the matrix data type and calls the\n        appropriate backend. For details, see :func:`backend.idd_diffsnorm` and\n        :func:`backend.idz_diffsnorm`.\n\n    Parameters\n    ----------\n    A : :class:`scipy.sparse.linalg.LinearOperator`\n        First matrix given as a :class:`scipy.sparse.linalg.LinearOperator` with the\n        `matvec` and `rmatvec` methods (to apply the matrix and its adjoint).\n    B : :class:`scipy.sparse.linalg.LinearOperator`\n        Second matrix given as a :class:`scipy.sparse.linalg.LinearOperator` with\n        the `matvec` and `rmatvec` methods (to apply the matrix and its adjoint).\n    its : int, optional\n        Number of power method iterations.\n\n    Returns\n    -------\n    float\n        Spectral norm estimate of matrix difference.\n    ')
    stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 805, 4))
    
    # 'from scipy.sparse.linalg import aslinearoperator' statement (line 805)
    update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/linalg/')
    import_21644 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 805, 4), 'scipy.sparse.linalg')

    if (type(import_21644) is not StypyTypeError):

        if (import_21644 != 'pyd_module'):
            __import__(import_21644)
            sys_modules_21645 = sys.modules[import_21644]
            import_from_module(stypy.reporting.localization.Localization(__file__, 805, 4), 'scipy.sparse.linalg', sys_modules_21645.module_type_store, module_type_store, ['aslinearoperator'])
            nest_module(stypy.reporting.localization.Localization(__file__, 805, 4), __file__, sys_modules_21645, sys_modules_21645.module_type_store, module_type_store)
        else:
            from scipy.sparse.linalg import aslinearoperator

            import_from_module(stypy.reporting.localization.Localization(__file__, 805, 4), 'scipy.sparse.linalg', None, module_type_store, ['aslinearoperator'], [aslinearoperator])

    else:
        # Assigning a type to the variable 'scipy.sparse.linalg' (line 805)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 805, 4), 'scipy.sparse.linalg', import_21644)

    remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/linalg/')
    
    
    # Assigning a Call to a Name (line 806):
    
    # Assigning a Call to a Name (line 806):
    
    # Call to aslinearoperator(...): (line 806)
    # Processing the call arguments (line 806)
    # Getting the type of 'A' (line 806)
    A_21647 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 806, 25), 'A', False)
    # Processing the call keyword arguments (line 806)
    kwargs_21648 = {}
    # Getting the type of 'aslinearoperator' (line 806)
    aslinearoperator_21646 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 806, 8), 'aslinearoperator', False)
    # Calling aslinearoperator(args, kwargs) (line 806)
    aslinearoperator_call_result_21649 = invoke(stypy.reporting.localization.Localization(__file__, 806, 8), aslinearoperator_21646, *[A_21647], **kwargs_21648)
    
    # Assigning a type to the variable 'A' (line 806)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 806, 4), 'A', aslinearoperator_call_result_21649)
    
    # Assigning a Call to a Name (line 807):
    
    # Assigning a Call to a Name (line 807):
    
    # Call to aslinearoperator(...): (line 807)
    # Processing the call arguments (line 807)
    # Getting the type of 'B' (line 807)
    B_21651 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 807, 25), 'B', False)
    # Processing the call keyword arguments (line 807)
    kwargs_21652 = {}
    # Getting the type of 'aslinearoperator' (line 807)
    aslinearoperator_21650 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 807, 8), 'aslinearoperator', False)
    # Calling aslinearoperator(args, kwargs) (line 807)
    aslinearoperator_call_result_21653 = invoke(stypy.reporting.localization.Localization(__file__, 807, 8), aslinearoperator_21650, *[B_21651], **kwargs_21652)
    
    # Assigning a type to the variable 'B' (line 807)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 807, 4), 'B', aslinearoperator_call_result_21653)
    
    # Assigning a Attribute to a Tuple (line 808):
    
    # Assigning a Subscript to a Name (line 808):
    
    # Obtaining the type of the subscript
    int_21654 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 808, 4), 'int')
    # Getting the type of 'A' (line 808)
    A_21655 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 808, 11), 'A')
    # Obtaining the member 'shape' of a type (line 808)
    shape_21656 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 808, 11), A_21655, 'shape')
    # Obtaining the member '__getitem__' of a type (line 808)
    getitem___21657 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 808, 4), shape_21656, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 808)
    subscript_call_result_21658 = invoke(stypy.reporting.localization.Localization(__file__, 808, 4), getitem___21657, int_21654)
    
    # Assigning a type to the variable 'tuple_var_assignment_20830' (line 808)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 808, 4), 'tuple_var_assignment_20830', subscript_call_result_21658)
    
    # Assigning a Subscript to a Name (line 808):
    
    # Obtaining the type of the subscript
    int_21659 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 808, 4), 'int')
    # Getting the type of 'A' (line 808)
    A_21660 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 808, 11), 'A')
    # Obtaining the member 'shape' of a type (line 808)
    shape_21661 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 808, 11), A_21660, 'shape')
    # Obtaining the member '__getitem__' of a type (line 808)
    getitem___21662 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 808, 4), shape_21661, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 808)
    subscript_call_result_21663 = invoke(stypy.reporting.localization.Localization(__file__, 808, 4), getitem___21662, int_21659)
    
    # Assigning a type to the variable 'tuple_var_assignment_20831' (line 808)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 808, 4), 'tuple_var_assignment_20831', subscript_call_result_21663)
    
    # Assigning a Name to a Name (line 808):
    # Getting the type of 'tuple_var_assignment_20830' (line 808)
    tuple_var_assignment_20830_21664 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 808, 4), 'tuple_var_assignment_20830')
    # Assigning a type to the variable 'm' (line 808)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 808, 4), 'm', tuple_var_assignment_20830_21664)
    
    # Assigning a Name to a Name (line 808):
    # Getting the type of 'tuple_var_assignment_20831' (line 808)
    tuple_var_assignment_20831_21665 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 808, 4), 'tuple_var_assignment_20831')
    # Assigning a type to the variable 'n' (line 808)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 808, 7), 'n', tuple_var_assignment_20831_21665)
    
    # Assigning a Lambda to a Name (line 809):
    
    # Assigning a Lambda to a Name (line 809):

    @norecursion
    def _stypy_temp_lambda_10(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_stypy_temp_lambda_10'
        module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_10', 809, 14, True)
        # Passed parameters checking function
        _stypy_temp_lambda_10.stypy_localization = localization
        _stypy_temp_lambda_10.stypy_type_of_self = None
        _stypy_temp_lambda_10.stypy_type_store = module_type_store
        _stypy_temp_lambda_10.stypy_function_name = '_stypy_temp_lambda_10'
        _stypy_temp_lambda_10.stypy_param_names_list = ['x']
        _stypy_temp_lambda_10.stypy_varargs_param_name = None
        _stypy_temp_lambda_10.stypy_kwargs_param_name = None
        _stypy_temp_lambda_10.stypy_call_defaults = defaults
        _stypy_temp_lambda_10.stypy_call_varargs = varargs
        _stypy_temp_lambda_10.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_10', ['x'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Stacktrace push for error reporting
        localization.set_stack_trace('_stypy_temp_lambda_10', ['x'], arguments)
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of the lambda function code ##################

        
        # Call to matvec(...): (line 809)
        # Processing the call arguments (line 809)
        # Getting the type of 'x' (line 809)
        x_21668 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 809, 34), 'x', False)
        # Processing the call keyword arguments (line 809)
        kwargs_21669 = {}
        # Getting the type of 'A' (line 809)
        A_21666 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 809, 24), 'A', False)
        # Obtaining the member 'matvec' of a type (line 809)
        matvec_21667 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 809, 24), A_21666, 'matvec')
        # Calling matvec(args, kwargs) (line 809)
        matvec_call_result_21670 = invoke(stypy.reporting.localization.Localization(__file__, 809, 24), matvec_21667, *[x_21668], **kwargs_21669)
        
        # Assigning the return type of the lambda function
        # Assigning a type to the variable 'stypy_return_type' (line 809)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 809, 14), 'stypy_return_type', matvec_call_result_21670)
        
        # ################# End of the lambda function code ##################

        # Stacktrace pop (error reporting)
        localization.unset_stack_trace()
        
        # Storing the return type of function '_stypy_temp_lambda_10' in the type store
        # Getting the type of 'stypy_return_type' (line 809)
        stypy_return_type_21671 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 809, 14), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_21671)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_stypy_temp_lambda_10'
        return stypy_return_type_21671

    # Assigning a type to the variable '_stypy_temp_lambda_10' (line 809)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 809, 14), '_stypy_temp_lambda_10', _stypy_temp_lambda_10)
    # Getting the type of '_stypy_temp_lambda_10' (line 809)
    _stypy_temp_lambda_10_21672 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 809, 14), '_stypy_temp_lambda_10')
    # Assigning a type to the variable 'matvec1' (line 809)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 809, 4), 'matvec1', _stypy_temp_lambda_10_21672)
    
    # Assigning a Lambda to a Name (line 810):
    
    # Assigning a Lambda to a Name (line 810):

    @norecursion
    def _stypy_temp_lambda_11(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_stypy_temp_lambda_11'
        module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_11', 810, 15, True)
        # Passed parameters checking function
        _stypy_temp_lambda_11.stypy_localization = localization
        _stypy_temp_lambda_11.stypy_type_of_self = None
        _stypy_temp_lambda_11.stypy_type_store = module_type_store
        _stypy_temp_lambda_11.stypy_function_name = '_stypy_temp_lambda_11'
        _stypy_temp_lambda_11.stypy_param_names_list = ['x']
        _stypy_temp_lambda_11.stypy_varargs_param_name = None
        _stypy_temp_lambda_11.stypy_kwargs_param_name = None
        _stypy_temp_lambda_11.stypy_call_defaults = defaults
        _stypy_temp_lambda_11.stypy_call_varargs = varargs
        _stypy_temp_lambda_11.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_11', ['x'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Stacktrace push for error reporting
        localization.set_stack_trace('_stypy_temp_lambda_11', ['x'], arguments)
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of the lambda function code ##################

        
        # Call to rmatvec(...): (line 810)
        # Processing the call arguments (line 810)
        # Getting the type of 'x' (line 810)
        x_21675 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 810, 35), 'x', False)
        # Processing the call keyword arguments (line 810)
        kwargs_21676 = {}
        # Getting the type of 'A' (line 810)
        A_21673 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 810, 25), 'A', False)
        # Obtaining the member 'rmatvec' of a type (line 810)
        rmatvec_21674 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 810, 25), A_21673, 'rmatvec')
        # Calling rmatvec(args, kwargs) (line 810)
        rmatvec_call_result_21677 = invoke(stypy.reporting.localization.Localization(__file__, 810, 25), rmatvec_21674, *[x_21675], **kwargs_21676)
        
        # Assigning the return type of the lambda function
        # Assigning a type to the variable 'stypy_return_type' (line 810)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 810, 15), 'stypy_return_type', rmatvec_call_result_21677)
        
        # ################# End of the lambda function code ##################

        # Stacktrace pop (error reporting)
        localization.unset_stack_trace()
        
        # Storing the return type of function '_stypy_temp_lambda_11' in the type store
        # Getting the type of 'stypy_return_type' (line 810)
        stypy_return_type_21678 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 810, 15), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_21678)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_stypy_temp_lambda_11'
        return stypy_return_type_21678

    # Assigning a type to the variable '_stypy_temp_lambda_11' (line 810)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 810, 15), '_stypy_temp_lambda_11', _stypy_temp_lambda_11)
    # Getting the type of '_stypy_temp_lambda_11' (line 810)
    _stypy_temp_lambda_11_21679 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 810, 15), '_stypy_temp_lambda_11')
    # Assigning a type to the variable 'matveca1' (line 810)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 810, 4), 'matveca1', _stypy_temp_lambda_11_21679)
    
    # Assigning a Lambda to a Name (line 811):
    
    # Assigning a Lambda to a Name (line 811):

    @norecursion
    def _stypy_temp_lambda_12(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_stypy_temp_lambda_12'
        module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_12', 811, 14, True)
        # Passed parameters checking function
        _stypy_temp_lambda_12.stypy_localization = localization
        _stypy_temp_lambda_12.stypy_type_of_self = None
        _stypy_temp_lambda_12.stypy_type_store = module_type_store
        _stypy_temp_lambda_12.stypy_function_name = '_stypy_temp_lambda_12'
        _stypy_temp_lambda_12.stypy_param_names_list = ['x']
        _stypy_temp_lambda_12.stypy_varargs_param_name = None
        _stypy_temp_lambda_12.stypy_kwargs_param_name = None
        _stypy_temp_lambda_12.stypy_call_defaults = defaults
        _stypy_temp_lambda_12.stypy_call_varargs = varargs
        _stypy_temp_lambda_12.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_12', ['x'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Stacktrace push for error reporting
        localization.set_stack_trace('_stypy_temp_lambda_12', ['x'], arguments)
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of the lambda function code ##################

        
        # Call to matvec(...): (line 811)
        # Processing the call arguments (line 811)
        # Getting the type of 'x' (line 811)
        x_21682 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 811, 34), 'x', False)
        # Processing the call keyword arguments (line 811)
        kwargs_21683 = {}
        # Getting the type of 'B' (line 811)
        B_21680 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 811, 24), 'B', False)
        # Obtaining the member 'matvec' of a type (line 811)
        matvec_21681 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 811, 24), B_21680, 'matvec')
        # Calling matvec(args, kwargs) (line 811)
        matvec_call_result_21684 = invoke(stypy.reporting.localization.Localization(__file__, 811, 24), matvec_21681, *[x_21682], **kwargs_21683)
        
        # Assigning the return type of the lambda function
        # Assigning a type to the variable 'stypy_return_type' (line 811)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 811, 14), 'stypy_return_type', matvec_call_result_21684)
        
        # ################# End of the lambda function code ##################

        # Stacktrace pop (error reporting)
        localization.unset_stack_trace()
        
        # Storing the return type of function '_stypy_temp_lambda_12' in the type store
        # Getting the type of 'stypy_return_type' (line 811)
        stypy_return_type_21685 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 811, 14), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_21685)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_stypy_temp_lambda_12'
        return stypy_return_type_21685

    # Assigning a type to the variable '_stypy_temp_lambda_12' (line 811)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 811, 14), '_stypy_temp_lambda_12', _stypy_temp_lambda_12)
    # Getting the type of '_stypy_temp_lambda_12' (line 811)
    _stypy_temp_lambda_12_21686 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 811, 14), '_stypy_temp_lambda_12')
    # Assigning a type to the variable 'matvec2' (line 811)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 811, 4), 'matvec2', _stypy_temp_lambda_12_21686)
    
    # Assigning a Lambda to a Name (line 812):
    
    # Assigning a Lambda to a Name (line 812):

    @norecursion
    def _stypy_temp_lambda_13(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_stypy_temp_lambda_13'
        module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_13', 812, 15, True)
        # Passed parameters checking function
        _stypy_temp_lambda_13.stypy_localization = localization
        _stypy_temp_lambda_13.stypy_type_of_self = None
        _stypy_temp_lambda_13.stypy_type_store = module_type_store
        _stypy_temp_lambda_13.stypy_function_name = '_stypy_temp_lambda_13'
        _stypy_temp_lambda_13.stypy_param_names_list = ['x']
        _stypy_temp_lambda_13.stypy_varargs_param_name = None
        _stypy_temp_lambda_13.stypy_kwargs_param_name = None
        _stypy_temp_lambda_13.stypy_call_defaults = defaults
        _stypy_temp_lambda_13.stypy_call_varargs = varargs
        _stypy_temp_lambda_13.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_13', ['x'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Stacktrace push for error reporting
        localization.set_stack_trace('_stypy_temp_lambda_13', ['x'], arguments)
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of the lambda function code ##################

        
        # Call to rmatvec(...): (line 812)
        # Processing the call arguments (line 812)
        # Getting the type of 'x' (line 812)
        x_21689 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 812, 35), 'x', False)
        # Processing the call keyword arguments (line 812)
        kwargs_21690 = {}
        # Getting the type of 'B' (line 812)
        B_21687 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 812, 25), 'B', False)
        # Obtaining the member 'rmatvec' of a type (line 812)
        rmatvec_21688 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 812, 25), B_21687, 'rmatvec')
        # Calling rmatvec(args, kwargs) (line 812)
        rmatvec_call_result_21691 = invoke(stypy.reporting.localization.Localization(__file__, 812, 25), rmatvec_21688, *[x_21689], **kwargs_21690)
        
        # Assigning the return type of the lambda function
        # Assigning a type to the variable 'stypy_return_type' (line 812)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 812, 15), 'stypy_return_type', rmatvec_call_result_21691)
        
        # ################# End of the lambda function code ##################

        # Stacktrace pop (error reporting)
        localization.unset_stack_trace()
        
        # Storing the return type of function '_stypy_temp_lambda_13' in the type store
        # Getting the type of 'stypy_return_type' (line 812)
        stypy_return_type_21692 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 812, 15), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_21692)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_stypy_temp_lambda_13'
        return stypy_return_type_21692

    # Assigning a type to the variable '_stypy_temp_lambda_13' (line 812)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 812, 15), '_stypy_temp_lambda_13', _stypy_temp_lambda_13)
    # Getting the type of '_stypy_temp_lambda_13' (line 812)
    _stypy_temp_lambda_13_21693 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 812, 15), '_stypy_temp_lambda_13')
    # Assigning a type to the variable 'matveca2' (line 812)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 812, 4), 'matveca2', _stypy_temp_lambda_13_21693)
    
    
    # Call to _is_real(...): (line 813)
    # Processing the call arguments (line 813)
    # Getting the type of 'A' (line 813)
    A_21695 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 813, 16), 'A', False)
    # Processing the call keyword arguments (line 813)
    kwargs_21696 = {}
    # Getting the type of '_is_real' (line 813)
    _is_real_21694 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 813, 7), '_is_real', False)
    # Calling _is_real(args, kwargs) (line 813)
    _is_real_call_result_21697 = invoke(stypy.reporting.localization.Localization(__file__, 813, 7), _is_real_21694, *[A_21695], **kwargs_21696)
    
    # Testing the type of an if condition (line 813)
    if_condition_21698 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 813, 4), _is_real_call_result_21697)
    # Assigning a type to the variable 'if_condition_21698' (line 813)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 813, 4), 'if_condition_21698', if_condition_21698)
    # SSA begins for if statement (line 813)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to idd_diffsnorm(...): (line 814)
    # Processing the call arguments (line 814)
    # Getting the type of 'm' (line 815)
    m_21701 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 815, 12), 'm', False)
    # Getting the type of 'n' (line 815)
    n_21702 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 815, 15), 'n', False)
    # Getting the type of 'matveca1' (line 815)
    matveca1_21703 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 815, 18), 'matveca1', False)
    # Getting the type of 'matveca2' (line 815)
    matveca2_21704 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 815, 28), 'matveca2', False)
    # Getting the type of 'matvec1' (line 815)
    matvec1_21705 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 815, 38), 'matvec1', False)
    # Getting the type of 'matvec2' (line 815)
    matvec2_21706 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 815, 47), 'matvec2', False)
    # Processing the call keyword arguments (line 814)
    # Getting the type of 'its' (line 815)
    its_21707 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 815, 60), 'its', False)
    keyword_21708 = its_21707
    kwargs_21709 = {'its': keyword_21708}
    # Getting the type of 'backend' (line 814)
    backend_21699 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 814, 15), 'backend', False)
    # Obtaining the member 'idd_diffsnorm' of a type (line 814)
    idd_diffsnorm_21700 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 814, 15), backend_21699, 'idd_diffsnorm')
    # Calling idd_diffsnorm(args, kwargs) (line 814)
    idd_diffsnorm_call_result_21710 = invoke(stypy.reporting.localization.Localization(__file__, 814, 15), idd_diffsnorm_21700, *[m_21701, n_21702, matveca1_21703, matveca2_21704, matvec1_21705, matvec2_21706], **kwargs_21709)
    
    # Assigning a type to the variable 'stypy_return_type' (line 814)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 814, 8), 'stypy_return_type', idd_diffsnorm_call_result_21710)
    # SSA branch for the else part of an if statement (line 813)
    module_type_store.open_ssa_branch('else')
    
    # Call to idz_diffsnorm(...): (line 817)
    # Processing the call arguments (line 817)
    # Getting the type of 'm' (line 818)
    m_21713 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 818, 12), 'm', False)
    # Getting the type of 'n' (line 818)
    n_21714 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 818, 15), 'n', False)
    # Getting the type of 'matveca1' (line 818)
    matveca1_21715 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 818, 18), 'matveca1', False)
    # Getting the type of 'matveca2' (line 818)
    matveca2_21716 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 818, 28), 'matveca2', False)
    # Getting the type of 'matvec1' (line 818)
    matvec1_21717 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 818, 38), 'matvec1', False)
    # Getting the type of 'matvec2' (line 818)
    matvec2_21718 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 818, 47), 'matvec2', False)
    # Processing the call keyword arguments (line 817)
    # Getting the type of 'its' (line 818)
    its_21719 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 818, 60), 'its', False)
    keyword_21720 = its_21719
    kwargs_21721 = {'its': keyword_21720}
    # Getting the type of 'backend' (line 817)
    backend_21711 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 817, 15), 'backend', False)
    # Obtaining the member 'idz_diffsnorm' of a type (line 817)
    idz_diffsnorm_21712 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 817, 15), backend_21711, 'idz_diffsnorm')
    # Calling idz_diffsnorm(args, kwargs) (line 817)
    idz_diffsnorm_call_result_21722 = invoke(stypy.reporting.localization.Localization(__file__, 817, 15), idz_diffsnorm_21712, *[m_21713, n_21714, matveca1_21715, matveca2_21716, matvec1_21717, matvec2_21718], **kwargs_21721)
    
    # Assigning a type to the variable 'stypy_return_type' (line 817)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 817, 8), 'stypy_return_type', idz_diffsnorm_call_result_21722)
    # SSA join for if statement (line 813)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'estimate_spectral_norm_diff(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'estimate_spectral_norm_diff' in the type store
    # Getting the type of 'stypy_return_type' (line 780)
    stypy_return_type_21723 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 780, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_21723)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'estimate_spectral_norm_diff'
    return stypy_return_type_21723

# Assigning a type to the variable 'estimate_spectral_norm_diff' (line 780)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 780, 0), 'estimate_spectral_norm_diff', estimate_spectral_norm_diff)

@norecursion
def svd(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'True' (line 821)
    True_21724 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 821, 26), 'True')
    defaults = [True_21724]
    # Create a new context for function 'svd'
    module_type_store = module_type_store.open_function_context('svd', 821, 0, False)
    
    # Passed parameters checking function
    svd.stypy_localization = localization
    svd.stypy_type_of_self = None
    svd.stypy_type_store = module_type_store
    svd.stypy_function_name = 'svd'
    svd.stypy_param_names_list = ['A', 'eps_or_k', 'rand']
    svd.stypy_varargs_param_name = None
    svd.stypy_kwargs_param_name = None
    svd.stypy_call_defaults = defaults
    svd.stypy_call_varargs = varargs
    svd.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'svd', ['A', 'eps_or_k', 'rand'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'svd', localization, ['A', 'eps_or_k', 'rand'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'svd(...)' code ##################

    str_21725 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 867, (-1)), 'str', '\n    Compute SVD of a matrix via an ID.\n\n    An SVD of a matrix `A` is a factorization::\n\n        A = numpy.dot(U, numpy.dot(numpy.diag(S), V.conj().T))\n\n    where `U` and `V` have orthonormal columns and `S` is nonnegative.\n\n    The SVD can be computed to any relative precision or rank (depending on the\n    value of `eps_or_k`).\n\n    See also :func:`interp_decomp` and :func:`id_to_svd`.\n\n    ..  This function automatically detects the form of the input parameters and\n        passes them to the appropriate backend. For details, see\n        :func:`backend.iddp_svd`, :func:`backend.iddp_asvd`,\n        :func:`backend.iddp_rsvd`, :func:`backend.iddr_svd`,\n        :func:`backend.iddr_asvd`, :func:`backend.iddr_rsvd`,\n        :func:`backend.idzp_svd`, :func:`backend.idzp_asvd`,\n        :func:`backend.idzp_rsvd`, :func:`backend.idzr_svd`,\n        :func:`backend.idzr_asvd`, and :func:`backend.idzr_rsvd`.\n\n    Parameters\n    ----------\n    A : :class:`numpy.ndarray` or :class:`scipy.sparse.linalg.LinearOperator`\n        Matrix to be factored, given as either a :class:`numpy.ndarray` or a\n        :class:`scipy.sparse.linalg.LinearOperator` with the `matvec` and\n        `rmatvec` methods (to apply the matrix and its adjoint).\n    eps_or_k : float or int\n        Relative error (if `eps_or_k < 1`) or rank (if `eps_or_k >= 1`) of\n        approximation.\n    rand : bool, optional\n        Whether to use random sampling if `A` is of type :class:`numpy.ndarray`\n        (randomized algorithms are always used if `A` is of type\n        :class:`scipy.sparse.linalg.LinearOperator`).\n\n    Returns\n    -------\n    U : :class:`numpy.ndarray`\n        Left singular vectors.\n    S : :class:`numpy.ndarray`\n        Singular values.\n    V : :class:`numpy.ndarray`\n        Right singular vectors.\n    ')
    stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 868, 4))
    
    # 'from scipy.sparse.linalg import LinearOperator' statement (line 868)
    update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/linalg/')
    import_21726 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 868, 4), 'scipy.sparse.linalg')

    if (type(import_21726) is not StypyTypeError):

        if (import_21726 != 'pyd_module'):
            __import__(import_21726)
            sys_modules_21727 = sys.modules[import_21726]
            import_from_module(stypy.reporting.localization.Localization(__file__, 868, 4), 'scipy.sparse.linalg', sys_modules_21727.module_type_store, module_type_store, ['LinearOperator'])
            nest_module(stypy.reporting.localization.Localization(__file__, 868, 4), __file__, sys_modules_21727, sys_modules_21727.module_type_store, module_type_store)
        else:
            from scipy.sparse.linalg import LinearOperator

            import_from_module(stypy.reporting.localization.Localization(__file__, 868, 4), 'scipy.sparse.linalg', None, module_type_store, ['LinearOperator'], [LinearOperator])

    else:
        # Assigning a type to the variable 'scipy.sparse.linalg' (line 868)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 868, 4), 'scipy.sparse.linalg', import_21726)

    remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/linalg/')
    
    
    # Assigning a Call to a Name (line 870):
    
    # Assigning a Call to a Name (line 870):
    
    # Call to _is_real(...): (line 870)
    # Processing the call arguments (line 870)
    # Getting the type of 'A' (line 870)
    A_21729 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 870, 20), 'A', False)
    # Processing the call keyword arguments (line 870)
    kwargs_21730 = {}
    # Getting the type of '_is_real' (line 870)
    _is_real_21728 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 870, 11), '_is_real', False)
    # Calling _is_real(args, kwargs) (line 870)
    _is_real_call_result_21731 = invoke(stypy.reporting.localization.Localization(__file__, 870, 11), _is_real_21728, *[A_21729], **kwargs_21730)
    
    # Assigning a type to the variable 'real' (line 870)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 870, 4), 'real', _is_real_call_result_21731)
    
    
    # Call to isinstance(...): (line 872)
    # Processing the call arguments (line 872)
    # Getting the type of 'A' (line 872)
    A_21733 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 872, 18), 'A', False)
    # Getting the type of 'np' (line 872)
    np_21734 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 872, 21), 'np', False)
    # Obtaining the member 'ndarray' of a type (line 872)
    ndarray_21735 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 872, 21), np_21734, 'ndarray')
    # Processing the call keyword arguments (line 872)
    kwargs_21736 = {}
    # Getting the type of 'isinstance' (line 872)
    isinstance_21732 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 872, 7), 'isinstance', False)
    # Calling isinstance(args, kwargs) (line 872)
    isinstance_call_result_21737 = invoke(stypy.reporting.localization.Localization(__file__, 872, 7), isinstance_21732, *[A_21733, ndarray_21735], **kwargs_21736)
    
    # Testing the type of an if condition (line 872)
    if_condition_21738 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 872, 4), isinstance_call_result_21737)
    # Assigning a type to the variable 'if_condition_21738' (line 872)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 872, 4), 'if_condition_21738', if_condition_21738)
    # SSA begins for if statement (line 872)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    
    # Getting the type of 'eps_or_k' (line 873)
    eps_or_k_21739 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 873, 11), 'eps_or_k')
    int_21740 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 873, 22), 'int')
    # Applying the binary operator '<' (line 873)
    result_lt_21741 = python_operator(stypy.reporting.localization.Localization(__file__, 873, 11), '<', eps_or_k_21739, int_21740)
    
    # Testing the type of an if condition (line 873)
    if_condition_21742 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 873, 8), result_lt_21741)
    # Assigning a type to the variable 'if_condition_21742' (line 873)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 873, 8), 'if_condition_21742', if_condition_21742)
    # SSA begins for if statement (line 873)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Name to a Name (line 874):
    
    # Assigning a Name to a Name (line 874):
    # Getting the type of 'eps_or_k' (line 874)
    eps_or_k_21743 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 874, 18), 'eps_or_k')
    # Assigning a type to the variable 'eps' (line 874)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 874, 12), 'eps', eps_or_k_21743)
    
    # Getting the type of 'rand' (line 875)
    rand_21744 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 875, 15), 'rand')
    # Testing the type of an if condition (line 875)
    if_condition_21745 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 875, 12), rand_21744)
    # Assigning a type to the variable 'if_condition_21745' (line 875)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 875, 12), 'if_condition_21745', if_condition_21745)
    # SSA begins for if statement (line 875)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Getting the type of 'real' (line 876)
    real_21746 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 876, 19), 'real')
    # Testing the type of an if condition (line 876)
    if_condition_21747 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 876, 16), real_21746)
    # Assigning a type to the variable 'if_condition_21747' (line 876)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 876, 16), 'if_condition_21747', if_condition_21747)
    # SSA begins for if statement (line 876)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Tuple (line 877):
    
    # Assigning a Subscript to a Name (line 877):
    
    # Obtaining the type of the subscript
    int_21748 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 877, 20), 'int')
    
    # Call to iddp_asvd(...): (line 877)
    # Processing the call arguments (line 877)
    # Getting the type of 'eps' (line 877)
    eps_21751 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 877, 48), 'eps', False)
    # Getting the type of 'A' (line 877)
    A_21752 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 877, 53), 'A', False)
    # Processing the call keyword arguments (line 877)
    kwargs_21753 = {}
    # Getting the type of 'backend' (line 877)
    backend_21749 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 877, 30), 'backend', False)
    # Obtaining the member 'iddp_asvd' of a type (line 877)
    iddp_asvd_21750 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 877, 30), backend_21749, 'iddp_asvd')
    # Calling iddp_asvd(args, kwargs) (line 877)
    iddp_asvd_call_result_21754 = invoke(stypy.reporting.localization.Localization(__file__, 877, 30), iddp_asvd_21750, *[eps_21751, A_21752], **kwargs_21753)
    
    # Obtaining the member '__getitem__' of a type (line 877)
    getitem___21755 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 877, 20), iddp_asvd_call_result_21754, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 877)
    subscript_call_result_21756 = invoke(stypy.reporting.localization.Localization(__file__, 877, 20), getitem___21755, int_21748)
    
    # Assigning a type to the variable 'tuple_var_assignment_20832' (line 877)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 877, 20), 'tuple_var_assignment_20832', subscript_call_result_21756)
    
    # Assigning a Subscript to a Name (line 877):
    
    # Obtaining the type of the subscript
    int_21757 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 877, 20), 'int')
    
    # Call to iddp_asvd(...): (line 877)
    # Processing the call arguments (line 877)
    # Getting the type of 'eps' (line 877)
    eps_21760 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 877, 48), 'eps', False)
    # Getting the type of 'A' (line 877)
    A_21761 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 877, 53), 'A', False)
    # Processing the call keyword arguments (line 877)
    kwargs_21762 = {}
    # Getting the type of 'backend' (line 877)
    backend_21758 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 877, 30), 'backend', False)
    # Obtaining the member 'iddp_asvd' of a type (line 877)
    iddp_asvd_21759 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 877, 30), backend_21758, 'iddp_asvd')
    # Calling iddp_asvd(args, kwargs) (line 877)
    iddp_asvd_call_result_21763 = invoke(stypy.reporting.localization.Localization(__file__, 877, 30), iddp_asvd_21759, *[eps_21760, A_21761], **kwargs_21762)
    
    # Obtaining the member '__getitem__' of a type (line 877)
    getitem___21764 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 877, 20), iddp_asvd_call_result_21763, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 877)
    subscript_call_result_21765 = invoke(stypy.reporting.localization.Localization(__file__, 877, 20), getitem___21764, int_21757)
    
    # Assigning a type to the variable 'tuple_var_assignment_20833' (line 877)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 877, 20), 'tuple_var_assignment_20833', subscript_call_result_21765)
    
    # Assigning a Subscript to a Name (line 877):
    
    # Obtaining the type of the subscript
    int_21766 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 877, 20), 'int')
    
    # Call to iddp_asvd(...): (line 877)
    # Processing the call arguments (line 877)
    # Getting the type of 'eps' (line 877)
    eps_21769 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 877, 48), 'eps', False)
    # Getting the type of 'A' (line 877)
    A_21770 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 877, 53), 'A', False)
    # Processing the call keyword arguments (line 877)
    kwargs_21771 = {}
    # Getting the type of 'backend' (line 877)
    backend_21767 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 877, 30), 'backend', False)
    # Obtaining the member 'iddp_asvd' of a type (line 877)
    iddp_asvd_21768 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 877, 30), backend_21767, 'iddp_asvd')
    # Calling iddp_asvd(args, kwargs) (line 877)
    iddp_asvd_call_result_21772 = invoke(stypy.reporting.localization.Localization(__file__, 877, 30), iddp_asvd_21768, *[eps_21769, A_21770], **kwargs_21771)
    
    # Obtaining the member '__getitem__' of a type (line 877)
    getitem___21773 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 877, 20), iddp_asvd_call_result_21772, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 877)
    subscript_call_result_21774 = invoke(stypy.reporting.localization.Localization(__file__, 877, 20), getitem___21773, int_21766)
    
    # Assigning a type to the variable 'tuple_var_assignment_20834' (line 877)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 877, 20), 'tuple_var_assignment_20834', subscript_call_result_21774)
    
    # Assigning a Name to a Name (line 877):
    # Getting the type of 'tuple_var_assignment_20832' (line 877)
    tuple_var_assignment_20832_21775 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 877, 20), 'tuple_var_assignment_20832')
    # Assigning a type to the variable 'U' (line 877)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 877, 20), 'U', tuple_var_assignment_20832_21775)
    
    # Assigning a Name to a Name (line 877):
    # Getting the type of 'tuple_var_assignment_20833' (line 877)
    tuple_var_assignment_20833_21776 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 877, 20), 'tuple_var_assignment_20833')
    # Assigning a type to the variable 'V' (line 877)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 877, 23), 'V', tuple_var_assignment_20833_21776)
    
    # Assigning a Name to a Name (line 877):
    # Getting the type of 'tuple_var_assignment_20834' (line 877)
    tuple_var_assignment_20834_21777 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 877, 20), 'tuple_var_assignment_20834')
    # Assigning a type to the variable 'S' (line 877)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 877, 26), 'S', tuple_var_assignment_20834_21777)
    # SSA branch for the else part of an if statement (line 876)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Call to a Tuple (line 879):
    
    # Assigning a Subscript to a Name (line 879):
    
    # Obtaining the type of the subscript
    int_21778 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 879, 20), 'int')
    
    # Call to idzp_asvd(...): (line 879)
    # Processing the call arguments (line 879)
    # Getting the type of 'eps' (line 879)
    eps_21781 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 879, 48), 'eps', False)
    # Getting the type of 'A' (line 879)
    A_21782 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 879, 53), 'A', False)
    # Processing the call keyword arguments (line 879)
    kwargs_21783 = {}
    # Getting the type of 'backend' (line 879)
    backend_21779 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 879, 30), 'backend', False)
    # Obtaining the member 'idzp_asvd' of a type (line 879)
    idzp_asvd_21780 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 879, 30), backend_21779, 'idzp_asvd')
    # Calling idzp_asvd(args, kwargs) (line 879)
    idzp_asvd_call_result_21784 = invoke(stypy.reporting.localization.Localization(__file__, 879, 30), idzp_asvd_21780, *[eps_21781, A_21782], **kwargs_21783)
    
    # Obtaining the member '__getitem__' of a type (line 879)
    getitem___21785 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 879, 20), idzp_asvd_call_result_21784, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 879)
    subscript_call_result_21786 = invoke(stypy.reporting.localization.Localization(__file__, 879, 20), getitem___21785, int_21778)
    
    # Assigning a type to the variable 'tuple_var_assignment_20835' (line 879)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 879, 20), 'tuple_var_assignment_20835', subscript_call_result_21786)
    
    # Assigning a Subscript to a Name (line 879):
    
    # Obtaining the type of the subscript
    int_21787 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 879, 20), 'int')
    
    # Call to idzp_asvd(...): (line 879)
    # Processing the call arguments (line 879)
    # Getting the type of 'eps' (line 879)
    eps_21790 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 879, 48), 'eps', False)
    # Getting the type of 'A' (line 879)
    A_21791 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 879, 53), 'A', False)
    # Processing the call keyword arguments (line 879)
    kwargs_21792 = {}
    # Getting the type of 'backend' (line 879)
    backend_21788 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 879, 30), 'backend', False)
    # Obtaining the member 'idzp_asvd' of a type (line 879)
    idzp_asvd_21789 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 879, 30), backend_21788, 'idzp_asvd')
    # Calling idzp_asvd(args, kwargs) (line 879)
    idzp_asvd_call_result_21793 = invoke(stypy.reporting.localization.Localization(__file__, 879, 30), idzp_asvd_21789, *[eps_21790, A_21791], **kwargs_21792)
    
    # Obtaining the member '__getitem__' of a type (line 879)
    getitem___21794 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 879, 20), idzp_asvd_call_result_21793, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 879)
    subscript_call_result_21795 = invoke(stypy.reporting.localization.Localization(__file__, 879, 20), getitem___21794, int_21787)
    
    # Assigning a type to the variable 'tuple_var_assignment_20836' (line 879)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 879, 20), 'tuple_var_assignment_20836', subscript_call_result_21795)
    
    # Assigning a Subscript to a Name (line 879):
    
    # Obtaining the type of the subscript
    int_21796 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 879, 20), 'int')
    
    # Call to idzp_asvd(...): (line 879)
    # Processing the call arguments (line 879)
    # Getting the type of 'eps' (line 879)
    eps_21799 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 879, 48), 'eps', False)
    # Getting the type of 'A' (line 879)
    A_21800 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 879, 53), 'A', False)
    # Processing the call keyword arguments (line 879)
    kwargs_21801 = {}
    # Getting the type of 'backend' (line 879)
    backend_21797 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 879, 30), 'backend', False)
    # Obtaining the member 'idzp_asvd' of a type (line 879)
    idzp_asvd_21798 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 879, 30), backend_21797, 'idzp_asvd')
    # Calling idzp_asvd(args, kwargs) (line 879)
    idzp_asvd_call_result_21802 = invoke(stypy.reporting.localization.Localization(__file__, 879, 30), idzp_asvd_21798, *[eps_21799, A_21800], **kwargs_21801)
    
    # Obtaining the member '__getitem__' of a type (line 879)
    getitem___21803 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 879, 20), idzp_asvd_call_result_21802, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 879)
    subscript_call_result_21804 = invoke(stypy.reporting.localization.Localization(__file__, 879, 20), getitem___21803, int_21796)
    
    # Assigning a type to the variable 'tuple_var_assignment_20837' (line 879)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 879, 20), 'tuple_var_assignment_20837', subscript_call_result_21804)
    
    # Assigning a Name to a Name (line 879):
    # Getting the type of 'tuple_var_assignment_20835' (line 879)
    tuple_var_assignment_20835_21805 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 879, 20), 'tuple_var_assignment_20835')
    # Assigning a type to the variable 'U' (line 879)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 879, 20), 'U', tuple_var_assignment_20835_21805)
    
    # Assigning a Name to a Name (line 879):
    # Getting the type of 'tuple_var_assignment_20836' (line 879)
    tuple_var_assignment_20836_21806 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 879, 20), 'tuple_var_assignment_20836')
    # Assigning a type to the variable 'V' (line 879)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 879, 23), 'V', tuple_var_assignment_20836_21806)
    
    # Assigning a Name to a Name (line 879):
    # Getting the type of 'tuple_var_assignment_20837' (line 879)
    tuple_var_assignment_20837_21807 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 879, 20), 'tuple_var_assignment_20837')
    # Assigning a type to the variable 'S' (line 879)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 879, 26), 'S', tuple_var_assignment_20837_21807)
    # SSA join for if statement (line 876)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA branch for the else part of an if statement (line 875)
    module_type_store.open_ssa_branch('else')
    
    # Getting the type of 'real' (line 881)
    real_21808 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 881, 19), 'real')
    # Testing the type of an if condition (line 881)
    if_condition_21809 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 881, 16), real_21808)
    # Assigning a type to the variable 'if_condition_21809' (line 881)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 881, 16), 'if_condition_21809', if_condition_21809)
    # SSA begins for if statement (line 881)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Tuple (line 882):
    
    # Assigning a Subscript to a Name (line 882):
    
    # Obtaining the type of the subscript
    int_21810 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 882, 20), 'int')
    
    # Call to iddp_svd(...): (line 882)
    # Processing the call arguments (line 882)
    # Getting the type of 'eps' (line 882)
    eps_21813 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 882, 47), 'eps', False)
    # Getting the type of 'A' (line 882)
    A_21814 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 882, 52), 'A', False)
    # Processing the call keyword arguments (line 882)
    kwargs_21815 = {}
    # Getting the type of 'backend' (line 882)
    backend_21811 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 882, 30), 'backend', False)
    # Obtaining the member 'iddp_svd' of a type (line 882)
    iddp_svd_21812 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 882, 30), backend_21811, 'iddp_svd')
    # Calling iddp_svd(args, kwargs) (line 882)
    iddp_svd_call_result_21816 = invoke(stypy.reporting.localization.Localization(__file__, 882, 30), iddp_svd_21812, *[eps_21813, A_21814], **kwargs_21815)
    
    # Obtaining the member '__getitem__' of a type (line 882)
    getitem___21817 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 882, 20), iddp_svd_call_result_21816, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 882)
    subscript_call_result_21818 = invoke(stypy.reporting.localization.Localization(__file__, 882, 20), getitem___21817, int_21810)
    
    # Assigning a type to the variable 'tuple_var_assignment_20838' (line 882)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 882, 20), 'tuple_var_assignment_20838', subscript_call_result_21818)
    
    # Assigning a Subscript to a Name (line 882):
    
    # Obtaining the type of the subscript
    int_21819 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 882, 20), 'int')
    
    # Call to iddp_svd(...): (line 882)
    # Processing the call arguments (line 882)
    # Getting the type of 'eps' (line 882)
    eps_21822 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 882, 47), 'eps', False)
    # Getting the type of 'A' (line 882)
    A_21823 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 882, 52), 'A', False)
    # Processing the call keyword arguments (line 882)
    kwargs_21824 = {}
    # Getting the type of 'backend' (line 882)
    backend_21820 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 882, 30), 'backend', False)
    # Obtaining the member 'iddp_svd' of a type (line 882)
    iddp_svd_21821 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 882, 30), backend_21820, 'iddp_svd')
    # Calling iddp_svd(args, kwargs) (line 882)
    iddp_svd_call_result_21825 = invoke(stypy.reporting.localization.Localization(__file__, 882, 30), iddp_svd_21821, *[eps_21822, A_21823], **kwargs_21824)
    
    # Obtaining the member '__getitem__' of a type (line 882)
    getitem___21826 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 882, 20), iddp_svd_call_result_21825, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 882)
    subscript_call_result_21827 = invoke(stypy.reporting.localization.Localization(__file__, 882, 20), getitem___21826, int_21819)
    
    # Assigning a type to the variable 'tuple_var_assignment_20839' (line 882)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 882, 20), 'tuple_var_assignment_20839', subscript_call_result_21827)
    
    # Assigning a Subscript to a Name (line 882):
    
    # Obtaining the type of the subscript
    int_21828 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 882, 20), 'int')
    
    # Call to iddp_svd(...): (line 882)
    # Processing the call arguments (line 882)
    # Getting the type of 'eps' (line 882)
    eps_21831 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 882, 47), 'eps', False)
    # Getting the type of 'A' (line 882)
    A_21832 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 882, 52), 'A', False)
    # Processing the call keyword arguments (line 882)
    kwargs_21833 = {}
    # Getting the type of 'backend' (line 882)
    backend_21829 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 882, 30), 'backend', False)
    # Obtaining the member 'iddp_svd' of a type (line 882)
    iddp_svd_21830 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 882, 30), backend_21829, 'iddp_svd')
    # Calling iddp_svd(args, kwargs) (line 882)
    iddp_svd_call_result_21834 = invoke(stypy.reporting.localization.Localization(__file__, 882, 30), iddp_svd_21830, *[eps_21831, A_21832], **kwargs_21833)
    
    # Obtaining the member '__getitem__' of a type (line 882)
    getitem___21835 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 882, 20), iddp_svd_call_result_21834, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 882)
    subscript_call_result_21836 = invoke(stypy.reporting.localization.Localization(__file__, 882, 20), getitem___21835, int_21828)
    
    # Assigning a type to the variable 'tuple_var_assignment_20840' (line 882)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 882, 20), 'tuple_var_assignment_20840', subscript_call_result_21836)
    
    # Assigning a Name to a Name (line 882):
    # Getting the type of 'tuple_var_assignment_20838' (line 882)
    tuple_var_assignment_20838_21837 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 882, 20), 'tuple_var_assignment_20838')
    # Assigning a type to the variable 'U' (line 882)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 882, 20), 'U', tuple_var_assignment_20838_21837)
    
    # Assigning a Name to a Name (line 882):
    # Getting the type of 'tuple_var_assignment_20839' (line 882)
    tuple_var_assignment_20839_21838 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 882, 20), 'tuple_var_assignment_20839')
    # Assigning a type to the variable 'V' (line 882)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 882, 23), 'V', tuple_var_assignment_20839_21838)
    
    # Assigning a Name to a Name (line 882):
    # Getting the type of 'tuple_var_assignment_20840' (line 882)
    tuple_var_assignment_20840_21839 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 882, 20), 'tuple_var_assignment_20840')
    # Assigning a type to the variable 'S' (line 882)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 882, 26), 'S', tuple_var_assignment_20840_21839)
    # SSA branch for the else part of an if statement (line 881)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Call to a Tuple (line 884):
    
    # Assigning a Subscript to a Name (line 884):
    
    # Obtaining the type of the subscript
    int_21840 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 884, 20), 'int')
    
    # Call to idzp_svd(...): (line 884)
    # Processing the call arguments (line 884)
    # Getting the type of 'eps' (line 884)
    eps_21843 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 884, 47), 'eps', False)
    # Getting the type of 'A' (line 884)
    A_21844 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 884, 52), 'A', False)
    # Processing the call keyword arguments (line 884)
    kwargs_21845 = {}
    # Getting the type of 'backend' (line 884)
    backend_21841 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 884, 30), 'backend', False)
    # Obtaining the member 'idzp_svd' of a type (line 884)
    idzp_svd_21842 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 884, 30), backend_21841, 'idzp_svd')
    # Calling idzp_svd(args, kwargs) (line 884)
    idzp_svd_call_result_21846 = invoke(stypy.reporting.localization.Localization(__file__, 884, 30), idzp_svd_21842, *[eps_21843, A_21844], **kwargs_21845)
    
    # Obtaining the member '__getitem__' of a type (line 884)
    getitem___21847 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 884, 20), idzp_svd_call_result_21846, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 884)
    subscript_call_result_21848 = invoke(stypy.reporting.localization.Localization(__file__, 884, 20), getitem___21847, int_21840)
    
    # Assigning a type to the variable 'tuple_var_assignment_20841' (line 884)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 884, 20), 'tuple_var_assignment_20841', subscript_call_result_21848)
    
    # Assigning a Subscript to a Name (line 884):
    
    # Obtaining the type of the subscript
    int_21849 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 884, 20), 'int')
    
    # Call to idzp_svd(...): (line 884)
    # Processing the call arguments (line 884)
    # Getting the type of 'eps' (line 884)
    eps_21852 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 884, 47), 'eps', False)
    # Getting the type of 'A' (line 884)
    A_21853 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 884, 52), 'A', False)
    # Processing the call keyword arguments (line 884)
    kwargs_21854 = {}
    # Getting the type of 'backend' (line 884)
    backend_21850 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 884, 30), 'backend', False)
    # Obtaining the member 'idzp_svd' of a type (line 884)
    idzp_svd_21851 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 884, 30), backend_21850, 'idzp_svd')
    # Calling idzp_svd(args, kwargs) (line 884)
    idzp_svd_call_result_21855 = invoke(stypy.reporting.localization.Localization(__file__, 884, 30), idzp_svd_21851, *[eps_21852, A_21853], **kwargs_21854)
    
    # Obtaining the member '__getitem__' of a type (line 884)
    getitem___21856 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 884, 20), idzp_svd_call_result_21855, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 884)
    subscript_call_result_21857 = invoke(stypy.reporting.localization.Localization(__file__, 884, 20), getitem___21856, int_21849)
    
    # Assigning a type to the variable 'tuple_var_assignment_20842' (line 884)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 884, 20), 'tuple_var_assignment_20842', subscript_call_result_21857)
    
    # Assigning a Subscript to a Name (line 884):
    
    # Obtaining the type of the subscript
    int_21858 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 884, 20), 'int')
    
    # Call to idzp_svd(...): (line 884)
    # Processing the call arguments (line 884)
    # Getting the type of 'eps' (line 884)
    eps_21861 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 884, 47), 'eps', False)
    # Getting the type of 'A' (line 884)
    A_21862 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 884, 52), 'A', False)
    # Processing the call keyword arguments (line 884)
    kwargs_21863 = {}
    # Getting the type of 'backend' (line 884)
    backend_21859 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 884, 30), 'backend', False)
    # Obtaining the member 'idzp_svd' of a type (line 884)
    idzp_svd_21860 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 884, 30), backend_21859, 'idzp_svd')
    # Calling idzp_svd(args, kwargs) (line 884)
    idzp_svd_call_result_21864 = invoke(stypy.reporting.localization.Localization(__file__, 884, 30), idzp_svd_21860, *[eps_21861, A_21862], **kwargs_21863)
    
    # Obtaining the member '__getitem__' of a type (line 884)
    getitem___21865 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 884, 20), idzp_svd_call_result_21864, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 884)
    subscript_call_result_21866 = invoke(stypy.reporting.localization.Localization(__file__, 884, 20), getitem___21865, int_21858)
    
    # Assigning a type to the variable 'tuple_var_assignment_20843' (line 884)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 884, 20), 'tuple_var_assignment_20843', subscript_call_result_21866)
    
    # Assigning a Name to a Name (line 884):
    # Getting the type of 'tuple_var_assignment_20841' (line 884)
    tuple_var_assignment_20841_21867 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 884, 20), 'tuple_var_assignment_20841')
    # Assigning a type to the variable 'U' (line 884)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 884, 20), 'U', tuple_var_assignment_20841_21867)
    
    # Assigning a Name to a Name (line 884):
    # Getting the type of 'tuple_var_assignment_20842' (line 884)
    tuple_var_assignment_20842_21868 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 884, 20), 'tuple_var_assignment_20842')
    # Assigning a type to the variable 'V' (line 884)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 884, 23), 'V', tuple_var_assignment_20842_21868)
    
    # Assigning a Name to a Name (line 884):
    # Getting the type of 'tuple_var_assignment_20843' (line 884)
    tuple_var_assignment_20843_21869 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 884, 20), 'tuple_var_assignment_20843')
    # Assigning a type to the variable 'S' (line 884)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 884, 26), 'S', tuple_var_assignment_20843_21869)
    # SSA join for if statement (line 881)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 875)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA branch for the else part of an if statement (line 873)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Call to a Name (line 886):
    
    # Assigning a Call to a Name (line 886):
    
    # Call to int(...): (line 886)
    # Processing the call arguments (line 886)
    # Getting the type of 'eps_or_k' (line 886)
    eps_or_k_21871 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 886, 20), 'eps_or_k', False)
    # Processing the call keyword arguments (line 886)
    kwargs_21872 = {}
    # Getting the type of 'int' (line 886)
    int_21870 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 886, 16), 'int', False)
    # Calling int(args, kwargs) (line 886)
    int_call_result_21873 = invoke(stypy.reporting.localization.Localization(__file__, 886, 16), int_21870, *[eps_or_k_21871], **kwargs_21872)
    
    # Assigning a type to the variable 'k' (line 886)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 886, 12), 'k', int_call_result_21873)
    
    
    # Getting the type of 'k' (line 887)
    k_21874 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 887, 15), 'k')
    
    # Call to min(...): (line 887)
    # Processing the call arguments (line 887)
    # Getting the type of 'A' (line 887)
    A_21876 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 887, 23), 'A', False)
    # Obtaining the member 'shape' of a type (line 887)
    shape_21877 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 887, 23), A_21876, 'shape')
    # Processing the call keyword arguments (line 887)
    kwargs_21878 = {}
    # Getting the type of 'min' (line 887)
    min_21875 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 887, 19), 'min', False)
    # Calling min(args, kwargs) (line 887)
    min_call_result_21879 = invoke(stypy.reporting.localization.Localization(__file__, 887, 19), min_21875, *[shape_21877], **kwargs_21878)
    
    # Applying the binary operator '>' (line 887)
    result_gt_21880 = python_operator(stypy.reporting.localization.Localization(__file__, 887, 15), '>', k_21874, min_call_result_21879)
    
    # Testing the type of an if condition (line 887)
    if_condition_21881 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 887, 12), result_gt_21880)
    # Assigning a type to the variable 'if_condition_21881' (line 887)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 887, 12), 'if_condition_21881', if_condition_21881)
    # SSA begins for if statement (line 887)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 888)
    # Processing the call arguments (line 888)
    str_21883 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 888, 33), 'str', 'Approximation rank %s exceeds min(A.shape) =  %s ')
    
    # Obtaining an instance of the builtin type 'tuple' (line 889)
    tuple_21884 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 889, 43), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 889)
    # Adding element type (line 889)
    # Getting the type of 'k' (line 889)
    k_21885 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 889, 43), 'k', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 889, 43), tuple_21884, k_21885)
    # Adding element type (line 889)
    
    # Call to min(...): (line 889)
    # Processing the call arguments (line 889)
    # Getting the type of 'A' (line 889)
    A_21887 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 889, 50), 'A', False)
    # Obtaining the member 'shape' of a type (line 889)
    shape_21888 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 889, 50), A_21887, 'shape')
    # Processing the call keyword arguments (line 889)
    kwargs_21889 = {}
    # Getting the type of 'min' (line 889)
    min_21886 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 889, 46), 'min', False)
    # Calling min(args, kwargs) (line 889)
    min_call_result_21890 = invoke(stypy.reporting.localization.Localization(__file__, 889, 46), min_21886, *[shape_21888], **kwargs_21889)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 889, 43), tuple_21884, min_call_result_21890)
    
    # Applying the binary operator '%' (line 888)
    result_mod_21891 = python_operator(stypy.reporting.localization.Localization(__file__, 888, 33), '%', str_21883, tuple_21884)
    
    # Processing the call keyword arguments (line 888)
    kwargs_21892 = {}
    # Getting the type of 'ValueError' (line 888)
    ValueError_21882 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 888, 22), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 888)
    ValueError_call_result_21893 = invoke(stypy.reporting.localization.Localization(__file__, 888, 22), ValueError_21882, *[result_mod_21891], **kwargs_21892)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 888, 16), ValueError_call_result_21893, 'raise parameter', BaseException)
    # SSA join for if statement (line 887)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Getting the type of 'rand' (line 890)
    rand_21894 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 890, 15), 'rand')
    # Testing the type of an if condition (line 890)
    if_condition_21895 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 890, 12), rand_21894)
    # Assigning a type to the variable 'if_condition_21895' (line 890)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 890, 12), 'if_condition_21895', if_condition_21895)
    # SSA begins for if statement (line 890)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Getting the type of 'real' (line 891)
    real_21896 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 891, 19), 'real')
    # Testing the type of an if condition (line 891)
    if_condition_21897 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 891, 16), real_21896)
    # Assigning a type to the variable 'if_condition_21897' (line 891)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 891, 16), 'if_condition_21897', if_condition_21897)
    # SSA begins for if statement (line 891)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Tuple (line 892):
    
    # Assigning a Subscript to a Name (line 892):
    
    # Obtaining the type of the subscript
    int_21898 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 892, 20), 'int')
    
    # Call to iddr_asvd(...): (line 892)
    # Processing the call arguments (line 892)
    # Getting the type of 'A' (line 892)
    A_21901 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 892, 48), 'A', False)
    # Getting the type of 'k' (line 892)
    k_21902 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 892, 51), 'k', False)
    # Processing the call keyword arguments (line 892)
    kwargs_21903 = {}
    # Getting the type of 'backend' (line 892)
    backend_21899 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 892, 30), 'backend', False)
    # Obtaining the member 'iddr_asvd' of a type (line 892)
    iddr_asvd_21900 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 892, 30), backend_21899, 'iddr_asvd')
    # Calling iddr_asvd(args, kwargs) (line 892)
    iddr_asvd_call_result_21904 = invoke(stypy.reporting.localization.Localization(__file__, 892, 30), iddr_asvd_21900, *[A_21901, k_21902], **kwargs_21903)
    
    # Obtaining the member '__getitem__' of a type (line 892)
    getitem___21905 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 892, 20), iddr_asvd_call_result_21904, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 892)
    subscript_call_result_21906 = invoke(stypy.reporting.localization.Localization(__file__, 892, 20), getitem___21905, int_21898)
    
    # Assigning a type to the variable 'tuple_var_assignment_20844' (line 892)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 892, 20), 'tuple_var_assignment_20844', subscript_call_result_21906)
    
    # Assigning a Subscript to a Name (line 892):
    
    # Obtaining the type of the subscript
    int_21907 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 892, 20), 'int')
    
    # Call to iddr_asvd(...): (line 892)
    # Processing the call arguments (line 892)
    # Getting the type of 'A' (line 892)
    A_21910 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 892, 48), 'A', False)
    # Getting the type of 'k' (line 892)
    k_21911 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 892, 51), 'k', False)
    # Processing the call keyword arguments (line 892)
    kwargs_21912 = {}
    # Getting the type of 'backend' (line 892)
    backend_21908 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 892, 30), 'backend', False)
    # Obtaining the member 'iddr_asvd' of a type (line 892)
    iddr_asvd_21909 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 892, 30), backend_21908, 'iddr_asvd')
    # Calling iddr_asvd(args, kwargs) (line 892)
    iddr_asvd_call_result_21913 = invoke(stypy.reporting.localization.Localization(__file__, 892, 30), iddr_asvd_21909, *[A_21910, k_21911], **kwargs_21912)
    
    # Obtaining the member '__getitem__' of a type (line 892)
    getitem___21914 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 892, 20), iddr_asvd_call_result_21913, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 892)
    subscript_call_result_21915 = invoke(stypy.reporting.localization.Localization(__file__, 892, 20), getitem___21914, int_21907)
    
    # Assigning a type to the variable 'tuple_var_assignment_20845' (line 892)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 892, 20), 'tuple_var_assignment_20845', subscript_call_result_21915)
    
    # Assigning a Subscript to a Name (line 892):
    
    # Obtaining the type of the subscript
    int_21916 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 892, 20), 'int')
    
    # Call to iddr_asvd(...): (line 892)
    # Processing the call arguments (line 892)
    # Getting the type of 'A' (line 892)
    A_21919 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 892, 48), 'A', False)
    # Getting the type of 'k' (line 892)
    k_21920 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 892, 51), 'k', False)
    # Processing the call keyword arguments (line 892)
    kwargs_21921 = {}
    # Getting the type of 'backend' (line 892)
    backend_21917 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 892, 30), 'backend', False)
    # Obtaining the member 'iddr_asvd' of a type (line 892)
    iddr_asvd_21918 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 892, 30), backend_21917, 'iddr_asvd')
    # Calling iddr_asvd(args, kwargs) (line 892)
    iddr_asvd_call_result_21922 = invoke(stypy.reporting.localization.Localization(__file__, 892, 30), iddr_asvd_21918, *[A_21919, k_21920], **kwargs_21921)
    
    # Obtaining the member '__getitem__' of a type (line 892)
    getitem___21923 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 892, 20), iddr_asvd_call_result_21922, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 892)
    subscript_call_result_21924 = invoke(stypy.reporting.localization.Localization(__file__, 892, 20), getitem___21923, int_21916)
    
    # Assigning a type to the variable 'tuple_var_assignment_20846' (line 892)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 892, 20), 'tuple_var_assignment_20846', subscript_call_result_21924)
    
    # Assigning a Name to a Name (line 892):
    # Getting the type of 'tuple_var_assignment_20844' (line 892)
    tuple_var_assignment_20844_21925 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 892, 20), 'tuple_var_assignment_20844')
    # Assigning a type to the variable 'U' (line 892)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 892, 20), 'U', tuple_var_assignment_20844_21925)
    
    # Assigning a Name to a Name (line 892):
    # Getting the type of 'tuple_var_assignment_20845' (line 892)
    tuple_var_assignment_20845_21926 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 892, 20), 'tuple_var_assignment_20845')
    # Assigning a type to the variable 'V' (line 892)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 892, 23), 'V', tuple_var_assignment_20845_21926)
    
    # Assigning a Name to a Name (line 892):
    # Getting the type of 'tuple_var_assignment_20846' (line 892)
    tuple_var_assignment_20846_21927 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 892, 20), 'tuple_var_assignment_20846')
    # Assigning a type to the variable 'S' (line 892)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 892, 26), 'S', tuple_var_assignment_20846_21927)
    # SSA branch for the else part of an if statement (line 891)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Call to a Tuple (line 894):
    
    # Assigning a Subscript to a Name (line 894):
    
    # Obtaining the type of the subscript
    int_21928 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 894, 20), 'int')
    
    # Call to idzr_asvd(...): (line 894)
    # Processing the call arguments (line 894)
    # Getting the type of 'A' (line 894)
    A_21931 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 894, 48), 'A', False)
    # Getting the type of 'k' (line 894)
    k_21932 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 894, 51), 'k', False)
    # Processing the call keyword arguments (line 894)
    kwargs_21933 = {}
    # Getting the type of 'backend' (line 894)
    backend_21929 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 894, 30), 'backend', False)
    # Obtaining the member 'idzr_asvd' of a type (line 894)
    idzr_asvd_21930 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 894, 30), backend_21929, 'idzr_asvd')
    # Calling idzr_asvd(args, kwargs) (line 894)
    idzr_asvd_call_result_21934 = invoke(stypy.reporting.localization.Localization(__file__, 894, 30), idzr_asvd_21930, *[A_21931, k_21932], **kwargs_21933)
    
    # Obtaining the member '__getitem__' of a type (line 894)
    getitem___21935 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 894, 20), idzr_asvd_call_result_21934, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 894)
    subscript_call_result_21936 = invoke(stypy.reporting.localization.Localization(__file__, 894, 20), getitem___21935, int_21928)
    
    # Assigning a type to the variable 'tuple_var_assignment_20847' (line 894)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 894, 20), 'tuple_var_assignment_20847', subscript_call_result_21936)
    
    # Assigning a Subscript to a Name (line 894):
    
    # Obtaining the type of the subscript
    int_21937 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 894, 20), 'int')
    
    # Call to idzr_asvd(...): (line 894)
    # Processing the call arguments (line 894)
    # Getting the type of 'A' (line 894)
    A_21940 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 894, 48), 'A', False)
    # Getting the type of 'k' (line 894)
    k_21941 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 894, 51), 'k', False)
    # Processing the call keyword arguments (line 894)
    kwargs_21942 = {}
    # Getting the type of 'backend' (line 894)
    backend_21938 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 894, 30), 'backend', False)
    # Obtaining the member 'idzr_asvd' of a type (line 894)
    idzr_asvd_21939 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 894, 30), backend_21938, 'idzr_asvd')
    # Calling idzr_asvd(args, kwargs) (line 894)
    idzr_asvd_call_result_21943 = invoke(stypy.reporting.localization.Localization(__file__, 894, 30), idzr_asvd_21939, *[A_21940, k_21941], **kwargs_21942)
    
    # Obtaining the member '__getitem__' of a type (line 894)
    getitem___21944 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 894, 20), idzr_asvd_call_result_21943, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 894)
    subscript_call_result_21945 = invoke(stypy.reporting.localization.Localization(__file__, 894, 20), getitem___21944, int_21937)
    
    # Assigning a type to the variable 'tuple_var_assignment_20848' (line 894)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 894, 20), 'tuple_var_assignment_20848', subscript_call_result_21945)
    
    # Assigning a Subscript to a Name (line 894):
    
    # Obtaining the type of the subscript
    int_21946 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 894, 20), 'int')
    
    # Call to idzr_asvd(...): (line 894)
    # Processing the call arguments (line 894)
    # Getting the type of 'A' (line 894)
    A_21949 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 894, 48), 'A', False)
    # Getting the type of 'k' (line 894)
    k_21950 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 894, 51), 'k', False)
    # Processing the call keyword arguments (line 894)
    kwargs_21951 = {}
    # Getting the type of 'backend' (line 894)
    backend_21947 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 894, 30), 'backend', False)
    # Obtaining the member 'idzr_asvd' of a type (line 894)
    idzr_asvd_21948 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 894, 30), backend_21947, 'idzr_asvd')
    # Calling idzr_asvd(args, kwargs) (line 894)
    idzr_asvd_call_result_21952 = invoke(stypy.reporting.localization.Localization(__file__, 894, 30), idzr_asvd_21948, *[A_21949, k_21950], **kwargs_21951)
    
    # Obtaining the member '__getitem__' of a type (line 894)
    getitem___21953 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 894, 20), idzr_asvd_call_result_21952, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 894)
    subscript_call_result_21954 = invoke(stypy.reporting.localization.Localization(__file__, 894, 20), getitem___21953, int_21946)
    
    # Assigning a type to the variable 'tuple_var_assignment_20849' (line 894)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 894, 20), 'tuple_var_assignment_20849', subscript_call_result_21954)
    
    # Assigning a Name to a Name (line 894):
    # Getting the type of 'tuple_var_assignment_20847' (line 894)
    tuple_var_assignment_20847_21955 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 894, 20), 'tuple_var_assignment_20847')
    # Assigning a type to the variable 'U' (line 894)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 894, 20), 'U', tuple_var_assignment_20847_21955)
    
    # Assigning a Name to a Name (line 894):
    # Getting the type of 'tuple_var_assignment_20848' (line 894)
    tuple_var_assignment_20848_21956 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 894, 20), 'tuple_var_assignment_20848')
    # Assigning a type to the variable 'V' (line 894)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 894, 23), 'V', tuple_var_assignment_20848_21956)
    
    # Assigning a Name to a Name (line 894):
    # Getting the type of 'tuple_var_assignment_20849' (line 894)
    tuple_var_assignment_20849_21957 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 894, 20), 'tuple_var_assignment_20849')
    # Assigning a type to the variable 'S' (line 894)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 894, 26), 'S', tuple_var_assignment_20849_21957)
    # SSA join for if statement (line 891)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA branch for the else part of an if statement (line 890)
    module_type_store.open_ssa_branch('else')
    
    # Getting the type of 'real' (line 896)
    real_21958 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 896, 19), 'real')
    # Testing the type of an if condition (line 896)
    if_condition_21959 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 896, 16), real_21958)
    # Assigning a type to the variable 'if_condition_21959' (line 896)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 896, 16), 'if_condition_21959', if_condition_21959)
    # SSA begins for if statement (line 896)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Tuple (line 897):
    
    # Assigning a Subscript to a Name (line 897):
    
    # Obtaining the type of the subscript
    int_21960 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 897, 20), 'int')
    
    # Call to iddr_svd(...): (line 897)
    # Processing the call arguments (line 897)
    # Getting the type of 'A' (line 897)
    A_21963 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 897, 47), 'A', False)
    # Getting the type of 'k' (line 897)
    k_21964 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 897, 50), 'k', False)
    # Processing the call keyword arguments (line 897)
    kwargs_21965 = {}
    # Getting the type of 'backend' (line 897)
    backend_21961 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 897, 30), 'backend', False)
    # Obtaining the member 'iddr_svd' of a type (line 897)
    iddr_svd_21962 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 897, 30), backend_21961, 'iddr_svd')
    # Calling iddr_svd(args, kwargs) (line 897)
    iddr_svd_call_result_21966 = invoke(stypy.reporting.localization.Localization(__file__, 897, 30), iddr_svd_21962, *[A_21963, k_21964], **kwargs_21965)
    
    # Obtaining the member '__getitem__' of a type (line 897)
    getitem___21967 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 897, 20), iddr_svd_call_result_21966, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 897)
    subscript_call_result_21968 = invoke(stypy.reporting.localization.Localization(__file__, 897, 20), getitem___21967, int_21960)
    
    # Assigning a type to the variable 'tuple_var_assignment_20850' (line 897)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 897, 20), 'tuple_var_assignment_20850', subscript_call_result_21968)
    
    # Assigning a Subscript to a Name (line 897):
    
    # Obtaining the type of the subscript
    int_21969 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 897, 20), 'int')
    
    # Call to iddr_svd(...): (line 897)
    # Processing the call arguments (line 897)
    # Getting the type of 'A' (line 897)
    A_21972 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 897, 47), 'A', False)
    # Getting the type of 'k' (line 897)
    k_21973 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 897, 50), 'k', False)
    # Processing the call keyword arguments (line 897)
    kwargs_21974 = {}
    # Getting the type of 'backend' (line 897)
    backend_21970 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 897, 30), 'backend', False)
    # Obtaining the member 'iddr_svd' of a type (line 897)
    iddr_svd_21971 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 897, 30), backend_21970, 'iddr_svd')
    # Calling iddr_svd(args, kwargs) (line 897)
    iddr_svd_call_result_21975 = invoke(stypy.reporting.localization.Localization(__file__, 897, 30), iddr_svd_21971, *[A_21972, k_21973], **kwargs_21974)
    
    # Obtaining the member '__getitem__' of a type (line 897)
    getitem___21976 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 897, 20), iddr_svd_call_result_21975, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 897)
    subscript_call_result_21977 = invoke(stypy.reporting.localization.Localization(__file__, 897, 20), getitem___21976, int_21969)
    
    # Assigning a type to the variable 'tuple_var_assignment_20851' (line 897)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 897, 20), 'tuple_var_assignment_20851', subscript_call_result_21977)
    
    # Assigning a Subscript to a Name (line 897):
    
    # Obtaining the type of the subscript
    int_21978 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 897, 20), 'int')
    
    # Call to iddr_svd(...): (line 897)
    # Processing the call arguments (line 897)
    # Getting the type of 'A' (line 897)
    A_21981 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 897, 47), 'A', False)
    # Getting the type of 'k' (line 897)
    k_21982 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 897, 50), 'k', False)
    # Processing the call keyword arguments (line 897)
    kwargs_21983 = {}
    # Getting the type of 'backend' (line 897)
    backend_21979 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 897, 30), 'backend', False)
    # Obtaining the member 'iddr_svd' of a type (line 897)
    iddr_svd_21980 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 897, 30), backend_21979, 'iddr_svd')
    # Calling iddr_svd(args, kwargs) (line 897)
    iddr_svd_call_result_21984 = invoke(stypy.reporting.localization.Localization(__file__, 897, 30), iddr_svd_21980, *[A_21981, k_21982], **kwargs_21983)
    
    # Obtaining the member '__getitem__' of a type (line 897)
    getitem___21985 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 897, 20), iddr_svd_call_result_21984, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 897)
    subscript_call_result_21986 = invoke(stypy.reporting.localization.Localization(__file__, 897, 20), getitem___21985, int_21978)
    
    # Assigning a type to the variable 'tuple_var_assignment_20852' (line 897)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 897, 20), 'tuple_var_assignment_20852', subscript_call_result_21986)
    
    # Assigning a Name to a Name (line 897):
    # Getting the type of 'tuple_var_assignment_20850' (line 897)
    tuple_var_assignment_20850_21987 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 897, 20), 'tuple_var_assignment_20850')
    # Assigning a type to the variable 'U' (line 897)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 897, 20), 'U', tuple_var_assignment_20850_21987)
    
    # Assigning a Name to a Name (line 897):
    # Getting the type of 'tuple_var_assignment_20851' (line 897)
    tuple_var_assignment_20851_21988 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 897, 20), 'tuple_var_assignment_20851')
    # Assigning a type to the variable 'V' (line 897)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 897, 23), 'V', tuple_var_assignment_20851_21988)
    
    # Assigning a Name to a Name (line 897):
    # Getting the type of 'tuple_var_assignment_20852' (line 897)
    tuple_var_assignment_20852_21989 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 897, 20), 'tuple_var_assignment_20852')
    # Assigning a type to the variable 'S' (line 897)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 897, 26), 'S', tuple_var_assignment_20852_21989)
    # SSA branch for the else part of an if statement (line 896)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Call to a Tuple (line 899):
    
    # Assigning a Subscript to a Name (line 899):
    
    # Obtaining the type of the subscript
    int_21990 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 899, 20), 'int')
    
    # Call to idzr_svd(...): (line 899)
    # Processing the call arguments (line 899)
    # Getting the type of 'A' (line 899)
    A_21993 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 899, 47), 'A', False)
    # Getting the type of 'k' (line 899)
    k_21994 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 899, 50), 'k', False)
    # Processing the call keyword arguments (line 899)
    kwargs_21995 = {}
    # Getting the type of 'backend' (line 899)
    backend_21991 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 899, 30), 'backend', False)
    # Obtaining the member 'idzr_svd' of a type (line 899)
    idzr_svd_21992 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 899, 30), backend_21991, 'idzr_svd')
    # Calling idzr_svd(args, kwargs) (line 899)
    idzr_svd_call_result_21996 = invoke(stypy.reporting.localization.Localization(__file__, 899, 30), idzr_svd_21992, *[A_21993, k_21994], **kwargs_21995)
    
    # Obtaining the member '__getitem__' of a type (line 899)
    getitem___21997 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 899, 20), idzr_svd_call_result_21996, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 899)
    subscript_call_result_21998 = invoke(stypy.reporting.localization.Localization(__file__, 899, 20), getitem___21997, int_21990)
    
    # Assigning a type to the variable 'tuple_var_assignment_20853' (line 899)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 899, 20), 'tuple_var_assignment_20853', subscript_call_result_21998)
    
    # Assigning a Subscript to a Name (line 899):
    
    # Obtaining the type of the subscript
    int_21999 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 899, 20), 'int')
    
    # Call to idzr_svd(...): (line 899)
    # Processing the call arguments (line 899)
    # Getting the type of 'A' (line 899)
    A_22002 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 899, 47), 'A', False)
    # Getting the type of 'k' (line 899)
    k_22003 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 899, 50), 'k', False)
    # Processing the call keyword arguments (line 899)
    kwargs_22004 = {}
    # Getting the type of 'backend' (line 899)
    backend_22000 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 899, 30), 'backend', False)
    # Obtaining the member 'idzr_svd' of a type (line 899)
    idzr_svd_22001 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 899, 30), backend_22000, 'idzr_svd')
    # Calling idzr_svd(args, kwargs) (line 899)
    idzr_svd_call_result_22005 = invoke(stypy.reporting.localization.Localization(__file__, 899, 30), idzr_svd_22001, *[A_22002, k_22003], **kwargs_22004)
    
    # Obtaining the member '__getitem__' of a type (line 899)
    getitem___22006 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 899, 20), idzr_svd_call_result_22005, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 899)
    subscript_call_result_22007 = invoke(stypy.reporting.localization.Localization(__file__, 899, 20), getitem___22006, int_21999)
    
    # Assigning a type to the variable 'tuple_var_assignment_20854' (line 899)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 899, 20), 'tuple_var_assignment_20854', subscript_call_result_22007)
    
    # Assigning a Subscript to a Name (line 899):
    
    # Obtaining the type of the subscript
    int_22008 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 899, 20), 'int')
    
    # Call to idzr_svd(...): (line 899)
    # Processing the call arguments (line 899)
    # Getting the type of 'A' (line 899)
    A_22011 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 899, 47), 'A', False)
    # Getting the type of 'k' (line 899)
    k_22012 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 899, 50), 'k', False)
    # Processing the call keyword arguments (line 899)
    kwargs_22013 = {}
    # Getting the type of 'backend' (line 899)
    backend_22009 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 899, 30), 'backend', False)
    # Obtaining the member 'idzr_svd' of a type (line 899)
    idzr_svd_22010 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 899, 30), backend_22009, 'idzr_svd')
    # Calling idzr_svd(args, kwargs) (line 899)
    idzr_svd_call_result_22014 = invoke(stypy.reporting.localization.Localization(__file__, 899, 30), idzr_svd_22010, *[A_22011, k_22012], **kwargs_22013)
    
    # Obtaining the member '__getitem__' of a type (line 899)
    getitem___22015 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 899, 20), idzr_svd_call_result_22014, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 899)
    subscript_call_result_22016 = invoke(stypy.reporting.localization.Localization(__file__, 899, 20), getitem___22015, int_22008)
    
    # Assigning a type to the variable 'tuple_var_assignment_20855' (line 899)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 899, 20), 'tuple_var_assignment_20855', subscript_call_result_22016)
    
    # Assigning a Name to a Name (line 899):
    # Getting the type of 'tuple_var_assignment_20853' (line 899)
    tuple_var_assignment_20853_22017 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 899, 20), 'tuple_var_assignment_20853')
    # Assigning a type to the variable 'U' (line 899)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 899, 20), 'U', tuple_var_assignment_20853_22017)
    
    # Assigning a Name to a Name (line 899):
    # Getting the type of 'tuple_var_assignment_20854' (line 899)
    tuple_var_assignment_20854_22018 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 899, 20), 'tuple_var_assignment_20854')
    # Assigning a type to the variable 'V' (line 899)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 899, 23), 'V', tuple_var_assignment_20854_22018)
    
    # Assigning a Name to a Name (line 899):
    # Getting the type of 'tuple_var_assignment_20855' (line 899)
    tuple_var_assignment_20855_22019 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 899, 20), 'tuple_var_assignment_20855')
    # Assigning a type to the variable 'S' (line 899)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 899, 26), 'S', tuple_var_assignment_20855_22019)
    # SSA join for if statement (line 896)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 890)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 873)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA branch for the else part of an if statement (line 872)
    module_type_store.open_ssa_branch('else')
    
    
    # Call to isinstance(...): (line 900)
    # Processing the call arguments (line 900)
    # Getting the type of 'A' (line 900)
    A_22021 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 900, 20), 'A', False)
    # Getting the type of 'LinearOperator' (line 900)
    LinearOperator_22022 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 900, 23), 'LinearOperator', False)
    # Processing the call keyword arguments (line 900)
    kwargs_22023 = {}
    # Getting the type of 'isinstance' (line 900)
    isinstance_22020 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 900, 9), 'isinstance', False)
    # Calling isinstance(args, kwargs) (line 900)
    isinstance_call_result_22024 = invoke(stypy.reporting.localization.Localization(__file__, 900, 9), isinstance_22020, *[A_22021, LinearOperator_22022], **kwargs_22023)
    
    # Testing the type of an if condition (line 900)
    if_condition_22025 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 900, 9), isinstance_call_result_22024)
    # Assigning a type to the variable 'if_condition_22025' (line 900)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 900, 9), 'if_condition_22025', if_condition_22025)
    # SSA begins for if statement (line 900)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Attribute to a Tuple (line 901):
    
    # Assigning a Subscript to a Name (line 901):
    
    # Obtaining the type of the subscript
    int_22026 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 901, 8), 'int')
    # Getting the type of 'A' (line 901)
    A_22027 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 901, 15), 'A')
    # Obtaining the member 'shape' of a type (line 901)
    shape_22028 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 901, 15), A_22027, 'shape')
    # Obtaining the member '__getitem__' of a type (line 901)
    getitem___22029 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 901, 8), shape_22028, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 901)
    subscript_call_result_22030 = invoke(stypy.reporting.localization.Localization(__file__, 901, 8), getitem___22029, int_22026)
    
    # Assigning a type to the variable 'tuple_var_assignment_20856' (line 901)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 901, 8), 'tuple_var_assignment_20856', subscript_call_result_22030)
    
    # Assigning a Subscript to a Name (line 901):
    
    # Obtaining the type of the subscript
    int_22031 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 901, 8), 'int')
    # Getting the type of 'A' (line 901)
    A_22032 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 901, 15), 'A')
    # Obtaining the member 'shape' of a type (line 901)
    shape_22033 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 901, 15), A_22032, 'shape')
    # Obtaining the member '__getitem__' of a type (line 901)
    getitem___22034 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 901, 8), shape_22033, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 901)
    subscript_call_result_22035 = invoke(stypy.reporting.localization.Localization(__file__, 901, 8), getitem___22034, int_22031)
    
    # Assigning a type to the variable 'tuple_var_assignment_20857' (line 901)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 901, 8), 'tuple_var_assignment_20857', subscript_call_result_22035)
    
    # Assigning a Name to a Name (line 901):
    # Getting the type of 'tuple_var_assignment_20856' (line 901)
    tuple_var_assignment_20856_22036 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 901, 8), 'tuple_var_assignment_20856')
    # Assigning a type to the variable 'm' (line 901)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 901, 8), 'm', tuple_var_assignment_20856_22036)
    
    # Assigning a Name to a Name (line 901):
    # Getting the type of 'tuple_var_assignment_20857' (line 901)
    tuple_var_assignment_20857_22037 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 901, 8), 'tuple_var_assignment_20857')
    # Assigning a type to the variable 'n' (line 901)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 901, 11), 'n', tuple_var_assignment_20857_22037)
    
    # Assigning a Lambda to a Name (line 902):
    
    # Assigning a Lambda to a Name (line 902):

    @norecursion
    def _stypy_temp_lambda_14(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_stypy_temp_lambda_14'
        module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_14', 902, 17, True)
        # Passed parameters checking function
        _stypy_temp_lambda_14.stypy_localization = localization
        _stypy_temp_lambda_14.stypy_type_of_self = None
        _stypy_temp_lambda_14.stypy_type_store = module_type_store
        _stypy_temp_lambda_14.stypy_function_name = '_stypy_temp_lambda_14'
        _stypy_temp_lambda_14.stypy_param_names_list = ['x']
        _stypy_temp_lambda_14.stypy_varargs_param_name = None
        _stypy_temp_lambda_14.stypy_kwargs_param_name = None
        _stypy_temp_lambda_14.stypy_call_defaults = defaults
        _stypy_temp_lambda_14.stypy_call_varargs = varargs
        _stypy_temp_lambda_14.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_14', ['x'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Stacktrace push for error reporting
        localization.set_stack_trace('_stypy_temp_lambda_14', ['x'], arguments)
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of the lambda function code ##################

        
        # Call to matvec(...): (line 902)
        # Processing the call arguments (line 902)
        # Getting the type of 'x' (line 902)
        x_22040 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 902, 36), 'x', False)
        # Processing the call keyword arguments (line 902)
        kwargs_22041 = {}
        # Getting the type of 'A' (line 902)
        A_22038 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 902, 27), 'A', False)
        # Obtaining the member 'matvec' of a type (line 902)
        matvec_22039 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 902, 27), A_22038, 'matvec')
        # Calling matvec(args, kwargs) (line 902)
        matvec_call_result_22042 = invoke(stypy.reporting.localization.Localization(__file__, 902, 27), matvec_22039, *[x_22040], **kwargs_22041)
        
        # Assigning the return type of the lambda function
        # Assigning a type to the variable 'stypy_return_type' (line 902)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 902, 17), 'stypy_return_type', matvec_call_result_22042)
        
        # ################# End of the lambda function code ##################

        # Stacktrace pop (error reporting)
        localization.unset_stack_trace()
        
        # Storing the return type of function '_stypy_temp_lambda_14' in the type store
        # Getting the type of 'stypy_return_type' (line 902)
        stypy_return_type_22043 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 902, 17), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_22043)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_stypy_temp_lambda_14'
        return stypy_return_type_22043

    # Assigning a type to the variable '_stypy_temp_lambda_14' (line 902)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 902, 17), '_stypy_temp_lambda_14', _stypy_temp_lambda_14)
    # Getting the type of '_stypy_temp_lambda_14' (line 902)
    _stypy_temp_lambda_14_22044 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 902, 17), '_stypy_temp_lambda_14')
    # Assigning a type to the variable 'matvec' (line 902)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 902, 8), 'matvec', _stypy_temp_lambda_14_22044)
    
    # Assigning a Lambda to a Name (line 903):
    
    # Assigning a Lambda to a Name (line 903):

    @norecursion
    def _stypy_temp_lambda_15(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_stypy_temp_lambda_15'
        module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_15', 903, 18, True)
        # Passed parameters checking function
        _stypy_temp_lambda_15.stypy_localization = localization
        _stypy_temp_lambda_15.stypy_type_of_self = None
        _stypy_temp_lambda_15.stypy_type_store = module_type_store
        _stypy_temp_lambda_15.stypy_function_name = '_stypy_temp_lambda_15'
        _stypy_temp_lambda_15.stypy_param_names_list = ['x']
        _stypy_temp_lambda_15.stypy_varargs_param_name = None
        _stypy_temp_lambda_15.stypy_kwargs_param_name = None
        _stypy_temp_lambda_15.stypy_call_defaults = defaults
        _stypy_temp_lambda_15.stypy_call_varargs = varargs
        _stypy_temp_lambda_15.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_15', ['x'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Stacktrace push for error reporting
        localization.set_stack_trace('_stypy_temp_lambda_15', ['x'], arguments)
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of the lambda function code ##################

        
        # Call to rmatvec(...): (line 903)
        # Processing the call arguments (line 903)
        # Getting the type of 'x' (line 903)
        x_22047 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 903, 38), 'x', False)
        # Processing the call keyword arguments (line 903)
        kwargs_22048 = {}
        # Getting the type of 'A' (line 903)
        A_22045 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 903, 28), 'A', False)
        # Obtaining the member 'rmatvec' of a type (line 903)
        rmatvec_22046 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 903, 28), A_22045, 'rmatvec')
        # Calling rmatvec(args, kwargs) (line 903)
        rmatvec_call_result_22049 = invoke(stypy.reporting.localization.Localization(__file__, 903, 28), rmatvec_22046, *[x_22047], **kwargs_22048)
        
        # Assigning the return type of the lambda function
        # Assigning a type to the variable 'stypy_return_type' (line 903)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 903, 18), 'stypy_return_type', rmatvec_call_result_22049)
        
        # ################# End of the lambda function code ##################

        # Stacktrace pop (error reporting)
        localization.unset_stack_trace()
        
        # Storing the return type of function '_stypy_temp_lambda_15' in the type store
        # Getting the type of 'stypy_return_type' (line 903)
        stypy_return_type_22050 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 903, 18), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_22050)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_stypy_temp_lambda_15'
        return stypy_return_type_22050

    # Assigning a type to the variable '_stypy_temp_lambda_15' (line 903)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 903, 18), '_stypy_temp_lambda_15', _stypy_temp_lambda_15)
    # Getting the type of '_stypy_temp_lambda_15' (line 903)
    _stypy_temp_lambda_15_22051 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 903, 18), '_stypy_temp_lambda_15')
    # Assigning a type to the variable 'matveca' (line 903)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 903, 8), 'matveca', _stypy_temp_lambda_15_22051)
    
    
    # Getting the type of 'eps_or_k' (line 904)
    eps_or_k_22052 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 904, 11), 'eps_or_k')
    int_22053 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 904, 22), 'int')
    # Applying the binary operator '<' (line 904)
    result_lt_22054 = python_operator(stypy.reporting.localization.Localization(__file__, 904, 11), '<', eps_or_k_22052, int_22053)
    
    # Testing the type of an if condition (line 904)
    if_condition_22055 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 904, 8), result_lt_22054)
    # Assigning a type to the variable 'if_condition_22055' (line 904)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 904, 8), 'if_condition_22055', if_condition_22055)
    # SSA begins for if statement (line 904)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Name to a Name (line 905):
    
    # Assigning a Name to a Name (line 905):
    # Getting the type of 'eps_or_k' (line 905)
    eps_or_k_22056 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 905, 18), 'eps_or_k')
    # Assigning a type to the variable 'eps' (line 905)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 905, 12), 'eps', eps_or_k_22056)
    
    # Getting the type of 'real' (line 906)
    real_22057 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 906, 15), 'real')
    # Testing the type of an if condition (line 906)
    if_condition_22058 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 906, 12), real_22057)
    # Assigning a type to the variable 'if_condition_22058' (line 906)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 906, 12), 'if_condition_22058', if_condition_22058)
    # SSA begins for if statement (line 906)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Tuple (line 907):
    
    # Assigning a Subscript to a Name (line 907):
    
    # Obtaining the type of the subscript
    int_22059 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 907, 16), 'int')
    
    # Call to iddp_rsvd(...): (line 907)
    # Processing the call arguments (line 907)
    # Getting the type of 'eps' (line 907)
    eps_22062 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 907, 44), 'eps', False)
    # Getting the type of 'm' (line 907)
    m_22063 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 907, 49), 'm', False)
    # Getting the type of 'n' (line 907)
    n_22064 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 907, 52), 'n', False)
    # Getting the type of 'matveca' (line 907)
    matveca_22065 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 907, 55), 'matveca', False)
    # Getting the type of 'matvec' (line 907)
    matvec_22066 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 907, 64), 'matvec', False)
    # Processing the call keyword arguments (line 907)
    kwargs_22067 = {}
    # Getting the type of 'backend' (line 907)
    backend_22060 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 907, 26), 'backend', False)
    # Obtaining the member 'iddp_rsvd' of a type (line 907)
    iddp_rsvd_22061 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 907, 26), backend_22060, 'iddp_rsvd')
    # Calling iddp_rsvd(args, kwargs) (line 907)
    iddp_rsvd_call_result_22068 = invoke(stypy.reporting.localization.Localization(__file__, 907, 26), iddp_rsvd_22061, *[eps_22062, m_22063, n_22064, matveca_22065, matvec_22066], **kwargs_22067)
    
    # Obtaining the member '__getitem__' of a type (line 907)
    getitem___22069 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 907, 16), iddp_rsvd_call_result_22068, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 907)
    subscript_call_result_22070 = invoke(stypy.reporting.localization.Localization(__file__, 907, 16), getitem___22069, int_22059)
    
    # Assigning a type to the variable 'tuple_var_assignment_20858' (line 907)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 907, 16), 'tuple_var_assignment_20858', subscript_call_result_22070)
    
    # Assigning a Subscript to a Name (line 907):
    
    # Obtaining the type of the subscript
    int_22071 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 907, 16), 'int')
    
    # Call to iddp_rsvd(...): (line 907)
    # Processing the call arguments (line 907)
    # Getting the type of 'eps' (line 907)
    eps_22074 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 907, 44), 'eps', False)
    # Getting the type of 'm' (line 907)
    m_22075 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 907, 49), 'm', False)
    # Getting the type of 'n' (line 907)
    n_22076 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 907, 52), 'n', False)
    # Getting the type of 'matveca' (line 907)
    matveca_22077 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 907, 55), 'matveca', False)
    # Getting the type of 'matvec' (line 907)
    matvec_22078 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 907, 64), 'matvec', False)
    # Processing the call keyword arguments (line 907)
    kwargs_22079 = {}
    # Getting the type of 'backend' (line 907)
    backend_22072 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 907, 26), 'backend', False)
    # Obtaining the member 'iddp_rsvd' of a type (line 907)
    iddp_rsvd_22073 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 907, 26), backend_22072, 'iddp_rsvd')
    # Calling iddp_rsvd(args, kwargs) (line 907)
    iddp_rsvd_call_result_22080 = invoke(stypy.reporting.localization.Localization(__file__, 907, 26), iddp_rsvd_22073, *[eps_22074, m_22075, n_22076, matveca_22077, matvec_22078], **kwargs_22079)
    
    # Obtaining the member '__getitem__' of a type (line 907)
    getitem___22081 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 907, 16), iddp_rsvd_call_result_22080, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 907)
    subscript_call_result_22082 = invoke(stypy.reporting.localization.Localization(__file__, 907, 16), getitem___22081, int_22071)
    
    # Assigning a type to the variable 'tuple_var_assignment_20859' (line 907)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 907, 16), 'tuple_var_assignment_20859', subscript_call_result_22082)
    
    # Assigning a Subscript to a Name (line 907):
    
    # Obtaining the type of the subscript
    int_22083 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 907, 16), 'int')
    
    # Call to iddp_rsvd(...): (line 907)
    # Processing the call arguments (line 907)
    # Getting the type of 'eps' (line 907)
    eps_22086 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 907, 44), 'eps', False)
    # Getting the type of 'm' (line 907)
    m_22087 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 907, 49), 'm', False)
    # Getting the type of 'n' (line 907)
    n_22088 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 907, 52), 'n', False)
    # Getting the type of 'matveca' (line 907)
    matveca_22089 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 907, 55), 'matveca', False)
    # Getting the type of 'matvec' (line 907)
    matvec_22090 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 907, 64), 'matvec', False)
    # Processing the call keyword arguments (line 907)
    kwargs_22091 = {}
    # Getting the type of 'backend' (line 907)
    backend_22084 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 907, 26), 'backend', False)
    # Obtaining the member 'iddp_rsvd' of a type (line 907)
    iddp_rsvd_22085 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 907, 26), backend_22084, 'iddp_rsvd')
    # Calling iddp_rsvd(args, kwargs) (line 907)
    iddp_rsvd_call_result_22092 = invoke(stypy.reporting.localization.Localization(__file__, 907, 26), iddp_rsvd_22085, *[eps_22086, m_22087, n_22088, matveca_22089, matvec_22090], **kwargs_22091)
    
    # Obtaining the member '__getitem__' of a type (line 907)
    getitem___22093 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 907, 16), iddp_rsvd_call_result_22092, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 907)
    subscript_call_result_22094 = invoke(stypy.reporting.localization.Localization(__file__, 907, 16), getitem___22093, int_22083)
    
    # Assigning a type to the variable 'tuple_var_assignment_20860' (line 907)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 907, 16), 'tuple_var_assignment_20860', subscript_call_result_22094)
    
    # Assigning a Name to a Name (line 907):
    # Getting the type of 'tuple_var_assignment_20858' (line 907)
    tuple_var_assignment_20858_22095 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 907, 16), 'tuple_var_assignment_20858')
    # Assigning a type to the variable 'U' (line 907)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 907, 16), 'U', tuple_var_assignment_20858_22095)
    
    # Assigning a Name to a Name (line 907):
    # Getting the type of 'tuple_var_assignment_20859' (line 907)
    tuple_var_assignment_20859_22096 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 907, 16), 'tuple_var_assignment_20859')
    # Assigning a type to the variable 'V' (line 907)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 907, 19), 'V', tuple_var_assignment_20859_22096)
    
    # Assigning a Name to a Name (line 907):
    # Getting the type of 'tuple_var_assignment_20860' (line 907)
    tuple_var_assignment_20860_22097 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 907, 16), 'tuple_var_assignment_20860')
    # Assigning a type to the variable 'S' (line 907)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 907, 22), 'S', tuple_var_assignment_20860_22097)
    # SSA branch for the else part of an if statement (line 906)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Call to a Tuple (line 909):
    
    # Assigning a Subscript to a Name (line 909):
    
    # Obtaining the type of the subscript
    int_22098 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 909, 16), 'int')
    
    # Call to idzp_rsvd(...): (line 909)
    # Processing the call arguments (line 909)
    # Getting the type of 'eps' (line 909)
    eps_22101 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 909, 44), 'eps', False)
    # Getting the type of 'm' (line 909)
    m_22102 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 909, 49), 'm', False)
    # Getting the type of 'n' (line 909)
    n_22103 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 909, 52), 'n', False)
    # Getting the type of 'matveca' (line 909)
    matveca_22104 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 909, 55), 'matveca', False)
    # Getting the type of 'matvec' (line 909)
    matvec_22105 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 909, 64), 'matvec', False)
    # Processing the call keyword arguments (line 909)
    kwargs_22106 = {}
    # Getting the type of 'backend' (line 909)
    backend_22099 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 909, 26), 'backend', False)
    # Obtaining the member 'idzp_rsvd' of a type (line 909)
    idzp_rsvd_22100 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 909, 26), backend_22099, 'idzp_rsvd')
    # Calling idzp_rsvd(args, kwargs) (line 909)
    idzp_rsvd_call_result_22107 = invoke(stypy.reporting.localization.Localization(__file__, 909, 26), idzp_rsvd_22100, *[eps_22101, m_22102, n_22103, matveca_22104, matvec_22105], **kwargs_22106)
    
    # Obtaining the member '__getitem__' of a type (line 909)
    getitem___22108 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 909, 16), idzp_rsvd_call_result_22107, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 909)
    subscript_call_result_22109 = invoke(stypy.reporting.localization.Localization(__file__, 909, 16), getitem___22108, int_22098)
    
    # Assigning a type to the variable 'tuple_var_assignment_20861' (line 909)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 909, 16), 'tuple_var_assignment_20861', subscript_call_result_22109)
    
    # Assigning a Subscript to a Name (line 909):
    
    # Obtaining the type of the subscript
    int_22110 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 909, 16), 'int')
    
    # Call to idzp_rsvd(...): (line 909)
    # Processing the call arguments (line 909)
    # Getting the type of 'eps' (line 909)
    eps_22113 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 909, 44), 'eps', False)
    # Getting the type of 'm' (line 909)
    m_22114 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 909, 49), 'm', False)
    # Getting the type of 'n' (line 909)
    n_22115 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 909, 52), 'n', False)
    # Getting the type of 'matveca' (line 909)
    matveca_22116 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 909, 55), 'matveca', False)
    # Getting the type of 'matvec' (line 909)
    matvec_22117 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 909, 64), 'matvec', False)
    # Processing the call keyword arguments (line 909)
    kwargs_22118 = {}
    # Getting the type of 'backend' (line 909)
    backend_22111 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 909, 26), 'backend', False)
    # Obtaining the member 'idzp_rsvd' of a type (line 909)
    idzp_rsvd_22112 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 909, 26), backend_22111, 'idzp_rsvd')
    # Calling idzp_rsvd(args, kwargs) (line 909)
    idzp_rsvd_call_result_22119 = invoke(stypy.reporting.localization.Localization(__file__, 909, 26), idzp_rsvd_22112, *[eps_22113, m_22114, n_22115, matveca_22116, matvec_22117], **kwargs_22118)
    
    # Obtaining the member '__getitem__' of a type (line 909)
    getitem___22120 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 909, 16), idzp_rsvd_call_result_22119, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 909)
    subscript_call_result_22121 = invoke(stypy.reporting.localization.Localization(__file__, 909, 16), getitem___22120, int_22110)
    
    # Assigning a type to the variable 'tuple_var_assignment_20862' (line 909)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 909, 16), 'tuple_var_assignment_20862', subscript_call_result_22121)
    
    # Assigning a Subscript to a Name (line 909):
    
    # Obtaining the type of the subscript
    int_22122 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 909, 16), 'int')
    
    # Call to idzp_rsvd(...): (line 909)
    # Processing the call arguments (line 909)
    # Getting the type of 'eps' (line 909)
    eps_22125 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 909, 44), 'eps', False)
    # Getting the type of 'm' (line 909)
    m_22126 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 909, 49), 'm', False)
    # Getting the type of 'n' (line 909)
    n_22127 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 909, 52), 'n', False)
    # Getting the type of 'matveca' (line 909)
    matveca_22128 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 909, 55), 'matveca', False)
    # Getting the type of 'matvec' (line 909)
    matvec_22129 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 909, 64), 'matvec', False)
    # Processing the call keyword arguments (line 909)
    kwargs_22130 = {}
    # Getting the type of 'backend' (line 909)
    backend_22123 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 909, 26), 'backend', False)
    # Obtaining the member 'idzp_rsvd' of a type (line 909)
    idzp_rsvd_22124 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 909, 26), backend_22123, 'idzp_rsvd')
    # Calling idzp_rsvd(args, kwargs) (line 909)
    idzp_rsvd_call_result_22131 = invoke(stypy.reporting.localization.Localization(__file__, 909, 26), idzp_rsvd_22124, *[eps_22125, m_22126, n_22127, matveca_22128, matvec_22129], **kwargs_22130)
    
    # Obtaining the member '__getitem__' of a type (line 909)
    getitem___22132 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 909, 16), idzp_rsvd_call_result_22131, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 909)
    subscript_call_result_22133 = invoke(stypy.reporting.localization.Localization(__file__, 909, 16), getitem___22132, int_22122)
    
    # Assigning a type to the variable 'tuple_var_assignment_20863' (line 909)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 909, 16), 'tuple_var_assignment_20863', subscript_call_result_22133)
    
    # Assigning a Name to a Name (line 909):
    # Getting the type of 'tuple_var_assignment_20861' (line 909)
    tuple_var_assignment_20861_22134 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 909, 16), 'tuple_var_assignment_20861')
    # Assigning a type to the variable 'U' (line 909)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 909, 16), 'U', tuple_var_assignment_20861_22134)
    
    # Assigning a Name to a Name (line 909):
    # Getting the type of 'tuple_var_assignment_20862' (line 909)
    tuple_var_assignment_20862_22135 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 909, 16), 'tuple_var_assignment_20862')
    # Assigning a type to the variable 'V' (line 909)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 909, 19), 'V', tuple_var_assignment_20862_22135)
    
    # Assigning a Name to a Name (line 909):
    # Getting the type of 'tuple_var_assignment_20863' (line 909)
    tuple_var_assignment_20863_22136 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 909, 16), 'tuple_var_assignment_20863')
    # Assigning a type to the variable 'S' (line 909)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 909, 22), 'S', tuple_var_assignment_20863_22136)
    # SSA join for if statement (line 906)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA branch for the else part of an if statement (line 904)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Call to a Name (line 911):
    
    # Assigning a Call to a Name (line 911):
    
    # Call to int(...): (line 911)
    # Processing the call arguments (line 911)
    # Getting the type of 'eps_or_k' (line 911)
    eps_or_k_22138 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 911, 20), 'eps_or_k', False)
    # Processing the call keyword arguments (line 911)
    kwargs_22139 = {}
    # Getting the type of 'int' (line 911)
    int_22137 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 911, 16), 'int', False)
    # Calling int(args, kwargs) (line 911)
    int_call_result_22140 = invoke(stypy.reporting.localization.Localization(__file__, 911, 16), int_22137, *[eps_or_k_22138], **kwargs_22139)
    
    # Assigning a type to the variable 'k' (line 911)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 911, 12), 'k', int_call_result_22140)
    
    # Getting the type of 'real' (line 912)
    real_22141 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 912, 15), 'real')
    # Testing the type of an if condition (line 912)
    if_condition_22142 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 912, 12), real_22141)
    # Assigning a type to the variable 'if_condition_22142' (line 912)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 912, 12), 'if_condition_22142', if_condition_22142)
    # SSA begins for if statement (line 912)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Tuple (line 913):
    
    # Assigning a Subscript to a Name (line 913):
    
    # Obtaining the type of the subscript
    int_22143 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 913, 16), 'int')
    
    # Call to iddr_rsvd(...): (line 913)
    # Processing the call arguments (line 913)
    # Getting the type of 'm' (line 913)
    m_22146 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 913, 44), 'm', False)
    # Getting the type of 'n' (line 913)
    n_22147 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 913, 47), 'n', False)
    # Getting the type of 'matveca' (line 913)
    matveca_22148 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 913, 50), 'matveca', False)
    # Getting the type of 'matvec' (line 913)
    matvec_22149 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 913, 59), 'matvec', False)
    # Getting the type of 'k' (line 913)
    k_22150 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 913, 67), 'k', False)
    # Processing the call keyword arguments (line 913)
    kwargs_22151 = {}
    # Getting the type of 'backend' (line 913)
    backend_22144 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 913, 26), 'backend', False)
    # Obtaining the member 'iddr_rsvd' of a type (line 913)
    iddr_rsvd_22145 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 913, 26), backend_22144, 'iddr_rsvd')
    # Calling iddr_rsvd(args, kwargs) (line 913)
    iddr_rsvd_call_result_22152 = invoke(stypy.reporting.localization.Localization(__file__, 913, 26), iddr_rsvd_22145, *[m_22146, n_22147, matveca_22148, matvec_22149, k_22150], **kwargs_22151)
    
    # Obtaining the member '__getitem__' of a type (line 913)
    getitem___22153 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 913, 16), iddr_rsvd_call_result_22152, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 913)
    subscript_call_result_22154 = invoke(stypy.reporting.localization.Localization(__file__, 913, 16), getitem___22153, int_22143)
    
    # Assigning a type to the variable 'tuple_var_assignment_20864' (line 913)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 913, 16), 'tuple_var_assignment_20864', subscript_call_result_22154)
    
    # Assigning a Subscript to a Name (line 913):
    
    # Obtaining the type of the subscript
    int_22155 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 913, 16), 'int')
    
    # Call to iddr_rsvd(...): (line 913)
    # Processing the call arguments (line 913)
    # Getting the type of 'm' (line 913)
    m_22158 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 913, 44), 'm', False)
    # Getting the type of 'n' (line 913)
    n_22159 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 913, 47), 'n', False)
    # Getting the type of 'matveca' (line 913)
    matveca_22160 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 913, 50), 'matveca', False)
    # Getting the type of 'matvec' (line 913)
    matvec_22161 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 913, 59), 'matvec', False)
    # Getting the type of 'k' (line 913)
    k_22162 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 913, 67), 'k', False)
    # Processing the call keyword arguments (line 913)
    kwargs_22163 = {}
    # Getting the type of 'backend' (line 913)
    backend_22156 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 913, 26), 'backend', False)
    # Obtaining the member 'iddr_rsvd' of a type (line 913)
    iddr_rsvd_22157 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 913, 26), backend_22156, 'iddr_rsvd')
    # Calling iddr_rsvd(args, kwargs) (line 913)
    iddr_rsvd_call_result_22164 = invoke(stypy.reporting.localization.Localization(__file__, 913, 26), iddr_rsvd_22157, *[m_22158, n_22159, matveca_22160, matvec_22161, k_22162], **kwargs_22163)
    
    # Obtaining the member '__getitem__' of a type (line 913)
    getitem___22165 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 913, 16), iddr_rsvd_call_result_22164, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 913)
    subscript_call_result_22166 = invoke(stypy.reporting.localization.Localization(__file__, 913, 16), getitem___22165, int_22155)
    
    # Assigning a type to the variable 'tuple_var_assignment_20865' (line 913)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 913, 16), 'tuple_var_assignment_20865', subscript_call_result_22166)
    
    # Assigning a Subscript to a Name (line 913):
    
    # Obtaining the type of the subscript
    int_22167 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 913, 16), 'int')
    
    # Call to iddr_rsvd(...): (line 913)
    # Processing the call arguments (line 913)
    # Getting the type of 'm' (line 913)
    m_22170 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 913, 44), 'm', False)
    # Getting the type of 'n' (line 913)
    n_22171 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 913, 47), 'n', False)
    # Getting the type of 'matveca' (line 913)
    matveca_22172 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 913, 50), 'matveca', False)
    # Getting the type of 'matvec' (line 913)
    matvec_22173 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 913, 59), 'matvec', False)
    # Getting the type of 'k' (line 913)
    k_22174 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 913, 67), 'k', False)
    # Processing the call keyword arguments (line 913)
    kwargs_22175 = {}
    # Getting the type of 'backend' (line 913)
    backend_22168 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 913, 26), 'backend', False)
    # Obtaining the member 'iddr_rsvd' of a type (line 913)
    iddr_rsvd_22169 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 913, 26), backend_22168, 'iddr_rsvd')
    # Calling iddr_rsvd(args, kwargs) (line 913)
    iddr_rsvd_call_result_22176 = invoke(stypy.reporting.localization.Localization(__file__, 913, 26), iddr_rsvd_22169, *[m_22170, n_22171, matveca_22172, matvec_22173, k_22174], **kwargs_22175)
    
    # Obtaining the member '__getitem__' of a type (line 913)
    getitem___22177 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 913, 16), iddr_rsvd_call_result_22176, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 913)
    subscript_call_result_22178 = invoke(stypy.reporting.localization.Localization(__file__, 913, 16), getitem___22177, int_22167)
    
    # Assigning a type to the variable 'tuple_var_assignment_20866' (line 913)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 913, 16), 'tuple_var_assignment_20866', subscript_call_result_22178)
    
    # Assigning a Name to a Name (line 913):
    # Getting the type of 'tuple_var_assignment_20864' (line 913)
    tuple_var_assignment_20864_22179 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 913, 16), 'tuple_var_assignment_20864')
    # Assigning a type to the variable 'U' (line 913)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 913, 16), 'U', tuple_var_assignment_20864_22179)
    
    # Assigning a Name to a Name (line 913):
    # Getting the type of 'tuple_var_assignment_20865' (line 913)
    tuple_var_assignment_20865_22180 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 913, 16), 'tuple_var_assignment_20865')
    # Assigning a type to the variable 'V' (line 913)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 913, 19), 'V', tuple_var_assignment_20865_22180)
    
    # Assigning a Name to a Name (line 913):
    # Getting the type of 'tuple_var_assignment_20866' (line 913)
    tuple_var_assignment_20866_22181 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 913, 16), 'tuple_var_assignment_20866')
    # Assigning a type to the variable 'S' (line 913)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 913, 22), 'S', tuple_var_assignment_20866_22181)
    # SSA branch for the else part of an if statement (line 912)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Call to a Tuple (line 915):
    
    # Assigning a Subscript to a Name (line 915):
    
    # Obtaining the type of the subscript
    int_22182 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 915, 16), 'int')
    
    # Call to idzr_rsvd(...): (line 915)
    # Processing the call arguments (line 915)
    # Getting the type of 'm' (line 915)
    m_22185 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 915, 44), 'm', False)
    # Getting the type of 'n' (line 915)
    n_22186 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 915, 47), 'n', False)
    # Getting the type of 'matveca' (line 915)
    matveca_22187 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 915, 50), 'matveca', False)
    # Getting the type of 'matvec' (line 915)
    matvec_22188 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 915, 59), 'matvec', False)
    # Getting the type of 'k' (line 915)
    k_22189 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 915, 67), 'k', False)
    # Processing the call keyword arguments (line 915)
    kwargs_22190 = {}
    # Getting the type of 'backend' (line 915)
    backend_22183 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 915, 26), 'backend', False)
    # Obtaining the member 'idzr_rsvd' of a type (line 915)
    idzr_rsvd_22184 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 915, 26), backend_22183, 'idzr_rsvd')
    # Calling idzr_rsvd(args, kwargs) (line 915)
    idzr_rsvd_call_result_22191 = invoke(stypy.reporting.localization.Localization(__file__, 915, 26), idzr_rsvd_22184, *[m_22185, n_22186, matveca_22187, matvec_22188, k_22189], **kwargs_22190)
    
    # Obtaining the member '__getitem__' of a type (line 915)
    getitem___22192 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 915, 16), idzr_rsvd_call_result_22191, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 915)
    subscript_call_result_22193 = invoke(stypy.reporting.localization.Localization(__file__, 915, 16), getitem___22192, int_22182)
    
    # Assigning a type to the variable 'tuple_var_assignment_20867' (line 915)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 915, 16), 'tuple_var_assignment_20867', subscript_call_result_22193)
    
    # Assigning a Subscript to a Name (line 915):
    
    # Obtaining the type of the subscript
    int_22194 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 915, 16), 'int')
    
    # Call to idzr_rsvd(...): (line 915)
    # Processing the call arguments (line 915)
    # Getting the type of 'm' (line 915)
    m_22197 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 915, 44), 'm', False)
    # Getting the type of 'n' (line 915)
    n_22198 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 915, 47), 'n', False)
    # Getting the type of 'matveca' (line 915)
    matveca_22199 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 915, 50), 'matveca', False)
    # Getting the type of 'matvec' (line 915)
    matvec_22200 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 915, 59), 'matvec', False)
    # Getting the type of 'k' (line 915)
    k_22201 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 915, 67), 'k', False)
    # Processing the call keyword arguments (line 915)
    kwargs_22202 = {}
    # Getting the type of 'backend' (line 915)
    backend_22195 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 915, 26), 'backend', False)
    # Obtaining the member 'idzr_rsvd' of a type (line 915)
    idzr_rsvd_22196 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 915, 26), backend_22195, 'idzr_rsvd')
    # Calling idzr_rsvd(args, kwargs) (line 915)
    idzr_rsvd_call_result_22203 = invoke(stypy.reporting.localization.Localization(__file__, 915, 26), idzr_rsvd_22196, *[m_22197, n_22198, matveca_22199, matvec_22200, k_22201], **kwargs_22202)
    
    # Obtaining the member '__getitem__' of a type (line 915)
    getitem___22204 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 915, 16), idzr_rsvd_call_result_22203, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 915)
    subscript_call_result_22205 = invoke(stypy.reporting.localization.Localization(__file__, 915, 16), getitem___22204, int_22194)
    
    # Assigning a type to the variable 'tuple_var_assignment_20868' (line 915)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 915, 16), 'tuple_var_assignment_20868', subscript_call_result_22205)
    
    # Assigning a Subscript to a Name (line 915):
    
    # Obtaining the type of the subscript
    int_22206 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 915, 16), 'int')
    
    # Call to idzr_rsvd(...): (line 915)
    # Processing the call arguments (line 915)
    # Getting the type of 'm' (line 915)
    m_22209 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 915, 44), 'm', False)
    # Getting the type of 'n' (line 915)
    n_22210 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 915, 47), 'n', False)
    # Getting the type of 'matveca' (line 915)
    matveca_22211 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 915, 50), 'matveca', False)
    # Getting the type of 'matvec' (line 915)
    matvec_22212 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 915, 59), 'matvec', False)
    # Getting the type of 'k' (line 915)
    k_22213 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 915, 67), 'k', False)
    # Processing the call keyword arguments (line 915)
    kwargs_22214 = {}
    # Getting the type of 'backend' (line 915)
    backend_22207 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 915, 26), 'backend', False)
    # Obtaining the member 'idzr_rsvd' of a type (line 915)
    idzr_rsvd_22208 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 915, 26), backend_22207, 'idzr_rsvd')
    # Calling idzr_rsvd(args, kwargs) (line 915)
    idzr_rsvd_call_result_22215 = invoke(stypy.reporting.localization.Localization(__file__, 915, 26), idzr_rsvd_22208, *[m_22209, n_22210, matveca_22211, matvec_22212, k_22213], **kwargs_22214)
    
    # Obtaining the member '__getitem__' of a type (line 915)
    getitem___22216 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 915, 16), idzr_rsvd_call_result_22215, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 915)
    subscript_call_result_22217 = invoke(stypy.reporting.localization.Localization(__file__, 915, 16), getitem___22216, int_22206)
    
    # Assigning a type to the variable 'tuple_var_assignment_20869' (line 915)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 915, 16), 'tuple_var_assignment_20869', subscript_call_result_22217)
    
    # Assigning a Name to a Name (line 915):
    # Getting the type of 'tuple_var_assignment_20867' (line 915)
    tuple_var_assignment_20867_22218 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 915, 16), 'tuple_var_assignment_20867')
    # Assigning a type to the variable 'U' (line 915)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 915, 16), 'U', tuple_var_assignment_20867_22218)
    
    # Assigning a Name to a Name (line 915):
    # Getting the type of 'tuple_var_assignment_20868' (line 915)
    tuple_var_assignment_20868_22219 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 915, 16), 'tuple_var_assignment_20868')
    # Assigning a type to the variable 'V' (line 915)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 915, 19), 'V', tuple_var_assignment_20868_22219)
    
    # Assigning a Name to a Name (line 915):
    # Getting the type of 'tuple_var_assignment_20869' (line 915)
    tuple_var_assignment_20869_22220 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 915, 16), 'tuple_var_assignment_20869')
    # Assigning a type to the variable 'S' (line 915)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 915, 22), 'S', tuple_var_assignment_20869_22220)
    # SSA join for if statement (line 912)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 904)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA branch for the else part of an if statement (line 900)
    module_type_store.open_ssa_branch('else')
    # Getting the type of '_TYPE_ERROR' (line 917)
    _TYPE_ERROR_22221 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 917, 14), '_TYPE_ERROR')
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 917, 8), _TYPE_ERROR_22221, 'raise parameter', BaseException)
    # SSA join for if statement (line 900)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 872)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Obtaining an instance of the builtin type 'tuple' (line 918)
    tuple_22222 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 918, 11), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 918)
    # Adding element type (line 918)
    # Getting the type of 'U' (line 918)
    U_22223 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 918, 11), 'U')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 918, 11), tuple_22222, U_22223)
    # Adding element type (line 918)
    # Getting the type of 'S' (line 918)
    S_22224 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 918, 14), 'S')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 918, 11), tuple_22222, S_22224)
    # Adding element type (line 918)
    # Getting the type of 'V' (line 918)
    V_22225 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 918, 17), 'V')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 918, 11), tuple_22222, V_22225)
    
    # Assigning a type to the variable 'stypy_return_type' (line 918)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 918, 4), 'stypy_return_type', tuple_22222)
    
    # ################# End of 'svd(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'svd' in the type store
    # Getting the type of 'stypy_return_type' (line 821)
    stypy_return_type_22226 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 821, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_22226)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'svd'
    return stypy_return_type_22226

# Assigning a type to the variable 'svd' (line 821)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 821, 0), 'svd', svd)

@norecursion
def estimate_rank(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'estimate_rank'
    module_type_store = module_type_store.open_function_context('estimate_rank', 921, 0, False)
    
    # Passed parameters checking function
    estimate_rank.stypy_localization = localization
    estimate_rank.stypy_type_of_self = None
    estimate_rank.stypy_type_store = module_type_store
    estimate_rank.stypy_function_name = 'estimate_rank'
    estimate_rank.stypy_param_names_list = ['A', 'eps']
    estimate_rank.stypy_varargs_param_name = None
    estimate_rank.stypy_kwargs_param_name = None
    estimate_rank.stypy_call_defaults = defaults
    estimate_rank.stypy_call_varargs = varargs
    estimate_rank.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'estimate_rank', ['A', 'eps'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'estimate_rank', localization, ['A', 'eps'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'estimate_rank(...)' code ##################

    str_22227 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 949, (-1)), 'str', '\n    Estimate matrix rank to a specified relative precision using randomized\n    methods.\n\n    The matrix `A` can be given as either a :class:`numpy.ndarray` or a\n    :class:`scipy.sparse.linalg.LinearOperator`, with different algorithms used\n    for each case. If `A` is of type :class:`numpy.ndarray`, then the output\n    rank is typically about 8 higher than the actual numerical rank.\n\n    ..  This function automatically detects the form of the input parameters and\n        passes them to the appropriate backend. For details,\n        see :func:`backend.idd_estrank`, :func:`backend.idd_findrank`,\n        :func:`backend.idz_estrank`, and :func:`backend.idz_findrank`.\n\n    Parameters\n    ----------\n    A : :class:`numpy.ndarray` or :class:`scipy.sparse.linalg.LinearOperator`\n        Matrix whose rank is to be estimated, given as either a\n        :class:`numpy.ndarray` or a :class:`scipy.sparse.linalg.LinearOperator`\n        with the `rmatvec` method (to apply the matrix adjoint).\n    eps : float\n        Relative error for numerical rank definition.\n\n    Returns\n    -------\n    int\n        Estimated matrix rank.\n    ')
    stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 950, 4))
    
    # 'from scipy.sparse.linalg import LinearOperator' statement (line 950)
    update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/linalg/')
    import_22228 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 950, 4), 'scipy.sparse.linalg')

    if (type(import_22228) is not StypyTypeError):

        if (import_22228 != 'pyd_module'):
            __import__(import_22228)
            sys_modules_22229 = sys.modules[import_22228]
            import_from_module(stypy.reporting.localization.Localization(__file__, 950, 4), 'scipy.sparse.linalg', sys_modules_22229.module_type_store, module_type_store, ['LinearOperator'])
            nest_module(stypy.reporting.localization.Localization(__file__, 950, 4), __file__, sys_modules_22229, sys_modules_22229.module_type_store, module_type_store)
        else:
            from scipy.sparse.linalg import LinearOperator

            import_from_module(stypy.reporting.localization.Localization(__file__, 950, 4), 'scipy.sparse.linalg', None, module_type_store, ['LinearOperator'], [LinearOperator])

    else:
        # Assigning a type to the variable 'scipy.sparse.linalg' (line 950)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 950, 4), 'scipy.sparse.linalg', import_22228)

    remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/linalg/')
    
    
    # Assigning a Call to a Name (line 952):
    
    # Assigning a Call to a Name (line 952):
    
    # Call to _is_real(...): (line 952)
    # Processing the call arguments (line 952)
    # Getting the type of 'A' (line 952)
    A_22231 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 952, 20), 'A', False)
    # Processing the call keyword arguments (line 952)
    kwargs_22232 = {}
    # Getting the type of '_is_real' (line 952)
    _is_real_22230 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 952, 11), '_is_real', False)
    # Calling _is_real(args, kwargs) (line 952)
    _is_real_call_result_22233 = invoke(stypy.reporting.localization.Localization(__file__, 952, 11), _is_real_22230, *[A_22231], **kwargs_22232)
    
    # Assigning a type to the variable 'real' (line 952)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 952, 4), 'real', _is_real_call_result_22233)
    
    
    # Call to isinstance(...): (line 954)
    # Processing the call arguments (line 954)
    # Getting the type of 'A' (line 954)
    A_22235 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 954, 18), 'A', False)
    # Getting the type of 'np' (line 954)
    np_22236 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 954, 21), 'np', False)
    # Obtaining the member 'ndarray' of a type (line 954)
    ndarray_22237 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 954, 21), np_22236, 'ndarray')
    # Processing the call keyword arguments (line 954)
    kwargs_22238 = {}
    # Getting the type of 'isinstance' (line 954)
    isinstance_22234 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 954, 7), 'isinstance', False)
    # Calling isinstance(args, kwargs) (line 954)
    isinstance_call_result_22239 = invoke(stypy.reporting.localization.Localization(__file__, 954, 7), isinstance_22234, *[A_22235, ndarray_22237], **kwargs_22238)
    
    # Testing the type of an if condition (line 954)
    if_condition_22240 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 954, 4), isinstance_call_result_22239)
    # Assigning a type to the variable 'if_condition_22240' (line 954)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 954, 4), 'if_condition_22240', if_condition_22240)
    # SSA begins for if statement (line 954)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Getting the type of 'real' (line 955)
    real_22241 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 955, 11), 'real')
    # Testing the type of an if condition (line 955)
    if_condition_22242 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 955, 8), real_22241)
    # Assigning a type to the variable 'if_condition_22242' (line 955)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 955, 8), 'if_condition_22242', if_condition_22242)
    # SSA begins for if statement (line 955)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 956):
    
    # Assigning a Call to a Name (line 956):
    
    # Call to idd_estrank(...): (line 956)
    # Processing the call arguments (line 956)
    # Getting the type of 'eps' (line 956)
    eps_22245 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 956, 39), 'eps', False)
    # Getting the type of 'A' (line 956)
    A_22246 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 956, 44), 'A', False)
    # Processing the call keyword arguments (line 956)
    kwargs_22247 = {}
    # Getting the type of 'backend' (line 956)
    backend_22243 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 956, 19), 'backend', False)
    # Obtaining the member 'idd_estrank' of a type (line 956)
    idd_estrank_22244 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 956, 19), backend_22243, 'idd_estrank')
    # Calling idd_estrank(args, kwargs) (line 956)
    idd_estrank_call_result_22248 = invoke(stypy.reporting.localization.Localization(__file__, 956, 19), idd_estrank_22244, *[eps_22245, A_22246], **kwargs_22247)
    
    # Assigning a type to the variable 'rank' (line 956)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 956, 12), 'rank', idd_estrank_call_result_22248)
    # SSA branch for the else part of an if statement (line 955)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Call to a Name (line 958):
    
    # Assigning a Call to a Name (line 958):
    
    # Call to idz_estrank(...): (line 958)
    # Processing the call arguments (line 958)
    # Getting the type of 'eps' (line 958)
    eps_22251 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 958, 39), 'eps', False)
    # Getting the type of 'A' (line 958)
    A_22252 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 958, 44), 'A', False)
    # Processing the call keyword arguments (line 958)
    kwargs_22253 = {}
    # Getting the type of 'backend' (line 958)
    backend_22249 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 958, 19), 'backend', False)
    # Obtaining the member 'idz_estrank' of a type (line 958)
    idz_estrank_22250 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 958, 19), backend_22249, 'idz_estrank')
    # Calling idz_estrank(args, kwargs) (line 958)
    idz_estrank_call_result_22254 = invoke(stypy.reporting.localization.Localization(__file__, 958, 19), idz_estrank_22250, *[eps_22251, A_22252], **kwargs_22253)
    
    # Assigning a type to the variable 'rank' (line 958)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 958, 12), 'rank', idz_estrank_call_result_22254)
    # SSA join for if statement (line 955)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'rank' (line 959)
    rank_22255 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 959, 11), 'rank')
    int_22256 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 959, 19), 'int')
    # Applying the binary operator '==' (line 959)
    result_eq_22257 = python_operator(stypy.reporting.localization.Localization(__file__, 959, 11), '==', rank_22255, int_22256)
    
    # Testing the type of an if condition (line 959)
    if_condition_22258 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 959, 8), result_eq_22257)
    # Assigning a type to the variable 'if_condition_22258' (line 959)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 959, 8), 'if_condition_22258', if_condition_22258)
    # SSA begins for if statement (line 959)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 961):
    
    # Assigning a Call to a Name (line 961):
    
    # Call to min(...): (line 961)
    # Processing the call arguments (line 961)
    # Getting the type of 'A' (line 961)
    A_22260 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 961, 23), 'A', False)
    # Obtaining the member 'shape' of a type (line 961)
    shape_22261 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 961, 23), A_22260, 'shape')
    # Processing the call keyword arguments (line 961)
    kwargs_22262 = {}
    # Getting the type of 'min' (line 961)
    min_22259 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 961, 19), 'min', False)
    # Calling min(args, kwargs) (line 961)
    min_call_result_22263 = invoke(stypy.reporting.localization.Localization(__file__, 961, 19), min_22259, *[shape_22261], **kwargs_22262)
    
    # Assigning a type to the variable 'rank' (line 961)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 961, 12), 'rank', min_call_result_22263)
    # SSA join for if statement (line 959)
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'rank' (line 962)
    rank_22264 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 962, 15), 'rank')
    # Assigning a type to the variable 'stypy_return_type' (line 962)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 962, 8), 'stypy_return_type', rank_22264)
    # SSA branch for the else part of an if statement (line 954)
    module_type_store.open_ssa_branch('else')
    
    
    # Call to isinstance(...): (line 963)
    # Processing the call arguments (line 963)
    # Getting the type of 'A' (line 963)
    A_22266 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 963, 20), 'A', False)
    # Getting the type of 'LinearOperator' (line 963)
    LinearOperator_22267 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 963, 23), 'LinearOperator', False)
    # Processing the call keyword arguments (line 963)
    kwargs_22268 = {}
    # Getting the type of 'isinstance' (line 963)
    isinstance_22265 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 963, 9), 'isinstance', False)
    # Calling isinstance(args, kwargs) (line 963)
    isinstance_call_result_22269 = invoke(stypy.reporting.localization.Localization(__file__, 963, 9), isinstance_22265, *[A_22266, LinearOperator_22267], **kwargs_22268)
    
    # Testing the type of an if condition (line 963)
    if_condition_22270 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 963, 9), isinstance_call_result_22269)
    # Assigning a type to the variable 'if_condition_22270' (line 963)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 963, 9), 'if_condition_22270', if_condition_22270)
    # SSA begins for if statement (line 963)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Attribute to a Tuple (line 964):
    
    # Assigning a Subscript to a Name (line 964):
    
    # Obtaining the type of the subscript
    int_22271 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 964, 8), 'int')
    # Getting the type of 'A' (line 964)
    A_22272 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 964, 15), 'A')
    # Obtaining the member 'shape' of a type (line 964)
    shape_22273 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 964, 15), A_22272, 'shape')
    # Obtaining the member '__getitem__' of a type (line 964)
    getitem___22274 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 964, 8), shape_22273, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 964)
    subscript_call_result_22275 = invoke(stypy.reporting.localization.Localization(__file__, 964, 8), getitem___22274, int_22271)
    
    # Assigning a type to the variable 'tuple_var_assignment_20870' (line 964)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 964, 8), 'tuple_var_assignment_20870', subscript_call_result_22275)
    
    # Assigning a Subscript to a Name (line 964):
    
    # Obtaining the type of the subscript
    int_22276 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 964, 8), 'int')
    # Getting the type of 'A' (line 964)
    A_22277 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 964, 15), 'A')
    # Obtaining the member 'shape' of a type (line 964)
    shape_22278 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 964, 15), A_22277, 'shape')
    # Obtaining the member '__getitem__' of a type (line 964)
    getitem___22279 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 964, 8), shape_22278, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 964)
    subscript_call_result_22280 = invoke(stypy.reporting.localization.Localization(__file__, 964, 8), getitem___22279, int_22276)
    
    # Assigning a type to the variable 'tuple_var_assignment_20871' (line 964)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 964, 8), 'tuple_var_assignment_20871', subscript_call_result_22280)
    
    # Assigning a Name to a Name (line 964):
    # Getting the type of 'tuple_var_assignment_20870' (line 964)
    tuple_var_assignment_20870_22281 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 964, 8), 'tuple_var_assignment_20870')
    # Assigning a type to the variable 'm' (line 964)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 964, 8), 'm', tuple_var_assignment_20870_22281)
    
    # Assigning a Name to a Name (line 964):
    # Getting the type of 'tuple_var_assignment_20871' (line 964)
    tuple_var_assignment_20871_22282 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 964, 8), 'tuple_var_assignment_20871')
    # Assigning a type to the variable 'n' (line 964)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 964, 11), 'n', tuple_var_assignment_20871_22282)
    
    # Assigning a Attribute to a Name (line 965):
    
    # Assigning a Attribute to a Name (line 965):
    # Getting the type of 'A' (line 965)
    A_22283 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 965, 18), 'A')
    # Obtaining the member 'rmatvec' of a type (line 965)
    rmatvec_22284 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 965, 18), A_22283, 'rmatvec')
    # Assigning a type to the variable 'matveca' (line 965)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 965, 8), 'matveca', rmatvec_22284)
    
    # Getting the type of 'real' (line 966)
    real_22285 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 966, 11), 'real')
    # Testing the type of an if condition (line 966)
    if_condition_22286 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 966, 8), real_22285)
    # Assigning a type to the variable 'if_condition_22286' (line 966)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 966, 8), 'if_condition_22286', if_condition_22286)
    # SSA begins for if statement (line 966)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to idd_findrank(...): (line 967)
    # Processing the call arguments (line 967)
    # Getting the type of 'eps' (line 967)
    eps_22289 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 967, 40), 'eps', False)
    # Getting the type of 'm' (line 967)
    m_22290 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 967, 45), 'm', False)
    # Getting the type of 'n' (line 967)
    n_22291 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 967, 48), 'n', False)
    # Getting the type of 'matveca' (line 967)
    matveca_22292 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 967, 51), 'matveca', False)
    # Processing the call keyword arguments (line 967)
    kwargs_22293 = {}
    # Getting the type of 'backend' (line 967)
    backend_22287 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 967, 19), 'backend', False)
    # Obtaining the member 'idd_findrank' of a type (line 967)
    idd_findrank_22288 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 967, 19), backend_22287, 'idd_findrank')
    # Calling idd_findrank(args, kwargs) (line 967)
    idd_findrank_call_result_22294 = invoke(stypy.reporting.localization.Localization(__file__, 967, 19), idd_findrank_22288, *[eps_22289, m_22290, n_22291, matveca_22292], **kwargs_22293)
    
    # Assigning a type to the variable 'stypy_return_type' (line 967)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 967, 12), 'stypy_return_type', idd_findrank_call_result_22294)
    # SSA branch for the else part of an if statement (line 966)
    module_type_store.open_ssa_branch('else')
    
    # Call to idz_findrank(...): (line 969)
    # Processing the call arguments (line 969)
    # Getting the type of 'eps' (line 969)
    eps_22297 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 969, 40), 'eps', False)
    # Getting the type of 'm' (line 969)
    m_22298 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 969, 45), 'm', False)
    # Getting the type of 'n' (line 969)
    n_22299 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 969, 48), 'n', False)
    # Getting the type of 'matveca' (line 969)
    matveca_22300 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 969, 51), 'matveca', False)
    # Processing the call keyword arguments (line 969)
    kwargs_22301 = {}
    # Getting the type of 'backend' (line 969)
    backend_22295 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 969, 19), 'backend', False)
    # Obtaining the member 'idz_findrank' of a type (line 969)
    idz_findrank_22296 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 969, 19), backend_22295, 'idz_findrank')
    # Calling idz_findrank(args, kwargs) (line 969)
    idz_findrank_call_result_22302 = invoke(stypy.reporting.localization.Localization(__file__, 969, 19), idz_findrank_22296, *[eps_22297, m_22298, n_22299, matveca_22300], **kwargs_22301)
    
    # Assigning a type to the variable 'stypy_return_type' (line 969)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 969, 12), 'stypy_return_type', idz_findrank_call_result_22302)
    # SSA join for if statement (line 966)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA branch for the else part of an if statement (line 963)
    module_type_store.open_ssa_branch('else')
    # Getting the type of '_TYPE_ERROR' (line 971)
    _TYPE_ERROR_22303 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 971, 14), '_TYPE_ERROR')
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 971, 8), _TYPE_ERROR_22303, 'raise parameter', BaseException)
    # SSA join for if statement (line 963)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 954)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'estimate_rank(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'estimate_rank' in the type store
    # Getting the type of 'stypy_return_type' (line 921)
    stypy_return_type_22304 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 921, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_22304)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'estimate_rank'
    return stypy_return_type_22304

# Assigning a type to the variable 'estimate_rank' (line 921)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 921, 0), 'estimate_rank', estimate_rank)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
