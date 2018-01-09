
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
30: '''
31: Direct wrappers for Fortran `id_dist` backend.
32: '''
33: 
34: import scipy.linalg._interpolative as _id
35: import numpy as np
36: 
37: _RETCODE_ERROR = RuntimeError("nonzero return code")
38: 
39: 
40: #------------------------------------------------------------------------------
41: # id_rand.f
42: #------------------------------------------------------------------------------
43: 
44: def id_srand(n):
45:     '''
46:     Generate standard uniform pseudorandom numbers via a very efficient lagged
47:     Fibonacci method.
48: 
49:     :param n:
50:         Number of pseudorandom numbers to generate.
51:     :type n: int
52: 
53:     :return:
54:         Pseudorandom numbers.
55:     :rtype: :class:`numpy.ndarray`
56:     '''
57:     return _id.id_srand(n)
58: 
59: 
60: def id_srandi(t):
61:     '''
62:     Initialize seed values for :func:`id_srand` (any appropriately random
63:     numbers will do).
64: 
65:     :param t:
66:         Array of 55 seed values.
67:     :type t: :class:`numpy.ndarray`
68:     '''
69:     t = np.asfortranarray(t)
70:     _id.id_srandi(t)
71: 
72: 
73: def id_srando():
74:     '''
75:     Reset seed values to their original values.
76:     '''
77:     _id.id_srando()
78: 
79: 
80: #------------------------------------------------------------------------------
81: # idd_frm.f
82: #------------------------------------------------------------------------------
83: 
84: def idd_frm(n, w, x):
85:     '''
86:     Transform real vector via a composition of Rokhlin's random transform,
87:     random subselection, and an FFT.
88: 
89:     In contrast to :func:`idd_sfrm`, this routine works best when the length of
90:     the transformed vector is the power-of-two integer output by
91:     :func:`idd_frmi`, or when the length is not specified but instead
92:     determined a posteriori from the output. The returned transformed vector is
93:     randomly permuted.
94: 
95:     :param n:
96:         Greatest power-of-two integer satisfying `n <= x.size` as obtained from
97:         :func:`idd_frmi`; `n` is also the length of the output vector.
98:     :type n: int
99:     :param w:
100:         Initialization array constructed by :func:`idd_frmi`.
101:     :type w: :class:`numpy.ndarray`
102:     :param x:
103:         Vector to be transformed.
104:     :type x: :class:`numpy.ndarray`
105: 
106:     :return:
107:         Transformed vector.
108:     :rtype: :class:`numpy.ndarray`
109:     '''
110:     return _id.idd_frm(n, w, x)
111: 
112: 
113: def idd_sfrm(l, n, w, x):
114:     '''
115:     Transform real vector via a composition of Rokhlin's random transform,
116:     random subselection, and an FFT.
117: 
118:     In contrast to :func:`idd_frm`, this routine works best when the length of
119:     the transformed vector is known a priori.
120: 
121:     :param l:
122:         Length of transformed vector, satisfying `l <= n`.
123:     :type l: int
124:     :param n:
125:         Greatest power-of-two integer satisfying `n <= x.size` as obtained from
126:         :func:`idd_sfrmi`.
127:     :type n: int
128:     :param w:
129:         Initialization array constructed by :func:`idd_sfrmi`.
130:     :type w: :class:`numpy.ndarray`
131:     :param x:
132:         Vector to be transformed.
133:     :type x: :class:`numpy.ndarray`
134: 
135:     :return:
136:         Transformed vector.
137:     :rtype: :class:`numpy.ndarray`
138:     '''
139:     return _id.idd_sfrm(l, n, w, x)
140: 
141: 
142: def idd_frmi(m):
143:     '''
144:     Initialize data for :func:`idd_frm`.
145: 
146:     :param m:
147:         Length of vector to be transformed.
148:     :type m: int
149: 
150:     :return:
151:         Greatest power-of-two integer `n` satisfying `n <= m`.
152:     :rtype: int
153:     :return:
154:         Initialization array to be used by :func:`idd_frm`.
155:     :rtype: :class:`numpy.ndarray`
156:     '''
157:     return _id.idd_frmi(m)
158: 
159: 
160: def idd_sfrmi(l, m):
161:     '''
162:     Initialize data for :func:`idd_sfrm`.
163: 
164:     :param l:
165:         Length of output transformed vector.
166:     :type l: int
167:     :param m:
168:         Length of the vector to be transformed.
169:     :type m: int
170: 
171:     :return:
172:         Greatest power-of-two integer `n` satisfying `n <= m`.
173:     :rtype: int
174:     :return:
175:         Initialization array to be used by :func:`idd_sfrm`.
176:     :rtype: :class:`numpy.ndarray`
177:     '''
178:     return _id.idd_sfrmi(l, m)
179: 
180: 
181: #------------------------------------------------------------------------------
182: # idd_id.f
183: #------------------------------------------------------------------------------
184: 
185: def iddp_id(eps, A):
186:     '''
187:     Compute ID of a real matrix to a specified relative precision.
188: 
189:     :param eps:
190:         Relative precision.
191:     :type eps: float
192:     :param A:
193:         Matrix.
194:     :type A: :class:`numpy.ndarray`
195: 
196:     :return:
197:         Rank of ID.
198:     :rtype: int
199:     :return:
200:         Column index array.
201:     :rtype: :class:`numpy.ndarray`
202:     :return:
203:         Interpolation coefficients.
204:     :rtype: :class:`numpy.ndarray`
205:     '''
206:     A = np.asfortranarray(A)
207:     k, idx, rnorms = _id.iddp_id(eps, A)
208:     n = A.shape[1]
209:     proj = A.T.ravel()[:k*(n-k)].reshape((k, n-k), order='F')
210:     return k, idx, proj
211: 
212: 
213: def iddr_id(A, k):
214:     '''
215:     Compute ID of a real matrix to a specified rank.
216: 
217:     :param A:
218:         Matrix.
219:     :type A: :class:`numpy.ndarray`
220:     :param k:
221:         Rank of ID.
222:     :type k: int
223: 
224:     :return:
225:         Column index array.
226:     :rtype: :class:`numpy.ndarray`
227:     :return:
228:         Interpolation coefficients.
229:     :rtype: :class:`numpy.ndarray`
230:     '''
231:     A = np.asfortranarray(A)
232:     idx, rnorms = _id.iddr_id(A, k)
233:     n = A.shape[1]
234:     proj = A.T.ravel()[:k*(n-k)].reshape((k, n-k), order='F')
235:     return idx, proj
236: 
237: 
238: def idd_reconid(B, idx, proj):
239:     '''
240:     Reconstruct matrix from real ID.
241: 
242:     :param B:
243:         Skeleton matrix.
244:     :type B: :class:`numpy.ndarray`
245:     :param idx:
246:         Column index array.
247:     :type idx: :class:`numpy.ndarray`
248:     :param proj:
249:         Interpolation coefficients.
250:     :type proj: :class:`numpy.ndarray`
251: 
252:     :return:
253:         Reconstructed matrix.
254:     :rtype: :class:`numpy.ndarray`
255:     '''
256:     B = np.asfortranarray(B)
257:     if proj.size > 0:
258:         return _id.idd_reconid(B, idx, proj)
259:     else:
260:         return B[:, np.argsort(idx)]
261: 
262: 
263: def idd_reconint(idx, proj):
264:     '''
265:     Reconstruct interpolation matrix from real ID.
266: 
267:     :param idx:
268:         Column index array.
269:     :type idx: :class:`numpy.ndarray`
270:     :param proj:
271:         Interpolation coefficients.
272:     :type proj: :class:`numpy.ndarray`
273: 
274:     :return:
275:         Interpolation matrix.
276:     :rtype: :class:`numpy.ndarray`
277:     '''
278:     return _id.idd_reconint(idx, proj)
279: 
280: 
281: def idd_copycols(A, k, idx):
282:     '''
283:     Reconstruct skeleton matrix from real ID.
284: 
285:     :param A:
286:         Original matrix.
287:     :type A: :class:`numpy.ndarray`
288:     :param k:
289:         Rank of ID.
290:     :type k: int
291:     :param idx:
292:         Column index array.
293:     :type idx: :class:`numpy.ndarray`
294: 
295:     :return:
296:         Skeleton matrix.
297:     :rtype: :class:`numpy.ndarray`
298:     '''
299:     A = np.asfortranarray(A)
300:     return _id.idd_copycols(A, k, idx)
301: 
302: 
303: #------------------------------------------------------------------------------
304: # idd_id2svd.f
305: #------------------------------------------------------------------------------
306: 
307: def idd_id2svd(B, idx, proj):
308:     '''
309:     Convert real ID to SVD.
310: 
311:     :param B:
312:         Skeleton matrix.
313:     :type B: :class:`numpy.ndarray`
314:     :param idx:
315:         Column index array.
316:     :type idx: :class:`numpy.ndarray`
317:     :param proj:
318:         Interpolation coefficients.
319:     :type proj: :class:`numpy.ndarray`
320: 
321:     :return:
322:         Left singular vectors.
323:     :rtype: :class:`numpy.ndarray`
324:     :return:
325:         Right singular vectors.
326:     :rtype: :class:`numpy.ndarray`
327:     :return:
328:         Singular values.
329:     :rtype: :class:`numpy.ndarray`
330:     '''
331:     B = np.asfortranarray(B)
332:     U, V, S, ier = _id.idd_id2svd(B, idx, proj)
333:     if ier:
334:         raise _RETCODE_ERROR
335:     return U, V, S
336: 
337: 
338: #------------------------------------------------------------------------------
339: # idd_snorm.f
340: #------------------------------------------------------------------------------
341: 
342: def idd_snorm(m, n, matvect, matvec, its=20):
343:     '''
344:     Estimate spectral norm of a real matrix by the randomized power method.
345: 
346:     :param m:
347:         Matrix row dimension.
348:     :type m: int
349:     :param n:
350:         Matrix column dimension.
351:     :type n: int
352:     :param matvect:
353:         Function to apply the matrix transpose to a vector, with call signature
354:         `y = matvect(x)`, where `x` and `y` are the input and output vectors,
355:         respectively.
356:     :type matvect: function
357:     :param matvec:
358:         Function to apply the matrix to a vector, with call signature
359:         `y = matvec(x)`, where `x` and `y` are the input and output vectors,
360:         respectively.
361:     :type matvec: function
362:     :param its:
363:         Number of power method iterations.
364:     :type its: int
365: 
366:     :return:
367:         Spectral norm estimate.
368:     :rtype: float
369:     '''
370:     snorm, v = _id.idd_snorm(m, n, matvect, matvec, its)
371:     return snorm
372: 
373: 
374: def idd_diffsnorm(m, n, matvect, matvect2, matvec, matvec2, its=20):
375:     '''
376:     Estimate spectral norm of the difference of two real matrices by the
377:     randomized power method.
378: 
379:     :param m:
380:         Matrix row dimension.
381:     :type m: int
382:     :param n:
383:         Matrix column dimension.
384:     :type n: int
385:     :param matvect:
386:         Function to apply the transpose of the first matrix to a vector, with
387:         call signature `y = matvect(x)`, where `x` and `y` are the input and
388:         output vectors, respectively.
389:     :type matvect: function
390:     :param matvect2:
391:         Function to apply the transpose of the second matrix to a vector, with
392:         call signature `y = matvect2(x)`, where `x` and `y` are the input and
393:         output vectors, respectively.
394:     :type matvect2: function
395:     :param matvec:
396:         Function to apply the first matrix to a vector, with call signature
397:         `y = matvec(x)`, where `x` and `y` are the input and output vectors,
398:         respectively.
399:     :type matvec: function
400:     :param matvec2:
401:         Function to apply the second matrix to a vector, with call signature
402:         `y = matvec2(x)`, where `x` and `y` are the input and output vectors,
403:         respectively.
404:     :type matvec2: function
405:     :param its:
406:         Number of power method iterations.
407:     :type its: int
408: 
409:     :return:
410:         Spectral norm estimate of matrix difference.
411:     :rtype: float
412:     '''
413:     return _id.idd_diffsnorm(m, n, matvect, matvect2, matvec, matvec2, its)
414: 
415: 
416: #------------------------------------------------------------------------------
417: # idd_svd.f
418: #------------------------------------------------------------------------------
419: 
420: def iddr_svd(A, k):
421:     '''
422:     Compute SVD of a real matrix to a specified rank.
423: 
424:     :param A:
425:         Matrix.
426:     :type A: :class:`numpy.ndarray`
427:     :param k:
428:         Rank of SVD.
429:     :type k: int
430: 
431:     :return:
432:         Left singular vectors.
433:     :rtype: :class:`numpy.ndarray`
434:     :return:
435:         Right singular vectors.
436:     :rtype: :class:`numpy.ndarray`
437:     :return:
438:         Singular values.
439:     :rtype: :class:`numpy.ndarray`
440:     '''
441:     A = np.asfortranarray(A)
442:     U, V, S, ier = _id.iddr_svd(A, k)
443:     if ier:
444:         raise _RETCODE_ERROR
445:     return U, V, S
446: 
447: 
448: def iddp_svd(eps, A):
449:     '''
450:     Compute SVD of a real matrix to a specified relative precision.
451: 
452:     :param eps:
453:         Relative precision.
454:     :type eps: float
455:     :param A:
456:         Matrix.
457:     :type A: :class:`numpy.ndarray`
458: 
459:     :return:
460:         Left singular vectors.
461:     :rtype: :class:`numpy.ndarray`
462:     :return:
463:         Right singular vectors.
464:     :rtype: :class:`numpy.ndarray`
465:     :return:
466:         Singular values.
467:     :rtype: :class:`numpy.ndarray`
468:     '''
469:     A = np.asfortranarray(A)
470:     m, n = A.shape
471:     k, iU, iV, iS, w, ier = _id.iddp_svd(eps, A)
472:     if ier:
473:         raise _RETCODE_ERROR
474:     U = w[iU-1:iU+m*k-1].reshape((m, k), order='F')
475:     V = w[iV-1:iV+n*k-1].reshape((n, k), order='F')
476:     S = w[iS-1:iS+k-1]
477:     return U, V, S
478: 
479: 
480: #------------------------------------------------------------------------------
481: # iddp_aid.f
482: #------------------------------------------------------------------------------
483: 
484: def iddp_aid(eps, A):
485:     '''
486:     Compute ID of a real matrix to a specified relative precision using random
487:     sampling.
488: 
489:     :param eps:
490:         Relative precision.
491:     :type eps: float
492:     :param A:
493:         Matrix.
494:     :type A: :class:`numpy.ndarray`
495: 
496:     :return:
497:         Rank of ID.
498:     :rtype: int
499:     :return:
500:         Column index array.
501:     :rtype: :class:`numpy.ndarray`
502:     :return:
503:         Interpolation coefficients.
504:     :rtype: :class:`numpy.ndarray`
505:     '''
506:     A = np.asfortranarray(A)
507:     m, n = A.shape
508:     n2, w = idd_frmi(m)
509:     proj = np.empty(n*(2*n2 + 1) + n2 + 1, order='F')
510:     k, idx, proj = _id.iddp_aid(eps, A, w, proj)
511:     proj = proj[:k*(n-k)].reshape((k, n-k), order='F')
512:     return k, idx, proj
513: 
514: 
515: def idd_estrank(eps, A):
516:     '''
517:     Estimate rank of a real matrix to a specified relative precision using
518:     random sampling.
519: 
520:     The output rank is typically about 8 higher than the actual rank.
521: 
522:     :param eps:
523:         Relative precision.
524:     :type eps: float
525:     :param A:
526:         Matrix.
527:     :type A: :class:`numpy.ndarray`
528: 
529:     :return:
530:         Rank estimate.
531:     :rtype: int
532:     '''
533:     A = np.asfortranarray(A)
534:     m, n = A.shape
535:     n2, w = idd_frmi(m)
536:     ra = np.empty(n*n2 + (n + 1)*(n2 + 1), order='F')
537:     k, ra = _id.idd_estrank(eps, A, w, ra)
538:     return k
539: 
540: 
541: #------------------------------------------------------------------------------
542: # iddp_asvd.f
543: #------------------------------------------------------------------------------
544: 
545: def iddp_asvd(eps, A):
546:     '''
547:     Compute SVD of a real matrix to a specified relative precision using random
548:     sampling.
549: 
550:     :param eps:
551:         Relative precision.
552:     :type eps: float
553:     :param A:
554:         Matrix.
555:     :type A: :class:`numpy.ndarray`
556: 
557:     :return:
558:         Left singular vectors.
559:     :rtype: :class:`numpy.ndarray`
560:     :return:
561:         Right singular vectors.
562:     :rtype: :class:`numpy.ndarray`
563:     :return:
564:         Singular values.
565:     :rtype: :class:`numpy.ndarray`
566:     '''
567:     A = np.asfortranarray(A)
568:     m, n = A.shape
569:     n2, winit = _id.idd_frmi(m)
570:     w = np.empty(
571:         max((min(m, n) + 1)*(3*m + 5*n + 1) + 25*min(m, n)**2,
572:             (2*n + 1)*(n2 + 1)),
573:         order='F')
574:     k, iU, iV, iS, w, ier = _id.iddp_asvd(eps, A, winit, w)
575:     if ier:
576:         raise _RETCODE_ERROR
577:     U = w[iU-1:iU+m*k-1].reshape((m, k), order='F')
578:     V = w[iV-1:iV+n*k-1].reshape((n, k), order='F')
579:     S = w[iS-1:iS+k-1]
580:     return U, V, S
581: 
582: 
583: #------------------------------------------------------------------------------
584: # iddp_rid.f
585: #------------------------------------------------------------------------------
586: 
587: def iddp_rid(eps, m, n, matvect):
588:     '''
589:     Compute ID of a real matrix to a specified relative precision using random
590:     matrix-vector multiplication.
591: 
592:     :param eps:
593:         Relative precision.
594:     :type eps: float
595:     :param m:
596:         Matrix row dimension.
597:     :type m: int
598:     :param n:
599:         Matrix column dimension.
600:     :type n: int
601:     :param matvect:
602:         Function to apply the matrix transpose to a vector, with call signature
603:         `y = matvect(x)`, where `x` and `y` are the input and output vectors,
604:         respectively.
605:     :type matvect: function
606: 
607:     :return:
608:         Rank of ID.
609:     :rtype: int
610:     :return:
611:         Column index array.
612:     :rtype: :class:`numpy.ndarray`
613:     :return:
614:         Interpolation coefficients.
615:     :rtype: :class:`numpy.ndarray`
616:     '''
617:     proj = np.empty(m + 1 + 2*n*(min(m, n) + 1), order='F')
618:     k, idx, proj, ier = _id.iddp_rid(eps, m, n, matvect, proj)
619:     if ier != 0:
620:         raise _RETCODE_ERROR
621:     proj = proj[:k*(n-k)].reshape((k, n-k), order='F')
622:     return k, idx, proj
623: 
624: 
625: def idd_findrank(eps, m, n, matvect):
626:     '''
627:     Estimate rank of a real matrix to a specified relative precision using
628:     random matrix-vector multiplication.
629: 
630:     :param eps:
631:         Relative precision.
632:     :type eps: float
633:     :param m:
634:         Matrix row dimension.
635:     :type m: int
636:     :param n:
637:         Matrix column dimension.
638:     :type n: int
639:     :param matvect:
640:         Function to apply the matrix transpose to a vector, with call signature
641:         `y = matvect(x)`, where `x` and `y` are the input and output vectors,
642:         respectively.
643:     :type matvect: function
644: 
645:     :return:
646:         Rank estimate.
647:     :rtype: int
648:     '''
649:     k, ra, ier = _id.idd_findrank(eps, m, n, matvect)
650:     if ier:
651:         raise _RETCODE_ERROR
652:     return k
653: 
654: 
655: #------------------------------------------------------------------------------
656: # iddp_rsvd.f
657: #------------------------------------------------------------------------------
658: 
659: def iddp_rsvd(eps, m, n, matvect, matvec):
660:     '''
661:     Compute SVD of a real matrix to a specified relative precision using random
662:     matrix-vector multiplication.
663: 
664:     :param eps:
665:         Relative precision.
666:     :type eps: float
667:     :param m:
668:         Matrix row dimension.
669:     :type m: int
670:     :param n:
671:         Matrix column dimension.
672:     :type n: int
673:     :param matvect:
674:         Function to apply the matrix transpose to a vector, with call signature
675:         `y = matvect(x)`, where `x` and `y` are the input and output vectors,
676:         respectively.
677:     :type matvect: function
678:     :param matvec:
679:         Function to apply the matrix to a vector, with call signature
680:         `y = matvec(x)`, where `x` and `y` are the input and output vectors,
681:         respectively.
682:     :type matvec: function
683: 
684:     :return:
685:         Left singular vectors.
686:     :rtype: :class:`numpy.ndarray`
687:     :return:
688:         Right singular vectors.
689:     :rtype: :class:`numpy.ndarray`
690:     :return:
691:         Singular values.
692:     :rtype: :class:`numpy.ndarray`
693:     '''
694:     k, iU, iV, iS, w, ier = _id.iddp_rsvd(eps, m, n, matvect, matvec)
695:     if ier:
696:         raise _RETCODE_ERROR
697:     U = w[iU-1:iU+m*k-1].reshape((m, k), order='F')
698:     V = w[iV-1:iV+n*k-1].reshape((n, k), order='F')
699:     S = w[iS-1:iS+k-1]
700:     return U, V, S
701: 
702: 
703: #------------------------------------------------------------------------------
704: # iddr_aid.f
705: #------------------------------------------------------------------------------
706: 
707: def iddr_aid(A, k):
708:     '''
709:     Compute ID of a real matrix to a specified rank using random sampling.
710: 
711:     :param A:
712:         Matrix.
713:     :type A: :class:`numpy.ndarray`
714:     :param k:
715:         Rank of ID.
716:     :type k: int
717: 
718:     :return:
719:         Column index array.
720:     :rtype: :class:`numpy.ndarray`
721:     :return:
722:         Interpolation coefficients.
723:     :rtype: :class:`numpy.ndarray`
724:     '''
725:     A = np.asfortranarray(A)
726:     m, n = A.shape
727:     w = iddr_aidi(m, n, k)
728:     idx, proj = _id.iddr_aid(A, k, w)
729:     if k == n:
730:         proj = np.array([], dtype='float64', order='F')
731:     else:
732:         proj = proj.reshape((k, n-k), order='F')
733:     return idx, proj
734: 
735: 
736: def iddr_aidi(m, n, k):
737:     '''
738:     Initialize array for :func:`iddr_aid`.
739: 
740:     :param m:
741:         Matrix row dimension.
742:     :type m: int
743:     :param n:
744:         Matrix column dimension.
745:     :type n: int
746:     :param k:
747:         Rank of ID.
748:     :type k: int
749: 
750:     :return:
751:         Initialization array to be used by :func:`iddr_aid`.
752:     :rtype: :class:`numpy.ndarray`
753:     '''
754:     return _id.iddr_aidi(m, n, k)
755: 
756: 
757: #------------------------------------------------------------------------------
758: # iddr_asvd.f
759: #------------------------------------------------------------------------------
760: 
761: def iddr_asvd(A, k):
762:     '''
763:     Compute SVD of a real matrix to a specified rank using random sampling.
764: 
765:     :param A:
766:         Matrix.
767:     :type A: :class:`numpy.ndarray`
768:     :param k:
769:         Rank of SVD.
770:     :type k: int
771: 
772:     :return:
773:         Left singular vectors.
774:     :rtype: :class:`numpy.ndarray`
775:     :return:
776:         Right singular vectors.
777:     :rtype: :class:`numpy.ndarray`
778:     :return:
779:         Singular values.
780:     :rtype: :class:`numpy.ndarray`
781:     '''
782:     A = np.asfortranarray(A)
783:     m, n = A.shape
784:     w = np.empty((2*k + 28)*m + (6*k + 21)*n + 25*k**2 + 100, order='F')
785:     w_ = iddr_aidi(m, n, k)
786:     w[:w_.size] = w_
787:     U, V, S, ier = _id.iddr_asvd(A, k, w)
788:     if ier != 0:
789:         raise _RETCODE_ERROR
790:     return U, V, S
791: 
792: 
793: #------------------------------------------------------------------------------
794: # iddr_rid.f
795: #------------------------------------------------------------------------------
796: 
797: def iddr_rid(m, n, matvect, k):
798:     '''
799:     Compute ID of a real matrix to a specified rank using random matrix-vector
800:     multiplication.
801: 
802:     :param m:
803:         Matrix row dimension.
804:     :type m: int
805:     :param n:
806:         Matrix column dimension.
807:     :type n: int
808:     :param matvect:
809:         Function to apply the matrix transpose to a vector, with call signature
810:         `y = matvect(x)`, where `x` and `y` are the input and output vectors,
811:         respectively.
812:     :type matvect: function
813:     :param k:
814:         Rank of ID.
815:     :type k: int
816: 
817:     :return:
818:         Column index array.
819:     :rtype: :class:`numpy.ndarray`
820:     :return:
821:         Interpolation coefficients.
822:     :rtype: :class:`numpy.ndarray`
823:     '''
824:     idx, proj = _id.iddr_rid(m, n, matvect, k)
825:     proj = proj[:k*(n-k)].reshape((k, n-k), order='F')
826:     return idx, proj
827: 
828: 
829: #------------------------------------------------------------------------------
830: # iddr_rsvd.f
831: #------------------------------------------------------------------------------
832: 
833: def iddr_rsvd(m, n, matvect, matvec, k):
834:     '''
835:     Compute SVD of a real matrix to a specified rank using random matrix-vector
836:     multiplication.
837: 
838:     :param m:
839:         Matrix row dimension.
840:     :type m: int
841:     :param n:
842:         Matrix column dimension.
843:     :type n: int
844:     :param matvect:
845:         Function to apply the matrix transpose to a vector, with call signature
846:         `y = matvect(x)`, where `x` and `y` are the input and output vectors,
847:         respectively.
848:     :type matvect: function
849:     :param matvec:
850:         Function to apply the matrix to a vector, with call signature
851:         `y = matvec(x)`, where `x` and `y` are the input and output vectors,
852:         respectively.
853:     :type matvec: function
854:     :param k:
855:         Rank of SVD.
856:     :type k: int
857: 
858:     :return:
859:         Left singular vectors.
860:     :rtype: :class:`numpy.ndarray`
861:     :return:
862:         Right singular vectors.
863:     :rtype: :class:`numpy.ndarray`
864:     :return:
865:         Singular values.
866:     :rtype: :class:`numpy.ndarray`
867:     '''
868:     U, V, S, ier = _id.iddr_rsvd(m, n, matvect, matvec, k)
869:     if ier != 0:
870:         raise _RETCODE_ERROR
871:     return U, V, S
872: 
873: 
874: #------------------------------------------------------------------------------
875: # idz_frm.f
876: #------------------------------------------------------------------------------
877: 
878: def idz_frm(n, w, x):
879:     '''
880:     Transform complex vector via a composition of Rokhlin's random transform,
881:     random subselection, and an FFT.
882: 
883:     In contrast to :func:`idz_sfrm`, this routine works best when the length of
884:     the transformed vector is the power-of-two integer output by
885:     :func:`idz_frmi`, or when the length is not specified but instead
886:     determined a posteriori from the output. The returned transformed vector is
887:     randomly permuted.
888: 
889:     :param n:
890:         Greatest power-of-two integer satisfying `n <= x.size` as obtained from
891:         :func:`idz_frmi`; `n` is also the length of the output vector.
892:     :type n: int
893:     :param w:
894:         Initialization array constructed by :func:`idz_frmi`.
895:     :type w: :class:`numpy.ndarray`
896:     :param x:
897:         Vector to be transformed.
898:     :type x: :class:`numpy.ndarray`
899: 
900:     :return:
901:         Transformed vector.
902:     :rtype: :class:`numpy.ndarray`
903:     '''
904:     return _id.idz_frm(n, w, x)
905: 
906: 
907: def idz_sfrm(l, n, w, x):
908:     '''
909:     Transform complex vector via a composition of Rokhlin's random transform,
910:     random subselection, and an FFT.
911: 
912:     In contrast to :func:`idz_frm`, this routine works best when the length of
913:     the transformed vector is known a priori.
914: 
915:     :param l:
916:         Length of transformed vector, satisfying `l <= n`.
917:     :type l: int
918:     :param n:
919:         Greatest power-of-two integer satisfying `n <= x.size` as obtained from
920:         :func:`idz_sfrmi`.
921:     :type n: int
922:     :param w:
923:         Initialization array constructed by :func:`idd_sfrmi`.
924:     :type w: :class:`numpy.ndarray`
925:     :param x:
926:         Vector to be transformed.
927:     :type x: :class:`numpy.ndarray`
928: 
929:     :return:
930:         Transformed vector.
931:     :rtype: :class:`numpy.ndarray`
932:     '''
933:     return _id.idz_sfrm(l, n, w, x)
934: 
935: 
936: def idz_frmi(m):
937:     '''
938:     Initialize data for :func:`idz_frm`.
939: 
940:     :param m:
941:         Length of vector to be transformed.
942:     :type m: int
943: 
944:     :return:
945:         Greatest power-of-two integer `n` satisfying `n <= m`.
946:     :rtype: int
947:     :return:
948:         Initialization array to be used by :func:`idz_frm`.
949:     :rtype: :class:`numpy.ndarray`
950:     '''
951:     return _id.idz_frmi(m)
952: 
953: 
954: def idz_sfrmi(l, m):
955:     '''
956:     Initialize data for :func:`idz_sfrm`.
957: 
958:     :param l:
959:         Length of output transformed vector.
960:     :type l: int
961:     :param m:
962:         Length of the vector to be transformed.
963:     :type m: int
964: 
965:     :return:
966:         Greatest power-of-two integer `n` satisfying `n <= m`.
967:     :rtype: int
968:     :return:
969:         Initialization array to be used by :func:`idz_sfrm`.
970:     :rtype: :class:`numpy.ndarray`
971:     '''
972:     return _id.idz_sfrmi(l, m)
973: 
974: 
975: #------------------------------------------------------------------------------
976: # idz_id.f
977: #------------------------------------------------------------------------------
978: 
979: def idzp_id(eps, A):
980:     '''
981:     Compute ID of a complex matrix to a specified relative precision.
982: 
983:     :param eps:
984:         Relative precision.
985:     :type eps: float
986:     :param A:
987:         Matrix.
988:     :type A: :class:`numpy.ndarray`
989: 
990:     :return:
991:         Rank of ID.
992:     :rtype: int
993:     :return:
994:         Column index array.
995:     :rtype: :class:`numpy.ndarray`
996:     :return:
997:         Interpolation coefficients.
998:     :rtype: :class:`numpy.ndarray`
999:     '''
1000:     A = np.asfortranarray(A)
1001:     k, idx, rnorms = _id.idzp_id(eps, A)
1002:     n = A.shape[1]
1003:     proj = A.T.ravel()[:k*(n-k)].reshape((k, n-k), order='F')
1004:     return k, idx, proj
1005: 
1006: 
1007: def idzr_id(A, k):
1008:     '''
1009:     Compute ID of a complex matrix to a specified rank.
1010: 
1011:     :param A:
1012:         Matrix.
1013:     :type A: :class:`numpy.ndarray`
1014:     :param k:
1015:         Rank of ID.
1016:     :type k: int
1017: 
1018:     :return:
1019:         Column index array.
1020:     :rtype: :class:`numpy.ndarray`
1021:     :return:
1022:         Interpolation coefficients.
1023:     :rtype: :class:`numpy.ndarray`
1024:     '''
1025:     A = np.asfortranarray(A)
1026:     idx, rnorms = _id.idzr_id(A, k)
1027:     n = A.shape[1]
1028:     proj = A.T.ravel()[:k*(n-k)].reshape((k, n-k), order='F')
1029:     return idx, proj
1030: 
1031: 
1032: def idz_reconid(B, idx, proj):
1033:     '''
1034:     Reconstruct matrix from complex ID.
1035: 
1036:     :param B:
1037:         Skeleton matrix.
1038:     :type B: :class:`numpy.ndarray`
1039:     :param idx:
1040:         Column index array.
1041:     :type idx: :class:`numpy.ndarray`
1042:     :param proj:
1043:         Interpolation coefficients.
1044:     :type proj: :class:`numpy.ndarray`
1045: 
1046:     :return:
1047:         Reconstructed matrix.
1048:     :rtype: :class:`numpy.ndarray`
1049:     '''
1050:     B = np.asfortranarray(B)
1051:     if proj.size > 0:
1052:         return _id.idz_reconid(B, idx, proj)
1053:     else:
1054:         return B[:, np.argsort(idx)]
1055: 
1056: 
1057: def idz_reconint(idx, proj):
1058:     '''
1059:     Reconstruct interpolation matrix from complex ID.
1060: 
1061:     :param idx:
1062:         Column index array.
1063:     :type idx: :class:`numpy.ndarray`
1064:     :param proj:
1065:         Interpolation coefficients.
1066:     :type proj: :class:`numpy.ndarray`
1067: 
1068:     :return:
1069:         Interpolation matrix.
1070:     :rtype: :class:`numpy.ndarray`
1071:     '''
1072:     return _id.idz_reconint(idx, proj)
1073: 
1074: 
1075: def idz_copycols(A, k, idx):
1076:     '''
1077:     Reconstruct skeleton matrix from complex ID.
1078: 
1079:     :param A:
1080:         Original matrix.
1081:     :type A: :class:`numpy.ndarray`
1082:     :param k:
1083:         Rank of ID.
1084:     :type k: int
1085:     :param idx:
1086:         Column index array.
1087:     :type idx: :class:`numpy.ndarray`
1088: 
1089:     :return:
1090:         Skeleton matrix.
1091:     :rtype: :class:`numpy.ndarray`
1092:     '''
1093:     A = np.asfortranarray(A)
1094:     return _id.idz_copycols(A, k, idx)
1095: 
1096: 
1097: #------------------------------------------------------------------------------
1098: # idz_id2svd.f
1099: #------------------------------------------------------------------------------
1100: 
1101: def idz_id2svd(B, idx, proj):
1102:     '''
1103:     Convert complex ID to SVD.
1104: 
1105:     :param B:
1106:         Skeleton matrix.
1107:     :type B: :class:`numpy.ndarray`
1108:     :param idx:
1109:         Column index array.
1110:     :type idx: :class:`numpy.ndarray`
1111:     :param proj:
1112:         Interpolation coefficients.
1113:     :type proj: :class:`numpy.ndarray`
1114: 
1115:     :return:
1116:         Left singular vectors.
1117:     :rtype: :class:`numpy.ndarray`
1118:     :return:
1119:         Right singular vectors.
1120:     :rtype: :class:`numpy.ndarray`
1121:     :return:
1122:         Singular values.
1123:     :rtype: :class:`numpy.ndarray`
1124:     '''
1125:     B = np.asfortranarray(B)
1126:     U, V, S, ier = _id.idz_id2svd(B, idx, proj)
1127:     if ier:
1128:         raise _RETCODE_ERROR
1129:     return U, V, S
1130: 
1131: 
1132: #------------------------------------------------------------------------------
1133: # idz_snorm.f
1134: #------------------------------------------------------------------------------
1135: 
1136: def idz_snorm(m, n, matveca, matvec, its=20):
1137:     '''
1138:     Estimate spectral norm of a complex matrix by the randomized power method.
1139: 
1140:     :param m:
1141:         Matrix row dimension.
1142:     :type m: int
1143:     :param n:
1144:         Matrix column dimension.
1145:     :type n: int
1146:     :param matveca:
1147:         Function to apply the matrix adjoint to a vector, with call signature
1148:         `y = matveca(x)`, where `x` and `y` are the input and output vectors,
1149:         respectively.
1150:     :type matveca: function
1151:     :param matvec:
1152:         Function to apply the matrix to a vector, with call signature
1153:         `y = matvec(x)`, where `x` and `y` are the input and output vectors,
1154:         respectively.
1155:     :type matvec: function
1156:     :param its:
1157:         Number of power method iterations.
1158:     :type its: int
1159: 
1160:     :return:
1161:         Spectral norm estimate.
1162:     :rtype: float
1163:     '''
1164:     snorm, v = _id.idz_snorm(m, n, matveca, matvec, its)
1165:     return snorm
1166: 
1167: 
1168: def idz_diffsnorm(m, n, matveca, matveca2, matvec, matvec2, its=20):
1169:     '''
1170:     Estimate spectral norm of the difference of two complex matrices by the
1171:     randomized power method.
1172: 
1173:     :param m:
1174:         Matrix row dimension.
1175:     :type m: int
1176:     :param n:
1177:         Matrix column dimension.
1178:     :type n: int
1179:     :param matveca:
1180:         Function to apply the adjoint of the first matrix to a vector, with
1181:         call signature `y = matveca(x)`, where `x` and `y` are the input and
1182:         output vectors, respectively.
1183:     :type matveca: function
1184:     :param matveca2:
1185:         Function to apply the adjoint of the second matrix to a vector, with
1186:         call signature `y = matveca2(x)`, where `x` and `y` are the input and
1187:         output vectors, respectively.
1188:     :type matveca2: function
1189:     :param matvec:
1190:         Function to apply the first matrix to a vector, with call signature
1191:         `y = matvec(x)`, where `x` and `y` are the input and output vectors,
1192:         respectively.
1193:     :type matvec: function
1194:     :param matvec2:
1195:         Function to apply the second matrix to a vector, with call signature
1196:         `y = matvec2(x)`, where `x` and `y` are the input and output vectors,
1197:         respectively.
1198:     :type matvec2: function
1199:     :param its:
1200:         Number of power method iterations.
1201:     :type its: int
1202: 
1203:     :return:
1204:         Spectral norm estimate of matrix difference.
1205:     :rtype: float
1206:     '''
1207:     return _id.idz_diffsnorm(m, n, matveca, matveca2, matvec, matvec2, its)
1208: 
1209: 
1210: #------------------------------------------------------------------------------
1211: # idz_svd.f
1212: #------------------------------------------------------------------------------
1213: 
1214: def idzr_svd(A, k):
1215:     '''
1216:     Compute SVD of a complex matrix to a specified rank.
1217: 
1218:     :param A:
1219:         Matrix.
1220:     :type A: :class:`numpy.ndarray`
1221:     :param k:
1222:         Rank of SVD.
1223:     :type k: int
1224: 
1225:     :return:
1226:         Left singular vectors.
1227:     :rtype: :class:`numpy.ndarray`
1228:     :return:
1229:         Right singular vectors.
1230:     :rtype: :class:`numpy.ndarray`
1231:     :return:
1232:         Singular values.
1233:     :rtype: :class:`numpy.ndarray`
1234:     '''
1235:     A = np.asfortranarray(A)
1236:     U, V, S, ier = _id.idzr_svd(A, k)
1237:     if ier:
1238:         raise _RETCODE_ERROR
1239:     return U, V, S
1240: 
1241: 
1242: def idzp_svd(eps, A):
1243:     '''
1244:     Compute SVD of a complex matrix to a specified relative precision.
1245: 
1246:     :param eps:
1247:         Relative precision.
1248:     :type eps: float
1249:     :param A:
1250:         Matrix.
1251:     :type A: :class:`numpy.ndarray`
1252: 
1253:     :return:
1254:         Left singular vectors.
1255:     :rtype: :class:`numpy.ndarray`
1256:     :return:
1257:         Right singular vectors.
1258:     :rtype: :class:`numpy.ndarray`
1259:     :return:
1260:         Singular values.
1261:     :rtype: :class:`numpy.ndarray`
1262:     '''
1263:     A = np.asfortranarray(A)
1264:     m, n = A.shape
1265:     k, iU, iV, iS, w, ier = _id.idzp_svd(eps, A)
1266:     if ier:
1267:         raise _RETCODE_ERROR
1268:     U = w[iU-1:iU+m*k-1].reshape((m, k), order='F')
1269:     V = w[iV-1:iV+n*k-1].reshape((n, k), order='F')
1270:     S = w[iS-1:iS+k-1]
1271:     return U, V, S
1272: 
1273: 
1274: #------------------------------------------------------------------------------
1275: # idzp_aid.f
1276: #------------------------------------------------------------------------------
1277: 
1278: def idzp_aid(eps, A):
1279:     '''
1280:     Compute ID of a complex matrix to a specified relative precision using
1281:     random sampling.
1282: 
1283:     :param eps:
1284:         Relative precision.
1285:     :type eps: float
1286:     :param A:
1287:         Matrix.
1288:     :type A: :class:`numpy.ndarray`
1289: 
1290:     :return:
1291:         Rank of ID.
1292:     :rtype: int
1293:     :return:
1294:         Column index array.
1295:     :rtype: :class:`numpy.ndarray`
1296:     :return:
1297:         Interpolation coefficients.
1298:     :rtype: :class:`numpy.ndarray`
1299:     '''
1300:     A = np.asfortranarray(A)
1301:     m, n = A.shape
1302:     n2, w = idz_frmi(m)
1303:     proj = np.empty(n*(2*n2 + 1) + n2 + 1, dtype='complex128', order='F')
1304:     k, idx, proj = _id.idzp_aid(eps, A, w, proj)
1305:     proj = proj[:k*(n-k)].reshape((k, n-k), order='F')
1306:     return k, idx, proj
1307: 
1308: 
1309: def idz_estrank(eps, A):
1310:     '''
1311:     Estimate rank of a complex matrix to a specified relative precision using
1312:     random sampling.
1313: 
1314:     The output rank is typically about 8 higher than the actual rank.
1315: 
1316:     :param eps:
1317:         Relative precision.
1318:     :type eps: float
1319:     :param A:
1320:         Matrix.
1321:     :type A: :class:`numpy.ndarray`
1322: 
1323:     :return:
1324:         Rank estimate.
1325:     :rtype: int
1326:     '''
1327:     A = np.asfortranarray(A)
1328:     m, n = A.shape
1329:     n2, w = idz_frmi(m)
1330:     ra = np.empty(n*n2 + (n + 1)*(n2 + 1), dtype='complex128', order='F')
1331:     k, ra = _id.idz_estrank(eps, A, w, ra)
1332:     return k
1333: 
1334: 
1335: #------------------------------------------------------------------------------
1336: # idzp_asvd.f
1337: #------------------------------------------------------------------------------
1338: 
1339: def idzp_asvd(eps, A):
1340:     '''
1341:     Compute SVD of a complex matrix to a specified relative precision using
1342:     random sampling.
1343: 
1344:     :param eps:
1345:         Relative precision.
1346:     :type eps: float
1347:     :param A:
1348:         Matrix.
1349:     :type A: :class:`numpy.ndarray`
1350: 
1351:     :return:
1352:         Left singular vectors.
1353:     :rtype: :class:`numpy.ndarray`
1354:     :return:
1355:         Right singular vectors.
1356:     :rtype: :class:`numpy.ndarray`
1357:     :return:
1358:         Singular values.
1359:     :rtype: :class:`numpy.ndarray`
1360:     '''
1361:     A = np.asfortranarray(A)
1362:     m, n = A.shape
1363:     n2, winit = _id.idz_frmi(m)
1364:     w = np.empty(
1365:         max((min(m, n) + 1)*(3*m + 5*n + 11) + 8*min(m, n)**2,
1366:             (2*n + 1)*(n2 + 1)),
1367:         dtype=np.complex128, order='F')
1368:     k, iU, iV, iS, w, ier = _id.idzp_asvd(eps, A, winit, w)
1369:     if ier:
1370:         raise _RETCODE_ERROR
1371:     U = w[iU-1:iU+m*k-1].reshape((m, k), order='F')
1372:     V = w[iV-1:iV+n*k-1].reshape((n, k), order='F')
1373:     S = w[iS-1:iS+k-1]
1374:     return U, V, S
1375: 
1376: 
1377: #------------------------------------------------------------------------------
1378: # idzp_rid.f
1379: #------------------------------------------------------------------------------
1380: 
1381: def idzp_rid(eps, m, n, matveca):
1382:     '''
1383:     Compute ID of a complex matrix to a specified relative precision using
1384:     random matrix-vector multiplication.
1385: 
1386:     :param eps:
1387:         Relative precision.
1388:     :type eps: float
1389:     :param m:
1390:         Matrix row dimension.
1391:     :type m: int
1392:     :param n:
1393:         Matrix column dimension.
1394:     :type n: int
1395:     :param matveca:
1396:         Function to apply the matrix adjoint to a vector, with call signature
1397:         `y = matveca(x)`, where `x` and `y` are the input and output vectors,
1398:         respectively.
1399:     :type matveca: function
1400: 
1401:     :return:
1402:         Rank of ID.
1403:     :rtype: int
1404:     :return:
1405:         Column index array.
1406:     :rtype: :class:`numpy.ndarray`
1407:     :return:
1408:         Interpolation coefficients.
1409:     :rtype: :class:`numpy.ndarray`
1410:     '''
1411:     proj = np.empty(
1412:         m + 1 + 2*n*(min(m, n) + 1),
1413:         dtype=np.complex128, order='F')
1414:     k, idx, proj, ier = _id.idzp_rid(eps, m, n, matveca, proj)
1415:     if ier:
1416:         raise _RETCODE_ERROR
1417:     proj = proj[:k*(n-k)].reshape((k, n-k), order='F')
1418:     return k, idx, proj
1419: 
1420: 
1421: def idz_findrank(eps, m, n, matveca):
1422:     '''
1423:     Estimate rank of a complex matrix to a specified relative precision using
1424:     random matrix-vector multiplication.
1425: 
1426:     :param eps:
1427:         Relative precision.
1428:     :type eps: float
1429:     :param m:
1430:         Matrix row dimension.
1431:     :type m: int
1432:     :param n:
1433:         Matrix column dimension.
1434:     :type n: int
1435:     :param matveca:
1436:         Function to apply the matrix adjoint to a vector, with call signature
1437:         `y = matveca(x)`, where `x` and `y` are the input and output vectors,
1438:         respectively.
1439:     :type matveca: function
1440: 
1441:     :return:
1442:         Rank estimate.
1443:     :rtype: int
1444:     '''
1445:     k, ra, ier = _id.idz_findrank(eps, m, n, matveca)
1446:     if ier:
1447:         raise _RETCODE_ERROR
1448:     return k
1449: 
1450: 
1451: #------------------------------------------------------------------------------
1452: # idzp_rsvd.f
1453: #------------------------------------------------------------------------------
1454: 
1455: def idzp_rsvd(eps, m, n, matveca, matvec):
1456:     '''
1457:     Compute SVD of a complex matrix to a specified relative precision using
1458:     random matrix-vector multiplication.
1459: 
1460:     :param eps:
1461:         Relative precision.
1462:     :type eps: float
1463:     :param m:
1464:         Matrix row dimension.
1465:     :type m: int
1466:     :param n:
1467:         Matrix column dimension.
1468:     :type n: int
1469:     :param matveca:
1470:         Function to apply the matrix adjoint to a vector, with call signature
1471:         `y = matveca(x)`, where `x` and `y` are the input and output vectors,
1472:         respectively.
1473:     :type matveca: function
1474:     :param matvec:
1475:         Function to apply the matrix to a vector, with call signature
1476:         `y = matvec(x)`, where `x` and `y` are the input and output vectors,
1477:         respectively.
1478:     :type matvec: function
1479: 
1480:     :return:
1481:         Left singular vectors.
1482:     :rtype: :class:`numpy.ndarray`
1483:     :return:
1484:         Right singular vectors.
1485:     :rtype: :class:`numpy.ndarray`
1486:     :return:
1487:         Singular values.
1488:     :rtype: :class:`numpy.ndarray`
1489:     '''
1490:     k, iU, iV, iS, w, ier = _id.idzp_rsvd(eps, m, n, matveca, matvec)
1491:     if ier:
1492:         raise _RETCODE_ERROR
1493:     U = w[iU-1:iU+m*k-1].reshape((m, k), order='F')
1494:     V = w[iV-1:iV+n*k-1].reshape((n, k), order='F')
1495:     S = w[iS-1:iS+k-1]
1496:     return U, V, S
1497: 
1498: 
1499: #------------------------------------------------------------------------------
1500: # idzr_aid.f
1501: #------------------------------------------------------------------------------
1502: 
1503: def idzr_aid(A, k):
1504:     '''
1505:     Compute ID of a complex matrix to a specified rank using random sampling.
1506: 
1507:     :param A:
1508:         Matrix.
1509:     :type A: :class:`numpy.ndarray`
1510:     :param k:
1511:         Rank of ID.
1512:     :type k: int
1513: 
1514:     :return:
1515:         Column index array.
1516:     :rtype: :class:`numpy.ndarray`
1517:     :return:
1518:         Interpolation coefficients.
1519:     :rtype: :class:`numpy.ndarray`
1520:     '''
1521:     A = np.asfortranarray(A)
1522:     m, n = A.shape
1523:     w = idzr_aidi(m, n, k)
1524:     idx, proj = _id.idzr_aid(A, k, w)
1525:     if k == n:
1526:         proj = np.array([], dtype='complex128', order='F')
1527:     else:
1528:         proj = proj.reshape((k, n-k), order='F')
1529:     return idx, proj
1530: 
1531: 
1532: def idzr_aidi(m, n, k):
1533:     '''
1534:     Initialize array for :func:`idzr_aid`.
1535: 
1536:     :param m:
1537:         Matrix row dimension.
1538:     :type m: int
1539:     :param n:
1540:         Matrix column dimension.
1541:     :type n: int
1542:     :param k:
1543:         Rank of ID.
1544:     :type k: int
1545: 
1546:     :return:
1547:         Initialization array to be used by :func:`idzr_aid`.
1548:     :rtype: :class:`numpy.ndarray`
1549:     '''
1550:     return _id.idzr_aidi(m, n, k)
1551: 
1552: 
1553: #------------------------------------------------------------------------------
1554: # idzr_asvd.f
1555: #------------------------------------------------------------------------------
1556: 
1557: def idzr_asvd(A, k):
1558:     '''
1559:     Compute SVD of a complex matrix to a specified rank using random sampling.
1560: 
1561:     :param A:
1562:         Matrix.
1563:     :type A: :class:`numpy.ndarray`
1564:     :param k:
1565:         Rank of SVD.
1566:     :type k: int
1567: 
1568:     :return:
1569:         Left singular vectors.
1570:     :rtype: :class:`numpy.ndarray`
1571:     :return:
1572:         Right singular vectors.
1573:     :rtype: :class:`numpy.ndarray`
1574:     :return:
1575:         Singular values.
1576:     :rtype: :class:`numpy.ndarray`
1577:     '''
1578:     A = np.asfortranarray(A)
1579:     m, n = A.shape
1580:     w = np.empty(
1581:         (2*k + 22)*m + (6*k + 21)*n + 8*k**2 + 10*k + 90,
1582:         dtype='complex128', order='F')
1583:     w_ = idzr_aidi(m, n, k)
1584:     w[:w_.size] = w_
1585:     U, V, S, ier = _id.idzr_asvd(A, k, w)
1586:     if ier:
1587:         raise _RETCODE_ERROR
1588:     return U, V, S
1589: 
1590: 
1591: #------------------------------------------------------------------------------
1592: # idzr_rid.f
1593: #------------------------------------------------------------------------------
1594: 
1595: def idzr_rid(m, n, matveca, k):
1596:     '''
1597:     Compute ID of a complex matrix to a specified rank using random
1598:     matrix-vector multiplication.
1599: 
1600:     :param m:
1601:         Matrix row dimension.
1602:     :type m: int
1603:     :param n:
1604:         Matrix column dimension.
1605:     :type n: int
1606:     :param matveca:
1607:         Function to apply the matrix adjoint to a vector, with call signature
1608:         `y = matveca(x)`, where `x` and `y` are the input and output vectors,
1609:         respectively.
1610:     :type matveca: function
1611:     :param k:
1612:         Rank of ID.
1613:     :type k: int
1614: 
1615:     :return:
1616:         Column index array.
1617:     :rtype: :class:`numpy.ndarray`
1618:     :return:
1619:         Interpolation coefficients.
1620:     :rtype: :class:`numpy.ndarray`
1621:     '''
1622:     idx, proj = _id.idzr_rid(m, n, matveca, k)
1623:     proj = proj[:k*(n-k)].reshape((k, n-k), order='F')
1624:     return idx, proj
1625: 
1626: 
1627: #------------------------------------------------------------------------------
1628: # idzr_rsvd.f
1629: #------------------------------------------------------------------------------
1630: 
1631: def idzr_rsvd(m, n, matveca, matvec, k):
1632:     '''
1633:     Compute SVD of a complex matrix to a specified rank using random
1634:     matrix-vector multiplication.
1635: 
1636:     :param m:
1637:         Matrix row dimension.
1638:     :type m: int
1639:     :param n:
1640:         Matrix column dimension.
1641:     :type n: int
1642:     :param matveca:
1643:         Function to apply the matrix adjoint to a vector, with call signature
1644:         `y = matveca(x)`, where `x` and `y` are the input and output vectors,
1645:         respectively.
1646:     :type matveca: function
1647:     :param matvec:
1648:         Function to apply the matrix to a vector, with call signature
1649:         `y = matvec(x)`, where `x` and `y` are the input and output vectors,
1650:         respectively.
1651:     :type matvec: function
1652:     :param k:
1653:         Rank of SVD.
1654:     :type k: int
1655: 
1656:     :return:
1657:         Left singular vectors.
1658:     :rtype: :class:`numpy.ndarray`
1659:     :return:
1660:         Right singular vectors.
1661:     :rtype: :class:`numpy.ndarray`
1662:     :return:
1663:         Singular values.
1664:     :rtype: :class:`numpy.ndarray`
1665:     '''
1666:     U, V, S, ier = _id.idzr_rsvd(m, n, matveca, matvec, k)
1667:     if ier:
1668:         raise _RETCODE_ERROR
1669:     return U, V, S
1670: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_29778 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 32, (-1)), 'str', '\nDirect wrappers for Fortran `id_dist` backend.\n')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 34, 0))

# 'import scipy.linalg._interpolative' statement (line 34)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/linalg/')
import_29779 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 34, 0), 'scipy.linalg._interpolative')

if (type(import_29779) is not StypyTypeError):

    if (import_29779 != 'pyd_module'):
        __import__(import_29779)
        sys_modules_29780 = sys.modules[import_29779]
        import_module(stypy.reporting.localization.Localization(__file__, 34, 0), '_id', sys_modules_29780.module_type_store, module_type_store)
    else:
        import scipy.linalg._interpolative as _id

        import_module(stypy.reporting.localization.Localization(__file__, 34, 0), '_id', scipy.linalg._interpolative, module_type_store)

else:
    # Assigning a type to the variable 'scipy.linalg._interpolative' (line 34)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 34, 0), 'scipy.linalg._interpolative', import_29779)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/linalg/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 35, 0))

# 'import numpy' statement (line 35)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/linalg/')
import_29781 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 35, 0), 'numpy')

if (type(import_29781) is not StypyTypeError):

    if (import_29781 != 'pyd_module'):
        __import__(import_29781)
        sys_modules_29782 = sys.modules[import_29781]
        import_module(stypy.reporting.localization.Localization(__file__, 35, 0), 'np', sys_modules_29782.module_type_store, module_type_store)
    else:
        import numpy as np

        import_module(stypy.reporting.localization.Localization(__file__, 35, 0), 'np', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 35)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 35, 0), 'numpy', import_29781)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/linalg/')


# Assigning a Call to a Name (line 37):

# Assigning a Call to a Name (line 37):

# Call to RuntimeError(...): (line 37)
# Processing the call arguments (line 37)
str_29784 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 30), 'str', 'nonzero return code')
# Processing the call keyword arguments (line 37)
kwargs_29785 = {}
# Getting the type of 'RuntimeError' (line 37)
RuntimeError_29783 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 17), 'RuntimeError', False)
# Calling RuntimeError(args, kwargs) (line 37)
RuntimeError_call_result_29786 = invoke(stypy.reporting.localization.Localization(__file__, 37, 17), RuntimeError_29783, *[str_29784], **kwargs_29785)

# Assigning a type to the variable '_RETCODE_ERROR' (line 37)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 37, 0), '_RETCODE_ERROR', RuntimeError_call_result_29786)

@norecursion
def id_srand(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'id_srand'
    module_type_store = module_type_store.open_function_context('id_srand', 44, 0, False)
    
    # Passed parameters checking function
    id_srand.stypy_localization = localization
    id_srand.stypy_type_of_self = None
    id_srand.stypy_type_store = module_type_store
    id_srand.stypy_function_name = 'id_srand'
    id_srand.stypy_param_names_list = ['n']
    id_srand.stypy_varargs_param_name = None
    id_srand.stypy_kwargs_param_name = None
    id_srand.stypy_call_defaults = defaults
    id_srand.stypy_call_varargs = varargs
    id_srand.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'id_srand', ['n'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'id_srand', localization, ['n'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'id_srand(...)' code ##################

    str_29787 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, (-1)), 'str', '\n    Generate standard uniform pseudorandom numbers via a very efficient lagged\n    Fibonacci method.\n\n    :param n:\n        Number of pseudorandom numbers to generate.\n    :type n: int\n\n    :return:\n        Pseudorandom numbers.\n    :rtype: :class:`numpy.ndarray`\n    ')
    
    # Call to id_srand(...): (line 57)
    # Processing the call arguments (line 57)
    # Getting the type of 'n' (line 57)
    n_29790 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 24), 'n', False)
    # Processing the call keyword arguments (line 57)
    kwargs_29791 = {}
    # Getting the type of '_id' (line 57)
    _id_29788 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 11), '_id', False)
    # Obtaining the member 'id_srand' of a type (line 57)
    id_srand_29789 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 57, 11), _id_29788, 'id_srand')
    # Calling id_srand(args, kwargs) (line 57)
    id_srand_call_result_29792 = invoke(stypy.reporting.localization.Localization(__file__, 57, 11), id_srand_29789, *[n_29790], **kwargs_29791)
    
    # Assigning a type to the variable 'stypy_return_type' (line 57)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 57, 4), 'stypy_return_type', id_srand_call_result_29792)
    
    # ################# End of 'id_srand(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'id_srand' in the type store
    # Getting the type of 'stypy_return_type' (line 44)
    stypy_return_type_29793 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_29793)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'id_srand'
    return stypy_return_type_29793

# Assigning a type to the variable 'id_srand' (line 44)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 44, 0), 'id_srand', id_srand)

@norecursion
def id_srandi(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'id_srandi'
    module_type_store = module_type_store.open_function_context('id_srandi', 60, 0, False)
    
    # Passed parameters checking function
    id_srandi.stypy_localization = localization
    id_srandi.stypy_type_of_self = None
    id_srandi.stypy_type_store = module_type_store
    id_srandi.stypy_function_name = 'id_srandi'
    id_srandi.stypy_param_names_list = ['t']
    id_srandi.stypy_varargs_param_name = None
    id_srandi.stypy_kwargs_param_name = None
    id_srandi.stypy_call_defaults = defaults
    id_srandi.stypy_call_varargs = varargs
    id_srandi.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'id_srandi', ['t'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'id_srandi', localization, ['t'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'id_srandi(...)' code ##################

    str_29794 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, (-1)), 'str', '\n    Initialize seed values for :func:`id_srand` (any appropriately random\n    numbers will do).\n\n    :param t:\n        Array of 55 seed values.\n    :type t: :class:`numpy.ndarray`\n    ')
    
    # Assigning a Call to a Name (line 69):
    
    # Assigning a Call to a Name (line 69):
    
    # Call to asfortranarray(...): (line 69)
    # Processing the call arguments (line 69)
    # Getting the type of 't' (line 69)
    t_29797 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 26), 't', False)
    # Processing the call keyword arguments (line 69)
    kwargs_29798 = {}
    # Getting the type of 'np' (line 69)
    np_29795 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 8), 'np', False)
    # Obtaining the member 'asfortranarray' of a type (line 69)
    asfortranarray_29796 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 69, 8), np_29795, 'asfortranarray')
    # Calling asfortranarray(args, kwargs) (line 69)
    asfortranarray_call_result_29799 = invoke(stypy.reporting.localization.Localization(__file__, 69, 8), asfortranarray_29796, *[t_29797], **kwargs_29798)
    
    # Assigning a type to the variable 't' (line 69)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 69, 4), 't', asfortranarray_call_result_29799)
    
    # Call to id_srandi(...): (line 70)
    # Processing the call arguments (line 70)
    # Getting the type of 't' (line 70)
    t_29802 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 18), 't', False)
    # Processing the call keyword arguments (line 70)
    kwargs_29803 = {}
    # Getting the type of '_id' (line 70)
    _id_29800 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 4), '_id', False)
    # Obtaining the member 'id_srandi' of a type (line 70)
    id_srandi_29801 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 70, 4), _id_29800, 'id_srandi')
    # Calling id_srandi(args, kwargs) (line 70)
    id_srandi_call_result_29804 = invoke(stypy.reporting.localization.Localization(__file__, 70, 4), id_srandi_29801, *[t_29802], **kwargs_29803)
    
    
    # ################# End of 'id_srandi(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'id_srandi' in the type store
    # Getting the type of 'stypy_return_type' (line 60)
    stypy_return_type_29805 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_29805)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'id_srandi'
    return stypy_return_type_29805

# Assigning a type to the variable 'id_srandi' (line 60)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 60, 0), 'id_srandi', id_srandi)

@norecursion
def id_srando(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'id_srando'
    module_type_store = module_type_store.open_function_context('id_srando', 73, 0, False)
    
    # Passed parameters checking function
    id_srando.stypy_localization = localization
    id_srando.stypy_type_of_self = None
    id_srando.stypy_type_store = module_type_store
    id_srando.stypy_function_name = 'id_srando'
    id_srando.stypy_param_names_list = []
    id_srando.stypy_varargs_param_name = None
    id_srando.stypy_kwargs_param_name = None
    id_srando.stypy_call_defaults = defaults
    id_srando.stypy_call_varargs = varargs
    id_srando.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'id_srando', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'id_srando', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'id_srando(...)' code ##################

    str_29806 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 76, (-1)), 'str', '\n    Reset seed values to their original values.\n    ')
    
    # Call to id_srando(...): (line 77)
    # Processing the call keyword arguments (line 77)
    kwargs_29809 = {}
    # Getting the type of '_id' (line 77)
    _id_29807 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 4), '_id', False)
    # Obtaining the member 'id_srando' of a type (line 77)
    id_srando_29808 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 77, 4), _id_29807, 'id_srando')
    # Calling id_srando(args, kwargs) (line 77)
    id_srando_call_result_29810 = invoke(stypy.reporting.localization.Localization(__file__, 77, 4), id_srando_29808, *[], **kwargs_29809)
    
    
    # ################# End of 'id_srando(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'id_srando' in the type store
    # Getting the type of 'stypy_return_type' (line 73)
    stypy_return_type_29811 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_29811)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'id_srando'
    return stypy_return_type_29811

# Assigning a type to the variable 'id_srando' (line 73)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 73, 0), 'id_srando', id_srando)

@norecursion
def idd_frm(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'idd_frm'
    module_type_store = module_type_store.open_function_context('idd_frm', 84, 0, False)
    
    # Passed parameters checking function
    idd_frm.stypy_localization = localization
    idd_frm.stypy_type_of_self = None
    idd_frm.stypy_type_store = module_type_store
    idd_frm.stypy_function_name = 'idd_frm'
    idd_frm.stypy_param_names_list = ['n', 'w', 'x']
    idd_frm.stypy_varargs_param_name = None
    idd_frm.stypy_kwargs_param_name = None
    idd_frm.stypy_call_defaults = defaults
    idd_frm.stypy_call_varargs = varargs
    idd_frm.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'idd_frm', ['n', 'w', 'x'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'idd_frm', localization, ['n', 'w', 'x'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'idd_frm(...)' code ##################

    str_29812 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 109, (-1)), 'str', "\n    Transform real vector via a composition of Rokhlin's random transform,\n    random subselection, and an FFT.\n\n    In contrast to :func:`idd_sfrm`, this routine works best when the length of\n    the transformed vector is the power-of-two integer output by\n    :func:`idd_frmi`, or when the length is not specified but instead\n    determined a posteriori from the output. The returned transformed vector is\n    randomly permuted.\n\n    :param n:\n        Greatest power-of-two integer satisfying `n <= x.size` as obtained from\n        :func:`idd_frmi`; `n` is also the length of the output vector.\n    :type n: int\n    :param w:\n        Initialization array constructed by :func:`idd_frmi`.\n    :type w: :class:`numpy.ndarray`\n    :param x:\n        Vector to be transformed.\n    :type x: :class:`numpy.ndarray`\n\n    :return:\n        Transformed vector.\n    :rtype: :class:`numpy.ndarray`\n    ")
    
    # Call to idd_frm(...): (line 110)
    # Processing the call arguments (line 110)
    # Getting the type of 'n' (line 110)
    n_29815 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 23), 'n', False)
    # Getting the type of 'w' (line 110)
    w_29816 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 26), 'w', False)
    # Getting the type of 'x' (line 110)
    x_29817 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 29), 'x', False)
    # Processing the call keyword arguments (line 110)
    kwargs_29818 = {}
    # Getting the type of '_id' (line 110)
    _id_29813 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 11), '_id', False)
    # Obtaining the member 'idd_frm' of a type (line 110)
    idd_frm_29814 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 110, 11), _id_29813, 'idd_frm')
    # Calling idd_frm(args, kwargs) (line 110)
    idd_frm_call_result_29819 = invoke(stypy.reporting.localization.Localization(__file__, 110, 11), idd_frm_29814, *[n_29815, w_29816, x_29817], **kwargs_29818)
    
    # Assigning a type to the variable 'stypy_return_type' (line 110)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 110, 4), 'stypy_return_type', idd_frm_call_result_29819)
    
    # ################# End of 'idd_frm(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'idd_frm' in the type store
    # Getting the type of 'stypy_return_type' (line 84)
    stypy_return_type_29820 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_29820)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'idd_frm'
    return stypy_return_type_29820

# Assigning a type to the variable 'idd_frm' (line 84)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 84, 0), 'idd_frm', idd_frm)

@norecursion
def idd_sfrm(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'idd_sfrm'
    module_type_store = module_type_store.open_function_context('idd_sfrm', 113, 0, False)
    
    # Passed parameters checking function
    idd_sfrm.stypy_localization = localization
    idd_sfrm.stypy_type_of_self = None
    idd_sfrm.stypy_type_store = module_type_store
    idd_sfrm.stypy_function_name = 'idd_sfrm'
    idd_sfrm.stypy_param_names_list = ['l', 'n', 'w', 'x']
    idd_sfrm.stypy_varargs_param_name = None
    idd_sfrm.stypy_kwargs_param_name = None
    idd_sfrm.stypy_call_defaults = defaults
    idd_sfrm.stypy_call_varargs = varargs
    idd_sfrm.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'idd_sfrm', ['l', 'n', 'w', 'x'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'idd_sfrm', localization, ['l', 'n', 'w', 'x'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'idd_sfrm(...)' code ##################

    str_29821 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 138, (-1)), 'str', "\n    Transform real vector via a composition of Rokhlin's random transform,\n    random subselection, and an FFT.\n\n    In contrast to :func:`idd_frm`, this routine works best when the length of\n    the transformed vector is known a priori.\n\n    :param l:\n        Length of transformed vector, satisfying `l <= n`.\n    :type l: int\n    :param n:\n        Greatest power-of-two integer satisfying `n <= x.size` as obtained from\n        :func:`idd_sfrmi`.\n    :type n: int\n    :param w:\n        Initialization array constructed by :func:`idd_sfrmi`.\n    :type w: :class:`numpy.ndarray`\n    :param x:\n        Vector to be transformed.\n    :type x: :class:`numpy.ndarray`\n\n    :return:\n        Transformed vector.\n    :rtype: :class:`numpy.ndarray`\n    ")
    
    # Call to idd_sfrm(...): (line 139)
    # Processing the call arguments (line 139)
    # Getting the type of 'l' (line 139)
    l_29824 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 24), 'l', False)
    # Getting the type of 'n' (line 139)
    n_29825 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 27), 'n', False)
    # Getting the type of 'w' (line 139)
    w_29826 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 30), 'w', False)
    # Getting the type of 'x' (line 139)
    x_29827 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 33), 'x', False)
    # Processing the call keyword arguments (line 139)
    kwargs_29828 = {}
    # Getting the type of '_id' (line 139)
    _id_29822 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 11), '_id', False)
    # Obtaining the member 'idd_sfrm' of a type (line 139)
    idd_sfrm_29823 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 139, 11), _id_29822, 'idd_sfrm')
    # Calling idd_sfrm(args, kwargs) (line 139)
    idd_sfrm_call_result_29829 = invoke(stypy.reporting.localization.Localization(__file__, 139, 11), idd_sfrm_29823, *[l_29824, n_29825, w_29826, x_29827], **kwargs_29828)
    
    # Assigning a type to the variable 'stypy_return_type' (line 139)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 139, 4), 'stypy_return_type', idd_sfrm_call_result_29829)
    
    # ################# End of 'idd_sfrm(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'idd_sfrm' in the type store
    # Getting the type of 'stypy_return_type' (line 113)
    stypy_return_type_29830 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_29830)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'idd_sfrm'
    return stypy_return_type_29830

# Assigning a type to the variable 'idd_sfrm' (line 113)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 113, 0), 'idd_sfrm', idd_sfrm)

@norecursion
def idd_frmi(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'idd_frmi'
    module_type_store = module_type_store.open_function_context('idd_frmi', 142, 0, False)
    
    # Passed parameters checking function
    idd_frmi.stypy_localization = localization
    idd_frmi.stypy_type_of_self = None
    idd_frmi.stypy_type_store = module_type_store
    idd_frmi.stypy_function_name = 'idd_frmi'
    idd_frmi.stypy_param_names_list = ['m']
    idd_frmi.stypy_varargs_param_name = None
    idd_frmi.stypy_kwargs_param_name = None
    idd_frmi.stypy_call_defaults = defaults
    idd_frmi.stypy_call_varargs = varargs
    idd_frmi.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'idd_frmi', ['m'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'idd_frmi', localization, ['m'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'idd_frmi(...)' code ##################

    str_29831 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 156, (-1)), 'str', '\n    Initialize data for :func:`idd_frm`.\n\n    :param m:\n        Length of vector to be transformed.\n    :type m: int\n\n    :return:\n        Greatest power-of-two integer `n` satisfying `n <= m`.\n    :rtype: int\n    :return:\n        Initialization array to be used by :func:`idd_frm`.\n    :rtype: :class:`numpy.ndarray`\n    ')
    
    # Call to idd_frmi(...): (line 157)
    # Processing the call arguments (line 157)
    # Getting the type of 'm' (line 157)
    m_29834 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 24), 'm', False)
    # Processing the call keyword arguments (line 157)
    kwargs_29835 = {}
    # Getting the type of '_id' (line 157)
    _id_29832 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 11), '_id', False)
    # Obtaining the member 'idd_frmi' of a type (line 157)
    idd_frmi_29833 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 157, 11), _id_29832, 'idd_frmi')
    # Calling idd_frmi(args, kwargs) (line 157)
    idd_frmi_call_result_29836 = invoke(stypy.reporting.localization.Localization(__file__, 157, 11), idd_frmi_29833, *[m_29834], **kwargs_29835)
    
    # Assigning a type to the variable 'stypy_return_type' (line 157)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 157, 4), 'stypy_return_type', idd_frmi_call_result_29836)
    
    # ################# End of 'idd_frmi(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'idd_frmi' in the type store
    # Getting the type of 'stypy_return_type' (line 142)
    stypy_return_type_29837 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_29837)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'idd_frmi'
    return stypy_return_type_29837

# Assigning a type to the variable 'idd_frmi' (line 142)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 142, 0), 'idd_frmi', idd_frmi)

@norecursion
def idd_sfrmi(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'idd_sfrmi'
    module_type_store = module_type_store.open_function_context('idd_sfrmi', 160, 0, False)
    
    # Passed parameters checking function
    idd_sfrmi.stypy_localization = localization
    idd_sfrmi.stypy_type_of_self = None
    idd_sfrmi.stypy_type_store = module_type_store
    idd_sfrmi.stypy_function_name = 'idd_sfrmi'
    idd_sfrmi.stypy_param_names_list = ['l', 'm']
    idd_sfrmi.stypy_varargs_param_name = None
    idd_sfrmi.stypy_kwargs_param_name = None
    idd_sfrmi.stypy_call_defaults = defaults
    idd_sfrmi.stypy_call_varargs = varargs
    idd_sfrmi.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'idd_sfrmi', ['l', 'm'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'idd_sfrmi', localization, ['l', 'm'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'idd_sfrmi(...)' code ##################

    str_29838 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 177, (-1)), 'str', '\n    Initialize data for :func:`idd_sfrm`.\n\n    :param l:\n        Length of output transformed vector.\n    :type l: int\n    :param m:\n        Length of the vector to be transformed.\n    :type m: int\n\n    :return:\n        Greatest power-of-two integer `n` satisfying `n <= m`.\n    :rtype: int\n    :return:\n        Initialization array to be used by :func:`idd_sfrm`.\n    :rtype: :class:`numpy.ndarray`\n    ')
    
    # Call to idd_sfrmi(...): (line 178)
    # Processing the call arguments (line 178)
    # Getting the type of 'l' (line 178)
    l_29841 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 25), 'l', False)
    # Getting the type of 'm' (line 178)
    m_29842 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 28), 'm', False)
    # Processing the call keyword arguments (line 178)
    kwargs_29843 = {}
    # Getting the type of '_id' (line 178)
    _id_29839 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 11), '_id', False)
    # Obtaining the member 'idd_sfrmi' of a type (line 178)
    idd_sfrmi_29840 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 178, 11), _id_29839, 'idd_sfrmi')
    # Calling idd_sfrmi(args, kwargs) (line 178)
    idd_sfrmi_call_result_29844 = invoke(stypy.reporting.localization.Localization(__file__, 178, 11), idd_sfrmi_29840, *[l_29841, m_29842], **kwargs_29843)
    
    # Assigning a type to the variable 'stypy_return_type' (line 178)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 178, 4), 'stypy_return_type', idd_sfrmi_call_result_29844)
    
    # ################# End of 'idd_sfrmi(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'idd_sfrmi' in the type store
    # Getting the type of 'stypy_return_type' (line 160)
    stypy_return_type_29845 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_29845)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'idd_sfrmi'
    return stypy_return_type_29845

# Assigning a type to the variable 'idd_sfrmi' (line 160)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 160, 0), 'idd_sfrmi', idd_sfrmi)

@norecursion
def iddp_id(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'iddp_id'
    module_type_store = module_type_store.open_function_context('iddp_id', 185, 0, False)
    
    # Passed parameters checking function
    iddp_id.stypy_localization = localization
    iddp_id.stypy_type_of_self = None
    iddp_id.stypy_type_store = module_type_store
    iddp_id.stypy_function_name = 'iddp_id'
    iddp_id.stypy_param_names_list = ['eps', 'A']
    iddp_id.stypy_varargs_param_name = None
    iddp_id.stypy_kwargs_param_name = None
    iddp_id.stypy_call_defaults = defaults
    iddp_id.stypy_call_varargs = varargs
    iddp_id.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'iddp_id', ['eps', 'A'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'iddp_id', localization, ['eps', 'A'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'iddp_id(...)' code ##################

    str_29846 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 205, (-1)), 'str', '\n    Compute ID of a real matrix to a specified relative precision.\n\n    :param eps:\n        Relative precision.\n    :type eps: float\n    :param A:\n        Matrix.\n    :type A: :class:`numpy.ndarray`\n\n    :return:\n        Rank of ID.\n    :rtype: int\n    :return:\n        Column index array.\n    :rtype: :class:`numpy.ndarray`\n    :return:\n        Interpolation coefficients.\n    :rtype: :class:`numpy.ndarray`\n    ')
    
    # Assigning a Call to a Name (line 206):
    
    # Assigning a Call to a Name (line 206):
    
    # Call to asfortranarray(...): (line 206)
    # Processing the call arguments (line 206)
    # Getting the type of 'A' (line 206)
    A_29849 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 206, 26), 'A', False)
    # Processing the call keyword arguments (line 206)
    kwargs_29850 = {}
    # Getting the type of 'np' (line 206)
    np_29847 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 206, 8), 'np', False)
    # Obtaining the member 'asfortranarray' of a type (line 206)
    asfortranarray_29848 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 206, 8), np_29847, 'asfortranarray')
    # Calling asfortranarray(args, kwargs) (line 206)
    asfortranarray_call_result_29851 = invoke(stypy.reporting.localization.Localization(__file__, 206, 8), asfortranarray_29848, *[A_29849], **kwargs_29850)
    
    # Assigning a type to the variable 'A' (line 206)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 206, 4), 'A', asfortranarray_call_result_29851)
    
    # Assigning a Call to a Tuple (line 207):
    
    # Assigning a Subscript to a Name (line 207):
    
    # Obtaining the type of the subscript
    int_29852 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 207, 4), 'int')
    
    # Call to iddp_id(...): (line 207)
    # Processing the call arguments (line 207)
    # Getting the type of 'eps' (line 207)
    eps_29855 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 207, 33), 'eps', False)
    # Getting the type of 'A' (line 207)
    A_29856 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 207, 38), 'A', False)
    # Processing the call keyword arguments (line 207)
    kwargs_29857 = {}
    # Getting the type of '_id' (line 207)
    _id_29853 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 207, 21), '_id', False)
    # Obtaining the member 'iddp_id' of a type (line 207)
    iddp_id_29854 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 207, 21), _id_29853, 'iddp_id')
    # Calling iddp_id(args, kwargs) (line 207)
    iddp_id_call_result_29858 = invoke(stypy.reporting.localization.Localization(__file__, 207, 21), iddp_id_29854, *[eps_29855, A_29856], **kwargs_29857)
    
    # Obtaining the member '__getitem__' of a type (line 207)
    getitem___29859 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 207, 4), iddp_id_call_result_29858, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 207)
    subscript_call_result_29860 = invoke(stypy.reporting.localization.Localization(__file__, 207, 4), getitem___29859, int_29852)
    
    # Assigning a type to the variable 'tuple_var_assignment_29628' (line 207)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 207, 4), 'tuple_var_assignment_29628', subscript_call_result_29860)
    
    # Assigning a Subscript to a Name (line 207):
    
    # Obtaining the type of the subscript
    int_29861 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 207, 4), 'int')
    
    # Call to iddp_id(...): (line 207)
    # Processing the call arguments (line 207)
    # Getting the type of 'eps' (line 207)
    eps_29864 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 207, 33), 'eps', False)
    # Getting the type of 'A' (line 207)
    A_29865 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 207, 38), 'A', False)
    # Processing the call keyword arguments (line 207)
    kwargs_29866 = {}
    # Getting the type of '_id' (line 207)
    _id_29862 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 207, 21), '_id', False)
    # Obtaining the member 'iddp_id' of a type (line 207)
    iddp_id_29863 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 207, 21), _id_29862, 'iddp_id')
    # Calling iddp_id(args, kwargs) (line 207)
    iddp_id_call_result_29867 = invoke(stypy.reporting.localization.Localization(__file__, 207, 21), iddp_id_29863, *[eps_29864, A_29865], **kwargs_29866)
    
    # Obtaining the member '__getitem__' of a type (line 207)
    getitem___29868 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 207, 4), iddp_id_call_result_29867, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 207)
    subscript_call_result_29869 = invoke(stypy.reporting.localization.Localization(__file__, 207, 4), getitem___29868, int_29861)
    
    # Assigning a type to the variable 'tuple_var_assignment_29629' (line 207)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 207, 4), 'tuple_var_assignment_29629', subscript_call_result_29869)
    
    # Assigning a Subscript to a Name (line 207):
    
    # Obtaining the type of the subscript
    int_29870 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 207, 4), 'int')
    
    # Call to iddp_id(...): (line 207)
    # Processing the call arguments (line 207)
    # Getting the type of 'eps' (line 207)
    eps_29873 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 207, 33), 'eps', False)
    # Getting the type of 'A' (line 207)
    A_29874 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 207, 38), 'A', False)
    # Processing the call keyword arguments (line 207)
    kwargs_29875 = {}
    # Getting the type of '_id' (line 207)
    _id_29871 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 207, 21), '_id', False)
    # Obtaining the member 'iddp_id' of a type (line 207)
    iddp_id_29872 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 207, 21), _id_29871, 'iddp_id')
    # Calling iddp_id(args, kwargs) (line 207)
    iddp_id_call_result_29876 = invoke(stypy.reporting.localization.Localization(__file__, 207, 21), iddp_id_29872, *[eps_29873, A_29874], **kwargs_29875)
    
    # Obtaining the member '__getitem__' of a type (line 207)
    getitem___29877 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 207, 4), iddp_id_call_result_29876, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 207)
    subscript_call_result_29878 = invoke(stypy.reporting.localization.Localization(__file__, 207, 4), getitem___29877, int_29870)
    
    # Assigning a type to the variable 'tuple_var_assignment_29630' (line 207)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 207, 4), 'tuple_var_assignment_29630', subscript_call_result_29878)
    
    # Assigning a Name to a Name (line 207):
    # Getting the type of 'tuple_var_assignment_29628' (line 207)
    tuple_var_assignment_29628_29879 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 207, 4), 'tuple_var_assignment_29628')
    # Assigning a type to the variable 'k' (line 207)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 207, 4), 'k', tuple_var_assignment_29628_29879)
    
    # Assigning a Name to a Name (line 207):
    # Getting the type of 'tuple_var_assignment_29629' (line 207)
    tuple_var_assignment_29629_29880 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 207, 4), 'tuple_var_assignment_29629')
    # Assigning a type to the variable 'idx' (line 207)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 207, 7), 'idx', tuple_var_assignment_29629_29880)
    
    # Assigning a Name to a Name (line 207):
    # Getting the type of 'tuple_var_assignment_29630' (line 207)
    tuple_var_assignment_29630_29881 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 207, 4), 'tuple_var_assignment_29630')
    # Assigning a type to the variable 'rnorms' (line 207)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 207, 12), 'rnorms', tuple_var_assignment_29630_29881)
    
    # Assigning a Subscript to a Name (line 208):
    
    # Assigning a Subscript to a Name (line 208):
    
    # Obtaining the type of the subscript
    int_29882 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 208, 16), 'int')
    # Getting the type of 'A' (line 208)
    A_29883 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 208, 8), 'A')
    # Obtaining the member 'shape' of a type (line 208)
    shape_29884 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 208, 8), A_29883, 'shape')
    # Obtaining the member '__getitem__' of a type (line 208)
    getitem___29885 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 208, 8), shape_29884, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 208)
    subscript_call_result_29886 = invoke(stypy.reporting.localization.Localization(__file__, 208, 8), getitem___29885, int_29882)
    
    # Assigning a type to the variable 'n' (line 208)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 208, 4), 'n', subscript_call_result_29886)
    
    # Assigning a Call to a Name (line 209):
    
    # Assigning a Call to a Name (line 209):
    
    # Call to reshape(...): (line 209)
    # Processing the call arguments (line 209)
    
    # Obtaining an instance of the builtin type 'tuple' (line 209)
    tuple_29901 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 209, 42), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 209)
    # Adding element type (line 209)
    # Getting the type of 'k' (line 209)
    k_29902 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 42), 'k', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 209, 42), tuple_29901, k_29902)
    # Adding element type (line 209)
    # Getting the type of 'n' (line 209)
    n_29903 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 45), 'n', False)
    # Getting the type of 'k' (line 209)
    k_29904 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 47), 'k', False)
    # Applying the binary operator '-' (line 209)
    result_sub_29905 = python_operator(stypy.reporting.localization.Localization(__file__, 209, 45), '-', n_29903, k_29904)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 209, 42), tuple_29901, result_sub_29905)
    
    # Processing the call keyword arguments (line 209)
    str_29906 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 209, 57), 'str', 'F')
    keyword_29907 = str_29906
    kwargs_29908 = {'order': keyword_29907}
    
    # Obtaining the type of the subscript
    # Getting the type of 'k' (line 209)
    k_29887 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 24), 'k', False)
    # Getting the type of 'n' (line 209)
    n_29888 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 27), 'n', False)
    # Getting the type of 'k' (line 209)
    k_29889 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 29), 'k', False)
    # Applying the binary operator '-' (line 209)
    result_sub_29890 = python_operator(stypy.reporting.localization.Localization(__file__, 209, 27), '-', n_29888, k_29889)
    
    # Applying the binary operator '*' (line 209)
    result_mul_29891 = python_operator(stypy.reporting.localization.Localization(__file__, 209, 24), '*', k_29887, result_sub_29890)
    
    slice_29892 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 209, 11), None, result_mul_29891, None)
    
    # Call to ravel(...): (line 209)
    # Processing the call keyword arguments (line 209)
    kwargs_29896 = {}
    # Getting the type of 'A' (line 209)
    A_29893 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 11), 'A', False)
    # Obtaining the member 'T' of a type (line 209)
    T_29894 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 209, 11), A_29893, 'T')
    # Obtaining the member 'ravel' of a type (line 209)
    ravel_29895 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 209, 11), T_29894, 'ravel')
    # Calling ravel(args, kwargs) (line 209)
    ravel_call_result_29897 = invoke(stypy.reporting.localization.Localization(__file__, 209, 11), ravel_29895, *[], **kwargs_29896)
    
    # Obtaining the member '__getitem__' of a type (line 209)
    getitem___29898 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 209, 11), ravel_call_result_29897, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 209)
    subscript_call_result_29899 = invoke(stypy.reporting.localization.Localization(__file__, 209, 11), getitem___29898, slice_29892)
    
    # Obtaining the member 'reshape' of a type (line 209)
    reshape_29900 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 209, 11), subscript_call_result_29899, 'reshape')
    # Calling reshape(args, kwargs) (line 209)
    reshape_call_result_29909 = invoke(stypy.reporting.localization.Localization(__file__, 209, 11), reshape_29900, *[tuple_29901], **kwargs_29908)
    
    # Assigning a type to the variable 'proj' (line 209)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 209, 4), 'proj', reshape_call_result_29909)
    
    # Obtaining an instance of the builtin type 'tuple' (line 210)
    tuple_29910 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 210, 11), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 210)
    # Adding element type (line 210)
    # Getting the type of 'k' (line 210)
    k_29911 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 11), 'k')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 210, 11), tuple_29910, k_29911)
    # Adding element type (line 210)
    # Getting the type of 'idx' (line 210)
    idx_29912 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 14), 'idx')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 210, 11), tuple_29910, idx_29912)
    # Adding element type (line 210)
    # Getting the type of 'proj' (line 210)
    proj_29913 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 19), 'proj')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 210, 11), tuple_29910, proj_29913)
    
    # Assigning a type to the variable 'stypy_return_type' (line 210)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 210, 4), 'stypy_return_type', tuple_29910)
    
    # ################# End of 'iddp_id(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'iddp_id' in the type store
    # Getting the type of 'stypy_return_type' (line 185)
    stypy_return_type_29914 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_29914)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'iddp_id'
    return stypy_return_type_29914

# Assigning a type to the variable 'iddp_id' (line 185)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 185, 0), 'iddp_id', iddp_id)

@norecursion
def iddr_id(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'iddr_id'
    module_type_store = module_type_store.open_function_context('iddr_id', 213, 0, False)
    
    # Passed parameters checking function
    iddr_id.stypy_localization = localization
    iddr_id.stypy_type_of_self = None
    iddr_id.stypy_type_store = module_type_store
    iddr_id.stypy_function_name = 'iddr_id'
    iddr_id.stypy_param_names_list = ['A', 'k']
    iddr_id.stypy_varargs_param_name = None
    iddr_id.stypy_kwargs_param_name = None
    iddr_id.stypy_call_defaults = defaults
    iddr_id.stypy_call_varargs = varargs
    iddr_id.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'iddr_id', ['A', 'k'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'iddr_id', localization, ['A', 'k'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'iddr_id(...)' code ##################

    str_29915 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 230, (-1)), 'str', '\n    Compute ID of a real matrix to a specified rank.\n\n    :param A:\n        Matrix.\n    :type A: :class:`numpy.ndarray`\n    :param k:\n        Rank of ID.\n    :type k: int\n\n    :return:\n        Column index array.\n    :rtype: :class:`numpy.ndarray`\n    :return:\n        Interpolation coefficients.\n    :rtype: :class:`numpy.ndarray`\n    ')
    
    # Assigning a Call to a Name (line 231):
    
    # Assigning a Call to a Name (line 231):
    
    # Call to asfortranarray(...): (line 231)
    # Processing the call arguments (line 231)
    # Getting the type of 'A' (line 231)
    A_29918 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 231, 26), 'A', False)
    # Processing the call keyword arguments (line 231)
    kwargs_29919 = {}
    # Getting the type of 'np' (line 231)
    np_29916 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 231, 8), 'np', False)
    # Obtaining the member 'asfortranarray' of a type (line 231)
    asfortranarray_29917 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 231, 8), np_29916, 'asfortranarray')
    # Calling asfortranarray(args, kwargs) (line 231)
    asfortranarray_call_result_29920 = invoke(stypy.reporting.localization.Localization(__file__, 231, 8), asfortranarray_29917, *[A_29918], **kwargs_29919)
    
    # Assigning a type to the variable 'A' (line 231)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 231, 4), 'A', asfortranarray_call_result_29920)
    
    # Assigning a Call to a Tuple (line 232):
    
    # Assigning a Subscript to a Name (line 232):
    
    # Obtaining the type of the subscript
    int_29921 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 232, 4), 'int')
    
    # Call to iddr_id(...): (line 232)
    # Processing the call arguments (line 232)
    # Getting the type of 'A' (line 232)
    A_29924 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 30), 'A', False)
    # Getting the type of 'k' (line 232)
    k_29925 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 33), 'k', False)
    # Processing the call keyword arguments (line 232)
    kwargs_29926 = {}
    # Getting the type of '_id' (line 232)
    _id_29922 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 18), '_id', False)
    # Obtaining the member 'iddr_id' of a type (line 232)
    iddr_id_29923 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 232, 18), _id_29922, 'iddr_id')
    # Calling iddr_id(args, kwargs) (line 232)
    iddr_id_call_result_29927 = invoke(stypy.reporting.localization.Localization(__file__, 232, 18), iddr_id_29923, *[A_29924, k_29925], **kwargs_29926)
    
    # Obtaining the member '__getitem__' of a type (line 232)
    getitem___29928 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 232, 4), iddr_id_call_result_29927, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 232)
    subscript_call_result_29929 = invoke(stypy.reporting.localization.Localization(__file__, 232, 4), getitem___29928, int_29921)
    
    # Assigning a type to the variable 'tuple_var_assignment_29631' (line 232)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 232, 4), 'tuple_var_assignment_29631', subscript_call_result_29929)
    
    # Assigning a Subscript to a Name (line 232):
    
    # Obtaining the type of the subscript
    int_29930 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 232, 4), 'int')
    
    # Call to iddr_id(...): (line 232)
    # Processing the call arguments (line 232)
    # Getting the type of 'A' (line 232)
    A_29933 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 30), 'A', False)
    # Getting the type of 'k' (line 232)
    k_29934 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 33), 'k', False)
    # Processing the call keyword arguments (line 232)
    kwargs_29935 = {}
    # Getting the type of '_id' (line 232)
    _id_29931 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 18), '_id', False)
    # Obtaining the member 'iddr_id' of a type (line 232)
    iddr_id_29932 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 232, 18), _id_29931, 'iddr_id')
    # Calling iddr_id(args, kwargs) (line 232)
    iddr_id_call_result_29936 = invoke(stypy.reporting.localization.Localization(__file__, 232, 18), iddr_id_29932, *[A_29933, k_29934], **kwargs_29935)
    
    # Obtaining the member '__getitem__' of a type (line 232)
    getitem___29937 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 232, 4), iddr_id_call_result_29936, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 232)
    subscript_call_result_29938 = invoke(stypy.reporting.localization.Localization(__file__, 232, 4), getitem___29937, int_29930)
    
    # Assigning a type to the variable 'tuple_var_assignment_29632' (line 232)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 232, 4), 'tuple_var_assignment_29632', subscript_call_result_29938)
    
    # Assigning a Name to a Name (line 232):
    # Getting the type of 'tuple_var_assignment_29631' (line 232)
    tuple_var_assignment_29631_29939 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 4), 'tuple_var_assignment_29631')
    # Assigning a type to the variable 'idx' (line 232)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 232, 4), 'idx', tuple_var_assignment_29631_29939)
    
    # Assigning a Name to a Name (line 232):
    # Getting the type of 'tuple_var_assignment_29632' (line 232)
    tuple_var_assignment_29632_29940 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 4), 'tuple_var_assignment_29632')
    # Assigning a type to the variable 'rnorms' (line 232)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 232, 9), 'rnorms', tuple_var_assignment_29632_29940)
    
    # Assigning a Subscript to a Name (line 233):
    
    # Assigning a Subscript to a Name (line 233):
    
    # Obtaining the type of the subscript
    int_29941 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 233, 16), 'int')
    # Getting the type of 'A' (line 233)
    A_29942 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 233, 8), 'A')
    # Obtaining the member 'shape' of a type (line 233)
    shape_29943 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 233, 8), A_29942, 'shape')
    # Obtaining the member '__getitem__' of a type (line 233)
    getitem___29944 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 233, 8), shape_29943, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 233)
    subscript_call_result_29945 = invoke(stypy.reporting.localization.Localization(__file__, 233, 8), getitem___29944, int_29941)
    
    # Assigning a type to the variable 'n' (line 233)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 233, 4), 'n', subscript_call_result_29945)
    
    # Assigning a Call to a Name (line 234):
    
    # Assigning a Call to a Name (line 234):
    
    # Call to reshape(...): (line 234)
    # Processing the call arguments (line 234)
    
    # Obtaining an instance of the builtin type 'tuple' (line 234)
    tuple_29960 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 234, 42), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 234)
    # Adding element type (line 234)
    # Getting the type of 'k' (line 234)
    k_29961 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 234, 42), 'k', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 234, 42), tuple_29960, k_29961)
    # Adding element type (line 234)
    # Getting the type of 'n' (line 234)
    n_29962 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 234, 45), 'n', False)
    # Getting the type of 'k' (line 234)
    k_29963 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 234, 47), 'k', False)
    # Applying the binary operator '-' (line 234)
    result_sub_29964 = python_operator(stypy.reporting.localization.Localization(__file__, 234, 45), '-', n_29962, k_29963)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 234, 42), tuple_29960, result_sub_29964)
    
    # Processing the call keyword arguments (line 234)
    str_29965 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 234, 57), 'str', 'F')
    keyword_29966 = str_29965
    kwargs_29967 = {'order': keyword_29966}
    
    # Obtaining the type of the subscript
    # Getting the type of 'k' (line 234)
    k_29946 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 234, 24), 'k', False)
    # Getting the type of 'n' (line 234)
    n_29947 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 234, 27), 'n', False)
    # Getting the type of 'k' (line 234)
    k_29948 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 234, 29), 'k', False)
    # Applying the binary operator '-' (line 234)
    result_sub_29949 = python_operator(stypy.reporting.localization.Localization(__file__, 234, 27), '-', n_29947, k_29948)
    
    # Applying the binary operator '*' (line 234)
    result_mul_29950 = python_operator(stypy.reporting.localization.Localization(__file__, 234, 24), '*', k_29946, result_sub_29949)
    
    slice_29951 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 234, 11), None, result_mul_29950, None)
    
    # Call to ravel(...): (line 234)
    # Processing the call keyword arguments (line 234)
    kwargs_29955 = {}
    # Getting the type of 'A' (line 234)
    A_29952 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 234, 11), 'A', False)
    # Obtaining the member 'T' of a type (line 234)
    T_29953 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 234, 11), A_29952, 'T')
    # Obtaining the member 'ravel' of a type (line 234)
    ravel_29954 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 234, 11), T_29953, 'ravel')
    # Calling ravel(args, kwargs) (line 234)
    ravel_call_result_29956 = invoke(stypy.reporting.localization.Localization(__file__, 234, 11), ravel_29954, *[], **kwargs_29955)
    
    # Obtaining the member '__getitem__' of a type (line 234)
    getitem___29957 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 234, 11), ravel_call_result_29956, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 234)
    subscript_call_result_29958 = invoke(stypy.reporting.localization.Localization(__file__, 234, 11), getitem___29957, slice_29951)
    
    # Obtaining the member 'reshape' of a type (line 234)
    reshape_29959 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 234, 11), subscript_call_result_29958, 'reshape')
    # Calling reshape(args, kwargs) (line 234)
    reshape_call_result_29968 = invoke(stypy.reporting.localization.Localization(__file__, 234, 11), reshape_29959, *[tuple_29960], **kwargs_29967)
    
    # Assigning a type to the variable 'proj' (line 234)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 234, 4), 'proj', reshape_call_result_29968)
    
    # Obtaining an instance of the builtin type 'tuple' (line 235)
    tuple_29969 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 235, 11), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 235)
    # Adding element type (line 235)
    # Getting the type of 'idx' (line 235)
    idx_29970 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 11), 'idx')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 235, 11), tuple_29969, idx_29970)
    # Adding element type (line 235)
    # Getting the type of 'proj' (line 235)
    proj_29971 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 16), 'proj')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 235, 11), tuple_29969, proj_29971)
    
    # Assigning a type to the variable 'stypy_return_type' (line 235)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 235, 4), 'stypy_return_type', tuple_29969)
    
    # ################# End of 'iddr_id(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'iddr_id' in the type store
    # Getting the type of 'stypy_return_type' (line 213)
    stypy_return_type_29972 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_29972)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'iddr_id'
    return stypy_return_type_29972

# Assigning a type to the variable 'iddr_id' (line 213)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 213, 0), 'iddr_id', iddr_id)

@norecursion
def idd_reconid(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'idd_reconid'
    module_type_store = module_type_store.open_function_context('idd_reconid', 238, 0, False)
    
    # Passed parameters checking function
    idd_reconid.stypy_localization = localization
    idd_reconid.stypy_type_of_self = None
    idd_reconid.stypy_type_store = module_type_store
    idd_reconid.stypy_function_name = 'idd_reconid'
    idd_reconid.stypy_param_names_list = ['B', 'idx', 'proj']
    idd_reconid.stypy_varargs_param_name = None
    idd_reconid.stypy_kwargs_param_name = None
    idd_reconid.stypy_call_defaults = defaults
    idd_reconid.stypy_call_varargs = varargs
    idd_reconid.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'idd_reconid', ['B', 'idx', 'proj'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'idd_reconid', localization, ['B', 'idx', 'proj'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'idd_reconid(...)' code ##################

    str_29973 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 255, (-1)), 'str', '\n    Reconstruct matrix from real ID.\n\n    :param B:\n        Skeleton matrix.\n    :type B: :class:`numpy.ndarray`\n    :param idx:\n        Column index array.\n    :type idx: :class:`numpy.ndarray`\n    :param proj:\n        Interpolation coefficients.\n    :type proj: :class:`numpy.ndarray`\n\n    :return:\n        Reconstructed matrix.\n    :rtype: :class:`numpy.ndarray`\n    ')
    
    # Assigning a Call to a Name (line 256):
    
    # Assigning a Call to a Name (line 256):
    
    # Call to asfortranarray(...): (line 256)
    # Processing the call arguments (line 256)
    # Getting the type of 'B' (line 256)
    B_29976 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 256, 26), 'B', False)
    # Processing the call keyword arguments (line 256)
    kwargs_29977 = {}
    # Getting the type of 'np' (line 256)
    np_29974 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 256, 8), 'np', False)
    # Obtaining the member 'asfortranarray' of a type (line 256)
    asfortranarray_29975 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 256, 8), np_29974, 'asfortranarray')
    # Calling asfortranarray(args, kwargs) (line 256)
    asfortranarray_call_result_29978 = invoke(stypy.reporting.localization.Localization(__file__, 256, 8), asfortranarray_29975, *[B_29976], **kwargs_29977)
    
    # Assigning a type to the variable 'B' (line 256)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 256, 4), 'B', asfortranarray_call_result_29978)
    
    
    # Getting the type of 'proj' (line 257)
    proj_29979 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 257, 7), 'proj')
    # Obtaining the member 'size' of a type (line 257)
    size_29980 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 257, 7), proj_29979, 'size')
    int_29981 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 257, 19), 'int')
    # Applying the binary operator '>' (line 257)
    result_gt_29982 = python_operator(stypy.reporting.localization.Localization(__file__, 257, 7), '>', size_29980, int_29981)
    
    # Testing the type of an if condition (line 257)
    if_condition_29983 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 257, 4), result_gt_29982)
    # Assigning a type to the variable 'if_condition_29983' (line 257)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 257, 4), 'if_condition_29983', if_condition_29983)
    # SSA begins for if statement (line 257)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to idd_reconid(...): (line 258)
    # Processing the call arguments (line 258)
    # Getting the type of 'B' (line 258)
    B_29986 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 258, 31), 'B', False)
    # Getting the type of 'idx' (line 258)
    idx_29987 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 258, 34), 'idx', False)
    # Getting the type of 'proj' (line 258)
    proj_29988 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 258, 39), 'proj', False)
    # Processing the call keyword arguments (line 258)
    kwargs_29989 = {}
    # Getting the type of '_id' (line 258)
    _id_29984 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 258, 15), '_id', False)
    # Obtaining the member 'idd_reconid' of a type (line 258)
    idd_reconid_29985 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 258, 15), _id_29984, 'idd_reconid')
    # Calling idd_reconid(args, kwargs) (line 258)
    idd_reconid_call_result_29990 = invoke(stypy.reporting.localization.Localization(__file__, 258, 15), idd_reconid_29985, *[B_29986, idx_29987, proj_29988], **kwargs_29989)
    
    # Assigning a type to the variable 'stypy_return_type' (line 258)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 258, 8), 'stypy_return_type', idd_reconid_call_result_29990)
    # SSA branch for the else part of an if statement (line 257)
    module_type_store.open_ssa_branch('else')
    
    # Obtaining the type of the subscript
    slice_29991 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 260, 15), None, None, None)
    
    # Call to argsort(...): (line 260)
    # Processing the call arguments (line 260)
    # Getting the type of 'idx' (line 260)
    idx_29994 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 260, 31), 'idx', False)
    # Processing the call keyword arguments (line 260)
    kwargs_29995 = {}
    # Getting the type of 'np' (line 260)
    np_29992 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 260, 20), 'np', False)
    # Obtaining the member 'argsort' of a type (line 260)
    argsort_29993 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 260, 20), np_29992, 'argsort')
    # Calling argsort(args, kwargs) (line 260)
    argsort_call_result_29996 = invoke(stypy.reporting.localization.Localization(__file__, 260, 20), argsort_29993, *[idx_29994], **kwargs_29995)
    
    # Getting the type of 'B' (line 260)
    B_29997 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 260, 15), 'B')
    # Obtaining the member '__getitem__' of a type (line 260)
    getitem___29998 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 260, 15), B_29997, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 260)
    subscript_call_result_29999 = invoke(stypy.reporting.localization.Localization(__file__, 260, 15), getitem___29998, (slice_29991, argsort_call_result_29996))
    
    # Assigning a type to the variable 'stypy_return_type' (line 260)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 260, 8), 'stypy_return_type', subscript_call_result_29999)
    # SSA join for if statement (line 257)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'idd_reconid(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'idd_reconid' in the type store
    # Getting the type of 'stypy_return_type' (line 238)
    stypy_return_type_30000 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 238, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_30000)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'idd_reconid'
    return stypy_return_type_30000

# Assigning a type to the variable 'idd_reconid' (line 238)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 238, 0), 'idd_reconid', idd_reconid)

@norecursion
def idd_reconint(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'idd_reconint'
    module_type_store = module_type_store.open_function_context('idd_reconint', 263, 0, False)
    
    # Passed parameters checking function
    idd_reconint.stypy_localization = localization
    idd_reconint.stypy_type_of_self = None
    idd_reconint.stypy_type_store = module_type_store
    idd_reconint.stypy_function_name = 'idd_reconint'
    idd_reconint.stypy_param_names_list = ['idx', 'proj']
    idd_reconint.stypy_varargs_param_name = None
    idd_reconint.stypy_kwargs_param_name = None
    idd_reconint.stypy_call_defaults = defaults
    idd_reconint.stypy_call_varargs = varargs
    idd_reconint.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'idd_reconint', ['idx', 'proj'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'idd_reconint', localization, ['idx', 'proj'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'idd_reconint(...)' code ##################

    str_30001 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 277, (-1)), 'str', '\n    Reconstruct interpolation matrix from real ID.\n\n    :param idx:\n        Column index array.\n    :type idx: :class:`numpy.ndarray`\n    :param proj:\n        Interpolation coefficients.\n    :type proj: :class:`numpy.ndarray`\n\n    :return:\n        Interpolation matrix.\n    :rtype: :class:`numpy.ndarray`\n    ')
    
    # Call to idd_reconint(...): (line 278)
    # Processing the call arguments (line 278)
    # Getting the type of 'idx' (line 278)
    idx_30004 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 278, 28), 'idx', False)
    # Getting the type of 'proj' (line 278)
    proj_30005 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 278, 33), 'proj', False)
    # Processing the call keyword arguments (line 278)
    kwargs_30006 = {}
    # Getting the type of '_id' (line 278)
    _id_30002 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 278, 11), '_id', False)
    # Obtaining the member 'idd_reconint' of a type (line 278)
    idd_reconint_30003 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 278, 11), _id_30002, 'idd_reconint')
    # Calling idd_reconint(args, kwargs) (line 278)
    idd_reconint_call_result_30007 = invoke(stypy.reporting.localization.Localization(__file__, 278, 11), idd_reconint_30003, *[idx_30004, proj_30005], **kwargs_30006)
    
    # Assigning a type to the variable 'stypy_return_type' (line 278)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 278, 4), 'stypy_return_type', idd_reconint_call_result_30007)
    
    # ################# End of 'idd_reconint(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'idd_reconint' in the type store
    # Getting the type of 'stypy_return_type' (line 263)
    stypy_return_type_30008 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 263, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_30008)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'idd_reconint'
    return stypy_return_type_30008

# Assigning a type to the variable 'idd_reconint' (line 263)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 263, 0), 'idd_reconint', idd_reconint)

@norecursion
def idd_copycols(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'idd_copycols'
    module_type_store = module_type_store.open_function_context('idd_copycols', 281, 0, False)
    
    # Passed parameters checking function
    idd_copycols.stypy_localization = localization
    idd_copycols.stypy_type_of_self = None
    idd_copycols.stypy_type_store = module_type_store
    idd_copycols.stypy_function_name = 'idd_copycols'
    idd_copycols.stypy_param_names_list = ['A', 'k', 'idx']
    idd_copycols.stypy_varargs_param_name = None
    idd_copycols.stypy_kwargs_param_name = None
    idd_copycols.stypy_call_defaults = defaults
    idd_copycols.stypy_call_varargs = varargs
    idd_copycols.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'idd_copycols', ['A', 'k', 'idx'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'idd_copycols', localization, ['A', 'k', 'idx'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'idd_copycols(...)' code ##################

    str_30009 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 298, (-1)), 'str', '\n    Reconstruct skeleton matrix from real ID.\n\n    :param A:\n        Original matrix.\n    :type A: :class:`numpy.ndarray`\n    :param k:\n        Rank of ID.\n    :type k: int\n    :param idx:\n        Column index array.\n    :type idx: :class:`numpy.ndarray`\n\n    :return:\n        Skeleton matrix.\n    :rtype: :class:`numpy.ndarray`\n    ')
    
    # Assigning a Call to a Name (line 299):
    
    # Assigning a Call to a Name (line 299):
    
    # Call to asfortranarray(...): (line 299)
    # Processing the call arguments (line 299)
    # Getting the type of 'A' (line 299)
    A_30012 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 299, 26), 'A', False)
    # Processing the call keyword arguments (line 299)
    kwargs_30013 = {}
    # Getting the type of 'np' (line 299)
    np_30010 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 299, 8), 'np', False)
    # Obtaining the member 'asfortranarray' of a type (line 299)
    asfortranarray_30011 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 299, 8), np_30010, 'asfortranarray')
    # Calling asfortranarray(args, kwargs) (line 299)
    asfortranarray_call_result_30014 = invoke(stypy.reporting.localization.Localization(__file__, 299, 8), asfortranarray_30011, *[A_30012], **kwargs_30013)
    
    # Assigning a type to the variable 'A' (line 299)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 299, 4), 'A', asfortranarray_call_result_30014)
    
    # Call to idd_copycols(...): (line 300)
    # Processing the call arguments (line 300)
    # Getting the type of 'A' (line 300)
    A_30017 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 300, 28), 'A', False)
    # Getting the type of 'k' (line 300)
    k_30018 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 300, 31), 'k', False)
    # Getting the type of 'idx' (line 300)
    idx_30019 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 300, 34), 'idx', False)
    # Processing the call keyword arguments (line 300)
    kwargs_30020 = {}
    # Getting the type of '_id' (line 300)
    _id_30015 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 300, 11), '_id', False)
    # Obtaining the member 'idd_copycols' of a type (line 300)
    idd_copycols_30016 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 300, 11), _id_30015, 'idd_copycols')
    # Calling idd_copycols(args, kwargs) (line 300)
    idd_copycols_call_result_30021 = invoke(stypy.reporting.localization.Localization(__file__, 300, 11), idd_copycols_30016, *[A_30017, k_30018, idx_30019], **kwargs_30020)
    
    # Assigning a type to the variable 'stypy_return_type' (line 300)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 300, 4), 'stypy_return_type', idd_copycols_call_result_30021)
    
    # ################# End of 'idd_copycols(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'idd_copycols' in the type store
    # Getting the type of 'stypy_return_type' (line 281)
    stypy_return_type_30022 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 281, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_30022)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'idd_copycols'
    return stypy_return_type_30022

# Assigning a type to the variable 'idd_copycols' (line 281)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 281, 0), 'idd_copycols', idd_copycols)

@norecursion
def idd_id2svd(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'idd_id2svd'
    module_type_store = module_type_store.open_function_context('idd_id2svd', 307, 0, False)
    
    # Passed parameters checking function
    idd_id2svd.stypy_localization = localization
    idd_id2svd.stypy_type_of_self = None
    idd_id2svd.stypy_type_store = module_type_store
    idd_id2svd.stypy_function_name = 'idd_id2svd'
    idd_id2svd.stypy_param_names_list = ['B', 'idx', 'proj']
    idd_id2svd.stypy_varargs_param_name = None
    idd_id2svd.stypy_kwargs_param_name = None
    idd_id2svd.stypy_call_defaults = defaults
    idd_id2svd.stypy_call_varargs = varargs
    idd_id2svd.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'idd_id2svd', ['B', 'idx', 'proj'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'idd_id2svd', localization, ['B', 'idx', 'proj'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'idd_id2svd(...)' code ##################

    str_30023 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 330, (-1)), 'str', '\n    Convert real ID to SVD.\n\n    :param B:\n        Skeleton matrix.\n    :type B: :class:`numpy.ndarray`\n    :param idx:\n        Column index array.\n    :type idx: :class:`numpy.ndarray`\n    :param proj:\n        Interpolation coefficients.\n    :type proj: :class:`numpy.ndarray`\n\n    :return:\n        Left singular vectors.\n    :rtype: :class:`numpy.ndarray`\n    :return:\n        Right singular vectors.\n    :rtype: :class:`numpy.ndarray`\n    :return:\n        Singular values.\n    :rtype: :class:`numpy.ndarray`\n    ')
    
    # Assigning a Call to a Name (line 331):
    
    # Assigning a Call to a Name (line 331):
    
    # Call to asfortranarray(...): (line 331)
    # Processing the call arguments (line 331)
    # Getting the type of 'B' (line 331)
    B_30026 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 331, 26), 'B', False)
    # Processing the call keyword arguments (line 331)
    kwargs_30027 = {}
    # Getting the type of 'np' (line 331)
    np_30024 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 331, 8), 'np', False)
    # Obtaining the member 'asfortranarray' of a type (line 331)
    asfortranarray_30025 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 331, 8), np_30024, 'asfortranarray')
    # Calling asfortranarray(args, kwargs) (line 331)
    asfortranarray_call_result_30028 = invoke(stypy.reporting.localization.Localization(__file__, 331, 8), asfortranarray_30025, *[B_30026], **kwargs_30027)
    
    # Assigning a type to the variable 'B' (line 331)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 331, 4), 'B', asfortranarray_call_result_30028)
    
    # Assigning a Call to a Tuple (line 332):
    
    # Assigning a Subscript to a Name (line 332):
    
    # Obtaining the type of the subscript
    int_30029 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 332, 4), 'int')
    
    # Call to idd_id2svd(...): (line 332)
    # Processing the call arguments (line 332)
    # Getting the type of 'B' (line 332)
    B_30032 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 332, 34), 'B', False)
    # Getting the type of 'idx' (line 332)
    idx_30033 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 332, 37), 'idx', False)
    # Getting the type of 'proj' (line 332)
    proj_30034 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 332, 42), 'proj', False)
    # Processing the call keyword arguments (line 332)
    kwargs_30035 = {}
    # Getting the type of '_id' (line 332)
    _id_30030 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 332, 19), '_id', False)
    # Obtaining the member 'idd_id2svd' of a type (line 332)
    idd_id2svd_30031 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 332, 19), _id_30030, 'idd_id2svd')
    # Calling idd_id2svd(args, kwargs) (line 332)
    idd_id2svd_call_result_30036 = invoke(stypy.reporting.localization.Localization(__file__, 332, 19), idd_id2svd_30031, *[B_30032, idx_30033, proj_30034], **kwargs_30035)
    
    # Obtaining the member '__getitem__' of a type (line 332)
    getitem___30037 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 332, 4), idd_id2svd_call_result_30036, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 332)
    subscript_call_result_30038 = invoke(stypy.reporting.localization.Localization(__file__, 332, 4), getitem___30037, int_30029)
    
    # Assigning a type to the variable 'tuple_var_assignment_29633' (line 332)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 332, 4), 'tuple_var_assignment_29633', subscript_call_result_30038)
    
    # Assigning a Subscript to a Name (line 332):
    
    # Obtaining the type of the subscript
    int_30039 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 332, 4), 'int')
    
    # Call to idd_id2svd(...): (line 332)
    # Processing the call arguments (line 332)
    # Getting the type of 'B' (line 332)
    B_30042 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 332, 34), 'B', False)
    # Getting the type of 'idx' (line 332)
    idx_30043 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 332, 37), 'idx', False)
    # Getting the type of 'proj' (line 332)
    proj_30044 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 332, 42), 'proj', False)
    # Processing the call keyword arguments (line 332)
    kwargs_30045 = {}
    # Getting the type of '_id' (line 332)
    _id_30040 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 332, 19), '_id', False)
    # Obtaining the member 'idd_id2svd' of a type (line 332)
    idd_id2svd_30041 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 332, 19), _id_30040, 'idd_id2svd')
    # Calling idd_id2svd(args, kwargs) (line 332)
    idd_id2svd_call_result_30046 = invoke(stypy.reporting.localization.Localization(__file__, 332, 19), idd_id2svd_30041, *[B_30042, idx_30043, proj_30044], **kwargs_30045)
    
    # Obtaining the member '__getitem__' of a type (line 332)
    getitem___30047 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 332, 4), idd_id2svd_call_result_30046, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 332)
    subscript_call_result_30048 = invoke(stypy.reporting.localization.Localization(__file__, 332, 4), getitem___30047, int_30039)
    
    # Assigning a type to the variable 'tuple_var_assignment_29634' (line 332)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 332, 4), 'tuple_var_assignment_29634', subscript_call_result_30048)
    
    # Assigning a Subscript to a Name (line 332):
    
    # Obtaining the type of the subscript
    int_30049 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 332, 4), 'int')
    
    # Call to idd_id2svd(...): (line 332)
    # Processing the call arguments (line 332)
    # Getting the type of 'B' (line 332)
    B_30052 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 332, 34), 'B', False)
    # Getting the type of 'idx' (line 332)
    idx_30053 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 332, 37), 'idx', False)
    # Getting the type of 'proj' (line 332)
    proj_30054 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 332, 42), 'proj', False)
    # Processing the call keyword arguments (line 332)
    kwargs_30055 = {}
    # Getting the type of '_id' (line 332)
    _id_30050 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 332, 19), '_id', False)
    # Obtaining the member 'idd_id2svd' of a type (line 332)
    idd_id2svd_30051 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 332, 19), _id_30050, 'idd_id2svd')
    # Calling idd_id2svd(args, kwargs) (line 332)
    idd_id2svd_call_result_30056 = invoke(stypy.reporting.localization.Localization(__file__, 332, 19), idd_id2svd_30051, *[B_30052, idx_30053, proj_30054], **kwargs_30055)
    
    # Obtaining the member '__getitem__' of a type (line 332)
    getitem___30057 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 332, 4), idd_id2svd_call_result_30056, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 332)
    subscript_call_result_30058 = invoke(stypy.reporting.localization.Localization(__file__, 332, 4), getitem___30057, int_30049)
    
    # Assigning a type to the variable 'tuple_var_assignment_29635' (line 332)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 332, 4), 'tuple_var_assignment_29635', subscript_call_result_30058)
    
    # Assigning a Subscript to a Name (line 332):
    
    # Obtaining the type of the subscript
    int_30059 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 332, 4), 'int')
    
    # Call to idd_id2svd(...): (line 332)
    # Processing the call arguments (line 332)
    # Getting the type of 'B' (line 332)
    B_30062 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 332, 34), 'B', False)
    # Getting the type of 'idx' (line 332)
    idx_30063 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 332, 37), 'idx', False)
    # Getting the type of 'proj' (line 332)
    proj_30064 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 332, 42), 'proj', False)
    # Processing the call keyword arguments (line 332)
    kwargs_30065 = {}
    # Getting the type of '_id' (line 332)
    _id_30060 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 332, 19), '_id', False)
    # Obtaining the member 'idd_id2svd' of a type (line 332)
    idd_id2svd_30061 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 332, 19), _id_30060, 'idd_id2svd')
    # Calling idd_id2svd(args, kwargs) (line 332)
    idd_id2svd_call_result_30066 = invoke(stypy.reporting.localization.Localization(__file__, 332, 19), idd_id2svd_30061, *[B_30062, idx_30063, proj_30064], **kwargs_30065)
    
    # Obtaining the member '__getitem__' of a type (line 332)
    getitem___30067 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 332, 4), idd_id2svd_call_result_30066, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 332)
    subscript_call_result_30068 = invoke(stypy.reporting.localization.Localization(__file__, 332, 4), getitem___30067, int_30059)
    
    # Assigning a type to the variable 'tuple_var_assignment_29636' (line 332)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 332, 4), 'tuple_var_assignment_29636', subscript_call_result_30068)
    
    # Assigning a Name to a Name (line 332):
    # Getting the type of 'tuple_var_assignment_29633' (line 332)
    tuple_var_assignment_29633_30069 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 332, 4), 'tuple_var_assignment_29633')
    # Assigning a type to the variable 'U' (line 332)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 332, 4), 'U', tuple_var_assignment_29633_30069)
    
    # Assigning a Name to a Name (line 332):
    # Getting the type of 'tuple_var_assignment_29634' (line 332)
    tuple_var_assignment_29634_30070 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 332, 4), 'tuple_var_assignment_29634')
    # Assigning a type to the variable 'V' (line 332)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 332, 7), 'V', tuple_var_assignment_29634_30070)
    
    # Assigning a Name to a Name (line 332):
    # Getting the type of 'tuple_var_assignment_29635' (line 332)
    tuple_var_assignment_29635_30071 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 332, 4), 'tuple_var_assignment_29635')
    # Assigning a type to the variable 'S' (line 332)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 332, 10), 'S', tuple_var_assignment_29635_30071)
    
    # Assigning a Name to a Name (line 332):
    # Getting the type of 'tuple_var_assignment_29636' (line 332)
    tuple_var_assignment_29636_30072 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 332, 4), 'tuple_var_assignment_29636')
    # Assigning a type to the variable 'ier' (line 332)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 332, 13), 'ier', tuple_var_assignment_29636_30072)
    
    # Getting the type of 'ier' (line 333)
    ier_30073 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 333, 7), 'ier')
    # Testing the type of an if condition (line 333)
    if_condition_30074 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 333, 4), ier_30073)
    # Assigning a type to the variable 'if_condition_30074' (line 333)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 333, 4), 'if_condition_30074', if_condition_30074)
    # SSA begins for if statement (line 333)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # Getting the type of '_RETCODE_ERROR' (line 334)
    _RETCODE_ERROR_30075 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 334, 14), '_RETCODE_ERROR')
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 334, 8), _RETCODE_ERROR_30075, 'raise parameter', BaseException)
    # SSA join for if statement (line 333)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Obtaining an instance of the builtin type 'tuple' (line 335)
    tuple_30076 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 335, 11), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 335)
    # Adding element type (line 335)
    # Getting the type of 'U' (line 335)
    U_30077 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 335, 11), 'U')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 335, 11), tuple_30076, U_30077)
    # Adding element type (line 335)
    # Getting the type of 'V' (line 335)
    V_30078 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 335, 14), 'V')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 335, 11), tuple_30076, V_30078)
    # Adding element type (line 335)
    # Getting the type of 'S' (line 335)
    S_30079 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 335, 17), 'S')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 335, 11), tuple_30076, S_30079)
    
    # Assigning a type to the variable 'stypy_return_type' (line 335)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 335, 4), 'stypy_return_type', tuple_30076)
    
    # ################# End of 'idd_id2svd(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'idd_id2svd' in the type store
    # Getting the type of 'stypy_return_type' (line 307)
    stypy_return_type_30080 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 307, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_30080)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'idd_id2svd'
    return stypy_return_type_30080

# Assigning a type to the variable 'idd_id2svd' (line 307)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 307, 0), 'idd_id2svd', idd_id2svd)

@norecursion
def idd_snorm(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    int_30081 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 342, 41), 'int')
    defaults = [int_30081]
    # Create a new context for function 'idd_snorm'
    module_type_store = module_type_store.open_function_context('idd_snorm', 342, 0, False)
    
    # Passed parameters checking function
    idd_snorm.stypy_localization = localization
    idd_snorm.stypy_type_of_self = None
    idd_snorm.stypy_type_store = module_type_store
    idd_snorm.stypy_function_name = 'idd_snorm'
    idd_snorm.stypy_param_names_list = ['m', 'n', 'matvect', 'matvec', 'its']
    idd_snorm.stypy_varargs_param_name = None
    idd_snorm.stypy_kwargs_param_name = None
    idd_snorm.stypy_call_defaults = defaults
    idd_snorm.stypy_call_varargs = varargs
    idd_snorm.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'idd_snorm', ['m', 'n', 'matvect', 'matvec', 'its'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'idd_snorm', localization, ['m', 'n', 'matvect', 'matvec', 'its'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'idd_snorm(...)' code ##################

    str_30082 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 369, (-1)), 'str', '\n    Estimate spectral norm of a real matrix by the randomized power method.\n\n    :param m:\n        Matrix row dimension.\n    :type m: int\n    :param n:\n        Matrix column dimension.\n    :type n: int\n    :param matvect:\n        Function to apply the matrix transpose to a vector, with call signature\n        `y = matvect(x)`, where `x` and `y` are the input and output vectors,\n        respectively.\n    :type matvect: function\n    :param matvec:\n        Function to apply the matrix to a vector, with call signature\n        `y = matvec(x)`, where `x` and `y` are the input and output vectors,\n        respectively.\n    :type matvec: function\n    :param its:\n        Number of power method iterations.\n    :type its: int\n\n    :return:\n        Spectral norm estimate.\n    :rtype: float\n    ')
    
    # Assigning a Call to a Tuple (line 370):
    
    # Assigning a Subscript to a Name (line 370):
    
    # Obtaining the type of the subscript
    int_30083 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 370, 4), 'int')
    
    # Call to idd_snorm(...): (line 370)
    # Processing the call arguments (line 370)
    # Getting the type of 'm' (line 370)
    m_30086 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 370, 29), 'm', False)
    # Getting the type of 'n' (line 370)
    n_30087 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 370, 32), 'n', False)
    # Getting the type of 'matvect' (line 370)
    matvect_30088 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 370, 35), 'matvect', False)
    # Getting the type of 'matvec' (line 370)
    matvec_30089 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 370, 44), 'matvec', False)
    # Getting the type of 'its' (line 370)
    its_30090 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 370, 52), 'its', False)
    # Processing the call keyword arguments (line 370)
    kwargs_30091 = {}
    # Getting the type of '_id' (line 370)
    _id_30084 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 370, 15), '_id', False)
    # Obtaining the member 'idd_snorm' of a type (line 370)
    idd_snorm_30085 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 370, 15), _id_30084, 'idd_snorm')
    # Calling idd_snorm(args, kwargs) (line 370)
    idd_snorm_call_result_30092 = invoke(stypy.reporting.localization.Localization(__file__, 370, 15), idd_snorm_30085, *[m_30086, n_30087, matvect_30088, matvec_30089, its_30090], **kwargs_30091)
    
    # Obtaining the member '__getitem__' of a type (line 370)
    getitem___30093 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 370, 4), idd_snorm_call_result_30092, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 370)
    subscript_call_result_30094 = invoke(stypy.reporting.localization.Localization(__file__, 370, 4), getitem___30093, int_30083)
    
    # Assigning a type to the variable 'tuple_var_assignment_29637' (line 370)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 370, 4), 'tuple_var_assignment_29637', subscript_call_result_30094)
    
    # Assigning a Subscript to a Name (line 370):
    
    # Obtaining the type of the subscript
    int_30095 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 370, 4), 'int')
    
    # Call to idd_snorm(...): (line 370)
    # Processing the call arguments (line 370)
    # Getting the type of 'm' (line 370)
    m_30098 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 370, 29), 'm', False)
    # Getting the type of 'n' (line 370)
    n_30099 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 370, 32), 'n', False)
    # Getting the type of 'matvect' (line 370)
    matvect_30100 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 370, 35), 'matvect', False)
    # Getting the type of 'matvec' (line 370)
    matvec_30101 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 370, 44), 'matvec', False)
    # Getting the type of 'its' (line 370)
    its_30102 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 370, 52), 'its', False)
    # Processing the call keyword arguments (line 370)
    kwargs_30103 = {}
    # Getting the type of '_id' (line 370)
    _id_30096 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 370, 15), '_id', False)
    # Obtaining the member 'idd_snorm' of a type (line 370)
    idd_snorm_30097 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 370, 15), _id_30096, 'idd_snorm')
    # Calling idd_snorm(args, kwargs) (line 370)
    idd_snorm_call_result_30104 = invoke(stypy.reporting.localization.Localization(__file__, 370, 15), idd_snorm_30097, *[m_30098, n_30099, matvect_30100, matvec_30101, its_30102], **kwargs_30103)
    
    # Obtaining the member '__getitem__' of a type (line 370)
    getitem___30105 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 370, 4), idd_snorm_call_result_30104, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 370)
    subscript_call_result_30106 = invoke(stypy.reporting.localization.Localization(__file__, 370, 4), getitem___30105, int_30095)
    
    # Assigning a type to the variable 'tuple_var_assignment_29638' (line 370)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 370, 4), 'tuple_var_assignment_29638', subscript_call_result_30106)
    
    # Assigning a Name to a Name (line 370):
    # Getting the type of 'tuple_var_assignment_29637' (line 370)
    tuple_var_assignment_29637_30107 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 370, 4), 'tuple_var_assignment_29637')
    # Assigning a type to the variable 'snorm' (line 370)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 370, 4), 'snorm', tuple_var_assignment_29637_30107)
    
    # Assigning a Name to a Name (line 370):
    # Getting the type of 'tuple_var_assignment_29638' (line 370)
    tuple_var_assignment_29638_30108 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 370, 4), 'tuple_var_assignment_29638')
    # Assigning a type to the variable 'v' (line 370)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 370, 11), 'v', tuple_var_assignment_29638_30108)
    # Getting the type of 'snorm' (line 371)
    snorm_30109 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 371, 11), 'snorm')
    # Assigning a type to the variable 'stypy_return_type' (line 371)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 371, 4), 'stypy_return_type', snorm_30109)
    
    # ################# End of 'idd_snorm(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'idd_snorm' in the type store
    # Getting the type of 'stypy_return_type' (line 342)
    stypy_return_type_30110 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 342, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_30110)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'idd_snorm'
    return stypy_return_type_30110

# Assigning a type to the variable 'idd_snorm' (line 342)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 342, 0), 'idd_snorm', idd_snorm)

@norecursion
def idd_diffsnorm(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    int_30111 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 374, 64), 'int')
    defaults = [int_30111]
    # Create a new context for function 'idd_diffsnorm'
    module_type_store = module_type_store.open_function_context('idd_diffsnorm', 374, 0, False)
    
    # Passed parameters checking function
    idd_diffsnorm.stypy_localization = localization
    idd_diffsnorm.stypy_type_of_self = None
    idd_diffsnorm.stypy_type_store = module_type_store
    idd_diffsnorm.stypy_function_name = 'idd_diffsnorm'
    idd_diffsnorm.stypy_param_names_list = ['m', 'n', 'matvect', 'matvect2', 'matvec', 'matvec2', 'its']
    idd_diffsnorm.stypy_varargs_param_name = None
    idd_diffsnorm.stypy_kwargs_param_name = None
    idd_diffsnorm.stypy_call_defaults = defaults
    idd_diffsnorm.stypy_call_varargs = varargs
    idd_diffsnorm.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'idd_diffsnorm', ['m', 'n', 'matvect', 'matvect2', 'matvec', 'matvec2', 'its'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'idd_diffsnorm', localization, ['m', 'n', 'matvect', 'matvect2', 'matvec', 'matvec2', 'its'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'idd_diffsnorm(...)' code ##################

    str_30112 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 412, (-1)), 'str', '\n    Estimate spectral norm of the difference of two real matrices by the\n    randomized power method.\n\n    :param m:\n        Matrix row dimension.\n    :type m: int\n    :param n:\n        Matrix column dimension.\n    :type n: int\n    :param matvect:\n        Function to apply the transpose of the first matrix to a vector, with\n        call signature `y = matvect(x)`, where `x` and `y` are the input and\n        output vectors, respectively.\n    :type matvect: function\n    :param matvect2:\n        Function to apply the transpose of the second matrix to a vector, with\n        call signature `y = matvect2(x)`, where `x` and `y` are the input and\n        output vectors, respectively.\n    :type matvect2: function\n    :param matvec:\n        Function to apply the first matrix to a vector, with call signature\n        `y = matvec(x)`, where `x` and `y` are the input and output vectors,\n        respectively.\n    :type matvec: function\n    :param matvec2:\n        Function to apply the second matrix to a vector, with call signature\n        `y = matvec2(x)`, where `x` and `y` are the input and output vectors,\n        respectively.\n    :type matvec2: function\n    :param its:\n        Number of power method iterations.\n    :type its: int\n\n    :return:\n        Spectral norm estimate of matrix difference.\n    :rtype: float\n    ')
    
    # Call to idd_diffsnorm(...): (line 413)
    # Processing the call arguments (line 413)
    # Getting the type of 'm' (line 413)
    m_30115 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 413, 29), 'm', False)
    # Getting the type of 'n' (line 413)
    n_30116 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 413, 32), 'n', False)
    # Getting the type of 'matvect' (line 413)
    matvect_30117 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 413, 35), 'matvect', False)
    # Getting the type of 'matvect2' (line 413)
    matvect2_30118 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 413, 44), 'matvect2', False)
    # Getting the type of 'matvec' (line 413)
    matvec_30119 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 413, 54), 'matvec', False)
    # Getting the type of 'matvec2' (line 413)
    matvec2_30120 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 413, 62), 'matvec2', False)
    # Getting the type of 'its' (line 413)
    its_30121 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 413, 71), 'its', False)
    # Processing the call keyword arguments (line 413)
    kwargs_30122 = {}
    # Getting the type of '_id' (line 413)
    _id_30113 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 413, 11), '_id', False)
    # Obtaining the member 'idd_diffsnorm' of a type (line 413)
    idd_diffsnorm_30114 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 413, 11), _id_30113, 'idd_diffsnorm')
    # Calling idd_diffsnorm(args, kwargs) (line 413)
    idd_diffsnorm_call_result_30123 = invoke(stypy.reporting.localization.Localization(__file__, 413, 11), idd_diffsnorm_30114, *[m_30115, n_30116, matvect_30117, matvect2_30118, matvec_30119, matvec2_30120, its_30121], **kwargs_30122)
    
    # Assigning a type to the variable 'stypy_return_type' (line 413)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 413, 4), 'stypy_return_type', idd_diffsnorm_call_result_30123)
    
    # ################# End of 'idd_diffsnorm(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'idd_diffsnorm' in the type store
    # Getting the type of 'stypy_return_type' (line 374)
    stypy_return_type_30124 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 374, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_30124)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'idd_diffsnorm'
    return stypy_return_type_30124

# Assigning a type to the variable 'idd_diffsnorm' (line 374)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 374, 0), 'idd_diffsnorm', idd_diffsnorm)

@norecursion
def iddr_svd(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'iddr_svd'
    module_type_store = module_type_store.open_function_context('iddr_svd', 420, 0, False)
    
    # Passed parameters checking function
    iddr_svd.stypy_localization = localization
    iddr_svd.stypy_type_of_self = None
    iddr_svd.stypy_type_store = module_type_store
    iddr_svd.stypy_function_name = 'iddr_svd'
    iddr_svd.stypy_param_names_list = ['A', 'k']
    iddr_svd.stypy_varargs_param_name = None
    iddr_svd.stypy_kwargs_param_name = None
    iddr_svd.stypy_call_defaults = defaults
    iddr_svd.stypy_call_varargs = varargs
    iddr_svd.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'iddr_svd', ['A', 'k'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'iddr_svd', localization, ['A', 'k'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'iddr_svd(...)' code ##################

    str_30125 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 440, (-1)), 'str', '\n    Compute SVD of a real matrix to a specified rank.\n\n    :param A:\n        Matrix.\n    :type A: :class:`numpy.ndarray`\n    :param k:\n        Rank of SVD.\n    :type k: int\n\n    :return:\n        Left singular vectors.\n    :rtype: :class:`numpy.ndarray`\n    :return:\n        Right singular vectors.\n    :rtype: :class:`numpy.ndarray`\n    :return:\n        Singular values.\n    :rtype: :class:`numpy.ndarray`\n    ')
    
    # Assigning a Call to a Name (line 441):
    
    # Assigning a Call to a Name (line 441):
    
    # Call to asfortranarray(...): (line 441)
    # Processing the call arguments (line 441)
    # Getting the type of 'A' (line 441)
    A_30128 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 441, 26), 'A', False)
    # Processing the call keyword arguments (line 441)
    kwargs_30129 = {}
    # Getting the type of 'np' (line 441)
    np_30126 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 441, 8), 'np', False)
    # Obtaining the member 'asfortranarray' of a type (line 441)
    asfortranarray_30127 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 441, 8), np_30126, 'asfortranarray')
    # Calling asfortranarray(args, kwargs) (line 441)
    asfortranarray_call_result_30130 = invoke(stypy.reporting.localization.Localization(__file__, 441, 8), asfortranarray_30127, *[A_30128], **kwargs_30129)
    
    # Assigning a type to the variable 'A' (line 441)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 441, 4), 'A', asfortranarray_call_result_30130)
    
    # Assigning a Call to a Tuple (line 442):
    
    # Assigning a Subscript to a Name (line 442):
    
    # Obtaining the type of the subscript
    int_30131 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 442, 4), 'int')
    
    # Call to iddr_svd(...): (line 442)
    # Processing the call arguments (line 442)
    # Getting the type of 'A' (line 442)
    A_30134 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 442, 32), 'A', False)
    # Getting the type of 'k' (line 442)
    k_30135 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 442, 35), 'k', False)
    # Processing the call keyword arguments (line 442)
    kwargs_30136 = {}
    # Getting the type of '_id' (line 442)
    _id_30132 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 442, 19), '_id', False)
    # Obtaining the member 'iddr_svd' of a type (line 442)
    iddr_svd_30133 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 442, 19), _id_30132, 'iddr_svd')
    # Calling iddr_svd(args, kwargs) (line 442)
    iddr_svd_call_result_30137 = invoke(stypy.reporting.localization.Localization(__file__, 442, 19), iddr_svd_30133, *[A_30134, k_30135], **kwargs_30136)
    
    # Obtaining the member '__getitem__' of a type (line 442)
    getitem___30138 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 442, 4), iddr_svd_call_result_30137, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 442)
    subscript_call_result_30139 = invoke(stypy.reporting.localization.Localization(__file__, 442, 4), getitem___30138, int_30131)
    
    # Assigning a type to the variable 'tuple_var_assignment_29639' (line 442)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 442, 4), 'tuple_var_assignment_29639', subscript_call_result_30139)
    
    # Assigning a Subscript to a Name (line 442):
    
    # Obtaining the type of the subscript
    int_30140 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 442, 4), 'int')
    
    # Call to iddr_svd(...): (line 442)
    # Processing the call arguments (line 442)
    # Getting the type of 'A' (line 442)
    A_30143 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 442, 32), 'A', False)
    # Getting the type of 'k' (line 442)
    k_30144 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 442, 35), 'k', False)
    # Processing the call keyword arguments (line 442)
    kwargs_30145 = {}
    # Getting the type of '_id' (line 442)
    _id_30141 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 442, 19), '_id', False)
    # Obtaining the member 'iddr_svd' of a type (line 442)
    iddr_svd_30142 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 442, 19), _id_30141, 'iddr_svd')
    # Calling iddr_svd(args, kwargs) (line 442)
    iddr_svd_call_result_30146 = invoke(stypy.reporting.localization.Localization(__file__, 442, 19), iddr_svd_30142, *[A_30143, k_30144], **kwargs_30145)
    
    # Obtaining the member '__getitem__' of a type (line 442)
    getitem___30147 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 442, 4), iddr_svd_call_result_30146, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 442)
    subscript_call_result_30148 = invoke(stypy.reporting.localization.Localization(__file__, 442, 4), getitem___30147, int_30140)
    
    # Assigning a type to the variable 'tuple_var_assignment_29640' (line 442)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 442, 4), 'tuple_var_assignment_29640', subscript_call_result_30148)
    
    # Assigning a Subscript to a Name (line 442):
    
    # Obtaining the type of the subscript
    int_30149 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 442, 4), 'int')
    
    # Call to iddr_svd(...): (line 442)
    # Processing the call arguments (line 442)
    # Getting the type of 'A' (line 442)
    A_30152 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 442, 32), 'A', False)
    # Getting the type of 'k' (line 442)
    k_30153 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 442, 35), 'k', False)
    # Processing the call keyword arguments (line 442)
    kwargs_30154 = {}
    # Getting the type of '_id' (line 442)
    _id_30150 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 442, 19), '_id', False)
    # Obtaining the member 'iddr_svd' of a type (line 442)
    iddr_svd_30151 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 442, 19), _id_30150, 'iddr_svd')
    # Calling iddr_svd(args, kwargs) (line 442)
    iddr_svd_call_result_30155 = invoke(stypy.reporting.localization.Localization(__file__, 442, 19), iddr_svd_30151, *[A_30152, k_30153], **kwargs_30154)
    
    # Obtaining the member '__getitem__' of a type (line 442)
    getitem___30156 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 442, 4), iddr_svd_call_result_30155, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 442)
    subscript_call_result_30157 = invoke(stypy.reporting.localization.Localization(__file__, 442, 4), getitem___30156, int_30149)
    
    # Assigning a type to the variable 'tuple_var_assignment_29641' (line 442)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 442, 4), 'tuple_var_assignment_29641', subscript_call_result_30157)
    
    # Assigning a Subscript to a Name (line 442):
    
    # Obtaining the type of the subscript
    int_30158 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 442, 4), 'int')
    
    # Call to iddr_svd(...): (line 442)
    # Processing the call arguments (line 442)
    # Getting the type of 'A' (line 442)
    A_30161 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 442, 32), 'A', False)
    # Getting the type of 'k' (line 442)
    k_30162 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 442, 35), 'k', False)
    # Processing the call keyword arguments (line 442)
    kwargs_30163 = {}
    # Getting the type of '_id' (line 442)
    _id_30159 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 442, 19), '_id', False)
    # Obtaining the member 'iddr_svd' of a type (line 442)
    iddr_svd_30160 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 442, 19), _id_30159, 'iddr_svd')
    # Calling iddr_svd(args, kwargs) (line 442)
    iddr_svd_call_result_30164 = invoke(stypy.reporting.localization.Localization(__file__, 442, 19), iddr_svd_30160, *[A_30161, k_30162], **kwargs_30163)
    
    # Obtaining the member '__getitem__' of a type (line 442)
    getitem___30165 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 442, 4), iddr_svd_call_result_30164, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 442)
    subscript_call_result_30166 = invoke(stypy.reporting.localization.Localization(__file__, 442, 4), getitem___30165, int_30158)
    
    # Assigning a type to the variable 'tuple_var_assignment_29642' (line 442)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 442, 4), 'tuple_var_assignment_29642', subscript_call_result_30166)
    
    # Assigning a Name to a Name (line 442):
    # Getting the type of 'tuple_var_assignment_29639' (line 442)
    tuple_var_assignment_29639_30167 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 442, 4), 'tuple_var_assignment_29639')
    # Assigning a type to the variable 'U' (line 442)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 442, 4), 'U', tuple_var_assignment_29639_30167)
    
    # Assigning a Name to a Name (line 442):
    # Getting the type of 'tuple_var_assignment_29640' (line 442)
    tuple_var_assignment_29640_30168 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 442, 4), 'tuple_var_assignment_29640')
    # Assigning a type to the variable 'V' (line 442)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 442, 7), 'V', tuple_var_assignment_29640_30168)
    
    # Assigning a Name to a Name (line 442):
    # Getting the type of 'tuple_var_assignment_29641' (line 442)
    tuple_var_assignment_29641_30169 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 442, 4), 'tuple_var_assignment_29641')
    # Assigning a type to the variable 'S' (line 442)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 442, 10), 'S', tuple_var_assignment_29641_30169)
    
    # Assigning a Name to a Name (line 442):
    # Getting the type of 'tuple_var_assignment_29642' (line 442)
    tuple_var_assignment_29642_30170 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 442, 4), 'tuple_var_assignment_29642')
    # Assigning a type to the variable 'ier' (line 442)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 442, 13), 'ier', tuple_var_assignment_29642_30170)
    
    # Getting the type of 'ier' (line 443)
    ier_30171 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 443, 7), 'ier')
    # Testing the type of an if condition (line 443)
    if_condition_30172 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 443, 4), ier_30171)
    # Assigning a type to the variable 'if_condition_30172' (line 443)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 443, 4), 'if_condition_30172', if_condition_30172)
    # SSA begins for if statement (line 443)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # Getting the type of '_RETCODE_ERROR' (line 444)
    _RETCODE_ERROR_30173 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 444, 14), '_RETCODE_ERROR')
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 444, 8), _RETCODE_ERROR_30173, 'raise parameter', BaseException)
    # SSA join for if statement (line 443)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Obtaining an instance of the builtin type 'tuple' (line 445)
    tuple_30174 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 445, 11), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 445)
    # Adding element type (line 445)
    # Getting the type of 'U' (line 445)
    U_30175 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 445, 11), 'U')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 445, 11), tuple_30174, U_30175)
    # Adding element type (line 445)
    # Getting the type of 'V' (line 445)
    V_30176 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 445, 14), 'V')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 445, 11), tuple_30174, V_30176)
    # Adding element type (line 445)
    # Getting the type of 'S' (line 445)
    S_30177 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 445, 17), 'S')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 445, 11), tuple_30174, S_30177)
    
    # Assigning a type to the variable 'stypy_return_type' (line 445)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 445, 4), 'stypy_return_type', tuple_30174)
    
    # ################# End of 'iddr_svd(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'iddr_svd' in the type store
    # Getting the type of 'stypy_return_type' (line 420)
    stypy_return_type_30178 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 420, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_30178)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'iddr_svd'
    return stypy_return_type_30178

# Assigning a type to the variable 'iddr_svd' (line 420)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 420, 0), 'iddr_svd', iddr_svd)

@norecursion
def iddp_svd(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'iddp_svd'
    module_type_store = module_type_store.open_function_context('iddp_svd', 448, 0, False)
    
    # Passed parameters checking function
    iddp_svd.stypy_localization = localization
    iddp_svd.stypy_type_of_self = None
    iddp_svd.stypy_type_store = module_type_store
    iddp_svd.stypy_function_name = 'iddp_svd'
    iddp_svd.stypy_param_names_list = ['eps', 'A']
    iddp_svd.stypy_varargs_param_name = None
    iddp_svd.stypy_kwargs_param_name = None
    iddp_svd.stypy_call_defaults = defaults
    iddp_svd.stypy_call_varargs = varargs
    iddp_svd.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'iddp_svd', ['eps', 'A'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'iddp_svd', localization, ['eps', 'A'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'iddp_svd(...)' code ##################

    str_30179 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 468, (-1)), 'str', '\n    Compute SVD of a real matrix to a specified relative precision.\n\n    :param eps:\n        Relative precision.\n    :type eps: float\n    :param A:\n        Matrix.\n    :type A: :class:`numpy.ndarray`\n\n    :return:\n        Left singular vectors.\n    :rtype: :class:`numpy.ndarray`\n    :return:\n        Right singular vectors.\n    :rtype: :class:`numpy.ndarray`\n    :return:\n        Singular values.\n    :rtype: :class:`numpy.ndarray`\n    ')
    
    # Assigning a Call to a Name (line 469):
    
    # Assigning a Call to a Name (line 469):
    
    # Call to asfortranarray(...): (line 469)
    # Processing the call arguments (line 469)
    # Getting the type of 'A' (line 469)
    A_30182 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 469, 26), 'A', False)
    # Processing the call keyword arguments (line 469)
    kwargs_30183 = {}
    # Getting the type of 'np' (line 469)
    np_30180 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 469, 8), 'np', False)
    # Obtaining the member 'asfortranarray' of a type (line 469)
    asfortranarray_30181 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 469, 8), np_30180, 'asfortranarray')
    # Calling asfortranarray(args, kwargs) (line 469)
    asfortranarray_call_result_30184 = invoke(stypy.reporting.localization.Localization(__file__, 469, 8), asfortranarray_30181, *[A_30182], **kwargs_30183)
    
    # Assigning a type to the variable 'A' (line 469)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 469, 4), 'A', asfortranarray_call_result_30184)
    
    # Assigning a Attribute to a Tuple (line 470):
    
    # Assigning a Subscript to a Name (line 470):
    
    # Obtaining the type of the subscript
    int_30185 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 470, 4), 'int')
    # Getting the type of 'A' (line 470)
    A_30186 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 470, 11), 'A')
    # Obtaining the member 'shape' of a type (line 470)
    shape_30187 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 470, 11), A_30186, 'shape')
    # Obtaining the member '__getitem__' of a type (line 470)
    getitem___30188 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 470, 4), shape_30187, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 470)
    subscript_call_result_30189 = invoke(stypy.reporting.localization.Localization(__file__, 470, 4), getitem___30188, int_30185)
    
    # Assigning a type to the variable 'tuple_var_assignment_29643' (line 470)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 470, 4), 'tuple_var_assignment_29643', subscript_call_result_30189)
    
    # Assigning a Subscript to a Name (line 470):
    
    # Obtaining the type of the subscript
    int_30190 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 470, 4), 'int')
    # Getting the type of 'A' (line 470)
    A_30191 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 470, 11), 'A')
    # Obtaining the member 'shape' of a type (line 470)
    shape_30192 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 470, 11), A_30191, 'shape')
    # Obtaining the member '__getitem__' of a type (line 470)
    getitem___30193 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 470, 4), shape_30192, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 470)
    subscript_call_result_30194 = invoke(stypy.reporting.localization.Localization(__file__, 470, 4), getitem___30193, int_30190)
    
    # Assigning a type to the variable 'tuple_var_assignment_29644' (line 470)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 470, 4), 'tuple_var_assignment_29644', subscript_call_result_30194)
    
    # Assigning a Name to a Name (line 470):
    # Getting the type of 'tuple_var_assignment_29643' (line 470)
    tuple_var_assignment_29643_30195 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 470, 4), 'tuple_var_assignment_29643')
    # Assigning a type to the variable 'm' (line 470)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 470, 4), 'm', tuple_var_assignment_29643_30195)
    
    # Assigning a Name to a Name (line 470):
    # Getting the type of 'tuple_var_assignment_29644' (line 470)
    tuple_var_assignment_29644_30196 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 470, 4), 'tuple_var_assignment_29644')
    # Assigning a type to the variable 'n' (line 470)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 470, 7), 'n', tuple_var_assignment_29644_30196)
    
    # Assigning a Call to a Tuple (line 471):
    
    # Assigning a Subscript to a Name (line 471):
    
    # Obtaining the type of the subscript
    int_30197 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 471, 4), 'int')
    
    # Call to iddp_svd(...): (line 471)
    # Processing the call arguments (line 471)
    # Getting the type of 'eps' (line 471)
    eps_30200 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 471, 41), 'eps', False)
    # Getting the type of 'A' (line 471)
    A_30201 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 471, 46), 'A', False)
    # Processing the call keyword arguments (line 471)
    kwargs_30202 = {}
    # Getting the type of '_id' (line 471)
    _id_30198 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 471, 28), '_id', False)
    # Obtaining the member 'iddp_svd' of a type (line 471)
    iddp_svd_30199 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 471, 28), _id_30198, 'iddp_svd')
    # Calling iddp_svd(args, kwargs) (line 471)
    iddp_svd_call_result_30203 = invoke(stypy.reporting.localization.Localization(__file__, 471, 28), iddp_svd_30199, *[eps_30200, A_30201], **kwargs_30202)
    
    # Obtaining the member '__getitem__' of a type (line 471)
    getitem___30204 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 471, 4), iddp_svd_call_result_30203, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 471)
    subscript_call_result_30205 = invoke(stypy.reporting.localization.Localization(__file__, 471, 4), getitem___30204, int_30197)
    
    # Assigning a type to the variable 'tuple_var_assignment_29645' (line 471)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 471, 4), 'tuple_var_assignment_29645', subscript_call_result_30205)
    
    # Assigning a Subscript to a Name (line 471):
    
    # Obtaining the type of the subscript
    int_30206 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 471, 4), 'int')
    
    # Call to iddp_svd(...): (line 471)
    # Processing the call arguments (line 471)
    # Getting the type of 'eps' (line 471)
    eps_30209 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 471, 41), 'eps', False)
    # Getting the type of 'A' (line 471)
    A_30210 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 471, 46), 'A', False)
    # Processing the call keyword arguments (line 471)
    kwargs_30211 = {}
    # Getting the type of '_id' (line 471)
    _id_30207 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 471, 28), '_id', False)
    # Obtaining the member 'iddp_svd' of a type (line 471)
    iddp_svd_30208 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 471, 28), _id_30207, 'iddp_svd')
    # Calling iddp_svd(args, kwargs) (line 471)
    iddp_svd_call_result_30212 = invoke(stypy.reporting.localization.Localization(__file__, 471, 28), iddp_svd_30208, *[eps_30209, A_30210], **kwargs_30211)
    
    # Obtaining the member '__getitem__' of a type (line 471)
    getitem___30213 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 471, 4), iddp_svd_call_result_30212, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 471)
    subscript_call_result_30214 = invoke(stypy.reporting.localization.Localization(__file__, 471, 4), getitem___30213, int_30206)
    
    # Assigning a type to the variable 'tuple_var_assignment_29646' (line 471)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 471, 4), 'tuple_var_assignment_29646', subscript_call_result_30214)
    
    # Assigning a Subscript to a Name (line 471):
    
    # Obtaining the type of the subscript
    int_30215 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 471, 4), 'int')
    
    # Call to iddp_svd(...): (line 471)
    # Processing the call arguments (line 471)
    # Getting the type of 'eps' (line 471)
    eps_30218 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 471, 41), 'eps', False)
    # Getting the type of 'A' (line 471)
    A_30219 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 471, 46), 'A', False)
    # Processing the call keyword arguments (line 471)
    kwargs_30220 = {}
    # Getting the type of '_id' (line 471)
    _id_30216 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 471, 28), '_id', False)
    # Obtaining the member 'iddp_svd' of a type (line 471)
    iddp_svd_30217 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 471, 28), _id_30216, 'iddp_svd')
    # Calling iddp_svd(args, kwargs) (line 471)
    iddp_svd_call_result_30221 = invoke(stypy.reporting.localization.Localization(__file__, 471, 28), iddp_svd_30217, *[eps_30218, A_30219], **kwargs_30220)
    
    # Obtaining the member '__getitem__' of a type (line 471)
    getitem___30222 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 471, 4), iddp_svd_call_result_30221, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 471)
    subscript_call_result_30223 = invoke(stypy.reporting.localization.Localization(__file__, 471, 4), getitem___30222, int_30215)
    
    # Assigning a type to the variable 'tuple_var_assignment_29647' (line 471)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 471, 4), 'tuple_var_assignment_29647', subscript_call_result_30223)
    
    # Assigning a Subscript to a Name (line 471):
    
    # Obtaining the type of the subscript
    int_30224 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 471, 4), 'int')
    
    # Call to iddp_svd(...): (line 471)
    # Processing the call arguments (line 471)
    # Getting the type of 'eps' (line 471)
    eps_30227 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 471, 41), 'eps', False)
    # Getting the type of 'A' (line 471)
    A_30228 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 471, 46), 'A', False)
    # Processing the call keyword arguments (line 471)
    kwargs_30229 = {}
    # Getting the type of '_id' (line 471)
    _id_30225 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 471, 28), '_id', False)
    # Obtaining the member 'iddp_svd' of a type (line 471)
    iddp_svd_30226 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 471, 28), _id_30225, 'iddp_svd')
    # Calling iddp_svd(args, kwargs) (line 471)
    iddp_svd_call_result_30230 = invoke(stypy.reporting.localization.Localization(__file__, 471, 28), iddp_svd_30226, *[eps_30227, A_30228], **kwargs_30229)
    
    # Obtaining the member '__getitem__' of a type (line 471)
    getitem___30231 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 471, 4), iddp_svd_call_result_30230, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 471)
    subscript_call_result_30232 = invoke(stypy.reporting.localization.Localization(__file__, 471, 4), getitem___30231, int_30224)
    
    # Assigning a type to the variable 'tuple_var_assignment_29648' (line 471)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 471, 4), 'tuple_var_assignment_29648', subscript_call_result_30232)
    
    # Assigning a Subscript to a Name (line 471):
    
    # Obtaining the type of the subscript
    int_30233 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 471, 4), 'int')
    
    # Call to iddp_svd(...): (line 471)
    # Processing the call arguments (line 471)
    # Getting the type of 'eps' (line 471)
    eps_30236 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 471, 41), 'eps', False)
    # Getting the type of 'A' (line 471)
    A_30237 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 471, 46), 'A', False)
    # Processing the call keyword arguments (line 471)
    kwargs_30238 = {}
    # Getting the type of '_id' (line 471)
    _id_30234 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 471, 28), '_id', False)
    # Obtaining the member 'iddp_svd' of a type (line 471)
    iddp_svd_30235 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 471, 28), _id_30234, 'iddp_svd')
    # Calling iddp_svd(args, kwargs) (line 471)
    iddp_svd_call_result_30239 = invoke(stypy.reporting.localization.Localization(__file__, 471, 28), iddp_svd_30235, *[eps_30236, A_30237], **kwargs_30238)
    
    # Obtaining the member '__getitem__' of a type (line 471)
    getitem___30240 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 471, 4), iddp_svd_call_result_30239, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 471)
    subscript_call_result_30241 = invoke(stypy.reporting.localization.Localization(__file__, 471, 4), getitem___30240, int_30233)
    
    # Assigning a type to the variable 'tuple_var_assignment_29649' (line 471)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 471, 4), 'tuple_var_assignment_29649', subscript_call_result_30241)
    
    # Assigning a Subscript to a Name (line 471):
    
    # Obtaining the type of the subscript
    int_30242 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 471, 4), 'int')
    
    # Call to iddp_svd(...): (line 471)
    # Processing the call arguments (line 471)
    # Getting the type of 'eps' (line 471)
    eps_30245 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 471, 41), 'eps', False)
    # Getting the type of 'A' (line 471)
    A_30246 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 471, 46), 'A', False)
    # Processing the call keyword arguments (line 471)
    kwargs_30247 = {}
    # Getting the type of '_id' (line 471)
    _id_30243 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 471, 28), '_id', False)
    # Obtaining the member 'iddp_svd' of a type (line 471)
    iddp_svd_30244 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 471, 28), _id_30243, 'iddp_svd')
    # Calling iddp_svd(args, kwargs) (line 471)
    iddp_svd_call_result_30248 = invoke(stypy.reporting.localization.Localization(__file__, 471, 28), iddp_svd_30244, *[eps_30245, A_30246], **kwargs_30247)
    
    # Obtaining the member '__getitem__' of a type (line 471)
    getitem___30249 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 471, 4), iddp_svd_call_result_30248, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 471)
    subscript_call_result_30250 = invoke(stypy.reporting.localization.Localization(__file__, 471, 4), getitem___30249, int_30242)
    
    # Assigning a type to the variable 'tuple_var_assignment_29650' (line 471)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 471, 4), 'tuple_var_assignment_29650', subscript_call_result_30250)
    
    # Assigning a Name to a Name (line 471):
    # Getting the type of 'tuple_var_assignment_29645' (line 471)
    tuple_var_assignment_29645_30251 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 471, 4), 'tuple_var_assignment_29645')
    # Assigning a type to the variable 'k' (line 471)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 471, 4), 'k', tuple_var_assignment_29645_30251)
    
    # Assigning a Name to a Name (line 471):
    # Getting the type of 'tuple_var_assignment_29646' (line 471)
    tuple_var_assignment_29646_30252 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 471, 4), 'tuple_var_assignment_29646')
    # Assigning a type to the variable 'iU' (line 471)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 471, 7), 'iU', tuple_var_assignment_29646_30252)
    
    # Assigning a Name to a Name (line 471):
    # Getting the type of 'tuple_var_assignment_29647' (line 471)
    tuple_var_assignment_29647_30253 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 471, 4), 'tuple_var_assignment_29647')
    # Assigning a type to the variable 'iV' (line 471)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 471, 11), 'iV', tuple_var_assignment_29647_30253)
    
    # Assigning a Name to a Name (line 471):
    # Getting the type of 'tuple_var_assignment_29648' (line 471)
    tuple_var_assignment_29648_30254 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 471, 4), 'tuple_var_assignment_29648')
    # Assigning a type to the variable 'iS' (line 471)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 471, 15), 'iS', tuple_var_assignment_29648_30254)
    
    # Assigning a Name to a Name (line 471):
    # Getting the type of 'tuple_var_assignment_29649' (line 471)
    tuple_var_assignment_29649_30255 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 471, 4), 'tuple_var_assignment_29649')
    # Assigning a type to the variable 'w' (line 471)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 471, 19), 'w', tuple_var_assignment_29649_30255)
    
    # Assigning a Name to a Name (line 471):
    # Getting the type of 'tuple_var_assignment_29650' (line 471)
    tuple_var_assignment_29650_30256 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 471, 4), 'tuple_var_assignment_29650')
    # Assigning a type to the variable 'ier' (line 471)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 471, 22), 'ier', tuple_var_assignment_29650_30256)
    
    # Getting the type of 'ier' (line 472)
    ier_30257 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 472, 7), 'ier')
    # Testing the type of an if condition (line 472)
    if_condition_30258 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 472, 4), ier_30257)
    # Assigning a type to the variable 'if_condition_30258' (line 472)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 472, 4), 'if_condition_30258', if_condition_30258)
    # SSA begins for if statement (line 472)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # Getting the type of '_RETCODE_ERROR' (line 473)
    _RETCODE_ERROR_30259 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 473, 14), '_RETCODE_ERROR')
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 473, 8), _RETCODE_ERROR_30259, 'raise parameter', BaseException)
    # SSA join for if statement (line 472)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 474):
    
    # Assigning a Call to a Name (line 474):
    
    # Call to reshape(...): (line 474)
    # Processing the call arguments (line 474)
    
    # Obtaining an instance of the builtin type 'tuple' (line 474)
    tuple_30275 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 474, 34), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 474)
    # Adding element type (line 474)
    # Getting the type of 'm' (line 474)
    m_30276 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 474, 34), 'm', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 474, 34), tuple_30275, m_30276)
    # Adding element type (line 474)
    # Getting the type of 'k' (line 474)
    k_30277 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 474, 37), 'k', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 474, 34), tuple_30275, k_30277)
    
    # Processing the call keyword arguments (line 474)
    str_30278 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 474, 47), 'str', 'F')
    keyword_30279 = str_30278
    kwargs_30280 = {'order': keyword_30279}
    
    # Obtaining the type of the subscript
    # Getting the type of 'iU' (line 474)
    iU_30260 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 474, 10), 'iU', False)
    int_30261 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 474, 13), 'int')
    # Applying the binary operator '-' (line 474)
    result_sub_30262 = python_operator(stypy.reporting.localization.Localization(__file__, 474, 10), '-', iU_30260, int_30261)
    
    # Getting the type of 'iU' (line 474)
    iU_30263 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 474, 15), 'iU', False)
    # Getting the type of 'm' (line 474)
    m_30264 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 474, 18), 'm', False)
    # Getting the type of 'k' (line 474)
    k_30265 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 474, 20), 'k', False)
    # Applying the binary operator '*' (line 474)
    result_mul_30266 = python_operator(stypy.reporting.localization.Localization(__file__, 474, 18), '*', m_30264, k_30265)
    
    # Applying the binary operator '+' (line 474)
    result_add_30267 = python_operator(stypy.reporting.localization.Localization(__file__, 474, 15), '+', iU_30263, result_mul_30266)
    
    int_30268 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 474, 22), 'int')
    # Applying the binary operator '-' (line 474)
    result_sub_30269 = python_operator(stypy.reporting.localization.Localization(__file__, 474, 21), '-', result_add_30267, int_30268)
    
    slice_30270 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 474, 8), result_sub_30262, result_sub_30269, None)
    # Getting the type of 'w' (line 474)
    w_30271 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 474, 8), 'w', False)
    # Obtaining the member '__getitem__' of a type (line 474)
    getitem___30272 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 474, 8), w_30271, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 474)
    subscript_call_result_30273 = invoke(stypy.reporting.localization.Localization(__file__, 474, 8), getitem___30272, slice_30270)
    
    # Obtaining the member 'reshape' of a type (line 474)
    reshape_30274 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 474, 8), subscript_call_result_30273, 'reshape')
    # Calling reshape(args, kwargs) (line 474)
    reshape_call_result_30281 = invoke(stypy.reporting.localization.Localization(__file__, 474, 8), reshape_30274, *[tuple_30275], **kwargs_30280)
    
    # Assigning a type to the variable 'U' (line 474)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 474, 4), 'U', reshape_call_result_30281)
    
    # Assigning a Call to a Name (line 475):
    
    # Assigning a Call to a Name (line 475):
    
    # Call to reshape(...): (line 475)
    # Processing the call arguments (line 475)
    
    # Obtaining an instance of the builtin type 'tuple' (line 475)
    tuple_30297 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 475, 34), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 475)
    # Adding element type (line 475)
    # Getting the type of 'n' (line 475)
    n_30298 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 475, 34), 'n', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 475, 34), tuple_30297, n_30298)
    # Adding element type (line 475)
    # Getting the type of 'k' (line 475)
    k_30299 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 475, 37), 'k', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 475, 34), tuple_30297, k_30299)
    
    # Processing the call keyword arguments (line 475)
    str_30300 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 475, 47), 'str', 'F')
    keyword_30301 = str_30300
    kwargs_30302 = {'order': keyword_30301}
    
    # Obtaining the type of the subscript
    # Getting the type of 'iV' (line 475)
    iV_30282 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 475, 10), 'iV', False)
    int_30283 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 475, 13), 'int')
    # Applying the binary operator '-' (line 475)
    result_sub_30284 = python_operator(stypy.reporting.localization.Localization(__file__, 475, 10), '-', iV_30282, int_30283)
    
    # Getting the type of 'iV' (line 475)
    iV_30285 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 475, 15), 'iV', False)
    # Getting the type of 'n' (line 475)
    n_30286 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 475, 18), 'n', False)
    # Getting the type of 'k' (line 475)
    k_30287 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 475, 20), 'k', False)
    # Applying the binary operator '*' (line 475)
    result_mul_30288 = python_operator(stypy.reporting.localization.Localization(__file__, 475, 18), '*', n_30286, k_30287)
    
    # Applying the binary operator '+' (line 475)
    result_add_30289 = python_operator(stypy.reporting.localization.Localization(__file__, 475, 15), '+', iV_30285, result_mul_30288)
    
    int_30290 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 475, 22), 'int')
    # Applying the binary operator '-' (line 475)
    result_sub_30291 = python_operator(stypy.reporting.localization.Localization(__file__, 475, 21), '-', result_add_30289, int_30290)
    
    slice_30292 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 475, 8), result_sub_30284, result_sub_30291, None)
    # Getting the type of 'w' (line 475)
    w_30293 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 475, 8), 'w', False)
    # Obtaining the member '__getitem__' of a type (line 475)
    getitem___30294 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 475, 8), w_30293, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 475)
    subscript_call_result_30295 = invoke(stypy.reporting.localization.Localization(__file__, 475, 8), getitem___30294, slice_30292)
    
    # Obtaining the member 'reshape' of a type (line 475)
    reshape_30296 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 475, 8), subscript_call_result_30295, 'reshape')
    # Calling reshape(args, kwargs) (line 475)
    reshape_call_result_30303 = invoke(stypy.reporting.localization.Localization(__file__, 475, 8), reshape_30296, *[tuple_30297], **kwargs_30302)
    
    # Assigning a type to the variable 'V' (line 475)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 475, 4), 'V', reshape_call_result_30303)
    
    # Assigning a Subscript to a Name (line 476):
    
    # Assigning a Subscript to a Name (line 476):
    
    # Obtaining the type of the subscript
    # Getting the type of 'iS' (line 476)
    iS_30304 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 476, 10), 'iS')
    int_30305 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 476, 13), 'int')
    # Applying the binary operator '-' (line 476)
    result_sub_30306 = python_operator(stypy.reporting.localization.Localization(__file__, 476, 10), '-', iS_30304, int_30305)
    
    # Getting the type of 'iS' (line 476)
    iS_30307 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 476, 15), 'iS')
    # Getting the type of 'k' (line 476)
    k_30308 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 476, 18), 'k')
    # Applying the binary operator '+' (line 476)
    result_add_30309 = python_operator(stypy.reporting.localization.Localization(__file__, 476, 15), '+', iS_30307, k_30308)
    
    int_30310 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 476, 20), 'int')
    # Applying the binary operator '-' (line 476)
    result_sub_30311 = python_operator(stypy.reporting.localization.Localization(__file__, 476, 19), '-', result_add_30309, int_30310)
    
    slice_30312 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 476, 8), result_sub_30306, result_sub_30311, None)
    # Getting the type of 'w' (line 476)
    w_30313 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 476, 8), 'w')
    # Obtaining the member '__getitem__' of a type (line 476)
    getitem___30314 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 476, 8), w_30313, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 476)
    subscript_call_result_30315 = invoke(stypy.reporting.localization.Localization(__file__, 476, 8), getitem___30314, slice_30312)
    
    # Assigning a type to the variable 'S' (line 476)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 476, 4), 'S', subscript_call_result_30315)
    
    # Obtaining an instance of the builtin type 'tuple' (line 477)
    tuple_30316 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 477, 11), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 477)
    # Adding element type (line 477)
    # Getting the type of 'U' (line 477)
    U_30317 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 477, 11), 'U')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 477, 11), tuple_30316, U_30317)
    # Adding element type (line 477)
    # Getting the type of 'V' (line 477)
    V_30318 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 477, 14), 'V')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 477, 11), tuple_30316, V_30318)
    # Adding element type (line 477)
    # Getting the type of 'S' (line 477)
    S_30319 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 477, 17), 'S')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 477, 11), tuple_30316, S_30319)
    
    # Assigning a type to the variable 'stypy_return_type' (line 477)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 477, 4), 'stypy_return_type', tuple_30316)
    
    # ################# End of 'iddp_svd(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'iddp_svd' in the type store
    # Getting the type of 'stypy_return_type' (line 448)
    stypy_return_type_30320 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 448, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_30320)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'iddp_svd'
    return stypy_return_type_30320

# Assigning a type to the variable 'iddp_svd' (line 448)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 448, 0), 'iddp_svd', iddp_svd)

@norecursion
def iddp_aid(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'iddp_aid'
    module_type_store = module_type_store.open_function_context('iddp_aid', 484, 0, False)
    
    # Passed parameters checking function
    iddp_aid.stypy_localization = localization
    iddp_aid.stypy_type_of_self = None
    iddp_aid.stypy_type_store = module_type_store
    iddp_aid.stypy_function_name = 'iddp_aid'
    iddp_aid.stypy_param_names_list = ['eps', 'A']
    iddp_aid.stypy_varargs_param_name = None
    iddp_aid.stypy_kwargs_param_name = None
    iddp_aid.stypy_call_defaults = defaults
    iddp_aid.stypy_call_varargs = varargs
    iddp_aid.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'iddp_aid', ['eps', 'A'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'iddp_aid', localization, ['eps', 'A'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'iddp_aid(...)' code ##################

    str_30321 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 505, (-1)), 'str', '\n    Compute ID of a real matrix to a specified relative precision using random\n    sampling.\n\n    :param eps:\n        Relative precision.\n    :type eps: float\n    :param A:\n        Matrix.\n    :type A: :class:`numpy.ndarray`\n\n    :return:\n        Rank of ID.\n    :rtype: int\n    :return:\n        Column index array.\n    :rtype: :class:`numpy.ndarray`\n    :return:\n        Interpolation coefficients.\n    :rtype: :class:`numpy.ndarray`\n    ')
    
    # Assigning a Call to a Name (line 506):
    
    # Assigning a Call to a Name (line 506):
    
    # Call to asfortranarray(...): (line 506)
    # Processing the call arguments (line 506)
    # Getting the type of 'A' (line 506)
    A_30324 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 506, 26), 'A', False)
    # Processing the call keyword arguments (line 506)
    kwargs_30325 = {}
    # Getting the type of 'np' (line 506)
    np_30322 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 506, 8), 'np', False)
    # Obtaining the member 'asfortranarray' of a type (line 506)
    asfortranarray_30323 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 506, 8), np_30322, 'asfortranarray')
    # Calling asfortranarray(args, kwargs) (line 506)
    asfortranarray_call_result_30326 = invoke(stypy.reporting.localization.Localization(__file__, 506, 8), asfortranarray_30323, *[A_30324], **kwargs_30325)
    
    # Assigning a type to the variable 'A' (line 506)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 506, 4), 'A', asfortranarray_call_result_30326)
    
    # Assigning a Attribute to a Tuple (line 507):
    
    # Assigning a Subscript to a Name (line 507):
    
    # Obtaining the type of the subscript
    int_30327 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 507, 4), 'int')
    # Getting the type of 'A' (line 507)
    A_30328 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 507, 11), 'A')
    # Obtaining the member 'shape' of a type (line 507)
    shape_30329 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 507, 11), A_30328, 'shape')
    # Obtaining the member '__getitem__' of a type (line 507)
    getitem___30330 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 507, 4), shape_30329, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 507)
    subscript_call_result_30331 = invoke(stypy.reporting.localization.Localization(__file__, 507, 4), getitem___30330, int_30327)
    
    # Assigning a type to the variable 'tuple_var_assignment_29651' (line 507)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 507, 4), 'tuple_var_assignment_29651', subscript_call_result_30331)
    
    # Assigning a Subscript to a Name (line 507):
    
    # Obtaining the type of the subscript
    int_30332 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 507, 4), 'int')
    # Getting the type of 'A' (line 507)
    A_30333 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 507, 11), 'A')
    # Obtaining the member 'shape' of a type (line 507)
    shape_30334 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 507, 11), A_30333, 'shape')
    # Obtaining the member '__getitem__' of a type (line 507)
    getitem___30335 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 507, 4), shape_30334, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 507)
    subscript_call_result_30336 = invoke(stypy.reporting.localization.Localization(__file__, 507, 4), getitem___30335, int_30332)
    
    # Assigning a type to the variable 'tuple_var_assignment_29652' (line 507)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 507, 4), 'tuple_var_assignment_29652', subscript_call_result_30336)
    
    # Assigning a Name to a Name (line 507):
    # Getting the type of 'tuple_var_assignment_29651' (line 507)
    tuple_var_assignment_29651_30337 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 507, 4), 'tuple_var_assignment_29651')
    # Assigning a type to the variable 'm' (line 507)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 507, 4), 'm', tuple_var_assignment_29651_30337)
    
    # Assigning a Name to a Name (line 507):
    # Getting the type of 'tuple_var_assignment_29652' (line 507)
    tuple_var_assignment_29652_30338 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 507, 4), 'tuple_var_assignment_29652')
    # Assigning a type to the variable 'n' (line 507)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 507, 7), 'n', tuple_var_assignment_29652_30338)
    
    # Assigning a Call to a Tuple (line 508):
    
    # Assigning a Subscript to a Name (line 508):
    
    # Obtaining the type of the subscript
    int_30339 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 508, 4), 'int')
    
    # Call to idd_frmi(...): (line 508)
    # Processing the call arguments (line 508)
    # Getting the type of 'm' (line 508)
    m_30341 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 508, 21), 'm', False)
    # Processing the call keyword arguments (line 508)
    kwargs_30342 = {}
    # Getting the type of 'idd_frmi' (line 508)
    idd_frmi_30340 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 508, 12), 'idd_frmi', False)
    # Calling idd_frmi(args, kwargs) (line 508)
    idd_frmi_call_result_30343 = invoke(stypy.reporting.localization.Localization(__file__, 508, 12), idd_frmi_30340, *[m_30341], **kwargs_30342)
    
    # Obtaining the member '__getitem__' of a type (line 508)
    getitem___30344 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 508, 4), idd_frmi_call_result_30343, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 508)
    subscript_call_result_30345 = invoke(stypy.reporting.localization.Localization(__file__, 508, 4), getitem___30344, int_30339)
    
    # Assigning a type to the variable 'tuple_var_assignment_29653' (line 508)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 508, 4), 'tuple_var_assignment_29653', subscript_call_result_30345)
    
    # Assigning a Subscript to a Name (line 508):
    
    # Obtaining the type of the subscript
    int_30346 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 508, 4), 'int')
    
    # Call to idd_frmi(...): (line 508)
    # Processing the call arguments (line 508)
    # Getting the type of 'm' (line 508)
    m_30348 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 508, 21), 'm', False)
    # Processing the call keyword arguments (line 508)
    kwargs_30349 = {}
    # Getting the type of 'idd_frmi' (line 508)
    idd_frmi_30347 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 508, 12), 'idd_frmi', False)
    # Calling idd_frmi(args, kwargs) (line 508)
    idd_frmi_call_result_30350 = invoke(stypy.reporting.localization.Localization(__file__, 508, 12), idd_frmi_30347, *[m_30348], **kwargs_30349)
    
    # Obtaining the member '__getitem__' of a type (line 508)
    getitem___30351 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 508, 4), idd_frmi_call_result_30350, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 508)
    subscript_call_result_30352 = invoke(stypy.reporting.localization.Localization(__file__, 508, 4), getitem___30351, int_30346)
    
    # Assigning a type to the variable 'tuple_var_assignment_29654' (line 508)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 508, 4), 'tuple_var_assignment_29654', subscript_call_result_30352)
    
    # Assigning a Name to a Name (line 508):
    # Getting the type of 'tuple_var_assignment_29653' (line 508)
    tuple_var_assignment_29653_30353 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 508, 4), 'tuple_var_assignment_29653')
    # Assigning a type to the variable 'n2' (line 508)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 508, 4), 'n2', tuple_var_assignment_29653_30353)
    
    # Assigning a Name to a Name (line 508):
    # Getting the type of 'tuple_var_assignment_29654' (line 508)
    tuple_var_assignment_29654_30354 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 508, 4), 'tuple_var_assignment_29654')
    # Assigning a type to the variable 'w' (line 508)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 508, 8), 'w', tuple_var_assignment_29654_30354)
    
    # Assigning a Call to a Name (line 509):
    
    # Assigning a Call to a Name (line 509):
    
    # Call to empty(...): (line 509)
    # Processing the call arguments (line 509)
    # Getting the type of 'n' (line 509)
    n_30357 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 509, 20), 'n', False)
    int_30358 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 509, 23), 'int')
    # Getting the type of 'n2' (line 509)
    n2_30359 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 509, 25), 'n2', False)
    # Applying the binary operator '*' (line 509)
    result_mul_30360 = python_operator(stypy.reporting.localization.Localization(__file__, 509, 23), '*', int_30358, n2_30359)
    
    int_30361 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 509, 30), 'int')
    # Applying the binary operator '+' (line 509)
    result_add_30362 = python_operator(stypy.reporting.localization.Localization(__file__, 509, 23), '+', result_mul_30360, int_30361)
    
    # Applying the binary operator '*' (line 509)
    result_mul_30363 = python_operator(stypy.reporting.localization.Localization(__file__, 509, 20), '*', n_30357, result_add_30362)
    
    # Getting the type of 'n2' (line 509)
    n2_30364 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 509, 35), 'n2', False)
    # Applying the binary operator '+' (line 509)
    result_add_30365 = python_operator(stypy.reporting.localization.Localization(__file__, 509, 20), '+', result_mul_30363, n2_30364)
    
    int_30366 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 509, 40), 'int')
    # Applying the binary operator '+' (line 509)
    result_add_30367 = python_operator(stypy.reporting.localization.Localization(__file__, 509, 38), '+', result_add_30365, int_30366)
    
    # Processing the call keyword arguments (line 509)
    str_30368 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 509, 49), 'str', 'F')
    keyword_30369 = str_30368
    kwargs_30370 = {'order': keyword_30369}
    # Getting the type of 'np' (line 509)
    np_30355 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 509, 11), 'np', False)
    # Obtaining the member 'empty' of a type (line 509)
    empty_30356 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 509, 11), np_30355, 'empty')
    # Calling empty(args, kwargs) (line 509)
    empty_call_result_30371 = invoke(stypy.reporting.localization.Localization(__file__, 509, 11), empty_30356, *[result_add_30367], **kwargs_30370)
    
    # Assigning a type to the variable 'proj' (line 509)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 509, 4), 'proj', empty_call_result_30371)
    
    # Assigning a Call to a Tuple (line 510):
    
    # Assigning a Subscript to a Name (line 510):
    
    # Obtaining the type of the subscript
    int_30372 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 510, 4), 'int')
    
    # Call to iddp_aid(...): (line 510)
    # Processing the call arguments (line 510)
    # Getting the type of 'eps' (line 510)
    eps_30375 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 510, 32), 'eps', False)
    # Getting the type of 'A' (line 510)
    A_30376 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 510, 37), 'A', False)
    # Getting the type of 'w' (line 510)
    w_30377 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 510, 40), 'w', False)
    # Getting the type of 'proj' (line 510)
    proj_30378 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 510, 43), 'proj', False)
    # Processing the call keyword arguments (line 510)
    kwargs_30379 = {}
    # Getting the type of '_id' (line 510)
    _id_30373 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 510, 19), '_id', False)
    # Obtaining the member 'iddp_aid' of a type (line 510)
    iddp_aid_30374 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 510, 19), _id_30373, 'iddp_aid')
    # Calling iddp_aid(args, kwargs) (line 510)
    iddp_aid_call_result_30380 = invoke(stypy.reporting.localization.Localization(__file__, 510, 19), iddp_aid_30374, *[eps_30375, A_30376, w_30377, proj_30378], **kwargs_30379)
    
    # Obtaining the member '__getitem__' of a type (line 510)
    getitem___30381 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 510, 4), iddp_aid_call_result_30380, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 510)
    subscript_call_result_30382 = invoke(stypy.reporting.localization.Localization(__file__, 510, 4), getitem___30381, int_30372)
    
    # Assigning a type to the variable 'tuple_var_assignment_29655' (line 510)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 510, 4), 'tuple_var_assignment_29655', subscript_call_result_30382)
    
    # Assigning a Subscript to a Name (line 510):
    
    # Obtaining the type of the subscript
    int_30383 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 510, 4), 'int')
    
    # Call to iddp_aid(...): (line 510)
    # Processing the call arguments (line 510)
    # Getting the type of 'eps' (line 510)
    eps_30386 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 510, 32), 'eps', False)
    # Getting the type of 'A' (line 510)
    A_30387 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 510, 37), 'A', False)
    # Getting the type of 'w' (line 510)
    w_30388 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 510, 40), 'w', False)
    # Getting the type of 'proj' (line 510)
    proj_30389 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 510, 43), 'proj', False)
    # Processing the call keyword arguments (line 510)
    kwargs_30390 = {}
    # Getting the type of '_id' (line 510)
    _id_30384 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 510, 19), '_id', False)
    # Obtaining the member 'iddp_aid' of a type (line 510)
    iddp_aid_30385 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 510, 19), _id_30384, 'iddp_aid')
    # Calling iddp_aid(args, kwargs) (line 510)
    iddp_aid_call_result_30391 = invoke(stypy.reporting.localization.Localization(__file__, 510, 19), iddp_aid_30385, *[eps_30386, A_30387, w_30388, proj_30389], **kwargs_30390)
    
    # Obtaining the member '__getitem__' of a type (line 510)
    getitem___30392 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 510, 4), iddp_aid_call_result_30391, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 510)
    subscript_call_result_30393 = invoke(stypy.reporting.localization.Localization(__file__, 510, 4), getitem___30392, int_30383)
    
    # Assigning a type to the variable 'tuple_var_assignment_29656' (line 510)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 510, 4), 'tuple_var_assignment_29656', subscript_call_result_30393)
    
    # Assigning a Subscript to a Name (line 510):
    
    # Obtaining the type of the subscript
    int_30394 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 510, 4), 'int')
    
    # Call to iddp_aid(...): (line 510)
    # Processing the call arguments (line 510)
    # Getting the type of 'eps' (line 510)
    eps_30397 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 510, 32), 'eps', False)
    # Getting the type of 'A' (line 510)
    A_30398 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 510, 37), 'A', False)
    # Getting the type of 'w' (line 510)
    w_30399 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 510, 40), 'w', False)
    # Getting the type of 'proj' (line 510)
    proj_30400 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 510, 43), 'proj', False)
    # Processing the call keyword arguments (line 510)
    kwargs_30401 = {}
    # Getting the type of '_id' (line 510)
    _id_30395 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 510, 19), '_id', False)
    # Obtaining the member 'iddp_aid' of a type (line 510)
    iddp_aid_30396 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 510, 19), _id_30395, 'iddp_aid')
    # Calling iddp_aid(args, kwargs) (line 510)
    iddp_aid_call_result_30402 = invoke(stypy.reporting.localization.Localization(__file__, 510, 19), iddp_aid_30396, *[eps_30397, A_30398, w_30399, proj_30400], **kwargs_30401)
    
    # Obtaining the member '__getitem__' of a type (line 510)
    getitem___30403 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 510, 4), iddp_aid_call_result_30402, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 510)
    subscript_call_result_30404 = invoke(stypy.reporting.localization.Localization(__file__, 510, 4), getitem___30403, int_30394)
    
    # Assigning a type to the variable 'tuple_var_assignment_29657' (line 510)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 510, 4), 'tuple_var_assignment_29657', subscript_call_result_30404)
    
    # Assigning a Name to a Name (line 510):
    # Getting the type of 'tuple_var_assignment_29655' (line 510)
    tuple_var_assignment_29655_30405 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 510, 4), 'tuple_var_assignment_29655')
    # Assigning a type to the variable 'k' (line 510)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 510, 4), 'k', tuple_var_assignment_29655_30405)
    
    # Assigning a Name to a Name (line 510):
    # Getting the type of 'tuple_var_assignment_29656' (line 510)
    tuple_var_assignment_29656_30406 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 510, 4), 'tuple_var_assignment_29656')
    # Assigning a type to the variable 'idx' (line 510)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 510, 7), 'idx', tuple_var_assignment_29656_30406)
    
    # Assigning a Name to a Name (line 510):
    # Getting the type of 'tuple_var_assignment_29657' (line 510)
    tuple_var_assignment_29657_30407 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 510, 4), 'tuple_var_assignment_29657')
    # Assigning a type to the variable 'proj' (line 510)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 510, 12), 'proj', tuple_var_assignment_29657_30407)
    
    # Assigning a Call to a Name (line 511):
    
    # Assigning a Call to a Name (line 511):
    
    # Call to reshape(...): (line 511)
    # Processing the call arguments (line 511)
    
    # Obtaining an instance of the builtin type 'tuple' (line 511)
    tuple_30418 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 511, 35), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 511)
    # Adding element type (line 511)
    # Getting the type of 'k' (line 511)
    k_30419 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 511, 35), 'k', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 511, 35), tuple_30418, k_30419)
    # Adding element type (line 511)
    # Getting the type of 'n' (line 511)
    n_30420 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 511, 38), 'n', False)
    # Getting the type of 'k' (line 511)
    k_30421 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 511, 40), 'k', False)
    # Applying the binary operator '-' (line 511)
    result_sub_30422 = python_operator(stypy.reporting.localization.Localization(__file__, 511, 38), '-', n_30420, k_30421)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 511, 35), tuple_30418, result_sub_30422)
    
    # Processing the call keyword arguments (line 511)
    str_30423 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 511, 50), 'str', 'F')
    keyword_30424 = str_30423
    kwargs_30425 = {'order': keyword_30424}
    
    # Obtaining the type of the subscript
    # Getting the type of 'k' (line 511)
    k_30408 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 511, 17), 'k', False)
    # Getting the type of 'n' (line 511)
    n_30409 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 511, 20), 'n', False)
    # Getting the type of 'k' (line 511)
    k_30410 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 511, 22), 'k', False)
    # Applying the binary operator '-' (line 511)
    result_sub_30411 = python_operator(stypy.reporting.localization.Localization(__file__, 511, 20), '-', n_30409, k_30410)
    
    # Applying the binary operator '*' (line 511)
    result_mul_30412 = python_operator(stypy.reporting.localization.Localization(__file__, 511, 17), '*', k_30408, result_sub_30411)
    
    slice_30413 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 511, 11), None, result_mul_30412, None)
    # Getting the type of 'proj' (line 511)
    proj_30414 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 511, 11), 'proj', False)
    # Obtaining the member '__getitem__' of a type (line 511)
    getitem___30415 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 511, 11), proj_30414, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 511)
    subscript_call_result_30416 = invoke(stypy.reporting.localization.Localization(__file__, 511, 11), getitem___30415, slice_30413)
    
    # Obtaining the member 'reshape' of a type (line 511)
    reshape_30417 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 511, 11), subscript_call_result_30416, 'reshape')
    # Calling reshape(args, kwargs) (line 511)
    reshape_call_result_30426 = invoke(stypy.reporting.localization.Localization(__file__, 511, 11), reshape_30417, *[tuple_30418], **kwargs_30425)
    
    # Assigning a type to the variable 'proj' (line 511)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 511, 4), 'proj', reshape_call_result_30426)
    
    # Obtaining an instance of the builtin type 'tuple' (line 512)
    tuple_30427 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 512, 11), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 512)
    # Adding element type (line 512)
    # Getting the type of 'k' (line 512)
    k_30428 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 512, 11), 'k')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 512, 11), tuple_30427, k_30428)
    # Adding element type (line 512)
    # Getting the type of 'idx' (line 512)
    idx_30429 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 512, 14), 'idx')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 512, 11), tuple_30427, idx_30429)
    # Adding element type (line 512)
    # Getting the type of 'proj' (line 512)
    proj_30430 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 512, 19), 'proj')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 512, 11), tuple_30427, proj_30430)
    
    # Assigning a type to the variable 'stypy_return_type' (line 512)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 512, 4), 'stypy_return_type', tuple_30427)
    
    # ################# End of 'iddp_aid(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'iddp_aid' in the type store
    # Getting the type of 'stypy_return_type' (line 484)
    stypy_return_type_30431 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 484, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_30431)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'iddp_aid'
    return stypy_return_type_30431

# Assigning a type to the variable 'iddp_aid' (line 484)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 484, 0), 'iddp_aid', iddp_aid)

@norecursion
def idd_estrank(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'idd_estrank'
    module_type_store = module_type_store.open_function_context('idd_estrank', 515, 0, False)
    
    # Passed parameters checking function
    idd_estrank.stypy_localization = localization
    idd_estrank.stypy_type_of_self = None
    idd_estrank.stypy_type_store = module_type_store
    idd_estrank.stypy_function_name = 'idd_estrank'
    idd_estrank.stypy_param_names_list = ['eps', 'A']
    idd_estrank.stypy_varargs_param_name = None
    idd_estrank.stypy_kwargs_param_name = None
    idd_estrank.stypy_call_defaults = defaults
    idd_estrank.stypy_call_varargs = varargs
    idd_estrank.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'idd_estrank', ['eps', 'A'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'idd_estrank', localization, ['eps', 'A'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'idd_estrank(...)' code ##################

    str_30432 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 532, (-1)), 'str', '\n    Estimate rank of a real matrix to a specified relative precision using\n    random sampling.\n\n    The output rank is typically about 8 higher than the actual rank.\n\n    :param eps:\n        Relative precision.\n    :type eps: float\n    :param A:\n        Matrix.\n    :type A: :class:`numpy.ndarray`\n\n    :return:\n        Rank estimate.\n    :rtype: int\n    ')
    
    # Assigning a Call to a Name (line 533):
    
    # Assigning a Call to a Name (line 533):
    
    # Call to asfortranarray(...): (line 533)
    # Processing the call arguments (line 533)
    # Getting the type of 'A' (line 533)
    A_30435 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 533, 26), 'A', False)
    # Processing the call keyword arguments (line 533)
    kwargs_30436 = {}
    # Getting the type of 'np' (line 533)
    np_30433 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 533, 8), 'np', False)
    # Obtaining the member 'asfortranarray' of a type (line 533)
    asfortranarray_30434 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 533, 8), np_30433, 'asfortranarray')
    # Calling asfortranarray(args, kwargs) (line 533)
    asfortranarray_call_result_30437 = invoke(stypy.reporting.localization.Localization(__file__, 533, 8), asfortranarray_30434, *[A_30435], **kwargs_30436)
    
    # Assigning a type to the variable 'A' (line 533)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 533, 4), 'A', asfortranarray_call_result_30437)
    
    # Assigning a Attribute to a Tuple (line 534):
    
    # Assigning a Subscript to a Name (line 534):
    
    # Obtaining the type of the subscript
    int_30438 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 534, 4), 'int')
    # Getting the type of 'A' (line 534)
    A_30439 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 534, 11), 'A')
    # Obtaining the member 'shape' of a type (line 534)
    shape_30440 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 534, 11), A_30439, 'shape')
    # Obtaining the member '__getitem__' of a type (line 534)
    getitem___30441 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 534, 4), shape_30440, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 534)
    subscript_call_result_30442 = invoke(stypy.reporting.localization.Localization(__file__, 534, 4), getitem___30441, int_30438)
    
    # Assigning a type to the variable 'tuple_var_assignment_29658' (line 534)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 534, 4), 'tuple_var_assignment_29658', subscript_call_result_30442)
    
    # Assigning a Subscript to a Name (line 534):
    
    # Obtaining the type of the subscript
    int_30443 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 534, 4), 'int')
    # Getting the type of 'A' (line 534)
    A_30444 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 534, 11), 'A')
    # Obtaining the member 'shape' of a type (line 534)
    shape_30445 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 534, 11), A_30444, 'shape')
    # Obtaining the member '__getitem__' of a type (line 534)
    getitem___30446 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 534, 4), shape_30445, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 534)
    subscript_call_result_30447 = invoke(stypy.reporting.localization.Localization(__file__, 534, 4), getitem___30446, int_30443)
    
    # Assigning a type to the variable 'tuple_var_assignment_29659' (line 534)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 534, 4), 'tuple_var_assignment_29659', subscript_call_result_30447)
    
    # Assigning a Name to a Name (line 534):
    # Getting the type of 'tuple_var_assignment_29658' (line 534)
    tuple_var_assignment_29658_30448 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 534, 4), 'tuple_var_assignment_29658')
    # Assigning a type to the variable 'm' (line 534)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 534, 4), 'm', tuple_var_assignment_29658_30448)
    
    # Assigning a Name to a Name (line 534):
    # Getting the type of 'tuple_var_assignment_29659' (line 534)
    tuple_var_assignment_29659_30449 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 534, 4), 'tuple_var_assignment_29659')
    # Assigning a type to the variable 'n' (line 534)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 534, 7), 'n', tuple_var_assignment_29659_30449)
    
    # Assigning a Call to a Tuple (line 535):
    
    # Assigning a Subscript to a Name (line 535):
    
    # Obtaining the type of the subscript
    int_30450 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 535, 4), 'int')
    
    # Call to idd_frmi(...): (line 535)
    # Processing the call arguments (line 535)
    # Getting the type of 'm' (line 535)
    m_30452 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 535, 21), 'm', False)
    # Processing the call keyword arguments (line 535)
    kwargs_30453 = {}
    # Getting the type of 'idd_frmi' (line 535)
    idd_frmi_30451 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 535, 12), 'idd_frmi', False)
    # Calling idd_frmi(args, kwargs) (line 535)
    idd_frmi_call_result_30454 = invoke(stypy.reporting.localization.Localization(__file__, 535, 12), idd_frmi_30451, *[m_30452], **kwargs_30453)
    
    # Obtaining the member '__getitem__' of a type (line 535)
    getitem___30455 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 535, 4), idd_frmi_call_result_30454, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 535)
    subscript_call_result_30456 = invoke(stypy.reporting.localization.Localization(__file__, 535, 4), getitem___30455, int_30450)
    
    # Assigning a type to the variable 'tuple_var_assignment_29660' (line 535)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 535, 4), 'tuple_var_assignment_29660', subscript_call_result_30456)
    
    # Assigning a Subscript to a Name (line 535):
    
    # Obtaining the type of the subscript
    int_30457 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 535, 4), 'int')
    
    # Call to idd_frmi(...): (line 535)
    # Processing the call arguments (line 535)
    # Getting the type of 'm' (line 535)
    m_30459 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 535, 21), 'm', False)
    # Processing the call keyword arguments (line 535)
    kwargs_30460 = {}
    # Getting the type of 'idd_frmi' (line 535)
    idd_frmi_30458 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 535, 12), 'idd_frmi', False)
    # Calling idd_frmi(args, kwargs) (line 535)
    idd_frmi_call_result_30461 = invoke(stypy.reporting.localization.Localization(__file__, 535, 12), idd_frmi_30458, *[m_30459], **kwargs_30460)
    
    # Obtaining the member '__getitem__' of a type (line 535)
    getitem___30462 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 535, 4), idd_frmi_call_result_30461, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 535)
    subscript_call_result_30463 = invoke(stypy.reporting.localization.Localization(__file__, 535, 4), getitem___30462, int_30457)
    
    # Assigning a type to the variable 'tuple_var_assignment_29661' (line 535)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 535, 4), 'tuple_var_assignment_29661', subscript_call_result_30463)
    
    # Assigning a Name to a Name (line 535):
    # Getting the type of 'tuple_var_assignment_29660' (line 535)
    tuple_var_assignment_29660_30464 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 535, 4), 'tuple_var_assignment_29660')
    # Assigning a type to the variable 'n2' (line 535)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 535, 4), 'n2', tuple_var_assignment_29660_30464)
    
    # Assigning a Name to a Name (line 535):
    # Getting the type of 'tuple_var_assignment_29661' (line 535)
    tuple_var_assignment_29661_30465 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 535, 4), 'tuple_var_assignment_29661')
    # Assigning a type to the variable 'w' (line 535)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 535, 8), 'w', tuple_var_assignment_29661_30465)
    
    # Assigning a Call to a Name (line 536):
    
    # Assigning a Call to a Name (line 536):
    
    # Call to empty(...): (line 536)
    # Processing the call arguments (line 536)
    # Getting the type of 'n' (line 536)
    n_30468 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 536, 18), 'n', False)
    # Getting the type of 'n2' (line 536)
    n2_30469 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 536, 20), 'n2', False)
    # Applying the binary operator '*' (line 536)
    result_mul_30470 = python_operator(stypy.reporting.localization.Localization(__file__, 536, 18), '*', n_30468, n2_30469)
    
    # Getting the type of 'n' (line 536)
    n_30471 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 536, 26), 'n', False)
    int_30472 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 536, 30), 'int')
    # Applying the binary operator '+' (line 536)
    result_add_30473 = python_operator(stypy.reporting.localization.Localization(__file__, 536, 26), '+', n_30471, int_30472)
    
    # Getting the type of 'n2' (line 536)
    n2_30474 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 536, 34), 'n2', False)
    int_30475 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 536, 39), 'int')
    # Applying the binary operator '+' (line 536)
    result_add_30476 = python_operator(stypy.reporting.localization.Localization(__file__, 536, 34), '+', n2_30474, int_30475)
    
    # Applying the binary operator '*' (line 536)
    result_mul_30477 = python_operator(stypy.reporting.localization.Localization(__file__, 536, 25), '*', result_add_30473, result_add_30476)
    
    # Applying the binary operator '+' (line 536)
    result_add_30478 = python_operator(stypy.reporting.localization.Localization(__file__, 536, 18), '+', result_mul_30470, result_mul_30477)
    
    # Processing the call keyword arguments (line 536)
    str_30479 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 536, 49), 'str', 'F')
    keyword_30480 = str_30479
    kwargs_30481 = {'order': keyword_30480}
    # Getting the type of 'np' (line 536)
    np_30466 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 536, 9), 'np', False)
    # Obtaining the member 'empty' of a type (line 536)
    empty_30467 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 536, 9), np_30466, 'empty')
    # Calling empty(args, kwargs) (line 536)
    empty_call_result_30482 = invoke(stypy.reporting.localization.Localization(__file__, 536, 9), empty_30467, *[result_add_30478], **kwargs_30481)
    
    # Assigning a type to the variable 'ra' (line 536)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 536, 4), 'ra', empty_call_result_30482)
    
    # Assigning a Call to a Tuple (line 537):
    
    # Assigning a Subscript to a Name (line 537):
    
    # Obtaining the type of the subscript
    int_30483 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 537, 4), 'int')
    
    # Call to idd_estrank(...): (line 537)
    # Processing the call arguments (line 537)
    # Getting the type of 'eps' (line 537)
    eps_30486 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 537, 28), 'eps', False)
    # Getting the type of 'A' (line 537)
    A_30487 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 537, 33), 'A', False)
    # Getting the type of 'w' (line 537)
    w_30488 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 537, 36), 'w', False)
    # Getting the type of 'ra' (line 537)
    ra_30489 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 537, 39), 'ra', False)
    # Processing the call keyword arguments (line 537)
    kwargs_30490 = {}
    # Getting the type of '_id' (line 537)
    _id_30484 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 537, 12), '_id', False)
    # Obtaining the member 'idd_estrank' of a type (line 537)
    idd_estrank_30485 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 537, 12), _id_30484, 'idd_estrank')
    # Calling idd_estrank(args, kwargs) (line 537)
    idd_estrank_call_result_30491 = invoke(stypy.reporting.localization.Localization(__file__, 537, 12), idd_estrank_30485, *[eps_30486, A_30487, w_30488, ra_30489], **kwargs_30490)
    
    # Obtaining the member '__getitem__' of a type (line 537)
    getitem___30492 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 537, 4), idd_estrank_call_result_30491, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 537)
    subscript_call_result_30493 = invoke(stypy.reporting.localization.Localization(__file__, 537, 4), getitem___30492, int_30483)
    
    # Assigning a type to the variable 'tuple_var_assignment_29662' (line 537)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 537, 4), 'tuple_var_assignment_29662', subscript_call_result_30493)
    
    # Assigning a Subscript to a Name (line 537):
    
    # Obtaining the type of the subscript
    int_30494 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 537, 4), 'int')
    
    # Call to idd_estrank(...): (line 537)
    # Processing the call arguments (line 537)
    # Getting the type of 'eps' (line 537)
    eps_30497 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 537, 28), 'eps', False)
    # Getting the type of 'A' (line 537)
    A_30498 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 537, 33), 'A', False)
    # Getting the type of 'w' (line 537)
    w_30499 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 537, 36), 'w', False)
    # Getting the type of 'ra' (line 537)
    ra_30500 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 537, 39), 'ra', False)
    # Processing the call keyword arguments (line 537)
    kwargs_30501 = {}
    # Getting the type of '_id' (line 537)
    _id_30495 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 537, 12), '_id', False)
    # Obtaining the member 'idd_estrank' of a type (line 537)
    idd_estrank_30496 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 537, 12), _id_30495, 'idd_estrank')
    # Calling idd_estrank(args, kwargs) (line 537)
    idd_estrank_call_result_30502 = invoke(stypy.reporting.localization.Localization(__file__, 537, 12), idd_estrank_30496, *[eps_30497, A_30498, w_30499, ra_30500], **kwargs_30501)
    
    # Obtaining the member '__getitem__' of a type (line 537)
    getitem___30503 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 537, 4), idd_estrank_call_result_30502, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 537)
    subscript_call_result_30504 = invoke(stypy.reporting.localization.Localization(__file__, 537, 4), getitem___30503, int_30494)
    
    # Assigning a type to the variable 'tuple_var_assignment_29663' (line 537)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 537, 4), 'tuple_var_assignment_29663', subscript_call_result_30504)
    
    # Assigning a Name to a Name (line 537):
    # Getting the type of 'tuple_var_assignment_29662' (line 537)
    tuple_var_assignment_29662_30505 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 537, 4), 'tuple_var_assignment_29662')
    # Assigning a type to the variable 'k' (line 537)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 537, 4), 'k', tuple_var_assignment_29662_30505)
    
    # Assigning a Name to a Name (line 537):
    # Getting the type of 'tuple_var_assignment_29663' (line 537)
    tuple_var_assignment_29663_30506 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 537, 4), 'tuple_var_assignment_29663')
    # Assigning a type to the variable 'ra' (line 537)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 537, 7), 'ra', tuple_var_assignment_29663_30506)
    # Getting the type of 'k' (line 538)
    k_30507 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 538, 11), 'k')
    # Assigning a type to the variable 'stypy_return_type' (line 538)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 538, 4), 'stypy_return_type', k_30507)
    
    # ################# End of 'idd_estrank(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'idd_estrank' in the type store
    # Getting the type of 'stypy_return_type' (line 515)
    stypy_return_type_30508 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 515, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_30508)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'idd_estrank'
    return stypy_return_type_30508

# Assigning a type to the variable 'idd_estrank' (line 515)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 515, 0), 'idd_estrank', idd_estrank)

@norecursion
def iddp_asvd(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'iddp_asvd'
    module_type_store = module_type_store.open_function_context('iddp_asvd', 545, 0, False)
    
    # Passed parameters checking function
    iddp_asvd.stypy_localization = localization
    iddp_asvd.stypy_type_of_self = None
    iddp_asvd.stypy_type_store = module_type_store
    iddp_asvd.stypy_function_name = 'iddp_asvd'
    iddp_asvd.stypy_param_names_list = ['eps', 'A']
    iddp_asvd.stypy_varargs_param_name = None
    iddp_asvd.stypy_kwargs_param_name = None
    iddp_asvd.stypy_call_defaults = defaults
    iddp_asvd.stypy_call_varargs = varargs
    iddp_asvd.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'iddp_asvd', ['eps', 'A'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'iddp_asvd', localization, ['eps', 'A'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'iddp_asvd(...)' code ##################

    str_30509 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 566, (-1)), 'str', '\n    Compute SVD of a real matrix to a specified relative precision using random\n    sampling.\n\n    :param eps:\n        Relative precision.\n    :type eps: float\n    :param A:\n        Matrix.\n    :type A: :class:`numpy.ndarray`\n\n    :return:\n        Left singular vectors.\n    :rtype: :class:`numpy.ndarray`\n    :return:\n        Right singular vectors.\n    :rtype: :class:`numpy.ndarray`\n    :return:\n        Singular values.\n    :rtype: :class:`numpy.ndarray`\n    ')
    
    # Assigning a Call to a Name (line 567):
    
    # Assigning a Call to a Name (line 567):
    
    # Call to asfortranarray(...): (line 567)
    # Processing the call arguments (line 567)
    # Getting the type of 'A' (line 567)
    A_30512 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 567, 26), 'A', False)
    # Processing the call keyword arguments (line 567)
    kwargs_30513 = {}
    # Getting the type of 'np' (line 567)
    np_30510 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 567, 8), 'np', False)
    # Obtaining the member 'asfortranarray' of a type (line 567)
    asfortranarray_30511 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 567, 8), np_30510, 'asfortranarray')
    # Calling asfortranarray(args, kwargs) (line 567)
    asfortranarray_call_result_30514 = invoke(stypy.reporting.localization.Localization(__file__, 567, 8), asfortranarray_30511, *[A_30512], **kwargs_30513)
    
    # Assigning a type to the variable 'A' (line 567)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 567, 4), 'A', asfortranarray_call_result_30514)
    
    # Assigning a Attribute to a Tuple (line 568):
    
    # Assigning a Subscript to a Name (line 568):
    
    # Obtaining the type of the subscript
    int_30515 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 568, 4), 'int')
    # Getting the type of 'A' (line 568)
    A_30516 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 568, 11), 'A')
    # Obtaining the member 'shape' of a type (line 568)
    shape_30517 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 568, 11), A_30516, 'shape')
    # Obtaining the member '__getitem__' of a type (line 568)
    getitem___30518 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 568, 4), shape_30517, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 568)
    subscript_call_result_30519 = invoke(stypy.reporting.localization.Localization(__file__, 568, 4), getitem___30518, int_30515)
    
    # Assigning a type to the variable 'tuple_var_assignment_29664' (line 568)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 568, 4), 'tuple_var_assignment_29664', subscript_call_result_30519)
    
    # Assigning a Subscript to a Name (line 568):
    
    # Obtaining the type of the subscript
    int_30520 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 568, 4), 'int')
    # Getting the type of 'A' (line 568)
    A_30521 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 568, 11), 'A')
    # Obtaining the member 'shape' of a type (line 568)
    shape_30522 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 568, 11), A_30521, 'shape')
    # Obtaining the member '__getitem__' of a type (line 568)
    getitem___30523 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 568, 4), shape_30522, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 568)
    subscript_call_result_30524 = invoke(stypy.reporting.localization.Localization(__file__, 568, 4), getitem___30523, int_30520)
    
    # Assigning a type to the variable 'tuple_var_assignment_29665' (line 568)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 568, 4), 'tuple_var_assignment_29665', subscript_call_result_30524)
    
    # Assigning a Name to a Name (line 568):
    # Getting the type of 'tuple_var_assignment_29664' (line 568)
    tuple_var_assignment_29664_30525 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 568, 4), 'tuple_var_assignment_29664')
    # Assigning a type to the variable 'm' (line 568)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 568, 4), 'm', tuple_var_assignment_29664_30525)
    
    # Assigning a Name to a Name (line 568):
    # Getting the type of 'tuple_var_assignment_29665' (line 568)
    tuple_var_assignment_29665_30526 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 568, 4), 'tuple_var_assignment_29665')
    # Assigning a type to the variable 'n' (line 568)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 568, 7), 'n', tuple_var_assignment_29665_30526)
    
    # Assigning a Call to a Tuple (line 569):
    
    # Assigning a Subscript to a Name (line 569):
    
    # Obtaining the type of the subscript
    int_30527 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 569, 4), 'int')
    
    # Call to idd_frmi(...): (line 569)
    # Processing the call arguments (line 569)
    # Getting the type of 'm' (line 569)
    m_30530 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 569, 29), 'm', False)
    # Processing the call keyword arguments (line 569)
    kwargs_30531 = {}
    # Getting the type of '_id' (line 569)
    _id_30528 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 569, 16), '_id', False)
    # Obtaining the member 'idd_frmi' of a type (line 569)
    idd_frmi_30529 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 569, 16), _id_30528, 'idd_frmi')
    # Calling idd_frmi(args, kwargs) (line 569)
    idd_frmi_call_result_30532 = invoke(stypy.reporting.localization.Localization(__file__, 569, 16), idd_frmi_30529, *[m_30530], **kwargs_30531)
    
    # Obtaining the member '__getitem__' of a type (line 569)
    getitem___30533 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 569, 4), idd_frmi_call_result_30532, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 569)
    subscript_call_result_30534 = invoke(stypy.reporting.localization.Localization(__file__, 569, 4), getitem___30533, int_30527)
    
    # Assigning a type to the variable 'tuple_var_assignment_29666' (line 569)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 569, 4), 'tuple_var_assignment_29666', subscript_call_result_30534)
    
    # Assigning a Subscript to a Name (line 569):
    
    # Obtaining the type of the subscript
    int_30535 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 569, 4), 'int')
    
    # Call to idd_frmi(...): (line 569)
    # Processing the call arguments (line 569)
    # Getting the type of 'm' (line 569)
    m_30538 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 569, 29), 'm', False)
    # Processing the call keyword arguments (line 569)
    kwargs_30539 = {}
    # Getting the type of '_id' (line 569)
    _id_30536 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 569, 16), '_id', False)
    # Obtaining the member 'idd_frmi' of a type (line 569)
    idd_frmi_30537 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 569, 16), _id_30536, 'idd_frmi')
    # Calling idd_frmi(args, kwargs) (line 569)
    idd_frmi_call_result_30540 = invoke(stypy.reporting.localization.Localization(__file__, 569, 16), idd_frmi_30537, *[m_30538], **kwargs_30539)
    
    # Obtaining the member '__getitem__' of a type (line 569)
    getitem___30541 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 569, 4), idd_frmi_call_result_30540, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 569)
    subscript_call_result_30542 = invoke(stypy.reporting.localization.Localization(__file__, 569, 4), getitem___30541, int_30535)
    
    # Assigning a type to the variable 'tuple_var_assignment_29667' (line 569)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 569, 4), 'tuple_var_assignment_29667', subscript_call_result_30542)
    
    # Assigning a Name to a Name (line 569):
    # Getting the type of 'tuple_var_assignment_29666' (line 569)
    tuple_var_assignment_29666_30543 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 569, 4), 'tuple_var_assignment_29666')
    # Assigning a type to the variable 'n2' (line 569)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 569, 4), 'n2', tuple_var_assignment_29666_30543)
    
    # Assigning a Name to a Name (line 569):
    # Getting the type of 'tuple_var_assignment_29667' (line 569)
    tuple_var_assignment_29667_30544 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 569, 4), 'tuple_var_assignment_29667')
    # Assigning a type to the variable 'winit' (line 569)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 569, 8), 'winit', tuple_var_assignment_29667_30544)
    
    # Assigning a Call to a Name (line 570):
    
    # Assigning a Call to a Name (line 570):
    
    # Call to empty(...): (line 570)
    # Processing the call arguments (line 570)
    
    # Call to max(...): (line 571)
    # Processing the call arguments (line 571)
    
    # Call to min(...): (line 571)
    # Processing the call arguments (line 571)
    # Getting the type of 'm' (line 571)
    m_30549 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 571, 17), 'm', False)
    # Getting the type of 'n' (line 571)
    n_30550 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 571, 20), 'n', False)
    # Processing the call keyword arguments (line 571)
    kwargs_30551 = {}
    # Getting the type of 'min' (line 571)
    min_30548 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 571, 13), 'min', False)
    # Calling min(args, kwargs) (line 571)
    min_call_result_30552 = invoke(stypy.reporting.localization.Localization(__file__, 571, 13), min_30548, *[m_30549, n_30550], **kwargs_30551)
    
    int_30553 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 571, 25), 'int')
    # Applying the binary operator '+' (line 571)
    result_add_30554 = python_operator(stypy.reporting.localization.Localization(__file__, 571, 13), '+', min_call_result_30552, int_30553)
    
    int_30555 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 571, 29), 'int')
    # Getting the type of 'm' (line 571)
    m_30556 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 571, 31), 'm', False)
    # Applying the binary operator '*' (line 571)
    result_mul_30557 = python_operator(stypy.reporting.localization.Localization(__file__, 571, 29), '*', int_30555, m_30556)
    
    int_30558 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 571, 35), 'int')
    # Getting the type of 'n' (line 571)
    n_30559 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 571, 37), 'n', False)
    # Applying the binary operator '*' (line 571)
    result_mul_30560 = python_operator(stypy.reporting.localization.Localization(__file__, 571, 35), '*', int_30558, n_30559)
    
    # Applying the binary operator '+' (line 571)
    result_add_30561 = python_operator(stypy.reporting.localization.Localization(__file__, 571, 29), '+', result_mul_30557, result_mul_30560)
    
    int_30562 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 571, 41), 'int')
    # Applying the binary operator '+' (line 571)
    result_add_30563 = python_operator(stypy.reporting.localization.Localization(__file__, 571, 39), '+', result_add_30561, int_30562)
    
    # Applying the binary operator '*' (line 571)
    result_mul_30564 = python_operator(stypy.reporting.localization.Localization(__file__, 571, 12), '*', result_add_30554, result_add_30563)
    
    int_30565 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 571, 46), 'int')
    
    # Call to min(...): (line 571)
    # Processing the call arguments (line 571)
    # Getting the type of 'm' (line 571)
    m_30567 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 571, 53), 'm', False)
    # Getting the type of 'n' (line 571)
    n_30568 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 571, 56), 'n', False)
    # Processing the call keyword arguments (line 571)
    kwargs_30569 = {}
    # Getting the type of 'min' (line 571)
    min_30566 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 571, 49), 'min', False)
    # Calling min(args, kwargs) (line 571)
    min_call_result_30570 = invoke(stypy.reporting.localization.Localization(__file__, 571, 49), min_30566, *[m_30567, n_30568], **kwargs_30569)
    
    int_30571 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 571, 60), 'int')
    # Applying the binary operator '**' (line 571)
    result_pow_30572 = python_operator(stypy.reporting.localization.Localization(__file__, 571, 49), '**', min_call_result_30570, int_30571)
    
    # Applying the binary operator '*' (line 571)
    result_mul_30573 = python_operator(stypy.reporting.localization.Localization(__file__, 571, 46), '*', int_30565, result_pow_30572)
    
    # Applying the binary operator '+' (line 571)
    result_add_30574 = python_operator(stypy.reporting.localization.Localization(__file__, 571, 12), '+', result_mul_30564, result_mul_30573)
    
    int_30575 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 572, 13), 'int')
    # Getting the type of 'n' (line 572)
    n_30576 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 572, 15), 'n', False)
    # Applying the binary operator '*' (line 572)
    result_mul_30577 = python_operator(stypy.reporting.localization.Localization(__file__, 572, 13), '*', int_30575, n_30576)
    
    int_30578 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 572, 19), 'int')
    # Applying the binary operator '+' (line 572)
    result_add_30579 = python_operator(stypy.reporting.localization.Localization(__file__, 572, 13), '+', result_mul_30577, int_30578)
    
    # Getting the type of 'n2' (line 572)
    n2_30580 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 572, 23), 'n2', False)
    int_30581 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 572, 28), 'int')
    # Applying the binary operator '+' (line 572)
    result_add_30582 = python_operator(stypy.reporting.localization.Localization(__file__, 572, 23), '+', n2_30580, int_30581)
    
    # Applying the binary operator '*' (line 572)
    result_mul_30583 = python_operator(stypy.reporting.localization.Localization(__file__, 572, 12), '*', result_add_30579, result_add_30582)
    
    # Processing the call keyword arguments (line 571)
    kwargs_30584 = {}
    # Getting the type of 'max' (line 571)
    max_30547 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 571, 8), 'max', False)
    # Calling max(args, kwargs) (line 571)
    max_call_result_30585 = invoke(stypy.reporting.localization.Localization(__file__, 571, 8), max_30547, *[result_add_30574, result_mul_30583], **kwargs_30584)
    
    # Processing the call keyword arguments (line 570)
    str_30586 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 573, 14), 'str', 'F')
    keyword_30587 = str_30586
    kwargs_30588 = {'order': keyword_30587}
    # Getting the type of 'np' (line 570)
    np_30545 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 570, 8), 'np', False)
    # Obtaining the member 'empty' of a type (line 570)
    empty_30546 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 570, 8), np_30545, 'empty')
    # Calling empty(args, kwargs) (line 570)
    empty_call_result_30589 = invoke(stypy.reporting.localization.Localization(__file__, 570, 8), empty_30546, *[max_call_result_30585], **kwargs_30588)
    
    # Assigning a type to the variable 'w' (line 570)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 570, 4), 'w', empty_call_result_30589)
    
    # Assigning a Call to a Tuple (line 574):
    
    # Assigning a Subscript to a Name (line 574):
    
    # Obtaining the type of the subscript
    int_30590 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 574, 4), 'int')
    
    # Call to iddp_asvd(...): (line 574)
    # Processing the call arguments (line 574)
    # Getting the type of 'eps' (line 574)
    eps_30593 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 574, 42), 'eps', False)
    # Getting the type of 'A' (line 574)
    A_30594 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 574, 47), 'A', False)
    # Getting the type of 'winit' (line 574)
    winit_30595 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 574, 50), 'winit', False)
    # Getting the type of 'w' (line 574)
    w_30596 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 574, 57), 'w', False)
    # Processing the call keyword arguments (line 574)
    kwargs_30597 = {}
    # Getting the type of '_id' (line 574)
    _id_30591 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 574, 28), '_id', False)
    # Obtaining the member 'iddp_asvd' of a type (line 574)
    iddp_asvd_30592 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 574, 28), _id_30591, 'iddp_asvd')
    # Calling iddp_asvd(args, kwargs) (line 574)
    iddp_asvd_call_result_30598 = invoke(stypy.reporting.localization.Localization(__file__, 574, 28), iddp_asvd_30592, *[eps_30593, A_30594, winit_30595, w_30596], **kwargs_30597)
    
    # Obtaining the member '__getitem__' of a type (line 574)
    getitem___30599 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 574, 4), iddp_asvd_call_result_30598, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 574)
    subscript_call_result_30600 = invoke(stypy.reporting.localization.Localization(__file__, 574, 4), getitem___30599, int_30590)
    
    # Assigning a type to the variable 'tuple_var_assignment_29668' (line 574)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 574, 4), 'tuple_var_assignment_29668', subscript_call_result_30600)
    
    # Assigning a Subscript to a Name (line 574):
    
    # Obtaining the type of the subscript
    int_30601 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 574, 4), 'int')
    
    # Call to iddp_asvd(...): (line 574)
    # Processing the call arguments (line 574)
    # Getting the type of 'eps' (line 574)
    eps_30604 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 574, 42), 'eps', False)
    # Getting the type of 'A' (line 574)
    A_30605 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 574, 47), 'A', False)
    # Getting the type of 'winit' (line 574)
    winit_30606 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 574, 50), 'winit', False)
    # Getting the type of 'w' (line 574)
    w_30607 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 574, 57), 'w', False)
    # Processing the call keyword arguments (line 574)
    kwargs_30608 = {}
    # Getting the type of '_id' (line 574)
    _id_30602 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 574, 28), '_id', False)
    # Obtaining the member 'iddp_asvd' of a type (line 574)
    iddp_asvd_30603 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 574, 28), _id_30602, 'iddp_asvd')
    # Calling iddp_asvd(args, kwargs) (line 574)
    iddp_asvd_call_result_30609 = invoke(stypy.reporting.localization.Localization(__file__, 574, 28), iddp_asvd_30603, *[eps_30604, A_30605, winit_30606, w_30607], **kwargs_30608)
    
    # Obtaining the member '__getitem__' of a type (line 574)
    getitem___30610 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 574, 4), iddp_asvd_call_result_30609, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 574)
    subscript_call_result_30611 = invoke(stypy.reporting.localization.Localization(__file__, 574, 4), getitem___30610, int_30601)
    
    # Assigning a type to the variable 'tuple_var_assignment_29669' (line 574)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 574, 4), 'tuple_var_assignment_29669', subscript_call_result_30611)
    
    # Assigning a Subscript to a Name (line 574):
    
    # Obtaining the type of the subscript
    int_30612 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 574, 4), 'int')
    
    # Call to iddp_asvd(...): (line 574)
    # Processing the call arguments (line 574)
    # Getting the type of 'eps' (line 574)
    eps_30615 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 574, 42), 'eps', False)
    # Getting the type of 'A' (line 574)
    A_30616 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 574, 47), 'A', False)
    # Getting the type of 'winit' (line 574)
    winit_30617 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 574, 50), 'winit', False)
    # Getting the type of 'w' (line 574)
    w_30618 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 574, 57), 'w', False)
    # Processing the call keyword arguments (line 574)
    kwargs_30619 = {}
    # Getting the type of '_id' (line 574)
    _id_30613 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 574, 28), '_id', False)
    # Obtaining the member 'iddp_asvd' of a type (line 574)
    iddp_asvd_30614 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 574, 28), _id_30613, 'iddp_asvd')
    # Calling iddp_asvd(args, kwargs) (line 574)
    iddp_asvd_call_result_30620 = invoke(stypy.reporting.localization.Localization(__file__, 574, 28), iddp_asvd_30614, *[eps_30615, A_30616, winit_30617, w_30618], **kwargs_30619)
    
    # Obtaining the member '__getitem__' of a type (line 574)
    getitem___30621 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 574, 4), iddp_asvd_call_result_30620, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 574)
    subscript_call_result_30622 = invoke(stypy.reporting.localization.Localization(__file__, 574, 4), getitem___30621, int_30612)
    
    # Assigning a type to the variable 'tuple_var_assignment_29670' (line 574)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 574, 4), 'tuple_var_assignment_29670', subscript_call_result_30622)
    
    # Assigning a Subscript to a Name (line 574):
    
    # Obtaining the type of the subscript
    int_30623 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 574, 4), 'int')
    
    # Call to iddp_asvd(...): (line 574)
    # Processing the call arguments (line 574)
    # Getting the type of 'eps' (line 574)
    eps_30626 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 574, 42), 'eps', False)
    # Getting the type of 'A' (line 574)
    A_30627 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 574, 47), 'A', False)
    # Getting the type of 'winit' (line 574)
    winit_30628 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 574, 50), 'winit', False)
    # Getting the type of 'w' (line 574)
    w_30629 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 574, 57), 'w', False)
    # Processing the call keyword arguments (line 574)
    kwargs_30630 = {}
    # Getting the type of '_id' (line 574)
    _id_30624 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 574, 28), '_id', False)
    # Obtaining the member 'iddp_asvd' of a type (line 574)
    iddp_asvd_30625 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 574, 28), _id_30624, 'iddp_asvd')
    # Calling iddp_asvd(args, kwargs) (line 574)
    iddp_asvd_call_result_30631 = invoke(stypy.reporting.localization.Localization(__file__, 574, 28), iddp_asvd_30625, *[eps_30626, A_30627, winit_30628, w_30629], **kwargs_30630)
    
    # Obtaining the member '__getitem__' of a type (line 574)
    getitem___30632 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 574, 4), iddp_asvd_call_result_30631, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 574)
    subscript_call_result_30633 = invoke(stypy.reporting.localization.Localization(__file__, 574, 4), getitem___30632, int_30623)
    
    # Assigning a type to the variable 'tuple_var_assignment_29671' (line 574)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 574, 4), 'tuple_var_assignment_29671', subscript_call_result_30633)
    
    # Assigning a Subscript to a Name (line 574):
    
    # Obtaining the type of the subscript
    int_30634 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 574, 4), 'int')
    
    # Call to iddp_asvd(...): (line 574)
    # Processing the call arguments (line 574)
    # Getting the type of 'eps' (line 574)
    eps_30637 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 574, 42), 'eps', False)
    # Getting the type of 'A' (line 574)
    A_30638 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 574, 47), 'A', False)
    # Getting the type of 'winit' (line 574)
    winit_30639 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 574, 50), 'winit', False)
    # Getting the type of 'w' (line 574)
    w_30640 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 574, 57), 'w', False)
    # Processing the call keyword arguments (line 574)
    kwargs_30641 = {}
    # Getting the type of '_id' (line 574)
    _id_30635 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 574, 28), '_id', False)
    # Obtaining the member 'iddp_asvd' of a type (line 574)
    iddp_asvd_30636 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 574, 28), _id_30635, 'iddp_asvd')
    # Calling iddp_asvd(args, kwargs) (line 574)
    iddp_asvd_call_result_30642 = invoke(stypy.reporting.localization.Localization(__file__, 574, 28), iddp_asvd_30636, *[eps_30637, A_30638, winit_30639, w_30640], **kwargs_30641)
    
    # Obtaining the member '__getitem__' of a type (line 574)
    getitem___30643 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 574, 4), iddp_asvd_call_result_30642, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 574)
    subscript_call_result_30644 = invoke(stypy.reporting.localization.Localization(__file__, 574, 4), getitem___30643, int_30634)
    
    # Assigning a type to the variable 'tuple_var_assignment_29672' (line 574)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 574, 4), 'tuple_var_assignment_29672', subscript_call_result_30644)
    
    # Assigning a Subscript to a Name (line 574):
    
    # Obtaining the type of the subscript
    int_30645 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 574, 4), 'int')
    
    # Call to iddp_asvd(...): (line 574)
    # Processing the call arguments (line 574)
    # Getting the type of 'eps' (line 574)
    eps_30648 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 574, 42), 'eps', False)
    # Getting the type of 'A' (line 574)
    A_30649 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 574, 47), 'A', False)
    # Getting the type of 'winit' (line 574)
    winit_30650 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 574, 50), 'winit', False)
    # Getting the type of 'w' (line 574)
    w_30651 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 574, 57), 'w', False)
    # Processing the call keyword arguments (line 574)
    kwargs_30652 = {}
    # Getting the type of '_id' (line 574)
    _id_30646 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 574, 28), '_id', False)
    # Obtaining the member 'iddp_asvd' of a type (line 574)
    iddp_asvd_30647 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 574, 28), _id_30646, 'iddp_asvd')
    # Calling iddp_asvd(args, kwargs) (line 574)
    iddp_asvd_call_result_30653 = invoke(stypy.reporting.localization.Localization(__file__, 574, 28), iddp_asvd_30647, *[eps_30648, A_30649, winit_30650, w_30651], **kwargs_30652)
    
    # Obtaining the member '__getitem__' of a type (line 574)
    getitem___30654 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 574, 4), iddp_asvd_call_result_30653, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 574)
    subscript_call_result_30655 = invoke(stypy.reporting.localization.Localization(__file__, 574, 4), getitem___30654, int_30645)
    
    # Assigning a type to the variable 'tuple_var_assignment_29673' (line 574)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 574, 4), 'tuple_var_assignment_29673', subscript_call_result_30655)
    
    # Assigning a Name to a Name (line 574):
    # Getting the type of 'tuple_var_assignment_29668' (line 574)
    tuple_var_assignment_29668_30656 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 574, 4), 'tuple_var_assignment_29668')
    # Assigning a type to the variable 'k' (line 574)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 574, 4), 'k', tuple_var_assignment_29668_30656)
    
    # Assigning a Name to a Name (line 574):
    # Getting the type of 'tuple_var_assignment_29669' (line 574)
    tuple_var_assignment_29669_30657 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 574, 4), 'tuple_var_assignment_29669')
    # Assigning a type to the variable 'iU' (line 574)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 574, 7), 'iU', tuple_var_assignment_29669_30657)
    
    # Assigning a Name to a Name (line 574):
    # Getting the type of 'tuple_var_assignment_29670' (line 574)
    tuple_var_assignment_29670_30658 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 574, 4), 'tuple_var_assignment_29670')
    # Assigning a type to the variable 'iV' (line 574)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 574, 11), 'iV', tuple_var_assignment_29670_30658)
    
    # Assigning a Name to a Name (line 574):
    # Getting the type of 'tuple_var_assignment_29671' (line 574)
    tuple_var_assignment_29671_30659 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 574, 4), 'tuple_var_assignment_29671')
    # Assigning a type to the variable 'iS' (line 574)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 574, 15), 'iS', tuple_var_assignment_29671_30659)
    
    # Assigning a Name to a Name (line 574):
    # Getting the type of 'tuple_var_assignment_29672' (line 574)
    tuple_var_assignment_29672_30660 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 574, 4), 'tuple_var_assignment_29672')
    # Assigning a type to the variable 'w' (line 574)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 574, 19), 'w', tuple_var_assignment_29672_30660)
    
    # Assigning a Name to a Name (line 574):
    # Getting the type of 'tuple_var_assignment_29673' (line 574)
    tuple_var_assignment_29673_30661 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 574, 4), 'tuple_var_assignment_29673')
    # Assigning a type to the variable 'ier' (line 574)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 574, 22), 'ier', tuple_var_assignment_29673_30661)
    
    # Getting the type of 'ier' (line 575)
    ier_30662 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 575, 7), 'ier')
    # Testing the type of an if condition (line 575)
    if_condition_30663 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 575, 4), ier_30662)
    # Assigning a type to the variable 'if_condition_30663' (line 575)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 575, 4), 'if_condition_30663', if_condition_30663)
    # SSA begins for if statement (line 575)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # Getting the type of '_RETCODE_ERROR' (line 576)
    _RETCODE_ERROR_30664 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 576, 14), '_RETCODE_ERROR')
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 576, 8), _RETCODE_ERROR_30664, 'raise parameter', BaseException)
    # SSA join for if statement (line 575)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 577):
    
    # Assigning a Call to a Name (line 577):
    
    # Call to reshape(...): (line 577)
    # Processing the call arguments (line 577)
    
    # Obtaining an instance of the builtin type 'tuple' (line 577)
    tuple_30680 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 577, 34), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 577)
    # Adding element type (line 577)
    # Getting the type of 'm' (line 577)
    m_30681 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 577, 34), 'm', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 577, 34), tuple_30680, m_30681)
    # Adding element type (line 577)
    # Getting the type of 'k' (line 577)
    k_30682 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 577, 37), 'k', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 577, 34), tuple_30680, k_30682)
    
    # Processing the call keyword arguments (line 577)
    str_30683 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 577, 47), 'str', 'F')
    keyword_30684 = str_30683
    kwargs_30685 = {'order': keyword_30684}
    
    # Obtaining the type of the subscript
    # Getting the type of 'iU' (line 577)
    iU_30665 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 577, 10), 'iU', False)
    int_30666 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 577, 13), 'int')
    # Applying the binary operator '-' (line 577)
    result_sub_30667 = python_operator(stypy.reporting.localization.Localization(__file__, 577, 10), '-', iU_30665, int_30666)
    
    # Getting the type of 'iU' (line 577)
    iU_30668 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 577, 15), 'iU', False)
    # Getting the type of 'm' (line 577)
    m_30669 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 577, 18), 'm', False)
    # Getting the type of 'k' (line 577)
    k_30670 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 577, 20), 'k', False)
    # Applying the binary operator '*' (line 577)
    result_mul_30671 = python_operator(stypy.reporting.localization.Localization(__file__, 577, 18), '*', m_30669, k_30670)
    
    # Applying the binary operator '+' (line 577)
    result_add_30672 = python_operator(stypy.reporting.localization.Localization(__file__, 577, 15), '+', iU_30668, result_mul_30671)
    
    int_30673 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 577, 22), 'int')
    # Applying the binary operator '-' (line 577)
    result_sub_30674 = python_operator(stypy.reporting.localization.Localization(__file__, 577, 21), '-', result_add_30672, int_30673)
    
    slice_30675 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 577, 8), result_sub_30667, result_sub_30674, None)
    # Getting the type of 'w' (line 577)
    w_30676 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 577, 8), 'w', False)
    # Obtaining the member '__getitem__' of a type (line 577)
    getitem___30677 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 577, 8), w_30676, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 577)
    subscript_call_result_30678 = invoke(stypy.reporting.localization.Localization(__file__, 577, 8), getitem___30677, slice_30675)
    
    # Obtaining the member 'reshape' of a type (line 577)
    reshape_30679 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 577, 8), subscript_call_result_30678, 'reshape')
    # Calling reshape(args, kwargs) (line 577)
    reshape_call_result_30686 = invoke(stypy.reporting.localization.Localization(__file__, 577, 8), reshape_30679, *[tuple_30680], **kwargs_30685)
    
    # Assigning a type to the variable 'U' (line 577)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 577, 4), 'U', reshape_call_result_30686)
    
    # Assigning a Call to a Name (line 578):
    
    # Assigning a Call to a Name (line 578):
    
    # Call to reshape(...): (line 578)
    # Processing the call arguments (line 578)
    
    # Obtaining an instance of the builtin type 'tuple' (line 578)
    tuple_30702 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 578, 34), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 578)
    # Adding element type (line 578)
    # Getting the type of 'n' (line 578)
    n_30703 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 578, 34), 'n', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 578, 34), tuple_30702, n_30703)
    # Adding element type (line 578)
    # Getting the type of 'k' (line 578)
    k_30704 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 578, 37), 'k', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 578, 34), tuple_30702, k_30704)
    
    # Processing the call keyword arguments (line 578)
    str_30705 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 578, 47), 'str', 'F')
    keyword_30706 = str_30705
    kwargs_30707 = {'order': keyword_30706}
    
    # Obtaining the type of the subscript
    # Getting the type of 'iV' (line 578)
    iV_30687 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 578, 10), 'iV', False)
    int_30688 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 578, 13), 'int')
    # Applying the binary operator '-' (line 578)
    result_sub_30689 = python_operator(stypy.reporting.localization.Localization(__file__, 578, 10), '-', iV_30687, int_30688)
    
    # Getting the type of 'iV' (line 578)
    iV_30690 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 578, 15), 'iV', False)
    # Getting the type of 'n' (line 578)
    n_30691 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 578, 18), 'n', False)
    # Getting the type of 'k' (line 578)
    k_30692 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 578, 20), 'k', False)
    # Applying the binary operator '*' (line 578)
    result_mul_30693 = python_operator(stypy.reporting.localization.Localization(__file__, 578, 18), '*', n_30691, k_30692)
    
    # Applying the binary operator '+' (line 578)
    result_add_30694 = python_operator(stypy.reporting.localization.Localization(__file__, 578, 15), '+', iV_30690, result_mul_30693)
    
    int_30695 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 578, 22), 'int')
    # Applying the binary operator '-' (line 578)
    result_sub_30696 = python_operator(stypy.reporting.localization.Localization(__file__, 578, 21), '-', result_add_30694, int_30695)
    
    slice_30697 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 578, 8), result_sub_30689, result_sub_30696, None)
    # Getting the type of 'w' (line 578)
    w_30698 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 578, 8), 'w', False)
    # Obtaining the member '__getitem__' of a type (line 578)
    getitem___30699 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 578, 8), w_30698, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 578)
    subscript_call_result_30700 = invoke(stypy.reporting.localization.Localization(__file__, 578, 8), getitem___30699, slice_30697)
    
    # Obtaining the member 'reshape' of a type (line 578)
    reshape_30701 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 578, 8), subscript_call_result_30700, 'reshape')
    # Calling reshape(args, kwargs) (line 578)
    reshape_call_result_30708 = invoke(stypy.reporting.localization.Localization(__file__, 578, 8), reshape_30701, *[tuple_30702], **kwargs_30707)
    
    # Assigning a type to the variable 'V' (line 578)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 578, 4), 'V', reshape_call_result_30708)
    
    # Assigning a Subscript to a Name (line 579):
    
    # Assigning a Subscript to a Name (line 579):
    
    # Obtaining the type of the subscript
    # Getting the type of 'iS' (line 579)
    iS_30709 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 579, 10), 'iS')
    int_30710 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 579, 13), 'int')
    # Applying the binary operator '-' (line 579)
    result_sub_30711 = python_operator(stypy.reporting.localization.Localization(__file__, 579, 10), '-', iS_30709, int_30710)
    
    # Getting the type of 'iS' (line 579)
    iS_30712 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 579, 15), 'iS')
    # Getting the type of 'k' (line 579)
    k_30713 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 579, 18), 'k')
    # Applying the binary operator '+' (line 579)
    result_add_30714 = python_operator(stypy.reporting.localization.Localization(__file__, 579, 15), '+', iS_30712, k_30713)
    
    int_30715 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 579, 20), 'int')
    # Applying the binary operator '-' (line 579)
    result_sub_30716 = python_operator(stypy.reporting.localization.Localization(__file__, 579, 19), '-', result_add_30714, int_30715)
    
    slice_30717 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 579, 8), result_sub_30711, result_sub_30716, None)
    # Getting the type of 'w' (line 579)
    w_30718 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 579, 8), 'w')
    # Obtaining the member '__getitem__' of a type (line 579)
    getitem___30719 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 579, 8), w_30718, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 579)
    subscript_call_result_30720 = invoke(stypy.reporting.localization.Localization(__file__, 579, 8), getitem___30719, slice_30717)
    
    # Assigning a type to the variable 'S' (line 579)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 579, 4), 'S', subscript_call_result_30720)
    
    # Obtaining an instance of the builtin type 'tuple' (line 580)
    tuple_30721 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 580, 11), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 580)
    # Adding element type (line 580)
    # Getting the type of 'U' (line 580)
    U_30722 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 580, 11), 'U')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 580, 11), tuple_30721, U_30722)
    # Adding element type (line 580)
    # Getting the type of 'V' (line 580)
    V_30723 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 580, 14), 'V')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 580, 11), tuple_30721, V_30723)
    # Adding element type (line 580)
    # Getting the type of 'S' (line 580)
    S_30724 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 580, 17), 'S')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 580, 11), tuple_30721, S_30724)
    
    # Assigning a type to the variable 'stypy_return_type' (line 580)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 580, 4), 'stypy_return_type', tuple_30721)
    
    # ################# End of 'iddp_asvd(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'iddp_asvd' in the type store
    # Getting the type of 'stypy_return_type' (line 545)
    stypy_return_type_30725 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 545, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_30725)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'iddp_asvd'
    return stypy_return_type_30725

# Assigning a type to the variable 'iddp_asvd' (line 545)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 545, 0), 'iddp_asvd', iddp_asvd)

@norecursion
def iddp_rid(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'iddp_rid'
    module_type_store = module_type_store.open_function_context('iddp_rid', 587, 0, False)
    
    # Passed parameters checking function
    iddp_rid.stypy_localization = localization
    iddp_rid.stypy_type_of_self = None
    iddp_rid.stypy_type_store = module_type_store
    iddp_rid.stypy_function_name = 'iddp_rid'
    iddp_rid.stypy_param_names_list = ['eps', 'm', 'n', 'matvect']
    iddp_rid.stypy_varargs_param_name = None
    iddp_rid.stypy_kwargs_param_name = None
    iddp_rid.stypy_call_defaults = defaults
    iddp_rid.stypy_call_varargs = varargs
    iddp_rid.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'iddp_rid', ['eps', 'm', 'n', 'matvect'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'iddp_rid', localization, ['eps', 'm', 'n', 'matvect'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'iddp_rid(...)' code ##################

    str_30726 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 616, (-1)), 'str', '\n    Compute ID of a real matrix to a specified relative precision using random\n    matrix-vector multiplication.\n\n    :param eps:\n        Relative precision.\n    :type eps: float\n    :param m:\n        Matrix row dimension.\n    :type m: int\n    :param n:\n        Matrix column dimension.\n    :type n: int\n    :param matvect:\n        Function to apply the matrix transpose to a vector, with call signature\n        `y = matvect(x)`, where `x` and `y` are the input and output vectors,\n        respectively.\n    :type matvect: function\n\n    :return:\n        Rank of ID.\n    :rtype: int\n    :return:\n        Column index array.\n    :rtype: :class:`numpy.ndarray`\n    :return:\n        Interpolation coefficients.\n    :rtype: :class:`numpy.ndarray`\n    ')
    
    # Assigning a Call to a Name (line 617):
    
    # Assigning a Call to a Name (line 617):
    
    # Call to empty(...): (line 617)
    # Processing the call arguments (line 617)
    # Getting the type of 'm' (line 617)
    m_30729 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 617, 20), 'm', False)
    int_30730 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 617, 24), 'int')
    # Applying the binary operator '+' (line 617)
    result_add_30731 = python_operator(stypy.reporting.localization.Localization(__file__, 617, 20), '+', m_30729, int_30730)
    
    int_30732 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 617, 28), 'int')
    # Getting the type of 'n' (line 617)
    n_30733 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 617, 30), 'n', False)
    # Applying the binary operator '*' (line 617)
    result_mul_30734 = python_operator(stypy.reporting.localization.Localization(__file__, 617, 28), '*', int_30732, n_30733)
    
    
    # Call to min(...): (line 617)
    # Processing the call arguments (line 617)
    # Getting the type of 'm' (line 617)
    m_30736 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 617, 37), 'm', False)
    # Getting the type of 'n' (line 617)
    n_30737 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 617, 40), 'n', False)
    # Processing the call keyword arguments (line 617)
    kwargs_30738 = {}
    # Getting the type of 'min' (line 617)
    min_30735 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 617, 33), 'min', False)
    # Calling min(args, kwargs) (line 617)
    min_call_result_30739 = invoke(stypy.reporting.localization.Localization(__file__, 617, 33), min_30735, *[m_30736, n_30737], **kwargs_30738)
    
    int_30740 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 617, 45), 'int')
    # Applying the binary operator '+' (line 617)
    result_add_30741 = python_operator(stypy.reporting.localization.Localization(__file__, 617, 33), '+', min_call_result_30739, int_30740)
    
    # Applying the binary operator '*' (line 617)
    result_mul_30742 = python_operator(stypy.reporting.localization.Localization(__file__, 617, 31), '*', result_mul_30734, result_add_30741)
    
    # Applying the binary operator '+' (line 617)
    result_add_30743 = python_operator(stypy.reporting.localization.Localization(__file__, 617, 26), '+', result_add_30731, result_mul_30742)
    
    # Processing the call keyword arguments (line 617)
    str_30744 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 617, 55), 'str', 'F')
    keyword_30745 = str_30744
    kwargs_30746 = {'order': keyword_30745}
    # Getting the type of 'np' (line 617)
    np_30727 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 617, 11), 'np', False)
    # Obtaining the member 'empty' of a type (line 617)
    empty_30728 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 617, 11), np_30727, 'empty')
    # Calling empty(args, kwargs) (line 617)
    empty_call_result_30747 = invoke(stypy.reporting.localization.Localization(__file__, 617, 11), empty_30728, *[result_add_30743], **kwargs_30746)
    
    # Assigning a type to the variable 'proj' (line 617)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 617, 4), 'proj', empty_call_result_30747)
    
    # Assigning a Call to a Tuple (line 618):
    
    # Assigning a Subscript to a Name (line 618):
    
    # Obtaining the type of the subscript
    int_30748 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 618, 4), 'int')
    
    # Call to iddp_rid(...): (line 618)
    # Processing the call arguments (line 618)
    # Getting the type of 'eps' (line 618)
    eps_30751 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 618, 37), 'eps', False)
    # Getting the type of 'm' (line 618)
    m_30752 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 618, 42), 'm', False)
    # Getting the type of 'n' (line 618)
    n_30753 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 618, 45), 'n', False)
    # Getting the type of 'matvect' (line 618)
    matvect_30754 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 618, 48), 'matvect', False)
    # Getting the type of 'proj' (line 618)
    proj_30755 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 618, 57), 'proj', False)
    # Processing the call keyword arguments (line 618)
    kwargs_30756 = {}
    # Getting the type of '_id' (line 618)
    _id_30749 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 618, 24), '_id', False)
    # Obtaining the member 'iddp_rid' of a type (line 618)
    iddp_rid_30750 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 618, 24), _id_30749, 'iddp_rid')
    # Calling iddp_rid(args, kwargs) (line 618)
    iddp_rid_call_result_30757 = invoke(stypy.reporting.localization.Localization(__file__, 618, 24), iddp_rid_30750, *[eps_30751, m_30752, n_30753, matvect_30754, proj_30755], **kwargs_30756)
    
    # Obtaining the member '__getitem__' of a type (line 618)
    getitem___30758 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 618, 4), iddp_rid_call_result_30757, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 618)
    subscript_call_result_30759 = invoke(stypy.reporting.localization.Localization(__file__, 618, 4), getitem___30758, int_30748)
    
    # Assigning a type to the variable 'tuple_var_assignment_29674' (line 618)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 618, 4), 'tuple_var_assignment_29674', subscript_call_result_30759)
    
    # Assigning a Subscript to a Name (line 618):
    
    # Obtaining the type of the subscript
    int_30760 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 618, 4), 'int')
    
    # Call to iddp_rid(...): (line 618)
    # Processing the call arguments (line 618)
    # Getting the type of 'eps' (line 618)
    eps_30763 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 618, 37), 'eps', False)
    # Getting the type of 'm' (line 618)
    m_30764 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 618, 42), 'm', False)
    # Getting the type of 'n' (line 618)
    n_30765 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 618, 45), 'n', False)
    # Getting the type of 'matvect' (line 618)
    matvect_30766 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 618, 48), 'matvect', False)
    # Getting the type of 'proj' (line 618)
    proj_30767 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 618, 57), 'proj', False)
    # Processing the call keyword arguments (line 618)
    kwargs_30768 = {}
    # Getting the type of '_id' (line 618)
    _id_30761 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 618, 24), '_id', False)
    # Obtaining the member 'iddp_rid' of a type (line 618)
    iddp_rid_30762 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 618, 24), _id_30761, 'iddp_rid')
    # Calling iddp_rid(args, kwargs) (line 618)
    iddp_rid_call_result_30769 = invoke(stypy.reporting.localization.Localization(__file__, 618, 24), iddp_rid_30762, *[eps_30763, m_30764, n_30765, matvect_30766, proj_30767], **kwargs_30768)
    
    # Obtaining the member '__getitem__' of a type (line 618)
    getitem___30770 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 618, 4), iddp_rid_call_result_30769, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 618)
    subscript_call_result_30771 = invoke(stypy.reporting.localization.Localization(__file__, 618, 4), getitem___30770, int_30760)
    
    # Assigning a type to the variable 'tuple_var_assignment_29675' (line 618)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 618, 4), 'tuple_var_assignment_29675', subscript_call_result_30771)
    
    # Assigning a Subscript to a Name (line 618):
    
    # Obtaining the type of the subscript
    int_30772 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 618, 4), 'int')
    
    # Call to iddp_rid(...): (line 618)
    # Processing the call arguments (line 618)
    # Getting the type of 'eps' (line 618)
    eps_30775 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 618, 37), 'eps', False)
    # Getting the type of 'm' (line 618)
    m_30776 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 618, 42), 'm', False)
    # Getting the type of 'n' (line 618)
    n_30777 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 618, 45), 'n', False)
    # Getting the type of 'matvect' (line 618)
    matvect_30778 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 618, 48), 'matvect', False)
    # Getting the type of 'proj' (line 618)
    proj_30779 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 618, 57), 'proj', False)
    # Processing the call keyword arguments (line 618)
    kwargs_30780 = {}
    # Getting the type of '_id' (line 618)
    _id_30773 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 618, 24), '_id', False)
    # Obtaining the member 'iddp_rid' of a type (line 618)
    iddp_rid_30774 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 618, 24), _id_30773, 'iddp_rid')
    # Calling iddp_rid(args, kwargs) (line 618)
    iddp_rid_call_result_30781 = invoke(stypy.reporting.localization.Localization(__file__, 618, 24), iddp_rid_30774, *[eps_30775, m_30776, n_30777, matvect_30778, proj_30779], **kwargs_30780)
    
    # Obtaining the member '__getitem__' of a type (line 618)
    getitem___30782 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 618, 4), iddp_rid_call_result_30781, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 618)
    subscript_call_result_30783 = invoke(stypy.reporting.localization.Localization(__file__, 618, 4), getitem___30782, int_30772)
    
    # Assigning a type to the variable 'tuple_var_assignment_29676' (line 618)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 618, 4), 'tuple_var_assignment_29676', subscript_call_result_30783)
    
    # Assigning a Subscript to a Name (line 618):
    
    # Obtaining the type of the subscript
    int_30784 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 618, 4), 'int')
    
    # Call to iddp_rid(...): (line 618)
    # Processing the call arguments (line 618)
    # Getting the type of 'eps' (line 618)
    eps_30787 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 618, 37), 'eps', False)
    # Getting the type of 'm' (line 618)
    m_30788 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 618, 42), 'm', False)
    # Getting the type of 'n' (line 618)
    n_30789 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 618, 45), 'n', False)
    # Getting the type of 'matvect' (line 618)
    matvect_30790 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 618, 48), 'matvect', False)
    # Getting the type of 'proj' (line 618)
    proj_30791 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 618, 57), 'proj', False)
    # Processing the call keyword arguments (line 618)
    kwargs_30792 = {}
    # Getting the type of '_id' (line 618)
    _id_30785 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 618, 24), '_id', False)
    # Obtaining the member 'iddp_rid' of a type (line 618)
    iddp_rid_30786 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 618, 24), _id_30785, 'iddp_rid')
    # Calling iddp_rid(args, kwargs) (line 618)
    iddp_rid_call_result_30793 = invoke(stypy.reporting.localization.Localization(__file__, 618, 24), iddp_rid_30786, *[eps_30787, m_30788, n_30789, matvect_30790, proj_30791], **kwargs_30792)
    
    # Obtaining the member '__getitem__' of a type (line 618)
    getitem___30794 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 618, 4), iddp_rid_call_result_30793, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 618)
    subscript_call_result_30795 = invoke(stypy.reporting.localization.Localization(__file__, 618, 4), getitem___30794, int_30784)
    
    # Assigning a type to the variable 'tuple_var_assignment_29677' (line 618)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 618, 4), 'tuple_var_assignment_29677', subscript_call_result_30795)
    
    # Assigning a Name to a Name (line 618):
    # Getting the type of 'tuple_var_assignment_29674' (line 618)
    tuple_var_assignment_29674_30796 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 618, 4), 'tuple_var_assignment_29674')
    # Assigning a type to the variable 'k' (line 618)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 618, 4), 'k', tuple_var_assignment_29674_30796)
    
    # Assigning a Name to a Name (line 618):
    # Getting the type of 'tuple_var_assignment_29675' (line 618)
    tuple_var_assignment_29675_30797 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 618, 4), 'tuple_var_assignment_29675')
    # Assigning a type to the variable 'idx' (line 618)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 618, 7), 'idx', tuple_var_assignment_29675_30797)
    
    # Assigning a Name to a Name (line 618):
    # Getting the type of 'tuple_var_assignment_29676' (line 618)
    tuple_var_assignment_29676_30798 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 618, 4), 'tuple_var_assignment_29676')
    # Assigning a type to the variable 'proj' (line 618)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 618, 12), 'proj', tuple_var_assignment_29676_30798)
    
    # Assigning a Name to a Name (line 618):
    # Getting the type of 'tuple_var_assignment_29677' (line 618)
    tuple_var_assignment_29677_30799 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 618, 4), 'tuple_var_assignment_29677')
    # Assigning a type to the variable 'ier' (line 618)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 618, 18), 'ier', tuple_var_assignment_29677_30799)
    
    
    # Getting the type of 'ier' (line 619)
    ier_30800 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 619, 7), 'ier')
    int_30801 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 619, 14), 'int')
    # Applying the binary operator '!=' (line 619)
    result_ne_30802 = python_operator(stypy.reporting.localization.Localization(__file__, 619, 7), '!=', ier_30800, int_30801)
    
    # Testing the type of an if condition (line 619)
    if_condition_30803 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 619, 4), result_ne_30802)
    # Assigning a type to the variable 'if_condition_30803' (line 619)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 619, 4), 'if_condition_30803', if_condition_30803)
    # SSA begins for if statement (line 619)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # Getting the type of '_RETCODE_ERROR' (line 620)
    _RETCODE_ERROR_30804 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 620, 14), '_RETCODE_ERROR')
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 620, 8), _RETCODE_ERROR_30804, 'raise parameter', BaseException)
    # SSA join for if statement (line 619)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 621):
    
    # Assigning a Call to a Name (line 621):
    
    # Call to reshape(...): (line 621)
    # Processing the call arguments (line 621)
    
    # Obtaining an instance of the builtin type 'tuple' (line 621)
    tuple_30815 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 621, 35), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 621)
    # Adding element type (line 621)
    # Getting the type of 'k' (line 621)
    k_30816 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 621, 35), 'k', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 621, 35), tuple_30815, k_30816)
    # Adding element type (line 621)
    # Getting the type of 'n' (line 621)
    n_30817 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 621, 38), 'n', False)
    # Getting the type of 'k' (line 621)
    k_30818 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 621, 40), 'k', False)
    # Applying the binary operator '-' (line 621)
    result_sub_30819 = python_operator(stypy.reporting.localization.Localization(__file__, 621, 38), '-', n_30817, k_30818)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 621, 35), tuple_30815, result_sub_30819)
    
    # Processing the call keyword arguments (line 621)
    str_30820 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 621, 50), 'str', 'F')
    keyword_30821 = str_30820
    kwargs_30822 = {'order': keyword_30821}
    
    # Obtaining the type of the subscript
    # Getting the type of 'k' (line 621)
    k_30805 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 621, 17), 'k', False)
    # Getting the type of 'n' (line 621)
    n_30806 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 621, 20), 'n', False)
    # Getting the type of 'k' (line 621)
    k_30807 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 621, 22), 'k', False)
    # Applying the binary operator '-' (line 621)
    result_sub_30808 = python_operator(stypy.reporting.localization.Localization(__file__, 621, 20), '-', n_30806, k_30807)
    
    # Applying the binary operator '*' (line 621)
    result_mul_30809 = python_operator(stypy.reporting.localization.Localization(__file__, 621, 17), '*', k_30805, result_sub_30808)
    
    slice_30810 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 621, 11), None, result_mul_30809, None)
    # Getting the type of 'proj' (line 621)
    proj_30811 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 621, 11), 'proj', False)
    # Obtaining the member '__getitem__' of a type (line 621)
    getitem___30812 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 621, 11), proj_30811, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 621)
    subscript_call_result_30813 = invoke(stypy.reporting.localization.Localization(__file__, 621, 11), getitem___30812, slice_30810)
    
    # Obtaining the member 'reshape' of a type (line 621)
    reshape_30814 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 621, 11), subscript_call_result_30813, 'reshape')
    # Calling reshape(args, kwargs) (line 621)
    reshape_call_result_30823 = invoke(stypy.reporting.localization.Localization(__file__, 621, 11), reshape_30814, *[tuple_30815], **kwargs_30822)
    
    # Assigning a type to the variable 'proj' (line 621)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 621, 4), 'proj', reshape_call_result_30823)
    
    # Obtaining an instance of the builtin type 'tuple' (line 622)
    tuple_30824 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 622, 11), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 622)
    # Adding element type (line 622)
    # Getting the type of 'k' (line 622)
    k_30825 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 622, 11), 'k')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 622, 11), tuple_30824, k_30825)
    # Adding element type (line 622)
    # Getting the type of 'idx' (line 622)
    idx_30826 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 622, 14), 'idx')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 622, 11), tuple_30824, idx_30826)
    # Adding element type (line 622)
    # Getting the type of 'proj' (line 622)
    proj_30827 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 622, 19), 'proj')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 622, 11), tuple_30824, proj_30827)
    
    # Assigning a type to the variable 'stypy_return_type' (line 622)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 622, 4), 'stypy_return_type', tuple_30824)
    
    # ################# End of 'iddp_rid(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'iddp_rid' in the type store
    # Getting the type of 'stypy_return_type' (line 587)
    stypy_return_type_30828 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 587, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_30828)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'iddp_rid'
    return stypy_return_type_30828

# Assigning a type to the variable 'iddp_rid' (line 587)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 587, 0), 'iddp_rid', iddp_rid)

@norecursion
def idd_findrank(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'idd_findrank'
    module_type_store = module_type_store.open_function_context('idd_findrank', 625, 0, False)
    
    # Passed parameters checking function
    idd_findrank.stypy_localization = localization
    idd_findrank.stypy_type_of_self = None
    idd_findrank.stypy_type_store = module_type_store
    idd_findrank.stypy_function_name = 'idd_findrank'
    idd_findrank.stypy_param_names_list = ['eps', 'm', 'n', 'matvect']
    idd_findrank.stypy_varargs_param_name = None
    idd_findrank.stypy_kwargs_param_name = None
    idd_findrank.stypy_call_defaults = defaults
    idd_findrank.stypy_call_varargs = varargs
    idd_findrank.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'idd_findrank', ['eps', 'm', 'n', 'matvect'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'idd_findrank', localization, ['eps', 'm', 'n', 'matvect'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'idd_findrank(...)' code ##################

    str_30829 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 648, (-1)), 'str', '\n    Estimate rank of a real matrix to a specified relative precision using\n    random matrix-vector multiplication.\n\n    :param eps:\n        Relative precision.\n    :type eps: float\n    :param m:\n        Matrix row dimension.\n    :type m: int\n    :param n:\n        Matrix column dimension.\n    :type n: int\n    :param matvect:\n        Function to apply the matrix transpose to a vector, with call signature\n        `y = matvect(x)`, where `x` and `y` are the input and output vectors,\n        respectively.\n    :type matvect: function\n\n    :return:\n        Rank estimate.\n    :rtype: int\n    ')
    
    # Assigning a Call to a Tuple (line 649):
    
    # Assigning a Subscript to a Name (line 649):
    
    # Obtaining the type of the subscript
    int_30830 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 649, 4), 'int')
    
    # Call to idd_findrank(...): (line 649)
    # Processing the call arguments (line 649)
    # Getting the type of 'eps' (line 649)
    eps_30833 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 649, 34), 'eps', False)
    # Getting the type of 'm' (line 649)
    m_30834 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 649, 39), 'm', False)
    # Getting the type of 'n' (line 649)
    n_30835 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 649, 42), 'n', False)
    # Getting the type of 'matvect' (line 649)
    matvect_30836 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 649, 45), 'matvect', False)
    # Processing the call keyword arguments (line 649)
    kwargs_30837 = {}
    # Getting the type of '_id' (line 649)
    _id_30831 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 649, 17), '_id', False)
    # Obtaining the member 'idd_findrank' of a type (line 649)
    idd_findrank_30832 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 649, 17), _id_30831, 'idd_findrank')
    # Calling idd_findrank(args, kwargs) (line 649)
    idd_findrank_call_result_30838 = invoke(stypy.reporting.localization.Localization(__file__, 649, 17), idd_findrank_30832, *[eps_30833, m_30834, n_30835, matvect_30836], **kwargs_30837)
    
    # Obtaining the member '__getitem__' of a type (line 649)
    getitem___30839 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 649, 4), idd_findrank_call_result_30838, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 649)
    subscript_call_result_30840 = invoke(stypy.reporting.localization.Localization(__file__, 649, 4), getitem___30839, int_30830)
    
    # Assigning a type to the variable 'tuple_var_assignment_29678' (line 649)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 649, 4), 'tuple_var_assignment_29678', subscript_call_result_30840)
    
    # Assigning a Subscript to a Name (line 649):
    
    # Obtaining the type of the subscript
    int_30841 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 649, 4), 'int')
    
    # Call to idd_findrank(...): (line 649)
    # Processing the call arguments (line 649)
    # Getting the type of 'eps' (line 649)
    eps_30844 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 649, 34), 'eps', False)
    # Getting the type of 'm' (line 649)
    m_30845 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 649, 39), 'm', False)
    # Getting the type of 'n' (line 649)
    n_30846 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 649, 42), 'n', False)
    # Getting the type of 'matvect' (line 649)
    matvect_30847 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 649, 45), 'matvect', False)
    # Processing the call keyword arguments (line 649)
    kwargs_30848 = {}
    # Getting the type of '_id' (line 649)
    _id_30842 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 649, 17), '_id', False)
    # Obtaining the member 'idd_findrank' of a type (line 649)
    idd_findrank_30843 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 649, 17), _id_30842, 'idd_findrank')
    # Calling idd_findrank(args, kwargs) (line 649)
    idd_findrank_call_result_30849 = invoke(stypy.reporting.localization.Localization(__file__, 649, 17), idd_findrank_30843, *[eps_30844, m_30845, n_30846, matvect_30847], **kwargs_30848)
    
    # Obtaining the member '__getitem__' of a type (line 649)
    getitem___30850 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 649, 4), idd_findrank_call_result_30849, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 649)
    subscript_call_result_30851 = invoke(stypy.reporting.localization.Localization(__file__, 649, 4), getitem___30850, int_30841)
    
    # Assigning a type to the variable 'tuple_var_assignment_29679' (line 649)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 649, 4), 'tuple_var_assignment_29679', subscript_call_result_30851)
    
    # Assigning a Subscript to a Name (line 649):
    
    # Obtaining the type of the subscript
    int_30852 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 649, 4), 'int')
    
    # Call to idd_findrank(...): (line 649)
    # Processing the call arguments (line 649)
    # Getting the type of 'eps' (line 649)
    eps_30855 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 649, 34), 'eps', False)
    # Getting the type of 'm' (line 649)
    m_30856 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 649, 39), 'm', False)
    # Getting the type of 'n' (line 649)
    n_30857 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 649, 42), 'n', False)
    # Getting the type of 'matvect' (line 649)
    matvect_30858 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 649, 45), 'matvect', False)
    # Processing the call keyword arguments (line 649)
    kwargs_30859 = {}
    # Getting the type of '_id' (line 649)
    _id_30853 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 649, 17), '_id', False)
    # Obtaining the member 'idd_findrank' of a type (line 649)
    idd_findrank_30854 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 649, 17), _id_30853, 'idd_findrank')
    # Calling idd_findrank(args, kwargs) (line 649)
    idd_findrank_call_result_30860 = invoke(stypy.reporting.localization.Localization(__file__, 649, 17), idd_findrank_30854, *[eps_30855, m_30856, n_30857, matvect_30858], **kwargs_30859)
    
    # Obtaining the member '__getitem__' of a type (line 649)
    getitem___30861 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 649, 4), idd_findrank_call_result_30860, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 649)
    subscript_call_result_30862 = invoke(stypy.reporting.localization.Localization(__file__, 649, 4), getitem___30861, int_30852)
    
    # Assigning a type to the variable 'tuple_var_assignment_29680' (line 649)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 649, 4), 'tuple_var_assignment_29680', subscript_call_result_30862)
    
    # Assigning a Name to a Name (line 649):
    # Getting the type of 'tuple_var_assignment_29678' (line 649)
    tuple_var_assignment_29678_30863 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 649, 4), 'tuple_var_assignment_29678')
    # Assigning a type to the variable 'k' (line 649)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 649, 4), 'k', tuple_var_assignment_29678_30863)
    
    # Assigning a Name to a Name (line 649):
    # Getting the type of 'tuple_var_assignment_29679' (line 649)
    tuple_var_assignment_29679_30864 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 649, 4), 'tuple_var_assignment_29679')
    # Assigning a type to the variable 'ra' (line 649)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 649, 7), 'ra', tuple_var_assignment_29679_30864)
    
    # Assigning a Name to a Name (line 649):
    # Getting the type of 'tuple_var_assignment_29680' (line 649)
    tuple_var_assignment_29680_30865 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 649, 4), 'tuple_var_assignment_29680')
    # Assigning a type to the variable 'ier' (line 649)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 649, 11), 'ier', tuple_var_assignment_29680_30865)
    
    # Getting the type of 'ier' (line 650)
    ier_30866 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 650, 7), 'ier')
    # Testing the type of an if condition (line 650)
    if_condition_30867 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 650, 4), ier_30866)
    # Assigning a type to the variable 'if_condition_30867' (line 650)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 650, 4), 'if_condition_30867', if_condition_30867)
    # SSA begins for if statement (line 650)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # Getting the type of '_RETCODE_ERROR' (line 651)
    _RETCODE_ERROR_30868 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 651, 14), '_RETCODE_ERROR')
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 651, 8), _RETCODE_ERROR_30868, 'raise parameter', BaseException)
    # SSA join for if statement (line 650)
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'k' (line 652)
    k_30869 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 652, 11), 'k')
    # Assigning a type to the variable 'stypy_return_type' (line 652)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 652, 4), 'stypy_return_type', k_30869)
    
    # ################# End of 'idd_findrank(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'idd_findrank' in the type store
    # Getting the type of 'stypy_return_type' (line 625)
    stypy_return_type_30870 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 625, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_30870)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'idd_findrank'
    return stypy_return_type_30870

# Assigning a type to the variable 'idd_findrank' (line 625)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 625, 0), 'idd_findrank', idd_findrank)

@norecursion
def iddp_rsvd(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'iddp_rsvd'
    module_type_store = module_type_store.open_function_context('iddp_rsvd', 659, 0, False)
    
    # Passed parameters checking function
    iddp_rsvd.stypy_localization = localization
    iddp_rsvd.stypy_type_of_self = None
    iddp_rsvd.stypy_type_store = module_type_store
    iddp_rsvd.stypy_function_name = 'iddp_rsvd'
    iddp_rsvd.stypy_param_names_list = ['eps', 'm', 'n', 'matvect', 'matvec']
    iddp_rsvd.stypy_varargs_param_name = None
    iddp_rsvd.stypy_kwargs_param_name = None
    iddp_rsvd.stypy_call_defaults = defaults
    iddp_rsvd.stypy_call_varargs = varargs
    iddp_rsvd.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'iddp_rsvd', ['eps', 'm', 'n', 'matvect', 'matvec'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'iddp_rsvd', localization, ['eps', 'm', 'n', 'matvect', 'matvec'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'iddp_rsvd(...)' code ##################

    str_30871 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 693, (-1)), 'str', '\n    Compute SVD of a real matrix to a specified relative precision using random\n    matrix-vector multiplication.\n\n    :param eps:\n        Relative precision.\n    :type eps: float\n    :param m:\n        Matrix row dimension.\n    :type m: int\n    :param n:\n        Matrix column dimension.\n    :type n: int\n    :param matvect:\n        Function to apply the matrix transpose to a vector, with call signature\n        `y = matvect(x)`, where `x` and `y` are the input and output vectors,\n        respectively.\n    :type matvect: function\n    :param matvec:\n        Function to apply the matrix to a vector, with call signature\n        `y = matvec(x)`, where `x` and `y` are the input and output vectors,\n        respectively.\n    :type matvec: function\n\n    :return:\n        Left singular vectors.\n    :rtype: :class:`numpy.ndarray`\n    :return:\n        Right singular vectors.\n    :rtype: :class:`numpy.ndarray`\n    :return:\n        Singular values.\n    :rtype: :class:`numpy.ndarray`\n    ')
    
    # Assigning a Call to a Tuple (line 694):
    
    # Assigning a Subscript to a Name (line 694):
    
    # Obtaining the type of the subscript
    int_30872 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 694, 4), 'int')
    
    # Call to iddp_rsvd(...): (line 694)
    # Processing the call arguments (line 694)
    # Getting the type of 'eps' (line 694)
    eps_30875 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 694, 42), 'eps', False)
    # Getting the type of 'm' (line 694)
    m_30876 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 694, 47), 'm', False)
    # Getting the type of 'n' (line 694)
    n_30877 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 694, 50), 'n', False)
    # Getting the type of 'matvect' (line 694)
    matvect_30878 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 694, 53), 'matvect', False)
    # Getting the type of 'matvec' (line 694)
    matvec_30879 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 694, 62), 'matvec', False)
    # Processing the call keyword arguments (line 694)
    kwargs_30880 = {}
    # Getting the type of '_id' (line 694)
    _id_30873 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 694, 28), '_id', False)
    # Obtaining the member 'iddp_rsvd' of a type (line 694)
    iddp_rsvd_30874 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 694, 28), _id_30873, 'iddp_rsvd')
    # Calling iddp_rsvd(args, kwargs) (line 694)
    iddp_rsvd_call_result_30881 = invoke(stypy.reporting.localization.Localization(__file__, 694, 28), iddp_rsvd_30874, *[eps_30875, m_30876, n_30877, matvect_30878, matvec_30879], **kwargs_30880)
    
    # Obtaining the member '__getitem__' of a type (line 694)
    getitem___30882 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 694, 4), iddp_rsvd_call_result_30881, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 694)
    subscript_call_result_30883 = invoke(stypy.reporting.localization.Localization(__file__, 694, 4), getitem___30882, int_30872)
    
    # Assigning a type to the variable 'tuple_var_assignment_29681' (line 694)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 694, 4), 'tuple_var_assignment_29681', subscript_call_result_30883)
    
    # Assigning a Subscript to a Name (line 694):
    
    # Obtaining the type of the subscript
    int_30884 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 694, 4), 'int')
    
    # Call to iddp_rsvd(...): (line 694)
    # Processing the call arguments (line 694)
    # Getting the type of 'eps' (line 694)
    eps_30887 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 694, 42), 'eps', False)
    # Getting the type of 'm' (line 694)
    m_30888 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 694, 47), 'm', False)
    # Getting the type of 'n' (line 694)
    n_30889 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 694, 50), 'n', False)
    # Getting the type of 'matvect' (line 694)
    matvect_30890 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 694, 53), 'matvect', False)
    # Getting the type of 'matvec' (line 694)
    matvec_30891 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 694, 62), 'matvec', False)
    # Processing the call keyword arguments (line 694)
    kwargs_30892 = {}
    # Getting the type of '_id' (line 694)
    _id_30885 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 694, 28), '_id', False)
    # Obtaining the member 'iddp_rsvd' of a type (line 694)
    iddp_rsvd_30886 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 694, 28), _id_30885, 'iddp_rsvd')
    # Calling iddp_rsvd(args, kwargs) (line 694)
    iddp_rsvd_call_result_30893 = invoke(stypy.reporting.localization.Localization(__file__, 694, 28), iddp_rsvd_30886, *[eps_30887, m_30888, n_30889, matvect_30890, matvec_30891], **kwargs_30892)
    
    # Obtaining the member '__getitem__' of a type (line 694)
    getitem___30894 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 694, 4), iddp_rsvd_call_result_30893, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 694)
    subscript_call_result_30895 = invoke(stypy.reporting.localization.Localization(__file__, 694, 4), getitem___30894, int_30884)
    
    # Assigning a type to the variable 'tuple_var_assignment_29682' (line 694)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 694, 4), 'tuple_var_assignment_29682', subscript_call_result_30895)
    
    # Assigning a Subscript to a Name (line 694):
    
    # Obtaining the type of the subscript
    int_30896 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 694, 4), 'int')
    
    # Call to iddp_rsvd(...): (line 694)
    # Processing the call arguments (line 694)
    # Getting the type of 'eps' (line 694)
    eps_30899 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 694, 42), 'eps', False)
    # Getting the type of 'm' (line 694)
    m_30900 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 694, 47), 'm', False)
    # Getting the type of 'n' (line 694)
    n_30901 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 694, 50), 'n', False)
    # Getting the type of 'matvect' (line 694)
    matvect_30902 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 694, 53), 'matvect', False)
    # Getting the type of 'matvec' (line 694)
    matvec_30903 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 694, 62), 'matvec', False)
    # Processing the call keyword arguments (line 694)
    kwargs_30904 = {}
    # Getting the type of '_id' (line 694)
    _id_30897 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 694, 28), '_id', False)
    # Obtaining the member 'iddp_rsvd' of a type (line 694)
    iddp_rsvd_30898 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 694, 28), _id_30897, 'iddp_rsvd')
    # Calling iddp_rsvd(args, kwargs) (line 694)
    iddp_rsvd_call_result_30905 = invoke(stypy.reporting.localization.Localization(__file__, 694, 28), iddp_rsvd_30898, *[eps_30899, m_30900, n_30901, matvect_30902, matvec_30903], **kwargs_30904)
    
    # Obtaining the member '__getitem__' of a type (line 694)
    getitem___30906 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 694, 4), iddp_rsvd_call_result_30905, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 694)
    subscript_call_result_30907 = invoke(stypy.reporting.localization.Localization(__file__, 694, 4), getitem___30906, int_30896)
    
    # Assigning a type to the variable 'tuple_var_assignment_29683' (line 694)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 694, 4), 'tuple_var_assignment_29683', subscript_call_result_30907)
    
    # Assigning a Subscript to a Name (line 694):
    
    # Obtaining the type of the subscript
    int_30908 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 694, 4), 'int')
    
    # Call to iddp_rsvd(...): (line 694)
    # Processing the call arguments (line 694)
    # Getting the type of 'eps' (line 694)
    eps_30911 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 694, 42), 'eps', False)
    # Getting the type of 'm' (line 694)
    m_30912 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 694, 47), 'm', False)
    # Getting the type of 'n' (line 694)
    n_30913 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 694, 50), 'n', False)
    # Getting the type of 'matvect' (line 694)
    matvect_30914 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 694, 53), 'matvect', False)
    # Getting the type of 'matvec' (line 694)
    matvec_30915 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 694, 62), 'matvec', False)
    # Processing the call keyword arguments (line 694)
    kwargs_30916 = {}
    # Getting the type of '_id' (line 694)
    _id_30909 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 694, 28), '_id', False)
    # Obtaining the member 'iddp_rsvd' of a type (line 694)
    iddp_rsvd_30910 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 694, 28), _id_30909, 'iddp_rsvd')
    # Calling iddp_rsvd(args, kwargs) (line 694)
    iddp_rsvd_call_result_30917 = invoke(stypy.reporting.localization.Localization(__file__, 694, 28), iddp_rsvd_30910, *[eps_30911, m_30912, n_30913, matvect_30914, matvec_30915], **kwargs_30916)
    
    # Obtaining the member '__getitem__' of a type (line 694)
    getitem___30918 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 694, 4), iddp_rsvd_call_result_30917, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 694)
    subscript_call_result_30919 = invoke(stypy.reporting.localization.Localization(__file__, 694, 4), getitem___30918, int_30908)
    
    # Assigning a type to the variable 'tuple_var_assignment_29684' (line 694)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 694, 4), 'tuple_var_assignment_29684', subscript_call_result_30919)
    
    # Assigning a Subscript to a Name (line 694):
    
    # Obtaining the type of the subscript
    int_30920 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 694, 4), 'int')
    
    # Call to iddp_rsvd(...): (line 694)
    # Processing the call arguments (line 694)
    # Getting the type of 'eps' (line 694)
    eps_30923 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 694, 42), 'eps', False)
    # Getting the type of 'm' (line 694)
    m_30924 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 694, 47), 'm', False)
    # Getting the type of 'n' (line 694)
    n_30925 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 694, 50), 'n', False)
    # Getting the type of 'matvect' (line 694)
    matvect_30926 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 694, 53), 'matvect', False)
    # Getting the type of 'matvec' (line 694)
    matvec_30927 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 694, 62), 'matvec', False)
    # Processing the call keyword arguments (line 694)
    kwargs_30928 = {}
    # Getting the type of '_id' (line 694)
    _id_30921 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 694, 28), '_id', False)
    # Obtaining the member 'iddp_rsvd' of a type (line 694)
    iddp_rsvd_30922 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 694, 28), _id_30921, 'iddp_rsvd')
    # Calling iddp_rsvd(args, kwargs) (line 694)
    iddp_rsvd_call_result_30929 = invoke(stypy.reporting.localization.Localization(__file__, 694, 28), iddp_rsvd_30922, *[eps_30923, m_30924, n_30925, matvect_30926, matvec_30927], **kwargs_30928)
    
    # Obtaining the member '__getitem__' of a type (line 694)
    getitem___30930 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 694, 4), iddp_rsvd_call_result_30929, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 694)
    subscript_call_result_30931 = invoke(stypy.reporting.localization.Localization(__file__, 694, 4), getitem___30930, int_30920)
    
    # Assigning a type to the variable 'tuple_var_assignment_29685' (line 694)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 694, 4), 'tuple_var_assignment_29685', subscript_call_result_30931)
    
    # Assigning a Subscript to a Name (line 694):
    
    # Obtaining the type of the subscript
    int_30932 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 694, 4), 'int')
    
    # Call to iddp_rsvd(...): (line 694)
    # Processing the call arguments (line 694)
    # Getting the type of 'eps' (line 694)
    eps_30935 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 694, 42), 'eps', False)
    # Getting the type of 'm' (line 694)
    m_30936 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 694, 47), 'm', False)
    # Getting the type of 'n' (line 694)
    n_30937 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 694, 50), 'n', False)
    # Getting the type of 'matvect' (line 694)
    matvect_30938 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 694, 53), 'matvect', False)
    # Getting the type of 'matvec' (line 694)
    matvec_30939 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 694, 62), 'matvec', False)
    # Processing the call keyword arguments (line 694)
    kwargs_30940 = {}
    # Getting the type of '_id' (line 694)
    _id_30933 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 694, 28), '_id', False)
    # Obtaining the member 'iddp_rsvd' of a type (line 694)
    iddp_rsvd_30934 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 694, 28), _id_30933, 'iddp_rsvd')
    # Calling iddp_rsvd(args, kwargs) (line 694)
    iddp_rsvd_call_result_30941 = invoke(stypy.reporting.localization.Localization(__file__, 694, 28), iddp_rsvd_30934, *[eps_30935, m_30936, n_30937, matvect_30938, matvec_30939], **kwargs_30940)
    
    # Obtaining the member '__getitem__' of a type (line 694)
    getitem___30942 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 694, 4), iddp_rsvd_call_result_30941, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 694)
    subscript_call_result_30943 = invoke(stypy.reporting.localization.Localization(__file__, 694, 4), getitem___30942, int_30932)
    
    # Assigning a type to the variable 'tuple_var_assignment_29686' (line 694)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 694, 4), 'tuple_var_assignment_29686', subscript_call_result_30943)
    
    # Assigning a Name to a Name (line 694):
    # Getting the type of 'tuple_var_assignment_29681' (line 694)
    tuple_var_assignment_29681_30944 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 694, 4), 'tuple_var_assignment_29681')
    # Assigning a type to the variable 'k' (line 694)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 694, 4), 'k', tuple_var_assignment_29681_30944)
    
    # Assigning a Name to a Name (line 694):
    # Getting the type of 'tuple_var_assignment_29682' (line 694)
    tuple_var_assignment_29682_30945 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 694, 4), 'tuple_var_assignment_29682')
    # Assigning a type to the variable 'iU' (line 694)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 694, 7), 'iU', tuple_var_assignment_29682_30945)
    
    # Assigning a Name to a Name (line 694):
    # Getting the type of 'tuple_var_assignment_29683' (line 694)
    tuple_var_assignment_29683_30946 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 694, 4), 'tuple_var_assignment_29683')
    # Assigning a type to the variable 'iV' (line 694)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 694, 11), 'iV', tuple_var_assignment_29683_30946)
    
    # Assigning a Name to a Name (line 694):
    # Getting the type of 'tuple_var_assignment_29684' (line 694)
    tuple_var_assignment_29684_30947 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 694, 4), 'tuple_var_assignment_29684')
    # Assigning a type to the variable 'iS' (line 694)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 694, 15), 'iS', tuple_var_assignment_29684_30947)
    
    # Assigning a Name to a Name (line 694):
    # Getting the type of 'tuple_var_assignment_29685' (line 694)
    tuple_var_assignment_29685_30948 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 694, 4), 'tuple_var_assignment_29685')
    # Assigning a type to the variable 'w' (line 694)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 694, 19), 'w', tuple_var_assignment_29685_30948)
    
    # Assigning a Name to a Name (line 694):
    # Getting the type of 'tuple_var_assignment_29686' (line 694)
    tuple_var_assignment_29686_30949 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 694, 4), 'tuple_var_assignment_29686')
    # Assigning a type to the variable 'ier' (line 694)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 694, 22), 'ier', tuple_var_assignment_29686_30949)
    
    # Getting the type of 'ier' (line 695)
    ier_30950 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 695, 7), 'ier')
    # Testing the type of an if condition (line 695)
    if_condition_30951 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 695, 4), ier_30950)
    # Assigning a type to the variable 'if_condition_30951' (line 695)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 695, 4), 'if_condition_30951', if_condition_30951)
    # SSA begins for if statement (line 695)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # Getting the type of '_RETCODE_ERROR' (line 696)
    _RETCODE_ERROR_30952 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 696, 14), '_RETCODE_ERROR')
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 696, 8), _RETCODE_ERROR_30952, 'raise parameter', BaseException)
    # SSA join for if statement (line 695)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 697):
    
    # Assigning a Call to a Name (line 697):
    
    # Call to reshape(...): (line 697)
    # Processing the call arguments (line 697)
    
    # Obtaining an instance of the builtin type 'tuple' (line 697)
    tuple_30968 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 697, 34), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 697)
    # Adding element type (line 697)
    # Getting the type of 'm' (line 697)
    m_30969 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 697, 34), 'm', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 697, 34), tuple_30968, m_30969)
    # Adding element type (line 697)
    # Getting the type of 'k' (line 697)
    k_30970 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 697, 37), 'k', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 697, 34), tuple_30968, k_30970)
    
    # Processing the call keyword arguments (line 697)
    str_30971 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 697, 47), 'str', 'F')
    keyword_30972 = str_30971
    kwargs_30973 = {'order': keyword_30972}
    
    # Obtaining the type of the subscript
    # Getting the type of 'iU' (line 697)
    iU_30953 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 697, 10), 'iU', False)
    int_30954 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 697, 13), 'int')
    # Applying the binary operator '-' (line 697)
    result_sub_30955 = python_operator(stypy.reporting.localization.Localization(__file__, 697, 10), '-', iU_30953, int_30954)
    
    # Getting the type of 'iU' (line 697)
    iU_30956 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 697, 15), 'iU', False)
    # Getting the type of 'm' (line 697)
    m_30957 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 697, 18), 'm', False)
    # Getting the type of 'k' (line 697)
    k_30958 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 697, 20), 'k', False)
    # Applying the binary operator '*' (line 697)
    result_mul_30959 = python_operator(stypy.reporting.localization.Localization(__file__, 697, 18), '*', m_30957, k_30958)
    
    # Applying the binary operator '+' (line 697)
    result_add_30960 = python_operator(stypy.reporting.localization.Localization(__file__, 697, 15), '+', iU_30956, result_mul_30959)
    
    int_30961 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 697, 22), 'int')
    # Applying the binary operator '-' (line 697)
    result_sub_30962 = python_operator(stypy.reporting.localization.Localization(__file__, 697, 21), '-', result_add_30960, int_30961)
    
    slice_30963 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 697, 8), result_sub_30955, result_sub_30962, None)
    # Getting the type of 'w' (line 697)
    w_30964 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 697, 8), 'w', False)
    # Obtaining the member '__getitem__' of a type (line 697)
    getitem___30965 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 697, 8), w_30964, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 697)
    subscript_call_result_30966 = invoke(stypy.reporting.localization.Localization(__file__, 697, 8), getitem___30965, slice_30963)
    
    # Obtaining the member 'reshape' of a type (line 697)
    reshape_30967 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 697, 8), subscript_call_result_30966, 'reshape')
    # Calling reshape(args, kwargs) (line 697)
    reshape_call_result_30974 = invoke(stypy.reporting.localization.Localization(__file__, 697, 8), reshape_30967, *[tuple_30968], **kwargs_30973)
    
    # Assigning a type to the variable 'U' (line 697)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 697, 4), 'U', reshape_call_result_30974)
    
    # Assigning a Call to a Name (line 698):
    
    # Assigning a Call to a Name (line 698):
    
    # Call to reshape(...): (line 698)
    # Processing the call arguments (line 698)
    
    # Obtaining an instance of the builtin type 'tuple' (line 698)
    tuple_30990 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 698, 34), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 698)
    # Adding element type (line 698)
    # Getting the type of 'n' (line 698)
    n_30991 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 698, 34), 'n', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 698, 34), tuple_30990, n_30991)
    # Adding element type (line 698)
    # Getting the type of 'k' (line 698)
    k_30992 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 698, 37), 'k', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 698, 34), tuple_30990, k_30992)
    
    # Processing the call keyword arguments (line 698)
    str_30993 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 698, 47), 'str', 'F')
    keyword_30994 = str_30993
    kwargs_30995 = {'order': keyword_30994}
    
    # Obtaining the type of the subscript
    # Getting the type of 'iV' (line 698)
    iV_30975 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 698, 10), 'iV', False)
    int_30976 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 698, 13), 'int')
    # Applying the binary operator '-' (line 698)
    result_sub_30977 = python_operator(stypy.reporting.localization.Localization(__file__, 698, 10), '-', iV_30975, int_30976)
    
    # Getting the type of 'iV' (line 698)
    iV_30978 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 698, 15), 'iV', False)
    # Getting the type of 'n' (line 698)
    n_30979 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 698, 18), 'n', False)
    # Getting the type of 'k' (line 698)
    k_30980 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 698, 20), 'k', False)
    # Applying the binary operator '*' (line 698)
    result_mul_30981 = python_operator(stypy.reporting.localization.Localization(__file__, 698, 18), '*', n_30979, k_30980)
    
    # Applying the binary operator '+' (line 698)
    result_add_30982 = python_operator(stypy.reporting.localization.Localization(__file__, 698, 15), '+', iV_30978, result_mul_30981)
    
    int_30983 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 698, 22), 'int')
    # Applying the binary operator '-' (line 698)
    result_sub_30984 = python_operator(stypy.reporting.localization.Localization(__file__, 698, 21), '-', result_add_30982, int_30983)
    
    slice_30985 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 698, 8), result_sub_30977, result_sub_30984, None)
    # Getting the type of 'w' (line 698)
    w_30986 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 698, 8), 'w', False)
    # Obtaining the member '__getitem__' of a type (line 698)
    getitem___30987 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 698, 8), w_30986, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 698)
    subscript_call_result_30988 = invoke(stypy.reporting.localization.Localization(__file__, 698, 8), getitem___30987, slice_30985)
    
    # Obtaining the member 'reshape' of a type (line 698)
    reshape_30989 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 698, 8), subscript_call_result_30988, 'reshape')
    # Calling reshape(args, kwargs) (line 698)
    reshape_call_result_30996 = invoke(stypy.reporting.localization.Localization(__file__, 698, 8), reshape_30989, *[tuple_30990], **kwargs_30995)
    
    # Assigning a type to the variable 'V' (line 698)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 698, 4), 'V', reshape_call_result_30996)
    
    # Assigning a Subscript to a Name (line 699):
    
    # Assigning a Subscript to a Name (line 699):
    
    # Obtaining the type of the subscript
    # Getting the type of 'iS' (line 699)
    iS_30997 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 699, 10), 'iS')
    int_30998 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 699, 13), 'int')
    # Applying the binary operator '-' (line 699)
    result_sub_30999 = python_operator(stypy.reporting.localization.Localization(__file__, 699, 10), '-', iS_30997, int_30998)
    
    # Getting the type of 'iS' (line 699)
    iS_31000 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 699, 15), 'iS')
    # Getting the type of 'k' (line 699)
    k_31001 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 699, 18), 'k')
    # Applying the binary operator '+' (line 699)
    result_add_31002 = python_operator(stypy.reporting.localization.Localization(__file__, 699, 15), '+', iS_31000, k_31001)
    
    int_31003 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 699, 20), 'int')
    # Applying the binary operator '-' (line 699)
    result_sub_31004 = python_operator(stypy.reporting.localization.Localization(__file__, 699, 19), '-', result_add_31002, int_31003)
    
    slice_31005 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 699, 8), result_sub_30999, result_sub_31004, None)
    # Getting the type of 'w' (line 699)
    w_31006 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 699, 8), 'w')
    # Obtaining the member '__getitem__' of a type (line 699)
    getitem___31007 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 699, 8), w_31006, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 699)
    subscript_call_result_31008 = invoke(stypy.reporting.localization.Localization(__file__, 699, 8), getitem___31007, slice_31005)
    
    # Assigning a type to the variable 'S' (line 699)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 699, 4), 'S', subscript_call_result_31008)
    
    # Obtaining an instance of the builtin type 'tuple' (line 700)
    tuple_31009 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 700, 11), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 700)
    # Adding element type (line 700)
    # Getting the type of 'U' (line 700)
    U_31010 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 700, 11), 'U')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 700, 11), tuple_31009, U_31010)
    # Adding element type (line 700)
    # Getting the type of 'V' (line 700)
    V_31011 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 700, 14), 'V')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 700, 11), tuple_31009, V_31011)
    # Adding element type (line 700)
    # Getting the type of 'S' (line 700)
    S_31012 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 700, 17), 'S')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 700, 11), tuple_31009, S_31012)
    
    # Assigning a type to the variable 'stypy_return_type' (line 700)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 700, 4), 'stypy_return_type', tuple_31009)
    
    # ################# End of 'iddp_rsvd(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'iddp_rsvd' in the type store
    # Getting the type of 'stypy_return_type' (line 659)
    stypy_return_type_31013 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 659, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_31013)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'iddp_rsvd'
    return stypy_return_type_31013

# Assigning a type to the variable 'iddp_rsvd' (line 659)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 659, 0), 'iddp_rsvd', iddp_rsvd)

@norecursion
def iddr_aid(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'iddr_aid'
    module_type_store = module_type_store.open_function_context('iddr_aid', 707, 0, False)
    
    # Passed parameters checking function
    iddr_aid.stypy_localization = localization
    iddr_aid.stypy_type_of_self = None
    iddr_aid.stypy_type_store = module_type_store
    iddr_aid.stypy_function_name = 'iddr_aid'
    iddr_aid.stypy_param_names_list = ['A', 'k']
    iddr_aid.stypy_varargs_param_name = None
    iddr_aid.stypy_kwargs_param_name = None
    iddr_aid.stypy_call_defaults = defaults
    iddr_aid.stypy_call_varargs = varargs
    iddr_aid.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'iddr_aid', ['A', 'k'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'iddr_aid', localization, ['A', 'k'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'iddr_aid(...)' code ##################

    str_31014 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 724, (-1)), 'str', '\n    Compute ID of a real matrix to a specified rank using random sampling.\n\n    :param A:\n        Matrix.\n    :type A: :class:`numpy.ndarray`\n    :param k:\n        Rank of ID.\n    :type k: int\n\n    :return:\n        Column index array.\n    :rtype: :class:`numpy.ndarray`\n    :return:\n        Interpolation coefficients.\n    :rtype: :class:`numpy.ndarray`\n    ')
    
    # Assigning a Call to a Name (line 725):
    
    # Assigning a Call to a Name (line 725):
    
    # Call to asfortranarray(...): (line 725)
    # Processing the call arguments (line 725)
    # Getting the type of 'A' (line 725)
    A_31017 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 725, 26), 'A', False)
    # Processing the call keyword arguments (line 725)
    kwargs_31018 = {}
    # Getting the type of 'np' (line 725)
    np_31015 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 725, 8), 'np', False)
    # Obtaining the member 'asfortranarray' of a type (line 725)
    asfortranarray_31016 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 725, 8), np_31015, 'asfortranarray')
    # Calling asfortranarray(args, kwargs) (line 725)
    asfortranarray_call_result_31019 = invoke(stypy.reporting.localization.Localization(__file__, 725, 8), asfortranarray_31016, *[A_31017], **kwargs_31018)
    
    # Assigning a type to the variable 'A' (line 725)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 725, 4), 'A', asfortranarray_call_result_31019)
    
    # Assigning a Attribute to a Tuple (line 726):
    
    # Assigning a Subscript to a Name (line 726):
    
    # Obtaining the type of the subscript
    int_31020 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 726, 4), 'int')
    # Getting the type of 'A' (line 726)
    A_31021 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 726, 11), 'A')
    # Obtaining the member 'shape' of a type (line 726)
    shape_31022 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 726, 11), A_31021, 'shape')
    # Obtaining the member '__getitem__' of a type (line 726)
    getitem___31023 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 726, 4), shape_31022, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 726)
    subscript_call_result_31024 = invoke(stypy.reporting.localization.Localization(__file__, 726, 4), getitem___31023, int_31020)
    
    # Assigning a type to the variable 'tuple_var_assignment_29687' (line 726)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 726, 4), 'tuple_var_assignment_29687', subscript_call_result_31024)
    
    # Assigning a Subscript to a Name (line 726):
    
    # Obtaining the type of the subscript
    int_31025 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 726, 4), 'int')
    # Getting the type of 'A' (line 726)
    A_31026 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 726, 11), 'A')
    # Obtaining the member 'shape' of a type (line 726)
    shape_31027 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 726, 11), A_31026, 'shape')
    # Obtaining the member '__getitem__' of a type (line 726)
    getitem___31028 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 726, 4), shape_31027, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 726)
    subscript_call_result_31029 = invoke(stypy.reporting.localization.Localization(__file__, 726, 4), getitem___31028, int_31025)
    
    # Assigning a type to the variable 'tuple_var_assignment_29688' (line 726)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 726, 4), 'tuple_var_assignment_29688', subscript_call_result_31029)
    
    # Assigning a Name to a Name (line 726):
    # Getting the type of 'tuple_var_assignment_29687' (line 726)
    tuple_var_assignment_29687_31030 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 726, 4), 'tuple_var_assignment_29687')
    # Assigning a type to the variable 'm' (line 726)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 726, 4), 'm', tuple_var_assignment_29687_31030)
    
    # Assigning a Name to a Name (line 726):
    # Getting the type of 'tuple_var_assignment_29688' (line 726)
    tuple_var_assignment_29688_31031 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 726, 4), 'tuple_var_assignment_29688')
    # Assigning a type to the variable 'n' (line 726)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 726, 7), 'n', tuple_var_assignment_29688_31031)
    
    # Assigning a Call to a Name (line 727):
    
    # Assigning a Call to a Name (line 727):
    
    # Call to iddr_aidi(...): (line 727)
    # Processing the call arguments (line 727)
    # Getting the type of 'm' (line 727)
    m_31033 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 727, 18), 'm', False)
    # Getting the type of 'n' (line 727)
    n_31034 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 727, 21), 'n', False)
    # Getting the type of 'k' (line 727)
    k_31035 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 727, 24), 'k', False)
    # Processing the call keyword arguments (line 727)
    kwargs_31036 = {}
    # Getting the type of 'iddr_aidi' (line 727)
    iddr_aidi_31032 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 727, 8), 'iddr_aidi', False)
    # Calling iddr_aidi(args, kwargs) (line 727)
    iddr_aidi_call_result_31037 = invoke(stypy.reporting.localization.Localization(__file__, 727, 8), iddr_aidi_31032, *[m_31033, n_31034, k_31035], **kwargs_31036)
    
    # Assigning a type to the variable 'w' (line 727)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 727, 4), 'w', iddr_aidi_call_result_31037)
    
    # Assigning a Call to a Tuple (line 728):
    
    # Assigning a Subscript to a Name (line 728):
    
    # Obtaining the type of the subscript
    int_31038 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 728, 4), 'int')
    
    # Call to iddr_aid(...): (line 728)
    # Processing the call arguments (line 728)
    # Getting the type of 'A' (line 728)
    A_31041 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 728, 29), 'A', False)
    # Getting the type of 'k' (line 728)
    k_31042 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 728, 32), 'k', False)
    # Getting the type of 'w' (line 728)
    w_31043 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 728, 35), 'w', False)
    # Processing the call keyword arguments (line 728)
    kwargs_31044 = {}
    # Getting the type of '_id' (line 728)
    _id_31039 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 728, 16), '_id', False)
    # Obtaining the member 'iddr_aid' of a type (line 728)
    iddr_aid_31040 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 728, 16), _id_31039, 'iddr_aid')
    # Calling iddr_aid(args, kwargs) (line 728)
    iddr_aid_call_result_31045 = invoke(stypy.reporting.localization.Localization(__file__, 728, 16), iddr_aid_31040, *[A_31041, k_31042, w_31043], **kwargs_31044)
    
    # Obtaining the member '__getitem__' of a type (line 728)
    getitem___31046 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 728, 4), iddr_aid_call_result_31045, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 728)
    subscript_call_result_31047 = invoke(stypy.reporting.localization.Localization(__file__, 728, 4), getitem___31046, int_31038)
    
    # Assigning a type to the variable 'tuple_var_assignment_29689' (line 728)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 728, 4), 'tuple_var_assignment_29689', subscript_call_result_31047)
    
    # Assigning a Subscript to a Name (line 728):
    
    # Obtaining the type of the subscript
    int_31048 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 728, 4), 'int')
    
    # Call to iddr_aid(...): (line 728)
    # Processing the call arguments (line 728)
    # Getting the type of 'A' (line 728)
    A_31051 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 728, 29), 'A', False)
    # Getting the type of 'k' (line 728)
    k_31052 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 728, 32), 'k', False)
    # Getting the type of 'w' (line 728)
    w_31053 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 728, 35), 'w', False)
    # Processing the call keyword arguments (line 728)
    kwargs_31054 = {}
    # Getting the type of '_id' (line 728)
    _id_31049 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 728, 16), '_id', False)
    # Obtaining the member 'iddr_aid' of a type (line 728)
    iddr_aid_31050 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 728, 16), _id_31049, 'iddr_aid')
    # Calling iddr_aid(args, kwargs) (line 728)
    iddr_aid_call_result_31055 = invoke(stypy.reporting.localization.Localization(__file__, 728, 16), iddr_aid_31050, *[A_31051, k_31052, w_31053], **kwargs_31054)
    
    # Obtaining the member '__getitem__' of a type (line 728)
    getitem___31056 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 728, 4), iddr_aid_call_result_31055, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 728)
    subscript_call_result_31057 = invoke(stypy.reporting.localization.Localization(__file__, 728, 4), getitem___31056, int_31048)
    
    # Assigning a type to the variable 'tuple_var_assignment_29690' (line 728)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 728, 4), 'tuple_var_assignment_29690', subscript_call_result_31057)
    
    # Assigning a Name to a Name (line 728):
    # Getting the type of 'tuple_var_assignment_29689' (line 728)
    tuple_var_assignment_29689_31058 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 728, 4), 'tuple_var_assignment_29689')
    # Assigning a type to the variable 'idx' (line 728)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 728, 4), 'idx', tuple_var_assignment_29689_31058)
    
    # Assigning a Name to a Name (line 728):
    # Getting the type of 'tuple_var_assignment_29690' (line 728)
    tuple_var_assignment_29690_31059 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 728, 4), 'tuple_var_assignment_29690')
    # Assigning a type to the variable 'proj' (line 728)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 728, 9), 'proj', tuple_var_assignment_29690_31059)
    
    
    # Getting the type of 'k' (line 729)
    k_31060 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 729, 7), 'k')
    # Getting the type of 'n' (line 729)
    n_31061 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 729, 12), 'n')
    # Applying the binary operator '==' (line 729)
    result_eq_31062 = python_operator(stypy.reporting.localization.Localization(__file__, 729, 7), '==', k_31060, n_31061)
    
    # Testing the type of an if condition (line 729)
    if_condition_31063 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 729, 4), result_eq_31062)
    # Assigning a type to the variable 'if_condition_31063' (line 729)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 729, 4), 'if_condition_31063', if_condition_31063)
    # SSA begins for if statement (line 729)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 730):
    
    # Assigning a Call to a Name (line 730):
    
    # Call to array(...): (line 730)
    # Processing the call arguments (line 730)
    
    # Obtaining an instance of the builtin type 'list' (line 730)
    list_31066 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 730, 24), 'list')
    # Adding type elements to the builtin type 'list' instance (line 730)
    
    # Processing the call keyword arguments (line 730)
    str_31067 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 730, 34), 'str', 'float64')
    keyword_31068 = str_31067
    str_31069 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 730, 51), 'str', 'F')
    keyword_31070 = str_31069
    kwargs_31071 = {'dtype': keyword_31068, 'order': keyword_31070}
    # Getting the type of 'np' (line 730)
    np_31064 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 730, 15), 'np', False)
    # Obtaining the member 'array' of a type (line 730)
    array_31065 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 730, 15), np_31064, 'array')
    # Calling array(args, kwargs) (line 730)
    array_call_result_31072 = invoke(stypy.reporting.localization.Localization(__file__, 730, 15), array_31065, *[list_31066], **kwargs_31071)
    
    # Assigning a type to the variable 'proj' (line 730)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 730, 8), 'proj', array_call_result_31072)
    # SSA branch for the else part of an if statement (line 729)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Call to a Name (line 732):
    
    # Assigning a Call to a Name (line 732):
    
    # Call to reshape(...): (line 732)
    # Processing the call arguments (line 732)
    
    # Obtaining an instance of the builtin type 'tuple' (line 732)
    tuple_31075 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 732, 29), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 732)
    # Adding element type (line 732)
    # Getting the type of 'k' (line 732)
    k_31076 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 732, 29), 'k', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 732, 29), tuple_31075, k_31076)
    # Adding element type (line 732)
    # Getting the type of 'n' (line 732)
    n_31077 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 732, 32), 'n', False)
    # Getting the type of 'k' (line 732)
    k_31078 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 732, 34), 'k', False)
    # Applying the binary operator '-' (line 732)
    result_sub_31079 = python_operator(stypy.reporting.localization.Localization(__file__, 732, 32), '-', n_31077, k_31078)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 732, 29), tuple_31075, result_sub_31079)
    
    # Processing the call keyword arguments (line 732)
    str_31080 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 732, 44), 'str', 'F')
    keyword_31081 = str_31080
    kwargs_31082 = {'order': keyword_31081}
    # Getting the type of 'proj' (line 732)
    proj_31073 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 732, 15), 'proj', False)
    # Obtaining the member 'reshape' of a type (line 732)
    reshape_31074 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 732, 15), proj_31073, 'reshape')
    # Calling reshape(args, kwargs) (line 732)
    reshape_call_result_31083 = invoke(stypy.reporting.localization.Localization(__file__, 732, 15), reshape_31074, *[tuple_31075], **kwargs_31082)
    
    # Assigning a type to the variable 'proj' (line 732)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 732, 8), 'proj', reshape_call_result_31083)
    # SSA join for if statement (line 729)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Obtaining an instance of the builtin type 'tuple' (line 733)
    tuple_31084 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 733, 11), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 733)
    # Adding element type (line 733)
    # Getting the type of 'idx' (line 733)
    idx_31085 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 733, 11), 'idx')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 733, 11), tuple_31084, idx_31085)
    # Adding element type (line 733)
    # Getting the type of 'proj' (line 733)
    proj_31086 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 733, 16), 'proj')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 733, 11), tuple_31084, proj_31086)
    
    # Assigning a type to the variable 'stypy_return_type' (line 733)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 733, 4), 'stypy_return_type', tuple_31084)
    
    # ################# End of 'iddr_aid(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'iddr_aid' in the type store
    # Getting the type of 'stypy_return_type' (line 707)
    stypy_return_type_31087 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 707, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_31087)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'iddr_aid'
    return stypy_return_type_31087

# Assigning a type to the variable 'iddr_aid' (line 707)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 707, 0), 'iddr_aid', iddr_aid)

@norecursion
def iddr_aidi(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'iddr_aidi'
    module_type_store = module_type_store.open_function_context('iddr_aidi', 736, 0, False)
    
    # Passed parameters checking function
    iddr_aidi.stypy_localization = localization
    iddr_aidi.stypy_type_of_self = None
    iddr_aidi.stypy_type_store = module_type_store
    iddr_aidi.stypy_function_name = 'iddr_aidi'
    iddr_aidi.stypy_param_names_list = ['m', 'n', 'k']
    iddr_aidi.stypy_varargs_param_name = None
    iddr_aidi.stypy_kwargs_param_name = None
    iddr_aidi.stypy_call_defaults = defaults
    iddr_aidi.stypy_call_varargs = varargs
    iddr_aidi.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'iddr_aidi', ['m', 'n', 'k'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'iddr_aidi', localization, ['m', 'n', 'k'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'iddr_aidi(...)' code ##################

    str_31088 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 753, (-1)), 'str', '\n    Initialize array for :func:`iddr_aid`.\n\n    :param m:\n        Matrix row dimension.\n    :type m: int\n    :param n:\n        Matrix column dimension.\n    :type n: int\n    :param k:\n        Rank of ID.\n    :type k: int\n\n    :return:\n        Initialization array to be used by :func:`iddr_aid`.\n    :rtype: :class:`numpy.ndarray`\n    ')
    
    # Call to iddr_aidi(...): (line 754)
    # Processing the call arguments (line 754)
    # Getting the type of 'm' (line 754)
    m_31091 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 754, 25), 'm', False)
    # Getting the type of 'n' (line 754)
    n_31092 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 754, 28), 'n', False)
    # Getting the type of 'k' (line 754)
    k_31093 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 754, 31), 'k', False)
    # Processing the call keyword arguments (line 754)
    kwargs_31094 = {}
    # Getting the type of '_id' (line 754)
    _id_31089 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 754, 11), '_id', False)
    # Obtaining the member 'iddr_aidi' of a type (line 754)
    iddr_aidi_31090 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 754, 11), _id_31089, 'iddr_aidi')
    # Calling iddr_aidi(args, kwargs) (line 754)
    iddr_aidi_call_result_31095 = invoke(stypy.reporting.localization.Localization(__file__, 754, 11), iddr_aidi_31090, *[m_31091, n_31092, k_31093], **kwargs_31094)
    
    # Assigning a type to the variable 'stypy_return_type' (line 754)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 754, 4), 'stypy_return_type', iddr_aidi_call_result_31095)
    
    # ################# End of 'iddr_aidi(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'iddr_aidi' in the type store
    # Getting the type of 'stypy_return_type' (line 736)
    stypy_return_type_31096 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 736, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_31096)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'iddr_aidi'
    return stypy_return_type_31096

# Assigning a type to the variable 'iddr_aidi' (line 736)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 736, 0), 'iddr_aidi', iddr_aidi)

@norecursion
def iddr_asvd(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'iddr_asvd'
    module_type_store = module_type_store.open_function_context('iddr_asvd', 761, 0, False)
    
    # Passed parameters checking function
    iddr_asvd.stypy_localization = localization
    iddr_asvd.stypy_type_of_self = None
    iddr_asvd.stypy_type_store = module_type_store
    iddr_asvd.stypy_function_name = 'iddr_asvd'
    iddr_asvd.stypy_param_names_list = ['A', 'k']
    iddr_asvd.stypy_varargs_param_name = None
    iddr_asvd.stypy_kwargs_param_name = None
    iddr_asvd.stypy_call_defaults = defaults
    iddr_asvd.stypy_call_varargs = varargs
    iddr_asvd.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'iddr_asvd', ['A', 'k'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'iddr_asvd', localization, ['A', 'k'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'iddr_asvd(...)' code ##################

    str_31097 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 781, (-1)), 'str', '\n    Compute SVD of a real matrix to a specified rank using random sampling.\n\n    :param A:\n        Matrix.\n    :type A: :class:`numpy.ndarray`\n    :param k:\n        Rank of SVD.\n    :type k: int\n\n    :return:\n        Left singular vectors.\n    :rtype: :class:`numpy.ndarray`\n    :return:\n        Right singular vectors.\n    :rtype: :class:`numpy.ndarray`\n    :return:\n        Singular values.\n    :rtype: :class:`numpy.ndarray`\n    ')
    
    # Assigning a Call to a Name (line 782):
    
    # Assigning a Call to a Name (line 782):
    
    # Call to asfortranarray(...): (line 782)
    # Processing the call arguments (line 782)
    # Getting the type of 'A' (line 782)
    A_31100 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 782, 26), 'A', False)
    # Processing the call keyword arguments (line 782)
    kwargs_31101 = {}
    # Getting the type of 'np' (line 782)
    np_31098 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 782, 8), 'np', False)
    # Obtaining the member 'asfortranarray' of a type (line 782)
    asfortranarray_31099 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 782, 8), np_31098, 'asfortranarray')
    # Calling asfortranarray(args, kwargs) (line 782)
    asfortranarray_call_result_31102 = invoke(stypy.reporting.localization.Localization(__file__, 782, 8), asfortranarray_31099, *[A_31100], **kwargs_31101)
    
    # Assigning a type to the variable 'A' (line 782)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 782, 4), 'A', asfortranarray_call_result_31102)
    
    # Assigning a Attribute to a Tuple (line 783):
    
    # Assigning a Subscript to a Name (line 783):
    
    # Obtaining the type of the subscript
    int_31103 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 783, 4), 'int')
    # Getting the type of 'A' (line 783)
    A_31104 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 783, 11), 'A')
    # Obtaining the member 'shape' of a type (line 783)
    shape_31105 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 783, 11), A_31104, 'shape')
    # Obtaining the member '__getitem__' of a type (line 783)
    getitem___31106 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 783, 4), shape_31105, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 783)
    subscript_call_result_31107 = invoke(stypy.reporting.localization.Localization(__file__, 783, 4), getitem___31106, int_31103)
    
    # Assigning a type to the variable 'tuple_var_assignment_29691' (line 783)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 783, 4), 'tuple_var_assignment_29691', subscript_call_result_31107)
    
    # Assigning a Subscript to a Name (line 783):
    
    # Obtaining the type of the subscript
    int_31108 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 783, 4), 'int')
    # Getting the type of 'A' (line 783)
    A_31109 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 783, 11), 'A')
    # Obtaining the member 'shape' of a type (line 783)
    shape_31110 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 783, 11), A_31109, 'shape')
    # Obtaining the member '__getitem__' of a type (line 783)
    getitem___31111 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 783, 4), shape_31110, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 783)
    subscript_call_result_31112 = invoke(stypy.reporting.localization.Localization(__file__, 783, 4), getitem___31111, int_31108)
    
    # Assigning a type to the variable 'tuple_var_assignment_29692' (line 783)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 783, 4), 'tuple_var_assignment_29692', subscript_call_result_31112)
    
    # Assigning a Name to a Name (line 783):
    # Getting the type of 'tuple_var_assignment_29691' (line 783)
    tuple_var_assignment_29691_31113 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 783, 4), 'tuple_var_assignment_29691')
    # Assigning a type to the variable 'm' (line 783)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 783, 4), 'm', tuple_var_assignment_29691_31113)
    
    # Assigning a Name to a Name (line 783):
    # Getting the type of 'tuple_var_assignment_29692' (line 783)
    tuple_var_assignment_29692_31114 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 783, 4), 'tuple_var_assignment_29692')
    # Assigning a type to the variable 'n' (line 783)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 783, 7), 'n', tuple_var_assignment_29692_31114)
    
    # Assigning a Call to a Name (line 784):
    
    # Assigning a Call to a Name (line 784):
    
    # Call to empty(...): (line 784)
    # Processing the call arguments (line 784)
    int_31117 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 784, 18), 'int')
    # Getting the type of 'k' (line 784)
    k_31118 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 784, 20), 'k', False)
    # Applying the binary operator '*' (line 784)
    result_mul_31119 = python_operator(stypy.reporting.localization.Localization(__file__, 784, 18), '*', int_31117, k_31118)
    
    int_31120 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 784, 24), 'int')
    # Applying the binary operator '+' (line 784)
    result_add_31121 = python_operator(stypy.reporting.localization.Localization(__file__, 784, 18), '+', result_mul_31119, int_31120)
    
    # Getting the type of 'm' (line 784)
    m_31122 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 784, 28), 'm', False)
    # Applying the binary operator '*' (line 784)
    result_mul_31123 = python_operator(stypy.reporting.localization.Localization(__file__, 784, 17), '*', result_add_31121, m_31122)
    
    int_31124 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 784, 33), 'int')
    # Getting the type of 'k' (line 784)
    k_31125 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 784, 35), 'k', False)
    # Applying the binary operator '*' (line 784)
    result_mul_31126 = python_operator(stypy.reporting.localization.Localization(__file__, 784, 33), '*', int_31124, k_31125)
    
    int_31127 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 784, 39), 'int')
    # Applying the binary operator '+' (line 784)
    result_add_31128 = python_operator(stypy.reporting.localization.Localization(__file__, 784, 33), '+', result_mul_31126, int_31127)
    
    # Getting the type of 'n' (line 784)
    n_31129 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 784, 43), 'n', False)
    # Applying the binary operator '*' (line 784)
    result_mul_31130 = python_operator(stypy.reporting.localization.Localization(__file__, 784, 32), '*', result_add_31128, n_31129)
    
    # Applying the binary operator '+' (line 784)
    result_add_31131 = python_operator(stypy.reporting.localization.Localization(__file__, 784, 17), '+', result_mul_31123, result_mul_31130)
    
    int_31132 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 784, 47), 'int')
    # Getting the type of 'k' (line 784)
    k_31133 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 784, 50), 'k', False)
    int_31134 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 784, 53), 'int')
    # Applying the binary operator '**' (line 784)
    result_pow_31135 = python_operator(stypy.reporting.localization.Localization(__file__, 784, 50), '**', k_31133, int_31134)
    
    # Applying the binary operator '*' (line 784)
    result_mul_31136 = python_operator(stypy.reporting.localization.Localization(__file__, 784, 47), '*', int_31132, result_pow_31135)
    
    # Applying the binary operator '+' (line 784)
    result_add_31137 = python_operator(stypy.reporting.localization.Localization(__file__, 784, 45), '+', result_add_31131, result_mul_31136)
    
    int_31138 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 784, 57), 'int')
    # Applying the binary operator '+' (line 784)
    result_add_31139 = python_operator(stypy.reporting.localization.Localization(__file__, 784, 55), '+', result_add_31137, int_31138)
    
    # Processing the call keyword arguments (line 784)
    str_31140 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 784, 68), 'str', 'F')
    keyword_31141 = str_31140
    kwargs_31142 = {'order': keyword_31141}
    # Getting the type of 'np' (line 784)
    np_31115 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 784, 8), 'np', False)
    # Obtaining the member 'empty' of a type (line 784)
    empty_31116 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 784, 8), np_31115, 'empty')
    # Calling empty(args, kwargs) (line 784)
    empty_call_result_31143 = invoke(stypy.reporting.localization.Localization(__file__, 784, 8), empty_31116, *[result_add_31139], **kwargs_31142)
    
    # Assigning a type to the variable 'w' (line 784)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 784, 4), 'w', empty_call_result_31143)
    
    # Assigning a Call to a Name (line 785):
    
    # Assigning a Call to a Name (line 785):
    
    # Call to iddr_aidi(...): (line 785)
    # Processing the call arguments (line 785)
    # Getting the type of 'm' (line 785)
    m_31145 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 785, 19), 'm', False)
    # Getting the type of 'n' (line 785)
    n_31146 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 785, 22), 'n', False)
    # Getting the type of 'k' (line 785)
    k_31147 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 785, 25), 'k', False)
    # Processing the call keyword arguments (line 785)
    kwargs_31148 = {}
    # Getting the type of 'iddr_aidi' (line 785)
    iddr_aidi_31144 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 785, 9), 'iddr_aidi', False)
    # Calling iddr_aidi(args, kwargs) (line 785)
    iddr_aidi_call_result_31149 = invoke(stypy.reporting.localization.Localization(__file__, 785, 9), iddr_aidi_31144, *[m_31145, n_31146, k_31147], **kwargs_31148)
    
    # Assigning a type to the variable 'w_' (line 785)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 785, 4), 'w_', iddr_aidi_call_result_31149)
    
    # Assigning a Name to a Subscript (line 786):
    
    # Assigning a Name to a Subscript (line 786):
    # Getting the type of 'w_' (line 786)
    w__31150 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 786, 18), 'w_')
    # Getting the type of 'w' (line 786)
    w_31151 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 786, 4), 'w')
    # Getting the type of 'w_' (line 786)
    w__31152 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 786, 7), 'w_')
    # Obtaining the member 'size' of a type (line 786)
    size_31153 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 786, 7), w__31152, 'size')
    slice_31154 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 786, 4), None, size_31153, None)
    # Storing an element on a container (line 786)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 786, 4), w_31151, (slice_31154, w__31150))
    
    # Assigning a Call to a Tuple (line 787):
    
    # Assigning a Subscript to a Name (line 787):
    
    # Obtaining the type of the subscript
    int_31155 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 787, 4), 'int')
    
    # Call to iddr_asvd(...): (line 787)
    # Processing the call arguments (line 787)
    # Getting the type of 'A' (line 787)
    A_31158 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 787, 33), 'A', False)
    # Getting the type of 'k' (line 787)
    k_31159 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 787, 36), 'k', False)
    # Getting the type of 'w' (line 787)
    w_31160 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 787, 39), 'w', False)
    # Processing the call keyword arguments (line 787)
    kwargs_31161 = {}
    # Getting the type of '_id' (line 787)
    _id_31156 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 787, 19), '_id', False)
    # Obtaining the member 'iddr_asvd' of a type (line 787)
    iddr_asvd_31157 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 787, 19), _id_31156, 'iddr_asvd')
    # Calling iddr_asvd(args, kwargs) (line 787)
    iddr_asvd_call_result_31162 = invoke(stypy.reporting.localization.Localization(__file__, 787, 19), iddr_asvd_31157, *[A_31158, k_31159, w_31160], **kwargs_31161)
    
    # Obtaining the member '__getitem__' of a type (line 787)
    getitem___31163 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 787, 4), iddr_asvd_call_result_31162, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 787)
    subscript_call_result_31164 = invoke(stypy.reporting.localization.Localization(__file__, 787, 4), getitem___31163, int_31155)
    
    # Assigning a type to the variable 'tuple_var_assignment_29693' (line 787)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 787, 4), 'tuple_var_assignment_29693', subscript_call_result_31164)
    
    # Assigning a Subscript to a Name (line 787):
    
    # Obtaining the type of the subscript
    int_31165 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 787, 4), 'int')
    
    # Call to iddr_asvd(...): (line 787)
    # Processing the call arguments (line 787)
    # Getting the type of 'A' (line 787)
    A_31168 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 787, 33), 'A', False)
    # Getting the type of 'k' (line 787)
    k_31169 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 787, 36), 'k', False)
    # Getting the type of 'w' (line 787)
    w_31170 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 787, 39), 'w', False)
    # Processing the call keyword arguments (line 787)
    kwargs_31171 = {}
    # Getting the type of '_id' (line 787)
    _id_31166 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 787, 19), '_id', False)
    # Obtaining the member 'iddr_asvd' of a type (line 787)
    iddr_asvd_31167 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 787, 19), _id_31166, 'iddr_asvd')
    # Calling iddr_asvd(args, kwargs) (line 787)
    iddr_asvd_call_result_31172 = invoke(stypy.reporting.localization.Localization(__file__, 787, 19), iddr_asvd_31167, *[A_31168, k_31169, w_31170], **kwargs_31171)
    
    # Obtaining the member '__getitem__' of a type (line 787)
    getitem___31173 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 787, 4), iddr_asvd_call_result_31172, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 787)
    subscript_call_result_31174 = invoke(stypy.reporting.localization.Localization(__file__, 787, 4), getitem___31173, int_31165)
    
    # Assigning a type to the variable 'tuple_var_assignment_29694' (line 787)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 787, 4), 'tuple_var_assignment_29694', subscript_call_result_31174)
    
    # Assigning a Subscript to a Name (line 787):
    
    # Obtaining the type of the subscript
    int_31175 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 787, 4), 'int')
    
    # Call to iddr_asvd(...): (line 787)
    # Processing the call arguments (line 787)
    # Getting the type of 'A' (line 787)
    A_31178 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 787, 33), 'A', False)
    # Getting the type of 'k' (line 787)
    k_31179 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 787, 36), 'k', False)
    # Getting the type of 'w' (line 787)
    w_31180 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 787, 39), 'w', False)
    # Processing the call keyword arguments (line 787)
    kwargs_31181 = {}
    # Getting the type of '_id' (line 787)
    _id_31176 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 787, 19), '_id', False)
    # Obtaining the member 'iddr_asvd' of a type (line 787)
    iddr_asvd_31177 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 787, 19), _id_31176, 'iddr_asvd')
    # Calling iddr_asvd(args, kwargs) (line 787)
    iddr_asvd_call_result_31182 = invoke(stypy.reporting.localization.Localization(__file__, 787, 19), iddr_asvd_31177, *[A_31178, k_31179, w_31180], **kwargs_31181)
    
    # Obtaining the member '__getitem__' of a type (line 787)
    getitem___31183 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 787, 4), iddr_asvd_call_result_31182, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 787)
    subscript_call_result_31184 = invoke(stypy.reporting.localization.Localization(__file__, 787, 4), getitem___31183, int_31175)
    
    # Assigning a type to the variable 'tuple_var_assignment_29695' (line 787)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 787, 4), 'tuple_var_assignment_29695', subscript_call_result_31184)
    
    # Assigning a Subscript to a Name (line 787):
    
    # Obtaining the type of the subscript
    int_31185 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 787, 4), 'int')
    
    # Call to iddr_asvd(...): (line 787)
    # Processing the call arguments (line 787)
    # Getting the type of 'A' (line 787)
    A_31188 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 787, 33), 'A', False)
    # Getting the type of 'k' (line 787)
    k_31189 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 787, 36), 'k', False)
    # Getting the type of 'w' (line 787)
    w_31190 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 787, 39), 'w', False)
    # Processing the call keyword arguments (line 787)
    kwargs_31191 = {}
    # Getting the type of '_id' (line 787)
    _id_31186 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 787, 19), '_id', False)
    # Obtaining the member 'iddr_asvd' of a type (line 787)
    iddr_asvd_31187 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 787, 19), _id_31186, 'iddr_asvd')
    # Calling iddr_asvd(args, kwargs) (line 787)
    iddr_asvd_call_result_31192 = invoke(stypy.reporting.localization.Localization(__file__, 787, 19), iddr_asvd_31187, *[A_31188, k_31189, w_31190], **kwargs_31191)
    
    # Obtaining the member '__getitem__' of a type (line 787)
    getitem___31193 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 787, 4), iddr_asvd_call_result_31192, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 787)
    subscript_call_result_31194 = invoke(stypy.reporting.localization.Localization(__file__, 787, 4), getitem___31193, int_31185)
    
    # Assigning a type to the variable 'tuple_var_assignment_29696' (line 787)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 787, 4), 'tuple_var_assignment_29696', subscript_call_result_31194)
    
    # Assigning a Name to a Name (line 787):
    # Getting the type of 'tuple_var_assignment_29693' (line 787)
    tuple_var_assignment_29693_31195 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 787, 4), 'tuple_var_assignment_29693')
    # Assigning a type to the variable 'U' (line 787)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 787, 4), 'U', tuple_var_assignment_29693_31195)
    
    # Assigning a Name to a Name (line 787):
    # Getting the type of 'tuple_var_assignment_29694' (line 787)
    tuple_var_assignment_29694_31196 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 787, 4), 'tuple_var_assignment_29694')
    # Assigning a type to the variable 'V' (line 787)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 787, 7), 'V', tuple_var_assignment_29694_31196)
    
    # Assigning a Name to a Name (line 787):
    # Getting the type of 'tuple_var_assignment_29695' (line 787)
    tuple_var_assignment_29695_31197 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 787, 4), 'tuple_var_assignment_29695')
    # Assigning a type to the variable 'S' (line 787)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 787, 10), 'S', tuple_var_assignment_29695_31197)
    
    # Assigning a Name to a Name (line 787):
    # Getting the type of 'tuple_var_assignment_29696' (line 787)
    tuple_var_assignment_29696_31198 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 787, 4), 'tuple_var_assignment_29696')
    # Assigning a type to the variable 'ier' (line 787)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 787, 13), 'ier', tuple_var_assignment_29696_31198)
    
    
    # Getting the type of 'ier' (line 788)
    ier_31199 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 788, 7), 'ier')
    int_31200 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 788, 14), 'int')
    # Applying the binary operator '!=' (line 788)
    result_ne_31201 = python_operator(stypy.reporting.localization.Localization(__file__, 788, 7), '!=', ier_31199, int_31200)
    
    # Testing the type of an if condition (line 788)
    if_condition_31202 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 788, 4), result_ne_31201)
    # Assigning a type to the variable 'if_condition_31202' (line 788)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 788, 4), 'if_condition_31202', if_condition_31202)
    # SSA begins for if statement (line 788)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # Getting the type of '_RETCODE_ERROR' (line 789)
    _RETCODE_ERROR_31203 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 789, 14), '_RETCODE_ERROR')
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 789, 8), _RETCODE_ERROR_31203, 'raise parameter', BaseException)
    # SSA join for if statement (line 788)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Obtaining an instance of the builtin type 'tuple' (line 790)
    tuple_31204 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 790, 11), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 790)
    # Adding element type (line 790)
    # Getting the type of 'U' (line 790)
    U_31205 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 790, 11), 'U')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 790, 11), tuple_31204, U_31205)
    # Adding element type (line 790)
    # Getting the type of 'V' (line 790)
    V_31206 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 790, 14), 'V')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 790, 11), tuple_31204, V_31206)
    # Adding element type (line 790)
    # Getting the type of 'S' (line 790)
    S_31207 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 790, 17), 'S')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 790, 11), tuple_31204, S_31207)
    
    # Assigning a type to the variable 'stypy_return_type' (line 790)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 790, 4), 'stypy_return_type', tuple_31204)
    
    # ################# End of 'iddr_asvd(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'iddr_asvd' in the type store
    # Getting the type of 'stypy_return_type' (line 761)
    stypy_return_type_31208 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 761, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_31208)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'iddr_asvd'
    return stypy_return_type_31208

# Assigning a type to the variable 'iddr_asvd' (line 761)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 761, 0), 'iddr_asvd', iddr_asvd)

@norecursion
def iddr_rid(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'iddr_rid'
    module_type_store = module_type_store.open_function_context('iddr_rid', 797, 0, False)
    
    # Passed parameters checking function
    iddr_rid.stypy_localization = localization
    iddr_rid.stypy_type_of_self = None
    iddr_rid.stypy_type_store = module_type_store
    iddr_rid.stypy_function_name = 'iddr_rid'
    iddr_rid.stypy_param_names_list = ['m', 'n', 'matvect', 'k']
    iddr_rid.stypy_varargs_param_name = None
    iddr_rid.stypy_kwargs_param_name = None
    iddr_rid.stypy_call_defaults = defaults
    iddr_rid.stypy_call_varargs = varargs
    iddr_rid.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'iddr_rid', ['m', 'n', 'matvect', 'k'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'iddr_rid', localization, ['m', 'n', 'matvect', 'k'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'iddr_rid(...)' code ##################

    str_31209 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 823, (-1)), 'str', '\n    Compute ID of a real matrix to a specified rank using random matrix-vector\n    multiplication.\n\n    :param m:\n        Matrix row dimension.\n    :type m: int\n    :param n:\n        Matrix column dimension.\n    :type n: int\n    :param matvect:\n        Function to apply the matrix transpose to a vector, with call signature\n        `y = matvect(x)`, where `x` and `y` are the input and output vectors,\n        respectively.\n    :type matvect: function\n    :param k:\n        Rank of ID.\n    :type k: int\n\n    :return:\n        Column index array.\n    :rtype: :class:`numpy.ndarray`\n    :return:\n        Interpolation coefficients.\n    :rtype: :class:`numpy.ndarray`\n    ')
    
    # Assigning a Call to a Tuple (line 824):
    
    # Assigning a Subscript to a Name (line 824):
    
    # Obtaining the type of the subscript
    int_31210 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 824, 4), 'int')
    
    # Call to iddr_rid(...): (line 824)
    # Processing the call arguments (line 824)
    # Getting the type of 'm' (line 824)
    m_31213 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 824, 29), 'm', False)
    # Getting the type of 'n' (line 824)
    n_31214 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 824, 32), 'n', False)
    # Getting the type of 'matvect' (line 824)
    matvect_31215 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 824, 35), 'matvect', False)
    # Getting the type of 'k' (line 824)
    k_31216 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 824, 44), 'k', False)
    # Processing the call keyword arguments (line 824)
    kwargs_31217 = {}
    # Getting the type of '_id' (line 824)
    _id_31211 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 824, 16), '_id', False)
    # Obtaining the member 'iddr_rid' of a type (line 824)
    iddr_rid_31212 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 824, 16), _id_31211, 'iddr_rid')
    # Calling iddr_rid(args, kwargs) (line 824)
    iddr_rid_call_result_31218 = invoke(stypy.reporting.localization.Localization(__file__, 824, 16), iddr_rid_31212, *[m_31213, n_31214, matvect_31215, k_31216], **kwargs_31217)
    
    # Obtaining the member '__getitem__' of a type (line 824)
    getitem___31219 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 824, 4), iddr_rid_call_result_31218, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 824)
    subscript_call_result_31220 = invoke(stypy.reporting.localization.Localization(__file__, 824, 4), getitem___31219, int_31210)
    
    # Assigning a type to the variable 'tuple_var_assignment_29697' (line 824)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 824, 4), 'tuple_var_assignment_29697', subscript_call_result_31220)
    
    # Assigning a Subscript to a Name (line 824):
    
    # Obtaining the type of the subscript
    int_31221 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 824, 4), 'int')
    
    # Call to iddr_rid(...): (line 824)
    # Processing the call arguments (line 824)
    # Getting the type of 'm' (line 824)
    m_31224 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 824, 29), 'm', False)
    # Getting the type of 'n' (line 824)
    n_31225 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 824, 32), 'n', False)
    # Getting the type of 'matvect' (line 824)
    matvect_31226 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 824, 35), 'matvect', False)
    # Getting the type of 'k' (line 824)
    k_31227 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 824, 44), 'k', False)
    # Processing the call keyword arguments (line 824)
    kwargs_31228 = {}
    # Getting the type of '_id' (line 824)
    _id_31222 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 824, 16), '_id', False)
    # Obtaining the member 'iddr_rid' of a type (line 824)
    iddr_rid_31223 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 824, 16), _id_31222, 'iddr_rid')
    # Calling iddr_rid(args, kwargs) (line 824)
    iddr_rid_call_result_31229 = invoke(stypy.reporting.localization.Localization(__file__, 824, 16), iddr_rid_31223, *[m_31224, n_31225, matvect_31226, k_31227], **kwargs_31228)
    
    # Obtaining the member '__getitem__' of a type (line 824)
    getitem___31230 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 824, 4), iddr_rid_call_result_31229, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 824)
    subscript_call_result_31231 = invoke(stypy.reporting.localization.Localization(__file__, 824, 4), getitem___31230, int_31221)
    
    # Assigning a type to the variable 'tuple_var_assignment_29698' (line 824)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 824, 4), 'tuple_var_assignment_29698', subscript_call_result_31231)
    
    # Assigning a Name to a Name (line 824):
    # Getting the type of 'tuple_var_assignment_29697' (line 824)
    tuple_var_assignment_29697_31232 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 824, 4), 'tuple_var_assignment_29697')
    # Assigning a type to the variable 'idx' (line 824)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 824, 4), 'idx', tuple_var_assignment_29697_31232)
    
    # Assigning a Name to a Name (line 824):
    # Getting the type of 'tuple_var_assignment_29698' (line 824)
    tuple_var_assignment_29698_31233 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 824, 4), 'tuple_var_assignment_29698')
    # Assigning a type to the variable 'proj' (line 824)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 824, 9), 'proj', tuple_var_assignment_29698_31233)
    
    # Assigning a Call to a Name (line 825):
    
    # Assigning a Call to a Name (line 825):
    
    # Call to reshape(...): (line 825)
    # Processing the call arguments (line 825)
    
    # Obtaining an instance of the builtin type 'tuple' (line 825)
    tuple_31244 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 825, 35), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 825)
    # Adding element type (line 825)
    # Getting the type of 'k' (line 825)
    k_31245 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 825, 35), 'k', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 825, 35), tuple_31244, k_31245)
    # Adding element type (line 825)
    # Getting the type of 'n' (line 825)
    n_31246 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 825, 38), 'n', False)
    # Getting the type of 'k' (line 825)
    k_31247 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 825, 40), 'k', False)
    # Applying the binary operator '-' (line 825)
    result_sub_31248 = python_operator(stypy.reporting.localization.Localization(__file__, 825, 38), '-', n_31246, k_31247)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 825, 35), tuple_31244, result_sub_31248)
    
    # Processing the call keyword arguments (line 825)
    str_31249 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 825, 50), 'str', 'F')
    keyword_31250 = str_31249
    kwargs_31251 = {'order': keyword_31250}
    
    # Obtaining the type of the subscript
    # Getting the type of 'k' (line 825)
    k_31234 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 825, 17), 'k', False)
    # Getting the type of 'n' (line 825)
    n_31235 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 825, 20), 'n', False)
    # Getting the type of 'k' (line 825)
    k_31236 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 825, 22), 'k', False)
    # Applying the binary operator '-' (line 825)
    result_sub_31237 = python_operator(stypy.reporting.localization.Localization(__file__, 825, 20), '-', n_31235, k_31236)
    
    # Applying the binary operator '*' (line 825)
    result_mul_31238 = python_operator(stypy.reporting.localization.Localization(__file__, 825, 17), '*', k_31234, result_sub_31237)
    
    slice_31239 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 825, 11), None, result_mul_31238, None)
    # Getting the type of 'proj' (line 825)
    proj_31240 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 825, 11), 'proj', False)
    # Obtaining the member '__getitem__' of a type (line 825)
    getitem___31241 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 825, 11), proj_31240, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 825)
    subscript_call_result_31242 = invoke(stypy.reporting.localization.Localization(__file__, 825, 11), getitem___31241, slice_31239)
    
    # Obtaining the member 'reshape' of a type (line 825)
    reshape_31243 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 825, 11), subscript_call_result_31242, 'reshape')
    # Calling reshape(args, kwargs) (line 825)
    reshape_call_result_31252 = invoke(stypy.reporting.localization.Localization(__file__, 825, 11), reshape_31243, *[tuple_31244], **kwargs_31251)
    
    # Assigning a type to the variable 'proj' (line 825)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 825, 4), 'proj', reshape_call_result_31252)
    
    # Obtaining an instance of the builtin type 'tuple' (line 826)
    tuple_31253 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 826, 11), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 826)
    # Adding element type (line 826)
    # Getting the type of 'idx' (line 826)
    idx_31254 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 826, 11), 'idx')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 826, 11), tuple_31253, idx_31254)
    # Adding element type (line 826)
    # Getting the type of 'proj' (line 826)
    proj_31255 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 826, 16), 'proj')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 826, 11), tuple_31253, proj_31255)
    
    # Assigning a type to the variable 'stypy_return_type' (line 826)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 826, 4), 'stypy_return_type', tuple_31253)
    
    # ################# End of 'iddr_rid(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'iddr_rid' in the type store
    # Getting the type of 'stypy_return_type' (line 797)
    stypy_return_type_31256 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 797, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_31256)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'iddr_rid'
    return stypy_return_type_31256

# Assigning a type to the variable 'iddr_rid' (line 797)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 797, 0), 'iddr_rid', iddr_rid)

@norecursion
def iddr_rsvd(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'iddr_rsvd'
    module_type_store = module_type_store.open_function_context('iddr_rsvd', 833, 0, False)
    
    # Passed parameters checking function
    iddr_rsvd.stypy_localization = localization
    iddr_rsvd.stypy_type_of_self = None
    iddr_rsvd.stypy_type_store = module_type_store
    iddr_rsvd.stypy_function_name = 'iddr_rsvd'
    iddr_rsvd.stypy_param_names_list = ['m', 'n', 'matvect', 'matvec', 'k']
    iddr_rsvd.stypy_varargs_param_name = None
    iddr_rsvd.stypy_kwargs_param_name = None
    iddr_rsvd.stypy_call_defaults = defaults
    iddr_rsvd.stypy_call_varargs = varargs
    iddr_rsvd.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'iddr_rsvd', ['m', 'n', 'matvect', 'matvec', 'k'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'iddr_rsvd', localization, ['m', 'n', 'matvect', 'matvec', 'k'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'iddr_rsvd(...)' code ##################

    str_31257 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 867, (-1)), 'str', '\n    Compute SVD of a real matrix to a specified rank using random matrix-vector\n    multiplication.\n\n    :param m:\n        Matrix row dimension.\n    :type m: int\n    :param n:\n        Matrix column dimension.\n    :type n: int\n    :param matvect:\n        Function to apply the matrix transpose to a vector, with call signature\n        `y = matvect(x)`, where `x` and `y` are the input and output vectors,\n        respectively.\n    :type matvect: function\n    :param matvec:\n        Function to apply the matrix to a vector, with call signature\n        `y = matvec(x)`, where `x` and `y` are the input and output vectors,\n        respectively.\n    :type matvec: function\n    :param k:\n        Rank of SVD.\n    :type k: int\n\n    :return:\n        Left singular vectors.\n    :rtype: :class:`numpy.ndarray`\n    :return:\n        Right singular vectors.\n    :rtype: :class:`numpy.ndarray`\n    :return:\n        Singular values.\n    :rtype: :class:`numpy.ndarray`\n    ')
    
    # Assigning a Call to a Tuple (line 868):
    
    # Assigning a Subscript to a Name (line 868):
    
    # Obtaining the type of the subscript
    int_31258 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 868, 4), 'int')
    
    # Call to iddr_rsvd(...): (line 868)
    # Processing the call arguments (line 868)
    # Getting the type of 'm' (line 868)
    m_31261 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 868, 33), 'm', False)
    # Getting the type of 'n' (line 868)
    n_31262 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 868, 36), 'n', False)
    # Getting the type of 'matvect' (line 868)
    matvect_31263 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 868, 39), 'matvect', False)
    # Getting the type of 'matvec' (line 868)
    matvec_31264 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 868, 48), 'matvec', False)
    # Getting the type of 'k' (line 868)
    k_31265 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 868, 56), 'k', False)
    # Processing the call keyword arguments (line 868)
    kwargs_31266 = {}
    # Getting the type of '_id' (line 868)
    _id_31259 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 868, 19), '_id', False)
    # Obtaining the member 'iddr_rsvd' of a type (line 868)
    iddr_rsvd_31260 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 868, 19), _id_31259, 'iddr_rsvd')
    # Calling iddr_rsvd(args, kwargs) (line 868)
    iddr_rsvd_call_result_31267 = invoke(stypy.reporting.localization.Localization(__file__, 868, 19), iddr_rsvd_31260, *[m_31261, n_31262, matvect_31263, matvec_31264, k_31265], **kwargs_31266)
    
    # Obtaining the member '__getitem__' of a type (line 868)
    getitem___31268 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 868, 4), iddr_rsvd_call_result_31267, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 868)
    subscript_call_result_31269 = invoke(stypy.reporting.localization.Localization(__file__, 868, 4), getitem___31268, int_31258)
    
    # Assigning a type to the variable 'tuple_var_assignment_29699' (line 868)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 868, 4), 'tuple_var_assignment_29699', subscript_call_result_31269)
    
    # Assigning a Subscript to a Name (line 868):
    
    # Obtaining the type of the subscript
    int_31270 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 868, 4), 'int')
    
    # Call to iddr_rsvd(...): (line 868)
    # Processing the call arguments (line 868)
    # Getting the type of 'm' (line 868)
    m_31273 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 868, 33), 'm', False)
    # Getting the type of 'n' (line 868)
    n_31274 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 868, 36), 'n', False)
    # Getting the type of 'matvect' (line 868)
    matvect_31275 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 868, 39), 'matvect', False)
    # Getting the type of 'matvec' (line 868)
    matvec_31276 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 868, 48), 'matvec', False)
    # Getting the type of 'k' (line 868)
    k_31277 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 868, 56), 'k', False)
    # Processing the call keyword arguments (line 868)
    kwargs_31278 = {}
    # Getting the type of '_id' (line 868)
    _id_31271 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 868, 19), '_id', False)
    # Obtaining the member 'iddr_rsvd' of a type (line 868)
    iddr_rsvd_31272 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 868, 19), _id_31271, 'iddr_rsvd')
    # Calling iddr_rsvd(args, kwargs) (line 868)
    iddr_rsvd_call_result_31279 = invoke(stypy.reporting.localization.Localization(__file__, 868, 19), iddr_rsvd_31272, *[m_31273, n_31274, matvect_31275, matvec_31276, k_31277], **kwargs_31278)
    
    # Obtaining the member '__getitem__' of a type (line 868)
    getitem___31280 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 868, 4), iddr_rsvd_call_result_31279, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 868)
    subscript_call_result_31281 = invoke(stypy.reporting.localization.Localization(__file__, 868, 4), getitem___31280, int_31270)
    
    # Assigning a type to the variable 'tuple_var_assignment_29700' (line 868)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 868, 4), 'tuple_var_assignment_29700', subscript_call_result_31281)
    
    # Assigning a Subscript to a Name (line 868):
    
    # Obtaining the type of the subscript
    int_31282 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 868, 4), 'int')
    
    # Call to iddr_rsvd(...): (line 868)
    # Processing the call arguments (line 868)
    # Getting the type of 'm' (line 868)
    m_31285 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 868, 33), 'm', False)
    # Getting the type of 'n' (line 868)
    n_31286 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 868, 36), 'n', False)
    # Getting the type of 'matvect' (line 868)
    matvect_31287 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 868, 39), 'matvect', False)
    # Getting the type of 'matvec' (line 868)
    matvec_31288 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 868, 48), 'matvec', False)
    # Getting the type of 'k' (line 868)
    k_31289 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 868, 56), 'k', False)
    # Processing the call keyword arguments (line 868)
    kwargs_31290 = {}
    # Getting the type of '_id' (line 868)
    _id_31283 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 868, 19), '_id', False)
    # Obtaining the member 'iddr_rsvd' of a type (line 868)
    iddr_rsvd_31284 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 868, 19), _id_31283, 'iddr_rsvd')
    # Calling iddr_rsvd(args, kwargs) (line 868)
    iddr_rsvd_call_result_31291 = invoke(stypy.reporting.localization.Localization(__file__, 868, 19), iddr_rsvd_31284, *[m_31285, n_31286, matvect_31287, matvec_31288, k_31289], **kwargs_31290)
    
    # Obtaining the member '__getitem__' of a type (line 868)
    getitem___31292 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 868, 4), iddr_rsvd_call_result_31291, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 868)
    subscript_call_result_31293 = invoke(stypy.reporting.localization.Localization(__file__, 868, 4), getitem___31292, int_31282)
    
    # Assigning a type to the variable 'tuple_var_assignment_29701' (line 868)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 868, 4), 'tuple_var_assignment_29701', subscript_call_result_31293)
    
    # Assigning a Subscript to a Name (line 868):
    
    # Obtaining the type of the subscript
    int_31294 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 868, 4), 'int')
    
    # Call to iddr_rsvd(...): (line 868)
    # Processing the call arguments (line 868)
    # Getting the type of 'm' (line 868)
    m_31297 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 868, 33), 'm', False)
    # Getting the type of 'n' (line 868)
    n_31298 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 868, 36), 'n', False)
    # Getting the type of 'matvect' (line 868)
    matvect_31299 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 868, 39), 'matvect', False)
    # Getting the type of 'matvec' (line 868)
    matvec_31300 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 868, 48), 'matvec', False)
    # Getting the type of 'k' (line 868)
    k_31301 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 868, 56), 'k', False)
    # Processing the call keyword arguments (line 868)
    kwargs_31302 = {}
    # Getting the type of '_id' (line 868)
    _id_31295 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 868, 19), '_id', False)
    # Obtaining the member 'iddr_rsvd' of a type (line 868)
    iddr_rsvd_31296 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 868, 19), _id_31295, 'iddr_rsvd')
    # Calling iddr_rsvd(args, kwargs) (line 868)
    iddr_rsvd_call_result_31303 = invoke(stypy.reporting.localization.Localization(__file__, 868, 19), iddr_rsvd_31296, *[m_31297, n_31298, matvect_31299, matvec_31300, k_31301], **kwargs_31302)
    
    # Obtaining the member '__getitem__' of a type (line 868)
    getitem___31304 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 868, 4), iddr_rsvd_call_result_31303, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 868)
    subscript_call_result_31305 = invoke(stypy.reporting.localization.Localization(__file__, 868, 4), getitem___31304, int_31294)
    
    # Assigning a type to the variable 'tuple_var_assignment_29702' (line 868)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 868, 4), 'tuple_var_assignment_29702', subscript_call_result_31305)
    
    # Assigning a Name to a Name (line 868):
    # Getting the type of 'tuple_var_assignment_29699' (line 868)
    tuple_var_assignment_29699_31306 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 868, 4), 'tuple_var_assignment_29699')
    # Assigning a type to the variable 'U' (line 868)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 868, 4), 'U', tuple_var_assignment_29699_31306)
    
    # Assigning a Name to a Name (line 868):
    # Getting the type of 'tuple_var_assignment_29700' (line 868)
    tuple_var_assignment_29700_31307 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 868, 4), 'tuple_var_assignment_29700')
    # Assigning a type to the variable 'V' (line 868)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 868, 7), 'V', tuple_var_assignment_29700_31307)
    
    # Assigning a Name to a Name (line 868):
    # Getting the type of 'tuple_var_assignment_29701' (line 868)
    tuple_var_assignment_29701_31308 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 868, 4), 'tuple_var_assignment_29701')
    # Assigning a type to the variable 'S' (line 868)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 868, 10), 'S', tuple_var_assignment_29701_31308)
    
    # Assigning a Name to a Name (line 868):
    # Getting the type of 'tuple_var_assignment_29702' (line 868)
    tuple_var_assignment_29702_31309 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 868, 4), 'tuple_var_assignment_29702')
    # Assigning a type to the variable 'ier' (line 868)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 868, 13), 'ier', tuple_var_assignment_29702_31309)
    
    
    # Getting the type of 'ier' (line 869)
    ier_31310 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 869, 7), 'ier')
    int_31311 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 869, 14), 'int')
    # Applying the binary operator '!=' (line 869)
    result_ne_31312 = python_operator(stypy.reporting.localization.Localization(__file__, 869, 7), '!=', ier_31310, int_31311)
    
    # Testing the type of an if condition (line 869)
    if_condition_31313 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 869, 4), result_ne_31312)
    # Assigning a type to the variable 'if_condition_31313' (line 869)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 869, 4), 'if_condition_31313', if_condition_31313)
    # SSA begins for if statement (line 869)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # Getting the type of '_RETCODE_ERROR' (line 870)
    _RETCODE_ERROR_31314 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 870, 14), '_RETCODE_ERROR')
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 870, 8), _RETCODE_ERROR_31314, 'raise parameter', BaseException)
    # SSA join for if statement (line 869)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Obtaining an instance of the builtin type 'tuple' (line 871)
    tuple_31315 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 871, 11), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 871)
    # Adding element type (line 871)
    # Getting the type of 'U' (line 871)
    U_31316 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 871, 11), 'U')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 871, 11), tuple_31315, U_31316)
    # Adding element type (line 871)
    # Getting the type of 'V' (line 871)
    V_31317 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 871, 14), 'V')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 871, 11), tuple_31315, V_31317)
    # Adding element type (line 871)
    # Getting the type of 'S' (line 871)
    S_31318 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 871, 17), 'S')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 871, 11), tuple_31315, S_31318)
    
    # Assigning a type to the variable 'stypy_return_type' (line 871)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 871, 4), 'stypy_return_type', tuple_31315)
    
    # ################# End of 'iddr_rsvd(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'iddr_rsvd' in the type store
    # Getting the type of 'stypy_return_type' (line 833)
    stypy_return_type_31319 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 833, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_31319)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'iddr_rsvd'
    return stypy_return_type_31319

# Assigning a type to the variable 'iddr_rsvd' (line 833)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 833, 0), 'iddr_rsvd', iddr_rsvd)

@norecursion
def idz_frm(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'idz_frm'
    module_type_store = module_type_store.open_function_context('idz_frm', 878, 0, False)
    
    # Passed parameters checking function
    idz_frm.stypy_localization = localization
    idz_frm.stypy_type_of_self = None
    idz_frm.stypy_type_store = module_type_store
    idz_frm.stypy_function_name = 'idz_frm'
    idz_frm.stypy_param_names_list = ['n', 'w', 'x']
    idz_frm.stypy_varargs_param_name = None
    idz_frm.stypy_kwargs_param_name = None
    idz_frm.stypy_call_defaults = defaults
    idz_frm.stypy_call_varargs = varargs
    idz_frm.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'idz_frm', ['n', 'w', 'x'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'idz_frm', localization, ['n', 'w', 'x'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'idz_frm(...)' code ##################

    str_31320 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 903, (-1)), 'str', "\n    Transform complex vector via a composition of Rokhlin's random transform,\n    random subselection, and an FFT.\n\n    In contrast to :func:`idz_sfrm`, this routine works best when the length of\n    the transformed vector is the power-of-two integer output by\n    :func:`idz_frmi`, or when the length is not specified but instead\n    determined a posteriori from the output. The returned transformed vector is\n    randomly permuted.\n\n    :param n:\n        Greatest power-of-two integer satisfying `n <= x.size` as obtained from\n        :func:`idz_frmi`; `n` is also the length of the output vector.\n    :type n: int\n    :param w:\n        Initialization array constructed by :func:`idz_frmi`.\n    :type w: :class:`numpy.ndarray`\n    :param x:\n        Vector to be transformed.\n    :type x: :class:`numpy.ndarray`\n\n    :return:\n        Transformed vector.\n    :rtype: :class:`numpy.ndarray`\n    ")
    
    # Call to idz_frm(...): (line 904)
    # Processing the call arguments (line 904)
    # Getting the type of 'n' (line 904)
    n_31323 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 904, 23), 'n', False)
    # Getting the type of 'w' (line 904)
    w_31324 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 904, 26), 'w', False)
    # Getting the type of 'x' (line 904)
    x_31325 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 904, 29), 'x', False)
    # Processing the call keyword arguments (line 904)
    kwargs_31326 = {}
    # Getting the type of '_id' (line 904)
    _id_31321 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 904, 11), '_id', False)
    # Obtaining the member 'idz_frm' of a type (line 904)
    idz_frm_31322 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 904, 11), _id_31321, 'idz_frm')
    # Calling idz_frm(args, kwargs) (line 904)
    idz_frm_call_result_31327 = invoke(stypy.reporting.localization.Localization(__file__, 904, 11), idz_frm_31322, *[n_31323, w_31324, x_31325], **kwargs_31326)
    
    # Assigning a type to the variable 'stypy_return_type' (line 904)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 904, 4), 'stypy_return_type', idz_frm_call_result_31327)
    
    # ################# End of 'idz_frm(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'idz_frm' in the type store
    # Getting the type of 'stypy_return_type' (line 878)
    stypy_return_type_31328 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 878, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_31328)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'idz_frm'
    return stypy_return_type_31328

# Assigning a type to the variable 'idz_frm' (line 878)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 878, 0), 'idz_frm', idz_frm)

@norecursion
def idz_sfrm(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'idz_sfrm'
    module_type_store = module_type_store.open_function_context('idz_sfrm', 907, 0, False)
    
    # Passed parameters checking function
    idz_sfrm.stypy_localization = localization
    idz_sfrm.stypy_type_of_self = None
    idz_sfrm.stypy_type_store = module_type_store
    idz_sfrm.stypy_function_name = 'idz_sfrm'
    idz_sfrm.stypy_param_names_list = ['l', 'n', 'w', 'x']
    idz_sfrm.stypy_varargs_param_name = None
    idz_sfrm.stypy_kwargs_param_name = None
    idz_sfrm.stypy_call_defaults = defaults
    idz_sfrm.stypy_call_varargs = varargs
    idz_sfrm.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'idz_sfrm', ['l', 'n', 'w', 'x'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'idz_sfrm', localization, ['l', 'n', 'w', 'x'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'idz_sfrm(...)' code ##################

    str_31329 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 932, (-1)), 'str', "\n    Transform complex vector via a composition of Rokhlin's random transform,\n    random subselection, and an FFT.\n\n    In contrast to :func:`idz_frm`, this routine works best when the length of\n    the transformed vector is known a priori.\n\n    :param l:\n        Length of transformed vector, satisfying `l <= n`.\n    :type l: int\n    :param n:\n        Greatest power-of-two integer satisfying `n <= x.size` as obtained from\n        :func:`idz_sfrmi`.\n    :type n: int\n    :param w:\n        Initialization array constructed by :func:`idd_sfrmi`.\n    :type w: :class:`numpy.ndarray`\n    :param x:\n        Vector to be transformed.\n    :type x: :class:`numpy.ndarray`\n\n    :return:\n        Transformed vector.\n    :rtype: :class:`numpy.ndarray`\n    ")
    
    # Call to idz_sfrm(...): (line 933)
    # Processing the call arguments (line 933)
    # Getting the type of 'l' (line 933)
    l_31332 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 933, 24), 'l', False)
    # Getting the type of 'n' (line 933)
    n_31333 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 933, 27), 'n', False)
    # Getting the type of 'w' (line 933)
    w_31334 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 933, 30), 'w', False)
    # Getting the type of 'x' (line 933)
    x_31335 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 933, 33), 'x', False)
    # Processing the call keyword arguments (line 933)
    kwargs_31336 = {}
    # Getting the type of '_id' (line 933)
    _id_31330 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 933, 11), '_id', False)
    # Obtaining the member 'idz_sfrm' of a type (line 933)
    idz_sfrm_31331 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 933, 11), _id_31330, 'idz_sfrm')
    # Calling idz_sfrm(args, kwargs) (line 933)
    idz_sfrm_call_result_31337 = invoke(stypy.reporting.localization.Localization(__file__, 933, 11), idz_sfrm_31331, *[l_31332, n_31333, w_31334, x_31335], **kwargs_31336)
    
    # Assigning a type to the variable 'stypy_return_type' (line 933)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 933, 4), 'stypy_return_type', idz_sfrm_call_result_31337)
    
    # ################# End of 'idz_sfrm(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'idz_sfrm' in the type store
    # Getting the type of 'stypy_return_type' (line 907)
    stypy_return_type_31338 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 907, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_31338)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'idz_sfrm'
    return stypy_return_type_31338

# Assigning a type to the variable 'idz_sfrm' (line 907)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 907, 0), 'idz_sfrm', idz_sfrm)

@norecursion
def idz_frmi(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'idz_frmi'
    module_type_store = module_type_store.open_function_context('idz_frmi', 936, 0, False)
    
    # Passed parameters checking function
    idz_frmi.stypy_localization = localization
    idz_frmi.stypy_type_of_self = None
    idz_frmi.stypy_type_store = module_type_store
    idz_frmi.stypy_function_name = 'idz_frmi'
    idz_frmi.stypy_param_names_list = ['m']
    idz_frmi.stypy_varargs_param_name = None
    idz_frmi.stypy_kwargs_param_name = None
    idz_frmi.stypy_call_defaults = defaults
    idz_frmi.stypy_call_varargs = varargs
    idz_frmi.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'idz_frmi', ['m'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'idz_frmi', localization, ['m'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'idz_frmi(...)' code ##################

    str_31339 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 950, (-1)), 'str', '\n    Initialize data for :func:`idz_frm`.\n\n    :param m:\n        Length of vector to be transformed.\n    :type m: int\n\n    :return:\n        Greatest power-of-two integer `n` satisfying `n <= m`.\n    :rtype: int\n    :return:\n        Initialization array to be used by :func:`idz_frm`.\n    :rtype: :class:`numpy.ndarray`\n    ')
    
    # Call to idz_frmi(...): (line 951)
    # Processing the call arguments (line 951)
    # Getting the type of 'm' (line 951)
    m_31342 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 951, 24), 'm', False)
    # Processing the call keyword arguments (line 951)
    kwargs_31343 = {}
    # Getting the type of '_id' (line 951)
    _id_31340 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 951, 11), '_id', False)
    # Obtaining the member 'idz_frmi' of a type (line 951)
    idz_frmi_31341 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 951, 11), _id_31340, 'idz_frmi')
    # Calling idz_frmi(args, kwargs) (line 951)
    idz_frmi_call_result_31344 = invoke(stypy.reporting.localization.Localization(__file__, 951, 11), idz_frmi_31341, *[m_31342], **kwargs_31343)
    
    # Assigning a type to the variable 'stypy_return_type' (line 951)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 951, 4), 'stypy_return_type', idz_frmi_call_result_31344)
    
    # ################# End of 'idz_frmi(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'idz_frmi' in the type store
    # Getting the type of 'stypy_return_type' (line 936)
    stypy_return_type_31345 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 936, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_31345)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'idz_frmi'
    return stypy_return_type_31345

# Assigning a type to the variable 'idz_frmi' (line 936)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 936, 0), 'idz_frmi', idz_frmi)

@norecursion
def idz_sfrmi(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'idz_sfrmi'
    module_type_store = module_type_store.open_function_context('idz_sfrmi', 954, 0, False)
    
    # Passed parameters checking function
    idz_sfrmi.stypy_localization = localization
    idz_sfrmi.stypy_type_of_self = None
    idz_sfrmi.stypy_type_store = module_type_store
    idz_sfrmi.stypy_function_name = 'idz_sfrmi'
    idz_sfrmi.stypy_param_names_list = ['l', 'm']
    idz_sfrmi.stypy_varargs_param_name = None
    idz_sfrmi.stypy_kwargs_param_name = None
    idz_sfrmi.stypy_call_defaults = defaults
    idz_sfrmi.stypy_call_varargs = varargs
    idz_sfrmi.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'idz_sfrmi', ['l', 'm'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'idz_sfrmi', localization, ['l', 'm'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'idz_sfrmi(...)' code ##################

    str_31346 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 971, (-1)), 'str', '\n    Initialize data for :func:`idz_sfrm`.\n\n    :param l:\n        Length of output transformed vector.\n    :type l: int\n    :param m:\n        Length of the vector to be transformed.\n    :type m: int\n\n    :return:\n        Greatest power-of-two integer `n` satisfying `n <= m`.\n    :rtype: int\n    :return:\n        Initialization array to be used by :func:`idz_sfrm`.\n    :rtype: :class:`numpy.ndarray`\n    ')
    
    # Call to idz_sfrmi(...): (line 972)
    # Processing the call arguments (line 972)
    # Getting the type of 'l' (line 972)
    l_31349 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 972, 25), 'l', False)
    # Getting the type of 'm' (line 972)
    m_31350 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 972, 28), 'm', False)
    # Processing the call keyword arguments (line 972)
    kwargs_31351 = {}
    # Getting the type of '_id' (line 972)
    _id_31347 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 972, 11), '_id', False)
    # Obtaining the member 'idz_sfrmi' of a type (line 972)
    idz_sfrmi_31348 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 972, 11), _id_31347, 'idz_sfrmi')
    # Calling idz_sfrmi(args, kwargs) (line 972)
    idz_sfrmi_call_result_31352 = invoke(stypy.reporting.localization.Localization(__file__, 972, 11), idz_sfrmi_31348, *[l_31349, m_31350], **kwargs_31351)
    
    # Assigning a type to the variable 'stypy_return_type' (line 972)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 972, 4), 'stypy_return_type', idz_sfrmi_call_result_31352)
    
    # ################# End of 'idz_sfrmi(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'idz_sfrmi' in the type store
    # Getting the type of 'stypy_return_type' (line 954)
    stypy_return_type_31353 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 954, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_31353)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'idz_sfrmi'
    return stypy_return_type_31353

# Assigning a type to the variable 'idz_sfrmi' (line 954)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 954, 0), 'idz_sfrmi', idz_sfrmi)

@norecursion
def idzp_id(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'idzp_id'
    module_type_store = module_type_store.open_function_context('idzp_id', 979, 0, False)
    
    # Passed parameters checking function
    idzp_id.stypy_localization = localization
    idzp_id.stypy_type_of_self = None
    idzp_id.stypy_type_store = module_type_store
    idzp_id.stypy_function_name = 'idzp_id'
    idzp_id.stypy_param_names_list = ['eps', 'A']
    idzp_id.stypy_varargs_param_name = None
    idzp_id.stypy_kwargs_param_name = None
    idzp_id.stypy_call_defaults = defaults
    idzp_id.stypy_call_varargs = varargs
    idzp_id.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'idzp_id', ['eps', 'A'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'idzp_id', localization, ['eps', 'A'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'idzp_id(...)' code ##################

    str_31354 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 999, (-1)), 'str', '\n    Compute ID of a complex matrix to a specified relative precision.\n\n    :param eps:\n        Relative precision.\n    :type eps: float\n    :param A:\n        Matrix.\n    :type A: :class:`numpy.ndarray`\n\n    :return:\n        Rank of ID.\n    :rtype: int\n    :return:\n        Column index array.\n    :rtype: :class:`numpy.ndarray`\n    :return:\n        Interpolation coefficients.\n    :rtype: :class:`numpy.ndarray`\n    ')
    
    # Assigning a Call to a Name (line 1000):
    
    # Assigning a Call to a Name (line 1000):
    
    # Call to asfortranarray(...): (line 1000)
    # Processing the call arguments (line 1000)
    # Getting the type of 'A' (line 1000)
    A_31357 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1000, 26), 'A', False)
    # Processing the call keyword arguments (line 1000)
    kwargs_31358 = {}
    # Getting the type of 'np' (line 1000)
    np_31355 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1000, 8), 'np', False)
    # Obtaining the member 'asfortranarray' of a type (line 1000)
    asfortranarray_31356 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1000, 8), np_31355, 'asfortranarray')
    # Calling asfortranarray(args, kwargs) (line 1000)
    asfortranarray_call_result_31359 = invoke(stypy.reporting.localization.Localization(__file__, 1000, 8), asfortranarray_31356, *[A_31357], **kwargs_31358)
    
    # Assigning a type to the variable 'A' (line 1000)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1000, 4), 'A', asfortranarray_call_result_31359)
    
    # Assigning a Call to a Tuple (line 1001):
    
    # Assigning a Subscript to a Name (line 1001):
    
    # Obtaining the type of the subscript
    int_31360 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1001, 4), 'int')
    
    # Call to idzp_id(...): (line 1001)
    # Processing the call arguments (line 1001)
    # Getting the type of 'eps' (line 1001)
    eps_31363 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1001, 33), 'eps', False)
    # Getting the type of 'A' (line 1001)
    A_31364 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1001, 38), 'A', False)
    # Processing the call keyword arguments (line 1001)
    kwargs_31365 = {}
    # Getting the type of '_id' (line 1001)
    _id_31361 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1001, 21), '_id', False)
    # Obtaining the member 'idzp_id' of a type (line 1001)
    idzp_id_31362 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1001, 21), _id_31361, 'idzp_id')
    # Calling idzp_id(args, kwargs) (line 1001)
    idzp_id_call_result_31366 = invoke(stypy.reporting.localization.Localization(__file__, 1001, 21), idzp_id_31362, *[eps_31363, A_31364], **kwargs_31365)
    
    # Obtaining the member '__getitem__' of a type (line 1001)
    getitem___31367 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1001, 4), idzp_id_call_result_31366, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1001)
    subscript_call_result_31368 = invoke(stypy.reporting.localization.Localization(__file__, 1001, 4), getitem___31367, int_31360)
    
    # Assigning a type to the variable 'tuple_var_assignment_29703' (line 1001)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1001, 4), 'tuple_var_assignment_29703', subscript_call_result_31368)
    
    # Assigning a Subscript to a Name (line 1001):
    
    # Obtaining the type of the subscript
    int_31369 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1001, 4), 'int')
    
    # Call to idzp_id(...): (line 1001)
    # Processing the call arguments (line 1001)
    # Getting the type of 'eps' (line 1001)
    eps_31372 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1001, 33), 'eps', False)
    # Getting the type of 'A' (line 1001)
    A_31373 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1001, 38), 'A', False)
    # Processing the call keyword arguments (line 1001)
    kwargs_31374 = {}
    # Getting the type of '_id' (line 1001)
    _id_31370 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1001, 21), '_id', False)
    # Obtaining the member 'idzp_id' of a type (line 1001)
    idzp_id_31371 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1001, 21), _id_31370, 'idzp_id')
    # Calling idzp_id(args, kwargs) (line 1001)
    idzp_id_call_result_31375 = invoke(stypy.reporting.localization.Localization(__file__, 1001, 21), idzp_id_31371, *[eps_31372, A_31373], **kwargs_31374)
    
    # Obtaining the member '__getitem__' of a type (line 1001)
    getitem___31376 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1001, 4), idzp_id_call_result_31375, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1001)
    subscript_call_result_31377 = invoke(stypy.reporting.localization.Localization(__file__, 1001, 4), getitem___31376, int_31369)
    
    # Assigning a type to the variable 'tuple_var_assignment_29704' (line 1001)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1001, 4), 'tuple_var_assignment_29704', subscript_call_result_31377)
    
    # Assigning a Subscript to a Name (line 1001):
    
    # Obtaining the type of the subscript
    int_31378 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1001, 4), 'int')
    
    # Call to idzp_id(...): (line 1001)
    # Processing the call arguments (line 1001)
    # Getting the type of 'eps' (line 1001)
    eps_31381 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1001, 33), 'eps', False)
    # Getting the type of 'A' (line 1001)
    A_31382 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1001, 38), 'A', False)
    # Processing the call keyword arguments (line 1001)
    kwargs_31383 = {}
    # Getting the type of '_id' (line 1001)
    _id_31379 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1001, 21), '_id', False)
    # Obtaining the member 'idzp_id' of a type (line 1001)
    idzp_id_31380 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1001, 21), _id_31379, 'idzp_id')
    # Calling idzp_id(args, kwargs) (line 1001)
    idzp_id_call_result_31384 = invoke(stypy.reporting.localization.Localization(__file__, 1001, 21), idzp_id_31380, *[eps_31381, A_31382], **kwargs_31383)
    
    # Obtaining the member '__getitem__' of a type (line 1001)
    getitem___31385 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1001, 4), idzp_id_call_result_31384, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1001)
    subscript_call_result_31386 = invoke(stypy.reporting.localization.Localization(__file__, 1001, 4), getitem___31385, int_31378)
    
    # Assigning a type to the variable 'tuple_var_assignment_29705' (line 1001)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1001, 4), 'tuple_var_assignment_29705', subscript_call_result_31386)
    
    # Assigning a Name to a Name (line 1001):
    # Getting the type of 'tuple_var_assignment_29703' (line 1001)
    tuple_var_assignment_29703_31387 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1001, 4), 'tuple_var_assignment_29703')
    # Assigning a type to the variable 'k' (line 1001)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1001, 4), 'k', tuple_var_assignment_29703_31387)
    
    # Assigning a Name to a Name (line 1001):
    # Getting the type of 'tuple_var_assignment_29704' (line 1001)
    tuple_var_assignment_29704_31388 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1001, 4), 'tuple_var_assignment_29704')
    # Assigning a type to the variable 'idx' (line 1001)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1001, 7), 'idx', tuple_var_assignment_29704_31388)
    
    # Assigning a Name to a Name (line 1001):
    # Getting the type of 'tuple_var_assignment_29705' (line 1001)
    tuple_var_assignment_29705_31389 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1001, 4), 'tuple_var_assignment_29705')
    # Assigning a type to the variable 'rnorms' (line 1001)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1001, 12), 'rnorms', tuple_var_assignment_29705_31389)
    
    # Assigning a Subscript to a Name (line 1002):
    
    # Assigning a Subscript to a Name (line 1002):
    
    # Obtaining the type of the subscript
    int_31390 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1002, 16), 'int')
    # Getting the type of 'A' (line 1002)
    A_31391 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1002, 8), 'A')
    # Obtaining the member 'shape' of a type (line 1002)
    shape_31392 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1002, 8), A_31391, 'shape')
    # Obtaining the member '__getitem__' of a type (line 1002)
    getitem___31393 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1002, 8), shape_31392, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1002)
    subscript_call_result_31394 = invoke(stypy.reporting.localization.Localization(__file__, 1002, 8), getitem___31393, int_31390)
    
    # Assigning a type to the variable 'n' (line 1002)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1002, 4), 'n', subscript_call_result_31394)
    
    # Assigning a Call to a Name (line 1003):
    
    # Assigning a Call to a Name (line 1003):
    
    # Call to reshape(...): (line 1003)
    # Processing the call arguments (line 1003)
    
    # Obtaining an instance of the builtin type 'tuple' (line 1003)
    tuple_31409 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1003, 42), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 1003)
    # Adding element type (line 1003)
    # Getting the type of 'k' (line 1003)
    k_31410 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1003, 42), 'k', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1003, 42), tuple_31409, k_31410)
    # Adding element type (line 1003)
    # Getting the type of 'n' (line 1003)
    n_31411 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1003, 45), 'n', False)
    # Getting the type of 'k' (line 1003)
    k_31412 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1003, 47), 'k', False)
    # Applying the binary operator '-' (line 1003)
    result_sub_31413 = python_operator(stypy.reporting.localization.Localization(__file__, 1003, 45), '-', n_31411, k_31412)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1003, 42), tuple_31409, result_sub_31413)
    
    # Processing the call keyword arguments (line 1003)
    str_31414 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1003, 57), 'str', 'F')
    keyword_31415 = str_31414
    kwargs_31416 = {'order': keyword_31415}
    
    # Obtaining the type of the subscript
    # Getting the type of 'k' (line 1003)
    k_31395 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1003, 24), 'k', False)
    # Getting the type of 'n' (line 1003)
    n_31396 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1003, 27), 'n', False)
    # Getting the type of 'k' (line 1003)
    k_31397 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1003, 29), 'k', False)
    # Applying the binary operator '-' (line 1003)
    result_sub_31398 = python_operator(stypy.reporting.localization.Localization(__file__, 1003, 27), '-', n_31396, k_31397)
    
    # Applying the binary operator '*' (line 1003)
    result_mul_31399 = python_operator(stypy.reporting.localization.Localization(__file__, 1003, 24), '*', k_31395, result_sub_31398)
    
    slice_31400 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 1003, 11), None, result_mul_31399, None)
    
    # Call to ravel(...): (line 1003)
    # Processing the call keyword arguments (line 1003)
    kwargs_31404 = {}
    # Getting the type of 'A' (line 1003)
    A_31401 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1003, 11), 'A', False)
    # Obtaining the member 'T' of a type (line 1003)
    T_31402 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1003, 11), A_31401, 'T')
    # Obtaining the member 'ravel' of a type (line 1003)
    ravel_31403 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1003, 11), T_31402, 'ravel')
    # Calling ravel(args, kwargs) (line 1003)
    ravel_call_result_31405 = invoke(stypy.reporting.localization.Localization(__file__, 1003, 11), ravel_31403, *[], **kwargs_31404)
    
    # Obtaining the member '__getitem__' of a type (line 1003)
    getitem___31406 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1003, 11), ravel_call_result_31405, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1003)
    subscript_call_result_31407 = invoke(stypy.reporting.localization.Localization(__file__, 1003, 11), getitem___31406, slice_31400)
    
    # Obtaining the member 'reshape' of a type (line 1003)
    reshape_31408 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1003, 11), subscript_call_result_31407, 'reshape')
    # Calling reshape(args, kwargs) (line 1003)
    reshape_call_result_31417 = invoke(stypy.reporting.localization.Localization(__file__, 1003, 11), reshape_31408, *[tuple_31409], **kwargs_31416)
    
    # Assigning a type to the variable 'proj' (line 1003)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1003, 4), 'proj', reshape_call_result_31417)
    
    # Obtaining an instance of the builtin type 'tuple' (line 1004)
    tuple_31418 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1004, 11), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 1004)
    # Adding element type (line 1004)
    # Getting the type of 'k' (line 1004)
    k_31419 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1004, 11), 'k')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1004, 11), tuple_31418, k_31419)
    # Adding element type (line 1004)
    # Getting the type of 'idx' (line 1004)
    idx_31420 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1004, 14), 'idx')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1004, 11), tuple_31418, idx_31420)
    # Adding element type (line 1004)
    # Getting the type of 'proj' (line 1004)
    proj_31421 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1004, 19), 'proj')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1004, 11), tuple_31418, proj_31421)
    
    # Assigning a type to the variable 'stypy_return_type' (line 1004)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1004, 4), 'stypy_return_type', tuple_31418)
    
    # ################# End of 'idzp_id(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'idzp_id' in the type store
    # Getting the type of 'stypy_return_type' (line 979)
    stypy_return_type_31422 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 979, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_31422)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'idzp_id'
    return stypy_return_type_31422

# Assigning a type to the variable 'idzp_id' (line 979)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 979, 0), 'idzp_id', idzp_id)

@norecursion
def idzr_id(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'idzr_id'
    module_type_store = module_type_store.open_function_context('idzr_id', 1007, 0, False)
    
    # Passed parameters checking function
    idzr_id.stypy_localization = localization
    idzr_id.stypy_type_of_self = None
    idzr_id.stypy_type_store = module_type_store
    idzr_id.stypy_function_name = 'idzr_id'
    idzr_id.stypy_param_names_list = ['A', 'k']
    idzr_id.stypy_varargs_param_name = None
    idzr_id.stypy_kwargs_param_name = None
    idzr_id.stypy_call_defaults = defaults
    idzr_id.stypy_call_varargs = varargs
    idzr_id.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'idzr_id', ['A', 'k'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'idzr_id', localization, ['A', 'k'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'idzr_id(...)' code ##################

    str_31423 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1024, (-1)), 'str', '\n    Compute ID of a complex matrix to a specified rank.\n\n    :param A:\n        Matrix.\n    :type A: :class:`numpy.ndarray`\n    :param k:\n        Rank of ID.\n    :type k: int\n\n    :return:\n        Column index array.\n    :rtype: :class:`numpy.ndarray`\n    :return:\n        Interpolation coefficients.\n    :rtype: :class:`numpy.ndarray`\n    ')
    
    # Assigning a Call to a Name (line 1025):
    
    # Assigning a Call to a Name (line 1025):
    
    # Call to asfortranarray(...): (line 1025)
    # Processing the call arguments (line 1025)
    # Getting the type of 'A' (line 1025)
    A_31426 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1025, 26), 'A', False)
    # Processing the call keyword arguments (line 1025)
    kwargs_31427 = {}
    # Getting the type of 'np' (line 1025)
    np_31424 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1025, 8), 'np', False)
    # Obtaining the member 'asfortranarray' of a type (line 1025)
    asfortranarray_31425 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1025, 8), np_31424, 'asfortranarray')
    # Calling asfortranarray(args, kwargs) (line 1025)
    asfortranarray_call_result_31428 = invoke(stypy.reporting.localization.Localization(__file__, 1025, 8), asfortranarray_31425, *[A_31426], **kwargs_31427)
    
    # Assigning a type to the variable 'A' (line 1025)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1025, 4), 'A', asfortranarray_call_result_31428)
    
    # Assigning a Call to a Tuple (line 1026):
    
    # Assigning a Subscript to a Name (line 1026):
    
    # Obtaining the type of the subscript
    int_31429 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1026, 4), 'int')
    
    # Call to idzr_id(...): (line 1026)
    # Processing the call arguments (line 1026)
    # Getting the type of 'A' (line 1026)
    A_31432 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1026, 30), 'A', False)
    # Getting the type of 'k' (line 1026)
    k_31433 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1026, 33), 'k', False)
    # Processing the call keyword arguments (line 1026)
    kwargs_31434 = {}
    # Getting the type of '_id' (line 1026)
    _id_31430 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1026, 18), '_id', False)
    # Obtaining the member 'idzr_id' of a type (line 1026)
    idzr_id_31431 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1026, 18), _id_31430, 'idzr_id')
    # Calling idzr_id(args, kwargs) (line 1026)
    idzr_id_call_result_31435 = invoke(stypy.reporting.localization.Localization(__file__, 1026, 18), idzr_id_31431, *[A_31432, k_31433], **kwargs_31434)
    
    # Obtaining the member '__getitem__' of a type (line 1026)
    getitem___31436 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1026, 4), idzr_id_call_result_31435, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1026)
    subscript_call_result_31437 = invoke(stypy.reporting.localization.Localization(__file__, 1026, 4), getitem___31436, int_31429)
    
    # Assigning a type to the variable 'tuple_var_assignment_29706' (line 1026)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1026, 4), 'tuple_var_assignment_29706', subscript_call_result_31437)
    
    # Assigning a Subscript to a Name (line 1026):
    
    # Obtaining the type of the subscript
    int_31438 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1026, 4), 'int')
    
    # Call to idzr_id(...): (line 1026)
    # Processing the call arguments (line 1026)
    # Getting the type of 'A' (line 1026)
    A_31441 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1026, 30), 'A', False)
    # Getting the type of 'k' (line 1026)
    k_31442 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1026, 33), 'k', False)
    # Processing the call keyword arguments (line 1026)
    kwargs_31443 = {}
    # Getting the type of '_id' (line 1026)
    _id_31439 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1026, 18), '_id', False)
    # Obtaining the member 'idzr_id' of a type (line 1026)
    idzr_id_31440 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1026, 18), _id_31439, 'idzr_id')
    # Calling idzr_id(args, kwargs) (line 1026)
    idzr_id_call_result_31444 = invoke(stypy.reporting.localization.Localization(__file__, 1026, 18), idzr_id_31440, *[A_31441, k_31442], **kwargs_31443)
    
    # Obtaining the member '__getitem__' of a type (line 1026)
    getitem___31445 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1026, 4), idzr_id_call_result_31444, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1026)
    subscript_call_result_31446 = invoke(stypy.reporting.localization.Localization(__file__, 1026, 4), getitem___31445, int_31438)
    
    # Assigning a type to the variable 'tuple_var_assignment_29707' (line 1026)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1026, 4), 'tuple_var_assignment_29707', subscript_call_result_31446)
    
    # Assigning a Name to a Name (line 1026):
    # Getting the type of 'tuple_var_assignment_29706' (line 1026)
    tuple_var_assignment_29706_31447 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1026, 4), 'tuple_var_assignment_29706')
    # Assigning a type to the variable 'idx' (line 1026)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1026, 4), 'idx', tuple_var_assignment_29706_31447)
    
    # Assigning a Name to a Name (line 1026):
    # Getting the type of 'tuple_var_assignment_29707' (line 1026)
    tuple_var_assignment_29707_31448 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1026, 4), 'tuple_var_assignment_29707')
    # Assigning a type to the variable 'rnorms' (line 1026)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1026, 9), 'rnorms', tuple_var_assignment_29707_31448)
    
    # Assigning a Subscript to a Name (line 1027):
    
    # Assigning a Subscript to a Name (line 1027):
    
    # Obtaining the type of the subscript
    int_31449 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1027, 16), 'int')
    # Getting the type of 'A' (line 1027)
    A_31450 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1027, 8), 'A')
    # Obtaining the member 'shape' of a type (line 1027)
    shape_31451 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1027, 8), A_31450, 'shape')
    # Obtaining the member '__getitem__' of a type (line 1027)
    getitem___31452 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1027, 8), shape_31451, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1027)
    subscript_call_result_31453 = invoke(stypy.reporting.localization.Localization(__file__, 1027, 8), getitem___31452, int_31449)
    
    # Assigning a type to the variable 'n' (line 1027)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1027, 4), 'n', subscript_call_result_31453)
    
    # Assigning a Call to a Name (line 1028):
    
    # Assigning a Call to a Name (line 1028):
    
    # Call to reshape(...): (line 1028)
    # Processing the call arguments (line 1028)
    
    # Obtaining an instance of the builtin type 'tuple' (line 1028)
    tuple_31468 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1028, 42), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 1028)
    # Adding element type (line 1028)
    # Getting the type of 'k' (line 1028)
    k_31469 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1028, 42), 'k', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1028, 42), tuple_31468, k_31469)
    # Adding element type (line 1028)
    # Getting the type of 'n' (line 1028)
    n_31470 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1028, 45), 'n', False)
    # Getting the type of 'k' (line 1028)
    k_31471 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1028, 47), 'k', False)
    # Applying the binary operator '-' (line 1028)
    result_sub_31472 = python_operator(stypy.reporting.localization.Localization(__file__, 1028, 45), '-', n_31470, k_31471)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1028, 42), tuple_31468, result_sub_31472)
    
    # Processing the call keyword arguments (line 1028)
    str_31473 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1028, 57), 'str', 'F')
    keyword_31474 = str_31473
    kwargs_31475 = {'order': keyword_31474}
    
    # Obtaining the type of the subscript
    # Getting the type of 'k' (line 1028)
    k_31454 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1028, 24), 'k', False)
    # Getting the type of 'n' (line 1028)
    n_31455 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1028, 27), 'n', False)
    # Getting the type of 'k' (line 1028)
    k_31456 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1028, 29), 'k', False)
    # Applying the binary operator '-' (line 1028)
    result_sub_31457 = python_operator(stypy.reporting.localization.Localization(__file__, 1028, 27), '-', n_31455, k_31456)
    
    # Applying the binary operator '*' (line 1028)
    result_mul_31458 = python_operator(stypy.reporting.localization.Localization(__file__, 1028, 24), '*', k_31454, result_sub_31457)
    
    slice_31459 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 1028, 11), None, result_mul_31458, None)
    
    # Call to ravel(...): (line 1028)
    # Processing the call keyword arguments (line 1028)
    kwargs_31463 = {}
    # Getting the type of 'A' (line 1028)
    A_31460 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1028, 11), 'A', False)
    # Obtaining the member 'T' of a type (line 1028)
    T_31461 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1028, 11), A_31460, 'T')
    # Obtaining the member 'ravel' of a type (line 1028)
    ravel_31462 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1028, 11), T_31461, 'ravel')
    # Calling ravel(args, kwargs) (line 1028)
    ravel_call_result_31464 = invoke(stypy.reporting.localization.Localization(__file__, 1028, 11), ravel_31462, *[], **kwargs_31463)
    
    # Obtaining the member '__getitem__' of a type (line 1028)
    getitem___31465 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1028, 11), ravel_call_result_31464, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1028)
    subscript_call_result_31466 = invoke(stypy.reporting.localization.Localization(__file__, 1028, 11), getitem___31465, slice_31459)
    
    # Obtaining the member 'reshape' of a type (line 1028)
    reshape_31467 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1028, 11), subscript_call_result_31466, 'reshape')
    # Calling reshape(args, kwargs) (line 1028)
    reshape_call_result_31476 = invoke(stypy.reporting.localization.Localization(__file__, 1028, 11), reshape_31467, *[tuple_31468], **kwargs_31475)
    
    # Assigning a type to the variable 'proj' (line 1028)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1028, 4), 'proj', reshape_call_result_31476)
    
    # Obtaining an instance of the builtin type 'tuple' (line 1029)
    tuple_31477 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1029, 11), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 1029)
    # Adding element type (line 1029)
    # Getting the type of 'idx' (line 1029)
    idx_31478 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1029, 11), 'idx')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1029, 11), tuple_31477, idx_31478)
    # Adding element type (line 1029)
    # Getting the type of 'proj' (line 1029)
    proj_31479 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1029, 16), 'proj')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1029, 11), tuple_31477, proj_31479)
    
    # Assigning a type to the variable 'stypy_return_type' (line 1029)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1029, 4), 'stypy_return_type', tuple_31477)
    
    # ################# End of 'idzr_id(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'idzr_id' in the type store
    # Getting the type of 'stypy_return_type' (line 1007)
    stypy_return_type_31480 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1007, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_31480)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'idzr_id'
    return stypy_return_type_31480

# Assigning a type to the variable 'idzr_id' (line 1007)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1007, 0), 'idzr_id', idzr_id)

@norecursion
def idz_reconid(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'idz_reconid'
    module_type_store = module_type_store.open_function_context('idz_reconid', 1032, 0, False)
    
    # Passed parameters checking function
    idz_reconid.stypy_localization = localization
    idz_reconid.stypy_type_of_self = None
    idz_reconid.stypy_type_store = module_type_store
    idz_reconid.stypy_function_name = 'idz_reconid'
    idz_reconid.stypy_param_names_list = ['B', 'idx', 'proj']
    idz_reconid.stypy_varargs_param_name = None
    idz_reconid.stypy_kwargs_param_name = None
    idz_reconid.stypy_call_defaults = defaults
    idz_reconid.stypy_call_varargs = varargs
    idz_reconid.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'idz_reconid', ['B', 'idx', 'proj'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'idz_reconid', localization, ['B', 'idx', 'proj'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'idz_reconid(...)' code ##################

    str_31481 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1049, (-1)), 'str', '\n    Reconstruct matrix from complex ID.\n\n    :param B:\n        Skeleton matrix.\n    :type B: :class:`numpy.ndarray`\n    :param idx:\n        Column index array.\n    :type idx: :class:`numpy.ndarray`\n    :param proj:\n        Interpolation coefficients.\n    :type proj: :class:`numpy.ndarray`\n\n    :return:\n        Reconstructed matrix.\n    :rtype: :class:`numpy.ndarray`\n    ')
    
    # Assigning a Call to a Name (line 1050):
    
    # Assigning a Call to a Name (line 1050):
    
    # Call to asfortranarray(...): (line 1050)
    # Processing the call arguments (line 1050)
    # Getting the type of 'B' (line 1050)
    B_31484 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1050, 26), 'B', False)
    # Processing the call keyword arguments (line 1050)
    kwargs_31485 = {}
    # Getting the type of 'np' (line 1050)
    np_31482 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1050, 8), 'np', False)
    # Obtaining the member 'asfortranarray' of a type (line 1050)
    asfortranarray_31483 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1050, 8), np_31482, 'asfortranarray')
    # Calling asfortranarray(args, kwargs) (line 1050)
    asfortranarray_call_result_31486 = invoke(stypy.reporting.localization.Localization(__file__, 1050, 8), asfortranarray_31483, *[B_31484], **kwargs_31485)
    
    # Assigning a type to the variable 'B' (line 1050)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1050, 4), 'B', asfortranarray_call_result_31486)
    
    
    # Getting the type of 'proj' (line 1051)
    proj_31487 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1051, 7), 'proj')
    # Obtaining the member 'size' of a type (line 1051)
    size_31488 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1051, 7), proj_31487, 'size')
    int_31489 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1051, 19), 'int')
    # Applying the binary operator '>' (line 1051)
    result_gt_31490 = python_operator(stypy.reporting.localization.Localization(__file__, 1051, 7), '>', size_31488, int_31489)
    
    # Testing the type of an if condition (line 1051)
    if_condition_31491 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1051, 4), result_gt_31490)
    # Assigning a type to the variable 'if_condition_31491' (line 1051)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1051, 4), 'if_condition_31491', if_condition_31491)
    # SSA begins for if statement (line 1051)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to idz_reconid(...): (line 1052)
    # Processing the call arguments (line 1052)
    # Getting the type of 'B' (line 1052)
    B_31494 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1052, 31), 'B', False)
    # Getting the type of 'idx' (line 1052)
    idx_31495 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1052, 34), 'idx', False)
    # Getting the type of 'proj' (line 1052)
    proj_31496 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1052, 39), 'proj', False)
    # Processing the call keyword arguments (line 1052)
    kwargs_31497 = {}
    # Getting the type of '_id' (line 1052)
    _id_31492 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1052, 15), '_id', False)
    # Obtaining the member 'idz_reconid' of a type (line 1052)
    idz_reconid_31493 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1052, 15), _id_31492, 'idz_reconid')
    # Calling idz_reconid(args, kwargs) (line 1052)
    idz_reconid_call_result_31498 = invoke(stypy.reporting.localization.Localization(__file__, 1052, 15), idz_reconid_31493, *[B_31494, idx_31495, proj_31496], **kwargs_31497)
    
    # Assigning a type to the variable 'stypy_return_type' (line 1052)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1052, 8), 'stypy_return_type', idz_reconid_call_result_31498)
    # SSA branch for the else part of an if statement (line 1051)
    module_type_store.open_ssa_branch('else')
    
    # Obtaining the type of the subscript
    slice_31499 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 1054, 15), None, None, None)
    
    # Call to argsort(...): (line 1054)
    # Processing the call arguments (line 1054)
    # Getting the type of 'idx' (line 1054)
    idx_31502 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1054, 31), 'idx', False)
    # Processing the call keyword arguments (line 1054)
    kwargs_31503 = {}
    # Getting the type of 'np' (line 1054)
    np_31500 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1054, 20), 'np', False)
    # Obtaining the member 'argsort' of a type (line 1054)
    argsort_31501 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1054, 20), np_31500, 'argsort')
    # Calling argsort(args, kwargs) (line 1054)
    argsort_call_result_31504 = invoke(stypy.reporting.localization.Localization(__file__, 1054, 20), argsort_31501, *[idx_31502], **kwargs_31503)
    
    # Getting the type of 'B' (line 1054)
    B_31505 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1054, 15), 'B')
    # Obtaining the member '__getitem__' of a type (line 1054)
    getitem___31506 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1054, 15), B_31505, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1054)
    subscript_call_result_31507 = invoke(stypy.reporting.localization.Localization(__file__, 1054, 15), getitem___31506, (slice_31499, argsort_call_result_31504))
    
    # Assigning a type to the variable 'stypy_return_type' (line 1054)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1054, 8), 'stypy_return_type', subscript_call_result_31507)
    # SSA join for if statement (line 1051)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'idz_reconid(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'idz_reconid' in the type store
    # Getting the type of 'stypy_return_type' (line 1032)
    stypy_return_type_31508 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1032, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_31508)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'idz_reconid'
    return stypy_return_type_31508

# Assigning a type to the variable 'idz_reconid' (line 1032)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1032, 0), 'idz_reconid', idz_reconid)

@norecursion
def idz_reconint(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'idz_reconint'
    module_type_store = module_type_store.open_function_context('idz_reconint', 1057, 0, False)
    
    # Passed parameters checking function
    idz_reconint.stypy_localization = localization
    idz_reconint.stypy_type_of_self = None
    idz_reconint.stypy_type_store = module_type_store
    idz_reconint.stypy_function_name = 'idz_reconint'
    idz_reconint.stypy_param_names_list = ['idx', 'proj']
    idz_reconint.stypy_varargs_param_name = None
    idz_reconint.stypy_kwargs_param_name = None
    idz_reconint.stypy_call_defaults = defaults
    idz_reconint.stypy_call_varargs = varargs
    idz_reconint.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'idz_reconint', ['idx', 'proj'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'idz_reconint', localization, ['idx', 'proj'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'idz_reconint(...)' code ##################

    str_31509 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1071, (-1)), 'str', '\n    Reconstruct interpolation matrix from complex ID.\n\n    :param idx:\n        Column index array.\n    :type idx: :class:`numpy.ndarray`\n    :param proj:\n        Interpolation coefficients.\n    :type proj: :class:`numpy.ndarray`\n\n    :return:\n        Interpolation matrix.\n    :rtype: :class:`numpy.ndarray`\n    ')
    
    # Call to idz_reconint(...): (line 1072)
    # Processing the call arguments (line 1072)
    # Getting the type of 'idx' (line 1072)
    idx_31512 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1072, 28), 'idx', False)
    # Getting the type of 'proj' (line 1072)
    proj_31513 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1072, 33), 'proj', False)
    # Processing the call keyword arguments (line 1072)
    kwargs_31514 = {}
    # Getting the type of '_id' (line 1072)
    _id_31510 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1072, 11), '_id', False)
    # Obtaining the member 'idz_reconint' of a type (line 1072)
    idz_reconint_31511 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1072, 11), _id_31510, 'idz_reconint')
    # Calling idz_reconint(args, kwargs) (line 1072)
    idz_reconint_call_result_31515 = invoke(stypy.reporting.localization.Localization(__file__, 1072, 11), idz_reconint_31511, *[idx_31512, proj_31513], **kwargs_31514)
    
    # Assigning a type to the variable 'stypy_return_type' (line 1072)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1072, 4), 'stypy_return_type', idz_reconint_call_result_31515)
    
    # ################# End of 'idz_reconint(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'idz_reconint' in the type store
    # Getting the type of 'stypy_return_type' (line 1057)
    stypy_return_type_31516 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1057, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_31516)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'idz_reconint'
    return stypy_return_type_31516

# Assigning a type to the variable 'idz_reconint' (line 1057)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1057, 0), 'idz_reconint', idz_reconint)

@norecursion
def idz_copycols(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'idz_copycols'
    module_type_store = module_type_store.open_function_context('idz_copycols', 1075, 0, False)
    
    # Passed parameters checking function
    idz_copycols.stypy_localization = localization
    idz_copycols.stypy_type_of_self = None
    idz_copycols.stypy_type_store = module_type_store
    idz_copycols.stypy_function_name = 'idz_copycols'
    idz_copycols.stypy_param_names_list = ['A', 'k', 'idx']
    idz_copycols.stypy_varargs_param_name = None
    idz_copycols.stypy_kwargs_param_name = None
    idz_copycols.stypy_call_defaults = defaults
    idz_copycols.stypy_call_varargs = varargs
    idz_copycols.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'idz_copycols', ['A', 'k', 'idx'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'idz_copycols', localization, ['A', 'k', 'idx'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'idz_copycols(...)' code ##################

    str_31517 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1092, (-1)), 'str', '\n    Reconstruct skeleton matrix from complex ID.\n\n    :param A:\n        Original matrix.\n    :type A: :class:`numpy.ndarray`\n    :param k:\n        Rank of ID.\n    :type k: int\n    :param idx:\n        Column index array.\n    :type idx: :class:`numpy.ndarray`\n\n    :return:\n        Skeleton matrix.\n    :rtype: :class:`numpy.ndarray`\n    ')
    
    # Assigning a Call to a Name (line 1093):
    
    # Assigning a Call to a Name (line 1093):
    
    # Call to asfortranarray(...): (line 1093)
    # Processing the call arguments (line 1093)
    # Getting the type of 'A' (line 1093)
    A_31520 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1093, 26), 'A', False)
    # Processing the call keyword arguments (line 1093)
    kwargs_31521 = {}
    # Getting the type of 'np' (line 1093)
    np_31518 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1093, 8), 'np', False)
    # Obtaining the member 'asfortranarray' of a type (line 1093)
    asfortranarray_31519 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1093, 8), np_31518, 'asfortranarray')
    # Calling asfortranarray(args, kwargs) (line 1093)
    asfortranarray_call_result_31522 = invoke(stypy.reporting.localization.Localization(__file__, 1093, 8), asfortranarray_31519, *[A_31520], **kwargs_31521)
    
    # Assigning a type to the variable 'A' (line 1093)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1093, 4), 'A', asfortranarray_call_result_31522)
    
    # Call to idz_copycols(...): (line 1094)
    # Processing the call arguments (line 1094)
    # Getting the type of 'A' (line 1094)
    A_31525 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1094, 28), 'A', False)
    # Getting the type of 'k' (line 1094)
    k_31526 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1094, 31), 'k', False)
    # Getting the type of 'idx' (line 1094)
    idx_31527 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1094, 34), 'idx', False)
    # Processing the call keyword arguments (line 1094)
    kwargs_31528 = {}
    # Getting the type of '_id' (line 1094)
    _id_31523 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1094, 11), '_id', False)
    # Obtaining the member 'idz_copycols' of a type (line 1094)
    idz_copycols_31524 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1094, 11), _id_31523, 'idz_copycols')
    # Calling idz_copycols(args, kwargs) (line 1094)
    idz_copycols_call_result_31529 = invoke(stypy.reporting.localization.Localization(__file__, 1094, 11), idz_copycols_31524, *[A_31525, k_31526, idx_31527], **kwargs_31528)
    
    # Assigning a type to the variable 'stypy_return_type' (line 1094)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1094, 4), 'stypy_return_type', idz_copycols_call_result_31529)
    
    # ################# End of 'idz_copycols(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'idz_copycols' in the type store
    # Getting the type of 'stypy_return_type' (line 1075)
    stypy_return_type_31530 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1075, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_31530)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'idz_copycols'
    return stypy_return_type_31530

# Assigning a type to the variable 'idz_copycols' (line 1075)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1075, 0), 'idz_copycols', idz_copycols)

@norecursion
def idz_id2svd(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'idz_id2svd'
    module_type_store = module_type_store.open_function_context('idz_id2svd', 1101, 0, False)
    
    # Passed parameters checking function
    idz_id2svd.stypy_localization = localization
    idz_id2svd.stypy_type_of_self = None
    idz_id2svd.stypy_type_store = module_type_store
    idz_id2svd.stypy_function_name = 'idz_id2svd'
    idz_id2svd.stypy_param_names_list = ['B', 'idx', 'proj']
    idz_id2svd.stypy_varargs_param_name = None
    idz_id2svd.stypy_kwargs_param_name = None
    idz_id2svd.stypy_call_defaults = defaults
    idz_id2svd.stypy_call_varargs = varargs
    idz_id2svd.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'idz_id2svd', ['B', 'idx', 'proj'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'idz_id2svd', localization, ['B', 'idx', 'proj'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'idz_id2svd(...)' code ##################

    str_31531 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1124, (-1)), 'str', '\n    Convert complex ID to SVD.\n\n    :param B:\n        Skeleton matrix.\n    :type B: :class:`numpy.ndarray`\n    :param idx:\n        Column index array.\n    :type idx: :class:`numpy.ndarray`\n    :param proj:\n        Interpolation coefficients.\n    :type proj: :class:`numpy.ndarray`\n\n    :return:\n        Left singular vectors.\n    :rtype: :class:`numpy.ndarray`\n    :return:\n        Right singular vectors.\n    :rtype: :class:`numpy.ndarray`\n    :return:\n        Singular values.\n    :rtype: :class:`numpy.ndarray`\n    ')
    
    # Assigning a Call to a Name (line 1125):
    
    # Assigning a Call to a Name (line 1125):
    
    # Call to asfortranarray(...): (line 1125)
    # Processing the call arguments (line 1125)
    # Getting the type of 'B' (line 1125)
    B_31534 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1125, 26), 'B', False)
    # Processing the call keyword arguments (line 1125)
    kwargs_31535 = {}
    # Getting the type of 'np' (line 1125)
    np_31532 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1125, 8), 'np', False)
    # Obtaining the member 'asfortranarray' of a type (line 1125)
    asfortranarray_31533 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1125, 8), np_31532, 'asfortranarray')
    # Calling asfortranarray(args, kwargs) (line 1125)
    asfortranarray_call_result_31536 = invoke(stypy.reporting.localization.Localization(__file__, 1125, 8), asfortranarray_31533, *[B_31534], **kwargs_31535)
    
    # Assigning a type to the variable 'B' (line 1125)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1125, 4), 'B', asfortranarray_call_result_31536)
    
    # Assigning a Call to a Tuple (line 1126):
    
    # Assigning a Subscript to a Name (line 1126):
    
    # Obtaining the type of the subscript
    int_31537 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1126, 4), 'int')
    
    # Call to idz_id2svd(...): (line 1126)
    # Processing the call arguments (line 1126)
    # Getting the type of 'B' (line 1126)
    B_31540 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1126, 34), 'B', False)
    # Getting the type of 'idx' (line 1126)
    idx_31541 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1126, 37), 'idx', False)
    # Getting the type of 'proj' (line 1126)
    proj_31542 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1126, 42), 'proj', False)
    # Processing the call keyword arguments (line 1126)
    kwargs_31543 = {}
    # Getting the type of '_id' (line 1126)
    _id_31538 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1126, 19), '_id', False)
    # Obtaining the member 'idz_id2svd' of a type (line 1126)
    idz_id2svd_31539 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1126, 19), _id_31538, 'idz_id2svd')
    # Calling idz_id2svd(args, kwargs) (line 1126)
    idz_id2svd_call_result_31544 = invoke(stypy.reporting.localization.Localization(__file__, 1126, 19), idz_id2svd_31539, *[B_31540, idx_31541, proj_31542], **kwargs_31543)
    
    # Obtaining the member '__getitem__' of a type (line 1126)
    getitem___31545 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1126, 4), idz_id2svd_call_result_31544, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1126)
    subscript_call_result_31546 = invoke(stypy.reporting.localization.Localization(__file__, 1126, 4), getitem___31545, int_31537)
    
    # Assigning a type to the variable 'tuple_var_assignment_29708' (line 1126)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1126, 4), 'tuple_var_assignment_29708', subscript_call_result_31546)
    
    # Assigning a Subscript to a Name (line 1126):
    
    # Obtaining the type of the subscript
    int_31547 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1126, 4), 'int')
    
    # Call to idz_id2svd(...): (line 1126)
    # Processing the call arguments (line 1126)
    # Getting the type of 'B' (line 1126)
    B_31550 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1126, 34), 'B', False)
    # Getting the type of 'idx' (line 1126)
    idx_31551 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1126, 37), 'idx', False)
    # Getting the type of 'proj' (line 1126)
    proj_31552 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1126, 42), 'proj', False)
    # Processing the call keyword arguments (line 1126)
    kwargs_31553 = {}
    # Getting the type of '_id' (line 1126)
    _id_31548 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1126, 19), '_id', False)
    # Obtaining the member 'idz_id2svd' of a type (line 1126)
    idz_id2svd_31549 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1126, 19), _id_31548, 'idz_id2svd')
    # Calling idz_id2svd(args, kwargs) (line 1126)
    idz_id2svd_call_result_31554 = invoke(stypy.reporting.localization.Localization(__file__, 1126, 19), idz_id2svd_31549, *[B_31550, idx_31551, proj_31552], **kwargs_31553)
    
    # Obtaining the member '__getitem__' of a type (line 1126)
    getitem___31555 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1126, 4), idz_id2svd_call_result_31554, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1126)
    subscript_call_result_31556 = invoke(stypy.reporting.localization.Localization(__file__, 1126, 4), getitem___31555, int_31547)
    
    # Assigning a type to the variable 'tuple_var_assignment_29709' (line 1126)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1126, 4), 'tuple_var_assignment_29709', subscript_call_result_31556)
    
    # Assigning a Subscript to a Name (line 1126):
    
    # Obtaining the type of the subscript
    int_31557 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1126, 4), 'int')
    
    # Call to idz_id2svd(...): (line 1126)
    # Processing the call arguments (line 1126)
    # Getting the type of 'B' (line 1126)
    B_31560 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1126, 34), 'B', False)
    # Getting the type of 'idx' (line 1126)
    idx_31561 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1126, 37), 'idx', False)
    # Getting the type of 'proj' (line 1126)
    proj_31562 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1126, 42), 'proj', False)
    # Processing the call keyword arguments (line 1126)
    kwargs_31563 = {}
    # Getting the type of '_id' (line 1126)
    _id_31558 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1126, 19), '_id', False)
    # Obtaining the member 'idz_id2svd' of a type (line 1126)
    idz_id2svd_31559 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1126, 19), _id_31558, 'idz_id2svd')
    # Calling idz_id2svd(args, kwargs) (line 1126)
    idz_id2svd_call_result_31564 = invoke(stypy.reporting.localization.Localization(__file__, 1126, 19), idz_id2svd_31559, *[B_31560, idx_31561, proj_31562], **kwargs_31563)
    
    # Obtaining the member '__getitem__' of a type (line 1126)
    getitem___31565 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1126, 4), idz_id2svd_call_result_31564, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1126)
    subscript_call_result_31566 = invoke(stypy.reporting.localization.Localization(__file__, 1126, 4), getitem___31565, int_31557)
    
    # Assigning a type to the variable 'tuple_var_assignment_29710' (line 1126)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1126, 4), 'tuple_var_assignment_29710', subscript_call_result_31566)
    
    # Assigning a Subscript to a Name (line 1126):
    
    # Obtaining the type of the subscript
    int_31567 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1126, 4), 'int')
    
    # Call to idz_id2svd(...): (line 1126)
    # Processing the call arguments (line 1126)
    # Getting the type of 'B' (line 1126)
    B_31570 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1126, 34), 'B', False)
    # Getting the type of 'idx' (line 1126)
    idx_31571 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1126, 37), 'idx', False)
    # Getting the type of 'proj' (line 1126)
    proj_31572 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1126, 42), 'proj', False)
    # Processing the call keyword arguments (line 1126)
    kwargs_31573 = {}
    # Getting the type of '_id' (line 1126)
    _id_31568 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1126, 19), '_id', False)
    # Obtaining the member 'idz_id2svd' of a type (line 1126)
    idz_id2svd_31569 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1126, 19), _id_31568, 'idz_id2svd')
    # Calling idz_id2svd(args, kwargs) (line 1126)
    idz_id2svd_call_result_31574 = invoke(stypy.reporting.localization.Localization(__file__, 1126, 19), idz_id2svd_31569, *[B_31570, idx_31571, proj_31572], **kwargs_31573)
    
    # Obtaining the member '__getitem__' of a type (line 1126)
    getitem___31575 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1126, 4), idz_id2svd_call_result_31574, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1126)
    subscript_call_result_31576 = invoke(stypy.reporting.localization.Localization(__file__, 1126, 4), getitem___31575, int_31567)
    
    # Assigning a type to the variable 'tuple_var_assignment_29711' (line 1126)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1126, 4), 'tuple_var_assignment_29711', subscript_call_result_31576)
    
    # Assigning a Name to a Name (line 1126):
    # Getting the type of 'tuple_var_assignment_29708' (line 1126)
    tuple_var_assignment_29708_31577 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1126, 4), 'tuple_var_assignment_29708')
    # Assigning a type to the variable 'U' (line 1126)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1126, 4), 'U', tuple_var_assignment_29708_31577)
    
    # Assigning a Name to a Name (line 1126):
    # Getting the type of 'tuple_var_assignment_29709' (line 1126)
    tuple_var_assignment_29709_31578 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1126, 4), 'tuple_var_assignment_29709')
    # Assigning a type to the variable 'V' (line 1126)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1126, 7), 'V', tuple_var_assignment_29709_31578)
    
    # Assigning a Name to a Name (line 1126):
    # Getting the type of 'tuple_var_assignment_29710' (line 1126)
    tuple_var_assignment_29710_31579 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1126, 4), 'tuple_var_assignment_29710')
    # Assigning a type to the variable 'S' (line 1126)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1126, 10), 'S', tuple_var_assignment_29710_31579)
    
    # Assigning a Name to a Name (line 1126):
    # Getting the type of 'tuple_var_assignment_29711' (line 1126)
    tuple_var_assignment_29711_31580 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1126, 4), 'tuple_var_assignment_29711')
    # Assigning a type to the variable 'ier' (line 1126)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1126, 13), 'ier', tuple_var_assignment_29711_31580)
    
    # Getting the type of 'ier' (line 1127)
    ier_31581 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1127, 7), 'ier')
    # Testing the type of an if condition (line 1127)
    if_condition_31582 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1127, 4), ier_31581)
    # Assigning a type to the variable 'if_condition_31582' (line 1127)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1127, 4), 'if_condition_31582', if_condition_31582)
    # SSA begins for if statement (line 1127)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # Getting the type of '_RETCODE_ERROR' (line 1128)
    _RETCODE_ERROR_31583 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1128, 14), '_RETCODE_ERROR')
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 1128, 8), _RETCODE_ERROR_31583, 'raise parameter', BaseException)
    # SSA join for if statement (line 1127)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Obtaining an instance of the builtin type 'tuple' (line 1129)
    tuple_31584 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1129, 11), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 1129)
    # Adding element type (line 1129)
    # Getting the type of 'U' (line 1129)
    U_31585 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1129, 11), 'U')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1129, 11), tuple_31584, U_31585)
    # Adding element type (line 1129)
    # Getting the type of 'V' (line 1129)
    V_31586 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1129, 14), 'V')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1129, 11), tuple_31584, V_31586)
    # Adding element type (line 1129)
    # Getting the type of 'S' (line 1129)
    S_31587 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1129, 17), 'S')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1129, 11), tuple_31584, S_31587)
    
    # Assigning a type to the variable 'stypy_return_type' (line 1129)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1129, 4), 'stypy_return_type', tuple_31584)
    
    # ################# End of 'idz_id2svd(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'idz_id2svd' in the type store
    # Getting the type of 'stypy_return_type' (line 1101)
    stypy_return_type_31588 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1101, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_31588)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'idz_id2svd'
    return stypy_return_type_31588

# Assigning a type to the variable 'idz_id2svd' (line 1101)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1101, 0), 'idz_id2svd', idz_id2svd)

@norecursion
def idz_snorm(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    int_31589 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1136, 41), 'int')
    defaults = [int_31589]
    # Create a new context for function 'idz_snorm'
    module_type_store = module_type_store.open_function_context('idz_snorm', 1136, 0, False)
    
    # Passed parameters checking function
    idz_snorm.stypy_localization = localization
    idz_snorm.stypy_type_of_self = None
    idz_snorm.stypy_type_store = module_type_store
    idz_snorm.stypy_function_name = 'idz_snorm'
    idz_snorm.stypy_param_names_list = ['m', 'n', 'matveca', 'matvec', 'its']
    idz_snorm.stypy_varargs_param_name = None
    idz_snorm.stypy_kwargs_param_name = None
    idz_snorm.stypy_call_defaults = defaults
    idz_snorm.stypy_call_varargs = varargs
    idz_snorm.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'idz_snorm', ['m', 'n', 'matveca', 'matvec', 'its'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'idz_snorm', localization, ['m', 'n', 'matveca', 'matvec', 'its'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'idz_snorm(...)' code ##################

    str_31590 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1163, (-1)), 'str', '\n    Estimate spectral norm of a complex matrix by the randomized power method.\n\n    :param m:\n        Matrix row dimension.\n    :type m: int\n    :param n:\n        Matrix column dimension.\n    :type n: int\n    :param matveca:\n        Function to apply the matrix adjoint to a vector, with call signature\n        `y = matveca(x)`, where `x` and `y` are the input and output vectors,\n        respectively.\n    :type matveca: function\n    :param matvec:\n        Function to apply the matrix to a vector, with call signature\n        `y = matvec(x)`, where `x` and `y` are the input and output vectors,\n        respectively.\n    :type matvec: function\n    :param its:\n        Number of power method iterations.\n    :type its: int\n\n    :return:\n        Spectral norm estimate.\n    :rtype: float\n    ')
    
    # Assigning a Call to a Tuple (line 1164):
    
    # Assigning a Subscript to a Name (line 1164):
    
    # Obtaining the type of the subscript
    int_31591 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1164, 4), 'int')
    
    # Call to idz_snorm(...): (line 1164)
    # Processing the call arguments (line 1164)
    # Getting the type of 'm' (line 1164)
    m_31594 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1164, 29), 'm', False)
    # Getting the type of 'n' (line 1164)
    n_31595 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1164, 32), 'n', False)
    # Getting the type of 'matveca' (line 1164)
    matveca_31596 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1164, 35), 'matveca', False)
    # Getting the type of 'matvec' (line 1164)
    matvec_31597 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1164, 44), 'matvec', False)
    # Getting the type of 'its' (line 1164)
    its_31598 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1164, 52), 'its', False)
    # Processing the call keyword arguments (line 1164)
    kwargs_31599 = {}
    # Getting the type of '_id' (line 1164)
    _id_31592 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1164, 15), '_id', False)
    # Obtaining the member 'idz_snorm' of a type (line 1164)
    idz_snorm_31593 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1164, 15), _id_31592, 'idz_snorm')
    # Calling idz_snorm(args, kwargs) (line 1164)
    idz_snorm_call_result_31600 = invoke(stypy.reporting.localization.Localization(__file__, 1164, 15), idz_snorm_31593, *[m_31594, n_31595, matveca_31596, matvec_31597, its_31598], **kwargs_31599)
    
    # Obtaining the member '__getitem__' of a type (line 1164)
    getitem___31601 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1164, 4), idz_snorm_call_result_31600, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1164)
    subscript_call_result_31602 = invoke(stypy.reporting.localization.Localization(__file__, 1164, 4), getitem___31601, int_31591)
    
    # Assigning a type to the variable 'tuple_var_assignment_29712' (line 1164)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1164, 4), 'tuple_var_assignment_29712', subscript_call_result_31602)
    
    # Assigning a Subscript to a Name (line 1164):
    
    # Obtaining the type of the subscript
    int_31603 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1164, 4), 'int')
    
    # Call to idz_snorm(...): (line 1164)
    # Processing the call arguments (line 1164)
    # Getting the type of 'm' (line 1164)
    m_31606 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1164, 29), 'm', False)
    # Getting the type of 'n' (line 1164)
    n_31607 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1164, 32), 'n', False)
    # Getting the type of 'matveca' (line 1164)
    matveca_31608 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1164, 35), 'matveca', False)
    # Getting the type of 'matvec' (line 1164)
    matvec_31609 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1164, 44), 'matvec', False)
    # Getting the type of 'its' (line 1164)
    its_31610 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1164, 52), 'its', False)
    # Processing the call keyword arguments (line 1164)
    kwargs_31611 = {}
    # Getting the type of '_id' (line 1164)
    _id_31604 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1164, 15), '_id', False)
    # Obtaining the member 'idz_snorm' of a type (line 1164)
    idz_snorm_31605 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1164, 15), _id_31604, 'idz_snorm')
    # Calling idz_snorm(args, kwargs) (line 1164)
    idz_snorm_call_result_31612 = invoke(stypy.reporting.localization.Localization(__file__, 1164, 15), idz_snorm_31605, *[m_31606, n_31607, matveca_31608, matvec_31609, its_31610], **kwargs_31611)
    
    # Obtaining the member '__getitem__' of a type (line 1164)
    getitem___31613 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1164, 4), idz_snorm_call_result_31612, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1164)
    subscript_call_result_31614 = invoke(stypy.reporting.localization.Localization(__file__, 1164, 4), getitem___31613, int_31603)
    
    # Assigning a type to the variable 'tuple_var_assignment_29713' (line 1164)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1164, 4), 'tuple_var_assignment_29713', subscript_call_result_31614)
    
    # Assigning a Name to a Name (line 1164):
    # Getting the type of 'tuple_var_assignment_29712' (line 1164)
    tuple_var_assignment_29712_31615 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1164, 4), 'tuple_var_assignment_29712')
    # Assigning a type to the variable 'snorm' (line 1164)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1164, 4), 'snorm', tuple_var_assignment_29712_31615)
    
    # Assigning a Name to a Name (line 1164):
    # Getting the type of 'tuple_var_assignment_29713' (line 1164)
    tuple_var_assignment_29713_31616 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1164, 4), 'tuple_var_assignment_29713')
    # Assigning a type to the variable 'v' (line 1164)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1164, 11), 'v', tuple_var_assignment_29713_31616)
    # Getting the type of 'snorm' (line 1165)
    snorm_31617 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1165, 11), 'snorm')
    # Assigning a type to the variable 'stypy_return_type' (line 1165)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1165, 4), 'stypy_return_type', snorm_31617)
    
    # ################# End of 'idz_snorm(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'idz_snorm' in the type store
    # Getting the type of 'stypy_return_type' (line 1136)
    stypy_return_type_31618 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1136, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_31618)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'idz_snorm'
    return stypy_return_type_31618

# Assigning a type to the variable 'idz_snorm' (line 1136)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1136, 0), 'idz_snorm', idz_snorm)

@norecursion
def idz_diffsnorm(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    int_31619 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1168, 64), 'int')
    defaults = [int_31619]
    # Create a new context for function 'idz_diffsnorm'
    module_type_store = module_type_store.open_function_context('idz_diffsnorm', 1168, 0, False)
    
    # Passed parameters checking function
    idz_diffsnorm.stypy_localization = localization
    idz_diffsnorm.stypy_type_of_self = None
    idz_diffsnorm.stypy_type_store = module_type_store
    idz_diffsnorm.stypy_function_name = 'idz_diffsnorm'
    idz_diffsnorm.stypy_param_names_list = ['m', 'n', 'matveca', 'matveca2', 'matvec', 'matvec2', 'its']
    idz_diffsnorm.stypy_varargs_param_name = None
    idz_diffsnorm.stypy_kwargs_param_name = None
    idz_diffsnorm.stypy_call_defaults = defaults
    idz_diffsnorm.stypy_call_varargs = varargs
    idz_diffsnorm.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'idz_diffsnorm', ['m', 'n', 'matveca', 'matveca2', 'matvec', 'matvec2', 'its'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'idz_diffsnorm', localization, ['m', 'n', 'matveca', 'matveca2', 'matvec', 'matvec2', 'its'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'idz_diffsnorm(...)' code ##################

    str_31620 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1206, (-1)), 'str', '\n    Estimate spectral norm of the difference of two complex matrices by the\n    randomized power method.\n\n    :param m:\n        Matrix row dimension.\n    :type m: int\n    :param n:\n        Matrix column dimension.\n    :type n: int\n    :param matveca:\n        Function to apply the adjoint of the first matrix to a vector, with\n        call signature `y = matveca(x)`, where `x` and `y` are the input and\n        output vectors, respectively.\n    :type matveca: function\n    :param matveca2:\n        Function to apply the adjoint of the second matrix to a vector, with\n        call signature `y = matveca2(x)`, where `x` and `y` are the input and\n        output vectors, respectively.\n    :type matveca2: function\n    :param matvec:\n        Function to apply the first matrix to a vector, with call signature\n        `y = matvec(x)`, where `x` and `y` are the input and output vectors,\n        respectively.\n    :type matvec: function\n    :param matvec2:\n        Function to apply the second matrix to a vector, with call signature\n        `y = matvec2(x)`, where `x` and `y` are the input and output vectors,\n        respectively.\n    :type matvec2: function\n    :param its:\n        Number of power method iterations.\n    :type its: int\n\n    :return:\n        Spectral norm estimate of matrix difference.\n    :rtype: float\n    ')
    
    # Call to idz_diffsnorm(...): (line 1207)
    # Processing the call arguments (line 1207)
    # Getting the type of 'm' (line 1207)
    m_31623 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1207, 29), 'm', False)
    # Getting the type of 'n' (line 1207)
    n_31624 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1207, 32), 'n', False)
    # Getting the type of 'matveca' (line 1207)
    matveca_31625 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1207, 35), 'matveca', False)
    # Getting the type of 'matveca2' (line 1207)
    matveca2_31626 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1207, 44), 'matveca2', False)
    # Getting the type of 'matvec' (line 1207)
    matvec_31627 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1207, 54), 'matvec', False)
    # Getting the type of 'matvec2' (line 1207)
    matvec2_31628 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1207, 62), 'matvec2', False)
    # Getting the type of 'its' (line 1207)
    its_31629 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1207, 71), 'its', False)
    # Processing the call keyword arguments (line 1207)
    kwargs_31630 = {}
    # Getting the type of '_id' (line 1207)
    _id_31621 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1207, 11), '_id', False)
    # Obtaining the member 'idz_diffsnorm' of a type (line 1207)
    idz_diffsnorm_31622 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1207, 11), _id_31621, 'idz_diffsnorm')
    # Calling idz_diffsnorm(args, kwargs) (line 1207)
    idz_diffsnorm_call_result_31631 = invoke(stypy.reporting.localization.Localization(__file__, 1207, 11), idz_diffsnorm_31622, *[m_31623, n_31624, matveca_31625, matveca2_31626, matvec_31627, matvec2_31628, its_31629], **kwargs_31630)
    
    # Assigning a type to the variable 'stypy_return_type' (line 1207)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1207, 4), 'stypy_return_type', idz_diffsnorm_call_result_31631)
    
    # ################# End of 'idz_diffsnorm(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'idz_diffsnorm' in the type store
    # Getting the type of 'stypy_return_type' (line 1168)
    stypy_return_type_31632 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1168, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_31632)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'idz_diffsnorm'
    return stypy_return_type_31632

# Assigning a type to the variable 'idz_diffsnorm' (line 1168)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1168, 0), 'idz_diffsnorm', idz_diffsnorm)

@norecursion
def idzr_svd(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'idzr_svd'
    module_type_store = module_type_store.open_function_context('idzr_svd', 1214, 0, False)
    
    # Passed parameters checking function
    idzr_svd.stypy_localization = localization
    idzr_svd.stypy_type_of_self = None
    idzr_svd.stypy_type_store = module_type_store
    idzr_svd.stypy_function_name = 'idzr_svd'
    idzr_svd.stypy_param_names_list = ['A', 'k']
    idzr_svd.stypy_varargs_param_name = None
    idzr_svd.stypy_kwargs_param_name = None
    idzr_svd.stypy_call_defaults = defaults
    idzr_svd.stypy_call_varargs = varargs
    idzr_svd.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'idzr_svd', ['A', 'k'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'idzr_svd', localization, ['A', 'k'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'idzr_svd(...)' code ##################

    str_31633 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1234, (-1)), 'str', '\n    Compute SVD of a complex matrix to a specified rank.\n\n    :param A:\n        Matrix.\n    :type A: :class:`numpy.ndarray`\n    :param k:\n        Rank of SVD.\n    :type k: int\n\n    :return:\n        Left singular vectors.\n    :rtype: :class:`numpy.ndarray`\n    :return:\n        Right singular vectors.\n    :rtype: :class:`numpy.ndarray`\n    :return:\n        Singular values.\n    :rtype: :class:`numpy.ndarray`\n    ')
    
    # Assigning a Call to a Name (line 1235):
    
    # Assigning a Call to a Name (line 1235):
    
    # Call to asfortranarray(...): (line 1235)
    # Processing the call arguments (line 1235)
    # Getting the type of 'A' (line 1235)
    A_31636 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1235, 26), 'A', False)
    # Processing the call keyword arguments (line 1235)
    kwargs_31637 = {}
    # Getting the type of 'np' (line 1235)
    np_31634 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1235, 8), 'np', False)
    # Obtaining the member 'asfortranarray' of a type (line 1235)
    asfortranarray_31635 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1235, 8), np_31634, 'asfortranarray')
    # Calling asfortranarray(args, kwargs) (line 1235)
    asfortranarray_call_result_31638 = invoke(stypy.reporting.localization.Localization(__file__, 1235, 8), asfortranarray_31635, *[A_31636], **kwargs_31637)
    
    # Assigning a type to the variable 'A' (line 1235)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1235, 4), 'A', asfortranarray_call_result_31638)
    
    # Assigning a Call to a Tuple (line 1236):
    
    # Assigning a Subscript to a Name (line 1236):
    
    # Obtaining the type of the subscript
    int_31639 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1236, 4), 'int')
    
    # Call to idzr_svd(...): (line 1236)
    # Processing the call arguments (line 1236)
    # Getting the type of 'A' (line 1236)
    A_31642 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1236, 32), 'A', False)
    # Getting the type of 'k' (line 1236)
    k_31643 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1236, 35), 'k', False)
    # Processing the call keyword arguments (line 1236)
    kwargs_31644 = {}
    # Getting the type of '_id' (line 1236)
    _id_31640 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1236, 19), '_id', False)
    # Obtaining the member 'idzr_svd' of a type (line 1236)
    idzr_svd_31641 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1236, 19), _id_31640, 'idzr_svd')
    # Calling idzr_svd(args, kwargs) (line 1236)
    idzr_svd_call_result_31645 = invoke(stypy.reporting.localization.Localization(__file__, 1236, 19), idzr_svd_31641, *[A_31642, k_31643], **kwargs_31644)
    
    # Obtaining the member '__getitem__' of a type (line 1236)
    getitem___31646 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1236, 4), idzr_svd_call_result_31645, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1236)
    subscript_call_result_31647 = invoke(stypy.reporting.localization.Localization(__file__, 1236, 4), getitem___31646, int_31639)
    
    # Assigning a type to the variable 'tuple_var_assignment_29714' (line 1236)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1236, 4), 'tuple_var_assignment_29714', subscript_call_result_31647)
    
    # Assigning a Subscript to a Name (line 1236):
    
    # Obtaining the type of the subscript
    int_31648 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1236, 4), 'int')
    
    # Call to idzr_svd(...): (line 1236)
    # Processing the call arguments (line 1236)
    # Getting the type of 'A' (line 1236)
    A_31651 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1236, 32), 'A', False)
    # Getting the type of 'k' (line 1236)
    k_31652 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1236, 35), 'k', False)
    # Processing the call keyword arguments (line 1236)
    kwargs_31653 = {}
    # Getting the type of '_id' (line 1236)
    _id_31649 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1236, 19), '_id', False)
    # Obtaining the member 'idzr_svd' of a type (line 1236)
    idzr_svd_31650 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1236, 19), _id_31649, 'idzr_svd')
    # Calling idzr_svd(args, kwargs) (line 1236)
    idzr_svd_call_result_31654 = invoke(stypy.reporting.localization.Localization(__file__, 1236, 19), idzr_svd_31650, *[A_31651, k_31652], **kwargs_31653)
    
    # Obtaining the member '__getitem__' of a type (line 1236)
    getitem___31655 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1236, 4), idzr_svd_call_result_31654, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1236)
    subscript_call_result_31656 = invoke(stypy.reporting.localization.Localization(__file__, 1236, 4), getitem___31655, int_31648)
    
    # Assigning a type to the variable 'tuple_var_assignment_29715' (line 1236)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1236, 4), 'tuple_var_assignment_29715', subscript_call_result_31656)
    
    # Assigning a Subscript to a Name (line 1236):
    
    # Obtaining the type of the subscript
    int_31657 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1236, 4), 'int')
    
    # Call to idzr_svd(...): (line 1236)
    # Processing the call arguments (line 1236)
    # Getting the type of 'A' (line 1236)
    A_31660 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1236, 32), 'A', False)
    # Getting the type of 'k' (line 1236)
    k_31661 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1236, 35), 'k', False)
    # Processing the call keyword arguments (line 1236)
    kwargs_31662 = {}
    # Getting the type of '_id' (line 1236)
    _id_31658 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1236, 19), '_id', False)
    # Obtaining the member 'idzr_svd' of a type (line 1236)
    idzr_svd_31659 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1236, 19), _id_31658, 'idzr_svd')
    # Calling idzr_svd(args, kwargs) (line 1236)
    idzr_svd_call_result_31663 = invoke(stypy.reporting.localization.Localization(__file__, 1236, 19), idzr_svd_31659, *[A_31660, k_31661], **kwargs_31662)
    
    # Obtaining the member '__getitem__' of a type (line 1236)
    getitem___31664 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1236, 4), idzr_svd_call_result_31663, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1236)
    subscript_call_result_31665 = invoke(stypy.reporting.localization.Localization(__file__, 1236, 4), getitem___31664, int_31657)
    
    # Assigning a type to the variable 'tuple_var_assignment_29716' (line 1236)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1236, 4), 'tuple_var_assignment_29716', subscript_call_result_31665)
    
    # Assigning a Subscript to a Name (line 1236):
    
    # Obtaining the type of the subscript
    int_31666 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1236, 4), 'int')
    
    # Call to idzr_svd(...): (line 1236)
    # Processing the call arguments (line 1236)
    # Getting the type of 'A' (line 1236)
    A_31669 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1236, 32), 'A', False)
    # Getting the type of 'k' (line 1236)
    k_31670 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1236, 35), 'k', False)
    # Processing the call keyword arguments (line 1236)
    kwargs_31671 = {}
    # Getting the type of '_id' (line 1236)
    _id_31667 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1236, 19), '_id', False)
    # Obtaining the member 'idzr_svd' of a type (line 1236)
    idzr_svd_31668 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1236, 19), _id_31667, 'idzr_svd')
    # Calling idzr_svd(args, kwargs) (line 1236)
    idzr_svd_call_result_31672 = invoke(stypy.reporting.localization.Localization(__file__, 1236, 19), idzr_svd_31668, *[A_31669, k_31670], **kwargs_31671)
    
    # Obtaining the member '__getitem__' of a type (line 1236)
    getitem___31673 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1236, 4), idzr_svd_call_result_31672, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1236)
    subscript_call_result_31674 = invoke(stypy.reporting.localization.Localization(__file__, 1236, 4), getitem___31673, int_31666)
    
    # Assigning a type to the variable 'tuple_var_assignment_29717' (line 1236)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1236, 4), 'tuple_var_assignment_29717', subscript_call_result_31674)
    
    # Assigning a Name to a Name (line 1236):
    # Getting the type of 'tuple_var_assignment_29714' (line 1236)
    tuple_var_assignment_29714_31675 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1236, 4), 'tuple_var_assignment_29714')
    # Assigning a type to the variable 'U' (line 1236)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1236, 4), 'U', tuple_var_assignment_29714_31675)
    
    # Assigning a Name to a Name (line 1236):
    # Getting the type of 'tuple_var_assignment_29715' (line 1236)
    tuple_var_assignment_29715_31676 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1236, 4), 'tuple_var_assignment_29715')
    # Assigning a type to the variable 'V' (line 1236)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1236, 7), 'V', tuple_var_assignment_29715_31676)
    
    # Assigning a Name to a Name (line 1236):
    # Getting the type of 'tuple_var_assignment_29716' (line 1236)
    tuple_var_assignment_29716_31677 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1236, 4), 'tuple_var_assignment_29716')
    # Assigning a type to the variable 'S' (line 1236)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1236, 10), 'S', tuple_var_assignment_29716_31677)
    
    # Assigning a Name to a Name (line 1236):
    # Getting the type of 'tuple_var_assignment_29717' (line 1236)
    tuple_var_assignment_29717_31678 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1236, 4), 'tuple_var_assignment_29717')
    # Assigning a type to the variable 'ier' (line 1236)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1236, 13), 'ier', tuple_var_assignment_29717_31678)
    
    # Getting the type of 'ier' (line 1237)
    ier_31679 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1237, 7), 'ier')
    # Testing the type of an if condition (line 1237)
    if_condition_31680 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1237, 4), ier_31679)
    # Assigning a type to the variable 'if_condition_31680' (line 1237)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1237, 4), 'if_condition_31680', if_condition_31680)
    # SSA begins for if statement (line 1237)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # Getting the type of '_RETCODE_ERROR' (line 1238)
    _RETCODE_ERROR_31681 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1238, 14), '_RETCODE_ERROR')
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 1238, 8), _RETCODE_ERROR_31681, 'raise parameter', BaseException)
    # SSA join for if statement (line 1237)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Obtaining an instance of the builtin type 'tuple' (line 1239)
    tuple_31682 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1239, 11), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 1239)
    # Adding element type (line 1239)
    # Getting the type of 'U' (line 1239)
    U_31683 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1239, 11), 'U')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1239, 11), tuple_31682, U_31683)
    # Adding element type (line 1239)
    # Getting the type of 'V' (line 1239)
    V_31684 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1239, 14), 'V')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1239, 11), tuple_31682, V_31684)
    # Adding element type (line 1239)
    # Getting the type of 'S' (line 1239)
    S_31685 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1239, 17), 'S')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1239, 11), tuple_31682, S_31685)
    
    # Assigning a type to the variable 'stypy_return_type' (line 1239)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1239, 4), 'stypy_return_type', tuple_31682)
    
    # ################# End of 'idzr_svd(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'idzr_svd' in the type store
    # Getting the type of 'stypy_return_type' (line 1214)
    stypy_return_type_31686 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1214, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_31686)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'idzr_svd'
    return stypy_return_type_31686

# Assigning a type to the variable 'idzr_svd' (line 1214)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1214, 0), 'idzr_svd', idzr_svd)

@norecursion
def idzp_svd(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'idzp_svd'
    module_type_store = module_type_store.open_function_context('idzp_svd', 1242, 0, False)
    
    # Passed parameters checking function
    idzp_svd.stypy_localization = localization
    idzp_svd.stypy_type_of_self = None
    idzp_svd.stypy_type_store = module_type_store
    idzp_svd.stypy_function_name = 'idzp_svd'
    idzp_svd.stypy_param_names_list = ['eps', 'A']
    idzp_svd.stypy_varargs_param_name = None
    idzp_svd.stypy_kwargs_param_name = None
    idzp_svd.stypy_call_defaults = defaults
    idzp_svd.stypy_call_varargs = varargs
    idzp_svd.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'idzp_svd', ['eps', 'A'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'idzp_svd', localization, ['eps', 'A'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'idzp_svd(...)' code ##################

    str_31687 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1262, (-1)), 'str', '\n    Compute SVD of a complex matrix to a specified relative precision.\n\n    :param eps:\n        Relative precision.\n    :type eps: float\n    :param A:\n        Matrix.\n    :type A: :class:`numpy.ndarray`\n\n    :return:\n        Left singular vectors.\n    :rtype: :class:`numpy.ndarray`\n    :return:\n        Right singular vectors.\n    :rtype: :class:`numpy.ndarray`\n    :return:\n        Singular values.\n    :rtype: :class:`numpy.ndarray`\n    ')
    
    # Assigning a Call to a Name (line 1263):
    
    # Assigning a Call to a Name (line 1263):
    
    # Call to asfortranarray(...): (line 1263)
    # Processing the call arguments (line 1263)
    # Getting the type of 'A' (line 1263)
    A_31690 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1263, 26), 'A', False)
    # Processing the call keyword arguments (line 1263)
    kwargs_31691 = {}
    # Getting the type of 'np' (line 1263)
    np_31688 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1263, 8), 'np', False)
    # Obtaining the member 'asfortranarray' of a type (line 1263)
    asfortranarray_31689 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1263, 8), np_31688, 'asfortranarray')
    # Calling asfortranarray(args, kwargs) (line 1263)
    asfortranarray_call_result_31692 = invoke(stypy.reporting.localization.Localization(__file__, 1263, 8), asfortranarray_31689, *[A_31690], **kwargs_31691)
    
    # Assigning a type to the variable 'A' (line 1263)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1263, 4), 'A', asfortranarray_call_result_31692)
    
    # Assigning a Attribute to a Tuple (line 1264):
    
    # Assigning a Subscript to a Name (line 1264):
    
    # Obtaining the type of the subscript
    int_31693 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1264, 4), 'int')
    # Getting the type of 'A' (line 1264)
    A_31694 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1264, 11), 'A')
    # Obtaining the member 'shape' of a type (line 1264)
    shape_31695 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1264, 11), A_31694, 'shape')
    # Obtaining the member '__getitem__' of a type (line 1264)
    getitem___31696 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1264, 4), shape_31695, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1264)
    subscript_call_result_31697 = invoke(stypy.reporting.localization.Localization(__file__, 1264, 4), getitem___31696, int_31693)
    
    # Assigning a type to the variable 'tuple_var_assignment_29718' (line 1264)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1264, 4), 'tuple_var_assignment_29718', subscript_call_result_31697)
    
    # Assigning a Subscript to a Name (line 1264):
    
    # Obtaining the type of the subscript
    int_31698 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1264, 4), 'int')
    # Getting the type of 'A' (line 1264)
    A_31699 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1264, 11), 'A')
    # Obtaining the member 'shape' of a type (line 1264)
    shape_31700 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1264, 11), A_31699, 'shape')
    # Obtaining the member '__getitem__' of a type (line 1264)
    getitem___31701 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1264, 4), shape_31700, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1264)
    subscript_call_result_31702 = invoke(stypy.reporting.localization.Localization(__file__, 1264, 4), getitem___31701, int_31698)
    
    # Assigning a type to the variable 'tuple_var_assignment_29719' (line 1264)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1264, 4), 'tuple_var_assignment_29719', subscript_call_result_31702)
    
    # Assigning a Name to a Name (line 1264):
    # Getting the type of 'tuple_var_assignment_29718' (line 1264)
    tuple_var_assignment_29718_31703 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1264, 4), 'tuple_var_assignment_29718')
    # Assigning a type to the variable 'm' (line 1264)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1264, 4), 'm', tuple_var_assignment_29718_31703)
    
    # Assigning a Name to a Name (line 1264):
    # Getting the type of 'tuple_var_assignment_29719' (line 1264)
    tuple_var_assignment_29719_31704 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1264, 4), 'tuple_var_assignment_29719')
    # Assigning a type to the variable 'n' (line 1264)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1264, 7), 'n', tuple_var_assignment_29719_31704)
    
    # Assigning a Call to a Tuple (line 1265):
    
    # Assigning a Subscript to a Name (line 1265):
    
    # Obtaining the type of the subscript
    int_31705 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1265, 4), 'int')
    
    # Call to idzp_svd(...): (line 1265)
    # Processing the call arguments (line 1265)
    # Getting the type of 'eps' (line 1265)
    eps_31708 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1265, 41), 'eps', False)
    # Getting the type of 'A' (line 1265)
    A_31709 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1265, 46), 'A', False)
    # Processing the call keyword arguments (line 1265)
    kwargs_31710 = {}
    # Getting the type of '_id' (line 1265)
    _id_31706 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1265, 28), '_id', False)
    # Obtaining the member 'idzp_svd' of a type (line 1265)
    idzp_svd_31707 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1265, 28), _id_31706, 'idzp_svd')
    # Calling idzp_svd(args, kwargs) (line 1265)
    idzp_svd_call_result_31711 = invoke(stypy.reporting.localization.Localization(__file__, 1265, 28), idzp_svd_31707, *[eps_31708, A_31709], **kwargs_31710)
    
    # Obtaining the member '__getitem__' of a type (line 1265)
    getitem___31712 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1265, 4), idzp_svd_call_result_31711, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1265)
    subscript_call_result_31713 = invoke(stypy.reporting.localization.Localization(__file__, 1265, 4), getitem___31712, int_31705)
    
    # Assigning a type to the variable 'tuple_var_assignment_29720' (line 1265)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1265, 4), 'tuple_var_assignment_29720', subscript_call_result_31713)
    
    # Assigning a Subscript to a Name (line 1265):
    
    # Obtaining the type of the subscript
    int_31714 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1265, 4), 'int')
    
    # Call to idzp_svd(...): (line 1265)
    # Processing the call arguments (line 1265)
    # Getting the type of 'eps' (line 1265)
    eps_31717 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1265, 41), 'eps', False)
    # Getting the type of 'A' (line 1265)
    A_31718 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1265, 46), 'A', False)
    # Processing the call keyword arguments (line 1265)
    kwargs_31719 = {}
    # Getting the type of '_id' (line 1265)
    _id_31715 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1265, 28), '_id', False)
    # Obtaining the member 'idzp_svd' of a type (line 1265)
    idzp_svd_31716 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1265, 28), _id_31715, 'idzp_svd')
    # Calling idzp_svd(args, kwargs) (line 1265)
    idzp_svd_call_result_31720 = invoke(stypy.reporting.localization.Localization(__file__, 1265, 28), idzp_svd_31716, *[eps_31717, A_31718], **kwargs_31719)
    
    # Obtaining the member '__getitem__' of a type (line 1265)
    getitem___31721 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1265, 4), idzp_svd_call_result_31720, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1265)
    subscript_call_result_31722 = invoke(stypy.reporting.localization.Localization(__file__, 1265, 4), getitem___31721, int_31714)
    
    # Assigning a type to the variable 'tuple_var_assignment_29721' (line 1265)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1265, 4), 'tuple_var_assignment_29721', subscript_call_result_31722)
    
    # Assigning a Subscript to a Name (line 1265):
    
    # Obtaining the type of the subscript
    int_31723 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1265, 4), 'int')
    
    # Call to idzp_svd(...): (line 1265)
    # Processing the call arguments (line 1265)
    # Getting the type of 'eps' (line 1265)
    eps_31726 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1265, 41), 'eps', False)
    # Getting the type of 'A' (line 1265)
    A_31727 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1265, 46), 'A', False)
    # Processing the call keyword arguments (line 1265)
    kwargs_31728 = {}
    # Getting the type of '_id' (line 1265)
    _id_31724 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1265, 28), '_id', False)
    # Obtaining the member 'idzp_svd' of a type (line 1265)
    idzp_svd_31725 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1265, 28), _id_31724, 'idzp_svd')
    # Calling idzp_svd(args, kwargs) (line 1265)
    idzp_svd_call_result_31729 = invoke(stypy.reporting.localization.Localization(__file__, 1265, 28), idzp_svd_31725, *[eps_31726, A_31727], **kwargs_31728)
    
    # Obtaining the member '__getitem__' of a type (line 1265)
    getitem___31730 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1265, 4), idzp_svd_call_result_31729, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1265)
    subscript_call_result_31731 = invoke(stypy.reporting.localization.Localization(__file__, 1265, 4), getitem___31730, int_31723)
    
    # Assigning a type to the variable 'tuple_var_assignment_29722' (line 1265)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1265, 4), 'tuple_var_assignment_29722', subscript_call_result_31731)
    
    # Assigning a Subscript to a Name (line 1265):
    
    # Obtaining the type of the subscript
    int_31732 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1265, 4), 'int')
    
    # Call to idzp_svd(...): (line 1265)
    # Processing the call arguments (line 1265)
    # Getting the type of 'eps' (line 1265)
    eps_31735 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1265, 41), 'eps', False)
    # Getting the type of 'A' (line 1265)
    A_31736 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1265, 46), 'A', False)
    # Processing the call keyword arguments (line 1265)
    kwargs_31737 = {}
    # Getting the type of '_id' (line 1265)
    _id_31733 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1265, 28), '_id', False)
    # Obtaining the member 'idzp_svd' of a type (line 1265)
    idzp_svd_31734 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1265, 28), _id_31733, 'idzp_svd')
    # Calling idzp_svd(args, kwargs) (line 1265)
    idzp_svd_call_result_31738 = invoke(stypy.reporting.localization.Localization(__file__, 1265, 28), idzp_svd_31734, *[eps_31735, A_31736], **kwargs_31737)
    
    # Obtaining the member '__getitem__' of a type (line 1265)
    getitem___31739 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1265, 4), idzp_svd_call_result_31738, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1265)
    subscript_call_result_31740 = invoke(stypy.reporting.localization.Localization(__file__, 1265, 4), getitem___31739, int_31732)
    
    # Assigning a type to the variable 'tuple_var_assignment_29723' (line 1265)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1265, 4), 'tuple_var_assignment_29723', subscript_call_result_31740)
    
    # Assigning a Subscript to a Name (line 1265):
    
    # Obtaining the type of the subscript
    int_31741 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1265, 4), 'int')
    
    # Call to idzp_svd(...): (line 1265)
    # Processing the call arguments (line 1265)
    # Getting the type of 'eps' (line 1265)
    eps_31744 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1265, 41), 'eps', False)
    # Getting the type of 'A' (line 1265)
    A_31745 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1265, 46), 'A', False)
    # Processing the call keyword arguments (line 1265)
    kwargs_31746 = {}
    # Getting the type of '_id' (line 1265)
    _id_31742 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1265, 28), '_id', False)
    # Obtaining the member 'idzp_svd' of a type (line 1265)
    idzp_svd_31743 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1265, 28), _id_31742, 'idzp_svd')
    # Calling idzp_svd(args, kwargs) (line 1265)
    idzp_svd_call_result_31747 = invoke(stypy.reporting.localization.Localization(__file__, 1265, 28), idzp_svd_31743, *[eps_31744, A_31745], **kwargs_31746)
    
    # Obtaining the member '__getitem__' of a type (line 1265)
    getitem___31748 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1265, 4), idzp_svd_call_result_31747, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1265)
    subscript_call_result_31749 = invoke(stypy.reporting.localization.Localization(__file__, 1265, 4), getitem___31748, int_31741)
    
    # Assigning a type to the variable 'tuple_var_assignment_29724' (line 1265)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1265, 4), 'tuple_var_assignment_29724', subscript_call_result_31749)
    
    # Assigning a Subscript to a Name (line 1265):
    
    # Obtaining the type of the subscript
    int_31750 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1265, 4), 'int')
    
    # Call to idzp_svd(...): (line 1265)
    # Processing the call arguments (line 1265)
    # Getting the type of 'eps' (line 1265)
    eps_31753 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1265, 41), 'eps', False)
    # Getting the type of 'A' (line 1265)
    A_31754 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1265, 46), 'A', False)
    # Processing the call keyword arguments (line 1265)
    kwargs_31755 = {}
    # Getting the type of '_id' (line 1265)
    _id_31751 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1265, 28), '_id', False)
    # Obtaining the member 'idzp_svd' of a type (line 1265)
    idzp_svd_31752 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1265, 28), _id_31751, 'idzp_svd')
    # Calling idzp_svd(args, kwargs) (line 1265)
    idzp_svd_call_result_31756 = invoke(stypy.reporting.localization.Localization(__file__, 1265, 28), idzp_svd_31752, *[eps_31753, A_31754], **kwargs_31755)
    
    # Obtaining the member '__getitem__' of a type (line 1265)
    getitem___31757 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1265, 4), idzp_svd_call_result_31756, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1265)
    subscript_call_result_31758 = invoke(stypy.reporting.localization.Localization(__file__, 1265, 4), getitem___31757, int_31750)
    
    # Assigning a type to the variable 'tuple_var_assignment_29725' (line 1265)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1265, 4), 'tuple_var_assignment_29725', subscript_call_result_31758)
    
    # Assigning a Name to a Name (line 1265):
    # Getting the type of 'tuple_var_assignment_29720' (line 1265)
    tuple_var_assignment_29720_31759 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1265, 4), 'tuple_var_assignment_29720')
    # Assigning a type to the variable 'k' (line 1265)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1265, 4), 'k', tuple_var_assignment_29720_31759)
    
    # Assigning a Name to a Name (line 1265):
    # Getting the type of 'tuple_var_assignment_29721' (line 1265)
    tuple_var_assignment_29721_31760 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1265, 4), 'tuple_var_assignment_29721')
    # Assigning a type to the variable 'iU' (line 1265)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1265, 7), 'iU', tuple_var_assignment_29721_31760)
    
    # Assigning a Name to a Name (line 1265):
    # Getting the type of 'tuple_var_assignment_29722' (line 1265)
    tuple_var_assignment_29722_31761 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1265, 4), 'tuple_var_assignment_29722')
    # Assigning a type to the variable 'iV' (line 1265)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1265, 11), 'iV', tuple_var_assignment_29722_31761)
    
    # Assigning a Name to a Name (line 1265):
    # Getting the type of 'tuple_var_assignment_29723' (line 1265)
    tuple_var_assignment_29723_31762 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1265, 4), 'tuple_var_assignment_29723')
    # Assigning a type to the variable 'iS' (line 1265)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1265, 15), 'iS', tuple_var_assignment_29723_31762)
    
    # Assigning a Name to a Name (line 1265):
    # Getting the type of 'tuple_var_assignment_29724' (line 1265)
    tuple_var_assignment_29724_31763 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1265, 4), 'tuple_var_assignment_29724')
    # Assigning a type to the variable 'w' (line 1265)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1265, 19), 'w', tuple_var_assignment_29724_31763)
    
    # Assigning a Name to a Name (line 1265):
    # Getting the type of 'tuple_var_assignment_29725' (line 1265)
    tuple_var_assignment_29725_31764 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1265, 4), 'tuple_var_assignment_29725')
    # Assigning a type to the variable 'ier' (line 1265)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1265, 22), 'ier', tuple_var_assignment_29725_31764)
    
    # Getting the type of 'ier' (line 1266)
    ier_31765 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1266, 7), 'ier')
    # Testing the type of an if condition (line 1266)
    if_condition_31766 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1266, 4), ier_31765)
    # Assigning a type to the variable 'if_condition_31766' (line 1266)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1266, 4), 'if_condition_31766', if_condition_31766)
    # SSA begins for if statement (line 1266)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # Getting the type of '_RETCODE_ERROR' (line 1267)
    _RETCODE_ERROR_31767 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1267, 14), '_RETCODE_ERROR')
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 1267, 8), _RETCODE_ERROR_31767, 'raise parameter', BaseException)
    # SSA join for if statement (line 1266)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 1268):
    
    # Assigning a Call to a Name (line 1268):
    
    # Call to reshape(...): (line 1268)
    # Processing the call arguments (line 1268)
    
    # Obtaining an instance of the builtin type 'tuple' (line 1268)
    tuple_31783 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1268, 34), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 1268)
    # Adding element type (line 1268)
    # Getting the type of 'm' (line 1268)
    m_31784 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1268, 34), 'm', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1268, 34), tuple_31783, m_31784)
    # Adding element type (line 1268)
    # Getting the type of 'k' (line 1268)
    k_31785 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1268, 37), 'k', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1268, 34), tuple_31783, k_31785)
    
    # Processing the call keyword arguments (line 1268)
    str_31786 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1268, 47), 'str', 'F')
    keyword_31787 = str_31786
    kwargs_31788 = {'order': keyword_31787}
    
    # Obtaining the type of the subscript
    # Getting the type of 'iU' (line 1268)
    iU_31768 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1268, 10), 'iU', False)
    int_31769 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1268, 13), 'int')
    # Applying the binary operator '-' (line 1268)
    result_sub_31770 = python_operator(stypy.reporting.localization.Localization(__file__, 1268, 10), '-', iU_31768, int_31769)
    
    # Getting the type of 'iU' (line 1268)
    iU_31771 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1268, 15), 'iU', False)
    # Getting the type of 'm' (line 1268)
    m_31772 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1268, 18), 'm', False)
    # Getting the type of 'k' (line 1268)
    k_31773 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1268, 20), 'k', False)
    # Applying the binary operator '*' (line 1268)
    result_mul_31774 = python_operator(stypy.reporting.localization.Localization(__file__, 1268, 18), '*', m_31772, k_31773)
    
    # Applying the binary operator '+' (line 1268)
    result_add_31775 = python_operator(stypy.reporting.localization.Localization(__file__, 1268, 15), '+', iU_31771, result_mul_31774)
    
    int_31776 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1268, 22), 'int')
    # Applying the binary operator '-' (line 1268)
    result_sub_31777 = python_operator(stypy.reporting.localization.Localization(__file__, 1268, 21), '-', result_add_31775, int_31776)
    
    slice_31778 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 1268, 8), result_sub_31770, result_sub_31777, None)
    # Getting the type of 'w' (line 1268)
    w_31779 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1268, 8), 'w', False)
    # Obtaining the member '__getitem__' of a type (line 1268)
    getitem___31780 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1268, 8), w_31779, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1268)
    subscript_call_result_31781 = invoke(stypy.reporting.localization.Localization(__file__, 1268, 8), getitem___31780, slice_31778)
    
    # Obtaining the member 'reshape' of a type (line 1268)
    reshape_31782 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1268, 8), subscript_call_result_31781, 'reshape')
    # Calling reshape(args, kwargs) (line 1268)
    reshape_call_result_31789 = invoke(stypy.reporting.localization.Localization(__file__, 1268, 8), reshape_31782, *[tuple_31783], **kwargs_31788)
    
    # Assigning a type to the variable 'U' (line 1268)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1268, 4), 'U', reshape_call_result_31789)
    
    # Assigning a Call to a Name (line 1269):
    
    # Assigning a Call to a Name (line 1269):
    
    # Call to reshape(...): (line 1269)
    # Processing the call arguments (line 1269)
    
    # Obtaining an instance of the builtin type 'tuple' (line 1269)
    tuple_31805 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1269, 34), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 1269)
    # Adding element type (line 1269)
    # Getting the type of 'n' (line 1269)
    n_31806 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1269, 34), 'n', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1269, 34), tuple_31805, n_31806)
    # Adding element type (line 1269)
    # Getting the type of 'k' (line 1269)
    k_31807 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1269, 37), 'k', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1269, 34), tuple_31805, k_31807)
    
    # Processing the call keyword arguments (line 1269)
    str_31808 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1269, 47), 'str', 'F')
    keyword_31809 = str_31808
    kwargs_31810 = {'order': keyword_31809}
    
    # Obtaining the type of the subscript
    # Getting the type of 'iV' (line 1269)
    iV_31790 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1269, 10), 'iV', False)
    int_31791 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1269, 13), 'int')
    # Applying the binary operator '-' (line 1269)
    result_sub_31792 = python_operator(stypy.reporting.localization.Localization(__file__, 1269, 10), '-', iV_31790, int_31791)
    
    # Getting the type of 'iV' (line 1269)
    iV_31793 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1269, 15), 'iV', False)
    # Getting the type of 'n' (line 1269)
    n_31794 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1269, 18), 'n', False)
    # Getting the type of 'k' (line 1269)
    k_31795 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1269, 20), 'k', False)
    # Applying the binary operator '*' (line 1269)
    result_mul_31796 = python_operator(stypy.reporting.localization.Localization(__file__, 1269, 18), '*', n_31794, k_31795)
    
    # Applying the binary operator '+' (line 1269)
    result_add_31797 = python_operator(stypy.reporting.localization.Localization(__file__, 1269, 15), '+', iV_31793, result_mul_31796)
    
    int_31798 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1269, 22), 'int')
    # Applying the binary operator '-' (line 1269)
    result_sub_31799 = python_operator(stypy.reporting.localization.Localization(__file__, 1269, 21), '-', result_add_31797, int_31798)
    
    slice_31800 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 1269, 8), result_sub_31792, result_sub_31799, None)
    # Getting the type of 'w' (line 1269)
    w_31801 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1269, 8), 'w', False)
    # Obtaining the member '__getitem__' of a type (line 1269)
    getitem___31802 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1269, 8), w_31801, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1269)
    subscript_call_result_31803 = invoke(stypy.reporting.localization.Localization(__file__, 1269, 8), getitem___31802, slice_31800)
    
    # Obtaining the member 'reshape' of a type (line 1269)
    reshape_31804 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1269, 8), subscript_call_result_31803, 'reshape')
    # Calling reshape(args, kwargs) (line 1269)
    reshape_call_result_31811 = invoke(stypy.reporting.localization.Localization(__file__, 1269, 8), reshape_31804, *[tuple_31805], **kwargs_31810)
    
    # Assigning a type to the variable 'V' (line 1269)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1269, 4), 'V', reshape_call_result_31811)
    
    # Assigning a Subscript to a Name (line 1270):
    
    # Assigning a Subscript to a Name (line 1270):
    
    # Obtaining the type of the subscript
    # Getting the type of 'iS' (line 1270)
    iS_31812 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1270, 10), 'iS')
    int_31813 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1270, 13), 'int')
    # Applying the binary operator '-' (line 1270)
    result_sub_31814 = python_operator(stypy.reporting.localization.Localization(__file__, 1270, 10), '-', iS_31812, int_31813)
    
    # Getting the type of 'iS' (line 1270)
    iS_31815 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1270, 15), 'iS')
    # Getting the type of 'k' (line 1270)
    k_31816 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1270, 18), 'k')
    # Applying the binary operator '+' (line 1270)
    result_add_31817 = python_operator(stypy.reporting.localization.Localization(__file__, 1270, 15), '+', iS_31815, k_31816)
    
    int_31818 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1270, 20), 'int')
    # Applying the binary operator '-' (line 1270)
    result_sub_31819 = python_operator(stypy.reporting.localization.Localization(__file__, 1270, 19), '-', result_add_31817, int_31818)
    
    slice_31820 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 1270, 8), result_sub_31814, result_sub_31819, None)
    # Getting the type of 'w' (line 1270)
    w_31821 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1270, 8), 'w')
    # Obtaining the member '__getitem__' of a type (line 1270)
    getitem___31822 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1270, 8), w_31821, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1270)
    subscript_call_result_31823 = invoke(stypy.reporting.localization.Localization(__file__, 1270, 8), getitem___31822, slice_31820)
    
    # Assigning a type to the variable 'S' (line 1270)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1270, 4), 'S', subscript_call_result_31823)
    
    # Obtaining an instance of the builtin type 'tuple' (line 1271)
    tuple_31824 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1271, 11), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 1271)
    # Adding element type (line 1271)
    # Getting the type of 'U' (line 1271)
    U_31825 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1271, 11), 'U')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1271, 11), tuple_31824, U_31825)
    # Adding element type (line 1271)
    # Getting the type of 'V' (line 1271)
    V_31826 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1271, 14), 'V')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1271, 11), tuple_31824, V_31826)
    # Adding element type (line 1271)
    # Getting the type of 'S' (line 1271)
    S_31827 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1271, 17), 'S')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1271, 11), tuple_31824, S_31827)
    
    # Assigning a type to the variable 'stypy_return_type' (line 1271)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1271, 4), 'stypy_return_type', tuple_31824)
    
    # ################# End of 'idzp_svd(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'idzp_svd' in the type store
    # Getting the type of 'stypy_return_type' (line 1242)
    stypy_return_type_31828 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1242, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_31828)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'idzp_svd'
    return stypy_return_type_31828

# Assigning a type to the variable 'idzp_svd' (line 1242)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1242, 0), 'idzp_svd', idzp_svd)

@norecursion
def idzp_aid(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'idzp_aid'
    module_type_store = module_type_store.open_function_context('idzp_aid', 1278, 0, False)
    
    # Passed parameters checking function
    idzp_aid.stypy_localization = localization
    idzp_aid.stypy_type_of_self = None
    idzp_aid.stypy_type_store = module_type_store
    idzp_aid.stypy_function_name = 'idzp_aid'
    idzp_aid.stypy_param_names_list = ['eps', 'A']
    idzp_aid.stypy_varargs_param_name = None
    idzp_aid.stypy_kwargs_param_name = None
    idzp_aid.stypy_call_defaults = defaults
    idzp_aid.stypy_call_varargs = varargs
    idzp_aid.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'idzp_aid', ['eps', 'A'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'idzp_aid', localization, ['eps', 'A'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'idzp_aid(...)' code ##################

    str_31829 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1299, (-1)), 'str', '\n    Compute ID of a complex matrix to a specified relative precision using\n    random sampling.\n\n    :param eps:\n        Relative precision.\n    :type eps: float\n    :param A:\n        Matrix.\n    :type A: :class:`numpy.ndarray`\n\n    :return:\n        Rank of ID.\n    :rtype: int\n    :return:\n        Column index array.\n    :rtype: :class:`numpy.ndarray`\n    :return:\n        Interpolation coefficients.\n    :rtype: :class:`numpy.ndarray`\n    ')
    
    # Assigning a Call to a Name (line 1300):
    
    # Assigning a Call to a Name (line 1300):
    
    # Call to asfortranarray(...): (line 1300)
    # Processing the call arguments (line 1300)
    # Getting the type of 'A' (line 1300)
    A_31832 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1300, 26), 'A', False)
    # Processing the call keyword arguments (line 1300)
    kwargs_31833 = {}
    # Getting the type of 'np' (line 1300)
    np_31830 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1300, 8), 'np', False)
    # Obtaining the member 'asfortranarray' of a type (line 1300)
    asfortranarray_31831 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1300, 8), np_31830, 'asfortranarray')
    # Calling asfortranarray(args, kwargs) (line 1300)
    asfortranarray_call_result_31834 = invoke(stypy.reporting.localization.Localization(__file__, 1300, 8), asfortranarray_31831, *[A_31832], **kwargs_31833)
    
    # Assigning a type to the variable 'A' (line 1300)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1300, 4), 'A', asfortranarray_call_result_31834)
    
    # Assigning a Attribute to a Tuple (line 1301):
    
    # Assigning a Subscript to a Name (line 1301):
    
    # Obtaining the type of the subscript
    int_31835 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1301, 4), 'int')
    # Getting the type of 'A' (line 1301)
    A_31836 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1301, 11), 'A')
    # Obtaining the member 'shape' of a type (line 1301)
    shape_31837 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1301, 11), A_31836, 'shape')
    # Obtaining the member '__getitem__' of a type (line 1301)
    getitem___31838 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1301, 4), shape_31837, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1301)
    subscript_call_result_31839 = invoke(stypy.reporting.localization.Localization(__file__, 1301, 4), getitem___31838, int_31835)
    
    # Assigning a type to the variable 'tuple_var_assignment_29726' (line 1301)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1301, 4), 'tuple_var_assignment_29726', subscript_call_result_31839)
    
    # Assigning a Subscript to a Name (line 1301):
    
    # Obtaining the type of the subscript
    int_31840 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1301, 4), 'int')
    # Getting the type of 'A' (line 1301)
    A_31841 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1301, 11), 'A')
    # Obtaining the member 'shape' of a type (line 1301)
    shape_31842 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1301, 11), A_31841, 'shape')
    # Obtaining the member '__getitem__' of a type (line 1301)
    getitem___31843 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1301, 4), shape_31842, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1301)
    subscript_call_result_31844 = invoke(stypy.reporting.localization.Localization(__file__, 1301, 4), getitem___31843, int_31840)
    
    # Assigning a type to the variable 'tuple_var_assignment_29727' (line 1301)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1301, 4), 'tuple_var_assignment_29727', subscript_call_result_31844)
    
    # Assigning a Name to a Name (line 1301):
    # Getting the type of 'tuple_var_assignment_29726' (line 1301)
    tuple_var_assignment_29726_31845 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1301, 4), 'tuple_var_assignment_29726')
    # Assigning a type to the variable 'm' (line 1301)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1301, 4), 'm', tuple_var_assignment_29726_31845)
    
    # Assigning a Name to a Name (line 1301):
    # Getting the type of 'tuple_var_assignment_29727' (line 1301)
    tuple_var_assignment_29727_31846 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1301, 4), 'tuple_var_assignment_29727')
    # Assigning a type to the variable 'n' (line 1301)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1301, 7), 'n', tuple_var_assignment_29727_31846)
    
    # Assigning a Call to a Tuple (line 1302):
    
    # Assigning a Subscript to a Name (line 1302):
    
    # Obtaining the type of the subscript
    int_31847 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1302, 4), 'int')
    
    # Call to idz_frmi(...): (line 1302)
    # Processing the call arguments (line 1302)
    # Getting the type of 'm' (line 1302)
    m_31849 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1302, 21), 'm', False)
    # Processing the call keyword arguments (line 1302)
    kwargs_31850 = {}
    # Getting the type of 'idz_frmi' (line 1302)
    idz_frmi_31848 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1302, 12), 'idz_frmi', False)
    # Calling idz_frmi(args, kwargs) (line 1302)
    idz_frmi_call_result_31851 = invoke(stypy.reporting.localization.Localization(__file__, 1302, 12), idz_frmi_31848, *[m_31849], **kwargs_31850)
    
    # Obtaining the member '__getitem__' of a type (line 1302)
    getitem___31852 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1302, 4), idz_frmi_call_result_31851, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1302)
    subscript_call_result_31853 = invoke(stypy.reporting.localization.Localization(__file__, 1302, 4), getitem___31852, int_31847)
    
    # Assigning a type to the variable 'tuple_var_assignment_29728' (line 1302)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1302, 4), 'tuple_var_assignment_29728', subscript_call_result_31853)
    
    # Assigning a Subscript to a Name (line 1302):
    
    # Obtaining the type of the subscript
    int_31854 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1302, 4), 'int')
    
    # Call to idz_frmi(...): (line 1302)
    # Processing the call arguments (line 1302)
    # Getting the type of 'm' (line 1302)
    m_31856 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1302, 21), 'm', False)
    # Processing the call keyword arguments (line 1302)
    kwargs_31857 = {}
    # Getting the type of 'idz_frmi' (line 1302)
    idz_frmi_31855 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1302, 12), 'idz_frmi', False)
    # Calling idz_frmi(args, kwargs) (line 1302)
    idz_frmi_call_result_31858 = invoke(stypy.reporting.localization.Localization(__file__, 1302, 12), idz_frmi_31855, *[m_31856], **kwargs_31857)
    
    # Obtaining the member '__getitem__' of a type (line 1302)
    getitem___31859 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1302, 4), idz_frmi_call_result_31858, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1302)
    subscript_call_result_31860 = invoke(stypy.reporting.localization.Localization(__file__, 1302, 4), getitem___31859, int_31854)
    
    # Assigning a type to the variable 'tuple_var_assignment_29729' (line 1302)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1302, 4), 'tuple_var_assignment_29729', subscript_call_result_31860)
    
    # Assigning a Name to a Name (line 1302):
    # Getting the type of 'tuple_var_assignment_29728' (line 1302)
    tuple_var_assignment_29728_31861 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1302, 4), 'tuple_var_assignment_29728')
    # Assigning a type to the variable 'n2' (line 1302)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1302, 4), 'n2', tuple_var_assignment_29728_31861)
    
    # Assigning a Name to a Name (line 1302):
    # Getting the type of 'tuple_var_assignment_29729' (line 1302)
    tuple_var_assignment_29729_31862 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1302, 4), 'tuple_var_assignment_29729')
    # Assigning a type to the variable 'w' (line 1302)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1302, 8), 'w', tuple_var_assignment_29729_31862)
    
    # Assigning a Call to a Name (line 1303):
    
    # Assigning a Call to a Name (line 1303):
    
    # Call to empty(...): (line 1303)
    # Processing the call arguments (line 1303)
    # Getting the type of 'n' (line 1303)
    n_31865 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1303, 20), 'n', False)
    int_31866 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1303, 23), 'int')
    # Getting the type of 'n2' (line 1303)
    n2_31867 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1303, 25), 'n2', False)
    # Applying the binary operator '*' (line 1303)
    result_mul_31868 = python_operator(stypy.reporting.localization.Localization(__file__, 1303, 23), '*', int_31866, n2_31867)
    
    int_31869 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1303, 30), 'int')
    # Applying the binary operator '+' (line 1303)
    result_add_31870 = python_operator(stypy.reporting.localization.Localization(__file__, 1303, 23), '+', result_mul_31868, int_31869)
    
    # Applying the binary operator '*' (line 1303)
    result_mul_31871 = python_operator(stypy.reporting.localization.Localization(__file__, 1303, 20), '*', n_31865, result_add_31870)
    
    # Getting the type of 'n2' (line 1303)
    n2_31872 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1303, 35), 'n2', False)
    # Applying the binary operator '+' (line 1303)
    result_add_31873 = python_operator(stypy.reporting.localization.Localization(__file__, 1303, 20), '+', result_mul_31871, n2_31872)
    
    int_31874 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1303, 40), 'int')
    # Applying the binary operator '+' (line 1303)
    result_add_31875 = python_operator(stypy.reporting.localization.Localization(__file__, 1303, 38), '+', result_add_31873, int_31874)
    
    # Processing the call keyword arguments (line 1303)
    str_31876 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1303, 49), 'str', 'complex128')
    keyword_31877 = str_31876
    str_31878 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1303, 69), 'str', 'F')
    keyword_31879 = str_31878
    kwargs_31880 = {'dtype': keyword_31877, 'order': keyword_31879}
    # Getting the type of 'np' (line 1303)
    np_31863 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1303, 11), 'np', False)
    # Obtaining the member 'empty' of a type (line 1303)
    empty_31864 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1303, 11), np_31863, 'empty')
    # Calling empty(args, kwargs) (line 1303)
    empty_call_result_31881 = invoke(stypy.reporting.localization.Localization(__file__, 1303, 11), empty_31864, *[result_add_31875], **kwargs_31880)
    
    # Assigning a type to the variable 'proj' (line 1303)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1303, 4), 'proj', empty_call_result_31881)
    
    # Assigning a Call to a Tuple (line 1304):
    
    # Assigning a Subscript to a Name (line 1304):
    
    # Obtaining the type of the subscript
    int_31882 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1304, 4), 'int')
    
    # Call to idzp_aid(...): (line 1304)
    # Processing the call arguments (line 1304)
    # Getting the type of 'eps' (line 1304)
    eps_31885 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1304, 32), 'eps', False)
    # Getting the type of 'A' (line 1304)
    A_31886 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1304, 37), 'A', False)
    # Getting the type of 'w' (line 1304)
    w_31887 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1304, 40), 'w', False)
    # Getting the type of 'proj' (line 1304)
    proj_31888 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1304, 43), 'proj', False)
    # Processing the call keyword arguments (line 1304)
    kwargs_31889 = {}
    # Getting the type of '_id' (line 1304)
    _id_31883 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1304, 19), '_id', False)
    # Obtaining the member 'idzp_aid' of a type (line 1304)
    idzp_aid_31884 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1304, 19), _id_31883, 'idzp_aid')
    # Calling idzp_aid(args, kwargs) (line 1304)
    idzp_aid_call_result_31890 = invoke(stypy.reporting.localization.Localization(__file__, 1304, 19), idzp_aid_31884, *[eps_31885, A_31886, w_31887, proj_31888], **kwargs_31889)
    
    # Obtaining the member '__getitem__' of a type (line 1304)
    getitem___31891 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1304, 4), idzp_aid_call_result_31890, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1304)
    subscript_call_result_31892 = invoke(stypy.reporting.localization.Localization(__file__, 1304, 4), getitem___31891, int_31882)
    
    # Assigning a type to the variable 'tuple_var_assignment_29730' (line 1304)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1304, 4), 'tuple_var_assignment_29730', subscript_call_result_31892)
    
    # Assigning a Subscript to a Name (line 1304):
    
    # Obtaining the type of the subscript
    int_31893 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1304, 4), 'int')
    
    # Call to idzp_aid(...): (line 1304)
    # Processing the call arguments (line 1304)
    # Getting the type of 'eps' (line 1304)
    eps_31896 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1304, 32), 'eps', False)
    # Getting the type of 'A' (line 1304)
    A_31897 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1304, 37), 'A', False)
    # Getting the type of 'w' (line 1304)
    w_31898 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1304, 40), 'w', False)
    # Getting the type of 'proj' (line 1304)
    proj_31899 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1304, 43), 'proj', False)
    # Processing the call keyword arguments (line 1304)
    kwargs_31900 = {}
    # Getting the type of '_id' (line 1304)
    _id_31894 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1304, 19), '_id', False)
    # Obtaining the member 'idzp_aid' of a type (line 1304)
    idzp_aid_31895 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1304, 19), _id_31894, 'idzp_aid')
    # Calling idzp_aid(args, kwargs) (line 1304)
    idzp_aid_call_result_31901 = invoke(stypy.reporting.localization.Localization(__file__, 1304, 19), idzp_aid_31895, *[eps_31896, A_31897, w_31898, proj_31899], **kwargs_31900)
    
    # Obtaining the member '__getitem__' of a type (line 1304)
    getitem___31902 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1304, 4), idzp_aid_call_result_31901, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1304)
    subscript_call_result_31903 = invoke(stypy.reporting.localization.Localization(__file__, 1304, 4), getitem___31902, int_31893)
    
    # Assigning a type to the variable 'tuple_var_assignment_29731' (line 1304)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1304, 4), 'tuple_var_assignment_29731', subscript_call_result_31903)
    
    # Assigning a Subscript to a Name (line 1304):
    
    # Obtaining the type of the subscript
    int_31904 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1304, 4), 'int')
    
    # Call to idzp_aid(...): (line 1304)
    # Processing the call arguments (line 1304)
    # Getting the type of 'eps' (line 1304)
    eps_31907 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1304, 32), 'eps', False)
    # Getting the type of 'A' (line 1304)
    A_31908 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1304, 37), 'A', False)
    # Getting the type of 'w' (line 1304)
    w_31909 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1304, 40), 'w', False)
    # Getting the type of 'proj' (line 1304)
    proj_31910 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1304, 43), 'proj', False)
    # Processing the call keyword arguments (line 1304)
    kwargs_31911 = {}
    # Getting the type of '_id' (line 1304)
    _id_31905 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1304, 19), '_id', False)
    # Obtaining the member 'idzp_aid' of a type (line 1304)
    idzp_aid_31906 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1304, 19), _id_31905, 'idzp_aid')
    # Calling idzp_aid(args, kwargs) (line 1304)
    idzp_aid_call_result_31912 = invoke(stypy.reporting.localization.Localization(__file__, 1304, 19), idzp_aid_31906, *[eps_31907, A_31908, w_31909, proj_31910], **kwargs_31911)
    
    # Obtaining the member '__getitem__' of a type (line 1304)
    getitem___31913 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1304, 4), idzp_aid_call_result_31912, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1304)
    subscript_call_result_31914 = invoke(stypy.reporting.localization.Localization(__file__, 1304, 4), getitem___31913, int_31904)
    
    # Assigning a type to the variable 'tuple_var_assignment_29732' (line 1304)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1304, 4), 'tuple_var_assignment_29732', subscript_call_result_31914)
    
    # Assigning a Name to a Name (line 1304):
    # Getting the type of 'tuple_var_assignment_29730' (line 1304)
    tuple_var_assignment_29730_31915 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1304, 4), 'tuple_var_assignment_29730')
    # Assigning a type to the variable 'k' (line 1304)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1304, 4), 'k', tuple_var_assignment_29730_31915)
    
    # Assigning a Name to a Name (line 1304):
    # Getting the type of 'tuple_var_assignment_29731' (line 1304)
    tuple_var_assignment_29731_31916 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1304, 4), 'tuple_var_assignment_29731')
    # Assigning a type to the variable 'idx' (line 1304)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1304, 7), 'idx', tuple_var_assignment_29731_31916)
    
    # Assigning a Name to a Name (line 1304):
    # Getting the type of 'tuple_var_assignment_29732' (line 1304)
    tuple_var_assignment_29732_31917 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1304, 4), 'tuple_var_assignment_29732')
    # Assigning a type to the variable 'proj' (line 1304)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1304, 12), 'proj', tuple_var_assignment_29732_31917)
    
    # Assigning a Call to a Name (line 1305):
    
    # Assigning a Call to a Name (line 1305):
    
    # Call to reshape(...): (line 1305)
    # Processing the call arguments (line 1305)
    
    # Obtaining an instance of the builtin type 'tuple' (line 1305)
    tuple_31928 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1305, 35), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 1305)
    # Adding element type (line 1305)
    # Getting the type of 'k' (line 1305)
    k_31929 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1305, 35), 'k', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1305, 35), tuple_31928, k_31929)
    # Adding element type (line 1305)
    # Getting the type of 'n' (line 1305)
    n_31930 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1305, 38), 'n', False)
    # Getting the type of 'k' (line 1305)
    k_31931 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1305, 40), 'k', False)
    # Applying the binary operator '-' (line 1305)
    result_sub_31932 = python_operator(stypy.reporting.localization.Localization(__file__, 1305, 38), '-', n_31930, k_31931)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1305, 35), tuple_31928, result_sub_31932)
    
    # Processing the call keyword arguments (line 1305)
    str_31933 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1305, 50), 'str', 'F')
    keyword_31934 = str_31933
    kwargs_31935 = {'order': keyword_31934}
    
    # Obtaining the type of the subscript
    # Getting the type of 'k' (line 1305)
    k_31918 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1305, 17), 'k', False)
    # Getting the type of 'n' (line 1305)
    n_31919 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1305, 20), 'n', False)
    # Getting the type of 'k' (line 1305)
    k_31920 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1305, 22), 'k', False)
    # Applying the binary operator '-' (line 1305)
    result_sub_31921 = python_operator(stypy.reporting.localization.Localization(__file__, 1305, 20), '-', n_31919, k_31920)
    
    # Applying the binary operator '*' (line 1305)
    result_mul_31922 = python_operator(stypy.reporting.localization.Localization(__file__, 1305, 17), '*', k_31918, result_sub_31921)
    
    slice_31923 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 1305, 11), None, result_mul_31922, None)
    # Getting the type of 'proj' (line 1305)
    proj_31924 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1305, 11), 'proj', False)
    # Obtaining the member '__getitem__' of a type (line 1305)
    getitem___31925 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1305, 11), proj_31924, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1305)
    subscript_call_result_31926 = invoke(stypy.reporting.localization.Localization(__file__, 1305, 11), getitem___31925, slice_31923)
    
    # Obtaining the member 'reshape' of a type (line 1305)
    reshape_31927 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1305, 11), subscript_call_result_31926, 'reshape')
    # Calling reshape(args, kwargs) (line 1305)
    reshape_call_result_31936 = invoke(stypy.reporting.localization.Localization(__file__, 1305, 11), reshape_31927, *[tuple_31928], **kwargs_31935)
    
    # Assigning a type to the variable 'proj' (line 1305)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1305, 4), 'proj', reshape_call_result_31936)
    
    # Obtaining an instance of the builtin type 'tuple' (line 1306)
    tuple_31937 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1306, 11), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 1306)
    # Adding element type (line 1306)
    # Getting the type of 'k' (line 1306)
    k_31938 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1306, 11), 'k')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1306, 11), tuple_31937, k_31938)
    # Adding element type (line 1306)
    # Getting the type of 'idx' (line 1306)
    idx_31939 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1306, 14), 'idx')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1306, 11), tuple_31937, idx_31939)
    # Adding element type (line 1306)
    # Getting the type of 'proj' (line 1306)
    proj_31940 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1306, 19), 'proj')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1306, 11), tuple_31937, proj_31940)
    
    # Assigning a type to the variable 'stypy_return_type' (line 1306)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1306, 4), 'stypy_return_type', tuple_31937)
    
    # ################# End of 'idzp_aid(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'idzp_aid' in the type store
    # Getting the type of 'stypy_return_type' (line 1278)
    stypy_return_type_31941 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1278, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_31941)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'idzp_aid'
    return stypy_return_type_31941

# Assigning a type to the variable 'idzp_aid' (line 1278)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1278, 0), 'idzp_aid', idzp_aid)

@norecursion
def idz_estrank(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'idz_estrank'
    module_type_store = module_type_store.open_function_context('idz_estrank', 1309, 0, False)
    
    # Passed parameters checking function
    idz_estrank.stypy_localization = localization
    idz_estrank.stypy_type_of_self = None
    idz_estrank.stypy_type_store = module_type_store
    idz_estrank.stypy_function_name = 'idz_estrank'
    idz_estrank.stypy_param_names_list = ['eps', 'A']
    idz_estrank.stypy_varargs_param_name = None
    idz_estrank.stypy_kwargs_param_name = None
    idz_estrank.stypy_call_defaults = defaults
    idz_estrank.stypy_call_varargs = varargs
    idz_estrank.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'idz_estrank', ['eps', 'A'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'idz_estrank', localization, ['eps', 'A'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'idz_estrank(...)' code ##################

    str_31942 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1326, (-1)), 'str', '\n    Estimate rank of a complex matrix to a specified relative precision using\n    random sampling.\n\n    The output rank is typically about 8 higher than the actual rank.\n\n    :param eps:\n        Relative precision.\n    :type eps: float\n    :param A:\n        Matrix.\n    :type A: :class:`numpy.ndarray`\n\n    :return:\n        Rank estimate.\n    :rtype: int\n    ')
    
    # Assigning a Call to a Name (line 1327):
    
    # Assigning a Call to a Name (line 1327):
    
    # Call to asfortranarray(...): (line 1327)
    # Processing the call arguments (line 1327)
    # Getting the type of 'A' (line 1327)
    A_31945 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1327, 26), 'A', False)
    # Processing the call keyword arguments (line 1327)
    kwargs_31946 = {}
    # Getting the type of 'np' (line 1327)
    np_31943 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1327, 8), 'np', False)
    # Obtaining the member 'asfortranarray' of a type (line 1327)
    asfortranarray_31944 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1327, 8), np_31943, 'asfortranarray')
    # Calling asfortranarray(args, kwargs) (line 1327)
    asfortranarray_call_result_31947 = invoke(stypy.reporting.localization.Localization(__file__, 1327, 8), asfortranarray_31944, *[A_31945], **kwargs_31946)
    
    # Assigning a type to the variable 'A' (line 1327)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1327, 4), 'A', asfortranarray_call_result_31947)
    
    # Assigning a Attribute to a Tuple (line 1328):
    
    # Assigning a Subscript to a Name (line 1328):
    
    # Obtaining the type of the subscript
    int_31948 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1328, 4), 'int')
    # Getting the type of 'A' (line 1328)
    A_31949 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1328, 11), 'A')
    # Obtaining the member 'shape' of a type (line 1328)
    shape_31950 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1328, 11), A_31949, 'shape')
    # Obtaining the member '__getitem__' of a type (line 1328)
    getitem___31951 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1328, 4), shape_31950, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1328)
    subscript_call_result_31952 = invoke(stypy.reporting.localization.Localization(__file__, 1328, 4), getitem___31951, int_31948)
    
    # Assigning a type to the variable 'tuple_var_assignment_29733' (line 1328)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1328, 4), 'tuple_var_assignment_29733', subscript_call_result_31952)
    
    # Assigning a Subscript to a Name (line 1328):
    
    # Obtaining the type of the subscript
    int_31953 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1328, 4), 'int')
    # Getting the type of 'A' (line 1328)
    A_31954 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1328, 11), 'A')
    # Obtaining the member 'shape' of a type (line 1328)
    shape_31955 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1328, 11), A_31954, 'shape')
    # Obtaining the member '__getitem__' of a type (line 1328)
    getitem___31956 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1328, 4), shape_31955, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1328)
    subscript_call_result_31957 = invoke(stypy.reporting.localization.Localization(__file__, 1328, 4), getitem___31956, int_31953)
    
    # Assigning a type to the variable 'tuple_var_assignment_29734' (line 1328)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1328, 4), 'tuple_var_assignment_29734', subscript_call_result_31957)
    
    # Assigning a Name to a Name (line 1328):
    # Getting the type of 'tuple_var_assignment_29733' (line 1328)
    tuple_var_assignment_29733_31958 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1328, 4), 'tuple_var_assignment_29733')
    # Assigning a type to the variable 'm' (line 1328)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1328, 4), 'm', tuple_var_assignment_29733_31958)
    
    # Assigning a Name to a Name (line 1328):
    # Getting the type of 'tuple_var_assignment_29734' (line 1328)
    tuple_var_assignment_29734_31959 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1328, 4), 'tuple_var_assignment_29734')
    # Assigning a type to the variable 'n' (line 1328)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1328, 7), 'n', tuple_var_assignment_29734_31959)
    
    # Assigning a Call to a Tuple (line 1329):
    
    # Assigning a Subscript to a Name (line 1329):
    
    # Obtaining the type of the subscript
    int_31960 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1329, 4), 'int')
    
    # Call to idz_frmi(...): (line 1329)
    # Processing the call arguments (line 1329)
    # Getting the type of 'm' (line 1329)
    m_31962 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1329, 21), 'm', False)
    # Processing the call keyword arguments (line 1329)
    kwargs_31963 = {}
    # Getting the type of 'idz_frmi' (line 1329)
    idz_frmi_31961 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1329, 12), 'idz_frmi', False)
    # Calling idz_frmi(args, kwargs) (line 1329)
    idz_frmi_call_result_31964 = invoke(stypy.reporting.localization.Localization(__file__, 1329, 12), idz_frmi_31961, *[m_31962], **kwargs_31963)
    
    # Obtaining the member '__getitem__' of a type (line 1329)
    getitem___31965 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1329, 4), idz_frmi_call_result_31964, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1329)
    subscript_call_result_31966 = invoke(stypy.reporting.localization.Localization(__file__, 1329, 4), getitem___31965, int_31960)
    
    # Assigning a type to the variable 'tuple_var_assignment_29735' (line 1329)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1329, 4), 'tuple_var_assignment_29735', subscript_call_result_31966)
    
    # Assigning a Subscript to a Name (line 1329):
    
    # Obtaining the type of the subscript
    int_31967 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1329, 4), 'int')
    
    # Call to idz_frmi(...): (line 1329)
    # Processing the call arguments (line 1329)
    # Getting the type of 'm' (line 1329)
    m_31969 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1329, 21), 'm', False)
    # Processing the call keyword arguments (line 1329)
    kwargs_31970 = {}
    # Getting the type of 'idz_frmi' (line 1329)
    idz_frmi_31968 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1329, 12), 'idz_frmi', False)
    # Calling idz_frmi(args, kwargs) (line 1329)
    idz_frmi_call_result_31971 = invoke(stypy.reporting.localization.Localization(__file__, 1329, 12), idz_frmi_31968, *[m_31969], **kwargs_31970)
    
    # Obtaining the member '__getitem__' of a type (line 1329)
    getitem___31972 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1329, 4), idz_frmi_call_result_31971, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1329)
    subscript_call_result_31973 = invoke(stypy.reporting.localization.Localization(__file__, 1329, 4), getitem___31972, int_31967)
    
    # Assigning a type to the variable 'tuple_var_assignment_29736' (line 1329)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1329, 4), 'tuple_var_assignment_29736', subscript_call_result_31973)
    
    # Assigning a Name to a Name (line 1329):
    # Getting the type of 'tuple_var_assignment_29735' (line 1329)
    tuple_var_assignment_29735_31974 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1329, 4), 'tuple_var_assignment_29735')
    # Assigning a type to the variable 'n2' (line 1329)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1329, 4), 'n2', tuple_var_assignment_29735_31974)
    
    # Assigning a Name to a Name (line 1329):
    # Getting the type of 'tuple_var_assignment_29736' (line 1329)
    tuple_var_assignment_29736_31975 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1329, 4), 'tuple_var_assignment_29736')
    # Assigning a type to the variable 'w' (line 1329)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1329, 8), 'w', tuple_var_assignment_29736_31975)
    
    # Assigning a Call to a Name (line 1330):
    
    # Assigning a Call to a Name (line 1330):
    
    # Call to empty(...): (line 1330)
    # Processing the call arguments (line 1330)
    # Getting the type of 'n' (line 1330)
    n_31978 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1330, 18), 'n', False)
    # Getting the type of 'n2' (line 1330)
    n2_31979 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1330, 20), 'n2', False)
    # Applying the binary operator '*' (line 1330)
    result_mul_31980 = python_operator(stypy.reporting.localization.Localization(__file__, 1330, 18), '*', n_31978, n2_31979)
    
    # Getting the type of 'n' (line 1330)
    n_31981 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1330, 26), 'n', False)
    int_31982 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1330, 30), 'int')
    # Applying the binary operator '+' (line 1330)
    result_add_31983 = python_operator(stypy.reporting.localization.Localization(__file__, 1330, 26), '+', n_31981, int_31982)
    
    # Getting the type of 'n2' (line 1330)
    n2_31984 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1330, 34), 'n2', False)
    int_31985 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1330, 39), 'int')
    # Applying the binary operator '+' (line 1330)
    result_add_31986 = python_operator(stypy.reporting.localization.Localization(__file__, 1330, 34), '+', n2_31984, int_31985)
    
    # Applying the binary operator '*' (line 1330)
    result_mul_31987 = python_operator(stypy.reporting.localization.Localization(__file__, 1330, 25), '*', result_add_31983, result_add_31986)
    
    # Applying the binary operator '+' (line 1330)
    result_add_31988 = python_operator(stypy.reporting.localization.Localization(__file__, 1330, 18), '+', result_mul_31980, result_mul_31987)
    
    # Processing the call keyword arguments (line 1330)
    str_31989 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1330, 49), 'str', 'complex128')
    keyword_31990 = str_31989
    str_31991 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1330, 69), 'str', 'F')
    keyword_31992 = str_31991
    kwargs_31993 = {'dtype': keyword_31990, 'order': keyword_31992}
    # Getting the type of 'np' (line 1330)
    np_31976 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1330, 9), 'np', False)
    # Obtaining the member 'empty' of a type (line 1330)
    empty_31977 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1330, 9), np_31976, 'empty')
    # Calling empty(args, kwargs) (line 1330)
    empty_call_result_31994 = invoke(stypy.reporting.localization.Localization(__file__, 1330, 9), empty_31977, *[result_add_31988], **kwargs_31993)
    
    # Assigning a type to the variable 'ra' (line 1330)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1330, 4), 'ra', empty_call_result_31994)
    
    # Assigning a Call to a Tuple (line 1331):
    
    # Assigning a Subscript to a Name (line 1331):
    
    # Obtaining the type of the subscript
    int_31995 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1331, 4), 'int')
    
    # Call to idz_estrank(...): (line 1331)
    # Processing the call arguments (line 1331)
    # Getting the type of 'eps' (line 1331)
    eps_31998 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1331, 28), 'eps', False)
    # Getting the type of 'A' (line 1331)
    A_31999 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1331, 33), 'A', False)
    # Getting the type of 'w' (line 1331)
    w_32000 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1331, 36), 'w', False)
    # Getting the type of 'ra' (line 1331)
    ra_32001 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1331, 39), 'ra', False)
    # Processing the call keyword arguments (line 1331)
    kwargs_32002 = {}
    # Getting the type of '_id' (line 1331)
    _id_31996 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1331, 12), '_id', False)
    # Obtaining the member 'idz_estrank' of a type (line 1331)
    idz_estrank_31997 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1331, 12), _id_31996, 'idz_estrank')
    # Calling idz_estrank(args, kwargs) (line 1331)
    idz_estrank_call_result_32003 = invoke(stypy.reporting.localization.Localization(__file__, 1331, 12), idz_estrank_31997, *[eps_31998, A_31999, w_32000, ra_32001], **kwargs_32002)
    
    # Obtaining the member '__getitem__' of a type (line 1331)
    getitem___32004 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1331, 4), idz_estrank_call_result_32003, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1331)
    subscript_call_result_32005 = invoke(stypy.reporting.localization.Localization(__file__, 1331, 4), getitem___32004, int_31995)
    
    # Assigning a type to the variable 'tuple_var_assignment_29737' (line 1331)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1331, 4), 'tuple_var_assignment_29737', subscript_call_result_32005)
    
    # Assigning a Subscript to a Name (line 1331):
    
    # Obtaining the type of the subscript
    int_32006 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1331, 4), 'int')
    
    # Call to idz_estrank(...): (line 1331)
    # Processing the call arguments (line 1331)
    # Getting the type of 'eps' (line 1331)
    eps_32009 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1331, 28), 'eps', False)
    # Getting the type of 'A' (line 1331)
    A_32010 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1331, 33), 'A', False)
    # Getting the type of 'w' (line 1331)
    w_32011 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1331, 36), 'w', False)
    # Getting the type of 'ra' (line 1331)
    ra_32012 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1331, 39), 'ra', False)
    # Processing the call keyword arguments (line 1331)
    kwargs_32013 = {}
    # Getting the type of '_id' (line 1331)
    _id_32007 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1331, 12), '_id', False)
    # Obtaining the member 'idz_estrank' of a type (line 1331)
    idz_estrank_32008 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1331, 12), _id_32007, 'idz_estrank')
    # Calling idz_estrank(args, kwargs) (line 1331)
    idz_estrank_call_result_32014 = invoke(stypy.reporting.localization.Localization(__file__, 1331, 12), idz_estrank_32008, *[eps_32009, A_32010, w_32011, ra_32012], **kwargs_32013)
    
    # Obtaining the member '__getitem__' of a type (line 1331)
    getitem___32015 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1331, 4), idz_estrank_call_result_32014, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1331)
    subscript_call_result_32016 = invoke(stypy.reporting.localization.Localization(__file__, 1331, 4), getitem___32015, int_32006)
    
    # Assigning a type to the variable 'tuple_var_assignment_29738' (line 1331)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1331, 4), 'tuple_var_assignment_29738', subscript_call_result_32016)
    
    # Assigning a Name to a Name (line 1331):
    # Getting the type of 'tuple_var_assignment_29737' (line 1331)
    tuple_var_assignment_29737_32017 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1331, 4), 'tuple_var_assignment_29737')
    # Assigning a type to the variable 'k' (line 1331)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1331, 4), 'k', tuple_var_assignment_29737_32017)
    
    # Assigning a Name to a Name (line 1331):
    # Getting the type of 'tuple_var_assignment_29738' (line 1331)
    tuple_var_assignment_29738_32018 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1331, 4), 'tuple_var_assignment_29738')
    # Assigning a type to the variable 'ra' (line 1331)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1331, 7), 'ra', tuple_var_assignment_29738_32018)
    # Getting the type of 'k' (line 1332)
    k_32019 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1332, 11), 'k')
    # Assigning a type to the variable 'stypy_return_type' (line 1332)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1332, 4), 'stypy_return_type', k_32019)
    
    # ################# End of 'idz_estrank(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'idz_estrank' in the type store
    # Getting the type of 'stypy_return_type' (line 1309)
    stypy_return_type_32020 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1309, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_32020)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'idz_estrank'
    return stypy_return_type_32020

# Assigning a type to the variable 'idz_estrank' (line 1309)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1309, 0), 'idz_estrank', idz_estrank)

@norecursion
def idzp_asvd(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'idzp_asvd'
    module_type_store = module_type_store.open_function_context('idzp_asvd', 1339, 0, False)
    
    # Passed parameters checking function
    idzp_asvd.stypy_localization = localization
    idzp_asvd.stypy_type_of_self = None
    idzp_asvd.stypy_type_store = module_type_store
    idzp_asvd.stypy_function_name = 'idzp_asvd'
    idzp_asvd.stypy_param_names_list = ['eps', 'A']
    idzp_asvd.stypy_varargs_param_name = None
    idzp_asvd.stypy_kwargs_param_name = None
    idzp_asvd.stypy_call_defaults = defaults
    idzp_asvd.stypy_call_varargs = varargs
    idzp_asvd.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'idzp_asvd', ['eps', 'A'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'idzp_asvd', localization, ['eps', 'A'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'idzp_asvd(...)' code ##################

    str_32021 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1360, (-1)), 'str', '\n    Compute SVD of a complex matrix to a specified relative precision using\n    random sampling.\n\n    :param eps:\n        Relative precision.\n    :type eps: float\n    :param A:\n        Matrix.\n    :type A: :class:`numpy.ndarray`\n\n    :return:\n        Left singular vectors.\n    :rtype: :class:`numpy.ndarray`\n    :return:\n        Right singular vectors.\n    :rtype: :class:`numpy.ndarray`\n    :return:\n        Singular values.\n    :rtype: :class:`numpy.ndarray`\n    ')
    
    # Assigning a Call to a Name (line 1361):
    
    # Assigning a Call to a Name (line 1361):
    
    # Call to asfortranarray(...): (line 1361)
    # Processing the call arguments (line 1361)
    # Getting the type of 'A' (line 1361)
    A_32024 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1361, 26), 'A', False)
    # Processing the call keyword arguments (line 1361)
    kwargs_32025 = {}
    # Getting the type of 'np' (line 1361)
    np_32022 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1361, 8), 'np', False)
    # Obtaining the member 'asfortranarray' of a type (line 1361)
    asfortranarray_32023 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1361, 8), np_32022, 'asfortranarray')
    # Calling asfortranarray(args, kwargs) (line 1361)
    asfortranarray_call_result_32026 = invoke(stypy.reporting.localization.Localization(__file__, 1361, 8), asfortranarray_32023, *[A_32024], **kwargs_32025)
    
    # Assigning a type to the variable 'A' (line 1361)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1361, 4), 'A', asfortranarray_call_result_32026)
    
    # Assigning a Attribute to a Tuple (line 1362):
    
    # Assigning a Subscript to a Name (line 1362):
    
    # Obtaining the type of the subscript
    int_32027 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1362, 4), 'int')
    # Getting the type of 'A' (line 1362)
    A_32028 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1362, 11), 'A')
    # Obtaining the member 'shape' of a type (line 1362)
    shape_32029 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1362, 11), A_32028, 'shape')
    # Obtaining the member '__getitem__' of a type (line 1362)
    getitem___32030 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1362, 4), shape_32029, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1362)
    subscript_call_result_32031 = invoke(stypy.reporting.localization.Localization(__file__, 1362, 4), getitem___32030, int_32027)
    
    # Assigning a type to the variable 'tuple_var_assignment_29739' (line 1362)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1362, 4), 'tuple_var_assignment_29739', subscript_call_result_32031)
    
    # Assigning a Subscript to a Name (line 1362):
    
    # Obtaining the type of the subscript
    int_32032 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1362, 4), 'int')
    # Getting the type of 'A' (line 1362)
    A_32033 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1362, 11), 'A')
    # Obtaining the member 'shape' of a type (line 1362)
    shape_32034 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1362, 11), A_32033, 'shape')
    # Obtaining the member '__getitem__' of a type (line 1362)
    getitem___32035 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1362, 4), shape_32034, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1362)
    subscript_call_result_32036 = invoke(stypy.reporting.localization.Localization(__file__, 1362, 4), getitem___32035, int_32032)
    
    # Assigning a type to the variable 'tuple_var_assignment_29740' (line 1362)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1362, 4), 'tuple_var_assignment_29740', subscript_call_result_32036)
    
    # Assigning a Name to a Name (line 1362):
    # Getting the type of 'tuple_var_assignment_29739' (line 1362)
    tuple_var_assignment_29739_32037 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1362, 4), 'tuple_var_assignment_29739')
    # Assigning a type to the variable 'm' (line 1362)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1362, 4), 'm', tuple_var_assignment_29739_32037)
    
    # Assigning a Name to a Name (line 1362):
    # Getting the type of 'tuple_var_assignment_29740' (line 1362)
    tuple_var_assignment_29740_32038 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1362, 4), 'tuple_var_assignment_29740')
    # Assigning a type to the variable 'n' (line 1362)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1362, 7), 'n', tuple_var_assignment_29740_32038)
    
    # Assigning a Call to a Tuple (line 1363):
    
    # Assigning a Subscript to a Name (line 1363):
    
    # Obtaining the type of the subscript
    int_32039 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1363, 4), 'int')
    
    # Call to idz_frmi(...): (line 1363)
    # Processing the call arguments (line 1363)
    # Getting the type of 'm' (line 1363)
    m_32042 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1363, 29), 'm', False)
    # Processing the call keyword arguments (line 1363)
    kwargs_32043 = {}
    # Getting the type of '_id' (line 1363)
    _id_32040 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1363, 16), '_id', False)
    # Obtaining the member 'idz_frmi' of a type (line 1363)
    idz_frmi_32041 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1363, 16), _id_32040, 'idz_frmi')
    # Calling idz_frmi(args, kwargs) (line 1363)
    idz_frmi_call_result_32044 = invoke(stypy.reporting.localization.Localization(__file__, 1363, 16), idz_frmi_32041, *[m_32042], **kwargs_32043)
    
    # Obtaining the member '__getitem__' of a type (line 1363)
    getitem___32045 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1363, 4), idz_frmi_call_result_32044, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1363)
    subscript_call_result_32046 = invoke(stypy.reporting.localization.Localization(__file__, 1363, 4), getitem___32045, int_32039)
    
    # Assigning a type to the variable 'tuple_var_assignment_29741' (line 1363)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1363, 4), 'tuple_var_assignment_29741', subscript_call_result_32046)
    
    # Assigning a Subscript to a Name (line 1363):
    
    # Obtaining the type of the subscript
    int_32047 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1363, 4), 'int')
    
    # Call to idz_frmi(...): (line 1363)
    # Processing the call arguments (line 1363)
    # Getting the type of 'm' (line 1363)
    m_32050 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1363, 29), 'm', False)
    # Processing the call keyword arguments (line 1363)
    kwargs_32051 = {}
    # Getting the type of '_id' (line 1363)
    _id_32048 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1363, 16), '_id', False)
    # Obtaining the member 'idz_frmi' of a type (line 1363)
    idz_frmi_32049 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1363, 16), _id_32048, 'idz_frmi')
    # Calling idz_frmi(args, kwargs) (line 1363)
    idz_frmi_call_result_32052 = invoke(stypy.reporting.localization.Localization(__file__, 1363, 16), idz_frmi_32049, *[m_32050], **kwargs_32051)
    
    # Obtaining the member '__getitem__' of a type (line 1363)
    getitem___32053 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1363, 4), idz_frmi_call_result_32052, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1363)
    subscript_call_result_32054 = invoke(stypy.reporting.localization.Localization(__file__, 1363, 4), getitem___32053, int_32047)
    
    # Assigning a type to the variable 'tuple_var_assignment_29742' (line 1363)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1363, 4), 'tuple_var_assignment_29742', subscript_call_result_32054)
    
    # Assigning a Name to a Name (line 1363):
    # Getting the type of 'tuple_var_assignment_29741' (line 1363)
    tuple_var_assignment_29741_32055 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1363, 4), 'tuple_var_assignment_29741')
    # Assigning a type to the variable 'n2' (line 1363)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1363, 4), 'n2', tuple_var_assignment_29741_32055)
    
    # Assigning a Name to a Name (line 1363):
    # Getting the type of 'tuple_var_assignment_29742' (line 1363)
    tuple_var_assignment_29742_32056 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1363, 4), 'tuple_var_assignment_29742')
    # Assigning a type to the variable 'winit' (line 1363)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1363, 8), 'winit', tuple_var_assignment_29742_32056)
    
    # Assigning a Call to a Name (line 1364):
    
    # Assigning a Call to a Name (line 1364):
    
    # Call to empty(...): (line 1364)
    # Processing the call arguments (line 1364)
    
    # Call to max(...): (line 1365)
    # Processing the call arguments (line 1365)
    
    # Call to min(...): (line 1365)
    # Processing the call arguments (line 1365)
    # Getting the type of 'm' (line 1365)
    m_32061 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1365, 17), 'm', False)
    # Getting the type of 'n' (line 1365)
    n_32062 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1365, 20), 'n', False)
    # Processing the call keyword arguments (line 1365)
    kwargs_32063 = {}
    # Getting the type of 'min' (line 1365)
    min_32060 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1365, 13), 'min', False)
    # Calling min(args, kwargs) (line 1365)
    min_call_result_32064 = invoke(stypy.reporting.localization.Localization(__file__, 1365, 13), min_32060, *[m_32061, n_32062], **kwargs_32063)
    
    int_32065 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1365, 25), 'int')
    # Applying the binary operator '+' (line 1365)
    result_add_32066 = python_operator(stypy.reporting.localization.Localization(__file__, 1365, 13), '+', min_call_result_32064, int_32065)
    
    int_32067 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1365, 29), 'int')
    # Getting the type of 'm' (line 1365)
    m_32068 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1365, 31), 'm', False)
    # Applying the binary operator '*' (line 1365)
    result_mul_32069 = python_operator(stypy.reporting.localization.Localization(__file__, 1365, 29), '*', int_32067, m_32068)
    
    int_32070 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1365, 35), 'int')
    # Getting the type of 'n' (line 1365)
    n_32071 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1365, 37), 'n', False)
    # Applying the binary operator '*' (line 1365)
    result_mul_32072 = python_operator(stypy.reporting.localization.Localization(__file__, 1365, 35), '*', int_32070, n_32071)
    
    # Applying the binary operator '+' (line 1365)
    result_add_32073 = python_operator(stypy.reporting.localization.Localization(__file__, 1365, 29), '+', result_mul_32069, result_mul_32072)
    
    int_32074 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1365, 41), 'int')
    # Applying the binary operator '+' (line 1365)
    result_add_32075 = python_operator(stypy.reporting.localization.Localization(__file__, 1365, 39), '+', result_add_32073, int_32074)
    
    # Applying the binary operator '*' (line 1365)
    result_mul_32076 = python_operator(stypy.reporting.localization.Localization(__file__, 1365, 12), '*', result_add_32066, result_add_32075)
    
    int_32077 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1365, 47), 'int')
    
    # Call to min(...): (line 1365)
    # Processing the call arguments (line 1365)
    # Getting the type of 'm' (line 1365)
    m_32079 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1365, 53), 'm', False)
    # Getting the type of 'n' (line 1365)
    n_32080 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1365, 56), 'n', False)
    # Processing the call keyword arguments (line 1365)
    kwargs_32081 = {}
    # Getting the type of 'min' (line 1365)
    min_32078 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1365, 49), 'min', False)
    # Calling min(args, kwargs) (line 1365)
    min_call_result_32082 = invoke(stypy.reporting.localization.Localization(__file__, 1365, 49), min_32078, *[m_32079, n_32080], **kwargs_32081)
    
    int_32083 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1365, 60), 'int')
    # Applying the binary operator '**' (line 1365)
    result_pow_32084 = python_operator(stypy.reporting.localization.Localization(__file__, 1365, 49), '**', min_call_result_32082, int_32083)
    
    # Applying the binary operator '*' (line 1365)
    result_mul_32085 = python_operator(stypy.reporting.localization.Localization(__file__, 1365, 47), '*', int_32077, result_pow_32084)
    
    # Applying the binary operator '+' (line 1365)
    result_add_32086 = python_operator(stypy.reporting.localization.Localization(__file__, 1365, 12), '+', result_mul_32076, result_mul_32085)
    
    int_32087 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1366, 13), 'int')
    # Getting the type of 'n' (line 1366)
    n_32088 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1366, 15), 'n', False)
    # Applying the binary operator '*' (line 1366)
    result_mul_32089 = python_operator(stypy.reporting.localization.Localization(__file__, 1366, 13), '*', int_32087, n_32088)
    
    int_32090 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1366, 19), 'int')
    # Applying the binary operator '+' (line 1366)
    result_add_32091 = python_operator(stypy.reporting.localization.Localization(__file__, 1366, 13), '+', result_mul_32089, int_32090)
    
    # Getting the type of 'n2' (line 1366)
    n2_32092 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1366, 23), 'n2', False)
    int_32093 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1366, 28), 'int')
    # Applying the binary operator '+' (line 1366)
    result_add_32094 = python_operator(stypy.reporting.localization.Localization(__file__, 1366, 23), '+', n2_32092, int_32093)
    
    # Applying the binary operator '*' (line 1366)
    result_mul_32095 = python_operator(stypy.reporting.localization.Localization(__file__, 1366, 12), '*', result_add_32091, result_add_32094)
    
    # Processing the call keyword arguments (line 1365)
    kwargs_32096 = {}
    # Getting the type of 'max' (line 1365)
    max_32059 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1365, 8), 'max', False)
    # Calling max(args, kwargs) (line 1365)
    max_call_result_32097 = invoke(stypy.reporting.localization.Localization(__file__, 1365, 8), max_32059, *[result_add_32086, result_mul_32095], **kwargs_32096)
    
    # Processing the call keyword arguments (line 1364)
    # Getting the type of 'np' (line 1367)
    np_32098 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1367, 14), 'np', False)
    # Obtaining the member 'complex128' of a type (line 1367)
    complex128_32099 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1367, 14), np_32098, 'complex128')
    keyword_32100 = complex128_32099
    str_32101 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1367, 35), 'str', 'F')
    keyword_32102 = str_32101
    kwargs_32103 = {'dtype': keyword_32100, 'order': keyword_32102}
    # Getting the type of 'np' (line 1364)
    np_32057 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1364, 8), 'np', False)
    # Obtaining the member 'empty' of a type (line 1364)
    empty_32058 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1364, 8), np_32057, 'empty')
    # Calling empty(args, kwargs) (line 1364)
    empty_call_result_32104 = invoke(stypy.reporting.localization.Localization(__file__, 1364, 8), empty_32058, *[max_call_result_32097], **kwargs_32103)
    
    # Assigning a type to the variable 'w' (line 1364)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1364, 4), 'w', empty_call_result_32104)
    
    # Assigning a Call to a Tuple (line 1368):
    
    # Assigning a Subscript to a Name (line 1368):
    
    # Obtaining the type of the subscript
    int_32105 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1368, 4), 'int')
    
    # Call to idzp_asvd(...): (line 1368)
    # Processing the call arguments (line 1368)
    # Getting the type of 'eps' (line 1368)
    eps_32108 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1368, 42), 'eps', False)
    # Getting the type of 'A' (line 1368)
    A_32109 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1368, 47), 'A', False)
    # Getting the type of 'winit' (line 1368)
    winit_32110 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1368, 50), 'winit', False)
    # Getting the type of 'w' (line 1368)
    w_32111 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1368, 57), 'w', False)
    # Processing the call keyword arguments (line 1368)
    kwargs_32112 = {}
    # Getting the type of '_id' (line 1368)
    _id_32106 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1368, 28), '_id', False)
    # Obtaining the member 'idzp_asvd' of a type (line 1368)
    idzp_asvd_32107 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1368, 28), _id_32106, 'idzp_asvd')
    # Calling idzp_asvd(args, kwargs) (line 1368)
    idzp_asvd_call_result_32113 = invoke(stypy.reporting.localization.Localization(__file__, 1368, 28), idzp_asvd_32107, *[eps_32108, A_32109, winit_32110, w_32111], **kwargs_32112)
    
    # Obtaining the member '__getitem__' of a type (line 1368)
    getitem___32114 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1368, 4), idzp_asvd_call_result_32113, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1368)
    subscript_call_result_32115 = invoke(stypy.reporting.localization.Localization(__file__, 1368, 4), getitem___32114, int_32105)
    
    # Assigning a type to the variable 'tuple_var_assignment_29743' (line 1368)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1368, 4), 'tuple_var_assignment_29743', subscript_call_result_32115)
    
    # Assigning a Subscript to a Name (line 1368):
    
    # Obtaining the type of the subscript
    int_32116 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1368, 4), 'int')
    
    # Call to idzp_asvd(...): (line 1368)
    # Processing the call arguments (line 1368)
    # Getting the type of 'eps' (line 1368)
    eps_32119 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1368, 42), 'eps', False)
    # Getting the type of 'A' (line 1368)
    A_32120 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1368, 47), 'A', False)
    # Getting the type of 'winit' (line 1368)
    winit_32121 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1368, 50), 'winit', False)
    # Getting the type of 'w' (line 1368)
    w_32122 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1368, 57), 'w', False)
    # Processing the call keyword arguments (line 1368)
    kwargs_32123 = {}
    # Getting the type of '_id' (line 1368)
    _id_32117 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1368, 28), '_id', False)
    # Obtaining the member 'idzp_asvd' of a type (line 1368)
    idzp_asvd_32118 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1368, 28), _id_32117, 'idzp_asvd')
    # Calling idzp_asvd(args, kwargs) (line 1368)
    idzp_asvd_call_result_32124 = invoke(stypy.reporting.localization.Localization(__file__, 1368, 28), idzp_asvd_32118, *[eps_32119, A_32120, winit_32121, w_32122], **kwargs_32123)
    
    # Obtaining the member '__getitem__' of a type (line 1368)
    getitem___32125 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1368, 4), idzp_asvd_call_result_32124, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1368)
    subscript_call_result_32126 = invoke(stypy.reporting.localization.Localization(__file__, 1368, 4), getitem___32125, int_32116)
    
    # Assigning a type to the variable 'tuple_var_assignment_29744' (line 1368)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1368, 4), 'tuple_var_assignment_29744', subscript_call_result_32126)
    
    # Assigning a Subscript to a Name (line 1368):
    
    # Obtaining the type of the subscript
    int_32127 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1368, 4), 'int')
    
    # Call to idzp_asvd(...): (line 1368)
    # Processing the call arguments (line 1368)
    # Getting the type of 'eps' (line 1368)
    eps_32130 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1368, 42), 'eps', False)
    # Getting the type of 'A' (line 1368)
    A_32131 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1368, 47), 'A', False)
    # Getting the type of 'winit' (line 1368)
    winit_32132 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1368, 50), 'winit', False)
    # Getting the type of 'w' (line 1368)
    w_32133 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1368, 57), 'w', False)
    # Processing the call keyword arguments (line 1368)
    kwargs_32134 = {}
    # Getting the type of '_id' (line 1368)
    _id_32128 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1368, 28), '_id', False)
    # Obtaining the member 'idzp_asvd' of a type (line 1368)
    idzp_asvd_32129 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1368, 28), _id_32128, 'idzp_asvd')
    # Calling idzp_asvd(args, kwargs) (line 1368)
    idzp_asvd_call_result_32135 = invoke(stypy.reporting.localization.Localization(__file__, 1368, 28), idzp_asvd_32129, *[eps_32130, A_32131, winit_32132, w_32133], **kwargs_32134)
    
    # Obtaining the member '__getitem__' of a type (line 1368)
    getitem___32136 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1368, 4), idzp_asvd_call_result_32135, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1368)
    subscript_call_result_32137 = invoke(stypy.reporting.localization.Localization(__file__, 1368, 4), getitem___32136, int_32127)
    
    # Assigning a type to the variable 'tuple_var_assignment_29745' (line 1368)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1368, 4), 'tuple_var_assignment_29745', subscript_call_result_32137)
    
    # Assigning a Subscript to a Name (line 1368):
    
    # Obtaining the type of the subscript
    int_32138 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1368, 4), 'int')
    
    # Call to idzp_asvd(...): (line 1368)
    # Processing the call arguments (line 1368)
    # Getting the type of 'eps' (line 1368)
    eps_32141 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1368, 42), 'eps', False)
    # Getting the type of 'A' (line 1368)
    A_32142 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1368, 47), 'A', False)
    # Getting the type of 'winit' (line 1368)
    winit_32143 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1368, 50), 'winit', False)
    # Getting the type of 'w' (line 1368)
    w_32144 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1368, 57), 'w', False)
    # Processing the call keyword arguments (line 1368)
    kwargs_32145 = {}
    # Getting the type of '_id' (line 1368)
    _id_32139 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1368, 28), '_id', False)
    # Obtaining the member 'idzp_asvd' of a type (line 1368)
    idzp_asvd_32140 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1368, 28), _id_32139, 'idzp_asvd')
    # Calling idzp_asvd(args, kwargs) (line 1368)
    idzp_asvd_call_result_32146 = invoke(stypy.reporting.localization.Localization(__file__, 1368, 28), idzp_asvd_32140, *[eps_32141, A_32142, winit_32143, w_32144], **kwargs_32145)
    
    # Obtaining the member '__getitem__' of a type (line 1368)
    getitem___32147 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1368, 4), idzp_asvd_call_result_32146, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1368)
    subscript_call_result_32148 = invoke(stypy.reporting.localization.Localization(__file__, 1368, 4), getitem___32147, int_32138)
    
    # Assigning a type to the variable 'tuple_var_assignment_29746' (line 1368)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1368, 4), 'tuple_var_assignment_29746', subscript_call_result_32148)
    
    # Assigning a Subscript to a Name (line 1368):
    
    # Obtaining the type of the subscript
    int_32149 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1368, 4), 'int')
    
    # Call to idzp_asvd(...): (line 1368)
    # Processing the call arguments (line 1368)
    # Getting the type of 'eps' (line 1368)
    eps_32152 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1368, 42), 'eps', False)
    # Getting the type of 'A' (line 1368)
    A_32153 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1368, 47), 'A', False)
    # Getting the type of 'winit' (line 1368)
    winit_32154 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1368, 50), 'winit', False)
    # Getting the type of 'w' (line 1368)
    w_32155 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1368, 57), 'w', False)
    # Processing the call keyword arguments (line 1368)
    kwargs_32156 = {}
    # Getting the type of '_id' (line 1368)
    _id_32150 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1368, 28), '_id', False)
    # Obtaining the member 'idzp_asvd' of a type (line 1368)
    idzp_asvd_32151 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1368, 28), _id_32150, 'idzp_asvd')
    # Calling idzp_asvd(args, kwargs) (line 1368)
    idzp_asvd_call_result_32157 = invoke(stypy.reporting.localization.Localization(__file__, 1368, 28), idzp_asvd_32151, *[eps_32152, A_32153, winit_32154, w_32155], **kwargs_32156)
    
    # Obtaining the member '__getitem__' of a type (line 1368)
    getitem___32158 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1368, 4), idzp_asvd_call_result_32157, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1368)
    subscript_call_result_32159 = invoke(stypy.reporting.localization.Localization(__file__, 1368, 4), getitem___32158, int_32149)
    
    # Assigning a type to the variable 'tuple_var_assignment_29747' (line 1368)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1368, 4), 'tuple_var_assignment_29747', subscript_call_result_32159)
    
    # Assigning a Subscript to a Name (line 1368):
    
    # Obtaining the type of the subscript
    int_32160 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1368, 4), 'int')
    
    # Call to idzp_asvd(...): (line 1368)
    # Processing the call arguments (line 1368)
    # Getting the type of 'eps' (line 1368)
    eps_32163 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1368, 42), 'eps', False)
    # Getting the type of 'A' (line 1368)
    A_32164 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1368, 47), 'A', False)
    # Getting the type of 'winit' (line 1368)
    winit_32165 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1368, 50), 'winit', False)
    # Getting the type of 'w' (line 1368)
    w_32166 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1368, 57), 'w', False)
    # Processing the call keyword arguments (line 1368)
    kwargs_32167 = {}
    # Getting the type of '_id' (line 1368)
    _id_32161 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1368, 28), '_id', False)
    # Obtaining the member 'idzp_asvd' of a type (line 1368)
    idzp_asvd_32162 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1368, 28), _id_32161, 'idzp_asvd')
    # Calling idzp_asvd(args, kwargs) (line 1368)
    idzp_asvd_call_result_32168 = invoke(stypy.reporting.localization.Localization(__file__, 1368, 28), idzp_asvd_32162, *[eps_32163, A_32164, winit_32165, w_32166], **kwargs_32167)
    
    # Obtaining the member '__getitem__' of a type (line 1368)
    getitem___32169 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1368, 4), idzp_asvd_call_result_32168, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1368)
    subscript_call_result_32170 = invoke(stypy.reporting.localization.Localization(__file__, 1368, 4), getitem___32169, int_32160)
    
    # Assigning a type to the variable 'tuple_var_assignment_29748' (line 1368)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1368, 4), 'tuple_var_assignment_29748', subscript_call_result_32170)
    
    # Assigning a Name to a Name (line 1368):
    # Getting the type of 'tuple_var_assignment_29743' (line 1368)
    tuple_var_assignment_29743_32171 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1368, 4), 'tuple_var_assignment_29743')
    # Assigning a type to the variable 'k' (line 1368)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1368, 4), 'k', tuple_var_assignment_29743_32171)
    
    # Assigning a Name to a Name (line 1368):
    # Getting the type of 'tuple_var_assignment_29744' (line 1368)
    tuple_var_assignment_29744_32172 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1368, 4), 'tuple_var_assignment_29744')
    # Assigning a type to the variable 'iU' (line 1368)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1368, 7), 'iU', tuple_var_assignment_29744_32172)
    
    # Assigning a Name to a Name (line 1368):
    # Getting the type of 'tuple_var_assignment_29745' (line 1368)
    tuple_var_assignment_29745_32173 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1368, 4), 'tuple_var_assignment_29745')
    # Assigning a type to the variable 'iV' (line 1368)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1368, 11), 'iV', tuple_var_assignment_29745_32173)
    
    # Assigning a Name to a Name (line 1368):
    # Getting the type of 'tuple_var_assignment_29746' (line 1368)
    tuple_var_assignment_29746_32174 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1368, 4), 'tuple_var_assignment_29746')
    # Assigning a type to the variable 'iS' (line 1368)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1368, 15), 'iS', tuple_var_assignment_29746_32174)
    
    # Assigning a Name to a Name (line 1368):
    # Getting the type of 'tuple_var_assignment_29747' (line 1368)
    tuple_var_assignment_29747_32175 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1368, 4), 'tuple_var_assignment_29747')
    # Assigning a type to the variable 'w' (line 1368)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1368, 19), 'w', tuple_var_assignment_29747_32175)
    
    # Assigning a Name to a Name (line 1368):
    # Getting the type of 'tuple_var_assignment_29748' (line 1368)
    tuple_var_assignment_29748_32176 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1368, 4), 'tuple_var_assignment_29748')
    # Assigning a type to the variable 'ier' (line 1368)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1368, 22), 'ier', tuple_var_assignment_29748_32176)
    
    # Getting the type of 'ier' (line 1369)
    ier_32177 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1369, 7), 'ier')
    # Testing the type of an if condition (line 1369)
    if_condition_32178 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1369, 4), ier_32177)
    # Assigning a type to the variable 'if_condition_32178' (line 1369)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1369, 4), 'if_condition_32178', if_condition_32178)
    # SSA begins for if statement (line 1369)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # Getting the type of '_RETCODE_ERROR' (line 1370)
    _RETCODE_ERROR_32179 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1370, 14), '_RETCODE_ERROR')
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 1370, 8), _RETCODE_ERROR_32179, 'raise parameter', BaseException)
    # SSA join for if statement (line 1369)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 1371):
    
    # Assigning a Call to a Name (line 1371):
    
    # Call to reshape(...): (line 1371)
    # Processing the call arguments (line 1371)
    
    # Obtaining an instance of the builtin type 'tuple' (line 1371)
    tuple_32195 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1371, 34), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 1371)
    # Adding element type (line 1371)
    # Getting the type of 'm' (line 1371)
    m_32196 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1371, 34), 'm', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1371, 34), tuple_32195, m_32196)
    # Adding element type (line 1371)
    # Getting the type of 'k' (line 1371)
    k_32197 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1371, 37), 'k', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1371, 34), tuple_32195, k_32197)
    
    # Processing the call keyword arguments (line 1371)
    str_32198 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1371, 47), 'str', 'F')
    keyword_32199 = str_32198
    kwargs_32200 = {'order': keyword_32199}
    
    # Obtaining the type of the subscript
    # Getting the type of 'iU' (line 1371)
    iU_32180 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1371, 10), 'iU', False)
    int_32181 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1371, 13), 'int')
    # Applying the binary operator '-' (line 1371)
    result_sub_32182 = python_operator(stypy.reporting.localization.Localization(__file__, 1371, 10), '-', iU_32180, int_32181)
    
    # Getting the type of 'iU' (line 1371)
    iU_32183 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1371, 15), 'iU', False)
    # Getting the type of 'm' (line 1371)
    m_32184 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1371, 18), 'm', False)
    # Getting the type of 'k' (line 1371)
    k_32185 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1371, 20), 'k', False)
    # Applying the binary operator '*' (line 1371)
    result_mul_32186 = python_operator(stypy.reporting.localization.Localization(__file__, 1371, 18), '*', m_32184, k_32185)
    
    # Applying the binary operator '+' (line 1371)
    result_add_32187 = python_operator(stypy.reporting.localization.Localization(__file__, 1371, 15), '+', iU_32183, result_mul_32186)
    
    int_32188 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1371, 22), 'int')
    # Applying the binary operator '-' (line 1371)
    result_sub_32189 = python_operator(stypy.reporting.localization.Localization(__file__, 1371, 21), '-', result_add_32187, int_32188)
    
    slice_32190 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 1371, 8), result_sub_32182, result_sub_32189, None)
    # Getting the type of 'w' (line 1371)
    w_32191 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1371, 8), 'w', False)
    # Obtaining the member '__getitem__' of a type (line 1371)
    getitem___32192 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1371, 8), w_32191, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1371)
    subscript_call_result_32193 = invoke(stypy.reporting.localization.Localization(__file__, 1371, 8), getitem___32192, slice_32190)
    
    # Obtaining the member 'reshape' of a type (line 1371)
    reshape_32194 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1371, 8), subscript_call_result_32193, 'reshape')
    # Calling reshape(args, kwargs) (line 1371)
    reshape_call_result_32201 = invoke(stypy.reporting.localization.Localization(__file__, 1371, 8), reshape_32194, *[tuple_32195], **kwargs_32200)
    
    # Assigning a type to the variable 'U' (line 1371)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1371, 4), 'U', reshape_call_result_32201)
    
    # Assigning a Call to a Name (line 1372):
    
    # Assigning a Call to a Name (line 1372):
    
    # Call to reshape(...): (line 1372)
    # Processing the call arguments (line 1372)
    
    # Obtaining an instance of the builtin type 'tuple' (line 1372)
    tuple_32217 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1372, 34), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 1372)
    # Adding element type (line 1372)
    # Getting the type of 'n' (line 1372)
    n_32218 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1372, 34), 'n', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1372, 34), tuple_32217, n_32218)
    # Adding element type (line 1372)
    # Getting the type of 'k' (line 1372)
    k_32219 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1372, 37), 'k', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1372, 34), tuple_32217, k_32219)
    
    # Processing the call keyword arguments (line 1372)
    str_32220 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1372, 47), 'str', 'F')
    keyword_32221 = str_32220
    kwargs_32222 = {'order': keyword_32221}
    
    # Obtaining the type of the subscript
    # Getting the type of 'iV' (line 1372)
    iV_32202 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1372, 10), 'iV', False)
    int_32203 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1372, 13), 'int')
    # Applying the binary operator '-' (line 1372)
    result_sub_32204 = python_operator(stypy.reporting.localization.Localization(__file__, 1372, 10), '-', iV_32202, int_32203)
    
    # Getting the type of 'iV' (line 1372)
    iV_32205 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1372, 15), 'iV', False)
    # Getting the type of 'n' (line 1372)
    n_32206 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1372, 18), 'n', False)
    # Getting the type of 'k' (line 1372)
    k_32207 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1372, 20), 'k', False)
    # Applying the binary operator '*' (line 1372)
    result_mul_32208 = python_operator(stypy.reporting.localization.Localization(__file__, 1372, 18), '*', n_32206, k_32207)
    
    # Applying the binary operator '+' (line 1372)
    result_add_32209 = python_operator(stypy.reporting.localization.Localization(__file__, 1372, 15), '+', iV_32205, result_mul_32208)
    
    int_32210 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1372, 22), 'int')
    # Applying the binary operator '-' (line 1372)
    result_sub_32211 = python_operator(stypy.reporting.localization.Localization(__file__, 1372, 21), '-', result_add_32209, int_32210)
    
    slice_32212 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 1372, 8), result_sub_32204, result_sub_32211, None)
    # Getting the type of 'w' (line 1372)
    w_32213 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1372, 8), 'w', False)
    # Obtaining the member '__getitem__' of a type (line 1372)
    getitem___32214 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1372, 8), w_32213, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1372)
    subscript_call_result_32215 = invoke(stypy.reporting.localization.Localization(__file__, 1372, 8), getitem___32214, slice_32212)
    
    # Obtaining the member 'reshape' of a type (line 1372)
    reshape_32216 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1372, 8), subscript_call_result_32215, 'reshape')
    # Calling reshape(args, kwargs) (line 1372)
    reshape_call_result_32223 = invoke(stypy.reporting.localization.Localization(__file__, 1372, 8), reshape_32216, *[tuple_32217], **kwargs_32222)
    
    # Assigning a type to the variable 'V' (line 1372)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1372, 4), 'V', reshape_call_result_32223)
    
    # Assigning a Subscript to a Name (line 1373):
    
    # Assigning a Subscript to a Name (line 1373):
    
    # Obtaining the type of the subscript
    # Getting the type of 'iS' (line 1373)
    iS_32224 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1373, 10), 'iS')
    int_32225 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1373, 13), 'int')
    # Applying the binary operator '-' (line 1373)
    result_sub_32226 = python_operator(stypy.reporting.localization.Localization(__file__, 1373, 10), '-', iS_32224, int_32225)
    
    # Getting the type of 'iS' (line 1373)
    iS_32227 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1373, 15), 'iS')
    # Getting the type of 'k' (line 1373)
    k_32228 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1373, 18), 'k')
    # Applying the binary operator '+' (line 1373)
    result_add_32229 = python_operator(stypy.reporting.localization.Localization(__file__, 1373, 15), '+', iS_32227, k_32228)
    
    int_32230 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1373, 20), 'int')
    # Applying the binary operator '-' (line 1373)
    result_sub_32231 = python_operator(stypy.reporting.localization.Localization(__file__, 1373, 19), '-', result_add_32229, int_32230)
    
    slice_32232 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 1373, 8), result_sub_32226, result_sub_32231, None)
    # Getting the type of 'w' (line 1373)
    w_32233 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1373, 8), 'w')
    # Obtaining the member '__getitem__' of a type (line 1373)
    getitem___32234 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1373, 8), w_32233, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1373)
    subscript_call_result_32235 = invoke(stypy.reporting.localization.Localization(__file__, 1373, 8), getitem___32234, slice_32232)
    
    # Assigning a type to the variable 'S' (line 1373)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1373, 4), 'S', subscript_call_result_32235)
    
    # Obtaining an instance of the builtin type 'tuple' (line 1374)
    tuple_32236 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1374, 11), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 1374)
    # Adding element type (line 1374)
    # Getting the type of 'U' (line 1374)
    U_32237 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1374, 11), 'U')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1374, 11), tuple_32236, U_32237)
    # Adding element type (line 1374)
    # Getting the type of 'V' (line 1374)
    V_32238 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1374, 14), 'V')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1374, 11), tuple_32236, V_32238)
    # Adding element type (line 1374)
    # Getting the type of 'S' (line 1374)
    S_32239 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1374, 17), 'S')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1374, 11), tuple_32236, S_32239)
    
    # Assigning a type to the variable 'stypy_return_type' (line 1374)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1374, 4), 'stypy_return_type', tuple_32236)
    
    # ################# End of 'idzp_asvd(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'idzp_asvd' in the type store
    # Getting the type of 'stypy_return_type' (line 1339)
    stypy_return_type_32240 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1339, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_32240)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'idzp_asvd'
    return stypy_return_type_32240

# Assigning a type to the variable 'idzp_asvd' (line 1339)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1339, 0), 'idzp_asvd', idzp_asvd)

@norecursion
def idzp_rid(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'idzp_rid'
    module_type_store = module_type_store.open_function_context('idzp_rid', 1381, 0, False)
    
    # Passed parameters checking function
    idzp_rid.stypy_localization = localization
    idzp_rid.stypy_type_of_self = None
    idzp_rid.stypy_type_store = module_type_store
    idzp_rid.stypy_function_name = 'idzp_rid'
    idzp_rid.stypy_param_names_list = ['eps', 'm', 'n', 'matveca']
    idzp_rid.stypy_varargs_param_name = None
    idzp_rid.stypy_kwargs_param_name = None
    idzp_rid.stypy_call_defaults = defaults
    idzp_rid.stypy_call_varargs = varargs
    idzp_rid.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'idzp_rid', ['eps', 'm', 'n', 'matveca'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'idzp_rid', localization, ['eps', 'm', 'n', 'matveca'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'idzp_rid(...)' code ##################

    str_32241 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1410, (-1)), 'str', '\n    Compute ID of a complex matrix to a specified relative precision using\n    random matrix-vector multiplication.\n\n    :param eps:\n        Relative precision.\n    :type eps: float\n    :param m:\n        Matrix row dimension.\n    :type m: int\n    :param n:\n        Matrix column dimension.\n    :type n: int\n    :param matveca:\n        Function to apply the matrix adjoint to a vector, with call signature\n        `y = matveca(x)`, where `x` and `y` are the input and output vectors,\n        respectively.\n    :type matveca: function\n\n    :return:\n        Rank of ID.\n    :rtype: int\n    :return:\n        Column index array.\n    :rtype: :class:`numpy.ndarray`\n    :return:\n        Interpolation coefficients.\n    :rtype: :class:`numpy.ndarray`\n    ')
    
    # Assigning a Call to a Name (line 1411):
    
    # Assigning a Call to a Name (line 1411):
    
    # Call to empty(...): (line 1411)
    # Processing the call arguments (line 1411)
    # Getting the type of 'm' (line 1412)
    m_32244 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1412, 8), 'm', False)
    int_32245 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1412, 12), 'int')
    # Applying the binary operator '+' (line 1412)
    result_add_32246 = python_operator(stypy.reporting.localization.Localization(__file__, 1412, 8), '+', m_32244, int_32245)
    
    int_32247 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1412, 16), 'int')
    # Getting the type of 'n' (line 1412)
    n_32248 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1412, 18), 'n', False)
    # Applying the binary operator '*' (line 1412)
    result_mul_32249 = python_operator(stypy.reporting.localization.Localization(__file__, 1412, 16), '*', int_32247, n_32248)
    
    
    # Call to min(...): (line 1412)
    # Processing the call arguments (line 1412)
    # Getting the type of 'm' (line 1412)
    m_32251 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1412, 25), 'm', False)
    # Getting the type of 'n' (line 1412)
    n_32252 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1412, 28), 'n', False)
    # Processing the call keyword arguments (line 1412)
    kwargs_32253 = {}
    # Getting the type of 'min' (line 1412)
    min_32250 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1412, 21), 'min', False)
    # Calling min(args, kwargs) (line 1412)
    min_call_result_32254 = invoke(stypy.reporting.localization.Localization(__file__, 1412, 21), min_32250, *[m_32251, n_32252], **kwargs_32253)
    
    int_32255 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1412, 33), 'int')
    # Applying the binary operator '+' (line 1412)
    result_add_32256 = python_operator(stypy.reporting.localization.Localization(__file__, 1412, 21), '+', min_call_result_32254, int_32255)
    
    # Applying the binary operator '*' (line 1412)
    result_mul_32257 = python_operator(stypy.reporting.localization.Localization(__file__, 1412, 19), '*', result_mul_32249, result_add_32256)
    
    # Applying the binary operator '+' (line 1412)
    result_add_32258 = python_operator(stypy.reporting.localization.Localization(__file__, 1412, 14), '+', result_add_32246, result_mul_32257)
    
    # Processing the call keyword arguments (line 1411)
    # Getting the type of 'np' (line 1413)
    np_32259 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1413, 14), 'np', False)
    # Obtaining the member 'complex128' of a type (line 1413)
    complex128_32260 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1413, 14), np_32259, 'complex128')
    keyword_32261 = complex128_32260
    str_32262 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1413, 35), 'str', 'F')
    keyword_32263 = str_32262
    kwargs_32264 = {'dtype': keyword_32261, 'order': keyword_32263}
    # Getting the type of 'np' (line 1411)
    np_32242 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1411, 11), 'np', False)
    # Obtaining the member 'empty' of a type (line 1411)
    empty_32243 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1411, 11), np_32242, 'empty')
    # Calling empty(args, kwargs) (line 1411)
    empty_call_result_32265 = invoke(stypy.reporting.localization.Localization(__file__, 1411, 11), empty_32243, *[result_add_32258], **kwargs_32264)
    
    # Assigning a type to the variable 'proj' (line 1411)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1411, 4), 'proj', empty_call_result_32265)
    
    # Assigning a Call to a Tuple (line 1414):
    
    # Assigning a Subscript to a Name (line 1414):
    
    # Obtaining the type of the subscript
    int_32266 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1414, 4), 'int')
    
    # Call to idzp_rid(...): (line 1414)
    # Processing the call arguments (line 1414)
    # Getting the type of 'eps' (line 1414)
    eps_32269 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1414, 37), 'eps', False)
    # Getting the type of 'm' (line 1414)
    m_32270 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1414, 42), 'm', False)
    # Getting the type of 'n' (line 1414)
    n_32271 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1414, 45), 'n', False)
    # Getting the type of 'matveca' (line 1414)
    matveca_32272 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1414, 48), 'matveca', False)
    # Getting the type of 'proj' (line 1414)
    proj_32273 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1414, 57), 'proj', False)
    # Processing the call keyword arguments (line 1414)
    kwargs_32274 = {}
    # Getting the type of '_id' (line 1414)
    _id_32267 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1414, 24), '_id', False)
    # Obtaining the member 'idzp_rid' of a type (line 1414)
    idzp_rid_32268 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1414, 24), _id_32267, 'idzp_rid')
    # Calling idzp_rid(args, kwargs) (line 1414)
    idzp_rid_call_result_32275 = invoke(stypy.reporting.localization.Localization(__file__, 1414, 24), idzp_rid_32268, *[eps_32269, m_32270, n_32271, matveca_32272, proj_32273], **kwargs_32274)
    
    # Obtaining the member '__getitem__' of a type (line 1414)
    getitem___32276 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1414, 4), idzp_rid_call_result_32275, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1414)
    subscript_call_result_32277 = invoke(stypy.reporting.localization.Localization(__file__, 1414, 4), getitem___32276, int_32266)
    
    # Assigning a type to the variable 'tuple_var_assignment_29749' (line 1414)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1414, 4), 'tuple_var_assignment_29749', subscript_call_result_32277)
    
    # Assigning a Subscript to a Name (line 1414):
    
    # Obtaining the type of the subscript
    int_32278 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1414, 4), 'int')
    
    # Call to idzp_rid(...): (line 1414)
    # Processing the call arguments (line 1414)
    # Getting the type of 'eps' (line 1414)
    eps_32281 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1414, 37), 'eps', False)
    # Getting the type of 'm' (line 1414)
    m_32282 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1414, 42), 'm', False)
    # Getting the type of 'n' (line 1414)
    n_32283 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1414, 45), 'n', False)
    # Getting the type of 'matveca' (line 1414)
    matveca_32284 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1414, 48), 'matveca', False)
    # Getting the type of 'proj' (line 1414)
    proj_32285 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1414, 57), 'proj', False)
    # Processing the call keyword arguments (line 1414)
    kwargs_32286 = {}
    # Getting the type of '_id' (line 1414)
    _id_32279 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1414, 24), '_id', False)
    # Obtaining the member 'idzp_rid' of a type (line 1414)
    idzp_rid_32280 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1414, 24), _id_32279, 'idzp_rid')
    # Calling idzp_rid(args, kwargs) (line 1414)
    idzp_rid_call_result_32287 = invoke(stypy.reporting.localization.Localization(__file__, 1414, 24), idzp_rid_32280, *[eps_32281, m_32282, n_32283, matveca_32284, proj_32285], **kwargs_32286)
    
    # Obtaining the member '__getitem__' of a type (line 1414)
    getitem___32288 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1414, 4), idzp_rid_call_result_32287, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1414)
    subscript_call_result_32289 = invoke(stypy.reporting.localization.Localization(__file__, 1414, 4), getitem___32288, int_32278)
    
    # Assigning a type to the variable 'tuple_var_assignment_29750' (line 1414)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1414, 4), 'tuple_var_assignment_29750', subscript_call_result_32289)
    
    # Assigning a Subscript to a Name (line 1414):
    
    # Obtaining the type of the subscript
    int_32290 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1414, 4), 'int')
    
    # Call to idzp_rid(...): (line 1414)
    # Processing the call arguments (line 1414)
    # Getting the type of 'eps' (line 1414)
    eps_32293 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1414, 37), 'eps', False)
    # Getting the type of 'm' (line 1414)
    m_32294 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1414, 42), 'm', False)
    # Getting the type of 'n' (line 1414)
    n_32295 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1414, 45), 'n', False)
    # Getting the type of 'matveca' (line 1414)
    matveca_32296 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1414, 48), 'matveca', False)
    # Getting the type of 'proj' (line 1414)
    proj_32297 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1414, 57), 'proj', False)
    # Processing the call keyword arguments (line 1414)
    kwargs_32298 = {}
    # Getting the type of '_id' (line 1414)
    _id_32291 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1414, 24), '_id', False)
    # Obtaining the member 'idzp_rid' of a type (line 1414)
    idzp_rid_32292 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1414, 24), _id_32291, 'idzp_rid')
    # Calling idzp_rid(args, kwargs) (line 1414)
    idzp_rid_call_result_32299 = invoke(stypy.reporting.localization.Localization(__file__, 1414, 24), idzp_rid_32292, *[eps_32293, m_32294, n_32295, matveca_32296, proj_32297], **kwargs_32298)
    
    # Obtaining the member '__getitem__' of a type (line 1414)
    getitem___32300 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1414, 4), idzp_rid_call_result_32299, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1414)
    subscript_call_result_32301 = invoke(stypy.reporting.localization.Localization(__file__, 1414, 4), getitem___32300, int_32290)
    
    # Assigning a type to the variable 'tuple_var_assignment_29751' (line 1414)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1414, 4), 'tuple_var_assignment_29751', subscript_call_result_32301)
    
    # Assigning a Subscript to a Name (line 1414):
    
    # Obtaining the type of the subscript
    int_32302 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1414, 4), 'int')
    
    # Call to idzp_rid(...): (line 1414)
    # Processing the call arguments (line 1414)
    # Getting the type of 'eps' (line 1414)
    eps_32305 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1414, 37), 'eps', False)
    # Getting the type of 'm' (line 1414)
    m_32306 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1414, 42), 'm', False)
    # Getting the type of 'n' (line 1414)
    n_32307 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1414, 45), 'n', False)
    # Getting the type of 'matveca' (line 1414)
    matveca_32308 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1414, 48), 'matveca', False)
    # Getting the type of 'proj' (line 1414)
    proj_32309 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1414, 57), 'proj', False)
    # Processing the call keyword arguments (line 1414)
    kwargs_32310 = {}
    # Getting the type of '_id' (line 1414)
    _id_32303 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1414, 24), '_id', False)
    # Obtaining the member 'idzp_rid' of a type (line 1414)
    idzp_rid_32304 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1414, 24), _id_32303, 'idzp_rid')
    # Calling idzp_rid(args, kwargs) (line 1414)
    idzp_rid_call_result_32311 = invoke(stypy.reporting.localization.Localization(__file__, 1414, 24), idzp_rid_32304, *[eps_32305, m_32306, n_32307, matveca_32308, proj_32309], **kwargs_32310)
    
    # Obtaining the member '__getitem__' of a type (line 1414)
    getitem___32312 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1414, 4), idzp_rid_call_result_32311, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1414)
    subscript_call_result_32313 = invoke(stypy.reporting.localization.Localization(__file__, 1414, 4), getitem___32312, int_32302)
    
    # Assigning a type to the variable 'tuple_var_assignment_29752' (line 1414)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1414, 4), 'tuple_var_assignment_29752', subscript_call_result_32313)
    
    # Assigning a Name to a Name (line 1414):
    # Getting the type of 'tuple_var_assignment_29749' (line 1414)
    tuple_var_assignment_29749_32314 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1414, 4), 'tuple_var_assignment_29749')
    # Assigning a type to the variable 'k' (line 1414)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1414, 4), 'k', tuple_var_assignment_29749_32314)
    
    # Assigning a Name to a Name (line 1414):
    # Getting the type of 'tuple_var_assignment_29750' (line 1414)
    tuple_var_assignment_29750_32315 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1414, 4), 'tuple_var_assignment_29750')
    # Assigning a type to the variable 'idx' (line 1414)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1414, 7), 'idx', tuple_var_assignment_29750_32315)
    
    # Assigning a Name to a Name (line 1414):
    # Getting the type of 'tuple_var_assignment_29751' (line 1414)
    tuple_var_assignment_29751_32316 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1414, 4), 'tuple_var_assignment_29751')
    # Assigning a type to the variable 'proj' (line 1414)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1414, 12), 'proj', tuple_var_assignment_29751_32316)
    
    # Assigning a Name to a Name (line 1414):
    # Getting the type of 'tuple_var_assignment_29752' (line 1414)
    tuple_var_assignment_29752_32317 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1414, 4), 'tuple_var_assignment_29752')
    # Assigning a type to the variable 'ier' (line 1414)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1414, 18), 'ier', tuple_var_assignment_29752_32317)
    
    # Getting the type of 'ier' (line 1415)
    ier_32318 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1415, 7), 'ier')
    # Testing the type of an if condition (line 1415)
    if_condition_32319 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1415, 4), ier_32318)
    # Assigning a type to the variable 'if_condition_32319' (line 1415)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1415, 4), 'if_condition_32319', if_condition_32319)
    # SSA begins for if statement (line 1415)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # Getting the type of '_RETCODE_ERROR' (line 1416)
    _RETCODE_ERROR_32320 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1416, 14), '_RETCODE_ERROR')
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 1416, 8), _RETCODE_ERROR_32320, 'raise parameter', BaseException)
    # SSA join for if statement (line 1415)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 1417):
    
    # Assigning a Call to a Name (line 1417):
    
    # Call to reshape(...): (line 1417)
    # Processing the call arguments (line 1417)
    
    # Obtaining an instance of the builtin type 'tuple' (line 1417)
    tuple_32331 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1417, 35), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 1417)
    # Adding element type (line 1417)
    # Getting the type of 'k' (line 1417)
    k_32332 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1417, 35), 'k', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1417, 35), tuple_32331, k_32332)
    # Adding element type (line 1417)
    # Getting the type of 'n' (line 1417)
    n_32333 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1417, 38), 'n', False)
    # Getting the type of 'k' (line 1417)
    k_32334 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1417, 40), 'k', False)
    # Applying the binary operator '-' (line 1417)
    result_sub_32335 = python_operator(stypy.reporting.localization.Localization(__file__, 1417, 38), '-', n_32333, k_32334)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1417, 35), tuple_32331, result_sub_32335)
    
    # Processing the call keyword arguments (line 1417)
    str_32336 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1417, 50), 'str', 'F')
    keyword_32337 = str_32336
    kwargs_32338 = {'order': keyword_32337}
    
    # Obtaining the type of the subscript
    # Getting the type of 'k' (line 1417)
    k_32321 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1417, 17), 'k', False)
    # Getting the type of 'n' (line 1417)
    n_32322 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1417, 20), 'n', False)
    # Getting the type of 'k' (line 1417)
    k_32323 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1417, 22), 'k', False)
    # Applying the binary operator '-' (line 1417)
    result_sub_32324 = python_operator(stypy.reporting.localization.Localization(__file__, 1417, 20), '-', n_32322, k_32323)
    
    # Applying the binary operator '*' (line 1417)
    result_mul_32325 = python_operator(stypy.reporting.localization.Localization(__file__, 1417, 17), '*', k_32321, result_sub_32324)
    
    slice_32326 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 1417, 11), None, result_mul_32325, None)
    # Getting the type of 'proj' (line 1417)
    proj_32327 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1417, 11), 'proj', False)
    # Obtaining the member '__getitem__' of a type (line 1417)
    getitem___32328 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1417, 11), proj_32327, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1417)
    subscript_call_result_32329 = invoke(stypy.reporting.localization.Localization(__file__, 1417, 11), getitem___32328, slice_32326)
    
    # Obtaining the member 'reshape' of a type (line 1417)
    reshape_32330 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1417, 11), subscript_call_result_32329, 'reshape')
    # Calling reshape(args, kwargs) (line 1417)
    reshape_call_result_32339 = invoke(stypy.reporting.localization.Localization(__file__, 1417, 11), reshape_32330, *[tuple_32331], **kwargs_32338)
    
    # Assigning a type to the variable 'proj' (line 1417)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1417, 4), 'proj', reshape_call_result_32339)
    
    # Obtaining an instance of the builtin type 'tuple' (line 1418)
    tuple_32340 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1418, 11), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 1418)
    # Adding element type (line 1418)
    # Getting the type of 'k' (line 1418)
    k_32341 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1418, 11), 'k')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1418, 11), tuple_32340, k_32341)
    # Adding element type (line 1418)
    # Getting the type of 'idx' (line 1418)
    idx_32342 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1418, 14), 'idx')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1418, 11), tuple_32340, idx_32342)
    # Adding element type (line 1418)
    # Getting the type of 'proj' (line 1418)
    proj_32343 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1418, 19), 'proj')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1418, 11), tuple_32340, proj_32343)
    
    # Assigning a type to the variable 'stypy_return_type' (line 1418)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1418, 4), 'stypy_return_type', tuple_32340)
    
    # ################# End of 'idzp_rid(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'idzp_rid' in the type store
    # Getting the type of 'stypy_return_type' (line 1381)
    stypy_return_type_32344 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1381, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_32344)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'idzp_rid'
    return stypy_return_type_32344

# Assigning a type to the variable 'idzp_rid' (line 1381)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1381, 0), 'idzp_rid', idzp_rid)

@norecursion
def idz_findrank(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'idz_findrank'
    module_type_store = module_type_store.open_function_context('idz_findrank', 1421, 0, False)
    
    # Passed parameters checking function
    idz_findrank.stypy_localization = localization
    idz_findrank.stypy_type_of_self = None
    idz_findrank.stypy_type_store = module_type_store
    idz_findrank.stypy_function_name = 'idz_findrank'
    idz_findrank.stypy_param_names_list = ['eps', 'm', 'n', 'matveca']
    idz_findrank.stypy_varargs_param_name = None
    idz_findrank.stypy_kwargs_param_name = None
    idz_findrank.stypy_call_defaults = defaults
    idz_findrank.stypy_call_varargs = varargs
    idz_findrank.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'idz_findrank', ['eps', 'm', 'n', 'matveca'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'idz_findrank', localization, ['eps', 'm', 'n', 'matveca'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'idz_findrank(...)' code ##################

    str_32345 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1444, (-1)), 'str', '\n    Estimate rank of a complex matrix to a specified relative precision using\n    random matrix-vector multiplication.\n\n    :param eps:\n        Relative precision.\n    :type eps: float\n    :param m:\n        Matrix row dimension.\n    :type m: int\n    :param n:\n        Matrix column dimension.\n    :type n: int\n    :param matveca:\n        Function to apply the matrix adjoint to a vector, with call signature\n        `y = matveca(x)`, where `x` and `y` are the input and output vectors,\n        respectively.\n    :type matveca: function\n\n    :return:\n        Rank estimate.\n    :rtype: int\n    ')
    
    # Assigning a Call to a Tuple (line 1445):
    
    # Assigning a Subscript to a Name (line 1445):
    
    # Obtaining the type of the subscript
    int_32346 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1445, 4), 'int')
    
    # Call to idz_findrank(...): (line 1445)
    # Processing the call arguments (line 1445)
    # Getting the type of 'eps' (line 1445)
    eps_32349 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1445, 34), 'eps', False)
    # Getting the type of 'm' (line 1445)
    m_32350 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1445, 39), 'm', False)
    # Getting the type of 'n' (line 1445)
    n_32351 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1445, 42), 'n', False)
    # Getting the type of 'matveca' (line 1445)
    matveca_32352 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1445, 45), 'matveca', False)
    # Processing the call keyword arguments (line 1445)
    kwargs_32353 = {}
    # Getting the type of '_id' (line 1445)
    _id_32347 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1445, 17), '_id', False)
    # Obtaining the member 'idz_findrank' of a type (line 1445)
    idz_findrank_32348 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1445, 17), _id_32347, 'idz_findrank')
    # Calling idz_findrank(args, kwargs) (line 1445)
    idz_findrank_call_result_32354 = invoke(stypy.reporting.localization.Localization(__file__, 1445, 17), idz_findrank_32348, *[eps_32349, m_32350, n_32351, matveca_32352], **kwargs_32353)
    
    # Obtaining the member '__getitem__' of a type (line 1445)
    getitem___32355 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1445, 4), idz_findrank_call_result_32354, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1445)
    subscript_call_result_32356 = invoke(stypy.reporting.localization.Localization(__file__, 1445, 4), getitem___32355, int_32346)
    
    # Assigning a type to the variable 'tuple_var_assignment_29753' (line 1445)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1445, 4), 'tuple_var_assignment_29753', subscript_call_result_32356)
    
    # Assigning a Subscript to a Name (line 1445):
    
    # Obtaining the type of the subscript
    int_32357 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1445, 4), 'int')
    
    # Call to idz_findrank(...): (line 1445)
    # Processing the call arguments (line 1445)
    # Getting the type of 'eps' (line 1445)
    eps_32360 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1445, 34), 'eps', False)
    # Getting the type of 'm' (line 1445)
    m_32361 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1445, 39), 'm', False)
    # Getting the type of 'n' (line 1445)
    n_32362 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1445, 42), 'n', False)
    # Getting the type of 'matveca' (line 1445)
    matveca_32363 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1445, 45), 'matveca', False)
    # Processing the call keyword arguments (line 1445)
    kwargs_32364 = {}
    # Getting the type of '_id' (line 1445)
    _id_32358 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1445, 17), '_id', False)
    # Obtaining the member 'idz_findrank' of a type (line 1445)
    idz_findrank_32359 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1445, 17), _id_32358, 'idz_findrank')
    # Calling idz_findrank(args, kwargs) (line 1445)
    idz_findrank_call_result_32365 = invoke(stypy.reporting.localization.Localization(__file__, 1445, 17), idz_findrank_32359, *[eps_32360, m_32361, n_32362, matveca_32363], **kwargs_32364)
    
    # Obtaining the member '__getitem__' of a type (line 1445)
    getitem___32366 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1445, 4), idz_findrank_call_result_32365, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1445)
    subscript_call_result_32367 = invoke(stypy.reporting.localization.Localization(__file__, 1445, 4), getitem___32366, int_32357)
    
    # Assigning a type to the variable 'tuple_var_assignment_29754' (line 1445)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1445, 4), 'tuple_var_assignment_29754', subscript_call_result_32367)
    
    # Assigning a Subscript to a Name (line 1445):
    
    # Obtaining the type of the subscript
    int_32368 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1445, 4), 'int')
    
    # Call to idz_findrank(...): (line 1445)
    # Processing the call arguments (line 1445)
    # Getting the type of 'eps' (line 1445)
    eps_32371 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1445, 34), 'eps', False)
    # Getting the type of 'm' (line 1445)
    m_32372 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1445, 39), 'm', False)
    # Getting the type of 'n' (line 1445)
    n_32373 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1445, 42), 'n', False)
    # Getting the type of 'matveca' (line 1445)
    matveca_32374 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1445, 45), 'matveca', False)
    # Processing the call keyword arguments (line 1445)
    kwargs_32375 = {}
    # Getting the type of '_id' (line 1445)
    _id_32369 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1445, 17), '_id', False)
    # Obtaining the member 'idz_findrank' of a type (line 1445)
    idz_findrank_32370 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1445, 17), _id_32369, 'idz_findrank')
    # Calling idz_findrank(args, kwargs) (line 1445)
    idz_findrank_call_result_32376 = invoke(stypy.reporting.localization.Localization(__file__, 1445, 17), idz_findrank_32370, *[eps_32371, m_32372, n_32373, matveca_32374], **kwargs_32375)
    
    # Obtaining the member '__getitem__' of a type (line 1445)
    getitem___32377 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1445, 4), idz_findrank_call_result_32376, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1445)
    subscript_call_result_32378 = invoke(stypy.reporting.localization.Localization(__file__, 1445, 4), getitem___32377, int_32368)
    
    # Assigning a type to the variable 'tuple_var_assignment_29755' (line 1445)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1445, 4), 'tuple_var_assignment_29755', subscript_call_result_32378)
    
    # Assigning a Name to a Name (line 1445):
    # Getting the type of 'tuple_var_assignment_29753' (line 1445)
    tuple_var_assignment_29753_32379 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1445, 4), 'tuple_var_assignment_29753')
    # Assigning a type to the variable 'k' (line 1445)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1445, 4), 'k', tuple_var_assignment_29753_32379)
    
    # Assigning a Name to a Name (line 1445):
    # Getting the type of 'tuple_var_assignment_29754' (line 1445)
    tuple_var_assignment_29754_32380 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1445, 4), 'tuple_var_assignment_29754')
    # Assigning a type to the variable 'ra' (line 1445)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1445, 7), 'ra', tuple_var_assignment_29754_32380)
    
    # Assigning a Name to a Name (line 1445):
    # Getting the type of 'tuple_var_assignment_29755' (line 1445)
    tuple_var_assignment_29755_32381 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1445, 4), 'tuple_var_assignment_29755')
    # Assigning a type to the variable 'ier' (line 1445)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1445, 11), 'ier', tuple_var_assignment_29755_32381)
    
    # Getting the type of 'ier' (line 1446)
    ier_32382 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1446, 7), 'ier')
    # Testing the type of an if condition (line 1446)
    if_condition_32383 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1446, 4), ier_32382)
    # Assigning a type to the variable 'if_condition_32383' (line 1446)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1446, 4), 'if_condition_32383', if_condition_32383)
    # SSA begins for if statement (line 1446)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # Getting the type of '_RETCODE_ERROR' (line 1447)
    _RETCODE_ERROR_32384 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1447, 14), '_RETCODE_ERROR')
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 1447, 8), _RETCODE_ERROR_32384, 'raise parameter', BaseException)
    # SSA join for if statement (line 1446)
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'k' (line 1448)
    k_32385 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1448, 11), 'k')
    # Assigning a type to the variable 'stypy_return_type' (line 1448)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1448, 4), 'stypy_return_type', k_32385)
    
    # ################# End of 'idz_findrank(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'idz_findrank' in the type store
    # Getting the type of 'stypy_return_type' (line 1421)
    stypy_return_type_32386 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1421, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_32386)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'idz_findrank'
    return stypy_return_type_32386

# Assigning a type to the variable 'idz_findrank' (line 1421)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1421, 0), 'idz_findrank', idz_findrank)

@norecursion
def idzp_rsvd(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'idzp_rsvd'
    module_type_store = module_type_store.open_function_context('idzp_rsvd', 1455, 0, False)
    
    # Passed parameters checking function
    idzp_rsvd.stypy_localization = localization
    idzp_rsvd.stypy_type_of_self = None
    idzp_rsvd.stypy_type_store = module_type_store
    idzp_rsvd.stypy_function_name = 'idzp_rsvd'
    idzp_rsvd.stypy_param_names_list = ['eps', 'm', 'n', 'matveca', 'matvec']
    idzp_rsvd.stypy_varargs_param_name = None
    idzp_rsvd.stypy_kwargs_param_name = None
    idzp_rsvd.stypy_call_defaults = defaults
    idzp_rsvd.stypy_call_varargs = varargs
    idzp_rsvd.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'idzp_rsvd', ['eps', 'm', 'n', 'matveca', 'matvec'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'idzp_rsvd', localization, ['eps', 'm', 'n', 'matveca', 'matvec'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'idzp_rsvd(...)' code ##################

    str_32387 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1489, (-1)), 'str', '\n    Compute SVD of a complex matrix to a specified relative precision using\n    random matrix-vector multiplication.\n\n    :param eps:\n        Relative precision.\n    :type eps: float\n    :param m:\n        Matrix row dimension.\n    :type m: int\n    :param n:\n        Matrix column dimension.\n    :type n: int\n    :param matveca:\n        Function to apply the matrix adjoint to a vector, with call signature\n        `y = matveca(x)`, where `x` and `y` are the input and output vectors,\n        respectively.\n    :type matveca: function\n    :param matvec:\n        Function to apply the matrix to a vector, with call signature\n        `y = matvec(x)`, where `x` and `y` are the input and output vectors,\n        respectively.\n    :type matvec: function\n\n    :return:\n        Left singular vectors.\n    :rtype: :class:`numpy.ndarray`\n    :return:\n        Right singular vectors.\n    :rtype: :class:`numpy.ndarray`\n    :return:\n        Singular values.\n    :rtype: :class:`numpy.ndarray`\n    ')
    
    # Assigning a Call to a Tuple (line 1490):
    
    # Assigning a Subscript to a Name (line 1490):
    
    # Obtaining the type of the subscript
    int_32388 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1490, 4), 'int')
    
    # Call to idzp_rsvd(...): (line 1490)
    # Processing the call arguments (line 1490)
    # Getting the type of 'eps' (line 1490)
    eps_32391 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1490, 42), 'eps', False)
    # Getting the type of 'm' (line 1490)
    m_32392 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1490, 47), 'm', False)
    # Getting the type of 'n' (line 1490)
    n_32393 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1490, 50), 'n', False)
    # Getting the type of 'matveca' (line 1490)
    matveca_32394 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1490, 53), 'matveca', False)
    # Getting the type of 'matvec' (line 1490)
    matvec_32395 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1490, 62), 'matvec', False)
    # Processing the call keyword arguments (line 1490)
    kwargs_32396 = {}
    # Getting the type of '_id' (line 1490)
    _id_32389 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1490, 28), '_id', False)
    # Obtaining the member 'idzp_rsvd' of a type (line 1490)
    idzp_rsvd_32390 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1490, 28), _id_32389, 'idzp_rsvd')
    # Calling idzp_rsvd(args, kwargs) (line 1490)
    idzp_rsvd_call_result_32397 = invoke(stypy.reporting.localization.Localization(__file__, 1490, 28), idzp_rsvd_32390, *[eps_32391, m_32392, n_32393, matveca_32394, matvec_32395], **kwargs_32396)
    
    # Obtaining the member '__getitem__' of a type (line 1490)
    getitem___32398 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1490, 4), idzp_rsvd_call_result_32397, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1490)
    subscript_call_result_32399 = invoke(stypy.reporting.localization.Localization(__file__, 1490, 4), getitem___32398, int_32388)
    
    # Assigning a type to the variable 'tuple_var_assignment_29756' (line 1490)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1490, 4), 'tuple_var_assignment_29756', subscript_call_result_32399)
    
    # Assigning a Subscript to a Name (line 1490):
    
    # Obtaining the type of the subscript
    int_32400 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1490, 4), 'int')
    
    # Call to idzp_rsvd(...): (line 1490)
    # Processing the call arguments (line 1490)
    # Getting the type of 'eps' (line 1490)
    eps_32403 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1490, 42), 'eps', False)
    # Getting the type of 'm' (line 1490)
    m_32404 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1490, 47), 'm', False)
    # Getting the type of 'n' (line 1490)
    n_32405 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1490, 50), 'n', False)
    # Getting the type of 'matveca' (line 1490)
    matveca_32406 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1490, 53), 'matveca', False)
    # Getting the type of 'matvec' (line 1490)
    matvec_32407 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1490, 62), 'matvec', False)
    # Processing the call keyword arguments (line 1490)
    kwargs_32408 = {}
    # Getting the type of '_id' (line 1490)
    _id_32401 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1490, 28), '_id', False)
    # Obtaining the member 'idzp_rsvd' of a type (line 1490)
    idzp_rsvd_32402 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1490, 28), _id_32401, 'idzp_rsvd')
    # Calling idzp_rsvd(args, kwargs) (line 1490)
    idzp_rsvd_call_result_32409 = invoke(stypy.reporting.localization.Localization(__file__, 1490, 28), idzp_rsvd_32402, *[eps_32403, m_32404, n_32405, matveca_32406, matvec_32407], **kwargs_32408)
    
    # Obtaining the member '__getitem__' of a type (line 1490)
    getitem___32410 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1490, 4), idzp_rsvd_call_result_32409, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1490)
    subscript_call_result_32411 = invoke(stypy.reporting.localization.Localization(__file__, 1490, 4), getitem___32410, int_32400)
    
    # Assigning a type to the variable 'tuple_var_assignment_29757' (line 1490)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1490, 4), 'tuple_var_assignment_29757', subscript_call_result_32411)
    
    # Assigning a Subscript to a Name (line 1490):
    
    # Obtaining the type of the subscript
    int_32412 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1490, 4), 'int')
    
    # Call to idzp_rsvd(...): (line 1490)
    # Processing the call arguments (line 1490)
    # Getting the type of 'eps' (line 1490)
    eps_32415 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1490, 42), 'eps', False)
    # Getting the type of 'm' (line 1490)
    m_32416 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1490, 47), 'm', False)
    # Getting the type of 'n' (line 1490)
    n_32417 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1490, 50), 'n', False)
    # Getting the type of 'matveca' (line 1490)
    matveca_32418 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1490, 53), 'matveca', False)
    # Getting the type of 'matvec' (line 1490)
    matvec_32419 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1490, 62), 'matvec', False)
    # Processing the call keyword arguments (line 1490)
    kwargs_32420 = {}
    # Getting the type of '_id' (line 1490)
    _id_32413 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1490, 28), '_id', False)
    # Obtaining the member 'idzp_rsvd' of a type (line 1490)
    idzp_rsvd_32414 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1490, 28), _id_32413, 'idzp_rsvd')
    # Calling idzp_rsvd(args, kwargs) (line 1490)
    idzp_rsvd_call_result_32421 = invoke(stypy.reporting.localization.Localization(__file__, 1490, 28), idzp_rsvd_32414, *[eps_32415, m_32416, n_32417, matveca_32418, matvec_32419], **kwargs_32420)
    
    # Obtaining the member '__getitem__' of a type (line 1490)
    getitem___32422 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1490, 4), idzp_rsvd_call_result_32421, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1490)
    subscript_call_result_32423 = invoke(stypy.reporting.localization.Localization(__file__, 1490, 4), getitem___32422, int_32412)
    
    # Assigning a type to the variable 'tuple_var_assignment_29758' (line 1490)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1490, 4), 'tuple_var_assignment_29758', subscript_call_result_32423)
    
    # Assigning a Subscript to a Name (line 1490):
    
    # Obtaining the type of the subscript
    int_32424 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1490, 4), 'int')
    
    # Call to idzp_rsvd(...): (line 1490)
    # Processing the call arguments (line 1490)
    # Getting the type of 'eps' (line 1490)
    eps_32427 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1490, 42), 'eps', False)
    # Getting the type of 'm' (line 1490)
    m_32428 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1490, 47), 'm', False)
    # Getting the type of 'n' (line 1490)
    n_32429 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1490, 50), 'n', False)
    # Getting the type of 'matveca' (line 1490)
    matveca_32430 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1490, 53), 'matveca', False)
    # Getting the type of 'matvec' (line 1490)
    matvec_32431 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1490, 62), 'matvec', False)
    # Processing the call keyword arguments (line 1490)
    kwargs_32432 = {}
    # Getting the type of '_id' (line 1490)
    _id_32425 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1490, 28), '_id', False)
    # Obtaining the member 'idzp_rsvd' of a type (line 1490)
    idzp_rsvd_32426 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1490, 28), _id_32425, 'idzp_rsvd')
    # Calling idzp_rsvd(args, kwargs) (line 1490)
    idzp_rsvd_call_result_32433 = invoke(stypy.reporting.localization.Localization(__file__, 1490, 28), idzp_rsvd_32426, *[eps_32427, m_32428, n_32429, matveca_32430, matvec_32431], **kwargs_32432)
    
    # Obtaining the member '__getitem__' of a type (line 1490)
    getitem___32434 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1490, 4), idzp_rsvd_call_result_32433, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1490)
    subscript_call_result_32435 = invoke(stypy.reporting.localization.Localization(__file__, 1490, 4), getitem___32434, int_32424)
    
    # Assigning a type to the variable 'tuple_var_assignment_29759' (line 1490)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1490, 4), 'tuple_var_assignment_29759', subscript_call_result_32435)
    
    # Assigning a Subscript to a Name (line 1490):
    
    # Obtaining the type of the subscript
    int_32436 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1490, 4), 'int')
    
    # Call to idzp_rsvd(...): (line 1490)
    # Processing the call arguments (line 1490)
    # Getting the type of 'eps' (line 1490)
    eps_32439 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1490, 42), 'eps', False)
    # Getting the type of 'm' (line 1490)
    m_32440 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1490, 47), 'm', False)
    # Getting the type of 'n' (line 1490)
    n_32441 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1490, 50), 'n', False)
    # Getting the type of 'matveca' (line 1490)
    matveca_32442 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1490, 53), 'matveca', False)
    # Getting the type of 'matvec' (line 1490)
    matvec_32443 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1490, 62), 'matvec', False)
    # Processing the call keyword arguments (line 1490)
    kwargs_32444 = {}
    # Getting the type of '_id' (line 1490)
    _id_32437 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1490, 28), '_id', False)
    # Obtaining the member 'idzp_rsvd' of a type (line 1490)
    idzp_rsvd_32438 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1490, 28), _id_32437, 'idzp_rsvd')
    # Calling idzp_rsvd(args, kwargs) (line 1490)
    idzp_rsvd_call_result_32445 = invoke(stypy.reporting.localization.Localization(__file__, 1490, 28), idzp_rsvd_32438, *[eps_32439, m_32440, n_32441, matveca_32442, matvec_32443], **kwargs_32444)
    
    # Obtaining the member '__getitem__' of a type (line 1490)
    getitem___32446 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1490, 4), idzp_rsvd_call_result_32445, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1490)
    subscript_call_result_32447 = invoke(stypy.reporting.localization.Localization(__file__, 1490, 4), getitem___32446, int_32436)
    
    # Assigning a type to the variable 'tuple_var_assignment_29760' (line 1490)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1490, 4), 'tuple_var_assignment_29760', subscript_call_result_32447)
    
    # Assigning a Subscript to a Name (line 1490):
    
    # Obtaining the type of the subscript
    int_32448 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1490, 4), 'int')
    
    # Call to idzp_rsvd(...): (line 1490)
    # Processing the call arguments (line 1490)
    # Getting the type of 'eps' (line 1490)
    eps_32451 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1490, 42), 'eps', False)
    # Getting the type of 'm' (line 1490)
    m_32452 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1490, 47), 'm', False)
    # Getting the type of 'n' (line 1490)
    n_32453 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1490, 50), 'n', False)
    # Getting the type of 'matveca' (line 1490)
    matveca_32454 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1490, 53), 'matveca', False)
    # Getting the type of 'matvec' (line 1490)
    matvec_32455 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1490, 62), 'matvec', False)
    # Processing the call keyword arguments (line 1490)
    kwargs_32456 = {}
    # Getting the type of '_id' (line 1490)
    _id_32449 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1490, 28), '_id', False)
    # Obtaining the member 'idzp_rsvd' of a type (line 1490)
    idzp_rsvd_32450 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1490, 28), _id_32449, 'idzp_rsvd')
    # Calling idzp_rsvd(args, kwargs) (line 1490)
    idzp_rsvd_call_result_32457 = invoke(stypy.reporting.localization.Localization(__file__, 1490, 28), idzp_rsvd_32450, *[eps_32451, m_32452, n_32453, matveca_32454, matvec_32455], **kwargs_32456)
    
    # Obtaining the member '__getitem__' of a type (line 1490)
    getitem___32458 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1490, 4), idzp_rsvd_call_result_32457, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1490)
    subscript_call_result_32459 = invoke(stypy.reporting.localization.Localization(__file__, 1490, 4), getitem___32458, int_32448)
    
    # Assigning a type to the variable 'tuple_var_assignment_29761' (line 1490)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1490, 4), 'tuple_var_assignment_29761', subscript_call_result_32459)
    
    # Assigning a Name to a Name (line 1490):
    # Getting the type of 'tuple_var_assignment_29756' (line 1490)
    tuple_var_assignment_29756_32460 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1490, 4), 'tuple_var_assignment_29756')
    # Assigning a type to the variable 'k' (line 1490)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1490, 4), 'k', tuple_var_assignment_29756_32460)
    
    # Assigning a Name to a Name (line 1490):
    # Getting the type of 'tuple_var_assignment_29757' (line 1490)
    tuple_var_assignment_29757_32461 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1490, 4), 'tuple_var_assignment_29757')
    # Assigning a type to the variable 'iU' (line 1490)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1490, 7), 'iU', tuple_var_assignment_29757_32461)
    
    # Assigning a Name to a Name (line 1490):
    # Getting the type of 'tuple_var_assignment_29758' (line 1490)
    tuple_var_assignment_29758_32462 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1490, 4), 'tuple_var_assignment_29758')
    # Assigning a type to the variable 'iV' (line 1490)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1490, 11), 'iV', tuple_var_assignment_29758_32462)
    
    # Assigning a Name to a Name (line 1490):
    # Getting the type of 'tuple_var_assignment_29759' (line 1490)
    tuple_var_assignment_29759_32463 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1490, 4), 'tuple_var_assignment_29759')
    # Assigning a type to the variable 'iS' (line 1490)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1490, 15), 'iS', tuple_var_assignment_29759_32463)
    
    # Assigning a Name to a Name (line 1490):
    # Getting the type of 'tuple_var_assignment_29760' (line 1490)
    tuple_var_assignment_29760_32464 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1490, 4), 'tuple_var_assignment_29760')
    # Assigning a type to the variable 'w' (line 1490)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1490, 19), 'w', tuple_var_assignment_29760_32464)
    
    # Assigning a Name to a Name (line 1490):
    # Getting the type of 'tuple_var_assignment_29761' (line 1490)
    tuple_var_assignment_29761_32465 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1490, 4), 'tuple_var_assignment_29761')
    # Assigning a type to the variable 'ier' (line 1490)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1490, 22), 'ier', tuple_var_assignment_29761_32465)
    
    # Getting the type of 'ier' (line 1491)
    ier_32466 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1491, 7), 'ier')
    # Testing the type of an if condition (line 1491)
    if_condition_32467 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1491, 4), ier_32466)
    # Assigning a type to the variable 'if_condition_32467' (line 1491)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1491, 4), 'if_condition_32467', if_condition_32467)
    # SSA begins for if statement (line 1491)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # Getting the type of '_RETCODE_ERROR' (line 1492)
    _RETCODE_ERROR_32468 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1492, 14), '_RETCODE_ERROR')
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 1492, 8), _RETCODE_ERROR_32468, 'raise parameter', BaseException)
    # SSA join for if statement (line 1491)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 1493):
    
    # Assigning a Call to a Name (line 1493):
    
    # Call to reshape(...): (line 1493)
    # Processing the call arguments (line 1493)
    
    # Obtaining an instance of the builtin type 'tuple' (line 1493)
    tuple_32484 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1493, 34), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 1493)
    # Adding element type (line 1493)
    # Getting the type of 'm' (line 1493)
    m_32485 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1493, 34), 'm', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1493, 34), tuple_32484, m_32485)
    # Adding element type (line 1493)
    # Getting the type of 'k' (line 1493)
    k_32486 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1493, 37), 'k', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1493, 34), tuple_32484, k_32486)
    
    # Processing the call keyword arguments (line 1493)
    str_32487 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1493, 47), 'str', 'F')
    keyword_32488 = str_32487
    kwargs_32489 = {'order': keyword_32488}
    
    # Obtaining the type of the subscript
    # Getting the type of 'iU' (line 1493)
    iU_32469 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1493, 10), 'iU', False)
    int_32470 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1493, 13), 'int')
    # Applying the binary operator '-' (line 1493)
    result_sub_32471 = python_operator(stypy.reporting.localization.Localization(__file__, 1493, 10), '-', iU_32469, int_32470)
    
    # Getting the type of 'iU' (line 1493)
    iU_32472 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1493, 15), 'iU', False)
    # Getting the type of 'm' (line 1493)
    m_32473 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1493, 18), 'm', False)
    # Getting the type of 'k' (line 1493)
    k_32474 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1493, 20), 'k', False)
    # Applying the binary operator '*' (line 1493)
    result_mul_32475 = python_operator(stypy.reporting.localization.Localization(__file__, 1493, 18), '*', m_32473, k_32474)
    
    # Applying the binary operator '+' (line 1493)
    result_add_32476 = python_operator(stypy.reporting.localization.Localization(__file__, 1493, 15), '+', iU_32472, result_mul_32475)
    
    int_32477 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1493, 22), 'int')
    # Applying the binary operator '-' (line 1493)
    result_sub_32478 = python_operator(stypy.reporting.localization.Localization(__file__, 1493, 21), '-', result_add_32476, int_32477)
    
    slice_32479 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 1493, 8), result_sub_32471, result_sub_32478, None)
    # Getting the type of 'w' (line 1493)
    w_32480 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1493, 8), 'w', False)
    # Obtaining the member '__getitem__' of a type (line 1493)
    getitem___32481 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1493, 8), w_32480, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1493)
    subscript_call_result_32482 = invoke(stypy.reporting.localization.Localization(__file__, 1493, 8), getitem___32481, slice_32479)
    
    # Obtaining the member 'reshape' of a type (line 1493)
    reshape_32483 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1493, 8), subscript_call_result_32482, 'reshape')
    # Calling reshape(args, kwargs) (line 1493)
    reshape_call_result_32490 = invoke(stypy.reporting.localization.Localization(__file__, 1493, 8), reshape_32483, *[tuple_32484], **kwargs_32489)
    
    # Assigning a type to the variable 'U' (line 1493)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1493, 4), 'U', reshape_call_result_32490)
    
    # Assigning a Call to a Name (line 1494):
    
    # Assigning a Call to a Name (line 1494):
    
    # Call to reshape(...): (line 1494)
    # Processing the call arguments (line 1494)
    
    # Obtaining an instance of the builtin type 'tuple' (line 1494)
    tuple_32506 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1494, 34), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 1494)
    # Adding element type (line 1494)
    # Getting the type of 'n' (line 1494)
    n_32507 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1494, 34), 'n', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1494, 34), tuple_32506, n_32507)
    # Adding element type (line 1494)
    # Getting the type of 'k' (line 1494)
    k_32508 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1494, 37), 'k', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1494, 34), tuple_32506, k_32508)
    
    # Processing the call keyword arguments (line 1494)
    str_32509 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1494, 47), 'str', 'F')
    keyword_32510 = str_32509
    kwargs_32511 = {'order': keyword_32510}
    
    # Obtaining the type of the subscript
    # Getting the type of 'iV' (line 1494)
    iV_32491 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1494, 10), 'iV', False)
    int_32492 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1494, 13), 'int')
    # Applying the binary operator '-' (line 1494)
    result_sub_32493 = python_operator(stypy.reporting.localization.Localization(__file__, 1494, 10), '-', iV_32491, int_32492)
    
    # Getting the type of 'iV' (line 1494)
    iV_32494 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1494, 15), 'iV', False)
    # Getting the type of 'n' (line 1494)
    n_32495 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1494, 18), 'n', False)
    # Getting the type of 'k' (line 1494)
    k_32496 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1494, 20), 'k', False)
    # Applying the binary operator '*' (line 1494)
    result_mul_32497 = python_operator(stypy.reporting.localization.Localization(__file__, 1494, 18), '*', n_32495, k_32496)
    
    # Applying the binary operator '+' (line 1494)
    result_add_32498 = python_operator(stypy.reporting.localization.Localization(__file__, 1494, 15), '+', iV_32494, result_mul_32497)
    
    int_32499 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1494, 22), 'int')
    # Applying the binary operator '-' (line 1494)
    result_sub_32500 = python_operator(stypy.reporting.localization.Localization(__file__, 1494, 21), '-', result_add_32498, int_32499)
    
    slice_32501 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 1494, 8), result_sub_32493, result_sub_32500, None)
    # Getting the type of 'w' (line 1494)
    w_32502 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1494, 8), 'w', False)
    # Obtaining the member '__getitem__' of a type (line 1494)
    getitem___32503 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1494, 8), w_32502, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1494)
    subscript_call_result_32504 = invoke(stypy.reporting.localization.Localization(__file__, 1494, 8), getitem___32503, slice_32501)
    
    # Obtaining the member 'reshape' of a type (line 1494)
    reshape_32505 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1494, 8), subscript_call_result_32504, 'reshape')
    # Calling reshape(args, kwargs) (line 1494)
    reshape_call_result_32512 = invoke(stypy.reporting.localization.Localization(__file__, 1494, 8), reshape_32505, *[tuple_32506], **kwargs_32511)
    
    # Assigning a type to the variable 'V' (line 1494)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1494, 4), 'V', reshape_call_result_32512)
    
    # Assigning a Subscript to a Name (line 1495):
    
    # Assigning a Subscript to a Name (line 1495):
    
    # Obtaining the type of the subscript
    # Getting the type of 'iS' (line 1495)
    iS_32513 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1495, 10), 'iS')
    int_32514 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1495, 13), 'int')
    # Applying the binary operator '-' (line 1495)
    result_sub_32515 = python_operator(stypy.reporting.localization.Localization(__file__, 1495, 10), '-', iS_32513, int_32514)
    
    # Getting the type of 'iS' (line 1495)
    iS_32516 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1495, 15), 'iS')
    # Getting the type of 'k' (line 1495)
    k_32517 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1495, 18), 'k')
    # Applying the binary operator '+' (line 1495)
    result_add_32518 = python_operator(stypy.reporting.localization.Localization(__file__, 1495, 15), '+', iS_32516, k_32517)
    
    int_32519 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1495, 20), 'int')
    # Applying the binary operator '-' (line 1495)
    result_sub_32520 = python_operator(stypy.reporting.localization.Localization(__file__, 1495, 19), '-', result_add_32518, int_32519)
    
    slice_32521 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 1495, 8), result_sub_32515, result_sub_32520, None)
    # Getting the type of 'w' (line 1495)
    w_32522 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1495, 8), 'w')
    # Obtaining the member '__getitem__' of a type (line 1495)
    getitem___32523 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1495, 8), w_32522, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1495)
    subscript_call_result_32524 = invoke(stypy.reporting.localization.Localization(__file__, 1495, 8), getitem___32523, slice_32521)
    
    # Assigning a type to the variable 'S' (line 1495)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1495, 4), 'S', subscript_call_result_32524)
    
    # Obtaining an instance of the builtin type 'tuple' (line 1496)
    tuple_32525 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1496, 11), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 1496)
    # Adding element type (line 1496)
    # Getting the type of 'U' (line 1496)
    U_32526 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1496, 11), 'U')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1496, 11), tuple_32525, U_32526)
    # Adding element type (line 1496)
    # Getting the type of 'V' (line 1496)
    V_32527 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1496, 14), 'V')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1496, 11), tuple_32525, V_32527)
    # Adding element type (line 1496)
    # Getting the type of 'S' (line 1496)
    S_32528 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1496, 17), 'S')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1496, 11), tuple_32525, S_32528)
    
    # Assigning a type to the variable 'stypy_return_type' (line 1496)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1496, 4), 'stypy_return_type', tuple_32525)
    
    # ################# End of 'idzp_rsvd(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'idzp_rsvd' in the type store
    # Getting the type of 'stypy_return_type' (line 1455)
    stypy_return_type_32529 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1455, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_32529)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'idzp_rsvd'
    return stypy_return_type_32529

# Assigning a type to the variable 'idzp_rsvd' (line 1455)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1455, 0), 'idzp_rsvd', idzp_rsvd)

@norecursion
def idzr_aid(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'idzr_aid'
    module_type_store = module_type_store.open_function_context('idzr_aid', 1503, 0, False)
    
    # Passed parameters checking function
    idzr_aid.stypy_localization = localization
    idzr_aid.stypy_type_of_self = None
    idzr_aid.stypy_type_store = module_type_store
    idzr_aid.stypy_function_name = 'idzr_aid'
    idzr_aid.stypy_param_names_list = ['A', 'k']
    idzr_aid.stypy_varargs_param_name = None
    idzr_aid.stypy_kwargs_param_name = None
    idzr_aid.stypy_call_defaults = defaults
    idzr_aid.stypy_call_varargs = varargs
    idzr_aid.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'idzr_aid', ['A', 'k'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'idzr_aid', localization, ['A', 'k'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'idzr_aid(...)' code ##################

    str_32530 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1520, (-1)), 'str', '\n    Compute ID of a complex matrix to a specified rank using random sampling.\n\n    :param A:\n        Matrix.\n    :type A: :class:`numpy.ndarray`\n    :param k:\n        Rank of ID.\n    :type k: int\n\n    :return:\n        Column index array.\n    :rtype: :class:`numpy.ndarray`\n    :return:\n        Interpolation coefficients.\n    :rtype: :class:`numpy.ndarray`\n    ')
    
    # Assigning a Call to a Name (line 1521):
    
    # Assigning a Call to a Name (line 1521):
    
    # Call to asfortranarray(...): (line 1521)
    # Processing the call arguments (line 1521)
    # Getting the type of 'A' (line 1521)
    A_32533 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1521, 26), 'A', False)
    # Processing the call keyword arguments (line 1521)
    kwargs_32534 = {}
    # Getting the type of 'np' (line 1521)
    np_32531 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1521, 8), 'np', False)
    # Obtaining the member 'asfortranarray' of a type (line 1521)
    asfortranarray_32532 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1521, 8), np_32531, 'asfortranarray')
    # Calling asfortranarray(args, kwargs) (line 1521)
    asfortranarray_call_result_32535 = invoke(stypy.reporting.localization.Localization(__file__, 1521, 8), asfortranarray_32532, *[A_32533], **kwargs_32534)
    
    # Assigning a type to the variable 'A' (line 1521)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1521, 4), 'A', asfortranarray_call_result_32535)
    
    # Assigning a Attribute to a Tuple (line 1522):
    
    # Assigning a Subscript to a Name (line 1522):
    
    # Obtaining the type of the subscript
    int_32536 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1522, 4), 'int')
    # Getting the type of 'A' (line 1522)
    A_32537 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1522, 11), 'A')
    # Obtaining the member 'shape' of a type (line 1522)
    shape_32538 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1522, 11), A_32537, 'shape')
    # Obtaining the member '__getitem__' of a type (line 1522)
    getitem___32539 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1522, 4), shape_32538, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1522)
    subscript_call_result_32540 = invoke(stypy.reporting.localization.Localization(__file__, 1522, 4), getitem___32539, int_32536)
    
    # Assigning a type to the variable 'tuple_var_assignment_29762' (line 1522)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1522, 4), 'tuple_var_assignment_29762', subscript_call_result_32540)
    
    # Assigning a Subscript to a Name (line 1522):
    
    # Obtaining the type of the subscript
    int_32541 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1522, 4), 'int')
    # Getting the type of 'A' (line 1522)
    A_32542 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1522, 11), 'A')
    # Obtaining the member 'shape' of a type (line 1522)
    shape_32543 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1522, 11), A_32542, 'shape')
    # Obtaining the member '__getitem__' of a type (line 1522)
    getitem___32544 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1522, 4), shape_32543, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1522)
    subscript_call_result_32545 = invoke(stypy.reporting.localization.Localization(__file__, 1522, 4), getitem___32544, int_32541)
    
    # Assigning a type to the variable 'tuple_var_assignment_29763' (line 1522)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1522, 4), 'tuple_var_assignment_29763', subscript_call_result_32545)
    
    # Assigning a Name to a Name (line 1522):
    # Getting the type of 'tuple_var_assignment_29762' (line 1522)
    tuple_var_assignment_29762_32546 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1522, 4), 'tuple_var_assignment_29762')
    # Assigning a type to the variable 'm' (line 1522)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1522, 4), 'm', tuple_var_assignment_29762_32546)
    
    # Assigning a Name to a Name (line 1522):
    # Getting the type of 'tuple_var_assignment_29763' (line 1522)
    tuple_var_assignment_29763_32547 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1522, 4), 'tuple_var_assignment_29763')
    # Assigning a type to the variable 'n' (line 1522)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1522, 7), 'n', tuple_var_assignment_29763_32547)
    
    # Assigning a Call to a Name (line 1523):
    
    # Assigning a Call to a Name (line 1523):
    
    # Call to idzr_aidi(...): (line 1523)
    # Processing the call arguments (line 1523)
    # Getting the type of 'm' (line 1523)
    m_32549 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1523, 18), 'm', False)
    # Getting the type of 'n' (line 1523)
    n_32550 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1523, 21), 'n', False)
    # Getting the type of 'k' (line 1523)
    k_32551 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1523, 24), 'k', False)
    # Processing the call keyword arguments (line 1523)
    kwargs_32552 = {}
    # Getting the type of 'idzr_aidi' (line 1523)
    idzr_aidi_32548 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1523, 8), 'idzr_aidi', False)
    # Calling idzr_aidi(args, kwargs) (line 1523)
    idzr_aidi_call_result_32553 = invoke(stypy.reporting.localization.Localization(__file__, 1523, 8), idzr_aidi_32548, *[m_32549, n_32550, k_32551], **kwargs_32552)
    
    # Assigning a type to the variable 'w' (line 1523)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1523, 4), 'w', idzr_aidi_call_result_32553)
    
    # Assigning a Call to a Tuple (line 1524):
    
    # Assigning a Subscript to a Name (line 1524):
    
    # Obtaining the type of the subscript
    int_32554 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1524, 4), 'int')
    
    # Call to idzr_aid(...): (line 1524)
    # Processing the call arguments (line 1524)
    # Getting the type of 'A' (line 1524)
    A_32557 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1524, 29), 'A', False)
    # Getting the type of 'k' (line 1524)
    k_32558 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1524, 32), 'k', False)
    # Getting the type of 'w' (line 1524)
    w_32559 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1524, 35), 'w', False)
    # Processing the call keyword arguments (line 1524)
    kwargs_32560 = {}
    # Getting the type of '_id' (line 1524)
    _id_32555 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1524, 16), '_id', False)
    # Obtaining the member 'idzr_aid' of a type (line 1524)
    idzr_aid_32556 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1524, 16), _id_32555, 'idzr_aid')
    # Calling idzr_aid(args, kwargs) (line 1524)
    idzr_aid_call_result_32561 = invoke(stypy.reporting.localization.Localization(__file__, 1524, 16), idzr_aid_32556, *[A_32557, k_32558, w_32559], **kwargs_32560)
    
    # Obtaining the member '__getitem__' of a type (line 1524)
    getitem___32562 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1524, 4), idzr_aid_call_result_32561, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1524)
    subscript_call_result_32563 = invoke(stypy.reporting.localization.Localization(__file__, 1524, 4), getitem___32562, int_32554)
    
    # Assigning a type to the variable 'tuple_var_assignment_29764' (line 1524)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1524, 4), 'tuple_var_assignment_29764', subscript_call_result_32563)
    
    # Assigning a Subscript to a Name (line 1524):
    
    # Obtaining the type of the subscript
    int_32564 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1524, 4), 'int')
    
    # Call to idzr_aid(...): (line 1524)
    # Processing the call arguments (line 1524)
    # Getting the type of 'A' (line 1524)
    A_32567 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1524, 29), 'A', False)
    # Getting the type of 'k' (line 1524)
    k_32568 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1524, 32), 'k', False)
    # Getting the type of 'w' (line 1524)
    w_32569 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1524, 35), 'w', False)
    # Processing the call keyword arguments (line 1524)
    kwargs_32570 = {}
    # Getting the type of '_id' (line 1524)
    _id_32565 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1524, 16), '_id', False)
    # Obtaining the member 'idzr_aid' of a type (line 1524)
    idzr_aid_32566 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1524, 16), _id_32565, 'idzr_aid')
    # Calling idzr_aid(args, kwargs) (line 1524)
    idzr_aid_call_result_32571 = invoke(stypy.reporting.localization.Localization(__file__, 1524, 16), idzr_aid_32566, *[A_32567, k_32568, w_32569], **kwargs_32570)
    
    # Obtaining the member '__getitem__' of a type (line 1524)
    getitem___32572 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1524, 4), idzr_aid_call_result_32571, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1524)
    subscript_call_result_32573 = invoke(stypy.reporting.localization.Localization(__file__, 1524, 4), getitem___32572, int_32564)
    
    # Assigning a type to the variable 'tuple_var_assignment_29765' (line 1524)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1524, 4), 'tuple_var_assignment_29765', subscript_call_result_32573)
    
    # Assigning a Name to a Name (line 1524):
    # Getting the type of 'tuple_var_assignment_29764' (line 1524)
    tuple_var_assignment_29764_32574 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1524, 4), 'tuple_var_assignment_29764')
    # Assigning a type to the variable 'idx' (line 1524)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1524, 4), 'idx', tuple_var_assignment_29764_32574)
    
    # Assigning a Name to a Name (line 1524):
    # Getting the type of 'tuple_var_assignment_29765' (line 1524)
    tuple_var_assignment_29765_32575 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1524, 4), 'tuple_var_assignment_29765')
    # Assigning a type to the variable 'proj' (line 1524)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1524, 9), 'proj', tuple_var_assignment_29765_32575)
    
    
    # Getting the type of 'k' (line 1525)
    k_32576 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1525, 7), 'k')
    # Getting the type of 'n' (line 1525)
    n_32577 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1525, 12), 'n')
    # Applying the binary operator '==' (line 1525)
    result_eq_32578 = python_operator(stypy.reporting.localization.Localization(__file__, 1525, 7), '==', k_32576, n_32577)
    
    # Testing the type of an if condition (line 1525)
    if_condition_32579 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1525, 4), result_eq_32578)
    # Assigning a type to the variable 'if_condition_32579' (line 1525)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1525, 4), 'if_condition_32579', if_condition_32579)
    # SSA begins for if statement (line 1525)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 1526):
    
    # Assigning a Call to a Name (line 1526):
    
    # Call to array(...): (line 1526)
    # Processing the call arguments (line 1526)
    
    # Obtaining an instance of the builtin type 'list' (line 1526)
    list_32582 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1526, 24), 'list')
    # Adding type elements to the builtin type 'list' instance (line 1526)
    
    # Processing the call keyword arguments (line 1526)
    str_32583 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1526, 34), 'str', 'complex128')
    keyword_32584 = str_32583
    str_32585 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1526, 54), 'str', 'F')
    keyword_32586 = str_32585
    kwargs_32587 = {'dtype': keyword_32584, 'order': keyword_32586}
    # Getting the type of 'np' (line 1526)
    np_32580 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1526, 15), 'np', False)
    # Obtaining the member 'array' of a type (line 1526)
    array_32581 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1526, 15), np_32580, 'array')
    # Calling array(args, kwargs) (line 1526)
    array_call_result_32588 = invoke(stypy.reporting.localization.Localization(__file__, 1526, 15), array_32581, *[list_32582], **kwargs_32587)
    
    # Assigning a type to the variable 'proj' (line 1526)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1526, 8), 'proj', array_call_result_32588)
    # SSA branch for the else part of an if statement (line 1525)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Call to a Name (line 1528):
    
    # Assigning a Call to a Name (line 1528):
    
    # Call to reshape(...): (line 1528)
    # Processing the call arguments (line 1528)
    
    # Obtaining an instance of the builtin type 'tuple' (line 1528)
    tuple_32591 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1528, 29), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 1528)
    # Adding element type (line 1528)
    # Getting the type of 'k' (line 1528)
    k_32592 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1528, 29), 'k', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1528, 29), tuple_32591, k_32592)
    # Adding element type (line 1528)
    # Getting the type of 'n' (line 1528)
    n_32593 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1528, 32), 'n', False)
    # Getting the type of 'k' (line 1528)
    k_32594 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1528, 34), 'k', False)
    # Applying the binary operator '-' (line 1528)
    result_sub_32595 = python_operator(stypy.reporting.localization.Localization(__file__, 1528, 32), '-', n_32593, k_32594)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1528, 29), tuple_32591, result_sub_32595)
    
    # Processing the call keyword arguments (line 1528)
    str_32596 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1528, 44), 'str', 'F')
    keyword_32597 = str_32596
    kwargs_32598 = {'order': keyword_32597}
    # Getting the type of 'proj' (line 1528)
    proj_32589 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1528, 15), 'proj', False)
    # Obtaining the member 'reshape' of a type (line 1528)
    reshape_32590 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1528, 15), proj_32589, 'reshape')
    # Calling reshape(args, kwargs) (line 1528)
    reshape_call_result_32599 = invoke(stypy.reporting.localization.Localization(__file__, 1528, 15), reshape_32590, *[tuple_32591], **kwargs_32598)
    
    # Assigning a type to the variable 'proj' (line 1528)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1528, 8), 'proj', reshape_call_result_32599)
    # SSA join for if statement (line 1525)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Obtaining an instance of the builtin type 'tuple' (line 1529)
    tuple_32600 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1529, 11), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 1529)
    # Adding element type (line 1529)
    # Getting the type of 'idx' (line 1529)
    idx_32601 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1529, 11), 'idx')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1529, 11), tuple_32600, idx_32601)
    # Adding element type (line 1529)
    # Getting the type of 'proj' (line 1529)
    proj_32602 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1529, 16), 'proj')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1529, 11), tuple_32600, proj_32602)
    
    # Assigning a type to the variable 'stypy_return_type' (line 1529)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1529, 4), 'stypy_return_type', tuple_32600)
    
    # ################# End of 'idzr_aid(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'idzr_aid' in the type store
    # Getting the type of 'stypy_return_type' (line 1503)
    stypy_return_type_32603 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1503, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_32603)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'idzr_aid'
    return stypy_return_type_32603

# Assigning a type to the variable 'idzr_aid' (line 1503)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1503, 0), 'idzr_aid', idzr_aid)

@norecursion
def idzr_aidi(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'idzr_aidi'
    module_type_store = module_type_store.open_function_context('idzr_aidi', 1532, 0, False)
    
    # Passed parameters checking function
    idzr_aidi.stypy_localization = localization
    idzr_aidi.stypy_type_of_self = None
    idzr_aidi.stypy_type_store = module_type_store
    idzr_aidi.stypy_function_name = 'idzr_aidi'
    idzr_aidi.stypy_param_names_list = ['m', 'n', 'k']
    idzr_aidi.stypy_varargs_param_name = None
    idzr_aidi.stypy_kwargs_param_name = None
    idzr_aidi.stypy_call_defaults = defaults
    idzr_aidi.stypy_call_varargs = varargs
    idzr_aidi.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'idzr_aidi', ['m', 'n', 'k'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'idzr_aidi', localization, ['m', 'n', 'k'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'idzr_aidi(...)' code ##################

    str_32604 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1549, (-1)), 'str', '\n    Initialize array for :func:`idzr_aid`.\n\n    :param m:\n        Matrix row dimension.\n    :type m: int\n    :param n:\n        Matrix column dimension.\n    :type n: int\n    :param k:\n        Rank of ID.\n    :type k: int\n\n    :return:\n        Initialization array to be used by :func:`idzr_aid`.\n    :rtype: :class:`numpy.ndarray`\n    ')
    
    # Call to idzr_aidi(...): (line 1550)
    # Processing the call arguments (line 1550)
    # Getting the type of 'm' (line 1550)
    m_32607 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1550, 25), 'm', False)
    # Getting the type of 'n' (line 1550)
    n_32608 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1550, 28), 'n', False)
    # Getting the type of 'k' (line 1550)
    k_32609 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1550, 31), 'k', False)
    # Processing the call keyword arguments (line 1550)
    kwargs_32610 = {}
    # Getting the type of '_id' (line 1550)
    _id_32605 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1550, 11), '_id', False)
    # Obtaining the member 'idzr_aidi' of a type (line 1550)
    idzr_aidi_32606 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1550, 11), _id_32605, 'idzr_aidi')
    # Calling idzr_aidi(args, kwargs) (line 1550)
    idzr_aidi_call_result_32611 = invoke(stypy.reporting.localization.Localization(__file__, 1550, 11), idzr_aidi_32606, *[m_32607, n_32608, k_32609], **kwargs_32610)
    
    # Assigning a type to the variable 'stypy_return_type' (line 1550)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1550, 4), 'stypy_return_type', idzr_aidi_call_result_32611)
    
    # ################# End of 'idzr_aidi(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'idzr_aidi' in the type store
    # Getting the type of 'stypy_return_type' (line 1532)
    stypy_return_type_32612 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1532, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_32612)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'idzr_aidi'
    return stypy_return_type_32612

# Assigning a type to the variable 'idzr_aidi' (line 1532)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1532, 0), 'idzr_aidi', idzr_aidi)

@norecursion
def idzr_asvd(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'idzr_asvd'
    module_type_store = module_type_store.open_function_context('idzr_asvd', 1557, 0, False)
    
    # Passed parameters checking function
    idzr_asvd.stypy_localization = localization
    idzr_asvd.stypy_type_of_self = None
    idzr_asvd.stypy_type_store = module_type_store
    idzr_asvd.stypy_function_name = 'idzr_asvd'
    idzr_asvd.stypy_param_names_list = ['A', 'k']
    idzr_asvd.stypy_varargs_param_name = None
    idzr_asvd.stypy_kwargs_param_name = None
    idzr_asvd.stypy_call_defaults = defaults
    idzr_asvd.stypy_call_varargs = varargs
    idzr_asvd.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'idzr_asvd', ['A', 'k'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'idzr_asvd', localization, ['A', 'k'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'idzr_asvd(...)' code ##################

    str_32613 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1577, (-1)), 'str', '\n    Compute SVD of a complex matrix to a specified rank using random sampling.\n\n    :param A:\n        Matrix.\n    :type A: :class:`numpy.ndarray`\n    :param k:\n        Rank of SVD.\n    :type k: int\n\n    :return:\n        Left singular vectors.\n    :rtype: :class:`numpy.ndarray`\n    :return:\n        Right singular vectors.\n    :rtype: :class:`numpy.ndarray`\n    :return:\n        Singular values.\n    :rtype: :class:`numpy.ndarray`\n    ')
    
    # Assigning a Call to a Name (line 1578):
    
    # Assigning a Call to a Name (line 1578):
    
    # Call to asfortranarray(...): (line 1578)
    # Processing the call arguments (line 1578)
    # Getting the type of 'A' (line 1578)
    A_32616 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1578, 26), 'A', False)
    # Processing the call keyword arguments (line 1578)
    kwargs_32617 = {}
    # Getting the type of 'np' (line 1578)
    np_32614 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1578, 8), 'np', False)
    # Obtaining the member 'asfortranarray' of a type (line 1578)
    asfortranarray_32615 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1578, 8), np_32614, 'asfortranarray')
    # Calling asfortranarray(args, kwargs) (line 1578)
    asfortranarray_call_result_32618 = invoke(stypy.reporting.localization.Localization(__file__, 1578, 8), asfortranarray_32615, *[A_32616], **kwargs_32617)
    
    # Assigning a type to the variable 'A' (line 1578)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1578, 4), 'A', asfortranarray_call_result_32618)
    
    # Assigning a Attribute to a Tuple (line 1579):
    
    # Assigning a Subscript to a Name (line 1579):
    
    # Obtaining the type of the subscript
    int_32619 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1579, 4), 'int')
    # Getting the type of 'A' (line 1579)
    A_32620 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1579, 11), 'A')
    # Obtaining the member 'shape' of a type (line 1579)
    shape_32621 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1579, 11), A_32620, 'shape')
    # Obtaining the member '__getitem__' of a type (line 1579)
    getitem___32622 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1579, 4), shape_32621, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1579)
    subscript_call_result_32623 = invoke(stypy.reporting.localization.Localization(__file__, 1579, 4), getitem___32622, int_32619)
    
    # Assigning a type to the variable 'tuple_var_assignment_29766' (line 1579)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1579, 4), 'tuple_var_assignment_29766', subscript_call_result_32623)
    
    # Assigning a Subscript to a Name (line 1579):
    
    # Obtaining the type of the subscript
    int_32624 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1579, 4), 'int')
    # Getting the type of 'A' (line 1579)
    A_32625 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1579, 11), 'A')
    # Obtaining the member 'shape' of a type (line 1579)
    shape_32626 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1579, 11), A_32625, 'shape')
    # Obtaining the member '__getitem__' of a type (line 1579)
    getitem___32627 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1579, 4), shape_32626, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1579)
    subscript_call_result_32628 = invoke(stypy.reporting.localization.Localization(__file__, 1579, 4), getitem___32627, int_32624)
    
    # Assigning a type to the variable 'tuple_var_assignment_29767' (line 1579)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1579, 4), 'tuple_var_assignment_29767', subscript_call_result_32628)
    
    # Assigning a Name to a Name (line 1579):
    # Getting the type of 'tuple_var_assignment_29766' (line 1579)
    tuple_var_assignment_29766_32629 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1579, 4), 'tuple_var_assignment_29766')
    # Assigning a type to the variable 'm' (line 1579)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1579, 4), 'm', tuple_var_assignment_29766_32629)
    
    # Assigning a Name to a Name (line 1579):
    # Getting the type of 'tuple_var_assignment_29767' (line 1579)
    tuple_var_assignment_29767_32630 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1579, 4), 'tuple_var_assignment_29767')
    # Assigning a type to the variable 'n' (line 1579)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1579, 7), 'n', tuple_var_assignment_29767_32630)
    
    # Assigning a Call to a Name (line 1580):
    
    # Assigning a Call to a Name (line 1580):
    
    # Call to empty(...): (line 1580)
    # Processing the call arguments (line 1580)
    int_32633 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1581, 9), 'int')
    # Getting the type of 'k' (line 1581)
    k_32634 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1581, 11), 'k', False)
    # Applying the binary operator '*' (line 1581)
    result_mul_32635 = python_operator(stypy.reporting.localization.Localization(__file__, 1581, 9), '*', int_32633, k_32634)
    
    int_32636 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1581, 15), 'int')
    # Applying the binary operator '+' (line 1581)
    result_add_32637 = python_operator(stypy.reporting.localization.Localization(__file__, 1581, 9), '+', result_mul_32635, int_32636)
    
    # Getting the type of 'm' (line 1581)
    m_32638 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1581, 19), 'm', False)
    # Applying the binary operator '*' (line 1581)
    result_mul_32639 = python_operator(stypy.reporting.localization.Localization(__file__, 1581, 8), '*', result_add_32637, m_32638)
    
    int_32640 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1581, 24), 'int')
    # Getting the type of 'k' (line 1581)
    k_32641 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1581, 26), 'k', False)
    # Applying the binary operator '*' (line 1581)
    result_mul_32642 = python_operator(stypy.reporting.localization.Localization(__file__, 1581, 24), '*', int_32640, k_32641)
    
    int_32643 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1581, 30), 'int')
    # Applying the binary operator '+' (line 1581)
    result_add_32644 = python_operator(stypy.reporting.localization.Localization(__file__, 1581, 24), '+', result_mul_32642, int_32643)
    
    # Getting the type of 'n' (line 1581)
    n_32645 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1581, 34), 'n', False)
    # Applying the binary operator '*' (line 1581)
    result_mul_32646 = python_operator(stypy.reporting.localization.Localization(__file__, 1581, 23), '*', result_add_32644, n_32645)
    
    # Applying the binary operator '+' (line 1581)
    result_add_32647 = python_operator(stypy.reporting.localization.Localization(__file__, 1581, 8), '+', result_mul_32639, result_mul_32646)
    
    int_32648 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1581, 38), 'int')
    # Getting the type of 'k' (line 1581)
    k_32649 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1581, 40), 'k', False)
    int_32650 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1581, 43), 'int')
    # Applying the binary operator '**' (line 1581)
    result_pow_32651 = python_operator(stypy.reporting.localization.Localization(__file__, 1581, 40), '**', k_32649, int_32650)
    
    # Applying the binary operator '*' (line 1581)
    result_mul_32652 = python_operator(stypy.reporting.localization.Localization(__file__, 1581, 38), '*', int_32648, result_pow_32651)
    
    # Applying the binary operator '+' (line 1581)
    result_add_32653 = python_operator(stypy.reporting.localization.Localization(__file__, 1581, 36), '+', result_add_32647, result_mul_32652)
    
    int_32654 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1581, 47), 'int')
    # Getting the type of 'k' (line 1581)
    k_32655 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1581, 50), 'k', False)
    # Applying the binary operator '*' (line 1581)
    result_mul_32656 = python_operator(stypy.reporting.localization.Localization(__file__, 1581, 47), '*', int_32654, k_32655)
    
    # Applying the binary operator '+' (line 1581)
    result_add_32657 = python_operator(stypy.reporting.localization.Localization(__file__, 1581, 45), '+', result_add_32653, result_mul_32656)
    
    int_32658 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1581, 54), 'int')
    # Applying the binary operator '+' (line 1581)
    result_add_32659 = python_operator(stypy.reporting.localization.Localization(__file__, 1581, 52), '+', result_add_32657, int_32658)
    
    # Processing the call keyword arguments (line 1580)
    str_32660 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1582, 14), 'str', 'complex128')
    keyword_32661 = str_32660
    str_32662 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1582, 34), 'str', 'F')
    keyword_32663 = str_32662
    kwargs_32664 = {'dtype': keyword_32661, 'order': keyword_32663}
    # Getting the type of 'np' (line 1580)
    np_32631 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1580, 8), 'np', False)
    # Obtaining the member 'empty' of a type (line 1580)
    empty_32632 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1580, 8), np_32631, 'empty')
    # Calling empty(args, kwargs) (line 1580)
    empty_call_result_32665 = invoke(stypy.reporting.localization.Localization(__file__, 1580, 8), empty_32632, *[result_add_32659], **kwargs_32664)
    
    # Assigning a type to the variable 'w' (line 1580)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1580, 4), 'w', empty_call_result_32665)
    
    # Assigning a Call to a Name (line 1583):
    
    # Assigning a Call to a Name (line 1583):
    
    # Call to idzr_aidi(...): (line 1583)
    # Processing the call arguments (line 1583)
    # Getting the type of 'm' (line 1583)
    m_32667 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1583, 19), 'm', False)
    # Getting the type of 'n' (line 1583)
    n_32668 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1583, 22), 'n', False)
    # Getting the type of 'k' (line 1583)
    k_32669 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1583, 25), 'k', False)
    # Processing the call keyword arguments (line 1583)
    kwargs_32670 = {}
    # Getting the type of 'idzr_aidi' (line 1583)
    idzr_aidi_32666 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1583, 9), 'idzr_aidi', False)
    # Calling idzr_aidi(args, kwargs) (line 1583)
    idzr_aidi_call_result_32671 = invoke(stypy.reporting.localization.Localization(__file__, 1583, 9), idzr_aidi_32666, *[m_32667, n_32668, k_32669], **kwargs_32670)
    
    # Assigning a type to the variable 'w_' (line 1583)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1583, 4), 'w_', idzr_aidi_call_result_32671)
    
    # Assigning a Name to a Subscript (line 1584):
    
    # Assigning a Name to a Subscript (line 1584):
    # Getting the type of 'w_' (line 1584)
    w__32672 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1584, 18), 'w_')
    # Getting the type of 'w' (line 1584)
    w_32673 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1584, 4), 'w')
    # Getting the type of 'w_' (line 1584)
    w__32674 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1584, 7), 'w_')
    # Obtaining the member 'size' of a type (line 1584)
    size_32675 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1584, 7), w__32674, 'size')
    slice_32676 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 1584, 4), None, size_32675, None)
    # Storing an element on a container (line 1584)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1584, 4), w_32673, (slice_32676, w__32672))
    
    # Assigning a Call to a Tuple (line 1585):
    
    # Assigning a Subscript to a Name (line 1585):
    
    # Obtaining the type of the subscript
    int_32677 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1585, 4), 'int')
    
    # Call to idzr_asvd(...): (line 1585)
    # Processing the call arguments (line 1585)
    # Getting the type of 'A' (line 1585)
    A_32680 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1585, 33), 'A', False)
    # Getting the type of 'k' (line 1585)
    k_32681 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1585, 36), 'k', False)
    # Getting the type of 'w' (line 1585)
    w_32682 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1585, 39), 'w', False)
    # Processing the call keyword arguments (line 1585)
    kwargs_32683 = {}
    # Getting the type of '_id' (line 1585)
    _id_32678 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1585, 19), '_id', False)
    # Obtaining the member 'idzr_asvd' of a type (line 1585)
    idzr_asvd_32679 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1585, 19), _id_32678, 'idzr_asvd')
    # Calling idzr_asvd(args, kwargs) (line 1585)
    idzr_asvd_call_result_32684 = invoke(stypy.reporting.localization.Localization(__file__, 1585, 19), idzr_asvd_32679, *[A_32680, k_32681, w_32682], **kwargs_32683)
    
    # Obtaining the member '__getitem__' of a type (line 1585)
    getitem___32685 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1585, 4), idzr_asvd_call_result_32684, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1585)
    subscript_call_result_32686 = invoke(stypy.reporting.localization.Localization(__file__, 1585, 4), getitem___32685, int_32677)
    
    # Assigning a type to the variable 'tuple_var_assignment_29768' (line 1585)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1585, 4), 'tuple_var_assignment_29768', subscript_call_result_32686)
    
    # Assigning a Subscript to a Name (line 1585):
    
    # Obtaining the type of the subscript
    int_32687 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1585, 4), 'int')
    
    # Call to idzr_asvd(...): (line 1585)
    # Processing the call arguments (line 1585)
    # Getting the type of 'A' (line 1585)
    A_32690 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1585, 33), 'A', False)
    # Getting the type of 'k' (line 1585)
    k_32691 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1585, 36), 'k', False)
    # Getting the type of 'w' (line 1585)
    w_32692 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1585, 39), 'w', False)
    # Processing the call keyword arguments (line 1585)
    kwargs_32693 = {}
    # Getting the type of '_id' (line 1585)
    _id_32688 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1585, 19), '_id', False)
    # Obtaining the member 'idzr_asvd' of a type (line 1585)
    idzr_asvd_32689 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1585, 19), _id_32688, 'idzr_asvd')
    # Calling idzr_asvd(args, kwargs) (line 1585)
    idzr_asvd_call_result_32694 = invoke(stypy.reporting.localization.Localization(__file__, 1585, 19), idzr_asvd_32689, *[A_32690, k_32691, w_32692], **kwargs_32693)
    
    # Obtaining the member '__getitem__' of a type (line 1585)
    getitem___32695 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1585, 4), idzr_asvd_call_result_32694, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1585)
    subscript_call_result_32696 = invoke(stypy.reporting.localization.Localization(__file__, 1585, 4), getitem___32695, int_32687)
    
    # Assigning a type to the variable 'tuple_var_assignment_29769' (line 1585)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1585, 4), 'tuple_var_assignment_29769', subscript_call_result_32696)
    
    # Assigning a Subscript to a Name (line 1585):
    
    # Obtaining the type of the subscript
    int_32697 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1585, 4), 'int')
    
    # Call to idzr_asvd(...): (line 1585)
    # Processing the call arguments (line 1585)
    # Getting the type of 'A' (line 1585)
    A_32700 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1585, 33), 'A', False)
    # Getting the type of 'k' (line 1585)
    k_32701 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1585, 36), 'k', False)
    # Getting the type of 'w' (line 1585)
    w_32702 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1585, 39), 'w', False)
    # Processing the call keyword arguments (line 1585)
    kwargs_32703 = {}
    # Getting the type of '_id' (line 1585)
    _id_32698 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1585, 19), '_id', False)
    # Obtaining the member 'idzr_asvd' of a type (line 1585)
    idzr_asvd_32699 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1585, 19), _id_32698, 'idzr_asvd')
    # Calling idzr_asvd(args, kwargs) (line 1585)
    idzr_asvd_call_result_32704 = invoke(stypy.reporting.localization.Localization(__file__, 1585, 19), idzr_asvd_32699, *[A_32700, k_32701, w_32702], **kwargs_32703)
    
    # Obtaining the member '__getitem__' of a type (line 1585)
    getitem___32705 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1585, 4), idzr_asvd_call_result_32704, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1585)
    subscript_call_result_32706 = invoke(stypy.reporting.localization.Localization(__file__, 1585, 4), getitem___32705, int_32697)
    
    # Assigning a type to the variable 'tuple_var_assignment_29770' (line 1585)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1585, 4), 'tuple_var_assignment_29770', subscript_call_result_32706)
    
    # Assigning a Subscript to a Name (line 1585):
    
    # Obtaining the type of the subscript
    int_32707 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1585, 4), 'int')
    
    # Call to idzr_asvd(...): (line 1585)
    # Processing the call arguments (line 1585)
    # Getting the type of 'A' (line 1585)
    A_32710 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1585, 33), 'A', False)
    # Getting the type of 'k' (line 1585)
    k_32711 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1585, 36), 'k', False)
    # Getting the type of 'w' (line 1585)
    w_32712 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1585, 39), 'w', False)
    # Processing the call keyword arguments (line 1585)
    kwargs_32713 = {}
    # Getting the type of '_id' (line 1585)
    _id_32708 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1585, 19), '_id', False)
    # Obtaining the member 'idzr_asvd' of a type (line 1585)
    idzr_asvd_32709 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1585, 19), _id_32708, 'idzr_asvd')
    # Calling idzr_asvd(args, kwargs) (line 1585)
    idzr_asvd_call_result_32714 = invoke(stypy.reporting.localization.Localization(__file__, 1585, 19), idzr_asvd_32709, *[A_32710, k_32711, w_32712], **kwargs_32713)
    
    # Obtaining the member '__getitem__' of a type (line 1585)
    getitem___32715 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1585, 4), idzr_asvd_call_result_32714, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1585)
    subscript_call_result_32716 = invoke(stypy.reporting.localization.Localization(__file__, 1585, 4), getitem___32715, int_32707)
    
    # Assigning a type to the variable 'tuple_var_assignment_29771' (line 1585)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1585, 4), 'tuple_var_assignment_29771', subscript_call_result_32716)
    
    # Assigning a Name to a Name (line 1585):
    # Getting the type of 'tuple_var_assignment_29768' (line 1585)
    tuple_var_assignment_29768_32717 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1585, 4), 'tuple_var_assignment_29768')
    # Assigning a type to the variable 'U' (line 1585)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1585, 4), 'U', tuple_var_assignment_29768_32717)
    
    # Assigning a Name to a Name (line 1585):
    # Getting the type of 'tuple_var_assignment_29769' (line 1585)
    tuple_var_assignment_29769_32718 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1585, 4), 'tuple_var_assignment_29769')
    # Assigning a type to the variable 'V' (line 1585)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1585, 7), 'V', tuple_var_assignment_29769_32718)
    
    # Assigning a Name to a Name (line 1585):
    # Getting the type of 'tuple_var_assignment_29770' (line 1585)
    tuple_var_assignment_29770_32719 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1585, 4), 'tuple_var_assignment_29770')
    # Assigning a type to the variable 'S' (line 1585)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1585, 10), 'S', tuple_var_assignment_29770_32719)
    
    # Assigning a Name to a Name (line 1585):
    # Getting the type of 'tuple_var_assignment_29771' (line 1585)
    tuple_var_assignment_29771_32720 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1585, 4), 'tuple_var_assignment_29771')
    # Assigning a type to the variable 'ier' (line 1585)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1585, 13), 'ier', tuple_var_assignment_29771_32720)
    
    # Getting the type of 'ier' (line 1586)
    ier_32721 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1586, 7), 'ier')
    # Testing the type of an if condition (line 1586)
    if_condition_32722 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1586, 4), ier_32721)
    # Assigning a type to the variable 'if_condition_32722' (line 1586)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1586, 4), 'if_condition_32722', if_condition_32722)
    # SSA begins for if statement (line 1586)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # Getting the type of '_RETCODE_ERROR' (line 1587)
    _RETCODE_ERROR_32723 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1587, 14), '_RETCODE_ERROR')
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 1587, 8), _RETCODE_ERROR_32723, 'raise parameter', BaseException)
    # SSA join for if statement (line 1586)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Obtaining an instance of the builtin type 'tuple' (line 1588)
    tuple_32724 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1588, 11), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 1588)
    # Adding element type (line 1588)
    # Getting the type of 'U' (line 1588)
    U_32725 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1588, 11), 'U')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1588, 11), tuple_32724, U_32725)
    # Adding element type (line 1588)
    # Getting the type of 'V' (line 1588)
    V_32726 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1588, 14), 'V')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1588, 11), tuple_32724, V_32726)
    # Adding element type (line 1588)
    # Getting the type of 'S' (line 1588)
    S_32727 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1588, 17), 'S')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1588, 11), tuple_32724, S_32727)
    
    # Assigning a type to the variable 'stypy_return_type' (line 1588)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1588, 4), 'stypy_return_type', tuple_32724)
    
    # ################# End of 'idzr_asvd(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'idzr_asvd' in the type store
    # Getting the type of 'stypy_return_type' (line 1557)
    stypy_return_type_32728 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1557, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_32728)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'idzr_asvd'
    return stypy_return_type_32728

# Assigning a type to the variable 'idzr_asvd' (line 1557)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1557, 0), 'idzr_asvd', idzr_asvd)

@norecursion
def idzr_rid(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'idzr_rid'
    module_type_store = module_type_store.open_function_context('idzr_rid', 1595, 0, False)
    
    # Passed parameters checking function
    idzr_rid.stypy_localization = localization
    idzr_rid.stypy_type_of_self = None
    idzr_rid.stypy_type_store = module_type_store
    idzr_rid.stypy_function_name = 'idzr_rid'
    idzr_rid.stypy_param_names_list = ['m', 'n', 'matveca', 'k']
    idzr_rid.stypy_varargs_param_name = None
    idzr_rid.stypy_kwargs_param_name = None
    idzr_rid.stypy_call_defaults = defaults
    idzr_rid.stypy_call_varargs = varargs
    idzr_rid.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'idzr_rid', ['m', 'n', 'matveca', 'k'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'idzr_rid', localization, ['m', 'n', 'matveca', 'k'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'idzr_rid(...)' code ##################

    str_32729 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1621, (-1)), 'str', '\n    Compute ID of a complex matrix to a specified rank using random\n    matrix-vector multiplication.\n\n    :param m:\n        Matrix row dimension.\n    :type m: int\n    :param n:\n        Matrix column dimension.\n    :type n: int\n    :param matveca:\n        Function to apply the matrix adjoint to a vector, with call signature\n        `y = matveca(x)`, where `x` and `y` are the input and output vectors,\n        respectively.\n    :type matveca: function\n    :param k:\n        Rank of ID.\n    :type k: int\n\n    :return:\n        Column index array.\n    :rtype: :class:`numpy.ndarray`\n    :return:\n        Interpolation coefficients.\n    :rtype: :class:`numpy.ndarray`\n    ')
    
    # Assigning a Call to a Tuple (line 1622):
    
    # Assigning a Subscript to a Name (line 1622):
    
    # Obtaining the type of the subscript
    int_32730 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1622, 4), 'int')
    
    # Call to idzr_rid(...): (line 1622)
    # Processing the call arguments (line 1622)
    # Getting the type of 'm' (line 1622)
    m_32733 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1622, 29), 'm', False)
    # Getting the type of 'n' (line 1622)
    n_32734 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1622, 32), 'n', False)
    # Getting the type of 'matveca' (line 1622)
    matveca_32735 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1622, 35), 'matveca', False)
    # Getting the type of 'k' (line 1622)
    k_32736 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1622, 44), 'k', False)
    # Processing the call keyword arguments (line 1622)
    kwargs_32737 = {}
    # Getting the type of '_id' (line 1622)
    _id_32731 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1622, 16), '_id', False)
    # Obtaining the member 'idzr_rid' of a type (line 1622)
    idzr_rid_32732 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1622, 16), _id_32731, 'idzr_rid')
    # Calling idzr_rid(args, kwargs) (line 1622)
    idzr_rid_call_result_32738 = invoke(stypy.reporting.localization.Localization(__file__, 1622, 16), idzr_rid_32732, *[m_32733, n_32734, matveca_32735, k_32736], **kwargs_32737)
    
    # Obtaining the member '__getitem__' of a type (line 1622)
    getitem___32739 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1622, 4), idzr_rid_call_result_32738, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1622)
    subscript_call_result_32740 = invoke(stypy.reporting.localization.Localization(__file__, 1622, 4), getitem___32739, int_32730)
    
    # Assigning a type to the variable 'tuple_var_assignment_29772' (line 1622)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1622, 4), 'tuple_var_assignment_29772', subscript_call_result_32740)
    
    # Assigning a Subscript to a Name (line 1622):
    
    # Obtaining the type of the subscript
    int_32741 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1622, 4), 'int')
    
    # Call to idzr_rid(...): (line 1622)
    # Processing the call arguments (line 1622)
    # Getting the type of 'm' (line 1622)
    m_32744 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1622, 29), 'm', False)
    # Getting the type of 'n' (line 1622)
    n_32745 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1622, 32), 'n', False)
    # Getting the type of 'matveca' (line 1622)
    matveca_32746 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1622, 35), 'matveca', False)
    # Getting the type of 'k' (line 1622)
    k_32747 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1622, 44), 'k', False)
    # Processing the call keyword arguments (line 1622)
    kwargs_32748 = {}
    # Getting the type of '_id' (line 1622)
    _id_32742 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1622, 16), '_id', False)
    # Obtaining the member 'idzr_rid' of a type (line 1622)
    idzr_rid_32743 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1622, 16), _id_32742, 'idzr_rid')
    # Calling idzr_rid(args, kwargs) (line 1622)
    idzr_rid_call_result_32749 = invoke(stypy.reporting.localization.Localization(__file__, 1622, 16), idzr_rid_32743, *[m_32744, n_32745, matveca_32746, k_32747], **kwargs_32748)
    
    # Obtaining the member '__getitem__' of a type (line 1622)
    getitem___32750 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1622, 4), idzr_rid_call_result_32749, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1622)
    subscript_call_result_32751 = invoke(stypy.reporting.localization.Localization(__file__, 1622, 4), getitem___32750, int_32741)
    
    # Assigning a type to the variable 'tuple_var_assignment_29773' (line 1622)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1622, 4), 'tuple_var_assignment_29773', subscript_call_result_32751)
    
    # Assigning a Name to a Name (line 1622):
    # Getting the type of 'tuple_var_assignment_29772' (line 1622)
    tuple_var_assignment_29772_32752 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1622, 4), 'tuple_var_assignment_29772')
    # Assigning a type to the variable 'idx' (line 1622)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1622, 4), 'idx', tuple_var_assignment_29772_32752)
    
    # Assigning a Name to a Name (line 1622):
    # Getting the type of 'tuple_var_assignment_29773' (line 1622)
    tuple_var_assignment_29773_32753 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1622, 4), 'tuple_var_assignment_29773')
    # Assigning a type to the variable 'proj' (line 1622)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1622, 9), 'proj', tuple_var_assignment_29773_32753)
    
    # Assigning a Call to a Name (line 1623):
    
    # Assigning a Call to a Name (line 1623):
    
    # Call to reshape(...): (line 1623)
    # Processing the call arguments (line 1623)
    
    # Obtaining an instance of the builtin type 'tuple' (line 1623)
    tuple_32764 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1623, 35), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 1623)
    # Adding element type (line 1623)
    # Getting the type of 'k' (line 1623)
    k_32765 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1623, 35), 'k', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1623, 35), tuple_32764, k_32765)
    # Adding element type (line 1623)
    # Getting the type of 'n' (line 1623)
    n_32766 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1623, 38), 'n', False)
    # Getting the type of 'k' (line 1623)
    k_32767 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1623, 40), 'k', False)
    # Applying the binary operator '-' (line 1623)
    result_sub_32768 = python_operator(stypy.reporting.localization.Localization(__file__, 1623, 38), '-', n_32766, k_32767)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1623, 35), tuple_32764, result_sub_32768)
    
    # Processing the call keyword arguments (line 1623)
    str_32769 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1623, 50), 'str', 'F')
    keyword_32770 = str_32769
    kwargs_32771 = {'order': keyword_32770}
    
    # Obtaining the type of the subscript
    # Getting the type of 'k' (line 1623)
    k_32754 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1623, 17), 'k', False)
    # Getting the type of 'n' (line 1623)
    n_32755 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1623, 20), 'n', False)
    # Getting the type of 'k' (line 1623)
    k_32756 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1623, 22), 'k', False)
    # Applying the binary operator '-' (line 1623)
    result_sub_32757 = python_operator(stypy.reporting.localization.Localization(__file__, 1623, 20), '-', n_32755, k_32756)
    
    # Applying the binary operator '*' (line 1623)
    result_mul_32758 = python_operator(stypy.reporting.localization.Localization(__file__, 1623, 17), '*', k_32754, result_sub_32757)
    
    slice_32759 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 1623, 11), None, result_mul_32758, None)
    # Getting the type of 'proj' (line 1623)
    proj_32760 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1623, 11), 'proj', False)
    # Obtaining the member '__getitem__' of a type (line 1623)
    getitem___32761 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1623, 11), proj_32760, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1623)
    subscript_call_result_32762 = invoke(stypy.reporting.localization.Localization(__file__, 1623, 11), getitem___32761, slice_32759)
    
    # Obtaining the member 'reshape' of a type (line 1623)
    reshape_32763 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1623, 11), subscript_call_result_32762, 'reshape')
    # Calling reshape(args, kwargs) (line 1623)
    reshape_call_result_32772 = invoke(stypy.reporting.localization.Localization(__file__, 1623, 11), reshape_32763, *[tuple_32764], **kwargs_32771)
    
    # Assigning a type to the variable 'proj' (line 1623)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1623, 4), 'proj', reshape_call_result_32772)
    
    # Obtaining an instance of the builtin type 'tuple' (line 1624)
    tuple_32773 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1624, 11), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 1624)
    # Adding element type (line 1624)
    # Getting the type of 'idx' (line 1624)
    idx_32774 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1624, 11), 'idx')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1624, 11), tuple_32773, idx_32774)
    # Adding element type (line 1624)
    # Getting the type of 'proj' (line 1624)
    proj_32775 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1624, 16), 'proj')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1624, 11), tuple_32773, proj_32775)
    
    # Assigning a type to the variable 'stypy_return_type' (line 1624)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1624, 4), 'stypy_return_type', tuple_32773)
    
    # ################# End of 'idzr_rid(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'idzr_rid' in the type store
    # Getting the type of 'stypy_return_type' (line 1595)
    stypy_return_type_32776 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1595, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_32776)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'idzr_rid'
    return stypy_return_type_32776

# Assigning a type to the variable 'idzr_rid' (line 1595)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1595, 0), 'idzr_rid', idzr_rid)

@norecursion
def idzr_rsvd(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'idzr_rsvd'
    module_type_store = module_type_store.open_function_context('idzr_rsvd', 1631, 0, False)
    
    # Passed parameters checking function
    idzr_rsvd.stypy_localization = localization
    idzr_rsvd.stypy_type_of_self = None
    idzr_rsvd.stypy_type_store = module_type_store
    idzr_rsvd.stypy_function_name = 'idzr_rsvd'
    idzr_rsvd.stypy_param_names_list = ['m', 'n', 'matveca', 'matvec', 'k']
    idzr_rsvd.stypy_varargs_param_name = None
    idzr_rsvd.stypy_kwargs_param_name = None
    idzr_rsvd.stypy_call_defaults = defaults
    idzr_rsvd.stypy_call_varargs = varargs
    idzr_rsvd.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'idzr_rsvd', ['m', 'n', 'matveca', 'matvec', 'k'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'idzr_rsvd', localization, ['m', 'n', 'matveca', 'matvec', 'k'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'idzr_rsvd(...)' code ##################

    str_32777 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1665, (-1)), 'str', '\n    Compute SVD of a complex matrix to a specified rank using random\n    matrix-vector multiplication.\n\n    :param m:\n        Matrix row dimension.\n    :type m: int\n    :param n:\n        Matrix column dimension.\n    :type n: int\n    :param matveca:\n        Function to apply the matrix adjoint to a vector, with call signature\n        `y = matveca(x)`, where `x` and `y` are the input and output vectors,\n        respectively.\n    :type matveca: function\n    :param matvec:\n        Function to apply the matrix to a vector, with call signature\n        `y = matvec(x)`, where `x` and `y` are the input and output vectors,\n        respectively.\n    :type matvec: function\n    :param k:\n        Rank of SVD.\n    :type k: int\n\n    :return:\n        Left singular vectors.\n    :rtype: :class:`numpy.ndarray`\n    :return:\n        Right singular vectors.\n    :rtype: :class:`numpy.ndarray`\n    :return:\n        Singular values.\n    :rtype: :class:`numpy.ndarray`\n    ')
    
    # Assigning a Call to a Tuple (line 1666):
    
    # Assigning a Subscript to a Name (line 1666):
    
    # Obtaining the type of the subscript
    int_32778 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1666, 4), 'int')
    
    # Call to idzr_rsvd(...): (line 1666)
    # Processing the call arguments (line 1666)
    # Getting the type of 'm' (line 1666)
    m_32781 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1666, 33), 'm', False)
    # Getting the type of 'n' (line 1666)
    n_32782 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1666, 36), 'n', False)
    # Getting the type of 'matveca' (line 1666)
    matveca_32783 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1666, 39), 'matveca', False)
    # Getting the type of 'matvec' (line 1666)
    matvec_32784 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1666, 48), 'matvec', False)
    # Getting the type of 'k' (line 1666)
    k_32785 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1666, 56), 'k', False)
    # Processing the call keyword arguments (line 1666)
    kwargs_32786 = {}
    # Getting the type of '_id' (line 1666)
    _id_32779 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1666, 19), '_id', False)
    # Obtaining the member 'idzr_rsvd' of a type (line 1666)
    idzr_rsvd_32780 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1666, 19), _id_32779, 'idzr_rsvd')
    # Calling idzr_rsvd(args, kwargs) (line 1666)
    idzr_rsvd_call_result_32787 = invoke(stypy.reporting.localization.Localization(__file__, 1666, 19), idzr_rsvd_32780, *[m_32781, n_32782, matveca_32783, matvec_32784, k_32785], **kwargs_32786)
    
    # Obtaining the member '__getitem__' of a type (line 1666)
    getitem___32788 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1666, 4), idzr_rsvd_call_result_32787, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1666)
    subscript_call_result_32789 = invoke(stypy.reporting.localization.Localization(__file__, 1666, 4), getitem___32788, int_32778)
    
    # Assigning a type to the variable 'tuple_var_assignment_29774' (line 1666)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1666, 4), 'tuple_var_assignment_29774', subscript_call_result_32789)
    
    # Assigning a Subscript to a Name (line 1666):
    
    # Obtaining the type of the subscript
    int_32790 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1666, 4), 'int')
    
    # Call to idzr_rsvd(...): (line 1666)
    # Processing the call arguments (line 1666)
    # Getting the type of 'm' (line 1666)
    m_32793 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1666, 33), 'm', False)
    # Getting the type of 'n' (line 1666)
    n_32794 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1666, 36), 'n', False)
    # Getting the type of 'matveca' (line 1666)
    matveca_32795 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1666, 39), 'matveca', False)
    # Getting the type of 'matvec' (line 1666)
    matvec_32796 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1666, 48), 'matvec', False)
    # Getting the type of 'k' (line 1666)
    k_32797 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1666, 56), 'k', False)
    # Processing the call keyword arguments (line 1666)
    kwargs_32798 = {}
    # Getting the type of '_id' (line 1666)
    _id_32791 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1666, 19), '_id', False)
    # Obtaining the member 'idzr_rsvd' of a type (line 1666)
    idzr_rsvd_32792 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1666, 19), _id_32791, 'idzr_rsvd')
    # Calling idzr_rsvd(args, kwargs) (line 1666)
    idzr_rsvd_call_result_32799 = invoke(stypy.reporting.localization.Localization(__file__, 1666, 19), idzr_rsvd_32792, *[m_32793, n_32794, matveca_32795, matvec_32796, k_32797], **kwargs_32798)
    
    # Obtaining the member '__getitem__' of a type (line 1666)
    getitem___32800 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1666, 4), idzr_rsvd_call_result_32799, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1666)
    subscript_call_result_32801 = invoke(stypy.reporting.localization.Localization(__file__, 1666, 4), getitem___32800, int_32790)
    
    # Assigning a type to the variable 'tuple_var_assignment_29775' (line 1666)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1666, 4), 'tuple_var_assignment_29775', subscript_call_result_32801)
    
    # Assigning a Subscript to a Name (line 1666):
    
    # Obtaining the type of the subscript
    int_32802 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1666, 4), 'int')
    
    # Call to idzr_rsvd(...): (line 1666)
    # Processing the call arguments (line 1666)
    # Getting the type of 'm' (line 1666)
    m_32805 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1666, 33), 'm', False)
    # Getting the type of 'n' (line 1666)
    n_32806 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1666, 36), 'n', False)
    # Getting the type of 'matveca' (line 1666)
    matveca_32807 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1666, 39), 'matveca', False)
    # Getting the type of 'matvec' (line 1666)
    matvec_32808 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1666, 48), 'matvec', False)
    # Getting the type of 'k' (line 1666)
    k_32809 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1666, 56), 'k', False)
    # Processing the call keyword arguments (line 1666)
    kwargs_32810 = {}
    # Getting the type of '_id' (line 1666)
    _id_32803 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1666, 19), '_id', False)
    # Obtaining the member 'idzr_rsvd' of a type (line 1666)
    idzr_rsvd_32804 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1666, 19), _id_32803, 'idzr_rsvd')
    # Calling idzr_rsvd(args, kwargs) (line 1666)
    idzr_rsvd_call_result_32811 = invoke(stypy.reporting.localization.Localization(__file__, 1666, 19), idzr_rsvd_32804, *[m_32805, n_32806, matveca_32807, matvec_32808, k_32809], **kwargs_32810)
    
    # Obtaining the member '__getitem__' of a type (line 1666)
    getitem___32812 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1666, 4), idzr_rsvd_call_result_32811, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1666)
    subscript_call_result_32813 = invoke(stypy.reporting.localization.Localization(__file__, 1666, 4), getitem___32812, int_32802)
    
    # Assigning a type to the variable 'tuple_var_assignment_29776' (line 1666)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1666, 4), 'tuple_var_assignment_29776', subscript_call_result_32813)
    
    # Assigning a Subscript to a Name (line 1666):
    
    # Obtaining the type of the subscript
    int_32814 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1666, 4), 'int')
    
    # Call to idzr_rsvd(...): (line 1666)
    # Processing the call arguments (line 1666)
    # Getting the type of 'm' (line 1666)
    m_32817 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1666, 33), 'm', False)
    # Getting the type of 'n' (line 1666)
    n_32818 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1666, 36), 'n', False)
    # Getting the type of 'matveca' (line 1666)
    matveca_32819 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1666, 39), 'matveca', False)
    # Getting the type of 'matvec' (line 1666)
    matvec_32820 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1666, 48), 'matvec', False)
    # Getting the type of 'k' (line 1666)
    k_32821 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1666, 56), 'k', False)
    # Processing the call keyword arguments (line 1666)
    kwargs_32822 = {}
    # Getting the type of '_id' (line 1666)
    _id_32815 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1666, 19), '_id', False)
    # Obtaining the member 'idzr_rsvd' of a type (line 1666)
    idzr_rsvd_32816 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1666, 19), _id_32815, 'idzr_rsvd')
    # Calling idzr_rsvd(args, kwargs) (line 1666)
    idzr_rsvd_call_result_32823 = invoke(stypy.reporting.localization.Localization(__file__, 1666, 19), idzr_rsvd_32816, *[m_32817, n_32818, matveca_32819, matvec_32820, k_32821], **kwargs_32822)
    
    # Obtaining the member '__getitem__' of a type (line 1666)
    getitem___32824 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1666, 4), idzr_rsvd_call_result_32823, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1666)
    subscript_call_result_32825 = invoke(stypy.reporting.localization.Localization(__file__, 1666, 4), getitem___32824, int_32814)
    
    # Assigning a type to the variable 'tuple_var_assignment_29777' (line 1666)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1666, 4), 'tuple_var_assignment_29777', subscript_call_result_32825)
    
    # Assigning a Name to a Name (line 1666):
    # Getting the type of 'tuple_var_assignment_29774' (line 1666)
    tuple_var_assignment_29774_32826 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1666, 4), 'tuple_var_assignment_29774')
    # Assigning a type to the variable 'U' (line 1666)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1666, 4), 'U', tuple_var_assignment_29774_32826)
    
    # Assigning a Name to a Name (line 1666):
    # Getting the type of 'tuple_var_assignment_29775' (line 1666)
    tuple_var_assignment_29775_32827 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1666, 4), 'tuple_var_assignment_29775')
    # Assigning a type to the variable 'V' (line 1666)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1666, 7), 'V', tuple_var_assignment_29775_32827)
    
    # Assigning a Name to a Name (line 1666):
    # Getting the type of 'tuple_var_assignment_29776' (line 1666)
    tuple_var_assignment_29776_32828 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1666, 4), 'tuple_var_assignment_29776')
    # Assigning a type to the variable 'S' (line 1666)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1666, 10), 'S', tuple_var_assignment_29776_32828)
    
    # Assigning a Name to a Name (line 1666):
    # Getting the type of 'tuple_var_assignment_29777' (line 1666)
    tuple_var_assignment_29777_32829 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1666, 4), 'tuple_var_assignment_29777')
    # Assigning a type to the variable 'ier' (line 1666)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1666, 13), 'ier', tuple_var_assignment_29777_32829)
    
    # Getting the type of 'ier' (line 1667)
    ier_32830 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1667, 7), 'ier')
    # Testing the type of an if condition (line 1667)
    if_condition_32831 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1667, 4), ier_32830)
    # Assigning a type to the variable 'if_condition_32831' (line 1667)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1667, 4), 'if_condition_32831', if_condition_32831)
    # SSA begins for if statement (line 1667)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # Getting the type of '_RETCODE_ERROR' (line 1668)
    _RETCODE_ERROR_32832 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1668, 14), '_RETCODE_ERROR')
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 1668, 8), _RETCODE_ERROR_32832, 'raise parameter', BaseException)
    # SSA join for if statement (line 1667)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Obtaining an instance of the builtin type 'tuple' (line 1669)
    tuple_32833 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1669, 11), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 1669)
    # Adding element type (line 1669)
    # Getting the type of 'U' (line 1669)
    U_32834 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1669, 11), 'U')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1669, 11), tuple_32833, U_32834)
    # Adding element type (line 1669)
    # Getting the type of 'V' (line 1669)
    V_32835 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1669, 14), 'V')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1669, 11), tuple_32833, V_32835)
    # Adding element type (line 1669)
    # Getting the type of 'S' (line 1669)
    S_32836 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1669, 17), 'S')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1669, 11), tuple_32833, S_32836)
    
    # Assigning a type to the variable 'stypy_return_type' (line 1669)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1669, 4), 'stypy_return_type', tuple_32833)
    
    # ################# End of 'idzr_rsvd(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'idzr_rsvd' in the type store
    # Getting the type of 'stypy_return_type' (line 1631)
    stypy_return_type_32837 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1631, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_32837)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'idzr_rsvd'
    return stypy_return_type_32837

# Assigning a type to the variable 'idzr_rsvd' (line 1631)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1631, 0), 'idzr_rsvd', idzr_rsvd)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
