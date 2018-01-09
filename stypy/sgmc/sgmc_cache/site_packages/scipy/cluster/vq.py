
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: '''
2: ====================================================================
3: K-means clustering and vector quantization (:mod:`scipy.cluster.vq`)
4: ====================================================================
5: 
6: Provides routines for k-means clustering, generating code books
7: from k-means models, and quantizing vectors by comparing them with
8: centroids in a code book.
9: 
10: .. autosummary::
11:    :toctree: generated/
12: 
13:    whiten -- Normalize a group of observations so each feature has unit variance
14:    vq -- Calculate code book membership of a set of observation vectors
15:    kmeans -- Performs k-means on a set of observation vectors forming k clusters
16:    kmeans2 -- A different implementation of k-means with more methods
17:            -- for initializing centroids
18: 
19: Background information
20: ======================
21: The k-means algorithm takes as input the number of clusters to
22: generate, k, and a set of observation vectors to cluster.  It
23: returns a set of centroids, one for each of the k clusters.  An
24: observation vector is classified with the cluster number or
25: centroid index of the centroid closest to it.
26: 
27: A vector v belongs to cluster i if it is closer to centroid i than
28: any other centroids. If v belongs to i, we say centroid i is the
29: dominating centroid of v. The k-means algorithm tries to
30: minimize distortion, which is defined as the sum of the squared distances
31: between each observation vector and its dominating centroid.  Each
32: step of the k-means algorithm refines the choices of centroids to
33: reduce distortion. The change in distortion is used as a
34: stopping criterion: when the change is lower than a threshold, the
35: k-means algorithm is not making sufficient progress and
36: terminates. One can also define a maximum number of iterations.
37: 
38: Since vector quantization is a natural application for k-means,
39: information theory terminology is often used.  The centroid index
40: or cluster index is also referred to as a "code" and the table
41: mapping codes to centroids and vice versa is often referred as a
42: "code book". The result of k-means, a set of centroids, can be
43: used to quantize vectors. Quantization aims to find an encoding of
44: vectors that reduces the expected distortion.
45: 
46: All routines expect obs to be a M by N array where the rows are
47: the observation vectors. The codebook is a k by N array where the
48: i'th row is the centroid of code word i. The observation vectors
49: and centroids have the same feature dimension.
50: 
51: As an example, suppose we wish to compress a 24-bit color image
52: (each pixel is represented by one byte for red, one for blue, and
53: one for green) before sending it over the web.  By using a smaller
54: 8-bit encoding, we can reduce the amount of data by two
55: thirds. Ideally, the colors for each of the 256 possible 8-bit
56: encoding values should be chosen to minimize distortion of the
57: color. Running k-means with k=256 generates a code book of 256
58: codes, which fills up all possible 8-bit sequences.  Instead of
59: sending a 3-byte value for each pixel, the 8-bit centroid index
60: (or code word) of the dominating centroid is transmitted. The code
61: book is also sent over the wire so each 8-bit code can be
62: translated back to a 24-bit pixel value representation. If the
63: image of interest was of an ocean, we would expect many 24-bit
64: blues to be represented by 8-bit codes. If it was an image of a
65: human face, more flesh tone colors would be represented in the
66: code book.
67: 
68: '''
69: from __future__ import division, print_function, absolute_import
70: 
71: import warnings
72: import numpy as np
73: from collections import deque
74: from scipy._lib._util import _asarray_validated
75: from scipy._lib.six import xrange
76: from scipy.spatial.distance import cdist
77: 
78: from . import _vq
79: 
80: __docformat__ = 'restructuredtext'
81: 
82: __all__ = ['whiten', 'vq', 'kmeans', 'kmeans2']
83: 
84: 
85: class ClusterError(Exception):
86:     pass
87: 
88: 
89: def whiten(obs, check_finite=True):
90:     '''
91:     Normalize a group of observations on a per feature basis.
92: 
93:     Before running k-means, it is beneficial to rescale each feature
94:     dimension of the observation set with whitening. Each feature is
95:     divided by its standard deviation across all observations to give
96:     it unit variance.
97: 
98:     Parameters
99:     ----------
100:     obs : ndarray
101:         Each row of the array is an observation.  The
102:         columns are the features seen during each observation.
103: 
104:         >>> #         f0    f1    f2
105:         >>> obs = [[  1.,   1.,   1.],  #o0
106:         ...        [  2.,   2.,   2.],  #o1
107:         ...        [  3.,   3.,   3.],  #o2
108:         ...        [  4.,   4.,   4.]]  #o3
109: 
110:     check_finite : bool, optional
111:         Whether to check that the input matrices contain only finite numbers.
112:         Disabling may give a performance gain, but may result in problems
113:         (crashes, non-termination) if the inputs do contain infinities or NaNs.
114:         Default: True
115: 
116:     Returns
117:     -------
118:     result : ndarray
119:         Contains the values in `obs` scaled by the standard deviation
120:         of each column.
121: 
122:     Examples
123:     --------
124:     >>> from scipy.cluster.vq import whiten
125:     >>> features  = np.array([[1.9, 2.3, 1.7],
126:     ...                       [1.5, 2.5, 2.2],
127:     ...                       [0.8, 0.6, 1.7,]])
128:     >>> whiten(features)
129:     array([[ 4.17944278,  2.69811351,  7.21248917],
130:            [ 3.29956009,  2.93273208,  9.33380951],
131:            [ 1.75976538,  0.7038557 ,  7.21248917]])
132: 
133:     '''
134:     obs = _asarray_validated(obs, check_finite=check_finite)
135:     std_dev = obs.std(axis=0)
136:     zero_std_mask = std_dev == 0
137:     if zero_std_mask.any():
138:         std_dev[zero_std_mask] = 1.0
139:         warnings.warn("Some columns have standard deviation zero. "
140:                       "The values of these columns will not change.",
141:                       RuntimeWarning)
142:     return obs / std_dev
143: 
144: 
145: def vq(obs, code_book, check_finite=True):
146:     '''
147:     Assign codes from a code book to observations.
148: 
149:     Assigns a code from a code book to each observation. Each
150:     observation vector in the 'M' by 'N' `obs` array is compared with the
151:     centroids in the code book and assigned the code of the closest
152:     centroid.
153: 
154:     The features in `obs` should have unit variance, which can be
155:     achieved by passing them through the whiten function.  The code
156:     book can be created with the k-means algorithm or a different
157:     encoding algorithm.
158: 
159:     Parameters
160:     ----------
161:     obs : ndarray
162:         Each row of the 'M' x 'N' array is an observation.  The columns are
163:         the "features" seen during each observation. The features must be
164:         whitened first using the whiten function or something equivalent.
165:     code_book : ndarray
166:         The code book is usually generated using the k-means algorithm.
167:         Each row of the array holds a different code, and the columns are
168:         the features of the code.
169: 
170:          >>> #              f0    f1    f2   f3
171:          >>> code_book = [
172:          ...             [  1.,   2.,   3.,   4.],  #c0
173:          ...             [  1.,   2.,   3.,   4.],  #c1
174:          ...             [  1.,   2.,   3.,   4.]]  #c2
175: 
176:     check_finite : bool, optional
177:         Whether to check that the input matrices contain only finite numbers.
178:         Disabling may give a performance gain, but may result in problems
179:         (crashes, non-termination) if the inputs do contain infinities or NaNs.
180:         Default: True
181: 
182:     Returns
183:     -------
184:     code : ndarray
185:         A length M array holding the code book index for each observation.
186:     dist : ndarray
187:         The distortion (distance) between the observation and its nearest
188:         code.
189: 
190:     Examples
191:     --------
192:     >>> from numpy import array
193:     >>> from scipy.cluster.vq import vq
194:     >>> code_book = array([[1.,1.,1.],
195:     ...                    [2.,2.,2.]])
196:     >>> features  = array([[  1.9,2.3,1.7],
197:     ...                    [  1.5,2.5,2.2],
198:     ...                    [  0.8,0.6,1.7]])
199:     >>> vq(features,code_book)
200:     (array([1, 1, 0],'i'), array([ 0.43588989,  0.73484692,  0.83066239]))
201: 
202:     '''
203:     obs = _asarray_validated(obs, check_finite=check_finite)
204:     code_book = _asarray_validated(code_book, check_finite=check_finite)
205:     ct = np.common_type(obs, code_book)
206: 
207:     c_obs = obs.astype(ct, copy=False)
208:     c_code_book = code_book.astype(ct, copy=False)
209: 
210:     if np.issubdtype(ct, np.float64) or np.issubdtype(ct, np.float32):
211:         return _vq.vq(c_obs, c_code_book)
212:     return py_vq(obs, code_book, check_finite=False)
213: 
214: 
215: def py_vq(obs, code_book, check_finite=True):
216:     ''' Python version of vq algorithm.
217: 
218:     The algorithm computes the euclidian distance between each
219:     observation and every frame in the code_book.
220: 
221:     Parameters
222:     ----------
223:     obs : ndarray
224:         Expects a rank 2 array. Each row is one observation.
225:     code_book : ndarray
226:         Code book to use. Same format than obs. Should have same number of
227:         features (eg columns) than obs.
228:     check_finite : bool, optional
229:         Whether to check that the input matrices contain only finite numbers.
230:         Disabling may give a performance gain, but may result in problems
231:         (crashes, non-termination) if the inputs do contain infinities or NaNs.
232:         Default: True
233: 
234:     Returns
235:     -------
236:     code : ndarray
237:         code[i] gives the label of the ith obversation, that its code is
238:         code_book[code[i]].
239:     mind_dist : ndarray
240:         min_dist[i] gives the distance between the ith observation and its
241:         corresponding code.
242: 
243:     Notes
244:     -----
245:     This function is slower than the C version but works for
246:     all input types.  If the inputs have the wrong types for the
247:     C versions of the function, this one is called as a last resort.
248: 
249:     It is about 20 times slower than the C version.
250: 
251:     '''
252:     obs = _asarray_validated(obs, check_finite=check_finite)
253:     code_book = _asarray_validated(code_book, check_finite=check_finite)
254: 
255:     if obs.ndim != code_book.ndim:
256:         raise ValueError("Observation and code_book should have the same rank")
257: 
258:     if obs.ndim == 1:
259:         obs = obs[:, np.newaxis]
260:         code_book = code_book[:, np.newaxis]
261: 
262:     dist = cdist(obs, code_book)
263:     code = dist.argmin(axis=1)
264:     min_dist = dist[np.arange(len(code)), code]
265:     return code, min_dist
266: 
267: # py_vq2 was equivalent to py_vq
268: py_vq2 = np.deprecate(py_vq, old_name='py_vq2', new_name='py_vq')
269: 
270: 
271: def _kmeans(obs, guess, thresh=1e-5):
272:     ''' "raw" version of k-means.
273: 
274:     Returns
275:     -------
276:     code_book
277:         the lowest distortion codebook found.
278:     avg_dist
279:         the average distance a observation is from a code in the book.
280:         Lower means the code_book matches the data better.
281: 
282:     See Also
283:     --------
284:     kmeans : wrapper around k-means
285: 
286:     Examples
287:     --------
288:     Note: not whitened in this example.
289: 
290:     >>> from numpy import array
291:     >>> from scipy.cluster.vq import _kmeans
292:     >>> features  = array([[ 1.9,2.3],
293:     ...                    [ 1.5,2.5],
294:     ...                    [ 0.8,0.6],
295:     ...                    [ 0.4,1.8],
296:     ...                    [ 1.0,1.0]])
297:     >>> book = array((features[0],features[2]))
298:     >>> _kmeans(features,book)
299:     (array([[ 1.7       ,  2.4       ],
300:            [ 0.73333333,  1.13333333]]), 0.40563916697728591)
301: 
302:     '''
303: 
304:     code_book = np.asarray(guess)
305:     diff = np.inf
306:     prev_avg_dists = deque([diff], maxlen=2)
307:     while diff > thresh:
308:         # compute membership and distances between obs and code_book
309:         obs_code, distort = vq(obs, code_book, check_finite=False)
310:         prev_avg_dists.append(distort.mean(axis=-1))
311:         # recalc code_book as centroids of associated obs
312:         code_book, has_members = _vq.update_cluster_means(obs, obs_code,
313:                                                           code_book.shape[0])
314:         code_book = code_book[has_members]
315:         diff = prev_avg_dists[0] - prev_avg_dists[1]
316: 
317:     return code_book, prev_avg_dists[1]
318: 
319: 
320: def kmeans(obs, k_or_guess, iter=20, thresh=1e-5, check_finite=True):
321:     '''
322:     Performs k-means on a set of observation vectors forming k clusters.
323: 
324:     The k-means algorithm adjusts the centroids until sufficient
325:     progress cannot be made, i.e. the change in distortion since
326:     the last iteration is less than some threshold. This yields
327:     a code book mapping centroids to codes and vice versa.
328: 
329:     Distortion is defined as the sum of the squared differences
330:     between the observations and the corresponding centroid.
331: 
332:     Parameters
333:     ----------
334:     obs : ndarray
335:        Each row of the M by N array is an observation vector. The
336:        columns are the features seen during each observation.
337:        The features must be whitened first with the `whiten` function.
338: 
339:     k_or_guess : int or ndarray
340:        The number of centroids to generate. A code is assigned to
341:        each centroid, which is also the row index of the centroid
342:        in the code_book matrix generated.
343: 
344:        The initial k centroids are chosen by randomly selecting
345:        observations from the observation matrix. Alternatively,
346:        passing a k by N array specifies the initial k centroids.
347: 
348:     iter : int, optional
349:        The number of times to run k-means, returning the codebook
350:        with the lowest distortion. This argument is ignored if
351:        initial centroids are specified with an array for the
352:        ``k_or_guess`` parameter. This parameter does not represent the
353:        number of iterations of the k-means algorithm.
354: 
355:     thresh : float, optional
356:        Terminates the k-means algorithm if the change in
357:        distortion since the last k-means iteration is less than
358:        or equal to thresh.
359: 
360:     check_finite : bool, optional
361:         Whether to check that the input matrices contain only finite numbers.
362:         Disabling may give a performance gain, but may result in problems
363:         (crashes, non-termination) if the inputs do contain infinities or NaNs.
364:         Default: True
365: 
366:     Returns
367:     -------
368:     codebook : ndarray
369:        A k by N array of k centroids. The i'th centroid
370:        codebook[i] is represented with the code i. The centroids
371:        and codes generated represent the lowest distortion seen,
372:        not necessarily the globally minimal distortion.
373: 
374:     distortion : float
375:        The distortion between the observations passed and the
376:        centroids generated.
377: 
378:     See Also
379:     --------
380:     kmeans2 : a different implementation of k-means clustering
381:        with more methods for generating initial centroids but without
382:        using a distortion change threshold as a stopping criterion.
383: 
384:     whiten : must be called prior to passing an observation matrix
385:        to kmeans.
386: 
387:     Examples
388:     --------
389:     >>> from numpy import array
390:     >>> from scipy.cluster.vq import vq, kmeans, whiten
391:     >>> import matplotlib.pyplot as plt
392:     >>> features  = array([[ 1.9,2.3],
393:     ...                    [ 1.5,2.5],
394:     ...                    [ 0.8,0.6],
395:     ...                    [ 0.4,1.8],
396:     ...                    [ 0.1,0.1],
397:     ...                    [ 0.2,1.8],
398:     ...                    [ 2.0,0.5],
399:     ...                    [ 0.3,1.5],
400:     ...                    [ 1.0,1.0]])
401:     >>> whitened = whiten(features)
402:     >>> book = np.array((whitened[0],whitened[2]))
403:     >>> kmeans(whitened,book)
404:     (array([[ 2.3110306 ,  2.86287398],    # random
405:            [ 0.93218041,  1.24398691]]), 0.85684700941625547)
406: 
407:     >>> from numpy import random
408:     >>> random.seed((1000,2000))
409:     >>> codes = 3
410:     >>> kmeans(whitened,codes)
411:     (array([[ 2.3110306 ,  2.86287398],    # random
412:            [ 1.32544402,  0.65607529],
413:            [ 0.40782893,  2.02786907]]), 0.5196582527686241)
414: 
415:     >>> # Create 50 datapoints in two clusters a and b
416:     >>> pts = 50
417:     >>> a = np.random.multivariate_normal([0, 0], [[4, 1], [1, 4]], size=pts)
418:     >>> b = np.random.multivariate_normal([30, 10],
419:     ...                                   [[10, 2], [2, 1]],
420:     ...                                   size=pts)
421:     >>> features = np.concatenate((a, b))
422:     >>> # Whiten data
423:     >>> whitened = whiten(features)
424:     >>> # Find 2 clusters in the data
425:     >>> codebook, distortion = kmeans(whitened, 2)
426:     >>> # Plot whitened data and cluster centers in red
427:     >>> plt.scatter(whitened[:, 0], whitened[:, 1])
428:     >>> plt.scatter(codebook[:, 0], codebook[:, 1], c='r')
429:     >>> plt.show()
430:     '''
431:     obs = _asarray_validated(obs, check_finite=check_finite)
432:     if iter < 1:
433:         raise ValueError("iter must be at least 1, got %s" % iter)
434: 
435:     # Determine whether a count (scalar) or an initial guess (array) was passed.
436:     if not np.isscalar(k_or_guess):
437:         guess = _asarray_validated(k_or_guess, check_finite=check_finite)
438:         if guess.size < 1:
439:             raise ValueError("Asked for 0 clusters. Initial book was %s" %
440:                              guess)
441:         return _kmeans(obs, guess, thresh=thresh)
442: 
443:     # k_or_guess is a scalar, now verify that it's an integer
444:     k = int(k_or_guess)
445:     if k != k_or_guess:
446:         raise ValueError("If k_or_guess is a scalar, it must be an integer.")
447:     if k < 1:
448:         raise ValueError("Asked for %d clusters." % k)
449: 
450:     # initialize best distance value to a large value
451:     best_dist = np.inf
452:     for i in xrange(iter):
453:         # the initial code book is randomly selected from observations
454:         guess = _kpoints(obs, k)
455:         book, dist = _kmeans(obs, guess, thresh=thresh)
456:         if dist < best_dist:
457:             best_book = book
458:             best_dist = dist
459:     return best_book, best_dist
460: 
461: 
462: def _kpoints(data, k):
463:     '''Pick k points at random in data (one row = one observation).
464: 
465:     Parameters
466:     ----------
467:     data : ndarray
468:         Expect a rank 1 or 2 array. Rank 1 are assumed to describe one
469:         dimensional data, rank 2 multidimensional data, in which case one
470:         row is one observation.
471:     k : int
472:         Number of samples to generate.
473: 
474:     '''
475:     idx = np.random.choice(data.shape[0], size=k, replace=False)
476:     return data[idx]
477: 
478: 
479: def _krandinit(data, k):
480:     '''Returns k samples of a random variable which parameters depend on data.
481: 
482:     More precisely, it returns k observations sampled from a Gaussian random
483:     variable which mean and covariances are the one estimated from data.
484: 
485:     Parameters
486:     ----------
487:     data : ndarray
488:         Expect a rank 1 or 2 array. Rank 1 are assumed to describe one
489:         dimensional data, rank 2 multidimensional data, in which case one
490:         row is one observation.
491:     k : int
492:         Number of samples to generate.
493: 
494:     '''
495:     mu = data.mean(axis=0)
496: 
497:     if data.ndim == 1:
498:         cov = np.cov(data)
499:         x = np.random.randn(k)
500:         x *= np.sqrt(cov)
501:     elif data.shape[1] > data.shape[0]:
502:         # initialize when the covariance matrix is rank deficient
503:         _, s, vh = np.linalg.svd(data - mu, full_matrices=False)
504:         x = np.random.randn(k, s.size)
505:         sVh = s[:, None] * vh / np.sqrt(data.shape[0] - 1)
506:         x = x.dot(sVh)
507:     else:
508:         cov = np.atleast_2d(np.cov(data, rowvar=False))
509: 
510:         # k rows, d cols (one row = one obs)
511:         # Generate k sample of a random variable ~ Gaussian(mu, cov)
512:         x = np.random.randn(k, mu.size)
513:         x = x.dot(np.linalg.cholesky(cov).T)
514: 
515:     x += mu
516:     return x
517: 
518: _valid_init_meth = {'random': _krandinit, 'points': _kpoints}
519: 
520: 
521: def _missing_warn():
522:     '''Print a warning when called.'''
523:     warnings.warn("One of the clusters is empty. "
524:                   "Re-run kmeans with a different initialization.")
525: 
526: 
527: def _missing_raise():
528:     '''raise a ClusterError when called.'''
529:     raise ClusterError("One of the clusters is empty. "
530:                        "Re-run kmeans with a different initialization.")
531: 
532: _valid_miss_meth = {'warn': _missing_warn, 'raise': _missing_raise}
533: 
534: 
535: def kmeans2(data, k, iter=10, thresh=1e-5, minit='random',
536:             missing='warn', check_finite=True):
537:     '''
538:     Classify a set of observations into k clusters using the k-means algorithm.
539: 
540:     The algorithm attempts to minimize the Euclidian distance between
541:     observations and centroids. Several initialization methods are
542:     included.
543: 
544:     Parameters
545:     ----------
546:     data : ndarray
547:         A 'M' by 'N' array of 'M' observations in 'N' dimensions or a length
548:         'M' array of 'M' one-dimensional observations.
549:     k : int or ndarray
550:         The number of clusters to form as well as the number of
551:         centroids to generate. If `minit` initialization string is
552:         'matrix', or if a ndarray is given instead, it is
553:         interpreted as initial cluster to use instead.
554:     iter : int, optional
555:         Number of iterations of the k-means algorithm to run. Note
556:         that this differs in meaning from the iters parameter to
557:         the kmeans function.
558:     thresh : float, optional
559:         (not used yet)
560:     minit : str, optional
561:         Method for initialization. Available methods are 'random',
562:         'points', and 'matrix':
563: 
564:         'random': generate k centroids from a Gaussian with mean and
565:         variance estimated from the data.
566: 
567:         'points': choose k observations (rows) at random from data for
568:         the initial centroids.
569: 
570:         'matrix': interpret the k parameter as a k by M (or length k
571:         array for one-dimensional data) array of initial centroids.
572:     missing : str, optional
573:         Method to deal with empty clusters. Available methods are
574:         'warn' and 'raise':
575: 
576:         'warn': give a warning and continue.
577: 
578:         'raise': raise an ClusterError and terminate the algorithm.
579:     check_finite : bool, optional
580:         Whether to check that the input matrices contain only finite numbers.
581:         Disabling may give a performance gain, but may result in problems
582:         (crashes, non-termination) if the inputs do contain infinities or NaNs.
583:         Default: True
584: 
585:     Returns
586:     -------
587:     centroid : ndarray
588:         A 'k' by 'N' array of centroids found at the last iteration of
589:         k-means.
590:     label : ndarray
591:         label[i] is the code or index of the centroid the
592:         i'th observation is closest to.
593: 
594:     '''
595:     if int(iter) < 1:
596:         raise ValueError("Invalid iter (%s), "
597:                          "must be a positive integer." % iter)
598:     try:
599:         miss_meth = _valid_miss_meth[missing]
600:     except KeyError:
601:         raise ValueError("Unknown missing method %r" % (missing,))
602: 
603:     data = _asarray_validated(data, check_finite=check_finite)
604:     if data.ndim == 1:
605:         d = 1
606:     elif data.ndim == 2:
607:         d = data.shape[1]
608:     else:
609:         raise ValueError("Input of rank > 2 is not supported.")
610: 
611:     if data.size < 1:
612:         raise ValueError("Empty input is not supported.")
613: 
614:     # If k is not a single value it should be compatible with data's shape
615:     if minit == 'matrix' or not np.isscalar(k):
616:         code_book = np.array(k, copy=True)
617:         if data.ndim != code_book.ndim:
618:             raise ValueError("k array doesn't match data rank")
619:         nc = len(code_book)
620:         if data.ndim > 1 and code_book.shape[1] != d:
621:             raise ValueError("k array doesn't match data dimension")
622:     else:
623:         nc = int(k)
624: 
625:         if nc < 1:
626:             raise ValueError("Cannot ask kmeans2 for %d clusters"
627:                              " (k was %s)" % (nc, k))
628:         elif nc != k:
629:             warnings.warn("k was not an integer, was converted.")
630: 
631:         try:
632:             init_meth = _valid_init_meth[minit]
633:         except KeyError:
634:             raise ValueError("Unknown init method %r" % (minit,))
635:         else:
636:             code_book = init_meth(data, k)
637: 
638:     for i in xrange(iter):
639:         # Compute the nearest neighbor for each obs using the current code book
640:         label = vq(data, code_book)[0]
641:         # Update the code book by computing centroids
642:         new_code_book, has_members = _vq.update_cluster_means(data, label, nc)
643:         if not has_members.all():
644:             miss_meth()
645:             # Set the empty clusters to their previous positions
646:             new_code_book[~has_members] = code_book[~has_members]
647:         code_book = new_code_book
648: 
649:     return code_book, label
650: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_5747 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, (-1)), 'str', '\n====================================================================\nK-means clustering and vector quantization (:mod:`scipy.cluster.vq`)\n====================================================================\n\nProvides routines for k-means clustering, generating code books\nfrom k-means models, and quantizing vectors by comparing them with\ncentroids in a code book.\n\n.. autosummary::\n   :toctree: generated/\n\n   whiten -- Normalize a group of observations so each feature has unit variance\n   vq -- Calculate code book membership of a set of observation vectors\n   kmeans -- Performs k-means on a set of observation vectors forming k clusters\n   kmeans2 -- A different implementation of k-means with more methods\n           -- for initializing centroids\n\nBackground information\n======================\nThe k-means algorithm takes as input the number of clusters to\ngenerate, k, and a set of observation vectors to cluster.  It\nreturns a set of centroids, one for each of the k clusters.  An\nobservation vector is classified with the cluster number or\ncentroid index of the centroid closest to it.\n\nA vector v belongs to cluster i if it is closer to centroid i than\nany other centroids. If v belongs to i, we say centroid i is the\ndominating centroid of v. The k-means algorithm tries to\nminimize distortion, which is defined as the sum of the squared distances\nbetween each observation vector and its dominating centroid.  Each\nstep of the k-means algorithm refines the choices of centroids to\nreduce distortion. The change in distortion is used as a\nstopping criterion: when the change is lower than a threshold, the\nk-means algorithm is not making sufficient progress and\nterminates. One can also define a maximum number of iterations.\n\nSince vector quantization is a natural application for k-means,\ninformation theory terminology is often used.  The centroid index\nor cluster index is also referred to as a "code" and the table\nmapping codes to centroids and vice versa is often referred as a\n"code book". The result of k-means, a set of centroids, can be\nused to quantize vectors. Quantization aims to find an encoding of\nvectors that reduces the expected distortion.\n\nAll routines expect obs to be a M by N array where the rows are\nthe observation vectors. The codebook is a k by N array where the\ni\'th row is the centroid of code word i. The observation vectors\nand centroids have the same feature dimension.\n\nAs an example, suppose we wish to compress a 24-bit color image\n(each pixel is represented by one byte for red, one for blue, and\none for green) before sending it over the web.  By using a smaller\n8-bit encoding, we can reduce the amount of data by two\nthirds. Ideally, the colors for each of the 256 possible 8-bit\nencoding values should be chosen to minimize distortion of the\ncolor. Running k-means with k=256 generates a code book of 256\ncodes, which fills up all possible 8-bit sequences.  Instead of\nsending a 3-byte value for each pixel, the 8-bit centroid index\n(or code word) of the dominating centroid is transmitted. The code\nbook is also sent over the wire so each 8-bit code can be\ntranslated back to a 24-bit pixel value representation. If the\nimage of interest was of an ocean, we would expect many 24-bit\nblues to be represented by 8-bit codes. If it was an image of a\nhuman face, more flesh tone colors would be represented in the\ncode book.\n\n')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 71, 0))

# 'import warnings' statement (line 71)
import warnings

import_module(stypy.reporting.localization.Localization(__file__, 71, 0), 'warnings', warnings, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 72, 0))

# 'import numpy' statement (line 72)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/cluster/')
import_5748 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 72, 0), 'numpy')

if (type(import_5748) is not StypyTypeError):

    if (import_5748 != 'pyd_module'):
        __import__(import_5748)
        sys_modules_5749 = sys.modules[import_5748]
        import_module(stypy.reporting.localization.Localization(__file__, 72, 0), 'np', sys_modules_5749.module_type_store, module_type_store)
    else:
        import numpy as np

        import_module(stypy.reporting.localization.Localization(__file__, 72, 0), 'np', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 72)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 72, 0), 'numpy', import_5748)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/cluster/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 73, 0))

# 'from collections import deque' statement (line 73)
try:
    from collections import deque

except:
    deque = UndefinedType
import_from_module(stypy.reporting.localization.Localization(__file__, 73, 0), 'collections', None, module_type_store, ['deque'], [deque])

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 74, 0))

# 'from scipy._lib._util import _asarray_validated' statement (line 74)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/cluster/')
import_5750 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 74, 0), 'scipy._lib._util')

if (type(import_5750) is not StypyTypeError):

    if (import_5750 != 'pyd_module'):
        __import__(import_5750)
        sys_modules_5751 = sys.modules[import_5750]
        import_from_module(stypy.reporting.localization.Localization(__file__, 74, 0), 'scipy._lib._util', sys_modules_5751.module_type_store, module_type_store, ['_asarray_validated'])
        nest_module(stypy.reporting.localization.Localization(__file__, 74, 0), __file__, sys_modules_5751, sys_modules_5751.module_type_store, module_type_store)
    else:
        from scipy._lib._util import _asarray_validated

        import_from_module(stypy.reporting.localization.Localization(__file__, 74, 0), 'scipy._lib._util', None, module_type_store, ['_asarray_validated'], [_asarray_validated])

else:
    # Assigning a type to the variable 'scipy._lib._util' (line 74)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 74, 0), 'scipy._lib._util', import_5750)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/cluster/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 75, 0))

# 'from scipy._lib.six import xrange' statement (line 75)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/cluster/')
import_5752 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 75, 0), 'scipy._lib.six')

if (type(import_5752) is not StypyTypeError):

    if (import_5752 != 'pyd_module'):
        __import__(import_5752)
        sys_modules_5753 = sys.modules[import_5752]
        import_from_module(stypy.reporting.localization.Localization(__file__, 75, 0), 'scipy._lib.six', sys_modules_5753.module_type_store, module_type_store, ['xrange'])
        nest_module(stypy.reporting.localization.Localization(__file__, 75, 0), __file__, sys_modules_5753, sys_modules_5753.module_type_store, module_type_store)
    else:
        from scipy._lib.six import xrange

        import_from_module(stypy.reporting.localization.Localization(__file__, 75, 0), 'scipy._lib.six', None, module_type_store, ['xrange'], [xrange])

else:
    # Assigning a type to the variable 'scipy._lib.six' (line 75)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 75, 0), 'scipy._lib.six', import_5752)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/cluster/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 76, 0))

# 'from scipy.spatial.distance import cdist' statement (line 76)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/cluster/')
import_5754 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 76, 0), 'scipy.spatial.distance')

if (type(import_5754) is not StypyTypeError):

    if (import_5754 != 'pyd_module'):
        __import__(import_5754)
        sys_modules_5755 = sys.modules[import_5754]
        import_from_module(stypy.reporting.localization.Localization(__file__, 76, 0), 'scipy.spatial.distance', sys_modules_5755.module_type_store, module_type_store, ['cdist'])
        nest_module(stypy.reporting.localization.Localization(__file__, 76, 0), __file__, sys_modules_5755, sys_modules_5755.module_type_store, module_type_store)
    else:
        from scipy.spatial.distance import cdist

        import_from_module(stypy.reporting.localization.Localization(__file__, 76, 0), 'scipy.spatial.distance', None, module_type_store, ['cdist'], [cdist])

else:
    # Assigning a type to the variable 'scipy.spatial.distance' (line 76)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 76, 0), 'scipy.spatial.distance', import_5754)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/cluster/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 78, 0))

# 'from scipy.cluster import _vq' statement (line 78)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/cluster/')
import_5756 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 78, 0), 'scipy.cluster')

if (type(import_5756) is not StypyTypeError):

    if (import_5756 != 'pyd_module'):
        __import__(import_5756)
        sys_modules_5757 = sys.modules[import_5756]
        import_from_module(stypy.reporting.localization.Localization(__file__, 78, 0), 'scipy.cluster', sys_modules_5757.module_type_store, module_type_store, ['_vq'])
        nest_module(stypy.reporting.localization.Localization(__file__, 78, 0), __file__, sys_modules_5757, sys_modules_5757.module_type_store, module_type_store)
    else:
        from scipy.cluster import _vq

        import_from_module(stypy.reporting.localization.Localization(__file__, 78, 0), 'scipy.cluster', None, module_type_store, ['_vq'], [_vq])

else:
    # Assigning a type to the variable 'scipy.cluster' (line 78)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 78, 0), 'scipy.cluster', import_5756)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/cluster/')


# Assigning a Str to a Name (line 80):

# Assigning a Str to a Name (line 80):
str_5758 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 80, 16), 'str', 'restructuredtext')
# Assigning a type to the variable '__docformat__' (line 80)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 80, 0), '__docformat__', str_5758)

# Assigning a List to a Name (line 82):

# Assigning a List to a Name (line 82):
__all__ = ['whiten', 'vq', 'kmeans', 'kmeans2']
module_type_store.set_exportable_members(['whiten', 'vq', 'kmeans', 'kmeans2'])

# Obtaining an instance of the builtin type 'list' (line 82)
list_5759 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 82, 10), 'list')
# Adding type elements to the builtin type 'list' instance (line 82)
# Adding element type (line 82)
str_5760 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 82, 11), 'str', 'whiten')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 82, 10), list_5759, str_5760)
# Adding element type (line 82)
str_5761 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 82, 21), 'str', 'vq')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 82, 10), list_5759, str_5761)
# Adding element type (line 82)
str_5762 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 82, 27), 'str', 'kmeans')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 82, 10), list_5759, str_5762)
# Adding element type (line 82)
str_5763 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 82, 37), 'str', 'kmeans2')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 82, 10), list_5759, str_5763)

# Assigning a type to the variable '__all__' (line 82)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 82, 0), '__all__', list_5759)
# Declaration of the 'ClusterError' class
# Getting the type of 'Exception' (line 85)
Exception_5764 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 19), 'Exception')

class ClusterError(Exception_5764, ):
    pass

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 85, 0, False)
        # Assigning a type to the variable 'self' (line 86)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 86, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'ClusterError.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'ClusterError' (line 85)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 85, 0), 'ClusterError', ClusterError)

@norecursion
def whiten(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'True' (line 89)
    True_5765 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 29), 'True')
    defaults = [True_5765]
    # Create a new context for function 'whiten'
    module_type_store = module_type_store.open_function_context('whiten', 89, 0, False)
    
    # Passed parameters checking function
    whiten.stypy_localization = localization
    whiten.stypy_type_of_self = None
    whiten.stypy_type_store = module_type_store
    whiten.stypy_function_name = 'whiten'
    whiten.stypy_param_names_list = ['obs', 'check_finite']
    whiten.stypy_varargs_param_name = None
    whiten.stypy_kwargs_param_name = None
    whiten.stypy_call_defaults = defaults
    whiten.stypy_call_varargs = varargs
    whiten.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'whiten', ['obs', 'check_finite'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'whiten', localization, ['obs', 'check_finite'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'whiten(...)' code ##################

    str_5766 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 133, (-1)), 'str', '\n    Normalize a group of observations on a per feature basis.\n\n    Before running k-means, it is beneficial to rescale each feature\n    dimension of the observation set with whitening. Each feature is\n    divided by its standard deviation across all observations to give\n    it unit variance.\n\n    Parameters\n    ----------\n    obs : ndarray\n        Each row of the array is an observation.  The\n        columns are the features seen during each observation.\n\n        >>> #         f0    f1    f2\n        >>> obs = [[  1.,   1.,   1.],  #o0\n        ...        [  2.,   2.,   2.],  #o1\n        ...        [  3.,   3.,   3.],  #o2\n        ...        [  4.,   4.,   4.]]  #o3\n\n    check_finite : bool, optional\n        Whether to check that the input matrices contain only finite numbers.\n        Disabling may give a performance gain, but may result in problems\n        (crashes, non-termination) if the inputs do contain infinities or NaNs.\n        Default: True\n\n    Returns\n    -------\n    result : ndarray\n        Contains the values in `obs` scaled by the standard deviation\n        of each column.\n\n    Examples\n    --------\n    >>> from scipy.cluster.vq import whiten\n    >>> features  = np.array([[1.9, 2.3, 1.7],\n    ...                       [1.5, 2.5, 2.2],\n    ...                       [0.8, 0.6, 1.7,]])\n    >>> whiten(features)\n    array([[ 4.17944278,  2.69811351,  7.21248917],\n           [ 3.29956009,  2.93273208,  9.33380951],\n           [ 1.75976538,  0.7038557 ,  7.21248917]])\n\n    ')
    
    # Assigning a Call to a Name (line 134):
    
    # Assigning a Call to a Name (line 134):
    
    # Call to _asarray_validated(...): (line 134)
    # Processing the call arguments (line 134)
    # Getting the type of 'obs' (line 134)
    obs_5768 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 29), 'obs', False)
    # Processing the call keyword arguments (line 134)
    # Getting the type of 'check_finite' (line 134)
    check_finite_5769 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 47), 'check_finite', False)
    keyword_5770 = check_finite_5769
    kwargs_5771 = {'check_finite': keyword_5770}
    # Getting the type of '_asarray_validated' (line 134)
    _asarray_validated_5767 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 10), '_asarray_validated', False)
    # Calling _asarray_validated(args, kwargs) (line 134)
    _asarray_validated_call_result_5772 = invoke(stypy.reporting.localization.Localization(__file__, 134, 10), _asarray_validated_5767, *[obs_5768], **kwargs_5771)
    
    # Assigning a type to the variable 'obs' (line 134)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 134, 4), 'obs', _asarray_validated_call_result_5772)
    
    # Assigning a Call to a Name (line 135):
    
    # Assigning a Call to a Name (line 135):
    
    # Call to std(...): (line 135)
    # Processing the call keyword arguments (line 135)
    int_5775 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 135, 27), 'int')
    keyword_5776 = int_5775
    kwargs_5777 = {'axis': keyword_5776}
    # Getting the type of 'obs' (line 135)
    obs_5773 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 14), 'obs', False)
    # Obtaining the member 'std' of a type (line 135)
    std_5774 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 135, 14), obs_5773, 'std')
    # Calling std(args, kwargs) (line 135)
    std_call_result_5778 = invoke(stypy.reporting.localization.Localization(__file__, 135, 14), std_5774, *[], **kwargs_5777)
    
    # Assigning a type to the variable 'std_dev' (line 135)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 135, 4), 'std_dev', std_call_result_5778)
    
    # Assigning a Compare to a Name (line 136):
    
    # Assigning a Compare to a Name (line 136):
    
    # Getting the type of 'std_dev' (line 136)
    std_dev_5779 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 20), 'std_dev')
    int_5780 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 136, 31), 'int')
    # Applying the binary operator '==' (line 136)
    result_eq_5781 = python_operator(stypy.reporting.localization.Localization(__file__, 136, 20), '==', std_dev_5779, int_5780)
    
    # Assigning a type to the variable 'zero_std_mask' (line 136)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 136, 4), 'zero_std_mask', result_eq_5781)
    
    
    # Call to any(...): (line 137)
    # Processing the call keyword arguments (line 137)
    kwargs_5784 = {}
    # Getting the type of 'zero_std_mask' (line 137)
    zero_std_mask_5782 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 7), 'zero_std_mask', False)
    # Obtaining the member 'any' of a type (line 137)
    any_5783 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 137, 7), zero_std_mask_5782, 'any')
    # Calling any(args, kwargs) (line 137)
    any_call_result_5785 = invoke(stypy.reporting.localization.Localization(__file__, 137, 7), any_5783, *[], **kwargs_5784)
    
    # Testing the type of an if condition (line 137)
    if_condition_5786 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 137, 4), any_call_result_5785)
    # Assigning a type to the variable 'if_condition_5786' (line 137)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 137, 4), 'if_condition_5786', if_condition_5786)
    # SSA begins for if statement (line 137)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Num to a Subscript (line 138):
    
    # Assigning a Num to a Subscript (line 138):
    float_5787 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 138, 33), 'float')
    # Getting the type of 'std_dev' (line 138)
    std_dev_5788 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 8), 'std_dev')
    # Getting the type of 'zero_std_mask' (line 138)
    zero_std_mask_5789 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 16), 'zero_std_mask')
    # Storing an element on a container (line 138)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 138, 8), std_dev_5788, (zero_std_mask_5789, float_5787))
    
    # Call to warn(...): (line 139)
    # Processing the call arguments (line 139)
    str_5792 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 139, 22), 'str', 'Some columns have standard deviation zero. The values of these columns will not change.')
    # Getting the type of 'RuntimeWarning' (line 141)
    RuntimeWarning_5793 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 22), 'RuntimeWarning', False)
    # Processing the call keyword arguments (line 139)
    kwargs_5794 = {}
    # Getting the type of 'warnings' (line 139)
    warnings_5790 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 8), 'warnings', False)
    # Obtaining the member 'warn' of a type (line 139)
    warn_5791 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 139, 8), warnings_5790, 'warn')
    # Calling warn(args, kwargs) (line 139)
    warn_call_result_5795 = invoke(stypy.reporting.localization.Localization(__file__, 139, 8), warn_5791, *[str_5792, RuntimeWarning_5793], **kwargs_5794)
    
    # SSA join for if statement (line 137)
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'obs' (line 142)
    obs_5796 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 11), 'obs')
    # Getting the type of 'std_dev' (line 142)
    std_dev_5797 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 17), 'std_dev')
    # Applying the binary operator 'div' (line 142)
    result_div_5798 = python_operator(stypy.reporting.localization.Localization(__file__, 142, 11), 'div', obs_5796, std_dev_5797)
    
    # Assigning a type to the variable 'stypy_return_type' (line 142)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 142, 4), 'stypy_return_type', result_div_5798)
    
    # ################# End of 'whiten(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'whiten' in the type store
    # Getting the type of 'stypy_return_type' (line 89)
    stypy_return_type_5799 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_5799)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'whiten'
    return stypy_return_type_5799

# Assigning a type to the variable 'whiten' (line 89)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 89, 0), 'whiten', whiten)

@norecursion
def vq(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'True' (line 145)
    True_5800 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 36), 'True')
    defaults = [True_5800]
    # Create a new context for function 'vq'
    module_type_store = module_type_store.open_function_context('vq', 145, 0, False)
    
    # Passed parameters checking function
    vq.stypy_localization = localization
    vq.stypy_type_of_self = None
    vq.stypy_type_store = module_type_store
    vq.stypy_function_name = 'vq'
    vq.stypy_param_names_list = ['obs', 'code_book', 'check_finite']
    vq.stypy_varargs_param_name = None
    vq.stypy_kwargs_param_name = None
    vq.stypy_call_defaults = defaults
    vq.stypy_call_varargs = varargs
    vq.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'vq', ['obs', 'code_book', 'check_finite'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'vq', localization, ['obs', 'code_book', 'check_finite'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'vq(...)' code ##################

    str_5801 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 202, (-1)), 'str', '\n    Assign codes from a code book to observations.\n\n    Assigns a code from a code book to each observation. Each\n    observation vector in the \'M\' by \'N\' `obs` array is compared with the\n    centroids in the code book and assigned the code of the closest\n    centroid.\n\n    The features in `obs` should have unit variance, which can be\n    achieved by passing them through the whiten function.  The code\n    book can be created with the k-means algorithm or a different\n    encoding algorithm.\n\n    Parameters\n    ----------\n    obs : ndarray\n        Each row of the \'M\' x \'N\' array is an observation.  The columns are\n        the "features" seen during each observation. The features must be\n        whitened first using the whiten function or something equivalent.\n    code_book : ndarray\n        The code book is usually generated using the k-means algorithm.\n        Each row of the array holds a different code, and the columns are\n        the features of the code.\n\n         >>> #              f0    f1    f2   f3\n         >>> code_book = [\n         ...             [  1.,   2.,   3.,   4.],  #c0\n         ...             [  1.,   2.,   3.,   4.],  #c1\n         ...             [  1.,   2.,   3.,   4.]]  #c2\n\n    check_finite : bool, optional\n        Whether to check that the input matrices contain only finite numbers.\n        Disabling may give a performance gain, but may result in problems\n        (crashes, non-termination) if the inputs do contain infinities or NaNs.\n        Default: True\n\n    Returns\n    -------\n    code : ndarray\n        A length M array holding the code book index for each observation.\n    dist : ndarray\n        The distortion (distance) between the observation and its nearest\n        code.\n\n    Examples\n    --------\n    >>> from numpy import array\n    >>> from scipy.cluster.vq import vq\n    >>> code_book = array([[1.,1.,1.],\n    ...                    [2.,2.,2.]])\n    >>> features  = array([[  1.9,2.3,1.7],\n    ...                    [  1.5,2.5,2.2],\n    ...                    [  0.8,0.6,1.7]])\n    >>> vq(features,code_book)\n    (array([1, 1, 0],\'i\'), array([ 0.43588989,  0.73484692,  0.83066239]))\n\n    ')
    
    # Assigning a Call to a Name (line 203):
    
    # Assigning a Call to a Name (line 203):
    
    # Call to _asarray_validated(...): (line 203)
    # Processing the call arguments (line 203)
    # Getting the type of 'obs' (line 203)
    obs_5803 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 29), 'obs', False)
    # Processing the call keyword arguments (line 203)
    # Getting the type of 'check_finite' (line 203)
    check_finite_5804 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 47), 'check_finite', False)
    keyword_5805 = check_finite_5804
    kwargs_5806 = {'check_finite': keyword_5805}
    # Getting the type of '_asarray_validated' (line 203)
    _asarray_validated_5802 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 10), '_asarray_validated', False)
    # Calling _asarray_validated(args, kwargs) (line 203)
    _asarray_validated_call_result_5807 = invoke(stypy.reporting.localization.Localization(__file__, 203, 10), _asarray_validated_5802, *[obs_5803], **kwargs_5806)
    
    # Assigning a type to the variable 'obs' (line 203)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 203, 4), 'obs', _asarray_validated_call_result_5807)
    
    # Assigning a Call to a Name (line 204):
    
    # Assigning a Call to a Name (line 204):
    
    # Call to _asarray_validated(...): (line 204)
    # Processing the call arguments (line 204)
    # Getting the type of 'code_book' (line 204)
    code_book_5809 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 204, 35), 'code_book', False)
    # Processing the call keyword arguments (line 204)
    # Getting the type of 'check_finite' (line 204)
    check_finite_5810 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 204, 59), 'check_finite', False)
    keyword_5811 = check_finite_5810
    kwargs_5812 = {'check_finite': keyword_5811}
    # Getting the type of '_asarray_validated' (line 204)
    _asarray_validated_5808 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 204, 16), '_asarray_validated', False)
    # Calling _asarray_validated(args, kwargs) (line 204)
    _asarray_validated_call_result_5813 = invoke(stypy.reporting.localization.Localization(__file__, 204, 16), _asarray_validated_5808, *[code_book_5809], **kwargs_5812)
    
    # Assigning a type to the variable 'code_book' (line 204)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 204, 4), 'code_book', _asarray_validated_call_result_5813)
    
    # Assigning a Call to a Name (line 205):
    
    # Assigning a Call to a Name (line 205):
    
    # Call to common_type(...): (line 205)
    # Processing the call arguments (line 205)
    # Getting the type of 'obs' (line 205)
    obs_5816 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 24), 'obs', False)
    # Getting the type of 'code_book' (line 205)
    code_book_5817 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 29), 'code_book', False)
    # Processing the call keyword arguments (line 205)
    kwargs_5818 = {}
    # Getting the type of 'np' (line 205)
    np_5814 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 9), 'np', False)
    # Obtaining the member 'common_type' of a type (line 205)
    common_type_5815 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 205, 9), np_5814, 'common_type')
    # Calling common_type(args, kwargs) (line 205)
    common_type_call_result_5819 = invoke(stypy.reporting.localization.Localization(__file__, 205, 9), common_type_5815, *[obs_5816, code_book_5817], **kwargs_5818)
    
    # Assigning a type to the variable 'ct' (line 205)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 205, 4), 'ct', common_type_call_result_5819)
    
    # Assigning a Call to a Name (line 207):
    
    # Assigning a Call to a Name (line 207):
    
    # Call to astype(...): (line 207)
    # Processing the call arguments (line 207)
    # Getting the type of 'ct' (line 207)
    ct_5822 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 207, 23), 'ct', False)
    # Processing the call keyword arguments (line 207)
    # Getting the type of 'False' (line 207)
    False_5823 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 207, 32), 'False', False)
    keyword_5824 = False_5823
    kwargs_5825 = {'copy': keyword_5824}
    # Getting the type of 'obs' (line 207)
    obs_5820 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 207, 12), 'obs', False)
    # Obtaining the member 'astype' of a type (line 207)
    astype_5821 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 207, 12), obs_5820, 'astype')
    # Calling astype(args, kwargs) (line 207)
    astype_call_result_5826 = invoke(stypy.reporting.localization.Localization(__file__, 207, 12), astype_5821, *[ct_5822], **kwargs_5825)
    
    # Assigning a type to the variable 'c_obs' (line 207)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 207, 4), 'c_obs', astype_call_result_5826)
    
    # Assigning a Call to a Name (line 208):
    
    # Assigning a Call to a Name (line 208):
    
    # Call to astype(...): (line 208)
    # Processing the call arguments (line 208)
    # Getting the type of 'ct' (line 208)
    ct_5829 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 208, 35), 'ct', False)
    # Processing the call keyword arguments (line 208)
    # Getting the type of 'False' (line 208)
    False_5830 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 208, 44), 'False', False)
    keyword_5831 = False_5830
    kwargs_5832 = {'copy': keyword_5831}
    # Getting the type of 'code_book' (line 208)
    code_book_5827 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 208, 18), 'code_book', False)
    # Obtaining the member 'astype' of a type (line 208)
    astype_5828 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 208, 18), code_book_5827, 'astype')
    # Calling astype(args, kwargs) (line 208)
    astype_call_result_5833 = invoke(stypy.reporting.localization.Localization(__file__, 208, 18), astype_5828, *[ct_5829], **kwargs_5832)
    
    # Assigning a type to the variable 'c_code_book' (line 208)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 208, 4), 'c_code_book', astype_call_result_5833)
    
    
    # Evaluating a boolean operation
    
    # Call to issubdtype(...): (line 210)
    # Processing the call arguments (line 210)
    # Getting the type of 'ct' (line 210)
    ct_5836 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 21), 'ct', False)
    # Getting the type of 'np' (line 210)
    np_5837 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 25), 'np', False)
    # Obtaining the member 'float64' of a type (line 210)
    float64_5838 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 210, 25), np_5837, 'float64')
    # Processing the call keyword arguments (line 210)
    kwargs_5839 = {}
    # Getting the type of 'np' (line 210)
    np_5834 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 7), 'np', False)
    # Obtaining the member 'issubdtype' of a type (line 210)
    issubdtype_5835 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 210, 7), np_5834, 'issubdtype')
    # Calling issubdtype(args, kwargs) (line 210)
    issubdtype_call_result_5840 = invoke(stypy.reporting.localization.Localization(__file__, 210, 7), issubdtype_5835, *[ct_5836, float64_5838], **kwargs_5839)
    
    
    # Call to issubdtype(...): (line 210)
    # Processing the call arguments (line 210)
    # Getting the type of 'ct' (line 210)
    ct_5843 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 54), 'ct', False)
    # Getting the type of 'np' (line 210)
    np_5844 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 58), 'np', False)
    # Obtaining the member 'float32' of a type (line 210)
    float32_5845 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 210, 58), np_5844, 'float32')
    # Processing the call keyword arguments (line 210)
    kwargs_5846 = {}
    # Getting the type of 'np' (line 210)
    np_5841 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 40), 'np', False)
    # Obtaining the member 'issubdtype' of a type (line 210)
    issubdtype_5842 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 210, 40), np_5841, 'issubdtype')
    # Calling issubdtype(args, kwargs) (line 210)
    issubdtype_call_result_5847 = invoke(stypy.reporting.localization.Localization(__file__, 210, 40), issubdtype_5842, *[ct_5843, float32_5845], **kwargs_5846)
    
    # Applying the binary operator 'or' (line 210)
    result_or_keyword_5848 = python_operator(stypy.reporting.localization.Localization(__file__, 210, 7), 'or', issubdtype_call_result_5840, issubdtype_call_result_5847)
    
    # Testing the type of an if condition (line 210)
    if_condition_5849 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 210, 4), result_or_keyword_5848)
    # Assigning a type to the variable 'if_condition_5849' (line 210)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 210, 4), 'if_condition_5849', if_condition_5849)
    # SSA begins for if statement (line 210)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to vq(...): (line 211)
    # Processing the call arguments (line 211)
    # Getting the type of 'c_obs' (line 211)
    c_obs_5852 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 211, 22), 'c_obs', False)
    # Getting the type of 'c_code_book' (line 211)
    c_code_book_5853 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 211, 29), 'c_code_book', False)
    # Processing the call keyword arguments (line 211)
    kwargs_5854 = {}
    # Getting the type of '_vq' (line 211)
    _vq_5850 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 211, 15), '_vq', False)
    # Obtaining the member 'vq' of a type (line 211)
    vq_5851 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 211, 15), _vq_5850, 'vq')
    # Calling vq(args, kwargs) (line 211)
    vq_call_result_5855 = invoke(stypy.reporting.localization.Localization(__file__, 211, 15), vq_5851, *[c_obs_5852, c_code_book_5853], **kwargs_5854)
    
    # Assigning a type to the variable 'stypy_return_type' (line 211)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 211, 8), 'stypy_return_type', vq_call_result_5855)
    # SSA join for if statement (line 210)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to py_vq(...): (line 212)
    # Processing the call arguments (line 212)
    # Getting the type of 'obs' (line 212)
    obs_5857 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 17), 'obs', False)
    # Getting the type of 'code_book' (line 212)
    code_book_5858 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 22), 'code_book', False)
    # Processing the call keyword arguments (line 212)
    # Getting the type of 'False' (line 212)
    False_5859 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 46), 'False', False)
    keyword_5860 = False_5859
    kwargs_5861 = {'check_finite': keyword_5860}
    # Getting the type of 'py_vq' (line 212)
    py_vq_5856 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 11), 'py_vq', False)
    # Calling py_vq(args, kwargs) (line 212)
    py_vq_call_result_5862 = invoke(stypy.reporting.localization.Localization(__file__, 212, 11), py_vq_5856, *[obs_5857, code_book_5858], **kwargs_5861)
    
    # Assigning a type to the variable 'stypy_return_type' (line 212)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 212, 4), 'stypy_return_type', py_vq_call_result_5862)
    
    # ################# End of 'vq(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'vq' in the type store
    # Getting the type of 'stypy_return_type' (line 145)
    stypy_return_type_5863 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_5863)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'vq'
    return stypy_return_type_5863

# Assigning a type to the variable 'vq' (line 145)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 145, 0), 'vq', vq)

@norecursion
def py_vq(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'True' (line 215)
    True_5864 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 215, 39), 'True')
    defaults = [True_5864]
    # Create a new context for function 'py_vq'
    module_type_store = module_type_store.open_function_context('py_vq', 215, 0, False)
    
    # Passed parameters checking function
    py_vq.stypy_localization = localization
    py_vq.stypy_type_of_self = None
    py_vq.stypy_type_store = module_type_store
    py_vq.stypy_function_name = 'py_vq'
    py_vq.stypy_param_names_list = ['obs', 'code_book', 'check_finite']
    py_vq.stypy_varargs_param_name = None
    py_vq.stypy_kwargs_param_name = None
    py_vq.stypy_call_defaults = defaults
    py_vq.stypy_call_varargs = varargs
    py_vq.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'py_vq', ['obs', 'code_book', 'check_finite'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'py_vq', localization, ['obs', 'code_book', 'check_finite'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'py_vq(...)' code ##################

    str_5865 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 251, (-1)), 'str', ' Python version of vq algorithm.\n\n    The algorithm computes the euclidian distance between each\n    observation and every frame in the code_book.\n\n    Parameters\n    ----------\n    obs : ndarray\n        Expects a rank 2 array. Each row is one observation.\n    code_book : ndarray\n        Code book to use. Same format than obs. Should have same number of\n        features (eg columns) than obs.\n    check_finite : bool, optional\n        Whether to check that the input matrices contain only finite numbers.\n        Disabling may give a performance gain, but may result in problems\n        (crashes, non-termination) if the inputs do contain infinities or NaNs.\n        Default: True\n\n    Returns\n    -------\n    code : ndarray\n        code[i] gives the label of the ith obversation, that its code is\n        code_book[code[i]].\n    mind_dist : ndarray\n        min_dist[i] gives the distance between the ith observation and its\n        corresponding code.\n\n    Notes\n    -----\n    This function is slower than the C version but works for\n    all input types.  If the inputs have the wrong types for the\n    C versions of the function, this one is called as a last resort.\n\n    It is about 20 times slower than the C version.\n\n    ')
    
    # Assigning a Call to a Name (line 252):
    
    # Assigning a Call to a Name (line 252):
    
    # Call to _asarray_validated(...): (line 252)
    # Processing the call arguments (line 252)
    # Getting the type of 'obs' (line 252)
    obs_5867 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 252, 29), 'obs', False)
    # Processing the call keyword arguments (line 252)
    # Getting the type of 'check_finite' (line 252)
    check_finite_5868 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 252, 47), 'check_finite', False)
    keyword_5869 = check_finite_5868
    kwargs_5870 = {'check_finite': keyword_5869}
    # Getting the type of '_asarray_validated' (line 252)
    _asarray_validated_5866 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 252, 10), '_asarray_validated', False)
    # Calling _asarray_validated(args, kwargs) (line 252)
    _asarray_validated_call_result_5871 = invoke(stypy.reporting.localization.Localization(__file__, 252, 10), _asarray_validated_5866, *[obs_5867], **kwargs_5870)
    
    # Assigning a type to the variable 'obs' (line 252)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 252, 4), 'obs', _asarray_validated_call_result_5871)
    
    # Assigning a Call to a Name (line 253):
    
    # Assigning a Call to a Name (line 253):
    
    # Call to _asarray_validated(...): (line 253)
    # Processing the call arguments (line 253)
    # Getting the type of 'code_book' (line 253)
    code_book_5873 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 253, 35), 'code_book', False)
    # Processing the call keyword arguments (line 253)
    # Getting the type of 'check_finite' (line 253)
    check_finite_5874 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 253, 59), 'check_finite', False)
    keyword_5875 = check_finite_5874
    kwargs_5876 = {'check_finite': keyword_5875}
    # Getting the type of '_asarray_validated' (line 253)
    _asarray_validated_5872 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 253, 16), '_asarray_validated', False)
    # Calling _asarray_validated(args, kwargs) (line 253)
    _asarray_validated_call_result_5877 = invoke(stypy.reporting.localization.Localization(__file__, 253, 16), _asarray_validated_5872, *[code_book_5873], **kwargs_5876)
    
    # Assigning a type to the variable 'code_book' (line 253)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 253, 4), 'code_book', _asarray_validated_call_result_5877)
    
    
    # Getting the type of 'obs' (line 255)
    obs_5878 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 255, 7), 'obs')
    # Obtaining the member 'ndim' of a type (line 255)
    ndim_5879 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 255, 7), obs_5878, 'ndim')
    # Getting the type of 'code_book' (line 255)
    code_book_5880 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 255, 19), 'code_book')
    # Obtaining the member 'ndim' of a type (line 255)
    ndim_5881 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 255, 19), code_book_5880, 'ndim')
    # Applying the binary operator '!=' (line 255)
    result_ne_5882 = python_operator(stypy.reporting.localization.Localization(__file__, 255, 7), '!=', ndim_5879, ndim_5881)
    
    # Testing the type of an if condition (line 255)
    if_condition_5883 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 255, 4), result_ne_5882)
    # Assigning a type to the variable 'if_condition_5883' (line 255)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 255, 4), 'if_condition_5883', if_condition_5883)
    # SSA begins for if statement (line 255)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 256)
    # Processing the call arguments (line 256)
    str_5885 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 256, 25), 'str', 'Observation and code_book should have the same rank')
    # Processing the call keyword arguments (line 256)
    kwargs_5886 = {}
    # Getting the type of 'ValueError' (line 256)
    ValueError_5884 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 256, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 256)
    ValueError_call_result_5887 = invoke(stypy.reporting.localization.Localization(__file__, 256, 14), ValueError_5884, *[str_5885], **kwargs_5886)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 256, 8), ValueError_call_result_5887, 'raise parameter', BaseException)
    # SSA join for if statement (line 255)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'obs' (line 258)
    obs_5888 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 258, 7), 'obs')
    # Obtaining the member 'ndim' of a type (line 258)
    ndim_5889 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 258, 7), obs_5888, 'ndim')
    int_5890 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 258, 19), 'int')
    # Applying the binary operator '==' (line 258)
    result_eq_5891 = python_operator(stypy.reporting.localization.Localization(__file__, 258, 7), '==', ndim_5889, int_5890)
    
    # Testing the type of an if condition (line 258)
    if_condition_5892 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 258, 4), result_eq_5891)
    # Assigning a type to the variable 'if_condition_5892' (line 258)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 258, 4), 'if_condition_5892', if_condition_5892)
    # SSA begins for if statement (line 258)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Subscript to a Name (line 259):
    
    # Assigning a Subscript to a Name (line 259):
    
    # Obtaining the type of the subscript
    slice_5893 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 259, 14), None, None, None)
    # Getting the type of 'np' (line 259)
    np_5894 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 259, 21), 'np')
    # Obtaining the member 'newaxis' of a type (line 259)
    newaxis_5895 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 259, 21), np_5894, 'newaxis')
    # Getting the type of 'obs' (line 259)
    obs_5896 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 259, 14), 'obs')
    # Obtaining the member '__getitem__' of a type (line 259)
    getitem___5897 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 259, 14), obs_5896, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 259)
    subscript_call_result_5898 = invoke(stypy.reporting.localization.Localization(__file__, 259, 14), getitem___5897, (slice_5893, newaxis_5895))
    
    # Assigning a type to the variable 'obs' (line 259)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 259, 8), 'obs', subscript_call_result_5898)
    
    # Assigning a Subscript to a Name (line 260):
    
    # Assigning a Subscript to a Name (line 260):
    
    # Obtaining the type of the subscript
    slice_5899 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 260, 20), None, None, None)
    # Getting the type of 'np' (line 260)
    np_5900 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 260, 33), 'np')
    # Obtaining the member 'newaxis' of a type (line 260)
    newaxis_5901 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 260, 33), np_5900, 'newaxis')
    # Getting the type of 'code_book' (line 260)
    code_book_5902 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 260, 20), 'code_book')
    # Obtaining the member '__getitem__' of a type (line 260)
    getitem___5903 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 260, 20), code_book_5902, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 260)
    subscript_call_result_5904 = invoke(stypy.reporting.localization.Localization(__file__, 260, 20), getitem___5903, (slice_5899, newaxis_5901))
    
    # Assigning a type to the variable 'code_book' (line 260)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 260, 8), 'code_book', subscript_call_result_5904)
    # SSA join for if statement (line 258)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 262):
    
    # Assigning a Call to a Name (line 262):
    
    # Call to cdist(...): (line 262)
    # Processing the call arguments (line 262)
    # Getting the type of 'obs' (line 262)
    obs_5906 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 262, 17), 'obs', False)
    # Getting the type of 'code_book' (line 262)
    code_book_5907 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 262, 22), 'code_book', False)
    # Processing the call keyword arguments (line 262)
    kwargs_5908 = {}
    # Getting the type of 'cdist' (line 262)
    cdist_5905 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 262, 11), 'cdist', False)
    # Calling cdist(args, kwargs) (line 262)
    cdist_call_result_5909 = invoke(stypy.reporting.localization.Localization(__file__, 262, 11), cdist_5905, *[obs_5906, code_book_5907], **kwargs_5908)
    
    # Assigning a type to the variable 'dist' (line 262)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 262, 4), 'dist', cdist_call_result_5909)
    
    # Assigning a Call to a Name (line 263):
    
    # Assigning a Call to a Name (line 263):
    
    # Call to argmin(...): (line 263)
    # Processing the call keyword arguments (line 263)
    int_5912 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 263, 28), 'int')
    keyword_5913 = int_5912
    kwargs_5914 = {'axis': keyword_5913}
    # Getting the type of 'dist' (line 263)
    dist_5910 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 263, 11), 'dist', False)
    # Obtaining the member 'argmin' of a type (line 263)
    argmin_5911 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 263, 11), dist_5910, 'argmin')
    # Calling argmin(args, kwargs) (line 263)
    argmin_call_result_5915 = invoke(stypy.reporting.localization.Localization(__file__, 263, 11), argmin_5911, *[], **kwargs_5914)
    
    # Assigning a type to the variable 'code' (line 263)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 263, 4), 'code', argmin_call_result_5915)
    
    # Assigning a Subscript to a Name (line 264):
    
    # Assigning a Subscript to a Name (line 264):
    
    # Obtaining the type of the subscript
    
    # Obtaining an instance of the builtin type 'tuple' (line 264)
    tuple_5916 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 264, 20), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 264)
    # Adding element type (line 264)
    
    # Call to arange(...): (line 264)
    # Processing the call arguments (line 264)
    
    # Call to len(...): (line 264)
    # Processing the call arguments (line 264)
    # Getting the type of 'code' (line 264)
    code_5920 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 264, 34), 'code', False)
    # Processing the call keyword arguments (line 264)
    kwargs_5921 = {}
    # Getting the type of 'len' (line 264)
    len_5919 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 264, 30), 'len', False)
    # Calling len(args, kwargs) (line 264)
    len_call_result_5922 = invoke(stypy.reporting.localization.Localization(__file__, 264, 30), len_5919, *[code_5920], **kwargs_5921)
    
    # Processing the call keyword arguments (line 264)
    kwargs_5923 = {}
    # Getting the type of 'np' (line 264)
    np_5917 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 264, 20), 'np', False)
    # Obtaining the member 'arange' of a type (line 264)
    arange_5918 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 264, 20), np_5917, 'arange')
    # Calling arange(args, kwargs) (line 264)
    arange_call_result_5924 = invoke(stypy.reporting.localization.Localization(__file__, 264, 20), arange_5918, *[len_call_result_5922], **kwargs_5923)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 264, 20), tuple_5916, arange_call_result_5924)
    # Adding element type (line 264)
    # Getting the type of 'code' (line 264)
    code_5925 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 264, 42), 'code')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 264, 20), tuple_5916, code_5925)
    
    # Getting the type of 'dist' (line 264)
    dist_5926 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 264, 15), 'dist')
    # Obtaining the member '__getitem__' of a type (line 264)
    getitem___5927 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 264, 15), dist_5926, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 264)
    subscript_call_result_5928 = invoke(stypy.reporting.localization.Localization(__file__, 264, 15), getitem___5927, tuple_5916)
    
    # Assigning a type to the variable 'min_dist' (line 264)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 264, 4), 'min_dist', subscript_call_result_5928)
    
    # Obtaining an instance of the builtin type 'tuple' (line 265)
    tuple_5929 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 265, 11), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 265)
    # Adding element type (line 265)
    # Getting the type of 'code' (line 265)
    code_5930 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 265, 11), 'code')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 265, 11), tuple_5929, code_5930)
    # Adding element type (line 265)
    # Getting the type of 'min_dist' (line 265)
    min_dist_5931 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 265, 17), 'min_dist')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 265, 11), tuple_5929, min_dist_5931)
    
    # Assigning a type to the variable 'stypy_return_type' (line 265)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 265, 4), 'stypy_return_type', tuple_5929)
    
    # ################# End of 'py_vq(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'py_vq' in the type store
    # Getting the type of 'stypy_return_type' (line 215)
    stypy_return_type_5932 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 215, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_5932)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'py_vq'
    return stypy_return_type_5932

# Assigning a type to the variable 'py_vq' (line 215)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 215, 0), 'py_vq', py_vq)

# Assigning a Call to a Name (line 268):

# Assigning a Call to a Name (line 268):

# Call to deprecate(...): (line 268)
# Processing the call arguments (line 268)
# Getting the type of 'py_vq' (line 268)
py_vq_5935 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 268, 22), 'py_vq', False)
# Processing the call keyword arguments (line 268)
str_5936 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 268, 38), 'str', 'py_vq2')
keyword_5937 = str_5936
str_5938 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 268, 57), 'str', 'py_vq')
keyword_5939 = str_5938
kwargs_5940 = {'new_name': keyword_5939, 'old_name': keyword_5937}
# Getting the type of 'np' (line 268)
np_5933 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 268, 9), 'np', False)
# Obtaining the member 'deprecate' of a type (line 268)
deprecate_5934 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 268, 9), np_5933, 'deprecate')
# Calling deprecate(args, kwargs) (line 268)
deprecate_call_result_5941 = invoke(stypy.reporting.localization.Localization(__file__, 268, 9), deprecate_5934, *[py_vq_5935], **kwargs_5940)

# Assigning a type to the variable 'py_vq2' (line 268)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 268, 0), 'py_vq2', deprecate_call_result_5941)

@norecursion
def _kmeans(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    float_5942 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 271, 31), 'float')
    defaults = [float_5942]
    # Create a new context for function '_kmeans'
    module_type_store = module_type_store.open_function_context('_kmeans', 271, 0, False)
    
    # Passed parameters checking function
    _kmeans.stypy_localization = localization
    _kmeans.stypy_type_of_self = None
    _kmeans.stypy_type_store = module_type_store
    _kmeans.stypy_function_name = '_kmeans'
    _kmeans.stypy_param_names_list = ['obs', 'guess', 'thresh']
    _kmeans.stypy_varargs_param_name = None
    _kmeans.stypy_kwargs_param_name = None
    _kmeans.stypy_call_defaults = defaults
    _kmeans.stypy_call_varargs = varargs
    _kmeans.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_kmeans', ['obs', 'guess', 'thresh'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_kmeans', localization, ['obs', 'guess', 'thresh'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_kmeans(...)' code ##################

    str_5943 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 302, (-1)), 'str', ' "raw" version of k-means.\n\n    Returns\n    -------\n    code_book\n        the lowest distortion codebook found.\n    avg_dist\n        the average distance a observation is from a code in the book.\n        Lower means the code_book matches the data better.\n\n    See Also\n    --------\n    kmeans : wrapper around k-means\n\n    Examples\n    --------\n    Note: not whitened in this example.\n\n    >>> from numpy import array\n    >>> from scipy.cluster.vq import _kmeans\n    >>> features  = array([[ 1.9,2.3],\n    ...                    [ 1.5,2.5],\n    ...                    [ 0.8,0.6],\n    ...                    [ 0.4,1.8],\n    ...                    [ 1.0,1.0]])\n    >>> book = array((features[0],features[2]))\n    >>> _kmeans(features,book)\n    (array([[ 1.7       ,  2.4       ],\n           [ 0.73333333,  1.13333333]]), 0.40563916697728591)\n\n    ')
    
    # Assigning a Call to a Name (line 304):
    
    # Assigning a Call to a Name (line 304):
    
    # Call to asarray(...): (line 304)
    # Processing the call arguments (line 304)
    # Getting the type of 'guess' (line 304)
    guess_5946 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 304, 27), 'guess', False)
    # Processing the call keyword arguments (line 304)
    kwargs_5947 = {}
    # Getting the type of 'np' (line 304)
    np_5944 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 304, 16), 'np', False)
    # Obtaining the member 'asarray' of a type (line 304)
    asarray_5945 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 304, 16), np_5944, 'asarray')
    # Calling asarray(args, kwargs) (line 304)
    asarray_call_result_5948 = invoke(stypy.reporting.localization.Localization(__file__, 304, 16), asarray_5945, *[guess_5946], **kwargs_5947)
    
    # Assigning a type to the variable 'code_book' (line 304)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 304, 4), 'code_book', asarray_call_result_5948)
    
    # Assigning a Attribute to a Name (line 305):
    
    # Assigning a Attribute to a Name (line 305):
    # Getting the type of 'np' (line 305)
    np_5949 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 305, 11), 'np')
    # Obtaining the member 'inf' of a type (line 305)
    inf_5950 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 305, 11), np_5949, 'inf')
    # Assigning a type to the variable 'diff' (line 305)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 305, 4), 'diff', inf_5950)
    
    # Assigning a Call to a Name (line 306):
    
    # Assigning a Call to a Name (line 306):
    
    # Call to deque(...): (line 306)
    # Processing the call arguments (line 306)
    
    # Obtaining an instance of the builtin type 'list' (line 306)
    list_5952 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 306, 27), 'list')
    # Adding type elements to the builtin type 'list' instance (line 306)
    # Adding element type (line 306)
    # Getting the type of 'diff' (line 306)
    diff_5953 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 306, 28), 'diff', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 306, 27), list_5952, diff_5953)
    
    # Processing the call keyword arguments (line 306)
    int_5954 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 306, 42), 'int')
    keyword_5955 = int_5954
    kwargs_5956 = {'maxlen': keyword_5955}
    # Getting the type of 'deque' (line 306)
    deque_5951 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 306, 21), 'deque', False)
    # Calling deque(args, kwargs) (line 306)
    deque_call_result_5957 = invoke(stypy.reporting.localization.Localization(__file__, 306, 21), deque_5951, *[list_5952], **kwargs_5956)
    
    # Assigning a type to the variable 'prev_avg_dists' (line 306)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 306, 4), 'prev_avg_dists', deque_call_result_5957)
    
    
    # Getting the type of 'diff' (line 307)
    diff_5958 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 307, 10), 'diff')
    # Getting the type of 'thresh' (line 307)
    thresh_5959 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 307, 17), 'thresh')
    # Applying the binary operator '>' (line 307)
    result_gt_5960 = python_operator(stypy.reporting.localization.Localization(__file__, 307, 10), '>', diff_5958, thresh_5959)
    
    # Testing the type of an if condition (line 307)
    is_suitable_condition(stypy.reporting.localization.Localization(__file__, 307, 4), result_gt_5960)
    # SSA begins for while statement (line 307)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'while loop')
    
    # Assigning a Call to a Tuple (line 309):
    
    # Assigning a Subscript to a Name (line 309):
    
    # Obtaining the type of the subscript
    int_5961 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 309, 8), 'int')
    
    # Call to vq(...): (line 309)
    # Processing the call arguments (line 309)
    # Getting the type of 'obs' (line 309)
    obs_5963 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 309, 31), 'obs', False)
    # Getting the type of 'code_book' (line 309)
    code_book_5964 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 309, 36), 'code_book', False)
    # Processing the call keyword arguments (line 309)
    # Getting the type of 'False' (line 309)
    False_5965 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 309, 60), 'False', False)
    keyword_5966 = False_5965
    kwargs_5967 = {'check_finite': keyword_5966}
    # Getting the type of 'vq' (line 309)
    vq_5962 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 309, 28), 'vq', False)
    # Calling vq(args, kwargs) (line 309)
    vq_call_result_5968 = invoke(stypy.reporting.localization.Localization(__file__, 309, 28), vq_5962, *[obs_5963, code_book_5964], **kwargs_5967)
    
    # Obtaining the member '__getitem__' of a type (line 309)
    getitem___5969 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 309, 8), vq_call_result_5968, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 309)
    subscript_call_result_5970 = invoke(stypy.reporting.localization.Localization(__file__, 309, 8), getitem___5969, int_5961)
    
    # Assigning a type to the variable 'tuple_var_assignment_5736' (line 309)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 309, 8), 'tuple_var_assignment_5736', subscript_call_result_5970)
    
    # Assigning a Subscript to a Name (line 309):
    
    # Obtaining the type of the subscript
    int_5971 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 309, 8), 'int')
    
    # Call to vq(...): (line 309)
    # Processing the call arguments (line 309)
    # Getting the type of 'obs' (line 309)
    obs_5973 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 309, 31), 'obs', False)
    # Getting the type of 'code_book' (line 309)
    code_book_5974 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 309, 36), 'code_book', False)
    # Processing the call keyword arguments (line 309)
    # Getting the type of 'False' (line 309)
    False_5975 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 309, 60), 'False', False)
    keyword_5976 = False_5975
    kwargs_5977 = {'check_finite': keyword_5976}
    # Getting the type of 'vq' (line 309)
    vq_5972 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 309, 28), 'vq', False)
    # Calling vq(args, kwargs) (line 309)
    vq_call_result_5978 = invoke(stypy.reporting.localization.Localization(__file__, 309, 28), vq_5972, *[obs_5973, code_book_5974], **kwargs_5977)
    
    # Obtaining the member '__getitem__' of a type (line 309)
    getitem___5979 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 309, 8), vq_call_result_5978, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 309)
    subscript_call_result_5980 = invoke(stypy.reporting.localization.Localization(__file__, 309, 8), getitem___5979, int_5971)
    
    # Assigning a type to the variable 'tuple_var_assignment_5737' (line 309)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 309, 8), 'tuple_var_assignment_5737', subscript_call_result_5980)
    
    # Assigning a Name to a Name (line 309):
    # Getting the type of 'tuple_var_assignment_5736' (line 309)
    tuple_var_assignment_5736_5981 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 309, 8), 'tuple_var_assignment_5736')
    # Assigning a type to the variable 'obs_code' (line 309)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 309, 8), 'obs_code', tuple_var_assignment_5736_5981)
    
    # Assigning a Name to a Name (line 309):
    # Getting the type of 'tuple_var_assignment_5737' (line 309)
    tuple_var_assignment_5737_5982 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 309, 8), 'tuple_var_assignment_5737')
    # Assigning a type to the variable 'distort' (line 309)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 309, 18), 'distort', tuple_var_assignment_5737_5982)
    
    # Call to append(...): (line 310)
    # Processing the call arguments (line 310)
    
    # Call to mean(...): (line 310)
    # Processing the call keyword arguments (line 310)
    int_5987 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 310, 48), 'int')
    keyword_5988 = int_5987
    kwargs_5989 = {'axis': keyword_5988}
    # Getting the type of 'distort' (line 310)
    distort_5985 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 310, 30), 'distort', False)
    # Obtaining the member 'mean' of a type (line 310)
    mean_5986 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 310, 30), distort_5985, 'mean')
    # Calling mean(args, kwargs) (line 310)
    mean_call_result_5990 = invoke(stypy.reporting.localization.Localization(__file__, 310, 30), mean_5986, *[], **kwargs_5989)
    
    # Processing the call keyword arguments (line 310)
    kwargs_5991 = {}
    # Getting the type of 'prev_avg_dists' (line 310)
    prev_avg_dists_5983 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 310, 8), 'prev_avg_dists', False)
    # Obtaining the member 'append' of a type (line 310)
    append_5984 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 310, 8), prev_avg_dists_5983, 'append')
    # Calling append(args, kwargs) (line 310)
    append_call_result_5992 = invoke(stypy.reporting.localization.Localization(__file__, 310, 8), append_5984, *[mean_call_result_5990], **kwargs_5991)
    
    
    # Assigning a Call to a Tuple (line 312):
    
    # Assigning a Subscript to a Name (line 312):
    
    # Obtaining the type of the subscript
    int_5993 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 312, 8), 'int')
    
    # Call to update_cluster_means(...): (line 312)
    # Processing the call arguments (line 312)
    # Getting the type of 'obs' (line 312)
    obs_5996 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 312, 58), 'obs', False)
    # Getting the type of 'obs_code' (line 312)
    obs_code_5997 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 312, 63), 'obs_code', False)
    
    # Obtaining the type of the subscript
    int_5998 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 313, 74), 'int')
    # Getting the type of 'code_book' (line 313)
    code_book_5999 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 313, 58), 'code_book', False)
    # Obtaining the member 'shape' of a type (line 313)
    shape_6000 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 313, 58), code_book_5999, 'shape')
    # Obtaining the member '__getitem__' of a type (line 313)
    getitem___6001 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 313, 58), shape_6000, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 313)
    subscript_call_result_6002 = invoke(stypy.reporting.localization.Localization(__file__, 313, 58), getitem___6001, int_5998)
    
    # Processing the call keyword arguments (line 312)
    kwargs_6003 = {}
    # Getting the type of '_vq' (line 312)
    _vq_5994 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 312, 33), '_vq', False)
    # Obtaining the member 'update_cluster_means' of a type (line 312)
    update_cluster_means_5995 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 312, 33), _vq_5994, 'update_cluster_means')
    # Calling update_cluster_means(args, kwargs) (line 312)
    update_cluster_means_call_result_6004 = invoke(stypy.reporting.localization.Localization(__file__, 312, 33), update_cluster_means_5995, *[obs_5996, obs_code_5997, subscript_call_result_6002], **kwargs_6003)
    
    # Obtaining the member '__getitem__' of a type (line 312)
    getitem___6005 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 312, 8), update_cluster_means_call_result_6004, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 312)
    subscript_call_result_6006 = invoke(stypy.reporting.localization.Localization(__file__, 312, 8), getitem___6005, int_5993)
    
    # Assigning a type to the variable 'tuple_var_assignment_5738' (line 312)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 312, 8), 'tuple_var_assignment_5738', subscript_call_result_6006)
    
    # Assigning a Subscript to a Name (line 312):
    
    # Obtaining the type of the subscript
    int_6007 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 312, 8), 'int')
    
    # Call to update_cluster_means(...): (line 312)
    # Processing the call arguments (line 312)
    # Getting the type of 'obs' (line 312)
    obs_6010 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 312, 58), 'obs', False)
    # Getting the type of 'obs_code' (line 312)
    obs_code_6011 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 312, 63), 'obs_code', False)
    
    # Obtaining the type of the subscript
    int_6012 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 313, 74), 'int')
    # Getting the type of 'code_book' (line 313)
    code_book_6013 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 313, 58), 'code_book', False)
    # Obtaining the member 'shape' of a type (line 313)
    shape_6014 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 313, 58), code_book_6013, 'shape')
    # Obtaining the member '__getitem__' of a type (line 313)
    getitem___6015 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 313, 58), shape_6014, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 313)
    subscript_call_result_6016 = invoke(stypy.reporting.localization.Localization(__file__, 313, 58), getitem___6015, int_6012)
    
    # Processing the call keyword arguments (line 312)
    kwargs_6017 = {}
    # Getting the type of '_vq' (line 312)
    _vq_6008 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 312, 33), '_vq', False)
    # Obtaining the member 'update_cluster_means' of a type (line 312)
    update_cluster_means_6009 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 312, 33), _vq_6008, 'update_cluster_means')
    # Calling update_cluster_means(args, kwargs) (line 312)
    update_cluster_means_call_result_6018 = invoke(stypy.reporting.localization.Localization(__file__, 312, 33), update_cluster_means_6009, *[obs_6010, obs_code_6011, subscript_call_result_6016], **kwargs_6017)
    
    # Obtaining the member '__getitem__' of a type (line 312)
    getitem___6019 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 312, 8), update_cluster_means_call_result_6018, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 312)
    subscript_call_result_6020 = invoke(stypy.reporting.localization.Localization(__file__, 312, 8), getitem___6019, int_6007)
    
    # Assigning a type to the variable 'tuple_var_assignment_5739' (line 312)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 312, 8), 'tuple_var_assignment_5739', subscript_call_result_6020)
    
    # Assigning a Name to a Name (line 312):
    # Getting the type of 'tuple_var_assignment_5738' (line 312)
    tuple_var_assignment_5738_6021 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 312, 8), 'tuple_var_assignment_5738')
    # Assigning a type to the variable 'code_book' (line 312)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 312, 8), 'code_book', tuple_var_assignment_5738_6021)
    
    # Assigning a Name to a Name (line 312):
    # Getting the type of 'tuple_var_assignment_5739' (line 312)
    tuple_var_assignment_5739_6022 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 312, 8), 'tuple_var_assignment_5739')
    # Assigning a type to the variable 'has_members' (line 312)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 312, 19), 'has_members', tuple_var_assignment_5739_6022)
    
    # Assigning a Subscript to a Name (line 314):
    
    # Assigning a Subscript to a Name (line 314):
    
    # Obtaining the type of the subscript
    # Getting the type of 'has_members' (line 314)
    has_members_6023 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 314, 30), 'has_members')
    # Getting the type of 'code_book' (line 314)
    code_book_6024 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 314, 20), 'code_book')
    # Obtaining the member '__getitem__' of a type (line 314)
    getitem___6025 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 314, 20), code_book_6024, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 314)
    subscript_call_result_6026 = invoke(stypy.reporting.localization.Localization(__file__, 314, 20), getitem___6025, has_members_6023)
    
    # Assigning a type to the variable 'code_book' (line 314)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 314, 8), 'code_book', subscript_call_result_6026)
    
    # Assigning a BinOp to a Name (line 315):
    
    # Assigning a BinOp to a Name (line 315):
    
    # Obtaining the type of the subscript
    int_6027 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 315, 30), 'int')
    # Getting the type of 'prev_avg_dists' (line 315)
    prev_avg_dists_6028 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 315, 15), 'prev_avg_dists')
    # Obtaining the member '__getitem__' of a type (line 315)
    getitem___6029 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 315, 15), prev_avg_dists_6028, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 315)
    subscript_call_result_6030 = invoke(stypy.reporting.localization.Localization(__file__, 315, 15), getitem___6029, int_6027)
    
    
    # Obtaining the type of the subscript
    int_6031 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 315, 50), 'int')
    # Getting the type of 'prev_avg_dists' (line 315)
    prev_avg_dists_6032 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 315, 35), 'prev_avg_dists')
    # Obtaining the member '__getitem__' of a type (line 315)
    getitem___6033 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 315, 35), prev_avg_dists_6032, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 315)
    subscript_call_result_6034 = invoke(stypy.reporting.localization.Localization(__file__, 315, 35), getitem___6033, int_6031)
    
    # Applying the binary operator '-' (line 315)
    result_sub_6035 = python_operator(stypy.reporting.localization.Localization(__file__, 315, 15), '-', subscript_call_result_6030, subscript_call_result_6034)
    
    # Assigning a type to the variable 'diff' (line 315)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 315, 8), 'diff', result_sub_6035)
    # SSA join for while statement (line 307)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Obtaining an instance of the builtin type 'tuple' (line 317)
    tuple_6036 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 317, 11), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 317)
    # Adding element type (line 317)
    # Getting the type of 'code_book' (line 317)
    code_book_6037 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 317, 11), 'code_book')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 317, 11), tuple_6036, code_book_6037)
    # Adding element type (line 317)
    
    # Obtaining the type of the subscript
    int_6038 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 317, 37), 'int')
    # Getting the type of 'prev_avg_dists' (line 317)
    prev_avg_dists_6039 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 317, 22), 'prev_avg_dists')
    # Obtaining the member '__getitem__' of a type (line 317)
    getitem___6040 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 317, 22), prev_avg_dists_6039, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 317)
    subscript_call_result_6041 = invoke(stypy.reporting.localization.Localization(__file__, 317, 22), getitem___6040, int_6038)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 317, 11), tuple_6036, subscript_call_result_6041)
    
    # Assigning a type to the variable 'stypy_return_type' (line 317)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 317, 4), 'stypy_return_type', tuple_6036)
    
    # ################# End of '_kmeans(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_kmeans' in the type store
    # Getting the type of 'stypy_return_type' (line 271)
    stypy_return_type_6042 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 271, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_6042)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_kmeans'
    return stypy_return_type_6042

# Assigning a type to the variable '_kmeans' (line 271)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 271, 0), '_kmeans', _kmeans)

@norecursion
def kmeans(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    int_6043 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 320, 33), 'int')
    float_6044 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 320, 44), 'float')
    # Getting the type of 'True' (line 320)
    True_6045 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 320, 63), 'True')
    defaults = [int_6043, float_6044, True_6045]
    # Create a new context for function 'kmeans'
    module_type_store = module_type_store.open_function_context('kmeans', 320, 0, False)
    
    # Passed parameters checking function
    kmeans.stypy_localization = localization
    kmeans.stypy_type_of_self = None
    kmeans.stypy_type_store = module_type_store
    kmeans.stypy_function_name = 'kmeans'
    kmeans.stypy_param_names_list = ['obs', 'k_or_guess', 'iter', 'thresh', 'check_finite']
    kmeans.stypy_varargs_param_name = None
    kmeans.stypy_kwargs_param_name = None
    kmeans.stypy_call_defaults = defaults
    kmeans.stypy_call_varargs = varargs
    kmeans.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'kmeans', ['obs', 'k_or_guess', 'iter', 'thresh', 'check_finite'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'kmeans', localization, ['obs', 'k_or_guess', 'iter', 'thresh', 'check_finite'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'kmeans(...)' code ##################

    str_6046 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 430, (-1)), 'str', "\n    Performs k-means on a set of observation vectors forming k clusters.\n\n    The k-means algorithm adjusts the centroids until sufficient\n    progress cannot be made, i.e. the change in distortion since\n    the last iteration is less than some threshold. This yields\n    a code book mapping centroids to codes and vice versa.\n\n    Distortion is defined as the sum of the squared differences\n    between the observations and the corresponding centroid.\n\n    Parameters\n    ----------\n    obs : ndarray\n       Each row of the M by N array is an observation vector. The\n       columns are the features seen during each observation.\n       The features must be whitened first with the `whiten` function.\n\n    k_or_guess : int or ndarray\n       The number of centroids to generate. A code is assigned to\n       each centroid, which is also the row index of the centroid\n       in the code_book matrix generated.\n\n       The initial k centroids are chosen by randomly selecting\n       observations from the observation matrix. Alternatively,\n       passing a k by N array specifies the initial k centroids.\n\n    iter : int, optional\n       The number of times to run k-means, returning the codebook\n       with the lowest distortion. This argument is ignored if\n       initial centroids are specified with an array for the\n       ``k_or_guess`` parameter. This parameter does not represent the\n       number of iterations of the k-means algorithm.\n\n    thresh : float, optional\n       Terminates the k-means algorithm if the change in\n       distortion since the last k-means iteration is less than\n       or equal to thresh.\n\n    check_finite : bool, optional\n        Whether to check that the input matrices contain only finite numbers.\n        Disabling may give a performance gain, but may result in problems\n        (crashes, non-termination) if the inputs do contain infinities or NaNs.\n        Default: True\n\n    Returns\n    -------\n    codebook : ndarray\n       A k by N array of k centroids. The i'th centroid\n       codebook[i] is represented with the code i. The centroids\n       and codes generated represent the lowest distortion seen,\n       not necessarily the globally minimal distortion.\n\n    distortion : float\n       The distortion between the observations passed and the\n       centroids generated.\n\n    See Also\n    --------\n    kmeans2 : a different implementation of k-means clustering\n       with more methods for generating initial centroids but without\n       using a distortion change threshold as a stopping criterion.\n\n    whiten : must be called prior to passing an observation matrix\n       to kmeans.\n\n    Examples\n    --------\n    >>> from numpy import array\n    >>> from scipy.cluster.vq import vq, kmeans, whiten\n    >>> import matplotlib.pyplot as plt\n    >>> features  = array([[ 1.9,2.3],\n    ...                    [ 1.5,2.5],\n    ...                    [ 0.8,0.6],\n    ...                    [ 0.4,1.8],\n    ...                    [ 0.1,0.1],\n    ...                    [ 0.2,1.8],\n    ...                    [ 2.0,0.5],\n    ...                    [ 0.3,1.5],\n    ...                    [ 1.0,1.0]])\n    >>> whitened = whiten(features)\n    >>> book = np.array((whitened[0],whitened[2]))\n    >>> kmeans(whitened,book)\n    (array([[ 2.3110306 ,  2.86287398],    # random\n           [ 0.93218041,  1.24398691]]), 0.85684700941625547)\n\n    >>> from numpy import random\n    >>> random.seed((1000,2000))\n    >>> codes = 3\n    >>> kmeans(whitened,codes)\n    (array([[ 2.3110306 ,  2.86287398],    # random\n           [ 1.32544402,  0.65607529],\n           [ 0.40782893,  2.02786907]]), 0.5196582527686241)\n\n    >>> # Create 50 datapoints in two clusters a and b\n    >>> pts = 50\n    >>> a = np.random.multivariate_normal([0, 0], [[4, 1], [1, 4]], size=pts)\n    >>> b = np.random.multivariate_normal([30, 10],\n    ...                                   [[10, 2], [2, 1]],\n    ...                                   size=pts)\n    >>> features = np.concatenate((a, b))\n    >>> # Whiten data\n    >>> whitened = whiten(features)\n    >>> # Find 2 clusters in the data\n    >>> codebook, distortion = kmeans(whitened, 2)\n    >>> # Plot whitened data and cluster centers in red\n    >>> plt.scatter(whitened[:, 0], whitened[:, 1])\n    >>> plt.scatter(codebook[:, 0], codebook[:, 1], c='r')\n    >>> plt.show()\n    ")
    
    # Assigning a Call to a Name (line 431):
    
    # Assigning a Call to a Name (line 431):
    
    # Call to _asarray_validated(...): (line 431)
    # Processing the call arguments (line 431)
    # Getting the type of 'obs' (line 431)
    obs_6048 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 431, 29), 'obs', False)
    # Processing the call keyword arguments (line 431)
    # Getting the type of 'check_finite' (line 431)
    check_finite_6049 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 431, 47), 'check_finite', False)
    keyword_6050 = check_finite_6049
    kwargs_6051 = {'check_finite': keyword_6050}
    # Getting the type of '_asarray_validated' (line 431)
    _asarray_validated_6047 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 431, 10), '_asarray_validated', False)
    # Calling _asarray_validated(args, kwargs) (line 431)
    _asarray_validated_call_result_6052 = invoke(stypy.reporting.localization.Localization(__file__, 431, 10), _asarray_validated_6047, *[obs_6048], **kwargs_6051)
    
    # Assigning a type to the variable 'obs' (line 431)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 431, 4), 'obs', _asarray_validated_call_result_6052)
    
    
    # Getting the type of 'iter' (line 432)
    iter_6053 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 432, 7), 'iter')
    int_6054 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 432, 14), 'int')
    # Applying the binary operator '<' (line 432)
    result_lt_6055 = python_operator(stypy.reporting.localization.Localization(__file__, 432, 7), '<', iter_6053, int_6054)
    
    # Testing the type of an if condition (line 432)
    if_condition_6056 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 432, 4), result_lt_6055)
    # Assigning a type to the variable 'if_condition_6056' (line 432)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 432, 4), 'if_condition_6056', if_condition_6056)
    # SSA begins for if statement (line 432)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 433)
    # Processing the call arguments (line 433)
    str_6058 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 433, 25), 'str', 'iter must be at least 1, got %s')
    # Getting the type of 'iter' (line 433)
    iter_6059 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 433, 61), 'iter', False)
    # Applying the binary operator '%' (line 433)
    result_mod_6060 = python_operator(stypy.reporting.localization.Localization(__file__, 433, 25), '%', str_6058, iter_6059)
    
    # Processing the call keyword arguments (line 433)
    kwargs_6061 = {}
    # Getting the type of 'ValueError' (line 433)
    ValueError_6057 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 433, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 433)
    ValueError_call_result_6062 = invoke(stypy.reporting.localization.Localization(__file__, 433, 14), ValueError_6057, *[result_mod_6060], **kwargs_6061)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 433, 8), ValueError_call_result_6062, 'raise parameter', BaseException)
    # SSA join for if statement (line 432)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    
    # Call to isscalar(...): (line 436)
    # Processing the call arguments (line 436)
    # Getting the type of 'k_or_guess' (line 436)
    k_or_guess_6065 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 436, 23), 'k_or_guess', False)
    # Processing the call keyword arguments (line 436)
    kwargs_6066 = {}
    # Getting the type of 'np' (line 436)
    np_6063 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 436, 11), 'np', False)
    # Obtaining the member 'isscalar' of a type (line 436)
    isscalar_6064 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 436, 11), np_6063, 'isscalar')
    # Calling isscalar(args, kwargs) (line 436)
    isscalar_call_result_6067 = invoke(stypy.reporting.localization.Localization(__file__, 436, 11), isscalar_6064, *[k_or_guess_6065], **kwargs_6066)
    
    # Applying the 'not' unary operator (line 436)
    result_not__6068 = python_operator(stypy.reporting.localization.Localization(__file__, 436, 7), 'not', isscalar_call_result_6067)
    
    # Testing the type of an if condition (line 436)
    if_condition_6069 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 436, 4), result_not__6068)
    # Assigning a type to the variable 'if_condition_6069' (line 436)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 436, 4), 'if_condition_6069', if_condition_6069)
    # SSA begins for if statement (line 436)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 437):
    
    # Assigning a Call to a Name (line 437):
    
    # Call to _asarray_validated(...): (line 437)
    # Processing the call arguments (line 437)
    # Getting the type of 'k_or_guess' (line 437)
    k_or_guess_6071 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 437, 35), 'k_or_guess', False)
    # Processing the call keyword arguments (line 437)
    # Getting the type of 'check_finite' (line 437)
    check_finite_6072 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 437, 60), 'check_finite', False)
    keyword_6073 = check_finite_6072
    kwargs_6074 = {'check_finite': keyword_6073}
    # Getting the type of '_asarray_validated' (line 437)
    _asarray_validated_6070 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 437, 16), '_asarray_validated', False)
    # Calling _asarray_validated(args, kwargs) (line 437)
    _asarray_validated_call_result_6075 = invoke(stypy.reporting.localization.Localization(__file__, 437, 16), _asarray_validated_6070, *[k_or_guess_6071], **kwargs_6074)
    
    # Assigning a type to the variable 'guess' (line 437)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 437, 8), 'guess', _asarray_validated_call_result_6075)
    
    
    # Getting the type of 'guess' (line 438)
    guess_6076 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 438, 11), 'guess')
    # Obtaining the member 'size' of a type (line 438)
    size_6077 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 438, 11), guess_6076, 'size')
    int_6078 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 438, 24), 'int')
    # Applying the binary operator '<' (line 438)
    result_lt_6079 = python_operator(stypy.reporting.localization.Localization(__file__, 438, 11), '<', size_6077, int_6078)
    
    # Testing the type of an if condition (line 438)
    if_condition_6080 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 438, 8), result_lt_6079)
    # Assigning a type to the variable 'if_condition_6080' (line 438)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 438, 8), 'if_condition_6080', if_condition_6080)
    # SSA begins for if statement (line 438)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 439)
    # Processing the call arguments (line 439)
    str_6082 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 439, 29), 'str', 'Asked for 0 clusters. Initial book was %s')
    # Getting the type of 'guess' (line 440)
    guess_6083 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 440, 29), 'guess', False)
    # Applying the binary operator '%' (line 439)
    result_mod_6084 = python_operator(stypy.reporting.localization.Localization(__file__, 439, 29), '%', str_6082, guess_6083)
    
    # Processing the call keyword arguments (line 439)
    kwargs_6085 = {}
    # Getting the type of 'ValueError' (line 439)
    ValueError_6081 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 439, 18), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 439)
    ValueError_call_result_6086 = invoke(stypy.reporting.localization.Localization(__file__, 439, 18), ValueError_6081, *[result_mod_6084], **kwargs_6085)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 439, 12), ValueError_call_result_6086, 'raise parameter', BaseException)
    # SSA join for if statement (line 438)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to _kmeans(...): (line 441)
    # Processing the call arguments (line 441)
    # Getting the type of 'obs' (line 441)
    obs_6088 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 441, 23), 'obs', False)
    # Getting the type of 'guess' (line 441)
    guess_6089 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 441, 28), 'guess', False)
    # Processing the call keyword arguments (line 441)
    # Getting the type of 'thresh' (line 441)
    thresh_6090 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 441, 42), 'thresh', False)
    keyword_6091 = thresh_6090
    kwargs_6092 = {'thresh': keyword_6091}
    # Getting the type of '_kmeans' (line 441)
    _kmeans_6087 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 441, 15), '_kmeans', False)
    # Calling _kmeans(args, kwargs) (line 441)
    _kmeans_call_result_6093 = invoke(stypy.reporting.localization.Localization(__file__, 441, 15), _kmeans_6087, *[obs_6088, guess_6089], **kwargs_6092)
    
    # Assigning a type to the variable 'stypy_return_type' (line 441)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 441, 8), 'stypy_return_type', _kmeans_call_result_6093)
    # SSA join for if statement (line 436)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 444):
    
    # Assigning a Call to a Name (line 444):
    
    # Call to int(...): (line 444)
    # Processing the call arguments (line 444)
    # Getting the type of 'k_or_guess' (line 444)
    k_or_guess_6095 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 444, 12), 'k_or_guess', False)
    # Processing the call keyword arguments (line 444)
    kwargs_6096 = {}
    # Getting the type of 'int' (line 444)
    int_6094 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 444, 8), 'int', False)
    # Calling int(args, kwargs) (line 444)
    int_call_result_6097 = invoke(stypy.reporting.localization.Localization(__file__, 444, 8), int_6094, *[k_or_guess_6095], **kwargs_6096)
    
    # Assigning a type to the variable 'k' (line 444)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 444, 4), 'k', int_call_result_6097)
    
    
    # Getting the type of 'k' (line 445)
    k_6098 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 445, 7), 'k')
    # Getting the type of 'k_or_guess' (line 445)
    k_or_guess_6099 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 445, 12), 'k_or_guess')
    # Applying the binary operator '!=' (line 445)
    result_ne_6100 = python_operator(stypy.reporting.localization.Localization(__file__, 445, 7), '!=', k_6098, k_or_guess_6099)
    
    # Testing the type of an if condition (line 445)
    if_condition_6101 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 445, 4), result_ne_6100)
    # Assigning a type to the variable 'if_condition_6101' (line 445)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 445, 4), 'if_condition_6101', if_condition_6101)
    # SSA begins for if statement (line 445)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 446)
    # Processing the call arguments (line 446)
    str_6103 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 446, 25), 'str', 'If k_or_guess is a scalar, it must be an integer.')
    # Processing the call keyword arguments (line 446)
    kwargs_6104 = {}
    # Getting the type of 'ValueError' (line 446)
    ValueError_6102 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 446, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 446)
    ValueError_call_result_6105 = invoke(stypy.reporting.localization.Localization(__file__, 446, 14), ValueError_6102, *[str_6103], **kwargs_6104)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 446, 8), ValueError_call_result_6105, 'raise parameter', BaseException)
    # SSA join for if statement (line 445)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'k' (line 447)
    k_6106 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 447, 7), 'k')
    int_6107 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 447, 11), 'int')
    # Applying the binary operator '<' (line 447)
    result_lt_6108 = python_operator(stypy.reporting.localization.Localization(__file__, 447, 7), '<', k_6106, int_6107)
    
    # Testing the type of an if condition (line 447)
    if_condition_6109 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 447, 4), result_lt_6108)
    # Assigning a type to the variable 'if_condition_6109' (line 447)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 447, 4), 'if_condition_6109', if_condition_6109)
    # SSA begins for if statement (line 447)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 448)
    # Processing the call arguments (line 448)
    str_6111 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 448, 25), 'str', 'Asked for %d clusters.')
    # Getting the type of 'k' (line 448)
    k_6112 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 448, 52), 'k', False)
    # Applying the binary operator '%' (line 448)
    result_mod_6113 = python_operator(stypy.reporting.localization.Localization(__file__, 448, 25), '%', str_6111, k_6112)
    
    # Processing the call keyword arguments (line 448)
    kwargs_6114 = {}
    # Getting the type of 'ValueError' (line 448)
    ValueError_6110 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 448, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 448)
    ValueError_call_result_6115 = invoke(stypy.reporting.localization.Localization(__file__, 448, 14), ValueError_6110, *[result_mod_6113], **kwargs_6114)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 448, 8), ValueError_call_result_6115, 'raise parameter', BaseException)
    # SSA join for if statement (line 447)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Attribute to a Name (line 451):
    
    # Assigning a Attribute to a Name (line 451):
    # Getting the type of 'np' (line 451)
    np_6116 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 451, 16), 'np')
    # Obtaining the member 'inf' of a type (line 451)
    inf_6117 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 451, 16), np_6116, 'inf')
    # Assigning a type to the variable 'best_dist' (line 451)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 451, 4), 'best_dist', inf_6117)
    
    
    # Call to xrange(...): (line 452)
    # Processing the call arguments (line 452)
    # Getting the type of 'iter' (line 452)
    iter_6119 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 452, 20), 'iter', False)
    # Processing the call keyword arguments (line 452)
    kwargs_6120 = {}
    # Getting the type of 'xrange' (line 452)
    xrange_6118 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 452, 13), 'xrange', False)
    # Calling xrange(args, kwargs) (line 452)
    xrange_call_result_6121 = invoke(stypy.reporting.localization.Localization(__file__, 452, 13), xrange_6118, *[iter_6119], **kwargs_6120)
    
    # Testing the type of a for loop iterable (line 452)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 452, 4), xrange_call_result_6121)
    # Getting the type of the for loop variable (line 452)
    for_loop_var_6122 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 452, 4), xrange_call_result_6121)
    # Assigning a type to the variable 'i' (line 452)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 452, 4), 'i', for_loop_var_6122)
    # SSA begins for a for statement (line 452)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Call to a Name (line 454):
    
    # Assigning a Call to a Name (line 454):
    
    # Call to _kpoints(...): (line 454)
    # Processing the call arguments (line 454)
    # Getting the type of 'obs' (line 454)
    obs_6124 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 454, 25), 'obs', False)
    # Getting the type of 'k' (line 454)
    k_6125 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 454, 30), 'k', False)
    # Processing the call keyword arguments (line 454)
    kwargs_6126 = {}
    # Getting the type of '_kpoints' (line 454)
    _kpoints_6123 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 454, 16), '_kpoints', False)
    # Calling _kpoints(args, kwargs) (line 454)
    _kpoints_call_result_6127 = invoke(stypy.reporting.localization.Localization(__file__, 454, 16), _kpoints_6123, *[obs_6124, k_6125], **kwargs_6126)
    
    # Assigning a type to the variable 'guess' (line 454)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 454, 8), 'guess', _kpoints_call_result_6127)
    
    # Assigning a Call to a Tuple (line 455):
    
    # Assigning a Subscript to a Name (line 455):
    
    # Obtaining the type of the subscript
    int_6128 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 455, 8), 'int')
    
    # Call to _kmeans(...): (line 455)
    # Processing the call arguments (line 455)
    # Getting the type of 'obs' (line 455)
    obs_6130 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 455, 29), 'obs', False)
    # Getting the type of 'guess' (line 455)
    guess_6131 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 455, 34), 'guess', False)
    # Processing the call keyword arguments (line 455)
    # Getting the type of 'thresh' (line 455)
    thresh_6132 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 455, 48), 'thresh', False)
    keyword_6133 = thresh_6132
    kwargs_6134 = {'thresh': keyword_6133}
    # Getting the type of '_kmeans' (line 455)
    _kmeans_6129 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 455, 21), '_kmeans', False)
    # Calling _kmeans(args, kwargs) (line 455)
    _kmeans_call_result_6135 = invoke(stypy.reporting.localization.Localization(__file__, 455, 21), _kmeans_6129, *[obs_6130, guess_6131], **kwargs_6134)
    
    # Obtaining the member '__getitem__' of a type (line 455)
    getitem___6136 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 455, 8), _kmeans_call_result_6135, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 455)
    subscript_call_result_6137 = invoke(stypy.reporting.localization.Localization(__file__, 455, 8), getitem___6136, int_6128)
    
    # Assigning a type to the variable 'tuple_var_assignment_5740' (line 455)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 455, 8), 'tuple_var_assignment_5740', subscript_call_result_6137)
    
    # Assigning a Subscript to a Name (line 455):
    
    # Obtaining the type of the subscript
    int_6138 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 455, 8), 'int')
    
    # Call to _kmeans(...): (line 455)
    # Processing the call arguments (line 455)
    # Getting the type of 'obs' (line 455)
    obs_6140 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 455, 29), 'obs', False)
    # Getting the type of 'guess' (line 455)
    guess_6141 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 455, 34), 'guess', False)
    # Processing the call keyword arguments (line 455)
    # Getting the type of 'thresh' (line 455)
    thresh_6142 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 455, 48), 'thresh', False)
    keyword_6143 = thresh_6142
    kwargs_6144 = {'thresh': keyword_6143}
    # Getting the type of '_kmeans' (line 455)
    _kmeans_6139 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 455, 21), '_kmeans', False)
    # Calling _kmeans(args, kwargs) (line 455)
    _kmeans_call_result_6145 = invoke(stypy.reporting.localization.Localization(__file__, 455, 21), _kmeans_6139, *[obs_6140, guess_6141], **kwargs_6144)
    
    # Obtaining the member '__getitem__' of a type (line 455)
    getitem___6146 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 455, 8), _kmeans_call_result_6145, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 455)
    subscript_call_result_6147 = invoke(stypy.reporting.localization.Localization(__file__, 455, 8), getitem___6146, int_6138)
    
    # Assigning a type to the variable 'tuple_var_assignment_5741' (line 455)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 455, 8), 'tuple_var_assignment_5741', subscript_call_result_6147)
    
    # Assigning a Name to a Name (line 455):
    # Getting the type of 'tuple_var_assignment_5740' (line 455)
    tuple_var_assignment_5740_6148 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 455, 8), 'tuple_var_assignment_5740')
    # Assigning a type to the variable 'book' (line 455)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 455, 8), 'book', tuple_var_assignment_5740_6148)
    
    # Assigning a Name to a Name (line 455):
    # Getting the type of 'tuple_var_assignment_5741' (line 455)
    tuple_var_assignment_5741_6149 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 455, 8), 'tuple_var_assignment_5741')
    # Assigning a type to the variable 'dist' (line 455)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 455, 14), 'dist', tuple_var_assignment_5741_6149)
    
    
    # Getting the type of 'dist' (line 456)
    dist_6150 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 456, 11), 'dist')
    # Getting the type of 'best_dist' (line 456)
    best_dist_6151 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 456, 18), 'best_dist')
    # Applying the binary operator '<' (line 456)
    result_lt_6152 = python_operator(stypy.reporting.localization.Localization(__file__, 456, 11), '<', dist_6150, best_dist_6151)
    
    # Testing the type of an if condition (line 456)
    if_condition_6153 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 456, 8), result_lt_6152)
    # Assigning a type to the variable 'if_condition_6153' (line 456)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 456, 8), 'if_condition_6153', if_condition_6153)
    # SSA begins for if statement (line 456)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Name to a Name (line 457):
    
    # Assigning a Name to a Name (line 457):
    # Getting the type of 'book' (line 457)
    book_6154 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 457, 24), 'book')
    # Assigning a type to the variable 'best_book' (line 457)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 457, 12), 'best_book', book_6154)
    
    # Assigning a Name to a Name (line 458):
    
    # Assigning a Name to a Name (line 458):
    # Getting the type of 'dist' (line 458)
    dist_6155 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 458, 24), 'dist')
    # Assigning a type to the variable 'best_dist' (line 458)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 458, 12), 'best_dist', dist_6155)
    # SSA join for if statement (line 456)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Obtaining an instance of the builtin type 'tuple' (line 459)
    tuple_6156 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 459, 11), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 459)
    # Adding element type (line 459)
    # Getting the type of 'best_book' (line 459)
    best_book_6157 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 459, 11), 'best_book')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 459, 11), tuple_6156, best_book_6157)
    # Adding element type (line 459)
    # Getting the type of 'best_dist' (line 459)
    best_dist_6158 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 459, 22), 'best_dist')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 459, 11), tuple_6156, best_dist_6158)
    
    # Assigning a type to the variable 'stypy_return_type' (line 459)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 459, 4), 'stypy_return_type', tuple_6156)
    
    # ################# End of 'kmeans(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'kmeans' in the type store
    # Getting the type of 'stypy_return_type' (line 320)
    stypy_return_type_6159 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 320, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_6159)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'kmeans'
    return stypy_return_type_6159

# Assigning a type to the variable 'kmeans' (line 320)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 320, 0), 'kmeans', kmeans)

@norecursion
def _kpoints(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_kpoints'
    module_type_store = module_type_store.open_function_context('_kpoints', 462, 0, False)
    
    # Passed parameters checking function
    _kpoints.stypy_localization = localization
    _kpoints.stypy_type_of_self = None
    _kpoints.stypy_type_store = module_type_store
    _kpoints.stypy_function_name = '_kpoints'
    _kpoints.stypy_param_names_list = ['data', 'k']
    _kpoints.stypy_varargs_param_name = None
    _kpoints.stypy_kwargs_param_name = None
    _kpoints.stypy_call_defaults = defaults
    _kpoints.stypy_call_varargs = varargs
    _kpoints.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_kpoints', ['data', 'k'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_kpoints', localization, ['data', 'k'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_kpoints(...)' code ##################

    str_6160 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 474, (-1)), 'str', 'Pick k points at random in data (one row = one observation).\n\n    Parameters\n    ----------\n    data : ndarray\n        Expect a rank 1 or 2 array. Rank 1 are assumed to describe one\n        dimensional data, rank 2 multidimensional data, in which case one\n        row is one observation.\n    k : int\n        Number of samples to generate.\n\n    ')
    
    # Assigning a Call to a Name (line 475):
    
    # Assigning a Call to a Name (line 475):
    
    # Call to choice(...): (line 475)
    # Processing the call arguments (line 475)
    
    # Obtaining the type of the subscript
    int_6164 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 475, 38), 'int')
    # Getting the type of 'data' (line 475)
    data_6165 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 475, 27), 'data', False)
    # Obtaining the member 'shape' of a type (line 475)
    shape_6166 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 475, 27), data_6165, 'shape')
    # Obtaining the member '__getitem__' of a type (line 475)
    getitem___6167 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 475, 27), shape_6166, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 475)
    subscript_call_result_6168 = invoke(stypy.reporting.localization.Localization(__file__, 475, 27), getitem___6167, int_6164)
    
    # Processing the call keyword arguments (line 475)
    # Getting the type of 'k' (line 475)
    k_6169 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 475, 47), 'k', False)
    keyword_6170 = k_6169
    # Getting the type of 'False' (line 475)
    False_6171 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 475, 58), 'False', False)
    keyword_6172 = False_6171
    kwargs_6173 = {'replace': keyword_6172, 'size': keyword_6170}
    # Getting the type of 'np' (line 475)
    np_6161 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 475, 10), 'np', False)
    # Obtaining the member 'random' of a type (line 475)
    random_6162 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 475, 10), np_6161, 'random')
    # Obtaining the member 'choice' of a type (line 475)
    choice_6163 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 475, 10), random_6162, 'choice')
    # Calling choice(args, kwargs) (line 475)
    choice_call_result_6174 = invoke(stypy.reporting.localization.Localization(__file__, 475, 10), choice_6163, *[subscript_call_result_6168], **kwargs_6173)
    
    # Assigning a type to the variable 'idx' (line 475)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 475, 4), 'idx', choice_call_result_6174)
    
    # Obtaining the type of the subscript
    # Getting the type of 'idx' (line 476)
    idx_6175 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 476, 16), 'idx')
    # Getting the type of 'data' (line 476)
    data_6176 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 476, 11), 'data')
    # Obtaining the member '__getitem__' of a type (line 476)
    getitem___6177 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 476, 11), data_6176, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 476)
    subscript_call_result_6178 = invoke(stypy.reporting.localization.Localization(__file__, 476, 11), getitem___6177, idx_6175)
    
    # Assigning a type to the variable 'stypy_return_type' (line 476)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 476, 4), 'stypy_return_type', subscript_call_result_6178)
    
    # ################# End of '_kpoints(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_kpoints' in the type store
    # Getting the type of 'stypy_return_type' (line 462)
    stypy_return_type_6179 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 462, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_6179)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_kpoints'
    return stypy_return_type_6179

# Assigning a type to the variable '_kpoints' (line 462)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 462, 0), '_kpoints', _kpoints)

@norecursion
def _krandinit(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_krandinit'
    module_type_store = module_type_store.open_function_context('_krandinit', 479, 0, False)
    
    # Passed parameters checking function
    _krandinit.stypy_localization = localization
    _krandinit.stypy_type_of_self = None
    _krandinit.stypy_type_store = module_type_store
    _krandinit.stypy_function_name = '_krandinit'
    _krandinit.stypy_param_names_list = ['data', 'k']
    _krandinit.stypy_varargs_param_name = None
    _krandinit.stypy_kwargs_param_name = None
    _krandinit.stypy_call_defaults = defaults
    _krandinit.stypy_call_varargs = varargs
    _krandinit.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_krandinit', ['data', 'k'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_krandinit', localization, ['data', 'k'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_krandinit(...)' code ##################

    str_6180 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 494, (-1)), 'str', 'Returns k samples of a random variable which parameters depend on data.\n\n    More precisely, it returns k observations sampled from a Gaussian random\n    variable which mean and covariances are the one estimated from data.\n\n    Parameters\n    ----------\n    data : ndarray\n        Expect a rank 1 or 2 array. Rank 1 are assumed to describe one\n        dimensional data, rank 2 multidimensional data, in which case one\n        row is one observation.\n    k : int\n        Number of samples to generate.\n\n    ')
    
    # Assigning a Call to a Name (line 495):
    
    # Assigning a Call to a Name (line 495):
    
    # Call to mean(...): (line 495)
    # Processing the call keyword arguments (line 495)
    int_6183 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 495, 24), 'int')
    keyword_6184 = int_6183
    kwargs_6185 = {'axis': keyword_6184}
    # Getting the type of 'data' (line 495)
    data_6181 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 495, 9), 'data', False)
    # Obtaining the member 'mean' of a type (line 495)
    mean_6182 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 495, 9), data_6181, 'mean')
    # Calling mean(args, kwargs) (line 495)
    mean_call_result_6186 = invoke(stypy.reporting.localization.Localization(__file__, 495, 9), mean_6182, *[], **kwargs_6185)
    
    # Assigning a type to the variable 'mu' (line 495)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 495, 4), 'mu', mean_call_result_6186)
    
    
    # Getting the type of 'data' (line 497)
    data_6187 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 497, 7), 'data')
    # Obtaining the member 'ndim' of a type (line 497)
    ndim_6188 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 497, 7), data_6187, 'ndim')
    int_6189 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 497, 20), 'int')
    # Applying the binary operator '==' (line 497)
    result_eq_6190 = python_operator(stypy.reporting.localization.Localization(__file__, 497, 7), '==', ndim_6188, int_6189)
    
    # Testing the type of an if condition (line 497)
    if_condition_6191 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 497, 4), result_eq_6190)
    # Assigning a type to the variable 'if_condition_6191' (line 497)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 497, 4), 'if_condition_6191', if_condition_6191)
    # SSA begins for if statement (line 497)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 498):
    
    # Assigning a Call to a Name (line 498):
    
    # Call to cov(...): (line 498)
    # Processing the call arguments (line 498)
    # Getting the type of 'data' (line 498)
    data_6194 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 498, 21), 'data', False)
    # Processing the call keyword arguments (line 498)
    kwargs_6195 = {}
    # Getting the type of 'np' (line 498)
    np_6192 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 498, 14), 'np', False)
    # Obtaining the member 'cov' of a type (line 498)
    cov_6193 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 498, 14), np_6192, 'cov')
    # Calling cov(args, kwargs) (line 498)
    cov_call_result_6196 = invoke(stypy.reporting.localization.Localization(__file__, 498, 14), cov_6193, *[data_6194], **kwargs_6195)
    
    # Assigning a type to the variable 'cov' (line 498)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 498, 8), 'cov', cov_call_result_6196)
    
    # Assigning a Call to a Name (line 499):
    
    # Assigning a Call to a Name (line 499):
    
    # Call to randn(...): (line 499)
    # Processing the call arguments (line 499)
    # Getting the type of 'k' (line 499)
    k_6200 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 499, 28), 'k', False)
    # Processing the call keyword arguments (line 499)
    kwargs_6201 = {}
    # Getting the type of 'np' (line 499)
    np_6197 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 499, 12), 'np', False)
    # Obtaining the member 'random' of a type (line 499)
    random_6198 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 499, 12), np_6197, 'random')
    # Obtaining the member 'randn' of a type (line 499)
    randn_6199 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 499, 12), random_6198, 'randn')
    # Calling randn(args, kwargs) (line 499)
    randn_call_result_6202 = invoke(stypy.reporting.localization.Localization(__file__, 499, 12), randn_6199, *[k_6200], **kwargs_6201)
    
    # Assigning a type to the variable 'x' (line 499)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 499, 8), 'x', randn_call_result_6202)
    
    # Getting the type of 'x' (line 500)
    x_6203 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 500, 8), 'x')
    
    # Call to sqrt(...): (line 500)
    # Processing the call arguments (line 500)
    # Getting the type of 'cov' (line 500)
    cov_6206 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 500, 21), 'cov', False)
    # Processing the call keyword arguments (line 500)
    kwargs_6207 = {}
    # Getting the type of 'np' (line 500)
    np_6204 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 500, 13), 'np', False)
    # Obtaining the member 'sqrt' of a type (line 500)
    sqrt_6205 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 500, 13), np_6204, 'sqrt')
    # Calling sqrt(args, kwargs) (line 500)
    sqrt_call_result_6208 = invoke(stypy.reporting.localization.Localization(__file__, 500, 13), sqrt_6205, *[cov_6206], **kwargs_6207)
    
    # Applying the binary operator '*=' (line 500)
    result_imul_6209 = python_operator(stypy.reporting.localization.Localization(__file__, 500, 8), '*=', x_6203, sqrt_call_result_6208)
    # Assigning a type to the variable 'x' (line 500)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 500, 8), 'x', result_imul_6209)
    
    # SSA branch for the else part of an if statement (line 497)
    module_type_store.open_ssa_branch('else')
    
    
    
    # Obtaining the type of the subscript
    int_6210 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 501, 20), 'int')
    # Getting the type of 'data' (line 501)
    data_6211 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 501, 9), 'data')
    # Obtaining the member 'shape' of a type (line 501)
    shape_6212 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 501, 9), data_6211, 'shape')
    # Obtaining the member '__getitem__' of a type (line 501)
    getitem___6213 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 501, 9), shape_6212, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 501)
    subscript_call_result_6214 = invoke(stypy.reporting.localization.Localization(__file__, 501, 9), getitem___6213, int_6210)
    
    
    # Obtaining the type of the subscript
    int_6215 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 501, 36), 'int')
    # Getting the type of 'data' (line 501)
    data_6216 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 501, 25), 'data')
    # Obtaining the member 'shape' of a type (line 501)
    shape_6217 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 501, 25), data_6216, 'shape')
    # Obtaining the member '__getitem__' of a type (line 501)
    getitem___6218 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 501, 25), shape_6217, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 501)
    subscript_call_result_6219 = invoke(stypy.reporting.localization.Localization(__file__, 501, 25), getitem___6218, int_6215)
    
    # Applying the binary operator '>' (line 501)
    result_gt_6220 = python_operator(stypy.reporting.localization.Localization(__file__, 501, 9), '>', subscript_call_result_6214, subscript_call_result_6219)
    
    # Testing the type of an if condition (line 501)
    if_condition_6221 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 501, 9), result_gt_6220)
    # Assigning a type to the variable 'if_condition_6221' (line 501)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 501, 9), 'if_condition_6221', if_condition_6221)
    # SSA begins for if statement (line 501)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Tuple (line 503):
    
    # Assigning a Subscript to a Name (line 503):
    
    # Obtaining the type of the subscript
    int_6222 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 503, 8), 'int')
    
    # Call to svd(...): (line 503)
    # Processing the call arguments (line 503)
    # Getting the type of 'data' (line 503)
    data_6226 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 503, 33), 'data', False)
    # Getting the type of 'mu' (line 503)
    mu_6227 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 503, 40), 'mu', False)
    # Applying the binary operator '-' (line 503)
    result_sub_6228 = python_operator(stypy.reporting.localization.Localization(__file__, 503, 33), '-', data_6226, mu_6227)
    
    # Processing the call keyword arguments (line 503)
    # Getting the type of 'False' (line 503)
    False_6229 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 503, 58), 'False', False)
    keyword_6230 = False_6229
    kwargs_6231 = {'full_matrices': keyword_6230}
    # Getting the type of 'np' (line 503)
    np_6223 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 503, 19), 'np', False)
    # Obtaining the member 'linalg' of a type (line 503)
    linalg_6224 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 503, 19), np_6223, 'linalg')
    # Obtaining the member 'svd' of a type (line 503)
    svd_6225 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 503, 19), linalg_6224, 'svd')
    # Calling svd(args, kwargs) (line 503)
    svd_call_result_6232 = invoke(stypy.reporting.localization.Localization(__file__, 503, 19), svd_6225, *[result_sub_6228], **kwargs_6231)
    
    # Obtaining the member '__getitem__' of a type (line 503)
    getitem___6233 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 503, 8), svd_call_result_6232, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 503)
    subscript_call_result_6234 = invoke(stypy.reporting.localization.Localization(__file__, 503, 8), getitem___6233, int_6222)
    
    # Assigning a type to the variable 'tuple_var_assignment_5742' (line 503)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 503, 8), 'tuple_var_assignment_5742', subscript_call_result_6234)
    
    # Assigning a Subscript to a Name (line 503):
    
    # Obtaining the type of the subscript
    int_6235 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 503, 8), 'int')
    
    # Call to svd(...): (line 503)
    # Processing the call arguments (line 503)
    # Getting the type of 'data' (line 503)
    data_6239 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 503, 33), 'data', False)
    # Getting the type of 'mu' (line 503)
    mu_6240 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 503, 40), 'mu', False)
    # Applying the binary operator '-' (line 503)
    result_sub_6241 = python_operator(stypy.reporting.localization.Localization(__file__, 503, 33), '-', data_6239, mu_6240)
    
    # Processing the call keyword arguments (line 503)
    # Getting the type of 'False' (line 503)
    False_6242 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 503, 58), 'False', False)
    keyword_6243 = False_6242
    kwargs_6244 = {'full_matrices': keyword_6243}
    # Getting the type of 'np' (line 503)
    np_6236 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 503, 19), 'np', False)
    # Obtaining the member 'linalg' of a type (line 503)
    linalg_6237 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 503, 19), np_6236, 'linalg')
    # Obtaining the member 'svd' of a type (line 503)
    svd_6238 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 503, 19), linalg_6237, 'svd')
    # Calling svd(args, kwargs) (line 503)
    svd_call_result_6245 = invoke(stypy.reporting.localization.Localization(__file__, 503, 19), svd_6238, *[result_sub_6241], **kwargs_6244)
    
    # Obtaining the member '__getitem__' of a type (line 503)
    getitem___6246 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 503, 8), svd_call_result_6245, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 503)
    subscript_call_result_6247 = invoke(stypy.reporting.localization.Localization(__file__, 503, 8), getitem___6246, int_6235)
    
    # Assigning a type to the variable 'tuple_var_assignment_5743' (line 503)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 503, 8), 'tuple_var_assignment_5743', subscript_call_result_6247)
    
    # Assigning a Subscript to a Name (line 503):
    
    # Obtaining the type of the subscript
    int_6248 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 503, 8), 'int')
    
    # Call to svd(...): (line 503)
    # Processing the call arguments (line 503)
    # Getting the type of 'data' (line 503)
    data_6252 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 503, 33), 'data', False)
    # Getting the type of 'mu' (line 503)
    mu_6253 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 503, 40), 'mu', False)
    # Applying the binary operator '-' (line 503)
    result_sub_6254 = python_operator(stypy.reporting.localization.Localization(__file__, 503, 33), '-', data_6252, mu_6253)
    
    # Processing the call keyword arguments (line 503)
    # Getting the type of 'False' (line 503)
    False_6255 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 503, 58), 'False', False)
    keyword_6256 = False_6255
    kwargs_6257 = {'full_matrices': keyword_6256}
    # Getting the type of 'np' (line 503)
    np_6249 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 503, 19), 'np', False)
    # Obtaining the member 'linalg' of a type (line 503)
    linalg_6250 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 503, 19), np_6249, 'linalg')
    # Obtaining the member 'svd' of a type (line 503)
    svd_6251 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 503, 19), linalg_6250, 'svd')
    # Calling svd(args, kwargs) (line 503)
    svd_call_result_6258 = invoke(stypy.reporting.localization.Localization(__file__, 503, 19), svd_6251, *[result_sub_6254], **kwargs_6257)
    
    # Obtaining the member '__getitem__' of a type (line 503)
    getitem___6259 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 503, 8), svd_call_result_6258, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 503)
    subscript_call_result_6260 = invoke(stypy.reporting.localization.Localization(__file__, 503, 8), getitem___6259, int_6248)
    
    # Assigning a type to the variable 'tuple_var_assignment_5744' (line 503)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 503, 8), 'tuple_var_assignment_5744', subscript_call_result_6260)
    
    # Assigning a Name to a Name (line 503):
    # Getting the type of 'tuple_var_assignment_5742' (line 503)
    tuple_var_assignment_5742_6261 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 503, 8), 'tuple_var_assignment_5742')
    # Assigning a type to the variable '_' (line 503)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 503, 8), '_', tuple_var_assignment_5742_6261)
    
    # Assigning a Name to a Name (line 503):
    # Getting the type of 'tuple_var_assignment_5743' (line 503)
    tuple_var_assignment_5743_6262 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 503, 8), 'tuple_var_assignment_5743')
    # Assigning a type to the variable 's' (line 503)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 503, 11), 's', tuple_var_assignment_5743_6262)
    
    # Assigning a Name to a Name (line 503):
    # Getting the type of 'tuple_var_assignment_5744' (line 503)
    tuple_var_assignment_5744_6263 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 503, 8), 'tuple_var_assignment_5744')
    # Assigning a type to the variable 'vh' (line 503)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 503, 14), 'vh', tuple_var_assignment_5744_6263)
    
    # Assigning a Call to a Name (line 504):
    
    # Assigning a Call to a Name (line 504):
    
    # Call to randn(...): (line 504)
    # Processing the call arguments (line 504)
    # Getting the type of 'k' (line 504)
    k_6267 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 504, 28), 'k', False)
    # Getting the type of 's' (line 504)
    s_6268 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 504, 31), 's', False)
    # Obtaining the member 'size' of a type (line 504)
    size_6269 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 504, 31), s_6268, 'size')
    # Processing the call keyword arguments (line 504)
    kwargs_6270 = {}
    # Getting the type of 'np' (line 504)
    np_6264 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 504, 12), 'np', False)
    # Obtaining the member 'random' of a type (line 504)
    random_6265 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 504, 12), np_6264, 'random')
    # Obtaining the member 'randn' of a type (line 504)
    randn_6266 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 504, 12), random_6265, 'randn')
    # Calling randn(args, kwargs) (line 504)
    randn_call_result_6271 = invoke(stypy.reporting.localization.Localization(__file__, 504, 12), randn_6266, *[k_6267, size_6269], **kwargs_6270)
    
    # Assigning a type to the variable 'x' (line 504)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 504, 8), 'x', randn_call_result_6271)
    
    # Assigning a BinOp to a Name (line 505):
    
    # Assigning a BinOp to a Name (line 505):
    
    # Obtaining the type of the subscript
    slice_6272 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 505, 14), None, None, None)
    # Getting the type of 'None' (line 505)
    None_6273 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 505, 19), 'None')
    # Getting the type of 's' (line 505)
    s_6274 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 505, 14), 's')
    # Obtaining the member '__getitem__' of a type (line 505)
    getitem___6275 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 505, 14), s_6274, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 505)
    subscript_call_result_6276 = invoke(stypy.reporting.localization.Localization(__file__, 505, 14), getitem___6275, (slice_6272, None_6273))
    
    # Getting the type of 'vh' (line 505)
    vh_6277 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 505, 27), 'vh')
    # Applying the binary operator '*' (line 505)
    result_mul_6278 = python_operator(stypy.reporting.localization.Localization(__file__, 505, 14), '*', subscript_call_result_6276, vh_6277)
    
    
    # Call to sqrt(...): (line 505)
    # Processing the call arguments (line 505)
    
    # Obtaining the type of the subscript
    int_6281 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 505, 51), 'int')
    # Getting the type of 'data' (line 505)
    data_6282 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 505, 40), 'data', False)
    # Obtaining the member 'shape' of a type (line 505)
    shape_6283 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 505, 40), data_6282, 'shape')
    # Obtaining the member '__getitem__' of a type (line 505)
    getitem___6284 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 505, 40), shape_6283, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 505)
    subscript_call_result_6285 = invoke(stypy.reporting.localization.Localization(__file__, 505, 40), getitem___6284, int_6281)
    
    int_6286 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 505, 56), 'int')
    # Applying the binary operator '-' (line 505)
    result_sub_6287 = python_operator(stypy.reporting.localization.Localization(__file__, 505, 40), '-', subscript_call_result_6285, int_6286)
    
    # Processing the call keyword arguments (line 505)
    kwargs_6288 = {}
    # Getting the type of 'np' (line 505)
    np_6279 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 505, 32), 'np', False)
    # Obtaining the member 'sqrt' of a type (line 505)
    sqrt_6280 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 505, 32), np_6279, 'sqrt')
    # Calling sqrt(args, kwargs) (line 505)
    sqrt_call_result_6289 = invoke(stypy.reporting.localization.Localization(__file__, 505, 32), sqrt_6280, *[result_sub_6287], **kwargs_6288)
    
    # Applying the binary operator 'div' (line 505)
    result_div_6290 = python_operator(stypy.reporting.localization.Localization(__file__, 505, 30), 'div', result_mul_6278, sqrt_call_result_6289)
    
    # Assigning a type to the variable 'sVh' (line 505)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 505, 8), 'sVh', result_div_6290)
    
    # Assigning a Call to a Name (line 506):
    
    # Assigning a Call to a Name (line 506):
    
    # Call to dot(...): (line 506)
    # Processing the call arguments (line 506)
    # Getting the type of 'sVh' (line 506)
    sVh_6293 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 506, 18), 'sVh', False)
    # Processing the call keyword arguments (line 506)
    kwargs_6294 = {}
    # Getting the type of 'x' (line 506)
    x_6291 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 506, 12), 'x', False)
    # Obtaining the member 'dot' of a type (line 506)
    dot_6292 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 506, 12), x_6291, 'dot')
    # Calling dot(args, kwargs) (line 506)
    dot_call_result_6295 = invoke(stypy.reporting.localization.Localization(__file__, 506, 12), dot_6292, *[sVh_6293], **kwargs_6294)
    
    # Assigning a type to the variable 'x' (line 506)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 506, 8), 'x', dot_call_result_6295)
    # SSA branch for the else part of an if statement (line 501)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Call to a Name (line 508):
    
    # Assigning a Call to a Name (line 508):
    
    # Call to atleast_2d(...): (line 508)
    # Processing the call arguments (line 508)
    
    # Call to cov(...): (line 508)
    # Processing the call arguments (line 508)
    # Getting the type of 'data' (line 508)
    data_6300 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 508, 35), 'data', False)
    # Processing the call keyword arguments (line 508)
    # Getting the type of 'False' (line 508)
    False_6301 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 508, 48), 'False', False)
    keyword_6302 = False_6301
    kwargs_6303 = {'rowvar': keyword_6302}
    # Getting the type of 'np' (line 508)
    np_6298 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 508, 28), 'np', False)
    # Obtaining the member 'cov' of a type (line 508)
    cov_6299 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 508, 28), np_6298, 'cov')
    # Calling cov(args, kwargs) (line 508)
    cov_call_result_6304 = invoke(stypy.reporting.localization.Localization(__file__, 508, 28), cov_6299, *[data_6300], **kwargs_6303)
    
    # Processing the call keyword arguments (line 508)
    kwargs_6305 = {}
    # Getting the type of 'np' (line 508)
    np_6296 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 508, 14), 'np', False)
    # Obtaining the member 'atleast_2d' of a type (line 508)
    atleast_2d_6297 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 508, 14), np_6296, 'atleast_2d')
    # Calling atleast_2d(args, kwargs) (line 508)
    atleast_2d_call_result_6306 = invoke(stypy.reporting.localization.Localization(__file__, 508, 14), atleast_2d_6297, *[cov_call_result_6304], **kwargs_6305)
    
    # Assigning a type to the variable 'cov' (line 508)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 508, 8), 'cov', atleast_2d_call_result_6306)
    
    # Assigning a Call to a Name (line 512):
    
    # Assigning a Call to a Name (line 512):
    
    # Call to randn(...): (line 512)
    # Processing the call arguments (line 512)
    # Getting the type of 'k' (line 512)
    k_6310 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 512, 28), 'k', False)
    # Getting the type of 'mu' (line 512)
    mu_6311 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 512, 31), 'mu', False)
    # Obtaining the member 'size' of a type (line 512)
    size_6312 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 512, 31), mu_6311, 'size')
    # Processing the call keyword arguments (line 512)
    kwargs_6313 = {}
    # Getting the type of 'np' (line 512)
    np_6307 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 512, 12), 'np', False)
    # Obtaining the member 'random' of a type (line 512)
    random_6308 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 512, 12), np_6307, 'random')
    # Obtaining the member 'randn' of a type (line 512)
    randn_6309 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 512, 12), random_6308, 'randn')
    # Calling randn(args, kwargs) (line 512)
    randn_call_result_6314 = invoke(stypy.reporting.localization.Localization(__file__, 512, 12), randn_6309, *[k_6310, size_6312], **kwargs_6313)
    
    # Assigning a type to the variable 'x' (line 512)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 512, 8), 'x', randn_call_result_6314)
    
    # Assigning a Call to a Name (line 513):
    
    # Assigning a Call to a Name (line 513):
    
    # Call to dot(...): (line 513)
    # Processing the call arguments (line 513)
    
    # Call to cholesky(...): (line 513)
    # Processing the call arguments (line 513)
    # Getting the type of 'cov' (line 513)
    cov_6320 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 513, 37), 'cov', False)
    # Processing the call keyword arguments (line 513)
    kwargs_6321 = {}
    # Getting the type of 'np' (line 513)
    np_6317 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 513, 18), 'np', False)
    # Obtaining the member 'linalg' of a type (line 513)
    linalg_6318 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 513, 18), np_6317, 'linalg')
    # Obtaining the member 'cholesky' of a type (line 513)
    cholesky_6319 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 513, 18), linalg_6318, 'cholesky')
    # Calling cholesky(args, kwargs) (line 513)
    cholesky_call_result_6322 = invoke(stypy.reporting.localization.Localization(__file__, 513, 18), cholesky_6319, *[cov_6320], **kwargs_6321)
    
    # Obtaining the member 'T' of a type (line 513)
    T_6323 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 513, 18), cholesky_call_result_6322, 'T')
    # Processing the call keyword arguments (line 513)
    kwargs_6324 = {}
    # Getting the type of 'x' (line 513)
    x_6315 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 513, 12), 'x', False)
    # Obtaining the member 'dot' of a type (line 513)
    dot_6316 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 513, 12), x_6315, 'dot')
    # Calling dot(args, kwargs) (line 513)
    dot_call_result_6325 = invoke(stypy.reporting.localization.Localization(__file__, 513, 12), dot_6316, *[T_6323], **kwargs_6324)
    
    # Assigning a type to the variable 'x' (line 513)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 513, 8), 'x', dot_call_result_6325)
    # SSA join for if statement (line 501)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 497)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Getting the type of 'x' (line 515)
    x_6326 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 515, 4), 'x')
    # Getting the type of 'mu' (line 515)
    mu_6327 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 515, 9), 'mu')
    # Applying the binary operator '+=' (line 515)
    result_iadd_6328 = python_operator(stypy.reporting.localization.Localization(__file__, 515, 4), '+=', x_6326, mu_6327)
    # Assigning a type to the variable 'x' (line 515)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 515, 4), 'x', result_iadd_6328)
    
    # Getting the type of 'x' (line 516)
    x_6329 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 516, 11), 'x')
    # Assigning a type to the variable 'stypy_return_type' (line 516)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 516, 4), 'stypy_return_type', x_6329)
    
    # ################# End of '_krandinit(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_krandinit' in the type store
    # Getting the type of 'stypy_return_type' (line 479)
    stypy_return_type_6330 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 479, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_6330)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_krandinit'
    return stypy_return_type_6330

# Assigning a type to the variable '_krandinit' (line 479)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 479, 0), '_krandinit', _krandinit)

# Assigning a Dict to a Name (line 518):

# Assigning a Dict to a Name (line 518):

# Obtaining an instance of the builtin type 'dict' (line 518)
dict_6331 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 518, 19), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 518)
# Adding element type (key, value) (line 518)
str_6332 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 518, 20), 'str', 'random')
# Getting the type of '_krandinit' (line 518)
_krandinit_6333 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 518, 30), '_krandinit')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 518, 19), dict_6331, (str_6332, _krandinit_6333))
# Adding element type (key, value) (line 518)
str_6334 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 518, 42), 'str', 'points')
# Getting the type of '_kpoints' (line 518)
_kpoints_6335 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 518, 52), '_kpoints')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 518, 19), dict_6331, (str_6334, _kpoints_6335))

# Assigning a type to the variable '_valid_init_meth' (line 518)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 518, 0), '_valid_init_meth', dict_6331)

@norecursion
def _missing_warn(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_missing_warn'
    module_type_store = module_type_store.open_function_context('_missing_warn', 521, 0, False)
    
    # Passed parameters checking function
    _missing_warn.stypy_localization = localization
    _missing_warn.stypy_type_of_self = None
    _missing_warn.stypy_type_store = module_type_store
    _missing_warn.stypy_function_name = '_missing_warn'
    _missing_warn.stypy_param_names_list = []
    _missing_warn.stypy_varargs_param_name = None
    _missing_warn.stypy_kwargs_param_name = None
    _missing_warn.stypy_call_defaults = defaults
    _missing_warn.stypy_call_varargs = varargs
    _missing_warn.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_missing_warn', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_missing_warn', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_missing_warn(...)' code ##################

    str_6336 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 522, 4), 'str', 'Print a warning when called.')
    
    # Call to warn(...): (line 523)
    # Processing the call arguments (line 523)
    str_6339 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 523, 18), 'str', 'One of the clusters is empty. Re-run kmeans with a different initialization.')
    # Processing the call keyword arguments (line 523)
    kwargs_6340 = {}
    # Getting the type of 'warnings' (line 523)
    warnings_6337 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 523, 4), 'warnings', False)
    # Obtaining the member 'warn' of a type (line 523)
    warn_6338 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 523, 4), warnings_6337, 'warn')
    # Calling warn(args, kwargs) (line 523)
    warn_call_result_6341 = invoke(stypy.reporting.localization.Localization(__file__, 523, 4), warn_6338, *[str_6339], **kwargs_6340)
    
    
    # ################# End of '_missing_warn(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_missing_warn' in the type store
    # Getting the type of 'stypy_return_type' (line 521)
    stypy_return_type_6342 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 521, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_6342)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_missing_warn'
    return stypy_return_type_6342

# Assigning a type to the variable '_missing_warn' (line 521)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 521, 0), '_missing_warn', _missing_warn)

@norecursion
def _missing_raise(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_missing_raise'
    module_type_store = module_type_store.open_function_context('_missing_raise', 527, 0, False)
    
    # Passed parameters checking function
    _missing_raise.stypy_localization = localization
    _missing_raise.stypy_type_of_self = None
    _missing_raise.stypy_type_store = module_type_store
    _missing_raise.stypy_function_name = '_missing_raise'
    _missing_raise.stypy_param_names_list = []
    _missing_raise.stypy_varargs_param_name = None
    _missing_raise.stypy_kwargs_param_name = None
    _missing_raise.stypy_call_defaults = defaults
    _missing_raise.stypy_call_varargs = varargs
    _missing_raise.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_missing_raise', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_missing_raise', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_missing_raise(...)' code ##################

    str_6343 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 528, 4), 'str', 'raise a ClusterError when called.')
    
    # Call to ClusterError(...): (line 529)
    # Processing the call arguments (line 529)
    str_6345 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 529, 23), 'str', 'One of the clusters is empty. Re-run kmeans with a different initialization.')
    # Processing the call keyword arguments (line 529)
    kwargs_6346 = {}
    # Getting the type of 'ClusterError' (line 529)
    ClusterError_6344 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 529, 10), 'ClusterError', False)
    # Calling ClusterError(args, kwargs) (line 529)
    ClusterError_call_result_6347 = invoke(stypy.reporting.localization.Localization(__file__, 529, 10), ClusterError_6344, *[str_6345], **kwargs_6346)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 529, 4), ClusterError_call_result_6347, 'raise parameter', BaseException)
    
    # ################# End of '_missing_raise(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_missing_raise' in the type store
    # Getting the type of 'stypy_return_type' (line 527)
    stypy_return_type_6348 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 527, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_6348)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_missing_raise'
    return stypy_return_type_6348

# Assigning a type to the variable '_missing_raise' (line 527)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 527, 0), '_missing_raise', _missing_raise)

# Assigning a Dict to a Name (line 532):

# Assigning a Dict to a Name (line 532):

# Obtaining an instance of the builtin type 'dict' (line 532)
dict_6349 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 532, 19), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 532)
# Adding element type (key, value) (line 532)
str_6350 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 532, 20), 'str', 'warn')
# Getting the type of '_missing_warn' (line 532)
_missing_warn_6351 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 532, 28), '_missing_warn')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 532, 19), dict_6349, (str_6350, _missing_warn_6351))
# Adding element type (key, value) (line 532)
str_6352 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 532, 43), 'str', 'raise')
# Getting the type of '_missing_raise' (line 532)
_missing_raise_6353 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 532, 52), '_missing_raise')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 532, 19), dict_6349, (str_6352, _missing_raise_6353))

# Assigning a type to the variable '_valid_miss_meth' (line 532)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 532, 0), '_valid_miss_meth', dict_6349)

@norecursion
def kmeans2(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    int_6354 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 535, 26), 'int')
    float_6355 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 535, 37), 'float')
    str_6356 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 535, 49), 'str', 'random')
    str_6357 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 536, 20), 'str', 'warn')
    # Getting the type of 'True' (line 536)
    True_6358 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 536, 41), 'True')
    defaults = [int_6354, float_6355, str_6356, str_6357, True_6358]
    # Create a new context for function 'kmeans2'
    module_type_store = module_type_store.open_function_context('kmeans2', 535, 0, False)
    
    # Passed parameters checking function
    kmeans2.stypy_localization = localization
    kmeans2.stypy_type_of_self = None
    kmeans2.stypy_type_store = module_type_store
    kmeans2.stypy_function_name = 'kmeans2'
    kmeans2.stypy_param_names_list = ['data', 'k', 'iter', 'thresh', 'minit', 'missing', 'check_finite']
    kmeans2.stypy_varargs_param_name = None
    kmeans2.stypy_kwargs_param_name = None
    kmeans2.stypy_call_defaults = defaults
    kmeans2.stypy_call_varargs = varargs
    kmeans2.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'kmeans2', ['data', 'k', 'iter', 'thresh', 'minit', 'missing', 'check_finite'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'kmeans2', localization, ['data', 'k', 'iter', 'thresh', 'minit', 'missing', 'check_finite'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'kmeans2(...)' code ##################

    str_6359 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 594, (-1)), 'str', "\n    Classify a set of observations into k clusters using the k-means algorithm.\n\n    The algorithm attempts to minimize the Euclidian distance between\n    observations and centroids. Several initialization methods are\n    included.\n\n    Parameters\n    ----------\n    data : ndarray\n        A 'M' by 'N' array of 'M' observations in 'N' dimensions or a length\n        'M' array of 'M' one-dimensional observations.\n    k : int or ndarray\n        The number of clusters to form as well as the number of\n        centroids to generate. If `minit` initialization string is\n        'matrix', or if a ndarray is given instead, it is\n        interpreted as initial cluster to use instead.\n    iter : int, optional\n        Number of iterations of the k-means algorithm to run. Note\n        that this differs in meaning from the iters parameter to\n        the kmeans function.\n    thresh : float, optional\n        (not used yet)\n    minit : str, optional\n        Method for initialization. Available methods are 'random',\n        'points', and 'matrix':\n\n        'random': generate k centroids from a Gaussian with mean and\n        variance estimated from the data.\n\n        'points': choose k observations (rows) at random from data for\n        the initial centroids.\n\n        'matrix': interpret the k parameter as a k by M (or length k\n        array for one-dimensional data) array of initial centroids.\n    missing : str, optional\n        Method to deal with empty clusters. Available methods are\n        'warn' and 'raise':\n\n        'warn': give a warning and continue.\n\n        'raise': raise an ClusterError and terminate the algorithm.\n    check_finite : bool, optional\n        Whether to check that the input matrices contain only finite numbers.\n        Disabling may give a performance gain, but may result in problems\n        (crashes, non-termination) if the inputs do contain infinities or NaNs.\n        Default: True\n\n    Returns\n    -------\n    centroid : ndarray\n        A 'k' by 'N' array of centroids found at the last iteration of\n        k-means.\n    label : ndarray\n        label[i] is the code or index of the centroid the\n        i'th observation is closest to.\n\n    ")
    
    
    
    # Call to int(...): (line 595)
    # Processing the call arguments (line 595)
    # Getting the type of 'iter' (line 595)
    iter_6361 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 595, 11), 'iter', False)
    # Processing the call keyword arguments (line 595)
    kwargs_6362 = {}
    # Getting the type of 'int' (line 595)
    int_6360 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 595, 7), 'int', False)
    # Calling int(args, kwargs) (line 595)
    int_call_result_6363 = invoke(stypy.reporting.localization.Localization(__file__, 595, 7), int_6360, *[iter_6361], **kwargs_6362)
    
    int_6364 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 595, 19), 'int')
    # Applying the binary operator '<' (line 595)
    result_lt_6365 = python_operator(stypy.reporting.localization.Localization(__file__, 595, 7), '<', int_call_result_6363, int_6364)
    
    # Testing the type of an if condition (line 595)
    if_condition_6366 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 595, 4), result_lt_6365)
    # Assigning a type to the variable 'if_condition_6366' (line 595)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 595, 4), 'if_condition_6366', if_condition_6366)
    # SSA begins for if statement (line 595)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 596)
    # Processing the call arguments (line 596)
    str_6368 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 596, 25), 'str', 'Invalid iter (%s), must be a positive integer.')
    # Getting the type of 'iter' (line 597)
    iter_6369 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 597, 57), 'iter', False)
    # Applying the binary operator '%' (line 596)
    result_mod_6370 = python_operator(stypy.reporting.localization.Localization(__file__, 596, 25), '%', str_6368, iter_6369)
    
    # Processing the call keyword arguments (line 596)
    kwargs_6371 = {}
    # Getting the type of 'ValueError' (line 596)
    ValueError_6367 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 596, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 596)
    ValueError_call_result_6372 = invoke(stypy.reporting.localization.Localization(__file__, 596, 14), ValueError_6367, *[result_mod_6370], **kwargs_6371)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 596, 8), ValueError_call_result_6372, 'raise parameter', BaseException)
    # SSA join for if statement (line 595)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # SSA begins for try-except statement (line 598)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
    
    # Assigning a Subscript to a Name (line 599):
    
    # Assigning a Subscript to a Name (line 599):
    
    # Obtaining the type of the subscript
    # Getting the type of 'missing' (line 599)
    missing_6373 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 599, 37), 'missing')
    # Getting the type of '_valid_miss_meth' (line 599)
    _valid_miss_meth_6374 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 599, 20), '_valid_miss_meth')
    # Obtaining the member '__getitem__' of a type (line 599)
    getitem___6375 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 599, 20), _valid_miss_meth_6374, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 599)
    subscript_call_result_6376 = invoke(stypy.reporting.localization.Localization(__file__, 599, 20), getitem___6375, missing_6373)
    
    # Assigning a type to the variable 'miss_meth' (line 599)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 599, 8), 'miss_meth', subscript_call_result_6376)
    # SSA branch for the except part of a try statement (line 598)
    # SSA branch for the except 'KeyError' branch of a try statement (line 598)
    module_type_store.open_ssa_branch('except')
    
    # Call to ValueError(...): (line 601)
    # Processing the call arguments (line 601)
    str_6378 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 601, 25), 'str', 'Unknown missing method %r')
    
    # Obtaining an instance of the builtin type 'tuple' (line 601)
    tuple_6379 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 601, 56), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 601)
    # Adding element type (line 601)
    # Getting the type of 'missing' (line 601)
    missing_6380 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 601, 56), 'missing', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 601, 56), tuple_6379, missing_6380)
    
    # Applying the binary operator '%' (line 601)
    result_mod_6381 = python_operator(stypy.reporting.localization.Localization(__file__, 601, 25), '%', str_6378, tuple_6379)
    
    # Processing the call keyword arguments (line 601)
    kwargs_6382 = {}
    # Getting the type of 'ValueError' (line 601)
    ValueError_6377 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 601, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 601)
    ValueError_call_result_6383 = invoke(stypy.reporting.localization.Localization(__file__, 601, 14), ValueError_6377, *[result_mod_6381], **kwargs_6382)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 601, 8), ValueError_call_result_6383, 'raise parameter', BaseException)
    # SSA join for try-except statement (line 598)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 603):
    
    # Assigning a Call to a Name (line 603):
    
    # Call to _asarray_validated(...): (line 603)
    # Processing the call arguments (line 603)
    # Getting the type of 'data' (line 603)
    data_6385 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 603, 30), 'data', False)
    # Processing the call keyword arguments (line 603)
    # Getting the type of 'check_finite' (line 603)
    check_finite_6386 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 603, 49), 'check_finite', False)
    keyword_6387 = check_finite_6386
    kwargs_6388 = {'check_finite': keyword_6387}
    # Getting the type of '_asarray_validated' (line 603)
    _asarray_validated_6384 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 603, 11), '_asarray_validated', False)
    # Calling _asarray_validated(args, kwargs) (line 603)
    _asarray_validated_call_result_6389 = invoke(stypy.reporting.localization.Localization(__file__, 603, 11), _asarray_validated_6384, *[data_6385], **kwargs_6388)
    
    # Assigning a type to the variable 'data' (line 603)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 603, 4), 'data', _asarray_validated_call_result_6389)
    
    
    # Getting the type of 'data' (line 604)
    data_6390 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 604, 7), 'data')
    # Obtaining the member 'ndim' of a type (line 604)
    ndim_6391 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 604, 7), data_6390, 'ndim')
    int_6392 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 604, 20), 'int')
    # Applying the binary operator '==' (line 604)
    result_eq_6393 = python_operator(stypy.reporting.localization.Localization(__file__, 604, 7), '==', ndim_6391, int_6392)
    
    # Testing the type of an if condition (line 604)
    if_condition_6394 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 604, 4), result_eq_6393)
    # Assigning a type to the variable 'if_condition_6394' (line 604)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 604, 4), 'if_condition_6394', if_condition_6394)
    # SSA begins for if statement (line 604)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Num to a Name (line 605):
    
    # Assigning a Num to a Name (line 605):
    int_6395 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 605, 12), 'int')
    # Assigning a type to the variable 'd' (line 605)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 605, 8), 'd', int_6395)
    # SSA branch for the else part of an if statement (line 604)
    module_type_store.open_ssa_branch('else')
    
    
    # Getting the type of 'data' (line 606)
    data_6396 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 606, 9), 'data')
    # Obtaining the member 'ndim' of a type (line 606)
    ndim_6397 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 606, 9), data_6396, 'ndim')
    int_6398 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 606, 22), 'int')
    # Applying the binary operator '==' (line 606)
    result_eq_6399 = python_operator(stypy.reporting.localization.Localization(__file__, 606, 9), '==', ndim_6397, int_6398)
    
    # Testing the type of an if condition (line 606)
    if_condition_6400 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 606, 9), result_eq_6399)
    # Assigning a type to the variable 'if_condition_6400' (line 606)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 606, 9), 'if_condition_6400', if_condition_6400)
    # SSA begins for if statement (line 606)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Subscript to a Name (line 607):
    
    # Assigning a Subscript to a Name (line 607):
    
    # Obtaining the type of the subscript
    int_6401 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 607, 23), 'int')
    # Getting the type of 'data' (line 607)
    data_6402 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 607, 12), 'data')
    # Obtaining the member 'shape' of a type (line 607)
    shape_6403 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 607, 12), data_6402, 'shape')
    # Obtaining the member '__getitem__' of a type (line 607)
    getitem___6404 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 607, 12), shape_6403, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 607)
    subscript_call_result_6405 = invoke(stypy.reporting.localization.Localization(__file__, 607, 12), getitem___6404, int_6401)
    
    # Assigning a type to the variable 'd' (line 607)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 607, 8), 'd', subscript_call_result_6405)
    # SSA branch for the else part of an if statement (line 606)
    module_type_store.open_ssa_branch('else')
    
    # Call to ValueError(...): (line 609)
    # Processing the call arguments (line 609)
    str_6407 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 609, 25), 'str', 'Input of rank > 2 is not supported.')
    # Processing the call keyword arguments (line 609)
    kwargs_6408 = {}
    # Getting the type of 'ValueError' (line 609)
    ValueError_6406 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 609, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 609)
    ValueError_call_result_6409 = invoke(stypy.reporting.localization.Localization(__file__, 609, 14), ValueError_6406, *[str_6407], **kwargs_6408)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 609, 8), ValueError_call_result_6409, 'raise parameter', BaseException)
    # SSA join for if statement (line 606)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 604)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'data' (line 611)
    data_6410 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 611, 7), 'data')
    # Obtaining the member 'size' of a type (line 611)
    size_6411 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 611, 7), data_6410, 'size')
    int_6412 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 611, 19), 'int')
    # Applying the binary operator '<' (line 611)
    result_lt_6413 = python_operator(stypy.reporting.localization.Localization(__file__, 611, 7), '<', size_6411, int_6412)
    
    # Testing the type of an if condition (line 611)
    if_condition_6414 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 611, 4), result_lt_6413)
    # Assigning a type to the variable 'if_condition_6414' (line 611)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 611, 4), 'if_condition_6414', if_condition_6414)
    # SSA begins for if statement (line 611)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 612)
    # Processing the call arguments (line 612)
    str_6416 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 612, 25), 'str', 'Empty input is not supported.')
    # Processing the call keyword arguments (line 612)
    kwargs_6417 = {}
    # Getting the type of 'ValueError' (line 612)
    ValueError_6415 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 612, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 612)
    ValueError_call_result_6418 = invoke(stypy.reporting.localization.Localization(__file__, 612, 14), ValueError_6415, *[str_6416], **kwargs_6417)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 612, 8), ValueError_call_result_6418, 'raise parameter', BaseException)
    # SSA join for if statement (line 611)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'minit' (line 615)
    minit_6419 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 615, 7), 'minit')
    str_6420 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 615, 16), 'str', 'matrix')
    # Applying the binary operator '==' (line 615)
    result_eq_6421 = python_operator(stypy.reporting.localization.Localization(__file__, 615, 7), '==', minit_6419, str_6420)
    
    
    
    # Call to isscalar(...): (line 615)
    # Processing the call arguments (line 615)
    # Getting the type of 'k' (line 615)
    k_6424 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 615, 44), 'k', False)
    # Processing the call keyword arguments (line 615)
    kwargs_6425 = {}
    # Getting the type of 'np' (line 615)
    np_6422 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 615, 32), 'np', False)
    # Obtaining the member 'isscalar' of a type (line 615)
    isscalar_6423 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 615, 32), np_6422, 'isscalar')
    # Calling isscalar(args, kwargs) (line 615)
    isscalar_call_result_6426 = invoke(stypy.reporting.localization.Localization(__file__, 615, 32), isscalar_6423, *[k_6424], **kwargs_6425)
    
    # Applying the 'not' unary operator (line 615)
    result_not__6427 = python_operator(stypy.reporting.localization.Localization(__file__, 615, 28), 'not', isscalar_call_result_6426)
    
    # Applying the binary operator 'or' (line 615)
    result_or_keyword_6428 = python_operator(stypy.reporting.localization.Localization(__file__, 615, 7), 'or', result_eq_6421, result_not__6427)
    
    # Testing the type of an if condition (line 615)
    if_condition_6429 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 615, 4), result_or_keyword_6428)
    # Assigning a type to the variable 'if_condition_6429' (line 615)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 615, 4), 'if_condition_6429', if_condition_6429)
    # SSA begins for if statement (line 615)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 616):
    
    # Assigning a Call to a Name (line 616):
    
    # Call to array(...): (line 616)
    # Processing the call arguments (line 616)
    # Getting the type of 'k' (line 616)
    k_6432 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 616, 29), 'k', False)
    # Processing the call keyword arguments (line 616)
    # Getting the type of 'True' (line 616)
    True_6433 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 616, 37), 'True', False)
    keyword_6434 = True_6433
    kwargs_6435 = {'copy': keyword_6434}
    # Getting the type of 'np' (line 616)
    np_6430 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 616, 20), 'np', False)
    # Obtaining the member 'array' of a type (line 616)
    array_6431 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 616, 20), np_6430, 'array')
    # Calling array(args, kwargs) (line 616)
    array_call_result_6436 = invoke(stypy.reporting.localization.Localization(__file__, 616, 20), array_6431, *[k_6432], **kwargs_6435)
    
    # Assigning a type to the variable 'code_book' (line 616)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 616, 8), 'code_book', array_call_result_6436)
    
    
    # Getting the type of 'data' (line 617)
    data_6437 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 617, 11), 'data')
    # Obtaining the member 'ndim' of a type (line 617)
    ndim_6438 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 617, 11), data_6437, 'ndim')
    # Getting the type of 'code_book' (line 617)
    code_book_6439 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 617, 24), 'code_book')
    # Obtaining the member 'ndim' of a type (line 617)
    ndim_6440 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 617, 24), code_book_6439, 'ndim')
    # Applying the binary operator '!=' (line 617)
    result_ne_6441 = python_operator(stypy.reporting.localization.Localization(__file__, 617, 11), '!=', ndim_6438, ndim_6440)
    
    # Testing the type of an if condition (line 617)
    if_condition_6442 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 617, 8), result_ne_6441)
    # Assigning a type to the variable 'if_condition_6442' (line 617)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 617, 8), 'if_condition_6442', if_condition_6442)
    # SSA begins for if statement (line 617)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 618)
    # Processing the call arguments (line 618)
    str_6444 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 618, 29), 'str', "k array doesn't match data rank")
    # Processing the call keyword arguments (line 618)
    kwargs_6445 = {}
    # Getting the type of 'ValueError' (line 618)
    ValueError_6443 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 618, 18), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 618)
    ValueError_call_result_6446 = invoke(stypy.reporting.localization.Localization(__file__, 618, 18), ValueError_6443, *[str_6444], **kwargs_6445)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 618, 12), ValueError_call_result_6446, 'raise parameter', BaseException)
    # SSA join for if statement (line 617)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 619):
    
    # Assigning a Call to a Name (line 619):
    
    # Call to len(...): (line 619)
    # Processing the call arguments (line 619)
    # Getting the type of 'code_book' (line 619)
    code_book_6448 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 619, 17), 'code_book', False)
    # Processing the call keyword arguments (line 619)
    kwargs_6449 = {}
    # Getting the type of 'len' (line 619)
    len_6447 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 619, 13), 'len', False)
    # Calling len(args, kwargs) (line 619)
    len_call_result_6450 = invoke(stypy.reporting.localization.Localization(__file__, 619, 13), len_6447, *[code_book_6448], **kwargs_6449)
    
    # Assigning a type to the variable 'nc' (line 619)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 619, 8), 'nc', len_call_result_6450)
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'data' (line 620)
    data_6451 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 620, 11), 'data')
    # Obtaining the member 'ndim' of a type (line 620)
    ndim_6452 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 620, 11), data_6451, 'ndim')
    int_6453 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 620, 23), 'int')
    # Applying the binary operator '>' (line 620)
    result_gt_6454 = python_operator(stypy.reporting.localization.Localization(__file__, 620, 11), '>', ndim_6452, int_6453)
    
    
    
    # Obtaining the type of the subscript
    int_6455 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 620, 45), 'int')
    # Getting the type of 'code_book' (line 620)
    code_book_6456 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 620, 29), 'code_book')
    # Obtaining the member 'shape' of a type (line 620)
    shape_6457 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 620, 29), code_book_6456, 'shape')
    # Obtaining the member '__getitem__' of a type (line 620)
    getitem___6458 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 620, 29), shape_6457, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 620)
    subscript_call_result_6459 = invoke(stypy.reporting.localization.Localization(__file__, 620, 29), getitem___6458, int_6455)
    
    # Getting the type of 'd' (line 620)
    d_6460 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 620, 51), 'd')
    # Applying the binary operator '!=' (line 620)
    result_ne_6461 = python_operator(stypy.reporting.localization.Localization(__file__, 620, 29), '!=', subscript_call_result_6459, d_6460)
    
    # Applying the binary operator 'and' (line 620)
    result_and_keyword_6462 = python_operator(stypy.reporting.localization.Localization(__file__, 620, 11), 'and', result_gt_6454, result_ne_6461)
    
    # Testing the type of an if condition (line 620)
    if_condition_6463 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 620, 8), result_and_keyword_6462)
    # Assigning a type to the variable 'if_condition_6463' (line 620)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 620, 8), 'if_condition_6463', if_condition_6463)
    # SSA begins for if statement (line 620)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 621)
    # Processing the call arguments (line 621)
    str_6465 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 621, 29), 'str', "k array doesn't match data dimension")
    # Processing the call keyword arguments (line 621)
    kwargs_6466 = {}
    # Getting the type of 'ValueError' (line 621)
    ValueError_6464 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 621, 18), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 621)
    ValueError_call_result_6467 = invoke(stypy.reporting.localization.Localization(__file__, 621, 18), ValueError_6464, *[str_6465], **kwargs_6466)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 621, 12), ValueError_call_result_6467, 'raise parameter', BaseException)
    # SSA join for if statement (line 620)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA branch for the else part of an if statement (line 615)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Call to a Name (line 623):
    
    # Assigning a Call to a Name (line 623):
    
    # Call to int(...): (line 623)
    # Processing the call arguments (line 623)
    # Getting the type of 'k' (line 623)
    k_6469 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 623, 17), 'k', False)
    # Processing the call keyword arguments (line 623)
    kwargs_6470 = {}
    # Getting the type of 'int' (line 623)
    int_6468 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 623, 13), 'int', False)
    # Calling int(args, kwargs) (line 623)
    int_call_result_6471 = invoke(stypy.reporting.localization.Localization(__file__, 623, 13), int_6468, *[k_6469], **kwargs_6470)
    
    # Assigning a type to the variable 'nc' (line 623)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 623, 8), 'nc', int_call_result_6471)
    
    
    # Getting the type of 'nc' (line 625)
    nc_6472 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 625, 11), 'nc')
    int_6473 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 625, 16), 'int')
    # Applying the binary operator '<' (line 625)
    result_lt_6474 = python_operator(stypy.reporting.localization.Localization(__file__, 625, 11), '<', nc_6472, int_6473)
    
    # Testing the type of an if condition (line 625)
    if_condition_6475 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 625, 8), result_lt_6474)
    # Assigning a type to the variable 'if_condition_6475' (line 625)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 625, 8), 'if_condition_6475', if_condition_6475)
    # SSA begins for if statement (line 625)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 626)
    # Processing the call arguments (line 626)
    str_6477 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 626, 29), 'str', 'Cannot ask kmeans2 for %d clusters (k was %s)')
    
    # Obtaining an instance of the builtin type 'tuple' (line 627)
    tuple_6478 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 627, 46), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 627)
    # Adding element type (line 627)
    # Getting the type of 'nc' (line 627)
    nc_6479 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 627, 46), 'nc', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 627, 46), tuple_6478, nc_6479)
    # Adding element type (line 627)
    # Getting the type of 'k' (line 627)
    k_6480 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 627, 50), 'k', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 627, 46), tuple_6478, k_6480)
    
    # Applying the binary operator '%' (line 626)
    result_mod_6481 = python_operator(stypy.reporting.localization.Localization(__file__, 626, 29), '%', str_6477, tuple_6478)
    
    # Processing the call keyword arguments (line 626)
    kwargs_6482 = {}
    # Getting the type of 'ValueError' (line 626)
    ValueError_6476 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 626, 18), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 626)
    ValueError_call_result_6483 = invoke(stypy.reporting.localization.Localization(__file__, 626, 18), ValueError_6476, *[result_mod_6481], **kwargs_6482)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 626, 12), ValueError_call_result_6483, 'raise parameter', BaseException)
    # SSA branch for the else part of an if statement (line 625)
    module_type_store.open_ssa_branch('else')
    
    
    # Getting the type of 'nc' (line 628)
    nc_6484 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 628, 13), 'nc')
    # Getting the type of 'k' (line 628)
    k_6485 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 628, 19), 'k')
    # Applying the binary operator '!=' (line 628)
    result_ne_6486 = python_operator(stypy.reporting.localization.Localization(__file__, 628, 13), '!=', nc_6484, k_6485)
    
    # Testing the type of an if condition (line 628)
    if_condition_6487 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 628, 13), result_ne_6486)
    # Assigning a type to the variable 'if_condition_6487' (line 628)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 628, 13), 'if_condition_6487', if_condition_6487)
    # SSA begins for if statement (line 628)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to warn(...): (line 629)
    # Processing the call arguments (line 629)
    str_6490 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 629, 26), 'str', 'k was not an integer, was converted.')
    # Processing the call keyword arguments (line 629)
    kwargs_6491 = {}
    # Getting the type of 'warnings' (line 629)
    warnings_6488 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 629, 12), 'warnings', False)
    # Obtaining the member 'warn' of a type (line 629)
    warn_6489 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 629, 12), warnings_6488, 'warn')
    # Calling warn(args, kwargs) (line 629)
    warn_call_result_6492 = invoke(stypy.reporting.localization.Localization(__file__, 629, 12), warn_6489, *[str_6490], **kwargs_6491)
    
    # SSA join for if statement (line 628)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 625)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # SSA begins for try-except statement (line 631)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
    
    # Assigning a Subscript to a Name (line 632):
    
    # Assigning a Subscript to a Name (line 632):
    
    # Obtaining the type of the subscript
    # Getting the type of 'minit' (line 632)
    minit_6493 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 632, 41), 'minit')
    # Getting the type of '_valid_init_meth' (line 632)
    _valid_init_meth_6494 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 632, 24), '_valid_init_meth')
    # Obtaining the member '__getitem__' of a type (line 632)
    getitem___6495 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 632, 24), _valid_init_meth_6494, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 632)
    subscript_call_result_6496 = invoke(stypy.reporting.localization.Localization(__file__, 632, 24), getitem___6495, minit_6493)
    
    # Assigning a type to the variable 'init_meth' (line 632)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 632, 12), 'init_meth', subscript_call_result_6496)
    # SSA branch for the except part of a try statement (line 631)
    # SSA branch for the except 'KeyError' branch of a try statement (line 631)
    module_type_store.open_ssa_branch('except')
    
    # Call to ValueError(...): (line 634)
    # Processing the call arguments (line 634)
    str_6498 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 634, 29), 'str', 'Unknown init method %r')
    
    # Obtaining an instance of the builtin type 'tuple' (line 634)
    tuple_6499 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 634, 57), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 634)
    # Adding element type (line 634)
    # Getting the type of 'minit' (line 634)
    minit_6500 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 634, 57), 'minit', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 634, 57), tuple_6499, minit_6500)
    
    # Applying the binary operator '%' (line 634)
    result_mod_6501 = python_operator(stypy.reporting.localization.Localization(__file__, 634, 29), '%', str_6498, tuple_6499)
    
    # Processing the call keyword arguments (line 634)
    kwargs_6502 = {}
    # Getting the type of 'ValueError' (line 634)
    ValueError_6497 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 634, 18), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 634)
    ValueError_call_result_6503 = invoke(stypy.reporting.localization.Localization(__file__, 634, 18), ValueError_6497, *[result_mod_6501], **kwargs_6502)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 634, 12), ValueError_call_result_6503, 'raise parameter', BaseException)
    # SSA branch for the else branch of a try statement (line 631)
    module_type_store.open_ssa_branch('except else')
    
    # Assigning a Call to a Name (line 636):
    
    # Assigning a Call to a Name (line 636):
    
    # Call to init_meth(...): (line 636)
    # Processing the call arguments (line 636)
    # Getting the type of 'data' (line 636)
    data_6505 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 636, 34), 'data', False)
    # Getting the type of 'k' (line 636)
    k_6506 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 636, 40), 'k', False)
    # Processing the call keyword arguments (line 636)
    kwargs_6507 = {}
    # Getting the type of 'init_meth' (line 636)
    init_meth_6504 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 636, 24), 'init_meth', False)
    # Calling init_meth(args, kwargs) (line 636)
    init_meth_call_result_6508 = invoke(stypy.reporting.localization.Localization(__file__, 636, 24), init_meth_6504, *[data_6505, k_6506], **kwargs_6507)
    
    # Assigning a type to the variable 'code_book' (line 636)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 636, 12), 'code_book', init_meth_call_result_6508)
    # SSA join for try-except statement (line 631)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 615)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Call to xrange(...): (line 638)
    # Processing the call arguments (line 638)
    # Getting the type of 'iter' (line 638)
    iter_6510 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 638, 20), 'iter', False)
    # Processing the call keyword arguments (line 638)
    kwargs_6511 = {}
    # Getting the type of 'xrange' (line 638)
    xrange_6509 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 638, 13), 'xrange', False)
    # Calling xrange(args, kwargs) (line 638)
    xrange_call_result_6512 = invoke(stypy.reporting.localization.Localization(__file__, 638, 13), xrange_6509, *[iter_6510], **kwargs_6511)
    
    # Testing the type of a for loop iterable (line 638)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 638, 4), xrange_call_result_6512)
    # Getting the type of the for loop variable (line 638)
    for_loop_var_6513 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 638, 4), xrange_call_result_6512)
    # Assigning a type to the variable 'i' (line 638)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 638, 4), 'i', for_loop_var_6513)
    # SSA begins for a for statement (line 638)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Subscript to a Name (line 640):
    
    # Assigning a Subscript to a Name (line 640):
    
    # Obtaining the type of the subscript
    int_6514 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 640, 36), 'int')
    
    # Call to vq(...): (line 640)
    # Processing the call arguments (line 640)
    # Getting the type of 'data' (line 640)
    data_6516 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 640, 19), 'data', False)
    # Getting the type of 'code_book' (line 640)
    code_book_6517 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 640, 25), 'code_book', False)
    # Processing the call keyword arguments (line 640)
    kwargs_6518 = {}
    # Getting the type of 'vq' (line 640)
    vq_6515 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 640, 16), 'vq', False)
    # Calling vq(args, kwargs) (line 640)
    vq_call_result_6519 = invoke(stypy.reporting.localization.Localization(__file__, 640, 16), vq_6515, *[data_6516, code_book_6517], **kwargs_6518)
    
    # Obtaining the member '__getitem__' of a type (line 640)
    getitem___6520 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 640, 16), vq_call_result_6519, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 640)
    subscript_call_result_6521 = invoke(stypy.reporting.localization.Localization(__file__, 640, 16), getitem___6520, int_6514)
    
    # Assigning a type to the variable 'label' (line 640)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 640, 8), 'label', subscript_call_result_6521)
    
    # Assigning a Call to a Tuple (line 642):
    
    # Assigning a Subscript to a Name (line 642):
    
    # Obtaining the type of the subscript
    int_6522 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 642, 8), 'int')
    
    # Call to update_cluster_means(...): (line 642)
    # Processing the call arguments (line 642)
    # Getting the type of 'data' (line 642)
    data_6525 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 642, 62), 'data', False)
    # Getting the type of 'label' (line 642)
    label_6526 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 642, 68), 'label', False)
    # Getting the type of 'nc' (line 642)
    nc_6527 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 642, 75), 'nc', False)
    # Processing the call keyword arguments (line 642)
    kwargs_6528 = {}
    # Getting the type of '_vq' (line 642)
    _vq_6523 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 642, 37), '_vq', False)
    # Obtaining the member 'update_cluster_means' of a type (line 642)
    update_cluster_means_6524 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 642, 37), _vq_6523, 'update_cluster_means')
    # Calling update_cluster_means(args, kwargs) (line 642)
    update_cluster_means_call_result_6529 = invoke(stypy.reporting.localization.Localization(__file__, 642, 37), update_cluster_means_6524, *[data_6525, label_6526, nc_6527], **kwargs_6528)
    
    # Obtaining the member '__getitem__' of a type (line 642)
    getitem___6530 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 642, 8), update_cluster_means_call_result_6529, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 642)
    subscript_call_result_6531 = invoke(stypy.reporting.localization.Localization(__file__, 642, 8), getitem___6530, int_6522)
    
    # Assigning a type to the variable 'tuple_var_assignment_5745' (line 642)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 642, 8), 'tuple_var_assignment_5745', subscript_call_result_6531)
    
    # Assigning a Subscript to a Name (line 642):
    
    # Obtaining the type of the subscript
    int_6532 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 642, 8), 'int')
    
    # Call to update_cluster_means(...): (line 642)
    # Processing the call arguments (line 642)
    # Getting the type of 'data' (line 642)
    data_6535 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 642, 62), 'data', False)
    # Getting the type of 'label' (line 642)
    label_6536 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 642, 68), 'label', False)
    # Getting the type of 'nc' (line 642)
    nc_6537 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 642, 75), 'nc', False)
    # Processing the call keyword arguments (line 642)
    kwargs_6538 = {}
    # Getting the type of '_vq' (line 642)
    _vq_6533 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 642, 37), '_vq', False)
    # Obtaining the member 'update_cluster_means' of a type (line 642)
    update_cluster_means_6534 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 642, 37), _vq_6533, 'update_cluster_means')
    # Calling update_cluster_means(args, kwargs) (line 642)
    update_cluster_means_call_result_6539 = invoke(stypy.reporting.localization.Localization(__file__, 642, 37), update_cluster_means_6534, *[data_6535, label_6536, nc_6537], **kwargs_6538)
    
    # Obtaining the member '__getitem__' of a type (line 642)
    getitem___6540 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 642, 8), update_cluster_means_call_result_6539, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 642)
    subscript_call_result_6541 = invoke(stypy.reporting.localization.Localization(__file__, 642, 8), getitem___6540, int_6532)
    
    # Assigning a type to the variable 'tuple_var_assignment_5746' (line 642)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 642, 8), 'tuple_var_assignment_5746', subscript_call_result_6541)
    
    # Assigning a Name to a Name (line 642):
    # Getting the type of 'tuple_var_assignment_5745' (line 642)
    tuple_var_assignment_5745_6542 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 642, 8), 'tuple_var_assignment_5745')
    # Assigning a type to the variable 'new_code_book' (line 642)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 642, 8), 'new_code_book', tuple_var_assignment_5745_6542)
    
    # Assigning a Name to a Name (line 642):
    # Getting the type of 'tuple_var_assignment_5746' (line 642)
    tuple_var_assignment_5746_6543 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 642, 8), 'tuple_var_assignment_5746')
    # Assigning a type to the variable 'has_members' (line 642)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 642, 23), 'has_members', tuple_var_assignment_5746_6543)
    
    
    
    # Call to all(...): (line 643)
    # Processing the call keyword arguments (line 643)
    kwargs_6546 = {}
    # Getting the type of 'has_members' (line 643)
    has_members_6544 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 643, 15), 'has_members', False)
    # Obtaining the member 'all' of a type (line 643)
    all_6545 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 643, 15), has_members_6544, 'all')
    # Calling all(args, kwargs) (line 643)
    all_call_result_6547 = invoke(stypy.reporting.localization.Localization(__file__, 643, 15), all_6545, *[], **kwargs_6546)
    
    # Applying the 'not' unary operator (line 643)
    result_not__6548 = python_operator(stypy.reporting.localization.Localization(__file__, 643, 11), 'not', all_call_result_6547)
    
    # Testing the type of an if condition (line 643)
    if_condition_6549 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 643, 8), result_not__6548)
    # Assigning a type to the variable 'if_condition_6549' (line 643)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 643, 8), 'if_condition_6549', if_condition_6549)
    # SSA begins for if statement (line 643)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to miss_meth(...): (line 644)
    # Processing the call keyword arguments (line 644)
    kwargs_6551 = {}
    # Getting the type of 'miss_meth' (line 644)
    miss_meth_6550 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 644, 12), 'miss_meth', False)
    # Calling miss_meth(args, kwargs) (line 644)
    miss_meth_call_result_6552 = invoke(stypy.reporting.localization.Localization(__file__, 644, 12), miss_meth_6550, *[], **kwargs_6551)
    
    
    # Assigning a Subscript to a Subscript (line 646):
    
    # Assigning a Subscript to a Subscript (line 646):
    
    # Obtaining the type of the subscript
    
    # Getting the type of 'has_members' (line 646)
    has_members_6553 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 646, 53), 'has_members')
    # Applying the '~' unary operator (line 646)
    result_inv_6554 = python_operator(stypy.reporting.localization.Localization(__file__, 646, 52), '~', has_members_6553)
    
    # Getting the type of 'code_book' (line 646)
    code_book_6555 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 646, 42), 'code_book')
    # Obtaining the member '__getitem__' of a type (line 646)
    getitem___6556 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 646, 42), code_book_6555, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 646)
    subscript_call_result_6557 = invoke(stypy.reporting.localization.Localization(__file__, 646, 42), getitem___6556, result_inv_6554)
    
    # Getting the type of 'new_code_book' (line 646)
    new_code_book_6558 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 646, 12), 'new_code_book')
    
    # Getting the type of 'has_members' (line 646)
    has_members_6559 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 646, 27), 'has_members')
    # Applying the '~' unary operator (line 646)
    result_inv_6560 = python_operator(stypy.reporting.localization.Localization(__file__, 646, 26), '~', has_members_6559)
    
    # Storing an element on a container (line 646)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 646, 12), new_code_book_6558, (result_inv_6560, subscript_call_result_6557))
    # SSA join for if statement (line 643)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Name to a Name (line 647):
    
    # Assigning a Name to a Name (line 647):
    # Getting the type of 'new_code_book' (line 647)
    new_code_book_6561 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 647, 20), 'new_code_book')
    # Assigning a type to the variable 'code_book' (line 647)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 647, 8), 'code_book', new_code_book_6561)
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Obtaining an instance of the builtin type 'tuple' (line 649)
    tuple_6562 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 649, 11), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 649)
    # Adding element type (line 649)
    # Getting the type of 'code_book' (line 649)
    code_book_6563 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 649, 11), 'code_book')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 649, 11), tuple_6562, code_book_6563)
    # Adding element type (line 649)
    # Getting the type of 'label' (line 649)
    label_6564 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 649, 22), 'label')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 649, 11), tuple_6562, label_6564)
    
    # Assigning a type to the variable 'stypy_return_type' (line 649)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 649, 4), 'stypy_return_type', tuple_6562)
    
    # ################# End of 'kmeans2(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'kmeans2' in the type store
    # Getting the type of 'stypy_return_type' (line 535)
    stypy_return_type_6565 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 535, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_6565)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'kmeans2'
    return stypy_return_type_6565

# Assigning a type to the variable 'kmeans2' (line 535)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 535, 0), 'kmeans2', kmeans2)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
